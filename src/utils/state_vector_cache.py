"""
상태 벡터 캐시 모듈

이 모듈은 특성 벡터 캐싱 기능을 제공하여 반복적인 벡터 계산을 방지합니다.
싱글톤 패턴을 사용하여 전체 시스템에서 일관된 캐시를 유지합니다.
"""

import time
import os
import logging
import threading
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import numpy as np
import json
from datetime import datetime

from .unified_logging import get_logger
from ..utils.config_loader import ConfigProxy

# 로거 설정
logger = get_logger(__name__)

# 전역 캐시 인스턴스
_GLOBAL_CACHE_INSTANCE = None
_GLOBAL_CACHE_LOCK = threading.Lock()


def get_cache(
    config: Optional[Union[Dict[str, Any], ConfigProxy]] = None,
) -> "StateVectorCache":
    """
    전역 캐시 인스턴스를 반환합니다. 싱글톤 패턴 구현.

    Args:
        config: 설정 객체 (필요한 경우 새 인스턴스 생성에 사용)

    Returns:
        StateVectorCache: 전역 캐시 인스턴스
    """
    global _GLOBAL_CACHE_INSTANCE

    with _GLOBAL_CACHE_LOCK:
        if _GLOBAL_CACHE_INSTANCE is None:
            _GLOBAL_CACHE_INSTANCE = StateVectorCache(config)

    return _GLOBAL_CACHE_INSTANCE


class StateVectorCache:
    """
    상태 벡터 캐시 클래스

    로또 번호 조합의 특성 벡터를 캐싱하여 중복 계산을 방지합니다.
    성능 통계도 수집하여 캐시 효율성을 모니터링합니다.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigProxy]] = None):
        """
        초기화

        Args:
            config: 설정 객체
        """
        if config is None:
            raise KeyError("캐시 설정이 제공되지 않았습니다.")

        self.config = config
        self.logger = get_logger(__name__)

        # ConfigProxy에서 직접 딕셔너리로 변환 (필요한 경우)
        if isinstance(self.config, ConfigProxy):
            self.config = self.config.to_dict()

        if "caching" not in self.config:
            raise KeyError("설정에 'caching' 섹션이 없습니다.")

        caching_config = self.config["caching"]

        # 필수 캐시 설정 키 확인
        required_keys = [
            "enable_feature_cache",
            "max_cache_size",
            "cache_log_level",
            "cache_metrics",
        ]
        for key in required_keys:
            if key not in caching_config:
                raise KeyError(f"캐싱 설정에 필수 키가 없습니다: {key}")

        # 캐시 메트릭 설정 확인
        if (
            "save" not in caching_config["cache_metrics"]
            or "report_interval" not in caching_config["cache_metrics"]
        ):
            raise KeyError(
                "캐시 메트릭 설정에 필수 키가 없습니다: 'save' 또는 'report_interval'"
            )

        # 캐시 설정 - 명시적 접근 사용
        self.enabled = caching_config["enable_feature_cache"]
        self.max_size = caching_config["max_cache_size"]
        self.log_level = caching_config["cache_log_level"]

        # 영구 캐시 설정
        self.persistent_cache = caching_config.get("persistent_cache", True)
        self.cache_file = caching_config.get("cache_file", "state_vector_cache.npz")

        # 로그 수준 설정
        log_level_str = self.log_level
        if isinstance(log_level_str, str):
            if log_level_str == "DEBUG":
                self.logger.setLevel(logging.DEBUG)
            elif log_level_str == "INFO":
                self.logger.setLevel(logging.INFO)
            elif log_level_str == "WARNING":
                self.logger.setLevel(logging.WARNING)
            elif log_level_str == "ERROR":
                self.logger.setLevel(logging.ERROR)
            elif log_level_str == "CRITICAL":
                self.logger.setLevel(logging.CRITICAL)
        else:
            self.logger.setLevel(self.log_level)

        # 캐시 구조 및 통계
        self._cache: Dict[str, np.ndarray] = {}
        self._access_count: Dict[str, int] = {}
        self._last_access: Dict[str, float] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "inserts": 0,
            "size": 0,
            "hit_ratio": 0.0,
            "last_report_time": time.time(),
        }

        # 캐시 메트릭 설정
        metrics_config = caching_config["cache_metrics"]
        self.save_metrics = metrics_config["save"]
        self.report_interval = metrics_config["report_interval"]

        # 잠금 장치
        self._lock = threading.RLock()

        # 영구 캐시 로드 시도
        if self.persistent_cache:
            self._load_cache()

        self.logger.info(
            f"상태 벡터 캐시 초기화 완료 (최대 크기: {self.max_size}, 영구 캐시: {self.persistent_cache})"
        )

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        키에 해당하는 벡터를 캐시에서 가져옵니다.

        Args:
            key: 캐시 키

        Returns:
            벡터 배열 또는 None (캐시 미스)
        """
        if not self.enabled:
            return None

        with self._lock:
            if key in self._cache:
                # 캐시 히트
                self._stats["hits"] += 1
                self._access_count[key] = self._access_count.get(key, 0) + 1
                self._last_access[key] = time.time()

                # 로그 추가 (DEBUG 레벨)
                self.logger.debug(f"Cache hit for key: {key[:8]}...")

                if self._stats["hits"] % 100 == 0:
                    hit_ratio = self._get_hit_ratio()
                    self._stats["hit_ratio"] = hit_ratio
                    self.logger.info(f"캐시 히트율: {hit_ratio:.1%}")

                return self._cache[key]
            else:
                # 캐시 미스
                self._stats["misses"] += 1

                # 로그 추가 (DEBUG 레벨)
                self.logger.debug(f"Cache miss for key: {key[:8]}...")

                return None

    def set(self, key: str, vector: np.ndarray) -> None:
        """
        벡터를 캐시에 저장합니다.

        Args:
            key: 캐시 키
            vector: 저장할 벡터 배열
        """
        if not self.enabled:
            return

        with self._lock:
            # 캐시가 가득 찼으면 정리
            if len(self._cache) >= self.max_size:
                self._evict()

            # 캐시에 추가
            self._cache[key] = vector
            self._access_count[key] = 1
            self._last_access[key] = time.time()
            self._stats["inserts"] += 1
            self._stats["size"] = len(self._cache)

            if self._stats["inserts"] % 100 == 0:
                self.logger.debug(
                    f"캐시 삽입: {self._stats['inserts']}개, 크기: {self._stats['size']}개"
                )

            # 일정 간격으로 영구 캐시 저장
            if self.persistent_cache and self._stats["inserts"] % 500 == 0:
                self._save_cache()

    def get_or_compute(
        self, key: str, compute_func: Callable[[], np.ndarray]
    ) -> np.ndarray:
        """
        키에 해당하는 벡터를 캐시에서 가져오거나, 없으면 계산 후 저장합니다.

        Args:
            key: 캐시 키
            compute_func: 벡터를 계산하는 함수

        Returns:
            벡터 배열 (캐시에서 가져오거나 계산한 결과)
        """
        # 캐시에서 가져오기
        cached_vector = self.get(key)
        if cached_vector is not None:
            return cached_vector

        # 벡터 계산
        vector = compute_func()

        # 캐시에 저장
        self.set(key, vector)

        return vector

    def _evict(self) -> None:
        """
        캐시가 가득 찼을 때 항목을 제거합니다.
        가장 오래된 접근 또는 가장 적게 사용된 항목을 제거합니다.
        """
        if not self._cache:
            return

        with self._lock:
            # 각 항목에 대한 점수 계산 (접근 횟수와 마지막 접근 시간 기반)
            now = time.time()
            scores = {}
            for key in self._cache:
                # 접근 횟수가 적을수록, 마지막 접근이 오래되었을수록 점수가 낮음
                access_count = self._access_count.get(key, 0)
                last_access = self._last_access.get(key, 0)
                time_since_access = now - last_access

                # 가중치 설정 (시간에 더 큰 가중치 부여)
                time_weight = 0.7
                count_weight = 0.3

                # 정규화된 점수 계산
                time_score = 1.0 / (1.0 + time_since_access / 3600)  # 시간 단위 정규화
                count_score = min(1.0, access_count / 10.0)  # 접근 횟수 정규화

                # 최종 점수
                scores[key] = time_weight * time_score + count_weight * count_score

            # 점수가 가장 낮은 항목 찾기
            evict_key = min(scores, key=scores.get)

            # 항목 제거
            if evict_key in self._cache:
                del self._cache[evict_key]
            if evict_key in self._access_count:
                del self._access_count[evict_key]
            if evict_key in self._last_access:
                del self._last_access[evict_key]

            self._stats["evictions"] += 1
            self._stats["size"] = len(self._cache)

            self.logger.debug(
                f"캐시 항목 제거: {evict_key[:8]}... (점수: {scores[evict_key]:.4f})"
            )

    def clear(self) -> None:
        """
        캐시를 완전히 비웁니다.
        """
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
            self._last_access.clear()
            self._stats["size"] = 0
            self.logger.info("캐시 초기화 완료")

    def _get_hit_ratio(self) -> float:
        """
        캐시 히트율을 계산합니다.

        Returns:
            float: 캐시 히트율 (0-1 범위)
        """
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계를 반환합니다.

        Returns:
            Dict[str, Any]: 캐시 통계 정보
        """
        with self._lock:
            # 기본 통계 복사
            stats = self._stats.copy()

            # 현재 히트율 업데이트
            stats["hit_ratio"] = self._get_hit_ratio()

            # 캐시 사용량 추가
            stats["memory_usage_estimate"] = self._estimate_memory_usage()

            # 가장 많이 접근한 항목 추가
            stats["most_accessed"] = self._get_most_accessed_keys(5)

            # 추가 지표
            stats["cache_age_seconds"] = time.time() - stats.get(
                "last_report_time", time.time()
            )
            stats["filled_ratio"] = (
                stats["size"] / self.max_size if self.max_size > 0 else 0
            )
            stats["hit_miss_ratio"] = (
                stats["hits"] / stats["misses"] if stats["misses"] > 0 else float("inf")
            )
            stats["cache_efficiency"] = stats["hits"] / (
                stats["inserts"] + 1
            )  # 삽입 당 히트 비율
            stats["timestamp"] = time.time()

            return stats

    def _estimate_memory_usage(self) -> int:
        """
        캐시가 사용하는 메모리 양을 대략적으로 추정합니다.

        Returns:
            int: 추정 메모리 사용량 (바이트)
        """
        # 벡터 데이터 메모리 계산
        vector_bytes = sum(v.nbytes for v in self._cache.values())

        # 키와 메타데이터 메모리 계산 (대략적인 추정)
        key_bytes = sum(len(k) * 2 for k in self._cache.keys())  # 키는 문자열
        metadata_bytes = len(self._cache) * 24  # 접근 횟수와 마지막 접근 시간

        return vector_bytes + key_bytes + metadata_bytes

    def _get_most_accessed_keys(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        가장 많이 접근한 캐시 항목을 반환합니다.

        Args:
            count: 반환할 항목 수

        Returns:
            List[Dict[str, Any]]: 가장 많이 접근한 항목 정보
        """
        if not self._access_count:
            return []

        # 접근 횟수 기준 내림차순 정렬
        sorted_keys = sorted(
            self._access_count.keys(),
            key=lambda k: self._access_count.get(k, 0),
            reverse=True,
        )

        # 상위 N개 선택
        top_keys = sorted_keys[:count]

        # 결과 포맷팅
        result = []
        for key in top_keys:
            # 키의 첫 부분만 포함 (보안 및 가독성 목적)
            short_key = key[:8] + "..." if len(key) > 8 else key
            result.append(
                {
                    "key": short_key,
                    "access_count": self._access_count.get(key, 0),
                    "last_access": self._last_access.get(key, 0),
                }
            )

        return result

    def save_stats(self, path: Optional[str] = None) -> None:
        """
        캐시 통계를 파일에 저장합니다.

        Args:
            path: 저장할 파일 경로 (기본값: None, 기본 경로 사용)
        """
        if not self.save_metrics:
            return

        try:
            # 통계 데이터 준비
            stats = self.get_stats()

            # 현재 타임스탬프 추가
            stats["timestamp"] = time.time()
            stats["datetime"] = datetime.now().isoformat()

            # 기본 경로 설정
            if path is None:
                # ConfigProxy 객체인 경우
                if hasattr(self.config, "get") and callable(
                    getattr(self.config, "get")
                ):
                    paths = self.config.get("paths")
                    if (
                        paths
                        and hasattr(paths, "get")
                        and callable(getattr(paths, "get"))
                    ):
                        report_dir = paths.get(
                            "performance_report_dir", "data/result/performance_reports"
                        )
                    else:
                        report_dir = "data/result/performance_reports"
                else:
                    # 딕셔너리인 경우
                    report_dir = self.config.get("paths", {}).get(
                        "performance_report_dir", "data/result/performance_reports"
                    )

                # 디렉토리 생성
                os.makedirs(report_dir, exist_ok=True)

                # 파일 경로 구성
                path = os.path.join(report_dir, f"cache_performance_report.json")

            # 파일 저장
            with open(path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            self.logger.info(f"캐시 통계 저장 완료: {path}")

            # 마지막 리포트 시간 업데이트
            self._stats["last_report_time"] = time.time()

        except Exception as e:
            self.logger.error(f"캐시 통계 저장 실패: {e}")

    def generate_key(self, numbers: List[int], version: str = "2.0") -> str:
        """
        번호 조합에 대한 고유한 캐시 키를 생성합니다.

        Args:
            numbers: 번호 조합
            version: 키 생성 알고리즘 버전

        Returns:
            str: 캐시 키
        """
        try:
            # 정렬된 번호 문자열 생성
            sorted_numbers = sorted(numbers)
            numbers_str = "_".join(map(str, sorted_numbers))

            # 버전에 따른 다른 접두사 사용
            prefix = f"v{version.replace('.', '_')}_"

            # SHA-256 해시 계산
            key = hashlib.sha256((prefix + numbers_str).encode()).hexdigest()

            return key
        except Exception as e:
            self.logger.error(f"캐시 키 생성 실패: {e}")
            # 유일성을 보장하기 위한 대체 키
            return f"fallback_{time.time()}_{hash(tuple(sorted(numbers)))}"

    def report_if_needed(self, force: bool = False) -> None:
        """
        필요한 경우 캐시 통계를 보고합니다.

        Args:
            force: 강제 보고 여부
        """
        if not self.save_metrics:
            return

        now = time.time()
        last_report_time = self._stats.get("last_report_time", 0)
        elapsed = now - last_report_time

        if force or elapsed > self.report_interval:
            # 통계 저장
            self.save_stats()

            # 로그 출력
            hit_ratio = self._get_hit_ratio()
            self.logger.info(
                f"캐시 성능 - 히트율: {hit_ratio:.1%}, 크기: {self._stats['size']}/{self.max_size}"
            )

    def _load_cache(self) -> None:
        """
        영구 캐시 파일에서 캐시를 로드합니다.
        """
        if not self.persistent_cache:
            return

        try:
            # 캐시 디렉토리 설정
            cache_dir = None
            # ConfigProxy 객체인 경우
            if hasattr(self.config, "get") and callable(getattr(self.config, "get")):
                # ConfigProxy 객체인 경우 paths를 먼저 가져온 후 하위 키에 접근
                paths = self.config.get("paths")
                if paths and hasattr(paths, "get") and callable(getattr(paths, "get")):
                    cache_dir = paths.get("cache_dir", "data/cache")
                else:
                    cache_dir = "data/cache"
            else:
                # 딕셔너리인 경우
                cache_dir = self.config.get("paths", {}).get("cache_dir", "data/cache")

            # 캐시 파일 경로 설정
            cache_file = os.path.join(cache_dir, self.cache_file)

            # 캐시 디렉토리 생성
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)

            # 캐시 파일이 있는지 확인
            if not os.path.exists(cache_file):
                self.logger.info(f"캐시 파일이 없습니다: {cache_file}")
                return

            # 로딩 시작
            self.logger.info(f"캐시 파일 로드 중: {cache_file}")
            load_start = time.time()

            # NPZ 파일 로드 (압축 형식)
            cache_data = np.load(cache_file, allow_pickle=True)

            # 벡터 데이터 추출
            with self._lock:
                # 각 벡터 추출
                for key in cache_data.files:
                    vector = cache_data[key]

                    # 벡터 형태 보정 (객체 배열인 경우)
                    if vector.dtype == np.dtype("O"):
                        # 실제 벡터 추출 시도
                        try:
                            actual_vector = vector.item()
                            if isinstance(actual_vector, np.ndarray):
                                self._cache[key] = actual_vector
                                self._access_count[key] = 1
                                self._last_access[key] = time.time()
                        except:
                            continue
                    else:
                        # 일반 숫자형 벡터
                        self._cache[key] = vector
                        self._access_count[key] = 1
                        self._last_access[key] = time.time()

                # 통계 업데이트
                self._stats["size"] = len(self._cache)
                self._stats["inserts"] = len(self._cache)

            # 메타데이터 파일 경로
            meta_file = os.path.join(cache_dir, "cache_metadata.json")

            # 메타데이터 로드 (있는 경우)
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    # 접근 횟수와 마지막 접근 시간 복원
                    if "access_count" in metadata:
                        self._access_count.update(metadata["access_count"])

                    if "last_access" in metadata:
                        self._last_access.update(metadata["last_access"])

                    # 통계 복원
                    if "stats" in metadata:
                        for key, value in metadata["stats"].items():
                            if key in self._stats:
                                self._stats[key] = value

                    self.logger.info(f"캐시 메타데이터 로드 완료: {meta_file}")
                except Exception as e:
                    self.logger.warning(f"캐시 메타데이터 로드 실패: {e}")

            # 로드 시간 측정
            load_time = time.time() - load_start
            self.logger.info(
                f"캐시 로드 완료: {len(self._cache)}개 항목, {load_time:.2f}초 소요"
            )

            # 캐시 크기 제한 적용
            if len(self._cache) > self.max_size:
                self.logger.warning(
                    f"캐시 크기({len(self._cache)})가 최대 크기({self.max_size})를 초과합니다. 정리를 수행합니다."
                )
                # 초과분 제거
                for _ in range(len(self._cache) - self.max_size):
                    self._evict()

        except Exception as e:
            self.logger.error(f"캐시 로드 실패: {e}")
            # 오류 발생 시 빈 캐시로 시작
            self._cache = {}
            self._access_count = {}
            self._last_access = {}

    def _save_cache(self) -> None:
        """
        캐시를 영구 파일에 저장합니다.
        """
        if not self.persistent_cache or not self._cache:
            return

        try:
            # 캐시 디렉토리 설정
            cache_dir = None
            # ConfigProxy 객체인 경우
            if hasattr(self.config, "get") and callable(getattr(self.config, "get")):
                # ConfigProxy 객체인 경우 paths를 먼저 가져온 후 하위 키에 접근
                paths = self.config.get("paths")
                if paths and hasattr(paths, "get") and callable(getattr(paths, "get")):
                    cache_dir = paths.get("cache_dir", "data/cache")
                else:
                    cache_dir = "data/cache"
            else:
                # 딕셔너리인 경우
                cache_dir = self.config.get("paths", {}).get("cache_dir", "data/cache")

            # 캐시 디렉토리 생성
            os.makedirs(cache_dir, exist_ok=True)

            # 캐시 파일 경로 설정
            cache_file = os.path.join(cache_dir, self.cache_file)
            temp_file = cache_file + ".tmp"

            # 저장 시작
            save_start = time.time()
            self.logger.info(f"캐시 저장 중: {len(self._cache)}개 항목")

            # 임시 파일에 먼저 저장 (원자적 쓰기 위해)
            with self._lock:
                # NPZ 형식으로 저장 (압축 형식)
                # 벡터를 딕셔너리로 변환
                cache_dict = {key: value for key, value in self._cache.items()}
                np.savez_compressed(temp_file, **cache_dict)

            # 임시 파일을 실제 파일로 이동 (원자적 작업)
            if os.path.exists(temp_file):
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                os.rename(temp_file, cache_file)

            # 메타데이터 저장
            meta_file = os.path.join(cache_dir, "cache_metadata.json")

            with open(meta_file, "w", encoding="utf-8") as f:
                metadata = {
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                    "access_count": self._access_count,
                    "last_access": self._last_access,
                    "stats": self._stats,
                    "version": "2.0",
                    "cache_size": len(self._cache),
                }
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # 저장 시간 측정
            save_time = time.time() - save_start
            self.logger.info(
                f"캐시 저장 완료: {len(self._cache)}개 항목, {save_time:.2f}초 소요"
            )

        except Exception as e:
            self.logger.error(f"캐시 저장 실패: {e}")
            # 임시 파일 정리
            temp_file = os.path.join(cache_dir, self.cache_file + ".tmp")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def get_all_values(self) -> List[np.ndarray]:
        """
        캐시에 저장된 모든 벡터 반환

        Returns:
            List[np.ndarray]: 모든 캐시된 벡터 목록
        """
        with self._lock:
            return list(self._cache.values())

    def get_cache_metrics(self) -> Dict[str, Any]:
        """
        캐시 성능 지표를 반환합니다.

        Returns:
            Dict[str, Any]: 캐시 성능 지표
        """
        with self._lock:
            hit_ratio = self._get_hit_ratio()
            filled_ratio = len(self._cache) / self.max_size if self.max_size > 0 else 0

            metrics = {
                "hit_ratio": hit_ratio,
                "miss_ratio": 1.0 - hit_ratio,
                "filled_ratio": filled_ratio,
                "total_hits": self._stats["hits"],
                "total_misses": self._stats["misses"],
                "total_inserts": self._stats["inserts"],
                "total_evictions": self._stats["evictions"],
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_estimate": self._estimate_memory_usage(),
                "timestamp": time.time(),
                "cache_enabled": self.enabled,
                "persistent_cache": self.persistent_cache,
            }

            return metrics

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """
        캐시 진단 정보를 반환합니다.

        Returns:
            Dict[str, Any]: 캐시 진단 정보
        """
        with self._lock:
            # 기본 지표
            metrics = self.get_cache_metrics()

            # 추가 진단 정보
            vector_shapes = {}
            value_ranges = {}

            # 캐시된 벡터의 형태와 값 범위 샘플링
            sample_size = min(10, len(self._cache))
            if sample_size > 0:
                sampled_keys = list(self._cache.keys())[:sample_size]

                for key in sampled_keys:
                    vector = self._cache[key]
                    vector_shapes[key[:8]] = list(vector.shape)

                    if len(vector) > 0:
                        value_ranges[key[:8]] = {
                            "min": float(np.min(vector)),
                            "max": float(np.max(vector)),
                            "mean": float(np.mean(vector)),
                            "std": float(np.std(vector)),
                            "nan_count": int(np.isnan(vector).sum()),
                        }

            # 접근 패턴 분석
            access_patterns = {
                "avg_access_count": (
                    np.mean(list(self._access_count.values()))
                    if self._access_count
                    else 0
                ),
                "max_access_count": (
                    max(self._access_count.values()) if self._access_count else 0
                ),
                "min_access_count": (
                    min(self._access_count.values()) if self._access_count else 0
                ),
                "most_accessed": self._get_most_accessed_keys(3),
            }

            # 진단 정보 종합
            diagnostic = {
                "metrics": metrics,
                "vector_shapes": vector_shapes,
                "value_ranges": value_ranges,
                "access_patterns": access_patterns,
                "last_report_time": time.ctime(self._stats.get("last_report_time", 0)),
                "warnings": [],
            }

            # 경고 추가
            if (
                metrics["hit_ratio"] < 0.5
                and metrics["total_hits"] + metrics["total_misses"] > 100
            ):
                diagnostic["warnings"].append(
                    "낮은 히트율: 캐시 설정이나 키 생성 방식을 검토하세요."
                )

            if metrics["filled_ratio"] > 0.9:
                diagnostic["warnings"].append(
                    "캐시가 거의 가득 찼습니다: 최대 크기 증가를 고려하세요."
                )

            if metrics["memory_usage_estimate"] > 1024 * 1024 * 100:  # 100MB
                diagnostic["warnings"].append(
                    "높은 메모리 사용량: 캐시 크기 제한을 검토하세요."
                )

            return diagnostic
