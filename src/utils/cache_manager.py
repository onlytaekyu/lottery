"""
캐시 관리 모듈

메모리 및 디스크 캐시를 관리하는 클래스를 제공합니다.
"""

import os
import pickle
import time
from collections import OrderedDict
from typing import Any, Optional, Dict, Union
import numpy as np
import torch
import json
import hashlib
from pathlib import Path
from .error_handler import get_logger
import threading


class CacheManager:
    """
    메모리 및 디스크 캐시를 관리하는 클래스입니다.
    """

    def __init__(
        self,
        pattern_analyzer=None,  # 하위 호환성을 위해 유지
        max_memory_size: int = 1024 * 1024 * 10,  # 기본값 10MB
        max_disk_size: int = 1024 * 1024 * 100,  # 기본값 100MB
        cache_dir: str = "data/cache",
        enable_compression: bool = True,
        default_ttl: int = 300,
    ):
        """
        CacheManager 초기화

        Args:
            pattern_analyzer: 패턴 분석기 (하위 호환성을 위해 유지)
            max_memory_size: 최대 메모리 캐시 크기 (바이트)
            max_disk_size: 최대 디스크 캐시 크기 (바이트)
            cache_dir: 캐시 디렉토리 경로
            enable_compression: 압축 활성화 여부
            default_ttl: 기본 캐시 유효 시간 (초)
        """
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_compression = enable_compression
        self.default_ttl = default_ttl
        self.memory_cache = OrderedDict()
        self.disk_cache = {}
        self.pattern_analyzer = pattern_analyzer  # 하위 호환성을 위해 유지
        self.logger = get_logger(__name__)
        os.makedirs(cache_dir, exist_ok=True)

    def _normalize_key(self, key: Any) -> str:
        """
        키를 문자열 형식으로 정규화합니다.

        Args:
            key: 정규화할 키

        Returns:
            정규화된 문자열 키
        """
        if isinstance(key, tuple):
            return "_".join(str(k) for k in key)
        elif isinstance(key, (int, float)):
            return str(key)
        elif not isinstance(key, str):
            return str(key)
        return key

    def _get_cache_path(self, key: Any, mode: str = "json") -> Path:
        """
        주어진 키에 대한 캐시 파일 경로를 반환합니다.

        Args:
            key: 캐시 키
            mode: 저장 모드 ('json' 또는 'pickle')

        Returns:
            캐시 파일 경로
        """
        # 키 정규화
        key = self._normalize_key(key)

        # 키의 해시 생성
        key_hash = hashlib.md5(key.encode()).hexdigest()

        # 파일 확장자 설정
        extension = ".json" if mode == "json" else ".pkl"
        return self.cache_dir / f"{key_hash}{extension}"

    def _safe_serialize(self, value: Any) -> Any:
        """
        객체를 JSON 안전한 형식으로 직렬화합니다.

        Args:
            value: 직렬화할 값

        Returns:
            직렬화된 값
        """
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.str_):
            return str(value)
        elif isinstance(value, (set, frozenset)):
            return list(value)
        elif isinstance(value, dict):
            # 딕셔너리의 각 항목 재귀적으로 직렬화
            result = {}
            for k, v in value.items():
                # 키가 튜플인 경우 문자열로 변환
                if isinstance(k, tuple):
                    normalized_key = self._normalize_key(k)
                elif not isinstance(k, (str, int, float, bool, type(None))):
                    normalized_key = str(k)
                else:
                    normalized_key = k
                result[normalized_key] = self._safe_serialize(v)
            return result
        elif isinstance(value, (list, tuple)):
            return [self._safe_serialize(item) for item in value]
        elif hasattr(value, "to_dict") and callable(value.to_dict):
            return self._safe_serialize(value.to_dict())
        elif hasattr(value, "to_json") and callable(value.to_json):
            return self._safe_serialize(value.to_json())
        elif value is None or isinstance(value, (str, int, float, bool)):
            return value
        else:
            # 기타 객체는 문자열로 변환
            try:
                return str(value)
            except Exception as e:
                self.logger.warning(
                    f"직렬화할 수 없는 객체: {type(value)}, 오류: {str(e)}"
                )
                return "UNKNOWN_OBJECT"

    def set(self, key: Any, value: Any, mode: str = "json") -> bool:
        """
        캐시에 값을 저장합니다.

        Args:
            key: 저장할 키
            value: 저장할 값
            mode: 저장 방식 ('json' 또는 'pickle')

        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 키 정규화
            normalized_key = self._normalize_key(key)
            self.logger.debug(f"캐시 저장: {normalized_key} (모드: {mode})")

            # None 값은 저장하지 않음
            if value is None:
                self.logger.debug(f"None 값은 캐시에 저장하지 않음: {normalized_key}")
                return False

            if mode == "json":
                # JSON 직렬화
                serialized_value = self._safe_serialize(value)
                cache_path = self._get_cache_path(normalized_key, mode="json")

                # 임시 파일에 쓰고 성공하면 이름 변경 (atomic write)
                temp_path = cache_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(serialized_value, f, ensure_ascii=False, indent=2)

                # 성공적으로 쓰였으면 정상 파일명으로 변경
                temp_path.replace(cache_path)
                self.logger.debug(f"JSON 캐시 파일 저장 완료: {cache_path}")
            else:
                # 피클 직렬화
                if isinstance(value, torch.Tensor):
                    data = pickle.dumps(value.cpu().numpy())
                elif isinstance(value, np.ndarray):
                    data = pickle.dumps(value)
                else:
                    data = pickle.dumps(value)

                # 메모리 또는 디스크에 저장
                if len(data) < self.max_memory_size:
                    self.memory_cache[normalized_key] = (time.time(), data)
                    self.logger.debug(f"메모리 캐시에 저장 완료: {normalized_key}")
                else:
                    cache_path = self._get_cache_path(normalized_key, mode="pickle")
                    with open(cache_path, "wb") as f:
                        f.write(data)
                    self.disk_cache[normalized_key] = cache_path
                    self.logger.debug(f"디스크 캐시에 저장 완료: {cache_path}")

            return True
        except Exception as e:
            self.logger.error(f"캐시 저장 중 오류 발생: 키={key}, 오류={str(e)}")
            # 임시 파일 제거
            try:
                if "temp_path" in locals() and temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            return False

    def get(self, key: Any, default: Any = None) -> Any:
        """
        캐시에서 값을 조회합니다.

        Args:
            key: 조회할 키
            default: 기본값 (키가 없을 경우 반환)

        Returns:
            캐시된 값 또는 기본값
        """
        try:
            normalized_key = self._normalize_key(key)
            self.logger.debug(f"캐시 조회: {normalized_key}")

            # 메모리 캐시에서 먼저 조회
            if normalized_key in self.memory_cache:
                timestamp, data = self.memory_cache[normalized_key]
                # 캐시 만료 확인
                if time.time() - timestamp <= self.default_ttl:
                    self.logger.debug(f"메모리 캐시 히트: {normalized_key}")
                    return pickle.loads(data)
                else:
                    # 만료된 경우 제거
                    del self.memory_cache[normalized_key]
                    self.logger.debug(f"만료된 메모리 캐시 삭제: {normalized_key}")

            # JSON 파일 조회
            json_path = self._get_cache_path(normalized_key, mode="json")
            if json_path.exists():
                # 파일이 비어있는지 확인
                if json_path.stat().st_size == 0:
                    self.logger.warning(
                        f"빈 JSON 캐시 파일 발견: {normalized_key}, 삭제합니다."
                    )
                    json_path.unlink()
                    return default

                # 파일 내용 로드
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    self.logger.debug(f"JSON 캐시 파일 히트: {normalized_key}")
                    return self._process_loaded_data(data, normalized_key)
                except json.JSONDecodeError as json_error:
                    self.logger.warning(
                        f"손상된 JSON 캐시 파일: {normalized_key}, 삭제합니다. 오류: {str(json_error)}"
                    )
                    json_path.unlink()
                    return default

            # 피클 파일 조회
            pickle_path = self._get_cache_path(normalized_key, mode="pickle")
            if pickle_path.exists():
                try:
                    with open(pickle_path, "rb") as f:
                        data = pickle.load(f)
                    self.logger.debug(f"피클 캐시 파일 히트: {normalized_key}")
                    return data
                except Exception as e:
                    self.logger.warning(
                        f"손상된 피클 캐시 파일: {normalized_key}, 삭제합니다. 오류: {str(e)}"
                    )
                    pickle_path.unlink()
                    return default

            self.logger.debug(f"캐시 미스: {normalized_key}")
            return default

        except Exception as e:
            self.logger.error(f"캐시 조회 중 오류 발생: 키={key}, 오류={str(e)}")
            return default

    def _process_loaded_data(self, data: Any, key: str) -> Any:
        """
        로드된 데이터를 처리합니다.
        특정 데이터 형식에 대한 객체 변환을 수행합니다.

        Args:
            data: 로드된 데이터
            key: 캐시 키

        Returns:
            처리된 데이터
        """
        # PatternAnalysis 객체 변환
        if key.startswith("pattern_analysis_") and isinstance(data, dict):
            try:
                from ..shared.types import PatternAnalysis

                return PatternAnalysis.from_dict(data)
            except Exception as e:
                self.logger.warning(f"PatternAnalysis 변환 실패: {key}, 오류: {str(e)}")

        # DistributionPattern 객체 변환
        if (
            key.startswith("pattern_analysis_")
            and isinstance(data, dict)
            and "even_odd" in data
        ):
            try:
                from ..analysis.distribution_analyzer import DistributionPattern

                # 각 패턴 카테고리에 대한 객체 변환
                result = {}
                for category in ["even_odd", "low_high", "ranges"]:
                    if category in data and isinstance(data[category], list):
                        result[category] = [
                            (
                                DistributionPattern.from_dict(pattern)
                                if isinstance(pattern, dict)
                                else pattern
                            )
                            for pattern in data[category]
                        ]

                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"패턴 필터 변환 실패: {key}, 오류: {str(e)}")

        return data

    def delete(self, key: Any) -> bool:
        """
        캐시에서 항목 삭제

        Args:
            key: 삭제할 항목의 키

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            # 키 정규화
            normalized_key = self._normalize_key(key)
            self.logger.debug(f"캐시 삭제: {normalized_key}")

            # 메모리 캐시에서 삭제
            if normalized_key in self.memory_cache:
                del self.memory_cache[normalized_key]

            # JSON 캐시 파일 삭제
            json_file = self._get_cache_path(normalized_key, mode="json")
            if json_file.exists():
                json_file.unlink()

            # 피클 캐시 파일 삭제
            pickle_file = self._get_cache_path(normalized_key, mode="pickle")
            if pickle_file.exists():
                pickle_file.unlink()

            return True
        except Exception as e:
            self.logger.error(f"캐시 항목 삭제 실패: {key}, 오류: {str(e)}")
            return False

    def clear(self, key: Optional[Any] = None) -> bool:
        """
        캐시를 지웁니다. 특정 키 또는 모든 캐시를 지울 수 있습니다.

        Args:
            key: 지울 특정 키 (None인 경우 모든 캐시)

        Returns:
            bool: 지우기 성공 여부
        """
        try:
            if key:
                # 특정 키만 삭제
                return self.delete(key)
            else:
                # 모든 캐시 삭제
                # 메모리 캐시 초기화
                self.memory_cache.clear()
                self.disk_cache.clear()

                # 모든 캐시 파일 삭제
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()

                self.logger.info("모든 캐시가 삭제되었습니다.")
                return True
        except Exception as e:
            self.logger.error(f"캐시 지우기 실패: {str(e)}")
            return False

    def clear_old_cache(self, max_age_days: int = 7) -> int:
        """
        지정된 일수보다 오래된 캐시 파일을 지웁니다.

        Args:
            max_age_days: 최대 보관 일수

        Returns:
            int: 삭제된 파일 수
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            removed_count = 0

            # JSON 캐시 파일 정리
            for cache_file in self.cache_dir.glob("*.json"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    cache_file.unlink()
                    removed_count += 1

            # 피클 캐시 파일 정리
            for cache_file in self.cache_dir.glob("*.pkl"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    cache_file.unlink()
                    removed_count += 1

            self.logger.info(f"{removed_count}개의 오래된 캐시 파일이 삭제되었습니다.")
            return removed_count
        except Exception as e:
            self.logger.error(f"오래된 캐시 정리 중 오류 발생: {str(e)}")
            return 0

    def cleanup_expired(self) -> int:
        """
        만료된 캐시 항목을 정리합니다.

        Returns:
            int: 정리된 항목 수
        """
        try:
            current_time = time.time()
            cleaned_count = 0

            # 메모리 캐시 정리
            expired_keys = [
                key
                for key, (timestamp, _) in self.memory_cache.items()
                if current_time - timestamp > self.default_ttl
            ]
            for key in expired_keys:
                del self.memory_cache[key]
                cleaned_count += 1
                self.logger.debug(f"만료된 메모리 캐시 항목 제거: {key}")

            # 디스크 캐시 정리 (JSON 파일)
            for cache_file in self.cache_dir.glob("*.json"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > self.default_ttl:
                    cache_file.unlink()
                    cleaned_count += 1
                    self.logger.debug(f"만료된 JSON 캐시 파일 제거: {cache_file}")

            # 디스크 캐시 정리 (피클 파일)
            for cache_file in self.cache_dir.glob("*.pkl"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > self.default_ttl:
                    cache_file.unlink()
                    cleaned_count += 1
                    self.logger.debug(f"만료된 피클 캐시 파일 제거: {cache_file}")

            # 캐시 크기 제한 확인 (메모리)
            if len(self.memory_cache) > self.max_memory_size:
                # 가장 오래된 항목부터 제거
                items_to_remove = len(self.memory_cache) - self.max_memory_size
                for _ in range(items_to_remove):
                    self.memory_cache.popitem(last=False)
                    cleaned_count += 1
                self.logger.debug(
                    f"메모리 캐시 크기 제한으로 {items_to_remove}개 항목 제거"
                )

            return cleaned_count
        except Exception as e:
            self.logger.error(f"캐시 정리 중 오류 발생: {str(e)}")
            return 0

    # 하위 호환성을 위한 별칭 메서드
    def put(self, key: Any, value: Any) -> bool:
        """
        캐시에 값을 저장합니다 (set의 별칭).
        하위 호환성을 위해 유지됩니다.

        Args:
            key: 저장할 키
            value: 저장할 값

        Returns:
            bool: 저장 성공 여부
        """
        return self.set(key, value, mode="pickle")

    # 하위 호환성을 위한 메서드들
    def _get_cached_result(self, key):
        """
        캐싱된 결과를 가져옵니다. (하위 호환성 유지용)
        """
        if not self.pattern_analyzer:
            return self.get(key)

        result = self.get(key)
        if result is None and hasattr(self.pattern_analyzer, "tensor_cache"):
            result = self.pattern_analyzer.tensor_cache.get(key)
        if result is None and hasattr(self.pattern_analyzer, "thread_local_cache"):
            result = self.pattern_analyzer.thread_local_cache.get(key)
        return result

    def _cache_result(self, key, result):
        """
        결과를 캐시에 저장합니다. (하위 호환성 유지용)
        """
        if (
            self.pattern_analyzer
            and isinstance(result, torch.Tensor)
            and hasattr(self.pattern_analyzer, "tensor_cache")
        ):
            self.pattern_analyzer.tensor_cache.put(key, result)
        else:
            self.put(key, result)


class ThreadLocalCache:
    """스레드 로컬 캐시 관리자"""

    def __init__(self):
        self._local = threading.local()
        self._logger = get_logger(__name__)

    def get(self, key: str, default: Any = None) -> Any:
        """캐시된 값 조회"""
        try:
            return getattr(self._local, key, default)
        except Exception as e:
            self._logger.error(f"캐시 조회 중 오류 발생: {str(e)}")
            return default

    def set(self, key: str, value: Any) -> None:
        """캐시 값 설정"""
        try:
            setattr(self._local, key, value)
        except Exception as e:
            self._logger.error(f"캐시 설정 중 오류 발생: {str(e)}")

    def delete(self, key: str) -> None:
        """캐시 값 삭제"""
        try:
            delattr(self._local, key)
        except Exception as e:
            self._logger.error(f"캐시 삭제 중 오류 발생: {str(e)}")

    def clear(self) -> None:
        """모든 캐시 삭제"""
        try:
            self._local.__dict__.clear()
        except Exception as e:
            self._logger.error(f"캐시 초기화 중 오류 발생: {str(e)}")

    def get_all(self) -> Dict[str, Any]:
        """모든 캐시 값 조회"""
        try:
            return dict(self._local.__dict__)
        except Exception as e:
            self._logger.error(f"캐시 전체 조회 중 오류 발생: {str(e)}")
            return {}
