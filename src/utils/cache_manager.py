"""
DAEBAK_AI 로또 예측 시스템 - 캐시 관리 모듈

이 모듈은 메모리 및 디스크 캐시를 관리하고 최적화하는 기능을 제공합니다.
"""

import json
import os
import pickle
import time
from collections import OrderedDict
from typing import Any, Dict, Optional
import numpy as np
import torch
import hashlib
from pathlib import Path
from threading import RLock
from .unified_logging import get_logger


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
        self.lock = RLock()  # 스레드 안전성을 위한 RLock 추가
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
        elif isinstance(value, (np.integer, np.floating, np.bool_, np.str_)):
            return value.item()
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
        캐시에 값을 저장합니다. (스레드 안전)
        """
        with self.lock:  # RLock으로 임계 영역 보호
            try:
                # 키 정규화
                normalized_key = self._normalize_key(key)
                self.logger.debug(f"캐시 저장: {normalized_key} (모드: {mode})")

                # None 값은 저장하지 않음
                if value is None:
                    self.logger.debug(
                        f"None 값은 캐시에 저장하지 않음: {normalized_key}"
                    )
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

    def get(self, key: Any, default: Any = None, mode: str = "json") -> Any:
        """
        캐시에서 값을 가져옵니다. (스레드 안전)
        """
        with self.lock:  # RLock으로 임계 영역 보호
            try:
                # 키 정규화
                normalized_key = self._normalize_key(key)
                self.logger.debug(f"캐시 조회: {normalized_key} (모드: {mode})")

                if mode == "json":
                    # JSON 파일에서 로드
                    cache_path = self._get_cache_path(normalized_key, mode="json")
                    if cache_path.exists():
                        with open(cache_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        self.logger.debug(f"JSON 캐시 파일에서 로드 완료: {cache_path}")
                        return self._process_loaded_data(data, normalized_key)
                else:
                    # 메모리 캐시에서 확인
                    if normalized_key in self.memory_cache:
                        timestamp, data = self.memory_cache[normalized_key]
                        # TTL 확인
                        if time.time() - timestamp < self.default_ttl:
                            loaded_data = pickle.loads(data)
                            self.logger.debug(
                                f"메모리 캐시에서 로드 완료: {normalized_key}"
                            )
                            return loaded_data
                        else:
                            # 만료된 캐시 제거
                            del self.memory_cache[normalized_key]
                            self.logger.debug(
                                f"만료된 메모리 캐시 제거: {normalized_key}"
                            )

                    # 디스크 캐시에서 확인
                    if normalized_key in self.disk_cache:
                        cache_path = self.disk_cache[normalized_key]
                        if cache_path.exists():
                            with open(cache_path, "rb") as f:
                                data = pickle.load(f)
                            self.logger.debug(
                                f"디스크 캐시에서 로드 완료: {cache_path}"
                            )
                            return data

                # 캐시에 없는 경우 기본값 반환
                self.logger.debug(f"캐시에 없음: {normalized_key}")
                return default

            except Exception as e:
                self.logger.error(f"캐시 조회 중 오류 발생: 키={key}, 오류={str(e)}")
                return default

    def _process_loaded_data(self, data: Any, key: str) -> Any:
        """
        로드된 데이터를 처리하여 원래 형태로 복원합니다.

        Args:
            data: 로드된 데이터
            key: 캐시 키

        Returns:
            처리된 데이터
        """
        try:
            if isinstance(data, list) and len(data) > 0:
                if all(isinstance(item, (int, float)) for item in data):
                    return np.array(data)
            elif isinstance(data, dict):
                if "numbers" in data and isinstance(data["numbers"], list):
                    return data
            return data
        except Exception as e:
            self.logger.warning(f"데이터 처리 중 오류: 키={key}, 오류={str(e)}")
            return data

    def delete(self, key: Any) -> bool:
        """
        캐시에서 특정 키를 삭제합니다.

        Args:
            key: 삭제할 키

        Returns:
            삭제 성공 여부
        """
        with self.lock:
            try:
                normalized_key = self._normalize_key(key)
                deleted = False

                # 메모리 캐시에서 삭제
                if normalized_key in self.memory_cache:
                    del self.memory_cache[normalized_key]
                    deleted = True

                # 디스크 캐시에서 삭제
                if normalized_key in self.disk_cache:
                    cache_path = self.disk_cache[normalized_key]
                    if cache_path.exists():
                        cache_path.unlink()
                    del self.disk_cache[normalized_key]
                    deleted = True

                # JSON 캐시 파일 삭제
                json_path = self._get_cache_path(normalized_key, mode="json")
                if json_path.exists():
                    json_path.unlink()
                    deleted = True

                # 피클 캐시 파일 삭제
                pickle_path = self._get_cache_path(normalized_key, mode="pickle")
                if pickle_path.exists():
                    pickle_path.unlink()
                    deleted = True

                if deleted:
                    self.logger.debug(f"캐시 삭제 완료: {normalized_key}")
                return deleted
            except Exception as e:
                self.logger.error(f"캐시 삭제 중 오류 발생: 키={key}, 오류={str(e)}")
                return False

    def clear(self, key: Optional[Any] = None) -> bool:
        """
        캐시를 정리합니다.

        Args:
            key: 특정 키만 정리할 경우 지정 (None이면 전체 정리)

        Returns:
            정리 성공 여부
        """
        with self.lock:
            try:
                if key is not None:
                    # 특정 키만 삭제
                    return self.delete(key)
                else:
                    # 전체 캐시 정리
                    self.memory_cache.clear()
                    self.disk_cache.clear()

                    # 캐시 디렉토리의 모든 파일 삭제
                    for file_path in self.cache_dir.glob("*"):
                        if file_path.is_file():
                            file_path.unlink()

                    self.logger.info("전체 캐시 정리 완료")
                    return True
            except Exception as e:
                self.logger.error(f"캐시 정리 중 오류 발생: {str(e)}")
                return False

    def clear_old_cache(self, max_age_days: int = 7) -> int:
        """
        오래된 캐시를 정리합니다.

        Args:
            max_age_days: 최대 보관 일수

        Returns:
            삭제된 파일 수
        """
        with self.lock:
            try:
                current_time = time.time()
                max_age_seconds = max_age_days * 24 * 3600
                deleted_count = 0

                # 디스크 캐시 파일 확인
                for file_path in self.cache_dir.glob("*"):
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age_seconds:
                            file_path.unlink()
                            deleted_count += 1

                # 메모리 캐시에서 오래된 항목 제거
                expired_keys = []
                for key, (timestamp, _) in self.memory_cache.items():
                    if current_time - timestamp > max_age_seconds:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.memory_cache[key]
                    deleted_count += 1

                self.logger.info(f"오래된 캐시 정리 완료: {deleted_count}개 항목 삭제")
                return deleted_count
            except Exception as e:
                self.logger.error(f"오래된 캐시 정리 중 오류 발생: {str(e)}")
                return 0

    def cleanup_expired(self) -> int:
        """
        만료된 캐시 항목을 정리합니다.

        Returns:
            정리된 항목 수
        """
        with self.lock:
            try:
                current_time = time.time()
                expired_keys = []

                # 메모리 캐시에서 만료된 항목 찾기
                for key, (timestamp, _) in self.memory_cache.items():
                    if current_time - timestamp > self.default_ttl:
                        expired_keys.append(key)

                # 만료된 항목 제거
                for key in expired_keys:
                    del self.memory_cache[key]

                self.logger.debug(f"만료된 캐시 정리 완료: {len(expired_keys)}개 항목")
                return len(expired_keys)
            except Exception as e:
                self.logger.error(f"만료된 캐시 정리 중 오류 발생: {str(e)}")
                return 0

    # 하위 호환성을 위한 메서드들
    def put(self, key: Any, value: Any) -> bool:
        """하위 호환성을 위한 메서드"""
        return self.set(key, value)

    def _get_cached_result(self, key):
        """하위 호환성을 위한 메서드"""
        return self.get(key)

    def _cache_result(self, key, result):
        """하위 호환성을 위한 메서드"""
        return self.set(key, result)


# 전역 캐시 매니저 인스턴스
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """전역 캐시 매니저 인스턴스를 반환합니다."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
