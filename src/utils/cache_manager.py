"""
통합 캐시 관리 시스템

메모리 및 디스크 캐싱과 캐시 경로 관리를 모두 처리합니다.
"""

import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
import hashlib

from .unified_logging import get_logger

if TYPE_CHECKING:
    from .unified_config import DirectoryPaths

logger = get_logger(__name__)


class UnifiedCachePathManager:
    """모든 캐시 경로를 중앙에서 일관되게 관리하는 클래스"""

    def __init__(self, paths: "DirectoryPaths"):
        """
        Args:
            paths: DirectoryPaths 객체. 캐시 기본 경로를 포함합니다.
        """
        self.base_dir = Path(paths.cache_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ 통합 캐시 경로 관리자 초기화 (기본 경로: {self.base_dir})")

    def get_path(self, cache_type: str, filename: Optional[str] = None) -> Path:
        """
        지정된 유형의 캐시 경로를 가져오고, 필요 시 디렉토리를 생성합니다.

        Args:
            cache_type: 캐시 유형 (e.g., 'model', 'feature_vector')
            filename: 특정 파일 이름 (선택 사항)

        Returns:
            캐시 경로 (Path 객체)
        """
        type_dir = self.base_dir / cache_type
        type_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            return type_dir / filename
        return type_dir

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        특정 유형의 캐시 또는 전체 캐시를 정리합니다.

        Args:
            cache_type: 정리할 캐시 유형. None이면 전체 캐시를 정리합니다.
        """
        if cache_type:
            target_dir = self.base_dir / cache_type
            if target_dir.exists():
                shutil.rmtree(target_dir)
                logger.info(f"캐시 유형 '{cache_type}' 정리 완료.")
            else:
                logger.warning(f"캐시 유형 '{cache_type}'을(를) 찾을 수 없습니다.")
        else:
            shutil.rmtree(self.base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)  # 기본 디렉토리 다시 생성
            logger.info("모든 캐시 정리 완료.")


class CacheManager:
    """
    메모리와 디스크를 모두 사용하는 간단한 캐시 관리자.
    특정 캐시 유형에 대한 캐시 디렉토리를 관리합니다.
    """
    def __init__(
        self,
        path_manager: UnifiedCachePathManager,
        cache_type: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.cache_dir = path_manager.get_path(cache_type)
        
        self.memory_cache: Dict[str, Any] = {}
        self.max_memory_cache_size = self.config.get("max_memory_cache_size", 100)
        
        logger.debug(f"CacheManager for '{cache_type}' initialized at {self.cache_dir}")

    def get(self, key: str, use_disk: bool = True) -> Optional[Any]:
        """캐시에서 데이터를 가져옵니다. 메모리 캐시를 먼저 확인합니다."""
        if key in self.memory_cache:
            logger.debug(f"Memory cache hit for key: {key}")
            return self.memory_cache[key]

        if use_disk:
            file_path = self._get_file_path(key)
            if file_path.exists():
                logger.debug(f"Disk cache hit for key: {key}")
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                    # 디스크에서 읽은 데이터도 메모리에 캐싱
                    if len(self.memory_cache) < self.max_memory_cache_size:
                        self.memory_cache[key] = data
                    return data
                except Exception as e:
                    logger.warning(f"캐시 파일 로드 실패: {file_path}, {e}")
                    return None
        
        return None

    def set(self, key: str, value: Any, use_disk: bool = True):
        """캐시에 데이터를 저장합니다."""
        self._evict_if_needed()
        self.memory_cache[key] = value

        if use_disk:
            file_path = self._get_file_path(key)
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.error(f"캐시 파일 저장 실패: {file_path}, {e}")


    def _get_file_path(self, key: str) -> Path:
        """키를 해시하여 파일 경로를 생성합니다."""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.pkl"

    def _evict_if_needed(self):
        """메모리 캐시가 최대 크기를 초과하면 가장 오래된 항목을 제거합니다."""
        if len(self.memory_cache) >= self.max_memory_cache_size:
            try:
                # 간단한 LRU: 가장 먼저 추가된 항목 삭제
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]
                logger.debug(f"Evicted '{oldest_key}' from memory cache.")
            except StopIteration:
                pass # 캐시가 비어있는 경우

    def clear(self, clear_disk: bool = True):
        """캐시를 비웁니다."""
        self.memory_cache.clear()
        if clear_disk:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache for '{self.cache_dir.name}' cleared.")
