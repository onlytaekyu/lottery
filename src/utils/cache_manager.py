"""
개인용 단순하고 빠른 캐시 시스템
"""

import json
import pickle
import time
import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

from .unified_logging import get_logger
from .error_handler import get_error_handler
from .memory_manager import get_memory_manager
from .unified_config import get_config
from .cache_paths import get_cache_path_manager
from .factory import get_singleton_instance

logger = get_logger(__name__)
error_handler = get_error_handler()


class GPUCache:
    """개인용 단순 캐시 (서버 기능 제거)"""

    def __init__(self):
        self.path_manager = get_cache_path_manager()
        self.cache_dir = self.path_manager.get_path(
            "default_cache"
        )  # 기본 캐시 디렉토리

        config = get_config("main").get_nested("utils.cache_manager", {})
        self.max_memory_items = config.get("max_memory_items", 100)

        # 메모리 캐시 (간단)
        self.memory_cache: Dict[str, Any] = {}
        self.memory_manager = get_memory_manager()

        logger.info(
            f"✅ 개인용 단순 캐시 초기화: {self.cache_dir}, 인메모리 캐시 크기: {self.max_memory_items}"
        )

    @error_handler.auto_recoverable(max_retries=2, delay=0.1)
    def set(
        self,
        key: str,
        value: Any,
        use_disk: bool = False,
        category: str = "default_cache",
    ) -> bool:
        """캐시 저장 (개인용 간소화)"""
        try:
            if use_disk:
                # 카테고리별 디스크 저장
                category_dir = self.path_manager.get_path(category)
                cache_file = category_dir / f"{key}.cache"

                try:
                    # JSON 시도
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(value, f)
                except (TypeError, ValueError):
                    # JSON 실패시 Pickle
                    cache_file = category_dir / f"{key}.pkl"
                    with open(cache_file, "wb") as f:
                        pickle.dump(value, f)
            else:
                # 메모리 저장
                if len(self.memory_cache) >= self.max_memory_items:
                    # 가장 오래된 항목 제거 (간단한 LRU)
                    oldest_key = next(iter(self.memory_cache))
                    del self.memory_cache[oldest_key]

                self.memory_cache[key] = value

            return True

        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            return False

    @error_handler.auto_recoverable(max_retries=2, delay=0.1)
    def get(
        self, key: str, default: Any = None, category: str = "default_cache"
    ) -> Any:
        """캐시 조회 (개인용 간소화)"""
        try:
            # 메모리 캐시 확인
            if key in self.memory_cache:
                return self.memory_cache[key]

            # 카테고리별 디스크 캐시 확인
            category_dir = self.path_manager.get_path(category)
            json_file = category_dir / f"{key}.cache"
            pkl_file = category_dir / f"{key}.pkl"

            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif pkl_file.exists():
                file_size = os.path.getsize(pkl_file)
                if not self.memory_manager.check_available_memory(file_size):
                    logger.warning(
                        f"메모리 부족: 캐시 파일 '{key}.pkl' (크기: {file_size / 1024**2:.2f}MB) 로드 불가."
                    )
                    return default

                with open(pkl_file, "rb") as f:
                    return pickle.load(f)

            return default

        except Exception as e:
            logger.error(f"캐시 조회 실패: {e}")
            return default

    def delete(self, key: str, category: str = "default_cache") -> bool:
        """캐시 삭제"""
        deleted = False

        # 메모리에서 삭제
        if key in self.memory_cache:
            del self.memory_cache[key]
            deleted = True

        # 디스크에서 삭제
        category_dir = self.path_manager.get_path(category)
        for ext in [".cache", ".pkl"]:
            cache_file = category_dir / f"{key}{ext}"
            if cache_file.exists():
                cache_file.unlink()
                deleted = True

        return deleted

    def clear(self):
        """전체 캐시 정리"""
        self.memory_cache.clear()
        # 중앙 경로 관리자를 통해 전체 캐시 디렉토리 정리
        self.path_manager.clear_cache()
        logger.info("메모리 캐시 및 디스크 캐시 전체 정리 완료.")

    def get_simple_stats(self) -> Dict[str, Any]:
        """간단한 통계"""
        # 전체 디스크 캐시 파일 수 계산
        disk_files = sum(1 for _ in self.path_manager.base_dir.rglob("*.pkl")) + sum(
            1 for _ in self.path_manager.base_dir.rglob("*.cache")
        )

        return {
            "memory_items": len(self.memory_cache),
            "disk_files": disk_files,
            "cache_dir": str(self.path_manager.base_dir),
        }


def get_cache_manager() -> GPUCache:
    """간단한 캐시 관리자 반환"""
    return get_singleton_instance(GPUCache)


# 하위 호환성
class CacheManager(GPUCache):
    """하위 호환성 래퍼"""

    def put(self, key, value):
        return self.set(key, value)

    def _get_cached_result(self, key):
        return self.get(key)

    def _cache_result(self, key, result):
        return self.set(key, result)


# 하위 호환성 래퍼 추가
class SelfHealingCacheManager(GPUCache):
    """하위 호환성 래퍼 - __init__.py에서 참조되는 클래스"""

    pass


def get_self_healing_cache():
    return get_cache_manager()


def get_self_healing_cache_manager():
    """하위 호환성 함수"""
    return get_cache_manager()
