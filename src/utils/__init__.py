"""
로또 번호 예측 시스템 - 통합 유틸리티 (완전 최적화 버전)

중복 로그 초기화 문제를 완전히 해결한 최적화된 시스템
- 자동 로거 할당
- 메모리 최적화된 지연 로딩
- Thread-Safe 보장
"""

from typing import Any, Optional
import threading

# 핵심 시스템만 즉시 로드
from .cache_paths import get_cache_dir, CACHE_DIR

# 최적화된 로깅 시스템 로드
from .unified_logging import (
    get_logger as _get_logger,
    get_logging_stats,
    cleanup_logging,
    get_optimization_report,
)

# Lazy import 캐시 (Thread-Safe)
_imported_items = {}
_import_lock = threading.RLock()


def get_logger(*args, **kwargs):
    """
    최적화된 로거 반환 (중복 초기화 방지)

    이 함수를 통해 모든 로거가 중앙 집중적으로 관리됩니다.
    """
    return _get_logger(*args, **kwargs)


def load_config(config_name: str = "main"):
    """필요시에만 설정 로드 (unified_config 사용)"""
    from .unified_config import load_config as _load_config

    return _load_config(config_name)


def __getattr__(name: str) -> Any:
    """모듈 레벨 지연 로딩 (Thread-Safe)"""
    module_mapping = {
        "get_profiler": ("unified_performance", "get_profiler"),
        "get_system_stats": ("system_diagnostics", "get_system_stats"),
        "cleanup_resources": ("unified_performance", "cleanup_resources"),
        "get_cuda_optimizer": ("cuda_optimizers", "get_cuda_optimizer"),
        "HybridOptimizer": ("cuda_optimizers", "HybridOptimizer"),
        "get_memory_manager": ("memory_manager", "get_memory_manager"),
        "MemoryManager": ("memory_manager", "MemoryManager"),
        "start_performance_monitoring": ("unified_performance", "start_monitoring"),
        "stop_performance_monitoring": ("unified_performance", "stop_monitoring"),
        "get_cuda_statistics": ("cuda_optimizers", "get_cuda_memory_info"),
    }

    if name in module_mapping:
        module_name, item_name = module_mapping[name]
        cache_key = f"{module_name}.{item_name}"
        with _import_lock:
            if cache_key not in _imported_items:
                try:
                    module = __import__(
                        f"src.utils.{module_name}", fromlist=[item_name]
                    )
                    _imported_items[cache_key] = getattr(module, item_name)
                except ImportError as e:
                    logger = _get_logger("utils.lazy_import")
                    logger.error(f"Failed to import {module_name}.{item_name}: {e}")
                    raise
            return _imported_items[cache_key]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 즉시 로드하지 않고 함수로 제공
__all__ = [
    # 즉시 사용 가능한 항목
    "get_logger",
    "load_config",
    "CACHE_DIR",
    "get_cache_dir",
    "get_logging_stats",
    "cleanup_logging",
    "get_optimization_report",
    # 지연 로딩 항목
    "get_profiler",
    "get_system_stats",
    "cleanup_resources",
    "get_cuda_optimizer",
    "HybridOptimizer",
    "get_memory_manager",
    "MemoryManager",
    "start_performance_monitoring",
    "stop_performance_monitoring",
    "get_cuda_statistics",
]
