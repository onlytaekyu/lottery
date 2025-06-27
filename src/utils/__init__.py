"""
로또 번호 예측 시스템 - 통합 유틸리티 (최적화 버전)

필요한 모듈만 런타임에 로드하여 초기 로딩 시간을 최적화합니다.
"""

from typing import Any, Optional

# 핵심 시스템만 즉시 로드
from .cache_paths import get_cache_dir, CACHE_DIR

# Lazy import 캐시
_imported_items = {}

def get_logger(*args, **kwargs):
    """필요시에만 로거 로드"""
    if "logger" not in _imported_items:
        from .unified_logging import get_logger as _get_logger

        _imported_items["logger"] = _get_logger
    return _imported_items["logger"](*args, **kwargs)

def get_profiler():
    """필요시에만 프로파일러 로드"""
    if "profiler" not in _imported_items:
        from .unified_performance import get_profiler as _get_profiler

        _imported_items["profiler"] = _get_profiler
    return _imported_items["profiler"]()

def load_config(config_name: str = "main"):
    """필요시에만 설정 로드 (unified_config 사용)"""
    if "load_config" not in _imported_items:
        from .unified_config import load_config as _load_config

        _imported_items["load_config"] = _load_config
    return _imported_items["load_config"](config_name)

def __getattr__(name: str) -> Any:
    """모듈 레벨 지연 로딩"""
    # 자주 사용되는 항목들의 모듈 매핑
    module_mapping = {
        # 클래스들
        "ConfigProxy": ("unified_config", "ConfigProxy"),
        "UnifiedPerformanceTracker": (
            "unified_performance",
            "UnifiedPerformanceTracker",
        ),
        "StrictErrorHandler": ("error_handler_refactored", "StrictErrorHandler"),
        "MemoryManager": ("memory_manager", "MemoryManager"),
        "ModelSaver": ("model_saver", "ModelSaver"),
        "CacheManager": ("cache_manager", "CacheManager"),
        # 함수들
        "save_report": ("unified_report", "save_report"),
        "load_draw_history": ("data_loader", "load_draw_history"),
        "save_vector_bundle": ("vector_exporter", "save_vector_bundle"),
        "save_feature_names": ("feature_name_tracker", "save_feature_names"),
        "load_feature_names": ("feature_name_tracker", "load_feature_names"),
        "strict_error_handler": ("error_handler_refactored", "strict_error_handler"),
        "validate_vector": ("unified_validation", "validate_vector"),
    }

    if name in module_mapping:
        module_name, item_name = module_mapping[name]
        cache_key = f"{module_name}.{item_name}"

        if cache_key not in _imported_items:
            try:
                module = __import__(f"src.utils.{module_name}", fromlist=[item_name])
                _imported_items[cache_key] = getattr(module, item_name)
            except ImportError as e:
                logger = get_logger("utils.lazy_import")
                logger.error(f"Failed to import {module_name}.{item_name}: {e}")
                raise

        return _imported_items[cache_key]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def cleanup_resources():
    """리소스 정리"""
    global _imported_items
    _imported_items.clear()

def get_import_stats():
    """import 통계 반환"""
    return {
        "loaded_items": len(_imported_items),
        "loaded_modules": list(
            set(key.split(".")[0] for key in _imported_items.keys())
        ),
    }

# 즉시 로드하지 않고 함수로 제공
__all__ = [
    "get_logger",
    "get_profiler",
    "load_config",
    "CACHE_DIR",
    "get_cache_dir",
    "cleanup_resources",
]
