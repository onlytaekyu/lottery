"""
로또 번호 예측 시스템 - 통합 유틸리티 (완전 최적화 버전)

Lazy import 패턴을 적용하여 초기 로딩 시간을 최적화했습니다.
필요한 모듈만 런타임에 로드됩니다.
"""

from typing import Any, Optional

# 🚀 핵심 시스템만 즉시 로드 (가벼운 모듈들)
from .cache_paths import get_cache_dir, CACHE_DIR

# 🔄 Lazy import 캐시
_imported_items = {}


def get_logger(name: Optional[str] = None):
    """필요시에만 로거 로드"""
    if "logger" not in _imported_items:
        from .unified_logging import get_logger as _get_logger

        _imported_items["logger"] = _get_logger
    return _imported_items["logger"](name)


def get_profiler():
    """필요시에만 프로파일러 로드"""
    if "profiler" not in _imported_items:
        from .unified_performance import get_profiler as _get_profiler

        _imported_items["profiler"] = _get_profiler
    return _imported_items["profiler"]()


def get_config_manager():
    """필요시에만 설정 관리자 로드"""
    if "config_manager" not in _imported_items:
        from .unified_config import get_config_manager as _get_config_manager

        _imported_items["config_manager"] = _get_config_manager
    return _imported_items["config_manager"]()


def load_config(config_name: str = "main"):
    """필요시에만 설정 로드"""
    if "load_config" not in _imported_items:
        from .unified_config import load_config as _load_config

        _imported_items["load_config"] = _load_config
    return _imported_items["load_config"](config_name)


def get_heavy_module(module_name: str):
    """무거운 모듈을 필요시에만 로드"""
    heavy_modules = {
        "torch": lambda: __import__("torch"),
        "psutil": lambda: __import__("psutil"),
        "sklearn": lambda: __import__("sklearn"),
        "tensorrt": lambda: __import__("tensorrt"),
    }

    if module_name in heavy_modules:
        try:
            return heavy_modules[module_name]()
        except ImportError:
            logger = get_logger("utils.heavy_import")
            logger.warning(f"Heavy module {module_name} not available")
            return None
    else:
        raise ValueError(f"Unknown heavy module: {module_name}")


def __getattr__(name: str) -> Any:
    """모듈 레벨 지연 로딩"""
    # 자주 사용되는 클래스들
    lazy_classes = {
        "UnifiedPerformanceTracker": (
            "unified_performance",
            "UnifiedPerformanceTracker",
        ),
        "ConfigProxy": ("unified_config", "ConfigProxy"),
        "UnifiedValidationManager": ("unified_validation", "UnifiedValidationManager"),
        "UnifiedReportWriter": ("unified_report", "UnifiedReportWriter"),
        "StrictErrorHandler": ("error_handler_refactored", "StrictErrorHandler"),
        "SafeModeManager": ("error_handler_refactored", "SafeModeManager"),
        "MemoryManager": ("memory_manager", "MemoryManager"),
        "ModelSaver": ("model_saver", "ModelSaver"),
        "CacheManager": ("cache_manager", "CacheManager"),
        "Normalizer": ("normalizer", "Normalizer"),
    }

    # 자주 사용되는 함수들
    lazy_functions = {
        "performance_monitor": ("unified_performance", "performance_monitor"),
        "profile": ("unified_performance", "profile"),
        "validate_vector": ("unified_validation", "validate_vector"),
        "save_report": ("unified_report", "save_report"),
        "strict_error_handler": ("error_handler_refactored", "strict_error_handler"),
        "load_draw_history": ("data_loader", "load_draw_history"),
        "save_vector_bundle": ("vector_exporter", "save_vector_bundle"),
        "save_feature_names": ("feature_name_tracker", "save_feature_names"),
        "load_feature_names": ("feature_name_tracker", "load_feature_names"),
    }

    # 클래스 지연 로딩
    if name in lazy_classes:
        module_name, class_name = lazy_classes[name]
        cache_key = f"{module_name}.{class_name}"

        if cache_key not in _imported_items:
            try:
                module = __import__(f"src.utils.{module_name}", fromlist=[class_name])
                _imported_items[cache_key] = getattr(module, class_name)
            except ImportError as e:
                logger = get_logger("utils.lazy_import")
                logger.error(f"Failed to import {module_name}.{class_name}: {e}")
                raise

        return _imported_items[cache_key]

    # 함수 지연 로딩
    if name in lazy_functions:
        module_name, func_name = lazy_functions[name]
        cache_key = f"{module_name}.{func_name}"

        if cache_key not in _imported_items:
            try:
                module = __import__(f"src.utils.{module_name}", fromlist=[func_name])
                _imported_items[cache_key] = getattr(module, func_name)
            except ImportError as e:
                logger = get_logger("utils.lazy_import")
                logger.error(f"Failed to import {module_name}.{func_name}: {e}")
                raise

        return _imported_items[cache_key]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 🚀 편의 함수들
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


# 필수 export 목록 (지연 로딩)
__all__ = [
    # 핵심 함수들
    "get_logger",
    "get_profiler",
    "get_config_manager",
    "load_config",
    # 캐시 관련
    "get_cache_dir",
    "CACHE_DIR",
    # 유틸리티
    "get_heavy_module",
    "cleanup_resources",
    "get_import_stats",
]
