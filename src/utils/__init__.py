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


def get_profiler():
    """필요시에만 프로파일러 로드"""
    with _import_lock:
        if "profiler" not in _imported_items:
            from .unified_performance import get_profiler as _get_profiler

            _imported_items["profiler"] = _get_profiler
    return _imported_items["profiler"]()


def load_config(config_name: str = "main"):
    """필요시에만 설정 로드 (unified_config 사용)"""
    with _import_lock:
        if "load_config" not in _imported_items:
            from .unified_config import load_config as _load_config

            _imported_items["load_config"] = _load_config
    return _imported_items["load_config"](config_name)


def auto_assign_logger(module_name: str):
    """
    모듈에 자동으로 로거 할당

    각 모듈에서 개별적으로 logger = get_logger(__name__)를 호출하는 대신
    이 함수를 통해 자동으로 로거가 할당됩니다.

    Args:
        module_name: 모듈 이름 (__name__ 값)

    Returns:
        최적화된 로거 인스턴스
    """
    return get_logger(module_name)


def __getattr__(name: str) -> Any:
    """모듈 레벨 지연 로딩 (Thread-Safe)"""
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
        "OptimizedLoggerFactory": ("unified_logging", "OptimizedLoggerFactory"),
        # 함수들
        "save_report": ("unified_report", "save_report"),
        "load_draw_history": ("data_loader", "load_draw_history"),
        "save_vector_bundle": ("vector_exporter", "save_vector_bundle"),
        "save_feature_names": ("feature_name_tracker", "save_feature_names"),
        "load_feature_names": ("feature_name_tracker", "load_feature_names"),
        "strict_error_handler": ("error_handler_refactored", "strict_error_handler"),
        "validate_vector": ("unified_validation", "validate_vector"),
        "log_exception": ("unified_logging", "log_exception"),
        "log_exception_with_trace": ("unified_logging", "log_exception_with_trace"),
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
                    logger = get_logger("utils.lazy_import")
                    logger.error(f"Failed to import {module_name}.{item_name}: {e}")
                    raise

        return _imported_items[cache_key]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_system_stats() -> dict:
    """시스템 통계 반환"""
    return {
        "logging_stats": get_logging_stats(),
        "imported_items": len(_imported_items),
        "cached_modules": list(_imported_items.keys()),
    }


def cleanup_resources():
    """리소스 정리 (완전 정리)"""
    global _imported_items

    with _import_lock:
        # 로깅 시스템 정리
        cleanup_logging()

        # 캐시 정리
        _imported_items.clear()


# 최적화된 로깅 시스템 함수들 추가
def test_logging_system():
    """로깅 시스템 테스트 실행"""
    with _import_lock:
        if "test_logging_system" not in _imported_items:
            from .test_optimized_logging import run_logging_system_test

            _imported_items["test_logging_system"] = run_logging_system_test
    return _imported_items["test_logging_system"]()


def get_optimization_report():
    """최적화 보고서 반환"""
    with _import_lock:
        if "get_optimization_report" not in _imported_items:
            from .logging_monitor import get_optimization_report as _get_report

            _imported_items["get_optimization_report"] = _get_report
    return _imported_items["get_optimization_report"]()


def start_logging_monitoring(interval_seconds: int = 60):
    """로깅 모니터링 시작"""
    with _import_lock:
        if "start_logging_monitoring" not in _imported_items:
            from .logging_monitor import start_logging_monitoring as _start_monitoring

            _imported_items["start_logging_monitoring"] = _start_monitoring
    return _imported_items["start_logging_monitoring"](interval_seconds)


# 즉시 로드하지 않고 함수로 제공
__all__ = [
    "get_logger",
    "auto_assign_logger",
    "get_profiler",
    "load_config",
    "CACHE_DIR",
    "get_cache_dir",
    "get_system_stats",
    "cleanup_resources",
    "test_logging_system",
    "get_optimization_report",
    "start_logging_monitoring",
]
