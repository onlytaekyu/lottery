"""
로또 번호 예측 시스템 - 통합 유틸리티 (최적화 버전)

이 패키지는 중복 제거 및 통합을 통해 최적화된 유틸리티 기능을 제공합니다.
모든 구형 시스템은 통합 시스템으로 대체되었습니다.
Lazy import 패턴을 적용하여 초기 로딩 시간을 최적화했습니다.
"""

import sys
from typing import Any, Dict, Optional, TYPE_CHECKING

# 🚀 핵심 시스템만 즉시 로드 (가벼운 모듈들)
from .unified_logging import get_logger
from .cache_paths import get_cache_dir, CACHE_DIR

# 🔄 Lazy import 용 모듈 레지스트리
_lazy_modules = {
    # 통합 시스템들
    "unified_performance": {
        "module": "src.utils.unified_performance",
        "classes": [
            "UnifiedPerformanceTracker",
            "PerformanceMetrics",
            "Profiler",
            "PerformanceTracker",
            "MemoryTracker",
        ],
        "functions": [
            "performance_monitor",
            "get_performance_manager",
            "get_profiler",
            "profile",
            "clear_memory",
            "get_device_info",
        ],
    },
    "unified_config": {
        "module": "src.utils.unified_config",
        "classes": ["ConfigProxy", "CudaConfig"],
        "functions": ["get_config_manager", "load_config"],
    },
    "unified_validation": {
        "module": "src.utils.unified_validation",
        "classes": ["UnifiedValidationManager", "ValidationLevel"],
        "functions": ["validate_vector"],
    },
    "unified_report": {
        "module": "src.utils.unified_report",
        "classes": ["UnifiedReportWriter"],
        "functions": [
            "save_report",
            "save_performance_report",
            "save_analysis_report",
            "save_training_report",
            "save_evaluation_report",
            "save_analysis_performance_report",
            "get_system_info",
            "safe_convert",
        ],
    },
    "error_handler_refactored": {
        "module": "src.utils.error_handler_refactored",
        "classes": ["StrictErrorHandler", "SafeModeManager"],
        "functions": ["strict_error_handler", "get_error_handler"],
    },
    # 기존 시스템들
    "config_loader": {
        "module": "src.utils.config_loader",
        "functions": ["load_config", "get_default_config", "save_config"],
    },
    "data_loader": {
        "module": "src.utils.data_loader",
        "functions": ["load_draw_history"],
    },
    "vector_exporter": {
        "module": "src.utils.vector_exporter",
        "functions": ["save_vector_bundle"],
    },
    "memory_manager": {
        "module": "src.utils.memory_manager",
        "classes": ["ThreadLocalCache", "MemoryManager"],
        "functions": ["get_memory_manager"],
    },
    "model_saver": {"module": "src.utils.model_saver", "classes": ["ModelSaver"]},
    "cache_manager": {"module": "src.utils.cache_manager", "classes": ["CacheManager"]},
    "normalizer": {"module": "src.utils.normalizer", "classes": ["Normalizer"]},
    "feature_name_tracker": {
        "module": "src.utils.feature_name_tracker",
        "functions": ["save_feature_names", "load_feature_names"],
    },
}

# 🔄 Lazy import 캐시
_imported_modules = {}
_imported_items = {}


def _lazy_import(module_name: str, item_name: str) -> Any:
    """지연 로딩으로 모듈 항목을 가져옵니다."""
    cache_key = f"{module_name}.{item_name}"

    # 캐시에서 확인
    if cache_key in _imported_items:
        return _imported_items[cache_key]

    # 모듈 정보 확인
    if module_name not in _lazy_modules:
        raise ImportError(f"Unknown lazy module: {module_name}")

    module_info = _lazy_modules[module_name]

    # 모듈 로드
    if module_name not in _imported_modules:
        try:
            # 상대 경로를 절대 경로로 변환
            if module_info["module"].startswith("src.utils."):
                module_path = module_info["module"][4:]  # 'src.' 제거
                from . import __import__ as utils_import

                imported_module = __import__(module_path, fromlist=[item_name])
            else:
                imported_module = __import__(
                    module_info["module"], fromlist=[item_name]
                )

            _imported_modules[module_name] = imported_module
        except ImportError as e:
            logger = get_logger("utils.lazy_import")
            logger.error(f"Failed to import module {module_info['module']}: {e}")
            raise

    # 항목 가져오기
    imported_module = _imported_modules[module_name]
    if hasattr(imported_module, item_name):
        item = getattr(imported_module, item_name)
        _imported_items[cache_key] = item
        return item
    else:
        raise AttributeError(f"Module {module_name} has no attribute {item_name}")


def __getattr__(name: str) -> Any:
    """모듈 레벨 지연 로딩"""
    # 모든 lazy 모듈에서 해당 항목 검색
    for module_name, module_info in _lazy_modules.items():
        all_items = module_info.get("classes", []) + module_info.get("functions", [])
        if name in all_items:
            return _lazy_import(module_name, name)

    # 특별한 경우들 처리
    if name == "UnifiedLogger":
        from .unified_logging import UnifiedLogger

        return UnifiedLogger
    elif name == "LogLevel":
        from .unified_logging import LogLevel

        return LogLevel
    elif name == "log_exception":
        from .unified_logging import log_exception

        return log_exception

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 🚀 편의 함수들 (즉시 로드)
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


def initialize_unified_systems():
    """통합 시스템들을 초기화합니다."""
    logger = get_logger("utils.init")
    logger.info("통합 시스템 초기화 시작")

    try:
        # 설정 관리자 초기화
        get_config_manager = _lazy_import("unified_config", "get_config_manager")
        get_config_manager()
        logger.info("설정 관리자 초기화 완료")

        # 성능 추적기 초기화
        UnifiedPerformanceTracker = _lazy_import(
            "unified_performance", "UnifiedPerformanceTracker"
        )
        UnifiedPerformanceTracker()
        logger.info("성능 추적기 초기화 완료")

        # 검증 관리자 초기화
        UnifiedValidationManager = _lazy_import(
            "unified_validation", "UnifiedValidationManager"
        )
        UnifiedValidationManager()
        logger.info("검증 관리자 초기화 완료")

        # 보고서 작성기 초기화
        UnifiedReportWriter = _lazy_import("unified_report", "UnifiedReportWriter")
        UnifiedReportWriter()
        logger.info("보고서 작성기 초기화 완료")

        logger.info("✅ 통합 시스템 초기화 완료")
        return True

    except Exception as e:
        logger.error(f"❌ 통합 시스템 초기화 실패: {e}")
        return False


def get_system_status():
    """통합 시스템들의 상태를 반환합니다."""
    status = {}

    try:
        # 로깅 시스템 상태
        from .unified_logging import UnifiedLogger

        logger = UnifiedLogger()
        status["logging"] = {
            "initialized": logger._initialized,
            "active_loggers": len(logger._loggers),
            "active_handlers": len(logger._file_handlers),
        }

        # 성능 추적기 상태
        get_performance_manager = _lazy_import(
            "unified_performance", "get_performance_manager"
        )
        tracker = get_performance_manager()
        status["performance"] = {
            "active_sessions": len(tracker.sessions),
            "profiling_enabled": tracker.config.enable_profiling,
        }

        # 설정 관리자 상태
        get_config_manager = _lazy_import("unified_config", "get_config_manager")
        config_manager = get_config_manager()
        status["config"] = {
            "configs_loaded": len(config_manager._configs),
            "cache_hits": getattr(config_manager, "_cache_hits", 0),
        }

        # 검증 관리자 상태
        UnifiedValidationManager = _lazy_import(
            "unified_validation", "UnifiedValidationManager"
        )
        validation_manager = UnifiedValidationManager()
        status["validation"] = {
            "registered_validators": len(validation_manager.validators)
        }

    except Exception as e:
        status["error"] = str(e)

    return status


def cleanup_resources():
    """시스템 리소스 정리"""
    logger = get_logger("utils.cleanup")
    logger.info("시스템 리소스 정리 시작")

    try:
        # 메모리 정리
        clear_memory = _lazy_import("unified_performance", "clear_memory")
        clear_memory()
        logger.info("메모리 정리 완료")

        # 성능 추적기 정리
        get_performance_manager = _lazy_import(
            "unified_performance", "get_performance_manager"
        )
        tracker = get_performance_manager()
        tracker.clear()
        logger.info("성능 추적기 정리 완료")

        # 임포트 캐시 정리
        _imported_modules.clear()
        _imported_items.clear()
        logger.info("임포트 캐시 정리 완료")

        logger.info("✅ 시스템 리소스 정리 완료")
        return True

    except Exception as e:
        logger.error(f"❌ 시스템 리소스 정리 실패: {e}")
        return False


def get_import_stats():
    """Import 통계 반환"""
    return {
        "loaded_modules": len(_imported_modules),
        "cached_items": len(_imported_items),
        "available_lazy_modules": len(_lazy_modules),
        "loaded_module_names": list(_imported_modules.keys()),
    }


# 📋 전체 시스템 export
__all__ = [
    # 🎯 통합 시스템들 (Lazy loaded)
    "UnifiedLogger",
    "LogLevel",
    "get_logger",
    "log_exception",
    "UnifiedPerformanceTracker",
    "PerformanceMetrics",
    "performance_monitor",
    "get_performance_manager",
    "get_profiler",
    "profile",
    "clear_memory",
    "get_device_info",
    "Profiler",  # 호환성
    "PerformanceTracker",  # 호환성
    "MemoryTracker",  # 호환성
    "UnifiedValidationManager",
    "ValidationLevel",
    "validate_vector",
    "UnifiedReportWriter",
    "save_report",
    "save_performance_report",
    "save_analysis_report",
    "save_training_report",
    "save_evaluation_report",
    "save_analysis_performance_report",
    "get_system_info",
    "safe_convert",
    "StrictErrorHandler",
    "strict_error_handler",
    "SafeModeManager",
    "get_error_handler",
    "ConfigProxy",
    "get_config_manager",
    "load_config",
    "CudaConfig",
    # 🔧 기존 유지 시스템들 (Lazy loaded)
    "get_default_config",
    "save_config",
    "load_draw_history",
    "save_vector_bundle",
    "MemoryManager",
    "ThreadLocalCache",
    "get_memory_manager",
    "ModelSaver",
    "CacheManager",
    "Normalizer",
    "save_feature_names",
    "load_feature_names",
    # 🚀 시스템 관리 함수들 (즉시 로드)
    "get_cache_dir",
    "CACHE_DIR",
    "initialize_unified_systems",
    "get_system_status",
    "cleanup_resources",
    "get_heavy_module",
    "get_import_stats",
]

# 🎉 시스템 초기화 메시지
_logger = get_logger("utils")
_logger.info("✅ 통합 유틸리티 시스템 로드 완료 - Lazy import 패턴 적용으로 최적화됨")
