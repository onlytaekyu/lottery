"""
로또 번호 예측 시스템 - 통합 유틸리티

이 패키지는 중복 제거 및 통합을 통해 최적화된 유틸리티 기능을 제공합니다.
"""

# 통합 시스템들 import
from .unified_logging import UnifiedLogger, LogLevel, get_logger, log_exception
from .unified_performance import (
    UnifiedPerformanceTracker,
    PerformanceMetrics,
    performance_monitor,
    get_performance_manager,
    get_profiler,
    profile,
    clear_memory,
    get_device_info,
)
from .unified_config import (
    UnifiedConfigManager,
    get_paths,
    load_config as get_unified_config,
)
from .unified_validation import (
    UnifiedValidationManager,
    ValidationLevel,
    validate_vector,
)
from .unified_report import (
    UnifiedReportWriter,
    save_report,
    save_performance_report,
    save_analysis_report,
    save_training_report,
    save_evaluation_report,
    save_analysis_performance_report,
    get_system_info,
)
from .error_handler_refactored import (
    StrictErrorHandler,
    strict_error_handler,
    SafeModeManager,
)

# 기존 유지할 유틸리티들
from .config_loader import load_config, ConfigProxy
from .data_loader import load_draw_history
from .cache_paths import get_cache_dir, CACHE_DIR
from .vector_exporter import save_vector_bundle
from .memory_manager import MemoryManager, MemoryConfig
from .cuda_optimizers import CudaConfig
from .model_saver import ModelSaver
from .cache_manager import CacheManager
from .normalizer import Normalizer


# 편의 함수들
def initialize_unified_systems():
    """통합 시스템들을 초기화합니다."""
    logger = get_logger("utils.init")
    logger.info("통합 시스템 초기화 시작")

    try:
        # 설정 관리자 초기화
        config_manager = UnifiedConfigManager()
        logger.info("설정 관리자 초기화 완료")

        # 성능 추적기 초기화
        performance_tracker = UnifiedPerformanceTracker()
        logger.info("성능 추적기 초기화 완료")

        # 검증 관리자 초기화
        validation_manager = UnifiedValidationManager()
        logger.info("검증 관리자 초기화 완료")

        # 보고서 작성기 초기화
        report_writer = UnifiedReportWriter()
        logger.info("보고서 작성기 초기화 완료")

        logger.info("통합 시스템 초기화 완료")
        return True

    except Exception as e:
        logger.error(f"통합 시스템 초기화 실패: {e}")
        return False


def get_system_status():
    """통합 시스템들의 상태를 반환합니다."""
    status = {}

    try:
        # 로깅 시스템 상태
        logger = UnifiedLogger()
        status["logging"] = {
            "initialized": logger._initialized,
            "active_loggers": len(logger._loggers),
            "active_handlers": len(logger._file_handlers),
        }

        # 성능 추적기 상태
        tracker = get_performance_manager()
        status["performance"] = {
            "active_sessions": len(tracker.sessions),
            "profiling_enabled": tracker.config.enable_profiling,
        }

        # 설정 관리자 상태
        config_manager = UnifiedConfigManager()
        status["config"] = {
            "configs_loaded": len(config_manager._configs),
            "cache_hits": getattr(config_manager, "_cache_hits", 0),
        }

        # 검증 관리자 상태
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
        clear_memory()
        logger.info("메모리 정리 완료")

        # 성능 추적기 정리
        tracker = get_performance_manager()
        tracker.clear()
        logger.info("성능 추적기 정리 완료")

        logger.info("시스템 리소스 정리 완료")
        return True

    except Exception as e:
        logger.error(f"시스템 리소스 정리 실패: {e}")
        return False


# 전체 시스템 상태 확인
__all__ = [
    # 통합 시스템들
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
    "UnifiedConfigManager",
    "get_paths",
    "get_unified_config",
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
    "StrictErrorHandler",
    "strict_error_handler",
    "SafeModeManager",
    # 기존 유지 시스템들
    "load_config",
    "ConfigProxy",
    "load_draw_history",
    "get_cache_dir",
    "CACHE_DIR",
    "save_vector_bundle",
    "MemoryManager",
    "MemoryConfig",
    "CudaConfig",
    "ModelSaver",
    "CacheManager",
    "Normalizer",
    # 시스템 관리 함수들
    "initialize_unified_systems",
    "get_system_status",
    "cleanup_resources",
]
