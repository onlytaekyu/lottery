"""
로또 번호 예측 시스템 - 유틸리티

이 패키지는 로또 번호 예측 시스템에서 사용되는 유틸리티 기능을 제공합니다.
"""

# 통합 시스템들 import
from .unified_logging import UnifiedLogger, LogLevel, get_logger, log_exception
from .unified_performance import (
    UnifiedPerformanceTracker,
    PerformanceMetrics,
    performance_monitor,
)
from .unified_config import UnifiedConfigManager, get_config, get_paths
from .unified_validation import (
    UnifiedValidationManager,
    ValidationLevel,
    validate_vector,
)
from .error_handler_refactored import (
    StrictErrorHandler,
    strict_error_handler,
    SafeModeManager,
)


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
        tracker = UnifiedPerformanceTracker()
        status["performance"] = {
            "active_sessions": len(tracker.sessions),
            "profiling_enabled": tracker.config.profiling_enabled,
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


# 전체 시스템 상태 확인
__all__ = [
    "UnifiedLogger",
    "LogLevel",
    "get_logger",
    "log_exception",
    "UnifiedPerformanceTracker",
    "PerformanceMetrics",
    "performance_monitor",
    "UnifiedConfigManager",
    "get_config",
    "get_paths",
    "UnifiedValidationManager",
    "ValidationLevel",
    "validate_vector",
    "StrictErrorHandler",
    "strict_error_handler",
    "SafeModeManager",
    "initialize_unified_systems",
    "get_system_status",
]
