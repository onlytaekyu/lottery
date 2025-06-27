"""
에러 핸들링 유틸리티 (리팩토링된 버전)

통합 로깅 시스템을 기반으로 한 에러 핸들링 기능을 제공합니다.
중복 코드를 제거하고 통합된 시스템을 사용합니다.
"""

import sys
import traceback
from typing import Any, Callable
from functools import wraps

# 통합 시스템 사용
from .unified_logging import get_logger, log_exception, init_logging_system

class StrictErrorHandler:
    """엄격한 에러 처리 시스템"""

    def __init__(self, logger_name: str = "error"):
        self.logger = get_logger(logger_name)
        self.error_count = 0
        self.max_errors = 3  # 최대 허용 에러 수

    def handle_critical_error(
        self, error: Exception, context: str, exit_immediately: bool = True
    ):
        """치명적 에러 처리 - 즉시 시스템 종료"""
        self.error_count += 1

        error_msg = (
            f"🚨 치명적 에러 발생 [{self.error_count}/{self.max_errors}]: {context}"
        )
        self.logger.critical(error_msg)
        self.logger.critical(f"에러 유형: {type(error).__name__}")
        self.logger.critical(f"에러 메시지: {str(error)}")
        self.logger.critical(f"스택 트레이스:\n{traceback.format_exc()}")

        if exit_immediately or self.error_count >= self.max_errors:
            self.logger.critical("🛑 시스템을 즉시 종료합니다.")
            sys.exit(1)

    def handle_validation_error(self, error: Exception, context: str, data: Any = None):
        """데이터 검증 에러 처리"""
        self.logger.error(f"❌ 데이터 검증 실패: {context}")
        self.logger.error(f"에러: {str(error)}")
        if data is not None:
            self.logger.error(f"문제 데이터: {data}")

        # 검증 에러는 즉시 시스템 종료
        self.handle_critical_error(error, f"데이터 검증 실패: {context}")

def strict_error_handler(context: str, exit_on_error: bool = True):
    """
    엄격한 에러 처리 데코레이터

    Args:
        context: 에러 컨텍스트
        exit_on_error: 에러 시 시스템 종료 여부
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = StrictErrorHandler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_critical_error(e, context, exit_on_error)

        return wrapper

    return decorator

def validate_and_fail_fast(condition: bool, error_message: str, data: Any = None):
    """
    조건 검증 후 실패 시 즉시 시스템 종료

    Args:
        condition: 검증할 조건
        error_message: 에러 메시지
        data: 관련 데이터 (선택사항)
    """
    if not condition:
        error_handler = StrictErrorHandler()
        error = RuntimeError(error_message)
        error_handler.handle_validation_error(error, error_message, data)

class SafeModeManager:
    """안전 모드 관리자"""

    def __init__(self):
        self.safe_mode = False
        self.logger = get_logger("safe_mode")

    def enable_safe_mode(self, reason: str):
        """안전 모드 활성화"""
        self.safe_mode = True
        self.logger.warning(f"안전 모드 활성화: {reason}")

    def is_safe_mode(self) -> bool:
        """안전 모드 여부 확인"""
        return self.safe_mode

    def safe_execute(self, func: Callable, *args, **kwargs):
        """안전 모드에서 함수 실행"""
        if self.safe_mode:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"안전 모드에서 함수 실행 실패: {str(e)}")
                return None
        else:
            return func(*args, **kwargs)

# 편의 함수들 (기존 API 호환성)
def log_exception_with_trace(logger_name: str, exception: Exception, context: str = ""):
    """예외와 스택 트레이스를 함께 로깅 (통합 시스템 사용)"""
    log_exception(logger_name, exception, context)

def handle_exception(exc_type, exc_value, exc_traceback):
    """전역 예외 핸들러 (통합 시스템에서 자동 처리됨)"""
    # 통합 로깅 시스템에서 자동으로 처리하므로 별도 구현 불필요

# 전역 인스턴스
_global_error_handler = None
_global_safe_mode_manager = None

def get_error_handler() -> StrictErrorHandler:
    """전역 에러 핸들러 반환"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = StrictErrorHandler()
    return _global_error_handler

def get_safe_mode_manager() -> SafeModeManager:
    """전역 안전 모드 관리자 반환"""
    global _global_safe_mode_manager
    if _global_safe_mode_manager is None:
        _global_safe_mode_manager = SafeModeManager()
    return _global_safe_mode_manager

# 기존 코드와의 호환성을 위한 함수들
def setup_logger(*args, **kwargs):
    """기존 setup_logger 호환성 함수 (사용 권장하지 않음)"""
    logger = get_logger("deprecated_setup_logger")
    logger.warning(
        "setup_logger는 더 이상 사용되지 않습니다. unified_logging.get_logger를 사용하세요."
    )
    return get_logger(kwargs.get("name", "default"))

# 자동 초기화
init_logging_system()
