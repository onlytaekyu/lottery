"""
에러 핸들링 및 로깅 유틸리티

이 모듈은 로깅 설정 및 에러 핸들링 기능을 제공합니다.
"""

import logging
import sys
import traceback
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, Type, Callable
from functools import wraps

# 로거 캐시
_loggers = {}

# 허용된 로그 파일 목록
ALLOWED_LOG_FILES = {
    "file_usage": "logs/file_usage.log",
    "lottery": "logs/lottery.log",
}

# 로그 핸들러 캐시 (파일 경로별로 핸들러 재사용)
_log_handlers = {}

# 로깅 시스템 초기화 여부 플래그
_logging_initialized = False


def log_exception_with_trace(
    logger: logging.Logger, exception: Exception, context: str = ""
) -> None:
    """
    예외와 스택 트레이스를 함께 로깅합니다.

    Args:
        logger: 사용할 로거 인스턴스
        exception: 로깅할 예외 객체
        context: 예외 컨텍스트 설명 (선택사항)

    이 함수는 예외 내용과 전체 트레이스백을 로깅하여 디버깅을 용이하게 합니다.
    다음 정보를 포함합니다:
    - 컨텍스트 설명
    - 예외 유형과 메시지
    - 전체 스택 트레이스 (파일 이름, 줄 번호, 코드 포함)
    """
    # 컨텍스트가 있으면 컨텍스트와 함께 로깅
    if context:
        logger.error(f"{context} - 예외 발생: {repr(exception)}")
    else:
        logger.error(f"예외 발생: {repr(exception)}")

    # 전체 트레이스 로깅
    logger.error("트레이스백:\n" + traceback.format_exc())


def setup_logger(
    name: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    file_level: Union[int, str] = logging.DEBUG,
    console_level: Union[int, str] = logging.INFO,
    log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
) -> logging.Logger:
    """
    로거 설정

    Args:
        name: 로거 이름 (None이면 루트 로거)
        level: 기본 로깅 레벨
        log_file: 로그 파일 경로
        console: 콘솔 출력 여부
        file_level: 파일 로깅 레벨
        console_level: 콘솔 로깅 레벨
        log_format: 로그 포맷

    Returns:
        설정된 로거
    """
    # 로깅 레벨 문자열 변환
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper(), logging.DEBUG)
    if isinstance(console_level, str):
        console_level = getattr(logging, console_level.upper(), logging.INFO)

    # 로거 가져오기
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 포맷터 설정
    formatter = logging.Formatter(log_format)

    # 파일 핸들러 설정
    if log_file:
        # 로그 파일 경로 정규화
        log_file_norm = os.path.normpath(log_file)

        # 허용된 로그 파일인지 확인
        use_allowed_file = False
        for allowed_file in ALLOWED_LOG_FILES.values():
            if os.path.normpath(allowed_file) == log_file_norm:
                use_allowed_file = True
                break

        # 허용되지 않은 로그 파일은 lottery.log로 변경
        if not use_allowed_file:
            log_file = ALLOWED_LOG_FILES["lottery"]
            log_file_norm = os.path.normpath(log_file)
            logger.warning(
                f"허용되지 않은 로그 파일 요청: {log_file}. lottery.log로 대체합니다."
            )

        # 로그 디렉토리 생성
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 파일 핸들러가 이미 캐시에 있는지 확인
        if log_file_norm in _log_handlers:
            file_handler = _log_handlers[log_file_norm]
        else:
            # 새 파일 핸들러 생성
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            _log_handlers[log_file_norm] = file_handler

        logger.addHandler(file_handler)

    # 콘솔 핸들러 설정
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    로거 가져오기 (캐싱)

    Args:
        name: 로거 이름

    Returns:
        로거 인스턴스
    """
    global _loggers, _logging_initialized

    # 이미 생성된 로거가 있으면 반환
    if name in _loggers:
        return _loggers[name]

    # 로깅 시스템이 초기화되지 않았으면 초기화
    if not _logging_initialized:
        init_logging_system()

    # 새 로거 생성
    logger = logging.getLogger(name)

    # 핸들러가 없으면 기본 핸들러 추가
    if not logger.handlers:
        # 루트 로거에 핸들러가 있는지 확인
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            # 로그 파일 유형 결정 (data_loader, file_manager 등은 file_usage.log 사용)
            log_file_key = (
                "file_usage"
                if any(
                    x in name.lower()
                    for x in ["data", "file", "loader", "io", "cache", "disk"]
                )
                else "lottery"
            )

            # 기본 로그 파일 설정
            log_file = ALLOWED_LOG_FILES[log_file_key]

            # 기본 핸들러 추가
            setup_logger(
                name=None,  # 루트 로거
                level=logging.INFO,
                log_file=log_file,
                console=True,
                file_level=logging.DEBUG,
                console_level=logging.INFO,
            )

    # 캐시에 저장
    _loggers[name] = logger
    return logger


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    전역 예외 핸들러

    Args:
        exc_type: 예외 타입
        exc_value: 예외 값
        exc_traceback: 예외 트레이스백
    """
    # KeyboardInterrupt는 정상 출력
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # 로거 가져오기
    logger = get_logger("error")

    # 예외 로깅
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


# 전역 예외 핸들러 설정
sys.excepthook = handle_exception


# 로깅 시스템 초기화
def init_logging_system():
    """로깅 시스템을 초기화합니다."""
    global _logging_initialized

    # 이미 초기화된 경우 중복 초기화 방지
    if _logging_initialized:
        return

    # 로그 디렉토리 생성
    log_dir = Path(ALLOWED_LOG_FILES["lottery"]).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # 루트 로거 설정
    setup_logger(
        name=None,  # 루트 로거
        level=logging.INFO,
        log_file=ALLOWED_LOG_FILES["lottery"],
        console=True,
        file_level=logging.DEBUG,
        console_level=logging.INFO,
    )

    # 파일 사용 로거 설정
    file_usage_logger = logging.getLogger("file_usage")
    setup_logger(
        name="file_usage",
        level=logging.INFO,
        log_file=ALLOWED_LOG_FILES["file_usage"],
        console=False,
        file_level=logging.DEBUG,
    )

    logging.info("로깅 시스템이 초기화되었습니다")
    logging.info(f"허용된 로그 파일: {', '.join(ALLOWED_LOG_FILES.values())}")

    # 초기화 완료 플래그 설정
    _logging_initialized = True


class StrictErrorHandler:
    """엄격한 에러 처리 시스템"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
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
    """엄격한 에러 처리 데코레이터"""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = StrictErrorHandler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler.handle_critical_error(
                    e, f"{context} - {func.__name__}", exit_on_error
                )
                return None

        return wrapper

    return decorator


def validate_and_fail_fast(condition: bool, error_message: str, data: Any = None):
    """조건 검증 후 실패 시 즉시 종료"""
    if not condition:
        handler = StrictErrorHandler()
        error = ValueError(error_message)
        handler.handle_validation_error(error, error_message, data)


# 안전 모드 시스템
class SafeModeManager:
    """안전 모드 관리자"""

    def __init__(self):
        self.safe_mode = False
        self.logger = get_logger(__name__)

    def enable_safe_mode(self, reason: str):
        """안전 모드 활성화"""
        self.safe_mode = True
        self.logger.warning(f"🛡️ 안전 모드 활성화: {reason}")

    def is_safe_mode(self) -> bool:
        """안전 모드 상태 확인"""
        return self.safe_mode

    def safe_execute(self, func: Callable, *args, **kwargs):
        """안전 모드에서 함수 실행"""
        if self.safe_mode:
            self.logger.info("🛡️ 안전 모드에서 실행 중...")
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"안전 모드 실행 실패: {str(e)}")
                return None
        else:
            return func(*args, **kwargs)


# 전역 안전 모드 관리자
safe_mode_manager = SafeModeManager()
