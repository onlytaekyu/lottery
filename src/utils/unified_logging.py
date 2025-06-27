"""
통합 로깅 시스템

모든 로깅 관련 기능을 중앙화하여 중복 코드를 제거하고 일관성을 보장합니다.
"""

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    """로그 레벨 열거형"""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogConfig:
    """로깅 설정"""

    name: Optional[str] = None
    level: LogLevel = LogLevel.INFO
    console_output: bool = True
    file_output: bool = True
    console_level: LogLevel = LogLevel.INFO
    file_level: LogLevel = LogLevel.DEBUG
    format_string: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    encoding: str = "utf-8"


class UnifiedLogger:
    """통합 로거 클래스"""

    # 클래스 레벨 캐시 및 설정
    _loggers: Dict[str, logging.Logger] = {}
    _handlers: Dict[str, logging.Handler] = {}
    _initialized: bool = False
    _lock = threading.RLock()

    # 허용된 로그 파일 경로
    ALLOWED_LOG_FILES = {
        "main": "logs/lottery.log",
        "file_usage": "logs/file_usage.log",
        "model": "logs/model.log",
        "performance": "logs/performance.log",
        "error": "logs/error.log",
    }

    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self._ensure_log_directories()

    @classmethod
    def _ensure_log_directories(cls):
        """로그 디렉토리 생성"""
        for log_file in cls.ALLOWED_LOG_FILES.values():
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_log_file_for_logger(cls, logger_name: str) -> str:
        """로거 이름에 따른 적절한 로그 파일 반환"""
        # 로거 이름 기반 로그 파일 매핑
        name_lower = logger_name.lower()

        if any(
            keyword in name_lower for keyword in ["file", "data", "io", "cache", "disk"]
        ):
            return cls.ALLOWED_LOG_FILES["file_usage"]
        elif any(keyword in name_lower for keyword in ["model", "train", "inference"]):
            return cls.ALLOWED_LOG_FILES["model"]
        elif any(
            keyword in name_lower for keyword in ["performance", "profiler", "memory"]
        ):
            return cls.ALLOWED_LOG_FILES["performance"]
        elif any(
            keyword in name_lower for keyword in ["error", "exception", "critical"]
        ):
            return cls.ALLOWED_LOG_FILES["error"]
        else:
            return cls.ALLOWED_LOG_FILES["main"]

    @classmethod
    def _create_formatter(cls, config: LogConfig) -> logging.Formatter:
        """포맷터 생성"""
        return logging.Formatter(fmt=config.format_string, datefmt=config.date_format)

    @classmethod
    def _create_console_handler(cls, config: LogConfig) -> logging.StreamHandler:
        """콘솔 핸들러 생성"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(config.console_level.value)
        handler.setFormatter(cls._create_formatter(config))
        return handler

    @classmethod
    def _create_file_handler(cls, log_file: str, config: LogConfig) -> logging.Handler:
        """파일 핸들러 생성 (회전 로그)"""
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding=config.encoding,
        )
        handler.setLevel(config.file_level.value)
        handler.setFormatter(cls._create_formatter(config))
        return handler

    @classmethod
    def get_logger(
        cls, name: str, config: Optional[LogConfig] = None
    ) -> logging.Logger:
        """
        통합 로거 반환 (싱글톤 패턴)

        Args:
            name: 로거 이름
            config: 로깅 설정 (선택사항)

        Returns:
            설정된 로거 인스턴스
        """
        with cls._lock:
            # 이미 생성된 로거가 있으면 반환
            if name in cls._loggers:
                return cls._loggers[name]

            # 전역 초기화
            if not cls._initialized:
                cls._initialize_logging_system()

            # 새 로거 생성
            logger = logging.getLogger(name)
            logger.setLevel((config or LogConfig()).level.value)

            # 핸들러가 없으면 추가
            if not logger.handlers:
                cls._setup_logger_handlers(logger, name, config or LogConfig())

            # 캐시에 저장
            cls._loggers[name] = logger
            return logger

    @classmethod
    def _setup_logger_handlers(
        cls, logger: logging.Logger, name: str, config: LogConfig
    ):
        """로거에 핸들러 설정"""
        # 콘솔 핸들러 추가
        if config.console_output:
            console_handler = cls._create_console_handler(config)
            logger.addHandler(console_handler)

        # 파일 핸들러 추가
        if config.file_output:
            log_file = cls._get_log_file_for_logger(name)

            # 핸들러 재사용을 위한 캐시 확인
            if log_file not in cls._handlers:
                cls._handlers[log_file] = cls._create_file_handler(log_file, config)

            logger.addHandler(cls._handlers[log_file])

    @classmethod
    def _initialize_logging_system(cls):
        """로깅 시스템 전역 초기화"""
        if cls._initialized:
            return

        # 로그 디렉토리 생성
        cls._ensure_log_directories()

        # 루트 로거 기본 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # 전역 예외 핸들러 설정
        sys.excepthook = cls._global_exception_handler

        cls._initialized = True

        # 초기화 완료 로그
        init_logger = cls.get_logger("unified_logging")
        init_logger.info("통합 로깅 시스템 초기화 완료")
        init_logger.info(
            f"허용된 로그 파일: {', '.join(cls.ALLOWED_LOG_FILES.values())}"
        )

    @classmethod
    def _global_exception_handler(cls, exc_type, exc_value, exc_traceback):
        """전역 예외 핸들러"""
        # KeyboardInterrupt는 정상 처리
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # 에러 로거로 예외 기록
        error_logger = cls.get_logger("error")
        error_logger.critical(
            "처리되지 않은 예외 발생", exc_info=(exc_type, exc_value, exc_traceback)
        )

    @classmethod
    def log_exception(cls, logger_name: str, exception: Exception, context: str = ""):
        """
        예외 로깅 유틸리티

        Args:
            logger_name: 로거 이름
            exception: 예외 객체
            context: 컨텍스트 정보
        """
        logger = cls.get_logger(logger_name)

        if context:
            logger.error(f"{context} - 예외 발생: {repr(exception)}")
        else:
            logger.error(f"예외 발생: {repr(exception)}")

        logger.error("상세 트레이스백:", exc_info=True)

    @classmethod
    def get_logger_stats(cls) -> Dict[str, Any]:
        """로거 통계 반환"""
        return {
            "total_loggers": len(cls._loggers),
            "total_handlers": len(cls._handlers),
            "logger_names": list(cls._loggers.keys()),
            "log_files": list(cls._handlers.keys()),
            "initialized": cls._initialized,
        }


# 편의 함수들
def get_logger(name: str, config: Optional[LogConfig] = None) -> logging.Logger:
    """통합 로거 반환 (편의 함수)"""
    return UnifiedLogger.get_logger(name, config)


def log_exception(logger_name: str, exception: Exception, context: str = ""):
    """예외 로깅 편의 함수"""
    UnifiedLogger.log_exception(logger_name, exception, context)


def configure_performance_logging() -> logging.Logger:
    """성능 로깅 전용 설정"""
    config = LogConfig(
        level=LogLevel.DEBUG,
        console_output=False,  # 성능 로그는 파일만
        file_output=True,
        format_string="%(asctime)s - %(levelname)s - [PERF] %(name)s - %(message)s",
    )
    return UnifiedLogger.get_logger("performance", config)


def configure_error_logging() -> logging.Logger:
    """에러 로깅 전용 설정"""
    config = LogConfig(
        level=LogLevel.ERROR,
        console_output=True,
        file_output=True,
        format_string="%(asctime)s - %(levelname)s - [ERROR] %(name)s - %(message)s",
    )
    return UnifiedLogger.get_logger("error", config)


# 초기화 함수 (기존 코드와의 호환성)
def init_logging_system():
    """로깅 시스템 초기화 (기존 API 호환성)"""
    UnifiedLogger._initialize_logging_system()


# 전역 정리 함수
def cleanup_logging():
    """로깅 시스템 정리"""
    UnifiedLogger.cleanup()
