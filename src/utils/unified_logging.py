"""통합 로깅 시스템 (성능 최적화 버전)

필수 로깅 기능만 제공하는 간소화된 로깅 시스템
"""

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import atexit


class LogLevel(Enum):
    """로그 레벨"""

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
    format_string: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 3
    encoding: str = "utf-8"


class OptimizedLoggerFactory:
    """최적화된 로거 팩토리"""

    _loggers: Dict[str, logging.Logger] = {}
    _formatter: Optional[logging.Formatter] = None
    _lock = threading.RLock()

    @classmethod
    def get_logger(
        cls, name: str, config: Optional[LogConfig] = None
    ) -> logging.Logger:
        """로거 반환 (캐싱 최적화)"""
        if name in cls._loggers:
            return cls._loggers[name]

        with cls._lock:
            if name in cls._loggers:
                return cls._loggers[name]

            config = config or LogConfig()
            logger = cls._create_logger(name, config)
            cls._loggers[name] = logger
            return logger

    @classmethod
    def _create_logger(cls, name: str, config: LogConfig) -> logging.Logger:
        """로거 생성"""
        logger = logging.getLogger(name)

        if logger.handlers:
            return logger

        logger.setLevel(config.level.value)
        logger.propagate = False

        # 포매터 생성 (재사용)
        if cls._formatter is None:
            cls._formatter = logging.Formatter(config.format_string)

        # 콘솔 핸들러
        if config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(config.level.value)
            console_handler.setFormatter(cls._formatter)
            logger.addHandler(console_handler)

        # 파일 핸들러
        if config.file_output:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)

            log_file = logs_dir / f"{name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding=config.encoding,
            )
            file_handler.setLevel(config.level.value)
            file_handler.setFormatter(cls._formatter)
            logger.addHandler(file_handler)

        return logger

    @classmethod
    def cleanup(cls):
        """리소스 정리"""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()


# 전역 함수 (성능 최적화)
def get_logger(name: str, config: Optional[LogConfig] = None) -> logging.Logger:
    """로거 반환"""
    return OptimizedLoggerFactory.get_logger(name, config)


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    console_output: bool = True,
    file_output: bool = True,
    log_dir: str = "logs",
) -> None:
    """로깅 시스템 초기화"""
    config = LogConfig(
        level=level, console_output=console_output, file_output=file_output
    )

    # 루트 로거 설정
    root_logger = get_logger("root", config)

    # 기본 로거들 미리 생성
    for logger_name in ["main", "training", "analysis", "prediction"]:
        get_logger(logger_name, config)


# 종료 시 정리
atexit.register(OptimizedLoggerFactory.cleanup)


def log_exception(logger_name: str, exception: Exception, context: str = ""):
    """예외 로깅"""
    logger = get_logger(logger_name)
    message = f"{context}: {str(exception)}" if context else str(exception)
    logger.error(message, exc_info=True)


def get_optimization_report() -> str:
    """최적화 리포트 반환"""
    return "로깅 시스템 최적화 완료 - 캐싱 및 성능 개선"
