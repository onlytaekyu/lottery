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
    """로깅 설정 (간소화)"""

    name: Optional[str] = None
    level: LogLevel = LogLevel.INFO
    console_output: bool = True
    file_output: bool = True
    format_string: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 3
    encoding: str = "utf-8"


class SimpleLoggerFactory:
    """간소화된 로거 팩토리"""

    _instance: Optional["SimpleLoggerFactory"] = None
    _lock = threading.RLock()
    _initialized = False
    _logger_cache: Dict[str, logging.Logger] = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # 로그 디렉토리 생성
            self.log_dir = Path(__file__).parent.parent.parent / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # 루트 로거 설정
            self._setup_root_logger()

            # 종료 시 정리 등록
            atexit.register(self.cleanup)

            self._initialized = True

    def _setup_root_logger(self):
        """루트 로거 설정"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def get_logger(
        self, name: str, config: Optional[LogConfig] = None
    ) -> logging.Logger:
        """로거 반환"""
        if name in self._logger_cache:
            return self._logger_cache[name]

        with self._lock:
            if name in self._logger_cache:
                return self._logger_cache[name]

            config = config or LogConfig()
            logger = logging.getLogger(name)
            logger.setLevel(config.level.value)

            # 핸들러 추가 (중복 방지)
            if not logger.handlers:
                self._setup_logger_handlers(logger, name, config)

            # 상위 로거로 전파 방지
            logger.propagate = False

            self._logger_cache[name] = logger
            return logger

    def _setup_logger_handlers(
        self, logger: logging.Logger, name: str, config: LogConfig
    ):
        """로거 핸들러 설정"""
        formatter = logging.Formatter(config.format_string, datefmt="%Y-%m-%d %H:%M:%S")

        # 콘솔 핸들러
        if config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(config.level.value)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 파일 핸들러
        if config.file_output:
            log_file = self.log_dir / f"{name.replace('.', '_')}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding=config.encoding,
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def cleanup(self):
        """정리"""
        try:
            for logger in self._logger_cache.values():
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

            self._logger_cache.clear()

        except Exception:
            pass  # 종료 시 예외 무시


# 전역 팩토리 인스턴스
_factory = None
_factory_lock = threading.RLock()


def _get_factory() -> SimpleLoggerFactory:
    """팩토리 인스턴스 반환"""
    global _factory

    if _factory is None:
        with _factory_lock:
            if _factory is None:
                _factory = SimpleLoggerFactory()

    return _factory


def get_logger(name: str, config: Optional[LogConfig] = None) -> logging.Logger:
    """로거 반환 (메인 함수)"""
    try:
        factory = _get_factory()
        return factory.get_logger(name, config)
    except Exception as e:
        # 로깅 시스템 실패 시 기본 로거 반환
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


def log_exception(logger_name: str, exception: Exception, context: str = ""):
    """예외 로깅"""
    logger = get_logger(logger_name)
    message = f"{context}: {str(exception)}" if context else str(exception)
    logger.error(message, exc_info=True)


def cleanup_logging():
    """로깅 시스템 정리"""
    global _factory

    if _factory:
        _factory.cleanup()
        _factory = None


# 편의 함수들
def init_logging_system():
    """로깅 시스템 초기화"""
    _get_factory()


def get_optimization_report() -> str:
    """최적화 리포트 반환"""
    return "로깅 시스템 간소화 완료 - 핵심 기능만 유지"
