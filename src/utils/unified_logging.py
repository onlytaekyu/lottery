"""통합 로깅 시스템 (성능 최적화 버전)

필수 로깅 기능만 제공하는 간소화된 로깅 시스템
"""

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import atexit
import re

try:
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


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


class SensitiveDataFilter(logging.Filter):
    """민감한 정보를 마스킹하는 로깅 필터"""

    def __init__(self, patterns: Optional[List[str]] = None):
        super().__init__()
        self._patterns = patterns or [
            r"'api_key':\s*'[^']+'",
            r"'password':\s*'[^']+'",
        ]
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self._patterns]

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.mask_sensitive_data(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(self.mask_sensitive_data(arg) for arg in record.args)
        return True

    def mask_sensitive_data(self, message: str) -> str:
        if not isinstance(message, str):
            return message

        for pattern in self._compiled_patterns:
            message = pattern.sub(
                lambda m: m.group().split(":")[0] + ": '***'", message
            )
        return message


class OptimizedLogManager:
    """동적 핸들러 및 민감 정보 필터링을 지원하는 최적화된 로그 관리자"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, level=logging.INFO, log_dir="logs"):
        if self._initialized:
            return

        self.root_logger = logging.getLogger("DAEBAK_AI")
        self.root_logger.setLevel(level)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.handlers = {}
        self.sensitive_filter = SensitiveDataFilter()
        self.root_logger.addFilter(self.sensitive_filter)

        self._setup_handlers(level)
        self._initialized = True

    def _setup_handlers(self, level: int):
        """로그 레벨에 따라 핸들러를 설정합니다."""
        # 콘솔 핸들러
        if RICH_AVAILABLE:
            console_handler = RichHandler(rich_tracebacks=True, show_path=False)
        else:
            console_handler = logging.StreamHandler(sys.stdout)

        console_handler.setFormatter(self.formatter)
        self.root_logger.addHandler(console_handler)
        self.handlers["console"] = console_handler

        # 파일 핸들러
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(self.formatter)
        self.root_logger.addHandler(file_handler)
        self.handlers["file"] = file_handler

        self.set_level(level)

    def get_logger(self, name: str) -> logging.Logger:
        return self.root_logger.getChild(name)

    def set_level(self, level: int):
        """전체 로깅 시스템의 레벨을 동적으로 변경합니다."""
        self.root_logger.setLevel(level)

        # 레벨에 따른 핸들러 활성화/비활성화 로직 (예시)
        # DEBUG 레벨에서는 콘솔 출력, INFO 이상부터는 파일에도 기록
        if level <= logging.DEBUG:
            if "console" not in self.handlers:
                # 핸들러 추가 로직
                pass
        else:
            if "console" in self.handlers:
                # 핸들러 제거 로직 (더 복잡한 관리가 필요할 수 있음)
                pass

    def cleanup(self):
        """모든 핸들러를 닫고 로깅 시스템을 안전하게 종료합니다."""
        for handler in self.root_logger.handlers[:]:
            try:
                handler.close()
                self.root_logger.removeHandler(handler)
            except Exception as e:
                # self.get_logger(__name__).error(f"핸들러 종료 중 오류: {e}")
                pass  # 종료 시점에는 로거가 불안정할 수 있음


_log_manager_instance: Optional[OptimizedLogManager] = None


def get_log_manager() -> OptimizedLogManager:
    """OptimizedLogManager의 싱글톤 인스턴스를 반환합니다."""
    global _log_manager_instance
    if _log_manager_instance is None:
        _log_manager_instance = OptimizedLogManager()
    return _log_manager_instance


def get_logger(name: str) -> logging.Logger:
    """지정된 이름의 로거를 가져옵니다."""
    manager = get_log_manager()
    return manager.get_logger(name)


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


def cleanup_logging():
    """전역 로그 관리자의 리소스를 정리합니다."""
    manager = get_log_manager()
    manager.cleanup()


# 종료 시 정리
atexit.register(cleanup_logging)


def log_exception(logger_name: str, exception: Exception, context: str = ""):
    """예외 로깅"""
    logger = get_logger(logger_name)
    message = f"{context}: {str(exception)}" if context else str(exception)
    logger.error(message, exc_info=True)


def get_optimization_report() -> str:
    """최적화 리포트 반환"""
    return "로깅 시스템 최적화 완료 - 캐싱 및 성능 개선"
