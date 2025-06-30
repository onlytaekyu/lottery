"""
통합 로깅 시스템 - 완전 최적화 버전

중복 초기화 문제를 완전히 해결한 싱글톤 패턴 기반 로깅 시스템
- 전역 단일 초기화
- Thread-Safe 보장
- 메모리 최적화된 로거 캐싱
- 중복 핸들러 방지
"""

import logging
import logging.handlers
import sys
import threading
import weakref
from pathlib import Path
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import atexit
import hashlib
import time


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


class DuplicateMessageFilter(logging.Filter):
    """중복 메시지 필터 - 최강화 버전"""

    def __init__(self, interval: float = 10.0):
        super().__init__()
        self.interval = interval
        self.last_logged = {}
        self.message_counts = {}
        self.blocked_patterns = set()
        self.global_block_list = set()  # 영구 차단 목록

    def filter(self, record: logging.LogRecord) -> bool:
        """메시지 필터링 - 최강화 로직"""
        message = record.getMessage()
        current_time = time.time()

        # 🚨 영구 차단 패턴 (무한 반복 방지)
        permanent_block_patterns = [
            "동기화된 차원",
            "향상된 벡터화 시스템 연결 완료",
            "새로운 벡터화 인스턴스 생성",
            "기존 청사진 시스템 비활성화",
            "벡터 청사진 정의 완료",
            "벡터 청사진 시스템 초기화 완료",
            "168차원 표준 특성 이름 생성 완료",
            "🚀 벡터-이름 동시 생성 시스템 시작",
            "✅ 필수 특성 22개 실제 계산 완료",
            "🚀 슬라이딩 윈도우 샘플 생성 시작",
            "벡터 차원 축소:",
            "벡터 차원 확장:",
            "0값 특성",
            "특성 품질 개선 완료",
        ]

        for pattern in permanent_block_patterns:
            if pattern in message:
                pattern_key = f"permanent_{hash(pattern)}"

                # 패턴별로 첫 1회만 허용
                if pattern_key not in self.global_block_list:
                    self.global_block_list.add(pattern_key)
                    return True  # 첫 번째만 허용
                else:
                    return False  # 이후 모든 메시지 차단

        # 🔄 주기적 허용 패턴 (1시간마다)
        periodic_patterns = ["초기화 완료", "생성 완료", "연결 완료"]

        for pattern in periodic_patterns:
            if pattern in message:
                key = f"periodic_{hash(pattern)}"
                if key in self.last_logged:
                    if current_time - self.last_logged[key] < 3600:  # 1시간
                        return False
                self.last_logged[key] = current_time
                return True

        # ⚠️ 경고 메시지 (24시간마다)
        warning_patterns = [
            "설정에서",
            "찾을 수 없습니다",
            "벡터 차원",
            "차원 불일치",
            "필수 특성이 누락",
        ]

        for pattern in warning_patterns:
            if pattern in message:
                key = f"warning_{hash(pattern)}"
                if key in self.last_logged:
                    if current_time - self.last_logged[key] < 86400:  # 24시간
                        return False
                self.last_logged[key] = current_time
                return True

        # 기타 메시지는 모두 허용
        return True


class OptimizedLoggerFactory:
    """
    최적화된 로거 팩토리 - 완전 싱글톤 패턴

    전역에서 단 한 번만 초기화되며, 모든 로거 생성과 관리를 담당합니다.
    """

    # 클래스 레벨 상태 - 진정한 싱글톤
    _instance: Optional["OptimizedLoggerFactory"] = None
    _instance_lock = threading.RLock()
    _initialized = False
    _initialization_lock = threading.RLock()

    # 로거 및 핸들러 캐시
    _logger_cache: Dict[str, logging.Logger] = {}
    _handler_cache: Dict[str, logging.Handler] = {}
    _handler_refs: Set[str] = set()

    # 허용된 로그 파일 경로 (단일 파일로 통합)
    ALLOWED_LOG_FILES = {
        "main": "logs/lottery.log",
        # "file_usage": "logs/file_usage.log",  # 주석: 메인 로그에 통합
        # "model": "logs/model.log",            # 주석: 메인 로그에 통합
        # "performance": "logs/performance.log", # 주석: 메인 로그에 통합
        "error": "logs/error.log",
    }

    def __new__(cls):
        """진정한 싱글톤 구현"""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        """초기화 - 단 한 번만 실행"""
        with self._initialization_lock:
            if self._initialized:
                return

            self._global_config = LogConfig()
            self._ensure_log_directories()
            self._setup_root_logger()
            self._register_cleanup()

            self._initialized = True

    @classmethod
    def get_instance(cls) -> "OptimizedLoggerFactory":
        """팩토리 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_log_directories(self):
        """로그 디렉토리 생성 - 한 번만"""
        for log_file in self.ALLOWED_LOG_FILES.values():
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def _setup_root_logger(self):
        """루트 로거 설정 - 한 번만"""
        root_logger = logging.getLogger()

        # 기존 핸들러 제거 (중복 방지)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        root_logger.setLevel(logging.INFO)

        # 전역 예외 핸들러 설정
        if not hasattr(sys, "_original_excepthook"):
            sys._original_excepthook = sys.excepthook
            sys.excepthook = self._global_exception_handler

    def _register_cleanup(self):
        """정리 함수 등록"""
        atexit.register(self.cleanup)

    def get_logger(
        self, name: str, config: Optional[LogConfig] = None
    ) -> logging.Logger:
        """
        최적화된 로거 반환

        Args:
            name: 로거 이름
            config: 로깅 설정 (선택사항)

        Returns:
            캐시된 로거 인스턴스
        """
        # 캐시에서 먼저 확인
        if name in self._logger_cache:
            return self._logger_cache[name]

        with self._instance_lock:
            # Double-checked locking
            if name in self._logger_cache:
                return self._logger_cache[name]

            # 새 로거 생성
            logger = logging.getLogger(name)
            effective_config = config or self._global_config

            # 로거 레벨 설정
            logger.setLevel(effective_config.level.value)

            # 핸들러 설정 (중복 방지)
            if not logger.handlers:
                self._setup_logger_handlers(logger, name, effective_config)

            # 상위 로거로의 전파 방지 (중복 로그 방지)
            logger.propagate = False

            # 캐시에 저장
            self._logger_cache[name] = logger

            return logger

    def _setup_logger_handlers(
        self, logger: logging.Logger, name: str, config: LogConfig
    ):
        """로거 핸들러 설정 - 중복 방지"""
        handlers_added = []

        try:
            # 콘솔 핸들러
            if config.console_output:
                console_handler = self._get_or_create_console_handler(config)
                logger.addHandler(console_handler)
                handlers_added.append(console_handler)

            # 파일 핸들러
            if config.file_output:
                log_file = self._get_log_file_for_logger(name)
                file_handler = self._get_or_create_file_handler(log_file, config)
                logger.addHandler(file_handler)
                handlers_added.append(file_handler)

        except Exception as e:
            # 핸들러 설정 실패 시 정리
            for handler in handlers_added:
                logger.removeHandler(handler)
            raise RuntimeError(f"로거 핸들러 설정 실패: {e}")

    def _get_or_create_console_handler(
        self, config: LogConfig
    ) -> logging.StreamHandler:
        """콘솔 핸들러 캐시된 생성 (중복 메시지 필터링 포함)"""
        handler_key = "console"

        if handler_key in self._handler_cache:
            return self._handler_cache[handler_key]

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(config.console_level.value)
        handler.setFormatter(self._create_formatter(config))

        # 중복 메시지 필터 추가
        duplicate_filter = DuplicateMessageFilter(interval=10)
        handler.addFilter(duplicate_filter)

        self._handler_cache[handler_key] = handler
        self._handler_refs.add(handler_key)

        return handler

    def _get_or_create_file_handler(
        self, log_file: str, config: LogConfig
    ) -> logging.Handler:
        """파일 핸들러 캐시된 생성 (중복 메시지 필터링 포함)"""
        handler_key = f"file_{log_file}"

        if handler_key in self._handler_cache:
            return self._handler_cache[handler_key]

        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding=config.encoding,
        )
        handler.setLevel(config.file_level.value)
        handler.setFormatter(self._create_formatter(config))

        # 파일 로그에는 중복 필터링을 덜 엄격하게 적용
        duplicate_filter = DuplicateMessageFilter(interval=5)
        handler.addFilter(duplicate_filter)

        self._handler_cache[handler_key] = handler
        self._handler_refs.add(handler_key)

        return handler

    def _create_formatter(self, config: LogConfig) -> logging.Formatter:
        """포맷터 생성"""
        return logging.Formatter(fmt=config.format_string, datefmt=config.date_format)

    def _get_log_file_for_logger(self, logger_name: str) -> str:
        """로거 이름에 따른 적절한 로그 파일 반환 (단일 파일로 통합)"""
        name_lower = logger_name.lower()

        # 에러 로그만 별도 파일에 기록
        if any(keyword in name_lower for keyword in ["error", "exception", "critical"]):
            return self.ALLOWED_LOG_FILES["error"]
        else:
            # 모든 일반 로그는 메인 파일에 통합
            return self.ALLOWED_LOG_FILES["main"]

        # 주석: 기존 분류 로직 (사용 안함)
        # if any(keyword in name_lower for keyword in ["file", "data", "io", "cache", "disk"]):
        #     return self.ALLOWED_LOG_FILES["file_usage"]
        # elif any(keyword in name_lower for keyword in ["model", "train", "inference"]):
        #     return self.ALLOWED_LOG_FILES["model"]
        # elif any(keyword in name_lower for keyword in ["performance", "profiler", "memory"]):
        #     return self.ALLOWED_LOG_FILES["performance"]

    def _global_exception_handler(self, exc_type, exc_value, exc_traceback):
        """전역 예외 핸들러"""
        # KeyboardInterrupt는 정상 처리
        if issubclass(exc_type, KeyboardInterrupt):
            if hasattr(sys, "_original_excepthook"):
                sys._original_excepthook(exc_type, exc_value, exc_traceback)
            return

        # 에러 로거로 예외 기록
        try:
            error_logger = self.get_logger("error")
            error_logger.critical(
                "처리되지 않은 예외 발생", exc_info=(exc_type, exc_value, exc_traceback)
            )
        except:
            # 로거 자체에 문제가 있으면 기본 처리
            if hasattr(sys, "_original_excepthook"):
                sys._original_excepthook(exc_type, exc_value, exc_traceback)

    def log_exception(self, logger_name: str, exception: Exception, context: str = ""):
        """예외 로깅 유틸리티"""
        logger = self.get_logger(logger_name)

        if context:
            logger.error(f"{context} - 예외 발생: {repr(exception)}")
        else:
            logger.error(f"예외 발생: {repr(exception)}")

        logger.error("상세 트레이스백:", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """로깅 시스템 통계"""
        return {
            "initialized": self._initialized,
            "total_loggers": len(self._logger_cache),
            "total_handlers": len(self._handler_cache),
            "logger_names": list(self._logger_cache.keys()),
            "handler_keys": list(self._handler_refs),
            "log_files": list(self.ALLOWED_LOG_FILES.values()),
        }

    def cleanup(self):
        """시스템 정리"""
        with self._instance_lock:
            # 모든 핸들러 정리
            for handler in self._handler_cache.values():
                try:
                    handler.close()
                except:
                    pass

            # 캐시 정리
            self._logger_cache.clear()
            self._handler_cache.clear()
            self._handler_refs.clear()


# 전역 팩토리 인스턴스
_GLOBAL_FACTORY: Optional[OptimizedLoggerFactory] = None
_FACTORY_LOCK = threading.RLock()


def _get_factory() -> OptimizedLoggerFactory:
    """전역 팩토리 인스턴스 반환"""
    global _GLOBAL_FACTORY

    if _GLOBAL_FACTORY is None:
        with _FACTORY_LOCK:
            if _GLOBAL_FACTORY is None:
                _GLOBAL_FACTORY = OptimizedLoggerFactory()

    return _GLOBAL_FACTORY


# 공개 API 함수들
def get_logger(name: str, config: Optional[LogConfig] = None) -> logging.Logger:
    """
    최적화된 로거 반환 (공개 API)

    Args:
        name: 로거 이름
        config: 로깅 설정 (선택사항)

    Returns:
        캐시된 로거 인스턴스
    """
    factory = _get_factory()
    return factory.get_logger(name, config)


def log_exception(logger_name: str, exception: Exception, context: str = ""):
    """예외 로깅 편의 함수"""
    factory = _get_factory()
    factory.log_exception(logger_name, exception, context)


def log_exception_with_trace(logger_name: str, exception: Exception, context: str = ""):
    """트레이스백 포함 예외 로깅 (기존 API 호환성)"""
    log_exception(logger_name, exception, context)


def configure_performance_logging() -> logging.Logger:
    """성능 로깅 전용 설정"""
    config = LogConfig(
        level=LogLevel.DEBUG,
        console_output=False,  # 성능 로그는 파일만
        file_output=True,
        format_string="%(asctime)s - %(levelname)s - [PERF] %(name)s - %(message)s",
    )
    return get_logger("performance", config)


def configure_error_logging() -> logging.Logger:
    """에러 로깅 전용 설정"""
    config = LogConfig(
        level=LogLevel.ERROR,
        console_output=True,
        file_output=True,
        format_string="%(asctime)s - %(levelname)s - [ERROR] %(name)s - %(message)s",
    )
    return get_logger("error", config)


def init_logging_system():
    """로깅 시스템 초기화 (기존 API 호환성)"""
    # 팩토리 초기화만으로 충분
    _get_factory()


def get_logging_stats() -> Dict[str, Any]:
    """로깅 시스템 통계 반환"""
    factory = _get_factory()
    return factory.get_stats()


def cleanup_logging():
    """로깅 시스템 정리"""
    global _GLOBAL_FACTORY

    if _GLOBAL_FACTORY is not None:
        _GLOBAL_FACTORY.cleanup()
        _GLOBAL_FACTORY = None


# 하위 호환성을 위한 클래스 (기존 코드 지원)
class UnifiedLogger:
    """기존 UnifiedLogger API 호환성 클래스"""

    @classmethod
    def get_logger(
        cls, name: str, config: Optional[LogConfig] = None
    ) -> logging.Logger:
        """기존 API 호환성"""
        return get_logger(name, config)

    @classmethod
    def log_exception(cls, logger_name: str, exception: Exception, context: str = ""):
        """기존 API 호환성"""
        log_exception(logger_name, exception, context)

    @classmethod
    def get_logger_stats(cls) -> Dict[str, Any]:
        """기존 API 호환성"""
        return get_logging_stats()

    @classmethod
    def cleanup(cls):
        """기존 API 호환성"""
        cleanup_logging()


# 시스템 시작 시 자동 초기화
init_logging_system()
