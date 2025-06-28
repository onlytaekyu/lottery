"""
자동 로거 주입 시스템

모든 모듈에 자동으로 최적화된 로거를 주입하여 중복 초기화 문제를 완전히 해결합니다.
- 모듈 import 시 자동 로거 할당
- Thread-Safe 보장
- 메모리 최적화
"""

import sys
import threading
from typing import Dict, Set, Any
from types import ModuleType
import weakref

from .unified_logging import get_logger


class AutoLoggerInjector:
    """
    자동 로거 주입기

    모든 모듈에 자동으로 로거를 주입하여 중복 초기화를 방지합니다.
    """

    _instance = None
    _lock = threading.RLock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._injected_modules: Set[str] = set()
        self._logger_cache: Dict[str, Any] = {}
        self._module_refs: Dict[str, weakref.ref] = {}
        self._original_import = None

        self._setup_import_hook()
        self._initialized = True

    def _setup_import_hook(self):
        """import 훅 설정"""
        if self._original_import is not None:
            return

        # 안전한 import 훅 설정
        try:
            self._original_import = __builtins__.__import__
            __builtins__.__import__ = self._hooked_import
        except Exception:
            # 훅 설정 실패 시 조용히 무시
            pass

    def _hooked_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """import 훅 - 모듈 로드 시 자동 로거 주입"""
        if self._original_import is None:
            # 원본 import가 없으면 기본 import 사용
            return __import__(name, globals, locals, fromlist, level)

        module = self._original_import(name, globals, locals, fromlist, level)

        # 프로젝트 모듈만 처리
        if self._should_inject_logger(name, module):
            self._inject_logger(name, module)

        return module

    def _should_inject_logger(self, module_name: str, module: ModuleType) -> bool:
        """로거 주입 여부 판단"""
        # 이미 주입된 모듈은 스킵
        if module_name in self._injected_modules:
            return False

        # 프로젝트 모듈만 처리 (src.로 시작하는 모듈)
        if not module_name.startswith("src."):
            return False

        # 시스템 모듈 제외
        if module_name in [
            "src.utils.unified_logging",
            "src.utils.auto_logger_injector",
        ]:
            return False

        # 모듈이 실제로 존재하는지 확인
        if not hasattr(module, "__file__"):
            return False

        return True

    def _inject_logger(self, module_name: str, module: ModuleType):
        """모듈에 로거 주입"""
        with self._lock:
            try:
                # 이미 logger 속성이 있으면 교체
                if hasattr(module, "logger"):
                    # 기존 로거가 우리 시스템의 로거인지 확인
                    existing_logger = getattr(module, "logger")
                    if hasattr(existing_logger, "_optimized_logger_factory"):
                        return  # 이미 최적화된 로거

                # 새 로거 생성 및 주입
                logger = get_logger(module_name)

                # 최적화된 로거임을 표시
                logger._optimized_logger_factory = True

                # 모듈에 로거 할당
                setattr(module, "logger", logger)

                # 추적 정보 저장
                self._injected_modules.add(module_name)
                self._logger_cache[module_name] = logger
                self._module_refs[module_name] = weakref.ref(module)

            except Exception:
                # 주입 실패는 조용히 무시 (시스템 안정성 우선)
                pass

    def inject_logger_to_existing_modules(self):
        """이미 로드된 모든 모듈에 로거 주입"""
        with self._lock:
            for module_name, module in sys.modules.items():
                if self._should_inject_logger(module_name, module):
                    self._inject_logger(module_name, module)

    def get_injection_stats(self) -> Dict[str, Any]:
        """주입 통계 반환"""
        return {
            "injected_modules": len(self._injected_modules),
            "cached_loggers": len(self._logger_cache),
            "module_list": list(self._injected_modules),
            "active_refs": len(
                [ref for ref in self._module_refs.values() if ref() is not None]
            ),
        }

    def cleanup(self):
        """정리"""
        with self._lock:
            # import 훅 복원
            if self._original_import is not None:
                try:
                    __builtins__.__import__ = self._original_import
                    self._original_import = None
                except Exception:
                    pass

            # 캐시 정리
            self._injected_modules.clear()
            self._logger_cache.clear()
            self._module_refs.clear()


# 전역 인스턴스
_global_injector = None
_injector_lock = threading.RLock()


def get_auto_injector() -> AutoLoggerInjector:
    """전역 자동 주입기 반환"""
    global _global_injector

    if _global_injector is None:
        with _injector_lock:
            if _global_injector is None:
                _global_injector = AutoLoggerInjector()

    return _global_injector


def activate_auto_logger_injection():
    """자동 로거 주입 활성화"""
    try:
        injector = get_auto_injector()

        # 이미 로드된 모듈들에도 로거 주입
        injector.inject_logger_to_existing_modules()

        return injector
    except Exception:
        # 활성화 실패 시 조용히 무시
        return None


def get_injection_stats() -> Dict[str, Any]:
    """주입 통계 반환"""
    if _global_injector is not None:
        return _global_injector.get_injection_stats()
    return {"injected_modules": 0, "cached_loggers": 0, "module_list": []}


def cleanup_auto_injection():
    """자동 주입 시스템 정리"""
    global _global_injector

    if _global_injector is not None:
        _global_injector.cleanup()
        _global_injector = None


# 수동 로거 주입 함수 (필요한 경우)
def manual_inject_logger(module_name: str, module: ModuleType = None):
    """특정 모듈에 수동으로 로거 주입"""
    if module is None:
        module = sys.modules.get(module_name)

    if module is None:
        return False

    try:
        injector = get_auto_injector()
        injector._inject_logger(module_name, module)
        return True
    except Exception:
        return False


# 시스템 자동 활성화 (안전하게)
try:
    activate_auto_logger_injection()
except Exception:
    # 자동 활성화 실패 시 조용히 무시
    pass
