"""
고급 오류 처리 및 복구 시스템 (v4 - Circuit Breaker)

Circuit Breaker 패턴과 GPU 특화 오류 복구를 통해
시스템의 안정성과 복원력을 극대화하는 지능형 오류 처리기.
"""

import time
import functools
import torch
import gc
from typing import Callable, Any, Dict, Optional
from enum import Enum
import threading
import asyncio

from .unified_logging import get_logger
from .factory import get_singleton_instance

logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class AdvancedErrorHandler:
    """Circuit Breaker 및 GPU 오류 복구를 지원하는 고급 오류 처리기"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._lock = threading.Lock()

    @property
    def state(self):
        with self._lock:
            if self._state == CircuitBreakerState.OPEN and (
                time.time() - self._last_failure_time > self.recovery_timeout
            ):
                self._state = CircuitBreakerState.HALF_OPEN
            return self._state

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                self._last_failure_time = time.time()

    def record_success(self):
        with self._lock:
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED

    def auto_recoverable(self, max_retries: int = 3, delay: float = 1.0):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self.state == CircuitBreakerState.OPEN:
                    logger.error(
                        f"Circuit Breaker가 열려있습니다. '{func.__name__}' 호출을 차단합니다."
                    )
                    return None  # 또는 예외 발생

                retries = 0
                while retries < max_retries:
                    try:
                        result = func(*args, **kwargs)
                        if self.state == CircuitBreakerState.HALF_OPEN:
                            self.record_success()
                        return result

                    except torch.cuda.OutOfMemoryError as e:
                        logger.warning(
                            f"GPU 메모리 부족 오류 발생! (시도 {retries + 1}/{max_retries}) - 자동 복구 시도..."
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        time.sleep(delay * (retries + 1))  # 재시도 간격 증가

                    except Exception as e:
                        logger.error(
                            f"'{func.__name__}' 실행 중 오류 발생: {e}", exc_info=True
                        )
                        self.record_failure()
                        raise  # 일반 오류는 재시도하지 않고 전파

                    retries += 1

                logger.error(
                    f"'{func.__name__}'가 최대 재시도 횟수({max_retries})를 초과했습니다."
                )
                self.record_failure()
                return None

            return wrapper

        return decorator

    def async_auto_recoverable(self, max_retries: int = 3, delay: float = 1.0):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if self.state == CircuitBreakerState.OPEN:
                    logger.error(
                        f"Circuit Breaker가 열려있습니다. '{func.__name__}' 호출을 차단합니다."
                    )
                    return None

                retries = 0
                while retries < max_retries:
                    try:
                        result = await func(*args, **kwargs)
                        if self.state == CircuitBreakerState.HALF_OPEN:
                            self.record_success()
                        return result

                    except torch.cuda.OutOfMemoryError as e:
                        logger.warning(
                            f"GPU 메모리 부족 오류 발생! (시도 {retries + 1}/{max_retries}) - 자동 복구 시도..."
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        await asyncio.sleep(delay * (retries + 1))

                    except Exception as e:
                        logger.error(
                            f"'{func.__name__}' 실행 중 오류 발생: {e}", exc_info=True
                        )
                        self.record_failure()
                        raise

                    retries += 1

                logger.error(
                    f"'{func.__name__}'가 최대 재시도 횟수({max_retries})를 초과했습니다."
                )
                self.record_failure()
                return None

            return wrapper

        return decorator

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        데코레이터 대신 직접 호출할 수 있는 재시도 래퍼입니다.
        순환 참조를 피하기 위해 사용됩니다.
        """
        max_retries = kwargs.pop("max_retries", 3)
        delay = kwargs.pop("delay", 1.0)

        if self.state == CircuitBreakerState.OPEN:
            logger.error(
                f"Circuit Breaker가 열려있습니다. '{func.__name__}' 호출을 차단합니다."
            )
            return None

        retries = 0
        while retries < max_retries:
            try:
                result = func(*args, **kwargs)
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.record_success()
                return result

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(
                    f"GPU 메모리 부족 오류 발생! (시도 {retries + 1}/{max_retries}) - 자동 복구 시도..."
                )
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(delay * (retries + 1))

            except Exception as e:
                logger.error(f"'{func.__name__}' 실행 중 오류 발생: {e}", exc_info=True)
                self.record_failure()
                raise

            retries += 1

        logger.error(
            f"'{func.__name__}'가 최대 재시도 횟수({max_retries})를 초과했습니다."
        )
        self.record_failure()
        return None


def get_error_handler() -> AdvancedErrorHandler:
    """AdvancedErrorHandler의 싱글톤 인스턴스를 반환합니다."""
    return get_singleton_instance(AdvancedErrorHandler)


# 자동 복구 데코레이터 예제 (주석 처리)
# 실제 사용 시에는 auto_recovery_system에서 import 필요
# from .auto_recovery_system import auto_recoverable, RecoveryStrategy
#
# @auto_recoverable(strategy=RecoveryStrategy.GPU_OOM)
# def gpu_intensive_function():
#     # GPU 메모리 부족 시 자동 복구
#     pass
