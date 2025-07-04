"""자동 오류 복구 메커니즘 모듈

이 모듈은 예외 상황에서 자동으로 복구를 시도하는 메커니즘을 제공합니다.
"""

import time
import threading
import asyncio
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
import psutil
import gc

from .unified_logging import get_logger

logger = get_logger(__name__)


@dataclass
class RecoveryConfig:
    """복구 설정 클래스"""

    max_retries: int = 3
    retry_delay: float = 1.0
    max_memory_percent: float = 80.0
    max_cpu_percent: float = 90.0
    cleanup_threshold: int = 1000
    stats: Dict[str, Any] = field(default_factory=dict)


class ErrorRecovery:
    """오류 복구 관리자 클래스"""

    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()
        self._process = psutil.Process()
        self._lock = threading.Lock()
        self._error_count = 0
        self._last_cleanup = time.time()
        self._recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0,
            "cleanup_count": 0,
        }

    def _check_resources(self) -> bool:
        """시스템 리소스 상태 확인"""
        try:
            memory_percent = self._process.memory_percent()
            cpu_percent = self._process.cpu_percent()

            if memory_percent > self.config.max_memory_percent:
                logger.warning(f"메모리 사용량 임계값 초과: {memory_percent:.1f}%")
                return False
            if cpu_percent > self.config.max_cpu_percent:
                logger.warning(f"CPU 사용량 임계값 초과: {cpu_percent:.1f}%")
                return False

            return True
        except Exception as e:
            logger.error(f"리소스 확인 중 오류 발생: {str(e)}")
            return False

    def _cleanup(self):
        """리소스 정리"""
        try:
            with self._lock:
                if time.time() - self._last_cleanup < self.config.cleanup_threshold:
                    return

                gc.collect()
                self._last_cleanup = time.time()
                self._recovery_stats["cleanup_count"] += 1
                logger.info("리소스 정리 완료")
        except Exception as e:
            logger.error(f"리소스 정리 중 오류 발생: {str(e)}")

    def recover(
        self,
        func: Callable,
        *args,
        retry_exceptions: Optional[List[Type[Exception]]] = None,
        **kwargs,
    ) -> Any:
        """
        함수 실행 및 오류 복구

        Args:
            func: 실행할 함수
            retry_exceptions: 재시도할 예외 타입 리스트
            *args: 함수에 전달할 위치 인수
            **kwargs: 함수에 전달할 키워드 인수

        Returns:
            함수의 실행 결과

        Raises:
            Exception: 모든 재시도가 실패한 경우 마지막 예외가 발생
        """
        retry_exceptions = retry_exceptions or [Exception]
        retry_count = 0

        while retry_count < self.config.max_retries:
            try:
                if not self._check_resources():
                    self._cleanup()
                    if not self._check_resources():
                        raise RuntimeError("시스템 리소스 부족")

                result = func(*args, **kwargs)
                self._error_count = 0
                return result

            except tuple(retry_exceptions) as e:
                retry_count += 1
                self._error_count += 1
                self._recovery_stats["total_errors"] += 1
                last_exception = e

                if retry_count < self.config.max_retries:
                    logger.warning(
                        f"오류 발생, 재시도 {retry_count}/{self.config.max_retries}: {str(e)}"
                    )
                    time.sleep(self.config.retry_delay)
                    self._cleanup()
                else:
                    logger.error(f"최대 재시도 횟수 초과: {str(e)}")
                    self._recovery_stats["failed_recoveries"] += 1
                    raise last_exception

            except Exception as e:
                logger.error(f"예상치 못한 오류 발생: {str(e)}")
                self._recovery_stats["failed_recoveries"] += 1
                raise

        # 이 부분에는 도달하지 않지만, 정적 분석기를 위한 안전장치
        self._recovery_stats["failed_recoveries"] += 1
        raise RuntimeError("최대 재시도 횟수 초과")

    async def recover_async(
        self,
        func: Callable,
        *args,
        retry_exceptions: Optional[List[Type[Exception]]] = None,
        **kwargs,
    ) -> Any:
        """비동기 함수 실행 및 오류 복구"""
        retry_exceptions = retry_exceptions or [Exception]
        retry_count = 0

        while retry_count < self.config.max_retries:
            try:
                if not self._check_resources():
                    self._cleanup()
                    if not self._check_resources():
                        raise RuntimeError("시스템 리소스 부족")

                result = await func(*args, **kwargs)
                self._error_count = 0
                return result

            except tuple(retry_exceptions) as e:
                retry_count += 1
                self._error_count += 1
                self._recovery_stats["total_errors"] += 1

                if retry_count < self.config.max_retries:
                    logger.warning(
                        f"오류 발생, 재시도 {retry_count}/{self.config.max_retries}: {str(e)}"
                    )
                    await asyncio.sleep(self.config.retry_delay)
                    self._cleanup()
                else:
                    logger.error(f"최대 재시도 횟수 초과: {str(e)}")
                    self._recovery_stats["failed_recoveries"] += 1
                    raise

            except Exception as e:
                logger.error(f"예상치 못한 오류 발생: {str(e)}")
                self._recovery_stats["failed_recoveries"] += 1
                raise

        self._recovery_stats["recovered_errors"] += 1
        return None

    def get_stats(self) -> Dict[str, Any]:
        """복구 통계 반환"""
        return self._recovery_stats.copy()

    def reset_stats(self):
        """복구 통계 초기화"""
        self._recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0,
            "cleanup_count": 0,
        }
