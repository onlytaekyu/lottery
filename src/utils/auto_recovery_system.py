"""
자동 복구 시스템 강화
GPU OOM, CPU 과부하, 메모리 누수 등 자동 복구
"""

import torch
import psutil
import gc
import threading
import time
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import functools

from .unified_logging import get_logger
from .unified_config import get_config
from .factory import get_singleton_instance
# from .unified_performance_engine import get_compute_executor, ComputeStrategy  # 순환 참조 방지
from .unified_memory_manager import get_unified_memory_manager

logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    """서킷 브레이커 상태"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class RecoveryStrategy(Enum):
    """복구 전략 타입"""

    GPU_OOM = "gpu_oom"
    CPU_OVERLOAD = "cpu_overload"
    MEMORY_LEAK = "memory_leak"
    DISK_FULL = "disk_full"
    NETWORK_ERROR = "network_error"
    PROCESS_HANG = "process_hang"


class RecoveryAction(Enum):
    """복구 액션 타입"""

    CLEANUP_MEMORY = "cleanup_memory"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    FALLBACK_TO_CPU = "fallback_to_cpu"
    RESTART_PROCESS = "restart_process"
    SCALE_DOWN = "scale_down"
    RETRY_WITH_BACKOFF = "retry_with_backoff"


@dataclass
class RecoveryContext:
    """복구 컨텍스트"""

    error_type: str
    error_message: str
    strategy: RecoveryStrategy
    attempted_actions: List[RecoveryAction]
    success: bool = False
    recovery_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SystemMonitor:
    """시스템 모니터링"""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks: Dict[str, List[Callable]] = {}
        self.thresholds = {
            "gpu_memory_usage": 90.0,  # %
            "cpu_usage": 85.0,  # %
            "memory_usage": 80.0,  # %
            "disk_usage": 90.0,  # %
        }

    def start_monitoring(self, interval: float = 5.0):
        """모니터링 시작"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self.monitor_thread.start()
        logger.info("시스템 모니터링 시작")

    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("시스템 모니터링 중지")

    def register_callback(self, event_type: str, callback: Callable):
        """콜백 등록"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    def _monitor_loop(self, interval: float):
        """모니터링 루프"""
        while self.monitoring:
            try:
                # GPU 메모리 확인
                if torch.cuda.is_available():
                    gpu_usage = self._get_gpu_memory_usage()
                    if gpu_usage > self.thresholds["gpu_memory_usage"]:
                        self._trigger_callbacks("gpu_memory_high", {"usage": gpu_usage})

                # CPU 사용률 확인
                cpu_usage = psutil.cpu_percent(interval=1)
                if cpu_usage > self.thresholds["cpu_usage"]:
                    self._trigger_callbacks("cpu_usage_high", {"usage": cpu_usage})

                # 메모리 사용률 확인
                memory = psutil.virtual_memory()
                if memory.percent > self.thresholds["memory_usage"]:
                    self._trigger_callbacks(
                        "memory_usage_high", {"usage": memory.percent}
                    )

                # 디스크 사용률 확인
                disk = psutil.disk_usage("/")
                disk_usage = (disk.used / disk.total) * 100
                if disk_usage > self.thresholds["disk_usage"]:
                    self._trigger_callbacks("disk_usage_high", {"usage": disk_usage})

                time.sleep(interval)

            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
                time.sleep(interval)

    def _get_gpu_memory_usage(self) -> float:
        """GPU 메모리 사용률 반환"""
        if not torch.cuda.is_available():
            return 0.0

        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return (allocated / total) * 100

    def _trigger_callbacks(self, event_type: str, data: Dict[str, Any]):
        """콜백 실행"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"콜백 실행 오류: {e}")


class RecoveryActionExecutor:
    """복구 액션 실행기"""

    def __init__(self):
        self.memory_manager = get_unified_memory_manager()
        # self.compute_executor는 제거. 순환 참조의 원인.
        # 액션은 상태 변경에만 집중하고, 실제 실행은 호출자가 담당.

    def execute_action(self, action: RecoveryAction, context: RecoveryContext) -> bool:
        """복구 액션 실행"""
        try:
            if action == RecoveryAction.CLEANUP_MEMORY:
                return self._cleanup_memory(context)
            elif action == RecoveryAction.REDUCE_BATCH_SIZE:
                return self._reduce_batch_size(context)
            elif action == RecoveryAction.FALLBACK_TO_CPU:
                return self._fallback_to_cpu(context)
            elif action == RecoveryAction.RESTART_PROCESS:
                return self._restart_process(context)
            elif action == RecoveryAction.SCALE_DOWN:
                return self._scale_down(context)
            elif action == RecoveryAction.RETRY_WITH_BACKOFF:
                return self._retry_with_backoff(context)
            else:
                logger.warning(f"알 수 없는 복구 액션: {action}")
                return False

        except Exception as e:
            logger.error(f"복구 액션 실행 오류: {action}: {e}")
            return False

    def _cleanup_memory(self, context: RecoveryContext) -> bool:
        """메모리 정리"""
        logger.info("메모리 정리 시작")

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # CPU 메모리 정리
        gc.collect()

        # 통합 메모리 관리자 정리
        self.memory_manager.cleanup_all()

        logger.info("메모리 정리 완료")
        return True

    def _reduce_batch_size(self, context: RecoveryContext) -> bool:
        """배치 크기 감소"""
        logger.info("배치 크기 감소 시도")

        # 메타데이터에서 현재 배치 크기 확인
        current_batch_size = context.metadata.get("batch_size", 32)
        new_batch_size = max(1, current_batch_size // 2)

        context.metadata["batch_size"] = new_batch_size
        context.metadata["batch_size_reduced"] = True

        logger.info(f"배치 크기 감소: {current_batch_size} -> {new_batch_size}")
        return True

    def _fallback_to_cpu(self, context: RecoveryContext) -> bool:
        """CPU로 폴백 (상태 플래그 설정)"""
        logger.info("CPU 폴백 요청. 컨텍스트에 'force_cpu=True' 설정.")

        context.metadata["force_cpu"] = True
        context.metadata["fallback_to_cpu"] = True

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # compute_executor 직접 호출 로직 제거
        logger.info("CPU 폴백 상태 설정 완료")
        return True

    def _restart_process(self, context: RecoveryContext) -> bool:
        """프로세스 재시작"""
        logger.warning("프로세스 재시작 필요 - 수동 개입 필요")

        # 실제 재시작은 상위 시스템에서 처리
        context.metadata["restart_required"] = True

        return False  # 수동 개입 필요

    def _scale_down(self, context: RecoveryContext) -> bool:
        """스케일 다운"""
        logger.info("시스템 스케일 다운")

        # 동시 실행 작업 수 감소
        context.metadata["max_workers"] = max(
            1, context.metadata.get("max_workers", 4) // 2
        )
        context.metadata["scaled_down"] = True

        logger.info(f"최대 워커 수 감소: {context.metadata['max_workers']}")
        return True

    def _retry_with_backoff(self, context: RecoveryContext) -> bool:
        """백오프와 함께 재시도"""
        retry_count = context.metadata.get("retry_count", 0)
        max_retries = context.metadata.get("max_retries", 3)

        if retry_count >= max_retries:
            logger.error("최대 재시도 횟수 초과")
            return False

        # 지수적 백오프
        backoff_time = 2**retry_count
        logger.info(f"재시도 대기: {backoff_time}초")

        time.sleep(backoff_time)

        context.metadata["retry_count"] = retry_count + 1
        return True


class AutoRecoverySystem:
    """통합 자동 복구 시스템 (Circuit Breaker 포함)"""

    def __init__(self):
        """초기화"""
        # Circuit Breaker 설정
        self.circuit_breaker_config = get_config().get("circuit_breaker", {})
        self.failure_threshold = self.circuit_breaker_config.get("failure_threshold", 5)
        self.recovery_timeout = self.circuit_breaker_config.get("recovery_timeout", 60)
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._lock = threading.RLock()

        # 기존 시스템 모니터 및 액션 실행기
        self.monitor = get_singleton_instance(SystemMonitor)
        self.action_executor = get_singleton_instance(RecoveryActionExecutor)
        self.is_running = False

        # 복구 전략 매핑
        self.recovery_strategies = {
            RecoveryStrategy.GPU_OOM: [
                RecoveryAction.CLEANUP_MEMORY,
                RecoveryAction.REDUCE_BATCH_SIZE,
                RecoveryAction.FALLBACK_TO_CPU,
            ],
            RecoveryStrategy.CPU_OVERLOAD: [
                RecoveryAction.SCALE_DOWN,
                RecoveryAction.CLEANUP_MEMORY,
                RecoveryAction.RETRY_WITH_BACKOFF,
            ],
            RecoveryStrategy.MEMORY_LEAK: [
                RecoveryAction.CLEANUP_MEMORY,
                RecoveryAction.RESTART_PROCESS,
            ],
            RecoveryStrategy.DISK_FULL: [
                RecoveryAction.CLEANUP_MEMORY,
                RecoveryAction.SCALE_DOWN,
            ],
            RecoveryStrategy.NETWORK_ERROR: [RecoveryAction.RETRY_WITH_BACKOFF],
            RecoveryStrategy.PROCESS_HANG: [RecoveryAction.RESTART_PROCESS],
        }

        # 복구 히스토리
        self.recovery_history: List[RecoveryContext] = []

        # 콜백 등록
        self._register_callbacks()

        # 설정
        self.enable_auto_recovery = self.config.get("enable_auto_recovery", True)
        self.monitoring_interval = self.config.get("monitoring_interval", 5.0)

        logger.info(f"✅ 자동 복구 시스템 초기화 (활성화: {self.enable_auto_recovery})")

    @property
    def state(self):
        """Circuit Breaker 상태 확인 및 자동 전환"""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN and (
                time.time() - self._last_failure_time > self.recovery_timeout
            ):
                self._state = CircuitBreakerState.HALF_OPEN
                logger.info("서킷 브레이커 상태 변경: OPEN -> HALF_OPEN")
            return self._state

    def record_failure(self):
        """실패 기록 및 서킷 OPEN"""
        with self._lock:
            self._failure_count += 1
            logger.debug(f"실패 기록: {self._failure_count}/{self.failure_threshold}")
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                self._last_failure_time = time.time()
                logger.error(f"서킷 브레이커 OPEN! 임계값({self.failure_threshold}) 도달.")

    def record_success(self):
        """성공 기록 및 서킷 CLOSE"""
        with self._lock:
            if self._state != CircuitBreakerState.CLOSED:
                 logger.info("서킷 브레이커 상태 변경: -> CLOSED")
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED

    def start(self):
        """시스템 모니터링 시작"""
        if self.is_running:
            return
        self.monitor.start_monitoring()
        self._register_callbacks()
        self.is_running = True
        logger.info("✅ 통합 자동 복구 시스템 시작")

    def stop(self):
        """시스템 모니터링 중지"""
        if not self.is_running:
            return
        self.monitor.stop_monitoring()
        self.is_running = False
        logger.info("⏹️ 통합 자동 복구 시스템 중지")

    def handle_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> RecoveryContext:
        """에러 처리 및 복구"""
        if context is None:
            context = {}

        # 에러 타입 분석
        strategy = self._analyze_error(error)

        # 복구 컨텍스트 생성
        recovery_context = RecoveryContext(
            error_type=type(error).__name__,
            error_message=str(error),
            strategy=strategy,
            attempted_actions=[],
            metadata=context,
        )

        # 복구 실행
        start_time = time.time()
        recovery_context.success = self._execute_recovery(recovery_context)
        recovery_context.recovery_time = time.time() - start_time

        # 히스토리 저장
        self.recovery_history.append(recovery_context)

        # 결과 로깅
        if recovery_context.success:
            logger.info(
                f"복구 성공: {recovery_context.error_type} ({recovery_context.recovery_time:.2f}초)"
            )
        else:
            logger.error(f"복구 실패: {recovery_context.error_type}")

        return recovery_context

    def _analyze_error(self, error: Exception) -> RecoveryStrategy:
        """에러 분석 및 전략 결정"""
        error_type = type(error).__name__
        error_message = str(error).lower()

        # GPU 메모리 부족
        if error_type == "RuntimeError" and (
            "out of memory" in error_message or "cuda" in error_message
        ):
            return RecoveryStrategy.GPU_OOM

        # CPU 과부하
        if (
            error_type in ["TimeoutError", "ProcessLookupError"]
            or "cpu" in error_message
        ):
            return RecoveryStrategy.CPU_OVERLOAD

        # 메모리 누수
        if error_type == "MemoryError" or "memory" in error_message:
            return RecoveryStrategy.MEMORY_LEAK

        # 디스크 부족
        if "disk" in error_message or "space" in error_message:
            return RecoveryStrategy.DISK_FULL

        # 네트워크 오류
        if (
            error_type in ["ConnectionError", "TimeoutError"]
            or "network" in error_message
        ):
            return RecoveryStrategy.NETWORK_ERROR

        # 프로세스 행업
        if error_type in ["ProcessLookupError", "BrokenPipeError"]:
            return RecoveryStrategy.PROCESS_HANG

        # 기본값: CPU 과부하로 처리
        return RecoveryStrategy.CPU_OVERLOAD

    def _execute_recovery(self, context: RecoveryContext) -> bool:
        """복구 실행"""
        strategy = context.strategy
        actions = self.recovery_strategies.get(strategy, [])

        logger.info(f"복구 전략 실행: {strategy.value}")

        for action in actions:
            logger.info(f"복구 액션 시도: {action.value}")

            success = self.action_executor.execute_action(action, context)
            context.attempted_actions.append(action)

            if success:
                logger.info(f"복구 액션 성공: {action.value}")
                return True
            else:
                logger.warning(f"복구 액션 실패: {action.value}")

        return False

    def _register_callbacks(self):
        """모니터링 콜백 등록"""
        self.monitor.register_callback("gpu_memory_high", self._handle_gpu_memory_high)
        self.monitor.register_callback("cpu_usage_high", self._handle_cpu_usage_high)
        self.monitor.register_callback(
            "memory_usage_high", self._handle_memory_usage_high
        )
        self.monitor.register_callback("disk_usage_high", self._handle_disk_usage_high)

    def _handle_gpu_memory_high(self, data: Dict[str, Any]):
        """GPU 메모리 사용률 높음 처리"""
        logger.warning(f"GPU 메모리 사용률 높음: {data['usage']:.1f}%")

        # 예방적 복구
        context = RecoveryContext(
            error_type="PreventiveRecovery",
            error_message="GPU memory usage high",
            strategy=RecoveryStrategy.GPU_OOM,
            attempted_actions=[],
            metadata=data,
        )

        self.action_executor.execute_action(RecoveryAction.CLEANUP_MEMORY, context)

    def _handle_cpu_usage_high(self, data: Dict[str, Any]):
        """CPU 사용률 높음 처리"""
        logger.warning(f"CPU 사용률 높음: {data['usage']:.1f}%")

        context = RecoveryContext(
            error_type="PreventiveRecovery",
            error_message="CPU usage high",
            strategy=RecoveryStrategy.CPU_OVERLOAD,
            attempted_actions=[],
            metadata=data,
        )

        self.action_executor.execute_action(RecoveryAction.SCALE_DOWN, context)

    def _handle_memory_usage_high(self, data: Dict[str, Any]):
        """메모리 사용률 높음 처리"""
        logger.warning(f"메모리 사용률 높음: {data['usage']:.1f}%")

        context = RecoveryContext(
            error_type="PreventiveRecovery",
            error_message="Memory usage high",
            strategy=RecoveryStrategy.MEMORY_LEAK,
            attempted_actions=[],
            metadata=data,
        )

        self.action_executor.execute_action(RecoveryAction.CLEANUP_MEMORY, context)

    def _handle_disk_usage_high(self, data: Dict[str, Any]):
        """디스크 사용률 높음 처리"""
        logger.warning(f"디스크 사용률 높음: {data['usage']:.1f}%")

        context = RecoveryContext(
            error_type="PreventiveRecovery",
            error_message="Disk usage high",
            strategy=RecoveryStrategy.DISK_FULL,
            attempted_actions=[],
            metadata=data,
        )

        self.action_executor.execute_action(RecoveryAction.CLEANUP_MEMORY, context)

    def get_recovery_stats(self) -> Dict[str, Any]:
        """복구 통계 반환"""
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for ctx in self.recovery_history if ctx.success)

        strategy_stats = {}
        for ctx in self.recovery_history:
            strategy = ctx.strategy.value
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "successful": 0}
            strategy_stats[strategy]["total"] += 1
            if ctx.success:
                strategy_stats[strategy]["successful"] += 1

        return {
            "total_recoveries": total_recoveries,
            "successful_recoveries": successful_recoveries,
            "success_rate": (
                (successful_recoveries / total_recoveries * 100)
                if total_recoveries > 0
                else 0
            ),
            "strategy_stats": strategy_stats,
            "average_recovery_time": (
                sum(ctx.recovery_time for ctx in self.recovery_history)
                / total_recoveries
                if total_recoveries > 0
                else 0
            ),
        }

    @contextmanager
    def auto_recovery_context(self, context: Dict[str, Any] = None):
        """자동 복구 컨텍스트 매니저"""
        if context is None:
            context = {}

        try:
            yield
        except Exception as e:
            recovery_context = self.handle_error(e, context)
            if not recovery_context.success:
                raise  # 복구 실패시 원래 예외 재발생


# 싱글톤 인스턴스
def get_auto_recovery_system() -> AutoRecoverySystem:
    """AutoRecoverySystem의 싱글톤 인스턴스를 반환합니다."""
    return get_singleton_instance(AutoRecoverySystem)


def auto_recoverable(
    strategy: Optional[RecoveryStrategy] = None, 
    max_retries: int = 3,
    delay: float = 1.0
):
    """
    통합 자동 복구 데코레이터.
    GPU OOM, 서킷 브레이커, 재시도 로직을 통합 관리합니다.
    """
    recovery_system = get_auto_recovery_system()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 서킷 브레이커 확인
            if recovery_system.state == CircuitBreakerState.OPEN:
                logger.error(f"Circuit Breaker가 열려있어 '{func.__name__}' 호출을 차단합니다.")
                # OPEN 상태에서는 빠른 실패(fail-fast)를 반환
                raise ConnectionAbortedError("Circuit Breaker is open.")

            # 재시도 로직
            retries = 0
            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    # HALF_OPEN 상태에서 성공 시 서킷을 닫음
                    if recovery_system.state == CircuitBreakerState.HALF_OPEN:
                        recovery_system.record_success()
                    return result

                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"GPU OOM 발생! (시도 {retries + 1}/{max_retries}) - 자동 복구 시도...")
                    context = {"error": e, "args": args, "kwargs": kwargs}
                    recovery_context = recovery_system.handle_error(e, context)
                    
                    # 복구 액션으로 상태가 변경되었을 수 있으므로 재시도
                    if recovery_context.success:
                        logger.info("GPU OOM 복구 성공. 작업을 재시도합니다.")
                        time.sleep(delay)
                    else:
                        logger.error("GPU OOM 복구 실패. 작업을 중단합니다.")
                        recovery_system.record_failure()
                        raise e # 복구 실패 시 예외 다시 발생

                except Exception as e:
                    logger.error(f"'{func.__name__}' 실행 중 예측하지 못한 오류: {e}", exc_info=True)
                    recovery_system.record_failure()
                    raise e # 일반 오류는 재시도 없이 전파

                retries += 1
            
            # 최종 실패
            logger.error(f"'{func.__name__}'가 최대 재시도 횟수({max_retries})를 초과했습니다.")
            recovery_system.record_failure()
            raise RuntimeError(f"Function {func.__name__} failed after {max_retries} retries.")

        return wrapper
    return decorator


def handle_error_with_recovery(
    error: Exception, context: Dict[str, Any] = None
) -> RecoveryContext:
    """에러 처리 및 복구 편의 함수"""
    recovery_system = get_auto_recovery_system()
    return recovery_system.handle_error(error, context)


def start_auto_recovery():
    """자동 복구 시스템 시작 편의 함수"""
    recovery_system = get_auto_recovery_system()
    recovery_system.start()


def stop_auto_recovery():
    """자동 복구 시스템 정지 편의 함수"""
    recovery_system = get_auto_recovery_system()
    recovery_system.stop()
