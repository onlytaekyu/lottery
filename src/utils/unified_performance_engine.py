"""
통합 성능 최적화 엔진 (Unified Performance Engine)

기존 performance_optimizer.py와 compute_strategy.py의 중복 기능을 통합하여
GPU > 멀티쓰레드 > CPU 우선순위 처리와 성능 모니터링을 단일 시스템으로 제공합니다.
"""

import os
import time
import torch
import psutil
import threading
import numpy as np
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from .unified_logging import get_logger
from .unified_config import get_config
from .memory_manager import get_memory_manager
from .unified_memory_manager import get_unified_memory_manager
from .auto_recovery_system import auto_recoverable, RecoveryStrategy
from .factory import get_singleton_instance

logger = get_logger(__name__)


class ComputeStrategy(Enum):
    """연산 전략 열거형"""

    GPU = "gpu"
    MULTITHREAD_CPU = "multithread_cpu"
    SINGLE_CPU = "single_cpu"


class TaskType(Enum):
    """작업 유형 열거형"""

    TENSOR_COMPUTATION = "tensor_computation"
    DATA_PROCESSING = "data_processing"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"


@dataclass
class SystemState:
    """시스템 상태 정보"""

    gpu_available: bool
    gpu_memory_free: float  # GB
    gpu_utilization: float  # 0-100%
    cpu_count: int
    cpu_usage: float  # 0-100%
    memory_available: float  # GB
    memory_usage: float  # 0-100%


@dataclass
class ComputeRequest:
    """연산 요청 정보"""

    data_size: int  # bytes
    task_type: TaskType
    priority: int = 1  # 1=low, 5=high
    requires_gpu: bool = False
    max_workers: Optional[int] = None


class CUDAStreamManager:
    """CUDA 스트림 관리자 - GPU 최적화 강화"""

    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size
        self.streams = []
        self.current_stream = 0
        self.gpu_available = torch.cuda.is_available()

        if self.gpu_available:
            self.streams = [torch.cuda.Stream() for _ in range(pool_size)]
            logger.info(f"✅ CUDA 스트림 풀 초기화 완료: {pool_size}개 스트림")
        else:
            logger.warning("GPU 미사용 환경 - CUDA 스트림 관리자 비활성화")

    def get_next_stream(self) -> Optional[torch.cuda.Stream]:
        """다음 사용 가능한 CUDA 스트림 반환"""
        if not self.gpu_available or not self.streams:
            return None

        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return stream

    @contextmanager
    def stream_context(self):
        """CUDA 스트림 컨텍스트 관리자"""
        if not self.gpu_available:
            yield None
            return

        stream = self.get_next_stream()
        if stream is None:
            yield None
            return

        try:
            with torch.cuda.stream(stream):
                yield stream
        finally:
            if stream:
                stream.synchronize()

    def synchronize_all(self):
        """모든 스트림 동기화"""
        if self.gpu_available:
            for stream in self.streams:
                stream.synchronize()

    def get_stats(self) -> Dict[str, Any]:
        """스트림 관리자 통계"""
        return {
            "pool_size": self.pool_size,
            "gpu_available": self.gpu_available,
            "current_stream_index": self.current_stream,
            "active_streams": len(self.streams),
        }


class SystemMonitor:
    """실시간 시스템 상태 모니터링"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()

    def get_current_state(self) -> SystemState:
        """현재 시스템 상태 반환"""
        # CPU 정보
        cpu_count = os.cpu_count() or 1
        cpu_usage = psutil.cpu_percent(interval=0.1)

        # 메모리 정보
        memory = psutil.virtual_memory()
        memory_available = memory.available / (1024**3)  # GB
        memory_usage = memory.percent

        # GPU 정보
        gpu_memory_free = 0.0
        gpu_utilization = 0.0

        if self.gpu_available:
            try:
                gpu_memory_free = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                ) / (
                    1024**3
                )  # GB

                # GPU 사용률은 간단히 메모리 사용률로 추정
                total_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                gpu_utilization = (1 - gpu_memory_free / total_memory) * 100

            except Exception as e:
                logger.warning(f"GPU 상태 확인 실패: {e}")
                gpu_memory_free = 0.0
                gpu_utilization = 100.0

        return SystemState(
            gpu_available=self.gpu_available,
            gpu_memory_free=gpu_memory_free,
            gpu_utilization=gpu_utilization,
            cpu_count=cpu_count,
            cpu_usage=cpu_usage,
            memory_available=memory_available,
            memory_usage=memory_usage,
        )


class OptimalComputeSelector:
    """최적 연산 전략 선택기"""

    def __init__(self):
        self.monitor = SystemMonitor()
        self.config = get_config("main").get_nested(
            "utils.unified_performance_engine", {}
        )

        # 임계값 설정
        self.gpu_memory_threshold = self.config.get("gpu_memory_threshold_gb", 1.0)
        self.gpu_utilization_threshold = self.config.get(
            "gpu_utilization_threshold", 80.0
        )
        self.multithread_data_threshold = self.config.get(
            "multithread_data_threshold_mb", 10
        )
        self.cpu_usage_threshold = self.config.get("cpu_usage_threshold", 80.0)
        self.min_workers_for_multithread = self.config.get(
            "min_workers_for_multithread", 2
        )

        logger.info(
            f"✅ 최적 연산 전략 선택기 초기화 (GPU 임계값: {self.gpu_memory_threshold}GB)"
        )

    def select_strategy(self, request: ComputeRequest) -> ComputeStrategy:
        """최적 연산 전략 선택"""
        state = self.monitor.get_current_state()
        data_size_mb = request.data_size / (1024**2)

        # 1. GPU 전략 검토
        if self._should_use_gpu(request, state, data_size_mb):
            logger.debug(f"GPU 전략 선택 (데이터: {data_size_mb:.1f}MB)")
            return ComputeStrategy.GPU

        # 2. 멀티쓰레드 CPU 전략 검토
        if self._should_use_multithread_cpu(request, state, data_size_mb):
            logger.debug(
                f"멀티쓰레드 CPU 전략 선택 (데이터: {data_size_mb:.1f}MB, 워커: {state.cpu_count})"
            )
            return ComputeStrategy.MULTITHREAD_CPU

        # 3. 단일 CPU 전략 (기본값)
        logger.debug(f"단일 CPU 전략 선택 (데이터: {data_size_mb:.1f}MB)")
        return ComputeStrategy.SINGLE_CPU

    def _should_use_gpu(
        self, request: ComputeRequest, state: SystemState, data_size_mb: float
    ) -> bool:
        """GPU 사용 여부 결정"""
        if not state.gpu_available:
            return False

        # 강제 GPU 요구사항
        if request.requires_gpu:
            return True

        # GPU 메모리 부족
        if state.gpu_memory_free < self.gpu_memory_threshold:
            logger.debug(
                f"GPU 메모리 부족: {state.gpu_memory_free:.1f}GB < {self.gpu_memory_threshold}GB"
            )
            return False

        # GPU 과부하
        if state.gpu_utilization > self.gpu_utilization_threshold:
            logger.debug(
                f"GPU 과부하: {state.gpu_utilization:.1f}% > {self.gpu_utilization_threshold}%"
            )
            return False

        # 텐서 연산은 GPU 우선
        if request.task_type == TaskType.TENSOR_COMPUTATION:
            return True

        # 메모리 집약적 작업은 GPU 메모리 여유 시에만
        if request.task_type == TaskType.MEMORY_INTENSIVE:
            return state.gpu_memory_free > self.gpu_memory_threshold * 2

        # 데이터 크기 기반 결정 (큰 데이터는 GPU가 유리)
        return data_size_mb > self.multithread_data_threshold * 5

    def _should_use_multithread_cpu(
        self, request: ComputeRequest, state: SystemState, data_size_mb: float
    ) -> bool:
        """멀티쓰레드 CPU 사용 여부 결정"""
        # CPU 코어 수 부족
        if state.cpu_count < self.min_workers_for_multithread:
            return False

        # CPU 과부하
        if state.cpu_usage > self.cpu_usage_threshold:
            return False

        # 데이터 크기 기반 결정
        if data_size_mb > self.multithread_data_threshold:
            return True

        # I/O 집약적 작업은 멀티쓰레드가 유리
        if request.task_type == TaskType.IO_INTENSIVE:
            return True

        return False


class AutoPerformanceMonitor:
    """자동 성능 모니터링"""

    def __init__(self):
        self.performance_history = {}
        self.lock = threading.Lock()

    @contextmanager
    def track(self, name: str, track_gpu: bool = True, track_memory_pool: bool = True):
        """성능 추적 컨텍스트 매니저"""
        start_time = time.time()
        gpu_memory_start = 0
        memory_pool_start = {}

        # GPU 메모리 추적
        if track_gpu and torch.cuda.is_available():
            try:
                gpu_memory_start = torch.cuda.memory_allocated()
            except Exception:
                pass

        # 메모리 풀 추적
        if track_memory_pool:
            try:
                memory_manager = get_unified_memory_manager()
                memory_pool_start = memory_manager.get_pool_stats()
            except Exception:
                pass

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # 성능 정보 수집
            perf_info = {
                "duration": duration,
                "timestamp": end_time,
            }

            # GPU 메모리 사용량
            if track_gpu and torch.cuda.is_available():
                try:
                    gpu_memory_end = torch.cuda.memory_allocated()
                    perf_info["gpu_memory_used"] = gpu_memory_end - gpu_memory_start
                except Exception:
                    pass

            # 메모리 풀 사용량
            if track_memory_pool:
                try:
                    memory_manager = get_unified_memory_manager()
                    memory_pool_end = memory_manager.get_pool_stats()
                    perf_info["memory_pool_delta"] = {
                        pool_name: memory_pool_end.get(pool_name, 0)
                        - memory_pool_start.get(pool_name, 0)
                        for pool_name in memory_pool_start.keys()
                    }
                except Exception:
                    pass

            # 성능 히스토리 저장
            with self.lock:
                if name not in self.performance_history:
                    self.performance_history[name] = []
                self.performance_history[name].append(perf_info)

                # 최대 100개 기록 유지
                if len(self.performance_history[name]) > 100:
                    self.performance_history[name] = self.performance_history[name][
                        -100:
                    ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        with self.lock:
            summary = {}
            for name, history in self.performance_history.items():
                if not history:
                    continue

                durations = [h["duration"] for h in history]
                summary[name] = {
                    "count": len(history),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_duration": sum(durations),
                    "last_run": history[-1]["timestamp"],
                }

                # GPU 메모리 통계
                gpu_memories = [
                    h.get("gpu_memory_used", 0)
                    for h in history
                    if "gpu_memory_used" in h
                ]
                if gpu_memories:
                    summary[name]["avg_gpu_memory"] = sum(gpu_memories) / len(
                        gpu_memories
                    )
                    summary[name]["max_gpu_memory"] = max(gpu_memories)

            return summary

    def clear_history(self):
        """성능 히스토리 초기화"""
        with self.lock:
            self.performance_history.clear()


class UnifiedPerformanceEngine:
    """통합 성능 최적화 엔진"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.config = get_config("main").get_nested(
            "utils.unified_performance_engine", {}
        )

        # 컴포넌트 초기화
        self.selector = OptimalComputeSelector()
        self.monitor = AutoPerformanceMonitor()
        self.cuda_stream_manager = CUDAStreamManager() if self.gpu_available else None

        # 실행기 초기화
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

        # 메모리 관리자
        self.memory_manager = get_memory_manager() if self.gpu_available else None

        # 에러 핸들러 (지연 로딩)
        self.error_handler = None

        logger.info("✅ 통합 성능 최적화 엔진 초기화 완료")

    def _get_error_handler(self):
        """에러 핸들러 지연 로딩"""
        if self.error_handler is None:
            from .error_handler import get_error_handler

            self.error_handler = get_error_handler()
        return self.error_handler

    @auto_recoverable(strategy=RecoveryStrategy.GPU_OOM)
    def execute(
        self,
        func: Callable,
        data: Any,
        task_type: TaskType = TaskType.DATA_PROCESSING,
        **kwargs,
    ) -> Any:
        """
        최적 전략을 선택하여 함수를 실행합니다.

        Args:
            func: 실행할 함수
            data: 입력 데이터
            task_type: 작업 유형
            **kwargs: 추가 인수

        Returns:
            함수 실행 결과
        """
        # 연산 요청 생성
        request = ComputeRequest(
            data_size=self._estimate_data_size(data),
            task_type=task_type,
            requires_gpu=kwargs.get("requires_gpu", False),
            max_workers=kwargs.get("max_workers"),
        )

        # 최적 전략 선택
        strategy = self.selector.select_strategy(request)

        # 성능 모니터링과 함께 실행
        with self.monitor.track(f"{func.__name__}_{strategy.value}"):
            try:
                if strategy == ComputeStrategy.GPU:
                    return self._execute_gpu(func, data, **kwargs)
                elif strategy == ComputeStrategy.MULTITHREAD_CPU:
                    return self._execute_multithread_cpu(func, data, **kwargs)
                else:
                    return self._execute_single_cpu(func, data, **kwargs)
            except Exception as e:
                logger.warning(f"실행 전략 {strategy.value} 실패, 단일 CPU로 폴백: {e}")
                return self._execute_single_cpu(func, data, **kwargs)

    def _execute_gpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """GPU 실행"""
        device = torch.device("cuda")

        # CUDA 스트림 컨텍스트 사용
        if self.cuda_stream_manager:
            with self.cuda_stream_manager.stream_context() as stream:
                return self._execute_gpu_with_stream(
                    func, data, device, stream, **kwargs
                )
        else:
            return self._execute_gpu_with_stream(func, data, device, None, **kwargs)

    def _execute_gpu_with_stream(
        self,
        func: Callable,
        data: Any,
        device: torch.device,
        stream: Optional[torch.cuda.Stream],
        **kwargs,
    ) -> Any:
        """스트림을 사용한 GPU 실행"""
        # 데이터를 GPU로 이동
        if isinstance(data, torch.Tensor):
            data = data.to(device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(device, non_blocking=True)
        elif isinstance(data, list):
            # 리스트의 경우 텐서로 변환 시도
            try:
                data = torch.tensor(data, device=device)
            except Exception:
                # 변환 실패 시 CPU에서 실행
                return self._execute_single_cpu(func, data, **kwargs)

        # 함수 실행
        result = func(data, **kwargs)

        # 결과를 CPU로 이동 (필요한 경우)
        if isinstance(result, torch.Tensor) and result.is_cuda:
            result = result.cpu()

        return result

    def _execute_multithread_cpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """멀티쓰레드 CPU 실행"""
        if not isinstance(data, list):
            return self._execute_single_cpu(func, data, **kwargs)

        # 데이터 청크 분할
        chunk_size = max(1, len(data) // (os.cpu_count() or 1))
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 병렬 실행
        def func_wrapper(chunk):
            return func(chunk, **kwargs)

        futures = [self.executor.submit(func_wrapper, chunk) for chunk in chunks]
        results = [future.result() for future in futures]

        # 결과 병합
        if all(isinstance(r, list) for r in results):
            return [item for sublist in results for item in sublist]
        else:
            return results

    def _execute_single_cpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """단일 CPU 실행"""
        return func(data, **kwargs)

    def _estimate_data_size(self, data: Any) -> int:
        """데이터 크기 추정"""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, list):
            return len(data) * 8  # 대략적인 추정
        elif isinstance(data, dict):
            return len(str(data).encode())
        else:
            return len(str(data).encode())

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = {
            "performance_summary": self.monitor.get_performance_summary(),
            "system_state": self.selector.monitor.get_current_state().__dict__,
        }

        if self.cuda_stream_manager:
            stats["cuda_streams"] = self.cuda_stream_manager.get_stats()

        return stats

    def shutdown(self):
        """엔진 종료"""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.cuda_stream_manager:
            self.cuda_stream_manager.synchronize_all()
        logger.info("통합 성능 최적화 엔진 종료 완료")


# 싱글톤 인스턴스
_unified_performance_engine = None
_engine_lock = threading.Lock()


def get_unified_performance_engine() -> UnifiedPerformanceEngine:
    """통합 성능 최적화 엔진 싱글톤 인스턴스 반환"""
    global _unified_performance_engine
    if _unified_performance_engine is None:
        with _engine_lock:
            if _unified_performance_engine is None:
                _unified_performance_engine = UnifiedPerformanceEngine()
    return _unified_performance_engine


def smart_execute(
    func: Callable, data: Any, task_type: TaskType = TaskType.DATA_PROCESSING, **kwargs
) -> Any:
    """
    스마트 실행 함수 - 최적 전략을 자동 선택하여 함수 실행

    Args:
        func: 실행할 함수
        data: 입력 데이터
        task_type: 작업 유형
        **kwargs: 추가 인수

    Returns:
        함수 실행 결과
    """
    engine = get_unified_performance_engine()
    return engine.execute(func, data, task_type, **kwargs)


# 하위 호환성을 위한 별칭
def get_smart_computation_engine() -> UnifiedPerformanceEngine:
    """하위 호환성을 위한 별칭"""
    return get_unified_performance_engine()


def get_auto_performance_monitor() -> AutoPerformanceMonitor:
    """성능 모니터 반환"""
    engine = get_unified_performance_engine()
    return engine.monitor


def get_cuda_stream_manager() -> Optional[CUDAStreamManager]:
    """CUDA 스트림 관리자 반환"""
    engine = get_unified_performance_engine()
    return engine.cuda_stream_manager
