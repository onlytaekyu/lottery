"""
완전한 GPU 연산 우선순위 시스템
GPU > 멀티쓰레드 CPU > CPU 전략 구현
"""

import os
import torch
import psutil
import threading
from enum import Enum
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .unified_logging import get_logger
from .unified_config import get_config
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
        self.config = get_config("main").get_nested("utils.compute_strategy", {})

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
            logger.debug(
                f"CPU 과부하: {state.cpu_usage:.1f}% > {self.cpu_usage_threshold}%"
            )
            return False

        # I/O 집약적 작업은 멀티쓰레드 유리
        if request.task_type == TaskType.IO_INTENSIVE:
            return True

        # 데이터 처리 작업은 데이터 크기에 따라
        if request.task_type == TaskType.DATA_PROCESSING:
            return data_size_mb > self.multithread_data_threshold

        # 일반적인 경우: 중간 크기 이상 데이터
        return data_size_mb > self.multithread_data_threshold


class ComputeExecutor:
    """연산 전략별 실행기"""

    def __init__(self):
        self.selector = OptimalComputeSelector()
        self.gpu_executor = None
        self.cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    def execute(
        self,
        func: Callable,
        data: Any,
        task_type: TaskType = TaskType.DATA_PROCESSING,
        **kwargs,
    ) -> Any:
        """최적 전략으로 함수 실행"""

        # 요청 정보 생성
        data_size = self._estimate_data_size(data)
        request = ComputeRequest(data_size=data_size, task_type=task_type, **kwargs)

        # 전략 선택
        strategy = self.selector.select_strategy(request)

        # 전략별 실행
        if strategy == ComputeStrategy.GPU:
            return self._execute_gpu(func, data, **kwargs)
        elif strategy == ComputeStrategy.MULTITHREAD_CPU:
            return self._execute_multithread_cpu(func, data, **kwargs)
        else:
            return self._execute_single_cpu(func, data, **kwargs)

    def _execute_gpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """GPU에서 실행"""
        device = torch.device("cuda")

        # 데이터를 GPU로 이동
        if hasattr(data, "to"):
            data = data.to(device)
        elif isinstance(data, (list, tuple)):
            data = [item.to(device) if hasattr(item, "to") else item for item in data]

        try:
            with torch.cuda.amp.autocast():
                result = func(data, **kwargs)

            # 결과를 CPU로 이동 (필요시)
            if hasattr(result, "cpu"):
                result = result.cpu()

            return result

        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU 메모리 부족, CPU로 폴백")
            torch.cuda.empty_cache()
            return self._execute_multithread_cpu(func, data, **kwargs)

    def _execute_multithread_cpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """멀티쓰레드 CPU에서 실행"""
        # 데이터를 CPU로 이동
        if hasattr(data, "cpu"):
            data = data.cpu()

        # 리스트/배치 데이터인 경우 병렬 처리
        if isinstance(data, (list, tuple)) and len(data) > 1:
            futures = []
            for item in data:
                future = self.cpu_executor.submit(func, item, **kwargs)
                futures.append(future)

            results = [future.result() for future in futures]
            return results
        else:
            return func(data, **kwargs)

    def _execute_single_cpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """단일 CPU에서 실행"""
        # 데이터를 CPU로 이동
        if hasattr(data, "cpu"):
            data = data.cpu()

        return func(data, **kwargs)

    def _estimate_data_size(self, data: Any) -> int:
        """데이터 크기 추정"""
        if hasattr(data, "nbytes"):
            return data.nbytes
        elif hasattr(data, "element_size") and hasattr(data, "nelement"):
            return data.element_size() * data.nelement()
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_data_size(item) for item in data)
        else:
            return len(str(data).encode("utf-8"))

    def shutdown(self):
        """리소스 정리"""
        self.cpu_executor.shutdown(wait=True)


# 싱글톤 인스턴스
def get_compute_executor() -> ComputeExecutor:
    """ComputeExecutor 싱글톤 인스턴스 반환"""
    return get_singleton_instance(ComputeExecutor)


def get_optimal_compute_selector() -> OptimalComputeSelector:
    """OptimalComputeSelector 싱글톤 인스턴스 반환"""
    return get_singleton_instance(OptimalComputeSelector)


# 편의 함수
def smart_execute(
    func: Callable, data: Any, task_type: TaskType = TaskType.DATA_PROCESSING, **kwargs
) -> Any:
    """최적 전략으로 함수를 실행하는 편의 함수"""
    executor = get_compute_executor()
    return executor.execute(func, data, task_type, **kwargs)
