"""
고성능 연산 및 모니터링 엔진 (v3 - 초경량화)

GPU 우선 연산 처리와 자동화된 성능 모니터링을 위한 초경량화 모듈.
파일 크기와 의존성을 최소화하여 순환 참조 위험을 원천 차단합니다.
"""

import time
import torch
import numpy as np
import threading
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from .unified_logging import get_logger
from .memory_manager import get_memory_manager  # 메모리 관리자 import
from .unified_config import get_config
from .compute_strategy import get_compute_executor, smart_execute, TaskType
from .unified_memory_manager import get_unified_memory_manager
from .auto_recovery_system import auto_recoverable, RecoveryStrategy

# from .error_handler import get_error_handler # 지연 로딩

logger = get_logger(__name__)
# error_handler = get_error_handler() # 지연 로딩


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


class SmartComputationEngine:
    """GPU > 멀티쓰레드 > CPU 우선순위 자동 연산 엔진"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()

        config = get_config("main").get_nested("utils.performance_optimizer", {})
        self.cpu_threshold = config.get(
            "cpu_threshold", 10000
        )  # 이 크기 이하 데이터는 CPU 사용
        self.multithread_threshold = config.get(
            "multithread_threshold", 100
        )  # 이 개수 이상 리스트는 멀티쓰레드 사용

        self.executor = ThreadPoolExecutor()
        self.error_handler = None  # 지연 로딩
        self.memory_manager = get_memory_manager() if self.gpu_available else None

        # CUDA 스트림 관리자 추가
        self.cuda_stream_manager = CUDAStreamManager() if self.gpu_available else None

    def _get_error_handler(self):
        if self.error_handler is None:
            from .error_handler import get_error_handler

            self.error_handler = get_error_handler()
        return self.error_handler

    @auto_recoverable(strategy=RecoveryStrategy.GPU_OOM)
    def execute(self, func: Callable, data: Any, **kwargs) -> Any:
        """
        새로운 통합 시스템을 사용하여 최적의 실행 전략을 선택합니다.
        """
        # 새로운 compute_strategy 시스템 사용
        task_type = (
            TaskType.TENSOR_COMPUTATION
            if self._is_tensor_operation(data)
            else TaskType.DATA_PROCESSING
        )

        try:
            # 스마트 실행 사용 (GPU > 멀티쓰레드 CPU > CPU 자동 선택)
            return smart_execute(func, data, task_type, **kwargs)
        except Exception as e:
            logger.warning(f"스마트 실행 실패, 기존 방식으로 폴백: {e}")
            # 기존 방식으로 폴백
            return self._legacy_execute(func, data, **kwargs)

    def _is_tensor_operation(self, data: Any) -> bool:
        """텐서 연산 여부 판단"""
        return isinstance(data, (torch.Tensor, np.ndarray))

    def _legacy_execute(self, func: Callable, data: Any, **kwargs) -> Any:
        """기존 실행 방식 (폴백용)"""
        # 에러 핸들러 지연 로딩
        error_handler = self._get_error_handler()

        # 실제 실행 로직을 내부 함수로 감싸서 전달
        def _execute_logic():
            data_size = self._get_data_size(data)

            # GPU 사용 조건: GPU 사용 가능, 데이터 크기 임계값 이상, GPU 메모리 여유
            use_gpu = (
                self.gpu_available
                and data_size > self.cpu_threshold
                and self.memory_manager
                and self.memory_manager._gpu_memory_ok()
            )

            if use_gpu:
                try:
                    logger.debug(f"연산 '{func.__name__}'에 GPU 전략 선택")
                    return self._execute_gpu(func, data, **kwargs)
                except Exception as e:
                    logger.warning(f"GPU 연산 실패, CPU로 폴백: {e}")
                    # GPU 실패 시 CPU로 자동 폴백 (에러 핸들러가 재시도)
                    return self._execute_cpu_strategy(func, data, **kwargs)
            else:
                logger.debug(f"연산 '{func.__name__}'에 CPU 전략 선택")
                return self._execute_cpu_strategy(func, data, **kwargs)

        return error_handler.execute_with_retry(_execute_logic)

    def _execute_cpu_strategy(self, func: Callable, data: Any, **kwargs) -> Any:
        """데이터 특성에 따라 단일/멀티 스레드 CPU 실행을 결정합니다."""
        # 데이터가 리스트이고, 병렬 처리가 유리할 만큼 충분히 클 경우
        if isinstance(data, list) and len(data) > self.multithread_threshold:
            logger.debug(f"연산 '{func.__name__}'에 멀티스레드 CPU 전략 선택")
            return self._execute_cpu_multithread(func, data, **kwargs)
        else:
            logger.debug(f"연산 '{func.__name__}'에 단일 스레드 CPU 전략 선택")
            return self._execute_cpu_single_thread(func, data, **kwargs)

    def _execute_gpu(self, func: Callable, data: Any, **kwargs) -> Any:
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
        """CUDA 스트림을 사용한 GPU 실행"""
        # 데이터를 텐서로 변환하고 GPU로 이동
        if isinstance(data, np.ndarray):
            data_tensor = torch.from_numpy(data).to(device, non_blocking=True)
        else:  # 이미 텐서인 경우
            data_tensor = data.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            result = func(data_tensor, **kwargs)

        # 결과가 텐서이면 CPU로 다시 이동 (필요시)
        if isinstance(result, torch.Tensor):
            return result.cpu()
        return result

    def _execute_cpu_single_thread(self, func: Callable, data: Any, **kwargs) -> Any:
        """단일 스레드로 CPU에서 연산을 실행합니다."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        return func(data, **kwargs)

    def _execute_cpu_multithread(
        self, func: Callable, data_list: List[Any], **kwargs
    ) -> List[Any]:
        """ThreadPoolExecutor를 사용하여 멀티스레드로 CPU 연산을 실행합니다."""

        # 각 데이터 항목에 대해 func를 실행하는 래퍼
        def func_wrapper(item):
            if isinstance(item, torch.Tensor):
                item = item.cpu().numpy()
            return func(item, **kwargs)

        results = list(self.executor.map(func_wrapper, data_list))
        return results

    def _get_data_size(self, data: Any) -> int:
        if isinstance(data, np.ndarray):
            return data.nbytes
        if isinstance(data, torch.Tensor):
            return data.element_size() * data.nelement()
        if isinstance(data, (list, tuple)):
            # 리스트의 경우, 첫번째 요소의 크기를 기준으로 전체 크기 추정
            if not data:
                return 0
            # np.array나 torch.tensor가 아닌 일반 list의 경우, len()이 더 적합
            if not isinstance(data[0], (np.ndarray, torch.Tensor)):
                return len(data)

            sample_size = 1
            if isinstance(data[0], np.ndarray):
                sample_size = data[0].nbytes
            elif isinstance(data[0], torch.Tensor):
                sample_size = data[0].element_size() * data[0].nelement()
            return sample_size * len(data)
        return 1

    def shutdown(self):
        self.executor.shutdown()


class AutoPerformanceMonitor:
    """98% 자동화 성능 모니터. @contextmanager를 사용하여 특정 코드 블록의 성능을 추적합니다."""

    def __init__(self):
        self.performance_history = []
        self.gpu_available = torch.cuda.is_available()
        self.memory_manager = get_memory_manager() if self.gpu_available else None

    @contextmanager
    def track(self, name: str, track_gpu: bool = True, track_memory_pool: bool = True):
        """
        지정된 코드 블록의 실행 시간과 GPU 메모리 사용량을 추적합니다.
        """
        start_time = time.time()

        gpu_available = track_gpu and self.gpu_available
        mem_before = torch.cuda.memory_allocated() if gpu_available else 0

        # 메모리 풀 상태 추적
        pool_stats_before = None
        if (
            track_memory_pool
            and self.memory_manager
            and hasattr(self.memory_manager, "memory_pool")
        ):
            pool_stats_before = {
                "total_tensors": sum(
                    len(pool) for pool in self.memory_manager.memory_pool.values()
                ),
                "pool_keys": len(self.memory_manager.memory_pool),
            }

        try:
            yield
        finally:
            duration = time.time() - start_time
            mem_after = torch.cuda.memory_allocated() if gpu_available else 0
            mem_used = (mem_after - mem_before) / (1024 * 1024)  # MB

            log_msg = f"성능 모니터 [{name}]: 실행 시간={duration:.4f}s"

            if gpu_available:
                log_msg += f", GPU 메모리 사용량={mem_used:.2f}MB"

            # 메모리 풀 통계 추가
            if track_memory_pool and self.memory_manager and pool_stats_before:
                pool_stats_after = {
                    "total_tensors": sum(
                        len(pool) for pool in self.memory_manager.memory_pool.values()
                    ),
                    "pool_keys": len(self.memory_manager.memory_pool),
                }
                pool_change = (
                    pool_stats_after["total_tensors"]
                    - pool_stats_before["total_tensors"]
                )
                log_msg += f", 메모리 풀 변화={pool_change:+d}개"

            logger.info(log_msg)

            # 성능 히스토리 저장
            self.performance_history.append(
                {
                    "name": name,
                    "duration": duration,
                    "memory_used_mb": mem_used,
                    "timestamp": time.time(),
                }
            )

            # 히스토리 크기 제한 (최근 100개만 유지)
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 모니터링 요약 정보"""
        if not self.performance_history:
            return {"message": "성능 데이터 없음"}

        # 최근 10개 작업 분석
        recent_tasks = self.performance_history[-10:]

        total_duration = sum(task["duration"] for task in recent_tasks)
        avg_duration = total_duration / len(recent_tasks)
        max_duration = max(task["duration"] for task in recent_tasks)

        summary = {
            "total_tasks": len(self.performance_history),
            "recent_tasks": len(recent_tasks),
            "avg_duration": avg_duration,
            "max_duration": max_duration,
            "total_recent_duration": total_duration,
        }

        if self.gpu_available:
            total_memory = sum(task["memory_used_mb"] for task in recent_tasks)
            avg_memory = total_memory / len(recent_tasks)
            summary["avg_memory_mb"] = avg_memory
            summary["total_memory_mb"] = total_memory

        return summary

    def clear_history(self):
        """성능 히스토리 초기화"""
        self.performance_history.clear()
        logger.info("성능 모니터링 히스토리 초기화 완료")


# --- 싱글톤 인스턴스 ---
from .factory import get_singleton_instance


def get_smart_computation_engine() -> SmartComputationEngine:
    """SmartComputationEngine의 싱글톤 인스턴스를 반환합니다."""
    return get_singleton_instance(SmartComputationEngine)


def get_auto_performance_monitor() -> AutoPerformanceMonitor:
    """자동 성능 모니터 반환"""
    return AutoPerformanceMonitor()


# CUDA 스트림 관리자 싱글톤
_cuda_stream_manager = None


def get_cuda_stream_manager() -> CUDAStreamManager:
    """CUDA 스트림 관리자 싱글톤 반환"""
    global _cuda_stream_manager
    if _cuda_stream_manager is None:
        _cuda_stream_manager = CUDAStreamManager()
    return _cuda_stream_manager


# 공개 API
__all__ = [
    "SmartComputationEngine",
    "AutoPerformanceMonitor",
    "CUDAStreamManager",
    "get_smart_computation_engine",
    "get_auto_performance_monitor",
    "get_cuda_stream_manager",
]
