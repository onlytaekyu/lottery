"""
통합 성능 최적화 모듈

GPU 최우선 연산 처리, 멀티쓰레드 최적화, 메모리 관리, 프로파일링 기능을 제공합니다.
"""

import os
import time
import threading
import multiprocessing
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import gc
from functools import wraps
import logging
from pathlib import Path
import json
import numpy as np

# GPU 관련 import (선택적)
try:
    import torch
    import torch.nn as nn
    import torch.cuda.amp as amp

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# NVIDIA GPU 모니터링 (선택적)
try:
    import pynvml

    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False

# 순환 참조 방지를 위한 로거 설정
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""

    # 시간 관련
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None

    # CPU 관련
    cpu_usage_start: float = 0.0
    cpu_usage_end: float = 0.0
    cpu_count: int = field(default_factory=multiprocessing.cpu_count)

    # 메모리 관련
    memory_start: float = 0.0
    memory_end: float = 0.0
    memory_peak: float = 0.0
    memory_diff: float = 0.0

    # GPU 관련 (CUDA 사용 가능시)
    gpu_memory_start: Optional[float] = None
    gpu_memory_end: Optional[float] = None
    gpu_memory_peak: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_temperature: Optional[float] = None

    # 처리량 관련
    items_processed: int = 0
    throughput: Optional[float] = None  # items/second

    # 기타
    function_name: str = ""
    thread_id: int = field(default_factory=threading.get_ident)
    process_id: int = field(default_factory=os.getpid)

    def finalize(self):
        """메트릭 최종 계산"""
        if self.end_time is None:
            self.end_time = time.time()

        self.duration = self.end_time - self.start_time
        self.memory_diff = self.memory_end - self.memory_start

        if self.duration > 0 and self.items_processed > 0:
            self.throughput = self.items_processed / self.duration

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "function_name": self.function_name,
            "duration": self.duration,
            "cpu_usage_start": self.cpu_usage_start,
            "cpu_usage_end": self.cpu_usage_end,
            "memory_start_mb": self.memory_start,
            "memory_end_mb": self.memory_end,
            "memory_diff_mb": self.memory_diff,
            "memory_peak_mb": self.memory_peak,
            "gpu_memory_start_mb": self.gpu_memory_start,
            "gpu_memory_end_mb": self.gpu_memory_end,
            "gpu_memory_peak_mb": self.gpu_memory_peak,
            "gpu_utilization": self.gpu_utilization,
            "gpu_temperature": self.gpu_temperature,
            "items_processed": self.items_processed,
            "throughput": self.throughput,
            "thread_id": self.thread_id,
            "process_id": self.process_id,
        }


class GPUOptimizer:
    """GPU 최적화 클래스"""

    def __init__(self):
        self.device = None
        self.device_count = 0
        self.memory_fraction = 0.9
        self.amp_enabled = False

        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            self._setup_gpu_optimizations()
        else:
            self.device = torch.device("cpu") if TORCH_AVAILABLE else None
            logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")

    def _setup_gpu_optimizations(self):
        """GPU 최적화 설정"""
        if not TORCH_AVAILABLE:
            return

        try:
            # 메모리 풀 설정
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)

            # cuDNN 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # AMP 활성화
            self.amp_enabled = True

            logger.info(f"GPU 최적화 설정 완료 - {self.device_count}개 GPU 사용 가능")

        except Exception as e:
            logger.error(f"GPU 최적화 설정 실패: {e}")

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """GPU 메모리 정보 반환"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}

        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            memory_total = (
                torch.cuda.get_device_properties(0).total_memory / 1024**2
            )  # MB

            return {
                "allocated_mb": memory_allocated,
                "reserved_mb": memory_reserved,
                "total_mb": memory_total,
                "utilization_percent": (memory_allocated / memory_total) * 100,
            }
        except Exception as e:
            logger.error(f"GPU 메모리 정보 수집 실패: {e}")
            return {}

    def get_gpu_temperature(self) -> Optional[float]:
        """GPU 온도 반환"""
        if not PYNVML_AVAILABLE:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except Exception as e:
            logger.debug(f"GPU 온도 수집 실패: {e}")
            return None

    def optimize_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """텐서 GPU 최적화"""
        if not TORCH_AVAILABLE or self.device is None:
            return tensor

        try:
            # GPU로 이동
            if self.device.type == "cuda" and not tensor.is_cuda:
                tensor = tensor.to(self.device, non_blocking=True)

            # 메모리 레이아웃 최적화
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            return tensor

        except Exception as e:
            logger.error(f"텐서 최적화 실패: {e}")
            return tensor

    @contextmanager
    def autocast_context(self):
        """AMP autocast 컨텍스트"""
        if TORCH_AVAILABLE and self.amp_enabled and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

    def clear_gpu_cache(self):
        """GPU 캐시 정리"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class MemoryOptimizer:
    """메모리 최적화 클래스"""

    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_percent(self) -> float:
        """메모리 사용률 반환 (%)"""
        return self.process.memory_percent()

    def is_memory_critical(self) -> bool:
        """메모리 사용량이 임계점을 넘었는지 확인"""
        return self.get_memory_percent() > self.max_memory_percent

    def force_garbage_collection(self):
        """강제 가비지 컬렉션"""
        collected = gc.collect()
        logger.debug(f"가비지 컬렉션 완료: {collected}개 객체 정리")
        return collected

    @contextmanager
    def memory_monitor(self, threshold_mb: float = 100.0):
        """메모리 모니터링 컨텍스트"""
        start_memory = self.get_memory_usage()

        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            memory_diff = end_memory - start_memory

            if memory_diff > threshold_mb:
                logger.warning(f"메모리 사용량 증가: {memory_diff:.1f}MB")

            if self.is_memory_critical():
                logger.warning(
                    "메모리 사용량이 임계점을 넘었습니다. 가비지 컬렉션을 수행합니다."
                )
                self.force_garbage_collection()


class MultiThreadOptimizer:
    """멀티스레드 최적화 클래스"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.cpu_count = os.cpu_count() or 1

    def parallel_map(
        self,
        func: Callable,
        items: List[Any],
        use_processes: bool = False,
        chunk_size: Optional[int] = None,
    ) -> List[Any]:
        """병렬 맵 처리"""
        if len(items) <= 1:
            return [func(item) for item in items]

        # 청크 크기 자동 계산
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 2))

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        try:
            with executor_class(max_workers=self.max_workers) as executor:
                if use_processes and chunk_size > 1:
                    # 프로세스 풀에서는 청크 단위로 처리
                    chunks = [
                        items[i : i + chunk_size]
                        for i in range(0, len(items), chunk_size)
                    ]
                    chunk_func = lambda chunk: [func(item) for item in chunk]

                    futures = [executor.submit(chunk_func, chunk) for chunk in chunks]
                    results = []

                    for future in as_completed(futures):
                        results.extend(future.result())

                    # 원래 순서 복원
                    return results[: len(items)]
                else:
                    # 스레드 풀에서는 개별 처리
                    futures = [executor.submit(func, item) for item in items]
                    return [future.result() for future in futures]

        except Exception as e:
            logger.error(f"병렬 처리 실패: {e}")
            # 폴백: 순차 처리
            return [func(item) for item in items]

    def batch_process(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int = 100,
        progress_callback: Optional[Callable] = None,
    ) -> List[Any]:
        """배치 처리"""
        results = []
        total_batches = (len(items) + batch_size - 1) // batch_size

        for i, batch_start in enumerate(range(0, len(items), batch_size)):
            batch_end = min(batch_start + batch_size, len(items))
            batch = items[batch_start:batch_end]

            batch_results = [func(item) for item in batch]
            results.extend(batch_results)

            if progress_callback:
                progress_callback(i + 1, total_batches)

        return results


class PerformanceProfiler:
    """성능 프로파일러"""

    def __init__(self, save_to_file: bool = True, output_dir: str = "logs/performance"):
        self.save_to_file = save_to_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_optimizer = GPUOptimizer()
        self.memory_optimizer = MemoryOptimizer()

        self.metrics_history: List[PerformanceMetrics] = []
        self._lock = threading.Lock()

    def profile_function(
        self,
        func: Optional[Callable] = None,
        *,
        include_gpu: bool = True,
        include_memory: bool = True,
        items_count: Optional[int] = None,
    ):
        """함수 성능 프로파일링 데코레이터"""

        def decorator(f: Callable):
            @wraps(f)
            def wrapper(*args, **kwargs):
                metrics = PerformanceMetrics(function_name=f.__name__)

                # 시작 메트릭 수집
                metrics.start_time = time.time()

                if include_memory:
                    metrics.memory_start = self.memory_optimizer.get_memory_usage()
                    metrics.cpu_usage_start = psutil.cpu_percent()

                if (
                    include_gpu
                    and self.gpu_optimizer.device
                    and self.gpu_optimizer.device.type == "cuda"
                ):
                    gpu_info = self.gpu_optimizer.get_gpu_memory_info()
                    metrics.gpu_memory_start = gpu_info.get("allocated_mb", 0)
                    metrics.gpu_temperature = self.gpu_optimizer.get_gpu_temperature()

                try:
                    # 함수 실행
                    result = f(*args, **kwargs)

                    # 처리된 아이템 수 설정
                    if items_count is not None:
                        metrics.items_processed = items_count
                    elif hasattr(result, "__len__"):
                        try:
                            metrics.items_processed = len(result)
                        except:
                            pass

                    return result

                finally:
                    # 종료 메트릭 수집
                    metrics.end_time = time.time()

                    if include_memory:
                        metrics.memory_end = self.memory_optimizer.get_memory_usage()
                        metrics.cpu_usage_end = psutil.cpu_percent()

                        # 메모리 피크 추정
                        metrics.memory_peak = max(
                            metrics.memory_start, metrics.memory_end
                        )

                    if (
                        include_gpu
                        and self.gpu_optimizer.device
                        and self.gpu_optimizer.device.type == "cuda"
                    ):
                        gpu_info = self.gpu_optimizer.get_gpu_memory_info()
                        metrics.gpu_memory_end = gpu_info.get("allocated_mb", 0)
                        metrics.gpu_utilization = gpu_info.get("utilization_percent", 0)

                        if metrics.gpu_memory_start is not None:
                            metrics.gpu_memory_peak = max(
                                metrics.gpu_memory_start, metrics.gpu_memory_end
                            )

                    # 메트릭 최종 계산
                    metrics.finalize()

                    # 메트릭 저장
                    self._save_metrics(metrics)

                    # 로깅
                    self._log_performance(metrics)

            return wrapper

        if func is None:
            return decorator
        else:
            return decorator(func)

    def _save_metrics(self, metrics: PerformanceMetrics):
        """메트릭 저장"""
        with self._lock:
            self.metrics_history.append(metrics)

            # 파일 저장
            if self.save_to_file:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"performance_{timestamp}_{metrics.function_name}.json"
                filepath = self.output_dir / filename

                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(metrics.to_dict(), f, indent=2)
                except Exception as e:
                    logger.error(f"성능 메트릭 저장 실패: {e}")

    def _log_performance(self, metrics: PerformanceMetrics):
        """성능 로깅"""
        log_msg = f"🚀 {metrics.function_name} 성능 분석:"
        log_msg += f"\n  ⏱️  실행 시간: {metrics.duration:.3f}초"

        if metrics.memory_diff != 0:
            log_msg += f"\n  💾 메모리 변화: {metrics.memory_diff:+.1f}MB"

        if metrics.throughput:
            log_msg += f"\n  📊 처리량: {metrics.throughput:.1f} items/sec"

        if metrics.gpu_memory_start is not None:
            gpu_diff = (metrics.gpu_memory_end or 0) - metrics.gpu_memory_start
            log_msg += f"\n  🎮 GPU 메모리 변화: {gpu_diff:+.1f}MB"

        if metrics.gpu_temperature:
            log_msg += f"\n  🌡️  GPU 온도: {metrics.gpu_temperature}°C"

        logger.info(log_msg)

    def get_performance_summary(
        self, function_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """성능 요약 반환"""
        with self._lock:
            if function_name:
                filtered_metrics = [
                    m for m in self.metrics_history if m.function_name == function_name
                ]
            else:
                filtered_metrics = self.metrics_history

            if not filtered_metrics:
                return {}

            durations = [m.duration for m in filtered_metrics if m.duration]
            memory_diffs = [m.memory_diff for m in filtered_metrics]
            throughputs = [m.throughput for m in filtered_metrics if m.throughput]

            return {
                "function_name": function_name or "all",
                "call_count": len(filtered_metrics),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "avg_memory_diff": (
                    sum(memory_diffs) / len(memory_diffs) if memory_diffs else 0
                ),
                "avg_throughput": (
                    sum(throughputs) / len(throughputs) if throughputs else 0
                ),
                "total_items_processed": sum(
                    m.items_processed for m in filtered_metrics
                ),
            }

    def clear_history(self):
        """히스토리 정리"""
        with self._lock:
            self.metrics_history.clear()


# 전역 인스턴스
_global_profiler = None
_profiler_lock = threading.Lock()


def get_performance_profiler() -> PerformanceProfiler:
    """전역 성능 프로파일러 반환"""
    global _global_profiler

    if _global_profiler is None:
        with _profiler_lock:
            if _global_profiler is None:
                _global_profiler = PerformanceProfiler()

    return _global_profiler


# 편의 함수들
def profile(func: Optional[Callable] = None, **kwargs):
    """성능 프로파일링 데코레이터 (편의 함수)"""
    profiler = get_performance_profiler()
    return profiler.profile_function(func, **kwargs)


@contextmanager
def gpu_optimization():
    """GPU 최적화 컨텍스트"""
    optimizer = GPUOptimizer()
    try:
        with optimizer.autocast_context():
            yield optimizer
    finally:
        optimizer.clear_gpu_cache()


@contextmanager
def memory_optimization(threshold_mb: float = 100.0):
    """메모리 최적화 컨텍스트"""
    optimizer = MemoryOptimizer()
    with optimizer.memory_monitor(threshold_mb):
        yield optimizer


def parallel_map(func: Callable, items: List[Any], **kwargs) -> List[Any]:
    """병렬 맵 처리 (편의 함수)"""
    optimizer = MultiThreadOptimizer()
    return optimizer.parallel_map(func, items, **kwargs)


def get_system_performance_report() -> Dict[str, Any]:
    """시스템 성능 리포트 생성"""
    # CPU 정보
    cpu_info = {
        "count": os.cpu_count(),
        "usage_percent": psutil.cpu_percent(interval=1),
        "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
    }

    # 메모리 정보
    memory_info = psutil.virtual_memory()._asdict()
    memory_info["usage_gb"] = memory_info["used"] / 1024**3
    memory_info["available_gb"] = memory_info["available"] / 1024**3

    # GPU 정보
    gpu_optimizer = GPUOptimizer()
    gpu_info = gpu_optimizer.get_gpu_memory_info()
    gpu_info["temperature"] = gpu_optimizer.get_gpu_temperature()
    gpu_info["device_count"] = gpu_optimizer.device_count

    # 성능 프로파일러 요약
    profiler = get_performance_profiler()
    performance_summary = profiler.get_performance_summary()

    return {
        "timestamp": time.time(),
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "performance_summary": performance_summary,
    }


@dataclass
class ComputationStrategy:
    """연산 전략 정의"""

    strategy_type: str  # "gpu", "multithread", "cpu"
    device: Optional[str] = None
    workers: Optional[int] = None
    memory_efficient: bool = True
    use_amp: bool = False

    def __post_init__(self):
        if (
            self.strategy_type == "gpu"
            and TORCH_AVAILABLE
            and torch.cuda.is_available()
        ):
            self.device = "cuda"
            self.use_amp = True
        elif self.strategy_type == "multithread":
            self.workers = self.workers or min(multiprocessing.cpu_count(), 8)
        elif self.strategy_type == "cpu":
            self.workers = 1


class SmartComputationEngine:
    """스마트 연산 엔진 - GPU > 멀티쓰레드 > CPU 순 처리"""

    def __init__(self, gpu_memory_threshold: float = 0.8):
        self.gpu_memory_threshold = gpu_memory_threshold
        self.logger = logging.getLogger(__name__)
        self._initialize_strategies()

    def _initialize_strategies(self):
        """사용 가능한 연산 전략 초기화"""
        self.strategies = []

        # 1. GPU 전략 (최우선)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.strategies.append(ComputationStrategy("gpu"))
            self.logger.info("GPU 연산 전략 활성화")

        # 2. 멀티쓰레드 전략
        if multiprocessing.cpu_count() > 1:
            self.strategies.append(ComputationStrategy("multithread"))
            self.logger.info(
                f"멀티쓰레드 연산 전략 활성화 ({multiprocessing.cpu_count()}개 코어)"
            )

        # 3. CPU 전략 (기본)
        self.strategies.append(ComputationStrategy("cpu"))
        self.logger.info("CPU 연산 전략 활성화")

    def select_optimal_strategy(
        self, data_size: int, operation_type: str = "general"
    ) -> ComputationStrategy:
        """최적 연산 전략 선택"""

        # GPU 전략 우선 검토
        for strategy in self.strategies:
            if strategy.strategy_type == "gpu":
                if self._is_gpu_available_for_computation(data_size):
                    self.logger.info(f"GPU 전략 선택 (데이터 크기: {data_size})")
                    return strategy
                else:
                    self.logger.warning("GPU 메모리 부족 - 다른 전략 선택")

        # 멀티쓰레드 전략 검토
        for strategy in self.strategies:
            if strategy.strategy_type == "multithread":
                if data_size > 1000:  # 큰 데이터에 대해서만 멀티쓰레드 활용
                    self.logger.info(f"멀티쓰레드 전략 선택 (데이터 크기: {data_size})")
                    return strategy

        # CPU 전략 (기본)
        cpu_strategy = next(s for s in self.strategies if s.strategy_type == "cpu")
        self.logger.info(f"CPU 전략 선택 (데이터 크기: {data_size})")
        return cpu_strategy

    def _is_gpu_available_for_computation(self, data_size: int) -> bool:
        """GPU 연산 가능 여부 확인"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return False

        try:
            # GPU 메모리 사용률 확인
            memory_allocated = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_usage_ratio = memory_allocated / memory_total

            # 예상 메모리 사용량 계산 (대략적)
            estimated_memory_needed = data_size * 4 * 8  # float32 * 8 (여유분)
            available_memory = memory_total - memory_allocated

            return (
                memory_usage_ratio < self.gpu_memory_threshold
                and estimated_memory_needed < available_memory
            )
        except Exception as e:
            self.logger.error(f"GPU 메모리 확인 실패: {e}")
            return False

    def execute_computation(self, func: Callable, data: Any, **kwargs) -> Any:
        """최적 전략으로 연산 실행"""

        # 데이터 크기 추정
        data_size = self._estimate_data_size(data)

        # 최적 전략 선택
        strategy = self.select_optimal_strategy(
            data_size, kwargs.get("operation_type", "general")
        )

        # 전략별 실행
        if strategy.strategy_type == "gpu":
            return self._execute_gpu_computation(func, data, strategy, **kwargs)
        elif strategy.strategy_type == "multithread":
            return self._execute_multithread_computation(func, data, strategy, **kwargs)
        else:
            return self._execute_cpu_computation(func, data, strategy, **kwargs)

    def _estimate_data_size(self, data: Any) -> int:
        """데이터 크기 추정"""
        if hasattr(data, "__len__"):
            return len(data)
        elif isinstance(data, np.ndarray):
            return data.size
        elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            return data.numel()
        else:
            return 1

    def _execute_gpu_computation(
        self, func: Callable, data: Any, strategy: ComputationStrategy, **kwargs
    ) -> Any:
        """GPU 연산 실행"""
        try:
            device = torch.device(strategy.device)

            # 데이터를 GPU로 이동
            if isinstance(data, (list, tuple)):
                # 리스트/튜플 데이터 처리
                if all(isinstance(item, (int, float)) for item in data):
                    gpu_data = torch.tensor(data, device=device)
                else:
                    gpu_data = data  # 복합 데이터는 함수 내에서 처리
            elif isinstance(data, np.ndarray):
                gpu_data = torch.from_numpy(data).to(device)
            elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                gpu_data = data.to(device)
            else:
                gpu_data = data

            # AMP 컨텍스트 사용
            if strategy.use_amp:
                with torch.cuda.amp.autocast():
                    result = func(gpu_data, **kwargs)
            else:
                result = func(gpu_data, **kwargs)

            # 결과를 CPU로 이동 (필요시)
            if TORCH_AVAILABLE and isinstance(result, torch.Tensor) and result.is_cuda:
                result = result.cpu()

            return result

        except Exception as e:
            self.logger.error(f"GPU 연산 실패: {e}")
            # GPU 실패 시 멀티쓰레드로 폴백
            fallback_strategy = ComputationStrategy("multithread")
            return self._execute_multithread_computation(
                func, data, fallback_strategy, **kwargs
            )

    def _execute_multithread_computation(
        self, func: Callable, data: Any, strategy: ComputationStrategy, **kwargs
    ) -> Any:
        """멀티쓰레드 연산 실행"""
        try:
            workers = strategy.workers or 1
            if hasattr(data, "__len__") and len(data) > workers:
                # 데이터를 청크로 분할
                chunk_size = len(data) // workers
                chunks = [
                    data[i : i + chunk_size] for i in range(0, len(data), chunk_size)
                ]

                # 멀티쓰레드 실행
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(func, chunk, **kwargs) for chunk in chunks
                    ]
                    results = [future.result() for future in as_completed(futures)]

                # 결과 병합
                if isinstance(results[0], (list, tuple)):
                    merged_result = []
                    for result in results:
                        merged_result.extend(result)
                    return merged_result
                elif isinstance(results[0], np.ndarray):
                    return np.concatenate(results)
                else:
                    return results
            else:
                # 작은 데이터는 단일 스레드로 처리
                return func(data, **kwargs)

        except Exception as e:
            self.logger.error(f"멀티쓰레드 연산 실패: {e}")
            # 멀티쓰레드 실패 시 CPU로 폴백
            fallback_strategy = ComputationStrategy("cpu")
            return self._execute_cpu_computation(
                func, data, fallback_strategy, **kwargs
            )

    def _execute_cpu_computation(
        self, func: Callable, data: Any, strategy: ComputationStrategy, **kwargs
    ) -> Any:
        """CPU 연산 실행"""
        try:
            return func(data, **kwargs)
        except Exception as e:
            self.logger.error(f"CPU 연산 실패: {e}")
            raise


# 전역 스마트 연산 엔진 인스턴스
_smart_computation_engine = None


def get_smart_computation_engine() -> SmartComputationEngine:
    """스마트 연산 엔진 인스턴스 반환"""
    global _smart_computation_engine
    if _smart_computation_engine is None:
        _smart_computation_engine = SmartComputationEngine()
    return _smart_computation_engine


def smart_compute(func: Callable, data: Any, **kwargs) -> Any:
    """스마트 연산 실행 (GPU > 멀티쓰레드 > CPU 순)"""
    engine = get_smart_computation_engine()
    return engine.execute_computation(func, data, **kwargs)


def optimize_computation(
    data: Any, use_gpu: bool = True, use_multithread: bool = True
) -> Dict[str, Any]:
    """연산 최적화 정보 반환"""
    engine = get_smart_computation_engine()
    data_size = engine._estimate_data_size(data)
    strategy = engine.select_optimal_strategy(data_size)

    return {
        "data_size": data_size,
        "selected_strategy": strategy.strategy_type,
        "device": strategy.device,
        "workers": strategy.workers,
        "gpu_available": TORCH_AVAILABLE and torch.cuda.is_available(),
        "cpu_count": multiprocessing.cpu_count(),
        "memory_efficient": strategy.memory_efficient,
        "use_amp": strategy.use_amp,
    }
