"""
하이브리드 최적화 시스템

ProcessPool + MemoryPool + CUDA를 통합한 최적화 시스템입니다.
작업 특성에 따라 최적의 처리 방식을 자동으로 선택합니다.
"""

import time
import psutil
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps

from .error_handler_refactored import get_logger
from .process_pool_manager import ProcessPoolManager, ProcessPoolConfig
from .memory_manager import MemoryManager, MemoryConfig
from .cuda_optimizers import CudaOptimizer, CudaConfig

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """최적화 전략"""

    AUTO = "auto"
    CPU_PARALLEL = "cpu_parallel"
    MEMORY_OPTIMIZED = "memory_optimized"
    GPU_ACCELERATED = "gpu_accelerated"
    HYBRID = "hybrid"


@dataclass
class TaskInfo:
    """작업 정보"""

    function_type: str  # "analysis", "vectorize", "train", "inference"
    data_size: int = 0
    parallelizable: bool = True
    gpu_compatible: bool = False
    memory_intensive: bool = False
    cpu_intensive: bool = True


@dataclass
class HybridConfig:
    """하이브리드 최적화 설정"""

    auto_optimization: bool = True
    memory_threshold: float = 0.8  # 80% 메모리 사용량 임계점
    cpu_threshold: float = 75.0  # 75% CPU 사용량 임계점
    gpu_threshold: float = 80.0  # 80% GPU 사용량 임계점
    min_parallel_size: int = 100  # 병렬 처리 최소 데이터 크기
    enable_monitoring: bool = True


class HybridOptimizer:
    """하이브리드 최적화 관리자"""

    def __init__(self, config: Union[HybridConfig, Dict[str, Any]]):
        # 딕셔너리가 전달된 경우 HybridConfig로 변환
        if isinstance(config, dict):
            self.config = HybridConfig(
                auto_optimization=config.get("auto_optimization", True),
                memory_threshold=config.get("memory_threshold", 0.8),
                cpu_threshold=config.get("cpu_threshold", 75.0),
                gpu_threshold=config.get("gpu_threshold", 80.0),
                min_parallel_size=config.get("min_parallel_size", 100),
                enable_monitoring=config.get("enable_monitoring", True),
            )
        else:
            self.config = config

        self.process_pool = None
        self.memory_manager = None
        self.cuda_optimizer = None

        # 성능 통계
        self.optimization_stats = {
            "total_optimizations": 0,
            "strategy_usage": {},
            "performance_gains": {},
            "avg_speedup": 0.0,
        }

        self._initialize_components()

    def _initialize_components(self):
        """구성 요소 초기화"""
        try:
            # ProcessPool 초기화
            process_config = ProcessPoolConfig(
                max_workers=min(4, psutil.cpu_count()),
                enable_monitoring=self.config.enable_monitoring,
            )
            self.process_pool = ProcessPoolManager(process_config)
            logger.info("ProcessPool 초기화 완료")

            # MemoryManager 초기화
            memory_config = MemoryConfig(
                max_memory_usage=self.config.memory_threshold, use_memory_pooling=True
            )
            self.memory_manager = MemoryManager(memory_config)
            logger.info("MemoryManager 초기화 완료")

            # CUDA 최적화 초기화 (선택적)
            try:
                cuda_config = CudaConfig(use_amp=True, use_cudnn=True)
                self.cuda_optimizer = CudaOptimizer(cuda_config)
                logger.info("CUDA 최적화 초기화 완료")
            except Exception as e:
                logger.info(f"CUDA 최적화 초기화 실패 (GPU 없음): {e}")
                self.cuda_optimizer = None

        except Exception as e:
            logger.error(f"하이브리드 최적화 구성 요소 초기화 실패: {e}")
            raise

    def determine_optimal_level(
        self, data_size: int, available_memory: float, gpu_available: bool = False
    ) -> str:
        """데이터 크기와 시스템 리소스를 기반으로 최적 분석 레벨 결정"""
        try:
            # GPU 사용 가능하고 대용량 데이터인 경우
            if gpu_available and data_size > 5000 and available_memory > 0.7:
                return "maximum"

            # 중간 규모 데이터이고 메모리가 충분한 경우
            elif data_size > 1000 and available_memory > 0.5:
                return "balanced"

            # 소규모 데이터이거나 메모리가 부족한 경우
            elif data_size > 500 or available_memory > 0.3:
                return "basic"

            # 기본 처리
            else:
                return "standard"

        except Exception as e:
            logger.warning(f"최적화 레벨 결정 중 오류: {e}, 기본값 사용")
            return "standard"

    def _analyze_system_resources(self) -> Dict[str, float]:
        """시스템 리소스 분석"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # 메모리 사용률
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent

            # GPU 사용률 (가능한 경우)
            gpu_percent = 0.0
            if self.cuda_optimizer and self.cuda_optimizer.is_available():
                try:
                    gpu_percent = self.cuda_optimizer.get_gpu_utilization()
                except:
                    gpu_percent = 0.0

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "gpu_percent": gpu_percent,
            }
        except Exception as e:
            logger.error(f"시스템 리소스 분석 실패: {e}")
            return {"cpu_percent": 0.0, "memory_percent": 0.0, "gpu_percent": 0.0}

    def _select_optimization_strategy(
        self, task_info: TaskInfo, system_resources: Dict[str, float]
    ) -> OptimizationStrategy:
        """최적화 전략 선택"""

        if not self.config.auto_optimization:
            return OptimizationStrategy.AUTO

        # GPU 가속 조건 확인
        if (
            task_info.gpu_compatible
            and self.cuda_optimizer
            and system_resources["gpu_percent"] < self.config.gpu_threshold
        ):
            return OptimizationStrategy.GPU_ACCELERATED

        # 메모리 집약적 작업 확인
        if (
            task_info.memory_intensive
            or system_resources["memory_percent"] > self.config.memory_threshold
        ):
            return OptimizationStrategy.MEMORY_OPTIMIZED

        # 병렬 처리 조건 확인
        if (
            task_info.parallelizable
            and task_info.data_size >= self.config.min_parallel_size
            and system_resources["cpu_percent"] < self.config.cpu_threshold
        ):
            return OptimizationStrategy.CPU_PARALLEL

        # 하이브리드 전략 (기본)
        return OptimizationStrategy.HYBRID

    def optimize_execution(
        self,
        func: Callable,
        data: Any,
        task_info: TaskInfo,
        strategy: Optional[OptimizationStrategy] = None,
        **kwargs,
    ) -> Any:
        """
        최적화된 실행

        Args:
            func: 실행할 함수
            data: 입력 데이터
            task_info: 작업 정보
            strategy: 강제 전략 (None이면 자동 선택)
            **kwargs: 함수에 전달할 추가 인자

        Returns:
            실행 결과
        """
        start_time = time.time()

        # 시스템 리소스 분석
        system_resources = self._analyze_system_resources()

        # 전략 선택
        if strategy is None:
            strategy = self._select_optimization_strategy(task_info, system_resources)

        logger.info(f"최적화 전략 선택: {strategy.value}")

        try:
            # 전략별 실행
            if strategy == OptimizationStrategy.GPU_ACCELERATED:
                result = self._execute_gpu_accelerated(func, data, task_info, **kwargs)
            elif strategy == OptimizationStrategy.CPU_PARALLEL:
                result = self._execute_cpu_parallel(func, data, task_info, **kwargs)
            elif strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
                result = self._execute_memory_optimized(func, data, task_info, **kwargs)
            elif strategy == OptimizationStrategy.HYBRID:
                result = self._execute_hybrid(func, data, task_info, **kwargs)
            else:
                result = self._execute_standard(func, data, **kwargs)

            # 성능 통계 업데이트
            execution_time = time.time() - start_time
            self._update_performance_stats(strategy, execution_time)

            return result

        except Exception as e:
            logger.error(f"최적화 실행 실패 ({strategy.value}): {e}")
            # 폴백: 표준 실행
            return self._execute_standard(func, data, **kwargs)

    def _execute_gpu_accelerated(
        self, func: Callable, data: Any, task_info: TaskInfo, **kwargs
    ) -> Any:
        """GPU 가속 실행"""
        if not self.cuda_optimizer:
            return self._execute_standard(func, data, **kwargs)

        with self.cuda_optimizer.gpu_context():
            return func(data, **kwargs)

    def _execute_cpu_parallel(
        self, func: Callable, data: Any, task_info: TaskInfo, **kwargs
    ) -> Any:
        """CPU 병렬 실행"""
        if not self.process_pool:
            return self._execute_standard(func, data, **kwargs)

        # 데이터를 청크로 분할
        if isinstance(data, list) and len(data) > self.config.min_parallel_size:
            chunk_size = max(1, len(data) // self.process_pool.config.max_workers)
            chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

            results = self.process_pool.execute_parallel(func, chunks, **kwargs)

            # 결과 병합
            if results and isinstance(results[0], list):
                merged_result = []
                for result in results:
                    if result:
                        merged_result.extend(result)
                return merged_result
            else:
                return results
        else:
            return self._execute_standard(func, data, **kwargs)

    def _execute_memory_optimized(
        self, func: Callable, data: Any, task_info: TaskInfo, **kwargs
    ) -> Any:
        """메모리 최적화 실행"""
        if not self.memory_manager:
            return self._execute_standard(func, data, **kwargs)

        with self.memory_manager.memory_context():
            return func(data, **kwargs)

    def _execute_hybrid(
        self, func: Callable, data: Any, task_info: TaskInfo, **kwargs
    ) -> Any:
        """하이브리드 실행"""
        # 메모리 최적화 + 병렬 처리 조합
        if self.memory_manager and self.process_pool:
            with self.memory_manager.memory_context():
                return self._execute_cpu_parallel(func, data, task_info, **kwargs)
        else:
            return self._execute_standard(func, data, **kwargs)

    def _execute_standard(self, func: Callable, data: Any, **kwargs) -> Any:
        """표준 실행"""
        return func(data, **kwargs)

    def _update_performance_stats(
        self, strategy: OptimizationStrategy, execution_time: float
    ):
        """성능 통계 업데이트"""
        self.optimization_stats["total_optimizations"] += 1

        strategy_name = strategy.value
        if strategy_name not in self.optimization_stats["strategy_usage"]:
            self.optimization_stats["strategy_usage"][strategy_name] = 0
        self.optimization_stats["strategy_usage"][strategy_name] += 1

        # 성능 향상 추적 (기준 시간 대비)
        if strategy_name not in self.optimization_stats["performance_gains"]:
            self.optimization_stats["performance_gains"][strategy_name] = []
        self.optimization_stats["performance_gains"][strategy_name].append(
            execution_time
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        summary = self.optimization_stats.copy()

        # 평균 성능 계산
        for strategy, times in summary["performance_gains"].items():
            if times:
                summary["performance_gains"][strategy] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "count": len(times),
                }

        return summary

    def shutdown(self):
        """하이브리드 최적화 시스템 종료"""
        if self.process_pool:
            self.process_pool.shutdown()
        if self.memory_manager:
            self.memory_manager.cleanup()
        if self.cuda_optimizer:
            self.cuda_optimizer.cleanup()

        logger.info("하이브리드 최적화 시스템 종료 완료")

    def cleanup(self):
        """리소스 정리 (shutdown의 별칭)"""
        self.shutdown()


# 전역 하이브리드 최적화 인스턴스
_global_hybrid_optimizer = None


def get_hybrid_optimizer(config: Optional[HybridConfig] = None) -> HybridOptimizer:
    """전역 하이브리드 최적화 관리자 반환"""
    global _global_hybrid_optimizer

    if _global_hybrid_optimizer is None:
        if config is None:
            config = HybridConfig()
        _global_hybrid_optimizer = HybridOptimizer(config)

    return _global_hybrid_optimizer


def optimize(task_info: Dict[str, Any]):
    """
    최적화 데코레이터

    Args:
        task_info: 작업 정보 딕셔너리
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # TaskInfo 객체 생성
            info = TaskInfo(
                function_type=task_info.get("function_type", "unknown"),
                parallelizable=task_info.get("parallelizable", True),
                gpu_compatible=task_info.get("gpu_compatible", False),
                memory_intensive=task_info.get("memory_intensive", False),
                cpu_intensive=task_info.get("cpu_intensive", True),
            )

            # 데이터 크기 추정
            if args:
                if isinstance(args[0], (list, tuple)):
                    info.data_size = len(args[0])
                elif hasattr(args[0], "__len__"):
                    info.data_size = len(args[0])

            # 하이브리드 최적화 실행
            optimizer = get_hybrid_optimizer()
            return optimizer.optimize_execution(
                func, args[0] if args else None, info, **kwargs
            )

        return wrapper

    return decorator


def cleanup_hybrid_optimizer():
    """전역 하이브리드 최적화 시스템 정리"""
    global _global_hybrid_optimizer
    if _global_hybrid_optimizer:
        _global_hybrid_optimizer.shutdown()
        _global_hybrid_optimizer = None
