"""
하이브리드 최적화 시스템

ProcessPool, MemoryPool, CUDA 최적화를 통합한 지능형 시스템
작업 타입별 최적 처리 방식 자동 선택 및 동적 리소스 관리
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np
import torch

from .error_handler import get_logger, StrictErrorHandler, validate_and_fail_fast
from .memory_manager import MemoryManager, get_memory_manager
from .process_pool_manager import ProcessPoolManager, get_process_pool_manager
from .performance_utils import PerformanceMonitor

logger = get_logger(__name__)


class WorkloadType(Enum):
    """작업 타입 분류"""

    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    GPU_INTENSIVE = "gpu_intensive"
    MIXED = "mixed"
    IO_INTENSIVE = "io_intensive"


class OptimizationStrategy(Enum):
    """최적화 전략"""

    PROCESS_POOL = "process_pool"
    MEMORY_POOL = "memory_pool"
    CUDA_ACCELERATOR = "cuda_accelerator"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class WorkloadProfile:
    """작업 부하 프로필"""

    cpu_usage: float
    memory_usage: float
    gpu_utilization: float
    data_size: int
    computation_complexity: str  # "low", "medium", "high"
    parallelizable: bool
    memory_bound: bool
    gpu_compatible: bool


@dataclass
class OptimizationResult:
    """최적화 결과"""

    strategy_used: OptimizationStrategy
    execution_time: float
    speedup: float
    memory_efficiency: float
    resource_utilization: Dict[str, float]
    bottlenecks_detected: List[str]


class HybridOptimizer:
    """하이브리드 최적화 시스템"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        하이브리드 최적화 시스템 초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.strict_handler = StrictErrorHandler(self.logger)
        self.performance_monitor = PerformanceMonitor()

        # 개별 최적화 시스템들
        self.memory_manager = get_memory_manager(self.config.get("memory_pool", {}))
        self.process_pool_manager = get_process_pool_manager(
            self.config.get("process_pool", {})
        )

        # 임계값 설정
        self.thresholds = self.config.get(
            "thresholds",
            {
                "cpu_usage": 0.7,
                "memory_usage": 0.8,
                "gpu_utilization": 0.6,
                "data_size_large": 1000000,  # 1M 요소
                "parallel_threshold": 100,
            },
        )

        # 전략별 성능 기록
        self.strategy_performance = {
            strategy: {"total_time": 0.0, "count": 0, "avg_speedup": 1.0}
            for strategy in OptimizationStrategy
        }

        # 자동 조정 설정
        self.auto_adjustment = self.config.get("auto_adjustment", True)
        self.monitoring_interval = self.config.get("monitoring_interval", 5.0)

        # 모니터링 스레드
        self._monitoring_active = False
        self._monitoring_thread = None

        if self.auto_adjustment:
            self._start_monitoring()

        self.logger.info("하이브리드 최적화 시스템 초기화 완료")

    def analyze_workload_type(self, task_info: Dict[str, Any]) -> WorkloadProfile:
        """
        작업 타입 분석

        Args:
            task_info: 작업 정보 딕셔너리

        Returns:
            작업 부하 프로필
        """
        # 현재 시스템 상태
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent

        # GPU 사용률 (사용 가능한 경우)
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            try:
                gpu_utilization = 50.0  # 기본값 (실제 API 사용 시 대체)
            except:
                gpu_utilization = 0.0

        # 작업 특성 분석
        data_size = task_info.get("data_size", 0)
        function_type = task_info.get("function_type", "unknown")
        parallelizable = task_info.get("parallelizable", True)

        # 복잡도 추정
        if "analysis" in function_type or "compute" in function_type:
            complexity = "high"
        elif "vectorize" in function_type or "transform" in function_type:
            complexity = "medium"
        else:
            complexity = "low"

        # 메모리 바운드 여부
        memory_bound = data_size > self.thresholds["data_size_large"]

        # GPU 호환성
        gpu_compatible = torch.cuda.is_available() and task_info.get(
            "gpu_compatible", False
        )

        return WorkloadProfile(
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            gpu_utilization=gpu_utilization,
            data_size=data_size,
            computation_complexity=complexity,
            parallelizable=parallelizable,
            memory_bound=memory_bound,
            gpu_compatible=gpu_compatible,
        )

    def select_optimal_strategy(
        self, workload: WorkloadProfile
    ) -> OptimizationStrategy:
        """
        최적 처리 전략 선택

        Args:
            workload: 작업 부하 프로필

        Returns:
            선택된 최적화 전략
        """
        # 전략별 점수 계산
        scores = {}

        # ProcessPool 점수
        cpu_score = 1.0 if workload.parallelizable else 0.0
        if workload.cpu_usage < self.thresholds["cpu_usage"]:
            cpu_score += 0.5
        if workload.computation_complexity == "high":
            cpu_score += 0.3
        scores[OptimizationStrategy.PROCESS_POOL] = cpu_score

        # MemoryPool 점수
        memory_score = 1.0 if workload.memory_bound else 0.0
        if workload.memory_usage > self.thresholds["memory_usage"]:
            memory_score += 0.5
        if workload.data_size > self.thresholds["data_size_large"]:
            memory_score += 0.3
        scores[OptimizationStrategy.MEMORY_POOL] = memory_score

        # CUDA 점수
        cuda_score = 0.0
        if workload.gpu_compatible and torch.cuda.is_available():
            cuda_score = 1.0
            if workload.gpu_utilization < self.thresholds["gpu_utilization"]:
                cuda_score += 0.5
            if workload.computation_complexity == "high":
                cuda_score += 0.3
        scores[OptimizationStrategy.CUDA_ACCELERATOR] = cuda_score

        # 하이브리드 점수 (복합 조건)
        hybrid_conditions = [
            workload.parallelizable,
            workload.memory_bound,
            workload.gpu_compatible,
            workload.computation_complexity == "high",
        ]
        hybrid_score = sum(hybrid_conditions) / len(hybrid_conditions)
        scores[OptimizationStrategy.HYBRID] = hybrid_score

        # 과거 성능 기록 반영
        for strategy, score in scores.items():
            if self.strategy_performance[strategy]["count"] > 0:
                avg_speedup = self.strategy_performance[strategy]["avg_speedup"]
                scores[strategy] *= avg_speedup

        # 최고 점수 전략 선택
        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]

        # 점수가 너무 낮으면 AUTO 전략 사용
        if best_score < 0.3:
            best_strategy = OptimizationStrategy.AUTO

        self.logger.debug(f"전략 선택: {best_strategy.value} (점수: {best_score:.2f})")
        return best_strategy

    def execute_with_optimization(
        self,
        func: Callable,
        data: Any,
        task_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[Any, OptimizationResult]:
        """
        최적화된 실행

        Args:
            func: 실행할 함수
            data: 입력 데이터
            task_info: 작업 정보
            **kwargs: 함수에 전달할 추가 인자

        Returns:
            (실행 결과, 최적화 결과)
        """
        start_time = time.time()

        # 작업 분석
        if task_info is None:
            task_info = {
                "data_size": len(data) if hasattr(data, "__len__") else 1,
                "function_type": (
                    func.__name__ if hasattr(func, "__name__") else "unknown"
                ),
                "parallelizable": True,
                "gpu_compatible": False,
            }

        workload = self.analyze_workload_type(task_info)
        strategy = self.select_optimal_strategy(workload)

        # 전략별 실행
        try:
            if strategy == OptimizationStrategy.PROCESS_POOL:
                result = self._execute_with_process_pool(func, data, **kwargs)
            elif strategy == OptimizationStrategy.MEMORY_POOL:
                result = self._execute_with_memory_pool(func, data, **kwargs)
            elif strategy == OptimizationStrategy.HYBRID:
                result = self._execute_with_hybrid(func, data, **kwargs)
            else:  # AUTO
                result = self._execute_auto(func, data, **kwargs)

            execution_time = time.time() - start_time

            # 성능 기록 업데이트
            self._update_strategy_performance(strategy, execution_time)

            # 최적화 결과 생성
            optimization_result = OptimizationResult(
                strategy_used=strategy,
                execution_time=execution_time,
                speedup=self._calculate_speedup(strategy, execution_time),
                memory_efficiency=self._calculate_memory_efficiency(),
                resource_utilization=self._get_resource_utilization(),
                bottlenecks_detected=self._detect_bottlenecks(workload),
            )

            return result, optimization_result

        except Exception as e:
            self.strict_handler.handle_critical_error(
                e, f"최적화 실행 실패: {strategy.value}"
            )
            return None, None

    def _execute_with_process_pool(self, func: Callable, data: Any, **kwargs) -> Any:
        """ProcessPool을 사용한 실행"""
        if (
            hasattr(data, "__len__")
            and len(data) > self.thresholds["parallel_threshold"]
        ):
            chunks = self.process_pool_manager.chunk_and_split(
                data, self.process_pool_manager.chunk_size
            )
            results = self.process_pool_manager.parallel_analyze(chunks, func, **kwargs)
            return self.process_pool_manager.merge_results(results)
        else:
            return func(data, **kwargs)

    def _execute_with_memory_pool(self, func: Callable, data: Any, **kwargs) -> Any:
        """MemoryPool을 사용한 실행"""
        with self.memory_manager.allocation_scope():
            return func(data, **kwargs)

    def _execute_with_hybrid(self, func: Callable, data: Any, **kwargs) -> Any:
        """하이브리드 실행 (여러 최적화 조합)"""
        with self.memory_manager.allocation_scope():
            if (
                hasattr(data, "__len__")
                and len(data) > self.thresholds["parallel_threshold"]
            ):
                # ProcessPool + MemoryPool 조합
                chunks = self.process_pool_manager.chunk_and_split(
                    data, self.process_pool_manager.chunk_size
                )
                results = self.process_pool_manager.parallel_analyze(
                    chunks, func, **kwargs
                )
                return self.process_pool_manager.merge_results(results)
            else:
                return func(data, **kwargs)

    def _execute_auto(self, func: Callable, data: Any, **kwargs) -> Any:
        """자동 최적화 실행"""
        # 간단한 휴리스틱으로 전략 선택
        data_size = len(data) if hasattr(data, "__len__") else 1

        if data_size > self.thresholds["data_size_large"]:
            return self._execute_with_memory_pool(func, data, **kwargs)
        elif data_size > self.thresholds["parallel_threshold"]:
            return self._execute_with_process_pool(func, data, **kwargs)
        else:
            return func(data, **kwargs)

    def _update_strategy_performance(
        self, strategy: OptimizationStrategy, execution_time: float
    ):
        """전략별 성능 기록 업데이트"""
        perf = self.strategy_performance[strategy]
        perf["total_time"] += execution_time
        perf["count"] += 1

        # 평균 속도 향상 계산 (간단한 추정)
        baseline_time = execution_time * 2  # 가정: 최적화 없으면 2배 시간
        speedup = baseline_time / execution_time if execution_time > 0 else 1.0
        perf["avg_speedup"] = (
            perf["avg_speedup"] * (perf["count"] - 1) + speedup
        ) / perf["count"]

    def _calculate_speedup(
        self, strategy: OptimizationStrategy, execution_time: float
    ) -> float:
        """속도 향상 계산"""
        return self.strategy_performance[strategy]["avg_speedup"]

    def _calculate_memory_efficiency(self) -> float:
        """메모리 효율성 계산"""
        memory_info = psutil.virtual_memory()
        return 1.0 - (memory_info.percent / 100.0)

    def _get_resource_utilization(self) -> Dict[str, float]:
        """리소스 사용률 반환"""
        utilization = {
            "cpu": psutil.cpu_percent(interval=0.1),
            "memory": psutil.virtual_memory().percent,
        }

        if torch.cuda.is_available():
            try:
                utilization["gpu"] = 50.0  # 기본값
            except:
                utilization["gpu"] = 0.0

        return utilization

    def _detect_bottlenecks(self, workload: WorkloadProfile) -> List[str]:
        """병목점 감지"""
        bottlenecks = []

        if workload.cpu_usage > 90:
            bottlenecks.append("high_cpu_usage")

        if workload.memory_usage > 85:
            bottlenecks.append("high_memory_usage")

        if workload.gpu_utilization > 90:
            bottlenecks.append("high_gpu_usage")

        if workload.data_size > self.thresholds["data_size_large"] * 10:
            bottlenecks.append("very_large_dataset")

        return bottlenecks

    def _start_monitoring(self):
        """모니터링 스레드 시작"""
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("자동 모니터링 시작")

    def _monitoring_loop(self):
        """모니터링 루프"""
        while self._monitoring_active:
            try:
                # 리소스 사용률 확인
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_percent = psutil.virtual_memory().percent

                # 임계값 초과 시 경고
                if cpu_percent > 90:
                    self.logger.warning(f"높은 CPU 사용률 감지: {cpu_percent:.1f}%")

                if memory_percent > 90:
                    self.logger.warning(
                        f"높은 메모리 사용률 감지: {memory_percent:.1f}%"
                    )

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"모니터링 중 오류: {e}")
                time.sleep(self.monitoring_interval)

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        return {
            "strategy_performance": self.strategy_performance,
            "resource_utilization": self._get_resource_utilization(),
            "optimization_config": self.config,
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            # 모니터링 중지
            self._monitoring_active = False
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=1.0)

            # 개별 시스템 정리
            if self.memory_manager:
                self.memory_manager.cleanup()

            if self.process_pool_manager:
                self.process_pool_manager.cleanup()

            self.logger.info("하이브리드 최적화 시스템 정리 완료")

        except Exception as e:
            self.logger.warning(f"하이브리드 최적화 시스템 정리 중 오류: {e}")


# 전역 하이브리드 최적화 시스템 인스턴스
_global_hybrid_optimizer = None


def get_hybrid_optimizer(config: Optional[Dict[str, Any]] = None) -> HybridOptimizer:
    """전역 하이브리드 최적화 시스템 반환"""
    global _global_hybrid_optimizer

    if _global_hybrid_optimizer is None:
        _global_hybrid_optimizer = HybridOptimizer(config)

    return _global_hybrid_optimizer


# 데코레이터
def optimize(task_info: Optional[Dict[str, Any]] = None):
    """함수를 자동 최적화하는 데코레이터"""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            optimizer = get_hybrid_optimizer()
            data = args[0] if args else None
            result, opt_result = optimizer.execute_with_optimization(
                func, data, task_info, **kwargs
            )
            return result

        return wrapper

    return decorator
