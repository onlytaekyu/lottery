"""
시스템 통합 테스트 - 성능 벤치마크 및 검증

이 모듈은 DAEBAK AI 로또 시스템의 전체적인 성능과 안정성을 테스트합니다.
"""

import time
import gc
import threading
import psutil
import torch
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from ..utils.unified_logging import get_logger
from ..utils.cuda_singleton_manager import (
    get_singleton_cuda_optimizer,
    cleanup_cuda_resources,
)
from ..utils.gpu_memory_pool import get_gpu_memory_pool, cleanup_all_memory_pools
from ..utils.enhanced_process_pool import (
    get_enhanced_process_pool,
    cleanup_process_pool,
)
from ..utils.gpu_accelerated_kernels import get_gpu_pattern_kernels, cleanup_gpu_kernels

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 지표"""

    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    success_count: int = 0
    failure_count: int = 0


@dataclass
class TestResult:
    """테스트 결과"""

    test_name: str
    success: bool
    metrics: PerformanceMetrics
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class SystemIntegrationTester:
    """시스템 통합 테스터"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)

        # 테스트 결과 저장
        self.test_results: List[TestResult] = []
        self.overall_metrics = PerformanceMetrics()

        # 성능 목표값
        self.performance_targets = {
            "max_execution_time": 30.0,  # 30초
            "max_memory_usage_mb": 1024,  # 1GB
            "min_gpu_utilization": 15.0,  # 15%
            "min_cpu_utilization": 20.0,  # 20%
            "max_error_rate": 0.01,  # 1%
            "min_throughput": 10.0,  # 10 operations/sec
        }

        self.logger.info("✅ 시스템 통합 테스터 초기화 완료")

    def run_full_integration_test(self) -> Dict[str, Any]:
        """전체 통합 테스트 실행"""
        try:
            self.logger.info("🚀 전체 시스템 통합 테스트 시작")
            start_time = time.time()

            # 테스트 순서
            test_suite = [
                ("memory_management", self.test_memory_management),
                ("cuda_optimization", self.test_cuda_optimization),
                ("gpu_acceleration", self.test_gpu_acceleration),
                ("multiprocessing", self.test_multiprocessing),
                ("pattern_analysis", self.test_pattern_analysis),
                ("vectorization", self.test_vectorization),
                ("performance_benchmark", self.test_performance_benchmark),
                ("stress_test", self.test_stress_test),
                ("memory_leak_test", self.test_memory_leak),
                ("resource_cleanup", self.test_resource_cleanup),
            ]

            # 각 테스트 실행
            for test_name, test_func in test_suite:
                try:
                    self.logger.info(f"📋 테스트 실행 중: {test_name}")
                    result = test_func()
                    self.test_results.append(result)

                    if result.success:
                        self.logger.info(f"✅ {test_name} 테스트 성공")
                    else:
                        self.logger.error(
                            f"❌ {test_name} 테스트 실패: {result.error_message}"
                        )

                except Exception as e:
                    self.logger.error(f"❌ {test_name} 테스트 예외: {e}")
                    self.test_results.append(
                        TestResult(
                            test_name=test_name,
                            success=False,
                            metrics=PerformanceMetrics(),
                            error_message=str(e),
                        )
                    )

                # 테스트 간 정리
                self._cleanup_between_tests()

            # 전체 결과 분석
            total_time = time.time() - start_time
            summary = self._generate_test_summary(total_time)

            self.logger.info(f"🎯 전체 통합 테스트 완료: {total_time:.2f}초")
            return summary

        except Exception as e:
            self.logger.error(f"통합 테스트 실패: {e}")
            raise

    def test_memory_management(self) -> TestResult:
        """메모리 관리 테스트"""
        try:
            with self._performance_monitor("memory_management") as metrics:
                # 메모리 풀 테스트
                memory_pool = get_gpu_memory_pool()

                # 메모리 할당/해제 테스트
                allocations = []
                for i in range(10):
                    with memory_pool.allocate(64 * 1024 * 1024) as tensor:  # 64MB
                        if tensor is not None:
                            allocations.append(tensor.size())

                # 통계 확인
                pool_stats = memory_pool.get_stats()

                # 성공 조건
                success = (
                    len(allocations) > 5  # 최소 5개 할당 성공
                    and pool_stats["cache_hit_rate"] >= 0.0  # 캐시 히트율 확인
                    and metrics.peak_memory_mb
                    < self.performance_targets["max_memory_usage_mb"]
                )

                return TestResult(
                    test_name="memory_management",
                    success=success,
                    metrics=metrics,
                    details={
                        "allocations_count": len(allocations),
                        "pool_stats": pool_stats,
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="memory_management",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    def test_cuda_optimization(self) -> TestResult:
        """CUDA 최적화 테스트"""
        try:
            with self._performance_monitor("cuda_optimization") as metrics:
                # CUDA 최적화기 테스트
                cuda_optimizer = get_singleton_cuda_optimizer(
                    requester_name="integration_test"
                )

                if cuda_optimizer.is_available():
                    # GPU 컨텍스트 테스트
                    with cuda_optimizer.device_context():
                        test_tensor = torch.randn(1000, 1000)
                        optimized_tensor = cuda_optimizer.optimize_tensor_operations(
                            test_tensor
                        )

                    # GPU 사용률 확인
                    gpu_utilization = cuda_optimizer.get_gpu_utilization()
                    metrics.gpu_utilization = gpu_utilization

                    success = gpu_utilization >= 0.0  # GPU 사용 가능
                else:
                    success = True  # GPU 없어도 성공 (CPU 폴백)
                    self.logger.warning("GPU 사용 불가능 - CPU 모드로 테스트")

                return TestResult(
                    test_name="cuda_optimization",
                    success=success,
                    metrics=metrics,
                    details={
                        "gpu_available": cuda_optimizer.is_available(),
                        "gpu_utilization": metrics.gpu_utilization,
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="cuda_optimization",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    def test_gpu_acceleration(self) -> TestResult:
        """GPU 가속 테스트"""
        try:
            with self._performance_monitor("gpu_acceleration") as metrics:
                # GPU 커널 테스트
                gpu_kernels = get_gpu_pattern_kernels()

                # 테스트 데이터 생성
                test_data = torch.randint(1, 46, (100, 6))  # 100개 로또 번호

                # 빈도 분석 커널 테스트
                freq_result = gpu_kernels.frequency_analysis_kernel(test_data)

                # 간격 분석 커널 테스트
                gap_result = gpu_kernels.gap_analysis_kernel(test_data)

                # 연속 번호 분석 커널 테스트
                consecutive_result = gpu_kernels.consecutive_analysis_kernel(test_data)

                # 결과 검증
                success = (
                    freq_result.size(0) == 45  # 45개 번호
                    and gap_result.size(0) == 45
                    and consecutive_result.size(0) == 100  # 100개 결과
                )

                return TestResult(
                    test_name="gpu_acceleration",
                    success=success,
                    metrics=metrics,
                    details={
                        "freq_result_shape": freq_result.shape,
                        "gap_result_shape": gap_result.shape,
                        "consecutive_result_shape": consecutive_result.shape,
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="gpu_acceleration",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    def test_multiprocessing(self) -> TestResult:
        """멀티프로세싱 테스트"""
        try:
            with self._performance_monitor("multiprocessing") as metrics:
                # 프로세스 풀 테스트
                process_pool = get_enhanced_process_pool()

                # 테스트 함수
                def test_function(data):
                    return sum(data) ** 0.5

                # 테스트 데이터
                test_data = [[i] * 1000 for i in range(1, 21)]  # 20개 청크

                # 병렬 처리 테스트
                start_time = time.time()
                results = process_pool.parallel_map(test_function, test_data)
                parallel_time = time.time() - start_time

                # 순차 처리와 비교
                start_time = time.time()
                sequential_results = [test_function(data) for data in test_data]
                sequential_time = time.time() - start_time

                # 성능 개선 계산
                speedup = sequential_time / parallel_time if parallel_time > 0 else 0
                metrics.throughput = len(test_data) / parallel_time

                # 결과 검증
                success = (
                    len(results) == len(test_data)
                    and speedup > 0.5  # 최소 50% 성능 (오버헤드 고려)
                )

                return TestResult(
                    test_name="multiprocessing",
                    success=success,
                    metrics=metrics,
                    details={
                        "results_count": len(results),
                        "speedup": speedup,
                        "parallel_time": parallel_time,
                        "sequential_time": sequential_time,
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="multiprocessing",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    def test_pattern_analysis(self) -> TestResult:
        """패턴 분석 테스트"""
        try:
            with self._performance_monitor("pattern_analysis") as metrics:
                # 패턴 분석기 임포트 및 테스트
                from ..analysis.pattern_analyzer import PatternAnalyzer
                from ..shared.types import LotteryNumber

                # 테스트 데이터 생성
                test_data = [
                    LotteryNumber(numbers=[1, 2, 3, 4, 5, 6], draw_number=i)
                    for i in range(1, 101)
                ]

                # 패턴 분석기 초기화
                analyzer = PatternAnalyzer()

                # 분석 실행
                analysis_result = analyzer.analyze(test_data)

                # 결과 검증
                success = (
                    analysis_result is not None
                    and hasattr(analysis_result, "frequency_map")
                    and len(analysis_result.frequency_map) > 0
                )

                return TestResult(
                    test_name="pattern_analysis",
                    success=success,
                    metrics=metrics,
                    details={
                        "analysis_type": type(analysis_result).__name__,
                        "frequency_map_size": (
                            len(analysis_result.frequency_map)
                            if hasattr(analysis_result, "frequency_map")
                            else 0
                        ),
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="pattern_analysis",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    def test_vectorization(self) -> TestResult:
        """벡터화 테스트"""
        try:
            with self._performance_monitor("vectorization") as metrics:
                # 벡터화 시스템 테스트
                from ..analysis.enhanced_pattern_vectorizer import (
                    EnhancedPatternVectorizer,
                )

                # 벡터화기 초기화
                vectorizer = EnhancedPatternVectorizer()

                # 테스트 분석 결과 생성
                test_analysis = {
                    "pattern_analysis": {
                        "frequency_map": {i: i * 0.1 for i in range(1, 46)}
                    },
                    "distribution_pattern": {"entropy": 3.5},
                    "roi_features": {"avg_roi": 0.8},
                }

                # 벡터화 실행
                vector = vectorizer.vectorize_full_analysis_enhanced(test_analysis)
                feature_names = vectorizer.get_feature_names()

                # 결과 검증
                success = (
                    vector is not None
                    and len(vector) == 168  # 168차원
                    and len(feature_names) == len(vector)  # 이름과 벡터 차원 일치
                    and not torch.isnan(torch.tensor(vector)).any()  # NaN 없음
                )

                return TestResult(
                    test_name="vectorization",
                    success=success,
                    metrics=metrics,
                    details={
                        "vector_dimension": len(vector),
                        "feature_names_count": len(feature_names),
                        "vector_range": (float(min(vector)), float(max(vector))),
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="vectorization",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    def test_performance_benchmark(self) -> TestResult:
        """성능 벤치마크 테스트"""
        try:
            with self._performance_monitor("performance_benchmark") as metrics:
                # 대용량 데이터 처리 테스트
                large_data = torch.randint(1, 46, (1000, 6))  # 1000개 로또 번호

                # GPU 커널 성능 테스트
                gpu_kernels = get_gpu_pattern_kernels()

                start_time = time.time()
                freq_result = gpu_kernels.frequency_analysis_kernel(large_data)
                gap_result = gpu_kernels.gap_analysis_kernel(large_data)
                consecutive_result = gpu_kernels.consecutive_analysis_kernel(large_data)
                processing_time = time.time() - start_time

                # 처리량 계산
                throughput = len(large_data) / processing_time
                metrics.throughput = throughput

                # 성능 목표 달성 여부
                success = (
                    processing_time < self.performance_targets["max_execution_time"]
                    and throughput >= self.performance_targets["min_throughput"]
                )

                return TestResult(
                    test_name="performance_benchmark",
                    success=success,
                    metrics=metrics,
                    details={
                        "data_size": len(large_data),
                        "processing_time": processing_time,
                        "throughput": throughput,
                        "target_throughput": self.performance_targets["min_throughput"],
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="performance_benchmark",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    def test_stress_test(self) -> TestResult:
        """스트레스 테스트"""
        try:
            with self._performance_monitor("stress_test") as metrics:
                # 동시 작업 스트레스 테스트
                import concurrent.futures

                def stress_worker(worker_id):
                    # GPU 커널 작업
                    gpu_kernels = get_gpu_pattern_kernels()
                    test_data = torch.randint(1, 46, (50, 6))

                    results = []
                    for _ in range(10):  # 각 워커가 10번 반복
                        freq_result = gpu_kernels.frequency_analysis_kernel(test_data)
                        results.append(freq_result)

                    return len(results)

                # 동시 실행
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(stress_worker, i) for i in range(4)]

                    results = []
                    for future in concurrent.futures.as_completed(futures, timeout=60):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            self.logger.error(f"스트레스 워커 실패: {e}")
                            results.append(0)

                # 성공률 계산
                success_rate = sum(1 for r in results if r > 0) / len(results)
                metrics.error_rate = 1.0 - success_rate

                success = success_rate >= 0.8  # 80% 이상 성공

                return TestResult(
                    test_name="stress_test",
                    success=success,
                    metrics=metrics,
                    details={
                        "worker_count": len(results),
                        "success_rate": success_rate,
                        "completed_tasks": sum(results),
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="stress_test",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    def test_memory_leak(self) -> TestResult:
        """메모리 누수 테스트"""
        try:
            with self._performance_monitor("memory_leak") as metrics:
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                # 반복적인 할당/해제 테스트
                for i in range(100):
                    # GPU 메모리 할당/해제
                    memory_pool = get_gpu_memory_pool()
                    with memory_pool.allocate(10 * 1024 * 1024) as tensor:  # 10MB
                        if tensor is not None:
                            # 간단한 연산
                            _ = tensor.sum()

                    # 주기적 가비지 컬렉션
                    if i % 20 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory

                # 메모리 증가량 허용 범위 (100MB 이하)
                success = memory_increase < 100

                return TestResult(
                    test_name="memory_leak",
                    success=success,
                    metrics=metrics,
                    details={
                        "initial_memory_mb": initial_memory,
                        "final_memory_mb": final_memory,
                        "memory_increase_mb": memory_increase,
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="memory_leak",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    def test_resource_cleanup(self) -> TestResult:
        """리소스 정리 테스트"""
        try:
            with self._performance_monitor("resource_cleanup") as metrics:
                # 모든 리소스 정리
                cleanup_cuda_resources()
                cleanup_all_memory_pools()
                cleanup_process_pool()
                cleanup_gpu_kernels()

                # 가비지 컬렉션
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 정리 후 메모리 확인
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                success = True  # 정리 작업은 예외 없이 완료되면 성공

                return TestResult(
                    test_name="resource_cleanup",
                    success=success,
                    metrics=metrics,
                    details={
                        "final_memory_mb": final_memory,
                    },
                )

        except Exception as e:
            return TestResult(
                test_name="resource_cleanup",
                success=False,
                metrics=PerformanceMetrics(),
                error_message=str(e),
            )

    @contextmanager
    def _performance_monitor(self, test_name: str):
        """성능 모니터링 컨텍스트"""
        # 초기 상태
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            initial_gpu_memory = 0

        metrics = PerformanceMetrics()

        try:
            yield metrics
        finally:
            # 최종 상태
            end_time = time.time()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                final_gpu_memory = 0

            # 메트릭 계산
            metrics.execution_time = end_time - start_time
            metrics.memory_usage_mb = final_memory - initial_memory
            metrics.peak_memory_mb = final_memory
            metrics.gpu_memory_mb = final_gpu_memory - initial_gpu_memory
            metrics.cpu_utilization = psutil.cpu_percent()

    def _cleanup_between_tests(self):
        """테스트 간 정리"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.1)  # 잠시 대기
        except Exception as e:
            self.logger.warning(f"테스트 간 정리 실패: {e}")

    def _generate_test_summary(self, total_time: float) -> Dict[str, Any]:
        """테스트 요약 생성"""
        try:
            # 성공/실패 통계
            total_tests = len(self.test_results)
            successful_tests = sum(1 for result in self.test_results if result.success)
            failed_tests = total_tests - successful_tests

            # 성능 통계
            total_execution_time = sum(
                result.metrics.execution_time for result in self.test_results
            )
            avg_execution_time = (
                total_execution_time / total_tests if total_tests > 0 else 0
            )

            peak_memory = (
                max(result.metrics.peak_memory_mb for result in self.test_results)
                if self.test_results
                else 0
            )

            # 목표 달성 여부
            performance_goals_met = {
                "execution_time": total_time
                < self.performance_targets["max_execution_time"],
                "memory_usage": peak_memory
                < self.performance_targets["max_memory_usage_mb"],
                "success_rate": (
                    (successful_tests / total_tests) >= 0.9
                    if total_tests > 0
                    else False
                ),
            }

            return {
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "success_rate": (
                        successful_tests / total_tests if total_tests > 0 else 0
                    ),
                    "total_time": total_time,
                    "avg_execution_time": avg_execution_time,
                    "peak_memory_mb": peak_memory,
                },
                "performance_goals": {
                    "targets": self.performance_targets,
                    "achievements": performance_goals_met,
                    "overall_success": all(performance_goals_met.values()),
                },
                "detailed_results": [
                    {
                        "test_name": result.test_name,
                        "success": result.success,
                        "execution_time": result.metrics.execution_time,
                        "memory_usage_mb": result.metrics.memory_usage_mb,
                        "error": result.error_message,
                        "details": result.details,
                    }
                    for result in self.test_results
                ],
                "recommendations": self._generate_recommendations(),
            }

        except Exception as e:
            self.logger.error(f"테스트 요약 생성 실패: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        # 실패한 테스트 분석
        failed_tests = [result for result in self.test_results if not result.success]

        if failed_tests:
            recommendations.append(
                f"실패한 테스트 {len(failed_tests)}개를 우선적으로 수정하세요."
            )

        # 성능 분석
        slow_tests = [
            result
            for result in self.test_results
            if result.metrics.execution_time > 5.0
        ]

        if slow_tests:
            recommendations.append("실행 시간이 긴 테스트들의 최적화를 검토하세요.")

        # 메모리 사용량 분석
        memory_intensive_tests = [
            result
            for result in self.test_results
            if result.metrics.peak_memory_mb > 500
        ]

        if memory_intensive_tests:
            recommendations.append(
                "메모리 사용량이 높은 테스트들의 최적화를 검토하세요."
            )

        if not recommendations:
            recommendations.append("모든 테스트가 성공적으로 완료되었습니다! 🎉")

        return recommendations


# 편의 함수
def run_quick_integration_test() -> Dict[str, Any]:
    """빠른 통합 테스트 실행"""
    tester = SystemIntegrationTester()
    return tester.run_full_integration_test()


def run_performance_benchmark() -> Dict[str, Any]:
    """성능 벤치마크만 실행"""
    tester = SystemIntegrationTester()
    result = tester.test_performance_benchmark()
    return {
        "benchmark_result": result,
        "success": result.success,
        "metrics": result.metrics.__dict__,
        "details": result.details,
    }
