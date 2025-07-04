"""
GPU 우선순위 시스템 성능 벤치마크 모듈

DAEBAK_AI Utils 모듈의 GPU 최적화 성능을 측정하고 검증하는 시스템
"""

import time
import asyncio
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import psutil
from concurrent.futures import ThreadPoolExecutor
import gc

from .unified_logging import get_logger
from .performance_optimizer import (
    get_smart_computation_engine,
    smart_compute,
    optimize_computation,
)
from .memory_manager import get_memory_manager
from .normalizer import get_gpu_normalizer, smart_normalize
from .async_io import get_gpu_async_io_manager, smart_read_file, smart_write_file

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """벤치마크 결과 데이터 클래스"""

    test_name: str
    gpu_time: Optional[float] = None
    cpu_time: Optional[float] = None
    multithread_time: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    cpu_memory_used: Optional[float] = None
    throughput_gpu: Optional[float] = None
    throughput_cpu: Optional[float] = None
    speedup_ratio: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

    def calculate_speedup(self):
        """속도 향상 비율 계산"""
        if self.gpu_time and self.cpu_time and self.gpu_time > 0:
            self.speedup_ratio = self.cpu_time / self.gpu_time
        elif self.multithread_time and self.cpu_time and self.multithread_time > 0:
            self.speedup_ratio = self.cpu_time / self.multithread_time


class GPUBenchmarkSuite:
    """GPU 우선순위 시스템 벤치마크 스위트"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.computation_engine = get_smart_computation_engine()
        self.memory_manager = get_memory_manager()
        self.gpu_normalizer = get_gpu_normalizer()
        self.gpu_io_manager = get_gpu_async_io_manager()

        self.gpu_available = torch.cuda.is_available()
        self.results: List[BenchmarkResult] = []

        if self.gpu_available:
            self.logger.info("GPU 벤치마크 스위트 초기화 완료")
        else:
            self.logger.warning("GPU 없음 - CPU 성능만 측정")

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """모든 벤치마크 실행"""
        self.logger.info("🚀 GPU 우선순위 시스템 벤치마크 시작")

        # 1. 스마트 컴퓨테이션 벤치마크
        self._benchmark_smart_computation()

        # 2. 메모리 관리 벤치마크
        self._benchmark_memory_management()

        # 3. 정규화 벤치마크
        self._benchmark_normalization()

        # 4. 비동기 I/O 벤치마크
        asyncio.run(self._benchmark_async_io())

        # 5. 통합 성능 벤치마크
        self._benchmark_integrated_performance()

        # 결과 분석 및 리포트 생성
        return self._generate_performance_report()

    def _benchmark_smart_computation(self):
        """스마트 컴퓨테이션 벤치마크"""
        self.logger.info("📊 스마트 컴퓨테이션 벤치마크")

        # 테스트 데이터 생성
        test_sizes = [1000, 10000, 100000, 1000000]

        for size in test_sizes:
            result = BenchmarkResult(f"smart_computation_{size}")

            try:
                # 테스트 데이터
                data = np.random.randn(size).astype(np.float32)

                def computation_task(arr):
                    if isinstance(arr, torch.Tensor):
                        return torch.mean(arr**2 + torch.sin(arr))
                    else:
                        return np.mean(arr**2 + np.sin(arr))

                # GPU/멀티쓰레드/CPU 성능 측정
                if self.gpu_available:
                    start_time = time.time()
                    gpu_result = smart_compute(
                        computation_task, data, operation_type="computation"
                    )
                    result.gpu_time = time.time() - start_time
                    result.gpu_memory_used = self.memory_manager.get_memory_usage("gpu")

                # CPU 성능 측정
                start_time = time.time()
                cpu_result = computation_task(data)
                result.cpu_time = time.time() - start_time
                result.cpu_memory_used = self.memory_manager.get_memory_usage("cpu")

                # 처리량 계산
                if result.gpu_time:
                    result.throughput_gpu = size / result.gpu_time
                if result.cpu_time:
                    result.throughput_cpu = size / result.cpu_time

                result.calculate_speedup()

                self.logger.info(
                    f"데이터 크기 {size}: GPU {result.gpu_time:.3f}s, CPU {result.cpu_time:.3f}s, 속도향상 {result.speedup_ratio:.2f}x"
                )

            except Exception as e:
                result.success = False
                result.error_message = str(e)
                self.logger.error(f"스마트 컴퓨테이션 벤치마크 실패: {e}")

            self.results.append(result)

    def _benchmark_memory_management(self):
        """메모리 관리 벤치마크"""
        self.logger.info("💾 메모리 관리 벤치마크")

        tensor_sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]

        for size in tensor_sizes:
            result = BenchmarkResult(f"memory_management_{size[0]}x{size[1]}")

            try:
                # 스마트 메모리 할당 테스트
                start_time = time.time()

                if self.gpu_available:
                    # GPU 메모리 할당
                    tensor = self.memory_manager.smart_memory_allocation(
                        size, prefer_gpu=True
                    )
                    result.gpu_time = time.time() - start_time
                    result.gpu_memory_used = self.memory_manager.get_memory_usage("gpu")

                    # 메모리 정리
                    del tensor
                    torch.cuda.empty_cache()

                # CPU 메모리 할당
                start_time = time.time()
                cpu_tensor = self.memory_manager.allocate_cpu_memory(size)
                result.cpu_time = time.time() - start_time
                result.cpu_memory_used = self.memory_manager.get_memory_usage("cpu")

                result.calculate_speedup()

                del cpu_tensor
                gc.collect()

                self.logger.info(
                    f"메모리 할당 {size}: GPU {result.gpu_time:.3f}s, CPU {result.cpu_time:.3f}s"
                )

            except Exception as e:
                result.success = False
                result.error_message = str(e)
                self.logger.error(f"메모리 관리 벤치마크 실패: {e}")

            self.results.append(result)

    def _benchmark_normalization(self):
        """정규화 벤치마크"""
        self.logger.info("📏 정규화 벤치마크")

        data_sizes = [10000, 100000, 1000000]
        methods = ["zscore", "minmax", "robust"]

        for size in data_sizes:
            for method in methods:
                result = BenchmarkResult(f"normalization_{method}_{size}")

                try:
                    # 테스트 데이터 생성
                    data = np.random.randn(size, 10).astype(np.float32)

                    # GPU 정규화
                    if self.gpu_available:
                        start_time = time.time()
                        gpu_normalized = smart_normalize(data, method=method)
                        result.gpu_time = time.time() - start_time
                        result.gpu_memory_used = self.memory_manager.get_memory_usage(
                            "gpu"
                        )

                    # CPU 정규화 (기존 방식)
                    start_time = time.time()
                    if method == "zscore":
                        cpu_normalized = (data - np.mean(data, axis=0)) / np.std(
                            data, axis=0
                        )
                    elif method == "minmax":
                        min_val = np.min(data, axis=0)
                        max_val = np.max(data, axis=0)
                        cpu_normalized = (data - min_val) / (max_val - min_val)
                    else:  # robust
                        median = np.median(data, axis=0)
                        mad = np.median(np.abs(data - median), axis=0)
                        cpu_normalized = (data - median) / mad

                    result.cpu_time = time.time() - start_time
                    result.cpu_memory_used = self.memory_manager.get_memory_usage("cpu")

                    # 처리량 계산
                    if result.gpu_time:
                        result.throughput_gpu = size / result.gpu_time
                    if result.cpu_time:
                        result.throughput_cpu = size / result.cpu_time

                    result.calculate_speedup()

                    self.logger.info(
                        f"정규화 {method} {size}: GPU {result.gpu_time:.3f}s, CPU {result.cpu_time:.3f}s, 속도향상 {result.speedup_ratio:.2f}x"
                    )

                except Exception as e:
                    result.success = False
                    result.error_message = str(e)
                    self.logger.error(f"정규화 벤치마크 실패: {e}")

                self.results.append(result)

    async def _benchmark_async_io(self):
        """비동기 I/O 벤치마크"""
        self.logger.info("💽 비동기 I/O 벤치마크")

        # 테스트 파일 크기
        file_sizes = [
            1024 * 1024,
            10 * 1024 * 1024,
            100 * 1024 * 1024,
        ]  # 1MB, 10MB, 100MB
        test_dir = Path("data/benchmark_temp")
        test_dir.mkdir(parents=True, exist_ok=True)

        for size in file_sizes:
            result = BenchmarkResult(f"async_io_{size//1024//1024}MB")

            try:
                # 테스트 데이터 생성
                test_data = np.random.randint(0, 256, size, dtype=np.uint8).tobytes()
                test_file = test_dir / f"test_{size}.bin"

                # GPU 우선순위 I/O
                if self.gpu_available:
                    start_time = time.time()
                    await smart_write_file(test_file, test_data)
                    read_data = await smart_read_file(test_file, load_to_gpu=True)
                    result.gpu_time = time.time() - start_time
                    result.gpu_memory_used = self.memory_manager.get_memory_usage("gpu")

                # 일반 I/O
                start_time = time.time()
                with open(test_file, "wb") as f:
                    f.write(test_data)
                with open(test_file, "rb") as f:
                    cpu_data = f.read()
                result.cpu_time = time.time() - start_time
                result.cpu_memory_used = self.memory_manager.get_memory_usage("cpu")

                # 처리량 계산 (MB/s)
                size_mb = size / (1024 * 1024)
                if result.gpu_time:
                    result.throughput_gpu = size_mb / result.gpu_time
                if result.cpu_time:
                    result.throughput_cpu = size_mb / result.cpu_time

                result.calculate_speedup()

                self.logger.info(
                    f"I/O {size_mb:.1f}MB: GPU {result.gpu_time:.3f}s, CPU {result.cpu_time:.3f}s, 처리량 GPU {result.throughput_gpu:.1f}MB/s"
                )

                # 테스트 파일 정리
                test_file.unlink(missing_ok=True)

            except Exception as e:
                result.success = False
                result.error_message = str(e)
                self.logger.error(f"비동기 I/O 벤치마크 실패: {e}")

            self.results.append(result)

        # 테스트 디렉토리 정리
        try:
            test_dir.rmdir()
        except:
            pass

    def _benchmark_integrated_performance(self):
        """통합 성능 벤치마크"""
        self.logger.info("🔄 통합 성능 벤치마크")

        result = BenchmarkResult("integrated_performance")

        try:
            # 복합 작업 시뮬레이션
            data_size = 50000
            data = np.random.randn(data_size, 20).astype(np.float32)

            # GPU 통합 작업
            if self.gpu_available:
                start_time = time.time()

                # 1. 메모리 할당
                gpu_tensor = self.memory_manager.smart_memory_allocation(
                    (data_size, 20), prefer_gpu=True
                )

                # 2. 데이터 정규화
                normalized_data = smart_normalize(data, method="zscore")

                # 3. 연산 처리
                def complex_computation(arr):
                    if isinstance(arr, torch.Tensor):
                        return torch.mean(arr**2) + torch.std(arr) * torch.sum(
                            torch.sin(arr)
                        )
                    else:
                        return np.mean(arr**2) + np.std(arr) * np.sum(np.sin(arr))

                gpu_result = smart_compute(complex_computation, normalized_data)

                result.gpu_time = time.time() - start_time
                result.gpu_memory_used = self.memory_manager.get_memory_usage("gpu")

                # 메모리 정리
                del gpu_tensor
                torch.cuda.empty_cache()

            # CPU 통합 작업
            start_time = time.time()

            # 1. 메모리 할당
            cpu_array = np.zeros((data_size, 20), dtype=np.float32)

            # 2. 데이터 정규화
            normalized_cpu = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

            # 3. 연산 처리
            cpu_result = np.mean(normalized_cpu**2) + np.std(normalized_cpu) * np.sum(
                np.sin(normalized_cpu)
            )

            result.cpu_time = time.time() - start_time
            result.cpu_memory_used = self.memory_manager.get_memory_usage("cpu")

            result.calculate_speedup()

            self.logger.info(
                f"통합 성능: GPU {result.gpu_time:.3f}s, CPU {result.cpu_time:.3f}s, 속도향상 {result.speedup_ratio:.2f}x"
            )

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"통합 성능 벤치마크 실패: {e}")

        self.results.append(result)

    def _generate_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        self.logger.info("📋 성능 리포트 생성")

        # 성공한 결과만 필터링
        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            return {"error": "모든 벤치마크 실패"}

        # 통계 계산
        gpu_times = [r.gpu_time for r in successful_results if r.gpu_time is not None]
        cpu_times = [r.cpu_time for r in successful_results if r.cpu_time is not None]
        speedup_ratios = [
            r.speedup_ratio for r in successful_results if r.speedup_ratio is not None
        ]

        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": len(successful_results),
                "failed_tests": len(self.results) - len(successful_results),
                "gpu_available": self.gpu_available,
            },
            "performance_metrics": {
                "average_gpu_time": np.mean(gpu_times) if gpu_times else None,
                "average_cpu_time": np.mean(cpu_times) if cpu_times else None,
                "average_speedup": np.mean(speedup_ratios) if speedup_ratios else None,
                "max_speedup": np.max(speedup_ratios) if speedup_ratios else None,
                "min_speedup": np.min(speedup_ratios) if speedup_ratios else None,
            },
            "system_info": {
                "gpu_count": torch.cuda.device_count() if self.gpu_available else 0,
                "gpu_name": (
                    torch.cuda.get_device_name(0) if self.gpu_available else None
                ),
                "gpu_memory_total": (
                    torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if self.gpu_available
                    else None
                ),
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "gpu_time": r.gpu_time,
                    "cpu_time": r.cpu_time,
                    "speedup_ratio": r.speedup_ratio,
                    "throughput_gpu": r.throughput_gpu,
                    "throughput_cpu": r.throughput_cpu,
                    "success": r.success,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations(successful_results),
        }

        # 리포트 저장
        self._save_report(report)

        return report

    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """성능 개선 권장사항 생성"""
        recommendations = []

        # 속도 향상 분석
        speedup_ratios = [
            r.speedup_ratio for r in results if r.speedup_ratio is not None
        ]

        if speedup_ratios:
            avg_speedup = np.mean(speedup_ratios)

            if avg_speedup > 3.0:
                recommendations.append(
                    "✅ GPU 가속이 매우 효과적입니다. 현재 설정을 유지하세요."
                )
            elif avg_speedup > 1.5:
                recommendations.append(
                    "⚡ GPU 가속이 효과적입니다. 더 큰 데이터셋에서 더 나은 성능을 기대할 수 있습니다."
                )
            elif avg_speedup > 1.0:
                recommendations.append(
                    "🔧 GPU 가속 효과가 제한적입니다. 메모리 설정을 조정해보세요."
                )
            else:
                recommendations.append(
                    "⚠️ GPU 성능이 CPU보다 낮습니다. 시스템 설정을 확인하세요."
                )

        # GPU 메모리 사용률 분석
        gpu_memory_usage = [
            r.gpu_memory_used for r in results if r.gpu_memory_used is not None
        ]
        if gpu_memory_usage:
            avg_gpu_memory = np.mean(gpu_memory_usage)

            if avg_gpu_memory > 0.9:
                recommendations.append(
                    "🚨 GPU 메모리 사용률이 높습니다. 배치 크기를 줄이거나 메모리 정리를 늘리세요."
                )
            elif avg_gpu_memory > 0.7:
                recommendations.append(
                    "⚡ GPU 메모리 사용률이 적당합니다. 성능과 안정성의 균형이 좋습니다."
                )
            else:
                recommendations.append(
                    "💡 GPU 메모리 여유가 있습니다. 더 큰 배치 크기를 사용할 수 있습니다."
                )

        # 실패한 테스트 분석
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            recommendations.append(
                f"❌ {len(failed_tests)}개 테스트가 실패했습니다. 시스템 안정성을 확인하세요."
            )

        return recommendations

    def _save_report(self, report: Dict[str, Any]):
        """리포트 저장"""
        try:
            report_dir = Path("data/result/performance_reports")
            report_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"gpu_benchmark_report_{timestamp}.json"

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"성능 리포트 저장: {report_file}")

        except Exception as e:
            self.logger.error(f"리포트 저장 실패: {e}")


def run_gpu_benchmark() -> Dict[str, Any]:
    """GPU 벤치마크 실행 (편의 함수)"""
    benchmark_suite = GPUBenchmarkSuite()
    return benchmark_suite.run_all_benchmarks()


def quick_performance_check() -> Dict[str, Any]:
    """빠른 성능 체크"""
    logger.info("🔍 빠른 성능 체크 시작")

    try:
        # 기본 시스템 정보
        system_info = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        }

        if system_info["gpu_available"]:
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
            system_info["gpu_memory_gb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024**3)

        # 간단한 성능 테스트
        test_data = np.random.randn(10000, 10).astype(np.float32)

        # GPU 테스트
        gpu_time = None
        if system_info["gpu_available"]:
            start_time = time.time()
            gpu_result = smart_compute(
                lambda x: (
                    torch.mean(x**2) if isinstance(x, torch.Tensor) else np.mean(x**2)
                ),
                test_data,
            )
            gpu_time = time.time() - start_time

        # CPU 테스트
        start_time = time.time()
        cpu_result = np.mean(test_data**2)
        cpu_time = time.time() - start_time

        performance_info = {
            "gpu_time": gpu_time,
            "cpu_time": cpu_time,
            "speedup": gpu_time and (cpu_time / gpu_time),
        }

        result = {
            "system_info": system_info,
            "performance_info": performance_info,
            "status": "success",
        }

        logger.info(
            f"성능 체크 완료: GPU 가속 {performance_info['speedup']:.2f}x"
            if performance_info["speedup"]
            else "성능 체크 완료 (GPU 없음)"
        )

        return result

    except Exception as e:
        logger.error(f"성능 체크 실패: {e}")
        return {"status": "error", "error": str(e)}
