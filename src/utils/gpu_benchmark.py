"""
GPU 우선순위 시스템 성능 벤치마크 모듈

DAEBAK_AI Utils 모듈의 GPU 최적화 성능을 측정하고 검증하는 시스템
"""

import time
import torch
from typing import Dict, Any, Optional

from .unified_logging import get_logger
from .unified_memory_manager import get_unified_memory_manager
from .cuda_optimizers import get_cuda_optimizer
from .unified_performance_engine import SystemMonitor

logger = get_logger(__name__)

class GPUBenchmark:
    """
    GPU 성능 벤치마킹 도구.
    메모리 대역폭, 행렬 연산 속도 등 다양한 GPU 성능 지표를 측정합니다.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cuda_optimizer = None
        self.memory_manager = None
        self.system_monitor = None

        if torch.cuda.is_available():
            self.cuda_optimizer = get_cuda_optimizer()
            self.memory_manager = get_unified_memory_manager()
            self.system_monitor = SystemMonitor()
        else:
            logger.warning("GPU가 없어 GPUBenchmark를 실행할 수 없습니다.")

    def run_all_benchmarks(self) -> Optional[Dict[str, Any]]:
        if not torch.cuda.is_available() or not self.system_monitor:
            return None

        self.logger.info("GPU 벤치마크 시작...")
        results = {
            "system_info": self.system_monitor.get_current_state().__dict__,
            "matrix_multiplication": self._benchmark_matrix_multiplication(),
            "memory_bandwidth": self._benchmark_memory_bandwidth(),
        }
        self.logger.info("GPU 벤치마크 완료.")
        return results

    def _benchmark_matrix_multiplication(self) -> Dict[str, float]:
        sizes = self.config.get("matrix_sizes", [256, 1024, 4096])
        results = {}
        for size in sizes:
            a = torch.randn(size, size, device=self.device)
            b = torch.randn(size, size, device=self.device)
            
            start_time = time.time()
            for _ in range(10):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            duration = (time.time() - start_time) / 10
            
            gflops = (2 * size**3) / (duration * 1e9)
            results[f"size_{size}x{size}_gflops"] = round(gflops, 2)
            
        return results

    def _benchmark_memory_bandwidth(self) -> Dict[str, float]:
        sizes_gb = self.config.get("memory_sizes_gb", [0.5, 1, 2])
        results = {}
        for size_gb in sizes_gb:
            tensor_size = int(size_gb * (1024**3) / 4) # float32
            tensor = torch.randn(tensor_size, device=self.device)
            
            start_time = time.time()
            for _ in range(10):
                _ = tensor.clone()
            torch.cuda.synchronize()
            duration = (time.time() - start_time) / 10
            
            bandwidth_gb_s = (size_gb / duration)
            results[f"size_{size_gb}GB_bandwidth_GBs"] = round(bandwidth_gb_s, 2)
            
        return results

    def get_report(self) -> Optional[str]:
        results = self.run_all_benchmarks()
        if not results:
            return "GPU 벤치마크를 실행할 수 없습니다."

        report = "GPU Benchmark Report\n"
        report += "="*20 + "\n"
        for key, value in results.items():
            report += f"{key}:\n"
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    report += f"  {sub_key}: {sub_value}\n"
            else:
                report += f"  {value}\n"
        return report

def run_gpu_benchmark(config: Optional[Dict[str, Any]] = None) -> None:
    benchmark = GPUBenchmark(config)
    report = benchmark.get_report()
    if report:
        logger.info(report)
