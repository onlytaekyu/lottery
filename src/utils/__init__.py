"""
DAEBAK_AI 로또 예측 시스템 - Utils 모듈 (최종 성능 최적화 버전)

핵심 성능 기능에 집중한 간소화된 utils 모듈입니다.
GPU 최우선, 메모리 효율성, 실행 속도 최적화 완료.
"""

from pathlib import Path

# 핵심 시스템 (필수)
from .unified_logging import get_logger
from .unified_config import load_config
from .cache_paths import CACHE_DIR, get_cache_dir

# CUDA 최적화 (GPU 최우선)
from .cuda_optimizers import (
    CUDAOptimizer,
    AMPTrainer,
    get_cuda_optimizer,
    optimize_memory,
)

# 메모리 관리 (핵심)
from .memory_manager import MemoryManager, get_memory_manager, memory_managed_analysis

# 성능 최적화 (핵심)
from .performance_optimizer import (
    PerformanceProfiler,
    GPUOptimizer,
    MemoryOptimizer,
    MultiThreadOptimizer,
)

# 모델 통합 (간소화)
from .model_integrator import ModelIntegrator, IntegratorConfig

# 캐시 관리 (필수)
from .cache_manager import get_cache_manager

# 비동기 I/O (성능)
from .async_io import AsyncIOManager

# 프로세스 풀 (병렬 처리)
from .process_pool_manager import ProcessPoolManager


# 지연 로딩을 위한 함수들 (메모리 효율성)
def get_performance_profiler():
    """성능 프로파일러 반환"""
    return PerformanceProfiler()


def get_gpu_optimizer():
    """GPU 최적화기 반환"""
    return GPUOptimizer()


def get_memory_optimizer():
    """메모리 최적화기 반환"""
    return MemoryOptimizer()


def get_async_io_manager():
    """비동기 I/O 매니저 반환"""
    return AsyncIOManager()


def get_process_pool_manager():
    """프로세스 풀 매니저 반환"""
    return ProcessPoolManager()


def get_model_integrator(config=None):
    """모델 통합기 반환"""
    return ModelIntegrator(config)


# 편의 함수들
def init_utils():
    """Utils 모듈 초기화"""
    logger = get_logger(__name__)
    logger.info("DAEBAK_AI Utils 모듈 초기화 완료 (성능 최적화 버전)")


def cleanup_utils():
    """Utils 모듈 정리"""
    try:
        # 메모리 관리자 정리
        memory_manager = get_memory_manager()
        memory_manager.cleanup()

        # CUDA 메모리 정리
        optimize_memory()

        logger = get_logger(__name__)
        logger.info("Utils 모듈 정리 완료")
    except Exception as e:
        print(f"Utils 모듈 정리 중 오류: {e}")


# 성능 모니터링 함수
def get_system_status():
    """시스템 상태 반환"""
    try:
        memory_manager = get_memory_manager()
        cuda_optimizer = get_cuda_optimizer()

        return {
            "memory_info": memory_manager.get_memory_info(),
            "cuda_info": cuda_optimizer.get_memory_info(),
            "status": "optimal",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# 모듈 초기화
init_utils()

__all__ = [
    # 핵심 시스템
    "get_logger",
    "load_config",
    "CACHE_DIR",
    "get_cache_dir",
    # CUDA 최적화
    "CUDAOptimizer",
    "AMPTrainer",
    "get_cuda_optimizer",
    "optimize_memory",
    # 메모리 관리
    "MemoryManager",
    "get_memory_manager",
    "memory_managed_analysis",
    # 성능 최적화
    "PerformanceProfiler",
    "GPUOptimizer",
    "MemoryOptimizer",
    "MultiThreadOptimizer",
    "get_performance_profiler",
    "get_gpu_optimizer",
    "get_memory_optimizer",
    # 모델 통합
    "ModelIntegrator",
    "IntegratorConfig",
    "get_model_integrator",
    # 캐시 관리
    "get_cache_manager",
    # 비동기 I/O
    "AsyncIOManager",
    "get_async_io_manager",
    # 프로세스 풀
    "ProcessPoolManager",
    "get_process_pool_manager",
    # 편의 함수
    "init_utils",
    "cleanup_utils",
    "get_system_status",
]
