"""
src/utils 통합 모듈 (v4 - MCP 최적화)

MCP 서버 활용으로 완전 최적화된 utils 모듈 통합 시스템.
GPU 우선순위 연산, 중앙집중식 메모리 관리, 자동 복구 시스템 완전 구현.
"""

from typing import Any, Dict, Optional

# 지연 로딩 모듈 캐시
_lazy_modules: Dict[str, Any] = {}

# === 핵심 성능 최적화 모듈 ===


def get_smart_computation_engine():
    """GPU > 멀티쓰레드 CPU > CPU 우선순위 연산 엔진"""
    if "computation_engine" not in _lazy_modules:
        from .performance_optimizer import get_smart_computation_engine

        _lazy_modules["computation_engine"] = get_smart_computation_engine()
    return _lazy_modules["computation_engine"]


def get_gpu_vector_exporter():
    """GPU 가속 벡터 내보내기 시스템"""
    if "vector_exporter" not in _lazy_modules:
        from .vector_exporter import get_gpu_vector_exporter

        _lazy_modules["vector_exporter"] = get_gpu_vector_exporter()
    return _lazy_modules["vector_exporter"]


def get_advanced_model_saver():
    """고급 모델 저장 시스템"""
    if "model_saver" not in _lazy_modules:
        from .model_saver import get_advanced_model_saver

        _lazy_modules["model_saver"] = get_advanced_model_saver()
    return _lazy_modules["model_saver"]


def get_gpu_normalizer():
    """GPU 가속 정규화 시스템"""
    if "normalizer" not in _lazy_modules:
        from .normalizer import get_gpu_normalizer

        _lazy_modules["normalizer"] = get_gpu_normalizer()
    return _lazy_modules["normalizer"]


def get_cache_manager():
    """자가 복구 캐시 관리자"""
    if "cache_manager" not in _lazy_modules:
        from .cache_manager import SelfHealingCacheManager

        _lazy_modules["cache_manager"] = SelfHealingCacheManager()
    return _lazy_modules["cache_manager"]


# === 통합 메모리 관리 시스템 ===


def get_memory_manager():
    """기본 GPU 메모리 관리자 (하위 호환성)"""
    if "memory_manager" not in _lazy_modules:
        from .memory_manager import OptimizedMemoryManager

        _lazy_modules["memory_manager"] = OptimizedMemoryManager()
    return _lazy_modules["memory_manager"]


def get_unified_memory_manager():
    """통합 메모리 관리자 (최신 시스템)"""
    if "unified_memory" not in _lazy_modules:
        from .unified_memory_manager import get_unified_memory_manager

        _lazy_modules["unified_memory"] = get_unified_memory_manager()
    return _lazy_modules["unified_memory"]


def get_unified_async_manager():
    """통합 비동기 관리자"""
    if "unified_async" not in _lazy_modules:
        from .unified_async_manager import get_unified_async_manager

        _lazy_modules["unified_async"] = get_unified_async_manager()
    return _lazy_modules["unified_async"]


def get_auto_recovery_system():
    """자동 복구 시스템"""
    if "auto_recovery" not in _lazy_modules:
        from .auto_recovery_system import get_auto_recovery_system

        _lazy_modules["auto_recovery"] = get_auto_recovery_system()
    return _lazy_modules["auto_recovery"]


# === 연산 전략 시스템 ===


def get_compute_executor():
    """최적 연산 실행기"""
    if "compute_executor" not in _lazy_modules:
        from .compute_strategy import get_compute_executor

        _lazy_modules["compute_executor"] = get_compute_executor()
    return _lazy_modules["compute_executor"]


def get_cuda_optimizer():
    """CUDA 최적화기"""
    if "cuda_optimizer" not in _lazy_modules:
        from .cuda_optimizers import get_cuda_optimizer

        _lazy_modules["cuda_optimizer"] = get_cuda_optimizer()
    return _lazy_modules["cuda_optimizer"]


def get_cuda_stream_manager():
    """CUDA 스트림 관리자"""
    if "cuda_stream_manager" not in _lazy_modules:
        from .performance_optimizer import get_cuda_stream_manager

        _lazy_modules["cuda_stream_manager"] = get_cuda_stream_manager()
    return _lazy_modules["cuda_stream_manager"]


# === 하위 호환성 별칭 ===

# 메모리 관리자 별칭 통합
get_optimized_memory_manager = get_memory_manager
get_gpu_memory_manager = get_memory_manager

# 비동기 관리자 별칭 통합
get_async_io_manager = get_unified_async_manager
get_gpu_async_io_manager = get_unified_async_manager

# 캐시 관리자 별칭 통합
get_self_healing_cache_manager = get_cache_manager
get_gpu_cache_manager = get_cache_manager


# 정규화 별칭 통합
def IndependentGPUNormalizer():
    """하위 호환성 클래스 래퍼"""
    from .normalizer import IndependentGPUNormalizer

    return IndependentGPUNormalizer()


get_independent_gpu_normalizer = get_gpu_normalizer

# === 전체 시스템 초기화 ===


def initialize_all_systems():
    """모든 시스템 초기화 (테스트용)"""
    systems = [
        "computation_engine",
        "unified_memory",
        "unified_async",
        "auto_recovery",
        "compute_executor",
        "cuda_optimizer",
    ]

    for system in systems:
        if system == "computation_engine":
            get_smart_computation_engine()
        elif system == "unified_memory":
            get_unified_memory_manager()
        elif system == "unified_async":
            get_unified_async_manager()
        elif system == "auto_recovery":
            get_auto_recovery_system()
        elif system == "compute_executor":
            get_compute_executor()
        elif system == "cuda_optimizer":
            get_cuda_optimizer()

    return len(_lazy_modules)


# === 공개 API ===

__all__ = [
    # 핵심 시스템
    "get_smart_computation_engine",
    "get_unified_memory_manager",
    "get_unified_async_manager",
    "get_auto_recovery_system",
    "get_compute_executor",
    "get_cuda_optimizer",
    "get_cuda_stream_manager",
    # 유틸리티
    "get_gpu_vector_exporter",
    "get_advanced_model_saver",
    "get_gpu_normalizer",
    "get_cache_manager",
    # 하위 호환성
    "get_memory_manager",
    "get_optimized_memory_manager",
    "get_gpu_memory_manager",
    "get_async_io_manager",
    "get_gpu_async_io_manager",
    "get_self_healing_cache_manager",
    "get_gpu_cache_manager",
    "IndependentGPUNormalizer",
    "get_independent_gpu_normalizer",
    # 초기화
    "initialize_all_systems",
]
