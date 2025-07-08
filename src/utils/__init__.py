"""
`utils` 패키지 초기화 파일
자주 사용되는 유틸리티 함수 및 클래스를 쉽게 임포트할 수 있도록 설정합니다.
"""

# Logging
from .unified_logging import get_logger, setup_logging

# Performance and Memory
from .unified_performance_engine import (
    UnifiedPerformanceEngine,
    AutoPerformanceMonitor,
    get_unified_performance_engine,
    get_auto_performance_monitor
)
from .unified_memory_manager import (
    UnifiedMemoryManager,
    get_unified_memory_manager
)

# Configuration
from .unified_config import Config, Paths, get_config, get_paths

# GPU Acceleration
from .cuda_optimizers import CudaOptimizer, CudaConfig, get_cuda_optimizer
from .gpu_accelerated_kernels import GPUAcceleratedKernels
from .gpu_accelerated_utils import (
    is_gpu_available,
    get_gpu_memory_info,
    get_gpu_device_name,
)

# Caching
from .cache_manager import UnifiedCachePathManager, CacheManager

# Dependency Injection
from .dependency_injection import configure_dependencies, resolve

# Other Core Utilities
from .data_loader import DataLoader
from .model_saver import ModelSaver
from .normalizer import Normalizer
from .pattern_filter import PatternFilter
from .vector_exporter import VectorExporter
from .config_validator import ConfigValidator
from .auto_recovery_system import AutoRecoverySystem
from .enhanced_process_pool import (
    EnhancedProcessPool,
    DynamicBatchSizeController,
    get_enhanced_process_pool,
    get_dynamic_batch_controller
)
from .unified_feature_vector_validator import UnifiedFeatureVectorValidator


__all__ = [
    # Logging
    "get_logger",
    "setup_logging",
    # Performance and Memory
    "UnifiedPerformanceEngine",
    "AutoPerformanceMonitor",
    "get_unified_performance_engine",
    "get_auto_performance_monitor",
    "UnifiedMemoryManager",
    "get_unified_memory_manager",
    # Configuration
    "Config",
    "Paths",
    "get_config",
    "get_paths",
    # GPU Acceleration
    "CudaOptimizer",
    "CudaConfig",
    "get_cuda_optimizer",
    "GPUAcceleratedKernels",
    "is_gpu_available",
    "get_gpu_memory_info",
    "get_gpu_device_name",
    # Caching
    "UnifiedCachePathManager",
    "CacheManager",
    # DI
    "configure_dependencies",
    "resolve",
    # Other Core Utilities
    "DataLoader",
    "ModelSaver",
    "Normalizer",
    "PatternFilter",
    "VectorExporter",
    "ConfigValidator",
    "AutoRecoverySystem",
    "EnhancedProcessPool",
    "DynamicBatchSizeController",
    "get_enhanced_process_pool",
    "get_dynamic_batch_controller",
    "UnifiedFeatureVectorValidator",
]
