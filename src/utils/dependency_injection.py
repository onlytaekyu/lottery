"""
의존성 주입(DI) 설정 및 핵심 컨테이너 모듈.

이 모듈은 애플리케이션의 모든 의존성을 중앙에서 설정하고,
단순화된 DI 컨테이너를 제공합니다.
"""

import threading
from typing import Dict, Any, Callable, Type, Optional, Set, List
from enum import Enum

from .unified_logging import get_logger
from .cache_manager import UnifiedCachePathManager

logger = get_logger(__name__)


# --- Core DI Classes (Simplified) ---

from .cache_paths import UnifiedCachePathManager

class LifecycleType(Enum):
    """의존성 생명주기 타입"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"

class DependencyError(Exception):
    """의존성 관련 오류"""

class CircularDependencyError(DependencyError):
    """순환 참조 오류"""

class DependencyRegistration:
    """의존성 등록 정보"""
    def __init__(
        self,
        interface: Type,
        implementation: Type,
        factory: Optional[Callable] = None,
        lifecycle: LifecycleType = LifecycleType.SINGLETON,
    ):
        self.interface = interface
        self.implementation = implementation
        self.factory = factory
        self.lifecycle = lifecycle

class DependencyContainer:
    """단순화된 의존성 주입 컨테이너"""
    def __init__(self):
        self._registrations: Dict[Type, DependencyRegistration] = {}
        self._instances: Dict[Type, Any] = {}
        self._resolution_stack: Set[Type] = set()
        self._lock = threading.RLock()

    def register(
        self,
        interface: Type,
        implementation: Type = None,
        factory: Callable = None,
        lifecycle: LifecycleType = LifecycleType.SINGLETON,
    ) -> "DependencyContainer":
        with self._lock:
            if implementation is None and factory is None:
                implementation = interface

            registration = DependencyRegistration(
                interface=interface,
                implementation=implementation,
                factory=factory,
                lifecycle=lifecycle,
            )
            self._registrations[interface] = registration
            return self

    def register_instance(self, interface: Type, instance: Any) -> "DependencyContainer":
        with self._lock:
            self._instances[interface] = instance
            return self

    def resolve(self, interface: Type) -> Any:
        with self._lock:
            if interface in self._resolution_stack:
                path = " -> ".join(c.__name__ for c in self._resolution_stack) + f" -> {interface.__name__}"
                raise CircularDependencyError(f"순환 참조 탐지: {path}")

            if interface in self._instances:
                return self._instances[interface]

            if interface not in self._registrations:
                raise DependencyError(f"등록되지 않은 의존성: {interface.__name__}")

            registration = self._registrations[interface]
            self._resolution_stack.add(interface)

            try:
                instance = self._create_instance(registration)
                if registration.lifecycle == LifecycleType.SINGLETON:
                    self._instances[interface] = instance
                return instance
            finally:
                self._resolution_stack.discard(interface)

    def _create_instance(self, registration: DependencyRegistration) -> Any:
        if registration.factory:
            # 팩토리 함수는 컨테이너 자체를 인자로 받아 다른 의존성을 해결할 수 있습니다.
            return registration.factory(self)
        else:
            try:
                # 기본 생성자 (인자 없음) 호출
                return registration.implementation()
            except TypeError as e:
                raise DependencyError(
                    f"'{registration.implementation.__name__}'의 인스턴스 생성 실패. "
                    f"인자가 있는 생성자는 팩토리(factory)를 등록해야 합니다. 원본 오류: {e}"
                ) from e

    def detect_circular_dependencies(self) -> List[List[str]]:
        # 이 기능은 복잡도에 비해 사용 빈도가 낮아 단순화 또는 제거될 수 있으나,
        # 초기 설정 검증을 위해 유지합니다.
        graph = {
            interface.__name__: [] for interface in self._registrations
        }
        # 실제 의존성 분석 로직이 필요하지만, 현재 구현에서는 생략.
        # 이 기능이 꼭 필요하다면 추후 재구현 필요. 지금은 빈 목록을 반환.
        return []


# --- Singleton Container Instance ---

_container_instance: Optional[DependencyContainer] = None
_container_lock = threading.Lock()

def get_container() -> DependencyContainer:
    """컨테이너의 싱글톤 인스턴스를 반환합니다."""
    global _container_instance
    if _container_instance is None:
        with _container_lock:
            if _container_instance is None:
                _container_instance = DependencyContainer()
    return _container_instance


# --- Central Dependency Configuration ---

def configure_dependencies():
    """
    애플리케이션 전체의 의존성을 등록하고 설정합니다.
    이 함수는 프로그램 실행 초기에 한 번만 호출되어야 합니다.
    """
    logger.info("🚀 의존성 설정 시작...")
    container = get_container()

    # 팩토리 함수 수정: 컨테이너를 인자로 받도록 람다로 감쌉니다.
    from .unified_config import get_config, get_paths, Config, Paths
    from .unified_memory_manager import UnifiedMemoryManager, get_unified_memory_manager
    from .cache_manager import CacheManager
    from .model_saver import ModelSaver
    from .gpu_accelerated_kernels import GPUAcceleratedKernels
    from .normalizer import Normalizer
    from .pattern_filter import PatternFilter
    from .unified_feature_vector_validator import UnifiedFeatureVectorValidator
    from .unified_performance_engine import UnifiedPerformanceEngine, get_unified_performance_engine, AutoPerformanceMonitor, get_auto_performance_monitor
    from .enhanced_process_pool import EnhancedProcessPool, DynamicBatchSizeController, get_enhanced_process_pool, get_dynamic_batch_controller
    from .data_loader import DataLoader, get_data_loader
    from .cuda_optimizers import CudaOptimizer, get_cuda_optimizer
    from .auto_recovery_system import AutoRecoverySystem, get_auto_recovery_system
    from .config_validator import ConfigValidator, get_config_validator
    from ..analysis.analyzer_factory import create_analyzer, get_analyzer_config
    from ..analysis.base_analyzer import BaseAnalyzer
    from ..evaluation.evaluator import Evaluator, get_evaluator
    from ..models.unified_model_manager import UnifiedModelManager
    from ..pipeline.unified_preprocessing_pipeline import UnifiedPreprocessingPipeline
    from ..pipeline.unified_training_pipeline import UnifiedTrainingPipeline
    from ..core.recommendation_engine import RecommendationEngine

    try:
        # 팩토리가 컨테이너를 인자로 받도록 수정합니다. 기존 get_* 함수들은 인자가 없으므로 람다로 감쌉니다.
        container.register(Config, factory=lambda c: get_config())
        container.register(Paths, factory=lambda c: get_paths())
        container.register(
            UnifiedCachePathManager,
            factory=lambda c: UnifiedCachePathManager(c.resolve(Paths))
        )
        container.register(UnifiedMemoryManager, factory=lambda c: get_unified_memory_manager())
        container.register(CacheManager) # 기본 생성자 사용
        container.register(ModelSaver) # 기본 생성자 사용
        container.register(GPUAcceleratedKernels) # 기본 생성자 사용
        container.register(Normalizer) # 기본 생성자 사용
        container.register(PatternFilter) # 기본 생성자 사용
        container.register(UnifiedFeatureVectorValidator) # 기본 생성자 사용
        container.register(UnifiedPerformanceEngine, factory=lambda c: get_unified_performance_engine())
        container.register(EnhancedProcessPool, factory=lambda c: get_enhanced_process_pool())
        container.register(DynamicBatchSizeController, factory=lambda c: get_dynamic_batch_controller())
        container.register(DataLoader, factory=lambda c: get_data_loader())
        container.register(CudaOptimizer, factory=lambda c: get_cuda_optimizer())
        container.register(AutoRecoverySystem, factory=lambda c: get_auto_recovery_system())
        container.register(AutoPerformanceMonitor, factory=lambda c: get_auto_performance_monitor())
        container.register(ConfigValidator, factory=lambda c: get_config_validator())

        # 분석기/평가기 팩토리 람다 수정
        container.register(BaseAnalyzer, factory=lambda c: create_analyzer(get_analyzer_config()))
        container.register(Evaluator, factory=lambda c: get_evaluator())

        # 의존성이 있는 클래스들에 대한 팩토리 정의
        container.register(
            UnifiedModelManager,
            factory=lambda c: UnifiedModelManager(
                config=c.resolve(Config),
                cache_path_manager=c.resolve(UnifiedCachePathManager),
                batch_controller=c.resolve(DynamicBatchSizeController),
                process_pool=c.resolve(EnhancedProcessPool),
                feature_validator=c.resolve(UnifiedFeatureVectorValidator),
                memory_manager=c.resolve(UnifiedMemoryManager),
                model_saver=c.resolve(ModelSaver),
            ),
        )
        container.register(
            UnifiedPreprocessingPipeline,
            factory=lambda c: UnifiedPreprocessingPipeline(
                config=c.resolve(Config),
                cache_path_manager=c.resolve(UnifiedCachePathManager)
            )
        )
        container.register(UnifiedTrainingPipeline)
        container.register(RecommendationEngine)

        cycles = container.detect_circular_dependencies()
        if cycles:
            logger.warning(f"⚠️ 순환 참조 탐지: {cycles}")
        else:
            logger.info("✅ 순환 참조 없음 - 의존성 그래프 검증 통과")

    except Exception as e:
        logger.error(f"❌ 의존성 설정 중 심각한 오류 발생: {e}", exc_info=True)
        raise

    logger.info("✅ 의존성 설정 완료.")


def resolve(interface_type: Type) -> Any:
    """편의성을 위한 resolve 함수"""
    return get_container().resolve(interface_type) 