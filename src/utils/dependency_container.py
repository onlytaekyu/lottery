"""
완전한 순환 참조 해결을 위한 의존성 주입 컨테이너
"""

import threading
from typing import Dict, Any, Callable, Type, Optional, Set, List
from abc import ABC, abstractmethod
from enum import Enum

from .unified_logging import get_logger

logger = get_logger(__name__)


class LifecycleType(Enum):
    """의존성 생명주기 타입"""

    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class DependencyError(Exception):
    """의존성 관련 오류"""

    pass


class CircularDependencyError(DependencyError):
    """순환 참조 오류"""

    pass


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
        self.instance = None
        self.dependencies: List[Type] = []


class DependencyContainer:
    """의존성 주입 컨테이너"""

    def __init__(self):
        self._registrations: Dict[Type, DependencyRegistration] = {}
        self._instances: Dict[Type, Any] = {}
        self._resolution_stack: Set[Type] = set()
        self._lock = threading.RLock()

        logger.info("✅ 의존성 주입 컨테이너 초기화")

    def register(
        self,
        interface: Type,
        implementation: Type = None,
        factory: Callable = None,
        lifecycle: LifecycleType = LifecycleType.SINGLETON,
    ) -> "DependencyContainer":
        """의존성 등록"""
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
            logger.debug(
                f"의존성 등록: {interface.__name__} -> {implementation.__name__ if implementation else 'Factory'}"
            )

            return self

    def register_instance(
        self, interface: Type, instance: Any
    ) -> "DependencyContainer":
        """인스턴스 직접 등록"""
        with self._lock:
            self._instances[interface] = instance
            logger.debug(f"인스턴스 등록: {interface.__name__}")
            return self

    def resolve(self, interface: Type) -> Any:
        """의존성 해결"""
        with self._lock:
            # 순환 참조 검사
            if interface in self._resolution_stack:
                cycle_path = " -> ".join(
                    [cls.__name__ for cls in self._resolution_stack]
                )
                cycle_path += f" -> {interface.__name__}"
                raise CircularDependencyError(f"순환 참조 탐지: {cycle_path}")

            # 이미 생성된 인스턴스 확인
            if interface in self._instances:
                return self._instances[interface]

            # 등록된 의존성 확인
            if interface not in self._registrations:
                raise DependencyError(f"등록되지 않은 의존성: {interface.__name__}")

            registration = self._registrations[interface]

            # 해결 스택에 추가
            self._resolution_stack.add(interface)

            try:
                instance = self._create_instance(registration)

                # 싱글톤인 경우 캐시
                if registration.lifecycle == LifecycleType.SINGLETON:
                    self._instances[interface] = instance

                return instance

            finally:
                # 해결 스택에서 제거
                self._resolution_stack.discard(interface)

    def _create_instance(self, registration: DependencyRegistration) -> Any:
        """인스턴스 생성"""
        if registration.factory:
            # 팩토리 함수 사용
            return registration.factory()
        else:
            # 생성자 의존성 주입
            return self._create_with_constructor_injection(registration.implementation)

    def _create_with_constructor_injection(self, implementation: Type) -> Any:
        """생성자 의존성 주입으로 인스턴스 생성"""
        try:
            # 생성자 매개변수 분석
            import inspect

            signature = inspect.signature(implementation.__init__)
            parameters = signature.parameters

            # 'self' 매개변수 제외
            param_names = [name for name in parameters.keys() if name != "self"]

            if not param_names:
                # 의존성 없음
                return implementation()

            # 의존성 해결
            dependencies = {}
            for param_name in param_names:
                param = parameters[param_name]

                # 타입 힌트가 있는 경우
                if param.annotation != inspect.Parameter.empty:
                    dependency = self.resolve(param.annotation)
                    dependencies[param_name] = dependency
                else:
                    # 기본값이 있는 경우
                    if param.default != inspect.Parameter.empty:
                        dependencies[param_name] = param.default
                    else:
                        logger.warning(
                            f"의존성 해결 실패: {implementation.__name__}.{param_name}"
                        )

            return implementation(**dependencies)

        except Exception as e:
            logger.error(f"인스턴스 생성 실패: {implementation.__name__}: {e}")
            # 기본 생성자 시도
            return implementation()

    def is_registered(self, interface: Type) -> bool:
        """등록 여부 확인"""
        return interface in self._registrations or interface in self._instances

    def clear(self):
        """컨테이너 초기화"""
        with self._lock:
            self._registrations.clear()
            self._instances.clear()
            self._resolution_stack.clear()
            logger.info("의존성 컨테이너 초기화 완료")

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """의존성 그래프 반환"""
        graph = {}

        for interface, registration in self._registrations.items():
            interface_name = interface.__name__
            dependencies = []

            if registration.implementation:
                try:
                    import inspect

                    signature = inspect.signature(registration.implementation.__init__)
                    parameters = signature.parameters

                    for param_name, param in parameters.items():
                        if (
                            param_name != "self"
                            and param.annotation != inspect.Parameter.empty
                        ):
                            dependencies.append(param.annotation.__name__)

                except Exception:
                    pass

            graph[interface_name] = dependencies

        return graph

    def detect_circular_dependencies(self) -> List[List[str]]:
        """순환 참조 탐지"""
        graph = self.get_dependency_graph()
        cycles = []

        def dfs(node, path, visited):
            if node in path:
                # 순환 참조 발견
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)

            for dependency in graph.get(node, []):
                dfs(dependency, path, visited)

            path.pop()

        visited = set()
        for node in graph:
            if node not in visited:
                dfs(node, [], visited)

        return cycles


# 전역 컨테이너 인스턴스
_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """전역 의존성 컨테이너 반환"""
    return _container


def register(
    interface: Type,
    implementation: Type = None,
    factory: Callable = None,
    lifecycle: LifecycleType = LifecycleType.SINGLETON,
) -> DependencyContainer:
    """의존성 등록 (편의 함수)"""
    return _container.register(interface, implementation, factory, lifecycle)


def register_instance(interface: Type, instance: Any) -> DependencyContainer:
    """인스턴스 등록 (편의 함수)"""
    return _container.register_instance(interface, instance)


def resolve(interface: Type) -> Any:
    """의존성 해결 (편의 함수)"""
    return _container.resolve(interface)


def configure_utils_dependencies():
    """utils 모듈 의존성 설정"""
    from .memory_manager import MemoryManager
    from .cuda_optimizers import CUDAOptimizer
    from .performance_optimizer import PerformanceOptimizer
    from .cache_manager import CacheManager
    from .error_handler import ErrorHandler
    from .unified_config import UnifiedConfig, get_config
    from .unified_logging import UnifiedLogger

    # 핵심 의존성 등록
    register(UnifiedConfig, factory=lambda: get_config("main"))
    register(UnifiedLogger, factory=lambda: get_logger(__name__))

    # 메모리 관리
    register(MemoryManager)
    register(CUDAOptimizer)

    # 성능 최적화
    register(PerformanceOptimizer)
    register(CacheManager)

    # 에러 처리
    register(ErrorHandler)

    logger.info("✅ utils 모듈 의존성 설정 완료")


# 자동 설정 함수
def auto_configure():
    """자동 의존성 설정"""
    try:
        configure_utils_dependencies()

        # 순환 참조 검사
        cycles = _container.detect_circular_dependencies()
        if cycles:
            logger.warning(f"순환 참조 탐지: {cycles}")
        else:
            logger.info("✅ 순환 참조 없음 - 의존성 설정 완료")

    except Exception as e:
        logger.error(f"의존성 설정 실패: {e}")


# 모듈 로드 시 자동 설정
if __name__ != "__main__":
    auto_configure()
