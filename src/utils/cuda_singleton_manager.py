"""
CUDA 싱글톤 관리자

CUDA 최적화기의 중복 초기화 문제를 완전히 해결하는 싱글톤 시스템
- 전역 단일 CUDA 최적화기 인스턴스
- Thread-Safe 보장
- 메모리 최적화된 CUDA 리소스 관리
"""

import threading
import weakref
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass
import torch

from .unified_logging import get_logger

# 전역 변수들
_cuda_manager_instance = None
_cuda_manager_lock = threading.RLock()
_cuda_optimizers_cache: Dict[str, Any] = {}

logger = get_logger(__name__)


@dataclass
class CudaSingletonConfig:
    """CUDA 싱글톤 설정"""

    use_amp: bool = True
    use_cudnn: bool = True
    use_graphs: bool = False
    batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 256
    enable_memory_pool: bool = True
    enable_profiling: bool = False


class CudaSingletonManager:
    """
    CUDA 싱글톤 관리자

    모든 CUDA 최적화기를 중앙에서 관리하여 중복 초기화를 방지합니다.
    """

    _instance = None
    _lock = threading.RLock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._cuda_available = torch.cuda.is_available()
            self._primary_optimizer = None
            self._config = None
            self._initialization_count = 0
            self._active_references: Set[str] = set()
            self._device_info = {}

            # CUDA 사용 가능 시 기본 설정
            if self._cuda_available:
                self._initialize_cuda_environment()

            self._initialized = True
            logger.info("🚀 CUDA 싱글톤 관리자 초기화 완료")

    def _initialize_cuda_environment(self):
        """CUDA 환경 초기화 (한 번만)"""
        try:
            # 기본 CUDA 설정
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # 디바이스 정보 수집
            self._device_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.device_count() > 0
                    else "Unknown"
                ),
                "memory_total": (
                    torch.cuda.get_device_properties(0).total_memory
                    if torch.cuda.device_count() > 0
                    else 0
                ),
            }

            logger.info(
                f"✅ CUDA 환경 초기화 완료 - {self._device_info['device_name']}"
            )

        except Exception as e:
            logger.warning(f"CUDA 환경 초기화 중 오류: {e}")

    def get_cuda_optimizer(
        self,
        config: Optional[CudaSingletonConfig] = None,
        requester_name: str = "unknown",
    ) -> Optional[Any]:
        """
        CUDA 최적화기 반환 (싱글톤)

        Args:
            config: CUDA 설정 (첫 번째 호출 시에만 적용)
            requester_name: 요청하는 모듈명 (로깅용)

        Returns:
            CUDA 최적화기 인스턴스 또는 None
        """
        with self._lock:
            self._initialization_count += 1
            self._active_references.add(requester_name)

            # CUDA 사용 불가능
            if not self._cuda_available:
                if self._initialization_count == 1:  # 첫 번째 호출에서만 로그
                    logger.info("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
                return None

            # 첫 번째 호출 시 최적화기 생성
            if self._primary_optimizer is None:
                try:
                    # 설정 저장
                    self._config = config or CudaSingletonConfig()

                    # 실제 CUDA 최적화기 생성
                    self._primary_optimizer = self._create_cuda_optimizer()

                    logger.info(
                        f"✅ CUDA 최적화기 생성 완료 (요청자: {requester_name})"
                    )

                except Exception as e:
                    logger.error(f"CUDA 최적화기 생성 실패: {e}")
                    return None
            else:
                # 이미 생성된 경우 중복 방지 로그
                if len(self._active_references) <= 3:  # 처음 3개까지만 로그
                    logger.debug(
                        f"기존 CUDA 최적화기 재사용 (요청자: {requester_name})"
                    )

            return self._primary_optimizer

    def _create_cuda_optimizer(self) -> Any:
        """실제 CUDA 최적화기 생성"""
        try:
            # 동적 import로 순환 참조 방지
            from .cuda_optimizers import BaseCudaOptimizer, CudaConfig

            # CudaConfig 생성
            config = self._config or CudaSingletonConfig()
            cuda_config = CudaConfig(
                use_amp=config.use_amp,
                use_cudnn=config.use_cudnn,
                use_graphs=config.use_graphs,
                batch_size=config.batch_size,
                min_batch_size=config.min_batch_size,
                max_batch_size=config.max_batch_size,
            )

            # BaseCudaOptimizer 생성 (내부 로깅 억제)
            optimizer = BaseCudaOptimizer(cuda_config)

            return optimizer

        except Exception as e:
            logger.error(f"CUDA 최적화기 생성 중 오류: {e}")
            raise

    def is_cuda_available(self) -> bool:
        """CUDA 사용 가능 여부"""
        return self._cuda_available

    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        return self._device_info.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        with self._lock:
            return {
                "cuda_available": self._cuda_available,
                "initialization_count": self._initialization_count,
                "active_references": list(self._active_references),
                "optimizer_created": self._primary_optimizer is not None,
                "device_info": self._device_info,
            }

    def cleanup(self):
        """정리 작업"""
        with self._lock:
            if self._primary_optimizer is not None:
                try:
                    if hasattr(self._primary_optimizer, "cleanup"):
                        self._primary_optimizer.cleanup()
                except Exception as e:
                    logger.warning(f"CUDA 최적화기 정리 중 오류: {e}")

                self._primary_optimizer = None

            self._active_references.clear()
            logger.info("CUDA 싱글톤 관리자 정리 완료")


# 전역 함수들
def get_cuda_singleton_manager() -> CudaSingletonManager:
    """전역 CUDA 싱글톤 관리자 반환"""
    global _cuda_manager_instance

    if _cuda_manager_instance is None:
        with _cuda_manager_lock:
            if _cuda_manager_instance is None:
                _cuda_manager_instance = CudaSingletonManager()

    return _cuda_manager_instance


def get_singleton_cuda_optimizer(
    config: Optional[CudaSingletonConfig] = None, requester_name: str = "unknown"
) -> Optional[Any]:
    """
    싱글톤 CUDA 최적화기 반환

    Args:
        config: CUDA 설정
        requester_name: 요청하는 모듈명

    Returns:
        CUDA 최적화기 인스턴스 또는 None
    """
    manager = get_cuda_singleton_manager()
    return manager.get_cuda_optimizer(config, requester_name)


def is_cuda_available() -> bool:
    """CUDA 사용 가능 여부 확인"""
    manager = get_cuda_singleton_manager()
    return manager.is_cuda_available()


def get_cuda_device_info() -> Dict[str, Any]:
    """CUDA 디바이스 정보 반환"""
    manager = get_cuda_singleton_manager()
    return manager.get_device_info()


def get_cuda_statistics() -> Dict[str, Any]:
    """CUDA 싱글톤 통계 반환"""
    manager = get_cuda_singleton_manager()
    return manager.get_statistics()


def cleanup_cuda_singleton():
    """CUDA 싱글톤 정리"""
    global _cuda_manager_instance

    if _cuda_manager_instance is not None:
        _cuda_manager_instance.cleanup()
        _cuda_manager_instance = None


# 모듈 종료 시 정리
import atexit

atexit.register(cleanup_cuda_singleton)
