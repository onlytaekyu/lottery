"""
CUDA 싱글톤 매니저 - 중복 초기화 방지 및 리소스 관리

이 모듈은 CUDA 리소스의 중복 초기화를 방지하고 효율적인 메모리 관리를 제공합니다.
"""

import threading
import time
from typing import Dict, Any, Optional, List, Set
from contextlib import contextmanager
import torch
import torch.cuda
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


class CudaSingletonConfig:
    """CUDA 싱글톤 설정 클래스"""

    def __init__(
        self,
        use_amp: bool = True,
        batch_size: int = 64,
        use_cudnn: bool = True,
        memory_fraction: float = 0.8,
        device_id: int = 0,
        enable_profiling: bool = False,
    ):
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.use_cudnn = use_cudnn
        self.memory_fraction = memory_fraction
        self.device_id = device_id
        self.enable_profiling = enable_profiling


class CudaSingletonManager:
    """CUDA 리소스 싱글톤 관리자"""

    _instance = None
    _lock = threading.RLock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if CudaSingletonManager._initialized:
            return

        with CudaSingletonManager._lock:
            if CudaSingletonManager._initialized:
                return

            self.logger = get_logger(__name__)
            self.device = None
            self.is_available = False
            self.memory_pools = {}
            self.stream_pools = {}
            self.active_contexts = set()
            self.config = None
            self.requesters = {}  # 요청자별 사용량 추적

            # CUDA 초기화
            self._initialize_cuda()

            CudaSingletonManager._initialized = True
            self.logger.info("✅ CUDA 싱글톤 매니저 초기화 완료")

    def _initialize_cuda(self):
        """CUDA 환경 초기화"""
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.is_available = True

                # CUDA 메모리 설정
                torch.cuda.empty_cache()

                # cuDNN 최적화
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True

                # 메모리 풀 초기화
                self._initialize_memory_pools()

                self.logger.info(f"🚀 CUDA 초기화 완료: {torch.cuda.get_device_name()}")
            else:
                self.logger.warning("⚠️ CUDA 사용 불가능")
                self.is_available = False

        except Exception as e:
            self.logger.error(f"CUDA 초기화 실패: {e}")
            self.is_available = False

    def _initialize_memory_pools(self):
        """메모리 풀 초기화"""
        try:
            if self.is_available:
                # 기본 메모리 풀들
                pool_sizes = {
                    "small": 256 * 1024 * 1024,  # 256MB
                    "medium": 512 * 1024 * 1024,  # 512MB
                    "large": 1024 * 1024 * 1024,  # 1GB
                }

                for pool_name, size in pool_sizes.items():
                    try:
                        # 메모리 풀 생성
                        pool_tensor = torch.empty(
                            size // 4, dtype=torch.float32, device=self.device
                        )
                        self.memory_pools[pool_name] = {
                            "tensor": pool_tensor,
                            "size": size,
                            "allocated": 0,
                            "free_blocks": [],
                        }
                        self.logger.debug(
                            f"메모리 풀 '{pool_name}' 생성: {size // (1024*1024)}MB"
                        )
                    except torch.cuda.OutOfMemoryError:
                        self.logger.warning(
                            f"메모리 풀 '{pool_name}' 생성 실패: 메모리 부족"
                        )

        except Exception as e:
            self.logger.error(f"메모리 풀 초기화 실패: {e}")

    def register_requester(
        self, requester_name: str, config: Optional[CudaSingletonConfig] = None
    ):
        """요청자 등록"""
        with self._lock:
            if requester_name not in self.requesters:
                self.requesters[requester_name] = {
                    "config": config,
                    "memory_usage": 0,
                    "active_contexts": 0,
                    "last_access": time.time(),
                }
                self.logger.debug(f"요청자 등록: {requester_name}")

    def get_cuda_optimizer(self, requester_name: str) -> "CudaOptimizer":
        """CUDA 최적화기 반환"""
        self.register_requester(requester_name)
        return CudaOptimizer(self, requester_name)

    @contextmanager
    def device_context(self, requester_name: str):
        """디바이스 컨텍스트 관리"""
        if not self.is_available:
            yield None
            return

        context_id = f"{requester_name}_{time.time()}"
        self.active_contexts.add(context_id)

        try:
            with torch.cuda.device(self.device):
                yield self.device
        finally:
            self.active_contexts.discard(context_id)

    @contextmanager
    def memory_scope(self, requester_name: str, size_mb: int = 100):
        """메모리 스코프 관리"""
        if not self.is_available:
            yield None
            return

        allocated_memory = None
        try:
            # 메모리 할당
            size_bytes = size_mb * 1024 * 1024
            allocated_memory = torch.empty(
                size_bytes // 4, dtype=torch.float32, device=self.device
            )

            # 사용량 추적
            if requester_name in self.requesters:
                self.requesters[requester_name]["memory_usage"] += size_bytes
                self.requesters[requester_name]["last_access"] = time.time()

            yield allocated_memory

        except torch.cuda.OutOfMemoryError:
            self.logger.warning(f"메모리 할당 실패: {size_mb}MB 요청")
            yield None
        finally:
            # 메모리 해제
            if allocated_memory is not None:
                del allocated_memory
                torch.cuda.empty_cache()

                # 사용량 추적 업데이트
                if requester_name in self.requesters:
                    self.requesters[requester_name]["memory_usage"] -= size_bytes

    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 반환"""
        if not self.is_available:
            return {"available": False}

        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()

            return {
                "available": True,
                "allocated": allocated,
                "reserved": reserved,
                "max_allocated": max_allocated,
                "free": reserved - allocated,
                "device_name": torch.cuda.get_device_name(),
                "device_count": torch.cuda.device_count(),
            }
        except Exception as e:
            self.logger.error(f"메모리 정보 조회 실패: {e}")
            return {"available": False, "error": str(e)}

    def cleanup_inactive_requesters(self, timeout_seconds: int = 300):
        """비활성 요청자 정리"""
        current_time = time.time()
        inactive_requesters = []

        with self._lock:
            for requester_name, info in self.requesters.items():
                if current_time - info["last_access"] > timeout_seconds:
                    inactive_requesters.append(requester_name)

            for requester_name in inactive_requesters:
                del self.requesters[requester_name]
                self.logger.debug(f"비활성 요청자 정리: {requester_name}")

    def get_utilization_stats(self) -> Dict[str, Any]:
        """사용률 통계 반환"""
        try:
            if not self.is_available:
                return {"available": False}

            # GPU 사용률 (근사치)
            memory_info = self.get_memory_info()
            utilization = (
                (memory_info["allocated"] / memory_info["reserved"]) * 100
                if memory_info["reserved"] > 0
                else 0
            )

            return {
                "available": True,
                "gpu_utilization": utilization,
                "active_contexts": len(self.active_contexts),
                "registered_requesters": len(self.requesters),
                "memory_info": memory_info,
                "requester_stats": {
                    name: {
                        "memory_usage": info["memory_usage"],
                        "active_contexts": info["active_contexts"],
                        "last_access": info["last_access"],
                    }
                    for name, info in self.requesters.items()
                },
            }
        except Exception as e:
            self.logger.error(f"사용률 통계 조회 실패: {e}")
            return {"available": False, "error": str(e)}

    def force_cleanup(self):
        """강제 정리"""
        try:
            if self.is_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.info("🧹 CUDA 리소스 강제 정리 완료")
        except Exception as e:
            self.logger.error(f"강제 정리 실패: {e}")


class CudaOptimizer:
    """CUDA 최적화기 - 싱글톤 매니저 기반"""

    def __init__(self, singleton_manager: CudaSingletonManager, requester_name: str):
        self.singleton_manager = singleton_manager
        self.requester_name = requester_name
        self.logger = get_logger(__name__)

    def is_available(self) -> bool:
        """CUDA 사용 가능 여부 반환"""
        return self.singleton_manager.is_available

    @contextmanager
    def device_context(self):
        """디바이스 컨텍스트"""
        with self.singleton_manager.device_context(self.requester_name) as device:
            yield device

    @contextmanager
    def memory_scope(self, size_mb: int = 100):
        """메모리 스코프"""
        with self.singleton_manager.memory_scope(
            self.requester_name, size_mb
        ) as memory:
            yield memory

    def get_gpu_utilization(self) -> float:
        """GPU 사용률 반환"""
        stats = self.singleton_manager.get_utilization_stats()
        return stats.get("gpu_utilization", 0.0)

    def optimize_tensor_operations(self, tensor: torch.Tensor) -> torch.Tensor:
        """텐서 연산 최적화"""
        if not self.is_available():
            return tensor

        try:
            with self.device_context():
                # GPU로 이동
                gpu_tensor = tensor.to(self.singleton_manager.device)

                # 최적화된 연산 수행
                # 예: 메모리 레이아웃 최적화
                if gpu_tensor.is_contiguous():
                    return gpu_tensor
                else:
                    return gpu_tensor.contiguous()

        except Exception as e:
            self.logger.error(f"텐서 최적화 실패: {e}")
            return tensor


# 전역 싱글톤 인스턴스
_cuda_singleton = None


def get_singleton_cuda_optimizer(
    config: Optional[CudaSingletonConfig] = None, requester_name: str = "default"
) -> CudaOptimizer:
    """싱글톤 CUDA 최적화기 반환"""
    global _cuda_singleton

    if _cuda_singleton is None:
        _cuda_singleton = CudaSingletonManager()

    if config:
        _cuda_singleton.config = config

    return _cuda_singleton.get_cuda_optimizer(requester_name)


def get_cuda_memory_info() -> Dict[str, Any]:
    """CUDA 메모리 정보 반환"""
    global _cuda_singleton

    if _cuda_singleton is None:
        _cuda_singleton = CudaSingletonManager()

    return _cuda_singleton.get_memory_info()


def cleanup_cuda_resources():
    """CUDA 리소스 정리"""
    global _cuda_singleton

    if _cuda_singleton is not None:
        _cuda_singleton.force_cleanup()
        _cuda_singleton.cleanup_inactive_requesters()
