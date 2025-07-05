"""
중앙집중식 메모리 관리 시스템
GPU/CPU 메모리 풀 통합 관리
"""

import torch
import threading
import psutil
import gc
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import weakref
from contextlib import contextmanager

from .unified_logging import get_logger
from .unified_config import get_config
from .factory import get_singleton_instance

logger = get_logger(__name__)


class DeviceType(Enum):
    """디바이스 타입"""

    GPU = "gpu"
    CPU = "cpu"


class AllocationStrategy(Enum):
    """메모리 할당 전략"""

    PREFER_GPU = "prefer_gpu"
    PREFER_CPU = "prefer_cpu"
    BALANCED = "balanced"
    SMART = "smart"


@dataclass
class MemoryAllocation:
    """메모리 할당 정보"""

    id: str
    device: DeviceType
    size: int
    tensor: Optional[torch.Tensor]
    metadata: Dict[str, Any]
    created_at: float
    last_accessed: float


class GPUMemoryPool:
    """GPU 메모리 풀"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.available = torch.cuda.is_available()
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.free_blocks: List[Tuple[int, torch.Tensor]] = []
        self._lock = threading.Lock()

        if self.available:
            self.total_memory = torch.cuda.get_device_properties(device_id).total_memory
            logger.info(
                f"✅ GPU 메모리 풀 초기화 (디바이스: {device_id}, 총 메모리: {self.total_memory / 1024**3:.1f}GB)"
            )
        else:
            self.total_memory = 0
            logger.warning("GPU 사용 불가 - CPU 전용 모드")

    def allocate(
        self, size: int, dtype: torch.dtype = torch.float32, allocation_id: str = None
    ) -> Optional[torch.Tensor]:
        """GPU 메모리 할당"""
        if not self.available:
            return None

        with self._lock:
            try:
                # 재사용 가능한 블록 검색
                tensor = self._find_reusable_block(size, dtype)

                if tensor is None:
                    # 새 텐서 할당
                    tensor = torch.empty(size, dtype=dtype, device=self.device)

                # 할당 정보 저장
                if allocation_id:
                    import time

                    allocation = MemoryAllocation(
                        id=allocation_id,
                        device=DeviceType.GPU,
                        size=tensor.element_size() * tensor.nelement(),
                        tensor=tensor,
                        metadata={"dtype": str(dtype)},
                        created_at=time.time(),
                        last_accessed=time.time(),
                    )
                    self.allocations[allocation_id] = allocation

                return tensor

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"GPU 메모리 부족 (요청: {size})")
                self._cleanup_unused_blocks()
                return None

    def deallocate(self, allocation_id: str) -> bool:
        """GPU 메모리 해제"""
        with self._lock:
            if allocation_id in self.allocations:
                allocation = self.allocations[allocation_id]

                # 재사용 풀에 추가
                if allocation.tensor is not None:
                    size = (
                        allocation.tensor.element_size() * allocation.tensor.nelement()
                    )
                    self.free_blocks.append((size, allocation.tensor))

                del self.allocations[allocation_id]
                return True
        return False

    def _find_reusable_block(
        self, size: int, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """재사용 가능한 메모리 블록 검색"""
        element_size = torch.tensor([], dtype=dtype).element_size()
        required_bytes = size * element_size

        for i, (block_size, tensor) in enumerate(self.free_blocks):
            if block_size >= required_bytes:
                # 블록 제거 후 반환
                self.free_blocks.pop(i)
                return tensor[:size]

        return None

    def _cleanup_unused_blocks(self):
        """사용하지 않는 블록 정리"""
        self.free_blocks.clear()
        torch.cuda.empty_cache()
        logger.info("GPU 메모리 블록 정리 완료")

    def get_memory_info(self) -> Dict[str, Any]:
        """GPU 메모리 정보 반환"""
        if not self.available:
            return {"available": False}

        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)

        return {
            "available": True,
            "total": self.total_memory,
            "allocated": allocated,
            "reserved": reserved,
            "free": self.total_memory - reserved,
            "utilization": (allocated / self.total_memory) * 100,
            "active_allocations": len(self.allocations),
            "free_blocks": len(self.free_blocks),
        }


class CPUMemoryPool:
    """CPU 메모리 풀"""

    def __init__(self):
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.free_blocks: List[Tuple[int, torch.Tensor]] = []
        self._lock = threading.Lock()

        # 시스템 메모리 정보
        memory = psutil.virtual_memory()
        self.total_memory = memory.total

        logger.info(
            f"✅ CPU 메모리 풀 초기화 (총 메모리: {self.total_memory / 1024**3:.1f}GB)"
        )

    def allocate(
        self, size: int, dtype: torch.dtype = torch.float32, allocation_id: str = None
    ) -> torch.Tensor:
        """CPU 메모리 할당"""
        with self._lock:
            try:
                # 재사용 가능한 블록 검색
                tensor = self._find_reusable_block(size, dtype)

                if tensor is None:
                    # 새 텐서 할당
                    tensor = torch.empty(size, dtype=dtype, device="cpu")

                # 할당 정보 저장
                if allocation_id:
                    import time

                    allocation = MemoryAllocation(
                        id=allocation_id,
                        device=DeviceType.CPU,
                        size=tensor.element_size() * tensor.nelement(),
                        tensor=tensor,
                        metadata={"dtype": str(dtype)},
                        created_at=time.time(),
                        last_accessed=time.time(),
                    )
                    self.allocations[allocation_id] = allocation

                return tensor

            except Exception as e:
                logger.error(f"CPU 메모리 할당 실패: {e}")
                self._cleanup_unused_blocks()
                raise

    def deallocate(self, allocation_id: str) -> bool:
        """CPU 메모리 해제"""
        with self._lock:
            if allocation_id in self.allocations:
                allocation = self.allocations[allocation_id]

                # 재사용 풀에 추가
                if allocation.tensor is not None:
                    size = (
                        allocation.tensor.element_size() * allocation.tensor.nelement()
                    )
                    self.free_blocks.append((size, allocation.tensor))

                del self.allocations[allocation_id]
                return True
        return False

    def _find_reusable_block(
        self, size: int, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """재사용 가능한 메모리 블록 검색"""
        element_size = torch.tensor([], dtype=dtype).element_size()
        required_bytes = size * element_size

        for i, (block_size, tensor) in enumerate(self.free_blocks):
            if block_size >= required_bytes and tensor.dtype == dtype:
                # 블록 제거 후 반환
                self.free_blocks.pop(i)
                return tensor[:size]

        return None

    def _cleanup_unused_blocks(self):
        """사용하지 않는 블록 정리"""
        self.free_blocks.clear()
        gc.collect()
        logger.info("CPU 메모리 블록 정리 완료")

    def get_memory_info(self) -> Dict[str, Any]:
        """CPU 메모리 정보 반환"""
        memory = psutil.virtual_memory()

        return {
            "total": self.total_memory,
            "available": memory.available,
            "used": memory.used,
            "utilization": memory.percent,
            "active_allocations": len(self.allocations),
            "free_blocks": len(self.free_blocks),
        }


class UnifiedMemoryManager:
    """통합 메모리 관리자"""

    def __init__(self):
        self.config = get_config("main").get_nested("utils.memory", {})

        # 메모리 풀 초기화
        self.gpu_pool = GPUMemoryPool()
        self.cpu_pool = CPUMemoryPool()

        # 할당 추적
        self.allocation_tracker: Dict[str, DeviceType] = {}
        self._allocation_counter = 0
        self._lock = threading.Lock()

        # 설정
        self.gpu_memory_threshold = (
            self.config.get("gpu_memory_threshold_gb", 1.0) * 1024**3
        )
        self.cpu_memory_threshold = (
            self.config.get("cpu_memory_threshold_gb", 2.0) * 1024**3
        )
        self.default_strategy = AllocationStrategy(
            self.config.get("default_allocation_strategy", "smart")
        )

        logger.info(
            f"✅ 통합 메모리 관리자 초기화 (전략: {self.default_strategy.value})"
        )

    def smart_allocate(
        self,
        size: int,
        dtype: torch.dtype = torch.float32,
        prefer_device: DeviceType = None,
        strategy: AllocationStrategy = None,
    ) -> Tuple[torch.Tensor, str]:
        """스마트 메모리 할당"""
        strategy = strategy or self.default_strategy
        allocation_id = self._generate_allocation_id()

        # 전략에 따른 디바이스 선택
        device = self._select_device(size, dtype, prefer_device, strategy)

        # 메모리 할당
        if device == DeviceType.GPU:
            tensor = self.gpu_pool.allocate(size, dtype, allocation_id)
            if tensor is None:
                # GPU 할당 실패시 CPU로 폴백
                logger.warning("GPU 할당 실패, CPU로 폴백")
                tensor = self.cpu_pool.allocate(size, dtype, allocation_id)
                device = DeviceType.CPU
        else:
            tensor = self.cpu_pool.allocate(size, dtype, allocation_id)

        # 할당 추적
        with self._lock:
            self.allocation_tracker[allocation_id] = device

        logger.debug(f"메모리 할당: {allocation_id} ({device.value}, {size} 요소)")
        return tensor, allocation_id

    def deallocate(self, allocation_id: str) -> bool:
        """메모리 해제"""
        with self._lock:
            if allocation_id not in self.allocation_tracker:
                return False

            device = self.allocation_tracker[allocation_id]

            # 디바이스별 해제
            if device == DeviceType.GPU:
                success = self.gpu_pool.deallocate(allocation_id)
            else:
                success = self.cpu_pool.deallocate(allocation_id)

            if success:
                del self.allocation_tracker[allocation_id]
                logger.debug(f"메모리 해제: {allocation_id} ({device.value})")

            return success

    def _select_device(
        self,
        size: int,
        dtype: torch.dtype,
        prefer_device: DeviceType,
        strategy: AllocationStrategy,
    ) -> DeviceType:
        """최적 디바이스 선택"""
        element_size = torch.tensor([], dtype=dtype).element_size()
        required_bytes = size * element_size

        # GPU 메모리 정보
        gpu_info = self.gpu_pool.get_memory_info()
        cpu_info = self.cpu_pool.get_memory_info()

        # 선호 디바이스가 지정된 경우
        if prefer_device:
            if prefer_device == DeviceType.GPU and gpu_info["available"]:
                if gpu_info["free"] > required_bytes:
                    return DeviceType.GPU
            elif prefer_device == DeviceType.CPU:
                if cpu_info["available"] > required_bytes:
                    return DeviceType.CPU

        # 전략별 선택
        if strategy == AllocationStrategy.PREFER_GPU:
            if gpu_info["available"] and gpu_info["free"] > required_bytes:
                return DeviceType.GPU
        elif strategy == AllocationStrategy.PREFER_CPU:
            if cpu_info["available"] > required_bytes:
                return DeviceType.CPU
        elif strategy == AllocationStrategy.SMART:
            # 스마트 선택 로직
            if gpu_info["available"]:
                gpu_utilization = gpu_info["utilization"]
                cpu_utilization = cpu_info["utilization"]

                # GPU 메모리 여유가 충분하고 사용률이 낮은 경우
                if (
                    gpu_info["free"] > required_bytes * 2
                    and gpu_utilization < 70
                    and required_bytes > 1024**2
                ):  # 1MB 이상
                    return DeviceType.GPU

                # CPU 사용률이 높은 경우 GPU 선택
                if cpu_utilization > 80 and gpu_utilization < 80:
                    return DeviceType.GPU

        # 기본값: CPU
        return DeviceType.CPU

    def _generate_allocation_id(self) -> str:
        """할당 ID 생성"""
        with self._lock:
            self._allocation_counter += 1
            return f"alloc_{self._allocation_counter:06d}"

    @contextmanager
    def temporary_allocation(
        self,
        size: int,
        dtype: torch.dtype = torch.float32,
        prefer_device: DeviceType = None,
    ):
        """임시 메모리 할당 컨텍스트 매니저"""
        tensor, allocation_id = self.smart_allocate(size, dtype, prefer_device)
        try:
            yield tensor
        finally:
            self.deallocate(allocation_id)

    def get_memory_status(self) -> Dict[str, Any]:
        """전체 메모리 상태 반환"""
        gpu_info = self.gpu_pool.get_memory_info()
        cpu_info = self.cpu_pool.get_memory_info()

        return {
            "gpu": gpu_info,
            "cpu": cpu_info,
            "total_allocations": len(self.allocation_tracker),
            "allocation_distribution": {
                "gpu": sum(
                    1
                    for device in self.allocation_tracker.values()
                    if device == DeviceType.GPU
                ),
                "cpu": sum(
                    1
                    for device in self.allocation_tracker.values()
                    if device == DeviceType.CPU
                ),
            },
        }

    def cleanup_all(self):
        """모든 메모리 정리"""
        with self._lock:
            # 모든 할당 해제
            allocation_ids = list(self.allocation_tracker.keys())
            for allocation_id in allocation_ids:
                self.deallocate(allocation_id)

            # 메모리 풀 정리
            self.gpu_pool._cleanup_unused_blocks()
            self.cpu_pool._cleanup_unused_blocks()

            logger.info("✅ 모든 메모리 정리 완료")

    def optimize_memory_usage(self):
        """메모리 사용량 최적화"""
        # 사용하지 않는 할당 정리
        current_time = time.time()
        stale_allocations = []

        for allocation_id, device in self.allocation_tracker.items():
            if device == DeviceType.GPU:
                allocation = self.gpu_pool.allocations.get(allocation_id)
            else:
                allocation = self.cpu_pool.allocations.get(allocation_id)

            if allocation and (current_time - allocation.last_accessed) > 300:  # 5분
                stale_allocations.append(allocation_id)

        # 오래된 할당 해제
        for allocation_id in stale_allocations:
            self.deallocate(allocation_id)
            logger.debug(f"오래된 할당 해제: {allocation_id}")

        # 메모리 풀 최적화
        self.gpu_pool._cleanup_unused_blocks()
        self.cpu_pool._cleanup_unused_blocks()

        logger.info(f"메모리 최적화 완료 (해제된 할당: {len(stale_allocations)})")


# 싱글톤 인스턴스
def get_unified_memory_manager() -> UnifiedMemoryManager:
    """통합 메모리 관리자 싱글톤 인스턴스 반환"""
    return get_singleton_instance(UnifiedMemoryManager)


# 편의 함수
def smart_allocate(
    size: int, dtype: torch.dtype = torch.float32, prefer_device: DeviceType = None
) -> Tuple[torch.Tensor, str]:
    """스마트 메모리 할당 편의 함수"""
    manager = get_unified_memory_manager()
    return manager.smart_allocate(size, dtype, prefer_device)


def deallocate(allocation_id: str) -> bool:
    """메모리 해제 편의 함수"""
    manager = get_unified_memory_manager()
    return manager.deallocate(allocation_id)


def get_memory_status() -> Dict[str, Any]:
    """메모리 상태 조회 편의 함수"""
    manager = get_unified_memory_manager()
    return manager.get_memory_status()


def temporary_allocation(
    size: int, dtype: torch.dtype = torch.float32, prefer_device: DeviceType = None
):
    """임시 메모리 할당 편의 함수"""
    manager = get_unified_memory_manager()
    return manager.temporary_allocation(size, dtype, prefer_device)
