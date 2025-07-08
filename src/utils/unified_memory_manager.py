"""
중앙집중식 메모리 관리 시스템
GPU/CPU 메모리 풀 통합 관리
⚠️ gpu_memory_pool.py의 고급 기능이 통합되었습니다.
"""

import torch
import threading
import psutil
import gc
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from dataclasses import asdict

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


class PoolType(Enum):
    """메모리 풀 타입 (gpu_memory_pool.py에서 통합)"""
    SMALL = "small"  # 64MB 이하
    MEDIUM = "medium"  # 64MB ~ 256MB
    LARGE = "large"  # 256MB ~ 1GB
    XLARGE = "xlarge"  # 1GB 이상


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


@dataclass
class MemoryBlock:
    """메모리 블록 정보 (gpu_memory_pool.py에서 통합)"""
    tensor: torch.Tensor
    size: int
    pool_type: PoolType
    allocated_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    ref_count: int = 0
    is_free: bool = True


@dataclass
class _GPUPoolStats:
    """GPU 메모리 풀 통계 관리"""
    total_allocated: int = 0
    total_freed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    peak_usage: int = 0
    allocation_count: int = 0
    deallocation_count: int = 0

    def record_allocation(self, size: int, from_cache: bool):
        self.allocation_count += 1
        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.total_allocated += size

    def record_deallocation(self, size: int):
        self.deallocation_count += 1
        self.total_freed += size

    def update_peak_usage(self, current_usage: int):
        self.peak_usage = max(self.peak_usage, current_usage)

    def get_stats(self) -> Dict[str, Any]:
        return asdict(self)


class _GPUPoolCleaner:
    """GPU 메모리 풀의 백그라운드 정리 작업 관리"""

    def __init__(self, pool, lock: threading.RLock, shutdown_event: threading.Event, interval: int):
        self.pool = pool
        self.lock = lock
        self.shutdown_event = shutdown_event
        self.interval = interval
        self.thread = None

    def start(self):
        """정리 스레드 시작"""
        if self.thread is None:
            self.thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.thread.start()
            logger.info("GPU 메모리 풀 정리 스레드 시작")

    def stop(self):
        """정리 스레드 중지"""
        if self.thread and self.thread.is_alive():
            self.shutdown_event.set()
            self.thread.join(timeout=5.0)
            logger.info("GPU 메모리 풀 정리 스레드 중지")

    def _cleanup_worker(self):
        """주기적으로 오래된 블록 정리"""
        while not self.shutdown_event.is_set():
            try:
                self.shutdown_event.wait(self.interval)
                if not self.shutdown_event.is_set():
                    self._periodic_cleanup()
            except Exception as e:
                logger.error(f"정리 스레드 오류: {e}")

    def _periodic_cleanup(self):
        """오래된 미사용 블록 정리"""
        with self.lock:
            current_time = time.time()
            cleanup_threshold = 600  # 10분
            total_freed = 0
            
            for pool_type, blocks in self.pool.pools.items():
                freed_count = 0
                
                # 사용되지 않는 오래된 블록들 정리
                for block in blocks[:]:  # 복사본으로 순회
                    if (block.is_free and 
                        current_time - block.last_used > cleanup_threshold and
                        block.ref_count <= 0):
                        blocks.remove(block)
                        total_freed += block.size
                        freed_count += 1
                        del block
                
                if freed_count > 0:
                    logger.debug(f"풀 {pool_type.value}에서 {freed_count}개 블록 정리")
            
            if total_freed > 0:
                self.pool.stats.record_deallocation(total_freed)
                torch.cuda.empty_cache()
                logger.info(f"주기적 정리 완료: {total_freed / 1024**2:.1f}MB 해제")

    def _emergency_cleanup(self):
        """긴급 메모리 정리"""
        with self.lock:
            logger.warning("긴급 메모리 정리 시작")
            
            # 오래된 블록들을 정리
            current_time = time.time()
            cleanup_threshold = 300  # 5분
            
            for pool_type, blocks in self.pool.pools.items():
                blocks_to_remove = []
                
                for block in blocks:
                    if (block.is_free and 
                        current_time - block.last_used > cleanup_threshold and
                        block.ref_count <= 0):
                        blocks_to_remove.append(block)
                
                for block in blocks_to_remove:
                    blocks.remove(block)
                    self.pool.stats.record_deallocation(block.size)
                    del block
                
                if blocks_to_remove:
                    logger.info(f"풀 {pool_type.value}에서 {len(blocks_to_remove)}개 블록 정리")
            
            # GPU 캐시 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("긴급 메모리 정리 완료")


class GPUMemoryPool:
    """
    GPU 메모리 풀 (gpu_memory_pool.py 통합)
    고급 메모리 풀링, 정리 스레드, 통계 시스템 포함
    """

    def __init__(
        self,
        device_id: int = 0,
        max_pool_size: int = 2 * 1024 * 1024 * 1024,  # 2GB
        cleanup_interval: int = 60,  # 60초
        enable_monitoring: bool = True,
    ):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.max_pool_size = max_pool_size
        self.cleanup_interval = cleanup_interval
        self.enable_monitoring = enable_monitoring
        self.available = torch.cuda.is_available()

        self.allocations: Dict[str, MemoryAllocation] = {}
        self.free_blocks: List[Tuple[int, torch.Tensor]] = []
        self._lock = threading.RLock()

        # 고급 메모리 풀들 (gpu_memory_pool.py에서 통합)
        self.pools: Dict[PoolType, List[MemoryBlock]] = {
            PoolType.SMALL: [],
            PoolType.MEDIUM: [],
            PoolType.LARGE: [],
            PoolType.XLARGE: [],
        }

        # 통계 정보 (분리된 클래스 사용)
        self.stats = _GPUPoolStats()

        # 활성 할당 추적 (gpu_memory_pool.py에서 통합)
        self.active_allocations: Dict[int, MemoryBlock] = {}

        # 정리 스레드 (분리된 클래스 사용)
        self.shutdown_event = threading.Event()
        self.cleaner = _GPUPoolCleaner(
            self, self._lock, self.shutdown_event, cleanup_interval
        )

        if self.available:
            self.total_memory = torch.cuda.get_device_properties(device_id).total_memory
            logger.info(f"✅ GPU 메모리 풀 초기화 (디바이스: {device_id}, 총 메모리: {self.total_memory / 1024**3:.1f}GB)")
            
            # 고급 풀 초기화 (gpu_memory_pool.py에서 통합)
            self._initialize_advanced_pools()
            
            if enable_monitoring:
                self.cleaner.start()
        else:
            self.total_memory = 0
            logger.warning("GPU 사용 불가 - CPU 전용 모드")

    def _initialize_advanced_pools(self):
        """고급 메모리 풀 초기화 (gpu_memory_pool.py에서 통합)"""
        try:
            # 디바이스 메모리 정보 확인
            available_memory = self.total_memory - torch.cuda.memory_allocated(self.device_id)
            safe_pool_size = min(self.max_pool_size, int(available_memory * 0.5))

            # 풀별 초기 할당
            pool_configs = {
                PoolType.SMALL: {"size": 64 * 1024 * 1024, "count": 8},  # 64MB x 8
                PoolType.MEDIUM: {"size": 256 * 1024 * 1024, "count": 4},  # 256MB x 4
                PoolType.LARGE: {"size": 512 * 1024 * 1024, "count": 2},  # 512MB x 2
                PoolType.XLARGE: {"size": 1024 * 1024 * 1024, "count": 1},  # 1GB x 1
            }

            total_required = sum(config["size"] * config["count"] for config in pool_configs.values())

            if total_required > safe_pool_size:
                logger.warning(f"풀 크기 조정: {total_required} -> {safe_pool_size}")
                scale_factor = safe_pool_size / total_required
                for pool_type, config in pool_configs.items():
                    config["count"] = max(1, int(config["count"] * scale_factor))

            # 실제 풀 생성
            for pool_type, config in pool_configs.items():
                for _ in range(config["count"]):
                    try:
                        tensor = torch.empty(
                            config["size"] // 4,  # float32 기준
                            dtype=torch.float32,
                            device=self.device,
                        )

                        block = MemoryBlock(
                            tensor=tensor,
                            size=config["size"],
                            pool_type=pool_type,
                            is_free=True,
                        )

                        self.pools[pool_type].append(block)
                        self.stats.record_allocation(config["size"], from_cache=False)

                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"풀 {pool_type.value} 생성 실패: 메모리 부족")
                        break

            logger.info(f"고급 메모리 풀 초기화 완료: {self._get_pool_summary()}")

        except Exception as e:
            logger.error(f"고급 메모리 풀 초기화 실패: {e}")

    def _get_pool_type(self, size: int) -> PoolType:
        """크기에 따른 풀 타입 결정 (gpu_memory_pool.py에서 통합)"""
        if size <= 64 * 1024 * 1024:
            return PoolType.SMALL
        elif size <= 256 * 1024 * 1024:
            return PoolType.MEDIUM
        elif size <= 1024 * 1024 * 1024:
            return PoolType.LARGE
        else:
            return PoolType.XLARGE

    @contextmanager
    def allocate_context(self, size: int, dtype: torch.dtype = torch.float32):
        """메모리 할당 컨텍스트 매니저 (gpu_memory_pool.py에서 통합)"""
        allocated_block = None
        try:
            allocated_block = self._allocate_advanced_block(size, dtype)
            if allocated_block:
                yield allocated_block.tensor
            else:
                # 풀에서 할당 실패 시 직접 할당
                tensor = torch.empty(size // 4, dtype=dtype, device=self.device)
                yield tensor
        except Exception as e:
            logger.error(f"메모리 할당 실패: {e}")
            yield None
        finally:
            if allocated_block:
                self._deallocate_advanced_block(allocated_block)

    def _allocate_advanced_block(self, size: int, dtype: torch.dtype) -> Optional[MemoryBlock]:
        """고급 메모리 블록 할당 (gpu_memory_pool.py에서 통합)"""
        with self._lock:
            pool_type = self._get_pool_type(size)

            # 해당 풀에서 사용 가능한 블록 찾기
            for block in self.pools[pool_type]:
                if block.is_free and block.size >= size:
                    block.is_free = False
                    block.last_used = time.time()
                    block.ref_count += 1
                    
                    # 통계 업데이트
                    self.stats.record_allocation(size, from_cache=True)
                    
                    # 활성 할당 추적
                    self.active_allocations[id(block)] = block
                    
                    return block

            # 사용 가능한 블록이 없으면 새로 할당 시도
            try:
                tensor = torch.empty(size // 4, dtype=dtype, device=self.device)
                
                block = MemoryBlock(
                    tensor=tensor,
                    size=size,
                    pool_type=pool_type,
                    is_free=False,
                    ref_count=1,
                )
                
                # 통계 업데이트
                self.stats.record_allocation(size, from_cache=False)
                
                # 활성 할당 추적
                self.active_allocations[id(block)] = block
                
                return block
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"GPU 메모리 부족 (요청: {size})")
                self._emergency_cleanup()
                return None

    def _deallocate_advanced_block(self, block: MemoryBlock):
        """고급 메모리 블록 해제 (gpu_memory_pool.py에서 통합)"""
        with self._lock:
            if not block.is_free:
                block.is_free = True
                block.ref_count -= 1
                if block.ref_count == 0:
                    self.stats.record_deallocation(block.size)
                
                # 활성 할당에서 제거
                if id(block) in self.active_allocations:
                    del self.active_allocations[id(block)]

    def _emergency_cleanup(self):
        """긴급 메모리 정리 (Cleaner에 위임)"""
        self.cleaner._emergency_cleanup()

    def shutdown(self):
        """메모리 풀 종료 및 정리"""
        logger.info(f"GPU 메모리 풀 종료 시작 (디바이스: {self.device_id})")
        self.cleaner.stop()
        
        with self._lock:
            # 모든 블록 해제
            for pool_type, blocks in self.pools.items():
                for block in blocks:
                    if block.tensor is not None:
                        del block.tensor
                blocks.clear()
            
            # 활성 할당 정리
            self.active_allocations.clear()
            self.allocations.clear()
            self.free_blocks.clear()
        
        # GPU 캐시 정리
        if self.available:
            torch.cuda.empty_cache()

        logger.info(f"GPU 메모리 풀 종료 완료 (디바이스: {self.device_id})")


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
