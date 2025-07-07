"""
GPU 메모리 풀링 시스템 - 메모리 누수 방지 및 효율적 관리

이 모듈은 GPU 메모리의 효율적인 할당과 해제를 관리하여 메모리 누수를 방지합니다.
"""

import threading
import time
import gc
from typing import Dict, Any, Optional, List, Tuple, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.cuda
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


class PoolType(Enum):
    """메모리 풀 타입"""

    SMALL = "small"  # 64MB 이하
    MEDIUM = "medium"  # 64MB ~ 256MB
    LARGE = "large"  # 256MB ~ 1GB
    XLARGE = "xlarge"  # 1GB 이상


@dataclass
class MemoryBlock:
    """메모리 블록 정보"""

    tensor: torch.Tensor
    size: int
    pool_type: PoolType
    allocated_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    ref_count: int = 0
    is_free: bool = True


class GPUMemoryPool:
    """GPU 메모리 풀 관리자"""

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

        self.logger = get_logger(__name__)
        self._lock = threading.RLock()

        # 메모리 풀들
        self.pools: Dict[PoolType, List[MemoryBlock]] = {
            PoolType.SMALL: [],
            PoolType.MEDIUM: [],
            PoolType.LARGE: [],
            PoolType.XLARGE: [],
        }

        # 통계 정보
        self.stats = {
            "total_allocated": 0,
            "total_freed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "peak_usage": 0,
            "allocation_count": 0,
            "deallocation_count": 0,
        }

        # 활성 할당 추적
        self.active_allocations: Dict[int, MemoryBlock] = {}

        # 정리 스레드
        self.cleanup_thread = None
        self.shutdown_event = threading.Event()

        # 초기화
        self._initialize_pools()
        if enable_monitoring:
            self._start_cleanup_thread()

        self.logger.info(f"✅ GPU 메모리 풀 초기화 완료 (디바이스: {device_id})")

    def _initialize_pools(self):
        """메모리 풀 초기화"""
        try:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA 사용 불가능")
                return

            # 디바이스 메모리 정보 확인
            total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
            available_memory = total_memory - torch.cuda.memory_allocated(
                self.device_id
            )

            # 풀 크기 계산 (사용 가능한 메모리의 50% 이하)
            safe_pool_size = min(self.max_pool_size, int(available_memory * 0.5))

            # 풀별 초기 할당
            pool_configs = {
                PoolType.SMALL: {"size": 64 * 1024 * 1024, "count": 8},  # 64MB x 8
                PoolType.MEDIUM: {"size": 256 * 1024 * 1024, "count": 4},  # 256MB x 4
                PoolType.LARGE: {"size": 512 * 1024 * 1024, "count": 2},  # 512MB x 2
                PoolType.XLARGE: {"size": 1024 * 1024 * 1024, "count": 1},  # 1GB x 1
            }

            total_required = sum(
                config["size"] * config["count"] for config in pool_configs.values()
            )

            if total_required > safe_pool_size:
                self.logger.warning(
                    f"풀 크기 조정: {total_required} -> {safe_pool_size}"
                )
                # 풀 크기를 비례적으로 줄임
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
                        self.stats["total_allocated"] += config["size"]

                    except torch.cuda.OutOfMemoryError:
                        self.logger.warning(
                            f"풀 {pool_type.value} 생성 실패: 메모리 부족"
                        )
                        break

            self.logger.info(f"메모리 풀 초기화 완료: {self._get_pool_summary()}")

        except Exception as e:
            self.logger.error(f"메모리 풀 초기화 실패: {e}")

    def _get_pool_type(self, size: int) -> PoolType:
        """크기에 따른 풀 타입 결정"""
        if size <= 64 * 1024 * 1024:
            return PoolType.SMALL
        elif size <= 256 * 1024 * 1024:
            return PoolType.MEDIUM
        elif size <= 1024 * 1024 * 1024:
            return PoolType.LARGE
        else:
            return PoolType.XLARGE

    @contextmanager
    def allocate(self, size: int, dtype: torch.dtype = torch.float32):
        """메모리 할당 컨텍스트 매니저"""
        allocated_block = None
        try:
            allocated_block = self._allocate_block(size, dtype)
            if allocated_block:
                yield allocated_block.tensor
            else:
                # 풀에서 할당 실패 시 직접 할당
                tensor = torch.empty(size // 4, dtype=dtype, device=self.device)
                yield tensor
        except Exception as e:
            self.logger.error(f"메모리 할당 실패: {e}")
            yield None
        finally:
            if allocated_block:
                self._deallocate_block(allocated_block)

    def _allocate_block(self, size: int, dtype: torch.dtype) -> Optional[MemoryBlock]:
        """메모리 블록 할당"""
        with self._lock:
            pool_type = self._get_pool_type(size)

            # 해당 풀에서 사용 가능한 블록 찾기
            for block in self.pools[pool_type]:
                if block.is_free and block.size >= size:
                    block.is_free = False
                    block.ref_count += 1
                    block.last_used = time.time()

                    # 활성 할당에 추가
                    self.active_allocations[id(block)] = block

                    self.stats["cache_hits"] += 1
                    self.stats["allocation_count"] += 1

                    return block

            # 풀에 적절한 블록이 없으면 새로 생성
            try:
                tensor = torch.empty(size // 4, dtype=dtype, device=self.device)
                block = MemoryBlock(
                    tensor=tensor,
                    size=size,
                    pool_type=pool_type,
                    is_free=False,
                    ref_count=1,
                )

                self.pools[pool_type].append(block)
                self.active_allocations[id(block)] = block

                self.stats["cache_misses"] += 1
                self.stats["allocation_count"] += 1
                self.stats["total_allocated"] += size

                return block

            except torch.cuda.OutOfMemoryError:
                self.logger.warning(f"메모리 할당 실패: {size} bytes")
                # 정리 후 재시도
                self._emergency_cleanup()
                return None

    def _deallocate_block(self, block: MemoryBlock):
        """메모리 블록 해제"""
        with self._lock:
            if id(block) in self.active_allocations:
                block.ref_count -= 1
                if block.ref_count <= 0:
                    block.is_free = True
                    block.ref_count = 0
                    del self.active_allocations[id(block)]
                    self.stats["deallocation_count"] += 1

    def _emergency_cleanup(self):
        """긴급 정리"""
        try:
            self.logger.warning("🚨 긴급 메모리 정리 시작")

            # 사용하지 않는 블록들 정리
            cleaned_count = 0
            for pool_type, blocks in self.pools.items():
                to_remove = []
                for i, block in enumerate(blocks):
                    if (
                        block.is_free and time.time() - block.last_used > 30
                    ):  # 30초 이상 미사용
                        to_remove.append(i)
                        cleaned_count += 1

                # 역순으로 제거
                for i in reversed(to_remove):
                    del blocks[i]

            # PyTorch 캐시 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # 가비지 컬렉션
            gc.collect()

            self.logger.info(f"긴급 정리 완료: {cleaned_count}개 블록 정리")

        except Exception as e:
            self.logger.error(f"긴급 정리 실패: {e}")

    def _start_cleanup_thread(self):
        """정리 스레드 시작"""

        def cleanup_worker():
            while not self.shutdown_event.wait(self.cleanup_interval):
                try:
                    self._periodic_cleanup()
                except Exception as e:
                    self.logger.error(f"정기 정리 실패: {e}")

        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        self.logger.debug("정리 스레드 시작")

    def _periodic_cleanup(self):
        """정기 정리"""
        with self._lock:
            current_time = time.time()
            total_cleaned = 0

            for pool_type, blocks in self.pools.items():
                to_remove = []
                for i, block in enumerate(blocks):
                    # 5분 이상 미사용 블록 정리
                    if (
                        block.is_free
                        and current_time - block.last_used > 300
                        and len(blocks) > 1
                    ):  # 최소 1개는 유지
                        to_remove.append(i)
                        total_cleaned += 1

                # 역순으로 제거
                for i in reversed(to_remove):
                    del blocks[i]

            if total_cleaned > 0:
                self.logger.debug(f"정기 정리: {total_cleaned}개 블록 정리")
                torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        with self._lock:
            # 현재 메모리 사용량
            current_usage = sum(
                sum(block.size for block in blocks if not block.is_free)
                for blocks in self.pools.values()
            )

            # 풀별 통계
            pool_stats = {}
            for pool_type, blocks in self.pools.items():
                pool_stats[pool_type.value] = {
                    "total_blocks": len(blocks),
                    "free_blocks": sum(1 for block in blocks if block.is_free),
                    "used_blocks": sum(1 for block in blocks if not block.is_free),
                    "total_size": sum(block.size for block in blocks),
                    "used_size": sum(
                        block.size for block in blocks if not block.is_free
                    ),
                }

            return {
                **self.stats,
                "current_usage": current_usage,
                "active_allocations": len(self.active_allocations),
                "pool_stats": pool_stats,
                "cache_hit_rate": (
                    self.stats["cache_hits"]
                    / (self.stats["cache_hits"] + self.stats["cache_misses"])
                    if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
                    else 0
                ),
            }

    def _get_pool_summary(self) -> str:
        """풀 요약 정보"""
        summary = []
        for pool_type, blocks in self.pools.items():
            if blocks:
                total_size = sum(block.size for block in blocks)
                summary.append(
                    f"{pool_type.value}: {len(blocks)}개 블록 ({total_size // (1024*1024)}MB)"
                )
        return ", ".join(summary)

    def shutdown(self):
        """메모리 풀 종료"""
        try:
            self.logger.info("GPU 메모리 풀 종료 시작")

            # 정리 스레드 종료
            if self.cleanup_thread:
                self.shutdown_event.set()
                self.cleanup_thread.join(timeout=5)

            # 모든 블록 해제
            with self._lock:
                for pool_type, blocks in self.pools.items():
                    blocks.clear()

                self.active_allocations.clear()

            # CUDA 캐시 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            self.logger.info("✅ GPU 메모리 풀 종료 완료")

        except Exception as e:
            self.logger.error(f"메모리 풀 종료 실패: {e}")


# 전역 메모리 풀 인스턴스
_memory_pools: Dict[int, GPUMemoryPool] = {}
_pool_lock = threading.Lock()


def get_gpu_memory_pool(device_id: int = 0) -> GPUMemoryPool:
    """GPU 메모리 풀 반환 (싱글톤)"""
    global _memory_pools

    with _pool_lock:
        if device_id not in _memory_pools:
            _memory_pools[device_id] = GPUMemoryPool(device_id=device_id)
        return _memory_pools[device_id]


@contextmanager
def gpu_memory_scope(
    size_mb: int, device_id: int = 0, dtype: torch.dtype = torch.float32
):
    """GPU 메모리 스코프 컨텍스트 매니저"""
    pool = get_gpu_memory_pool(device_id)
    size_bytes = size_mb * 1024 * 1024

    with pool.allocate(size_bytes, dtype) as tensor:
        yield tensor


def cleanup_all_memory_pools():
    """모든 메모리 풀 정리"""
    global _memory_pools

    with _pool_lock:
        for device_id, pool in _memory_pools.items():
            pool.shutdown()
        _memory_pools.clear()


def get_memory_pool_stats() -> Dict[str, Any]:
    """모든 메모리 풀 통계 반환"""
    global _memory_pools

    with _pool_lock:
        return {
            f"device_{device_id}": pool.get_stats()
            for device_id, pool in _memory_pools.items()
        }
