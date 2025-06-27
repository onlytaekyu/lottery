"""CUDA 메모리 관리 모듈

이 모듈은 CUDA 메모리 관리를 위한 클래스를 제공합니다.
메모리 사용 효율성을 높이고 메모리 누수를 방지하기 위한 기능들이 포함되어 있습니다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, TypeVar, Generic
from threading import Lock, Thread
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import zlib
import time
import gc
import torch
from pathlib import Path
import threading
import queue
from collections import OrderedDict
import psutil
import sys
import platform
import logging
import weakref
import traceback
from contextlib import contextmanager
import os
import random
import json

# 상대경로 임포트로 변경
from .error_handler_refactored import get_logger
from .unified_performance import get_profiler
from .config_loader import ConfigProxy

logger = get_logger(__name__)


# CUDA 설정 플랫폼별 최적화
def get_cuda_alloc_config():
    """플랫폼에 따른 CUDA 메모리 할당 설정 반환"""
    if platform.system() == "Windows":
        return "max_split_size_mb:512"
    else:
        return "expandable_segments:True,max_split_size_mb:512"


# CUDA 설정 적용
if torch.cuda.is_available():
    cuda_config = get_cuda_alloc_config()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cuda_config
    logger.info(f"CUDA 메모리 설정: {cuda_config}")


T = TypeVar("T")


@dataclass
class ThreadLocalConfig:
    """스레드 로컬 설정 클래스"""

    max_size: int = 1000
    max_memory: int = 1 << 30  # 1GB
    ttl: float = 3600.0  # 1시간
    cleanup_interval: float = 300.0  # 5분
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreadLocalCacheEntry(Generic[T]):
    """스레드 로컬 캐시 항목 클래스"""

    value: T
    size: int
    access_count: int = 0
    last_access: float = 0.0
    ttl: float = 0.0


class ThreadLocalCache(Generic[T]):
    """스레드 로컬 캐시 클래스"""

    def __init__(self, config: Optional[ThreadLocalConfig] = None):
        """
        스레드 로컬 캐시 초기화

        Args:
            config: 스레드 로컬 캐시 설정
        """
        self.config = config or ThreadLocalConfig()
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._stats = {
            "hit_count": 0,
            "miss_count": 0,
            "eviction_count": 0,
            "memory_usage": 0,
        }
        self._local = threading.local()
        self._last_cleanup = time.time()

    def _get_cache(self) -> OrderedDict[str, ThreadLocalCacheEntry[T]]:
        """스레드별 캐시 가져오기"""
        if not hasattr(self._local, "cache"):
            self._local.cache = OrderedDict()
        return self._local.cache

    def _cleanup(self):
        """만료된 캐시 항목 정리"""
        try:
            current_time = time.time()
            if current_time - self._last_cleanup < self.config.cleanup_interval:
                return

            with self._lock:
                cache = self._get_cache()
                expired_keys = []
                for key, entry in cache.items():
                    if entry.ttl > 0 and current_time - entry.last_access > entry.ttl:
                        expired_keys.append(key)
                        self._stats["eviction_count"] += 1
                        self._stats["memory_usage"] -= entry.size

                for key in expired_keys:
                    del cache[key]

                self._last_cleanup = current_time
                logger.info(f"캐시 정리 완료: {len(expired_keys)}개 항목 제거")

        except Exception as e:
            logger.error(f"캐시 정리 실패: {str(e)}")

    def get(self, key: str) -> Optional[T]:
        """
        캐시에서 값을 가져옵니다.

        Args:
            key: 캐시 키

        Returns:
            캐시된 값 또는 None
        """
        with self._lock:
            cache = self._get_cache()
            if key in cache:
                entry = cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                self._stats["hit_count"] += 1
                self._hit_count += 1
                return entry.value
            self._stats["miss_count"] += 1
            self._miss_count += 1
            return None

    def put(self, key: str, value: T, size: Optional[int] = None) -> bool:
        """값 캐싱"""
        try:
            self._cleanup()
            cache = self._get_cache()

            # 크기 계산
            if size is None:
                size = self._estimate_size(value)

            # 메모리 제한 확인
            while (
                len(cache) >= self.config.max_size
                or self._stats["memory_usage"] + size > self.config.max_memory  # type: ignore
            ):
                if not cache:
                    return False

                # LRU 항목 제거
                _, entry = cache.popitem(last=False)
                self._stats["eviction_count"] += 1
                self._stats["memory_usage"] -= entry.size  # type: ignore

            # 새 항목 추가
            cache[key] = ThreadLocalCacheEntry(
                value=value,
                size=size,
                last_access=time.time(),
                ttl=self.config.ttl,
            )
            self._stats["memory_usage"] += size  # type: ignore
            return True

        except Exception as e:
            logger.error(f"캐시 값 저장 실패: {str(e)}")
            return False

    def _estimate_size(self, value: T) -> int:
        """값의 대략적인 크기 추정"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
            else:
                return 1 << 10  # 기본 크기 1KB
        except Exception:
            return 1 << 10  # 오류 발생 시 기본 크기

    def clear(self):
        """캐시 비우기"""
        try:
            with self._lock:
                cache = self._get_cache()
                self._stats["memory_usage"] = 0
                cache.clear()
                logger.info("캐시 초기화 완료")
        except Exception as e:
            logger.error(f"캐시 초기화 실패: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        stats = {}
        with self._lock:
            stats["hit_count"] = self._hit_count
            stats["miss_count"] = self._miss_count
            stats["eviction_count"] = self._eviction_count
            stats["access_count"] = self._hit_count + self._miss_count
            stats["hit_rate"] = (
                (self._hit_count / (self._hit_count + self._miss_count)) * 100
                if (self._hit_count + self._miss_count) > 0
                else 0
            )
            stats["entry_count"] = len(self._get_cache())
            stats["memory_usage"] = self._stats["memory_usage"]
            stats["memory_limit"] = self.config.max_memory
            stats["usage_percent"] = (  # type: ignore
                100 * self._stats["memory_usage"] / self.config.max_memory
                if self.config.max_memory > 0
                else 0
            )
            stats["items_count"] = len(self._get_cache())
            stats["device"] = "cpu"  # type: ignore
            return stats

    def reset_stats(self):
        """통계 초기화"""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0


# 메모리 관리 설정 클래스
@dataclass
class MemoryConfig:
    """메모리 관리 설정 클래스"""

    # 메모리 사용량 관련 설정
    max_memory_usage: float = 0.85  # 안전을 위해 85%로 제한
    cache_size: int = 256 * 1024 * 1024  # 캐시 크기 (256MB)
    min_batch_size: int = 1
    max_batch_size: int = 128  # 최대 배치 크기 제한
    optimal_batch_size: int = 16  # 최적 배치 크기
    memory_track_interval: int = 5  # 메모리 추적 간격
    memory_frags_threshold: float = 0.3
    memory_usage_warning: float = 0.95  # 경고 임계값

    # 워커 관련 설정
    num_workers: int = 2  # 워커 수 줄임 (메모리 부담 감소)
    prefetch_factor: int = 2

    # 메모리 풀 관련 설정
    pool_size: int = 32
    compression_threshold: int = 16 << 20  # 16MB
    alignment_size: int = 256

    # 최적화 관련 설정
    use_memory_pooling: bool = True
    use_memory_alignment: bool = True
    use_memory_compression: bool = True
    use_memory_prefetching: bool = False
    use_memory_reuse: bool = True
    use_memory_tracking: bool = True
    use_memory_optimization: bool = True
    use_memory_compaction: bool = True
    use_memory_pinning: bool = False
    use_memory_streaming: bool = False
    use_memory_events: bool = False
    use_memory_graphs: bool = False
    use_memory_peer_access: bool = False
    use_memory_unified: bool = False
    use_memory_multi_gpu: bool = False
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    # 메모리 풀 통계
    memory_pool_sizes: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_usage: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_hits: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_misses: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_evictions: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_compressions: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_decompressions: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_prefetches: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_reuses: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_alignments: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_errors: Dict[Tuple[torch.Size, torch.dtype], int] = field(
        default_factory=dict
    )
    memory_pool_stats: Dict[str, Any] = field(default_factory=dict)
    auto_cleanup_interval: float = 60.0  # 자동 정리 간격

    # 플랫폼별 메모리 설정
    platform: str = platform.system()

    # 플랫폼별 최적화 설정
    use_pinned_memory: bool = field(init=False)
    use_shared_memory: bool = field(init=False)
    cuda_memory_config: str = field(init=False)

    def __post_init__(self):
        """초기화 후 처리"""
        # 플랫폼별 최적화 설정 초기화
        if self.platform == "Windows":
            # Windows는 일부 CUDA 기능 제한
            self.use_pinned_memory = False
            self.use_shared_memory = False
            self.cuda_memory_config = "max_split_size_mb:512"
        else:
            # Linux/MacOS
            self.use_pinned_memory = True
            self.use_shared_memory = True
            self.cuda_memory_config = "expandable_segments:True,max_split_size_mb:512"

        # 메모리 설정 검증
        self.max_memory_usage = max(0.1, min(self.max_memory_usage, 0.95))
        self.cache_size = max(1 << 20, min(self.cache_size, 1 << 30))  # 1MB ~ 1GB
        self.min_batch_size = max(1, min(self.min_batch_size, self.max_batch_size))
        self.max_batch_size = max(self.min_batch_size, min(self.max_batch_size, 256))
        self.optimal_batch_size = max(
            self.min_batch_size, min(self.optimal_batch_size, self.max_batch_size)
        )
        self.memory_track_interval = max(1, min(self.memory_track_interval, 60))
        self.memory_frags_threshold = max(0.1, min(self.memory_frags_threshold, 0.5))
        self.memory_usage_warning = max(0.5, min(self.memory_usage_warning, 0.98))

        # 워커 설정 검증
        self.num_workers = max(1, min(self.num_workers, os.cpu_count() or 4))
        self.prefetch_factor = max(1, min(self.prefetch_factor, 4))

        # 풀 설정 검증
        self.pool_size = max(10, min(self.pool_size, 128))
        self.compression_threshold = max(
            1 << 10, min(self.compression_threshold, 1 << 28)
        )
        self.alignment_size = max(32, min(self.alignment_size, 512))

        # GPU ID 검증
        if torch.cuda.is_available():
            self.gpu_ids = [
                gpu_id
                for gpu_id in self.gpu_ids
                if 0 <= gpu_id < torch.cuda.device_count()
            ]
            if not self.gpu_ids:
                self.gpu_ids = [0]
        else:
            self.gpu_ids = []

    def update(self, new_config: Dict[str, Any]):
        """설정 업데이트"""
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.__post_init__()  # 업데이트된 값 검증


@dataclass
class CacheEntry:
    """캐시 항목 클래스"""

    tensor: torch.Tensor
    size: int
    access_count: int = 0
    last_access: float = 0.0
    compressed: bool = False
    compressed_data: Optional[bytes] = None
    creation_time: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)

    def update_access(self):
        """접근 정보 업데이트"""
        self.access_count += 1
        self.last_access = time.time()


class MemoryPool:
    """메모리 풀"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.pools: Dict[Tuple[torch.Size, torch.dtype], List[torch.Tensor]] = {}
        self.allocated_tensors: Dict[int, Tuple[torch.Size, torch.dtype]] = {}
        self.lock = Lock()
        self._initialize()

    def _initialize(self):
        """메모리 풀 초기화"""
        try:
            # 기본 텐서 타입에 대한 풀 생성
            if torch.cuda.is_available() and self.config.use_memory_pooling:
                # GPU 풀 설정
                torch.backends.cudnn.benchmark = True
                for device_id in self.config.gpu_ids:
                    with torch.cuda.device(device_id):
                        if hasattr(torch.cuda, "empty_cache"):
                            torch.cuda.empty_cache()
                logger.info(f"CUDA 메모리 풀 초기화 완료: {self.config.gpu_ids}")
        except Exception as e:
            logger.error(f"메모리 풀 초기화 실패: {str(e)}")

    def _create_pool(self, shape: Tuple[int, ...], dtype: torch.dtype):
        """특정 크기와 타입의 풀 생성"""
        try:
            if (torch.Size(shape), dtype) not in self.pools:
                self.pools[(torch.Size(shape), dtype)] = []
                logger.debug(f"새 메모리 풀 생성: 형태={shape}, 타입={dtype}")
        except Exception as e:
            logger.error(f"메모리 풀 생성 실패: {str(e)}")

    def _align_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """텐서 정렬"""
        if not self.config.use_memory_alignment:
            return tensor

        try:
            shape = tensor.shape
            dtype = tensor.dtype
            size = tensor.numel() * tensor.element_size()

            # 정렬 크기 계산
            if size < self.config.alignment_size:
                return tensor

            # 정렬 필요 없는 경우
            if size % self.config.alignment_size == 0:
                return tensor

            # 정렬 크기 계산
            aligned_size = (
                (size + self.config.alignment_size - 1)
                // self.config.alignment_size
                * self.config.alignment_size
            )

            # 필요한 경우에만 새 텐서 생성
            if aligned_size != size:
                # 계정 크기 기록
                self.config.memory_pool_alignments[(torch.Size(shape), dtype)] = (
                    self.config.memory_pool_alignments.get(
                        (torch.Size(shape), dtype), 0
                    )
                    + 1
                )
                # 정렬된 크기로 새 텐서 생성
                new_shape = (aligned_size // tensor.element_size(),)
                aligned_tensor = torch.empty(
                    new_shape, dtype=dtype, device=tensor.device
                )
                return aligned_tensor.view(shape)
            return tensor
        except Exception as e:
            logger.error(f"텐서 정렬 실패: {str(e)}")
            return tensor

    def get_tensor(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """풀에서 텐서 가져오기"""
        try:
            with self.lock:
                key = (torch.Size(shape), dtype)
                if key not in self.pools:
                    self._create_pool(shape, dtype)
                    self.config.memory_pool_misses[key] = (
                        self.config.memory_pool_misses.get(key, 0) + 1
                    )
                    return None

                if not self.pools[key]:
                    self.config.memory_pool_misses[key] = (
                        self.config.memory_pool_misses.get(key, 0) + 1
                    )
                    return None

                tensor = self.pools[key].pop()
                self.allocated_tensors[id(tensor)] = key
                self.config.memory_pool_hits[key] = (
                    self.config.memory_pool_hits.get(key, 0) + 1
                )
                return tensor
        except Exception as e:
            logger.error(f"풀에서 텐서 가져오기 실패: {str(e)}")
            return None

    def return_tensor(self, tensor: torch.Tensor):
        """텐서를 풀로 반환"""
        try:
            tensor_id = id(tensor)
            with self.lock:
                if tensor_id not in self.allocated_tensors:
                    return

                key = self.allocated_tensors[tensor_id]
                # 풀 크기 제한
                if len(self.pools[key]) >= self.config.pool_size:
                    # 풀이 꽉 찬 경우, 가장 오래된 텐서 제거
                    self.pools[key].pop(0)
                    self.config.memory_pool_evictions[key] = (
                        self.config.memory_pool_evictions.get(key, 0) + 1
                    )

                self.pools[key].append(tensor)
                del self.allocated_tensors[tensor_id]
                self.config.memory_pool_reuses[key] = (
                    self.config.memory_pool_reuses.get(key, 0) + 1
                )
        except Exception as e:
            logger.error(f"텐서 풀 반환 실패: {str(e)}")

    def cleanup(self):
        """모든 풀 정리"""
        try:
            with self.lock:
                self.pools.clear()
                self.allocated_tensors.clear()
                logger.info("메모리 풀 정리 완료")
        except Exception as e:
            logger.error(f"메모리 풀 정리 실패: {str(e)}")


class TensorCache:
    """텐서 캐시"""

    def __init__(
        self,
        max_memory_size: int = 1 * 1024 * 1024 * 1024,  # 1GB로 기본 크기 줄임
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.max_memory_size = max_memory_size
        self.device = device
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = Lock()
        self.current_memory_usage = 0
        self.hit_count = 0
        self.miss_count = 0
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "insertions": 0,
        }

    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """텐서 캐싱"""
        try:
            tensor_size = tensor.numel() * tensor.element_size()

            with self.lock:
                # 이미 있는 키인 경우 업데이트
                if key in self.cache:
                    old_entry = self.cache[key]
                    self.current_memory_usage -= old_entry.size

                # 메모리 부족 시 공간 확보
                while (
                    self.current_memory_usage + tensor_size > self.max_memory_size
                    and self.cache
                ):
                    self._evict()

                # 여전히 공간이 부족하면 실패
                if self.current_memory_usage + tensor_size > self.max_memory_size:
                    return False

                # 텐서 복사 (분리)
                cached_tensor = tensor.clone().detach()
                entry = CacheEntry(
                    tensor=cached_tensor,
                    size=tensor_size,
                )
                self.cache[key] = entry
                self.current_memory_usage += tensor_size
                self.stats["insertions"] += 1
                return True
        except Exception as e:
            logger.error(f"텐서 캐싱 실패: {str(e)}")
            return False

    def get(self, key: str) -> Optional[torch.Tensor]:
        """캐시된 텐서 가져오기"""
        try:
            with self.lock:
                if key in self.cache:
                    entry = self.cache[key]
                    entry.update_access()
                    self.hit_count += 1
                    return entry.tensor
                self.miss_count += 1
                return None
        except Exception as e:
            logger.error(f"캐시된 텐서 가져오기 실패: {str(e)}")
            return None

    def _evict(self) -> bool:
        """LRU 항목 제거"""
        try:
            if not self.cache:
                return False

            # 가장 오래전에 접근한 항목 찾기
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_access)
            entry = self.cache.pop(lru_key)
            self.current_memory_usage -= entry.size
            self.stats["evictions"] += 1
            return True
        except Exception as e:
            logger.error(f"캐시 항목 제거 실패: {str(e)}")
            return False

    def clear(self):
        """캐시 비우기"""
        try:
            with self.lock:
                self.cache.clear()
                self.current_memory_usage = 0
                logger.info("텐서 캐시 초기화 완료")
        except Exception as e:
            logger.error(f"텐서 캐시 초기화 실패: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self.lock:
            stats = {}
            stats["hit_count"] = self.hit_count
            stats["miss_count"] = self.miss_count
            stats["access_count"] = self.hit_count + self.miss_count
            stats["hit_rate"] = (
                (self.hit_count / (self.hit_count + self.miss_count)) * 100
                if (self.hit_count + self.miss_count) > 0
                else 0
            )
            stats["cache_size"] = self.current_memory_usage
            stats["memory_usage"] = self.current_memory_usage
            stats["memory_limit"] = self.max_memory_size
            stats["usage_percent"] = (  # type: ignore
                100 * self.current_memory_usage / self.max_memory_size
                if self.max_memory_size > 0
                else 0
            )
            stats["items_count"] = len(self.cache)
            stats["device"] = self.device  # type: ignore
            return stats


class MemoryManager:
    """메모리 관리자"""

    _instance = None

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        메모리 관리자 초기화

        Args:
            config: 메모리 관리 설정 (기본값: None)
        """
        self.config = config or MemoryConfig()
        self.lock = threading.RLock()
        self.memory_pool = MemoryPool(self.config)

        # 로그 디렉토리 설정
        self.log_dir = Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 메모리 사용량 로그 파일
        self.memory_log_file = self.log_dir / "memory_usage.csv"

        # 워커 스레드
        self.worker_threads = []
        self.allocation_queue = queue.Queue()
        self.deallocation_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.cleanup_thread = None
        self.cache_manager_thread = None
        self.auto_cleanup_thread = None
        self.memory_monitor_thread = None

        # 스레드 풀 초기화
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 텐서 캐시
        self.tensor_cache = TensorCache(
            max_memory_size=self.config.cache_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # 메모리 사용량 모니터링
        self.memory_usage = {
            "gpu": 0.0,
            "ram": 0.0,
            "gpu_allocated": 0,
            "gpu_reserved": 0,
            "gpu_max_allocated": 0,
            "gpu_max_reserved": 0,
        }

        # GPU 메모리 사용량 추적
        self.gpu_memory_allocated = []
        self.gpu_memory_reserved = []
        self.timestamps = []

        # 메모리 릭 감지
        self.allocations = {}
        self.allocation_counts = {}
        self.deallocations = {}
        self.deallocation_counts = {}

        # 배치 크기 동적 조정 관련
        self.optimal_batch_size = self.config.optimal_batch_size
        self.current_batch_size = self.optimal_batch_size

        # 스케일링 설정
        self.scaling_factors = {
            "memory_low": 1.2,  # 메모리 여유 시 배치 크기 증가 비율
            "memory_high": 0.8,  # 메모리 부족 시 배치 크기 감소 비율
        }

        # 스레드 로컬 캐시
        self.thread_local_cache = ThreadLocalCache(ThreadLocalConfig())

        # 메모리 이벤트 리스너
        self.memory_event_listeners = []

        # 메모리 그래프 활성화 (CUDA만 해당)
        self.memory_graphs_enabled = (
            torch.cuda.is_available() and self.config.use_memory_graphs
        )
        self.current_stream = None
        self.current_graph = None

        # 캐시 히트/미스 카운터
        self._cache_hits = 0
        self._cache_misses = 0

        # 마지막 메모리 확인 시간 및 경고 상태 추적
        self._last_cleanup_time = time.time()
        self._memory_warning_issued = False
        self._memory_warning_count = 0
        self._last_warning_time = time.time()

        # CUDA 설정 초기화
        self._initial_cuda_setup()

        # 워커 스레드 시작
        self._start_worker_threads()
        self._start_cleanup_thread()

    def _initial_cuda_setup(self):
        """초기 CUDA 설정"""
        if not torch.cuda.is_available():
            logger.info("CUDA를 사용할 수 없습니다")
            return

        # CUDA 할당 설정
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.config.cuda_memory_config
        logger.info(f"CUDA 메모리 설정: {self.config.cuda_memory_config}")

    def _start_worker_threads(self):
        """작업자 스레드 시작"""
        try:
            if self.worker_threads:
                logger.debug("작업자 스레드가 이미 실행 중입니다")
                return

            # 스레드 중지 플래그 초기화
            self.stop_event.clear()

            # 메모리 정리 스레드 시작
            self.cleanup_thread = Thread(
                target=self._cleanup_loop, daemon=True, name="MemoryCleanupThread"
            )
            self.cleanup_thread.start()

            # 작업자 스레드 시작
            for i in range(self.config.num_workers):
                thread = Thread(
                    target=self._worker_loop,
                    daemon=True,
                    name=f"MemoryWorkerThread-{i}",
                )
                thread.start()
                self.worker_threads.append(thread)

            logger.info(
                f"메모리 관리자 작업자 스레드 {len(self.worker_threads)}개 시작됨"
            )
        except Exception as e:
            logger.error(f"작업자 스레드 시작 중 오류: {str(e)}")

    def _cleanup_loop(self):
        """메모리 정리 루프"""
        try:
            while not self.stop_event.is_set():
                try:
                    # 일정 시간마다 메모리 사용량 확인 및 캐시 정리
                    self._cleanup_memory()
                    time.sleep(self.config.memory_track_interval)
                except Exception as e:
                    logger.error(f"메모리 정리 루프 실행 중 오류: {str(e)}")
                    time.sleep(
                        self.config.memory_track_interval * 2
                    )  # 오류 발생 시 더 오래 대기
        except Exception as e:
            logger.error(f"메모리 정리 루프 종료: {str(e)}")

    def _worker_loop(self):
        """작업자 스레드 루프"""
        try:
            while not self.stop_event.is_set():
                try:
                    # 할당 큐에서 작업 가져오기 (1초 타임아웃)
                    try:
                        task = self.allocation_queue.get(timeout=1.0)
                        self._process_allocation_task(task)
                        self.allocation_queue.task_done()
                    except Empty:
                        pass

                    # 해제 큐에서 작업 가져오기 (타임아웃 없음)
                    try:
                        task = self.deallocation_queue.get_nowait()
                        self._process_deallocation_task(task)
                        self.deallocation_queue.task_done()
                    except Empty:
                        pass

                except Exception as e:
                    logger.error(f"작업자 루프 실행 중 오류: {str(e)}")

                # 잠시 대기
                time.sleep(0.01)
        except Exception as e:
            logger.error(f"작업자 루프 종료: {str(e)}")

    def _start_cleanup_thread(self):
        """자동 정리 스레드 시작"""
        try:
            # 이미 실행 중인 스레드가 있으면 중지
            if hasattr(self, "cleanup_thread") and self.cleanup_thread is not None:
                self.stop_event.set()
                if self.cleanup_thread.is_alive():
                    self.cleanup_thread.join(timeout=1.0)

            # 정지 이벤트 초기화
            self.stop_event = threading.Event()

            # 자동 정리 스레드 생성 및 시작
            self.cleanup_thread = threading.Thread(
                target=self._auto_cleanup_task, daemon=True, name="MemoryAutoCleanup"
            )
            self.cleanup_thread.start()
            logger.debug("자동 정리 스레드 시작됨")
        except Exception as e:
            logger.error(f"자동 정리 스레드 초기화 중 오류 발생: {str(e)}")
            self.cleanup_thread = None

    def _memory_monitor_task(self):
        """메모리 모니터링 작업 (워커 스레드)"""
        try:
            while not self.stop_event.is_set():
                try:
                    # 메모리 통계 업데이트
                    self._update_gpu_memory_stats()

                    # 메모리 사용량 로깅
                    if torch.cuda.is_available():
                        logger.debug(
                            f"GPU 메모리: 할당={torch.cuda.memory_allocated() / (1024**2):.1f}MB, "
                            f"예약={torch.cuda.memory_reserved() / (1024**2):.1f}MB"
                        )

                    # 시스템 메모리 사용량
                    ram_usage = psutil.virtual_memory().percent
                    logger.debug(f"시스템 메모리 사용률: {ram_usage:.1f}%")

                    # 메모리 사용량이 높은 경우 정리 요청
                    if ram_usage > 90 or (
                        torch.cuda.is_available()
                        and torch.cuda.memory_allocated()
                        / torch.cuda.get_device_properties(0).total_memory
                        > 0.9
                    ):
                        logger.warning("메모리 사용량이 높습니다. 정리 요청...")
                        self._cleanup_internal()

                    # 일정 시간 대기
                    time.sleep(self.config.memory_track_interval)
                except Exception as e:
                    logger.error(f"메모리 모니터링 중 오류: {str(e)}")
                    time.sleep(10)  # 오류 발생 시 대기 시간 증가
        except Exception as e:
            logger.error(f"메모리 모니터링 태스크 실패: {str(e)}")

    def _cache_manager_task(self):
        """캐시 관리 작업 (워커 스레드)"""
        try:
            # 처음에 잠시 대기하여 다른 초기화가 완료되도록 함
            time.sleep(5)

            while not self.stop_event.is_set():
                try:
                    # 캐시 항목 정리 (30분 이상 사용되지 않은 항목)
                    with self.lock:
                        now = time.time()
                        expired_keys = []

                        # 오래된 항목 식별
                        for key, entry in self.allocations.items():
                            if now - entry.last_access > 1800:  # 30분
                                expired_keys.append(key)

                        # 만료된 항목 제거
                        for key in expired_keys:
                            del self.allocations[key]

                        if expired_keys:
                            logger.info(
                                f"{len(expired_keys)}개의 만료된 캐시 항목 제거됨"
                            )

                    # 일정 시간 대기
                    time.sleep(300)  # 5분마다 검사
                except Exception as e:
                    logger.error(f"캐시 관리 중 오류: {str(e)}")
                    time.sleep(60)  # 오류 발생 시 대기 시간 증가
        except Exception as e:
            logger.error(f"캐시 관리 태스크 실패: {str(e)}")

    def _auto_cleanup_task(self):
        """자동 정리 작업 (워커 스레드)"""
        try:
            # 처음에 잠시 대기하여 다른 초기화가 완료되도록 함
            time.sleep(10)

            while not self.stop_event.is_set():
                try:
                    # 메모리 사용량 확인 및 정리
                    if self.should_cleanup():
                        logger.info("자동 정리 시작...")
                        self._cleanup_internal()
                        logger.info("자동 정리 완료")

                    # 일정 시간 대기
                    time.sleep(self.config.auto_cleanup_interval)
                except Exception as e:
                    logger.error(f"자동 정리 중 오류: {str(e)}")
                    time.sleep(60)  # 오류 발생 시 대기 시간 증가
        except Exception as e:
            logger.error(f"자동 정리 태스크 실패: {str(e)}")

    @contextmanager
    def allocation_scope(self):
        """
        메모리 할당 스코프 (컨텍스트 매니저)

        이 컨텍스트 내에서 생성된 텐서는 추적되어 관리됩니다.
        """
        try:
            # 진입 시 메모리 상태 기록
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                before_allocated = torch.cuda.memory_allocated()
                before_reserved = torch.cuda.memory_reserved()

            # 컨텍스트 내부 코드 실행
            yield

        finally:
            # 종료 시 메모리 정리 고려
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                after_allocated = torch.cuda.memory_allocated()
                after_reserved = torch.cuda.memory_reserved()

                delta_allocated = after_allocated - before_allocated
                delta_reserved = after_reserved - before_reserved

                # 변화량이 크면 기록
                if abs(delta_allocated) > 10 * 1024 * 1024:  # 10MB 이상
                    logger.debug(
                        f"메모리 변화: 할당={delta_allocated/(1024*1024):.1f}MB, "
                        f"예약={delta_reserved/(1024*1024):.1f}MB"
                    )

    def should_cleanup(self) -> bool:
        """
        메모리 정리 필요 여부 확인

        Returns:
            정리 필요 여부
        """
        try:
            # 마지막 정리 후 일정 시간이 지나지 않았다면 정리하지 않음
            if time.time() - self._last_cleanup_time < 5.0:  # 최소 5초 간격
                return False

            if torch.cuda.is_available():
                # GPU 메모리 사용률 검사
                memory_usage = (
                    torch.cuda.memory_allocated()
                    / torch.cuda.get_device_properties(0).total_memory
                )

                # 경고 임계값 초과 시 경고 발생
                if (
                    memory_usage > self.config.memory_usage_warning
                    and not self._memory_warning_issued
                ):
                    self._memory_warning_issued = True
                    self._memory_warning_count += 1
                    self._last_warning_time = time.time()
                    logger.warning(
                        f"GPU 메모리 사용률 높음: {memory_usage*100:.1f}% "
                        f"(경고 {self._memory_warning_count}회 발생)"
                    )

                # 경고 상태 복구
                if (
                    self._memory_warning_issued
                    and memory_usage < self.config.memory_usage_warning - 0.1
                ):
                    self._memory_warning_issued = False

                return memory_usage > self.config.max_memory_usage
            else:
                # CPU 메모리 사용률 검사
                memory_usage = psutil.virtual_memory().percent / 100.0
                return memory_usage > self.config.max_memory_usage
        except Exception as e:
            logger.error(f"메모리 정리 필요 여부 확인 중 오류 발생: {str(e)}")
            return False

    def cleanup(self) -> None:
        """메모리 정리"""
        try:
            with self.lock:
                start_time = time.time()

                # 캐시된 텐서 정리
                for key in list(self.allocations.keys()):
                    del self.allocations[key]
                self.allocations.clear()

                # 텐서 캐시 정리
                if hasattr(self, "tensor_cache"):
                    self.tensor_cache.clear()

                # CUDA 캐시 정리
                if torch.cuda.is_available():
                    # 진행 중인 CUDA 작업 동기화
                    torch.cuda.synchronize()

                    # CUDA 캐시 정리
                    torch.cuda.empty_cache()

                    # GPU 메모리 상태 로깅
                    memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                    memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    logger.debug(
                        f"CUDA 메모리 정리 후: 할당={memory_allocated:.1f}MB, 예약={memory_reserved:.1f}MB"
                    )

                # 가비지 컬렉션 실행
                gc.collect()

                # 풀 정리 고려
                if hasattr(self, "memory_pool"):
                    if random.random() < 0.1:  # 10% 확률로 풀도 완전 정리
                        self.memory_pool.cleanup()
                        logger.debug("메모리 풀 정리 완료")

                duration = time.time() - start_time
                logger.info(f"메모리 정리 완료 (소요 시간: {duration:.2f}초)")

        except Exception as e:
            logger.error(f"메모리 정리 중 오류 발생: {str(e)}")
            traceback.print_exc()

    def get_memory_usage(self, memory_type: str = "gpu") -> float:
        """
        시스템/GPU 메모리 사용률을 반환합니다.

        Args:
            memory_type: "gpu" 또는 "ram" (기본값: "gpu")

        Returns:
            메모리 사용률 (0.0 ~ 1.0)
        """
        try:
            if memory_type == "gpu" and torch.cuda.is_available():
                # GPU 메모리 사용률
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_allocated = torch.cuda.memory_allocated(0)
                return memory_allocated / total_memory
            else:
                # 시스템 메모리 사용률
                return psutil.virtual_memory().percent / 100.0
        except Exception as e:
            logger.error(f"메모리 사용량 확인 실패: {str(e)}")
            return 0.0

    def _update_gpu_memory_stats(self):
        """GPU 메모리 상태 업데이트"""
        if torch.cuda.is_available():
            try:
                current_device = torch.cuda.current_device()
                self.gpu_memory_allocated = torch.cuda.memory_allocated(
                    current_device
                ) / (1024 * 1024)
                self.gpu_memory_reserved = torch.cuda.memory_reserved(
                    current_device
                ) / (1024 * 1024)

                # OOM 위험 감지
                if self.gpu_memory_allocated > 0.95 * torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024 * 1024):
                    logger.warning(
                        f"OOM 위험! GPU 메모리 사용량 높음: {self.gpu_memory_allocated:.1f}MB"
                    )
                    # 긴급 메모리 정리
                    self.cleanup()
            except Exception as e:
                logger.error(f"GPU 메모리 통계 업데이트 오류: {str(e)}")

    def clear_cache(self):
        """메모리 캐시를 정리합니다."""
        try:
            # 텐서 캐시 정리
            if hasattr(self, "tensor_cache"):
                self.tensor_cache.clear()

            # 스레드 로컬 캐시 정리
            if hasattr(self, "thread_local_cache"):
                self.thread_local_cache.clear()

            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA 캐시가 정리되었습니다.")

            # 그 외 할당 캐시 정리
            with self.lock:
                if hasattr(self, "allocations"):
                    self.allocations.clear()

            # 가비지 컬렉터 실행
            gc.collect()

            logger.info("모든 메모리 캐시가 정리되었습니다.")
            return True

        except Exception as e:
            logger.error(f"캐시 정리 중 오류 발생: {str(e)}")
            return False

    def _cleanup_internal(self):
        """내부 정리 로직"""
        try:
            # 정리 스레드 종료
            if hasattr(self, "stop_event"):
                self.stop_event.set()

            # 실제 정리 작업 수행
            self._perform_cleanup()

            logger.info("내부 정리 완료")
        except Exception as e:
            # 소멸자에서 호출될 수 있으므로 최대한 예외 안전하게 처리
            try:
                logger.error(f"내부 정리 중 오류: {str(e)}")
            except Exception:
                pass

    def _perform_cleanup(self):
        """자원 정리 작업"""
        try:
            # 시작 시간 기록
            start_time = time.time()

            # 1. CUDA 캐시 정리
            if torch.cuda.is_available():
                # 캐시 정리 전 메모리 사용량
                before_memory = torch.cuda.memory_allocated() / (1024 * 1024)

                # 캐시 정리
                torch.cuda.empty_cache()

                # 동기화 및 캐시 정리 후 메모리 사용량
                torch.cuda.synchronize()
                after_memory = torch.cuda.memory_allocated() / (1024 * 1024)

                # 메모리 변화 로깅
                if before_memory - after_memory > 5:  # 5MB 이상 절약된 경우만 로깅
                    logger.debug(
                        f"CUDA 캐시 정리: {before_memory:.1f}MB → {after_memory:.1f}MB "
                        f"(절약: {before_memory - after_memory:.1f}MB)"
                    )

            # 2. 가비지 컬렉션 강제 실행
            gc.collect()

            # 3. 메모리 풀 정리 (20% 확률)
            if hasattr(self, "memory_pool") and random.random() < 0.2:
                self.memory_pool.cleanup()

            # 4. 스레드 로컬 캐시 정리
            if hasattr(self, "thread_local_cache"):
                self.thread_local_cache._cleanup()

            # 소요 시간 계산 및 로깅
            duration = time.time() - start_time
            logger.debug(f"정리 작업 완료 (소요 시간: {duration:.2f}초)")

        except Exception as e:
            logger.error(f"정리 작업 중 오류 발생: {str(e)}")

    def _process_allocation_task(self, task):
        """할당 작업 처리"""
        if task is None:
            return  # 종료 신호

        try:
            # 작업 처리 로직
            pass  # 실제 구현 내용은 여기에 추가
        except Exception as e:
            logger.error(f"할당 작업 처리 중 오류: {str(e)}")

    def _process_deallocation_task(self, task):
        """해제 작업 처리"""
        if task is None:
            return  # 종료 신호

        try:
            # 작업 처리 로직
            pass  # 실제 구현 내용은 여기에 추가
        except Exception as e:
            logger.error(f"해제 작업 처리 중 오류: {str(e)}")

    def __del__(self):
        """소멸자"""
        try:
            # 정리 스레드 종료
            if hasattr(self, "stop_event"):
                self.stop_event.set()

            # 메모리 정리
            self._cleanup_memory()

            # 스레드 풀 종료
            if hasattr(self, "executor") and self.executor is not None:
                self.executor.shutdown(wait=False)

            logger.debug("메모리 관리자 종료")
        except Exception as e:
            logger.error(f"메모리 관리자 종료 중 오류: {str(e)}")

    def _cleanup_memory(self):
        """메모리 정리"""
        try:
            with self.lock:
                start_time = time.time()

                # 캐시된 텐서 정리
                for key in list(self.allocations.keys()):
                    del self.allocations[key]
                self.allocations.clear()

                # 텐서 캐시 정리
                if hasattr(self, "tensor_cache"):
                    self.tensor_cache.clear()

                # CUDA 캐시 정리
                if torch.cuda.is_available():
                    # 진행 중인 CUDA 작업 동기화
                    torch.cuda.synchronize()

                    # CUDA 캐시 정리
                    torch.cuda.empty_cache()

                    # GPU 메모리 상태 로깅
                    memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                    memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    logger.debug(
                        f"CUDA 메모리 정리 후: 할당={memory_allocated:.1f}MB, 예약={memory_reserved:.1f}MB"
                    )

                # 가비지 컬렉션 실행
                gc.collect()

                # 풀 정리 고려
                if hasattr(self, "memory_pool"):
                    if random.random() < 0.1:  # 10% 확률로 풀도 완전 정리
                        self.memory_pool.cleanup()
                        logger.debug("메모리 풀 정리 완료")

                duration = time.time() - start_time
                logger.info(f"메모리 정리 완료 (소요 시간: {duration:.2f}초)")

                return True
        except Exception as e:
            logger.error(f"메모리 정리 실패: {str(e)}")
            return False

    def check_memory_for_batch(
        self, batch_size: int, operation_name: str = "", memory_type: str = "gpu"
    ) -> bool:
        """
        특정 배치 크기의 작업에 충분한 메모리가 있는지 확인

        Args:
            batch_size: 확인할 배치 크기
            operation_name: 작업 이름 (로깅용)
            memory_type: 확인할 메모리 유형 ("gpu" 또는 "ram")

        Returns:
            충분한 메모리가 있으면 True, 아니면 False
        """
        try:
            # 현재 메모리 사용량 확인
            current_usage = self.get_memory_usage(memory_type)

            # 배치 크기에 따른 예상 메모리 사용량 추정
            # 0.01은 배치당 메모리 증가율 추정치 (실제로는 모델과 데이터에 따라 조정 필요)
            estimated_increase = 0.01 * batch_size

            # 최대 허용 메모리 사용량
            max_allowed = self.config.max_memory_usage

            # 예상 메모리 사용량이 허용 범위 내인지 확인
            has_enough_memory = (current_usage + estimated_increase) <= max_allowed

            if not has_enough_memory:
                logger.warning(
                    f"메모리 부족 예상: {operation_name}, 현재={current_usage:.2f}, "
                    f"예상 증가={estimated_increase:.2f}, 최대 허용={max_allowed:.2f}"
                )

                # 부족한 경우 메모리 정리 시도
                self._cleanup_internal()

                # 정리 후 다시 확인
                current_usage = self.get_memory_usage(memory_type)
                has_enough_memory = (current_usage + estimated_increase) <= max_allowed

                if has_enough_memory:
                    logger.info(
                        f"메모리 정리 후 충분한 메모리 확보됨: {operation_name}"
                    )

            return has_enough_memory

        except Exception as e:
            logger.error(f"메모리 확인 중 오류 발생: {str(e)}")
            return True  # 오류 발생 시 기본적으로 진행 허용

    def get_safe_batch_size(
        self, max_batch_size: Optional[int] = None, memory_type: str = "gpu"
    ) -> int:
        """
        현재 메모리 상태에 따라 안전하게 사용할 수 있는 배치 크기를 계산합니다.

        Args:
            max_batch_size: 최대 배치 크기 제한 (기본값: None, 설정의 max_batch_size 사용)
            memory_type: 확인할 메모리 유형 ("gpu" 또는 "ram")

        Returns:
            안전한 배치 크기
        """
        try:
            # 최대 배치 크기가 지정되지 않은 경우 설정에서 가져옴
            if max_batch_size is None:
                max_batch_size = self.config.max_batch_size

            # 현재 메모리 사용량 확인
            current_usage = self.get_memory_usage(memory_type)

            # 최대 허용 메모리 사용량
            max_allowed = self.config.max_memory_usage

            # 사용 가능한 메모리 비율
            available_memory_ratio = max(0.0, (max_allowed - current_usage))

            # 메모리가 부족한 경우 정리 시도
            if available_memory_ratio < 0.1:  # 10% 미만의 메모리만 사용 가능한 경우
                logger.warning(
                    f"메모리 부족, 정리 시도 (사용 가능: {available_memory_ratio:.1%})"
                )
                self._cleanup_internal()
                current_usage = self.get_memory_usage(memory_type)
                available_memory_ratio = max(0.0, (max_allowed - current_usage))
                logger.info(f"정리 후 사용 가능한 메모리: {available_memory_ratio:.1%}")

            # 배치당 예상 메모리 사용률
            estimated_memory_per_batch = 0.01  # 배치당 메모리 증가율 추정치

            # 안전한 배치 크기 계산 (최소한 설정의 min_batch_size 이상)
            safe_batch_size = max(
                self.config.min_batch_size,
                min(
                    max_batch_size,
                    int(
                        available_memory_ratio / estimated_memory_per_batch * 0.8
                    ),  # 80%만 사용
                ),
            )

            # 로그 기록
            logger.debug(
                f"안전한 배치 크기: {safe_batch_size}, "
                f"사용 가능한 메모리: {available_memory_ratio:.1%}, "
                f"최대 배치 크기: {max_batch_size}"
            )

            return safe_batch_size

        except Exception as e:
            logger.error(f"안전한 배치 크기 계산 중 오류 발생: {str(e)}")
            # 오류 발생 시 보수적인 값 반환
            return max(1, min(8, getattr(self.config, "min_batch_size", 1)))


def cleanup_analysis():
    """명시적 메모리 해제 함수"""
    try:
        # Python 가비지 컬렉션 실행
        gc.collect()

        # CUDA 메모리 해제
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 메모리 사용량 로깅
        memory_info = psutil.virtual_memory()
        logger.info(
            f"메모리 정리 완료 - 사용량: {memory_info.percent:.1f}%, 사용가능: {memory_info.available / (1024**3):.1f}GB"
        )

    except Exception as e:
        logger.error(f"메모리 정리 중 오류: {str(e)}")


@contextmanager
def memory_managed_analysis():
    """메모리 관리 컨텍스트 매니저"""
    try:
        # 시작 전 메모리 상태 기록
        start_memory = psutil.virtual_memory()
        logger.info(f"분석 시작 - 메모리 사용량: {start_memory.percent:.1f}%")

        yield

    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        raise
    finally:
        # 종료 후 메모리 정리
        cleanup_analysis()

        # 최종 메모리 상태 기록
        end_memory = psutil.virtual_memory()
        logger.info(f"분석 완료 - 메모리 사용량: {end_memory.percent:.1f}%")
