"""메모리 관리 모듈 (GPU 우선순위 버전)

GPU 메모리 최우선 관리와 스마트 폴백 시스템을 제공하는 고성능 메모리 관리 시스템
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Tuple
import threading
import time
import gc
import torch
import psutil
import platform
from contextlib import contextmanager
import os
import numpy as np

from .unified_logging import get_logger

logger = get_logger(__name__)

# CUDA 메모리 설정
if torch.cuda.is_available() and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    if platform.system() == "Windows":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,max_split_size_mb:512"
        )


@dataclass
class MemoryConfig:
    """메모리 관리 설정"""

    max_memory_usage: float = 0.85
    optimal_batch_size: int = 256
    min_batch_size: int = 16
    max_batch_size: int = 512
    cleanup_interval: float = 60.0
    preallocation_fraction: float = 0.5
    gpu_fallback_threshold: float = 0.9  # GPU 메모리 사용률 임계값
    auto_fallback_enabled: bool = True

    def __post_init__(self):
        self.max_memory_usage = max(0.1, min(self.max_memory_usage, 0.95))
        self.min_batch_size = max(1, min(self.min_batch_size, self.max_batch_size))
        self.max_batch_size = max(self.min_batch_size, self.max_batch_size)
        self.optimal_batch_size = max(
            self.min_batch_size, min(self.optimal_batch_size, self.max_batch_size)
        )
        self.gpu_fallback_threshold = max(0.5, min(self.gpu_fallback_threshold, 0.99))


class SmartMemoryAllocator:
    """스마트 메모리 할당기 - GPU 우선순위 with 자동 폴백"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.gpu_available = torch.cuda.is_available()
        self.fallback_history = []  # 폴백 히스토리 추적

        if self.gpu_available:
            self.logger.info("GPU 메모리 할당기 초기화 완료")
        else:
            self.logger.info("CPU 메모리 할당기 초기화 완료")

    def smart_allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        prefer_gpu: bool = True,
    ) -> torch.Tensor:
        """스마트 텐서 할당 - GPU 우선순위 with 자동 폴백"""

        # 메모리 사용량 추정
        element_size = torch.tensor([], dtype=dtype).element_size()
        estimated_memory = np.prod(shape) * element_size

        # GPU 할당 시도
        if prefer_gpu and self.gpu_available:
            try:
                if self._can_allocate_on_gpu(estimated_memory):
                    tensor = torch.zeros(shape, dtype=dtype, device="cuda")
                    self.logger.debug(
                        f"GPU 메모리 할당 성공: {shape}, {estimated_memory/1024**2:.1f}MB"
                    )
                    return tensor
                else:
                    self.logger.warning(
                        f"GPU 메모리 부족 - CPU로 폴백: {estimated_memory/1024**2:.1f}MB"
                    )
                    self._record_fallback("insufficient_memory", estimated_memory)
            except torch.cuda.OutOfMemoryError as e:
                self.logger.warning(f"GPU 메모리 할당 실패 - CPU로 폴백: {e}")
                self._record_fallback("cuda_oom", estimated_memory)
                # GPU 캐시 정리 시도
                torch.cuda.empty_cache()

        # CPU 할당 (폴백)
        try:
            tensor = torch.zeros(shape, dtype=dtype, device="cpu")
            self.logger.debug(
                f"CPU 메모리 할당 성공: {shape}, {estimated_memory/1024**2:.1f}MB"
            )
            return tensor
        except Exception as e:
            self.logger.error(f"CPU 메모리 할당 실패: {e}")
            raise MemoryError(f"메모리 할당 실패: {estimated_memory/1024**2:.1f}MB")

    def _can_allocate_on_gpu(self, estimated_memory: int) -> bool:
        """GPU 메모리 할당 가능 여부 확인"""
        if not self.gpu_available:
            return False

        try:
            # 현재 GPU 메모리 사용률 확인
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            usage_ratio = allocated / total

            # 임계값 초과 시 폴백
            if usage_ratio > self.config.gpu_fallback_threshold:
                return False

            # 사용 가능한 메모리 확인
            available = total - allocated
            safety_margin = total * 0.1  # 10% 여유분

            return (estimated_memory + safety_margin) < available

        except Exception as e:
            self.logger.error(f"GPU 메모리 확인 실패: {e}")
            return False

    def _record_fallback(self, reason: str, memory_size: int):
        """폴백 히스토리 기록"""
        fallback_info = {
            "timestamp": time.time(),
            "reason": reason,
            "memory_size_mb": memory_size / (1024**2),
            "gpu_usage": self._get_gpu_usage(),
        }
        self.fallback_history.append(fallback_info)

        # 히스토리 크기 제한 (최근 100개만 보관)
        if len(self.fallback_history) > 100:
            self.fallback_history = self.fallback_history[-100:]

    def _get_gpu_usage(self) -> float:
        """GPU 사용률 반환"""
        if not self.gpu_available:
            return 0.0

        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total if total > 0 else 0.0
        except Exception:
            return 0.0

    def get_fallback_statistics(self) -> Dict[str, Any]:
        """폴백 통계 반환"""
        if not self.fallback_history:
            return {"total_fallbacks": 0, "reasons": {}}

        reasons = {}
        for entry in self.fallback_history:
            reason = entry["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1

        return {
            "total_fallbacks": len(self.fallback_history),
            "reasons": reasons,
            "recent_fallbacks": self.fallback_history[-10:],  # 최근 10개
            "average_fallback_memory_mb": sum(
                entry["memory_size_mb"] for entry in self.fallback_history
            )
            / len(self.fallback_history),
        }

    def optimize_tensor_placement(
        self, tensor: torch.Tensor, prefer_gpu: bool = True
    ) -> torch.Tensor:
        """텐서 배치 최적화"""
        if not prefer_gpu or not self.gpu_available:
            return tensor.cpu()

        # 이미 GPU에 있으면 그대로 반환
        if tensor.is_cuda:
            return tensor

        # GPU로 이동 시도
        try:
            if self._can_allocate_on_gpu(tensor.numel() * tensor.element_size()):
                return tensor.cuda()
            else:
                self.logger.debug("GPU 메모리 부족 - CPU에 유지")
                return tensor
        except Exception as e:
            self.logger.warning(f"GPU 이동 실패: {e}")
            return tensor


class MemoryManager:
    """메모리 관리자 (GPU 우선순위 버전)"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, config: Optional[MemoryConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[MemoryConfig] = None):
        if hasattr(self, "_initialized"):
            return
        self.config = config or MemoryConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._running = False
        self._last_cleanup = time.time()
        self._initialized = True

        # 스마트 메모리 할당기 초기화
        self.smart_allocator = SmartMemoryAllocator(self.config)

        if torch.cuda.is_available():
            self._setup_gpu_memory_pool()
        logger.info(f"메모리 관리자 초기화 완료: {self.device}")

    def _setup_gpu_memory_pool(self):
        try:
            torch.cuda.set_per_process_memory_fraction(
                self.config.preallocation_fraction
            )
            dummy_tensor = torch.zeros(1024, 1024, device=self.device)
            del dummy_tensor
            torch.cuda.empty_cache()
            logger.info("GPU 메모리 풀 설정 완료")
        except Exception as e:
            logger.error(f"GPU 메모리 풀 설정 실패: {e}")

    def smart_memory_allocation(
        self,
        tensor_size: Union[int, Tuple[int, ...]],
        dtype: torch.dtype = torch.float32,
        prefer_gpu: bool = True,
    ) -> torch.Tensor:
        """스마트 메모리 할당 - GPU 부족 시 자동 CPU 폴백"""
        if isinstance(tensor_size, int):
            shape = (tensor_size,)
        else:
            shape = tensor_size

        return self.smart_allocator.smart_allocate_tensor(shape, dtype, prefer_gpu)

    def allocate_gpu_memory(
        self,
        tensor_size: Union[int, Tuple[int, ...]],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """GPU 메모리 할당 (실패 시 예외 발생)"""
        if isinstance(tensor_size, int):
            shape = (tensor_size,)
        else:
            shape = tensor_size

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다")

        try:
            return torch.zeros(shape, dtype=dtype, device="cuda")
        except torch.cuda.OutOfMemoryError as e:
            # 메모리 정리 후 재시도
            torch.cuda.empty_cache()
            try:
                return torch.zeros(shape, dtype=dtype, device="cuda")
            except torch.cuda.OutOfMemoryError:
                raise torch.cuda.OutOfMemoryError(f"GPU 메모리 할당 실패: {shape}")

    def allocate_cpu_memory(
        self,
        tensor_size: Union[int, Tuple[int, ...]],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """CPU 메모리 할당"""
        if isinstance(tensor_size, int):
            shape = (tensor_size,)
        else:
            shape = tensor_size

        return torch.zeros(shape, dtype=dtype, device="cpu")

    def should_cleanup(self) -> bool:
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_usage = allocated / total if total > 0 else 0
                if gpu_usage > self.config.max_memory_usage:
                    return True
            cpu_usage = psutil.virtual_memory().percent / 100
            return cpu_usage > self.config.max_memory_usage
        except Exception as e:
            logger.error(f"메모리 사용률 확인 실패: {e}")
            return True

    def cleanup(self) -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            self._last_cleanup = time.time()
            logger.debug("메모리 정리 완료")
        except Exception as e:
            logger.error(f"메모리 정리 실패: {e}")

    def get_memory_usage(self, memory_type: str = "gpu") -> float:
        try:
            if memory_type == "gpu" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return allocated / total if total > 0 else 0
            elif memory_type == "cpu":
                return psutil.virtual_memory().percent / 100
            return 0.0
        except Exception as e:
            logger.error(f"메모리 사용률 조회 실패: {e}")
            return 0.0

    def get_available_memory(self, memory_type: str = "gpu") -> float:
        try:
            if memory_type == "gpu" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return (total - allocated) / (1024**3)
            elif memory_type == "cpu":
                return psutil.virtual_memory().available / (1024**3)
            return 0.0
        except Exception as e:
            logger.error(f"사용 가능한 메모리 조회 실패: {e}")
            return 0.0

    def get_safe_batch_size(
        self, max_batch_size: Optional[int] = None, memory_type: str = "gpu"
    ) -> int:
        try:
            if max_batch_size is None:
                max_batch_size = self.config.max_batch_size
            memory_usage = self.get_memory_usage(memory_type)
            if memory_usage > 0.8:
                return max(self.config.min_batch_size, max_batch_size // 4)
            elif memory_usage > 0.6:
                return max(self.config.min_batch_size, max_batch_size // 2)
            return max_batch_size
        except Exception as e:
            logger.error(f"안전한 배치 크기 계산 실패: {e}")
            return self.config.min_batch_size

    @contextmanager
    def allocation_scope(self):
        try:
            yield
        finally:
            if self.should_cleanup():
                self.cleanup()

    @contextmanager
    def batch_processing(self):
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_memory_info(self) -> Dict[str, Any]:
        info = {
            "device": str(self.device),
            "cpu_usage": self.get_memory_usage("cpu"),
            "cpu_available_gb": self.get_available_memory("cpu"),
        }
        if torch.cuda.is_available():
            info.update(
                {
                    "gpu_usage": self.get_memory_usage("gpu"),
                    "gpu_available_gb": self.get_available_memory("gpu"),
                    "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                }
            )

        # 스마트 할당기 통계 추가
        if hasattr(self, "smart_allocator"):
            info["fallback_statistics"] = self.smart_allocator.get_fallback_statistics()

        return info

    def shutdown(self):
        self._running = False
        self.cleanup()
        logger.info("메모리 관리자 종료 완료")


def get_memory_manager(config: Optional[MemoryConfig] = None) -> MemoryManager:
    return MemoryManager(config)


@contextmanager
def memory_managed_analysis():
    manager = get_memory_manager()
    try:
        with manager.allocation_scope():
            yield manager
    finally:
        manager.cleanup()


def cleanup_analysis():
    get_memory_manager().cleanup()
