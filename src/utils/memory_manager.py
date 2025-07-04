"""메모리 관리 모듈 (성능 최적화 버전)

GPU 메모리 최우선 관리와 핵심 기능만 제공하는 간소화된 메모리 관리 시스템
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import threading
import time
import gc
import torch
import psutil
import platform
from contextlib import contextmanager
import os

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
    """메모리 관리 설정 (간소화)"""

    # 메모리 사용률 제한
    max_memory_usage: float = 0.85  # 85% 제한

    # 배치 크기 설정
    optimal_batch_size: int = 256
    min_batch_size: int = 16
    max_batch_size: int = 512

    # 정리 간격
    cleanup_interval: float = 60.0  # 1분

    # GPU 메모리 풀 사전 할당
    preallocation_fraction: float = 0.5

    def __post_init__(self):
        """설정 검증"""
        self.max_memory_usage = max(0.1, min(self.max_memory_usage, 0.95))
        self.min_batch_size = max(1, min(self.min_batch_size, self.max_batch_size))
        self.max_batch_size = max(self.min_batch_size, self.max_batch_size)
        self.optimal_batch_size = max(
            self.min_batch_size, min(self.optimal_batch_size, self.max_batch_size)
        )


class MemoryManager:
    """메모리 관리자 (간소화)"""

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
        self._cleanup_thread = None
        self._running = False
        self._last_cleanup = time.time()
        self._initialized = True

        # GPU 메모리 풀 초기화
        if torch.cuda.is_available():
            self._setup_gpu_memory_pool()

        # 자동 정리 시작
        self._start_cleanup_thread()

        logger.info(f"메모리 관리자 초기화 완료: {self.device}")

    def _setup_gpu_memory_pool(self):
        """GPU 메모리 풀 설정"""
        try:
            # GPU 메모리 사전 할당
            torch.cuda.set_per_process_memory_fraction(
                self.config.preallocation_fraction
            )

            # 메모리 풀 워밍업
            dummy_tensor = torch.zeros(1024, 1024, device=self.device)
            del dummy_tensor
            torch.cuda.empty_cache()

            logger.info("GPU 메모리 풀 설정 완료")
        except Exception as e:
            logger.error(f"GPU 메모리 풀 설정 실패: {e}")

    def _start_cleanup_thread(self):
        """자동 정리 스레드 시작"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._running = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop, daemon=True
            )
            self._cleanup_thread.start()

    def _cleanup_loop(self):
        """자동 정리 루프"""
        while self._running:
            try:
                time.sleep(self.config.cleanup_interval)
                if self.should_cleanup():
                    self.cleanup()
            except Exception as e:
                logger.error(f"자동 정리 오류: {e}")

    def should_cleanup(self) -> bool:
        """정리 필요 여부 확인"""
        try:
            # GPU 메모리 사용률 확인
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_usage = allocated / total if total > 0 else 0

                if gpu_usage > self.config.max_memory_usage:
                    return True

            # CPU 메모리 사용률 확인
            cpu_usage = psutil.virtual_memory().percent / 100
            if cpu_usage > self.config.max_memory_usage:
                return True

            return False

        except Exception as e:
            logger.error(f"메모리 사용률 확인 실패: {e}")
            return True

    def cleanup(self) -> None:
        """메모리 정리"""
        try:
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # CPU 메모리 정리
            gc.collect()

            self._last_cleanup = time.time()
            logger.debug("메모리 정리 완료")

        except Exception as e:
            logger.error(f"메모리 정리 실패: {e}")

    def get_memory_usage(self, memory_type: str = "gpu") -> float:
        """메모리 사용률 반환"""
        try:
            if memory_type == "gpu" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return allocated / total if total > 0 else 0
            elif memory_type == "cpu":
                return psutil.virtual_memory().percent / 100
            else:
                return 0.0
        except Exception as e:
            logger.error(f"메모리 사용률 조회 실패: {e}")
            return 0.0

    def get_available_memory(self, memory_type: str = "gpu") -> float:
        """사용 가능한 메모리 반환 (GB)"""
        try:
            if memory_type == "gpu" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return (total - allocated) / (1024**3)  # GB
            elif memory_type == "cpu":
                return psutil.virtual_memory().available / (1024**3)  # GB
            else:
                return 0.0
        except Exception as e:
            logger.error(f"사용 가능한 메모리 조회 실패: {e}")
            return 0.0

    def check_gpu_memory(self, required_bytes: int) -> bool:
        """GPU 메모리 충분성 확인"""
        if not torch.cuda.is_available():
            return False

        try:
            available_bytes = self.get_available_memory("gpu") * (1024**3)
            return available_bytes >= required_bytes
        except Exception as e:
            logger.error(f"GPU 메모리 확인 실패: {e}")
            return False

    def get_safe_batch_size(
        self, max_batch_size: Optional[int] = None, memory_type: str = "gpu"
    ) -> int:
        """안전한 배치 크기 반환"""
        try:
            if max_batch_size is None:
                max_batch_size = self.config.max_batch_size

            # 메모리 사용률 기반 배치 크기 조정
            memory_usage = self.get_memory_usage(memory_type)

            if memory_usage > 0.8:  # 80% 이상 사용 시
                return max(self.config.min_batch_size, max_batch_size // 4)
            elif memory_usage > 0.6:  # 60% 이상 사용 시
                return max(self.config.min_batch_size, max_batch_size // 2)
            else:
                return max_batch_size

        except Exception as e:
            logger.error(f"안전한 배치 크기 계산 실패: {e}")
            return self.config.min_batch_size

    @contextmanager
    def allocation_scope(self):
        """메모리 할당 스코프"""
        try:
            yield
        finally:
            if self.should_cleanup():
                self.cleanup()

    @contextmanager
    def batch_processing(self):
        """배치 처리 스코프"""
        try:
            yield
        finally:
            # 배치 처리 후 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 반환"""
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

        return info

    def shutdown(self):
        """메모리 관리자 종료"""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)

        # 최종 정리
        self.cleanup()
        logger.info("메모리 관리자 종료 완료")

    def __del__(self):
        """소멸자"""
        try:
            self.shutdown()
        except:
            pass


# 편의 함수들
def get_memory_manager(config: Optional[MemoryConfig] = None) -> MemoryManager:
    """메모리 관리자 반환"""
    return MemoryManager(config)


@contextmanager
def memory_managed_analysis():
    """메모리 관리 분석 컨텍스트"""
    manager = get_memory_manager()
    try:
        with manager.allocation_scope():
            yield manager
    finally:
        manager.cleanup()


def cleanup_analysis():
    """분석 후 정리"""
    manager = get_memory_manager()
    manager.cleanup()
