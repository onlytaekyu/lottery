"""
개인용 로또 예측에 최적화된 간단한 메모리 관리자
"""

import torch
import gc
import psutil
import time
from typing import Optional, Dict, Any, Tuple, Union

from .unified_logging import get_logger
from .error_handler import get_error_handler
from .unified_config import get_config
from .factory import get_singleton_instance

logger = get_logger(__name__)
error_handler = get_error_handler()


class GPUMemoryManager:
    """개인용 간단 메모리 관리자 (서버 기능 제거)"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()

        # 설정에서 값 로드, 없으면 기본값 사용
        config = get_config("main").get_nested("utils.memory_manager", {})
        self.gpu_threshold = config.get(
            "gpu_threshold", 0.8
        )  # 80% 이상 사용시 CPU 폴백
        self.log_interval_seconds = config.get(
            "log_interval_seconds", 600
        )  # 10분에 한 번 로그

        # 간단한 카운터만 (통계 최소화)
        self.stats = {"gpu_allocations": 0, "cpu_fallbacks": 0}

        self.last_log_time = time.time()

        # 메모리 풀링 시스템 추가
        self.memory_pool = {}  # 크기별 메모리 풀
        self.pool_enabled = config.get("enable_memory_pool", True)
        self.max_pool_size = config.get("max_pool_size", 10)  # 풀당 최대 텐서 수

        logger.info(
            f"✅ 개인용 메모리 관리자 초기화 (장치: {self.device}, GPU 임계값: {self.gpu_threshold * 100}%, 메모리 풀: {'활성화' if self.pool_enabled else '비활성화'})"
        )

    @error_handler.auto_recoverable(max_retries=2, delay=0.5)
    def smart_allocate(
        self, size: Union[int, Tuple], prefer_gpu: bool = True
    ) -> torch.Tensor:
        """스마트 메모리 할당 (개인용 간소화)"""

        if isinstance(size, int):
            shape = (size,)
        else:
            shape = size

        # 메모리 풀에서 재사용 가능한 텐서 확인
        if self.pool_enabled:
            pooled_tensor = self._get_from_pool(shape, prefer_gpu)
            if pooled_tensor is not None:
                return pooled_tensor

        # GPU 사용 조건: GPU 가능, 사용자 선호, 메모리 여유
        if prefer_gpu and self.gpu_available and self._gpu_memory_ok():
            try:
                tensor = torch.zeros(shape, device=self.device)
                self.stats["gpu_allocations"] += 1
                return tensor
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU 메모리 부족, CPU 사용")
                self.stats["cpu_fallbacks"] += 1
                return torch.zeros(shape, device="cpu")
        else:
            # CPU 할당
            return torch.zeros(shape, device="cpu")

    def _get_from_pool(self, shape: Tuple, prefer_gpu: bool) -> Optional[torch.Tensor]:
        """메모리 풀에서 재사용 가능한 텐서 반환"""
        device_key = "gpu" if prefer_gpu and self.gpu_available else "cpu"
        pool_key = f"{device_key}_{shape}"

        if pool_key in self.memory_pool and self.memory_pool[pool_key]:
            tensor = self.memory_pool[pool_key].pop()
            tensor.zero_()  # 텐서 초기화
            return tensor

        return None

    def return_to_pool(self, tensor: torch.Tensor):
        """사용 완료된 텐서를 메모리 풀에 반환"""
        if not self.pool_enabled:
            return

        device_key = "gpu" if tensor.device.type == "cuda" else "cpu"
        pool_key = f"{device_key}_{tensor.shape}"

        if pool_key not in self.memory_pool:
            self.memory_pool[pool_key] = []

        # 풀 크기 제한
        if len(self.memory_pool[pool_key]) < self.max_pool_size:
            self.memory_pool[pool_key].append(tensor)

    def clear_memory_pool(self):
        """메모리 풀 정리"""
        if self.memory_pool:
            total_tensors = sum(len(pool) for pool in self.memory_pool.values())
            self.memory_pool.clear()
            logger.info(f"메모리 풀 정리 완료: {total_tensors}개 텐서 해제")

    def _gpu_memory_ok(self) -> bool:
        """GPU 메모리 상태 간단 체크"""
        if not self.gpu_available:
            return False

        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated / total) < self.gpu_threshold
        except:
            return False

    @error_handler.auto_recoverable(max_retries=2, delay=0.1)
    def check_available_memory(self, size_in_bytes: int, for_gpu: bool = True) -> bool:
        """
        특정 크기의 데이터를 할당할 수 있는지 미리 확인합니다.
        """
        if not for_gpu:
            # CPU 메모리 확인 (psutil 사용)
            available_cpu = psutil.virtual_memory().available
            return size_in_bytes < available_cpu

        if not self.gpu_available:
            return False

        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated + size_in_bytes) / total < self.gpu_threshold
        except:
            return False

    def cleanup(self):
        """간단한 메모리 정리"""
        # 메모리 풀 정리
        self.clear_memory_pool()

        gc.collect()
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_simple_stats(self) -> Dict[str, Any]:
        """간단한 통계만"""
        cpu_memory = psutil.virtual_memory()
        result = {
            "gpu_allocations": self.stats["gpu_allocations"],
            "cpu_fallbacks": self.stats["cpu_fallbacks"],
            "cpu_memory_percent": cpu_memory.percent,
        }

        if self.gpu_available:
            try:
                allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                result["gpu_memory_gb"] = f"{allocated:.1f}/{total:.1f}"
            except:
                pass

        # 메모리 풀 통계 추가
        if self.pool_enabled:
            total_pooled = sum(len(pool) for pool in self.memory_pool.values())
            result["memory_pool_tensors"] = total_pooled
            result["memory_pool_keys"] = len(self.memory_pool)

        return result

    def get_gpu_usage(self) -> Dict[str, Any]:
        """GPU 메모리 사용량 조회 (개인용 간소화)"""
        if not self.gpu_available:
            return {"status": "GPU not available"}

        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
        }

    def release_cuda_cache(self, extra_msg: str = ""):
        """
        CUDA 캐시를 정리하고 로그를 남깁니다. (중앙 관리)
        """
        if self.gpu_available:
            try:
                gc.collect()
                torch.cuda.empty_cache()
                msg = "CUDA 캐시가 정리되었습니다."
                if extra_msg:
                    msg += f" (트리거: {extra_msg})"
                logger.debug(msg)
            except Exception as e:
                logger.error(f"CUDA 캐시 정리 중 오류 발생: {e}")

    def _log_status(self):
        """메모리 상태 로깅"""
        if self.gpu_available:
            usage = self.get_gpu_usage()
            logger.info(f"GPU 메모리 사용량: {usage}")


def get_memory_manager() -> GPUMemoryManager:
    """간단한 메모리 관리자 반환"""
    return get_singleton_instance(GPUMemoryManager)


# 하위 호환성
get_optimized_memory_manager = get_memory_manager


# 하위 호환성 래퍼 추가
class OptimizedMemoryManager(GPUMemoryManager):
    """하위 호환성 래퍼 - __init__.py에서 참조되는 클래스"""

    pass


def get_optimized_memory_manager():
    """하위 호환성 함수"""
    return get_memory_manager()
