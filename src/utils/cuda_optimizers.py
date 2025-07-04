"""CUDA 최적화 모듈 (성능 최적화 버전)

GPU 최우선 연산 처리와 메모리 효율성에 집중한 간소화된 CUDA 최적화 시스템
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
import logging
from contextlib import contextmanager

# 로거 설정
logger = logging.getLogger(__name__)

# GPU 메모리 풀 초기화 상태
_memory_pool_initialized = False
_memory_pool_lock = threading.RLock()


@dataclass
class CudaConfig:
    """CUDA 최적화 설정 (간소화)"""

    # 배치 크기 설정
    batch_size: int = 256
    min_batch_size: int = 16
    max_batch_size: int = 512

    # 최적화 플래그
    use_amp: bool = True
    use_cudnn: bool = True

    def __post_init__(self):
        """CUDA 설정 초기화"""
        if not torch.cuda.is_available():
            logger.warning("CUDA 비활성화 - CPU 모드")
            self.use_amp = False
            self.use_cudnn = False
        else:
            self._setup_cuda()

    def _setup_cuda(self):
        """CUDA 최적화 설정"""
        try:
            # cuDNN 최적화
            if self.use_cudnn:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # GPU 메모리 풀 설정
            setup_cuda_memory_pool()

            logger.info("CUDA 최적화 설정 완료")
        except Exception as e:
            logger.error(f"CUDA 설정 실패: {e}")


class CUDAOptimizer:
    """CUDA 최적화기 (간소화)"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, config: Optional[CudaConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[CudaConfig] = None):
        if hasattr(self, "_initialized"):
            return

        self.config = config or CudaConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = True

        logger.info(f"CUDA 최적화기 초기화 완료: {self.device}")

    def is_cuda_available(self) -> bool:
        """CUDA 사용 가능 여부"""
        return torch.cuda.is_available()

    @contextmanager
    def device_context(self):
        """디바이스 컨텍스트 관리자"""
        if self.is_cuda_available():
            with torch.cuda.device(self.device):
                yield
        else:
            yield

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """모델 최적화"""
        if not self.is_cuda_available():
            return model

        try:
            # GPU로 이동
            model = model.to(self.device)

            # AMP 적용
            if self.config.use_amp:
                model = self._apply_amp(model)

            return model

        except Exception as e:
            logger.error(f"모델 최적화 실패: {e}")
            return model

    def _apply_amp(self, model: nn.Module) -> nn.Module:
        """AMP 적용"""
        try:
            # 모델을 half precision으로 변환
            return model.half()
        except Exception as e:
            logger.warning(f"AMP 적용 실패: {e}")
            return model

    def optimize_memory(self) -> bool:
        """GPU 메모리 최적화"""
        if not self.is_cuda_available():
            return False

        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
        except Exception as e:
            logger.error(f"메모리 최적화 실패: {e}")
            return False

    def get_memory_info(self) -> Dict[str, Any]:
        """GPU 메모리 정보"""
        if not self.is_cuda_available():
            return {"available": False}

        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

            return {
                "available": True,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "utilization": allocated / total if total > 0 else 0,
            }
        except Exception as e:
            logger.error(f"메모리 정보 조회 실패: {e}")
            return {"available": False}


class AMPTrainer:
    """AMP 트레이너 (간소화)"""

    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def use_amp(self) -> bool:
        """AMP 사용 여부"""
        return (
            torch.cuda.is_available()
            and self.config.get("use_amp", True)
            and self.scaler is not None
        )

    def train_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """AMP를 사용한 훈련 스텝"""

        if not self.use_amp:
            # 일반 훈련
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs, batch.get("labels"))
            loss.backward()
            optimizer.step()
            return {"loss": loss.item()}

        # AMP 훈련
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = loss_fn(outputs, batch.get("labels"))

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        return {"loss": loss.item()}

    def get_device(self) -> torch.device:
        """디바이스 반환"""
        return self.device


def setup_cuda_memory_pool():
    """CUDA 메모리 풀 설정"""
    global _memory_pool_initialized

    if _memory_pool_initialized or not torch.cuda.is_available():
        return

    with _memory_pool_lock:
        if _memory_pool_initialized:
            return

        try:
            # GPU 메모리 풀 설정
            torch.cuda.set_per_process_memory_fraction(0.8)  # 80% 사용
            _memory_pool_initialized = True
            logger.info("CUDA 메모리 풀 초기화 완료")

        except Exception as e:
            logger.error(f"CUDA 메모리 풀 설정 실패: {e}")


def get_optimal_batch_size(
    model: nn.Module, input_shape: tuple, max_memory_fraction: float = 0.8
) -> int:
    """최적 배치 크기 계산"""

    if not torch.cuda.is_available():
        return 32  # CPU 기본값

    try:
        # GPU 메모리 정보
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory * max_memory_fraction

        # 모델 크기 추정
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters())

        # 입력 데이터 크기 추정
        input_memory = 1
        for dim in input_shape:
            input_memory *= dim
        input_memory *= 4  # float32 기준

        # 최적 배치 크기 계산
        batch_size = int(available_memory / (model_memory + input_memory))

        # 범위 제한
        return max(16, min(batch_size, 512))

    except Exception as e:
        logger.error(f"배치 크기 계산 실패: {e}")
        return 32


# 편의 함수들
def get_cuda_optimizer(config: Optional[CudaConfig] = None) -> CUDAOptimizer:
    """CUDA 최적화기 반환"""
    return CUDAOptimizer(config)


def optimize_memory():
    """메모리 최적화 실행"""
    optimizer = get_cuda_optimizer()
    return optimizer.optimize_memory()


def get_device_info() -> Dict[str, Any]:
    """디바이스 정보 반환"""
    optimizer = get_cuda_optimizer()
    return optimizer.get_memory_info()
