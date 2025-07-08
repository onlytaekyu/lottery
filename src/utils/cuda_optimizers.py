"""
CUDA 최적화 관리자

AMP (Automatic Mixed Precision), TF32, Cudnn Benchmark 등
핵심적인 CUDA 최적화 설정을 관리하고 컨텍스트를 제공합니다.
"""

import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import threading

from .unified_logging import get_logger
from .factory import get_singleton_instance

logger = get_logger(__name__)


@dataclass
class CudaConfig:
    """CUDA 최적화 설정"""
    use_amp: bool = True
    use_cudnn_benchmark: bool = True
    enable_tf32: bool = True

    def __post_init__(self):
        if not torch.cuda.is_available():
            self.use_amp = False
        if self.use_cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.enable_tf32
            torch.backends.cudnn.allow_tf32 = self.enable_tf32


class CudaManager:
    """CUDA 최적화 관리 및 컨텍스트 제공"""
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[CudaConfig] = None):
        if hasattr(self, "initialized"):
            return

        self.config = config or CudaConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)
        self.initialized = True
        logger.info(f"✅ CUDA Manager 초기화 (AMP: {self.config.use_amp}, TF32: {self.config.enable_tf32})")

    @property
    def is_available(self) -> bool:
        """CUDA 사용 가능 여부"""
        return torch.cuda.is_available()

    @contextmanager
    def amp_context(self, enabled: Optional[bool] = None):
        """
        AMP (Automatic Mixed Precision) 컨텍스트를 제공합니다.
        'enabled' 인수로 컨텍스트의 활성화 여부를 오버라이드할 수 있습니다.
        """
        use_amp = self.config.use_amp if enabled is None else enabled
        with torch.cuda.amp.autocast(enabled=use_amp):
            yield

    def set_tf32_enabled(self, enabled: bool):
        """TF32 사용 여부를 동적으로 설정합니다."""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.backends.cudnn.allow_tf32 = enabled
            self.config.enable_tf32 = enabled
            logger.info(f"TF32 사용 설정 변경: {'활성화' if enabled else '비활성화'}")

    def get_stats(self) -> Dict[str, Any]:
        """현재 CUDA 설정을 반환합니다."""
        return {
            "cuda_available": self.is_available,
            "device_name": torch.cuda.get_device_name(0) if self.is_available else "N/A",
            "amp_enabled": self.config.use_amp,
            "cudnn_benchmark_enabled": self.config.use_cudnn_benchmark,
            "tf32_enabled": self.config.enable_tf32,
        }


def get_cuda_optimizer(config: Optional[CudaConfig] = None) -> CudaManager:
    """CudaManager의 싱글톤 인스턴스를 반환합니다."""
    # GPUOptimizer -> CudaManager로 이름 변경
    return get_singleton_instance(CudaManager, config=config)


class AMPTrainer:
    """AMP를 사용한 학습을 위한 간단한 래퍼"""

    def __init__(self, config: Optional[CudaConfig] = None):
        self.cuda_manager = get_cuda_optimizer(config)

    def train_step(self, model, optimizer, loss_fn, batch_data):
        """AMP를 적용하여 한 스텝의 학습을 진행합니다."""
        inputs, targets = batch_data
        inputs = inputs.to(self.cuda_manager.device)
        targets = targets.to(self.cuda_manager.device)

        with self.cuda_manager.amp_context():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        self.cuda_manager.amp_scaler.scale(loss).backward()
        self.cuda_manager.amp_scaler.step(optimizer)
        self.cuda_manager.amp_scaler.update()

        return loss.item()
