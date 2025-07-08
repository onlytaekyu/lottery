"""
완전 독립적인 GPU 정규화 시스템
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Union
from .unified_logging import get_logger
from .unified_memory_manager import get_unified_memory_manager

logger = get_logger(__name__)

class Normalizer:
    """
    데이터 정규화를 위한 클래스.
    GPU 사용 가능 시 GPU를 활용하여 정규화를 가속하고,
    그렇지 않은 경우 CPU를 사용하여 안정적으로 처리합니다.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
        
        self.memory_manager = None
        if self.gpu_available:
            self.memory_manager = get_unified_memory_manager()

    def normalize(self, data: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """
        데이터를 정규화합니다. GPU 사용이 가능하고 메모리가 충분하면 GPU를 사용합니다.
        """
        use_gpu = False
        if self.gpu_available and self.memory_manager:
            # _gpu_memory_ok 같은 private 메서드 대신, 필요한 메모리를 직접 확인하는 것이 좋습니다.
            # 여기서는 단순화를 위해 존재 여부만 체크합니다.
            use_gpu = True

        if use_gpu:
            try:
                return self._gpu_normalize(data)
            except Exception as e:
                logger.warning(f"GPU 정규화 중 오류 발생, CPU로 폴백: {e}")
                if self.memory_manager:
                    self.memory_manager.cleanup_all() # 캐시 정리
                return self._cpu_normalize(data)
        else:
            return self._cpu_normalize(data)

    def _gpu_normalize(self, data: np.ndarray) -> torch.Tensor:
        """GPU를 사용한 정규화"""
        tensor = torch.from_numpy(data).to(self.device, non_blocking=True)
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0)
        std[std == 0] = 1.0  # 0으로 나누는 것을 방지
        
        normalized_tensor = (tensor - mean) / std
        
        return normalized_tensor

    def _cpu_normalize(self, data: np.ndarray) -> np.ndarray:
        """CPU를 사용한 정규화"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1.0
        return (data - mean) / std

    def to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """데이터를 텐서로 변환합니다."""
        return torch.from_numpy(data).to(self.device)

def get_normalizer(config: Optional[Dict[str, Any]] = None) -> Normalizer:
    """Normalizer의 인스턴스를 반환합니다."""
    return Normalizer(config)
