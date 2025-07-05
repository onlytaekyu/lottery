"""
완전 독립적인 GPU 정규화 시스템
"""

import numpy as np
import torch
from typing import Union, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .unified_logging import get_logger
from .unified_config import get_config

# from .memory_manager import get_memory_manager # 지연 로딩
from .error_handler import get_error_handler

logger = get_logger(__name__)
error_handler = get_error_handler()


@dataclass
class NormalizationConfig:
    """정규화 설정"""

    gpu_threshold: int = 5000  # 이 크기 이상만 GPU 사용
    zero_std_fallback: float = 1e-8
    identical_scores_fallback: float = 0.5


class GPUNormalizer:
    """완전 독립적인 GPU 정규화기 (순환 참조 없음)"""

    def __init__(self):
        config_dict = get_config("main").get_nested("utils.normalizer", {})
        self.config = NormalizationConfig(**config_dict)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()
        self.memory_manager = None  # 지연 로딩
        self.error_handler = get_error_handler()  # 에러 핸들러는 데코레이터 때문에 유지

        logger.info(f"✅ GPU 정규화기 초기화: {self.device}")

    def _get_memory_manager(self):
        if self.memory_manager is None:
            from .memory_manager import get_memory_manager

            self.memory_manager = get_memory_manager()
        return self.memory_manager

    @error_handler.auto_recoverable(max_retries=2, delay=0.1)
    def smart_normalize(
        self,
        data: Union[np.ndarray, list, torch.Tensor],
        method: str = "zscore",
        axis: int = 0,
    ) -> np.ndarray:
        """독립적인 스마트 정규화 (performance_optimizer 의존성 없음)"""

        # 데이터 전처리
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # 데이터 크기 기반 GPU/CPU 선택
        data_size = data.nbytes if hasattr(data, "nbytes") else len(data)

        if (
            self.gpu_available
            and data_size > self.config.gpu_threshold
            and self._get_memory_manager()._gpu_memory_ok()
        ):

            return self._gpu_normalize(data, method, axis)
        else:
            return self._cpu_normalize(data, method, axis)

    def _gpu_normalize(self, data: np.ndarray, method: str, axis: int) -> np.ndarray:
        """GPU 정규화 (직접 구현)"""
        try:
            # GPU 텐서로 변환
            tensor = torch.from_numpy(data).float().to(self.device)

            if method == "zscore":
                mean = torch.mean(tensor, dim=axis, keepdim=True)
                std = torch.std(tensor, dim=axis, keepdim=True)
                std = torch.where(
                    std == 0,
                    torch.tensor(self.config.zero_std_fallback, device=self.device),
                    std,
                )
                normalized = (tensor - mean) / std

            elif method == "minmax":
                min_val = torch.min(tensor, dim=axis, keepdim=True)[0]
                max_val = torch.max(tensor, dim=axis, keepdim=True)[0]
                range_val = max_val - min_val
                range_val = torch.where(
                    range_val == 0, torch.ones_like(range_val), range_val
                )
                normalized = (tensor - min_val) / range_val

            elif method == "robust":
                median = torch.median(tensor, dim=axis, keepdim=True)[0]
                mad = torch.median(torch.abs(tensor - median), dim=axis, keepdim=True)[
                    0
                ]
                mad = torch.where(
                    mad == 0,
                    torch.tensor(self.config.zero_std_fallback, device=self.device),
                    mad,
                )
                normalized = (tensor - median) / mad
            else:
                normalized = tensor

            result = normalized.cpu().numpy()

            # GPU 메모리 정리 (중앙 관리 메서드 사용)
            del tensor, normalized
            self._get_memory_manager().release_cuda_cache("normalizer._gpu_normalize")

            return result

        except Exception as e:
            logger.warning(f"GPU 정규화 실패, CPU 폴백: {e}")
            # 에러 핸들러가 재시도 하므로, 여기서 바로 폴백하지 않고 예외를 다시 발생시킴
            raise

    def _cpu_normalize(self, data: np.ndarray, method: str, axis: int) -> np.ndarray:
        """CPU 정규화"""
        if method == "zscore":
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            std[std == 0] = self.config.zero_std_fallback
            return (data - mean) / std

        elif method == "minmax":
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            return (data - min_val) / range_val

        elif method == "robust":
            median = np.median(data, axis=axis, keepdims=True)
            mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
            mad[mad == 0] = self.config.zero_std_fallback
            return (data - median) / mad

        return data

    def batch_normalize(self, data_list: list, method: str = "zscore") -> list:
        """배치 정규화"""
        return [self.smart_normalize(data, method) for data in data_list]


# 싱글톤 인스턴스
_gpu_normalizer: Optional[GPUNormalizer] = None


def get_gpu_normalizer() -> GPUNormalizer:
    """독립적 정규화기 반환"""
    global _gpu_normalizer
    if _gpu_normalizer is None:
        _gpu_normalizer = GPUNormalizer()
    return _gpu_normalizer


# 하위 호환성 함수들
def smart_normalize(data, method="zscore", **kwargs) -> np.ndarray:
    normalizer = get_gpu_normalizer()
    return normalizer.smart_normalize(data, method, **kwargs)


def z_score_normalize(array, **kwargs):
    return smart_normalize(array, method="zscore", **kwargs)


def min_max_normalize(array, **kwargs):
    return smart_normalize(array, method="minmax", **kwargs)


# 기존 Normalizer 클래스 (하위 호환성)
class Normalizer(GPUNormalizer):
    """하위 호환성 래퍼"""

    pass


# 하위 호환성 래퍼 추가
class IndependentGPUNormalizer(GPUNormalizer):
    """하위 호환성 래퍼 - __init__.py에서 참조되는 클래스"""

    pass


def get_independent_gpu_normalizer():
    """하위 호환성 함수"""
    return get_gpu_normalizer()


# get_independent_normalizer를 get_gpu_normalizer로 대체
get_independent_normalizer = get_gpu_normalizer
