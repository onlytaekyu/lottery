"""
모든 트레이너의 기본 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..utils.unified_config import ConfigProxy
from ..utils.cuda_optimizers import AMPTrainer
from ..shared.types import LotteryNumber
from ..utils.error_handler_refactored import get_logger


class BaseTrainer(ABC):
    def __init__(self, config: ConfigProxy):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.amp_trainer = AMPTrainer(config)
        self._data: Optional[List[LotteryNumber]] = None
        self.device = self.amp_trainer.get_device()

    @abstractmethod
    def set_data(self, draw_data: List[LotteryNumber]) -> None:
        """학습 데이터를 설정합니다."""
        self._data = draw_data

    @abstractmethod
    def train(self) -> None:
        """모델을 학습합니다."""
        if self._data is None:
            raise ValueError("학습 데이터가 설정되지 않았습니다.")

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """모델을 평가합니다."""
        if self._data is None:
            raise ValueError("평가 데이터가 설정되지 않았습니다.")

    @abstractmethod
    def predict(self) -> List[List[int]]:
        """예측을 수행합니다."""
        if self._data is None:
            raise ValueError("예측 데이터가 설정되지 않았습니다.")

    def get_device_info(self) -> Dict[str, Any]:
        """현재 디바이스 정보를 반환합니다."""
        return self.amp_trainer.get_device_info()

    def optimize_memory(self) -> None:
        """메모리를 최적화합니다."""
        import gc
        import torch

        # 가비지 컬렉션 실행
        gc.collect()

        # GPU 메모리 정리 (CUDA 사용 가능한 경우)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("메모리 최적화 완료")
