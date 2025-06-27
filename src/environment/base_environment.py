"""
로또 환경 모듈

이 모듈은 강화학습 환경을 위한 기본 인터페이스와 구현을 제공합니다.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Protocol
import abc
from dataclasses import dataclass
from datetime import datetime

# 상대 경로 사용으로 수정
from ..utils.error_handler_refactored import get_logger
from src.shared.types import LotteryNumber, PatternAnalysis

logger = get_logger(__name__)


class BaseEnvironment(abc.ABC):
    """강화학습 환경을 위한 기본 인터페이스"""

    def __init__(
        self,
        train_data: Optional[List[Any]] = None,
        max_steps: int = 6,
    ):
        """
        초기화

        Args:
            train_data: 학습 데이터 (없으면 None)
            max_steps: 최대 스텝 수
        """
        self.logger = get_logger(__name__)
        self.train_data = train_data or []
        self.max_steps = max_steps
        self.steps = 0

        # 상태 및 행동 공간 크기 (기본값)
        self.state_size = 45
        self.action_size = 45

        # 임베딩 차원 (기본값)
        self.embedding_dim = 32

        # 선택된 행동 추적
        self.selected = np.zeros(self.action_size, dtype=np.int32)

    @abc.abstractmethod
    def reset(self) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        환경 초기화

        Returns:
            초기 상태 또는 (초기 상태, 추가 정보) 튜플
        """
        pass

    @abc.abstractmethod
    def step(
        self, action: int
    ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, Any]], float, bool, Dict]:
        """
        환경 진행

        Args:
            action: 선택할 행동

        Returns:
            새로운 상태 또는 (새로운 상태, 추가 정보) 튜플
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        pass

    def _calculate_reward(self, action: int) -> float:
        """
        보상 계산

        Args:
            action: 선택한 행동

        Returns:
            reward: 보상값
        """
        return 0.0  # 기본 구현은 보상을 0으로 반환

    def close(self) -> None:
        """환경 정리"""
        pass


def ensure_embedding_dim(env_object, default_dim=32):
    """
    환경 객체에 embedding_dim 속성이 없으면 추가

    Args:
        env_object: 환경 객체
        default_dim: 기본 임베딩 차원
    """
    try:
        if not hasattr(env_object, "embedding_dim"):
            logger.info(f"임베딩 차원 속성 없음, 기본값({default_dim}) 적용")
            env_object.embedding_dim = default_dim
    except Exception as e:
        logger.error(f"임베딩 차원 설정 중 오류: {str(e)}")
