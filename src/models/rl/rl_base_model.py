"""
강화학습 모델 기본 클래스

이 모듈은 강화학습 알고리즘을 위한 기본 클래스를 제공합니다.
모든 RL 모델들의 공통 기능을 구현합니다.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time

from ..base_model import ModelWithAMP
from ...utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class RLBaseModel(ModelWithAMP):
    """
    강화학습 모델 기본 클래스

    모든 RL 모델의 공통 기능을 구현합니다.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, model_name: str = "RLBaseModel"
    ):
        """
        RL 기본 모델 초기화

        Args:
            config: 모델 설정
            model_name: 모델 이름
        """
        super().__init__(config)
        self.model_name = model_name
        self.config = config or {}

        # 훈련 관련 기본 속성
        self.is_trained = False
        self.train_episode_rewards = []
        self.eval_episode_rewards = []

    def _ensure_directory(self, path: str) -> None:
        """
        디렉토리 생성 확인

        Args:
            path: 파일 경로
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def save(self, path: str) -> bool:
        """
        모델 저장 (공통 로직)

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        if not self.is_trained:
            logger.warning("학습되지 않은 모델은 저장할 수 없습니다.")
            return False

        try:
            # 디렉토리 확인
            self._ensure_directory(path)

            # 모델 파일 경로
            model_path = path

            # 메타데이터 파일 경로
            meta_path = f"{os.path.splitext(path)[0]}.meta.json"

            # 모델 저장 (하위 클래스에서 구현)
            self._save_model(model_path)

            # 메타데이터 저장
            metadata = {
                "model_type": self.model_name,
                "metadata": self.metadata,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 모델별 추가 메타데이터 추가
            metadata.update(self._get_additional_metadata())

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"{self.model_name} 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"{self.model_name} 모델 저장 중 오류: {str(e)}")
            return False

    def load(self, path: str) -> bool:
        """
        모델 로드 (공통 로직)

        Args:
            path: 모델 파일 경로

        Returns:
            성공 여부
        """
        try:
            # 모델 파일 확인
            if not os.path.exists(path):
                logger.error(f"모델 파일이 존재하지 않습니다: {path}")
                return False

            # 메타데이터 파일 경로
            meta_path = f"{os.path.splitext(path)[0]}.meta.json"
            if not os.path.exists(meta_path):
                logger.warning(f"메타데이터 파일이 존재하지 않습니다: {meta_path}")

            # 모델 로드 (하위 클래스에서 구현)
            self._load_model(path)

            # 메타데이터 로드
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    self.metadata = metadata.get("metadata", {})

                    # 모델별 추가 메타데이터 로드
                    self._load_additional_metadata(metadata)

            # 로드 완료 표시
            self.is_trained = True
            logger.info(f"{self.model_name} 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"{self.model_name} 모델 로드 중 오류: {str(e)}")
            return False

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 평가 (공통 로직)

        Args:
            X: 상태 데이터
            y: 보상 데이터
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 에피소드 수 설정
        episodes = kwargs.get("episodes", 10)

        # 평가 수행 (하위 클래스에서 구현)
        eval_results = self._evaluate_model(X, y, episodes=episodes, **kwargs)

        # 결과 로깅
        logger.info(
            f"{self.model_name} 평가 완료: "
            f"평균 보상={eval_results.get('average_reward', 0):.4f}, "
            f"표준편차={eval_results.get('std_reward', 0):.4f}, "
            f"최대 보상={eval_results.get('max_reward', 0):.4f}"
        )

        return eval_results

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        모델 예측 수행 (공통 로직)

        Args:
            X: 상태 데이터
            **kwargs: 추가 매개변수

        Returns:
            예측된 액션 또는 점수
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 예측 수행 (하위 클래스에서 구현)
        return self._predict(X, **kwargs)

    def _save_model(self, path: str) -> None:
        """모델 저장 구현 (하위 클래스에서 오버라이드)"""
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def _load_model(self, path: str) -> None:
        """모델 로드 구현 (하위 클래스에서 오버라이드)"""
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def _evaluate_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """모델 평가 구현 (하위 클래스에서 오버라이드)"""
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """예측 구현 (하위 클래스에서 오버라이드)"""
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def _get_additional_metadata(self) -> Dict[str, Any]:
        """
        추가 메타데이터 획득 (하위 클래스에서 오버라이드 가능)

        Returns:
            추가 메타데이터 딕셔너리
        """
        return {}

    def _load_additional_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        추가 메타데이터 로드 (하위 클래스에서 오버라이드 가능)

        Args:
            metadata: 메타데이터 딕셔너리
        """
        pass
