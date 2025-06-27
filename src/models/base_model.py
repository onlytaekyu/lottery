"""
기본 모델 클래스 (Base Model Class)

이 모듈은 모든 모델이 상속받는 기본 모델 클래스를 정의합니다.
표준 인터페이스를 제공하여 모델 간 호환성을 보장합니다.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import time
import importlib

from ..utils.error_handler_refactored import get_logger
from ..utils.model_saver import save_model, load_model
from ..shared.types import LotteryNumber, ModelPrediction

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    모든 DAEBAK_AI 모델의 기본 클래스

    이 클래스는 로또 번호 예측 모델이 구현해야 하는 기본 인터페이스를 정의합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        기본 모델 초기화

        Args:
            config: 모델 설정
        """
        # 기본 설정
        self.config = config or {}

        # 장치 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 상태
        self.is_trained = False
        self.training_history = []
        self.model_name = self.__class__.__name__

        # 모델 메타데이터
        self.metadata = {
            "model_type": self.model_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0.0",
        }

        logger.info(f"{self.model_name} 초기화: 장치 {self.device}")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 훈련 (표준 인터페이스)

        Args:
            X: 특성 벡터
            y: 레이블/타겟
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과 및 메타데이터
        """
        raise NotImplementedError("fit 메서드를 구현해야 합니다.")

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        모델 예측 수행 (표준 인터페이스)

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            예측값
        """
        raise NotImplementedError("predict 메서드를 구현해야 합니다.")

    @abstractmethod
    def save(self, path: str) -> bool:
        """
        모델 저장 (표준 인터페이스)

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        raise NotImplementedError("save 메서드를 구현해야 합니다.")

    @abstractmethod
    def load(self, path: str) -> bool:
        """
        모델 로드 (표준 인터페이스)

        Args:
            path: 로드할 모델 경로

        Returns:
            성공 여부
        """
        raise NotImplementedError("load 메서드를 구현해야 합니다.")

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 평가 (선택적 구현)

        Args:
            X: 특성 벡터
            y: 레이블/타겟
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        if not self.is_trained:
            return {"error": "훈련되지 않은 모델은 평가할 수 없습니다."}

        return {
            "message": "evaluate 메서드가 구현되지 않았습니다.",
            "status": "not_implemented",
        }

    def _ensure_directory(self, path: str) -> None:
        """
        저장 경로의 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.

        Args:
            path: 파일 경로
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"디렉토리 생성: {directory}")

    def get_feature_vector(
        self,
        feature_path: str = "data/cache/feature_vector_full.npy",
        names_path: str = "data/cache/feature_vector_full.names.json",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        특성 벡터와 특성 이름을 로드합니다.

        Args:
            feature_path: 특성 벡터 파일 경로
            names_path: 특성 이름 파일 경로

        Returns:
            특성 벡터와 특성 이름의 튜플
        """
        try:
            # 벡터 데이터 로드
            if not os.path.exists(feature_path):
                raise FileNotFoundError(
                    f"특성 벡터 파일이 존재하지 않습니다: {feature_path}"
                )

            vector = np.load(feature_path)

            # 특성 이름 로드
            if not os.path.exists(names_path):
                raise FileNotFoundError(
                    f"특성 이름 파일이 존재하지 않습니다: {names_path}"
                )

            with open(names_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)

            logger.info(
                f"특성 벡터 로드 완료: {feature_path}, 형태={vector.shape}, 특성 수={len(feature_names)}"
            )
            return vector, feature_names

        except Exception as e:
            logger.error(f"특성 벡터 로드 중 오류: {e}")
            raise


class ModelWithAMP(BaseModel):
    """
    Automatic Mixed Precision (AMP)를 지원하는 모델 기본 클래스

    PyTorch 모델에 AMP를 적용하기 위한 공통 기능을 제공합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        AMP 지원 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # AMP 관련 설정
        self.use_amp = (
            self.config.get("use_amp", True) if torch.cuda.is_available() else False
        )

        # Scaler 초기화
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info(f"{self.model_name}: AMP 활성화됨")
        else:
            # Linter 오류 방지를 위해 None 대신 더미 스케일러 객체 생성
            self.scaler = _DummyScaler()
            logger.info(f"{self.model_name}: AMP 비활성화됨")

    def train_step_with_amp(self, model, inputs, targets, optimizer, loss_fn, **kwargs):
        """
        AMP를 적용한 훈련 단계

        Args:
            model: 훈련할 모델
            inputs: 입력 데이터
            targets: 타겟 데이터
            optimizer: 옵티마이저
            loss_fn: 손실 함수
            **kwargs: 추가 매개변수

        Returns:
            손실값
        """
        # 모델을 훈련 모드로 설정
        model.train()

        # 그래디언트 초기화
        optimizer.zero_grad()

        if self.use_amp:
            # AMP 컨텍스트에서 순전파
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            # 스케일된 그래디언트로 역전파
            self.scaler.scale(loss).backward()

            # 스케일된 그래디언트로 옵티마이저 스텝
            self.scaler.step(optimizer)

            # 스케일러 업데이트
            self.scaler.update()
        else:
            # 일반 순전파 및 역전파
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        return loss.item()


class _DummyScaler:
    """
    AMP가 비활성화되었을 때 사용하는 더미 스케일러 클래스

    AMP 스케일러 인터페이스를 모방하여 linter 오류를 방지합니다.
    """

    def scale(self, loss):
        """
        손실값 스케일링 (더미 구현)

        Args:
            loss: 손실값

        Returns:
            원본 손실값 (스케일링 없음)
        """
        return loss

    def step(self, optimizer):
        """
        옵티마이저 스텝 (더미 구현)

        Args:
            optimizer: 옵티마이저
        """
        optimizer.step()

    def update(self):
        """
        스케일러 업데이트 (더미 구현)
        """
        pass


class EnsembleBaseModel(BaseModel):
    """
    앙상블 모델 기본 클래스

    여러 모델의 예측을 결합하는 앙상블 모델을 위한 기본 클래스입니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        앙상블 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 및 가중치 목록
        self.models = []
        self.weights = []

    def add_model(self, model: BaseModel, weight: float = 1.0):
        """
        앙상블에 모델 추가

        Args:
            model: 추가할 모델
            weight: 모델 가중치 (기본값: 1.0)
        """
        self.models.append(model)
        self.weights.append(weight)
        logger.info(f"앙상블에 {model.model_name} 추가 (가중치: {weight})")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모든 구성 모델 훈련

        Args:
            X: 특성 벡터
            y: 레이블/타겟
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과
        """
        results = {}

        for i, model in enumerate(self.models):
            logger.info(
                f"앙상블 구성 모델 {i+1}/{len(self.models)} ({model.model_name}) 훈련 중..."
            )
            result = model.fit(X, y, **kwargs)
            results[model.model_name] = result

        self.is_trained = all(model.is_trained for model in self.models)

        return {
            "ensemble_results": results,
            "is_trained": self.is_trained,
            "model_count": len(self.models),
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        앙상블 예측 수행

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            앙상블 예측값
        """
        if not self.models:
            raise ValueError("앙상블에 모델이 없습니다.")

        if not self.is_trained:
            raise ValueError("훈련되지 않은 앙상블 모델입니다.")

        predictions = []

        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X, **kwargs)
            predictions.append((pred, weight))

        return self._combine_predictions(predictions)

    def _combine_predictions(self, predictions) -> np.ndarray:
        """
        모델 예측값 결합

        Args:
            predictions: (예측값, 가중치) 튜플 목록

        Returns:
            결합된 예측값
        """
        # 기본 구현: 가중 평균
        weighted_preds = [pred * weight for pred, weight in predictions]
        total_weight = sum(self.weights)

        if total_weight == 0:
            return predictions[0][0]  # 가중치가 모두 0이면 첫 번째 예측 반환

        return sum(weighted_preds) / total_weight

    def save(self, path: str) -> bool:
        """
        앙상블 모델 저장

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        try:
            # 디렉토리 확인
            self._ensure_directory(path)

            # 각 구성 모델 저장
            ensemble_data = {
                "model_paths": [],
                "weights": self.weights,
                "metadata": self.metadata,
            }

            for i, model in enumerate(self.models):
                model_path = f"{os.path.splitext(path)[0]}_model_{i}.pt"
                success = model.save(model_path)
                if success:
                    ensemble_data["model_paths"].append(model_path)

            # 앙상블 메타데이터 저장
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ensemble_data, f, ensure_ascii=False, indent=2)

            logger.info(f"앙상블 모델 저장 완료: {path} (모델 {len(self.models)}개)")
            return True

        except Exception as e:
            logger.error(f"앙상블 모델 저장 중 오류: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        앙상블 모델 로드

        Args:
            path: 모델 경로

        Returns:
            성공 여부
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"앙상블 모델 파일이 존재하지 않습니다: {path}")

            # 앙상블 메타데이터 로드
            with open(path, "r", encoding="utf-8") as f:
                ensemble_data = json.load(f)

            model_paths = ensemble_data.get("model_paths", [])
            self.weights = ensemble_data.get("weights", [])
            self.metadata = ensemble_data.get("metadata", {})

            # 모델 목록 초기화
            self.models = []

            # 각 구성 모델 로드
            for model_path in model_paths:
                if not os.path.exists(model_path):
                    logger.warning(f"구성 모델 파일이 존재하지 않습니다: {model_path}")
                    continue

                # 모델 타입 추론 및 인스턴스 생성
                model_type = os.path.basename(model_path).split("_")[0]
                model_instance = self._create_model_instance(model_type)

                if model_instance and model_instance.load(model_path):
                    self.models.append(model_instance)

            # 훈련 상태 갱신
            self.is_trained = len(self.models) > 0

            logger.info(
                f"앙상블 모델 로드 완료: {path} (모델 {len(self.models)}개/{len(model_paths)}개)"
            )
            return self.is_trained

        except Exception as e:
            logger.error(f"앙상블 모델 로드 중 오류: {e}")
            return False

    def _create_model_instance(self, model_type: str) -> Optional[BaseModel]:
        """
        모델 타입에 따른 인스턴스 생성

        Args:
            model_type: 모델 타입 문자열

        Returns:
            모델 인스턴스 또는 None
        """
        try:
            # 모델 모듈 동적 임포트
            for module_path in [
                f"..models.ml.{model_type.lower()}_model",
                f"..models.dl.{model_type.lower()}_model",
                f"..models.rl.{model_type.lower()}_model",
                f"..models.bayesian.{model_type.lower()}_model",
                f"..models.meta.{model_type.lower()}_model",
            ]:
                try:
                    module = importlib.import_module(module_path, package=__package__)
                    model_class = getattr(module, model_type)
                    return model_class(self.config)
                except (ImportError, AttributeError):
                    continue

            logger.warning(
                f"모델 타입에 해당하는 클래스를 찾을 수 없습니다: {model_type}"
            )
            return None

        except Exception as e:
            logger.error(f"모델 인스턴스 생성 중 오류: {e}")
            return None


__all__ = ["BaseModel", "ModelWithAMP", "EnsembleBaseModel"]
