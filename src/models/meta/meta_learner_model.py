"""
메타 러닝 모델

이 모듈은 여러 모델의 예측 결과를 조합하는 메타 러닝 모델을 구현합니다.
모델별 성능에 기반한 가중치를 자동으로 조정하여 최적의 예측을 수행합니다.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time
import importlib

# scikit-learn 모델 미리 임포트
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ..base_model import BaseModel, EnsembleBaseModel
from ...utils.error_handler import get_logger

logger = get_logger(__name__)


class MetaLearnerModel(EnsembleBaseModel):
    """
    메타 러닝 모델

    여러 기본 모델의 예측을 조합하여 최종 예측을 생성하는 모델입니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        메타 러닝 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 설정
        self.config = config or {}
        meta_config = self.config.get("meta_learner", {})

        # 모델 하이퍼파라미터
        self.learning_rate = meta_config.get("learning_rate", 0.01)
        self.num_epochs = meta_config.get("num_epochs", 100)
        self.batch_size = meta_config.get("batch_size", 32)
        self.weight_decay = meta_config.get("weight_decay", 0.0001)
        self.patience = meta_config.get("patience", 10)
        self.auto_adjust = meta_config.get("auto_adjust", True)
        self.adjustment_frequency = meta_config.get("adjustment_frequency", 5)

        # 모델 가중치 (초기에는 균등 가중치)
        self.weights = []

        # 모델 이름
        self.model_name = "MetaLearnerModel"

        # 가중치 조정을 위한 변수
        self.adjustment_count = 0
        self.performance_history = []

        logger.info(
            f"메타 러닝 모델 초기화 완료: 자동 가중치 조정={self.auto_adjust}, "
            f"조정 주기={self.adjustment_frequency}"
        )

    def add_model(self, model: BaseModel, weight: float = 1.0):
        """
        앙상블에 모델 추가

        Args:
            model: 추가할 모델
            weight: 모델 가중치 (기본값: 1.0)
        """
        super().add_model(model, weight)

        # 가중치 정규화
        if self.weights:
            self._normalize_weights()

    def _normalize_weights(self):
        """
        가중치 정규화 (합이 1이 되도록)
        """
        weight_sum = sum(self.weights)
        if weight_sum > 0:
            self.weights = [w / weight_sum for w in self.weights]
            logger.info(f"가중치 정규화: {self.weights}")

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        메타 모델 예측 수행

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            예측값
        """
        if not self.models:
            raise ValueError("앙상블에 모델이 없습니다.")

        if not self.is_trained:
            raise ValueError("훈련되지 않은 앙상블 모델입니다.")

        predictions = []

        # 각 모델의 예측 수행
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            logger.info(
                f"모델 {i+1}/{len(self.models)} ({model.model_name}) 예측 중... (가중치: {weight:.4f})"
            )
            pred = model.predict(X, **kwargs)
            predictions.append((pred, weight))

        # 가중 평균으로 예측 결합
        return self._combine_predictions(predictions)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        메타 모델 훈련

        Args:
            X: 특성 벡터
            y: 타겟 벡터
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과
        """
        # 로깅
        logger.info(f"메타 러닝 모델 훈련 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 훈련 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 각 모델 훈련
        results = {}
        for i, model in enumerate(self.models):
            logger.info(
                f"구성 모델 {i+1}/{len(self.models)} ({model.model_name}) 훈련 중..."
            )
            result = model.fit(X_train, y_train, **kwargs)
            results[model.model_name] = result

        # 검증 세트에서 각 모델의 성능 평가
        performances = []
        for i, model in enumerate(self.models):
            logger.info(
                f"구성 모델 {i+1}/{len(self.models)} ({model.model_name}) 성능 평가 중..."
            )
            evaluation = model.evaluate(X_val, y_val)

            # RMSE를 성능 지표로 사용 (낮을수록 좋음)
            rmse = evaluation.get("rmse", 0.0)
            performances.append((i, model.model_name, rmse))

        # 성능 기록
        self.performance_history.append(performances)

        # 성능 기반 가중치 조정
        if self.auto_adjust:
            self._adjust_weights_by_performance(performances)

        # 훈련 완료 표시
        self.is_trained = True

        # 결과 반환
        return {
            "ensemble_results": results,
            "model_performances": performances,
            "weights": self.weights,
            "is_trained": self.is_trained,
        }

    def _adjust_weights_by_performance(
        self, performances: List[Tuple[int, str, float]]
    ):
        """
        성능 기반 가중치 조정

        Args:
            performances: (모델 인덱스, 모델 이름, RMSE) 튜플 목록
        """
        # RMSE 추출 (낮을수록 좋음)
        rmse_values = [rmse for _, _, rmse in performances]

        # 역수 변환 (높을수록 좋음)
        if all(rmse > 0 for rmse in rmse_values):
            inverse_rmse = [1.0 / rmse for rmse in rmse_values]

            # 정규화
            total = sum(inverse_rmse)
            new_weights = [val / total for val in inverse_rmse]

            # 가중치 업데이트
            self.weights = new_weights

            # 로깅
            weight_info = [
                f"{name}: {weight:.4f}"
                for (_, name, _), weight in zip(performances, new_weights)
            ]
            logger.info(f"성능 기반 가중치 조정: {', '.join(weight_info)}")
        else:
            logger.warning("일부 RMSE 값이 0이어서 가중치 조정 불가")

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        메타 모델 평가

        Args:
            X: 특성 벡터
            y: 타겟 벡터
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        if not self.is_trained:
            raise ValueError("훈련되지 않은 모델은 평가할 수 없습니다.")

        # 개별 모델 평가
        model_results = []
        for i, model in enumerate(self.models):
            logger.info(
                f"구성 모델 {i+1}/{len(self.models)} ({model.model_name}) 평가 중..."
            )
            result = model.evaluate(X, y, **kwargs)
            model_results.append((model.model_name, result))

        # 앙상블 예측
        ensemble_pred = self.predict(X, **kwargs)

        # 앙상블 성능 평가
        rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
        mae = mean_absolute_error(y, ensemble_pred)
        r2 = r2_score(y, ensemble_pred)

        logger.info(f"메타 러닝 모델 평가: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        # 자동 가중치 조정 (adjustment_frequency 주기로)
        if self.auto_adjust:
            self.adjustment_count += 1
            if self.adjustment_count >= self.adjustment_frequency:
                # 모델별 RMSE 추출
                performances = [
                    (i, model_name, result.get("rmse", 0.0))
                    for i, (model_name, result) in enumerate(model_results)
                ]

                # 가중치 조정
                self._adjust_weights_by_performance(performances)

                # 카운터 리셋
                self.adjustment_count = 0

        return {
            "ensemble_rmse": rmse,
            "ensemble_mae": mae,
            "ensemble_r2": r2,
            "model_results": model_results,
            "weights": self.weights,
        }

    def save(self, path: str) -> bool:
        """
        메타 모델 저장

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
                "config": {
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "weight_decay": self.weight_decay,
                    "patience": self.patience,
                    "auto_adjust": self.auto_adjust,
                    "adjustment_frequency": self.adjustment_frequency,
                },
                "performance_history": self.performance_history,
                "is_trained": self.is_trained,
            }

            for i, model in enumerate(self.models):
                model_path = f"{os.path.splitext(path)[0]}_model_{i}.pt"
                success = model.save(model_path)
                if success:
                    ensemble_data["model_paths"].append(model_path)

            # 메타 모델 메타데이터 저장
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ensemble_data, f, ensure_ascii=False, indent=2)

            logger.info(f"메타 러닝 모델 저장 완료: {path} (모델 {len(self.models)}개)")
            return True

        except Exception as e:
            logger.error(f"메타 러닝 모델 저장 중 오류: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        메타 모델 로드

        Args:
            path: 모델 경로

        Returns:
            성공 여부
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(path):
                raise FileNotFoundError(f"메타 모델 파일이 존재하지 않습니다: {path}")

            # 메타데이터 로드
            with open(path, "r", encoding="utf-8") as f:
                ensemble_data = json.load(f)

            # 설정 로드
            config = ensemble_data.get("config", {})
            self.learning_rate = config.get("learning_rate", self.learning_rate)
            self.num_epochs = config.get("num_epochs", self.num_epochs)
            self.batch_size = config.get("batch_size", self.batch_size)
            self.weight_decay = config.get("weight_decay", self.weight_decay)
            self.patience = config.get("patience", self.patience)
            self.auto_adjust = config.get("auto_adjust", self.auto_adjust)
            self.adjustment_frequency = config.get(
                "adjustment_frequency", self.adjustment_frequency
            )

            # 가중치 및 메타데이터 로드
            self.weights = ensemble_data.get("weights", [])
            self.metadata = ensemble_data.get("metadata", {})
            self.performance_history = ensemble_data.get("performance_history", [])
            self.is_trained = ensemble_data.get("is_trained", False)

            # 모델 로드
            model_paths = ensemble_data.get("model_paths", [])
            self.models = []

            for model_path in model_paths:
                if not os.path.exists(model_path):
                    logger.warning(f"구성 모델 파일이 존재하지 않습니다: {model_path}")
                    continue

                # 모델 타입 추론 및 인스턴스 생성
                model_type = os.path.basename(model_path).split("_")[0]
                model_instance = self._create_model_instance(model_type)

                if model_instance and model_instance.load(model_path):
                    self.models.append(model_instance)

            logger.info(
                f"메타 러닝 모델 로드 완료: {path} (모델 {len(self.models)}개/{len(model_paths)}개)"
            )
            return len(self.models) > 0

        except Exception as e:
            logger.error(f"메타 러닝 모델 로드 중 오류: {e}")
            return False

    def combine_predictions_with_weights(
        self, predictions: List[np.ndarray], weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        여러 모델의 예측을 가중치에 따라 조합

        Args:
            predictions: 예측값 리스트
            weights: 가중치 리스트 (None인 경우 self.weights 사용)

        Returns:
            조합된 예측값
        """
        if weights is None:
            weights = self.weights

        if len(predictions) != len(weights):
            raise ValueError(
                f"예측값 수({len(predictions)})와 가중치 수({len(weights)})가 일치하지 않습니다."
            )

        # 가중 평균 계산
        weighted_sum = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_sum += pred * weight

        return weighted_sum

    def _combine_predictions(self, predictions) -> np.ndarray:
        """
        모델 예측값 결합

        Args:
            predictions: (예측값, 가중치) 튜플 목록

        Returns:
            결합된 예측값
        """
        # 기본 구현: 가중 평균
        preds = [pred for pred, _ in predictions]
        weights = [weight for _, weight in predictions]

        return self.combine_predictions_with_weights(preds, weights)


class MetaRegressionModel(MetaLearnerModel):
    """
    메타 회귀 모델

    기본 모델의 예측을 입력으로 사용하여 최종 예측을 생성하는 메타 회귀 모델입니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        메타 회귀 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 이름 업데이트
        self.model_name = "MetaRegressionModel"

        # 메타 회귀 모델
        self.meta_model = None

        logger.info(f"메타 회귀 모델 초기화 완료")

    def _build_meta_model(self, input_dim: int):
        """
        메타 회귀 모델 구성

        Args:
            input_dim: 입력 차원 (기본 모델 수)
        """
        # 메타 회귀 모델 초기화 (Ridge 회귀)
        self.meta_model = Ridge(alpha=self.weight_decay)

        logger.info(f"메타 회귀 모델 구성 완료: 입력 차원={input_dim}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        메타 회귀 모델 훈련

        Args:
            X: 특성 벡터
            y: 타겟 벡터
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과
        """
        # 로깅
        logger.info(f"메타 회귀 모델 훈련 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 훈련 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 각 기본 모델 훈련
        results = {}
        for i, model in enumerate(self.models):
            logger.info(
                f"구성 모델 {i+1}/{len(self.models)} ({model.model_name}) 훈련 중..."
            )
            result = model.fit(X_train, y_train, **kwargs)
            results[model.model_name] = result

        # 각 모델의 예측 생성
        meta_X_train = []
        meta_X_val = []

        for i, model in enumerate(self.models):
            logger.info(
                f"구성 모델 {i+1}/{len(self.models)} ({model.model_name}) 예측 생성 중..."
            )
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            meta_X_train.append(train_pred)
            meta_X_val.append(val_pred)

        # 예측 배열 변환
        meta_X_train = np.column_stack(meta_X_train)
        meta_X_val = np.column_stack(meta_X_val)

        # 메타 모델 구성
        self._build_meta_model(len(self.models))

        # 메타 모델 훈련
        logger.info("메타 회귀 모델 훈련 중...")

        # Ridge 모델 훈련
        self.meta_model.fit(meta_X_train, y_train)

        # 가중치 설정 (Ridge 모델의 계수 사용)
        self.weights = np.abs(self.meta_model.coef_)
        self._normalize_weights()

        # 검증 성능 평가
        meta_y_val_pred = self.meta_model.predict(meta_X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, meta_y_val_pred))
        logger.info(f"메타 회귀 모델 검증 RMSE: {val_rmse:.4f}")

        # 훈련 완료 표시
        self.is_trained = True

        return {
            "ensemble_results": results,
            "meta_model_weights": self.weights,
            "validation_rmse": val_rmse,
            "is_trained": self.is_trained,
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        메타 회귀 모델 예측 수행

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            예측값
        """
        if not self.models:
            raise ValueError("앙상블에 모델이 없습니다.")

        if not self.is_trained:
            raise ValueError("훈련되지 않은 모델입니다.")

        # 각 기본 모델의 예측 생성
        base_predictions = []
        for i, model in enumerate(self.models):
            logger.info(
                f"구성 모델 {i+1}/{len(self.models)} ({model.model_name}) 예측 중..."
            )
            pred = model.predict(X, **kwargs)
            base_predictions.append(pred)

        # 예측 배열 변환
        meta_X = np.column_stack(base_predictions)

        # 메타 모델 예측
        return self.meta_model.predict(meta_X)

    def save(self, path: str) -> bool:
        """
        메타 회귀 모델 저장

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        try:
            # 상위 클래스 저장 메서드 호출
            success = super().save(path)

            if success and self.meta_model is not None:
                # 메타 모델 저장
                import pickle

                meta_model_path = f"{os.path.splitext(path)[0]}_meta_model.pkl"
                with open(meta_model_path, "wb") as f:
                    pickle.dump(self.meta_model, f)

                logger.info(f"메타 회귀 모델 저장 완료: {meta_model_path}")

            return success

        except Exception as e:
            logger.error(f"메타 회귀 모델 저장 중 오류: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        메타 회귀 모델 로드

        Args:
            path: 모델 경로

        Returns:
            성공 여부
        """
        try:
            # 상위 클래스 로드 메서드 호출
            success = super().load(path)

            if success:
                # 메타 모델 로드
                import pickle

                meta_model_path = f"{os.path.splitext(path)[0]}_meta_model.pkl"

                if os.path.exists(meta_model_path):
                    with open(meta_model_path, "rb") as f:
                        self.meta_model = pickle.load(f)

                    logger.info(f"메타 회귀 모델 로드 완료: {meta_model_path}")
                else:
                    logger.warning(
                        f"메타 모델 파일이 존재하지 않습니다: {meta_model_path}"
                    )
                    return False

            return success

        except Exception as e:
            logger.error(f"메타 회귀 모델 로드 중 오류: {e}")
            return False


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
