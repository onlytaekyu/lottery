"""
RandomForest 모델

이 모듈은 RandomForest 알고리즘 기반 로또 번호 예측 모델을 구현합니다.
MLBaseModel을 상속받아 공통 인터페이스를 구현합니다.
"""

import os
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# ML 라이브러리 임포트
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from .ml_models import MLBaseModel
from ...utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class RandomForestModel(MLBaseModel):
    """
    RandomForest 기반 로또 번호 예측 모델

    앙상블 기법인 랜덤 포레스트를 사용하여 로또 번호의 당첨 확률을 예측합니다.
    다수의 결정 트리를 사용하여 과적합을 방지하고 안정적인 예측을 제공합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        RandomForest 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config, model_name="RandomForestModel")

        # 기본 하이퍼파라미터 설정
        self.default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }

        # 설정에서 하이퍼파라미터 업데이트 (있는 경우)
        self.params = self.default_params.copy()
        if config and "random_forest" in config:
            self.params.update(config["random_forest"])

        logger.info(f"RandomForest 모델 초기화 완료: {self.params}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        RandomForest 모델 학습

        Args:
            X: 특성 벡터
            y: 타겟 점수
            **kwargs: 추가 매개변수 (feature_names 등)

        Returns:
            학습 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"RandomForest 모델 학습 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 기본 매개변수 설정
        feature_names = kwargs.get(
            "feature_names", [f"feature_{i}" for i in range(X.shape[1])]
        )

        # 특성 이름 저장
        self.feature_names = feature_names

        # 훈련/검증 데이터 준비
        if "eval_set" in kwargs:
            eval_set = kwargs["eval_set"]
            X_val, y_val = eval_set[0]
        else:
            # 검증 세트가 없으면 훈련 데이터의 20%를 검증에 사용
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X, y = X_train, y_train

        # 모델 초기화
        self.model = RandomForestRegressor(**self.params)

        # 학습 시작
        start_time = time.time()
        self.model.fit(X, y)

        # 학습 시간 계산
        train_time = time.time() - start_time

        # 평가 수행
        y_pred = self.model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # 특성 중요도 계산
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))

        # 상위 10개 중요 특성 로깅
        top_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        logger.info(f"상위 10개 중요 특성: {top_features}")

        # 메타데이터 업데이트
        self.metadata.update(
            {
                "train_samples": X.shape[0],
                "features": X.shape[1],
                "eval_rmse": rmse,
                "eval_mae": mae,
                "eval_r2": r2,
                "train_time": train_time,
                "feature_importance": feature_importance,
            }
        )

        # 훈련 완료 표시
        self.is_trained = True
        logger.info(
            f"RandomForest 모델 학습 완료: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, "
            f"소요 시간={train_time:.2f}초"
        )

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        RandomForest 모델 예측 수행

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            예측 점수
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        logger.info(f"RandomForest 예측 수행: 입력 형태={X.shape}")

        # 예측 수행
        predictions = self.model.predict(X)

        return predictions

    def _save_model(self, path: str) -> None:
        """
        RandomForest 모델 저장 구현

        Args:
            path: 저장 경로
        """
        joblib.dump(self.model, path)

    def _load_model(self, path: str) -> None:
        """
        RandomForest 모델 로드 구현

        Args:
            path: 모델 파일 경로
        """
        self.model = joblib.load(path)

    def _get_feature_importance(self, importance_type: str) -> Dict[str, float]:
        """
        RandomForest 특성 중요도 계산 구현

        Args:
            importance_type: 중요도 유형 (무시됨)

        Returns:
            특성 이름과 중요도 딕셔너리
        """
        # RandomForest는 importance_type을 무시하고 기본 feature_importances_를 사용
        if not self.is_trained or self.model is None:
            return {}

        # 특성 이름이 없는 경우
        if not self.feature_names:
            return {}

        # 특성 중요도 계산
        importance = self.model.feature_importances_

        # 결과 반환
        return dict(zip(self.feature_names, importance))
