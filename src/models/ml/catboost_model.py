"""
CatBoost 모델

이 모듈은 CatBoost 알고리즘 기반 로또 번호 예측 모델을 구현합니다.
MLBaseModel을 상속받아 공통 인터페이스를 구현합니다.
"""

import os
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# ML 라이브러리 임포트
try:
    from catboost import CatBoost, Pool
except ImportError:
    raise ImportError(
        "CatBoost 라이브러리가 설치되지 않았습니다. 'pip install catboost'로 설치하세요."
    )

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .ml_models import MLBaseModel
from ...utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class CatBoostModel(MLBaseModel):
    """
    CatBoost 기반 로또 번호 예측 모델

    고성능 그래디언트 부스팅 알고리즘을 사용하여 로또 번호의 당첨 확률을 예측합니다.
    범주형 변수 처리에 강점이 있는 알고리즘입니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        CatBoost 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config, model_name="CatBoostModel")

        # 기본 하이퍼파라미터 설정
        self.default_params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "verbose": False,
        }

        # 설정에서 하이퍼파라미터 업데이트 (있는 경우)
        self.params = self.default_params.copy()
        if config and "catboost" in config:
            self.params.update(config["catboost"])

        logger.info(f"CatBoost 모델 초기화 완료: {self.params}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        CatBoost 모델 학습

        Args:
            X: 특성 벡터
            y: 타겟 점수
            **kwargs: 추가 매개변수 (eval_set, early_stopping_rounds 등)

        Returns:
            학습 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"CatBoost 모델 학습 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 기본 매개변수 설정
        early_stopping_rounds = kwargs.get("early_stopping_rounds", 50)
        feature_names = kwargs.get(
            "feature_names", [f"feature_{i}" for i in range(X.shape[1])]
        )
        categorical_features = kwargs.get("categorical_features", None)

        # 특성 이름 저장
        self.feature_names = feature_names

        # 훈련/검증 데이터 준비
        if "eval_set" in kwargs:
            eval_set = kwargs["eval_set"]
            X_val, y_val = eval_set[0]
            train_pool = Pool(
                X, y, feature_names=feature_names, cat_features=categorical_features
            )
            eval_pool = Pool(
                X_val,
                y_val,
                feature_names=feature_names,
                cat_features=categorical_features,
            )
        else:
            # 검증 세트가 없으면 훈련 데이터의 20%를 검증에 사용
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            train_pool = Pool(
                X_train,
                y_train,
                feature_names=feature_names,
                cat_features=categorical_features,
            )
            eval_pool = Pool(
                X_val,
                y_val,
                feature_names=feature_names,
                cat_features=categorical_features,
            )

        # 모델 초기화
        self.model = CatBoost(self.params)

        # 학습 시작
        start_time = time.time()
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=early_stopping_rounds,
            verbose=self.params.get("verbose", False),
            plot=False,
        )

        # 학습 시간 계산
        train_time = time.time() - start_time

        # 특성 중요도 계산
        feature_importance = dict(
            zip(feature_names, self.model.get_feature_importance(train_pool))
        )

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
                "best_iteration": self.model.get_best_iteration(),
                "best_score": self.model.get_best_score(),
                "train_time": train_time,
                "feature_importance": feature_importance,
            }
        )

        # 훈련 완료 표시
        self.is_trained = True
        logger.info(
            f"CatBoost 모델 학습 완료: 최적 반복={self.model.get_best_iteration()}, "
            f"최적 점수={self.model.get_best_score()}, 소요 시간={train_time:.2f}초"
        )

        return {
            "best_iteration": self.model.get_best_iteration(),
            "best_score": self.model.get_best_score(),
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        CatBoost 모델 예측 수행

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

        logger.info(f"CatBoost 예측 수행: 입력 형태={X.shape}")

        # 예측 옵션
        prediction_type = kwargs.get("prediction_type", "RawFormulaVal")
        thread_count = kwargs.get("thread_count", -1)

        # 데이터셋 준비
        categorical_features = kwargs.get("categorical_features", None)
        if self.feature_names and len(self.feature_names) == X.shape[1]:
            data_pool = Pool(
                X, feature_names=self.feature_names, cat_features=categorical_features
            )
            predictions = self.model.predict(
                data_pool, prediction_type=prediction_type, thread_count=thread_count
            )
        else:
            # 특성 이름이 없거나 차원이 맞지 않는 경우
            predictions = self.model.predict(
                X, prediction_type=prediction_type, thread_count=thread_count
            )

        return predictions

    def _save_model(self, path: str) -> None:
        """
        CatBoost 모델 저장 구현

        Args:
            path: 저장 경로
        """
        self.model.save_model(path, format="cbm")

    def _load_model(self, path: str) -> None:
        """
        CatBoost 모델 로드 구현

        Args:
            path: 모델 파일 경로
        """
        self.model = CatBoost()
        self.model.load_model(path, format="cbm")

    def _get_feature_importance(self, importance_type: str) -> Dict[str, float]:
        """
        CatBoost 특성 중요도 계산 구현

        Args:
            importance_type: 중요도 유형 ('PredictionValuesChange', 'LossFunctionChange', 등)

        Returns:
            특성 이름과 중요도 딕셔너리
        """
        if importance_type not in ["PredictionValuesChange", "LossFunctionChange"]:
            importance_type = "PredictionValuesChange"  # 기본값

        # 모델이 학습되었는지 확인
        if not self.is_trained or self.model is None:
            return {}

        # 특성 이름이 없는 경우
        if not self.feature_names:
            return {}

        # 특성 중요도 계산
        importance = self.model.get_feature_importance(type=importance_type)

        # 결과 반환
        return dict(zip(self.feature_names, importance))
