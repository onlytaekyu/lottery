"""
LightGBM 모델

이 모듈은 LightGBM 알고리즘 기반 로또 번호 예측 모델을 구현합니다.
feature_vector_full.npy 또는 필터링된 벡터를 입력으로 사용하여 점수를 예측합니다.
"""

import os
import json
import numpy as np
import lightgbm as lgb
import joblib
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time

from ..base_model import BaseModel
from ...utils.unified_logging import get_logger
from ...utils.model_saver import save_model, load_model

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    """
    LightGBM 기반 로또 번호 예측 모델

    빠르고 효율적인 그래디언트 부스팅 구현을 통해 로또 번호의 점수를 예측합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        LightGBM 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)
        self.model = None
        self.feature_names = []
        self.model_name = "LightGBMModel"

        # 기본 하이퍼파라미터 설정
        self.default_params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        # 설정에서 하이퍼파라미터 업데이트 (있는 경우)
        self.params = self.default_params.copy()
        if config and "lightgbm" in config:
            self.params.update(config["lightgbm"])

        # GPU 자동 감지 및 파라미터 적용
        if config and config.get("use_gpu", False):
            self.params["device_type"] = "gpu"

        logger.info(f"LightGBM 모델 초기화 완료: {self.params}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        LightGBM 모델 학습

        Args:
            X: 특성 벡터
            y: 타겟 점수
            **kwargs: 추가 매개변수 (eval_set, early_stopping_rounds 등)

        Returns:
            학습 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"LightGBM 모델 학습 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 기본 매개변수 설정
        early_stopping_rounds = kwargs.get("early_stopping_rounds", 50)
        num_boost_round = kwargs.get("num_boost_round", 1000)
        feature_names = kwargs.get(
            "feature_names", [f"feature_{i}" for i in range(X.shape[1])]
        )

        # 특성 이름 저장
        self.feature_names = feature_names

        # 훈련/검증 데이터 준비
        if "eval_set" in kwargs:
            eval_set = kwargs["eval_set"]
            train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
            valid_data = lgb.Dataset(
                eval_set[0][0], label=eval_set[0][1], feature_name=feature_names
            )
        else:
            # 검증 세트가 없으면 훈련 데이터의 20%를 검증에 사용
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            valid_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names)

        # 학습 시작
        start_time = time.time()
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100,
        )

        # 학습 시간 계산
        train_time = time.time() - start_time

        # 특성 중요도 계산
        importance = self.model.feature_importance(importance_type="gain")
        feature_importance = dict(zip(feature_names, importance.tolist()))

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
                "best_iteration": self.model.best_iteration,
                "best_score": self.model.best_score,
                "train_time": train_time,
                "feature_importance": feature_importance,
            }
        )

        # 훈련 완료 표시
        self.is_trained = True
        logger.info(
            f"LightGBM 모델 학습 완료: 최적 반복={self.model.best_iteration}, 최적 점수={self.model.best_score}, 소요 시간={train_time:.2f}초"
        )

        return {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        LightGBM 모델 예측 수행

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

        logger.info(f"LightGBM 예측 수행: 입력 형태={X.shape}")

        # 예측 수행
        predictions = self.model.predict(X)

        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        LightGBM 모델 평가

        Args:
            X: 특성 벡터
            y: 실제 타겟
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 예측 수행
        y_pred = self.predict(X)

        # 평가 메트릭 계산
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        logger.info(f"LightGBM 모델 평가: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        return {"rmse": rmse, "mae": mae, "r2": r2, "model_type": self.model_name}

    def save(self, path: str) -> bool:
        """
        LightGBM 모델 저장

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        if not self.is_trained or self.model is None:
            logger.warning("학습되지 않은 모델은 저장할 수 없습니다.")
            return False

        try:
            # 디렉토리 확인
            self._ensure_directory(path)

            # 모델 파일 경로
            model_path = path

            # 메타데이터 파일 경로
            meta_path = f"{os.path.splitext(path)[0]}.meta.json"

            # 모델 저장
            self.model.save_model(model_path)

            # 메타데이터 저장
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_type": self.model_name,
                        "feature_names": self.feature_names,
                        "params": self.params,
                        "metadata": self.metadata,
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            logger.info(f"LightGBM 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"LightGBM 모델 저장 중 오류: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        LightGBM 모델 로드

        Args:
            path: 모델 경로

        Returns:
            성공 여부
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(path):
                raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {path}")

            # 메타데이터 파일 경로
            meta_path = f"{os.path.splitext(path)[0]}.meta.json"

            # 모델 로드
            self.model = lgb.Booster(model_file=path)

            # 메타데이터 로드
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    self.feature_names = meta_data.get("feature_names", [])
                    self.params = meta_data.get("params", self.default_params)
                    self.metadata = meta_data.get("metadata", {})

            # 훈련 완료 표시
            self.is_trained = True
            logger.info(f"LightGBM 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"LightGBM 모델 로드 중 오류: {e}")
            return False

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        특성 중요도 반환

        Args:
            importance_type: 중요도 타입 (gain, split)

        Returns:
            특성 이름 및 중요도 딕셔너리
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        importance = self.model.feature_importance(importance_type=importance_type)

        if not self.feature_names:
            # 특성 이름이 없으면 자동 생성
            self.feature_names = [f"feature_{i}" for i in range(len(importance))]

        return dict(zip(self.feature_names, importance.tolist()))


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
