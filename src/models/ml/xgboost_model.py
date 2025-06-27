"""
XGBoost 모델

이 모듈은 XGBoost 알고리즘 기반 로또 번호 예측 모델을 구현합니다.
feature_vector_full.npy 또는 필터링된 벡터를 입력으로 사용하여 점수를 예측합니다.
"""

import os
import json
import numpy as np
import xgboost as xgb
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time

from ..base_model import BaseModel
from ...utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost 기반 로또 번호 예측 모델

    고성능 그래디언트 부스팅 구현을 통해 로또 번호의 당첨 확률을 예측합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        XGBoost 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)
        self.model = None
        self.feature_names = []
        self.model_name = "XGBoostModel"

        # 기본 하이퍼파라미터 설정
        self.default_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "auto",
            "verbosity": 0,
        }

        # 설정에서 하이퍼파라미터 업데이트 (있는 경우)
        self.params = self.default_params.copy()
        if config and "xgboost" in config:
            self.params.update(config["xgboost"])

        logger.info(f"XGBoost 모델 초기화 완료: {self.params}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        XGBoost 모델 학습

        Args:
            X: 특성 벡터
            y: 타겟 점수
            **kwargs: 추가 매개변수 (eval_set, early_stopping_rounds 등)

        Returns:
            학습 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"XGBoost 모델 학습 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 기본 매개변수 설정
        early_stopping_rounds = kwargs.get("early_stopping_rounds", 50)
        num_boost_round = kwargs.get("num_boost_round", 1000)
        feature_names = kwargs.get(
            "feature_names", [f"feature_{i}" for i in range(X.shape[1])]
        )

        # 특성 이름 저장
        self.feature_names = feature_names

        # DMatrix 생성
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

        # 평가 세트 설정
        evals = []
        if "eval_set" in kwargs:
            eval_set = kwargs["eval_set"]
            dval = xgb.DMatrix(
                eval_set[0][0], label=eval_set[0][1], feature_names=feature_names
            )
            evals = [(dtrain, "train"), (dval, "valid")]
        else:
            # 검증 세트가 없으면 훈련 데이터를 평가에 사용
            evals = [(dtrain, "train")]

        # 학습 시작
        start_time = time.time()
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100,
        )

        # 학습 시간 계산
        train_time = time.time() - start_time

        # 특성 중요도 계산
        importance = self.model.get_score(importance_type="gain")

        # 모든 특성에 대한 중요도를 0으로 초기화
        feature_importance = {name: 0.0 for name in feature_names}

        # 모델이 반환한 중요도로 업데이트
        feature_importance.update(importance)

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
            f"XGBoost 모델 학습 완료: 최적 반복={self.model.best_iteration}, 최적 점수={self.model.best_score}, 소요 시간={train_time:.2f}초"
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
        XGBoost 모델 예측 수행

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

        logger.info(f"XGBoost 예측 수행: 입력 형태={X.shape}")

        # DMatrix 변환
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)

        # 예측 수행
        predictions = self.model.predict(dtest)

        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        XGBoost 모델 평가

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

        logger.info(f"XGBoost 모델 평가: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        return {"rmse": rmse, "mae": mae, "r2": r2, "model_type": self.model_name}

    def save(self, path: str) -> bool:
        """
        XGBoost 모델 저장

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

            logger.info(f"XGBoost 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"XGBoost 모델 저장 중 오류: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        XGBoost 모델 로드

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
            self.model = xgb.Booster()
            self.model.load_model(path)

            # 메타데이터 로드
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    self.feature_names = meta_data.get("feature_names", [])
                    self.params = meta_data.get("params", self.default_params)
                    self.metadata = meta_data.get("metadata", {})

            # 훈련 완료 표시
            self.is_trained = True
            logger.info(f"XGBoost 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"XGBoost 모델 로드 중 오류: {e}")
            return False

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        특성 중요도 반환

        Args:
            importance_type: 중요도 타입 (gain, weight, cover, total_gain, total_cover)

        Returns:
            특성 이름 및 중요도 딕셔너리
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 모델에서 특성 중요도 가져오기
        importance = self.model.get_score(importance_type=importance_type)

        # 모든 특성에 대한 중요도를 0으로 초기화
        all_importance = {name: 0.0 for name in self.feature_names}

        # 모델이 반환한 중요도로 업데이트
        all_importance.update(importance)

        return all_importance


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
