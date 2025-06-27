"""
ML 모델

이 모듈은 머신러닝 기반 로또 번호 예측 모델을 구현합니다.
LightGBM과 XGBoost 알고리즘을 통합하여 효율적으로 관리합니다.
"""

import os
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# ML 라이브러리 임포트
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..base_model import BaseModel
from ...utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class MLBaseModel(BaseModel):
    """
    머신러닝 모델의 기본 클래스

    LightGBM과 XGBoost 모델의 공통 메서드를 구현합니다.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, model_name: str = "MLBaseModel"
    ):
        """
        ML 기본 모델 초기화

        Args:
            config: 모델 설정
            model_name: 모델 이름
        """
        super().__init__(config)
        self.model = None
        self.feature_names = []
        self.model_name = model_name
        self.params = {}
        self.default_params = {}

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 평가 (공통 로직)

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
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        logger.info(
            f"{self.model_name} 모델 평가: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
        )

        return {"rmse": rmse, "mae": mae, "r2": r2, "model_type": self.model_name}

    def save(self, path: str) -> bool:
        """
        모델 저장 (공통 로직)

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

            # 모델 저장 (하위 클래스에서 구현)
            self._save_model(model_path)

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
                    self.feature_names = metadata.get("feature_names", [])
                    self.params = metadata.get("params", {})
                    self.metadata = metadata.get("metadata", {})

            # 로드 완료 표시
            self.is_trained = True
            logger.info(f"{self.model_name} 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"{self.model_name} 모델 로드 중 오류: {str(e)}")
            return False

    def _save_model(self, path: str) -> None:
        """모델 저장 구현 (하위 클래스에서 오버라이드)"""
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def _load_model(self, path: str) -> None:
        """모델 로드 구현 (하위 클래스에서 오버라이드)"""
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        특성 중요도 반환 (공통 로직)

        Args:
            importance_type: 중요도 유형 ('gain', 'split', 'weight')

        Returns:
            특성 이름과 중요도 딕셔너리
        """
        if not self.is_trained or self.model is None:
            logger.warning("학습되지 않은 모델의 특성 중요도를 얻을 수 없습니다.")
            return {}

        try:
            # 하위 클래스에서 구현
            return self._get_feature_importance(importance_type)

        except Exception as e:
            logger.error(f"특성 중요도 계산 중 오류: {str(e)}")
            return {}

    def _get_feature_importance(self, importance_type: str) -> Dict[str, float]:
        """특성 중요도 계산 구현 (하위 클래스에서 오버라이드)"""
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")


class LightGBMModel(MLBaseModel):
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
        super().__init__(config, model_name="LightGBMModel")

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

    def _save_model(self, path: str) -> None:
        """
        LightGBM 모델 저장 구현

        Args:
            path: 저장 경로
        """
        self.model.save_model(path)

    def _load_model(self, path: str) -> None:
        """
        LightGBM 모델 로드 구현

        Args:
            path: 모델 파일 경로
        """
        self.model = lgb.Booster(model_file=path)

    def _get_feature_importance(self, importance_type: str) -> Dict[str, float]:
        """
        LightGBM 특성 중요도 계산 구현

        Args:
            importance_type: 중요도 유형 ('gain', 'split')

        Returns:
            특성 이름과 중요도 딕셔너리
        """
        importance = self.model.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names, importance.tolist()))


class XGBoostModel(MLBaseModel):
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
        super().__init__(config, model_name="XGBoostModel")

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

    def _save_model(self, path: str) -> None:
        """
        XGBoost 모델 저장 구현

        Args:
            path: 저장 경로
        """
        self.model.save_model(path)

    def _load_model(self, path: str) -> None:
        """
        XGBoost 모델 로드 구현

        Args:
            path: 모델 파일 경로
        """
        self.model = xgb.Booster()
        self.model.load_model(path)

    def _get_feature_importance(self, importance_type: str) -> Dict[str, float]:
        """
        XGBoost 특성 중요도 계산 구현

        Args:
            importance_type: 중요도 유형 ('gain', 'weight', 'cover', 'total_gain', 'total_cover')

        Returns:
            특성 이름과 중요도 딕셔너리
        """
        importance = self.model.get_score(importance_type=importance_type)

        # 모든 특성에 대한 중요도를 0으로 초기화
        result = {name: 0.0 for name in self.feature_names}

        # 모델이 반환한 중요도로 업데이트
        result.update(importance)

        return result
