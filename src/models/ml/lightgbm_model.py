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

        # 3자리 예측 모드 지원 활성화
        self.supports_3digit_mode = True
        self.three_digit_model = None
        self.three_digit_feature_names = []

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

        # 3자리 모드 전용 파라미터
        self.three_digit_params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",  # 다중 분류
            "metric": "multi_logloss",
            "num_class": 14190,  # C(45,3) = 14190
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.75,
            "bagging_freq": 3,
            "verbose": -1,
        }

        # 설정에서 하이퍼파라미터 업데이트 (있는 경우)
        self.params = self.default_params.copy()
        if config and "lightgbm" in config:
            self.params.update(config["lightgbm"])

        # 3자리 모드 파라미터 업데이트
        if config and "lightgbm_3digit" in config:
            self.three_digit_params.update(config["lightgbm_3digit"])

        # GPU 자동 감지 및 파라미터 적용
        if config and config.get("use_gpu", False):
            self.params["device_type"] = "gpu"
            self.three_digit_params["device_type"] = "gpu"

        logger.info(f"LightGBM 모델 초기화 완료: {self.params}")
        logger.info(f"3자리 모드 지원: {self.supports_3digit_mode}")

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

    # ===== 3자리 예측 모드 구현 =====

    def fit_3digit_mode(
        self, X: np.ndarray, y_3digit: np.ndarray, **kwargs
    ) -> Dict[str, Any]:
        """
        3자리 모드 전용 훈련

        Args:
            X: 특성 벡터
            y_3digit: 3자리 조합 레이블 (원-핫 인코딩 또는 클래스 인덱스)
            **kwargs: 추가 매개변수

        Returns:
            Dict[str, Any]: 훈련 결과
        """
        try:
            logger.info(
                f"LightGBM 3자리 모드 훈련 시작: X={X.shape}, y={y_3digit.shape}"
            )

            # 클래스 레이블 변환 (원-핫 → 클래스 인덱스)
            if y_3digit.ndim > 1 and y_3digit.shape[1] > 1:
                y_classes = np.argmax(y_3digit, axis=1)
            else:
                y_classes = y_3digit.astype(int)

            # 기본 매개변수 설정
            early_stopping_rounds = kwargs.get("early_stopping_rounds", 30)
            num_boost_round = kwargs.get("num_boost_round", 500)
            feature_names = kwargs.get(
                "feature_names", [f"3digit_feature_{i}" for i in range(X.shape[1])]
            )

            # 특성 이름 저장
            self.three_digit_feature_names = feature_names

            # 훈련/검증 데이터 준비
            if "eval_set" in kwargs:
                eval_set = kwargs["eval_set"]
                train_data = lgb.Dataset(X, label=y_classes, feature_name=feature_names)
                valid_data = lgb.Dataset(
                    eval_set[0][0], label=eval_set[0][1], feature_name=feature_names
                )
            else:
                from sklearn.model_selection import train_test_split

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_classes, test_size=0.2, random_state=42, stratify=y_classes
                )
                train_data = lgb.Dataset(
                    X_train, label=y_train, feature_name=feature_names
                )
                valid_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names)

            # 3자리 모델 훈련
            start_time = time.time()
            self.three_digit_model = lgb.train(
                self.three_digit_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=50,
            )

            train_time = time.time() - start_time

            # 메타데이터 업데이트
            self.metadata["3digit_mode"] = {
                "train_samples": X.shape[0],
                "features": X.shape[1],
                "num_classes": self.three_digit_params["num_class"],
                "best_iteration": self.three_digit_model.best_iteration,
                "best_score": self.three_digit_model.best_score,
                "train_time": train_time,
            }

            logger.info(
                f"3자리 모드 훈련 완료: 반복={self.three_digit_model.best_iteration}, "
                f"점수={self.three_digit_model.best_score}, 시간={train_time:.2f}초"
            )

            return {
                "best_iteration": self.three_digit_model.best_iteration,
                "best_score": self.three_digit_model.best_score,
                "train_time": train_time,
                "num_classes": self.three_digit_params["num_class"],
                "is_trained": True,
            }

        except Exception as e:
            logger.error(f"3자리 모드 훈련 중 오류: {e}")
            return {"error": str(e)}

    def predict_3digit_combinations(
        self, X: np.ndarray, top_k: int = 100, **kwargs
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        3자리 조합 예측

        Args:
            X: 특성 벡터
            top_k: 상위 k개 조합 반환
            **kwargs: 추가 매개변수

        Returns:
            List[Tuple[Tuple[int, int, int], float]]: (3자리 조합, 신뢰도) 리스트
        """
        if not self.is_3digit_mode:
            logger.error("3자리 모드가 활성화되지 않았습니다.")
            return []

        if self.three_digit_model is None:
            logger.error("3자리 모델이 훈련되지 않았습니다.")
            return []

        try:
            logger.info(f"LightGBM 3자리 예측 수행: 입력={X.shape}, top_k={top_k}")

            # 예측 수행 (확률)
            predictions = self.three_digit_model.predict(X)

            # 단일 샘플인 경우 차원 조정
            if predictions.ndim == 1:
                predictions = predictions.reshape(1, -1)

            # 3자리 조합 생성
            from itertools import combinations

            all_3digit_combos = list(combinations(range(1, 46), 3))

            results = []

            # 각 샘플에 대해 상위 k개 예측
            for sample_idx in range(predictions.shape[0]):
                sample_probs = predictions[sample_idx]

                # 상위 k개 클래스 인덱스 선택
                top_indices = np.argsort(sample_probs)[-top_k:][::-1]

                # 조합과 신뢰도 매핑
                for idx in top_indices:
                    if idx < len(all_3digit_combos):
                        combo = all_3digit_combos[idx]
                        confidence = float(sample_probs[idx])
                        results.append((combo, confidence))

            # 신뢰도 기준 정렬
            results.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"3자리 예측 완료: {len(results)}개 결과")
            return results[:top_k]

        except Exception as e:
            logger.error(f"3자리 예측 중 오류: {e}")
            return []

    def evaluate_3digit_mode(
        self, X: np.ndarray, y_3digit: np.ndarray, **kwargs
    ) -> Dict[str, Any]:
        """
        3자리 모드 평가

        Args:
            X: 특성 벡터
            y_3digit: 3자리 조합 레이블
            **kwargs: 추가 매개변수

        Returns:
            Dict[str, Any]: 평가 결과
        """
        if not self.is_3digit_mode or self.three_digit_model is None:
            return {"error": "3자리 모드가 활성화되지 않거나 모델이 훈련되지 않음"}

        try:
            # 클래스 레이블 변환
            if y_3digit.ndim > 1 and y_3digit.shape[1] > 1:
                y_true = np.argmax(y_3digit, axis=1)
            else:
                y_true = y_3digit.astype(int)

            # 예측 수행
            predictions = self.three_digit_model.predict(X)
            y_pred = np.argmax(predictions, axis=1)

            # 정확도 계산
            accuracy = np.mean(y_true == y_pred)

            # Top-k 정확도 계산
            top_k_accuracies = {}
            for k in [1, 5, 10, 20, 50]:
                top_k_pred = np.argsort(predictions, axis=1)[:, -k:]
                top_k_acc = np.mean(
                    [y_true[i] in top_k_pred[i] for i in range(len(y_true))]
                )
                top_k_accuracies[f"top_{k}_accuracy"] = top_k_acc

            # 신뢰도 통계
            max_probs = np.max(predictions, axis=1)
            confidence_stats = {
                "mean_confidence": np.mean(max_probs),
                "std_confidence": np.std(max_probs),
                "min_confidence": np.min(max_probs),
                "max_confidence": np.max(max_probs),
            }

            results = {
                "accuracy": accuracy,
                "samples_evaluated": len(y_true),
                **top_k_accuracies,
                **confidence_stats,
            }

            logger.info(f"3자리 모드 평가 완료: 정확도={accuracy:.4f}")
            return results

        except Exception as e:
            logger.error(f"3자리 모드 평가 중 오류: {e}")
            return {"error": str(e)}

    def _extract_window_features(self, window_data: List) -> np.ndarray:
        """
        LightGBM용 윈도우 특성 추출 (BaseModel 메서드 재정의)

        Args:
            window_data: 윈도우 당첨 번호 데이터

        Returns:
            np.ndarray: 추출된 특성 벡터
        """
        try:
            # 기본 특성 추출
            base_features = super()._extract_window_features(window_data)

            # LightGBM 특화 추가 특성
            all_numbers = []
            for draw in window_data:
                all_numbers.extend(draw.numbers)

            if all_numbers:
                # 고급 통계 특성
                advanced_features = np.array(
                    [
                        np.percentile(all_numbers, 25),  # Q1
                        np.percentile(all_numbers, 75),  # Q3
                        np.var(all_numbers),  # 분산
                        len(np.unique(all_numbers)) / len(all_numbers),  # 고유성 비율
                        np.sum(np.array(all_numbers) % 2)
                        / len(all_numbers),  # 홀수 비율
                    ]
                )
            else:
                advanced_features = np.zeros(5)

            # 특성 결합
            combined_features = np.concatenate([base_features, advanced_features])

            return combined_features

        except Exception as e:
            logger.error(f"LightGBM 윈도우 특성 추출 중 오류: {e}")
            return super()._extract_window_features(window_data)


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
