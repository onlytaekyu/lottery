"""
Optimized LightGBM Preprocessor
LightGBM 최적화 전처리기 - GPU 가속, 카테고리 임베딩, 하이퍼파라미터 튜닝
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import lightgbm as lgb
import optuna
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import itertools
import warnings

warnings.filterwarnings("ignore")

from ..utils.unified_logging import get_logger
from ..utils.unified_memory_manager import get_unified_memory_manager
from ..shared.types import PreprocessedData

logger = get_logger(__name__)


class CategoricalEmbedder:
    """카테고리 특성 임베딩"""

    def __init__(self, embedding_dim: int = 32, method: str = "target_guided"):
        self.embedding_dim = embedding_dim
        self.method = method
        self.encoders = {}
        self.target_encoders = {}

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        categorical_features: List[int] = None,
    ):
        """임베딩 학습"""

        if categorical_features is None:
            # 자동으로 카테고리 특성 감지
            categorical_features = self._detect_categorical_features(X)

        for feat_idx in categorical_features:
            feature_data = X[:, feat_idx]

            # 라벨 인코딩
            le = LabelEncoder()
            encoded_data = le.fit_transform(feature_data.astype(str))
            self.encoders[feat_idx] = le

            # 타겟 기반 인코딩
            if y is not None and self.method == "target_guided":
                te = TargetEncoder(smooth="auto")
                te.fit(encoded_data.reshape(-1, 1), y)
                self.target_encoders[feat_idx] = te

    def transform(self, X: np.ndarray) -> np.ndarray:
        """임베딩 변환"""

        X_embedded = X.copy()

        for feat_idx, encoder in self.encoders.items():
            feature_data = X[:, feat_idx]

            # 라벨 인코딩
            try:
                encoded_data = encoder.transform(feature_data.astype(str))
            except ValueError:
                # 새로운 카테고리 처리
                encoded_data = np.zeros(len(feature_data))

            # 타겟 인코딩 적용
            if feat_idx in self.target_encoders:
                target_encoded = self.target_encoders[feat_idx].transform(
                    encoded_data.reshape(-1, 1)
                )
                X_embedded[:, feat_idx] = target_encoded.flatten()
            else:
                X_embedded[:, feat_idx] = encoded_data

        return X_embedded

    def _detect_categorical_features(self, X: np.ndarray) -> List[int]:
        """카테고리 특성 자동 감지"""
        categorical_features = []

        for i in range(X.shape[1]):
            unique_values = len(np.unique(X[:, i]))
            total_values = len(X[:, i])

            # 고유값 비율이 10% 이하이면 카테고리로 간주
            if unique_values / total_values <= 0.1 and unique_values <= 50:
                categorical_features.append(i)

        return categorical_features


class FeatureInteractionGenerator:
    """특성 상호작용 생성기"""

    def __init__(
        self, max_interactions: int = 50, selection_method: str = "lightgbm_importance"
    ):
        self.max_interactions = max_interactions
        self.selection_method = selection_method
        self.interaction_features = []
        self.feature_importance = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """상호작용 특성 학습"""

        # 기본 특성 중요도 계산
        self.feature_importance = self._calculate_feature_importance(X, y)

        # 상위 특성들 선택
        n_top_features = min(20, X.shape[1])
        top_features = np.argsort(self.feature_importance)[-n_top_features:]

        # 2차 상호작용 생성
        interaction_candidates = []
        for i, j in itertools.combinations(top_features, 2):
            interaction_candidates.append((i, j))

        # 상호작용 중요도 평가
        interaction_scores = []
        for i, j in interaction_candidates:
            interaction_feature = X[:, i] * X[:, j]
            score = self._evaluate_interaction(interaction_feature, y)
            interaction_scores.append((score, i, j))

        # 상위 상호작용 선택
        interaction_scores.sort(reverse=True)
        self.interaction_features = [
            (i, j) for _, i, j in interaction_scores[: self.max_interactions]
        ]

        logger.info(f"생성된 상호작용 특성: {len(self.interaction_features)}개")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """상호작용 특성 변환"""

        if not self.interaction_features:
            return X

        # 상호작용 특성 생성
        interaction_matrix = []
        for i, j in self.interaction_features:
            interaction_feature = X[:, i] * X[:, j]
            interaction_matrix.append(interaction_feature)

        interaction_matrix = np.array(interaction_matrix).T

        # 원본 특성과 결합
        X_enhanced = np.hstack([X, interaction_matrix])

        return X_enhanced

    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """특성 중요도 계산"""

        if self.selection_method == "lightgbm_importance":
            # LightGBM으로 특성 중요도 계산
            train_data = lgb.Dataset(X, label=y)
            params = {
                "objective": "multiclass" if len(np.unique(y)) > 2 else "binary",
                "num_class": len(np.unique(y)) if len(np.unique(y)) > 2 else 1,
                "metric": (
                    "multi_logloss" if len(np.unique(y)) > 2 else "binary_logloss"
                ),
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "verbose": -1,
            }

            model = lgb.train(params, train_data, num_boost_round=100)
            return model.feature_importance()

        else:
            # 상호정보량 기반 중요도
            selector = SelectKBest(score_func=mutual_info_classif, k="all")
            selector.fit(X, y)
            return selector.scores_

    def _evaluate_interaction(
        self, interaction_feature: np.ndarray, y: np.ndarray
    ) -> float:
        """상호작용 특성 평가"""

        # 상호정보량 계산
        from sklearn.feature_selection import mutual_info_classif

        score = mutual_info_classif(interaction_feature.reshape(-1, 1), y)[0]
        return score


class OptimizedLightGBMPreprocessor:
    """최적화된 LightGBM 전처리기"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.memory_manager = get_unified_memory_manager()

        # GPU 설정
        self.gpu_params = {
            "device_type": "gpu",
            "gpu_platform_id": 0,
            "gpu_use_dp": True,
            "max_bin": 255,
            "num_gpu": 1,
        }

        # 컴포넌트 초기화
        self.cat_embedder = CategoricalEmbedder(
            embedding_dim=32, method="target_guided"
        )

        self.interaction_generator = FeatureInteractionGenerator(
            max_interactions=50, selection_method="lightgbm_importance"
        )

        self.feature_selector = None
        self.best_params = None

    def preprocess_for_lightgbm(
        self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        LightGBM 최적화 전처리

        Args:
            X: 입력 특성
            y: 타겟 변수
            feature_names: 특성 이름 목록

        Returns:
            Tuple[np.ndarray, Dict]: 전처리된 데이터와 최적 파라미터
        """

        try:
            self.logger.info("LightGBM 최적화 전처리 시작")

            # 1. 카테고리 임베딩
            self.logger.info("카테고리 임베딩 적용...")
            self.cat_embedder.fit(X, y)
            X_embedded = self.cat_embedder.transform(X)

            # 2. 특성 상호작용 생성
            self.logger.info("특성 상호작용 생성...")
            self.interaction_generator.fit(X_embedded, y)
            X_enhanced = self.interaction_generator.transform(X_embedded)

            # 3. 특성 선택
            self.logger.info("특성 선택...")
            X_selected = self._select_features(X_enhanced, y, k_best=100)

            # 4. 하이퍼파라미터 최적화
            self.logger.info("하이퍼파라미터 최적화...")
            self.best_params = self._optimize_hyperparameters(
                X_selected, y, n_trials=100
            )

            self.logger.info("전처리 완료")

            return X_selected, self.best_params

        except Exception as e:
            self.logger.error(f"LightGBM 전처리 중 오류: {e}")
            # 오류 발생 시 원본 데이터와 기본 파라미터 반환
            return X, {}

    def _select_features(
        self, X: np.ndarray, y: np.ndarray, k_best: int = 100
    ) -> np.ndarray:
        """특성 선택"""

        if X.shape[1] <= k_best:
            return X

        # 상호정보량 기반 특성 선택
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif, k=min(k_best, X.shape[1])
        )

        X_selected = self.feature_selector.fit_transform(X, y)

        self.logger.info(f"특성 선택 완료: {X.shape[1]} -> {X_selected.shape[1]}")
        return X_selected

    def _optimize_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optuna를 이용한 하이퍼파라미터 최적화"""

        def objective(trial):
            # 하이퍼파라미터 탐색 공간 정의
            params = {
                "objective": "multiclass" if len(np.unique(y)) > 2 else "binary",
                "num_class": len(np.unique(y)) if len(np.unique(y)) > 2 else 1,
                "metric": (
                    "multi_logloss" if len(np.unique(y)) > 2 else "binary_logloss"
                ),
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 10, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "verbose": -1,
            }

            # GPU 설정 추가
            if self.gpu_params["device_type"] == "gpu":
                params.update(self.gpu_params)

            # 교차 검증으로 성능 평가
            train_data = lgb.Dataset(X, label=y)
            cv_results = lgb.cv(
                params,
                train_data,
                num_boost_round=200,
                nfold=5,
                stratified=True,
                shuffle=True,
                seed=42,
                return_cvbooster=False,
                eval_train_metric=False,
            )

            # 최적화 대상 메트릭 반환
            if len(np.unique(y)) > 2:
                return min(cv_results["valid multi_logloss-mean"])
            else:
                return min(cv_results["valid binary_logloss-mean"])

        # Optuna 최적화 실행
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # 최적 파라미터 반환
        best_params = study.best_params
        best_params.update(
            {
                "objective": "multiclass" if len(np.unique(y)) > 2 else "binary",
                "num_class": len(np.unique(y)) if len(np.unique(y)) > 2 else 1,
                "metric": (
                    "multi_logloss" if len(np.unique(y)) > 2 else "binary_logloss"
                ),
                "boosting_type": "gbdt",
                "verbose": -1,
            }
        )

        # GPU 설정 추가
        if self.gpu_params["device_type"] == "gpu":
            best_params.update(self.gpu_params)

        self.logger.info(
            f"하이퍼파라미터 최적화 완료 - 최적 점수: {study.best_value:.4f}"
        )

        return best_params

    def create_optimized_model(
        self, X: np.ndarray, y: np.ndarray
    ) -> lgb.LGBMClassifier:
        """최적화된 LightGBM 모델 생성"""

        if self.best_params is None:
            raise ValueError("먼저 전처리를 수행하여 최적 파라미터를 찾아야 합니다.")

        # LightGBM 분류기 생성
        model = lgb.LGBMClassifier(**self.best_params)

        # 모델 학습
        model.fit(X, y)

        return model

    def transform_new_data(self, X: np.ndarray) -> np.ndarray:
        """새로운 데이터 변환"""

        # 학습된 전처리 파이프라인 적용
        X_embedded = self.cat_embedder.transform(X)
        X_enhanced = self.interaction_generator.transform(X_embedded)

        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_enhanced)
        else:
            X_selected = X_enhanced

        return X_selected

    def get_feature_importance_analysis(
        self, model: lgb.LGBMClassifier, feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """특성 중요도 분석"""

        importance_gain = model.feature_importances_
        importance_split = model.booster_.feature_importance(importance_type="split")

        # 특성 이름 생성
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_gain))]

        # 중요도 분석 결과
        analysis = {
            "feature_names": feature_names,
            "importance_gain": importance_gain,
            "importance_split": importance_split,
            "top_features_gain": [],
            "top_features_split": [],
        }

        # 상위 특성 추출
        top_indices_gain = np.argsort(importance_gain)[-20:][::-1]
        top_indices_split = np.argsort(importance_split)[-20:][::-1]

        for idx in top_indices_gain:
            analysis["top_features_gain"].append(
                {"feature": feature_names[idx], "importance": importance_gain[idx]}
            )

        for idx in top_indices_split:
            analysis["top_features_split"].append(
                {"feature": feature_names[idx], "importance": importance_split[idx]}
            )

        return analysis

    def print_optimization_summary(self):
        """최적화 결과 요약 출력"""

        print("=" * 60)
        print("🚀 LightGBM 최적화 결과")
        print("=" * 60)

        if self.best_params:
            print("📊 최적 하이퍼파라미터:")
            for param, value in self.best_params.items():
                if param not in ["objective", "metric", "boosting_type", "verbose"]:
                    print(f"  • {param}: {value}")

        print(f"\n🔧 적용된 최적화 기법:")
        print(f"  • 카테고리 임베딩: Target-guided encoding")
        print(
            f"  • 특성 상호작용: 상위 {self.interaction_generator.max_interactions}개 생성"
        )
        print(f"  • 특성 선택: 상호정보량 기반")
        print(
            f"  • GPU 가속: {'활성화' if self.gpu_params['device_type'] == 'gpu' else '비활성화'}"
        )

        print("=" * 60)

    def preprocess(self, data: np.ndarray) -> PreprocessedData:
        """
        데이터를 전처리합니다.
        """
        # ... existing code ...
