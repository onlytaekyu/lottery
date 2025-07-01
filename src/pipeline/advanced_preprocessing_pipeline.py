"""
고급 전처리 파이프라인

Phase 1: 즉시 적용 가능한 ML/DL 성능 최적화
- Smart Feature Selection (상관관계 기반 중복 제거)
- Advanced Scaling & Normalization (모델별 특화)
- Intelligent Outlier Handling (Winsorizing)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
import warnings

from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_config
from ..utils.memory_manager import get_memory_manager
from ..utils.unified_performance import performance_monitor

logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """전처리 설정"""

    # Feature Selection
    variance_threshold: float = 0.001  # 개선: 0.01 → 0.001
    correlation_threshold: float = 0.95  # 신규: 상관관계 임계값
    max_features: int = 80  # 목표: 95개 → 80개

    # Outlier Handling
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)  # 1%, 99% 분위수
    outlier_method: str = "winsorize"  # "winsorize", "iqr", "zscore"

    # Model-specific scaling
    enable_model_specific_scaling: bool = True


class AdvancedPreprocessingPipeline:
    """고급 전처리 파이프라인"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = get_config("main") if config is None else get_config("main")
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()

        # 전처리 설정
        self.preprocessing_config = PreprocessingConfig()

        # 스케일러 초기화
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "quantile": QuantileTransformer(output_distribution="uniform"),
            "power": PowerTransformer(method="yeo-johnson"),
        }

        # 통계 저장
        self.preprocessing_stats = {}
        self.feature_importance_scores = None
        self.selected_features_mask = None

        self.logger.info("고급 전처리 파이프라인 초기화 완료")

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_model: str = "general",
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        전처리 파이프라인 실행

        Args:
            X: 특성 행렬
            feature_names: 특성 이름 리스트
            target_model: 타겟 모델 ("lightgbm", "autoencoder", "tcn", "random_forest", "general")

        Returns:
            Tuple[전처리된 특성, 선택된 특성 이름, 통계 정보]
        """
        self.logger.info(
            f"전처리 파이프라인 시작: {X.shape}, 타겟 모델: {target_model}"
        )

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # 1. 기본 검증
        X_processed, feature_names = self._validate_input(X, feature_names)

        # 2. Phase 1: 핵심 최적화
        X_processed, feature_names = self._apply_phase1_optimization(
            X_processed, feature_names
        )

        # 3. 모델별 특화 전처리
        X_processed, feature_names = self._apply_model_specific_preprocessing(
            X_processed, feature_names, target_model
        )

        # 4. 최종 검증
        X_processed = self._final_validation(X_processed)

        # 통계 정보 수집
        stats_info = self._collect_preprocessing_stats(X, X_processed, feature_names)

        self.logger.info(f"전처리 완료: {X.shape} → {X_processed.shape}")
        return X_processed, feature_names, stats_info

    def _validate_input(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """입력 데이터 검증"""
        # NaN/Inf 처리
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            self.logger.warning("NaN/Inf 값 발견, 0으로 대체")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 특성 이름 길이 맞춤
        if len(feature_names) != X.shape[1]:
            self.logger.warning(
                f"특성 이름 개수 불일치: {len(feature_names)} vs {X.shape[1]}"
            )
            feature_names = (
                feature_names[: X.shape[1]]
                if len(feature_names) > X.shape[1]
                else feature_names
                + [f"feature_{i}" for i in range(len(feature_names), X.shape[1])]
            )

        return X, feature_names

    def _apply_phase1_optimization(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Phase 1: 핵심 특성 최적화"""
        self.logger.info("Phase 1 최적화 적용 중...")

        # 1. Smart Feature Selection
        X, feature_names = self._smart_feature_selection(X, feature_names)

        # 2. Intelligent Outlier Handling
        X = self._handle_outliers_intelligently(X)

        return X, feature_names

    def _smart_feature_selection(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """스마트 특성 선택"""
        original_count = X.shape[1]

        # 1. Variance Threshold (개선된 임계값)
        variance_selector = VarianceThreshold(
            threshold=self.preprocessing_config.variance_threshold
        )
        X_var_selected = variance_selector.fit_transform(X)
        var_mask = variance_selector.get_support()

        selected_features = [
            name for name, selected in zip(feature_names, var_mask) if selected
        ]

        self.logger.info(
            f"분산 기반 선택: {original_count} → {len(selected_features)}개"
        )

        # 2. Correlation-based removal (신규)
        X_corr_selected, selected_features = self._remove_highly_correlated_features(
            X_var_selected, selected_features
        )

        self.logger.info(f"상관관계 기반 제거 후: {len(selected_features)}개")

        return X_corr_selected, selected_features

    def _remove_highly_correlated_features(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """고도로 상관된 특성 제거"""
        if X.shape[1] <= 1:
            return X, feature_names

        # 상관관계 행렬 계산
        try:
            corr_matrix = np.corrcoef(X, rowvar=False)

            # 제거할 특성 인덱스 찾기
            to_remove = set()
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    if (
                        abs(corr_matrix[i, j])
                        > self.preprocessing_config.correlation_threshold
                    ):
                        # 분산이 작은 특성 제거
                        var_i = np.var(X[:, i])
                        var_j = np.var(X[:, j])
                        to_remove.add(i if var_i < var_j else j)

            # 특성 제거
            keep_indices = [i for i in range(X.shape[1]) if i not in to_remove]
            X_filtered = X[:, keep_indices]
            filtered_names = [feature_names[i] for i in keep_indices]

            self.logger.info(f"상관관계 제거: {len(to_remove)}개 특성 제거")
            return X_filtered, filtered_names

        except Exception as e:
            self.logger.warning(f"상관관계 제거 실패: {e}")
            return X, feature_names

    def _handle_outliers_intelligently(self, X: np.ndarray) -> np.ndarray:
        """지능적 이상치 처리"""
        method = self.preprocessing_config.outlier_method

        if method == "winsorize":
            return self._winsorize_outliers(X)
        elif method == "iqr":
            return self._iqr_outlier_handling(X)
        elif method == "zscore":
            return self._zscore_outlier_handling(X)
        else:
            self.logger.warning(f"알 수 없는 이상치 처리 방법: {method}")
            return X

    def _winsorize_outliers(self, X: np.ndarray) -> np.ndarray:
        """Winsorizing 이상치 처리"""
        limits = self.preprocessing_config.winsorize_limits

        X_winsorized = np.zeros_like(X)
        outlier_count = 0

        for i in range(X.shape[1]):
            column = X[:, i]
            lower_bound = np.percentile(column, limits[0] * 100)
            upper_bound = np.percentile(column, limits[1] * 100)

            # 클리핑
            clipped = np.clip(column, lower_bound, upper_bound)
            outlier_count += np.sum((column < lower_bound) | (column > upper_bound))

            X_winsorized[:, i] = clipped

        self.logger.info(f"Winsorizing 완료: {outlier_count}개 이상치 처리")
        return X_winsorized

    def _iqr_outlier_handling(self, X: np.ndarray) -> np.ndarray:
        """IQR 기반 이상치 처리"""
        X_processed = np.zeros_like(X)

        for i in range(X.shape[1]):
            column = X[:, i]
            Q1 = np.percentile(column, 25)
            Q3 = np.percentile(column, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            X_processed[:, i] = np.clip(column, lower_bound, upper_bound)

        return X_processed

    def _zscore_outlier_handling(self, X: np.ndarray) -> np.ndarray:
        """Z-Score 기반 이상치 처리"""
        X_processed = np.zeros_like(X)

        for i in range(X.shape[1]):
            column = X[:, i]
            z_scores = np.abs(stats.zscore(column))

            # Z-Score 2.5 이상인 값들을 중간값으로 대체
            median_val = np.median(column)
            outlier_mask = z_scores > 2.5

            X_processed[:, i] = np.where(outlier_mask, median_val, column)

        return X_processed

    def _apply_model_specific_preprocessing(
        self, X: np.ndarray, feature_names: List[str], target_model: str
    ) -> Tuple[np.ndarray, List[str]]:
        """모델별 특화 전처리"""
        if not self.preprocessing_config.enable_model_specific_scaling:
            return X, feature_names

        if target_model == "lightgbm":
            return self._optimize_for_lightgbm(X, feature_names)
        elif target_model == "autoencoder":
            return self._optimize_for_autoencoder(X, feature_names)
        elif target_model == "tcn":
            return self._optimize_for_tcn(X, feature_names)
        elif target_model == "random_forest":
            return self._optimize_for_random_forest(X, feature_names)
        else:
            # General preprocessing
            scaler = self.scalers["robust"]
            X_scaled = scaler.fit_transform(X)
            return X_scaled, feature_names

    def _optimize_for_lightgbm(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """LightGBM 최적화"""
        # 1. RobustScaler 적용
        scaler = self.scalers["robust"]
        X_scaled = scaler.fit_transform(X)

        # 2. 특성 선택 (상위 70개)
        if X.shape[1] > 70:
            # 분산 기준으로 상위 특성 선택
            variances = np.var(X_scaled, axis=0)
            top_indices = np.argsort(variances)[-70:]

            X_selected = X_scaled[:, top_indices]
            selected_names = [feature_names[i] for i in top_indices]

            self.logger.info(
                f"LightGBM 특성 선택: {X.shape[1]} → {len(selected_names)}개"
            )
            return X_selected, selected_names

        return X_scaled, feature_names

    def _optimize_for_autoencoder(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """AutoEncoder 최적화"""
        # 1. MinMax(0,1) 정규화 (강제)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)

        # 2. PCA 차원 축소 (95% 분산 보존)
        if X.shape[1] > 100:
            pca = PCA(n_components=0.95)
            X_pca = pca.fit_transform(X_scaled)

            # 새로운 특성 이름 생성
            pca_names = [f"pca_component_{i}" for i in range(X_pca.shape[1])]

            self.logger.info(f"AutoEncoder PCA: {X.shape[1]} → {X_pca.shape[1]}차원")
            return X_pca, pca_names

        return X_scaled, feature_names

    def _optimize_for_tcn(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """TCN 최적화"""
        # 1. StandardScaler 적용
        scaler = self.scalers["standard"]
        X_scaled = scaler.fit_transform(X)

        return X_scaled, feature_names

    def _optimize_for_random_forest(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Random Forest 최적화"""
        # 트리 기반 모델은 스케일링 불필요
        self.logger.info("Random Forest: 모든 특성 유지 (스케일링 없음)")
        return X, feature_names

    def _final_validation(self, X: np.ndarray) -> np.ndarray:
        """최종 검증"""
        # NaN/Inf 재검사
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            self.logger.warning("최종 검증에서 NaN/Inf 발견, 수정")
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        return X

    def _collect_preprocessing_stats(
        self, X_original: np.ndarray, X_processed: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """전처리 통계 수집"""
        stats = {
            "original_shape": X_original.shape,
            "processed_shape": X_processed.shape,
            "feature_reduction_ratio": 1 - (X_processed.shape[1] / X_original.shape[1]),
            "selected_features": feature_names,
            "preprocessing_config": {
                "variance_threshold": self.preprocessing_config.variance_threshold,
                "correlation_threshold": self.preprocessing_config.correlation_threshold,
                "outlier_method": self.preprocessing_config.outlier_method,
                "winsorize_limits": self.preprocessing_config.winsorize_limits,
            },
        }

        # 특성별 통계
        if X_processed.size > 0:
            stats["feature_stats"] = {
                "mean": np.mean(X_processed, axis=0).tolist(),
                "std": np.std(X_processed, axis=0).tolist(),
                "min": np.min(X_processed, axis=0).tolist(),
                "max": np.max(X_processed, axis=0).tolist(),
            }

        return stats
