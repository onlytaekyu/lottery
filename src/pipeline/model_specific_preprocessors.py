"""
모델별 특화 전처리기

각 모델의 특성에 맞는 최적화된 전처리를 제공합니다:
- LightGBMPreprocessor: RobustScaler + 특성 선택 + Feature interaction
- AutoEncoderPreprocessor: MinMax(0,1) + PCA + 노이즈 제거
- TCNPreprocessor: StandardScaler + 시계열 윈도우 + 시간 특성
- RandomForestPreprocessor: 원본 유지 + 범주형 인코딩
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings

from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_config
from ..utils.memory_manager import get_memory_manager
from ..utils.unified_performance import performance_monitor
from ..utils.strict_validation import StrictPreprocessor, SafeTargetEncoder

logger = get_logger(__name__)


class BasePreprocessor:
    """공통 전처리 Base, StrictPreprocessor 래핑"""

    def __init__(self):
        self.strict_proc = StrictPreprocessor()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):  # noqa: D401,E501
        """Train 데이터에 대해서만 통계 적합"""
        self.strict_proc.fit_on_train_only(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        return self.strict_proc.transform_both(X)

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:  # noqa: E501
        return self.fit(X, y).transform(X)


class AutoEncoderPreprocessor(BasePreprocessor):
    """AE 전용: PCA 168→100, MinMax(0,1) 정규화, 대칭성 확보"""

    def __init__(self, n_components: int = 100):
        super().__init__()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):  # noqa: D401,E501
        # 이상치 클리핑 (IQR 방식)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1 + 1e-8
        X_clip = np.clip(X, Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        X_proc = super().fit_transform(X_clip, y)
        self.pca.fit(X_proc)
        X_pca = self.pca.transform(X_proc)
        self.scaler.fit(np.abs(X_pca))  # 대칭성 확보를 위해 절대값 기반 학습
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        # 동일 클리핑 적용
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1 + 1e-8
        X_clip = np.clip(X, Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        X_proc = super().transform(X_clip)
        X_pca = self.pca.transform(X_proc)
        X_scaled = self.scaler.transform(np.abs(X_pca))
        return X_scaled


class LightGBMPreprocessor(BasePreprocessor):
    """LightGBM 전용: PCA 금지, Target Encoding 포함"""

    def __init__(self, top_k_features: int | None = 200):
        super().__init__()
        self.target_encoder: SafeTargetEncoder | None = None
        self.top_k_features = top_k_features
        self.selected_idx: list[int] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):  # noqa: D401,E501
        # 첫 열이 범주형이라고 가정 (예시). 실제 구현 시 config 필요
        cat_feature = X[:, 0]
        num_features = X[:, 1:]
        super().fit(num_features)
        self.target_encoder = SafeTargetEncoder()
        _ = self.target_encoder.fit_transform_with_kfold(cat_feature, y)
        # 중요도 대용으로 분산 사용 → 상위 top_k_features 선택
        if (
            self.top_k_features is not None
            and num_features.shape[1] > self.top_k_features
        ):
            variances = np.var(num_features, axis=0)
            self.selected_idx = np.argsort(variances)[-self.top_k_features :]
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        if self.target_encoder is None:
            raise RuntimeError("TargetEncoder not fitted.")
        cat_feature = X[:, 0]
        num_features = X[:, 1:]
        num_proc = super().transform(num_features)
        if self.selected_idx is not None:
            num_proc = num_proc[:, self.selected_idx]
        enc_vals = self.target_encoder.transform(cat_feature)
        return np.column_stack([enc_vals, num_proc])


# --------------------- TCN Temporal Augmentation ---------------------


def _month_one_hot(month: int) -> np.ndarray:
    vec = np.zeros(12, dtype=float)
    vec[month - 1] = 1.0
    return vec


def tcn_temporal_features(
    draw_numbers: np.ndarray, window_size: int = 50
) -> np.ndarray:  # noqa: E501
    """회차 기반 시계열 특화 특성 생성(이동평균, 계절성, 속도, ACF 포함)"""
    draw_numbers = np.asarray(draw_numbers, dtype=int)
    max_draw = draw_numbers.max()
    min_draw = draw_numbers.min()
    norm_draw = (draw_numbers - min_draw) / (max_draw - min_draw + 1e-8)

    # delta features
    deltas = np.diff(draw_numbers, prepend=draw_numbers[0])
    avg_gap = deltas.mean() if len(deltas) else 1.0
    delta_norm = deltas / (avg_gap + 1e-8)

    # 이동평균 & ACF1
    ma = np.convolve(draw_numbers, np.ones(window_size) / window_size, mode="same")
    acf1 = np.concatenate([[0], np.diff(draw_numbers)]) / (avg_gap + 1e-8)

    # seasonal features (월별 ; 여기서는 회차 번호를 월로 매핑 예시)
    months = (draw_numbers % 12) + 1
    month_oh = np.vstack([_month_one_hot(m) for m in months])

    # trend velocity
    rolling_avg = np.convolve(
        draw_numbers, np.ones(window_size) / window_size, mode="same"
    )
    velocity = np.diff(rolling_avg, prepend=rolling_avg[0]) / (window_size + 1e-8)

    features = np.column_stack([norm_draw, delta_norm, velocity, ma, acf1])
    features = np.concatenate([features, month_oh], axis=1)
    return features


class TCNPreprocessor(BasePreprocessor):
    """TCN 전용: 시계열 특화 temporal_features 추가"""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):  # noqa: D401,E501
        # TCN은 주로 정규화 정도만 필요
        return super().fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        base_proc = super().transform(X)
        temporal = tcn_temporal_features(np.arange(X.shape[0]))
        # 길이 맞추기
        if temporal.shape[0] != base_proc.shape[0]:
            warnings.warn("Temporal feature length mismatch; resizing via repeat.")
            temporal = np.resize(temporal, (base_proc.shape[0], temporal.shape[1]))
        return np.concatenate([base_proc, temporal], axis=1)


class RandomForestPreprocessor(BasePreprocessor):
    """Random Forest 전용: 전체 특성 보존, 별도 변환 없음"""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        return X


# 팩토리 함수
def create_model_preprocessor(
    model_type: str, config: Optional[Dict[str, Any]] = None
) -> BasePreprocessor:
    """모델별 전처리기 생성"""
    preprocessors = {
        "lightgbm": LightGBMPreprocessor,
        "autoencoder": AutoEncoderPreprocessor,
        "tcn": TCNPreprocessor,
        "random_forest": RandomForestPreprocessor,
    }

    if model_type.lower() not in preprocessors:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    return preprocessors[model_type.lower()](config)


# 이전 시스템 호환성을 위한 별칭
BaseModelPreprocessor = BasePreprocessor
