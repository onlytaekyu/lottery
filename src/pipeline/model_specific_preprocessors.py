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

logger = get_logger(__name__)


class BaseModelPreprocessor(ABC):
    """모델별 전처리기 기본 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = get_config("main") if config is None else get_config("main")
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()

        # 전처리 컴포넌트들
        self.scaler = None
        self.feature_selector = None
        self.dimensionality_reducer = None

        # 메타데이터
        self.is_fitted = False
        self.feature_names_in = None
        self.feature_names_out = None
        self.preprocessing_stats = {}

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "BaseModelPreprocessor":
        """전처리기 학습"""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """데이터 변환"""
        pass

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """학습 및 변환"""
        return self.fit(X, y, feature_names).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """출력 특성 이름 반환"""
        return self.feature_names_out or []

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """전처리 통계 반환"""
        return self.preprocessing_stats


class LightGBMPreprocessor(BaseModelPreprocessor):
    """LightGBM 최적화 전처리기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # LightGBM 특화 설정
        self.max_features = 70
        self.enable_feature_interactions = True

        # 컴포넌트 초기화
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.interaction_features = []

        self.logger.info("LightGBM 전처리기 초기화 완료")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LightGBMPreprocessor":
        """LightGBM 전처리기 학습"""
        self.logger.info(f"LightGBM 전처리기 학습 시작: {X.shape}")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.feature_names_in = feature_names

        # 1. RobustScaler 학습
        X_scaled = self.scaler.fit_transform(X)

        # 2. 특성 선택기 학습
        if X.shape[1] > self.max_features:
            if y is not None:
                # 지도 학습 기반 특성 선택
                self.feature_selector = SelectKBest(
                    score_func=f_classif, k=self.max_features
                )
                self.feature_selector.fit(X_scaled, y)
            else:
                # 분산 기반 특성 선택
                variances = np.var(X_scaled, axis=0)
                top_indices = np.argsort(variances)[-self.max_features :]
                self.selected_feature_indices = top_indices

        # 3. Feature Interaction 패턴 학습
        if self.enable_feature_interactions:
            self._learn_interaction_patterns(X_scaled, y)

        self.is_fitted = True
        self.logger.info("LightGBM 전처리기 학습 완료")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """LightGBM 전처리 변환"""
        if not self.is_fitted:
            raise ValueError("전처리기가 학습되지 않았습니다.")

        # 1. 스케일링
        X_scaled = self.scaler.transform(X)

        # 2. 특성 선택
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        elif hasattr(self, "selected_feature_indices"):
            X_selected = X_scaled[:, self.selected_feature_indices]
        else:
            X_selected = X_scaled

        return X_selected

    def _learn_interaction_patterns(self, X: np.ndarray, y: Optional[np.ndarray]):
        """상호작용 패턴 학습"""
        if X.shape[1] < 2:
            return

        # 상위 특성들만 사용 (계산 효율성)
        n_top_features = min(15, X.shape[1])
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-n_top_features:]

        # 유용한 상호작용 패턴 찾기
        from itertools import combinations

        interaction_candidates = []
        for i, j in combinations(top_indices, 2):
            # 곱셈 상호작용
            interaction = X[:, i] * X[:, j]

            # 상호작용의 유용성 평가 (분산 기준)
            interaction_var = np.var(interaction)
            if interaction_var > 1e-6:  # 의미있는 분산을 가진 경우만
                interaction_candidates.append((i, j, interaction_var))

        # 상위 상호작용들 선택
        interaction_candidates.sort(key=lambda x: x[2], reverse=True)
        self.interaction_features = interaction_candidates[:10]  # 상위 10개

        self.logger.info(f"상호작용 패턴 학습 완료: {len(self.interaction_features)}개")


class AutoEncoderPreprocessor(BaseModelPreprocessor):
    """AutoEncoder 최적화 전처리기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # AutoEncoder 특화 설정
        self.feature_range = (0, 1)
        self.pca_components = 0.95

        # 컴포넌트 초기화
        self.scaler = MinMaxScaler(feature_range=self.feature_range)
        self.pca = None

        self.logger.info("AutoEncoder 전처리기 초기화 완료")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "AutoEncoderPreprocessor":
        """AutoEncoder 전처리기 학습"""
        self.logger.info(f"AutoEncoder 전처리기 학습 시작: {X.shape}")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.feature_names_in = feature_names

        # 1. MinMax(0,1) 스케일링 학습
        X_scaled = self.scaler.fit_transform(X)

        # 2. PCA 학습 (차원 축소)
        if X.shape[1] > 50 and self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
            X_pca = self.pca.fit_transform(X_scaled)
            self.logger.info(f"PCA 적용: {X.shape[1]} → {X_pca.shape[1]}차원")

        self.is_fitted = True
        self.logger.info("AutoEncoder 전처리기 학습 완료")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """AutoEncoder 전처리 변환"""
        if not self.is_fitted:
            raise ValueError("전처리기가 학습되지 않았습니다.")

        # 1. MinMax 스케일링 (강제 0-1 범위)
        X_scaled = self.scaler.transform(X)

        # 2. PCA 변환
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)

        return X_scaled


class TCNPreprocessor(BaseModelPreprocessor):
    """TCN 최적화 전처리기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # TCN 특화 설정
        self.sequence_length = 50

        # 컴포넌트 초기화
        self.scaler = StandardScaler()

        self.logger.info("TCN 전처리기 초기화 완료")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "TCNPreprocessor":
        """TCN 전처리기 학습"""
        self.logger.info(f"TCN 전처리기 학습 시작: {X.shape}")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.feature_names_in = feature_names

        # StandardScaler 학습
        X_scaled = self.scaler.fit_transform(X)

        self.is_fitted = True
        self.logger.info("TCN 전처리기 학습 완료")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """TCN 전처리 변환"""
        if not self.is_fitted:
            raise ValueError("전처리기가 학습되지 않았습니다.")

        # StandardScaler 적용
        X_scaled = self.scaler.transform(X)

        return X_scaled


class RandomForestPreprocessor(BaseModelPreprocessor):
    """Random Forest 최적화 전처리기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.logger.info("Random Forest 전처리기 초기화 완료")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "RandomForestPreprocessor":
        """Random Forest 전처리기 학습"""
        self.logger.info(f"Random Forest 전처리기 학습 시작: {X.shape}")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.feature_names_in = feature_names

        self.is_fitted = True
        self.logger.info("Random Forest 전처리기 학습 완료")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Random Forest 전처리 변환"""
        if not self.is_fitted:
            raise ValueError("전처리기가 학습되지 않았습니다.")

        # Random Forest는 원본 데이터 그대로 사용
        return X


# 팩토리 함수
def create_model_preprocessor(
    model_type: str, config: Optional[Dict[str, Any]] = None
) -> BaseModelPreprocessor:
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
