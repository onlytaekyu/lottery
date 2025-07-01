# 1. 표준 라이브러리
import os
from typing import Any, Dict, Tuple, Optional

# 2. 서드파티
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# 3. 프로젝트 내부
from ..analysis.negative_sample_generator import NegativeSampleGenerator
from ..pipeline.model_specific_preprocessors import (
    LightGBMPreprocessor,
    TCNPreprocessor,
    AutoEncoderPreprocessor,
)
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """데이터 전처리 통합 파이프라인"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scaler_type = self.config.get("scaler", "minmax")  # minmax, zscore
        self.use_pca = self.config.get("use_pca", False)
        self.pca_components = self.config.get("pca_components", 50)
        self.test_size = self.config.get("test_size", 0.2)
        self.val_size = self.config.get("val_size", 0.1)
        self.random_state = self.config.get("random_state", 42)
        self.negative_sample_size = self.config.get("negative_sample_size", 10000)
        self.model_type = self.config.get("model_type", "lightgbm")

    def run(self, raw_data: np.ndarray, labels: np.ndarray = None) -> Dict[str, Any]:
        # 1. 특성 벡터화 (여기서는 이미 벡터라고 가정)
        X = raw_data.copy()
        y = labels.copy() if labels is not None else None

        # 2. 정규화/스케일링
        if self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. 차원 축소 (옵션)
        if self.use_pca:
            pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X_proc = pca.fit_transform(X_scaled)
        else:
            X_proc = X_scaled

        # 4. 데이터 분할 (train/val/test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_proc,
            y,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            stratify=y,
        )
        val_ratio = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_ratio,
            random_state=self.random_state,
            stratify=y_temp,
        )

        # 5. Negative sampling (비당첨 샘플)
        neg_gen = NegativeSampleGenerator(self.config)
        neg_samples = neg_gen.generate_samples([], self.negative_sample_size)[
            "raw_path"
        ]
        # (실제 사용 시 draw_data 필요, 여기선 예시)

        # 6. 모델별 입력 변환
        if self.model_type.lower() == "lightgbm":
            preproc = LightGBMPreprocessor()
            X_train_model = preproc.fit_transform(X_train, y_train)
            X_val_model = preproc.transform(X_val)
            X_test_model = preproc.transform(X_test)
        elif self.model_type.lower() in ["tcn", "lstm"]:
            preproc = TCNPreprocessor()
            X_train_model = preproc.fit_transform(X_train, y_train)
            X_val_model = preproc.transform(X_val)
            X_test_model = preproc.transform(X_test)
        elif self.model_type.lower() == "autoencoder":
            preproc = AutoEncoderPreprocessor()
            X_train_model = preproc.fit_transform(X_train, y_train)
            X_val_model = preproc.transform(X_val)
            X_test_model = preproc.transform(X_test)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")

        return {
            "X_train": X_train_model,
            "X_val": X_val_model,
            "X_test": X_test_model,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "neg_samples_path": neg_samples,
        }
