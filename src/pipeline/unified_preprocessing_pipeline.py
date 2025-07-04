# 1. 표준 라이브러리
import os
import pickle
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import warnings

# 2. 서드파티
import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# 3. 프로젝트 내부
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_config
from ..utils.memory_manager import get_memory_manager
from ..utils.unified_performance import performance_monitor
from ..utils.cache_paths import get_cache_dir
from ..analysis.negative_sample_generator import NegativeSampleGenerator
from .model_specific_preprocessors import (
    LightGBMPreprocessor,
    TCNPreprocessor,
    AutoEncoderPreprocessor,
)

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    X_processed: np.ndarray
    feature_names: List[str]
    preprocessing_stats: Dict[str, Any]
    model_type: str
    processing_time: float
    cache_key: Optional[str] = None


class UnifiedPreprocessingPipeline:
    """
    통합 전처리 파이프라인 (GPU/배치/캐시/병렬/모델별 최적화)
    지원 모델: LightGBM(GPU), TCN(PyTorch), AutoEncoder(PyTorch), RandomForest(cuML)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = get_config("main") if config is None else config
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()
        self.cache_dir = get_cache_dir() / "preprocessing"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_caching = self.config.get("preprocessing", {}).get(
            "enable_caching", True
        )
        self.use_gpu = self.config.get("use_gpu", False)
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "quantile": QuantileTransformer(output_distribution="uniform"),
            "power": PowerTransformer(method="yeo-johnson"),
        }
        self.variance_threshold = self.config.get("variance_threshold", 0.001)
        self.correlation_threshold = self.config.get("correlation_threshold", 0.95)
        self.winsorize_limits = self.config.get("winsorize_limits", (0.01, 0.99))
        self.outlier_method = self.config.get("outlier_method", "winsorize")
        self.pca_components = self.config.get("pca_components", 50)
        self.use_pca = self.config.get("use_pca", False)
        self.negative_sample_size = self.config.get("negative_sample_size", 10000)
        self.performance_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "processing_times": [],
            "feature_reduction_ratios": [],
        }
        self.logger.info("통합 전처리 파이프라인 초기화 완료")

    @performance_monitor
    def preprocess_for_model(
        self,
        X: np.ndarray,
        feature_names: List[str],
        model_type: str,
        y: Optional[np.ndarray] = None,
        use_cache: bool = True,
    ) -> PreprocessingResult:
        start_time = datetime.now()
        cache_key = None
        if use_cache and self.enable_caching:
            cache_key = self._generate_cache_key(X, feature_names, model_type)
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats["cache_hits"] += 1
                self.logger.info(f"캐시에서 전처리 결과 로드: {cache_key}")
                return cached_result
        # GPU/CPU 선택
        X_proc = self._to_device(X)
        # 1. Feature selection (variance)
        X_proc, feature_names = self._variance_filter(X_proc, feature_names)
        # 2. Outlier
        X_proc = self._handle_outliers(X_proc)
        # 3. Scaling
        X_proc = self._scale(X_proc)
        # 4. PCA
        if self.use_pca:
            X_proc = self._pca(X_proc)
        # 5. Negative sampling (optional)
        neg_samples = None
        if self.negative_sample_size > 0:
            neg_gen = NegativeSampleGenerator(self.config)
            neg_samples = neg_gen.generate_samples([], self.negative_sample_size)[
                "raw_path"
            ]
        # 6. Model-specific
        X_final = self._model_specific(X_proc, y, model_type)
        # 7. CPU로 복귀
        if CUPY_AVAILABLE and isinstance(X_final, cp.ndarray):
            X_final = cp.asnumpy(X_final)
        processing_time = (datetime.now() - start_time).total_seconds()
        result = PreprocessingResult(
            X_processed=X_final,
            feature_names=feature_names,
            preprocessing_stats={
                "processing_time": processing_time,
                "neg_samples_path": neg_samples,
            },
            model_type=model_type,
            processing_time=processing_time,
            cache_key=cache_key,
        )
        if use_cache and self.enable_caching and cache_key:
            self._save_to_cache(cache_key, result)
        self.performance_stats["total_processed"] += 1
        self.performance_stats["processing_times"].append(processing_time)
        return result

    def preprocess_batch(
        self,
        datasets: List[Tuple[np.ndarray, List[str], str]],
        y: Optional[np.ndarray] = None,
        parallel: bool = True,
    ) -> List[PreprocessingResult]:
        results = []
        # 병렬 처리(향후 확장), 현재는 순차
        for X, feature_names, model_type in datasets:
            results.append(self.preprocess_for_model(X, feature_names, model_type, y))
        return results

    def _to_device(self, X):
        if self.use_gpu:
            if CUPY_AVAILABLE:
                return cp.asarray(X)
            elif TORCH_AVAILABLE:
                return torch.tensor(X, device="cuda")
        return X

    def _variance_filter(self, X, feature_names):
        if isinstance(X, np.ndarray):
            selector = VarianceThreshold(self.variance_threshold)
            X_var = selector.fit_transform(X)
            mask = selector.get_support()
            feature_names = [f for f, m in zip(feature_names, mask) if m]
            return X_var, feature_names
        # cuPy/torch 지원(간단화)
        return X, feature_names

    def _handle_outliers(self, X):
        if self.outlier_method == "winsorize":
            return self._winsorize(X)
        # TODO: IQR, zscore 등 추가
        return X

    def _winsorize(self, X):
        if isinstance(X, np.ndarray):
            lower, upper = np.quantile(X, self.winsorize_limits, axis=0)
            X = np.clip(X, lower, upper)
            return X
        # cuPy 지원
        if CUPY_AVAILABLE and isinstance(X, cp.ndarray):
            lower, upper = cp.quantile(X, self.winsorize_limits, axis=0)
            X = cp.clip(X, lower, upper)
            return X
        return X

    def _scale(self, X):
        scaler = self.scalers.get(self.config.get("scaler", "minmax"), MinMaxScaler())
        if isinstance(X, np.ndarray):
            return scaler.fit_transform(X)
        if CUPY_AVAILABLE and isinstance(X, cp.ndarray):
            # cuPy용 스케일러(간단화)
            X_np = cp.asnumpy(X)
            X_scaled = scaler.fit_transform(X_np)
            return cp.asarray(X_scaled)
        return X

    def _pca(self, X):
        if isinstance(X, np.ndarray):
            pca = PCA(n_components=self.pca_components)
            return pca.fit_transform(X)
        if CUPY_AVAILABLE and isinstance(X, cp.ndarray):
            # cuML PCA 등 확장 가능
            return X  # TODO: cuML PCA 적용
        return X

    def _model_specific(self, X, y, model_type):
        mt = model_type.lower()
        if mt == "lightgbm":
            preproc = LightGBMPreprocessor()
            return preproc.fit_transform(X, y)
        elif mt in ["tcn"]:
            preproc = TCNPreprocessor()
            return preproc.fit_transform(X, y)
        elif mt == "autoencoder":
            preproc = AutoEncoderPreprocessor()
            return preproc.fit_transform(X, y)
        elif mt == "random_forest":
            # cuML/sklearn 분기 등 추가 가능
            return X
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    def _generate_cache_key(self, X, feature_names, model_type):
        return f"{model_type}_{len(feature_names)}_{hash(X.tobytes())}"

    def _load_from_cache(self, cache_key):
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, cache_key, result):
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    def get_performance_report(self) -> Dict[str, Any]:
        return self.performance_stats
