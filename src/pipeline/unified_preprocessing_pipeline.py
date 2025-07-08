# 1. 표준 라이브러리
import asyncio
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

# 2. 서드파티
import numpy as np

CUPY_AVAILABLE = False
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    pass
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
from ..utils.unified_memory_manager import get_unified_memory_manager
from ..utils.cuda_optimizers import get_cuda_optimizer
from ..utils.enhanced_process_pool import get_enhanced_process_pool
from ..utils.unified_async_manager import get_unified_async_manager
from ..utils.cache_manager import UnifiedCachePathManager
from ..utils.cache_manager import CacheManager
from ..analysis.negative_sample_generator import NegativeSampleGenerator
from .model_specific_preprocessors import (
    LightGBMPreprocessor,
    TCNPreprocessor,
    AutoEncoderPreprocessor,
)

logger = get_logger(__name__)


@dataclass
class PreprocessingOptimizationConfig:
    """전처리 파이프라인 최적화 설정"""
    # GPU 최적화
    enable_gpu_acceleration: bool = True
    prefer_cupy: bool = True
    gpu_memory_limit: float = 0.8
    
    # 병렬 처리
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 8
    batch_size: int = 10000
    
    # 비동기 처리
    enable_async: bool = True
    async_chunk_size: int = 5000
    
    # 메모리 관리
    enable_memory_optimization: bool = True
    memory_efficient_mode: bool = True
    
    # 캐시 설정
    enable_advanced_cache: bool = True
    cache_ttl: int = 3600  # 1시간
    max_cache_size: int = 1000
    
    # 성능 모니터링
    enable_monitoring: bool = True
    profiling_enabled: bool = True


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
    통합 전처리 파이프라인 v2.0 (완전 최적화)
    - GPU/배치/캐시/병렬/모델별/비동기 최적화
    - 지원 모델: LightGBM(GPU), TCN(PyTorch), AutoEncoder(PyTorch), RandomForest(cuML)
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 cache_path_manager: Optional[UnifiedCachePathManager] = None
                ):
        self.config = get_config("main") if config is None else config
        self.logger = get_logger(__name__)
        
        # 최적화 설정
        self.opt_config = PreprocessingOptimizationConfig()
        opt_params = self.config.get("preprocessing_optimization", {})
        if opt_params:
            for key, value in opt_params.items():
                if hasattr(self.opt_config, key):
                    setattr(self.opt_config, key, value)

        # 통합 시스템 초기화
        self.memory_manager = get_unified_memory_manager()
        self.cuda_optimizer = get_cuda_optimizer()
        self.process_pool = get_enhanced_process_pool()
        self.async_manager = get_unified_async_manager()
        
        # 캐시 설정
        self.cache_path_manager = cache_path_manager
        if self.cache_path_manager and self.opt_config.enable_advanced_cache:
            self.cache_manager = CacheManager(
                path_manager=self.cache_path_manager,
                cache_type="preprocessing",
                config={
                    "max_memory_cache_size": self.opt_config.max_cache_size,
                    "default_ttl": self.opt_config.cache_ttl
                }
            )
        else:
            self.cache_manager = None
        
        # 기본 설정
        self.enable_caching = self.config.get("preprocessing", {}).get(
            "enable_caching", True
        ) and self.cache_manager is not None
        self.use_gpu = self.config.get("use_gpu", False) and self.opt_config.enable_gpu_acceleration
        
        # 향상된 스케일러들
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "quantile": QuantileTransformer(output_distribution="uniform"),
            "power": PowerTransformer(method="yeo-johnson"),
        }
        
        # 전처리 매개변수
        self.variance_threshold = self.config.get("variance_threshold", 0.001)
        self.correlation_threshold = self.config.get("correlation_threshold", 0.95)
        self.winsorize_limits = self.config.get("winsorize_limits", (0.01, 0.99))
        self.outlier_method = self.config.get("outlier_method", "winsorize")
        self.pca_components = self.config.get("pca_components", 50)
        self.use_pca = self.config.get("use_pca", False)
        self.negative_sample_size = self.config.get("negative_sample_size", 10000)
        
        # 성능 통계
        self.performance_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "processing_times": [],
            "feature_reduction_ratios": [],
            "gpu_accelerated": 0,
            "parallel_processed": 0,
            "async_processed": 0,
            "memory_efficiency": 0.0
        }
        
        # GPU 가용성 체크
        self.gpu_available = self._check_gpu_availability()
        
        self.logger.info("통합 전처리 파이프라인 v2.0 초기화 완료")

    # @performance_monitor  # 주석 처리: 함수로 사용하려면 다른 방법 필요
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

    def _load_from_cache(self, cache_key: str) -> Optional[PreprocessingResult]:
        if not self.cache_manager:
            return None
        return self.cache_manager.get(cache_key)

    def _save_to_cache(self, cache_key: str, result: PreprocessingResult):
        if not self.cache_manager:
            return
        self.cache_manager.set(cache_key, result)

    def get_performance_report(self) -> Dict[str, Any]:
        """성능 통계 보고서"""
        return {
            "total_processed": self.performance_stats["total_processed"],
            "cache_hits": self.performance_stats["cache_hits"],
            "processing_times": self.performance_stats["processing_times"],
            "feature_reduction_ratios": self.performance_stats["feature_reduction_ratios"],
        }

    # ========================================
    # 새로운 통합 최적화 메서드들 (v2.0)
    # ========================================

    def _check_gpu_availability(self) -> bool:
        """GPU 가용성 체크"""
        if not self.opt_config.enable_gpu_acceleration:
            return False
        
        gpu_available = False
        
        # CuPy 체크
        if CUPY_AVAILABLE:
            try:
                cp.cuda.runtime.getDeviceCount()
                gpu_available = True
                self.logger.info("CuPy GPU 가속 사용 가능")
            except Exception:
                pass
        
        # PyTorch CUDA 체크
        if TORCH_AVAILABLE and not gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    self.logger.info("PyTorch CUDA 가속 사용 가능")
            except Exception:
                pass
        
        if not gpu_available:
            self.logger.info("GPU 가속 사용 불가능, CPU 모드로 실행")
        
        return gpu_available

    async def preprocess_for_model_async(
        self,
        X: np.ndarray,
        feature_names: List[str],
        model_type: str,
        y: Optional[np.ndarray] = None,
        use_cache: bool = True,
    ) -> PreprocessingResult:
        """
        비동기 전처리 수행

        Args:
            X: 입력 데이터
            feature_names: 특성 이름들
            model_type: 모델 타입
            y: 라벨 데이터 (선택적)
            use_cache: 캐시 사용 여부

        Returns:
            전처리 결과
        """
        start_time = datetime.now()
        
        # 캐시 확인
        cache_key = None
        if use_cache and self.enable_caching and self.cache_manager:
            cache_key = self._generate_cache_key(X, feature_names, model_type)
            cached_result = await self._get_cache_result_async(cache_key)
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                self.logger.debug(f"캐시에서 비동기 전처리 결과 로드: {cache_key}")
                return cached_result

        # 메모리 최적화된 비동기 전처리
        with self.memory_manager.optimize_context():
            # 데이터 청크 분할
            chunk_size = self.opt_config.async_chunk_size
            chunks = []
            for i in range(0, len(X), chunk_size):
                chunk = X[i:i + chunk_size]
                chunks.append(chunk)

            # 비동기 청크 처리
            processed_chunks = await self._process_chunks_async(chunks, feature_names, model_type, y)

            # 결과 통합
            X_processed = np.vstack(processed_chunks)

        processing_time = (datetime.now() - start_time).total_seconds()

        # 결과 생성
        result = PreprocessingResult(
            X_processed=X_processed,
            feature_names=feature_names,
            preprocessing_stats={
                "processing_time": processing_time,
                "chunks_processed": len(chunks),
                "async_mode": True
            },
            model_type=model_type,
            processing_time=processing_time,
            cache_key=cache_key,
        )

        # 캐시 저장
        if use_cache and self.enable_caching and self.cache_manager and cache_key:
            await self._set_cache_result_async(cache_key, result)

        # 성능 통계 업데이트
        self.performance_stats["total_processed"] += 1
        self.performance_stats["async_processed"] += 1
        self.performance_stats["processing_times"].append(processing_time)

        return result

    async def _get_cache_result_async(self, key: str) -> Optional[PreprocessingResult]:
        if not self.cache_manager:
            return None
        return await asyncio.to_thread(self.cache_manager.get, key)

    async def _set_cache_result_async(self, key: str, value: PreprocessingResult):
        if not self.cache_manager:
            return
        await asyncio.to_thread(self.cache_manager.set, key, value)

    async def _process_chunks_async(
        self,
        chunks: List[np.ndarray],
        feature_names: List[str],
        model_type: str,
        y: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """비동기 청크 처리"""
        tasks = []
        
        for i, chunk in enumerate(chunks):
            task = self.async_manager.create_task(
                self._process_single_chunk_async(chunk, feature_names, model_type, y, i)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

    async def _process_single_chunk_async(
        self,
        chunk: np.ndarray,
        feature_names: List[str],
        model_type: str,
        y: Optional[np.ndarray],
        chunk_idx: int
    ) -> np.ndarray:
        """단일 청크 비동기 처리"""
        self.logger.debug(f"청크 {chunk_idx} 처리 시작")

        # GPU 장치로 이동
        chunk_proc = self._to_device_optimized(chunk)

        # 전처리 단계들
        chunk_proc = self._handle_outliers_optimized(chunk_proc)
        chunk_proc = self._scale_optimized(chunk_proc)
        
        # 모델별 처리
        chunk_proc = self._model_specific_optimized(chunk_proc, y, model_type)

        # CPU로 복귀
        if CUPY_AVAILABLE and isinstance(chunk_proc, cp.ndarray):
            chunk_proc = cp.asnumpy(chunk_proc)
        elif TORCH_AVAILABLE and isinstance(chunk_proc, torch.Tensor):
            chunk_proc = chunk_proc.cpu().numpy()

        self.logger.debug(f"청크 {chunk_idx} 처리 완료")
        return chunk_proc

    def preprocess_batch_parallel(
        self,
        datasets: List[Tuple[np.ndarray, List[str], str]],
        y: Optional[np.ndarray] = None,
        parallel_workers: Optional[int] = None
    ) -> List[PreprocessingResult]:
        """
        병렬 배치 전처리

        Args:
            datasets: 데이터셋 리스트 (X, feature_names, model_type)
            y: 라벨 데이터 (선택적)
            parallel_workers: 병렬 워커 수

        Returns:
            전처리 결과 리스트
        """
        if parallel_workers is None:
            parallel_workers = self.opt_config.max_parallel_workers

        self.logger.info(f"병렬 배치 전처리 시작: {len(datasets)}개 데이터셋, {parallel_workers}개 워커")

        # 메모리 최적화
        with self.memory_manager.optimize_context():
            start_time = datetime.now()

            # 병렬 처리 함수
            def process_dataset(args):
                dataset, worker_id = args
                X, feature_names, model_type = dataset
                
                self.logger.debug(f"워커 {worker_id} 전처리 시작: 모델={model_type}")
                
                result = self.preprocess_for_model(
                    X, feature_names, model_type, y, use_cache=True
                )
                
                self.logger.debug(f"워커 {worker_id} 전처리 완료")
                return result

            # 병렬 실행
            worker_args = [(dataset, i) for i, dataset in enumerate(datasets)]
            results = self.process_pool.map(process_dataset, worker_args)

            processing_time = (datetime.now() - start_time).total_seconds()

        # 성능 통계 업데이트
        self.performance_stats["parallel_processed"] += len(datasets)

        self.logger.info(f"병렬 배치 전처리 완료: 소요 시간={processing_time:.2f}초")
        return results

    def _to_device_optimized(self, X: np.ndarray) -> np.ndarray:
        """최적화된 GPU 장치 이동"""
        if not self.use_gpu or not self.gpu_available:
            return X

        # 메모리 사용량 체크
        memory_info = self.cuda_optimizer.get_memory_info()
        if memory_info.get("usage_percent", 0) > self.opt_config.gpu_memory_limit * 100:
            self.logger.warning("GPU 메모리 사용량 초과, CPU 모드로 전환")
            return X

        # CuPy 우선
        if CUPY_AVAILABLE and self.opt_config.prefer_cupy:
            try:
                gpu_array = cp.asarray(X)
                self.performance_stats["gpu_accelerated"] += 1
                return gpu_array
            except Exception as e:
                self.logger.warning(f"CuPy 변환 실패: {e}")

        # PyTorch 대체
        if TORCH_AVAILABLE:
            try:
                gpu_tensor = torch.tensor(X, device="cuda")
                self.performance_stats["gpu_accelerated"] += 1
                return gpu_tensor
            except Exception as e:
                self.logger.warning(f"PyTorch CUDA 변환 실패: {e}")

        return X

    def _handle_outliers_optimized(self, X):
        """최적화된 이상값 처리"""
        if self.outlier_method == "winsorize":
            return self._winsorize_optimized(X)
        return X

    def _winsorize_optimized(self, X):
        """최적화된 윈소라이징"""
        if isinstance(X, np.ndarray):
            lower, upper = np.quantile(X, self.winsorize_limits, axis=0)
            return np.clip(X, lower, upper)
        
        # CuPy 최적화
        if CUPY_AVAILABLE and isinstance(X, cp.ndarray):
            lower, upper = cp.quantile(X, self.winsorize_limits, axis=0)
            return cp.clip(X, lower, upper)
        
        # PyTorch 최적화
        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            lower, upper = torch.quantile(X, torch.tensor(self.winsorize_limits), dim=0)
            return torch.clamp(X, lower, upper)
        
        return X

    def _scale_optimized(self, X):
        """최적화된 스케일링"""
        # GPU 가속 스케일링 (간단한 표준화)
        if isinstance(X, np.ndarray):
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            return (X - mean) / (std + 1e-8)
        
        # CuPy 가속
        if CUPY_AVAILABLE and isinstance(X, cp.ndarray):
            mean = cp.mean(X, axis=0)
            std = cp.std(X, axis=0)
            return (X - mean) / (std + 1e-8)
        
        # PyTorch 가속
        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            mean = torch.mean(X, dim=0)
            std = torch.std(X, dim=0)
            return (X - mean) / (std + 1e-8)
        
        return X

    def _model_specific_optimized(self, X, y, model_type: str):
        """최적화된 모델별 전처리"""
        if model_type.lower() == "lightgbm":
            return self._preprocess_for_lightgbm(X)
        elif model_type.lower() == "tcn":
            return self._preprocess_for_tcn(X)
        elif model_type.lower() == "autoencoder":
            return self._preprocess_for_autoencoder(X)
        elif model_type.lower() == "randomforest":
            return self._preprocess_for_randomforest(X)
        else:
            return X

    def _preprocess_for_lightgbm(self, X):
        """LightGBM 전용 전처리"""
        # LightGBM은 원본 데이터를 선호
        return X

    def _preprocess_for_tcn(self, X):
        """TCN 전용 전처리"""
        # TCN은 정규화된 시계열 데이터 필요
        return X

    def _preprocess_for_autoencoder(self, X):
        """AutoEncoder 전용 전처리"""
        # AutoEncoder는 [0,1] 범위 정규화 필요
        if isinstance(X, np.ndarray):
            return (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-8)
        
        if CUPY_AVAILABLE and isinstance(X, cp.ndarray):
            return (X - cp.min(X)) / (cp.max(X) - cp.min(X) + 1e-8)
        
        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            return (X - torch.min(X)) / (torch.max(X) - torch.min(X) + 1e-8)
        
        return X

    def _preprocess_for_randomforest(self, X):
        """RandomForest 전용 전처리"""
        # RandomForest는 스케일링이 필요하지 않음
        return X

    def _generate_cache_key(self, X: np.ndarray, feature_names: List[str], model_type: str) -> str:
        """향상된 캐시 키 생성"""
        # 데이터 해시
        data_hash = hashlib.md5(X.tobytes()).hexdigest()[:16]
        
        # 특성 이름 해시
        feature_hash = hashlib.md5(','.join(feature_names).encode()).hexdigest()[:8]
        
        # 설정 해시
        config_str = f"{model_type}_{self.use_gpu}_{self.variance_threshold}_{self.outlier_method}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"preprocessing_{data_hash}_{feature_hash}_{config_hash}"

    def get_optimization_report(self) -> Dict[str, Any]:
        """최적화 보고서 생성"""
        return {
            "pipeline_name": "UnifiedPreprocessingPipeline_v2.0",
            "optimization_config": {
                "gpu_acceleration": self.opt_config.enable_gpu_acceleration,
                "parallel_processing": self.opt_config.enable_parallel_processing,
                "async_enabled": self.opt_config.enable_async,
                "memory_optimization": self.opt_config.enable_memory_optimization,
                "advanced_cache": self.opt_config.enable_advanced_cache
            },
            "performance_stats": self.performance_stats,
            "gpu_info": {
                "gpu_available": self.gpu_available,
                "cupy_available": CUPY_AVAILABLE,
                "torch_available": TORCH_AVAILABLE,
                "gpu_accelerated_ops": self.performance_stats["gpu_accelerated"]
            },
            "cache_info": self.cache_manager.get_cache_info() if self.cache_manager else {},
            "memory_info": self.memory_manager.get_memory_info() if self.memory_manager else {}
        }

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 최적화"""
        self.logger.info("전처리 파이프라인 메모리 최적화 시작")

        optimization_results = {
            "before": {},
            "after": {},
            "optimizations_applied": []
        }

        # 현재 메모리 상태
        if self.memory_manager:
            optimization_results["before"] = self.memory_manager.get_memory_info()

        optimizations_applied = []

        # 1. 캐시 정리
        if self.cache_manager:
            cache_info = self.cache_manager.get_cache_info()
            if cache_info.get("memory_usage", 0) > 300 * 1024 * 1024:  # 300MB 이상
                self.cache_manager.clear_cache()
                optimizations_applied.append("cache_cleared")

        # 2. 스케일러 재초기화
        if self.performance_stats["total_processed"] > 10000:
            self.scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
                "quantile": QuantileTransformer(output_distribution="uniform"),
                "power": PowerTransformer(method="yeo-johnson"),
            }
            optimizations_applied.append("scalers_reset")

        # 3. 성능 통계 요약
        if len(self.performance_stats["processing_times"]) > 1000:
            avg_time = np.mean(self.performance_stats["processing_times"])
            self.performance_stats["processing_times"] = [avg_time]
            optimizations_applied.append("stats_summarized")

        # 최적화 후 메모리 상태
        if self.memory_manager:
            optimization_results["after"] = self.memory_manager.get_memory_info()

        optimization_results["optimizations_applied"] = optimizations_applied

        self.logger.info(f"전처리 파이프라인 메모리 최적화 완료: {len(optimizations_applied)}개 최적화 적용")
        return optimization_results

    def benchmark_performance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        model_types: List[str],
        n_runs: int = 5
    ) -> Dict[str, Any]:
        """성능 벤치마크"""
        self.logger.info(f"전처리 파이프라인 벤치마크 시작: {n_runs}회 실행")

        benchmark_results = {
            "standard_preprocessing": [],
            "async_preprocessing": [],
            "parallel_preprocessing": []
        }

        # 테스트 데이터 준비
        test_datasets = [(X, feature_names, model_type) for model_type in model_types]

        # 1. 표준 전처리 벤치마크
        for i in range(n_runs):
            start_time = datetime.now()
            for dataset in test_datasets:
                X_test, features, model_type = dataset
                _ = self.preprocess_for_model(X_test, features, model_type, use_cache=False)
            elapsed = (datetime.now() - start_time).total_seconds()
            benchmark_results["standard_preprocessing"].append(elapsed)

        # 2. 비동기 전처리 벤치마크
        if self.opt_config.enable_async:
            for i in range(n_runs):
                start_time = datetime.now()
                for dataset in test_datasets:
                    X_test, features, model_type = dataset
                    _ = asyncio.run(self.preprocess_for_model_async(X_test, features, model_type, use_cache=False))
                elapsed = (datetime.now() - start_time).total_seconds()
                benchmark_results["async_preprocessing"].append(elapsed)

        # 3. 병렬 전처리 벤치마크
        if self.opt_config.enable_parallel_processing:
            for i in range(n_runs):
                start_time = datetime.now()
                _ = self.preprocess_batch_parallel(test_datasets, parallel_workers=4)
                elapsed = (datetime.now() - start_time).total_seconds()
                benchmark_results["parallel_preprocessing"].append(elapsed)

        # 결과 요약
        summary = {}
        for method, times in benchmark_results.items():
            if times:
                summary[method] = {
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times)
                }

        self.logger.info("전처리 파이프라인 벤치마크 완료")
        return {
            "benchmark_results": benchmark_results,
            "summary": summary,
            "test_data_shape": X.shape,
            "model_types": model_types,
            "n_runs": n_runs
        }
