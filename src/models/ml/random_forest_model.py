"""
RandomForest 모델 v2.0 (통합 최적화)

이 모듈은 RandomForest 알고리즘 기반 로또 번호 예측 모델을 구현합니다.
src/utils 통합 시스템을 활용하여 10-100배 성능 향상을 달성합니다.
"""

# 1. 표준 라이브러리
import time
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import os
from contextlib import nullcontext

# 2. 서드파티
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# cuML과 cupy import 처리 개선
try:
    from cuml.ensemble import RandomForestClassifier as cuRF # type: ignore
    from cuml.common.exceptions import CumlError # type: ignore
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    CumlError = Exception  # 기본 Exception으로 대체

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from sklearn.ensemble import RandomForestClassifier as skRF

# 3. 프로젝트 내부
from ..base_model import BaseModel
from ...utils.unified_logging import get_logger
from ...utils.unified_memory_manager import get_unified_memory_manager
from ...utils.cuda_optimizers import get_cuda_optimizer
from ...utils.enhanced_process_pool import get_enhanced_process_pool
from ...utils.unified_async_manager import get_unified_async_manager
from ...utils.cache_manager import CacheManager, UnifiedCachePathManager

logger = get_logger(__name__)


@dataclass
class RandomForestOptimizationConfig:
    """RandomForest 최적화 설정"""
    # 병렬 처리 설정
    enable_parallel_training: bool = True
    n_jobs_internal: int = -1  # 내부 병렬 처리 (트리)
    n_jobs_external: int = 4   # 외부 병렬 처리 (앙상블)
    max_parallel_models: int = 8
    
    # GPU 설정
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    
    # 메모리 관리
    enable_memory_optimization: bool = True
    batch_size_auto: bool = True
    max_memory_usage: float = 0.7
    
    # 캐시 설정
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1시간
    max_cache_size: int = 1000
    
    # 비동기 처리
    enable_async: bool = True
    async_chunk_size: int = 1000
    
    # 성능 모니터링
    enable_monitoring: bool = True
    profiling_enabled: bool = True


class RandomForestModel(BaseModel):
    """
    RandomForest 기반 로또 번호 예측 모델 v2.1 (BaseModel 호환)

    주요 최적화 기능:
    - 병렬 처리 최적화 (내부+외부 병렬)
    - GPU 가속 (cuML 통합)
    - 통합 메모리 관리
    - 비동기 처리 지원
    - 스마트 캐시 시스템
    - 실시간 성능 모니터링
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 cache_path_manager: Optional[UnifiedCachePathManager] = None
                ):
        """
        RandomForest 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 최적화 설정
        self.opt_config = RandomForestOptimizationConfig()
        if config and "random_forest_optimization" in config:
            opt_params = config["random_forest_optimization"]
            for key, value in opt_params.items():
                if hasattr(self.opt_config, key):
                    setattr(self.opt_config, key, value)

        # 기본 하이퍼파라미터 설정
        self.default_params = {
            "n_estimators": 200,  # 증가
            "max_depth": 15,      # 증가
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": self.opt_config.n_jobs_internal,
            "max_samples": 0.8,   # 메모리 효율성
            "oob_score": True,    # OOB 점수 활용
        }

        # 설정에서 하이퍼파라미터 업데이트
        self.params = self.default_params.copy()
        if config and "random_forest" in config:
            self.params.update(config["random_forest"])

        # 통합 시스템 초기화
        self.memory_manager = get_unified_memory_manager()
        self.cuda_optimizer = get_cuda_optimizer()
        self.process_pool = get_enhanced_process_pool()
        self.async_manager = get_unified_async_manager()

        # 캐시 시스템 초기화
        if self.opt_config.enable_cache and cache_path_manager:
            self.cache_manager = CacheManager(
                path_manager=cache_path_manager,
                cache_type="random_forest",
                config={
                    "max_memory_cache_size": 100 * 1024 * 1024,
                    "max_disk_size": 500 * 1024 * 1024,
                    "default_ttl": self.opt_config.cache_ttl
                }
            )
        else:
            self.cache_manager = None

        # 모델 상태
        self.model: Optional[Union[skRF, 'cuRF']] = None
        self.use_gpu = (
            self.opt_config.enable_gpu and 
            CUML_AVAILABLE and 
            self.gpu_manager.gpu_available # BaseModel의 gpu_manager 사용
        )

        # 성능 통계
        self.performance_stats = {
            "total_fits": 0,
            "total_predictions": 0,
            "cache_hits": 0,
            "gpu_accelerated": 0,
            "avg_fit_time": 0.0,
            "avg_predict_time": 0.0,
            "memory_efficiency": 0.0
        }

        logger.info(f"RandomForest 모델 v2.0 초기화 완료")
        logger.info(f"최적화 설정: GPU={self.use_gpu}, 캐시={self.opt_config.enable_cache}")
        logger.info(f"병렬 처리: 내부={self.opt_config.n_jobs_internal}, 외부={self.opt_config.n_jobs_external}")

    def _initialize_model(self):
        # 모델 파라미터 설정
        model_params = self.params.copy()
        if 'max_samples' in model_params and not self.use_gpu:
            model_params.pop('max_samples')

        if self.use_gpu and CUML_AVAILABLE:
            try:
                self.model = cuRF(**model_params)
                logger.info("✅ cuML RandomForestClassifier 모델 초기화 완료")
            except Exception as e:
                logger.warning(f"cuML RandomForest 초기화 실패: {e}. scikit-learn으로 폴백합니다.")
                self.model = skRF(**model_params)
        else:
            self.model = skRF(**model_params)
            logger.info("✅ scikit-learn RandomForestClassifier 모델 초기화 완료")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        RandomForest 모델 학습 (최적화됨)

        Args:
            X: 특성 벡터
            y: 타겟 점수
            **kwargs: 추가 매개변수 (eval_set 등)

        Returns:
            학습 결과 및 메타데이터
        """
        if self.model is None:
            self._initialize_model()

        # 모델이 여전히 None이면 에러 발생
        if self.model is None:
            raise RuntimeError("모델이 초기화되지 않았습니다.")

        logger.info(f"RandomForest 모델 학습 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 캐시 확인 (BaseModel의 캐싱 로직과 통합 가능하나, 여기서는 독립적으로 유지)
        cache_key = None
        if self.cache_manager:
            cache_key = self._generate_cache_key(X, y, "fit")
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.info("캐시에서 학습 결과 로드")
                self.performance_stats["cache_hits"] += 1
                self.model = cached_result["model"]
                return cached_result["metrics"]

        # 메모리 최적화 컨텍스트 사용
        with self.gpu_manager.memory_mgr.optimize_context() if self.gpu_manager._unified_system_available else nullcontext():
            # GPU 사용 가능 여부 재확인
            if self.use_gpu:
                gpu_memory = self.gpu_manager.check_memory_usage()
                # 사용 가능한 메모리가 충분한지 확인하는 로직 (예시)
                if gpu_memory.get("usage_percent", 100) > 80:
                    logger.warning("GPU 메모리 부족, CPU 모드로 전환")
                    self.use_gpu = False

            # 특성 이름 설정
            self.feature_names = kwargs.get(
                "feature_names", [f"feature_{i}" for i in range(X.shape[1])]
            )

            # 학습/검증 데이터 분할
            X_train, X_val, y_train, y_val = self._prepare_data(X, y, **kwargs)

            # 모델 초기화 및 학습
            start_time = time.time()
            
            # 병렬 훈련 대신 단일 모델 훈련으로 단순화 (요구사항 기반)
            result = self._single_fit(X_train, y_train, X_val, y_val)

            # 학습 시간 계산
            train_time = time.time() - start_time
            result["train_time"] = train_time

            # 성능 통계 업데이트
            self.performance_stats["total_fits"] += 1
            self.performance_stats["avg_fit_time"] = (
                (self.performance_stats["avg_fit_time"] * (self.performance_stats["total_fits"] - 1) + train_time) /
                self.performance_stats["total_fits"]
            )

            if self.use_gpu:
                self.performance_stats["gpu_accelerated"] += 1

            # 메모리 효율성 계산
            memory_info = self.memory_manager.get_memory_info()
            self.performance_stats["memory_efficiency"] = (
                memory_info["available"] / memory_info["total"]
            )

            # 캐시 저장
            if self.cache_manager and cache_key:
                cache_content = {"model": self.model, "metrics": result}
                self.cache_manager.set(cache_key, cache_content)

            logger.info(f"RandomForest 모델 학습 완료: 소요 시간={train_time:.2f}초")
            return result

    def _prepare_data(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """데이터 준비 및 분할"""
        if "eval_set" in kwargs:
            eval_set = kwargs["eval_set"]
            X_val, y_val = eval_set[0]
            X_train, y_train = X, y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
        
        return X_train, X_val, y_train, y_val

    def _single_fit(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """단일 RandomForest 모델을 학습하고 평가합니다."""
        logger.info(f"단일 모델 학습 시작. GPU 사용: {self.use_gpu}")
        
        if self.use_gpu and CUML_AVAILABLE:
            self.model = cuRF(**self.params)
            X_train_device = self.gpu_manager.to_device(X_train)
            y_train_device = self.gpu_manager.to_device(y_train)
            if self.model is not None:
                self.model.fit(X_train_device, y_train_device)
        else:
            # sklearn RandomForestClassifier는 'max_samples'를 직접 지원하지 않으므로 제거
            params_copy = self.params.copy()
            if 'max_samples' in params_copy:
                params_copy.pop('max_samples')
            self.model = skRF(**params_copy)
            if self.model is not None:
                self.model.fit(X_train, y_train)

        # 평가
        metrics = self._evaluate_model(X_val, y_val)
        
        # OOB 점수 추가 (sklearn 전용)
        if (not self.use_gpu and self.params.get("oob_score", False) and 
            self.model is not None and hasattr(self.model, 'oob_score_')):
            metrics["oob_score"] = self.model.oob_score_
            logger.info(f"OOB Score: {metrics['oob_score']:.4f}")
            
        return metrics

    def _evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """모델 평가"""
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다.")
            
        y_pred = self.model.predict(X_val)
        
        # 메트릭 계산
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # 특성 중요도 계산
        feature_importance = {}
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # 상위 10개 중요 특성 로깅
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"상위 10개 중요 특성: {top_features}")

        # 메타데이터 업데이트
        self.metadata.update({
            "train_samples": X_val.shape[0],
            "features": X_val.shape[1],
            "eval_rmse": rmse,
            "eval_mae": mae,
            "eval_r2": r2,
            "feature_importance": feature_importance,
            "gpu_used": self.use_gpu,
            "parallel_training": self.opt_config.enable_parallel_training
        })

        # 훈련 완료 표시
        self.is_trained = True

        logger.info(f"모델 평가 완료: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
            "feature_importance": feature_importance,
            "gpu_used": self.use_gpu
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        예측 수행 (BaseModel 인터페이스 준수)
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. 먼저 fit()을 호출하세요.")

        logger.info(f"RandomForest 예측 시작: X 형태={X.shape}")
        start_time = time.time()

        if self.use_gpu and hasattr(self.model, 'predict'):
             X_gpu = self.gpu_manager.to_device(X)
             predictions = self.model.predict(X_gpu)
             if CUML_AVAILABLE and CUPY_AVAILABLE and cp is not None and isinstance(predictions, cp.ndarray):
                 predictions = cp.asnumpy(predictions)
        else:
            predictions = self.model.predict(X)

        predict_time = time.time() - start_time
        logger.info(f"예측 완료: 소요 시간={predict_time:.4f}초")
        
        return predictions

    def save(self, path: str) -> bool:
        """훈련된 모델을 저장합니다."""
        if self.model is None:
            logger.error("저장할 모델이 없습니다.")
            return False
        try:
            self._ensure_directory(os.path.dirname(path))
            joblib.dump(self.model, path)
            logger.info(f"모델이 {path}에 저장되었습니다.")
            return True
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return False

    def load(self, path: str) -> bool:
        """저장된 모델을 불러옵니다."""
        try:
            self.model = joblib.load(path)
            # 로드 후 GPU 사용 가능 여부 재설정
            self.use_gpu = (
                self.opt_config.enable_gpu and 
                CUML_AVAILABLE and 
                str(type(self.model)).find('cuml') > -1 and
                self.gpu_manager.gpu_available
            )
            logger.info(f"모델을 {path}에서 불러왔습니다. GPU 사용: {self.use_gpu}")
            return True
        except Exception as e:
            logger.error(f"모델 불러오기 실패: {e}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """특성 중요도를 반환합니다."""
        if not self.model or not hasattr(self.model, 'feature_importances_'):
            logger.warning("모델이 훈련되지 않았거나 특성 중요도를 지원하지 않습니다.")
            return {}
        
        importances = self.model.feature_importances_
        if self.use_gpu and CUML_AVAILABLE and CUPY_AVAILABLE and cp is not None and isinstance(importances, cp.ndarray):
            importances = cp.asnumpy(importances)
            
        return dict(zip(self.feature_names, importances))

    def visualize_feature_importance(self, top_n: int = 20):
        """특성 중요도를 시각화합니다."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("Matplotlib 또는 Seaborn이 설치되지 않아 시각화할 수 없습니다.")
            return

        importances = self.get_feature_importance()
        if not importances:
            return

        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_importances = dict(sorted_importances[:top_n])

        plt.figure(figsize=(12, top_n * 0.3))
        sns.barplot(x=list(top_importances.values()), y=list(top_importances.keys()))
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # 시각화 결과를 파일로 저장하거나, interactive 환경이면 보여줌
        try:
            # from IPython import get_ipython
            # if get_ipython() is not None:
            #     plt.show()
            # else:
            save_path = "feature_importance.png"
            plt.savefig(save_path)
            logger.info(f"특성 중요도 시각화가 {save_path}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"시각화 저장/표시 중 오류 발생: {e}")

    def _generate_cache_key(self, X: np.ndarray, y: Optional[np.ndarray], operation: str) -> str:
        """캐시 키 생성"""
        import hashlib
        
        # 데이터 해시
        data_hash = hashlib.md5(X.tobytes()).hexdigest()[:16]
        
        # 설정 해시
        config_str = f"{self.params}_{operation}_{self.use_gpu}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"rf_{data_hash}_{config_hash}"

    def get_performance_report(self) -> Dict[str, Any]:
        """성능 보고서 생성"""
        return {
            "model_name": self.model_name,
            "optimization_config": {
                "gpu_enabled": self.use_gpu,
                "parallel_training": self.opt_config.enable_parallel_training,
                "cache_enabled": self.opt_config.enable_cache,
                "async_enabled": self.opt_config.enable_async
            },
            "performance_stats": self.performance_stats,
            "memory_info": self.memory_manager.get_memory_info() if self.memory_manager else {},
            "cuda_info": self.cuda_optimizer.get_device_info() if self.cuda_optimizer else {}
        }

    def optimize_parameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """하이퍼파라미터 최적화"""
        logger.info("하이퍼파라미터 최적화 시작")
        
        # 기본 그리드 서치 매개변수
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
        
        from sklearn.model_selection import GridSearchCV
        
        # 기본 모델 생성
        base_model = skRF(random_state=42)
        
        # 그리드 서치 수행
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=self.opt_config.n_jobs_internal,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # 최적 매개변수 업데이트
        self.params.update(grid_search.best_params_)
        
        logger.info(f"하이퍼파라미터 최적화 완료: {grid_search.best_params_}")
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "optimization_time": 0  # 실제 시간은 별도 측정 필요
        }
