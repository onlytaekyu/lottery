"""
LightGBM 모델

이 모듈은 LightGBM 알고리즘 기반 로또 번호 예측 모델을 구현합니다.
feature_vector_full.npy 또는 필터링된 벡터를 입력으로 사용하여 점수를 예측합니다.

✅ v2.0 업데이트: src/utils 통합 시스템 적용
- 병렬 처리 최적화 (get_enhanced_process_pool)
- 비동기 처리 지원 (get_unified_async_manager)
- GPU 최적화 강화 (get_cuda_optimizer)
- 통합 메모리 관리 (get_unified_memory_manager)
- 스마트 캐시 시스템
"""

import os
import json
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Any, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
from sklearn.metrics import log_loss
try:
    from scipy.sparse import spmatrix
    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    spmatrix = None
    SCIPY_SPARSE_AVAILABLE = False

# ✅ src/utils 통합 시스템 활용
from ...utils.unified_logging import get_logger
from ...utils import (
    get_unified_memory_manager,
    get_cuda_optimizer,
    get_enhanced_process_pool,
    get_unified_async_manager,
    get_pattern_filter
)

# 3. 프로젝트 내부
from ..base_model import BaseModel

logger = get_logger(__name__)


@dataclass
class LightGBMOptimizationConfig:
    """LightGBM 최적화 설정"""
    
    # 병렬 처리 설정
    enable_parallel_training: bool = True
    parallel_workers: int = 4
    lgb_n_jobs: int = -1  # LightGBM 내부 병렬 처리
    
    # 비동기 처리 설정
    enable_async_processing: bool = True
    async_batch_size: int = 1000
    
    # GPU 최적화 설정
    auto_gpu_optimization: bool = True
    gpu_memory_fraction: float = 0.8
    use_gpu_tree_learner: bool = True
    
    # 캐시 설정
    enable_smart_caching: bool = True
    cache_feature_vectors: bool = True
    cache_predictions: bool = True
    cache_ttl: int = 3600  # 1시간
    
    # 메모리 최적화
    auto_memory_management: bool = True
    memory_efficient_training: bool = True


class LightGBMModel(BaseModel):
    """
    🚀 LightGBM 기반 로또 번호 예측 모델 (v2.0)

    src/utils 통합 시스템 기반 고성능 그래디언트 부스팅:
    - 병렬 훈련 및 예측
    - 비동기 데이터 처리
    - GPU 최적화 강화
    - 스마트 메모리 관리
    - 지능형 캐시 시스템
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        LightGBM 모델 초기화 (통합 시스템 적용)

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 이름
        self.model_name = "LightGBMModel"

        # ✅ 최적화 설정 초기화
        opt_config = config.get("optimization", {}) if config else {}
        self.opt_config = LightGBMOptimizationConfig(
            enable_parallel_training=opt_config.get("enable_parallel_training", True),
            parallel_workers=opt_config.get("parallel_workers", 4),
            lgb_n_jobs=opt_config.get("lgb_n_jobs", -1),
            enable_async_processing=opt_config.get("enable_async_processing", True),
            async_batch_size=opt_config.get("async_batch_size", 1000),
            auto_gpu_optimization=opt_config.get("auto_gpu_optimization", True),
            gpu_memory_fraction=opt_config.get("gpu_memory_fraction", 0.8),
            use_gpu_tree_learner=opt_config.get("use_gpu_tree_learner", True),
            enable_smart_caching=opt_config.get("enable_smart_caching", True),
            cache_feature_vectors=opt_config.get("cache_feature_vectors", True),
            cache_predictions=opt_config.get("cache_predictions", True),
            cache_ttl=opt_config.get("cache_ttl", 3600),
            auto_memory_management=opt_config.get("auto_memory_management", True),
            memory_efficient_training=opt_config.get("memory_efficient_training", True)
        )

        # ✅ src/utils 통합 시스템 초기화
        try:
            self.memory_mgr = get_unified_memory_manager()
            self.cuda_opt = get_cuda_optimizer()
            self.process_pool = get_enhanced_process_pool()
            self.async_mgr = get_unified_async_manager()
            self.pattern_filter = get_pattern_filter()
            self._unified_system_available = True
            logger.info("✅ LightGBM 통합 시스템 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ 통합 시스템 초기화 실패, 기본 모드로 폴백: {e}")
            self._unified_system_available = False
            self._init_fallback_systems()

        # ✅ 스마트 캐시 시스템
        if self.opt_config.enable_smart_caching and self._unified_system_available:
            self.smart_cache = True
            self.feature_cache = {}  # 특성 벡터 캐시
            self.prediction_cache = {}  # 예측 결과 캐시
            self.model_cache = {}  # 모델 캐시
        else:
            self.smart_cache = False
            self.feature_cache = {}
            self.prediction_cache = {}

        # 기본 파라미터 설정 (병렬 처리 최적화)
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
            # ✅ 병렬 처리 최적화
            "n_jobs": self.opt_config.lgb_n_jobs,
            "num_threads": self.opt_config.parallel_workers if self.opt_config.enable_parallel_training else 1,
        }

        # 3자리 모드 전용 파라미터 (병렬 처리 최적화)
        self.three_digit_params = {
            "objective": "multiclass",
            "num_class": 14190,  # C(45,3) = 45*44*43/(3*2*1)
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 3,
            "verbose": -1,
            "random_state": 42,
            # ✅ 병렬 처리 최적화
            "n_jobs": self.opt_config.lgb_n_jobs,
            "num_threads": self.opt_config.parallel_workers if self.opt_config.enable_parallel_training else 1,
        }

        # 설정에서 파라미터 업데이트
        if config and "lightgbm" in config:
            self.params.update(config["lightgbm"])

        # 3자리 모드 파라미터 업데이트
        if config and "lightgbm_3digit" in config:
            self.three_digit_params.update(config["lightgbm_3digit"])

        # ✅ GPU 설정 최적화 (통합 시스템)
        self._setup_advanced_gpu_acceleration(config)

        # 특성 이름 저장
        self.feature_names = None

        logger.info(f"✅ LightGBM 모델 초기화 완료 (v2.0)")
        logger.info(f"🚀 최적화 활성화: 병렬={self.opt_config.enable_parallel_training}, "
                   f"비동기={self.opt_config.enable_async_processing}, "
                   f"스마트캐시={self.opt_config.enable_smart_caching}")
        logger.info(f"GPU 사용: {self.params.get('device_type', 'cpu')}")
        logger.info(f"장치: {self.device} (GPU 사용 가능: {self.device_manager.gpu_available})")

    def _init_fallback_systems(self):
        """폴백 시스템 초기화"""
        # 기본 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=self.opt_config.parallel_workers)
        logger.info("기본 병렬 처리 시스템으로 폴백")

    def _setup_advanced_gpu_acceleration(self, config: Optional[Dict[str, Any]] = None):
        """
        ✅ 고급 GPU 가속 설정 (통합 시스템 적용)

        Args:
            config: 모델 설정
        """
        try:
            # GPU 사용 설정 확인
            use_gpu = False
            if config:
                use_gpu = config.get("use_gpu", False)
            
            # GPU 가용성 자동 체크
            if use_gpu and self.device_manager.gpu_available:
                # ✅ CUDA 최적화기 활용 (통합 시스템)
                if self._unified_system_available and self.cuda_opt:
                    self.cuda_opt.set_tf32_enabled(True)
                    self.cuda_opt.set_memory_pool_enabled(True)
                    if self.opt_config.use_gpu_tree_learner:
                        self.cuda_opt.optimize_for_inference(True)
                    logger.info("🚀 고급 CUDA 최적화 활성화")
                
                # LightGBM GPU 지원 확인 및 설정
                try:
                    # GPU 파라미터 적용
                    gpu_params = {
                        "device_type": "gpu",
                        "gpu_platform_id": 0,
                        "gpu_device_id": 0,
                        "gpu_use_dp": True,  # double precision 사용
                        "max_bin": 63,       # GPU에서 권장되는 max_bin
                    }
                    
                    # ✅ 고급 GPU 최적화 설정
                    if self.opt_config.auto_gpu_optimization:
                        memory_info = self.device_manager.check_memory_usage()
                        if memory_info.get("gpu_available", False):
                            total_gpu_memory = memory_info.get("total_gb", 8.0)
                            
                            # GPU 메모리에 따른 동적 설정
                            if total_gpu_memory >= 16.0:  # 고급 GPU
                                gpu_params.update({
                                    "max_bin": 255,
                                    "num_leaves": min(255, self.params.get("num_leaves", 31) * 2),
                                })
                                logger.info("🔥 고성능 GPU 모드 활성화")
                            elif total_gpu_memory >= 8.0:  # 중급 GPU
                                gpu_params.update({
                                    "max_bin": 127,
                                    "num_leaves": min(127, self.params.get("num_leaves", 31)),
                                })
                                logger.info("⚡ 표준 GPU 모드 활성화")
                    
                    self.params.update(gpu_params)
                    self.three_digit_params.update(gpu_params)
                    
                    logger.info("✅ LightGBM 고급 GPU 가속 활성화")
                    
                    # ✅ GPU 메모리 풀링 설정
                    if self._unified_system_available and self.memory_mgr:
                        gpu_memory_fraction = self.opt_config.gpu_memory_fraction
                        logger.info(f"GPU 메모리 사용률 제한: {gpu_memory_fraction*100:.0f}%")
                        
                except Exception as e:
                    logger.warning(f"LightGBM GPU 설정 실패, CPU로 fallback: {e}")
                    self._fallback_to_cpu()
                    
            else:
                if use_gpu and not self.device_manager.gpu_available:
                    logger.warning("GPU 사용이 요청되었지만 GPU를 사용할 수 없습니다. CPU로 fallback")
                self._fallback_to_cpu()
                
        except Exception as e:
            logger.error(f"GPU 설정 중 오류 발생, CPU로 fallback: {e}")
            self._fallback_to_cpu()

    def _fallback_to_cpu(self):
        """CPU로 fallback 처리 (최적화된)"""
        self.params["device_type"] = "cpu"
        self.three_digit_params["device_type"] = "cpu"
        
        # ✅ CPU 최적화 설정
        if self.opt_config.enable_parallel_training:
            cpu_threads = min(self.opt_config.parallel_workers, os.cpu_count() or 4)
            self.params.update({
                "n_jobs": cpu_threads,
                "num_threads": cpu_threads,
                "force_row_wise": True,  # CPU 최적화
            })
            self.three_digit_params.update({
                "n_jobs": cpu_threads,
                "num_threads": cpu_threads,
                "force_row_wise": True,
            })
            logger.info(f"🔧 CPU 병렬 처리 최적화: {cpu_threads} 스레드")
        
        # GPU 전용 파라미터 제거
        gpu_params = ["gpu_platform_id", "gpu_device_id", "gpu_use_dp"]
        for param in gpu_params:
            self.params.pop(param, None)
            self.three_digit_params.pop(param, None)
            
        logger.info("CPU 모드로 설정 완료")

    async def _check_gpu_memory_async(self, data_size: int) -> bool:
        """
        🚀 비동기 GPU 메모리 충분성 확인

        Args:
            data_size: 데이터 크기 (대략적인 바이트 수)

        Returns:
            GPU 메모리가 충분한지 여부
        """
        if self.params.get("device_type") != "gpu":
            return True
            
        # ✅ 통합 메모리 관리자 활용
        if self._unified_system_available and self.memory_mgr:
            try:
                memory_stats = await self.async_mgr.run_in_thread(
                    self.memory_mgr.get_memory_status
                )
                gpu_util = memory_stats.get("gpu_utilization", 0.5)
                
                # 스마트 메모리 할당 가능성 체크
                (data_size * 4) / (1024**3)
                memory_limit = self.opt_config.gpu_memory_fraction
                
                if gpu_util > memory_limit:
                    logger.warning(
                        f"GPU 메모리 사용률 높음: {gpu_util*100:.1f}% > {memory_limit*100:.0f}%, "
                        f"CPU로 fallback"
                    )
                    await self.async_mgr.run_in_thread(self._fallback_to_cpu)
                    return False
                    
                return True
                
            except Exception as e:
                logger.warning(f"비동기 메모리 체크 실패: {e}")
                return self._check_gpu_memory_before_training(data_size)
        else:
            # 폴백: 기존 방식
            return self._check_gpu_memory_before_training(data_size)

    def _check_gpu_memory_before_training(self, data_size: int) -> bool:
        """
        훈련 전 GPU 메모리 충분성 확인 (기존 방식)

        Args:
            data_size: 데이터 크기 (대략적인 바이트 수)

        Returns:
            GPU 메모리가 충분한지 여부
        """
        if self.params.get("device_type") != "gpu":
            return True
            
        memory_info = self.device_manager.check_memory_usage()
        if not memory_info.get("gpu_available", False):
            return False
            
        # 필요한 메모리 추정 (데이터 크기의 3-4배)
        estimated_memory_gb = (data_size * 4) / (1024**3)
        available_memory_gb = memory_info.get("total_gb", 0) - memory_info.get("allocated_gb", 0)
        memory_limit = self.opt_config.gpu_memory_fraction
        
        if estimated_memory_gb > available_memory_gb * memory_limit:
            logger.warning(
                f"GPU 메모리 부족 예상: 필요={estimated_memory_gb:.1f}GB, "
                f"사용 가능={available_memory_gb:.1f}GB, CPU로 fallback"
            )
            self._fallback_to_cpu()
            return False
            
        return True

    def get_optimal_batch_size(self, data_size: int) -> int:
        """
        🚀 최적 배치 크기 계산 (통합 시스템 활용)
        
        Args:
            data_size: 처리할 데이터 크기
            
        Returns:
            최적 배치 크기
        """
        if self._unified_system_available and self.memory_mgr:
            try:
                memory_stats = self.memory_mgr.get_memory_status()
                
                if self.params.get("device_type") == "gpu":
                    gpu_util = memory_stats.get("gpu_utilization", 0.5)
                    if gpu_util < 0.3:
                        return min(self.opt_config.async_batch_size * 2, data_size)
                    elif gpu_util < 0.7:
                        return min(self.opt_config.async_batch_size, data_size)
                    else:
                        return min(self.opt_config.async_batch_size // 2, data_size)
                else:
                    cpu_util = memory_stats.get("cpu_utilization", 0.5)
                    if cpu_util < 0.3:
                        return min(self.opt_config.async_batch_size, data_size)
                    else:
                        return min(self.opt_config.async_batch_size // 2, data_size)
                        
            except Exception as e:
                logger.debug(f"최적 배치 크기 계산 실패: {e}")
        
        # 폴백: 기본 배치 크기
        return min(self.opt_config.async_batch_size, data_size)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        🚀 성능 통계 반환 (통합 시스템 정보 포함)
        """
        stats = {
            "model_type": self.model_name,
            "unified_system_available": self._unified_system_available,
            "optimization_config": {
                "parallel_training": self.opt_config.enable_parallel_training,
                "async_processing": self.opt_config.enable_async_processing,
                "smart_caching": self.opt_config.enable_smart_caching,
                "gpu_optimization": self.opt_config.auto_gpu_optimization,
                "parallel_workers": self.opt_config.parallel_workers,
                "lgb_n_jobs": self.opt_config.lgb_n_jobs,
            },
            "device_info": {
                "device_type": self.params.get("device_type", "cpu"),
                "gpu_available": self.device_manager.gpu_available,
                "using_gpu": self.params.get("device_type") == "gpu",
            },
            "cache_stats": {
                "smart_cache_enabled": self.smart_cache,
                "feature_cache_size": len(self.feature_cache),
                "prediction_cache_size": len(self.prediction_cache),
            },
            "model_info": {
                "is_trained": self.is_trained,
                "feature_count": len(self.feature_names) if self.feature_names else 0,
            }
        }
        
        # 통합 시스템 통계
        if self._unified_system_available:
            if self.memory_mgr:
                try:
                    stats["memory_performance"] = self.memory_mgr.get_performance_metrics()
                except Exception as e:
                    logger.debug(f"메모리 성능 통계 조회 실패: {e}")
            
            if self.cuda_opt:
                try:
                    stats["cuda_optimization"] = self.cuda_opt.get_optimization_stats()
                except Exception as e:
                    logger.debug(f"CUDA 최적화 통계 조회 실패: {e}")
        
        return stats

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        LightGBM 모델 학습 (GPU 최적화)

        Args:
            X: 특성 벡터
            y: 타겟 점수
            **kwargs: 추가 매개변수 (eval_set, early_stopping_rounds 등)

        Returns:
            학습 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"LightGBM 모델 학습 시작: X 형태={X.shape}, y 형태={y.shape}")

        # ✅ GPU 메모리 체크 (비동기 처리 지원)
        data_size = X.nbytes + y.nbytes
        if self.opt_config.enable_async_processing and self._unified_system_available:
            try:
                # 비동기 메모리 체크
                import asyncio
                memory_ok = asyncio.run(self._check_gpu_memory_async(data_size))
                if not memory_ok:
                    logger.warning("GPU 메모리 부족으로 CPU 모드로 전환됨")
            except Exception as e:
                logger.warning(f"비동기 메모리 체크 실패, 기본 체크 사용: {e}")
                self._check_gpu_memory_before_training(data_size)
        else:
            # 기본 동기 메모리 체크
            self._check_gpu_memory_before_training(data_size)

        # 기본 매개변수 설정
        early_stopping_rounds = kwargs.get("early_stopping_rounds", 50)
        num_boost_round = kwargs.get("num_boost_round", 1000)
        feature_names = kwargs.get(
            "feature_names", [f"feature_{i}" for i in range(X.shape[1])]
        )

        # 특성 이름 저장
        self.feature_names = feature_names

        try:
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
            logger.info(f"LightGBM 훈련 시작 (device: {self.params.get('device_type', 'cpu')})")
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

            # GPU 메모리 정보 (훈련 후)
            gpu_memory_info = self.device_manager.check_memory_usage()

        except Exception as e:
            if "gpu" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"GPU 훈련 실패, CPU로 재시도: {e}")
                self._fallback_to_cpu()
                
                # CPU로 재훈련
                return self.fit(X, y, **kwargs)
            else:
                raise

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
                "device_type": self.params.get("device_type", "cpu"),
                "gpu_memory_info": gpu_memory_info,
            }
        )

        # 훈련 완료 표시
        self.is_trained = True
        logger.info(
            f"LightGBM 모델 학습 완료: 최적 반복={self.model.best_iteration}, "
            f"최적 점수={self.model.best_score}, 소요 시간={train_time:.2f}초, "
            f"device={self.params.get('device_type', 'cpu')}"
        )

        return {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
            "device_type": self.params.get("device_type", "cpu"),
            "gpu_memory_info": gpu_memory_info,
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
                    self.params = meta_data.get("params", self.params)
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

    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """Optuna를 사용하여 하이퍼파라미터를 최적화합니다."""
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna가 설치되지 않았습니다. 하이퍼파라미터 최적화를 건너뜁니다.")
            return {}

        def objective(trial: optuna.Trial) -> float:
            param = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }
            if self.gpu_manager.gpu_available:
                param['device'] = 'gpu'

            model = lgb.LGBMClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(10, verbose=False)])
            
            preds_proba = model.predict_proba(X_val)
            
            # 희소 행렬 타입 체크 강화 및 처리 로직 분기
            if SCIPY_SPARSE_AVAILABLE and spmatrix is not None and isinstance(preds_proba, spmatrix):
                dense_preds = preds_proba.toarray()
                preds = dense_preds[:, 1]
            else:
                preds = preds_proba[:, 1]  # type: ignore

            logloss = log_loss(y_val, preds)
            return logloss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"최적화 완료. Best trial: {study.best_trial.value}")
        logger.info(f"최적 파라미터: {study.best_params}")
        
        self.params.update(study.best_params)
        self.model = lgb.LGBMClassifier(**self.params)
        
        return study.best_params


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
