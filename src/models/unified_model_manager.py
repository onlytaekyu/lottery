"""
통합 모델 관리자

TCN, AutoEncoder, LightGBM 모델들을 통합하여 GPU 최적화된 예측 시스템을 제공합니다.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from ..utils.unified_logging import get_logger
from ..utils.cache_manager import UnifiedCachePathManager, CacheManager
from ..utils.model_saver import ModelSaver
from ..utils.unified_memory_manager import UnifiedMemoryManager
from ..utils.enhanced_process_pool import EnhancedProcessPool, DynamicBatchSizeController
from ..utils.unified_feature_vector_validator import UnifiedFeatureVectorValidator

# 모델 임포트
from .dl.tcn_model import TCNModel
from .dl.autoencoder_model import AutoencoderModel
from .ml.lightgbm_model import LightGBMModel


logger = get_logger(__name__)


class UnifiedModelManager:
    """통합 모델 관리자 - GPU 최적화된 모델 통합 시스템"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_path_manager: UnifiedCachePathManager = None,
        batch_controller: DynamicBatchSizeController = None,
        process_pool: EnhancedProcessPool = None,
        feature_validator: UnifiedFeatureVectorValidator = None,
        memory_manager: UnifiedMemoryManager = None,
        model_saver: ModelSaver = None,
    ):
        self.config = config or {}
        self.logger = get_logger(__name__)

        # GPU 가용성 확인
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_available else "cpu")

        # 유틸리티 초기화 (주입)
        self.cache_path_manager = cache_path_manager
        self.batch_controller = batch_controller
        self.process_pool = process_pool
        self.feature_validator = feature_validator
        self.memory_manager = memory_manager
        self.model_saver = model_saver

        # 모델 인스턴스들
        self.models = {}
        self.model_configs = {
            "tcn": {
                "input_dim": 168,
                "num_channels": [128, 64, 32],
                "kernel_size": 3,
                "dropout": 0.2,
                "use_gpu": self.gpu_available,
            },
            "autoencoder": {
                "input_dim": 168,
                "hidden_dims": [128, 64, 32],
                "latent_dim": 16,
                "dropout_rate": 0.2,
                "use_gpu": self.gpu_available,
            },
            "lightgbm": {
                "boosting_type": "gbdt",
                "objective": "regression",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "use_gpu": self.gpu_available,
            },
        }

        # 설정 업데이트
        if config:
            for model_name in self.model_configs:
                if model_name in config:
                    self.model_configs[model_name].update(config[model_name])

        # 모델 성능 통계
        self.model_stats = {
            "tcn": {"predictions": 0, "avg_time": 0.0, "accuracy": 0.0},
            "autoencoder": {"predictions": 0, "avg_time": 0.0, "anomaly_rate": 0.0},
            "lightgbm": {"predictions": 0, "avg_time": 0.0, "feature_importance": {}},
        }

        # 앙상블 가중치
        self.ensemble_weights = {
            "tcn": 0.4,
            "autoencoder": 0.3,
            "lightgbm": 0.3,
        }

        logger.info(f"✅ 통합 모델 관리자 초기화 완료 (GPU: {self.gpu_available})")

    def initialize_models(self, force_reload: bool = False) -> Dict[str, bool]:
        """모든 모델 초기화"""
        results = {}

        if force_reload:
            self.models.clear()

        # TCN 모델 초기화
        if "tcn" not in self.models or force_reload:
            try:
                tcn_cache_manager = None
                if self.cache_path_manager:
                    tcn_cache_manager = CacheManager(
                        path_manager=self.cache_path_manager,
                        cache_type="tcn_model",
                        config=self.config.get("caching", {})
                    )
                self.models["tcn"] = TCNModel(
                    config=self.model_configs["tcn"],
                    cuda_optimizer=self.cuda_optimizer,
                    memory_manager=self.memory_manager,
                    cache_manager=tcn_cache_manager,
                )
                results["tcn"] = True
                logger.info("✅ TCN 모델 초기화 완료")
            except Exception as e:
                logger.error(f"❌ TCN 모델 초기화 실패: {e}")
                results["tcn"] = False

        # AutoEncoder 모델 초기화
        if "autoencoder" not in self.models or force_reload:
            try:
                ae_cache_manager = None
                if self.cache_path_manager:
                    ae_cache_manager = CacheManager(
                        path_manager=self.cache_path_manager,
                        cache_type="autoencoder_model",
                        config=self.config.get("caching", {})
                    )
                self.models["autoencoder"] = AutoencoderModel(
                    config=self.model_configs["autoencoder"],
                    cuda_optimizer=self.cuda_optimizer,
                    memory_manager=self.memory_manager,
                    cache_manager=ae_cache_manager,
                )
                results["autoencoder"] = True
                logger.info("✅ AutoEncoder 모델 초기화 완료")
            except Exception as e:
                logger.error(f"❌ AutoEncoder 모델 초기화 실패: {e}")
                results["autoencoder"] = False

        # LightGBM 모델 초기화
        if "lightgbm" not in self.models or force_reload:
            try:
                lgbm_cache_manager = None
                if self.cache_path_manager:
                    lgbm_cache_manager = CacheManager(
                        path_manager=self.cache_path_manager,
                        cache_type="lightgbm_model",
                        config=self.config.get("caching", {})
                    )
                self.models["lightgbm"] = LightGBMModel(
                    config=self.model_configs["lightgbm"], cache_manager=lgbm_cache_manager
                )
                results["lightgbm"] = True
                logger.info("✅ LightGBM 모델 초기화 완료")
            except Exception as e:
                logger.error(f"❌ LightGBM 모델 초기화 실패: {e}")
                results["lightgbm"] = False

        return results

    def fit_all_models(
        self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, **kwargs
    ) -> Dict[str, Any]:
        """모든 모델 학습"""
        if not self.models:
            self.initialize_models()

        # 입력 검증
        if self.feature_validator and not self.feature_validator.validate_all(
            X, [f"feature_{i}" for i in range(X.shape[1])]
        ):
            raise ValueError("입력 데이터가 검증에 실패했습니다")

        # 동적 배치 크기 조정
        optimal_batch_size = self.batch_controller.get_current_batch_size() if self.batch_controller else 32

        results = {}
        start_time = time.time()

        # 병렬 학습 (CPU 집약적 모델들)
        executor = self.process_pool.get_executor() if self.process_pool else ThreadPoolExecutor(max_workers=3)
        with executor:
            futures = {}

            # TCN 모델 학습
            if "tcn" in self.models:
                futures["tcn"] = executor.submit(
                    self._fit_model_with_monitoring,
                    "tcn",
                    X,
                    y,
                    validation_split,
                    **kwargs,
                )

            # AutoEncoder 모델 학습
            if "autoencoder" in self.models:
                futures["autoencoder"] = executor.submit(
                    self._fit_model_with_monitoring,
                    "autoencoder",
                    X,
                    y,
                    validation_split,
                    **kwargs,
                )

            # LightGBM 모델 학습 (별도 스레드)
            if "lightgbm" in self.models:
                futures["lightgbm"] = executor.submit(
                    self._fit_model_with_monitoring,
                    "lightgbm",
                    X,
                    y,
                    validation_split,
                    **kwargs,
                )

            # 결과 수집
            for model_name, future in futures.items():
                try:
                    results[model_name] = future.result(timeout=3600)  # 1시간 타임아웃
                    logger.info(f"✅ {model_name} 모델 학습 완료")
                except Exception as e:
                    logger.error(f"❌ {model_name} 모델 학습 실패: {e}")
                    results[model_name] = {"success": False, "error": str(e)}

        total_time = time.time() - start_time
        results["total_training_time"] = total_time
        results["batch_size_used"] = optimal_batch_size

        logger.info(f"🎯 전체 모델 학습 완료: {total_time:.2f}초")
        return results

    def _fit_model_with_monitoring(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """모니터링과 함께 모델 학습"""
        try:
            start_time = time.time()

            # 메모리 사용량 체크
            if self.memory_manager and not self.memory_manager.check_available_memory(X.nbytes * 2):
                raise MemoryError(f"{model_name} 모델 학습을 위한 메모리가 부족합니다")

            # 모델 학습
            model = self.models[model_name]
            fit_result = model.fit(X, y, validation_split=validation_split, **kwargs)

            # 성능 통계 업데이트
            training_time = time.time() - start_time
            self.model_stats[model_name]["avg_time"] = training_time
            if self.batch_controller:
                self.batch_controller.report_success(training_time)
            
            return {"success": True, "result": fit_result, "training_time": training_time}

        except Exception as e:
            if self.batch_controller:
                self.batch_controller.report_failure()
            logger.error(f"❌ {model_name} 모델 학습 실패: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def predict_ensemble(
        self, X: np.ndarray, use_weights: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """가중치 기반 앙상블 예측"""
        if not self.models:
            self.initialize_models()
        
        predictions = {}
        for model_name, model in self.models.items():
            try:
                start_time = time.time()
                predictions[model_name] = model.predict(X, **kwargs)
                elapsed_time = time.time() - start_time
                
                # 통계 업데이트
                stats = self.model_stats[model_name]
                stats["avg_time"] = (stats["avg_time"] * stats["predictions"] + elapsed_time) / (stats["predictions"] + 1)
                stats["predictions"] += 1
                
            except Exception as e:
                logger.error(f"{model_name} 예측 실패: {e}")
                predictions[model_name] = None
        
        # None 값 제거
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_predictions:
            raise RuntimeError("모든 모델에서 예측에 실패했습니다.")
            
        return self._combine_predictions(valid_predictions, use_weights)

    def _combine_predictions(self, predictions: Dict[str, Any], use_weights: bool) -> Optional[np.ndarray]:
        """가중치를 사용하여 예측 결합"""
        if not predictions:
            return None

        final_prediction = None
        total_weight = 0.0

        for model_name, pred in predictions.items():
            weight = self.ensemble_weights[model_name] if use_weights else 1.0
            
            if final_prediction is None:
                final_prediction = pred * weight
            else:
                # 예측 결과의 shape이 다를 경우 브로드캐스팅 또는 리샘플링 필요
                if final_prediction.shape != pred.shape:
                    logger.warning(f"예측 결과 shape 불일치: {final_prediction.shape} vs {pred.shape}")
                    # 간단한 해결책: 작은 쪽에 맞춰 자르기 (더 나은 방법 필요)
                    min_len = min(final_prediction.shape[0], pred.shape[0])
                    final_prediction = final_prediction[:min_len]
                    pred = pred[:min_len]
                
                final_prediction += pred * weight
            
            total_weight += weight
            
        if total_weight > 0 and final_prediction is not None:
            return final_prediction / total_weight
        return final_prediction


    def save_all_models(self, save_dir: Optional[str] = None) -> Dict[str, bool]:
        """모든 모델 저장"""
        if not self.model_saver:
            logger.warning("ModelSaver가 없어 모델을 저장할 수 없습니다.")
            return {}

        results = {}
        for model_name, model in self.models.items():
            try:
                self.model_saver.save_model(model, model_name)
                results[model_name] = True
            except Exception as e:
                logger.error(f"{model_name} 모델 저장 실패: {e}")
                results[model_name] = False
        return results

    def load_all_models(self, load_dir: Optional[str] = None) -> Dict[str, bool]:
        """모든 모델 로드"""
        if not self.model_saver:
            logger.warning("ModelSaver가 없어 모델을 로드할 수 없습니다.")
            return {}
            
        results = {}
        for model_name in self.model_configs.keys():
            try:
                self.models[model_name] = self.model_saver.load_model(model_name)
                results[model_name] = True
            except Exception as e:
                logger.error(f"{model_name} 모델 로드 실패: {e}")
                results[model_name] = False
        return results

    def get_model_stats(self) -> Dict[str, Any]:
        """모델별 성능 통계를 반환합니다."""
        # 추가: 실시간으로 평균 시간 다시 계산
        for model_name, stats in self.model_stats.items():
            if "total_time" in stats and "predictions" in stats and stats["predictions"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["predictions"]
        return self.model_stats

    def update_ensemble_weights(self, new_weights: Dict[str, float]):
        """앙상블 가중치를 업데이트하고 정규화합니다."""
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # 정규화
            logger.info(f"가중치 총합이 1이 아니므로 정규화합니다. 합계: {total_weight}")
            new_weights = {k: v / total_weight for k, v in new_weights.items()}

        for model_name, weight in new_weights.items():
            if model_name in self.ensemble_weights:
                self.ensemble_weights[model_name] = weight
        logger.info(f"앙상블 가중치 업데이트 완료: {self.ensemble_weights}")
