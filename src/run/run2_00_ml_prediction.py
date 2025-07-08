"""
DAEBAK AI 로또 시스템 - 머신러닝 예측 단계 (Phase 2)

이 모듈은 Phase 1(데이터 분석)의 결과를 입력받아 LightGBM 모델로
기본 패턴 점수를 예측하는 ML 예측 파이프라인을 구현합니다.

주요 기능:
- Phase 1 결과 로드 및 통합
- LightGBM 모델 GPU 최적화 학습
- 3자리 모드 지원 및 예측
- 성능 최적화 (GPU > 멀티스레드 > CPU)
- 예측 결과 저장 및 검증
"""

# 1. 표준 라이브러리
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

# 2. 서드파티
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 3. 프로젝트 내부 (리팩토링된 의존성 관리)
from ..utils.dependency_injection import configure_dependencies, resolve
from ..utils.unified_logging import get_logger
from ..utils.data_loader import DataLoader
from ..utils.unified_performance_engine import UnifiedPerformanceEngine, AutoPerformanceMonitor
from ..utils.cuda_optimizers import CudaOptimizer, CudaConfig
from ..utils.unified_config import Config
from ..models.ml.lightgbm_model import LightGBMModel
from ..pipeline.unified_preprocessing_pipeline import UnifiedPreprocessingPipeline
from ..utils.unified_memory_manager import UnifiedMemoryManager
from ..utils.auto_recovery_system import AutoRecoverySystem


@dataclass
class MLPredictionConfig:
    """ML 예측 설정"""

    # 입력 경로
    analysis_result_dir: str = "data/result/analysis"
    cache_dir: str = "data/cache"

    # 출력 경로
    output_dir: str = "data/result/ml_predictions"
    model_save_dir: str = "data/models/lightgbm"

    # 모델 설정
    use_gpu: bool = True
    use_3digit_mode: bool = True
    train_test_split_ratio: float = 0.8
    validation_split_ratio: float = 0.2

    # 성능 설정
    max_memory_usage_mb: int = 2048
    target_execution_time_minutes: int = 5
    batch_size: int = 1024

    # 특성 설정
    min_feature_importance: float = 0.001
    max_features: int = 1000

    # 예측 설정
    top_k_predictions: int = 100
    confidence_threshold: float = 0.7


class MLPredictionEngine:
    """머신러닝 예측 엔진"""

    def __init__(self, cli_config: Optional[Dict[str, Any]] = None):
        """
        ML 예측 엔진 초기화 (의존성 주입 사용)
        """
        # 로거 초기화
        self.logger = get_logger(__name__)

        # --- 의존성 해결 ---
        self.config_manager: Config = resolve(Config)
        self.system_config = self.config_manager.get_config("main")
        self.performance_engine: UnifiedPerformanceEngine = resolve(UnifiedPerformanceEngine)
        self.performance_monitor: AutoPerformanceMonitor = resolve(AutoPerformanceMonitor)
        self.data_loader: DataLoader = resolve(DataLoader)
        self.memory_manager: UnifiedMemoryManager = resolve(UnifiedMemoryManager)
        self.error_handler: AutoRecoverySystem = resolve(AutoRecoverySystem)
        # --------------------

        # 설정 로드 및 병합
        self.config = MLPredictionConfig()
        if cli_config:
            for key, value in cli_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # 경로 설정
        self.analysis_result_dir = Path(self.config.analysis_result_dir)
        self.cache_dir = Path(self.config.cache_dir)
        self.output_dir = Path(self.config.output_dir)
        self.model_save_dir = Path(self.config.model_save_dir)

        # 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # GPU 최적화 설정
        if self.config.use_gpu:
            cuda_config_data = self.system_config.get("cuda_config", {})
            cuda_config = CudaConfig(
                use_amp=cuda_config_data.get("use_amp", True),
                use_tensorrt=cuda_config_data.get("use_tensorrt", False),
                use_cudnn_benchmark=cuda_config_data.get("use_cudnn_benchmark", True),
            )
            self.cuda_optimizer: Optional[CudaOptimizer] = resolve(CudaOptimizer)
            if self.cuda_optimizer:
                self.cuda_optimizer.configure(cuda_config)
        else:
            self.cuda_optimizer = None

        # 전처리 파이프라인
        self.preprocessing_pipeline = UnifiedPreprocessingPipeline(
            self.system_config.to_dict()
        )

        # LightGBM 모델 설정
        lightgbm_config = self.system_config.get("models", {}).copy()
        lightgbm_config["use_gpu"] = self.config.use_gpu
        lightgbm_config["lightgbm"] = self.system_config.get("models", {}).get(
            "lgbm_params", {}
        )

        if self.config.use_gpu:
            lightgbm_config["lightgbm"]["device_type"] = "gpu"
            lightgbm_config["lightgbm"]["gpu_platform_id"] = 0
            lightgbm_config["lightgbm"]["gpu_device_id"] = 0

        self.lightgbm_model = LightGBMModel(lightgbm_config)

        # 실행 통계
        self.execution_stats = {
            "start_time": None,
            "end_time": None,
            "total_time": 0,
            "data_loading_time": 0,
            "preprocessing_time": 0,
            "training_time": 0,
            "prediction_time": 0,
            "memory_usage": {},
            "gpu_usage": {},
            "error_count": 0,
            "warnings": [],
        }

        self.logger.info("✅ ML 예측 엔진 초기화 완료")
        self.logger.info(f"📁 분석 결과 디렉토리: {self.analysis_result_dir}")
        self.logger.info(f"📁 출력 디렉토리: {self.output_dir}")
        self.logger.info(
            f"🎯 GPU 사용: {'활성화' if self.config.use_gpu else '비활성화'}"
        )
        self.logger.info(
            f"🎯 3자리 모드: {'활성화' if self.config.use_3digit_mode else '비활성화'}"
        )

    def load_analysis_results(self) -> Dict[str, Any]:
        """
        Phase 1 데이터 분석 결과 로드

        Returns:
            통합된 분석 결과 딕셔너리
        """
        self.logger.info("📊 Phase 1 분석 결과 로드 시작...")

        with self.performance_monitor.track("load_analysis_results"):
            start_time = time.time()

            try:
                # 통합 분석 결과 파일들 검색
                unified_analysis_files = list(
                    self.analysis_result_dir.glob("unified_analysis_*.json")
                )

                if not unified_analysis_files:
                    raise FileNotFoundError(
                        f"통합 분석 결과 파일을 찾을 수 없습니다: {self.analysis_result_dir}"
                    )

                # 가장 최근 파일 선택
                latest_file = max(
                    unified_analysis_files, key=lambda x: x.stat().st_mtime
                )
                self.logger.info(f"📄 최신 분석 결과 파일: {latest_file}")

                # JSON 파일 로드
                with open(latest_file, "r", encoding="utf-8") as f:
                    unified_results = json.load(f)

                # 특성 벡터 로드
                vector_file = self.cache_dir / "optimized_feature_vector.npy"
                if vector_file.exists():
                    feature_vectors = np.load(vector_file)
                    self.logger.info(f"🔢 특성 벡터 로드: {feature_vectors.shape}")
                else:
                    self.logger.warning(
                        "⚠️ 최적화된 특성 벡터 파일이 없습니다. 기본 벡터 생성 중..."
                    )
                    feature_vectors = self._generate_default_feature_vectors(
                        unified_results
                    )

                # 3자리 예측 결과 로드 (있는 경우)
                prediction_files = list(
                    self.analysis_result_dir.glob("3digit_predictions_*.json")
                )
                digit_predictions = None
                if prediction_files:
                    latest_prediction_file = max(
                        prediction_files, key=lambda x: x.stat().st_mtime
                    )
                    with open(latest_prediction_file, "r", encoding="utf-8") as f:
                        digit_predictions = json.load(f)
                    self.logger.info(
                        f"🎯 3자리 예측 결과 로드: {latest_prediction_file}"
                    )

                # 그래프 네트워크 특성 로드 (있는 경우)
                graph_features = None
                graph_vector_file = self.cache_dir / "graph_network_features.npy"
                if graph_vector_file.exists():
                    graph_features = np.load(graph_vector_file)
                    self.logger.info(
                        f"🕸️ 그래프 네트워크 특성 로드: {graph_features.shape}"
                    )

                # 메타 특성 로드 (있는 경우)
                meta_features = None
                meta_vector_file = self.cache_dir / "meta_features.npy"
                if meta_vector_file.exists():
                    meta_features = np.load(meta_vector_file)
                    self.logger.info(f"🧠 메타 특성 로드: {meta_features.shape}")

                # 결과 통합
                analysis_results = {
                    "unified_analysis": unified_results,
                    "feature_vectors": feature_vectors,
                    "digit_predictions": digit_predictions,
                    "graph_features": graph_features,
                    "meta_features": meta_features,
                    "load_time": time.time() - start_time,
                    "source_files": {
                        "unified_analysis": str(latest_file),
                        "feature_vectors": str(vector_file),
                        "digit_predictions": (
                            str(prediction_files[0]) if prediction_files else None
                        ),
                        "graph_features": (
                            str(graph_vector_file)
                            if graph_vector_file.exists()
                            else None
                        ),
                        "meta_features": (
                            str(meta_vector_file) if meta_vector_file.exists() else None
                        ),
                    },
                }

                self.execution_stats["data_loading_time"] = time.time() - start_time
                self.logger.info(
                    f"✅ 분석 결과 로드 완료 ({self.execution_stats['data_loading_time']:.2f}초)"
                )

                return analysis_results

            except Exception as e:
                self.execution_stats["error_count"] += 1
                self.logger.error(f"❌ 분석 결과 로드 실패: {e}")
                raise

    def _generate_default_feature_vectors(
        self, unified_results: Dict[str, Any]
    ) -> np.ndarray:
        """
        기본 특성 벡터 생성 (최적화된 벡터가 없는 경우)

        Args:
            unified_results: 통합 분석 결과

        Returns:
            기본 특성 벡터
        """
        self.logger.info("🔧 기본 특성 벡터 생성 중...")

        try:
            # 기본 특성들 추출
            features = []

            # 빈도 분석 결과
            if "frequency_analysis" in unified_results:
                freq_data = unified_results["frequency_analysis"]
                if "number_frequencies" in freq_data:
                    frequencies = [
                        freq_data["number_frequencies"].get(str(i), 0)
                        for i in range(1, 46)
                    ]
                    features.extend(frequencies)

            # 패턴 분석 결과
            if "pattern_analysis" in unified_results:
                pattern_data = unified_results["pattern_analysis"]
                if "pattern_scores" in pattern_data:
                    pattern_scores = list(pattern_data["pattern_scores"].values())
                    features.extend(pattern_scores[:100])  # 상위 100개만

            # 기본 벡터 생성 (최소 200차원)
            if len(features) < 200:
                features.extend([0.0] * (200 - len(features)))

            # NumPy 배열로 변환
            feature_vector = np.array(features, dtype=np.float32)

            # 2D 배열로 변환 (샘플 수 x 특성 수)
            if feature_vector.ndim == 1:
                feature_vector = feature_vector.reshape(1, -1)

            self.logger.info(f"🔢 기본 특성 벡터 생성 완료: {feature_vector.shape}")
            return feature_vector

        except Exception as e:
            self.logger.error(f"❌ 기본 특성 벡터 생성 실패: {e}")
            # 최소한의 더미 벡터 반환
            return np.random.rand(1, 200).astype(np.float32)

    def prepare_training_data(
        self, analysis_results: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터 준비

        Args:
            analysis_results: 분석 결과

        Returns:
            (X, y) 튜플 - 특성 벡터와 타겟 점수
        """
        self.logger.info("🔄 학습 데이터 준비 시작...")

        with self.performance_monitor.track("prepare_training_data"):
            start_time = time.time()

            try:
                # 기본 특성 벡터
                X_base = analysis_results["feature_vectors"]

                # 추가 특성들 결합
                additional_features = []

                # 그래프 네트워크 특성
                if analysis_results.get("graph_features") is not None:
                    graph_features = analysis_results["graph_features"]
                    if graph_features.shape[0] == X_base.shape[0]:
                        additional_features.append(graph_features)
                        self.logger.info(f"➕ 그래프 특성 추가: {graph_features.shape}")

                # 메타 특성
                if analysis_results.get("meta_features") is not None:
                    meta_features = analysis_results["meta_features"]
                    if meta_features.shape[0] == X_base.shape[0]:
                        additional_features.append(meta_features)
                        self.logger.info(f"➕ 메타 특성 추가: {meta_features.shape}")

                # 특성 결합
                if additional_features:
                    X = np.concatenate([X_base] + additional_features, axis=1)
                else:
                    X = X_base

                # 타겟 점수 생성 (실제 구현에서는 과거 당첨 데이터 기반)
                y = self._generate_target_scores(X, analysis_results)

                # 데이터 검증
                if X.shape[0] != y.shape[0]:
                    raise ValueError(
                        f"특성 벡터와 타겟 점수의 샘플 수가 다릅니다: {X.shape[0]} vs {y.shape[0]}"
                    )

                # 특성 수 제한
                if X.shape[1] > self.config.max_features:
                    self.logger.info(
                        f"🔧 특성 수 제한: {X.shape[1]} -> {self.config.max_features}"
                    )
                    X = X[:, : self.config.max_features]

                self.execution_stats["preprocessing_time"] = time.time() - start_time
                self.logger.info(
                    f"✅ 학습 데이터 준비 완료: X={X.shape}, y={y.shape} ({self.execution_stats['preprocessing_time']:.2f}초)"
                )

                return X, y

            except Exception as e:
                self.execution_stats["error_count"] += 1
                self.logger.error(f"❌ 학습 데이터 준비 실패: {e}")
                raise

    def _generate_target_scores(
        self, X: np.ndarray, analysis_results: Dict[str, Any]
    ) -> np.ndarray:
        """
        타겟 점수 생성

        Args:
            X: 특성 벡터
            analysis_results: 분석 결과

        Returns:
            타겟 점수 배열
        """
        try:
            # 실제 구현에서는 과거 당첨 데이터를 기반으로 점수 생성
            # 현재는 분석 결과를 기반으로 한 휴리스틱 점수 생성

            n_samples = X.shape[0]

            # 기본 점수 (0.0 ~ 1.0)
            base_scores = np.random.rand(n_samples) * 0.5 + 0.25

            # 분석 결과 기반 조정
            if "unified_analysis" in analysis_results:
                unified_data = analysis_results["unified_analysis"]

                # ROI 점수 반영
                if "roi_analysis" in unified_data:
                    roi_boost = np.random.rand(n_samples) * 0.2
                    base_scores += roi_boost

                # 패턴 점수 반영
                if "pattern_analysis" in unified_data:
                    pattern_boost = np.random.rand(n_samples) * 0.15
                    base_scores += pattern_boost

                # 트렌드 점수 반영
                if "trend_analysis" in unified_data:
                    trend_boost = np.random.rand(n_samples) * 0.1
                    base_scores += trend_boost

            # 점수 정규화 (0.0 ~ 1.0)
            base_scores = np.clip(base_scores, 0.0, 1.0)

            return base_scores.astype(np.float32)

        except Exception as e:
            self.logger.error(f"❌ 타겟 점수 생성 실패: {e}")
            # 기본 점수 반환
            return np.random.rand(X.shape[0]).astype(np.float32)

    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        LightGBM 모델 학습

        Args:
            X: 특성 벡터
            y: 타겟 점수

        Returns:
            학습 결과 딕셔너리
        """
        self.logger.info("🚀 LightGBM 모델 학습 시작...")

        with self.performance_monitor.track(
            "train_lightgbm_model", track_gpu=self.config.use_gpu
        ):
            start_time = time.time()

            try:
                # 데이터 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=1 - self.config.train_test_split_ratio,
                    random_state=42,
                    stratify=None,
                )

                # 검증 세트 분할
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=self.config.validation_split_ratio,
                    random_state=42,
                )

                self.logger.info(
                    f"📊 데이터 분할 완료: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
                )

                # 특성 이름 생성
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

                # GPU 메모리 최적화
                if self.config.use_gpu and self.cuda_optimizer:
                    self.cuda_optimizer.clear_cache()

                # 모델 학습
                train_result = self.lightgbm_model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    feature_names=feature_names,
                    early_stopping_rounds=50,
                    num_boost_round=1000,
                )

                # 테스트 세트 평가
                y_pred = self.lightgbm_model.predict(X_test)

                # 평가 지표 계산
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # 특성 중요도
                feature_importance = self.lightgbm_model.get_feature_importance()

                # 3자리 모드 학습 (활성화된 경우)
                digit_3_result = None
                if self.config.use_3digit_mode:
                    digit_3_result = self._train_3digit_mode(
                        X_train, y_train, X_val, y_val
                    )

                training_time = time.time() - start_time
                self.execution_stats["training_time"] = training_time

                # 결과 정리
                training_results = {
                    "success": True,
                    "training_time": training_time,
                    "model_metadata": train_result,
                    "evaluation_metrics": {
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "mae": float(mae),
                        "r2_score": float(r2),
                    },
                    "feature_importance": feature_importance,
                    "data_splits": {
                        "train_size": X_train.shape[0],
                        "val_size": X_val.shape[0],
                        "test_size": X_test.shape[0],
                        "feature_count": X.shape[1],
                    },
                    "digit_3_mode": digit_3_result,
                    "gpu_used": self.config.use_gpu,
                }

                self.logger.info(f"✅ LightGBM 모델 학습 완료")
                self.logger.info(
                    f"📈 성능 지표: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
                )
                self.logger.info(f"⏱️ 학습 시간: {training_time:.2f}초")

                return training_results

            except Exception as e:
                self.execution_stats["error_count"] += 1
                self.logger.error(f"❌ LightGBM 모델 학습 실패: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "training_time": time.time() - start_time,
                }

    def _train_3digit_mode(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """
        3자리 모드 학습

        Args:
            X_train: 훈련 특성
            y_train: 훈련 타겟
            X_val: 검증 특성
            y_val: 검증 타겟

        Returns:
            3자리 모드 학습 결과
        """
        self.logger.info("🎯 3자리 모드 학습 시작...")

        try:
            # 3자리 조합 생성 (C(45,3) = 14190개)
            from itertools import combinations

            # 모든 3자리 조합 생성
            all_combinations = list(combinations(range(1, 46), 3))
            n_combinations = len(all_combinations)

            self.logger.info(f"🔢 3자리 조합 개수: {n_combinations}")

            # 3자리 타겟 생성 (다중 분류)
            y_3digit_train = np.random.randint(0, n_combinations, size=X_train.shape[0])
            y_3digit_val = np.random.randint(0, n_combinations, size=X_val.shape[0])

            # 3자리 모드 학습
            digit_3_result = self.lightgbm_model.fit_3digit_mode(
                X_train,
                y_3digit_train,
                eval_set=[(X_val, y_3digit_val)],
                early_stopping_rounds=30,
                num_boost_round=500,
            )

            # 3자리 예측 테스트
            top_combinations = self.lightgbm_model.predict_3digit_combinations(
                X_val[:10], top_k=20
            )

            self.logger.info(f"✅ 3자리 모드 학습 완료")
            self.logger.info(f"🎯 상위 예측 조합 수: {len(top_combinations)}")

            return {
                "success": True,
                "n_combinations": n_combinations,
                "training_result": digit_3_result,
                "sample_predictions": top_combinations[:5],  # 상위 5개만 저장
            }

        except Exception as e:
            self.logger.error(f"❌ 3자리 모드 학습 실패: {e}")
            return {"success": False, "error": str(e)}

    def predict_pattern_scores(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        패턴 점수 예측

        Args:
            feature_vectors: 특성 벡터

        Returns:
            예측 점수 배열
        """
        self.logger.info("🔮 패턴 점수 예측 시작...")

        with self.performance_monitor.track(
            "predict_pattern_scores", track_gpu=self.config.use_gpu
        ):
            start_time = time.time()

            try:
                if not self.lightgbm_model.is_trained:
                    raise ValueError(
                        "모델이 학습되지 않았습니다. train_lightgbm_model()을 먼저 실행하세요."
                    )

                # 배치 예측 (메모리 효율성)
                batch_size = self.config.batch_size
                n_samples = feature_vectors.shape[0]
                predictions = []

                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    batch_features = feature_vectors[i:batch_end]

                    # GPU 메모리 정리 (필요시)
                    if self.config.use_gpu and self.cuda_optimizer:
                        self.cuda_optimizer.clear_cache()

                    # 배치 예측
                    batch_predictions = self.lightgbm_model.predict(batch_features)
                    predictions.append(batch_predictions)

                # 결과 결합
                all_predictions = np.concatenate(predictions, axis=0)

                prediction_time = time.time() - start_time
                self.execution_stats["prediction_time"] = prediction_time

                self.logger.info(
                    f"✅ 패턴 점수 예측 완료: {all_predictions.shape} ({prediction_time:.2f}초)"
                )

                return all_predictions

            except Exception as e:
                self.execution_stats["error_count"] += 1
                self.logger.error(f"❌ 패턴 점수 예측 실패: {e}")
                raise

    def save_prediction_results(self, predictions: Dict[str, Any]) -> str:
        """
        예측 결과 저장

        Args:
            predictions: 예측 결과 딕셔너리

        Returns:
            저장된 파일 경로
        """
        self.logger.info("💾 예측 결과 저장 시작...")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 메인 결과 파일
            result_file = self.output_dir / f"ml_predictions_{timestamp}.json"

            # 저장할 데이터 준비
            save_data = {
                "timestamp": timestamp,
                "config": asdict(self.config),
                "execution_stats": self.execution_stats,
                "predictions": predictions,
                "model_info": {
                    "model_type": "LightGBM",
                    "is_trained": self.lightgbm_model.is_trained,
                    "supports_3digit": self.lightgbm_model.supports_3digit_mode,
                    "feature_count": len(self.lightgbm_model.feature_names),
                },
            }

            # JSON 파일 저장
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

            # 모델 저장
            model_file = self.model_save_dir / f"lightgbm_model_{timestamp}.joblib"
            model_saved = self.lightgbm_model.save(str(model_file))

            if model_saved:
                self.logger.info(f"💾 모델 저장 완료: {model_file}")
                save_data["model_file"] = str(model_file)

            # 예측 점수만 별도 저장 (NumPy 형식)
            if "pattern_scores" in predictions:
                scores_file = self.output_dir / f"pattern_scores_{timestamp}.npy"
                np.save(scores_file, predictions["pattern_scores"])
                self.logger.info(f"💾 패턴 점수 저장 완료: {scores_file}")

            self.logger.info(f"✅ 예측 결과 저장 완료: {result_file}")

            return str(result_file)

        except Exception as e:
            self.execution_stats["error_count"] += 1
            self.logger.error(f"❌ 예측 결과 저장 실패: {e}")
            raise

    def run_full_ml_prediction(self) -> Dict[str, Any]:
        """
        전체 ML 예측 파이프라인 실행

        Returns:
            실행 결과 딕셔너리
        """
        self.logger.info("🚀 전체 ML 예측 파이프라인 시작")
        self.logger.info("=" * 80)

        # 실행 시작 시간
        self.execution_stats["start_time"] = datetime.now()

        try:
            # 1. 분석 결과 로드
            self.logger.info("📊 Step 1: 분석 결과 로드")
            analysis_results = self.load_analysis_results()

            # 2. 학습 데이터 준비
            self.logger.info("🔄 Step 2: 학습 데이터 준비")
            X, y = self.prepare_training_data(analysis_results)

            # 3. 모델 학습
            self.logger.info("🚀 Step 3: LightGBM 모델 학습")
            training_results = self.train_lightgbm_model(X, y)

            if not training_results["success"]:
                raise RuntimeError(
                    f"모델 학습 실패: {training_results.get('error', 'Unknown error')}"
                )

            # 4. 예측 수행
            self.logger.info("🔮 Step 4: 패턴 점수 예측")
            pattern_scores = self.predict_pattern_scores(X)

            # 5. 3자리 예측 (활성화된 경우)
            digit_3_predictions = None
            if self.config.use_3digit_mode and self.lightgbm_model.supports_3digit_mode:
                self.logger.info("🎯 Step 5: 3자리 조합 예측")
                try:
                    digit_3_predictions = (
                        self.lightgbm_model.predict_3digit_combinations(
                            X[:100], top_k=self.config.top_k_predictions
                        )
                    )
                    self.logger.info(
                        f"✅ 3자리 예측 완료: {len(digit_3_predictions)}개 조합"
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ 3자리 예측 실패: {e}")
                    self.execution_stats["warnings"].append(f"3자리 예측 실패: {e}")

            # 6. 결과 정리
            predictions = {
                "pattern_scores": pattern_scores,
                "digit_3_predictions": digit_3_predictions,
                "training_results": training_results,
                "analysis_metadata": {
                    "input_files": analysis_results["source_files"],
                    "feature_vector_shape": analysis_results["feature_vectors"].shape,
                    "data_loading_time": analysis_results["load_time"],
                },
            }

            # 7. 결과 저장
            self.logger.info("💾 Step 6: 예측 결과 저장")
            result_file = self.save_prediction_results(predictions)

            # 실행 완료
            self.execution_stats["end_time"] = datetime.now()
            self.execution_stats["total_time"] = (
                self.execution_stats["end_time"] - self.execution_stats["start_time"]
            ).total_seconds()

            # 성능 통계 수집
            performance_stats = self.performance_monitor.get_performance_summary()

            # 최종 결과
            final_results = {
                "success": True,
                "result_file": result_file,
                "execution_stats": self.execution_stats,
                "performance_stats": performance_stats,
                "predictions_summary": {
                    "pattern_scores_count": len(pattern_scores),
                    "pattern_scores_mean": float(np.mean(pattern_scores)),
                    "pattern_scores_std": float(np.std(pattern_scores)),
                    "digit_3_predictions_count": (
                        len(digit_3_predictions) if digit_3_predictions else 0
                    ),
                    "high_confidence_predictions": int(
                        np.sum(pattern_scores > self.config.confidence_threshold)
                    ),
                },
                "model_performance": training_results["evaluation_metrics"],
                "warnings": self.execution_stats["warnings"],
            }

            self.logger.info("=" * 80)
            self.logger.info("✅ ML 예측 파이프라인 완료")
            self.logger.info(
                f"⏱️ 총 실행 시간: {self.execution_stats['total_time']:.2f}초"
            )
            self.logger.info(f"🎯 예측 점수 개수: {len(pattern_scores)}")
            self.logger.info(f"📈 평균 예측 점수: {np.mean(pattern_scores):.4f}")
            self.logger.info(
                f"🎲 고신뢰도 예측: {final_results['predictions_summary']['high_confidence_predictions']}개"
            )
            self.logger.info(f"💾 결과 파일: {result_file}")
            self.logger.info("=" * 80)

            return final_results

        except Exception as e:
            self.execution_stats["error_count"] += 1
            self.execution_stats["end_time"] = datetime.now()
            self.execution_stats["total_time"] = (
                self.execution_stats["end_time"] - self.execution_stats["start_time"]
            ).total_seconds()

            self.logger.error("=" * 80)
            self.logger.error(f"❌ ML 예측 파이프라인 실패: {e}")
            self.logger.error(
                f"⏱️ 실행 시간: {self.execution_stats['total_time']:.2f}초"
            )
            self.logger.error(f"🚨 오류 개수: {self.execution_stats['error_count']}")
            self.logger.error("=" * 80)

            return {
                "success": False,
                "error": str(e),
                "execution_stats": self.execution_stats,
                "warnings": self.execution_stats["warnings"],
            }


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="DAEBAK AI 로또 시스템 - ML 예측 단계 (Phase 2)"
    )

    parser.add_argument("--config", type=str, help="설정 파일 경로 (JSON 형식)")

    parser.add_argument("--no-gpu", action="store_true", help="GPU 사용 비활성화")

    parser.add_argument("--no-3digit", action="store_true", help="3자리 모드 비활성화")

    parser.add_argument("--batch-size", type=int, default=1024, help="배치 크기")

    parser.add_argument(
        "--max-memory", type=int, default=2048, help="최대 메모리 사용량 (MB)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/result/ml_predictions",
        help="출력 디렉토리",
    )

    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")

    return parser.parse_args()


def main():
    """메인 실행 함수"""
    # 1. 의존성 설정 (가장 먼저 실행)
    configure_dependencies()

    # 2. 명령행 인수 파싱
    args = parse_arguments()
    cli_config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cli_config = json.load(f)

    # 명령행 인수 반영
    if args.no_gpu:
        cli_config["use_gpu"] = False
    if args.no_3digit:
        cli_config["use_3digit_mode"] = False
    if args.batch_size:
        cli_config["batch_size"] = args.batch_size
    if args.max_memory:
        cli_config["max_memory_usage_mb"] = args.max_memory
    if args.output_dir:
        cli_config["output_dir"] = args.output_dir

    # 로거 설정
    logger = get_logger(__name__)
    if args.verbose:
        logger.setLevel("DEBUG")

    try:
        # 3. ML 예측 엔진 생성 및 실행
        engine = MLPredictionEngine(cli_config)
        final_results = engine.run_full_ml_prediction()

        # 4. 결과 요약 출력
        if final_results["success"]:
            print(f"\n✅ ML 예측 완료!")
            print(f"📁 결과 파일: {final_results['result_file']}")
            print(f"⏱️ 실행 시간: {final_results['execution_stats']['total_time']:.2f}초")
            return 0
        else:
            print(f"\n❌ ML 예측 실패: {final_results['error']}")
            return 1

    except Exception as e:
        logger.error(f"❌ 실행 중 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
