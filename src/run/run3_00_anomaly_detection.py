"""
DAEBAK AI 로또 시스템 - 이상감지 단계 (Phase 3)

이 모듈은 Phase 1(데이터 분석)과 Phase 2(ML 예측)의 결과를 입력받아
AutoEncoder 모델을 사용해 이상 패턴을 감지하고 감점하는 이상감지 파이프라인을 구현합니다.

주요 기능:
- Phase 1, 2 결과 로드 및 통합
- AutoEncoder 모델 GPU 최적화 학습
- 정상 패턴 학습 및 이상치 감지
- 재구성 오류 기반 감점 시스템
- 동적 임계값 조정 및 성능 최적화
"""

# 1. 표준 라이브러리
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings

# 2. 서드파티
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# 3. 프로젝트 내부 (리팩토링된 의존성 관리)
from ..utils.dependency_injection import configure_dependencies, resolve
from ..utils.unified_logging import get_logger
from ..utils.data_loader import DataLoader
from ..utils.unified_performance_engine import UnifiedPerformanceEngine, AutoPerformanceMonitor
from ..utils.cuda_optimizers import CudaOptimizer, CudaConfig
from ..utils.unified_config import Config
from ..models.dl.autoencoder_model import AutoencoderModel
from ..pipeline.optimized_autoencoder_preprocessor import OptimizedAutoEncoderPreprocessor
from ..utils.unified_memory_manager import UnifiedMemoryManager
from ..utils.auto_recovery_system import AutoRecoverySystem

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class AnomalyDetectionConfig:
    """이상감지 설정"""

    # 입력 경로
    analysis_result_dir: str = "data/result/analysis"
    ml_predictions_dir: str = "data/result/ml_predictions"
    cache_dir: str = "data/cache"

    # 출력 경로
    output_dir: str = "data/result/anomaly_detection"
    model_save_dir: str = "data/models/autoencoder"

    # 모델 설정
    use_gpu: bool = True
    latent_dim: int = 32
    hidden_dims: List[int] = None
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100

    # 이상감지 설정
    zscore_threshold: float = 2.5
    anomaly_penalty_factor: float = 0.3
    min_penalty: float = 0.1
    max_penalty: float = 0.8

    # 성능 설정
    max_memory_usage_mb: int = 2048
    target_execution_time_minutes: int = 5
    early_stopping_patience: int = 10

    # 검증 설정
    validation_split: float = 0.2
    contamination_rate: float = 0.1

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class AnomalyDetectionEngine:
    """이상감지 엔진"""

    def __init__(self, cli_config: Optional[Dict[str, Any]] = None):
        """
        이상감지 엔진 초기화 (의존성 주입 사용)
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
        self.config = AnomalyDetectionConfig()
        if cli_config:
            for key, value in cli_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # 경로 설정
        self.analysis_result_dir = Path(self.config.analysis_result_dir)
        self.ml_predictions_dir = Path(self.config.ml_predictions_dir)
        self.cache_dir = Path(self.config.cache_dir)
        self.output_dir = Path(self.config.output_dir)
        self.model_save_dir = Path(self.config.model_save_dir)

        # 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # GPU 최적화 설정
        self.device = torch.device(
            "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        )

        if self.config.use_gpu and torch.cuda.is_available():
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

        # 전처리기 초기화
        preprocessor_config = {
            "input_dim": 200,  # 기본값, 실제 데이터에 따라 조정
            "latent_dim": self.config.latent_dim,
            "hidden_dims": self.config.hidden_dims,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "use_gpu": self.config.use_gpu,
        }
        self.preprocessor = OptimizedAutoEncoderPreprocessor(preprocessor_config)

        # AutoEncoder 모델 설정
        autoencoder_config = {
            "use_gpu": self.config.use_gpu,
            "autoencoder": {
                "input_dim": 200,  # 실제 데이터에 따라 조정
                "hidden_dims": self.config.hidden_dims,
                "latent_dim": self.config.latent_dim,
                "learning_rate": self.config.learning_rate,
                "dropout_rate": 0.2,
                "batch_norm": True,
                "zscore_threshold": self.config.zscore_threshold,
            },
        }
        self.autoencoder_model = AutoencoderModel(autoencoder_config)

        # 스케일러
        self.scaler = StandardScaler()

        # 실행 통계
        self.execution_stats = {
            "start_time": None,
            "end_time": None,
            "total_time": 0,
            "data_loading_time": 0,
            "preprocessing_time": 0,
            "training_time": 0,
            "detection_time": 0,
            "memory_usage": {},
            "gpu_usage": {},
            "error_count": 0,
            "warnings": [],
        }

        # 이상감지 통계
        self.anomaly_stats = {
            "total_samples": 0,
            "anomaly_count": 0,
            "anomaly_rate": 0.0,
            "mean_reconstruction_error": 0.0,
            "std_reconstruction_error": 0.0,
            "threshold_value": 0.0,
            "penalty_applied": 0,
        }

        self.logger.info("✅ 이상감지 엔진 초기화 완료")
        self.logger.info(f"📁 분석 결과 디렉토리: {self.analysis_result_dir}")
        self.logger.info(f"📁 ML 예측 디렉토리: {self.ml_predictions_dir}")
        self.logger.info(f"📁 출력 디렉토리: {self.output_dir}")
        self.logger.info(
            f"🎯 GPU 사용: {'활성화' if self.config.use_gpu else '비활성화'}"
        )
        self.logger.info(f"🧠 잠재 차원: {self.config.latent_dim}")
        self.logger.info(f"📊 Z-Score 임계값: {self.config.zscore_threshold}")

    def load_previous_results(self) -> Dict[str, Any]:
        """
        이전 단계 결과 로드 (run1, run2)

        Returns:
            통합된 이전 결과 딕셔너리
        """
        self.logger.info("📊 이전 단계 결과 로드 시작...")

        with self.performance_monitor.track("load_previous_results"):
            start_time = time.time()

            try:
                # Phase 1 결과 로드
                unified_analysis_files = list(
                    self.analysis_result_dir.glob("unified_analysis_*.json")
                )
                if not unified_analysis_files:
                    raise FileNotFoundError(
                        f"Phase 1 분석 결과 파일을 찾을 수 없습니다: {self.analysis_result_dir}"
                    )

                latest_analysis_file = max(
                    unified_analysis_files, key=lambda x: x.stat().st_mtime
                )
                self.logger.info(f"📄 최신 분석 결과 파일: {latest_analysis_file}")

                with open(latest_analysis_file, "r", encoding="utf-8") as f:
                    analysis_results = json.load(f)

                # Phase 2 결과 로드
                ml_prediction_files = list(
                    self.ml_predictions_dir.glob("ml_predictions_*.json")
                )
                if not ml_prediction_files:
                    raise FileNotFoundError(
                        f"Phase 2 ML 예측 결과 파일을 찾을 수 없습니다: {self.ml_predictions_dir}"
                    )

                latest_ml_file = max(
                    ml_prediction_files, key=lambda x: x.stat().st_mtime
                )
                self.logger.info(f"📄 최신 ML 예측 파일: {latest_ml_file}")

                with open(latest_ml_file, "r", encoding="utf-8") as f:
                    ml_results = json.load(f)

                # 특성 벡터 로드
                feature_vectors = None
                vector_file = self.cache_dir / "optimized_feature_vector.npy"
                if vector_file.exists():
                    feature_vectors = np.load(vector_file)
                    self.logger.info(f"🔢 특성 벡터 로드: {feature_vectors.shape}")

                # 패턴 점수 로드
                pattern_scores = None
                score_files = list(self.ml_predictions_dir.glob("pattern_scores_*.npy"))
                if score_files:
                    latest_score_file = max(
                        score_files, key=lambda x: x.stat().st_mtime
                    )
                    pattern_scores = np.load(latest_score_file)
                    self.logger.info(f"🎯 패턴 점수 로드: {pattern_scores.shape}")

                # 추가 특성 로드
                additional_features = {}

                # 그래프 네트워크 특성
                graph_vector_file = self.cache_dir / "graph_network_features.npy"
                if graph_vector_file.exists():
                    additional_features["graph_features"] = np.load(graph_vector_file)
                    self.logger.info(
                        f"🕸️ 그래프 특성 로드: {additional_features['graph_features'].shape}"
                    )

                # 메타 특성
                meta_vector_file = self.cache_dir / "meta_features.npy"
                if meta_vector_file.exists():
                    additional_features["meta_features"] = np.load(meta_vector_file)
                    self.logger.info(
                        f"🧠 메타 특성 로드: {additional_features['meta_features'].shape}"
                    )

                # 결과 통합
                combined_results = {
                    "analysis_results": analysis_results,
                    "ml_results": ml_results,
                    "feature_vectors": feature_vectors,
                    "pattern_scores": pattern_scores,
                    "additional_features": additional_features,
                    "load_time": time.time() - start_time,
                    "source_files": {
                        "analysis": str(latest_analysis_file),
                        "ml_predictions": str(latest_ml_file),
                        "feature_vectors": (
                            str(vector_file) if vector_file.exists() else None
                        ),
                        "pattern_scores": (
                            str(latest_score_file) if score_files else None
                        ),
                    },
                }

                self.execution_stats["data_loading_time"] = time.time() - start_time
                self.logger.info(
                    f"✅ 이전 단계 결과 로드 완료 ({self.execution_stats['data_loading_time']:.2f}초)"
                )

                return combined_results

            except Exception as e:
                self.execution_stats["error_count"] += 1
                self.logger.error(f"❌ 이전 단계 결과 로드 실패: {e}")
                raise

    def prepare_autoencoder_data(
        self, all_results: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        AutoEncoder 학습용 데이터 준비

        Args:
            all_results: 모든 이전 결과

        Returns:
            (X, feature_names) 튜플 - 특성 벡터와 특성 이름
        """
        self.logger.info("🔄 AutoEncoder 학습 데이터 준비 시작...")

        with self.performance_monitor.track("prepare_autoencoder_data"):
            start_time = time.time()

            try:
                # 기본 특성 벡터
                X_base = all_results["feature_vectors"]
                if X_base is None:
                    raise ValueError("기본 특성 벡터가 없습니다")

                # 특성 리스트 초기화
                feature_matrices = [X_base]
                feature_names = [f"base_feature_{i}" for i in range(X_base.shape[1])]

                # ML 예측 점수 추가
                if all_results["pattern_scores"] is not None:
                    pattern_scores = all_results["pattern_scores"]
                    if pattern_scores.ndim == 1:
                        pattern_scores = pattern_scores.reshape(-1, 1)

                    # 샘플 수 맞춤
                    if pattern_scores.shape[0] != X_base.shape[0]:
                        if pattern_scores.shape[0] > X_base.shape[0]:
                            pattern_scores = pattern_scores[: X_base.shape[0]]
                        else:
                            # 패턴 점수가 적으면 반복
                            repeat_count = (
                                X_base.shape[0] + pattern_scores.shape[0] - 1
                            ) // pattern_scores.shape[0]
                            pattern_scores = np.tile(pattern_scores, (repeat_count, 1))[
                                : X_base.shape[0]
                            ]

                    feature_matrices.append(pattern_scores)
                    feature_names.extend(
                        [f"ml_score_{i}" for i in range(pattern_scores.shape[1])]
                    )
                    self.logger.info(f"➕ ML 예측 점수 추가: {pattern_scores.shape}")

                # 추가 특성들 결합
                additional_features = all_results.get("additional_features", {})

                for feature_name, feature_data in additional_features.items():
                    if (
                        feature_data is not None
                        and feature_data.shape[0] == X_base.shape[0]
                    ):
                        feature_matrices.append(feature_data)
                        feature_names.extend(
                            [
                                f"{feature_name}_{i}"
                                for i in range(feature_data.shape[1])
                            ]
                        )
                        self.logger.info(
                            f"➕ {feature_name} 추가: {feature_data.shape}"
                        )

                # 모든 특성 결합
                X_combined = np.concatenate(feature_matrices, axis=1)

                # 데이터 검증
                if np.any(np.isnan(X_combined)) or np.any(np.isinf(X_combined)):
                    self.logger.warning(
                        "⚠️ 데이터에 NaN 또는 Inf 값이 있습니다. 0으로 대체합니다."
                    )
                    X_combined = np.nan_to_num(
                        X_combined, nan=0.0, posinf=0.0, neginf=0.0
                    )

                # 스케일링
                X_scaled = self.scaler.fit_transform(X_combined)

                # AutoEncoder 입력 차원 업데이트
                self.config.input_dim = X_scaled.shape[1]
                self.autoencoder_model.config["autoencoder"]["input_dim"] = (
                    X_scaled.shape[1]
                )

                self.execution_stats["preprocessing_time"] = time.time() - start_time
                self.logger.info(
                    f"✅ AutoEncoder 데이터 준비 완료: {X_scaled.shape} ({self.execution_stats['preprocessing_time']:.2f}초)"
                )

                return X_scaled, feature_names

            except Exception as e:
                self.execution_stats["error_count"] += 1
                self.logger.error(f"❌ AutoEncoder 데이터 준비 실패: {e}")
                raise

    def train_autoencoder_model(self, X: np.ndarray) -> Dict[str, Any]:
        """
        AutoEncoder 모델 학습

        Args:
            X: 학습 데이터

        Returns:
            학습 결과 딕셔너리
        """
        self.logger.info("🚀 AutoEncoder 모델 학습 시작...")

        with self.performance_monitor.track(
            "train_autoencoder_model", track_gpu=self.config.use_gpu
        ):
            start_time = time.time()

            try:
                # 데이터 분할
                from sklearn.model_selection import train_test_split

                X_train, X_val = train_test_split(
                    X, test_size=self.config.validation_split, random_state=42
                )

                self.logger.info(
                    f"📊 데이터 분할 완료: Train={X_train.shape}, Val={X_val.shape}"
                )

                # GPU 메모리 최적화
                if self.config.use_gpu and self.cuda_optimizer:
                    self.cuda_optimizer.clear_cache()

                # 모델 학습 (정상 패턴 학습)
                # y는 None으로 설정 (비지도 학습)
                train_result = self.autoencoder_model.fit(
                    X_train,
                    y=None,
                    validation_data=(X_val, None),
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    early_stopping_patience=self.config.early_stopping_patience,
                    verbose=True,
                )

                # 재구성 오류 통계 계산
                reconstruction_errors = self._compute_reconstruction_errors(X_train)

                # 임계값 계산 (Z-score 기반)
                mean_error = np.mean(reconstruction_errors)
                std_error = np.std(reconstruction_errors)
                threshold = mean_error + self.config.zscore_threshold * std_error

                # 통계 업데이트
                self.anomaly_stats.update(
                    {
                        "mean_reconstruction_error": float(mean_error),
                        "std_reconstruction_error": float(std_error),
                        "threshold_value": float(threshold),
                    }
                )

                training_time = time.time() - start_time
                self.execution_stats["training_time"] = training_time

                # 결과 정리
                training_results = {
                    "success": True,
                    "training_time": training_time,
                    "model_metadata": train_result,
                    "reconstruction_stats": {
                        "mean_error": float(mean_error),
                        "std_error": float(std_error),
                        "threshold": float(threshold),
                        "zscore_threshold": self.config.zscore_threshold,
                    },
                    "data_splits": {
                        "train_size": X_train.shape[0],
                        "val_size": X_val.shape[0],
                        "feature_count": X.shape[1],
                    },
                    "model_config": {
                        "input_dim": X.shape[1],
                        "latent_dim": self.config.latent_dim,
                        "hidden_dims": self.config.hidden_dims,
                    },
                    "gpu_used": self.config.use_gpu,
                }

                self.logger.info(f"✅ AutoEncoder 모델 학습 완료")
                self.logger.info(
                    f"📈 재구성 오류 통계: 평균={mean_error:.4f}, 표준편차={std_error:.4f}"
                )
                self.logger.info(f"🎯 이상치 임계값: {threshold:.4f}")
                self.logger.info(f"⏱️ 학습 시간: {training_time:.2f}초")

                return training_results

            except Exception as e:
                self.execution_stats["error_count"] += 1
                self.logger.error(f"❌ AutoEncoder 모델 학습 실패: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "training_time": time.time() - start_time,
                }

    def _compute_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        재구성 오류 계산

        Args:
            X: 입력 데이터

        Returns:
            재구성 오류 배열
        """
        try:
            # 배치 단위로 재구성 오류 계산
            batch_size = self.config.batch_size
            errors = []

            for i in range(0, X.shape[0], batch_size):
                batch_end = min(i + batch_size, X.shape[0])
                batch_X = X[i:batch_end]

                # 재구성
                reconstructed = self.autoencoder_model.predict(batch_X)

                # MSE 계산
                batch_errors = np.mean((batch_X - reconstructed) ** 2, axis=1)
                errors.extend(batch_errors)

            return np.array(errors)

        except Exception as e:
            self.logger.error(f"❌ 재구성 오류 계산 실패: {e}")
            return np.zeros(X.shape[0])

    def detect_anomalies(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        이상치 감지

        Args:
            feature_vectors: 특성 벡터

        Returns:
            이상치 점수 배열
        """
        self.logger.info("🔍 이상치 감지 시작...")

        with self.performance_monitor.track(
            "detect_anomalies", track_gpu=self.config.use_gpu
        ):
            start_time = time.time()

            try:
                if not self.autoencoder_model.is_trained:
                    raise ValueError("AutoEncoder 모델이 학습되지 않았습니다.")

                # 데이터 스케일링
                X_scaled = self.scaler.transform(feature_vectors)

                # 재구성 오류 계산
                reconstruction_errors = self._compute_reconstruction_errors(X_scaled)

                # 이상치 점수 정규화 (0~1 범위)
                min_error = np.min(reconstruction_errors)
                max_error = np.max(reconstruction_errors)

                if max_error > min_error:
                    anomaly_scores = (reconstruction_errors - min_error) / (
                        max_error - min_error
                    )
                else:
                    anomaly_scores = np.zeros_like(reconstruction_errors)

                # 이상치 통계 업데이트
                threshold = self.anomaly_stats["threshold_value"]
                anomaly_count = np.sum(reconstruction_errors > threshold)

                self.anomaly_stats.update(
                    {
                        "total_samples": len(reconstruction_errors),
                        "anomaly_count": int(anomaly_count),
                        "anomaly_rate": float(
                            anomaly_count / len(reconstruction_errors)
                        ),
                    }
                )

                detection_time = time.time() - start_time
                self.execution_stats["detection_time"] = detection_time

                self.logger.info(
                    f"✅ 이상치 감지 완료: {anomaly_count}/{len(reconstruction_errors)} ({self.anomaly_stats['anomaly_rate']:.1%})"
                )
                self.logger.info(f"⏱️ 감지 시간: {detection_time:.2f}초")

                return anomaly_scores

            except Exception as e:
                self.execution_stats["error_count"] += 1
                self.logger.error(f"❌ 이상치 감지 실패: {e}")
                raise

    def apply_anomaly_penalties(
        self, base_scores: np.ndarray, anomaly_scores: np.ndarray
    ) -> np.ndarray:
        """
        이상치 점수에 따른 감점 적용

        Args:
            base_scores: 기본 점수
            anomaly_scores: 이상치 점수

        Returns:
            감점 적용된 점수
        """
        self.logger.info("⚖️ 이상치 감점 적용 시작...")

        try:
            # 감점 계수 계산 (이상치 점수에 비례)
            penalty_factors = anomaly_scores * self.config.anomaly_penalty_factor

            # 감점 범위 제한
            penalty_factors = np.clip(
                penalty_factors, self.config.min_penalty, self.config.max_penalty
            )

            # 감점 적용
            penalized_scores = base_scores * (1 - penalty_factors)

            # 감점 통계
            penalty_applied = np.sum(penalty_factors > 0)
            avg_penalty = (
                np.mean(penalty_factors[penalty_factors > 0])
                if penalty_applied > 0
                else 0
            )

            self.anomaly_stats["penalty_applied"] = int(penalty_applied)

            self.logger.info(
                f"✅ 감점 적용 완료: {penalty_applied}/{len(base_scores)} 샘플에 감점"
            )
            self.logger.info(f"📊 평균 감점률: {avg_penalty:.1%}")

            return penalized_scores

        except Exception as e:
            self.logger.error(f"❌ 감점 적용 실패: {e}")
            return base_scores

    def save_anomaly_results(self, results: Dict[str, Any]) -> str:
        """
        이상감지 결과 저장

        Args:
            results: 이상감지 결과 딕셔너리

        Returns:
            저장된 파일 경로
        """
        self.logger.info("💾 이상감지 결과 저장 시작...")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 메인 결과 파일
            result_file = self.output_dir / f"anomaly_detection_{timestamp}.json"

            # 저장할 데이터 준비
            save_data = {
                "timestamp": timestamp,
                "config": asdict(self.config),
                "execution_stats": self.execution_stats,
                "anomaly_stats": self.anomaly_stats,
                "results": results,
                "model_info": {
                    "model_type": "AutoEncoder",
                    "is_trained": self.autoencoder_model.is_trained,
                    "input_dim": self.autoencoder_model.input_dim,
                    "latent_dim": self.config.latent_dim,
                    "hidden_dims": self.config.hidden_dims,
                },
            }

            # JSON 파일 저장
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

            # 모델 저장
            model_file = self.model_save_dir / f"autoencoder_model_{timestamp}.pt"
            model_saved = self.autoencoder_model.save(str(model_file))

            if model_saved:
                self.logger.info(f"💾 모델 저장 완료: {model_file}")
                save_data["model_file"] = str(model_file)

            # 이상치 점수만 별도 저장 (NumPy 형식)
            if "anomaly_scores" in results:
                scores_file = self.output_dir / f"anomaly_scores_{timestamp}.npy"
                np.save(scores_file, results["anomaly_scores"])
                self.logger.info(f"💾 이상치 점수 저장 완료: {scores_file}")

            # 감점 적용된 점수 저장
            if "penalized_scores" in results:
                penalized_file = self.output_dir / f"penalized_scores_{timestamp}.npy"
                np.save(penalized_file, results["penalized_scores"])
                self.logger.info(f"💾 감점 점수 저장 완료: {penalized_file}")

            self.logger.info(f"✅ 이상감지 결과 저장 완료: {result_file}")

            return str(result_file)

        except Exception as e:
            self.execution_stats["error_count"] += 1
            self.logger.error(f"❌ 이상감지 결과 저장 실패: {e}")
            raise

    def run_full_anomaly_detection(self) -> Dict[str, Any]:
        """
        전체 이상감지 파이프라인 실행

        Returns:
            실행 결과 딕셔너리
        """
        self.logger.info("🚀 전체 이상감지 파이프라인 시작")
        self.logger.info("=" * 80)

        # 실행 시작 시간
        self.execution_stats["start_time"] = datetime.now()

        try:
            # 1. 이전 결과 로드
            self.logger.info("📊 Step 1: 이전 단계 결과 로드")
            previous_results = self.load_previous_results()

            # 2. AutoEncoder 학습 데이터 준비
            self.logger.info("🔄 Step 2: AutoEncoder 학습 데이터 준비")
            X, feature_names = self.prepare_autoencoder_data(previous_results)

            # 3. AutoEncoder 모델 학습
            self.logger.info("🚀 Step 3: AutoEncoder 모델 학습")
            training_results = self.train_autoencoder_model(X)

            if not training_results["success"]:
                raise RuntimeError(
                    f"AutoEncoder 학습 실패: {training_results.get('error', 'Unknown error')}"
                )

            # 4. 이상치 감지
            self.logger.info("🔍 Step 4: 이상치 감지")
            anomaly_scores = self.detect_anomalies(X)

            # 5. 감점 적용
            self.logger.info("⚖️ Step 5: 감점 적용")
            base_scores = previous_results["pattern_scores"]
            if base_scores is None:
                # 기본 점수 생성
                base_scores = np.random.rand(len(anomaly_scores)) * 0.5 + 0.5
                self.logger.warning("⚠️ 기본 점수가 없어 임시 점수를 생성했습니다.")

            # 샘플 수 맞춤
            if len(base_scores) != len(anomaly_scores):
                if len(base_scores) > len(anomaly_scores):
                    base_scores = base_scores[: len(anomaly_scores)]
                else:
                    repeat_count = (len(anomaly_scores) + len(base_scores) - 1) // len(
                        base_scores
                    )
                    base_scores = np.tile(base_scores, repeat_count)[
                        : len(anomaly_scores)
                    ]

            penalized_scores = self.apply_anomaly_penalties(base_scores, anomaly_scores)

            # 6. 결과 정리
            detection_results = {
                "anomaly_scores": anomaly_scores,
                "penalized_scores": penalized_scores,
                "base_scores": base_scores,
                "training_results": training_results,
                "feature_names": feature_names,
                "previous_results_metadata": {
                    "source_files": previous_results["source_files"],
                    "data_loading_time": previous_results["load_time"],
                },
            }

            # 7. 결과 저장
            self.logger.info("💾 Step 6: 이상감지 결과 저장")
            result_file = self.save_anomaly_results(detection_results)

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
                "anomaly_stats": self.anomaly_stats,
                "performance_stats": performance_stats,
                "detection_summary": {
                    "total_samples": self.anomaly_stats["total_samples"],
                    "anomaly_count": self.anomaly_stats["anomaly_count"],
                    "anomaly_rate": self.anomaly_stats["anomaly_rate"],
                    "penalty_applied": self.anomaly_stats["penalty_applied"],
                    "mean_penalty_reduction": float(
                        np.mean(base_scores - penalized_scores)
                    ),
                },
                "model_performance": training_results["reconstruction_stats"],
                "warnings": self.execution_stats["warnings"],
            }

            self.logger.info("=" * 80)
            self.logger.info("✅ 이상감지 파이프라인 완료")
            self.logger.info(
                f"⏱️ 총 실행 시간: {self.execution_stats['total_time']:.2f}초"
            )
            self.logger.info(
                f"🔍 이상치 감지: {self.anomaly_stats['anomaly_count']}/{self.anomaly_stats['total_samples']} ({self.anomaly_stats['anomaly_rate']:.1%})"
            )
            self.logger.info(
                f"⚖️ 감점 적용: {self.anomaly_stats['penalty_applied']}개 샘플"
            )
            self.logger.info(
                f"📊 평균 점수 감소: {final_results['detection_summary']['mean_penalty_reduction']:.4f}"
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
            self.logger.error(f"❌ 이상감지 파이프라인 실패: {e}")
            self.logger.error(
                f"⏱️ 실행 시간: {self.execution_stats['total_time']:.2f}초"
            )
            self.logger.error(f"🚨 오류 개수: {self.execution_stats['error_count']}")
            self.logger.error("=" * 80)

            return {
                "success": False,
                "error": str(e),
                "execution_stats": self.execution_stats,
                "anomaly_stats": self.anomaly_stats,
                "warnings": self.execution_stats["warnings"],
            }


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="DAEBAK AI 로또 시스템 - 이상감지 단계 (Phase 3)"
    )

    parser.add_argument("--config", type=str, help="설정 파일 경로 (JSON 형식)")

    parser.add_argument("--no-gpu", action="store_true", help="GPU 사용 비활성화")

    parser.add_argument(
        "--latent-dim", type=int, default=32, help="AutoEncoder 잠재 차원"
    )

    parser.add_argument("--batch-size", type=int, default=256, help="배치 크기")

    parser.add_argument("--epochs", type=int, default=100, help="학습 에포크 수")

    parser.add_argument(
        "--zscore-threshold", type=float, default=2.5, help="Z-score 임계값"
    )

    parser.add_argument(
        "--penalty-factor", type=float, default=0.3, help="이상치 감점 계수"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/result/anomaly_detection",
        help="출력 디렉토리",
    )

    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")

    return parser.parse_args()


def main():
    """메인 실행 함수"""
    # 1. 의존성 설정
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
    if args.latent_dim:
        cli_config["latent_dim"] = args.latent_dim
    if args.batch_size:
        cli_config["batch_size"] = args.batch_size
    if args.epochs:
        cli_config["epochs"] = args.epochs
    if args.zscore_threshold:
        cli_config["zscore_threshold"] = args.zscore_threshold
    if args.penalty_factor:
        cli_config["anomaly_penalty_factor"] = args.penalty_factor
    if args.output_dir:
        cli_config["output_dir"] = args.output_dir

    # 로거 설정
    logger = get_logger(__name__)
    if args.verbose:
        logger.setLevel("DEBUG")

    try:
        # 3. 이상감지 엔진 생성 및 실행
        engine = AnomalyDetectionEngine(cli_config)
        final_results = engine.run_full_anomaly_detection()

        # 4. 결과 요약 출력
        if final_results["success"]:
            print(f"\n✅ 이상감지 완료!")
            print(f"📁 결과 파일: {final_results['result_file']}")
            print(f"⏱️ 실행 시간: {final_results['execution_stats']['total_time']:.2f}초")
            print(f"🔍 이상치 감지율: {final_results['anomaly_stats']['anomaly_rate']:.1%}")
            return 0
        else:
            print(f"\n❌ 이상감지 실패: {final_results['error']}")
            return 1

    except Exception as e:
        logger.error(f"❌ 실행 중 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
