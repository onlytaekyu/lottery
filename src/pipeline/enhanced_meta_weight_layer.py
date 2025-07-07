"""
Enhanced Meta Weight Layer
강화된 메타 가중치 레이어 - 동적 가중치 학습, 신뢰도 기반 앙상블, ROI 조정
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

from ..utils.unified_logging import get_logger
from ..utils.memory_manager import MemoryManager
from ..utils.cuda_singleton_manager import CudaSingletonManager

logger = get_logger(__name__)


@dataclass
class MetaWeightConfig:
    """메타 가중치 설정"""

    num_models: int = 4  # LightGBM, AutoEncoder, TCN, RandomForest
    adaptation_rate: float = 0.01
    momentum: float = 0.9
    confidence_threshold: float = 0.7
    roi_weight: float = 0.3
    diversity_weight: float = 0.2
    performance_weight: float = 0.5
    device: str = "cuda"


class DynamicWeightLearner(nn.Module):
    """동적 가중치 학습기"""

    def __init__(self, config: MetaWeightConfig):
        super().__init__()
        self.config = config
        self.num_models = config.num_models
        self.adaptation_rate = config.adaptation_rate
        self.momentum = config.momentum

        # 가중치 네트워크
        self.weight_network = nn.Sequential(
            nn.Linear(self.num_models * 3, 64),  # 성능, 신뢰도, 다양성 메트릭
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.num_models),
            nn.Softmax(dim=-1),
        )

        # 이동 평균 가중치
        self.register_buffer(
            "moving_weights", torch.ones(self.num_models) / self.num_models
        )

    def forward(self, model_metrics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_metrics: (batch_size, num_models, 3) - 성능, 신뢰도, 다양성
        Returns:
            weights: (batch_size, num_models) - 동적 가중치
        """

        # 메트릭 평탄화
        flattened_metrics = model_metrics.view(model_metrics.size(0), -1)

        # 동적 가중치 계산
        dynamic_weights = self.weight_network(flattened_metrics)

        # 이동 평균 업데이트
        batch_mean_weights = dynamic_weights.mean(dim=0)
        self.moving_weights = (
            self.momentum * self.moving_weights
            + (1 - self.momentum) * batch_mean_weights
        )

        return dynamic_weights

    def get_stable_weights(self) -> torch.Tensor:
        """안정화된 가중치 반환"""
        return self.moving_weights.clone()


class ConfidenceEstimator:
    """신뢰도 추정기"""

    def __init__(self, uncertainty_method: str = "monte_carlo_dropout"):
        self.uncertainty_method = uncertainty_method

    def estimate_confidence(
        self,
        model_predictions: Dict[str, np.ndarray],
        model_objects: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """
        모델별 신뢰도 추정

        Args:
            model_predictions: 모델별 예측 결과
            model_objects: 모델 객체들 (MC Dropout용)

        Returns:
            Dict[str, float]: 모델별 신뢰도 점수
        """

        confidence_scores = {}

        for model_name, predictions in model_predictions.items():
            if self.uncertainty_method == "monte_carlo_dropout":
                confidence = self._monte_carlo_confidence(
                    predictions, model_objects.get(model_name)
                )
            elif self.uncertainty_method == "prediction_variance":
                confidence = self._prediction_variance_confidence(predictions)
            elif self.uncertainty_method == "entropy_based":
                confidence = self._entropy_based_confidence(predictions)
            else:
                confidence = 0.5  # 기본값

            confidence_scores[model_name] = confidence

        return confidence_scores

    def _monte_carlo_confidence(self, predictions: np.ndarray, model: Any) -> float:
        """Monte Carlo Dropout 기반 신뢰도"""

        if model is None:
            return 0.5

        try:
            # 다중 샘플링 (모델이 dropout을 지원하는 경우)
            if hasattr(model, "enable_dropout"):
                model.enable_dropout()
                samples = []
                for _ in range(10):  # 10회 샘플링
                    sample_pred = model.predict(predictions)
                    samples.append(sample_pred)

                # 예측 분산 계산
                samples = np.array(samples)
                prediction_variance = np.var(samples, axis=0)
                confidence = 1.0 / (1.0 + np.mean(prediction_variance))

                model.disable_dropout()
            else:
                # 예측 확률의 최대값을 신뢰도로 사용
                if len(predictions.shape) > 1:
                    confidence = np.mean(np.max(predictions, axis=1))
                else:
                    confidence = 0.7  # 기본값

        except Exception as e:
            logger.warning(f"Monte Carlo 신뢰도 계산 중 오류: {e}")
            confidence = 0.5

        return float(confidence)

    def _prediction_variance_confidence(self, predictions: np.ndarray) -> float:
        """예측 분산 기반 신뢰도"""

        if len(predictions.shape) > 1:
            # 다중 클래스 예측의 경우
            prediction_variance = np.var(predictions, axis=1)
            confidence = 1.0 / (1.0 + np.mean(prediction_variance))
        else:
            # 단일 예측의 경우
            confidence = 1.0 / (1.0 + np.var(predictions))

        return float(confidence)

    def _entropy_based_confidence(self, predictions: np.ndarray) -> float:
        """엔트로피 기반 신뢰도"""

        if len(predictions.shape) > 1:
            # 확률 분포의 엔트로피 계산
            entropy = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
            max_entropy = np.log(predictions.shape[1])
            confidence = 1.0 - np.mean(entropy) / max_entropy
        else:
            # 이진 분류의 경우
            prob = np.clip(predictions, 0.01, 0.99)
            entropy = -prob * np.log(prob) - (1 - prob) * np.log(1 - prob)
            confidence = 1.0 - np.mean(entropy) / np.log(2)

        return float(confidence)


class ROIBasedWeightAdjuster:
    """ROI 기반 가중치 조정기"""

    def __init__(self, target_metric: str = "profit_ratio"):
        self.target_metric = target_metric
        self.roi_history = {}
        self.performance_history = {}

    def calculate_roi_weights(
        self,
        model_predictions: Dict[str, np.ndarray],
        actual_results: Optional[np.ndarray] = None,
        historical_roi: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, float]:
        """
        ROI 기반 가중치 계산

        Args:
            model_predictions: 모델별 예측 결과
            actual_results: 실제 결과 (선택사항)
            historical_roi: 모델별 과거 ROI 데이터

        Returns:
            Dict[str, float]: 모델별 ROI 가중치
        """

        roi_weights = {}

        # 과거 ROI 데이터 업데이트
        if historical_roi:
            self.roi_history.update(historical_roi)

        # 각 모델의 ROI 성능 평가
        for model_name, predictions in model_predictions.items():
            if model_name in self.roi_history and self.roi_history[model_name]:
                # 과거 ROI 평균 계산
                avg_roi = np.mean(self.roi_history[model_name][-10:])  # 최근 10회
                roi_weights[model_name] = max(0.1, avg_roi)  # 최소 0.1 보장
            else:
                # 기본 가중치
                roi_weights[model_name] = 0.5

        # 가중치 정규화
        total_weight = sum(roi_weights.values())
        if total_weight > 0:
            roi_weights = {k: v / total_weight for k, v in roi_weights.items()}

        return roi_weights

    def update_roi_history(self, model_name: str, roi_value: float):
        """ROI 히스토리 업데이트"""

        if model_name not in self.roi_history:
            self.roi_history[model_name] = []

        self.roi_history[model_name].append(roi_value)

        # 히스토리 크기 제한 (최대 50개)
        if len(self.roi_history[model_name]) > 50:
            self.roi_history[model_name] = self.roi_history[model_name][-50:]

    def get_roi_analysis(self) -> Dict[str, Any]:
        """ROI 분석 결과 반환"""

        analysis = {}

        for model_name, roi_values in self.roi_history.items():
            if roi_values:
                analysis[model_name] = {
                    "mean_roi": np.mean(roi_values),
                    "std_roi": np.std(roi_values),
                    "max_roi": np.max(roi_values),
                    "min_roi": np.min(roi_values),
                    "recent_trend": (
                        np.mean(roi_values[-5:])
                        if len(roi_values) >= 5
                        else np.mean(roi_values)
                    ),
                    "stability": 1.0 / (1.0 + np.std(roi_values)),
                }

        return analysis


class EnhancedMetaWeightLayer:
    """강화된 메타 가중치 레이어"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.memory_manager = MemoryManager()
        self.cuda_manager = CudaSingletonManager()

        # 디바이스 설정
        self.device = torch.device(
            "cuda" if self.cuda_manager.is_available() else "cpu"
        )

        # 메타 가중치 설정
        self.meta_config = MetaWeightConfig(
            num_models=config.get("num_models", 4),
            adaptation_rate=config.get("adaptation_rate", 0.01),
            momentum=config.get("momentum", 0.9),
            confidence_threshold=config.get("confidence_threshold", 0.7),
            roi_weight=config.get("roi_weight", 0.3),
            diversity_weight=config.get("diversity_weight", 0.2),
            performance_weight=config.get("performance_weight", 0.5),
            device=str(self.device),
        )

        # 컴포넌트 초기화
        self.dynamic_weights = DynamicWeightLearner(self.meta_config).to(self.device)
        self.confidence_estimator = ConfidenceEstimator(
            uncertainty_method=config.get("uncertainty_method", "monte_carlo_dropout")
        )
        self.roi_adjuster = ROIBasedWeightAdjuster(
            target_metric=config.get("target_metric", "profit_ratio")
        )

        # 옵티마이저
        self.optimizer = optim.Adam(
            self.dynamic_weights.parameters(), lr=self.meta_config.adaptation_rate
        )

        # 성능 메트릭 저장
        self.performance_history = []
        self.model_names = ["lightgbm", "autoencoder", "tcn", "randomforest"]

    def compute_ensemble_prediction(
        self,
        model_predictions: Dict[str, np.ndarray],
        model_objects: Optional[Dict[str, Any]] = None,
        historical_roi: Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        앙상블 예측 계산

        Args:
            model_predictions: 모델별 예측 결과
            model_objects: 모델 객체들 (신뢰도 계산용)
            historical_roi: 모델별 과거 ROI 데이터

        Returns:
            Tuple[np.ndarray, Dict]: 앙상블 예측과 메타데이터
        """

        try:
            self.logger.info("앙상블 예측 계산 시작")

            with self.memory_manager.get_context("meta_weight_ensemble"):
                # 1. 신뢰도 추정
                confidence_scores = self.confidence_estimator.estimate_confidence(
                    model_predictions, model_objects
                )

                # 2. ROI 기반 가중치 계산
                roi_weights = self.roi_adjuster.calculate_roi_weights(
                    model_predictions, historical_roi=historical_roi
                )

                # 3. 다양성 점수 계산
                diversity_scores = self._calculate_diversity_scores(model_predictions)

                # 4. 성능 점수 계산
                performance_scores = self._calculate_performance_scores(
                    model_predictions
                )

                # 5. 동적 가중치 계산
                dynamic_weights = self._compute_dynamic_weights(
                    performance_scores, confidence_scores, diversity_scores
                )

                # 6. 최종 앙상블 예측
                ensemble_prediction = self._weighted_ensemble_prediction(
                    model_predictions, dynamic_weights
                )

                # 메타데이터 생성
                metadata = {
                    "dynamic_weights": dynamic_weights,
                    "confidence_scores": confidence_scores,
                    "roi_weights": roi_weights,
                    "diversity_scores": diversity_scores,
                    "performance_scores": performance_scores,
                    "ensemble_confidence": np.mean(list(confidence_scores.values())),
                    "weight_entropy": self._calculate_weight_entropy(dynamic_weights),
                }

                self.logger.info("앙상블 예측 계산 완료")

                return ensemble_prediction, metadata

        except Exception as e:
            self.logger.error(f"앙상블 예측 계산 중 오류: {e}")
            raise

    def _calculate_diversity_scores(
        self, model_predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """모델 다양성 점수 계산"""

        diversity_scores = {}
        predictions_array = np.array(list(model_predictions.values()))

        for i, model_name in enumerate(model_predictions.keys()):
            # 다른 모델들과의 상관관계 계산
            other_predictions = np.delete(predictions_array, i, axis=0)
            correlations = []

            for other_pred in other_predictions:
                if len(model_predictions[model_name].shape) > 1:
                    # 다중 클래스 예측
                    corr = np.corrcoef(
                        model_predictions[model_name].flatten(), other_pred.flatten()
                    )[0, 1]
                else:
                    # 단일 예측
                    corr = np.corrcoef(model_predictions[model_name], other_pred)[0, 1]

                if not np.isnan(corr):
                    correlations.append(abs(corr))

            # 다양성 점수 = 1 - 평균 상관관계
            diversity_scores[model_name] = (
                1.0 - np.mean(correlations) if correlations else 0.5
            )

        return diversity_scores

    def _calculate_performance_scores(
        self, model_predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """모델 성능 점수 계산 (과거 성능 기반)"""

        performance_scores = {}

        # 과거 성능 히스토리가 있는 경우 사용
        if self.performance_history:
            latest_performance = self.performance_history[-1]
            for model_name in model_predictions.keys():
                performance_scores[model_name] = latest_performance.get(model_name, 0.5)
        else:
            # 기본 성능 점수
            for model_name in model_predictions.keys():
                performance_scores[model_name] = 0.5

        return performance_scores

    def _compute_dynamic_weights(
        self,
        performance_scores: Dict[str, float],
        confidence_scores: Dict[str, float],
        diversity_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """동적 가중치 계산"""

        # 메트릭 텐서 생성
        model_metrics = []
        for model_name in self.model_names:
            if model_name in performance_scores:
                metrics = [
                    performance_scores.get(model_name, 0.5),
                    confidence_scores.get(model_name, 0.5),
                    diversity_scores.get(model_name, 0.5),
                ]
                model_metrics.append(metrics)
            else:
                model_metrics.append([0.5, 0.5, 0.5])

        model_metrics = torch.FloatTensor([model_metrics]).to(self.device)

        # 동적 가중치 계산
        self.dynamic_weights.eval()
        with torch.no_grad():
            weights_tensor = self.dynamic_weights(model_metrics)
            weights_array = weights_tensor.cpu().numpy().flatten()

        # 딕셔너리 형태로 변환
        dynamic_weights = {}
        for i, model_name in enumerate(self.model_names):
            if i < len(weights_array):
                dynamic_weights[model_name] = float(weights_array[i])
            else:
                dynamic_weights[model_name] = 0.25  # 기본값

        return dynamic_weights

    def _weighted_ensemble_prediction(
        self, model_predictions: Dict[str, np.ndarray], weights: Dict[str, float]
    ) -> np.ndarray:
        """가중 앙상블 예측"""

        ensemble_prediction = None
        total_weight = 0.0

        for model_name, predictions in model_predictions.items():
            weight = weights.get(model_name, 0.0)

            if weight > 0:
                if ensemble_prediction is None:
                    ensemble_prediction = weight * predictions
                else:
                    ensemble_prediction += weight * predictions
                total_weight += weight

        # 가중치 정규화
        if total_weight > 0:
            ensemble_prediction /= total_weight

        return ensemble_prediction

    def _calculate_weight_entropy(self, weights: Dict[str, float]) -> float:
        """가중치 엔트로피 계산 (다양성 측정)"""

        weight_values = np.array(list(weights.values()))
        weight_values = weight_values / np.sum(weight_values)  # 정규화

        # 엔트로피 계산
        entropy = -np.sum(weight_values * np.log(weight_values + 1e-8))
        max_entropy = np.log(len(weight_values))

        return entropy / max_entropy  # 정규화된 엔트로피

    def update_performance_history(self, model_performance: Dict[str, float]):
        """성능 히스토리 업데이트"""

        self.performance_history.append(model_performance.copy())

        # 히스토리 크기 제한
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def train_dynamic_weights(
        self,
        model_predictions: Dict[str, np.ndarray],
        actual_results: np.ndarray,
        epochs: int = 10,
    ):
        """동적 가중치 학습"""

        self.logger.info("동적 가중치 학습 시작")

        # 학습 데이터 준비
        performance_scores = {}
        for model_name, predictions in model_predictions.items():
            if len(predictions.shape) > 1:
                # 다중 클래스
                pred_labels = np.argmax(predictions, axis=1)
            else:
                # 이진 분류
                pred_labels = (predictions > 0.5).astype(int)

            accuracy = accuracy_score(actual_results, pred_labels)
            performance_scores[model_name] = accuracy

        # 성능 히스토리 업데이트
        self.update_performance_history(performance_scores)

        # 신뢰도 및 다양성 점수 계산
        confidence_scores = self.confidence_estimator.estimate_confidence(
            model_predictions
        )
        diversity_scores = self._calculate_diversity_scores(model_predictions)

        # 학습 루프
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # 메트릭 텐서 생성
            model_metrics = []
            for model_name in self.model_names:
                metrics = [
                    performance_scores.get(model_name, 0.5),
                    confidence_scores.get(model_name, 0.5),
                    diversity_scores.get(model_name, 0.5),
                ]
                model_metrics.append(metrics)

            model_metrics = torch.FloatTensor([model_metrics]).to(self.device)

            # 가중치 예측
            predicted_weights = self.dynamic_weights(model_metrics)

            # 손실 계산 (성능 기반)
            target_weights = torch.FloatTensor([list(performance_scores.values())]).to(
                self.device
            )
            target_weights = torch.softmax(target_weights, dim=-1)

            loss = nn.functional.kl_div(
                torch.log(predicted_weights + 1e-8),
                target_weights,
                reduction="batchmean",
            )

            # 역전파
            loss.backward()
            self.optimizer.step()

            if epoch % 5 == 0:
                self.logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        self.logger.info("동적 가중치 학습 완료")

    def get_weight_analysis(self) -> Dict[str, Any]:
        """가중치 분석 결과 반환"""

        # 현재 안정화된 가중치
        stable_weights = self.dynamic_weights.get_stable_weights().cpu().numpy()

        # ROI 분석
        roi_analysis = self.roi_adjuster.get_roi_analysis()

        analysis = {
            "stable_weights": {
                model_name: float(weight)
                for model_name, weight in zip(self.model_names, stable_weights)
            },
            "roi_analysis": roi_analysis,
            "performance_trend": (
                self.performance_history[-10:]
                if len(self.performance_history) >= 10
                else self.performance_history
            ),
            "weight_stability": float(1.0 / (1.0 + np.std(stable_weights))),
            "ensemble_diversity": float(
                self._calculate_weight_entropy({"weights": stable_weights})
            ),
        }

        return analysis

    def print_optimization_summary(self, metadata: Dict[str, Any]):
        """최적화 결과 요약 출력"""

        print("=" * 60)
        print("⚖️  Meta Weight Layer 최적화 결과")
        print("=" * 60)

        print(f"📊 동적 가중치:")
        for model_name, weight in metadata["dynamic_weights"].items():
            print(f"  • {model_name}: {weight:.4f}")

        print(f"\n📈 신뢰도 점수:")
        for model_name, confidence in metadata["confidence_scores"].items():
            print(f"  • {model_name}: {confidence:.4f}")

        print(f"\n🔧 적용된 최적화 기법:")
        print(f"  • 동적 가중치 학습: 적응률 {self.meta_config.adaptation_rate}")
        print(f"  • 신뢰도 기반 앙상블: {self.confidence_estimator.uncertainty_method}")
        print(f"  • ROI 기반 조정: {self.roi_adjuster.target_metric}")
        print(f"  • GPU 가속: {'활성화' if self.device.type == 'cuda' else '비활성화'}")

        print(f"\n📊 앙상블 메트릭:")
        print(f"  • 앙상블 신뢰도: {metadata['ensemble_confidence']:.4f}")
        print(f"  • 가중치 엔트로피: {metadata['weight_entropy']:.4f}")

        print("=" * 60)
