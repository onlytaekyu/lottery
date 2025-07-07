"""
적응형 가중치 시스템 (Adaptive Weight System)

성과 기반 가중치 적응과 메타 러닝을 통한 동적 전략 가중치 조정 시스템입니다.

주요 기능:
- 실시간 성과 기반 가중치 조정
- 메타 러닝 기반 가중치 최적화
- 다중 목표 최적화 (ROI, 적중률, 리스크)
- 시간 가중 성과 평가
- 적응형 학습률 조정
- 가중치 안정성 보장
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import math
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

from ..shared.types import LotteryNumber, ModelPrediction
from ..utils.unified_logging import get_logger
from ..utils.cache_paths import get_cache_dir
from ..monitoring.performance_tracker import PerformanceMetrics

logger = get_logger(__name__)


@dataclass
class WeightUpdateConfig:
    """가중치 업데이트 설정"""

    learning_rate: float = 0.01  # 학습률
    momentum: float = 0.9  # 모멘텀
    decay_rate: float = 0.95  # 시간 감쇠율
    min_weight: float = 0.05  # 최소 가중치
    max_weight: float = 0.6  # 최대 가중치
    stability_threshold: float = 0.1  # 안정성 임계값
    adaptation_window: int = 20  # 적응 윈도우 크기
    performance_memory: int = 100  # 성능 메모리 크기
    multi_objective_weights: Dict[str, float] = None  # 다중 목표 가중치


@dataclass
class StrategyWeight:
    """전략 가중치 정보"""

    strategy: str
    current_weight: float
    target_weight: float
    momentum: float
    performance_score: float
    stability_score: float
    confidence: float
    last_update: datetime
    update_count: int


@dataclass
class WeightOptimizationResult:
    """가중치 최적화 결과"""

    optimized_weights: Dict[str, float]
    expected_performance: float
    optimization_score: float
    convergence_iterations: int
    stability_metrics: Dict[str, float]
    recommendations: List[str]


class MetaLearner:
    """메타 러닝 시스템"""

    def __init__(self, config: WeightUpdateConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # 메타 모델 초기화
        self.meta_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        # 학습 데이터 저장소
        self.training_data = []
        self.model_trained = False

        # 특성 엔지니어링
        self.feature_extractors = {
            "performance_trend": self._extract_performance_trend,
            "volatility_features": self._extract_volatility_features,
            "correlation_features": self._extract_correlation_features,
            "temporal_features": self._extract_temporal_features,
        }

    def extract_features(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> np.ndarray:
        """성능 이력에서 특성 추출"""
        try:
            features = []

            for extractor_name, extractor_func in self.feature_extractors.items():
                try:
                    feature_vector = extractor_func(performance_history)
                    features.extend(feature_vector)
                except Exception as e:
                    self.logger.warning(f"특성 추출 실패 ({extractor_name}): {str(e)}")
                    # 기본값으로 채움
                    features.extend([0.0] * 10)  # 기본 특성 크기

            return np.array(features)

        except Exception as e:
            self.logger.error(f"특성 추출 중 오류: {str(e)}")
            return np.zeros(40)  # 기본 특성 벡터

    def _extract_performance_trend(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> List[float]:
        """성능 트렌드 특성 추출"""
        features = []

        for strategy, metrics_list in performance_history.items():
            if len(metrics_list) >= 2:
                # ROI 트렌드
                roi_values = [m.roi for m in metrics_list[-10:]]
                roi_trend = (
                    np.polyfit(range(len(roi_values)), roi_values, 1)[0]
                    if len(roi_values) > 1
                    else 0
                )
                features.append(roi_trend)

                # 승률 트렌드
                win_rate_values = [m.win_rate for m in metrics_list[-10:]]
                win_rate_trend = (
                    np.polyfit(range(len(win_rate_values)), win_rate_values, 1)[0]
                    if len(win_rate_values) > 1
                    else 0
                )
                features.append(win_rate_trend)
            else:
                features.extend([0.0, 0.0])

        # 고정 크기로 패딩
        while len(features) < 10:
            features.append(0.0)

        return features[:10]

    def _extract_volatility_features(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> List[float]:
        """변동성 특성 추출"""
        features = []

        for strategy, metrics_list in performance_history.items():
            if len(metrics_list) >= 2:
                # ROI 변동성
                roi_values = [m.roi for m in metrics_list[-10:]]
                roi_volatility = np.std(roi_values) if len(roi_values) > 1 else 0
                features.append(roi_volatility)

                # 샤프 비율 평균
                sharpe_values = [m.sharpe_ratio for m in metrics_list[-10:]]
                avg_sharpe = np.mean(sharpe_values) if sharpe_values else 0
                features.append(avg_sharpe)
            else:
                features.extend([0.0, 0.0])

        # 고정 크기로 패딩
        while len(features) < 10:
            features.append(0.0)

        return features[:10]

    def _extract_correlation_features(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> List[float]:
        """상관관계 특성 추출"""
        features = []

        strategies = list(performance_history.keys())

        # 전략 간 ROI 상관관계
        if len(strategies) >= 2:
            roi_data = {}
            for strategy, metrics_list in performance_history.items():
                roi_data[strategy] = [m.roi for m in metrics_list[-10:]]

            # 상관관계 계산
            correlations = []
            for i in range(len(strategies)):
                for j in range(i + 1, len(strategies)):
                    strat1, strat2 = strategies[i], strategies[j]
                    if len(roi_data[strat1]) > 1 and len(roi_data[strat2]) > 1:
                        corr = np.corrcoef(roi_data[strat1], roi_data[strat2])[0, 1]
                        correlations.append(corr if not np.isnan(corr) else 0)

            # 평균 상관관계
            avg_correlation = np.mean(correlations) if correlations else 0
            features.append(avg_correlation)
        else:
            features.append(0.0)

        # 고정 크기로 패딩
        while len(features) < 10:
            features.append(0.0)

        return features[:10]

    def _extract_temporal_features(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> List[float]:
        """시간적 특성 추출"""
        features = []

        for strategy, metrics_list in performance_history.items():
            if metrics_list:
                # 최근 성과 vs 과거 성과
                recent_roi = (
                    np.mean([m.roi for m in metrics_list[-5:]])
                    if len(metrics_list) >= 5
                    else 0
                )
                past_roi = (
                    np.mean([m.roi for m in metrics_list[-10:-5]])
                    if len(metrics_list) >= 10
                    else recent_roi
                )

                performance_change = recent_roi - past_roi
                features.append(performance_change)

                # 성과 일관성
                roi_values = [m.roi for m in metrics_list[-10:]]
                consistency = 1 / (1 + np.std(roi_values)) if len(roi_values) > 1 else 0
                features.append(consistency)
            else:
                features.extend([0.0, 0.0])

        # 고정 크기로 패딩
        while len(features) < 10:
            features.append(0.0)

        return features[:10]

    def train_meta_model(self, training_data: List[Dict[str, Any]]) -> bool:
        """메타 모델 학습"""
        try:
            if len(training_data) < 10:
                self.logger.warning("학습 데이터가 부족합니다")
                return False

            # 특성과 타겟 분리
            X = []
            y = []

            for data in training_data:
                features = data.get("features", [])
                target = data.get("target_performance", 0)

                if features and target is not None:
                    X.append(features)
                    y.append(target)

            if len(X) < 10:
                self.logger.warning("유효한 학습 데이터가 부족합니다")
                return False

            X = np.array(X)
            y = np.array(y)

            # 모델 학습
            self.meta_model.fit(X, y)

            # 성능 평가
            y_pred = self.meta_model.predict(X)
            mse = mean_squared_error(y, y_pred)

            self.model_trained = True
            self.logger.info(f"메타 모델 학습 완료 - MSE: {mse:.4f}")

            return True

        except Exception as e:
            self.logger.error(f"메타 모델 학습 중 오류: {str(e)}")
            return False

    def predict_performance(self, features: np.ndarray) -> float:
        """성능 예측"""
        try:
            if not self.model_trained:
                return 0.0

            prediction = self.meta_model.predict(features.reshape(1, -1))[0]
            return float(prediction)

        except Exception as e:
            self.logger.error(f"성능 예측 중 오류: {str(e)}")
            return 0.0

    def get_feature_importance(self) -> Dict[str, float]:
        """특성 중요도 반환"""
        try:
            if not self.model_trained:
                return {}

            importance = self.meta_model.feature_importances_
            feature_names = [f"feature_{i}" for i in range(len(importance))]

            return dict(zip(feature_names, importance))

        except Exception as e:
            self.logger.error(f"특성 중요도 계산 중 오류: {str(e)}")
            return {}


class AdaptiveWeightSystem:
    """적응형 가중치 시스템"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        적응형 가중치 시스템 초기화

        Args:
            config: 설정 객체
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # 가중치 업데이트 설정
        weight_config = self.config.get("weight_update", {})
        multi_obj_weights = weight_config.get(
            "multi_objective_weights",
            {"roi": 0.4, "win_rate": 0.3, "stability": 0.2, "risk": 0.1},
        )

        self.update_config = WeightUpdateConfig(
            learning_rate=weight_config.get("learning_rate", 0.01),
            momentum=weight_config.get("momentum", 0.9),
            decay_rate=weight_config.get("decay_rate", 0.95),
            min_weight=weight_config.get("min_weight", 0.05),
            max_weight=weight_config.get("max_weight", 0.6),
            stability_threshold=weight_config.get("stability_threshold", 0.1),
            adaptation_window=weight_config.get("adaptation_window", 20),
            performance_memory=weight_config.get("performance_memory", 100),
            multi_objective_weights=multi_obj_weights,
        )

        # 메타 러닝 시스템
        self.meta_learner = MetaLearner(self.update_config)

        # 전략 가중치 관리
        self.strategy_weights = {}
        self.weight_history = defaultdict(deque)
        self.performance_history = defaultdict(deque)

        # 최적화 이력
        self.optimization_history = []

        # 데이터 저장 경로
        self.cache_dir = Path(get_cache_dir()) / "adaptive_weights"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 초기 가중치 설정
        self.initialize_weights()

        self.logger.info("적응형 가중치 시스템 초기화 완료")

    def initialize_weights(self, strategies: Optional[List[str]] = None) -> None:
        """가중치 초기화"""
        try:
            if strategies is None:
                strategies = [
                    "frequency_based",
                    "cluster_analysis",
                    "trend_following",
                    "ai_ensemble",
                    "contrarian",
                ]

            # 균등 가중치로 초기화
            equal_weight = 1.0 / len(strategies)

            for strategy in strategies:
                self.strategy_weights[strategy] = StrategyWeight(
                    strategy=strategy,
                    current_weight=equal_weight,
                    target_weight=equal_weight,
                    momentum=0.0,
                    performance_score=0.0,
                    stability_score=1.0,
                    confidence=0.5,
                    last_update=datetime.now(),
                    update_count=0,
                )

            self.logger.info(f"가중치 초기화 완료: {len(strategies)}개 전략")

        except Exception as e:
            self.logger.error(f"가중치 초기화 중 오류: {str(e)}")

    def update_weights(
        self, performance_metrics: Dict[str, PerformanceMetrics]
    ) -> Dict[str, float]:
        """성과 기반 가중치 업데이트"""
        try:
            self.logger.info("성과 기반 가중치 업데이트 시작")

            # 성과 점수 계산
            performance_scores = self._calculate_performance_scores(performance_metrics)

            # 타겟 가중치 계산
            target_weights = self._calculate_target_weights(performance_scores)

            # 가중치 업데이트 (모멘텀 적용)
            updated_weights = self._apply_momentum_update(target_weights)

            # 안정성 검사
            stable_weights = self._ensure_weight_stability(updated_weights)

            # 가중치 정규화
            normalized_weights = self._normalize_weights(stable_weights)

            # 가중치 객체 업데이트
            self._update_weight_objects(normalized_weights, performance_scores)

            # 이력 저장
            self._save_weight_history(normalized_weights)

            self.logger.info("성과 기반 가중치 업데이트 완료")
            return normalized_weights

        except Exception as e:
            self.logger.error(f"가중치 업데이트 중 오류: {str(e)}")
            return self.get_current_weights()

    def _calculate_performance_scores(
        self, performance_metrics: Dict[str, PerformanceMetrics]
    ) -> Dict[str, float]:
        """성과 점수 계산"""
        try:
            scores = {}

            for strategy, metrics in performance_metrics.items():
                # 다중 목표 점수 계산
                roi_score = self._normalize_score(metrics.roi, -1.0, 1.0)
                win_rate_score = metrics.win_rate
                stability_score = (
                    1.0 / (1.0 + metrics.volatility) if metrics.volatility > 0 else 1.0
                )
                risk_score = 1.0 - min(1.0, max(0.0, metrics.max_drawdown))

                # 가중 평균 계산
                weighted_score = (
                    self.update_config.multi_objective_weights["roi"] * roi_score
                    + self.update_config.multi_objective_weights["win_rate"]
                    * win_rate_score
                    + self.update_config.multi_objective_weights["stability"]
                    * stability_score
                    + self.update_config.multi_objective_weights["risk"] * risk_score
                )

                # 시간 가중 적용
                time_weight = self._calculate_time_weight(metrics.timestamp)
                final_score = weighted_score * time_weight

                scores[strategy] = max(0.0, min(1.0, final_score))

            return scores

        except Exception as e:
            self.logger.error(f"성과 점수 계산 중 오류: {str(e)}")
            return {}

    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """점수 정규화"""
        try:
            if max_val == min_val:
                return 0.5

            normalized = (value - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))

        except Exception as e:
            return 0.5

    def _calculate_time_weight(self, timestamp: datetime) -> float:
        """시간 가중치 계산"""
        try:
            now = datetime.now()
            time_diff = (now - timestamp).total_seconds() / 3600  # 시간 단위

            # 지수 감쇠
            time_weight = math.exp(-time_diff * 0.1)  # 10시간 반감기

            return max(0.1, min(1.0, time_weight))

        except Exception as e:
            return 1.0

    def _calculate_target_weights(
        self, performance_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """타겟 가중치 계산"""
        try:
            if not performance_scores:
                return self.get_current_weights()

            # 소프트맥스 기반 가중치 계산
            scores = np.array(list(performance_scores.values()))

            # 온도 매개변수 적용 (과도한 집중 방지)
            temperature = 2.0
            exp_scores = np.exp(scores / temperature)
            softmax_weights = exp_scores / np.sum(exp_scores)

            # 최소/최대 가중치 제한 적용
            clipped_weights = np.clip(
                softmax_weights,
                self.update_config.min_weight,
                self.update_config.max_weight,
            )

            # 정규화
            normalized_weights = clipped_weights / np.sum(clipped_weights)

            # 딕셔너리로 변환
            target_weights = {}
            for i, strategy in enumerate(performance_scores.keys()):
                target_weights[strategy] = float(normalized_weights[i])

            return target_weights

        except Exception as e:
            self.logger.error(f"타겟 가중치 계산 중 오류: {str(e)}")
            return self.get_current_weights()

    def _apply_momentum_update(
        self, target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """모멘텀 적용 가중치 업데이트"""
        try:
            updated_weights = {}

            for strategy, target_weight in target_weights.items():
                if strategy in self.strategy_weights:
                    current_weight = self.strategy_weights[strategy].current_weight
                    current_momentum = self.strategy_weights[strategy].momentum

                    # 그래디언트 계산
                    gradient = target_weight - current_weight

                    # 모멘텀 업데이트
                    new_momentum = (
                        self.update_config.momentum * current_momentum
                        + self.update_config.learning_rate * gradient
                    )

                    # 새로운 가중치 계산
                    new_weight = current_weight + new_momentum

                    updated_weights[strategy] = new_weight

                    # 모멘텀 저장
                    self.strategy_weights[strategy].momentum = new_momentum
                else:
                    updated_weights[strategy] = target_weight

            return updated_weights

        except Exception as e:
            self.logger.error(f"모멘텀 업데이트 중 오류: {str(e)}")
            return target_weights

    def _ensure_weight_stability(self, weights: Dict[str, float]) -> Dict[str, float]:
        """가중치 안정성 보장"""
        try:
            stable_weights = {}

            for strategy, weight in weights.items():
                if strategy in self.strategy_weights:
                    current_weight = self.strategy_weights[strategy].current_weight

                    # 변화량 제한
                    max_change = self.update_config.stability_threshold
                    weight_change = weight - current_weight

                    if abs(weight_change) > max_change:
                        # 변화량 제한 적용
                        limited_change = np.sign(weight_change) * max_change
                        stable_weight = current_weight + limited_change
                    else:
                        stable_weight = weight

                    stable_weights[strategy] = stable_weight
                else:
                    stable_weights[strategy] = weight

            return stable_weights

        except Exception as e:
            self.logger.error(f"가중치 안정성 보장 중 오류: {str(e)}")
            return weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """가중치 정규화"""
        try:
            total_weight = sum(weights.values())

            if total_weight == 0:
                # 균등 분배
                equal_weight = 1.0 / len(weights)
                return {strategy: equal_weight for strategy in weights.keys()}

            # 정규화
            normalized = {
                strategy: weight / total_weight for strategy, weight in weights.items()
            }

            # 최소/최대 가중치 제한 재적용
            clipped = {}
            for strategy, weight in normalized.items():
                clipped[strategy] = max(
                    self.update_config.min_weight,
                    min(self.update_config.max_weight, weight),
                )

            # 재정규화
            total_clipped = sum(clipped.values())
            if total_clipped > 0:
                final_weights = {
                    strategy: weight / total_clipped
                    for strategy, weight in clipped.items()
                }
            else:
                equal_weight = 1.0 / len(clipped)
                final_weights = {strategy: equal_weight for strategy in clipped.keys()}

            return final_weights

        except Exception as e:
            self.logger.error(f"가중치 정규화 중 오류: {str(e)}")
            return weights

    def _update_weight_objects(
        self, weights: Dict[str, float], performance_scores: Dict[str, float]
    ) -> None:
        """가중치 객체 업데이트"""
        try:
            current_time = datetime.now()

            for strategy, weight in weights.items():
                if strategy in self.strategy_weights:
                    weight_obj = self.strategy_weights[strategy]

                    # 가중치 업데이트
                    weight_obj.target_weight = weight
                    weight_obj.current_weight = weight
                    weight_obj.performance_score = performance_scores.get(strategy, 0.0)
                    weight_obj.last_update = current_time
                    weight_obj.update_count += 1

                    # 안정성 점수 계산
                    if len(self.weight_history[strategy]) > 1:
                        recent_weights = list(self.weight_history[strategy])[-5:]
                        weight_variance = (
                            np.var(recent_weights) if len(recent_weights) > 1 else 0
                        )
                        weight_obj.stability_score = 1.0 / (1.0 + weight_variance)

                    # 신뢰도 계산
                    weight_obj.confidence = min(1.0, weight_obj.update_count / 10.0)

        except Exception as e:
            self.logger.error(f"가중치 객체 업데이트 중 오류: {str(e)}")

    def _save_weight_history(self, weights: Dict[str, float]) -> None:
        """가중치 이력 저장"""
        try:
            for strategy, weight in weights.items():
                self.weight_history[strategy].append(weight)

                # 이력 크기 제한
                if (
                    len(self.weight_history[strategy])
                    > self.update_config.performance_memory
                ):
                    self.weight_history[strategy].popleft()

        except Exception as e:
            self.logger.error(f"가중치 이력 저장 중 오류: {str(e)}")

    def optimize_weights_with_meta_learning(
        self,
        performance_history: Dict[str, List[PerformanceMetrics]],
        target_performance: float = 0.1,
    ) -> WeightOptimizationResult:
        """메타 러닝을 통한 가중치 최적화"""
        try:
            self.logger.info("메타 러닝 기반 가중치 최적화 시작")

            # 특성 추출
            features = self.meta_learner.extract_features(performance_history)

            # 메타 모델 예측
            predicted_performance = self.meta_learner.predict_performance(features)

            # 최적화 목적함수 정의
            def objective(weights):
                """가중치 최적화 목적함수"""
                try:
                    # 제약 조건 확인
                    if np.sum(weights) != 1.0 or np.any(
                        weights < self.update_config.min_weight
                    ):
                        return 1e10

                    # 예상 성능 계산
                    expected_perf = 0
                    for i, strategy in enumerate(self.strategy_weights.keys()):
                        if (
                            strategy in performance_history
                            and performance_history[strategy]
                        ):
                            recent_metrics = performance_history[strategy][-1]
                            strategy_score = self._calculate_strategy_score(
                                recent_metrics
                            )
                            expected_perf += weights[i] * strategy_score

                    # 목표 성능과의 차이 + 다양성 패널티
                    performance_loss = abs(expected_perf - target_performance)
                    diversity_penalty = -np.sum(
                        weights * np.log(weights + 1e-10)
                    )  # 엔트로피

                    return performance_loss - 0.1 * diversity_penalty

                except Exception as e:
                    return 1e10

            # 제약 조건 설정
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

            bounds = [
                (self.update_config.min_weight, self.update_config.max_weight)
                for _ in range(len(self.strategy_weights))
            ]

            # 초기값 설정
            initial_weights = np.array(
                [w.current_weight for w in self.strategy_weights.values()]
            )

            # 최적화 실행
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    initial_weights,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000, "ftol": 1e-6},
                )

            # 결과 처리
            if result.success:
                optimized_weights = result.x
                optimization_score = -result.fun
                convergence_iterations = result.nit
            else:
                self.logger.warning("가중치 최적화 실패, 현재 가중치 유지")
                optimized_weights = initial_weights
                optimization_score = 0.0
                convergence_iterations = 0

            # 가중치 딕셔너리 생성
            weight_dict = {}
            for i, strategy in enumerate(self.strategy_weights.keys()):
                weight_dict[strategy] = float(optimized_weights[i])

            # 안정성 메트릭 계산
            stability_metrics = self._calculate_stability_metrics(weight_dict)

            # 추천사항 생성
            recommendations = self._generate_optimization_recommendations(
                weight_dict, predicted_performance, optimization_score
            )

            # 결과 객체 생성
            optimization_result = WeightOptimizationResult(
                optimized_weights=weight_dict,
                expected_performance=predicted_performance,
                optimization_score=optimization_score,
                convergence_iterations=convergence_iterations,
                stability_metrics=stability_metrics,
                recommendations=recommendations,
            )

            # 최적화 이력 저장
            self.optimization_history.append(
                {
                    "timestamp": datetime.now(),
                    "result": optimization_result,
                    "target_performance": target_performance,
                }
            )

            self.logger.info("메타 러닝 기반 가중치 최적화 완료")
            return optimization_result

        except Exception as e:
            self.logger.error(f"가중치 최적화 중 오류: {str(e)}")
            return WeightOptimizationResult({}, 0, 0, 0, {}, ["최적화 실패"])

    def _calculate_strategy_score(self, metrics: PerformanceMetrics) -> float:
        """전략 점수 계산"""
        try:
            # 다중 목표 점수
            roi_score = self._normalize_score(metrics.roi, -1.0, 1.0)
            win_rate_score = metrics.win_rate
            stability_score = (
                1.0 / (1.0 + metrics.volatility) if metrics.volatility > 0 else 1.0
            )
            risk_score = 1.0 - min(1.0, max(0.0, metrics.max_drawdown))

            # 가중 평균
            weighted_score = (
                self.update_config.multi_objective_weights["roi"] * roi_score
                + self.update_config.multi_objective_weights["win_rate"]
                * win_rate_score
                + self.update_config.multi_objective_weights["stability"]
                * stability_score
                + self.update_config.multi_objective_weights["risk"] * risk_score
            )

            return max(0.0, min(1.0, weighted_score))

        except Exception as e:
            return 0.0

    def _calculate_stability_metrics(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """안정성 메트릭 계산"""
        try:
            metrics = {}

            # 가중치 분산
            weight_values = list(weights.values())
            metrics["weight_variance"] = float(np.var(weight_values))

            # 가중치 엔트로피 (다양성)
            entropy = -np.sum([w * np.log(w + 1e-10) for w in weight_values])
            metrics["weight_entropy"] = float(entropy)

            # 최대 가중치 비율
            max_weight = max(weight_values)
            metrics["max_weight_ratio"] = float(max_weight)

            # 유효 전략 수
            effective_strategies = 1 / np.sum([w**2 for w in weight_values])
            metrics["effective_strategies"] = float(effective_strategies)

            return metrics

        except Exception as e:
            self.logger.error(f"안정성 메트릭 계산 중 오류: {str(e)}")
            return {}

    def _generate_optimization_recommendations(
        self,
        weights: Dict[str, float],
        predicted_performance: float,
        optimization_score: float,
    ) -> List[str]:
        """최적화 추천사항 생성"""
        recommendations = []

        try:
            # 성능 기반 추천
            if predicted_performance < 0:
                recommendations.append(
                    "예상 성능이 음수입니다. 전략 재검토가 필요합니다."
                )
            elif predicted_performance < 0.05:
                recommendations.append(
                    "예상 성능이 낮습니다. 보다 보수적인 접근을 권장합니다."
                )

            # 가중치 분포 기반 추천
            max_weight = max(weights.values())
            if max_weight > 0.5:
                recommendations.append(
                    "특정 전략에 과도하게 집중되어 있습니다. 다양성을 고려하세요."
                )

            min_weight = min(weights.values())
            if min_weight < 0.1:
                recommendations.append(
                    "일부 전략의 가중치가 너무 낮습니다. 포트폴리오 균형을 검토하세요."
                )

            # 최적화 점수 기반 추천
            if optimization_score < 0.1:
                recommendations.append(
                    "최적화 효과가 제한적입니다. 목표 성능을 재조정하거나 전략을 변경하세요."
                )

            # 긍정적 추천
            if predicted_performance > 0.1 and optimization_score > 0.2:
                recommendations.append(
                    "최적화 결과가 양호합니다. 현재 가중치를 적용해보세요."
                )

        except Exception as e:
            recommendations.append("추천사항 생성 중 오류가 발생했습니다.")

        return recommendations

    def get_current_weights(self) -> Dict[str, float]:
        """현재 가중치 반환"""
        try:
            return {
                strategy: weight_obj.current_weight
                for strategy, weight_obj in self.strategy_weights.items()
            }
        except Exception as e:
            self.logger.error(f"현재 가중치 반환 중 오류: {str(e)}")
            return {}

    def get_weight_summary(self) -> Dict[str, Any]:
        """가중치 요약 정보 반환"""
        try:
            summary = {
                "current_weights": self.get_current_weights(),
                "weight_objects": {},
                "stability_metrics": {},
                "optimization_history_count": len(self.optimization_history),
                "last_update": None,
            }

            # 가중치 객체 정보
            for strategy, weight_obj in self.strategy_weights.items():
                summary["weight_objects"][strategy] = {
                    "current_weight": weight_obj.current_weight,
                    "target_weight": weight_obj.target_weight,
                    "performance_score": weight_obj.performance_score,
                    "stability_score": weight_obj.stability_score,
                    "confidence": weight_obj.confidence,
                    "update_count": weight_obj.update_count,
                    "last_update": weight_obj.last_update.isoformat(),
                }

                if summary[
                    "last_update"
                ] is None or weight_obj.last_update > datetime.fromisoformat(
                    summary["last_update"]
                ):
                    summary["last_update"] = weight_obj.last_update.isoformat()

            # 안정성 메트릭
            current_weights = self.get_current_weights()
            if current_weights:
                summary["stability_metrics"] = self._calculate_stability_metrics(
                    current_weights
                )

            return summary

        except Exception as e:
            self.logger.error(f"가중치 요약 정보 반환 중 오류: {str(e)}")
            return {}

    def save_weights(self) -> None:
        """가중치 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 가중치 데이터 준비
            weight_data = {
                "timestamp": timestamp,
                "strategy_weights": {},
                "weight_history": {},
                "optimization_history": self.optimization_history,
                "config": asdict(self.update_config),
            }

            # 전략 가중치
            for strategy, weight_obj in self.strategy_weights.items():
                weight_data["strategy_weights"][strategy] = {
                    "strategy": weight_obj.strategy,
                    "current_weight": weight_obj.current_weight,
                    "target_weight": weight_obj.target_weight,
                    "momentum": weight_obj.momentum,
                    "performance_score": weight_obj.performance_score,
                    "stability_score": weight_obj.stability_score,
                    "confidence": weight_obj.confidence,
                    "last_update": weight_obj.last_update.isoformat(),
                    "update_count": weight_obj.update_count,
                }

            # 가중치 이력
            for strategy, history in self.weight_history.items():
                weight_data["weight_history"][strategy] = list(history)

            # 파일 저장
            weights_file = self.cache_dir / f"adaptive_weights_{timestamp}.json"
            with open(weights_file, "w", encoding="utf-8") as f:
                json.dump(weight_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"가중치 저장 완료: {weights_file}")

        except Exception as e:
            self.logger.error(f"가중치 저장 중 오류: {str(e)}")

    def load_weights(self, date: Optional[str] = None) -> bool:
        """가중치 로드"""
        try:
            if date:
                pattern = f"adaptive_weights_{date}*.json"
            else:
                pattern = "adaptive_weights_*.json"

            weight_files = list(self.cache_dir.glob(pattern))
            if not weight_files:
                self.logger.warning(f"가중치 파일을 찾을 수 없습니다: {pattern}")
                return False

            # 최신 파일 선택
            latest_file = max(weight_files, key=lambda x: x.name)

            with open(latest_file, "r", encoding="utf-8") as f:
                weight_data = json.load(f)

            # 전략 가중치 복원
            for strategy, data in weight_data.get("strategy_weights", {}).items():
                self.strategy_weights[strategy] = StrategyWeight(
                    strategy=data["strategy"],
                    current_weight=data["current_weight"],
                    target_weight=data["target_weight"],
                    momentum=data["momentum"],
                    performance_score=data["performance_score"],
                    stability_score=data["stability_score"],
                    confidence=data["confidence"],
                    last_update=datetime.fromisoformat(data["last_update"]),
                    update_count=data["update_count"],
                )

            # 가중치 이력 복원
            for strategy, history in weight_data.get("weight_history", {}).items():
                self.weight_history[strategy] = deque(
                    history, maxlen=self.update_config.performance_memory
                )

            # 최적화 이력 복원
            self.optimization_history = weight_data.get("optimization_history", [])

            self.logger.info(f"가중치 로드 완료: {latest_file}")
            return True

        except Exception as e:
            self.logger.error(f"가중치 로드 중 오류: {str(e)}")
            return False
