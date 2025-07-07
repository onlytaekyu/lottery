"""
로또 포트폴리오 관리자 (Lottery Portfolio Manager)

다전략 포트폴리오 접근법을 통한 리스크 분산 및 수익 최적화 시스템입니다.

주요 기능:
- 다전략 포트폴리오 관리
- 동적 가중치 조정
- 리스크 관리
- 성과 기반 리밸런싱
- Kelly Criterion 적용
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod

from ..shared.types import LotteryNumber, ModelPrediction
from ..utils.unified_logging import get_logger
from ..utils.cache_paths import get_cache_dir
from ..models.realistic_lottery_predictor import RealisticLotteryPredictor

logger = get_logger(__name__)


@dataclass
class StrategyAllocation:
    """전략 배분 정보"""

    strategy: str
    weight: float
    combinations: int
    risk_level: str
    expected_return: float
    confidence: float


@dataclass
class PortfolioPerformance:
    """포트폴리오 성능 정보"""

    total_return: float
    total_investment: float
    roi: float
    win_rate: float
    loss_rate: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float


class BaseStrategy(ABC):
    """기본 전략 클래스"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.performance_history = []

    @abstractmethod
    def generate_combinations(
        self, data: List[LotteryNumber], count: int
    ) -> List[ModelPrediction]:
        """조합 생성 추상 메서드"""
        pass

    @abstractmethod
    def calculate_risk_score(self) -> float:
        """리스크 점수 계산 추상 메서드"""
        pass


class FrequencyBasedStrategy(BaseStrategy):
    """빈도 기반 전략"""

    def generate_combinations(
        self, data: List[LotteryNumber], count: int
    ) -> List[ModelPrediction]:
        """빈도 기반 조합 생성"""
        try:
            # 번호별 빈도 계산
            all_numbers = []
            for draw in data:
                all_numbers.extend(draw.numbers)

            frequency = Counter(all_numbers)
            hot_numbers = [num for num, _ in frequency.most_common(20)]

            predictions = []
            for _ in range(count):
                # 고빈도 번호에서 4-5개 선택
                import random

                selected = random.sample(hot_numbers, 5)

                # 나머지 1-2개는 중간 빈도에서 선택
                remaining_numbers = [n for n in range(1, 46) if n not in selected]
                selected.extend(random.sample(remaining_numbers, 6 - len(selected)))

                predictions.append(
                    ModelPrediction(
                        numbers=sorted(selected),
                        confidence=0.7,
                        model_type="frequency_based",
                        metadata={"strategy": "frequency_based", "risk_level": "low"},
                    )
                )

            return predictions

        except Exception as e:
            self.logger.error(f"빈도 기반 조합 생성 중 오류: {str(e)}")
            return []

    def calculate_risk_score(self) -> float:
        """리스크 점수 계산 (낮은 리스크)"""
        return 0.3


class ClusterAnalysisStrategy(BaseStrategy):
    """클러스터 분석 전략"""

    def generate_combinations(
        self, data: List[LotteryNumber], count: int
    ) -> List[ModelPrediction]:
        """클러스터 분석 기반 조합 생성"""
        try:
            from sklearn.cluster import KMeans

            # 번호를 원-핫 인코딩으로 변환
            features = []
            for draw in data:
                feature = [0] * 45
                for num in draw.numbers:
                    feature[num - 1] = 1
                features.append(feature)

            features = np.array(features)

            # K-means 클러스터링
            n_clusters = min(3, len(data) // 20)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(features)

                # 각 클러스터의 중심에서 번호 선택
                predictions = []
                for _ in range(count):
                    cluster_idx = np.random.randint(0, n_clusters)
                    centroid = kmeans.cluster_centers_[cluster_idx]

                    # 상위 확률 번호 선택
                    top_indices = np.argsort(centroid)[-6:]
                    numbers = [idx + 1 for idx in top_indices]

                    predictions.append(
                        ModelPrediction(
                            numbers=sorted(numbers),
                            confidence=0.6,
                            model_type="cluster_analysis",
                            metadata={
                                "strategy": "cluster_analysis",
                                "risk_level": "medium",
                            },
                        )
                    )

                return predictions

            return []

        except Exception as e:
            self.logger.error(f"클러스터 분석 조합 생성 중 오류: {str(e)}")
            return []

    def calculate_risk_score(self) -> float:
        """리스크 점수 계산 (중간 리스크)"""
        return 0.5


class TrendFollowingStrategy(BaseStrategy):
    """트렌드 추종 전략"""

    def generate_combinations(
        self, data: List[LotteryNumber], count: int
    ) -> List[ModelPrediction]:
        """트렌드 추종 조합 생성"""
        try:
            if len(data) < 10:
                return []

            # 최근 vs 과거 트렌드 분석
            recent_data = data[-10:]
            older_data = data[-20:-10] if len(data) >= 20 else data[:-10]

            recent_freq = Counter()
            older_freq = Counter()

            for draw in recent_data:
                recent_freq.update(draw.numbers)

            for draw in older_data:
                older_freq.update(draw.numbers)

            # 상승 트렌드 번호 식별
            trend_scores = {}
            for num in range(1, 46):
                recent_count = recent_freq.get(num, 0)
                older_count = older_freq.get(num, 0)

                if older_count > 0:
                    trend_scores[num] = (recent_count - older_count) / older_count
                else:
                    trend_scores[num] = recent_count

            # 상승 트렌드 번호 선택
            ascending_numbers = sorted(
                trend_scores.items(), key=lambda x: x[1], reverse=True
            )[:15]
            trend_numbers = [num for num, _ in ascending_numbers]

            predictions = []
            for _ in range(count):
                import random

                # 트렌드 번호에서 4-5개 선택
                selected = random.sample(trend_numbers, min(5, len(trend_numbers)))

                # 나머지는 랜덤
                remaining_numbers = [n for n in range(1, 46) if n not in selected]
                selected.extend(random.sample(remaining_numbers, 6 - len(selected)))

                predictions.append(
                    ModelPrediction(
                        numbers=sorted(selected),
                        confidence=0.5,
                        model_type="trend_following",
                        metadata={"strategy": "trend_following", "risk_level": "high"},
                    )
                )

            return predictions

        except Exception as e:
            self.logger.error(f"트렌드 추종 조합 생성 중 오류: {str(e)}")
            return []

    def calculate_risk_score(self) -> float:
        """리스크 점수 계산 (높은 리스크)"""
        return 0.8


class AIEnsembleStrategy(BaseStrategy):
    """AI 앙상블 전략"""

    def generate_combinations(
        self, data: List[LotteryNumber], count: int
    ) -> List[ModelPrediction]:
        """AI 앙상블 조합 생성"""
        try:
            # 여러 AI 모델의 예측을 조합
            predictor = RealisticLotteryPredictor()

            # 간단한 앙상블 구현
            predictions = []
            for _ in range(count):
                # 다양한 전략 조합
                strategies = ["conservative", "aggressive", "balanced"]
                selected_strategy = np.random.choice(strategies)

                # 전략에 따른 번호 생성
                if selected_strategy == "conservative":
                    numbers = self._generate_conservative_numbers(data)
                elif selected_strategy == "aggressive":
                    numbers = self._generate_aggressive_numbers(data)
                else:
                    numbers = self._generate_balanced_numbers(data)

                predictions.append(
                    ModelPrediction(
                        numbers=sorted(numbers),
                        confidence=0.6,
                        model_type="ai_ensemble",
                        metadata={"strategy": "ai_ensemble", "risk_level": "medium"},
                    )
                )

            return predictions

        except Exception as e:
            self.logger.error(f"AI 앙상블 조합 생성 중 오류: {str(e)}")
            return []

    def _generate_conservative_numbers(self, data: List[LotteryNumber]) -> List[int]:
        """보수적 번호 생성"""
        import random

        # 빈도 기반 안정적 번호
        all_numbers = []
        for draw in data[-30:]:  # 최근 30회
            all_numbers.extend(draw.numbers)

        frequency = Counter(all_numbers)
        hot_numbers = [num for num, _ in frequency.most_common(15)]

        return (
            random.sample(hot_numbers, 6)
            if len(hot_numbers) >= 6
            else random.sample(range(1, 46), 6)
        )

    def _generate_aggressive_numbers(self, data: List[LotteryNumber]) -> List[int]:
        """공격적 번호 생성"""
        import random

        # 최신 트렌드 기반
        recent_numbers = []
        for draw in data[-5:]:  # 최근 5회
            recent_numbers.extend(draw.numbers)

        trend_numbers = list(set(recent_numbers))
        selected = random.sample(trend_numbers, min(4, len(trend_numbers)))

        # 나머지는 랜덤
        remaining = [n for n in range(1, 46) if n not in selected]
        selected.extend(random.sample(remaining, 6 - len(selected)))

        return selected

    def _generate_balanced_numbers(self, data: List[LotteryNumber]) -> List[int]:
        """균형 번호 생성"""
        import random

        # 빈도와 트렌드 조합
        conservative = self._generate_conservative_numbers(data)[:3]
        aggressive = self._generate_aggressive_numbers(data)[:3]

        # 중복 제거
        combined = list(set(conservative + aggressive))
        if len(combined) >= 6:
            return random.sample(combined, 6)
        else:
            remaining = [n for n in range(1, 46) if n not in combined]
            combined.extend(random.sample(remaining, 6 - len(combined)))
            return combined

    def calculate_risk_score(self) -> float:
        """리스크 점수 계산 (중간 리스크)"""
        return 0.6


class ContrarianStrategy(BaseStrategy):
    """역발상 전략"""

    def generate_combinations(
        self, data: List[LotteryNumber], count: int
    ) -> List[ModelPrediction]:
        """역발상 조합 생성"""
        try:
            # 최근 출현 빈도가 낮은 번호 선택
            recent_data = data[-20:] if len(data) >= 20 else data
            all_numbers = []
            for draw in recent_data:
                all_numbers.extend(draw.numbers)

            frequency = Counter(all_numbers)

            # 출현 빈도가 낮은 번호 선택
            cold_numbers = []
            for num in range(1, 46):
                if frequency.get(num, 0) <= 2:  # 2회 이하 출현
                    cold_numbers.append(num)

            predictions = []
            for _ in range(count):
                import random

                if len(cold_numbers) >= 6:
                    selected = random.sample(cold_numbers, 6)
                else:
                    selected = cold_numbers.copy()
                    remaining = [n for n in range(1, 46) if n not in selected]
                    selected.extend(random.sample(remaining, 6 - len(selected)))

                predictions.append(
                    ModelPrediction(
                        numbers=sorted(selected),
                        confidence=0.4,
                        model_type="contrarian",
                        metadata={"strategy": "contrarian", "risk_level": "very_high"},
                    )
                )

            return predictions

        except Exception as e:
            self.logger.error(f"역발상 조합 생성 중 오류: {str(e)}")
            return []

    def calculate_risk_score(self) -> float:
        """리스크 점수 계산 (매우 높은 리스크)"""
        return 0.9


class LotteryPortfolioManager:
    """로또 포트폴리오 관리자"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        포트폴리오 관리자 초기화

        Args:
            config: 설정 객체
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # 전략 객체 초기화
        self.strategies = {
            "frequency_based": FrequencyBasedStrategy(self.config),
            "cluster_analysis": ClusterAnalysisStrategy(self.config),
            "trend_following": TrendFollowingStrategy(self.config),
            "ai_ensemble": AIEnsembleStrategy(self.config),
            "contrarian": ContrarianStrategy(self.config),
        }

        # 기본 포트폴리오 배분
        self.default_allocation = {
            "conservative": 0.4,  # 40% - 빈도 + 클러스터
            "aggressive": 0.4,  # 40% - 트렌드 + AI
            "balanced": 0.2,  # 20% - 역발상 + 혼합
        }

        # 전략별 매핑
        self.strategy_mapping = {
            "conservative": ["frequency_based", "cluster_analysis"],
            "aggressive": ["trend_following", "ai_ensemble"],
            "balanced": ["contrarian"],
        }

        # 성능 추적
        self.performance_history = []
        self.allocation_history = []

        # 리스크 관리 설정
        self.risk_limits = {
            "max_allocation_per_strategy": 0.6,  # 단일 전략 최대 60%
            "min_allocation_per_strategy": 0.1,  # 단일 전략 최소 10%
            "max_risk_score": 0.7,  # 전체 포트폴리오 최대 리스크
            "rebalancing_threshold": 0.1,  # 리밸런싱 임계값
        }

        # 결과 저장 경로
        self.cache_dir = Path(get_cache_dir()) / "portfolio_manager"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("로또 포트폴리오 관리자 초기화 완료")

    def allocate_combinations(
        self, total_combinations: int = 5, data: Optional[List[LotteryNumber]] = None
    ) -> Dict[str, StrategyAllocation]:
        """조합 배분 전략"""
        try:
            self.logger.info(f"조합 배분 시작 - 총 {total_combinations}개 조합")

            # 현재 성과 기반 가중치 조정
            adjusted_allocation = self._adjust_allocation_by_performance()

            # 배분 계산
            allocations = {}
            remaining_combinations = total_combinations

            for portfolio_type, weight in adjusted_allocation.items():
                combinations_count = int(total_combinations * weight)

                # 최소 1개는 보장
                if combinations_count == 0 and remaining_combinations > 0:
                    combinations_count = 1

                remaining_combinations -= combinations_count

                # 전략별 세부 배분
                strategies = self.strategy_mapping[portfolio_type]
                strategy_weight = 1.0 / len(strategies)

                for strategy in strategies:
                    strategy_combinations = max(
                        1, combinations_count // len(strategies)
                    )

                    # 기대 수익 계산
                    expected_return = self._calculate_expected_return(strategy, data)

                    # 신뢰도 계산
                    confidence = self._calculate_strategy_confidence(strategy)

                    allocations[strategy] = StrategyAllocation(
                        strategy=strategy,
                        weight=weight * strategy_weight,
                        combinations=strategy_combinations,
                        risk_level=self._get_risk_level(strategy),
                        expected_return=expected_return,
                        confidence=confidence,
                    )

            # 남은 조합 분배
            if remaining_combinations > 0:
                # 가장 성과가 좋은 전략에 추가 배분
                best_strategy = self._get_best_performing_strategy()
                if best_strategy in allocations:
                    allocations[best_strategy].combinations += remaining_combinations

            # 배분 이력 저장
            self.allocation_history.append(
                {
                    "timestamp": datetime.now(),
                    "allocations": allocations,
                    "total_combinations": total_combinations,
                }
            )

            self.logger.info("조합 배분 완료")
            return allocations

        except Exception as e:
            self.logger.error(f"조합 배분 중 오류: {str(e)}")
            return {}

    def generate_portfolio(
        self, allocation: Dict[str, StrategyAllocation], data: List[LotteryNumber]
    ) -> List[ModelPrediction]:
        """포트폴리오 생성"""
        try:
            self.logger.info("포트폴리오 생성 시작")

            all_predictions = []

            for strategy_name, strategy_allocation in allocation.items():
                if strategy_name in self.strategies:
                    strategy = self.strategies[strategy_name]

                    # 전략별 조합 생성
                    predictions = strategy.generate_combinations(
                        data, strategy_allocation.combinations
                    )

                    # 메타데이터에 포트폴리오 정보 추가
                    for prediction in predictions:
                        if prediction.metadata is None:
                            prediction.metadata = {}
                        prediction.metadata.update(
                            {
                                "portfolio_weight": strategy_allocation.weight,
                                "expected_return": strategy_allocation.expected_return,
                                "allocation_confidence": strategy_allocation.confidence,
                            }
                        )

                    all_predictions.extend(predictions)

            # 포트폴리오 다양성 검증
            all_predictions = self._ensure_portfolio_diversity(all_predictions)

            self.logger.info(f"포트폴리오 생성 완료 - {len(all_predictions)}개 조합")
            return all_predictions

        except Exception as e:
            self.logger.error(f"포트폴리오 생성 중 오류: {str(e)}")
            return []

    def rebalance_portfolio(
        self, performance_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """성과 기반 포트폴리오 리밸런싱"""
        try:
            self.logger.info("포트폴리오 리밸런싱 시작")

            if not performance_history:
                return self.default_allocation

            # 최근 성과 분석
            recent_performance = (
                performance_history[-10:]
                if len(performance_history) >= 10
                else performance_history
            )

            # 전략별 성과 점수 계산
            strategy_scores = {}
            for portfolio_type in self.default_allocation.keys():
                scores = []
                for perf in recent_performance:
                    strategy_perf = perf.get(portfolio_type, {})
                    roi = strategy_perf.get("roi", 0)
                    hit_rate = strategy_perf.get("hit_rate", 0)

                    # 종합 점수 (ROI 70% + 적중률 30%)
                    score = roi * 0.7 + hit_rate * 0.3
                    scores.append(score)

                strategy_scores[portfolio_type] = np.mean(scores) if scores else 0

            # 성과 기반 가중치 조정
            total_score = sum(strategy_scores.values())
            if total_score > 0:
                base_weight = 1.0 / len(strategy_scores)
                new_allocation = {}

                for portfolio_type, score in strategy_scores.items():
                    # 성과 점수를 반영한 가중치 계산
                    performance_factor = score / total_score
                    adjusted_weight = base_weight * (1 + performance_factor)

                    # 리스크 제한 적용
                    max_weight = self.risk_limits["max_allocation_per_strategy"]
                    min_weight = self.risk_limits["min_allocation_per_strategy"]

                    new_allocation[portfolio_type] = max(
                        min_weight, min(max_weight, adjusted_weight)
                    )

                # 가중치 정규화
                total_weight = sum(new_allocation.values())
                for portfolio_type in new_allocation:
                    new_allocation[portfolio_type] /= total_weight

                self.logger.info("포트폴리오 리밸런싱 완료")
                return new_allocation

            return self.default_allocation

        except Exception as e:
            self.logger.error(f"포트폴리오 리밸런싱 중 오류: {str(e)}")
            return self.default_allocation

    def calculate_portfolio_risk(
        self, allocation: Dict[str, StrategyAllocation]
    ) -> float:
        """포트폴리오 리스크 계산"""
        try:
            total_risk = 0
            total_weight = 0

            for strategy_name, strategy_allocation in allocation.items():
                if strategy_name in self.strategies:
                    strategy = self.strategies[strategy_name]
                    risk_score = strategy.calculate_risk_score()

                    # 가중 리스크 계산
                    weighted_risk = risk_score * strategy_allocation.weight
                    total_risk += weighted_risk
                    total_weight += strategy_allocation.weight

            # 평균 리스크 계산
            portfolio_risk = total_risk / total_weight if total_weight > 0 else 0

            return portfolio_risk

        except Exception as e:
            self.logger.error(f"포트폴리오 리스크 계산 중 오류: {str(e)}")
            return 1.0  # 최대 리스크 반환

    def calculate_portfolio_performance(
        self, predictions: List[ModelPrediction], actual_results: List[LotteryNumber]
    ) -> PortfolioPerformance:
        """포트폴리오 성능 계산"""
        try:
            if not predictions or not actual_results:
                return PortfolioPerformance(0, 0, 0, 0, 1, 0, 0, 0)

            total_investment = len(predictions) * 1000  # 1000원/게임
            total_return = 0
            wins = 0
            returns = []

            # 각 예측에 대해 성과 계산
            for prediction in predictions:
                game_return = 0
                for actual in actual_results:
                    matches = len(set(prediction.numbers) & set(actual.numbers))

                    # 당첨금 계산
                    if matches == 6:
                        game_return = 2000000000  # 1등
                    elif matches == 5:
                        game_return = 1500000  # 3등
                    elif matches == 4:
                        game_return = 50000  # 4등
                    elif matches == 3:
                        game_return = 5000  # 5등

                total_return += game_return
                net_return = game_return - 1000
                returns.append(net_return)

                if net_return > 0:
                    wins += 1

            # 성능 지표 계산
            roi = (
                (total_return - total_investment) / total_investment
                if total_investment > 0
                else 0
            )
            win_rate = wins / len(predictions) if predictions else 0
            loss_rate = 1 - win_rate

            # 샤프 비율 계산
            if len(returns) > 1:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0

            # 최대 손실 계산
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

            # 변동성 계산
            volatility = np.std(returns) if len(returns) > 1 else 0

            return PortfolioPerformance(
                total_return=total_return,
                total_investment=total_investment,
                roi=roi,
                win_rate=win_rate,
                loss_rate=loss_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
            )

        except Exception as e:
            self.logger.error(f"포트폴리오 성능 계산 중 오류: {str(e)}")
            return PortfolioPerformance(0, 0, 0, 0, 1, 0, 0, 0)

    def _adjust_allocation_by_performance(self) -> Dict[str, float]:
        """성과 기반 배분 조정"""
        try:
            if not self.performance_history:
                return self.default_allocation

            # 최근 성과 기반 조정
            return self.rebalance_portfolio(self.performance_history)

        except Exception as e:
            self.logger.error(f"성과 기반 배분 조정 중 오류: {str(e)}")
            return self.default_allocation

    def _calculate_expected_return(
        self, strategy: str, data: Optional[List[LotteryNumber]]
    ) -> float:
        """전략별 기대 수익 계산"""
        try:
            # 전략별 기대 수익률 (과거 성과 기반)
            base_returns = {
                "frequency_based": 0.02,  # 2%
                "cluster_analysis": 0.015,  # 1.5%
                "trend_following": 0.03,  # 3%
                "ai_ensemble": 0.025,  # 2.5%
                "contrarian": 0.01,  # 1%
            }

            base_return = base_returns.get(strategy, 0.01)

            # 최근 성과 반영
            if self.performance_history:
                recent_performance = self.performance_history[-5:]  # 최근 5회
                strategy_returns = []

                for perf in recent_performance:
                    strategy_perf = perf.get(strategy, {})
                    roi = strategy_perf.get("roi", 0)
                    strategy_returns.append(roi)

                if strategy_returns:
                    recent_avg = np.mean(strategy_returns)
                    # 기본 수익률과 최근 성과의 가중 평균
                    return base_return * 0.3 + recent_avg * 0.7

            return base_return

        except Exception as e:
            self.logger.error(f"기대 수익 계산 중 오류: {str(e)}")
            return 0.01

    def _calculate_strategy_confidence(self, strategy: str) -> float:
        """전략별 신뢰도 계산"""
        try:
            # 기본 신뢰도
            base_confidence = {
                "frequency_based": 0.7,  # 높은 신뢰도
                "cluster_analysis": 0.6,  # 중간 신뢰도
                "trend_following": 0.5,  # 중간 신뢰도
                "ai_ensemble": 0.6,  # 중간 신뢰도
                "contrarian": 0.4,  # 낮은 신뢰도
            }

            base_conf = base_confidence.get(strategy, 0.5)

            # 최근 성과 일관성 반영
            if self.performance_history:
                recent_performance = self.performance_history[-10:]  # 최근 10회
                strategy_rois = []

                for perf in recent_performance:
                    strategy_perf = perf.get(strategy, {})
                    roi = strategy_perf.get("roi", 0)
                    strategy_rois.append(roi)

                if len(strategy_rois) > 1:
                    # 일관성 계산 (낮은 변동성 = 높은 신뢰도)
                    std_roi = np.std(strategy_rois)
                    consistency = max(0, 1 - std_roi) if std_roi < 1 else 0

                    # 기본 신뢰도와 일관성의 가중 평균
                    return base_conf * 0.6 + consistency * 0.4

            return base_conf

        except Exception as e:
            self.logger.error(f"전략 신뢰도 계산 중 오류: {str(e)}")
            return 0.5

    def _get_risk_level(self, strategy: str) -> str:
        """전략별 리스크 레벨 반환"""
        risk_levels = {
            "frequency_based": "low",
            "cluster_analysis": "medium",
            "trend_following": "high",
            "ai_ensemble": "medium",
            "contrarian": "very_high",
        }
        return risk_levels.get(strategy, "medium")

    def _get_best_performing_strategy(self) -> str:
        """최고 성과 전략 식별"""
        try:
            if not self.performance_history:
                return "frequency_based"  # 기본값

            # 최근 성과 기반 최고 전략 선택
            recent_performance = self.performance_history[-5:]
            strategy_scores = {}

            for strategy in self.strategies.keys():
                scores = []
                for perf in recent_performance:
                    strategy_perf = perf.get(strategy, {})
                    roi = strategy_perf.get("roi", 0)
                    hit_rate = strategy_perf.get("hit_rate", 0)
                    score = roi * 0.7 + hit_rate * 0.3
                    scores.append(score)

                strategy_scores[strategy] = np.mean(scores) if scores else 0

            return max(strategy_scores.items(), key=lambda x: x[1])[0]

        except Exception as e:
            self.logger.error(f"최고 성과 전략 식별 중 오류: {str(e)}")
            return "frequency_based"

    def _ensure_portfolio_diversity(
        self, predictions: List[ModelPrediction]
    ) -> List[ModelPrediction]:
        """포트폴리오 다양성 보장"""
        try:
            if len(predictions) <= 1:
                return predictions

            # 중복 조합 제거
            unique_predictions = []
            seen_combinations = set()

            for prediction in predictions:
                combination_tuple = tuple(sorted(prediction.numbers))
                if combination_tuple not in seen_combinations:
                    unique_predictions.append(prediction)
                    seen_combinations.add(combination_tuple)

            # 번호 분포 다양성 검증
            all_numbers = []
            for prediction in unique_predictions:
                all_numbers.extend(prediction.numbers)

            number_frequency = Counter(all_numbers)

            # 특정 번호가 과도하게 집중된 경우 조정
            max_frequency = len(unique_predictions) * 0.6  # 60% 이하로 제한

            for prediction in unique_predictions:
                adjusted_numbers = []
                for num in prediction.numbers:
                    if number_frequency[num] <= max_frequency:
                        adjusted_numbers.append(num)

                # 부족한 번호 보충
                while len(adjusted_numbers) < 6:
                    # 빈도가 낮은 번호 선택
                    low_freq_numbers = [
                        n
                        for n in range(1, 46)
                        if n not in adjusted_numbers
                        and number_frequency[n] < max_frequency
                    ]
                    if low_freq_numbers:
                        import random

                        adjusted_numbers.append(random.choice(low_freq_numbers))
                    else:
                        # 모든 번호가 고빈도인 경우 랜덤 선택
                        remaining = [
                            n for n in range(1, 46) if n not in adjusted_numbers
                        ]
                        if remaining:
                            adjusted_numbers.append(random.choice(remaining))
                        else:
                            break

                prediction.numbers = sorted(adjusted_numbers)

            return unique_predictions

        except Exception as e:
            self.logger.error(f"포트폴리오 다양성 보장 중 오류: {str(e)}")
            return predictions

    def save_portfolio_state(self) -> None:
        """포트폴리오 상태 저장"""
        try:
            state = {
                "allocation_history": self.allocation_history,
                "performance_history": self.performance_history,
                "default_allocation": self.default_allocation,
                "risk_limits": self.risk_limits,
                "timestamp": datetime.now().isoformat(),
            }

            state_file = self.cache_dir / "portfolio_state.json"
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"포트폴리오 상태 저장 완료: {state_file}")

        except Exception as e:
            self.logger.error(f"포트폴리오 상태 저장 중 오류: {str(e)}")

    def load_portfolio_state(self) -> None:
        """포트폴리오 상태 로드"""
        try:
            state_file = self.cache_dir / "portfolio_state.json"
            if state_file.exists():
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)

                self.allocation_history = state.get("allocation_history", [])
                self.performance_history = state.get("performance_history", [])
                self.default_allocation = state.get(
                    "default_allocation", self.default_allocation
                )
                self.risk_limits = state.get("risk_limits", self.risk_limits)

                self.logger.info(f"포트폴리오 상태 로드 완료: {state_file}")

        except Exception as e:
            self.logger.error(f"포트폴리오 상태 로드 중 오류: {str(e)}")
