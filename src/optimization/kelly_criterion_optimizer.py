"""
Kelly Criterion 기반 베팅 최적화 시스템

현실적 로또 시스템에서 최적 베팅 크기와 포트폴리오 배분을 계산
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.unified_logging import get_logger
from ..utils.cache_paths import get_cache_dir
from ..utils.memory_manager import get_memory_manager
from ..models.realistic_lottery_predictor import PrizeGrade, PredictionResult

logger = get_logger(__name__)


class RiskLevel(Enum):
    """리스크 레벨"""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class KellyResult:
    """Kelly 계산 결과"""

    optimal_fraction: float
    expected_return: float
    win_probability: float
    loss_probability: float
    risk_adjusted_fraction: float
    recommendation: str


@dataclass
class PortfolioAllocation:
    """포트폴리오 배분"""

    strategy_allocations: Dict[str, float]
    total_investment: float
    expected_return: float
    risk_score: float
    kelly_fractions: Dict[str, float]


class KellyCriterionOptimizer:
    """Kelly Criterion 베팅 최적화"""

    def __init__(self, config_path: str = "config/realistic_lottery_config.yaml"):
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()
        self.cache_dir = get_cache_dir("kelly_optimization")

        # 설정 로드
        self.config = self._load_config(config_path)

        # Kelly 설정
        self.kelly_fraction = self.config.get("risk_management", {}).get(
            "kelly_fraction", 0.25
        )
        self.max_bet_fraction = 0.5  # 최대 베팅 비율
        self.min_bet_fraction = 0.01  # 최소 베팅 비율

        # 등급별 상금 설정 (평균값)
        self.prize_amounts = {
            PrizeGrade.FIRST: 2000000000,  # 20억
            PrizeGrade.SECOND: 50000000,  # 5천만
            PrizeGrade.THIRD: 1500000,  # 150만
            PrizeGrade.FOURTH: 50000,  # 5만
            PrizeGrade.FIFTH: 5000,  # 5천
        }

        self.cost_per_combination = 1000  # 1천원

        self.logger.info("✅ Kelly Criterion 최적화 시스템 초기화 완료")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("realistic_lottery", {})
        except FileNotFoundError:
            self.logger.warning(f"설정 파일 없음: {config_path}, 기본 설정 사용")
            return {
                "risk_management": {"kelly_fraction": 0.25},
                "portfolio_allocation": {
                    "conservative": 0.4,
                    "aggressive": 0.4,
                    "balanced": 0.2,
                },
            }

    def calculate_optimal_bet_size(
        self,
        win_probability: float,
        odds: float,
        bankroll: float,
        risk_level: RiskLevel = RiskLevel.MODERATE,
    ) -> KellyResult:
        """최적 베팅 크기 계산"""
        try:
            # Kelly Criterion 공식: f* = (bp - q) / b
            # f*: 최적 베팅 비율
            # b: 배당률 - 1
            # p: 승률
            # q: 패율 (1-p)

            b = odds - 1  # 순 배당률
            p = win_probability
            q = 1 - p

            # Kelly 비율 계산
            if b <= 0 or p <= 0:
                kelly_fraction = 0.0
            else:
                kelly_fraction = (b * p - q) / b

            # 음수인 경우 베팅하지 않음
            if kelly_fraction <= 0:
                kelly_fraction = 0.0

            # 최대/최소 비율 제한
            kelly_fraction = max(
                self.min_bet_fraction, min(kelly_fraction, self.max_bet_fraction)
            )

            # 리스크 레벨에 따른 조정
            risk_adjusted_fraction = self._adjust_for_risk_level(
                kelly_fraction, risk_level
            )

            # 기대 수익 계산
            expected_return = p * (odds - 1) - q

            # 추천사항 생성
            recommendation = self._generate_recommendation(
                kelly_fraction, risk_adjusted_fraction, expected_return, p
            )

            return KellyResult(
                optimal_fraction=kelly_fraction,
                expected_return=expected_return,
                win_probability=p,
                loss_probability=q,
                risk_adjusted_fraction=risk_adjusted_fraction,
                recommendation=recommendation,
            )

        except Exception as e:
            self.logger.error(f"최적 베팅 크기 계산 오류: {e}")
            return KellyResult(
                optimal_fraction=0.0,
                expected_return=0.0,
                win_probability=0.0,
                loss_probability=1.0,
                risk_adjusted_fraction=0.0,
                recommendation="계산 오류로 인해 베팅을 권장하지 않습니다.",
            )

    def _adjust_for_risk_level(
        self, kelly_fraction: float, risk_level: RiskLevel
    ) -> float:
        """리스크 레벨에 따른 Kelly 비율 조정"""
        try:
            if risk_level == RiskLevel.CONSERVATIVE:
                # 보수적: Kelly 결과의 50%
                return kelly_fraction * 0.5
            elif risk_level == RiskLevel.MODERATE:
                # 중간: Kelly 결과의 75%
                return kelly_fraction * 0.75
            elif risk_level == RiskLevel.AGGRESSIVE:
                # 공격적: Kelly 결과의 100%
                return kelly_fraction
            else:
                return kelly_fraction * 0.75

        except Exception as e:
            self.logger.error(f"리스크 레벨 조정 오류: {e}")
            return kelly_fraction * 0.5

    def _generate_recommendation(
        self,
        kelly_fraction: float,
        risk_adjusted_fraction: float,
        expected_return: float,
        win_probability: float,
    ) -> str:
        """추천사항 생성"""
        try:
            if kelly_fraction <= 0:
                return "베팅을 권장하지 않습니다. 기대값이 음수입니다."
            elif kelly_fraction < 0.05:
                return "매우 소량 베팅을 권장합니다."
            elif kelly_fraction < 0.15:
                return "적당한 베팅을 권장합니다."
            elif kelly_fraction < 0.3:
                return "적극적인 베팅을 고려할 수 있습니다."
            else:
                return "매우 공격적인 베팅이지만 리스크가 높습니다."

        except Exception as e:
            self.logger.error(f"추천사항 생성 오류: {e}")
            return "추천사항 생성 중 오류가 발생했습니다."

    def portfolio_kelly(
        self, predictions: List[PredictionResult], bankroll: float
    ) -> PortfolioAllocation:
        """포트폴리오 Kelly 최적화"""
        try:
            strategy_allocations = {}
            kelly_fractions = {}
            total_expected_return = 0.0
            total_risk_score = 0.0

            # 각 전략별 Kelly 계산
            for prediction in predictions:
                strategy_name = prediction.strategy_type

                # 전략별 승률 및 배당률 계산
                win_prob, odds = self._calculate_strategy_metrics(prediction)

                # Kelly 계산
                kelly_result = self.calculate_optimal_bet_size(
                    win_prob,
                    odds,
                    bankroll,
                    self._get_risk_level_from_strategy(strategy_name),
                )

                # 결과 저장
                kelly_fractions[strategy_name] = kelly_result.risk_adjusted_fraction
                total_expected_return += kelly_result.expected_return

                # 리스크 점수 계산 (변동성 기반)
                risk_score = self._calculate_risk_score(prediction, kelly_result)
                total_risk_score += risk_score

            # 포트폴리오 정규화
            total_kelly = sum(kelly_fractions.values())
            if total_kelly > 0:
                # Kelly 비율에 따른 배분
                for strategy in kelly_fractions:
                    strategy_allocations[strategy] = (
                        kelly_fractions[strategy] / total_kelly
                    )
            else:
                # 기본 배분 사용
                default_allocation = self.config.get("portfolio_allocation", {})
                strategy_allocations = {
                    "conservative": default_allocation.get("conservative", 0.4),
                    "aggressive": default_allocation.get("aggressive", 0.4),
                    "balanced": default_allocation.get("balanced", 0.2),
                }

            # 총 투자금 계산 (Kelly 기반)
            optimal_investment = bankroll * min(total_kelly, self.max_bet_fraction)

            return PortfolioAllocation(
                strategy_allocations=strategy_allocations,
                total_investment=optimal_investment,
                expected_return=total_expected_return,
                risk_score=total_risk_score / len(predictions) if predictions else 0,
                kelly_fractions=kelly_fractions,
            )

        except Exception as e:
            self.logger.error(f"포트폴리오 Kelly 최적화 오류: {e}")
            return PortfolioAllocation(
                strategy_allocations={},
                total_investment=0.0,
                expected_return=0.0,
                risk_score=1.0,
                kelly_fractions={},
            )

    def _calculate_strategy_metrics(
        self, prediction: PredictionResult
    ) -> Tuple[float, float]:
        """전략별 승률 및 배당률 계산"""
        try:
            # 등급별 확률과 상금을 이용한 전체 승률 계산
            total_win_prob = 0.0
            total_expected_prize = 0.0

            for grade, probability in prediction.grade_probabilities.items():
                if grade in self.prize_amounts:
                    total_win_prob += probability
                    total_expected_prize += probability * self.prize_amounts[grade]

            # 승률이 0인 경우 최소값 설정
            win_probability = max(total_win_prob, 0.001)

            # 배당률 계산 (기대상금 / 투자비용)
            odds = (
                total_expected_prize / self.cost_per_combination
                if self.cost_per_combination > 0
                else 1.0
            )
            odds = max(odds, 1.0)  # 최소 1.0 (손실 없음)

            return win_probability, odds

        except Exception as e:
            self.logger.error(f"전략 지표 계산 오류: {e}")
            return 0.001, 1.0

    def _get_risk_level_from_strategy(self, strategy_name: str) -> RiskLevel:
        """전략명에서 리스크 레벨 추출"""
        try:
            if "conservative" in strategy_name.lower():
                return RiskLevel.CONSERVATIVE
            elif "aggressive" in strategy_name.lower():
                return RiskLevel.AGGRESSIVE
            else:
                return RiskLevel.MODERATE

        except Exception as e:
            self.logger.error(f"리스크 레벨 추출 오류: {e}")
            return RiskLevel.MODERATE

    def _calculate_risk_score(
        self, prediction: PredictionResult, kelly_result: KellyResult
    ) -> float:
        """리스크 점수 계산"""
        try:
            # 여러 요소를 고려한 리스크 점수
            # 1. 신뢰도 (낮을수록 리스크 높음)
            confidence_risk = 1.0 - prediction.confidence_score

            # 2. Kelly 비율 (높을수록 리스크 높음)
            kelly_risk = kelly_result.optimal_fraction

            # 3. 승률 (낮을수록 리스크 높음)
            probability_risk = 1.0 - kelly_result.win_probability

            # 가중 평균으로 최종 리스크 점수 계산
            risk_score = (
                confidence_risk * 0.4 + kelly_risk * 0.3 + probability_risk * 0.3
            )

            return max(0.0, min(1.0, risk_score))

        except Exception as e:
            self.logger.error(f"리스크 점수 계산 오류: {e}")
            return 0.5

    def calculate_position_sizing(
        self,
        predictions: List[PredictionResult],
        bankroll: float,
        max_positions: int = 5,
    ) -> Dict[str, Any]:
        """포지션 크기 계산"""
        try:
            # 포트폴리오 최적화
            portfolio = self.portfolio_kelly(predictions, bankroll)

            # 각 포지션별 크기 계산
            position_sizes = {}
            total_combinations = 0

            for strategy, allocation in portfolio.strategy_allocations.items():
                # 해당 전략의 투자금
                strategy_investment = portfolio.total_investment * allocation

                # 조합 수 계산 (1000원 단위)
                combinations_count = int(
                    strategy_investment / self.cost_per_combination
                )
                combinations_count = max(1, combinations_count)  # 최소 1개

                position_sizes[strategy] = {
                    "combinations_count": combinations_count,
                    "investment_amount": combinations_count * self.cost_per_combination,
                    "allocation_percentage": allocation * 100,
                    "kelly_fraction": portfolio.kelly_fractions.get(strategy, 0.0),
                }

                total_combinations += combinations_count

            # 최대 포지션 수 제한
            if total_combinations > max_positions:
                # 비례적으로 축소
                scale_factor = max_positions / total_combinations
                for strategy in position_sizes:
                    position_sizes[strategy]["combinations_count"] = max(
                        1,
                        int(
                            position_sizes[strategy]["combinations_count"]
                            * scale_factor
                        ),
                    )
                    position_sizes[strategy]["investment_amount"] = (
                        position_sizes[strategy]["combinations_count"]
                        * self.cost_per_combination
                    )

            return {
                "position_sizes": position_sizes,
                "total_investment": sum(
                    pos["investment_amount"] for pos in position_sizes.values()
                ),
                "total_combinations": sum(
                    pos["combinations_count"] for pos in position_sizes.values()
                ),
                "expected_return": portfolio.expected_return,
                "risk_score": portfolio.risk_score,
                "recommendation": self._generate_position_recommendation(portfolio),
            }

        except Exception as e:
            self.logger.error(f"포지션 크기 계산 오류: {e}")
            return {}

    def _generate_position_recommendation(self, portfolio: PortfolioAllocation) -> str:
        """포지션 추천사항 생성"""
        try:
            if portfolio.expected_return <= 0:
                return "기대수익이 음수입니다. 투자를 권장하지 않습니다."
            elif portfolio.risk_score > 0.8:
                return "고위험 포트폴리오입니다. 신중한 투자를 권장합니다."
            elif portfolio.risk_score > 0.6:
                return "중위험 포트폴리오입니다. 적절한 리스크 관리가 필요합니다."
            elif portfolio.risk_score > 0.4:
                return "중저위험 포트폴리오입니다. 안정적인 투자가 가능합니다."
            else:
                return "저위험 포트폴리오입니다. 보수적인 투자 전략입니다."

        except Exception as e:
            self.logger.error(f"포지션 추천사항 생성 오류: {e}")
            return "추천사항 생성 중 오류가 발생했습니다."

    def dynamic_kelly_adjustment(
        self, historical_performance: List[float], current_kelly: float
    ) -> float:
        """동적 Kelly 조정"""
        try:
            if not historical_performance or len(historical_performance) < 3:
                return current_kelly

            # 최근 성과 분석
            recent_performance = historical_performance[-10:]  # 최근 10회

            # 성과 지표 계산
            win_rate = sum(1 for p in recent_performance if p > 0) / len(
                recent_performance
            )
            avg_return = np.mean(recent_performance)
            volatility = np.std(recent_performance)

            # 조정 팩터 계산
            adjustment_factor = 1.0

            # 승률 기반 조정
            if win_rate < 0.3:
                adjustment_factor *= 0.7  # 승률 낮으면 보수적으로
            elif win_rate > 0.7:
                adjustment_factor *= 1.2  # 승률 높으면 적극적으로

            # 변동성 기반 조정
            if volatility > 0.5:
                adjustment_factor *= 0.8  # 변동성 높으면 보수적으로
            elif volatility < 0.2:
                adjustment_factor *= 1.1  # 변동성 낮으면 적극적으로

            # 평균 수익 기반 조정
            if avg_return < -0.2:
                adjustment_factor *= 0.6  # 손실 지속시 매우 보수적으로
            elif avg_return > 0.2:
                adjustment_factor *= 1.3  # 수익 지속시 적극적으로

            # 최종 Kelly 비율 계산
            adjusted_kelly = current_kelly * adjustment_factor

            # 범위 제한
            adjusted_kelly = max(
                self.min_bet_fraction, min(adjusted_kelly, self.max_bet_fraction)
            )

            return adjusted_kelly

        except Exception as e:
            self.logger.error(f"동적 Kelly 조정 오류: {e}")
            return current_kelly * 0.8  # 오류 시 보수적으로

    def risk_parity_allocation(
        self, predictions: List[PredictionResult]
    ) -> Dict[str, float]:
        """리스크 패리티 배분"""
        try:
            if not predictions:
                return {}

            # 각 전략의 리스크 계산
            strategy_risks = {}
            for prediction in predictions:
                # 리스크 프록시: (1 - 신뢰도) + 변동성
                risk = (1 - prediction.confidence_score) + self._estimate_volatility(
                    prediction
                )
                strategy_risks[prediction.strategy_type] = max(
                    0.01, risk
                )  # 최소 리스크 설정

            # 리스크 역수로 가중치 계산
            risk_weights = {}
            total_inv_risk = sum(1 / risk for risk in strategy_risks.values())

            for strategy, risk in strategy_risks.items():
                risk_weights[strategy] = (1 / risk) / total_inv_risk

            return risk_weights

        except Exception as e:
            self.logger.error(f"리스크 패리티 배분 오류: {e}")
            return {}

    def _estimate_volatility(self, prediction: PredictionResult) -> float:
        """변동성 추정"""
        try:
            # 등급별 확률 분산을 변동성 프록시로 사용
            probabilities = list(prediction.grade_probabilities.values())
            if not probabilities:
                return 0.5

            mean_prob = np.mean(probabilities)
            variance = np.var(probabilities)

            # 정규화된 변동성 (0-1 범위)
            volatility = min(1.0, variance / (mean_prob + 0.001))

            return volatility

        except Exception as e:
            self.logger.error(f"변동성 추정 오류: {e}")
            return 0.5

    def generate_optimization_report(
        self, predictions: List[PredictionResult], bankroll: float
    ) -> Dict[str, Any]:
        """최적화 리포트 생성"""
        try:
            # 포트폴리오 최적화
            portfolio = self.portfolio_kelly(predictions, bankroll)

            # 포지션 크기 계산
            position_sizing = self.calculate_position_sizing(predictions, bankroll)

            # 리스크 패리티 배분
            risk_parity = self.risk_parity_allocation(predictions)

            # 리포트 구성
            report = {
                "optimization_summary": {
                    "total_investment": portfolio.total_investment,
                    "expected_return": portfolio.expected_return,
                    "risk_score": portfolio.risk_score,
                    "kelly_efficiency": sum(portfolio.kelly_fractions.values()),
                },
                "strategy_allocations": {
                    "kelly_based": portfolio.strategy_allocations,
                    "risk_parity": risk_parity,
                    "position_sizes": position_sizing.get("position_sizes", {}),
                },
                "risk_metrics": {
                    "portfolio_risk": portfolio.risk_score,
                    "diversification_ratio": self._calculate_diversification_ratio(
                        portfolio
                    ),
                    "max_drawdown_estimate": self._estimate_max_drawdown(portfolio),
                },
                "recommendations": {
                    "portfolio": self._generate_position_recommendation(portfolio),
                    "risk_management": self._generate_risk_management_advice(portfolio),
                    "rebalancing": self._generate_rebalancing_advice(portfolio),
                },
            }

            return report

        except Exception as e:
            self.logger.error(f"최적화 리포트 생성 오류: {e}")
            return {}

    def _calculate_diversification_ratio(self, portfolio: PortfolioAllocation) -> float:
        """다양화 비율 계산"""
        try:
            allocations = list(portfolio.strategy_allocations.values())
            if not allocations:
                return 0.0

            # 허핀달 지수 (낮을수록 다양화)
            herfindahl_index = sum(w**2 for w in allocations)

            # 다양화 비율 (1에 가까울수록 잘 다양화됨)
            diversification_ratio = (
                (1 - herfindahl_index) / (1 - 1 / len(allocations))
                if len(allocations) > 1
                else 0
            )

            return diversification_ratio

        except Exception as e:
            self.logger.error(f"다양화 비율 계산 오류: {e}")
            return 0.0

    def _estimate_max_drawdown(self, portfolio: PortfolioAllocation) -> float:
        """최대 손실 추정"""
        try:
            # 포트폴리오 리스크 기반 최대 손실 추정
            # 단순화된 계산: 리스크 점수 * 2 (최악의 경우 시나리오)
            max_drawdown = min(0.8, portfolio.risk_score * 2)

            return max_drawdown

        except Exception as e:
            self.logger.error(f"최대 손실 추정 오류: {e}")
            return 0.5

    def _generate_risk_management_advice(self, portfolio: PortfolioAllocation) -> str:
        """리스크 관리 조언 생성"""
        try:
            if portfolio.risk_score > 0.8:
                return "고위험 포트폴리오입니다. 스톱로스 설정과 포지션 크기 축소를 권장합니다."
            elif portfolio.risk_score > 0.6:
                return "중위험 포트폴리오입니다. 정기적인 리밸런싱과 성과 모니터링이 필요합니다."
            else:
                return "저위험 포트폴리오입니다. 현재 리스크 수준을 유지하세요."

        except Exception as e:
            self.logger.error(f"리스크 관리 조언 생성 오류: {e}")
            return "리스크 관리 조언 생성 중 오류가 발생했습니다."

    def _generate_rebalancing_advice(self, portfolio: PortfolioAllocation) -> str:
        """리밸런싱 조언 생성"""
        try:
            # Kelly 효율성 기반 조언
            kelly_efficiency = sum(portfolio.kelly_fractions.values())

            if kelly_efficiency < 0.1:
                return "Kelly 효율성이 낮습니다. 전략 재검토가 필요합니다."
            elif kelly_efficiency > 0.5:
                return "Kelly 효율성이 높습니다. 현재 배분을 유지하되 과도한 레버리지에 주의하세요."
            else:
                return "적절한 Kelly 효율성입니다. 월 1회 리밸런싱을 권장합니다."

        except Exception as e:
            self.logger.error(f"리밸런싱 조언 생성 오류: {e}")
            return "리밸런싱 조언 생성 중 오류가 발생했습니다."
