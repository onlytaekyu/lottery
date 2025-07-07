"""
A/B 테스트 프레임워크 (A/B Testing Framework)

전략 간 성과 비교와 통계적 유의성 검정을 위한 A/B 테스트 시스템입니다.

주요 기능:
- 전략 간 성과 비교
- 통계적 유의성 검정
- 다중 전략 비교 (A/B/C/D 테스트)
- 베이지안 A/B 테스트
- 효과 크기 계산
- 테스트 종료 조건 판정
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import math
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import warnings

from ..shared.types import LotteryNumber, ModelPrediction
from ..utils.unified_logging import get_logger
from ..utils.cache_paths import get_cache_dir
from ..monitoring.performance_tracker import PerformanceMetrics

logger = get_logger(__name__)


@dataclass
class ABTestConfig:
    """A/B 테스트 설정"""

    significance_level: float = 0.05  # 유의수준
    power: float = 0.8  # 검정력
    minimum_sample_size: int = 30  # 최소 표본 크기
    maximum_duration_days: int = 30  # 최대 테스트 기간
    effect_size_threshold: float = 0.1  # 최소 효과 크기
    confidence_level: float = 0.95  # 신뢰도
    early_stopping: bool = True  # 조기 종료 허용
    bayesian_threshold: float = 0.95  # 베이지안 확률 임계값


@dataclass
class ABTestResult:
    """A/B 테스트 결과"""

    test_id: str
    strategy_a: str
    strategy_b: str
    sample_size_a: int
    sample_size_b: int
    mean_performance_a: float
    mean_performance_b: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    statistical_power: float
    test_duration_days: int
    recommendation: str
    bayesian_probability: Optional[float] = None


@dataclass
class MultipleTestResult:
    """다중 비교 테스트 결과"""

    test_id: str
    strategies: List[str]
    sample_sizes: Dict[str, int]
    mean_performances: Dict[str, float]
    pairwise_comparisons: List[ABTestResult]
    overall_p_value: float
    best_strategy: str
    effect_sizes: Dict[str, float]
    recommendations: List[str]


class StatisticalTests:
    """통계적 검정 도구"""

    @staticmethod
    def t_test(group_a: List[float], group_b: List[float]) -> Tuple[float, float]:
        """독립 표본 t-검정"""
        try:
            statistic, p_value = ttest_ind(group_a, group_b, equal_var=False)
            return float(statistic), float(p_value)
        except Exception:
            return 0.0, 1.0

    @staticmethod
    def mann_whitney_u(
        group_a: List[float], group_b: List[float]
    ) -> Tuple[float, float]:
        """Mann-Whitney U 검정 (비모수)"""
        try:
            statistic, p_value = mannwhitneyu(group_a, group_b, alternative="two-sided")
            return float(statistic), float(p_value)
        except Exception:
            return 0.0, 1.0

    @staticmethod
    def cohens_d(group_a: List[float], group_b: List[float]) -> float:
        """Cohen's d 효과 크기 계산"""
        try:
            mean_a, mean_b = np.mean(group_a), np.mean(group_b)
            std_a, std_b = np.std(group_a, ddof=1), np.std(group_b, ddof=1)
            n_a, n_b = len(group_a), len(group_b)

            # 합동 표준편차
            pooled_std = math.sqrt(
                ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
            )

            if pooled_std == 0:
                return 0.0

            return (mean_a - mean_b) / pooled_std
        except Exception:
            return 0.0

    @staticmethod
    def calculate_confidence_interval(
        group_a: List[float], group_b: List[float], confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """평균 차이의 신뢰구간 계산"""
        try:
            mean_diff = np.mean(group_a) - np.mean(group_b)

            # 표준오차 계산
            se_a = np.std(group_a, ddof=1) / math.sqrt(len(group_a))
            se_b = np.std(group_b, ddof=1) / math.sqrt(len(group_b))
            se_diff = math.sqrt(se_a**2 + se_b**2)

            # t-분포 임계값
            df = len(group_a) + len(group_b) - 2
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, df)

            # 신뢰구간
            margin_error = t_critical * se_diff
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error

            return (ci_lower, ci_upper)
        except Exception:
            return (0.0, 0.0)

    @staticmethod
    def calculate_power(
        effect_size: float, sample_size: int, alpha: float = 0.05
    ) -> float:
        """검정력 계산"""
        try:
            from scipy.stats import norm

            # 비중심성 매개변수
            ncp = effect_size * math.sqrt(sample_size / 2)

            # 임계값
            z_alpha = norm.ppf(1 - alpha / 2)

            # 검정력
            power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)

            return max(0.0, min(1.0, power))
        except Exception:
            return 0.0


class BayesianABTest:
    """베이지안 A/B 테스트"""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def calculate_bayesian_probability(
        self, successes_a: int, trials_a: int, successes_b: int, trials_b: int
    ) -> float:
        """베이지안 확률 계산 (A가 B보다 좋을 확률)"""
        try:
            # 베타 분포 매개변수
            alpha_a = self.prior_alpha + successes_a
            beta_a = self.prior_beta + trials_a - successes_a

            alpha_b = self.prior_alpha + successes_b
            beta_b = self.prior_beta + trials_b - successes_b

            # 몬테카를로 시뮬레이션
            n_samples = 10000
            samples_a = np.random.beta(alpha_a, beta_a, n_samples)
            samples_b = np.random.beta(alpha_b, beta_b, n_samples)

            # A가 B보다 좋을 확률
            prob_a_better = np.mean(samples_a > samples_b)

            return float(prob_a_better)
        except Exception:
            return 0.5

    def calculate_credible_interval(
        self, successes: int, trials: int, credible_level: float = 0.95
    ) -> Tuple[float, float]:
        """신용구간 계산"""
        try:
            alpha = self.prior_alpha + successes
            beta = self.prior_beta + trials - successes

            lower_percentile = (1 - credible_level) / 2 * 100
            upper_percentile = (1 + credible_level) / 2 * 100

            ci_lower = stats.beta.ppf(lower_percentile / 100, alpha, beta)
            ci_upper = stats.beta.ppf(upper_percentile / 100, alpha, beta)

            return (ci_lower, ci_upper)
        except Exception:
            return (0.0, 1.0)


class ABTestingFramework:
    """A/B 테스트 프레임워크"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        A/B 테스트 프레임워크 초기화

        Args:
            config: 설정 객체
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # 테스트 설정
        test_config = self.config.get("ab_test", {})
        self.test_config = ABTestConfig(
            significance_level=test_config.get("significance_level", 0.05),
            power=test_config.get("power", 0.8),
            minimum_sample_size=test_config.get("minimum_sample_size", 30),
            maximum_duration_days=test_config.get("maximum_duration_days", 30),
            effect_size_threshold=test_config.get("effect_size_threshold", 0.1),
            confidence_level=test_config.get("confidence_level", 0.95),
            early_stopping=test_config.get("early_stopping", True),
            bayesian_threshold=test_config.get("bayesian_threshold", 0.95),
        )

        # 통계 도구
        self.stats = StatisticalTests()
        self.bayesian = BayesianABTest()

        # 테스트 이력
        self.test_history = []
        self.active_tests = {}

        # 데이터 저장 경로
        self.cache_dir = Path(get_cache_dir()) / "ab_testing"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("A/B 테스트 프레임워크 초기화 완료")

    def run_ab_test(
        self,
        strategy_a: str,
        strategy_b: str,
        performance_data_a: List[float],
        performance_data_b: List[float],
        test_id: Optional[str] = None,
    ) -> ABTestResult:
        """A/B 테스트 실행"""
        try:
            if test_id is None:
                test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.logger.info(f"A/B 테스트 실행: {strategy_a} vs {strategy_b}")

            # 데이터 검증
            if len(performance_data_a) < self.test_config.minimum_sample_size:
                raise ValueError(
                    f"전략 A의 표본 크기가 부족합니다: {len(performance_data_a)} < {self.test_config.minimum_sample_size}"
                )

            if len(performance_data_b) < self.test_config.minimum_sample_size:
                raise ValueError(
                    f"전략 B의 표본 크기가 부족합니다: {len(performance_data_b)} < {self.test_config.minimum_sample_size}"
                )

            # 기본 통계
            mean_a = np.mean(performance_data_a)
            mean_b = np.mean(performance_data_b)

            # 정규성 검정
            is_normal_a = self._test_normality(performance_data_a)
            is_normal_b = self._test_normality(performance_data_b)

            # 적절한 검정 선택
            if is_normal_a and is_normal_b:
                statistic, p_value = self.stats.t_test(
                    performance_data_a, performance_data_b
                )
                test_type = "t-test"
            else:
                statistic, p_value = self.stats.mann_whitney_u(
                    performance_data_a, performance_data_b
                )
                test_type = "Mann-Whitney U"

            # 효과 크기 계산
            effect_size = self.stats.cohens_d(performance_data_a, performance_data_b)

            # 신뢰구간 계산
            ci_lower, ci_upper = self.stats.calculate_confidence_interval(
                performance_data_a,
                performance_data_b,
                self.test_config.confidence_level,
            )

            # 유의성 판정
            is_significant = p_value < self.test_config.significance_level

            # 검정력 계산
            sample_size = min(len(performance_data_a), len(performance_data_b))
            statistical_power = self.stats.calculate_power(
                abs(effect_size), sample_size, self.test_config.significance_level
            )

            # 베이지안 확률 (승률 기반)
            wins_a = sum(1 for x in performance_data_a if x > 0)
            wins_b = sum(1 for x in performance_data_b if x > 0)

            bayesian_prob = self.bayesian.calculate_bayesian_probability(
                wins_a, len(performance_data_a), wins_b, len(performance_data_b)
            )

            # 추천사항 생성
            recommendation = self._generate_ab_recommendation(
                strategy_a,
                strategy_b,
                mean_a,
                mean_b,
                is_significant,
                effect_size,
                statistical_power,
                bayesian_prob,
            )

            # 결과 객체 생성
            result = ABTestResult(
                test_id=test_id,
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                sample_size_a=len(performance_data_a),
                sample_size_b=len(performance_data_b),
                mean_performance_a=mean_a,
                mean_performance_b=mean_b,
                effect_size=effect_size,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                is_significant=is_significant,
                statistical_power=statistical_power,
                test_duration_days=0,  # 실제 구현에서는 계산
                recommendation=recommendation,
                bayesian_probability=bayesian_prob,
            )

            # 이력 저장
            self.test_history.append(result)

            self.logger.info(f"A/B 테스트 완료: {test_id}")
            return result

        except Exception as e:
            self.logger.error(f"A/B 테스트 실행 중 오류: {str(e)}")
            return ABTestResult(
                test_id=test_id or "error",
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                sample_size_a=0,
                sample_size_b=0,
                mean_performance_a=0.0,
                mean_performance_b=0.0,
                effect_size=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                statistical_power=0.0,
                test_duration_days=0,
                recommendation="테스트 실행 실패",
            )

    def run_multiple_comparison(
        self, strategies: Dict[str, List[float]], test_id: Optional[str] = None
    ) -> MultipleTestResult:
        """다중 전략 비교 테스트"""
        try:
            if test_id is None:
                test_id = f"multi_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.logger.info(f"다중 비교 테스트 실행: {list(strategies.keys())}")

            strategy_names = list(strategies.keys())
            if len(strategy_names) < 2:
                raise ValueError("최소 2개 이상의 전략이 필요합니다")

            # 기본 통계
            sample_sizes = {name: len(data) for name, data in strategies.items()}
            mean_performances = {
                name: np.mean(data) for name, data in strategies.items()
            }

            # 쌍별 비교
            pairwise_comparisons = []
            for i in range(len(strategy_names)):
                for j in range(i + 1, len(strategy_names)):
                    strategy_a = strategy_names[i]
                    strategy_b = strategy_names[j]

                    comparison = self.run_ab_test(
                        strategy_a,
                        strategy_b,
                        strategies[strategy_a],
                        strategies[strategy_b],
                        f"{test_id}_pair_{i}_{j}",
                    )
                    pairwise_comparisons.append(comparison)

            # 전체 F-검정 (ANOVA)
            try:
                f_statistic, overall_p_value = stats.f_oneway(*strategies.values())
                overall_p_value = float(overall_p_value)
            except Exception:
                overall_p_value = 1.0

            # 최고 전략 선택
            best_strategy = max(mean_performances.items(), key=lambda x: x[1])[0]

            # 효과 크기 계산 (최고 전략 대비)
            effect_sizes = {}
            best_data = strategies[best_strategy]

            for name, data in strategies.items():
                if name != best_strategy:
                    effect_size = self.stats.cohens_d(best_data, data)
                    effect_sizes[name] = effect_size
                else:
                    effect_sizes[name] = 0.0

            # 추천사항 생성
            recommendations = self._generate_multiple_recommendations(
                strategy_names,
                mean_performances,
                pairwise_comparisons,
                overall_p_value,
                best_strategy,
            )

            # 결과 객체 생성
            result = MultipleTestResult(
                test_id=test_id,
                strategies=strategy_names,
                sample_sizes=sample_sizes,
                mean_performances=mean_performances,
                pairwise_comparisons=pairwise_comparisons,
                overall_p_value=overall_p_value,
                best_strategy=best_strategy,
                effect_sizes=effect_sizes,
                recommendations=recommendations,
            )

            self.logger.info(f"다중 비교 테스트 완료: {test_id}")
            return result

        except Exception as e:
            self.logger.error(f"다중 비교 테스트 중 오류: {str(e)}")
            return MultipleTestResult(
                test_id=test_id or "error",
                strategies=[],
                sample_sizes={},
                mean_performances={},
                pairwise_comparisons=[],
                overall_p_value=1.0,
                best_strategy="",
                effect_sizes={},
                recommendations=["테스트 실행 실패"],
            )

    def _test_normality(self, data: List[float], alpha: float = 0.05) -> bool:
        """정규성 검정"""
        try:
            if len(data) < 8:
                return False  # 표본이 너무 작음

            # Shapiro-Wilk 검정
            statistic, p_value = stats.shapiro(data)
            return p_value > alpha
        except Exception:
            return False

    def _generate_ab_recommendation(
        self,
        strategy_a: str,
        strategy_b: str,
        mean_a: float,
        mean_b: float,
        is_significant: bool,
        effect_size: float,
        statistical_power: float,
        bayesian_prob: float,
    ) -> str:
        """A/B 테스트 추천사항 생성"""
        try:
            if not is_significant:
                if statistical_power < 0.8:
                    return f"통계적으로 유의한 차이가 없습니다. 검정력이 낮으므로({statistical_power:.2f}) 더 많은 데이터가 필요합니다."
                else:
                    return f"통계적으로 유의한 차이가 없습니다. 두 전략의 성능이 유사합니다."

            # 유의한 차이가 있는 경우
            better_strategy = strategy_a if mean_a > mean_b else strategy_b
            worse_strategy = strategy_b if mean_a > mean_b else strategy_a

            # 효과 크기 해석
            if abs(effect_size) < 0.2:
                effect_desc = "작은"
            elif abs(effect_size) < 0.5:
                effect_desc = "중간"
            elif abs(effect_size) < 0.8:
                effect_desc = "큰"
            else:
                effect_desc = "매우 큰"

            # 베이지안 확률 고려
            bayesian_desc = ""
            if bayesian_prob > 0.95:
                bayesian_desc = " (베이지안 분석에서도 95% 이상 확신)"
            elif bayesian_prob > 0.9:
                bayesian_desc = " (베이지안 분석에서도 90% 이상 확신)"

            return f"{better_strategy}이 {worse_strategy}보다 통계적으로 유의하게 우수합니다. 효과 크기: {effect_desc} (Cohen's d = {effect_size:.3f}){bayesian_desc}"

        except Exception:
            return "추천사항 생성 중 오류가 발생했습니다."

    def _generate_multiple_recommendations(
        self,
        strategies: List[str],
        mean_performances: Dict[str, float],
        pairwise_comparisons: List[ABTestResult],
        overall_p_value: float,
        best_strategy: str,
    ) -> List[str]:
        """다중 비교 추천사항 생성"""
        recommendations = []

        try:
            # 전체 차이 유의성
            if overall_p_value < self.test_config.significance_level:
                recommendations.append(
                    f"전략 간 전체적으로 유의한 차이가 있습니다 (p = {overall_p_value:.4f})"
                )
            else:
                recommendations.append(
                    f"전략 간 전체적으로 유의한 차이가 없습니다 (p = {overall_p_value:.4f})"
                )

            # 최고 전략
            best_performance = mean_performances[best_strategy]
            recommendations.append(
                f"최고 성과 전략: {best_strategy} (평균 성과: {best_performance:.4f})"
            )

            # 유의한 쌍별 비교
            significant_pairs = [
                comp for comp in pairwise_comparisons if comp.is_significant
            ]

            if significant_pairs:
                recommendations.append(
                    f"총 {len(significant_pairs)}개의 전략 쌍에서 유의한 차이 발견"
                )

                # 상위 3개 유의한 차이
                sorted_pairs = sorted(
                    significant_pairs, key=lambda x: abs(x.effect_size), reverse=True
                )
                for i, pair in enumerate(sorted_pairs[:3]):
                    better = (
                        pair.strategy_a
                        if pair.mean_performance_a > pair.mean_performance_b
                        else pair.strategy_b
                    )
                    worse = (
                        pair.strategy_b
                        if pair.mean_performance_a > pair.mean_performance_b
                        else pair.strategy_a
                    )
                    recommendations.append(
                        f"{i+1}. {better} > {worse} (효과 크기: {abs(pair.effect_size):.3f})"
                    )
            else:
                recommendations.append("유의한 전략 쌍 차이를 발견하지 못했습니다")

            # 실용적 추천
            if len(strategies) > 3:
                recommendations.append(
                    "전략이 많습니다. 상위 2-3개 전략에 집중하는 것을 권장합니다"
                )

        except Exception:
            recommendations.append("추천사항 생성 중 오류가 발생했습니다")

        return recommendations

    def calculate_required_sample_size(
        self, effect_size: float, power: float = 0.8, alpha: float = 0.05
    ) -> int:
        """필요 표본 크기 계산"""
        try:
            # Cohen의 공식 사용
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)

            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

            return max(self.test_config.minimum_sample_size, int(math.ceil(n)))

        except Exception:
            return self.test_config.minimum_sample_size

    def should_stop_test_early(
        self,
        performance_data_a: List[float],
        performance_data_b: List[float],
        current_duration_days: int,
    ) -> Tuple[bool, str]:
        """조기 종료 조건 확인"""
        try:
            if not self.test_config.early_stopping:
                return False, "조기 종료가 비활성화되어 있습니다"

            # 최소 표본 크기 확인
            if (
                len(performance_data_a) < self.test_config.minimum_sample_size
                or len(performance_data_b) < self.test_config.minimum_sample_size
            ):
                return False, "최소 표본 크기에 도달하지 않았습니다"

            # 최대 기간 확인
            if current_duration_days >= self.test_config.maximum_duration_days:
                return True, "최대 테스트 기간에 도달했습니다"

            # 통계적 유의성 확인
            _, p_value = self.stats.t_test(performance_data_a, performance_data_b)

            if p_value < self.test_config.significance_level:
                # 효과 크기 확인
                effect_size = abs(
                    self.stats.cohens_d(performance_data_a, performance_data_b)
                )

                if effect_size >= self.test_config.effect_size_threshold:
                    return (
                        True,
                        f"통계적 유의성과 실용적 효과 크기 달성 (p={p_value:.4f}, d={effect_size:.3f})",
                    )

            # 베이지안 조건 확인
            wins_a = sum(1 for x in performance_data_a if x > 0)
            wins_b = sum(1 for x in performance_data_b if x > 0)

            bayesian_prob = self.bayesian.calculate_bayesian_probability(
                wins_a, len(performance_data_a), wins_b, len(performance_data_b)
            )

            if bayesian_prob > self.test_config.bayesian_threshold or bayesian_prob < (
                1 - self.test_config.bayesian_threshold
            ):
                return True, f"베이지안 확률이 임계값 도달 ({bayesian_prob:.3f})"

            return False, "조기 종료 조건을 만족하지 않습니다"

        except Exception as e:
            return False, f"조기 종료 확인 중 오류: {str(e)}"

    def get_test_summary(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """테스트 요약 정보 반환"""
        try:
            if test_id:
                # 특정 테스트 요약
                test_results = [t for t in self.test_history if t.test_id == test_id]
                if not test_results:
                    return {"error": f"테스트 ID {test_id}를 찾을 수 없습니다"}

                result = test_results[0]
                return {
                    "test_id": result.test_id,
                    "strategies": [result.strategy_a, result.strategy_b],
                    "sample_sizes": [result.sample_size_a, result.sample_size_b],
                    "mean_performances": [
                        result.mean_performance_a,
                        result.mean_performance_b,
                    ],
                    "effect_size": result.effect_size,
                    "p_value": result.p_value,
                    "is_significant": result.is_significant,
                    "recommendation": result.recommendation,
                    "bayesian_probability": result.bayesian_probability,
                }
            else:
                # 전체 테스트 요약
                if not self.test_history:
                    return {"message": "테스트 이력이 없습니다"}

                total_tests = len(self.test_history)
                significant_tests = sum(
                    1 for t in self.test_history if t.is_significant
                )

                # 평균 효과 크기
                avg_effect_size = np.mean(
                    [abs(t.effect_size) for t in self.test_history]
                )

                # 최근 테스트
                recent_test = self.test_history[-1]

                return {
                    "total_tests": total_tests,
                    "significant_tests": significant_tests,
                    "significance_rate": (
                        significant_tests / total_tests if total_tests > 0 else 0
                    ),
                    "average_effect_size": avg_effect_size,
                    "recent_test": {
                        "test_id": recent_test.test_id,
                        "strategies": [recent_test.strategy_a, recent_test.strategy_b],
                        "is_significant": recent_test.is_significant,
                        "recommendation": recent_test.recommendation,
                    },
                }

        except Exception as e:
            return {"error": f"테스트 요약 생성 중 오류: {str(e)}"}

    def save_test_results(self) -> None:
        """테스트 결과 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 테스트 이력 저장
            results_data = []
            for result in self.test_history:
                results_data.append(asdict(result))

            results_file = self.cache_dir / f"ab_test_results_{timestamp}.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"A/B 테스트 결과 저장 완료: {results_file}")

        except Exception as e:
            self.logger.error(f"테스트 결과 저장 중 오류: {str(e)}")

    def load_test_results(self, date: Optional[str] = None) -> bool:
        """테스트 결과 로드"""
        try:
            if date:
                pattern = f"ab_test_results_{date}*.json"
            else:
                pattern = "ab_test_results_*.json"

            result_files = list(self.cache_dir.glob(pattern))
            if not result_files:
                self.logger.warning(f"테스트 결과 파일을 찾을 수 없습니다: {pattern}")
                return False

            # 최신 파일 선택
            latest_file = max(result_files, key=lambda x: x.name)

            with open(latest_file, "r", encoding="utf-8") as f:
                results_data = json.load(f)

            # 결과 복원
            self.test_history = []
            for data in results_data:
                # 튜플 복원
                if "confidence_interval" in data and isinstance(
                    data["confidence_interval"], list
                ):
                    data["confidence_interval"] = tuple(data["confidence_interval"])

                result = ABTestResult(**data)
                self.test_history.append(result)

            self.logger.info(f"A/B 테스트 결과 로드 완료: {latest_file}")
            return True

        except Exception as e:
            self.logger.error(f"테스트 결과 로드 중 오류: {str(e)}")
            return False
