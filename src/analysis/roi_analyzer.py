"""
ROI 기반 분석기 모듈

이 모듈은 로또 번호 조합의 ROI(투자수익률), 기대값, 위험 분석 기능을 제공합니다.
"""

import numpy as np
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import Counter, defaultdict
import math
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

from src.analysis.base_analyzer import BaseAnalyzer
from src.shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from src.utils.unified_performance import performance_monitor
from src.utils.unified_config import ConfigProxy

logger = get_logger(__name__)


class ROIMetrics:
    """ROI 분석 결과를 담는 클래스"""

    def __init__(
        self,
        expected_value: float,
        roi: float,
        win_probability: float,
        payback_period: int,
        risk_score: float = 0.0,
    ):
        """
        ROIMetrics 초기화

        Args:
            expected_value: 기대값
            roi: 투자수익률
            win_probability: 당첨 확률
            payback_period: 투자금 회수 기간 (주)
            risk_score: 위험 점수
        """
        self.expected_value = expected_value
        self.roi = roi
        self.win_probability = win_probability
        self.payback_period = payback_period
        self.risk_score = risk_score

    def to_dict(self) -> Dict[str, float]:
        """
        ROI 메트릭을 사전 형태로 변환합니다.

        Returns:
            Dict[str, float]: ROI 메트릭 사전
        """
        return {
            "expected_value": self.expected_value,
            "roi": self.roi,
            "win_probability": self.win_probability,
            "payback_period": self.payback_period,
            "risk_score": self.risk_score,
        }


class ROIAnalyzer(BaseAnalyzer):
    """ROI 기반 분석기 클래스"""

    def __init__(self, config: dict):
        """
        ROIAnalyzer 초기화

        Args:
            config: 분석에 사용할 설정
        """
        super().__init__(config or {}, "roi")

        self.logger.info("ROIAnalyzer 초기화 완료")

        # 성능 최적화 시스템 초기화
        from src.utils.memory_manager import get_memory_manager

        self.memory_manager = get_memory_manager()

        # 티켓 가격 설정 (기본값 1,000원)
        self.ticket_price = 1000
        try:
            self.ticket_price = self.config["lottery"]["ticket_price"]
        except (KeyError, TypeError):
            logger.warning(
                "티켓 가격 설정을 찾을 수 없습니다. 기본값 1,000원을 사용합니다."
            )

        # 소수 목록 생성 (45 이하)
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

        # 상금 구조 설정
        self.prize_structure = {
            6: 2000000000,  # 1등 (평균 20억)
            5: 1000000,  # 2등 (평균 100만원)
            4: 50000,  # 3등 (평균 5만원)
            3: 5000,  # 4등 (평균 5천원)
        }

    @performance_monitor
    def analyze(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        과거 로또 당첨 번호의 ROI 관련 패턴을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: ROI 분석 결과
        """
        with self.memory_manager.allocation_scope():
            logger.info(f"ROI 기반 분석 시작: {len(historical_data)}개 데이터")

            # 캐시 키 생성
            cache_key = self._create_cache_key("roi_analysis", len(historical_data))

            # 캐시 확인
            cached_result = self._check_cache(cache_key)
            if cached_result:
                logger.info("캐시된 ROI 분석 결과 사용")
                return cached_result

            # 분석 수행
            results = {}

            # ROI 그룹 분석
            results["roi_pattern_groups"] = self.identify_roi_pattern_groups(
                historical_data
            )

            # 번호별 ROI 점수
            results["number_roi_scores"] = self.calculate_number_roi_scores(
                historical_data
            )

            # 패턴별 ROI 추세
            results["roi_trend_by_pattern"] = self.calculate_roi_trend_by_pattern(
                historical_data
            )

            # 위험도 매트릭스
            results["risk_matrix"] = self.calculate_risk_matrix(historical_data)

            # ROI 기반 번호 그룹
            results["roi_number_groups"] = self.analyze_roi_number_groups(
                historical_data
            )

            # 추가 기능들
            results["roi_group_score"] = self.calculate_roi_group_score(historical_data)
            results["roi_cluster_score"] = self.calculate_roi_cluster_score(
                historical_data
            )
            # 보너스 관련 플래그 제거됨
            results["roi_pattern_group_id"] = self.calculate_roi_pattern_group_id(
                historical_data
            )

            # 결과 캐싱
            self._save_to_cache(cache_key, results)

            logger.info("ROI 기반 분석 완료")
            return results

    def analyze_combination(
        self, historical_data: List[LotteryNumber], target_numbers: List[int]
    ) -> ROIMetrics:
        """
        특정 번호 조합의 ROI를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            target_numbers: 분석할 번호 조합(6개 번호)

        Returns:
            ROIMetrics: ROI 분석 결과
        """
        # 유효성 검사
        if len(target_numbers) != 6:
            raise ValueError("분석할 번호 조합은 6개 번호여야 합니다.")

        # 내부 분석 메서드 호출
        return self._analyze_impl(historical_data, target_numbers)

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> ROIMetrics:
        """
        특정 번호 조합의 ROI를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            args: 첫 번째 인자는 분석할 번호 조합(6개 번호)이어야 함
            kwargs: 추가 매개변수 (사용되지 않음)

        Returns:
            ROIMetrics: ROI 분석 결과
        """
        # 첫 번째 인자가 없으면 예외 발생
        if not args:
            raise ValueError("분석할 번호 조합이 제공되지 않았습니다.")

        # 첫 번째 인자를 target_numbers로 사용
        target_numbers = args[0]

        # 분석 수행
        win_probability = self._calculate_win_probability(target_numbers)
        expected_value = self._calculate_expected_value(historical_data, target_numbers)
        roi = (expected_value / self.ticket_price) - 1.0
        risk_score = self._calculate_risk_score(historical_data, target_numbers)

        # 투자금 회수 기간 계산 (주)
        payback_period = (
            int(self.ticket_price / expected_value) if expected_value > 0 else 999
        )

        return ROIMetrics(
            expected_value=expected_value,
            roi=roi,
            win_probability=win_probability,
            payback_period=payback_period,
            risk_score=risk_score,
        )

    def _calculate_win_probability(self, target_numbers: List[int]) -> float:
        """
        번호 조합의 당첨 확률을 계산합니다.

        Args:
            target_numbers: 분석할 번호 조합

        Returns:
            float: 당첨 확률
        """
        # 로또 6/45 당첨 확률: 1/8,145,060
        return 1.0 / 8145060

    def _calculate_expected_value(
        self, historical_data: List[LotteryNumber], target_numbers: List[int]
    ) -> float:
        """
        번호 조합의 기대값을 계산합니다.

        Args:
            historical_data: 과거 당첨 번호 목록
            target_numbers: 분석할 번호 조합

        Returns:
            float: 기대값
        """
        expected_value = 0.0

        # 과거 데이터를 기반으로 각 등수별 당첨 확률 추정
        total_matches = {3: 0, 4: 0, 5: 0, 6: 0}

        for draw in historical_data:
            matches = len(set(target_numbers) & set(draw.numbers))
            if matches >= 3:
                total_matches[matches] += 1

        total_draws = len(historical_data)

        # 각 등수별 기대값 계산
        for matches, count in total_matches.items():
            if count > 0:
                probability = count / total_draws
                prize = self.prize_structure.get(matches, 0)
                expected_value += probability * prize

        return expected_value

    def _calculate_risk_score(
        self, historical_data: List[LotteryNumber], target_numbers: List[int]
    ) -> float:
        """
        번호 조합의 위험 점수를 계산합니다.

        Args:
            historical_data: 과거 당첨 번호 목록
            target_numbers: 분석할 번호 조합

        Returns:
            float: 위험 점수 (0-1, 낮을수록 안전)
        """
        risk_factors = []

        # 1. 번호 분산도 위험
        number_variance = np.var(target_numbers)
        normalized_variance = min(number_variance / 200, 1.0)  # 정규화
        risk_factors.append(normalized_variance)

        # 2. 연속 번호 위험
        sorted_numbers = sorted(target_numbers)
        consecutive_count = 0
        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] - sorted_numbers[i - 1] == 1:
                consecutive_count += 1
        consecutive_risk = min(consecutive_count / 3, 1.0)  # 3개 이상이면 최대 위험
        risk_factors.append(consecutive_risk)

        # 3. 홀짝 불균형 위험
        odd_count = sum(1 for n in target_numbers if n % 2 == 1)
        balance_risk = abs(odd_count - 3) / 3  # 3:3이 이상적
        risk_factors.append(balance_risk)

        # 4. 과거 출현 빈도 위험
        frequency_risk = self._calculate_frequency_risk(historical_data, target_numbers)
        risk_factors.append(frequency_risk)

        # 전체 위험 점수 (가중 평균)
        weights = [0.3, 0.2, 0.2, 0.3]
        total_risk = sum(w * r for w, r in zip(weights, risk_factors))

        return min(total_risk, 1.0)

    def _calculate_frequency_risk(
        self, historical_data: List[LotteryNumber], target_numbers: List[int]
    ) -> float:
        """번호별 출현 빈도 기반 위험도 계산"""
        if not historical_data:
            return 0.5  # 데이터 없으면 중간 위험도

        # 각 번호의 출현 빈도 계산
        frequency_count = Counter()
        for draw in historical_data:
            frequency_count.update(draw.numbers)

        total_draws = len(historical_data)
        target_frequencies = [
            frequency_count.get(num, 0) / total_draws for num in target_numbers
        ]

        # 평균 출현 빈도와의 차이로 위험도 계산
        expected_frequency = 6 / 45  # 이론적 출현 빈도
        frequency_deviations = [
            abs(freq - expected_frequency) for freq in target_frequencies
        ]

        return min(np.mean(frequency_deviations) * 5, 1.0)  # 스케일링

    def calculate_number_roi_scores(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        번호별 ROI 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 번호별 ROI 점수
        """
        logger.info("번호별 ROI 점수 계산 중...")

        # 번호별 출현 통계
        number_stats = defaultdict(
            lambda: {"count": 0, "recent_count": 0, "positions": []}
        )

        recent_threshold = min(
            50, len(historical_data) // 4
        )  # 최근 25% 또는 최대 50회차

        for i, draw in enumerate(historical_data):
            is_recent = i >= len(historical_data) - recent_threshold

            for pos, number in enumerate(sorted(draw.numbers)):
                number_stats[number]["count"] += 1
                number_stats[number]["positions"].append(pos)

                if is_recent:
                    number_stats[number]["recent_count"] += 1

        # ROI 점수 계산
        roi_scores = {}
        total_draws = len(historical_data)

        for number in range(1, 46):
            stats = number_stats[number]

            # 기본 출현 빈도 점수
            frequency_score = stats["count"] / total_draws if total_draws > 0 else 0

            # 최근 출현 빈도 점수 (가중치 2배)
            recent_score = (
                stats["recent_count"] / recent_threshold if recent_threshold > 0 else 0
            )

            # 위치 다양성 점수
            position_diversity = (
                len(set(stats["positions"])) / 6 if stats["positions"] else 0
            )

            # 종합 ROI 점수
            roi_score = (
                frequency_score * 0.4 + recent_score * 0.4 + position_diversity * 0.2
            ) * 100

            roi_scores[f"number_{number}"] = {
                "roi_score": roi_score,
                "frequency": frequency_score,
                "recent_frequency": recent_score,
                "position_diversity": position_diversity,
                "total_appearances": stats["count"],
            }

        logger.info("번호별 ROI 점수 계산 완료")
        return {
            "individual_scores": roi_scores,
            "top_10_numbers": sorted(
                roi_scores.items(), key=lambda x: x[1]["roi_score"], reverse=True
            )[:10],
            "analysis_period": total_draws,
            "recent_period": recent_threshold,
        }

    def calculate_roi_trend_by_pattern(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        패턴별 ROI 추세를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 패턴별 ROI 추세
        """
        logger.info("패턴별 ROI 추세 계산 중...")

        # 패턴 그룹 정의
        pattern_groups = {
            "odd_dominant": [],  # 홀수 우세 (4개 이상)
            "even_dominant": [],  # 짝수 우세 (4개 이상)
            "low_numbers": [],  # 낮은 번호 우세 (1-22 범위에 4개 이상)
            "high_numbers": [],  # 높은 번호 우세 (23-45 범위에 4개 이상)
            "consecutive_numbers": [],  # 연속 번호 포함 (2개 이상)
            "wide_range": [],  # 넓은 번호 범위 (범위 > 22)
            "narrow_range": [],  # 좁은 번호 범위 (범위 <= 22)
            "prime_heavy": [],  # 소수 우세 (3개 이상)
            "multiples_of_3": [],  # 3의 배수 우세 (3개 이상)
        }

        # 각 그룹에 회차 분류
        for draw in historical_data:
            numbers = sorted(draw.numbers)
            draw_no = draw.draw_no

            # 홀짝 패턴
            odd_count = sum(1 for num in numbers if num % 2 == 1)
            if odd_count >= 4:
                pattern_groups["odd_dominant"].append(draw_no)
            elif odd_count <= 2:
                pattern_groups["even_dominant"].append(draw_no)

            # 고저 패턴
            low_count = sum(1 for num in numbers if num <= 22)
            if low_count >= 4:
                pattern_groups["low_numbers"].append(draw_no)
            elif low_count <= 2:
                pattern_groups["high_numbers"].append(draw_no)

            # 연속 번호 패턴
            consecutive_count = sum(
                1 for i in range(1, len(numbers)) if numbers[i] - numbers[i - 1] == 1
            )
            if consecutive_count >= 2:
                pattern_groups["consecutive_numbers"].append(draw_no)

            # 범위 패턴
            number_range = max(numbers) - min(numbers)
            if number_range > 22:
                pattern_groups["wide_range"].append(draw_no)
            else:
                pattern_groups["narrow_range"].append(draw_no)

            # 소수 패턴
            prime_count = sum(1 for num in numbers if num in self.primes)
            if prime_count >= 3:
                pattern_groups["prime_heavy"].append(draw_no)

            # 3의 배수 패턴
            multiple_of_3_count = sum(1 for num in numbers if num % 3 == 0)
            if multiple_of_3_count >= 3:
                pattern_groups["multiples_of_3"].append(draw_no)

        # 패턴별 ROI 추세 계산
        result = {}
        for pattern, draws in pattern_groups.items():
            if len(draws) > 0:
                trend_score = self.calculate_roi_trend(historical_data, draws)
                result[f"{pattern}_roi_trend"] = {
                    "trend_score": trend_score,
                    "occurrence_count": len(draws),
                    "frequency": len(draws) / len(historical_data),
                    "recent_occurrences": len(
                        [d for d in draws if d > len(historical_data) - 20]
                    ),
                }

        logger.info("패턴별 ROI 추세 계산 완료")
        return result

    def calculate_roi_trend(
        self, historical_data: List[LotteryNumber], target_draws: List[int]
    ) -> float:
        """특정 회차들의 ROI 추세를 계산합니다."""
        if not target_draws:
            return 0.0

        # 시간 가중 점수 계산
        total_draws = len(historical_data)
        weighted_score = 0.0
        total_weight = 0.0

        for draw_no in target_draws:
            # 최근일수록 높은 가중치
            position_from_end = total_draws - draw_no
            weight = 1.0 / (1.0 + position_from_end * 0.1)  # 지수적 감소

            weighted_score += weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def calculate_risk_matrix(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """위험도 매트릭스를 계산합니다."""
        logger.info("위험도 매트릭스 계산 중...")

        risk_matrix = {}

        # 번호별 위험도 계산
        for number in range(1, 46):
            combinations_with_number = [
                draw for draw in historical_data if number in draw.numbers
            ]

            if combinations_with_number:
                # 해당 번호가 포함된 조합들의 특성 분석
                risk_factors = []

                for draw in combinations_with_number:
                    numbers = draw.numbers

                    # 분산 위험
                    variance_risk = min(np.var(numbers) / 200, 1.0)

                    # 연속성 위험
                    sorted_nums = sorted(numbers)
                    consecutive_risk = (
                        sum(
                            1
                            for i in range(1, len(sorted_nums))
                            if sorted_nums[i] - sorted_nums[i - 1] == 1
                        )
                        / 3
                    )

                    risk_factors.append((variance_risk + consecutive_risk) / 2)

                risk_matrix[f"number_{number}"] = {
                    "average_risk": np.mean(risk_factors),
                    "risk_std": np.std(risk_factors),
                    "sample_size": len(combinations_with_number),
                }
            else:
                risk_matrix[f"number_{number}"] = {
                    "average_risk": 0.5,  # 중간 위험도
                    "risk_std": 0.0,
                    "sample_size": 0,
                }

        logger.info("위험도 매트릭스 계산 완료")
        return risk_matrix

    def analyze_roi_number_groups(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """ROI 기반 번호 그룹을 분석합니다."""
        logger.info("ROI 기반 번호 그룹 분석 중...")

        # 번호별 ROI 점수 가져오기
        roi_scores = self.calculate_number_roi_scores(historical_data)
        individual_scores = roi_scores["individual_scores"]

        # 점수 기준으로 그룹 분류
        high_roi_numbers = []
        medium_roi_numbers = []
        low_roi_numbers = []

        score_threshold_high = 60
        score_threshold_low = 40

        for number_key, score_data in individual_scores.items():
            number = int(number_key.split("_")[1])
            roi_score = score_data["roi_score"]

            if roi_score >= score_threshold_high:
                high_roi_numbers.append(number)
            elif roi_score >= score_threshold_low:
                medium_roi_numbers.append(number)
            else:
                low_roi_numbers.append(number)

        logger.info("ROI 기반 번호 그룹 분석 완료")
        return {
            "high_roi_group": {
                "numbers": sorted(high_roi_numbers),
                "count": len(high_roi_numbers),
                "average_score": (
                    np.mean(
                        [
                            individual_scores[f"number_{n}"]["roi_score"]
                            for n in high_roi_numbers
                        ]
                    )
                    if high_roi_numbers
                    else 0
                ),
            },
            "medium_roi_group": {
                "numbers": sorted(medium_roi_numbers),
                "count": len(medium_roi_numbers),
                "average_score": (
                    np.mean(
                        [
                            individual_scores[f"number_{n}"]["roi_score"]
                            for n in medium_roi_numbers
                        ]
                    )
                    if medium_roi_numbers
                    else 0
                ),
            },
            "low_roi_group": {
                "numbers": sorted(low_roi_numbers),
                "count": len(low_roi_numbers),
                "average_score": (
                    np.mean(
                        [
                            individual_scores[f"number_{n}"]["roi_score"]
                            for n in low_roi_numbers
                        ]
                    )
                    if low_roi_numbers
                    else 0
                ),
            },
        }

    def identify_roi_pattern_groups(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """ROI 패턴 그룹을 식별합니다."""
        logger.info("ROI 패턴 그룹 식별 중...")

        pattern_performance = {}

        # 다양한 패턴의 성과 추적
        patterns = {
            "balanced_odd_even": lambda nums: abs(
                sum(1 for n in nums if n % 2 == 1) - 3
            )
            <= 1,
            "wide_range": lambda nums: max(nums) - min(nums) > 25,
            "has_consecutive": lambda nums: any(
                nums[i + 1] - nums[i] == 1 for i in range(len(nums) - 1)
            ),
            "prime_rich": lambda nums: sum(1 for n in nums if n in self.primes) >= 3,
            "decade_spread": lambda nums: len(set(n // 10 for n in nums)) >= 4,
        }

        for pattern_name, pattern_func in patterns.items():
            matching_draws = []

            for draw in historical_data:
                sorted_numbers = sorted(draw.numbers)
                if pattern_func(sorted_numbers):
                    matching_draws.append(draw.draw_no)

            if matching_draws:
                # 패턴의 최근 성과 계산
                recent_performance = len(
                    [d for d in matching_draws if d > len(historical_data) - 30]
                )
                total_performance = len(matching_draws)

                pattern_performance[pattern_name] = {
                    "total_occurrences": total_performance,
                    "recent_occurrences": recent_performance,
                    "frequency": total_performance / len(historical_data),
                    "recent_frequency": recent_performance
                    / min(30, len(historical_data)),
                    "roi_trend": self.calculate_roi_trend(
                        historical_data, matching_draws
                    ),
                }

        logger.info("ROI 패턴 그룹 식별 완료")
        return pattern_performance

    def calculate_roi_group_score(self, historical_data: List[LotteryNumber]) -> float:
        """ROI 그룹별 점수를 계산합니다."""
        with performance_monitor("calculate_roi_group_score"):
            logger.info("ROI 그룹별 점수 계산 중...")

            if len(historical_data) < 50:
                logger.warning("ROI 그룹별 점수 계산을 위한 충분한 데이터가 없습니다.")
                return 0.0

            # ROI 그룹 분석 결과 가져오기
            roi_groups = self.analyze_roi_number_groups(historical_data)

            # 각 그룹의 가중 점수 계산
            high_roi_weight = 0.5
            medium_roi_weight = 0.3
            low_roi_weight = 0.2

            total_score = (
                roi_groups["high_roi_group"]["average_score"] * high_roi_weight
                + roi_groups["medium_roi_group"]["average_score"] * medium_roi_weight
                + roi_groups["low_roi_group"]["average_score"] * low_roi_weight
            )

            logger.info("ROI 그룹별 점수 계산 완료")
            return total_score

    def calculate_roi_cluster_score(
        self, historical_data: List[LotteryNumber]
    ) -> float:
        """ROI 클러스터 점수를 계산합니다."""
        with performance_monitor("calculate_roi_cluster_score"):
            logger.info("ROI 클러스터 점수 계산 중...")

            if len(historical_data) < 50:
                logger.warning(
                    "ROI 클러스터 점수 계산을 위한 충분한 데이터가 없습니다."
                )
                return 0.0

            # 패턴별 ROI 추세 가져오기
            roi_trends = self.calculate_roi_trend_by_pattern(historical_data)

            # 클러스터 점수 계산 (각 패턴의 가중 평균)
            pattern_weights = {
                "odd_dominant_roi_trend": 0.15,
                "even_dominant_roi_trend": 0.15,
                "low_numbers_roi_trend": 0.15,
                "high_numbers_roi_trend": 0.15,
                "consecutive_numbers_roi_trend": 0.10,
                "wide_range_roi_trend": 0.10,
                "narrow_range_roi_trend": 0.10,
                "prime_heavy_roi_trend": 0.10,
            }

            total_score = 0.0
            for pattern, weight in pattern_weights.items():
                if pattern in roi_trends:
                    total_score += roi_trends[pattern]["trend_score"] * weight

            logger.info("ROI 클러스터 점수 계산 완료")
            return total_score

    # calculate_low_risk_bonus_flag 메서드 완전 삭제됨

    def calculate_roi_pattern_group_id(
        self, historical_data: List[LotteryNumber]
    ) -> int:
        """ROI 패턴 그룹 ID를 계산합니다."""
        logger.info("ROI 패턴 그룹 ID 계산 중...")

        # 패턴 그룹 성과 분석
        pattern_groups = self.identify_roi_pattern_groups(historical_data)

        if not pattern_groups:
            return 0

        # 가장 성과가 좋은 패턴 그룹 찾기
        best_pattern = max(
            pattern_groups.items(),
            key=lambda x: x[1]["roi_trend"] * x[1]["recent_frequency"],
        )

        # 패턴명을 ID로 변환
        pattern_to_id = {
            "balanced_odd_even": 1,
            "wide_range": 2,
            "has_consecutive": 3,
            "prime_rich": 4,
            "decade_spread": 5,
        }

        group_id = pattern_to_id.get(best_pattern[0], 0)

        logger.info(f"ROI 패턴 그룹 ID: {group_id} (패턴: {best_pattern[0]})")
        return group_id
