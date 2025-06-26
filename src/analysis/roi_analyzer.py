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
from src.utils.error_handler import get_logger

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
        super().__init__(config, analyzer_type="roi")

        # 티켓 가격 설정 (기본값 1,000원)
        self.ticket_price = 1000
        if (
            isinstance(config, dict)
            and "lottery" in config
            and "ticket_price" in config["lottery"]
        ):
            self.ticket_price = config["lottery"]["ticket_price"]

        # 소수 목록 생성 (45 이하)
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

    def analyze(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        과거 로또 당첨 번호의 ROI 관련 패턴을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: ROI 분석 결과
        """
        # 캐시 키 생성
        cache_key = self._create_cache_key("roi_analysis", len(historical_data))

        # 캐시 확인
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 분석 결과 사용: {cache_key}")
            return cached_result

        # 분석 수행
        self.logger.info(f"ROI 기반 분석 시작: {len(historical_data)}개 데이터")

        results = {}

        # ROI 그룹 분석
        results["roi_pattern_groups"] = self.identify_roi_pattern_groups(
            historical_data
        )

        # 번호별 ROI 점수
        results["number_roi_scores"] = self.calculate_number_roi_scores(historical_data)

        # 패턴별 ROI 추세
        results["roi_trend_by_pattern"] = self.calculate_roi_trend_by_pattern(
            historical_data
        )

        # 위험도 매트릭스
        results["risk_matrix"] = self.calculate_risk_matrix(historical_data)

        # ROI 기반 번호 그룹
        results["roi_number_groups"] = self.analyze_roi_number_groups(historical_data)

        # 추가 기능 1: ROI 그룹 점수
        results["roi_group_score"] = self.calculate_roi_group_score(historical_data)

        # 추가 기능 2: ROI 클러스터 점수
        results["roi_cluster_score"] = self.calculate_roi_cluster_score(historical_data)

        # 추가 기능 3: 저위험 보너스 플래그
        results["low_risk_bonus_flag"] = self.calculate_low_risk_bonus_flag(
            historical_data
        )

        # 추가 기능 4: ROI 패턴 그룹 ID
        results["roi_pattern_group_id"] = self.calculate_roi_pattern_group_id(
            historical_data
        )

        # 결과 캐싱
        self._save_to_cache(cache_key, results)

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
        if expected_value > 0:
            payback_period = int(np.ceil(self.ticket_price / expected_value))
        else:
            payback_period = float("inf")

        result = ROIMetrics(
            expected_value=expected_value,
            roi=roi,
            win_probability=win_probability,
            payback_period=payback_period,
            risk_score=risk_score,
        )

        return result

    def _calculate_win_probability(self, target_numbers: List[int]) -> float:
        """
        번호 조합의 당첨 확률을 계산합니다.

        Args:
            target_numbers: 분석할 번호 조합(6개 번호)

        Returns:
            float: 당첨 확률
        """
        # 로또 6/45 당첨 확률: 1 / 8,145,060
        return 1.0 / 8145060

    def _calculate_expected_value(
        self, historical_data: List[LotteryNumber], target_numbers: List[int]
    ) -> float:
        """
        번호 조합의 기대값을 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            target_numbers: 분석할 번호 조합(6개 번호)

        Returns:
            float: 기대값
        """
        # 최근 100회 당첨금 기준
        recent_data = (
            historical_data[-100:] if len(historical_data) >= 100 else historical_data
        )

        # 1등 당첨금 계산 (실제 데이터 없으면 기본값 사용)
        first_prize_avg = 2000000000  # 20억원 기본값

        # 번호 출현 빈도 기반 보정
        number_frequency = Counter()
        for draw in recent_data:
            number_frequency.update(draw.numbers)

        # 선택 번호의 빈도 점수
        frequency_score = sum(
            number_frequency.get(num, 0) for num in target_numbers
        ) / len(recent_data)

        # 빈도 점수에 따른 보정 (0.8 ~ 1.2)
        freq_adjustment = 0.8 + (frequency_score / (6 * len(recent_data))) * 0.4

        # 당첨 확률
        win_probability = self._calculate_win_probability(target_numbers)

        # 기대값 = 확률 * 당첨금 * 보정계수
        expected_value = win_probability * first_prize_avg * freq_adjustment

        return expected_value

    def _calculate_risk_score(
        self, historical_data: List[LotteryNumber], target_numbers: List[int]
    ) -> float:
        """
        번호 조합의 위험 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            target_numbers: 분석할 번호 조합(6개 번호)

        Returns:
            float: 위험 점수 (0-1 범위, 높을수록 위험)
        """
        # 위험 지표들
        risk_factors = []

        # 1. 번호 분포 위험 (너무 편중된 경우)
        sorted_numbers = sorted(target_numbers)
        ranges = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)]
        range_counts = [0] * len(ranges)

        for num in sorted_numbers:
            for i, (start, end) in enumerate(ranges):
                if start <= num <= end:
                    range_counts[i] += 1
                    break

        # 분포 편중도 계산 (0에 가까울수록 고른 분포)
        distribution_skew = (
            np.std(range_counts) / np.mean(range_counts)
            if np.mean(range_counts) > 0
            else 0
        )
        distribution_risk = min(1.0, distribution_skew / 2)
        risk_factors.append(distribution_risk)

        # 2. 연속 번호 위험
        consecutive_count = 0
        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                consecutive_count += 1

        consecutive_risk = consecutive_count / 5.0  # 최대 5개 연속 가능
        risk_factors.append(consecutive_risk)

        # 중복 위험 (과거 당첨 번호와 많이 중복되는 경우)
        max_overlap = 0
        for draw in historical_data:
            overlap = len(set(draw.numbers) & set(target_numbers))
            max_overlap = max(max_overlap, overlap)

        overlap_risk = max_overlap / 6.0
        risk_factors.append(overlap_risk)

        # 종합 위험 점수 (0-1 범위)
        risk_score = np.mean(risk_factors)

        return risk_score

    def calculate_number_roi_scores(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        각 번호별 ROI 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 번호별 ROI 점수 (0-1 범위)
        """
        with self.performance_tracker.track("calculate_number_roi_scores"):
            self.logger.info("번호별 ROI 점수 계산 중...")

            # 결과 저장용 딕셔너리
            result = {}

            # 충분한 데이터가 있는지 확인
            if len(historical_data) < 30:
                self.logger.warning(
                    "번호별 ROI 점수 계산을 위한 충분한 데이터가 없습니다. (최소 30회차 필요)"
                )
                return result

            # 전체 데이터에서의 번호별 출현 빈도
            overall_frequency = Counter()
            for draw in historical_data:
                overall_frequency.update(draw.numbers)

            # 최근 데이터 30회에서의 번호별 출현 빈도
            recent_frequency = Counter()
            recent_data = historical_data[-30:]
            for draw in recent_data:
                recent_frequency.update(draw.numbers)

            # 번호별 ROI 점수 계산
            for num in range(1, 46):
                # 전체 빈도 (정규화)
                overall_norm = (
                    overall_frequency[num] / len(historical_data)
                    if historical_data
                    else 0
                )

                # 최근 빈도 (정규화)
                recent_norm = (
                    recent_frequency[num] / len(recent_data)
                    if len(recent_data) > 0
                    else 0
                )

                # ROI 점수 계산 (최근 추세 반영)
                # 전체 빈도와 최근 빈도의 조합으로 계산
                if overall_norm > 0:
                    # 최근 상승 추세면 ROI 높게 평가
                    trend_factor = recent_norm / overall_norm
                    roi_score = overall_norm * (0.5 + 0.5 * min(2.0, trend_factor))
                else:
                    roi_score = recent_norm

                # 0-1 범위로 정규화
                roi_score = min(1.0, roi_score * 6)  # 이론적 최대 빈도의 6배

                result[f"num_{num}"] = float(roi_score)

            # 상위/하위 ROI 번호 선별
            high_roi_numbers = sorted(
                [(num, result[f"num_{num}"]) for num in range(1, 46)],
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            low_roi_numbers = sorted(
                [(num, result[f"num_{num}"]) for num in range(1, 46)],
                key=lambda x: x[1],
            )[:10]

            # 결과에 상위/하위 ROI 번호 추가
            result["high_roi_numbers"] = [num for num, _ in high_roi_numbers]
            result["high_roi_scores"] = [score for _, score in high_roi_numbers]
            result["low_roi_numbers"] = [num for num, _ in low_roi_numbers]
            result["low_roi_scores"] = [score for _, score in low_roi_numbers]

            self.logger.info("번호별 ROI 점수 계산 완료")
            return result

    def calculate_roi_trend_by_pattern(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        패턴별 ROI 추세 계산

        과거 데이터를 분석하여 각 패턴별 ROI 추세를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 패턴별 ROI 추세 정보
        """
        with self.performance_tracker.track("calculate_roi_trend_by_pattern"):
            self.logger.info("패턴별 ROI 추세 계산 중...")

            # 결과 저장용 딕셔너리
            result = {}

            # 충분한 데이터가 있는지 확인
            if len(historical_data) < 50:
                self.logger.warning(
                    "패턴별 ROI 추세 계산을 위한 충분한 데이터가 없습니다. (최소 50회차 필요)"
                )
                return result

            # 패턴 그룹 정의 (각 패턴은 번호 특성을 기반으로 함)
            pattern_groups = {
                "odd_dominant": [],  # 홀수 우세 패턴
                "even_dominant": [],  # 짝수 우세 패턴
                "low_numbers": [],  # 낮은 번호 우세 패턴
                "high_numbers": [],  # 높은 번호 우세 패턴
                "balanced_segments": [],  # 균형 있는 구간 분포
                "consecutive_numbers": [],  # 연속 번호 포함
                "wide_range": [],  # 넓은 번호 범위
                "narrow_range": [],  # 좁은 번호 범위
                "prime_heavy": [],  # 소수 우세 패턴
                "multiples_of_3": [],  # 3의 배수 우세 패턴
            }

            # 각 패턴 그룹에 회차 분류
            for draw in historical_data:
                numbers = draw.numbers
                draw_no = draw.draw_no

                # 홀짝 비율
                odd_count = sum(1 for num in numbers if num % 2 == 1)
                even_count = len(numbers) - odd_count

                # 낮은/높은 번호 비율
                low_count = sum(1 for num in numbers if num <= 22)
                high_count = len(numbers) - low_count

                # 패턴 그룹 분류
                if odd_count > even_count:
                    pattern_groups["odd_dominant"].append(draw_no)
                elif even_count > odd_count:
                    pattern_groups["even_dominant"].append(draw_no)
                else:
                    pattern_groups["balanced_segments"].append(draw_no)

                # 낮은/높은 번호 그룹 분류
                if low_count > high_count:
                    pattern_groups["low_numbers"].append(draw_no)
                elif high_count > low_count:
                    pattern_groups["high_numbers"].append(draw_no)
                else:
                    pattern_groups["balanced_segments"].append(draw_no)

                # 연속 번호 패턴 분류
                consecutive_count = 0
                for i in range(1, len(numbers)):
                    if numbers[i] == numbers[i - 1] + 1:
                        consecutive_count += 1
                if consecutive_count > 0:
                    pattern_groups["consecutive_numbers"].append(draw_no)

                # 넓은/좁은 번호 범위 분류
                if max(numbers) - min(numbers) > 22:
                    pattern_groups["wide_range"].append(draw_no)
                else:
                    pattern_groups["narrow_range"].append(draw_no)

                # 소수 패턴 분류
                prime_count = sum(1 for num in numbers if num in self.primes)
                if prime_count > 0:
                    pattern_groups["prime_heavy"].append(draw_no)

                # 3의 배수 패턴 분류
                multiple_of_3_count = sum(1 for num in numbers if num % 3 == 0)
                if multiple_of_3_count > 0:
                    pattern_groups["multiples_of_3"].append(draw_no)

            # 패턴별 ROI 추세 계산
            for pattern, draws in pattern_groups.items():
                if len(draws) > 0:
                    result[f"{pattern}_roi_trend"] = self.calculate_roi_trend(
                        historical_data, draws
                    )

            self.logger.info("패턴별 ROI 추세 계산 완료")
            return result

    def calculate_roi_group_score(self, historical_data: List[LotteryNumber]) -> float:
        """
        ROI 그룹별 점수 계산

        과거 데이터를 분석하여 ROI가 높은 그룹별 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            float: ROI 그룹별 점수
        """
        with self.performance_tracker.track("calculate_roi_group_score"):
            self.logger.info("ROI 그룹별 점수 계산 중...")

            # 결과 저장용 변수
            total_score = 0.0

            # 충분한 데이터가 있는지 확인
            if len(historical_data) < 50:
                self.logger.warning(
                    "ROI 그룹별 점수 계산을 위한 충분한 데이터가 없습니다. (최소 50회차 필요)"
                )
                return total_score

            # ROI 그룹 정의 (각 그룹은 번호 특성을 기반으로 함)
            roi_groups = {
                "odd_dominant": [],  # 홀수 우세 그룹
                "even_dominant": [],  # 짝수 우세 그룹
                "low_numbers": [],  # 낮은 번호 우세 그룹
                "high_numbers": [],  # 높은 번호 우세 그룹
                "balanced_segments": [],  # 균형 있는 구간 분포
                "consecutive_numbers": [],  # 연속 번호 포함
                "wide_range": [],  # 넓은 번호 범위
                "narrow_range": [],  # 좁은 번호 범위
                "prime_heavy": [],  # 소수 우세 그룹
                "multiples_of_3": [],  # 3의 배수 우세 그룹
            }

            # 각 그룹에 회차 분류
            for draw in historical_data:
                numbers = draw.numbers
                draw_no = draw.draw_no

                # 그룹 분류
                if sum(1 for num in numbers if num % 2 == 1) > len(numbers) / 2:
                    roi_groups["odd_dominant"].append(draw_no)
                else:
                    roi_groups["even_dominant"].append(draw_no)

                if sum(1 for num in numbers if num <= 22) > len(numbers) / 2:
                    roi_groups["low_numbers"].append(draw_no)
                else:
                    roi_groups["high_numbers"].append(draw_no)

                if sum(1 for num in numbers if num % 3 == 0) > len(numbers) / 2:
                    roi_groups["multiples_of_3"].append(draw_no)

                # 연속 번호 패턴 분류
                consecutive_count = 0
                for i in range(1, len(numbers)):
                    if numbers[i] == numbers[i - 1] + 1:
                        consecutive_count += 1
                if consecutive_count > 0:
                    roi_groups["consecutive_numbers"].append(draw_no)

                # 균형 있는 구간 분포 패턴 분류
                if max(numbers) - min(numbers) > 22:
                    roi_groups["wide_range"].append(draw_no)
                else:
                    roi_groups["narrow_range"].append(draw_no)

                # 소수 패턴 분류
                prime_count = sum(1 for num in numbers if num in self.primes)
                if prime_count > 0:
                    roi_groups["prime_heavy"].append(draw_no)

            # 그룹별 ROI 점수 계산
            for group, draws in roi_groups.items():
                if len(draws) > 0:
                    total_score += self.calculate_roi_trend(historical_data, draws)

            self.logger.info("ROI 그룹별 점수 계산 완료")
            return total_score

    def calculate_roi_cluster_score(
        self, historical_data: List[LotteryNumber]
    ) -> float:
        """
        ROI 클러스터 점수 계산

        과거 데이터를 분석하여 ROI 클러스터 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            float: ROI 클러스터 점수
        """
        with self.performance_tracker.track("calculate_roi_cluster_score"):
            self.logger.info("ROI 클러스터 점수 계산 중...")

            # 결과 저장용 변수
            total_score = 0.0

            # 충분한 데이터가 있는지 확인
            if len(historical_data) < 50:
                self.logger.warning(
                    "ROI 클러스터 점수 계산을 위한 충분한 데이터가 없습니다. (최소 50회차 필요)"
                )
                return total_score

            # 번호별 ROI 점수 계산
            number_roi_scores = self.calculate_number_roi_scores(historical_data)

            # 클러스터 정의 (각 클러스터는 번호 특성을 기반으로 함)
            clusters = {
                "odd_dominant": [],  # 홀수 우세 클러스터
                "even_dominant": [],  # 짝수 우세 클러스터
                "low_numbers": [],  # 낮은 번호 우세 클러스터
                "high_numbers": [],  # 높은 번호 우세 클러스터
                "balanced_segments": [],  # 균형 있는 구간 분포
                "consecutive_numbers": [],  # 연속 번호 포함
                "wide_range": [],  # 넓은 번호 범위
                "narrow_range": [],  # 좁은 번호 범위
                "prime_heavy": [],  # 소수 우세 클러스터
                "multiples_of_3": [],  # 3의 배수 우세 클러스터
            }

            # 각 클러스터에 회차 분류
            for draw in historical_data:
                numbers = draw.numbers
                draw_no = draw.draw_no

                # 클러스터 분류
                if sum(1 for num in numbers if num % 2 == 1) > len(numbers) / 2:
                    clusters["odd_dominant"].append(draw_no)
                else:
                    clusters["even_dominant"].append(draw_no)

                if sum(1 for num in numbers if num <= 22) > len(numbers) / 2:
                    clusters["low_numbers"].append(draw_no)
                else:
                    clusters["high_numbers"].append(draw_no)

                if sum(1 for num in numbers if num % 3 == 0) > len(numbers) / 2:
                    clusters["multiples_of_3"].append(draw_no)

                # 연속 번호 패턴 분류
                consecutive_count = 0
                for i in range(1, len(numbers)):
                    if numbers[i] == numbers[i - 1] + 1:
                        consecutive_count += 1
                if consecutive_count > 0:
                    clusters["consecutive_numbers"].append(draw_no)

                # 균형 있는 구간 분포 패턴 분류
                if max(numbers) - min(numbers) > 22:
                    clusters["wide_range"].append(draw_no)
                else:
                    clusters["narrow_range"].append(draw_no)

                # 소수 패턴 분류
                prime_count = sum(1 for num in numbers if num in self.primes)
                if prime_count > 0:
                    clusters["prime_heavy"].append(draw_no)

            # 클러스터별 ROI 점수 계산
            for cluster, draws in clusters.items():
                if len(draws) > 0:
                    total_score += self.calculate_roi_trend(historical_data, draws)

            self.logger.info("ROI 클러스터 점수 계산 완료")
            return total_score
