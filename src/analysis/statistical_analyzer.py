"""
출현/위치 통계 분석기 모듈

이 모듈은 로또 번호의 출현 빈도, 위치 통계, 간격 등을 분석하는 기능을 제공합니다.
"""

import numpy as np
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import Counter, defaultdict

from src.analysis.base_analyzer import BaseAnalyzer
from src.shared.types import LotteryNumber
from src.utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class StatisticalAnalyzer(BaseAnalyzer):
    """출현/위치 통계 분석기 클래스"""

    def __init__(self, config: dict):
        """
        StatisticalAnalyzer 초기화

        Args:
            config: 분석에 사용할 설정
        """
        super().__init__(config, analyzer_type="statistical")

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        과거 로또 당첨 번호의 출현/위치 통계를 분석하는 내부 구현

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 분석 매개변수

        Returns:
            Dict[str, Any]: 출현/위치 통계 분석 결과
        """
        # 분석 수행
        self.logger.info(f"출현/위치 통계 분석 시작: {len(historical_data)}개 데이터")

        results = {}

        # 위치 빈도 분석
        results["position_frequency"] = self._analyze_position_frequency(
            historical_data
        )

        # 위치별 분산 계산
        position_variance = self._analyze_position_variance(historical_data)

        # 위치 분산 평균을 float으로 직접 추출
        results["position_variance_avg"] = float(position_variance["average_variance"])

        # 위치별 엔트로피 및 표준편차 계산
        self._add_position_entropy_and_std(historical_data, results)

        # 위치별 엔트로피와 표준편차를 메인 결과 딕셔너리에 직접 추가 (이미 추가되었음)
        # 정규화 처리
        max_theoretical_entropy = np.log2(45)  # 45개 번호의 최대 이론적 엔트로피
        max_std = 15.0  # 표준편차의 합리적인 최대값 (1-45 범위에서의 일반적 최대값)

        for i in range(1, 7):
            # 엔트로피 0-1 범위로 정규화
            if f"position_entropy_{i}" in results:
                raw_entropy = results[f"position_entropy_{i}"]
                results[f"position_entropy_{i}"] = min(
                    1.0, max(0.0, raw_entropy / max_theoretical_entropy)
                )

            # 표준편차 0-1 범위로 정규화
            if f"position_std_{i}" in results:
                raw_std = results[f"position_std_{i}"]
                results[f"position_std_{i}"] = min(1.0, max(0.0, raw_std / max_std))

        # 간격 표준편차
        results["gap_stddev"] = self._analyze_gap_std(historical_data)

        # Hot-Cold 번호 분석
        hot_numbers, cold_numbers = self._identify_hot_cold_numbers(historical_data)
        results["hot_numbers"] = list(hot_numbers)
        results["cold_numbers"] = list(cold_numbers)

        # Hot-Cold 혼합 점수
        results["hot_cold_mix_score"] = self._analyze_hot_cold_mix_score(
            historical_data
        )

        # 번호 출현 빈도
        results["number_frequency"] = self._calculate_number_frequencies(
            historical_data
        )

        # 위치별 통계
        results["position_stats"] = self._calculate_position_number_stats(
            historical_data
        )

        # 합계 분포 분석
        results["sum_distribution"] = self._analyze_number_sum_distribution(
            historical_data
        )

        # 홀짝 분포 분석
        results["odd_even_distribution"] = self._analyze_odd_even_distribution(
            historical_data
        )

        return results

    def _analyze_position_frequency(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        위치별 번호 출현 빈도를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 위치별 번호 출현 빈도
        """
        position_counters = [Counter() for _ in range(6)]

        for draw in historical_data:
            for i, num in enumerate(sorted(draw.numbers)):
                position_counters[i][num] += 1

        # 결과 변환
        result = {}
        for i, counter in enumerate(position_counters):
            position_key = f"position_{i+1}"
            # 각 위치에서의 번호별 출현 빈도 (상위 10개만)
            result[position_key] = {
                str(num): count / len(historical_data)
                for num, count in counter.most_common(10)
            }

        # 모든 위치에서의 전체 번호 출현 빈도
        all_counters = Counter()
        for counter in position_counters:
            all_counters.update(counter)

        result["all_positions"] = {
            str(num): count / (len(historical_data) * 6)
            for num, count in all_counters.most_common()
        }

        # 위치 빈도 행렬 계산 (통계 계산용)
        position_matrix = np.zeros((45, 6))
        for i, counter in enumerate(position_counters):
            for num in range(1, 46):
                position_matrix[num - 1, i] = counter[num] / len(historical_data)

        result["position_matrix"] = position_matrix.tolist()

        return result

    def _analyze_position_variance(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        위치별 번호 분산을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 위치별 번호 분산
        """
        # 위치별 번호 목록
        position_numbers = [[] for _ in range(6)]

        for draw in historical_data:
            sorted_numbers = sorted(draw.numbers)
            for i, num in enumerate(sorted_numbers):
                position_numbers[i].append(num)

        # 위치별 분산 계산
        result = {}
        for i, numbers in enumerate(position_numbers):
            variance = np.var(numbers)
            result[f"position_{i+1}_variance"] = float(variance)

        # 전체 위치 분산 평균
        result["average_variance"] = float(
            np.mean([result[f"position_{i+1}_variance"] for i in range(6)])
        )

        return result

    def _analyze_gap_std(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        번호 간격의 표준편차를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 간격 표준편차 분석 결과
        """
        result = {}

        # 모든 당첨 조합의 간격 계산
        all_gaps = []
        for draw in historical_data:
            sorted_numbers = sorted(draw.numbers)
            gaps = [
                sorted_numbers[i] - sorted_numbers[i - 1]
                for i in range(1, len(sorted_numbers))
            ]
            all_gaps.extend(gaps)

            # 각 조합별 간격 표준편차
            if len(gaps) > 0:
                result[f"draw_{draw.draw_no}_gap_std"] = float(np.std(gaps))

        # 전체 간격 표준편차
        result["overall_gap_std"] = float(np.std(all_gaps))

        # 최근 N회 간격 표준편차 추세
        recent_draws = min(30, len(historical_data))
        recent_gap_stds = []

        for i in range(1, recent_draws + 1):
            if i <= len(historical_data):
                draw = historical_data[-i]
                sorted_numbers = sorted(draw.numbers)
                gaps = [
                    sorted_numbers[j] - sorted_numbers[j - 1]
                    for j in range(1, len(sorted_numbers))
                ]
                if len(gaps) > 0:
                    recent_gap_stds.append(np.std(gaps))

        if recent_gap_stds:
            result["recent_gap_std_avg"] = float(np.mean(recent_gap_stds))
        else:
            result["recent_gap_std_avg"] = 0.0

        return result

    def _identify_hot_cold_numbers(
        self, historical_data: List[LotteryNumber]
    ) -> Tuple[Set[int], Set[int]]:
        """
        핫 넘버와 콜드 넘버를 식별합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Tuple[Set[int], Set[int]]: 핫 넘버와 콜드 넘버 집합
        """
        # 전체 기간 동안의 번호별 출현 횟수 계산
        frequency = Counter()
        for draw in historical_data:
            frequency.update(draw.numbers)

        # 최근 20% 기간 동안의 번호별 출현 횟수 계산
        recent_count = max(1, int(len(historical_data) * 0.2))
        recent_frequency = Counter()
        for draw in historical_data[-recent_count:]:
            recent_frequency.update(draw.numbers)

        # 핫 넘버: 전체 기간과 최근 기간 모두에서 상위 30% 출현
        all_numbers = set(range(1, 46))
        hot_threshold = int(45 * 0.3)  # 상위 30%

        hot_all_time = {num for num, _ in frequency.most_common(hot_threshold)}
        hot_recent = {num for num, _ in recent_frequency.most_common(hot_threshold)}

        hot_numbers = hot_all_time.intersection(hot_recent)

        # 콜드 넘버: 전체 기간과 최근 기간 모두에서 하위 30% 출현
        cold_threshold = int(45 * 0.3)  # 하위 30%

        cold_all_time = {
            num for num, _ in frequency.most_common()[: -cold_threshold - 1 : -1]
        }
        cold_recent = {
            num for num, _ in recent_frequency.most_common()[: -cold_threshold - 1 : -1]
        }

        cold_numbers = cold_all_time.intersection(cold_recent)

        return hot_numbers, cold_numbers

    def _analyze_hot_cold_mix_score(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        핫 넘버와 콜드 넘버의 혼합 점수를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 핫-콜드 혼합 점수
        """
        hot_numbers, cold_numbers = self._identify_hot_cold_numbers(historical_data)

        result = {}

        # 당첨 번호에서 핫 넘버와 콜드 넘버의 비율 계산
        hot_ratios = []
        cold_ratios = []
        balanced_ratios = []

        for draw in historical_data:
            numbers = set(draw.numbers)
            hot_count = len(numbers.intersection(hot_numbers))
            cold_count = len(numbers.intersection(cold_numbers))
            balanced_count = len(numbers) - hot_count - cold_count

            hot_ratio = hot_count / len(numbers)
            cold_ratio = cold_count / len(numbers)
            balanced_ratio = balanced_count / len(numbers)

            hot_ratios.append(hot_ratio)
            cold_ratios.append(cold_ratio)
            balanced_ratios.append(balanced_ratio)

        # 평균 비율 계산
        result["avg_hot_ratio"] = float(np.mean(hot_ratios))
        result["avg_cold_ratio"] = float(np.mean(cold_ratios))
        result["avg_balanced_ratio"] = float(np.mean(balanced_ratios))

        # 최적 혼합 비율 (과거 당첨 번호에서 가장 많이 나온 비율)
        # 예: (2 hot, 1 cold, 3 balanced)가 가장 많이 당첨되었다면 이를 최적 비율로 간주
        optimal_mix = (0, 0, 0)  # (hot, cold, balanced)
        max_count = 0

        mix_counter = Counter()
        for draw in historical_data:
            numbers = set(draw.numbers)
            hot_count = len(numbers.intersection(hot_numbers))
            cold_count = len(numbers.intersection(cold_numbers))
            balanced_count = len(numbers) - hot_count - cold_count

            mix = (hot_count, cold_count, balanced_count)
            mix_counter[mix] += 1

            if mix_counter[mix] > max_count:
                max_count = mix_counter[mix]
                optimal_mix = mix

        result["optimal_hot_count"] = optimal_mix[0]
        result["optimal_cold_count"] = optimal_mix[1]
        result["optimal_balanced_count"] = optimal_mix[2]
        result["optimal_mix_frequency"] = max_count / len(historical_data)

        return result

    def _calculate_number_frequencies(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        번호별 출현 빈도를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 번호별 출현 빈도
        """
        frequency = Counter()
        total_draws = len(historical_data)

        for draw in historical_data:
            frequency.update(draw.numbers)

        # 각 번호의 출현 빈도를 백분율로 변환
        result = {}
        expected_frequency = total_draws * 6 / 45  # 랜덤 추출 시 기대 빈도

        for num in range(1, 46):
            actual_frequency = frequency[num]
            normalized_frequency = actual_frequency / expected_frequency
            result[f"num_{num}"] = float(normalized_frequency)

        return result

    def _calculate_position_number_stats(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Dict[str, float]]:
        """
        위치별 번호 통계를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Dict[str, float]]: 위치별 번호 통계
        """
        # 위치별 번호 목록
        position_numbers = [[] for _ in range(6)]

        for draw in historical_data:
            sorted_numbers = sorted(draw.numbers)
            for i, num in enumerate(sorted_numbers):
                position_numbers[i].append(num)

        # 위치별 통계 계산
        result = {}
        for i, numbers in enumerate(position_numbers):
            position_key = f"position_{i+1}"
            stats = {
                "mean": float(np.mean(numbers)),
                "median": float(np.median(numbers)),
                "std": float(np.std(numbers)),
                "min": float(min(numbers)),
                "max": float(max(numbers)),
                "range": float(max(numbers) - min(numbers)),
            }
            result[position_key] = stats

        return result

    def _analyze_number_sum_distribution(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        당첨 번호 합계 분포를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 합계 분포 분석 결과
        """
        # 각 당첨 조합의 합계 계산
        sums = [sum(draw.numbers) for draw in historical_data]

        # 합계 통계 계산
        result = {
            "mean_sum": float(np.mean(sums)),
            "median_sum": float(np.median(sums)),
            "std_sum": float(np.std(sums)),
            "min_sum": float(min(sums)),
            "max_sum": float(max(sums)),
            "range_sum": float(max(sums) - min(sums)),
        }

        # 합계 구간별 빈도 계산
        sum_bins = {
            "sum_70_100": 0,
            "sum_101_125": 0,
            "sum_126_150": 0,
            "sum_151_175": 0,
            "sum_176_200": 0,
            "sum_201_225": 0,
            "sum_226_255": 0,
        }

        for total in sums:
            if 70 <= total <= 100:
                sum_bins["sum_70_100"] += 1
            elif 101 <= total <= 125:
                sum_bins["sum_101_125"] += 1
            elif 126 <= total <= 150:
                sum_bins["sum_126_150"] += 1
            elif 151 <= total <= 175:
                sum_bins["sum_151_175"] += 1
            elif 176 <= total <= 200:
                sum_bins["sum_176_200"] += 1
            elif 201 <= total <= 225:
                sum_bins["sum_201_225"] += 1
            elif 226 <= total <= 255:
                sum_bins["sum_226_255"] += 1

        # 구간별 빈도를 백분율로 변환
        for key, count in sum_bins.items():
            result[key] = float(count / len(sums))

        return result

    def _analyze_odd_even_distribution(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        홀짝 번호 분포를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 홀짝 분포 분석 결과
        """
        # 각 당첨 조합의 홀짝 개수 계산
        odd_counts = []
        even_counts = []

        for draw in historical_data:
            odd = sum(1 for num in draw.numbers if num % 2 == 1)
            even = len(draw.numbers) - odd

            odd_counts.append(odd)
            even_counts.append(even)

        # 홀짝 조합 빈도 계산
        odd_even_counter = Counter(zip(odd_counts, even_counts))
        total_draws = len(historical_data)

        result = {}

        # 각 조합의 빈도를 백분율로 변환
        for (odd, even), count in odd_even_counter.items():
            result[f"odd_{odd}_even_{even}"] = float(count / total_draws)

        # 홀수/짝수 평균
        result["avg_odd"] = float(np.mean(odd_counts))
        result["avg_even"] = float(np.mean(even_counts))

        # 가장 많이 나온 홀짝 조합
        most_common = odd_even_counter.most_common(1)
        if most_common:
            (odd, even), count = most_common[0]
            result["most_common_odd"] = odd
            result["most_common_even"] = even
            result["most_common_frequency"] = float(count / total_draws)

        return result

    def _add_position_entropy_and_std(
        self, historical_data: List[LotteryNumber], results: Dict[str, Any]
    ) -> None:
        """
        위치별 엔트로피와 표준편차를 계산하여 결과에 추가합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            results: 결과를 저장할 딕셔너리
        """
        # 위치별 번호 목록
        position_numbers = [[] for _ in range(6)]

        for draw in historical_data:
            sorted_numbers = sorted(draw.numbers)
            for i, num in enumerate(sorted_numbers):
                position_numbers[i].append(num)

        # 위치별 번호 출현 빈도 계산
        position_counts = [Counter() for _ in range(6)]
        for i, numbers in enumerate(position_numbers):
            position_counts[i].update(numbers)

        # 위치별 엔트로피 계산
        for i in range(6):
            # 표준편차 계산 및 저장 (float로 변환)
            std_value = float(np.std(position_numbers[i]))
            results[f"position_std_{i+1}"] = std_value

            # 번호별 확률 계산
            total_count = len(position_numbers[i])
            probs = {}
            for num in range(1, 46):
                count = position_counts[i][num]
                probs[num] = count / total_count if total_count > 0 else 0

            # 섀넌 엔트로피 계산: H = -Σ(p_i * log2(p_i))
            entropy = 0.0
            for num, prob in probs.items():
                if prob > 0:  # 0 확률은 무시
                    entropy -= prob * np.log2(prob)

            # 결과 저장 (float로 변환)
            results[f"position_entropy_{i+1}"] = float(entropy)

        self.logger.info("위치별 엔트로피 및 표준편차 계산 완료")
