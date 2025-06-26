"""
유사도/중복성 분석기 모듈

이 모듈은 로또 번호 조합의 과거 당첨 번호와의 유사도, 중복성 등을 분석하는 기능을 제공합니다.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np

from ..analysis.base_analyzer import BaseAnalyzer
from ..shared.types import LotteryNumber
from ..utils.error_handler import get_logger

logger = get_logger(__name__)


class OverlapAnalyzer(BaseAnalyzer):
    """유사도/중복성 분석기 클래스"""

    def __init__(self, config: dict):
        """
        OverlapAnalyzer 초기화

        Args:
            config: 분석에 사용할 설정
        """
        super().__init__(config, analyzer_type="overlap")

    def analyze(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        과거 로또 당첨 번호의 유사도/중복성을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 유사도/중복성 분석 결과
        """
        # 캐시 키 생성
        cache_key = self._create_cache_key("overlap_analysis", len(historical_data))

        # 캐시 확인
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 분석 결과 사용: {cache_key}")
            return cached_result

        # 분석 수행
        self.logger.info(f"유사도/중복성 분석 시작: {len(historical_data)}개 데이터")

        results = {}

        # 중복 플래그 분석
        results["duplicate_flag"] = self._analyze_duplicate_flag(historical_data)

        # 과거 당첨 번호와의 최대 중복도
        results["max_overlap_with_past"] = self._analyze_max_overlap_with_past(
            historical_data
        )

        # 최근성 점수
        results["recency_score"] = self._analyze_recency_score(historical_data)

        # 중복 패턴 분석
        results["overlap_patterns"] = self._analyze_overlap_patterns(historical_data)

        # 인접 회차 유사도
        results["adjacent_draw_similarity"] = self._analyze_adjacent_draw_similarity(
            historical_data
        )

        # 최근 N회 중복 패턴
        results["recent_overlap_patterns"] = self._analyze_recent_overlap_patterns(
            historical_data
        )

        # 정확한 일치 여부 분석 (새로 추가)
        exact_match_results = self._analyze_exact_match_in_history(historical_data)
        results["exact_match_in_history"] = exact_match_results["exact_match_list"]
        results["exact_match_count"] = exact_match_results["exact_match_count"]

        # 과거 최대 중복 번호 수 분석 (새로 추가)
        results["num_overlap_with_past_max"] = self._analyze_num_overlap_with_past_max(
            historical_data
        )

        # 인기 패턴과의 중복 분석 (새로 추가)
        results["overlap_with_hot_patterns"] = self._analyze_overlap_with_hot_patterns(
            historical_data
        )

        # 3자리 및 4자리 중복 패턴 분석 (신규 추가)
        results["overlap_3_4_digit_patterns"] = (
            self._analyze_3_4_digit_overlap_patterns(historical_data)
        )

        # 중복 패턴 ROI 상관관계 분석 (신규 추가)
        results["overlap_roi_correlation"] = self._analyze_overlap_roi_correlation(
            historical_data
        )

        # 결과 캐싱
        self._save_to_cache(cache_key, results)

        return results

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        과거 로또 당첨 번호의 유사도/중복성을 분석하는 내부 구현입니다.
        BaseAnalyzer의 추상 메서드를 구현합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 매개변수

        Returns:
            Dict[str, Any]: 유사도/중복성 분석 결과
        """
        # 분석 수행
        self.logger.info(f"유사도/중복성 분석 시작: {len(historical_data)}개 데이터")

        results = {}

        # 중복 플래그 분석
        results["duplicate_flag"] = self._analyze_duplicate_flag(historical_data)

        # 과거 당첨 번호와의 최대 중복도
        results["max_overlap_with_past"] = self._analyze_max_overlap_with_past(
            historical_data
        )

        # 최근성 점수
        results["recency_score"] = self._analyze_recency_score(historical_data)

        # 중복 패턴 분석
        results["overlap_patterns"] = self._analyze_overlap_patterns(historical_data)

        # 인접 회차 유사도
        results["adjacent_draw_similarity"] = self._analyze_adjacent_draw_similarity(
            historical_data
        )

        # 최근 N회 중복 패턴
        results["recent_overlap_patterns"] = self._analyze_recent_overlap_patterns(
            historical_data
        )

        # 정확한 일치 여부 분석 (새로 추가)
        exact_match_results = self._analyze_exact_match_in_history(historical_data)
        results["exact_match_in_history"] = exact_match_results["exact_match_list"]
        results["exact_match_count"] = exact_match_results["exact_match_count"]

        # 과거 최대 중복 번호 수 분석 (새로 추가)
        results["num_overlap_with_past_max"] = self._analyze_num_overlap_with_past_max(
            historical_data
        )

        # 인기 패턴과의 중복 분석 (새로 추가)
        results["overlap_with_hot_patterns"] = self._analyze_overlap_with_hot_patterns(
            historical_data
        )

        return results

    def _analyze_duplicate_flag(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        중복 조합 여부를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 중복 조합 분석 결과
        """
        result = {}

        # 각 조합이 이전에 동일하게 나왔는지 확인
        duplicate_flags = []
        duplicate_indices = []

        for i, draw in enumerate(historical_data):
            is_duplicate = False
            duplicate_idx = -1

            # 현재 조합과 이전 모든 조합 비교
            for j in range(i):
                if set(draw.numbers) == set(historical_data[j].numbers):
                    is_duplicate = True
                    duplicate_idx = j
                    break

            duplicate_flags.append(is_duplicate)
            duplicate_indices.append(duplicate_idx)

        # 중복 조합 개수 및 비율
        duplicate_count = sum(duplicate_flags)
        duplicate_ratio = (
            duplicate_count / len(historical_data) if historical_data else 0
        )

        # 중복 조합 회차 정보
        duplicate_rounds = []
        for i, is_duplicate in enumerate(duplicate_flags):
            if is_duplicate:
                # 중복 회차 및 원본 회차
                draw_no = historical_data[i].draw_no
                original_draw_no = historical_data[duplicate_indices[i]].draw_no
                duplicate_rounds.append((draw_no, original_draw_no))

        # 결과 취합
        result["duplicate_count"] = duplicate_count
        result["duplicate_ratio"] = float(duplicate_ratio)
        result["duplicate_rounds"] = duplicate_rounds

        # 중복이 가장 많이 발생한 조합
        if duplicate_count > 0:
            # 조합별 출현 횟수
            combination_counter = Counter()
            for draw in historical_data:
                combination_counter[tuple(sorted(draw.numbers))] += 1

            # 가장 많이 중복된 조합
            most_common = combination_counter.most_common(5)
            result["most_duplicated"] = []
            result["most_duplicated_counts"] = []

            for comb, count in most_common:
                if count > 1:  # 중복된 조합만 포함
                    result["most_duplicated"].append(list(comb))
                    result["most_duplicated_counts"].append(count)

        return result

    def _analyze_max_overlap_with_past(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        과거 당첨 번호와의 최대 중복도를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 최대 중복도 분석 결과
        """
        result = {}

        # 각 조합과 과거 모든 조합 간의 중복 번호 수
        max_overlaps = []
        avg_overlaps = []

        for i, draw in enumerate(historical_data):
            if i == 0:  # 첫 번째 회차는 이전 회차가 없음
                max_overlaps.append(0)
                avg_overlaps.append(0)
                continue

            # 현재 조합과 이전 모든 조합의 중복 번호 수
            overlaps = []
            for j in range(i):
                overlap = len(set(draw.numbers) & set(historical_data[j].numbers))
                overlaps.append(overlap)

            # 최대 중복 번호 수와 평균 중복 번호 수
            max_overlap = max(overlaps)
            avg_overlap = sum(overlaps) / len(overlaps)

            max_overlaps.append(max_overlap)
            avg_overlaps.append(avg_overlap)

        # 통계 계산
        result["max_overlap_avg"] = float(
            np.mean(max_overlaps[1:])
        )  # 첫 번째 회차 제외
        result["max_overlap_max"] = int(max(max_overlaps[1:]))
        result["max_overlap_min"] = int(min(max_overlaps[1:]))
        result["max_overlap_std"] = float(np.std(max_overlaps[1:]))

        result["avg_overlap_avg"] = float(np.mean(avg_overlaps[1:]))
        result["avg_overlap_max"] = float(max(avg_overlaps[1:]))
        result["avg_overlap_min"] = float(min(avg_overlaps[1:]))
        result["avg_overlap_std"] = float(np.std(avg_overlaps[1:]))

        # 각 중복 수준(0~6)별 빈도
        overlap_counts = [0] * 7  # 0~6개 중복

        for max_overlap in max_overlaps[1:]:  # 첫 번째 회차 제외
            overlap_counts[max_overlap] += 1

        # 중복 수준별 비율
        total_draws = len(max_overlaps[1:])
        for i in range(7):
            result[f"overlap_{i}_count"] = overlap_counts[i]
            result[f"overlap_{i}_ratio"] = float(overlap_counts[i] / total_draws)

        return result

    def _analyze_recency_score(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        번호별 최근성 점수를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 최근성 점수 분석 결과
        """
        result = {}

        # 각 번호의 마지막 출현 회차
        last_appearance = {}

        # 역순으로 순회하며 각 번호의 첫 출현 회차 기록
        total_draws = len(historical_data)

        for num in range(1, 46):
            last_appearance[num] = -1  # 기본값: 출현하지 않음

            for i in range(total_draws - 1, -1, -1):
                if num in historical_data[i].numbers:
                    last_appearance[num] = i
                    break

        # 최근성 점수 계산
        for num in range(1, 46):
            if last_appearance[num] >= 0:
                # 정규화된 최근성 점수 (0~1): 최근에 출현할수록 높음
                recency = last_appearance[num] / (total_draws - 1)
                result[f"num_{num}"] = float(recency)
            else:
                # 한 번도 출현하지 않은 경우
                result[f"num_{num}"] = 0.0

        # 최근성 기준 상위/하위 번호
        recency_scores = [(num, result[f"num_{num}"]) for num in range(1, 46)]

        # 가장 최근에 출현한 번호들
        most_recent = sorted(recency_scores, key=lambda x: x[1], reverse=True)[:10]
        result["most_recent"] = [int(num) for num, _ in most_recent]
        result["most_recent_scores"] = [float(score) for _, score in most_recent]

        # 가장 오래전에 출현한 번호들
        least_recent = sorted(
            [x for x in recency_scores if x[1] > 0], key=lambda x: x[1]
        )[:10]
        result["least_recent"] = [int(num) for num, _ in least_recent]
        result["least_recent_scores"] = [float(score) for _, score in least_recent]

        # 한 번도 출현하지 않은 번호들
        never_appeared = [num for num in range(1, 46) if result[f"num_{num}"] == 0.0]
        result["never_appeared"] = never_appeared

        return result

    def _analyze_overlap_patterns(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        중복 패턴을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 중복 패턴 분석 결과
        """
        result = {}

        # 특정 간격으로 중복되는 패턴 분석
        lags = [1, 2, 3, 4, 5, 10, 20]  # 분석할 간격

        for lag in lags:
            if len(historical_data) <= lag:
                continue

            lag_key = f"lag_{lag}"
            result[lag_key] = {}

            # 각 회차와 이전 회차의 중복 번호 수
            overlaps = []

            for i in range(lag, len(historical_data)):
                current_draw = set(historical_data[i].numbers)
                previous_draw = set(historical_data[i - lag].numbers)

                overlap = len(current_draw & previous_draw)
                overlaps.append(overlap)

            # 중복 수준별 빈도
            overlap_counts = [0] * 7  # 0~6개 중복
            for overlap in overlaps:
                overlap_counts[overlap] += 1

            # 통계 계산
            result[lag_key]["avg_overlap"] = float(np.mean(overlaps))
            result[lag_key]["std_overlap"] = float(np.std(overlaps))
            result[lag_key]["max_overlap"] = int(max(overlaps))

            # 중복 수준별 비율
            total_draws = len(overlaps)
            for i in range(7):
                result[lag_key][f"overlap_{i}_count"] = overlap_counts[i]
                result[lag_key][f"overlap_{i}_ratio"] = float(
                    overlap_counts[i] / total_draws
                )

        return result

    def _analyze_adjacent_draw_similarity(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        인접 회차 간 유사도를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 인접 회차 유사도 분석 결과
        """
        result = {}

        # 인접 회차 간 유사도 계산
        jaccard_similarities = []
        cosine_similarities = []
        overlap_counts = []

        for i in range(1, len(historical_data)):
            current_draw = set(historical_data[i].numbers)
            previous_draw = set(historical_data[i - 1].numbers)

            # Jaccard 유사도: 교집합 크기 / 합집합 크기
            jaccard = len(current_draw & previous_draw) / len(
                current_draw | previous_draw
            )
            jaccard_similarities.append(jaccard)

            # 중복 번호 수
            overlap = len(current_draw & previous_draw)
            overlap_counts.append(overlap)

            # 코사인 유사도를 위한 벡터 표현
            vec1 = np.zeros(45)
            vec2 = np.zeros(45)

            for num in current_draw:
                vec1[num - 1] = 1

            for num in previous_draw:
                vec2[num - 1] = 1

            # 코사인 유사도: 두 벡터의 내적 / (두 벡터의 크기 곱)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            cosine = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
            cosine_similarities.append(cosine)

        # 통계 계산
        result["jaccard_avg"] = float(np.mean(jaccard_similarities))
        result["jaccard_std"] = float(np.std(jaccard_similarities))
        result["jaccard_max"] = float(max(jaccard_similarities))
        result["jaccard_min"] = float(min(jaccard_similarities))

        result["cosine_avg"] = float(np.mean(cosine_similarities))
        result["cosine_std"] = float(np.std(cosine_similarities))
        result["cosine_max"] = float(max(cosine_similarities))
        result["cosine_min"] = float(min(cosine_similarities))

        result["overlap_avg"] = float(np.mean(overlap_counts))
        result["overlap_std"] = float(np.std(overlap_counts))

        # 중복 수준별 빈도
        overlap_counters = Counter(overlap_counts)
        total_draws = len(overlap_counts)

        for i in range(7):  # 0~6개 중복
            count = overlap_counters.get(i, 0)
            result[f"overlap_{i}_count"] = count
            result[f"overlap_{i}_ratio"] = float(count / total_draws)

        # 가장 유사한 인접 회차 찾기
        if jaccard_similarities:
            max_jaccard_idx = np.argmax(jaccard_similarities)
            max_jaccard_draw1 = historical_data[max_jaccard_idx + 1].draw_no
            max_jaccard_draw2 = historical_data[max_jaccard_idx].draw_no

            result["most_similar_draws"] = [max_jaccard_draw1, max_jaccard_draw2]
            result["most_similar_jaccard"] = float(
                jaccard_similarities[max_jaccard_idx]
            )
            result["most_similar_overlap"] = int(overlap_counts[max_jaccard_idx])

        return result

    def _analyze_recent_overlap_patterns(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        최근 회차의 중복 패턴을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 최근 중복 패턴 분석 결과
        """
        result = {}

        # 분석할 최근 회차 수
        recent_counts = [10, 20, 30, 50]

        for count in recent_counts:
            if len(historical_data) <= count:
                continue

            count_key = f"recent_{count}"
            result[count_key] = {}

            # 최근 회차 데이터
            recent_data = historical_data[-count:]

            # 번호별 출현 횟수
            number_counts = Counter()
            for draw in recent_data:
                number_counts.update(draw.numbers)

            # 번호별 출현 빈도
            for num in range(1, 46):
                freq = number_counts.get(num, 0) / count
                result[count_key][f"num_{num}_freq"] = float(freq)

            # 가장 많이 출현한 번호
            most_common = number_counts.most_common(10)
            result[count_key]["most_common"] = [num for num, _ in most_common]
            result[count_key]["most_common_counts"] = [
                count for _, count in most_common
            ]

            # 번호 쌍 출현 횟수
            pair_counts = Counter()
            for draw in recent_data:
                numbers = sorted(draw.numbers)
                for i in range(len(numbers)):
                    for j in range(i + 1, len(numbers)):
                        pair = (numbers[i], numbers[j])
                        pair_counts[pair] += 1

            # 가장 많이 출현한 번호 쌍
            most_common_pairs = pair_counts.most_common(10)
            result[count_key]["most_common_pairs"] = [
                list(pair) for pair, _ in most_common_pairs
            ]
            result[count_key]["most_common_pair_counts"] = [
                count for _, count in most_common_pairs
            ]

            # 완전 중복 조합 (최근 count 회차 내에서 완전히 동일한 조합)
            combination_counts = Counter()
            for draw in recent_data:
                combination = tuple(sorted(draw.numbers))
                combination_counts[combination] += 1

            # 중복 조합 찾기
            duplicate_combinations = []
            for comb, count in combination_counts.items():
                if count > 1:
                    duplicate_combinations.append((list(comb), count))

            if duplicate_combinations:
                result[count_key]["duplicate_combinations"] = [
                    comb for comb, _ in duplicate_combinations
                ]
                result[count_key]["duplicate_counts"] = [
                    count for _, count in duplicate_combinations
                ]
                result[count_key]["has_duplicates"] = True
            else:
                result[count_key]["has_duplicates"] = False

        return result

    def _analyze_exact_match_in_history(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        각 회차의 번호 조합이 이전 회차와 정확히 일치하는지 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 정확한 일치 여부 분석 결과
        """
        result = {}

        # 정확한 일치 여부 리스트
        exact_match_list = []
        exact_match_indices = []

        # 각 회차의 정확한 일치 여부 확인
        for i, draw in enumerate(historical_data):
            # 현재 회차의 번호 집합
            current_set = frozenset(draw.numbers)

            # 이전 회차들과 비교
            is_exact_match = False
            match_idx = -1

            for j in range(i):
                prev_set = frozenset(historical_data[j].numbers)
                if current_set == prev_set:
                    is_exact_match = True
                    match_idx = j
                    break

            exact_match_list.append(is_exact_match)
            exact_match_indices.append(match_idx if is_exact_match else -1)

        # 정확한 일치 회차 수
        exact_match_count = sum(exact_match_list)

        # 일치 회차 정보
        match_details = []
        for i, is_match in enumerate(exact_match_list):
            if is_match:
                current_draw_no = getattr(historical_data[i], "draw_no", i)
                match_draw_no = getattr(
                    historical_data[exact_match_indices[i]],
                    "draw_no",
                    exact_match_indices[i],
                )
                match_details.append(
                    {
                        "draw_no": current_draw_no,
                        "match_draw_no": match_draw_no,
                        "numbers": historical_data[i].numbers,
                    }
                )

        # 결과 취합
        result["exact_match_list"] = exact_match_list
        result["exact_match_count"] = exact_match_count
        result["exact_match_ratio"] = (
            float(exact_match_count / len(historical_data)) if historical_data else 0.0
        )
        result["match_details"] = match_details

        return result

    def _analyze_num_overlap_with_past_max(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        각 회차의 번호 조합과 이전 모든 회차 간의 최대 중복 번호 수를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 최대 중복 번호 수 분석 결과
        """
        result = {}

        # 각 회차의 최대 중복 번호 수
        max_overlap_counts = []
        max_overlap_draw_indices = []

        for i, draw in enumerate(historical_data):
            if i == 0:  # 첫 번째 회차는 이전 회차가 없음
                max_overlap_counts.append(0)
                max_overlap_draw_indices.append(-1)
                continue

            # 현재 회차의 번호 집합
            current_set = set(draw.numbers)

            # 이전 회차들과 비교하여 최대 중복 수 찾기
            max_overlap = 0
            max_overlap_idx = -1

            for j in range(i):
                prev_set = set(historical_data[j].numbers)
                overlap = len(current_set & prev_set)

                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_idx = j

            max_overlap_counts.append(max_overlap)
            max_overlap_draw_indices.append(max_overlap_idx)

        # 최대 중복 번호 수 통계
        if len(max_overlap_counts) > 1:  # 첫 번째 회차 제외
            overlap_stats = max_overlap_counts[1:]
            result["average"] = float(np.mean(overlap_stats))
            result["max"] = int(max(overlap_stats))
            result["min"] = int(min(overlap_stats))
            result["std"] = float(np.std(overlap_stats))

            # 분포 계산
            distribution = {}
            for overlap in range(7):  # 0~6개 중복
                count = overlap_stats.count(overlap)
                distribution[str(overlap)] = count

            result["distribution"] = distribution

            # 높은 중복 수를 가진 회차 정보
            high_overlap_draws = []
            high_overlap_threshold = 4  # 4개 이상 중복을 높은 중복으로 간주

            for i in range(1, len(historical_data)):
                if max_overlap_counts[i] >= high_overlap_threshold:
                    draw_no = getattr(historical_data[i], "draw_no", i)
                    match_draw_no = getattr(
                        historical_data[max_overlap_draw_indices[i]],
                        "draw_no",
                        max_overlap_draw_indices[i],
                    )
                    high_overlap_draws.append(
                        {
                            "draw_no": draw_no,
                            "match_draw_no": match_draw_no,
                            "overlap_count": max_overlap_counts[i],
                            "numbers": historical_data[i].numbers,
                            "matching_numbers": list(
                                set(historical_data[i].numbers)
                                & set(
                                    historical_data[max_overlap_draw_indices[i]].numbers
                                )
                            ),
                        }
                    )

            result["high_overlap_draws"] = high_overlap_draws
        else:
            # 데이터가 충분하지 않은 경우
            result["average"] = 0.0
            result["max"] = 0
            result["min"] = 0
            result["std"] = 0.0
            result["distribution"] = {"0": 0}
            result["high_overlap_draws"] = []

        return result

    def _analyze_overlap_with_hot_patterns(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        각 회차의 번호 조합과 인기 번호 패턴 간의 중복을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 인기 패턴 중복 분석 결과
        """
        result = {}

        # 인기 번호 목록 생성
        hot_numbers = self._get_hot_numbers(historical_data)

        if not hot_numbers:
            self.logger.warning(
                "인기 번호 목록을 생성할 수 없습니다. 데이터가 부족합니다."
            )
            result["error"] = "인기 번호 목록을 생성할 수 없습니다."
            return result

        # 인기 번호 목록 저장
        result["hot_numbers"] = hot_numbers

        # 각 회차별 인기 번호와의 중복 계산
        overlap_counts = []
        high_overlap_draws = []

        for i, draw in enumerate(historical_data):
            # 현재 회차와 인기 번호 간 중복 수
            overlap = len(set(draw.numbers) & set(hot_numbers))
            overlap_counts.append(overlap)

            # 높은 중복을 가진 회차 기록 (4개 이상)
            if overlap >= 4:
                draw_no = getattr(draw, "draw_no", i)
                high_overlap_draws.append(draw_no)

        # 중복 분포 계산
        overlap_distribution = {}
        for i in range(7):  # 0~6개 중복
            overlap_distribution[str(i)] = overlap_counts.count(i)

        # 결과 취합
        result["avg_count"] = float(np.mean(overlap_counts)) if overlap_counts else 0.0
        result["high_overlap_draws"] = high_overlap_draws
        result["hot_overlap_distribution"] = overlap_distribution

        return result

    def _get_hot_numbers(self, historical_data: List[LotteryNumber]) -> List[int]:
        """
        인기 번호(출현 빈도가 높은 번호)를 추출합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            List[int]: 인기 번호 목록 (상위 20개)
        """
        # 번호별 출현 빈도 계산
        number_counts = Counter()
        for draw in historical_data:
            for number in draw.numbers:
                number_counts[number] += 1

        # 출현 빈도가 높은 상위 20개 번호 반환
        hot_numbers = [num for num, count in number_counts.most_common(20)]
        return hot_numbers

    def _analyze_3_4_digit_overlap_patterns(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        과거 전체 회차 간 3자리 및 4자리 중복 패턴 통계 분석

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 3자리 및 4자리 중복 패턴 분석 결과
        """
        result = {}

        # 3자리 패턴 분석
        overlap_3_patterns = {}
        overlap_3_frequency = Counter()
        overlap_3_time_gaps = []

        # 4자리 패턴 분석
        overlap_4_patterns = {}
        overlap_4_frequency = Counter()
        overlap_4_time_gaps = []

        # 모든 회차 조합 간 비교
        for i in range(len(historical_data)):
            current_numbers = set(historical_data[i].numbers)
            current_draw_no = historical_data[i].draw_no

            for j in range(i + 1, len(historical_data)):
                other_numbers = set(historical_data[j].numbers)
                other_draw_no = historical_data[j].draw_no

                # 중복 번호 찾기
                overlap = current_numbers & other_numbers
                overlap_count = len(overlap)

                # 시간 간격 계산
                time_gap = abs(current_draw_no - other_draw_no)

                # 3자리 중복 패턴
                if overlap_count == 3:
                    pattern_key = tuple(sorted(overlap))
                    overlap_3_frequency[pattern_key] += 1
                    overlap_3_time_gaps.append(time_gap)

                    if pattern_key not in overlap_3_patterns:
                        overlap_3_patterns[pattern_key] = []
                    overlap_3_patterns[pattern_key].append(
                        {
                            "draw1": current_draw_no,
                            "draw2": other_draw_no,
                            "time_gap": time_gap,
                        }
                    )

                # 4자리 중복 패턴
                if overlap_count == 4:
                    pattern_key = tuple(sorted(overlap))
                    overlap_4_frequency[pattern_key] += 1
                    overlap_4_time_gaps.append(time_gap)

                    if pattern_key not in overlap_4_patterns:
                        overlap_4_patterns[pattern_key] = []
                    overlap_4_patterns[pattern_key].append(
                        {
                            "draw1": current_draw_no,
                            "draw2": other_draw_no,
                            "time_gap": time_gap,
                        }
                    )

        # 3자리 패턴 통계
        result["overlap_3_patterns"] = {
            "total_patterns": len(overlap_3_patterns),
            "total_occurrences": sum(overlap_3_frequency.values()),
            "most_frequent": dict(overlap_3_frequency.most_common(10)),
            "avg_frequency": (
                float(np.mean(list(overlap_3_frequency.values())))
                if overlap_3_frequency
                else 0.0
            ),
            "frequency_std": (
                float(np.std(list(overlap_3_frequency.values())))
                if overlap_3_frequency
                else 0.0
            ),
        }

        # 4자리 패턴 통계 (희귀도 분석)
        result["overlap_4_patterns"] = {
            "total_patterns": len(overlap_4_patterns),
            "total_occurrences": sum(overlap_4_frequency.values()),
            "rarity_score": 1.0 / (len(overlap_4_patterns) + 1),  # 희귀도 점수
            "most_frequent": dict(overlap_4_frequency.most_common(5)),
            "avg_frequency": (
                float(np.mean(list(overlap_4_frequency.values())))
                if overlap_4_frequency
                else 0.0
            ),
        }

        # 시간 간격 분석
        result["overlap_time_gap_analysis"] = {
            "gap_3_avg": (
                float(np.mean(overlap_3_time_gaps)) if overlap_3_time_gaps else 0.0
            ),
            "gap_3_std": (
                float(np.std(overlap_3_time_gaps)) if overlap_3_time_gaps else 0.0
            ),
            "gap_4_avg": (
                float(np.mean(overlap_4_time_gaps)) if overlap_4_time_gaps else 0.0
            ),
            "gap_4_std": (
                float(np.std(overlap_4_time_gaps)) if overlap_4_time_gaps else 0.0
            ),
            "gap_consistency_3": (
                1.0 / (1.0 + float(np.std(overlap_3_time_gaps)))
                if overlap_3_time_gaps
                else 0.0
            ),
            "gap_consistency_4": (
                1.0 / (1.0 + float(np.std(overlap_4_time_gaps)))
                if overlap_4_time_gaps
                else 0.0
            ),
        }

        # 패턴 일관성 분석
        pattern_consistency_3 = 0.0
        if overlap_3_patterns:
            consistent_patterns = sum(
                1
                for pattern_data in overlap_3_patterns.values()
                if len(pattern_data) > 1
            )
            pattern_consistency_3 = consistent_patterns / len(overlap_3_patterns)

        pattern_consistency_4 = 0.0
        if overlap_4_patterns:
            consistent_patterns = sum(
                1
                for pattern_data in overlap_4_patterns.values()
                if len(pattern_data) > 1
            )
            pattern_consistency_4 = consistent_patterns / len(overlap_4_patterns)

        result["pattern_consistency"] = {
            "consistency_3": float(pattern_consistency_3),
            "consistency_4": float(pattern_consistency_4),
        }

        return result

    def _analyze_overlap_roi_correlation(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        3~4자리 중복 패턴별 실제 ROI 성능 상관관계 분석

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 중복 패턴 ROI 상관관계 분석 결과
        """
        result = {}

        # 패턴별 ROI 점수 계산을 위한 기본 설정
        ticket_price = 1000  # 기본 로또 티켓 가격
        prize_structure = {
            6: 2000000000,  # 1등 평균 상금 (20억)
            5: 1500000,  # 2등 평균 상금 (150만)
            4: 50000,  # 3등 평균 상금 (5만)
            3: 5000,  # 4등 평균 상금 (5천)
        }

        # 3자리 중복 패턴 ROI 분석
        overlap_3_roi_scores = []
        overlap_4_roi_scores = []

        # 모든 회차에 대해 중복 패턴과 실제 성과 분석
        for i in range(len(historical_data)):
            current_numbers = set(historical_data[i].numbers)

            # 이전 회차들과의 3자리, 4자리 중복 확인
            has_3_overlap = False
            has_4_overlap = False

            for j in range(max(0, i - 50), i):  # 최근 50회차 내에서 중복 확인
                other_numbers = set(historical_data[j].numbers)
                overlap_count = len(current_numbers & other_numbers)

                if overlap_count == 3:
                    has_3_overlap = True
                elif overlap_count == 4:
                    has_4_overlap = True

            # ROI 점수 계산 (단순화된 백테스트 방식)
            # 실제로는 다음 회차 당첨 여부를 확인해야 하지만,
            # 여기서는 패턴의 통계적 성과를 추정
            if i < len(historical_data) - 1:
                next_draw = historical_data[i + 1]
                match_count = len(current_numbers & set(next_draw.numbers))

                # 예상 수익 계산
                expected_return = 0
                if match_count in prize_structure:
                    expected_return = prize_structure[match_count]

                roi = (expected_return - ticket_price) / ticket_price

                if has_3_overlap:
                    overlap_3_roi_scores.append(roi)
                if has_4_overlap:
                    overlap_4_roi_scores.append(roi)

        # 상관관계 계산
        overlap_3_roi_correlation = 0.0
        if overlap_3_roi_scores:
            # 3자리 중복 패턴의 평균 ROI
            avg_3_roi = np.mean(overlap_3_roi_scores)
            # 전체 평균 ROI와의 차이를 상관관계로 근사
            overall_avg_roi = -0.5  # 로또의 일반적인 기대 ROI
            overlap_3_roi_correlation = min(
                1.0, max(-1.0, (avg_3_roi - overall_avg_roi) * 2)
            )

        overlap_4_roi_correlation = 0.0
        if overlap_4_roi_scores:
            # 4자리 중복 패턴의 평균 ROI
            avg_4_roi = np.mean(overlap_4_roi_scores)
            # 전체 평균 ROI와의 차이를 상관관계로 근사
            overall_avg_roi = -0.5  # 로또의 일반적인 기대 ROI
            overlap_4_roi_correlation = min(
                1.0, max(-1.0, (avg_4_roi - overall_avg_roi) * 2)
            )

        result["overlap_3_roi_correlation"] = float(overlap_3_roi_correlation)
        result["overlap_4_roi_correlation"] = float(overlap_4_roi_correlation)

        # 추가 통계 정보
        result["overlap_3_roi_stats"] = {
            "sample_count": len(overlap_3_roi_scores),
            "avg_roi": (
                float(np.mean(overlap_3_roi_scores)) if overlap_3_roi_scores else 0.0
            ),
            "std_roi": (
                float(np.std(overlap_3_roi_scores)) if overlap_3_roi_scores else 0.0
            ),
            "positive_roi_ratio": (
                float(
                    sum(1 for roi in overlap_3_roi_scores if roi > 0)
                    / len(overlap_3_roi_scores)
                )
                if overlap_3_roi_scores
                else 0.0
            ),
        }

        result["overlap_4_roi_stats"] = {
            "sample_count": len(overlap_4_roi_scores),
            "avg_roi": (
                float(np.mean(overlap_4_roi_scores)) if overlap_4_roi_scores else 0.0
            ),
            "std_roi": (
                float(np.std(overlap_4_roi_scores)) if overlap_4_roi_scores else 0.0
            ),
            "positive_roi_ratio": (
                float(
                    sum(1 for roi in overlap_4_roi_scores if roi > 0)
                    / len(overlap_4_roi_scores)
                )
                if overlap_4_roi_scores
                else 0.0
            ),
        }

        return result
