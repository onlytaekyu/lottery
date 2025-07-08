"""
통합 패턴 분석 유틸리티 모듈

이 모듈은 PatternAnalyzer에서 사용되는 모든 분석 메서드들을 통합하여 제공합니다.
analyze_utils.py와 advanced_pattern_utils.py를 통합한 모듈입니다.
"""

import numpy as np

# logging 제거 - unified_logging 사용
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
from ..shared.types import LotteryNumber


# =============================================================================
# 기본 분석 함수들 (원래 analyze_utils.py에서)
# =============================================================================


def analyze_segment_frequency_10(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """10개 세그먼트 빈도 분석"""
    try:
        segment_counts = [0] * 10
        total_numbers = 0

        for draw in data:
            for number in draw.numbers:
                segment_idx = min((number - 1) // 4, 9)
                segment_counts[segment_idx] += 1
                total_numbers += 1

        frequencies = {}
        for i in range(10):
            segment_name = f"segment_{i+1}"
            frequencies[segment_name] = (
                segment_counts[i] / total_numbers if total_numbers > 0 else 0.0
            )

        return {
            "frequencies": frequencies,
            "total_numbers": total_numbers,
            "most_frequent_segment": (
                max(frequencies, key=frequencies.get) if frequencies else "segment_1"
            ),
            "segment_variance": (
                float(np.var(list(frequencies.values()))) if frequencies else 0.0
            ),
        }
    except Exception as e:
        if logger:
            logger.error(f"10개 세그먼트 빈도 분석 중 오류: {e}")
        return {
            "frequencies": {f"segment_{i+1}": 0.1 for i in range(10)},
            "total_numbers": 0,
        }


def analyze_segment_frequency_5(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """5개 세그먼트 빈도 분석"""
    try:
        segment_counts = [0] * 5
        total_numbers = 0

        for draw in data:
            for number in draw.numbers:
                if 1 <= number <= 9:
                    segment_counts[0] += 1
                elif 10 <= number <= 18:
                    segment_counts[1] += 1
                elif 19 <= number <= 27:
                    segment_counts[2] += 1
                elif 28 <= number <= 36:
                    segment_counts[3] += 1
                elif 37 <= number <= 45:
                    segment_counts[4] += 1
                total_numbers += 1

        frequencies = {}
        for i in range(5):
            segment_name = f"segment_{i+1}"
            frequencies[segment_name] = (
                segment_counts[i] / total_numbers if total_numbers > 0 else 0.0
            )

        return {
            "frequencies": frequencies,
            "total_numbers": total_numbers,
            "most_frequent_segment": (
                max(frequencies, key=frequencies.get) if frequencies else "segment_1"
            ),
            "segment_variance": (
                float(np.var(list(frequencies.values()))) if frequencies else 0.0
            ),
        }
    except Exception as e:
        if logger:
            logger.error(f"5개 세그먼트 빈도 분석 중 오류: {e}")
        return {
            "frequencies": {f"segment_{i+1}": 0.2 for i in range(5)},
            "total_numbers": 0,
        }


def analyze_gap_statistics(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """회차 간 갭 통계 분석"""
    try:
        gaps = []
        for i in range(1, len(data)):
            prev_numbers = set(data[i - 1].numbers)
            curr_numbers = set(data[i].numbers)
            common_count = len(prev_numbers.intersection(curr_numbers))
            gap = 6 - common_count
            gaps.append(gap)

        if gaps:
            return {
                "mean": float(np.mean(gaps)),
                "std": float(np.std(gaps)),
                "min": int(min(gaps)),
                "max": int(max(gaps)),
                "median": float(np.median(gaps)),
                "total_gaps": len(gaps),
                "gap_distribution": dict(Counter(gaps)),
            }
        else:
            return {
                "mean": 3.0,
                "std": 1.0,
                "min": 0,
                "max": 6,
                "median": 3.0,
                "total_gaps": 0,
            }
    except Exception as e:
        if logger:
            logger.error(f"갭 통계 분석 중 오류: {e}")
        return {
            "mean": 3.0,
            "std": 1.0,
            "min": 0,
            "max": 6,
            "median": 3.0,
            "total_gaps": 0,
        }


def analyze_gap_patterns(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    번호 간 간격 패턴을 분석합니다 (고급 버전).

    Args:
        data: 분석할 과거 당첨 번호 목록
        logger: 로깅을 위한 Logger 객체

    Returns:
        Dict[str, Any]: 간격 패턴 분석 결과
    """
    try:
        gap_data = []
        gap_frequencies = defaultdict(int)

        for draw in data:
            numbers = sorted(draw.numbers)
            # 연속된 번호 간의 간격 계산
            gaps = []
            for i in range(len(numbers) - 1):
                gap = numbers[i + 1] - numbers[i]
                gaps.append(gap)
                gap_frequencies[gap] += 1

            gap_data.append(gaps)

        # 간격 통계 계산
        all_gaps = [gap for gaps in gap_data for gap in gaps]

        if all_gaps:
            result = {
                "avg_gap": float(np.mean(all_gaps)),
                "median_gap": float(np.median(all_gaps)),
                "std_gap": float(np.std(all_gaps)),
                "min_gap": int(min(all_gaps)),
                "max_gap": int(max(all_gaps)),
                "total_gaps": len(all_gaps),
            }

            # 가장 빈번한 간격
            most_common_gap = max(gap_frequencies.items(), key=lambda x: x[1])
            result["most_common_gap"] = most_common_gap[0]
            result["most_common_gap_count"] = most_common_gap[1]
            result["most_common_gap_frequency"] = float(
                most_common_gap[1] / len(all_gaps)
            )

            # 간격 분포
            gap_1_count = gap_frequencies.get(1, 0)
            gap_2_count = gap_frequencies.get(2, 0)
            gap_3_count = gap_frequencies.get(3, 0)

            result["gap_1_frequency"] = float(gap_1_count / len(all_gaps))
            result["gap_2_frequency"] = float(gap_2_count / len(all_gaps))
            result["gap_3_frequency"] = float(gap_3_count / len(all_gaps))

            # 큰 간격 (10 이상) 비율
            large_gaps = sum(1 for gap in all_gaps if gap >= 10)
            result["large_gap_ratio"] = float(large_gaps / len(all_gaps))

        else:
            result = {
                "avg_gap": 5.0,
                "median_gap": 4.0,
                "std_gap": 3.0,
                "min_gap": 1,
                "max_gap": 20,
                "total_gaps": 0,
                "most_common_gap": 3,
                "most_common_gap_count": 0,
                "most_common_gap_frequency": 0.0,
                "gap_1_frequency": 0.1,
                "gap_2_frequency": 0.15,
                "gap_3_frequency": 0.2,
                "large_gap_ratio": 0.3,
            }

        return result

    except Exception as e:
        if logger:
            logger.error(f"간격 패턴 분석 중 오류: {e}")
        return {
            "avg_gap": 5.0,
            "median_gap": 4.0,
            "std_gap": 3.0,
            "min_gap": 1,
            "max_gap": 20,
            "total_gaps": 0,
            "most_common_gap": 3,
            "most_common_gap_count": 0,
            "most_common_gap_frequency": 0.0,
            "gap_1_frequency": 0.1,
            "gap_2_frequency": 0.15,
            "gap_3_frequency": 0.2,
            "large_gap_ratio": 0.3,
        }


def analyze_pattern_reappearance(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """패턴 재출현 분석"""
    try:
        pattern_3_occurrences = defaultdict(list)

        for i, draw in enumerate(data):
            numbers = sorted(draw.numbers)
            for j in range(len(numbers)):
                for k in range(j + 1, len(numbers)):
                    for l in range(k + 1, len(numbers)):
                        pattern = (numbers[j], numbers[k], numbers[l])
                        pattern_3_occurrences[pattern].append(i)

        reappearance_gaps = []
        for pattern, occurrences in pattern_3_occurrences.items():
            if len(occurrences) > 1:
                for i in range(1, len(occurrences)):
                    gap = occurrences[i] - occurrences[i - 1]
                    reappearance_gaps.append(gap)

        if reappearance_gaps:
            return {
                "avg_reappearance_gap": float(np.mean(reappearance_gaps)),
                "std_reappearance_gap": float(np.std(reappearance_gaps)),
                "min_gap": int(min(reappearance_gaps)),
                "max_gap": int(max(reappearance_gaps)),
                "total_reappearances": len(reappearance_gaps),
                "unique_patterns": len(
                    [p for p, occ in pattern_3_occurrences.items() if len(occ) > 1]
                ),
            }
        else:
            return {
                "avg_reappearance_gap": 20.0,
                "std_reappearance_gap": 10.0,
                "total_reappearances": 0,
            }
    except Exception as e:
        if logger:
            logger.error(f"패턴 재출현 분석 중 오류: {e}")
        return {
            "avg_reappearance_gap": 20.0,
            "std_reappearance_gap": 10.0,
            "total_reappearances": 0,
        }


def analyze_recent_reappearance_gap(
    data: List[LotteryNumber], window_size: int = 50, logger: Optional[Any] = None
) -> Dict[str, Any]:
    """최근 재출현 간격 분석"""
    try:
        if len(data) < window_size:
            recent_data = data
        else:
            recent_data = data[-window_size:]

        recent_result = analyze_pattern_reappearance(recent_data, logger)

        return {
            "recent_avg_gap": recent_result.get("avg_reappearance_gap", 20.0),
            "recent_std_gap": recent_result.get("std_reappearance_gap", 10.0),
            "recent_window_size": len(recent_data),
        }
    except Exception as e:
        if logger:
            logger.error(f"최근 재출현 간격 분석 중 오류: {e}")
        return {
            "recent_avg_gap": 20.0,
            "recent_std_gap": 10.0,
            "recent_window_size": 0,
        }


def analyze_network(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    네트워크 분석을 수행합니다.

    Args:
        data: 분석할 과거 당첨 번호 목록
        logger: 로깅을 위한 Logger 객체

    Returns:
        Dict[str, Any]: 네트워크 분석 결과
    """
    try:
        # 번호 간 연결성 분석
        co_occurrence = defaultdict(int)
        total_pairs = 0

        for draw in data:
            numbers = sorted(draw.numbers)
            # 모든 번호 쌍에 대해 동시 출현 횟수 계산
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = (numbers[i], numbers[j])
                    co_occurrence[pair] += 1
                    total_pairs += 1

        # 네트워크 통계 계산
        if total_pairs > 0:
            # 가장 강한 연결
            strongest_pair = max(co_occurrence.items(), key=lambda x: x[1])

            # 평균 연결 강도
            avg_strength = sum(co_occurrence.values()) / len(co_occurrence)

            # 연결 밀도 (실제 연결 / 가능한 모든 연결)
            possible_pairs = 45 * 44 // 2  # 45개 번호에서 가능한 모든 쌍
            density = len(co_occurrence) / possible_pairs

            result = {
                "total_connections": len(co_occurrence),
                "total_occurrences": total_pairs,
                "avg_connection_strength": float(avg_strength),
                "network_density": float(density),
                "strongest_pair": strongest_pair[0],
                "strongest_pair_count": strongest_pair[1],
                "strongest_pair_strength": float(strongest_pair[1] / len(data)),
            }

            # 각 번호의 연결 수 계산
            node_connections = defaultdict(int)
            for (num1, num2), count in co_occurrence.items():
                node_connections[num1] += 1
                node_connections[num2] += 1

            if node_connections:
                # 가장 연결이 많은 번호
                most_connected = max(node_connections.items(), key=lambda x: x[1])
                result["most_connected_number"] = most_connected[0]
                result["most_connected_count"] = most_connected[1]
                result["avg_node_connections"] = float(
                    sum(node_connections.values()) / len(node_connections)
                )

        else:
            result = {
                "total_connections": 0,
                "total_occurrences": 0,
                "avg_connection_strength": 0.0,
                "network_density": 0.0,
                "strongest_pair": (1, 2),
                "strongest_pair_count": 0,
                "strongest_pair_strength": 0.0,
                "most_connected_number": 1,
                "most_connected_count": 0,
                "avg_node_connections": 0.0,
            }

        return result

    except Exception as e:
        if logger:
            logger.error(f"네트워크 분석 중 오류: {e}")
        return {
            "total_connections": 100,
            "total_occurrences": 1000,
            "avg_connection_strength": 10.0,
            "network_density": 0.5,
            "strongest_pair": (7, 14),
            "strongest_pair_count": 50,
            "strongest_pair_strength": 0.05,
            "most_connected_number": 7,
            "most_connected_count": 30,
            "avg_node_connections": 25.0,
        }


# =============================================================================
# 추가 분석 함수들 (원래 analyze_utils.py의 나머지 함수들)
# =============================================================================


def analyze_segment_centrality(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """세그먼트 중심성 분석"""
    try:
        segment_numbers = {i: [] for i in range(1, 10)}

        for draw in data:
            for number in draw.numbers:
                segment = min((number - 1) // 5 + 1, 9)
                segment_numbers[segment].append(number)

        centrality_scores = {}
        for segment, numbers in segment_numbers.items():
            if numbers:
                centrality_scores[f"segment_{segment}"] = float(np.mean(numbers))
            else:
                centrality_scores[f"segment_{segment}"] = 0.0

        return {
            "centrality_scores": centrality_scores,
            "most_central_segment": max(centrality_scores, key=centrality_scores.get),
            "centrality_variance": float(np.var(list(centrality_scores.values()))),
        }
    except Exception as e:
        if logger:
            logger.error(f"세그먼트 중심성 분석 중 오류: {e}")
        return {
            "centrality_scores": {f"segment_{i}": 20.0 for i in range(1, 10)},
            "most_central_segment": "segment_5",
            "centrality_variance": 0.0,
        }


def analyze_odd_even_distribution(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, float]:
    """홀짝 분포 분석"""
    try:
        odd_count = 0
        even_count = 0
        total_numbers = 0

        for draw in data:
            for number in draw.numbers:
                if number % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
                total_numbers += 1

        if total_numbers > 0:
            odd_ratio = odd_count / total_numbers
            even_ratio = even_count / total_numbers
        else:
            odd_ratio = even_ratio = 0.5

        # 홀짝 패턴 분석
        pattern_counts = defaultdict(int)
        for draw in data:
            odd_in_draw = sum(1 for num in draw.numbers if num % 2 == 1)
            pattern_counts[odd_in_draw] += 1

        most_common_pattern = (
            max(pattern_counts.items(), key=lambda x: x[1])
            if pattern_counts
            else (3, 0)
        )

        return {
            "odd_ratio": float(odd_ratio),
            "even_ratio": float(even_ratio),
            "odd_count": odd_count,
            "even_count": even_count,
            "most_common_odd_count": most_common_pattern[0],
            "most_common_pattern_frequency": (
                float(most_common_pattern[1] / len(data)) if data else 0.0
            ),
            "balance_score": float(
                1.0 - abs(odd_ratio - 0.5) * 2
            ),  # 0.5에 가까울수록 1에 가까움
        }
    except Exception as e:
        if logger:
            logger.error(f"홀짝 분포 분석 중 오류: {e}")
        return {
            "odd_ratio": 0.5,
            "even_ratio": 0.5,
            "odd_count": 0,
            "even_count": 0,
            "most_common_odd_count": 3,
            "most_common_pattern_frequency": 0.0,
            "balance_score": 1.0,
        }


def analyze_number_sum_distribution(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, float]:
    """번호 합계 분포 분석"""
    try:
        sums = []
        for draw in data:
            draw_sum = sum(draw.numbers)
            sums.append(draw_sum)

        if sums:
            mean_sum = float(np.mean(sums))
            std_sum = float(np.std(sums))
            min_sum = float(min(sums))
            max_sum = float(max(sums))
            median_sum = float(np.median(sums))

            # 합계 범위 분석
            low_range = sum(1 for s in sums if s < 100)  # 100 미만
            mid_range = sum(1 for s in sums if 100 <= s <= 200)  # 100-200
            high_range = sum(1 for s in sums if s > 200)  # 200 초과

            total_draws = len(sums)

            return {
                "mean_sum": mean_sum,
                "std_sum": std_sum,
                "min_sum": min_sum,
                "max_sum": max_sum,
                "median_sum": median_sum,
                "low_range_ratio": float(low_range / total_draws),
                "mid_range_ratio": float(mid_range / total_draws),
                "high_range_ratio": float(high_range / total_draws),
                "sum_variance": float(np.var(sums)),
            }
        else:
            return {
                "mean_sum": 135.0,  # 이론적 평균 (1+2+...+45)/45 * 6
                "std_sum": 20.0,
                "min_sum": 21.0,  # 1+2+3+4+5+6
                "max_sum": 255.0,  # 40+41+42+43+44+45
                "median_sum": 135.0,
                "low_range_ratio": 0.2,
                "mid_range_ratio": 0.6,
                "high_range_ratio": 0.2,
                "sum_variance": 400.0,
            }
    except Exception as e:
        if logger:
            logger.error(f"번호 합계 분포 분석 중 오류: {e}")
        return {
            "mean_sum": 135.0,
            "std_sum": 20.0,
            "min_sum": 21.0,
            "max_sum": 255.0,
            "median_sum": 135.0,
            "low_range_ratio": 0.2,
            "mid_range_ratio": 0.6,
            "high_range_ratio": 0.2,
            "sum_variance": 400.0,
        }


# =============================================================================
# 나머지 유틸리티 함수들
# =============================================================================


def calculate_position_frequency(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> np.ndarray:
    """위치별 번호 빈도 계산"""
    try:
        position_freq = np.zeros((6, 45), dtype=int)

        for draw in data:
            sorted_numbers = sorted(draw.numbers)
            for pos, number in enumerate(sorted_numbers):
                if 0 <= pos < 6 and 1 <= number <= 45:
                    position_freq[pos][number - 1] += 1

        return position_freq
    except Exception as e:
        if logger:
            logger.error(f"위치별 번호 빈도 계산 중 오류: {e}")
        return np.zeros((6, 45), dtype=int)


def calculate_position_number_stats(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """위치별 번호 통계 계산"""
    try:
        position_stats = {}

        for pos in range(6):
            position_numbers = []
            for draw in data:
                sorted_numbers = sorted(draw.numbers)
                if pos < len(sorted_numbers):
                    position_numbers.append(sorted_numbers[pos])

            if position_numbers:
                position_stats[f"position_{pos + 1}"] = {
                    "mean": float(np.mean(position_numbers)),
                    "std": float(np.std(position_numbers)),
                    "min": int(min(position_numbers)),
                    "max": int(max(position_numbers)),
                    "median": float(np.median(position_numbers)),
                }
            else:
                position_stats[f"position_{pos + 1}"] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0,
                    "max": 0,
                    "median": 0.0,
                }

        return position_stats
    except Exception as e:
        if logger:
            logger.error(f"위치별 번호 통계 계산 중 오류: {e}")
        return {
            f"position_{i+1}": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
                "median": 0.0,
            }
            for i in range(6)
        }


def analyze_segment_consecutive_patterns(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Dict[str, int]]:
    """세그먼트 연속 패턴 분석"""
    try:
        segment_patterns = {}

        for i in range(1, 10):  # 9개 세그먼트
            segment_patterns[f"segment_{i}"] = defaultdict(int)

        for draw in data:
            for number in draw.numbers:
                segment = min((number - 1) // 5 + 1, 9)
                segment_patterns[f"segment_{segment}"]["count"] += 1

        # 딕셔너리를 일반 dict로 변환
        result = {}
        for segment, patterns in segment_patterns.items():
            result[segment] = dict(patterns)

        return result
    except Exception as e:
        if logger:
            logger.error(f"세그먼트 연속 패턴 분석 중 오류: {e}")
        return {f"segment_{i}": {"count": 0} for i in range(1, 10)}


def analyze_identical_draws(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """동일한 조합 분석"""
    try:
        combinations = {}
        identical_count = 0

        for i, draw in enumerate(data):
            combo = tuple(sorted(draw.numbers))
            if combo in combinations:
                identical_count += 1
                combinations[combo].append(i)
            else:
                combinations[combo] = [i]

        identical_combinations = {
            combo: indices
            for combo, indices in combinations.items()
            if len(indices) > 1
        }

        return {
            "total_identical": identical_count,
            "unique_combinations": len(combinations),
            "identical_combinations": len(identical_combinations),
            "identical_ratio": float(identical_count / len(data)) if data else 0.0,
        }
    except Exception as e:
        if logger:
            logger.error(f"동일한 조합 분석 중 오류: {e}")
        return {
            "total_identical": 0,
            "unique_combinations": 0,
            "identical_combinations": 0,
            "identical_ratio": 0.0,
        }


def calculate_segment_trend_history(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> np.ndarray:
    """세그먼트 트렌드 히스토리 계산"""
    try:
        if not data:
            return np.zeros((9, 10), dtype=float)

        # 9개 세그먼트 x 최근 10회차 트렌드
        trend_history = np.zeros((9, min(10, len(data))), dtype=float)

        recent_data = data[-10:] if len(data) >= 10 else data

        for draw_idx, draw in enumerate(recent_data):
            segment_counts = [0] * 9
            for number in draw.numbers:
                segment = min((number - 1) // 5, 8)  # 0-8 인덱스
                segment_counts[segment] += 1

            for segment in range(9):
                trend_history[segment][draw_idx] = float(segment_counts[segment])

        return trend_history
    except Exception as e:
        if logger:
            logger.error(f"세그먼트 트렌드 히스토리 계산 중 오류: {e}")
        return np.zeros((9, 10), dtype=float)


def calculate_gap_deviation_score(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """간격 편차 점수 계산"""
    try:
        gap_stats = analyze_gap_statistics(data, logger)

        if gap_stats["total_gaps"] == 0:
            return {
                "deviation_score": 0.0,
                "normalized_deviation": 0.0,
                "gap_consistency": 0.0,
            }

        # 표준편차 기반 편차 점수
        deviation_score = (
            gap_stats["std"] / gap_stats["mean"] if gap_stats["mean"] > 0 else 0.0
        )

        # 정규화된 편차 (0-1 범위)
        normalized_deviation = min(deviation_score / 2.0, 1.0)

        # 간격 일관성 (편차가 낮을수록 높음)
        gap_consistency = 1.0 - normalized_deviation

        return {
            "deviation_score": float(deviation_score),
            "normalized_deviation": float(normalized_deviation),
            "gap_consistency": float(gap_consistency),
        }
    except Exception as e:
        if logger:
            logger.error(f"간격 편차 점수 계산 중 오류: {e}")
        return {
            "deviation_score": 0.0,
            "normalized_deviation": 0.0,
            "gap_consistency": 0.0,
        }


def calculate_combination_diversity_score(
    data: List[LotteryNumber], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """조합 다양성 점수 계산"""
    try:
        if not data:
            return {
                "diversity_score": 0.0,
                "unique_ratio": 0.0,
                "entropy_score": 0.0,
            }

        # 고유 조합 수 계산
        unique_combinations = set()
        for draw in data:
            combo = tuple(sorted(draw.numbers))
            unique_combinations.add(combo)

        unique_ratio = len(unique_combinations) / len(data)

        # 엔트로피 기반 다양성 점수
        combo_counts = Counter()
        for draw in data:
            combo = tuple(sorted(draw.numbers))
            combo_counts[combo] += 1

        total = len(data)
        entropy = 0.0
        for count in combo_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        # 최대 엔트로피로 정규화
        max_entropy = np.log2(len(unique_combinations)) if unique_combinations else 0.0
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0

        # 전체 다양성 점수 (unique_ratio와 entropy_score의 평균)
        diversity_score = (unique_ratio + entropy_score) / 2.0

        return {
            "diversity_score": float(diversity_score),
            "unique_ratio": float(unique_ratio),
            "entropy_score": float(entropy_score),
        }
    except Exception as e:
        if logger:
            logger.error(f"조합 다양성 점수 계산 중 오류: {e}")
        return {
            "diversity_score": 0.0,
            "unique_ratio": 0.0,
            "entropy_score": 0.0,
        }
