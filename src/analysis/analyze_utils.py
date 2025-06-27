"""
분석 유틸리티 모듈

PatternAnalyzer에서 사용되는 다양한 분석 메서드들을 제공합니다.
"""

import numpy as np
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional
from ..shared.types import LotteryNumber


def analyze_segment_frequency_10(
    data: List[LotteryNumber], logger: logging.Logger = None
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
    data: List[LotteryNumber], logger: logging.Logger = None
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
    data: List[LotteryNumber], logger: logging.Logger = None
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


def analyze_pattern_reappearance(
    data: List[LotteryNumber], logger: logging.Logger = None
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
    data: List[LotteryNumber], window_size: int = 50, logger: logging.Logger = None
) -> Dict[str, Any]:
    """최근 재출현 간격 분석"""
    try:
        recent_data = data[-window_size:] if len(data) >= window_size else data
        recent_result = analyze_pattern_reappearance(recent_data, logger)

        return {
            "recent_avg_gap": recent_result["avg_reappearance_gap"],
            "recent_std_gap": recent_result["std_reappearance_gap"],
            "recent_reappearances": recent_result["total_reappearances"],
            "analysis_window": len(recent_data),
            "reappearance_rate": (
                recent_result["total_reappearances"] / len(recent_data)
                if recent_data
                else 0.0
            ),
        }
    except Exception as e:
        if logger:
            logger.error(f"최근 재출현 간격 분석 중 오류: {e}")
        return {
            "recent_avg_gap": 15.0,
            "recent_std_gap": 8.0,
            "recent_reappearances": 0,
        }


def analyze_segment_centrality(
    data: List[LotteryNumber], logger: logging.Logger = None
) -> Dict[str, Any]:
    """세그먼트 중심성 분석"""
    try:
        segment_connections = defaultdict(int)
        total_connections = 0

        for draw in data:
            numbers = sorted(draw.numbers)
            segments = []
            for num in numbers:
                if 1 <= num <= 9:
                    segments.append(0)
                elif 10 <= num <= 18:
                    segments.append(1)
                elif 19 <= num <= 27:
                    segments.append(2)
                elif 28 <= num <= 36:
                    segments.append(3)
                else:
                    segments.append(4)

            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    if segments[i] != segments[j]:
                        pair = tuple(sorted([segments[i], segments[j]]))
                        segment_connections[pair] += 1
                        total_connections += 1

        centrality_scores = {}
        for i in range(5):
            connections = sum(
                count for pair, count in segment_connections.items() if i in pair
            )
            centrality_scores[f"segment_{i+1}"] = (
                connections / total_connections if total_connections > 0 else 0.0
            )

        return {
            "centrality_scores": centrality_scores,
            "total_connections": total_connections,
            "most_central_segment": (
                max(centrality_scores, key=centrality_scores.get)
                if centrality_scores
                else "segment_1"
            ),
            "avg_centrality": (
                float(np.mean(list(centrality_scores.values())))
                if centrality_scores
                else 0.0
            ),
        }
    except Exception as e:
        if logger:
            logger.error(f"세그먼트 중심성 분석 중 오류: {e}")
        return {
            "centrality_scores": {f"segment_{i+1}": 0.2 for i in range(5)},
            "total_connections": 0,
        }


def analyze_segment_consecutive_patterns(
    data: List[LotteryNumber], logger: logging.Logger = None
) -> Dict[str, Dict[str, int]]:
    """세그먼트 연속 패턴 분석"""
    try:
        consecutive_patterns = {}

        for i in range(5):
            segment_name = f"segment_{i+1}"
            consecutive_patterns[segment_name] = {
                "single": 0,
                "double": 0,
                "triple": 0,
                "quad_plus": 0,
            }

        for draw in data:
            numbers = sorted(draw.numbers)
            segment_counts = [0] * 5

            for num in numbers:
                if 1 <= num <= 9:
                    segment_counts[0] += 1
                elif 10 <= num <= 18:
                    segment_counts[1] += 1
                elif 19 <= num <= 27:
                    segment_counts[2] += 1
                elif 28 <= num <= 36:
                    segment_counts[3] += 1
                else:
                    segment_counts[4] += 1

            for i, count in enumerate(segment_counts):
                segment_name = f"segment_{i+1}"
                if count == 1:
                    consecutive_patterns[segment_name]["single"] += 1
                elif count == 2:
                    consecutive_patterns[segment_name]["double"] += 1
                elif count == 3:
                    consecutive_patterns[segment_name]["triple"] += 1
                elif count >= 4:
                    consecutive_patterns[segment_name]["quad_plus"] += 1

        return consecutive_patterns
    except Exception as e:
        if logger:
            logger.error(f"세그먼트 연속 패턴 분석 중 오류: {e}")
        return {
            f"segment_{i+1}": {"single": 0, "double": 0, "triple": 0, "quad_plus": 0}
            for i in range(5)
        }


def analyze_identical_draws(
    data: List[LotteryNumber], logger: logging.Logger = None
) -> Dict[str, Any]:
    """동일한 당첨 번호 조합 분석"""
    try:
        combinations = {}
        duplicate_count = 0
        duplicate_pairs = []

        for i, draw in enumerate(data):
            combination = tuple(sorted(draw.numbers))

            if combination in combinations:
                duplicate_count += 1
                duplicate_pairs.append((combinations[combination], i))
            else:
                combinations[combination] = i

        return {
            "duplicate_count": duplicate_count,
            "duplicate_ratio": duplicate_count / len(data) if data else 0.0,
            "unique_combinations": len(combinations),
            "duplicate_pairs": duplicate_pairs,
            "total_draws": len(data),
        }
    except Exception as e:
        if logger:
            logger.error(f"동일 조합 분석 중 오류: {e}")
        return {"duplicate_count": 0, "duplicate_ratio": 0.0, "unique_combinations": 0}


def calculate_position_frequency(
    data: List[LotteryNumber], logger: logging.Logger = None
) -> np.ndarray:
    """위치별 번호 빈도 계산"""
    try:
        position_freq = np.zeros((6, 45), dtype=float)

        for draw in data:
            sorted_numbers = sorted(draw.numbers)
            for pos, number in enumerate(sorted_numbers):
                if 1 <= number <= 45 and pos < 6:
                    position_freq[pos][number - 1] += 1

        for pos in range(6):
            total = np.sum(position_freq[pos])
            if total > 0:
                position_freq[pos] /= total

        return position_freq
    except Exception as e:
        if logger:
            logger.error(f"위치별 번호 빈도 계산 중 오류: {e}")
        return np.ones((6, 45), dtype=float) / 45


def calculate_position_number_stats(
    data: List[LotteryNumber], logger: logging.Logger = None
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
                position_stats[f"position_{pos+1}"] = {
                    "mean": float(np.mean(position_numbers)),
                    "std": float(np.std(position_numbers)),
                    "min": int(min(position_numbers)),
                    "max": int(max(position_numbers)),
                    "median": float(np.median(position_numbers)),
                    "mode": int(Counter(position_numbers).most_common(1)[0][0]),
                }
            else:
                position_stats[f"position_{pos+1}"] = {
                    "mean": 23.0,
                    "std": 13.0,
                    "min": 1,
                    "max": 45,
                    "median": 23.0,
                    "mode": 23,
                }

        return position_stats
    except Exception as e:
        if logger:
            logger.error(f"위치별 번호 통계 계산 중 오류: {e}")
        return {f"position_{pos+1}": {"mean": 23.0, "std": 13.0} for pos in range(6)}


def calculate_segment_trend_history(
    data: List[LotteryNumber], logger: logging.Logger = None
) -> np.ndarray:
    """세그먼트 추세 히스토리 계산"""
    try:
        segment_history = np.zeros((5, len(data)), dtype=float)

        for i, draw in enumerate(data):
            segment_counts = [0] * 5

            for number in draw.numbers:
                if 1 <= number <= 9:
                    segment_counts[0] += 1
                elif 10 <= number <= 18:
                    segment_counts[1] += 1
                elif 19 <= number <= 27:
                    segment_counts[2] += 1
                elif 28 <= number <= 36:
                    segment_counts[3] += 1
                else:
                    segment_counts[4] += 1

            for seg in range(5):
                segment_history[seg][i] = segment_counts[seg] / 6.0

        return segment_history
    except Exception as e:
        if logger:
            logger.error(f"세그먼트 추세 히스토리 계산 중 오류: {e}")
        return np.random.rand(5, len(data) if data else 100) * 0.5


def calculate_gap_deviation_score(
    data: List[LotteryNumber], logger: logging.Logger = None
) -> Dict[str, Any]:
    """갭 편차 점수 계산"""
    try:
        gap_stats = analyze_gap_statistics(data, logger)
        expected_gap_mean = 3.0
        actual_gap_mean = gap_stats["mean"]
        deviation_score = abs(actual_gap_mean - expected_gap_mean) / expected_gap_mean

        return {
            "deviation_score": float(deviation_score),
            "actual_mean": actual_gap_mean,
            "expected_mean": expected_gap_mean,
            "gap_std": gap_stats["std"],
            "normalized_score": min(1.0, deviation_score),
        }
    except Exception as e:
        if logger:
            logger.error(f"갭 편차 점수 계산 중 오류: {e}")
        return {"deviation_score": 0.0, "actual_mean": 3.0, "expected_mean": 3.0}


def calculate_combination_diversity_score(
    data: List[LotteryNumber], logger: logging.Logger = None
) -> Dict[str, Any]:
    """조합 다양성 점수 계산"""
    try:
        number_counts = defaultdict(int)
        total_numbers = 0

        for draw in data:
            for number in draw.numbers:
                number_counts[number] += 1
                total_numbers += 1

        frequencies = [count / total_numbers for count in number_counts.values()]
        entropy = -sum(f * np.log2(f) for f in frequencies if f > 0)
        max_entropy = np.log2(45)
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "diversity_score": float(diversity_score),
            "entropy": float(entropy),
            "max_entropy": float(max_entropy),
            "unique_numbers": len(number_counts),
            "number_variance": float(np.var(list(number_counts.values()))),
            "total_draws": len(data),
        }
    except Exception as e:
        if logger:
            logger.error(f"조합 다양성 점수 계산 중 오류: {e}")
        return {
            "diversity_score": 0.8,
            "entropy": 4.0,
            "max_entropy": 5.49,
            "unique_numbers": 45,
        }
