"""
로또 번호 분포 분석 모듈

이 모듈은 로또 번호의 다양한 분포 패턴(홀짝, 고저, 범위 등)을 분석하는 기능을 제공합니다.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .base_analyzer import BaseAnalyzer
from src.shared.types import LotteryNumber


@dataclass
class DistributionPattern:
    """분포 패턴을 나타내는 데이터 클래스"""

    pattern: Tuple[int, int]
    frequency: float
    percentile: float

    @classmethod
    def from_dict(cls, data: Dict) -> "DistributionPattern":
        """딕셔너리에서 DistributionPattern 객체를 생성합니다."""
        return cls(
            pattern=tuple(data.get("pattern", (0, 0))),
            frequency=data.get("frequency", 0.0),
            percentile=data.get("percentile", 0.0),
        )

    def to_dict(self) -> Dict:
        """DistributionPattern 객체를 딕셔너리로 변환합니다."""
        return {
            "pattern": self.pattern,
            "frequency": self.frequency,
            "percentile": self.percentile,
        }


class DistributionAnalyzer(BaseAnalyzer[Dict[str, List[DistributionPattern]]]):
    """로또 번호의 분포 패턴을 분석하는 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        DistributionAnalyzer 초기화

        Args:
            config: 분포 분석에 사용할 설정
        """
        super().__init__(config or {}, "distribution")

    def analyze(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, List[DistributionPattern]]:
        """
        다양한 분포 패턴을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, List[DistributionPattern]]: 분석된 분포 패턴들
        """
        with self.performance_tracker.track("distribution_pattern_analysis"):
            # 캐시 확인
            cache_key = self._create_cache_key(
                "distribution_analysis", len(historical_data)
            )
            cached_result = self._check_cache(cache_key)

            if cached_result:
                return cached_result

            # 분석 수행
            even_odd_patterns = self._analyze_even_odd(historical_data)
            low_high_patterns = self._analyze_low_high(historical_data)
            range_patterns = self._analyze_range_distribution(historical_data)
            sum_patterns = self._analyze_sum_distribution(historical_data)
            gap_patterns = self._analyze_consecutive_gaps(historical_data)

            result = {
                "even_odd": even_odd_patterns,
                "low_high": low_high_patterns,
                "ranges": range_patterns,
                "sum_ranges": sum_patterns,
                "gaps": gap_patterns,
            }

            # 결과 캐싱
            self.cache_manager.set(cache_key, result)

            return result

    def _check_cache(
        self, cache_key: str
    ) -> Optional[Dict[str, List[DistributionPattern]]]:
        """캐시된 분석 결과를 확인하고 반환합니다."""
        try:
            cached_result = self.cache_manager.get(cache_key)

            if cached_result:
                # 캐시된 결과가 딕셔너리 형태로 반환되면 DistributionPattern 객체로 변환
                if isinstance(cached_result, dict):
                    result = {}
                    for key, patterns in cached_result.items():
                        if isinstance(patterns, list):
                            result[key] = [
                                (
                                    DistributionPattern.from_dict(pattern)
                                    if isinstance(pattern, dict)
                                    else pattern
                                )
                                for pattern in patterns
                            ]
                        else:
                            result[key] = patterns
                    return result
                return cached_result
        except Exception as e:
            print(f"캐시 데이터 액세스 오류: {str(e)}")
            # 오류 발생 시 캐시 무시

        return None

    def _analyze_even_odd(self, data: List[LotteryNumber]) -> List[DistributionPattern]:
        """홀짝 분포 패턴을 분석합니다."""
        pattern_counts = {}
        total_draws = len(data)

        # 패턴 빈도 카운트
        for draw in data:
            even_count = sum(1 for n in draw.numbers if n % 2 == 0)
            odd_count = 6 - even_count
            pattern = (even_count, odd_count)

            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # 빈도 및 백분위 계산
        patterns = []
        for pattern, count in pattern_counts.items():
            frequency = count / total_draws
            patterns.append((pattern, frequency))

        # 빈도별 정렬
        patterns.sort(key=lambda x: x[1], reverse=True)

        # 백분위 계산
        total_freq = sum(freq for _, freq in patterns)
        cum_freq = 0
        result = []

        for pattern, freq in patterns:
            cum_freq += freq
            percentile = cum_freq / total_freq
            result.append(
                DistributionPattern(
                    pattern=pattern, frequency=freq, percentile=percentile
                )
            )

        return result

    def _analyze_low_high(self, data: List[LotteryNumber]) -> List[DistributionPattern]:
        """고저 분포 패턴을 분석합니다."""
        pattern_counts = {}
        total_draws = len(data)

        # 중간값 기준(1-23 낮음, 24-45 높음)
        for draw in data:
            low_count = sum(1 for n in draw.numbers if 1 <= n <= 23)
            high_count = 6 - low_count
            pattern = (low_count, high_count)

            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # 빈도 및 백분위 계산
        patterns = []
        for pattern, count in pattern_counts.items():
            frequency = count / total_draws
            patterns.append((pattern, frequency))

        # 빈도별 정렬
        patterns.sort(key=lambda x: x[1], reverse=True)

        # 백분위 계산
        total_freq = sum(freq for _, freq in patterns)
        cum_freq = 0
        result = []

        for pattern, freq in patterns:
            cum_freq += freq
            percentile = cum_freq / total_freq
            result.append(
                DistributionPattern(
                    pattern=pattern, frequency=freq, percentile=percentile
                )
            )

        return result

    def _analyze_range_distribution(
        self, data: List[LotteryNumber]
    ) -> List[DistributionPattern]:
        """번호 범위 분포 패턴을 분석합니다."""
        range_counts = {i: 0 for i in range(5)}  # 5개 범위
        total_draws = len(data)

        # 범위 정의
        ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]

        # 각 범위별 번호 수 계산
        for draw in data:
            for num in draw.numbers:
                for i, (start, end) in enumerate(ranges):
                    if start <= num <= end:
                        range_counts[i] += 1
                        break

        # 빈도 계산
        total_numbers = total_draws * 6
        frequencies = {i: count / total_numbers for i, count in range_counts.items()}

        # 패턴 생성
        patterns = []
        for i, (start, end) in enumerate(ranges):
            patterns.append(
                DistributionPattern(
                    pattern=(start, end),
                    frequency=frequencies[i],
                    percentile=sum(
                        1 for f in frequencies.values() if f <= frequencies[i]
                    )
                    / len(frequencies),
                )
            )

        return sorted(patterns, key=lambda p: p.frequency, reverse=True)

    def _analyze_sum_distribution(
        self, data: List[LotteryNumber]
    ) -> List[DistributionPattern]:
        """합계 분포 패턴을 분석합니다."""
        sum_counts = {}
        total_draws = len(data)

        # 합계 범위 정의 (일반적으로 합계는 21~255 사이의 값)
        sum_ranges = {
            "very_low": (21, 100),
            "low": (101, 125),
            "medium": (126, 155),
            "high": (156, 180),
            "very_high": (181, 255),
        }
        range_counts = {k: 0 for k in sum_ranges}

        # 각 번호조합의 합계 계산 및 카운트
        for draw in data:
            total_sum = sum(draw.numbers)
            for range_name, (start, end) in sum_ranges.items():
                if start <= total_sum <= end:
                    range_counts[range_name] += 1
                    break

        # 패턴 생성
        patterns = []
        for range_name, count in range_counts.items():
            frequency = count / total_draws
            patterns.append(
                DistributionPattern(
                    pattern=sum_ranges[range_name],  # 범위 튜플
                    frequency=frequency,
                    percentile=sum(
                        1 for f in range_counts.values() if f / total_draws <= frequency
                    )
                    / len(range_counts),
                )
            )

        return sorted(patterns, key=lambda p: p.frequency, reverse=True)

    def _analyze_consecutive_gaps(
        self, data: List[LotteryNumber]
    ) -> List[DistributionPattern]:
        """연속된 번호간 간격 패턴을 분석합니다."""
        # 간격 범위 정의
        gap_ranges = [(1, 3), (4, 6), (7, 10), (11, 15), (16, 44)]
        gap_counts = {gap_range: 0 for gap_range in gap_ranges}
        total_gaps = 0

        # 각 번호조합의 연속 간격 분석
        for draw in data:
            sorted_numbers = sorted(draw.numbers)
            for i in range(len(sorted_numbers) - 1):
                gap = sorted_numbers[i + 1] - sorted_numbers[i]
                total_gaps += 1
                for gap_range in gap_ranges:
                    if gap_range[0] <= gap <= gap_range[1]:
                        gap_counts[gap_range] += 1
                        break

        # 패턴 생성
        patterns = []
        for gap_range, count in gap_counts.items():
            frequency = count / total_gaps
            patterns.append(
                DistributionPattern(
                    pattern=gap_range,
                    frequency=frequency,
                    percentile=sum(
                        1 for f in gap_counts.values() if f / total_gaps <= frequency
                    )
                    / len(gap_counts),
                )
            )

        return sorted(patterns, key=lambda p: p.frequency, reverse=True)
