"""
로또 번호 분포 분석 모듈

이 모듈은 로또 번호의 다양한 분포 패턴(홀짝, 고저, 범위 등)을 분석하는 기능을 제공합니다.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

from .base_analyzer import BaseAnalyzer
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


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

        # 성능 최적화 시스템 초기화
        from ..utils.memory_manager import get_memory_manager

        self.memory_manager = get_memory_manager()

        # 분포 분석 설정
        self.range_segments = 9  # 1-45를 9개 구간으로 분할
        self.sum_ranges = [(50, 100), (101, 150), (151, 200), (201, 250), (251, 300)]

        logger.info("DistributionAnalyzer 초기화 완료")

    def analyze(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        다양한 분포 패턴을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 분석된 분포 패턴들 (직렬화 가능한 형태)
        """
        with self.memory_manager.allocation_scope():
            logger.info(f"분포 패턴 분석 시작: {len(historical_data)}개 데이터")

            # 캐시 확인
            cache_key = self._create_cache_key(
                "distribution_analysis", len(historical_data)
            )
            cached_result = self._check_cache(cache_key)

            if cached_result:
                logger.info("캐시된 분포 분석 결과 사용")
                return cached_result

            try:
                # 분석 수행
                even_odd_patterns = self._analyze_even_odd(historical_data)
                low_high_patterns = self._analyze_low_high(historical_data)
                range_patterns = self._analyze_range_distribution(historical_data)
                sum_patterns = self._analyze_sum_distribution(historical_data)
                gap_patterns = self._analyze_consecutive_gaps(historical_data)

                # 추가 분포 분석
                prime_patterns = self._analyze_prime_distribution(historical_data)
                decade_patterns = self._analyze_decade_distribution(historical_data)
                position_patterns = self._analyze_position_distribution(historical_data)

                # DistributionPattern 객체들을 딕셔너리로 변환
                result = {
                    "even_odd": [
                        p.to_dict() if hasattr(p, "to_dict") else p
                        for p in even_odd_patterns
                    ],
                    "low_high": [
                        p.to_dict() if hasattr(p, "to_dict") else p
                        for p in low_high_patterns
                    ],
                    "ranges": [
                        p.to_dict() if hasattr(p, "to_dict") else p
                        for p in range_patterns
                    ],
                    "sum_ranges": [
                        p.to_dict() if hasattr(p, "to_dict") else p
                        for p in sum_patterns
                    ],
                    "gaps": [
                        p.to_dict() if hasattr(p, "to_dict") else p
                        for p in gap_patterns
                    ],
                    "prime_distribution": [
                        p.to_dict() if hasattr(p, "to_dict") else p
                        for p in prime_patterns
                    ],
                    "decade_distribution": [
                        p.to_dict() if hasattr(p, "to_dict") else p
                        for p in decade_patterns
                    ],
                    "position_distribution": [
                        p.to_dict() if hasattr(p, "to_dict") else p
                        for p in position_patterns
                    ],
                }

                # 결과 검증 - 함수 객체가 없는지 확인
                for key, patterns in result.items():
                    if isinstance(patterns, list):
                        for i, pattern in enumerate(patterns):
                            if callable(pattern):
                                logger.error(f"함수 객체 발견: {key}[{i}] = {pattern}")
                                # 함수 객체를 문자열로 변환
                                result[key][i] = str(pattern)

                # 결과 캐싱
                if hasattr(self, "cache_manager"):
                    self.cache_manager.set(cache_key, result)

                logger.info("분포 패턴 분석 완료")
                return result

            except Exception as e:
                logger.error(f"분포 분석 중 오류: {e}")
                return {"error": str(e), "analysis_type": "distribution"}

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """BaseAnalyzer 인터페이스 구현"""
        return self.analyze(historical_data)

    def _check_cache(
        self, cache_key: str
    ) -> Optional[Dict[str, List[DistributionPattern]]]:
        """캐시된 분석 결과를 확인하고 반환합니다."""
        try:
            if hasattr(self, "cache_manager"):
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
            logger.warning(f"캐시 데이터 액세스 오류: {str(e)}")

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
        return self._calculate_percentiles(patterns)

    def _analyze_low_high(self, data: List[LotteryNumber]) -> List[DistributionPattern]:
        """고저 분포 패턴을 분석합니다."""
        pattern_counts = {}
        total_draws = len(data)

        # 중간값 기준(1-22 낮음, 23-45 높음)
        for draw in data:
            low_count = sum(1 for n in draw.numbers if 1 <= n <= 22)
            high_count = 6 - low_count
            pattern = (low_count, high_count)

            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # 빈도 및 백분위 계산
        patterns = []
        for pattern, count in pattern_counts.items():
            frequency = count / total_draws
            patterns.append((pattern, frequency))

        patterns.sort(key=lambda x: x[1], reverse=True)
        return self._calculate_percentiles(patterns)

    def _analyze_range_distribution(
        self, data: List[LotteryNumber]
    ) -> List[DistributionPattern]:
        """범위별 분포 패턴을 분석합니다."""
        pattern_counts = {}
        total_draws = len(data)

        # 9개 구간으로 분할 (1-5, 6-10, ..., 41-45)
        ranges = [(i * 5 + 1, (i + 1) * 5) for i in range(9)]

        for draw in data:
            range_counts = [0] * 9
            for number in draw.numbers:
                for i, (start, end) in enumerate(ranges):
                    if start <= number <= end:
                        range_counts[i] += 1
                        break

            # 가장 많은 번호를 포함한 상위 2개 구간 패턴
            sorted_ranges = sorted(
                enumerate(range_counts), key=lambda x: x[1], reverse=True
            )
            top_ranges = tuple(sorted([sorted_ranges[0][0], sorted_ranges[1][0]]))

            pattern_counts[top_ranges] = pattern_counts.get(top_ranges, 0) + 1

        patterns = []
        for pattern, count in pattern_counts.items():
            frequency = count / total_draws
            patterns.append((pattern, frequency))

        patterns.sort(key=lambda x: x[1], reverse=True)
        return self._calculate_percentiles(patterns)

    def _analyze_sum_distribution(
        self, data: List[LotteryNumber]
    ) -> List[DistributionPattern]:
        """합계 분포 패턴을 분석합니다."""
        pattern_counts = {}
        total_draws = len(data)

        for draw in data:
            total_sum = sum(draw.numbers)

            # 합계 범위 분류
            range_id = -1
            for i, (min_sum, max_sum) in enumerate(self.sum_ranges):
                if min_sum <= total_sum <= max_sum:
                    range_id = i
                    break

            if range_id == -1:  # 범위 밖
                if total_sum < 50:
                    range_id = -2  # 매우 낮음
                else:
                    range_id = -3  # 매우 높음

            pattern = (range_id, total_sum // 20)  # 20단위로 그룹화
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        patterns = []
        for pattern, count in pattern_counts.items():
            frequency = count / total_draws
            patterns.append((pattern, frequency))

        patterns.sort(key=lambda x: x[1], reverse=True)
        return self._calculate_percentiles(patterns)

    def _analyze_consecutive_gaps(
        self, data: List[LotteryNumber]
    ) -> List[DistributionPattern]:
        """연속 번호 갭 패턴을 분석합니다."""
        pattern_counts = {}
        total_draws = len(data)

        for draw in data:
            sorted_numbers = sorted(draw.numbers)
            gaps = []

            for i in range(1, len(sorted_numbers)):
                gap = sorted_numbers[i] - sorted_numbers[i - 1]
                gaps.append(gap)

            # 갭 패턴 분류
            consecutive_count = sum(1 for gap in gaps if gap == 1)
            large_gap_count = sum(1 for gap in gaps if gap > 10)

            pattern = (consecutive_count, large_gap_count)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        patterns = []
        for pattern, count in pattern_counts.items():
            frequency = count / total_draws
            patterns.append((pattern, frequency))

        patterns.sort(key=lambda x: x[1], reverse=True)
        return self._calculate_percentiles(patterns)

    def _analyze_prime_distribution(
        self, data: List[LotteryNumber]
    ) -> List[DistributionPattern]:
        """소수 분포 패턴을 분석합니다."""
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
        pattern_counts = {}
        total_draws = len(data)

        for draw in data:
            prime_count = sum(1 for n in draw.numbers if n in primes)
            composite_count = 6 - prime_count
            pattern = (prime_count, composite_count)

            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        patterns = []
        for pattern, count in pattern_counts.items():
            frequency = count / total_draws
            patterns.append((pattern, frequency))

        patterns.sort(key=lambda x: x[1], reverse=True)
        return self._calculate_percentiles(patterns)

    def _analyze_decade_distribution(
        self, data: List[LotteryNumber]
    ) -> List[DistributionPattern]:
        """10의 자리 분포 패턴을 분석합니다."""
        pattern_counts = {}
        total_draws = len(data)

        for draw in data:
            decade_counts = [0] * 5  # 0-9, 10-19, 20-29, 30-39, 40-45

            for number in draw.numbers:
                decade = min(number // 10, 4)  # 40-45는 4번째 그룹
                decade_counts[decade] += 1

            # 상위 2개 10의 자리 패턴
            sorted_decades = sorted(
                enumerate(decade_counts), key=lambda x: x[1], reverse=True
            )
            pattern = tuple(sorted([sorted_decades[0][0], sorted_decades[1][0]]))

            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        patterns = []
        for pattern, count in pattern_counts.items():
            frequency = count / total_draws
            patterns.append((pattern, frequency))

        patterns.sort(key=lambda x: x[1], reverse=True)
        return self._calculate_percentiles(patterns)

    def _analyze_position_distribution(
        self, data: List[LotteryNumber]
    ) -> List[DistributionPattern]:
        """위치별 번호 분포 패턴을 분석합니다."""
        pattern_counts = {}
        total_draws = len(data)

        for draw in data:
            sorted_numbers = sorted(draw.numbers)

            # 첫 번째와 마지막 번호의 위치 패턴
            first_range = (sorted_numbers[0] - 1) // 15  # 0, 1, 2 (1-15, 16-30, 31-45)
            last_range = (sorted_numbers[-1] - 1) // 15

            pattern = (first_range, last_range)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        patterns = []
        for pattern, count in pattern_counts.items():
            frequency = count / total_draws
            patterns.append((pattern, frequency))

        patterns.sort(key=lambda x: x[1], reverse=True)
        return self._calculate_percentiles(patterns)

    def _calculate_percentiles(
        self, patterns: List[Tuple[Tuple[int, int], float]]
    ) -> List[DistributionPattern]:
        """패턴 리스트에서 백분위를 계산합니다."""
        total_freq = sum(freq for _, freq in patterns)
        cum_freq = 0
        result = []

        for pattern, freq in patterns:
            cum_freq += freq
            percentile = cum_freq / total_freq if total_freq > 0 else 0
            result.append(
                DistributionPattern(
                    pattern=pattern, frequency=freq, percentile=percentile
                )
            )

        return result

    def get_distribution_summary(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """분포 분석 요약 정보를 반환합니다."""
        analysis_result = self.analyze(historical_data)

        summary = {}
        for category, patterns in analysis_result.items():
            if patterns:
                most_common = patterns[0]
                summary[category] = {
                    "most_common_pattern": most_common.pattern,
                    "frequency": most_common.frequency,
                    "total_patterns": len(patterns),
                    "diversity_score": self._calculate_diversity_score(patterns),
                }

        return summary

    def _calculate_diversity_score(self, patterns: List[DistributionPattern]) -> float:
        """패턴 다양성 점수를 계산합니다."""
        if not patterns:
            return 0.0

        # 엔트로피 기반 다양성 계산
        entropy = 0.0
        for pattern in patterns:
            if pattern.frequency > 0:
                entropy -= pattern.frequency * math.log2(pattern.frequency)

        # 정규화 (0-1 범위)
        max_entropy = math.log2(len(patterns)) if len(patterns) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0
