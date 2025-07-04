import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from ..shared.types import LotteryNumber
from ..utils.unified_performance import performance_monitor
from ..utils.unified_logging import get_logger
from dataclasses import dataclass


@dataclass
class DatasetSummary:
    """데이터셋 요약 통계"""

    total_count: int
    train_count: int
    validation_count: int
    train_ratio: float
    validation_ratio: float
    number_distribution: Dict[
        int, Dict[str, float]
    ]  # 번호별 출현 비율 (전체/학습/검증)
    even_odd_ratio: Dict[str, Tuple[float, float]]  # 짝수-홀수 비율 (학습/검증)
    high_low_ratio: Dict[str, Tuple[float, float]]  # 고범위-저범위 비율 (학습/검증)
    range_distribution: Dict[str, Dict[Tuple[int, int], float]]  # 번호 범위별 분포


class DatasetSplitter:
    """로또 데이터셋 분할 클래스"""

    def __init__(self):
        """초기화"""
        self.logger = get_logger(__name__)
        # 통합 성능 모니터링 사용

    def split_dataset(
        self,
        data: List[LotteryNumber],
        train_ratio: float = 0.8,
        random_split: bool = False,
    ) -> Tuple[List[LotteryNumber], List[LotteryNumber], DatasetSummary]:
        """
        데이터셋을 학습/검증 세트로 분할

        Args:
            data: 로또 번호 데이터
            train_ratio: 학습 데이터 비율
            random_split: 무작위 분할 여부 (시간순 분할이 기본값)

        Returns:
            학습 데이터, 검증 데이터, 데이터셋 요약 통계
        """
        self.logger.info(f"데이터셋 분할 시작 (학습 비율: {train_ratio:.2f})")

        with performance_monitor("split_dataset"):
            # 데이터 유효성 검사
            if not data:
                self.logger.error("빈 데이터셋")
                return [], [], self._create_empty_summary()

            total_count = len(data)

            # 분할 인덱스 계산
            split_idx = int(total_count * train_ratio)

            if random_split:
                # 무작위 분할
                indices = np.random.permutation(total_count)
                train_indices = indices[:split_idx]
                val_indices = indices[split_idx:]

                train_data = [data[i] for i in train_indices]
                val_data = [data[i] for i in val_indices]
            else:
                # 시간순 분할 (기본값)
                train_data = data[:split_idx]
                val_data = data[split_idx:]

            # 데이터셋 요약 통계 생성
            summary = self._create_dataset_summary(data, train_data, val_data)

            self.logger.info(
                f"데이터셋 분할 완료: 학습 {len(train_data)}개, 검증 {len(val_data)}개"
            )
            return train_data, val_data, summary

    def _create_dataset_summary(
        self,
        full_data: List[LotteryNumber],
        train_data: List[LotteryNumber],
        val_data: List[LotteryNumber],
    ) -> DatasetSummary:
        """데이터셋 요약 통계 생성"""
        total_count = len(full_data)
        train_count = len(train_data)
        val_count = len(val_data)

        # 번호별 출현 비율
        full_freq = self._calculate_number_distribution(full_data)
        train_freq = self._calculate_number_distribution(train_data)
        val_freq = self._calculate_number_distribution(val_data)

        number_distribution = {}
        for num in range(1, 46):
            number_distribution[num] = {
                "full": full_freq.get(num, 0),
                "train": train_freq.get(num, 0),
                "validation": val_freq.get(num, 0),
            }

        # 짝수-홀수 비율
        train_even_odd = self._calculate_even_odd_ratio(train_data)
        val_even_odd = self._calculate_even_odd_ratio(val_data)

        # 고범위-저범위 비율
        train_high_low = self._calculate_high_low_ratio(train_data)
        val_high_low = self._calculate_high_low_ratio(val_data)

        # 번호 범위별 분포
        train_range_dist = self._calculate_range_distribution(train_data)
        val_range_dist = self._calculate_range_distribution(val_data)

        return DatasetSummary(
            total_count=total_count,
            train_count=train_count,
            validation_count=val_count,
            train_ratio=train_count / total_count if total_count else 0,
            validation_ratio=val_count / total_count if total_count else 0,
            number_distribution=number_distribution,
            even_odd_ratio={"train": train_even_odd, "validation": val_even_odd},
            high_low_ratio={"train": train_high_low, "validation": val_high_low},
            range_distribution={
                "train": train_range_dist,
                "validation": val_range_dist,
            },
        )

    def _calculate_number_distribution(
        self, data: List[LotteryNumber]
    ) -> Dict[int, float]:
        """번호별 출현 비율 계산"""
        if not data:
            return {}

        counts = {}
        for draw in data:
            for num in draw.numbers:
                counts[num] = counts.get(num, 0) + 1

        total_draws = len(data)
        return {num: count / (total_draws * 6) for num, count in counts.items()}

    def _calculate_even_odd_ratio(
        self, data: List[LotteryNumber]
    ) -> Tuple[float, float]:
        """짝수-홀수 비율 계산"""
        if not data:
            return (0.5, 0.5)

        even_count = 0
        odd_count = 0

        for draw in data:
            for num in draw.numbers:
                if num % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1

        total = even_count + odd_count
        if total == 0:
            return (0.5, 0.5)

        return (even_count / total, odd_count / total)

    def _calculate_high_low_ratio(
        self, data: List[LotteryNumber]
    ) -> Tuple[float, float]:
        """고범위(24-45)-저범위(1-23) 비율 계산"""
        if not data:
            return (0.5, 0.5)

        low_count = 0
        high_count = 0

        for draw in data:
            for num in draw.numbers:
                if num <= 23:
                    low_count += 1
                else:
                    high_count += 1

        total = low_count + high_count
        if total == 0:
            return (0.5, 0.5)

        return (low_count / total, high_count / total)

    def _calculate_range_distribution(
        self, data: List[LotteryNumber]
    ) -> Dict[Tuple[int, int], float]:
        """번호 범위별 분포 계산"""
        if not data:
            return {}

        ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
        range_counts = {r: 0 for r in ranges}

        for draw in data:
            for num in draw.numbers:
                for r in ranges:
                    if r[0] <= num <= r[1]:
                        range_counts[r] = range_counts[r] + 1
                        break

        total = sum(range_counts.values())
        if total == 0:
            return {r: 0.0 for r in ranges}

        return {r: count / total for r, count in range_counts.items()}

    def _create_empty_summary(self) -> DatasetSummary:
        """빈 데이터셋 요약 통계 생성"""
        number_distribution = {
            num: {"full": 0.0, "train": 0.0, "validation": 0.0} for num in range(1, 46)
        }
        ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
        empty_range_dist = {r: 0.0 for r in ranges}

        return DatasetSummary(
            total_count=0,
            train_count=0,
            validation_count=0,
            train_ratio=0,
            validation_ratio=0,
            number_distribution=number_distribution,
            even_odd_ratio={"train": (0.5, 0.5), "validation": (0.5, 0.5)},
            high_low_ratio={"train": (0.5, 0.5), "validation": (0.5, 0.5)},
            range_distribution={
                "train": empty_range_dist,
                "validation": empty_range_dist,
            },
        )

    def get_distribution_difference(
        self, train_data: List[LotteryNumber], val_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """학습 및 검증 데이터셋 간의 분포 차이 계산"""
        train_freq = self._calculate_number_distribution(train_data)
        val_freq = self._calculate_number_distribution(val_data)

        # KL 발산 계산 (분포 차이 측정)
        kl_div = 0.0
        for num in range(1, 46):
            p = train_freq.get(num, 1e-10)  # 0 방지
            q = val_freq.get(num, 1e-10)  # 0 방지
            if p > 0 and q > 0:
                kl_div += p * np.log(p / q)

        # 짝수-홀수 비율 차이
        train_even_odd = self._calculate_even_odd_ratio(train_data)
        val_even_odd = self._calculate_even_odd_ratio(val_data)
        even_odd_diff = abs(train_even_odd[0] - val_even_odd[0])

        # 고범위-저범위 비율 차이
        train_high_low = self._calculate_high_low_ratio(train_data)
        val_high_low = self._calculate_high_low_ratio(val_data)
        high_low_diff = abs(train_high_low[0] - val_high_low[0])

        return {
            "kl_divergence": kl_div,
            "even_odd_difference": even_odd_diff,
            "high_low_difference": high_low_diff,
        }
