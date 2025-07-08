"""
트렌드/시계열 분석기 모듈

이 모듈은 로또 번호의 시간에 따른 패턴 변화와 추세를 분석하는 기능을 제공합니다.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from collections import Counter

# logging 제거 - unified_logging 사용
import time

from ..utils.unified_logging import get_logger
from ..shared.types import LotteryNumber
from ..utils.unified_config import ConfigProxy
from .base_analyzer import BaseAnalyzer

# 로거 설정
logger = get_logger(__name__)


class TrendAnalyzer(BaseAnalyzer):
    """
    추세 분석기 클래스

    로또 당첨 번호의 시계열적 추세를 분석합니다.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigProxy]] = None):
        """
        초기화

        Args:
            config: 설정 객체
        """
        super().__init__(config, analyzer_type="trend")
        self.trend_window = 50
        self.segment_window = 20
        
        logger.info("TrendAnalyzer가 CPU 모드로 초기화되었습니다.")

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        추세 분석 구현

        Args:
            historical_data: 과거 당첨 번호 데이터
            *args, **kwargs: 추가 매개변수

        Returns:
            Dict[str, Any]: 추세 분석 결과
        """
        self.logger.info(f"추세 분석 시작: {len(historical_data)}개 데이터")

        # 데이터가 너무 적으면 분석 불가
        if len(historical_data) < self.trend_window:
            self.logger.warning(
                f"분석할 데이터가 부족합니다. 최소 {self.trend_window}개 필요함 (현재: {len(historical_data)}개)"
            )
            return {"error": "데이터 부족", "trend_features": {}}

        # 최근 데이터만 사용 (설정에 따라)
        window_size = kwargs.get("window_size", self.trend_window)
        use_all_data = kwargs.get("use_all_data", False)

        if not use_all_data and len(historical_data) > window_size:
            analysis_data = historical_data[-window_size:]
        else:
            analysis_data = historical_data

        # 회차별 번호 추출
        draw_numbers = [lottery.numbers for lottery in analysis_data]

        # 추세 특성 추출
        trend_features = self._extract_trend_features(draw_numbers)

        # 결과 반환
        return {
            "trend_features": trend_features,
            "data_count": len(historical_data),
            "window_size": window_size,
            "timestamp": int(time.time()),
        }

    def _extract_trend_features(self, draw_numbers: List[List[int]]) -> Dict[str, Any]:
        """
        추세 특성 추출

        Args:
            draw_numbers: 회차별 당첨 번호

        Returns:
            Dict[str, Any]: 추세 특성
        """
        features = {}

        # 번호별 출현 빈도 추세
        features.update(self._analyze_frequency_trend(draw_numbers))

        # 세그먼트 반복 패턴
        features.update(self._analyze_segment_repeat_patterns(draw_numbers))

        # 연속성 패턴
        features.update(self._analyze_consecutive_patterns(draw_numbers))

        return features

    def _analyze_frequency_trend(
        self, draw_numbers: List[List[int]]
    ) -> Dict[str, float]:
        """
        번호별 출현 빈도의 추세를 분석합니다.

        Args:
            draw_numbers: 회차별 당첨 번호

        Returns:
            Dict[str, float]: 빈도 추세 분석 결과
        """
        # 전체 번호 출현 빈도 계산
        all_numbers = [num for draw in draw_numbers for num in draw]
        frequency = Counter(all_numbers)

        # 최근 절반과 이전 절반의 빈도 비교
        mid_point = len(draw_numbers) // 2
        recent_numbers = [num for draw in draw_numbers[mid_point:] for num in draw]
        old_numbers = [num for draw in draw_numbers[:mid_point] for num in draw]

        recent_freq = Counter(recent_numbers)
        old_freq = Counter(old_numbers)

        # 추세 점수 계산
        trend_score = 0.0
        total_numbers = len(set(all_numbers))

        for num in range(1, 46):
            recent_count = recent_freq.get(num, 0)
            old_count = old_freq.get(num, 0)

            if old_count > 0:
                trend_ratio = recent_count / old_count
                trend_score += abs(trend_ratio - 1.0)

        return {
            "frequency_trend_score": (
                trend_score / total_numbers if total_numbers > 0 else 0.0
            ),
            "most_frequent_number": frequency.most_common(1)[0][0] if frequency else 0,
            "frequency_variance": (
                np.var(list(frequency.values())) if frequency else 0.0
            ),
        }

    def _analyze_consecutive_patterns(
        self, draw_numbers: List[List[int]]
    ) -> Dict[str, float]:
        """
        연속성 패턴 분석

        Args:
            draw_numbers: 회차별 당첨 번호

        Returns:
            Dict[str, float]: 연속성 패턴 분석 결과
        """
        consecutive_counts = []

        for numbers in draw_numbers:
            sorted_numbers = sorted(numbers)
            consecutive_count = 0

            for i in range(len(sorted_numbers) - 1):
                if sorted_numbers[i + 1] - sorted_numbers[i] == 1:
                    consecutive_count += 1

            consecutive_counts.append(consecutive_count)

        return {
            "avg_consecutive_count": (
                np.mean(consecutive_counts) if consecutive_counts else 0.0
            ),
            "consecutive_variance": (
                np.var(consecutive_counts) if consecutive_counts else 0.0
            ),
            "max_consecutive_count": (
                max(consecutive_counts) if consecutive_counts else 0
            ),
        }

    def _analyze_segment_repeat_patterns(
        self, draw_numbers: List[List[int]]
    ) -> Dict[str, float]:
        """
        세그먼트 반복 패턴 분석 (TrendAnalyzer 특화 버전)

        Args:
            draw_numbers: 회차별 당첨 번호

        Returns:
            Dict[str, float]: 세그먼트 반복 패턴 분석 결과
        """
        # 구간별 번호 출현 세그먼트 분석 (1-9, 10-19, 20-29, 30-39, 40-45)
        segments = [
            (1, 9),
            (10, 19),
            (20, 29),
            (30, 39),
            (40, 45),
        ]

        # 최근 N회차 세그먼트 패턴
        window_size = min(self.segment_window, len(draw_numbers))
        recent_draws = draw_numbers[-window_size:]

        # 각 회차별 세그먼트 카운트
        segment_counts = []
        for numbers in recent_draws:
            counts = [0] * len(segments)
            for num in numbers:
                for i, (start, end) in enumerate(segments):
                    if start <= num <= end:
                        counts[i] += 1
                        break
            segment_counts.append(tuple(counts))

        # 세그먼트 패턴 반복률 계산
        unique_patterns = len(set(segment_counts))
        repeat_ratio = (
            1.0 - (unique_patterns / len(segment_counts)) if segment_counts else 0.5
        )

        return {
            "segment_repeat_score": float(repeat_ratio),
        }
