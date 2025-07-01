"""
베이스 트렌드 분석기 모듈

TrendAnalyzer와 TrendSequenceGenerator의 공통 기능을 제공하는 베이스 클래스입니다.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from collections import Counter, defaultdict
from scipy import stats

# logging 제거 - unified_logging 사용
import time
from datetime import datetime

from ..utils.unified_logging import get_logger
from ..shared.types import LotteryNumber
from ..utils.unified_config import ConfigProxy
from .base_analyzer import BaseAnalyzer


class BaseTrendAnalyzer(BaseAnalyzer[Dict[str, Any]]):
    """
    트렌드 분석 베이스 클래스

    TrendAnalyzer와 TrendSequenceGenerator의 공통 기능을 제공합니다.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], ConfigProxy]] = None,
        analyzer_type: str = "trend",
    ):
        """
        초기화

        Args:
            config: 설정 객체
            analyzer_type: 분석기 타입
        """
        super().__init__(config, analyzer_type)

        # 분석 설정
        try:
            self.trend_window = self.config["trend_analysis"]["window_size"]
        except (KeyError, TypeError):
            self.trend_window = 10  # 기본 윈도우 크기

        try:
            self.regression_window = self.config["trend_analysis"]["regression_window"]
        except (KeyError, TypeError):
            self.regression_window = 20  # 기본 회귀 윈도우 크기

        try:
            self.segment_window = self.config["trend_analysis"]["segment_window"]
        except (KeyError, TypeError):
            self.segment_window = 15  # 기본 세그먼트 윈도우 크기

    def _analyze_position_trends(
        self, draw_numbers: List[List[int]]
    ) -> Dict[str, float]:
        """
        위치별 추세 분석

        Args:
            draw_numbers: 회차별 당첨 번호

        Returns:
            Dict[str, float]: 위치별 추세 분석 결과
        """
        # 각 위치별 숫자 추출
        positions = [[] for _ in range(6)]

        for numbers in draw_numbers:
            sorted_numbers = sorted(numbers)
            for i, num in enumerate(sorted_numbers):
                positions[i].append(num)

        # 각 위치별 추세선 기울기 계산
        slopes = {}

        for i, pos_numbers in enumerate(positions):
            # 최소 3개 이상의 데이터가 필요
            if len(pos_numbers) >= 3:
                # 시간 축 (0, 1, 2, ...)
                x = np.arange(len(pos_numbers))
                # 선형 회귀로 기울기 계산
                slope, _ = np.polyfit(x, pos_numbers, 1)
                # 정규화: -1(하락) ~ 1(상승)
                normalized_slope = np.clip(slope / 5.0, -1.0, 1.0)
                slopes[f"position_trend_slope_{i+1}"] = float(normalized_slope)
            else:
                slopes[f"position_trend_slope_{i+1}"] = 0.0

        return slopes

    def _analyze_delta_patterns(
        self, draw_numbers: List[List[int]]
    ) -> Dict[str, float]:
        """
        증감 패턴 분석

        Args:
            draw_numbers: 회차별 당첨 번호

        Returns:
            Dict[str, float]: 증감 패턴 분석 결과
        """
        if len(draw_numbers) < 2:
            return {"delta_mean": 0.0, "delta_std": 0.0}

        # 회차간 평균 변화량 계산
        deltas = []
        for i in range(1, len(draw_numbers)):
            prev_set = set(draw_numbers[i - 1])
            curr_set = set(draw_numbers[i])

            # 달라진 번호 수
            changed = len(prev_set.symmetric_difference(curr_set)) / 2
            deltas.append(changed)

        # 평균 및 표준편차 계산
        delta_mean = np.mean(deltas) if deltas else 0.0
        delta_std = np.std(deltas) if len(deltas) > 1 else 0.0

        # 정규화: 0 ~ 1 범위로
        normalized_mean = delta_mean / 6.0  # 최대 6개 모두 변경 가능
        normalized_std = min(delta_std / 3.0, 1.0)  # 표준편차 정규화

        return {
            "delta_mean": float(normalized_mean),
            "delta_std": float(normalized_std),
        }

    def _analyze_segment_repeat_patterns(
        self, draw_numbers: List[List[int]]
    ) -> Dict[str, float]:
        """
        세그먼트 반복 패턴 분석

        Args:
            draw_numbers: 회차별 당첨 번호

        Returns:
            Dict[str, float]: 세그먼트 반복 패턴 분석 결과
        """
        if len(draw_numbers) < 2:
            return {"segment_repeat_score": 0.5}

        # 각 회차를 세그먼트로 분할 (1-9, 10-18, 19-27, 28-36, 37-45)
        segment_patterns = []

        for numbers in draw_numbers:
            segments = [0] * 5  # 5개 세그먼트
            for num in numbers:
                segment_idx = min((num - 1) // 9, 4)  # 0-4 인덱스
                segments[segment_idx] += 1
            segment_patterns.append(tuple(segments))

        # 반복 패턴 점수 계산
        pattern_counts = Counter(segment_patterns)
        total_patterns = len(segment_patterns)

        # 가장 빈번한 패턴의 비율
        if pattern_counts:
            max_count = max(pattern_counts.values())
            repeat_score = max_count / total_patterns
        else:
            repeat_score = 0.0

        return {"segment_repeat_score": float(repeat_score)}

    def _extract_trend_features(
        self, draw_numbers: List[List[int]]
    ) -> Dict[str, float]:
        """
        트렌드 특성 추출 (공통 메서드)

        Args:
            draw_numbers: 회차별 당첨 번호

        Returns:
            Dict[str, float]: 트렌드 특성
        """
        # 1. 위치별 추세 분석
        position_trends = self._analyze_position_trends(draw_numbers)

        # 2. 증감 패턴 분석
        delta_patterns = self._analyze_delta_patterns(draw_numbers)

        # 3. 세그먼트 반복 분석
        segment_patterns = self._analyze_segment_repeat_patterns(draw_numbers)

        # 특성 통합
        trend_features = {
            **position_trends,
            **delta_patterns,
            **segment_patterns,
        }

        return trend_features

    def _create_trend_vector(self, trend_features: Dict[str, float]) -> np.ndarray:
        """
        트렌드 특성을 벡터로 변환

        Args:
            trend_features: 트렌드 특성 딕셔너리

        Returns:
            np.ndarray: 9차원 트렌드 벡터
        """
        trend_vector = np.array(
            [
                trend_features.get("position_trend_slope_1", 0.0),
                trend_features.get("position_trend_slope_2", 0.0),
                trend_features.get("position_trend_slope_3", 0.0),
                trend_features.get("position_trend_slope_4", 0.0),
                trend_features.get("position_trend_slope_5", 0.0),
                trend_features.get("position_trend_slope_6", 0.0),
                trend_features.get("delta_mean", 0.0),
                trend_features.get("delta_std", 0.0),
                trend_features.get("segment_repeat_score", 0.5),
            ],
            dtype=np.float32,
        )

        return trend_vector

    def _normalize_features(self, sequence_tensor: np.ndarray) -> np.ndarray:
        """
        특성별 정규화

        Args:
            sequence_tensor: 시퀀스 텐서

        Returns:
            np.ndarray: 정규화된 시퀀스 텐서
        """
        try:
            # 각 특성별로 정규화
            normalized_tensor = sequence_tensor.copy()

            # 특성별 평균과 표준편차 계산
            for feature_idx in range(sequence_tensor.shape[-1]):
                feature_data = sequence_tensor[:, :, feature_idx]

                # 평균과 표준편차 계산
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)

                # 표준편차가 0이 아닌 경우에만 정규화
                if std_val > 1e-8:
                    normalized_tensor[:, :, feature_idx] = (
                        feature_data - mean_val
                    ) / std_val
                else:
                    normalized_tensor[:, :, feature_idx] = feature_data - mean_val

            return normalized_tensor

        except Exception as e:
            self.logger.warning(f"특성 정규화 중 오류: {e}")
            return sequence_tensor

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        기본 분석 구현 (하위 클래스에서 오버라이드)

        Args:
            historical_data: 과거 당첨 번호 데이터
            *args, **kwargs: 추가 매개변수

        Returns:
            Dict[str, Any]: 분석 결과
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")
