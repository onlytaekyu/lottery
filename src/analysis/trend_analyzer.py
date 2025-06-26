"""
트렌드/시계열 분석기 모듈

이 모듈은 로또 번호의 시간에 따른 패턴 변화와 추세를 분석하는 기능을 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Tuple, Optional, Union
from collections import Counter, defaultdict
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
import json
import logging
import time
from datetime import datetime

from ..utils.error_handler import get_logger
from ..shared.types import LotteryNumber
from ..utils.config_loader import ConfigProxy
from .base_analyzer import BaseAnalyzer

# 로거 설정
logger = get_logger(__name__)


class TrendAnalyzer(BaseAnalyzer[Dict[str, Any]]):
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
        super().__init__(config, "trend")

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

        # 1. 위치별 추세 분석
        position_trends = self._analyze_position_trends(draw_numbers)

        # 2. 증감 패턴 분석
        delta_patterns = self._analyze_delta_patterns(draw_numbers)

        # 3. 세그먼트 반복 분석
        segment_patterns = self._analyze_segment_repeat_patterns(draw_numbers)

        # 분석 결과 통합
        trend_features = {
            **position_trends,
            **delta_patterns,
            **segment_patterns,
        }

        # 결과 반환
        return {
            "trend_features": trend_features,
            "data_count": len(historical_data),
            "window_size": window_size,
            "timestamp": int(time.time()),
        }

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
