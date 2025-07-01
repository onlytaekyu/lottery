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

# logging 제거 - unified_logging 사용
import time
from datetime import datetime

from ..utils.unified_logging import get_logger
from ..shared.types import LotteryNumber
from ..utils.unified_config import ConfigProxy
from .base_trend_analyzer import BaseTrendAnalyzer

# 로거 설정
logger = get_logger(__name__)


class TrendAnalyzer(BaseTrendAnalyzer):
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

        # 베이스 클래스의 공통 메서드 사용
        trend_features = self._extract_trend_features(draw_numbers)

        # 결과 반환
        return {
            "trend_features": trend_features,
            "data_count": len(historical_data),
            "window_size": window_size,
            "timestamp": int(time.time()),
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
