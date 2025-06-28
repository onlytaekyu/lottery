"""
중복 패턴 ROI 분석 모듈

이 모듈은 3자리 및 4자리 중복 패턴과 ROI 성능 간의 상관관계를 분석합니다.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import Counter

from .base_analyzer import BaseAnalyzer
from ..shared.types import LotteryNumber
from ..utils import get_logger

# 최적화된 로거 자동 할당
logger = get_logger(__name__)


class OverlapROIAnalyzer(BaseAnalyzer):
    """중복 패턴 ROI 분석기 클래스"""

    def __init__(self, config: dict):
        """
        OverlapROIAnalyzer 초기화

        Args:
            config: 분석에 사용할 설정
        """
        super().__init__(config, analyzer_type="overlap_roi")

    def analyze_overlap_pattern_roi(
        self, historical_data: List[LotteryNumber], overlap_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        중복 패턴과 ROI 성능 간의 상관관계 분석

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            overlap_patterns: 중복 패턴 분석 결과

        Returns:
            Dict[str, Any]: 중복 패턴 ROI 분석 결과
        """
        result = {}

        try:
            self.logger.info("중복 패턴 ROI 분석 시작...")

            # 중복 패턴 데이터 추출
            overlap_3_4_data = overlap_patterns.get("overlap_3_4_digit_patterns", {})
            overlap_3_patterns = overlap_3_4_data.get("overlap_3_patterns", {})
            overlap_4_patterns = overlap_3_4_data.get("overlap_4_patterns", {})

            # 3자리 중복 패턴 ROI 효과 분석
            result["overlap_3_roi_effect"] = self._analyze_pattern_roi_effect(
                historical_data, overlap_3_patterns, pattern_type="3_digit"
            )

            # 4자리 중복 패턴 ROI 효과 분석
            result["overlap_4_roi_effect"] = self._analyze_pattern_roi_effect(
                historical_data, overlap_4_patterns, pattern_type="4_digit"
            )

            # 중복 패턴 기반 ROI 예측 모델
            result["roi_prediction_model"] = self._build_overlap_roi_prediction_model(
                historical_data, overlap_3_4_data
            )

            # 중복 패턴 ROI 성능 요약
            result["performance_summary"] = {
                "avg_3_pattern_roi": result["overlap_3_roi_effect"].get("avg_roi", 0.0),
                "avg_4_pattern_roi": result["overlap_4_roi_effect"].get("avg_roi", 0.0),
                "pattern_correlation": self._calculate_pattern_correlation(
                    result["overlap_3_roi_effect"], result["overlap_4_roi_effect"]
                ),
                "recommendation": self._generate_roi_recommendation(
                    result["overlap_3_roi_effect"], result["overlap_4_roi_effect"]
                ),
            }

            self.logger.info("중복 패턴 ROI 분석 완료")
            return result

        except Exception as e:
            self.logger.error(f"중복 패턴 ROI 분석 중 오류 발생: {e}")
            return {
                "overlap_3_roi_effect": {"avg_roi": 0.0, "sample_count": 0},
                "overlap_4_roi_effect": {"avg_roi": 0.0, "sample_count": 0},
                "roi_prediction_model": {"accuracy": 0.0, "confidence": 0.0},
                "performance_summary": {
                    "avg_3_pattern_roi": 0.0,
                    "avg_4_pattern_roi": 0.0,
                    "pattern_correlation": 0.0,
                    "recommendation": "insufficient_data",
                },
            }

    def _analyze_pattern_roi_effect(
        self,
        historical_data: List[LotteryNumber],
        patterns: Dict[str, Any],
        pattern_type: str,
    ) -> Dict[str, Any]:
        """
        특정 중복 패턴의 ROI 효과 분석

        Args:
            historical_data: 과거 당첨 번호 목록
            patterns: 패턴 데이터
            pattern_type: 패턴 유형 ("3_digit" 또는 "4_digit")

        Returns:
            Dict[str, Any]: 패턴 ROI 효과 분석 결과
        """
        roi_scores = []
        pattern_occurrences = []

        # 패턴별 ROI 계산
        most_frequent = patterns.get("most_frequent", {})

        for pattern, frequency in most_frequent.items():
            if frequency < 2:  # 최소 2회 이상 출현한 패턴만 분석
                continue

            # 해당 패턴이 포함된 회차들의 ROI 계산
            pattern_roi = self._calculate_pattern_specific_roi(
                historical_data, pattern, pattern_type
            )

            if pattern_roi is not None:
                roi_scores.append(pattern_roi)
                pattern_occurrences.append(frequency)

        # 통계 계산
        if roi_scores:
            avg_roi = np.mean(roi_scores)
            std_roi = np.std(roi_scores)
            positive_ratio = sum(1 for roi in roi_scores if roi > 0) / len(roi_scores)

            # 가중 평균 ROI (빈도로 가중)
            weighted_roi = np.average(roi_scores, weights=pattern_occurrences)
        else:
            avg_roi = std_roi = positive_ratio = weighted_roi = 0.0

        return {
            "avg_roi": float(avg_roi),
            "std_roi": float(std_roi),
            "weighted_roi": float(weighted_roi),
            "positive_ratio": float(positive_ratio),
            "sample_count": len(roi_scores),
            "pattern_type": pattern_type,
        }

    def _calculate_pattern_specific_roi(
        self, historical_data: List[LotteryNumber], pattern: Any, pattern_type: str
    ) -> Optional[float]:
        """
        특정 패턴의 ROI 계산

        Args:
            historical_data: 과거 당첨 번호 목록
            pattern: 분석할 패턴
            pattern_type: 패턴 유형

        Returns:
            Optional[float]: 패턴의 ROI (계산 불가능한 경우 None)
        """
        try:
            # 패턴을 번호 리스트로 변환
            if isinstance(pattern, (tuple, list)):
                pattern_numbers = list(pattern)
            else:
                return None

            # 해당 패턴이 포함된 회차들 찾기
            matching_draws = []
            for draw in historical_data:
                if self._pattern_matches_draw(
                    pattern_numbers, draw.numbers, pattern_type
                ):
                    matching_draws.append(draw)

            if len(matching_draws) < 2:
                return None

            # 매칭된 회차들의 평균 ROI 계산
            total_roi = 0.0
            for draw in matching_draws:
                # 간단한 ROI 계산 (실제로는 더 복잡한 계산이 필요)
                roi = self._calculate_simple_roi(draw.numbers)
                total_roi += roi

            return total_roi / len(matching_draws)

        except Exception as e:
            self.logger.warning(f"패턴별 ROI 계산 중 오류: {e}")
            return None

    def _pattern_matches_draw(
        self, pattern_numbers: List[int], draw_numbers: List[int], pattern_type: str
    ) -> bool:
        """
        패턴이 특정 회차와 매칭되는지 확인

        Args:
            pattern_numbers: 패턴 번호 리스트
            draw_numbers: 회차 번호 리스트
            pattern_type: 패턴 유형

        Returns:
            bool: 매칭 여부
        """
        overlap = set(pattern_numbers) & set(draw_numbers)
        required_overlap = 3 if pattern_type == "3_digit" else 4
        return len(overlap) >= required_overlap

    def _calculate_simple_roi(self, numbers: List[int]) -> float:
        """
        간단한 ROI 계산

        Args:
            numbers: 번호 리스트

        Returns:
            float: 간단한 ROI 점수
        """
        # 번호 특성 기반 간단한 ROI 점수 계산
        # 실제로는 더 복잡한 ROI 계산 로직이 필요

        # 홀짝 균형
        odd_count = sum(1 for num in numbers if num % 2 == 1)
        balance_score = 1.0 - abs(odd_count - 3) / 3.0

        # 번호 범위
        range_score = (max(numbers) - min(numbers)) / 45.0

        # 연속 번호
        consecutive_score = 0.0
        for i in range(1, len(numbers)):
            if numbers[i] == numbers[i - 1] + 1:
                consecutive_score += 0.1

        # 종합 점수
        roi_score = (
            balance_score * 0.4 + range_score * 0.4 + consecutive_score * 0.2
        ) - 0.5

        return roi_score

    def _build_overlap_roi_prediction_model(
        self, historical_data: List[LotteryNumber], overlap_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        중복 패턴 기반 ROI 예측 모델 구축

        Args:
            historical_data: 과거 당첨 번호 목록
            overlap_data: 중복 패턴 데이터

        Returns:
            Dict[str, Any]: 예측 모델 정보
        """
        try:
            # 간단한 예측 모델 (실제로는 더 복잡한 ML 모델 필요)
            model_accuracy = 0.0
            model_confidence = 0.0

            # 3자리 패턴 기반 예측 정확도
            overlap_3_patterns = overlap_data.get("overlap_3_patterns", {})
            if overlap_3_patterns.get("total_patterns", 0) > 0:
                model_accuracy += 0.3
                model_confidence += 0.2

            # 4자리 패턴 기반 예측 정확도
            overlap_4_patterns = overlap_data.get("overlap_4_patterns", {})
            if overlap_4_patterns.get("total_patterns", 0) > 0:
                model_accuracy += 0.4
                model_confidence += 0.3

            # 시간 간격 일관성
            time_gap_data = overlap_data.get("overlap_time_gap_analysis", {})
            gap_consistency = (
                time_gap_data.get("gap_consistency_3", 0.0)
                + time_gap_data.get("gap_consistency_4", 0.0)
            ) / 2.0

            model_confidence += gap_consistency * 0.5

            return {
                "accuracy": min(1.0, model_accuracy),
                "confidence": min(1.0, model_confidence),
                "features_used": ["3_digit_patterns", "4_digit_patterns", "time_gaps"],
                "model_type": "simple_statistical",
            }

        except Exception as e:
            self.logger.warning(f"ROI 예측 모델 구축 중 오류: {e}")
            return {
                "accuracy": 0.0,
                "confidence": 0.0,
                "features_used": [],
                "model_type": "none",
            }

    def _calculate_pattern_correlation(
        self, pattern_3_effect: Dict[str, Any], pattern_4_effect: Dict[str, Any]
    ) -> float:
        """
        3자리와 4자리 패턴 간의 상관관계 계산

        Args:
            pattern_3_effect: 3자리 패턴 효과
            pattern_4_effect: 4자리 패턴 효과

        Returns:
            float: 상관관계 (-1 ~ 1)
        """
        try:
            roi_3 = pattern_3_effect.get("avg_roi", 0.0)
            roi_4 = pattern_4_effect.get("avg_roi", 0.0)

            # 간단한 상관관계 계산
            if roi_3 == 0.0 and roi_4 == 0.0:
                return 0.0

            # 두 패턴의 ROI 방향성 비교
            if (roi_3 > 0 and roi_4 > 0) or (roi_3 < 0 and roi_4 < 0):
                correlation = min(1.0, abs(roi_3 + roi_4) / 2.0)
            else:
                correlation = -min(1.0, abs(roi_3 - roi_4) / 2.0)

            return correlation

        except Exception as e:
            self.logger.warning(f"패턴 상관관계 계산 중 오류: {e}")
            return 0.0

    def _generate_roi_recommendation(
        self, pattern_3_effect: Dict[str, Any], pattern_4_effect: Dict[str, Any]
    ) -> str:
        """
        ROI 분석 결과 기반 추천 생성

        Args:
            pattern_3_effect: 3자리 패턴 효과
            pattern_4_effect: 4자리 패턴 효과

        Returns:
            str: 추천 사항
        """
        roi_3 = pattern_3_effect.get("avg_roi", 0.0)
        roi_4 = pattern_4_effect.get("avg_roi", 0.0)
        sample_3 = pattern_3_effect.get("sample_count", 0)
        sample_4 = pattern_4_effect.get("sample_count", 0)

        if sample_3 < 5 and sample_4 < 5:
            return "insufficient_data"
        elif roi_3 > 0.1 and roi_4 > 0.1:
            return "use_both_patterns"
        elif roi_3 > roi_4 and roi_3 > 0.05:
            return "prefer_3_digit_patterns"
        elif roi_4 > roi_3 and roi_4 > 0.05:
            return "prefer_4_digit_patterns"
        elif roi_3 < -0.1 and roi_4 < -0.1:
            return "avoid_overlap_patterns"
        else:
            return "neutral_effect"

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        BaseAnalyzer의 추상 메서드 구현

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 매개변수

        Returns:
            Dict[str, Any]: 분석 결과
        """
        # 기본 분석 수행
        if args and isinstance(args[0], dict):
            overlap_patterns = args[0]
            return self.analyze_overlap_pattern_roi(historical_data, overlap_patterns)
        else:
            self.logger.warning("중복 패턴 데이터가 제공되지 않았습니다.")
            return {
                "overlap_3_roi_effect": {"avg_roi": 0.0, "sample_count": 0},
                "overlap_4_roi_effect": {"avg_roi": 0.0, "sample_count": 0},
                "roi_prediction_model": {"accuracy": 0.0, "confidence": 0.0},
                "performance_summary": {
                    "avg_3_pattern_roi": 0.0,
                    "avg_4_pattern_roi": 0.0,
                    "pattern_correlation": 0.0,
                    "recommendation": "no_data_provided",
                },
            }
