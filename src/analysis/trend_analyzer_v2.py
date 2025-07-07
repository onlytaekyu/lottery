#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
회차별 트렌드 분석기 V2

기존 TrendAnalyzer와 완전히 독립적으로 동작하는 새로운 트렌드 분석기입니다.
회차 번호(draw_no) 기준으로만 트렌드를 분석하며, 계절성 분석은 제외합니다.

주요 기능:
- 회차별 번호 출현 트렌드 분석
- 트렌드 변화점 탐지
- 트렌드 강도 측정
- 가중 트렌드 분석 (최근 회차 가중)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import json
from pathlib import Path

from .base_analyzer import BaseAnalyzer
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


class TrendAnalyzerV2(BaseAnalyzer):
    """
    회차별 트렌드 분석기 V2

    기존 TrendAnalyzer와 독립적으로 동작하며, 회차 기준 트렌드 분석에 특화됩니다.
    계절성 제거, 순수 트렌드만 분석합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        super().__init__(config, "TrendAnalyzerV2")

        # 트렌드 분석 설정
        self.trend_config = {
            "min_trend_length": 5,  # 최소 트렌드 길이
            "trend_threshold": 0.6,  # 트렌드 임계값
            "weight_decay": 0.95,  # 가중치 감쇠율
            "change_point_sensitivity": 0.3,  # 변화점 민감도
            "trend_strength_window": 10,  # 트렌드 강도 측정 윈도우
        }

        # 설정 오버라이드
        if config and "trend_analyzer_v2" in config:
            self.trend_config.update(config["trend_analyzer_v2"])

        self.logger.info("TrendAnalyzerV2 초기화 완료")

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        회차별 트렌드 분석 메인 구현

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 트렌드 분석 결과
        """
        try:
            self.logger.info(
                f"TrendAnalyzerV2 분석 시작: {len(historical_data)}개 회차"
            )

            # 1. 회차별 번호 출현 트렌드 분석
            draw_trends = self.analyze_draw_number_trends(historical_data)

            # 2. 트렌드 변화점 탐지
            change_points = self.detect_trend_change_points(historical_data)

            # 3. 트렌드 강도 측정
            trend_strength = self.calculate_trend_strength(historical_data)

            # 4. 가중 트렌드 분석
            weighted_trends = self.apply_weighted_trend_analysis(historical_data)

            # 5. 종합 트렌드 점수 계산
            comprehensive_score = self._calculate_comprehensive_trend_score(
                draw_trends, change_points, trend_strength, weighted_trends
            )

            # 결과 통합
            analysis_result = {
                "analyzer_version": "TrendAnalyzerV2_v1.0",
                "analysis_timestamp": datetime.now().isoformat(),
                "data_range": {
                    "total_draws": len(historical_data),
                    "start_draw": historical_data[0].draw_no if historical_data else 0,
                    "end_draw": historical_data[-1].draw_no if historical_data else 0,
                },
                "draw_number_trends": draw_trends,
                "trend_change_points": change_points,
                "trend_strength_metrics": trend_strength,
                "weighted_trend_analysis": weighted_trends,
                "comprehensive_trend_score": comprehensive_score,
                "trend_summary": self._generate_trend_summary(
                    draw_trends, change_points, trend_strength, weighted_trends
                ),
            }

            self.logger.info("TrendAnalyzerV2 분석 완료")
            return analysis_result

        except Exception as e:
            self.logger.error(f"TrendAnalyzerV2 분석 중 오류 발생: {e}")
            raise

    def analyze_draw_number_trends(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        회차별 번호 출현 트렌드 분석

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 회차별 번호 트렌드 분석 결과
        """
        try:
            # 회차별 번호 출현 매트릭스 생성
            draw_matrix = self._create_draw_matrix(data)

            # 각 번호별 트렌드 분석
            number_trends = {}
            for number in range(1, 46):  # 로또 번호 1-45
                trend_data = self._analyze_single_number_trend(draw_matrix, number)
                number_trends[str(number)] = trend_data

            # 전체 트렌드 패턴 분석
            overall_patterns = self._analyze_overall_trend_patterns(draw_matrix)

            return {
                "individual_number_trends": number_trends,
                "overall_trend_patterns": overall_patterns,
                "trend_analysis_config": self.trend_config,
            }

        except Exception as e:
            self.logger.error(f"회차별 번호 트렌드 분석 실패: {e}")
            return {}

    def detect_trend_change_points(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        트렌드 변화점 탐지

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 트렌드 변화점 분석 결과
        """
        try:
            # 회차별 번호 출현 매트릭스 생성
            draw_matrix = self._create_draw_matrix(data)

            # 각 번호별 변화점 탐지
            change_points = {}
            for number in range(1, 46):
                change_points[str(number)] = self._detect_single_number_change_points(
                    draw_matrix, number
                )

            # 전체 시스템 변화점 탐지
            system_change_points = self._detect_system_change_points(draw_matrix)

            return {
                "individual_change_points": change_points,
                "system_change_points": system_change_points,
                "change_point_summary": self._summarize_change_points(
                    change_points, system_change_points
                ),
            }

        except Exception as e:
            self.logger.error(f"트렌드 변화점 탐지 실패: {e}")
            return {}

    def calculate_trend_strength(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        트렌드 강도 측정

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 트렌드 강도 분석 결과
        """
        try:
            # 회차별 번호 출현 매트릭스 생성
            draw_matrix = self._create_draw_matrix(data)

            # 각 번호별 트렌드 강도 계산
            strength_metrics = {}
            for number in range(1, 46):
                strength_metrics[str(number)] = (
                    self._calculate_single_number_trend_strength(draw_matrix, number)
                )

            # 전체 시스템 트렌드 강도
            system_strength = self._calculate_system_trend_strength(draw_matrix)

            return {
                "individual_trend_strength": strength_metrics,
                "system_trend_strength": system_strength,
                "strength_distribution": self._analyze_strength_distribution(
                    strength_metrics
                ),
            }

        except Exception as e:
            self.logger.error(f"트렌드 강도 측정 실패: {e}")
            return {}

    def apply_weighted_trend_analysis(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        가중 트렌드 분석 (최근 회차 가중)

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 가중 트렌드 분석 결과
        """
        try:
            # 회차별 가중치 계산
            weights = self._calculate_draw_weights(len(data))

            # 가중 출현 빈도 계산
            weighted_frequencies = self._calculate_weighted_frequencies(data, weights)

            # 가중 트렌드 방향 분석
            weighted_trends = self._analyze_weighted_trends(data, weights)

            # 가중 예측 점수 계산
            prediction_scores = self._calculate_weighted_prediction_scores(
                weighted_frequencies, weighted_trends
            )

            return {
                "weighted_frequencies": weighted_frequencies,
                "weighted_trend_directions": weighted_trends,
                "prediction_scores": prediction_scores,
                "weight_configuration": {
                    "decay_rate": self.trend_config["weight_decay"],
                    "total_data_points": len(data),
                },
            }

        except Exception as e:
            self.logger.error(f"가중 트렌드 분석 실패: {e}")
            return {}

    def _create_draw_matrix(self, data: List[LotteryNumber]) -> np.ndarray:
        """회차별 번호 출현 매트릭스 생성"""
        matrix = np.zeros((len(data), 46))  # 0번 인덱스는 사용하지 않음

        for i, draw in enumerate(data):
            for number in draw.numbers:
                if 1 <= number <= 45:
                    matrix[i, number] = 1

        return matrix

    def _analyze_single_number_trend(
        self, matrix: np.ndarray, number: int
    ) -> Dict[str, Any]:
        """단일 번호의 트렌드 분석"""
        appearances = matrix[:, number]

        # 출현 간격 분석
        appearance_indices = np.where(appearances == 1)[0]
        if len(appearance_indices) > 1:
            intervals = np.diff(appearance_indices)
            avg_interval = np.mean(intervals)
            interval_trend = np.polyfit(range(len(intervals)), intervals, 1)[0]
        else:
            avg_interval = len(matrix)
            interval_trend = 0

        # 최근 트렌드 분석
        recent_window = min(20, len(appearances))
        recent_trend = np.sum(appearances[-recent_window:]) / recent_window

        # 장기 트렌드 분석
        long_term_trend = np.sum(appearances) / len(appearances)

        return {
            "total_appearances": int(np.sum(appearances)),
            "appearance_rate": float(long_term_trend),
            "recent_trend": float(recent_trend),
            "average_interval": float(avg_interval),
            "interval_trend": float(interval_trend),
            "trend_direction": (
                "increasing" if recent_trend > long_term_trend else "decreasing"
            ),
        }

    def _analyze_overall_trend_patterns(self, matrix: np.ndarray) -> Dict[str, Any]:
        """전체 트렌드 패턴 분석"""
        # 회차별 총 출현 패턴 (항상 6개이므로 다른 메트릭 사용)
        # 번호 범위별 출현 패턴 분석
        low_numbers = np.sum(matrix[:, 1:16], axis=1)  # 1-15
        mid_numbers = np.sum(matrix[:, 16:31], axis=1)  # 16-30
        high_numbers = np.sum(matrix[:, 31:46], axis=1)  # 31-45

        return {
            "range_distribution_trends": {
                "low_range_trend": float(
                    np.polyfit(range(len(low_numbers)), low_numbers, 1)[0]
                ),
                "mid_range_trend": float(
                    np.polyfit(range(len(mid_numbers)), mid_numbers, 1)[0]
                ),
                "high_range_trend": float(
                    np.polyfit(range(len(high_numbers)), high_numbers, 1)[0]
                ),
            },
            "pattern_stability": {
                "low_range_stability": float(np.std(low_numbers)),
                "mid_range_stability": float(np.std(mid_numbers)),
                "high_range_stability": float(np.std(high_numbers)),
            },
        }

    def _detect_single_number_change_points(
        self, matrix: np.ndarray, number: int
    ) -> List[Dict[str, Any]]:
        """단일 번호의 변화점 탐지"""
        appearances = matrix[:, number]
        change_points = []

        # 슬라이딩 윈도우를 사용한 변화점 탐지
        window_size = self.trend_config["trend_strength_window"]
        threshold = self.trend_config["change_point_sensitivity"]

        for i in range(window_size, len(appearances) - window_size):
            before_window = appearances[i - window_size : i]
            after_window = appearances[i : i + window_size]

            before_rate = np.mean(before_window)
            after_rate = np.mean(after_window)

            if abs(before_rate - after_rate) > threshold:
                change_points.append(
                    {
                        "draw_index": i,
                        "before_rate": float(before_rate),
                        "after_rate": float(after_rate),
                        "change_magnitude": float(abs(before_rate - after_rate)),
                        "change_direction": (
                            "increase" if after_rate > before_rate else "decrease"
                        ),
                    }
                )

        return change_points

    def _detect_system_change_points(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """시스템 전체 변화점 탐지"""
        # 전체 시스템의 패턴 변화 탐지
        system_changes = []

        # 번호 범위별 분포 변화 탐지
        low_numbers = np.sum(matrix[:, 1:16], axis=1)
        mid_numbers = np.sum(matrix[:, 16:31], axis=1)
        high_numbers = np.sum(matrix[:, 31:46], axis=1)

        window_size = self.trend_config["trend_strength_window"]

        for i in range(window_size, len(matrix) - window_size):
            # 각 범위의 변화 분석
            ranges = [low_numbers, mid_numbers, high_numbers]
            range_names = ["low", "mid", "high"]

            for range_data, range_name in zip(ranges, range_names):
                before_avg = np.mean(range_data[i - window_size : i])
                after_avg = np.mean(range_data[i : i + window_size])

                if abs(before_avg - after_avg) > 0.5:  # 시스템 변화 임계값
                    system_changes.append(
                        {
                            "draw_index": i,
                            "range_affected": range_name,
                            "before_average": float(before_avg),
                            "after_average": float(after_avg),
                            "change_magnitude": float(abs(before_avg - after_avg)),
                        }
                    )

        return system_changes

    def _summarize_change_points(
        self, individual_changes: Dict[str, List], system_changes: List
    ) -> Dict[str, Any]:
        """변화점 요약"""
        total_individual_changes = sum(
            len(changes) for changes in individual_changes.values()
        )

        return {
            "total_individual_change_points": total_individual_changes,
            "total_system_change_points": len(system_changes),
            "most_volatile_numbers": sorted(
                [(num, len(changes)) for num, changes in individual_changes.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "change_point_density": (
                total_individual_changes / len(individual_changes)
                if individual_changes
                else 0
            ),
        }

    def _calculate_single_number_trend_strength(
        self, matrix: np.ndarray, number: int
    ) -> Dict[str, float]:
        """단일 번호의 트렌드 강도 계산"""
        appearances = matrix[:, number]

        # 트렌드 강도 메트릭들
        # 1. 선형 트렌드 강도
        x = np.arange(len(appearances))
        linear_trend = np.polyfit(x, appearances, 1)[0]

        # 2. 최근 vs 장기 트렌드 비교
        recent_window = min(20, len(appearances))
        recent_rate = np.mean(appearances[-recent_window:])
        long_term_rate = np.mean(appearances)
        trend_acceleration = recent_rate - long_term_rate

        # 3. 트렌드 일관성 (변동성의 역수)
        trend_consistency = 1.0 / (np.std(appearances) + 1e-6)

        return {
            "linear_trend_strength": float(abs(linear_trend)),
            "trend_acceleration": float(trend_acceleration),
            "trend_consistency": float(trend_consistency),
            "composite_strength": float(
                abs(linear_trend) + abs(trend_acceleration) + trend_consistency
            ),
        }

    def _calculate_system_trend_strength(self, matrix: np.ndarray) -> Dict[str, float]:
        """시스템 전체 트렌드 강도 계산"""
        # 번호 범위별 트렌드 강도
        low_numbers = np.sum(matrix[:, 1:16], axis=1)
        mid_numbers = np.sum(matrix[:, 16:31], axis=1)
        high_numbers = np.sum(matrix[:, 31:46], axis=1)

        x = np.arange(len(matrix))

        return {
            "low_range_trend_strength": float(abs(np.polyfit(x, low_numbers, 1)[0])),
            "mid_range_trend_strength": float(abs(np.polyfit(x, mid_numbers, 1)[0])),
            "high_range_trend_strength": float(abs(np.polyfit(x, high_numbers, 1)[0])),
            "overall_system_stability": float(
                1.0
                / (
                    np.std(low_numbers)
                    + np.std(mid_numbers)
                    + np.std(high_numbers)
                    + 1e-6
                )
            ),
        }

    def _analyze_strength_distribution(
        self, strength_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """트렌드 강도 분포 분석"""
        composite_strengths = [
            metrics["composite_strength"] for metrics in strength_metrics.values()
        ]

        return {
            "mean_strength": float(np.mean(composite_strengths)),
            "median_strength": float(np.median(composite_strengths)),
            "std_strength": float(np.std(composite_strengths)),
            "strongest_numbers": sorted(
                [
                    (num, metrics["composite_strength"])
                    for num, metrics in strength_metrics.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "weakest_numbers": sorted(
                [
                    (num, metrics["composite_strength"])
                    for num, metrics in strength_metrics.items()
                ],
                key=lambda x: x[1],
            )[:10],
        }

    def _calculate_draw_weights(self, total_draws: int) -> np.ndarray:
        """회차별 가중치 계산 (최근 회차일수록 높은 가중치)"""
        weights = np.array(
            [
                self.trend_config["weight_decay"] ** (total_draws - i - 1)
                for i in range(total_draws)
            ]
        )
        return weights / np.sum(weights)  # 정규화

    def _calculate_weighted_frequencies(
        self, data: List[LotteryNumber], weights: np.ndarray
    ) -> Dict[str, float]:
        """가중 출현 빈도 계산"""
        weighted_freq = defaultdict(float)

        for i, draw in enumerate(data):
            for number in draw.numbers:
                weighted_freq[str(number)] += weights[i]

        return dict(weighted_freq)

    def _analyze_weighted_trends(
        self, data: List[LotteryNumber], weights: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """가중 트렌드 방향 분석"""
        # 시간 윈도우별 가중 출현율 계산
        window_size = 10
        trends = {}

        for number in range(1, 46):
            number_str = str(number)
            window_rates = []

            for i in range(window_size, len(data)):
                window_data = data[i - window_size : i]
                window_weights = weights[i - window_size : i]

                appearances = sum(
                    weights[j]
                    for j, draw in enumerate(window_data)
                    if number in draw.numbers
                )
                total_weight = sum(window_weights)

                window_rates.append(
                    appearances / total_weight if total_weight > 0 else 0
                )

            if len(window_rates) > 1:
                trend_slope = np.polyfit(range(len(window_rates)), window_rates, 1)[0]
                trends[number_str] = {
                    "trend_slope": float(trend_slope),
                    "recent_rate": float(window_rates[-1]) if window_rates else 0.0,
                    "trend_direction": (
                        "increasing" if trend_slope > 0 else "decreasing"
                    ),
                }
            else:
                trends[number_str] = {
                    "trend_slope": 0.0,
                    "recent_rate": 0.0,
                    "trend_direction": "stable",
                }

        return trends

    def _calculate_weighted_prediction_scores(
        self, frequencies: Dict[str, float], trends: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """가중 예측 점수 계산"""
        scores = {}

        for number_str in frequencies:
            frequency_score = frequencies[number_str]
            trend_score = (
                trends[number_str]["trend_slope"] if number_str in trends else 0
            )

            # 복합 점수 계산 (빈도 + 트렌드)
            composite_score = frequency_score + (trend_score * 0.3)  # 트렌드 가중치 30%
            scores[number_str] = float(composite_score)

        return scores

    def _calculate_comprehensive_trend_score(
        self,
        draw_trends: Dict,
        change_points: Dict,
        trend_strength: Dict,
        weighted_trends: Dict,
    ) -> Dict[str, float]:
        """종합 트렌드 점수 계산"""
        comprehensive_scores = {}

        for number in range(1, 46):
            number_str = str(number)

            # 각 분석 결과에서 점수 추출
            trend_score = 0.0
            strength_score = 0.0
            change_score = 0.0
            weighted_score = 0.0

            # 트렌드 분석 점수
            if number_str in draw_trends.get("individual_number_trends", {}):
                trend_data = draw_trends["individual_number_trends"][number_str]
                trend_score = trend_data.get("recent_trend", 0) * 0.4

            # 강도 분석 점수
            if number_str in trend_strength.get("individual_trend_strength", {}):
                strength_data = trend_strength["individual_trend_strength"][number_str]
                strength_score = strength_data.get("composite_strength", 0) * 0.3

            # 변화점 분석 점수 (변화점이 적을수록 안정적)
            if number_str in change_points.get("individual_change_points", {}):
                change_count = len(
                    change_points["individual_change_points"][number_str]
                )
                change_score = max(0, 1.0 - change_count * 0.1) * 0.2

            # 가중 트렌드 점수
            if number_str in weighted_trends.get("prediction_scores", {}):
                weighted_score = weighted_trends["prediction_scores"][number_str] * 0.1

            # 종합 점수
            comprehensive_scores[number_str] = (
                trend_score + strength_score + change_score + weighted_score
            )

        return comprehensive_scores

    def _generate_trend_summary(
        self,
        draw_trends: Dict,
        change_points: Dict,
        trend_strength: Dict,
        weighted_trends: Dict,
    ) -> Dict[str, Any]:
        """트렌드 분석 요약 생성"""
        # 상위 추천 번호 (종합 점수 기준)
        comprehensive_scores = self._calculate_comprehensive_trend_score(
            draw_trends, change_points, trend_strength, weighted_trends
        )

        top_numbers = sorted(
            comprehensive_scores.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "top_recommended_numbers": [
                {"number": int(num), "score": score} for num, score in top_numbers
            ],
            "trend_analysis_summary": {
                "total_numbers_analyzed": 45,
                "analysis_methods": [
                    "draw_trends",
                    "change_points",
                    "trend_strength",
                    "weighted_trends",
                ],
                "recommendation_confidence": (
                    "high" if top_numbers[0][1] > 0.5 else "medium"
                ),
            },
            "system_health": {
                "overall_stability": trend_strength.get(
                    "system_trend_strength", {}
                ).get("overall_system_stability", 0),
                "change_point_density": change_points.get(
                    "change_point_summary", {}
                ).get("change_point_density", 0),
            },
        }

    def save_analysis_results(
        self, analysis_results: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """분석 결과를 파일로 저장"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trend_analyzer_v2_results_{timestamp}.json"

            output_dir = Path("data/result/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / filename

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"TrendAnalyzerV2 분석 결과 저장: {output_file}")
            return str(output_file)

        except Exception as e:
            self.logger.error(f"분석 결과 저장 실패: {e}")
            raise


# 팩토리 함수
def create_trend_analyzer_v2(
    config: Optional[Dict[str, Any]] = None,
) -> TrendAnalyzerV2:
    """TrendAnalyzerV2 인스턴스 생성"""
    return TrendAnalyzerV2(config)


# 편의 함수
def analyze_trends_v2(
    historical_data: List[LotteryNumber], config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """회차별 트렌드 분석 수행"""
    analyzer = create_trend_analyzer_v2(config)
    return analyzer.analyze(historical_data)
