"""
로또 번호 패턴 분석기 모듈

이 모듈은 로또 번호의 패턴을 분석하는 기능을 제공합니다.
"""

import sys
import json
import datetime
import random
import networkx as nx
import numpy as np
import hashlib
import os
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple, TypedDict, Optional, Union, cast
from collections import Counter, defaultdict
import time
import logging
from pathlib import Path
import math

from ..utils.error_handler import get_logger
from ..shared.types import LotteryNumber, PatternAnalysis
from ..utils.memory_manager import MemoryManager
from ..analysis.base_analyzer import BaseAnalyzer
from ..utils.config_loader import ConfigProxy
from ..utils.report_writer import safe_convert
from ..utils.performance_report_writer import save_analysis_performance_report
from ..shared.graph_utils import calculate_pair_frequency, calculate_pair_centrality


# 로그 설정
logger = get_logger(__name__)


# 패턴 특성을 위한 TypedDict 정의
class PatternFeatures(TypedDict, total=False):
    max_consecutive_length: int
    total_sum: int
    odd_count: int
    even_count: int
    gap_avg: float
    gap_std: float
    range_counts: list[int]
    cluster_overlap_ratio: float
    frequent_pair_score: float
    roi_weight: float
    consecutive_score: float
    trend_score_avg: float
    trend_score_max: float
    trend_score_min: float
    risk_score: float


class PatternAnalyzer(BaseAnalyzer[PatternAnalysis]):
    """로또 번호의 패턴을 분석하는 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        PatternAnalyzer 초기화

        Args:
            config: 패턴 분석에 사용할 설정
        """
        super().__init__(config or {}, "pattern")
        self.memory_manager = MemoryManager()
        self.pattern_stats = {}
        self.scoped_analyses = {}  # 스코프별 분석 결과 저장
        self.logger = get_logger(__name__)  # 로거 명시적 초기화

        # ConfigProxy로 변환
        if not isinstance(self.config, ConfigProxy):
            self.config = ConfigProxy(self.config)

    def load_data(self, limit: Optional[int] = None) -> List[LotteryNumber]:
        """
        로또 당첨 번호 데이터 로드

        Args:
            limit: 로드할 최대 데이터 수 (None이면 전체 로드)

        Returns:
            로또 당첨 번호 데이터 리스트
        """
        # 순환 참조 방지를 위한 지연 임포트
        from ..utils.data_loader import load_draw_history

        data = load_draw_history()
        if limit is not None and limit > 0:
            data = data[-limit:]

        return data

    def analyze_patterns(self, historical_data: List[LotteryNumber]) -> PatternAnalysis:
        """
        과거 로또 당첨 번호의 패턴을 분석합니다 (analyze 메서드의 별칭).

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            PatternAnalysis: 패턴 분석 결과
        """
        return self.analyze(historical_data)

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> PatternAnalysis:
        """
        과거 로또 당첨 번호의 패턴을 분석하는 내부 구현입니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 매개변수

        Returns:
            PatternAnalysis: 패턴 분석 결과
        """
        self.logger.info(f"패턴 분석 시작: {len(historical_data)}개 데이터")

        result = PatternAnalysis()
        result.metadata = {}

        # 연속 번호 길이 분포 분석
        consecutive_length_distribution = self.analyze_consecutive_length_distribution(
            historical_data
        )

        # 홀짝 분포 분석
        odd_even_distribution = self.analyze_odd_even_distribution(historical_data)

        # 합계 분포 분석
        sum_distribution = self.analyze_number_sum_distribution(historical_data)

        # 네트워크 분석
        network_analysis = self.analyze_network(historical_data)

        # 간격 패턴 분석
        gap_patterns = self.analyze_gap_patterns(historical_data)

        # 세그먼트 엔트로피 계산
        segment_entropy = self.calculate_segment_entropy(historical_data)

        # 클러스터 분포 분석
        cluster_result = self._find_number_clusters(historical_data)
        cluster_distribution = self.calculate_cluster_distribution(cluster_result)

        # 가중치 빈도 계산
        weighted_frequencies = self._calculate_weighted_frequencies(historical_data)

        # 빈발 쌍 분석 - graph_utils 모듈 사용
        frequent_pairs = calculate_pair_frequency(historical_data, logger=self.logger)

        # 세그먼트 10 히스토리
        segment_10_history = self.generate_segment_10_history(historical_data)

        # 세그먼트 5 히스토리
        segment_5_history = self.generate_segment_5_history(historical_data)

        # 세그먼트 중심성 분석
        segment_centrality = self.analyze_segment_centrality(historical_data)

        # 위치별 빈도 계산
        position_frequency = self.calculate_position_frequency(historical_data)

        # 위치별 번호 통계
        position_number_stats = self.calculate_position_number_stats(historical_data)

        # 격차 편차 점수
        gap_deviation_scores = self.calculate_gap_deviation_score(historical_data)

        # 다양성 점수
        diversity_scores = self.calculate_combination_diversity_score(historical_data)

        # GNN 엣지 가중치 가져오기
        gnn_edge_weights = self._get_gnn_edge_weights(historical_data)

        # 트렌드 분석기에서 트렌드 특성 가져오기
        trend_features = self._get_trend_features(historical_data)

        # 중복/유사성 분석기에서 중복 특성 가져오기
        overlap_features = self._get_overlap_features(historical_data)

        # ROI 분석기에서 ROI 특성 가져오기
        roi_features = self._get_roi_features(historical_data)

        # 추천기 물리적 구조 특성 분석
        physical_structure_features = self._analyze_physical_structure(historical_data)

        # 패턴 통계에 연속 번호 길이 분포 저장
        self.pattern_stats["consecutive_length_distribution"] = (
            consecutive_length_distribution
        )
        self.pattern_stats["odd_even_distribution"] = odd_even_distribution
        self.pattern_stats["sum_distribution"] = sum_distribution
        self.pattern_stats["network_analysis"] = network_analysis
        self.pattern_stats["gap_patterns"] = gap_patterns
        self.pattern_stats["segment_entropy"] = segment_entropy
        self.pattern_stats["cluster_distribution"] = cluster_distribution
        self.pattern_stats["weighted_frequencies"] = weighted_frequencies
        self.pattern_stats["frequent_pairs"] = frequent_pairs
        self.pattern_stats["segment_10_history"] = segment_10_history
        self.pattern_stats["segment_5_history"] = segment_5_history
        self.pattern_stats["segment_centrality"] = segment_centrality
        self.pattern_stats["position_frequency"] = position_frequency.tolist()
        self.pattern_stats["position_number_stats"] = position_number_stats
        self.pattern_stats["gap_deviation_scores"] = gap_deviation_scores
        self.pattern_stats["diversity_scores"] = diversity_scores
        self.pattern_stats["gnn_edge_weights"] = gnn_edge_weights
        self.pattern_stats["trend_features"] = trend_features
        self.pattern_stats["overlap_features"] = overlap_features
        self.pattern_stats["roi_features"] = roi_features
        self.pattern_stats["physical_structure_features"] = physical_structure_features

        # 메타데이터 설정
        result.metadata["consecutive_length_distribution"] = (
            consecutive_length_distribution
        )
        result.metadata["odd_even_distribution"] = odd_even_distribution
        result.metadata["sum_distribution"] = sum_distribution
        result.metadata["network_analysis"] = network_analysis
        result.metadata["gap_patterns"] = gap_patterns
        result.metadata["segment_entropy"] = segment_entropy
        result.metadata["cluster_distribution"] = cluster_distribution
        result.metadata["weighted_frequencies"] = weighted_frequencies
        result.metadata["frequent_pairs"] = frequent_pairs
        result.metadata["segment_10_history"] = segment_10_history
        result.metadata["segment_5_history"] = segment_5_history
        result.metadata["segment_centrality"] = segment_centrality
        result.metadata["position_frequency"] = position_frequency.tolist()
        result.metadata["position_number_stats"] = position_number_stats
        result.metadata["gap_deviation_scores"] = gap_deviation_scores
        result.metadata["diversity_scores"] = diversity_scores
        result.metadata["gnn_edge_weights"] = gnn_edge_weights
        result.metadata["trend_features"] = trend_features
        result.metadata["overlap_features"] = overlap_features
        result.metadata["roi_features"] = roi_features
        result.metadata["physical_structure_features"] = physical_structure_features

        # 결과 설정
        result.pattern_stats = self.pattern_stats
        result.pattern_vectors = None  # 벡터화는 별도 함수에서 수행

        self.logger.info("패턴 분석 완료")
        return result

    def _analyze_physical_structure(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        로또 추첨기의 물리적 구조와 관련된 특성을 분석합니다.

        물리적 구조 특성에는 번호 간 거리 분산, 연속 쌍 비율, 각 위치별 Z-점수,
        이항분포 매칭 점수, 번호 표준편차 점수 등이 포함됩니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 물리적 구조 특성 분석 결과
        """
        self.logger.info("물리적 구조 특성 분석 시작...")

        # 결과 저장용 딕셔너리
        physical_structure_features = {}

        try:
            # 1. 거리 분산 (distance_variance) 계산
            distance_variance = self._calculate_distance_variance(historical_data)
            physical_structure_features["distance_variance"] = distance_variance

            # 2. 연속 쌍 비율 (sequential_pair_rate) 계산
            sequential_pair_rate = self._calculate_sequential_pair_rate(historical_data)
            physical_structure_features["sequential_pair_rate"] = sequential_pair_rate

            # 3. 위치별 Z-점수 계산
            z_scores = self._calculate_position_z_scores(historical_data)
            physical_structure_features.update(z_scores)

            # 4. 이항분포 매칭 점수 계산
            binomial_match_score = self._calculate_binomial_match_score(historical_data)
            physical_structure_features["binomial_match_score"] = binomial_match_score

            # 5. 번호 표준편차 점수 계산
            number_std_score = self._calculate_number_std_score(historical_data)
            physical_structure_features["number_std_score"] = number_std_score

            self.logger.info("물리적 구조 특성 분석 완료")

        except Exception as e:
            self.logger.error(f"물리적 구조 특성 분석 중 오류 발생: {str(e)}")
            # 기본값 설정
            physical_structure_features = {
                "distance_variance": {"average": 0.0, "std": 0.0},
                "sequential_pair_rate": {"avg_rate": 0.0},
                "zscore_num1": 0.0,
                "zscore_num2": 0.0,
                "zscore_num3": 0.0,
                "zscore_num4": 0.0,
                "zscore_num5": 0.0,
                "zscore_num6": 0.0,
                "binomial_match_score": 0.0,
                "number_std_score": 0.0,
            }

        return physical_structure_features

    def _calculate_distance_variance(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        각 당첨 조합의 번호 간 거리 분산을 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 거리 분산 결과
                - 각 회차별 거리 분산
                - 평균, 표준편차 등 통계
        """
        try:
            # 결과 저장용 딕셔너리
            result = {}
            all_variances = []

            # 각 회차별로 거리 분산 계산
            for draw in historical_data:
                if not hasattr(draw, "draw_no") or not draw.numbers:
                    continue

                draw_key = f"draw_{draw.draw_no}"
                sorted_numbers = sorted(draw.numbers)

                # 인접한 번호 간 거리 계산
                distances = [
                    sorted_numbers[i + 1] - sorted_numbers[i]
                    for i in range(len(sorted_numbers) - 1)
                ]

                # 거리의 분산 계산
                if len(distances) > 0:
                    variance = float(np.var(distances))
                    result[draw_key] = variance
                    all_variances.append(variance)

            # 전체 통계 계산
            if all_variances:
                result["average"] = float(np.mean(all_variances))
                result["std"] = float(np.std(all_variances))
                result["min"] = float(np.min(all_variances))
                result["max"] = float(np.max(all_variances))
            else:
                result["average"] = 0.0
                result["std"] = 0.0
                result["min"] = 0.0
                result["max"] = 0.0

            return result

        except Exception as e:
            self.logger.error(f"거리 분산 계산 중 오류 발생: {str(e)}")
            return {"average": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    def _calculate_sequential_pair_rate(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        각 당첨 조합에서 연속된 번호 쌍의 비율을 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 연속 쌍 비율 결과
                - 각 회차별 연속 쌍 수
                - 평균 비율 등 통계
        """
        try:
            # 결과 저장용 딕셔너리
            result = {}
            all_sequential_pairs = []

            # 각 회차별로 연속 쌍 계산
            for draw in historical_data:
                if not hasattr(draw, "draw_no") or not draw.numbers:
                    continue

                draw_key = f"draw_{draw.draw_no}"
                sorted_numbers = sorted(draw.numbers)

                # 연속된 번호 쌍 카운트
                sequential_pairs = sum(
                    1
                    for i in range(len(sorted_numbers) - 1)
                    if sorted_numbers[i + 1] - sorted_numbers[i] == 1
                )

                result[draw_key] = sequential_pairs
                all_sequential_pairs.append(sequential_pairs)

            # 평균 연속 쌍 비율 계산
            if all_sequential_pairs:
                result["avg_rate"] = float(np.mean(all_sequential_pairs))
                result["max_rate"] = float(np.max(all_sequential_pairs))
                result["min_rate"] = float(np.min(all_sequential_pairs))
                result["std_rate"] = float(np.std(all_sequential_pairs))
            else:
                result["avg_rate"] = 0.0
                result["max_rate"] = 0.0
                result["min_rate"] = 0.0
                result["std_rate"] = 0.0

            return result

        except Exception as e:
            self.logger.error(f"연속 쌍 비율 계산 중 오류 발생: {str(e)}")
            return {"avg_rate": 0.0, "max_rate": 0.0, "min_rate": 0.0, "std_rate": 0.0}

    def _calculate_position_z_scores(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        각 위치(1~6)별 번호의 Z-점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 각 위치별 Z-점수
        """
        try:
            # 위치별 번호 저장
            position_numbers = [[] for _ in range(6)]

            # 최근 회차의 번호 (Z-점수 계산 대상)
            latest_draw = historical_data[-1] if historical_data else None

            if not latest_draw or not latest_draw.numbers:
                return {f"zscore_num{i+1}": 0.0 for i in range(6)}

            latest_numbers = sorted(latest_draw.numbers)

            # 위치별 번호 수집 (최근 회차 제외)
            for draw in historical_data[:-1]:
                sorted_numbers = sorted(draw.numbers)
                for i, num in enumerate(sorted_numbers):
                    if i < 6:  # 안전 검사
                        position_numbers[i].append(num)

            # 위치별 Z-점수 계산
            z_scores = {}
            for i in range(6):
                if position_numbers[i]:
                    mean = np.mean(position_numbers[i])
                    std = np.std(position_numbers[i])
                    # 표준편차가 0이 아닌 경우에만 Z-점수 계산
                    if std > 0 and i < len(latest_numbers):
                        z_score = (latest_numbers[i] - mean) / std
                    else:
                        z_score = 0.0
                else:
                    z_score = 0.0

                z_scores[f"zscore_num{i+1}"] = float(z_score)

            return z_scores

        except Exception as e:
            self.logger.error(f"위치별 Z-점수 계산 중 오류 발생: {str(e)}")
            return {f"zscore_num{i+1}": 0.0 for i in range(6)}

    def _calculate_binomial_match_score(
        self, historical_data: List[LotteryNumber]
    ) -> float:
        """
        각 당첨 조합이 이론적 이항분포와 얼마나 일치하는지 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            float: 이항분포 매칭 점수 (0~1)
        """
        try:
            # 최근 회차의 번호
            latest_draw = historical_data[-1] if historical_data else None

            if not latest_draw or not latest_draw.numbers:
                return 0.0

            # 이론적 이항분포 생성 (45개 번호에 대한)
            # 중앙에 가장 높은 확률, 양 끝에 낮은 확률을 가지는 분포
            x = np.arange(1, 46)
            theoretical_dist = self._generate_binomial_distribution(45)

            # 실제 번호 분포 (히스토그램)
            actual_hist = np.zeros(45)
            for num in latest_draw.numbers:
                if 1 <= num <= 45:
                    actual_hist[num - 1] = 1

            # 정규화
            theoretical_dist_norm = theoretical_dist / np.sum(theoretical_dist)
            actual_hist_norm = actual_hist / np.sum(actual_hist)

            # KL 발산 계산 (작을수록 유사)
            # 0에 가까운 값을 피하기 위해 스무딩 적용
            epsilon = 1e-10
            actual_hist_smoothed = actual_hist_norm + epsilon
            theoretical_dist_smoothed = theoretical_dist_norm + epsilon

            # 정규화
            actual_hist_smoothed = actual_hist_smoothed / np.sum(actual_hist_smoothed)
            theoretical_dist_smoothed = theoretical_dist_smoothed / np.sum(
                theoretical_dist_smoothed
            )

            # KL 발산 계산
            kl_div = np.sum(
                actual_hist_smoothed
                * np.log(actual_hist_smoothed / theoretical_dist_smoothed)
            )

            # 점수로 변환 (KL 발산이 작을수록 높은 점수)
            score = 1.0 / (1.0 + kl_div)

            return float(score)

        except Exception as e:
            self.logger.error(f"이항분포 매칭 점수 계산 중 오류 발생: {str(e)}")
            return 0.0

    def _generate_binomial_distribution(self, size: int) -> np.ndarray:
        """
        이항분포와 유사한 확률 분포를 생성합니다.

        Args:
            size: 분포 크기

        Returns:
            np.ndarray: 이항분포 확률 벡터
        """
        # 이항분포 파라미터
        n = size - 1
        p = 0.5

        # 이항분포 확률 질량 함수
        x = np.arange(size)
        from scipy.stats import binom

        probabilities = binom.pmf(x, n, p)

        return probabilities

    def _calculate_number_std_score(
        self, historical_data: List[LotteryNumber]
    ) -> float:
        """
        각 당첨 조합의 번호 표준편차 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            float: 번호 표준편차 점수 (0~1)
        """
        try:
            # 모든 회차의 번호 표준편차 계산
            std_scores = []

            # 이론적 최대 표준편차 (1,2,3,4,5,45 조합)
            max_theoretical_std = np.std([1, 2, 3, 4, 5, 45])

            for draw in historical_data:
                if not draw.numbers:
                    continue

                # 번호 표준편차 계산
                std_dev = float(np.std(draw.numbers))

                # 정규화 (최대 이론값으로 나눔)
                normalized_score = std_dev / max_theoretical_std
                std_scores.append(min(1.0, normalized_score))

            # 평균 점수 계산
            if std_scores:
                return float(np.mean(std_scores))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"번호 표준편차 점수 계산 중 오류 발생: {str(e)}")
            return 0.0

    def _calculate_weighted_frequencies(
        self, data: List[LotteryNumber]
    ) -> Dict[int, float]:
        """
        가중치가 적용된 번호 출현 빈도를 계산합니다.
        장기, 중기, 단기로 구분하여 가중치를 적용합니다.

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            가중치가 적용된 번호별 출현 빈도 (0~1 사이의 값)
        """
        if not data:
            return {num: 1.0 / 45 for num in range(1, 46)}

        # 설정에서 가중치 가져오기
        try:
            long_term_weight = self.config["frequency_weights"]["long_term"]
            mid_term_weight = self.config["frequency_weights"]["mid_term"]
            short_term_weight = self.config["frequency_weights"]["short_term"]

            # 기간 비율 설정
            mid_term_ratio = self.config["periods"][
                "mid_term"
            ]  # 중기 = 전체 데이터의 2.5%
            short_term_ratio = self.config["periods"][
                "short_term"
            ]  # 단기 = 전체 데이터의 1.2%
        except KeyError as e:
            raise KeyError(f"필수 설정값이 누락되었습니다: {e}")

        # 각 기간별 데이터 분할
        total_count = len(data)
        mid_idx = max(1, int(total_count * (1 - mid_term_ratio)))
        short_idx = max(1, int(total_count * (1 - short_term_ratio)))

        long_term_data = data[:]  # 전체 데이터
        mid_term_data = data[mid_idx:]  # 중기 데이터
        short_term_data = data[short_idx:]  # 단기 데이터

        # 각 기간별 빈도 계산
        long_term_freq = self._calculate_frequencies(long_term_data)
        mid_term_freq = self._calculate_frequencies(mid_term_data)
        short_term_freq = self._calculate_frequencies(short_term_data)

        # 가중치 적용한 빈도 계산
        weighted_freq = {}
        for num in range(1, 46):
            weighted_freq[num] = (
                long_term_weight * long_term_freq.get(num, 0)
                + mid_term_weight * mid_term_freq.get(num, 0)
                + short_term_weight * short_term_freq.get(num, 0)
            )

        return weighted_freq

    def _calculate_recency_map(self, data: List[LotteryNumber]) -> Dict[int, float]:
        """최근성 점수를 계산합니다."""
        # 초기 설정
        total_draws = len(data)
        recency_map = {num: 1.0 for num in range(1, 46)}  # 모든 번호의 초기 점수는 1.0

        # 가장 최근 데이터가 가장 큰 가중치를 갖도록 함
        for i, draw in enumerate(data):
            normalized_weight = i / total_draws  # 0에 가까울수록 오래된 데이터

            # 각 번호마다 등장 위치에 따른 최근성 점수 계산
            for num in draw.numbers:
                current_score = recency_map.get(num, 1.0)
                # 최근에 등장할수록 점수가 낮아짐 (가중치는 데이터가 최신일수록 증가)
                recency_map[num] = min(current_score, 1.0 - normalized_weight)

        # 점수 범위 정규화 (0.1 ~ 1.0)
        min_score = min(recency_map.values()) if recency_map else 0.1
        max_score = max(recency_map.values()) if recency_map else 1.0
        score_range = max_score - min_score

        if score_range > 0:
            for num in recency_map:
                normalized_score = (recency_map[num] - min_score) / score_range
                recency_map[num] = 0.1 + normalized_score * 0.9  # 0.1 ~ 1.0 범위로 조정

        return recency_map

    def _calculate_frequencies(self, data: List[LotteryNumber]) -> Dict[int, float]:
        """각 번호의 빈도를 계산합니다."""
        number_counts = {}
        total_draws = len(data)

        for draw in data:
            for num in draw.numbers:
                number_counts[num] = number_counts.get(num, 0) + 1

        return {num: count / total_draws for num, count in number_counts.items()}

    def _detect_trending_numbers(self, data: List[LotteryNumber]) -> List[int]:
        """출현 빈도가 꾸준히 증가하는 번호를 감지합니다."""
        total_draws = len(data)
        window_size = max(
            10, int(total_draws * 0.05)
        )  # 데이터의 5%를 트렌드 분석에 사용
        windows = []

        # 슬라이딩 윈도우 생성
        for i in range(0, total_draws - window_size + 1, window_size // 2):
            window_data = data[i : i + window_size]
            window_freq = self._calculate_frequencies(window_data)
            windows.append(window_freq)

        # 트렌드 분석
        trending_numbers = []
        for num in range(1, 46):
            appearances = [freq.get(num, 0) for freq in windows]
            if len(appearances) < 2:
                continue

            # 선형 회귀를 사용한 트렌드 계산
            x = np.arange(len(appearances))
            slope = np.polyfit(x, appearances, 1)[0]

            # 트렌드가 일관되게 양수인지 확인
            if slope > 0 and all(
                appearances[i] <= appearances[i + 1]
                for i in range(len(appearances) - 1)
            ):
                trending_numbers.append(num)

        return trending_numbers

    def _identify_hot_cold_numbers(
        self, frequency: Dict[int, float]
    ) -> Tuple[Set[int], Set[int]]:
        """빈도를 기반으로 인기 및 비인기 번호를 식별합니다."""
        sorted_numbers = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        hot_threshold = np.percentile([f for _, f in sorted_numbers], 75)
        cold_threshold = np.percentile([f for _, f in sorted_numbers], 25)

        hot_numbers = {num for num, freq in sorted_numbers if freq >= hot_threshold}
        cold_numbers = {num for num, freq in sorted_numbers if freq <= cold_threshold}

        return hot_numbers, cold_numbers

    def _find_number_clusters(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        번호 간 동시 출현 패턴을 기반으로 클러스터를 찾습니다.

        KMeans 클러스터링 알고리즘을 사용하여 번호를 그룹화합니다.

        Returns:
            Dict[str, Any]: 클러스터링 결과
                - clusters: 클러스터 목록 [[번호1, 번호2, ...], ...]
                - cluster_labels: 각 번호가 속한 클러스터 ID (번호 -> 클러스터 ID 매핑)
                - cluster_size_distribution: 각 클러스터의 크기
                - cooccurrence_matrix: 동시 출현 행렬 (분석용)
        """
        try:
            self.logger.info("KMeans 알고리즘으로 번호 클러스터링 시작...")

            # 1. 동시 출현 행렬 생성
            cooccurrence_matrix = np.zeros((45, 45), dtype=np.float32)

            # 모든 당첨 세트에서 동시 출현 횟수 계산
            for draw in data:
                numbers = [num - 1 for num in draw.numbers]  # 0-인덱스로 변환
                for i in range(len(numbers)):
                    for j in range(i + 1, len(numbers)):
                        cooccurrence_matrix[numbers[i], numbers[j]] += 1
                        cooccurrence_matrix[numbers[j], numbers[i]] += 1

            # 동시 출현 행렬 정규화
            total_draws = len(data)
            if total_draws > 0:
                cooccurrence_matrix /= total_draws

            # 2. KMeans 클러스터링 적용
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            # 특성 정규화
            scaler = StandardScaler()
            scaled_matrix = scaler.fit_transform(cooccurrence_matrix)

            # 최적 클러스터 수 결정 (5-9 사이에서 실루엣 점수 기준)
            from sklearn.metrics import silhouette_score

            best_n_clusters = 5
            best_score = -1

            for n_clusters in range(5, 10):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(scaled_matrix)

                    # 충분한 데이터가 있는 경우만 실루엣 점수 계산
                    if len(set(cluster_labels)) > 1:
                        score = silhouette_score(scaled_matrix, cluster_labels)
                        self.logger.info(
                            f"클러스터 수 {n_clusters}의 실루엣 점수: {score:.4f}"
                        )

                        if score > best_score:
                            best_score = score
                            best_n_clusters = n_clusters
                except Exception as e:
                    self.logger.warning(f"클러스터 수 {n_clusters} 평가 중 오류: {e}")

            # 최적 클러스터 수로 최종 클러스터링
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_matrix)

            # 3. 결과 포맷팅
            # 클러스터 목록 구성 (1-인덱스로 변환)
            clusters = [[] for _ in range(best_n_clusters)]
            for i in range(45):
                clusters[cluster_labels[i]].append(i + 1)  # 1-인덱스로 변환

            # 클러스터 ID 맵핑
            cluster_labels_map = {str(i + 1): int(cluster_labels[i]) for i in range(45)}

            # 클러스터 크기 분포
            cluster_size_distribution = {
                f"cluster_{i}": len(cluster) for i, cluster in enumerate(clusters)
            }

            self.logger.info(
                f"번호 클러스터링 완료: {best_n_clusters}개 클러스터, 실루엣 점수: {best_score:.4f}"
            )

            # 4. 클러스터 배정 결과 저장 (선택적)
            try:
                import os
                import json

                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

                # 결과 디렉토리가 없으면 생성
                data_dir = os.path.join(project_root, "data")
                os.makedirs(data_dir, exist_ok=True)

                # 클러스터 배정 저장
                cluster_file = os.path.join(data_dir, "cluster_assignments.json")
                with open(cluster_file, "w") as f:
                    json.dump({"number_clusters": cluster_labels_map}, f, indent=2)

                self.logger.info(f"클러스터 배정 결과 저장 완료: {cluster_file}")
            except Exception as e:
                self.logger.warning(f"클러스터 배정 결과 저장 중 오류: {e}")

            return {
                "clusters": clusters,
                "cluster_labels": cluster_labels_map,
                "cluster_size_distribution": cluster_size_distribution,
                "cooccurrence_matrix": cooccurrence_matrix.tolist(),
            }

        except Exception as e:
            self.logger.error(f"번호 클러스터링 중 오류 발생: {e}")
            # 오류 발생 시 기본값 반환
            default_clusters = [
                [i for i in range(1, 10)],
                [i for i in range(10, 19)],
                [i for i in range(19, 28)],
                [i for i in range(28, 37)],
                [i for i in range(37, 46)],
            ]

            default_labels = {str(i): (i - 1) // 9 for i in range(1, 46)}
            default_distribution = {
                f"cluster_{i}": len(cluster)
                for i, cluster in enumerate(default_clusters)
            }

            return {
                "clusters": default_clusters,
                "cluster_labels": default_labels,
                "cluster_size_distribution": default_distribution,
                "cooccurrence_matrix": [[0.0 for _ in range(45)] for _ in range(45)],
            }

    def _calculate_roi_metrics(
        self, data: List[LotteryNumber]
    ) -> Dict[Tuple[int, int], float]:
        """
        번호 쌍의 패턴 메트릭을 계산합니다.
        이 메트릭은 특정 번호 쌍이 포함된 추첨 번호의 패턴을 나타냅니다.

        예전에는 ROI 메트릭이었으나, 순수 패턴 기반 분석으로 변경되었습니다.

        Args:
            data: 분석할 로또 당첨 번호 목록

        Returns:
            번호 쌍별 패턴 메트릭
        """
        # 각 번호 쌍의 출현 횟수 계산
        pair_occurrences = {}
        total_draws = len(data)

        # 모든 가능한 번호 쌍에 대한 패턴 계산 준비
        for i in range(1, 46):
            for j in range(i + 1, 46):
                pair = (i, j)
                pair_occurrences[pair] = 0

        # 과거 추첨에서 번호 쌍의 출현 횟수 계산
        for draw in data:
            numbers = draw.numbers
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    num1, num2 = min(numbers[i], numbers[j]), max(
                        numbers[i], numbers[j]
                    )
                    pair = (num1, num2)
                    pair_occurrences[pair] = pair_occurrences.get(pair, 0) + 1

        # 패턴 점수 계산 (이전의 ROI 대신 패턴 중요도 점수)
        pair_pattern_score = {}
        for pair, occurrences in pair_occurrences.items():
            if total_draws > 0:
                frequency = occurrences / total_draws
                # 수정: 단순 빈도 기반 점수로 대체
                significance = 1.0 + frequency * 5.0  # 빈도가 높을수록 중요도 증가
                pair_pattern_score[pair] = significance
            else:
                pair_pattern_score[pair] = 1.0

        return pair_pattern_score

    def analyze_consecutive_length_distribution(
        self, data: List[LotteryNumber]
    ) -> PatternAnalysis:
        """
        연속 번호 길이 분포 분석

        Args:
            data: 로또 당첨 번호 데이터 목록

        Returns:
            패턴 분석 결과 (consecutive_length_distribution 포함)
        """
        # 결과 객체 초기화
        result = PatternAnalysis(
            frequency_map={},
            recency_map={},
        )

        # 각 당첨 번호 세트의 최대 연속 번호 길이 계산
        consecutive_counts = {}  # 길이별 개수 카운트

        for draw in data:
            # 정렬된 번호 목록
            sorted_numbers = sorted(draw.numbers)

            # 최대 연속 번호 길이 계산
            max_length = self.get_max_consecutive_length(sorted_numbers)

            # 카운트 증가
            if max_length in consecutive_counts:
                consecutive_counts[max_length] += 1
            else:
                consecutive_counts[max_length] = 1

        # 전체 데이터 수
        total_count = len(data)

        # 각 길이별 비율 계산 - 키를 문자열로 변환
        consecutive_distribution = {}
        for length, count in consecutive_counts.items():
            # 문자열 키로 저장
            consecutive_distribution[str(length)] = count / total_count

        # 결과 저장
        result.metadata = {}
        result.metadata["consecutive_length_distribution"] = consecutive_distribution

        # 내부 상태에도 저장
        self.pattern_stats = {}
        self.pattern_stats["consecutive_length_distribution"] = consecutive_distribution

        return result

    def get_max_consecutive_length(self, numbers: List[int]) -> int:
        """
        번호 목록에서 최대 연속 번호 길이 찾기

        Args:
            numbers: 정렬된 번호 목록

        Returns:
            최대 연속 번호 길이
        """
        if not numbers:
            return 0

        sorted_numbers = sorted(numbers)
        max_length = 1
        current_length = 1

        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 1

        return max_length

    def score_by_consecutive_pattern(self, numbers: List[int]) -> float:
        """
        연속된 번호 패턴에 따라 점수를 계산합니다.
        연속 번호가 많을수록 낮은 점수를 반환합니다.

        Args:
            numbers: 점수를 계산할 번호 목록

        Returns:
            0~1 사이의 점수 (높을수록 선호되는 패턴)
        """
        # 설정에서 최대 연속 번호 개수 가져오기
        try:
            max_consecutive = self.config["filters"]["max_consecutive"]
        except KeyError:
            raise KeyError("설정에서 'filters.max_consecutive' 키를 찾을 수 없습니다.")

        # 연속 번호 길이 계산
        consecutive_length = self.get_max_consecutive_length(sorted(numbers))

        # 연속 번호가 없는 경우
        if consecutive_length <= 1:
            return 1.0

        # 최대 허용 연속 번호를 초과하는 경우
        if consecutive_length > max_consecutive:
            return 0.0

        # 연속 번호에 따른 감점 (2개 연속: -0.1, 3개 연속: -0.3, 4개 연속: -0.6)
        penalty = ((consecutive_length - 1) ** 2) / (max_consecutive**2)

        # 최종 점수 계산 (1 - 패널티)
        return max(0.0, 1.0 - penalty)

    def extract_pattern_features(
        self, numbers: list[int], historical_data: Optional[list[LotteryNumber]] = None
    ) -> dict[str, Any]:
        """
        로또 번호 조합에서 패턴 특성 벡터 추출

        Args:
            numbers: 분석할 번호 목록
            historical_data: 참조할 과거 데이터 (선택 사항)

        Returns:
            추출된 패턴 특성 벡터
        """
        # 정렬된 번호
        sorted_numbers = sorted(numbers)

        # 1. 최대 연속 길이
        max_consecutive_length = self.get_max_consecutive_length(sorted_numbers)

        # 2. 번호 합계
        total_sum = sum(sorted_numbers)

        # 3. 홀수/짝수 개수
        odd_count = sum(1 for num in sorted_numbers if num % 2 == 1)
        even_count = len(sorted_numbers) - odd_count

        # 4. 간격 평균 및 표준편차
        gaps = [
            sorted_numbers[i + 1] - sorted_numbers[i]
            for i in range(len(sorted_numbers) - 1)
        ]
        gap_avg = sum(gaps) / len(gaps) if gaps else 0
        gap_std = np.std(gaps) if gaps else 0

        # 5. 범위별 분포 (1-9, 10-19, 20-29, 30-39, 40-45)
        range_counts = [0, 0, 0, 0, 0]
        for num in sorted_numbers:
            if 1 <= num <= 9:
                range_counts[0] += 1
            elif 10 <= num <= 19:
                range_counts[1] += 1
            elif 20 <= num <= 29:
                range_counts[2] += 1
            elif 30 <= num <= 39:
                range_counts[3] += 1
            else:  # 40-45
                range_counts[4] += 1

        # 고급 특성들 (과거 데이터가 있을 경우)
        cluster_overlap_ratio = 0.0
        frequent_pair_score = 0.0
        roi_weight = 0.0

        if historical_data:
            # 패턴 분석 수행
            pattern_analysis = self.analyze(historical_data)

            # 6. 클러스터 오버랩 비율
            if pattern_analysis.clusters:
                overlap_count = 0
                for cluster in pattern_analysis.clusters:
                    cluster_set = set(cluster)
                    numbers_set = set(sorted_numbers)
                    overlap = cluster_set.intersection(numbers_set)
                    if len(overlap) >= 2:  # 클러스터와 2개 이상 일치
                        overlap_count += len(overlap)

                cluster_overlap_ratio = (
                    overlap_count / len(sorted_numbers) if overlap_count > 0 else 0
                )

            # 7. 빈도 페어 점수
            if pattern_analysis.pair_frequency:
                pair_scores = []
                for i in range(len(sorted_numbers)):
                    for j in range(i + 1, len(sorted_numbers)):
                        num1, num2 = min(sorted_numbers[i], sorted_numbers[j]), max(
                            sorted_numbers[i], sorted_numbers[j]
                        )
                        pair = (num1, num2)
                        if pair in pattern_analysis.pair_frequency:
                            pair_scores.append(pattern_analysis.pair_frequency[pair])

                frequent_pair_score = (
                    sum(pair_scores) / len(pair_scores) if pair_scores else 0
                )

            # 8. ROI 가중치 (수익성 점수)
            if pattern_analysis.roi_matrix:
                roi_scores = []
                for i in range(len(sorted_numbers)):
                    for j in range(i + 1, len(sorted_numbers)):
                        num1, num2 = min(sorted_numbers[i], sorted_numbers[j]), max(
                            sorted_numbers[i], sorted_numbers[j]
                        )
                        pair = (num1, num2)
                        if pair in pattern_analysis.roi_matrix:
                            roi_scores.append(pattern_analysis.roi_matrix[pair])

                roi_weight = sum(roi_scores) / len(roi_scores) if roi_scores else 0

        # 9. 연속 번호 점수
        consecutive_score = self.score_by_consecutive_pattern(sorted_numbers)

        # 10. 새로운 특성: 트렌드 점수 (과거 데이터가 있을 경우)
        trend_score_avg = 0.0
        trend_score_max = 0.0
        trend_score_min = 0.0

        if historical_data and len(historical_data) > 0:
            # 트렌드 점수 계산
            trend_scores = self.get_number_trend_scores(historical_data)
            number_trend_scores = [trend_scores.get(num, 0.0) for num in sorted_numbers]

            if number_trend_scores:
                trend_score_avg = sum(number_trend_scores) / len(number_trend_scores)
                trend_score_max = max(number_trend_scores)
                trend_score_min = min(number_trend_scores)

        # 11. 새로운 특성: 위험도 점수
        risk_score = self.calculate_risk_score(sorted_numbers, historical_data)

        # 결과 딕셔너리 생성
        result = {
            "max_consecutive_length": max_consecutive_length,
            "total_sum": total_sum,
            "odd_count": odd_count,
            "even_count": even_count,
            "gap_avg": gap_avg,
            "gap_std": gap_std,
            "range_counts": range_counts,
            "cluster_overlap_ratio": cluster_overlap_ratio,
            "frequent_pair_score": frequent_pair_score,
            "roi_weight": roi_weight,
            "consecutive_score": consecutive_score,
            "trend_score_avg": trend_score_avg,
            "trend_score_max": trend_score_max,
            "trend_score_min": trend_score_min,
            "risk_score": risk_score,
        }

        return result

    def vectorize_pattern_features(
        self, features: Union[PatternFeatures, Dict[str, Any]]
    ) -> np.ndarray:
        """
        패턴 특성 딕셔너리를 모델 입력용 특성 벡터로 변환

        Args:
            features: extract_pattern_features()로 추출한 특성 딕셔너리

        Returns:
            고정 길이 벡터 (numpy 배열)
        """
        # 안전하게 값 추출 (기본값 제공)
        max_consecutive_length = features.get("max_consecutive_length", 0)
        total_sum = features.get("total_sum", 0)
        odd_count = features.get("odd_count", 0)
        even_count = features.get("even_count", 0)
        gap_avg = features.get("gap_avg", 0.0)
        gap_std = features.get("gap_std", 0.0)
        range_counts = features.get("range_counts", [0, 0, 0, 0, 0])
        cluster_overlap_ratio = features.get("cluster_overlap_ratio", 0.0)
        frequent_pair_score = features.get("frequent_pair_score", 0.0)
        roi_weight = features.get("roi_weight", 0.0)
        consecutive_score = features.get("consecutive_score", 0.0)
        trend_score_avg = features.get("trend_score_avg", 0.0)
        trend_score_max = features.get("trend_score_max", 0.0)
        trend_score_min = features.get("trend_score_min", 0.0)
        risk_score = features.get("risk_score", 0.0)

        # 고정된 순서로 특성 추출 (일관성을 위해)
        feature_vector = np.array(
            [
                max_consecutive_length / 6.0,  # 정규화: 최대 6
                total_sum
                / 270.0,  # 정규화: 최대 합계 (45+44+43+42+41+40=255, 약간 여유)
                odd_count / 6.0,
                even_count / 6.0,
                gap_avg / 20.0,  # 정규화: 일반적인 최대 간격
                gap_std / 15.0,  # 정규화: 일반적인 최대 표준편차
                *[count / 6.0 for count in range_counts],  # 5개 요소
                cluster_overlap_ratio,  # 이미 0-1 범위
                frequent_pair_score * 10.0,  # 강화: 일반적으로 작은 값
                roi_weight / 2.0,  # 정규화: 일반적인 최대값
                consecutive_score + 0.3,  # 시프트: -0.3~0.25 → 0~0.55
                trend_score_avg * 10.0,  # 강화: 일반적으로 작은 값
                trend_score_max * 10.0,  # 강화: 일반적으로 작은 값
                trend_score_min * 10.0,  # 강화: 일반적으로 작은 값
                risk_score,  # 이미 0-1 범위
            ]
        )

        return feature_vector

    def pattern_penalty(self, features: PatternFeatures) -> float:
        """
        패턴 특성에 따른 패널티 점수를 계산합니다.
        특정 패턴이 통계적으로 불리할 경우 높은 패널티를 적용합니다.

        Args:
            features: 패턴 특성 객체

        Returns:
            0~1 사이의 패널티 점수 (낮을수록 선호되는 패턴)
        """
        penalty = 0.0

        # 연속된 번호에 대한 패널티
        if "max_consecutive_length" in features:
            try:
                max_consecutive = self.config["filters"]["max_consecutive"]
            except KeyError:
                max_consecutive = 4  # 기본값
                self.logger.warning(
                    "'filters.max_consecutive' 설정을 찾을 수 없습니다. 기본값(4)을 사용합니다."
                )

            if features["max_consecutive_length"] > max_consecutive:
                penalty += 0.3
            elif features["max_consecutive_length"] == max_consecutive:
                penalty += 0.1

        # 합계에 대한 패널티
        if "total_sum" in features:
            try:
                min_sum = self.config["filters"]["min_sum"]
            except KeyError:
                min_sum = 90  # 기본값
                self.logger.warning(
                    "'filters.min_sum' 설정을 찾을 수 없습니다. 기본값(90)을 사용합니다."
                )

            try:
                max_sum = self.config["filters"]["max_sum"]
            except KeyError:
                max_sum = 210  # 기본값
                self.logger.warning(
                    "'filters.max_sum' 설정을 찾을 수 없습니다. 기본값(210)을 사용합니다."
                )

            if features["total_sum"] < min_sum or features["total_sum"] > max_sum:
                penalty += 0.3

        # 홀짝 비율에 대한 패널티
        if "odd_count" in features and "even_count" in features:
            try:
                min_even = self.config["filters"]["min_even_numbers"]
            except KeyError:
                min_even = 2  # 기본값
                self.logger.warning(
                    "'filters.min_even_numbers' 설정을 찾을 수 없습니다. 기본값(2)을 사용합니다."
                )

            try:
                max_even = self.config["filters"]["max_even_numbers"]
            except KeyError:
                max_even = 4  # 기본값
                self.logger.warning(
                    "'filters.max_even_numbers' 설정을 찾을 수 없습니다. 기본값(4)을 사용합니다."
                )

            even_count = features["even_count"]
            if even_count < min_even or even_count > max_even:
                penalty += 0.2

        # 다른 패턴 특성에 따른 패널티...
        if "roi_weight" in features and features["roi_weight"] < 0.3:
            penalty += 0.15

        # 트렌드 점수에 따른 패널티
        if "trend_score_avg" in features:
            try:
                trend_threshold = self.config["pattern_features"]["thresholds"][
                    "trend_score_threshold"
                ]
            except KeyError:
                trend_threshold = 0.75  # 기본값
                self.logger.warning(
                    "'pattern_features.thresholds.trend_score_threshold' 설정을 찾을 수 없습니다. 기본값(0.75)을 사용합니다."
                )

            if features["trend_score_avg"] < trend_threshold:
                penalty += 0.1

        # 리스크 점수에 따른 패널티
        if "risk_score" in features:
            try:
                risk_threshold = self.config["pattern_features"]["thresholds"][
                    "risk_score_threshold"
                ]
            except KeyError:
                risk_threshold = 0.65  # 기본값
                self.logger.warning(
                    "'pattern_features.thresholds.risk_score_threshold' 설정을 찾을 수 없습니다. 기본값(0.65)을 사용합니다."
                )

            if features["risk_score"] > risk_threshold:
                penalty += 0.2

        # 최종 패널티는 0~1 사이로 제한
        return min(1.0, penalty)

    # 트렌드 분석 메서드 추가
    def get_number_trend_scores(
        self,
        historical_data: List[LotteryNumber],
        window_size: int = 30,
        alpha: float = 0.1,
    ) -> Dict[int, float]:
        """
        각 번호의 출현 트렌드를 EWMA(Exponentially Weighted Moving Average)를 사용하여 계산합니다.

        Args:
            historical_data: 과거 당첨 번호 데이터
            window_size: 분석할 최근 회차 수
            alpha: EWMA 가중치 파라미터 (0-1 사이, 작을수록 과거 값의 영향이 큼)

        Returns:
            각 번호의 트렌드 점수 (높을수록 최근에 자주 등장)
        """
        # 최근 window_size 회차 데이터만 사용
        recent_data = (
            historical_data[-window_size:]
            if len(historical_data) > window_size
            else historical_data
        )

        # 모든 번호에 대한 EWMA 계산을 위한 준비
        number_appearances = {num: [] for num in range(1, 46)}

        # 각 회차별로 번호 출현 여부 기록 (1: 출현, 0: 미출현)
        for draw in recent_data:
            draw_numbers = set(draw.numbers)
            for num in range(1, 46):
                number_appearances[num].append(1 if num in draw_numbers else 0)

        # EWMA 계산
        trend_scores = {}
        for num in range(1, 46):
            if not number_appearances[num]:
                trend_scores[num] = 0.0
                continue

            # EWMA 계산
            ewma = number_appearances[num][0]
            for i in range(1, len(number_appearances[num])):
                ewma = alpha * number_appearances[num][i] + (1 - alpha) * ewma

            trend_scores[num] = ewma

        # 정규화 (합이 1이 되도록)
        total_score = sum(trend_scores.values())
        if total_score > 0:
            trend_scores = {
                num: score / total_score for num, score in trend_scores.items()
            }

        return trend_scores

    # 위험도 점수 계산 메서드 추가
    def calculate_risk_score(
        self, numbers: List[int], historical_data: Optional[List[LotteryNumber]] = None
    ) -> float:
        """
        번호 조합의 위험도 점수를 계산합니다.

        Args:
            numbers: 평가할 번호 목록
            historical_data: 참조할 과거 데이터 (선택 사항)

        Returns:
            위험도 점수 (0-1 사이, 1에 가까울수록 위험)
        """
        if not numbers or len(numbers) != 6:
            return 1.0  # 잘못된 형식의 번호는 최대 위험도

        sorted_numbers = sorted(numbers)
        risk_score = 0.0

        # 1. 합계 위험도 (70 미만 또는 210 초과)
        total_sum = sum(sorted_numbers)
        if total_sum < 70:
            risk_score += 0.25
        elif total_sum > 210:
            risk_score += 0.25
        elif total_sum < 90 or total_sum > 170:
            risk_score += 0.1

        # 2. 홀짝 불균형 위험도
        odd_count = sum(1 for num in sorted_numbers if num % 2 == 1)
        even_count = 6 - odd_count

        if odd_count == 6 or even_count == 6:
            risk_score += 0.25  # 모두 홀수 또는 모두 짝수
        elif odd_count == 5 or even_count == 5:
            risk_score += 0.1  # 5:1 불균형

        # 3. 연속 번호 위험도
        max_consecutive = self.get_max_consecutive_length(sorted_numbers)
        if max_consecutive >= 5:
            risk_score += 0.25  # 5개 이상 연속
        elif max_consecutive == 4:
            risk_score += 0.15  # 4개 연속
        elif max_consecutive == 3:
            risk_score += 0.05  # 3개 연속

        # 4. 번호 분포 위험도 (1-9, 10-19, 20-29, 30-39, 40-45)
        range_counts = [0] * 5
        for num in sorted_numbers:
            if 1 <= num <= 9:
                range_counts[0] += 1
            elif 10 <= num <= 19:
                range_counts[1] += 1
            elif 20 <= num <= 29:
                range_counts[2] += 1
            elif 30 <= num <= 39:
                range_counts[3] += 1
            else:
                range_counts[4] += 1

        # 한 구간에 3개 이상 몰린 경우
        if max(range_counts) >= 4:
            risk_score += 0.2
        elif max(range_counts) == 3:
            risk_score += 0.1

        # 5. 과거 빈도 기반 위험도 (선택적)
        if historical_data:
            repeat_count = 0
            # 과거 동일한 번호 조합 출현 횟수 확인
            for draw in historical_data:
                common_numbers = set(draw.numbers).intersection(set(numbers))
                if len(common_numbers) >= 5:  # 5개 이상 일치
                    repeat_count += 1

            if repeat_count > 0:
                risk_score += min(0.2, repeat_count * 0.05)  # 최대 0.2까지 추가

        # 최종 위험도 점수 (0-1 범위로 제한)
        return min(1.0, risk_score)

    def get_number_frequencies(self, data: List[LotteryNumber]) -> Dict[int, int]:
        """
        각 번호별 출현 횟수를 계산합니다.

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            각 번호의 출현 횟수 (번호 -> 출현 횟수)
        """
        frequencies = {}
        for draw in data:
            for num in draw.numbers:
                frequencies[num] = frequencies.get(num, 0) + 1

        return frequencies

    def analyze_scope(
        self, historical_data: List[LotteryNumber], scope: str = "full"
    ) -> PatternAnalysis:
        """
        지정된 스코프에 대한 패턴 분석을 수행합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            scope: 분석 스코프 ("full", "mid", "short" 중 하나)

        Returns:
            PatternAnalysis: 패턴 분석 결과
        """
        with self.performance_tracker.track(f"pattern_analysis_{scope}"):
            try:
                # 캐시 키 생성
                cache_key = self._create_cache_key(
                    f"pattern_analysis_{scope}", len(historical_data)
                )

                # 캐시 확인
                cached_result = self._check_cache(cache_key)
                if cached_result:
                    # 스코프별 결과 저장
                    self.scoped_analyses[scope] = cached_result
                    return cached_result

                # 패턴 분석 수행
                result = self.analyze(historical_data)

                # 결과 객체에 스코프 정보 추가
                result.metadata["scope"] = scope
                result.metadata["data_count"] = len(historical_data)

                # 스코프별 결과 저장
                self.scoped_analyses[scope] = result

                # 결과 캐싱
                self.cache_manager.set(cache_key, result)

                return result
            except Exception as e:
                self.logger.error(f"{scope} 스코프 패턴 분석 중 오류 발생: {str(e)}")
                return PatternAnalysis(
                    frequency_map={num: 1.0 / 45 for num in range(1, 46)},
                    recency_map={num: 0.5 for num in range(1, 46)},
                )

    def run_all_analyses(self, draw_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        모든 분석을 수행하고 결과를 통합합니다.

        Args:
            draw_data: 분석할 로또 번호 데이터

        Returns:
            Dict[str, Any]: 통합된 분석 결과
        """
        try:
            # 분석 시작 시간 기록
            start_time = datetime.now()
            self.logger.info(f"전체 분석 시작 (데이터: {len(draw_data)}개 회차)")

            # 스코프별 분석 수행
            self.logger.info("기본 패턴 분석 수행 중...")
            full_analysis = self.analyze_scope(draw_data, "full")
            recent_100 = self.analyze_scope(
                draw_data[-100:] if len(draw_data) >= 100 else draw_data, "recent_100"
            )
            last_year_count = min(52, len(draw_data))
            last_year = self.analyze_scope(draw_data[-last_year_count:], "last_year")
            last_month_count = min(4, len(draw_data))
            last_month = self.analyze_scope(draw_data[-last_month_count:], "last_month")

            # 세그먼트 빈도 분석
            self.logger.info("세그먼트 빈도 분석 수행 중...")
            segment_10_frequency = self.analyze_segment_frequency_10(draw_data)
            segment_5_frequency = self.analyze_segment_frequency_5(draw_data)

            # 갭 통계 분석
            self.logger.info("갭 통계 분석 수행 중...")
            gap_statistics = self.analyze_gap_statistics(draw_data)

            # 명시적으로 gap_stddev 필드 추가
            if "std" in gap_statistics:
                gap_stddev = gap_statistics["std"]
            else:
                # 직접 계산
                gaps = []
                for i in range(1, len(draw_data)):
                    prev_numbers = set(draw_data[i - 1].numbers)
                    curr_numbers = set(draw_data[i].numbers)
                    gap = 6 - len(prev_numbers.intersection(curr_numbers))
                    gaps.append(gap)
                gap_stddev = np.std(gaps) if gaps else 0.0

            # 필수 필드 추가
            self.logger.info("필수 필드 gap_stddev 계산 중...")

            # 패턴 재출현 분석
            self.logger.info("패턴 재출현 분석 수행 중...")
            pattern_reappearance = self.analyze_pattern_reappearance(draw_data)

            # 최근 재출현 간격 분석
            self.logger.info("최근 재출현 간격 분석 수행 중...")
            recent_reappearance_gap = self.analyze_recent_reappearance_gap(draw_data)

            # 회차별 구간 빈도 히스토리 생성
            self.logger.info("구간 빈도 히스토리 생성 중...")
            segment_10_history = self.generate_segment_10_history(draw_data)
            segment_5_history = self.generate_segment_5_history(draw_data)

            # 세그먼트 중심성 분석
            self.logger.info("세그먼트 중심성 분석 수행 중...")
            segment_centrality = self.analyze_segment_centrality(draw_data)

            # 세그먼트 연속 패턴 분석
            self.logger.info("세그먼트 연속 패턴 분석 수행 중...")
            segment_consecutive_patterns = self.analyze_segment_consecutive_patterns(
                draw_data
            )

            # 중복 당첨 번호 분석
            self.logger.info("중복 당첨 번호 분석 수행 중...")
            identical_draw_check = self.analyze_identical_draws(draw_data)

            # 필수 필드: 중복 플래그
            duplicate_flag = (
                1 if identical_draw_check.get("duplicate_count", 0) > 0 else 0
            )

            # ===== 추가 분석 항목 실행 =====
            # 위치별 번호 빈도 계산
            self.logger.info("위치별 번호 빈도 계산 중...")
            position_frequency = self.calculate_position_frequency(draw_data)

            # 위치별 번호 통계 계산
            self.logger.info("위치별 번호 통계 계산 중...")
            position_number_stats = self.calculate_position_number_stats(draw_data)

            # 세그먼트 추세 히스토리 계산
            self.logger.info("세그먼트 추세 히스토리 계산 중...")
            segment_trend_history = self.calculate_segment_trend_history(draw_data)

            # 필수 필드: 세그먼트 엔트로피 계산
            self.logger.info("세그먼트 엔트로피 계산 중...")
            segment_entropy = self.calculate_segment_entropy(draw_data)

            # 필수 필드: ROI 그룹 점수 계산
            self.logger.info("ROI 패턴 그룹 식별 중...")
            roi_pattern_groups = self.identify_roi_pattern_groups(draw_data)
            roi_group_score = 0.0
            if (
                isinstance(roi_pattern_groups, dict)
                and "group_scores" in roi_pattern_groups
            ):
                if isinstance(roi_pattern_groups["group_scores"], dict):
                    roi_group_score = sum(
                        roi_pattern_groups["group_scores"].values()
                    ) / max(1, len(roi_pattern_groups["group_scores"]))

            # 필수 필드: Hot-Cold 믹스 점수 계산
            self.logger.info("Hot-Cold 믹스 점수 계산 중...")
            hot_cold_mix_score = 0.0

            # 빈도에 따라 핫/콜드 번호 식별
            frequencies = self._calculate_frequencies(draw_data)
            hot_numbers, cold_numbers = self._identify_hot_cold_numbers(frequencies)

            # 최근 100회 데이터에서 핫/콜드 믹스 패턴 분석
            recent_draws = draw_data[-100:] if len(draw_data) >= 100 else draw_data
            hot_cold_mix_counts = []

            for draw in recent_draws:
                hot_count = sum(1 for num in draw.numbers if num in hot_numbers)
                cold_count = sum(1 for num in draw.numbers if num in cold_numbers)
                # 핫/콜드 믹스 비율 (0: 모두 핫 또는 모두 콜드, 1: 균등 믹스)
                mix_ratio = min(hot_count, cold_count) / 3  # 최대 3:3 균형이면 1.0
                hot_cold_mix_counts.append(mix_ratio)

            hot_cold_mix_score = sum(hot_cold_mix_counts) / max(
                1, len(hot_cold_mix_counts)
            )

            # 갭 편차 점수 계산
            self.logger.info("갭 편차 점수 계산 중...")
            gap_deviation_score = self.calculate_gap_deviation_score(draw_data)

            # 조합 다양성 점수 계산
            self.logger.info("조합 다양성 점수 계산 중...")
            combination_diversity = self.calculate_combination_diversity_score(
                draw_data
            )

            # ROI 트렌드 계산
            self.logger.info("ROI 트렌드 계산 중...")
            roi_trend_by_pattern = self.calculate_roi_trend_by_pattern(draw_data)

            # 모든 분석 결과를 하나의 딕셔너리로 통합
            combined_result = {
                # 기존 분석 결과
                "frequency_analysis": {
                    "full": full_analysis.frequency_map,
                    "recent_100": recent_100.frequency_map,
                    "last_year": last_year.frequency_map,
                    "last_month": last_month.frequency_map,
                },
                "recency_analysis": {
                    "full": full_analysis.recency_map,
                    "recent_100": recent_100.recency_map,
                    "last_year": last_year.recency_map,
                    "last_month": last_month.recency_map,
                },
                "segment_10_frequency": segment_10_frequency,
                "segment_5_frequency": segment_5_frequency,
                "gap_statistics": gap_statistics,
                "pattern_reappearance": pattern_reappearance,
                "recent_reappearance_gap": recent_reappearance_gap,
                "segment_centrality": segment_centrality,
                "segment_consecutive_patterns": segment_consecutive_patterns,
                "identical_draws": identical_draw_check,
                # 추가 분석 결과
                "position_frequency": (
                    position_frequency.tolist()
                    if isinstance(position_frequency, np.ndarray)
                    else position_frequency
                ),
                "position_number_stats": position_number_stats,
                "segment_trend_history": (
                    segment_trend_history.tolist()
                    if isinstance(segment_trend_history, np.ndarray)
                    else segment_trend_history
                ),
                "gap_deviation_score": gap_deviation_score,
                "combination_diversity": combination_diversity,
                "roi_trend_by_pattern": roi_trend_by_pattern,
                "roi_pattern_groups": roi_pattern_groups,
                # 필수 필드 명시적 추가
                "gap_stddev": gap_stddev,
                "hot_cold_mix_score": hot_cold_mix_score,
                "segment_entropy": segment_entropy,
                "roi_group_score": roi_group_score,
                "duplicate_flag": duplicate_flag,
                "pair_centrality": segment_centrality,  # 기존 segment_centrality를 pair_centrality로도 사용
            }

            # 성능 보고서 저장
            try:
                self._save_analysis_performance_report(
                    combined_result, draw_data, start_time
                )
            except Exception as e:
                self.logger.warning(f"성능 보고서 저장 중 오류: {str(e)}")

            return combined_result

        except Exception as e:
            self.logger.error(f"전체 분석 중 오류 발생: {str(e)}")
            # 성능 보고서에 오류 기록
            try:
                self._save_analysis_performance_report(
                    {}, draw_data, start_time, str(e)
                )
            except Exception as log_error:
                self.logger.error(f"오류 로깅 중 추가 오류 발생: {str(log_error)}")

            # 최소한의 결과 반환
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _save_analysis_performance_report(
        self,
        analysis_result: Dict[str, Any],
        draw_data: List[LotteryNumber],
        start_time: datetime,
        error: Optional[str] = None,
    ) -> None:
        """
        패턴 분석 성능 보고서 저장
        performance_report_writer.py의 함수를 사용하여 성능 리포트를 저장합니다.

        Args:
            analysis_result: 분석 결과
            draw_data: 분석한 당첨 번호 데이터
            start_time: 분석 시작 시간
            error: 에러 메시지 (있는 경우)
        """
        try:
            # 성능 데이터 수집
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # 분석 메트릭스 정보 (필수 필드만 포함)
            data_metrics = {
                "record_count": len(draw_data) if draw_data else 0,
                "execution_time_sec": execution_time,
                "status": "error" if error else "success",
            }

            # 오류 정보 추가
            if error:
                data_metrics["error"] = error

            # 성능 리포트 저장 함수 호출
            from ..utils.performance_report_writer import (
                save_analysis_performance_report,
            )

            save_analysis_performance_report(
                None,  # profiler
                self.performance_tracker,
                self.config,
                "pattern_analyzer",
                data_metrics,
            )

            self.logger.info("패턴 분석 성능 보고서 저장 완료")

        except Exception as e:
            self.logger.error(f"성능 보고서 저장 중 오류 발생: {e}")

    def get_analysis_by_scope(self, scope: str = "full") -> Optional[PatternAnalysis]:
        """
        특정 스코프의 패턴 분석 결과를 반환합니다.

        Args:
            scope: 분석 스코프 ("full", "mid", "short" 중 하나)

        Returns:
            PatternAnalysis: 패턴 분석 결과 또는 None
        """
        return self.scoped_analyses.get(scope)

    def get_all_analyses(self) -> Dict[str, PatternAnalysis]:
        """
        모든 스코프의 패턴 분석 결과를 반환합니다.

        Returns:
            Dict[str, PatternAnalysis]: 스코프별 패턴 분석 결과
        """
        return self.scoped_analyses

    def blend_frequency_maps(
        self,
        long_weight: float = 0.6,
        mid_weight: float = 0.25,
        short_weight: float = 0.15,
    ) -> Dict[int, float]:
        """
        다양한 스코프의 빈도 맵을 가중치에 따라 혼합합니다.

        Args:
            long_weight: 전체 스코프의 가중치
            mid_weight: 중기 스코프의 가중치
            short_weight: 단기 스코프의 가중치

        Returns:
            Dict[int, float]: 혼합된 빈도 맵
        """
        # 각 스코프의 빈도 맵 가져오기
        full_freq = self.scoped_analyses.get(
            "full", PatternAnalysis(frequency_map={}, recency_map={})
        ).frequency_map
        mid_freq = self.scoped_analyses.get(
            "mid", PatternAnalysis(frequency_map={}, recency_map={})
        ).frequency_map
        short_freq = self.scoped_analyses.get(
            "short", PatternAnalysis(frequency_map={}, recency_map={})
        ).frequency_map

        # 스코프별 데이터가 없는 경우 기본값 설정
        for num in range(1, 46):
            if num not in full_freq:
                full_freq[num] = 1.0 / 45
            if num not in mid_freq:
                mid_freq[num] = 1.0 / 45
            if num not in short_freq:
                short_freq[num] = 1.0 / 45

        # 가중치 혼합 빈도 계산
        blended_freq = {}
        for num in range(1, 46):
            blended_freq[num] = (
                full_freq.get(num, 0) * long_weight
                + mid_freq.get(num, 0) * mid_weight
                + short_freq.get(num, 0) * short_weight
            )

        return blended_freq

    def save_analyses_to_cache(self, cache_dir: str = "data/cache") -> bool:
        """
        모든 스코프의 패턴 분석 결과를 캐시 파일로 저장합니다.
        BaseAnalyzer의 캐시 메서드를 활용합니다.

        Args:
            cache_dir: 캐시 디렉토리 경로

        Returns:
            bool: 저장 성공 여부
        """
        try:
            success = True
            for scope, analysis in self.scoped_analyses.items():
                # 캐시 키 생성
                cache_key = self._create_cache_key(f"pattern_analysis_{scope}", 0)

                # 캐시에 저장 (BaseAnalyzer 메서드 활용)
                self._save_to_cache(cache_key, analysis)
                self.logger.info(f"{scope} 스코프 분석 결과 캐시 저장 완료")

            return success
        except Exception as e:
            self.logger.error(f"분석 결과 저장 중 오류 발생: {str(e)}")
            return False

    def load_analyses_from_cache(self, cache_dir: str = "data/cache") -> bool:
        """
        모든 스코프의 패턴 분석 결과를 캐시 파일에서 로드합니다.
        BaseAnalyzer의 캐시 메서드를 활용합니다.

        Args:
            cache_dir: 캐시 디렉토리 경로

        Returns:
            bool: 로드 성공 여부
        """
        try:
            loaded_count = 0
            for scope in ["full", "mid", "short"]:
                # 캐시 키 생성
                cache_key = self._create_cache_key(f"pattern_analysis_{scope}", 0)

                # 캐시에서 로드 (BaseAnalyzer 메서드 활용)
                analysis = self._check_cache(cache_key)
                if analysis:
                    self.scoped_analyses[scope] = analysis
                    loaded_count += 1
                    self.logger.info(f"{scope} 스코프 분석 결과 캐시 로드 완료")

            return loaded_count > 0
        except Exception as e:
            self.logger.error(f"분석 결과 로드 중 오류 발생: {str(e)}")
            return False

    def write_analysis_report(
        self, output_dir: str = "data/test_results", test_mode: bool = False
    ) -> str:
        """
        이 메서드는 더 이상 사용되지 않으며, 분석 결과를 텍스트나 JSON 파일로 저장하지 않습니다.
        대신 벡터 형태로만 저장됩니다.

        성능 리포트는 performance_report_writer.py에서 담당합니다.

        Args:
            output_dir: 사용되지 않음
            test_mode: 사용되지 않음

        Returns:
            str: 경고 메시지
        """
        self.logger.warning(
            "write_analysis_report 메서드는 더 이상 사용되지 않습니다. "
            "분석 결과는 벡터 형태로만 저장되며, 성능 리포트는 performance_report_writer.py에서 처리합니다."
        )
        return "분석 결과는 벡터 형태로만, 성능 리포트는 별도로 저장됩니다."

    def analyze_odd_even_distribution(
        self, data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        당첨 번호의 홀짝 비율 분포를 분석합니다.

        Args:
            data: 로또 당첨 번호 데이터 목록

        Returns:
            홀짝 비율별 분포 (예: "3:3" -> 0.45)
        """
        distribution = {}
        total_draws = len(data)

        for draw in data:
            # 홀수/짝수 개수 계산
            odd_count = sum(1 for num in draw.numbers if num % 2 == 1)
            even_count = len(draw.numbers) - odd_count

            # 비율 문자열 생성 (예: "3:3", "4:2")
            ratio = f"{odd_count}:{even_count}"

            # 분포 누적
            distribution[ratio] = distribution.get(ratio, 0) + 1

        # 비율로 변환
        if total_draws > 0:
            distribution = {
                ratio: count / total_draws for ratio, count in distribution.items()
            }

        return distribution

    def analyze_number_sum_distribution(
        self, data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        당첨 번호의 합계 분포를 분석합니다.

        Args:
            data: 로또 당첨 번호 데이터 목록

        Returns:
            합계별 분포 (예: "125" -> 0.03)
        """
        distribution = {}
        total_draws = len(data)

        for draw in data:
            # 번호 합계 계산
            num_sum = sum(draw.numbers)

            # 합계 문자열
            sum_str = str(num_sum)

            # 분포 누적
            distribution[sum_str] = distribution.get(sum_str, 0) + 1

        # 비율로 변환
        if total_draws > 0:
            distribution = {
                sum_str: count / total_draws for sum_str, count in distribution.items()
            }

        return distribution

    def analyze_network(self, lottery_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        로또 번호의 동시 출현 네트워크를 구성하고 분석합니다.

        Args:
            lottery_data: 로또 당첨 번호 전체 데이터

        Returns:
            Dict[str, Any]: 네트워크 분석 결과
        """
        with self.performance_tracker.track("network_analysis"):
            # 1. 동시 출현 그래프 구성
            G = nx.Graph()

            # 모든 로또 번호 노드 추가 (1~45)
            for i in range(1, 46):
                G.add_node(i)

            # 동시 출현 엣지 추가
            edge_weights = defaultdict(int)

            for draw in lottery_data:
                numbers = sorted(draw.numbers)
                # 모든 번호 쌍에 대해 엣지 가중치 계산
                for i in range(len(numbers)):
                    for j in range(i + 1, len(numbers)):
                        edge = (numbers[i], numbers[j])
                        edge_weights[edge] += 1

            # 가중치가 부여된 엣지를 그래프에 추가
            for (source, target), weight in edge_weights.items():
                G.add_edge(source, target, weight=weight)

            # 2. 네트워크 분석
            # 2.1 모든 동시 출현 엣지 (내림차순 정렬)
            edges = []
            for (source, target), weight in sorted(
                edge_weights.items(), key=lambda x: x[1], reverse=True
            ):
                edges.append({"source": source, "target": target, "weight": weight})

            # 2.2 노드별 연결 정보
            nodes_data = []
            for node in range(1, 46):
                neighbors = list(G.neighbors(node))
                # NetworkX 2.8.4 버전과의 호환성을 위해 degree 계산 방식 변경
                degree = len(neighbors)
                weighted_degree = sum(
                    G[node][neighbor]["weight"] for neighbor in neighbors
                )

                nodes_data.append(
                    {
                        "number": node,
                        "connections": sorted(neighbors),
                        "degree": degree,
                        "weighted_degree": weighted_degree,
                    }
                )

            # 노드를 weighted_degree 기준으로 정렬
            nodes_data = sorted(
                nodes_data, key=lambda x: x["weighted_degree"], reverse=True
            )

            # 2.3 중심성 측정
            # 2.3.1 degree centrality
            degree_centrality = nx.degree_centrality(G)

            # 2.3.2 weighted degree (수동 계산)
            weighted_degree = {}
            for node in range(1, 46):
                neighbors = list(G.neighbors(node))
                weighted_degree[node] = sum(
                    G[node][neighbor]["weight"] for neighbor in neighbors
                )

            # 2.3.3 eigenvector centrality
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, weight="weight")
            except:
                # 수렴 실패 시 최대 반복 횟수 증가
                eigenvector_centrality = nx.eigenvector_centrality(
                    G, weight="weight", max_iter=1000
                )

            # 중심성 결과 통합
            centrality = {
                "degree": {str(k): v for k, v in degree_centrality.items()},
                "weighted_degree": {str(k): v for k, v in weighted_degree.items()},
                "eigenvector": {str(k): v for k, v in eigenvector_centrality.items()},
            }

            # 3. 결과 반환 - Graph 객체는 JSON 직렬화 불가능하므로 제외
            return {"edges": edges, "nodes": nodes_data, "centrality": centrality}

    def analyze_segment_frequency_10(
        self, data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        10개 구간으로 나누어 번호별 출현 빈도 분석

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 10개 구간별 빈도 정보
        """
        with self.performance_tracker.track("analyze_segment_frequency_10"):
            # 10개 구간 정의 (1-5, 6-10, ..., 41-45)
            segments = {i: [] for i in range(1, 11)}
            segment_size = 5  # 각 구간 크기 (마지막 구간은 1-5만)

            # 모든 번호를 구간별로 분류
            for num in range(1, 46):
                segment_idx = (num - 1) // segment_size + 1
                segments[segment_idx].append(num)

            # 각 구간별 출현 빈도 계산
            segment_frequencies = {}

            # 전체 번호 빈도 계산
            number_freqs = self.get_number_frequencies(data)

            # 구간별 빈도 합산
            for segment_idx, numbers in segments.items():
                segment_freq = sum(number_freqs.get(num, 0) for num in numbers)
                segment_frequencies[str(segment_idx)] = segment_freq

            return segment_frequencies

    def analyze_segment_frequency_5(
        self, data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        5개 구간으로 나누어 번호별 출현 빈도 분석

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 5개 구간별 빈도 정보
        """
        with self.performance_tracker.track("analyze_segment_frequency_5"):
            # 5개 구간 정의 (1-9, 10-18, 19-27, 28-36, 37-45)
            segments = {i: [] for i in range(1, 6)}
            segment_size = 9  # 각 구간 크기

            # 모든 번호를 구간별로 분류
            for num in range(1, 46):
                segment_idx = (num - 1) // segment_size + 1
                segments[segment_idx].append(num)

            # 각 구간별 출현 빈도 계산
            segment_frequencies = {}

            # 전체 번호 빈도 계산
            number_freqs = self.get_number_frequencies(data)

            # 구간별 빈도 합산
            for segment_idx, numbers in segments.items():
                segment_freq = sum(number_freqs.get(num, 0) for num in numbers)
                segment_frequencies[str(segment_idx)] = segment_freq

            return segment_frequencies

    def analyze_gap_statistics(self, data: List[LotteryNumber]) -> Dict[str, float]:
        """
        번호 간 간격 통계 분석 (평균 간격, 표준편차 등)

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 간격 통계 정보
        """
        with self.performance_tracker.track("analyze_gap_statistics"):
            all_gaps = []

            # 각 회차별 번호 간격 계산
            for draw in data:
                numbers = sorted(draw.numbers)

                # 번호 간 간격 계산
                gaps = [numbers[i] - numbers[i - 1] for i in range(1, len(numbers))]
                all_gaps.extend(gaps)

            # 간격 통계 계산
            if all_gaps:
                gap_avg = float(np.mean(all_gaps))
                gap_std = float(np.std(all_gaps))
                gap_min = float(np.min(all_gaps))
                gap_max = float(np.max(all_gaps))

                return {"avg": gap_avg, "std": gap_std, "min": gap_min, "max": gap_max}

            return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    def analyze_pattern_reappearance(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        패턴(구조, 클러스터 등) 재출현 간격 분석

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 패턴 재출현 통계 정보
        """
        with self.performance_tracker.track("analyze_pattern_reappearance"):
            result = {}

            # 1. 홀짝 패턴 재출현 분석
            odd_even_patterns = {}
            for i, draw in enumerate(data):
                numbers = draw.numbers
                odd_count = sum(1 for num in numbers if num % 2 == 1)
                even_count = len(numbers) - odd_count
                pattern_key = f"{odd_count}:{even_count}"

                if pattern_key in odd_even_patterns:
                    # 마지막 출현 이후 경과한 회차 수 계산
                    gap = i - odd_even_patterns[pattern_key][-1]
                    odd_even_patterns[pattern_key].append(i)
                    odd_even_patterns[pattern_key + "_gaps"].append(gap)
                else:
                    odd_even_patterns[pattern_key] = [i]
                    odd_even_patterns[pattern_key + "_gaps"] = []

            # 홀짝 패턴별 평균 재출현 간격
            odd_even_reappearance = {}
            for key in odd_even_patterns:
                if "_gaps" in key and odd_even_patterns[key]:
                    pattern = key.replace("_gaps", "")
                    odd_even_reappearance[pattern] = {
                        "avg_gap": float(np.mean(odd_even_patterns[key])),
                        "intervals": odd_even_patterns[key],
                    }

            result["odd_even"] = odd_even_reappearance

            # 2. 구간 분포 패턴 재출현 분석
            # 구간 정의 (1-15, 16-30, 31-45)
            range_patterns = {}
            for i, draw in enumerate(data):
                numbers = draw.numbers
                low = sum(1 for num in numbers if 1 <= num <= 15)
                mid = sum(1 for num in numbers if 16 <= num <= 30)
                high = sum(1 for num in numbers if 31 <= num <= 45)
                pattern_key = f"{low}:{mid}:{high}"

                if pattern_key in range_patterns:
                    gap = i - range_patterns[pattern_key][-1]
                    range_patterns[pattern_key].append(i)
                    range_patterns[pattern_key + "_gaps"].append(gap)
                else:
                    range_patterns[pattern_key] = [i]
                    range_patterns[pattern_key + "_gaps"] = []

            # 구간 패턴별 평균 재출현 간격
            range_reappearance = {}
            for key in range_patterns:
                if "_gaps" in key and range_patterns[key]:
                    pattern = key.replace("_gaps", "")
                    range_reappearance[pattern] = {
                        "avg_gap": float(np.mean(range_patterns[key])),
                        "intervals": range_patterns[key],
                    }

            result["range"] = range_reappearance

            # 3. 연속 번호 패턴 재출현 분석
            consec_patterns = {}
            for i, draw in enumerate(data):
                numbers = sorted(draw.numbers)
                max_consec = self.get_max_consecutive_length(numbers)
                pattern_key = f"consec_{max_consec}"

                if pattern_key in consec_patterns:
                    gap = i - consec_patterns[pattern_key][-1]
                    consec_patterns[pattern_key].append(i)
                    consec_patterns[pattern_key + "_gaps"].append(gap)
                else:
                    consec_patterns[pattern_key] = [i]
                    consec_patterns[pattern_key + "_gaps"] = []

            # 연속 패턴별 평균 재출현 간격
            consec_reappearance = {}
            for key in consec_patterns:
                if "_gaps" in key and consec_patterns[key]:
                    pattern = key.replace("_gaps", "")
                    consec_reappearance[pattern] = {
                        "avg_gap": float(np.mean(consec_patterns[key])),
                        "intervals": consec_patterns[key],
                    }

            result["consecutive"] = consec_reappearance

            return result

    def analyze_recent_reappearance_gap(
        self, data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        각 번호별 최근 재출현 간격 분석

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 번호별 최근 재출현 간격
        """
        with self.performance_tracker.track("analyze_recent_reappearance_gap"):
            # 각 번호별 최근 출현 회차 기록
            latest_appearance = {}

            # 번호별 최근 재출현 간격
            recent_gaps = {}

            # 최근 회차부터 역순으로 처리
            for i, draw in enumerate(data):
                draw_idx = i
                for num in draw.numbers:
                    if num in latest_appearance:
                        # 번호가 이전에 나온 적이 있으면 간격 계산
                        gap = draw_idx - latest_appearance[num]
                        recent_gaps[str(num)] = gap

                    # 최근 출현 회차 업데이트
                    latest_appearance[num] = draw_idx

            # 모든 번호(1~45)에 대해 기본값 설정
            for num in range(1, 46):
                if str(num) not in recent_gaps:
                    recent_gaps[str(num)] = len(data) if num in latest_appearance else 0

            return recent_gaps

    # 새로운 메서드 추가: 회차별 10구간 빈도 히스토리 생성
    def generate_segment_10_history(
        self, data: List[LotteryNumber]
    ) -> Dict[str, List[int]]:
        """
        회차별 10구간 빈도 히스토리 생성

        각 회차마다 10구간별 번호 출현 횟수를 계산합니다:
        - 구간 1: 1~5
        - 구간 2: 6~10
        - 구간 3: 11~15
        - 구간 4: 16~20
        - 구간 5: 21~25
        - 구간 6: 26~30
        - 구간 7: 31~35
        - 구간 8: 36~40
        - 구간 9: 41~45

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, List[int]]: 회차별 10구간 빈도 벡터
            {회차번호: [구간1 개수, 구간2 개수, ..., 구간9 개수]}
        """
        with self.performance_tracker.track("generate_segment_10_history"):
            # 구간 정의 (1-5, 6-10, ..., 41-45)
            segment_size = 5  # 각 구간 크기
            segment_count = 9  # 총 구간 수

            # 회차별 구간 빈도 벡터 저장을 위한 딕셔너리
            segment_history = {}

            # 각 회차별로 구간 빈도 계산
            for draw in data:
                if not hasattr(draw, "draw_no") or not draw.draw_no:
                    continue

                # 현재 회차의 빈도 벡터 초기화
                frequency_vector = [0] * segment_count

                # 번호별로 해당 구간 카운트 증가
                for num in draw.numbers:
                    segment_idx = (num - 1) // segment_size
                    if 0 <= segment_idx < segment_count:
                        frequency_vector[segment_idx] += 1

                # 회차 번호를 문자열로 변환하여 저장
                segment_history[str(draw.draw_no)] = frequency_vector

            return segment_history

    # 새로운 메서드 추가: 회차별 5구간 빈도 히스토리 생성
    def generate_segment_5_history(
        self, data: List[LotteryNumber]
    ) -> Dict[str, List[int]]:
        """
        회차별 5구간 빈도 히스토리 생성

        각 회차마다 5구간별 번호 출현 횟수를 계산합니다:
        - 구간 1: 1~9
        - 구간 2: 10~18
        - 구간 3: 19~27
        - 구간 4: 28~36
        - 구간 5: 37~45

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, List[int]]: 회차별 5구간 빈도 벡터
            {회차번호: [구간1 개수, 구간2 개수, ..., 구간5 개수]}
        """
        with self.performance_tracker.track("generate_segment_5_history"):
            # 구간 정의 (1-9, 10-18, ..., 37-45)
            segment_size = 9  # 각 구간 크기
            segment_count = 5  # 총 구간 수

            # 회차별 구간 빈도 벡터 저장을 위한 딕셔너리
            segment_history = {}

            # 각 회차별로 구간 빈도 계산
            for draw in data:
                if not hasattr(draw, "draw_no") or not draw.draw_no:
                    continue

                # 현재 회차의 빈도 벡터 초기화
                frequency_vector = [0] * segment_count

                # 번호별로 해당 구간 카운트 증가
                for num in draw.numbers:
                    segment_idx = (num - 1) // segment_size
                    if 0 <= segment_idx < segment_count:
                        frequency_vector[segment_idx] += 1

                # 회차 번호를 문자열로 변환하여 저장
                segment_history[str(draw.draw_no)] = frequency_vector

            return segment_history

    # 새로운 메서드 추가: 세그먼트 별 중심성 분석
    def analyze_segment_centrality(self, data: List[LotteryNumber]) -> Dict[str, float]:
        """
        세그먼트 별 중심성 분석

        기존 네트워크 그래프에서 각 세그먼트(1~5, 6~10, ..., 41~45)별로
        그룹화하여 중심성 점수를 계산합니다.

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 세그먼트별 중심성 점수
            {"1~5": 0.21, "6~10": 0.28, ...}
        """
        with self.performance_tracker.track("analyze_segment_centrality"):
            # 1. 네트워크 그래프 구성 (analyze_network 메서드와 유사하게)
            G = nx.Graph()

            # 모든 로또 번호 노드 추가 (1~45)
            for i in range(1, 46):
                G.add_node(i)

            # 동시 출현 엣지 추가
            edge_weights = defaultdict(int)

            for draw in data:
                numbers = sorted(draw.numbers)
                # 모든 번호 쌍에 대해 엣지 가중치 계산
                for i in range(len(numbers)):
                    for j in range(i + 1, len(numbers)):
                        edge = (numbers[i], numbers[j])
                        edge_weights[edge] += 1

            # 가중치가 부여된 엣지를 그래프에 추가
            for (source, target), weight in edge_weights.items():
                G.add_edge(source, target, weight=weight)

            # 2. 중심성 계산
            try:
                degree_centrality = nx.degree_centrality(G)
                eigenvector_centrality = nx.eigenvector_centrality(
                    G, weight="weight", max_iter=1000
                )
            except Exception as e:
                self.logger.error(f"중심성 계산 중 오류: {str(e)}")
                # 오류 발생 시 빈 결과 반환
                return {}

            # 3. 세그먼트 정의 (10개 구간)
            segment_size = 5  # 각 구간 크기
            segments = {}

            for i in range(1, 10):  # 9개 세그먼트 (1~5, 6~10, ..., 41~45)
                start_num = (i - 1) * segment_size + 1
                end_num = i * segment_size
                segment_key = f"{start_num}~{end_num}"
                segments[segment_key] = []

                # 해당 세그먼트에 속하는 번호 추가
                for num in range(start_num, end_num + 1):
                    if num <= 45:  # 45를 초과하는 번호는 없음
                        segments[segment_key].append(num)

            # 4. 세그먼트별 중심성 계산
            segment_centrality = {}

            for segment_key, numbers in segments.items():
                # 해당 세그먼트 내 번호들의 중심성 평균 계산
                degree_sum = sum(degree_centrality.get(num, 0) for num in numbers)
                eigenvector_sum = sum(
                    eigenvector_centrality.get(num, 0) for num in numbers
                )

                # 번호 개수로 정규화
                count = len(numbers)
                if count > 0:
                    segment_centrality[segment_key] = {
                        "degree": degree_sum / count,
                        "eigenvector": eigenvector_sum / count,
                    }

                    # 간편한 사용을 위해 eigenvector 중심성을 기본값으로 설정
                    segment_centrality[segment_key] = eigenvector_sum / count

            return segment_centrality

    # 새로운 메서드 추가: 세그먼트 별 연속 패턴 분석
    def analyze_segment_consecutive_patterns(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Dict[str, int]]:
        """
        세그먼트 별 연속 패턴 분석

        각 당첨 번호에서 연속된 번호 그룹을 추출하고, 어떤 세그먼트에 속하는지 분석합니다.

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Dict[str, int]]: 세그먼트별 연속 패턴 통계
            {
                "1~5": {"count_2": 37, "count_3": 12},
                "6~10": {"count_2": 25, "count_3": 8},
                ...
            }
        """
        with self.performance_tracker.track("analyze_segment_consecutive_patterns"):
            # 1. 세그먼트 정의 (10개 구간)
            segment_size = 5  # 각 구간 크기
            segments = {}
            segment_consecutive_counts = {}

            for i in range(1, 10):  # 9개 세그먼트 (1~5, 6~10, ..., 41~45)
                start_num = (i - 1) * segment_size + 1
                end_num = i * segment_size
                segment_key = f"{start_num}~{end_num}"
                segments[segment_key] = set(range(start_num, end_num + 1))

                # 각 세그먼트별 연속 패턴 카운트 초기화
                segment_consecutive_counts[segment_key] = {
                    "count_2": 0,
                    "count_3": 0,
                    "count_4+": 0,
                }

            # 2. 각 당첨 번호에서 연속 패턴 분석
            for draw in data:
                sorted_numbers = sorted(draw.numbers)

                # 연속된 번호 그룹 찾기
                consecutive_groups = []
                current_group = [sorted_numbers[0]]

                for i in range(1, len(sorted_numbers)):
                    if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                        # 연속된 번호
                        current_group.append(sorted_numbers[i])
                    else:
                        # 연속이 끊김
                        if (
                            len(current_group) >= 2
                        ):  # 2개 이상 연속된 경우만 그룹으로 간주
                            consecutive_groups.append(current_group)
                        current_group = [sorted_numbers[i]]

                # 마지막 그룹 확인
                if len(current_group) >= 2:
                    consecutive_groups.append(current_group)

                # 각 연속 그룹이 어떤 세그먼트에 속하는지 확인
                for group in consecutive_groups:
                    group_len = len(group)
                    if group_len < 2:
                        continue

                    # 그룹이 어느 세그먼트에 속하는지 확인
                    for segment_key, segment_numbers in segments.items():
                        # 그룹의 모든 번호가 현재 세그먼트에 속하는지 확인
                        if all(num in segment_numbers for num in group):
                            # 연속 패턴 길이에 따라 카운트 증가
                            if group_len == 2:
                                segment_consecutive_counts[segment_key]["count_2"] += 1
                            elif group_len == 3:
                                segment_consecutive_counts[segment_key]["count_3"] += 1
                            else:  # 4개 이상
                                segment_consecutive_counts[segment_key]["count_4+"] += 1
                            break

            return segment_consecutive_counts

    # 여기에 새로운 메서드 추가
    def analyze_identical_draws(self, draw_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        중복 당첨 번호 세트를 분석합니다.

        각 당첨 번호 세트가 과거에 이미 나왔는지 확인하고,
        중복된 세트를 추적합니다.

        Args:
            draw_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 중복 당첨 번호 분석 결과
        """
        with self.performance_tracker.track("analyze_identical_draws"):
            # 당첨 번호 세트별 등장 회차 기록
            combination_appearances = {}

            # 결과 저장용 딕셔너리
            duplicates = {}

            # 각 회차별로 중복 검사
            for draw in draw_data:
                # 번호를 튜플로 변환 (불변 키로 사용하기 위함)
                numbers_tuple = tuple(sorted(draw.numbers))
                numbers_key = ",".join(map(str, numbers_tuple))

                # 회차 번호 또는 인덱스
                draw_id = (
                    draw.draw_no if hasattr(draw, "draw_no") and draw.draw_no else 0
                )

                # 이 조합이 이미 등장했는지 확인
                if numbers_tuple in combination_appearances:
                    # 이미 등장한 적이 있는 조합
                    combination_appearances[numbers_tuple].append(draw_id)

                    # 중복 딕셔너리에 추가
                    duplicates[numbers_key] = combination_appearances[numbers_tuple]
                else:
                    # 처음 등장하는 조합
                    combination_appearances[numbers_tuple] = [draw_id]

            # 총 중복 개수
            total_duplicates = len(duplicates)

            # 결과 구성
            result = {
                "total_duplicates": total_duplicates,
                "duplicates": duplicates,
                "has_duplicates": total_duplicates > 0,
            }

            # 디버그 로그
            if total_duplicates > 0:
                self.logger.info(f"총 {total_duplicates}개의 중복 당첨 번호 세트 발견")
                for numbers, occurrences in duplicates.items():
                    self.logger.info(f"번호 세트 {numbers}: {occurrences}회차에 등장")
            else:
                self.logger.info("중복된 당첨 번호 세트가 없습니다.")

            return result

    def is_combination_duplicate(
        self, numbers: List[int], draw_data: List[LotteryNumber]
    ) -> bool:
        """
        주어진 번호 조합이 과거에 당첨된 적 있는지 확인합니다.

        Args:
            numbers: 확인할 번호 조합
            draw_data: 과거 당첨 번호 데이터

        Returns:
            bool: 중복 여부 (True: 과거에 당첨됨, False: 처음 등장)
        """
        numbers_tuple = tuple(sorted(numbers))
        count = 0

        for draw in draw_data:
            draw_tuple = tuple(sorted(draw.numbers))
            if draw_tuple == numbers_tuple:
                count += 1
                # 2번 이상 등장하면 중복으로 판단
                if count >= 2:
                    return True

        return False

    def load_from_cache(self, numbers: List[int]) -> Optional[Dict[str, Any]]:
        """
        주어진 번호에 대한 캐시된 분석 결과 로드

        Args:
            numbers: 로또 번호 리스트

        Returns:
            캐시된 패턴 분석 결과 (없으면 None)
        """
        try:
            # 캐시 키 생성 (BaseAnalyzer 메서드 활용)
            sorted_numbers = sorted(numbers)
            numbers_str = "_".join(map(str, sorted_numbers))
            cache_key = self._create_cache_key(f"pattern_analysis_{numbers_str}", 0)

            # 캐시에서 로드 (BaseAnalyzer 메서드 활용)
            result = self._check_cache(cache_key)
            if result:
                self.logger.info(f"캐시된 패턴 분석 결과 로드: {cache_key}")
                return result

            return None
        except Exception as e:
            self.logger.warning(f"캐시 로드 실패: {str(e)}")
            return None

    def _compute_numbers_hash(self, numbers: List[int]) -> str:
        """
        번호 리스트에 대한 고유 해시 생성 (BaseAnalyzer 메서드 활용)

        Args:
            numbers: 로또 번호 리스트

        Returns:
            고유 해시 문자열
        """
        # 정렬된 번호 목록으로 해시 생성
        sorted_numbers = sorted(numbers)
        numbers_str = "_".join(map(str, sorted_numbers))
        # BaseAnalyzer의 _create_cache_key 메서드로 대체될 수 있음
        return f"pattern_analysis_{numbers_str}"

    def analyze_gap_patterns(self, data: List[LotteryNumber]) -> Dict[int, int]:
        """
        연속된 번호 간의 간격 패턴을 분석합니다.

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[int, int]: 간격별 출현 횟수 (e.g., {1: 빈도, 2: 빈도, ..., 10+: 빈도})
        """
        with self.performance_tracker.track("analyze_gap_patterns"):
            gap_counts = {i: 0 for i in range(1, 11)}  # 1~10까지의 간격
            gap_counts[11] = 0  # 11 이상의 간격을 위한 키

            total_gaps = 0

            for draw in data:
                numbers = sorted(draw.numbers)

                # 연속된 번호 간의 간격 계산
                for i in range(1, len(numbers)):
                    gap = numbers[i] - numbers[i - 1]
                    total_gaps += 1

                    if gap <= 10:
                        gap_counts[gap] += 1
                    else:
                        gap_counts[11] += 1  # 11 이상의 간격은 하나로 그룹화

            # 빈도를 비율로 변환
            if total_gaps > 0:
                gap_patterns = {
                    gap: count / total_gaps for gap, count in gap_counts.items()
                }
            else:
                gap_patterns = {gap: 0.0 for gap in gap_counts.keys()}

            self.logger.info(f"간격 패턴 분석 완료: {len(gap_patterns)}개 패턴")
            return gap_patterns

    def generate_probability_matrix(
        self, data: List[LotteryNumber]
    ) -> Dict[Tuple[int, int], float]:
        """
        번호 간 동시 출현 확률 행렬을 생성합니다.

        Args:
            data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[Tuple[int, int], float]: 번호 쌍별 동시 출현 확률
        """
        with self.performance_tracker.track("generate_probability_matrix"):
            # 확률 행렬 초기화 (45x45)
            probability_matrix = {}
            for i in range(1, 46):
                for j in range(1, 46):
                    probability_matrix[(i, j)] = 0.0

            # 번호별 출현 횟수
            number_counts = {num: 0 for num in range(1, 46)}

            # 동시 출현 횟수
            cooccurrence_counts = {}
            for i in range(1, 46):
                for j in range(1, 46):
                    cooccurrence_counts[(i, j)] = 0

            # 데이터 분석
            total_draws = len(data)
            if total_draws == 0:
                self.logger.warning("데이터가 없어 확률 행렬을 생성할 수 없습니다.")
                return probability_matrix

            for draw in data:
                numbers = draw.numbers

                # 각 번호 출현 횟수 증가
                for num in numbers:
                    number_counts[num] += 1

                # 번호 쌍 동시 출현 횟수 증가
                for i in numbers:
                    for j in numbers:
                        if i != j:  # 자기 자신과의 동시 출현은 계산하지 않음
                            cooccurrence_counts[(i, j)] += 1

            # 조건부 확률 계산 P(j|i) = P(i,j) / P(i)
            for i in range(1, 46):
                for j in range(1, 46):
                    if i != j and number_counts[i] > 0:
                        probability_matrix[(i, j)] = (
                            cooccurrence_counts[(i, j)] / number_counts[i]
                        )

            # 행렬 대칭성 및 범위 검증
            self._validate_probability_matrix(probability_matrix)

            self.logger.info(f"확률 행렬 생성 완료: {len(probability_matrix)}개 원소")
            return probability_matrix

    def _validate_probability_matrix(
        self, matrix: Dict[Tuple[int, int], float]
    ) -> None:
        """
        확률 행렬의 유효성을 검증합니다.

        Args:
            matrix: 검증할 확률 행렬
        """
        # 모든 값이 [0,1] 범위 내에 있는지 확인
        invalid_values = 0
        for key, value in matrix.items():
            if not (0 <= value <= 1):
                invalid_values += 1
                matrix[key] = max(0, min(value, 1))  # 값을 범위 내로 조정

        if invalid_values > 0:
            self.logger.warning(
                f"{invalid_values}개의 확률 값이 [0,1] 범위를 벗어나 조정되었습니다."
            )

    def calculate_segment_entropy(
        self, data: List[LotteryNumber], segments: int = 5
    ) -> Dict[str, float]:
        """
        각 세그먼트의 엔트로피를 계산하여 다양성을 측정합니다.

        Args:
            data: 과거 당첨 번호 목록
            segments: 세그먼트 수 (기본값: 5 - 1-9, 10-19, 20-29, 30-39, 40-45)

        Returns:
            Dict[str, float]: 각 세그먼트의 엔트로피 및 종합 지표
        """
        with self.performance_tracker.track("calculate_segment_entropy"):
            segment_size = 45 // segments
            segment_counts = np.zeros((segments, len(data)))

            # 각 세그먼트별 번호 출현 빈도 집계
            for draw_idx, draw in enumerate(data):
                for num in draw.numbers:
                    seg_idx = min((num - 1) // segment_size, segments - 1)
                    segment_counts[seg_idx, draw_idx] += 1

            # 세그먼트별 엔트로피 계산
            segment_entropy = {}
            segment_entropy_values = np.zeros(segments)

            for seg_idx in range(segments):
                segment_name = f"segment_{seg_idx+1}"
                # 회차별 해당 세그먼트 번호 출현 확률 계산
                segment_probs = segment_counts[seg_idx] / 6  # 각 회차에 6개 번호 출현
                segment_probs = segment_probs[segment_probs > 0]  # 0인 확률 제외

                if len(segment_probs) > 0:
                    # 섀넌 엔트로피 계산: -sum(p * log2(p))
                    entropy = -np.sum(segment_probs * np.log2(segment_probs + 1e-10))
                    segment_entropy[segment_name] = float(entropy)
                    segment_entropy_values[seg_idx] = entropy
                else:
                    segment_entropy[segment_name] = 0.0
                    segment_entropy_values[seg_idx] = 0.0

            # 전체 엔트로피 평균 및 표준편차 계산
            segment_entropy["mean"] = float(np.mean(segment_entropy_values))
            segment_entropy["std"] = float(np.std(segment_entropy_values))
            segment_entropy["min"] = float(np.min(segment_entropy_values))
            segment_entropy["max"] = float(np.max(segment_entropy_values))

            # 편차 지수 (최대와 최소의 차이를 평균으로 나눈 값)
            if segment_entropy["mean"] > 0:
                segment_entropy["deviation_index"] = float(
                    (segment_entropy["max"] - segment_entropy["min"])
                    / segment_entropy["mean"]
                )
            else:
                segment_entropy["deviation_index"] = 0.0

            self.logger.info(f"세그먼트 엔트로피 계산 완료: {segments}개 세그먼트")
            return segment_entropy

    def calculate_cluster_distribution(
        self, cluster_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        클러스터 분포 및 품질 지표를 계산합니다.

        Args:
            cluster_result: _find_number_clusters 메서드의 반환 값

        Returns:
            Dict[str, Any]: 클러스터 분포 및 품질 지표
        """
        self.logger.info("클러스터 품질 지표 계산 중...")

        # 클러스터 데이터 가져오기
        clusters = cluster_result.get("clusters", [])
        cluster_labels = cluster_result.get("cluster_labels", {})
        cluster_size_distribution = cluster_result.get("cluster_size_distribution", {})
        cooccurrence_matrix = cluster_result.get("cooccurrence_matrix", [])

        # 결과 저장용 딕셔너리
        result = {}

        if not clusters:
            self.logger.warning("클러스터가 없어 분포를 계산할 수 없습니다.")
            return {"no_clusters": 0, "silhouette_score": 0.0, "balance_score": 0.0}

        # 1. 기본 클러스터 정보
        result["total_clusters"] = len(clusters)
        result["cluster_size_distribution"] = cluster_size_distribution

        # 클러스터 크기 계산
        cluster_sizes = [len(cluster) for cluster in clusters]
        result["mean_size"] = (
            float(sum(cluster_sizes) / len(clusters)) if clusters else 0.0
        )
        result["max_size"] = float(max(cluster_sizes)) if clusters else 0.0
        result["min_size"] = float(min(cluster_sizes)) if clusters else 0.0
        result["largest_cluster_size"] = float(max(cluster_sizes)) if clusters else 0.0

        # 2. 클러스터 균형 점수
        if max(cluster_sizes) > 0:
            result["balance_score"] = float(min(cluster_sizes) / max(cluster_sizes))
            result["cluster_size_ratio"] = float(
                max(cluster_sizes) / min(cluster_sizes)
            )
        else:
            result["balance_score"] = 0.0
            result["cluster_size_ratio"] = 0.0

        # 3. 클러스터 엔트로피 점수 (Shannon 엔트로피)
        try:
            # 클러스터 크기의 확률 분포 계산
            total_numbers = sum(cluster_sizes)
            probabilities = [size / total_numbers for size in cluster_sizes]

            # 엔트로피 계산: -Σ(p_i * log2(p_i))
            import math

            entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)

            # 최대 가능 엔트로피로 정규화 (균등 분포의 엔트로피)
            max_entropy = math.log2(len(clusters)) if len(clusters) > 0 else 1.0
            result["cluster_entropy_score"] = (
                float(entropy / max_entropy) if max_entropy > 0 else 0.0
            )
        except Exception as e:
            self.logger.warning(f"클러스터 엔트로피 계산 중 오류: {e}")
            result["cluster_entropy_score"] = 0.0

        # 4. 클러스터 간 거리 통계
        try:
            if isinstance(cooccurrence_matrix, list) and len(cooccurrence_matrix) > 0:
                # NumPy 배열로 변환
                matrix = np.array(cooccurrence_matrix)

                # 클러스터 중심점 계산
                centroids = []
                for cluster in clusters:
                    # 0-인덱스로 변환
                    indices = [num - 1 for num in cluster]
                    # 클러스터에 속한 번호들의 동시 출현 벡터 평균
                    if indices:
                        centroid = np.mean(matrix[indices], axis=0)
                        centroids.append(centroid)

                # 클러스터 간 거리 계산
                distances = []
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        # 유클리드 거리 계산
                        dist = np.linalg.norm(centroids[i] - centroids[j])
                        distances.append(dist)

                if distances:
                    result["avg_distance_between_clusters"] = float(np.mean(distances))
                    result["distance_variance"] = float(np.var(distances))
                else:
                    result["avg_distance_between_clusters"] = 0.0
                    result["distance_variance"] = 0.0
            else:
                result["avg_distance_between_clusters"] = 0.0
                result["distance_variance"] = 0.0
        except Exception as e:
            self.logger.warning(f"클러스터 간 거리 계산 중 오류: {e}")
            result["avg_distance_between_clusters"] = 0.0
            result["distance_variance"] = 0.0

        # 5. 실루엣 점수 계산
        try:
            from sklearn.metrics import silhouette_score

            # 클러스터 라벨 준비
            if (
                cluster_labels
                and isinstance(cooccurrence_matrix, list)
                and len(cooccurrence_matrix) > 0
            ):
                # NumPy 배열로 변환
                matrix = np.array(cooccurrence_matrix)

                # 클러스터 라벨 배열 생성
                labels_array = np.zeros(45, dtype=int)
                for num_str, cluster_id in cluster_labels.items():
                    num = int(num_str) - 1  # 0-인덱스로 변환
                    if 0 <= num < 45:
                        labels_array[num] = cluster_id

                # 실루엣 점수 계산 (클러스터가 2개 이상이고 각 클러스터에 2개 이상의 샘플이 있어야 함)
                unique_labels = set(labels_array)
                if len(unique_labels) >= 2 and all(
                    np.sum(labels_array == label) >= 2 for label in unique_labels
                ):
                    score = silhouette_score(matrix, labels_array)
                    result["silhouette_score"] = float(score)
                else:
                    result["silhouette_score"] = 0.0
            else:
                result["silhouette_score"] = 0.0
        except Exception as e:
            self.logger.warning(f"실루엣 점수 계산 중 오류: {e}")
            result["silhouette_score"] = 0.0

        # 로그 출력
        self.logger.info(f"클러스터 분석 완료: {len(clusters)}개 클러스터")
        self.logger.info(f"실루엣 점수: {result.get('silhouette_score', 0.0):.4f}")
        self.logger.info(f"균형 점수: {result.get('balance_score', 0.0):.4f}")
        self.logger.info(
            f"엔트로피 점수: {result.get('cluster_entropy_score', 0.0):.4f}"
        )
        self.logger.info(
            f"클러스터 간 평균 거리: {result.get('avg_distance_between_clusters', 0.0):.4f}"
        )

        return result

    def calculate_position_frequency(
        self, historical_data: List[LotteryNumber]
    ) -> np.ndarray:
        """
        각 위치별 번호 출현 빈도를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            np.ndarray: 위치별 번호 출현 빈도 (6x45 행렬)
        """
        with self.performance_tracker.track("calculate_position_frequency"):
            # 6개 위치, 45개 번호에 대한 빈도 행렬 초기화 (0-indexed)
            position_matrix = np.zeros((6, 45), dtype=np.int32)

            # 각 회차의 당첨 번호에 대해 위치별 빈도 계산
            for draw in historical_data:
                # 오름차순 정렬된 번호 사용
                sorted_numbers = sorted(draw.numbers)
                for position, number in enumerate(sorted_numbers):
                    # 인덱스는 0부터 시작하므로 번호에서 1을 빼줌
                    position_matrix[position, number - 1] += 1

            # 결과 저장
            try:
                cache_dir = Path(self.config["paths"]["cache_dir"])
                cache_dir.mkdir(parents=True, exist_ok=True)

                file_path = cache_dir / "position_number_matrix.npy"
                np.save(file_path, position_matrix)
                self.logger.info(f"위치별 번호 빈도 행렬 저장 완료: {file_path}")
            except Exception as e:
                self.logger.error(f"위치별 번호 빈도 행렬 저장 중 오류 발생: {e}")

            return position_matrix

    def calculate_position_number_stats(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Dict[str, float]]:
        """
        각 위치별, 번호별 출현 확률을 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Dict[str, float]]: 위치별 번호 출현 확률
        """
        with self.performance_tracker.track("calculate_position_number_stats"):
            # 위치별 번호 빈도 행렬 계산
            position_matrix = self.calculate_position_frequency(historical_data)
            total_draws = len(historical_data)

            # 위치별 번호 확률 계산
            position_stats = {}
            for position in range(6):
                position_key = f"position_{position + 1}"
                position_stats[position_key] = {}

                for number in range(1, 46):
                    number_key = str(number)
                    # 해당 위치에서 특정 번호의 출현 확률
                    probability = float(
                        position_matrix[position, number - 1] / total_draws
                    )
                    position_stats[position_key][number_key] = probability

            # 특이한 출현 패턴 로깅
            for position in range(6):
                position_key = f"position_{position + 1}"
                # 가장 높은 확률을 가진 번호 찾기
                max_prob_number = max(
                    position_stats[position_key].items(), key=lambda x: x[1]
                )
                self.logger.info(
                    f"위치 {position + 1}에서 가장 자주 출현한 번호: {max_prob_number[0]} "
                    f"(확률: {max_prob_number[1]:.4f})"
                )

            return position_stats

    def calculate_segment_trend_history(
        self, historical_data: List[LotteryNumber], segments: int = 5
    ) -> np.ndarray:
        """
        세그먼트별 출현 추세 시계열을 생성합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            segments: 세그먼트 수 (기본값: 5)

        Returns:
            np.ndarray: 세그먼트별 추세 시계열 (segments x n_turns)
        """
        with self.performance_tracker.track("calculate_segment_trend_history"):
            n_turns = len(historical_data)
            segment_size = 45 // segments

            # 세그먼트별 추세 행렬 초기화
            trend_matrix = np.zeros((segments, n_turns))

            # 각 회차에 대해 세그먼트별 번호 출현 수 계산
            for turn_idx, draw in enumerate(historical_data):
                segment_counts = [0] * segments

                for number in draw.numbers:
                    # 각 번호가 속한 세그먼트 인덱스 계산 (0-indexed)
                    segment_idx = min((number - 1) // segment_size, segments - 1)
                    segment_counts[segment_idx] += 1

                # 해당 회차의 세그먼트별 출현 수 저장
                for seg_idx in range(segments):
                    trend_matrix[seg_idx, turn_idx] = segment_counts[seg_idx]

            # 결과 저장
            try:
                cache_dir = Path(self.config["paths"]["cache_dir"])
                cache_dir.mkdir(parents=True, exist_ok=True)

                file_path = cache_dir / "segment_history_matrix.npy"
                np.save(file_path, trend_matrix)

                # 추세 점수 계산 (최근 30회차의 기울기)
                if n_turns >= 30:
                    recent_trends = trend_matrix[:, -30:]
                    trend_scores = []

                    for seg_idx in range(segments):
                        # 간단한 선형 추세 계산 (기울기)
                        x = np.arange(30)
                        y = recent_trends[seg_idx]
                        slope, _ = np.polyfit(x, y, 1)
                        trend_scores.append(float(slope))

                    mean_trend = np.mean(trend_scores)
                    self.logger.info(f"세그먼트 평균 추세 점수: {mean_trend:.4f}")

                self.logger.info(
                    f"세그먼트 추세 행렬 저장 완료: {file_path} (형태: {trend_matrix.shape})"
                )
            except Exception as e:
                self.logger.error(f"세그먼트 추세 행렬 저장 중 오류 발생: {e}")

            return trend_matrix

    def calculate_gap_deviation_score(
        self, draw_data: List[LotteryNumber]
    ) -> Dict[int, float]:
        """
        각 번호의 간격 표준편차 점수를 계산합니다.

        Args:
            draw_data: 분석할 로또 번호 데이터

        Returns:
            Dict[int, float]: 번호별 간격 표준편차 점수
        """
        self.logger.info("간격 표준편차 점수 계산 중...")

        # 각 번호의 출현 회차 기록
        number_appearances = {num: [] for num in range(1, 46)}

        # 모든 회차에서 각 번호의 출현 여부 기록
        for idx, draw in enumerate(draw_data):
            for num in draw.numbers:
                number_appearances[num].append(idx)

        # 각 번호별 연속 출현 간격의 표준편차 계산
        gap_deviation_scores = {}

        for number, appearances in number_appearances.items():
            if len(appearances) >= 2:
                # 연속 출현 간격 계산
                gaps = [
                    appearances[i] - appearances[i - 1]
                    for i in range(1, len(appearances))
                ]

                # 간격의 표준편차
                if gaps:
                    std_dev = np.std(gaps)
                    mean_gap = np.mean(gaps)

                    # 변동 계수 (CV: Coefficient of Variation)
                    # 간격 크기의 상대적 변동성 측정
                    cv = std_dev / mean_gap if mean_gap > 0 else 0

                    # 정규화된 점수 (0~1 사이)
                    # 변동 계수가 낮을수록 일정한 간격으로 출현
                    normalized_score = 1.0 / (1.0 + cv) if cv > 0 else 1.0

                    gap_deviation_scores[number] = normalized_score
                else:
                    gap_deviation_scores[number] = 0.0
            else:
                # 출현 기록이 부족한 경우
                gap_deviation_scores[number] = 0.0

        self.logger.info(
            f"간격 표준편차 점수 계산 완료: {len(gap_deviation_scores)}개 번호"
        )
        return gap_deviation_scores

    def calculate_combination_diversity_score(
        self, draw_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        각 회차의 조합 다양성 점수를 계산합니다.

        자카드 유사도 기반으로 이전 회차들과의 유사도를 측정하여
        얼마나 다양한 조합인지 평가합니다.

        Args:
            draw_data: 분석할 로또 번호 데이터

        Returns:
            Dict[str, float]: 회차별 조합 다양성 점수
        """
        self.logger.info("조합 다양성 점수 계산 중...")

        diversity_scores = {}

        # 각 회차에 대해 다양성 점수 계산
        for i, current_draw in enumerate(draw_data):
            # 첫 회차는 비교 대상이 없으므로 건너뜀
            if i == 0:
                continue

            # 현재 회차 번호 세트
            current_set = set(current_draw.numbers)

            # 이전 회차와 비교
            similarities = []
            for j in range(max(0, i - 10), i):  # 최대 10개 이전 회차와 비교
                prev_draw = draw_data[j]
                prev_set = set(prev_draw.numbers)

                # 자카드 유사도 = 교집합 크기 / 합집합 크기
                intersection = len(current_set.intersection(prev_set))
                union = len(current_set.union(prev_set))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)

            # 평균 자카드 유사도 계산
            avg_similarity = (
                sum(similarities) / len(similarities) if similarities else 0
            )

            # 다양성 = 1 - 유사도 (유사도가 낮을수록 다양성이 높음)
            diversity = 1.0 - avg_similarity

            # 회차 번호를 키로 다양성 점수 저장
            draw_key = str(current_draw.draw_no)
            diversity_scores[draw_key] = float(diversity)

        return diversity_scores

    def _get_trend_features(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        TrendAnalyzer를 사용하여 트렌드 특성을 가져옵니다.
        위치별 트렌드 기울기, delta 통계, 세그먼트 반복 점수 등이 포함됩니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 트렌드 특성
        """
        from ..analysis.trend_analyzer import TrendAnalyzer

        try:
            # TrendAnalyzer 인스턴스 생성
            trend_analyzer = TrendAnalyzer(self.config)

            # 트렌드 분석 수행
            trend_analysis = trend_analyzer.analyze(historical_data)

            # 필요한 트렌드 특성 추출
            if "trend_features" in trend_analysis:
                trend_features = trend_analysis["trend_features"]
                self.logger.info("트렌드 특성 추출 완료")

                # 필수 키 확인 및 누락된 경우 경고 출력
                required_keys = [
                    "position_trend_slope_1",
                    "position_trend_slope_2",
                    "position_trend_slope_3",
                    "position_trend_slope_4",
                    "position_trend_slope_5",
                    "position_trend_slope_6",
                    "delta_mean",
                    "delta_std",
                    "segment_repeat_score",
                ]

                missing_keys = [
                    key for key in required_keys if key not in trend_features
                ]
                if missing_keys:
                    self.logger.warning(
                        f"트렌드 특성에서 다음 키가 누락되었습니다: {missing_keys}"
                    )

                return trend_features
            else:
                self.logger.warning(
                    "TrendAnalyzer에서 trend_features를 찾을 수 없습니다."
                )
                # 기본값 생성
                default_features = {
                    "position_trend_slope_1": 0.0,
                    "position_trend_slope_2": 0.0,
                    "position_trend_slope_3": 0.0,
                    "position_trend_slope_4": 0.0,
                    "position_trend_slope_5": 0.0,
                    "position_trend_slope_6": 0.0,
                    "delta_mean": 0.0,
                    "delta_std": 0.5,
                    "segment_repeat_score": 0.5,
                }
                return default_features

        except Exception as e:
            self.logger.error(f"트렌드 특성 추출 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본값 반환
            default_features = {
                "position_trend_slope_1": 0.0,
                "position_trend_slope_2": 0.0,
                "position_trend_slope_3": 0.0,
                "position_trend_slope_4": 0.0,
                "position_trend_slope_5": 0.0,
                "position_trend_slope_6": 0.0,
                "delta_mean": 0.0,
                "delta_std": 0.5,
                "segment_repeat_score": 0.5,
            }
            return default_features

    def _get_overlap_features(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        OverlapAnalyzer를 사용하여 중복/유사성 특성을 가져옵니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 중복/유사성 특성
        """
        from ..analysis.overlap_analyzer import OverlapAnalyzer

        try:
            # OverlapAnalyzer 인스턴스 생성
            overlap_analyzer = OverlapAnalyzer(self.config)

            # 중복/유사성 분석 수행
            overlap_analysis = overlap_analyzer.analyze(historical_data)

            # 필요한 특성 추출
            overlap_features = {}

            # 1. 정확한 일치 통계
            if "exact_match_count" in overlap_analysis:
                overlap_features["exact_match_count"] = overlap_analysis[
                    "exact_match_count"
                ]
                # 마지막 요소가 정확한 일치인지 확인 (가장 최근 회차)
                if (
                    "exact_match_in_history" in overlap_analysis
                    and overlap_analysis["exact_match_in_history"]
                ):
                    overlap_features["latest_is_exact_match"] = bool(
                        overlap_analysis["exact_match_in_history"][-1]
                    )
                else:
                    overlap_features["latest_is_exact_match"] = False

            # 2. 과거 최대 중복 번호 수 통계
            if "num_overlap_with_past_max" in overlap_analysis:
                max_overlap = overlap_analysis["num_overlap_with_past_max"]
                if isinstance(max_overlap, dict):
                    for key in ["average", "max", "min", "std"]:
                        if key in max_overlap:
                            overlap_features[f"max_overlap_{key}"] = max_overlap[key]

            # 3. 인기 패턴과의 중복 통계
            if "overlap_with_hot_patterns" in overlap_analysis:
                hot_patterns = overlap_analysis["overlap_with_hot_patterns"]
                if isinstance(hot_patterns, dict):
                    if "avg_count" in hot_patterns:
                        overlap_features["hot_overlap_avg"] = hot_patterns["avg_count"]
                    if "hot_numbers" in hot_patterns:
                        overlap_features["hot_numbers"] = hot_patterns["hot_numbers"]
                    if "hot_overlap_distribution" in hot_patterns:
                        overlap_features["hot_overlap_dist"] = hot_patterns[
                            "hot_overlap_distribution"
                        ]

            # 4. 인접 회차 유사도 통계
            if "adjacent_draw_similarity" in overlap_analysis:
                adj_sim = overlap_analysis["adjacent_draw_similarity"]
                if isinstance(adj_sim, dict):
                    for key in ["jaccard_avg", "cosine_avg", "overlap_avg"]:
                        if key in adj_sim:
                            overlap_features[key] = adj_sim[key]

            self.logger.info("중복/유사성 특성 추출 완료")
            return overlap_features

        except Exception as e:
            self.logger.error(f"중복/유사성 특성 추출 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본값 반환
            return {
                "exact_match_count": 0,
                "latest_is_exact_match": False,
                "max_overlap_average": 0.0,
                "max_overlap_max": 0,
                "hot_overlap_avg": 0.0,
            }

    def _get_roi_features(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        ROIAnalyzer를 사용하여 ROI 특성을 가져옵니다.
        ROI 그룹 점수, 클러스터 점수, 저위험 보너스 플래그 등이 포함됩니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: ROI 특성
        """
        from ..analysis.roi_analyzer import ROIAnalyzer

        try:
            # ROIAnalyzer 인스턴스 생성
            roi_analyzer = ROIAnalyzer(self.config)

            # ROI 분석 수행
            roi_analysis = roi_analyzer.analyze(historical_data)

            # 필요한 ROI 특성 추출
            roi_features = {}

            # 1. ROI 그룹 점수
            if "roi_group_score" in roi_analysis:
                roi_features["roi_group_score"] = roi_analysis["roi_group_score"]

            # 2. ROI 클러스터 점수
            if "roi_cluster_score" in roi_analysis:
                roi_features["roi_cluster_assignments"] = roi_analysis[
                    "roi_cluster_score"
                ].get("cluster_assignments", {})
                roi_features["roi_cluster_averages"] = roi_analysis[
                    "roi_cluster_score"
                ].get("cluster_averages", {})

            # 3. 저위험 보너스 플래그
            if "low_risk_bonus_flag" in roi_analysis:
                roi_features["low_risk_bonus_flag"] = roi_analysis[
                    "low_risk_bonus_flag"
                ].get("low_risk_bonus_flag", {})
                roi_features["low_risk_count"] = roi_analysis[
                    "low_risk_bonus_flag"
                ].get("low_risk_count", 0)

            # 4. ROI 패턴 그룹 ID
            if "roi_pattern_group_id" in roi_analysis:
                roi_features["roi_pattern_group_id"] = roi_analysis[
                    "roi_pattern_group_id"
                ]

            self.logger.info("ROI 특성 추출 완료")
            return roi_features

        except Exception as e:
            self.logger.error(f"ROI 특성 추출 중 오류 발생: {str(e)}")
            return {}
