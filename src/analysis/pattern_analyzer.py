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

        # 가중치 빈도 계산
        weighted_frequencies = self._calculate_weighted_frequencies(historical_data)

        # 최근성 맵 계산
        recency_map = self._calculate_recency_map(historical_data)

        result = PatternAnalysis(
            frequency_map=weighted_frequencies, recency_map=recency_map
        )
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

        # 번호 간 조건부 상호작용 분석
        conditional_interaction_features = (
            self._analyze_number_conditional_relationships(historical_data)
        )

        # 홀짝 미세 편향성 분석
        odd_even_micro_bias = self._detect_odd_even_micro_bias(historical_data)

        # 구간별 미세 편향성 분석
        range_micro_bias = self._detect_range_micro_bias(historical_data)

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
        self.pattern_stats["conditional_interaction_features"] = (
            conditional_interaction_features
        )
        self.pattern_stats["odd_even_micro_bias"] = odd_even_micro_bias
        self.pattern_stats["range_micro_bias"] = range_micro_bias
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
        result.metadata["conditional_interaction_features"] = (
            conditional_interaction_features
        )
        result.metadata["odd_even_micro_bias"] = odd_even_micro_bias
        result.metadata["range_micro_bias"] = range_micro_bias
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
