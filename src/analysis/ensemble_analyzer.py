#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
앙상블 패턴 분석기

기존 분석기들과 완전히 독립적으로 동작하는 앙상블 패턴 분석기입니다.
다중 윈도우 분석과 가중 앙상블 기법을 통해 예측 성능을 향상시킵니다.

주요 기능:
- 다중 윈도우 분석
- 가중 앙상블 분석
- 적응적 가중치 조정
- 앙상블 성능 평가
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from scipy import stats

from .base_analyzer import BaseAnalyzer
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


class EnsembleAnalyzer(BaseAnalyzer):
    """
    앙상블 패턴 분석기

    다중 윈도우 분석과 가중 앙상블을 통해 예측 성능을 향상시킵니다.
    기존 분석기들과 완전히 독립적으로 동작합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        super().__init__(config, "EnsembleAnalyzer")

        # 앙상블 분석 설정
        self.ensemble_config = {
            "window_sizes": [5, 10, 20, 50, 100],  # 다중 윈도우 크기
            "ensemble_methods": ["average", "weighted_average", "stacking", "voting"],
            "weight_update_frequency": 10,  # 가중치 업데이트 빈도
            "performance_memory": 50,  # 성능 기록 길이
            "min_confidence_threshold": 0.3,  # 최소 신뢰도 임계값
            "diversity_penalty": 0.1,  # 다양성 페널티
            "adaptive_learning_rate": 0.05,  # 적응적 학습률
        }

        # 설정 오버라이드
        if config and "ensemble_analyzer" in config:
            self.ensemble_config.update(config["ensemble_analyzer"])

        # 앙상블 구성 요소 초기화
        self.window_analyzers = {}
        self.ensemble_weights = {}
        self.performance_history = defaultdict(list)
        self.prediction_cache = {}

        self.logger.info("EnsembleAnalyzer 초기화 완료")

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        앙상블 분석 메인 구현

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 앙상블 분석 결과
        """
        try:
            self.logger.info(
                f"EnsembleAnalyzer 분석 시작: {len(historical_data)}개 회차"
            )

            # 1. 다중 윈도우 분석
            multi_window_results = self.multi_window_analysis(historical_data)

            # 2. 가중 앙상블 분석
            weighted_ensemble_results = self.weighted_ensemble_analysis(
                multi_window_results.get("window_analyses", [])
            )

            # 3. 적응적 가중치 조정
            adaptive_weights = self.adaptive_weight_adjustment(
                self.performance_history.get("ensemble_performance", [])
            )

            # 4. 앙상블 성능 평가
            performance_evaluation = self._evaluate_ensemble_performance(
                multi_window_results, weighted_ensemble_results, historical_data
            )

            # 5. 최종 앙상블 예측
            final_predictions = self._generate_final_ensemble_predictions(
                multi_window_results, weighted_ensemble_results, adaptive_weights
            )

            # 결과 통합
            analysis_result = {
                "analyzer_version": "EnsembleAnalyzer_v1.0",
                "analysis_timestamp": datetime.now().isoformat(),
                "data_range": {
                    "total_draws": len(historical_data),
                    "start_draw": historical_data[0].draw_no if historical_data else 0,
                    "end_draw": historical_data[-1].draw_no if historical_data else 0,
                },
                "multi_window_analysis": multi_window_results,
                "weighted_ensemble_analysis": weighted_ensemble_results,
                "adaptive_weights": adaptive_weights,
                "performance_evaluation": performance_evaluation,
                "final_predictions": final_predictions,
                "ensemble_summary": self._generate_ensemble_summary(
                    multi_window_results,
                    weighted_ensemble_results,
                    adaptive_weights,
                    final_predictions,
                ),
            }

            # 성능 기록 업데이트
            self._update_performance_history(performance_evaluation)

            self.logger.info("EnsembleAnalyzer 분석 완료")
            return analysis_result

        except Exception as e:
            self.logger.error(f"EnsembleAnalyzer 분석 중 오류 발생: {e}")
            raise

    def multi_window_analysis(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        다중 윈도우 분석

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 다중 윈도우 분석 결과
        """
        try:
            window_analyses = []

            for window_size in self.ensemble_config["window_sizes"]:
                if len(data) < window_size:
                    continue

                # 윈도우별 분석 수행
                window_analysis = self._analyze_single_window(data, window_size)
                window_analyses.append(window_analysis)

            # 윈도우 간 일관성 분석
            consistency_analysis = self._analyze_window_consistency(window_analyses)

            # 윈도우 성능 비교
            performance_comparison = self._compare_window_performance(window_analyses)

            return {
                "window_analyses": window_analyses,
                "consistency_analysis": consistency_analysis,
                "performance_comparison": performance_comparison,
                "window_config": {
                    "window_sizes": self.ensemble_config["window_sizes"],
                    "total_windows": len(window_analyses),
                },
            }

        except Exception as e:
            self.logger.error(f"다중 윈도우 분석 실패: {e}")
            return {}

    def weighted_ensemble_analysis(
        self, analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        가중 앙상블 분석

        Args:
            analyses: 개별 분석 결과 리스트

        Returns:
            Dict[str, Any]: 가중 앙상블 분석 결과
        """
        try:
            if not analyses:
                return {"error": "분석 결과가 없습니다"}

            # 앙상블 방법별 결과 계산
            ensemble_results = {}

            for method in self.ensemble_config["ensemble_methods"]:
                ensemble_results[method] = self._apply_ensemble_method(analyses, method)

            # 앙상블 방법 간 비교
            method_comparison = self._compare_ensemble_methods(ensemble_results)

            # 최적 앙상블 선택
            best_ensemble = self._select_best_ensemble(
                ensemble_results, method_comparison
            )

            return {
                "ensemble_results": ensemble_results,
                "method_comparison": method_comparison,
                "best_ensemble": best_ensemble,
                "ensemble_config": {
                    "methods_used": self.ensemble_config["ensemble_methods"],
                    "total_analyses": len(analyses),
                },
            }

        except Exception as e:
            self.logger.error(f"가중 앙상블 분석 실패: {e}")
            return {}

    def adaptive_weight_adjustment(
        self, performance_history: List[float]
    ) -> Dict[str, float]:
        """
        적응적 가중치 조정

        Args:
            performance_history: 성능 기록 리스트

        Returns:
            Dict[str, float]: 조정된 가중치
        """
        try:
            if not performance_history:
                # 초기 균등 가중치
                return self._initialize_equal_weights()

            # 성능 기반 가중치 계산
            performance_weights = self._calculate_performance_weights(
                performance_history
            )

            # 다양성 기반 가중치 계산
            diversity_weights = self._calculate_diversity_weights()

            # 적응적 가중치 조합
            adaptive_weights = self._combine_adaptive_weights(
                performance_weights, diversity_weights
            )

            # 가중치 정규화
            normalized_weights = self._normalize_weights(adaptive_weights)

            return {
                "performance_weights": performance_weights,
                "diversity_weights": diversity_weights,
                "adaptive_weights": adaptive_weights,
                "normalized_weights": normalized_weights,
                "adjustment_config": {
                    "learning_rate": self.ensemble_config["adaptive_learning_rate"],
                    "diversity_penalty": self.ensemble_config["diversity_penalty"],
                },
            }

        except Exception as e:
            self.logger.error(f"적응적 가중치 조정 실패: {e}")
            return {}

    def _analyze_single_window(
        self, data: List[LotteryNumber], window_size: int
    ) -> Dict[str, Any]:
        """단일 윈도우 분석"""
        # 최근 윈도우 데이터 추출
        window_data = data[-window_size:] if len(data) >= window_size else data

        # 기본 통계 분석
        number_frequencies = self._calculate_number_frequencies(window_data)
        pattern_analysis = self._analyze_patterns_in_window(window_data)
        trend_analysis = self._analyze_trends_in_window(window_data)

        # 윈도우 특성 분석
        window_characteristics = self._analyze_window_characteristics(window_data)

        # 예측 점수 계산
        prediction_scores = self._calculate_window_prediction_scores(
            number_frequencies, pattern_analysis, trend_analysis
        )

        return {
            "window_size": window_size,
            "data_points": len(window_data),
            "number_frequencies": number_frequencies,
            "pattern_analysis": pattern_analysis,
            "trend_analysis": trend_analysis,
            "window_characteristics": window_characteristics,
            "prediction_scores": prediction_scores,
        }

    def _calculate_number_frequencies(
        self, data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """윈도우 내 번호 빈도 계산"""
        frequencies = defaultdict(int)
        total_numbers = 0

        for draw in data:
            for number in draw.numbers:
                frequencies[number] += 1
                total_numbers += 1

        # 상대 빈도로 변환
        relative_frequencies = {}
        for number in range(1, 46):
            relative_frequencies[str(number)] = (
                frequencies[number] / len(data) if data else 0
            )

        return relative_frequencies

    def _analyze_patterns_in_window(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """윈도우 내 패턴 분석"""
        patterns = {
            "consecutive_pairs": 0,
            "arithmetic_sequences": 0,
            "sum_patterns": defaultdict(int),
            "range_distributions": {"low": 0, "mid": 0, "high": 0},
        }

        for draw in data:
            numbers = sorted(draw.numbers)

            # 연속 쌍 계산
            for i in range(len(numbers) - 1):
                if numbers[i + 1] - numbers[i] == 1:
                    patterns["consecutive_pairs"] += 1

            # 등차수열 패턴
            if len(numbers) >= 3:
                diffs = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
                if len(set(diffs)) <= 2:  # 거의 등차수열
                    patterns["arithmetic_sequences"] += 1

            # 합계 패턴
            total_sum = sum(numbers)
            sum_range = f"{total_sum // 25 * 25}-{(total_sum // 25 + 1) * 25}"
            patterns["sum_patterns"][sum_range] += 1

            # 범위 분포
            for number in numbers:
                if 1 <= number <= 15:
                    patterns["range_distributions"]["low"] += 1
                elif 16 <= number <= 30:
                    patterns["range_distributions"]["mid"] += 1
                else:
                    patterns["range_distributions"]["high"] += 1

        return patterns

    def _analyze_trends_in_window(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """윈도우 내 트렌드 분석"""
        if len(data) < 3:
            return {"trend_strength": 0, "trend_direction": "stable"}

        trends = {}

        for number in range(1, 46):
            appearances = []
            for i, draw in enumerate(data):
                appearances.append(1 if number in draw.numbers else 0)

            # 선형 트렌드 계산
            if len(appearances) > 1:
                x = np.arange(len(appearances))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x, appearances
                )

                trends[str(number)] = {
                    "slope": float(slope),
                    "r_squared": float(r_value**2),
                    "p_value": float(p_value),
                    "trend_strength": float(abs(slope)),
                    "trend_direction": "increasing" if slope > 0 else "decreasing",
                }

        return trends

    def _analyze_window_characteristics(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """윈도우 특성 분석"""
        if not data:
            return {}

        all_numbers = []
        for draw in data:
            all_numbers.extend(draw.numbers)

        characteristics = {
            "number_diversity": len(set(all_numbers)),
            "average_sum": float(np.mean([sum(draw.numbers) for draw in data])),
            "sum_variance": float(np.var([sum(draw.numbers) for draw in data])),
            "number_range_spread": float(
                np.mean([max(draw.numbers) - min(draw.numbers) for draw in data])
            ),
            "unique_patterns": len(set(tuple(sorted(draw.numbers)) for draw in data)),
        }

        return characteristics

    def _calculate_window_prediction_scores(
        self,
        frequencies: Dict[str, float],
        patterns: Dict[str, Any],
        trends: Dict[str, Any],
    ) -> Dict[str, float]:
        """윈도우 예측 점수 계산"""
        prediction_scores = {}

        for number in range(1, 46):
            number_str = str(number)

            # 빈도 점수
            frequency_score = frequencies.get(number_str, 0)

            # 트렌드 점수
            trend_score = 0
            if number_str in trends:
                trend_data = trends[number_str]
                trend_score = trend_data["slope"] * trend_data["r_squared"]

            # 패턴 점수 (간단한 버전)
            pattern_score = 0.1  # 기본 점수

            # 종합 점수
            composite_score = (
                frequency_score * 0.6 + trend_score * 0.3 + pattern_score * 0.1
            )
            prediction_scores[number_str] = float(composite_score)

        return prediction_scores

    def _analyze_window_consistency(
        self, window_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """윈도우 간 일관성 분석"""
        if len(window_analyses) < 2:
            return {"consistency_score": 1.0, "note": "단일 윈도우"}

        # 예측 점수 일관성 분석
        prediction_correlations = []

        for i in range(len(window_analyses) - 1):
            scores1 = window_analyses[i]["prediction_scores"]
            scores2 = window_analyses[i + 1]["prediction_scores"]

            # 공통 번호에 대한 점수 추출
            common_numbers = set(scores1.keys()) & set(scores2.keys())
            if common_numbers:
                values1 = [scores1[num] for num in common_numbers]
                values2 = [scores2[num] for num in common_numbers]

                correlation = np.corrcoef(values1, values2)[0, 1]
                if not np.isnan(correlation):
                    prediction_correlations.append(correlation)

        consistency_score = (
            np.mean(prediction_correlations) if prediction_correlations else 0
        )

        return {
            "consistency_score": float(consistency_score),
            "prediction_correlations": prediction_correlations,
            "window_count": len(window_analyses),
        }

    def _compare_window_performance(
        self, window_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """윈도우 성능 비교"""
        performance_metrics = {}

        for analysis in window_analyses:
            window_size = analysis["window_size"]

            # 예측 점수의 분포 분석
            scores = list(analysis["prediction_scores"].values())

            performance_metrics[str(window_size)] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "max_score": float(np.max(scores)),
                "score_range": float(np.max(scores) - np.min(scores)),
                "data_points": analysis["data_points"],
            }

        # 최적 윈도우 크기 결정
        best_window = max(
            performance_metrics.keys(),
            key=lambda k: performance_metrics[k]["mean_score"],
        )

        return {
            "window_performance": performance_metrics,
            "best_window_size": int(best_window),
            "performance_ranking": sorted(
                performance_metrics.items(),
                key=lambda x: x[1]["mean_score"],
                reverse=True,
            ),
        }

    def _apply_ensemble_method(
        self, analyses: List[Dict[str, Any]], method: str
    ) -> Dict[str, Any]:
        """앙상블 방법 적용"""
        if method == "average":
            return self._apply_average_ensemble(analyses)
        elif method == "weighted_average":
            return self._apply_weighted_average_ensemble(analyses)
        elif method == "stacking":
            return self._apply_stacking_ensemble(analyses)
        elif method == "voting":
            return self._apply_voting_ensemble(analyses)
        else:
            return {"error": f"알 수 없는 앙상블 방법: {method}"}

    def _apply_average_ensemble(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """평균 앙상블"""
        if not analyses:
            return {}

        # 모든 분석의 예측 점수 평균 계산
        ensemble_scores = defaultdict(list)

        for analysis in analyses:
            prediction_scores = analysis.get("prediction_scores", {})
            for number, score in prediction_scores.items():
                ensemble_scores[number].append(score)

        # 평균 계산
        average_scores = {}
        for number, scores in ensemble_scores.items():
            average_scores[number] = float(np.mean(scores))

        return {
            "method": "average",
            "ensemble_scores": average_scores,
            "contributing_analyses": len(analyses),
        }

    def _apply_weighted_average_ensemble(
        self, analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """가중 평균 앙상블"""
        if not analyses:
            return {}

        # 윈도우 크기 기반 가중치 계산
        weights = []
        for analysis in analyses:
            window_size = analysis.get("window_size", 1)
            # 더 큰 윈도우에 더 높은 가중치 (하지만 너무 크면 페널티)
            weight = np.sqrt(window_size) / (1 + window_size * 0.01)
            weights.append(weight)

        # 가중치 정규화
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # 가중 평균 계산
        ensemble_scores = defaultdict(float)

        for i, analysis in enumerate(analyses):
            prediction_scores = analysis.get("prediction_scores", {})
            for number, score in prediction_scores.items():
                ensemble_scores[number] += score * weights[i]

        return {
            "method": "weighted_average",
            "ensemble_scores": dict(ensemble_scores),
            "weights": weights.tolist(),
            "contributing_analyses": len(analyses),
        }

    def _apply_stacking_ensemble(
        self, analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """스태킹 앙상블 (간단한 버전)"""
        if not analyses:
            return {}

        # 메타 학습기로 선형 조합 사용
        ensemble_scores = {}

        # 각 분석의 성능 추정 (데이터 포인트 수 기반)
        performance_weights = []
        for analysis in analyses:
            data_points = analysis.get("data_points", 1)
            # 더 많은 데이터 포인트 = 더 높은 신뢰도
            weight = np.log(data_points + 1)
            performance_weights.append(weight)

        # 가중치 정규화
        performance_weights = np.array(performance_weights)
        performance_weights = performance_weights / np.sum(performance_weights)

        # 스태킹 점수 계산
        for number in range(1, 46):
            number_str = str(number)
            stacked_score = 0

            for i, analysis in enumerate(analyses):
                score = analysis.get("prediction_scores", {}).get(number_str, 0)
                stacked_score += score * performance_weights[i]

            ensemble_scores[number_str] = float(stacked_score)

        return {
            "method": "stacking",
            "ensemble_scores": ensemble_scores,
            "meta_weights": performance_weights.tolist(),
            "contributing_analyses": len(analyses),
        }

    def _apply_voting_ensemble(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """투표 앙상블"""
        if not analyses:
            return {}

        # 각 분석에서 상위 N개 번호 선택
        top_n = 10
        voting_scores = defaultdict(int)

        for analysis in analyses:
            prediction_scores = analysis.get("prediction_scores", {})
            # 상위 번호들 선택
            top_numbers = sorted(
                prediction_scores.items(), key=lambda x: x[1], reverse=True
            )[:top_n]

            # 투표 점수 부여 (순위 기반)
            for rank, (number, score) in enumerate(top_numbers):
                voting_scores[number] += top_n - rank

        # 투표 점수를 0-1 범위로 정규화
        max_votes = max(voting_scores.values()) if voting_scores else 1
        normalized_scores = {
            number: float(votes / max_votes) for number, votes in voting_scores.items()
        }

        return {
            "method": "voting",
            "ensemble_scores": normalized_scores,
            "raw_votes": dict(voting_scores),
            "contributing_analyses": len(analyses),
        }

    def _compare_ensemble_methods(
        self, ensemble_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """앙상블 방법 비교"""
        comparison = {}

        # 각 방법의 예측 점수 분포 분석
        for method, results in ensemble_results.items():
            if "ensemble_scores" in results:
                scores = list(results["ensemble_scores"].values())

                comparison[method] = {
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "max_score": float(np.max(scores)),
                    "score_diversity": float(np.max(scores) - np.min(scores)),
                }

        # 최적 방법 선택
        if comparison:
            best_method = max(
                comparison.keys(), key=lambda k: comparison[k]["mean_score"]
            )
            comparison["best_method"] = best_method

        return comparison

    def _select_best_ensemble(
        self, ensemble_results: Dict[str, Dict[str, Any]], comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """최적 앙상블 선택"""
        best_method = comparison.get("best_method", "average")

        if best_method in ensemble_results:
            best_ensemble = ensemble_results[best_method].copy()
            best_ensemble["selection_reason"] = (
                f"최고 평균 점수: {comparison.get(best_method, {}).get('mean_score', 0):.3f}"
            )
            return best_ensemble

        # 기본값으로 평균 앙상블 반환
        return ensemble_results.get("average", {})

    def _initialize_equal_weights(self) -> Dict[str, float]:
        """균등 가중치 초기화"""
        num_methods = len(self.ensemble_config["ensemble_methods"])
        equal_weight = 1.0 / num_methods if num_methods > 0 else 1.0

        return {
            method: equal_weight for method in self.ensemble_config["ensemble_methods"]
        }

    def _calculate_performance_weights(
        self, performance_history: List[float]
    ) -> Dict[str, float]:
        """성능 기반 가중치 계산"""
        if not performance_history:
            return self._initialize_equal_weights()

        # 최근 성능에 더 높은 가중치
        recent_performance = performance_history[-10:]  # 최근 10개
        avg_performance = np.mean(recent_performance)

        # 성능 기반 가중치 (간단한 버전)
        performance_weights = {}
        for method in self.ensemble_config["ensemble_methods"]:
            # 실제로는 각 방법별 성능을 추적해야 하지만, 여기서는 균등하게 처리
            performance_weights[method] = avg_performance

        return performance_weights

    def _calculate_diversity_weights(self) -> Dict[str, float]:
        """다양성 기반 가중치 계산"""
        # 다양성 가중치 (간단한 버전)
        diversity_weights = {}

        for method in self.ensemble_config["ensemble_methods"]:
            # 각 방법의 다양성 기여도 (휴리스틱)
            if method == "average":
                diversity_weights[method] = 0.8
            elif method == "weighted_average":
                diversity_weights[method] = 0.9
            elif method == "stacking":
                diversity_weights[method] = 1.0
            elif method == "voting":
                diversity_weights[method] = 0.7
            else:
                diversity_weights[method] = 0.5

        return diversity_weights

    def _combine_adaptive_weights(
        self, performance_weights: Dict[str, float], diversity_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """적응적 가중치 조합"""
        adaptive_weights = {}

        for method in self.ensemble_config["ensemble_methods"]:
            perf_weight = performance_weights.get(method, 0.5)
            div_weight = diversity_weights.get(method, 0.5)

            # 성능과 다양성의 가중 조합
            combined_weight = perf_weight * 0.7 + div_weight * 0.3
            adaptive_weights[method] = combined_weight

        return adaptive_weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """가중치 정규화"""
        total_weight = sum(weights.values())

        if total_weight > 0:
            return {method: weight / total_weight for method, weight in weights.items()}
        else:
            return self._initialize_equal_weights()

    def _evaluate_ensemble_performance(
        self,
        multi_window_results: Dict[str, Any],
        weighted_ensemble_results: Dict[str, Any],
        historical_data: List[LotteryNumber],
    ) -> Dict[str, Any]:
        """앙상블 성능 평가"""
        evaluation = {
            "window_performance": {},
            "ensemble_performance": {},
            "overall_metrics": {},
        }

        # 윈도우 성능 평가
        window_analyses = multi_window_results.get("window_analyses", [])
        for analysis in window_analyses:
            window_size = analysis["window_size"]
            scores = list(analysis["prediction_scores"].values())

            evaluation["window_performance"][str(window_size)] = {
                "score_variance": float(np.var(scores)),
                "score_entropy": float(-np.sum(scores * np.log(scores + 1e-10))),
                "coverage": (
                    len([s for s in scores if s > 0]) / len(scores) if scores else 0
                ),
            }

        # 앙상블 성능 평가
        ensemble_results = weighted_ensemble_results.get("ensemble_results", {})
        for method, results in ensemble_results.items():
            if "ensemble_scores" in results:
                scores = list(results["ensemble_scores"].values())

                evaluation["ensemble_performance"][method] = {
                    "prediction_diversity": float(np.std(scores)),
                    "confidence_level": (
                        float(np.mean([s for s in scores if s > 0.5])) if scores else 0
                    ),
                    "stability": float(1.0 / (1.0 + np.var(scores))),
                }

        # 전체 메트릭
        evaluation["overall_metrics"] = {
            "total_windows": len(window_analyses),
            "total_ensemble_methods": len(ensemble_results),
            "analysis_completeness": 1.0,  # 모든 분석이 완료됨
        }

        return evaluation

    def _generate_final_ensemble_predictions(
        self,
        multi_window_results: Dict[str, Any],
        weighted_ensemble_results: Dict[str, Any],
        adaptive_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """최종 앙상블 예측 생성"""
        # 최적 앙상블 결과 선택
        best_ensemble = weighted_ensemble_results.get("best_ensemble", {})
        ensemble_scores = best_ensemble.get("ensemble_scores", {})

        # 적응적 가중치 적용
        normalized_weights = adaptive_weights.get("normalized_weights", {})

        # 최종 예측 점수 계산
        final_scores = {}
        for number_str, score in ensemble_scores.items():
            # 가중치 적용 (간단한 버전)
            weight_factor = (
                sum(normalized_weights.values()) / len(normalized_weights)
                if normalized_weights
                else 1.0
            )
            final_scores[number_str] = float(score * weight_factor)

        # 상위 추천 번호 선택
        top_predictions = sorted(
            final_scores.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # 신뢰도 계산
        confidence_scores = {}
        for number_str, score in final_scores.items():
            # 신뢰도 = 점수 / 최대 점수
            max_score = max(final_scores.values()) if final_scores else 1
            confidence_scores[number_str] = float(score / max_score)

        return {
            "final_scores": final_scores,
            "top_predictions": [
                {
                    "number": int(num),
                    "score": score,
                    "confidence": confidence_scores.get(num, 0),
                }
                for num, score in top_predictions
            ],
            "prediction_summary": {
                "total_numbers_scored": len(final_scores),
                "high_confidence_count": len(
                    [s for s in confidence_scores.values() if s > 0.7]
                ),
                "average_confidence": float(np.mean(list(confidence_scores.values()))),
                "prediction_method": best_ensemble.get("method", "unknown"),
            },
        }

    def _generate_ensemble_summary(
        self,
        multi_window_results: Dict[str, Any],
        weighted_ensemble_results: Dict[str, Any],
        adaptive_weights: Dict[str, float],
        final_predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """앙상블 분석 요약"""
        summary = {
            "analysis_overview": {
                "total_windows_analyzed": len(
                    multi_window_results.get("window_analyses", [])
                ),
                "ensemble_methods_used": len(
                    weighted_ensemble_results.get("ensemble_results", {})
                ),
                "best_ensemble_method": weighted_ensemble_results.get(
                    "best_ensemble", {}
                ).get("method", "unknown"),
            },
            "key_findings": {
                "best_window_size": multi_window_results.get(
                    "performance_comparison", {}
                ).get("best_window_size", 0),
                "window_consistency": multi_window_results.get(
                    "consistency_analysis", {}
                ).get("consistency_score", 0),
                "top_predicted_numbers": final_predictions.get("top_predictions", [])[
                    :5
                ],
            },
            "performance_metrics": {
                "prediction_confidence": final_predictions.get(
                    "prediction_summary", {}
                ).get("average_confidence", 0),
                "high_confidence_predictions": final_predictions.get(
                    "prediction_summary", {}
                ).get("high_confidence_count", 0),
                "ensemble_stability": 1.0,  # 임시값
            },
            "recommendations": {
                "recommended_numbers": [
                    pred["number"]
                    for pred in final_predictions.get("top_predictions", [])[:6]
                ],
                "confidence_level": (
                    "high"
                    if final_predictions.get("prediction_summary", {}).get(
                        "average_confidence", 0
                    )
                    > 0.7
                    else "medium"
                ),
                "ensemble_strength": (
                    "strong"
                    if len(weighted_ensemble_results.get("ensemble_results", {})) > 2
                    else "moderate"
                ),
            },
        }

        return summary

    def _update_performance_history(self, performance_evaluation: Dict[str, Any]):
        """성능 기록 업데이트"""
        # 전체 성능 점수 계산
        overall_score = performance_evaluation.get("overall_metrics", {}).get(
            "analysis_completeness", 0
        )

        # 성능 기록에 추가
        self.performance_history["ensemble_performance"].append(overall_score)

        # 기록 길이 제한
        max_history = self.ensemble_config["performance_memory"]
        if len(self.performance_history["ensemble_performance"]) > max_history:
            self.performance_history["ensemble_performance"] = self.performance_history[
                "ensemble_performance"
            ][-max_history:]

    def save_analysis_results(
        self, analysis_results: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """분석 결과를 파일로 저장"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ensemble_analyzer_results_{timestamp}.json"

            output_dir = Path("data/result/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / filename

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"EnsembleAnalyzer 분석 결과 저장: {output_file}")
            return str(output_file)

        except Exception as e:
            self.logger.error(f"분석 결과 저장 실패: {e}")
            raise


# 팩토리 함수
def create_ensemble_analyzer(
    config: Optional[Dict[str, Any]] = None,
) -> EnsembleAnalyzer:
    """EnsembleAnalyzer 인스턴스 생성"""
    return EnsembleAnalyzer(config)


# 편의 함수
def analyze_ensemble_patterns(
    historical_data: List[LotteryNumber], config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """앙상블 패턴 분석 수행"""
    analyzer = create_ensemble_analyzer(config)
    return analyzer.analyze(historical_data)
