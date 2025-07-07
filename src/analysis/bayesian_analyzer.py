#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
베이지안 확률 분석기

기존 분석기들과 완전히 독립적으로 동작하는 베이지안 확률 분석기입니다.
베이지안 추론을 통해 번호 출현 확률을 분석하고 조건부 확률을 계산합니다.

주요 기능:
- 베이지안 확률 추론
- 조건부 확률 분석
- 사후 확률 업데이트
- 확률 분포 모델링
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import json
from pathlib import Path
from scipy import stats
from scipy.special import beta, gamma

from .base_analyzer import BaseAnalyzer
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


class BayesianAnalyzer(BaseAnalyzer):
    """
    베이지안 확률 분석기

    베이지안 추론을 통한 번호 출현 확률 분석 및 조건부 확률 계산에 특화됩니다.
    기존 분석기들과 완전히 독립적으로 동작합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        super().__init__(config, "BayesianAnalyzer")

        # 베이지안 분석 설정
        self.bayesian_config = {
            "prior_alpha": 1.0,  # 베타 분포의 알파 매개변수
            "prior_beta": 1.0,  # 베타 분포의 베타 매개변수
            "confidence_level": 0.95,  # 신뢰도 수준
            "mcmc_samples": 1000,  # MCMC 샘플 수
            "burn_in": 100,  # 번인 기간
            "condition_window": 20,  # 조건부 확률 계산 윈도우
            "update_threshold": 0.01,  # 업데이트 임계값
        }

        # 설정 오버라이드
        if config and "bayesian_analyzer" in config:
            self.bayesian_config.update(config["bayesian_analyzer"])

        # 사전 확률 분포 초기화
        self.prior_distributions = self._initialize_prior_distributions()

        # 사후 확률 분포 (업데이트됨)
        self.posterior_distributions = {}

        self.logger.info("BayesianAnalyzer 초기화 완료")

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        베이지안 확률 분석 메인 구현

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 베이지안 분석 결과
        """
        try:
            self.logger.info(
                f"BayesianAnalyzer 분석 시작: {len(historical_data)}개 회차"
            )

            # 1. 베이지안 확률 추론
            probability_inference = self.bayesian_probability_inference(historical_data)

            # 2. 조건부 확률 분석
            conditional_analysis = self.conditional_probability_analysis(
                historical_data
            )

            # 3. 사후 확률 업데이트
            posterior_updates = self.update_posterior_probabilities(historical_data)

            # 4. 확률 분포 모델링
            distribution_modeling = self._model_probability_distributions(
                historical_data
            )

            # 5. 베이지안 예측
            bayesian_predictions = self._generate_bayesian_predictions(
                probability_inference, conditional_analysis, posterior_updates
            )

            # 결과 통합
            analysis_result = {
                "analyzer_version": "BayesianAnalyzer_v1.0",
                "analysis_timestamp": datetime.now().isoformat(),
                "data_range": {
                    "total_draws": len(historical_data),
                    "start_draw": historical_data[0].draw_no if historical_data else 0,
                    "end_draw": historical_data[-1].draw_no if historical_data else 0,
                },
                "bayesian_inference": probability_inference,
                "conditional_analysis": conditional_analysis,
                "posterior_updates": posterior_updates,
                "distribution_modeling": distribution_modeling,
                "bayesian_predictions": bayesian_predictions,
                "analysis_summary": self._generate_bayesian_summary(
                    probability_inference,
                    conditional_analysis,
                    posterior_updates,
                    bayesian_predictions,
                ),
            }

            self.logger.info("BayesianAnalyzer 분석 완료")
            return analysis_result

        except Exception as e:
            self.logger.error(f"BayesianAnalyzer 분석 중 오류 발생: {e}")
            raise

    def bayesian_probability_inference(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        베이지안 확률 추론

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 베이지안 확률 추론 결과
        """
        try:
            # 번호별 출현 횟수 계산
            number_counts = self._count_number_occurrences(data)

            # 베이지안 추론 수행
            bayesian_probabilities = {}
            credible_intervals = {}

            for number in range(1, 46):
                # 관찰된 데이터
                successes = number_counts.get(number, 0)
                trials = len(data)

                # 베타 분포 매개변수 업데이트
                alpha_post = self.bayesian_config["prior_alpha"] + successes
                beta_post = self.bayesian_config["prior_beta"] + trials - successes

                # 사후 확률 계산
                posterior_mean = alpha_post / (alpha_post + beta_post)
                posterior_var = (alpha_post * beta_post) / (
                    (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)
                )

                # 신뢰구간 계산
                confidence = self.bayesian_config["confidence_level"]
                lower_bound = stats.beta.ppf(
                    (1 - confidence) / 2, alpha_post, beta_post
                )
                upper_bound = stats.beta.ppf(
                    (1 + confidence) / 2, alpha_post, beta_post
                )

                bayesian_probabilities[str(number)] = {
                    "posterior_mean": float(posterior_mean),
                    "posterior_variance": float(posterior_var),
                    "alpha_posterior": float(alpha_post),
                    "beta_posterior": float(beta_post),
                    "observed_frequency": (
                        float(successes / trials) if trials > 0 else 0.0
                    ),
                }

                credible_intervals[str(number)] = {
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "confidence_level": confidence,
                }

            # 베이지안 모델 비교
            model_comparison = self._compare_bayesian_models(
                data, bayesian_probabilities
            )

            return {
                "individual_probabilities": bayesian_probabilities,
                "credible_intervals": credible_intervals,
                "model_comparison": model_comparison,
                "inference_config": self.bayesian_config,
            }

        except Exception as e:
            self.logger.error(f"베이지안 확률 추론 실패: {e}")
            return {}

    def conditional_probability_analysis(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        조건부 확률 분석

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 조건부 확률 분석 결과
        """
        try:
            # 조건부 확률 계산
            conditional_probs = {}

            # 1. 이전 번호 출현 조건부 확률
            prev_number_conditions = self._calculate_previous_number_conditions(data)

            # 2. 번호 범위 조건부 확률
            range_conditions = self._calculate_range_conditions(data)

            # 3. 패턴 조건부 확률
            pattern_conditions = self._calculate_pattern_conditions(data)

            # 4. 시간적 조건부 확률
            temporal_conditions = self._calculate_temporal_conditions(data)

            conditional_probs = {
                "previous_number_conditions": prev_number_conditions,
                "range_conditions": range_conditions,
                "pattern_conditions": pattern_conditions,
                "temporal_conditions": temporal_conditions,
            }

            # 조건부 확률 종합 분석
            comprehensive_analysis = self._analyze_conditional_dependencies(
                conditional_probs
            )

            return {
                "conditional_probabilities": conditional_probs,
                "dependency_analysis": comprehensive_analysis,
                "condition_strength": self._measure_condition_strength(
                    conditional_probs
                ),
            }

        except Exception as e:
            self.logger.error(f"조건부 확률 분석 실패: {e}")
            return {}

    def update_posterior_probabilities(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        사후 확률 업데이트

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 사후 확률 업데이트 결과
        """
        try:
            # 순차적 베이지안 업데이트
            sequential_updates = self._perform_sequential_updates(data)

            # 배치 베이지안 업데이트
            batch_updates = self._perform_batch_updates(data)

            # 적응적 업데이트
            adaptive_updates = self._perform_adaptive_updates(data)

            # 업데이트 수렴성 분석
            convergence_analysis = self._analyze_update_convergence(sequential_updates)

            return {
                "sequential_updates": sequential_updates,
                "batch_updates": batch_updates,
                "adaptive_updates": adaptive_updates,
                "convergence_analysis": convergence_analysis,
                "update_summary": self._summarize_updates(
                    sequential_updates, batch_updates, adaptive_updates
                ),
            }

        except Exception as e:
            self.logger.error(f"사후 확률 업데이트 실패: {e}")
            return {}

    def _initialize_prior_distributions(self) -> Dict[str, Dict[str, float]]:
        """사전 확률 분포 초기화"""
        priors = {}

        # 각 번호에 대한 균등 사전 분포 (베타 분포)
        for number in range(1, 46):
            priors[str(number)] = {
                "alpha": self.bayesian_config["prior_alpha"],
                "beta": self.bayesian_config["prior_beta"],
                "distribution_type": "beta",
            }

        return priors

    def _count_number_occurrences(self, data: List[LotteryNumber]) -> Dict[int, int]:
        """번호별 출현 횟수 계산"""
        counts = defaultdict(int)

        for draw in data:
            for number in draw.numbers:
                counts[number] += 1

        return dict(counts)

    def _compare_bayesian_models(
        self, data: List[LotteryNumber], probabilities: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """베이지안 모델 비교"""
        # 모델 1: 균등 사전 분포
        # 모델 2: 정보적 사전 분포
        # 모델 3: 계층적 베이지안 모델

        models = {}

        # 모델 1: 균등 사전
        uniform_prior_log_likelihood = self._calculate_log_likelihood(
            data, probabilities, "uniform"
        )
        models["uniform_prior"] = {
            "log_likelihood": uniform_prior_log_likelihood,
            "aic": -2 * uniform_prior_log_likelihood + 2 * 45,  # 45개 번호
            "bic": -2 * uniform_prior_log_likelihood + np.log(len(data)) * 45,
        }

        # 모델 2: 정보적 사전 (로또 특성 반영)
        informative_prior_log_likelihood = self._calculate_log_likelihood(
            data, probabilities, "informative"
        )
        models["informative_prior"] = {
            "log_likelihood": informative_prior_log_likelihood,
            "aic": -2 * informative_prior_log_likelihood + 2 * 45,
            "bic": -2 * informative_prior_log_likelihood + np.log(len(data)) * 45,
        }

        # 최적 모델 선택
        best_model = min(models.keys(), key=lambda k: models[k]["aic"])

        return {
            "models": models,
            "best_model": best_model,
            "model_weights": self._calculate_model_weights(models),
        }

    def _calculate_log_likelihood(
        self,
        data: List[LotteryNumber],
        probabilities: Dict[str, Dict[str, float]],
        prior_type: str,
    ) -> float:
        """로그 우도 계산"""
        log_likelihood = 0.0

        for draw in data:
            for number in draw.numbers:
                prob = probabilities[str(number)]["posterior_mean"]
                log_likelihood += np.log(
                    prob + 1e-10
                )  # 수치적 안정성을 위한 작은 값 추가

        return log_likelihood

    def _calculate_model_weights(
        self, models: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """모델 가중치 계산 (AIC 기반)"""
        aic_values = [model["aic"] for model in models.values()]
        min_aic = min(aic_values)

        weights = {}
        total_weight = 0

        for model_name, model_info in models.items():
            weight = np.exp(-(model_info["aic"] - min_aic) / 2)
            weights[model_name] = weight
            total_weight += weight

        # 정규화
        for model_name in weights:
            weights[model_name] /= total_weight

        return weights

    def _calculate_previous_number_conditions(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """이전 번호 출현 조건부 확률"""
        conditions = {}

        for i in range(1, len(data)):
            prev_numbers = set(data[i - 1].numbers)
            curr_numbers = set(data[i].numbers)

            for prev_num in prev_numbers:
                for curr_num in curr_numbers:
                    key = f"{prev_num}_to_{curr_num}"
                    if key not in conditions:
                        conditions[key] = {"count": 0, "total": 0}
                    conditions[key]["count"] += 1

            # 총 가능한 조건 수 계산
            for prev_num in prev_numbers:
                for possible_num in range(1, 46):
                    key = f"{prev_num}_to_{possible_num}"
                    if key not in conditions:
                        conditions[key] = {"count": 0, "total": 0}
                    conditions[key]["total"] += 1

        # 조건부 확률 계산
        conditional_probs = {}
        for key, counts in conditions.items():
            if counts["total"] > 0:
                conditional_probs[key] = counts["count"] / counts["total"]
            else:
                conditional_probs[key] = 0.0

        return conditional_probs

    def _calculate_range_conditions(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """번호 범위 조건부 확률"""
        ranges = {
            "low": list(range(1, 16)),  # 1-15
            "mid": list(range(16, 31)),  # 16-30
            "high": list(range(31, 46)),  # 31-45
        }

        range_conditions = {}

        for draw in data:
            range_counts = {r: 0 for r in ranges}

            for number in draw.numbers:
                for range_name, range_numbers in ranges.items():
                    if number in range_numbers:
                        range_counts[range_name] += 1

            # 범위별 조건부 확률 계산
            for range_name, count in range_counts.items():
                if range_name not in range_conditions:
                    range_conditions[range_name] = []
                range_conditions[range_name].append(count)

        # 통계 계산
        range_stats = {}
        for range_name, counts in range_conditions.items():
            range_stats[range_name] = {
                "mean": float(np.mean(counts)),
                "std": float(np.std(counts)),
                "distribution": counts,
            }

        return range_stats

    def _calculate_pattern_conditions(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """패턴 조건부 확률"""
        patterns = {
            "consecutive": 0,  # 연속 번호
            "arithmetic": 0,  # 등차수열
            "sum_range": {},  # 합계 범위
        }

        for draw in data:
            numbers = sorted(draw.numbers)

            # 연속 번호 패턴
            consecutive_count = 0
            for i in range(len(numbers) - 1):
                if numbers[i + 1] - numbers[i] == 1:
                    consecutive_count += 1

            if consecutive_count > 0:
                patterns["consecutive"] += 1

            # 등차수열 패턴 (간단한 버전)
            if len(numbers) >= 3:
                diffs = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
                if len(set(diffs)) == 1:  # 모든 차이가 같음
                    patterns["arithmetic"] += 1

            # 합계 범위
            total_sum = sum(numbers)
            sum_range = f"{total_sum // 50 * 50}-{(total_sum // 50 + 1) * 50}"
            if sum_range not in patterns["sum_range"]:
                patterns["sum_range"][sum_range] = 0
            patterns["sum_range"][sum_range] += 1

        # 확률로 변환
        total_draws = len(data)
        pattern_probs = {
            "consecutive_probability": patterns["consecutive"] / total_draws,
            "arithmetic_probability": patterns["arithmetic"] / total_draws,
            "sum_range_distribution": {
                k: v / total_draws for k, v in patterns["sum_range"].items()
            },
        }

        return pattern_probs

    def _calculate_temporal_conditions(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """시간적 조건부 확률"""
        window_size = self.bayesian_config["condition_window"]
        temporal_conditions = {}

        for number in range(1, 46):
            number_str = str(number)
            appearances = []

            # 번호 출현 위치 찾기
            for i, draw in enumerate(data):
                if number in draw.numbers:
                    appearances.append(i)

            # 출현 간격 분석
            if len(appearances) > 1:
                intervals = np.diff(appearances)
                temporal_conditions[number_str] = {
                    "mean_interval": float(np.mean(intervals)),
                    "std_interval": float(np.std(intervals)),
                    "last_appearance": appearances[-1] if appearances else -1,
                    "expected_next": (
                        appearances[-1] + np.mean(intervals) if appearances else -1
                    ),
                }
            else:
                temporal_conditions[number_str] = {
                    "mean_interval": len(data),
                    "std_interval": 0.0,
                    "last_appearance": appearances[0] if appearances else -1,
                    "expected_next": -1,
                }

        return temporal_conditions

    def _analyze_conditional_dependencies(
        self, conditional_probs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """조건부 의존성 분석"""
        dependencies = {
            "strong_dependencies": [],
            "weak_dependencies": [],
            "independence_tests": {},
        }

        # 강한 의존성 탐지 (임계값 기반)
        threshold_strong = 0.7
        threshold_weak = 0.3

        # 이전 번호 조건부 확률에서 강한 의존성 찾기
        prev_conditions = conditional_probs.get("previous_number_conditions", {})
        for condition, prob in prev_conditions.items():
            if prob > threshold_strong:
                dependencies["strong_dependencies"].append(
                    {
                        "condition": condition,
                        "probability": prob,
                        "type": "previous_number",
                    }
                )
            elif prob > threshold_weak:
                dependencies["weak_dependencies"].append(
                    {
                        "condition": condition,
                        "probability": prob,
                        "type": "previous_number",
                    }
                )

        return dependencies

    def _measure_condition_strength(
        self, conditional_probs: Dict[str, Any]
    ) -> Dict[str, float]:
        """조건 강도 측정"""
        strength_metrics = {}

        # 이전 번호 조건 강도
        prev_conditions = conditional_probs.get("previous_number_conditions", {})
        if prev_conditions:
            probs = list(prev_conditions.values())
            strength_metrics["previous_number_strength"] = float(np.std(probs))

        # 범위 조건 강도
        range_conditions = conditional_probs.get("range_conditions", {})
        if range_conditions:
            range_stds = [stats["std"] for stats in range_conditions.values()]
            strength_metrics["range_condition_strength"] = float(np.mean(range_stds))

        # 패턴 조건 강도
        pattern_conditions = conditional_probs.get("pattern_conditions", {})
        if pattern_conditions:
            pattern_probs = [
                pattern_conditions.get("consecutive_probability", 0),
                pattern_conditions.get("arithmetic_probability", 0),
            ]
            strength_metrics["pattern_condition_strength"] = float(
                np.std(pattern_probs)
            )

        return strength_metrics

    def _perform_sequential_updates(
        self, data: List[LotteryNumber]
    ) -> List[Dict[str, Any]]:
        """순차적 베이지안 업데이트"""
        updates = []
        current_alpha = {
            str(i): self.bayesian_config["prior_alpha"] for i in range(1, 46)
        }
        current_beta = {
            str(i): self.bayesian_config["prior_beta"] for i in range(1, 46)
        }

        for i, draw in enumerate(data):
            # 각 번호에 대해 업데이트
            for number in range(1, 46):
                number_str = str(number)

                if number in draw.numbers:
                    current_alpha[number_str] += 1
                else:
                    current_beta[number_str] += 1

            # 주요 업데이트 포인트만 저장 (메모리 효율성)
            if i % 10 == 0 or i == len(data) - 1:
                update_snapshot = {
                    "draw_index": i,
                    "alpha_values": current_alpha.copy(),
                    "beta_values": current_beta.copy(),
                    "posterior_means": {
                        num: current_alpha[num]
                        / (current_alpha[num] + current_beta[num])
                        for num in current_alpha
                    },
                }
                updates.append(update_snapshot)

        return updates

    def _perform_batch_updates(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """배치 베이지안 업데이트"""
        # 전체 데이터를 한 번에 사용하여 업데이트
        number_counts = self._count_number_occurrences(data)
        total_draws = len(data)

        batch_updates = {}

        for number in range(1, 46):
            number_str = str(number)
            successes = number_counts.get(number, 0)

            alpha_post = self.bayesian_config["prior_alpha"] + successes
            beta_post = self.bayesian_config["prior_beta"] + total_draws - successes

            batch_updates[number_str] = {
                "alpha_posterior": alpha_post,
                "beta_posterior": beta_post,
                "posterior_mean": alpha_post / (alpha_post + beta_post),
                "posterior_variance": (alpha_post * beta_post)
                / ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)),
            }

        return batch_updates

    def _perform_adaptive_updates(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """적응적 베이지안 업데이트"""
        # 최근 데이터에 더 높은 가중치를 부여
        adaptive_updates = {}

        # 지수적 가중치 계산
        weights = np.exp(np.linspace(-1, 0, len(data)))
        weights = weights / np.sum(weights)

        # 가중 업데이트
        for number in range(1, 46):
            number_str = str(number)
            weighted_successes = 0

            for i, draw in enumerate(data):
                if number in draw.numbers:
                    weighted_successes += weights[i]

            # 가중된 시행 횟수
            weighted_trials = np.sum(weights)

            alpha_post = self.bayesian_config["prior_alpha"] + weighted_successes
            beta_post = (
                self.bayesian_config["prior_beta"]
                + weighted_trials
                - weighted_successes
            )

            adaptive_updates[number_str] = {
                "weighted_successes": float(weighted_successes),
                "weighted_trials": float(weighted_trials),
                "alpha_posterior": float(alpha_post),
                "beta_posterior": float(beta_post),
                "posterior_mean": float(alpha_post / (alpha_post + beta_post)),
            }

        return adaptive_updates

    def _analyze_update_convergence(
        self, sequential_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """업데이트 수렴성 분석"""
        convergence_metrics = {}

        if len(sequential_updates) < 2:
            return {"convergence_status": "insufficient_data"}

        # 각 번호별 수렴성 분석
        for number in range(1, 46):
            number_str = str(number)

            # 시간에 따른 사후 평균 변화
            posterior_means = [
                update["posterior_means"][number_str] for update in sequential_updates
            ]

            # 수렴 지표 계산
            if len(posterior_means) > 1:
                # 변화율 계산
                changes = np.diff(posterior_means)
                convergence_metrics[number_str] = {
                    "final_mean": float(posterior_means[-1]),
                    "mean_change_rate": float(np.mean(np.abs(changes))),
                    "max_change": float(np.max(np.abs(changes))),
                    "converged": bool(
                        np.abs(changes[-1]) < self.bayesian_config["update_threshold"]
                    ),
                }

        # 전체 시스템 수렴성
        converged_count = sum(
            1
            for metrics in convergence_metrics.values()
            if metrics.get("converged", False)
        )
        total_numbers = len(convergence_metrics)

        return {
            "individual_convergence": convergence_metrics,
            "system_convergence": {
                "converged_ratio": (
                    converged_count / total_numbers if total_numbers > 0 else 0
                ),
                "converged_count": converged_count,
                "total_numbers": total_numbers,
            },
        }

    def _summarize_updates(
        self,
        sequential: List[Dict[str, Any]],
        batch: Dict[str, Any],
        adaptive: Dict[str, Any],
    ) -> Dict[str, Any]:
        """업데이트 요약"""
        summary = {
            "update_methods": {
                "sequential": len(sequential),
                "batch": len(batch),
                "adaptive": len(adaptive),
            },
            "comparison": {},
        }

        # 방법별 결과 비교
        if sequential and batch:
            final_sequential = sequential[-1]["posterior_means"]
            batch_means = {k: v["posterior_mean"] for k, v in batch.items()}

            differences = []
            for number_str in final_sequential:
                if number_str in batch_means:
                    diff = abs(final_sequential[number_str] - batch_means[number_str])
                    differences.append(diff)

            summary["comparison"]["sequential_vs_batch"] = {
                "mean_difference": float(np.mean(differences)) if differences else 0,
                "max_difference": float(np.max(differences)) if differences else 0,
            }

        return summary

    def _model_probability_distributions(
        self, data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """확률 분포 모델링"""
        distributions = {}

        # 번호별 출현 분포 모델링
        number_counts = self._count_number_occurrences(data)

        # 베타 분포 모델링
        beta_models = {}
        for number in range(1, 46):
            count = number_counts.get(number, 0)
            total = len(data)

            # 베타 분포 매개변수 추정
            alpha = count + 1
            beta = total - count + 1

            beta_models[str(number)] = {
                "alpha": alpha,
                "beta": beta,
                "mean": alpha / (alpha + beta),
                "variance": (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)),
            }

        # 전체 시스템 분포 모델링
        all_counts = list(number_counts.values())
        if all_counts:
            system_distribution = {
                "mean": float(np.mean(all_counts)),
                "std": float(np.std(all_counts)),
                "min": int(np.min(all_counts)),
                "max": int(np.max(all_counts)),
                "distribution_type": "empirical",
            }
        else:
            system_distribution = {"distribution_type": "empty"}

        distributions = {
            "individual_beta_models": beta_models,
            "system_distribution": system_distribution,
        }

        return distributions

    def _generate_bayesian_predictions(
        self,
        inference: Dict[str, Any],
        conditional: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """베이지안 예측 생성"""
        predictions = {}

        # 개별 번호 예측
        individual_predictions = {}
        probabilities = inference.get("individual_probabilities", {})

        for number_str, prob_data in probabilities.items():
            # 베이지안 예측 점수 계산
            posterior_mean = prob_data["posterior_mean"]
            posterior_var = prob_data["posterior_variance"]

            # 불확실성을 고려한 예측 점수
            uncertainty_penalty = np.sqrt(posterior_var)
            prediction_score = posterior_mean - uncertainty_penalty * 0.5

            individual_predictions[number_str] = {
                "prediction_score": float(prediction_score),
                "confidence": float(1.0 / (1.0 + posterior_var)),
                "posterior_mean": float(posterior_mean),
                "uncertainty": float(uncertainty_penalty),
            }

        # 상위 추천 번호
        top_predictions = sorted(
            individual_predictions.items(),
            key=lambda x: x[1]["prediction_score"],
            reverse=True,
        )[:10]

        predictions = {
            "individual_predictions": individual_predictions,
            "top_recommendations": [
                {
                    "number": int(num),
                    "score": data["prediction_score"],
                    "confidence": data["confidence"],
                }
                for num, data in top_predictions
            ],
            "prediction_summary": {
                "total_numbers_analyzed": len(individual_predictions),
                "high_confidence_count": len(
                    [
                        p
                        for p in individual_predictions.values()
                        if p["confidence"] > 0.7
                    ]
                ),
                "average_confidence": float(
                    np.mean([p["confidence"] for p in individual_predictions.values()])
                ),
            },
        }

        return predictions

    def _generate_bayesian_summary(
        self,
        inference: Dict[str, Any],
        conditional: Dict[str, Any],
        updates: Dict[str, Any],
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """베이지안 분석 요약"""
        summary = {
            "analysis_overview": {
                "total_numbers_analyzed": 45,
                "analysis_methods": [
                    "bayesian_inference",
                    "conditional_analysis",
                    "posterior_updates",
                ],
                "confidence_level": self.bayesian_config["confidence_level"],
            },
            "key_findings": {
                "best_model": inference.get("model_comparison", {}).get(
                    "best_model", "unknown"
                ),
                "top_predicted_numbers": predictions.get("top_recommendations", [])[:5],
                "system_convergence": updates.get("convergence_analysis", {}).get(
                    "system_convergence", {}
                ),
            },
            "recommendation_confidence": {
                "overall_confidence": predictions.get("prediction_summary", {}).get(
                    "average_confidence", 0
                ),
                "high_confidence_numbers": predictions.get(
                    "prediction_summary", {}
                ).get("high_confidence_count", 0),
            },
        }

        return summary

    def save_analysis_results(
        self, analysis_results: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """분석 결과를 파일로 저장"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bayesian_analyzer_results_{timestamp}.json"

            output_dir = Path("data/result/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / filename

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"BayesianAnalyzer 분석 결과 저장: {output_file}")
            return str(output_file)

        except Exception as e:
            self.logger.error(f"분석 결과 저장 실패: {e}")
            raise


# 팩토리 함수
def create_bayesian_analyzer(
    config: Optional[Dict[str, Any]] = None,
) -> BayesianAnalyzer:
    """BayesianAnalyzer 인스턴스 생성"""
    return BayesianAnalyzer(config)


# 편의 함수
def analyze_bayesian_probabilities(
    historical_data: List[LotteryNumber], config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """베이지안 확률 분석 수행"""
    analyzer = create_bayesian_analyzer(config)
    return analyzer.analyze(historical_data)
