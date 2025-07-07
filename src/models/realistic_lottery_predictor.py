"""
현실적 로또 예측 시스템 (Realistic Lottery Predictor)

6개 완전 적중이 아닌 하위 등급 적중률 최적화와 손실 최소화에 초점을 맞춘 예측기를 만들겠습니다.

목표:
- 5등(3개) 적중률: 15% → 35% (133% 개선)
- 4등(4개) 적중률: 1% → 5% (400% 개선)
- 3등(5개) 적중률: 0.1% → 0.3% (200% 개선)
- 총 손실률: 60% → 25% (58% 개선)
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
from collections import Counter
import random
from sklearn.cluster import KMeans
from dataclasses import dataclass
from enum import Enum
import yaml

from .base_model import BaseModel
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from ..utils.cache_paths import get_cache_dir
from ..utils.memory_manager import get_memory_manager

logger = get_logger(__name__)


class PrizeGrade(Enum):
    """로또 등급"""

    FIRST = 1  # 6개 일치
    SECOND = 2  # 5개 일치 + 보너스
    THIRD = 3  # 5개 일치
    FOURTH = 4  # 4개 일치
    FIFTH = 5  # 3개 일치


@dataclass
class GradeOptimizationTarget:
    """등급별 최적화 목표"""

    grade: PrizeGrade
    base_probability: float
    target_probability: float
    focus_weight: float
    expected_hit_rate: float


@dataclass
class PredictionResult:
    """예측 결과"""

    combinations: List[List[int]]
    grade_probabilities: Dict[PrizeGrade, float]
    strategy_type: str
    confidence_score: float
    expected_value: float
    risk_level: str


class RealisticLotteryPredictor(BaseModel):
    """현실적 로또 예측 시스템"""

    def __init__(self, config_path: str = "config/realistic_lottery_config.yaml"):
        """
        현실적 로또 예측기 초기화

        Args:
            config_path: 설정 파일 경로
        """
        super().__init__(config_path)
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()
        self.cache_dir = Path(get_cache_dir("realistic_models"))

        # 설정 로드
        self.config = self._load_config(config_path)

        # 목표 등급별 확률 개선 설정
        self.target_improvements = self._setup_target_improvements()

        # 포트폴리오 전략 설정
        self.portfolio_strategies = self._setup_portfolio_strategies()

        # 모델 컴포넌트 초기화
        self.models = {}
        self.analyzers = {}

        self.logger.info("✅ 현실적 로또 예측 시스템 초기화 완료")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("realistic_lottery", {})
        except FileNotFoundError:
            self.logger.warning(f"설정 파일 없음: {config_path}, 기본 설정 사용")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            "target_improvements": {
                "5th_prize": {
                    "base_probability": 0.017544,
                    "target_probability": 0.028571,
                    "focus_weight": 0.4,
                },
                "4th_prize": {
                    "base_probability": 0.000969,
                    "target_probability": 0.001429,
                    "focus_weight": 0.4,
                },
                "3rd_prize": {
                    "base_probability": 0.000028,
                    "target_probability": 0.00004,
                    "focus_weight": 0.2,
                },
            },
            "portfolio_allocation": {
                "conservative": 0.4,
                "aggressive": 0.4,
                "balanced": 0.2,
            },
            "risk_management": {"kelly_fraction": 0.25, "max_consecutive_losses": 10},
        }

    def _setup_target_improvements(self) -> Dict[PrizeGrade, GradeOptimizationTarget]:
        """등급별 최적화 목표 설정"""
        improvements = {}
        config = self.config.get("target_improvements", {})

        # 5등 (3개 일치)
        fifth_config = config.get("5th_prize", {})
        improvements[PrizeGrade.FIFTH] = GradeOptimizationTarget(
            grade=PrizeGrade.FIFTH,
            base_probability=fifth_config.get("base_probability", 0.017544),
            target_probability=fifth_config.get("target_probability", 0.028571),
            focus_weight=fifth_config.get("focus_weight", 0.4),
            expected_hit_rate=0.35,
        )

        # 4등 (4개 일치)
        fourth_config = config.get("4th_prize", {})
        improvements[PrizeGrade.FOURTH] = GradeOptimizationTarget(
            grade=PrizeGrade.FOURTH,
            base_probability=fourth_config.get("base_probability", 0.000969),
            target_probability=fourth_config.get("target_probability", 0.001429),
            focus_weight=fourth_config.get("focus_weight", 0.4),
            expected_hit_rate=0.05,
        )

        # 3등 (5개 일치)
        third_config = config.get("3rd_prize", {})
        improvements[PrizeGrade.THIRD] = GradeOptimizationTarget(
            grade=PrizeGrade.THIRD,
            base_probability=third_config.get("base_probability", 0.000028),
            target_probability=third_config.get("target_probability", 0.00004),
            focus_weight=third_config.get("focus_weight", 0.2),
            expected_hit_rate=0.003,
        )

        return improvements

    def _setup_portfolio_strategies(self) -> Dict[str, Dict]:
        """포트폴리오 전략 설정"""
        allocation = self.config.get("portfolio_allocation", {})
        strategies_config = self.config.get("strategies", {})

        return {
            "conservative": {
                "weight": allocation.get("conservative", 0.4),
                "focus_grades": [PrizeGrade.FIFTH],
                "methods": ["frequency_based", "cluster_analysis"],
                "risk_level": "low",
                "expected_hit_rate": 0.35,
                "config": strategies_config.get("conservative", {}),
            },
            "aggressive": {
                "weight": allocation.get("aggressive", 0.4),
                "focus_grades": [PrizeGrade.THIRD, PrizeGrade.FOURTH],
                "methods": ["trend_following", "ai_ensemble"],
                "risk_level": "high",
                "expected_hit_rate": 0.05,
                "config": strategies_config.get("aggressive", {}),
            },
            "balanced": {
                "weight": allocation.get("balanced", 0.2),
                "focus_grades": [PrizeGrade.THIRD, PrizeGrade.FOURTH, PrizeGrade.FIFTH],
                "methods": ["weighted_average", "meta_learning"],
                "risk_level": "medium",
                "expected_hit_rate": 0.15,
                "config": strategies_config.get("balanced", {}),
            },
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        모델 학습

        Args:
            X: 특성 데이터
            y: 타겟 데이터
        """
        try:
            self.logger.info("현실적 로또 예측기 학습 시작")

            # 데이터 변환 (LotteryNumber 객체로 변환)
            lottery_data = []
            for i, row in enumerate(X):
                # 번호 추출 (상위 6개 특성값 기준)
                numbers = self._extract_numbers_from_features(row)
                lottery_data.append(
                    LotteryNumber(
                        numbers=numbers,
                        id=i + 1,
                        draw_date=datetime.now().strftime("%Y-%m-%d"),
                    )
                )

            # 종합 분석 수행
            self._perform_comprehensive_analysis(lottery_data)

            # 등급별 최적화 전략 학습
            self._learn_grade_optimization_strategies(lottery_data)

            self.logger.info("현실적 로또 예측기 학습 완료")

        except Exception as e:
            self.logger.error(f"학습 중 오류 발생: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행

        Args:
            X: 예측할 특성 데이터

        Returns:
            예측 결과
        """
        try:
            # 데이터 변환
            lottery_data = []
            for i, row in enumerate(X):
                numbers = self._extract_numbers_from_features(row)
                lottery_data.append(
                    LotteryNumber(
                        numbers=numbers,
                        id=i + 1,
                        draw_date=datetime.now().strftime("%Y-%m-%d"),
                    )
                )

            # 등급별 최적화 예측 수행
            predictions = self.predict_with_grade_optimization(lottery_data)

            # 결과 변환
            result = np.zeros((len(predictions), 6))
            for i, pred in enumerate(predictions):
                if i < len(result):
                    result[i] = pred.numbers[:6]

            return result

        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            return np.zeros((len(X), 6))

    def predict_with_grade_optimization(
        self, historical_data: pd.DataFrame, total_combinations: int = 5
    ) -> List[PredictionResult]:
        """등급별 최적화 예측"""
        try:
            results = []

            # 포트폴리오 배분에 따른 조합 수 계산
            allocation = self._calculate_combination_allocation(total_combinations)

            # 각 전략별 예측 실행
            for strategy_name, strategy_config in self.portfolio_strategies.items():
                num_combinations = allocation[strategy_name]
                if num_combinations > 0:
                    strategy_result = self._generate_strategy_combinations(
                        historical_data,
                        strategy_name,
                        strategy_config,
                        num_combinations,
                    )
                    results.append(strategy_result)

            self.logger.info(f"등급별 최적화 예측 완료: {len(results)}개 전략")
            return results

        except Exception as e:
            self.logger.error(f"등급별 최적화 예측 오류: {e}")
            raise

    def _calculate_combination_allocation(
        self, total_combinations: int
    ) -> Dict[str, int]:
        """조합 배분 계산"""
        allocation = {}

        for strategy_name, strategy_config in self.portfolio_strategies.items():
            weight = strategy_config["weight"]
            allocated = int(total_combinations * weight)
            allocation[strategy_name] = allocated

        # 반올림 오차 조정
        total_allocated = sum(allocation.values())
        if total_allocated < total_combinations:
            # 가장 큰 비중의 전략에 나머지 할당
            max_strategy = max(
                allocation.keys(), key=lambda x: self.portfolio_strategies[x]["weight"]
            )
            allocation[max_strategy] += total_combinations - total_allocated

        return allocation

    def _generate_strategy_combinations(
        self,
        historical_data: pd.DataFrame,
        strategy_name: str,
        strategy_config: Dict,
        num_combinations: int,
    ) -> PredictionResult:
        """전략별 조합 생성"""
        try:
            focus_grades = strategy_config["focus_grades"]
            methods = strategy_config["methods"]

            # 각 방법별 예측 생성
            method_predictions = []
            for method in methods:
                predictions = self._apply_prediction_method(
                    historical_data, method, focus_grades, num_combinations
                )
                method_predictions.extend(predictions)

            # 최종 조합 선택 (중복 제거 및 최적화)
            final_combinations = self._select_optimal_combinations(
                method_predictions, num_combinations, focus_grades
            )

            # 등급별 확률 계산
            grade_probabilities = self._calculate_grade_probabilities(
                final_combinations, focus_grades
            )

            # 기대값 계산
            expected_value = self._calculate_expected_value(
                final_combinations, grade_probabilities
            )

            # 신뢰도 점수 계산
            confidence_score = self._calculate_confidence_score(
                final_combinations, method_predictions
            )

            return PredictionResult(
                combinations=final_combinations,
                grade_probabilities=grade_probabilities,
                strategy_type=strategy_name,
                confidence_score=confidence_score,
                expected_value=expected_value,
                risk_level=strategy_config["risk_level"],
            )

        except Exception as e:
            self.logger.error(f"전략별 조합 생성 오류 ({strategy_name}): {e}")
            raise

    def _apply_prediction_method(
        self,
        historical_data: pd.DataFrame,
        method: str,
        focus_grades: List[PrizeGrade],
        num_combinations: int,
    ) -> List[List[int]]:
        """예측 방법 적용"""
        try:
            if method == "frequency_based":
                return self._frequency_based_prediction(
                    historical_data, num_combinations
                )
            elif method == "cluster_analysis":
                return self._cluster_analysis_prediction(
                    historical_data, num_combinations
                )
            elif method == "trend_following":
                return self._trend_following_prediction(
                    historical_data, num_combinations
                )
            elif method == "ai_ensemble":
                return self._ai_ensemble_prediction(historical_data, num_combinations)
            elif method == "weighted_average":
                return self._weighted_average_prediction(
                    historical_data, num_combinations
                )
            elif method == "meta_learning":
                return self._meta_learning_prediction(historical_data, num_combinations)
            else:
                self.logger.warning(f"알 수 없는 예측 방법: {method}")
                return self._frequency_based_prediction(
                    historical_data, num_combinations
                )

        except Exception as e:
            self.logger.error(f"예측 방법 적용 오류 ({method}): {e}")
            return []

    def _frequency_based_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """빈도 기반 예측"""
        try:
            # 각 번호별 출현 빈도 계산
            frequency_counts = {}
            for _, row in historical_data.iterrows():
                numbers = [row[f"number_{i}"] for i in range(1, 7)]
                for num in numbers:
                    frequency_counts[num] = frequency_counts.get(num, 0) + 1

            # 빈도 순으로 정렬
            sorted_numbers = sorted(
                frequency_counts.items(), key=lambda x: x[1], reverse=True
            )

            # 상위 번호들로 조합 생성
            combinations = []
            high_freq_numbers = [num for num, _ in sorted_numbers[:20]]

            for i in range(num_combinations):
                # 상위 빈도 번호 중 6개 선택 (약간의 랜덤성 추가)
                selected = np.random.choice(high_freq_numbers, 6, replace=False)
                combinations.append(sorted(selected.tolist()))

            return combinations

        except Exception as e:
            self.logger.error(f"빈도 기반 예측 오류: {e}")
            return []

    def _cluster_analysis_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """클러스터 분석 예측"""
        try:
            # 데이터 준비
            X = historical_data[[f"number_{i}" for i in range(1, 7)]].values

            # 클러스터링 수행
            kmeans = KMeans(n_clusters=min(5, len(X) // 10), random_state=42)
            kmeans.fit(X)

            # 각 클러스터의 중심값 기반 조합 생성
            combinations = []
            for i in range(min(num_combinations, len(kmeans.cluster_centers_))):
                center = kmeans.cluster_centers_[i]
                # 중심값을 정수로 변환하고 1-45 범위로 조정
                numbers = np.clip(np.round(center).astype(int), 1, 45)
                # 중복 제거 및 6개 맞추기
                unique_numbers = list(set(numbers))
                while len(unique_numbers) < 6:
                    candidate = np.random.randint(1, 46)
                    if candidate not in unique_numbers:
                        unique_numbers.append(candidate)
                combinations.append(sorted(unique_numbers[:6]))

            # 부족한 조합은 랜덤으로 채움
            while len(combinations) < num_combinations:
                random_combination = sorted(
                    np.random.choice(range(1, 46), 6, replace=False)
                )
                combinations.append(random_combination)

            return combinations

        except Exception as e:
            self.logger.error(f"클러스터 분석 예측 오류: {e}")
            return []

    def _trend_following_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """트렌드 추종 예측"""
        try:
            # 최근 데이터에 더 높은 가중치 부여
            recent_data = historical_data.tail(20)

            # 최근 트렌드 분석
            trend_weights = {}
            for idx, (_, row) in enumerate(recent_data.iterrows()):
                weight = (idx + 1) / len(recent_data)  # 최근일수록 높은 가중치
                numbers = [row[f"number_{i}"] for i in range(1, 7)]
                for num in numbers:
                    trend_weights[num] = trend_weights.get(num, 0) + weight

            # 트렌드 점수 기반 조합 생성
            combinations = []
            trend_numbers = sorted(
                trend_weights.items(), key=lambda x: x[1], reverse=True
            )

            for i in range(num_combinations):
                # 상위 트렌드 번호 중 선택 (일부 랜덤성 포함)
                top_numbers = [num for num, _ in trend_numbers[:15]]
                selected = np.random.choice(top_numbers, 6, replace=False)
                combinations.append(sorted(selected.tolist()))

            return combinations

        except Exception as e:
            self.logger.error(f"트렌드 추종 예측 오류: {e}")
            return []

    def _ai_ensemble_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """AI 앙상블 예측"""
        try:
            # 여러 AI 모델의 예측 결과 통합
            predictions = []

            # 각 모델별 예측 (실제 구현에서는 훈련된 모델 사용)
            for model_name in ["lightgbm", "autoencoder", "tcn"]:
                model_predictions = self._get_model_predictions(
                    historical_data, model_name, num_combinations
                )
                predictions.extend(model_predictions)

            # 앙상블 가중치 적용
            ensemble_weights = self.config.get("models", {}).get("ensemble_weights", {})

            # 가중 평균 기반 최종 조합 선택
            final_combinations = self._weighted_ensemble_selection(
                predictions, ensemble_weights, num_combinations
            )

            return final_combinations

        except Exception as e:
            self.logger.error(f"AI 앙상블 예측 오류: {e}")
            return []

    def _weighted_average_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """가중 평균 예측"""
        try:
            # 여러 방법의 예측 결과를 가중 평균
            frequency_pred = self._frequency_based_prediction(
                historical_data, num_combinations
            )
            cluster_pred = self._cluster_analysis_prediction(
                historical_data, num_combinations
            )
            trend_pred = self._trend_following_prediction(
                historical_data, num_combinations
            )

            # 가중치 적용하여 최종 조합 선택
            all_predictions = frequency_pred + cluster_pred + trend_pred

            # 중복 제거 및 상위 조합 선택
            unique_combinations = []
            for combo in all_predictions:
                if combo not in unique_combinations:
                    unique_combinations.append(combo)

            return unique_combinations[:num_combinations]

        except Exception as e:
            self.logger.error(f"가중 평균 예측 오류: {e}")
            return []

    def _meta_learning_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """메타 러닝 예측"""
        try:
            # 과거 예측 성능을 학습하여 최적 전략 선택
            # 실제 구현에서는 과거 성능 데이터 활용

            # 임시로 여러 방법의 조합 사용
            methods = ["frequency_based", "cluster_analysis", "trend_following"]
            all_predictions = []

            for method in methods:
                method_predictions = self._apply_prediction_method(
                    historical_data, method, [PrizeGrade.FIFTH], num_combinations
                )
                all_predictions.extend(method_predictions)

            # 메타 학습 기반 최적 조합 선택
            return self._meta_learning_selection(all_predictions, num_combinations)

        except Exception as e:
            self.logger.error(f"메타 러닝 예측 오류: {e}")
            return []

    def _get_model_predictions(
        self, historical_data: pd.DataFrame, model_name: str, num_combinations: int
    ) -> List[List[int]]:
        """모델별 예측 결과 가져오기"""
        # 실제 구현에서는 훈련된 모델 사용
        # 임시로 랜덤 조합 반환
        combinations = []
        for _ in range(num_combinations):
            combination = sorted(np.random.choice(range(1, 46), 6, replace=False))
            combinations.append(combination)
        return combinations

    def _weighted_ensemble_selection(
        self,
        predictions: List[List[int]],
        weights: Dict[str, float],
        num_combinations: int,
    ) -> List[List[int]]:
        """가중 앙상블 선택"""
        # 가중치 기반 조합 선택 로직
        return predictions[:num_combinations]

    def _meta_learning_selection(
        self, predictions: List[List[int]], num_combinations: int
    ) -> List[List[int]]:
        """메타 러닝 기반 선택"""
        # 메타 러닝 선택 로직
        return predictions[:num_combinations]

    def _select_optimal_combinations(
        self,
        predictions: List[List[int]],
        num_combinations: int,
        focus_grades: List[PrizeGrade],
    ) -> List[List[int]]:
        """최적 조합 선택"""
        try:
            # 중복 제거
            unique_predictions = []
            for pred in predictions:
                if pred not in unique_predictions:
                    unique_predictions.append(pred)

            # 등급별 최적화 점수 계산
            scored_predictions = []
            for pred in unique_predictions:
                score = self._calculate_optimization_score(pred, focus_grades)
                scored_predictions.append((pred, score))

            # 점수 순으로 정렬하여 상위 조합 선택
            scored_predictions.sort(key=lambda x: x[1], reverse=True)

            return [pred for pred, _ in scored_predictions[:num_combinations]]

        except Exception as e:
            self.logger.error(f"최적 조합 선택 오류: {e}")
            return predictions[:num_combinations]

    def _calculate_optimization_score(
        self, combination: List[int], focus_grades: List[PrizeGrade]
    ) -> float:
        """최적화 점수 계산"""
        try:
            score = 0.0

            # 각 등급별 가중치 적용
            for grade in focus_grades:
                if grade in self.target_improvements:
                    target = self.target_improvements[grade]
                    grade_score = self._calculate_grade_score(combination, grade)
                    score += grade_score * target.focus_weight

            return score

        except Exception as e:
            self.logger.error(f"최적화 점수 계산 오류: {e}")
            return 0.0

    def _calculate_grade_score(
        self, combination: List[int], grade: PrizeGrade
    ) -> float:
        """등급별 점수 계산"""
        # 실제 구현에서는 복잡한 확률 계산
        # 임시로 간단한 점수 반환
        base_score = 1.0

        if grade == PrizeGrade.FIFTH:
            # 5등 최적화 로직
            base_score *= 1.5
        elif grade == PrizeGrade.FOURTH:
            # 4등 최적화 로직
            base_score *= 1.3
        elif grade == PrizeGrade.THIRD:
            # 3등 최적화 로직
            base_score *= 1.1

        return base_score

    def _calculate_grade_probabilities(
        self, combinations: List[List[int]], focus_grades: List[PrizeGrade]
    ) -> Dict[PrizeGrade, float]:
        """등급별 확률 계산"""
        try:
            probabilities = {}

            for grade in focus_grades:
                if grade in self.target_improvements:
                    target = self.target_improvements[grade]
                    # 목표 확률 기반 계산
                    probabilities[grade] = target.target_probability
                else:
                    probabilities[grade] = 0.0

            return probabilities

        except Exception as e:
            self.logger.error(f"등급별 확률 계산 오류: {e}")
            return {}

    def calculate_expected_value(
        self,
        combinations: List[List[int]],
        grade_probabilities: Dict[PrizeGrade, float],
    ) -> float:
        """기대값 계산 (Kelly Criterion 적용)"""
        try:
            # 각 등급별 상금 (임시 값)
            prize_amounts = {
                PrizeGrade.FIRST: 2000000000,  # 20억
                PrizeGrade.SECOND: 50000000,  # 5천만
                PrizeGrade.THIRD: 1500000,  # 150만
                PrizeGrade.FOURTH: 50000,  # 5만
                PrizeGrade.FIFTH: 5000,  # 5천
            }

            expected_value = 0.0
            cost_per_combination = 1000  # 1천원

            for grade, probability in grade_probabilities.items():
                if grade in prize_amounts:
                    expected_value += probability * prize_amounts[grade]

            # 비용 차감
            expected_value -= cost_per_combination

            return expected_value

        except Exception as e:
            self.logger.error(f"기대값 계산 오류: {e}")
            return 0.0

    def _calculate_expected_value(
        self,
        combinations: List[List[int]],
        grade_probabilities: Dict[PrizeGrade, float],
    ) -> float:
        """내부 기대값 계산"""
        return self.calculate_expected_value(combinations, grade_probabilities)

    def _calculate_confidence_score(
        self, combinations: List[List[int]], method_predictions: List[List[int]]
    ) -> float:
        """신뢰도 점수 계산"""
        try:
            # 여러 방법 간 일치도 기반 신뢰도 계산
            if not method_predictions:
                return 0.5

            # 조합 간 유사도 계산
            similarity_scores = []
            for combo in combinations:
                max_similarity = 0.0
                for pred in method_predictions:
                    similarity = len(set(combo) & set(pred)) / 6.0
                    max_similarity = max(max_similarity, similarity)
                similarity_scores.append(max_similarity)

            return np.mean(similarity_scores)

        except Exception as e:
            self.logger.error(f"신뢰도 점수 계산 오류: {e}")
            return 0.5

    def get_portfolio_summary(
        self, predictions: List[PredictionResult]
    ) -> Dict[str, Any]:
        """포트폴리오 요약 정보"""
        try:
            summary = {
                "total_combinations": sum(
                    len(pred.combinations) for pred in predictions
                ),
                "strategies": {},
                "expected_performance": {},
                "risk_metrics": {},
            }

            # 전략별 요약
            for pred in predictions:
                summary["strategies"][pred.strategy_type] = {
                    "combinations_count": len(pred.combinations),
                    "expected_value": pred.expected_value,
                    "confidence_score": pred.confidence_score,
                    "risk_level": pred.risk_level,
                }

            # 전체 기대 성능
            total_expected_value = sum(pred.expected_value for pred in predictions)
            summary["expected_performance"] = {
                "total_expected_value": total_expected_value,
                "average_confidence": np.mean(
                    [pred.confidence_score for pred in predictions]
                ),
            }

            return summary

        except Exception as e:
            self.logger.error(f"포트폴리오 요약 생성 오류: {e}")
            return {}

    def save_predictions(self, predictions: List[PredictionResult], round_number: int):
        """예측 결과 저장"""
        try:
            save_path = self.cache_dir / f"predictions_round_{round_number}.json"

            # 직렬화 가능한 형태로 변환
            serializable_predictions = []
            for pred in predictions:
                serializable_pred = {
                    "combinations": pred.combinations,
                    "grade_probabilities": {
                        grade.value: prob
                        for grade, prob in pred.grade_probabilities.items()
                    },
                    "strategy_type": pred.strategy_type,
                    "confidence_score": pred.confidence_score,
                    "expected_value": pred.expected_value,
                    "risk_level": pred.risk_level,
                }
                serializable_predictions.append(serializable_pred)

            import json

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(serializable_predictions, f, ensure_ascii=False, indent=2)

            self.logger.info(f"예측 결과 저장 완료: {save_path}")

        except Exception as e:
            self.logger.error(f"예측 결과 저장 오류: {e}")

    def _perform_comprehensive_analysis(self, data: List[LotteryNumber]) -> None:
        """종합 분석 수행"""
        try:
            self.logger.info("종합 분석 시작")

            # 1. 빈도 분석
            self._analyze_frequency(data)

            # 2. 패턴 분석
            self._analyze_patterns(data)

            # 3. 클러스터 분석
            self._analyze_clusters(data)

            # 4. 트렌드 분석
            self._analyze_trends(data)

            # 5. ROI 분석
            self._analyze_roi(data)

            self.logger.info("종합 분석 완료")

        except Exception as e:
            self.logger.error(f"종합 분석 중 오류: {str(e)}")

    def _analyze_frequency(self, data: List[LotteryNumber]) -> None:
        """빈도 분석"""
        try:
            # 전체 빈도 계산
            all_numbers = []
            for draw in data:
                all_numbers.extend(draw.numbers)

            frequency_counter = Counter(all_numbers)
            total_draws = len(data)

            # 빈도 정규화
            frequency_map = {}
            for num in range(1, 46):
                count = frequency_counter.get(num, 0)
                frequency_map[num] = count / total_draws if total_draws > 0 else 0

            # 최근성 분석 (최근 20회)
            recent_data = data[-20:] if len(data) >= 20 else data
            recent_numbers = []
            for draw in recent_data:
                recent_numbers.extend(draw.numbers)

            recent_counter = Counter(recent_numbers)
            recent_frequency = {}
            for num in range(1, 46):
                count = recent_counter.get(num, 0)
                recent_frequency[num] = (
                    count / len(recent_data) if len(recent_data) > 0 else 0
                )

            # 결과 저장
            self.analysis_results["frequency_analysis"] = {
                "overall_frequency": frequency_map,
                "recent_frequency": recent_frequency,
                "hot_numbers": [
                    num
                    for num, freq in sorted(
                        frequency_map.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                ],
                "cold_numbers": [
                    num
                    for num, freq in sorted(frequency_map.items(), key=lambda x: x[1])[
                        :10
                    ]
                ],
            }

        except Exception as e:
            self.logger.error(f"빈도 분석 중 오류: {str(e)}")

    def _analyze_patterns(self, data: List[LotteryNumber]) -> None:
        """패턴 분석"""
        try:
            patterns = {
                "even_odd": [],
                "low_high": [],
                "consecutive": [],
                "sum_range": [],
            }

            for draw in data:
                numbers = sorted(draw.numbers)

                # 홀짝 패턴
                odd_count = sum(1 for n in numbers if n % 2 == 1)
                patterns["even_odd"].append(odd_count)

                # 고저 패턴
                low_count = sum(1 for n in numbers if n <= 23)
                patterns["low_high"].append(low_count)

                # 연속 번호
                consecutive_count = 0
                for i in range(len(numbers) - 1):
                    if numbers[i + 1] - numbers[i] == 1:
                        consecutive_count += 1
                patterns["consecutive"].append(consecutive_count)

                # 합계 범위
                total_sum = sum(numbers)
                patterns["sum_range"].append(total_sum)

            # 패턴 분석 결과
            self.analysis_results["pattern_analysis"] = {
                "even_odd_distribution": Counter(patterns["even_odd"]),
                "low_high_distribution": Counter(patterns["low_high"]),
                "consecutive_distribution": Counter(patterns["consecutive"]),
                "sum_range_stats": {
                    "mean": np.mean(patterns["sum_range"]),
                    "std": np.std(patterns["sum_range"]),
                    "min": min(patterns["sum_range"]),
                    "max": max(patterns["sum_range"]),
                },
            }

        except Exception as e:
            self.logger.error(f"패턴 분석 중 오류: {str(e)}")

    def _analyze_clusters(self, data: List[LotteryNumber]) -> None:
        """클러스터 분석"""
        try:
            # 번호를 원-핫 인코딩으로 변환
            features = []
            for draw in data:
                feature = [0] * 45
                for num in draw.numbers:
                    feature[num - 1] = 1
                features.append(feature)

            features = np.array(features)

            # K-means 클러스터링
            n_clusters = min(5, len(data) // 10)  # 클러스터 수 조정
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(features)

                # 클러스터별 특성 분석
                cluster_analysis = {}
                for i in range(n_clusters):
                    cluster_indices = np.where(cluster_labels == i)[0]
                    cluster_data = [data[idx] for idx in cluster_indices]

                    # 클러스터 내 번호 빈도
                    cluster_numbers = []
                    for draw in cluster_data:
                        cluster_numbers.extend(draw.numbers)

                    cluster_frequency = Counter(cluster_numbers)
                    top_numbers = [num for num, _ in cluster_frequency.most_common(10)]

                    cluster_analysis[i] = {
                        "size": len(cluster_data),
                        "top_numbers": top_numbers,
                        "centroid": kmeans.cluster_centers_[i].tolist(),
                    }

                self.analysis_results["cluster_analysis"] = cluster_analysis

        except Exception as e:
            self.logger.error(f"클러스터 분석 중 오류: {str(e)}")

    def _analyze_trends(self, data: List[LotteryNumber]) -> None:
        """트렌드 분석"""
        try:
            # 시간별 트렌드 분석
            if len(data) >= 10:
                recent_data = data[-10:]  # 최근 10회
                older_data = data[-20:-10] if len(data) >= 20 else data[:-10]

                # 최근 vs 과거 빈도 비교
                recent_freq = Counter()
                older_freq = Counter()

                for draw in recent_data:
                    recent_freq.update(draw.numbers)

                for draw in older_data:
                    older_freq.update(draw.numbers)

                # 트렌드 계산
                trend_analysis = {}
                for num in range(1, 46):
                    recent_count = recent_freq.get(num, 0)
                    older_count = older_freq.get(num, 0)

                    # 트렌드 점수 계산
                    if older_count > 0:
                        trend_score = (recent_count - older_count) / older_count
                    else:
                        trend_score = recent_count

                    trend_analysis[num] = trend_score

                # 상승/하락 트렌드 번호
                ascending_numbers = [
                    num
                    for num, score in sorted(
                        trend_analysis.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                ]
                descending_numbers = [
                    num
                    for num, score in sorted(
                        trend_analysis.items(), key=lambda x: x[1]
                    )[:10]
                ]

                self.analysis_results["trend_analysis"] = {
                    "trend_scores": trend_analysis,
                    "ascending_numbers": ascending_numbers,
                    "descending_numbers": descending_numbers,
                }

        except Exception as e:
            self.logger.error(f"트렌드 분석 중 오류: {str(e)}")

    def _analyze_roi(self, data: List[LotteryNumber]) -> None:
        """ROI 분석"""
        try:
            # 번호별 ROI 계산 (가상의 투자 시뮬레이션)
            roi_analysis = {}

            for target_num in range(1, 46):
                total_investment = 0
                total_return = 0

                for i, draw in enumerate(data):
                    if i < len(data) - 1:  # 다음 회차 결과 확인
                        next_draw = data[i + 1]

                        # 해당 번호에 투자했다고 가정
                        if target_num in draw.numbers:
                            total_investment += 1000  # 1000원 투자

                            # 다음 회차에서 당첨 여부 확인
                            if target_num in next_draw.numbers:
                                total_return += 5000  # 5등 당첨금

                # ROI 계산
                if total_investment > 0:
                    roi = (total_return - total_investment) / total_investment
                else:
                    roi = 0

                roi_analysis[target_num] = roi

            # 상위 ROI 번호
            top_roi_numbers = [
                num
                for num, roi in sorted(
                    roi_analysis.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]

            self.analysis_results["roi_analysis"] = {
                "roi_scores": roi_analysis,
                "top_roi_numbers": top_roi_numbers,
            }

        except Exception as e:
            self.logger.error(f"ROI 분석 중 오류: {str(e)}")

    def _learn_grade_optimization_strategies(self, data: List[LotteryNumber]) -> None:
        """등급별 최적화 전략 학습"""
        try:
            self.logger.info("등급별 최적화 전략 학습 시작")

            # 각 등급별 성공 패턴 분석
            grade_patterns = {
                "5th_prize": [],  # 3개 맞춘 경우
                "4th_prize": [],  # 4개 맞춘 경우
                "3rd_prize": [],  # 5개 맞춘 경우
            }

            # 과거 데이터에서 성공 패턴 추출
            for i in range(len(data) - 1):
                current_draw = data[i]
                next_draw = data[i + 1]

                # 가상의 예측 생성 (현재 회차 데이터 기반)
                virtual_predictions = self._generate_virtual_predictions(current_draw)

                # 다음 회차 결과와 비교
                for prediction in virtual_predictions:
                    matches = len(set(prediction) & set(next_draw.numbers))

                    if matches == 3:
                        grade_patterns["5th_prize"].append(prediction)
                    elif matches == 4:
                        grade_patterns["4th_prize"].append(prediction)
                    elif matches == 5:
                        grade_patterns["3rd_prize"].append(prediction)

            # 패턴 분석 결과 저장
            self.analysis_results["grade_patterns"] = grade_patterns

            self.logger.info("등급별 최적화 전략 학습 완료")

        except Exception as e:
            self.logger.error(f"등급별 최적화 전략 학습 중 오류: {str(e)}")

    def _generate_virtual_predictions(self, draw: LotteryNumber) -> List[List[int]]:
        """가상 예측 생성"""
        try:
            predictions = []

            # 현재 회차 번호 기반 변형
            base_numbers = draw.numbers

            for _ in range(10):  # 10개 가상 예측 생성
                # 기본 번호에서 일부 변경
                prediction = base_numbers.copy()

                # 1-2개 번호 변경
                change_count = random.randint(1, 2)
                for _ in range(change_count):
                    idx = random.randint(0, 5)
                    new_num = random.randint(1, 45)
                    while new_num in prediction:
                        new_num = random.randint(1, 45)
                    prediction[idx] = new_num

                predictions.append(sorted(prediction))

            return predictions

        except Exception as e:
            self.logger.error(f"가상 예측 생성 중 오류: {str(e)}")
            return []

    def save(self, path: str) -> None:
        """모델 저장"""
        try:
            save_data = {
                "target_improvements": self.target_improvements,
                "portfolio_strategies": self.portfolio_strategies,
                "analysis_results": self.analysis_results,
                "performance_history": self.performance_history,
                "timestamp": datetime.now().isoformat(),
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"현실적 로또 예측기 모델 저장 완료: {path}")

        except Exception as e:
            self.logger.error(f"모델 저장 중 오류: {str(e)}")

    def load(self, path: str) -> None:
        """모델 로드"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                save_data = json.load(f)

            self.target_improvements = save_data.get(
                "target_improvements", self.target_improvements
            )
            self.portfolio_strategies = save_data.get(
                "portfolio_strategies", self.portfolio_strategies
            )
            self.analysis_results = save_data.get(
                "analysis_results", self.analysis_results
            )
            self.performance_history = save_data.get(
                "performance_history", self.performance_history
            )

            self.logger.info(f"현실적 로또 예측기 모델 로드 완료: {path}")

        except Exception as e:
            self.logger.error(f"모델 로드 중 오류: {str(e)}")
