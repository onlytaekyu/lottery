"""
전략 생성기 (Strategy Generator)

다양한 로또 예측 전략을 생성하고 최적화하는 모듈입니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.cluster import KMeans
from collections import Counter
import random

from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


class PrizeGrade:
    """로또 등급"""
    FIRST = 1   # 6개 일치
    SECOND = 2  # 5개 일치 + 보너스
    THIRD = 3   # 5개 일치
    FOURTH = 4  # 4개 일치
    FIFTH = 5   # 3개 일치


class StrategyGenerator:
    """로또 예측 전략 생성기"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        전략 생성기 초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # 전략별 설정
        self.strategy_configs = self.config.get("strategies", {})
        
        # 모델 예측 캐시
        self.model_predictions_cache = {}
        
        self.logger.info("전략 생성기 초기화 완료")

    def generate_strategy_combinations(
        self,
        historical_data: pd.DataFrame,
        strategy_name: str,
        strategy_config: Dict,
        num_combinations: int,
    ) -> Dict[str, Any]:
        """
        전략별 조합 생성

        Args:
            historical_data: 과거 데이터
            strategy_name: 전략 이름
            strategy_config: 전략 설정
            num_combinations: 생성할 조합 수

        Returns:
            생성된 조합과 메타데이터
        """
        self.logger.info(f"전략 '{strategy_name}' 조합 생성 시작 (조합 수: {num_combinations})")

        focus_grades = strategy_config.get("focus_grades", [])
        methods = strategy_config.get("methods", [])
        risk_level = strategy_config.get("risk_level", "medium")

        # 방법별 예측 수행
        all_predictions = []
        method_weights = {}

        for method in methods:
            try:
                predictions = self._apply_prediction_method(
                    historical_data, method, focus_grades, num_combinations
                )
                all_predictions.extend(predictions)
                method_weights[method] = 1.0 / len(methods)  # 균등 가중치
                
                self.logger.debug(f"방법 '{method}': {len(predictions)}개 조합 생성")
            except Exception as e:
                self.logger.warning(f"방법 '{method}' 실패: {e}")
                continue

        # 최적 조합 선택
        if all_predictions:
            optimal_combinations = self._select_optimal_combinations(
                all_predictions, num_combinations, focus_grades
            )
        else:
            self.logger.warning(f"전략 '{strategy_name}'에서 조합 생성 실패, 랜덤 조합 사용")
            optimal_combinations = self._generate_random_combinations(num_combinations)

        # 등급별 확률 계산
        grade_probabilities = self._calculate_grade_probabilities(
            optimal_combinations, focus_grades
        )

        # 신뢰도 점수 계산
        confidence_score = self._calculate_confidence_score(
            optimal_combinations, all_predictions
        )

        # 기대값 계산
        expected_value = self._calculate_expected_value(
            optimal_combinations, grade_probabilities
        )

        result = {
            "combinations": optimal_combinations,
            "grade_probabilities": grade_probabilities,
            "strategy_type": strategy_name,
            "confidence_score": confidence_score,
            "expected_value": expected_value,
            "risk_level": risk_level,
            "method_weights": method_weights,
        }

        self.logger.info(
            f"전략 '{strategy_name}' 완료: {len(optimal_combinations)}개 조합, "
            f"신뢰도={confidence_score:.3f}, 기대값={expected_value:.3f}"
        )

        return result

    def _apply_prediction_method(
        self,
        historical_data: pd.DataFrame,
        method: str,
        focus_grades: List[int],
        num_combinations: int,
    ) -> List[List[int]]:
        """
        예측 방법 적용

        Args:
            historical_data: 과거 데이터
            method: 예측 방법
            focus_grades: 목표 등급
            num_combinations: 조합 수

        Returns:
            예측된 조합 리스트
        """
        method_map = {
            "frequency_based": self._frequency_based_prediction,
            "cluster_analysis": self._cluster_analysis_prediction,
            "trend_following": self._trend_following_prediction,
            "ai_ensemble": self._ai_ensemble_prediction,
            "weighted_average": self._weighted_average_prediction,
            "meta_learning": self._meta_learning_prediction,
        }

        if method not in method_map:
            self.logger.warning(f"알 수 없는 예측 방법: {method}")
            return self._generate_random_combinations(num_combinations)

        try:
            return method_map[method](historical_data, num_combinations)
        except Exception as e:
            self.logger.error(f"예측 방법 '{method}' 실행 실패: {e}")
            return self._generate_random_combinations(num_combinations)

    def _frequency_based_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """빈도 기반 예측"""
        try:
            # 전체 번호 빈도 계산
            all_numbers = []
            for _, row in historical_data.iterrows():
                numbers = [int(x) for x in str(row.get('numbers', '')).split(',') if x.strip().isdigit()]
                all_numbers.extend(numbers)

            if not all_numbers:
                return self._generate_random_combinations(num_combinations)

            # 번호별 빈도 계산
            number_freq = Counter(all_numbers)
            
            # 상위 빈도 번호들 선택 (20개 정도)
            high_freq_numbers = [num for num, _ in number_freq.most_common(20)]
            
            # 중위 빈도 번호들 선택
            sorted_numbers = sorted(number_freq.items(), key=lambda x: x[1])
            mid_range = len(sorted_numbers) // 3
            mid_freq_numbers = [num for num, _ in sorted_numbers[mid_range:2*mid_range]]

            combinations = []
            for _ in range(num_combinations):
                # 고빈도 3-4개, 중빈도 2-3개 조합
                high_count = random.randint(3, 4)
                mid_count = 6 - high_count
                
                selected_high = random.sample(high_freq_numbers, min(high_count, len(high_freq_numbers)))
                selected_mid = random.sample(mid_freq_numbers, min(mid_count, len(mid_freq_numbers)))
                
                combination = selected_high + selected_mid
                if len(combination) == 6:
                    combinations.append(sorted(combination))

            return combinations[:num_combinations] if combinations else self._generate_random_combinations(num_combinations)

        except Exception as e:
            self.logger.error(f"빈도 기반 예측 실패: {e}")
            return self._generate_random_combinations(num_combinations)

    def _cluster_analysis_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """클러스터 분석 기반 예측"""
        try:
            # 데이터 준비
            data_matrix = []
            for _, row in historical_data.iterrows():
                numbers = [int(x) for x in str(row.get('numbers', '')).split(',') if x.strip().isdigit()]
                if len(numbers) >= 6:
                    # 45차원 원-핫 벡터 생성
                    vector = np.zeros(45)
                    for num in numbers[:6]:
                        if 1 <= num <= 45:
                            vector[num-1] = 1
                    data_matrix.append(vector)

            if len(data_matrix) < 10:
                return self._generate_random_combinations(num_combinations)

            data_matrix = np.array(data_matrix)

            # K-means 클러스터링
            n_clusters = min(5, len(data_matrix) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit_predict(data_matrix)

            # 클러스터 중심 기반 조합 생성
            combinations = []
            cluster_centers = kmeans.cluster_centers_

            for i in range(num_combinations):
                center_idx = i % len(cluster_centers)
                center = cluster_centers[center_idx]
                
                # 상위 확률 번호 선택
                top_indices = np.argsort(center)[-12:]  # 상위 12개
                selected_numbers = []
                
                # 확률적 선택
                for idx in top_indices:
                    if random.random() < center[idx] * 2:  # 확률 조정
                        selected_numbers.append(idx + 1)
                        if len(selected_numbers) == 6:
                            break
                
                # 부족한 경우 랜덤 추가
                while len(selected_numbers) < 6:
                    num = random.randint(1, 45)
                    if num not in selected_numbers:
                        selected_numbers.append(num)

                combinations.append(sorted(selected_numbers))

            return combinations

        except Exception as e:
            self.logger.error(f"클러스터 분석 예측 실패: {e}")
            return self._generate_random_combinations(num_combinations)

    def _trend_following_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """트렌드 추종 예측"""
        try:
            # 최근 데이터에 더 높은 가중치
            recent_weight = 3
            old_weight = 1
            weighted_numbers = []

            for i, (_, row) in enumerate(historical_data.iterrows()):
                numbers = [int(x) for x in str(row.get('numbers', '')).split(',') if x.strip().isdigit()]
                weight = recent_weight if i >= len(historical_data) * 0.7 else old_weight
                
                for num in numbers:
                    if 1 <= num <= 45:
                        weighted_numbers.extend([num] * weight)

            if not weighted_numbers:
                return self._generate_random_combinations(num_combinations)

            # 가중 빈도 계산
            weighted_freq = Counter(weighted_numbers)
            
            # 트렌드 점수 계산 (최근 상승 트렌드)
            trend_scores = {}
            recent_data = historical_data.tail(10)
            
            for num in range(1, 46):
                recent_count = 0
                for _, row in recent_data.iterrows():
                    numbers = [int(x) for x in str(row.get('numbers', '')).split(',') if x.strip().isdigit()]
                    if num in numbers:
                        recent_count += 1
                trend_scores[num] = recent_count

            # 트렌드와 빈도 결합
            combined_scores = {}
            for num in range(1, 46):
                freq_score = weighted_freq.get(num, 0)
                trend_score = trend_scores.get(num, 0)
                combined_scores[num] = freq_score * 0.7 + trend_score * 0.3

            # 상위 번호들로 조합 생성
            top_numbers = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            combinations = []
            for _ in range(num_combinations):
                # 상위 15개에서 확률적 선택
                candidate_pool = [num for num, _ in top_numbers[:15]]
                combination = random.sample(candidate_pool, min(6, len(candidate_pool)))
                
                # 부족한 경우 추가
                while len(combination) < 6:
                    num = random.randint(1, 45)
                    if num not in combination:
                        combination.append(num)
                
                combinations.append(sorted(combination))

            return combinations

        except Exception as e:
            self.logger.error(f"트렌드 추종 예측 실패: {e}")
            return self._generate_random_combinations(num_combinations)

    def _ai_ensemble_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """AI 앙상블 예측"""
        try:
            # 여러 방법의 예측 결합
            methods = ["frequency_based", "cluster_analysis", "trend_following"]
            all_predictions = []
            
            for method in methods:
                if method != "ai_ensemble":  # 무한 재귀 방지
                    predictions = self._apply_prediction_method(
                        historical_data, method, [], num_combinations // len(methods) + 1
                    )
                    all_predictions.extend(predictions)

            if not all_predictions:
                return self._generate_random_combinations(num_combinations)

            # 번호별 등장 빈도 계산
            number_scores = Counter()
            for combination in all_predictions:
                for num in combination:
                    number_scores[num] += 1

            # 상위 번호들로 새로운 조합 생성
            top_numbers = [num for num, _ in number_scores.most_common(20)]
            
            combinations = []
            for _ in range(num_combinations):
                # 확률적 선택으로 다양성 확보
                selected = []
                for num in top_numbers:
                    if random.random() < 0.4:  # 40% 확률로 선택
                        selected.append(num)
                        if len(selected) == 6:
                            break
                
                # 부족한 경우 랜덤 추가
                while len(selected) < 6:
                    num = random.randint(1, 45)
                    if num not in selected:
                        selected.append(num)
                
                combinations.append(sorted(selected))

            return combinations

        except Exception as e:
            self.logger.error(f"AI 앙상블 예측 실패: {e}")
            return self._generate_random_combinations(num_combinations)

    def _weighted_average_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """가중 평균 예측"""
        try:
            # 빈도와 트렌드 기반 예측 결합
            freq_predictions = self._frequency_based_prediction(historical_data, num_combinations)
            trend_predictions = self._trend_following_prediction(historical_data, num_combinations)
            
            # 두 예측의 가중 결합
            combined_numbers = Counter()
            
            # 빈도 기반 예측에 60% 가중치
            for combination in freq_predictions:
                for num in combination:
                    combined_numbers[num] += 0.6
                    
            # 트렌드 기반 예측에 40% 가중치
            for combination in trend_predictions:
                for num in combination:
                    combined_numbers[num] += 0.4

            # 상위 번호들로 조합 생성
            top_numbers = [num for num, _ in combined_numbers.most_common(18)]
            
            combinations = []
            for _ in range(num_combinations):
                combination = random.sample(top_numbers, min(6, len(top_numbers)))
                combinations.append(sorted(combination))

            return combinations

        except Exception as e:
            self.logger.error(f"가중 평균 예측 실패: {e}")
            return self._generate_random_combinations(num_combinations)

    def _meta_learning_prediction(
        self, historical_data: pd.DataFrame, num_combinations: int
    ) -> List[List[int]]:
        """메타 러닝 예측"""
        try:
            # 과거 성과 기반으로 방법 가중치 조정
            methods = ["frequency_based", "cluster_analysis", "trend_following"]
            method_weights = {"frequency_based": 0.4, "cluster_analysis": 0.3, "trend_following": 0.3}
            
            # 각 방법별 예측 수행
            weighted_numbers = Counter()
            
            for method in methods:
                if method != "meta_learning":  # 무한 재귀 방지
                    predictions = self._apply_prediction_method(
                        historical_data, method, [], num_combinations
                    )
                    weight = method_weights.get(method, 0.33)
                    
                    for combination in predictions:
                        for num in combination:
                            weighted_numbers[num] += weight

            # 가중치 기반 조합 생성
            sorted_numbers = sorted(weighted_numbers.items(), key=lambda x: x[1], reverse=True)
            top_numbers = [num for num, _ in sorted_numbers[:20]]
            
            combinations = []
            for _ in range(num_combinations):
                # 확률적 선택
                selected = []
                for num in top_numbers:
                    selection_prob = weighted_numbers[num] / max(weighted_numbers.values())
                    if random.random() < selection_prob * 0.5:
                        selected.append(num)
                        if len(selected) == 6:
                            break
                
                # 부족한 경우 상위 번호에서 추가
                while len(selected) < 6:
                    for num in top_numbers:
                        if num not in selected:
                            selected.append(num)
                            break
                    if len(selected) < 6:
                        # 그래도 부족하면 랜덤 추가
                        num = random.randint(1, 45)
                        if num not in selected:
                            selected.append(num)
                
                combinations.append(sorted(selected))

            return combinations

        except Exception as e:
            self.logger.error(f"메타 러닝 예측 실패: {e}")
            return self._generate_random_combinations(num_combinations)

    def _generate_random_combinations(self, num_combinations: int) -> List[List[int]]:
        """랜덤 조합 생성 (fallback)"""
        combinations = []
        for _ in range(num_combinations):
            combination = sorted(random.sample(range(1, 46), 6))
            combinations.append(combination)
        return combinations

    def _select_optimal_combinations(
        self,
        predictions: List[List[int]],
        num_combinations: int,
        focus_grades: List[int],
    ) -> List[List[int]]:
        """최적 조합 선택"""
        if not predictions:
            return self._generate_random_combinations(num_combinations)

        # 조합별 점수 계산
        scored_combinations = []
        for combination in predictions:
            score = self._calculate_optimization_score(combination, focus_grades)
            scored_combinations.append((combination, score))

        # 점수 기준 정렬 및 상위 선택
        scored_combinations.sort(key=lambda x: x[1], reverse=True)
        
        # 중복 제거하면서 상위 조합 선택
        selected = []
        seen = set()
        
        for combination, score in scored_combinations:
            combo_tuple = tuple(sorted(combination))
            if combo_tuple not in seen:
                selected.append(combination)
                seen.add(combo_tuple)
                if len(selected) >= num_combinations:
                    break

        # 부족한 경우 랜덤으로 채우기
        while len(selected) < num_combinations:
            random_combo = sorted(random.sample(range(1, 46), 6))
            combo_tuple = tuple(random_combo)
            if combo_tuple not in seen:
                selected.append(random_combo)
                seen.add(combo_tuple)

        return selected[:num_combinations]

    def _calculate_optimization_score(
        self, combination: List[int], focus_grades: List[int]
    ) -> float:
        """최적화 점수 계산"""
        try:
            score = 0.0
            
            # 기본 다양성 점수
            numbers = sorted(combination)
            
            # 번호 분포 점수 (1-45 범위에서 고르게 분포)
            ranges = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            distribution_score = 1.0 / (1.0 + np.std(ranges))
            score += distribution_score * 0.3
            
            # 홀짝 균형 점수
            odd_count = sum(1 for num in numbers if num % 2 == 1)
            balance_score = 1.0 - abs(odd_count - 3) / 3.0
            score += balance_score * 0.2
            
            # 구간별 분포 점수 (1-15, 16-30, 31-45)
            range_counts = [0, 0, 0]
            for num in numbers:
                if num <= 15:
                    range_counts[0] += 1
                elif num <= 30:
                    range_counts[1] += 1
                else:
                    range_counts[2] += 1
            
            range_score = 1.0 - np.std(range_counts) / 2.0
            score += range_score * 0.2
            
            # 등급별 특화 점수
            for grade in focus_grades:
                grade_score = self._calculate_grade_score(combination, grade)
                score += grade_score * 0.3 / len(focus_grades) if focus_grades else 0

            return max(0.0, score)

        except Exception as e:
            self.logger.error(f"최적화 점수 계산 실패: {e}")
            return 0.0

    def _calculate_grade_score(self, combination: List[int], grade: int) -> float:
        """등급별 점수 계산"""
        try:
            # 간단한 휴리스틱 기반 점수
            if grade == PrizeGrade.FIFTH:  # 3개 일치 목표
                # 자주 나오는 번호 패턴 선호
                common_numbers = [1, 7, 14, 21, 28, 35, 42]  # 예시
                common_count = sum(1 for num in combination if num in common_numbers)
                return common_count / len(combination)
                
            elif grade == PrizeGrade.FOURTH:  # 4개 일치 목표
                # 중간 빈도 번호 선호
                mid_numbers = list(range(10, 36))
                mid_count = sum(1 for num in combination if num in mid_numbers)
                return mid_count / len(combination)
                
            elif grade == PrizeGrade.THIRD:  # 5개 일치 목표
                # 균형잡힌 분포 선호
                return self._calculate_optimization_score(combination, []) * 0.5
                
            return 0.5  # 기본 점수

        except Exception as e:
            self.logger.error(f"등급 점수 계산 실패: {e}")
            return 0.0

    def _calculate_grade_probabilities(
        self, combinations: List[List[int]], focus_grades: List[int]
    ) -> Dict[int, float]:
        """등급별 확률 계산"""
        probabilities = {}
        
        # 기본 확률 (실제 로또 확률 기반)
        base_probabilities = {
            PrizeGrade.FIFTH: 0.017544,   # 3개 일치
            PrizeGrade.FOURTH: 0.000969,  # 4개 일치
            PrizeGrade.THIRD: 0.000028,   # 5개 일치
            PrizeGrade.SECOND: 0.000002,  # 5개 일치 + 보너스
            PrizeGrade.FIRST: 0.0000000715,  # 6개 일치
        }
        
        # 조합 수에 따른 확률 조정
        num_combinations = len(combinations)
        
        for grade in [PrizeGrade.FIFTH, PrizeGrade.FOURTH, PrizeGrade.THIRD, PrizeGrade.SECOND, PrizeGrade.FIRST]:
            base_prob = base_probabilities.get(grade, 0)
            
            # 조합 수에 비례하여 확률 증가 (단순 모델)
            adjusted_prob = min(1.0, base_prob * num_combinations)
            
            # 포커스 등급인 경우 추가 보정
            if grade in focus_grades:
                adjusted_prob *= 1.5  # 50% 추가 보정
                
            probabilities[grade] = adjusted_prob

        return probabilities

    def _calculate_confidence_score(
        self, combinations: List[List[int]], method_predictions: List[List[int]]
    ) -> float:
        """신뢰도 점수 계산"""
        try:
            if not method_predictions:
                return 0.5

            # 선택된 조합이 여러 방법에서 얼마나 일치하는지 계산
            total_overlap = 0
            total_possible = 0

            for selected_combo in combinations:
                overlap_scores = []
                
                for method_combo in method_predictions:
                    # 두 조합 간 겹치는 번호 수
                    overlap = len(set(selected_combo) & set(method_combo))
                    overlap_scores.append(overlap / 6.0)  # 0-1 정규화
                
                if overlap_scores:
                    total_overlap += max(overlap_scores)  # 최대 겹침 점수
                    total_possible += 1

            confidence = total_overlap / total_possible if total_possible > 0 else 0.5
            return min(1.0, max(0.0, confidence))

        except Exception as e:
            self.logger.error(f"신뢰도 점수 계산 실패: {e}")
            return 0.5

    def _calculate_expected_value(
        self,
        combinations: List[List[int]],
        grade_probabilities: Dict[int, float],
    ) -> float:
        """기대값 계산"""
        try:
            # 등급별 상금 (원 단위, 대략적)
            prize_amounts = {
                PrizeGrade.FIRST: 2000000000,    # 1등: 20억
                PrizeGrade.SECOND: 50000000,     # 2등: 5천만
                PrizeGrade.THIRD: 1500000,       # 3등: 150만
                PrizeGrade.FOURTH: 50000,        # 4등: 5만
                PrizeGrade.FIFTH: 5000,          # 5등: 5천
            }
            
            # 로또 구입비
            cost_per_game = 1000
            total_cost = len(combinations) * cost_per_game
            
            # 기대 수익 계산
            expected_winnings = 0
            for grade, probability in grade_probabilities.items():
                prize = prize_amounts.get(grade, 0)
                expected_winnings += probability * prize
            
            # 기대값 = (기대 수익 - 비용) / 비용
            expected_value = (expected_winnings - total_cost) / total_cost
            
            return expected_value

        except Exception as e:
            self.logger.error(f"기대값 계산 실패: {e}")
            return -0.5  # 기본적으로 손실 예상 