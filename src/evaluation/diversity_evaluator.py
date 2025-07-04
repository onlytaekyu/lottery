"""
다양성 평가 모듈 (Diversity Evaluator)

이 모듈은 로또 번호 추천 시스템의 다양성 평가 기능을 제공합니다.
여러 추천 번호 조합 간의 다양성을 측정하고 최적화합니다.
"""

# 1. 표준 라이브러리
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
import itertools
from collections import Counter

# 2. 서드파티
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 3. 프로젝트 내부
from ..shared.types import LotteryNumber, ModelPrediction
from ..utils.unified_logging import get_logger
from ..analysis.pattern_analysis_utils import calculate_combination_diversity_score

logger = get_logger(__name__)


class DiversityEvaluator:
    """
    다양성 평가 클래스

    로또 번호 조합들의 다양성을 다양한 메트릭으로 평가합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        다양성 평가기 초기화

        Args:
            config: 설정 객체
        """
        # 기본 설정
        default_config = {
            "diversity_metrics": [
                "hamming_distance",
                "jaccard_similarity",
                "entropy_score",
                "clustering_diversity",
                "position_diversity",
                "sum_diversity",
                "pattern_diversity",
            ],
            "clustering": {"n_clusters": 5, "random_state": 42},
            "weights": {
                "hamming_distance": 0.2,
                "jaccard_similarity": 0.15,
                "entropy_score": 0.2,
                "clustering_diversity": 0.15,
                "position_diversity": 0.1,
                "sum_diversity": 0.1,
                "pattern_diversity": 0.1,
            },
            "save_results": True,
            "results_dir": "data/result/diversity_analysis",
        }

        # 설정 병합
        self.config = default_config.copy()
        if config:
            diversity_config = config.get("diversity_evaluation", config)
            self.config.update(diversity_config)

        # 결과 저장 디렉토리 생성
        if self.config["save_results"]:
            results_dir = Path(self.config["results_dir"])
            results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("다양성 평가기 초기화 완료")

    def evaluate_diversity(
        self, combinations: List[List[int]], save_results: bool = None
    ) -> Dict[str, Any]:
        """
        번호 조합들의 다양성 종합 평가

        Args:
            combinations: 평가할 번호 조합 리스트
            save_results: 결과 저장 여부 (None이면 설정값 사용)

        Returns:
            다양성 평가 결과
        """
        logger.info(f"다양성 평가 시작: {len(combinations)}개 조합")
        start_time = time.time()

        if len(combinations) < 2:
            logger.warning("다양성 평가를 위해서는 최소 2개 이상의 조합이 필요합니다.")
            return {"error": "조합 수 부족", "combinations_count": len(combinations)}

        # 각 메트릭별 평가
        results = {}

        for metric in self.config["diversity_metrics"]:
            try:
                if metric == "hamming_distance":
                    score = self._calculate_hamming_diversity(combinations)
                elif metric == "jaccard_similarity":
                    score = self._calculate_jaccard_diversity(combinations)
                elif metric == "entropy_score":
                    score = self._calculate_entropy_diversity(combinations)
                elif metric == "clustering_diversity":
                    score = self._calculate_clustering_diversity(combinations)
                elif metric == "position_diversity":
                    score = self._calculate_position_diversity(combinations)
                elif metric == "sum_diversity":
                    score = self._calculate_sum_diversity(combinations)
                elif metric == "pattern_diversity":
                    score = self._calculate_pattern_diversity(combinations)
                else:
                    logger.warning(f"알 수 없는 다양성 메트릭: {metric}")
                    continue

                results[metric] = score
                logger.info(f"{metric}: {score:.4f}")

            except Exception as e:
                logger.error(f"{metric} 계산 중 오류: {e}")
                results[metric] = 0.0

        # 가중 평균 다양성 점수 계산
        weighted_score = self._calculate_weighted_diversity_score(results)

        # 최종 결과 구성
        final_results = {
            "overall_diversity_score": weighted_score,
            "metric_scores": results,
            "combinations_count": len(combinations),
            "evaluation_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
        }

        # 결과 저장
        if save_results or (save_results is None and self.config["save_results"]):
            self._save_diversity_results(final_results, combinations)

        logger.info(f"다양성 평가 완료: 종합 점수 {weighted_score:.4f}")
        return final_results

    def _calculate_hamming_diversity(self, combinations: List[List[int]]) -> float:
        """
        해밍 거리 기반 다양성 계산

        Args:
            combinations: 번호 조합 리스트

        Returns:
            해밍 다양성 점수 (0~1, 높을수록 다양함)
        """
        # 번호를 바이너리 벡터로 변환
        binary_vectors = []
        for combo in combinations:
            vector = np.zeros(45)  # 로또 번호 1~45
            for num in combo:
                if 1 <= num <= 45:
                    vector[num - 1] = 1
            binary_vectors.append(vector)

        binary_vectors = np.array(binary_vectors)

        # 해밍 거리 계산
        distances = pdist(binary_vectors, metric="hamming")

        # 평균 해밍 거리를 다양성 점수로 사용
        avg_distance = np.mean(distances)

        return float(avg_distance)

    def _calculate_jaccard_diversity(self, combinations: List[List[int]]) -> float:
        """
        자카드 유사도 기반 다양성 계산

        Args:
            combinations: 번호 조합 리스트

        Returns:
            자카드 다양성 점수 (0~1, 높을수록 다양함)
        """
        if len(combinations) < 2:
            return 0.0

        similarities = []

        # 모든 조합 쌍에 대해 자카드 유사도 계산
        for i in range(len(combinations)):
            for j in range(i + 1, len(combinations)):
                set1 = set(combinations[i])
                set2 = set(combinations[j])

                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))

                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)

        # 평균 유사도를 구한 후 1에서 빼서 다양성으로 변환
        avg_similarity = np.mean(similarities) if similarities else 0.0
        diversity = 1.0 - avg_similarity

        return float(diversity)

    def _calculate_entropy_diversity(self, combinations: List[List[int]]) -> float:
        """
        엔트로피 기반 다양성 계산

        Args:
            combinations: 번호 조합 리스트

        Returns:
            엔트로피 다양성 점수 (0~1, 높을수록 다양함)
        """
        # 모든 번호의 출현 빈도 계산
        all_numbers = []
        for combo in combinations:
            all_numbers.extend(combo)

        # 번호별 출현 빈도
        number_counts = Counter(all_numbers)

        # 확률 분포 계산
        total_count = len(all_numbers)
        probabilities = [count / total_count for count in number_counts.values()]

        # 엔트로피 계산
        entropy_score = entropy(probabilities, base=2)

        # 최대 엔트로피로 정규화 (로또 번호 45개)
        max_entropy = np.log2(45)
        normalized_entropy = entropy_score / max_entropy if max_entropy > 0 else 0.0

        return float(normalized_entropy)

    def _calculate_clustering_diversity(self, combinations: List[List[int]]) -> float:
        """
        클러스터링 기반 다양성 계산

        Args:
            combinations: 번호 조합 리스트

        Returns:
            클러스터링 다양성 점수 (0~1, 높을수록 다양함)
        """
        if len(combinations) < self.config["clustering"]["n_clusters"]:
            # 조합 수가 클러스터 수보다 적으면 모든 조합이 서로 다른 클러스터
            return 1.0

        # 번호를 특성 벡터로 변환
        features = []
        for combo in combinations:
            # 기본 통계 특성
            feature_vector = [
                np.mean(combo),  # 평균
                np.std(combo),  # 표준편차
                min(combo),  # 최솟값
                max(combo),  # 최댓값
                sum(combo),  # 합계
                len([n for n in combo if n % 2 == 1]),  # 홀수 개수
                len([n for n in combo if n <= 22]),  # 작은 수 개수
            ]
            features.append(feature_vector)

        features = np.array(features)

        # 특성 정규화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        try:
            # K-means 클러스터링
            n_clusters = min(self.config["clustering"]["n_clusters"], len(combinations))
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.config["clustering"]["random_state"],
                n_init=10,
            )
            cluster_labels = kmeans.fit_predict(features_scaled)

            # 실루엣 점수를 다양성 지표로 사용
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(features_scaled, cluster_labels)
                # 실루엣 점수를 0~1 범위로 변환 (-1~1 -> 0~1)
                diversity_score = (silhouette_avg + 1) / 2
            else:
                diversity_score = 0.0

        except Exception as e:
            logger.warning(f"클러스터링 다양성 계산 중 오류: {e}")
            diversity_score = 0.0

        return float(diversity_score)

    def _calculate_position_diversity(self, combinations: List[List[int]]) -> float:
        """
        위치 기반 다양성 계산

        Args:
            combinations: 번호 조합 리스트

        Returns:
            위치 다양성 점수 (0~1, 높을수록 다양함)
        """
        # 각 위치별 번호 분포 분석
        position_distributions = []

        for pos in range(6):  # 로또 번호 6개 위치
            position_numbers = []
            for combo in combinations:
                sorted_combo = sorted(combo)
                if pos < len(sorted_combo):
                    position_numbers.append(sorted_combo[pos])

            if position_numbers:
                # 해당 위치의 번호 다양성 (표준편차 사용)
                diversity = np.std(position_numbers)
                position_distributions.append(diversity)

        # 전체 위치의 평균 다양성
        avg_position_diversity = (
            np.mean(position_distributions) if position_distributions else 0.0
        )

        # 정규화 (로또 번호 범위 고려)
        max_std = np.sqrt(((45 - 1) ** 2) / 12)  # 균등분포의 표준편차
        normalized_diversity = avg_position_diversity / max_std if max_std > 0 else 0.0

        return float(min(normalized_diversity, 1.0))

    def _calculate_sum_diversity(self, combinations: List[List[int]]) -> float:
        """
        합계 기반 다양성 계산

        Args:
            combinations: 번호 조합 리스트

        Returns:
            합계 다양성 점수 (0~1, 높을수록 다양함)
        """
        # 각 조합의 합계 계산
        sums = [sum(combo) for combo in combinations]

        # 합계의 표준편차를 다양성으로 사용
        sum_diversity = np.std(sums)

        # 정규화 (로또 번호 합계 범위: 21~255)
        min_sum = 1 + 2 + 3 + 4 + 5 + 6  # 21
        max_sum = 40 + 41 + 42 + 43 + 44 + 45  # 255
        max_std = np.sqrt(((max_sum - min_sum) ** 2) / 12)
        normalized_diversity = sum_diversity / max_std if max_std > 0 else 0.0

        return float(min(normalized_diversity, 1.0))

    def _calculate_pattern_diversity(self, combinations: List[List[int]]) -> float:
        """
        패턴 기반 다양성 계산

        Args:
            combinations: 번호 조합 리스트

        Returns:
            패턴 다양성 점수 (0~1, 높을수록 다양함)
        """
        try:
            # 기존 패턴 분석 유틸리티 사용
            pattern_scores = []

            for combo in combinations:
                pattern_result = calculate_combination_diversity_score(combo)
                pattern_scores.append(pattern_result.get("diversity_score", 0.0))

            # 패턴 점수들의 다양성 (표준편차)
            pattern_diversity = np.std(pattern_scores) if pattern_scores else 0.0

            # 정규화 (0~1 범위)
            max_pattern_std = 0.5  # 경험적 최댓값
            normalized_diversity = (
                pattern_diversity / max_pattern_std if max_pattern_std > 0 else 0.0
            )

            return float(min(normalized_diversity, 1.0))

        except Exception as e:
            logger.warning(f"패턴 다양성 계산 중 오류: {e}")
            return 0.0

    def _calculate_weighted_diversity_score(
        self, metric_scores: Dict[str, float]
    ) -> float:
        """
        가중 평균 다양성 점수 계산

        Args:
            metric_scores: 각 메트릭별 점수

        Returns:
            가중 평균 다양성 점수
        """
        weights = self.config["weights"]
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, score in metric_scores.items():
            if metric in weights:
                weight = weights[metric]
                weighted_sum += score * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def optimize_diversity(
        self,
        candidate_combinations: List[List[int]],
        target_count: int,
        method: str = "greedy",
    ) -> List[List[int]]:
        """
        다양성을 최대화하는 조합 선택

        Args:
            candidate_combinations: 후보 조합들
            target_count: 선택할 조합 수
            method: 최적화 방법 ("greedy", "random", "clustering")

        Returns:
            다양성이 최대화된 조합들
        """
        logger.info(
            f"다양성 최적화 시작: {len(candidate_combinations)}개 후보 중 {target_count}개 선택"
        )

        if len(candidate_combinations) <= target_count:
            return candidate_combinations

        if method == "greedy":
            return self._greedy_diversity_selection(
                candidate_combinations, target_count
            )
        elif method == "random":
            return self._random_diversity_selection(
                candidate_combinations, target_count
            )
        elif method == "clustering":
            return self._clustering_diversity_selection(
                candidate_combinations, target_count
            )
        else:
            logger.warning(f"알 수 없는 최적화 방법: {method}, greedy 방법 사용")
            return self._greedy_diversity_selection(
                candidate_combinations, target_count
            )

    def _greedy_diversity_selection(
        self, candidates: List[List[int]], target_count: int
    ) -> List[List[int]]:
        """
        탐욕적 다양성 선택

        Args:
            candidates: 후보 조합들
            target_count: 선택할 조합 수

        Returns:
            선택된 조합들
        """
        if not candidates:
            return []

        selected = [candidates[0]]  # 첫 번째 조합 선택
        remaining = candidates[1:]

        for _ in range(target_count - 1):
            if not remaining:
                break

            best_candidate = None
            best_diversity = -1

            # 각 후보에 대해 현재 선택된 조합들과의 다양성 계산
            for candidate in remaining:
                test_combinations = selected + [candidate]
                diversity_result = self.evaluate_diversity(
                    test_combinations, save_results=False
                )
                diversity_score = diversity_result.get("overall_diversity_score", 0.0)

                if diversity_score > best_diversity:
                    best_diversity = diversity_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        logger.info(f"탐욕적 선택 완료: {len(selected)}개 조합 선택")
        return selected

    def _random_diversity_selection(
        self, candidates: List[List[int]], target_count: int
    ) -> List[List[int]]:
        """
        랜덤 다양성 선택 (여러 번 시도하여 최적 선택)

        Args:
            candidates: 후보 조합들
            target_count: 선택할 조합 수

        Returns:
            선택된 조합들
        """
        best_selection = None
        best_diversity = -1
        num_trials = min(100, len(candidates) // target_count)  # 시도 횟수

        for _ in range(num_trials):
            # 랜덤 선택
            selected_indices = np.random.choice(
                len(candidates), size=target_count, replace=False
            )
            selected = [candidates[i] for i in selected_indices]

            # 다양성 평가
            diversity_result = self.evaluate_diversity(selected, save_results=False)
            diversity_score = diversity_result.get("overall_diversity_score", 0.0)

            if diversity_score > best_diversity:
                best_diversity = diversity_score
                best_selection = selected

        logger.info(f"랜덤 선택 완료: {num_trials}번 시도 중 최적 선택")
        return best_selection or candidates[:target_count]

    def _clustering_diversity_selection(
        self, candidates: List[List[int]], target_count: int
    ) -> List[List[int]]:
        """
        클러스터링 기반 다양성 선택

        Args:
            candidates: 후보 조합들
            target_count: 선택할 조합 수

        Returns:
            선택된 조합들
        """
        # 특성 벡터 생성
        features = []
        for combo in candidates:
            feature_vector = [
                np.mean(combo),
                np.std(combo),
                min(combo),
                max(combo),
                sum(combo),
                len([n for n in combo if n % 2 == 1]),
                len([n for n in combo if n <= 22]),
            ]
            features.append(feature_vector)

        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        try:
            # 클러스터링
            n_clusters = min(target_count, len(candidates))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)

            # 각 클러스터에서 중심에 가장 가까운 조합 선택
            selected = []
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    # 클러스터 중심에 가장 가까운 점 찾기
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    distances = [
                        np.linalg.norm(features_scaled[i] - cluster_center)
                        for i in cluster_indices
                    ]
                    closest_idx = cluster_indices[np.argmin(distances)]
                    selected.append(candidates[closest_idx])

            logger.info(f"클러스터링 선택 완료: {len(selected)}개 조합 선택")
            return selected

        except Exception as e:
            logger.error(f"클러스터링 선택 중 오류: {e}")
            return candidates[:target_count]

    def _save_diversity_results(
        self, results: Dict[str, Any], combinations: List[List[int]]
    ) -> None:
        """
        다양성 평가 결과 저장

        Args:
            results: 평가 결과
            combinations: 평가된 조합들
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(self.config["results_dir"])

            # 결과 파일 저장
            results_file = results_dir / f"diversity_evaluation_{timestamp}.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 조합 파일 저장
            combinations_file = results_dir / f"evaluated_combinations_{timestamp}.json"
            with open(combinations_file, "w", encoding="utf-8") as f:
                json.dump(combinations, f, ensure_ascii=False, indent=2)

            logger.info(f"다양성 평가 결과 저장 완료: {results_file}")

        except Exception as e:
            logger.error(f"다양성 평가 결과 저장 중 오류: {e}")

    def compare_diversity_methods(
        self, combinations: List[List[int]], methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        다양한 다양성 측정 방법 비교

        Args:
            combinations: 평가할 조합들
            methods: 비교할 방법들 (None이면 모든 방법)

        Returns:
            방법별 비교 결과
        """
        if methods is None:
            methods = self.config["diversity_metrics"]

        comparison_results = {}

        for method in methods:
            # 임시로 해당 메트릭만 활성화
            temp_config = self.config.copy()
            temp_config["diversity_metrics"] = [method]
            temp_config["weights"] = {method: 1.0}

            # 임시 평가기 생성
            temp_evaluator = DiversityEvaluator({"diversity_evaluation": temp_config})

            # 다양성 평가
            result = temp_evaluator.evaluate_diversity(combinations, save_results=False)
            comparison_results[method] = result.get("overall_diversity_score", 0.0)

        logger.info("다양성 방법 비교 완료")
        return comparison_results


def get_diversity_evaluator(
    config: Optional[Dict[str, Any]] = None,
) -> DiversityEvaluator:
    """
    다양성 평가기 팩토리 함수

    Args:
        config: 설정 객체

    Returns:
        다양성 평가기 인스턴스
    """
    return DiversityEvaluator(config)


# 편의 함수들
def calculate_diversity_score(combinations: List[List[int]]) -> float:
    """
    간단한 다양성 점수 계산 함수

    Args:
        combinations: 번호 조합들

    Returns:
        다양성 점수 (0~1)
    """
    evaluator = DiversityEvaluator()
    result = evaluator.evaluate_diversity(combinations, save_results=False)
    return result.get("overall_diversity_score", 0.0)


def select_diverse_combinations(
    candidates: List[List[int]], count: int, method: str = "greedy"
) -> List[List[int]]:
    """
    다양성을 고려한 조합 선택 함수

    Args:
        candidates: 후보 조합들
        count: 선택할 개수
        method: 선택 방법

    Returns:
        선택된 조합들
    """
    evaluator = DiversityEvaluator()
    return evaluator.optimize_diversity(candidates, count, method)
