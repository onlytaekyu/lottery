"""
구조/쌍/클러스터 분석기 모듈

이 모듈은 로또 번호의 구조적 패턴, 쌍 관계, 클러스터 구조 등을 분석하는 기능을 제공합니다.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from src.analysis.base_analyzer import BaseAnalyzer
from src.shared.types import LotteryNumber
from src.utils.error_handler_refactored import get_logger
from src.shared.graph_utils import calculate_pair_frequency, calculate_pair_centrality
from src.utils.unified_performance import performance_monitor

logger = get_logger(__name__)


class StructuralAnalyzer(BaseAnalyzer):
    """구조/쌍/클러스터 분석기 클래스"""

    def __init__(self, config: dict):
        """
        StructuralAnalyzer 초기화

        Args:
            config: 분석에 사용할 설정
        """
        super().__init__(config, name="structural")

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        BaseAnalyzer의 추상 메서드 구현

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 매개변수

        Returns:
            Dict[str, Any]: 구조적 패턴 분석 결과
        """
        return self.analyze(historical_data)

    def analyze(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        과거 로또 당첨 번호의 구조적 패턴을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 구조적 패턴 분석 결과
        """
        # 캐시 키 생성
        cache_key = self._create_cache_key("structural_analysis", len(historical_data))

        # 캐시 확인
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 분석 결과 사용: {cache_key}")
            return cached_result

        # 분석 수행
        self.logger.info(f"구조/쌍/클러스터 분석 시작: {len(historical_data)}개 데이터")

        results = {}

        # 번호 쌍 중심성 분석
        results["pair_centrality"] = self._analyze_pair_centrality(historical_data)

        # 클러스터 분석
        results["cluster_analysis"] = self._analyze_clusters(historical_data)

        # 실루엣 점수 (클러스터 품질)
        results["silhouette_score"] = self._calculate_silhouette_score(historical_data)

        # 응집성 점수
        results["cohesiveness_score"] = self._calculate_cohesiveness_score(
            historical_data
        )

        # 연속성 점수
        results["consecutiveness_score"] = self._calculate_consecutiveness_score(
            historical_data
        )

        # 쌍 빈도 분석 - graph_utils 사용
        with performance_monitor("calculate_pair_frequency"):
            # 쌍 빈도 계산
            pair_freq_tuples = calculate_pair_frequency(
                historical_data, logger=self.logger
            )

            # 결과 가공 및 포맷팅
            pair_frequency_result = {}

            # 상위 빈출 쌍 (상위 20개)
            top_pairs = sorted(
                pair_freq_tuples.items(), key=lambda x: x[1], reverse=True
            )[:20]
            for i, ((num1, num2), freq) in enumerate(top_pairs):
                pair_key = f"top_pair_{i+1}"
                pair_frequency_result[pair_key] = f"{num1}-{num2}"
                pair_frequency_result[f"{pair_key}_count"] = int(
                    freq * len(historical_data)
                )
                pair_frequency_result[f"{pair_key}_percentage"] = float(freq)

            # 전체 쌍 평균 출현 빈도
            total_pairs = len(pair_freq_tuples)
            avg_frequency = (
                sum(pair_freq_tuples.values()) / total_pairs if total_pairs > 0 else 0
            )

            pair_frequency_result["total_pair_types"] = total_pairs
            pair_frequency_result["avg_pair_frequency"] = float(avg_frequency)

            # 이론적으로 가능한 모든 쌍 (45C2 = 990개)
            theoretical_pairs = 990
            coverage = total_pairs / theoretical_pairs

            pair_frequency_result["pair_coverage"] = float(coverage)

            results["pair_frequency"] = pair_frequency_result
            self.logger.info(f"쌍 빈도 분석 완료: {total_pairs}개 쌍")

        # 번호 삼중항 빈도
        results["triplet_frequency"] = self._calculate_triplet_frequency(
            historical_data
        )

        # 네트워크 분석
        results["network_analysis"] = self._analyze_number_network(historical_data)

        # 범위 분포 분석
        results["range_distribution"] = self._analyze_range_distribution(
            historical_data
        )

        # 결과 캐싱
        self._save_to_cache(cache_key, results)

        return results

    def _analyze_pair_centrality(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        번호 쌍의 중심성을 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 번호 쌍 중심성 분석 결과
        """
        with performance_monitor("analyze_pair_centrality"):
            # graph_utils를 사용하여 쌍 빈도 계산
            pair_freq_tuples = calculate_pair_frequency(
                historical_data, logger=self.logger
            )

            # graph_utils를 사용하여 중심성 계산
            node_centrality = calculate_pair_centrality(
                pair_freq_tuples, logger=self.logger
            )

            # 결과 저장용 딕셔너리
            result = {}

            # 번호별 중심성 점수
            for num in range(1, 46):
                num_str = str(num)
                # 원래 키 형식을 유지하기 위해 가독성 있는 키로 변환
                result[f"degree_centrality_{num_str}"] = float(
                    node_centrality.get(num, 0)
                )

                # betweenness와 closeness 중심성은 graph_utils에서 제공하지 않으므로
                # 노드 중심성으로 대체
                result[f"betweenness_centrality_{num_str}"] = float(
                    node_centrality.get(num, 0)
                )
                result[f"closeness_centrality_{num_str}"] = float(
                    node_centrality.get(num, 0)
                )

            # 전체 중심성 평균
            if node_centrality:
                result["avg_degree_centrality"] = float(
                    np.mean(list(node_centrality.values()))
                )
                result["avg_betweenness_centrality"] = result[
                    "avg_degree_centrality"
                ]  # 동일한 값 사용
                result["avg_closeness_centrality"] = result[
                    "avg_degree_centrality"
                ]  # 동일한 값 사용
            else:
                result["avg_degree_centrality"] = 0.0
                result["avg_betweenness_centrality"] = 0.0
                result["avg_closeness_centrality"] = 0.0

            # 상위 중심성 번호
            top_centrality = sorted(
                node_centrality.items(), key=lambda x: x[1], reverse=True
            )[:5]
            result["top_degree_numbers"] = [int(num) for num, _ in top_centrality]
            result["top_betweenness_numbers"] = result[
                "top_degree_numbers"
            ]  # 동일한 값 사용

            self.logger.info("쌍 중심성 분석 완료")
            return result

    def _analyze_clusters(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        번호 클러스터를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 클러스터 분석 결과
        """
        # 번호 쌍 출현 횟수 계산
        pair_counter = Counter()

        for draw in historical_data:
            numbers = sorted(draw.numbers)
            # 모든 가능한 쌍 생성
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = (numbers[i], numbers[j])
                    pair_counter[pair] += 1

        # 번호 간 유사도 행렬 생성 (1-45)
        similarity_matrix = np.zeros((45, 45))

        for (num1, num2), count in pair_counter.items():
            # 정규화된 유사도 (0-1 범위)
            similarity = count / len(historical_data)
            similarity_matrix[num1 - 1, num2 - 1] = similarity
            similarity_matrix[num2 - 1, num1 - 1] = similarity  # 대칭 행렬

        # 유사도를 거리로 변환 (1 - 유사도)
        distance_matrix = 1 - similarity_matrix

        # KMeans 클러스터링 (다양한 k 값 시도)
        best_k = 5  # 기본값
        best_score = -1

        for k in range(3, 9):  # 3-8개 클러스터 시도
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(similarity_matrix)

                # 실루엣 점수 계산
                score = silhouette_score(similarity_matrix, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue

        # 최적 k로 클러스터링
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(similarity_matrix)

        # 클러스터 결과 정리
        clusters = defaultdict(list)
        for i in range(45):
            num = i + 1
            cluster_id = labels[i]
            clusters[cluster_id].append(num)

        # DBSCAN 클러스터링 (밀도 기반)
        try:
            dbscan = DBSCAN(eps=0.3, min_samples=2)
            dbscan_labels = dbscan.fit_predict(distance_matrix)

            dbscan_clusters = defaultdict(list)
            for i in range(45):
                num = i + 1
                cluster_id = dbscan_labels[i]
                dbscan_clusters[cluster_id].append(num)
        except:
            dbscan_clusters = {-1: list(range(1, 46))}

        # 결과 취합
        result = {
            "kmeans": {str(k): v for k, v in clusters.items()},
            "kmeans_k": best_k,
            "kmeans_silhouette_score": float(best_score),
            "dbscan": {str(k): v for k, v in dbscan_clusters.items() if k != -1},
            "dbscan_outliers": dbscan_clusters.get(-1, []),
        }

        return result

    def _calculate_silhouette_score(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        클러스터 품질(실루엣 점수)을 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 실루엣 점수 분석 결과
        """
        # 번호 쌍 출현 횟수 계산
        pair_counter = Counter()

        for draw in historical_data:
            numbers = sorted(draw.numbers)
            # 모든 가능한 쌍 생성
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = (numbers[i], numbers[j])
                    pair_counter[pair] += 1

        # 번호 간 유사도 행렬 생성 (1-45)
        similarity_matrix = np.zeros((45, 45))

        for (num1, num2), count in pair_counter.items():
            # 정규화된 유사도 (0-1 범위)
            similarity = count / len(historical_data)
            similarity_matrix[num1 - 1, num2 - 1] = similarity
            similarity_matrix[num2 - 1, num1 - 1] = similarity  # 대칭 행렬

        # 다양한 k 값에 대한 실루엣 점수 계산
        result = {}

        for k in range(3, 9):  # 3-8개 클러스터 시도
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(similarity_matrix)

                # 실루엣 점수 계산
                score = silhouette_score(similarity_matrix, labels)
                result[f"silhouette_k{k}"] = float(score)
            except:
                result[f"silhouette_k{k}"] = float(0)

        # 최적 k 찾기
        best_k = max(range(3, 9), key=lambda k: result[f"silhouette_k{k}"])
        result["best_k"] = best_k
        result["best_silhouette_score"] = result[f"silhouette_k{best_k}"]

        return result

    def _calculate_cohesiveness_score(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        번호 조합의 응집성 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 응집성 점수 분석 결과
        """
        # 번호 쌍 출현 횟수 계산
        pair_counter = Counter()

        for draw in historical_data:
            numbers = sorted(draw.numbers)
            # 모든 가능한 쌍 생성
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = (numbers[i], numbers[j])
                    pair_counter[pair] += 1

        # 각 당첨 조합의 응집성 점수 계산
        cohesiveness_scores = []

        for draw in historical_data:
            numbers = sorted(draw.numbers)

            # 조합 내 모든 쌍의 출현 빈도 합계
            pair_frequency_sum = 0
            pair_count = 0

            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = (numbers[i], numbers[j])
                    pair_frequency_sum += pair_counter[pair]
                    pair_count += 1

            # 평균 쌍 빈도 (응집성 점수)
            avg_pair_frequency = (
                pair_frequency_sum / pair_count if pair_count > 0 else 0
            )
            normalized_score = avg_pair_frequency / len(historical_data)

            cohesiveness_scores.append(normalized_score)

        # 결과 취합
        result = {
            "avg_cohesiveness": float(np.mean(cohesiveness_scores)),
            "median_cohesiveness": float(np.median(cohesiveness_scores)),
            "std_cohesiveness": float(np.std(cohesiveness_scores)),
            "min_cohesiveness": float(min(cohesiveness_scores)),
            "max_cohesiveness": float(max(cohesiveness_scores)),
        }

        # 백분위수 계산
        percentiles = [10, 25, 50, 75, 90, 95]
        for p in percentiles:
            result[f"percentile_{p}"] = float(np.percentile(cohesiveness_scores, p))

        return result

    def _calculate_consecutiveness_score(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        번호 조합의 연속성 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 연속성 점수 분석 결과
        """
        # 각 당첨 조합의 연속성 점수 계산
        consecutive_counts = []
        consecutive_percentages = []

        for draw in historical_data:
            numbers = sorted(draw.numbers)

            # 연속 번호 카운트
            consecutive_count = 0
            for i in range(1, len(numbers)):
                if numbers[i] == numbers[i - 1] + 1:
                    consecutive_count += 1

            # 연속 번호 비율
            consecutive_percentage = consecutive_count / (len(numbers) - 1)

            consecutive_counts.append(consecutive_count)
            consecutive_percentages.append(consecutive_percentage)

        # 결과 취합
        result = {
            "avg_consecutive_count": float(np.mean(consecutive_counts)),
            "median_consecutive_count": float(np.median(consecutive_counts)),
            "std_consecutive_count": float(np.std(consecutive_counts)),
            "max_consecutive_count": float(max(consecutive_counts)),
            "avg_consecutive_percentage": float(np.mean(consecutive_percentages)),
            "median_consecutive_percentage": float(np.median(consecutive_percentages)),
            "std_consecutive_percentage": float(np.std(consecutive_percentages)),
        }

        # 연속 번호 개수별 분포
        consecutive_counter = Counter(consecutive_counts)
        total_draws = len(historical_data)

        for count, frequency in consecutive_counter.items():
            result[f"consecutive_{count}_count"] = frequency
            result[f"consecutive_{count}_percentage"] = float(frequency / total_draws)

        return result

    def _calculate_triplet_frequency(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        번호 삼중항의 출현 빈도를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 번호 삼중항 빈도 분석 결과
        """
        # 번호 삼중항 출현 횟수 계산
        triplet_counter = Counter()

        for draw in historical_data:
            numbers = sorted(draw.numbers)
            # 모든 가능한 삼중항 생성
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    for k in range(j + 1, len(numbers)):
                        triplet = (numbers[i], numbers[j], numbers[k])
                        triplet_counter[triplet] += 1

        # 결과 취합
        result = {}

        # 상위 빈출 삼중항 (상위 10개)
        top_triplets = triplet_counter.most_common(10)
        for i, ((num1, num2, num3), count) in enumerate(top_triplets):
            triplet_key = f"top_triplet_{i+1}"
            result[triplet_key] = f"{num1}-{num2}-{num3}"
            result[f"{triplet_key}_count"] = count
            result[f"{triplet_key}_percentage"] = float(count / len(historical_data))

        # 전체 삼중항 평균 출현 빈도
        total_triplets = len(triplet_counter)
        total_frequency = sum(triplet_counter.values())
        avg_frequency = total_frequency / total_triplets if total_triplets > 0 else 0

        result["total_triplet_types"] = total_triplets
        result["avg_triplet_frequency"] = float(avg_frequency / len(historical_data))

        # 이론적으로 가능한 모든 삼중항 (45C3 = 14,190개)
        theoretical_triplets = 14190
        coverage = total_triplets / theoretical_triplets

        result["triplet_coverage"] = float(coverage)

        return result

    def _analyze_number_network(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        번호 네트워크를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 네트워크 분석 결과
        """
        # 번호 쌍 출현 횟수 계산
        pair_counter = Counter()

        for draw in historical_data:
            numbers = sorted(draw.numbers)
            # 모든 가능한 쌍 생성
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = (numbers[i], numbers[j])
                    pair_counter[pair] += 1

        # 그래프 생성
        G = nx.Graph()

        # 노드 추가 (1-45 모든 번호)
        for num in range(1, 46):
            G.add_node(num)

        # 엣지 추가 (상위 30% 빈도 쌍만)
        threshold = np.percentile(list(pair_counter.values()), 70)

        for (num1, num2), count in pair_counter.items():
            if count >= threshold:
                # 정규화된 가중치 (0-1 범위)
                weight = count / len(historical_data)
                G.add_edge(num1, num2, weight=weight)

        # 네트워크 지표 계산
        result = {}

        # 밀도 (density) - 가능한 모든 연결 중 실제 연결의 비율
        density = nx.density(G)
        result["network_density"] = float(density)

        # 전이성 (transitivity) - 삼각형 구조의 비율
        transitivity = nx.transitivity(G)
        result["network_transitivity"] = float(transitivity)

        # 평균 군집 계수 (average clustering coefficient)
        avg_clustering = nx.average_clustering(G, weight="weight")
        result["avg_clustering_coefficient"] = float(avg_clustering)

        # 연결된 컴포넌트 수
        connected_components = list(nx.connected_components(G))
        result["connected_component_count"] = len(connected_components)

        # 가장 큰 컴포넌트의 크기
        if connected_components:
            largest_component = max(connected_components, key=len)
            result["largest_component_size"] = len(largest_component)
            result["largest_component_ratio"] = float(len(largest_component) / 45)
        else:
            result["largest_component_size"] = 0
            result["largest_component_ratio"] = 0.0

        # 컴포넌트별 크기 분포
        component_sizes = [len(c) for c in connected_components]
        if component_sizes:
            result["avg_component_size"] = float(np.mean(component_sizes))
            result["median_component_size"] = float(np.median(component_sizes))
        else:
            result["avg_component_size"] = 0.0
            result["median_component_size"] = 0.0

        return result

    def _analyze_range_distribution(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        번호 범위 분포를 분석합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 범위 분포 분석 결과
        """
        # 번호 범위 구간 정의
        ranges = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)]

        # 각 당첨 조합의 범위 분포 계산
        range_distributions = []

        for draw in historical_data:
            range_counts = [0] * len(ranges)

            for num in draw.numbers:
                for i, (start, end) in enumerate(ranges):
                    if start <= num <= end:
                        range_counts[i] += 1
                        break

            range_distributions.append(range_counts)

        # 결과 취합
        result = {}

        # 각 범위별 평균 출현 개수
        range_distributions = np.array(range_distributions)
        for i, (start, end) in enumerate(ranges):
            range_key = f"range_{start}_{end}"

            result[f"{range_key}_avg"] = float(np.mean(range_distributions[:, i]))
            result[f"{range_key}_std"] = float(np.std(range_distributions[:, i]))

            # 해당 범위에서 0, 1, 2, 3개 이상 번호가 나오는 비율
            for count in range(4):
                count_pct = np.sum(range_distributions[:, i] == count) / len(
                    historical_data
                )
                result[f"{range_key}_count{count}"] = float(count_pct)

        # 범위 분포 패턴 빈도
        pattern_counter = Counter(tuple(dist) for dist in range_distributions)
        total_draws = len(historical_data)

        # 상위 5개 패턴
        top_patterns = pattern_counter.most_common(5)
        for i, (pattern, count) in enumerate(top_patterns):
            pattern_str = "-".join(str(c) for c in pattern)
            result[f"top_pattern_{i+1}"] = pattern_str
            result[f"top_pattern_{i+1}_count"] = count
            result[f"top_pattern_{i+1}_percentage"] = float(count / total_draws)

        # 균형 점수 - 모든 범위에 고르게 분포된 정도
        balance_scores = []

        for dist in range_distributions:
            # 이상적인 균형: 모든 범위에 균등하게 분포 (엔트로피 최대화)
            # 정규화된 엔트로피 계산
            distribution = np.array(dist) / sum(dist)
            entropy = -np.sum(distribution * np.log2(distribution + 1e-10))
            max_entropy = np.log2(len(ranges))
            balance_score = entropy / max_entropy

            balance_scores.append(balance_score)

        result["avg_balance_score"] = float(np.mean(balance_scores))
        result["median_balance_score"] = float(np.median(balance_scores))
        result["std_balance_score"] = float(np.std(balance_scores))

        return result
