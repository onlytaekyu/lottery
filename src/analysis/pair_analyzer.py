"""
번호 쌍 분석기 모듈

이 모듈은 로또 번호 쌍의 패턴을 분석하는 기능을 제공합니다.
주로 중심성 지표, ROI 기반 패턴, 그래프 가중치 계산 등을 수행합니다.
"""

import os
import json
import datetime
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from collections import Counter, defaultdict

from ..utils.error_handler_refactored import get_logger
from ..shared.types import LotteryNumber
from ..analysis.base_analyzer import BaseAnalyzer
from ..utils.unified_config import ConfigProxy
from ..shared.graph_utils import calculate_pair_frequency, calculate_pair_centrality

# 로그 설정
logger = get_logger(__name__)


class PairAnalyzer(BaseAnalyzer[Dict[str, Any]]):
    """로또 번호 쌍의 패턴을 분석하는 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        PairAnalyzer 초기화

        Args:
            config: 번호 쌍 분석에 사용할 설정
        """
        super().__init__(config or {}, "pair")
        self.logger = get_logger(__name__)

        # ConfigProxy로 변환
        if not isinstance(self.config, ConfigProxy):
            self.config = ConfigProxy(self.config)

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        실제 번호 쌍 분석을 구현하는 내부 메서드 (BaseAnalyzer 추상 메서드 구현)

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 분석 매개변수

        Returns:
            Dict[str, Any]: 분석 결과
        """
        with performance_monitor("pair_analyze"):
            self.logger.info("번호 쌍 분석 시작...")

            # 쌍 빈도 계산 (graph_utils 사용)
            with performance_monitor("calculate_pair_frequency"):
                pair_freq_tuples = calculate_pair_frequency(
                    historical_data, logger=self.logger
                )

                # 문자열 키 형식으로 변환 (기존 코드와 호환성 유지)
                pair_frequency = {}
                for (num1, num2), freq in pair_freq_tuples.items():
                    pair_key = f"{num1}-{num2}"
                    pair_frequency[pair_key] = int(
                        freq * len(historical_data)
                    )  # 원래 빈도 수로 복원

                self.logger.info(f"번호 쌍 빈도 계산 완료: {len(pair_frequency)}개 쌍")

            # 쌍 중심성 계산
            with performance_monitor("calculate_pair_centrality"):
                # 튜플 키로 변환
                pair_freq_for_centrality = {}
                for pair_key, count in pair_frequency.items():
                    num1, num2 = map(int, pair_key.split("-"))
                    pair_freq_for_centrality[(num1, num2)] = count / len(
                        historical_data
                    )  # 정규화된 빈도로 변환

                # graph_utils의 calculate_pair_centrality 사용
                node_centrality = calculate_pair_centrality(
                    pair_freq_for_centrality, logger=self.logger
                )

                # 엣지 중심성으로 변환
                pair_centrality = {}

                # 그래프 재구성
                G = nx.Graph()

                # 노드와 간선 추가
                for (num1, num2), weight in pair_freq_for_centrality.items():
                    if not G.has_node(num1):
                        G.add_node(num1)
                    if not G.has_node(num2):
                        G.add_node(num2)
                    G.add_edge(num1, num2, weight=weight)

                # 엣지 중심성 계산
                for edge in G.edges():
                    num1, num2 = edge
                    # 쌍의 중심성 = 두 노드 중심성의 평균
                    edge_centrality = (
                        node_centrality.get(num1, 0.0) + node_centrality.get(num2, 0.0)
                    ) / 2

                    # 문자열 키로 저장 (JSON 직렬화를 위해)
                    pair_key = f"({min(num1, num2)},{max(num1, num2)})"
                    pair_centrality[pair_key] = float(edge_centrality)

                # 결과 정규화 (0-1 범위)
                if pair_centrality:
                    max_centrality = max(pair_centrality.values())
                    if max_centrality > 0:
                        pair_centrality = {
                            k: v / max_centrality for k, v in pair_centrality.items()
                        }

                # 중심성 상위 50개만 선택
                top_pairs = sorted(
                    pair_centrality.items(), key=lambda x: x[1], reverse=True
                )[:50]
                pair_centrality = {k: v for k, v in top_pairs}

                self.logger.info(f"쌍 중심성 분석 완료 (상위 50개 쌍)")

            # 쌍 ROI 점수 계산
            pair_roi_score = self.calculate_pair_roi_scores(historical_data)

            # 빈번한 3개 번호 조합 계산
            frequent_triples = self.calculate_frequent_triples(historical_data)

            # 번호 클러스터 그룹 계산
            pair_cluster_group_id = self.calculate_number_clusters(historical_data)

            # 쌍 그래프 가중치 계산
            pair_graph_weights = self.calculate_pair_graph_weights(
                pair_frequency, pair_roi_score
            )

            self.logger.info("번호 쌍 분석 완료")

            return {
                "pair_frequency": pair_frequency,
                "pair_centrality": pair_centrality,
                "pair_roi_score": pair_roi_score,
                "frequent_triples": frequent_triples["frequent_triples"],
                "pair_cluster_group_id": pair_cluster_group_id,
                "pair_graph_weights": pair_graph_weights["pair_graph_weights"],
                "graph_stats": pair_graph_weights["graph_stats"],
                "strong_pairs": pair_graph_weights["strong_pairs"],
                "triple_patterns": frequent_triples["triple_patterns"],
                "recent_top_triples": frequent_triples.get("recent_top_triples", []),
            }

    def calculate_pair_roi_scores(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, float]:
        """
        각 번호 쌍의 ROI 점수를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, float]: 번호 쌍별 ROI 점수
        """
        with performance_monitor("calculate_pair_roi_scores"):
            # 1. 각 번호 쌍의 출현 회차 기록
            pair_appearances = defaultdict(list)
            for turn, draw in enumerate(historical_data):
                numbers = draw.numbers
                for i in range(len(numbers)):
                    for j in range(i + 1, len(numbers)):
                        num1, num2 = min(numbers[i], numbers[j]), max(
                            numbers[i], numbers[j]
                        )
                        pair = (num1, num2)
                        pair_appearances[pair].append(turn)

            # 2. 각 번호 쌍의 ROI 계산
            pair_roi_scores = {}
            total_turns = len(historical_data)

            for pair, appearances in pair_appearances.items():
                if len(appearances) < 2:
                    continue

                # 출현 간격 계산
                gaps = [
                    appearances[i] - appearances[i - 1]
                    for i in range(1, len(appearances))
                ]
                avg_gap = sum(gaps) / len(gaps) if gaps else 0

                # 기대 ROI 계산 - 간단한 모델링: 출현 빈도와 일관성 기반
                frequency = len(appearances) / total_turns
                consistency = 1 - (np.std(gaps) / avg_gap if avg_gap > 0 else 1)
                roi_score = frequency * (0.7 + 0.3 * consistency)

                # 문자열 키로 저장
                pair_key = f"({pair[0]},{pair[1]})"
                pair_roi_scores[pair_key] = float(roi_score)

            # 3. 결과 정규화 (0-1 범위)
            if pair_roi_scores:
                max_roi = max(pair_roi_scores.values())
                if max_roi > 0:
                    pair_roi_scores = {
                        k: v / max_roi for k, v in pair_roi_scores.items()
                    }

            # 4. ROI 상위 100개만 선택
            top_pairs = sorted(
                pair_roi_scores.items(), key=lambda x: x[1], reverse=True
            )[:100]
            pair_roi_scores = {k: v for k, v in top_pairs}

            self.logger.info(f"쌍 ROI 점수 분석 완료 (상위 100개 쌍)")

            return pair_roi_scores

    def calculate_pair_graph_weights(
        self, pair_frequency: Dict[str, int], pair_roi_score: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        번호 쌍 그래프의 가중치를 계산합니다.

        Args:
            pair_frequency: 번호 쌍의 출현 빈도
            pair_roi_score: 번호 쌍의 ROI 점수

        Returns:
            Dict[str, Any]: 쌍 그래프 가중치 및 통계
        """
        with performance_monitor("calculate_pair_graph_weights"):
            # 1. 그래프 생성
            G = nx.Graph()

            # 모든 로또 번호를 노드로 추가
            for i in range(1, 46):
                G.add_node(i)

            # 2. 엣지(번호 쌍) 추가 및 가중치 계산
            pair_counts = defaultdict(int)
            for pair, count in pair_frequency.items():
                num1, num2 = map(int, pair.split("-"))
                pair_counts[tuple(sorted((num1, num2)))] += count

            # 3. 가중치 정규화 및 그래프에 추가
            total_draws = sum(pair_counts.values()) / 15  # 총 쌍 수로 나누어 정규화
            edge_data = {}

            for pair, count in pair_counts.items():
                num1, num2 = pair
                frequency = count / total_draws  # 출현 빈도로 정규화

                # ROI 점수 가져오기 (없으면 기본값 0.5 사용)
                roi_key = f"({num1},{num2})"
                roi_score = pair_roi_score.get(roi_key, 0.5)

                # 가중치 계산: 빈도와 ROI 점수의 가중 평균
                alpha = 0.6  # 빈도 가중치
                beta = 0.4  # ROI 가중치
                combined_weight = alpha * frequency + beta * roi_score

                # 그래프에 엣지 추가
                G.add_edge(
                    num1,
                    num2,
                    weight=combined_weight,
                    frequency=frequency,
                    roi_score=roi_score,
                )

                # 결과 저장
                edge_data[roi_key] = {
                    "frequency": float(frequency),
                    "roi_score": float(roi_score),
                    "combined_weight": float(combined_weight),
                }

            # 4. 그래프 통계 계산
            density = nx.density(G)
            avg_clustering = nx.average_clustering(G, weight="weight")

            # 가중치 분포 통계
            weight_values = [data["weight"] for _, _, data in G.edges(data=True)]
            weight_stats = {
                "min": float(min(weight_values)) if weight_values else 0,
                "max": float(max(weight_values)) if weight_values else 0,
                "mean": float(np.mean(weight_values)) if weight_values else 0,
                "median": float(np.median(weight_values)) if weight_values else 0,
                "std": float(np.std(weight_values)) if weight_values else 0,
            }

            # 5. 연결 강도가 가장 높은 쌍(엣지) 선택
            top_edges = sorted(
                [(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
                key=lambda x: x[2],
                reverse=True,
            )[:50]

            strong_pairs = {f"({u},{v})": float(w) for u, v, w in top_edges}

            # 6. 인접 행렬(adjacency matrix) 추출
            adj_matrix_sparse = nx.to_scipy_sparse_array(G, weight="weight")

            # 7. [새로 추가] 45x45 크기의 정규화된 행렬 생성 (GNN/RL 모델용)
            pair_graph_matrix = np.zeros((45, 45), dtype=np.float32)

            # 그래프의 모든 엣지를 행렬에 반영
            for u, v, data in G.edges(data=True):
                # 인덱스는 0부터 시작하므로 -1 조정
                pair_graph_matrix[u - 1, v - 1] = data["weight"]
                pair_graph_matrix[v - 1, u - 1] = data["weight"]  # 대칭성 유지

            # 대각 요소(자기 자신과의 연결)는 0으로 설정
            np.fill_diagonal(pair_graph_matrix, 0.0)

            # 행렬 정규화 (로그 스케일링 적용)
            # 0인 값은 유지하고, 0이 아닌 값에 대해서만 로그 스케일링 적용
            non_zero_mask = pair_graph_matrix > 0
            if np.any(non_zero_mask):
                # 로그 스케일링 (log(1+x))
                pair_graph_matrix[non_zero_mask] = np.log1p(
                    pair_graph_matrix[non_zero_mask]
                )

                # 0-1 범위로 다시 정규화
                if np.max(pair_graph_matrix) > 0:
                    pair_graph_matrix = pair_graph_matrix / np.max(pair_graph_matrix)

            # 8. [새로 추가] 저차원 벡터 생성 (차원 축소)
            try:
                from sklearn.decomposition import PCA

                # 45x45 행렬을 1차원으로 펼침
                flattened_matrix = pair_graph_matrix.flatten()

                # PCA를 사용하여 차원 축소 (128차원)
                pca_dim = 128
                if len(flattened_matrix) > pca_dim:
                    pca = PCA(n_components=pca_dim, random_state=42)
                    compressed_vector = pca.fit_transform(
                        pair_graph_matrix.reshape(1, -1)
                    )[0]
                else:
                    # 행렬이 이미 충분히 작다면 그대로 사용
                    compressed_vector = flattened_matrix

                # 엣지 리스트 형식으로 변환 (PyTorch Geometric / DGL 호환용)
                edge_list = []
                for u in range(45):
                    for v in range(u + 1, 45):  # 중복 방지를 위해 상삼각 행렬만 처리
                        if pair_graph_matrix[u, v] > 0:
                            edge_list.append((u, v, float(pair_graph_matrix[u, v])))

            except Exception as e:
                self.logger.warning(f"차원 축소 중 오류 발생: {e}")
                compressed_vector = np.zeros(128, dtype=np.float32)
                edge_list = []

            # 결과 저장
            result = {
                "graph_stats": {
                    "density": float(density),
                    "avg_clustering": float(avg_clustering),
                    "node_count": G.number_of_nodes(),
                    "edge_count": G.number_of_edges(),
                    "weight_stats": weight_stats,
                },
                "strong_pairs": strong_pairs,
                "pair_graph_weights": edge_data,  # 모든 엣지 데이터 추가
                "pair_graph_matrix": pair_graph_matrix.tolist(),  # 45x45 행렬 추가
                "pair_graph_compressed_vector": compressed_vector.tolist(),  # 저차원 벡터 추가
                "pair_graph_edge_list": edge_list,  # 엣지 리스트 추가
            }

            self.logger.info(
                f"쌍 그래프 가중치 분석 완료 (밀도: {density:.4f}, 평균 클러스터링: {avg_clustering:.4f})"
            )

            # 인접 행렬과 압축 벡터를 NumPy 배열로 저장
            try:
                import os

                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                cache_dir = Path(project_root) / self.config["paths"]["cache_dir"]
                graph_dir = cache_dir / "graph_inputs"
                graph_dir.mkdir(parents=True, exist_ok=True)

                # 원본 인접 행렬 저장
                adj_matrix = adj_matrix_sparse.toarray()
                np.save(cache_dir / "pair_adjacency_matrix.npy", adj_matrix)

                # GNN/RL 모델용 행렬 저장
                np.save(graph_dir / "pair_graph_weights.npy", pair_graph_matrix)

                # 압축 벡터 저장
                np.save(graph_dir / "pair_graph_compressed.npy", compressed_vector)

                # 엣지 리스트 저장
                with open(
                    graph_dir / "pair_graph_edge_list.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(edge_list, f)

                self.logger.info(f"그래프 행렬 및 벡터 저장 완료: {graph_dir}")
            except Exception as e:
                self.logger.error(f"그래프 데이터 저장 중 오류 발생: {e}")

            return result

    def calculate_frequent_triples(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        빈번하게 함께 출현하는 3개 번호 조합을 계산

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 빈번한 3개 번호 조합 정보
        """
        with performance_monitor("calculate_frequent_triples"):
            self.logger.info("빈번한 3개 번호 조합 분석 중...")

            # 결과 저장용 딕셔너리
            result = {
                "triple_frequency": {},
                "triple_win_ratio": {},
                "triple_patterns": {},
            }

            # 3개 번호 조합 빈도 계산
            triple_count = {}
            total_draws = len(historical_data)

            # 모든 회차에서 3개 번호 조합 추출
            for draw in historical_data:
                numbers = draw.numbers

                # 가능한 모든 3개 조합 생성
                for i in range(len(numbers)):
                    for j in range(i + 1, len(numbers)):
                        for k in range(j + 1, len(numbers)):
                            # 세 번호를 정렬하여 키 생성
                            triple = (numbers[i], numbers[j], numbers[k])
                            triple_key = f"{numbers[i]}-{numbers[j]}-{numbers[k]}"
                            triple_count[triple_key] = (
                                triple_count.get(triple_key, 0) + 1
                            )

            # 결과를 빈도 기준으로 정렬
            sorted_triples = sorted(
                triple_count.items(), key=lambda x: x[1], reverse=True
            )

            # 상위 10개 조합 선택 (요구사항에 맞게 포맷 변경)
            top_triples = {}
            for idx, (triple_key, count) in enumerate(sorted_triples[:10]):
                numbers = list(map(int, triple_key.split("-")))
                triplet_id = f"triplet_{idx+1}"
                top_triples[triplet_id] = numbers

            # 상위 50개 조합의 세부 정보 (기존 분석 유지)
            detailed_triples = []
            for triple_key, count in sorted_triples[:50]:
                numbers = tuple(map(int, triple_key.split("-")))
                frequency = count / total_draws

                # 승률 계산 (최근 50회차 기준)
                recent_wins = 0
                recent_draws = (
                    historical_data[-50:]
                    if len(historical_data) >= 50
                    else historical_data
                )

                for draw in recent_draws:
                    draw_set = set(draw.numbers)
                    if all(num in draw_set for num in numbers):
                        recent_wins += 1

                win_ratio = recent_wins / len(recent_draws) if recent_draws else 0

                detailed_triples.append(
                    {
                        "numbers": numbers,
                        "frequency": frequency,
                        "count": count,
                        "win_ratio": win_ratio,
                    }
                )

            result["detailed_triples"] = detailed_triples

            # 3개 번호 조합의 다양한 패턴 분석
            pattern_types = {
                "consecutive": 0,  # 연속된 3개 번호
                "same_segment": 0,  # 같은 세그먼트에 있는 3개 번호
                "arithmetic_seq": 0,  # 등차수열 형태 (예: 1-3-5, 10-20-30)
                "odd_heavy": 0,  # 홀수가 2개 이상
                "even_heavy": 0,  # 짝수가 2개 이상
                "high_variance": 0,  # 번호 간 편차가 큰 경우
                "low_variance": 0,  # 번호 간 편차가 작은 경우
            }

            # 상위 조합에서 패턴 분석
            for triple in detailed_triples:
                numbers = triple["numbers"]

                # 연속 번호 검사
                if numbers[1] == numbers[0] + 1 and numbers[2] == numbers[1] + 1:
                    pattern_types["consecutive"] += 1

                # 같은 세그먼트 검사
                segments = [self._get_segment(num) for num in numbers]
                if segments[0] == segments[1] == segments[2]:
                    pattern_types["same_segment"] += 1

                # 등차수열 검사
                if numbers[1] - numbers[0] == numbers[2] - numbers[1]:
                    pattern_types["arithmetic_seq"] += 1

                # 홀수/짝수 비율 검사
                odd_count = sum(1 for n in numbers if n % 2 == 1)
                if odd_count >= 2:
                    pattern_types["odd_heavy"] += 1
                else:
                    pattern_types["even_heavy"] += 1

                # 분산 검사
                variance = np.var(numbers)
                if variance > 100:
                    pattern_types["high_variance"] += 1
                elif variance < 20:
                    pattern_types["low_variance"] += 1

            result["triple_patterns"] = pattern_types

            # 최근 10회차에서 나온 조합 분석
            recent_draws = (
                historical_data[-10:] if len(historical_data) >= 10 else historical_data
            )
            recent_triples = set()

            for draw in recent_draws:
                numbers = draw.numbers
                for i in range(len(numbers)):
                    for j in range(i + 1, len(numbers)):
                        for k in range(j + 1, len(numbers)):
                            triple = (numbers[i], numbers[j], numbers[k])
                            recent_triples.add(triple)

            # 최근 나온 조합 중 가장 빈번한 조합 선택
            recent_triple_counts = {
                triple: triple_count.get(f"{triple[0]}-{triple[1]}-{triple[2]}", 0)
                for triple in recent_triples
            }

            top_recent_triples = sorted(
                recent_triple_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]

            result["recent_top_triples"] = [
                {"numbers": triple, "count": count}
                for triple, count in top_recent_triples
            ]

            # 요구사항에 맞는 형식으로 반환 값 설정
            return {**result, "frequent_triples": top_triples}  # 요구사항 형식

    def _is_prime(self, n: int) -> bool:
        """
        주어진 숫자가 소수인지 확인

        Args:
            n: 확인할 숫자

        Returns:
            bool: 소수 여부
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def _get_segment(self, number: int) -> int:
        """
        번호가 속한 세그먼트(1~9) 반환

        Args:
            number: 확인할 번호 (1~45)

        Returns:
            int: 세그먼트 번호 (1~9)
        """
        if 1 <= number <= 5:
            return 1
        elif 6 <= number <= 10:
            return 2
        elif 11 <= number <= 15:
            return 3
        elif 16 <= number <= 20:
            return 4
        elif 21 <= number <= 25:
            return 5
        elif 26 <= number <= 30:
            return 6
        elif 31 <= number <= 35:
            return 7
        elif 36 <= number <= 40:
            return 8
        elif 41 <= number <= 45:
            return 9
        else:
            return 0  # 잘못된 번호

    def identify_roi_pattern_groups(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        ROI 기반 패턴 그룹 식별

        과거 데이터를 분석하여 ROI가 높은 패턴 그룹을 식별합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            Dict[str, Any]: 패턴 그룹 정보
        """
        with performance_monitor("identify_roi_pattern_groups"):
            self.logger.info("ROI 패턴 그룹 식별 중...")

            # 결과 저장용 딕셔너리
            result = {
                "high_roi_groups": [],
                "pattern_groups": {},
                "group_statistics": {},
            }

            # 충분한 데이터가 있는지 확인
            if len(historical_data) < 50:
                self.logger.warning(
                    "ROI 패턴 그룹 식별을 위한 충분한 데이터가 없습니다. (최소 50회차 필요)"
                )
                return result

            # 패턴 그룹 정의 (각 패턴은 번호 특성을 기반으로 함)
            pattern_groups = {
                "odd_dominant": [],  # 홀수 우세 패턴
                "even_dominant": [],  # 짝수 우세 패턴
                "low_numbers": [],  # 낮은 번호 우세 패턴
                "high_numbers": [],  # 높은 번호 우세 패턴
                "balanced_segments": [],  # 균형 있는 구간 분포
                "consecutive_numbers": [],  # 연속 번호 포함
                "wide_range": [],  # 넓은 번호 범위
                "narrow_range": [],  # 좁은 번호 범위
                "prime_heavy": [],  # 소수 우세 패턴
                "multiples_of_3": [],  # 3의 배수 우세 패턴
            }

            # 각 회차를 패턴 그룹으로 분류
            for draw in historical_data:
                numbers = sorted(draw.numbers)

                # 홀수/짝수 분류
                odd_count = sum(1 for n in numbers if n % 2 == 1)
                if odd_count >= 4:
                    pattern_groups["odd_dominant"].append(draw.draw_no)
                elif odd_count <= 2:
                    pattern_groups["even_dominant"].append(draw.draw_no)

                # 번호 범위 분류
                low_count = sum(1 for n in numbers if n <= 22)
                if low_count >= 4:
                    pattern_groups["low_numbers"].append(draw.draw_no)
                elif low_count <= 2:
                    pattern_groups["high_numbers"].append(draw.draw_no)

                # 구간 분포 분류
                segments = [0] * 5
                for num in numbers:
                    if num <= 9:
                        segments[0] += 1
                    elif num <= 18:
                        segments[1] += 1
                    elif num <= 27:
                        segments[2] += 1
                    elif num <= 36:
                        segments[3] += 1
                    else:
                        segments[4] += 1

                # 균형 있는 구간 분포 여부
                if max(segments) <= 2 and min(segments) > 0:
                    pattern_groups["balanced_segments"].append(draw.draw_no)

                # 연속 번호 포함 여부
                has_consecutive = False
                for i in range(len(numbers) - 1):
                    if numbers[i + 1] == numbers[i] + 1:
                        has_consecutive = True
                        break

                if has_consecutive:
                    pattern_groups["consecutive_numbers"].append(draw.draw_no)

                # 번호 범위
                range_width = numbers[-1] - numbers[0]
                if range_width >= 38:
                    pattern_groups["wide_range"].append(draw.draw_no)
                elif range_width <= 28:
                    pattern_groups["narrow_range"].append(draw.draw_no)

                # 소수 패턴
                prime_count = sum(1 for n in numbers if self._is_prime(n))
                if prime_count >= 3:
                    pattern_groups["prime_heavy"].append(draw.draw_no)

                # 3의 배수 패턴
                multiples_of_3_count = sum(1 for n in numbers if n % 3 == 0)
                if multiples_of_3_count >= 3:
                    pattern_groups["multiples_of_3"].append(draw.draw_no)

            # 각 패턴 그룹의 ROI 계산
            group_statistics = {}

            for group_name, draw_numbers in pattern_groups.items():
                # 그룹에 속한 회차 수가 적으면 건너뜀
                if len(draw_numbers) < 10:
                    group_statistics[group_name] = {
                        "count": len(draw_numbers),
                        "frequency": len(draw_numbers) / len(historical_data),
                        "roi_estimate": 0.0,
                        "recent_performance": 0.0,
                    }
                    continue

                # 해당 그룹의 빈도
                frequency = len(draw_numbers) / len(historical_data)

                # ROI 추정 (단순화된 모델)
                # 실제 ROI는 당첨 확률 및 상금에 따라 달라짐
                # 여기서는 빈도와 패턴 일관성을 기반으로 추정

                # 최근 성능 (최근 20회차 기준)
                recent_draws = set(draw.draw_no for draw in historical_data[-20:])
                recent_hits = len([d for d in draw_numbers if d in recent_draws])
                recent_performance = recent_hits / 20 if recent_draws else 0

                # ROI 추정 (0.5가 손익분기점)
                roi_estimate = 0.5 + (recent_performance - frequency) * 2
                roi_estimate = max(0.0, min(1.0, roi_estimate))

                group_statistics[group_name] = {
                    "count": len(draw_numbers),
                    "frequency": frequency,
                    "roi_estimate": roi_estimate,
                    "recent_performance": recent_performance,
                }

            # ROI가 높은 그룹 식별 (ROI 0.6 이상)
            high_roi_groups = []

            for group_name, stats in group_statistics.items():
                if stats["roi_estimate"] >= 0.6:
                    high_roi_groups.append(
                        {
                            "name": group_name,
                            "roi": stats["roi_estimate"],
                            "count": stats["count"],
                            "recent_performance": stats["recent_performance"],
                        }
                    )

            # ROI 기준으로 정렬
            high_roi_groups.sort(key=lambda x: x["roi"], reverse=True)

            # 결과 저장
            result["high_roi_groups"] = high_roi_groups
            result["pattern_groups"] = {
                k: v for k, v in pattern_groups.items() if len(v) > 0
            }
            result["group_statistics"] = group_statistics

            self.logger.info(
                f"ROI 패턴 그룹 식별 완료: 높은 ROI 그룹 {len(high_roi_groups)}개"
            )
            return result

    def save_pair_centrality(
        self, pair_centrality: Dict[str, float], cache_dir: Path
    ) -> Path:
        """
        번호 쌍 중심성 점수를 파일로 저장

        Args:
            pair_centrality: 번호 쌍 중심성 점수
            cache_dir: 캐시 디렉토리 경로

        Returns:
            Path: 저장된 파일 경로
        """
        with performance_monitor("save_pair_centrality"):
            self.logger.info("번호 쌍 중심성 점수 저장 중...")

            # 디렉토리 생성
            cache_dir.mkdir(parents=True, exist_ok=True)

            # JSON 형식으로 저장
            json_path = cache_dir.joinpath("pair_centrality.json")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pair_centrality, f, ensure_ascii=False, indent=2)

            # NumPy 배열로 변환하여 저장
            # 45x45 행렬 형태로 변환
            matrix = np.zeros((45, 45), dtype=np.float32)

            for pair_key, score in pair_centrality.items():
                # 다양한 키 형식 처리
                try:
                    if "-" in pair_key:
                        # 'num1-num2' 형식
                        num1, num2 = map(int, pair_key.split("-"))
                    elif "(" in pair_key and ")" in pair_key and "," in pair_key:
                        # '(num1,num2)' 형식
                        nums_str = pair_key.strip("()").split(",")
                        num1, num2 = int(nums_str[0]), int(nums_str[1])
                    else:
                        self.logger.warning(f"지원되지 않는 쌍 키 형식: {pair_key}")
                        continue

                    # 0-기반 인덱스로 변환
                    idx1, idx2 = num1 - 1, num2 - 1
                    matrix[idx1, idx2] = score
                    matrix[idx2, idx1] = score  # 대칭 행렬
                except Exception as e:
                    self.logger.warning(f"쌍 키 변환 오류 ({pair_key}): {e}")
                    continue

            # NumPy 배열로 저장
            npy_path = cache_dir.joinpath("pair_centrality.npy")
            np.save(npy_path, matrix)

            self.logger.info(f"번호 쌍 중심성 점수 저장 완료: {json_path}, {npy_path}")
            return npy_path

    def calculate_number_clusters(
        self, historical_data: List[LotteryNumber], n_clusters: int = 5
    ) -> Dict[str, int]:
        """
        각 번호를 클러스터 그룹에 할당합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            n_clusters: 클러스터 그룹 수

        Returns:
            Dict[str, int]: 번호를 키로, 클러스터 ID를 값으로 하는 딕셔너리
        """
        with performance_monitor("calculate_number_clusters"):
            self.logger.info(f"번호 클러스터링 시작 (그룹 수: {n_clusters})...")

            # 1. 각 번호의 빈도 및 ROI 점수 계산
            number_freq = defaultdict(int)
            number_recency = defaultdict(list)

            # 빈도 및 최근성 계산
            for idx, draw in enumerate(historical_data):
                for num in draw.numbers:
                    number_freq[num] += 1
                    number_recency[num].append(idx)

            # 2. 특성 벡터 생성
            features = np.zeros((45, 3))

            for num in range(1, 46):
                # 빈도 특성
                features[num - 1, 0] = number_freq.get(num, 0) / len(historical_data)

                # 최근성 특성
                recency_list = number_recency.get(num, [])
                if recency_list:
                    # 최근 등장 인덱스
                    last_idx = max(recency_list)
                    features[num - 1, 1] = last_idx / len(historical_data)

                    # 평균 간격
                    if len(recency_list) > 1:
                        gaps = [
                            recency_list[i] - recency_list[i - 1]
                            for i in range(1, len(recency_list))
                        ]
                        features[num - 1, 2] = np.mean(gaps) / len(historical_data)

            # 3. 특성 정규화
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # 4. K-means 클러스터링
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)

            # 5. 결과 형식화
            result = {}
            for num in range(1, 46):
                result[str(num)] = int(clusters[num - 1])

            self.logger.info(f"번호 클러스터링 완료: {n_clusters}개 그룹")
            return result
