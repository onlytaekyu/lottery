"""
그래프 네트워크 분석기
- 번호 간 관계를 그래프로 모델링
- 커뮤니티 탐지, 중심성 분석, Node2Vec 임베딩 등을 포함합니다.
"""

import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict

# 기존 시스템 import (순환 참조 방지)
from ..utils.unified_logging import get_logger
from ..shared.types import LotteryNumber
from ..analysis.base_analyzer import BaseAnalyzer

logger = get_logger(__name__)


class GraphNetworkAnalyzer(BaseAnalyzer):
    """
    그래프 네트워크 분석기
    - 번호 간 동시 출현 관계를 그래프로 모델링
    - 커뮤니티 탐지로 번호 그룹 발견
    - 다양한 중심성 지표 계산
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {}, "graph_network")
        self.graph = None
        self.communities = None
        self.centrality_metrics = {}
        self.node_embeddings = {}

        # 설정 초기화
        self.min_co_occurrence = self.config.get("min_co_occurrence", 2)
        self.use_node2vec = self.config.get("use_node2vec", False)
        self.embedding_dim = self.config.get("embedding_dim", 64)

        logger.info("✅ GraphNetworkAnalyzer 초기화 완료")

    def _analyze_impl(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """그래프 네트워크 분석 메인 로직"""
        try:
            logger.info(f"🔗 그래프 네트워크 분석 시작: {len(data)}개 회차")

            # 1. 번호 공출현 그래프 생성
            co_occurrence_graph = self._build_co_occurrence_graph(data)

            # 2. 커뮤니티 탐지
            communities = self._detect_communities(co_occurrence_graph)

            # 3. 중심성 분석
            centrality_analysis = self._analyze_centrality(co_occurrence_graph)

            # 4. 그래프 임베딩 (선택적)
            embeddings = self._generate_graph_embeddings(co_occurrence_graph)

            # 5. 그래프 통계
            graph_statistics = self._calculate_graph_statistics(co_occurrence_graph)

            # 6. 번호별 그래프 특성
            number_graph_features = self._extract_number_graph_features(
                co_occurrence_graph, communities, centrality_analysis
            )

            result = {
                "graph_statistics": graph_statistics,
                "communities": communities,
                "centrality_analysis": centrality_analysis,
                "embeddings": embeddings,
                "number_graph_features": number_graph_features,
                "co_occurrence_matrix": self._graph_to_matrix(co_occurrence_graph),
            }

            # 내부 상태 저장
            self.graph = co_occurrence_graph
            self.communities = communities
            self.centrality_metrics = centrality_analysis
            self.node_embeddings = embeddings

            logger.info("✅ 그래프 네트워크 분석 완료")
            return result

        except Exception as e:
            logger.error(f"그래프 네트워크 분석 실패: {e}")
            return self._get_fallback_result()

    def _build_co_occurrence_graph(self, data: List[LotteryNumber]) -> nx.Graph:
        """번호 공출현 그래프 생성"""
        try:
            # 공출현 횟수 계산
            co_occurrence_counts = defaultdict(int)

            for draw in data:
                numbers = sorted(draw.numbers)
                # 모든 번호 쌍의 공출현 횟수 계산
                for i in range(len(numbers)):
                    for j in range(i + 1, len(numbers)):
                        pair = (numbers[i], numbers[j])
                        co_occurrence_counts[pair] += 1

            # 그래프 생성
            G = nx.Graph()

            # 모든 번호를 노드로 추가
            for num in range(1, 46):
                G.add_node(num)

            # 최소 공출현 횟수 이상인 쌍만 엣지로 추가
            for (num1, num2), count in co_occurrence_counts.items():
                if count >= self.min_co_occurrence:
                    # 가중치는 공출현 횟수
                    weight = count / len(data)  # 정규화
                    G.add_edge(num1, num2, weight=weight, count=count)

            logger.info(
                f"그래프 생성 완료: {G.number_of_nodes()}개 노드, {G.number_of_edges()}개 엣지"
            )
            return G

        except Exception as e:
            logger.error(f"그래프 생성 실패: {e}")
            return nx.Graph()

    def _detect_communities(self, graph: nx.Graph) -> Dict[str, Any]:
        """커뮤니티 탐지 (Louvain 알고리즘)"""
        try:
            communities_result = {}

            # 1. NetworkX 내장 커뮤니티 탐지
            try:
                from networkx.algorithms import community

                communities_greedy = community.greedy_modularity_communities(graph)
                communities_result["greedy"] = [list(c) for c in communities_greedy]

                # 모듈러리티 계산
                modularity = community.modularity(graph, communities_greedy)
                communities_result["modularity"] = modularity

            except ImportError:
                logger.warning("NetworkX 커뮤니티 모듈 없음")

            # 2. Louvain 알고리즘 (선택적)
            try:
                import community as community_louvain

                partition = community_louvain.best_partition(graph)

                # 커뮤니티별 노드 그룹화
                communities_louvain = defaultdict(list)
                for node, comm_id in partition.items():
                    communities_louvain[comm_id].append(node)

                communities_result["louvain"] = list(communities_louvain.values())
                communities_result["louvain_modularity"] = community_louvain.modularity(
                    partition, graph
                )

            except ImportError:
                logger.warning("Louvain 라이브러리 없음, 기본 알고리즘 사용")

            # 3. 커뮤니티 통계
            if "greedy" in communities_result:
                communities_stats = self._analyze_community_statistics(
                    communities_result["greedy"], graph
                )
                communities_result["statistics"] = communities_stats

            logger.info(
                f"커뮤니티 탐지 완료: {len(communities_result.get('greedy', []))}개 커뮤니티"
            )
            return communities_result

        except Exception as e:
            logger.error(f"커뮤니티 탐지 실패: {e}")
            return {"error": str(e)}

    def _analyze_centrality(self, graph: nx.Graph) -> Dict[str, Any]:
        """다양한 중심성 지표 분석"""
        try:
            centrality_result = {}

            # 1. 차수 중심성 (Degree Centrality)
            degree_centrality = nx.degree_centrality(graph)
            centrality_result["degree"] = degree_centrality

            # 2. 근접 중심성 (Closeness Centrality)
            try:
                closeness_centrality = nx.closeness_centrality(graph)
                centrality_result["closeness"] = closeness_centrality
            except:
                logger.warning("근접 중심성 계산 실패")

            # 3. 매개 중심성 (Betweenness Centrality)
            try:
                betweenness_centrality = nx.betweenness_centrality(graph)
                centrality_result["betweenness"] = betweenness_centrality
            except:
                logger.warning("매개 중심성 계산 실패")

            # 4. 고유벡터 중심성 (Eigenvector Centrality)
            try:
                eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=500)
                centrality_result["eigenvector"] = eigenvector_centrality
            except:
                logger.warning("고유벡터 중심성 계산 실패")

            # 5. PageRank
            try:
                pagerank = nx.pagerank(graph)
                centrality_result["pagerank"] = pagerank
            except:
                logger.warning("PageRank 계산 실패")

            # 6. 중심성 통계
            centrality_stats = self._analyze_centrality_statistics(centrality_result)
            centrality_result["statistics"] = centrality_stats

            logger.info(f"중심성 분석 완료: {len(centrality_result)}개 지표")
            return centrality_result

        except Exception as e:
            logger.error(f"중심성 분석 실패: {e}")
            return {"error": str(e)}

    def _generate_graph_embeddings(self, graph: nx.Graph) -> Dict[str, Any]:
        """그래프 임베딩 생성 (Node2Vec)"""
        try:
            embeddings_result = {}

            if not self.use_node2vec:
                logger.info("Node2Vec 사용 안함")
                return embeddings_result

            # Node2Vec 임베딩 (선택적)
            try:
                from node2vec import Node2Vec

                # Node2Vec 모델 생성
                node2vec = Node2Vec(
                    graph,
                    dimensions=self.embedding_dim,
                    walk_length=30,
                    num_walks=200,
                    workers=4,
                )

                # 모델 학습
                model = node2vec.fit(window=10, min_count=1, batch_words=4)

                # 임베딩 벡터 추출
                embeddings = {}
                for node in graph.nodes():
                    try:
                        embeddings[node] = model.wv[str(node)].tolist()
                    except KeyError:
                        embeddings[node] = [0.0] * self.embedding_dim

                embeddings_result["node2vec"] = embeddings
                embeddings_result["embedding_dim"] = self.embedding_dim

                logger.info(f"Node2Vec 임베딩 완료: {len(embeddings)}개 노드")

            except ImportError:
                logger.warning("Node2Vec 라이브러리 없음")
            except Exception as e:
                logger.warning(f"Node2Vec 실행 실패: {e}")

            # 대체 임베딩 방법 (인접 행렬 기반)
            if "node2vec" not in embeddings_result:
                adjacency_embeddings = self._generate_adjacency_embeddings(graph)
                embeddings_result["adjacency"] = adjacency_embeddings

            return embeddings_result

        except Exception as e:
            logger.error(f"그래프 임베딩 생성 실패: {e}")
            return {"error": str(e)}

    def _generate_adjacency_embeddings(self, graph: nx.Graph) -> Dict[int, List[float]]:
        """인접 행렬 기반 임베딩 생성"""
        try:
            # 인접 행렬 생성
            adjacency_matrix = nx.adjacency_matrix(
                graph, nodelist=range(1, 46)
            ).toarray()

            # SVD 분해로 차원 축소
            from sklearn.decomposition import TruncatedSVD

            svd = TruncatedSVD(n_components=min(32, adjacency_matrix.shape[0] - 1))
            embeddings_matrix = svd.fit_transform(adjacency_matrix)

            # 노드별 임베딩 딕셔너리 생성
            embeddings = {}
            for i, node in enumerate(range(1, 46)):
                embeddings[node] = embeddings_matrix[i].tolist()

            logger.info(f"인접 행렬 임베딩 완료: {len(embeddings)}개 노드")
            return embeddings

        except Exception as e:
            logger.error(f"인접 행렬 임베딩 생성 실패: {e}")
            return {}

    def _calculate_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """그래프 통계 계산"""
        try:
            stats = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "is_connected": nx.is_connected(graph),
                "number_of_components": nx.number_connected_components(graph),
            }

            # 연결된 그래프인 경우 추가 통계
            if stats["is_connected"]:
                stats["diameter"] = nx.diameter(graph)
                stats["average_path_length"] = nx.average_shortest_path_length(graph)
                stats["radius"] = nx.radius(graph)

            # 클러스터링 계수
            stats["average_clustering"] = nx.average_clustering(graph)
            stats["transitivity"] = nx.transitivity(graph)

            # 차수 통계
            degrees = dict(graph.degree())
            degree_values = list(degrees.values())
            stats["degree_statistics"] = {
                "mean": np.mean(degree_values),
                "std": np.std(degree_values),
                "max": np.max(degree_values),
                "min": np.min(degree_values),
                "median": np.median(degree_values),
            }

            return stats

        except Exception as e:
            logger.error(f"그래프 통계 계산 실패: {e}")
            return {"error": str(e)}

    def _extract_number_graph_features(
        self, graph: nx.Graph, communities: Dict[str, Any], centrality: Dict[str, Any]
    ) -> Dict[int, Dict[str, float]]:
        """번호별 그래프 특성 추출"""
        try:
            number_features = {}

            for num in range(1, 46):
                features = {}

                # 기본 그래프 특성
                features["degree"] = graph.degree(num)
                features["weighted_degree"] = graph.degree(num, weight="weight")

                # 중심성 지표
                for centrality_type, centrality_values in centrality.items():
                    if centrality_type != "statistics" and isinstance(
                        centrality_values, dict
                    ):
                        features[f"centrality_{centrality_type}"] = (
                            centrality_values.get(num, 0.0)
                        )

                # 클러스터링 계수
                features["clustering_coefficient"] = nx.clustering(graph, num)

                # 커뮤니티 정보
                if "greedy" in communities:
                    community_id = self._find_community_id(num, communities["greedy"])
                    features["community_id"] = community_id
                    features["community_size"] = (
                        len(communities["greedy"][community_id])
                        if community_id >= 0
                        else 0
                    )

                # 이웃 노드 통계
                neighbors = list(graph.neighbors(num))
                features["neighbor_count"] = len(neighbors)
                if neighbors:
                    features["neighbor_avg"] = np.mean(neighbors)
                    features["neighbor_std"] = np.std(neighbors)

                number_features[num] = features

            return number_features

        except Exception as e:
            logger.error(f"번호별 그래프 특성 추출 실패: {e}")
            return {}

    def _find_community_id(self, node: int, communities: List[List[int]]) -> int:
        """노드가 속한 커뮤니티 ID 찾기"""
        for i, community in enumerate(communities):
            if node in community:
                return i
        return -1

    def _analyze_community_statistics(
        self, communities: List[List[int]], graph: nx.Graph
    ) -> Dict[str, Any]:
        """커뮤니티 통계 분석"""
        try:
            stats = {
                "community_count": len(communities),
                "community_sizes": [len(comm) for comm in communities],
                "largest_community_size": max(len(comm) for comm in communities),
                "smallest_community_size": min(len(comm) for comm in communities),
                "average_community_size": np.mean([len(comm) for comm in communities]),
            }

            # 커뮤니티 내 밀도
            community_densities = []
            for community in communities:
                subgraph = graph.subgraph(community)
                community_densities.append(nx.density(subgraph))

            stats["community_densities"] = community_densities
            stats["average_community_density"] = np.mean(community_densities)

            return stats

        except Exception as e:
            logger.error(f"커뮤니티 통계 분석 실패: {e}")
            return {"error": str(e)}

    def _analyze_centrality_statistics(
        self, centrality_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """중심성 통계 분석"""
        try:
            stats = {}

            for centrality_type, centrality_values in centrality_result.items():
                if isinstance(centrality_values, dict):
                    values = list(centrality_values.values())
                    stats[centrality_type] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "max": np.max(values),
                        "min": np.min(values),
                        "median": np.median(values),
                        "top_5_nodes": sorted(
                            centrality_values.items(), key=lambda x: x[1], reverse=True
                        )[:5],
                    }

            return stats

        except Exception as e:
            logger.error(f"중심성 통계 분석 실패: {e}")
            return {"error": str(e)}

    def _graph_to_matrix(self, graph: nx.Graph) -> List[List[float]]:
        """그래프를 인접 행렬로 변환"""
        try:
            adjacency_matrix = nx.adjacency_matrix(
                graph, nodelist=range(1, 46), weight="weight"
            )
            return adjacency_matrix.toarray().tolist()
        except Exception as e:
            logger.error(f"그래프 행렬 변환 실패: {e}")
            return [[0.0] * 45 for _ in range(45)]

    def _get_fallback_result(self) -> Dict[str, Any]:
        """폴백 결과 반환"""
        return {
            "graph_statistics": {"error": "analysis_failed"},
            "communities": {"error": "analysis_failed"},
            "centrality_analysis": {"error": "analysis_failed"},
            "embeddings": {"error": "analysis_failed"},
            "number_graph_features": {},
            "co_occurrence_matrix": [[0.0] * 45 for _ in range(45)],
        }

    def get_graph_features_vector(
        self, number_graph_features: Dict[int, Dict[str, float]]
    ) -> np.ndarray:
        """그래프 특성을 벡터로 변환"""
        try:
            # 특성 이름 정의
            feature_names = [
                "degree",
                "weighted_degree",
                "clustering_coefficient",
                "centrality_degree",
                "centrality_closeness",
                "centrality_betweenness",
                "centrality_eigenvector",
                "centrality_pagerank",
                "community_id",
                "community_size",
                "neighbor_count",
                "neighbor_avg",
                "neighbor_std",
            ]

            # 모든 번호의 특성을 벡터로 변환
            all_features = []
            for num in range(1, 46):
                number_features = number_graph_features.get(num, {})
                feature_vector = []

                for feature_name in feature_names:
                    value = number_features.get(feature_name, 0.0)
                    feature_vector.append(float(value))

                all_features.extend(feature_vector)

            return np.array(all_features, dtype=np.float32)

        except Exception as e:
            logger.error(f"그래프 특성 벡터 변환 실패: {e}")
            return np.zeros(45 * 13, dtype=np.float32)  # 45개 번호 * 13개 특성
