#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
그래프 기반 분석 유틸리티 모듈

로또 번호 간의 관계를 그래프로 모델링하고 분석하는 함수들을 제공합니다.
성능 최적화를 위한 캐싱, 메모리 효율성, 병렬 처리 기능을 포함합니다.
"""

import logging
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional, Union
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 타입 힌트
LotteryNumber = Any  # 실제 LotteryNumber 클래스 가져오기가 어려울 수 있어 Any로 대체

# 성능 최적화를 위한 전역 캐시
_CACHE = {}
_CACHE_SIZE_LIMIT = 1000
_MEMORY_POOL = {}


def _generate_cache_key(data, func_name: str, **kwargs) -> str:
    """데이터와 매개변수를 기반으로 캐시 키를 생성합니다."""
    try:
        # 데이터 해시 생성
        if hasattr(data, "__len__"):
            data_hash = hashlib.md5(f"{len(data)}_{type(data)}".encode()).hexdigest()[
                :8
            ]
        else:
            data_hash = hashlib.md5(str(data).encode()).hexdigest()[:8]

        # 매개변수 해시 생성
        params_str = json.dumps(kwargs, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        return f"{func_name}_{data_hash}_{params_hash}"
    except Exception:
        # 해시 생성 실패 시 기본 키 반환
        return f"{func_name}_default_{hash(str(kwargs)) % 10000}"


def _manage_cache_size():
    """캐시 크기를 관리하고 필요시 정리합니다."""
    global _CACHE
    if len(_CACHE) > _CACHE_SIZE_LIMIT:
        # LRU 방식으로 오래된 항목 제거 (간단한 구현)
        keys_to_remove = list(_CACHE.keys())[: -_CACHE_SIZE_LIMIT // 2]
        for key in keys_to_remove:
            del _CACHE[key]


def cached_analysis(func):
    """분석 함수에 대한 캐싱 데코레이터"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 캐시 키 생성
        cache_key = _generate_cache_key(
            args[0] if args else None, func.__name__, **kwargs
        )

        # 캐시에서 결과 확인
        if cache_key in _CACHE:
            if kwargs.get("logger"):
                kwargs["logger"].debug(f"캐시에서 {func.__name__} 결과 반환")
            return _CACHE[cache_key]

        # 함수 실행
        result = func(*args, **kwargs)

        # 캐시에 저장
        _CACHE[cache_key] = result
        _manage_cache_size()

        if kwargs.get("logger"):
            kwargs["logger"].debug(f"{func.__name__} 결과를 캐시에 저장")

        return result

    return wrapper


@cached_analysis
def calculate_pair_frequency(
    data: List[LotteryNumber],
    logger: Optional[logging.Logger] = None,
    normalize_method: str = "max",
    chunk_size: int = 100,
) -> Dict[Tuple[int, int], float]:
    """
    번호 쌍의 정규화된 동시 출현 빈도를 계산합니다. (최적화된 버전)

    Args:
        data: 로또 당첨 번호 목록
        logger: 로깅을 위한 Logger 객체 (선택 사항)
        normalize_method: 정규화 방법 ("max", "total", "relative")
        chunk_size: 청크 단위 처리 크기

    Returns:
        Dict[Tuple[int, int], float]: 번호 쌍의 정규화된 빈도 맵
    """
    if not data:
        if logger:
            logger.warning("빈 데이터로 인해 빈 쌍 빈도 맵을 반환합니다.")
        return {}

    # 청크 단위로 처리하여 메모리 효율성 향상
    pair_counter = Counter()
    total_draws = len(data)

    if logger:
        logger.info(f"총 {total_draws}개 추첨 데이터를 {chunk_size} 단위로 처리 시작")

    # 청크 단위 처리
    for i in range(0, total_draws, chunk_size):
        chunk = data[i : min(i + chunk_size, total_draws)]

        # 청크 내 모든 번호 쌍 카운팅
        for lottery_number in chunk:
            numbers = (
                lottery_number.numbers
                if hasattr(lottery_number, "numbers")
                else lottery_number
            )
            # 정렬된 번호로 모든 가능한 쌍 조합 생성
            sorted_numbers = sorted(numbers)
            for pair in combinations(sorted_numbers, 2):
                pair_counter[pair] += 1

    # 정규화 방법에 따른 처리
    if normalize_method == "max":
        max_count = max(pair_counter.values()) if pair_counter else 1
        pair_frequency = {
            pair: count / max_count for pair, count in pair_counter.items()
        }
    elif normalize_method == "total":
        total_pairs = sum(pair_counter.values())
        pair_frequency = {
            pair: count / total_pairs for pair, count in pair_counter.items()
        }
    elif normalize_method == "relative":
        # 각 번호별 출현 빈도 대비 상대적 빈도
        number_freq = Counter()
        for lottery_number in data:
            numbers = (
                lottery_number.numbers
                if hasattr(lottery_number, "numbers")
                else lottery_number
            )
            for number in numbers:
                number_freq[number] += 1

        pair_frequency = {}
        for (num1, num2), count in pair_counter.items():
            expected_freq = (number_freq[num1] * number_freq[num2]) / (total_draws**2)
            pair_frequency[(num1, num2)] = count / max(expected_freq, 1)
    else:
        # 기본값: max 방법 사용
        max_count = max(pair_counter.values()) if pair_counter else 1
        pair_frequency = {
            pair: count / max_count for pair, count in pair_counter.items()
        }

    if logger:
        logger.info(
            f"총 {len(pair_frequency)}개의 번호 쌍 빈도 계산 완료 (정규화: {normalize_method})"
        )

    return pair_frequency


@cached_analysis
def calculate_pair_centrality(
    pair_freq: Dict[Tuple[int, int], float],
    logger: Optional[logging.Logger] = None,
    centrality_types: List[str] = None,
    parallel: bool = True,
) -> Dict[int, float]:
    """
    쌍 빈도를 기반으로 가중치 그래프를 구성하고 중심성을 계산합니다. (최적화된 버전)

    Args:
        pair_freq: 번호 쌍의 빈도 맵
        logger: 로깅을 위한 Logger 객체 (선택 사항)
        centrality_types: 계산할 중심성 유형 리스트
        parallel: 병렬 처리 사용 여부

    Returns:
        Dict[int, float]: 노드별 중심성 점수
    """
    if not pair_freq:
        if logger:
            logger.warning("빈 쌍 빈도 맵으로 인해 빈 중심성 맵을 반환합니다.")
        return {}

    # 기본 중심성 유형 설정
    if centrality_types is None:
        centrality_types = ["degree", "closeness", "betweenness"]

    # NetworkX 그래프 생성
    G = nx.Graph()

    # 모든 노드와 간선 추가
    for (node1, node2), weight in pair_freq.items():
        G.add_edge(node1, node2, weight=weight)

    # 노드가 없으면 빈 결과 반환
    if not G.nodes():
        if logger:
            logger.warning("그래프에 노드가 없습니다. 빈 중심성 맵을 반환합니다.")
        return {}

    centrality_results = {}

    def calculate_single_centrality(centrality_type):
        """단일 중심성 계산 함수"""
        try:
            if centrality_type == "degree":
                return nx.degree_centrality(G)
            elif centrality_type == "closeness":
                return nx.closeness_centrality(G)
            elif centrality_type == "betweenness":
                # 큰 그래프의 경우 샘플링 사용
                if len(G.nodes()) > 100:
                    return nx.betweenness_centrality(
                        G, k=min(100, len(G.nodes())), weight="weight"
                    )
                else:
                    return nx.betweenness_centrality(G, weight="weight")
            elif centrality_type == "eigenvector":
                try:
                    return nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
                except:
                    return {node: 0.0 for node in G.nodes()}
            else:
                return {node: 0.0 for node in G.nodes()}
        except Exception as e:
            if logger:
                logger.warning(f"{centrality_type} 중심성 계산 중 오류: {str(e)}")
            return {node: 0.0 for node in G.nodes()}

    # 병렬 처리 또는 순차 처리
    if parallel and len(centrality_types) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(centrality_types))) as executor:
            future_to_type = {
                executor.submit(calculate_single_centrality, c_type): c_type
                for c_type in centrality_types
            }

            for future in as_completed(future_to_type):
                c_type = future_to_type[future]
                try:
                    centrality_results[c_type] = future.result()
                except Exception as e:
                    if logger:
                        logger.warning(f"{c_type} 중심성 병렬 계산 중 오류: {str(e)}")
                    centrality_results[c_type] = {node: 0.0 for node in G.nodes()}
    else:
        # 순차 처리
        for c_type in centrality_types:
            centrality_results[c_type] = calculate_single_centrality(c_type)

    # 중심성 점수 결합 (평균)
    centrality = {}
    for node in G.nodes():
        scores = [
            centrality_results[c_type].get(node, 0.0) for c_type in centrality_types
        ]
        centrality[node] = sum(scores) / len(scores) if scores else 0.0

    if logger:
        logger.info(
            f"총 {len(centrality)}개 노드의 중심성 계산 완료 (유형: {centrality_types})"
        )

    return centrality


@cached_analysis
def calculate_segment_entropy(
    data: List[LotteryNumber],
    segments: int = 5,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    세그먼트별 엔트로피를 계산합니다. (통합 최적화된 버전)

    Args:
        data: 로또 당첨 번호 목록
        segments: 세그먼트 수
        logger: 로깅을 위한 Logger 객체

    Returns:
        np.ndarray: 세그먼트별 엔트로피 벡터
    """
    if not data:
        if logger:
            logger.warning("빈 데이터로 인해 0 벡터를 반환합니다.")
        return np.zeros(segments)

    # 세그먼트 크기 계산
    segment_size = 45 // segments
    segment_entropy = np.zeros(segments)

    # 각 세그먼트별 번호 출현 빈도 계산
    for seg_idx in range(segments):
        start_num = seg_idx * segment_size + 1
        end_num = min((seg_idx + 1) * segment_size, 45)

        # 세그먼트 내 번호들의 출현 빈도 계산
        segment_counts = Counter()
        total_segment_appearances = 0

        for lottery_number in data:
            numbers = (
                lottery_number.numbers
                if hasattr(lottery_number, "numbers")
                else lottery_number
            )
            for number in numbers:
                if start_num <= number <= end_num:
                    segment_counts[number] += 1
                    total_segment_appearances += 1

        # 엔트로피 계산
        if total_segment_appearances > 0:
            entropy = 0.0
            for count in segment_counts.values():
                if count > 0:
                    prob = count / total_segment_appearances
                    entropy -= prob * np.log2(prob)
            segment_entropy[seg_idx] = entropy
        else:
            segment_entropy[seg_idx] = 0.0

    if logger:
        logger.info(f"{segments}개 세그먼트의 엔트로피 계산 완료")

    return segment_entropy


@cached_analysis
def calculate_number_gaps(
    data: List[LotteryNumber], logger: Optional[logging.Logger] = None
) -> Dict[int, Dict[str, float]]:
    """
    각 번호의 출현 간격을 계산합니다. (통합 최적화된 버전)

    Args:
        data: 로또 당첨 번호 목록
        logger: 로깅을 위한 Logger 객체

    Returns:
        Dict[int, Dict[str, float]]: 번호별 간격 통계
    """
    if not data:
        if logger:
            logger.warning("빈 데이터로 인해 빈 간격 맵을 반환합니다.")
        return {}

    # 각 번호의 출현 위치 기록
    number_positions = defaultdict(list)

    for idx, lottery_number in enumerate(data):
        numbers = (
            lottery_number.numbers
            if hasattr(lottery_number, "numbers")
            else lottery_number
        )
        for number in numbers:
            number_positions[number].append(idx)

    # 각 번호의 간격 통계 계산
    gap_stats = {}
    for number in range(1, 46):
        positions = number_positions.get(number, [])

        if len(positions) < 2:
            gap_stats[number] = {
                "avg_gap": len(data),  # 전체 기간
                "min_gap": len(data),
                "max_gap": len(data),
                "gap_variance": 0.0,
                "last_appearance": positions[0] if positions else -1,
            }
        else:
            gaps = [positions[i] - positions[i - 1] for i in range(1, len(positions))]
            gap_stats[number] = {
                "avg_gap": np.mean(gaps),
                "min_gap": min(gaps),
                "max_gap": max(gaps),
                "gap_variance": np.var(gaps),
                "last_appearance": positions[-1],
            }

    if logger:
        logger.info(f"45개 번호의 출현 간격 계산 완료")

    return gap_stats


@cached_analysis
def calculate_cluster_distribution(
    data: List[LotteryNumber],
    n_clusters: int = 5,
    embedding_dim: int = 3,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[int, int], Dict[str, float]]:
    """
    클러스터 분포를 계산합니다. (통합 최적화된 버전)

    Args:
        data: 로또 당첨 번호 목록
        n_clusters: 클러스터 수
        embedding_dim: 임베딩 차원
        logger: 로깅을 위한 Logger 객체

    Returns:
        Tuple[Dict[int, int], Dict[str, float]]: 번호별 클러스터 할당 및 품질 지표
    """
    if not data:
        if logger:
            logger.warning("빈 데이터로 인해 기본 클러스터 분포를 반환합니다.")
        return {i: 0 for i in range(1, 46)}, {"silhouette_score": 0.0}

    try:
        # 번호 출현 빈도 행렬 생성
        occurrence_matrix = np.zeros((45, len(data)))
        for i, lottery_number in enumerate(data):
            numbers = (
                lottery_number.numbers
                if hasattr(lottery_number, "numbers")
                else lottery_number
            )
            for number in numbers:
                occurrence_matrix[number - 1, i] = 1

        # 차원 축소 (t-SNE)
        if len(data) > 30:  # 충분한 데이터가 있을 때만 t-SNE 사용
            tsne = TSNE(
                n_components=min(embedding_dim, 3),
                perplexity=min(30, len(data) // 4),
                random_state=42,
                method="barnes_hut",
                learning_rate="auto",
                n_iter=500,  # 성능을 위해 반복 횟수 줄임
            )
            embeddings = tsne.fit_transform(occurrence_matrix)
        else:
            # 데이터가 적으면 PCA 사용
            from sklearn.decomposition import PCA

            pca = PCA(n_components=min(embedding_dim, len(data), 45))
            embeddings = pca.fit_transform(occurrence_matrix)

        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # 번호별 클러스터 할당
        number_clusters = {i + 1: int(label) for i, label in enumerate(cluster_labels)}

        # 품질 지표 계산
        quality_metrics = {}
        try:
            if len(set(cluster_labels)) > 1:
                quality_metrics["silhouette_score"] = silhouette_score(
                    embeddings, cluster_labels
                )
            else:
                quality_metrics["silhouette_score"] = 0.0
        except Exception as e:
            if logger:
                logger.warning(f"실루엣 점수 계산 중 오류: {str(e)}")
            quality_metrics["silhouette_score"] = 0.0

        # 클러스터 균형 점수
        cluster_counts = Counter(cluster_labels)
        cluster_sizes = list(cluster_counts.values())
        balance_score = (
            1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
            if cluster_sizes
            else 0.0
        )
        quality_metrics["balance_score"] = max(0.0, balance_score)

        if logger:
            logger.info(
                f"클러스터 분포 계산 완료 (클러스터 수: {n_clusters}, 실루엣 점수: {quality_metrics['silhouette_score']:.4f})"
            )

        return number_clusters, quality_metrics

    except Exception as e:
        if logger:
            logger.error(f"클러스터 분포 계산 중 오류: {str(e)}")
        return {i: 0 for i in range(1, 46)}, {"silhouette_score": 0.0, "error": str(e)}


def calculate_cluster_score(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    클러스터링 결과에 대한 다양한 품질 지표를 계산합니다.

    Args:
        similarity_matrix: 유사도 행렬
        labels: 클러스터링 결과 레이블
        logger: 로깅을 위한 Logger 객체 (선택 사항)

    Returns:
        Dict[str, float]: 클러스터 품질 지표 맵
    """
    # 결과 초기화
    result = {
        "silhouette_score": 0.0,
        "avg_distance_between_clusters": 0.0,
        "balance_score": 0.0,
        "largest_cluster_size_norm": 0.0,
        "cluster_count_norm": 0.0,
        "cluster_entropy_score": 0.0,
        "cohesiveness_score": 0.0,
    }

    # 데이터 검증
    if similarity_matrix.shape[0] != len(labels):
        if logger:
            logger.error(
                f"유사도 행렬 크기({similarity_matrix.shape[0]})와 레이블 수({len(labels)})가 일치하지 않습니다."
            )
        return result

    # 유효한 클러스터 식별
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(labels)

    # 클러스터가 하나만 있거나 없는 경우
    if n_clusters <= 1:
        if logger:
            logger.warning(
                f"클러스터 수가 부족합니다 ({n_clusters}). 품질 지표를 계산할 수 없습니다."
            )
        return result

    # 실루엣 점수 계산
    try:
        # 거리 행렬로 변환 (유사도가 아닌 경우)
        distance_matrix = 1 - similarity_matrix
        sil_score = silhouette_score(distance_matrix, labels)
        result["silhouette_score"] = max(0.0, sil_score)  # 음수값 방지
    except Exception as e:
        if logger:
            logger.warning(f"실루엣 점수 계산 중 오류 발생: {str(e)}")

    # 클러스터 간 평균 거리 계산
    try:
        inter_cluster_distances = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_i_indices = np.where(labels == unique_labels[i])[0]
                cluster_j_indices = np.where(labels == unique_labels[j])[0]

                if len(cluster_i_indices) > 0 and len(cluster_j_indices) > 0:
                    # 두 클러스터 간의 모든 쌍의 거리 계산
                    distances = []
                    for idx_i in cluster_i_indices:
                        for idx_j in cluster_j_indices:
                            distances.append(distance_matrix[idx_i, idx_j])

                    if distances:
                        inter_cluster_distances.append(np.mean(distances))

        if inter_cluster_distances:
            result["avg_distance_between_clusters"] = np.mean(inter_cluster_distances)
    except Exception as e:
        if logger:
            logger.warning(f"클러스터 간 거리 계산 중 오류 발생: {str(e)}")

    # 클러스터 균형 점수 계산
    try:
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        avg_size = np.mean(cluster_sizes)
        if avg_size > 0:
            balance_variance = np.var(cluster_sizes) / (avg_size**2)
            result["balance_score"] = max(0.0, 1.0 - balance_variance)
    except Exception as e:
        if logger:
            logger.warning(f"균형 점수 계산 중 오류 발생: {str(e)}")

    # 가장 큰 클러스터 크기 정규화
    try:
        largest_cluster_size = max([np.sum(labels == label) for label in unique_labels])
        result["largest_cluster_size_norm"] = 1.0 - (largest_cluster_size / n_samples)
    except Exception as e:
        if logger:
            logger.warning(f"최대 클러스터 크기 계산 중 오류 발생: {str(e)}")

    # 클러스터 수 정규화 (적절한 범위: 3-10)
    try:
        optimal_range = (3, 10)
        if optimal_range[0] <= n_clusters <= optimal_range[1]:
            result["cluster_count_norm"] = 1.0
        else:
            distance_from_range = min(
                abs(n_clusters - optimal_range[0]), abs(n_clusters - optimal_range[1])
            )
            result["cluster_count_norm"] = max(0.0, 1.0 - distance_from_range / 10.0)
    except Exception as e:
        if logger:
            logger.warning(f"클러스터 수 정규화 중 오류 발생: {str(e)}")

    # 클러스터 엔트로피 점수
    try:
        cluster_probs = (
            np.array([np.sum(labels == label) for label in unique_labels]) / n_samples
        )
        entropy = -np.sum(cluster_probs * np.log2(cluster_probs + 1e-10))
        max_entropy = np.log2(n_clusters)
        result["cluster_entropy_score"] = (
            entropy / max_entropy if max_entropy > 0 else 0.0
        )
    except Exception as e:
        if logger:
            logger.warning(f"클러스터 엔트로피 계산 중 오류 발생: {str(e)}")

    # 응집성 점수 (클러스터 내 평균 유사도)
    try:
        cohesiveness_scores = []
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 1:
                cluster_similarities = []
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        idx_i, idx_j = cluster_indices[i], cluster_indices[j]
                        cluster_similarities.append(similarity_matrix[idx_i, idx_j])

                if cluster_similarities:
                    cohesiveness_scores.append(np.mean(cluster_similarities))

        if cohesiveness_scores:
            result["cohesiveness_score"] = np.mean(cohesiveness_scores)
    except Exception as e:
        if logger:
            logger.warning(f"응집성 점수 계산 중 오류 발생: {str(e)}")

    if logger:
        logger.info(
            f"클러스터 품질 지표 계산 완료: 실루엣={result['silhouette_score']:.4f}"
        )

    return result


def clear_cache():
    """캐시를 비웁니다."""
    global _CACHE, _MEMORY_POOL
    _CACHE.clear()
    _MEMORY_POOL.clear()


def get_cache_stats() -> Dict[str, Any]:
    """캐시 통계를 반환합니다."""
    return {
        "cache_size": len(_CACHE),
        "cache_limit": _CACHE_SIZE_LIMIT,
        "memory_pool_size": len(_MEMORY_POOL),
    }
