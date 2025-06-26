#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
그래프 분석 유틸리티

이 모듈은 로또 데이터 분석에 사용되는 그래프 기반 분석 함수들을 제공합니다.
쌍 빈도, 중심성, 클러스터 품질 지표 등의 계산을 담당합니다.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from itertools import combinations
from collections import Counter
import logging
from sklearn.metrics import silhouette_score

# 타입 정의
LotteryNumber = Any  # 실제 LotteryNumber 클래스 가져오기가 어려울 수 있어 Any로 대체


def calculate_pair_frequency(
    data: List[LotteryNumber], logger: Optional[logging.Logger] = None
) -> Dict[Tuple[int, int], float]:
    """
    번호 쌍의 정규화된 동시 출현 빈도를 계산합니다.

    Args:
        data: 로또 당첨 번호 목록
        logger: 로깅을 위한 Logger 객체 (선택 사항)

    Returns:
        Dict[Tuple[int, int], float]: 번호 쌍의 정규화된 빈도 맵
    """
    # 쌍 카운터 초기화
    pair_counter: Counter = Counter()

    # 모든 번호 쌍 카운팅
    for lottery_number in data:
        numbers = (
            lottery_number.numbers
            if hasattr(lottery_number, "numbers")
            else lottery_number
        )
        # 모든 가능한 쌍 조합 생성
        for pair in combinations(sorted(numbers), 2):
            pair_counter[pair] += 1

    # 정규화를 위한 최대 카운트 계산
    max_count = max(pair_counter.values()) if pair_counter else 1

    # 정규화된 빈도 계산
    pair_frequency: Dict[Tuple[int, int], float] = {}
    for pair, count in pair_counter.items():
        pair_frequency[pair] = count / max_count

    if logger:
        logger.info(f"총 {len(pair_frequency)}개의 번호 쌍 빈도 계산 완료")

    return pair_frequency


def calculate_pair_centrality(
    pair_freq: Dict[Tuple[int, int], float], logger: Optional[logging.Logger] = None
) -> Dict[int, float]:
    """
    쌍 빈도를 기반으로 가중치 그래프를 구성하고 중심성을 계산합니다.

    Args:
        pair_freq: 번호 쌍의 빈도 맵
        logger: 로깅을 위한 Logger 객체 (선택 사항)

    Returns:
        Dict[int, float]: 노드별 중심성 점수
    """
    # NetworkX 그래프 생성
    G = nx.Graph()

    # 모든 노드와 간선 추가
    for (node1, node2), weight in pair_freq.items():
        if not G.has_node(node1):
            G.add_node(node1)
        if not G.has_node(node2):
            G.add_node(node2)
        G.add_edge(node1, node2, weight=weight)

    # 노드가 없으면 빈 결과 반환
    if not G.nodes():
        if logger:
            logger.warning("그래프에 노드가 없습니다. 빈 중심성 맵을 반환합니다.")
        return {}

    # 중심성 계산
    try:
        # 연결 중심성 (Degree Centrality)
        degree_centrality = nx.degree_centrality(G)

        # 근접 중심성 (Closeness Centrality)
        # 비연결 그래프인 경우 예외 처리
        try:
            closeness_centrality = nx.closeness_centrality(G)
        except:
            if logger:
                logger.warning("근접 중심성 계산 중 오류 발생. 기본값 사용.")
            closeness_centrality = {node: 0.0 for node in G.nodes()}

        # 매개 중심성 (Betweenness Centrality)
        # 계산이 오래 걸릴 수 있으므로 작은 그래프에만 적용
        if len(G.nodes()) <= 100:
            betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
        else:
            if logger:
                logger.info(
                    f"노드 수가 많아({len(G.nodes())}) 매개 중심성 계산을 건너뜁니다."
                )
            betweenness_centrality = {node: 0.0 for node in G.nodes()}

        # 중심성 점수 결합 (평균)
        centrality: Dict[int, float] = {}
        for node in G.nodes():
            centrality[node] = (
                degree_centrality.get(node, 0.0)
                + closeness_centrality.get(node, 0.0)
                + betweenness_centrality.get(node, 0.0)
            ) / 3.0

        if logger:
            logger.info(f"총 {len(centrality)}개 노드의 중심성 계산 완료")

        return centrality

    except Exception as e:
        if logger:
            logger.error(f"중심성 계산 중 오류 발생: {str(e)}")
        # 오류 발생 시 빈 딕셔너리 반환
        return {}


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

    # 클러스터 크기 계산 및 균형 점수
    try:
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        max_cluster_size = max(cluster_sizes)

        # 최대 클러스터 크기 정규화
        result["largest_cluster_size_norm"] = 1.0 - (max_cluster_size / n_samples)

        # 클러스터 수 정규화 (최대 10개 기준)
        result["cluster_count_norm"] = min(n_clusters / 10.0, 1.0)

        # 균형 점수 계산 (크기의 표준편차 기반)
        if n_clusters > 1:
            size_std = np.std(cluster_sizes)
            expected_size = n_samples / n_clusters
            # 표준편차가 0에 가까울수록 균형이 좋음
            result["balance_score"] = 1.0 - min(size_std / expected_size, 1.0)

        # 클러스터 엔트로피 점수 계산
        cluster_probs = [size / n_samples for size in cluster_sizes]
        entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in cluster_probs])
        max_entropy = np.log2(n_clusters) if n_clusters > 0 else 1.0

        # 엔트로피 정규화 (0-1 범위)
        result["cluster_entropy_score"] = (
            entropy / max_entropy if max_entropy > 0 else 0.0
        )
    except Exception as e:
        if logger:
            logger.warning(f"클러스터 크기 및 균형 계산 중 오류 발생: {str(e)}")

    # 응집도 점수 계산 (클러스터 내 유사도 평균)
    try:
        intra_cluster_similarities = []
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) > 1:
                # 클러스터 내 모든 쌍의 유사도 계산
                cluster_similarities = []
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        cluster_similarities.append(
                            similarity_matrix[indices[i], indices[j]]
                        )

                if cluster_similarities:
                    intra_cluster_similarities.append(np.mean(cluster_similarities))

        if intra_cluster_similarities:
            result["cohesiveness_score"] = np.mean(intra_cluster_similarities)
    except Exception as e:
        if logger:
            logger.warning(f"응집도 점수 계산 중 오류 발생: {str(e)}")

    if logger:
        logger.info(
            f"클러스터 품질 지표 계산 완료: {n_clusters}개 클러스터, 실루엣 점수 {result['silhouette_score']:.4f}"
        )

    return result
