"""
로또 번호의 클러스터를 분석하기 위한 모듈

이 모듈은 로또 번호들 간의 관계를 그래프로 모델링하고
클러스터링을 수행하여 패턴을 추출합니다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Set, Optional, Any
from pathlib import Path
from ..utils.error_handler import get_logger
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
import json

logger = get_logger(__name__)


def build_pair_graph(draw_history: List[Any]) -> np.ndarray:
    """
    로또 번호 사이의 동시 출현 관계를 인접 행렬로 구성

    Args:
        draw_history: 로또 번호 이력

    Returns:
        인접 행렬 (45x45)
    """
    # 인접 행렬 초기화 (45x45)
    adjacency_matrix = np.zeros((45, 45))

    # 모든 로또 번호 데이터 처리
    for draw in draw_history:
        # 숫자 리스트로 변환 (numbers 속성이 있는 경우)
        numbers = draw.numbers if hasattr(draw, "numbers") else draw

        # 모든 번호 쌍에 대해 연결 증가
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                num1, num2 = numbers[i], numbers[j]
                if 1 <= num1 <= 45 and 1 <= num2 <= 45:
                    adjacency_matrix[num1 - 1, num2 - 1] += 1
                    adjacency_matrix[num2 - 1, num1 - 1] += 1

    # 정규화
    if np.max(adjacency_matrix) > 0:
        adjacency_matrix = adjacency_matrix / np.max(adjacency_matrix)

    return adjacency_matrix


def create_default_gnn_model() -> nn.Module:
    """
    기본 GNN 모델을 생성합니다.

    Returns:
        nn.Module: 기본 GNN 모델
    """
    logger.info("기본 GNN 모델 생성 중...")

    class DefaultGNNModel(nn.Module):
        def __init__(self, input_dim=45, embedding_dim=32, hidden_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(input_dim + 1, embedding_dim)  # +1은 패딩용
            self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.fc = nn.Linear(hidden_dim, embedding_dim)

        def forward(self, x):
            # x: [batch_size, seq_len]
            x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
            x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.mean(dim=2)  # global average pooling
            x = self.fc(x)
            return x

    model = DefaultGNNModel()
    logger.info("기본 GNN 모델 생성 완료")
    return model


class ClusterAnalyzer:
    """
    로또 번호의 클러스터를 분석하는 클래스

    이 클래스는 로또 번호 간의 관계를 분석하고,
    번호들을 의미 있는 클러스터로 그룹화합니다.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        초기화

        Args:
            config: 설정 객체
        """
        self.logger = get_logger(__name__)

        # DBSCAN 사용 여부
        self.use_dbscan = False
        if "clustering" in config and "use_dbscan" in config["clustering"]:
            self.use_dbscan = config["clustering"]["use_dbscan"]

        # DBSCAN 파라미터
        self.eps = 0.5
        self.min_samples = 3
        if "clustering" in config and "dbscan" in config["clustering"]:
            dbscan_config = config["clustering"]["dbscan"]
            if "eps" in dbscan_config:
                self.eps = dbscan_config["eps"]
            if "min_samples" in dbscan_config:
                self.min_samples = dbscan_config["min_samples"]

        # KMeans 파라미터
        self.n_clusters = 4
        if "clustering" in config and "n_clusters" in config["clustering"]:
            self.n_clusters = config["clustering"]["n_clusters"]

        # 클러스터 품질 최소 기준
        self.min_silhouette_score = 0.1
        if "clustering" in config and "min_silhouette_score" in config["clustering"]:
            self.min_silhouette_score = config["clustering"]["min_silhouette_score"]

        # 자동 클러스터 수 탐색 설정
        self.auto_adjust_clusters = True
        self.min_clusters = 2
        self.max_clusters = 8
        if "clustering" in config and "auto_adjust_clusters" in config["clustering"]:
            self.auto_adjust_clusters = config["clustering"]["auto_adjust_clusters"]
            if "min_clusters" in config["clustering"]:
                self.min_clusters = config["clustering"]["min_clusters"]
            if "max_clusters" in config["clustering"]:
                self.max_clusters = config["clustering"]["max_clusters"]

        # 결과 저장 경로
        self.result_path = "data/result/analysis"
        if "paths" in config and "analysis_result_dir" in config["paths"]:
            self.result_path = config["paths"]["analysis_result_dir"]

    def _compute_cluster_quality(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        클러스터 품질 지표 계산

        Args:
            embeddings: 임베딩 벡터
            labels: 클러스터 레이블

        Returns:
            품질 지표 딕셔너리
        """
        quality_metrics = {}

        # 클러스터 개수
        unique_labels = np.unique(labels)
        if -1 in unique_labels:  # DBSCAN에서는 -1이 노이즈 포인트
            unique_labels = unique_labels[unique_labels != -1]
        cluster_count = len(unique_labels)
        quality_metrics["cluster_count"] = int(cluster_count)

        # 실루엣 점수 계산
        silhouette = 0.0
        if cluster_count > 1 and len(embeddings) > cluster_count:
            try:
                # -1 레이블(노이즈)을 제외하고 계산
                valid_indices = labels != -1
                if np.sum(valid_indices) > cluster_count:
                    silhouette = float(
                        silhouette_score(
                            embeddings[valid_indices],
                            labels[valid_indices],
                            sample_size=min(1000, np.sum(valid_indices)),
                        )
                    )
            except Exception as e:
                self.logger.warning(f"실루엣 점수 계산 실패: {e}")

        quality_metrics["silhouette_score"] = silhouette

        # 클러스터 스코어 계산 (silhouette score 기반)
        cluster_score = 0.0
        if cluster_count > 1:
            # 실루엣 점수를 0~1 범위로 변환 (-1~1 범위에서)
            cluster_score = (silhouette + 1) / 2
        else:
            # 클러스터가 1개인 경우 또는 거리가 0인 경우 경고 출력
            self.logger.warning(
                "클러스터 수가 1개이거나 거리가 0입니다. cluster_score=0.0으로 설정"
            )
            cluster_score = 0.0

        quality_metrics["cluster_score"] = float(cluster_score)

        # 클러스터 간 거리
        avg_distance = 0.0
        if cluster_count > 1:
            # 클러스터 중심점 계산
            centroids = []
            for label in unique_labels:
                if label == -1:  # 노이즈 무시
                    continue
                mask = labels == label
                if np.sum(mask) > 0:
                    centroid = np.mean(embeddings[mask], axis=0)
                    centroids.append(centroid)

            # 중심점 간 거리 계산
            if len(centroids) > 1:
                distances = []
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        dist = np.linalg.norm(centroids[i] - centroids[j])
                        distances.append(dist)
                if distances:
                    avg_distance = float(np.mean(distances))

        quality_metrics["avg_distance_between_clusters"] = avg_distance

        # 클러스터 크기 균형성
        cluster_sizes = []
        for label in unique_labels:
            if label == -1:  # 노이즈 무시
                continue
            cluster_sizes.append(np.sum(labels == label))

        std_size = 0.0
        balance_score = 1.0
        if cluster_sizes:
            std_size = float(np.std(cluster_sizes))
            mean_size = float(np.mean(cluster_sizes))
            if mean_size > 0:
                # 균형 점수: 1이면 완벽하게 균형, 0에 가까울수록 불균형
                balance_score = 1.0 - min(1.0, std_size / mean_size)

        quality_metrics["std_of_cluster_size"] = std_size
        quality_metrics["balance_score"] = float(balance_score)

        # 최대 클러스터 크기
        largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0
        quality_metrics["largest_cluster_size"] = float(largest_cluster_size)

        # 가장 작은 클러스터 크기
        smallest_cluster_size = min(cluster_sizes) if cluster_sizes else 0
        quality_metrics["smallest_cluster_size"] = float(smallest_cluster_size)

        # 클러스터 크기 비율 (가장 큰 / 가장 작은)
        size_ratio = 0.0
        if smallest_cluster_size > 0:
            size_ratio = largest_cluster_size / smallest_cluster_size
        quality_metrics["cluster_size_ratio"] = float(size_ratio)

        # 클러스터 엔트로피 점수 (새로 추가)
        entropy_score = 0.0
        if cluster_sizes:
            # 클러스터 크기 비율 계산
            total_size = sum(cluster_sizes)
            size_ratios = [size / total_size for size in cluster_sizes]

            # 엔트로피 계산 (균등한 분포일수록 높음)
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in size_ratios)
            max_entropy = np.log2(len(cluster_sizes)) if len(cluster_sizes) > 0 else 1

            # 정규화된 엔트로피 (0-1 범위)
            entropy_score = entropy / max_entropy if max_entropy > 0 else 0

        quality_metrics["cluster_entropy_score"] = float(entropy_score)

        # 응집도 점수 (새로 추가)
        cohesiveness_score = 0.0
        if cluster_count > 1:
            # 클러스터 내 거리 평균 계산
            intra_cluster_distances = []
            for label in unique_labels:
                if label == -1:  # 노이즈 무시
                    continue
                cluster_points = embeddings[labels == label]
                if len(cluster_points) > 1:
                    # 클러스터 내 모든 점 간의 거리 계산
                    for i in range(len(cluster_points)):
                        for j in range(i + 1, len(cluster_points)):
                            dist = np.linalg.norm(cluster_points[i] - cluster_points[j])
                            intra_cluster_distances.append(dist)

            # 내부 거리와 외부 거리의 비율로 응집도 계산
            if intra_cluster_distances and avg_distance > 0:
                avg_intra_distance = np.mean(intra_cluster_distances)
                cohesiveness_score = 1.0 - (avg_intra_distance / (avg_distance + 0.001))
                # 0-1 범위로 제한
                cohesiveness_score = max(0.0, min(1.0, cohesiveness_score))

        quality_metrics["cohesiveness_score"] = float(cohesiveness_score)

        # 정규화된 클러스터 수
        norm_cluster_count = 0.0
        if self.max_clusters > self.min_clusters:
            norm_cluster_count = (cluster_count - self.min_clusters) / (
                self.max_clusters - self.min_clusters
            )
            norm_cluster_count = max(0.0, min(1.0, norm_cluster_count))
        quality_metrics["cluster_count_norm"] = float(norm_cluster_count)

        # 정규화된 최대 클러스터 크기
        if len(embeddings) > 0:
            norm_largest_size = largest_cluster_size / len(embeddings)
        else:
            norm_largest_size = 0.0
        quality_metrics["largest_cluster_size_norm"] = float(norm_largest_size)

        # 정규화된 클러스터 크기 비율
        norm_size_ratio = 0.0
        if size_ratio > 0:
            # 비율을 0-1 범위로 변환 (비율이 1에 가까울수록 균형)
            norm_size_ratio = 1.0 / (1.0 + np.log(1 + size_ratio))
        quality_metrics["cluster_size_ratio_norm"] = float(norm_size_ratio)

        return quality_metrics

    def analyze_clusters(self, draw_history: List[List[int]]) -> Dict[str, Any]:
        """
        로또 번호의 클러스터를 분석합니다.

        Args:
            draw_history: 로또 번호 이력

        Returns:
            Dict[str, Any]: 클러스터 분석 결과
        """
        # 결과 초기화
        result = {
            "clusters": [],
            "adjacency_matrix": None,
            "embedding": None,
            "cluster_embedding_quality": {},
            "cluster_groups": {},
        }

        # 충분한 데이터가 있는지 확인
        if not draw_history or len(draw_history) < 5:
            self.logger.warning("분석할 충분한 데이터가 없습니다.")
            return result

        # 동시 출현 그래프 구성
        adjacency_matrix = build_pair_graph(draw_history)
        result["adjacency_matrix"] = adjacency_matrix.tolist()

        # 그래프 특성 추출 (연결 중심성)
        graph_features = np.sum(adjacency_matrix, axis=1)

        # 번호별 출현 빈도 계산
        number_counts = np.zeros(45)
        for draw in draw_history:
            for num in draw:
                if 1 <= num <= 45:
                    number_counts[num - 1] += 1

        # 빈도 정규화
        if len(draw_history) > 0:
            node_features = number_counts / len(draw_history)
        else:
            node_features = number_counts

        # 그래프 임베딩을 위한 특성 행렬 구성
        # 빈도와 연결성을 결합한 특성
        embeddings = np.column_stack((node_features, graph_features))
        result["embedding"] = embeddings.tolist()

        # 클러스터링 수행
        cluster_labels = np.zeros(len(embeddings), dtype=np.int32)

        # 자동 클러스터 수 조정 기능 강화
        if self.auto_adjust_clusters and not self.use_dbscan:
            # 최적의 클러스터 수 탐색 (min_clusters ~ max_clusters)
            best_score = -1
            best_labels = np.zeros(len(embeddings), dtype=np.int32)
            best_n_clusters = self.n_clusters  # 기본값

            for n_clusters in range(
                self.min_clusters, min(self.max_clusters + 1, len(embeddings))
            ):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings)

                    # 클러스터가 2개 이상일 때만 실루엣 점수 계산
                    if len(np.unique(labels)) > 1:
                        try:
                            score = silhouette_score(embeddings, labels)
                            self.logger.info(
                                f"클러스터 수 {n_clusters}의 실루엣 점수: {score:.3f}"
                            )

                            if score > best_score:
                                best_score = score
                                best_labels = labels
                                best_n_clusters = n_clusters
                        except:
                            pass
                except Exception as e:
                    self.logger.warning(f"클러스터 수 {n_clusters} 시도 중 오류: {e}")

            # 최적의 클러스터 수를 발견했으면 사용
            if best_score > 0:
                self.logger.info(
                    f"최적의 클러스터 수: {best_n_clusters} (실루엣 점수: {best_score:.3f})"
                )
                cluster_labels = best_labels
            else:
                # 기본 클러스터 수 사용
                self.logger.warning(
                    f"최적의 클러스터 수를 찾지 못했습니다. 기본값 {self.n_clusters} 사용"
                )
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
        elif self.use_dbscan:
            # DBSCAN 사용 (밀도 기반 클러스터링)
            try:
                dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                cluster_labels = dbscan.fit_predict(embeddings)
                self.logger.info(
                    f"DBSCAN 클러스터링 결과: {len(np.unique(cluster_labels))}개 클러스터 (노이즈 포함)"
                )
            except Exception as e:
                self.logger.error(f"DBSCAN 클러스터링 실패: {e}")
        else:
            # 자동 조정 없이 기본 KMeans 사용
            try:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
            except Exception as e:
                self.logger.error(f"KMeans 클러스터링 실패: {e}")

        # 클러스터 품질 계산
        quality_metrics = self._compute_cluster_quality(embeddings, cluster_labels)
        result["cluster_embedding_quality"] = quality_metrics

        # 클러스터 품질 확인 및 경고
        silhouette = quality_metrics.get("silhouette_score", 0.0)
        cluster_count = quality_metrics.get("cluster_count", 0)

        if silhouette < self.min_silhouette_score:
            self.logger.warning(
                f"낮은 클러스터 품질: 실루엣 점수 {silhouette:.3f} < {self.min_silhouette_score}"
            )
            if not self.use_dbscan:
                self.logger.info(f"DBSCAN으로 클러스터링 재시도")
                try:
                    # DBSCAN으로 재시도
                    dbscan = DBSCAN(eps=0.4, min_samples=3)  # 더 관대한 파라미터
                    new_labels = dbscan.fit_predict(embeddings)
                    new_quality = self._compute_cluster_quality(embeddings, new_labels)

                    if new_quality.get("silhouette_score", 0.0) > silhouette:
                        self.logger.info(
                            f"DBSCAN 결과가 더 좋음: {new_quality.get('silhouette_score', 0.0):.3f} > {silhouette:.3f}"
                        )
                        cluster_labels = new_labels
                        result["cluster_embedding_quality"] = new_quality
                except Exception as e:
                    self.logger.error(f"DBSCAN 재시도 실패: {e}")

        if cluster_count <= 1:
            self.logger.warning(
                f"클러스터가 하나뿐입니다. 클러스터 기반 특성이 유용하지 않을 수 있습니다."
            )

        # 클러스터 결과 저장
        for i in range(1, 46):  # 로또 번호 1-45
            cluster_id = int(cluster_labels[i - 1])
            result["clusters"].append(
                {
                    "number": i,
                    "cluster": cluster_id,
                    "frequency": float(node_features[i - 1]),
                    "connections": float(graph_features[i - 1]),
                }
            )

        # 클러스터별 번호 그룹화
        cluster_groups = {}
        for i, label in enumerate(cluster_labels):
            label_str = str(label)
            if label_str not in cluster_groups:
                cluster_groups[label_str] = []
            cluster_groups[label_str].append(i + 1)  # 인덱스를 로또 번호(1-45)로 변환

        # 클러스터 그룹 정보 추가
        result["cluster_groups"] = cluster_groups

        # 클러스터 품질 메트릭스를 특성 벡터로 변환하여 저장
        try:
            cluster_feature_vector = self.create_cluster_feature_vector(
                result["cluster_embedding_quality"]
            )
            result["cluster_feature_vector"] = cluster_feature_vector.tolist()
        except Exception as e:
            self.logger.error(f"클러스터 특성 벡터 생성 실패: {e}")
            result["cluster_feature_vector"] = []

        return result

    def create_cluster_feature_vector(
        self, quality_metrics: Dict[str, Any]
    ) -> np.ndarray:
        """
        클러스터 품질 지표를 특성 벡터로 변환합니다.

        Args:
            quality_metrics: 클러스터 품질 지표 딕셔너리

        Returns:
            np.ndarray: 클러스터 특성 벡터 (12차원)
        """
        # 12개 특성으로 구성된 벡터 생성 (기존 8개에서 확장)
        feature_vector = np.zeros(12, dtype=np.float32)

        # 1. 클러스터 스코어
        feature_vector[0] = quality_metrics.get("cluster_score", 0.0)

        # 2. 실루엣 점수 (0-1로 정규화)
        silhouette = quality_metrics.get("silhouette_score", 0.0)
        feature_vector[1] = (silhouette + 1) / 2  # -1~1 범위를 0~1로 변환

        # 3. 클러스터 간 평균 거리 (정규화)
        avg_distance = quality_metrics.get("avg_distance_between_clusters", 0.0)
        feature_vector[2] = min(avg_distance / 2.0, 1.0)  # 최대 2.0으로 가정하고 정규화

        # 4. 클러스터 크기 표준편차 (정규화)
        std_size = quality_metrics.get("std_of_cluster_size", 0.0)
        feature_vector[3] = min(std_size / 10.0, 1.0)  # 최대 10으로 가정하고 정규화

        # 5. 균형 점수 (이미 0-1 범위)
        feature_vector[4] = quality_metrics.get("balance_score", 1.0)

        # 6. 클러스터 개수 (정규화)
        cluster_count = quality_metrics.get("cluster_count", 0)
        feature_vector[5] = quality_metrics.get(
            "cluster_count_norm", min(cluster_count / 8.0, 1.0)
        )

        # 7. 최대 클러스터 크기 (정규화)
        feature_vector[6] = quality_metrics.get("largest_cluster_size_norm", 0.0)

        # 8. 클러스터 크기 비율 (정규화)
        feature_vector[7] = quality_metrics.get("cluster_size_ratio_norm", 0.0)

        # 9. 클러스터 엔트로피 점수 (새로 추가)
        feature_vector[8] = quality_metrics.get("cluster_entropy_score", 0.5)

        # 10. 응집도 점수 (새로 추가)
        feature_vector[9] = quality_metrics.get("cohesiveness_score", 0.5)

        # 11. 최소 클러스터 크기 (정규화)
        smallest_size = quality_metrics.get("smallest_cluster_size", 0.0)
        if smallest_size > 0:
            feature_vector[10] = min(smallest_size / 10.0, 1.0)  # 최대 10으로 가정
        else:
            feature_vector[10] = 0.0

        # 12. 클러스터 분포 균일성 (실루엣과 엔트로피의 조합)
        feature_vector[11] = (feature_vector[1] + feature_vector[8]) / 2.0

        return feature_vector

    def extract_cluster_features(
        self, data: List[List[int]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        클러스터 분석 결과를 10차원 특성 벡터로 추출

        Args:
            data: 로또 번호 이력 데이터

        Returns:
            Tuple[np.ndarray, List[str]]: 10차원 클러스터 특성 벡터와 특성 이름
        """
        self.logger.info("클러스터 특성 추출 시작...")

        try:
            # 데이터 부족 시 기본값 반환
            if not data or len(data) < 10:
                self.logger.warning("데이터 부족으로 기본 클러스터 특성 반환")
                return self._get_default_cluster_features()

            # 1. 번호별 임베딩 생성
            embeddings = self._create_number_embeddings(data)

            # 2. 여러 K 값으로 클러스터링 수행 및 품질 지표 계산
            cluster_qualities = {}

            # K-means 클러스터링 (k=3,4,5)
            for k in [3, 4, 5]:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings)

                    # 품질 지표 계산
                    quality = self._compute_cluster_quality(embeddings, labels)
                    cluster_qualities[f"k{k}"] = quality

                except Exception as e:
                    self.logger.warning(f"K={k} 클러스터링 실패: {e}")
                    cluster_qualities[f"k{k}"] = {
                        "silhouette_score": 0.0,
                        "cluster_score": 0.0,
                    }

            # 3. 10차원 특성 벡터 구성
            features = []
            feature_names = []

            # 실루엣 점수 (3개)
            for k in [3, 4, 5]:
                silhouette = cluster_qualities.get(f"k{k}", {}).get(
                    "silhouette_score", 0.0
                )
                # -1~1 범위를 0~1로 정규화
                normalized_silhouette = max(0.0, (silhouette + 1) / 2)
                features.append(normalized_silhouette)
                feature_names.append(f"cluster_silhouette_k{k}")

            # 칼린스키-하라바즈 지수 (3개)
            for k in [3, 4, 5]:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings)

                    from sklearn.metrics import calinski_harabasz_score

                    ch_score = calinski_harabasz_score(embeddings, labels)
                    # 정규화 (일반적으로 0~1000 범위)
                    normalized_ch = min(1.0, ch_score / 1000.0)
                    features.append(normalized_ch)
                    feature_names.append(f"cluster_ch_score_k{k}")

                except Exception as e:
                    self.logger.warning(
                        f"칼린스키-하라바즈 점수 계산 실패 (k={k}): {e}"
                    )
                    features.append(0.0)
                    feature_names.append(f"cluster_ch_score_k{k}")

            # 클러스터 균형도 (1개)
            try:
                # 최적 K 선택 (실루엣 점수 기준)
                best_k = 3
                best_silhouette = -1
                for k in [3, 4, 5]:
                    sil = cluster_qualities.get(f"k{k}", {}).get("silhouette_score", -1)
                    if sil > best_silhouette:
                        best_silhouette = sil
                        best_k = k

                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)

                # 클러스터별 크기 균형도 계산
                unique, counts = np.unique(labels, return_counts=True)
                balance_score = (
                    1.0 - np.std(counts) / np.mean(counts) if len(counts) > 1 else 0.0
                )
                balance_score = max(0.0, min(1.0, balance_score))

                features.append(balance_score)
                feature_names.append("cluster_balance_score")

            except Exception as e:
                self.logger.warning(f"클러스터 균형도 계산 실패: {e}")
                features.append(0.0)
                feature_names.append("cluster_balance_score")

            # 클러스터 분리도 (1개)
            try:
                # 클러스터 중심 간 평균 거리
                centers = kmeans.cluster_centers_
                if len(centers) > 1:
                    distances = []
                    for i in range(len(centers)):
                        for j in range(i + 1, len(centers)):
                            dist = np.linalg.norm(centers[i] - centers[j])
                            distances.append(dist)

                    separation_score = np.mean(distances)
                    # 정규화 (임베딩 차원 고려)
                    separation_score = min(1.0, separation_score / 10.0)
                else:
                    separation_score = 0.0

                features.append(separation_score)
                feature_names.append("cluster_separation_score")

            except Exception as e:
                self.logger.warning(f"클러스터 분리도 계산 실패: {e}")
                features.append(0.0)
                feature_names.append("cluster_separation_score")

            # 클러스터 내부 분산 (1개)
            try:
                intra_variance = 0.0
                for cluster_id in unique:
                    cluster_points = embeddings[labels == cluster_id]
                    if len(cluster_points) > 1:
                        cluster_center = np.mean(cluster_points, axis=0)
                        variance = np.mean(
                            np.sum((cluster_points - cluster_center) ** 2, axis=1)
                        )
                        intra_variance += variance

                intra_variance /= len(unique)
                # 정규화
                intra_variance = min(1.0, intra_variance / 100.0)

                features.append(intra_variance)
                feature_names.append("cluster_intra_variance")

            except Exception as e:
                self.logger.warning(f"클러스터 내부 분산 계산 실패: {e}")
                features.append(0.0)
                feature_names.append("cluster_intra_variance")

            # 클러스터 간 거리 (1개)
            try:
                inter_distance = 0.0
                if len(centers) > 1:
                    min_distances = []
                    for i, center in enumerate(centers):
                        other_centers = np.delete(centers, i, axis=0)
                        min_dist = np.min(
                            [np.linalg.norm(center - other) for other in other_centers]
                        )
                        min_distances.append(min_dist)

                    inter_distance = np.mean(min_distances)
                    # 정규화
                    inter_distance = min(1.0, inter_distance / 10.0)

                features.append(inter_distance)
                feature_names.append("cluster_inter_distance")

            except Exception as e:
                self.logger.warning(f"클러스터 간 거리 계산 실패: {e}")
                features.append(0.0)
                feature_names.append("cluster_inter_distance")

            # 최종 검증 (정확히 10차원인지 확인)
            if len(features) != 10:
                self.logger.warning(f"클러스터 특성 차원 불일치: {len(features)} != 10")
                # 10차원으로 조정
                if len(features) < 10:
                    features.extend([0.0] * (10 - len(features)))
                    feature_names.extend(
                        [f"cluster_padding_{i}" for i in range(10 - len(feature_names))]
                    )
                else:
                    features = features[:10]
                    feature_names = feature_names[:10]

            result_vector = np.array(features, dtype=np.float32)

            # NaN/Inf 처리
            result_vector = np.nan_to_num(
                result_vector, nan=0.0, posinf=1.0, neginf=0.0
            )
            result_vector = np.clip(result_vector, 0.0, 1.0)

            self.logger.info(f"클러스터 특성 추출 완료: {len(result_vector)}차원")
            return result_vector, feature_names

        except Exception as e:
            self.logger.error(f"클러스터 특성 추출 중 오류: {e}")
            return self._get_default_cluster_features()

    def _create_number_embeddings(self, data: List[List[int]]) -> np.ndarray:
        """
        번호별 임베딩 생성

        Args:
            data: 로또 번호 이력

        Returns:
            np.ndarray: 45x4 임베딩 행렬 (각 번호당 4차원 특성)
        """
        embeddings = np.zeros((45, 4), dtype=np.float32)

        try:
            # 각 번호(1~45)에 대한 특성 계산
            for num in range(1, 46):
                # 1. 출현 빈도
                frequency = sum(1 for draw in data if num in draw)
                embeddings[num - 1, 0] = frequency / len(data) if data else 0.0

                # 2. 평균 위치 (번호가 나타나는 평균 위치)
                positions = []
                for draw in data:
                    if num in draw:
                        try:
                            pos = draw.index(num)
                            positions.append(pos)
                        except ValueError:
                            continue

                avg_position = (
                    np.mean(positions) / 5.0 if positions else 0.5
                )  # 0~1 정규화
                embeddings[num - 1, 1] = avg_position

                # 3. 위치 분산
                pos_variance = (
                    np.var(positions) / 25.0 if len(positions) > 1 else 0.0
                )  # 0~1 정규화
                embeddings[num - 1, 2] = min(1.0, pos_variance)

                # 4. 최근 출현율 (최근 20회 중 출현 비율)
                recent_data = data[-20:] if len(data) >= 20 else data
                recent_frequency = sum(1 for draw in recent_data if num in draw)
                recent_rate = (
                    recent_frequency / len(recent_data) if recent_data else 0.0
                )
                embeddings[num - 1, 3] = recent_rate

        except Exception as e:
            self.logger.warning(f"임베딩 생성 중 오류: {e}")

        return embeddings

    def _get_default_cluster_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        기본 클러스터 특성 반환 (데이터 부족 시)

        Returns:
            Tuple[np.ndarray, List[str]]: 기본 10차원 벡터와 특성 이름
        """
        features = [0.0] * 10
        feature_names = [
            "cluster_silhouette_k3",
            "cluster_silhouette_k4",
            "cluster_silhouette_k5",
            "cluster_ch_score_k3",
            "cluster_ch_score_k4",
            "cluster_ch_score_k5",
            "cluster_balance_score",
            "cluster_separation_score",
            "cluster_intra_variance",
            "cluster_inter_distance",
        ]

        return np.array(features, dtype=np.float32), feature_names


__all__ = ["ClusterAnalyzer", "build_pair_graph", "create_default_gnn_model"]
