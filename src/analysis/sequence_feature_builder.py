"""
시퀀스 특성 생성 모듈

이 모듈은 LSTM, GNN 등의 딥러닝 모델을 위한 시퀀스 데이터를 생성하는 기능을 제공합니다.
"""

import numpy as np
import pickle
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Set

# logging 제거 - unified_logging 사용
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from ..utils.unified_logging import get_logger
from ..utils.unified_performance import performance_monitor
from ..utils.unified_config import ConfigProxy
from ..shared.types import LotteryNumber
from .pattern_analyzer import PatternAnalyzer
from .enhanced_pattern_vectorizer import EnhancedPatternVectorizer
from .base_feature_extractor import BaseFeatureExtractor


# 로거 설정
logger = get_logger(__name__)


class SequenceFeatureBuilder(BaseFeatureExtractor):
    """시퀀스 특성 생성 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        초기화

        Args:
            config: 설정
        """
        super().__init__(config)
        self.config = ConfigProxy(config or {})
        self.pattern_analyzer = PatternAnalyzer(config)
        self.pattern_vectorizer = EnhancedPatternVectorizer(config)

        # 캐시 디렉토리 설정
        try:
            self.cache_dir = self.config["paths"]["cache_dir"]
        except KeyError:
            logger.warning(
                "설정에서 'paths.cache_dir'를 찾을 수 없습니다. 기본값('data/cache')을 사용합니다."
            )
            self.cache_dir = "data/cache"

        self.sequence_dir = Path(self.cache_dir)
        self.lstm_dir = self.sequence_dir / "lstm"
        self.gnn_dir = self.sequence_dir / "gnn"

        # 디렉토리 생성
        self.lstm_dir.mkdir(exist_ok=True, parents=True)
        self.gnn_dir.mkdir(exist_ok=True, parents=True)

    def extract_features(self, data: Any, *args, **kwargs) -> Any:
        """
        베이스 클래스의 추상 메서드 구현

        Args:
            data: 입력 데이터 (List[LotteryNumber])

        Returns:
            시퀀스 특성 추출 결과
        """
        if isinstance(data, list):
            # 기본 매개변수 설정
            lstm_seq_length = kwargs.get("lstm_seq_length", 5)
            gnn_seq_length = kwargs.get("gnn_seq_length", 10)
            build_lstm = kwargs.get("build_lstm", True)
            build_gnn = kwargs.get("build_gnn", True)

            return self.build_sequences(
                data, lstm_seq_length, gnn_seq_length, build_lstm, build_gnn
            )
        else:
            raise ValueError(
                "SequenceFeatureBuilder는 List[LotteryNumber] 형태의 데이터를 필요로 합니다."
            )

    # @profile("build_all_sequences")
    def build_sequences(
        self,
        draw_data: List[LotteryNumber],
        lstm_seq_length: int = 5,
        gnn_seq_length: int = 10,
        build_lstm: bool = True,
        build_gnn: bool = True,
    ) -> Dict[str, Any]:
        """
        모든 시퀀스 특성 생성

        Args:
            draw_data: 당첨 번호 데이터
            lstm_seq_length: LSTM 시퀀스 길이
            gnn_seq_length: GNN 시퀀스 길이
            build_lstm: LSTM 시퀀스 생성 여부
            build_gnn: GNN 시퀀스 생성 여부

        Returns:
            생성된 시퀀스 정보
        """
        self.logger.info("시퀀스 특성 생성 시작")

        # 결과 저장 딕셔너리
        result = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data_count": len(draw_data),
        }

        # LSTM 시퀀스 생성
        if build_lstm:
            result["lstm"] = self._build_lstm_sequences(draw_data, lstm_seq_length)

        # GNN 시퀀스 생성
        if build_gnn:
            result["gnn"] = self._build_gnn_sequences(draw_data, gnn_seq_length)

        self.logger.info("모든 시퀀스 특성 생성 완료")
        return result

    # @profile("build_lstm_sequences")
    def _build_lstm_sequences(
        self, draw_data: List[LotteryNumber], sequence_length: int = 5
    ) -> Dict[str, Any]:
        """
        LSTM 입력용 시퀀스 생성

        Args:
            draw_data: 당첨 번호 데이터
            sequence_length: 시퀀스 길이 (몇 회차를 연속으로 볼지)

        Returns:
            시퀀스 정보
        """
        self.logger.info(f"LSTM 시퀀스 생성 시작 (시퀀스 길이: {sequence_length})")

        # 메모리 추적 시작
        # 성능 모니터링은 performance_monitor에서 처리

        start_time = time.time()

        # 모든 회차의 패턴 벡터 생성
        vectors = self._generate_pattern_vectors(draw_data)

        # 벡터 형태 확인
        if len(vectors) < sequence_length + 1:
            self.logger.error(
                f"데이터가 부족합니다. 최소 {sequence_length + 1}회차 이상 필요함"
            )
            return {
                "success": False,
                "error": f"데이터가 부족합니다. 최소 {sequence_length + 1}회차 이상 필요함",
            }

        # 시퀀스 데이터 구성
        X_sequences = []
        Y_targets = []

        # 시퀀스 생성 (i번째 시퀀스: i-sequence_length부터 i-1까지, 타겟: i)
        for i in range(sequence_length, len(vectors)):
            # 입력 시퀀스 (이전 sequence_length개 회차)
            X_seq = vectors[i - sequence_length : i]

            # 타겟 (현재 회차)
            Y_target = vectors[i]

            X_sequences.append(X_seq)
            Y_targets.append(Y_target)

        # 배열로 변환
        X_array = np.array(X_sequences, dtype=np.float32)
        Y_array = np.array(Y_targets, dtype=np.float32)

        # 메모리 추적 중지
        # 성능 모니터링은 performance_monitor에서 처리
        memory_log = {"memory_used_mb": 0.0}  # 기본값

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"LSTM 시퀀스 생성 완료: {len(X_sequences)}개 ({elapsed_time:.2f}초)"
        )
        self.logger.info(
            f"X 시퀀스 형태: {X_array.shape}, Y 타겟 형태: {Y_array.shape}"
        )
        self.logger.info(f"메모리 사용: {memory_log['memory_used_mb']:.2f}MB")

        # 결과 저장
        x_path, y_path = self._save_lstm_sequences(X_array, Y_array, sequence_length)

        return {
            "success": True,
            "elapsed_time": elapsed_time,
            "x_shape": X_array.shape,
            "y_shape": Y_array.shape,
            "memory_used_mb": memory_log["memory_used_mb"],
            "x_path": x_path,
            "y_path": y_path,
        }

    # @profile("build_gnn_sequences")
    def _build_gnn_sequences(
        self, draw_data: List[LotteryNumber], sequence_length: int = 10
    ) -> Dict[str, Any]:
        """
        GNN 입력용 그래프 시퀀스 생성

        Args:
            draw_data: 당첨 번호 데이터
            sequence_length: 시퀀스 길이 (몇 회차를 연속으로 볼지)

        Returns:
            시퀀스 정보
        """
        self.logger.info(
            f"GNN 그래프 시퀀스 생성 시작 (시퀀스 길이: {sequence_length})"
        )

        # 메모리 추적 시작
        # 성능 모니터링은 performance_monitor에서 처리

        start_time = time.time()

        # 모든 회차의 엣지 추출
        all_edges = self._extract_all_edges(draw_data)

        # 회차 수 확인
        if len(all_edges) < sequence_length + 1:
            self.logger.error(
                f"데이터가 부족합니다. 최소 {sequence_length + 1}회차 이상 필요함"
            )
            return {
                "success": False,
                "error": f"데이터가 부족합니다. 최소 {sequence_length + 1}회차 이상 필요함",
            }

        # 시퀀스 데이터 구성
        X_graph_sequences = []
        Y_node_labels = []

        # 시퀀스 생성 (i번째 시퀀스: i-sequence_length부터 i-1까지, 타겟: i)
        for i in range(sequence_length, len(all_edges)):
            # 입력 그래프 시퀀스 (이전 sequence_length개 회차)
            X_graph_seq = all_edges[i - sequence_length : i]

            # 타겟 (현재 회차의 노드 점수)
            Y_node_label = self._create_node_labels(draw_data[i])

            X_graph_sequences.append(X_graph_seq)
            Y_node_labels.append(Y_node_label)

        # 라벨 배열로 변환
        Y_array = np.array(Y_node_labels, dtype=np.float32)

        # 메모리 추적 중지
        # 성능 모니터링은 performance_monitor에서 처리
        memory_log = {"memory_used_mb": 0.0}  # 기본값

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"GNN 그래프 시퀀스 생성 완료: {len(X_graph_sequences)}개 ({elapsed_time:.2f}초)"
        )
        self.logger.info(f"Y 라벨 형태: {Y_array.shape}")
        self.logger.info(f"메모리 사용: {memory_log['memory_used_mb']:.2f}MB")

        # 결과 저장
        x_path, y_path = self._save_gnn_sequences(
            X_graph_sequences, Y_array, sequence_length
        )

        # 추가로 인접 행렬 생성은 건너뜁니다 (오류 방지)
        # adj_matrices = self.build_adjacency_matrices([X_graph_sequences[:10]])  # 샘플 데이터만

        return {
            "success": True,
            "elapsed_time": elapsed_time,
            "sequence_count": len(X_graph_sequences),
            "y_shape": Y_array.shape,
            "memory_used_mb": memory_log["memory_used_mb"],
            "x_path": x_path,
            "y_path": y_path,
        }

    def _generate_pattern_vectors(
        self, draw_data: List[LotteryNumber]
    ) -> List[np.ndarray]:
        """
        각 회차별 패턴 벡터 생성

        Args:
            draw_data: 당첨 번호 데이터

        Returns:
            각 회차별 패턴 벡터 목록
        """
        self.logger.info(f"패턴 벡터 생성 시작: {len(draw_data)}회차")

        # 벡터 목록
        vectors = []

        # 회차별 패턴 벡터 생성
        for i, draw in enumerate(draw_data):
            # 패턴 특성 추출 (현재 회차까지의 데이터만 사용)
            features = self.pattern_analyzer.extract_pattern_features(
                draw.numbers, draw_data[: i + 1]
            )

            # 특성 벡터화
            vector = self.pattern_analyzer.vectorize_pattern_features(features)

            # 정규화 (0~1 범위로)
            vector = self._normalize_vector(vector, draw_data)

            vectors.append(vector)

            # 진행 상황 로깅 (10% 단위로)
            if i % max(1, len(draw_data) // 10) == 0:
                self.logger.info(
                    f"패턴 벡터 생성 진행: {i}/{len(draw_data)} ({i/len(draw_data)*100:.1f}%)"
                )

        self.logger.info(f"패턴 벡터 생성 완료: {len(vectors)}개")
        return vectors

    def _normalize_vector(
        self, vector: np.ndarray, draw_data: List[LotteryNumber]
    ) -> np.ndarray:
        """
        특성 벡터 정규화

        Args:
            vector: 원본 벡터
            draw_data: 당첨 번호 데이터

        Returns:
            정규화된 벡터
        """
        # 벡터 복사
        normalized = vector.copy()

        # 회차 수 기반 정규화를 적용할 특성 인덱스
        # (임시로 벡터 구조를 기반으로 인덱스 할당, 실제 구현에서는 정확한 인덱스 필요)
        draw_count_based_indices = [0, 1, 2, 3, 4]  # 빈도, 최근성 등

        # 합계 기반 정규화를 적용할 특성 인덱스
        sum_based_indices = [5, 6, 7, 8]  # 합계, 간격 등

        # 최대값 기반 정규화를 적용할 특성 인덱스
        max_based_indices = list(range(9, len(vector)))  # 나머지 특성

        # 회차 수 기반 정규화
        for idx in draw_count_based_indices:
            if idx < len(normalized):
                normalized[idx] = normalized[idx] / len(draw_data)

        # 합계 기반 정규화 (최대 270 = 45+44+43+42+41+40)
        for idx in sum_based_indices:
            if idx < len(normalized):
                normalized[idx] = normalized[idx] / 270.0

        # 최대값 기반 정규화
        for idx in max_based_indices:
            if idx < len(normalized):
                # 이미 0~1 범위면 그대로 유지, 그 외의 경우 45로 나눔
                if normalized[idx] > 1.0:
                    normalized[idx] = normalized[idx] / 45.0

        # 0~1 범위 벗어나는 값 보정
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized

    def _save_lstm_sequences(
        self, X_sequences: np.ndarray, Y_targets: np.ndarray, sequence_length: int
    ) -> Tuple[str, str]:
        """
        LSTM 시퀀스 데이터 저장

        Args:
            X_sequences: 입력 시퀀스
            Y_targets: 타겟 벡터
            sequence_length: 시퀀스 길이

        Returns:
            저장된 파일 경로 튜플
        """
        # 타임스탬프
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 파일명
        x_file_name = f"X_lstm_seq_len{sequence_length}_{timestamp}.npy"
        y_file_name = f"Y_lstm_target_len{sequence_length}_{timestamp}.npy"

        # 경로
        x_path = self.lstm_dir / x_file_name
        y_path = self.lstm_dir / y_file_name

        # 저장
        np.save(x_path, X_sequences)
        np.save(y_path, Y_targets)

        # 가장 최신 버전용 링크 (덮어쓰기)
        latest_x_path = self.lstm_dir / "X_lstm_seq.npy"
        latest_y_path = self.lstm_dir / "Y_lstm_target.npy"

        np.save(latest_x_path, X_sequences)
        np.save(latest_y_path, Y_targets)

        self.logger.info(f"LSTM 시퀀스 데이터 저장 완료:")
        self.logger.info(f"  - X 시퀀스: {x_path}")
        self.logger.info(f"  - Y 타겟: {y_path}")
        self.logger.info(f"  - 최신 버전: {latest_x_path}, {latest_y_path}")

        return str(x_path), str(y_path)

    def _extract_all_edges(
        self, draw_data: List[LotteryNumber]
    ) -> List[List[Tuple[int, int]]]:
        """
        모든 회차의 그래프 엣지 추출

        Args:
            draw_data: 당첨 번호 데이터

        Returns:
            각 회차별 엣지 목록
        """
        self.logger.info(f"그래프 엣지 추출 시작: {len(draw_data)}회차")

        all_edges = []

        # 각 회차별 엣지 추출
        for i, draw in enumerate(draw_data):
            edges = self._extract_edges(draw.numbers)
            all_edges.append(edges)

            # 진행 상황 로깅 (10% 단위로)
            if i % max(1, len(draw_data) // 10) == 0:
                self.logger.info(
                    f"그래프 엣지 추출 진행: {i}/{len(draw_data)} ({i/len(draw_data)*100:.1f}%)"
                )

        self.logger.info(f"그래프 엣지 추출 완료: {len(all_edges)}회차")
        return all_edges

    def _extract_edges(self, numbers: List[int]) -> List[Tuple[int, int]]:
        """
        단일 회차의 그래프 엣지 추출

        Args:
            numbers: 번호 목록

        Returns:
            엣지 목록 (노드 쌍)
        """
        edges = []

        # 모든 번호 쌍으로 엣지 생성
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                # 노드 번호는 1~45 (0-인덱스가 아님)
                edges.append((numbers[i], numbers[j]))

        return edges

    def _create_node_labels(self, draw: LotteryNumber) -> np.ndarray:
        """
        노드 라벨 생성

        Args:
            draw: 당첨 번호 데이터

        Returns:
            노드 라벨 (45차원 벡터, 당첨된 번호는 1, 그 외는 0)
        """
        # 45개 노드에 대한 라벨 (0: 비당첨, 1: 당첨)
        labels = np.zeros(45, dtype=np.float32)

        # 당첨 번호 표시
        for num in draw.numbers:
            # 인덱스는 0부터 시작하므로 1을 빼줌
            labels[num - 1] = 1.0

        return labels

    def _save_gnn_sequences(
        self,
        X_graph_sequences: List[List[Tuple[int, int]]],
        Y_node_labels: np.ndarray,
        sequence_length: int,
    ) -> Tuple[str, str]:
        """
        GNN 그래프 시퀀스 데이터 저장

        Args:
            X_graph_sequences: 그래프 엣지 시퀀스
            Y_node_labels: 노드 라벨
            sequence_length: 시퀀스 길이

        Returns:
            저장된 파일 경로 튜플
        """
        # 타임스탬프
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 파일명
        x_file_name = f"X_gnn_seq_len{sequence_length}_{timestamp}.pkl"
        y_file_name = f"Y_gnn_labels_len{sequence_length}_{timestamp}.npy"

        # 경로
        x_path = self.gnn_dir / x_file_name
        y_path = self.gnn_dir / y_file_name

        # 저장
        with open(x_path, "wb") as f:
            pickle.dump(X_graph_sequences, f)

        np.save(y_path, Y_node_labels)

        # 가장 최신 버전용 링크 (덮어쓰기)
        latest_x_path = self.gnn_dir / "X_gnn_seq.pkl"
        latest_y_path = self.gnn_dir / "Y_gnn_labels.npy"

        with open(latest_x_path, "wb") as f:
            pickle.dump(X_graph_sequences, f)

        np.save(latest_y_path, Y_node_labels)

        self.logger.info(f"GNN 그래프 시퀀스 데이터 저장 완료:")
        self.logger.info(f"  - X 그래프 시퀀스: {x_path}")
        self.logger.info(f"  - Y 노드 라벨: {y_path}")
        self.logger.info(f"  - 최신 버전: {latest_x_path}, {latest_y_path}")

        return str(x_path), str(y_path)

    def build_adjacency_matrices(
        self, graph_sequences: List[List[List[Tuple[int, int]]]]
    ) -> List[List[np.ndarray]]:
        """
        인접 행렬 생성

        Args:
            graph_sequences: 그래프 엣지 시퀀스

        Returns:
            인접 행렬 시퀀스
        """
        self.logger.info("인접 행렬 생성 시작")

        all_matrices = []

        # 각 시퀀스에 대한 인접 행렬 생성
        for i, sequence in enumerate(graph_sequences):
            matrices = []

            # 시퀀스 내 각 그래프에 대한 인접 행렬 생성
            for edges in sequence:
                # 45x45 인접 행렬 (노드 번호는 1~45)
                adj_matrix = np.zeros((45, 45), dtype=np.float32)

                # 엣지 설정
                for edge in edges:
                    # 엣지 구조가 (src, dst) 형태인지 확인
                    if isinstance(edge, tuple) and len(edge) == 2:
                        src, dst = edge
                        # 인덱스는 0부터 시작하므로 1을 빼줌
                        adj_matrix[src - 1, dst - 1] = 1.0
                        adj_matrix[dst - 1, src - 1] = 1.0  # 무방향 그래프
                    else:
                        self.logger.warning(f"잘못된 엣지 형식: {edge}")

                matrices.append(adj_matrix)

            all_matrices.append(matrices)

            # 진행 상황 로깅 (10% 단위로)
            if i % max(1, len(graph_sequences) // 10) == 0:
                self.logger.info(
                    f"인접 행렬 생성 진행: {i}/{len(graph_sequences)} ({i/len(graph_sequences)*100:.1f}%)"
                )

        self.logger.info(f"인접 행렬 생성 완료: {len(all_matrices)}개 시퀀스")

        # 인접 행렬 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adj_path = self.gnn_dir / f"adjacency_matrices_{timestamp}.pkl"

        with open(adj_path, "wb") as f:
            pickle.dump(all_matrices, f)

        self.logger.info(f"인접 행렬 저장 완료: {adj_path}")

        return all_matrices


def build_sequences(
    draw_data: List[LotteryNumber],
    lstm_seq_length: int = 5,
    gnn_seq_length: int = 10,
    build_lstm: bool = True,
    build_gnn: bool = True,
) -> Dict[str, Any]:
    """
    모든 시퀀스 생성 (모듈 레벨 함수)

    Args:
        draw_data: 당첨 번호 데이터
        lstm_seq_length: LSTM 시퀀스 길이
        gnn_seq_length: GNN 시퀀스 길이
        build_lstm: LSTM 시퀀스 생성 여부
        build_gnn: GNN 시퀀스 생성 여부

    Returns:
        생성된 시퀀스 정보
    """
    builder = SequenceFeatureBuilder()
    return builder.build_sequences(
        draw_data,
        lstm_seq_length=lstm_seq_length,
        gnn_seq_length=gnn_seq_length,
        build_lstm=build_lstm,
        build_gnn=build_gnn,
    )
