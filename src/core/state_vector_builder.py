import numpy as np
import torch
import json
from typing import Dict, List, Any, Tuple, Optional, Union
import gc
import logging
from pathlib import Path
from ..shared.types import LotteryNumber, PatternAnalysis
from ..utils.memory_manager import MemoryManager, MemoryConfig
from ..utils.error_handler_refactored import get_logger
from ..utils.normalizer import Normalizer
from ..utils.unified_config import ConfigProxy

# 로거 설정
logger = get_logger(__name__)


class ClusterAnalyzer:
    """클러스터 분석을 위한 간단한 구현"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """클러스터 분석기 초기화"""
        self.config = config or {}

    def analyze_clusters(self, number_lists: List[List[int]]) -> Dict[str, Any]:
        """클러스터 분석 수행

        Args:
            number_lists: 번호 리스트들

        Returns:
            클러스터 분석 결과
        """
        # 간단한 구현 - 동시 출현 행렬 기반 임베딩 생성
        num_nodes = 45

        # 동시 출현 행렬 계산
        cooccurrence = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for numbers in number_lists:
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    # 인덱스는 0부터 시작하므로 번호에서 1을 뺌
                    idx1, idx2 = numbers[i] - 1, numbers[j] - 1
                    cooccurrence[idx1, idx2] += 1
                    cooccurrence[idx2, idx1] += 1  # 대칭 행렬

        # 단순한 특이값 분해로 임베딩 생성
        try:
            U, s, Vh = np.linalg.svd(cooccurrence, full_matrices=False)
            # 차원 축소 (32차원 사용)
            embedding_dim = min(32, U.shape[1])
            embeddings = U[:, :embedding_dim] * np.sqrt(s[:embedding_dim])
        except Exception as e:
            logger.warning(f"SVD 계산 실패, 기본 임베딩 사용: {str(e)}")
            embeddings = np.random.normal(0, 0.1, (num_nodes, 32))

        return {"embeddings": embeddings, "cooccurrence": cooccurrence}


def build_pair_graph(data: List[List[int]]) -> np.ndarray:
    """번호 쌍 그래프 구축

    Args:
        data: 과거 로또 번호 데이터

    Returns:
        인접 행렬
    """
    num_nodes = 45
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 번호 쌍 빈도 계산
    for numbers in data:
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                # 0부터 시작하는 인덱스로 변환
                idx1, idx2 = numbers[i] - 1, numbers[j] - 1
                adjacency[idx1, idx2] += 1
                adjacency[idx2, idx1] += 1  # 대칭 행렬

    # 정규화
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
    adjacency = adjacency / row_sums

    return adjacency


class StateVectorBuilder:
    """
    상태 벡터 생성기

    이 클래스는 로또 번호 추천을 위한 상태 벡터를 생성합니다.
    상태 벡터는 패턴 분석, 클러스터 분석, 트렌드 등을 통합한 정보를 포함하며,
    강화학습 및 GNN 모델의 입력으로 사용됩니다.

    이 클래스는 내부에 정의된 ClusterAnalyzer 클래스를 사용하여
    번호 간 관계를 모델링합니다.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        device: Optional[str] = None,
        memory_manager: Optional[MemoryManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # 장치 설정 - GPU 우선
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"상태 벡터 빌더 초기화 - 사용 장치: {self.device}")

        # 설정 객체 초기화
        self.config = ConfigProxy(config or {})

        # 정규화 유틸리티 초기화
        self.normalizer = Normalizer(config)

        # 로거 설정
        self.logger = get_logger(__name__)

        # 특성 벡터 캐시 초기화
        from ..utils.state_vector_cache import get_cache

        self.vector_cache = get_cache(config)

        # 메모리 관리자 초기화 (제공되지 않은 경우)
        self.embedding_dim = embedding_dim
        if memory_manager is None:
            memory_config = MemoryConfig(
                max_memory_usage=0.8,  # 80% 최대 메모리 사용량
                cache_size=128 * 1024 * 1024,  # 128MB 캐시
                memory_track_interval=5,
                memory_usage_warning=0.9,
                pool_size=16,  # 작은 메모리 풀 설정
                gpu_ids=[0] if torch.cuda.is_available() else [],
            )
            self.memory_manager = MemoryManager(memory_config)
        else:
            self.memory_manager = memory_manager

        # 클러스터 분석기 초기화
        self.cluster_analyzer = ClusterAnalyzer()

        # 캐시 초기화
        self._state_vector_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # 오류 복구 변수
        self._error_count = 0
        self._max_errors = 3

        # 특성 이름 초기화
        self.feature_names = []

    def build_state_vector(
        self,
        draw_history: List[LotteryNumber],
        pattern_analysis: PatternAnalysis,
        cluster_embeddings: Optional[np.ndarray] = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        모든 패턴/클러스터/트렌드 데이터를 통합하는 상태 벡터 생성

        이 메서드는 ClusterAnalyzer의 결과를 활용하여 번호 간 관계를 효과적으로 모델링합니다.
        cluster_embeddings 매개변수를 통해 사전 계산된 클러스터 임베딩을 전달하거나,
        제공되지 않은 경우 자동으로 계산합니다.

        Args:
            draw_history: 로또 당첨 번호 이력
            pattern_analysis: 패턴 분석 결과
            cluster_embeddings: 사전 계산된 클러스터 임베딩 (선택적)
            use_cache: 캐시 사용 여부

        Returns:
            np.ndarray: 통합된 상태 벡터

        Raises:
            ValueError: 입력 데이터가 유효하지 않거나 필수 속성이 누락된 경우
            RuntimeError: 상태 벡터 생성 중 오류 발생 시
        """
        # 입력 검증
        if not draw_history:
            error_msg = "당첨 번호 이력이 비어 있습니다."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 특성 이름 초기화
        self.feature_names = []

        # 캐시 키 생성 (최근 5회 데이터 기반)
        if use_cache and len(draw_history) >= 5:
            cache_key = self._generate_cache_key(draw_history[-5:])

            # 중앙 집중식 캐시 확인
            cached_vector = self.vector_cache.get(cache_key)
            if cached_vector is not None:
                self.logger.debug(f"상태 벡터 캐시 히트: 키 {cache_key[:10]}...")

                # 특성 이름도 로드
                from ..utils.feature_name_tracker import load_feature_names

                cache_dir = Path(self.config.safe_get("paths.cache_dir", "data/cache"))

                # 새로운 네이밍 규칙의 파일 확인
                feature_names_file = cache_dir / "state_vector.names.json"
                if not feature_names_file.exists():
                    # 기존 네이밍 규칙의 파일 확인
                    feature_names_file = cache_dir / "state_vector_feature_names.json"

                if feature_names_file.exists():
                    self.feature_names = load_feature_names(str(feature_names_file))
                else:
                    # 기본 이름 생성
                    self.feature_names = [
                        f"feature_{i}" for i in range(len(cached_vector))
                    ]
                    self.logger.warning(
                        "특성 이름 파일을 찾을 수 없어 기본 이름을 사용합니다."
                    )

                return cached_vector

            # 로컬 캐시 확인 (이전 방식 - 호환성 유지)
            if cache_key in self._state_vector_cache:
                self._cache_hits += 1
                return self._state_vector_cache[cache_key]
            self._cache_misses += 1

        # 메모리 관리자 스코프 내에서 실행
        with self.memory_manager.allocation_scope():
            # 빈도 통계 (정규화)
            freq_stats = self._normalize_frequency(pattern_analysis.frequency_map)
            # 특성 이름 추가
            freq_names = [f"number_{i+1}_freq" for i in range(45)]
            self.feature_names.extend(freq_names)

            # 시간적 트렌드 벡터 (최근 20회 추첨 트렌드)
            trend_vector, trend_names = self._extract_trend_vector(draw_history[-20:])
            self.feature_names.extend(trend_names)

            # 분포 프로필 (간격/위치 분포)
            distribution, dist_names = self._create_distribution_profile(
                draw_history, pattern_analysis
            )
            self.feature_names.extend(dist_names)

            # 클러스터 임베딩 생성 또는 사용
            if cluster_embeddings is None:
                # 클러스터 임베딩 계산
                cluster_embeddings = self._compute_cluster_embeddings(draw_history)

            # 클러스터 임베딩 평균 계산
            if torch.cuda.is_available() and self.device == "cuda":
                # GPU 가속 평균 계산
                cluster_tensor = torch.tensor(
                    cluster_embeddings, dtype=torch.float32, device="cuda"
                )
                cluster_mean = cluster_tensor.mean(dim=0).cpu().numpy()
            else:
                cluster_mean = cluster_embeddings.mean(axis=0)

            # 특성 이름 추가 (클러스터 임베딩)
            cluster_names = [f"cluster_embedding_{i}" for i in range(len(cluster_mean))]
            self.feature_names.extend(cluster_names)

            # 모든 벡터 결합
            state_vector = np.concatenate(
                [
                    freq_stats,  # 번호별 빈도 (45)
                    trend_vector,  # 트렌드 분석 (20+)
                    distribution,  # 분포 프로필 (20+)
                    cluster_mean,  # 클러스터 임베딩 (32)
                ]
            )

            # 캐시에 상태 벡터 저장
            if use_cache and len(draw_history) >= 5:
                cache_key = self._generate_cache_key(draw_history[-5:])
                # 중앙 집중식 캐시에 저장
                self.vector_cache.set(cache_key, state_vector)
                # 로컬 캐시에도 저장 (이전 방식 - 호환성 유지)
                self._state_vector_cache[cache_key] = state_vector

                # 특성 이름도 저장
                from ..utils.feature_name_tracker import save_feature_names

                cache_dir = Path(self.config.safe_get("paths.cache_dir", "data/cache"))
                cache_dir.mkdir(parents=True, exist_ok=True)

                # 새로운 네이밍 규칙으로 저장
                feature_names_file = cache_dir / "state_vector.names.json"
                save_feature_names(self.feature_names, str(feature_names_file))

                # 기존 네이밍 규칙으로도 저장 (호환성 유지)
                legacy_file = cache_dir / "state_vector_feature_names.json"
                save_feature_names(self.feature_names, str(legacy_file))

                self.logger.debug(
                    f"상태 벡터 특성 이름 {len(self.feature_names)}개 저장 완료"
                )

            return state_vector

    def _compute_cluster_embeddings(
        self, draw_history: List[LotteryNumber]
    ) -> np.ndarray:
        """
        로또 번호에 대한 클러스터 임베딩을 계산

        이 메서드는 ClusterAnalyzer를 사용하여 번호 간 관계를 분석하고
        임베딩을 생성합니다.

        Args:
            draw_history: 로또 당첨 번호 이력

        Returns:
            np.ndarray: 클러스터 임베딩 (45 x embedding_dim)

        Raises:
            ValueError: 입력 데이터가 유효하지 않거나 필수 속성이 누락된 경우
            RuntimeError: 임베딩 계산 중 오류 발생 시
        """
        cooccurrence = None
        adjacency_matrix = None

        try:
            # 번호 목록으로 변환
            number_lists = []
            for draw in draw_history:
                # LotteryNumber 객체에서 numbers 속성만 추출하여 정렬
                if hasattr(draw, "numbers"):
                    number_lists.append(sorted(draw.numbers))  # 번호 오름차순 정렬 보장
                # 리스트나 다른 타입인 경우
                elif isinstance(draw, list) and all(isinstance(n, int) for n in draw):
                    number_lists.append(sorted(draw))  # 리스트인 경우 정렬
                else:
                    error_msg = f"지원되지 않는 데이터 형식: {type(draw)}"
                    logger.error(error_msg)
                    raise TypeError(error_msg)

            # 유효한 데이터가 없는 경우
            if not number_lists:
                error_msg = "유효한 번호 데이터가 없습니다."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 클러스터 분석기를 사용하여 클러스터 정보 획득
            cluster_result = self.cluster_analyzer.analyze_clusters(number_lists)

            # 임베딩 추출
            embeddings = np.array(cluster_result.get("embeddings", []))

            # 임베딩 크기 확인
            if embeddings.size == 0 or embeddings.shape[0] != 45:
                error_msg = f"클러스터 임베딩 생성 실패: 예상 행 수=45, 실제={embeddings.shape[0] if embeddings.size > 0 else 0}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # 인접 행렬을 사용하여 임베딩 보강
            adjacency_matrix = build_pair_graph(number_lists)

            # 고유값 분해로 보조 임베딩 생성
            eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)
            # 상위 8개 고유벡터 선택
            top_eigenvectors = eigenvectors[:, -8:]

            # 임베딩에 통합
            if embeddings.shape[1] >= 8:
                embeddings[:, :8] = 0.5 * embeddings[:, :8] + 0.5 * top_eigenvectors

            return embeddings

        finally:
            # 메모리 해제
            if cooccurrence is not None:
                del cooccurrence
            if adjacency_matrix is not None:
                del adjacency_matrix

            # 가비지 컬렉션 실행
            gc.collect()

            # CUDA 메모리 정리 (필요시)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_cache_key(self, recent_draws: List[LotteryNumber]) -> str:
        """캐시 키 생성"""
        # 최근 5회 당첨 번호와 회차 정보로 키 생성
        key_parts = []
        for draw in recent_draws:
            # draw_no 속성이 있으면 사용
            if hasattr(draw, "draw_no"):
                draw_id = draw.draw_no
            # 없으면 fallback
            else:
                draw_id = "unknown"

            # 번호는 항상 오름차순으로 정렬 (numbers 속성이 있는지 확인)
            if hasattr(draw, "numbers"):
                numbers_str = "_".join(str(n) for n in sorted(draw.numbers))
                key_parts.append(f"{draw_id}:{numbers_str}")
            else:
                key_parts.append(f"{draw_id}:unknown_numbers")
        return "|".join(key_parts)

    def _normalize_frequency(
        self, frequency: Dict[int, Union[int, float]]
    ) -> np.ndarray:
        """번호별 빈도를 정규화된 벡터로 변환

        Args:
            frequency: 번호별 빈도 맵

        Returns:
            np.ndarray: 정규화된 빈도 벡터
        """
        # 번호별 정규화된 빈도를 담을 벡터
        freq_vector = np.zeros(45, dtype=np.float32)

        # 데이터가 있는 경우
        if frequency:
            # 빈도 값 채우기
            for num, freq in frequency.items():
                if 1 <= num <= 45:
                    freq_vector[num - 1] = freq

            # 정규화 방법 결정
            normalization_method = self.config.safe_get(
                "normalization.feature_vector",
                "zscore",  # 설정에 맞게 기본값 변경: zscore
            )

            # 정규화 방식에 따라 처리
            if normalization_method.lower() == "zscore":
                # Z-Score 정규화 적용
                freq_vector, stats = self.normalizer.z_score_normalize(
                    freq_vector, return_stats=True
                )

                # 정규화 통계 로깅
                mean_val = float(np.mean(stats["mean"]))
                std_val = float(np.mean(stats["std"]))
                self.logger.debug(
                    f"빈도 벡터 Z-Score 정규화: 평균={mean_val:.4f}, 표준편차={std_val:.4f}"
                )

                # 이상치 완화를 위한 클리핑 (-3 ~ 3 범위로 제한)
                freq_vector = np.clip(freq_vector, -3.0, 3.0)

            elif normalization_method.lower() == "minmax":
                # Min-Max 정규화 적용
                freq_vector, stats = self.normalizer.min_max_normalize(
                    freq_vector, return_stats=True
                )

                # 정규화 통계 로깅
                min_val = float(np.mean(stats["min"]))
                max_val = float(np.mean(stats["max"]))
                self.logger.debug(
                    f"빈도 벡터 Min-Max 정규화: 범위={min_val:.4f}~{max_val:.4f}"
                )

            else:
                # 기존 Min-Max 정규화 - 합이 1이 되도록 (fallback)
                freq_sum = freq_vector.sum()
                if freq_sum > 0:
                    freq_vector = freq_vector / freq_sum
                self.logger.warning(
                    f"알 수 없는 정규화 방식: {normalization_method}, 기본 정규화 적용"
                )
        else:
            # 데이터가 없는 경우 균일 분포
            freq_vector = np.ones(45, dtype=np.float32) / 45

        return freq_vector

    def _extract_trend_vector(
        self, recent_draws: List[LotteryNumber]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        최근 추첨 결과로부터 트렌드 특성 벡터를 추출합니다.

        Args:
            recent_draws: 최근 추첨 이력

        Returns:
            트렌드 특성 벡터, 특성 이름 목록
        """
        # 홀짝 구성 히스토리 (5회)
        odd_even_history = []
        # 구간 분포 히스토리 (5회, 3개 구간)
        range_history = []
        # 합계 히스토리 (5회)
        sum_history = []

        # 히스토리 구성
        for draw in recent_draws[-5:]:
            numbers = draw.numbers
            # 홀짝 비율
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            even_count = 6 - odd_count
            odd_even_ratio = odd_count / 6.0
            odd_even_history.append(odd_even_ratio)

            # 구간 분포 (3개 구간: 1-15, 16-30, 31-45)
            r1 = sum(1 for n in numbers if 1 <= n <= 15)
            r2 = sum(1 for n in numbers if 16 <= n <= 30)
            r3 = sum(1 for n in numbers if 31 <= n <= 45)
            range_history.append([r1 / 6.0, r2 / 6.0, r3 / 6.0])

            # 합계
            total_sum = sum(numbers)
            # 일반적인 합계 범위: 21-279, 평균 약 150
            normalized_sum = (total_sum - 21) / (279 - 21)
            sum_history.append(normalized_sum)

        # 트렌드 추출: 홀짝 균형 추세
        odd_even_trend = np.array(odd_even_history[-5:], dtype=np.float32)

        # 트렌드 추출: 구간 분포 추세
        range_trend = np.array(range_history[-5:], dtype=np.float32).flatten()

        # 트렌드 추출: 합계 추세
        sum_trend = np.array(sum_history[-5:], dtype=np.float32)

        # 결합된 벡터
        trend_vector = np.concatenate([odd_even_trend, range_trend, sum_trend])

        # 특성 이름 생성
        feature_names = []
        # 홀짝 추세 (5개)
        for i in range(5):
            feature_names.append(f"odd_even_trend_{i+1}")
        # 구간 분포 추세 (5회 x 3구간 = 15개)
        for i in range(5):
            for j in range(3):
                feature_names.append(f"range_trend_{i+1}_{j+1}")
        # 합계 추세 (5개)
        for i in range(5):
            feature_names.append(f"sum_trend_{i+1}")

        return trend_vector, feature_names

    def _create_distribution_profile(
        self, draw_history: List[LotteryNumber], pattern_analysis: PatternAnalysis
    ) -> Tuple[np.ndarray, List[str]]:
        """추첨 분포 프로필 생성"""
        try:
            profile = np.zeros(10)
            feature_names = []

            if not draw_history:
                return profile, feature_names

            # 최근 10회 추첨 데이터만 사용
            recent_draws = (
                draw_history[-10:] if len(draw_history) > 10 else draw_history
            )

            # 1. 홀짝 비율 (0-1 정규화)
            odd_counts = []
            for draw in recent_draws:
                # LotteryNumber 객체인 경우
                if hasattr(draw, "numbers"):
                    numbers = draw.numbers
                # 리스트인 경우
                elif isinstance(draw, list) and all(isinstance(n, int) for n in draw):
                    numbers = draw
                else:
                    logger.warning(
                        f"지원되지 않는 데이터 형식: {type(draw)}, 건너뜁니다."
                    )
                    continue

                odd_counts.append(sum(1 for num in numbers if num % 2 == 1))

            profile[0] = (
                np.mean(odd_counts) / 6 if odd_counts else 0
            )  # 6개 번호 중 홀수 비율
            feature_names.append("odd_ratio")

            # 2. 합계 분포 (100-200 범위가 일반적)
            sums = []
            for draw in recent_draws:
                # LotteryNumber 객체인 경우
                if hasattr(draw, "numbers"):
                    numbers = draw.numbers
                # 리스트인 경우
                elif isinstance(draw, list) and all(isinstance(n, int) for n in draw):
                    numbers = draw
                else:
                    continue

                sums.append(sum(numbers))

            profile[1] = (
                np.mean(sums) / 270 if sums else 0
            )  # 최대 가능 합계 270으로 정규화
            feature_names.append("sum_norm")

            # 3. 범위별 분포 (1-9, 10-19, 20-29, 30-39, 40-45)
            ranges = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)]
            range_counts = np.zeros(5)

            valid_draws = []  # 유효한 추첨 데이터만 추적
            for i, (start, end) in enumerate(ranges):
                for draw in recent_draws:
                    # LotteryNumber 객체인 경우
                    if hasattr(draw, "numbers"):
                        numbers = draw.numbers
                    # 리스트인 경우
                    elif isinstance(draw, list) and all(
                        isinstance(n, int) for n in draw
                    ):
                        numbers = draw
                    else:
                        continue

                    if draw not in valid_draws:
                        valid_draws.append(draw)

                    range_counts[i] += sum(1 for num in numbers if start <= num <= end)

            # 각 범위별 평균 비율 계산
            total_counts = (
                len(valid_draws) * 6 if valid_draws else 1
            )  # 총 번호 수 (0으로 나누기 방지)
            for i in range(5):
                profile[i + 2] = range_counts[i] / total_counts
                feature_names.append(f"range_{i+1}_ratio")

            # 4. 연속 번호 평균 개수
            consecutive_counts = []
            for draw in recent_draws:
                # LotteryNumber 객체인 경우
                if hasattr(draw, "numbers"):
                    numbers = draw.numbers
                # 리스트인 경우
                elif isinstance(draw, list) and all(isinstance(n, int) for n in draw):
                    numbers = draw
                else:
                    continue

                sorted_nums = sorted(numbers)
                count = sum(
                    1
                    for i in range(len(sorted_nums) - 1)
                    if sorted_nums[i + 1] - sorted_nums[i] == 1
                )
                consecutive_counts.append(count)

            profile[7] = (
                np.mean(consecutive_counts) / 5 if consecutive_counts else 0
            )  # 최대 5쌍의 연속성
            feature_names.append("consecutive_count_ratio")

            # 5. 번호 간 평균 간격
            gap_means = []
            for draw in recent_draws:
                # LotteryNumber 객체인 경우
                if hasattr(draw, "numbers"):
                    numbers = draw.numbers
                # 리스트인 경우
                elif isinstance(draw, list) and all(isinstance(n, int) for n in draw):
                    numbers = draw
                else:
                    continue

                sorted_nums = sorted(numbers)
                gaps = [
                    sorted_nums[i + 1] - sorted_nums[i]
                    for i in range(len(sorted_nums) - 1)
                ]
                gap_means.append(np.mean(gaps) if gaps else 0)

            profile[8] = (
                np.mean(gap_means) / 9 if gap_means else 0
            )  # 평균 간격 정규화 (최대 간격 가정 9)
            feature_names.append("avg_gap_norm")

            # 6. 핫/콜드 번호 비율
            hot_cold_ratio = 0
            for draw in recent_draws:
                # LotteryNumber 객체인 경우
                if hasattr(draw, "numbers"):
                    numbers = draw.numbers
                # 리스트인 경우
                elif isinstance(draw, list) and all(isinstance(n, int) for n in draw):
                    numbers = draw
                else:
                    continue

                hot_count = sum(
                    1 for num in numbers if num in pattern_analysis.hot_numbers
                )
                cold_count = sum(
                    1 for num in numbers if num in pattern_analysis.cold_numbers
                )
                hot_cold_ratio += (
                    hot_count - cold_count + 6
                ) / 12  # -6~6 -> 0~1 정규화

            profile[9] = hot_cold_ratio / len(valid_draws) if valid_draws else 0
            feature_names.append("hot_cold_ratio")

            return profile, feature_names
        except Exception as e:
            logger.error(f"분포 프로필 생성 중 오류: {str(e)}")
            return np.zeros(10), []

    def _cleanup_memory(self) -> None:
        """메모리 정리 및 캐시 초기화"""
        try:
            # 텐서 캐시 정리
            self._state_vector_cache = {}

            # CUDA 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 가비지 컬렉션 실행
            gc.collect()

            logger.info("메모리 정리 완료")
        except Exception as e:
            logger.error(f"메모리 정리 중 오류: {str(e)}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_ratio": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0
            ),
            "cache_size": len(self._state_vector_cache),
            "memory_usage_mb": (
                sum(arr.nbytes for arr in self._state_vector_cache.values())
                / (1024 * 1024)
                if self._state_vector_cache
                else 0
            ),
        }

    def __del__(self):
        """소멸자: 자원 정리"""
        self._cleanup_memory()

    def get_feature_names(self) -> List[str]:
        """
        현재 벡터에 사용된 특성 이름 목록을 반환합니다.

        Returns:
            특성 이름 목록
        """
        # 특성 이름이 비어 있으면 기본 이름 생성
        if not self.feature_names:
            self.logger.warning("특성 이름이 비어 있습니다. 기본 이름을 생성합니다.")
            # 일반적인 상태 벡터 크기는 약 100-110 차원
            default_size = 110
            return [f"feature_{i}" for i in range(default_size)]

        return self.feature_names
