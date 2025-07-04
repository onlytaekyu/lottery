"""
로또 번호 선택을 위한 강화학습 환경 통합 모듈

임베딩 속성 관련 오류를 해결한 환경 클래스 제공
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import os
from ..utils.unified_logging import get_logger
from ..shared.types import LotteryNumber, PatternAnalysis
from .base_environment import BaseEnvironment

logger = get_logger(__name__)


class LotteryEnvironment(BaseEnvironment):
    """로또 번호 선택을 위한 강화학습 환경"""

    def __init__(
        self,
        train_data: List[Union[Dict[str, Any], LotteryNumber]],
        pattern_analysis: Optional[PatternAnalysis] = None,
        max_steps: int = 6,
    ):
        """
        초기화

        Args:
            train_data: 학습 데이터셋 (Dictionary 또는 LotteryNumber 객체 리스트)
            pattern_analysis: 패턴 분석 결과
            max_steps: 최대 스텝 수 (선택할 번호 개수)
        """
        # Dictionary 형태를 LotteryNumber 객체로 변환
        lottery_data = self._convert_to_lottery_numbers(train_data)

        super().__init__(lottery_data, max_steps)
        self.pattern_analysis = pattern_analysis
        self.logger = logger  # 클래스 내에서 사용할 로거 설정

        # 각 번호별 빈도 정보 정규화
        self.normalized_frequency = self._normalize_frequency()

        # ROI 지표 정규화
        self.normalized_roi = self._normalize_roi_metrics()

        self.reset()

    def _convert_to_lottery_numbers(
        self, data: List[Union[Dict[str, Any], LotteryNumber]]
    ) -> List[LotteryNumber]:
        """
        데이터를 LotteryNumber 객체 리스트로 변환

        Args:
            data: Dictionary 또는 LotteryNumber 객체 리스트

        Returns:
            LotteryNumber 객체 리스트
        """
        lottery_numbers = []

        for item in data:
            if isinstance(item, LotteryNumber):
                lottery_numbers.append(item)
            elif isinstance(item, dict) and "numbers" in item:
                # Dictionary에서 LotteryNumber 객체 생성
                lottery_numbers.append(
                    LotteryNumber(
                        draw_no=item.get("draw_no", 0),
                        numbers=item.get("numbers", []),
                        date=item.get("date", ""),
                    )
                )

        return lottery_numbers

    def _normalize_frequency(self) -> np.ndarray:
        """빈도 정보 정규화"""
        freq = np.zeros(45)
        if self.pattern_analysis is None:
            return freq

        for num, value in self.pattern_analysis.frequency_map.items():
            if 1 <= num <= 45:
                freq[num - 1] = value

        # 정규화 (0~1 범위로)
        if np.sum(freq) > 0:
            freq = freq / np.max(freq)

        return freq

    def _normalize_roi_metrics(self) -> np.ndarray:
        """ROI 지표 정규화"""
        roi_matrix = np.zeros((45, 45))

        # 패턴 분석 결과가 없으면 기본값 반환
        if self.pattern_analysis is None:
            return roi_matrix

        # PatternAnalysis에 roi_metrics가 없는 경우를 위한 안전한 접근
        roi_metrics = getattr(self.pattern_analysis, "pair_frequency", {})

        for (num1, num2), value in roi_metrics.items():
            if 1 <= num1 <= 45 and 1 <= num2 <= 45:
                roi_matrix[num1 - 1, num2 - 1] = value
                roi_matrix[num2 - 1, num1 - 1] = value  # 대칭 행렬

        # 정규화 (0~1 범위로)
        if np.max(roi_matrix) > 0:
            roi_matrix = roi_matrix / np.max(roi_matrix)

        return roi_matrix

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        # 선택된 번호 초기화
        self.selected = np.zeros(45, dtype=np.int32)
        self.steps = 0
        return self.selected.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경 진행

        Args:
            action: 선택할 번호 인덱스 (0-44)

        Returns:
            state: 새로운 상태
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        if not (0 <= action < 45):
            raise ValueError(f"유효하지 않은 액션: {action}")

        # 이미 선택된 번호인 경우 패널티
        if self.selected[action] == 1:
            return self.selected.copy(), -1.0, False, {"error": "already_selected"}

        # 번호 선택
        self.selected[action] = 1
        self.steps += 1

        # 보상 계산
        reward = self._calculate_reward(action)

        # 종료 여부 확인
        done = self.steps >= self.max_steps

        # 추가 정보
        info = {
            "steps": self.steps,
            "selected_numbers": [i + 1 for i, v in enumerate(self.selected) if v == 1],
        }

        return self.selected.copy(), reward, done, info

    def _calculate_reward(self, action: int) -> float:
        """
        보상 계산

        Args:
            action: 선택한 번호 인덱스 (0-44)

        Returns:
            reward: 보상값
        """
        reward = 0.0

        # 패턴 분석 결과가 없으면 기본 보상만 반환
        if self.pattern_analysis is None:
            return reward

        # 1. 빈도 기반 보상
        reward += self.normalized_frequency[action] * 0.2

        # 2. 클러스터 준수 보상
        cluster_reward = 0.0
        # 안전한 클러스터 접근
        clusters = getattr(self.pattern_analysis, "clusters", [])
        if isinstance(clusters, list):
            for cluster in clusters:
                if action + 1 in cluster:  # 0-based -> 1-based
                    selected_in_cluster = sum(
                        1 for n in cluster if self.selected[n - 1] == 1
                    )
                    if selected_in_cluster > 0:
                        cluster_reward = 0.3 * (selected_in_cluster / len(cluster))
                        break
        reward += cluster_reward

        # 3. 핫넘버/콜드넘버 보상
        num = action + 1  # 1-based 인덱스로 변환
        hot_numbers = getattr(self.pattern_analysis, "hot_numbers", set())
        cold_numbers = getattr(self.pattern_analysis, "cold_numbers", set())

        if num in hot_numbers:
            reward += 0.2
        if num in cold_numbers:
            reward -= 0.1

        # 4. 분포 적합성 보상
        # - 현재 선택된 번호 세트가 평균 합계에 근접하는지
        # - 간격 분포가 적절한지
        selected_nums = [i + 1 for i, v in enumerate(self.selected) if v == 1]
        selected_nums.append(action + 1)

        if len(selected_nums) >= 2:
            # 간격 계산
            sorted_nums = sorted(selected_nums)
            gaps = [
                sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)
            ]
            avg_gap = np.mean(gaps)

            # 이상적인 간격 (통계적 평균)에 가까울수록 높은 보상
            ideal_gap = 8.0  # 통계 기반 조정
            gap_diff = abs(avg_gap - ideal_gap)
            # 부동소수점 계산을 먼저 수행한 후 결과를 변수에 저장하여 타입 문제 해결
            similarity_ratio = float(1 - gap_diff / ideal_gap)
            gap_reward = 0.2 * max(0.0, similarity_ratio)
            reward += gap_reward

        # 5. ROI 지표 기반 보상
        selected_indices = [
            i for i, v in enumerate(self.selected) if v == 1 and i != action
        ]
        roi_bonus = 0.0
        for idx in selected_indices:
            roi_bonus += self.normalized_roi[action, idx]

        # max 함수의 인자를 모두 정수로 변환하여 타입 문제 해결
        reward += roi_bonus * 0.3 / max(1, int(len(selected_indices)))

        return reward


class EnhancedLotteryEnvironment(LotteryEnvironment):
    """GNN 임베딩을 활용한 향상된 로또 환경"""

    def __init__(
        self,
        train_data: List[Union[Dict[str, Any], LotteryNumber]],
        pattern_analysis: Optional[PatternAnalysis] = None,
        embeddings: Optional[np.ndarray] = None,
        max_steps: int = 6,
    ):
        """
        초기화

        Args:
            train_data: 학습 데이터셋 (Dictionary 또는 LotteryNumber 객체 리스트)
            pattern_analysis: 패턴 분석 결과
            embeddings: GNN 임베딩 (45 x D 크기)
            max_steps: 최대 스텝 수 (선택할 번호 개수)
        """
        super().__init__(train_data, pattern_analysis, max_steps)
        self.logger = logger  # 로거 설정

        # 클러스터 임베딩 로드
        self.cluster_embeddings = np.zeros((45, 32))  # 기본값으로 초기화
        if os.path.exists("data/cluster_embeddings.npy"):
            try:
                self.cluster_embeddings = np.load("data/cluster_embeddings.npy")
                self.logger.info(
                    "클러스터 임베딩 로드 성공: data/cluster_embeddings.npy"
                )
            except Exception as e:
                self.logger.error(f"클러스터 임베딩 로드 실패: {str(e)}")
                # 기본값인 self.cluster_embeddings 유지
        else:
            self.logger.warning(
                "클러스터 임베딩 파일이 없습니다. 기본 임베딩을 사용합니다."
            )

        # GNN 임베딩 저장 (None인 경우 기본값 사용)
        if embeddings is None:
            self.logger.warning("임베딩 없음, 임의 임베딩 생성")
            self.embeddings = np.random.normal(0, 0.1, (45, 32))
        else:
            self.embeddings = embeddings

        # 임베딩 차원 설정 - 오류 방지를 위한 예외 처리 추가
        try:
            if len(self.embeddings.shape) > 1:
                self.embedding_dim = self.embeddings.shape[1]
            else:
                # 1차원 배열이면 기본값 설정
                self.logger.warning(
                    "임베딩이 1차원 배열입니다. 기본 차원(32)을 사용합니다."
                )
                self.embedding_dim = 32
        except Exception as e:
            self.logger.error(f"임베딩 차원 설정 실패: {str(e)}")
            self.embedding_dim = 32

        self.current_numbers = []

    def get_embedding(self, numbers: List[int]) -> np.ndarray:
        """
        선택된 번호들의 평균 임베딩 계산

        Args:
            numbers: 숫자 리스트 (1-45)

        Returns:
            평균 임베딩 벡터
        """
        # 번호가 비어있는 경우 기본값 반환
        if not numbers:
            return np.zeros(self.embedding_dim)

        # 각 번호에 대한 임베딩 추출
        embeddings = []
        for num in numbers:
            if 1 <= num <= 45:
                embeddings.append(self.embeddings[num - 1])

        # 평균 임베딩 계산
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)

    def get_cluster_embedding_vector(self, numbers: List[int]) -> np.ndarray:
        """
        선택된 번호들의 클러스터 임베딩 벡터 추출

        Args:
            numbers: 선택된 번호 리스트

        Returns:
            클러스터 임베딩 벡터
        """
        if not numbers:
            return np.zeros(32)  # 기본 임베딩 차원

        embeddings = []
        for num in numbers:
            if 1 <= num <= 45:
                embeddings.append(self.cluster_embeddings[num - 1])

        # 평균 임베딩 반환
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(32)

    def get_state(self) -> np.ndarray:
        """
        현재 상태 벡터 반환 (선택된 번호 + GNN 클러스터 임베딩)

        Returns:
            상태 벡터 (선택 여부 + 클러스터 임베딩)
        """
        freq_vector = self.compute_frequency_vector(self.current_numbers)
        cluster_vector = self.get_cluster_embedding_vector(self.current_numbers)
        return np.concatenate([freq_vector, cluster_vector])

    def compute_frequency_vector(self, numbers: List[int]) -> np.ndarray:
        """
        번호 빈도 벡터 계산

        Args:
            numbers: 선택된 번호 리스트

        Returns:
            빈도 벡터
        """
        # 기본 빈도 벡터 (정규화된 번호별 출현 빈도)
        freq_vector = self.normalized_frequency.copy()

        # 이미 선택된 번호는 표시
        selected_vector = np.zeros(45)
        for num in numbers:
            if 1 <= num <= 45:
                selected_vector[num - 1] = 1.0

        # 빈도 벡터와 선택 벡터 결합
        combined_vector = np.concatenate([freq_vector, selected_vector])
        return combined_vector

    def reset(self) -> np.ndarray:
        """
        환경 초기화

        Returns:
            상태 벡터 (선택 상태 + 클러스터 임베딩)
        """
        # 기본 초기화
        self.selected = np.zeros(45, dtype=np.int32)
        self.steps = 0
        self.current_numbers = []

        # 향상된 상태 벡터 반환
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경 진행

        Args:
            action: 선택할 번호 인덱스 (0-44)

        Returns:
            combined_state: 새로운 상태 (선택 상태 + 임베딩)
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        if not (0 <= action < 45):
            raise ValueError(f"유효하지 않은 액션: {action}")

        # 이미 선택된 번호인 경우 패널티
        if self.selected[action] == 1:
            return self.get_state(), -1.0, False, {"error": "already_selected"}

        # 번호 선택
        self.selected[action] = 1
        self.steps += 1

        # 선택된 번호 추가 (1-based)
        self.current_numbers.append(action + 1)

        # 보상 계산
        reward = self._calculate_reward(action)

        # 종료 여부 확인
        done = self.steps >= self.max_steps

        # 추가 정보
        info = {
            "steps": self.steps,
            "selected_numbers": self.current_numbers,
        }

        # 향상된 상태 반환
        return self.get_state(), reward, done, info
