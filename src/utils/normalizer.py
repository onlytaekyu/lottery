"""
데이터 정규화 유틸리티 모듈

이 모듈은 Z-Score, Min-Max 등 다양한 정규화 방식을 구현한 함수들을 제공합니다.
DAEBAK_AI 시스템 전체에서 일관된 정규화 적용을 위해 사용됩니다.
"""

import numpy as np
from typing import List, Union, Dict, Any, Optional
from ..utils.unified_logging import get_logger
from ..utils.unified_config import ConfigProxy

# 로거 설정
logger = get_logger(__name__)


class Normalizer:
    """
    다양한 정규화 방식을 제공하는 클래스

    정규화 방식:
    - Z-Score: (x - mean) / std
    - Min-Max: (x - min) / (max - min)

    모든 정규화 함수는 원본 배열의 통계를 계산하고 결과와 함께 반환합니다.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigProxy]] = None):
        """
        Normalizer 초기화

        Args:
            config: 설정 객체 또는 딕셔너리
        """
        self.config = config if config is not None else {}
        self.logger = get_logger(__name__)

        # 설정에서 fallback 값 가져오기
        if isinstance(config, dict):
            self.zero_std_fallback = (
                config.get("normalization", {})
                .get("fallback", {})
                .get("zero_std", 1e-8)
            )
            self.identical_scores_fallback = (
                config.get("normalization", {})
                .get("fallback", {})
                .get("identical_scores", 0.5)
            )
        elif config is not None and hasattr(config, "safe_get"):
            self.zero_std_fallback = config.safe_get(
                "normalization.fallback.zero_std", 1e-8
            )
            self.identical_scores_fallback = config.safe_get(
                "normalization.fallback.identical_scores", 0.5
            )
        else:
            self.zero_std_fallback = 1e-8
            self.identical_scores_fallback = 0.5

    def z_score_normalize(self, array, axis=0, return_stats=False):
        """
        Z-Score 정규화 수행 (x - mean) / std

        Args:
            array: 정규화할 배열 또는 리스트
            axis: 정규화를 수행할 축 (0: 열 방향, 1: 행 방향)
            return_stats: 평균 및 표준편차 등 통계값 반환 여부

        Returns:
            정규화된 배열 또는 (정규화된 배열, 통계 정보) 튜플
        """
        try:
            # NumPy 배열로 변환
            if not isinstance(array, np.ndarray):
                array = np.array(array, dtype=np.float64)

            # 차원 확인 및 처리
            if array.ndim == 1:
                array = array.reshape(-1, 1)

            # 평균 및 표준편차 계산
            mean = np.mean(array, axis=axis, keepdims=True)
            std = np.std(array, axis=axis, keepdims=True)

            # 0 표준편차 처리
            std[std == 0] = self.zero_std_fallback

            # 정규화 수행
            normalized = (array - mean) / std

            # 로깅
            self.logger.debug(
                f"Z-Score 정규화 완료: shape={array.shape}, mean={np.mean(mean):.4f}, std={np.mean(std):.4f}"
            )

            if return_stats:
                stats = {"mean": mean, "std": std}
                return normalized, stats
            else:
                return normalized

        except Exception as e:
            self.logger.error(f"Z-Score 정규화 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 반환
            if return_stats:
                stats = {"mean": np.zeros_like(array), "std": np.ones_like(array)}
                return array, stats
            else:
                return array

    def min_max_normalize(
        self, array, feature_range=(0, 1), axis=0, return_stats=False
    ):
        """
        Min-Max 정규화 수행 (x - min) / (max - min)

        Args:
            array: 정규화할 배열 또는 리스트
            feature_range: 변환할 범위 (min, max)
            axis: 정규화를 수행할 축 (0: 열 방향, 1: 행 방향)
            return_stats: 최소값 및 최대값 등 통계값 반환 여부

        Returns:
            정규화된 배열 또는 (정규화된 배열, 통계 정보) 튜플
        """
        try:
            # NumPy 배열로 변환
            if not isinstance(array, np.ndarray):
                array = np.array(array, dtype=np.float64)

            # 차원 확인 및 처리
            if array.ndim == 1:
                array = array.reshape(-1, 1)

            # 최소값 및 최대값 계산
            min_val = np.min(array, axis=axis, keepdims=True)
            max_val = np.max(array, axis=axis, keepdims=True)

            # 동일한 값 처리
            if np.array_equal(min_val, max_val):
                self.logger.warning(
                    "Min-Max 정규화: 모든 값이 동일합니다. 기본값으로 대체합니다."
                )
                normalized = np.full_like(array, self.identical_scores_fallback)
            else:
                # 정규화 범위
                min_range, max_range = feature_range

                # 정규화 수행
                normalized = (array - min_val) / (max_val - min_val) * (
                    max_range - min_range
                ) + min_range

            # 로깅
            self.logger.debug(
                f"Min-Max 정규화 완료: shape={array.shape}, min={np.mean(min_val):.4f}, max={np.mean(max_val):.4f}"
            )

            if return_stats:
                stats = {"min": min_val, "max": max_val}
                return normalized, stats
            else:
                return normalized

        except Exception as e:
            self.logger.error(f"Min-Max 정규화 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 반환
            if return_stats:
                stats = {"min": np.zeros_like(array), "max": np.ones_like(array)}
                return array, stats
            else:
                return array

    def z_score_filter(self, array, threshold=2.5):
        """
        Z-Score 기반 이상치 필터링 (지정된 임계값 범위 벗어나는 값 제외)

        Args:
            array: 필터링할 배열 또는 리스트
            threshold: Z-Score 임계값 (기본값: 2.5)

        Returns:
            필터링된 인덱스 목록 (정상 데이터 인덱스)
        """
        try:
            # NumPy 배열로 변환
            if not isinstance(array, np.ndarray):
                array = np.array(array, dtype=np.float64)

            # 1차원 확인
            if array.ndim > 1:
                array = array.flatten()

            # 평균 및 표준편차 계산
            mean = np.mean(array)
            std = np.std(array)

            # 표준편차가 0인 경우 처리
            if std == 0:
                self.logger.warning(
                    "Z-Score 필터링: 표준편차가 0입니다. 모든 인덱스를 반환합니다."
                )
                return list(range(len(array)))

            # Z-Score 계산
            z_scores = np.abs((array - mean) / std)

            # 임계값 이내의 인덱스 추출
            normal_indices = np.where(z_scores < threshold)[0].tolist()

            # 필터링 결과 로깅
            filtered_count = len(array) - len(normal_indices)
            if filtered_count > 0:
                self.logger.info(
                    f"Z-Score 필터링: 총 {len(array)}개 중 {filtered_count}개 제외됨 (임계값: {threshold})"
                )

            return normal_indices

        except Exception as e:
            self.logger.error(f"Z-Score 필터링 중 오류 발생: {str(e)}")
            # 오류 발생 시 모든 인덱스 반환
            return list(range(len(array) if hasattr(array, "__len__") else 0))

    def filter_outliers_by_zscore(
        self,
        candidates: List[Dict[str, Any]],
        score_key: str = "score",
        threshold: float = 2.5,
    ) -> List[Dict[str, Any]]:
        """
        Z-Score 기반 이상치 필터링을 딕셔너리 리스트에 적용

        Args:
            candidates: 후보 딕셔너리 리스트
            score_key: 점수를 포함하는 키 이름
            threshold: Z-Score 임계값 (기본값: 2.5)

        Returns:
            필터링된 후보 목록
        """
        try:
            if not candidates:
                return candidates

            # 점수 추출
            scores = []
            for i, candidate in enumerate(candidates):
                if score_key not in candidate:
                    self.logger.warning(f"후보 {i}에 점수 키 '{score_key}'가 없습니다.")
                    scores.append(0.0)
                else:
                    scores.append(float(candidate[score_key]))

            # 정상 인덱스 추출
            normal_indices = self.z_score_filter(np.array(scores), threshold)

            # 필터링된 후보 반환
            filtered_candidates = [candidates[i] for i in normal_indices]

            # 로깅
            self.logger.info(
                f"이상치 필터링: 총 {len(candidates)}개 중 {len(candidates) - len(filtered_candidates)}개 제외됨"
            )

            return filtered_candidates

        except Exception as e:
            self.logger.error(f"이상치 필터링 중 오류 발생: {str(e)}")
            return candidates

    def normalize_array(self, array, method="zscore", **kwargs):
        """
        설정에 따라 적절한 정규화 방식 적용

        Args:
            array: 정규화할 배열 또는 리스트
            method: 정규화 방식 ("zscore" 또는 "minmax")
            **kwargs: 각 정규화 함수에 전달할 추가 인자

        Returns:
            정규화된 배열
        """
        if method.lower() == "zscore":
            return self.z_score_normalize(array, **kwargs)
        elif method.lower() == "minmax":
            return self.min_max_normalize(array, **kwargs)
        else:
            self.logger.warning(f"알 수 없는 정규화 방식: {method}, 기본값 zscore 사용")
            return self.z_score_normalize(array, **kwargs)


# 모듈 레벨 함수 (단순 래퍼)
def z_score_normalize(array, **kwargs):
    """Z-Score 정규화 수행 - 모듈 레벨 함수"""
    normalizer = Normalizer()
    return normalizer.z_score_normalize(array, **kwargs)


def min_max_normalize(array, **kwargs):
    """Min-Max 정규화 수행 - 모듈 레벨 함수"""
    normalizer = Normalizer()
    return normalizer.min_max_normalize(array, **kwargs)


def z_score_filter(array, threshold=2.5):
    """Z-Score 기반 이상치 필터링 - 모듈 레벨 함수"""
    normalizer = Normalizer()
    return normalizer.z_score_filter(array, threshold)
