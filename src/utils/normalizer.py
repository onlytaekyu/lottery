"""
정규화 모듈 (GPU 우선순위 버전)

GPU 최우선 처리와 스마트 컴퓨테이션을 활용한 고성능 정규화 시스템
"""

import numpy as np
import torch
from typing import Dict, List, Any, Union, Optional, Tuple
from dataclasses import dataclass

from .unified_logging import get_logger
from .unified_config import ConfigProxy
from .performance_optimizer import smart_compute, get_smart_computation_engine

logger = get_logger(__name__)


@dataclass
class NormalizationConfig:
    """정규화 설정"""

    zero_std_fallback: float = 1e-8
    identical_scores_fallback: float = 0.5
    use_gpu_acceleration: bool = True
    batch_size_threshold: int = 1000  # GPU 사용 임계값


class GPUNormalizer:
    """GPU 우선순위 정규화 클래스"""

    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigProxy]] = None):
        self.logger = get_logger(__name__)
        self.computation_engine = get_smart_computation_engine()

        # 설정 초기화
        if isinstance(config, dict):
            self.config = NormalizationConfig(
                zero_std_fallback=config.get("normalization.fallback.zero_std", 1e-8),
                identical_scores_fallback=config.get(
                    "normalization.fallback.identical_scores", 0.5
                ),
                use_gpu_acceleration=config.get("normalization.use_gpu", True),
            )
        elif config is not None and hasattr(config, "safe_get"):
            self.config = NormalizationConfig(
                zero_std_fallback=config.safe_get(
                    "normalization.fallback.zero_std", 1e-8
                ),
                identical_scores_fallback=config.safe_get(
                    "normalization.fallback.identical_scores", 0.5
                ),
                use_gpu_acceleration=config.safe_get("normalization.use_gpu", True),
            )
        else:
            self.config = NormalizationConfig()

    def smart_normalize(
        self,
        data: Union[np.ndarray, list, torch.Tensor],
        method: str = "zscore",
        axis: int = 0,
        return_stats: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        스마트 정규화 - GPU > 멀티쓰레드 > CPU 순으로 처리

        Args:
            data: 정규화할 데이터
            method: 정규화 방법 ("zscore", "minmax", "robust")
            axis: 정규화 축
            return_stats: 통계 정보 반환 여부
            **kwargs: 추가 매개변수

        Returns:
            정규화된 데이터 또는 (데이터, 통계) 튜플
        """
        try:
            # 데이터 전처리
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
            elif isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)

            # 차원 처리
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            # 정규화 방법 선택
            if method.lower() == "zscore":
                return self._smart_zscore_normalize(data, axis, return_stats, **kwargs)
            elif method.lower() == "minmax":
                return self._smart_minmax_normalize(data, axis, return_stats, **kwargs)
            elif method.lower() == "robust":
                return self._smart_robust_normalize(data, axis, return_stats, **kwargs)
            else:
                raise ValueError(f"지원하지 않는 정규화 방법: {method}")

        except Exception as e:
            self.logger.error(f"스마트 정규화 실패: {e}")
            if return_stats:
                return data, {}
            return data

    def _smart_zscore_normalize(
        self, data: np.ndarray, axis: int, return_stats: bool, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """GPU 우선순위 Z-Score 정규화"""

        def zscore_computation(arr):
            if isinstance(arr, torch.Tensor):
                # GPU 텐서 처리
                mean = torch.mean(arr, dim=axis, keepdim=True)
                std = torch.std(arr, dim=axis, keepdim=True)

                # 0 표준편차 처리
                std = torch.where(
                    std == 0,
                    torch.tensor(self.config.zero_std_fallback, device=arr.device),
                    std,
                )

                normalized = (arr - mean) / std

                return normalized.cpu().numpy(), {
                    "mean": mean.cpu().numpy(),
                    "std": std.cpu().numpy(),
                }
            else:
                # NumPy 배열 처리
                mean = np.mean(arr, axis=axis, keepdims=True)
                std = np.std(arr, axis=axis, keepdims=True)

                # 0 표준편차 처리
                std[std == 0] = self.config.zero_std_fallback

                normalized = (arr - mean) / std

                return normalized, {"mean": mean, "std": std}

        try:
            # 스마트 컴퓨테이션 실행
            result = smart_compute(
                zscore_computation, data, operation_type="normalization"
            )

            if isinstance(result, tuple) and len(result) == 2:
                normalized, stats = result
                self.logger.debug(f"Z-Score 정규화 완료: shape={normalized.shape}")

                if return_stats:
                    return normalized, stats
                return normalized
            else:
                # 단일 결과인 경우
                if return_stats:
                    return result, {}
                return result

        except Exception as e:
            self.logger.error(f"Z-Score 정규화 실패: {e}")
            # 폴백 처리
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            std[std == 0] = self.config.zero_std_fallback
            normalized = (data - mean) / std

            if return_stats:
                return normalized, {"mean": mean, "std": std}
            return normalized

    def _smart_minmax_normalize(
        self,
        data: np.ndarray,
        axis: int,
        return_stats: bool,
        feature_range: Tuple[float, float] = (0, 1),
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """GPU 우선순위 Min-Max 정규화"""

        def minmax_computation(arr):
            if isinstance(arr, torch.Tensor):
                # GPU 텐서 처리
                min_val = torch.min(arr, dim=axis, keepdim=True)[0]
                max_val = torch.max(arr, dim=axis, keepdim=True)[0]

                # 동일한 값 처리
                range_val = max_val - min_val
                range_val = torch.where(
                    range_val == 0, torch.ones_like(range_val), range_val
                )

                min_range, max_range = feature_range
                normalized = (arr - min_val) / range_val * (
                    max_range - min_range
                ) + min_range

                return normalized.cpu().numpy(), {
                    "min": min_val.cpu().numpy(),
                    "max": max_val.cpu().numpy(),
                }
            else:
                # NumPy 배열 처리
                min_val = np.min(arr, axis=axis, keepdims=True)
                max_val = np.max(arr, axis=axis, keepdims=True)

                # 동일한 값 처리
                if np.array_equal(min_val, max_val):
                    normalized = np.full_like(
                        arr, self.config.identical_scores_fallback
                    )
                else:
                    min_range, max_range = feature_range
                    normalized = (arr - min_val) / (max_val - min_val) * (
                        max_range - min_range
                    ) + min_range

                return normalized, {"min": min_val, "max": max_val}

        try:
            # 스마트 컴퓨테이션 실행
            result = smart_compute(
                minmax_computation, data, operation_type="normalization"
            )

            if isinstance(result, tuple) and len(result) == 2:
                normalized, stats = result
                self.logger.debug(f"Min-Max 정규화 완료: shape={normalized.shape}")

                if return_stats:
                    return normalized, stats
                return normalized
            else:
                if return_stats:
                    return result, {}
                return result

        except Exception as e:
            self.logger.error(f"Min-Max 정규화 실패: {e}")
            # 폴백 처리
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)

            if np.array_equal(min_val, max_val):
                normalized = np.full_like(data, self.config.identical_scores_fallback)
            else:
                min_range, max_range = feature_range
                normalized = (data - min_val) / (max_val - min_val) * (
                    max_range - min_range
                ) + min_range

            if return_stats:
                return normalized, {"min": min_val, "max": max_val}
            return normalized

    def _smart_robust_normalize(
        self, data: np.ndarray, axis: int, return_stats: bool, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """GPU 우선순위 Robust 정규화 (중위수 및 MAD 사용)"""

        def robust_computation(arr):
            if isinstance(arr, torch.Tensor):
                # GPU 텐서 처리
                median = torch.median(arr, dim=axis, keepdim=True)[0]
                mad = torch.median(torch.abs(arr - median), dim=axis, keepdim=True)[0]

                # MAD가 0인 경우 처리
                mad = torch.where(
                    mad == 0,
                    torch.tensor(self.config.zero_std_fallback, device=arr.device),
                    mad,
                )

                normalized = (arr - median) / mad

                return normalized.cpu().numpy(), {
                    "median": median.cpu().numpy(),
                    "mad": mad.cpu().numpy(),
                }
            else:
                # NumPy 배열 처리
                median = np.median(arr, axis=axis, keepdims=True)
                mad = np.median(np.abs(arr - median), axis=axis, keepdims=True)

                # MAD가 0인 경우 처리
                mad[mad == 0] = self.config.zero_std_fallback

                normalized = (arr - median) / mad

                return normalized, {"median": median, "mad": mad}

        try:
            # 스마트 컴퓨테이션 실행
            result = smart_compute(
                robust_computation, data, operation_type="normalization"
            )

            if isinstance(result, tuple) and len(result) == 2:
                normalized, stats = result
                self.logger.debug(f"Robust 정규화 완료: shape={normalized.shape}")

                if return_stats:
                    return normalized, stats
                return normalized
            else:
                if return_stats:
                    return result, {}
                return result

        except Exception as e:
            self.logger.error(f"Robust 정규화 실패: {e}")
            # 폴백 처리
            median = np.median(data, axis=axis, keepdims=True)
            mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
            mad[mad == 0] = self.config.zero_std_fallback
            normalized = (data - median) / mad

            if return_stats:
                return normalized, {"median": median, "mad": mad}
            return normalized

    def smart_outlier_filter(
        self,
        data: Union[np.ndarray, list],
        threshold: float = 2.5,
        method: str = "zscore",
    ) -> List[int]:
        """
        스마트 이상치 필터링 - GPU 우선순위 처리

        Args:
            data: 필터링할 데이터
            threshold: 임계값
            method: 필터링 방법 ("zscore", "iqr", "isolation")

        Returns:
            정상 데이터 인덱스 목록
        """
        try:
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)

            if data.ndim > 1:
                data = data.flatten()

            def outlier_computation(arr):
                if isinstance(arr, torch.Tensor):
                    if method == "zscore":
                        mean = torch.mean(arr)
                        std = torch.std(arr)
                        if std == 0:
                            return torch.arange(len(arr), device=arr.device)
                        z_scores = torch.abs((arr - mean) / std)
                        return torch.where(z_scores < threshold)[0]
                    elif method == "iqr":
                        q1 = torch.quantile(arr, 0.25)
                        q3 = torch.quantile(arr, 0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        return torch.where((arr >= lower_bound) & (arr <= upper_bound))[
                            0
                        ]
                else:
                    if method == "zscore":
                        mean = np.mean(arr)
                        std = np.std(arr)
                        if std == 0:
                            return np.arange(len(arr))
                        z_scores = np.abs((arr - mean) / std)
                        return np.where(z_scores < threshold)[0]
                    elif method == "iqr":
                        q1 = np.percentile(arr, 25)
                        q3 = np.percentile(arr, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        return np.where((arr >= lower_bound) & (arr <= upper_bound))[0]

            # 스마트 컴퓨테이션 실행
            result = smart_compute(
                outlier_computation, data, operation_type="filtering"
            )

            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy()

            normal_indices = (
                result.tolist() if hasattr(result, "tolist") else list(result)
            )

            filtered_count = len(data) - len(normal_indices)
            if filtered_count > 0:
                self.logger.info(
                    f"이상치 필터링 완료: {len(data)}개 중 {filtered_count}개 제외"
                )

            return normal_indices

        except Exception as e:
            self.logger.error(f"이상치 필터링 실패: {e}")
            return list(range(len(data)))


# 전역 정규화 인스턴스
_gpu_normalizer = None


def get_gpu_normalizer(
    config: Optional[Union[Dict[str, Any], ConfigProxy]] = None,
) -> GPUNormalizer:
    """GPU 정규화 인스턴스 반환"""
    global _gpu_normalizer
    if _gpu_normalizer is None:
        _gpu_normalizer = GPUNormalizer(config)
    return _gpu_normalizer


# 통합 정규화 함수들 (하위 호환성 유지)
def smart_normalize(
    data: Union[np.ndarray, list, torch.Tensor], method: str = "zscore", **kwargs
) -> np.ndarray:
    """통합 정규화 함수 - GPU 우선순위 처리"""
    normalizer = get_gpu_normalizer()
    return normalizer.smart_normalize(data, method, **kwargs)


def z_score_normalize(array, **kwargs):
    """Z-Score 정규화 (하위 호환성)"""
    return smart_normalize(array, method="zscore", **kwargs)


def min_max_normalize(array, **kwargs):
    """Min-Max 정규화 (하위 호환성)"""
    return smart_normalize(array, method="minmax", **kwargs)


def z_score_filter(array, threshold=2.5):
    """Z-Score 필터링 (하위 호환성)"""
    normalizer = get_gpu_normalizer()
    return normalizer.smart_outlier_filter(array, threshold, method="zscore")


# 기존 Normalizer 클래스 (하위 호환성 유지)
class Normalizer(GPUNormalizer):
    """기존 Normalizer 클래스 (하위 호환성)"""

    def normalize_array(self, array, method="zscore", **kwargs):
        """배열 정규화 (하위 호환성)"""
        return self.smart_normalize(array, method, **kwargs)

    def z_score_normalize(self, array, axis=0, return_stats=False):
        """Z-Score 정규화 (하위 호환성)"""
        return self.smart_normalize(
            array, method="zscore", axis=axis, return_stats=return_stats
        )

    def min_max_normalize(
        self, array, feature_range=(0, 1), axis=0, return_stats=False
    ):
        """Min-Max 정규화 (하위 호환성)"""
        return self.smart_normalize(
            array,
            method="minmax",
            axis=axis,
            return_stats=return_stats,
            feature_range=feature_range,
        )

    def z_score_filter(self, array, threshold=2.5):
        """Z-Score 필터링 (하위 호환성)"""
        return self.smart_outlier_filter(array, threshold, method="zscore")

    def filter_outliers_by_zscore(
        self,
        candidates: List[Dict[str, Any]],
        score_key: str = "score",
        threshold: float = 2.5,
    ) -> List[Dict[str, Any]]:
        """딕셔너리 리스트 이상치 필터링 (하위 호환성)"""
        try:
            if not candidates:
                return candidates

            # 점수 추출
            scores = [candidate.get(score_key, 0) for candidate in candidates]

            # 이상치 인덱스 필터링
            normal_indices = self.smart_outlier_filter(
                scores, threshold, method="zscore"
            )

            # 필터링된 후보 반환
            return [candidates[i] for i in normal_indices]

        except Exception as e:
            self.logger.error(f"후보 이상치 필터링 실패: {e}")
            return candidates
