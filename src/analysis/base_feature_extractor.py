"""
베이스 특성 추출기 모듈

FeatureExtractor와 SequenceFeatureBuilder의 공통 기능을 제공하는 베이스 클래스입니다.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# logging 제거 - unified_logging 사용
from datetime import datetime
from abc import ABC, abstractmethod

from ..utils.unified_logging import get_logger
from ..shared.types import LotteryNumber
from ..utils.unified_config import ConfigProxy
from ..utils.unified_performance import performance_monitor
from ..utils.memory_manager import get_memory_manager
from ..utils.cache_manager import CacheManager


class BaseFeatureExtractor(ABC):
    """
    특성 추출 베이스 클래스

    FeatureExtractor와 SequenceFeatureBuilder의 공통 기능을 제공합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        초기화

        Args:
            config: 특성 추출 설정
        """
        self.config = ConfigProxy(config or {})
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()
        self.cache_manager = CacheManager()

        # 캐시 디렉토리 설정
        try:
            self.cache_dir = Path(self.config["paths"]["cache_dir"])
        except (KeyError, TypeError):
            self.logger.warning(
                "설정에서 'paths.cache_dir'를 찾을 수 없습니다. 기본값('data/cache')을 사용합니다."
            )
            self.cache_dir = Path("data/cache")

        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 특성 추출 설정
        self.max_features = self.config.get("feature_extraction", {}).get(
            "max_features", 100
        )
        self.feature_selection_method = self.config.get("feature_extraction", {}).get(
            "selection_method", "mutual_info"
        )
        self.scaling_method = self.config.get("feature_extraction", {}).get(
            "scaling_method", "standard"
        )

    def safe_index_access(self, data: Any, index: int = 0) -> Union[float, int]:
        """
        타입 안전한 인덱스 접근 함수

        Args:
            data: 접근할 데이터
            index: 접근할 인덱스 (기본값: 0)

        Returns:
            인덱스에 해당하는 값 또는 기본값
        """
        if isinstance(data, (list, tuple, np.ndarray)) and len(data) > index:
            return data[index]
        elif isinstance(data, (int, float, bool, complex)):
            return data
        else:
            try:
                return float(np.asarray(data).flat[0])
            except:
                return 0.0

    def extract_basic_number_features(self, numbers: List[int]) -> np.ndarray:
        """
        기본 번호 특성 추출

        Args:
            numbers: 당첨 번호 리스트

        Returns:
            np.ndarray: 기본 특성 벡터
        """
        try:
            # 기본 통계 특성
            features = []

            # 1. 합계
            total_sum = sum(numbers)
            features.append(total_sum)

            # 2. 평균
            mean_val = np.mean(numbers)
            features.append(mean_val)

            # 3. 분산
            variance = np.var(numbers)
            features.append(variance)

            # 4. 홀짝 비율
            odd_count = sum(1 for num in numbers if num % 2 == 1)
            odd_ratio = odd_count / len(numbers)
            features.append(odd_ratio)

            # 5. 범위
            range_val = max(numbers) - min(numbers)
            features.append(range_val)

            # 6. 연속 번호 개수
            sorted_numbers = sorted(numbers)
            consecutive_count = 0
            for i in range(1, len(sorted_numbers)):
                if sorted_numbers[i] - sorted_numbers[i - 1] == 1:
                    consecutive_count += 1
            features.append(consecutive_count)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            self.logger.warning(f"기본 번호 특성 추출 중 오류: {e}")
            return np.zeros(6, dtype=np.float32)

    def extract_segment_features(self, numbers: List[int]) -> np.ndarray:
        """
        세그먼트 기반 특성 추출

        Args:
            numbers: 당첨 번호 리스트

        Returns:
            np.ndarray: 세그먼트 특성 벡터
        """
        try:
            # 5개 세그먼트로 분할 (1-9, 10-18, 19-27, 28-36, 37-45)
            segment_counts = [0] * 5

            for num in numbers:
                if 1 <= num <= 9:
                    segment_counts[0] += 1
                elif 10 <= num <= 18:
                    segment_counts[1] += 1
                elif 19 <= num <= 27:
                    segment_counts[2] += 1
                elif 28 <= num <= 36:
                    segment_counts[3] += 1
                elif 37 <= num <= 45:
                    segment_counts[4] += 1

            # 정규화 (총 6개 번호 중 각 세그먼트 비율)
            segment_ratios = [count / len(numbers) for count in segment_counts]

            return np.array(segment_ratios, dtype=np.float32)

        except Exception as e:
            self.logger.warning(f"세그먼트 특성 추출 중 오류: {e}")
            return np.zeros(5, dtype=np.float32)

    def extract_gap_features(self, numbers: List[int]) -> np.ndarray:
        """
        간격 기반 특성 추출

        Args:
            numbers: 당첨 번호 리스트

        Returns:
            np.ndarray: 간격 특성 벡터
        """
        try:
            sorted_numbers = sorted(numbers)
            gaps = []

            # 연속된 번호 간의 간격 계산
            for i in range(1, len(sorted_numbers)):
                gap = sorted_numbers[i] - sorted_numbers[i - 1]
                gaps.append(gap)

            if gaps:
                features = [
                    np.mean(gaps),  # 평균 간격
                    np.std(gaps),  # 간격 표준편차
                    min(gaps),  # 최소 간격
                    max(gaps),  # 최대 간격
                    len([g for g in gaps if g == 1]),  # 연속 간격 개수
                ]
            else:
                features = [0.0, 0.0, 0.0, 0.0, 0.0]

            return np.array(features, dtype=np.float32)

        except Exception as e:
            self.logger.warning(f"간격 특성 추출 중 오류: {e}")
            return np.zeros(5, dtype=np.float32)

    def normalize_features(
        self, features: np.ndarray, method: str = "standard"
    ) -> np.ndarray:
        """
        특성 정규화

        Args:
            features: 정규화할 특성 배열
            method: 정규화 방법 ("standard", "minmax", "robust")

        Returns:
            np.ndarray: 정규화된 특성 배열
        """
        try:
            if method == "standard":
                # 표준화 (평균 0, 표준편차 1)
                mean_val = np.mean(features, axis=0)
                std_val = np.std(features, axis=0)
                std_val = np.where(std_val == 0, 1, std_val)  # 0으로 나누기 방지
                return (features - mean_val) / std_val

            elif method == "minmax":
                # 최소-최대 정규화 (0-1 범위)
                min_val = np.min(features, axis=0)
                max_val = np.max(features, axis=0)
                range_val = max_val - min_val
                range_val = np.where(range_val == 0, 1, range_val)  # 0으로 나누기 방지
                return (features - min_val) / range_val

            elif method == "robust":
                # 로버스트 정규화 (중간값과 IQR 사용)
                median_val = np.median(features, axis=0)
                q75 = np.percentile(features, 75, axis=0)
                q25 = np.percentile(features, 25, axis=0)
                iqr = q75 - q25
                iqr = np.where(iqr == 0, 1, iqr)  # 0으로 나누기 방지
                return (features - median_val) / iqr

            else:
                self.logger.warning(f"알 수 없는 정규화 방법: {method}")
                return features

        except Exception as e:
            self.logger.warning(f"특성 정규화 중 오류: {e}")
            return features

    def create_cache_key(self, data: Any, prefix: str = "feature") -> str:
        """
        캐시 키 생성

        Args:
            data: 캐시 키 생성에 사용할 데이터
            prefix: 키 접두사

        Returns:
            str: 캐시 키
        """
        try:
            import hashlib

            # 데이터를 문자열로 변환
            if isinstance(data, dict):
                data_str = str(sorted(data.items()))
            elif isinstance(data, list):
                data_str = str(data)
            else:
                data_str = str(data)

            # 해시 생성
            hash_obj = hashlib.md5(data_str.encode())
            return f"{prefix}_{hash_obj.hexdigest()[:16]}"

        except Exception as e:
            self.logger.warning(f"캐시 키 생성 중 오류: {e}")
            return f"{prefix}_{int(datetime.now().timestamp())}"

    def save_features_to_cache(self, features: np.ndarray, cache_key: str) -> bool:
        """
        특성을 캐시에 저장

        Args:
            features: 저장할 특성 배열
            cache_key: 캐시 키

        Returns:
            bool: 저장 성공 여부
        """
        try:
            cache_path = self.cache_dir / f"{cache_key}.npy"
            np.save(cache_path, features)
            self.logger.debug(f"특성을 캐시에 저장: {cache_path}")
            return True

        except Exception as e:
            self.logger.warning(f"특성 캐시 저장 중 오류: {e}")
            return False

    def load_features_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """
        캐시에서 특성 로드

        Args:
            cache_key: 캐시 키

        Returns:
            Optional[np.ndarray]: 로드된 특성 배열 (없으면 None)
        """
        try:
            cache_path = self.cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                features = np.load(cache_path)
                self.logger.debug(f"캐시에서 특성 로드: {cache_path}")
                return features
            return None

        except Exception as e:
            self.logger.warning(f"특성 캐시 로드 중 오류: {e}")
            return None

    @abstractmethod
    def extract_features(self, data: Any, *args, **kwargs) -> Any:
        """
        특성 추출 메서드 (하위 클래스에서 구현)

        Args:
            data: 특성 추출할 데이터
            *args, **kwargs: 추가 매개변수

        Returns:
            Any: 추출된 특성
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")
