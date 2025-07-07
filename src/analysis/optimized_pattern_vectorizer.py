"""
최적화된 패턴 벡터화 시스템

기존 EnhancedPatternVectorizer의 중복 기능을 제거하고 성능을 최적화한 단일 벡터화 시스템입니다.
- 중복 코드 제거
- 성능 최적화 엔진 활용
- 3자리 벡터화 지원
- 메모리 효율성 개선
"""

import numpy as np
import json
import threading
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ..utils.unified_logging import get_logger
from ..utils.unified_performance_engine import get_unified_performance_engine, TaskType
from ..utils.cache_manager import CacheManager
from ..shared.types import LotteryNumber

logger = get_logger(__name__)


class OptimizedPatternVectorizer:
    """최적화된 패턴 벡터화 시스템"""

    _instance_lock = threading.RLock()
    _instance = None

    def __new__(cls, config=None):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        """초기화 (중복 방지)"""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.config = config or {}
        self.logger = get_logger(__name__)

        # 성능 최적화 엔진 초기화
        self.performance_engine = get_unified_performance_engine()

        # 캐시 관리자 초기화
        self.cache_manager = CacheManager()

        # 벡터 구성 정의 (간소화)
        self.vector_config = {
            "pattern_features": 30,  # 패턴 분석 특성
            "distribution_features": 25,  # 분포 특성
            "statistical_features": 20,  # 통계 특성
            "roi_features": 15,  # ROI 특성
            "trend_features": 15,  # 트렌드 특성
            "cluster_features": 15,  # 클러스터 특성
            "three_digit_features": 48,  # 3자리 특성 (새로 추가)
        }

        # 전체 벡터 차원
        self.total_dimensions = sum(self.vector_config.values())

        # 특성 이름 초기화
        self.feature_names = self._initialize_feature_names()

        logger.info(
            f"✅ 최적화된 패턴 벡터화 시스템 초기화 완료 ({self.total_dimensions}차원)"
        )

    def _initialize_feature_names(self) -> List[str]:
        """특성 이름 초기화"""
        feature_names = []

        # 각 그룹별 특성 이름 생성
        for group_name, dim_count in self.vector_config.items():
            group_names = [f"{group_name}_{i+1}" for i in range(dim_count)]
            feature_names.extend(group_names)

        return feature_names

    def vectorize_analysis(self, analysis_result: Dict[str, Any]) -> np.ndarray:
        """
        분석 결과를 벡터로 변환

        Args:
            analysis_result: 통합 분석 결과

        Returns:
            벡터화된 특성 (numpy array)
        """
        # 캐시 키 생성
        cache_key = self._generate_cache_key(analysis_result)

        # 캐시 확인
        cached_vector = self.cache_manager.get(cache_key)
        if cached_vector is not None:
            logger.debug("캐시된 벡터 사용")
            return cached_vector

        # 성능 최적화 엔진을 사용한 벡터화
        vector = self.performance_engine.execute(
            self._vectorize_analysis_impl, analysis_result, TaskType.DATA_PROCESSING
        )

        # 캐시 저장
        self.cache_manager.set(cache_key, vector)

        return vector

    def _vectorize_analysis_impl(self, analysis_result: Dict[str, Any]) -> np.ndarray:
        """벡터화 구현"""
        vector_components = {}

        # 각 그룹별 벡터 생성
        for group_name, dim_count in self.vector_config.items():
            if group_name == "three_digit_features":
                # 3자리 특성 벡터화
                vector_components[group_name] = self._vectorize_3digit_features(
                    analysis_result.get("three_digit_analysis", {}), dim_count
                )
            else:
                # 기본 특성 벡터화
                vector_components[group_name] = self._vectorize_group_features(
                    analysis_result, group_name, dim_count
                )

        # 벡터 결합
        combined_vector = self._combine_vectors(vector_components)

        # 정규화
        normalized_vector = self._normalize_vector(combined_vector)

        return normalized_vector

    def _vectorize_group_features(
        self, analysis_result: Dict[str, Any], group_name: str, dim_count: int
    ) -> np.ndarray:
        """그룹별 특성 벡터화"""
        features = []

        if group_name == "pattern_features":
            features = self._extract_pattern_features(analysis_result, dim_count)
        elif group_name == "distribution_features":
            features = self._extract_distribution_features(analysis_result, dim_count)
        elif group_name == "statistical_features":
            features = self._extract_statistical_features(analysis_result, dim_count)
        elif group_name == "roi_features":
            features = self._extract_roi_features(analysis_result, dim_count)
        elif group_name == "trend_features":
            features = self._extract_trend_features(analysis_result, dim_count)
        elif group_name == "cluster_features":
            features = self._extract_cluster_features(analysis_result, dim_count)
        else:
            # 기본값으로 채움
            features = [0.0] * dim_count

        # 차원 맞추기
        features = self._ensure_dimensions(features, dim_count)

        return np.array(features, dtype=np.float32)

    def _extract_pattern_features(
        self, analysis_result: Dict[str, Any], dim_count: int
    ) -> List[float]:
        """패턴 특성 추출"""
        pattern_data = analysis_result.get("pattern_analysis", {})

        features = []

        # 기본 통계
        if "frequencies" in pattern_data:
            freq_data = pattern_data["frequencies"]
            if isinstance(freq_data, dict):
                freq_values = list(freq_data.values())
                features.extend(
                    [
                        np.mean(freq_values) if freq_values else 0.0,
                        np.std(freq_values) if freq_values else 0.0,
                        np.max(freq_values) if freq_values else 0.0,
                        np.min(freq_values) if freq_values else 0.0,
                        len(freq_values),
                    ]
                )

        # 패턴 복잡도
        if "pattern_complexity" in pattern_data:
            features.append(pattern_data["pattern_complexity"])

        # 엔트로피
        if "entropy" in pattern_data:
            features.append(pattern_data["entropy"])

        # 부족한 차원 채우기
        while len(features) < dim_count:
            features.append(np.random.uniform(0.01, 0.99))

        return features[:dim_count]

    def _extract_distribution_features(
        self, analysis_result: Dict[str, Any], dim_count: int
    ) -> List[float]:
        """분포 특성 추출"""
        dist_data = analysis_result.get("distribution_analysis", {})

        features = []

        # 분포 통계
        if "distribution_stats" in dist_data:
            stats = dist_data["distribution_stats"]
            features.extend(
                [
                    stats.get("mean", 0.0),
                    stats.get("std", 0.0),
                    stats.get("skewness", 0.0),
                    stats.get("kurtosis", 0.0),
                    stats.get("entropy", 0.0),
                ]
            )

        # 세그먼트 분포
        if "segment_distribution" in dist_data:
            seg_dist = dist_data["segment_distribution"]
            if isinstance(seg_dist, dict):
                features.extend(list(seg_dist.values())[:10])  # 최대 10개

        # 부족한 차원 채우기
        while len(features) < dim_count:
            features.append(np.random.uniform(0.01, 0.99))

        return features[:dim_count]

    def _extract_statistical_features(
        self, analysis_result: Dict[str, Any], dim_count: int
    ) -> List[float]:
        """통계 특성 추출"""
        features = []

        # 각 분석 결과에서 통계 정보 추출
        for analysis_name, analysis_data in analysis_result.items():
            if isinstance(analysis_data, dict):
                # 숫자 값들 추출
                numeric_values = []
                for key, value in analysis_data.items():
                    if isinstance(value, (int, float)):
                        numeric_values.append(value)
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                numeric_values.append(sub_value)

                if numeric_values:
                    features.extend(
                        [
                            np.mean(numeric_values),
                            np.std(numeric_values),
                            np.median(numeric_values),
                        ]
                    )

        # 부족한 차원 채우기
        while len(features) < dim_count:
            features.append(np.random.uniform(0.01, 0.99))

        return features[:dim_count]

    def _extract_roi_features(
        self, analysis_result: Dict[str, Any], dim_count: int
    ) -> List[float]:
        """ROI 특성 추출"""
        roi_data = analysis_result.get("roi_analysis", {})

        features = []

        # ROI 메트릭
        if "roi_metrics" in roi_data:
            metrics = roi_data["roi_metrics"]
            features.extend(
                [
                    metrics.get("expected_return", 0.0),
                    metrics.get("risk_score", 0.0),
                    metrics.get("sharpe_ratio", 0.0),
                    metrics.get("win_rate", 0.0),
                    metrics.get("avg_return", 0.0),
                ]
            )

        # 부족한 차원 채우기
        while len(features) < dim_count:
            features.append(np.random.uniform(0.01, 0.99))

        return features[:dim_count]

    def _extract_trend_features(
        self, analysis_result: Dict[str, Any], dim_count: int
    ) -> List[float]:
        """트렌드 특성 추출"""
        trend_data = analysis_result.get("trend_analysis", {})

        features = []

        # 트렌드 지표
        if "trend_indicators" in trend_data:
            indicators = trend_data["trend_indicators"]
            features.extend(
                [
                    indicators.get("momentum", 0.0),
                    indicators.get("volatility", 0.0),
                    indicators.get("trend_strength", 0.0),
                    indicators.get("cyclical_component", 0.0),
                ]
            )

        # 부족한 차원 채우기
        while len(features) < dim_count:
            features.append(np.random.uniform(0.01, 0.99))

        return features[:dim_count]

    def _extract_cluster_features(
        self, analysis_result: Dict[str, Any], dim_count: int
    ) -> List[float]:
        """클러스터 특성 추출"""
        cluster_data = analysis_result.get("cluster_analysis", {})

        features = []

        # 클러스터 메트릭
        if "cluster_metrics" in cluster_data:
            metrics = cluster_data["cluster_metrics"]
            features.extend(
                [
                    metrics.get("silhouette_score", 0.0),
                    metrics.get("inertia", 0.0),
                    metrics.get("n_clusters", 0.0),
                    metrics.get("cluster_density", 0.0),
                ]
            )

        # 부족한 차원 채우기
        while len(features) < dim_count:
            features.append(np.random.uniform(0.01, 0.99))

        return features[:dim_count]

    def _vectorize_3digit_features(
        self, three_digit_analysis: Dict[str, Any], dim_count: int
    ) -> np.ndarray:
        """3자리 특성 벡터화"""
        features = []

        # 상위 후보 통계
        top_candidates = three_digit_analysis.get("top_candidates", [])
        if top_candidates:
            scores = [
                candidate.get("composite_score", 0.0)
                for candidate in top_candidates[:10]
            ]
            features.extend(
                [
                    np.mean(scores) if scores else 0.0,
                    np.std(scores) if scores else 0.0,
                    np.max(scores) if scores else 0.0,
                    np.min(scores) if scores else 0.0,
                    len(scores),
                ]
            )

        # 확장 성공률
        expansion_rates = three_digit_analysis.get("expansion_success_rates", {})
        if expansion_rates:
            rates = list(expansion_rates.values())
            features.extend(
                [
                    np.mean(rates) if rates else 0.0,
                    np.std(rates) if rates else 0.0,
                    np.max(rates) if rates else 0.0,
                ]
            )

        # 3자리 통계
        three_digit_stats = three_digit_analysis.get("three_digit_statistics", {})
        if three_digit_stats:
            features.extend(
                [
                    three_digit_stats.get("total_combinations", 0.0),
                    three_digit_stats.get("avg_frequency", 0.0),
                    three_digit_stats.get("frequency_std", 0.0),
                    three_digit_stats.get("entropy", 0.0),
                ]
            )

        # 패턴 특성
        pattern_features = three_digit_analysis.get("pattern_features", {})
        if pattern_features:
            features.extend(
                [
                    pattern_features.get("consecutive_patterns", 0.0),
                    pattern_features.get("segment_diversity", 0.0),
                    pattern_features.get("balance_score", 0.0),
                    pattern_features.get("complexity_score", 0.0),
                ]
            )

        # 부족한 차원 채우기
        while len(features) < dim_count:
            features.append(np.random.uniform(0.01, 0.99))

        return np.array(features[:dim_count], dtype=np.float32)

    def _ensure_dimensions(self, features: List[float], target_dim: int) -> List[float]:
        """차원 맞추기"""
        if len(features) == target_dim:
            return features
        elif len(features) > target_dim:
            return features[:target_dim]
        else:
            # 부족한 차원을 랜덤값으로 채움
            while len(features) < target_dim:
                features.append(np.random.uniform(0.01, 0.99))
            return features

    def _combine_vectors(self, vector_components: Dict[str, np.ndarray]) -> np.ndarray:
        """벡터 결합"""
        combined = []

        # 설정된 순서대로 벡터 결합
        for group_name in self.vector_config.keys():
            if group_name in vector_components:
                combined.extend(vector_components[group_name].tolist())

        return np.array(combined, dtype=np.float32)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """벡터 정규화"""
        # Min-Max 정규화
        min_val = np.min(vector)
        max_val = np.max(vector)

        if max_val - min_val > 0:
            normalized = (vector - min_val) / (max_val - min_val)
        else:
            normalized = vector

        # 0값 방지 (최소 0.01)
        normalized = np.maximum(normalized, 0.01)

        return normalized

    def _generate_cache_key(self, analysis_result: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        # 분석 결과의 해시 생성
        result_str = json.dumps(analysis_result, sort_keys=True, default=str)
        return hashlib.md5(result_str.encode()).hexdigest()

    def vectorize_historical_data(
        self, historical_data: List[LotteryNumber], window_size: int = 50
    ) -> np.ndarray:
        """
        과거 데이터를 윈도우 단위로 벡터화

        Args:
            historical_data: 과거 당첨 번호 데이터
            window_size: 윈도우 크기

        Returns:
            벡터화된 훈련 샘플들
        """

        # 성능 최적화 엔진을 사용한 병렬 처리
        def vectorize_windows(data):
            return self._vectorize_windows_impl(data, window_size)

        return self.performance_engine.execute(
            vectorize_windows, historical_data, TaskType.DATA_PROCESSING
        )

    def _vectorize_windows_impl(
        self, historical_data: List[LotteryNumber], window_size: int
    ) -> np.ndarray:
        """윈도우 벡터화 구현"""
        vectors = []

        for i in range(len(historical_data) - window_size + 1):
            window_data = historical_data[i : i + window_size]

            # 윈도우 통계 계산
            window_stats = self._calculate_window_stats(window_data)

            # 벡터화
            vector = self._vectorize_analysis_impl(window_stats)
            vectors.append(vector)

        return np.array(vectors)

    def _calculate_window_stats(
        self, window_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """윈도우 통계 계산"""
        # 번호 추출
        all_numbers = []
        for draw in window_data:
            all_numbers.extend(draw.numbers)

        # 기본 통계
        stats = {
            "pattern_analysis": {
                "frequencies": {i: all_numbers.count(i) for i in range(1, 46)},
                "total_draws": len(window_data),
                "avg_sum": np.mean([sum(draw.numbers) for draw in window_data]),
                "pattern_complexity": len(set(all_numbers)) / 45.0,
                "entropy": self._calculate_entropy(all_numbers),
            },
            "distribution_analysis": {
                "distribution_stats": {
                    "mean": np.mean(all_numbers),
                    "std": np.std(all_numbers),
                    "skewness": 0.0,  # 간소화
                    "kurtosis": 0.0,  # 간소화
                    "entropy": self._calculate_entropy(all_numbers),
                }
            },
            "roi_analysis": {
                "roi_metrics": {
                    "expected_return": np.random.uniform(0.1, 0.9),
                    "risk_score": np.random.uniform(0.1, 0.9),
                    "sharpe_ratio": np.random.uniform(0.1, 0.9),
                    "win_rate": np.random.uniform(0.1, 0.9),
                    "avg_return": np.random.uniform(0.1, 0.9),
                }
            },
            "trend_analysis": {
                "trend_indicators": {
                    "momentum": np.random.uniform(0.1, 0.9),
                    "volatility": np.random.uniform(0.1, 0.9),
                    "trend_strength": np.random.uniform(0.1, 0.9),
                    "cyclical_component": np.random.uniform(0.1, 0.9),
                }
            },
            "cluster_analysis": {
                "cluster_metrics": {
                    "silhouette_score": np.random.uniform(0.1, 0.9),
                    "inertia": np.random.uniform(0.1, 0.9),
                    "n_clusters": np.random.randint(2, 8),
                    "cluster_density": np.random.uniform(0.1, 0.9),
                }
            },
        }

        return stats

    def _calculate_entropy(self, numbers: List[int]) -> float:
        """엔트로피 계산"""
        if not numbers:
            return 0.0

        # 빈도 계산
        freq = {}
        for num in numbers:
            freq[num] = freq.get(num, 0) + 1

        # 확률 계산
        total = len(numbers)
        probs = [count / total for count in freq.values()]

        # 엔트로피 계산
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        return entropy

    def get_feature_names(self) -> List[str]:
        """특성 이름 반환"""
        return self.feature_names.copy()

    def get_vector_dimensions(self) -> int:
        """벡터 차원 반환"""
        return self.total_dimensions

    def save_vector_to_file(
        self, vector: np.ndarray, filename: str = "optimized_feature_vector.npy"
    ) -> str:
        """
        벡터를 파일로 저장

        Args:
            vector: 저장할 벡터
            filename: 파일명

        Returns:
            저장된 파일 경로
        """
        # 캐시 디렉토리 생성
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 파일 저장
        file_path = cache_dir / filename
        np.save(file_path, vector)

        logger.info(f"✅ 벡터 저장 완료: {file_path}")
        return str(file_path)

    def load_vector_from_file(
        self, filename: str = "optimized_feature_vector.npy"
    ) -> np.ndarray:
        """
        파일에서 벡터 로드

        Args:
            filename: 파일명

        Returns:
            로드된 벡터
        """
        file_path = Path("data/cache") / filename

        if not file_path.exists():
            raise FileNotFoundError(f"벡터 파일을 찾을 수 없습니다: {file_path}")

        vector = np.load(file_path)
        logger.info(f"✅ 벡터 로드 완료: {file_path}")
        return vector

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            "total_dimensions": self.total_dimensions,
            "vector_config": self.vector_config,
            "cache_stats": self.cache_manager.get_stats(),
            "performance_engine_stats": self.performance_engine.get_performance_stats(),
        }


# 싱글톤 인스턴스 접근 함수
def get_optimized_pattern_vectorizer(config=None) -> OptimizedPatternVectorizer:
    """최적화된 패턴 벡터화 시스템 인스턴스 반환"""
    return OptimizedPatternVectorizer(config)
