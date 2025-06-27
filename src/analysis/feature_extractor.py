"""
특성 추출 엔진

다양한 패턴 분석 결과로부터 머신러닝/딥러닝 모델에 사용할 특성들을 추출하는 모듈입니다.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, cast
from dataclasses import dataclass
from collections import Counter, defaultdict
import math
from datetime import datetime
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.shared.types import LotteryNumber
from src.utils.error_handler_refactored import get_logger
from src.utils.unified_performance import performance_monitor
from src.utils.memory_manager import get_memory_manager
from src.utils.cache_manager import CacheManager

logger = get_logger(__name__)


def safe_index_access(data, index=0):
    """
    타입 안전한 인덱스 접근 함수

    Args:
        data: 접근할 데이터
        index: 접근할 인덱스 (기본값: 0)

    Returns:
        인덱스에 해당하는 값 또는 None
    """
    if hasattr(data, "__getitem__") and hasattr(data, "__len__"):
        return data[index] if len(data) > index else None  # type: ignore
    return data


@dataclass
class FeatureGroup:
    """특성 그룹을 나타내는 데이터 클래스"""

    name: str
    features: np.ndarray
    feature_names: List[str]
    importance_scores: Optional[np.ndarray] = None
    description: str = ""


@dataclass
class FeatureExtractionResult:
    """특성 추출 결과"""

    feature_matrix: np.ndarray
    feature_names: List[str]
    feature_groups: Dict[str, FeatureGroup]
    extraction_metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]


class FeatureExtractor:
    """특성 추출 엔진 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        FeatureExtractor 초기화

        Args:
            config: 특성 추출 설정
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()
        self.cache_manager = CacheManager()

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

        # 특성 그룹 정의
        self.feature_groups_config = {
            "basic_statistics": {
                "enabled": True,
                "max_features": 20,
                "description": "기본 통계적 특성",
            },
            "pattern_features": {
                "enabled": True,
                "max_features": 25,
                "description": "패턴 기반 특성",
            },
            "time_series_features": {
                "enabled": True,
                "max_features": 15,
                "description": "시계열 특성",
            },
            "graph_features": {
                "enabled": True,
                "max_features": 20,
                "description": "그래프 기반 특성",
            },
            "roi_features": {
                "enabled": True,
                "max_features": 15,
                "description": "ROI 기반 특성",
            },
            "distribution_features": {
                "enabled": True,
                "max_features": 10,
                "description": "분포 기반 특성",
            },
        }

        # 스케일러 초기화
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }

        logger.info("FeatureExtractor 초기화 완료")

    @performance_monitor
    def extract_all_features(
        self, analysis_results: Dict[str, Any]
    ) -> FeatureExtractionResult:
        """
        모든 특성을 추출합니다.

        Args:
            analysis_results: 분석 결과 딕셔너리

        Returns:
            FeatureExtractionResult: 특성 추출 결과
        """
        with self.memory_manager.allocation_scope():
            logger.info("전체 특성 추출 시작")

            # 캐시 확인
            cache_key = self._create_cache_key(analysis_results)
            cached_result = self.cache_manager.get(cache_key)

            if cached_result:
                logger.info("캐시된 특성 추출 결과 사용")
                return cached_result

            feature_groups = {}
            all_features = []
            all_feature_names = []
            extraction_metadata = {}

            # 1. 기본 통계적 특성
            if self.feature_groups_config["basic_statistics"]["enabled"]:
                basic_group = self._extract_basic_statistics(analysis_results)
                feature_groups["basic_statistics"] = basic_group
                all_features.append(basic_group.features)
                all_feature_names.extend(basic_group.feature_names)

            # 2. 패턴 기반 특성
            if self.feature_groups_config["pattern_features"]["enabled"]:
                pattern_group = self._extract_pattern_features(analysis_results)
                feature_groups["pattern_features"] = pattern_group
                all_features.append(pattern_group.features)
                all_feature_names.extend(pattern_group.feature_names)

            # 3. 시계열 특성
            if self.feature_groups_config["time_series_features"]["enabled"]:
                time_group = self._extract_time_series_features(analysis_results)
                feature_groups["time_series_features"] = time_group
                all_features.append(time_group.features)
                all_feature_names.extend(time_group.feature_names)

            # 4. 그래프 기반 특성
            if self.feature_groups_config["graph_features"]["enabled"]:
                graph_group = self._extract_graph_features(analysis_results)
                feature_groups["graph_features"] = graph_group
                all_features.append(graph_group.features)
                all_feature_names.extend(graph_group.feature_names)

            # 5. ROI 기반 특성
            if self.feature_groups_config["roi_features"]["enabled"]:
                roi_group = self._extract_roi_features(analysis_results)
                feature_groups["roi_features"] = roi_group
                all_features.append(roi_group.features)
                all_feature_names.extend(roi_group.feature_names)

            # 6. 분포 기반 특성
            if self.feature_groups_config["distribution_features"]["enabled"]:
                dist_group = self._extract_distribution_features(analysis_results)
                feature_groups["distribution_features"] = dist_group
                all_features.append(dist_group.features)
                all_feature_names.extend(dist_group.feature_names)

            # 특성 행렬 결합
            if all_features:
                feature_matrix = np.concatenate(all_features, axis=0)
            else:
                feature_matrix = np.array([])

            # 품질 메트릭 계산
            quality_metrics = self._calculate_quality_metrics(
                feature_matrix, all_feature_names
            )

            # 추출 메타데이터
            extraction_metadata = {
                "total_features": len(all_feature_names),
                "feature_groups_count": len(feature_groups),
                "extraction_time": datetime.now().isoformat(),
                "config_used": dict(self.config),
            }

            result = FeatureExtractionResult(
                feature_matrix=feature_matrix,
                feature_names=all_feature_names,
                feature_groups=feature_groups,
                extraction_metadata=extraction_metadata,
                quality_metrics=quality_metrics,
            )

            # 결과 캐싱
            self.cache_manager.set(cache_key, result)

            logger.info(f"특성 추출 완료: {len(all_feature_names)}개 특성")
            return result

    def _extract_basic_statistics(
        self, analysis_results: Dict[str, Any]
    ) -> FeatureGroup:
        """기본 통계적 특성 추출"""
        features = []
        feature_names = []

        # 패턴 분석 결과에서 기본 통계 추출
        if "pattern_analysis" in analysis_results:
            pattern_data = analysis_results["pattern_analysis"]

            # 빈도 통계
            if "frequency_stats" in pattern_data:
                freq_stats = pattern_data["frequency_stats"]
                features.extend(
                    [
                        freq_stats.get("mean", 0.0),
                        freq_stats.get("std", 0.0),
                        freq_stats.get("min", 0.0),
                        freq_stats.get("max", 0.0),
                        freq_stats.get("median", 0.0),
                    ]
                )
                feature_names.extend(
                    ["freq_mean", "freq_std", "freq_min", "freq_max", "freq_median"]
                )

            # 간격 통계
            if "gap_stats" in pattern_data:
                gap_stats = pattern_data["gap_stats"]
                features.extend(
                    [
                        gap_stats.get("mean_gap", 0.0),
                        gap_stats.get("std_gap", 0.0),
                        gap_stats.get("max_gap", 0.0),
                    ]
                )
                feature_names.extend(["gap_mean", "gap_std", "gap_max"])

            # 연속성 통계
            if "consecutive_stats" in pattern_data:
                cons_stats = pattern_data["consecutive_stats"]
                features.extend(
                    [
                        cons_stats.get("max_consecutive", 0.0),
                        cons_stats.get("avg_consecutive", 0.0),
                        cons_stats.get("consecutive_ratio", 0.0),
                    ]
                )
                feature_names.extend(
                    ["max_consecutive", "avg_consecutive", "consecutive_ratio"]
                )

        # 분포 분석 결과에서 통계 추출
        if "distribution_analysis" in analysis_results:
            dist_data = analysis_results["distribution_analysis"]

            # 홀짝 분포
            if "even_odd" in dist_data:
                even_odd_patterns = dist_data["even_odd"]
                if even_odd_patterns:
                    # 가장 빈번한 홀짝 패턴
                    most_common = max(even_odd_patterns, key=lambda x: x.frequency)
                    features.extend(
                        [
                            most_common.pattern[0],  # 짝수 개수
                            most_common.pattern[1],  # 홀수 개수
                            most_common.frequency,  # 빈도
                        ]
                    )
                    feature_names.extend(
                        [
                            "most_common_even_count",
                            "most_common_odd_count",
                            "odd_even_frequency",
                        ]
                    )

            # 고저 분포
            if "low_high" in dist_data:
                low_high_patterns = dist_data["low_high"]
                if low_high_patterns:
                    most_common = max(low_high_patterns, key=lambda x: x.frequency)
                    features.extend(
                        [
                            most_common.pattern[0],  # 저위 개수
                            most_common.pattern[1],  # 고위 개수
                            most_common.frequency,  # 빈도
                        ]
                    )
                    feature_names.extend(
                        [
                            "most_common_low_count",
                            "most_common_high_count",
                            "low_high_frequency",
                        ]
                    )

        # 부족한 특성을 0으로 패딩
        target_size = self.feature_groups_config["basic_statistics"]["max_features"]
        while len(features) < target_size:
            features.append(0.0)
            feature_names.append(f"basic_padding_{len(features)}")

        # 초과 특성 제거
        features = features[:target_size]
        feature_names = feature_names[:target_size]

        return FeatureGroup(
            name="basic_statistics",
            features=np.array(features),
            feature_names=feature_names,
            description="기본 통계적 특성",
        )

    def _extract_pattern_features(
        self, analysis_results: Dict[str, Any]
    ) -> FeatureGroup:
        """패턴 기반 특성 추출"""
        features = []
        feature_names = []

        if "pattern_analysis" in analysis_results:
            pattern_data = analysis_results["pattern_analysis"]

            # 클러스터 특성
            if "cluster_info" in pattern_data:
                cluster_info = pattern_data["cluster_info"]
                features.extend(
                    [
                        cluster_info.get("cluster_count", 0.0),
                        cluster_info.get("avg_cluster_size", 0.0),
                        cluster_info.get("cluster_density", 0.0),
                        cluster_info.get("silhouette_score", 0.0),
                    ]
                )
                feature_names.extend(
                    [
                        "cluster_count",
                        "avg_cluster_size",
                        "cluster_density",
                        "silhouette_score",
                    ]
                )

            # 트렌드 특성
            if "trend_scores" in pattern_data:
                trend_scores = pattern_data["trend_scores"]
                if isinstance(trend_scores, dict):
                    trend_values = list(trend_scores.values())
                    features.extend(
                        [
                            np.mean(trend_values) if trend_values else 0.0,
                            np.std(trend_values) if trend_values else 0.0,
                            max(trend_values) if trend_values else 0.0,
                            min(trend_values) if trend_values else 0.0,
                        ]
                    )
                    feature_names.extend(
                        ["trend_mean", "trend_std", "trend_max", "trend_min"]
                    )

            # 위험도 특성
            if "risk_scores" in pattern_data:
                risk_data = pattern_data["risk_scores"]
                if isinstance(risk_data, dict):
                    features.extend(
                        [
                            risk_data.get("overall_risk", 0.0),
                            risk_data.get("frequency_risk", 0.0),
                            risk_data.get("pattern_risk", 0.0),
                        ]
                    )
                    feature_names.extend(
                        ["overall_risk", "frequency_risk", "pattern_risk"]
                    )

        # 쌍 분석 특성
        if "pair_analysis" in analysis_results:
            pair_data = analysis_results["pair_analysis"]

            if "pair_frequency" in pair_data:
                pair_freq = pair_data["pair_frequency"]
                if isinstance(pair_freq, dict) and pair_freq:
                    freq_values = list(pair_freq.values())
                    features.extend(
                        [
                            np.mean(freq_values),
                            np.std(freq_values),
                            max(freq_values),
                            len(freq_values),
                        ]
                    )
                    feature_names.extend(
                        [
                            "pair_freq_mean",
                            "pair_freq_std",
                            "pair_freq_max",
                            "pair_count",
                        ]
                    )

            if "centrality_scores" in pair_data:
                centrality = pair_data["centrality_scores"]
                if isinstance(centrality, dict) and centrality:
                    cent_values = list(centrality.values())
                    features.extend(
                        [np.mean(cent_values), np.std(cent_values), max(cent_values)]
                    )
                    feature_names.extend(
                        ["centrality_mean", "centrality_std", "centrality_max"]
                    )

        # 부족한 특성을 0으로 패딩
        target_size = self.feature_groups_config["pattern_features"]["max_features"]
        while len(features) < target_size:
            features.append(0.0)
            feature_names.append(f"pattern_padding_{len(features)}")

        # 초과 특성 제거
        features = features[:target_size]
        feature_names = feature_names[:target_size]

        return FeatureGroup(
            name="pattern_features",
            features=np.array(features),
            feature_names=feature_names,
            description="패턴 기반 특성",
        )

    def _extract_time_series_features(
        self, analysis_results: Dict[str, Any]
    ) -> FeatureGroup:
        """시계열 특성 추출"""
        features = []
        feature_names = []

        # 트렌드 분석 특성
        if "trend_analysis" in analysis_results:
            trend_data = analysis_results["trend_analysis"]

            if "trend_direction" in trend_data:
                trend_dir = trend_data["trend_direction"]
                features.extend(
                    [
                        trend_dir.get("upward_trend", 0.0),
                        trend_dir.get("downward_trend", 0.0),
                        trend_dir.get("stable_trend", 0.0),
                    ]
                )
                feature_names.extend(["upward_trend", "downward_trend", "stable_trend"])

            if "seasonality" in trend_data:
                seasonality = trend_data["seasonality"]
                features.extend(
                    [
                        seasonality.get("seasonal_strength", 0.0),
                        seasonality.get("period", 0.0),
                    ]
                )
                feature_names.extend(["seasonal_strength", "seasonal_period"])

        # 최근성 특성
        if "recency_analysis" in analysis_results:
            recency_data = analysis_results["recency_analysis"]

            if "recent_frequency" in recency_data:
                recent_freq = recency_data["recent_frequency"]
                if isinstance(recent_freq, dict):
                    freq_values = list(recent_freq.values())
                    features.extend(
                        [
                            np.mean(freq_values) if freq_values else 0.0,
                            np.std(freq_values) if freq_values else 0.0,
                        ]
                    )
                    feature_names.extend(["recent_freq_mean", "recent_freq_std"])

            if "gap_since_last" in recency_data:
                gap_data = recency_data["gap_since_last"]
                if isinstance(gap_data, dict):
                    gap_values = list(gap_data.values())
                    features.extend(
                        [
                            np.mean(gap_values) if gap_values else 0.0,
                            max(gap_values) if gap_values else 0.0,
                            min(gap_values) if gap_values else 0.0,
                        ]
                    )
                    feature_names.extend(["gap_mean", "gap_max", "gap_min"])

        # 부족한 특성을 0으로 패딩
        target_size = self.feature_groups_config["time_series_features"]["max_features"]
        while len(features) < target_size:
            features.append(0.0)
            feature_names.append(f"time_padding_{len(features)}")

        # 초과 특성 제거
        features = features[:target_size]
        feature_names = feature_names[:target_size]

        return FeatureGroup(
            name="time_series_features",
            features=np.array(features),
            feature_names=feature_names,
            description="시계열 특성",
        )

    def _extract_graph_features(self, analysis_results: Dict[str, Any]) -> FeatureGroup:
        """그래프 기반 특성 추출"""
        features = []
        feature_names = []

        # 그래프 구조 특성
        if "graph_analysis" in analysis_results:
            graph_data = analysis_results["graph_analysis"]

            if "network_metrics" in graph_data:
                network = graph_data["network_metrics"]
                features.extend(
                    [
                        network.get("density", 0.0),
                        network.get("clustering_coefficient", 0.0),
                        network.get("average_path_length", 0.0),
                        network.get("diameter", 0.0),
                    ]
                )
                feature_names.extend(
                    [
                        "graph_density",
                        "clustering_coeff",
                        "avg_path_length",
                        "graph_diameter",
                    ]
                )

            if "centrality_measures" in graph_data:
                centrality = graph_data["centrality_measures"]
                if isinstance(centrality, dict):
                    # 중심성 측정값들의 통계
                    cent_values = list(centrality.values())
                    features.extend(
                        [
                            np.mean(cent_values) if cent_values else 0.0,
                            np.std(cent_values) if cent_values else 0.0,
                            max(cent_values) if cent_values else 0.0,
                        ]
                    )
                    feature_names.extend(
                        ["centrality_mean", "centrality_std", "centrality_max"]
                    )

        # 연결성 특성
        if "connectivity_analysis" in analysis_results:
            conn_data = analysis_results["connectivity_analysis"]

            features.extend(
                [
                    conn_data.get("connected_components", 0.0),
                    conn_data.get("edge_count", 0.0),
                    conn_data.get("node_count", 0.0),
                ]
            )
            feature_names.extend(["connected_components", "edge_count", "node_count"])

        # 부족한 특성을 0으로 패딩
        target_size = self.feature_groups_config["graph_features"]["max_features"]
        while len(features) < target_size:
            features.append(0.0)
            feature_names.append(f"graph_padding_{len(features)}")

        # 초과 특성 제거
        features = features[:target_size]
        feature_names = feature_names[:target_size]

        return FeatureGroup(
            name="graph_features",
            features=np.array(features),
            feature_names=feature_names,
            description="그래프 기반 특성",
        )

    def _extract_roi_features(self, analysis_results: Dict[str, Any]) -> FeatureGroup:
        """ROI 기반 특성 추출"""
        features = []
        feature_names = []

        if "roi_analysis" in analysis_results:
            roi_data = analysis_results["roi_analysis"]

            # ROI 점수 특성
            if "number_roi_scores" in roi_data:
                roi_scores = roi_data["number_roi_scores"]
                if isinstance(roi_scores, dict) and "scores" in roi_scores:
                    scores = roi_scores["scores"]
                    if scores:
                        score_values = list(scores.values())
                        features.extend(
                            [
                                np.mean(score_values),
                                np.std(score_values),
                                max(score_values),
                                min(score_values),
                            ]
                        )
                        feature_names.extend(
                            [
                                "roi_score_mean",
                                "roi_score_std",
                                "roi_score_max",
                                "roi_score_min",
                            ]
                        )

            # ROI 트렌드 특성
            if "roi_trend_by_pattern" in roi_data:
                roi_trend = roi_data["roi_trend_by_pattern"]
                if isinstance(roi_trend, dict):
                    # 패턴별 ROI 트렌드 통계
                    trend_values = []
                    for pattern_data in roi_trend.values():
                        if (
                            isinstance(pattern_data, dict)
                            and "trend_score" in pattern_data
                        ):
                            trend_values.append(pattern_data["trend_score"])

                    if trend_values:
                        features.extend(
                            [
                                np.mean(trend_values),
                                np.std(trend_values),
                                max(trend_values),
                            ]
                        )
                        feature_names.extend(
                            ["roi_trend_mean", "roi_trend_std", "roi_trend_max"]
                        )

            # 위험 매트릭스 특성
            if "risk_matrix" in roi_data:
                risk_matrix = roi_data["risk_matrix"]
                if isinstance(risk_matrix, dict):
                    features.extend(
                        [
                            risk_matrix.get("overall_risk", 0.0),
                            risk_matrix.get("volatility", 0.0),
                            risk_matrix.get("max_drawdown", 0.0),
                        ]
                    )
                    feature_names.extend(["overall_risk", "volatility", "max_drawdown"])

        # 부족한 특성을 0으로 패딩
        target_size = self.feature_groups_config["roi_features"]["max_features"]
        while len(features) < target_size:
            features.append(0.0)
            feature_names.append(f"roi_padding_{len(features)}")

        # 초과 특성 제거
        features = features[:target_size]
        feature_names = feature_names[:target_size]

        return FeatureGroup(
            name="roi_features",
            features=np.array(features),
            feature_names=feature_names,
            description="ROI 기반 특성",
        )

    def _extract_distribution_features(
        self, analysis_results: Dict[str, Any]
    ) -> FeatureGroup:
        """분포 기반 특성 추출"""
        features = []
        feature_names = []

        if "distribution_analysis" in analysis_results:
            dist_data = analysis_results["distribution_analysis"]

            # 범위 분포 특성
            if "ranges" in dist_data:
                range_patterns = dist_data["ranges"]
                if range_patterns:
                    # 가장 빈번한 범위 패턴들의 통계
                    frequencies = [pattern.frequency for pattern in range_patterns]
                    features.extend(
                        [np.mean(frequencies), np.std(frequencies), max(frequencies)]
                    )
                    feature_names.extend(
                        ["range_freq_mean", "range_freq_std", "range_freq_max"]
                    )

            # 합계 분포 특성
            if "sum_ranges" in dist_data:
                sum_patterns = dist_data["sum_ranges"]
                if sum_patterns:
                    frequencies = [pattern.frequency for pattern in sum_patterns]
                    features.extend([np.mean(frequencies), np.std(frequencies)])
                    feature_names.extend(["sum_freq_mean", "sum_freq_std"])

            # 소수 분포 특성
            if "prime_distribution" in dist_data:
                prime_patterns = dist_data["prime_distribution"]
                if prime_patterns:
                    frequencies = [pattern.frequency for pattern in prime_patterns]
                    features.extend([np.mean(frequencies), max(frequencies)])
                    feature_names.extend(["prime_freq_mean", "prime_freq_max"])

        # 부족한 특성을 0으로 패딩
        target_size = self.feature_groups_config["distribution_features"][
            "max_features"
        ]
        while len(features) < target_size:
            features.append(0.0)
            feature_names.append(f"dist_padding_{len(features)}")

        # 초과 특성 제거
        features = features[:target_size]
        feature_names = feature_names[:target_size]

        return FeatureGroup(
            name="distribution_features",
            features=np.array(features),
            feature_names=feature_names,
            description="분포 기반 특성",
        )

    @performance_monitor
    def optimize_feature_selection(
        self,
        features: np.ndarray,
        feature_names: List[str],
        target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        특성 선택 최적화

        Args:
            features: 특성 행렬
            feature_names: 특성 이름 리스트
            target: 타겟 변수 (있는 경우)

        Returns:
            Tuple[선택된 특성, 선택된 특성 이름, 중요도 점수]
        """
        logger.info(f"특성 선택 최적화 시작: {features.shape[0]}개 특성")

        if features.size == 0:
            logger.warning("빈 특성 행렬")
            return features, feature_names, np.array([])

        # 특성이 1차원인 경우 2차원으로 변환
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 타겟이 없는 경우 분산 기반 선택
        if target is None:
            return self._variance_based_selection(features, feature_names)

        # 타겟이 있는 경우 지도 학습 기반 선택
        return self._supervised_feature_selection(features, feature_names, target)

    def _variance_based_selection(
        self, features: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """분산 기반 특성 선택"""
        # 분산 계산
        variances = (
            np.var(features, axis=0)
            if features.shape[0] > 1
            else np.ones(features.shape[1])
        )

        # 분산이 0인 특성 제거
        non_zero_var_mask = variances > 1e-8

        selected_features = features[:, non_zero_var_mask]

        # numpy.where를 사용하여 타입 안전한 방식으로 특성 이름 선택
        selected_indices = np.where(non_zero_var_mask)[0]
        selected_names = [
            safe_index_access(feature_names, i)
            for i in selected_indices
            if safe_index_access(feature_names, i) is not None
        ]

        importance_scores = variances[non_zero_var_mask]

        # 분산 기준으로 상위 특성 선택
        if len(selected_names) > self.max_features:
            top_indices = np.argsort(importance_scores)[-self.max_features :]
            selected_features = selected_features[:, top_indices]
            selected_names = [selected_names[i] for i in top_indices]
            importance_scores = importance_scores[top_indices]

        logger.info(f"분산 기반 특성 선택 완료: {len(selected_names)}개 특성 선택")
        return selected_features, selected_names, importance_scores

    def _supervised_feature_selection(
        self, features: np.ndarray, feature_names: List[str], target: np.ndarray
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """지도 학습 기반 특성 선택"""
        try:
            # 특성 선택 방법에 따라 선택
            if self.feature_selection_method == "mutual_info":
                selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(self.max_features, features.shape[1]),
                )
            else:  # f_classif
                selector = SelectKBest(
                    score_func=f_classif, k=min(self.max_features, features.shape[1])
                )

            # 특성 선택 수행
            selected_features = selector.fit_transform(features.T, target).T
            selected_mask = selector.get_support()

            # numpy.where를 사용하여 타입 안전한 방식으로 특성 이름 선택
            selected_indices = np.where(selected_mask)[0]
            selected_names = [
                safe_index_access(feature_names, i)
                for i in selected_indices
                if safe_index_access(feature_names, i) is not None
            ]

            importance_scores = selector.scores_[selected_mask]

            logger.info(
                f"지도 학습 기반 특성 선택 완료: {len(selected_names)}개 특성 선택"
            )
            return selected_features, selected_names, importance_scores

        except Exception as e:
            logger.warning(f"지도 학습 기반 특성 선택 실패: {e}, 분산 기반으로 대체")
            return self._variance_based_selection(features, feature_names)

    def _calculate_quality_metrics(
        self, features: np.ndarray, feature_names: List[str]
    ) -> Dict[str, float]:
        """특성 품질 메트릭 계산"""
        if features.size == 0:
            return {
                "feature_count": 0,
                "mean_variance": 0.0,
                "correlation_score": 0.0,
                "missing_ratio": 0.0,
            }

        # 특성이 1차원인 경우 2차원으로 변환
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 분산 계산
        variances = (
            np.var(features, axis=0)
            if features.shape[0] > 1
            else np.ones(features.shape[1])
        )
        mean_variance = np.mean(variances)

        # 상관관계 점수 (특성 간 평균 상관관계)
        try:
            if features.shape[0] > 1 and features.shape[1] > 1:
                corr_matrix = np.corrcoef(features, rowvar=False)
                # 대각선 제외한 상관관계의 절댓값 평균
                mask = np.array(~np.eye(corr_matrix.shape[0], dtype=bool), dtype=bool)
                correlation_score = np.mean(np.abs(corr_matrix[mask]))
            else:
                correlation_score = 0.0
        except:
            correlation_score = 0.0

        # 결측값 비율
        missing_ratio = np.mean(np.isnan(features)) if features.size > 0 else 0.0

        return {
            "feature_count": len(feature_names),
            "mean_variance": float(mean_variance),
            "correlation_score": float(correlation_score),
            "missing_ratio": float(missing_ratio),
        }

    def _create_cache_key(self, analysis_results: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        import hashlib
        import json

        # 분석 결과의 해시 생성
        result_str = json.dumps(analysis_results, sort_keys=True, default=str)
        config_str = json.dumps(dict(self.config), sort_keys=True, default=str)

        combined_str = result_str + config_str
        return hashlib.md5(combined_str.encode()).hexdigest()

    def scale_features(
        self, features: np.ndarray, method: Optional[str] = None
    ) -> np.ndarray:
        """
        특성 스케일링

        Args:
            features: 스케일링할 특성 행렬
            method: 스케일링 방법 ('standard', 'minmax', 'robust')

        Returns:
            스케일링된 특성 행렬
        """
        if method is None:
            method = self.scaling_method

        if features.size == 0:
            return features

        # 특성이 1차원인 경우 2차원으로 변환
        if features.ndim == 1:
            features = features.reshape(1, -1)
            was_1d = True
        else:
            was_1d = False

        try:
            scaler = self.scalers[method]
            scaled_features = scaler.fit_transform(features.T).T

            if was_1d:
                scaled_features = scaled_features.flatten()

            return scaled_features

        except Exception as e:
            logger.warning(f"특성 스케일링 실패: {e}")
            return features
