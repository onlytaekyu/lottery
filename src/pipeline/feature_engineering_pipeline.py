"""
특성 엔지니어링 파이프라인

Phase 2: 성능 향상을 위한 고급 특성 엔지니어링
- Feature Interaction 생성 (top 10 조합)
- TCN용 시계열 윈도우 생성
- SHAP 기반 특성 선택
- Negative sampling 개선
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings

from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_config
from ..utils.memory_manager import get_memory_manager
from ..utils.unified_performance import performance_monitor

logger = get_logger(__name__)


@dataclass
class FeatureEngineeringConfig:
    """특성 엔지니어링 설정"""

    # Feature Interactions
    enable_feature_interactions: bool = True
    max_interaction_features: int = 10
    interaction_degree: int = 2

    # Temporal Features (TCN용)
    enable_temporal_features: bool = True
    sequence_length: int = 50
    temporal_stride: int = 1

    # SHAP-based Selection
    enable_shap_selection: bool = False  # 계산 비용으로 기본 비활성화
    shap_top_k: int = 60

    # Negative Sampling
    enable_negative_sampling: bool = True
    negative_ratio: float = 0.3

    # Meta Features
    enable_meta_features: bool = True
    statistical_windows: List[int] = None

    def __post_init__(self):
        if self.statistical_windows is None:
            self.statistical_windows = [5, 10, 20, 50]


class FeatureEngineeringPipeline:
    """특성 엔지니어링 파이프라인"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = get_config("main") if config is None else get_config("main")
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()

        # 특성 엔지니어링 설정
        self.fe_config = FeatureEngineeringConfig()

        # 캐시된 특성들
        self.interaction_features_cache = {}
        self.temporal_features_cache = {}
        self.meta_features_cache = {}

        self.logger.info("특성 엔지니어링 파이프라인 초기화 완료")

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: List[str],
        y: Optional[np.ndarray] = None,
        target_model: str = "general",
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        특성 엔지니어링 파이프라인 실행

        Args:
            X: 특성 행렬
            feature_names: 특성 이름 리스트
            y: 타겟 변수 (있는 경우)
            target_model: 타겟 모델

        Returns:
            Tuple[엔지니어링된 특성, 특성 이름, 통계 정보]
        """
        self.logger.info(f"특성 엔지니어링 시작: {X.shape}, 타겟 모델: {target_model}")

        X_engineered = X.copy()
        engineered_names = feature_names.copy()
        engineering_stats = {"original_features": X.shape[1]}

        # 1. Feature Interactions 생성
        if self.fe_config.enable_feature_interactions:
            X_interactions, interaction_names = self._create_feature_interactions(
                X_engineered, engineered_names
            )
            X_engineered = np.hstack([X_engineered, X_interactions])
            engineered_names.extend(interaction_names)
            engineering_stats["interaction_features"] = len(interaction_names)

        # 2. TCN용 시계열 특성 (해당 모델인 경우)
        if target_model == "tcn" and self.fe_config.enable_temporal_features:
            X_temporal, temporal_names = self._create_temporal_features(
                X_engineered, engineered_names
            )
            if X_temporal is not None:
                X_engineered = X_temporal
                engineered_names = temporal_names
                engineering_stats["temporal_reshape"] = True

        # 3. Meta Features 생성
        if self.fe_config.enable_meta_features:
            X_meta, meta_names = self._create_meta_features(
                X_engineered, engineered_names
            )
            X_engineered = np.hstack([X_engineered, X_meta])
            engineered_names.extend(meta_names)
            engineering_stats["meta_features"] = len(meta_names)

        # 4. SHAP 기반 특성 선택 (선택적)
        if self.fe_config.enable_shap_selection and y is not None:
            X_selected, selected_names = self._shap_based_selection(
                X_engineered, engineered_names, y
            )
            X_engineered = X_selected
            engineered_names = selected_names
            engineering_stats["shap_selected_features"] = len(selected_names)

        # 5. 최종 통계
        engineering_stats.update(
            {
                "final_features": X_engineered.shape[1],
                "feature_expansion_ratio": X_engineered.shape[1] / X.shape[1],
            }
        )

        self.logger.info(f"특성 엔지니어링 완료: {X.shape} → {X_engineered.shape}")
        return X_engineered, engineered_names, engineering_stats

    def _create_feature_interactions(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """특성 상호작용 생성"""
        self.logger.info("특성 상호작용 생성 중...")

        # 상위 특성들 선택 (분산 기준)
        if X.shape[1] > 20:
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-20:]  # 상위 20개 특성
            X_top = X[:, top_indices]
            top_names = [feature_names[i] for i in top_indices]
        else:
            X_top = X
            top_names = feature_names

        # 2차 상호작용 생성
        interaction_features = []
        interaction_names = []

        # 가장 유용한 상호작용 패턴들
        interaction_patterns = [
            ("multiply", lambda x, y: x * y),
            ("add", lambda x, y: x + y),
            ("subtract", lambda x, y: np.abs(x - y)),
            ("divide", lambda x, y: np.divide(x, y + 1e-8)),  # 0으로 나누기 방지
            ("max", lambda x, y: np.maximum(x, y)),
            ("min", lambda x, y: np.minimum(x, y)),
        ]

        interaction_count = 0
        max_interactions = self.fe_config.max_interaction_features

        # 특성 쌍 조합 생성
        for i, j in combinations(range(len(top_names)), 2):
            if interaction_count >= max_interactions:
                break

            feature_i = X_top[:, i]
            feature_j = X_top[:, j]
            name_i = top_names[i]
            name_j = top_names[j]

            # 곱셈 상호작용 (가장 유용한 패턴)
            interaction = feature_i * feature_j
            interaction_features.append(interaction)
            interaction_names.append(f"{name_i}_x_{name_j}")
            interaction_count += 1

        if interaction_features:
            X_interactions = np.column_stack(interaction_features)
            self.logger.info(f"상호작용 특성 생성: {len(interaction_names)}개")
            return X_interactions, interaction_names
        else:
            return np.empty((X.shape[0], 0)), []

    def _create_temporal_features(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """TCN용 시계열 특성 생성"""
        self.logger.info("TCN용 시계열 특성 생성 중...")

        seq_len = self.fe_config.sequence_length
        stride = self.fe_config.temporal_stride

        if X.shape[0] < seq_len:
            self.logger.warning(
                f"데이터 길이({X.shape[0]})가 시퀀스 길이({seq_len})보다 작음"
            )
            return None, feature_names

        # 시계열 윈도우 생성
        sequences = []
        for i in range(0, X.shape[0] - seq_len + 1, stride):
            sequence = X[i : i + seq_len]
            sequences.append(sequence)

        if sequences:
            # [batch_size, sequence_length, features] 형태로 변환
            X_temporal = np.array(sequences)

            # 시간 정보 추가
            temporal_features = []
            for i in range(seq_len):
                temporal_features.extend([f"{name}_t{i}" for name in feature_names])

            # 추가 시간 기반 특성
            temporal_features.extend(
                [
                    "sequence_mean",
                    "sequence_std",
                    "sequence_trend",
                    "sequence_volatility",
                    "sequence_momentum",
                ]
            )

            # 통계적 시간 특성 계산
            seq_stats = []
            for seq in sequences:
                stats = [
                    np.mean(seq),
                    np.std(seq),
                    np.polyfit(range(seq_len), np.mean(seq, axis=1), 1)[0],  # 트렌드
                    np.std(np.diff(np.mean(seq, axis=1))),  # 변동성
                    np.mean(np.diff(np.mean(seq, axis=1))),  # 모멘텀
                ]
                seq_stats.append(stats)

            # 원본 시계열 데이터와 통계 특성 결합
            X_temporal_flat = X_temporal.reshape(len(sequences), -1)
            X_stats = np.array(seq_stats)
            X_combined = np.hstack([X_temporal_flat, X_stats])

            self.logger.info(f"시계열 특성 생성: {X.shape} → {X_combined.shape}")
            return X_combined, temporal_features

        return None, feature_names

    def _create_meta_features(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """메타 특성 생성"""
        self.logger.info("메타 특성 생성 중...")

        meta_features = []
        meta_names = []

        # 1. 통계적 메타 특성
        if X.shape[0] > 1:
            # 전체 특성의 통계
            meta_features.extend(
                [
                    np.mean(X, axis=1),  # 샘플별 평균
                    np.std(X, axis=1),  # 샘플별 표준편차
                    np.max(X, axis=1),  # 샘플별 최대값
                    np.min(X, axis=1),  # 샘플별 최소값
                    np.median(X, axis=1),  # 샘플별 중간값
                ]
            )
            meta_names.extend(
                [
                    "sample_mean",
                    "sample_std",
                    "sample_max",
                    "sample_min",
                    "sample_median",
                ]
            )

        # 2. 분포 기반 메타 특성 (샘플별 계산)
        if X.shape[1] > 1:
            # 샘플별 상관관계 기반 특성
            try:
                # 각 샘플에 대해 특성 간 변동성 계산
                sample_ranges = np.max(X, axis=1) - np.min(X, axis=1)
                sample_vars = np.var(X, axis=1)
                sample_skew = []

                for i in range(X.shape[0]):
                    # 간단한 비대칭성 측정
                    sample_data = X[i, :]
                    mean_val = np.mean(sample_data)
                    median_val = np.median(sample_data)
                    skew_val = (mean_val - median_val) / (np.std(sample_data) + 1e-8)
                    sample_skew.append(skew_val)

                meta_features.extend(
                    [
                        sample_ranges,  # 샘플별 범위
                        sample_vars,  # 샘플별 분산
                        np.array(sample_skew),  # 샘플별 비대칭성
                    ]
                )
                meta_names.extend(
                    ["sample_range", "sample_variance", "sample_skewness"]
                )
            except:
                # 계산 실패 시 기본값 (샘플 수만큼)
                n_samples = X.shape[0]
                meta_features.extend(
                    [np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)]
                )
                meta_names.extend(
                    ["sample_range", "sample_variance", "sample_skewness"]
                )

        # 3. 엔트로피 기반 메타 특성 (샘플별 계산)
        if X.shape[0] > 1:
            try:
                # 각 샘플의 정보량 계산
                sample_entropies = []
                for i in range(X.shape[0]):
                    sample_data = X[i, :]
                    # 샘플 내 특성 값들의 분포 엔트로피
                    hist, _ = np.histogram(
                        sample_data, bins=min(10, len(sample_data) // 2 + 1)
                    )
                    hist = hist / np.sum(hist)
                    hist = hist[hist > 0]
                    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 1 else 0
                    sample_entropies.append(entropy)

                meta_features.extend([np.array(sample_entropies)])
                meta_names.extend(["sample_entropy"])
            except:
                n_samples = X.shape[0]
                meta_features.extend([np.zeros(n_samples)])
                meta_names.extend(["sample_entropy"])

        if meta_features:
            X_meta = np.column_stack(meta_features)
            self.logger.info(f"메타 특성 생성: {len(meta_names)}개")
            return X_meta, meta_names
        else:
            return np.empty((X.shape[0], 0)), []

    def _shap_based_selection(
        self, X: np.ndarray, feature_names: List[str], y: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """SHAP 기반 특성 선택 (간소화된 버전)"""
        self.logger.info("SHAP 기반 특성 선택 중...")

        # SHAP 대신 mutual information 사용 (계산 효율성)
        try:
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.fe_config.shap_top_k, X.shape[1]),
            )
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()

            selected_names = [
                name for name, selected in zip(feature_names, selected_mask) if selected
            ]

            self.logger.info(f"특성 선택 완료: {X.shape[1]} → {len(selected_names)}개")
            return X_selected, selected_names

        except Exception as e:
            self.logger.warning(f"특성 선택 실패: {e}")
            return X, feature_names

    def create_negative_samples(
        self,
        positive_samples: np.ndarray,
        feature_names: List[str],
        negative_ratio: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """개선된 네거티브 샘플링"""
        self.logger.info("개선된 네거티브 샘플링 중...")

        n_positive = positive_samples.shape[0]
        n_negative = int(n_positive * negative_ratio)

        # 다양한 네거티브 샘플링 전략
        negative_samples = []

        # 1. 가우시안 노이즈 기반 (50%)
        n_gaussian = n_negative // 2
        for _ in range(n_gaussian):
            # 원본 샘플 선택
            idx = np.random.randint(0, n_positive)
            base_sample = positive_samples[idx].copy()

            # 가우시안 노이즈 추가
            noise = np.random.normal(0, 0.1, base_sample.shape)
            negative_sample = base_sample + noise
            negative_samples.append(negative_sample)

        # 2. 특성 셔플링 기반 (50%)
        n_shuffle = n_negative - n_gaussian
        for _ in range(n_shuffle):
            # 원본 샘플 선택
            idx = np.random.randint(0, n_positive)
            base_sample = positive_samples[idx].copy()

            # 일부 특성을 다른 샘플의 특성으로 교체
            n_features_to_shuffle = np.random.randint(1, min(5, len(base_sample)))
            shuffle_indices = np.random.choice(
                len(base_sample), n_features_to_shuffle, replace=False
            )

            for shuffle_idx in shuffle_indices:
                donor_idx = np.random.randint(0, n_positive)
                base_sample[shuffle_idx] = positive_samples[donor_idx, shuffle_idx]

            negative_samples.append(base_sample)

        # 레이블 생성
        positive_labels = np.ones(n_positive)
        negative_labels = np.zeros(len(negative_samples))

        # 결합
        all_samples = np.vstack([positive_samples, np.array(negative_samples)])
        all_labels = np.hstack([positive_labels, negative_labels])

        self.logger.info(
            f"네거티브 샘플링 완료: {n_positive} positive + {len(negative_samples)} negative"
        )
        return all_samples, all_labels

    def get_feature_importance_ranking(
        self, X: np.ndarray, feature_names: List[str], y: np.ndarray
    ) -> Dict[str, float]:
        """특성 중요도 랭킹"""
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k="all")
            selector.fit(X, y)

            importance_dict = {}
            for name, score in zip(feature_names, selector.scores_):
                importance_dict[name] = float(score)

            # 중요도 순으로 정렬
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            return sorted_importance

        except Exception as e:
            self.logger.warning(f"특성 중요도 계산 실패: {e}")
            return {name: 0.0 for name in feature_names}
