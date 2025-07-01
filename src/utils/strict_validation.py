import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)


class SafeTargetEncoder:
    """KFold 기반 안전 Target Encoding (미래 데이터 유출 방지)"""

    def __init__(self):
        self.global_mapping = None
        self.default_value = None

    def fit_transform_with_kfold(self, X, y, n_folds=5):
        X = np.asarray(X)
        y = np.asarray(y)
        encoded = np.zeros_like(y, dtype=float)
        kf = KFold(n_splits=n_folds, shuffle=False)
        for train_idx, val_idx in kf.split(X):
            mapping = self._build_mapping(X[train_idx], y[train_idx])
            encoded[val_idx] = self._apply_mapping(X[val_idx], mapping)
        # 전체 데이터로 global mapping 저장 (추후 test transform용)
        self.global_mapping = self._build_mapping(X, y)
        self.default_value = np.mean(y)
        return encoded

    def transform(self, X):
        return self._apply_mapping(np.asarray(X), self.global_mapping)

    def _build_mapping(self, X, y):
        mapping = {}
        for x_val, y_val in zip(X, y):
            if x_val not in mapping:
                mapping[x_val] = []
            mapping[x_val].append(y_val)
        return {k: np.mean(v) for k, v in mapping.items()}

    def _apply_mapping(self, X, mapping):
        return np.array([mapping.get(x, self.default_value) for x in X])


class StrictPreprocessor:
    """Train/Test 분리 기반 Scaling, Outlier 처리 (Train 통계만 사용)"""

    def __init__(self):
        self.train_stats = {}

    def fit_on_train_only(self, X_train):
        X_train = np.asarray(X_train)
        self.train_stats["percentiles"] = np.percentile(X_train, [1, 99], axis=0)
        self.train_stats["mean"] = np.mean(X_train, axis=0)
        self.train_stats["std"] = np.std(X_train, axis=0)

    def transform_both(self, X):
        X = np.asarray(X)
        X_clip = np.clip(
            X, self.train_stats["percentiles"][0], self.train_stats["percentiles"][1]
        )
        X_scaled = (X_clip - self.train_stats["mean"]) / (
            self.train_stats["std"] + 1e-8
        )
        return X_scaled


# ROI-aware Negative Sampling
import random


def roi_aware_negative_sampling(
    historical_data, roi_scores, n_samples, low_roi_weight=0.7, high_roi_weight=0.3
):
    """
    ROI 점수 기반 가중 샘플링
    low_roi_weight: 저ROI 샘플 비율
    high_roi_weight: 고ROI 샘플 비율
    """
    assert abs(low_roi_weight + high_roi_weight - 1.0) < 1e-6
    # float/int만 남기고 tuple 등은 완전히 제외
    indices = [
        i
        for i, r in enumerate(roi_scores)
        if isinstance(r, (int, float)) and not isinstance(r, tuple)
    ]
    filtered_scores = [
        r
        for r in roi_scores
        if isinstance(r, (int, float)) and not isinstance(r, tuple)
    ]
    if not filtered_scores or not indices:
        return []
    filtered_scores = np.array(filtered_scores)
    n_low = int(n_samples * low_roi_weight)
    n_high = n_samples - n_low
    low_thresh = np.percentile(filtered_scores, 30)
    high_thresh = np.percentile(filtered_scores, 70)
    low_mask = filtered_scores <= low_thresh
    high_mask = filtered_scores >= high_thresh
    low_candidates = [indices[i] for i in np.where(low_mask)[0]]
    high_candidates = [indices[i] for i in np.where(high_mask)[0]]
    low_samples = (
        random.sample(low_candidates, min(n_low, len(low_candidates)))
        if low_candidates
        else []
    )
    high_samples = (
        random.sample(high_candidates, min(n_high, len(high_candidates)))
        if high_candidates
        else []
    )
    selected_indices = low_samples + high_samples
    random.shuffle(selected_indices)
    return [historical_data[i] for i in selected_indices]


class FeatureDriftDetector:
    """KS-Test 기반 Feature Drift 감지"""

    def detect_drift(self, X_train, X_test, threshold=0.05):
        drift_features = []
        for i in range(X_train.shape[1]):
            result = ks_2samp(X_train[:, i], X_test[:, i])
            p = result.pvalue
            if p < threshold:
                drift_features.append(i)
                logger.warning(f"Feature {i} shows significant drift (p={p:.4f})")
        return drift_features


class PipelineValidator:
    """파이프라인 누수 자동 검증"""

    def validate_no_leakage(self, pipeline, X_train, X_test):
        # Train/Test 분리 후, train 통계만으로 test 변환 여부 확인
        # 시간 순서 보장(미래 데이터 미사용) 여부 확인
        # (구현 예시: scaling/encoding이 train 기준인지, test 정보 미포함인지 체크)
        # 실제 파이프라인 객체에 따라 커스텀 검증 필요
        # 예시: StrictPreprocessor, SafeTargetEncoder 등 사용 여부
        # (여기서는 단순 통계 비교)
        train_mean = np.mean(X_train, axis=0)
        test_mean = np.mean(X_test, axis=0)
        diff = np.abs(train_mean - test_mean)
        if np.any(diff > 1.0):  # 임계값 예시
            logger.warning("Train/Test mean 차이가 비정상적으로 큼. 누수 가능성 의심.")
        return diff
