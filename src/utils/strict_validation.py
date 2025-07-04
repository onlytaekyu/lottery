import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp
import logging
from typing import Dict, Any, Callable
from .unified_logging import get_logger

logger = logging.getLogger(__name__)


class AutoRecoverySystem:
    """
    검증 과정에서 발견된 오류를 자동으로 복구하는 시스템.
    """

    def __init__(self):
        self.recovery_strategies: Dict[str, Callable] = {
            "data_drift": self.fix_data_drift,
            "data_leakage": self.fix_data_leakage,
            "validation_error": self.fix_validation_error,
        }
        self.logger = get_logger(__name__)

    def recover(self, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        주어진 오류 유형에 따라 자동 복구를 시도합니다.

        Args:
            error_type: 오류 유형 ('data_drift', 'data_leakage' 등)
            context: 복구에 필요한 데이터 및 메타정보

        Returns:
            복구된 컨텍스트
        """
        recovery_func = self.recovery_strategies.get(error_type)
        if recovery_func:
            self.logger.info(f"'{error_type}' 오류에 대한 자동 복구를 시작합니다...")
            try:
                recovered_context = recovery_func(context)
                self.logger.info(f"✅ 자동 복구 완료: {error_type}")
                return recovered_context
            except Exception as e:
                self.logger.error(f"'{error_type}' 자동 복구 중 오류 발생: {e}")
                raise
        else:
            self.logger.warning(f"'{error_type}'에 대한 복구 전략이 없습니다.")
            return context

    def fix_data_drift(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        데이터 분포 변화(Drift)를 보정합니다.
        (예시: 테스트 데이터의 분포를 학습 데이터 분포와 유사하게 변환)
        """
        X_train = context.get("X_train")
        X_test = context.get("X_test")
        drift_features = context.get("drift_features", [])

        if X_train is None or X_test is None or not drift_features:
            return context

        X_test_recovered = np.copy(X_test)
        for col_idx in drift_features:
            train_mean, train_std = np.mean(X_train[:, col_idx]), np.std(
                X_train[:, col_idx]
            )
            test_mean, test_std = np.mean(X_test_recovered[:, col_idx]), np.std(
                X_test_recovered[:, col_idx]
            )

            # 평균과 표준편차를 학습 데이터 기준으로 조정
            if test_std > 1e-6:
                X_test_recovered[:, col_idx] = (
                    X_test_recovered[:, col_idx] - test_mean
                ) / test_std * train_std + train_mean

        context["X_test"] = X_test_recovered
        self.logger.info(f"{len(drift_features)}개 특성에 대한 분포 보정 적용 완료.")
        return context

    def fix_data_leakage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 유출(Leakage) 문제를 해결합니다. (예시: 중복 샘플 제거)"""
        X_train = context.get("X_train")
        X_test = context.get("X_test")

        if X_train is None or X_test is None:
            return context

        # DataFrame으로 변환하여 중복 제거
        import pandas as pd

        train_df = pd.DataFrame(X_train)
        test_df = pd.DataFrame(X_test)

        # 테스트 데이터에만 존재하는 샘플만 남김
        merged_df = pd.merge(train_df, test_df, how="inner")
        if not merged_df.empty:
            test_df_dedup = test_df.drop_duplicates().merge(
                train_df.drop_duplicates(), how="left", indicator=True
            )
            test_df_dedup = test_df_dedup[test_df_dedup["_merge"] == "left_only"].drop(
                columns=["_merge"]
            )
            context["X_test"] = test_df_dedup.to_numpy()
            self.logger.info(
                f"{len(test_df) - len(test_df_dedup)}개의 중복 샘플이 테스트 데이터에서 제거되었습니다."
            )

        return context

    def fix_validation_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """일반적인 데이터 유효성 오류를 수정합니다. (예시: 결측치 채우기)"""
        data = context.get("data")
        if data is None:
            return context

        # 예시: Numpy 배열의 NaN 값을 평균으로 대체
        if isinstance(data, np.ndarray) and np.isnan(data).any():
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            context["data"] = imputer.fit_transform(data)
            self.logger.info("결측치가 평균값으로 자동 대체되었습니다.")

        return context


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
            # 각 fold의 train 데이터로만 매핑 생성 → 미래 데이터 유출 차단
            mapping = self._build_mapping(X[train_idx], y[train_idx])
            default_val = np.mean(y[train_idx])  # fold-specific default
            encoded[val_idx] = np.array(
                [mapping.get(x, default_val) for x in X[val_idx]]
            )
        # 전체 데이터로 global mapping 저장 (추후 test transform용)
        self.global_mapping = self._build_mapping(X, y)
        self.default_value = np.mean(y)
        return encoded

    def transform(self, X):
        """Test/Inference 시 호출. fit_transform_with_kfold 이후 사용."""
        if self.global_mapping is None or self.default_value is None:
            raise RuntimeError(
                "SafeTargetEncoder: fit_transform_with_kfold를 먼저 호출해야 합니다."
            )
        return np.array(
            [self.global_mapping.get(x, self.default_value) for x in np.asarray(X)]
        )

    def _build_mapping(self, X, y):
        mapping = {}
        for x_val, y_val in zip(X, y):
            if x_val not in mapping:
                mapping[x_val] = []
            mapping[x_val].append(y_val)
        return {k: np.mean(v) for k, v in mapping.items()}


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
            p = float(ks_2samp(X_train[:, i], X_test[:, i]).pvalue)
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


class StrictTrainTestSplitter:
    """시간순 데이터 분할 (미래 데이터 유출 방지)"""

    def split(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7) -> tuple:
        n = len(X)
        split_idx = int(n * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test


class LeakageDetector:
    """데이터 누수 자동 검증기 (통계/시간/패턴 기반)"""

    def check_leakage(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray = None,
        y_test: np.ndarray = None,
    ) -> dict:
        result = {}
        # 1. 시간순 분할 여부
        if hasattr(X_train, "index") and hasattr(X_test, "index"):
            if X_train.index.max() >= X_test.index.min():
                result["time_order_violation"] = True
        # 2. 통계적 분포 차이
        train_mean = np.mean(X_train, axis=0)
        test_mean = np.mean(X_test, axis=0)
        diff = np.abs(train_mean - test_mean)
        result["mean_diff"] = diff.tolist()
        # 3. 중복 샘플 체크
        if hasattr(X_train, "tolist") and hasattr(X_test, "tolist"):
            overlap = set(map(tuple, X_train.tolist())) & set(
                map(tuple, X_test.tolist())
            )
            result["overlap_count"] = len(overlap)
        # 4. 경고
        if np.any(diff > 1.0) or result.get("overlap_count", 0) > 0:
            logger.warning("데이터 누수 가능성 감지!")
        return result


def auto_recoverable(error_type: str):
    """
    검증 함수에 자동 복구 기능을 추가하는 데코레이터.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"'{func.__name__}' 실행 중 오류 발생: {e}. 자동 복구를 시도합니다."
                )
                # self와 context를 인자에서 추출해야 함
                # 이 예시에서는 detector 객체가 첫 번째 인자라고 가정
                detector_instance = args[0]

                # context 구성 (실제 사용 시에는 더 정교한 방법 필요)
                context = {}
                if "X_train" in kwargs and "X_test" in kwargs:
                    context["X_train"] = kwargs["X_train"]
                    context["X_test"] = kwargs["X_test"]

                recovery_system = AutoRecoverySystem()
                recovered_context = recovery_system.recover(error_type, context)

                # 복구된 context로 원 함수 재시도
                kwargs.update(recovered_context)
                return func(*args, **kwargs)

        return wrapper

    return decorator
