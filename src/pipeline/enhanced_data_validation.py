"""
Enhanced Data Validation System
강화된 데이터 검증 시스템 - 데이터 품질 및 모델 입력 검증
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

from ..utils.unified_logging import get_logger
from ..utils.memory_manager import MemoryManager
from ..shared.types import LotteryNumber

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    """데이터 품질 보고서"""

    nan_ratio: float
    inf_ratio: float
    zero_ratio: float
    scale_imbalance: float
    outlier_ratio: float
    class_imbalance_ratio: Optional[float] = None
    feature_correlation_issues: List[str] = None
    data_drift_score: float = 0.0
    quality_score: float = 0.0
    recommendations: List[str] = None


class EnhancedDataValidator:
    """강화된 데이터 검증기"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.memory_manager = MemoryManager()

        # 검증 임계값 설정
        self.thresholds = {
            "nan_ratio_max": 0.05,  # NaN 비율 5% 이하
            "inf_ratio_max": 0.001,  # Inf 비율 0.1% 이하
            "scale_imbalance_max": 100.0,  # 스케일 불균형 100배 이하
            "outlier_ratio_max": 0.1,  # 이상값 비율 10% 이하
            "class_imbalance_max": 10.0,  # 클래스 불균형 10배 이하
            "correlation_threshold": 0.95,  # 상관관계 임계값
            "drift_threshold": 0.3,  # 데이터 드리프트 임계값
        }

        # 스케일러 초기화
        self.scalers = {"standard": StandardScaler(), "robust": RobustScaler()}

        # 결측값 처리기 초기화
        self.imputers = {
            "simple": SimpleImputer(strategy="median"),
            "knn": KNNImputer(n_neighbors=5),
        }

    def enhanced_data_validation(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> DataQualityReport:
        """
        강화된 데이터 검증 수행

        Args:
            X: 입력 데이터
            y: 타겟 데이터 (선택사항)
            feature_names: 피처 이름 목록 (선택사항)

        Returns:
            DataQualityReport: 데이터 품질 보고서
        """
        try:
            self.logger.info("강화된 데이터 검증 시작")

            with self.memory_manager.get_context("data_validation"):
                # 1. 기본 데이터 품질 체크
                nan_ratio = self._check_nan_values(X)
                inf_ratio = self._check_inf_values(X)
                zero_ratio = self._check_zero_values(X)

                # 2. 피처 스케일 불균형 체크
                scale_imbalance = self._check_scale_imbalance(X)

                # 3. 이상값 검출
                outlier_ratio = self._detect_outliers(X)

                # 4. 타겟 분포 분석
                class_imbalance_ratio = None
                if y is not None:
                    class_imbalance_ratio = self._analyze_target_distribution(y)

                # 5. 피처 상관관계 분석
                correlation_issues = self._check_feature_correlation(X, feature_names)

                # 6. 데이터 드리프트 검출
                drift_score = self._detect_data_drift(X)

                # 7. 전체 품질 점수 계산
                quality_score = self._calculate_quality_score(
                    nan_ratio,
                    inf_ratio,
                    scale_imbalance,
                    outlier_ratio,
                    class_imbalance_ratio,
                    len(correlation_issues),
                    drift_score,
                )

                # 8. 개선 권장사항 생성
                recommendations = self._generate_recommendations(
                    nan_ratio,
                    inf_ratio,
                    scale_imbalance,
                    outlier_ratio,
                    class_imbalance_ratio,
                    correlation_issues,
                    drift_score,
                )

                # 보고서 생성
                report = DataQualityReport(
                    nan_ratio=nan_ratio,
                    inf_ratio=inf_ratio,
                    zero_ratio=zero_ratio,
                    scale_imbalance=scale_imbalance,
                    outlier_ratio=outlier_ratio,
                    class_imbalance_ratio=class_imbalance_ratio,
                    feature_correlation_issues=correlation_issues,
                    data_drift_score=drift_score,
                    quality_score=quality_score,
                    recommendations=recommendations,
                )

                self.logger.info(f"데이터 검증 완료 - 품질 점수: {quality_score:.3f}")
                return report

        except Exception as e:
            self.logger.error(f"데이터 검증 중 오류: {e}")
            raise

    def _check_nan_values(self, X: np.ndarray) -> float:
        """NaN 값 비율 체크"""
        if X.size == 0:
            return 0.0
        nan_count = np.isnan(X).sum()
        return nan_count / X.size

    def _check_inf_values(self, X: np.ndarray) -> float:
        """Inf 값 비율 체크"""
        if X.size == 0:
            return 0.0
        inf_count = np.isinf(X).sum()
        return inf_count / X.size

    def _check_zero_values(self, X: np.ndarray) -> float:
        """0 값 비율 체크"""
        if X.size == 0:
            return 0.0
        zero_count = (X == 0).sum()
        return zero_count / X.size

    def _check_scale_imbalance(self, X: np.ndarray) -> float:
        """피처 스케일 불균형 체크"""
        if X.shape[1] < 2:
            return 1.0

        # 각 피처의 표준편차 계산
        feature_stds = np.std(X, axis=0)

        # 0인 표준편차 제외
        non_zero_stds = feature_stds[feature_stds > 1e-10]

        if len(non_zero_stds) < 2:
            return 1.0

        # 최대/최소 비율 계산
        scale_imbalance = np.max(non_zero_stds) / np.min(non_zero_stds)
        return scale_imbalance

    def _detect_outliers(self, X: np.ndarray) -> float:
        """이상값 검출 (IQR 방법)"""
        outlier_count = 0
        total_count = X.size

        for col in range(X.shape[1]):
            col_data = X[:, col]

            # IQR 계산
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1

            # 이상값 범위 설정
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 이상값 카운트
            outliers = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_count += outliers.sum()

        return outlier_count / total_count

    def _analyze_target_distribution(self, y: np.ndarray) -> float:
        """타겟 분포 분석 (클래스 불균형 체크)"""
        if y is None or len(y) == 0:
            return 1.0

        # 연속형 타겟인 경우 구간화
        if len(np.unique(y)) > 10:
            y_binned = pd.cut(y, bins=5, labels=False)
            unique_values, counts = np.unique(y_binned, return_counts=True)
        else:
            unique_values, counts = np.unique(y, return_counts=True)

        if len(counts) < 2:
            return 1.0

        # 클래스 불균형 비율 계산
        class_distribution = counts / len(y)
        imbalance_ratio = np.max(class_distribution) / np.min(class_distribution)

        return imbalance_ratio

    def _check_feature_correlation(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> List[str]:
        """피처 간 상관관계 분석"""
        if X.shape[1] < 2:
            return []

        correlation_issues = []

        try:
            # 상관관계 매트릭스 계산
            corr_matrix = np.corrcoef(X.T)

            # 대각선 제외하고 높은 상관관계 찾기
            n_features = X.shape[1]
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if (
                        abs(corr_matrix[i, j])
                        > self.thresholds["correlation_threshold"]
                    ):
                        feature_i = (
                            feature_names[i] if feature_names else f"feature_{i}"
                        )
                        feature_j = (
                            feature_names[j] if feature_names else f"feature_{j}"
                        )
                        correlation_issues.append(
                            f"{feature_i} - {feature_j}: {corr_matrix[i, j]:.3f}"
                        )

        except Exception as e:
            self.logger.warning(f"상관관계 분석 중 오류: {e}")

        return correlation_issues

    def _detect_data_drift(self, X: np.ndarray) -> float:
        """데이터 드리프트 검출 (통계적 방법)"""
        if X.shape[0] < 100:  # 샘플 수가 적으면 드리프트 검출 불가
            return 0.0

        try:
            # 데이터를 두 그룹으로 나누어 분포 비교
            mid_point = X.shape[0] // 2
            X_first_half = X[:mid_point]
            X_second_half = X[mid_point:]

            drift_scores = []

            for col in range(X.shape[1]):
                # Kolmogorov-Smirnov 테스트
                ks_stat, p_value = stats.ks_2samp(
                    X_first_half[:, col], X_second_half[:, col]
                )
                drift_scores.append(ks_stat)

            # 평균 드리프트 점수 반환
            return np.mean(drift_scores)

        except Exception as e:
            self.logger.warning(f"데이터 드리프트 검출 중 오류: {e}")
            return 0.0

    def _calculate_quality_score(
        self,
        nan_ratio: float,
        inf_ratio: float,
        scale_imbalance: float,
        outlier_ratio: float,
        class_imbalance_ratio: Optional[float],
        correlation_issues_count: int,
        drift_score: float,
    ) -> float:
        """전체 데이터 품질 점수 계산 (0-1 스케일)"""

        # 각 메트릭별 점수 계산 (1이 최고, 0이 최악)
        nan_score = max(0, 1 - nan_ratio / self.thresholds["nan_ratio_max"])
        inf_score = max(0, 1 - inf_ratio / self.thresholds["inf_ratio_max"])

        scale_score = max(
            0, 1 - min(scale_imbalance / self.thresholds["scale_imbalance_max"], 1)
        )
        outlier_score = max(0, 1 - outlier_ratio / self.thresholds["outlier_ratio_max"])

        class_score = 1.0
        if class_imbalance_ratio is not None:
            class_score = max(
                0,
                1
                - min(
                    class_imbalance_ratio / self.thresholds["class_imbalance_max"], 1
                ),
            )

        correlation_score = max(0, 1 - min(correlation_issues_count / 10, 1))
        drift_score_normalized = max(
            0, 1 - drift_score / self.thresholds["drift_threshold"]
        )

        # 가중 평균 계산
        weights = [0.2, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1]  # 각 메트릭 가중치
        scores = [
            nan_score,
            inf_score,
            scale_score,
            outlier_score,
            class_score,
            correlation_score,
            drift_score_normalized,
        ]

        quality_score = np.average(scores, weights=weights)
        return quality_score

    def _generate_recommendations(
        self,
        nan_ratio: float,
        inf_ratio: float,
        scale_imbalance: float,
        outlier_ratio: float,
        class_imbalance_ratio: Optional[float],
        correlation_issues: List[str],
        drift_score: float,
    ) -> List[str]:
        """데이터 품질 개선 권장사항 생성"""

        recommendations = []

        # NaN 값 처리 권장사항
        if nan_ratio > self.thresholds["nan_ratio_max"]:
            recommendations.append(
                f"NaN 비율이 {nan_ratio:.3f}로 높습니다. KNN Imputer 또는 고급 결측값 처리 기법 사용을 권장합니다."
            )

        # Inf 값 처리 권장사항
        if inf_ratio > self.thresholds["inf_ratio_max"]:
            recommendations.append(
                f"Inf 값이 {inf_ratio:.3f} 비율로 존재합니다. 수치 안정성을 위해 클리핑 또는 변환이 필요합니다."
            )

        # 스케일 불균형 처리 권장사항
        if scale_imbalance > self.thresholds["scale_imbalance_max"]:
            recommendations.append(
                f"피처 스케일 불균형이 {scale_imbalance:.1f}배입니다. RobustScaler 또는 StandardScaler 사용을 권장합니다."
            )

        # 이상값 처리 권장사항
        if outlier_ratio > self.thresholds["outlier_ratio_max"]:
            recommendations.append(
                f"이상값 비율이 {outlier_ratio:.3f}로 높습니다. 이상값 제거 또는 변환을 고려하세요."
            )

        # 클래스 불균형 처리 권장사항
        if (
            class_imbalance_ratio
            and class_imbalance_ratio > self.thresholds["class_imbalance_max"]
        ):
            recommendations.append(
                f"클래스 불균형이 {class_imbalance_ratio:.1f}배입니다. SMOTE, 가중치 조정 또는 언더샘플링을 고려하세요."
            )

        # 상관관계 문제 권장사항
        if len(correlation_issues) > 5:
            recommendations.append(
                f"{len(correlation_issues)}개의 높은 상관관계가 발견되었습니다. 차원 축소 또는 피처 선택을 고려하세요."
            )

        # 데이터 드리프트 권장사항
        if drift_score > self.thresholds["drift_threshold"]:
            recommendations.append(
                f"데이터 드리프트 점수가 {drift_score:.3f}입니다. 데이터 분포 변화를 모니터링하고 모델 재학습을 고려하세요."
            )

        if not recommendations:
            recommendations.append(
                "데이터 품질이 양호합니다. 추가 전처리 없이 모델 학습을 진행할 수 있습니다."
            )

        return recommendations

    def auto_fix_data_quality(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """자동 데이터 품질 개선"""

        self.logger.info("자동 데이터 품질 개선 시작")
        X_fixed = X.copy()
        y_fixed = y.copy() if y is not None else None

        try:
            # 1. Inf 값 처리
            X_fixed = np.where(np.isinf(X_fixed), np.nan, X_fixed)

            # 2. NaN 값 처리 (KNN Imputer 사용)
            if np.isnan(X_fixed).any():
                self.logger.info("NaN 값 처리 중...")
                X_fixed = self.imputers["knn"].fit_transform(X_fixed)

            # 3. 이상값 처리 (IQR 기반 클리핑)
            for col in range(X_fixed.shape[1]):
                Q1 = np.percentile(X_fixed[:, col], 25)
                Q3 = np.percentile(X_fixed[:, col], 75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                X_fixed[:, col] = np.clip(X_fixed[:, col], lower_bound, upper_bound)

            # 4. 스케일 정규화
            X_fixed = self.scalers["robust"].fit_transform(X_fixed)

            self.logger.info("자동 데이터 품질 개선 완료")
            return X_fixed, y_fixed

        except Exception as e:
            self.logger.error(f"자동 데이터 품질 개선 중 오류: {e}")
            return X, y

    def print_quality_report(self, report: DataQualityReport):
        """데이터 품질 보고서 출력"""

        print("=" * 60)
        print("📊 데이터 품질 보고서")
        print("=" * 60)

        print(f"🎯 전체 품질 점수: {report.quality_score:.3f}/1.000")
        print()

        print("📈 세부 메트릭:")
        print(f"  • NaN 비율: {report.nan_ratio:.4f} ({report.nan_ratio*100:.2f}%)")
        print(f"  • Inf 비율: {report.inf_ratio:.4f} ({report.inf_ratio*100:.2f}%)")
        print(f"  • Zero 비율: {report.zero_ratio:.4f} ({report.zero_ratio*100:.2f}%)")
        print(f"  • 스케일 불균형: {report.scale_imbalance:.2f}배")
        print(
            f"  • 이상값 비율: {report.outlier_ratio:.4f} ({report.outlier_ratio*100:.2f}%)"
        )

        if report.class_imbalance_ratio:
            print(f"  • 클래스 불균형: {report.class_imbalance_ratio:.2f}배")

        print(f"  • 상관관계 문제: {len(report.feature_correlation_issues)}개")
        print(f"  • 데이터 드리프트: {report.data_drift_score:.3f}")
        print()

        if report.recommendations:
            print("💡 개선 권장사항:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")

        print("=" * 60)
