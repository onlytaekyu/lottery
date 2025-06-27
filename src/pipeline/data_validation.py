"""
데이터 검증 시스템

로또 데이터의 품질을 검증하고 이상치를 탐지하는 모듈입니다.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import math
from datetime import datetime, timedelta
from pathlib import Path

from src.shared.types import LotteryNumber
from src.utils.error_handler_refactored import get_logger
from src.utils.unified_performance import performance_monitor
from src.utils.memory_manager import get_memory_manager
from src.utils.unified_report import save_analysis_performance_report

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """검증 결과를 담는 데이터 클래스"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    quality_score: float


@dataclass
class DataQualityReport:
    """데이터 품질 보고서"""

    total_draws: int
    valid_draws: int
    invalid_draws: int
    missing_draws: List[int]
    duplicate_draws: List[int]
    anomalous_draws: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    recommendations: List[str]


class DataValidator:
    """로또 데이터 검증 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        DataValidator 초기화

        Args:
            config: 검증 설정
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()

        # 검증 기준 설정
        self.min_number = 1
        self.max_number = 45
        self.numbers_per_draw = 6
        self.bonus_required = True

        # 이상치 탐지 임계값
        self.z_score_threshold = 2.5
        self.frequency_outlier_threshold = 3.0
        self.gap_outlier_threshold = 100  # 회차

        logger.info("DataValidator 초기화 완료")

    def validate_lottery_data(self, data: List[LotteryNumber]) -> ValidationResult:
        """
        로또 데이터 전체 검증

        Args:
            data: 검증할 로또 데이터

        Returns:
            ValidationResult: 검증 결과
        """
        with self.memory_manager.allocation_scope():
            logger.info(f"로또 데이터 검증 시작: {len(data)}개 회차")

            errors = []
            warnings = []
            anomalies = []
            statistics = {}

            # 1. 기본 구조 검증
            struct_errors, struct_warnings = self._validate_structure(data)
            errors.extend(struct_errors)
            warnings.extend(struct_warnings)

            # 2. 번호 범위 검증
            range_errors, range_warnings = self._validate_number_ranges(data)
            errors.extend(range_errors)
            warnings.extend(range_warnings)

            # 3. 중복 및 누락 검증
            dup_errors, dup_warnings = self._validate_duplicates_and_missing(data)
            errors.extend(dup_errors)
            warnings.extend(dup_warnings)

            # 4. 시계열 순서 검증
            seq_errors, seq_warnings = self._validate_sequence_order(data)
            errors.extend(seq_errors)
            warnings.extend(seq_warnings)

            # 5. 통계적 이상치 탐지
            anomalies = self.detect_anomalies(data)

            # 6. 통계 정보 수집
            statistics = self._collect_statistics(data)

            # 7. 품질 점수 계산
            quality_score = self._calculate_quality_score(
                len(errors), len(warnings), len(anomalies), len(data)
            )

            is_valid = len(errors) == 0

            logger.info(
                f"데이터 검증 완료: {'유효' if is_valid else '오류 발견'} "
                f"(오류: {len(errors)}, 경고: {len(warnings)}, 이상치: {len(anomalies)})"
            )

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                statistics=statistics,
                anomalies=anomalies,
                quality_score=quality_score,
            )

    def _validate_structure(
        self, data: List[LotteryNumber]
    ) -> Tuple[List[str], List[str]]:
        """기본 구조 검증"""
        errors = []
        warnings = []

        if not data:
            errors.append("데이터가 비어있습니다")
            return errors, warnings

        for i, draw in enumerate(data):
            # 번호 개수 검증
            if len(draw.numbers) != self.numbers_per_draw:
                errors.append(
                    f"회차 {draw.draw_no}: 번호 개수 오류 "
                    f"(예상: {self.numbers_per_draw}, 실제: {len(draw.numbers)})"
                )

            # 중복 번호 검증
            if len(set(draw.numbers)) != len(draw.numbers):
                errors.append(f"회차 {draw.draw_no}: 중복 번호 존재")

            # 보너스 번호 검증
            if self.bonus_required and not hasattr(draw, "bonus") or draw.bonus is None:
                warnings.append(f"회차 {draw.draw_no}: 보너스 번호 누락")

        return errors, warnings

    def _validate_number_ranges(
        self, data: List[LotteryNumber]
    ) -> Tuple[List[str], List[str]]:
        """번호 범위 검증"""
        errors = []
        warnings = []

        for draw in data:
            # 메인 번호 범위 검증
            for num in draw.numbers:
                if not isinstance(num, int):
                    errors.append(f"회차 {draw.draw_no}: 번호가 정수가 아님 ({num})")
                elif not (self.min_number <= num <= self.max_number):
                    errors.append(
                        f"회차 {draw.draw_no}: 번호 범위 초과 "
                        f"({num}, 유효 범위: {self.min_number}-{self.max_number})"
                    )

            # 보너스 번호 범위 검증
            if hasattr(draw, "bonus") and draw.bonus is not None:
                if not isinstance(draw.bonus, int):
                    errors.append(f"회차 {draw.draw_no}: 보너스 번호가 정수가 아님")
                elif not (self.min_number <= draw.bonus <= self.max_number):
                    errors.append(f"회차 {draw.draw_no}: 보너스 번호 범위 초과")
                elif draw.bonus in draw.numbers:
                    errors.append(
                        f"회차 {draw.draw_no}: 보너스 번호가 메인 번호와 중복"
                    )

        return errors, warnings

    def _validate_duplicates_and_missing(
        self, data: List[LotteryNumber]
    ) -> Tuple[List[str], List[str]]:
        """중복 및 누락 검증"""
        errors = []
        warnings = []

        # 회차 번호 중복 검증
        draw_numbers = [draw.draw_no for draw in data]
        duplicates = [num for num, count in Counter(draw_numbers).items() if count > 1]

        if duplicates:
            errors.append(f"중복된 회차 번호: {duplicates}")

        # 연속 회차 누락 검증
        if len(data) > 1:
            sorted_data = sorted(data, key=lambda x: x.draw_no)
            missing_draws = []

            for i in range(len(sorted_data) - 1):
                current = sorted_data[i].draw_no
                next_draw = sorted_data[i + 1].draw_no

                if next_draw - current > 1:
                    missing_range = list(range(current + 1, next_draw))
                    missing_draws.extend(missing_range)

            if missing_draws:
                warnings.append(
                    f"누락된 회차: {missing_draws[:10]}{'...' if len(missing_draws) > 10 else ''}"
                )

        return errors, warnings

    def _validate_sequence_order(
        self, data: List[LotteryNumber]
    ) -> Tuple[List[str], List[str]]:
        """시계열 순서 검증"""
        errors = []
        warnings = []

        if len(data) < 2:
            return errors, warnings

        # 날짜 순서 검증 (날짜 정보가 있는 경우)
        dated_draws = [draw for draw in data if hasattr(draw, "date") and draw.date]

        if dated_draws and len(dated_draws) > 1:
            sorted_by_date = sorted(dated_draws, key=lambda x: x.date)
            sorted_by_draw_no = sorted(dated_draws, key=lambda x: x.draw_no)

            if sorted_by_date != sorted_by_draw_no:
                warnings.append("회차 번호와 날짜 순서가 일치하지 않습니다")

        # 회차 번호 순서 검증
        draw_numbers = [draw.draw_no for draw in data]
        if draw_numbers != sorted(draw_numbers):
            warnings.append("회차 번호가 순서대로 정렬되지 않았습니다")

        return errors, warnings

    def detect_anomalies(self, data: List[LotteryNumber]) -> List[Dict[str, Any]]:
        """통계적 이상치 탐지"""
        anomalies = []

        if len(data) < 10:  # 최소 데이터 요구사항
            return anomalies

        # 1. 번호 빈도 이상치
        frequency_anomalies = self._detect_frequency_anomalies(data)
        anomalies.extend(frequency_anomalies)

        # 2. 합계 이상치
        sum_anomalies = self._detect_sum_anomalies(data)
        anomalies.extend(sum_anomalies)

        # 3. 연속 번호 이상치
        consecutive_anomalies = self._detect_consecutive_anomalies(data)
        anomalies.extend(consecutive_anomalies)

        # 4. 홀짝 분포 이상치
        odd_even_anomalies = self._detect_odd_even_anomalies(data)
        anomalies.extend(odd_even_anomalies)

        # 5. 범위 분포 이상치
        range_anomalies = self._detect_range_anomalies(data)
        anomalies.extend(range_anomalies)

        return anomalies

    def _detect_frequency_anomalies(
        self, data: List[LotteryNumber]
    ) -> List[Dict[str, Any]]:
        """번호 빈도 이상치 탐지"""
        anomalies = []

        # 번호별 출현 빈도 계산
        frequency = Counter()
        for draw in data:
            frequency.update(draw.numbers)

        # 빈도 통계
        frequencies = list(frequency.values())
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)

        if std_freq == 0:
            return anomalies

        # Z-score 기반 이상치 탐지
        for number, freq in frequency.items():
            z_score = abs(freq - mean_freq) / std_freq

            if z_score > self.frequency_outlier_threshold:
                anomalies.append(
                    {
                        "type": "frequency_outlier",
                        "number": number,
                        "frequency": freq,
                        "z_score": z_score,
                        "description": f"번호 {number}의 출현 빈도({freq})가 비정상적",
                    }
                )

        return anomalies

    def _detect_sum_anomalies(self, data: List[LotteryNumber]) -> List[Dict[str, Any]]:
        """합계 이상치 탐지"""
        anomalies = []

        sums = [sum(draw.numbers) for draw in data]
        mean_sum = np.mean(sums)
        std_sum = np.std(sums)

        if std_sum == 0:
            return anomalies

        for i, draw in enumerate(data):
            draw_sum = sum(draw.numbers)
            z_score = abs(draw_sum - mean_sum) / std_sum

            if z_score > self.z_score_threshold:
                anomalies.append(
                    {
                        "type": "sum_outlier",
                        "draw_no": draw.draw_no,
                        "sum": draw_sum,
                        "z_score": z_score,
                        "description": f"회차 {draw.draw_no}의 번호 합계({draw_sum})가 비정상적",
                    }
                )

        return anomalies

    def _detect_consecutive_anomalies(
        self, data: List[LotteryNumber]
    ) -> List[Dict[str, Any]]:
        """연속 번호 이상치 탐지"""
        anomalies = []

        for draw in data:
            sorted_numbers = sorted(draw.numbers)
            consecutive_count = 0
            max_consecutive = 0

            for i in range(len(sorted_numbers) - 1):
                if sorted_numbers[i + 1] - sorted_numbers[i] == 1:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count + 1)
                else:
                    consecutive_count = 0

            # 4개 이상 연속 번호는 이상치로 간주
            if max_consecutive >= 4:
                anomalies.append(
                    {
                        "type": "consecutive_outlier",
                        "draw_no": draw.draw_no,
                        "consecutive_count": max_consecutive,
                        "numbers": sorted_numbers,
                        "description": f"회차 {draw.draw_no}에 {max_consecutive}개 연속 번호 존재",
                    }
                )

        return anomalies

    def _detect_odd_even_anomalies(
        self, data: List[LotteryNumber]
    ) -> List[Dict[str, Any]]:
        """홀짝 분포 이상치 탐지"""
        anomalies = []

        for draw in data:
            odd_count = sum(1 for n in draw.numbers if n % 2 == 1)
            even_count = 6 - odd_count

            # 극단적인 홀짝 분포 (0:6 또는 6:0)
            if odd_count == 0 or even_count == 0:
                anomalies.append(
                    {
                        "type": "odd_even_outlier",
                        "draw_no": draw.draw_no,
                        "odd_count": odd_count,
                        "even_count": even_count,
                        "description": f"회차 {draw.draw_no}의 홀짝 분포가 극단적 ({odd_count}:{even_count})",
                    }
                )

        return anomalies

    def _detect_range_anomalies(
        self, data: List[LotteryNumber]
    ) -> List[Dict[str, Any]]:
        """범위 분포 이상치 탐지"""
        anomalies = []

        for draw in data:
            # 번호 범위 (최대값 - 최소값)
            number_range = max(draw.numbers) - min(draw.numbers)

            # 범위가 너무 작은 경우 (모든 번호가 10 이내)
            if number_range <= 10:
                anomalies.append(
                    {
                        "type": "range_outlier",
                        "draw_no": draw.draw_no,
                        "range": number_range,
                        "numbers": sorted(draw.numbers),
                        "description": f"회차 {draw.draw_no}의 번호 범위({number_range})가 너무 작음",
                    }
                )

        return anomalies

    def _collect_statistics(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """데이터 통계 정보 수집"""
        if not data:
            return {}

        # 기본 통계
        total_draws = len(data)
        draw_numbers = [draw.draw_no for draw in data]

        # 번호 빈도 통계
        all_numbers = []
        for draw in data:
            all_numbers.extend(draw.numbers)

        frequency = Counter(all_numbers)

        # 합계 통계
        sums = [sum(draw.numbers) for draw in data]

        # 홀짝 분포 통계
        odd_even_distributions = []
        for draw in data:
            odd_count = sum(1 for n in draw.numbers if n % 2 == 1)
            odd_even_distributions.append((odd_count, 6 - odd_count))

        return {
            "total_draws": total_draws,
            "draw_range": {"min": min(draw_numbers), "max": max(draw_numbers)},
            "number_frequency": {
                "most_common": frequency.most_common(5),
                "least_common": frequency.most_common()[-5:],
                "mean_frequency": np.mean(list(frequency.values())),
                "std_frequency": np.std(list(frequency.values())),
            },
            "sum_statistics": {
                "mean": np.mean(sums),
                "std": np.std(sums),
                "min": min(sums),
                "max": max(sums),
                "median": np.median(sums),
            },
            "odd_even_distribution": {
                "most_common_pattern": Counter(odd_even_distributions).most_common(1)[0]
            },
        }

    def _calculate_quality_score(
        self, error_count: int, warning_count: int, anomaly_count: int, total_draws: int
    ) -> float:
        """데이터 품질 점수 계산 (0-100)"""
        if total_draws == 0:
            return 0.0

        # 가중치 설정
        error_weight = 10.0
        warning_weight = 2.0
        anomaly_weight = 1.0

        # 점수 계산
        penalty = (
            error_count * error_weight
            + warning_count * warning_weight
            + anomaly_count * anomaly_weight
        )

        # 정규화 (총 회차 수 대비)
        normalized_penalty = penalty / total_draws

        # 품질 점수 (최대 100점에서 페널티 차감)
        quality_score = max(0.0, 100.0 - normalized_penalty * 10)

        return round(quality_score, 2)

    def generate_quality_report(self, data: List[LotteryNumber]) -> DataQualityReport:
        """데이터 품질 보고서 생성"""
        validation_result = self.validate_lottery_data(data)

        # 유효하지 않은 회차 찾기
        invalid_draws = []
        for error in validation_result.errors:
            if "회차" in error:
                try:
                    draw_no = int(error.split("회차 ")[1].split(":")[0])
                    invalid_draws.append(draw_no)
                except (IndexError, ValueError):
                    pass

        # 중복 회차 찾기
        draw_numbers = [draw.draw_no for draw in data]
        duplicates = [num for num, count in Counter(draw_numbers).items() if count > 1]

        # 누락 회차 찾기
        missing_draws = []
        if len(data) > 1:
            sorted_data = sorted(data, key=lambda x: x.draw_no)
            for i in range(len(sorted_data) - 1):
                current = sorted_data[i].draw_no
                next_draw = sorted_data[i + 1].draw_no
                if next_draw - current > 1:
                    missing_range = list(range(current + 1, next_draw))
                    missing_draws.extend(missing_range)

        # 품질 메트릭 계산
        quality_metrics = {
            "completeness": (
                (len(data) - len(missing_draws)) / (len(data) + len(missing_draws))
                if (len(data) + len(missing_draws)) > 0
                else 1.0
            ),
            "accuracy": (
                1.0 - (len(validation_result.errors) / len(data))
                if len(data) > 0
                else 0.0
            ),
            "consistency": (
                1.0 - (len(duplicates) / len(data)) if len(data) > 0 else 1.0
            ),
            "anomaly_rate": (
                len(validation_result.anomalies) / len(data) if len(data) > 0 else 0.0
            ),
        }

        # 개선 권장사항
        recommendations = []
        if validation_result.errors:
            recommendations.append("데이터 오류를 수정하세요")
        if missing_draws:
            recommendations.append("누락된 회차 데이터를 보완하세요")
        if duplicates:
            recommendations.append("중복된 회차 데이터를 제거하세요")
        if validation_result.anomalies:
            recommendations.append("이상치 데이터를 검토하세요")
        if validation_result.quality_score < 90:
            recommendations.append("전반적인 데이터 품질 개선이 필요합니다")

        return DataQualityReport(
            total_draws=len(data),
            valid_draws=len(data) - len(invalid_draws),
            invalid_draws=len(invalid_draws),
            missing_draws=missing_draws,
            duplicate_draws=duplicates,
            anomalous_draws=[
                {
                    "draw_no": anomaly.get("draw_no"),
                    "type": anomaly.get("type"),
                    "description": anomaly.get("description"),
                }
                for anomaly in validation_result.anomalies
                if "draw_no" in anomaly
            ],
            quality_metrics=quality_metrics,
            recommendations=recommendations,
        )

    def save_quality_report(
        self, report: DataQualityReport, output_path: Optional[Path] = None
    ) -> Path:
        """품질 보고서를 파일로 저장"""
        if output_path is None:
            output_path = Path(
                "data/result/performance_reports/data_quality_report.json"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_draws": report.total_draws,
                "valid_draws": report.valid_draws,
                "invalid_draws": report.invalid_draws,
                "quality_score": sum(report.quality_metrics.values())
                / len(report.quality_metrics)
                * 100,
            },
            "details": {
                "missing_draws": report.missing_draws[:20],  # 최대 20개만 저장
                "duplicate_draws": report.duplicate_draws,
                "anomalous_draws": report.anomalous_draws[:20],  # 최대 20개만 저장
                "quality_metrics": report.quality_metrics,
                "recommendations": report.recommendations,
            },
        }

        save_analysis_performance_report(
            profiler=None,  # 프로파일러 없음
            performance_tracker=None,  # 성능 추적기 없음
            config={},  # 빈 설정
            module_name="data_validation",
            data_metrics=report_data,
        )
        logger.info(f"데이터 품질 보고서 저장: {output_path}")

        return output_path
