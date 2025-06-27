"""
통합 데이터 검증 시스템

벡터 무결성, 패턴 검증, 파일 검증 등의 중복 코드를 제거하고 통합합니다.
"""

import os
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from .unified_logging import get_logger
from .unified_performance import performance_monitor

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """검증 레벨"""

    BASIC = "basic"  # 기본 검증
    STRICT = "strict"  # 엄격한 검증
    COMPLETE = "complete"  # 완전한 검증


@dataclass
class ValidationResult:
    """검증 결과"""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str):
        """에러 추가"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """경고 추가"""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult"):
        """다른 검증 결과와 병합"""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)

    def get_summary(self) -> str:
        """검증 결과 요약"""
        status = "통과" if self.is_valid else "실패"
        return f"검증 {status} (에러: {len(self.errors)}, 경고: {len(self.warnings)})"


class BaseValidator(ABC):
    """기본 검증기 인터페이스"""

    def __init__(self, level: ValidationLevel = ValidationLevel.BASIC):
        self.level = level
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """데이터 검증"""
        pass

    def get_name(self) -> str:
        """검증기 이름 반환"""
        return self.__class__.__name__


class VectorValidator(BaseValidator):
    """벡터 데이터 검증기"""

    def __init__(self, level: ValidationLevel = ValidationLevel.BASIC):
        super().__init__(level)
        self.min_dimensions = 70
        self.max_dimensions = 1000
        self.allowed_dtypes = [np.float32, np.float64, np.int32, np.int64]

    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """벡터 데이터 검증"""
        result = ValidationResult(is_valid=True)

        with performance_monitor(f"vector_validation_{self.level.value}"):
            try:
                # 기본 타입 검증
                if not isinstance(data, np.ndarray):
                    result.add_error(
                        f"벡터 데이터가 numpy 배열이 아닙니다: {type(data)}"
                    )
                    return result

                # 차원 검증
                if data.ndim != 2:
                    result.add_error(f"벡터는 2차원이어야 합니다: {data.ndim}차원")
                    return result

                rows, cols = data.shape
                result.metadata.update(
                    {
                        "shape": data.shape,
                        "dtype": str(data.dtype),
                        "memory_usage_mb": data.nbytes / (1024 * 1024),
                    }
                )

                # 최소 차원 검증
                if cols < self.min_dimensions:
                    result.add_error(
                        f"벡터 차원이 최소 요구사항보다 작습니다: {cols} < {self.min_dimensions}"
                    )

                # 최대 차원 검증 (경고)
                if cols > self.max_dimensions:
                    result.add_warning(
                        f"벡터 차원이 매우 큽니다: {cols} > {self.max_dimensions}"
                    )

                # 데이터 타입 검증
                if data.dtype not in self.allowed_dtypes:
                    result.add_warning(f"권장되지 않는 데이터 타입: {data.dtype}")

                # 엄격한 검증
                if self.level in [ValidationLevel.STRICT, ValidationLevel.COMPLETE]:
                    self._strict_validation(data, result)

                # 완전한 검증
                if self.level == ValidationLevel.COMPLETE:
                    self._complete_validation(data, result)

            except Exception as e:
                result.add_error(f"벡터 검증 중 예외 발생: {str(e)}")

        return result

    def _strict_validation(self, data: np.ndarray, result: ValidationResult):
        """엄격한 검증"""
        # NaN/Inf 검증
        if np.isnan(data).any():
            result.add_error("벡터에 NaN 값이 포함되어 있습니다")

        if np.isinf(data).any():
            result.add_error("벡터에 무한대 값이 포함되어 있습니다")

        # 제로 벡터 검증
        zero_rows = np.all(data == 0, axis=1)
        if zero_rows.any():
            zero_count = np.sum(zero_rows)
            result.add_warning(f"제로 벡터가 {zero_count}개 발견되었습니다")

    def _complete_validation(self, data: np.ndarray, result: ValidationResult):
        """완전한 검증"""
        # 통계적 이상치 검증
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        # 분산이 너무 작은 특성 검증
        low_variance_features = np.sum(stds < 1e-6)
        if low_variance_features > 0:
            result.add_warning(
                f"분산이 매우 작은 특성이 {low_variance_features}개 있습니다"
            )

        # 값 범위 검증
        data_min, data_max = np.min(data), np.max(data)
        if data_max - data_min > 1e6:
            result.add_warning(
                f"값의 범위가 매우 큽니다: [{data_min:.2e}, {data_max:.2e}]"
            )

        result.metadata.update(
            {
                "min_value": float(data_min),
                "max_value": float(data_max),
                "mean_value": float(np.mean(data)),
                "std_value": float(np.std(data)),
                "low_variance_features": int(low_variance_features),
            }
        )


class UnifiedValidationManager:
    """통합 검증 관리자"""

    def __init__(self):
        self.logger = get_logger("validation")
        self.validators: Dict[str, BaseValidator] = {}
        self._register_default_validators()

    def _register_default_validators(self):
        """기본 검증기 등록"""
        self.register_validator("vector", VectorValidator())

    def register_validator(self, name: str, validator: BaseValidator):
        """검증기 등록"""
        self.validators[name] = validator
        self.logger.debug(f"검증기 등록: {name}")

    def validate(self, validator_name: str, data: Any, **kwargs) -> ValidationResult:
        """데이터 검증"""
        if validator_name not in self.validators:
            result = ValidationResult(is_valid=False)
            result.add_error(f"알 수 없는 검증기: {validator_name}")
            return result

        validator = self.validators[validator_name]

        try:
            result = validator.validate(data, **kwargs)

            # 로깅
            if result.is_valid:
                self.logger.debug(f"검증 성공: {validator_name}")
            else:
                self.logger.error(
                    f"검증 실패: {validator_name} - {result.get_summary()}"
                )
                for error in result.errors:
                    self.logger.error(f"  에러: {error}")

            for warning in result.warnings:
                self.logger.warning(f"  경고: {warning}")

            return result

        except Exception as e:
            self.logger.error(f"검증 중 예외 발생: {validator_name} - {str(e)}")
            result = ValidationResult(is_valid=False)
            result.add_error(f"검증 중 예외 발생: {str(e)}")
            return result


# 편의 함수들
_global_validation_manager = None


def get_validation_manager() -> UnifiedValidationManager:
    """전역 검증 관리자 반환"""
    global _global_validation_manager
    if _global_validation_manager is None:
        _global_validation_manager = UnifiedValidationManager()
    return _global_validation_manager


def validate_vector(
    vector_data: np.ndarray, level: ValidationLevel = ValidationLevel.BASIC
) -> ValidationResult:
    """벡터 검증 (편의 함수)"""
    validator = VectorValidator(level)
    return validator.validate(vector_data)


def strict_validate_and_fail_fast(
    validator_name: str, data: Any, context: str = "", **kwargs
):
    """
    엄격한 검증 후 실패 시 즉시 예외 발생

    Args:
        validator_name: 검증기 이름
        data: 검증할 데이터
        context: 컨텍스트 정보
        **kwargs: 추가 인자

    Raises:
        RuntimeError: 검증 실패 시
    """
    manager = get_validation_manager()
    result = manager.validate(validator_name, data, **kwargs)

    if not result.is_valid:
        error_msg = f"데이터 검증 실패"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {result.get_summary()}"

        logger.error(error_msg)
        for error in result.errors:
            logger.error(f"  - {error}")

        raise RuntimeError(error_msg)
