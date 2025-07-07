#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
설정 검증 시스템

각 설정 파일별로 타입, 범위, 필수 여부를 검증하는 시스템입니다.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml
import json

from .unified_logging import get_logger

logger = get_logger(__name__)


class ConfigType(Enum):
    """설정 타입 열거형"""

    MAIN = "main"
    REALISTIC_LOTTERY = "realistic_lottery"
    PATTERN_ANALYSIS = "pattern_analysis"


@dataclass
class ValidationRule:
    """설정 검증 규칙"""

    key_path: str  # 설정 키 경로 (예: "analysis.micro_bias.bias_threshold")
    required: bool = False  # 필수 여부
    data_type: type = str  # 데이터 타입
    min_value: Optional[Union[int, float]] = None  # 최소값
    max_value: Optional[Union[int, float]] = None  # 최대값
    allowed_values: Optional[List[Any]] = None  # 허용값 목록
    description: str = ""  # 설명


class ConfigValidator:
    """설정 검증기"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.validation_rules = self._initialize_validation_rules()

    def _initialize_validation_rules(self) -> Dict[ConfigType, List[ValidationRule]]:
        """검증 규칙 초기화"""
        rules = {
            ConfigType.MAIN: self._get_main_config_rules(),
            ConfigType.REALISTIC_LOTTERY: self._get_realistic_lottery_rules(),
            ConfigType.PATTERN_ANALYSIS: self._get_pattern_analysis_rules(),
        }
        return rules

    def _get_main_config_rules(self) -> List[ValidationRule]:
        """메인 설정 파일 검증 규칙"""
        return [
            # 데이터 설정
            ValidationRule(
                key_path="data.max_number",
                required=True,
                data_type=int,
                min_value=1,
                max_value=100,
                description="최대 번호 (로또: 45)",
            ),
            ValidationRule(
                key_path="data.min_number",
                required=True,
                data_type=int,
                min_value=1,
                max_value=45,
                description="최소 번호 (로또: 1)",
            ),
            ValidationRule(
                key_path="data.sequence_length",
                required=True,
                data_type=int,
                min_value=6,
                max_value=6,
                description="시퀀스 길이 (로또: 6)",
            ),
            # 실행 설정
            ValidationRule(
                key_path="execution.max_threads",
                required=True,
                data_type=int,
                min_value=1,
                max_value=32,
                description="최대 스레드 수",
            ),
            ValidationRule(
                key_path="execution.use_gpu",
                required=True,
                data_type=bool,
                description="GPU 사용 여부",
            ),
            # 최적화 설정
            ValidationRule(
                key_path="optimization.memory.max_memory_usage",
                required=True,
                data_type=float,
                min_value=0.1,
                max_value=0.95,
                description="최대 메모리 사용률",
            ),
            ValidationRule(
                key_path="optimization.process_pool.max_workers",
                required=True,
                data_type=int,
                min_value=1,
                max_value=16,
                description="프로세스 풀 최대 워커 수",
            ),
            # 분석 설정
            ValidationRule(
                key_path="analysis.enable_position_bias_analysis",
                required=True,
                data_type=bool,
                description="위치 편향 분석 활성화",
            ),
            ValidationRule(
                key_path="analysis.enable_micro_bias_analysis",
                required=True,
                data_type=bool,
                description="미세 편향 분석 활성화",
            ),
        ]

    def _get_realistic_lottery_rules(self) -> List[ValidationRule]:
        """현실적 로또 설정 검증 규칙"""
        return [
            # 목표 개선률 (실제 파일 구조에 맞춤)
            ValidationRule(
                key_path="realistic_lottery.target_improvements.5th_prize.base_probability",
                required=True,
                data_type=float,
                min_value=0.0,
                max_value=1.0,
                description="5등 기본 확률",
            ),
            ValidationRule(
                key_path="realistic_lottery.target_improvements.5th_prize.target_probability",
                required=True,
                data_type=float,
                min_value=0.0,
                max_value=1.0,
                description="5등 목표 확률",
            ),
            ValidationRule(
                key_path="realistic_lottery.target_improvements.4th_prize.base_probability",
                required=True,
                data_type=float,
                min_value=0.0,
                max_value=1.0,
                description="4등 기본 확률",
            ),
            ValidationRule(
                key_path="realistic_lottery.target_improvements.3rd_prize.base_probability",
                required=True,
                data_type=float,
                min_value=0.0,
                max_value=1.0,
                description="3등 기본 확률",
            ),
            # 포트폴리오 배분
            ValidationRule(
                key_path="realistic_lottery.portfolio_allocation.conservative",
                required=True,
                data_type=float,
                min_value=0.0,
                max_value=1.0,
                description="보수적 전략 배분",
            ),
            ValidationRule(
                key_path="realistic_lottery.portfolio_allocation.aggressive",
                required=True,
                data_type=float,
                min_value=0.0,
                max_value=1.0,
                description="공격적 전략 배분",
            ),
            ValidationRule(
                key_path="realistic_lottery.portfolio_allocation.balanced",
                required=True,
                data_type=float,
                min_value=0.0,
                max_value=1.0,
                description="균형 전략 배분",
            ),
            # 리스크 관리
            ValidationRule(
                key_path="realistic_lottery.risk_management.kelly_fraction",
                required=True,
                data_type=float,
                min_value=0.01,
                max_value=1.0,
                description="Kelly 비율",
            ),
            ValidationRule(
                key_path="realistic_lottery.risk_management.max_consecutive_losses",
                required=True,
                data_type=int,
                min_value=1,
                max_value=50,
                description="최대 연속 손실 횟수",
            ),
            # 성과 목표
            ValidationRule(
                key_path="realistic_lottery.performance_goals.hit_rate_5th",
                required=True,
                data_type=float,
                min_value=0.0,
                max_value=1.0,
                description="5등 목표 적중률",
            ),
            ValidationRule(
                key_path="realistic_lottery.performance_goals.total_loss_target",
                required=True,
                data_type=float,
                min_value=0.0,
                max_value=1.0,
                description="총 손실 목표",
            ),
        ]

    def _get_pattern_analysis_rules(self) -> List[ValidationRule]:
        """패턴 분석 설정 검증 규칙"""
        return [
            # 위치 편향 분석
            ValidationRule(
                key_path="new_pattern_analysis.position_bias.enabled",
                required=True,
                data_type=bool,
                description="위치 편향 분석 활성화",
            ),
            ValidationRule(
                key_path="new_pattern_analysis.position_bias.position_count",
                required=True,
                data_type=int,
                min_value=6,
                max_value=6,
                description="위치 개수",
            ),
            # 시간적 주기성 분석
            ValidationRule(
                key_path="new_pattern_analysis.overlap_time_gaps.analysis_window",
                required=True,
                data_type=int,
                min_value=10,
                max_value=1000,
                description="분석 윈도우 크기",
            ),
            ValidationRule(
                key_path="new_pattern_analysis.overlap_time_gaps.min_overlap_count",
                required=True,
                data_type=int,
                min_value=1,
                max_value=10,
                description="최소 중복 개수",
            ),
            # 조건부 상호작용 분석
            ValidationRule(
                key_path="new_pattern_analysis.conditional_interaction.significance_level",
                required=True,
                data_type=float,
                min_value=0.001,
                max_value=0.1,
                description="유의수준",
            ),
            ValidationRule(
                key_path="new_pattern_analysis.conditional_interaction.top_k_pairs",
                required=True,
                data_type=int,
                min_value=1,
                max_value=100,
                description="상위 K개 쌍",
            ),
            # 미세 편향성 분석
            ValidationRule(
                key_path="new_pattern_analysis.micro_bias.bias_threshold",
                required=True,
                data_type=float,
                min_value=0.01,
                max_value=0.5,
                description="편향 임계값",
            ),
            ValidationRule(
                key_path="new_pattern_analysis.micro_bias.moving_average_window",
                required=True,
                data_type=int,
                min_value=10,
                max_value=200,
                description="이동평균 윈도우",
            ),
        ]

    def validate_config(
        self, config: Dict[str, Any], config_type: ConfigType
    ) -> Tuple[bool, List[str]]:
        """
        설정 검증 실행

        Args:
            config: 검증할 설정 딕셔너리
            config_type: 설정 타입

        Returns:
            Tuple[bool, List[str]]: (검증 성공 여부, 오류 메시지 목록)
        """
        errors = []
        rules = self.validation_rules.get(config_type, [])

        for rule in rules:
            try:
                error = self._validate_single_rule(config, rule)
                if error:
                    errors.append(error)
            except Exception as e:
                errors.append(f"규칙 검증 중 오류 ({rule.key_path}): {e}")

        # 포트폴리오 배분 합계 검증 (realistic_lottery만)
        if config_type == ConfigType.REALISTIC_LOTTERY:
            portfolio_error = self._validate_portfolio_allocation(config)
            if portfolio_error:
                errors.append(portfolio_error)

        success = len(errors) == 0

        if success:
            self.logger.info(f"✅ {config_type.value} 설정 검증 성공")
        else:
            self.logger.error(
                f"❌ {config_type.value} 설정 검증 실패: {len(errors)}개 오류"
            )
            for error in errors:
                self.logger.error(f"  - {error}")

        return success, errors

    def _validate_single_rule(
        self, config: Dict[str, Any], rule: ValidationRule
    ) -> Optional[str]:
        """단일 규칙 검증"""
        keys = rule.key_path.split(".")
        current = config

        # 키 경로 탐색
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                if rule.required:
                    return f"필수 키 누락: {rule.key_path}"
                else:
                    return None  # 선택적 키는 누락 허용
            current = current[key]

        value = current

        # 타입 검증
        if not isinstance(value, rule.data_type):
            return f"타입 오류 ({rule.key_path}): {type(value).__name__} != {rule.data_type.__name__}"

        # 범위 검증 (숫자 타입만)
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                return f"최소값 위반 ({rule.key_path}): {value} < {rule.min_value}"
            if rule.max_value is not None and value > rule.max_value:
                return f"최대값 위반 ({rule.key_path}): {value} > {rule.max_value}"

        # 허용값 검증
        if rule.allowed_values is not None and value not in rule.allowed_values:
            return (
                f"허용값 위반 ({rule.key_path}): {value} not in {rule.allowed_values}"
            )

        return None

    def _validate_portfolio_allocation(self, config: Dict[str, Any]) -> Optional[str]:
        """포트폴리오 배분 합계 검증"""
        try:
            portfolio = config.get("realistic_lottery", {}).get(
                "portfolio_allocation", {}
            )
            conservative = portfolio.get("conservative", 0)
            aggressive = portfolio.get("aggressive", 0)
            balanced = portfolio.get("balanced", 0)

            total = conservative + aggressive + balanced
            if abs(total - 1.0) > 0.01:  # 1% 오차 허용
                return f"포트폴리오 배분 합계 오류: {total} != 1.0"

            return None
        except Exception as e:
            return f"포트폴리오 배분 검증 오류: {e}"

    def validate_config_file(
        self, config_path: str, config_type: ConfigType
    ) -> Tuple[bool, List[str]]:
        """
        설정 파일 직접 검증

        Args:
            config_path: 설정 파일 경로
            config_type: 설정 타입

        Returns:
            Tuple[bool, List[str]]: (검증 성공 여부, 오류 메시지 목록)
        """
        try:
            if not os.path.exists(config_path):
                return False, [f"설정 파일 없음: {config_path}"]

            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    config = yaml.safe_load(f)
                elif config_path.endswith(".json"):
                    config = json.load(f)
                else:
                    return False, [f"지원하지 않는 파일 형식: {config_path}"]

            if not isinstance(config, dict):
                return False, [f"설정 파일 형식 오류: 딕셔너리가 아님"]

            return self.validate_config(config, config_type)

        except Exception as e:
            return False, [f"설정 파일 로드 오류: {e}"]

    def generate_validation_report(
        self, results: Dict[str, Tuple[bool, List[str]]]
    ) -> str:
        """검증 결과 리포트 생성"""
        report_lines = ["📋 설정 검증 리포트", "=" * 50, ""]

        total_configs = len(results)
        success_count = sum(1 for success, _ in results.values() if success)

        report_lines.append(f"총 설정 파일: {total_configs}")
        report_lines.append(f"검증 성공: {success_count}")
        report_lines.append(f"검증 실패: {total_configs - success_count}")
        report_lines.append("")

        for config_name, (success, errors) in results.items():
            status = "✅ 성공" if success else "❌ 실패"
            report_lines.append(f"{config_name}: {status}")

            if errors:
                for error in errors:
                    report_lines.append(f"  - {error}")
            report_lines.append("")

        return "\n".join(report_lines)


# 싱글톤 인스턴스
_validator_instance = None


def get_config_validator() -> ConfigValidator:
    """설정 검증기 싱글톤 인스턴스 반환"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ConfigValidator()
    return _validator_instance


def validate_all_configs() -> Dict[str, Tuple[bool, List[str]]]:
    """모든 설정 파일 검증"""
    validator = get_config_validator()

    configs_to_validate = [
        ("config/config.yaml", ConfigType.MAIN),
        ("config/realistic_lottery_config.yaml", ConfigType.REALISTIC_LOTTERY),
        ("config/new_pattern_analysis_config.yaml", ConfigType.PATTERN_ANALYSIS),
    ]

    results = {}

    for config_path, config_type in configs_to_validate:
        if os.path.exists(config_path):
            success, errors = validator.validate_config_file(config_path, config_type)
            results[config_path] = (success, errors)
        else:
            results[config_path] = (False, [f"파일 없음: {config_path}"])

    return results


if __name__ == "__main__":
    # 테스트 실행
    results = validate_all_configs()
    validator = get_config_validator()
    report = validator.generate_validation_report(results)
    print(report)
