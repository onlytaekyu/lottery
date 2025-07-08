#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
설정 검증 시스템

각 설정 파일별로 타입, 범위, 필수 여부를 검증하는 시스템입니다.
"""

from typing import Dict, Any, Type, Optional, Tuple
from enum import Enum
import yaml
from pydantic import BaseModel, Field, ValidationError

from .unified_logging import get_logger

logger = get_logger(__name__)


class ConfigType(Enum):
    """설정 타입 열거형"""
    MAIN = "main"
    REALISTIC_LOTTERY = "realistic_lottery"
    PATTERN_ANALYSIS = "pattern_analysis"


# --- Pydantic 스키마 정의 ---

class DataConfig(BaseModel):
    max_number: int = Field(..., ge=1, le=100, description="최대 번호 (로또: 45)")
    min_number: int = Field(..., ge=1, le=45, description="최소 번호 (로또: 1)")
    sequence_length: int = Field(..., ge=6, le=6, description="시퀀스 길이 (로또: 6)")

class ExecutionConfig(BaseModel):
    max_threads: int = Field(..., ge=1, le=32, description="최대 스레드 수")
    use_gpu: bool = Field(..., description="GPU 사용 여부")

class OptimizationMemoryConfig(BaseModel):
    max_memory_usage: float = Field(..., ge=0.1, le=0.95, description="최대 메모리 사용률")

class OptimizationProcessPoolConfig(BaseModel):
    max_workers: int = Field(..., ge=1, le=16, description="프로세스 풀 최대 워커 수")

class OptimizationConfig(BaseModel):
    memory: OptimizationMemoryConfig
    process_pool: OptimizationProcessPoolConfig

class AnalysisConfig(BaseModel):
    enable_position_bias_analysis: bool
    enable_micro_bias_analysis: bool

class MainConfig(BaseModel):
    """메인 설정 파일 스키마"""
    data: DataConfig
    execution: ExecutionConfig
    optimization: OptimizationConfig
    analysis: AnalysisConfig

class PrizeImprovement(BaseModel):
    base_probability: float = Field(..., ge=0.0, le=1.0)
    target_probability: float = Field(..., ge=0.0, le=1.0)

class TargetImprovements(BaseModel):
    fifth_prize: PrizeImprovement = Field(..., alias="5th_prize")
    fourth_prize: PrizeImprovement = Field(..., alias="4th_prize")
    third_prize: PrizeImprovement = Field(..., alias="3rd_prize")

class PortfolioAllocation(BaseModel):
    conservative: float = Field(..., ge=0.0, le=1.0)
    aggressive: float = Field(..., ge=0.0, le=1.0)
    balanced: float = Field(..., ge=0.0, le=1.0)

class RiskManagement(BaseModel):
    kelly_fraction: float = Field(..., ge=0.01, le=1.0)
    max_consecutive_losses: int = Field(..., ge=1, le=50)

class PerformanceGoals(BaseModel):
    hit_rate_5th: float = Field(..., ge=0.0, le=1.0)
    total_loss_target: float = Field(..., ge=0.0, le=1.0)

class RealisticLotteryConfig(BaseModel):
    """현실적 로또 설정 파일 스키마"""
    target_improvements: TargetImprovements
    portfolio_allocation: PortfolioAllocation
    risk_management: RiskManagement
    performance_goals: PerformanceGoals

class PositionBias(BaseModel):
    enabled: bool
    position_count: int = Field(..., ge=6, le=6)

class TemporalPeriodicity(BaseModel):
    enabled: bool
    default_period: int

class NewPatternAnalysisConfig(BaseModel):
    position_bias: PositionBias
    temporal_periodicity: TemporalPeriodicity

class PatternAnalysisConfig(BaseModel):
    """패턴 분석 설정 파일 스키마"""
    new_pattern_analysis: NewPatternAnalysisConfig


# --- 검증기 클래스 ---

class ConfigValidator:
    """Pydantic 모델을 사용한 설정 검증기"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.schema_map: Dict[ConfigType, Type[BaseModel]] = {
            ConfigType.MAIN: MainConfig,
            ConfigType.REALISTIC_LOTTERY: RealisticLotteryConfig,
            ConfigType.PATTERN_ANALYSIS: PatternAnalysisConfig,
        }

    def validate_config(self, config_data: Dict[str, Any], config_type: ConfigType) -> Tuple[bool, Optional[str]]:
        """
        주어진 설정 데이터를 Pydantic 모델로 검증합니다.

        Args:
            config_data: 검증할 설정 데이터 딕셔너리.
            config_type: 설정 타입.

        Returns:
            (is_valid, error_message) 튜플.
        """
        schema = self.schema_map.get(config_type)
        if not schema:
            return False, f"알 수 없는 설정 타입: {config_type.value}"

        try:
            # realistic_lottery.yml 파일의 키 형식에 맞게 데이터 변환
            if config_type == ConfigType.REALISTIC_LOTTERY and "realistic_lottery" in config_data:
                config_data = config_data["realistic_lottery"]

            schema.model_validate(config_data)
            logger.info(f"✅ '{config_type.value}' 설정 검증 성공.")
            return True, None
        except ValidationError as e:
            logger.error(f"❌ '{config_type.value}' 설정 검증 실패.")
            return False, str(e)

    def validate_config_file(self, config_path: str, config_type: ConfigType) -> Tuple[bool, Optional[str]]:
        """
        설정 파일을 읽고 검증합니다.

        Args:
            config_path: 설정 파일 경로.
            config_type: 설정 타입.

        Returns:
            (is_valid, error_message) 튜플.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            return False, f"설정 파일 로드 실패: {e}"

        return self.validate_config(config_data, config_type)


def get_config_validator() -> ConfigValidator:
    """ConfigValidator의 싱글톤 인스턴스를 반환합니다."""
    # 필요하다면 여기에 싱글톤 로직 추가
    return ConfigValidator()


def validate_all_configs(config_dir: str = "config") -> Dict[str, Any]:
    """
    'config' 디렉토리의 모든 YAML 파일을 검증합니다.
    파일 이름을 기반으로 ConfigType을 추론합니다.
    """
    validator = get_config_validator()
    results = {}
    
    config_files = {
        "config.yaml": ConfigType.MAIN,
        "realistic_lottery.yml": ConfigType.REALISTIC_LOTTERY,
        "pattern_analysis.yml": ConfigType.PATTERN_ANALYSIS,
    }

    for filename, config_type in config_files.items():
        file_path = f"{config_dir}/{filename}"
        is_valid, error_msg = validator.validate_config_file(file_path, config_type)
        results[filename] = {"is_valid": is_valid, "errors": error_msg}

    return results


if __name__ == "__main__":
    # 테스트 실행
    results = validate_all_configs()
    validator = get_config_validator()
    # report = validator.generate_validation_report(results) # 기존 리포트 생성 로직은 제거됨
    print(results)
