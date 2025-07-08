"""
DAEBAK_AI 로또 예측 모델 패키지

이 패키지는 로또 번호 예측을 위한 다양한 모델 구현을 포함합니다.
통합된 인터페이스를 통해 일관된 API를 제공합니다.
"""

from .base_model import BaseModel, ModelWithAMP, EnsembleBaseModel

# 남은 모델만 import
from .ml import (
    MLBaseModel,
    LightGBMModel,
    RandomForestModel,
)
from .dl import AutoencoderModel, TCNModel

# 통합 모델 관리자
from .unified_model_manager import UnifiedModelManager, get_unified_model_manager

from typing import Any, Dict, Optional, Type
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)

# 사용 가능한 모델 클래스들을 딕셔너리에 매핑
# 키는 설정 파일이나 코드에서 사용할 모델의 이름입니다.
AVAILABLE_MODELS: Dict[str, Type[BaseModel]] = {
    "LightGBMModel": LightGBMModel,
    "RandomForestModel": RandomForestModel,
    # "AutoEncoderModel": AutoEncoderModel,
    # "TCNModel": TCNModel,
}

def create_model(model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
    """
    모델 팩토리 함수

    Args:
        model_name (str): 생성할 모델의 이름.
        config (Optional[Dict[str, Any]]): 모델에 전달할 설정.

    Returns:
        BaseModel: 생성된 모델 인스턴스.

    Raises:
        ValueError: 요청된 모델을 사용할 수 없는 경우.
    """

    if model_name not in AVAILABLE_MODELS:
        logger.error(f"알 수 없는 모델 이름: '{model_name}'. "
                     f"사용 가능한 모델: {list(AVAILABLE_MODELS.keys())}")
        raise ValueError(f"모델 '{model_name}'을(를) 찾을 수 없습니다.")

    model_class = AVAILABLE_MODELS[model_name]
    
    try:
        logger.info(f"'{model_name}' 모델 인스턴스 생성 중...")
        # 모델 클래스에 설정을 전달하여 인스턴스 생성
        instance = model_class(config=config)
        logger.info(f"'{model_name}' 모델이 성공적으로 생성되었습니다.")
        return instance
    except Exception as e:
        logger.error(f"'{model_name}' 모델 생성 중 오류 발생: {e}", exc_info=True)
        raise

__all__ = [
    "BaseModel",
    "ModelWithAMP",
    "EnsembleBaseModel",
    "MLBaseModel",
    "LightGBMModel",
    "RandomForestModel",
    "AutoencoderModel",
    "TCNModel",
    "UnifiedModelManager",
    "get_unified_model_manager",
    "create_model",
]
