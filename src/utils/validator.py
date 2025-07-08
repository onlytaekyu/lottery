"""
데이터 및 모델의 유효성을 검증하기 위한 간단한 유틸리티 함수 및 클래스를 제공합니다.
이 모듈은 오버엔지니어링을 피하고, 단순하고 명확한 검증 로직에 집중합니다.
"""

from typing import Any, Dict, List

from .unified_logging import get_logger

logger = get_logger(__name__)

class SimpleValidator:
    """
    단순한 데이터 유효성 검증기.
    필요에 따라 구체적인 검증 규칙을 추가할 수 있습니다.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        logger.info("SimpleValidator initialized.")

    def validate_lottery_numbers(self, numbers: List[int]) -> bool:
        """
        단일 로또 번호 조합의 유효성을 검증합니다.
        - 6개의 숫자인지 확인
        - 1~45 범위 내에 있는지 확인
        - 중복이 없는지 확인
        """
        if len(numbers) != 6:
            logger.warning(f"Invalid number count: {len(numbers)}")
            return False
        if not all(1 <= n <= 45 for n in numbers):
            logger.warning(f"Numbers out of range: {numbers}")
            return False
        if len(set(numbers)) != 6:
            logger.warning(f"Duplicate numbers found: {numbers}")
            return False
        
        return True

def get_validator(config: Dict[str, Any] = None) -> SimpleValidator:
    """SimpleValidator의 인스턴스를 반환합니다."""
    return SimpleValidator(config) 