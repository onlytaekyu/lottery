"""
설정 로드 및 관리 유틸리티 (하위 호환성)

새 코드는 utils.unified_config 를 직접 사용하세요.
이 모듈은 기존 코드의 import 호환을 위해 최소 래퍼만 제공합니다.
"""

from __future__ import annotations

from .unified_config import (
    ConfigProxy,
    load_config as unified_load_config,
    get_config_manager,
)
from .error_handler_refactored import get_logger

logger = get_logger(__name__)

__all__ = [
    "load_config",
    "get_default_config",
    "ConfigProxy",
    "get_config_manager",
]


def load_config(config_path: str | None = None):
    """하위 호환성을 위한 래퍼 함수"""
    logger.warning(
        "config_loader.load_config은 deprecated입니다. utils.unified_config.load_config을 사용하세요."
    )
    if config_path is None:
        return unified_load_config("main")
    return ConfigProxy.from_file(config_path)


def get_default_config():
    """하위 호환성을 위한 기본 설정 반환"""
    logger.warning("config_loader.get_default_config은 deprecated입니다.")
    return {
        "paths": {
            "data_dir": "data",
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "model_dir": "savedModels",
            "cache_dir": "data/cache",
            "result_dir": "data/result",
            "log_dir": "logs",
        },
        "data": {
            "lottery_file": "lottery.csv",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
    }
