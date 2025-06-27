"""
설정 로드 및 관리 유틸리티

이 모듈은 YAML 설정 파일을 로드하고 검증하는 기능을 제공합니다.
통합 설정 관리를 위해 unified_config.py와 연동됩니다.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# unified_config에서 ConfigProxy와 관련 함수들을 import
from .unified_config import ConfigProxy, load_config as unified_load_config

from .error_handler_refactored import get_logger, log_exception_with_trace

# 로거 설정
logger = get_logger(__name__)


# 하위 호환성을 위한 함수들
def load_config(config_path: Optional[str] = None) -> ConfigProxy:
    """
    설정 파일을 로드합니다. (하위 호환성을 위한 래퍼)

    Args:
        config_path: 설정 파일 경로 (기본값: None, 메인 설정 사용)

    Returns:
        ConfigProxy: 설정 프록시 객체
    """
    try:
        if config_path:
            # 사용자 지정 경로가 있는 경우
            return ConfigProxy.from_file(config_path)
        else:
            # 기본 설정 사용
            return unified_load_config("main")
    except Exception as e:
        log_exception_with_trace(logger, e, f"설정 로드 실패 ({config_path})")
        raise


def get_default_config() -> Dict[str, Any]:
    """
    기본 설정 반환

    Returns:
        기본 설정 딕셔너리
    """
    return {
        "paths": {
            "data_dir": "data",
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "model_dir": "savedModels",
            "model_save_dir": "savedModels",
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
        "caching": {
            "enable_feature_cache": True,
            "max_cache_size": 10000,
            "cache_log_level": "INFO",
            "cache_metrics": {
                "save": True,
                "report_interval": 1000,
                "file_path": "logs/cache_stats.json",
            },
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 100,
            "early_stopping": True,
            "patience": 10,
        },
        "models": {
            "rl": {
                "hidden_dim": 128,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "epsilon_decay": 0.995,
            },
            "statistical": {
                "use_frequency": True,
                "use_patterns": True,
                "min_confidence": 0.3,
            },
            "lstm": {
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.2,
            },
            "lgbm_params": {
                "learning_rate": 0.1,
                "n_estimators": 100,
                "num_leaves": 31,
                "max_depth": -1,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.8,
                "objective": "binary",
            },
            "xgb_params": {
                "learning_rate": 0.1,
                "n_estimators": 100,
                "max_depth": 6,
                "min_child_weight": 1,
                "gamma": 0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
            },
        },
        "reporting": {"enable_performance_report": True, "report_dir": "logs/reports"},
    }


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    설정을 YAML 파일로 저장

    Args:
        config: 저장할 설정
        config_path: 저장할 파일 경로

    Returns:
        성공 여부
    """
    try:
        # 디렉토리 생성
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # YAML 파일로 저장
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"설정 파일 저장 성공: {config_file}")
        return True

    except Exception as e:
        log_exception_with_trace(logger, e, f"설정 파일 저장 오류 ({config_path})")
        raise IOError(f"설정 파일 저장 실패: {str(e)}")


def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 설정과 기본 설정을 병합합니다.

    Args:
        config: 사용자 설정

    Returns:
        병합된 설정
    """
    result = get_default_config()

    def _deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                _deep_update(d[k], v)
            else:
                d[k] = v

    _deep_update(result, config)
    return result
