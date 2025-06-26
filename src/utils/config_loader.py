"""
설정 파일 로더 (Config Loader)

이 모듈은 YAML 형식의 설정 파일을 로드하는 기능을 제공합니다.
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Generic, cast, Union, List, Type
import json

from .error_handler import get_logger, log_exception_with_trace

# 로거 설정
logger = get_logger(__name__)

T = TypeVar("T")


class ConfigProxy:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._logger = logging.getLogger(__name__)

    def get(self, key: str) -> Any:
        """
        설정값을 가져옵니다. 키가 없으면 KeyError를 발생시킵니다.
        점 표기법으로 중첩된 키를 지원합니다 (예: "frequency_weights.long_term").
        """
        try:
            # 키가 문자열인지 확인
            if not isinstance(key, str):
                raise TypeError(f"설정 키는 문자열이어야 합니다: {type(key)}")

            if "." in key:
                parts = key.split(".")
                current = self._config

                # 중첩 구조 탐색
                for part in parts[:-1]:
                    if not isinstance(current, dict) or part not in current:
                        raise KeyError(
                            f"설정 키를 찾을 수 없음: {key} (중간 경로 {part}가 없음)"
                        )
                    current = current[part]

                # 최종 키 확인
                last_key = parts[-1]
                if not isinstance(current, dict) or last_key not in current:
                    raise KeyError(
                        f"설정 키를 찾을 수 없음: {key} (최종 키 {last_key}가 없음)"
                    )

                return current[last_key]
            else:
                if key not in self._config:
                    raise KeyError(f"설정 키를 찾을 수 없음: {key}")
                return self._config[key]
        except KeyError as e:
            log_exception_with_trace(self._logger, e, f"설정 키 접근 오류 ({key})")
            raise
        except Exception as e:
            log_exception_with_trace(
                self._logger, e, f"설정 키 접근 중 예외 발생 ({key})"
            )
            raise

    def get_typed(self, key: str, expected_type: Type) -> Any:
        """
        설정값을 가져오고 지정한 타입인지 확인합니다.
        타입이 일치하지 않으면 TypeError를 발생시킵니다.

        Args:
            key: 설정 키
            expected_type: 기대하는 값의 타입

        Returns:
            설정값

        Raises:
            KeyError: 키가 없는 경우
            TypeError: 값의 타입이 일치하지 않는 경우
        """
        value = self.get(key)
        if not isinstance(value, expected_type):
            error_msg = f"설정 '{key}'의 타입이 일치하지 않습니다. 기대: {expected_type.__name__}, 실제: {type(value).__name__}"
            self._logger.error(error_msg)
            raise TypeError(error_msg)
        return value

    def __getitem__(self, key: str) -> Any:
        """
        딕셔너리 스타일로 설정값을 가져옵니다. 키가 없으면 KeyError를 발생시킵니다.
        중첩된 딕셔너리에도 접근 가능합니다 (예: config["models"]["lightgbm"]).
        """
        try:
            # 키가 문자열인지 확인
            if not isinstance(key, str):
                raise TypeError(f"설정 키는 문자열이어야 합니다: {type(key)}")

            if key not in self._config:
                raise KeyError(f"설정 키를 찾을 수 없음: {key}")
            return self._config[key]
        except KeyError as e:
            log_exception_with_trace(self._logger, e, f"설정 키 접근 오류 ({key})")
            raise
        except Exception as e:
            log_exception_with_trace(
                self._logger, e, f"설정 키 접근 중 예외 발생 ({key})"
            )
            raise

    def __setitem__(self, key: str, value: Any) -> None:
        """딕셔너리 스타일로 설정값을 설정합니다."""
        self._config[key] = value

    def get_nested(self, *keys: str) -> Any:
        """중첩된 설정값을 가져옵니다. 키가 없으면 KeyError를 발생시킵니다."""
        try:
            current: Any = self._config
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    raise KeyError(
                        f"설정 키를 찾을 수 없음: {'.'.join(keys)} (중간 경로 {key}가 없음)"
                    )
                current = current[key]
            return current
        except KeyError as e:
            log_exception_with_trace(
                self._logger, e, f"중첩 설정 키 접근 오류 ({'.'.join(keys)})"
            )
            raise

    def safe_get(self, key: str, default: Any = None) -> Any:
        """
        설정값을 안전하게 가져옵니다. 키가 없으면 기본값을 반환합니다.
        점 표기법으로 중첩된 키를 지원합니다 (예: "frequency_weights.long_term").

        경고: 이 메서드는 기본값을 사용하기 때문에 오류를 숨길 수 있습니다.
        중요한 설정 키의 경우 get() 메서드를 사용하는 것이 좋습니다.
        """
        try:
            # 키가 문자열인지 확인
            if not isinstance(key, str):
                self._logger.warning(
                    f"잘못된 키 타입: {type(key)}. 문자열이어야 합니다. 기본값 {default} 반환"
                )
                return default

            return self.get(key)
        except KeyError:
            self._logger.warning(
                f"설정 키를 찾을 수 없어 기본값 사용: {key} -> {default}"
            )
            return default
        except Exception as e:
            self._logger.warning(
                f"설정 접근 중 예외 발생: {str(e)}. 기본값 {default} 반환"
            )
            return default

    def safe_get_typed(self, key: str, expected_type: Type, default: Any = None) -> Any:
        """
        설정값을 안전하게 가져오고 지정한 타입인지 확인합니다.
        키가 없거나 타입이 일치하지 않으면 기본값을 반환합니다.

        Args:
            key: 설정 키
            expected_type: 기대하는 값의 타입
            default: 키가 없거나 타입이 일치하지 않을 때 반환할 기본값

        Returns:
            설정값 또는 기본값
        """
        try:
            value = self.get(key)
            if not isinstance(value, expected_type):
                self._logger.warning(
                    f"설정 '{key}'의 타입이 일치하지 않습니다. 기대: {expected_type.__name__}, 실제: {type(value).__name__}. 기본값 반환"
                )
                return default
            return value
        except KeyError:
            self._logger.warning(
                f"설정 키를 찾을 수 없어 기본값 사용: {key} -> {default}"
            )
            return default
        except Exception as e:
            self._logger.warning(
                f"설정 접근 중 예외 발생: {str(e)}. 기본값 {default} 반환"
            )
            return default

    def update(self, key: str, value: Any) -> None:
        """설정값을 업데이트합니다."""
        self._config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """현재 설정을 딕셔너리로 반환합니다."""
        return self._config.copy()

    @classmethod
    def from_file(cls, path: str | Path) -> "ConfigProxy":
        """파일에서 설정을 로드합니다. 실패 시 예외를 발생시킵니다."""
        logger = get_logger(__name__)
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return cls(config)
        except Exception as e:
            log_exception_with_trace(logger, e, f"설정 파일 로드 실패 ({path})")
            raise ValueError(f"설정 파일 로드 실패: {str(e)}")

    def save(self, path: str | Path) -> None:
        """현재 설정을 파일로 저장합니다."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_exception_with_trace(self._logger, e, f"설정 파일 저장 실패 ({path})")
            raise IOError(f"설정 파일 저장 실패: {str(e)}")

    def validate_critical_paths(self, critical_keys: List[str] = None) -> bool:
        """
        중요한 설정 키가 존재하는지 검증합니다.

        Args:
            critical_keys: 검증할 중요 키 목록 (기본값: None, 모든 필수 키 검증)

        Returns:
            bool: 모든 키가 유효한지 여부

        Raises:
            KeyError: 필수 키가 없는 경우
        """
        if critical_keys is None:
            # 기본 필수 키 목록
            critical_keys = [
                "paths.data_dir",
                "paths.model_save_dir",
                "paths.cache_dir",
                "paths.result_dir",
                "caching.max_cache_size",
                "caching.enable_feature_cache",
            ]

        missing_keys = []
        for key in critical_keys:
            try:
                self.get(key)
            except KeyError:
                missing_keys.append(key)

        if missing_keys:
            error_msg = f"필수 설정 키가 없습니다: {missing_keys}"
            self._logger.error(error_msg)
            raise KeyError(error_msg)

        return True

    def validate_keys(self, keys: List[str]) -> bool:
        """
        지정된 설정 키 목록이 존재하는지 검증합니다.

        Args:
            keys: 검증할 키 목록

        Returns:
            bool: 모든 키가 유효한지 여부

        Raises:
            KeyError: 키가 없는 경우
        """
        missing_keys = []

        for key in keys:
            try:
                # 점(.) 표기법이 있는 키인 경우
                if "." in key:
                    parts = key.split(".")
                    current = self._config

                    # 경로 탐색
                    valid_key = True
                    for part in parts:
                        if not isinstance(current, dict) or part not in current:
                            valid_key = False
                            break
                        current = current[part]

                    if not valid_key:
                        missing_keys.append(key)
                else:
                    # 단일 키인 경우
                    if key not in self._config:
                        missing_keys.append(key)
            except Exception:
                missing_keys.append(key)

        if missing_keys:
            error_msg = f"다음 설정 키를 찾을 수 없습니다: {missing_keys}"
            self._logger.error(error_msg)
            raise KeyError(error_msg)

        return True

    def get_with_fallback(
        self, key: str, fallback_key: str, default: Any = None
    ) -> Any:
        """
        주 키에서 값을 가져오고, 없으면 대체 키에서 가져옵니다.
        두 키 모두 없으면 기본값을 반환합니다.

        Args:
            key: 주 키
            fallback_key: 대체 키
            default: 기본값

        Returns:
            설정값 또는 기본값
        """
        try:
            return self.get(key)
        except KeyError:
            self._logger.warning(
                f"주 키 '{key}'를 찾을 수 없어 대체 키 '{fallback_key}' 사용"
            )
            try:
                return self.get(fallback_key)
            except KeyError:
                self._logger.warning(
                    f"대체 키 '{fallback_key}'도 찾을 수 없어 기본값 사용"
                )
                return default

    def get_typed_with_validation(
        self, key: str, expected_type: Type, validator=None, default: Any = None
    ) -> Any:
        """
        설정값을 가져오고 타입과 유효성을 검증합니다.

        Args:
            key: 설정 키
            expected_type: 기대하는 값의 타입
            validator: 유효성 검증 함수 (선택적)
            default: 기본값 (키가 없거나 유효하지 않은 경우)

        Returns:
            검증된 설정값 또는 기본값
        """
        try:
            value = self.get(key)

            # 타입 검증
            if not isinstance(value, expected_type):
                self._logger.warning(
                    f"키 '{key}'의 타입이 {expected_type.__name__}이 아닙니다: {type(value).__name__}"
                )
                return default

            # 유효성 검증 (제공된 경우)
            if validator is not None and callable(validator):
                if not validator(value):
                    self._logger.warning(
                        f"키 '{key}'의 값이 유효하지 않습니다: {value}"
                    )
                    return default

            return value
        except KeyError:
            self._logger.warning(f"키 '{key}'를 찾을 수 없어 기본값 사용: {default}")
            return default


def load_config(config_path: Optional[str] = None) -> ConfigProxy:
    """
    YAML 설정 파일 로드

    Args:
        config_path: 설정 파일 경로 (기본값: config/config.yaml)

    Returns:
        설정 프록시 객체

    Raises:
        FileNotFoundError: 설정 파일이 존재하지 않을 때
        ValueError: 설정 파일 형식이 올바르지 않을 때
        IOError: 파일 읽기에 실패했을 때
    """
    # 기본 설정 파일 경로
    if config_path is None:
        config_path = "config/config.yaml"

    # 상대 경로를 절대 경로로 변환
    config_file = Path(config_path)
    if not config_file.is_absolute():
        # 프로젝트 루트 디렉토리 기준 경로 설정
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / config_path

    # 설정 파일 존재 여부 확인
    if not config_file.exists():
        error_msg = f"설정 파일을 찾을 수 없습니다: {config_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # YAML 파일 로드
        with open(config_file, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                if config is None:
                    error_msg = f"설정 파일이 비어있습니다: {config_file}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            except yaml.YAMLError as e:
                log_exception_with_trace(logger, e, f"YAML 파싱 오류 ({config_file})")
                raise ValueError(f"YAML 파싱 오류: {str(e)}")

        logger.info(f"설정 파일 로드 성공: {config_file}")
        config_proxy = ConfigProxy(config)

        # 중요 경로 유효성 검증
        if not config_proxy.validate_critical_paths():
            logger.warning(
                "일부 중요 설정 키가 누락되었습니다. 기본값을 사용하거나 오류가 발생할 수 있습니다."
            )

        return config_proxy

    except (IOError, OSError) as e:
        log_exception_with_trace(logger, e, f"설정 파일 읽기 오류 ({config_file})")
        raise IOError(f"설정 파일 읽기 오류: {str(e)}")


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
