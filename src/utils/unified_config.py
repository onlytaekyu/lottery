"""
통합 설정 관리 시스템

설정 로드, 검증, 캐싱 기능을 통합하여 중복 코드를 제거합니다.
"""

import os
import yaml
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .unified_logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class ConfigPath:
    """설정 파일 경로 정보"""

    main_config: str = "config/config.yaml"
    optimization_config: str = "config/optimization.yaml"
    pattern_analysis_config: str = "config/new_pattern_analysis_config.yaml"

    def __post_init__(self):
        """경로 정규화"""
        self.main_config = os.path.normpath(self.main_config)
        self.optimization_config = os.path.normpath(self.optimization_config)
        self.pattern_analysis_config = os.path.normpath(self.pattern_analysis_config)


@dataclass
class DirectoryPaths:
    """디렉토리 경로 설정"""

    # 데이터 디렉토리
    data_root: str = "data"
    cache_dir: str = "data/cache"
    raw_data_dir: str = "data/raw"
    result_dir: str = "data/result"
    predictions_dir: str = "data/predictions"

    # 로그 디렉토리
    logs_dir: str = "logs"

    # 모델 디렉토리
    models_dir: str = "savedModels"

    # 임시 디렉토리
    temp_dir: str = "temp"

    def __post_init__(self):
        """경로 정규화 및 생성"""
        for field_name, field_value in self.__dict__.items():
            normalized_path = os.path.normpath(field_value)
            setattr(self, field_name, normalized_path)

            # 디렉토리 생성
            Path(normalized_path).mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, filename: str) -> str:
        """캐시 파일 경로 반환"""
        return os.path.join(self.cache_dir, filename)

    def get_result_path(self, filename: str) -> str:
        """결과 파일 경로 반환"""
        return os.path.join(self.result_dir, filename)

    def get_model_path(self, filename: str) -> str:
        """모델 파일 경로 반환"""
        return os.path.join(self.models_dir, filename)


class ConfigValidator(ABC):
    """설정 검증 인터페이스"""

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> bool:
        """설정 검증"""

    @abstractmethod
    def get_required_keys(self) -> List[str]:
        """필수 키 목록 반환"""


class DefaultConfigValidator(ConfigValidator):
    """기본 설정 검증기"""

    def __init__(self, required_sections: Optional[List[str]] = None):
        self.required_sections = required_sections or [
            "paths",
            "caching",
            "training",
            "analysis",
        ]

    def validate(self, config: Dict[str, Any]) -> bool:
        """설정 검증"""
        try:
            # 필수 섹션 확인
            for section in self.required_sections:
                if section not in config:
                    logger.error(f"필수 설정 섹션 누락: {section}")
                    return False

            # 경로 설정 검증
            if "paths" in config:
                paths_config = config["paths"]
                required_paths = ["feature_vector_path", "name_file_path"]
                for path_key in required_paths:
                    if path_key not in paths_config:
                        logger.error(f"필수 경로 설정 누락: paths.{path_key}")
                        return False

            # 벡터 설정 검증
            if "vector" in config:
                vector_config = config["vector"]
                if "min_required_dimension" in vector_config:
                    min_dim = vector_config["min_required_dimension"]
                    if not isinstance(min_dim, int) or min_dim < 1:
                        logger.error(f"잘못된 최소 벡터 차원: {min_dim}")
                        return False

            return True

        except Exception as e:
            logger.error(f"설정 검증 중 오류 발생: {e}")
            return False

    def get_required_keys(self) -> List[str]:
        """필수 키 목록 반환"""
        return [
            "paths.feature_vector_path",
            "paths.name_file_path",
            "caching.enabled",
            "training.use_filtered_vector",
            "analysis.enable_pattern_analysis",
        ]


class UnifiedConfigManager:
    """통합 설정 관리자"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """싱글톤 패턴 구현"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = get_logger("config")
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._validators: Dict[str, ConfigValidator] = {}
        self._watchers: Dict[str, float] = {}  # 파일 수정 시간 추적
        self._cache_enabled = True
        self._auto_reload = False

        # 기본 경로 설정
        self.config_paths = ConfigPath()
        self.directory_paths = DirectoryPaths()

        # 기본 검증기 등록
        self.register_validator("main", DefaultConfigValidator())

        self._initialized = True
        self.logger.info("통합 설정 관리자 초기화 완료")

    def register_validator(self, config_name: str, validator: ConfigValidator):
        """설정 검증기 등록"""
        self._validators[config_name] = validator
        self.logger.debug(f"설정 검증기 등록: {config_name}")

    def load_config(
        self,
        config_name: str,
        file_path: Optional[str] = None,
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """
        설정 파일 로드

        Args:
            config_name: 설정 이름
            file_path: 설정 파일 경로 (None이면 기본 경로 사용)
            force_reload: 강제 재로드 여부

        Returns:
            설정 딕셔너리
        """
        with self._lock:
            # 캐시된 설정 확인
            if (
                not force_reload
                and self._cache_enabled
                and config_name in self._configs
            ):
                if not self._auto_reload or not self._is_file_modified(
                    config_name, file_path
                ):
                    return self._configs[config_name].copy()

            # 파일 경로 결정
            if file_path is None:
                file_path = self._get_default_config_path(config_name)

            # 설정 파일 로드
            try:
                config = self._load_config_file(file_path)

                # 설정 검증
                if config_name in self._validators:
                    if not self._validators[config_name].validate(config):
                        raise ValueError(f"설정 검증 실패: {config_name}")

                # 캐시에 저장
                self._configs[config_name] = config
                self._watchers[config_name] = (
                    os.path.getmtime(file_path) if os.path.exists(file_path) else 0
                )

                self.logger.info(f"설정 로드 완료: {config_name} ({file_path})")
                return config.copy()

            except Exception as e:
                self.logger.error(f"설정 로드 실패: {config_name} - {e}")
                raise RuntimeError(f"설정 로드 실패: {config_name}") from e

    def _get_default_config_path(self, config_name: str) -> str:
        """기본 설정 파일 경로 반환"""
        path_mapping = {
            "main": self.config_paths.main_config,
            "optimization": self.config_paths.optimization_config,
            "pattern_analysis": self.config_paths.pattern_analysis_config,
        }

        if config_name in path_mapping:
            return path_mapping[config_name]
        else:
            return f"config/{config_name}.yaml"

    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {file_path}")

        file_ext = Path(file_path).suffix.lower()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_ext in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif file_ext == ".json":
                    return json.load(f) or {}
                else:
                    raise ValueError(f"지원되지 않는 설정 파일 형식: {file_ext}")

        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파싱 오류: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 오류: {e}")

    def _is_file_modified(
        self, config_name: str, file_path: Optional[str] = None
    ) -> bool:
        """파일 수정 여부 확인"""
        if config_name not in self._watchers:
            return True

        current_path = file_path or self._get_default_config_path(config_name)

        if not os.path.exists(current_path):
            return False

        current_mtime = os.path.getmtime(current_path)
        return current_mtime > self._watchers[config_name]

    def get_config(self, config_name: str = "main") -> "ConfigProxy":
        """
        설정 프록시 반환

        Args:
            config_name: 설정 이름

        Returns:
            설정 프록시 객체
        """
        config = self.load_config(config_name)
        return ConfigProxy(config, config_name, self)

    def get_nested_value(
        self, config: Dict[str, Any], key_path: str, default: Any = None
    ) -> Any:
        """
        중첩된 키 경로로 값 조회

        Args:
            config: 설정 딕셔너리
            key_path: 점으로 구분된 키 경로 (예: "training.batch_size")
            default: 기본값

        Returns:
            설정 값
        """
        try:
            keys = key_path.split(".")
            value = config

            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    if default is None:
                        raise KeyError(f"설정 키 누락: {key_path}")
                    return default

            return value

        except Exception as e:
            if default is None:
                self.logger.error(f"설정 키 접근 실패: {key_path} - {e}")
                raise KeyError(f"설정 키 누락: {key_path}")
            return default


class ConfigProxy:
    """설정 프록시 클래스 (기존 API 호환성)"""

    def __init__(
        self, config: Dict[str, Any], config_name: str, manager: UnifiedConfigManager
    ):
        self._config = config
        self._config_name = config_name
        self._manager = manager

    def __getitem__(self, key: str) -> Any:
        """딕셔너리 스타일 접근"""
        if key not in self._config:
            logger.error(f"[ERROR] 설정 키 누락: config['{key}']")
            raise KeyError(f"설정 키 누락: {key}")
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """안전한 키 접근 (기본값 지원)"""
        if default is None and key not in self._config:
            logger.error(f"[ERROR] 설정 키 누락: config.get('{key}')")
            raise KeyError(f"설정 키 누락: {key}")
        return self._config.get(key, default)

    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """중첩된 키 경로로 값 조회"""
        return self._manager.get_nested_value(self._config, key_path, default)


# 편의 함수들
def get_config_manager() -> UnifiedConfigManager:
    """통합 설정 관리자 반환"""
    return UnifiedConfigManager()


def load_config(
    config_name: str = "main", file_path: Optional[str] = None
) -> ConfigProxy:
    """설정 로드 (편의 함수)"""
    manager = UnifiedConfigManager()
    return manager.get_config(config_name)


def get_directory_paths() -> DirectoryPaths:
    """디렉토리 경로 설정 반환"""
    manager = UnifiedConfigManager()
    return manager.directory_paths


def get_paths() -> DirectoryPaths:
    """경로 설정 반환 (별칭)"""
    return get_directory_paths()


def get_config(config_name: str = "main") -> ConfigProxy:
    """설정 반환 (편의 함수)"""
    manager = UnifiedConfigManager()
    return manager.get_config(config_name)
