"""
통합 설정 관리 시스템

설정 로드, 검증, 캐싱 기능을 통합하여 중복 코드를 제거합니다.
타입 안전성, 동적 설정 변경, 스키마 검증 기능을 포함합니다.
"""

import os
import yaml
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, TypeVar, Union, Set, Type, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import time
import hashlib
from collections import defaultdict
import jsonschema
from jsonschema import validate, ValidationError
import torch
import psutil

from .unified_logging import get_logger

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigChangeType(Enum):
    """설정 변경 타입"""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class ConfigChange:
    """설정 변경 정보"""

    change_type: ConfigChangeType
    key_path: str
    old_value: Any = None
    new_value: Any = None
    timestamp: float = field(default_factory=time.time)


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

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """JSON 스키마 반환"""


class EnhancedConfigValidator(ConfigValidator):
    """향상된 설정 검증기 - JSON 스키마 기반"""

    def __init__(self, required_sections: Optional[List[str]] = None):
        self.required_sections = required_sections or [
            "paths",
            "caching",
            "training",
            "analysis",
        ]

        # JSON 스키마 정의
        self.schema = {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "object",
                    "properties": {
                        "feature_vector_path": {"type": "string"},
                        "name_file_path": {"type": "string"},
                        "data_dir": {"type": "string"},
                        "cache_dir": {"type": "string"},
                    },
                    "required": ["feature_vector_path", "name_file_path"],
                },
                "caching": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "max_size": {"type": "integer", "minimum": 1},
                        "ttl": {"type": "number", "minimum": 0},
                    },
                    "required": ["enabled"],
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "use_filtered_vector": {"type": "boolean"},
                        "batch_size": {"type": "integer", "minimum": 1},
                        "learning_rate": {"type": "number", "minimum": 0},
                        "epochs": {"type": "integer", "minimum": 1},
                        "use_amp": {"type": "boolean"},
                    },
                    "required": ["use_filtered_vector"],
                },
                "analysis": {
                    "type": "object",
                    "properties": {
                        "enable_pattern_analysis": {"type": "boolean"},
                        "min_pattern_frequency": {"type": "integer", "minimum": 1},
                        "max_pattern_length": {"type": "integer", "minimum": 1},
                    },
                    "required": ["enable_pattern_analysis"],
                },
                "vector": {
                    "type": "object",
                    "properties": {
                        "min_required_dimension": {"type": "integer", "minimum": 1},
                        "max_dimension": {"type": "integer", "minimum": 1},
                        "allow_mismatch": {"type": "boolean"},
                    },
                },
                "cuda": {
                    "type": "object",
                    "properties": {
                        "use_cuda": {"type": "boolean"},
                        "gpu_ids": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                        "use_amp": {"type": "boolean"},
                        "use_tensorrt": {"type": "boolean"},
                    },
                },
                "system": {
                    "type": "object",
                    "properties": {
                        "gpu_available": {"type": "boolean"},
                        "gpu_count": {"type": "integer"},
                        "cpu_count": {"type": "integer"},
                    },
                },
            },
            "required": ["paths", "caching", "training", "analysis"],
        }

    def validate(self, config: Dict[str, Any]) -> bool:
        """JSON 스키마 기반 설정 검증"""
        try:
            # JSON 스키마 검증
            validate(instance=config, schema=self.schema)

            # 추가 비즈니스 로직 검증
            return self._validate_business_rules(config)

        except ValidationError as e:
            logger.error(f"설정 스키마 검증 실패: {e.message}")
            return False
        except Exception as e:
            logger.error(f"설정 검증 중 오류 발생: {e}")
            return False

    def _validate_business_rules(self, config: Dict[str, Any]) -> bool:
        """비즈니스 규칙 검증"""
        try:
            # 벡터 차원 검증
            if "vector" in config:
                vector_config = config["vector"]
                min_dim = vector_config.get("min_required_dimension", 1)
                max_dim = vector_config.get("max_dimension", 1000)

                if min_dim > max_dim:
                    logger.error(
                        f"최소 벡터 차원({min_dim})이 최대 차원({max_dim})보다 큽니다."
                    )
                    return False

            # 학습 설정 검증
            if "training" in config:
                training_config = config["training"]
                batch_size = training_config.get("batch_size", 32)
                learning_rate = training_config.get("learning_rate", 0.001)

                if batch_size > 1024:
                    logger.warning(f"배치 크기가 매우 큽니다: {batch_size}")

                if learning_rate > 0.1:
                    logger.warning(f"학습률이 매우 큽니다: {learning_rate}")

            # 경로 존재성 검증
            if "paths" in config:
                paths_config = config["paths"]
                for path_key, path_value in paths_config.items():
                    if isinstance(path_value, str) and path_key.endswith("_dir"):
                        Path(path_value).mkdir(parents=True, exist_ok=True)

            return True

        except Exception as e:
            logger.error(f"비즈니스 규칙 검증 실패: {e}")
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

    def get_schema(self) -> Dict[str, Any]:
        """JSON 스키마 반환"""
        return self.schema


@dataclass
class CudaConfig:
    """CUDA 최적화 설정"""

    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 2
    batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 256
    optimal_batch_size: Optional[int] = None
    max_memory_usage: float = 0.8
    use_amp: bool = True
    use_cudnn: bool = True
    use_graphs: bool = False
    engine_cache_dir: str = "./data/cache/tensorrt/engines"

    def __post_init__(self):
        self._validate_types()
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if not cuda_available:
            logger.warning("CUDA 비활성화 - CPU 모드")
            self.use_cudnn = False
            self.use_graphs = False
        else:
            self._setup_cuda_optimizations()

        if self.optimal_batch_size is None:
            self.optimal_batch_size = self.batch_size

        if self.min_batch_size > self.max_batch_size:
            self.min_batch_size = min(self.min_batch_size, self.max_batch_size)

        os.makedirs(self.engine_cache_dir, exist_ok=True)

    def _validate_types(self):
        if not isinstance(self.gpu_ids, list):
            raise TypeError("gpu_ids는 리스트여야 합니다.")
        if not all(isinstance(gpu_id, int) and gpu_id >= 0 for gpu_id in self.gpu_ids):
            raise ValueError("gpu_ids의 모든 값은 0 이상의 정수여야 합니다.")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size는 1 이상의 정수여야 합니다.")
        if (
            not isinstance(self.max_memory_usage, (int, float))
            or not 0 < self.max_memory_usage <= 1
        ):
            raise ValueError("max_memory_usage는 0과 1 사이의 숫자여야 합니다.")

    def _setup_cuda_optimizations(self):
        try:
            import torch

            if hasattr(torch.cuda, "memory_pool"):
                torch.cuda.empty_cache()
            if self.use_cudnn:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            logger.debug("CUDA 최적화 설정 완료")
        except Exception as e:
            logger.error(f"CUDA 최적화 설정 실패: {e}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CudaConfig":
        return cls(**data)


@dataclass
class SystemInfo:
    """시스템의 하드웨어 및 소프트웨어 정보를 동적으로 탐지하는 싱글톤 클래스."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.refresh()
        self._initialized = True

    def refresh(self):
        """시스템 정보를 다시 수집하여 최신 상태로 업데이트합니다."""
        self.cpu_count = os.cpu_count() or 1
        self.cpu_usage = psutil.cpu_percent()

        mem = psutil.virtual_memory()
        self.memory_total_gb = mem.total / (1024**3)
        self.memory_available_gb = mem.available / (1024**3)

        try:
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
                self.gpu_devices = []
                for i in range(self.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self.gpu_devices.append(
                        {
                            "name": props.name,
                            "memory_total_mb": props.total_memory / (1024**2),
                        }
                    )
            else:
                self.gpu_count = 0
                self.gpu_devices = []
        except Exception as e:
            logger.warning(f"GPU 정보 탐지 실패: {e}")
            self.gpu_available = False
            self.gpu_count = 0
            self.gpu_devices = []

    def to_dict(self) -> Dict[str, Any]:
        """수집된 시스템 정보를 딕셔너리로 반환합니다."""
        return {
            "cpu_count": self.cpu_count,
            "cpu_usage_percent": self.cpu_usage,
            "memory_total_gb": round(self.memory_total_gb, 2),
            "memory_available_gb": round(self.memory_available_gb, 2),
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu_devices": self.gpu_devices,
        }


class UnifiedConfigManager:
    """동적 시스템 정보를 통합하는 설정 관리자"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.file_timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.config_paths = ConfigPath()
        self.directory_paths = DirectoryPaths()
        self.system_info = SystemInfo()
        self.validator = None  # 검증기 초기화
        logger.info("✅ 통합 설정 관리자 초기화 (동적 시스템 정보 탐지 활성화)")

    def load_config(
        self,
        config_name: str,
        file_path: Optional[str] = None,
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """
        설정 파일을 로드하고, 동적으로 탐지된 시스템 정보를 주입합니다.
        """
        with self._lock:
            if file_path is None:
                file_path = self._get_default_config_path(config_name)

            is_modified = self._is_file_modified(config_name, file_path)

            if (
                not force_reload
                and config_name in self.config_cache
                and not is_modified
            ):
                logger.debug(f"캐시된 설정 반환: {config_name}")
                return self.config_cache[config_name]

            logger.info(
                f"설정 로드 시작: {config_name} (강제 새로고침: {force_reload})"
            )

            # 파일에서 설정 로드
            config_data = self._load_config_file(file_path)

            # 동적 시스템 정보 주입
            self.system_info.refresh()
            config_data["system"] = self.system_info.to_dict()

            # 검증
            if self.validator and not self.validator.validate(config_data):
                raise ValueError(f"'{config_name}' 설정 파일이 유효하지 않습니다.")

            self.config_cache[config_name] = config_data
            self.file_timestamps[config_name] = (
                os.path.getmtime(file_path) if os.path.exists(file_path) else 0
            )
            logger.info(f"✅ 설정 '{config_name}' 로드 및 동적 시스템 정보 통합 완료.")

            return self.config_cache[config_name]

    def _get_default_config_path(self, config_name: str) -> str:
        path_mapping = {
            "main": self.config_paths.main_config,
            "optimization": self.config_paths.optimization_config,
            "pattern_analysis": self.config_paths.pattern_analysis_config,
        }
        return path_mapping.get(config_name, f"config/{config_name}.yaml")

    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            logger.error(f"설정 파일이 존재하지 않습니다: {file_path}")
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith((".yaml", ".yml")):
                    return yaml.safe_load(f) or {}
                elif file_path.endswith(".json"):
                    return json.load(f) or {}
                else:
                    logger.error(f"지원하지 않는 파일 형식: {file_path}")
                    return {}
        except Exception as e:
            logger.error(f"설정 파일 로드 실패 {file_path}: {e}")
            return {}

    def _is_file_modified(
        self, config_name: str, file_path: Optional[str] = None
    ) -> bool:
        if file_path is None:
            file_path = self._get_default_config_path(config_name)

        if not os.path.exists(file_path):
            return False

        current_mtime = os.path.getmtime(file_path)
        cached_mtime = self.file_timestamps.get(config_name, 0)
        return current_mtime > cached_mtime

    def get_config(self, config_name: str = "main") -> "ConfigProxy":
        config = self.load_config(config_name)
        return ConfigProxy(config, config_name, self)

    def get_nested_value(
        self, config: Dict[str, Any], key_path: str, default: Any = None
    ) -> Any:
        keys = key_path.split(".")
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                if default is None:
                    raise KeyError(f"설정 키 누락: {key_path}")
                return default
        return current


class ConfigProxy:
    """설정 프록시 클래스"""

    def __init__(
        self, config: Dict[str, Any], config_name: str, manager: UnifiedConfigManager
    ):
        self.config = config
        self.config_name = config_name
        self.manager = manager

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def get_nested(self, key_path: str, default: Any = None) -> Any:
        return self.manager.get_nested_value(self.config, key_path, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()


def get_config_manager() -> UnifiedConfigManager:
    """설정 관리자 싱글톤 인스턴스 반환"""
    return UnifiedConfigManager()


def load_config(
    config_name: str = "main", file_path: Optional[str] = None
) -> ConfigProxy:
    """설정 로드"""
    manager = get_config_manager()
    return manager.get_config(config_name)


def get_directory_paths() -> DirectoryPaths:
    """디렉토리 경로 반환"""
    manager = get_config_manager()
    return manager.directory_paths


def get_paths() -> DirectoryPaths:
    """디렉토리 경로 반환 (별칭)"""
    return get_directory_paths()


def get_config(config_name: str = "main") -> ConfigProxy:
    """설정 반환"""
    return load_config(config_name)
