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
    """통합 CUDA 최적화 설정 클래스 - 타입 안전성 강화"""

    # 기본 GPU 설정
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 2

    # 배치 크기 관련 설정
    batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 256
    optimal_batch_size: Optional[int] = None

    # 메모리 설정
    max_memory_usage: float = 0.8
    output_size: Optional[List[int]] = None
    input_size: Optional[List[int]] = None
    normalize_inputs: bool = False

    # AMP (자동 혼합 정밀도) 관련 설정
    use_amp: bool = True
    fp16_mode: bool = False

    # cuDNN 관련 설정
    use_cudnn: bool = True

    # CUDA 그래프 관련 설정
    use_graphs: bool = False

    # TensorRT 캐시 설정
    engine_cache_dir: str = "./data/cache/tensorrt/engines"
    onnx_cache_dir: str = "./data/cache/tensorrt/onnx"
    calibration_cache_dir: str = "./data/cache/tensorrt/calibration"
    engine_cache_prefix: str = "engine"
    engine_cache_suffix: str = ".trt"
    engine_cache_version: str = "v1"
    onnx_cache_prefix: str = "model"
    onnx_cache_suffix: str = ".onnx"
    onnx_cache_version: str = "v1"

    def __post_init__(self):
        """초기화 후처리 - 타입 검증 포함"""
        # 타입 검증
        self._validate_types()

        # CUDA 사용 가능성 확인
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if not cuda_available:
            logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 설정됩니다.")
            self.use_cudnn = False
            self.use_graphs = False
        else:
            # CUDA 최적화 설정 적용
            self._setup_cuda_optimizations()

        # optimal_batch_size가 설정되지 않은 경우 batch_size 값 사용
        if self.optimal_batch_size is None:
            self.optimal_batch_size = self.batch_size

        # 배치 크기 검증
        if self.min_batch_size > self.max_batch_size:
            logger.warning(
                "min_batch_size가 max_batch_size보다 큽니다. 값을 조정합니다."
            )
            self.min_batch_size = min(self.min_batch_size, self.max_batch_size)

        # 디렉토리 경로를 절대 경로로 변환
        for dir_path in ["engine_cache_dir", "onnx_cache_dir", "calibration_cache_dir"]:
            current_path = getattr(self, dir_path)
            if isinstance(current_path, str) and not os.path.isabs(current_path):
                # 상대 경로를 절대 경로로 변환
                new_path = Path(__file__).parent.parent.parent / current_path
                setattr(self, dir_path, str(new_path))

        # 디렉토리 생성
        for dir_path in [
            self.engine_cache_dir,
            self.onnx_cache_dir,
            self.calibration_cache_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    def _validate_types(self):
        """타입 검증"""
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
        """CUDA 최적화 설정"""
        try:
            import torch

            # GPU 메모리 풀 설정
            if hasattr(torch.cuda, "memory_pool"):
                torch.cuda.empty_cache()

            # cuDNN 최적화
            if self.use_cudnn:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            logger.debug("CUDA 최적화 설정 완료 (unified_config)")
        except Exception as e:
            logger.error(f"CUDA 최적화 설정 실패: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CudaConfig":
        """딕셔너리로부터 생성"""
        return cls(**data)


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

        # 동적 설정 변경 지원
        self._change_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._change_history: List[ConfigChange] = []
        self._max_history_size = 1000

        # 기본 경로 설정
        self.config_paths = ConfigPath()
        self.directory_paths = DirectoryPaths()

        # 기본 검증기 등록
        self.register_validator("main", EnhancedConfigValidator())

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

    def add_change_listener(
        self, config_name: str, callback: Callable[[ConfigChange], None]
    ):
        """설정 변경 리스너 추가"""
        self._change_listeners[config_name].append(callback)
        self.logger.debug(f"설정 변경 리스너 추가: {config_name}")

    def remove_change_listener(
        self, config_name: str, callback: Callable[[ConfigChange], None]
    ):
        """설정 변경 리스너 제거"""
        if config_name in self._change_listeners:
            try:
                self._change_listeners[config_name].remove(callback)
                self.logger.debug(f"설정 변경 리스너 제거: {config_name}")
            except ValueError:
                pass

    def update_config_value(
        self, config_name: str, key_path: str, new_value: Any
    ) -> bool:
        """설정 값 동적 업데이트"""
        try:
            with self._lock:
                if config_name not in self._configs:
                    self.load_config(config_name)

                config = self._configs[config_name]
                old_value = self.get_nested_value(config, key_path, None)

                # 중첩된 키 경로로 값 설정
                keys = key_path.split(".")
                current = config

                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                current[keys[-1]] = new_value

                # 변경 기록
                change = ConfigChange(
                    change_type=(
                        ConfigChangeType.MODIFIED
                        if old_value is not None
                        else ConfigChangeType.ADDED
                    ),
                    key_path=key_path,
                    old_value=old_value,
                    new_value=new_value,
                )

                self._record_change(config_name, change)
                self.logger.info(
                    f"설정 값 업데이트: {config_name}.{key_path} = {new_value}"
                )

                return True

        except Exception as e:
            self.logger.error(f"설정 값 업데이트 실패: {config_name}.{key_path} - {e}")
            return False

    def _record_change(self, config_name: str, change: ConfigChange):
        """설정 변경 기록"""
        # 변경 히스토리 추가
        self._change_history.append(change)

        # 히스토리 크기 제한
        if len(self._change_history) > self._max_history_size:
            self._change_history = self._change_history[-self._max_history_size :]

        # 리스너들에게 알림
        for callback in self._change_listeners.get(config_name, []):
            try:
                callback(change)
            except Exception as e:
                self.logger.error(f"설정 변경 리스너 호출 실패: {e}")

    def get_change_history(
        self, config_name: Optional[str] = None, limit: int = 100
    ) -> List[ConfigChange]:
        """설정 변경 히스토리 조회"""
        if config_name is None:
            return self._change_history[-limit:]
        else:
            # 특정 설정의 변경만 필터링
            filtered = [
                change
                for change in self._change_history
                if change.key_path.startswith(f"{config_name}.")
            ]
            return filtered[-limit:]

    def validate_config_change(
        self, config_name: str, key_path: str, new_value: Any
    ) -> bool:
        """설정 변경 사전 검증"""
        try:
            if config_name not in self._validators:
                return True  # 검증기가 없으면 허용

            # 임시 설정으로 검증
            temp_config = self._configs.get(config_name, {}).copy()
            keys = key_path.split(".")
            current = temp_config

            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            current[keys[-1]] = new_value

            return self._validators[config_name].validate(temp_config)

        except Exception as e:
            self.logger.error(f"설정 변경 검증 실패: {e}")
            return False

    def get_config_schema(self, config_name: str) -> Optional[Dict[str, Any]]:
        """설정 스키마 반환"""
        if config_name in self._validators:
            return self._validators[config_name].get_schema()
        return None

    def export_config(
        self, config_name: str, file_path: str, format: str = "yaml"
    ) -> bool:
        """설정을 파일로 내보내기"""
        try:
            if config_name not in self._configs:
                self.load_config(config_name)

            config = self._configs[config_name]

            with open(file_path, "w", encoding="utf-8") as f:
                if format.lower() == "yaml":
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                elif format.lower() == "json":
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"지원되지 않는 형식: {format}")

            self.logger.info(f"설정 내보내기 완료: {config_name} -> {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"설정 내보내기 실패: {e}")
            return False


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
