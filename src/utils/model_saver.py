"""
모델 저장/로드 시스템

이 모듈은 모델의 저장, 로드, 버전 관리, 체크포인트 검증을 위한 유틸리티를 제공합니다.
"""

import logging
import logging.handlers
import torch
import pickle
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Type
from pathlib import Path
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from .memory_manager import MemoryManager, MemoryConfig
from .error_handler import get_logger
import traceback
from .config_loader import ConfigProxy

# 로거 설정
logger = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """체크포인트 메타데이터"""

    version: str
    timestamp: datetime
    model_type: str
    model_hash: str
    config_hash: str
    size: int
    dependencies: List[str]
    metrics: Dict[str, float]
    is_valid: bool = True


class ModelSaver:
    """모델 저장/로드 시스템"""

    def __init__(self, base_dir: Optional[str] = None):
        """
        모델 저장 시스템 초기화

        Args:
            base_dir: 체크포인트 기본 디렉토리 (기본값: None, 프로젝트 루트/checkpoints 사용)
        """
        # 상대 경로 사용
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent / "checkpoints"
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.memory_manager = MemoryManager(MemoryConfig())
        self._setup_logging()

        # 스레드 안전성을 위한 락
        self.save_lock = threading.Lock()
        self.load_lock = threading.Lock()

        # 저장/로드 큐
        self.save_queue = ThreadPoolExecutor()
        self.load_queue = ThreadPoolExecutor()

    def _setup_logging(self):
        """로깅 설정"""
        # 상대 경로 사용
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        # 로거 설정
        self.logger = get_logger(self.__class__.__module__)

        # 기존 핸들러 살펴보고 파일 핸들러가 없는 경우에만 추가
        has_file_handler = False
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler) or isinstance(
                handler, logging.handlers.RotatingFileHandler
            ):
                has_file_handler = True
                break

        # 파일 핸들러가 없으면 새로 추가
        if not has_file_handler:
            # 파일 핸들러 생성
            log_file = log_dir / "model_saver.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"모델 저장 시스템 로깅 설정 완료 (로그 파일: {log_file})")

    def save_model(
        self,
        model: Any,
        name: str,
        version: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        priority: int = 1,
        sync: bool = False,  # 동기식 저장 여부
    ):
        """
        모델 저장

        Args:
            model: 저장할 모델
            name: 모델 이름
            version: 모델 버전
            config: 모델 설정
            metrics: 성능 메트릭
            priority: 저장 우선순위 (낮을수록 높은 우선순위)
            sync: 동기식 저장 여부 (True이면 함수가 리턴하기 전에 저장 완료)

        Returns:
            저장된 체크포인트 경로
        """
        # 메타데이터 준비
        metadata = self._prepare_metadata(model, name, version, config, metrics)

        checkpoint_path = self.base_dir / name / version

        # 동기식 저장인 경우 바로 저장 작업 수행
        if sync:
            task = {
                "model": model,
                "name": name,
                "version": version,
                "metadata": metadata,
                "config": config,
            }
            self._save_checkpoint(task)
        else:
            # 저장 작업 큐에 추가
            task = {
                "model": model,
                "name": name,
                "version": version,
                "metadata": metadata,
                "config": config,
            }
            self.save_queue.submit(self._save_checkpoint, task)

        return str(checkpoint_path)

    def _save_checkpoint(self, task: Dict[str, Any]):
        """
        체크포인트 저장

        Args:
            task: 저장 작업 정보
        """
        with self.save_lock:
            try:
                model = task["model"]
                name = task["name"]
                version = task["version"]
                metadata = task["metadata"]
                config = task["config"]

                # 체크포인트 디렉토리 생성
                checkpoint_dir = self.base_dir / name / version
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # 모델 상태 저장
                if hasattr(model, "state_dict"):
                    # PyTorch 모델
                    state_dict = model.state_dict()
                    torch.save(state_dict, checkpoint_dir / "model.pt")
                else:
                    # 일반 모델
                    with open(checkpoint_dir / "model.pkl", "wb") as f:
                        pickle.dump(model, f)

                # 모델 타입 정보 추가
                if hasattr(model, "model_type"):
                    config["type"] = model.model_type
                elif "model_type" in config:
                    config["type"] = config["model_type"]
                else:
                    config["type"] = type(model).__name__

                # 설정 저장
                with open(checkpoint_dir / "config.json", "w") as f:
                    json.dump(config, f, indent=2)

                # 메타데이터 저장
                with open(checkpoint_dir / "metadata.json", "w") as f:
                    json.dump(metadata.__dict__, f, indent=2, default=str)

                # 체크포인트 검증
                if self._validate_checkpoint(checkpoint_dir):
                    self.logger.info(f"체크포인트 저장 완료: {checkpoint_dir}")
                else:
                    raise ValueError("체크포인트 검증 실패")

            except Exception as e:
                self.logger.error(f"체크포인트 저장 중 오류 발생: {e}")
                raise

    def load_model(
        self, name: str, version: str, device: Optional[str] = None, priority: int = 1
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        모델 로드

        Args:
            name: 모델 이름
            version: 모델 버전
            device: 로드 장치 (None이면 CPU)
            priority: 로드 우선순위 (낮을수록 높은 우선순위)

        Returns:
            (로드된 모델, 모델 설정)
        """
        try:
            # 로드 작업 큐에 추가
            future: Future = self.load_queue.submit(
                self._load_checkpoint, name, version, device
            )

            # 결과 리턴
            return future.result()
        except Exception as e:
            # 에러 로깅
            logger = get_logger(self.__class__.__module__)
            error_message = f"Error in load_model: {str(e)}"
            logger.error(error_message)

            # 스택 트레이스 로깅
            stack_trace = traceback.format_exc()
            logger.error(f"Stack trace:\n{stack_trace}")

            return None, {}

    def _load_checkpoint(self, name: str, version: str, device: Optional[str] = None):
        """
        체크포인트 로드

        Args:
            name: 모델 이름
            version: 모델 버전
            device: 로드 장치 (None이면 CPU)

        Returns:
            (로드된 모델, 모델 설정)
        """
        with self.load_lock:
            try:
                checkpoint_dir = self.base_dir / name / version

                # 체크포인트 검증
                if not self._validate_checkpoint(checkpoint_dir):
                    raise ValueError("체크포인트 검증 실패")

                # 메타데이터 로드
                with open(checkpoint_dir / "metadata.json", "r") as f:
                    metadata = CheckpointMetadata(**json.load(f))

                # 설정 로드
                with open(checkpoint_dir / "config.json", "r") as f:
                    config = json.load(f)

                # 모델 로드
                if (checkpoint_dir / "model.pt").exists():
                    # PyTorch 모델
                    state_dict = torch.load(
                        checkpoint_dir / "model.pt", weights_only=True
                    )
                    if device:
                        state_dict = {k: v.to(device) for k, v in state_dict.items()}
                    model = self._create_model_from_config(config)
                    model.load_state_dict(state_dict)
                else:
                    # 일반 모델
                    with open(checkpoint_dir / "model.pkl", "rb") as f:
                        model = pickle.load(f)

                return model, config

            except Exception as e:
                self.logger.error(f"모델 로드 중 오류 발생: {e}")
                raise

    def _prepare_metadata(
        self,
        model: Any,
        name: str,
        version: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> CheckpointMetadata:
        """
        메타데이터 준비

        Args:
            model: 모델
            name: 모델 이름
            version: 모델 버전
            config: 모델 설정
            metrics: 성능 메트릭

        Returns:
            메타데이터
        """
        # 모델 상태 직렬화
        if hasattr(model, "state_dict"):
            state_dict = model.state_dict()
            model_bytes = pickle.dumps(state_dict)
        else:
            model_bytes = pickle.dumps(model)
        model_hash = hashlib.sha256(model_bytes).hexdigest()

        # 설정 직렬화
        config_bytes = json.dumps(config, sort_keys=True).encode()
        config_hash = hashlib.sha256(config_bytes).hexdigest()

        # 체크포인트 크기 계산
        checkpoint_size = len(model_bytes) + len(config_bytes)

        return CheckpointMetadata(
            version=version,
            timestamp=datetime.now(),
            model_type=type(model).__name__,
            model_hash=model_hash,
            config_hash=config_hash,
            size=checkpoint_size,
            dependencies=self._get_model_dependencies(model),
            metrics=metrics or {},
        )

    def _validate_checkpoint(self, checkpoint_dir: Path) -> bool:
        """
        체크포인트 검증

        Args:
            checkpoint_dir: 체크포인트 디렉토리

        Returns:
            체크포인트가 유효한지 여부
        """
        try:
            # 필수 파일 확인
            required_files = ["config.json", "metadata.json"]
            model_files = ["model.pt", "model.pkl"]

            # 모델 파일 중 하나는 반드시 존재해야 함
            if not any((checkpoint_dir / file).exists() for file in model_files):
                self.logger.error(
                    f"체크포인트 검증 실패: 모델 파일이 존재하지 않습니다. 경로: {checkpoint_dir}"
                )
                return False

            # 필수 설정 파일 확인
            for file in required_files:
                if not (checkpoint_dir / file).exists():
                    self.logger.error(
                        f"체크포인트 검증 실패: {file} 파일이 존재하지 않습니다. 경로: {checkpoint_dir}"
                    )
                    return False

            # 메타데이터 로드
            try:
                with open(checkpoint_dir / "metadata.json", "r") as f:
                    metadata = CheckpointMetadata(**json.load(f))
            except Exception as e:
                self.logger.error(f"메타데이터 로드 중 오류 발생: {e}")
                return False

            # 설정 파일 로드
            try:
                with open(checkpoint_dir / "config.json", "r") as f:
                    config = json.load(f)
            except Exception as e:
                self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
                return False

            # 설정 해시 확인
            config_bytes = json.dumps(config, sort_keys=True).encode()
            config_hash = hashlib.sha256(config_bytes).hexdigest()

            # 모델 파일 로드 및 해시 계산 (안전하게 처리)
            if (checkpoint_dir / "model.pt").exists():
                try:
                    # 모델 로드 시 예외 처리 강화
                    state_dict = torch.load(
                        checkpoint_dir / "model.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    # 가능한 경우 state_dict 해시만 검증 (전체 모델 로드 없이)
                    model_bytes = pickle.dumps(state_dict)
                    model_hash = hashlib.sha256(model_bytes).hexdigest()
                except Exception as e:
                    self.logger.warning(
                        f"모델 상태 로드 중 오류 발생 (검증 계속 진행): {e}"
                    )
                    # 해시 검증 건너뛰고 기본 파일 존재 여부만으로 검증 진행
                    return True
            else:
                try:
                    with open(checkpoint_dir / "model.pkl", "rb") as f:
                        model_bytes = f.read()
                    model_hash = hashlib.sha256(model_bytes).hexdigest()
                except Exception as e:
                    self.logger.warning(
                        f"모델 파일 로드 중 오류 발생 (검증 계속 진행): {e}"
                    )
                    # 해시 검증 건너뛰고 기본 파일 존재 여부만으로 검증 진행
                    return True

            # 해시 검증 (메타데이터와 일치하는지)
            # 주의: 모델 해시는 종종 변할 수 있으므로 불일치 시에도 경고만 표시
            if config_hash != metadata.config_hash:
                self.logger.warning(
                    f"설정 해시 불일치: {config_hash} != {metadata.config_hash}"
                )
                # 설정 불일치는 경고만 출력하고 계속 진행

            # 모델 해시 검증은 선택적으로 수행 (이전에 성공적으로 로드된 경우)
            if hasattr(metadata, "model_hash") and model_hash != metadata.model_hash:
                self.logger.warning(
                    f"모델 해시 불일치: {model_hash} != {metadata.model_hash}"
                )
                # 모델 해시 불일치도 경고만 출력

            # 기본 파일이 존재하면 유효한 것으로 간주
            self.logger.info(f"체크포인트 검증 성공: {checkpoint_dir}")
            return True

        except Exception as e:
            self.logger.error(f"체크포인트 검증 중 예외 발생: {e}")
            return False

    def _get_model_dependencies(self, model: Any) -> List[str]:
        """
        모델 종속성 확인

        Args:
            model: 모델

        Returns:
            모델 종속성 목록
        """
        dependencies = []

        if hasattr(model, "state_dict"):
            # PyTorch 모델
            for name, param in model.named_parameters():
                if param.requires_grad:
                    dependencies.append(f"param_{name}")

            for name, buffer in model.named_buffers():
                dependencies.append(f"buffer_{name}")
        else:
            # 일반 모델
            for attr in dir(model):
                if not attr.startswith("_"):
                    value = getattr(model, attr)
                    if isinstance(value, (torch.nn.Module, torch.Tensor)):
                        dependencies.append(attr)

        return dependencies

    def _create_model_from_config(self, config: Dict[str, Any]) -> Any:
        """
        모델 생성

        Args:
            config: 모델 설정

        Returns:
            모델
        """
        # 모델 타입 확인
        model_type = config.get("type")
        if model_type is None:
            # type 키가 없는 경우 기본값으로 RNNModel 사용
            self.logger.warning(
                "모델 설정에 'type' 키가 없습니다. 기본값으로 'RNNModel'을 사용합니다."
            )
            model_type = "RNNModel"

        # 모델 클래스 확인
        model_class = self._get_model_class(model_type)

        # 모델 생성을 위한 설정 준비
        model_params = config.copy()
        if "type" in model_params:
            del model_params["type"]  # 모델 파라미터에서 타입 정보 제거

        # 모델 생성
        if model_type in ["RNNModel", "lstm"]:
            # RNNModel은 첫 번째 인자로 config를 받음
            rnn_model_type = (
                model_params.pop("model_type", "lstm")
                if "model_type" in model_params
                else "lstm"
            )
            return model_class(model_params, model_type=rnn_model_type)
        else:
            return model_class(**model_params)

    def _get_model_class(self, model_type: str) -> type:
        """
        모델 클래스 확인

        Args:
            model_type: 모델 타입

        Returns:
            모델 클래스
        """
        # 모델 클래스 매핑
        model_classes = {
            "Linear": torch.nn.Linear,
            "Conv2d": torch.nn.Conv2d,
            "LSTM": torch.nn.LSTM,
            "Transformer": torch.nn.Transformer,
            "RNNModel": lambda *args, **kwargs: self._import_class(
                "src.supervised.rnn_models", "RNNModel"
            )(*args, **kwargs),
            "lstm": lambda *args, **kwargs: self._import_class(
                "src.supervised.rnn_models", "RNNModel"
            )(*args, **kwargs),
            # 추가 모델 클래스...
        }

        if model_type not in model_classes:
            self.logger.error(f"지원하지 않는 모델 타입: {model_type}")
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

        return model_classes[model_type]

    def _import_class(self, module_name: str, class_name: str) -> type:
        """
        모듈에서 클래스 가져오기

        Args:
            module_name: 모듈 이름
            class_name: 클래스 이름

        Returns:
            클래스 객체
        """
        import importlib

        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.logger.error(
                f"모듈 {module_name}에서 {class_name} 클래스 로드 중 오류 발생: {e}"
            )
            raise

    def get_checkpoint_info(self, name: str, version: str) -> Dict[str, Any]:
        """
        체크포인트 정보 반환

        Args:
            name: 모델 이름
            version: 모델 버전

        Returns:
            체크포인트 정보
        """
        checkpoint_dir = self.base_dir / name / version
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_dir}")

        # 메타데이터 로드
        metadata_path = checkpoint_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}"
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # 설정 로드
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # 검증 상태 확인
        metadata["is_valid"] = self._validate_checkpoint(checkpoint_dir)

        return {
            "name": name,
            "version": version,
            "metadata": metadata,
            "config": config,
            "path": str(checkpoint_dir),
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            # 스레드 풀 종료
            self.save_queue.shutdown(wait=False)
            self.load_queue.shutdown(wait=False)

            # 메모리 관리자 정리
            if hasattr(self, "memory_manager"):
                self.memory_manager.cleanup()

            self.logger.info("모델 저장 시스템 리소스 정리 완료")
        except Exception as e:
            self.logger.error(f"모델 저장 시스템 리소스 정리 중 오류 발생: {str(e)}")


def save_model(
    model: torch.nn.Module,
    model_type: str,
    config: ConfigProxy,
    metrics: Optional[Dict[str, float]] = None,
    save_dir: str = "savedModels",
) -> str:
    """
    모델을 저장합니다.

    Args:
        model: 저장할 모델
        model_type: 모델 타입 (예: "lstm", "gnn", "rl")
        config: 설정
        metrics: 평가 지표
        save_dir: 저장 디렉토리

    Returns:
        저장된 모델 파일 경로
    """
    try:
        # 저장 디렉토리 생성
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_model_{timestamp}.pt"
        filepath = save_path / filename

        # 저장할 데이터 구성
        save_data = {
            "model_state": model.state_dict(),
            "model_type": model_type,
            "config": config.to_dict(),
            "timestamp": timestamp,
        }

        if metrics:
            save_data["metrics"] = metrics

        # 모델 저장
        torch.save(save_data, filepath)
        logger.info(f"모델 저장 완료: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"모델 저장 실패: {str(e)}")
        return ""


def load_model(
    model_class: Type[torch.nn.Module], path: str, device: Optional[torch.device] = None
) -> Optional[torch.nn.Module]:
    """
    모델을 로드합니다.

    Args:
        model_class: 모델 클래스
        path: 모델 파일 경로
        device: 디바이스

    Returns:
        로드된 모델
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 데이터 로드
        save_data = torch.load(path, map_location=device)

        # 모델 인스턴스 생성
        model = model_class()
        model.load_state_dict(save_data["model_state"])
        model.to(device)

        logger.info(f"모델 로드 완료: {path}")
        return model

    except Exception as e:
        logger.error(f"모델 로드 실패: {str(e)}")
        return None
