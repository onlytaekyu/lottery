"""CUDA 최적화 모듈 (최적화 버전)

이 모듈은 CUDA를 활용한 최적화 클래스들을 제공합니다.
자동 배치 크기 조정, 메모리 효율적인 연산, 연산 스케줄링 등의 기능을 포함합니다.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from threading import Lock, Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from ..utils.config_loader import ConfigProxy
from torch.utils.data import DataLoader

# 통합 설정에서 CudaConfig import
from .unified_config import CudaConfig

# 스레드 로컬 캐시 가져오기
from .memory_manager import (
    MemoryManager,
    MemoryConfig,
    ThreadLocalCache,
    ThreadLocalConfig,
)

# 비동기 IO 관련 임포트
from .async_io import AsyncIOManager, AsyncIOConfig

# 성능 추적 관련 임포트
from .unified_performance import Profiler

# 오류 복구 관련 임포트
from .error_recovery import ErrorRecovery, RecoveryConfig

# 로거 설정
from .error_handler_refactored import get_logger

logger = get_logger(__name__)


class BaseCudaOptimizer:
    """CUDA 최적화 클래스 (통합 버전)"""

    def __init__(self, config: Optional[CudaConfig] = None):
        self.config = config or CudaConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = Lock()
        self._current_model: Optional[nn.Module] = None

        # 메모리 관리자 초기화
        self._memory_manager = MemoryManager(MemoryConfig())

        # 스레드 로컬 캐시 초기화
        self._thread_cache = ThreadLocalCache(
            ThreadLocalConfig(
                max_size=1000,
                max_memory=1 << 30,  # 1GB
                ttl=3600.0,  # 1시간
            )
        )

        # 비동기 IO 관리자 초기화
        self._async_io = AsyncIOManager(
            AsyncIOConfig(
                chunk_size=1 << 20,  # 1MB
                max_concurrent_ops=4,
                compression_level=6,
                compression_threshold=1 << 20,  # 1MB
            )
        )

        # 프로파일러 초기화
        self._profiler = Profiler()

        # 에러 복구 메커니즘 초기화
        self._error_recovery = ErrorRecovery(RecoveryConfig())

        # 기본 설정값
        num_workers = 2
        prefetch_factor = 2

        self._inference_queue: Queue = Queue(maxsize=num_workers * prefetch_factor)
        self._result_queue: Queue = Queue(maxsize=num_workers * prefetch_factor)
        self._worker_threads: List[Any] = []
        self._inference_streams: List[Any] = []
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._cuda_events: List[Any] = []
        self._cuda_graphs: List[Any] = []
        self._custom_kernels: Dict[str, Any] = {}
        self._initialize()

    def _initialize(self):
        """초기화"""
        try:
            with self._profiler.profile("initialize"):
                # CUDA 최적화 기본 설정
                if torch.cuda.is_available():
                    # CUDA 최적화 설정
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

                    # CUDA 스트림 초기화
                    for _ in range(2):
                        stream = torch.cuda.Stream()
                        self._inference_streams.append(stream)

                    # CUDA 이벤트 초기화
                    for _ in range(4):
                        event = torch.cuda.Event(enable_timing=True)
                        self._cuda_events.append(event)

                    # CUDA 그래프 초기화
                    if self.config.use_graphs:
                        self._initialize_cuda_graphs()

            logger.info("CUDA 최적화기 초기화 완료")
        except Exception as e:
            logger.error(f"CUDA 최적화기 초기화 실패: {str(e)}")

    def _initialize_cuda_graphs(self):
        """CUDA 그래프 초기화"""
        try:
            if torch.cuda.is_available():
                # CUDA 그래프 생성
                graph = torch.cuda.CUDAGraph()
                self._cuda_graphs.append(graph)
                logger.info("CUDA 그래프 초기화 완료")
        except Exception as e:
            logger.error(f"CUDA 그래프 초기화 실패: {str(e)}")

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """모델 최적화"""
        try:
            with self._profiler.profile("optimize_model"):
                # AMP 적용
                if self.config.use_amp:
                    model = self._apply_amp(model)

                # 텐서 메모리 최적화
                model = self._optimize_tensor_memory(model)

                # 모델을 GPU로 이동
                if torch.cuda.is_available():
                    model = model.to(self.device)

                self._current_model = model
                return model

        except Exception as e:
            logger.error(f"모델 최적화 실패: {str(e)}")
            return model

    def _apply_amp(self, model: nn.Module) -> nn.Module:
        """AMP(자동 혼합 정밀도) 적용"""
        try:
            if torch.cuda.is_available() and self.config.use_amp:
                # 모델을 half precision으로 변환
                model = model.half()
                logger.info("AMP 적용 완료")
            return model
        except Exception as e:
            logger.error(f"AMP 적용 실패: {str(e)}")
            return model

    def _optimize_tensor_memory(self, model: nn.Module) -> nn.Module:
        """텐서 메모리 최적화"""
        try:
            # 메모리 압축 및 최적화
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.contiguous()

            logger.info("텐서 메모리 최적화 완료")
            return model
        except Exception as e:
            logger.error(f"텐서 메모리 최적화 실패: {str(e)}")
            return model

    def optimize_inference(
        self, model: nn.Module, input_data: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """추론 최적화"""
        try:
            with self._profiler.profile("optimize_inference"):
                # 입력 데이터를 GPU로 이동
                if torch.cuda.is_available():
                    input_data = input_data.to(self.device)

                # 동기 추론 실행
                return self._run_sync_inference(input_data)

        except Exception as e:
            logger.error(f"추론 최적화 실패: {str(e)}")
            return None

    def _run_sync_inference(self, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """동기 추론 실행"""
        try:
            if self._current_model is None:
                logger.error("모델이 설정되지 않았습니다.")
                return None

            with torch.no_grad():
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self._current_model(input_tensor)
                else:
                    output = self._current_model(input_tensor)

            return output

        except Exception as e:
            logger.error(f"동기 추론 실패: {str(e)}")
            return None

    def cleanup(self):
        """리소스 정리"""
        try:
            # 스레드 풀 정리
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=True)

            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 메모리 관리자 정리
            if hasattr(self, "_memory_manager"):
                self._memory_manager.cleanup()

            logger.info("CUDA 최적화기 정리 완료")

        except Exception as e:
            logger.error(f"CUDA 최적화기 정리 실패: {str(e)}")

    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except Exception:
            pass


class AMPTrainer:
    """AMP 트레이너 (통합 버전)"""

    def __init__(self, config: ConfigProxy):
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    @property
    def use_amp(self) -> bool:
        """AMP 사용 여부"""
        return (
            self.config["training"]["use_amp"]
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

    def train_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """훈련 스텝"""
        try:
            if self.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(batch["input"])
                    loss = loss_fn(outputs, batch["target"])

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(batch["input"])
                loss = loss_fn(outputs, batch["target"])
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            return {"loss": loss.item()}

        except Exception as e:
            logger.error(f"훈련 스텝 실패: {str(e)}")
            return {"loss": float("inf")}

    def evaluate(
        self, model: torch.nn.Module, loss_fn: Callable, dataloader: DataLoader
    ) -> Dict[str, float]:
        """평가"""
        try:
            model.eval()
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch in dataloader:
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch["input"])
                            loss = loss_fn(outputs, batch["target"])
                    else:
                        outputs = model(batch["input"])
                        loss = loss_fn(outputs, batch["target"])

                    total_loss += loss.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
            return {"avg_loss": avg_loss}

        except Exception as e:
            logger.error(f"평가 실패: {str(e)}")
            return {"avg_loss": float("inf")}

    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        try:
            if torch.cuda.is_available():
                return {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved(),
                }
            else:
                return {"device": "cpu"}
        except Exception as e:
            logger.error(f"디바이스 정보 조회 실패: {str(e)}")
            return {"error": str(e)}

    def get_device(self) -> torch.device:
        """디바이스 반환"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def optimize_memory(self) -> None:
        """메모리 최적화"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.error(f"메모리 최적화 실패: {str(e)}")


# 유틸리티 함수들
def get_device_info() -> Dict[str, Any]:
    """디바이스 정보 반환"""
    try:
        if torch.cuda.is_available():
            return {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
                "memory_cached": torch.cuda.memory_cached(),
            }
        else:
            return {"device": "cpu"}
    except Exception as e:
        logger.error(f"디바이스 정보 조회 실패: {str(e)}")
        return {"error": str(e)}


def optimize_memory():
    """메모리 최적화"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.error(f"메모리 최적화 실패: {str(e)}")


def setup_cuda_memory_pool():
    """CUDA 메모리 풀 설정"""
    try:
        if torch.cuda.is_available():
            # 메모리 풀 설정
            torch.cuda.empty_cache()

            # 메모리 할당 전략 설정
            torch.cuda.set_per_process_memory_fraction(0.8)

            logger.info("CUDA 메모리 풀 설정 완료")
    except Exception as e:
        logger.error(f"CUDA 메모리 풀 설정 실패: {str(e)}")


def get_cuda_memory_info() -> Dict[str, Any]:
    """CUDA 메모리 정보 반환"""
    try:
        if torch.cuda.is_available():
            return {
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
                "memory_cached": torch.cuda.memory_cached(),
                "max_memory_allocated": torch.cuda.max_memory_allocated(),
                "max_memory_reserved": torch.cuda.max_memory_reserved(),
            }
        else:
            return {"device": "cpu"}
    except Exception as e:
        logger.error(f"CUDA 메모리 정보 조회 실패: {str(e)}")
        return {"error": str(e)}
