"""CUDA 최적화 모듈

이 모듈은 CUDA를 활용한 최적화 클래스들을 제공합니다.
자동 배치 크기 조정, 메모리 효율적인 연산, 연산 스케줄링 등의 기능을 포함합니다.
"""

import time
import torch
import torch.nn as nn
import gc
import psutil
import logging
import os
import platform
import sys
import traceback
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from threading import Lock, Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from ..utils.config_loader import ConfigProxy
from torch.utils.data import DataLoader

# 스레드 로컬 캐시 가져오기 (memory_manager에서 가져오도록 변경)
from .memory_manager import (
    MemoryManager,
    MemoryConfig,
    ThreadLocalCache,
    ThreadLocalConfig,
)

# 비동기 IO 관련 임포트
from .async_io import AsyncIOManager, AsyncIOConfig

# 성능 추적 관련 임포트
from .unified_performance import get_profiler

# 오류 복구 관련 임포트
from .error_recovery import ErrorRecovery, RecoveryConfig

# 로거 설정
from .error_handler_refactored import get_logger

logger = get_logger(__name__)


@dataclass
class CudaConfig:
    """CUDA 최적화 설정"""

    # 배치 크기 관련 설정
    batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 256
    optimal_batch_size: Optional[int] = None

    # AMP (자동 혼합 정밀도) 관련 설정
    use_amp: bool = True  # 자동 혼합 정밀도 사용 여부

    # cuDNN 관련 설정
    use_cudnn: bool = True  # cuDNN 사용 여부

    # CUDA 그래프 관련 설정
    use_graphs: bool = False  # CUDA 그래프 사용 여부

    # 추가 설정들은 필요한 경우 추가할 수 있습니다.

    def __post_init__(self):
        """설정 검증"""
        if not torch.cuda.is_available():
            logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 설정됩니다.")
            self.use_amp = False
            self.use_cudnn = False
            self.use_graphs = False

        # optimal_batch_size가 설정되지 않은 경우 batch_size 값 사용
        if self.optimal_batch_size is None:
            self.optimal_batch_size = self.batch_size


class BaseCudaOptimizer:
    """CUDA 최적화 클래스"""

    def __init__(self, config: Optional[CudaConfig] = None):
        self.config = config or CudaConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = Lock()
        self._current_model: Optional[nn.Module] = None

        # 메모리 관리자 초기화 - 기본 MemoryConfig 사용
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
        self._profiler = Profiler(
            ProfilerConfig(
                enable_cpu_profiling=True,
                enable_memory_profiling=True,
                enable_thread_profiling=True,
                sampling_interval=0.1,
                memory_threshold=1 << 30,  # 1GB
                cpu_threshold=80.0,
                thread_threshold=100,
            )
        )

        # 에러 복구 메커니즘 초기화
        self._error_recovery = ErrorRecovery(RecoveryConfig())  # 기본 매개변수 사용

        # 기본 설정값
        num_workers = 2
        prefetch_factor = 2
        num_streams = 2

        self._inference_queue: Queue = Queue(maxsize=num_workers * prefetch_factor)
        self._result_queue: Queue = Queue(maxsize=num_workers * prefetch_factor)
        self._worker_threads: List[Any] = []  # type: ignore
        self._inference_streams: List[Any] = []  # type: ignore
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._cuda_events: List[Any] = []  # type: ignore
        self._cuda_graphs: List[Any] = []  # type: ignore
        self._custom_kernels: Dict[str, Any] = {}  # type: ignore
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
                    for _ in range(2):  # 기본 스트림 수 2개로 설정
                        stream = torch.cuda.Stream()
                        self._inference_streams.append(stream)

                    # CUDA 이벤트 초기화
                    for _ in range(4):  # 기본 이벤트 수 4개로 설정
                        event = torch.cuda.Event(enable_timing=True)
                        self._cuda_events.append(event)

                    # CUDA 그래프 초기화
                    if self.config.use_graphs:
                        self._initialize_cuda_graphs()

            logger.info("CUDA 최적화기 초기화 완료")
        except Exception as e:
            logger.error(f"CUDA 최적화기 초기화 실패: {str(e)}")

    def _initialize_custom_kernels(self):
        """커스텀 CUDA 커널 초기화"""
        try:
            # 예시: 행렬 곱셈 커널
            if "matmul" not in self._custom_kernels:
                self._custom_kernels["matmul"] = self._create_matmul_kernel()

            # 예시: 텐서 연산 커널
            if "tensor_ops" not in self._custom_kernels:
                self._custom_kernels["tensor_ops"] = self._create_tensor_ops_kernel()

        except Exception as e:
            logger.error(f"커스텀 CUDA 커널 초기화 실패: {str(e)}")

    def _create_matmul_kernel(self) -> Any:  # type: ignore
        """행렬 곱셈 커널 생성"""
        try:
            # CUDA 커널 코드
            kernel_code = """
            __global__ void matmul_kernel(
                const float* A,
                const float* B,
                float* C,
                const int M,
                const int N,
                const int K
            ) {
                const int row = blockIdx.y * blockDim.y + threadIdx.y;
                const int col = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (row < M && col < N) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += A[row * K + k] * B[k * N + col];
                    }
                    C[row * N + col] = sum;
                }
            }
            """
            return self._compile_kernel(kernel_code, "matmul_kernel")
        except Exception as e:
            logger.error(f"행렬 곱셈 커널 생성 실패: {str(e)}")
            return None

    def _create_tensor_ops_kernel(self) -> Any:  # type: ignore
        """텐서 연산 커널 생성"""
        try:
            # CUDA 커널 코드
            kernel_code = """
            __global__ void tensor_ops_kernel(
                const float* input,
                float* output,
                const int size,
                const float scale
            ) {
                const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size) {
                    output[idx] = input[idx] * scale;
                }
            }
            """
            return self._compile_kernel(kernel_code, "tensor_ops_kernel")
        except Exception as e:
            logger.error(f"텐서 연산 커널 생성 실패: {str(e)}")
            return None

    def _compile_kernel(self, kernel_code: str, kernel_name: str) -> Any:  # type: ignore
        """커널 컴파일"""
        try:
            # CUDA 커널 컴파일
            from numba import cuda

            return cuda.jit(kernel_code)
        except Exception as e:
            logger.error(f"커널 컴파일 실패: {str(e)}")
            return None

    def _initialize_cuda_graphs(self):
        """CUDA 그래프 초기화"""
        try:
            if not self.config.use_graphs or not torch.cuda.is_available():
                return

            # CUDA 그래프 초기화
            for _ in range(2):  # 기본 그래프 수 2개로 설정
                graph = torch.cuda.CUDAGraph()
                self._cuda_graphs.append(graph)

            logger.info("CUDA 그래프 초기화 완료")
        except Exception as e:
            logger.error(f"CUDA 그래프 초기화 실패: {str(e)}")

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """모델 최적화"""
        try:
            with self._lock:
                with self._profiler.profile("optimize_model"):
                    # 메모리 사용량 확인
                    memory_usage = self._memory_manager.get_memory_usage()

                    # AMP 적용 (자동 혼합 정밀도)
                    if self.config.use_amp:
                        model = self._apply_amp(model)

                    # cuDNN 최적화 적용
                    if self.config.use_cudnn:
                        # cuDNN 벤치마크 및 설정
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cudnn.deterministic = False
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True

                    # CUDA 그래프 적용
                    if self.config.use_graphs and torch.cuda.is_available():
                        # 고급 메모리 최적화 적용 (메모리 여유가 있는 경우)
                        if memory_usage < 0.7:  # 메모리 사용률이 70% 미만인 경우
                            try:
                                # 텐서 메모리 관리 최적화
                                self._optimize_tensor_memory(model)
                            except Exception as e:
                                logger.warning(f"텐서 메모리 최적화 실패: {str(e)}")

                    self._current_model = model
                    return model
        except Exception as e:
            logger.error(f"모델 최적화 실패: {str(e)}")
            return model

    def _apply_amp(self, model: nn.Module) -> nn.Module:
        """AMP(자동 혼합 정밀도) 적용"""
        try:
            if self.config.use_amp:
                model = model.half()  # FP16으로 변환
            return model
        except Exception as e:
            logger.error(f"AMP 적용 실패: {str(e)}")
            return model

    def _optimize_tensor_memory(self, model: nn.Module) -> nn.Module:
        """텐서 메모리 최적화"""
        try:
            # 연속적인 메모리 접근을 위한 텐서 재배치
            for param in model.parameters():
                if param.is_cuda:
                    param.data = param.data.contiguous()

            # 메모리 포맷 최적화 (채널 우선 포맷)
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.data = module.weight.data.to(
                        device="cuda", memory_format=torch.channels_last
                    )
                    if module.bias is not None:
                        module.bias.data = module.bias.data.to(
                            device="cuda", memory_format=torch.channels_last
                        )

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
                # 입력 데이터 전송
                input_tensor = input_data.to(self.device)

                # AMP 활성화 설정
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    with torch.no_grad():
                        output = self._run_sync_inference(input_tensor)

                return output
        except Exception as e:
            logger.error(f"추론 최적화 실패: {str(e)}")
            return None

    def _run_sync_inference(self, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """동기 추론 실행"""
        try:
            if self._current_model is None:
                return None

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                with torch.no_grad():
                    output = self._current_model(input_tensor)

            return output
        except Exception as e:
            logger.error(f"동기 추론 실행 실패: {str(e)}")
            return None

    def _start_worker_threads(self):
        """워커 스레드 시작"""
        try:
            for i in range(2):
                worker = Thread(target=self._worker_loop, args=(i,), daemon=True)
                worker.start()
                self._worker_threads.append(worker)

        except Exception as e:
            logger.error(f"워커 스레드 시작 실패: {str(e)}")

    def _worker_loop(self, worker_id: int):
        """워커 루프"""
        try:
            while True:
                # 입력 데이터 가져오기
                input_tensor = self._inference_queue.get()
                if input_tensor is None:
                    break

                # 추론 실행
                with torch.cuda.stream(self._inference_streams[worker_id]):
                    output = self._run_sync_inference(input_tensor)

                # 결과 저장
                if output is not None:
                    self._result_queue.put(output)

        except Exception as e:
            logger.error(f"워커 루프 실행 실패: {str(e)}")

    def cleanup(self):
        """리소스 정리"""
        try:
            with self._profiler.profile("cleanup"):
                # 워커 스레드 종료
                for _ in range(2):
                    self._inference_queue.put(None)

                for worker in self._worker_threads:
                    worker.join()

                # 큐 비우기
                while not self._inference_queue.empty():
                    try:
                        self._inference_queue.get_nowait()
                    except:
                        pass

                while not self._result_queue.empty():
                    try:
                        self._result_queue.get_nowait()
                    except:
                        pass

                # 리소스 해제
                self._executor.shutdown(wait=True)
                self._memory_manager.cleanup()
                self._thread_cache.clear()
                torch.cuda.empty_cache()

                logger.info("리소스 정리 완료")
        except Exception as e:
            logger.error(f"리소스 정리 실패: {str(e)}")

    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"소멸자 실행 실패: {str(e)}")


"""
CUDA 최적화 및 AMP(Automatic Mixed Precision) 유틸리티
"""


class AMPTrainer:
    """AMP(Automatic Mixed Precision) 트레이너"""

    def __init__(self, config: ConfigProxy):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    @property
    def use_amp(self) -> bool:
        """AMP 사용 여부"""
        return (
            self.config.safe_get("use_amp", default=True)
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
        """AMP를 사용한 학습 스텝"""
        model.train()
        optimizer.zero_grad()

        try:
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(batch)
                    loss = loss_fn(output, batch["target"])

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = model(batch)
                loss = loss_fn(output, batch["target"])
                loss.backward()
                optimizer.step()

            return {"loss": loss.item()}

        except Exception as e:
            logger.error(f"학습 스텝 중 오류 발생: {str(e)}")
            return {"loss": float("inf")}

    def evaluate(
        self, model: torch.nn.Module, loss_fn: Callable, dataloader: DataLoader
    ) -> Dict[str, float]:
        """AMP를 사용한 평가"""
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                try:
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            output = model(batch)
                            loss = loss_fn(output, batch["target"])
                    else:
                        output = model(batch)
                        loss = loss_fn(output, batch["target"])

                    total_loss += loss.item()
                    count += 1

                except Exception as e:
                    logger.error(f"평가 중 오류 발생: {str(e)}")
                    continue

        return {"loss": total_loss / count if count > 0 else float("inf")}

    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보를 반환합니다."""
        device_info = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            ),
        }

        # CUDA 정보 추가
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            device_info["cuda_version"] = f"{props.major}.{props.minor}"
            device_info["total_memory"] = props.total_memory
        else:
            device_info["cuda_version"] = "N/A"

        return device_info

    def get_device(self) -> torch.device:
        """현재 사용 중인 장치를 반환합니다."""
        return self.device

    def optimize_memory(self) -> None:
        """메모리를 최적화합니다."""
        import gc

        # 가비지 컬렉션 실행
        gc.collect()

        # GPU 메모리 정리 (CUDA 사용 가능한 경우)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_device_info() -> Dict[str, Any]:
    """현재 디바이스 정보를 반환"""
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "amp_enabled": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                "gpu_memory_reserved": torch.cuda.memory_reserved(0),
            }
        )

    return info


def optimize_memory():
    """CUDA 메모리 최적화"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
