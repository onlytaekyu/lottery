"""TensorRT 최적화 모듈

이 모듈은 TensorRT 최적화를 위한 클래스를 제공합니다.
"""

import torch
import torch.nn as nn
import torch.onnx
import os
import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Sequence
from pathlib import Path
from threading import Lock

# TensorRT를 전역적으로 가져오기 - 모든 메서드에서 공유
try:
    import tensorrt as trt  # type: ignore # 타입 체커가 TensorRT 모듈 분석을 건너뛰도록 설정

    # TensorRT 속성이 실제로 있는지 확인
    # RuntimeError 방지를 위한 최소 필수 속성 체크
    required_attr = [
        "Logger",
        "Builder",
        "Runtime",
        "NetworkDefinitionCreationFlag",
        "BuilderFlag",
        "OnnxParser",
    ]
    TENSORRT_AVAILABLE = all(hasattr(trt, attr) for attr in required_attr)

    if TENSORRT_AVAILABLE:
        try:
            # TensorRT 상수 - trt가 사용 가능한 경우에만 설정
            network_flag = getattr(trt, "NetworkDefinitionCreationFlag", None)
            EXPLICIT_BATCH = (
                1 << int(getattr(network_flag, "EXPLICIT_BATCH", 0))
                if network_flag
                else 0
            )
        except (AttributeError, TypeError):
            EXPLICIT_BATCH = 0
            TENSORRT_AVAILABLE = False
    else:
        EXPLICIT_BATCH = 0
except (ImportError, AttributeError):
    trt = None
    TENSORRT_AVAILABLE = False
    EXPLICIT_BATCH = 0

# 상대 경로 임포트로 변경
from .error_handler_refactored import get_logger
from .memory_manager import MemoryManager, MemoryConfig
from .async_io import AsyncIOManager, AsyncIOConfig
from .error_recovery import ErrorRecovery, RecoveryConfig

logger = get_logger(__name__)

# CudaConfig는 unified_config.py에서 import됨
from .unified_config import CudaConfig


class BaseCudaOptimizer:
    """CUDA 최적화 기본 클래스"""

    def __init__(self, config: CudaConfig):
        """
        CUDA 최적화기 초기화

        Args:
            config: CUDA 설정
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = Lock()

    def optimize_model(self, model: nn.Module, input_data: torch.Tensor) -> nn.Module:
        """모델 최적화 (하위 클래스에서 구현)"""
        return model

    def optimize_inference(
        self, model: nn.Module, input_data: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """추론 최적화 (하위 클래스에서 구현)"""
        return model(input_data)

    def cleanup(self):
        """리소스 정리 (하위 클래스에서 구현)"""


# 하위 호환성을 위한 별칭
TensorRTConfig = CudaConfig


class LayerType:
    """TensorRT 레이어 타입"""

    CONVOLUTION = "Convolution"
    BATCH_NORMALIZATION = "BatchNormalization"
    RELU = "ReLU"
    POOLING = "Pooling"
    MATRIX_MULTIPLY = "MatrixMultiply"
    ELEMENTWISE = "ElementWise"
    FULLY_CONNECTED = "FullyConnected"


class ElementWiseOperation:
    """요소별 연산 타입"""

    SUM = "Sum"
    PROD = "Prod"
    MAX = "Max"
    MIN = "Min"
    SUB = "Sub"
    DIV = "Div"
    POW = "Pow"


class PoolingType:
    """풀링 연산 타입"""

    MAX = "Max"
    AVERAGE = "Average"


class TensorRTBase(BaseCudaOptimizer):
    """TensorRT 최적화 기본 클래스"""

    def __init__(self, config: CudaConfig):
        """
        TensorRT 최적화기 초기화

        Args:
            config: CUDA 설정 (TensorRT 설정 포함)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._engine: Any = None
        self._context: Any = None
        self._is_initialized = False
        self.logger = get_logger(__name__)
        self._use_tensorrt = TENSORRT_AVAILABLE  # TensorRT 설치 여부에 따라 설정
        self._engine_initialized = False  # 엔진 초기화 상태 플래그
        self._fallback_to_pytorch = (
            not TENSORRT_AVAILABLE
        )  # TensorRT 사용 불가능하면 PyTorch 폴백

        # 스레드 로컬 캐시 초기화
        self._thread_cache = {}

        # TensorRT 로거 초기화
        self._trt_logger = None
        if TENSORRT_AVAILABLE:
            logger_class = getattr(trt, "Logger", None)
            if logger_class:
                warning_level = getattr(logger_class, "WARNING", 0)
                self._trt_logger = logger_class(warning_level)
        else:
            logger.warning("TensorRT 모듈을 로드할 수 없습니다. PyTorch를 사용합니다.")

        # 디렉토리 생성
        for dir_path in [
            self.config.engine_cache_dir,
            self.config.onnx_cache_dir,
            self.config.calibration_cache_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

        # 부모 클래스 초기화 (BaseCudaOptimizer)
        super().__init__(config)

        # 메모리 관리자 초기화 - 메모리 사용량 최적화
        self._memory_manager = MemoryManager(
            MemoryConfig(
                max_memory_usage=0.65,  # 더 보수적인 메모리 사용 (0.75 → 0.65)
                cache_size=1 << 30,  # 1GB 캐시로 감소 (4GB → 1GB)
                memory_track_interval=30,  # 기존 설정 유지
                memory_frags_threshold=0.4,  # 기존 설정 유지
                memory_usage_warning=0.9,  # 경고 임계값 상향 (0.85 → 0.9)
                num_workers=min(
                    self.config.num_workers, 4
                ),  # 워커 수 추가 제한 (8 → 4)
                prefetch_factor=1,  # 프리페치 팩터 추가 감소 (2 → 1)
                pool_size=32,  # 풀 크기 추가 감소 (50 → 32)
                compression_threshold=1 << 21,  # 압축 임계값 감소 (4MB → 2MB)
                alignment_size=64,  # 정렬 크기 추가 감소 (128 → 64)
                use_memory_pooling=True,
                use_memory_alignment=True,
                use_memory_compression=True,
                use_memory_prefetching=False,  # 메모리 프리페칭 비활성화
                use_memory_reuse=True,
                use_memory_tracking=True,
                use_memory_optimization=True,
                use_memory_compaction=True,
                use_memory_pinning=False,  # 메모리 고정 비활성화
                use_memory_streaming=False,  # 메모리 스트리밍 비활성화
                use_memory_events=False,  # 메모리 이벤트 비활성화
                use_memory_graphs=False,  # 그래프 메모리 비활성화
                use_memory_peer_access=False,  # 피어 액세스 비활성화
                use_memory_unified=False,
                use_memory_multi_gpu=False,
                gpu_ids=self.config.gpu_ids,
            )
        )

        # 비동기 IO 관리자 초기화 - 메모리 효율성 개선
        self._async_io = AsyncIOManager(
            AsyncIOConfig(
                chunk_size=1 << 18,  # 청크 크기 감소 (1MB → 256KB)
                max_concurrent_ops=2,  # 동시 작업 수 감소 (4 → 2)
                compression_level=9,  # 압축 수준 증가 (6 → 9)
                compression_threshold=1 << 16,  # 압축 임계값 감소 (1MB → 64KB)
                retry_count=2,  # 재시도 횟수 감소 (3 → 2)
                retry_delay=0.5,  # 재시도 딜레이 감소 (1.0 → 0.5)
                timeout=15.0,  # 타임아웃 감소 (30.0 → 15.0)
                use_compression=True,
            )
        )

        # 프로파일러 초기화 - 가벼운 설정
        self._profiler = Profiler(
            ProfilerConfig(
                enable_cpu_profiling=False,  # CPU 프로파일링 비활성화
                enable_memory_profiling=True,
                enable_thread_profiling=False,  # 스레드 프로파일링 비활성화
                sampling_interval=0.5,  # 샘플링 간격 증가 (0.1 → 0.5)
                memory_threshold=1 << 28,  # 메모리 임계값 감소 (1GB → 256MB)
                cpu_threshold=90.0,  # CPU 임계값 증가 (80.0 → 90.0)
                thread_threshold=200,  # 스레드 임계값 증가 (100 → 200)
            )
        )

        # 에러 복구 메커니즘 초기화
        self._error_recovery = ErrorRecovery(
            RecoveryConfig(
                max_retries=2,  # 재시도 횟수 감소 (3 → 2)
                retry_delay=0.5,  # 재시도 딜레이 감소 (1.0 → 0.5)
                max_memory_percent=70.0,  # 메모리 제한 감소 (80.0 → 70.0)
                max_cpu_percent=85.0,  # CPU 제한 감소 (90.0 → 85.0)
                cleanup_threshold=500,  # 정리 임계값 감소 (1000 → 500)
            )
        )

        self._initialize()

    def _initialize(self):
        """초기화"""
        try:
            # 디렉토리 생성
            os.makedirs(self.config.engine_cache_dir, exist_ok=True)
            os.makedirs(self.config.onnx_cache_dir, exist_ok=True)
            os.makedirs(self.config.calibration_cache_dir, exist_ok=True)

            # CUDA 메모리 정리
            torch.cuda.empty_cache()

            # TensorRT 초기화 성공 여부 확인
            if self._use_tensorrt:
                # TensorRT 가용성 확인
                try:
                    import tensorrt as trt

                    logger.info(f"TensorRT 버전 {trt.__version__} 사용")

                    # 엔진 캐시 로드
                    if self._load_engine_cache():
                        self._engine_initialized = True
                        logger.info("TensorRT 엔진 캐시 로드 성공")
                    else:
                        logger.info(
                            "TensorRT 엔진 캐시 로드 실패, 필요 시 새 엔진을 생성합니다"
                        )
                except (ImportError, AttributeError) as e:
                    logger.warning(
                        f"TensorRT 초기화 실패: {str(e)}. PyTorch를 사용합니다."
                    )
                    self._use_tensorrt = False
                    self._fallback_to_pytorch = True
            else:
                logger.info("PyTorch로 추론을 진행합니다")
        except Exception as e:
            logger.error(f"초기화 실패: {str(e)}")
            self._use_tensorrt = False
            self._fallback_to_pytorch = True

    def _load_engine_cache(self):
        """엔진 캐시 로드"""
        try:
            with self._profiler.profile("load_engine_cache"):
                # TensorRT 사용 불가능하면 즉시 실패
                if not TENSORRT_AVAILABLE:
                    logger.warning(
                        "TensorRT를 사용할 수 없어 엔진 캐시 로드를 건너뜁니다."
                    )
                    return False

                engine_path = self._get_engine_path()

                # 엔진 파일 존재 확인
                if not os.path.exists(engine_path):
                    logger.warning(f"엔진 파일이 존재하지 않습니다: {engine_path}")
                    return False

                # 엔진 해시 및 캐시 유효성 확인
                engine_hash = self._get_engine_hash(engine_path)
                if not self._is_cache_valid(engine_hash):
                    logger.warning(f"엔진 캐시가 유효하지 않습니다: {engine_path}")
                    return False

                # 엔진 로드 시도
                if not self._load_engine(engine_path):
                    logger.warning(f"엔진 로드에 실패했습니다: {engine_path}")
                    return False

                logger.info(f"엔진 캐시 로드 성공: {engine_path}")
                return True
        except Exception as e:
            logger.error(f"엔진 캐시 로드 실패: {str(e)}")
            return False

    async def _save_engine_cache(self, engine_path: str):
        """엔진 캐시 저장"""
        try:
            with self._profiler.profile("save_engine_cache"):
                engine_hash = self._get_engine_hash(engine_path)
                cache_path = os.path.join(
                    self.config.engine_cache_dir,
                    f"{self.config.engine_cache_prefix}_{engine_hash}{self.config.engine_cache_suffix}",
                )

                # 비동기로 엔진 파일 저장
                with open(engine_path, "rb") as f:
                    engine_data = f.read()
                await self._async_io.write_file(cache_path, engine_data)

                # 캐시 메타데이터 저장
                metadata = {
                    "engine_hash": engine_hash,
                    "timestamp": time.time(),
                    "config": self.config.__dict__,
                }
                metadata_path = os.path.join(
                    self.config.engine_cache_dir,
                    f"{self.config.engine_cache_prefix}_{engine_hash}_metadata.json",
                )
                await self._async_io.write_json(metadata_path, metadata)

                logger.info(f"엔진 캐시 저장 완료: {cache_path}")
                return True
        except Exception as e:
            logger.error(f"엔진 캐시 저장 실패: {str(e)}")
            return False

    def _load_engine(self, engine_path: str):
        """엔진 로드"""
        try:
            with self._profiler.profile("load_engine"):
                # 파일 존재 확인
                if not os.path.exists(engine_path):
                    logger.error(f"엔진 파일이 존재하지 않습니다: {engine_path}")
                    return False

                # 비동기로 엔진 파일 로드
                engine_data = self._async_io.sync_read_file(engine_path)
                if engine_data is None:
                    logger.error(f"엔진 파일 로드 실패: {engine_path}")
                    return False

                # 엔진 생성
                if self._trt_logger is None and TENSORRT_AVAILABLE:
                    logger_class = getattr(trt, "Logger", None)
                    if logger_class:
                        warning_level = getattr(logger_class, "WARNING", 0)
                        self._trt_logger = logger_class(warning_level)

                if not TENSORRT_AVAILABLE:
                    logger.error("TensorRT를 사용할 수 없습니다.")
                    return False

                runtime_class = getattr(trt, "Runtime", None)
                if not runtime_class:
                    logger.error("TensorRT Runtime 클래스를 찾을 수 없습니다.")
                    return False

                runtime = runtime_class(self._trt_logger)

                # engine_data가 None이 아닌지 확인
                if engine_data is not None:
                    self._trt_model = runtime.deserialize_cuda_engine(engine_data)
                    if self._trt_model is None:
                        logger.error("엔진 역직렬화 실패")
                        return False
                    self._trt_context = self._trt_model.create_execution_context()
                    logger.info(f"엔진 로드 완료: {engine_path}")
                    return True
                else:
                    logger.error("역직렬화 중단 - 엔진 데이터가 None입니다")
                    return False
        except Exception as e:
            logger.error(f"엔진 로드 실패: {str(e)}")
            return False

    def optimize_model(self, model: nn.Module, input_data: torch.Tensor) -> nn.Module:
        """모델 최적화"""
        try:
            with self._lock:
                with self._profiler.profile("optimize_model"):
                    # 메모리 사용량 확인 및 최적화
                    if torch.cuda.is_available():
                        allocated_mb = torch.cuda.memory_allocated() / (1024**2)
                        total_mb = torch.cuda.get_device_properties(0).total_memory / (
                            1024**2
                        )
                        usage_percent = allocated_mb / total_mb * 100
                        logger.info(
                            f"Memory before optimization: allocated={allocated_mb:.2f}MB / total={total_mb:.2f}MB ({usage_percent:.1f}%)"
                        )

                        # 메모리 부족 시 PyTorch 폴백 고려
                        if allocated_mb > 0.98 * total_mb:
                            logger.info(
                                f"GPU memory usage is at {usage_percent:.1f}%, skipping TensorRT optimization (fallback to PyTorch)"
                            )
                            self._use_tensorrt = False
                            self._fallback_to_pytorch = True
                            return model
                        elif allocated_mb > 0.9 * total_mb:
                            logger.info(
                                f"GPU memory usage high ({usage_percent:.1f}%), but proceeding with TensorRT"
                            )

                    # 메모리 관리자를 통한 추가 메모리 검사
                    if not self._memory_manager.check_memory_for_batch(
                        batch_size=1, operation_name="optimize_model", memory_type="gpu"
                    ):
                        logger.info(
                            "TensorRT optimization skipped due to GPU memory constraint (non-critical)."
                        )
                        self._use_tensorrt = False
                        self._fallback_to_pytorch = True
                        return model

                    # 입력 형태 로깅
                    input_shape = list(input_data.shape)
                    engine_path = self._get_engine_path()
                    logger.info(
                        f"[TensorRT] 입력 형태에 맞게 모델 최적화: {input_shape}, 저장 경로: {engine_path}"
                    )

                    # 모델을 CUDA로 이동
                    if torch.cuda.is_available():
                        model = model.to(self.device)

                    # TensorRT 사용 불가능하면 PyTorch 사용
                    if not self._use_tensorrt:
                        logger.info(
                            "TensorRT 비활성화됨, PyTorch 모델을 직접 사용합니다"
                        )
                        self._current_model = model
                        return model

                    # TensorRT 최적화 시도
                    # ONNX 변환
                    onnx_path = self._convert_to_onnx(model, input_data)
                    if not onnx_path:
                        # ONNX 변환 실패 - PyTorch 모델 직접 사용
                        logger.info("ONNX 변환 실패, PyTorch 모델을 직접 사용합니다")
                        self._fallback_to_pytorch = True
                        self._current_model = model
                        return model

                    # 엔진 생성
                    engine_path = self._create_engine(onnx_path)
                    if not engine_path:
                        logger.info(
                            "TensorRT 엔진 생성 실패, PyTorch 모델을 직접 사용합니다"
                        )
                        self._fallback_to_pytorch = True
                        self._current_model = model
                        return model

                    # 엔진 로드
                    if not self._load_engine(engine_path):
                        logger.info(
                            "TensorRT 엔진 로드 실패, PyTorch 모델을 직접 사용합니다"
                        )
                        self._fallback_to_pytorch = True
                        self._current_model = model
                        return model

                    # 최적화 성공
                    logger.info("TensorRT 최적화 성공")
                    self._engine_initialized = True
                    self._fallback_to_pytorch = False
                    self._current_model = model

                    # 메모리 사용량 확인
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 캐시 정리
                        allocated_mb = torch.cuda.memory_allocated() / (1024**2)
                        total_mb = torch.cuda.get_device_properties(0).total_memory / (
                            1024**2
                        )
                        usage_percent = allocated_mb / total_mb * 100
                        logger.info(
                            f"Memory after optimization: allocated={allocated_mb:.2f}MB / total={total_mb:.2f}MB ({usage_percent:.1f}%)"
                        )

                    return model

        except Exception as e:
            logger.error(f"모델 최적화 실패: {str(e)}")
            # 오류 발생 시 원본 모델 반환
            self._fallback_to_pytorch = True
            self._current_model = model
            return model

    def optimize_inference(
        self, model: nn.Module, input_data: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """추론 최적화"""
        try:
            with self._profiler.profile("optimize_inference"):
                # 입력 데이터 전처리
                input_tensor = self._preprocess_data(input_data)
                if input_tensor is None:
                    return None

                # TensorRT 추론 가능 여부 확인 (우선 사용)
                can_use_tensorrt = (
                    self._use_tensorrt
                    and not self._fallback_to_pytorch
                    and hasattr(self, "_trt_model")
                    and self._trt_model is not None
                    and hasattr(self, "_trt_context")
                    and self._trt_context is not None
                )

                # 메모리 상태 확인
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / (1024**2)
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (
                        1024**2
                    )
                    usage_percent = current_memory / total_memory * 100

                    # 상세한 메모리 사용량 로깅
                    logger.info(
                        f"Memory during inference: {current_memory:.2f}MB / {total_memory:.2f}MB ({usage_percent:.1f}%)"
                    )

                    # 메모리 사용량이 높을 때 경고 및 조치
                    if usage_percent > 80.0:
                        logger.warning(
                            f"GPU memory usage high: {current_memory:.2f}MB / {total_memory:.2f}MB ({usage_percent:.1f}%) - running memory cleanup"
                        )
                        torch.cuda.empty_cache()

                # TensorRT 또는 PyTorch 추론
                if can_use_tensorrt:
                    try:
                        output = self._run_trt_inference(input_tensor)
                        if output is not None:
                            return output
                        # TensorRT 추론 실패 시 PyTorch로 폴백
                        logger.warning("TensorRT 추론 실패, PyTorch로 폴백")
                        self._fallback_to_pytorch = True
                        return self._run_pytorch_inference(input_tensor)
                    except Exception as e:
                        logger.warning(
                            f"TensorRT 추론 중 오류 발생: {str(e)}, PyTorch로 폴백"
                        )
                        self._fallback_to_pytorch = True
                        return self._run_pytorch_inference(input_tensor)
                else:
                    # PyTorch 추론
                    return self._run_pytorch_inference(input_tensor)

        except Exception as e:
            logger.error(
                f"추론 최적화 실패: {e} | shape={input_tensor.shape}, dtype={input_tensor.dtype}, device={input_tensor.device}, min={input_tensor.min()}, max={input_tensor.max()}"
            )
            # 오류 발생 시 PyTorch 추론 시도
            try:
                return self._run_pytorch_inference(input_tensor)
            except:
                return None

    def cleanup(self):
        """리소스 정리 - Python 종료 시 안전한 정리"""
        try:
            # Python 종료 상태 확인
            import sys

            if sys is None or getattr(sys, "meta_path", None) is None:
                # Python이 종료 중이면 중요한 정리만 수행
                if hasattr(self, "_trt_context"):
                    self._trt_context = None
                if hasattr(self, "_trt_model"):
                    self._trt_model = None
                return

            with self._profiler.profile("cleanup"):
                # TensorRT 리소스 해제
                if hasattr(self, "_trt_context") and self._trt_context is not None:
                    self._trt_context = None
                if hasattr(self, "_trt_model") and self._trt_model is not None:
                    self._trt_model = None

                # 모델 참조 해제
                if hasattr(self, "_current_model") and self._current_model is not None:
                    self._current_model = None

                # 메모리 관리자 정리
                if hasattr(self, "_memory_manager"):
                    self._memory_manager.cleanup()

                # 캐시 정리
                if hasattr(self, "_thread_cache"):
                    self._thread_cache.clear()

                # CUDA 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # 모든 CUDA 연산 완료 대기
                    torch.cuda.empty_cache()  # 캐시 메모리 해제

                # 가비지 컬렉션 강제 실행
                import gc

                gc.collect()

                logger.info("리소스 정리 완료")
        except Exception as e:
            # Python 종료 시에는 로깅도 실패할 수 있음
            try:
                logger.error(f"리소스 정리 실패: {str(e)}")
            except:
                pass

    def __del__(self):
        """소멸자 - Python 종료 시 안전한 정리"""
        try:
            # Python 종료 상태 확인
            import sys

            if sys is None or getattr(sys, "meta_path", None) is None:
                # Python이 종료 중이면 정리 작업을 건너뜀
                return

            if hasattr(self, "cleanup"):
                self.cleanup()
        except Exception:
            # Python 종료 시에는 로깅도 실패할 수 있으므로 조용히 무시
            pass

    def _can_fuse_conv_relu(self, layer: Any) -> bool:
        """Conv + ReLU 융합 가능 여부 확인"""
        try:
            return (
                layer.type == LayerType.CONVOLUTION
                and layer.get_output(0).get_consumers()[0].type == LayerType.RELU
            )
        except Exception as e:
            logger.error(f"Conv + ReLU 융합 가능 여부 확인 실패: {str(e)}")
            return False

    def _can_fuse_conv_bn_relu(self, layer: Any) -> bool:
        """Conv + BN + ReLU 융합 가능 여부 확인"""
        try:
            return (
                layer.type == LayerType.CONVOLUTION
                and layer.get_output(0).get_consumers()[0].type
                == LayerType.BATCH_NORMALIZATION
                and layer.get_output(0)
                .get_consumers()[0]
                .get_output(0)
                .get_consumers()[0]
                .type
                == LayerType.RELU
            )
        except Exception as e:
            logger.error(f"Conv + BN + ReLU 융합 가능 여부 확인 실패: {str(e)}")
            return False

    def _can_fuse_conv_bn(self, layer: Any) -> bool:
        """Conv + BN 융합 가능 여부 확인"""
        try:
            return (
                layer.type == LayerType.CONVOLUTION
                and layer.get_output(0).get_consumers()[0].type
                == LayerType.BATCH_NORMALIZATION
            )
        except Exception as e:
            logger.error(f"Conv + BN 융합 가능 여부 확인 실패: {str(e)}")
            return False

    def _fuse_conv_relu(self, layer: Any):
        """Conv + ReLU 융합"""
        try:
            # ReLU 레이어 가져오기
            relu_layer = layer.get_output(0).get_consumers()[0]
            # 융합
            layer.set_output(0, relu_layer.get_output(0))
        except Exception as e:
            logger.error(f"Conv + ReLU 융합 실패: {str(e)}")

    def _fuse_conv_bn_relu(self, layer: Any):
        """Conv + BN + ReLU 융합"""
        try:
            # BN 레이어 가져오기
            bn_layer = layer.get_output(0).get_consumers()[0]
            # ReLU 레이어 가져오기
            relu_layer = bn_layer.get_output(0).get_consumers()[0]
            # 융합
            layer.set_output(0, relu_layer.get_output(0))
        except Exception as e:
            logger.error(f"Conv + BN + ReLU 융합 실패: {str(e)}")

    def _fuse_conv_bn(self, layer: Any):
        """Conv + BN 융합"""
        try:
            # BN 레이어 가져오기
            bn_layer = layer.get_output(0).get_consumers()[0]
            # 융합
            layer.set_output(0, bn_layer.get_output(0))
        except Exception as e:
            logger.error(f"Conv + BN 융합 실패: {str(e)}")

    def _can_fuse_conv_add_relu(self, layer: Any) -> bool:
        """Conv + Add + ReLU 융합 가능 여부 확인"""
        try:
            return (
                layer.type == LayerType.CONVOLUTION
                and layer.get_output(0).get_consumers()[0].type == LayerType.ELEMENTWISE
                and layer.get_output(0).get_consumers()[0].get_operation()
                == ElementWiseOperation.SUM
                and layer.get_output(0)
                .get_consumers()[0]
                .get_output(0)
                .get_consumers()[0]
                .type
                == LayerType.RELU
            )
        except Exception as e:
            logger.error(f"Conv + Add + ReLU 융합 가능 여부 확인 실패: {str(e)}")
            return False

    def _can_fuse_conv_maxpool_relu(self, layer: Any) -> bool:
        """Conv + MaxPool + ReLU 융합 가능 여부 확인"""
        try:
            return (
                layer.type == LayerType.CONVOLUTION
                and layer.get_output(0).get_consumers()[0].type == LayerType.POOLING
                and layer.get_output(0).get_consumers()[0].get_pooling_type()
                == PoolingType.MAX
                and layer.get_output(0)
                .get_consumers()[0]
                .get_output(0)
                .get_consumers()[0]
                .type
                == LayerType.RELU
            )
        except Exception as e:
            logger.error(f"Conv + MaxPool + ReLU 융합 가능 여부 확인 실패: {str(e)}")
            return False

    def _can_fuse_conv_avgpool_relu(self, layer: Any) -> bool:
        """Conv + AvgPool + ReLU 융합 가능 여부 확인"""
        try:
            return (
                layer.type == LayerType.CONVOLUTION
                and layer.get_output(0).get_consumers()[0].type == LayerType.POOLING
                and layer.get_output(0).get_consumers()[0].get_pooling_type()
                == PoolingType.AVERAGE
                and layer.get_output(0)
                .get_consumers()[0]
                .get_output(0)
                .get_consumers()[0]
                .type
                == LayerType.RELU
            )
        except Exception as e:
            logger.error(f"Conv + AvgPool + ReLU 융합 가능 여부 확인 실패: {str(e)}")
            return False

    def _can_fuse_matmul_add_relu(self, layer: Any) -> bool:
        """MatMul + Add + ReLU 융합 가능 여부 확인"""
        try:
            return (
                layer.type == LayerType.MATRIX_MULTIPLY
                and layer.get_output(0).get_consumers()[0].type == LayerType.ELEMENTWISE
                and layer.get_output(0).get_consumers()[0].get_operation()
                == ElementWiseOperation.SUM
                and layer.get_output(0)
                .get_consumers()[0]
                .get_output(0)
                .get_consumers()[0]
                .type
                == LayerType.RELU
            )
        except Exception as e:
            logger.error(f"MatMul + Add + ReLU 융합 가능 여부 확인 실패: {str(e)}")
            return False

    def _can_fuse_fc_relu(self, layer: Any) -> bool:
        """FC + ReLU 융합 가능 여부 확인"""
        try:
            return (
                layer.type == LayerType.FULLY_CONNECTED
                and layer.get_output(0).get_consumers()[0].type == LayerType.RELU
            )
        except Exception as e:
            logger.error(f"FC + ReLU 융합 가능 여부 확인 실패: {str(e)}")
            return False

    def _can_fuse_fc_bn_relu(self, layer: Any) -> bool:
        """FC + BN + ReLU 융합 가능 여부 확인"""
        try:
            return (
                layer.type == LayerType.FULLY_CONNECTED
                and layer.get_output(0).get_consumers()[0].type
                == LayerType.BATCH_NORMALIZATION
                and layer.get_output(0)
                .get_consumers()[0]
                .get_output(0)
                .get_consumers()[0]
                .type
                == LayerType.RELU
            )
        except Exception as e:
            logger.error(f"FC + BN + ReLU 융합 가능 여부 확인 실패: {str(e)}")
            return False

    def _fuse_conv_add_relu(self, layer: Any):
        """Conv + Add + ReLU 융합"""
        try:
            # Add 레이어 가져오기
            add_layer = layer.get_output(0).get_consumers()[0]
            # ReLU 레이어 가져오기
            relu_layer = add_layer.get_output(0).get_consumers()[0]
            # 융합
            layer.set_output(0, relu_layer.get_output(0))
        except Exception as e:
            logger.error(f"Conv + Add + ReLU 융합 실패: {str(e)}")

    def _fuse_conv_maxpool_relu(self, layer: Any):
        """Conv + MaxPool + ReLU 융합"""
        try:
            # MaxPool 레이어 가져오기
            maxpool_layer = layer.get_output(0).get_consumers()[0]
            # ReLU 레이어 가져오기
            relu_layer = maxpool_layer.get_output(0).get_consumers()[0]
            # 융합
            layer.set_output(0, relu_layer.get_output(0))
        except Exception as e:
            logger.error(f"Conv + MaxPool + ReLU 융합 실패: {str(e)}")

    def _fuse_conv_avgpool_relu(self, layer: Any):
        """Conv + AvgPool + ReLU 융합"""
        try:
            # AvgPool 레이어 가져오기
            avgpool_layer = layer.get_output(0).get_consumers()[0]
            # ReLU 레이어 가져오기
            relu_layer = avgpool_layer.get_output(0).get_consumers()[0]
            # 융합
            layer.set_output(0, relu_layer.get_output(0))
        except Exception as e:
            logger.error(f"Conv + AvgPool + ReLU 융합 실패: {str(e)}")

    def _fuse_matmul_add_relu(self, layer: Any):
        """MatMul + Add + ReLU 융합"""
        try:
            # Add 레이어 가져오기
            add_layer = layer.get_output(0).get_consumers()[0]
            # ReLU 레이어 가져오기
            relu_layer = add_layer.get_output(0).get_consumers()[0]
            # 융합
            layer.set_output(0, relu_layer.get_output(0))
        except Exception as e:
            logger.error(f"MatMul + Add + ReLU 융합 실패: {str(e)}")

    def _fuse_fc_relu(self, layer: Any):
        """FC + ReLU 융합"""
        try:
            # ReLU 레이어 가져오기
            relu_layer = layer.get_output(0).get_consumers()[0]
            # 융합
            layer.set_output(0, relu_layer.get_output(0))
        except Exception as e:
            logger.error(f"FC + ReLU 융합 실패: {str(e)}")

    def _fuse_fc_bn_relu(self, layer: Any):
        """FC + BN + ReLU 융합"""
        try:
            # BN 레이어 가져오기
            bn_layer = layer.get_output(0).get_consumers()[0]
            # ReLU 레이어 가져오기
            relu_layer = bn_layer.get_output(0).get_consumers()[0]
            # 융합
            layer.set_output(0, relu_layer.get_output(0))
        except Exception as e:
            logger.error(f"FC + BN + ReLU 융합 실패: {str(e)}")

    def _adjust_batch_size(
        self, input_data: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """배치 크기 조정"""
        try:
            if batch_size == input_data.size(0):
                return input_data

            # 배치 크기 조정
            if batch_size < input_data.size(0):
                return input_data[:batch_size]
            else:
                # 패딩으로 배치 크기 증가
                padding = torch.zeros(
                    (batch_size - input_data.size(0),) + input_data.shape[1:],
                    dtype=input_data.dtype,
                    device=input_data.device,
                )
                return torch.cat([input_data, padding], dim=0)
        except Exception as e:
            logger.error(f"배치 크기 조정 실패: {str(e)}")
            return input_data

    def _get_engine_path(self) -> str:
        """엔진 파일 경로 반환"""
        return os.path.join(
            self.config.engine_cache_dir,
            f"{self.config.engine_cache_prefix}_{self.config.engine_cache_version}{self.config.engine_cache_suffix}",
        )

    def _get_engine_hash(self, engine_path: str) -> str:
        """엔진 파일의 해시값 반환"""
        with open(engine_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _is_cache_valid(self, engine_hash: str) -> bool:
        """캐시 유효성 검사"""
        metadata_path = os.path.join(
            self.config.engine_cache_dir,
            f"{self.config.engine_cache_prefix}_{engine_hash}_metadata.json",
        )

        # 메타데이터 파일이 없으면 메타데이터 생성 (자동 복구)
        if not os.path.exists(metadata_path):
            try:
                # 엔진 파일 확인
                engine_path = self._get_engine_path()
                if not os.path.exists(engine_path):
                    logger.warning(f"엔진 파일이 존재하지 않습니다: {engine_path}")
                    return False

                # 메타데이터 파일 자동 생성
                logger.info(f"메타데이터 파일 자동 생성: {metadata_path}")
                metadata = {
                    "engine_hash": engine_hash,
                    "timestamp": time.time(),
                    "config": {
                        k: str(v)
                        for k, v in self.config.__dict__.items()
                        if not k.startswith("_")
                    },
                    "auto_generated": True,
                    "version": self.config.engine_cache_version,
                }

                # 메타데이터 파일 저장
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"메타데이터 파일 자동 생성 완료: {metadata_path}")
                return True
            except Exception as e:
                logger.warning(f"메타데이터 자동 생성 실패: {str(e)}")
                return False

        # 메타데이터 파일이 있으면 정상 검증
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                # 해시 검사만 수행 (TTL 제한 없음)
                is_valid = metadata.get("engine_hash") == engine_hash
                if is_valid:
                    logger.info(f"엔진 캐시 유효성 검사 성공: {engine_hash}")
                else:
                    logger.warning(
                        f"엔진 해시 불일치: {metadata.get('engine_hash')} != {engine_hash}"
                    )
                return is_valid
        except Exception as e:
            logger.warning(f"메타데이터 파일 읽기 오류: {str(e)}")
            # 읽기 오류 시 메타데이터 재생성 시도
            try:
                os.remove(metadata_path)
                logger.info(f"손상된 메타데이터 파일 삭제: {metadata_path}")
                # 재귀적으로 다시 유효성 검사 수행 (한 번만)
                return self._is_cache_valid(engine_hash)
            except Exception as ex:
                logger.error(f"손상된 메타데이터 파일 삭제 실패: {str(ex)}")
                return False

    def _run_trt_inference(self, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            # TensorRT 사용 가능 여부 확인
            if not TENSORRT_AVAILABLE:
                logger.error("TensorRT를 사용할 수 없습니다.")
                return None

            # 메모리 확인 및 최적화
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 이전 CUDA 작업 완료 대기

            # 출력 크기 설정
            if self.config.output_size is None:
                self.config.output_size = list(input_tensor.shape)

            # 출력 저장소 준비
            output_storage = torch.empty(
                self.config.output_size, dtype=torch.float32, device=self.device
            )

            # TensorRT 컨텍스트 확인
            if not hasattr(self, "_trt_context") or self._trt_context is None:
                logger.error("TensorRT context가 초기화되지 않았습니다.")
                return None

            # 입력 텐서 형태 및 메모리 연속성 확인
            if not input_tensor.is_contiguous():
                logger.debug("입력 텐서가 연속적이지 않아 연속적으로 변환합니다.")
                input_tensor = input_tensor.contiguous()

            # 데이터 포인터 유효 여부 확인
            if not input_tensor.is_cuda:
                logger.debug("입력 텐서가 CUDA 텐서가 아니어서 CUDA로 변환합니다.")
                input_tensor = input_tensor.cuda()

            # 추론 실행
            self._trt_context.execute_v2(
                [input_tensor.data_ptr(), output_storage.data_ptr()]
            )

            # 출력 처리
            output = output_storage.to(self.device)
            logger.debug(
                f"TRT Inference: output shape={output.shape}, dtype={output.dtype}"
            )

            # 메모리 효율성을 위해 불필요한 텐서 해제
            del output_storage

            return output
        except Exception as e:
            logger.error(
                f"TensorRT 추론 실패: {e} | shape={input_tensor.shape}, dtype={input_tensor.dtype}, device={input_tensor.device}"
            )
            return None

    def _run_pytorch_inference(
        self, input_tensor: torch.Tensor
    ) -> Optional[torch.Tensor]:
        try:
            # 모델 초기화 검사
            if self._current_model is None:
                logger.error("현재 모델이 초기화되지 않았습니다")
                return None

            # 메모리 효율성을 위한 AMP 사용
            with torch.cuda.amp.autocast(enabled=True):
                output = self._current_model(input_tensor)

            # 출력이 튜플/리스트인 경우 첫 번째 항목만 반환
            if isinstance(output, (tuple, list)):
                output = output[0]

            return output
        except Exception as e:
            logger.error(
                f"PyTorch 추론 실행 실패: {str(e)} | shape={input_tensor.shape}, dtype={input_tensor.dtype}"
            )
            return None

    def _preprocess_data(self, input_data: torch.Tensor) -> torch.Tensor:
        """입력 데이터 전처리 - 메모리 효율성 개선"""
        try:
            # 데이터 타입 변환 (메모리 효율성)
            if input_data.dtype != torch.float32:
                input_data = input_data.float()

            # 메모리 연속성 확보
            input_data = input_data.contiguous()

            # 배치 크기 최적화 (선택적)
            if (
                hasattr(self.config, "max_batch_size")
                and input_data.shape[0] > self.config.max_batch_size
            ):
                logger.warning(
                    f"배치 크기 {input_data.shape[0]}가 최대 배치 크기 {self.config.max_batch_size}를 초과하여 잘랐습니다"
                )
                input_data = input_data[: self.config.max_batch_size]

            # 데이터 정규화 (필요한 경우)
            if (
                hasattr(self.config, "normalize_inputs")
                and self.config.normalize_inputs
            ):
                input_data = (input_data - input_data.mean()) / (
                    input_data.std() + 1e-8
                )

            logger.debug(
                f"Preprocess: shape={input_data.shape}, dtype={input_data.dtype}, "
                f"min={input_data.min().item():.4f}, max={input_data.max().item():.4f}"
            )
            return input_data
        except Exception as e:
            logger.error(f"데이터 전처리 실패: {str(e)}")
            # 원본 반환
            return input_data

    def is_available(self) -> bool:
        """TensorRT 사용 가능 여부 확인"""
        return self._is_initialized and self._engine is not None

    def optimize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        텐서 최적화

        Args:
            tensor: 입력 텐서

        Returns:
            최적화된 텐서
        """
        if not self.is_available():
            return tensor

        try:
            # CUDA 텐서로 변환
            if not tensor.is_cuda:
                tensor = tensor.cuda()

            # TensorRT 엔진으로 추론
            output = self._engine.run([tensor])
            return output[0]

        except Exception as e:
            self.logger.error(f"TensorRT 최적화 중 오류 발생: {str(e)}")
            return tensor

    def optimize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if not self.is_available():
            return batch
        return batch.to(self.device)

    def zeros(self, shape: Union[Sequence[int], int]) -> torch.Tensor:
        try:
            if isinstance(shape, (list, tuple)):
                shape_list = [int(x) for x in shape]
                if len(shape_list) == 1:
                    return torch.zeros((shape_list[0],), device=self.device)
                return torch.zeros(tuple(shape_list), device=self.device)
            elif isinstance(shape, int):
                return torch.zeros((int(shape),), device=self.device)
            else:
                raise ValueError(f"지원하지 않는 shape 타입: {type(shape)}")
        except Exception as e:
            self.logger.error(f"텐서 생성 중 오류 발생: {str(e)}")
            if isinstance(shape, (list, tuple)):
                shape_list = [int(x) for x in shape]
                if len(shape_list) == 1:
                    return torch.zeros((shape_list[0],))  # CPU 텐서로 폴백
                return torch.zeros(tuple(shape_list))  # CPU 텐서로 폴백
            elif isinstance(shape, int):
                return torch.zeros((int(shape),))  # CPU 텐서로 폴백
            else:
                raise ValueError(f"지원하지 않는 shape 타입: {type(shape)}")

    def run(self, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.is_available():
            return input_tensor
        if self._engine is None:
            self._initialize_engine()
        if self._engine is None:  # 초기화 실패
            return None
        if self._context is None:
            try:
                self._context = self._engine.create_execution_context()
            except Exception as e:
                self.logger.error(f"실행 컨텍스트 생성 중 오류 발생: {str(e)}")
                return None
        try:
            return self._context.execute_v2([input_tensor.data_ptr()])[0]
        except Exception as e:
            self.logger.error(f"텐서 실행 중 오류 발생: {str(e)}")
            return None

    def _initialize_engine(self) -> Optional[Union[bool, str]]:
        """TensorRT 엔진 초기화

        Returns:
            Optional[Union[bool, str]]: 초기화 성공 여부 또는 엔진 경로, 오류 시 None
        """
        if not self.is_available():
            return None
        try:
            # TensorRT 모듈 사용 가능 여부 확인
            if not TENSORRT_AVAILABLE:
                logger.error("TensorRT를 사용할 수 없습니다.")
                return None

            # TensorRT 로거 사용 (이미 초기화된 로거 사용)
            if self._trt_logger is None:
                logger_class = getattr(trt, "Logger", None)
                if logger_class:
                    warning_level = getattr(logger_class, "WARNING", 0)
                    self._trt_logger = logger_class(warning_level)

            # ONNX 파일 경로
            onnx_path = f"{self.config.onnx_cache_dir}/{self.config.onnx_cache_prefix}{self.config.onnx_cache_version}{self.config.onnx_cache_suffix}"

            # ONNX 파일 존재 확인
            if not os.path.exists(onnx_path):
                logger.error(f"ONNX 파일이 존재하지 않습니다: {onnx_path}")
                return False

            # TensorRT 빌더 생성
            builder_class = getattr(trt, "Builder", None)
            if not builder_class:
                logger.error("TensorRT Builder 클래스를 찾을 수 없습니다.")
                return False

            builder = builder_class(self._trt_logger)

            # 네트워크 생성
            network_flag = getattr(trt, "NetworkDefinitionCreationFlag", None)
            explicit_batch = (
                1 << int(getattr(network_flag, "EXPLICIT_BATCH", 0))
                if network_flag
                else 0
            )
            network = builder.create_network(explicit_batch)

            # ONNX 파서 생성
            parser_class = getattr(trt, "OnnxParser", None)
            if not parser_class:
                logger.error("TensorRT OnnxParser 클래스를 찾을 수 없습니다.")
                return False

            parser = parser_class(network, self._trt_logger)

            # ONNX 모델 로드
            with open(onnx_path, "rb") as model:
                model_data = model.read()
                if not model_data:
                    logger.error(f"ONNX 모델 파일이 비어 있습니다: {onnx_path}")
                    return False

                if not parser.parse(model_data):
                    error_msgs = ""
                    for i in range(parser.num_errors):
                        error_msgs += f"{parser.get_error(i)}\n"
                    logger.error(f"ONNX 모델 파싱 실패: {error_msgs}")
                    return False

            # 엔진 빌드
            config = builder.create_builder_config()

            # 워크스페이스 메모리 설정
            try:
                # 1GB 워크스페이스 메모리로 감소 (2GB → 1GB)
                memory_size = 1 << 30

                # TensorRT 버전에 따른 메모리 설정 방식 처리
                if hasattr(trt, "MemoryPoolType"):
                    try:
                        # TensorRT 10.0 이상 버전 처리 - 속성 접근 전 안전하게 검사
                        memory_pool_type = getattr(trt, "MemoryPoolType", None)
                        workspace_attr = getattr(memory_pool_type, "WORKSPACE", None)

                        if memory_pool_type is not None and workspace_attr is not None:
                            config.set_memory_pool_limit(workspace_attr, memory_size)
                            logger.info("TensorRT MemoryPoolType.WORKSPACE 설정 완료")
                        else:
                            raise AttributeError("WORKSPACE 속성을 찾을 수 없습니다")
                    except (AttributeError, TypeError) as attr_err:
                        logger.warning(
                            f"MemoryPoolType.WORKSPACE 설정 실패: {attr_err}. 대체 방법 사용"
                        )
                        # 이전 버전 호환성 메서드 시도
                        if hasattr(config, "set_memory_pool_limit"):
                            config.set_memory_pool_limit(memory_size)
                        elif hasattr(config, "max_workspace_size"):
                            config.max_workspace_size = memory_size
                else:
                    # 이전 TensorRT 버전 처리
                    if hasattr(config, "max_workspace_size"):
                        config.max_workspace_size = memory_size
                        logger.info(
                            f"TensorRT 이전 버전 max_workspace_size={memory_size} 설정 완료"
                        )
                    else:
                        logger.warning(
                            "TensorRT 워크스페이스 메모리 설정 방법을 찾을 수 없습니다. 기본 설정 사용"
                        )
            except Exception as e:
                logger.warning(
                    f"워크스페이스 메모리 설정 중 오류: {str(e)}. 기본 설정 사용"
                )

            # FP16 모드 설정
            if self.config.fp16_mode:
                builder_flag = getattr(trt, "BuilderFlag", None)
                if builder_flag:
                    fp16_flag = getattr(builder_flag, "FP16", None)
                    if fp16_flag is not None:
                        config.set_flag(fp16_flag)

            # 동적 입력을 위한 최적화 프로필 설정
            profile = builder.create_optimization_profile()

            # 입력 정보 가져오기
            input_name = "input"
            if self.config.input_size is None:
                # 기본 입력 크기 설정
                self.config.input_size = [1, 1, 28, 28]  # 기본 MNIST 크기
                logger.warning(
                    f"입력 크기가 설정되지 않았습니다. 기본값 사용: {self.config.input_size}"
                )

            # 최적화 프로필 설정 (min, opt, max)
            min_shape = tuple(self.config.input_size)
            opt_shape = tuple(self.config.input_size)
            max_shape = tuple(
                [d * 2 if i == 0 else d for i, d in enumerate(self.config.input_size)]
            )  # 배치 크기만 2배로

            # 프로필에 입력 설정 추가
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)

            # 프로필 설정 추가
            config.add_optimization_profile(profile)

            # 엔진 빌드
            try:
                # build_serialized_network 메서드 사용 (최신 TensorRT)
                serialized_network = builder.build_serialized_network(network, config)

                if serialized_network is None:
                    logger.error("TensorRT 직렬화 네트워크 생성 실패")
                    return None

                runtime_class = getattr(trt, "Runtime", None)
                if not runtime_class:
                    logger.error("TensorRT Runtime 클래스를 찾을 수 없습니다.")
                    return None

                runtime = runtime_class(self._trt_logger)
                engine = runtime.deserialize_cuda_engine(serialized_network)

                if engine is None:
                    logger.error("TensorRT 엔진 생성 실패")
                    return None

                # 엔진 파일 저장
                engine_path = self._get_engine_path()
                try:
                    with open(engine_path, "wb") as f:
                        f.write(engine.serialize())

                    # 메타데이터 파일 생성
                    engine_hash = self._get_engine_hash(engine_path)
                    metadata_path = os.path.join(
                        self.config.engine_cache_dir,
                        f"{self.config.engine_cache_prefix}_{engine_hash}_metadata.json",
                    )

                    # 메타데이터 파일 내용 생성
                    metadata = {
                        "engine_hash": engine_hash,
                        "timestamp": time.time(),
                        "config": {
                            k: str(v)
                            for k, v in self.config.__dict__.items()
                            if not k.startswith("_")
                        },
                        "auto_generated": False,
                        "version": self.config.engine_cache_version,
                    }

                    # 디렉토리 생성 확인
                    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

                    # 메타데이터 파일 저장
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"엔진 메타데이터 파일 생성: {metadata_path}")

                except Exception as e:
                    logger.error(f"엔진 파일 저장 실패: {str(e)}")
                    return None

                logger.info(f"엔진 생성 완료: {engine_path}")
                return engine_path
            except Exception as e:
                logger.error(f"엔진 빌드 실패: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"엔진 생성 중 오류 발생: {str(e)}")
            return None

    def _setup_dynamic_axes(self):
        self.dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if self.config.input_size is not None and len(self.config.input_size) > 2:
            self.dynamic_axes["input"][2] = "height"
            self.dynamic_axes["input"][3] = "width"
        # ... 추가적인 설정 ...

    def _create_engine(self, onnx_path: str) -> Optional[str]:
        """ONNX 모델로부터 TensorRT 엔진 생성"""
        try:
            # ONNX 파일 존재 확인
            if not os.path.exists(onnx_path):
                logger.error(f"ONNX 파일이 존재하지 않습니다: {onnx_path}")
                return None

            # TensorRT 모듈 가져오기 시도
            if not TENSORRT_AVAILABLE:
                logger.warning(
                    "TensorRT 모듈이 설치되어 있지 않습니다. PyTorch 모델을 직접 사용합니다."
                )
                return None

            # TensorRT 로거 사용 (이미 초기화된 로거 사용)
            if self._trt_logger is None:
                logger_class = getattr(trt, "Logger", None)
                if logger_class:
                    warning_level = getattr(logger_class, "WARNING", 0)
                    self._trt_logger = logger_class(warning_level)

            # 입력 및 출력 형태 로깅
            input_shape = (
                self.config.input_size if self.config.input_size else "알 수 없음"
            )
            engine_path = self._get_engine_path()
            logger.info(
                f"[TensorRT] ONNX 파일({onnx_path})로부터 엔진 생성, 입력 형태: {input_shape}, 엔진 경로: {engine_path}"
            )

            # TensorRT 빌더 및 네트워크 생성
            builder_class = getattr(trt, "Builder", None)
            if not builder_class:
                logger.error("TensorRT Builder 클래스를 찾을 수 없습니다.")
                return None

            builder = builder_class(self._trt_logger)
            network = builder.create_network(EXPLICIT_BATCH)

            parser_class = getattr(trt, "OnnxParser", None)
            if not parser_class:
                logger.error("TensorRT OnnxParser 클래스를 찾을 수 없습니다.")
                return None

            parser = parser_class(network, self._trt_logger)

            # ONNX 모델 로드
            with open(onnx_path, "rb") as model:
                model_data = model.read()
                if not model_data:
                    logger.error(f"ONNX 모델 파일이 비어 있습니다: {onnx_path}")
                    return None

                if not parser.parse(model_data):
                    error_msgs = ""
                    for i in range(parser.num_errors):
                        error_msgs += f"{parser.get_error(i)}\n"
                    logger.error(f"ONNX 모델 파싱 실패: {error_msgs}")
                    return None

            # 엔진 빌드
            config = builder.create_builder_config()

            # 워크스페이스 메모리 설정
            try:
                # 1GB 워크스페이스 메모리로 감소 (2GB → 1GB)
                memory_size = 1 << 30

                # TensorRT 버전에 따른 메모리 설정 방식 처리
                if hasattr(trt, "MemoryPoolType"):
                    try:
                        # TensorRT 10.0 이상 버전 처리 - 속성 접근 전 안전하게 검사
                        memory_pool_type = getattr(trt, "MemoryPoolType", None)
                        workspace_attr = getattr(memory_pool_type, "WORKSPACE", None)

                        if memory_pool_type is not None and workspace_attr is not None:
                            config.set_memory_pool_limit(workspace_attr, memory_size)
                            logger.info("TensorRT MemoryPoolType.WORKSPACE 설정 완료")
                        else:
                            raise AttributeError("WORKSPACE 속성을 찾을 수 없습니다")
                    except (AttributeError, TypeError) as attr_err:
                        logger.warning(
                            f"MemoryPoolType.WORKSPACE 설정 실패: {attr_err}. 대체 방법 사용"
                        )
                        # 이전 버전 호환성 메서드 시도
                        if hasattr(config, "set_memory_pool_limit"):
                            config.set_memory_pool_limit(memory_size)
                        elif hasattr(config, "max_workspace_size"):
                            config.max_workspace_size = memory_size
                else:
                    # 이전 TensorRT 버전 처리
                    if hasattr(config, "max_workspace_size"):
                        config.max_workspace_size = memory_size
                        logger.info(
                            f"TensorRT 이전 버전 max_workspace_size={memory_size} 설정 완료"
                        )
                    else:
                        logger.warning(
                            "TensorRT 워크스페이스 메모리 설정 방법을 찾을 수 없습니다. 기본 설정 사용"
                        )
            except Exception as e:
                logger.warning(
                    f"워크스페이스 메모리 설정 중 오류: {str(e)}. 기본 설정 사용"
                )

            # FP16 모드 설정
            if self.config.fp16_mode:
                builder_flag = getattr(trt, "BuilderFlag", None)
                if builder_flag:
                    fp16_flag = getattr(builder_flag, "FP16", None)
                    if fp16_flag is not None:
                        config.set_flag(fp16_flag)

            # 동적 입력을 위한 최적화 프로필 설정
            profile = builder.create_optimization_profile()

            # 입력 정보 가져오기
            input_name = "input"
            if self.config.input_size is None:
                # 기본 입력 크기 설정
                self.config.input_size = [1, 1, 28, 28]  # 기본 MNIST 크기
                logger.warning(
                    f"입력 크기가 설정되지 않았습니다. 기본값 사용: {self.config.input_size}"
                )

            # 최적화 프로필 설정 (min, opt, max)
            min_shape = tuple(self.config.input_size)
            opt_shape = tuple(self.config.input_size)
            max_shape = tuple(
                [d * 2 if i == 0 else d for i, d in enumerate(self.config.input_size)]
            )  # 배치 크기만 2배로

            # 프로필에 입력 설정 추가
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)

            # 프로필 설정 추가
            config.add_optimization_profile(profile)

            # 엔진 빌드
            try:
                # build_serialized_network 메서드 사용 (최신 TensorRT)
                serialized_network = builder.build_serialized_network(network, config)

                if serialized_network is None:
                    logger.error("TensorRT 직렬화 네트워크 생성 실패")
                    return None

                runtime_class = getattr(trt, "Runtime", None)
                if not runtime_class:
                    logger.error("TensorRT Runtime 클래스를 찾을 수 없습니다.")
                    return None

                runtime = runtime_class(self._trt_logger)
                engine = runtime.deserialize_cuda_engine(serialized_network)

                if engine is None:
                    logger.error("TensorRT 엔진 생성 실패")
                    return None

                # 엔진 파일 저장
                engine_path = self._get_engine_path()
                try:
                    with open(engine_path, "wb") as f:
                        f.write(engine.serialize())

                    # 메타데이터 파일 생성
                    engine_hash = self._get_engine_hash(engine_path)
                    metadata_path = os.path.join(
                        self.config.engine_cache_dir,
                        f"{self.config.engine_cache_prefix}_{engine_hash}_metadata.json",
                    )

                    # 메타데이터 파일 내용 생성
                    metadata = {
                        "engine_hash": engine_hash,
                        "timestamp": time.time(),
                        "config": {
                            k: str(v)
                            for k, v in self.config.__dict__.items()
                            if not k.startswith("_")
                        },
                        "auto_generated": False,
                        "version": self.config.engine_cache_version,
                    }

                    # 디렉토리 생성 확인
                    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

                    # 메타데이터 파일 저장
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"엔진 메타데이터 파일 생성: {metadata_path}")

                except Exception as e:
                    logger.error(f"엔진 파일 저장 실패: {str(e)}")
                    return None

                logger.info(f"엔진 생성 완료: {engine_path}")
                return engine_path
            except Exception as e:
                logger.error(f"엔진 빌드 실패: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"엔진 생성 중 오류 발생: {str(e)}")
            return None

    def _convert_to_onnx(self, model: nn.Module, input_data: torch.Tensor) -> str:
        """PyTorch 모델을 ONNX로 변환

        Args:
            model: 변환할 PyTorch 모델
            input_data: 모델 입력 예제 데이터

        Returns:
            ONNX 파일 경로 또는 빈 문자열(변환 실패시)
        """
        try:
            # 입력 정보 저장
            if self.config.input_size is None:
                self.config.input_size = list(input_data.shape)
                self.logger.info(f"입력 크기 설정: {self.config.input_size}")

            # 동적 축 설정 (배치 크기만 동적으로)
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

            # 더미 입력 생성
            dummy_input = input_data.clone().float()

            # ONNX 파일 경로
            onnx_path = os.path.join(self.config.onnx_cache_dir, "model.onnx")

            # ONNX 모듈 가져오기 시도
            try:
                # onnx 모듈은 직접 사용하지 않지만, 설치되어 있는지 확인하기 위해 시도
                import importlib.util

                if importlib.util.find_spec("onnx") is None:
                    raise ImportError("onnx 모듈이 설치되어 있지 않습니다")

                # ONNX 내보내기 (일관된 입력/출력 이름 사용)
                torch.onnx.export(
                    model,
                    (dummy_input,),
                    onnx_path,
                    input_names=["input"],  # TensorRT에서 사용할 이름과 일치시킴
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    opset_version=11,  # 호환성을 위한 ONNX 버전 지정
                    do_constant_folding=True,  # 상수 폴딩 최적화
                    verbose=False,
                )

                self.logger.info(
                    f"ONNX 내보내기 완료: {onnx_path}, 입력 shape={dummy_input.shape}, dtype={dummy_input.dtype}"
                )
                return onnx_path
            except ImportError as e:
                self.logger.warning(
                    f"ONNX 모듈이 설치되어 있지 않습니다: {str(e)}. PyTorch 모델을 직접 사용합니다."
                )
                self._current_model = model  # ONNX 변환 대신 PyTorch 모델 직접 사용
                return ""
        except Exception as e:
            self.logger.error(f"ONNX 변환 실패: {str(e)}")
            return ""
