"""비동기 IO 처리 모듈 (GPU 우선순위 버전)

GPU 메모리 우선순위와 스마트 컴퓨테이션을 활용한 고성능 비동기 I/O 시스템
"""

import asyncio
import aiofiles
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import pickle
import zlib
import numpy as np
import torch
from dataclasses import dataclass, field
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from .unified_logging import get_logger
from .performance_optimizer import get_smart_computation_engine
from .memory_manager import get_memory_manager

logger = get_logger(__name__)


@dataclass
class AsyncIOConfig:
    """비동기 IO 설정 클래스 (GPU 우선순위)"""

    chunk_size: int = 1 << 20  # 1MB
    max_concurrent_ops: int = 4
    compression_level: int = 6
    compression_threshold: int = 1 << 20  # 1MB
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    use_compression: bool = True
    use_encryption: bool = False
    encryption_key: str = ""
    encryption_iv: str = ""
    # GPU 관련 설정
    prefer_gpu_memory: bool = True
    gpu_memory_threshold: float = 0.8
    auto_memory_management: bool = True
    parallel_gpu_ops: bool = True
    stats: Dict[str, Any] = field(default_factory=dict)


class GPUAsyncIOManager:
    """GPU 우선순위 비동기 IO 관리자 클래스"""

    def __init__(self, config: Optional[AsyncIOConfig] = None):
        self.config = config or AsyncIOConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_ops)
        self.computation_engine = get_smart_computation_engine()
        self.memory_manager = get_memory_manager()
        self.gpu_available = torch.cuda.is_available()

        self._stats = {
            "read_ops": 0,
            "write_ops": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "read_time": 0.0,
            "write_time": 0.0,
            "errors": 0,
            "retries": 0,
            "gpu_ops": 0,
            "cpu_fallbacks": 0,
            "memory_transfers": 0,
        }

        # GPU 메모리 풀 초기화
        if self.gpu_available and self.config.prefer_gpu_memory:
            self._initialize_gpu_memory_pool()

    def _initialize_gpu_memory_pool(self):
        """GPU 메모리 풀 초기화"""
        try:
            # 메모리 풀 사전 할당
            with self.memory_manager.allocation_scope():
                dummy_tensor = torch.zeros(1024, 1024, device="cuda")
                del dummy_tensor
                torch.cuda.empty_cache()
            logger.info("GPU 메모리 풀 초기화 완료")
        except Exception as e:
            logger.warning(f"GPU 메모리 풀 초기화 실패: {e}")

    async def smart_read_file(
        self, file_path: Union[str, Path], load_to_gpu: bool = True
    ) -> Optional[Union[bytes, torch.Tensor]]:
        """스마트 파일 읽기 - GPU 메모리 우선순위"""
        start_time = time.time()
        retry_count = 0

        while retry_count < self.config.retry_count:
            try:
                async with self._semaphore:
                    # 파일 크기 확인
                    file_size = Path(file_path).stat().st_size

                    # GPU 메모리 할당 가능 여부 확인
                    can_use_gpu = (
                        load_to_gpu
                        and self.gpu_available
                        and self._can_allocate_gpu_memory(file_size)
                    )

                    # 파일 읽기
                    async with aiofiles.open(file_path, "rb") as f:
                        data = await f.read()

                    # GPU 메모리로 로드
                    if can_use_gpu:
                        try:
                            # NumPy 배열로 변환 후 GPU 텐서로 변환
                            np_data = np.frombuffer(data, dtype=np.uint8)
                            gpu_tensor = torch.from_numpy(np_data).cuda()

                            self._stats["gpu_ops"] += 1
                            logger.debug(
                                f"파일을 GPU 메모리로 로드: {file_path}, 크기: {file_size/1024**2:.1f}MB"
                            )

                            self._update_read_stats(len(data), start_time)
                            return gpu_tensor

                        except Exception as e:
                            logger.warning(f"GPU 로드 실패, CPU로 폴백: {e}")
                            self._stats["cpu_fallbacks"] += 1

                    # CPU 메모리 사용
                    self._update_read_stats(len(data), start_time)
                    return data

            except Exception as e:
                retry_count += 1
                self._stats["errors"] += 1
                self._stats["retries"] += 1
                if retry_count < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_delay)
                    logger.warning(
                        f"파일 읽기 재시도 {retry_count}/{self.config.retry_count}: {e}"
                    )
                else:
                    logger.error(f"파일 읽기 실패: {e}")
                    return None

        return None

    async def smart_write_file(
        self, file_path: Union[str, Path], data: Union[bytes, torch.Tensor, np.ndarray]
    ) -> bool:
        """스마트 파일 쓰기 - GPU 데이터 자동 처리"""
        start_time = time.time()
        retry_count = 0

        # 데이터 타입에 따른 처리
        if isinstance(data, torch.Tensor):
            if data.is_cuda:
                # GPU 텐서를 CPU로 이동
                data = data.cpu().numpy().tobytes()
                self._stats["memory_transfers"] += 1
            else:
                data = data.numpy().tobytes()
        elif isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, bytes):
            # 기타 타입은 pickle로 직렬화
            data = pickle.dumps(data)

        # 데이터 압축 (GPU 가속 가능 시)
        if (
            self.config.use_compression
            and len(data) > self.config.compression_threshold
        ):
            data = await self._smart_compress(data)

        while retry_count < self.config.retry_count:
            try:
                async with self._semaphore:
                    # 디렉토리 생성
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(data)

                    self._update_write_stats(len(data), start_time)
                    return True

            except Exception as e:
                retry_count += 1
                self._stats["errors"] += 1
                self._stats["retries"] += 1
                if retry_count < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_delay)
                    logger.warning(
                        f"파일 쓰기 재시도 {retry_count}/{self.config.retry_count}: {e}"
                    )
                else:
                    logger.error(f"파일 쓰기 실패: {e}")
                    return False

        return False

    async def _smart_compress(self, data: bytes) -> bytes:
        """스마트 압축 - GPU 가속 가능 시 활용"""
        try:
            # GPU에서 압축 처리 가능한지 확인
            if self.gpu_available and len(data) > 10 * 1024 * 1024:  # 10MB 이상
                # 대용량 데이터는 멀티스레드로 압축
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    compressed = await loop.run_in_executor(
                        executor, zlib.compress, data, self.config.compression_level
                    )
                return compressed
            else:
                # 소용량 데이터는 직접 압축
                return zlib.compress(data, level=self.config.compression_level)
        except Exception as e:
            logger.warning(f"압축 실패, 원본 데이터 사용: {e}")
            return data

    def _can_allocate_gpu_memory(self, size_bytes: int) -> bool:
        """GPU 메모리 할당 가능 여부 확인"""
        if not self.gpu_available:
            return False

        try:
            gpu_usage = self.memory_manager.get_memory_usage("gpu")
            return gpu_usage < self.config.gpu_memory_threshold
        except Exception:
            return False

    def _update_read_stats(self, bytes_read: int, start_time: float):
        """읽기 통계 업데이트"""
        self._stats["read_ops"] += 1
        self._stats["read_bytes"] += bytes_read
        self._stats["read_time"] += time.time() - start_time

    def _update_write_stats(self, bytes_written: int, start_time: float):
        """쓰기 통계 업데이트"""
        self._stats["write_ops"] += 1
        self._stats["write_bytes"] += bytes_written
        self._stats["write_time"] += time.time() - start_time

    async def smart_read_tensor(
        self, file_path: Union[str, Path], dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """텐서 파일 스마트 읽기 - GPU 우선순위"""
        try:
            # 파일 읽기
            data = await self.smart_read_file(file_path, load_to_gpu=True)

            if data is None:
                return None

            if isinstance(data, torch.Tensor):
                # 이미 GPU 텐서인 경우
                return data.to(dtype)
            else:
                # bytes 데이터를 텐서로 변환
                if data.startswith(b"PK"):  # ZIP 형식 (PyTorch 저장 형식)
                    # PyTorch 형식 로드
                    with open(file_path, "rb") as f:
                        tensor = torch.load(
                            f, map_location="cuda" if self.gpu_available else "cpu"
                        )
                    return tensor.to(dtype)
                else:
                    # NumPy 형식으로 가정
                    np_data = np.frombuffer(data, dtype=np.float32)
                    tensor = torch.from_numpy(np_data).to(dtype)

                    # GPU로 이동 (가능한 경우)
                    if self.gpu_available and self._can_allocate_gpu_memory(
                        tensor.numel() * tensor.element_size()
                    ):
                        tensor = tensor.cuda()

                    return tensor

        except Exception as e:
            logger.error(f"텐서 읽기 실패: {e}")
            return None

    async def smart_write_tensor(
        self, file_path: Union[str, Path], tensor: torch.Tensor, format: str = "pytorch"
    ) -> bool:
        """텐서 파일 스마트 쓰기"""
        try:
            if format.lower() == "pytorch":
                # PyTorch 형식으로 저장
                def save_pytorch():
                    torch.save(tensor, file_path)

                # 비동기 실행
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    await loop.run_in_executor(executor, save_pytorch)

                return True
            else:
                # NumPy 형식으로 저장
                return await self.smart_write_file(file_path, tensor)

        except Exception as e:
            logger.error(f"텐서 쓰기 실패: {e}")
            return False

    async def parallel_tensor_operations(
        self, operations: List[Tuple[str, Union[str, Path], Any]]
    ) -> List[Any]:
        """병렬 텐서 연산 - GPU 메모리 효율적 처리"""
        results = []

        # GPU 메모리 사용량에 따라 배치 크기 조정
        gpu_usage = (
            self.memory_manager.get_memory_usage("gpu") if self.gpu_available else 1.0
        )

        if gpu_usage > 0.8:
            batch_size = 1  # 메모리 부족 시 순차 처리
        elif gpu_usage > 0.6:
            batch_size = 2  # 적당한 병렬 처리
        else:
            batch_size = min(4, len(operations))  # 최대 병렬 처리

        # 배치별 처리
        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]

            # 병렬 실행
            tasks = []
            for op_type, file_path, data in batch:
                if op_type == "read":
                    tasks.append(self.smart_read_tensor(file_path))
                elif op_type == "write":
                    tasks.append(self.smart_write_tensor(file_path, data))
                elif op_type == "read_file":
                    tasks.append(self.smart_read_file(file_path))
                elif op_type == "write_file":
                    tasks.append(self.smart_write_file(file_path, data))

            # 배치 결과 수집
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)

            # GPU 메모리 정리 (필요시)
            if self.gpu_available and self.config.auto_memory_management:
                if self.memory_manager.should_cleanup():
                    self.memory_manager.cleanup()

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self._stats.copy()

        # 추가 메트릭 계산
        if stats["read_ops"] > 0:
            stats["avg_read_time"] = stats["read_time"] / stats["read_ops"]
            stats["avg_read_throughput_mbps"] = (
                (stats["read_bytes"] / (1024**2)) / stats["read_time"]
                if stats["read_time"] > 0
                else 0
            )

        if stats["write_ops"] > 0:
            stats["avg_write_time"] = stats["write_time"] / stats["write_ops"]
            stats["avg_write_throughput_mbps"] = (
                (stats["write_bytes"] / (1024**2)) / stats["write_time"]
                if stats["write_time"] > 0
                else 0
            )

        # GPU 활용률
        total_ops = stats["read_ops"] + stats["write_ops"]
        if total_ops > 0:
            stats["gpu_utilization_percent"] = (stats["gpu_ops"] / total_ops) * 100
            stats["cpu_fallback_rate"] = (stats["cpu_fallbacks"] / total_ops) * 100

        return stats


# 전역 GPU 비동기 IO 관리자
_gpu_async_io_manager = None


def get_gpu_async_io_manager(
    config: Optional[AsyncIOConfig] = None,
) -> GPUAsyncIOManager:
    """GPU 비동기 IO 관리자 인스턴스 반환"""
    global _gpu_async_io_manager
    if _gpu_async_io_manager is None:
        _gpu_async_io_manager = GPUAsyncIOManager(config)
    return _gpu_async_io_manager


# 편의 함수들
async def smart_read_file(
    file_path: Union[str, Path], load_to_gpu: bool = True
) -> Optional[Union[bytes, torch.Tensor]]:
    """스마트 파일 읽기 (편의 함수)"""
    manager = get_gpu_async_io_manager()
    return await manager.smart_read_file(file_path, load_to_gpu)


async def smart_write_file(
    file_path: Union[str, Path], data: Union[bytes, torch.Tensor, np.ndarray]
) -> bool:
    """스마트 파일 쓰기 (편의 함수)"""
    manager = get_gpu_async_io_manager()
    return await manager.smart_write_file(file_path, data)


async def smart_read_tensor(
    file_path: Union[str, Path], dtype: torch.dtype = torch.float32
) -> Optional[torch.Tensor]:
    """스마트 텐서 읽기 (편의 함수)"""
    manager = get_gpu_async_io_manager()
    return await manager.smart_read_tensor(file_path, dtype)


async def smart_write_tensor(
    file_path: Union[str, Path], tensor: torch.Tensor, format: str = "pytorch"
) -> bool:
    """스마트 텐서 쓰기 (편의 함수)"""
    manager = get_gpu_async_io_manager()
    return await manager.smart_write_tensor(file_path, tensor, format)


# 기존 AsyncIOManager 클래스 (하위 호환성 유지)
class AsyncIOManager(GPUAsyncIOManager):
    """기존 AsyncIOManager 클래스 (하위 호환성)"""

    async def read_file(self, file_path: Union[str, Path]) -> Optional[bytes]:
        """파일 비동기 읽기 (하위 호환성)"""
        result = await self.smart_read_file(file_path, load_to_gpu=False)
        if isinstance(result, torch.Tensor):
            return result.cpu().numpy().tobytes()
        return result

    async def write_file(self, file_path: Union[str, Path], data: bytes) -> bool:
        """파일 비동기 쓰기 (하위 호환성)"""
        return await self.smart_write_file(file_path, data)

    def sync_read_file(self, file_path: Union[str, Path]) -> Optional[bytes]:
        """파일 동기식 읽기 (하위 호환성)"""
        start_time = time.time()
        retry_count = 0

        while retry_count < self.config.retry_count:
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                    self._update_read_stats(len(data), start_time)
                    return data
            except Exception as e:
                retry_count += 1
                self._stats["errors"] += 1
                self._stats["retries"] += 1
                if retry_count < self.config.retry_count:
                    time.sleep(self.config.retry_delay)
                    logger.warning(
                        f"파일 읽기 재시도 {retry_count}/{self.config.retry_count}: {e}"
                    )
                else:
                    logger.error(f"파일 읽기 실패: {e}")
                    return None
        return None

    async def read_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """JSON 파일 비동기 읽기 (하위 호환성)"""
        data = await self.read_file(file_path)
        if data is not None:
            try:
                return json.loads(data)
            except Exception as e:
                logger.error(f"JSON 파싱 실패: {e}")
                return None
        return None

    async def write_json(
        self, file_path: Union[str, Path], data: Dict[str, Any]
    ) -> bool:
        """JSON 파일 비동기 쓰기 (하위 호환성)"""
        try:
            json_data = json.dumps(data)
            return await self.write_file(file_path, json_data.encode())
        except Exception as e:
            logger.error(f"JSON 직렬화 실패: {e}")
            return False

    async def read_pickle(self, file_path: Union[str, Path]) -> Optional[Any]:
        """Pickle 파일 비동기 읽기 (하위 호환성)"""
        data = await self.read_file(file_path)
        if data is not None:
            try:
                return pickle.loads(data)
            except Exception as e:
                logger.error(f"Pickle 파싱 실패: {e}")
                return None
        return None

    async def write_pickle(self, file_path: Union[str, Path], data: Any) -> bool:
        """Pickle 파일 비동기 쓰기 (하위 호환성)"""
        try:
            pickle_data = pickle.dumps(data)
            return await self.write_file(file_path, pickle_data)
        except Exception as e:
            logger.error(f"Pickle 직렬화 실패: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환 (하위 호환성)"""
        return self.get_performance_stats()

    def reset_stats(self):
        """통계 초기화 (하위 호환성)"""
        self._stats = {
            "read_ops": 0,
            "write_ops": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "read_time": 0.0,
            "write_time": 0.0,
            "errors": 0,
            "retries": 0,
            "gpu_ops": 0,
            "cpu_fallbacks": 0,
            "memory_transfers": 0,
        }
