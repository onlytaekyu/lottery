"""
개인용 간단하고 확실한 비동기 I/O
"""

import asyncio
import aiofiles
import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from .unified_logging import get_logger
from .error_handler import get_error_handler
from .factory import get_singleton_instance

logger = get_logger(__name__)
error_handler = get_error_handler()


class GPUAsyncIO:
    """개인용 간단 비동기 I/O (GPU 메모리 확실히 정리)"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
        logger.info(f"✅ 개인용 비동기 I/O 초기화 (GPU: {self.gpu_available})")
        if self.gpu_available:
            torch.cuda.empty_cache()

    @asynccontextmanager
    async def gpu_safe_context(self):
        """GPU 메모리 확실히 정리하는 컨텍스트"""
        try:
            yield
        finally:
            # 개인용: 확실한 GPU 메모리 정리
            if self.gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    @error_handler.async_auto_recoverable(max_retries=2, delay=0.2)
    async def smart_write_file(
        self, file_path: Union[str, Path], data: Union[bytes, torch.Tensor, np.ndarray]
    ) -> bool:
        """간단하고 확실한 파일 쓰기"""

        async with self.gpu_safe_context():
            try:
                # 데이터 타입 처리
                if isinstance(data, torch.Tensor):
                    if data.is_cuda:
                        data = data.cpu().numpy().tobytes()
                    else:
                        data = data.numpy().tobytes()
                elif isinstance(data, np.ndarray):
                    data = data.tobytes()
                elif not isinstance(data, bytes):
                    data = str(data).encode()

                # 디렉토리 생성
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)

                # 비동기 쓰기
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(data)

                return True

            except Exception as e:
                logger.error(f"파일 쓰기 실패: {e}")
                return False

    @error_handler.async_auto_recoverable(max_retries=2, delay=0.2)
    async def smart_read_file(
        self, file_path: Union[str, Path], load_to_gpu: bool = False
    ) -> Optional[bytes]:
        """간단하고 확실한 파일 읽기"""

        async with self.gpu_safe_context():
            try:
                async with aiofiles.open(file_path, "rb") as f:
                    data = await f.read()

                # GPU 로드 요청시
                if load_to_gpu and self.gpu_available:
                    try:
                        np_data = np.frombuffer(data, dtype=np.uint8)
                        gpu_tensor = torch.from_numpy(np_data).cuda()
                        return gpu_tensor
                    except:
                        # GPU 실패시 그냥 bytes 반환
                        pass

                return data

            except Exception as e:
                logger.error(f"파일 읽기 실패: {e}")
                return None

    @error_handler.async_auto_recoverable(max_retries=2, delay=0.2)
    async def zero_copy_read(
        self, data_path: Union[str, Path], target_device: str = "cuda"
    ) -> Optional[torch.Tensor]:
        """
        Pinned Memory를 활용하여 데이터를 GPU에 효율적으로 로드합니다. (Zero-Copy Read)
        """
        if target_device != "cuda" or not self.gpu_available:
            # GPU 타겟이 아니면 일반 읽기 수행 후 CPU 텐서로 반환
            cpu_data = await self.smart_read_file(data_path, load_to_gpu=False)
            if isinstance(cpu_data, bytes):
                return torch.from_numpy(np.frombuffer(cpu_data, dtype=np.float32))
            return None

        try:
            # 1. 비동기로 파일 읽기
            async with aiofiles.open(data_path, "rb") as f:
                buffer = await f.read()

            # 2. 버퍼를 numpy 배열로 변환
            #    (실제 데이터 타입에 맞게 조정 필요, 여기서는 float32로 가정)
            host_array = np.frombuffer(buffer, dtype=np.float32)

            # 3. Pinned memory 텐서 생성
            pinned_tensor = torch.empty(
                host_array.shape, dtype=torch.float32, pin_memory=True
            )
            pinned_tensor.copy_(torch.from_numpy(host_array))

            # 4. 비동기적으로 GPU에 복사
            gpu_tensor = await asyncio.to_thread(
                lambda: pinned_tensor.to(self.device, non_blocking=True)
            )

            # 5. 스트림 동기화 (복사 완료 보장)
            torch.cuda.synchronize()

            logger.debug(f"Zero-copy read 성공: {data_path} -> {target_device}")
            return gpu_tensor

        except Exception as e:
            logger.error(f"Zero-copy read 실패: {data_path}, 오류: {e}", exc_info=True)
            return None

    def cleanup(self):
        """정리"""
        if self.gpu_available:
            torch.cuda.empty_cache()


def get_gpu_async_io_manager() -> GPUAsyncIO:
    """간단한 비동기 I/O 관리자 반환"""
    return get_singleton_instance(GPUAsyncIO)


# 하위 호환성
class AsyncIOManager(GPUAsyncIO):
    pass


def get_async_io_manager():
    return get_gpu_async_io_manager()
