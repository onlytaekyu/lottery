"""비동기 IO 처리 모듈

이 모듈은 파일 입출력 비동기 처리를 위한 클래스를 제공합니다.
"""

import asyncio
import aiofiles
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import pickle
import zlib
from dataclasses import dataclass, field
import time

from .error_handler_refactored import get_logger

logger = get_logger(__name__)


@dataclass
class AsyncIOConfig:
    """비동기 IO 설정 클래스"""

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
    stats: Dict[str, Any] = field(default_factory=dict)


class AsyncIOManager:
    """비동기 IO 관리자 클래스"""

    def __init__(self, config: Optional[AsyncIOConfig] = None):
        self.config = config or AsyncIOConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_ops)
        self._stats = {
            "read_ops": 0,
            "write_ops": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "read_time": 0.0,
            "write_time": 0.0,
            "errors": 0,
            "retries": 0,
        }

    async def read_file(self, file_path: Union[str, Path]) -> Optional[bytes]:
        """파일 비동기 읽기"""
        start_time = time.time()
        retry_count = 0

        while retry_count < self.config.retry_count:
            try:
                async with self._semaphore:
                    async with aiofiles.open(file_path, "rb") as f:
                        data = await f.read()
                        self._stats["read_ops"] += 1
                        self._stats["read_bytes"] += len(data)
                        self._stats["read_time"] += time.time() - start_time
                        return data
            except Exception as e:
                retry_count += 1
                self._stats["errors"] += 1
                self._stats["retries"] += 1
                if retry_count < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_delay)
                    logger.warning(
                        f"파일 읽기 재시도 {retry_count}/{self.config.retry_count}: {str(e)}"
                    )
                else:
                    logger.error(f"파일 읽기 실패: {str(e)}")
                    return None

    def sync_read_file(self, file_path: Union[str, Path]) -> Optional[bytes]:
        """파일 동기식 읽기 - 비동기 메서드의 동기 버전"""
        start_time = time.time()
        retry_count = 0

        while retry_count < self.config.retry_count:
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                    self._stats["read_ops"] += 1
                    self._stats["read_bytes"] += len(data)
                    self._stats["read_time"] += time.time() - start_time
                    return data
            except Exception as e:
                retry_count += 1
                self._stats["errors"] += 1
                self._stats["retries"] += 1
                if retry_count < self.config.retry_count:
                    time.sleep(self.config.retry_delay)
                    logger.warning(
                        f"파일 읽기 재시도 {retry_count}/{self.config.retry_count}: {str(e)}"
                    )
                else:
                    logger.error(f"파일 읽기 실패: {str(e)}")
                    return None

        return None

    async def write_file(self, file_path: Union[str, Path], data: bytes) -> bool:
        """파일 비동기 쓰기"""
        start_time = time.time()
        retry_count = 0

        # 데이터 압축
        if (
            self.config.use_compression
            and len(data) > self.config.compression_threshold
        ):
            data = zlib.compress(data, level=self.config.compression_level)

        while retry_count < self.config.retry_count:
            try:
                async with self._semaphore:
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(data)
                        self._stats["write_ops"] += 1
                        self._stats["write_bytes"] += len(data)
                        self._stats["write_time"] += time.time() - start_time
                        return True
            except Exception as e:
                retry_count += 1
                self._stats["errors"] += 1
                self._stats["retries"] += 1
                if retry_count < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_delay)
                    logger.warning(
                        f"파일 쓰기 재시도 {retry_count}/{self.config.retry_count}: {str(e)}"
                    )
                else:
                    logger.error(f"파일 쓰기 실패: {str(e)}")
                    return False
        return False

    async def read_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """JSON 파일 비동기 읽기"""
        data = await self.read_file(file_path)
        if data is not None:
            try:
                return json.loads(data)
            except Exception as e:
                logger.error(f"JSON 파싱 실패: {str(e)}")
                return None
        return None

    async def write_json(
        self, file_path: Union[str, Path], data: Dict[str, Any]
    ) -> bool:
        """JSON 파일 비동기 쓰기"""
        try:
            json_data = json.dumps(data)
            return await self.write_file(file_path, json_data.encode())
        except Exception as e:
            logger.error(f"JSON 직렬화 실패: {str(e)}")
            return False

    async def read_pickle(self, file_path: Union[str, Path]) -> Optional[Any]:
        """Pickle 파일 비동기 읽기"""
        data = await self.read_file(file_path)
        if data is not None:
            try:
                return pickle.loads(data)
            except Exception as e:
                logger.error(f"Pickle 파싱 실패: {str(e)}")
                return None
        return None

    async def write_pickle(self, file_path: Union[str, Path], data: Any) -> bool:
        """Pickle 파일 비동기 쓰기"""
        try:
            pickle_data = pickle.dumps(data)
            return await self.write_file(file_path, pickle_data)
        except Exception as e:
            logger.error(f"Pickle 직렬화 실패: {str(e)}")
            return False

    async def read_lines(self, file_path: Union[str, Path]) -> Optional[List[str]]:
        """파일 라인별 비동기 읽기"""
        try:
            async with aiofiles.open(file_path, "r") as f:
                return await f.readlines()
        except Exception as e:
            logger.error(f"파일 라인 읽기 실패: {str(e)}")
            return None

    async def write_lines(self, file_path: Union[str, Path], lines: List[str]) -> bool:
        """파일 라인별 비동기 쓰기"""
        try:
            async with aiofiles.open(file_path, "w") as f:
                await f.writelines(lines)
                return True
        except Exception as e:
            logger.error(f"파일 라인 쓰기 실패: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """IO 통계 반환"""
        return self._stats.copy()

    def reset_stats(self):
        """IO 통계 초기화"""
        self._stats = {
            "read_ops": 0,
            "write_ops": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "read_time": 0.0,
            "write_time": 0.0,
            "errors": 0,
            "retries": 0,
        }
