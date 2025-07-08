"""
비동기 처리 완전 통합 시스템
모든 I/O 작업을 비동기로 통합 관리
"""

import asyncio
import aiofiles
import aiohttp
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager
import pickle
import json

from .unified_logging import get_logger
from .unified_config import get_config
from .factory import get_singleton_instance

logger = get_logger(__name__)


class AsyncTaskType(Enum):
    """비동기 작업 타입"""

    FILE_IO = "file_io"
    NETWORK_IO = "network_io"
    CACHE_OPERATION = "cache_operation"
    DATA_PROCESSING = "data_processing"
    MODEL_OPERATION = "model_operation"


@dataclass
class AsyncTask:
    """비동기 작업 정보"""

    id: str
    task_type: AsyncTaskType
    coroutine: Awaitable
    priority: int = 1
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            import time

            self.created_at = time.time()


class AsyncFileManager:
    """비동기 파일 관리자"""

    def __init__(self):
        self.file_locks: Dict[str, asyncio.Lock] = {}
        self.open_files: Dict[str, aiofiles.threadpool.text.AsyncTextIOWrapper] = {}

    async def read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """비동기 파일 읽기"""
        async with aiofiles.open(file_path, "r", encoding=encoding) as f:
            content = await f.read()
        logger.debug(f"파일 읽기 완료: {file_path}")
        return content

    async def write_file(self, file_path: str, content: str, encoding: str = "utf-8"):
        """비동기 파일 쓰기"""
        # 파일별 락 생성
        if file_path not in self.file_locks:
            self.file_locks[file_path] = asyncio.Lock()

        async with self.file_locks[file_path]:
            async with aiofiles.open(file_path, "w", encoding=encoding) as f:
                await f.write(content)

        logger.debug(f"파일 쓰기 완료: {file_path}")

    async def read_binary_file(self, file_path: str) -> bytes:
        """비동기 바이너리 파일 읽기"""
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()
        logger.debug(f"바이너리 파일 읽기 완료: {file_path}")
        return content

    async def write_binary_file(self, file_path: str, content: bytes):
        """비동기 바이너리 파일 쓰기"""
        if file_path not in self.file_locks:
            self.file_locks[file_path] = asyncio.Lock()

        async with self.file_locks[file_path]:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)

        logger.debug(f"바이너리 파일 쓰기 완료: {file_path}")

    async def read_json_file(self, file_path: str) -> Dict[str, Any]:
        """비동기 JSON 파일 읽기"""
        content = await self.read_file(file_path)
        return json.loads(content)

    async def write_json_file(self, file_path: str, data: Dict[str, Any]):
        """비동기 JSON 파일 쓰기"""
        content = json.dumps(data, indent=2, ensure_ascii=False)
        await self.write_file(file_path, content)

    async def read_pickle_file(self, file_path: str) -> Any:
        """비동기 피클 파일 읽기"""
        content = await self.read_binary_file(file_path)
        return pickle.loads(content)

    async def write_pickle_file(self, file_path: str, data: Any):
        """비동기 피클 파일 쓰기"""
        content = pickle.dumps(data)
        await self.write_binary_file(file_path, content)


class AsyncCacheManager:
    """비동기 캐시 관리자"""

    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_locks: Dict[str, asyncio.Lock] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = 1000
        self.ttl_seconds = 3600  # 1시간

    async def get(self, key: str, default: Any = None) -> Any:
        """비동기 캐시 조회"""
        if key not in self.cache:
            return default

        # TTL 확인
        metadata = self.cache_metadata.get(key, {})
        if self._is_expired(metadata):
            await self.delete(key)
            return default

        # 접근 시간 업데이트
        import time

        metadata["last_accessed"] = time.time()

        return self.cache[key]

    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """비동기 캐시 저장"""
        # 캐시 크기 제한 확인
        if len(self.cache) >= self.max_cache_size:
            await self._evict_oldest()

        # 키별 락 생성
        if key not in self.cache_locks:
            self.cache_locks[key] = asyncio.Lock()

        async with self.cache_locks[key]:
            self.cache[key] = value

            import time

            self.cache_metadata[key] = {
                "created_at": time.time(),
                "last_accessed": time.time(),
                "ttl": ttl or self.ttl_seconds,
                "size": len(str(value)),
            }

        logger.debug(f"캐시 저장: {key}")

    async def delete(self, key: str) -> bool:
        """비동기 캐시 삭제"""
        if key in self.cache:
            del self.cache[key]
            del self.cache_metadata[key]
            if key in self.cache_locks:
                del self.cache_locks[key]
            logger.debug(f"캐시 삭제: {key}")
            return True
        return False

    async def clear(self):
        """비동기 캐시 전체 삭제"""
        self.cache.clear()
        self.cache_metadata.clear()
        self.cache_locks.clear()
        logger.info("캐시 전체 삭제 완료")

    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """캐시 만료 확인"""
        if not metadata:
            return True

        import time

        created_at = metadata.get("created_at", 0)
        ttl = metadata.get("ttl", self.ttl_seconds)

        return (time.time() - created_at) > ttl

    async def _evict_oldest(self):
        """가장 오래된 캐시 항목 제거 - 최적화된 LRU 알고리즘"""
        if not self.cache:
            return
        
        # 가장 오래된 캐시 항목 찾기
        oldest_key = None
        oldest_time = float('inf')
        
        for key, metadata in self.cache_metadata.items():
            if key in self.cache:
                last_accessed = metadata.get('last_accessed', 0)
                if last_accessed < oldest_time:
                    oldest_time = last_accessed
                    oldest_key = key
        
        # 가장 오래된 항목 제거
        if oldest_key:
            await self.delete(oldest_key)
            logger.debug(f"캐시 LRU 제거: {oldest_key}")
        
        # 메모리 부족 시 추가 정리
        if len(self.cache) > self.max_cache_size * 0.9:
            # 상위 10% 제거
            evict_count = max(1, int(len(self.cache) * 0.1))
            sorted_items = sorted(
                self.cache_metadata.items(),
                key=lambda x: x[1].get('last_accessed', 0)
            )
            
            for key, _ in sorted_items[:evict_count]:
                if key in self.cache:
                    await self.delete(key)
                    logger.debug(f"캐시 배치 제거: {key}")
        if not self.cache_metadata:
            return

        oldest_key = min(
            self.cache_metadata.keys(),
            key=lambda k: self.cache_metadata[k].get("last_accessed", 0),
        )

        await self.delete(oldest_key)
        logger.debug(f"오래된 캐시 제거: {oldest_key}")


class AsyncDataLoader:
    """비동기 데이터 로더"""

    def __init__(self):
        self.cache_manager = AsyncCacheManager()
        self.file_manager = AsyncFileManager()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def load_data(
        self, source: str, cache_key: str = None, force_reload: bool = False
    ) -> Any:
        """비동기 데이터 로딩"""
        # 캐시 확인
        if cache_key and not force_reload:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data is not None:
                logger.debug(f"캐시에서 데이터 로드: {cache_key}")
                return cached_data

        # 데이터 소스 타입 판단
        if source.startswith(("http://", "https://")):
            data = await self._load_from_url(source)
        elif source.endswith(".json"):
            data = await self.file_manager.read_json_file(source)
        elif source.endswith(".pkl"):
            data = await self.file_manager.read_pickle_file(source)
        else:
            data = await self.file_manager.read_file(source)

        # 캐시 저장
        if cache_key:
            await self.cache_manager.set(cache_key, data)

        logger.debug(f"데이터 로드 완료: {source}")
        return data

    async def _load_from_url(self, url: str) -> Any:
        """URL에서 데이터 로드"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        async with self.session.get(url) as response:
            if response.content_type == "application/json":
                return await response.json()
            else:
                return await response.text()

    async def save_data(self, data: Any, destination: str, cache_key: str = None):
        """비동기 데이터 저장"""
        if destination.endswith(".json"):
            await self.file_manager.write_json_file(destination, data)
        elif destination.endswith(".pkl"):
            await self.file_manager.write_pickle_file(destination, data)
        else:
            await self.file_manager.write_file(destination, str(data))

        # 캐시 업데이트
        if cache_key:
            await self.cache_manager.set(cache_key, data)

        logger.debug(f"데이터 저장 완료: {destination}")

    async def batch_load(
        self, sources: List[str], cache_keys: List[str] = None
    ) -> List[Any]:
        """배치 데이터 로딩"""
        if cache_keys is None:
            cache_keys = [None] * len(sources)

        tasks = []
        for source, cache_key in zip(sources, cache_keys):
            task = self.load_data(source, cache_key)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"데이터 로드 실패: {sources[i]}: {result}")
                results[i] = None

        return results


class AsyncTaskManager:
    """비동기 작업 관리자"""

    def __init__(self):
        self.tasks: Dict[str, AsyncTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent_tasks = 10
        self.task_counter = 0
        self._lock = asyncio.Lock()

    async def submit_task(
        self,
        coroutine: Awaitable,
        task_type: AsyncTaskType = AsyncTaskType.DATA_PROCESSING,
        priority: int = 1,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """비동기 작업 제출"""
        async with self._lock:
            self.task_counter += 1
            task_id = f"task_{self.task_counter:06d}"

        task = AsyncTask(
            id=task_id,
            task_type=task_type,
            coroutine=coroutine,
            priority=priority,
            timeout=timeout,
            callback=callback,
        )

        self.tasks[task_id] = task
        await self.task_queue.put(task)

        logger.debug(f"작업 제출: {task_id} ({task_type.value})")
        return task_id

    async def get_task_result(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Any:
        """작업 결과 조회"""
        if task_id not in self.running_tasks:
            raise ValueError(f"작업을 찾을 수 없음: {task_id}")

        task = self.running_tasks[task_id]

        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"작업 시간 초과: {task_id}")
            task.cancel()
            raise
        finally:
            # 정리
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            if task_id in self.tasks:
                del self.tasks[task_id]

    async def cancel_task(self, task_id: str) -> bool:
        """작업 취소"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            del self.running_tasks[task_id]
            logger.debug(f"작업 취소: {task_id}")
            return True
        return False

    async def start_worker(self):
        """작업 워커 시작"""
        while True:
            try:
                # 큐에서 작업 가져오기
                task = await self.task_queue.get()

                # 동시 실행 제한 확인
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    # 큐에 다시 넣기
                    await self.task_queue.put(task)
                    await asyncio.sleep(0.1)
                    continue

                # 작업 실행
                async_task = asyncio.create_task(self._execute_task(task))
                self.running_tasks[task.id] = async_task

                # 큐 작업 완료 표시
                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"작업 워커 오류: {e}")
                await asyncio.sleep(1)

    async def _execute_task(self, task: AsyncTask) -> Any:
        """작업 실행"""
        try:
            if task.timeout:
                result = await asyncio.wait_for(task.coroutine, timeout=task.timeout)
            else:
                result = await task.coroutine

            # 콜백 실행
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(result)
                    else:
                        task.callback(result)
                except Exception as e:
                    logger.error(f"콜백 실행 오류: {e}")

            logger.debug(f"작업 완료: {task.id}")
            return result

        except Exception as e:
            logger.error(f"작업 실행 오류: {task.id}: {e}")
            raise
        finally:
            # 정리
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            if task.id in self.tasks:
                del self.tasks[task.id]


class UnifiedAsyncManager:
    """통합 비동기 관리자"""

    def __init__(self):
        self.config = get_config("main").get_nested("utils.async", {})

        # 하위 관리자들
        self.file_manager = AsyncFileManager()
        self.cache_manager = AsyncCacheManager()
        self.data_loader = AsyncDataLoader()
        self.task_manager = AsyncTaskManager()

        # 설정
        self.task_manager.max_concurrent_tasks = self.config.get(
            "max_concurrent_tasks", 10
        )
        self.cache_manager.max_cache_size = self.config.get("max_cache_size", 1000)
        self.cache_manager.ttl_seconds = self.config.get("cache_ttl_seconds", 3600)

        # 워커 시작
        self._worker_task = None

        logger.info("✅ 통합 비동기 관리자 초기화")

    async def start(self):
        """비동기 관리자 시작"""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self.task_manager.start_worker())
            logger.info("비동기 워커 시작")

    async def stop(self):
        """비동기 관리자 정지"""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            logger.info("비동기 워커 정지")

    @asynccontextmanager
    async def session(self):
        """비동기 세션 컨텍스트"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    # 편의 메서드들
    async def read_file(self, file_path: str) -> str:
        """파일 읽기"""
        return await self.file_manager.read_file(file_path)

    async def write_file(self, file_path: str, content: str):
        """파일 쓰기"""
        await self.file_manager.write_file(file_path, content)

    async def cache_get(self, key: str, default: Any = None) -> Any:
        """캐시 조회"""
        return await self.cache_manager.get(key, default)

    async def cache_set(self, key: str, value: Any, ttl: Optional[float] = None):
        """캐시 저장"""
        await self.cache_manager.set(key, value, ttl)

    async def load_data(self, source: str, cache_key: str = None) -> Any:
        """데이터 로딩"""
        return await self.data_loader.load_data(source, cache_key)

    async def submit_task(self, coroutine: Awaitable, **kwargs) -> str:
        """작업 제출"""
        return await self.task_manager.submit_task(coroutine, **kwargs)


# 싱글톤 인스턴스
def get_unified_async_manager() -> UnifiedAsyncManager:
    """통합 비동기 관리자 싱글톤 인스턴스 반환"""
    return get_singleton_instance(UnifiedAsyncManager)


# 편의 함수들
async def async_read_file(file_path: str) -> str:
    """비동기 파일 읽기 편의 함수"""
    manager = get_unified_async_manager()
    return await manager.read_file(file_path)


async def async_write_file(file_path: str, content: str):
    """비동기 파일 쓰기 편의 함수"""
    manager = get_unified_async_manager()
    await manager.write_file(file_path, content)


async def async_cache_get(key: str, default: Any = None) -> Any:
    """비동기 캐시 조회 편의 함수"""
    manager = get_unified_async_manager()
    return await manager.cache_get(key, default)


async def async_cache_set(key: str, value: Any, ttl: Optional[float] = None):
    """비동기 캐시 저장 편의 함수"""
    manager = get_unified_async_manager()
    await manager.cache_set(key, value, ttl)


async def async_load_data(source: str, cache_key: str = None) -> Any:
    """비동기 데이터 로딩 편의 함수"""
    manager = get_unified_async_manager()
    return await manager.load_data(source, cache_key)
