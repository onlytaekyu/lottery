"""
고급 병렬 처리 관리자 (v3 - GPU 워커 통합)

CPU 및 GPU 워커 풀을 동적으로 관리하여,
작업 유형에 따라 최적의 리소스를 할당하는 고성능 병렬 처리 시스템.
"""

import os
import sys
import time
import psutil
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager

from .unified_logging import get_logger, log_exception

logger = get_logger(__name__)


def gpu_worker_process(task_queue: mp.Queue, result_queue: mp.Queue, gpu_id: int):
    """GPU 워커 프로세스 함수 (상태 보고 기능 추가)"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"✅ GPU 워커 시작 (PID: {os.getpid()}, GPU: {gpu_id})")

    while True:
        try:
            task_id, func, args, kwargs = task_queue.get()
            if func is None:
                break

            result_queue.put(("busy", gpu_id, task_id))
            result = func(*args, **kwargs, device=device)
            result_queue.put(("done", gpu_id, task_id, result))

        except Exception as e:
            log_exception(e, f"GPU 워커 (GPU: {gpu_id}) 오류 발생")
            result_queue.put(("error", gpu_id, task_id, e))


@dataclass
class ParallelManagerConfig:
    """고급 병렬 처리 관리자 설정"""

    cpu_workers: int = min(4, psutil.cpu_count(logical=False) or 1)
    gpu_workers: int = torch.cuda.device_count()
    chunk_size: int = 100
    timeout: float = 300.0
    memory_limit_mb: int = 1024
    enable_monitoring: bool = True
    auto_restart: bool = True
    restart_threshold: int = 100  # 작업 수행 후 재시작


class AdvancedParallelManager:
    """CPU 및 GPU 리소스를 지능적으로 통합 관리하는 병렬 처리기 (v4)"""

    def __init__(self, config: ParallelManagerConfig, max_retries: int = 3):
        self.config = config
        self.max_retries = max_retries
        self.cpu_executor = ProcessPoolExecutor(max_workers=self.config.cpu_workers)

        # GPU 워커 초기화
        self.gpu_task_queue = mp.Queue()
        self.gpu_result_queue = mp.Queue()
        self.gpu_processes = []
        self.gpu_worker_status = (
            {}
        )  # {gpu_id: {'status': 'idle'/'busy', 'last_job_time': float}}

        if self.config.gpu_workers > 0:
            for i in range(self.config.gpu_workers):
                proc = mp.Process(
                    target=gpu_worker_process,
                    args=(self.gpu_task_queue, self.gpu_result_queue, i),
                )
                proc.start()
                self.gpu_processes.append(proc)
                self.gpu_worker_status[i] = {
                    "status": "idle",
                    "last_job_time": time.time(),
                }

        self.thread_executor = ThreadPoolExecutor()

    def _get_idle_gpu_worker(self) -> Optional[int]:
        """가장 오랫동안 유휴 상태였던 GPU 워커 ID를 반환합니다."""
        idle_workers = {
            gpu_id: data
            for gpu_id, data in self.gpu_worker_status.items()
            if data["status"] == "idle"
        }
        if not idle_workers:
            return None
        # 마지막 작업 시간 기준으로 정렬하여 가장 오래된 워커 선택
        return min(
            idle_workers, key=lambda gpu_id: idle_workers[gpu_id]["last_job_time"]
        )

    def _execute_on_gpu(self, func: Callable, tasks: List[Any]) -> List[Any]:
        """GPU 워커 풀에서 작업 실행 (지능적 분배 및 재시도)"""
        num_tasks = len(tasks)
        results = [None] * num_tasks
        retries = [0] * num_tasks

        task_map = {i: task for i, task in enumerate(tasks)}

        while task_map:
            # 유휴 GPU 워커에게 작업 할당
            idle_worker_id = self._get_idle_gpu_worker()
            if idle_worker_id is not None and task_map:
                task_id, task_args = task_map.popitem()
                self.gpu_worker_status[idle_worker_id]["status"] = "busy"
                self.gpu_task_queue.put((task_id, func, task_args, {}))

            # 결과 처리
            status, gpu_id, task_id, data = self.gpu_result_queue.get()

            if status == "done":
                results[task_id] = data
                self.gpu_worker_status[gpu_id]["status"] = "idle"
                self.gpu_worker_status[gpu_id]["last_job_time"] = time.time()
            elif status == "error":
                logger.warning(f"GPU 작업 {task_id} 실패. 재시도합니다. (오류: {data})")
                self.gpu_worker_status[gpu_id]["status"] = "idle"
                if retries[task_id] < self.max_retries:
                    retries[task_id] += 1
                    task_map[task_id] = tasks[task_id]  # 작업을 다시 큐에 넣음
                else:
                    logger.error(
                        f"GPU 작업 {task_id} 최대 재시도 횟수 초과. 최종 실패 처리."
                    )
                    results[task_id] = data  # 예외를 결과로 저장
            elif status == "busy":
                self.gpu_worker_status[gpu_id]["status"] = "busy"

        return results

    def shutdown(self):
        """모든 워커 풀 종료"""
        self.cpu_executor.shutdown(wait=True)
        if self.config.gpu_workers > 0:
            for _ in range(self.config.gpu_workers):
                self.gpu_task_queue.put((None, None, None, None))
            for proc in self.gpu_processes:
                proc.join()
        self.thread_executor.shutdown(wait=True)
        logger.info("모든 병렬 처리 관리자 리소스가 종료되었습니다.")

    def execute_parallel(
        self, func: Callable, tasks: List[Any], prefer_gpu: bool = False
    ) -> List[Any]:
        """
        작업을 병렬로 실행 (GPU 우선 또는 CPU)
        """
        if prefer_gpu and self.config.gpu_workers > 0:
            return self._execute_on_gpu(func, tasks)
        else:
            return self._execute_on_cpu(func, tasks)

    def _execute_on_cpu(self, func: Callable, tasks: List[Any]) -> List[Any]:
        """CPU 프로세스 풀에서 작업 실행"""
        futures = [self.cpu_executor.submit(func, *task) for task in tasks]
        # 결과를 원래 순서대로 정렬
        results = [None] * len(tasks)
        future_map = {future: i for i, future in enumerate(futures)}

        for future in as_completed(future_map):
            index = future_map[future]
            try:
                results[index] = future.result()
            except Exception as e:
                logger.error(f"CPU 작업 {index} 실패: {e}")
                results[index] = e  # 예외를 결과로 저장

        return results


# (기존 get_process_pool_manager는 get_parallel_manager로 이름 변경 및 수정)
_manager_instance = None


def get_parallel_manager(
    config: Optional[ParallelManagerConfig] = None,
) -> "AdvancedParallelManager":
    global _manager_instance
    if _manager_instance is None:
        cfg = config or ParallelManagerConfig()
        _manager_instance = AdvancedParallelManager(cfg)
    return _manager_instance


@contextmanager
def process_pool_context(config: Optional[ParallelManagerConfig] = None):
    """ProcessPool 컨텍스트 매니저"""
    manager = None
    try:
        if config is None:
            config = ParallelManagerConfig()
        manager = AdvancedParallelManager(config)
        yield manager
    finally:
        if manager:
            manager.shutdown()


def parallel_map(
    func: Callable, data_list: List[Any], chunk_size: Optional[int] = None, **kwargs
) -> List[Any]:
    """
    간편한 병렬 매핑 함수

    Args:
        func: 적용할 함수
        data_list: 데이터 리스트
        chunk_size: 청크 크기 (기본값: CPU 코어 수)
        **kwargs: 함수에 전달할 추가 인자

    Returns:
        결과 리스트
    """
    if chunk_size is None:
        chunk_size = max(1, len(data_list) // psutil.cpu_count())

    # 데이터를 청크로 분할
    chunks = [
        data_list[i : i + chunk_size] for i in range(0, len(data_list), chunk_size)
    ]

    with process_pool_context() as manager:
        chunk_results = manager.execute_parallel(func, chunks, **kwargs)

    # 결과 평탄화
    results = []
    for chunk_result in chunk_results:
        if chunk_result is not None:
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
            else:
                results.append(chunk_result)

    return results


def cleanup_process_pool():
    """전역 ProcessPool 정리"""
    global _manager_instance
    if _manager_instance:
        _manager_instance.shutdown()
        _manager_instance = None
