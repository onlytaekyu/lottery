"""
고성능 멀티프로세싱 시스템 - 로또 분석 최적화
⚠️ process_pool_manager.py의 GPU 워커 관리 기능이 통합되었습니다.

이 모듈은 로또 번호 분석을 위한 고성능 멀티프로세싱 시스템을 제공합니다.
GPU 메모리 기반 동적 배치 크기 조정 및 지능적 GPU 워커 관리 기능이 포함되어 있습니다.
"""

import os
import time
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from queue import Queue, Empty
import psutil

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.unified_logging import get_logger, log_exception

logger = get_logger(__name__)


def gpu_worker_process(task_queue: mp.Queue, result_queue: mp.Queue, gpu_id: int):
    """GPU 워커 프로세스 함수 (process_pool_manager.py에서 통합)"""
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
class ProcessingTask:
    """처리 작업 정의"""

    task_id: str
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ProcessingResult:
    """처리 결과"""

    task_id: str
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None


@dataclass
class ParallelManagerConfig:
    """고급 병렬 처리 관리자 설정 (process_pool_manager.py에서 통합)"""

    cpu_workers: int = min(4, psutil.cpu_count(logical=False) or 1)
    gpu_workers: int = torch.cuda.device_count() if TORCH_AVAILABLE else 0
    chunk_size: int = 100
    timeout: float = 300.0
    memory_limit_mb: int = 1024
    enable_monitoring: bool = True
    auto_restart: bool = True
    restart_threshold: int = 100  # 작업 수행 후 재시작


class EnhancedProcessPool:
    """
    고성능 프로세스 풀 관리자 (통합 버전)
    ⚠️ process_pool_manager.py의 GPU 워커 관리 기능이 통합되었습니다.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        chunk_size: int = 100,
        timeout: float = 300,
        enable_monitoring: bool = True,
        memory_limit_mb: int = 1024,
        cpu_affinity: bool = False,
        config: Optional[ParallelManagerConfig] = None,
        max_retries: int = 3,
    ):
        # 기존 설정
        self.max_workers = max_workers or min(8, os.cpu_count() or 1)
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.enable_monitoring = enable_monitoring
        self.memory_limit_mb = memory_limit_mb
        self.cpu_affinity = cpu_affinity
        self.max_retries = max_retries

        # 통합된 설정 (process_pool_manager.py에서)
        self.config = config or ParallelManagerConfig()

        self.logger = get_logger(__name__)

        # 기존 프로세스 풀
        self.process_pool = None
        self.thread_pool = None

        # 기존 작업 큐
        self.task_queue = Queue()
        self.result_queue = Queue()

        # 통합된 GPU 워커 관리 (process_pool_manager.py에서)
        self.gpu_task_queue = mp.Queue()
        self.gpu_result_queue = mp.Queue()
        self.gpu_processes = []
        self.gpu_worker_status = {}  # {gpu_id: {'status': 'idle'/'busy', 'last_job_time': float}}

        # 기존 모니터링
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "peak_memory_usage": 0,
            "active_workers": 0,
            # 통합된 GPU 통계
            "gpu_tasks_submitted": 0,
            "gpu_tasks_completed": 0,
            "gpu_tasks_failed": 0,
            "gpu_workers_active": 0,
        }

        # 기존 워커 상태 추적
        self.worker_stats = {}
        self.active_tasks = {}

        # 초기화
        self._initialize_pools()
        self._initialize_gpu_workers()  # GPU 워커 초기화 추가

        if enable_monitoring:
            self._start_monitoring()

        logger.info(f"✅ 통합 고성능 프로세스 풀 초기화 완료 (CPU 워커: {self.max_workers}개, GPU 워커: {self.config.gpu_workers}개)")

    def _initialize_pools(self):
        """프로세스 및 스레드 풀 초기화"""
        try:
            # 프로세스 풀 초기화
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context("spawn"),  # Windows 호환성
            )

            # 스레드 풀 초기화 (I/O 작업용)
            self.thread_pool = ThreadPoolExecutor(max_workers=min(4, self.max_workers))

            # CPU 선호도 설정 (Linux/Unix)
            if self.cpu_affinity and hasattr(os, "sched_setaffinity"):
                try:
                    available_cpus = list(range(os.cpu_count()))
                    os.sched_setaffinity(0, available_cpus[: self.max_workers])
                    logger.debug(f"CPU 선호도 설정: {available_cpus[:self.max_workers]}")
                except Exception as e:
                    logger.warning(f"CPU 선호도 설정 실패: {e}")

        except Exception as e:
            logger.error(f"프로세스 풀 초기화 실패: {e}")
            raise

    def _initialize_gpu_workers(self):
        """GPU 워커 초기화 (process_pool_manager.py에서 통합)"""
        if not TORCH_AVAILABLE or self.config.gpu_workers <= 0:
            logger.info("GPU 워커 사용 안함 (CUDA 미사용 또는 GPU 없음)")
            return

        try:
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

            self.stats["gpu_workers_active"] = self.config.gpu_workers
            logger.info(f"✅ GPU 워커 {self.config.gpu_workers}개 초기화 완료")

        except Exception as e:
            logger.error(f"GPU 워커 초기화 실패: {e}")

    def _get_idle_gpu_worker(self) -> Optional[int]:
        """가장 오랫동안 유휴 상태였던 GPU 워커 ID를 반환 (process_pool_manager.py에서 통합)"""
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
        """GPU 워커 풀에서 작업 실행 (process_pool_manager.py에서 통합)"""
        num_tasks = len(tasks)
        results = [None] * num_tasks
        retries = [0] * num_tasks

        task_map = {i: task for i, task in enumerate(tasks)}
        self.stats["gpu_tasks_submitted"] += num_tasks

        while task_map:
            # 유휴 GPU 워커에게 작업 할당
            idle_worker_id = self._get_idle_gpu_worker()
            if idle_worker_id is not None and task_map:
                task_id, task_args = task_map.popitem()
                self.gpu_worker_status[idle_worker_id]["status"] = "busy"
                self.gpu_task_queue.put((task_id, func, task_args, {}))

            # 결과 처리
            try:
                status, gpu_id, task_id, data = self.gpu_result_queue.get(timeout=1)

                if status == "done":
                    results[task_id] = data
                    self.gpu_worker_status[gpu_id]["status"] = "idle"
                    self.gpu_worker_status[gpu_id]["last_job_time"] = time.time()
                    self.stats["gpu_tasks_completed"] += 1
                elif status == "error":
                    logger.warning(f"GPU 작업 {task_id} 실패. 재시도합니다. (오류: {data})")
                    self.gpu_worker_status[gpu_id]["status"] = "idle"
                    if retries[task_id] < self.max_retries:
                        retries[task_id] += 1
                        task_map[task_id] = tasks[task_id]  # 작업을 다시 큐에 넣음
                    else:
                        logger.error(f"GPU 작업 {task_id} 최대 재시도 횟수 초과. 최종 실패 처리.")
                        results[task_id] = data  # 예외를 결과로 저장
                        self.stats["gpu_tasks_failed"] += 1
                elif status == "busy":
                    self.gpu_worker_status[gpu_id]["status"] = "busy"

            except Empty:
                # 타임아웃 발생, 계속 진행
                continue

        return results

    def _execute_on_cpu(self, func: Callable, tasks: List[Any]) -> List[Any]:
        """CPU 프로세스 풀에서 작업 실행 (process_pool_manager.py에서 통합)"""
        if not self.process_pool:
            raise RuntimeError("프로세스 풀이 초기화되지 않았습니다")
            
        futures = [self.process_pool.submit(func, *task) for task in tasks]
        results = [None] * len(tasks)
        future_map = {future: i for i, future in enumerate(futures)}

        for future in as_completed(future_map):
            index = future_map[future]
            try:
                results[index] = future.result()
                self.stats["tasks_completed"] += 1
            except Exception as e:
                logger.error(f"CPU 작업 {index} 실패: {e}")
                results[index] = e  # 예외를 결과로 저장
                self.stats["tasks_failed"] += 1

        return results

    def execute_parallel(
        self, func: Callable, tasks: List[Any], prefer_gpu: bool = False
    ) -> List[Any]:
        """
        작업을 병렬로 실행 (GPU 우선 또는 CPU) (process_pool_manager.py에서 통합)
        """
        if prefer_gpu and self.config.gpu_workers > 0 and TORCH_AVAILABLE:
            logger.info(f"🚀 GPU 병렬 실행: {len(tasks)}개 작업")
            return self._execute_on_gpu(func, tasks)
        else:
            logger.info(f"🚀 CPU 병렬 실행: {len(tasks)}개 작업")
            self.stats["tasks_submitted"] += len(tasks)
            return self._execute_on_cpu(func, tasks)

    def parallel_analyze_chunks(
        self,
        data_chunks: List[Any],
        analyzer_func: Callable,
        combine_func: Optional[Callable] = None,
        prefer_gpu: bool = False,
        **kwargs,
    ) -> Any:
        """
        데이터 청크를 병렬로 분석 (통합 버전)

        Args:
            data_chunks: 분석할 데이터 청크 리스트
            analyzer_func: 분석 함수
            combine_func: 결과 결합 함수 (None이면 리스트로 반환)
            prefer_gpu: GPU 사용 선호 여부
            **kwargs: 분석 함수에 전달할 추가 인자

        Returns:
            분석 결과 (결합된 결과 또는 결과 리스트)
        """
        try:
            logger.info(f"🚀 통합 병렬 분석 시작: {len(data_chunks)}개 청크 (GPU 선호: {prefer_gpu})")
            start_time = time.time()

            if prefer_gpu and self.config.gpu_workers > 0 and TORCH_AVAILABLE:
                # GPU 병렬 처리
                results = self._execute_on_gpu(analyzer_func, data_chunks)
            else:
                # 기존 CPU 병렬 처리
                if not self.process_pool:
                    # 프로세스 풀이 없으면 순차 실행
                    results = []
                    for chunk in data_chunks:
                        try:
                            result = analyzer_func(chunk, **kwargs)
                            results.append(result)
                        except Exception as e:
                            logger.error(f"순차 분석 중 오류: {e}")
                            results.append(None)
                else:
                    # 프로세스 풀 사용
                    futures = []
                    for i, chunk in enumerate(data_chunks):
                        task_id = f"chunk_analysis_{i}_{time.time()}"

                        future = self.process_pool.submit(
                            self._execute_analysis_task, task_id, analyzer_func, chunk, kwargs
                        )

                        futures.append((task_id, future))
                        self.stats["tasks_submitted"] += 1

                    # 결과 수집
                    results = []
                    failed_tasks = []

                    for task_id, future in futures:
                        try:
                            result = future.result(timeout=self.timeout)
                            if result.success:
                                results.append(result.result)
                                self.stats["tasks_completed"] += 1
                            else:
                                failed_tasks.append((task_id, result.error))
                                self.stats["tasks_failed"] += 1

                        except Exception as e:
                            failed_tasks.append((task_id, str(e)))
                            self.stats["tasks_failed"] += 1
                            logger.error(f"작업 {task_id} 실패: {e}")

                    if failed_tasks:
                        logger.warning(f"실패한 작업: {len(failed_tasks)}개")
                        for task_id, error in failed_tasks:
                            logger.debug(f"실패 작업 {task_id}: {error}")

            execution_time = time.time() - start_time
            self.stats["total_execution_time"] += execution_time

            # 결과 결합
            if combine_func and callable(combine_func):
                logger.debug("결과 결합 중...")
                final_result = combine_func(results)
                logger.info(f"✅ 통합 병렬 분석 완료 (실행 시간: {execution_time:.2f}초)")
                return final_result
            else:
                logger.info(f"✅ 통합 병렬 분석 완료 (실행 시간: {execution_time:.2f}초, 결과: {len(results)}개)")
                return results

        except Exception as e:
            logger.error(f"통합 병렬 분석 실패: {e}")
            raise

    def parallel_map(
        self,
        func: Callable,
        iterable: List[Any],
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> List[Any]:
        """
        함수를 iterable에 병렬로 매핑

        Args:
            func: 적용할 함수
            iterable: 데이터 리스트
            chunksize: 청크 크기
            **kwargs: 함수에 전달할 추가 인자

        Returns:
            결과 리스트
        """
        try:
            chunk_size = chunksize or max(1, len(iterable) // (self.max_workers * 2))

            # 데이터 청킹
            chunks = [
                iterable[i : i + chunk_size]
                for i in range(0, len(iterable), chunk_size)
            ]

            # 청크 처리 함수
            def process_chunk(chunk):
                return [func(item, **kwargs) for item in chunk]

            # 결과 결합 함수
            def combine_results(chunk_results):
                combined = []
                for chunk_result in chunk_results:
                    combined.extend(chunk_result)
                return combined

            return self.parallel_analyze_chunks(chunks, process_chunk, combine_results)

        except Exception as e:
            logger.error(f"병렬 매핑 실패: {e}")
            raise

    def adaptive_parallel_processing(
        self,
        data: List[Any],
        func: Callable,
        min_chunk_size: int = 10,
        max_chunk_size: int = 1000,
        target_execution_time: float = 1.0,
        **kwargs,
    ) -> List[Any]:
        """
        적응형 병렬 처리 - 성능에 따라 청크 크기 자동 조정

        Args:
            data: 처리할 데이터
            func: 처리 함수
            min_chunk_size: 최소 청크 크기
            max_chunk_size: 최대 청크 크기
            target_execution_time: 목표 실행 시간 (초)
            **kwargs: 함수 인자

        Returns:
            처리 결과 리스트
        """
        try:
            if len(data) <= min_chunk_size:
                # 데이터가 작으면 단일 스레드로 처리
                return [func(item, **kwargs) for item in data]

            # 초기 청크 크기 결정
            initial_chunk_size = min(
                max_chunk_size, max(min_chunk_size, len(data) // self.max_workers)
            )

            # 성능 테스트를 위한 작은 샘플 처리
            test_size = min(initial_chunk_size, len(data) // 10)
            if test_size > 0:
                test_data = data[:test_size]

                start_time = time.time()
                self.parallel_map(func, test_data, test_size, **kwargs)
                test_time = time.time() - start_time

                if test_time > 0:
                    # 목표 시간에 맞춰 청크 크기 조정
                    optimal_chunk_size = int(
                        test_size * target_execution_time / test_time
                    )
                    optimal_chunk_size = max(
                        min_chunk_size, min(max_chunk_size, optimal_chunk_size)
                    )
                else:
                    optimal_chunk_size = initial_chunk_size
            else:
                optimal_chunk_size = initial_chunk_size

            self.logger.debug(f"적응형 청크 크기: {optimal_chunk_size}")

            # 전체 데이터 처리
            return self.parallel_map(func, data, optimal_chunk_size, **kwargs)

        except Exception as e:
            self.logger.error(f"적응형 병렬 처리 실패: {e}")
            # 폴백: 단일 스레드 처리
            return [func(item, **kwargs) for item in data]

    def pipeline_processing(
        self,
        data: List[Any],
        pipeline_stages: List[Callable],
        stage_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Any]:
        """
        파이프라인 병렬 처리

        Args:
            data: 입력 데이터
            pipeline_stages: 파이프라인 단계 함수들
            stage_configs: 각 단계별 설정

        Returns:
            최종 처리 결과
        """
        try:
            self.logger.info(f"🔄 파이프라인 처리 시작: {len(pipeline_stages)}단계")

            current_data = data
            stage_configs = stage_configs or [{}] * len(pipeline_stages)

            for i, (stage_func, config) in enumerate(
                zip(pipeline_stages, stage_configs)
            ):
                self.logger.debug(
                    f"파이프라인 단계 {i+1}/{len(pipeline_stages)} 처리 중"
                )

                start_time = time.time()
                current_data = self.parallel_map(stage_func, current_data, **config)
                stage_time = time.time() - start_time

                self.logger.debug(f"단계 {i+1} 완료: {stage_time:.2f}초")

            self.logger.info("✅ 파이프라인 처리 완료")
            return current_data

        except Exception as e:
            self.logger.error(f"파이프라인 처리 실패: {e}")
            raise

    def _execute_analysis_task(
        self, task_id: str, func: Callable, data: Any, kwargs: Dict[str, Any]
    ) -> ProcessingResult:
        """분석 작업 실행"""
        start_time = time.time()
        worker_id = f"worker_{os.getpid()}"

        try:
            # 메모리 사용량 모니터링
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 작업 실행
            result = func(data, **kwargs)

            # 최종 메모리 사용량
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_memory - initial_memory

            execution_time = time.time() - start_time

            # 메모리 제한 체크
            if final_memory > self.memory_limit_mb:
                self.logger.warning(
                    f"메모리 제한 초과: {final_memory:.1f}MB > {self.memory_limit_mb}MB"
                )

            return ProcessingResult(
                task_id=task_id,
                result=result,
                success=True,
                execution_time=execution_time,
                worker_id=worker_id,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"작업 실행 실패: {str(e)}"

            return ProcessingResult(
                task_id=task_id,
                result=None,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                worker_id=worker_id,
            )

    def _start_monitoring(self):
        """모니터링 스레드 시작"""

        def monitor_worker():
            while True:
                try:
                    # 시스템 리소스 모니터링
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent

                    # 통계 업데이트
                    self.stats["peak_memory_usage"] = max(
                        self.stats["peak_memory_usage"], memory_percent
                    )

                    # 로그 출력 (5분마다)
                    if int(time.time()) % 300 == 0:
                        self.logger.debug(
                            f"시스템 상태 - CPU: {cpu_percent:.1f}%, "
                            f"메모리: {memory_percent:.1f}%, "
                            f"완료된 작업: {self.stats['tasks_completed']}"
                        )

                    time.sleep(10)  # 10초마다 모니터링

                except Exception as e:
                    self.logger.error(f"모니터링 오류: {e}")
                    time.sleep(30)

        monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        monitor_thread.start()
        self.logger.debug("모니터링 스레드 시작")

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        try:
            # 시스템 정보 추가
            cpu_count = os.cpu_count()
            memory_info = psutil.virtual_memory()

            return {
                **self.stats,
                "max_workers": self.max_workers,
                "chunk_size": self.chunk_size,
                "timeout": self.timeout,
                "system_info": {
                    "cpu_count": cpu_count,
                    "memory_total_gb": memory_info.total / (1024**3),
                    "memory_available_gb": memory_info.available / (1024**3),
                    "memory_percent": memory_info.percent,
                },
                "efficiency": {
                    "success_rate": (
                        self.stats["tasks_completed"]
                        / max(1, self.stats["tasks_submitted"])
                    )
                    * 100,
                    "avg_execution_time": (
                        self.stats["total_execution_time"]
                        / max(1, self.stats["tasks_completed"])
                    ),
                    "throughput": (
                        self.stats["tasks_completed"]
                        / max(1, self.stats["total_execution_time"])
                    ),
                },
            }
        except Exception as e:
            self.logger.error(f"통계 조회 실패: {e}")
            return self.stats

    def shutdown(self, wait: bool = True):
        """
        모든 리소스 종료 (통합 버전)
        """
        try:
            # 기존 프로세스 풀 종료
            if self.process_pool:
                self.process_pool.shutdown(wait=wait)
                self.process_pool = None

            # 스레드 풀 종료
            if self.thread_pool:
                self.thread_pool.shutdown(wait=wait)
                self.thread_pool = None

            # GPU 워커 프로세스 종료 (process_pool_manager.py에서 통합)
            if self.config.gpu_workers > 0 and self.gpu_processes:
                logger.info(f"GPU 워커 {len(self.gpu_processes)}개 종료 중...")
                
                # 종료 신호 전송
                for _ in range(self.config.gpu_workers):
                    self.gpu_task_queue.put((None, None, None, None))
                
                # 프로세스 종료 대기
                for proc in self.gpu_processes:
                    if proc.is_alive():
                        proc.join(timeout=5)
                        if proc.is_alive():
                            logger.warning(f"GPU 워커 프로세스 {proc.pid} 강제 종료")
                            proc.terminate()
                
                self.gpu_processes.clear()
                self.gpu_worker_status.clear()
                
                logger.info("✅ GPU 워커 종료 완료")

            # 모니터링 중지
            if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2)

            logger.info("✅ 통합 프로세스 풀 종료 완료")

        except Exception as e:
            logger.error(f"프로세스 풀 종료 중 오류: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# 전역 프로세스 풀 인스턴스
_process_pool = None
_pool_lock = threading.Lock()


def get_enhanced_process_pool(
    config: Optional[Dict[str, Any]] = None,
) -> EnhancedProcessPool:
    """고성능 프로세스 풀 반환 (싱글톤)"""
    global _process_pool

    with _pool_lock:
        if _process_pool is None:
            pool_config = config or {}
            _process_pool = EnhancedProcessPool(**pool_config)
        return _process_pool


def cleanup_process_pool():
    """프로세스 풀 정리"""
    global _process_pool

    with _pool_lock:
        if _process_pool is not None:
            _process_pool.shutdown()
            _process_pool = None


# 편의 함수들
def parallel_lottery_analysis(
    data_chunks: List[Any], analyzer_func: Callable, **kwargs
) -> List[Any]:
    """로또 분석을 위한 병렬 처리 편의 함수"""
    pool = get_enhanced_process_pool()
    return pool.parallel_analyze_chunks(data_chunks, analyzer_func, **kwargs)


class DynamicBatchSizeController:
    """GPU 메모리 기반 동적 배치 크기 조정기"""

    def __init__(
        self,
        initial_batch_size: int = 2000,
        min_batch_size: int = 500,
        max_batch_size: int = 5000,
        memory_threshold: float = 0.8,
        adjustment_factor: float = 0.7,
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adjustment_factor = adjustment_factor

        # GPU 가용성 확인
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()

        # 성능 통계
        self.oom_count = 0
        self.successful_batches = 0
        self.total_processing_time = 0.0
        self.last_adjustment_time = time.time()

        # 스레드 안전성
        self.lock = threading.Lock()

        if self.gpu_available:
            logger.info(
                f"✅ GPU 동적 배치 크기 조정 활성화 - 초기 크기: {initial_batch_size}"
            )
        else:
            logger.info(f"⚠️ CPU 기반 배치 처리 - 고정 크기: {initial_batch_size}")

    def get_current_batch_size(self) -> int:
        """현재 최적 배치 크기 반환"""
        with self.lock:
            if self.gpu_available:
                try:
                    memory_usage = self._get_gpu_memory_usage()
                    if memory_usage > self.memory_threshold:
                        self._reduce_batch_size()
                    elif (
                        memory_usage < 0.5
                        and self.current_batch_size < self.max_batch_size
                    ):
                        self._increase_batch_size()
                except Exception as e:
                    logger.warning(f"GPU 메모리 확인 실패: {e}")

            return self.current_batch_size

    def handle_oom(self) -> int:
        """OutOfMemory 발생시 배치 크기 감소"""
        with self.lock:
            self.oom_count += 1
            old_size = self.current_batch_size

            self.current_batch_size = max(
                self.min_batch_size, int(self.current_batch_size * 0.5)
            )

            logger.warning(
                f"🚨 OOM 발생 - 배치 크기 조정: {old_size} → {self.current_batch_size}"
            )
            return self.current_batch_size

    def report_success(self, processing_time: float):
        """성공적인 배치 처리 보고"""
        with self.lock:
            self.successful_batches += 1
            self.total_processing_time += processing_time

            if (
                self.successful_batches % 10 == 0
                and time.time() - self.last_adjustment_time > 60
            ):
                self._consider_batch_increase()

    def _get_gpu_memory_usage(self) -> float:
        """GPU 메모리 사용률 반환"""
        if not self.gpu_available:
            return 0.0

        try:
            if TORCH_AVAILABLE:
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                return max(allocated, cached) / total
            return 0.0
        except Exception as e:
            logger.debug(f"GPU 메모리 사용률 확인 실패: {e}")
            return 0.0

    def _reduce_batch_size(self):
        """배치 크기 감소"""
        old_size = self.current_batch_size
        self.current_batch_size = max(
            self.min_batch_size, int(self.current_batch_size * self.adjustment_factor)
        )

        if old_size != self.current_batch_size:
            logger.info(f"📉 배치 크기 감소: {old_size} → {self.current_batch_size}")
            self.last_adjustment_time = time.time()

    def _increase_batch_size(self):
        """배치 크기 증가"""
        old_size = self.current_batch_size
        self.current_batch_size = min(
            self.max_batch_size, int(self.current_batch_size * 1.2)
        )

        if old_size != self.current_batch_size:
            logger.info(f"📈 배치 크기 증가: {old_size} → {self.current_batch_size}")
            self.last_adjustment_time = time.time()

    def _consider_batch_increase(self):
        """배치 크기 증가 고려"""
        if (
            self.oom_count == 0
            and self.current_batch_size < self.max_batch_size
            and self._get_gpu_memory_usage() < 0.6
        ):
            self._increase_batch_size()

    def get_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        with self.lock:
            avg_time = self.total_processing_time / max(1, self.successful_batches)

            return {
                "current_batch_size": self.current_batch_size,
                "oom_count": self.oom_count,
                "successful_batches": self.successful_batches,
                "average_processing_time": avg_time,
                "gpu_memory_usage": self._get_gpu_memory_usage(),
                "gpu_available": self.gpu_available,
            }


# 싱글톤 인스턴스
_batch_controller_instance = None


def get_dynamic_batch_controller(**kwargs) -> DynamicBatchSizeController:
    """동적 배치 크기 컨트롤러 싱글톤 인스턴스 반환"""
    global _batch_controller_instance

    if _batch_controller_instance is None:
        _batch_controller_instance = DynamicBatchSizeController(**kwargs)

    return _batch_controller_instance


def adaptive_lottery_processing(
    lottery_data: List[Any], processing_func: Callable, **kwargs
) -> List[Any]:
    """적응형 로또 데이터 처리 편의 함수 (동적 배치 크기 지원)"""
    pool = get_enhanced_process_pool()

    # 동적 배치 크기 컨트롤러 사용
    batch_controller = get_dynamic_batch_controller()
    optimal_batch_size = batch_controller.get_current_batch_size()

    # 배치 크기를 동적으로 조정하여 처리
    try:
        start_time = time.time()
        result = pool.adaptive_parallel_processing(
            lottery_data, processing_func, max_chunk_size=optimal_batch_size, **kwargs
        )
        processing_time = time.time() - start_time
        batch_controller.report_success(processing_time)
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            batch_controller.handle_oom()
            # 줄어든 배치 크기로 재시도
            new_batch_size = batch_controller.get_current_batch_size()
            return pool.adaptive_parallel_processing(
                lottery_data, processing_func, max_chunk_size=new_batch_size, **kwargs
            )
        raise


# 하위 호환성을 위한 별칭
CPUBatchProcessor = EnhancedProcessPool
