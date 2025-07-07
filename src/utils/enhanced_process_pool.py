"""
고성능 멀티프로세싱 시스템 - 로또 분석 최적화

이 모듈은 로또 번호 분석을 위한 고성능 멀티프로세싱 시스템을 제공합니다.
"""

import os
import time
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from queue import Queue, Empty
import pickle
import psutil
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


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


class EnhancedProcessPool:
    """고성능 프로세스 풀 관리자"""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        chunk_size: int = 100,
        timeout: float = 300,
        enable_monitoring: bool = True,
        memory_limit_mb: int = 1024,
        cpu_affinity: bool = False,
    ):
        self.max_workers = max_workers or min(8, os.cpu_count() or 1)
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.enable_monitoring = enable_monitoring
        self.memory_limit_mb = memory_limit_mb
        self.cpu_affinity = cpu_affinity

        self.logger = get_logger(__name__)

        # 프로세스 풀
        self.process_pool = None
        self.thread_pool = None

        # 작업 큐
        self.task_queue = Queue()
        self.result_queue = Queue()

        # 모니터링
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "peak_memory_usage": 0,
            "active_workers": 0,
        }

        # 워커 상태 추적
        self.worker_stats = {}
        self.active_tasks = {}

        # 초기화
        self._initialize_pools()

        if enable_monitoring:
            self._start_monitoring()

        self.logger.info(
            f"✅ 고성능 프로세스 풀 초기화 완료 (워커: {self.max_workers}개)"
        )

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
                    self.logger.debug(
                        f"CPU 선호도 설정: {available_cpus[:self.max_workers]}"
                    )
                except Exception as e:
                    self.logger.warning(f"CPU 선호도 설정 실패: {e}")

        except Exception as e:
            self.logger.error(f"프로세스 풀 초기화 실패: {e}")
            raise

    def parallel_analyze_chunks(
        self,
        data_chunks: List[Any],
        analyzer_func: Callable,
        combine_func: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """
        데이터 청크를 병렬로 분석

        Args:
            data_chunks: 분석할 데이터 청크 리스트
            analyzer_func: 분석 함수
            combine_func: 결과 결합 함수 (None이면 리스트로 반환)
            **kwargs: 분석 함수에 전달할 추가 인자

        Returns:
            분석 결과 (결합된 결과 또는 결과 리스트)
        """
        try:
            self.logger.info(f"🚀 병렬 분석 시작: {len(data_chunks)}개 청크")
            start_time = time.time()

            # 작업 제출
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
                    self.logger.error(f"작업 {task_id} 실패: {e}")

            execution_time = time.time() - start_time
            self.stats["total_execution_time"] += execution_time

            if failed_tasks:
                self.logger.warning(f"실패한 작업: {len(failed_tasks)}개")
                for task_id, error in failed_tasks:
                    self.logger.debug(f"실패 작업 {task_id}: {error}")

            # 결과 결합
            if combine_func and results:
                combined_result = combine_func(results)
                self.logger.info(f"✅ 병렬 분석 완료: {execution_time:.2f}초 (결합됨)")
                return combined_result
            else:
                self.logger.info(
                    f"✅ 병렬 분석 완료: {execution_time:.2f}초 ({len(results)}개 결과)"
                )
                return results

        except Exception as e:
            self.logger.error(f"병렬 분석 실패: {e}")
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
            self.logger.error(f"병렬 매핑 실패: {e}")
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
                test_result = self.parallel_map(func, test_data, test_size, **kwargs)
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
            memory_used = final_memory - initial_memory

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
        """프로세스 풀 종료"""
        try:
            self.logger.info("프로세스 풀 종료 시작")

            if self.process_pool:
                self.process_pool.shutdown(wait=wait)

            if self.thread_pool:
                self.thread_pool.shutdown(wait=wait)

            self.logger.info("✅ 프로세스 풀 종료 완료")

        except Exception as e:
            self.logger.error(f"프로세스 풀 종료 실패: {e}")

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


def adaptive_lottery_processing(
    lottery_data: List[Any], processing_func: Callable, **kwargs
) -> List[Any]:
    """적응형 로또 데이터 처리 편의 함수"""
    pool = get_enhanced_process_pool()
    return pool.adaptive_parallel_processing(lottery_data, processing_func, **kwargs)
