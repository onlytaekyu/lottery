"""
ProcessPool 관리자

CPU 집약적 작업을 병렬 처리하기 위한 프로세스 풀 시스템입니다.
메모리 효율성과 성능 최적화를 위해 설계되었습니다.
"""

import os
import sys
import time
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager

from .unified_logging import get_logger, log_exception_with_trace

logger = get_logger(__name__)


@dataclass
class ProcessPoolConfig:
    """ProcessPool 설정"""

    max_workers: int = min(4, psutil.cpu_count())
    chunk_size: int = 100
    timeout: float = 300.0
    memory_limit_mb: int = 1024
    enable_monitoring: bool = True
    auto_restart: bool = True
    restart_threshold: int = 100  # 작업 수행 후 재시작


class ProcessPoolManager:
    """프로세스 풀 관리자"""

    def __init__(self, config: Union[ProcessPoolConfig, Dict[str, Any]]):
        # 딕셔너리가 전달된 경우 ProcessPoolConfig로 변환
        if isinstance(config, dict):
            self.config = ProcessPoolConfig(
                max_workers=config.get("max_workers", min(4, psutil.cpu_count())),
                chunk_size=config.get("chunk_size", 100),
                timeout=config.get("timeout", 300.0),
                memory_limit_mb=config.get("memory_limit_mb", 1024),
                enable_monitoring=config.get("enable_monitoring", True),
                auto_restart=config.get("auto_restart", True),
                restart_threshold=config.get("restart_threshold", 100),
            )
        else:
            self.config = config

        self.executor = None
        self.task_count = 0
        self.performance_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_time": 0.0,
            "avg_task_time": 0.0,
        }
        self._initialize_pool()

    def _initialize_pool(self):
        """프로세스 풀 초기화"""
        try:
            self.executor = ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                mp_context=mp.get_context("spawn"),  # Windows 호환성
            )
            # 전역 인스턴스인지 확인하여 중복 로그 방지
            global _global_process_pool_manager, _initialization_logged
            if self is _global_process_pool_manager and not _initialization_logged:
                logger.info(
                    f"ProcessPool 초기화 완료: {self.config.max_workers}개 워커"
                )
                _initialization_logged = True
            else:
                logger.debug(
                    f"ProcessPool 초기화 완료: {self.config.max_workers}개 워커"
                )
        except Exception as e:
            logger.error(f"ProcessPool 초기화 실패: {e}")
            raise

    def _check_memory_usage(self) -> bool:
        """메모리 사용량 확인"""
        try:
            memory_info = psutil.virtual_memory()
            memory_usage_mb = (memory_info.total - memory_info.available) / (
                1024 * 1024
            )

            if memory_usage_mb > self.config.memory_limit_mb:
                logger.warning(
                    f"메모리 사용량 초과: {memory_usage_mb:.1f}MB > {self.config.memory_limit_mb}MB"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"메모리 확인 실패: {e}")
            return True  # 확인 실패 시 계속 진행

    def _restart_pool_if_needed(self):
        """필요시 프로세스 풀 재시작"""
        if (
            self.config.auto_restart
            and self.task_count >= self.config.restart_threshold
        ):

            logger.info(f"프로세스 풀 재시작 ({self.task_count}개 작업 완료)")
            self.shutdown()
            self._initialize_pool()
            self.task_count = 0

    def execute_parallel(
        self, func: Callable, data_chunks: List[Any], **kwargs
    ) -> List[Any]:
        """
        병렬 작업 실행

        Args:
            func: 실행할 함수
            data_chunks: 데이터 청크 리스트
            **kwargs: 함수에 전달할 추가 인자

        Returns:
            결과 리스트
        """
        if not self.executor:
            raise RuntimeError("ProcessPool이 초기화되지 않았습니다")

        if not self._check_memory_usage():
            logger.warning("메모리 부족으로 순차 처리로 전환")
            return self._execute_sequential(func, data_chunks, **kwargs)

        start_time = time.time()
        results = []

        try:
            # 작업 제출
            future_to_chunk = {
                self.executor.submit(func, chunk, **kwargs): chunk
                for chunk in data_chunks
            }

            self.performance_stats["total_tasks"] += len(data_chunks)

            # 결과 수집
            for future in as_completed(future_to_chunk, timeout=self.config.timeout):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.performance_stats["completed_tasks"] += 1
                except Exception as e:
                    logger.error(f"작업 실행 실패: {e}")
                    self.performance_stats["failed_tasks"] += 1
                    # 실패한 작업은 None으로 처리
                    results.append(None)

            # 성능 통계 업데이트
            elapsed_time = time.time() - start_time
            self.performance_stats["total_time"] += elapsed_time
            self.performance_stats["avg_task_time"] = self.performance_stats[
                "total_time"
            ] / max(1, self.performance_stats["completed_tasks"])

            self.task_count += len(data_chunks)
            self._restart_pool_if_needed()

            logger.info(
                f"병렬 처리 완료: {len(data_chunks)}개 청크, {elapsed_time:.2f}초"
            )
            return results

        except Exception as e:
            logger.error(f"병렬 처리 실패: {e}")
            # 실패 시 순차 처리로 폴백
            return self._execute_sequential(func, data_chunks, **kwargs)

    def _execute_sequential(
        self, func: Callable, data_chunks: List[Any], **kwargs
    ) -> List[Any]:
        """순차 처리 폴백"""
        logger.info("순차 처리 모드로 실행")
        results = []

        for chunk in data_chunks:
            try:
                result = func(chunk, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"순차 처리 실패: {e}")
                results.append(None)

        return results

    def execute_single(self, func: Callable, data: Any, **kwargs) -> Any:
        """단일 작업 실행"""
        if not self.executor:
            raise RuntimeError("ProcessPool이 초기화되지 않았습니다")

        try:
            future = self.executor.submit(func, data, **kwargs)
            result = future.result(timeout=self.config.timeout)
            self.task_count += 1
            self._restart_pool_if_needed()
            return result
        except Exception as e:
            logger.error(f"단일 작업 실행 실패: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.performance_stats.copy()
        stats.update(
            {
                "current_task_count": self.task_count,
                "max_workers": self.config.max_workers,
                "success_rate": (
                    stats["completed_tasks"] / max(1, stats["total_tasks"]) * 100
                ),
            }
        )
        return stats

    def shutdown(self, wait: bool = True):
        """프로세스 풀 종료"""
        if self.executor:
            try:
                self.executor.shutdown(wait=wait)
                logger.info("ProcessPool 종료 완료")
            except Exception as e:
                logger.error(f"ProcessPool 종료 실패: {e}")
            finally:
                self.executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# 전역 ProcessPool 관리자 인스턴스
_global_process_pool_manager = None
_initialization_logged = False  # 초기화 로그 중복 방지


def get_process_pool_manager(
    config: Optional[Union[ProcessPoolConfig, Dict[str, Any]]] = None,
) -> ProcessPoolManager:
    """전역 ProcessPool 관리자 반환 (싱글톤 패턴)"""
    global _global_process_pool_manager, _initialization_logged

    if _global_process_pool_manager is None:
        if config is None:
            config = ProcessPoolConfig()
        _global_process_pool_manager = ProcessPoolManager(config)
        if not _initialization_logged:
            # config가 딕셔너리인 경우 max_workers 키로 접근
            if isinstance(config, dict):
                max_workers = config.get("max_workers", 4)
            else:
                max_workers = config.max_workers
            logger.info(f"전역 ProcessPool 관리자 생성: {max_workers}개 워커")
            _initialization_logged = True
    else:
        logger.debug("기존 전역 ProcessPool 관리자 재사용")

    return _global_process_pool_manager


@contextmanager
def process_pool_context(config: Optional[ProcessPoolConfig] = None):
    """ProcessPool 컨텍스트 매니저"""
    manager = None
    try:
        if config is None:
            config = ProcessPoolConfig()
        manager = ProcessPoolManager(config)
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
    global _global_process_pool_manager, _initialization_logged
    if _global_process_pool_manager:
        _global_process_pool_manager.shutdown()
        _global_process_pool_manager = None
        _initialization_logged = False  # 로그 플래그 초기화
