"""
ProcessPool 기반 CPU 집약적 작업 최적화 시스템

이 모듈은 multiprocessing.Pool을 사용하여 CPU 집약적 작업을
병렬로 처리하는 시스템을 제공합니다.
"""

import os
import time
import pickle
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from pathlib import Path
import numpy as np
import logging
from functools import partial
import gc

from .error_handler_refactored import get_logger, StrictErrorHandler, validate_and_fail_fast
from .memory_manager import MemoryManager
from .performance_utils import PerformanceMonitor

logger = get_logger(__name__)


class ProcessPoolManager:
    """ProcessPool 기반 병렬 처리 관리자"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        ProcessPool 관리자 초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.strict_handler = StrictErrorHandler(self.logger)
        self.performance_monitor = PerformanceMonitor()

        # CPU 코어 수 기반 워커 수 설정
        self.max_workers = self._calculate_optimal_workers()
        self.chunk_size = self.config.get("chunk_size", 100)
        self.timeout = self.config.get("timeout", 300)

        # 메모리 관리
        self.memory_manager = MemoryManager()

        # 성능 통계
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_time": 0.0,
            "avg_speedup": 0.0,
        }

        self.logger.info(f"ProcessPool 관리자 초기화: {self.max_workers}개 워커")

    def _calculate_optimal_workers(self) -> int:
        """최적 워커 수 계산"""
        cpu_cores = cpu_count()

        # 설정에서 지정된 값이 있으면 사용
        if "max_workers" in self.config:
            configured_workers = self.config["max_workers"]
            if configured_workers == "auto":
                return max(1, cpu_cores - 1)  # 1개 코어는 시스템용으로 남김
            else:
                return min(configured_workers, cpu_cores)

        # 기본값: CPU 코어 수 - 1 (최소 1개)
        return max(1, cpu_cores - 1)

    def parallel_analyze(
        self, data_chunks: List[Any], analysis_func: Callable, **kwargs
    ) -> List[Any]:
        """
        병렬 분석 수행

        Args:
            data_chunks: 분석할 데이터 청크들
            analysis_func: 분석 함수
            **kwargs: 분석 함수에 전달할 추가 인자

        Returns:
            분석 결과 리스트
        """
        start_time = time.time()

        with self.performance_monitor.track_stage("parallel_analyze"):
            self.logger.info(
                f"병렬 분석 시작: {len(data_chunks)}개 청크, {self.max_workers}개 워커"
            )

            try:
                # 함수와 kwargs를 결합한 partial 함수 생성
                if kwargs:
                    worker_func = partial(analysis_func, **kwargs)
                else:
                    worker_func = analysis_func

                # ProcessPoolExecutor 사용
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # 작업 제출
                    futures = {
                        executor.submit(worker_func, chunk): i
                        for i, chunk in enumerate(data_chunks)
                    }

                    # 결과 수집
                    results = [None] * len(data_chunks)
                    completed = 0

                    for future in as_completed(futures, timeout=self.timeout):
                        chunk_index = futures[future]
                        try:
                            result = future.result()
                            results[chunk_index] = result
                            completed += 1

                            if completed % 10 == 0:  # 10개마다 진행상황 로그
                                self.logger.info(
                                    f"진행률: {completed}/{len(data_chunks)} ({completed/len(data_chunks)*100:.1f}%)"
                                )

                        except Exception as e:
                            self.logger.error(f"청크 {chunk_index} 처리 실패: {str(e)}")
                            self.stats["failed_tasks"] += 1
                            results[chunk_index] = None

                # 실패한 작업 검증
                failed_count = sum(1 for r in results if r is None)
                if failed_count > 0:
                    self.logger.warning(f"실패한 작업: {failed_count}개")

                    # 실패율이 20% 이상이면 치명적 오류로 처리
                    if failed_count / len(data_chunks) > 0.2:
                        self.strict_handler.handle_critical_error(
                            RuntimeError(
                                f"병렬 분석 실패율 과다: {failed_count}/{len(data_chunks)}"
                            ),
                            "병렬 분석 실패",
                        )

                # 통계 업데이트
                duration = time.time() - start_time
                self.stats["total_tasks"] += len(data_chunks)
                self.stats["completed_tasks"] += len(data_chunks) - failed_count
                self.stats["total_time"] += duration

                # 예상 속도 향상 계산 (단순 추정)
                estimated_serial_time = duration * self.max_workers
                speedup = estimated_serial_time / duration if duration > 0 else 1.0
                self.stats["avg_speedup"] = speedup

                self.logger.info(
                    f"병렬 분석 완료: {duration:.2f}초, 예상 속도향상: {speedup:.1f}x"
                )

                return results

            except Exception as e:
                self.strict_handler.handle_critical_error(e, "병렬 분석 실행 실패")
                return []

    def parallel_vectorize(
        self, patterns: List[Any], vectorize_func: Callable, **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        """
        병렬 벡터화 수행

        Args:
            patterns: 벡터화할 패턴들
            vectorize_func: 벡터화 함수
            **kwargs: 벡터화 함수에 전달할 추가 인자

        Returns:
            (벡터 배열, 특성 이름 리스트)
        """
        with self.performance_monitor.track_stage("parallel_vectorize"):
            self.logger.info(f"병렬 벡터화 시작: {len(patterns)}개 패턴")

            # 패턴을 청크로 분할
            chunks = self.chunk_and_split(patterns, self.chunk_size)

            # 병렬 벡터화 실행
            chunk_results = self.parallel_analyze(chunks, vectorize_func, **kwargs)

            # 결과 병합
            vectors, feature_names = self.merge_vectorization_results(chunk_results)

            validate_and_fail_fast(
                vectors is not None and len(vectors) > 0, "병렬 벡터화 결과가 비어 있음"
            )

            self.logger.info(f"병렬 벡터화 완료: {vectors.shape}")
            return vectors, feature_names

    def chunk_and_process(
        self, data: List[Any], chunk_size: int, process_func: Callable, **kwargs
    ) -> List[Any]:
        """
        데이터를 청크로 분할하여 병렬 처리

        Args:
            data: 처리할 데이터
            chunk_size: 청크 크기
            process_func: 처리 함수
            **kwargs: 처리 함수에 전달할 추가 인자

        Returns:
            처리 결과 리스트
        """
        with self.performance_monitor.track_stage("chunk_and_process"):
            # 데이터를 청크로 분할
            chunks = self.chunk_and_split(data, chunk_size)

            # 병렬 처리
            chunk_results = self.parallel_analyze(chunks, process_func, **kwargs)

            # 결과 병합
            merged_results = self.merge_results(chunk_results)

            return merged_results

    def chunk_and_split(self, data: List[Any], chunk_size: int) -> List[List[Any]]:
        """데이터를 청크로 분할"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            chunks.append(chunk)

        self.logger.debug(f"데이터 분할: {len(data)}개 -> {len(chunks)}개 청크")
        return chunks

    def merge_results(self, chunk_results: List[Any]) -> List[Any]:
        """청크 결과 병합"""
        merged = []
        for chunk_result in chunk_results:
            if chunk_result is not None:
                if isinstance(chunk_result, list):
                    merged.extend(chunk_result)
                else:
                    merged.append(chunk_result)

        self.logger.debug(
            f"결과 병합: {len(chunk_results)}개 청크 -> {len(merged)}개 결과"
        )
        return merged

    def merge_vectorization_results(
        self, chunk_results: List[Tuple]
    ) -> Tuple[np.ndarray, List[str]]:
        """벡터화 결과 병합"""
        vectors_list = []
        feature_names = None

        for chunk_result in chunk_results:
            if chunk_result is not None and len(chunk_result) == 2:
                chunk_vectors, chunk_names = chunk_result

                if chunk_vectors is not None:
                    vectors_list.append(chunk_vectors)

                # 첫 번째 유효한 특성 이름을 사용
                if feature_names is None and chunk_names is not None:
                    feature_names = chunk_names

        # 벡터 결합
        if vectors_list:
            combined_vectors = np.vstack(vectors_list)
        else:
            combined_vectors = np.array([])

        if feature_names is None:
            feature_names = []

        self.logger.debug(
            f"벡터화 결과 병합: {combined_vectors.shape}, {len(feature_names)}개 특성"
        )
        return combined_vectors, feature_names

    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """리소스 할당 최적화"""
        # CPU 사용률 확인
        cpu_percent = self.performance_monitor.get_cpu_usage()
        memory_percent = self.performance_monitor.get_memory_usage()

        # 동적 워커 수 조정
        if cpu_percent < 50:  # CPU 사용률이 낮으면 워커 수 증가
            new_workers = min(self.max_workers + 1, cpu_count())
        elif cpu_percent > 90:  # CPU 사용률이 높으면 워커 수 감소
            new_workers = max(1, self.max_workers - 1)
        else:
            new_workers = self.max_workers

        # 메모리 사용률이 높으면 청크 크기 감소
        if memory_percent > 80:
            new_chunk_size = max(50, self.chunk_size // 2)
        elif memory_percent < 50:
            new_chunk_size = min(200, self.chunk_size * 2)
        else:
            new_chunk_size = self.chunk_size

        # 설정 업데이트
        old_workers = self.max_workers
        old_chunk_size = self.chunk_size

        self.max_workers = new_workers
        self.chunk_size = new_chunk_size

        optimization_result = {
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "workers_changed": old_workers != new_workers,
            "chunk_size_changed": old_chunk_size != new_chunk_size,
            "old_workers": old_workers,
            "new_workers": new_workers,
            "old_chunk_size": old_chunk_size,
            "new_chunk_size": new_chunk_size,
        }

        if (
            optimization_result["workers_changed"]
            or optimization_result["chunk_size_changed"]
        ):
            self.logger.info(
                f"리소스 할당 최적화: 워커 {old_workers}->{new_workers}, 청크크기 {old_chunk_size}->{new_chunk_size}"
            )

        return optimization_result

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            **self.stats,
            "current_workers": self.max_workers,
            "current_chunk_size": self.chunk_size,
            "cpu_usage": self.performance_monitor.get_cpu_usage(),
            "memory_usage": self.performance_monitor.get_memory_usage(),
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            # 메모리 정리
            gc.collect()

            # 성능 모니터 정리
            if hasattr(self.performance_monitor, "cleanup"):
                self.performance_monitor.cleanup()

            self.logger.info("ProcessPool 관리자 정리 완료")

        except Exception as e:
            self.logger.warning(f"ProcessPool 정리 중 오류: {e}")

    def __del__(self):
        """소멸자"""
        self.cleanup()


# 전역 ProcessPool 관리자 인스턴스
_global_process_pool_manager = None


def get_process_pool_manager(
    config: Optional[Dict[str, Any]] = None,
) -> ProcessPoolManager:
    """전역 ProcessPool 관리자 반환"""
    global _global_process_pool_manager

    if _global_process_pool_manager is None:
        _global_process_pool_manager = ProcessPoolManager(config)

    return _global_process_pool_manager


# 편의 함수들
def parallel_map(
    func: Callable, data: List[Any], chunk_size: int = 100, **kwargs
) -> List[Any]:
    """병렬 맵 함수"""
    manager = get_process_pool_manager()
    chunks = manager.chunk_and_split(data, chunk_size)
    return manager.parallel_analyze(chunks, func, **kwargs)


def parallel_reduce(
    func: Callable,
    data: List[Any],
    reduce_func: Callable,
    chunk_size: int = 100,
    **kwargs,
) -> Any:
    """병렬 맵-리듀스 함수"""
    # 병렬 맵 단계
    mapped_results = parallel_map(func, data, chunk_size, **kwargs)

    # 리듀스 단계
    return reduce_func(mapped_results)


# 데코레이터
def parallelize(chunk_size: int = 100, max_workers: Optional[int] = None):
    """함수를 병렬화하는 데코레이터"""

    def decorator(func: Callable):
        def wrapper(data: List[Any], **kwargs):
            config = {"chunk_size": chunk_size}
            if max_workers is not None:
                config["max_workers"] = max_workers

            manager = ProcessPoolManager(config)
            chunks = manager.chunk_and_split(data, chunk_size)
            return manager.parallel_analyze(chunks, func, **kwargs)

        return wrapper

    return decorator
