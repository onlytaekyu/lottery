"""
동적 배치 크기 조정 컨트롤러
"""

import torch
from typing import Dict, Any, List, Callable
from torch.utils.data import DataLoader
import joblib
from concurrent.futures import ThreadPoolExecutor
import os
import gc
import time

from .error_handler_refactored import get_logger
from .unified_config import ConfigProxy
from .performance_utils import MemoryTracker

logger = get_logger(__name__)

class DynamicBatchSizeController:
    """동적 배치 크기 조정 컨트롤러"""

    def __init__(
        self,
        config: ConfigProxy,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 1024,
        memory_threshold: float = 0.9,
    ):
        self.config = config
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.memory_tracker = MemoryTracker()

    def adjust_batch_size(self, dataloader: DataLoader) -> int:
        """배치 크기 동적 조정"""
        try:
            self.memory_tracker.start()

            # 현재 메모리 사용량 확인
            memory_usage = (
                torch.cuda.memory_allocated()
                / torch.cuda.get_device_properties(0).total_memory
            )

            if memory_usage > self.memory_threshold:
                # 메모리 사용량이 높으면 배치 크기 감소
                new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
                logger.info(
                    f"메모리 사용량 높음 ({memory_usage:.2%}). 배치 크기 감소: {self.current_batch_size} -> {new_batch_size}"
                )
            else:
                # 메모리 사용량이 낮으면 배치 크기 증가
                new_batch_size = min(self.max_batch_size, self.current_batch_size * 2)
                logger.info(
                    f"메모리 사용량 낮음 ({memory_usage:.2%}). 배치 크기 증가: {self.current_batch_size} -> {new_batch_size}"
                )

            self.current_batch_size = new_batch_size
            dataloader.batch_size = new_batch_size

            return new_batch_size

        except Exception as e:
            logger.error(f"배치 크기 조정 중 오류 발생: {str(e)}")
            return self.current_batch_size

        finally:
            self.memory_tracker.stop()

    def get_batch_size_info(self) -> Dict[str, Any]:
        """배치 크기 정보 반환"""
        return {
            "current_batch_size": self.current_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "memory_threshold": self.memory_threshold,
            "memory_usage": torch.cuda.memory_allocated()
            / torch.cuda.get_device_properties(0).total_memory,
        }

class CPUBatchProcessor:
    """CPU 기반 배치 처리 컨트롤러"""

    def __init__(
        self,
        n_jobs: int = -1,
        batch_size: int = 100,
        max_batch_size: int = 1000,
        backend: str = "threading",
    ):
        """
        CPU 기반 배치 처리 컨트롤러 초기화

        Args:
            n_jobs: 병렬 작업 수 (-1은 모든 CPU 사용)
            batch_size: 초기 배치 크기
            max_batch_size: 최대 배치 크기
            backend: 병렬 백엔드 ('threading' 또는 'multiprocessing')
        """
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count() or 4
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.backend = backend
        self.memory_tracker = MemoryTracker()
        self.logger = get_logger(__name__)

    def process_batches(
        self, items: List[Any], process_fn: Callable, *args, **kwargs
    ) -> List[Any]:
        """
        항목 목록을 배치로 처리

        Args:
            items: 처리할 항목 목록
            process_fn: 각 배치를 처리할 함수
            *args, **kwargs: process_fn에 전달할 추가 인자

        Returns:
            처리된 결과 목록
        """
        self.logger.info(
            f"배치 처리 시작: {len(items)} 항목, 배치 크기: {self.batch_size}"
        )

        # 배치로 분할
        batches = self._create_batches(items)
        results = []

        start_time = time.time()

        try:
            if self.backend == "threading":
                # 스레드 기반 병렬 처리 (I/O 바운드 작업에 적합)
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = []
                    for batch in batches:
                        future = executor.submit(process_fn, batch, *args, **kwargs)
                        futures.append(future)

                    # 결과 수집
                    for future in futures:
                        batch_result = future.result()
                        if isinstance(batch_result, list):
                            results.extend(batch_result)
                        else:
                            results.append(batch_result)
            else:
                # joblib 기반 병렬 처리 (CPU 바운드 작업에 적합)
                batch_results = joblib.Parallel(n_jobs=self.n_jobs)(
                    joblib.delayed(process_fn)(batch, *args, **kwargs)
                    for batch in batches
                )

                # 결과 합치기
                for batch_result in batch_results:
                    if isinstance(batch_result, list):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)

        except Exception as e:
            self.logger.error(f"배치 처리 중 오류 발생: {str(e)}")

        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"배치 처리 완료: {len(results)} 결과, 소요 시간: {elapsed_time:.2f}초"
            )

            # 메모리 정리
            gc.collect()

        return results

    def _create_batches(self, items: List[Any]) -> List[List[Any]]:
        """
        항목 목록을 배치로 분할

        Args:
            items: 처리할 항목 목록

        Returns:
            배치 목록
        """
        batches = []
        total_items = len(items)

        # 배치 크기 조정 (항목 수에 따라)
        if total_items < self.batch_size:
            self.batch_size = max(1, total_items)
        elif total_items > self.batch_size * 10:
            # 항목이 많은 경우 배치 크기 증가
            self.batch_size = min(self.max_batch_size, self.batch_size * 2)

        # 배치 생성
        for i in range(0, total_items, self.batch_size):
            end = min(i + self.batch_size, total_items)
            batches.append(items[i:end])

        self.logger.debug(
            f"배치 생성: {len(batches)}개 배치, 배치 크기: {self.batch_size}"
        )
        return batches
