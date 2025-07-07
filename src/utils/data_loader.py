"""
GPU 네이티브 데이터 로더 (v4 - 제로카피 최적화)

Pinned Memory와 제로카피(Zero-copy) 변환을 통해 CPU를 거치지 않고
데이터를 GPU로 직접 로딩하여, 로딩 속도를 극대화합니다.
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
import json
from collections import Counter, deque
import mmap
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import weakref
import aiofiles
from io import BytesIO

from .unified_logging import get_logger
from .async_io import get_async_io_manager
from .memory_manager import get_memory_manager
from .error_handler import get_error_handler
from .cache_manager import get_cache_manager
from .unified_config import get_config
from .factory import get_singleton_instance

logger = get_logger(__name__)
error_handler = get_error_handler()
GPU_AVAILABLE = torch.cuda.is_available()


class GPUNativeDataLoader:
    """제로카피를 활용하는 GPU 네이티브 데이터 로더"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_manager = get_memory_manager()  # 통합 메모리 관리자 사용
        self.cache_manager = get_cache_manager()  # 통합 캐시 관리자 사용

        # 설정에서 값 로드
        config = get_config("main").get_nested("utils.data_loader", {})
        self.max_workers = config.get("max_workers", 4)
        self.prefetch_factor = config.get("prefetch_factor", 2)
        self.dynamic_batch_size_mapping = config.get(
            "dynamic_batch_size_mapping",
            {
                "0.8": 512,
                "0.6": 256,
                "0.4": 128,
                "0.2": 64,
                "default": 32,
            },
        )

        # 누락된 속성 초기화 (사용자 요청)
        self.stats = {
            "cache_hits": 0,
            "loads": 0,
            "errors": 0,
            "gpu_direct_loads": 0,
            "avg_load_time_ms": 0,
        }
        self.io_manager = get_async_io_manager()
        self.pin_memory = GPU_AVAILABLE

        if self.device.type == "cuda":
            logger.info(
                "✅ GPU 네이티브 데이터 로더 초기화 (Pinned Memory 및 제로카피 활성화)"
            )
        else:
            logger.info("✅ GPU 네이티브 데이터 로더 초기화 (CPU 모드)")

    @error_handler.auto_recoverable(max_retries=2, delay=0.2)
    def gpu_native_load_csv(
        self, file_path: Union[str, Path], dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """
        CSV 파일을 Pinned Memory와 제로카피를 통해 GPU 텐서로 직접 로딩합니다.
        """
        if self.device.type != "cuda":
            logger.warning("GPU가 없어 일반 CPU 모드로 CSV를 로드합니다.")
            numpy_array = pd.read_csv(file_path).to_numpy()
            return torch.from_numpy(numpy_array).to(dtype)

        try:
            # 1. Pandas로 데이터를 읽고 NumPy 배열로 변환
            numpy_array = pd.read_csv(file_path).to_numpy(dtype=np.float32)

            # 2. Pinned Memory에 텐서 할당
            #    .pin_memory()는 새로운 Pinned 텐서를 생성
            pinned_tensor = torch.from_numpy(numpy_array).pin_memory()

            # 3. 비동기적으로 GPU로 전송
            #    Pinned Memory -> GPU 전송은 non_blocking=True일 때 비동기로 동작
            gpu_tensor = pinned_tensor.to(self.device, non_blocking=True, dtype=dtype)

            logger.debug(
                f"'{file_path}'를 GPU 네이티브 방식으로 성공적으로 로드했습니다."
            )
            return gpu_tensor

        except FileNotFoundError:
            logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return None
        except Exception as e:
            logger.error(f"'{file_path}' 로딩 중 오류 발생: {e}")
            return None

    @error_handler.auto_recoverable(max_retries=3, delay=0.5)
    async def gpu_native_load(
        self, data_path: Union[str, Path], dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """
        GPU 텐서를 직접 생성하는 네이티브 로더 (메모리 매핑 및 비동기 I/O 활용)
        """
        path_str = str(data_path)
        cache_key = f"{path_str}_{dtype}"

        # 통합 캐시 관리자에서 조회
        cached_tensor = self.cache_manager.get(cache_key)
        if cached_tensor is not None:
            self.stats["cache_hits"] += 1
            # 캐시된 텐서가 올바른 장치에 있는지 확인
            if cached_tensor.device.type != self.device.type:
                return cached_tensor.to(self.device, non_blocking=True)
            return cached_tensor

        start_time = time.perf_counter()
        tensor = None

        try:
            if GPU_AVAILABLE and str(data_path).endswith(".npy"):
                # 제로 카피 비동기 읽기 사용
                tensor = await self.io_manager.zero_copy_read(
                    data_path, target_device="cuda"
                )
                self.stats["gpu_direct_loads"] += 1
            else:
                # 일반 비동기 파일 읽기 후 CPU에서 텐서로 변환
                async with aiofiles.open(data_path, "rb") as f:
                    content = await f.read()

                if str(data_path).endswith(".npy"):
                    numpy_array = np.load(BytesIO(content))
                    tensor = torch.from_numpy(numpy_array).to(dtype)
                elif str(data_path).endswith(".csv"):
                    df = pd.read_csv(BytesIO(content))
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    tensor = torch.from_numpy(df[numeric_cols].values).to(dtype)
                else:
                    # Raw binary
                    tensor = torch.from_numpy(np.frombuffer(content, dtype=np.uint8))

            if (
                tensor is not None
                and self.device.type == "cuda"
                and tensor.device.type == "cpu"
            ):
                tensor = tensor.to(self.device, non_blocking=True)

            # 통합 캐시 관리자에 저장
            if tensor is not None:
                # 메모리에만 저장 (use_disk=False)
                self.cache_manager.set(cache_key, tensor, use_disk=False)

            load_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_stats(load_time_ms)
            logger.debug(f"데이터 로드 완료: {data_path} ({load_time_ms:.2f}ms)")
            return tensor

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"GPU 네이티브 로드 실패: {data_path} - {e}")
            return None

    def _update_stats(self, load_time_ms: float):
        """성능 통계 업데이트"""
        total = self.stats["loads"]
        self.stats["avg_load_time_ms"] = (
            self.stats["avg_load_time_ms"] * total + load_time_ms
        ) / (total + 1)
        self.stats["loads"] += 1

    def create_optimized_dataloader(
        self, dataset: Dataset, batch_size: int = 0, shuffle: bool = True
    ) -> TorchDataLoader:
        """
        최적화된 PyTorch DataLoader 생성 (동적 배치 크기 조정)
        """
        effective_batch_size = (
            batch_size if batch_size > 0 else self._get_dynamic_batch_size()
        )

        return TorchDataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            num_workers=self.max_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.max_workers > 0 else None,
            persistent_workers=True if self.max_workers > 0 else False,
        )

    def _get_dynamic_batch_size(self) -> int:
        """GPU 상태에 따른 동적 배치 크기 결정"""
        if not GPU_AVAILABLE:
            return 32

        total_mem = torch.cuda.get_device_properties(0).total_memory
        reserved_mem = torch.cuda.memory_reserved(0)
        free_mem = total_mem - reserved_mem
        free_ratio = free_mem / total_mem

        # 설정된 매핑을 기반으로 배치 크기 결정
        for threshold, batch_size in sorted(
            self.dynamic_batch_size_mapping.items(),
            key=lambda item: float(item[0]) if item[0] != "default" else -1,
            reverse=True,
        ):
            if threshold == "default":
                continue
            if free_ratio > float(threshold):
                return batch_size
        return self.dynamic_batch_size_mapping.get("default", 32)

    def get_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return self.stats

    def clear_cache(self):
        """캐시 비우기"""
        # 데이터 로더는 중앙 캐시를 직접 비우지 않음
        # 중앙 캐시 관리자를 통해 비워야 함
        # self.cache_manager.clear() # 필요한 경우 호출
        logger.info("데이터 로더 캐시는 중앙 cache_manager를 통해 관리됩니다.")


def get_data_loader() -> GPUNativeDataLoader:
    """GPU 네이티브 데이터 로더 반환 (싱글톤)"""
    return get_singleton_instance(GPUNativeDataLoader)


# --- 하위 호환성 및 편의 클래스/함수 ---


class LotteryDataset(Dataset):
    """로또 데이터셋"""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_prepare_data(file_path: str, **kwargs) -> Dataset:
    """데이터 로드 및 데이터셋 생성 통합 함수"""
    loader = get_data_loader()

    # 이 함수는 이제 비동기 로더를 사용하므로,
    # 실제 사용 시에는 `asyncio.run(loader.gpu_native_load(...))` 필요
    # 여기서는 개념적 예시를 위해 동기적으로 호출하는 것처럼 표현
    logger.warning("`load_and_prepare_data`는 비동기 컨텍스트에서 실행되어야 합니다.")

    # 예시: features와 labels를 파일에서 분리하여 로드한다고 가정
    # features = asyncio.run(loader.gpu_native_load(file_path + "_features.npy"))
    # labels = asyncio.run(loader.gpu_native_load(file_path + "_labels.npy"))

    # 임시 동기적 로드
    data = pd.read_csv(file_path)  # 실제로는 비동기로 읽어야 함
    features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    labels = torch.tensor(data.iloc[:, -1].values, dtype=torch.long)

    return LotteryDataset(features, labels)


# 기존 DataLoader 클래스는 GPUNativeDataLoader의 래퍼로 동작 가능
DataLoader = GPUNativeDataLoader


def load_draw_history(file_path: str = None) -> List:
    """
    로또 당첨 번호 이력을 로드합니다.

    Args:
        file_path: 데이터 파일 경로 (기본값: None, config에서 가져옴)

    Returns:
        List[LotteryNumber]: 로또 당첨 번호 리스트
    """
    from ..shared.types import LotteryNumber
    from pathlib import Path

    try:
        # 파일 경로 설정
        if file_path is None:
            # 기본 경로 사용
            config = get_config("main")
            data_config = config.get("data", {})
            file_path = data_config.get("historical_data_path", "data/raw/lottery.csv")

        # 파일 존재 확인
        if not Path(file_path).exists():
            logger.warning(f"데이터 파일이 존재하지 않습니다: {file_path}")
            return []

        # CSV 파일 로드
        df = pd.read_csv(file_path)

        # 필요한 컬럼이 있는지 확인
        required_columns = ["draw_no"]
        number_columns = ["num1", "num2", "num3", "num4", "num5", "num6"]

        if not all(col in df.columns for col in required_columns):
            logger.error(f"필수 컬럼이 없습니다: {required_columns}")
            return []

        # 번호 컬럼이 있는지 확인 (다양한 형태 지원)
        if all(col in df.columns for col in number_columns):
            # num1, num2, ... 형태
            pass
        elif "numbers" in df.columns:
            # numbers 컬럼에 모든 번호가 있는 경우
            number_columns = ["numbers"]
        else:
            logger.error("번호 컬럼을 찾을 수 없습니다")
            return []

        # LotteryNumber 객체 리스트 생성
        lottery_numbers = []

        for _, row in df.iterrows():
            try:
                draw_no = int(row["draw_no"])

                # 번호 추출
                if len(number_columns) == 6:  # num1, num2, ... 형태
                    numbers = [int(row[col]) for col in number_columns]
                else:  # numbers 컬럼 형태
                    numbers_str = str(row["numbers"])
                    if "," in numbers_str:
                        numbers = [int(x.strip()) for x in numbers_str.split(",")]
                    else:
                        # 공백으로 분리된 경우
                        numbers = [int(x) for x in numbers_str.split()]

                # 날짜 정보 (있는 경우)
                date = None
                if "date" in df.columns:
                    date = str(row["date"])
                elif "draw_date" in df.columns:
                    date = str(row["draw_date"])

                # LotteryNumber 객체 생성
                lottery_number = LotteryNumber(
                    draw_no=draw_no, numbers=numbers, date=date
                )

                lottery_numbers.append(lottery_number)

            except Exception as e:
                logger.warning(f"행 {row.name} 처리 실패: {e}")
                continue

        logger.info(f"로또 데이터 로드 완료: {len(lottery_numbers)}개 회차")
        return lottery_numbers

    except Exception as e:
        logger.error(f"로또 데이터 로드 실패: {e}")
        return []
