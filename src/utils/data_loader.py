"""
로또 번호 예측을 위한 데이터 로더 V2 (GPU 최적화 버전)

GPU 우선순위 연산과 메모리 효율성을 극대화한 데이터 로더입니다.
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
import json
from collections import Counter

from .unified_logging import get_logger
from ..shared.types import LotteryNumber

logger = get_logger(__name__)


class DataQualityValidator:
    """간단한 데이터 품질 검증 클래스"""

    def __init__(self):
        self.logger = get_logger(__name__)

    def validate_basic_quality(self, data: List[LotteryNumber]) -> bool:
        """기본 품질 검증"""
        if not data:
            self.logger.error("빈 데이터셋")
            return False

        # 중복 회차 검증
        draw_numbers = [item.draw_no for item in data]
        if len(set(draw_numbers)) != len(draw_numbers):
            self.logger.warning("중복 회차 발견")

        # 번호 범위 검증
        for item in data:
            if not all(1 <= num <= 45 for num in item.numbers):
                self.logger.error(f"회차 {item.draw_no}: 번호 범위 오류")
                return False

        return True


class DataTransformer:
    """GPU 우선순위 데이터 변환 클래스"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.logger = get_logger(__name__)

        if self.use_gpu:
            self.logger.info("GPU 가속 데이터 변환 활성화")
        else:
            self.logger.info("CPU 데이터 변환 모드")

    def transform_to_tensor(self, data: List[LotteryNumber]) -> torch.Tensor:
        """로또 데이터를 GPU 텐서로 변환"""
        if not data:
            return torch.empty(0, 6, device=self.device)

        # NumPy 배열로 먼저 변환
        numbers_array = np.array([item.numbers for item in data])

        # GPU 텐서로 변환
        tensor = torch.from_numpy(numbers_array).float().to(self.device)

        return tensor

    def normalize_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """GPU에서 정규화 수행"""
        if tensor.numel() == 0:
            return tensor

        # GPU에서 min-max 정규화
        tensor_min = tensor.min(dim=1, keepdim=True)[0]
        tensor_max = tensor.max(dim=1, keepdim=True)[0]

        # 0으로 나누기 방지
        range_tensor = tensor_max - tensor_min
        range_tensor = torch.where(
            range_tensor == 0, torch.ones_like(range_tensor), range_tensor
        )

        normalized = (tensor - tensor_min) / range_tensor
        return normalized


def load_draw_history(file_path: Optional[str] = None) -> List[LotteryNumber]:
    """로또 당첨 번호 이력을 로드합니다."""
    if file_path is None:
        file_path = str(Path(__file__).parent.parent.parent / "data/raw/lottery.csv")

    if not Path(file_path).exists():
        logger.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        return []

    try:
        logger.info(f"데이터 로드 중: {file_path}")
        df = pd.read_csv(file_path, encoding="utf-8")

        lottery_numbers = []
        for _, row in df.iterrows():
            try:
                numbers = [int(row[f"num{i}"]) for i in range(1, 7)]
                lottery_numbers.append(
                    LotteryNumber(draw_no=int(row["seqNum"]), numbers=numbers)
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"데이터 처리 중 오류 발생 (행: {row.name}): {e}")
                continue

        logger.info(f"데이터 로드 완료: {len(lottery_numbers)}개 회차")
        return lottery_numbers
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return []


def validate_lottery_data(data: List[LotteryNumber]) -> bool:
    """로또 데이터 기본 검증"""
    validator = DataQualityValidator()
    return validator.validate_basic_quality(data)


class LotteryDataset(Dataset):
    """PyTorch용 로또 데이터셋 클래스"""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class DataLoader:
    """데이터 로더 클래스"""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.data = None

    def load_data(self) -> List[LotteryNumber]:
        """데이터 로드"""
        if self.data is None:
            self.data = load_draw_history(self.file_path)
        return self.data

    def get_recent_data(self, count: int = 100) -> List[LotteryNumber]:
        """최근 데이터 조회"""
        data = self.load_data()
        return data[-count:] if len(data) > count else data


class LotteryJSONEncoder(json.JSONEncoder):
    """로또 데이터용 JSON 인코더"""

    def default(self, obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def create_dataloader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> TorchDataLoader:
    """PyTorch DataLoader 생성"""
    dataset = LotteryDataset(features, labels)
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_and_prepare_data(
    file_path: Optional[str] = None,
    validate: bool = True,
    transform: bool = False,
) -> Union[List[LotteryNumber], Tuple[torch.Tensor, torch.Tensor]]:
    """데이터 로드 및 준비"""
    raw_data = load_draw_history(file_path)

    if validate and not validate_lottery_data(raw_data):
        raise ValueError("데이터 검증 실패")

    if transform:
        # 간단한 변환 예시
        features = torch.randn(len(raw_data), 6)  # 실제 구현 필요
        labels = torch.randn(len(raw_data), 1)  # 실제 구현 필요
        return features, labels

    return raw_data
