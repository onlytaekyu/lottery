"""
로또 번호 예측을 위한 데이터 로더 V2 (모듈화 버전)

데이터 로딩, 검증, 변환의 책임을 각각의 전문 모듈로 분리하여
유지보수성과 확장성을 극대화한 버전입니다.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import List, Optional, Union, Tuple
from pathlib import Path

from .unified_logging import get_logger
from .data_validator import DataQualityValidator
from .data_transformer import DataTransformer
from ..shared.types import LotteryNumber

logger = get_logger(__name__)


def load_and_prepare_data(
    file_path: Optional[str] = None,
    validate: bool = True,
    transform: bool = True,
) -> Union[List[LotteryNumber], Tuple[torch.Tensor, torch.Tensor]]:
    """
    데이터를 로드하고, 선택적으로 검증 및 변환을 수행합니다.

    Args:
        file_path: 데이터 파일 경로.
        validate: 데이터 품질 검증 수행 여부.
        transform: 데이터 벡터 변환 수행 여부.

    Returns:
        변환 여부에 따라 LotteryNumber 리스트 또는 (특성, 레이블) 텐서 튜플.
    """
    # 1. 데이터 로딩
    raw_data = _load_raw_data(file_path)

    # 2. 데이터 검증 (선택 사항)
    if validate:
        validator = DataQualityValidator()
        validation_result = validator.validate_lottery_data(raw_data)
        validator.log_validation_result(validation_result)
        if not validation_result["is_valid"]:
            raise ValueError("데이터 품질 검증 실패. 자세한 내용은 로그를 확인하세요.")

    # 3. 데이터 변환 (선택 사항)
    if transform:
        transformer = DataTransformer()
        features, labels = transformer.transform_to_vectors(raw_data)

        # 텐서로 변환
        feature_tensor = torch.from_numpy(features).float()
        label_tensor = torch.from_numpy(labels).float().unsqueeze(1)

        return feature_tensor, label_tensor

    return raw_data


def _load_raw_data(file_path: Optional[str] = None) -> List[LotteryNumber]:
    """CSV 파일에서 원본 로또 데이터를 로드합니다."""
    if file_path is None:
        file_path = str(Path(__file__).parent.parent.parent / "data/raw/lottery.csv")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

    logger.info(f"데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path, encoding="utf-8")

    lottery_numbers = []
    for _, row in df.iterrows():
        try:
            numbers = [int(row[f"num{i}"]) for i in range(1, 7)]
            lottery_numbers.append(
                LotteryNumber(draw_no=int(row["seqNum"]), numbers=numbers)
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"데이터 처리 중 오류 발생 (행: {row.name}): {e}")
            continue

    logger.info(f"데이터 로드 완료: {len(lottery_numbers)}개 회차")
    return lottery_numbers


class LotteryDataset(Dataset):
    """PyTorch용 로또 데이터셋 클래스"""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def create_dataloader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> TorchDataLoader:
    """
    PyTorch DataLoader를 생성합니다.

    Args:
        features: 특성 텐서
        labels: 레이블 텐서
        batch_size: 배치 크기
        shuffle: 데이터 셔플 여부
        num_workers: 데이터 로딩 워커 수

    Returns:
        설정된 PyTorch DataLoader 객체
    """
    dataset = LotteryDataset(features, labels)
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
