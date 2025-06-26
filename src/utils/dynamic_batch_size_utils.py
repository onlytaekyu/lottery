"""
배치 크기 동적 조절 유틸리티

이 모듈은 학습 과정에서 메모리 사용량에 따라
배치 크기를 안전하게 조절하는 유틸리티 함수를 제공합니다.
"""

import torch
import gc
from typing import Optional, Union, Tuple
import logging
import numpy as np
from pathlib import Path

from .error_handler import get_logger

# 로거 설정
logger = get_logger(__name__)


def get_available_memory(
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[int, int]:
    """
    사용 가능한 메모리 확인

    Args:
        device: 장치 (기본값: 현재 활성 장치)

    Returns:
        (사용 가능한 메모리 크기, 총 메모리 크기) (바이트 단위)
    """
    # 장치 확인
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # GPU 메모리 확인
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            # 현재 장치의 인덱스
            index = device.index if device.index is not None else 0

            # 총 메모리
            total_memory = torch.cuda.get_device_properties(index).total_memory

            # 현재 할당된 메모리
            allocated_memory = torch.cuda.memory_allocated(index)

            # 현재 예약된 메모리 (캐시 포함)
            reserved_memory = torch.cuda.memory_reserved(index)

            # 사용 가능한 메모리 계산 (예약된 메모리를 고려하여 계산)
            available_memory = total_memory - reserved_memory

            return available_memory, total_memory
        except Exception as e:
            logger.warning(f"GPU 메모리 확인 실패: {str(e)}, CPU 사용")

    # CPU 메모리 확인
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.available, vm.total
    except ImportError:
        logger.warning("psutil 모듈을 찾을 수 없습니다. 기본값 반환.")
        return 8 * 1024**3, 16 * 1024**3  # 기본값: 사용 가능 8GB, 총 16GB


def get_safe_batch_size(
    initial_batch_size: int = 32,
    min_batch_size: int = 1,
    max_batch_size: int = 256,
    sample_size_bytes: Optional[int] = None,
    model_size_bytes: Optional[int] = None,
    memory_margin: float = 0.2,
    device: Optional[Union[str, torch.device]] = None,
) -> int:
    """
    안전한 배치 크기 계산

    Args:
        initial_batch_size: 초기 배치 크기
        min_batch_size: 최소 배치 크기
        max_batch_size: 최대 배치 크기
        sample_size_bytes: 샘플 하나의 크기 (바이트 단위, 없으면 추정)
        model_size_bytes: 모델 크기 (바이트 단위, 없으면 추정)
        memory_margin: 메모리 여유 공간 비율 (0~1)
        device: 장치

    Returns:
        안전한 배치 크기
    """
    # 장치 확인
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # 메모리 확인
    available_memory, total_memory = get_available_memory(device)

    # 메모리 여유 공간 고려
    available_memory = int(available_memory * (1 - memory_margin))

    # 샘플 크기와 모델 크기를 알 수 없는 경우
    if sample_size_bytes is None and model_size_bytes is None:
        # GPU 사용 시 보수적으로 메모리 할당
        if device.type == "cuda":
            # 일반적인 작은 모델 가정: 총 가용 메모리의 30%를 사용
            batch_size = min(
                max(
                    min_batch_size,
                    int(
                        available_memory * 0.3 / (4 * 1024 * 1024)
                    ),  # 샘플당 약 4MB 가정
                ),
                max_batch_size,
            )
        else:
            # CPU는 좀 더 큰 배치 크기 허용
            batch_size = min(max(initial_batch_size, min_batch_size), max_batch_size)
    else:
        # 샘플 크기 추정 (제공되지 않은 경우)
        if sample_size_bytes is None:
            sample_size_bytes = 4 * 1024 * 1024  # 기본값: 4MB

        # 모델 크기 추정 (제공되지 않은 경우)
        if model_size_bytes is None:
            model_size_bytes = int(total_memory * 0.2)  # 기본값: 총 메모리의 20%

        # 배치당 메모리 계산
        batch_memory = sample_size_bytes * initial_batch_size

        # 사용 가능한 메모리에 맞는 배치 크기 계산
        if available_memory > model_size_bytes + batch_memory:
            # 초기 배치 크기가 적합함
            batch_size = initial_batch_size
        else:
            # 사용 가능한 메모리에 맞게 배치 크기 조정
            usable_memory = max(0, available_memory - model_size_bytes)
            batch_size = max(min_batch_size, int(usable_memory / sample_size_bytes))
            batch_size = min(batch_size, max_batch_size)

    logger.info(
        f"계산된 안전 배치 크기: {batch_size} "
        f"(가용 메모리: {available_memory/(1024**3):.2f}GB, "
        f"장치: {device})"
    )

    return batch_size


def clean_memory():
    """
    메모리 정리

    PyTorch 캐시와 가비지 컬렉션을 실행하여 메모리를 정리합니다.
    """
    # PyTorch 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 가비지 컬렉션 실행
    gc.collect()


def estimate_model_size(model: torch.nn.Module) -> int:
    """
    모델 크기 추정

    Args:
        model: PyTorch 모델

    Returns:
        모델 크기 (바이트 단위)
    """
    # 모델 파라미터 메모리 계산
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # 버퍼 메모리 계산 (BatchNorm 등에 사용)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # 모델 복제본, 그래디언트 등의 추가 메모리 고려
    total_size = param_size + buffer_size
    total_size = int(
        total_size * 3
    )  # 가중치, 그래디언트, 옵티마이저 상태를 고려한 대략적인 추정

    return total_size


def adjust_for_mixed_precision(batch_size: int, enabled: bool = False) -> int:
    """
    혼합 정밀도(mixed precision) 학습을 위한 배치 크기 조정

    Args:
        batch_size: 기존 배치 크기
        enabled: 혼합 정밀도 사용 여부

    Returns:
        조정된 배치 크기
    """
    if not enabled:
        return batch_size

    # 혼합 정밀도 사용 시 메모리 사용량이 감소하므로 배치 크기 증가 가능
    adjusted_batch_size = int(batch_size * 1.6)  # 약 1.6배 증가 (float32 -> float16)

    logger.debug(f"혼합 정밀도 조정: {batch_size} -> {adjusted_batch_size}")
    return adjusted_batch_size
