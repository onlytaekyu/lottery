"""
GPU 가속 유틸리티 모음

이 모듈은 여러 분석기에서 공통적으로 사용될 수 있는
GPU 가속 계산 함수들을 제공합니다.
"""

from typing import Dict
import torch

def calculate_frequencies_gpu(data_tensor: torch.Tensor, min_length: int = 46) -> Dict[int, int]:
    """
    GPU를 사용하여 텐서에 포함된 각 숫자의 빈도를 계산합니다.

    Args:
        data_tensor (torch.Tensor): 빈도를 계산할 데이터가 포함된 1D 또는 2D 텐서.
                                     GPU 장치에 있어야 합니다.
        min_length (int): 빈도 계산을 위한 최소 길이. 로또 번호(1-45)를 위해 기본값은 46입니다.

    Returns:
        Dict[int, int]: 각 숫자를 키로, 빈도를 값으로 갖는 딕셔너리.
    """
    if not isinstance(data_tensor, torch.Tensor):
        raise TypeError("입력 데이터는 반드시 torch.Tensor여야 합니다.")

    if data_tensor.is_cuda is False:
        raise ValueError("입력 텐서는 반드시 CUDA 장치에 있어야 합니다.")

    if data_tensor.numel() == 0:
        return {}

    # 텐서를 1D로 만들고 bincount를 사용하여 빈도 계산
    frequencies = torch.bincount(data_tensor.flatten(), minlength=min_length)

    # GPU에서 CPU로 결과 이동 후 딕셔너리로 변환
    frequencies_cpu = frequencies.cpu().numpy()

    # 0번 인덱스는 무시하고 1부터의 결과만 사용
    return {i: int(frequencies_cpu[i]) for i in range(1, min_length)} 