"""DEPRECATED: moved to cuda_optimizers.py
This stub remains for backward compatibility.
"""

from __future__ import annotations

from .cuda_optimizers import (
    BatchSizeConfig,
    DynamicBatchSizeController,
    get_safe_batch_size,
)

__all__ = [
    "BatchSizeConfig",
    "DynamicBatchSizeController",
    "get_safe_batch_size",
]
