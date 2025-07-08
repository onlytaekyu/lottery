"""
패턴 필터링 모듈

이 모듈은 로또 번호 패턴을 필터링하는 기능을 제공합니다.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from ..utils.unified_logging import get_logger
from ..utils.unified_config import load_config
from ..utils.unified_memory_manager import get_unified_memory_manager

logger = get_logger(__name__)

GPU_AVAILABLE = torch.cuda.is_available()

_instance: Optional["PatternFilter"] = None


def get_pattern_filter(config: Optional[Dict[str, Any]] = None) -> "PatternFilter":
    global _instance
    if _instance is None:
        if config is None:
            config = load_config("analysis")
        _instance = PatternFilter(config)
    return _instance


class PatternFilter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("pattern_filter", {})
        self.logger = get_logger(__name__)

        self.gpu_available = GPU_AVAILABLE
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
        
        self.cuda_optimizer = None
        self.memory_manager = None
        if self.gpu_available:
            from .cuda_optimizers import get_cuda_optimizer
            self.cuda_optimizer = get_cuda_optimizer()
            self.memory_manager = get_unified_memory_manager()

        # 필터 설정
        self.min_frequency = self.config.get("min_frequency", 5)
        self.max_consecutive = self.config.get("max_consecutive", 4)

    def filter_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        if not patterns:
            return {}

        try:
            if self.gpu_available and self.cuda_optimizer:
                return self._filter_gpu(patterns)
            else:
                return self._filter_cpu(patterns)
        except Exception as e:
            self.logger.error(f"패턴 필터링 중 예상치 못한 오류 발생: {e}")
            return {}

    def _filter_gpu(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"GPU를 사용하여 {len(patterns)}개의 패턴 필터링 시작")
        
        filtered_patterns = {}
        # GPU 필터링 로직 구현 (기존 코드 참조)
        # ... (생략) ...

        if self.memory_manager:
            self.memory_manager.release_cuda_cache("pattern_filter._filter_gpu")
        
        return filtered_patterns

    def _filter_cpu(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"CPU를 사용하여 {len(patterns)}개의 패턴 필터링 시작")
        
        filtered_patterns = {}
        for key, info in patterns.items():
            if self._is_valid_cpu(info):
                filtered_patterns[key] = info
        
        return filtered_patterns

    def _is_valid_cpu(self, pattern_info: Dict[str, Any]) -> bool:
        frequency = pattern_info.get("frequency", 0)
        if frequency < self.min_frequency:
            return False
            
        numbers = pattern_info.get("numbers", [])
        if len(numbers) > 1:
            diffs = np.diff(sorted(numbers))
            consecutive_count = np.sum(diffs == 1)
            if consecutive_count >= self.max_consecutive:
                return False

        return True

    def clear_caches(self):
        # 이 클래스에서는 lru_cache를 사용하지 않으므로 비워둡니다.
        pass
