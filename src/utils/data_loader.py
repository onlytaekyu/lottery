"""
GPU 네이티브 데이터 로더 (v4 - 제로카피 최적화)

Pinned Memory와 제로카피(Zero-copy) 변환을 통해 CPU를 거치지 않고
데이터를 GPU로 직접 로딩하여, 로딩 속도를 극대화합니다.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from .unified_logging import get_logger
from .unified_memory_manager import get_unified_memory_manager
from .unified_config import get_paths

logger = get_logger(__name__)

class DataLoader:
    """
    데이터 로딩을 위한 단순화된 클래스.
    주로 CSV나 npy와 같은 파일들을 로드합니다.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.paths = get_paths()
        
        self.memory_manager = None
        if torch.cuda.is_available():
            self.memory_manager = get_unified_memory_manager()

    def load_lottery_data(self, file_name: str = "lotto_history.csv") -> pd.DataFrame:
        """CSV 형태의 로또 데이터를 로드합니다."""
        file_path = Path(self.paths.raw_data_dir) / file_name
        if not file_path.exists():
            logger.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
            raise FileNotFoundError(f"No data file found at {file_path}")
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"{file_path}에서 {len(df)}개의 데이터를 로드했습니다.")
            return df
        except Exception as e:
            logger.error(f"{file_path} 파일 로드 중 오류 발생: {e}")
            raise

    def load_feature_vectors(self, file_name: str = "feature_vectors.npy") -> Optional[np.ndarray]:
        """Numpy 형태로 저장된 특성 벡터를 로드합니다."""
        vector_path = Path(self.paths.cache_dir) / file_name
        if not vector_path.exists():
            logger.warning("사전 계산된 특성 벡터 파일을 찾을 수 없습니다: {vector_path}")
            return None
        
        try:
            logger.info(f"{vector_path}에서 특성 벡터를 로드합니다...")
            return np.load(vector_path, allow_pickle=True)
        except Exception as e:
            logger.error(f"{vector_path} 파일 로드 중 오류 발생: {e}")
            return None

def get_data_loader(config: Optional[Dict[str, Any]] = None) -> DataLoader:
    """DataLoader의 인스턴스를 반환합니다."""
    return DataLoader(config)
