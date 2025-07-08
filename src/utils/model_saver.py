"""
GPU 네이티브 모델 저장/로드 시스템으로 전환
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any

from .unified_logging import get_logger
from .unified_memory_manager import get_unified_memory_manager
from .unified_config import get_paths

logger = get_logger(__name__)


class ModelSaver:
    def __init__(self, base_path: Optional[str] = None):
        self.paths = get_paths()
        self.base_path = Path(base_path or self.paths.models_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.memory_manager = get_unified_memory_manager()

    def save(self, model: Any, name: str, metadata: Optional[Dict[str, Any]] = None):
        file_path = self.base_path / f"{name}.pt"
        try:
            # GPU에서 모델을 저장할 경우, CPU로 이동하여 저장하는 것이 안전
            if isinstance(model, torch.nn.Module) and next(model.parameters()).is_cuda:
                model.to('cpu')

            save_obj = {"model_state_dict": model.state_dict(), "metadata": metadata}
            torch.save(save_obj, file_path)
            
            logger.info(f"Model '{name}' saved to {file_path}")
            
            # 메모리 정리
            if self.memory_manager:
                self.memory_manager.cleanup_all()

        except Exception as e:
            logger.error(f"Failed to save model '{name}' to {file_path}: {e}")
            raise

    def load(self, model: Any, name: str) -> Optional[Dict[str, Any]]:
        file_path = self.base_path / f"{name}.pt"
        if not file_path.exists():
            logger.warning(f"Model file not found: {file_path}")
            return None
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            save_obj = torch.load(file_path, map_location=device)
            model.load_state_dict(save_obj["model_state_dict"])
            model.to(device) # 모델을 적절한 장치로 이동
            
            logger.info(f"Model '{name}' loaded from {file_path}")
            return save_obj.get("metadata")
            
        except Exception as e:
            logger.error(f"Failed to load model '{name}' from {file_path}: {e}")
            raise

def get_model_saver(base_path: Optional[str] = None) -> ModelSaver:
    return ModelSaver(base_path)
