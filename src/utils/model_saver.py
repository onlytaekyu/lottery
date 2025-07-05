"""
GPU 네이티브 모델 저장/로드 시스템으로 전환
"""

import torch
import asyncio
import io
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import hashlib
from datetime import datetime

from .unified_logging import get_logger
from .async_io import get_gpu_async_io_manager
from .memory_manager import get_memory_manager

logger = get_logger(__name__)


class GPUNativeModelSaver:
    """GPU 네이티브 모델 저장/로드 시스템"""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or "savedModels")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.async_io = get_gpu_async_io_manager()
        self.memory_manager = get_memory_manager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"✅ GPU 네이티브 모델 저장 시스템 초기화: {self.device}")

    async def gpu_save(
        self,
        model: torch.nn.Module,
        name: str,
        version: str = "latest",
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """GPU에서 직접 모델 직렬화 및 비동기 저장"""

        # 저장 경로 생성
        save_dir = self.base_dir / name / version
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. GPU 메모리에서 직접 직렬화
            buffer = io.BytesIO()

            # 모델을 GPU에 유지한 채로 저장
            model_state = model.state_dict()
            torch.save(model_state, buffer)
            buffer.seek(0)
            model_data = buffer.read()
            buffer.close()

            # 2. 메타데이터 준비
            metadata = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "model_size": len(model_data),
                "device": str(self.device),
                "config": config or {},
                "metrics": metrics or {},
                "gpu_serialized": True,
            }

            # 3. 비동기 파일 저장
            tasks = [
                # 모델 데이터
                self.async_io.smart_write_file(save_dir / "model.pt", model_data),
                # 메타데이터
                self.async_io.smart_write_file(
                    save_dir / "metadata.json", json.dumps(metadata, indent=2).encode()
                ),
                # 설정
                self.async_io.smart_write_file(
                    save_dir / "config.json",
                    json.dumps(config or {}, indent=2).encode(),
                ),
            ]

            await asyncio.gather(*tasks)

            logger.info(f"✅ GPU 모델 저장 완료: {save_dir}")
            return str(save_dir)

        except Exception as e:
            logger.error(f"❌ GPU 모델 저장 실패: {e}")
            raise

    async def gpu_load(
        self, name: str, version: str = "latest", device: Optional[str] = None
    ) -> Dict[str, Any]:
        """GPU로 직접 모델 로드"""

        load_dir = self.base_dir / name / version
        if not load_dir.exists():
            raise FileNotFoundError(f"모델을 찾을 수 없습니다: {load_dir}")

        target_device = device or str(self.device)

        try:
            # 1. 비동기 파일 로드
            tasks = [
                self.async_io.smart_read_file(load_dir / "model.pt"),
                self.async_io.smart_read_file(load_dir / "metadata.json"),
                self.async_io.smart_read_file(load_dir / "config.json"),
            ]

            model_data, metadata_data, config_data = await asyncio.gather(*tasks)

            # 2. 메타데이터 파싱
            if metadata_data is None or config_data is None:
                raise ValueError("모델 메타데이터 또는 설정 파일을 로드할 수 없습니다.")

            metadata = json.loads(metadata_data.decode())
            config = json.loads(config_data.decode())

            # 3. GPU로 직접 로드
            if model_data is None:
                raise ValueError("모델 데이터를 로드할 수 없습니다.")

            buffer = io.BytesIO(model_data)
            state_dict = torch.load(
                buffer, map_location=target_device, weights_only=True
            )
            buffer.close()

            logger.info(f"✅ GPU 모델 로드 완료: {load_dir}")

            return {
                "state_dict": state_dict,
                "metadata": metadata,
                "config": config,
                "device": target_device,
            }

        except Exception as e:
            logger.error(f"❌ GPU 모델 로드 실패: {e}")
            raise

    def sync_save(self, model, name: str, **kwargs) -> str:
        """동기식 저장 (하위 호환성)"""
        return asyncio.run(self.gpu_save(model, name, **kwargs))

    def sync_load(self, name: str, **kwargs) -> Dict[str, Any]:
        """동기식 로드 (하위 호환성)"""
        return asyncio.run(self.gpu_load(name, **kwargs))


# 싱글톤 인스턴스
_gpu_model_saver: Optional[GPUNativeModelSaver] = None


def get_advanced_model_saver(base_dir: Optional[str] = None) -> GPUNativeModelSaver:
    """GPU 모델 저장기 싱글톤 반환"""
    global _gpu_model_saver
    if _gpu_model_saver is None:
        _gpu_model_saver = GPUNativeModelSaver(base_dir)
    return _gpu_model_saver


# 편의 함수들 (하위 호환성)
def save_model(model, model_type: str, config=None, **kwargs) -> str:
    """편의 함수: 모델 저장"""
    saver = get_advanced_model_saver()
    return saver.sync_save(model, model_type, config=config, **kwargs)


def load_model(model_class, path: str, **kwargs):
    """편의 함수: 모델 로드"""
    saver = get_advanced_model_saver()
    data = saver.sync_load(path.split("/")[-2], **kwargs)

    model = model_class()
    model.load_state_dict(data["state_dict"])
    return model
