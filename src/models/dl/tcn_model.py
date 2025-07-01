# 1. 표준 라이브러리
import os, time
from typing import Any, Dict, Optional

# 2. 서드파티
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 3. 프로젝트 내부
from ..base_model import BaseModel
from ...utils.unified_logging import get_logger

logger = get_logger(__name__)


def _build_tcn_block(
    in_channels: int, out_channels: int, kernel_size: int, dilation: int
) -> nn.Module:
    """Dilated Conv + ReLU + Residual"""
    padding = (kernel_size - 1) * dilation
    conv = nn.Conv1d(
        in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
    )
    return nn.Sequential(conv, nn.ReLU())


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.block = _build_tcn_block(in_ch, out_ch, kernel_size, dilation)
        self.resample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.block(x)
        res = self.resample(x)
        return out + res


class _TCN(nn.Module):
    def __init__(
        self,
        num_numbers: int,
        seq_len: int = 50,
        levels: int = 4,
        kernel_size: int = 3,
        hidden: int = 64,
    ):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(levels):
            dilation = 2**i
            layers.append(_ResidualBlock(in_ch, hidden, kernel_size, dilation))
            in_ch = hidden
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(hidden * seq_len, num_numbers)

    def forward(self, x):  # x: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.network(x)
        x = x.flatten(1)
        return torch.sigmoid(self.head(x))  # 확률 반환


class TCNModel(BaseModel):
    """시계열 로또 번호 출현 확률 예측용 TCN 모델"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        seq_len = self.config.get("seq_len", 50)
        num_numbers = self.config.get("num_numbers", 45)
        self.lr = self.config.get("lr", 1e-3)
        self.model = _TCN(num_numbers, seq_len).to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, **kwargs):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            preds = self.model(X_t)
            loss = self.loss_fn(preds, y_t)
            loss.backward()
            self.optimizer.step()
            if epoch % 2 == 0:
                logger.info(f"TCN epoch {epoch}/{epochs} loss={loss.item():.4f}")
        self.is_trained = True
        return {"epochs": epochs, "final_loss": loss.item()}

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("모델이 훈련되지 않았습니다.")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            preds = self.model(X_t).cpu().numpy()
        return preds

    def save(self, path: str) -> bool:
        try:
            self._ensure_directory(path)
            torch.save(self.model.state_dict(), path)
            logger.info(f"TCN 모델 저장 완료: {path}")
            return True
        except Exception as e:
            logger.error(f"TCN 모델 저장 실패: {e}")
            return False

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.error(f"모델 파일 없음: {path}")
            return False
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.is_trained = True
        logger.info(f"TCN 모델 로드 완료: {path}")
        return True
