"""
Transformer 모델

이 모듈은 Transformer 아키텍처 기반 로또 번호 예측 모델을 구현합니다.
Attention 메커니즘을 활용하여 로또 번호 간의 복잡한 패턴을 학습합니다.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time

from ..base_model import ModelWithAMP
from ...utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder 모듈

    입력 특성에 대한 self-attention 기반 인코딩을 수행합니다.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Transformer Encoder 초기화

        Args:
            input_dim: 입력 차원
            d_model: 모델 차원
            nhead: 어텐션 헤드 수
            num_layers: 인코더 레이어 수
            dim_feedforward: 피드포워드 네트워크 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()

        # 입력 임베딩
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 포지셔널 인코딩
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer 인코더 레이어
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Transformer 인코더
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # 출력 차원
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 [batch_size, seq_len, input_dim]

        Returns:
            인코딩된 텐서 [batch_size, seq_len, d_model]
        """
        # 입력 임베딩
        x = self.input_embedding(x)

        # 포지셔널 인코딩
        x = self.pos_encoder(x)

        # Transformer 인코더
        output = self.transformer_encoder(x)

        return output


class PositionalEncoding(nn.Module):
    """
    포지셔널 인코딩 모듈

    시퀀스 내 위치 정보를 인코딩합니다.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        """
        포지셔널 인코딩 초기화

        Args:
            d_model: 모델 차원
            dropout: 드롭아웃 비율
            max_len: 최대 시퀀스 길이
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 포지셔널 인코딩 행렬 생성
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 등록된 버퍼로 저장 (파라미터가 아님)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]

        Returns:
            포지셔널 인코딩이 적용된 텐서
        """
        # 포지셔널 인코딩 적용
        x = x + self.pe[: x.size(1)].unsqueeze(0)

        return self.dropout(x)


class TransformerModel(ModelWithAMP):
    """
    Transformer 기반 로또 번호 예측 모델

    Attention 메커니즘을 활용하여 로또 번호 간의 패턴을 학습합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Transformer 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 설정
        self.config = config or {}
        transformer_config = self.config.get("transformer", {})

        # 모델 하이퍼파라미터
        self.input_dim = transformer_config.get("input_dim", 70)
        self.d_model = transformer_config.get("d_model", 128)
        self.nhead = transformer_config.get("nhead", 4)
        self.num_layers = transformer_config.get("num_layers", 2)
        self.dim_feedforward = transformer_config.get("dim_feedforward", 512)
        self.dropout = transformer_config.get("dropout", 0.1)
        self.output_dim = transformer_config.get("output_dim", 1)
        self.learning_rate = transformer_config.get("learning_rate", 0.001)

        # 모델 이름
        self.model_name = "TransformerModel"

        # 모델 구성
        self._build_model()

        logger.info(
            f"Transformer 모델 초기화 완료: 입력 차원={self.input_dim}, "
            f"모델 차원={self.d_model}, 어텐션 헤드={self.nhead}, "
            f"레이어 수={self.num_layers}, 드롭아웃={self.dropout}"
        )

    def _build_model(self):
        """모델 구성"""
        # Transformer 인코더
        self.encoder = TransformerEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        # 출력 레이어
        self.fc_out = nn.Linear(self.d_model, self.output_dim)

        # 손실 함수
        self.criterion = nn.MSELoss()

        # 옵티마이저
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # 장치로 이동
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 [batch_size, input_dim]

        Returns:
            예측 점수 [batch_size, output_dim]
        """
        # 입력 차원 변환 [batch_size, input_dim] -> [batch_size, 1, input_dim]
        x = x.unsqueeze(1)

        # 인코더 통과
        encoded = self.encoder(x)

        # 마지막 시퀀스 위치의 출력 사용
        output = encoded[:, 0, :]

        # 출력 레이어
        return self.fc_out(output)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 훈련

        Args:
            X: 특성 벡터
            y: 타겟 점수
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"Transformer 모델 훈련 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 훈련 모드로 설정
        self.train()

        # 훈련 매개변수
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 32)
        validation_split = kwargs.get("validation_split", 0.2)
        patience = kwargs.get("patience", 10)

        # 검증 세트 분리
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # PyTorch 텐서로 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # DataLoader 생성
        from torch.utils.data import TensorDataset, DataLoader

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 훈련 지표
        best_val_loss = float("inf")
        patience_counter = 0
        training_history = []

        # 훈련 시작
        start_time = time.time()

        for epoch in range(epochs):
            # 훈련 루프
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                # AMP를 적용한 훈련 단계
                loss = self.train_step_with_amp(
                    self, batch_X, batch_y, self.optimizer, self.criterion
                )
                train_loss += loss

            # 에포크 평균 손실
            train_loss /= len(train_loader)

            # 검증
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
            self.train()

            # 에포크 결과 기록
            training_history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

            # 로깅
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"에포크 {epoch+1}/{epochs}: 훈련 손실={train_loss:.4f}, 검증 손실={val_loss:.4f}"
                )

            # 조기 종료 확인
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # 최상의 모델 저장
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"조기 종료: 에포크 {epoch+1}")
                    break

        # 최상의 모델 복원
        self.load_state_dict(best_model_state)

        # 훈련 시간 계산
        train_time = time.time() - start_time

        # 모델 훈련 완료 표시
        self.is_trained = True
        self.training_history = training_history

        # 메타데이터 업데이트
        self.metadata.update(
            {
                "train_samples": X_train.shape[0],
                "val_samples": X_val.shape[0],
                "best_val_loss": best_val_loss,
                "epochs_trained": epoch + 1,
                "train_time": train_time,
            }
        )

        logger.info(
            f"Transformer 모델 훈련 완료: 최종 검증 손실={best_val_loss:.4f}, "
            f"에포크={epoch+1}/{epochs}, 소요 시간={train_time:.2f}초"
        )

        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        모델 예측 수행

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            예측 점수
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 평가 모드로 설정
        self.eval()

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 예측 수행
        with torch.no_grad():
            predictions = self(X_tensor)

        # NumPy 배열로 변환
        predictions_np = predictions.cpu().numpy().flatten()

        return predictions_np

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 평가

        Args:
            X: 특성 벡터
            y: 실제 타겟
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 예측 수행
        y_pred = self.predict(X)

        # 평가 메트릭 계산
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        logger.info(
            f"Transformer 모델 평가: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
        )

        return {"rmse": rmse, "mae": mae, "r2": r2, "model_type": self.model_name}

    def save(self, path: str) -> bool:
        """
        모델 저장

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        try:
            # 디렉토리 확인
            self._ensure_directory(path)

            # 저장할 데이터
            save_dict = {
                "model_state": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "output_dim": self.output_dim,
                    "learning_rate": self.learning_rate,
                },
                "metadata": self.metadata,
                "is_trained": self.is_trained,
            }

            # 모델 저장
            torch.save(save_dict, path)

            logger.info(f"Transformer 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"Transformer 모델 저장 중 오류: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        모델 로드

        Args:
            path: 모델 경로

        Returns:
            성공 여부
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(path):
                raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {path}")

            # 모델 로드
            checkpoint = torch.load(path, map_location=self.device)

            # 설정 업데이트
            config = checkpoint.get("config", {})
            self.input_dim = config.get("input_dim", self.input_dim)
            self.d_model = config.get("d_model", self.d_model)
            self.nhead = config.get("nhead", self.nhead)
            self.num_layers = config.get("num_layers", self.num_layers)
            self.dim_feedforward = config.get("dim_feedforward", self.dim_feedforward)
            self.dropout = config.get("dropout", self.dropout)
            self.output_dim = config.get("output_dim", self.output_dim)
            self.learning_rate = config.get("learning_rate", self.learning_rate)

            # 모델 재구성
            self._build_model()

            # 모델 가중치 로드
            self.load_state_dict(checkpoint["model_state"])

            # 메타데이터 로드
            self.metadata = checkpoint.get("metadata", {})
            self.is_trained = checkpoint.get("is_trained", False)

            logger.info(f"Transformer 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"Transformer 모델 로드 중 오류: {e}")
            return False


# math 모듈 임포트 추가
import math


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
