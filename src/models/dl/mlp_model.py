"""
MLP 딥러닝 모델

이 모듈은 다층 퍼셉트론(Multi-Layer Perceptron) 기반 로또 번호 예측 모델을 구현합니다.
특성 벡터를 입력으로 받아 점수를 예측하는 기본적인 딥러닝 모델입니다.
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
from ...utils.error_handler import get_logger

logger = get_logger(__name__)


class MLPNetwork(nn.Module):
    """
    다층 퍼셉트론 신경망

    여러 개의 완전 연결 레이어로 구성된 기본적인 신경망입니다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.3,
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        """
        MLP 네트워크 초기화

        Args:
            input_dim: 입력 차원
            hidden_dims: 은닉층 차원 목록
            output_dim: 출력 차원
            dropout_rate: 드롭아웃 비율
            activation: 활성화 함수 ('relu', 'leaky_relu', 'elu')
            batch_norm: 배치 정규화 사용 여부
        """
        super().__init__()

        # 활성화 함수 설정
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
            logger.warning(f"알 수 없는 활성화 함수: {activation}, ReLU로 대체합니다.")

        # 레이어 구성
        layers = []

        # 입력층 -> 첫 번째 은닉층
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout_rate))

        # 은닉층 간 연결
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))

        # 마지막 은닉층 -> 출력층
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # 순차 모델 생성
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서

        Returns:
            출력 텐서
        """
        return self.network(x)


class MLPModel(ModelWithAMP):
    """
    MLP 기반 로또 번호 예측 모델

    다층 퍼셉트론을 사용하여 로또 번호의 당첨 확률을 예측합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        MLP 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 설정
        self.config = config or {}
        mlp_config = self.config.get("mlp", {})

        # 모델 하이퍼파라미터
        self.input_dim = mlp_config.get("input_dim", 70)
        self.hidden_dims = mlp_config.get("hidden_dims", [128, 64, 32])
        self.output_dim = mlp_config.get("output_dim", 1)
        self.dropout_rate = mlp_config.get("dropout_rate", 0.3)
        self.activation = mlp_config.get("activation", "relu")
        self.batch_norm = mlp_config.get("batch_norm", True)

        # 학습 하이퍼파라미터
        self.learning_rate = mlp_config.get("learning_rate", 0.001)
        self.weight_decay = mlp_config.get("weight_decay", 1e-5)

        # 모델 이름
        self.model_name = "MLPModel"

        # 모델 구성
        self._build_model()

        logger.info(
            f"MLP 모델 초기화 완료: 입력 차원={self.input_dim}, "
            f"은닉층 차원={self.hidden_dims}, 출력 차원={self.output_dim}, "
            f"활성화 함수={self.activation}, 배치 정규화={self.batch_norm}"
        )

    def _build_model(self):
        """
        모델 구성
        """
        # MLP 네트워크
        self.model = MLPNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            batch_norm=self.batch_norm,
        ).to(self.device)

        # 손실 함수
        self.criterion = nn.MSELoss()

        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 훈련

        Args:
            X: 특성 벡터
            y: 타겟 값
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"MLP 모델 훈련 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 훈련 매개변수
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 64)
        validation_split = kwargs.get("validation_split", 0.2)
        patience = kwargs.get("patience", 10)

        # 훈련/검증 세트 분할
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # PyTorch 텐서로 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # 데이터 로더 생성
        from torch.utils.data import TensorDataset, DataLoader

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 훈련 지표
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []

        # 훈련 시작 시간
        start_time = time.time()

        # 훈련 루프
        for epoch in range(epochs):
            # 훈련 모드로 설정
            self.model.train()

            # 에포크 훈련 손실
            epoch_train_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in train_loader:
                # AMP를 사용한 훈련 단계
                loss = self.train_step_with_amp(
                    model=self.model,
                    inputs=batch_X,
                    targets=batch_y,
                    optimizer=self.optimizer,
                    loss_fn=self.criterion,
                )

                # 손실 누적
                epoch_train_loss += loss
                num_batches += 1

            # 평균 훈련 손실 계산
            epoch_train_loss /= num_batches
            train_losses.append(epoch_train_loss)

            # 검증 손실 계산
            self.model.eval()
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        val_predictions = self.model(X_val_tensor)
                        val_loss = self.criterion(val_predictions, y_val_tensor).item()
                else:
                    val_predictions = self.model(X_val_tensor)
                    val_loss = self.criterion(val_predictions, y_val_tensor).item()

            val_losses.append(val_loss)

            # 로깅
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"에포크 {epoch+1}/{epochs}: 훈련 손실={epoch_train_loss:.6f}, "
                    f"검증 손실={val_loss:.6f}"
                )

            # 조기 종료 확인
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # 최상의 모델 저장
                best_model_state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"조기 종료: 에포크 {epoch+1}")
                    break

        # 최상의 모델 복원
        self.model.load_state_dict(best_model_state["model"])
        self.optimizer.load_state_dict(best_model_state["optimizer"])

        # 훈련 시간 계산
        train_time = time.time() - start_time

        # 훈련 메트릭 저장
        self.training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_model_state["epoch"],
        }

        # 훈련 완료 표시
        self.is_trained = True

        # 메타데이터 업데이트
        self.metadata.update(
            {
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "best_val_loss": best_val_loss,
                "best_epoch": best_model_state["epoch"],
                "epochs_trained": epoch + 1,
                "train_time": train_time,
            }
        )

        logger.info(
            f"MLP 모델 훈련 완료: 최상의 검증 손실={best_val_loss:.6f} (에포크 {best_model_state['epoch']+1}), "
            f"소요 시간={train_time:.2f}초"
        )

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_model_state["epoch"],
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
            예측값
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 평가 모드로 설정
        self.model.eval()

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 예측 수행
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(X_tensor)
            else:
                predictions = self.model(X_tensor)

        # NumPy 배열로 변환
        predictions_np = predictions.cpu().numpy()

        # 차원 조정 (필요한 경우)
        if predictions_np.shape[1] == 1:
            predictions_np = predictions_np.flatten()

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

        logger.info(f"MLP 모델 평가: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "model_type": self.model_name,
        }

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
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dims": self.hidden_dims,
                    "output_dim": self.output_dim,
                    "dropout_rate": self.dropout_rate,
                    "activation": self.activation,
                    "batch_norm": self.batch_norm,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
                "metadata": self.metadata,
                "training_history": self.training_history,
                "is_trained": self.is_trained,
            }

            # 모델 저장
            torch.save(save_dict, path)

            logger.info(f"MLP 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"MLP 모델 저장 중 오류: {e}")
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
            self.hidden_dims = config.get("hidden_dims", self.hidden_dims)
            self.output_dim = config.get("output_dim", self.output_dim)
            self.dropout_rate = config.get("dropout_rate", self.dropout_rate)
            self.activation = config.get("activation", self.activation)
            self.batch_norm = config.get("batch_norm", self.batch_norm)
            self.learning_rate = config.get("learning_rate", self.learning_rate)
            self.weight_decay = config.get("weight_decay", self.weight_decay)

            # 모델 재구성
            self._build_model()

            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint["model_state"])

            # 옵티마이저 상태 로드
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            # 메타데이터 로드
            self.metadata = checkpoint.get("metadata", {})
            self.training_history = checkpoint.get("training_history", {})
            self.is_trained = checkpoint.get("is_trained", False)

            logger.info(f"MLP 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"MLP 모델 로드 중 오류: {e}")
            return False


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
