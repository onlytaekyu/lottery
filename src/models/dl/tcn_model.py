"""
TCN (Temporal Convolutional Network) 모델

이 모듈은 시계열 로또 번호 패턴 분석을 위한 정교한 TCN 모델을 구현합니다.
Dilated convolution과 residual connection을 활용한 시계열 예측 모델입니다.
"""

# 1. 표준 라이브러리
import os
import json
import time
from typing import Any, Dict, Optional, List, Tuple, Union

# 2. 서드파티
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 3. 프로젝트 내부
from ..base_model import ModelWithAMP
from ...utils.unified_logging import get_logger

logger = get_logger(__name__)


class Chomp1d(nn.Module):
    """
    Causal convolution을 위한 패딩 제거 모듈

    TCN에서 미래 정보 누출을 방지하기 위해 오른쪽 패딩을 제거합니다.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파: 오른쪽 패딩 제거

        Args:
            x: 입력 텐서 (batch, channels, seq_len)

        Returns:
            패딩이 제거된 텐서
        """
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN의 핵심 Temporal Block

    Dilated convolution + Weight normalization + Residual connection을 구현합니다.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        use_weight_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        # 첫 번째 dilated convolution
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if use_weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)

        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        # 두 번째 dilated convolution
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if use_weight_norm:
            self.conv2 = nn.utils.weight_norm(self.conv2)

        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        # 활성화 함수 설정
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        # Residual connection을 위한 차원 맞춤
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        # 네트워크 구성
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.activation,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.activation,
            self.dropout2,
        )

        self.init_weights()

    def init_weights(self):
        """가중치 초기화"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서

        Returns:
            출력 텐서 (residual connection 적용됨)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TCNNetwork(nn.Module):
    """
    정교한 TCN 네트워크 구현

    여러 개의 TemporalBlock을 쌓아서 긴 시계열 의존성을 학습합니다.
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
        use_skip_connections: bool = True,
        use_weight_norm: bool = True,
        output_dim: int = 45,
    ):
        """
        TCN 네트워크 초기화

        Args:
            input_dim: 입력 차원
            num_channels: 각 레이어의 채널 수 리스트
            kernel_size: 컨볼루션 커널 크기
            dropout: 드롭아웃 비율
            activation: 활성화 함수
            use_skip_connections: Skip connection 사용 여부
            use_weight_norm: Weight normalization 사용 여부
            output_dim: 출력 차원 (로또 번호 개수)
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    activation=activation,
                )
            )

        self.network = nn.Sequential(*layers)

        # 출력 레이어
        self.output_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(num_channels[-1], output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid(),  # 확률 출력
        )

        # Skip connections를 위한 추가 레이어
        self.use_skip_connections = use_skip_connections
        if use_skip_connections:
            self.skip_projection = nn.Conv1d(input_dim, num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 (batch, seq_len, input_dim) 또는 (batch, input_dim, seq_len)

        Returns:
            출력 텐서 (batch, output_dim)
        """
        # 입력 차원 조정
        if x.dim() == 2:  # (batch, seq_len) -> (batch, 1, seq_len)
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(2) > x.size(
            1
        ):  # (batch, seq_len, features) -> (batch, features, seq_len)
            x = x.transpose(1, 2)

        # TCN 네트워크 통과
        tcn_out = self.network(x)

        # Skip connection 적용
        if self.use_skip_connections:
            skip_out = self.skip_projection(x)
            tcn_out = tcn_out + skip_out

        # 출력 프로젝션
        output = self.output_projection(tcn_out)

        return output

    def get_receptive_field(self, kernel_size: int, num_levels: int) -> int:
        """
        TCN의 수용 영역 계산

        Args:
            kernel_size: 커널 크기
            num_levels: 레이어 수

        Returns:
            수용 영역 크기
        """
        return 1 + 2 * (kernel_size - 1) * (2**num_levels - 1)


class TCNModel(ModelWithAMP):
    """
    시계열 로또 번호 패턴 분석용 TCN 모델

    Temporal Convolutional Network를 사용하여 시계열 로또 번호 패턴을 학습하고
    미래 번호 출현 확률을 예측합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        TCN 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)
        self.device = torch.device(
            "cuda"
            if config and config.get("use_gpu", False) and torch.cuda.is_available()
            else "cpu"
        )

        # 모델 설정
        self.config = config or {}
        tcn_config = self.config.get("tcn", {})

        # TCN 아키텍처 설정
        self.input_dim = tcn_config.get("input_dim", 45)
        self.num_channels = tcn_config.get("num_channels", [64, 128, 256, 128, 64])
        self.kernel_size = tcn_config.get("kernel_size", 3)
        self.dropout = tcn_config.get("dropout", 0.2)
        self.activation = tcn_config.get("activation", "relu")
        self.use_skip_connections = tcn_config.get("use_skip_connections", True)
        self.use_weight_norm = tcn_config.get("use_weight_norm", True)
        self.output_dim = tcn_config.get("output_dim", 45)

        # 학습 설정
        self.learning_rate = tcn_config.get("learning_rate", 0.001)
        self.weight_decay = tcn_config.get("weight_decay", 1e-5)
        self.batch_size = tcn_config.get("batch_size", 64)

        # 시계열 설정
        self.sequence_length = tcn_config.get("sequence_length", 50)
        self.prediction_horizon = tcn_config.get("prediction_horizon", 1)
        self.use_attention = tcn_config.get("use_attention", False)

        # 모델 이름
        self.model_name = "TCNModel"

        # 모델 구성
        self._build_model()

        logger.info(f"TCN 모델 초기화 완료: {self.model_name}")
        logger.info(
            f"아키텍처: channels={self.num_channels}, kernel_size={self.kernel_size}"
        )
        logger.info(
            f"시퀀스 길이: {self.sequence_length}, 수용 영역: {self._get_receptive_field()}"
        )

    def _build_model(self):
        """TCN 모델 구성"""
        self.model = TCNNetwork(
            input_dim=self.input_dim,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            activation=self.activation,
            use_skip_connections=self.use_skip_connections,
            use_weight_norm=self.use_weight_norm,
            output_dim=self.output_dim,
        ).to(self.device)

        # 옵티마이저 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # 손실 함수
        self.criterion = nn.BCELoss()

        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

    def _get_receptive_field(self) -> int:
        """수용 영역 계산"""
        return self.model.get_receptive_field(self.kernel_size, len(self.num_channels))

    def _prepare_sequences(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        시계열 데이터를 시퀀스로 변환

        Args:
            X: 입력 데이터 (samples, features) 또는 (samples, seq_len, features)
            y: 타겟 데이터 (선택적)

        Returns:
            시퀀스로 변환된 데이터
        """
        if X.ndim == 2:
            # (samples, features) -> (samples, seq_len, features)로 변환
            if X.shape[0] < self.sequence_length:
                logger.warning(
                    f"데이터 샘플 수({X.shape[0]})가 시퀀스 길이({self.sequence_length})보다 작습니다."
                )
                # 패딩 또는 반복으로 시퀀스 길이 맞춤
                X_padded = np.pad(
                    X, ((self.sequence_length - X.shape[0], 0), (0, 0)), mode="edge"
                )
                X_sequences = X_padded.reshape(1, self.sequence_length, -1)
            else:
                X_sequences = []
                for i in range(len(X) - self.sequence_length + 1):
                    X_sequences.append(X[i : i + self.sequence_length])
                X_sequences = np.array(X_sequences)
        else:
            X_sequences = X

        if y is not None:
            if y.ndim == 1:
                # 시퀀스에 맞춰 y도 조정
                if len(y) >= len(X_sequences):
                    y_sequences = y[: len(X_sequences)]
                else:
                    y_sequences = y
            else:
                y_sequences = y
            return X_sequences, y_sequences

        return X_sequences, None

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        batch_size: int = None,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        PyTorch DataLoader 생성

        Args:
            X: 입력 데이터
            y: 타겟 데이터
            batch_size: 배치 크기
            shuffle: 셔플 여부

        Returns:
            DataLoader 객체
        """
        if batch_size is None:
            batch_size = self.batch_size

        X_tensor = torch.FloatTensor(X)

        if y is not None:
            y_tensor = torch.FloatTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        한 에포크 훈련

        Args:
            train_loader: 훈련 데이터 로더

        Returns:
            평균 훈련 손실
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            if len(batch_data) == 2:
                inputs, targets = batch_data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            else:
                inputs = batch_data[0].to(self.device)
                targets = None

            # AMP를 사용한 훈련 스텝
            if targets is not None:
                loss = self.train_step_with_amp(
                    self.model, inputs, targets, self.optimizer, self.criterion
                )
                total_loss += loss
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """
        한 에포크 검증

        Args:
            val_loader: 검증 데이터 로더

        Returns:
            평균 검증 손실
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1

        return total_loss / max(num_batches, 1)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        완전한 TCN 모델 훈련 구현

        Args:
            X: 입력 특성 벡터
            y: 타겟 레이블
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과 및 메타데이터
        """
        logger.info(f"TCN 모델 훈련 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 파라미터 설정
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", self.batch_size)
        validation_split = kwargs.get("validation_split", 0.2)
        patience = kwargs.get("patience", 15)

        # 시계열 시퀀스 준비
        X_sequences, y_sequences = self._prepare_sequences(X, y)
        if y_sequences is not None:
            logger.info(
                f"시퀀스 변환 완료: X_seq 형태={X_sequences.shape}, y_seq 형태={y_sequences.shape}"
            )
        else:
            logger.info(f"시퀀스 변환 완료: X_seq 형태={X_sequences.shape}, y_seq=None")

        # 훈련/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_sequences, y_sequences, test_size=validation_split, random_state=42
        )

        # DataLoader 생성
        train_loader = self._create_dataloader(
            X_train, y_train, batch_size, shuffle=True
        )
        val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)

        # 훈련 루프
        best_val_loss = float("inf")
        patience_counter = 0
        training_history = []

        start_time = time.time()

        for epoch in range(epochs):
            # 훈련 단계
            train_loss = self._train_epoch(train_loader)

            # 검증 단계
            val_loss = self._validate_epoch(val_loader)

            # 학습률 스케줄링
            self.scheduler.step(val_loss)

            # 히스토리 저장
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            training_history.append(epoch_info)

            # 로깅
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # 조기 종료 검사
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최적 모델 상태 저장
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"조기 종료: {patience} 에포크 동안 개선되지 않음")
                break

        # 최적 모델 상태 복원
        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)

        train_time = time.time() - start_time

        # 훈련 히스토리 저장
        self.training_history = training_history

        # 메타데이터 업데이트
        self.metadata.update(
            {
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "sequence_length": self.sequence_length,
                "receptive_field": self._get_receptive_field(),
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "epochs_trained": epoch + 1,
                "train_time": train_time,
                "architecture": {
                    "num_channels": self.num_channels,
                    "kernel_size": self.kernel_size,
                    "dropout": self.dropout,
                    "activation": self.activation,
                },
            }
        )

        # 훈련 완료 표시
        self.is_trained = True

        logger.info(
            f"TCN 모델 훈련 완료: "
            f"최적 검증 손실={best_val_loss:.4f}, "
            f"훈련 시간={train_time:.2f}초, "
            f"에포크={epoch + 1}"
        )

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs": epoch + 1,
            "train_time": train_time,
            "is_trained": True,
            "model_type": "TCNModel",
            "training_history": training_history,
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        TCN 모델 예측 수행

        Args:
            X: 입력 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            예측 확률
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        logger.info(f"TCN 예측 수행: 입력 형태={X.shape}")

        # 시퀀스 준비
        X_sequences, _ = self._prepare_sequences(X)

        # 예측 수행
        self.model.eval()
        predictions = []

        with torch.no_grad():
            # 배치 단위로 예측
            batch_size = kwargs.get("batch_size", self.batch_size)
            for i in range(0, len(X_sequences), batch_size):
                batch_X = X_sequences[i : i + batch_size]
                batch_tensor = torch.FloatTensor(batch_X).to(self.device)

                batch_pred = self.model(batch_tensor)
                predictions.append(batch_pred.cpu().numpy())

        predictions = np.vstack(predictions)

        logger.info(f"예측 완료: 출력 형태={predictions.shape}")
        return predictions

    def predict_sequence(self, X: np.ndarray, future_steps: int = 1) -> np.ndarray:
        """
        미래 시퀀스 예측

        Args:
            X: 입력 시퀀스
            future_steps: 예측할 미래 스텝 수

        Returns:
            미래 예측값
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")

        self.model.eval()
        future_predictions = []

        # 현재 시퀀스로 시작
        current_sequence = X.copy()

        with torch.no_grad():
            for _ in range(future_steps):
                # 시퀀스 준비
                seq_input, _ = self._prepare_sequences(current_sequence)
                seq_tensor = torch.FloatTensor(seq_input[-1:]).to(
                    self.device
                )  # 마지막 시퀀스만 사용

                # 예측
                pred = self.model(seq_tensor)
                pred_np = pred.cpu().numpy()[0]

                future_predictions.append(pred_np)

                # 시퀀스 업데이트 (예측값을 다음 입력으로 사용)
                current_sequence = np.vstack([current_sequence[1:], pred_np])

        return np.array(future_predictions)

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        완전한 모델 평가 구현

        Args:
            X: 입력 특성 벡터
            y: 실제 타겟
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")

        logger.info(f"TCN 모델 평가 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 시퀀스 준비
        X_sequences, y_sequences = self._prepare_sequences(X, y)

        # 예측 수행
        y_pred = self.predict(X_sequences)

        # None 체크
        if y_sequences is None:
            raise ValueError(
                "타겟 데이터가 None입니다. 평가를 위해서는 실제 타겟 값이 필요합니다."
            )

        # 차원 맞춤
        if y_sequences.ndim != y_pred.ndim:
            if y_sequences.ndim == 1 and y_pred.ndim == 2:
                y_sequences = y_sequences.reshape(-1, 1)
            elif y_sequences.ndim == 2 and y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

        # 길이 맞춤
        min_len = min(len(y_sequences), len(y_pred))
        y_sequences = y_sequences[:min_len]
        y_pred = y_pred[:min_len]

        # 메트릭 계산
        try:
            rmse = np.sqrt(mean_squared_error(y_sequences, y_pred))
            mae = mean_absolute_error(y_sequences, y_pred)
            r2 = r2_score(y_sequences, y_pred)
            mape = self._calculate_mape(y_sequences, y_pred)

            # 분류 메트릭 (확률 예측인 경우)
            if np.all((y_pred >= 0) & (y_pred <= 1)):
                y_pred_binary = (y_pred > 0.5).astype(int)
                y_true_binary = (y_sequences > 0.5).astype(int)

                from sklearn.metrics import precision_score, recall_score, f1_score

                precision = precision_score(
                    y_true_binary, y_pred_binary, average="macro", zero_division=0
                )
                recall = recall_score(
                    y_true_binary, y_pred_binary, average="macro", zero_division=0
                )
                f1 = f1_score(
                    y_true_binary, y_pred_binary, average="macro", zero_division=0
                )
            else:
                precision = recall = f1 = None

        except Exception as e:
            logger.error(f"메트릭 계산 중 오류: {e}")
            rmse = mae = r2 = mape = precision = recall = f1 = None

        results = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "model_type": "TCNModel",
            "test_samples": len(y_sequences),
            "sequence_length": self.sequence_length,
        }

        logger.info(f"평가 완료: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        return results

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        MAPE (Mean Absolute Percentage Error) 계산

        Args:
            y_true: 실제값
            y_pred: 예측값

        Returns:
            MAPE 값
        """
        try:
            # 0 값 처리
            mask = y_true != 0
            if not np.any(mask):
                return float("inf")

            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            return mape
        except:
            return float("inf")

    def save(self, path: str) -> bool:
        """
        완전한 모델 저장

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        try:
            self._ensure_directory(path)

            save_dict = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "num_channels": self.num_channels,
                    "kernel_size": self.kernel_size,
                    "dropout": self.dropout,
                    "activation": self.activation,
                    "use_skip_connections": self.use_skip_connections,
                    "use_weight_norm": self.use_weight_norm,
                    "output_dim": self.output_dim,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "sequence_length": self.sequence_length,
                    "prediction_horizon": self.prediction_horizon,
                    "use_attention": self.use_attention,
                },
                "metadata": self.metadata,
                "training_history": self.training_history,
                "is_trained": self.is_trained,
                "model_name": self.model_name,
            }

            torch.save(save_dict, path)
            logger.info(f"TCN 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"TCN 모델 저장 실패: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        완전한 모델 로드

        Args:
            path: 로드할 모델 경로

        Returns:
            성공 여부
        """
        try:
            if not os.path.exists(path):
                logger.error(f"모델 파일이 존재하지 않습니다: {path}")
                return False

            checkpoint = torch.load(path, map_location=self.device)

            # 설정 복원
            config = checkpoint.get("config", {})
            self.input_dim = config.get("input_dim", self.input_dim)
            self.num_channels = config.get("num_channels", self.num_channels)
            self.kernel_size = config.get("kernel_size", self.kernel_size)
            self.dropout = config.get("dropout", self.dropout)
            self.activation = config.get("activation", self.activation)
            self.use_skip_connections = config.get(
                "use_skip_connections", self.use_skip_connections
            )
            self.use_weight_norm = config.get("use_weight_norm", self.use_weight_norm)
            self.output_dim = config.get("output_dim", self.output_dim)
            self.learning_rate = config.get("learning_rate", self.learning_rate)
            self.weight_decay = config.get("weight_decay", self.weight_decay)
            self.batch_size = config.get("batch_size", self.batch_size)
            self.sequence_length = config.get("sequence_length", self.sequence_length)
            self.prediction_horizon = config.get(
                "prediction_horizon", self.prediction_horizon
            )
            self.use_attention = config.get("use_attention", self.use_attention)

            # 모델 재구성
            self._build_model()

            # 상태 복원
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            if "scheduler_state" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])

            # 메타데이터 복원
            self.metadata = checkpoint.get("metadata", self.metadata)
            self.training_history = checkpoint.get("training_history", [])
            self.is_trained = checkpoint.get("is_trained", False)
            self.model_name = checkpoint.get("model_name", self.model_name)

            self.model.to(self.device)

            logger.info(f"TCN 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"TCN 모델 로드 실패: {e}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """
        TCN의 시간대별 중요도 (Attention 가중치 기반)

        Returns:
            특성 중요도 딕셔너리
        """
        if not self.is_trained:
            return {"error": "모델이 학습되지 않았습니다."}

        # TCN의 경우 각 시간 스텝의 상대적 중요도를 계산
        importance = {}

        try:
            # 모델의 가중치를 분석하여 중요도 계산
            total_weights = 0
            layer_importance = {}

            for name, param in self.model.named_parameters():
                if "weight" in name and param.requires_grad:
                    weight_sum = torch.sum(torch.abs(param)).item()
                    layer_importance[name] = weight_sum
                    total_weights += weight_sum

            # 정규화
            for name, weight in layer_importance.items():
                importance[name] = weight / total_weights if total_weights > 0 else 0

        except Exception as e:
            logger.error(f"특성 중요도 계산 중 오류: {e}")
            importance = {"error": str(e)}

        return importance

    def visualize_predictions(
        self, X: np.ndarray, y: np.ndarray, save_path: str = None
    ):
        """
        예측 결과 시각화

        Args:
            X: 입력 데이터
            y: 실제 값
            save_path: 저장 경로 (선택적)
        """
        try:
            import matplotlib.pyplot as plt

            # 예측 수행
            y_pred = self.predict(X)

            # 시각화
            plt.figure(figsize=(15, 8))

            # 전체 예측 결과
            plt.subplot(2, 2, 1)
            plt.scatter(y, y_pred, alpha=0.6)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
            plt.xlabel("실제값")
            plt.ylabel("예측값")
            plt.title("실제값 vs 예측값")

            # 시계열 플롯 (처음 100개 샘플)
            plt.subplot(2, 2, 2)
            n_show = min(100, len(y))
            plt.plot(y[:n_show], label="실제값", linewidth=2)
            plt.plot(y_pred[:n_show], label="예측값", linewidth=2)
            plt.xlabel("시간")
            plt.ylabel("값")
            plt.title("시계열 예측 결과")
            plt.legend()

            # 오차 분포
            plt.subplot(2, 2, 3)
            errors = y - y_pred.flatten()
            plt.hist(errors, bins=50, alpha=0.7)
            plt.xlabel("예측 오차")
            plt.ylabel("빈도")
            plt.title("예측 오차 분포")

            # 훈련 히스토리
            if self.training_history:
                plt.subplot(2, 2, 4)
                epochs = [h["epoch"] for h in self.training_history]
                train_losses = [h["train_loss"] for h in self.training_history]
                val_losses = [h["val_loss"] for h in self.training_history]

                plt.plot(epochs, train_losses, label="훈련 손실")
                plt.plot(epochs, val_losses, label="검증 손실")
                plt.xlabel("에포크")
                plt.ylabel("손실")
                plt.title("훈련 히스토리")
                plt.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"시각화 저장 완료: {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib을 설치해야 시각화를 사용할 수 있습니다.")
        except Exception as e:
            logger.error(f"시각화 중 오류: {e}")

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        불확실성을 포함한 예측

        Args:
            X: 입력 데이터
            n_samples: 샘플링 횟수

        Returns:
            (평균 예측값, 예측 불확실성) 튜플
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")

        # Dropout을 활성화하여 여러 번 예측
        self.model.train()  # Dropout 활성화

        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.predict(X)
                predictions.append(pred)

        predictions = np.array(predictions)

        # 평균과 표준편차 계산
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        self.model.eval()  # 다시 평가 모드로

        return mean_pred, std_pred
