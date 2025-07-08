"""
TCN (Temporal Convolutional Network) 모델

이 모듈은 시계열 로또 번호 패턴 분석을 위한 정교한 TCN 모델을 구현합니다.
Dilated convolution과 residual connection을 활용한 시계열 예측 모델입니다.
"""

# 1. 표준 라이브러리
import os
import time
from typing import Any, Dict, Optional, List, Tuple
import asyncio
from dataclasses import dataclass

# 2. 서드파티
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 3. 프로젝트 내부
from ..base_model import ModelWithAMP
from ...utils.unified_logging import get_logger
from ...utils.unified_memory_manager import get_unified_memory_manager
from ...utils.cuda_optimizers import get_cuda_optimizer
from ...utils.enhanced_process_pool import get_enhanced_process_pool
from ...utils.unified_async_manager import get_unified_async_manager
from ...utils.cache_manager import CacheManager

# TensorRT 지원 (선택적)
try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class TCNOptimizationConfig:
    """TCN 최적화 설정"""
    # TensorRT 설정
    enable_tensorrt: bool = True
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    tensorrt_workspace_size: int = 1 << 30  # 1GB
    
    # 메모리 관리
    enable_memory_optimization: bool = True
    memory_pool_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    
    # 캐시 설정
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1시간
    max_cache_size: int = 1000
    
    # 비동기 처리
    enable_async: bool = True
    async_batch_size: int = 32
    
    # 성능 모니터링
    enable_monitoring: bool = True
    profiling_enabled: bool = True


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

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """
        TCN 모델 초기화 (통합 최적화)

        Args:
            config: 모델 설정
        """
        super().__init__(config)
        
        # 모델 설정
        self.config = config or {}
        tcn_config = self.config.get("tcn", {})

        # 최적화 설정
        self.opt_config = TCNOptimizationConfig()
        if "tcn_optimization" in self.config:
            opt_params = self.config["tcn_optimization"]
            for key, value in opt_params.items():
                if hasattr(self.opt_config, key):
                    setattr(self.opt_config, key, value)

        # 통합 시스템 초기화
        self.memory_manager = get_unified_memory_manager()
        self.cuda_optimizer = get_cuda_optimizer()
        self.process_pool = get_enhanced_process_pool()
        self.async_manager = get_unified_async_manager()

        # 캐시 시스템 초기화
        self.cache_manager = cache_manager

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
        self.base_batch_size = tcn_config.get("batch_size", 64)
        self.batch_size = self.base_batch_size

        # 시계열 설정
        self.sequence_length = tcn_config.get("sequence_length", 50)
        self.prediction_horizon = tcn_config.get("prediction_horizon", 1)
        self.use_attention = tcn_config.get("use_attention", False)

        # GPU 최적화 설정
        self.use_data_parallel = tcn_config.get("use_data_parallel", False)
        self.adaptive_batch_size = tcn_config.get("adaptive_batch_size", True)
        self.max_memory_usage = tcn_config.get("max_memory_usage", 0.8)

        # 모델 이름
        self.model_name = "TCNModel_v2"

        # TensorRT 설정
        self.tensorrt_model = None
        self.use_tensorrt = (
            self.opt_config.enable_tensorrt and 
            TENSORRT_AVAILABLE and 
            self.cuda_optimizer.is_available()
        )

        # 성능 통계
        self.performance_stats = {
            "total_fits": 0,
            "total_predictions": 0,
            "cache_hits": 0,
            "tensorrt_accelerated": 0,
            "avg_fit_time": 0.0,
            "avg_predict_time": 0.0,
            "memory_efficiency": 0.0
        }

        # 모델 구성
        self._build_model()

        logger.info(f"TCN 모델 v2.0 초기화 완료: {self.model_name}")
        logger.info(f"장치: {self.device} (GPU 사용 가능: {self.device_manager.gpu_available})")
        logger.info(f"아키텍처: channels={self.num_channels}, kernel_size={self.kernel_size}")
        logger.info(f"시퀀스 길이: {self.sequence_length}, 수용 영역: {self._get_receptive_field()}")
        logger.info(f"배치 크기: {self.batch_size} (적응형: {self.adaptive_batch_size})")
        logger.info(f"최적화 설정: TensorRT={self.use_tensorrt}, 캐시={self.opt_config.enable_cache}")

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

        # DataParallel 적용 (다중 GPU 사용 시)
        if self.use_data_parallel and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"DataParallel 적용: {torch.cuda.device_count()}개 GPU 사용")

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

    def _adjust_batch_size(self, current_memory_usage: float) -> int:
        """
        GPU 메모리 사용량에 따른 배치 크기 동적 조정

        Args:
            current_memory_usage: 현재 메모리 사용률 (0-1)

        Returns:
            조정된 배치 크기
        """
        if not self.adaptive_batch_size or not self.device_manager.gpu_available:
            return self.batch_size

        if current_memory_usage > self.max_memory_usage:
            # 메모리 사용량이 높으면 배치 크기 감소
            new_batch_size = max(8, self.batch_size // 2)
            logger.info(f"메모리 사용량 높음 ({current_memory_usage:.1%}), 배치 크기 감소: {self.batch_size} -> {new_batch_size}")
            self.batch_size = new_batch_size
        elif current_memory_usage < 0.5 and self.batch_size < self.base_batch_size:
            # 메모리 여유가 있으면 배치 크기 증가
            new_batch_size = min(self.base_batch_size, self.batch_size * 2)
            logger.info(f"메모리 여유 있음 ({current_memory_usage:.1%}), 배치 크기 증가: {self.batch_size} -> {new_batch_size}")
            self.batch_size = new_batch_size

        return self.batch_size

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
        PyTorch DataLoader 생성 (GPU 최적화)

        Args:
            X: 입력 데이터
            y: 타겟 데이터
            batch_size: 배치 크기
            shuffle: 셔플 여부

        Returns:
            DataLoader 객체
        """
        if batch_size is None:
            # GPU 메모리 사용량 확인 후 배치 크기 조정
            memory_info = self.device_manager.check_memory_usage()
            if memory_info.get("gpu_available", False):
                current_usage = memory_info.get("usage_percent", 0) / 100
                batch_size = self._adjust_batch_size(current_usage)
            else:
                batch_size = self.batch_size

        # 데이터를 tensor로 변환 (GPU로 이동은 훈련 시에 수행)
        X_tensor = torch.FloatTensor(X)

        if y is not None:
            y_tensor = torch.FloatTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        # DataLoader 설정 (GPU 사용 시 pin_memory=True)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=self.device_manager.gpu_available,
            num_workers=2 if self.device_manager.gpu_available else 0
        )

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        한 에포크 훈련 (GPU 최적화)

        Args:
            train_loader: 훈련 데이터 로더

        Returns:
            평균 훈련 손실
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            try:
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                    # GPU로 데이터 이동 (non_blocking=True로 최적화)
                    inputs = self.device_manager.to_device(inputs)
                    targets = self.device_manager.to_device(targets)
                else:
                    inputs = self.device_manager.to_device(batch_data[0])
                    targets = None

                # AMP를 사용한 훈련 스텝
                if targets is not None:
                    loss = self.train_step_with_amp(
                        self.model, inputs, targets, self.optimizer, self.criterion
                    )
                    total_loss += loss
                    num_batches += 1

                # 주기적으로 GPU 메모리 상태 확인
                if batch_idx % 50 == 0 and self.device_manager.gpu_available:
                    memory_info = self.device_manager.check_memory_usage()
                    if memory_info.get("usage_percent", 0) > 90:
                        logger.warning(f"GPU 메모리 사용량 높음: {memory_info.get('usage_percent', 0):.1f}%")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"배치 {batch_idx}에서 GPU 메모리 부족, 캐시 정리 후 배치 크기 감소")
                    self.device_manager.clear_cache()
                    # 배치 크기 감소
                    self.batch_size = max(8, self.batch_size // 2)
                    logger.info(f"배치 크기 감소: {self.batch_size}")
                    continue
                else:
                    raise

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

    # ========================================
    # 새로운 통합 최적화 메서드들 (v2.0)
    # ========================================

    async def predict_async(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        비동기 예측 메서드

        Args:
            X: 입력 데이터
            **kwargs: 추가 매개변수

        Returns:
            예측 결과
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")

        logger.info(f"비동기 예측 시작: 입력 형태={X.shape}")

        # 캐시 확인
        cache_key = None
        if self.cache_manager:
            cache_key = self._generate_cache_key(X, "predict_async")
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info("캐시에서 비동기 예측 결과 로드")
                self.performance_stats["cache_hits"] += 1
                return cached_result

        # 시퀀스 준비
        X_sequences, _ = self._prepare_sequences(X)

        # 비동기 예측 수행
        predictions = await self._async_predict_internal(X_sequences)

        # 캐시 저장
        if self.cache_manager and cache_key:
            self.cache_manager.set(cache_key, predictions)

        logger.info("비동기 예측 완료")
        return predictions

    async def _async_predict_internal(self, X_sequences: np.ndarray) -> np.ndarray:
        """내부 비동기 예측 구현"""
        chunk_size = self.opt_config.async_batch_size
        chunks = [X_sequences[i:i + chunk_size] for i in range(0, len(X_sequences), chunk_size)]
        
        tasks = []
        for i, chunk in enumerate(chunks):
            task = self.async_manager.create_task(
                self._predict_chunk_async(chunk, i)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return np.vstack(results)

    async def _predict_chunk_async(self, chunk: np.ndarray, chunk_idx: int) -> np.ndarray:
        """청크 비동기 예측"""
        logger.debug(f"TCN 청크 {chunk_idx} 예측 시작")
        
        with self.memory_manager.optimize_context():
            model_to_use = self.tensorrt_model if self.use_tensorrt and self.tensorrt_model else self.model
            
            model_to_use.eval()
            with torch.no_grad():
                chunk_tensor = torch.FloatTensor(chunk).to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = model_to_use(chunk_tensor)
                else:
                    predictions = model_to_use(chunk_tensor)
                
                result = predictions.cpu().numpy()
        
        logger.debug(f"TCN 청크 {chunk_idx} 예측 완료")
        return result

    def _generate_cache_key(self, X: np.ndarray, operation: str) -> str:
        """캐시 키 생성"""
        import hashlib
        
        # 데이터 해시
        data_hash = hashlib.md5(X.tobytes()).hexdigest()[:16]
        
        # 설정 해시
        config_str = f"{self.num_channels}_{self.kernel_size}_{self.sequence_length}_{operation}_{self.use_tensorrt}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"tcn_{data_hash}_{config_hash}"

    def optimize_with_tensorrt(self, sample_input: Optional[torch.Tensor] = None) -> bool:
        """
        TensorRT 최적화 수행

        Args:
            sample_input: 샘플 입력 텐서 (None이면 자동 생성)

        Returns:
            최적화 성공 여부
        """
        if not self.use_tensorrt or not TENSORRT_AVAILABLE:
            logger.warning("TensorRT 최적화가 비활성화되어 있거나 사용할 수 없습니다.")
            return False

        if not self.is_trained:
            logger.warning("모델이 학습되지 않았습니다. 학습 후 TensorRT 최적화를 수행하세요.")
            return False

        try:
            logger.info("TensorRT 최적화 시작")
            
            # 샘플 입력 생성 (없는 경우)
            if sample_input is None:
                sample_input = torch.randn(1, self.sequence_length, self.input_dim).to(self.device)
            
            # 모델을 평가 모드로 전환
            self.model.eval()
            
            # TensorRT 변환 설정
            if self.opt_config.tensorrt_precision == "fp16":
                precision = torch.half
            elif self.opt_config.tensorrt_precision == "int8":
                precision = torch.int8
            else:
                precision = torch.float32
            
            # TensorRT 모델 생성
            self.tensorrt_model = torch_tensorrt.compile(
                self.model,
                inputs=[torch_tensorrt.Input(
                    min_shape=sample_input.shape,
                    opt_shape=sample_input.shape,
                    max_shape=sample_input.shape,
                    dtype=precision
                )],
                enabled_precisions={precision},
                workspace_size=self.opt_config.tensorrt_workspace_size
            )
            
            # 테스트 실행
            with torch.no_grad():
                test_output = self.tensorrt_model(sample_input)
                logger.info(f"TensorRT 테스트 실행 성공: 출력 형태={test_output.shape}")
            
            logger.info("TensorRT 최적화 완료")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT 최적화 실패: {str(e)}")
            self.use_tensorrt = False
            self.tensorrt_model = None
            return False

    def predict_optimized(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        최적화된 예측 수행 (캐시 + TensorRT + 메모리 최적화)

        Args:
            X: 입력 데이터
            **kwargs: 추가 매개변수

        Returns:
            예측 결과
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")

        logger.info(f"최적화된 예측 시작: 입력 형태={X.shape}")

        # 캐시 확인
        cache_key = None
        if self.cache_manager:
            cache_key = self._generate_cache_key(X, "predict_optimized")
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info("캐시에서 최적화된 예측 결과 로드")
                self.performance_stats["cache_hits"] += 1
                return cached_result

        # 메모리 최적화된 예측
        with self.memory_manager.optimize_context():
            start_time = time.time()
            
            # 시퀀스 준비
            X_sequences, _ = self._prepare_sequences(X)
            
            # 최적화된 모델 선택
            model_to_use = self.tensorrt_model if self.use_tensorrt and self.tensorrt_model else self.model
            
            # 예측 수행
            predictions = []
            model_to_use.eval()
            
            with torch.no_grad():
                # 배치 크기 조정 (메모리 사용량 고려)
                memory_info = self.device_manager.check_memory_usage()
                if memory_info.get("gpu_available", False):
                    current_usage = memory_info.get("usage_percent", 0) / 100
                    batch_size = self._adjust_batch_size(current_usage)
                else:
                    batch_size = self.batch_size
                
                for i in range(0, len(X_sequences), batch_size):
                    batch_X = X_sequences[i:i + batch_size]
                    batch_tensor = torch.FloatTensor(batch_X).to(self.device)
                    
                    # AMP 사용 여부에 따라 예측 수행
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            batch_pred = model_to_use(batch_tensor)
                    else:
                        batch_pred = model_to_use(batch_tensor)
                    
                    predictions.append(batch_pred.cpu().numpy())
                
                # 결과 합치기
                predictions = np.vstack(predictions)
            
            predict_time = time.time() - start_time

        # 성능 통계 업데이트
        self.performance_stats["total_predictions"] += 1
        self.performance_stats["avg_predict_time"] = (
            (self.performance_stats["avg_predict_time"] * (self.performance_stats["total_predictions"] - 1) + predict_time) /
            self.performance_stats["total_predictions"]
        )

        # TensorRT 사용 시 통계 업데이트
        if self.use_tensorrt and self.tensorrt_model:
            self.performance_stats["tensorrt_accelerated"] += 1

        # 캐시 저장
        if self.cache_manager and cache_key:
            self.cache_manager.set(cache_key, predictions)

        logger.info(f"최적화된 예측 완료: 출력 형태={predictions.shape}, 소요 시간={predict_time:.3f}초")
        return predictions

    def get_performance_report(self) -> Dict[str, Any]:
        """성능 보고서 생성"""
        return {
            "model_name": self.model_name,
            "architecture": {
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "sequence_length": self.sequence_length,
                "output_dim": self.output_dim,
                "receptive_field": self._get_receptive_field()
            },
            "optimization_config": {
                "tensorrt_enabled": self.use_tensorrt,
                "tensorrt_model_loaded": self.tensorrt_model is not None,
                "cache_enabled": self.opt_config.enable_cache,
                "async_enabled": self.opt_config.enable_async,
                "memory_optimization": self.opt_config.enable_memory_optimization,
                "amp_enabled": self.use_amp
            },
            "performance_stats": self.performance_stats,
            "device_info": {
                "device": str(self.device),
                "gpu_available": self.device_manager.gpu_available,
                "data_parallel": self.use_data_parallel
            },
            "memory_info": self.memory_manager.get_memory_info() if self.memory_manager else {},
            "cuda_info": self.cuda_optimizer.get_device_info() if self.cuda_optimizer else {}
        }

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 최적화"""
        logger.info("메모리 사용량 최적화 시작")
        
        optimization_results = {
            "before": {},
            "after": {},
            "optimizations_applied": []
        }
        
        # 현재 메모리 사용량 체크
        if self.memory_manager:
            optimization_results["before"] = self.memory_manager.get_memory_info()
        
        optimizations_applied = []
        
        # 1. 모델 가중치 최적화
        if self.model:
            try:
                # 사용하지 않는 버퍼 정리
                torch.cuda.empty_cache()
                optimizations_applied.append("cuda_cache_clear")
                
                # 그래디언트 정리
                self.model.zero_grad()
                optimizations_applied.append("gradient_clear")
                
            except Exception as e:
                logger.warning(f"모델 최적화 중 오류: {e}")
        
        # 2. 배치 크기 조정
        if self.adaptive_batch_size:
            memory_info = self.device_manager.check_memory_usage()
            if memory_info.get("gpu_available", False):
                current_usage = memory_info.get("usage_percent", 0) / 100
                new_batch_size = self._adjust_batch_size(current_usage)
                if new_batch_size != self.batch_size:
                    optimizations_applied.append(f"batch_size_adjusted_{self.batch_size}_to_{new_batch_size}")
        
        # 3. 캐시 정리
        if self.cache_manager:
            cache_info = self.cache_manager.get_cache_info()
            if cache_info.get("memory_usage", 0) > 100 * 1024 * 1024:  # 100MB 이상
                self.cache_manager.clear_cache()
                optimizations_applied.append("cache_cleared")
        
        # 최적화 후 메모리 사용량 체크
        if self.memory_manager:
            optimization_results["after"] = self.memory_manager.get_memory_info()
        
        optimization_results["optimizations_applied"] = optimizations_applied
        
        logger.info(f"메모리 최적화 완료: {len(optimizations_applied)}개 최적화 적용")
        return optimization_results

    def benchmark_performance(self, X: np.ndarray, n_runs: int = 10) -> Dict[str, Any]:
        """성능 벤치마크 수행"""
        logger.info(f"성능 벤치마크 시작: {n_runs}회 실행")
        
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        benchmark_results = {
            "standard_predict": [],
            "optimized_predict": [],
            "async_predict": [],
            "tensorrt_predict": []
        }
        
        # 1. 표준 예측 벤치마크
        for i in range(n_runs):
            start_time = time.time()
            _ = self.predict(X)
            benchmark_results["standard_predict"].append(time.time() - start_time)
        
        # 2. 최적화된 예측 벤치마크
        for i in range(n_runs):
            start_time = time.time()
            _ = self.predict_optimized(X)
            benchmark_results["optimized_predict"].append(time.time() - start_time)
        
        # 3. 비동기 예측 벤치마크
        if self.opt_config.enable_async:
            for i in range(n_runs):
                start_time = time.time()
                _ = asyncio.run(self.predict_async(X))
                benchmark_results["async_predict"].append(time.time() - start_time)
        
        # 4. TensorRT 예측 벤치마크 (사용 가능한 경우)
        if self.use_tensorrt and self.tensorrt_model:
            for i in range(n_runs):
                start_time = time.time()
                _ = self.predict_optimized(X)  # TensorRT 사용
                benchmark_results["tensorrt_predict"].append(time.time() - start_time)
        
        # 결과 요약
        summary = {}
        for method, times in benchmark_results.items():
            if times:
                summary[method] = {
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times)
                }
        
        logger.info("성능 벤치마크 완료")
        return {
            "benchmark_results": benchmark_results,
            "summary": summary,
            "input_shape": X.shape,
            "n_runs": n_runs
        }
