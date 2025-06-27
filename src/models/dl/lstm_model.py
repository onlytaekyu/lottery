"""
LSTM 모델

이 모듈은 로또 번호 예측을 위한 LSTM 기반 모델을 구현합니다.
과거 당첨 번호의 시퀀스 패턴을 학습하여 향후 등장 가능성이 높은 번호를 예측합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
import time
from collections import deque
import gc
import random

from ..base_model import ModelWithAMP
from ...utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class LSTMNetwork(nn.Module):
    """LSTM 기반 로또 번호 예측 네트워크"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pattern_feature_dim: int = 0,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        LSTM 네트워크 초기화

        Args:
            input_dim: 입력 차원
            hidden_dim: 은닉층 차원
            output_dim: 출력 차원
            pattern_feature_dim: 패턴 특성 벡터 차원 (0이면 사용하지 않음)
            num_layers: LSTM 레이어 수
            dropout: 드롭아웃 비율
            bidirectional: 양방향 LSTM 사용 여부
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pattern_feature_dim = pattern_feature_dim
        self.use_pattern_features = pattern_feature_dim > 0

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # 드롭아웃
        self.dropout = nn.Dropout(dropout)

        # 패턴 특성 처리를 위한 별도의 레이어
        if self.use_pattern_features:
            self.pattern_fc = nn.Linear(pattern_feature_dim, hidden_dim)
            self.pattern_dropout = nn.Dropout(dropout)

        # 출력 레이어
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 패턴 특성을 사용하는 경우 출력 레이어 입력 차원 조정
        if self.use_pattern_features:
            self.fc = nn.Linear(lstm_out_dim + hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(lstm_out_dim, output_dim)

        # 활성화 함수
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, pattern_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 (batch_size, seq_len, input_dim)
            pattern_features: 패턴 특성 텐서 (batch_size, pattern_feature_dim)

        Returns:
            출력 텐서 (batch_size, output_dim)
        """
        # LSTM 출력
        # 정방향: h_t는 (batch_size, num_directions, hidden_dim)
        # 양방향: h_t는 (batch_size, 2, hidden_dim)
        lstm_out, (h_t, c_t) = self.lstm(x)

        # 마지막 타임스텝의 은닉 상태 사용
        if self.bidirectional:
            # 양방향인 경우 양쪽 방향의 은닉 상태 연결
            h_t = torch.cat((h_t[-2, :, :], h_t[-1, :, :]), dim=1)
        else:
            h_t = h_t[-1]

        # 드롭아웃 적용
        h_t = self.dropout(h_t)

        # 패턴 특성을 사용하는 경우
        if self.use_pattern_features and pattern_features is not None:
            # 패턴 특성 처리
            pattern_out = self.pattern_fc(pattern_features)
            pattern_out = F.relu(pattern_out)
            pattern_out = self.pattern_dropout(pattern_out)

            # LSTM 출력과 패턴 특성 결합
            combined = torch.cat([h_t, pattern_out], dim=1)

            # 최종 출력
            output = self.fc(combined)
        else:
            # 패턴 특성 없이 LSTM 출력만 사용
            output = self.fc(h_t)

        # 시그모이드 활성화 (각 번호의 출현 확률)
        return self.sigmoid(output)


class LSTMModel(ModelWithAMP):
    """LSTM 기반 로또 번호 예측 모델"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 설정
        self.config = config or {}
        lstm_config = self.config.get("lstm", {})

        # 모델 하이퍼파라미터
        self.input_dim = lstm_config.get("input_dim", 45)
        self.hidden_dim = lstm_config.get("hidden_dim", 128)
        self.output_dim = lstm_config.get("output_dim", 45)
        self.num_layers = lstm_config.get("num_layers", 2)
        self.dropout_rate = lstm_config.get("dropout_rate", 0.3)
        self.bidirectional = lstm_config.get("bidirectional", True)
        self.sequence_length = lstm_config.get("sequence_length", 5)

        # 패턴 특성 설정
        self.use_pattern_features = lstm_config.get("use_pattern_features", True)
        self.pattern_feature_dim = lstm_config.get("pattern_feature_dim", 16)

        # 학습 하이퍼파라미터
        self.learning_rate = lstm_config.get("learning_rate", 0.001)
        self.weight_decay = lstm_config.get("weight_decay", 1e-5)
        self.batch_size = lstm_config.get("batch_size", 32)

        # 모델 이름
        self.model_name = "LSTMModel"

        # 모델 구성
        self._build_model()

        logger.info(
            f"LSTM 모델 초기화 완료: 입력 차원={self.input_dim}, "
            f"은닉층 차원={self.hidden_dim}, 출력 차원={self.output_dim}, "
            f"LSTM 레이어 수={self.num_layers}, 양방향={self.bidirectional}"
        )

    def _build_model(self):
        """
        모델 구성
        """
        # LSTM 네트워크
        pattern_feature_dim = (
            self.pattern_feature_dim if self.use_pattern_features else 0
        )

        self.model = LSTMNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            pattern_feature_dim=pattern_feature_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            bidirectional=self.bidirectional,
        ).to(self.device)

        # 손실 함수
        self.criterion = nn.BCELoss()

        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 학습

        Args:
            X: 입력 시퀀스 특성 (samples, seq_length, features)
            y: 타겟 레이블 (samples, num_classes)
            **kwargs: 추가 매개변수

        Returns:
            학습 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"LSTM 모델 학습 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 학습 매개변수
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", self.batch_size)
        validation_split = kwargs.get("validation_split", 0.2)
        patience = kwargs.get("patience", 10)
        pattern_features = kwargs.get("pattern_features", None)

        # 입력 데이터 검증
        if len(X.shape) != 3:
            raise ValueError(
                f"X는 3차원이어야 합니다: (samples, seq_length, features), 현재: {X.shape}"
            )

        # 훈련/검증 세트 분할
        from sklearn.model_selection import train_test_split

        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )

            # 패턴 특성이 있는 경우 함께 분할
            if pattern_features is not None:
                pf_train, pf_val = train_test_split(
                    pattern_features, test_size=validation_split, random_state=42
                )
            else:
                pf_train = pf_val = None
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            pf_train = pattern_features
            pf_val = None

        # PyTorch 텐서로 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)

        if pattern_features is not None and self.use_pattern_features:
            pf_train_tensor = torch.FloatTensor(pf_train).to(self.device)
        else:
            pf_train_tensor = None

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

            if pf_val is not None and self.use_pattern_features:
                pf_val_tensor = torch.FloatTensor(pf_val).to(self.device)
            else:
                pf_val_tensor = None

        # 데이터 로더 생성
        train_dataset = torch.utils.data.TensorDataset(
            X_train_tensor,
            (
                y_train_tensor
                if pf_train_tensor is None
                else torch.utils.data.TensorDataset(y_train_tensor, pf_train_tensor)
            ),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        if X_val is not None:
            val_dataset = torch.utils.data.TensorDataset(
                X_val_tensor,
                (
                    y_val_tensor
                    if pf_val_tensor is None
                    else torch.utils.data.TensorDataset(y_val_tensor, pf_val_tensor)
                ),
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            val_loader = None

        # 훈련 시작
        logger.info("LSTM 모델 훈련 시작")
        start_time = time.time()

        best_val_loss = float("inf")
        no_improvement_count = 0
        training_history = []

        for epoch in range(epochs):
            # 훈련 모드
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                # 배치 데이터 추출
                inputs, targets = batch

                # 패턴 특성이 있는 경우
                if self.use_pattern_features and pf_train_tensor is not None:
                    targets, pattern_feats = targets
                else:
                    pattern_feats = None

                # AMP 사용 여부에 따라 훈련 단계 수행
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # 순전파
                        outputs = self.model(inputs, pattern_feats)
                        # 손실 계산
                        loss = self.criterion(outputs, targets)

                    # 역전파 및 최적화
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 순전파
                    outputs = self.model(inputs, pattern_feats)
                    # 손실 계산
                    loss = self.criterion(outputs, targets)

                    # 역전파 및 최적화
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # 배치 크기로 나누어 평균 훈련 손실 계산
            train_loss /= len(train_loader.dataset)

            # 검증 모드
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        # 배치 데이터 추출
                        inputs, targets = batch

                        # 패턴 특성이 있는 경우
                        if self.use_pattern_features and pf_val_tensor is not None:
                            targets, pattern_feats = targets
                        else:
                            pattern_feats = None

                        # 순전파
                        outputs = self.model(inputs, pattern_feats)
                        # 손실 계산
                        loss = self.criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)

                # 배치 크기로 나누어 평균 검증 손실 계산
                val_loss /= len(val_loader.dataset)

            # 에포크 결과 저장
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss if val_loader is not None else None,
            }
            training_history.append(epoch_result)

            # 로깅 (10 에포크마다)
            if (epoch + 1) % 10 == 0:
                val_msg = (
                    f", 검증 손실={val_loss:.6f}" if val_loader is not None else ""
                )
                logger.info(
                    f"에포크 {epoch + 1}/{epochs}: 훈련 손실={train_loss:.6f}{val_msg}"
                )

            # 조기 종료 검사
            if val_loader is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            elif val_loader is not None:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                logger.info(f"조기 종료: {patience}번 연속 개선 없음")
                break

        # 훈련 시간 계산
        train_time = time.time() - start_time

        # 훈련 완료 표시
        self.is_trained = True
        self.training_history = training_history

        # 메타데이터 업데이트
        self.metadata.update(
            {
                "train_samples": X_train.shape[0],
                "val_samples": X_val.shape[0] if X_val is not None else 0,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss if val_loader is not None else None,
                "best_val_loss": best_val_loss if val_loader is not None else None,
                "epochs_trained": epoch + 1,
                "train_time": train_time,
            }
        )

        logger.info(
            f"LSTM 모델 훈련 완료: 최종 훈련 손실={train_loss:.6f}, "
            f"소요 시간={train_time:.2f}초"
        )

        return {
            "train_loss": train_loss,
            "val_loss": val_loss if val_loader is not None else None,
            "best_val_loss": best_val_loss if val_loader is not None else None,
            "epochs": epoch + 1,
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        모델 예측 수행

        Args:
            X: 입력 시퀀스 특성 (samples, seq_length, features)
            **kwargs: 추가 매개변수

        Returns:
            예측 확률
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 로깅
        logger.info(f"LSTM 예측 수행: 입력 형태={X.shape}")

        # 배치 크기 설정
        batch_size = kwargs.get("batch_size", self.batch_size)
        pattern_features = kwargs.get("pattern_features", None)

        # 모델을 평가 모드로 설정
        self.model.eval()

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)

        if pattern_features is not None and self.use_pattern_features:
            pf_tensor = torch.FloatTensor(pattern_features).to(self.device)
        else:
            pf_tensor = None

        # 데이터 로더 생성
        if pf_tensor is not None:
            dataset = torch.utils.data.TensorDataset(X_tensor, pf_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(X_tensor)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # 예측 결과 저장
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                if pf_tensor is not None:
                    inputs, pattern_feats = batch
                else:
                    inputs = batch[0]
                    pattern_feats = None

                # 예측
                outputs = self.model(inputs, pattern_feats)
                predictions.append(outputs.cpu().numpy())

        # 배치별 결과 합치기
        return np.vstack(predictions)

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
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                    "num_layers": self.num_layers,
                    "dropout_rate": self.dropout_rate,
                    "bidirectional": self.bidirectional,
                    "use_pattern_features": self.use_pattern_features,
                    "pattern_feature_dim": self.pattern_feature_dim,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                },
                "metadata": self.metadata,
                "training_history": self.training_history,
                "is_trained": self.is_trained,
            }

            # 모델 저장
            torch.save(save_dict, path)

            logger.info(f"LSTM 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"LSTM 모델 저장 중 오류: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        모델 로드

        Args:
            path: 모델 파일 경로

        Returns:
            성공 여부
        """
        try:
            # 파일 존재 확인
            if not Path(path).exists():
                logger.error(f"모델 파일이 존재하지 않습니다: {path}")
                return False

            # 모델 데이터 로드
            checkpoint = torch.load(path, map_location=self.device)

            # 설정 로드
            config = checkpoint.get("config", {})
            self.input_dim = config.get("input_dim", self.input_dim)
            self.hidden_dim = config.get("hidden_dim", self.hidden_dim)
            self.output_dim = config.get("output_dim", self.output_dim)
            self.num_layers = config.get("num_layers", self.num_layers)
            self.dropout_rate = config.get("dropout_rate", self.dropout_rate)
            self.bidirectional = config.get("bidirectional", self.bidirectional)
            self.use_pattern_features = config.get(
                "use_pattern_features", self.use_pattern_features
            )
            self.pattern_feature_dim = config.get(
                "pattern_feature_dim", self.pattern_feature_dim
            )
            self.learning_rate = config.get("learning_rate", self.learning_rate)
            self.weight_decay = config.get("weight_decay", self.weight_decay)
            self.batch_size = config.get("batch_size", self.batch_size)

            # 모델 재구성
            self._build_model()

            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            # 메타데이터 로드
            self.metadata = checkpoint.get("metadata", {})
            self.training_history = checkpoint.get("training_history", [])
            self.is_trained = checkpoint.get("is_trained", False)

            logger.info(f"LSTM 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"LSTM 모델 로드 중 오류: {e}")
            return False

    def _ensure_directory(self, path: str) -> None:
        """
        저장 디렉토리 확인 및 생성

        Args:
            path: 파일 경로
        """
        directory = Path(path).parent
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
