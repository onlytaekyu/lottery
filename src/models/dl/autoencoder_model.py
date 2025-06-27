"""
Autoencoder 모델

이 모듈은 오토인코더 기반 이상 탐지 및 감점용 모델을 구현합니다.
이상치를 탐지하고 패턴에서 벗어난 로또 번호 조합에 감점을 부여합니다.
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


class AutoencoderNetwork(nn.Module):
    """
    오토인코더 신경망

    인코더와 디코더로 구성된 대칭형 오토인코더 네트워크입니다.
    이상 탐지에 활용합니다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        """
        오토인코더 네트워크 초기화

        Args:
            input_dim: 입력 차원
            hidden_dims: 인코더 은닉층 차원 목록 (디코더는 반대 순서로 구성)
            latent_dim: 잠재 표현 차원
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

        # 인코더 레이어 구성
        encoder_layers = []

        # 입력층 -> 첫 번째 은닉층
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if batch_norm:
            encoder_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        encoder_layers.append(self.activation)
        encoder_layers.append(nn.Dropout(dropout_rate))

        # 은닉층 간 연결
        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Dropout(dropout_rate))

        # 마지막 은닉층 -> 잠재 표현
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        if batch_norm:
            encoder_layers.append(nn.BatchNorm1d(latent_dim))
        encoder_layers.append(self.activation)

        # 인코더 모델 생성
        self.encoder = nn.Sequential(*encoder_layers)

        # 디코더 레이어 구성 (인코더의 역순)
        decoder_layers = []

        # 잠재 표현 -> 첫 번째 디코더 은닉층
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        if batch_norm:
            decoder_layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        decoder_layers.append(self.activation)
        decoder_layers.append(nn.Dropout(dropout_rate))

        # 디코더 은닉층 간 연결
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            if batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dims[i - 1]))
            decoder_layers.append(self.activation)
            decoder_layers.append(nn.Dropout(dropout_rate))

        # 마지막 디코더 은닉층 -> 출력층
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))

        # 디코더 모델 생성
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        인코딩 함수

        Args:
            x: 입력 텐서

        Returns:
            잠재 표현
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        디코딩 함수

        Args:
            z: 잠재 표현

        Returns:
            재구성된 출력
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파

        Args:
            x: 입력 텐서

        Returns:
            (재구성된 출력, 잠재 표현) 튜플
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoencoderModel(ModelWithAMP):
    """
    오토인코더 기반 로또 번호 이상 탐지 모델

    특성 벡터를 입력으로 받아 정상 패턴을 학습하고,
    재구성 오류를 통해 이상치를 탐지합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        오토인코더 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 설정
        self.config = config or {}
        ae_config = self.config.get("autoencoder", {})

        # 모델 하이퍼파라미터
        self.input_dim = ae_config.get("input_dim", 70)
        self.hidden_dims = ae_config.get("hidden_dims", [64, 32])
        self.latent_dim = ae_config.get("latent_dim", 16)
        self.dropout_rate = ae_config.get("dropout_rate", 0.2)
        self.activation = ae_config.get("activation", "relu")
        self.batch_norm = ae_config.get("batch_norm", True)

        # 학습 하이퍼파라미터
        self.learning_rate = ae_config.get("learning_rate", 0.001)
        self.weight_decay = ae_config.get("weight_decay", 1e-5)

        # 이상 탐지 임계값
        self.reconstruction_threshold = ae_config.get("reconstruction_threshold", None)
        self.zscore_threshold = ae_config.get("zscore_threshold", 2.5)

        # 모델 이름
        self.model_name = "AutoencoderModel"

        # 재구성 오류 통계
        self.error_mean = None
        self.error_std = None

        # 모델 구성
        self._build_model()

        logger.info(
            f"오토인코더 모델 초기화 완료: 입력 차원={self.input_dim}, "
            f"은닉층 차원={self.hidden_dims}, 잠재 차원={self.latent_dim}, "
            f"활성화 함수={self.activation}, 배치 정규화={self.batch_norm}"
        )

    def _build_model(self):
        """
        모델 구성
        """
        # 오토인코더 네트워크
        self.model = AutoencoderNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            batch_norm=self.batch_norm,
        ).to(self.device)

        # 손실 함수
        self.criterion = nn.MSELoss(reduction="none")

        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        모델 훈련

        Args:
            X: 특성 벡터 (정상 데이터)
            y: 사용되지 않음 (오토인코더는 비지도 학습)
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"오토인코더 모델 훈련 시작: X 형태={X.shape}")

        # 훈련 매개변수
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 64)
        validation_split = kwargs.get("validation_split", 0.2)
        patience = kwargs.get("patience", 10)

        # 훈련/검증 세트 분할
        from sklearn.model_selection import train_test_split

        X_train, X_val = train_test_split(
            X, test_size=validation_split, random_state=42
        )

        # PyTorch 텐서로 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        # 데이터 로더 생성
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, X_val_tensor)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # 훈련 시작
        logger.info("오토인코더 훈련 시작")
        start_time = time.time()

        best_val_loss = float("inf")
        no_improvement_count = 0
        training_history = []

        for epoch in range(epochs):
            # 훈련 모드
            self.model.train()
            train_loss = 0.0

            for inputs, _ in train_loader:
                # AMP 사용 여부에 따라 훈련 단계 수행
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # 순전파
                        outputs, _ = self.model(inputs)
                        # 손실 계산 (평균 MSE)
                        loss = self.criterion(outputs, inputs).mean()

                    # 역전파 및 최적화
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 순전파
                    outputs, _ = self.model(inputs)
                    # 손실 계산
                    loss = self.criterion(outputs, inputs).mean()

                    # 역전파 및 최적화
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # 배치 크기로 나누어 평균 훈련 손실 계산
            train_loss /= len(train_loader.dataset)

            # 검증 모드
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, _ in val_loader:
                    # 순전파
                    outputs, _ = self.model(inputs)
                    # 손실 계산
                    loss = self.criterion(outputs, inputs).mean()
                    val_loss += loss.item() * inputs.size(0)

            # 배치 크기로 나누어 평균 검증 손실 계산
            val_loss /= len(val_loader.dataset)

            # 에포크 결과 저장
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            training_history.append(epoch_result)

            # 로깅 (10 에포크마다)
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"에포크 {epoch + 1}/{epochs}: 훈련 손실={train_loss:.6f}, 검증 손실={val_loss:.6f}"
                )

            # 조기 종료 검사
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                logger.info(f"조기 종료: {patience}번 연속 개선 없음")
                break

        # 훈련 시간 계산
        train_time = time.time() - start_time

        # 재구성 오류 통계 계산
        self._compute_reconstruction_error_stats(X)

        # 훈련 완료 표시
        self.is_trained = True
        self.training_history = training_history

        # 메타데이터 업데이트
        self.metadata.update(
            {
                "train_samples": X_train.shape[0],
                "val_samples": X_val.shape[0],
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "epochs_trained": epoch + 1,
                "train_time": train_time,
                "error_mean": (
                    float(self.error_mean) if self.error_mean is not None else None
                ),
                "error_std": (
                    float(self.error_std) if self.error_std is not None else None
                ),
                "reconstruction_threshold": (
                    float(self.reconstruction_threshold)
                    if self.reconstruction_threshold is not None
                    else None
                ),
            }
        )

        logger.info(
            f"오토인코더 훈련 완료: 최종 훈련 손실={train_loss:.6f}, "
            f"최종 검증 손실={val_loss:.6f}, 소요 시간={train_time:.2f}초"
        )

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs": epoch + 1,
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
        }

    def _compute_reconstruction_error_stats(self, X: np.ndarray) -> None:
        """
        재구성 오류 통계 계산

        Args:
            X: 입력 데이터
        """
        # 모델을 평가 모드로 설정
        self.model.eval()

        # 배치 크기 설정
        batch_size = 128

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 데이터 로더 생성
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # 모든 샘플에 대한 재구성 오류 저장
        all_errors = []

        with torch.no_grad():
            for (inputs,) in dataloader:
                # 재구성
                outputs, _ = self.model(inputs)

                # 오류 계산 (MSE)
                errors = self.criterion(outputs, inputs)

                # 각 샘플의 평균 오류 계산
                sample_errors = errors.mean(dim=1).cpu().numpy()
                all_errors.extend(sample_errors)

        # 오류 통계 계산
        all_errors = np.array(all_errors)
        self.error_mean = np.mean(all_errors)
        self.error_std = np.std(all_errors)

        # 임계값 설정 (설정에 없는 경우)
        if self.reconstruction_threshold is None:
            # Z-점수 임계값 사용
            self.reconstruction_threshold = (
                self.error_mean + self.zscore_threshold * self.error_std
            )

        logger.info(
            f"재구성 오류 통계: 평균={self.error_mean:.6f}, 표준편차={self.error_std:.6f}, "
            f"임계값={self.reconstruction_threshold:.6f}"
        )

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        모델 예측 수행 (이상 점수 계산)

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            이상 점수 (높을수록 이상치)
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 로깅
        logger.info(f"오토인코더 예측 수행: 입력 형태={X.shape}")

        # 배치 크기 설정
        batch_size = kwargs.get("batch_size", 128)

        # 모델을 평가 모드로 설정
        self.model.eval()

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 데이터 로더 생성
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # 예측 결과 저장
        anomaly_scores = []

        with torch.no_grad():
            for (inputs,) in dataloader:
                # 재구성
                outputs, _ = self.model(inputs)

                # 오류 계산 (MSE)
                errors = self.criterion(outputs, inputs)

                # 각 샘플의 평균 오류 계산
                sample_errors = errors.mean(dim=1).cpu().numpy()

                # 이상 점수 계산 (정규화된 점수)
                if self.error_std > 0:
                    z_scores = (sample_errors - self.error_mean) / self.error_std
                    anomaly_scores.extend(z_scores)
                else:
                    # 표준편차가 0인 경우 (거의 발생하지 않음)
                    anomaly_scores.extend(np.zeros_like(sample_errors))

        return np.array(anomaly_scores)

    def get_latent_representation(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        잠재 표현 추출

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            잠재 표현 벡터
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 배치 크기 설정
        batch_size = kwargs.get("batch_size", 128)

        # 모델을 평가 모드로 설정
        self.model.eval()

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 데이터 로더 생성
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # 잠재 표현 저장
        latent_vectors = []

        with torch.no_grad():
            for (inputs,) in dataloader:
                # 인코딩
                latent = self.model.encode(inputs)
                latent_vectors.append(latent.cpu().numpy())

        return np.vstack(latent_vectors)

    def is_anomaly(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        이상치 여부 판단

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            이상치 여부 불리언 배열
        """
        # 이상 점수 계산
        anomaly_scores = self.predict(X, **kwargs)

        # 임계값 설정 (필요한 경우)
        threshold = kwargs.get("threshold", self.zscore_threshold)

        # 임계값이 None인 경우 기본값 설정
        if threshold is None:
            threshold = 2.0  # 기본 임계값
            logger.warning(f"임계값이 None입니다. 기본값 {threshold}를 사용합니다.")

        # 이상치 여부 판단
        return anomaly_scores > threshold

    def evaluate(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        모델 평가

        Args:
            X: 특성 벡터
            y: 실제 이상치 레이블 (있는 경우)
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 재구성 오류 계산
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs, latent = self.model(X_tensor)
            mse = self.criterion(outputs, X_tensor).mean(dim=1).cpu().numpy()

        # 평가 결과
        results = {
            "mean_reconstruction_error": float(np.mean(mse)),
            "median_reconstruction_error": float(np.median(mse)),
            "min_reconstruction_error": float(np.min(mse)),
            "max_reconstruction_error": float(np.max(mse)),
            "std_reconstruction_error": float(np.std(mse)),
        }

        # 레이블이 있는 경우 추가 평가
        if y is not None:
            from sklearn.metrics import (
                roc_auc_score,
                precision_score,
                recall_score,
                f1_score,
            )

            # 이상치 예측
            anomaly_scores = self.predict(X)
            pred_labels = anomaly_scores > self.zscore_threshold

            try:
                results.update(
                    {
                        "roc_auc": float(roc_auc_score(y, anomaly_scores)),
                        "precision": float(precision_score(y, pred_labels)),
                        "recall": float(recall_score(y, pred_labels)),
                        "f1": float(f1_score(y, pred_labels)),
                    }
                )
            except Exception as e:
                logger.warning(f"레이블 기반 평가 중 오류 발생: {e}")

        return results

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
                    "latent_dim": self.latent_dim,
                    "dropout_rate": self.dropout_rate,
                    "activation": self.activation,
                    "batch_norm": self.batch_norm,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
                "metadata": self.metadata,
                "training_history": self.training_history,
                "is_trained": self.is_trained,
                "error_mean": self.error_mean,
                "error_std": self.error_std,
                "reconstruction_threshold": self.reconstruction_threshold,
            }

            # 모델 저장
            torch.save(save_dict, path)

            logger.info(f"오토인코더 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"오토인코더 모델 저장 중 오류: {e}")
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
            if not os.path.exists(path):
                logger.error(f"모델 파일이 존재하지 않습니다: {path}")
                return False

            # 모델 데이터 로드
            checkpoint = torch.load(path, map_location=self.device)

            # 설정 로드
            config = checkpoint.get("config", {})
            self.input_dim = config.get("input_dim", self.input_dim)
            self.hidden_dims = config.get("hidden_dims", self.hidden_dims)
            self.latent_dim = config.get("latent_dim", self.latent_dim)
            self.dropout_rate = config.get("dropout_rate", self.dropout_rate)
            self.activation = config.get("activation", self.activation)
            self.batch_norm = config.get("batch_norm", self.batch_norm)
            self.learning_rate = config.get("learning_rate", self.learning_rate)
            self.weight_decay = config.get("weight_decay", self.weight_decay)

            # 모델 재구성
            self._build_model()

            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            # 메타데이터 로드
            self.metadata = checkpoint.get("metadata", {})
            self.training_history = checkpoint.get("training_history", [])
            self.is_trained = checkpoint.get("is_trained", False)
            self.error_mean = checkpoint.get("error_mean", None)
            self.error_std = checkpoint.get("error_std", None)
            self.reconstruction_threshold = checkpoint.get(
                "reconstruction_threshold", None
            )

            logger.info(f"오토인코더 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"오토인코더 모델 로드 중 오류: {e}")
            return False
