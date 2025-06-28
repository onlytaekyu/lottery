"""
베이지안 신경망 모델

이 모듈은 베이지안 신경망 기반 로또 번호 예측 모델을 구현합니다.
불확실성을 추정하여 보다 신뢰할 수 있는 로또 번호 추천을 제공합니다.
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
from ...utils import get_logger

# 최적화된 로거 자동 할당
logger = get_logger(__name__)


class BayesianLinear(nn.Module):
    """
    베이지안 선형 레이어

    가중치의 사전 분포와 변분 사후 분포를 사용한 베이지안 선형 레이어를 구현합니다.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 0.1):
        """
        베이지안 레이어 초기화

        Args:
            in_features: 입력 특성 수
            out_features: 출력 특성 수
            prior_sigma: 사전 분포의 표준편차
        """
        super().__init__()

        # 가중치 및 바이어스의 평균
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))

        # 가중치 및 바이어스의 로그 표준편차
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        # 사전 분포 파라미터
        self.prior_sigma = prior_sigma

        # 가중치 초기화
        self.reset_parameters()

        # 가중치 및 바이어스 샘플
        self.weight = None
        self.bias = None

    def reset_parameters(self):
        """
        가중치 초기화
        """
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.constant_(self.bias_rho, -5.0)

    def _reparameterize(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """
        재매개화 트릭

        Args:
            mu: 평균 텐서
            rho: 로그 표준편차 텐서

        Returns:
            샘플링된 텐서
        """
        # 로그 표준편차를 표준편차로 변환
        sigma = torch.log1p(torch.exp(rho))

        # 표준 정규 분포에서 샘플링
        epsilon = torch.randn_like(sigma)

        # 재매개화 트릭 적용
        return mu + sigma * epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서

        Returns:
            출력 텐서
        """
        # 가중치 및 바이어스 샘플링
        self.weight = self._reparameterize(self.weight_mu, self.weight_rho)
        self.bias = self._reparameterize(self.bias_mu, self.bias_rho)

        # 선형 변환 적용
        return F.linear(x, self.weight, self.bias)

    def kl_divergence(self) -> torch.Tensor:
        """
        사전 분포와 변분 사후 분포 간의 KL 발산 계산

        Returns:
            KL 발산 텐서
        """
        # 가중치와 바이어스의 표준편차 계산
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # 가중치의 KL 발산
        kl_weight = self._kl_normal(
            self.weight_mu,
            weight_sigma,
            torch.zeros_like(self.weight_mu),
            self.prior_sigma * torch.ones_like(weight_sigma),
        )

        # 바이어스의 KL 발산
        kl_bias = self._kl_normal(
            self.bias_mu,
            bias_sigma,
            torch.zeros_like(self.bias_mu),
            self.prior_sigma * torch.ones_like(bias_sigma),
        )

        # 전체 KL 발산
        return kl_weight + kl_bias

    def _kl_normal(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """
        두 정규 분포 간의 KL 발산 계산

        Args:
            mu1: 첫 번째 분포의 평균
            sigma1: 첫 번째 분포의 표준편차
            mu2: 두 번째 분포의 평균
            sigma2: 두 번째 분포의 표준편차

        Returns:
            KL 발산 텐서
        """
        return (
            torch.log(sigma2 / sigma1)
            + (sigma1.pow(2) + (mu1 - mu2).pow(2)) / (2 * sigma2.pow(2))
            - 0.5
        ).sum()


class BayesianNN(nn.Module):
    """
    베이지안 신경망

    불확실성을 추정하는 베이지안 신경망 모델입니다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        prior_sigma: float = 0.1,
        dropout_rate: float = 0.1,
    ):
        """
        베이지안 신경망 초기화

        Args:
            input_dim: 입력 차원
            hidden_dims: 은닉층 차원 목록
            output_dim: 출력 차원
            prior_sigma: 사전 분포의 표준편차
            dropout_rate: 드롭아웃 비율
        """
        super().__init__()

        # 레이어 구성
        layers = []

        # 입력층 -> 첫 번째 은닉층
        layers.append(BayesianLinear(input_dim, hidden_dims[0], prior_sigma))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # 은닉층 간 연결
        for i in range(len(hidden_dims) - 1):
            layers.append(
                BayesianLinear(hidden_dims[i], hidden_dims[i + 1], prior_sigma)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # 마지막 은닉층 -> 출력층
        layers.append(BayesianLinear(hidden_dims[-1], output_dim, prior_sigma))

        # 레이어 시퀀스
        self.layers = nn.Sequential(*layers)

        # 베이지안 레이어 리스트 (KL 발산 계산용)
        self.bayesian_layers = [
            layer for layer in layers if isinstance(layer, BayesianLinear)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서

        Returns:
            출력 텐서
        """
        return self.layers(x)

    def kl_divergence(self) -> torch.Tensor:
        """
        전체 네트워크의 KL 발산 계산

        Returns:
            KL 발산 텐서
        """
        return sum(layer.kl_divergence() for layer in self.bayesian_layers)


class BayesianNNModel(ModelWithAMP):
    """
    베이지안 신경망 기반 로또 번호 예측 모델

    불확실성을 고려한 로또 번호 예측을 수행합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        베이지안 신경망 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 설정
        self.config = config or {}
        bayesian_config = self.config.get("bayesian_nn", {})

        # 모델 하이퍼파라미터
        self.input_dim = bayesian_config.get("input_dim", 70)
        self.hidden_dims = bayesian_config.get("hidden_dims", [128, 64])
        self.output_dim = bayesian_config.get("output_dim", 1)
        self.prior_sigma = bayesian_config.get("prior_sigma", 0.1)
        self.dropout_rate = bayesian_config.get("dropout_rate", 0.1)
        self.learning_rate = bayesian_config.get("learning_rate", 0.001)
        self.kl_weight = bayesian_config.get("kl_weight", 0.01)  # KL 발산 가중치
        self.num_samples = bayesian_config.get("num_samples", 10)  # 예측 시 샘플 수

        # 모델 이름
        self.model_name = "BayesianNNModel"

        # 모델 구성
        self._build_model()

        logger.info(
            f"베이지안 신경망 모델 초기화 완료: 입력 차원={self.input_dim}, "
            f"은닉층 차원={self.hidden_dims}, 출력 차원={self.output_dim}, "
            f"사전 표준편차={self.prior_sigma}, 드롭아웃={self.dropout_rate}"
        )

    def _build_model(self):
        """모델 구성"""
        # 베이지안 신경망
        self.model = BayesianNN(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            prior_sigma=self.prior_sigma,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # 손실 함수
        self.criterion = nn.MSELoss()

        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

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
        logger.info(
            f"베이지안 신경망 모델 훈련 시작: X 형태={X.shape}, y 형태={y.shape}"
        )

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
            self.model.train()
            train_loss = 0.0
            train_mse = 0.0
            train_kl = 0.0

            for batch_X, batch_y in train_loader:
                # AMP 컨텍스트에서 훈련
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # 순전파
                        outputs = self.model(batch_X)

                        # 손실 계산 (MSE + KL 발산)
                        mse_loss = self.criterion(outputs, batch_y)
                        kl_loss = self.model.kl_divergence() / len(train_loader.dataset)
                        loss = mse_loss + self.kl_weight * kl_loss

                    # 그래디언트 초기화
                    self.optimizer.zero_grad()

                    # 스케일된 역전파
                    self.scaler.scale(loss).backward()

                    # 스케일된 옵티마이저 스텝
                    self.scaler.step(self.optimizer)

                    # 스케일러 업데이트
                    self.scaler.update()
                else:
                    # 순전파
                    outputs = self.model(batch_X)

                    # 손실 계산 (MSE + KL 발산)
                    mse_loss = self.criterion(outputs, batch_y)
                    kl_loss = self.model.kl_divergence() / len(train_loader.dataset)
                    loss = mse_loss + self.kl_weight * kl_loss

                    # 역전파 및 옵티마이저 스텝
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # 손실 누적
                train_loss += loss.item() * batch_X.size(0)
                train_mse += mse_loss.item() * batch_X.size(0)
                train_kl += kl_loss.item() * batch_X.size(0)

            # 에포크 평균 손실
            train_loss /= len(train_loader.dataset)
            train_mse /= len(train_loader.dataset)
            train_kl /= len(train_loader.dataset)

            # 검증
            val_loss, val_mse, val_uncertainty = self._evaluate_validation(
                X_val_tensor, y_val_tensor
            )

            # 에포크 결과 기록
            training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_mse": train_mse,
                    "train_kl": train_kl,
                    "val_loss": val_loss,
                    "val_mse": val_mse,
                    "val_uncertainty": val_uncertainty,
                }
            )

            # 로깅
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"에포크 {epoch+1}/{epochs}: "
                    f"훈련 손실={train_loss:.4f} (MSE={train_mse:.4f}, KL={train_kl:.4f}), "
                    f"검증 손실={val_loss:.4f}, 불확실성={val_uncertainty:.4f}"
                )

            # 조기 종료 확인
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # 최상의 모델 저장
                best_model_state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
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
            f"베이지안 신경망 모델 훈련 완료: 최종 검증 손실={best_val_loss:.4f}, "
            f"에포크={epoch+1}/{epochs}, 소요 시간={train_time:.2f}초"
        )

        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
        }

    def _evaluate_validation(
        self, X_val: torch.Tensor, y_val: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        검증 세트 평가

        Args:
            X_val: 검증 특성 텐서
            y_val: 검증 타겟 텐서

        Returns:
            검증 손실, MSE, 불확실성의 튜플
        """
        self.model.eval()

        # 여러 번 예측
        preds = []
        kl_divs = []

        for _ in range(self.num_samples):
            with torch.no_grad():
                # 순전파
                pred = self.model(X_val)
                kl_div = self.model.kl_divergence() / len(X_val)

                preds.append(pred)
                kl_divs.append(kl_div.item())

        # 예측값 스택
        preds = torch.stack(preds)

        # 평균 및 표준편차 계산
        mean_pred = preds.mean(dim=0)
        uncertainty = preds.std(dim=0).mean().item()

        # MSE 계산
        mse = self.criterion(mean_pred, y_val).item()

        # 평균 KL 발산
        mean_kl = np.mean(kl_divs)

        # 전체 손실
        loss = mse + self.kl_weight * mean_kl

        return loss, mse, uncertainty

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
        self.model.eval()

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 여러 번 예측
        preds = []

        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.model(X_tensor)
                preds.append(pred)

        # 예측값 스택
        preds = torch.stack(preds)

        # 평균 계산
        mean_pred = preds.mean(dim=0)

        # NumPy 배열로 변환
        predictions_np = mean_pred.cpu().numpy().flatten()

        return predictions_np

    def predict_with_uncertainty(
        self, X: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        불확실성을 포함한 모델 예측 수행

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            예측 점수와 불확실성의 튜플
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 평가 모드로 설정
        self.model.eval()

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 여러 번 예측
        preds = []

        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.model(X_tensor)
                preds.append(pred)

        # 예측값 스택
        preds = torch.stack(preds)

        # 평균 및 표준편차 계산
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)

        # NumPy 배열로 변환
        mean_np = mean_pred.cpu().numpy().flatten()
        std_np = std_pred.cpu().numpy().flatten()

        return mean_np, std_np

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
        y_pred, y_std = self.predict_with_uncertainty(X)

        # 평가 메트릭 계산
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mean_uncertainty = np.mean(y_std)

        logger.info(
            f"베이지안 신경망 모델 평가: RMSE={rmse:.4f}, MAE={mae:.4f}, "
            f"R2={r2:.4f}, 불확실성={mean_uncertainty:.4f}"
        )

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "uncertainty": mean_uncertainty,
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
                    "prior_sigma": self.prior_sigma,
                    "dropout_rate": self.dropout_rate,
                    "learning_rate": self.learning_rate,
                    "kl_weight": self.kl_weight,
                    "num_samples": self.num_samples,
                },
                "metadata": self.metadata,
                "is_trained": self.is_trained,
                "training_history": self.training_history,
            }

            # 모델 저장
            torch.save(save_dict, path)

            logger.info(f"베이지안 신경망 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"베이지안 신경망 모델 저장 중 오류: {e}")
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
            self.prior_sigma = config.get("prior_sigma", self.prior_sigma)
            self.dropout_rate = config.get("dropout_rate", self.dropout_rate)
            self.learning_rate = config.get("learning_rate", self.learning_rate)
            self.kl_weight = config.get("kl_weight", self.kl_weight)
            self.num_samples = config.get("num_samples", self.num_samples)

            # 모델 재구성
            self._build_model()

            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint["model_state"])

            # 옵티마이저 상태 로드
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            # 메타데이터 로드
            self.metadata = checkpoint.get("metadata", {})
            self.is_trained = checkpoint.get("is_trained", False)
            self.training_history = checkpoint.get("training_history", [])

            logger.info(f"베이지안 신경망 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"베이지안 신경망 모델 로드 중 오류: {e}")
            return False


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
