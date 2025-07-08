"""
Optimized AutoEncoder Preprocessor
AutoEncoder 최적화 전처리기 - VAE, Progressive Training, Denoising 강화
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

from ..utils.unified_logging import get_logger
from ..shared.types import PreprocessedData

logger = get_logger(__name__)


@dataclass
class VAEConfig:
    """VAE 설정"""

    input_dim: int = 168
    latent_dim: int = 64
    hidden_dims: List[int] = None
    beta: float = 1.0  # KL divergence weight
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 100
    device: str = "cuda"


class VariationalAutoEncoder(nn.Module):
    """Variational AutoEncoder"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims or [128, 100]

        # Encoder
        encoder_layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_var = nn.Linear(prev_dim, self.latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = self.latent_dim

        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """인코딩"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """재매개화 트릭"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """디코딩"""
        return self.decoder(z)

    def forward(self, x):
        """순전파"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """VAE 손실 함수"""
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + beta * kl_loss, recon_loss, kl_loss


class ProgressiveTrainingScheduler:
    """Progressive Training 스케줄러"""

    def __init__(self, stages: List[int] = None):
        self.stages = stages or [168, 128, 100, 64]
        self.current_stage = 0
        self.stage_epochs = []

    def get_current_stage_config(self, total_epochs: int) -> Tuple[int, int]:
        """현재 단계 설정 반환"""
        epochs_per_stage = total_epochs // len(self.stages)
        current_dim = self.stages[self.current_stage]
        return current_dim, epochs_per_stage

    def next_stage(self):
        """다음 단계로 이동"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False

    def is_final_stage(self) -> bool:
        """최종 단계 여부"""
        return self.current_stage == len(self.stages) - 1


class AdaptiveNoiseInjection:
    """적응형 노이즈 주입"""

    def __init__(self, noise_ratio: float = 0.1, adaptive_scaling: bool = True):
        self.noise_ratio = noise_ratio
        self.adaptive_scaling = adaptive_scaling
        self.current_noise_level = noise_ratio

    def inject_noise(
        self, x: torch.Tensor, epoch: int = 0, max_epochs: int = 100
    ) -> torch.Tensor:
        """노이즈 주입"""
        if self.adaptive_scaling:
            # 학습 진행에 따라 노이즈 레벨 감소
            decay_factor = 1 - (epoch / max_epochs)
            self.current_noise_level = self.noise_ratio * decay_factor

        noise = torch.randn_like(x) * self.current_noise_level
        return x + noise

    def get_current_noise_level(self) -> float:
        """현재 노이즈 레벨 반환"""
        return self.current_noise_level


class OptimizedAutoEncoderPreprocessor:
    """최적화된 AutoEncoder 전처리기"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)

        # 디바이스 설정
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # VAE 설정
        self.vae_config = VAEConfig(
            input_dim=config.get("input_dim", 168),
            latent_dim=config.get("latent_dim", 64),
            hidden_dims=config.get("hidden_dims", [128, 100]),
            beta=config.get("beta", 1.0),
            learning_rate=config.get("learning_rate", 0.001),
            batch_size=config.get("batch_size", 128),
            epochs=config.get("epochs", 100),
            device=str(self.device),
        )

        # 컴포넌트 초기화
        self.vae = None
        self.progressive_scheduler = ProgressiveTrainingScheduler(
            stages=config.get("progressive_stages", [168, 128, 100, 64])
        )
        self.noise_injection = AdaptiveNoiseInjection(
            noise_ratio=config.get("noise_ratio", 0.1),
            adaptive_scaling=config.get("adaptive_scaling", True),
        )

        # 스케일러
        self.scaler = StandardScaler()
        self.fitted = False

    def preprocess_for_autoencoder(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        AutoEncoder 최적화 전처리

        Args:
            X: 입력 데이터
            y: 타겟 데이터 (선택사항)

        Returns:
            Tuple[np.ndarray, Dict]: 전처리된 데이터와 메타데이터
        """

        try:
            self.logger.info("AutoEncoder 최적화 전처리 시작")

            with self.memory_manager.get_context("autoencoder_preprocessing"):
                # 1. 데이터 정규화
                self.logger.info("데이터 정규화...")
                X_scaled = self.scaler.fit_transform(X)

                # 2. VAE 학습
                self.logger.info("VAE 학습...")
                self.vae = self._train_vae(X_scaled)

                # 3. 잠재 표현 추출
                self.logger.info("잠재 표현 추출...")
                X_encoded = self._extract_latent_representation(X_scaled)

                # 4. 재구성 품질 평가
                reconstruction_quality = self._evaluate_reconstruction_quality(X_scaled)

                # 메타데이터 생성
                metadata = {
                    "latent_dim": self.vae_config.latent_dim,
                    "reconstruction_loss": reconstruction_quality[
                        "reconstruction_loss"
                    ],
                    "kl_divergence": reconstruction_quality["kl_divergence"],
                    "total_loss": reconstruction_quality["total_loss"],
                    "compression_ratio": X.shape[1] / X_encoded.shape[1],
                }

                self.fitted = True
                self.logger.info(
                    f"AutoEncoder 전처리 완료 - 압축률: {metadata['compression_ratio']:.2f}x"
                )

                return X_encoded, metadata

        except Exception as e:
            self.logger.error(f"AutoEncoder 전처리 중 오류: {e}")
            raise

    def _train_vae(self, X: np.ndarray) -> VariationalAutoEncoder:
        """VAE 학습"""

        # VAE 모델 초기화
        vae = VariationalAutoEncoder(self.vae_config).to(self.device)
        optimizer = optim.Adam(vae.parameters(), lr=self.vae_config.learning_rate)

        # 데이터 로더 생성
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.vae_config.batch_size, shuffle=True
        )

        # Progressive Training
        if self.progressive_scheduler:
            return self._progressive_train(vae, optimizer, dataloader)
        else:
            return self._standard_train(vae, optimizer, dataloader)

    def _progressive_train(
        self, vae: VariationalAutoEncoder, optimizer, dataloader
    ) -> VariationalAutoEncoder:
        """Progressive Training 실행"""

        self.logger.info("Progressive Training 시작")

        # 각 단계별 학습
        for stage in range(len(self.progressive_scheduler.stages)):
            current_dim, stage_epochs = (
                self.progressive_scheduler.get_current_stage_config(
                    self.vae_config.epochs
                )
            )

            self.logger.info(
                f"Stage {stage + 1}: 차원 {current_dim}, 에포크 {stage_epochs}"
            )

            # 단계별 VAE 설정 조정
            if stage > 0:
                vae = self._adjust_vae_for_stage(vae, current_dim)

            # 단계별 학습
            for epoch in range(stage_epochs):
                total_loss = 0
                vae.train()

                for batch_idx, (data,) in enumerate(dataloader):
                    optimizer.zero_grad()

                    # 노이즈 주입 (Denoising)
                    noisy_data = self.noise_injection.inject_noise(
                        data, epoch, stage_epochs
                    )

                    # 순전파
                    recon_batch, mu, log_var = vae(noisy_data)

                    # 손실 계산
                    loss, recon_loss, kl_loss = vae.loss_function(
                        recon_batch, data, mu, log_var, self.vae_config.beta
                    )

                    # 역전파
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # 에포크별 로그
                if epoch % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    self.logger.info(
                        f"Stage {stage + 1}, Epoch {epoch}: Loss = {avg_loss:.4f}"
                    )

            # 다음 단계로 이동
            self.progressive_scheduler.next_stage()

        return vae

    def _standard_train(
        self, vae: VariationalAutoEncoder, optimizer, dataloader
    ) -> VariationalAutoEncoder:
        """표준 학습 실행"""

        self.logger.info("표준 VAE 학습 시작")

        for epoch in range(self.vae_config.epochs):
            total_loss = 0
            vae.train()

            for batch_idx, (data,) in enumerate(dataloader):
                optimizer.zero_grad()

                # 노이즈 주입
                noisy_data = self.noise_injection.inject_noise(
                    data, epoch, self.vae_config.epochs
                )

                # 순전파
                recon_batch, mu, log_var = vae(noisy_data)

                # 손실 계산
                loss, recon_loss, kl_loss = vae.loss_function(
                    recon_batch, data, mu, log_var, self.vae_config.beta
                )

                # 역전파
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 에포크별 로그
            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                self.logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        return vae

    def _adjust_vae_for_stage(
        self, vae: VariationalAutoEncoder, target_dim: int
    ) -> VariationalAutoEncoder:
        """단계별 VAE 조정"""

        # 현재 단계에서는 기본 VAE 구조 유지
        # 실제 구현에서는 네트워크 구조를 동적으로 조정할 수 있음
        return vae

    def _extract_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """잠재 표현 추출"""
        
        if self.vae is None:
            raise ValueError("VAE 모델이 학습되지 않았습니다.")
            
        self.vae.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            mu, log_var = self.vae.encode(X_tensor)
            # 평균값을 잠재 표현으로 사용
            latent_representation = mu.cpu().numpy()
        
        return latent_representation

    def _evaluate_reconstruction_quality(self, X: np.ndarray) -> Dict[str, float]:
        """재구성 품질 평가"""

        if self.vae is None:
            raise ValueError("VAE 모델이 학습되지 않았습니다.")
            
        self.vae.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            recon_x, mu, log_var = self.vae(X_tensor)
            loss, recon_loss, kl_loss = self.vae.loss_function(
                recon_x, X_tensor, mu, log_var, self.vae_config.beta
            )

        return {
            "total_loss": loss.item() / len(X),
            "reconstruction_loss": recon_loss.item() / len(X),
            "kl_divergence": kl_loss.item() / len(X),
        }

    def transform_new_data(self, X: np.ndarray) -> np.ndarray:
        """새로운 데이터 변환"""

        if not self.fitted:
            raise ValueError("먼저 전처리를 수행해야 합니다.")

        # 정규화
        X_scaled = self.scaler.transform(X)

        # 잠재 표현 추출
        X_encoded = self._extract_latent_representation(X_scaled)

        return X_encoded

    def reconstruct_data(self, X_encoded: np.ndarray) -> np.ndarray:
        """데이터 재구성"""

        if not self.fitted:
            raise ValueError("먼저 전처리를 수행해야 합니다.")

        self.vae.eval()
        X_tensor = torch.FloatTensor(X_encoded).to(self.device)

        with torch.no_grad():
            reconstructed = self.vae.decode(X_tensor)
            reconstructed = reconstructed.cpu().numpy()

        # 역정규화
        reconstructed = self.scaler.inverse_transform(reconstructed)

        return reconstructed

    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """이상값 점수 계산"""

        if not self.fitted:
            raise ValueError("먼저 전처리를 수행해야 합니다.")

        # 정규화
        X_scaled = self.scaler.transform(X)

        # 재구성
        X_encoded = self._extract_latent_representation(X_scaled)
        X_reconstructed = self.reconstruct_data(X_encoded)
        X_reconstructed_scaled = self.scaler.transform(X_reconstructed)

        # 재구성 오류 계산
        reconstruction_errors = np.mean(
            (X_scaled - X_reconstructed_scaled) ** 2, axis=1
        )

        return reconstruction_errors

    def generate_synthetic_data(self, n_samples: int = 1000) -> np.ndarray:
        """합성 데이터 생성"""

        if not self.fitted:
            raise ValueError("먼저 전처리를 수행해야 합니다.")

        self.vae.eval()

        # 잠재 공간에서 샘플링
        z = torch.randn(n_samples, self.vae_config.latent_dim).to(self.device)

        with torch.no_grad():
            synthetic_data = self.vae.decode(z)
            synthetic_data = synthetic_data.cpu().numpy()

        # 역정규화
        synthetic_data = self.scaler.inverse_transform(synthetic_data)

        return synthetic_data

    def print_optimization_summary(self, metadata: Dict[str, Any]):
        """최적화 결과 요약 출력"""

        print("=" * 60)
        print("🧠 AutoEncoder 최적화 결과")
        print("=" * 60)

        print(f"📊 모델 구성:")
        print(f"  • 입력 차원: {self.vae_config.input_dim}")
        print(f"  • 잠재 차원: {self.vae_config.latent_dim}")
        print(f"  • 은닉층: {self.vae_config.hidden_dims}")
        print(f"  • 압축률: {metadata['compression_ratio']:.2f}x")

        print(f"\n📈 학습 결과:")
        print(f"  • 총 손실: {metadata['total_loss']:.4f}")
        print(f"  • 재구성 손실: {metadata['reconstruction_loss']:.4f}")
        print(f"  • KL 발산: {metadata['kl_divergence']:.4f}")

        print(f"\n🔧 적용된 최적화 기법:")
        print(f"  • Variational AutoEncoder: 확률적 잠재 표현")
        print(f"  • Progressive Training: {len(self.progressive_scheduler.stages)}단계")
        print(f"  • Denoising: 적응형 노이즈 주입")
        print(f"  • GPU 가속: {'활성화' if self.device.type == 'cuda' else '비활성화'}")

        print("=" * 60)

    def preprocess(self, data: np.ndarray) -> PreprocessedData:
        """
        데이터를 전처리합니다.

        Args:
            data: 입력 데이터

        Returns:
            PreprocessedData: 전처리된 데이터와 메타데이터
        """

        try:
            self.logger.info("AutoEncoder 최적화 전처리 시작")

            with self.memory_manager.get_context("autoencoder_preprocessing"):
                # 1. 데이터 정규화
                self.logger.info("데이터 정규화...")
                X_scaled = self.scaler.fit_transform(data)

                # 2. VAE 학습
                self.logger.info("VAE 학습...")
                self.vae = self._train_vae(X_scaled)

                # 3. 잠재 표현 추출
                self.logger.info("잠재 표현 추출...")
                X_encoded = self._extract_latent_representation(X_scaled)

                # 4. 재구성 품질 평가
                reconstruction_quality = self._evaluate_reconstruction_quality(X_scaled)

                # 메타데이터 생성
                metadata = {
                    "latent_dim": self.vae_config.latent_dim,
                    "reconstruction_loss": reconstruction_quality[
                        "reconstruction_loss"
                    ],
                    "kl_divergence": reconstruction_quality["kl_divergence"],
                    "total_loss": reconstruction_quality["total_loss"],
                    "compression_ratio": data.shape[1] / X_encoded.shape[1],
                }

                self.fitted = True
                self.logger.info(
                    f"AutoEncoder 전처리 완료 - 압축률: {metadata['compression_ratio']:.2f}x"
                )

                return PreprocessedData(
                    data=X_encoded,
                    metadata=metadata
                )

        except Exception as e:
            self.logger.error(f"AutoEncoder 전처리 중 오류: {e}")
            raise
