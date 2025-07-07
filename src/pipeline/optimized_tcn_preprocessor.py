"""
Optimized TCN Preprocessor
TCN 최적화 전처리기 - 다중 해상도 윈도우, 시계열 증강, Attention 기반 특성 선택
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from scipy import signal
import warnings

warnings.filterwarnings("ignore")

from ..utils.unified_logging import get_logger
from ..utils.memory_manager import MemoryManager
from ..utils.cuda_singleton_manager import CudaSingletonManager

logger = get_logger(__name__)


@dataclass
class TCNConfig:
    """TCN 설정"""

    input_dim: int = 168
    output_dim: int = 64
    num_channels: List[int] = None
    kernel_size: int = 3
    dropout: float = 0.2
    sequence_length: int = 50
    attention_heads: int = 8
    device: str = "cuda"


class TimeSeriesAugmentation:
    """시계열 증강"""

    def __init__(self, augmentation_methods: List[str] = None):
        self.methods = augmentation_methods or [
            "time_warping",
            "magnitude_warping",
            "window_slicing",
        ]

    def augment(
        self, X: np.ndarray, method: str = "time_warping", **kwargs
    ) -> np.ndarray:
        """시계열 증강 적용"""

        if method == "time_warping":
            return self._time_warping(X, **kwargs)
        elif method == "magnitude_warping":
            return self._magnitude_warping(X, **kwargs)
        elif method == "window_slicing":
            return self._window_slicing(X, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 증강 방법: {method}")

    def _time_warping(self, X: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """시간 왜곡"""

        seq_len = X.shape[0]

        # 시간 왜곡 함수 생성
        time_steps = np.arange(seq_len)
        warp_steps = np.random.normal(0, sigma, seq_len)
        warped_time = time_steps + warp_steps

        # 경계 조건 처리
        warped_time = np.clip(warped_time, 0, seq_len - 1)

        # 보간을 통한 왜곡 적용
        warped_X = np.zeros_like(X)
        for i in range(X.shape[1]):
            warped_X[:, i] = np.interp(time_steps, warped_time, X[:, i])

        return warped_X

    def _magnitude_warping(self, X: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """크기 왜곡"""

        # 스무딩 커브 생성
        seq_len = X.shape[0]
        warp_curve = np.random.normal(1.0, sigma, seq_len)

        # 가우시안 필터로 스무딩
        warp_curve = signal.gaussian(seq_len, std=seq_len / 5) * warp_curve
        warp_curve = warp_curve / np.mean(warp_curve)  # 평균 1로 정규화

        # 크기 왜곡 적용
        warped_X = X * warp_curve.reshape(-1, 1)

        return warped_X

    def _window_slicing(self, X: np.ndarray, slice_ratio: float = 0.8) -> np.ndarray:
        """윈도우 슬라이싱"""

        seq_len = X.shape[0]
        slice_len = int(seq_len * slice_ratio)

        # 랜덤 시작점 선택
        start_idx = np.random.randint(0, seq_len - slice_len + 1)

        # 슬라이싱 후 원본 길이로 복원 (패딩 또는 반복)
        sliced_X = X[start_idx : start_idx + slice_len]

        if slice_len < seq_len:
            # 패딩으로 원본 길이 복원
            pad_len = seq_len - slice_len
            pad_X = np.zeros((pad_len, X.shape[1]))
            sliced_X = np.vstack([sliced_X, pad_X])

        return sliced_X


class TemporalAttentionSelector(nn.Module):
    """시계열 Attention 기반 특성 선택기"""

    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config
        self.sequence_length = config.sequence_length
        self.attention_heads = config.attention_heads
        self.input_dim = config.input_dim

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=self.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Feature importance 계산용 레이어
        self.feature_importance = nn.Linear(self.input_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            attended_features: (batch_size, sequence_length, input_dim)
            attention_weights: (batch_size, sequence_length, sequence_length)
        """

        # Self-attention
        attended_features, attention_weights = self.attention(x, x, x)

        # Feature importance 계산
        feature_importance = torch.sigmoid(self.feature_importance(attended_features))

        # 중요도 기반 특성 선택
        selected_features = attended_features * feature_importance

        return selected_features, attention_weights

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """특성 중요도 반환"""

        with torch.no_grad():
            attended_features, _ = self.attention(x, x, x)
            importance = torch.sigmoid(self.feature_importance(attended_features))

        return importance.mean(dim=(0, 1))  # 배치와 시퀀스 차원에서 평균


class OptimizedTCNPreprocessor:
    """최적화된 TCN 전처리기"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.memory_manager = MemoryManager()
        self.cuda_manager = CudaSingletonManager()

        # 디바이스 설정
        self.device = torch.device(
            "cuda" if self.cuda_manager.is_available() else "cpu"
        )

        # TCN 설정
        self.tcn_config = TCNConfig(
            input_dim=config.get("input_dim", 168),
            output_dim=config.get("output_dim", 64),
            num_channels=config.get("num_channels", [100, 100, 100]),
            kernel_size=config.get("kernel_size", 3),
            dropout=config.get("dropout", 0.2),
            sequence_length=config.get("sequence_length", 50),
            attention_heads=config.get("attention_heads", 8),
            device=str(self.device),
        )

        # 다중 해상도 윈도우 설정
        self.multi_resolution_windows = config.get(
            "multi_resolution_windows", [10, 25, 50, 100]
        )

        # 컴포넌트 초기화
        self.time_augmentation = TimeSeriesAugmentation(
            ["time_warping", "magnitude_warping", "window_slicing"]
        )

        self.attention_selector = TemporalAttentionSelector(self.tcn_config).to(
            self.device
        )

        # 스케일러
        self.scaler = StandardScaler()
        self.fitted = False

    def preprocess_for_tcn(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        TCN 최적화 전처리

        Args:
            X: 입력 데이터 (samples, features)
            y: 타겟 데이터 (선택사항)

        Returns:
            Tuple[np.ndarray, Dict]: 전처리된 시계열 데이터와 메타데이터
        """

        try:
            self.logger.info("TCN 최적화 전처리 시작")

            with self.memory_manager.get_context("tcn_preprocessing"):
                # 1. 데이터 정규화
                self.logger.info("데이터 정규화...")
                X_scaled = self.scaler.fit_transform(X)

                # 2. 시계열 시퀀스 생성
                self.logger.info("시계열 시퀀스 생성...")
                X_sequences = self._create_sequences(X_scaled)

                # 3. 다중 해상도 특성 추출
                self.logger.info("다중 해상도 특성 추출...")
                X_multi_resolution = self._extract_multi_resolution_features(
                    X_sequences
                )

                # 4. 시계열 증강
                self.logger.info("시계열 증강...")
                X_augmented = self._apply_time_series_augmentation(X_multi_resolution)

                # 5. Attention 기반 특성 선택
                self.logger.info("Attention 기반 특성 선택...")
                X_selected, attention_weights = self._apply_attention_selection(
                    X_augmented
                )

                # 6. 최종 시퀀스 형태로 변환
                X_final = self._format_for_tcn(X_selected)

                # 메타데이터 생성
                metadata = {
                    "sequence_length": self.tcn_config.sequence_length,
                    "input_dim": X_final.shape[-1],
                    "multi_resolution_windows": self.multi_resolution_windows,
                    "attention_heads": self.tcn_config.attention_heads,
                    "augmentation_methods": self.time_augmentation.methods,
                    "compression_ratio": X.shape[1] / X_final.shape[-1],
                }

                self.fitted = True
                self.logger.info(
                    f"TCN 전처리 완료 - 시퀀스 길이: {X_final.shape[1]}, 특성 수: {X_final.shape[2]}"
                )

                return X_final, metadata

        except Exception as e:
            self.logger.error(f"TCN 전처리 중 오류: {e}")
            raise

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """시계열 시퀀스 생성"""

        sequence_length = self.tcn_config.sequence_length

        if len(X) < sequence_length:
            # 데이터가 부족한 경우 패딩
            pad_length = sequence_length - len(X)
            X_padded = np.pad(X, ((0, pad_length), (0, 0)), mode="edge")
            sequences = X_padded.reshape(1, sequence_length, -1)
        else:
            # 슬라이딩 윈도우로 시퀀스 생성
            sequences = []
            for i in range(len(X) - sequence_length + 1):
                sequences.append(X[i : i + sequence_length])
            sequences = np.array(sequences)

        return sequences

    def _extract_multi_resolution_features(self, X_sequences: np.ndarray) -> np.ndarray:
        """다중 해상도 특성 추출"""

        multi_resolution_features = []

        for window_size in self.multi_resolution_windows:
            if window_size <= X_sequences.shape[1]:
                # 지정된 윈도우 크기로 다운샘플링
                step = max(1, X_sequences.shape[1] // window_size)
                downsampled = X_sequences[:, ::step, :]

                # 고정 길이로 맞춤
                if downsampled.shape[1] > window_size:
                    downsampled = downsampled[:, :window_size, :]
                elif downsampled.shape[1] < window_size:
                    # 패딩
                    pad_length = window_size - downsampled.shape[1]
                    downsampled = np.pad(
                        downsampled, ((0, 0), (0, pad_length), (0, 0)), mode="edge"
                    )

                multi_resolution_features.append(downsampled)

        # 모든 해상도 특성 연결
        if multi_resolution_features:
            # 시퀀스 차원에서 연결
            combined_features = np.concatenate(multi_resolution_features, axis=1)
        else:
            combined_features = X_sequences

        return combined_features

    def _apply_time_series_augmentation(self, X: np.ndarray) -> np.ndarray:
        """시계열 증강 적용"""

        augmented_sequences = []

        for i, sequence in enumerate(X):
            # 원본 시퀀스 추가
            augmented_sequences.append(sequence)

            # 각 증강 방법 적용 (50% 확률)
            for method in self.time_augmentation.methods:
                if np.random.random() < 0.5:
                    try:
                        augmented_seq = self.time_augmentation.augment(sequence, method)
                        augmented_sequences.append(augmented_seq)
                    except Exception as e:
                        self.logger.warning(f"증강 방법 {method} 적용 중 오류: {e}")

        return np.array(augmented_sequences)

    def _apply_attention_selection(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Attention 기반 특성 선택"""

        # PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Attention 적용
        self.attention_selector.eval()
        with torch.no_grad():
            selected_features, attention_weights = self.attention_selector(X_tensor)

        # NumPy 배열로 변환
        selected_features = selected_features.cpu().numpy()
        attention_weights = attention_weights.cpu().numpy()

        return selected_features, attention_weights

    def _format_for_tcn(self, X: np.ndarray) -> np.ndarray:
        """TCN 입력 형태로 포맷팅"""

        # TCN은 (batch_size, sequence_length, input_dim) 형태를 기대
        if len(X.shape) == 3:
            return X
        else:
            # 2D 배열인 경우 시퀀스 차원 추가
            return X.reshape(X.shape[0], 1, -1)

    def transform_new_data(self, X: np.ndarray) -> np.ndarray:
        """새로운 데이터 변환"""

        if not self.fitted:
            raise ValueError("먼저 전처리를 수행해야 합니다.")

        # 정규화
        X_scaled = self.scaler.transform(X)

        # 시퀀스 생성
        X_sequences = self._create_sequences(X_scaled)

        # 다중 해상도 특성 추출
        X_multi_resolution = self._extract_multi_resolution_features(X_sequences)

        # Attention 기반 특성 선택 (증강 제외)
        X_selected, _ = self._apply_attention_selection(X_multi_resolution)

        # 최종 형태로 변환
        X_final = self._format_for_tcn(X_selected)

        return X_final

    def get_temporal_patterns(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """시계열 패턴 분석"""

        if not self.fitted:
            raise ValueError("먼저 전처리를 수행해야 합니다.")

        # 변환
        X_transformed = self.transform_new_data(X)

        # 패턴 분석
        patterns = {}

        # 1. 시계열 트렌드
        patterns["trend"] = np.mean(X_transformed, axis=0)

        # 2. 변동성
        patterns["volatility"] = np.std(X_transformed, axis=0)

        # 3. 자기상관
        patterns["autocorrelation"] = []
        for i in range(X_transformed.shape[2]):
            feature_series = X_transformed[:, :, i].flatten()
            autocorr = np.correlate(feature_series, feature_series, mode="full")
            patterns["autocorrelation"].append(autocorr)

        patterns["autocorrelation"] = np.array(patterns["autocorrelation"])

        return patterns

    def get_attention_analysis(self, X: np.ndarray) -> Dict[str, Any]:
        """Attention 분석"""

        if not self.fitted:
            raise ValueError("먼저 전처리를 수행해야 합니다.")

        # 전처리
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled)
        X_multi_resolution = self._extract_multi_resolution_features(X_sequences)

        # Attention 가중치 추출
        X_tensor = torch.FloatTensor(X_multi_resolution).to(self.device)

        self.attention_selector.eval()
        with torch.no_grad():
            _, attention_weights = self.attention_selector(X_tensor)
            feature_importance = self.attention_selector.get_feature_importance(
                X_tensor
            )

        analysis = {
            "attention_weights": attention_weights.cpu().numpy(),
            "feature_importance": feature_importance.cpu().numpy(),
            "top_features": torch.topk(
                feature_importance, k=min(20, len(feature_importance))
            )
            .indices.cpu()
            .numpy(),
        }

        return analysis

    def print_optimization_summary(self, metadata: Dict[str, Any]):
        """최적화 결과 요약 출력"""

        print("=" * 60)
        print("⏰ TCN 최적화 결과")
        print("=" * 60)

        print(f"📊 시계열 구성:")
        print(f"  • 시퀀스 길이: {metadata['sequence_length']}")
        print(f"  • 입력 차원: {metadata['input_dim']}")
        print(f"  • 다중 해상도 윈도우: {metadata['multi_resolution_windows']}")
        print(f"  • Attention 헤드: {metadata['attention_heads']}")

        print(f"\n🔧 적용된 최적화 기법:")
        print(
            f"  • 다중 해상도 윈도우: {len(metadata['multi_resolution_windows'])}개 해상도"
        )
        print(f"  • 시계열 증강: {', '.join(metadata['augmentation_methods'])}")
        print(f"  • Attention 기반 특성 선택: {metadata['attention_heads']}헤드")
        print(f"  • GPU 가속: {'활성화' if self.device.type == 'cuda' else '비활성화'}")

        print(f"\n📈 압축 효율:")
        print(f"  • 특성 압축률: {metadata['compression_ratio']:.2f}x")

        print("=" * 60)
