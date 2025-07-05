"""
통합 특성 벡터 검증기 (v3 - GPU 가속)

GPU를 활용하여 대규모 특성 벡터의 차원, 값 범위, 이상치 등을
고속으로 검증하는 통합 시스템.
"""

import json
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from .unified_logging import get_logger
from .cuda_optimizers import get_cuda_optimizer
from .memory_manager import get_memory_manager
import threading

logger = get_logger(__name__)
GPU_AVAILABLE = torch.cuda.is_available()

# 필수 특성 목록 (전역 상수로 변경)
ESSENTIAL_FEATURES = ["frequency_mean", "recency_mean", "pattern_score"]


class GPUFeatureValidator:
    """GPU 가속 및 자동 메모리 관리를 지원하는 통합 피처 벡터 검증기"""

    def __init__(self, gpu_threshold: int = 10000):
        self.device = torch.device("cuda" if GPU_AVAILABLE else "cpu")
        self.gpu_threshold = gpu_threshold
        self.memory_manager = get_memory_manager()

        if GPU_AVAILABLE:
            logger.info(f"✅ GPU 피처 검증기 초기화 (GPU 임계값: {self.gpu_threshold})")
        else:
            logger.info("✅ 피처 검증기 초기화 (CPU 모드)")

    def validate_vectors(
        self, vectors: torch.Tensor, feature_names: List[str], z_threshold: float = 3.0
    ) -> bool:
        """
        피처 벡터의 유효성을 검증합니다. (필수 피처, NaN/Inf, 이상치)
        데이터 크기에 따라 GPU 사용을 자동으로 결정합니다.
        """
        if not self._check_essential_features(feature_names):
            return False

        use_gpu = GPU_AVAILABLE and vectors.numel() >= self.gpu_threshold

        # 메모리 관리자를 통해 텐서 할당/이동
        input_tensor = self.memory_manager.smart_allocate(
            vectors.shape, dtype=vectors.dtype, prefer_gpu=use_gpu
        )
        input_tensor.copy_(vectors)

        # 1. NaN/Inf 검증
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logger.error("검증 실패: 벡터에 NaN 또는 Inf 값이 존재합니다.")
            return False

        # 2. 이상치 탐지 (Z-score)
        mean = torch.mean(input_tensor, dim=0)
        std = torch.std(input_tensor, dim=0)
        z_scores = torch.abs((input_tensor - mean) / (std + 1e-8))

        if torch.any(z_scores > z_threshold):
            outlier_ratio = torch.mean(
                torch.any(z_scores > z_threshold, dim=1).float()
            ).item()
            logger.warning(
                f"이상치 감지: 전체 벡터의 {outlier_ratio:.2%}가 임계값을 벗어납니다."
            )
            # 실패로 처리하지 않고 경고만 기록 (정책에 따라 변경 가능)

        logger.info(f"피처 벡터 검증 완료 (Device: {input_tensor.device})")
        return True

    def _check_essential_features(self, feature_names: List[str]) -> bool:
        """필수 피처 존재 여부 확인"""
        missing = [f for f in ESSENTIAL_FEATURES if f not in feature_names]
        if missing:
            logger.error(f"필수 피처 누락: {missing}")
            return False
        return True


class UnifiedFeatureVectorValidator:
    """CPU 및 GPU 검증을 통합한 특성 벡터 검증기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.gpu_validator = GPUFeatureValidator() if GPU_AVAILABLE else None

    def validate_all(
        self, vectors: Union[np.ndarray, torch.Tensor], feature_names: List[str]
    ) -> bool:
        """모든 검증을 수행합니다."""
        if not self.check_essential_features(feature_names):
            return False

        if self.gpu_validator:
            if isinstance(vectors, np.ndarray):
                vectors = torch.from_numpy(vectors).to(
                    self.gpu_validator.device, non_blocking=True
                )

            results = self.gpu_validator.validate_vectors(vectors, feature_names)
            if not results:
                logger.error("GPU 벡터 검증 실패")
                return False
            logger.info("GPU 벡터 검증 완료")
        else:
            # CPU Fallback
            if isinstance(vectors, torch.Tensor):
                vectors = vectors.cpu().numpy()

            if np.isnan(vectors).any() or np.isinf(vectors).any():
                logger.error("CPU 벡터 검증 실패: NaN 또는 Inf 값이 존재합니다.")
                return False
            logger.info("CPU 벡터 검증 완료.")

        return True

    def check_essential_features(self, feature_names: List[str]) -> bool:
        """필수 특성 존재 여부 확인"""
        missing = [f for f in ESSENTIAL_FEATURES if f not in feature_names]
        if missing:
            logger.error(f"필수 특성 누락: {missing}")
            return False
        return True


# 기존 함수들은 필요시 이 클래스를 사용하도록 리팩토링하거나, 유지/제거
# 예시: get_validator() 팩토리 함수
_validator_instance: Optional[GPUFeatureValidator] = None
_lock = threading.Lock()


def get_feature_validator() -> GPUFeatureValidator:
    """GPUFeatureValidator의 싱글톤 인스턴스를 반환합니다."""
    global _validator_instance
    with _lock:
        if _validator_instance is None:
            _validator_instance = GPUFeatureValidator()
    return _validator_instance


def validate_feature_vector_with_config(
    config: Dict[str, Any], names_file_path: str
) -> List[str]:
    """설정 기반 특성 벡터 검증"""
    try:
        with open(names_file_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        missing_features = []
        for essential in ESSENTIAL_FEATURES:
            if essential not in feature_names:
                missing_features.append(essential)

        return missing_features
    except Exception as e:
        logger.error(f"특성 벡터 검증 실패: {e}")
        return []


def check_vector_dimensions(
    vector_path: str,
    names_path: str,
    raise_on_mismatch: bool = True,
    allow_mismatch: bool = False,
) -> bool:
    """벡터 차원과 특성 이름 수 일치 확인"""
    try:
        vectors = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            names = json.load(f)

        if vectors.shape[1] != len(names):
            error_msg = f"벡터 차원({vectors.shape[1]})과 특성 이름 수({len(names)})가 일치하지 않습니다"
            if raise_on_mismatch and not allow_mismatch:
                raise ValueError(error_msg)
            logger.warning(error_msg)
            return False

        return True
    except Exception as e:
        logger.error(f"차원 확인 실패: {e}")
        if raise_on_mismatch:
            raise
        return False


def create_feature_registry(
    config: Dict[str, Any], registry_path: str
) -> Dict[str, Any]:
    """특성 레지스트리 생성"""
    registry = {
        "essential_features": ESSENTIAL_FEATURES,
        "created_at": str(Path().cwd()),
        "version": "1.0",
    }

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    return registry


def check_feature_mapping_consistency(
    feature_names: List[str], registry: Dict[str, Any]
) -> List[str]:
    """특성 매핑 일관성 확인"""
    inconsistencies = []
    essential = registry.get("essential_features", [])

    for name in feature_names:
        if name not in essential and not any(
            prefix in name
            for prefix in ["frequency_", "recency_", "pattern_", "cluster_"]
        ):
            inconsistencies.append(name)

    return inconsistencies


def sync_vectors_and_names(vector_path: str, names_path: str) -> bool:
    """벡터와 이름 동기화"""
    try:
        return check_vector_dimensions(vector_path, names_path, raise_on_mismatch=False)
    except Exception as e:
        logger.error(f"동기화 실패: {e}")
        return False


def ensure_essential_features(feature_names: List[str]) -> List[str]:
    """필수 특성 보장"""
    result = feature_names.copy()
    for essential in ESSENTIAL_FEATURES:
        if essential not in result:
            result.append(essential)
    return result


def safe_float_conversion(value: Any) -> float:
    """안전한 float 변환"""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            return float(value)
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0


def detect_outliers(
    vector_path: str, names_path: str, z_threshold: float = 2.5
) -> Tuple[np.ndarray, List[int]]:
    """이상치 탐지"""
    try:
        vectors = np.load(vector_path)
        z_scores = np.abs(
            (vectors - np.mean(vectors, axis=0)) / (np.std(vectors, axis=0) + 1e-8)
        )
        outlier_mask = np.any(z_scores > z_threshold, axis=1)
        outlier_indices = np.where(outlier_mask)[0].tolist()

        return outlier_mask, outlier_indices
    except Exception as e:
        logger.error(f"이상치 탐지 실패: {e}")
        return np.array([]), []


def save_outlier_information(
    vector_path: str, outlier_mask: np.ndarray, outlier_indices: List[int]
) -> Tuple[str, str]:
    """이상치 정보 저장"""
    base_path = Path(vector_path).parent
    mask_path = base_path / "outlier_mask.npy"
    indices_path = base_path / "outlier_indices.json"

    np.save(mask_path, outlier_mask)
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(outlier_indices, f)

    return str(mask_path), str(indices_path)


def analyze_vector_statistics(vector_path: str, names_path: str) -> Dict[str, Any]:
    """벡터 통계 분석"""
    try:
        vectors = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            names = json.load(f)

        stats = {
            "feature_count": len(names),
            "sample_count": vectors.shape[0],
            "nan_rate": float(np.isnan(vectors).sum() / vectors.size),
            "vector_scale_mean": float(np.mean(np.abs(vectors))),
            "vector_scale_std": float(np.std(np.abs(vectors))),
            "feature_entropy_score": float(np.mean(np.var(vectors, axis=0))),
        }

        return stats
    except Exception as e:
        logger.error(f"통계 분석 실패: {e}")
        return {}
