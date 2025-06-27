"""
통합 특성 벡터 검증기

특성 벡터의 유효성을 검증하는 통합 모듈입니다.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from .error_handler_refactored import get_logger

logger = get_logger(__name__)

# 필수 특성 목록
ESSENTIAL_FEATURES = [
    "frequency_mean",
    "frequency_std",
    "recency_mean",
    "recency_std",
    "pattern_score",
    "cluster_score",
    "trend_score",
    "pair_score",
]

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
