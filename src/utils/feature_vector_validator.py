"""
특성 벡터 검증 모듈 (복구)

벡터 차원과 특성 이름의 일치성을 검증하는 모듈입니다.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from .error_handler_refactored import get_logger

logger = get_logger(__name__)

# 필수 특성 목록 (22개)
ESSENTIAL_FEATURES = [
    "gap_stddev",
    "pair_centrality",
    "hot_cold_mix_score",
    "segment_entropy",
    "position_entropy_1",
    "position_entropy_2",
    "position_entropy_3",
    "position_entropy_4",
    "position_entropy_5",
    "position_entropy_6",
    "position_std_1",
    "position_std_2",
    "position_std_3",
    "position_std_4",
    "position_std_5",
    "position_std_6",
    "distance_variance",
    "cohesiveness_score",
    "sequential_pair_rate",
    "number_spread",
    "pattern_complexity",
    "trend_strength",
]


def check_vector_dimensions(
    vector_path: str, names_path: str, raise_on_mismatch: bool = True
) -> bool:
    """벡터 차원과 특성 이름 수 일치 확인"""
    try:
        vector = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            names = json.load(f)

        vector_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]
        names_count = len(names)

        if vector_dim != names_count:
            error_msg = f"차원 불일치: 벡터({vector_dim}) != 이름({names_count})"
            if raise_on_mismatch:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)
                return False

        logger.info(f"✅ 벡터 차원 검증 통과: {vector_dim}차원")
        return True

    except Exception as e:
        logger.error(f"벡터 차원 검증 실패: {e}")
        if raise_on_mismatch:
            raise
        return False


def validate_essential_features(feature_names: List[str]) -> List[str]:
    """필수 특성 검증"""
    missing_features = []
    for essential in ESSENTIAL_FEATURES:
        if essential not in feature_names:
            missing_features.append(essential)

    if missing_features:
        logger.warning(f"누락된 필수 특성: {missing_features}")
    else:
        logger.info("✅ 모든 필수 특성 존재")

    return missing_features


def analyze_vector_quality(vector_path: str) -> Dict[str, float]:
    """벡터 품질 분석"""
    try:
        vector = np.load(vector_path)
        if vector.ndim > 1:
            vector = vector.flatten()

        # 0값 비율 계산
        zero_ratio = np.sum(vector == 0) / len(vector)

        # 엔트로피 계산
        hist, _ = np.histogram(vector, bins=20)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0

        # 분산 계산
        variance = float(np.var(vector))

        quality_metrics = {
            "zero_ratio": float(zero_ratio),
            "entropy": float(entropy),
            "variance": variance,
            "mean": float(np.mean(vector)),
            "std": float(np.std(vector)),
        }

        logger.info(
            f"벡터 품질 분석 완료: 0값비율={zero_ratio:.3f}, 엔트로피={entropy:.3f}"
        )
        return quality_metrics

    except Exception as e:
        logger.error(f"벡터 품질 분석 실패: {e}")
        return {}


def fix_vector_dimension_mismatch(
    vector_path: str, names_path: str, output_vector_path: str, output_names_path: str
) -> bool:
    """벡터 차원 불일치 수정"""
    try:
        vector = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            names = json.load(f)

        vector_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]
        names_count = len(names)

        if vector_dim == names_count:
            logger.info("차원이 이미 일치합니다")
            return True

        # 차원 조정
        if vector_dim > names_count:
            # 이름 추가
            for i in range(names_count, vector_dim):
                names.append(f"feature_{i}")
            logger.info(f"특성 이름 {vector_dim - names_count}개 추가")
        else:
            # 벡터 확장
            if vector.ndim == 1:
                extended_vector = np.zeros(names_count, dtype=vector.dtype)
                extended_vector[:vector_dim] = vector
            else:
                extended_vector = np.zeros(
                    (vector.shape[0], names_count), dtype=vector.dtype
                )
                extended_vector[:, :vector_dim] = vector

            vector = extended_vector
            logger.info(f"벡터 차원 {names_count - vector_dim}개 확장")

        # 저장
        np.save(output_vector_path, vector)
        with open(output_names_path, "w", encoding="utf-8") as f:
            json.dump(names, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 차원 불일치 수정 완료: {len(names)}차원")
        return True

    except Exception as e:
        logger.error(f"차원 불일치 수정 실패: {e}")
        return False
