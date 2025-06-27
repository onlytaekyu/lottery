#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
특성 벡터 내보내기 유틸리티

이 모듈은 특성 벡터와 관련 메타데이터를 표준화된 방식으로 저장하는 함수를 제공합니다.
벡터 데이터, 특성 이름, 인덱스 매핑 등을 일관된 방식으로 관리합니다.

🔧 중복 함수 통합:
- save_feature_names: feature_name_tracker.py에서 재사용
- save_feature_index_mapping: feature_name_tracker.py에서 재사용
- load_feature_names: feature_name_tracker.py에서 재사용
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..utils.unified_logging import get_logger

# 🔧 중복된 기능 통합 - 기존 함수 재사용
from .feature_name_tracker import (
    save_feature_names,
    load_feature_names,
    save_feature_index_mapping,
)

logger = get_logger(__name__)

def save_feature_vector_and_metadata(
    vector: np.ndarray,
    feature_names: List[str],
    base_path: str,
    save_index_map: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    특성 벡터와 관련 메타데이터를 표준화된 경로에 저장합니다.

    Args:
        vector: 특성 벡터 (2D 또는 1D 배열)
        feature_names: 특성 이름 목록
        base_path: 기본 파일 경로 (확장자 제외)
        save_index_map: 인덱스 매핑 저장 여부
        metadata: 추가 메타데이터 (선택 사항)

    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 벡터 차원 확인 및 유효성 검증
        if len(vector.shape) == 1:
            if vector.shape[0] != len(feature_names):
                logger.warning(
                    f"벡터 크기({vector.shape[0]})와 특성 이름 수({len(feature_names)})가 일치하지 않습니다."
                )
                return False
        else:
            if vector.shape[1] != len(feature_names):
                logger.warning(
                    f"벡터 열 수({vector.shape[1]})와 특성 이름 수({len(feature_names)})가 일치하지 않습니다."
                )
                return False

        # 디렉토리 확인 및 생성
        base_dir = os.path.dirname(base_path)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        # 1. 벡터 저장
        vector_path = f"{base_path}.npy"
        try:
            np.save(vector_path, vector)
            logger.info(f"벡터 저장 완료: {vector_path}, 형태: {vector.shape}")
        except Exception as e:
            logger.error(f"벡터 저장 중 오류 발생: {e}")
            return False

        # 2. 특성 이름 저장 (기존 함수 재사용)
        names_path = f"{base_path}.names.json"
        try:
            save_feature_names(feature_names, names_path)
        except Exception as e:
            logger.error(f"특성 이름 저장 실패: {e}")
            return False

        # 3. 인덱스 매핑 저장 (선택 사항, 기존 함수 재사용)
        if save_index_map:
            index_path = f"{base_path}.index.json"
            try:
                save_feature_index_mapping(feature_names, index_path)
            except Exception as e:
                logger.warning(f"인덱스 매핑 저장에 실패했으나 계속 진행합니다: {e}")

        # 4. 추가 메타데이터 저장 (선택 사항)
        if metadata:
            meta_path = f"{base_path}.meta.json"
            try:
                # 기본 메타데이터 추가
                metadata.update(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "vector_shape": list(vector.shape),
                        "feature_count": len(feature_names),
                    }
                )

                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                logger.info(f"메타데이터 저장 완료: {meta_path}")
            except Exception as e:
                logger.warning(f"메타데이터 저장 중 오류 발생: {e}")

        return True
    except Exception as e:
        logger.error(f"벡터 및 메타데이터 저장 중 오류 발생: {e}")
        return False

def save_vector_bundle(
    vector: np.ndarray,
    feature_names: List[str],
    base_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    apply_low_variance_filter: bool = True,
    low_variance_path: str = "data/cache/low_variance_features.json",
) -> bool:
    """
    벡터 번들 저장 (기존 함수 재사용)

    Args:
        vector: 저장할 벡터
        feature_names: 특성 이름 목록
        base_path: 기본 저장 경로
        metadata: 추가 메타데이터
        apply_low_variance_filter: 저분산 필터 적용 여부
        low_variance_path: 저분산 특성 정보 파일 경로

    Returns:
        저장 성공 여부
    """
    try:
        # 기존 함수 재사용하여 벡터와 메타데이터 저장
        return save_feature_vector_and_metadata(
            vector=vector,
            feature_names=feature_names,
            base_path=base_path,
            save_index_map=True,
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"벡터 번들 저장 실패: {e}")
        return False

def load_feature_vector(
    base_path: str = "data/cache/feature_vector_full",
    use_filtered: Optional[bool] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    특성 벡터와 이름을 로드합니다.

    Args:
        base_path: 기본 파일 경로 (확장자 제외)
        use_filtered: 필터링된 벡터 사용 여부
        config: 설정 객체

    Returns:
        (벡터, 특성 이름 목록) 튜플

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        ValueError: 벡터와 특성 이름 수가 일치하지 않을 때
    """
    try:
        # 파일 경로 결정
        if use_filtered:
            vector_path = f"{base_path}_filtered.npy"
            names_path = f"{base_path}_filtered.names.json"
        else:
            vector_path = f"{base_path}.npy"
            names_path = f"{base_path}.names.json"

        # 벡터 로드
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"벡터 파일이 존재하지 않습니다: {vector_path}")

        vector = np.load(vector_path)

        # 특성 이름 로드 (기존 함수 재사용)
        feature_names = load_feature_names(names_path)

        # 차원 일치 확인
        expected_features = (
            vector.shape[1] if len(vector.shape) > 1 else vector.shape[0]
        )
        if expected_features != len(feature_names):
            raise ValueError(
                f"벡터 특성 수({expected_features})와 특성 이름 수({len(feature_names)})가 일치하지 않습니다."
            )

        logger.info(
            f"특성 벡터 로드 완료: {vector.shape}, 특성 수: {len(feature_names)}"
        )
        return vector, feature_names

    except Exception as e:
        logger.error(f"특성 벡터 로드 실패: {e}")
        raise

def export_vector_with_filtering(
    vector: np.ndarray,
    feature_names: List[str],
    base_path: str,
    low_var_path: Optional[str] = "data/cache/low_variance_features.json",
    save_filtered: bool = True,
) -> bool:
    """
    원본 벡터와 저분산 필터링된 벡터를 함께 저장합니다.

    Args:
        vector: 특성 벡터 (2D 또는 1D 배열)
        feature_names: 특성 이름 목록
        base_path: 기본 파일 경로 (확장자 제외)
        low_var_path: 저분산 특성 정보 파일 경로
        save_filtered: 필터링된 벡터 저장 여부

    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 1. 원본 벡터 및 메타데이터 저장
        if not save_feature_vector_and_metadata(
            vector=vector,
            feature_names=feature_names,
            base_path=base_path,
            save_index_map=True,
        ):
            return False

        # 2. 저분산 필터링 적용 (선택 사항)
        if save_filtered and low_var_path and os.path.exists(low_var_path):
            try:
                # 저분산 특성 정보 로드
                with open(low_var_path, "r", encoding="utf-8") as f:
                    low_var_info = json.load(f)

                removed_feature_names = low_var_info.get("removed_feature_names", [])

                if removed_feature_names:
                    # 필터 마스크 생성
                    mask = [name not in removed_feature_names for name in feature_names]

                    # 필터링된 특성 이름
                    filtered_names = [
                        name for name, keep in zip(feature_names, mask) if keep
                    ]

                    # 벡터 필터링
                    if len(vector.shape) == 1:
                        # 1D 벡터
                        filtered_vector = vector[mask]
                    else:
                        # 2D 벡터
                        filtered_vector = vector[:, mask]

                    # 필터링된 벡터 및 메타데이터 저장
                    filtered_base_path = f"{base_path}_filtered"

                    metadata = {
                        "original_feature_count": len(feature_names),
                        "filtered_feature_count": len(filtered_names),
                        "removed_feature_count": len(removed_feature_names),
                        "removed_features": removed_feature_names,
                    }

                    save_feature_vector_and_metadata(
                        vector=filtered_vector,
                        feature_names=filtered_names,
                        base_path=filtered_base_path,
                        save_index_map=True,
                        metadata=metadata,
                    )

                    logger.info(
                        f"필터링된 벡터 저장 완료: {filtered_base_path}.npy "
                        f"(원본: {len(feature_names)}개 → 필터링: {len(filtered_names)}개 특성)"
                    )
            except Exception as e:
                logger.warning(f"저분산 필터링 적용 중 오류 발생: {e}")

        return True
    except Exception as e:
        logger.error(f"벡터 내보내기 중 오류 발생: {e}")
        return False

def export_gnn_state_inputs(
    pair_graph_vector: np.ndarray,
    feature_names: List[str],
    base_path: str = "data/cache/pair_graph_compressed_vector",
) -> bool:
    """
    GNN 입력용 그래프 벡터를 저장합니다.

    Args:
        pair_graph_vector: 쌍 그래프 벡터
        feature_names: 특성 이름 목록
        base_path: 기본 파일 경로 (확장자 제외)

    Returns:
        bool: 저장 성공 여부
    """
    return save_feature_vector_and_metadata(
        vector=pair_graph_vector,
        feature_names=feature_names,
        base_path=base_path,
        save_index_map=True,
        metadata={
            "vector_type": "graph_structure",
            "description": "GNN 모델 및 강화학습 에이전트 입력용 그래프 구조 벡터",
        },
    )
