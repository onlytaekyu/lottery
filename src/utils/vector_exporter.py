#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
특성 벡터 내보내기 유틸리티

이 모듈은 특성 벡터와 관련 메타데이터를 표준화된 방식으로 저장하는 함수를 제공합니다.
벡터 데이터, 특성 이름, 인덱스 매핑 등을 일관된 방식으로 관리합니다.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


def save_feature_names(feature_names: List[str], output_path: str) -> bool:
    """
    특성 이름 목록을 JSON 파일로 저장합니다.

    Args:
        feature_names: 특성 이름 목록
        output_path: 출력 파일 경로

    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 디렉토리 확인 및 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 임시 파일에 먼저 저장 (원자적 작업을 위해)
        temp_path = f"{output_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)

        # 임시 파일을 실제 파일로 이동
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)

        logger.info(f"특성 이름 {len(feature_names)}개 저장 완료: {output_path}")
        return True
    except Exception as e:
        logger.error(f"특성 이름 저장 중 오류 발생: {e}")
        # 임시 파일 정리
        if os.path.exists(f"{output_path}.tmp"):
            try:
                os.remove(f"{output_path}.tmp")
            except:
                pass
        return False


def save_feature_index_mapping(feature_names: List[str], output_path: str) -> bool:
    """
    특성 이름과 인덱스의 매핑을 JSON 파일로 저장합니다.

    Args:
        feature_names: 특성 이름 목록
        output_path: 출력 파일 경로

    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 디렉토리 확인 및 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 인덱스 매핑 생성
        index_mapping = {
            "index_to_name": {str(i): name for i, name in enumerate(feature_names)},
            "name_to_index": {name: str(i) for i, name in enumerate(feature_names)},
            "timestamp": datetime.now().isoformat(),
            "feature_count": len(feature_names),
        }

        # 임시 파일에 먼저 저장
        temp_path = f"{output_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(index_mapping, f, ensure_ascii=False, indent=2)

        # 임시 파일을 실제 파일로 이동
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)

        logger.info(f"특성 인덱스 매핑 저장 완료: {output_path}")
        return True
    except Exception as e:
        logger.error(f"특성 인덱스 매핑 저장 중 오류 발생: {e}")
        # 임시 파일 정리
        if os.path.exists(f"{output_path}.tmp"):
            try:
                os.remove(f"{output_path}.tmp")
            except:
                pass
        return False


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
            # 1D 벡터인 경우 차원 확인
            if vector.shape[0] != len(feature_names):
                logger.warning(
                    f"벡터 크기({vector.shape[0]})와 특성 이름 수({len(feature_names)})가 일치하지 않습니다."
                )
                return False
        else:
            # 2D 벡터인 경우 열 수 확인
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

        # 2. 특성 이름 저장
        names_path = f"{base_path}.names.json"
        if not save_feature_names(feature_names, names_path):
            return False

        # 3. 인덱스 매핑 저장 (선택 사항)
        if save_index_map:
            index_path = f"{base_path}.index.json"
            if not save_feature_index_mapping(feature_names, index_path):
                logger.warning("인덱스 매핑 저장에 실패했으나 계속 진행합니다.")

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


def save_vector_bundle(
    vector: np.ndarray,
    feature_names: List[str],
    base_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    apply_low_variance_filter: bool = True,
    low_variance_path: str = "data/cache/low_variance_features.json",
) -> bool:
    """
    벡터, 이름, 인덱스, 메타데이터를 한 번에 저장하고, 필요 시 필터링 버전도 생성합니다.

    Args:
        vector: 원본 벡터 (1D or 2D)
        feature_names: 특성 이름 목록
        base_path: 저장 경로의 베이스 (확장자 제외)
        metadata: 메타데이터 (선택)
        apply_low_variance_filter: True일 경우 필터링된 벡터도 함께 저장
        low_variance_path: 필터 기준 JSON 경로

    Returns:
        저장 성공 여부 (True/False)
    """
    try:
        # 1. 벡터 차원 확인 및 유효성 검증
        if len(vector.shape) == 1:
            # 1D 벡터인 경우 차원 확인
            if vector.shape[0] != len(feature_names):
                logger.warning(
                    f"벡터 크기({vector.shape[0]})와 특성 이름 수({len(feature_names)})가 일치하지 않습니다."
                )
                return False
        else:
            # 2D 벡터인 경우 열 수 확인
            if vector.shape[1] != len(feature_names):
                logger.warning(
                    f"벡터 열 수({vector.shape[1]})와 특성 이름 수({len(feature_names)})가 일치하지 않습니다."
                )
                return False

        # 2. 디렉토리 확인 및 생성
        base_dir = os.path.dirname(base_path)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        # 3. 원본 벡터 저장
        ok1 = save_feature_vector_and_metadata(
            vector=vector,
            feature_names=feature_names,
            base_path=base_path,
            save_index_map=True,
            metadata=metadata,
        )

        if not ok1:
            logger.error(f"[save_vector_bundle] 원본 벡터 저장 실패")
            return False

        # 4. 필터링된 벡터도 저장 (선택적)
        if apply_low_variance_filter:
            # 저분산 특성 정보 파일 존재 여부 확인
            if not os.path.exists(low_variance_path):
                logger.warning(
                    f"저분산 특성 정보 파일이 존재하지 않습니다: {low_variance_path}"
                )
                # 저장은 성공한 것으로 간주 (필터링만 건너뜀)
                return True

            try:
                # 저분산 특성 정보 로드
                with open(low_variance_path, "r", encoding="utf-8") as f:
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

                    filtered_metadata = {
                        "original_feature_count": len(feature_names),
                        "filtered_feature_count": len(filtered_names),
                        "removed_feature_count": len(removed_feature_names),
                        "removed_features": removed_feature_names,
                    }

                    # 원본 메타데이터가 있으면 병합
                    if metadata:
                        filtered_metadata.update(metadata)

                    # 필터링된 벡터 저장
                    ok2 = save_feature_vector_and_metadata(
                        vector=filtered_vector,
                        feature_names=filtered_names,
                        base_path=filtered_base_path,
                        save_index_map=True,
                        metadata=filtered_metadata,
                    )

                    if ok2:
                        logger.info(
                            f"필터링된 벡터 저장 완료: {filtered_base_path}.npy "
                            f"(원본: {len(feature_names)}개 → 필터링: {len(filtered_names)}개 특성)"
                        )
                    else:
                        logger.warning(f"필터링된 벡터 저장 실패")
            except Exception as e:
                logger.warning(f"저분산 필터링 적용 중 오류 발생: {e}")
                # 원본 벡터는 저장됐으므로 성공으로 간주
                return True

        logger.info(f"[save_vector_bundle] 벡터 번들 저장 완료: {base_path}")
        return True

    except Exception as e:
        logger.error(f"[save_vector_bundle] 저장 실패: {str(e)}")
        return False


def load_feature_vector(
    base_path: str = "data/cache/feature_vector_full",
    use_filtered: Optional[bool] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    특성 벡터와 특성 이름을 로드합니다.
    설정에 따라 필터링된 벡터 또는 원본 벡터를 로드합니다.

    Args:
        base_path: 벡터 파일의 기본 경로 (확장자 제외)
        use_filtered: 필터링된 벡터를 사용할지 여부 (None이면 config에서 확인)
        config: 설정 사전 (use_filtered가 None인 경우 사용)

    Returns:
        (특성 벡터, 특성 이름 목록) 튜플

    Raises:
        FileNotFoundError: 벡터 파일이 존재하지 않는 경우
        ValueError: 벡터와 특성 이름 차원이 일치하지 않는 경우
        RuntimeError: 설정 키가 누락된 경우
    """
    # 설정에서 필터링된 벡터 사용 여부 확인
    if use_filtered is None and config is not None:
        try:
            use_filtered = config["training"]["use_filtered_vector"]
        except KeyError as e:
            logger.error(f"[ERROR] 설정 키 누락: {str(e)}")
            raise RuntimeError("설정 키 누락으로 프로세스를 종료합니다.")

    # 기본값 설정 (use_filtered가 None이고 config도 None인 경우에만 기본값 사용)
    if use_filtered is None:
        logger.error(
            "[ERROR] 필터링 설정이 제공되지 않고 config에서도 찾을 수 없습니다."
        )
        raise RuntimeError("필터링 설정 누락으로 프로세스를 종료합니다.")

    # 파일 경로 설정
    vector_path = f"{base_path}_filtered.npy" if use_filtered else f"{base_path}.npy"
    names_path = (
        f"{base_path}_filtered.names.json"
        if use_filtered
        else f"{base_path}.names.json"
    )

    # 필터링된 벡터가 없는 경우 원본으로 폴백
    if use_filtered and not os.path.exists(vector_path):
        logger.warning(
            f"필터링된 벡터 파일({vector_path})이 존재하지 않습니다. 원본 벡터로 대체합니다."
        )
        vector_path = f"{base_path}.npy"
        names_path = f"{base_path}.names.json"

    # 벡터 로드
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"벡터 파일({vector_path})이 존재하지 않습니다.")

    vector = np.load(vector_path)

    # 특성 이름 로드
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"특성 이름 파일({names_path})이 존재하지 않습니다.")

    with open(names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # 차원 검증
    expected_dim = len(feature_names)
    actual_dim = vector.shape[1] if len(vector.shape) > 1 else vector.shape[0]

    if actual_dim != expected_dim:
        raise ValueError(
            f"벡터 차원({actual_dim})과 특성 이름 수({expected_dim})가 일치하지 않습니다."
        )

    logger.info(
        f"특성 벡터 로드 완료: {vector_path}, 형태={vector.shape}, 특성 수={len(feature_names)}"
    )
    return vector, feature_names
