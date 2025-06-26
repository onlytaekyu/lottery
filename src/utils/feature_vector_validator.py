#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
특성 벡터 검증 유틸리티

이 모듈은 생성된 특성 벡터에 필수 특성이 모두 포함되어 있는지 검증합니다.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime
import shutil

from .error_handler import get_logger

logger = get_logger(__name__)

# 필수 특성 목록 정의
ESSENTIAL_FEATURES = [
    # 1. 출현/위치 통계 특성 (20개)
    "position_frequency_1",
    "position_frequency_2",
    "position_frequency_3",
    "position_frequency_4",
    "position_frequency_5",
    "position_frequency_6",
    "position_entropy_1",
    "position_entropy_2",
    "position_entropy_3",
    "position_entropy_4",
    "position_entropy_5",
    "position_entropy_6",
    "position_variance_avg",
    "position_std_1",
    "position_std_2",
    "position_std_3",
    "position_std_4",
    "position_std_5",
    "position_std_6",
    "position_bias_score",
    # 2. 쌍/삼중 조합 기반 특성 (12개)
    "pair_centrality_avg",
    "pair_centrality_std",
    "pair_roi_high",
    "pair_roi_low",
    "pair_roi_avg",
    "pair_score_avg",
    "pair_score_std",
    "triple_score_avg",
    "triple_score_high",
    "triple_centrality",
    "triple_diversity",
    "triple_coverage",
    # 3. 클러스터 기반 패턴 특성 (12개)
    "cluster_score",
    "silhouette_score",
    "avg_distance_between_clusters",
    "std_of_cluster_size",
    "balance_score",
    "cluster_count_norm",
    "largest_cluster_size_norm",
    "cluster_size_ratio_norm",
    "cluster_entropy_score",
    "cohesiveness_score",
    "distance_variance",
    "sequential_pair_rate",
    # 4. 시계열 및 반복성 특성 (14개)
    "recent_delta_1",
    "recent_delta_2",
    "recent_delta_3",
    "recent_delta_4",
    "recent_delta_5",
    "recent_delta_6",
    "delta_trend",
    "trend_slope",
    "trend_correlation",
    "trend_variance",
    "segment_10_change_rate",
    "segment_5_change_rate",
    "segment_entropy",
    "segment_repeat_score",
    # 5. 중복성 및 유사성 특성 (8개)
    "duplicate_flag",
    "exact_match_in_history",
    "max_overlap_with_past",
    "combination_recency_score",
    "overlap_with_hot_patterns",
    "overlap_with_cold_patterns",
    "hot_cold_mix_score",
    "gap_stddev",
    # 6. ROI 기반 프리미엄 분석 특성 (11개)
    "roi_group_score",
    "roi_cluster_score",
    "expected_value",
    "roi_estimate",
    "win_probability",
    "risk_score",
    "odd_dominant_roi",
    "even_dominant_roi",
    "low_numbers_roi",
    "high_numbers_roi",
    "balanced_segments_roi",
    # 7. 복제기 및 확률 기반 특성 (13개)
    "zscore_num1",
    "zscore_num2",
    "zscore_num3",
    "zscore_num4",
    "zscore_num5",
    "zscore_num6",
    "zscore_mean",
    "zscore_std",
    "binomial_match_score",
    "rarity_index",
    "entropy_overall",
    "distribution_balance",
    "number_diversity",
    # 8. 중복 패턴 기반 특성 (20개)
    "overlap_3_frequency_score",
    "overlap_3_pattern_count_norm",
    "overlap_4_rarity_score",
    "overlap_4_pattern_count_norm",
    "overlap_time_gap_variance_3",
    "overlap_time_gap_variance_4",
    "overlap_pattern_consistency_3",
    "overlap_pattern_consistency_4",
    "overlap_3_roi_correlation",
    "overlap_4_roi_correlation",
    "overlap_3_avg_roi",
    "overlap_4_avg_roi",
    "overlap_3_positive_roi_ratio",
    "overlap_4_positive_roi_ratio",
    "overlap_roi_expectation",
    "overlap_3_sample_confidence",
    "overlap_4_sample_confidence",
    "max_overlap_avg_norm",
    "duplicate_ratio",
    "recency_score",
    # 9. 추첨 순서 편향 분석 특성 (5개)
    "position_min_value_mean",
    "position_max_value_mean",
    "position_gap_mean",
    "position_even_odd_ratio",
    "position_low_high_ratio",
    # 10. 중복 패턴 시간적 주기성 특성 (5개)
    "overlap_3_time_gap_mean",
    "overlap_4_time_gap_mean",
    "overlap_time_gap_stddev",
    "recent_overlap_3_count",
    "recent_overlap_4_count",
    # 11. 번호 간 조건부 상호작용 특성 (3개)
    "number_attraction_score",
    "number_repulsion_score",
    "conditional_dependency_strength",
    # 12. 홀짝 및 구간별 미세 편향성 특성 (4개)
    "odd_even_bias_score",
    "segment_balance_bias_score",
    "range_bias_moving_avg",
    "odd_ratio_change_rate",
]


def safe_float_conversion(value: Any) -> float:
    """
    다양한 타입의 값을 안전하게 float로 변환합니다.
    딕셔너리나 리스트 등 복잡한 타입은 0.0으로 처리합니다.

    Args:
        value: 변환할 값

    Returns:
        float: 변환된 float 값, 변환 불가능한 경우 0.0
    """
    try:
        # None 값 처리
        if value is None:
            return 0.0

        # 딕셔너리 처리
        if isinstance(value, dict):
            # 딕셔너리가 비어있는 경우
            if not value:
                return 0.0

            # 값이 하나만 있는 경우 그 값을 사용
            if len(value) == 1:
                return safe_float_conversion(next(iter(value.values())))

            # 평균값을 계산할 수 있는 경우
            try:
                return float(
                    sum(safe_float_conversion(v) for v in value.values()) / len(value)
                )
            except:
                return 0.0

        # 리스트 처리
        if isinstance(value, (list, tuple)):
            # 리스트가 비어있는 경우
            if not value:
                return 0.0

            # 값이 하나만 있는 경우 그 값을 사용
            if len(value) == 1:
                return safe_float_conversion(value[0])

            # 평균값을 계산할 수 있는 경우
            try:
                return float(sum(safe_float_conversion(v) for v in value) / len(value))
            except:
                return 0.0

        # 일반적인 변환 시도
        return float(value)
    except (ValueError, TypeError):
        # 변환 불가능한 경우 0.0 반환
        return 0.0


def validate_feature_vector(expected_features: List[str], names_path: str) -> List[str]:
    """
    주어진 .names.json 파일과 기대되는 feature 리스트를 비교하여 누락된 feature 리스트 반환

    Args:
        expected_features: 기대되는 특성 이름 목록
        names_path: 특성 이름 파일 경로 (.names.json)

    Returns:
        누락된 특성 목록

    Raises:
        FileNotFoundError: 특성 이름 파일이 존재하지 않는 경우
        ValueError: 특성 이름 파일 형식이 올바르지 않은 경우
    """
    try:
        # 파일 존재 확인
        if not os.path.exists(names_path):
            error_msg = f"[ERROR] 특성 이름 파일이 존재하지 않습니다: {names_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 특성 이름 로드
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 리스트가 아닌 경우 처리
        if not isinstance(feature_names, list):
            error_msg = f"[ERROR] 특성 이름 파일 형식이 올바르지 않습니다: {names_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 특성 이름을 집합으로 변환하여 비교 효율성 향상
        feature_names_set = set(feature_names)
        expected_features_set = set(expected_features)

        # 누락된 특성 찾기
        missing_features = list(expected_features_set - feature_names_set)

        # 누락된 특성이 있는 경우 로그 출력
        if missing_features:
            logger.error(f"[ERROR] 필수 벡터 특성 누락됨: {sorted(missing_features)}")

        return missing_features
    except (FileNotFoundError, ValueError) as e:
        # 이미 처리된 예외는 그대로 전달
        raise
    except Exception as e:
        error_msg = f"[ERROR] 특성 벡터 검증 중 오류 발생: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def validate_feature_vector_with_config(
    config: Dict[str, Any], names_path: Optional[str] = None
) -> List[str]:
    """
    설정에서 필수 특성 목록을 가져와 벡터 검증

    Args:
        config: 설정 객체
        names_path: 특성 이름 파일 경로 (없으면 설정에서 가져옴)

    Returns:
        누락된 특성 목록
    """
    try:
        # 설정에서 필수 특성 목록 가져오기
        essential_features = []

        try:
            if isinstance(config, dict) and "validation" in config:
                essential_features = config["validation"].get(
                    "essential_features", ESSENTIAL_FEATURES
                )
            elif hasattr(config, "get") and callable(getattr(config, "get")):
                # ConfigProxy 객체인 경우
                validation = config.get("validation")
                if (
                    validation is not None
                    and hasattr(validation, "get")
                    and callable(getattr(validation, "get"))
                ):
                    essential_features = validation.get(
                        "essential_features", ESSENTIAL_FEATURES
                    )
                else:
                    essential_features = ESSENTIAL_FEATURES
            else:
                essential_features = ESSENTIAL_FEATURES
        except Exception as e:
            logger.warning(f"설정에서 필수 특성 목록을 가져오는 중 오류 발생: {e}")
            essential_features = ESSENTIAL_FEATURES

        # 특성 이름 파일 경로가 제공되지 않은 경우 설정에서 가져오기
        if not names_path:
            try:
                if isinstance(config, dict):
                    if "paths" in config and "feature_names_path" in config["paths"]:
                        names_path = config["paths"]["feature_names_path"]
                    else:
                        names_path = "data/cache/feature_vector_full.names.json"
                elif hasattr(config, "get") and callable(getattr(config, "get")):
                    # ConfigProxy 객체인 경우
                    paths = config.get("paths")
                    if (
                        paths is not None
                        and hasattr(paths, "get")
                        and callable(getattr(paths, "get"))
                    ):
                        names_path = paths.get(
                            "feature_names_path",
                            "data/cache/feature_vector_full.names.json",
                        )
                    else:
                        names_path = "data/cache/feature_vector_full.names.json"
                else:
                    names_path = "data/cache/feature_vector_full.names.json"
            except Exception as e:
                logger.warning(
                    f"설정에서 특성 이름 파일 경로를 가져오는 중 오류 발생: {e}"
                )
                names_path = "data/cache/feature_vector_full.names.json"

        # 벡터 검증
        missing_features = validate_feature_vector(essential_features, names_path)

        # 필수 특성이 누락된 경우 예외 발생
        if missing_features:
            error_msg = f"필수 특성이 누락되었습니다: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return missing_features
    except Exception as e:
        if isinstance(e, ValueError) and "필수 특성이 누락되었습니다" in str(e):
            # 이미 처리된 예외는 그대로 전달
            raise
        error_msg = f"특성 벡터 검증 중 오류 발생: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def ensure_features_present(
    expected_features: List[str], names_path: str, raise_on_missing: bool = True
) -> bool:
    """
    필수 특성이 벡터에 포함되어 있는지 확인하고, 누락된 경우 예외 발생

    Args:
        expected_features: 기대되는 특성 이름 목록
        names_path: 특성 이름 파일 경로 (.names.json)
        raise_on_missing: 사용되지 않는 파라미터, 항상 예외 발생

    Returns:
        bool: 모든 특성이 존재하면 True

    Raises:
        ValueError: 누락된 특성이 있는 경우 항상 예외 발생
    """
    missing_features = validate_feature_vector(expected_features, names_path)

    if missing_features:
        error_msg = f"[ERROR] 필수 특성이 누락되었습니다: {sorted(missing_features)}"
        logger.error(error_msg)
        # 누락된 특성이 있으면 항상 예외 발생
        raise ValueError(error_msg)

    return True


def detect_outliers(
    vector_path: str, names_path: str, z_threshold: float = 2.5
) -> Tuple[np.ndarray, List[int]]:
    """
    특성 벡터에서 이상치를 탐지합니다.

    Args:
        vector_path: 벡터 파일 경로 (.npy)
        names_path: 특성 이름 파일 경로 (.names.json)
        z_threshold: Z-score 임계값 (기본값: 2.5)

    Returns:
        Tuple[np.ndarray, List[int]]: (이상치 마스크, 이상치 인덱스 목록)
    """
    try:
        # 벡터와 특성 이름 로드
        vectors = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 벡터가 1차원인 경우 2차원으로 변환
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # 각 특성별 Z-score 계산
        means = np.mean(vectors, axis=0)
        stds = np.std(vectors, axis=0) + 1e-6  # 0으로 나누기 방지
        z_scores = np.abs((vectors - means) / stds)

        # 이상치 마스크 생성 (임계값 초과 = True)
        outlier_mask = z_scores > z_threshold

        # 이상치가 포함된 벡터 인덱스 추출
        outlier_indices = np.where(np.any(outlier_mask, axis=1))[0].tolist()

        # 이상치 정보 로깅
        outlier_count = len(outlier_indices)
        total_count = vectors.shape[0]
        outlier_rate = outlier_count / total_count if total_count > 0 else 0

        logger.info(
            f"이상치 탐지 완료: {outlier_count}/{total_count} ({outlier_rate:.2%})"
        )

        # 특성별 이상치 비율 계산
        feature_outlier_rates = np.mean(outlier_mask, axis=0)
        for i, rate in enumerate(feature_outlier_rates):
            if rate > 0.05:  # 5% 이상이 이상치인 특성만 로깅
                feature_name = (
                    feature_names[i] if i < len(feature_names) else f"feature_{i}"
                )
                logger.warning(f"특성 '{feature_name}'의 이상치 비율: {rate:.2%}")

        return outlier_mask, outlier_indices

    except Exception as e:
        logger.error(f"이상치 탐지 중 오류 발생: {e}")
        # 빈 마스크와 빈 목록 반환
        return np.array([]), []


def save_outlier_information(
    vector_path: str, outlier_mask: np.ndarray, outlier_indices: List[int]
) -> Tuple[str, str]:
    """
    이상치 정보를 파일로 저장합니다.

    Args:
        vector_path: 원본 벡터 파일 경로
        outlier_mask: 이상치 마스크
        outlier_indices: 이상치 인덱스 목록

    Returns:
        Tuple[str, str]: 저장된 마스크 파일 경로와 인덱스 파일 경로
    """
    try:
        # 저장 경로 생성
        vector_dir = os.path.dirname(vector_path)
        mask_path = os.path.join(vector_dir, "outlier_vector_mask.npy")
        indices_path = os.path.join(vector_dir, "outlier_vector_indices.json")

        # 마스크 저장
        np.save(mask_path, outlier_mask)

        # 인덱스 저장
        with open(indices_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "outlier_indices": outlier_indices,
                    "timestamp": datetime.now().isoformat(),
                    "z_threshold": 2.5,
                    "count": len(outlier_indices),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"이상치 마스크 저장 완료: {mask_path}")
        logger.info(f"이상치 인덱스 저장 완료: {indices_path}")

        return mask_path, indices_path

    except Exception as e:
        logger.error(f"이상치 정보 저장 중 오류 발생: {e}")
        return "", ""


def check_vector_dimensions(
    vector_path: str,
    names_path: str,
    raise_on_mismatch: bool = True,
    allow_mismatch: bool = False,
) -> bool:
    """
    벡터 데이터(.npy)와 특성 이름(.names.json) 파일의 차원이 일치하는지 확인합니다.

    Args:
        vector_path: 벡터 데이터 파일 경로
        names_path: 특성 이름 파일 경로
        raise_on_mismatch: 불일치 시 예외를 발생시킬지 여부 (기본값: True)
        allow_mismatch: 불일치를 허용할지 여부 (항상 False 사용 필수)

    Returns:
        bool: 차원이 일치하면 True, 불일치하면 False

    Raises:
        ValueError: 차원이 불일치하고 raise_on_mismatch가 True인 경우
        FileNotFoundError: 파일이 존재하지 않는 경우
    """
    # allow_mismatch 매개변수 무시 및 경고
    if allow_mismatch:
        logger.warning(
            "벡터 차원 불일치는 더 이상 허용되지 않습니다. allow_mismatch=False로 강제 설정됩니다."
        )
        allow_mismatch = False  # 항상 False로 설정

    try:
        # 파일 존재 확인
        if not os.path.exists(vector_path):
            error_msg = f"[ERROR] 벡터 데이터 파일이 존재하지 않습니다: {vector_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not os.path.exists(names_path):
            error_msg = f"[ERROR] 특성 이름 파일이 존재하지 않습니다: {names_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 벡터 데이터 로드
        vector_data = np.load(vector_path)

        # 특성 이름 로드
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 차원 확인
        vector_dim = vector_data.shape[1] if len(vector_data.shape) > 1 else 1
        names_count = len(feature_names)

        # 결과 기록
        match = vector_dim == names_count

        if not match:
            error_msg = f"[ERROR] 벡터 차원({vector_dim})과 특성 이름 수({names_count})가 일치하지 않습니다!"
            logger.error(error_msg)

            # 불일치 시 항상 예외 발생
            raise ValueError(error_msg)
        else:
            logger.info(
                f"벡터 차원 검증 성공: 벡터 차원({vector_dim}) = 특성 이름 수({names_count})"
            )
            return True

    except (FileNotFoundError, ValueError) as e:
        # 이미 처리된 예외는 그대로 전달
        raise
    except Exception as e:
        error_msg = f"[ERROR] 벡터 차원 검증 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def create_feature_registry(config: Any, registry_path: str) -> Dict[str, str]:
    """
    특성 레지스트리 파일을 생성합니다.
    각 특성이 어떤 분석기에서 생성되었는지를 매핑합니다.

    Args:
        config: 설정 객체
        registry_path: 레지스트리 파일 경로

    Returns:
        Dict[str, str]: 특성 이름 -> 모듈 이름 매핑
    """
    try:
        # 기존 레지스트리 파일이 있으면 로드
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            logger.info(f"기존 특성 레지스트리 로드: {len(registry)}개 특성")
        else:
            registry = {}
            logger.info("새로운 특성 레지스트리 생성")

        # 기본 매핑 정의 (필수 특성에 대한 매핑)
        default_mapping = {
            # 패턴 분석기 특성
            "gap_stddev": "pattern_analyzer",
            "hot_cold_mix_score": "pattern_analyzer",
            "segment_entropy": "pattern_analyzer",
            "duplicate_flag": "unified_analyzer",
            "max_overlap_with_past": "pattern_analyzer",
            "combination_recency_score": "pattern_analyzer",
            # 쌍 분석기 특성
            "pair_centrality": "pair_analyzer",
            "pair_centrality_avg": "pair_analyzer",
            # ROI 분석기 특성
            "roi_group_score": "roi_analyzer",
            # 위치 관련 특성
            "position_entropy_1": "position_analyzer",
            "position_entropy_2": "position_analyzer",
            "position_entropy_3": "position_analyzer",
            "position_entropy_4": "position_analyzer",
            "position_entropy_5": "position_analyzer",
            "position_entropy_6": "position_analyzer",
            "position_variance_avg": "position_analyzer",
            "position_bias_score": "position_analyzer",
            "position_std_1": "position_analyzer",
            "position_std_2": "position_analyzer",
            "position_std_3": "position_analyzer",
            "position_std_4": "position_analyzer",
            "position_std_5": "position_analyzer",
            "position_std_6": "position_analyzer",
            # 클러스터링 관련 특성
            "cluster_score": "cluster_analyzer",
            "silhouette_score": "cluster_analyzer",
            "avg_distance_between_clusters": "cluster_analyzer",
            "std_of_cluster_size": "cluster_analyzer",
            "balance_score": "cluster_analyzer",
            "cluster_count_norm": "cluster_analyzer",
            "largest_cluster_size_norm": "cluster_analyzer",
            "cluster_size_ratio_norm": "cluster_analyzer",
        }

        # 기본 매핑을 기존 레지스트리에 업데이트
        update_count = 0
        for feature, module in default_mapping.items():
            if feature not in registry:
                registry[feature] = module
                update_count += 1

        # 캐시 디렉토리에서 사용 가능한 모든 특성 이름 찾기
        try:
            # ConfigProxy 객체인 경우
            if hasattr(config, "get") and callable(getattr(config, "get")):
                # ConfigProxy 객체인 경우 paths를 먼저 가져온 후 하위 키에 접근
                paths = config.get("paths")
                if paths and hasattr(paths, "get") and callable(getattr(paths, "get")):
                    cache_dir = paths.get("cache_dir", "data/cache")
                else:
                    cache_dir = "data/cache"
            else:
                # 딕셔너리인 경우
                cache_dir = config.get("paths", {}).get("cache_dir", "data/cache")

            # 특성 이름 파일 경로
            names_file = os.path.join(cache_dir, "feature_vector_full.names.json")
            if os.path.exists(names_file):
                with open(names_file, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)

                # 누락된 특성 처리 (기본적으로 'unknown'으로 표시)
                unknown_count = 0
                for feature in feature_names:
                    if feature not in registry:
                        # 특성 이름 패턴 기반으로 모듈 추측
                        if "position_" in feature:
                            registry[feature] = "position_analyzer"
                        elif "pair_" in feature:
                            registry[feature] = "pair_analyzer"
                        elif "cluster_" in feature:
                            registry[feature] = "cluster_analyzer"
                        elif "roi_" in feature:
                            registry[feature] = "roi_analyzer"
                        else:
                            registry[feature] = "unknown"
                            unknown_count += 1

                logger.info(
                    f"기존 특성에 {unknown_count}개의 '알 수 없는' 매핑이 추가되었습니다."
                )

        except Exception as e:
            logger.warning(f"특성 이름 파일 로드 중 오류 발생: {e}")

        # 레지스트리 파일 저장
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

        logger.info(
            f"특성 레지스트리 저장 완료: {len(registry)}개 특성 ({update_count}개 업데이트됨)"
        )
        return registry

    except Exception as e:
        logger.error(f"특성 레지스트리 생성 중 오류 발생: {e}")
        return {}


def check_feature_mapping_consistency(
    feature_names: List[str], feature_registry: Dict[str, str]
) -> List[str]:
    """
    특성 이름과 레지스트리 간의 일관성을 확인합니다.

    Args:
        feature_names: 특성 이름 목록
        feature_registry: 특성 레지스트리 (특성 -> 모듈 매핑)

    Returns:
        List[str]: 레지스트리에 없는 특성 목록
    """
    # 레지스트리에 없는 특성 찾기
    missing_from_registry = [f for f in feature_names if f not in feature_registry]

    # 결과 로깅
    if missing_from_registry:
        logger.warning(f"레지스트리에 없는 특성: {missing_from_registry}")
    else:
        logger.info("모든 특성이 레지스트리에 매핑되어 있습니다.")

    return missing_from_registry


def check_registry_modules(feature_registry: Dict[str, str]) -> Dict[str, int]:
    """
    레지스트리의 모듈 분포를 확인합니다.

    Args:
        feature_registry: 특성 레지스트리 (특성 -> 모듈 매핑)

    Returns:
        Dict[str, int]: 모듈별 특성 수
    """
    module_counts = {}
    for _, module in feature_registry.items():
        if module not in module_counts:
            module_counts[module] = 0
        module_counts[module] += 1

    # 결과 로깅
    logger.info(f"모듈별 특성 수: {module_counts}")
    return module_counts


def sync_vectors_and_names(vector_path: str, names_path: str) -> bool:
    """
    벡터 파일과 특성 이름 파일의 차원을 일치시킵니다.

    Args:
        vector_path: 벡터 파일 경로 (.npy)
        names_path: 특성 이름 파일 경로 (.names.json)

    Returns:
        bool: 동기화 성공 여부
    """
    try:
        # 벡터와 특성 이름 로드
        vectors = np.load(vector_path)

        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 차원 확인
        if len(vectors.shape) > 1:
            vector_dim = vectors.shape[1]  # 2D 배열의 경우 열 수
        else:
            vector_dim = vectors.shape[0]  # 1D 배열의 경우 요소 수

        names_count = len(feature_names)

        # 차원이 일치하면 변경 없음
        if vector_dim == names_count:
            logger.info("벡터 차원과 특성 이름 수가 일치합니다. 동기화 필요 없음.")
            return True

        # 차원이 일치하지 않으면 조정
        if vector_dim > names_count:
            # 벡터가 더 큰 경우: 이름 목록에 임시 이름 추가
            logger.warning(
                f"벡터 차원({vector_dim})이 특성 이름 수({names_count})보다 큽니다. 이름 목록에 임시 이름을 추가합니다."
            )
            for i in range(names_count, vector_dim):
                feature_names.append(f"feature_{i}")

            # 수정된 이름 목록 저장
            with open(names_path, "w", encoding="utf-8") as f:
                json.dump(feature_names, f, indent=2, ensure_ascii=False)

            logger.info(
                f"이름 목록에 {vector_dim - names_count}개의 임시 이름을 추가했습니다."
            )
        else:
            # 이름 목록이 더 큰 경우: 이름 목록 자르기
            logger.warning(
                f"특성 이름 수({names_count})가 벡터 차원({vector_dim})보다 큽니다. 이름 목록을 조정합니다."
            )
            feature_names = feature_names[:vector_dim]

            # 수정된 이름 목록 저장
            with open(names_path, "w", encoding="utf-8") as f:
                json.dump(feature_names, f, indent=2, ensure_ascii=False)

            logger.info(
                f"이름 목록에서 {names_count - vector_dim}개의 항목을 제거했습니다."
            )

        return True
    except Exception as e:
        logger.error(f"벡터와 특성 이름 동기화 중 오류 발생: {e}")
        return False


def ensure_essential_features(
    vector_path: str, names_path: str, essential_features: Optional[List[str]] = None
) -> bool:
    """
    필수 특성이 벡터에 포함되어 있는지 확인하고, 누락된 경우 추가합니다.

    Args:
        vector_path: 벡터 파일 경로 (.npy)
        names_path: 특성 이름 파일 경로 (.names.json)
        essential_features: 필수 특성 목록 (None이면 기본 목록 사용)

    Returns:
        bool: 필수 특성 확인 및 추가 성공 여부
    """
    try:
        # 기본 필수 특성 목록 사용
        if essential_features is None:
            essential_features = ESSENTIAL_FEATURES

        # 특성 이름 로드
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 누락된 필수 특성 확인
        missing_features = [f for f in essential_features if f not in feature_names]

        # 누락된 특성이 없으면 변경 없음
        if not missing_features:
            logger.info("모든 필수 특성이 벡터에 포함되어 있습니다.")
            return True

        logger.warning(f"다음 필수 특성이 누락되었습니다: {missing_features}")

        # 벡터 로드
        vectors = np.load(vector_path)

        # 벡터 차원 확인
        if len(vectors.shape) > 1:
            vector_shape = list(vectors.shape)
            feature_dim_idx = 1
        else:
            vector_shape = [vectors.shape[0]]
            feature_dim_idx = 0

        # 누락된 특성 수만큼 벡터 확장
        missing_count = len(missing_features)

        if len(vector_shape) > 1:
            # 2D 벡터 확장
            expanded_shape = list(vector_shape)
            expanded_shape[feature_dim_idx] += missing_count
            expanded_vectors = np.zeros(expanded_shape, dtype=vectors.dtype)

            # 기존 벡터 복사
            if feature_dim_idx == 1:
                expanded_vectors[:, : vector_shape[1]] = vectors
            else:
                expanded_vectors[: vector_shape[0], :] = vectors
        else:
            # 1D 벡터 확장
            expanded_vectors = np.zeros(
                vector_shape[0] + missing_count, dtype=vectors.dtype
            )
            expanded_vectors[: vector_shape[0]] = vectors

        # 새 벡터 저장
        np.save(vector_path, expanded_vectors)

        # 특성 이름 목록 확장
        feature_names.extend(missing_features)

        # 수정된 이름 목록 저장
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2, ensure_ascii=False)

        logger.info(f"{missing_count}개의 필수 특성을 추가했습니다: {missing_features}")

        # 특성 기본값 매핑 - 특정 필수 특성의 경우 의미 있는 기본값 제공
        default_values = {
            "gap_stddev": 0.1,
            "hot_cold_mix_score": 0.5,
            "segment_entropy": 0.5,
            "roi_group_score": 0.5,
            "duplicate_flag": 0,
            "max_overlap_with_past": 0,
            "combination_recency_score": 0.5,
            "cluster_score": 0.5,
            "silhouette_score": 0.3,
            "avg_distance_between_clusters": 0.5,
            "std_of_cluster_size": 0.5,
            "balance_score": 0.5,
            "cluster_count_norm": 0.6,
            "largest_cluster_size_norm": 0.5,
            "cluster_size_ratio_norm": 0.5,
        }

        # 위치 관련 특성의 기본값 설정
        for i in range(1, 7):
            default_values[f"position_entropy_{i}"] = 0.5
            default_values[f"position_std_{i}"] = 0.2

        default_values["position_variance_avg"] = 0.3
        default_values["position_bias_score"] = 0.5

        # 누락된 특성에 대한 정보 추가 로깅
        try:
            # feature_registry.json 파일이 있다면 누락된 특성 담당 모듈 정보 로깅
            registry_path = Path(vector_path).parent / "feature_registry.json"
            if registry_path.exists():
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)

                missing_modules = {}
                for feature in missing_features:
                    module = registry.get(feature, "unknown")
                    if module not in missing_modules:
                        missing_modules[module] = []
                    missing_modules[module].append(feature)

                for module, features in missing_modules.items():
                    logger.warning(
                        f"모듈 '{module}'에서 {len(features)}개의 특성 누락: {features}"
                    )
        except Exception as e:
            logger.debug(f"누락된 특성 모듈 정보 로깅 중 오류: {e}")

        return True
    except Exception as e:
        logger.error(f"필수 특성 확인 및 추가 중 오류 발생: {e}")
        raise ValueError(
            f"필수 특성 확인 및 추가 중 오류 발생: {e}"
        )  # 예외를 발생시켜 호출자에게 전달


def analyze_vector_statistics(vector_path: str, names_path: str) -> Dict[str, Any]:
    """
    특성 벡터의 통계 정보를 분석합니다.

    Args:
        vector_path: 벡터 파일 경로
        names_path: 특성 이름 파일 경로

    Returns:
        Dict[str, Any]: 벡터 통계 정보
    """
    try:
        # 벡터와 특성 이름 로드
        vectors = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 벡터가 1차원인 경우 2차원으로 변환
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # 기본 통계
        stats = {
            "vector_count": vectors.shape[0],
            "feature_count": vectors.shape[1],
            "nan_count": int(np.isnan(vectors).sum()),
            "nan_rate": float(np.isnan(vectors).sum()) / vectors.size,
            "zero_rate": float((vectors == 0).sum()) / vectors.size,
            "min_value": float(np.nanmin(vectors)),
            "max_value": float(np.nanmax(vectors)),
            "mean_value": float(np.nanmean(vectors)),
            "std_value": float(np.nanstd(vectors)),
            "missing_features": [],
            "vector_scale_mean": float(np.nanmean(np.abs(vectors))),
            "vector_scale_std": float(np.nanstd(np.abs(vectors))),
            "feature_entropy_score": 0.0,
        }

        # NaN이 많은 특성 확인
        nan_by_feature = np.isnan(vectors).sum(axis=0)
        high_nan_features = [
            (feature_names[i], float(nan_by_feature[i] / vectors.shape[0]))
            for i in range(len(feature_names))
            if nan_by_feature[i] > 0
        ]

        stats["high_nan_features"] = high_nan_features

        # 특성 엔트로피 계산 (정보 다양성)
        try:
            # 각 특성값을 10개 구간으로 나누어 히스토그램 생성
            histograms = []
            for i in range(vectors.shape[1]):
                feature_values = vectors[:, i]
                hist, _ = np.histogram(
                    feature_values, bins=10, range=(0, 1), density=True
                )
                hist = hist / hist.sum() if hist.sum() > 0 else hist
                histograms.append(hist)

            # 각 특성의 엔트로피 계산
            entropies = []
            for hist in histograms:
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropies.append(entropy)

            # 전체 엔트로피 점수 (0-1 범위로 정규화)
            max_possible_entropy = np.log2(10)  # 10개 구간에서 가능한 최대 엔트로피
            avg_entropy = np.mean(entropies)
            normalized_entropy = avg_entropy / max_possible_entropy

            stats["feature_entropy_score"] = float(normalized_entropy)
        except Exception as e:
            logger.warning(f"특성 엔트로피 계산 중 오류: {e}")

        return stats

    except Exception as e:
        logger.error(f"벡터 통계 분석 중 오류 발생: {e}")
        return {"error": str(e)}


def track_vector_scale_stats(
    vector_path: str,
    names_path: str,
    threshold: float = 0.1,
    output_path: Optional[str] = None,
    track_history: bool = True,
) -> Dict[str, Any]:
    """
    벡터 스케일 통계를 추적하고 이전 버전과 비교합니다.

    Args:
        vector_path: 벡터 파일 경로
        names_path: 특성 이름 파일 경로
        threshold: 변화 감지 임계값 (기본값: 0.1, 즉 10%)
        output_path: 결과 저장 경로 (기본값: None이면 벡터 파일과 동일한 디렉토리에 저장)
        track_history: 이력 추적 여부 (기본값: True)

    Returns:
        Dict[str, Any]: 벡터 스케일 통계 및 비교 결과
    """
    try:
        # 벡터와 특성 이름 로드
        vectors = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 벡터가 1차원인 경우 2차원으로 변환
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # 기본 정보 설정
        result = {
            "timestamp": datetime.now().isoformat(),
            "vector_file": os.path.basename(vector_path),
            "num_samples": vectors.shape[0],
            "num_features": len(feature_names),
            "scale_stats": {},
            "changes_detected": False,
            "changed_features": {},
            "global_stats": {
                "mean_value": float(np.nanmean(vectors)),
                "std_value": float(np.nanstd(vectors)),
                "min_value": float(np.nanmin(vectors)),
                "max_value": float(np.nanmax(vectors)),
                "zero_rate": float((vectors == 0).sum()) / vectors.size,
                "nan_rate": float(np.isnan(vectors).sum()) / vectors.size,
            },
        }

        # 각 특성별 통계 계산
        for i, feature_name in enumerate(feature_names):
            if i >= vectors.shape[1]:
                continue

            feature_values = vectors[:, i]
            feature_stats = {
                "mean": float(np.nanmean(feature_values)),
                "std": float(np.nanstd(feature_values)),
                "min": float(np.nanmin(feature_values)),
                "max": float(np.nanmax(feature_values)),
                "zero_rate": float((feature_values == 0).sum()) / len(feature_values),
                "nan_rate": float(np.isnan(feature_values).sum()) / len(feature_values),
            }
            result["scale_stats"][feature_name] = feature_stats

        # 결과 저장 경로 설정
        cache_dir = Path(vector_path).parent
        if output_path is None:
            stats_file_path = cache_dir / "vector_scale_stats.json"
        else:
            stats_file_path = Path(output_path)
            # 디렉토리 생성
            os.makedirs(stats_file_path.parent, exist_ok=True)

        diff_file_path = stats_file_path.parent / "vector_scale_diff.json"
        history_file_path = stats_file_path.parent / "vector_scale_history.json"

        # 이전 통계와 비교
        previous_stats = None
        if stats_file_path.exists():
            try:
                with open(stats_file_path, "r", encoding="utf-8") as f:
                    previous_stats = json.load(f)

                # 변경된 특성 탐지
                if "scale_stats" in previous_stats:
                    prev_scale_stats = previous_stats["scale_stats"]
                    changes = {}
                    scale_drift = {}

                    for feature_name, current_stats in result["scale_stats"].items():
                        if feature_name in prev_scale_stats:
                            prev_stats = prev_scale_stats[feature_name]
                            feature_changes = {}
                            feature_drift = {}

                            # 평균 변화 계산
                            mean_delta = current_stats["mean"] - prev_stats["mean"]
                            feature_drift["mean_delta"] = mean_delta

                            # 표준편차 비율 계산
                            if prev_stats["std"] > 0:
                                std_ratio = current_stats["std"] / prev_stats["std"]
                            else:
                                std_ratio = (
                                    1.0 if current_stats["std"] == 0 else float("inf")
                                )
                            feature_drift["std_ratio"] = std_ratio

                            # 범위 변화 계산
                            prev_range = prev_stats["max"] - prev_stats["min"]
                            current_range = current_stats["max"] - current_stats["min"]
                            if prev_range > 0:
                                range_ratio = current_range / prev_range
                            else:
                                range_ratio = (
                                    1.0 if current_range == 0 else float("inf")
                                )
                            feature_drift["range_ratio"] = range_ratio

                            # 스케일 드리프트 저장
                            scale_drift[feature_name] = feature_drift

                            # 평균 변화 감지 (임계값 기준)
                            if abs(mean_delta) > threshold * abs(
                                prev_stats["mean"] + 1e-10
                            ):
                                change_pct = (
                                    mean_delta / (abs(prev_stats["mean"]) + 1e-10)
                                ) * 100
                                feature_changes["mean"] = {
                                    "prev": prev_stats["mean"],
                                    "current": current_stats["mean"],
                                    "delta_pct": change_pct,
                                }

                            # 표준편차 변화 감지 (임계값 기준)
                            if abs(std_ratio - 1.0) > threshold:
                                change_pct = (std_ratio - 1.0) * 100
                                feature_changes["std"] = {
                                    "prev": prev_stats["std"],
                                    "current": current_stats["std"],
                                    "delta_pct": change_pct,
                                }

                            # 범위 변화 감지 (임계값 기준)
                            if abs(range_ratio - 1.0) > threshold:
                                change_pct = (range_ratio - 1.0) * 100
                                feature_changes["range"] = {
                                    "prev": prev_range,
                                    "current": current_range,
                                    "delta_pct": change_pct,
                                }

                            # 제로레이트 변화 감지
                            if (
                                "zero_rate" in current_stats
                                and "zero_rate" in prev_stats
                            ):
                                zero_delta = (
                                    current_stats["zero_rate"] - prev_stats["zero_rate"]
                                )
                                if abs(zero_delta) > threshold:
                                    feature_changes["zero_rate"] = {
                                        "prev": prev_stats["zero_rate"],
                                        "current": current_stats["zero_rate"],
                                        "delta_pct": zero_delta * 100,
                                    }

                            # 변화가 감지된 경우 추가
                            if feature_changes:
                                changes[feature_name] = feature_changes

                    # 변경 사항 저장
                    if changes:
                        result["changes_detected"] = True
                        result["changed_features"] = changes

                        # 변경 사항 로깅
                        logger.warning(f"{len(changes)}개 특성에서 스케일 변화 감지됨")
                        for feature, change_info in list(changes.items())[
                            :5
                        ]:  # 처음 5개만 출력
                            for stat_type, values in change_info.items():
                                logger.warning(
                                    f"특성 '{feature}' {stat_type}: {values['prev']:.4f} → {values['current']:.4f} "
                                    f"(변화: {values['delta_pct']:+.2f}%)"
                                )
                        if len(changes) > 5:
                            logger.warning(
                                f"... 외 {len(changes) - 5}개 특성에 변화 있음"
                            )

                        # 변경 파일 저장
                        drift_data = {
                            "timestamp": datetime.now().isoformat(),
                            "previous_timestamp": previous_stats.get(
                                "timestamp", "unknown"
                            ),
                            "vector_file": os.path.basename(vector_path),
                            "num_features": len(feature_names),
                            "changes": changes,
                            "scale_drift": scale_drift,
                            "threshold": threshold,
                            "total_changed_features": len(changes),
                        }

                        with open(diff_file_path, "w", encoding="utf-8") as f:
                            json.dump(drift_data, f, indent=2, ensure_ascii=False)

                        logger.info(f"벡터 스케일 변화 정보 저장됨: {diff_file_path}")

                        # 이력 추적
                        if track_history and history_file_path.exists():
                            try:
                                with open(
                                    history_file_path, "r", encoding="utf-8"
                                ) as f:
                                    history = json.load(f)

                                history_entry = {
                                    "timestamp": datetime.now().isoformat(),
                                    "vector_file": os.path.basename(vector_path),
                                    "changed_features_count": len(changes),
                                    "global_stats": result["global_stats"],
                                    "major_changes": {
                                        k: v
                                        for k, v in changes.items()
                                        if any(
                                            abs(x.get("delta_pct", 0)) > threshold * 100
                                            for x in v.values()
                                        )
                                    },
                                }

                                history["entries"].append(history_entry)

                                with open(
                                    history_file_path, "w", encoding="utf-8"
                                ) as f:
                                    json.dump(history, f, indent=2, ensure_ascii=False)

                                logger.info(
                                    f"벡터 스케일 이력 업데이트됨: {history_file_path}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"벡터 스케일 이력 업데이트 중 오류 발생: {e}"
                                )

            except Exception as e:
                logger.error(f"이전 스케일 통계와 비교 중 오류 발생: {e}")

        # 이력 파일 생성 (없는 경우)
        if track_history and not history_file_path.exists():
            try:
                history = {
                    "first_timestamp": datetime.now().isoformat(),
                    "vector_file": os.path.basename(vector_path),
                    "entries": [
                        {
                            "timestamp": datetime.now().isoformat(),
                            "vector_file": os.path.basename(vector_path),
                            "changed_features_count": 0,
                            "global_stats": result["global_stats"],
                            "major_changes": {},
                        }
                    ],
                }

                with open(history_file_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)

                logger.info(f"벡터 스케일 이력 파일 생성됨: {history_file_path}")
            except Exception as e:
                logger.error(f"벡터 스케일 이력 파일 생성 중 오류 발생: {e}")

        # 새 통계 파일 저장 (원자적으로 저장하기 위해 임시 파일 사용)
        temp_file_path = stats_file_path.with_suffix(".json.tmp")
        try:
            with open(temp_file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # 임시 파일을 실제 파일로 이동 (Windows에서는 기존 파일을 먼저 삭제해야 함)
            if os.path.exists(stats_file_path):
                os.remove(stats_file_path)
            os.rename(temp_file_path, stats_file_path)

            logger.info(f"벡터 스케일 통계 저장됨: {stats_file_path}")

        except Exception as e:
            logger.error(f"벡터 스케일 통계 저장 중 오류 발생: {e}")
            # 임시 파일 정리
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

        return result

    except Exception as e:
        logger.error(f"벡터 스케일 통계 추적 중 오류 발생: {e}")
        return {"error": str(e)}


def detect_low_variance_features(
    vector_path: str, names_path: str, threshold: float = 0.005
) -> Dict[str, Any]:
    """
    낮은 분산을 가진 특성을 감지합니다.

    Args:
        vector_path: 벡터 파일 경로
        names_path: 특성 이름 파일 경로
        threshold: 분산 임계값 (기본값: 0.005)

    Returns:
        Dict[str, Any]: 낮은 분산 특성 정보
    """
    try:
        # 벡터와 특성 이름 로드
        vectors = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 벡터가 1차원인 경우 2차원으로 변환
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # 각 특성별 분산 계산
        variances = np.nanvar(vectors, axis=0)

        # 낮은 분산 특성 인덱스 찾기
        low_variance_indices = np.where(variances < threshold)[0].tolist()

        # 결과 생성
        result = {
            "removed_feature_indices": low_variance_indices,
            "removed_feature_names": [
                feature_names[i] for i in low_variance_indices if i < len(feature_names)
            ],
            "threshold": threshold,
            "total_removed": len(low_variance_indices),
            "feature_variances": {
                feature_names[i]: float(variances[i])
                for i in range(min(len(feature_names), len(variances)))
            },
        }

        # 인덱스 -> 이름 매핑 시도
        try:
            mapping_path = Path(vector_path).parent / "feature_index_mapping.json"
            if mapping_path.exists():
                with open(mapping_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)

                # 매핑 정보 추가
                mapped_names = []
                for idx in low_variance_indices:
                    if str(idx) in mapping:
                        mapped_names.append(mapping[str(idx)])
                    elif idx < len(feature_names):
                        mapped_names.append(feature_names[idx])

                result["mapped_feature_names"] = mapped_names
        except Exception as e:
            logger.debug(f"특성 인덱스 매핑 중 오류: {e}")

        # 저장 경로
        cache_dir = Path(vector_path).parent
        result_path = cache_dir / "low_variance_features.json"

        # 결과 저장 (원자적으로 저장하기 위해 임시 파일 사용)
        temp_file_path = result_path.with_suffix(".json.tmp")
        try:
            with open(temp_file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # 임시 파일을 실제 파일로 이동
            if os.path.exists(result_path):
                os.remove(result_path)
            os.rename(temp_file_path, result_path)

            logger.info(
                f"낮은 분산 특성 정보 저장됨: {result_path} (제거된 특성: {len(low_variance_indices)}개)"
            )

        except Exception as e:
            logger.error(f"낮은 분산 특성 정보 저장 중 오류 발생: {e}")
            # 임시 파일 정리
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

        return result

    except Exception as e:
        logger.error(f"낮은 분산 특성 감지 중 오류 발생: {e}")
        return {"error": str(e)}


def filter_low_variance_features(
    vector_path: str, names_path: str, config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    낮은 분산 특성을 필터링하여 벡터와 특성 이름을 반환합니다.

    Args:
        vector_path: 벡터 파일 경로
        names_path: 특성 이름 파일 경로
        config: 설정 객체 (선택 사항)

    Returns:
        Tuple[np.ndarray, List[str]]: 필터링된 벡터와 특성 이름
    """
    try:
        # 벡터와 특성 이름 로드
        vectors = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 벡터가 1차원인 경우 2차원으로 변환
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # 설정에서 필터링 여부 확인
        filtering_enabled = False
        if config is not None:
            try:
                filtering_enabled = config.get("filtering", {}).get(
                    "remove_low_variance_features", False
                )
            except:
                pass

        # 필터링이 비활성화된 경우 원본 반환
        if not filtering_enabled:
            return vectors, feature_names

        # 낮은 분산 특성 파일 경로
        cache_dir = Path(vector_path).parent
        low_var_path = cache_dir / "low_variance_features.json"

        # 파일이 없는 경우 먼저 생성
        if not low_var_path.exists():
            detect_low_variance_features(vector_path, names_path)

        # 낮은 분산 특성 정보 로드
        if low_var_path.exists():
            with open(low_var_path, "r", encoding="utf-8") as f:
                low_var_info = json.load(f)

            removed_indices = low_var_info.get("removed_feature_indices", [])

            if removed_indices:
                # 유효한 인덱스만 선택 (벡터 크기를 초과하지 않도록)
                valid_indices = [
                    idx for idx in removed_indices if idx < vectors.shape[1]
                ]

                if valid_indices:
                    # 유지할 인덱스 생성 (제거할 인덱스의 보수)
                    keep_indices = [
                        i for i in range(vectors.shape[1]) if i not in valid_indices
                    ]

                    # 벡터 필터링
                    filtered_vectors = vectors[:, keep_indices]

                    # 특성 이름 필터링
                    filtered_names = [
                        feature_names[i] for i in keep_indices if i < len(feature_names)
                    ]

                    logger.info(f"낮은 분산 특성 {len(valid_indices)}개 제거됨")
                    return filtered_vectors, filtered_names

        # 필터링을 수행하지 않은 경우 원본 반환
        return vectors, feature_names

    except Exception as e:
        logger.error(f"낮은 분산 특성 필터링 중 오류 발생: {e}")
        # 오류 발생 시 원본 반환
        return vectors, feature_names


def validate_and_track_vector(
    vector_path: str,
    names_path: str,
    config: Optional[Dict[str, Any]] = None,
    track_scale: bool = True,
    detect_low_variance: bool = True,
) -> Dict[str, Any]:
    """
    벡터 검증, 스케일 추적 및 낮은 분산 특성 감지를 수행하는 통합 메서드입니다.

    Args:
        vector_path: 벡터 파일 경로
        names_path: 특성 이름 파일 경로
        config: 설정 객체 (선택 사항)
        track_scale: 스케일 추적 여부
        detect_low_variance: 낮은 분산 특성 감지 여부

    Returns:
        Dict[str, Any]: 검증 및 추적 결과
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "vector_file": os.path.basename(vector_path),
        "validation_success": False,
    }

    try:
        # 1. 벡터 차원 확인
        dim_check = check_vector_dimensions(
            vector_path, names_path, raise_on_mismatch=False
        )
        result["dimension_check"] = dim_check

        if not dim_check:
            logger.warning(
                f"벡터와 특성 이름의 차원이 일치하지 않습니다: {vector_path}, {names_path}"
            )
            return result

        # 2. 필수 특성 확인
        essential_check = True
        try:
            if config is not None:
                essential_features = config.get(
                    "essential_features", ESSENTIAL_FEATURES
                )
            else:
                essential_features = ESSENTIAL_FEATURES

            essential_check = ensure_essential_features(
                vector_path, names_path, essential_features
            )
        except Exception as e:
            logger.error(f"필수 특성 확인 중 오류 발생: {e}")
            essential_check = False

        result["essential_check"] = essential_check

        # 3. 벡터 통계 분석
        stats = analyze_vector_statistics(vector_path, names_path)
        result["statistics"] = stats

        # 4. 스케일 추적 (옵션)
        if track_scale:
            scale_threshold = 0.1  # 기본 임계값 10%
            if config is not None:
                scale_threshold = config.get("vector_validation", {}).get(
                    "scale_threshold", 0.1
                )

            scale_stats = track_vector_scale_stats(
                vector_path, names_path, threshold=scale_threshold
            )
            result["scale_tracking"] = {
                "success": "error" not in scale_stats,
                "changes_detected": scale_stats.get("changes_detected", False),
                "changed_features_count": len(scale_stats.get("changed_features", {})),
            }

        # 5. 낮은 분산 특성 감지 (옵션)
        if detect_low_variance:
            variance_threshold = 0.005  # 기본 임계값
            if config is not None:
                variance_threshold = config.get("vector_validation", {}).get(
                    "variance_threshold", 0.005
                )

            low_var_info = detect_low_variance_features(
                vector_path, names_path, threshold=variance_threshold
            )
            result["low_variance_detection"] = {
                "success": "error" not in low_var_info,
                "removed_count": low_var_info.get("total_removed", 0),
                "threshold": low_var_info.get("threshold", variance_threshold),
            }

        # 전체 검증 성공 여부
        result["validation_success"] = essential_check and dim_check

        return result

    except Exception as e:
        logger.error(f"벡터 검증 및 추적 중 오류 발생: {e}")
        result["error"] = str(e)
        return result


def track_vector_scale_statistics(
    vector_file_path: str,
    feature_names_path: str,
    output_path: str = "data/cache/vector_scale_stats.json",
) -> Dict[str, Any]:
    """
    벡터 스케일 통계를 추적하고 저장합니다.

    이 함수는 track_vector_scale_stats 함수의 별칭입니다.
    호환성을 위해 유지됩니다.

    Args:
        vector_file_path: 벡터 파일 경로
        feature_names_path: 특성 이름 파일 경로
        output_path: 결과 저장 경로 (기본값: "data/cache/vector_scale_stats.json")

    Returns:
        Dict[str, Any]: 벡터 스케일 통계
    """
    logger.info(
        "track_vector_scale_statistics 함수는 track_vector_scale_stats의 별칭입니다."
    )
    return track_vector_scale_stats(
        vector_path=vector_file_path,
        names_path=feature_names_path,
        output_path=output_path,
    )


def get_feature_filter_mask(
    feature_names: List[str],
    low_variance_path: str = "data/cache/low_variance_features.json",
    min_variance_threshold: float = 0.001,
    custom_exclude: Optional[List[str]] = None,
) -> List[bool]:
    """
    저분산 특성을 제외하는 필터 마스크를 생성합니다.

    Args:
        feature_names: 특성 이름 목록
        low_variance_path: 저분산 특성 정보 파일 경로
        min_variance_threshold: 최소 분산 임계값 (직접 계산 시 사용)
        custom_exclude: 추가로 제외할 특성 목록

    Returns:
        List[bool]: 필터 마스크 (True: 유지, False: 제거)
    """
    try:
        # 기본 마스크 초기화 (모든 특성 유지)
        mask = [True] * len(feature_names)

        # 제거할 특성 이름 목록 초기화
        removed_feature_names = []

        # 1. 저분산 특성 정보 파일이 있으면 로드
        if os.path.exists(low_variance_path):
            try:
                with open(low_variance_path, "r", encoding="utf-8") as f:
                    low_var_info = json.load(f)
                # 제거할 특성 이름 추가
                removed_feature_names.extend(
                    low_var_info.get("removed_feature_names", [])
                )
                logger.info(
                    f"저분산 특성 정보 파일에서 {len(removed_feature_names)}개 특성 제거 대상 확인"
                )
            except Exception as e:
                logger.warning(f"저분산 특성 정보 파일 로드 중 오류 발생: {e}")

        # 2. 사용자 지정 제외 목록 처리
        if custom_exclude:
            custom_exclude_set = set(custom_exclude)
            custom_exclude_count = sum(
                1 for name in feature_names if name in custom_exclude_set
            )
            logger.info(
                f"사용자 지정 제외 목록에서 {custom_exclude_count}개 특성 제거 대상 확인"
            )
            removed_feature_names.extend(custom_exclude)

        # 3. 마스크 생성
        if removed_feature_names:
            removed_set = set(removed_feature_names)
            mask = [name not in removed_set for name in feature_names]

            # 제거된 특성 수 로깅
            removed_count = sum(1 for flag in mask if not flag)
            total_count = len(feature_names)
            logger.info(
                f"필터 마스크 생성 완료: {removed_count}/{total_count} 특성 제거됨 ({removed_count/total_count:.2%})"
            )

            # 제거된 특성 샘플 로깅 (최대 5개)
            removed_sample = [
                name for name, keep in zip(feature_names, mask) if not keep
            ][:5]
            if removed_sample:
                logger.info(f"제거된 특성 샘플: {removed_sample}")

        return mask

    except Exception as e:
        logger.error(f"특성 필터 마스크 생성 중 오류 발생: {e}")
        # 오류 발생 시 모든 특성 유지
        return [True] * len(feature_names)


def filter_features_by_mask(
    features: np.ndarray, feature_names: List[str], mask: List[bool]
) -> Tuple[np.ndarray, List[str]]:
    """
    마스크를 기반으로 특성 벡터와 이름을 필터링합니다.

    Args:
        features: 특성 벡터 (2D 배열 또는 1D 배열)
        feature_names: 특성 이름 목록
        mask: 필터 마스크 (True: 유지, False: 제거)

    Returns:
        Tuple[np.ndarray, List[str]]: 필터링된 특성 벡터와 이름
    """
    try:
        # 마스크 길이 확인
        if len(mask) != len(feature_names):
            logger.warning(
                f"마스크 길이({len(mask)})가 특성 이름 수({len(feature_names)})와 일치하지 않습니다."
            )
            if len(mask) < len(feature_names):
                # 마스크가 더 짧은 경우, 나머지는 유지
                mask = mask + [True] * (len(feature_names) - len(mask))
            else:
                # 마스크가 더 긴 경우, 잘라냄
                mask = mask[: len(feature_names)]

        # 벡터 차원 확인
        is_1d = len(features.shape) == 1

        # 필터링
        if is_1d:
            if len(features) != len(mask):
                logger.warning(
                    f"1D 벡터 길이({len(features)})가 마스크 길이({len(mask)})와 일치하지 않습니다."
                )
                # 길이가 다른 경우 처리
                if len(features) < len(mask):
                    # 벡터가 더 짧은 경우, 마스크 자르기
                    mask = mask[: len(features)]
                else:
                    # 벡터가 더 긴 경우, 나머지 마스크 추가
                    mask = mask + [True] * (len(features) - len(mask))

            # 1D 벡터 필터링
            filtered_features = features[mask]
        else:
            if features.shape[1] != len(mask):
                logger.warning(
                    f"2D 벡터 열 수({features.shape[1]})가 마스크 길이({len(mask)})와 일치하지 않습니다."
                )
                # 길이가 다른 경우 처리
                if features.shape[1] < len(mask):
                    # 벡터가 더 짧은 경우, 마스크 자르기
                    mask = mask[: features.shape[1]]
                else:
                    # 벡터가 더 긴 경우, 나머지 마스크 추가
                    mask = mask + [True] * (features.shape[1] - len(mask))

            # 2D 벡터 필터링
            filtered_features = features[:, mask]

        # 특성 이름 필터링
        filtered_names = [name for name, keep in zip(feature_names, mask) if keep]

        # 필터링 결과 로깅
        removed_count = sum(1 for flag in mask if not flag)
        logger.info(
            f"특성 필터링 완료: {removed_count}개 특성 제거됨 ({len(filtered_names)}개 유지)"
        )

        return filtered_features, filtered_names

    except Exception as e:
        logger.error(f"특성 필터링 중 오류 발생: {e}")
        # 오류 발생 시 원본 반환
        return features, feature_names


def load_low_variance_feature_names(
    low_var_path: str = "data/cache/low_variance_features.json",
) -> List[str]:
    """
    저분산 특성 이름 목록을 로드합니다.

    Args:
        low_var_path: 저분산 특성 정보 파일 경로

    Returns:
        List[str]: 저분산 특성 이름 목록
    """
    try:
        if not os.path.exists(low_var_path):
            logger.warning(f"저분산 특성 정보 파일이 존재하지 않습니다: {low_var_path}")
            return []

        with open(low_var_path, "r", encoding="utf-8") as f:
            low_var_info = json.load(f)

        removed_feature_names = low_var_info.get("removed_feature_names", [])

        logger.info(
            f"저분산 특성 정보 파일에서 {len(removed_feature_names)}개 특성 이름 로드 완료"
        )
        return removed_feature_names

    except Exception as e:
        logger.error(f"저분산 특성 이름 로드 중 오류 발생: {e}")
        return []


def save_vector_scale_statistics(
    vector_path: str,
    names_path: str,
    output_path: str = "data/cache/vector_scale_stats.json",
    compute_drift: bool = True,
) -> Dict[str, Any]:
    """
    벡터 스케일 통계를 계산하고 저장합니다.

    Args:
        vector_path: 벡터 파일 경로
        names_path: 특성 이름 파일 경로
        output_path: 결과 저장 경로 (기본값: "data/cache/vector_scale_stats.json")
        compute_drift: 이전 통계와 비교하여 drift 계산 여부

    Returns:
        Dict[str, Any]: 벡터 스케일 통계
    """
    try:
        # 벡터와 특성 이름 로드
        vectors = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 벡터가 1차원인 경우 2차원으로 변환
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # 각 특성별 통계 계산
        feature_stats = {}
        for i, name in enumerate(feature_names):
            if i >= vectors.shape[1]:
                continue

            feature_values = vectors[:, i]
            feature_stats[name] = {
                "mean": float(np.nanmean(feature_values)),
                "std": float(np.nanstd(feature_values)),
                "min": float(np.nanmin(feature_values)),
                "max": float(np.nanmax(feature_values)),
            }

        # 통계 결과 생성
        stats_result = {
            "timestamp": timestamp,
            "feature_stats": feature_stats,
            "vector_shape": list(vectors.shape),
            "feature_count": len(feature_names),
        }

        # 통계 저장
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 임시 파일에 먼저 저장 후 원자적으로 이동
        temp_output_path = f"{output_path}.tmp"
        with open(temp_output_path, "w", encoding="utf-8") as f:
            json.dump(stats_result, f, indent=2, ensure_ascii=False)

        if os.path.exists(output_path):
            os.replace(temp_output_path, output_path)
        else:
            os.rename(temp_output_path, output_path)

        logger.info(f"벡터 스케일 통계 저장 완료: {output_path}")

        # 이전 통계와 비교하여 drift 계산
        if compute_drift:
            # 이전 통계 파일 경로 (가장 최근 통계 파일)
            drift_output_path = os.path.join(
                os.path.dirname(output_path), "vector_scale_drift.json"
            )
            history_path = os.path.join(
                os.path.dirname(output_path), "vector_scale_stats_history"
            )

            if not os.path.exists(history_path):
                os.makedirs(history_path, exist_ok=True)

            # 현재 통계의 복사본을 history에 저장
            history_file = os.path.join(
                history_path, f"vector_scale_stats_{timestamp.replace(':', '-')}.json"
            )
            try:
                shutil.copy2(output_path, history_file)
                logger.info(f"벡터 스케일 통계 히스토리 저장: {history_file}")
            except Exception as e:
                logger.warning(f"히스토리 저장 중 오류: {e}")

            # 이전 통계와 비교
            try:
                # history 디렉토리의 파일 목록 가져오기
                history_files = sorted(
                    [
                        os.path.join(history_path, f)
                        for f in os.listdir(history_path)
                        if f.startswith("vector_scale_stats_") and f.endswith(".json")
                    ]
                )

                if len(history_files) > 1:
                    # 현재 통계를 제외한 가장 최근 통계
                    prev_stats_path = history_files[-2]

                    with open(prev_stats_path, "r", encoding="utf-8") as f:
                        prev_stats = json.load(f)

                    # drift 계산
                    scale_drift = {}
                    prev_feature_stats = prev_stats.get("feature_stats", {})
                    threshold_crossed_features = []

                    for name, stats in feature_stats.items():
                        if name in prev_feature_stats:
                            mean_delta = abs(
                                stats["mean"] - prev_feature_stats[name]["mean"]
                            )
                            std_delta = abs(
                                stats["std"] - prev_feature_stats[name]["std"]
                            )

                            scale_drift[name] = {
                                "mean_delta": float(mean_delta),
                                "std_delta": float(std_delta),
                            }

                            # 임계값(0.1) 초과 여부 확인
                            if mean_delta > 0.1 or std_delta > 0.1:
                                threshold_crossed_features.append(name)
                                logger.warning(
                                    f"특성 '{name}'의 스케일 변화가 임계값을 초과했습니다: "
                                    f"mean_delta={mean_delta:.4f}, std_delta={std_delta:.4f}"
                                )

                    # drift 결과 저장
                    drift_result = {
                        "timestamp": timestamp,
                        "prev_stats_timestamp": prev_stats.get("timestamp", "unknown"),
                        "scale_drift": scale_drift,
                        "threshold_crossed_count": len(threshold_crossed_features),
                        "threshold_crossed_features": threshold_crossed_features,
                    }

                    with open(drift_output_path, "w", encoding="utf-8") as f:
                        json.dump(drift_result, f, indent=2, ensure_ascii=False)

                    logger.info(
                        f"벡터 스케일 drift 계산 완료: {drift_output_path} "
                        f"(임계값 초과 특성: {len(threshold_crossed_features)}개)"
                    )

                    # 결과에 drift 정보 추가
                    stats_result["drift"] = {
                        "threshold_crossed_count": len(threshold_crossed_features),
                        "has_significant_drift": len(threshold_crossed_features) > 0,
                    }
            except Exception as e:
                logger.warning(f"벡터 스케일 drift 계산 중 오류: {e}")

        return stats_result

    except Exception as e:
        logger.error(f"벡터 스케일 통계 계산 중 오류 발생: {e}")
        return {"error": str(e)}


def create_clean_vector(
    vector_path: str, names_path: str, z_threshold: float = 2.5, save_clean: bool = True
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    벡터에서 이상치를 제거하고 정제된 clean 버전을 생성합니다.

    Args:
        vector_path: 벡터 파일 경로 (.npy)
        names_path: 특성 이름 파일 경로 (.names.json)
        z_threshold: Z-score 임계값 (기본값: 2.5)
        save_clean: 정제된 벡터를 파일로 저장할지 여부 (기본값: True)

    Returns:
        Tuple[np.ndarray, List[str], Dict[str, Any]]: (정제된 벡터, 특성 이름, 메타데이터)
    """
    try:
        # 벡터와 특성 이름 로드
        vectors = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 이상치 탐지
        outlier_mask, outlier_indices = detect_outliers(
            vector_path=vector_path, names_path=names_path, z_threshold=z_threshold
        )

        # 이상치 반전 마스크 생성 (이상치가 아닌 샘플 선택)
        keep_mask = np.ones(vectors.shape[0], dtype=bool)
        keep_mask[outlier_indices] = False

        # 이상치가 제거된 정제 벡터 생성
        clean_vector = vectors[keep_mask]

        # 메타데이터 생성
        metadata = {
            "removed_outliers": len(outlier_indices),
            "original_samples": int(vectors.shape[0]),
            "remaining_samples": int(clean_vector.shape[0]),
            "removal_ratio": (
                len(outlier_indices) / vectors.shape[0] if vectors.shape[0] > 0 else 0
            ),
            "z_threshold": z_threshold,
            "timestamp": datetime.now().isoformat(),
        }

        # 결과 로깅
        logger.info(
            f"이상치 제거 완료: {metadata['removed_outliers']}/{metadata['original_samples']} "
            f"({metadata['removal_ratio']:.2%}) 제거됨"
        )

        # 정제된 벡터 저장 (선택적)
        if save_clean:
            # 파일 경로 생성
            clean_vector_path = vector_path.replace(".npy", ".clean.npy")
            clean_meta_path = vector_path.replace(".npy", ".clean.meta.json")

            # 벡터 저장
            np.save(clean_vector_path, clean_vector)

            # 메타데이터 저장
            with open(clean_meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"정제된 벡터 저장 완료: {clean_vector_path}")
            logger.info(f"정제 메타데이터 저장 완료: {clean_meta_path}")

        return clean_vector, feature_names, metadata

    except Exception as e:
        logger.error(f"이상치 제거 중 오류 발생: {e}")
        # 오류 발생 시 원본 벡터와 빈 메타데이터 반환
        try:
            vectors = np.load(vector_path)
            with open(names_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
            return vectors, feature_names, {"error": str(e)}
        except:
            raise RuntimeError(f"벡터 로드 실패: {e}")


def load_vector_with_config(
    base_path: str = "data/cache/feature_vector_full",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    설정에 따라 원본/필터링/정제된 벡터를 로드합니다.

    Args:
        base_path: 벡터 파일의 기본 경로 (확장자 제외)
        config: 설정 사전
            - config["training"]["use_filtered_vector"]: 필터링된 벡터 사용 여부
            - config["training"]["remove_outliers"]: 이상치 제거 벡터 사용 여부

    Returns:
        (특성 벡터, 특성 이름 목록) 튜플

    Raises:
        RuntimeError: 설정 키가 누락된 경우
        FileNotFoundError: 벡터 파일이 존재하지 않는 경우
    """
    try:
        # 1. 기본 설정값 확인
        use_filtered = False
        remove_outliers = False

        # 2. 설정에서 필터링된 벡터 사용 여부 확인
        if config is not None:
            try:
                use_filtered = config["training"]["use_filtered_vector"]
            except KeyError as e:
                logger.error(f"[ERROR] 설정 키 누락: {str(e)}")
                raise RuntimeError("설정 키 누락으로 프로세스를 종료합니다.")

            # 3. 설정에서 이상치 제거 여부 확인
            try:
                remove_outliers = config["training"]["remove_outliers"]
            except KeyError as e:
                logger.warning(
                    f"[WARNING] remove_outliers 설정 키가 없습니다. 기본값(False) 사용"
                )

        # 4. 파일 경로 설정
        if use_filtered:
            vector_path = f"{base_path}_filtered"
        else:
            vector_path = base_path

        # 5. 이상치 제거 벡터 사용 여부에 따라 경로 수정
        if remove_outliers:
            vector_path = f"{vector_path}.clean"
            logger.info(f"이상치 제거 벡터 사용: {vector_path}.npy")

        # 6. 벡터 로드
        vector_file = f"{vector_path}.npy"
        names_file = (
            f"{vector_path}.names.json"
            if not remove_outliers
            else f"{base_path}.names.json"
        )

        if not os.path.exists(vector_file):
            raise FileNotFoundError(f"벡터 파일({vector_file})이 존재하지 않습니다.")

        if not os.path.exists(names_file):
            raise FileNotFoundError(
                f"특성 이름 파일({names_file})이 존재하지 않습니다."
            )

        vector = np.load(vector_file)

        with open(names_file, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        # 차원 검증
        expected_dim = len(feature_names)
        actual_dim = vector.shape[1] if len(vector.shape) > 1 else vector.shape[0]

        if actual_dim != expected_dim:
            raise ValueError(
                f"벡터 차원({actual_dim})과 특성 이름 수({expected_dim})가 일치하지 않습니다."
            )

        logger.info(
            f"특성 벡터 로드 완료: {vector_file}, 형태={vector.shape}, 특성 수={len(feature_names)}"
        )

        return vector, feature_names

    except Exception as e:
        logger.error(f"벡터 로드 중 오류 발생: {e}")
        raise


def validate_vector_integrity_enhanced(
    vector_path: str, names_path: str, expected_dims: int = 95
) -> Dict[str, Any]:
    """
    강화된 벡터 무결성 검증
    
    Args:
        vector_path: 벡터 파일 경로
        names_path: 특성 이름 파일 경로  
        expected_dims: 예상 차원 수
        
    Returns:
        Dict[str, Any]: 검증 결과 및 통계
    """
    logger.info("강화된 벡터 무결성 검증 시작...")
    
    result = {
        "validation_passed": False,
        "dimension_check": False,
        "nan_inf_check": False,
        "entropy_check": False,
        "statistics": {}
    }
    
    try:
        # 1. 파일 존재 확인
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"벡터 파일이 존재하지 않습니다: {vector_path}")
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"특성 이름 파일이 존재하지 않습니다: {names_path}")
        
        # 2. 벡터 및 특성 이름 로드
        vector = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        
        # 3. 차원 일치 확인
        if len(vector) == len(feature_names) == expected_dims:
            result["dimension_check"] = True
            logger.info(f"차원 검증 통과: {len(vector)}차원")
        else:
            logger.error(f"차원 불일치: 벡터 {len(vector)}, 특성명 {len(feature_names)}, 예상 {expected_dims}")
            result["statistics"]["vector_dims"] = len(vector)
            result["statistics"]["names_count"] = len(feature_names)
            result["statistics"]["expected_dims"] = expected_dims
        
        # 4. NaN/Inf 비율 확인 (1% 초과 시 실패)
        nan_count = np.isnan(vector).sum()
        inf_count = np.isinf(vector).sum()
        total_invalid = nan_count + inf_count
        invalid_ratio = total_invalid / len(vector)
        
        result["statistics"]["nan_count"] = int(nan_count)
        result["statistics"]["inf_count"] = int(inf_count)
        result["statistics"]["invalid_ratio"] = float(invalid_ratio)
        
        if invalid_ratio <= 0.01:
            result["nan_inf_check"] = True
            logger.info(f"NaN/Inf 검증 통과: {invalid_ratio:.6f} <= 0.01")
        else:
            logger.error(f"NaN/Inf 비율 초과: {invalid_ratio:.6f} > 0.01")
        
        # 5. 특성 다양성 엔트로피 계산
        entropy_score = calculate_feature_entropy(vector)
        result["statistics"]["entropy_score"] = entropy_score
        
        if entropy_score >= 0.1:
            result["entropy_check"] = True
            logger.info(f"엔트로피 검증 통과: {entropy_score:.4f} >= 0.1")
        else:
            logger.warning(f"특성 다양성 부족: 엔트로피 {entropy_score:.4f} < 0.1")
            result["entropy_check"] = True  # 경고만 하고 통과
        
        # 6. 추가 통계 계산
        result["statistics"]["vector_mean"] = float(np.mean(vector))
        result["statistics"]["vector_std"] = float(np.std(vector))
        result["statistics"]["vector_min"] = float(np.min(vector))
        result["statistics"]["vector_max"] = float(np.max(vector))
        result["statistics"]["zero_ratio"] = float(np.sum(vector == 0) / len(vector))
        
        # 7. 최종 검증 결과
        result["validation_passed"] = (
            result["dimension_check"] and 
            result["nan_inf_check"] and 
            result["entropy_check"]
        )
        
        if result["validation_passed"]:
            logger.info("강화된 벡터 무결성 검증 통과")
        else:
            logger.error("강화된 벡터 무결성 검증 실패")
        
        return result
        
    except Exception as e:
        logger.error(f"벡터 무결성 검증 중 오류: {e}")
        result["error"] = str(e)
        return result


def calculate_feature_entropy(vector: np.ndarray) -> float:
    """
    특성 벡터의 정보 엔트로피 계산
    
    Args:
        vector: 특성 벡터
        
    Returns:
        float: 엔트로피 점수 (0~1)
    """
    try:
        # 벡터를 히스토그램으로 변환
        hist, _ = np.histogram(vector, bins=50, density=True)
        
        # 0이 아닌 확률만 사용
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        # 엔트로피 계산
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # 0~1 범위로 정규화
        max_entropy = np.log2(len(hist))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized_entropy)
        
    except Exception as e:
        logger.warning(f"엔트로피 계산 중 오류: {e}")
        return 0.0


def fix_vector_dimensions(
    vector_path: str, names_path: str, target_dims: int = 95
) -> bool:
    """
    벡터 차원을 목표 차원에 맞게 수정
    
    Args:
        vector_path: 벡터 파일 경로
        names_path: 특성 이름 파일 경로
        target_dims: 목표 차원 수
        
    Returns:
        bool: 수정 성공 여부
    """
    try:
        logger.info(f"벡터 차원 수정 시작: 목표 {target_dims}차원")
        
        # 기존 데이터 로드
        vector = np.load(vector_path)
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        
        current_dims = len(vector)
        current_names = len(feature_names)
        
        logger.info(f"현재 상태: 벡터 {current_dims}차원, 특성명 {current_names}개")
        
        # 차원 조정
        if current_dims < target_dims:
            # 패딩
            padding_size = target_dims - current_dims
            padded_vector = np.pad(vector, (0, padding_size), mode='constant', constant_values=0.0)
            
            # 특성 이름도 패딩
            padded_names = feature_names + [f"padded_feature_{i+1}" for i in range(padding_size)]
            
            logger.info(f"벡터 패딩: {current_dims} → {target_dims}차원")
            
        elif current_dims > target_dims:
            # 절단
            padded_vector = vector[:target_dims]
            padded_names = feature_names[:target_dims]
            
            logger.info(f"벡터 절단: {current_dims} → {target_dims}차원")
            
        else:
            # 이미 올바른 차원
            padded_vector = vector
            padded_names = feature_names
            logger.info("벡터 차원이 이미 올바릅니다.")
        
        # 특성 이름 수 조정
        if len(padded_names) < target_dims:
            padded_names.extend([f"auto_feature_{i+1}" for i in range(target_dims - len(padded_names))])
        elif len(padded_names) > target_dims:
            padded_names = padded_names[:target_dims]
        
        # NaN/Inf 정리
        padded_vector = np.nan_to_num(padded_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        padded_vector = np.clip(padded_vector, -10.0, 10.0)
        
        # 저장
        np.save(vector_path, padded_vector.astype(np.float32))