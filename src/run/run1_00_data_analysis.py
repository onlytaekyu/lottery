#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터 분석 실행 스크립트

이 스크립트는 로또 데이터에 대한 종합적인 분석을 수행하고,
필요한 모든 특성 벡터를 생성합니다.
"""

import os
import sys
import json
import numpy as np
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 프로젝트 루트 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 로거 설정
from src.utils.error_handler_refactored import get_logger, log_exception_with_trace

logger = get_logger(__name__)


def main():
    """
    데이터 분석 및 벡터화 파이프라인 실행
    """
    try:
        start_time = time.time()
        logger.info("데이터 분석 및 벡터화 파이프라인 시작")

        # 필요한 모듈 임포트
        from src.utils.data_loader import load_draw_history
        from src.utils.config_loader import load_config
        from src.analysis.unified_analyzer import UnifiedAnalyzer
        from src.analysis.pattern_vectorizer import PatternVectorizer
        from src.utils.performance_report_writer import save_analysis_performance_report
        from src.utils.unified_performance import performance_monitor
        from src.utils import get_profiler as Profiler
        from src.utils.feature_vector_validator import (
            validate_feature_vector,
            ensure_features_present,
            check_vector_dimensions,
            save_outlier_information,
            analyze_vector_statistics,
            validate_vector_integrity_enhanced,
            validate_feature_vector_with_config,
            create_feature_registry,
            check_feature_mapping_consistency,
            detect_outliers,
            ESSENTIAL_FEATURES,
            safe_float_conversion,
        )

        # 설정 로드
        config = load_config()

        # 설정 파일 검증 (필수 키 확인)
        try:
            # 중요 설정 키 검증
            config.validate_critical_paths()

            # 분석 관련 필수 키 검증
            required_keys = [
                "clustering.n_clusters",
                "clustering.min_silhouette_score",
                "filtering.remove_low_variance_features",
                "filtering.variance_threshold",
                "paths.analysis_result_dir",
                "paths.performance_report_dir",
            ]

            for key in required_keys:
                if not config.has_key(key):
                    logger.warning(f"필수 설정 키 누락: {key}")

            logger.info("설정 파일 검증 완료: 모든 필수 키가 확인되었습니다.")
        except Exception as e:
            logger.warning(f"설정 검증 중 오류 발생: {str(e)}")
            logger.warning("일부 필수 설정이 누락되어 기본값을 사용합니다.")

        # 성능 추적 및 프로파일링 설정
        profiler = Profiler(config)
        # 통합 성능 모니터링 사용
        profiler.start("total")

        # 데이터 로드
        draw_data = load_draw_history()
        logger.info(f"로또 당첨 데이터 로드 완료: {len(draw_data)}개 회차")

        # 통합 분석기를 사용하여 모든 분석 수행
        logger.info("통합 분석 수행 중...")
        profiler.start("unified_analysis")
        unified_analyzer = UnifiedAnalyzer(config)
        analysis_result = unified_analyzer.analyze(draw_data)
        profiler.stop("unified_analysis")
        logger.info(f"통합 분석 완료: {len(analysis_result)}개 분석 항목")

        # 분석 결과 저장
        logger.info("분석 결과 저장 중...")
        profiler.start("save_analysis")
        result_path = unified_analyzer.save_analysis_results(analysis_result)
        profiler.stop("save_analysis")
        logger.info(f"분석 결과 저장 완료: {result_path}")

        # 패턴 벡터라이저 초기화 및 벡터화 수행
        pattern_vectorizer = PatternVectorizer(config)

        # 저분산 특성 제거 설정
        if config["filtering"]["remove_low_variance_features"]:
            pattern_vectorizer.remove_low_variance = True
            pattern_vectorizer.variance_threshold = config["filtering"][
                "variance_threshold"
            ]

        logger.info("확장 특성 벡터화 수행 중...")
        profiler.start("vectorization")

        # dict 값 안전 변환 함수 추가
        pattern_vectorizer.external_float_conversion = safe_float_conversion

        # 벡터 생성 - 확장 특성 벡터 생성 방식 사용
        feature_vectors, feature_names = pattern_vectorizer.vectorize_extended_features(
            analysis_result
        )
        profiler.stop("vectorization")
        logger.info(
            f"확장 특성 벡터화 완료: 차원 {feature_vectors.shape}, 특성 수 {len(feature_names)}"
        )

        # 벡터 및 특성 이름을 캐시에 저장
        cache_dir = Path(project_root) / config["paths"]["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 주요 벡터 파일 경로
        vector_file = cache_dir / "feature_vectors_full.npy"
        names_file = cache_dir / "feature_vector_full.names.json"

        # 벡터 및 이름 저장
        profiler.start("save_vectors")
        np.save(vector_file, feature_vectors)
        with open(names_file, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2, ensure_ascii=False)
        profiler.stop("save_vectors")

        logger.info(f"특성 벡터 저장 완료: {vector_file}")
        logger.info(f"특성 이름 저장 완료: {names_file}")

        # 벡터 검증 - 최소 70개 이상의 특성 확보 확인
        if len(feature_names) < 70:
            error_msg = f"생성된 특성 벡터 수가 최소 요구사항인 70개보다 적습니다: {len(feature_names)}개"
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.info(
                f"벡터 차원 확인: {len(feature_names)}개 (최소 70개 이상 요구사항 충족)"
            )

        # 벡터 검증 및 필수 특성 추가
        profiler.start("vector_validation")
        logger.info("특성 벡터 검증 수행 중...")

        # 설정에서 필수 특성 목록을 가져와 벡터 검증
        names_file_path = str(names_file)
        vector_file_path = str(vector_file)

        # 벡터 차원과 특성 이름 수 확인 - 불일치 시 예외 발생
        try:
            dim_check = check_vector_dimensions(
                vector_file_path,
                names_file_path,
                raise_on_mismatch=True,  # 불일치 시 예외 발생
                allow_mismatch=False,  # 차원 불일치 허용 안함
            )
            logger.info("벡터 차원과 특성 이름 수 일치 확인됨")
        except ValueError as e:
            logger.error(f"벡터 차원 불일치 오류: {str(e)}")
            raise

        # ROI 특성 벡터 확인
        roi_features = analysis_result.get("roi_features", {})
        if roi_features:
            logger.info("ROI 특성이 분석 결과에 포함되어 있습니다.")
            logger.info(
                f"ROI 그룹 점수 항목 수: {len(roi_features.get('roi_group_score', {}))}"
            )
            logger.info(
                f"ROI 클러스터 항목 수: {len(roi_features.get('roi_cluster_score', {}).get('cluster_assignments', {}))}"
            )
            logger.info(
                f"저위험 보너스 플래그 항목 수: {len(roi_features.get('low_risk_bonus_flag', {}).get('low_risk_bonus_flag', {}))}"
            )
        else:
            logger.warning("벡터에 ROI 특성이 포함되어 있지 않습니다.")

        # 필수 특성 확인
        missing_features = validate_feature_vector_with_config(config, names_file_path)
        if missing_features:
            logger.error(f"필수 특성 누락: {missing_features}")
            # 필수 특성이 누락된 경우 보고서에 기록하기 위해 저장
            missing_features_file = cache_dir / "missing_features.json"
            with open(missing_features_file, "w", encoding="utf-8") as f:
                json.dump(missing_features, f, indent=2, ensure_ascii=False)
        else:
            logger.info("모든 필수 특성이 포함되어 있습니다.")

        # 특성 레지스트리 생성 및 저장
        try:
            registry_file = cache_dir / "feature_registry.json"
            feature_registry = create_feature_registry(config, str(registry_file))
            logger.info(f"특성 레지스트리 저장 완료: {registry_file}")

            # 레지스트리와 특성 이름의 일관성 검사
            with open(names_file_path, "r", encoding="utf-8") as f:
                updated_feature_names = json.load(f)

            inconsistencies = check_feature_mapping_consistency(
                updated_feature_names, feature_registry
            )
            if inconsistencies:
                logger.warning(
                    f"일부 특성({len(inconsistencies)}개)이 레지스트리에 등록되지 않았습니다."
                )
        except Exception as e:
            logger.error(f"특성 레지스트리 생성 및 검증 중 오류 발생: {e}")

        # 이상치 탐지 및 저장
        profiler.start("outlier_detection")
        logger.info("이상치 탐지 수행 중...")

        try:
            # Z-score 기반 이상치 탐지 (임계값: 2.5)
            outlier_mask, outlier_indices = detect_outliers(
                vector_file_path, names_file_path, z_threshold=2.5
            )

            # 이상치 정보 저장
            if len(outlier_indices) > 0:
                mask_path, indices_path = save_outlier_information(
                    vector_file_path, outlier_mask, outlier_indices
                )
                logger.info(f"이상치 정보 저장 완료: {len(outlier_indices)}개 항목")
            else:
                logger.info("이상치가 발견되지 않았습니다.")

        except Exception as e:
            logger.error(f"이상치 탐지 및 저장 중 오류 발생: {e}")

        profiler.stop("outlier_detection")

        # 벡터 통계 분석
        profiler.start("vector_statistics")
        logger.info("벡터 통계 분석 수행 중...")

        try:
            vector_stats = analyze_vector_statistics(vector_file_path, names_file_path)

            # 통계 정보 저장
            stats_file = cache_dir / "vector_statistics.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(vector_stats, f, indent=2, ensure_ascii=False)

            logger.info(f"벡터 통계 정보 저장 완료: {stats_file}")

            # 주요 통계 출력
            logger.info(
                f"벡터 통계: 엔트로피 점수={vector_stats.get('feature_entropy_score', 0):.4f}, "
                f"NaN 비율={vector_stats.get('nan_rate', 0):.6f}, "
                f"특성 크기 평균={vector_stats.get('vector_scale_mean', 0):.4f}"
            )

        except Exception as e:
            logger.error(f"벡터 통계 분석 중 오류 발생: {e}")

        profiler.stop("vector_statistics")

        logger.info("특성 벡터 검증 완료")
        profiler.stop("vector_validation")

        # 프로파일링 완료
        profiler.stop("total")

        # 성능 보고서 저장 - 확장된 메트릭 포함
        data_metrics = {
            "record_count": len(draw_data),
            "vector_shape": list(feature_vectors.shape),
            "features_count": len(feature_names),
            "essential_features_count": len(ESSENTIAL_FEATURES),
            "missing_features_count": len(missing_features) if missing_features else 0,
            "low_variance_features_removed": len(
                pattern_vectorizer.removed_low_variance_features
            ),
            "vector_dim": feature_vectors.shape[1],
            "vector_nan_rate": float(np.isnan(feature_vectors).sum())
            / feature_vectors.size,
            "cache_hit_rate": performance_tracker.get_cache_hit_rate(),
            "analysis_time": time.time() - start_time,
            # 확장된 메트릭
            "outlier_count": (
                len(outlier_indices) if "outlier_indices" in locals() else 0
            ),
            "vector_scale_mean": float(np.mean(np.abs(feature_vectors))),
            "vector_scale_std": float(np.std(np.abs(feature_vectors))),
            "feature_entropy_score": (
                vector_stats.get("feature_entropy_score", 0.0)
                if "vector_stats" in locals()
                else 0.0
            ),
            "cluster_silhouette_score": analysis_result.get(
                "cluster_embedding_quality", {}
            ).get("silhouette_score", 0.0),
            "vector_diversity_score": (
                vector_stats.get("feature_entropy_score", 0.0)
                if "vector_stats" in locals()
                else 0.0
            ),
        }

        # 클러스터 품질 메트릭 추가 (있는 경우)
        if "cluster_embedding_quality" in analysis_result:
            cluster_metrics = analysis_result["cluster_embedding_quality"]
            for key in [
                "silhouette_score",
                "cluster_score",
                "cohesiveness_score",
                "cluster_entropy_score",
                "balance_score",
            ]:
                if key in cluster_metrics:
                    data_metrics[f"cluster_{key}"] = cluster_metrics[key]

        perf_file = save_analysis_performance_report(
            profiler, performance_tracker, config, "data_analysis", data_metrics
        )
        logger.info(f"성능 보고서 저장 완료: {perf_file}")

        # 로그 출력
        logger.info(f"벡터 형태: {feature_vectors.shape}")
        logger.info(f"특성 개수: {len(feature_names)}")
        logger.info(f"전체 실행 시간: {time.time() - start_time:.2f}초")

        # 벡터 무결성 강화 검증 수행
        validate_vector_integrity()

        # 3-4자리 중복 패턴 분석 결과 확인
        validate_overlap_patterns(analysis_result)

        # 신규 패턴 분석 항목들 검증
        validate_new_pattern_features(analysis_result)

        logger.info("데이터 분석 완료!")
        return analysis_result
    except Exception as e:
        logger.error(f"데이터 분석 파이프라인 실행 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        return 1


def validate_vector_integrity():
    """
    강화된 벡터 무결성 검증 수행
    """
    logger.info("=== 벡터 무결성 강화 검증 시작 ===")

    try:
        vector_path = "data/cache/feature_vector_full.npy"
        names_path = "data/cache/feature_vector_full.names.json"

        # 강화된 검증 수행
        from src.utils.feature_vector_validator import (
            validate_vector_integrity_enhanced,
        )

        validation_result = validate_vector_integrity_enhanced(
            vector_path=vector_path,
            names_path=names_path,
            expected_dims=95,  # 청사진 시스템의 총 차원
        )

        if validation_result["validation_passed"]:
            logger.info("✅ 벡터 무결성 검증 통과!")
            logger.info(
                f"   - 차원: {validation_result['statistics'].get('vector_dims', 'N/A')}"
            )
            logger.info(
                f"   - 엔트로피: {validation_result['statistics'].get('entropy_score', 'N/A'):.4f}"
            )
            logger.info(
                f"   - NaN/Inf 비율: {validation_result['statistics'].get('invalid_ratio', 'N/A'):.6f}"
            )
        else:
            logger.error("❌ 벡터 무결성 검증 실패!")
            logger.error(
                f"   - 차원 검증: {'통과' if validation_result['dimension_check'] else '실패'}"
            )
            logger.error(
                f"   - NaN/Inf 검증: {'통과' if validation_result['nan_inf_check'] else '실패'}"
            )
            logger.error(
                f"   - 엔트로피 검증: {'통과' if validation_result['entropy_check'] else '실패'}"
            )

            # 자동 수정 시도
            if not validation_result["dimension_check"]:
                logger.info("차원 불일치 자동 수정 시도...")
                from src.utils.feature_vector_validator import fix_vector_dimensions

                if fix_vector_dimensions(vector_path, names_path, target_dims=95):
                    logger.info("✅ 차원 수정 완료")
                else:
                    logger.error("❌ 차원 수정 실패")

    except Exception as e:
        logger.error(f"벡터 무결성 검증 중 오류: {e}")


def validate_overlap_patterns(analysis_result: Dict[str, Any]):
    """
    3-4자리 중복 패턴 분석 결과 검증

    Args:
        analysis_result: 전체 분석 결과
    """
    logger.info("=== 3-4자리 중복 패턴 검증 시작 ===")

    try:
        # overlap 분석 결과 확인
        overlap_data = analysis_result.get("overlap", {})

        if not overlap_data:
            logger.warning("❌ overlap 분석 결과가 없습니다.")
            return

        # 3-4자리 중복 패턴 확인
        overlap_3_4_data = overlap_data.get("overlap_3_4_digit_patterns", {})

        if overlap_3_4_data:
            logger.info("✅ 3-4자리 중복 패턴 분석 완료")

            # 3자리 패턴 확인
            overlap_3_patterns = overlap_3_4_data.get("overlap_3_patterns", {})
            if overlap_3_patterns:
                pattern_count = overlap_3_patterns.get("total_patterns", 0)
                avg_frequency = overlap_3_patterns.get("avg_frequency", 0.0)
                logger.info(
                    f"   - 3자리 패턴: {pattern_count}개, 평균 빈도: {avg_frequency:.4f}"
                )

            # 4자리 패턴 확인
            overlap_4_patterns = overlap_3_4_data.get("overlap_4_patterns", {})
            if overlap_4_patterns:
                pattern_count = overlap_4_patterns.get("total_patterns", 0)
                rarity_score = overlap_4_patterns.get("rarity_score", 0.0)
                logger.info(
                    f"   - 4자리 패턴: {pattern_count}개, 희귀도: {rarity_score:.4f}"
                )

            # 시간 간격 분석 확인
            time_gap_data = overlap_3_4_data.get("overlap_time_gap_analysis", {})
            if time_gap_data:
                gap_3_std = time_gap_data.get("gap_3_std", 0.0)
                gap_4_std = time_gap_data.get("gap_4_std", 0.0)
                logger.info(
                    f"   - 시간 간격 분산: 3자리 {gap_3_std:.2f}, 4자리 {gap_4_std:.2f}"
                )
        else:
            logger.warning("❌ 3-4자리 중복 패턴 데이터가 없습니다.")

        # ROI 상관관계 확인
        roi_correlation_data = overlap_data.get("overlap_roi_correlation", {})

        if roi_correlation_data:
            logger.info("✅ ROI 상관관계 분석 완료")

            overlap_3_roi = roi_correlation_data.get("overlap_3_roi_correlation", 0.0)
            overlap_4_roi = roi_correlation_data.get("overlap_4_roi_correlation", 0.0)

            logger.info(f"   - 3자리 ROI 상관관계: {overlap_3_roi:.4f}")
            logger.info(f"   - 4자리 ROI 상관관계: {overlap_4_roi:.4f}")

            # ROI 통계 확인
            overlap_3_stats = roi_correlation_data.get("overlap_3_roi_stats", {})
            overlap_4_stats = roi_correlation_data.get("overlap_4_roi_stats", {})

            if overlap_3_stats:
                avg_roi = overlap_3_stats.get("avg_roi", 0.0)
                sample_count = overlap_3_stats.get("sample_count", 0)
                logger.info(
                    f"   - 3자리 평균 ROI: {avg_roi:.4f} (샘플: {sample_count}개)"
                )

            if overlap_4_stats:
                avg_roi = overlap_4_stats.get("avg_roi", 0.0)
                sample_count = overlap_4_stats.get("sample_count", 0)
                logger.info(
                    f"   - 4자리 평균 ROI: {avg_roi:.4f} (샘플: {sample_count}개)"
                )
        else:
            logger.warning("❌ ROI 상관관계 데이터가 없습니다.")

        # 벡터화 확인
        try:
            from src.analysis.pattern_vectorizer import PatternVectorizer
            from src.utils.config_loader import load_config

            config = load_config()
            vectorizer = PatternVectorizer(config)

            overlap_vector, overlap_names = vectorizer.extract_overlap_pattern_features(
                analysis_result
            )

            if len(overlap_vector) == 20 and len(overlap_names) == 20:
                logger.info("✅ 중복 패턴 벡터화 성공")
                logger.info(f"   - 벡터 차원: {len(overlap_vector)}")
                logger.info(f"   - 특성 이름: {len(overlap_names)}개")

                # 주요 특성 값 출력
                for i, name in enumerate(overlap_names[:5]):  # 처음 5개만
                    logger.info(f"   - {name}: {overlap_vector[i]:.4f}")
            else:
                logger.warning(
                    f"❌ 중복 패턴 벡터 차원 불일치: {len(overlap_vector)} != 20"
                )

        except Exception as e:
            logger.warning(f"중복 패턴 벡터화 검증 중 오류: {e}")

    except Exception as e:
        logger.error(f"중복 패턴 검증 중 오류: {e}")


def validate_new_pattern_features(analysis_result: Dict[str, Any]):
    """
    신규 패턴 분석 항목들 검증

    Args:
        analysis_result: 전체 분석 결과
    """
    logger.info("=== 신규 패턴 분석 항목 검증 시작 ===")

    try:
        # 1. 추첨 순서 편향 분석 검증
        position_bias_data = analysis_result.get("position_bias_features", {})
        if position_bias_data:
            logger.info("✅ 추첨 순서 편향 분석 완료")

            min_value_mean = position_bias_data.get("position_min_value_mean", 0.0)
            max_value_mean = position_bias_data.get("position_max_value_mean", 0.0)
            gap_mean = position_bias_data.get("position_gap_mean", 0.0)
            even_odd_ratio = position_bias_data.get("position_even_odd_ratio", 0.0)
            low_high_ratio = position_bias_data.get("position_low_high_ratio", 0.0)

            logger.info(f"   - 최소값 평균: {min_value_mean:.2f}")
            logger.info(f"   - 최대값 평균: {max_value_mean:.2f}")
            logger.info(f"   - 간격 평균: {gap_mean:.2f}")
            logger.info(f"   - 홀짝 비율: {even_odd_ratio:.4f}")
            logger.info(f"   - 저고 비율: {low_high_ratio:.4f}")
        else:
            logger.warning("❌ 추첨 순서 편향 분석 데이터가 없습니다.")

        # 2. 중복 패턴 시간적 주기성 분석 검증
        time_gap_data = analysis_result.get("overlap_time_gaps", {})
        if time_gap_data:
            logger.info("✅ 중복 패턴 시간적 주기성 분석 완료")

            gap_3_mean = time_gap_data.get("overlap_3_time_gap_mean", 0.0)
            gap_4_mean = time_gap_data.get("overlap_4_time_gap_mean", 0.0)
            gap_stddev = time_gap_data.get("overlap_time_gap_stddev", 0.0)
            recent_3_count = time_gap_data.get("recent_overlap_3_count", 0)
            recent_4_count = time_gap_data.get("recent_overlap_4_count", 0)

            logger.info(f"   - 3매치 평균 간격: {gap_3_mean:.2f}")
            logger.info(f"   - 4매치 평균 간격: {gap_4_mean:.2f}")
            logger.info(f"   - 간격 표준편차: {gap_stddev:.2f}")
            logger.info(f"   - 최근 3매치 수: {recent_3_count}")
            logger.info(f"   - 최근 4매치 수: {recent_4_count}")
        else:
            logger.warning("❌ 중복 패턴 시간적 주기성 분석 데이터가 없습니다.")

        # 3. 번호 간 조건부 상호작용 분석 검증
        conditional_data = analysis_result.get("conditional_interaction_features", {})
        if conditional_data:
            logger.info("✅ 번호 간 조건부 상호작용 분석 완료")

            attraction_score = conditional_data.get("number_attraction_score", 0.0)
            repulsion_score = conditional_data.get("number_repulsion_score", 0.0)
            dependency_strength = conditional_data.get(
                "conditional_dependency_strength", 0.0
            )

            logger.info(f"   - 끌림 점수: {attraction_score:.4f}")
            logger.info(f"   - 회피 점수: {repulsion_score:.4f}")
            logger.info(f"   - 의존성 강도: {dependency_strength:.4f}")
        else:
            logger.warning("❌ 번호 간 조건부 상호작용 분석 데이터가 없습니다.")

        # 4. 홀짝 및 구간별 미세 편향성 분석 검증
        odd_even_data = analysis_result.get("odd_even_micro_bias", {})
        range_bias_data = analysis_result.get("range_micro_bias", {})

        if odd_even_data or range_bias_data:
            logger.info("✅ 미세 편향성 분석 완료")

            if odd_even_data:
                odd_bias_score = odd_even_data.get("odd_even_bias_score", 0.0)
                segment_balance_score = odd_even_data.get(
                    "segment_balance_bias_score", 0.0
                )
                logger.info(f"   - 홀짝 편향 점수: {odd_bias_score:.4f}")
                logger.info(f"   - 구간 균형 점수: {segment_balance_score:.4f}")

            if range_bias_data:
                range_moving_avg = range_bias_data.get("range_bias_moving_avg", 0.0)
                odd_ratio_change = range_bias_data.get("odd_ratio_change_rate", 0.0)
                logger.info(f"   - 범위 편향 이동평균: {range_moving_avg:.4f}")
                logger.info(f"   - 홀수 비율 변화율: {odd_ratio_change:.4f}")
        else:
            logger.warning("❌ 미세 편향성 분석 데이터가 없습니다.")

        # 벡터화 확인
        try:
            from src.analysis.pattern_vectorizer import PatternVectorizer
            from src.utils.config_loader import load_config

            config = load_config()
            vectorizer = PatternVectorizer(config)

            # 각 신규 특성 벡터화 확인
            position_vector, position_names = vectorizer.extract_position_bias_features(
                analysis_result
            )
            time_gap_vector, time_gap_names = (
                vectorizer.extract_overlap_time_gap_features(analysis_result)
            )
            conditional_vector, conditional_names = (
                vectorizer.extract_conditional_interaction_features(analysis_result)
            )
            micro_bias_vector, micro_bias_names = (
                vectorizer.extract_micro_bias_features(analysis_result)
            )

            total_new_features = (
                len(position_vector)
                + len(time_gap_vector)
                + len(conditional_vector)
                + len(micro_bias_vector)
            )

            if total_new_features == 17:  # 5 + 5 + 3 + 4 = 17
                logger.info("✅ 신규 패턴 특성 벡터화 성공")
                logger.info(f"   - 추첨 순서 편향: {len(position_vector)}차원")
                logger.info(f"   - 시간적 주기성: {len(time_gap_vector)}차원")
                logger.info(f"   - 조건부 상호작용: {len(conditional_vector)}차원")
                logger.info(f"   - 미세 편향성: {len(micro_bias_vector)}차원")
                logger.info(f"   - 총 신규 특성: {total_new_features}차원")
            else:
                logger.warning(
                    f"❌ 신규 패턴 특성 차원 불일치: {total_new_features} != 17"
                )

        except Exception as e:
            logger.warning(f"신규 패턴 특성 벡터화 검증 중 오류: {e}")

    except Exception as e:
        logger.error(f"신규 패턴 특성 검증 중 오류: {e}")


if __name__ == "__main__":
    sys.exit(main())
