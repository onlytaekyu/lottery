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
        from src.utils.unified_report import save_analysis_performance_report

        # 🔧 통합 성능 추적기 사용
        from src.utils.unified_performance import get_profiler
        from src.utils.unified_feature_vector_validator import (
            validate_feature_vector_with_config,
            check_vector_dimensions,
            create_feature_registry,
            check_feature_mapping_consistency,
            sync_vectors_and_names,
            ensure_essential_features,
            safe_float_conversion,
            ESSENTIAL_FEATURES,
            detect_outliers,
            save_outlier_information,
            analyze_vector_statistics,
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

        # 🔧 통합 성능 추적기 사용
        profiler = get_profiler()
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
            logger.info(
                f"ROI 패턴 그룹 ID 항목 수: {len(roi_features.get('roi_pattern_group_id', {}))}"
            )

            # 특성 이름에 roi_features 관련 항목이 있는지 확인
            roi_feature_count = sum(
                1 for name in feature_names if name.startswith("roi_")
            )
            if roi_feature_count > 0:
                logger.info(
                    f"ROI 관련 특성 {roi_feature_count}개가 벡터에 포함되어 있습니다."
                )
            else:
                logger.warning("벡터에 ROI 특성이 포함되어 있지 않습니다.")
        else:
            logger.warning(
                "분석 결과에 ROI 특성이 없습니다. ROI 분석기가 올바르게 실행되었는지 확인하세요."
            )

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
            # 🔧 통합 성능 추적기 사용
            "cache_hit_rate": profiler.get_cache_hit_rate(),
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
            # 🔧 통합 성능 추적기 사용
            profiler,
            profiler,
            config,
            "data_analysis",
            data_metrics,
        )
        logger.info(f"성능 보고서 저장 완료: {perf_file}")

        # 로그 출력
        logger.info(f"벡터 형태: {feature_vectors.shape}")
        logger.info(f"특성 개수: {len(feature_names)}")
        logger.info(f"전체 실행 시간: {time.time() - start_time:.2f}초")
        return 0
    except Exception as e:
        logger.error(f"데이터 분석 파이프라인 실행 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
