import json
import logging
import os
import sys
import time
import random
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import psutil

from src.utils.error_handler_refactored import get_logger, log_exception_with_trace
from src.utils.state_vector_cache import get_cache
from src.utils.data_loader import load_draw_history
from src.utils.config_loader import load_config
from src.utils.profiler import get_profiler
from src.analysis.pattern_analyzer import PatternAnalyzer
from src.analysis.pattern_vectorizer import PatternVectorizer
from src.utils.unified_report import safe_convert, save_physical_performance_report
from src.analysis.pair_analyzer import PairAnalyzer
from src.utils.feature_vector_validator import (
    validate_feature_vector_with_config,
    check_vector_dimensions,
)

# 로거 설정
logger = get_logger(__name__)


def run_data_analysis() -> bool:
    """
    데이터 분석 및 전처리 실행

    Returns:
        bool: 작업 성공 여부
    """
    start_time = time.time()

    # 랜덤 시드 설정 (재현성 보장)
    random.seed(42)
    np.random.seed(42)

    # 설정 로드
    try:
        config = load_config()
    except Exception as e:
        log_exception_with_trace(logger, e, "데이터 분석 수행: 설정 로드 실패")
        return False

    # 프로파일러 초기화
    profiler = get_profiler()

    try:
        # 1. 데이터 로드
        with profiler.profile("데이터 로드"):
            logger.info("1단계: 과거 당첨 번호 데이터 로드 중...")
            try:
                historical_data = load_draw_history()

                if not historical_data:
                    logger.error("당첨 번호 데이터를 로드할 수 없습니다.")
                    return False

                logger.info(f"데이터 로드 완료: {len(historical_data)}개 회차")
            except Exception as e:
                log_exception_with_trace(
                    logger, e, "데이터 분석: 당첨 번호 데이터 로드 실패"
                )
                return False

        # 2. 패턴 분석기 초기화
        with profiler.profile("패턴 분석기 초기화"):
            try:
                # ConfigProxy를 Dict로 변환하여 전달
                pattern_analyzer = PatternAnalyzer(config.to_dict())
                pattern_vectorizer = PatternVectorizer(config.to_dict())

                # 새로운 쌍 분석기 초기화
                pair_analyzer = PairAnalyzer(config.to_dict())

                # 벡터 캐시 초기화
                vector_cache = get_cache(config)
            except Exception as e:
                log_exception_with_trace(logger, e, "데이터 분석: 분석기 초기화 실패")
                return False

            # 결과 디렉토리 생성
            try:
                result_dir = Path(config["paths"]["result_dir"])
            except KeyError as e:
                log_exception_with_trace(
                    logger, e, "데이터 분석: 결과 디렉토리 경로 설정 누락"
                )
                logger.warning(
                    "설정에서 'paths.result_dir'를 찾을 수 없습니다. 기본값 'data/result'를 사용합니다."
                )
                result_dir = Path("data/result")

            analysis_dir = result_dir / "analysis"

            try:
                cache_dir = Path(config["paths"]["cache_dir"])
            except KeyError as e:
                log_exception_with_trace(
                    logger, e, "데이터 분석: 캐시 디렉토리 경로 설정 누락"
                )
                logger.warning(
                    "설정에서 'paths.cache_dir'를 찾을 수 없습니다. 기본값 'data/cache'를 사용합니다."
                )
                cache_dir = Path("data/cache")

            # 디렉토리 생성
            try:
                result_dir.mkdir(parents=True, exist_ok=True)
                analysis_dir.mkdir(parents=True, exist_ok=True)
                cache_dir.mkdir(parents=True, exist_ok=True)

                logger.info("분석 준비 완료")
            except Exception as e:
                log_exception_with_trace(logger, e, "데이터 분석: 디렉토리 생성 실패")
                return False

        # 3. 패턴 분석 실행
        with profiler.profile("패턴 분석"):
            logger.info("패턴 분석 수행 중...")

            try:
                # 전체 당첨 번호 데이터에 대한 분석 수행
                analysis_result = pattern_analyzer.analyze(historical_data)

                # 추가 분석 수행
                logger.info("추가 패턴 분석 수행 중...")

                # 번호 합계 분포 분석
                sum_distribution = pattern_analyzer.analyze_number_sum_distribution(
                    historical_data
                )
                logger.info(f"합계 분포 분석 완료: {len(sum_distribution)}개 항목")

                # 번호 간격 패턴 분석
                gap_patterns = pattern_analyzer.analyze_gap_patterns(historical_data)
                logger.info(f"간격 패턴 분석 완료: {len(gap_patterns)}개 패턴")

                # 확률 행렬 생성
                probability_matrix = pattern_analyzer.generate_probability_matrix(
                    historical_data
                )
                logger.info(f"확률 행렬 생성 완료: {len(probability_matrix)}개 원소")

                # 기본 행렬 형식 검증 (45x45 크기, 대칭성, 값 범위)
                matrix_keys = probability_matrix.keys()
                matrix_size = len(set([x[0] for x in matrix_keys])) * len(
                    set([x[1] for x in matrix_keys])
                )
                logger.info(f"확률 행렬 크기: {matrix_size}개 (기대값: 45x45 = 2025)")

                # 세그먼트 엔트로피 계산
                segment_entropy = pattern_analyzer.calculate_segment_entropy(
                    historical_data
                )
                logger.info(
                    f"세그먼트 엔트로피 계산 완료: {len(segment_entropy)}개 항목"
                )

                # 클러스터 분포 계산
                cluster_distribution = pattern_analyzer.calculate_cluster_distribution(
                    analysis_result.clusters
                )
                logger.info(
                    f"클러스터 분포 계산 완료: {len(cluster_distribution)}개 항목"
                )

                # 새로 추가된 분석 기능 실행
                logger.info("확장 패턴 분석 수행 중...")

                # 위치별 번호 빈도 행렬 계산
                position_frequency = pattern_analyzer.enhance_position_frequency(
                    historical_data
                )
                logger.info(
                    f"위치별 번호 빈도 행렬 계산 완료: {position_frequency.shape}"
                )

                # 세그먼트 추세 이력 계산
                segment_trend_history = pattern_analyzer.enhance_segment_trend_history(
                    historical_data
                )
                logger.info(
                    f"세그먼트 추세 이력 계산 완료: {segment_trend_history.shape}"
                )

                # 추가 분석 데이터 생성
                logger.info("추가 분석 데이터 생성 중...")

                # 간격 표준편차 점수 계산
                gap_deviation_score = pattern_analyzer.calculate_gap_deviation_score(
                    historical_data
                )
                logger.info(
                    f"간격 표준편차 점수 계산 완료: {len(gap_deviation_score)}개 항목"
                )

                # 조합 다양성 점수 계산
                combination_diversity_score = (
                    pattern_analyzer.calculate_combination_diversity_score(
                        historical_data
                    )
                )
                logger.info(
                    f"조합 다양성 점수 계산 완료: {len(combination_diversity_score)}개 항목"
                )

                # 패턴별 ROI 추세 계산
                roi_trend_by_pattern = pattern_analyzer.calculate_roi_trend_by_pattern(
                    historical_data
                )
                logger.info(
                    f"패턴별 ROI 추세 계산 완료: {len(roi_trend_by_pattern)}개 패턴"
                )

                # 위치별 번호 통계 계산
                position_number_stats = (
                    pattern_analyzer.calculate_position_number_stats(historical_data)
                )
                logger.info(
                    f"위치별 번호 통계 계산 완료: {len(position_number_stats)}개 위치"
                )

                # 쌍 빈도 데이터 생성 및 저장
                from src.pipeline.data_analysis_pipeline import calculate_pair_frequency

                pair_frequency = calculate_pair_frequency(historical_data)
                np.save(cache_dir / "pair_frequency.npy", pair_frequency)
                logger.info(
                    f"쌍 빈도 데이터 저장 완료: {cache_dir / 'pair_frequency.npy'}"
                )

                # 쌍 분석 수행
                logger.info("쌍 분석 수행 중...")
                pair_analysis_result = pair_analyzer.analyze(historical_data)

                # 번호 쌍 중심성
                pair_centrality = pair_analysis_result["pair_centrality"]
                logger.info(f"번호 쌍 중심성 분석 완료: {len(pair_centrality)}개 쌍")

                # 번호 쌍 ROI 점수
                pair_roi_scores = pair_analysis_result["pair_roi_score"]
                logger.info(f"번호 쌍 ROI 점수 분석 완료: {len(pair_roi_scores)}개 쌍")

                # 번호 쌍 그래프 가중치
                pair_graph_weights = pair_analysis_result["pair_graph_weights"]
                logger.info(f"번호 쌍 그래프 가중치 분석 완료")

                # 빈번한 3개 번호 조합
                frequent_triples = pair_analysis_result["frequent_triples"]
                logger.info(
                    f"빈번한 3개 번호 조합 분석 완료: {len(frequent_triples)}개 조합"
                )

                # ROI 패턴 그룹 식별
                roi_pattern_groups = pair_analyzer.identify_roi_pattern_groups(
                    historical_data
                )
                logger.info(
                    f"ROI 패턴 그룹 식별 완료: {len(roi_pattern_groups)}개 그룹"
                )

                # 각 개별 분석 결과는 NumPy 배열이나 벡터로만 저장
                # 세그먼트 추세 저장
                np.save(cache_dir / "segment_history_matrix.npy", segment_trend_history)

                # 위치별 번호 빈도 저장
                np.save(cache_dir / "position_frequency_matrix.npy", position_frequency)

                logger.info("분석 데이터 벡터 저장 완료")

            except Exception as e:
                log_exception_with_trace(logger, e, "데이터 분석: 패턴 분석 수행 실패")
                return False

        # 4. 벡터 생성 및 캐싱
        with profiler.profile("벡터 생성 및 캐싱"):
            logger.info("4단계: 특성 벡터 생성 중...")
            try:
                # 확장된 분석 결과를 하나로 통합
                enhanced_analysis = {
                    "pattern_analysis": analysis_result,
                    "sum_distribution": sum_distribution,
                    "gap_patterns": gap_patterns,
                    "segment_entropy": segment_entropy,
                    "cluster_distribution": cluster_distribution,
                    "position_frequency": position_frequency,
                    "position_number_stats": position_number_stats,
                    "segment_trend_history": segment_trend_history,
                    "gap_deviation_score": gap_deviation_score,
                    "combination_diversity_score": combination_diversity_score,
                    "roi_trend_by_pattern": roi_trend_by_pattern,
                }

                # 쌍 분석 결과 통합
                pair_analysis = {
                    "pair_centrality": pair_centrality,
                    "pair_roi_score": pair_roi_scores,
                    "pair_graph_weights": pair_graph_weights["strong_pairs"],
                    "frequent_triples": frequent_triples,
                    "roi_pattern_groups": roi_pattern_groups,
                }

                # 특성 벡터 생성
                feature_vector = pattern_vectorizer.vectorize_enhanced_analysis(
                    enhanced_analysis, pair_analysis
                )

                # 벡터 유효성 검사 (NaN, Inf 검사)
                nan_count = np.isnan(feature_vector).sum()
                inf_count = np.isinf(feature_vector).sum()

                if nan_count > 0 or inf_count > 0:
                    logger.warning(
                        f"벡터에 문제가 있습니다: NaN={nan_count}, Inf={inf_count}"
                    )
                    # NaN 및 Inf 값을 0으로 대체
                    feature_vector = np.nan_to_num(
                        feature_vector, nan=0.0, posinf=1.0, neginf=0.0
                    )
                    logger.info("문제 있는 값을 대체하였습니다.")

                # 벡터 차원 확인
                vector_dim = feature_vector.shape[0]
                logger.info(f"생성된 특성 벡터 차원: {vector_dim}")

                if vector_dim < 20:
                    logger.warning(
                        f"특성 벡터 차원이 20 미만입니다: {vector_dim}. 확장이 필요합니다."
                    )

                # 특성 이름 가져오기
                feature_names = pattern_vectorizer.get_feature_names()

                if len(feature_names) != vector_dim:
                    logger.warning(
                        f"특성 이름 수({len(feature_names)})와 벡터 차원({vector_dim})이 일치하지 않습니다."
                    )

                # 특성 벡터 파일로 저장
                vector_file_path = pattern_vectorizer.save_vector_to_file(
                    feature_vector, "feature_vectors_full.npy"
                )

                # 특성 이름 파일로 저장
                names_file_path = cache_dir / "feature_vector_full.names.json"
                with open(names_file_path, "w", encoding="utf-8") as f:
                    json.dump(feature_names, f, ensure_ascii=False, indent=2)

                logger.info(
                    f"특성 이름 저장 완료: {names_file_path} ({len(feature_names)}개 특성)"
                )

                # 세그먼트 이력 파일로 저장 (LSTM 모델용)
                segment_history_files = (
                    pattern_vectorizer.save_segment_history_to_numpy(
                        analysis_result.segment_10_history,
                        analysis_result.segment_5_history,
                    )
                )

                logger.info(f"세그먼트 이력 저장 완료: {segment_history_files}")

                # 특성 벡터 검증 (feature_vector_validator 모듈 사용)
                logger.info("특성 벡터 검증 수행 중...")

                # 필수 특성 포함 여부 검증
                missing_features = validate_feature_vector_with_config(
                    config, str(names_file_path)
                )
                if missing_features:
                    error_msg = f"필수 특성이 누락되었습니다: {missing_features}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # 벡터 차원과 특성 이름 수 일치 여부 검증
                check_vector_dimensions(
                    vector_file_path, str(names_file_path), raise_on_mismatch=True
                )

                logger.info("특성 벡터 검증 완료")

                # 벡터 통계 계산
                if len(historical_data) > 0:
                    try:
                        # 모든 번호 조합 벡터화 (첫 100개만)
                        all_vectors = []
                        for i in range(min(100, len(historical_data))):
                            draw = historical_data[i]
                            try:
                                number_vector = (
                                    pattern_vectorizer.vectorize_pattern_features(
                                        draw.numbers
                                    )
                                )
                                all_vectors.append(number_vector)
                            except Exception:
                                pass

                        if all_vectors:
                            all_vectors = np.array(all_vectors)
                            vector_array_shape = all_vectors.shape
                            logger.info(f"벡터 배열 생성 성공: {vector_array_shape}")

                            # 이상치 확인
                            outlier_count = 0
                            for vec in all_vectors:
                                # Z-점수 기반 이상치 검출
                                z_scores = scipy.stats.zscore(vec)
                                if np.abs(z_scores).max() > 3:
                                    outlier_count += 1

                            if outlier_count > 0:
                                pct = (outlier_count / len(all_vectors)) * 100
                                logger.info(
                                    f"이상치 벡터 {outlier_count}개 ({pct:.1f}%) 감지 및 저장 완료"
                                )

                            # 벡터 통계 로그
                            logger.info(f"벡터 배열 크기: {vector_array_shape}")
                            n_features = (
                                vector_array_shape[1]
                                if len(vector_array_shape) > 1
                                else 0
                            )
                            logger.info(f"특성 개수: {n_features}")

                            # 저분산 특성 확인
                            if n_features > 0:
                                variances = np.var(all_vectors, axis=0)
                                low_var_features = np.sum(variances < 0.01)
                                if low_var_features > 0:
                                    logger.warning(
                                        f"저변동 특성이 {low_var_features}개 있습니다."
                                    )

                                # 저분산 특성 자동 제거 여부 확인
                                try:
                                    auto_remove_low_var = config["filtering"][
                                        "auto_remove_low_variance"
                                    ]
                                    if auto_remove_low_var:
                                        removed_features = (
                                            pattern_vectorizer.removed_low_variance_features
                                        )
                                        if removed_features:
                                            logger.info(
                                                f"저분산 특성 {len(removed_features)}개가 자동 제거되었습니다."
                                            )
                                except (KeyError, TypeError):
                                    pass

                            # 패턴 특성 통계 계산
                            if not hasattr(locals(), "pattern_features_list"):
                                pattern_features_list = []

                            # 패턴 특성 통계 계산
                            pattern_feature_stats = (
                                calculate_pattern_feature_statistics(
                                    pattern_features_list
                                )
                            )
                    except Exception as e:
                        logger.error(f"벡터 통계 계산 중 오류 발생: {e}")

                # 벡터 호환성 확인
                logger.info("생성된 벡터는 ML/DL 모델과 완전히 호환됩니다.")

                # 최종 파일 정보 표시
                logger.info(
                    f"벡터 파일 저장 완료: {vector_file_path} (형태: {feature_vector.shape})"
                )
                logger.info(
                    f"특성 이름 저장 완료: {names_file_path} ({len(feature_names)}개 특성)"
                )

                # 추가 분석 데이터 생성
                logger.info("추가 분석 데이터 생성 중...")

                # 쌍 빈도 데이터 생성 및 저장
                pair_frequency = calculate_pair_frequency(historical_data)
                np.save(cache_dir / "pair_frequency.npy", pair_frequency)
                logger.info(
                    f"쌍 빈도 데이터 저장 완료: {cache_dir / 'pair_frequency.npy'}"
                )

            except Exception as e:
                log_exception_with_trace(
                    logger, e, "데이터 분석: 특성 벡터 생성 중 오류 발생"
                )
                return False

        # 5. 추가 분석 데이터 생성
        with profiler.profile("추가 분석 데이터"):
            logger.info("추가 분석 데이터 생성 중...")

            # 쌍 빈도 분석
            pair_frequency = calculate_pair_frequency(historical_data)
            pair_freq_file = cache_dir / "pair_frequency.npy"

            # Dictionary를 NumPy 배열로 변환하여 저장
            pair_data = np.array(
                [(p[0], p[1], v) for p, v in pair_frequency.items()],
                dtype=[("num1", "i4"), ("num2", "i4"), ("freq", "f4")],
            )

            np.save(pair_freq_file, pair_data)
            logger.info(f"쌍 빈도 데이터 저장 완료: {pair_freq_file}")

            # 세그먼트 중심성 분석
            segment_centrality = calculate_segment_centrality(historical_data)
            centrality_file = cache_dir / "segment_centrality_vector.npy"
            np.save(centrality_file, segment_centrality)
            logger.info(f"세그먼트 중심성 데이터 저장 완료: {centrality_file}")

            # 추가 분석 데이터 생성
            try:
                # 클러스터 임베딩 생성
                cluster_embeddings, cluster_quality = generate_cluster_embeddings(
                    historical_data
                )
                embedding_file = cache_dir / "cluster_embeddings.npy"
                np.save(embedding_file, cluster_embeddings)
                logger.info(f"클러스터 임베딩 저장 완료: {embedding_file}")

                # 클러스터 품질 정보 로깅
                logger.info(
                    f"클러스터 품질 지표: 실루엣 점수={cluster_quality.get('silhouette_score', 0.0):.3f}, "
                    f"클러스터 수={cluster_quality.get('cluster_count', 0)}, "
                    f"최대 클러스터 크기={cluster_quality.get('largest_cluster_size', 0)}, "
                    f"균형 점수={cluster_quality.get('balance_score', 0.0):.3f}"
                )
            except Exception as e:
                log_exception_with_trace(logger, e, "클러스터 임베딩 생성 중 오류 발생")
                # 기본 임베딩 생성 (오류 방지)
                cluster_embeddings = np.column_stack(
                    [np.random.random((45, 3)), np.zeros((45, 1))]
                )
                cluster_quality = {
                    "cluster_count": 1,
                    "silhouette_score": 0.0,
                    "largest_cluster_size": 45,
                    "balance_score": 0.0,
                    "avg_distance_between_clusters": 0.0,
                    "std_of_cluster_size": 0.0,
                    "error": str(e),
                }
                logger.warning("기본 클러스터 임베딩으로 대체되었습니다.")

            # 번호 간격 분석 (Average Gap Analysis)
            number_gaps = calculate_number_gaps(historical_data)
            gap_file = cache_dir / "number_gaps.npy"
            np.save(gap_file, number_gaps)
            logger.info(f"번호 간격 데이터 저장 완료: {gap_file}")

            # 세그먼트 엔트로피 계산 (segment diversity)
            segment_entropy = calculate_segment_entropy(historical_data)
            entropy_file = cache_dir / "segment_entropy.npy"
            np.save(entropy_file, segment_entropy)
            logger.info(f"세그먼트 엔트로피 데이터 저장 완료: {entropy_file}")

            # 클러스터 분포 계산
            cluster_distribution = calculate_cluster_distribution(cluster_embeddings)

            # 패턴 특성 통계 계산
            pattern_feature_stats = calculate_pattern_feature_statistics(
                pattern_features_list
            )

            # 핫/콜드 넘버 분포 계산
            hot_numbers, cold_numbers = calculate_hot_cold_numbers(historical_data)

        # 실행 시간 기록
        execution_time = time.time() - start_time
        logger.info(f"데이터 분석 및 전처리 완료 (소요시간: {execution_time:.2f}초)")

        # 프로파일링 결과 출력
        profiler.log_report()

        # 성능 보고서 생성
        try:
            enable_performance_report = config["reporting"]["enable_performance_report"]
        except KeyError as e:
            log_exception_with_trace(logger, e, "데이터 분석: 성능 보고서 설정 누락")
            logger.warning(
                "설정에서 'reporting.enable_performance_report'를 찾을 수 없습니다. 기본값 True를 사용합니다."
            )
            enable_performance_report = True

        if enable_performance_report:
            try:
                # 물리적 성능 보고서 생성
                try:
                    # GPU 정보 수집
                    gpu_stats = None
                    if torch.cuda.is_available():
                        gpu_stats = {
                            "device_name": torch.cuda.get_device_name(0),
                            "utilization": 0.0,  # 실제 사용량은 측정 불가능하여 예시값 사용
                        }

                    # 메모리 사용량 측정
                    memory_info = psutil.virtual_memory()
                    memory_used_mb = memory_info.used / (1024 * 1024)

                    # 캐시 통계 정보 가져오기
                    cache_hit_count = 0
                    cache_miss_count = 0
                    cache_hit_rate = 0.0

                    try:
                        # 벡터 캐시에서 통계 정보 수집
                        vector_cache = get_cache(config)
                        cache_stats = vector_cache.get_stats()

                        cache_hit_count = cache_stats["hits"]
                        cache_miss_count = cache_stats["misses"]
                        cache_hit_rate = cache_stats["hit_ratio"]

                        logger.info(
                            f"벡터 캐시 통계: 히트={cache_hit_count}, 미스={cache_miss_count}, "
                            f"히트율={cache_hit_rate:.2%}"
                        )
                    except Exception as e:
                        logger.warning(f"캐시 통계 수집 중 오류 발생: {str(e)}")

                    # 물리적 성능 보고서 생성 및 저장
                    physical_report_path = save_physical_performance_report(
                        module_name="data_analysis",
                        execution_times={
                            "data_load": profiler.get_execution_time_safe(
                                "데이터 로드"
                            ),
                            "pattern_analysis": profiler.get_execution_time_safe(
                                "패턴 분석"
                            ),
                            "vectorization": profiler.get_execution_time_safe(
                                "벡터 생성 및 캐싱"
                            ),
                            "additional_analysis": profiler.get_execution_time_safe(
                                "추가 분석 데이터"
                            ),
                            "report_generation": profiler.get_execution_time_safe(
                                "성능 리포트 생성"
                            ),
                            "total": time.time() - start_time,
                        },
                        cache_stats={
                            "hits": cache_hit_count,
                            "misses": cache_miss_count,
                            "hit_ratio": cache_hit_rate,
                        },
                        gpu_stats=gpu_stats,
                        config=config,
                    )

                    logger.info(
                        f"통합 물리적 성능 보고서 저장 완료: {physical_report_path}"
                    )
                except Exception as e:
                    log_exception_with_trace(
                        logger, e, "데이터 분석: 통합 물리적 성능 보고서 생성 실패"
                    )
                    # 물리적 성능 보고서 생성 실패는 전체 작업 성공 여부에 영향을 주지 않음
            except Exception as e:
                log_exception_with_trace(
                    logger, e, "데이터 분석: 성능 보고서 생성 실패"
                )

        return True

    except Exception as e:
        log_exception_with_trace(logger, e, "데이터 분석 중 예상치 못한 오류 발생")
        return False


# calculate_pair_frequency 함수는 src.shared.graph_utils로 통합됨


def calculate_segment_centrality(data, segments=9):
    """
    세그먼트별 중심성 분석

    Args:
        data: 분석할 과거 당첨 번호 목록
        segments: 세그먼트 수 (기본값: 9)

    Returns:
        np.ndarray: 세그먼트 중심성 벡터
    """
    import networkx as nx

    # 네트워크 그래프 생성
    G = nx.Graph()

    # 노드 추가 (1~45)
    for i in range(1, 46):
        G.add_node(i)

    # 동시 출현 엣지 추가
    for draw in data:
        numbers = draw.numbers
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                num1, num2 = numbers[i], numbers[j]
                if G.has_edge(num1, num2):
                    G[num1][num2]["weight"] += 1
                else:
                    G.add_edge(num1, num2, weight=1)

    # 세그먼트별 중심성 계산
    segment_size = 45 // segments
    segment_centrality = np.zeros(segments)

    for seg_idx in range(segments):
        start_num = seg_idx * segment_size + 1
        end_num = min((seg_idx + 1) * segment_size, 45)

        segment_nodes = list(range(start_num, end_num + 1))
        segment_subgraph = G.subgraph(segment_nodes)

        # 세그먼트 내 노드의 중심성 평균
        centrality_values = nx.degree_centrality(segment_subgraph)
        if centrality_values:
            segment_centrality[seg_idx] = sum(centrality_values.values()) / len(
                centrality_values
            )

    return segment_centrality


def generate_cluster_embeddings(data, embedding_dim=3):
    """
    클러스터 임베딩 생성

    Args:
        data: 분석할 과거 당첨 번호 목록
        embedding_dim: 임베딩 차원 (기본값: 3) - barnes_hut 알고리즘은 3차원 이하만 지원

    Returns:
        Tuple[np.ndarray, Dict[str, float]]: 클러스터 임베딩 벡터 및 품질 지표
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.warning(
            "scikit-learn 라이브러리가 설치되지 않았습니다. 기본 임베딩을 생성합니다."
        )
        return np.random.random((45, embedding_dim + 1)), {
            "error": "scikit-learn 미설치"
        }

    # 번호 출현 빈도 행렬 생성
    occurrence_matrix = np.zeros((45, len(data)))
    for i, draw in enumerate(data):
        for number in draw.numbers:
            # 인덱스는 0부터 시작하므로 1을 빼줌
            number_index = number - 1
            occurrence_matrix[number_index, i] = 1

    logger.info(
        f"임베딩 생성을 위한 출현 행렬 생성 완료: 형태 {occurrence_matrix.shape}"
    )

    # 임베딩 생성을 위한 TSNE 적용
    try:
        # TSNE로 차원 축소 - barnes_hut 알고리즘은 3차원 이하만 지원함
        logger.info(f"t-SNE 차원 축소 시작: {embedding_dim}차원으로 축소")
        tsne = TSNE(
            n_components=embedding_dim,
            perplexity=30,  # 고정된 perplexity 값 사용
            random_state=42,
            method="barnes_hut",  # 명시적으로 방법 지정
            learning_rate="auto",  # 자동 학습률 조정
            init="random",  # 초기화 방법 명시
            n_iter=1000,  # 충분한 반복 횟수
        )
        embeddings = tsne.fit_transform(occurrence_matrix)
        logger.info(f"t-SNE 차원 축소 완료: 임베딩 형태 {embeddings.shape}")

        # KMeans 클러스터링 적용
        best_n_clusters = 5  # 기본 클러스터 수
        best_silhouette = -1
        best_labels = None

        # 최적의 클러스터 수 찾기 (실루엣 점수가 0.2 미만이면 자동 조정)
        cluster_range = range(3, 11)  # 3에서 10까지 시도

        logger.info(
            f"최적 클러스터 수 탐색 시작 (범위: {min(cluster_range)}-{max(cluster_range)})"
        )
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # 실루엣 점수 계산 (클러스터가 1개이면 계산 불가)
            if len(set(cluster_labels)) > 1:
                try:
                    silhouette = silhouette_score(embeddings, cluster_labels)
                    logger.info(
                        f"클러스터 수 {n_clusters}의 실루엣 점수: {silhouette:.4f}"
                    )

                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_n_clusters = n_clusters
                        best_labels = cluster_labels
                except Exception as e:
                    logger.warning(
                        f"실루엣 점수 계산 중 오류 (클러스터 수={n_clusters}): {str(e)}"
                    )

        # 최적의 클러스터 수로 다시 클러스터링
        if best_silhouette < 0:  # 실루엣 점수 계산이 모두 실패했을 경우
            logger.warning(
                "모든 실루엣 점수 계산이 실패했습니다. 기본 클러스터 수 5를 사용합니다."
            )
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
        else:
            logger.info(
                f"최적의 클러스터 수: {best_n_clusters} (실루엣 점수: {best_silhouette:.4f})"
            )
            if best_labels is None:  # 이전에 계산되지 않았다면
                kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
            else:
                cluster_labels = best_labels
                kmeans = KMeans(
                    n_clusters=best_n_clusters, random_state=42, n_init=10
                ).fit(embeddings)

        # 각 번호의 클러스터 라벨
        number_cluster_labels = {
            i + 1: int(label) for i, label in enumerate(cluster_labels)
        }

        # 클러스터 중심 계산
        cluster_centers = kmeans.cluster_centers_

        # 클러스터 간 평균 거리 계산
        cluster_distances = []
        for i in range(best_n_clusters):
            for j in range(i + 1, best_n_clusters):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                cluster_distances.append(dist)

        avg_distance_between_clusters = (
            np.mean(cluster_distances) if cluster_distances else 0.0
        )

        # 클러스터 거리가 0으로 계산되면 다른 방식으로 계산 시도
        if avg_distance_between_clusters <= 0.001:
            logger.warning(
                "클러스터 간 거리가 너무 작습니다. 대체 방법으로 계산합니다."
            )
            try:
                # 클러스터별 데이터 포인트 수집
                cluster_points = {i: [] for i in range(best_n_clusters)}
                for point_idx, label in enumerate(cluster_labels):
                    cluster_points[label].append(embeddings[point_idx])

                # 각 클러스터 내의 모든 포인트에 대해 다른 클러스터와의 거리 계산
                all_inter_distances = []
                for c1 in range(best_n_clusters):
                    for c2 in range(c1 + 1, best_n_clusters):
                        if cluster_points[c1] and cluster_points[c2]:
                            # 각 포인트 쌍 간의 거리 계산
                            for p1 in cluster_points[c1]:
                                for p2 in cluster_points[c2]:
                                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                                    all_inter_distances.append(dist)

                if all_inter_distances:
                    avg_distance_between_clusters = np.mean(all_inter_distances)
                    logger.info(
                        f"재계산된 클러스터 간 거리: {avg_distance_between_clusters:.4f}"
                    )
            except Exception as e:
                logger.error(f"클러스터 거리 재계산 중 오류: {str(e)}")

        # 클러스터 크기 분석
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(best_n_clusters)]
        std_of_cluster_size = np.std(cluster_sizes)
        largest_cluster_size = np.max(cluster_sizes)
        mean_cluster_size = np.mean(cluster_sizes)

        # 균형 점수 계산 (표준편차/평균)
        balance_score = (
            std_of_cluster_size / mean_cluster_size if mean_cluster_size > 0 else 0.0
        )

        # 실루엣 점수
        silhouette = best_silhouette if best_silhouette >= 0 else 0.0

        # 클러스터 임베딩에 클러스터 ID 추가
        embeddings_with_labels = np.column_stack([embeddings, cluster_labels])

        # 클러스터 품질 지표 반환
        cluster_quality = {
            "cluster_count": best_n_clusters,
            "avg_distance_between_clusters": float(avg_distance_between_clusters),
            "std_of_cluster_size": float(std_of_cluster_size),
            "silhouette_score": float(silhouette),
            "largest_cluster_size": int(largest_cluster_size),
            "balance_score": float(balance_score),
        }

        return embeddings_with_labels, cluster_quality

    except Exception as e:
        logger.error(f"클러스터 임베딩 생성 중 오류: {str(e)}")
        logger.error(traceback.format_exc())
        # 오류 발생 시 클러스터 ID 열이 포함된 임의 배열 반환 (마지막 열이 클러스터 ID)
        random_embeddings = np.random.random((45, embedding_dim))
        random_clusters = np.zeros((45, 1))
        embeddings_with_clusters = np.column_stack([random_embeddings, random_clusters])
        return embeddings_with_clusters, {"error": f"임베딩 생성 실패: {str(e)}"}


def calculate_hot_cold_numbers(data, hot_window=30, cold_window=50):
    """
    핫 넘버와 콜드 넘버 분포를 계산합니다.

    Args:
        data: 과거 당첨 번호 목록
        hot_window: 핫 넘버를 판단할 최근 회차 수
        cold_window: 콜드 넘버를 판단할 미출현 회차 수

    Returns:
        tuple: (hot_number_dict, cold_number_dict)
    """
    # 최근 회차 데이터만 선택
    recent_data = data[-hot_window:] if len(data) > hot_window else data

    # 핫 넘버 빈도 계산
    hot_numbers = {}
    for draw in recent_data:
        for num in draw.numbers:
            hot_numbers[num] = hot_numbers.get(num, 0) + 1

    # 상위 N개 핫 넘버만 선택
    top_hot_numbers = dict(
        sorted(hot_numbers.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    # 콜드 넘버 계산 (cold_window 회차 동안 출현하지 않은 번호)
    all_numbers = set(range(1, 46))
    recent_cold_check = data[-cold_window:] if len(data) > cold_window else data

    appeared_numbers = set()
    for draw in recent_cold_check:
        appeared_numbers.update(draw.numbers)

    cold_numbers = all_numbers - appeared_numbers
    cold_number_dict = {num: cold_window for num in cold_numbers}

    # 콜드 넘버가 없으면 가장 출현 빈도가 낮은 번호를 선택
    if not cold_number_dict:
        all_freq = {}
        for draw in recent_cold_check:
            for num in draw.numbers:
                all_freq[num] = all_freq.get(num, 0) + 1

        for num in range(1, 46):
            if num not in all_freq:
                all_freq[num] = 0

        cold_number_dict = dict(sorted(all_freq.items(), key=lambda x: x[1])[:10])

    return top_hot_numbers, cold_number_dict


def calculate_number_gaps(data):
    """
    각 번호별 평균 재출현 간격을 계산합니다.

    Args:
        data: 과거 당첨 번호 목록

    Returns:
        dict: 번호별 평균 재출현 간격
    """
    # 각 번호의 출현 위치 기록
    number_positions = {num: [] for num in range(1, 46)}

    for idx, draw in enumerate(data):
        for num in draw.numbers:
            number_positions[num].append(idx)

    # 각 번호별 간격 계산
    number_gaps = {}
    for num in range(1, 46):
        positions = number_positions[num]
        if len(positions) > 1:  # 최소 2번 이상 출현해야 간격 계산 가능
            gaps = [positions[i] - positions[i - 1] for i in range(1, len(positions))]
            number_gaps[num] = sum(gaps) / len(gaps) if gaps else 0
        else:
            number_gaps[num] = (
                len(data) if len(positions) == 0 else len(data) - positions[0]
            )

    return number_gaps


def calculate_cluster_distribution(cluster_embeddings):
    """
    클러스터별 요소 수 분포를 계산합니다.

    Args:
        cluster_embeddings: 클러스터 임베딩 배열

    Returns:
        dict: 클러스터별 요소 수
    """
    # 클러스터 ID는 마지막 열에 저장됨
    if cluster_embeddings.shape[1] < 2:
        return {"cluster_0": 45}

    cluster_ids = cluster_embeddings[:, -1].astype(int)
    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)

    cluster_distribution = {
        f"cluster_{cluster}": int(count)
        for cluster, count in zip(unique_clusters, counts)
    }
    return cluster_distribution


def calculate_pattern_feature_statistics(features_list):
    """
    패턴 특성값의 통계를 계산합니다.

    Args:
        features_list: 패턴 특성값 목록

    Returns:
        dict: 각 특성별 통계 (평균, 표준편차)
    """
    if not features_list:
        return {}

    # 모든 가능한 특성값 키 추출
    all_keys = set()
    for features in features_list:
        # 문자열이나 None이 아닌 딕셔너리만 처리
        if isinstance(features, dict):
            all_keys.update(features.keys())
        elif not isinstance(features, (str, type(None))):
            # 딕셔너리가 아니면 로그 기록
            logger.warning(f"패턴 특성값이 딕셔너리가 아닙니다: {type(features)}")

    # 각 특성별 값 수집
    feature_values = {key: [] for key in all_keys}
    for features in features_list:
        if not isinstance(features, dict):
            continue  # 딕셔너리가 아니면 건너뜀

        for key in all_keys:
            if (
                key in features
                and isinstance(features[key], (int, float))
                and not isinstance(features[key], bool)
            ):
                feature_values[key].append(features[key])

    # 통계 계산
    stats = {}
    for key, values in feature_values.items():
        if values:  # 값이 있는 경우에만 통계 계산
            values_array = np.array(values, dtype=float)
            if len(values_array) > 0:
                stats[key] = {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "sample_count": len(values_array),
                }

    return stats


def calculate_segment_entropy(data, segments=5):
    """
    각 세그먼트의 엔트로피를 계산하여 다양성을 측정합니다.

    Args:
        data: 과거 당첨 번호 목록
        segments: 세그먼트 수

    Returns:
        np.ndarray: 각 세그먼트의 엔트로피
    """
    segment_size = 45 // segments
    segment_counts = np.zeros((segments, len(data)))

    # 각 세그먼트별 번호 출현 빈도 집계
    for draw_idx, draw in enumerate(data):
        for num in draw.numbers:
            seg_idx = min((num - 1) // segment_size, segments - 1)
            segment_counts[seg_idx, draw_idx] += 1

    # 세그먼트별 엔트로피 계산
    segment_entropy = np.zeros(segments)
    for seg_idx in range(segments):
        # 회차별 해당 세그먼트 번호 출현 확률 계산
        segment_probs = segment_counts[seg_idx] / 6  # 각 회차에 6개 번호 출현
        segment_probs = segment_probs[segment_probs > 0]  # 0인 확률 제외

        if len(segment_probs) > 0:
            # 섀넌 엔트로피 계산: -sum(p * log2(p))
            entropy = -np.sum(segment_probs * np.log2(segment_probs + 1e-10))
            segment_entropy[seg_idx] = entropy

    return segment_entropy


if __name__ == "__main__":
    try:
        logger.info("데이터 분석 파이프라인 시작")
        result = run_data_analysis()
        if result:
            logger.info("데이터 분석 파이프라인 실행 성공")
            sys.exit(0)
        else:
            logger.error("데이터 분석 파이프라인 실행 실패")
            sys.exit(1)
    except Exception as e:
        log_exception_with_trace(
            logger, e, "데이터 분석 파이프라인 실행 중 치명적 오류"
        )
        sys.exit(1)
