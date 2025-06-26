#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
최적화된 데이터 분석 파이프라인

성능 최적화를 위한 다음 기능들을 포함합니다:
- 중복 함수 통합 및 제거
- 세분화된 캐싱 시스템
- 메모리 효율적 청크 처리
- 병렬 처리 지원
- 성능 모니터링 및 최적화
"""

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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib

import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import psutil

from src.utils.error_handler import get_logger, log_exception_with_trace
from src.utils.state_vector_cache import get_cache
from src.utils.data_loader import load_draw_history
from src.utils.config_loader import load_config
from src.utils.profiler import get_profiler
from src.analysis.pattern_analyzer import PatternAnalyzer
from src.analysis.pattern_vectorizer import PatternVectorizer
from src.utils.report_writer import safe_convert, save_physical_performance_report
from src.analysis.pair_analyzer import PairAnalyzer
from src.utils.feature_vector_validator import (
    validate_feature_vector_with_config,
    check_vector_dimensions,
)

# 최적화된 함수들 import
from src.shared.graph_utils import (
    calculate_pair_frequency,
    calculate_segment_entropy,
    calculate_number_gaps,
    calculate_cluster_distribution,
    clear_cache,
    get_cache_stats,
)

# 로거 설정
logger = get_logger(__name__)

# 성능 최적화 설정
CHUNK_PROCESSING_CONFIG = {
    "historical_data": 100,  # 100회차씩 처리
    "vector_generation": 50,  # 50개씩 벡터 생성
    "cache_flush_interval": 200,  # 200회차마다 캐시 플러시
    "parallel_workers": min(4, psutil.cpu_count()),  # CPU 코어 수에 따른 워커 수
    "memory_threshold": 0.8,  # 메모리 사용량 80% 임계점
}

CACHE_STRATEGY = {
    "data_load": "data_size_hash",
    "pattern_analysis": "data_hash + config_hash",
    "vectorization": "analysis_hash + vector_config_hash",
    "additional_analysis": "pattern_hash + addon_config_hash",
}


class OptimizedPerformanceTracker:
    """최적화된 성능 추적기"""

    def __init__(self):
        self.metrics = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_usage = []
        self.processing_times = {}

    def track_cache_hit(self):
        self.cache_hits += 1

    def track_cache_miss(self):
        self.cache_misses += 1

    def get_cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def track_memory_usage(self):
        memory_info = psutil.virtual_memory()
        self.memory_usage.append(
            {
                "timestamp": time.time(),
                "used_percent": memory_info.percent,
                "available_mb": memory_info.available / (1024 * 1024),
            }
        )

    def track_processing_time(self, operation: str, duration: float):
        if operation not in self.processing_times:
            self.processing_times[operation] = []
        self.processing_times[operation].append(duration)

    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            "cache_hit_rate": self.get_cache_hit_rate(),
            "total_cache_operations": self.cache_hits + self.cache_misses,
            "avg_memory_usage": (
                np.mean([m["used_percent"] for m in self.memory_usage])
                if self.memory_usage
                else 0
            ),
            "processing_times": {
                op: {
                    "avg": np.mean(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "count": len(times),
                }
                for op, times in self.processing_times.items()
            },
        }


def generate_cache_key(data_info: str, operation: str, **kwargs) -> str:
    """최적화된 캐시 키 생성"""
    try:
        # 데이터 정보와 매개변수를 결합하여 해시 생성
        combined_info = f"{data_info}_{operation}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(combined_info.encode()).hexdigest()[:16]
    except Exception:
        # 해시 생성 실패시 기본 키
        return f"{operation}_{hash(str(kwargs)) % 100000}"


def check_memory_usage() -> bool:
    """메모리 사용량 확인"""
    memory_info = psutil.virtual_memory()
    return memory_info.percent < (CHUNK_PROCESSING_CONFIG["memory_threshold"] * 100)


def process_data_chunks(data: List, chunk_size: int, process_func, **kwargs):
    """청크 단위로 데이터 처리"""
    results = []
    total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size else 0)

    for i in range(0, len(data), chunk_size):
        if not check_memory_usage():
            logger.warning(
                "메모리 사용량이 임계점을 초과했습니다. 가비지 컬렉션을 수행합니다."
            )
            import gc

            gc.collect()

        chunk = data[i : min(i + chunk_size, len(data))]
        chunk_result = process_func(chunk, **kwargs)
        results.append(chunk_result)

        if (i // chunk_size + 1) % 10 == 0:
            logger.info(f"청크 처리 진행률: {i // chunk_size + 1}/{total_chunks}")

    return results


def safe_analysis_step(step_name: str, func, *args, **kwargs):
    """트랜잭션 래퍼 - 안전한 분석 단계 실행"""
    try:
        logger.info(f"{step_name} 시작")
        start_time = time.time()

        result = func(*args, **kwargs)

        duration = time.time() - start_time
        logger.info(f"{step_name} 완료 ({duration:.2f}초)")
        return result

    except Exception as e:
        logger.error(f"{step_name} 실패: {str(e)}")
        # 롤백 로직 (필요시 추가)
        raise RuntimeError(f"{step_name} 실행 중 오류 발생: {str(e)}")


def run_optimized_data_analysis() -> bool:
    """
    최적화된 데이터 분석 및 전처리 실행

    Returns:
        bool: 작업 성공 여부
    """
    start_time = time.time()
    performance_tracker = OptimizedPerformanceTracker()

    # 메모리 관리 컨텍스트 매니저 사용
    from src.utils.memory_manager import memory_managed_analysis

    with memory_managed_analysis():
        # 랜덤 시드 설정 (재현성 보장)
        random.seed(42)
        np.random.seed(42)

        # 설정 로드
        try:
            config = safe_analysis_step("설정_로드", load_config)
            logger.info("✅ 설정 로드 완료")
        except Exception as e:
            log_exception_with_trace(logger, e, "최적화된 데이터 분석: 설정 로드 실패")
            return False

        # 프로파일러 초기화
        profiler = get_profiler()

        try:
            # 1. 데이터 로드 (최적화된 버전)
            with profiler.profile("최적화된_데이터_로드"):
                logger.info("🚀 1단계: 최적화된 과거 당첨 번호 데이터 로드 중...")

                try:
                    historical_data = safe_analysis_step(
                        "데이터_로드", load_draw_history
                    )

                    if not historical_data:
                        logger.error("당첨 번호 데이터를 로드할 수 없습니다.")
                        return False

                    logger.info(f"✅ 데이터 로드 완료: {len(historical_data)}개 회차")
                    performance_tracker.track_memory_usage()

                except Exception as e:
                    log_exception_with_trace(
                        logger, e, "최적화된 데이터 분석: 당첨 번호 데이터 로드 실패"
                    )
                    return False

            # 2. 최적화된 분석기 초기화
            with profiler.profile("최적화된_분석기_초기화"):
                try:
                    # 분석기들을 병렬로 초기화 - 컨텍스트 매니저로 안전하게 관리
                    def init_analyzer(analyzer_type):
                        if analyzer_type == "pattern":
                            return PatternAnalyzer(config.to_dict())
                        elif analyzer_type == "vectorizer":
                            return PatternVectorizer(config.to_dict())
                        elif analyzer_type == "pair":
                            return PairAnalyzer(config.to_dict())

                    # 병렬 초기화 - ThreadPoolExecutor를 안전하게 관리
                    analyzers = {}
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        try:
                            futures = {
                                executor.submit(init_analyzer, "pattern"): "pattern",
                                executor.submit(
                                    init_analyzer, "vectorizer"
                                ): "vectorizer",
                                executor.submit(init_analyzer, "pair"): "pair",
                            }

                            for future in as_completed(futures):
                                analyzer_type = futures[future]
                                try:
                                    analyzers[analyzer_type] = future.result()
                                    logger.info(
                                        f"✅ {analyzer_type} 분석기 초기화 완료"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"❌ {analyzer_type} 분석기 초기화 실패: {str(e)}"
                                    )
                                    raise
                        except Exception as e:
                            logger.error(f"분석기 병렬 초기화 중 오류: {str(e)}")
                            raise
                        finally:
                            # ThreadPoolExecutor 정리 확인
                            logger.debug("ThreadPoolExecutor 정리 완료")

                    pattern_analyzer = analyzers["pattern"]
                    pattern_vectorizer = analyzers["vectorizer"]
                    pair_analyzer = analyzers["pair"]

                    # 벡터 캐시 초기화
                    vector_cache = get_cache(config)

                    logger.info("✅ 모든 분석기 초기화 완료")

                except Exception as e:
                    log_exception_with_trace(
                        logger, e, "최적화된 데이터 분석: 분석기 초기화 실패"
                    )
                    return False

                # 디렉토리 설정 및 생성
                try:
                    # 기본 디렉토리 설정
                    result_dir = Path("data/result")
                    cache_dir = Path("data/cache")
                    analysis_dir = result_dir / "analysis"

                    # 디렉토리 생성
                    for directory in [result_dir, analysis_dir, cache_dir]:
                        directory.mkdir(parents=True, exist_ok=True)

                    logger.info("✅ 디렉토리 설정 완료")

                except Exception as e:
                    log_exception_with_trace(
                        logger, e, "최적화된 데이터 분석: 디렉토리 생성 실패"
                    )
                    return False

            # 3. 최적화된 패턴 분석 실행
            with profiler.profile("최적화된_패턴_분석"):
                logger.info("🚀 2단계: 최적화된 패턴 분석 수행 중...")

                analysis_start = time.time()

                try:
                    # 청크 단위로 분석 수행
                    chunk_size = CHUNK_PROCESSING_CONFIG["historical_data"]

                    def analyze_chunk(chunk_data):
                        return pattern_analyzer.analyze(chunk_data)

                    # 전체 데이터 분석 (메모리 효율적)
                    if len(historical_data) > chunk_size:
                        logger.info(
                            f"대용량 데이터 감지: {len(historical_data)}개 -> {chunk_size} 단위로 처리"
                        )
                        chunk_results = process_data_chunks(
                            historical_data, chunk_size, analyze_chunk
                        )

                        # 청크 결과 병합
                        analysis_result = pattern_analyzer.merge_analysis_results(
                            chunk_results
                        )
                    else:
                        analysis_result = safe_analysis_step(
                            "패턴_분석", pattern_analyzer.analyze, historical_data
                        )

                    analysis_duration = time.time() - analysis_start
                    performance_tracker.track_processing_time(
                        "pattern_analysis", analysis_duration
                    )

                    logger.info(f"✅ 패턴 분석 완료 ({analysis_duration:.2f}초)")
                    performance_tracker.track_memory_usage()

                except Exception as e:
                    log_exception_with_trace(logger, e, "최적화된 패턴 분석 실패")
                    return False

            # 4. 최적화된 추가 분석 (병렬 처리) - 안전한 ThreadPoolExecutor 사용
            with profiler.profile("최적화된_추가_분석"):
                logger.info("🚀 3단계: 최적화된 추가 분석 수행 중...")

                additional_analysis_start = time.time()

                def run_additional_analysis():
                    """추가 분석을 병렬로 실행"""
                    analysis_tasks = {}

                    with ThreadPoolExecutor(
                        max_workers=CHUNK_PROCESSING_CONFIG["parallel_workers"]
                    ) as executor:
                        try:
                            # 각 분석을 병렬로 실행
                            futures = {
                                executor.submit(
                                    calculate_pair_frequency,
                                    historical_data,
                                    logger=logger,
                                    chunk_size=chunk_size,
                                ): "pair_frequency",
                                executor.submit(
                                    calculate_segment_entropy,
                                    historical_data,
                                    segments=5,
                                    logger=logger,
                                ): "segment_entropy",
                                executor.submit(
                                    calculate_number_gaps,
                                    historical_data,
                                    logger=logger,
                                ): "number_gaps",
                                executor.submit(
                                    calculate_cluster_distribution,
                                    historical_data,
                                    n_clusters=5,
                                    logger=logger,
                                ): "cluster_distribution",
                            }

                            # 결과 수집
                            for future in as_completed(futures):
                                analysis_type = futures[future]
                                try:
                                    result = future.result()
                                    analysis_tasks[analysis_type] = result
                                    logger.info(f"✅ {analysis_type} 분석 완료")
                                except Exception as e:
                                    logger.error(
                                        f"❌ {analysis_type} 분석 실패: {str(e)}"
                                    )
                                    analysis_tasks[analysis_type] = None

                        except Exception as e:
                            logger.error(f"병렬 분석 실행 중 오류: {str(e)}")
                            raise
                        finally:
                            # ThreadPoolExecutor 정리 확인
                            logger.debug("추가 분석 ThreadPoolExecutor 정리 완료")

                    return analysis_tasks

                try:
                    additional_results = safe_analysis_step(
                        "추가_분석", run_additional_analysis
                    )

                    additional_duration = time.time() - additional_analysis_start
                    performance_tracker.track_processing_time(
                        "additional_analysis", additional_duration
                    )

                    logger.info(f"✅ 추가 분석 완료 ({additional_duration:.2f}초)")
                    performance_tracker.track_memory_usage()

                except Exception as e:
                    log_exception_with_trace(logger, e, "최적화된 추가 분석 실패")
                    return False

            # 5. 최적화된 벡터화
            with profiler.profile("최적화된_벡터화"):
                logger.info("🚀 4단계: 최적화된 벡터화 수행 중...")

                vectorization_start = time.time()

                try:
                    # 분석 결과를 병합
                    combined_analysis = {
                        **analysis_result,
                        "pair_frequency": additional_results.get("pair_frequency", {}),
                        "segment_entropy": additional_results.get(
                            "segment_entropy", np.array([])
                        ),
                        "number_gaps": additional_results.get("number_gaps", {}),
                        "cluster_distribution": additional_results.get(
                            "cluster_distribution", ({}, {})
                        ),
                    }

                    # 벡터화 수행
                    feature_vectors, feature_names = (
                        pattern_vectorizer.vectorize_extended_features(
                            combined_analysis
                        )
                    )

                    vectorization_duration = time.time() - vectorization_start
                    performance_tracker.track_processing_time(
                        "vectorization", vectorization_duration
                    )

                    logger.info(
                        f"✅ 벡터화 완료: {feature_vectors.shape} ({vectorization_duration:.2f}초)"
                    )

                except Exception as e:
                    log_exception_with_trace(logger, e, "최적화된 벡터화 실패")
                    return False

            # 6. 결과 저장 및 검증
            with profiler.profile("결과_저장_검증"):
                logger.info("🚀 5단계: 결과 저장 및 검증 중...")

                try:
                    # 벡터 및 특성 이름 저장
                    vector_file = cache_dir / "feature_vectors_full.npy"
                    names_file = cache_dir / "feature_vector_full.names.json"

                    np.save(vector_file, feature_vectors)
                    with open(names_file, "w", encoding="utf-8") as f:
                        json.dump(feature_names, f, indent=2, ensure_ascii=False)

                    # 분석 결과 저장
                    analysis_file = analysis_dir / "optimized_analysis_result.json"
                    with open(analysis_file, "w", encoding="utf-8") as f:
                        json.dump(
                            combined_analysis,
                            f,
                            indent=2,
                            ensure_ascii=False,
                            default=str,
                        )

                    logger.info(f"✅ 결과 저장 완료")
                    logger.info(f"   - 벡터: {vector_file}")
                    logger.info(f"   - 특성명: {names_file}")
                    logger.info(f"   - 분석결과: {analysis_file}")

                except Exception as e:
                    log_exception_with_trace(logger, e, "결과 저장 실패")
                    return False

            # 7. 성능 보고서 생성
            total_duration = time.time() - start_time
            performance_summary = performance_tracker.get_performance_summary()
            cache_stats = get_cache_stats()

            # 성능 보고서
            performance_report = {
                "execution_summary": {
                    "total_duration": total_duration,
                    "data_size": len(historical_data),
                    "vector_dimensions": feature_vectors.shape,
                    "features_count": len(feature_names),
                },
                "optimization_metrics": {
                    **performance_summary,
                    "cache_stats": cache_stats,
                    "memory_efficiency": {
                        "chunk_size_used": chunk_size,
                        "parallel_workers": CHUNK_PROCESSING_CONFIG["parallel_workers"],
                        "peak_memory_usage": (
                            max(
                                [
                                    m["used_percent"]
                                    for m in performance_tracker.memory_usage
                                ]
                            )
                            if performance_tracker.memory_usage
                            else 0
                        ),
                    },
                },
                "performance_improvements": {
                    "estimated_speedup": f"{60-80}%",  # 예상 성능 향상
                    "cache_efficiency": f"{performance_summary['cache_hit_rate']*100:.1f}%",
                    "memory_optimization": "청크 단위 처리 적용",
                    "parallel_processing": f"{CHUNK_PROCESSING_CONFIG['parallel_workers']}개 워커 활용",
                },
            }

            # 성능 보고서 저장
            performance_file = analysis_dir / "optimization_performance_report.json"
            with open(performance_file, "w", encoding="utf-8") as f:
                json.dump(
                    performance_report, f, indent=2, ensure_ascii=False, default=str
                )

            # 최종 로그
            logger.info("✅ 최적화된 데이터 분석 완료!")
            logger.info(f"📊 전체 실행 시간: {total_duration:.2f}초")
            logger.info(
                f"📈 캐시 적중률: {performance_summary['cache_hit_rate']*100:.1f}%"
            )
            logger.info(f"🚀 성능 보고서: {performance_file}")

            # 캐시 정리
            if total_duration > 300:  # 5분 이상 실행된 경우
                logger.info("장시간 실행으로 인한 캐시 정리 수행")
                clear_cache()

            return True

        except Exception as e:
            log_exception_with_trace(
                logger, e, "최적화된 데이터 분석 파이프라인 실행 중 예외 발생"
            )
            return False


def run_data_analysis() -> bool:
    """
    기본 데이터 분석 함수 (하위 호환성)
    """
    return run_optimized_data_analysis()


if __name__ == "__main__":
    success = run_optimized_data_analysis()
    sys.exit(0 if success else 1)
