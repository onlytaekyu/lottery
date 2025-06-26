#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
최적화된 데이터 분석 파이프라인

성능 최적화를 위한 다음 기능들을 포함합니다:
- 중복 함수 통합 및 제거
- 세분화된 캐싱 시스템
- 메모리 효율적 청크 처리
- 병렬 처리 지원 (ProcessPool 통합)
- 하이브리드 최적화 시스템
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

from src.utils.error_handler import (
    get_logger,
    log_exception_with_trace,
    StrictErrorHandler,
    strict_error_handler,
    validate_and_fail_fast,
)
from src.utils.state_vector_cache import get_cache
from src.utils.data_loader import load_draw_history, LotteryJSONEncoder
from src.utils.config_loader import load_config
from src.utils.profiler import get_profiler

# 최적화 시스템 import
from src.utils.process_pool_manager import get_process_pool_manager
from src.utils.hybrid_optimizer import get_hybrid_optimizer, optimize
from src.utils.memory_manager import get_memory_manager

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

# 전역 엄격한 에러 핸들러
strict_handler = StrictErrorHandler()

# 전역 최적화 시스템들
process_pool_manager = None
hybrid_optimizer = None
memory_manager = None


def initialize_optimization_systems(config: Dict[str, Any]):
    """최적화 시스템들 초기화"""
    global process_pool_manager, hybrid_optimizer, memory_manager

    try:
        # 최적화 설정 로드
        optimization_config = config.get("optimization", {})

        # ProcessPool 관리자 초기화
        process_pool_config = optimization_config.get("process_pool", {})
        process_pool_manager = get_process_pool_manager(process_pool_config)
        logger.info("ProcessPool 관리자 초기화 완료")

        # 메모리 관리자 초기화
        memory_config = optimization_config.get("memory_pool", {})
        memory_manager = get_memory_manager(memory_config)
        logger.info("메모리 관리자 초기화 완료")

        # 하이브리드 최적화 시스템 초기화
        hybrid_config = optimization_config.get("hybrid", {})
        hybrid_optimizer = get_hybrid_optimizer(hybrid_config)
        logger.info("하이브리드 최적화 시스템 초기화 완료")

    except Exception as e:
        logger.error(f"최적화 시스템 초기화 실패: {e}")
        # 최적화 시스템 없이도 동작할 수 있도록 None으로 설정
        process_pool_manager = None
        hybrid_optimizer = None
        memory_manager = None


class OptimizedPerformanceTracker:
    """최적화된 성능 추적기"""

    def __init__(self):
        self.metrics = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_usage = []
        self.processing_times = {}
        self.optimization_stats = {}

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

    def track_optimization_result(self, operation: str, strategy: str, speedup: float):
        """최적화 결과 추적"""
        if operation not in self.optimization_stats:
            self.optimization_stats[operation] = {}

        if strategy not in self.optimization_stats[operation]:
            self.optimization_stats[operation][strategy] = {
                "count": 0,
                "total_speedup": 0.0,
                "avg_speedup": 0.0,
            }

        stats = self.optimization_stats[operation][strategy]
        stats["count"] += 1
        stats["total_speedup"] += speedup
        stats["avg_speedup"] = stats["total_speedup"] / stats["count"]

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
            "optimization_stats": self.optimization_stats,
        }


@optimize(
    task_info={
        "function_type": "analysis",
        "parallelizable": True,
        "gpu_compatible": False,
    }
)
def optimized_pattern_analysis(
    data_chunk: List, analyzer: PatternAnalyzer
) -> Dict[str, Any]:
    """최적화된 패턴 분석"""
    try:
        # 메모리 관리 스코프 내에서 실행
        if memory_manager:
            with memory_manager.allocation_scope():
                return analyzer.analyze(data_chunk)
        else:
            return analyzer.analyze(data_chunk)
    except Exception as e:
        logger.error(f"패턴 분석 실패: {e}")
        return {}


@optimize(
    task_info={
        "function_type": "vectorize",
        "parallelizable": True,
        "gpu_compatible": False,
    }
)
def optimized_vectorization(
    patterns: List, vectorizer: PatternVectorizer
) -> Tuple[np.ndarray, List[str]]:
    """최적화된 벡터화"""
    try:
        if memory_manager:
            with memory_manager.allocation_scope():
                return vectorizer.vectorize(patterns)
        else:
            return vectorizer.vectorize(patterns)
    except Exception as e:
        logger.error(f"벡터화 실패: {e}")
        return np.array([]), []


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


def process_data_chunks_optimized(data: List, chunk_size: int, process_func, **kwargs):
    """최적화된 청크 단위 데이터 처리"""
    global process_pool_manager

    # ProcessPool이 사용 가능하고 데이터가 충분히 큰 경우 병렬 처리
    if process_pool_manager and len(data) > 200:
        try:
            chunks = process_pool_manager.chunk_and_split(data, chunk_size)
            results = process_pool_manager.parallel_analyze(
                chunks, process_func, **kwargs
            )
            return process_pool_manager.merge_results(results)
        except Exception as e:
            logger.warning(f"병렬 처리 실패, 순차 처리로 전환: {e}")

    # 순차 처리 (기존 방식)
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


@strict_error_handler("최적화된 데이터 분석 파이프라인", exit_on_error=True)
def run_optimized_data_analysis() -> bool:
    """
    최적화된 데이터 분석 및 전처리 실행

    Returns:
        bool: 작업 성공 여부 (실패 시 시스템 종료)
    """
    start_time = time.time()
    performance_tracker = OptimizedPerformanceTracker()

    # 메모리 관리 컨텍스트 매니저 사용
    from src.utils.memory_manager import memory_managed_analysis
    from src.utils.profiler import get_profiler

    # 프로파일러 초기화
    profiler = get_profiler()

    with memory_managed_analysis():
        with profiler.profile("전체_파이프라인"):
            logger.info("🚀 최적화된 데이터 분석 및 전처리 시작")

            # 1. 설정 로드 - 실패 시 즉시 종료
            with profiler.profile("설정_로드"):
                logger.info("설정_로드 시작")

                config = load_config()
                validate_and_fail_fast(
                    config is not None, "설정 파일 로드 실패 - 시스템을 종료합니다"
                )

                logger.info(f"설정_로드 완료 ({time.time() - start_time:.2f}초)")
                logger.info("✅ 설정 로드 완료")

                # 1.5. 최적화 시스템 초기화
                initialize_optimization_systems(config)
                logger.info("✅ 최적화 시스템 초기화 완료")

            # 2. 데이터 로드 및 검증 - 실패 시 즉시 종료
            with profiler.profile("데이터_로드"):
                logger.info("🚀 1단계: 최적화된 과거 당첨 번호 데이터 로드 중...")
                logger.info("데이터_로드 시작")

                historical_data = load_draw_history(
                    validate_data=True
                )  # 엄격한 검증 활성화
                validate_and_fail_fast(
                    historical_data and len(historical_data) > 0,
                    f"데이터 로드 실패 또는 빈 데이터: {len(historical_data) if historical_data else 0}개",
                    historical_data,
                )

                logger.info(f"데이터_로드 완료 ({time.time() - start_time:.2f}초)")
                logger.info(f"✅ 데이터 로드 완료: {len(historical_data)}개 회차")

            # 분석기 초기화 - 실패 시 즉시 종료
            performance_tracker.track_memory_usage()

            def init_analyzer(analyzer_type: str):
                """분석기 초기화 래퍼"""
                if analyzer_type == "pair":
                    from src.analysis.pair_analyzer import PairAnalyzer

                    analyzer = PairAnalyzer()
                    validate_and_fail_fast(
                        analyzer is not None, f"{analyzer_type} 분석기 초기화 실패"
                    )
                    return analyzer
                elif analyzer_type == "vectorizer":
                    from src.analysis.pattern_vectorizer import PatternVectorizer

                    analyzer = PatternVectorizer()
                    validate_and_fail_fast(
                        analyzer is not None, f"{analyzer_type} 분석기 초기화 실패"
                    )
                    return analyzer
                elif analyzer_type == "pattern":
                    from src.analysis.pattern_analyzer import PatternAnalyzer

                    analyzer = PatternAnalyzer()
                    validate_and_fail_fast(
                        analyzer is not None, f"{analyzer_type} 분석기 초기화 실패"
                    )
                    return analyzer
                else:
                    strict_handler.handle_critical_error(
                        ValueError(f"알 수 없는 분석기 타입: {analyzer_type}"),
                        "분석기 초기화 실패",
                    )
                    return None

            # 분석기들 초기화
            from src.analysis.pair_analyzer import PairAnalyzer
            from src.analysis.pattern_vectorizer import PatternVectorizer
            from src.analysis.pattern_analyzer import PatternAnalyzer

            pair_analyzer: PairAnalyzer = init_analyzer("pair")
            logger.info("✅ pair 분석기 초기화 완료")

            pattern_vectorizer: PatternVectorizer = init_analyzer("vectorizer")
            logger.info("✅ vectorizer 분석기 초기화 완료")

            pattern_analyzer: PatternAnalyzer = init_analyzer("pattern")
            logger.info("✅ pattern 분석기 초기화 완료")

            logger.info("✅ 모든 분석기 초기화 완료")

            # 디렉토리 설정 및 생성 - 실패 시 즉시 종료
            result_dir = Path("data/result")
            cache_dir = Path("data/cache")
            analysis_dir = result_dir / "analysis"

            # 디렉토리 생성
            for directory in [result_dir, analysis_dir, cache_dir]:
                directory.mkdir(parents=True, exist_ok=True)
                validate_and_fail_fast(
                    directory.exists(), f"디렉토리 생성 실패: {directory}"
                )

            logger.info("✅ 디렉토리 설정 완료")

            # 3. 최적화된 패턴 분석 실행 - 실패 시 즉시 종료
            with profiler.profile("최적화된_패턴_분석"):
                logger.info("🚀 2단계: 최적화된 패턴 분석 수행 중...")

                analysis_start = time.time()

                # 청크 단위로 분석 수행
                chunk_size = CHUNK_PROCESSING_CONFIG["historical_data"]

                def analyze_chunk(chunk_data):
                    result = pattern_analyzer.analyze(chunk_data)
                    validate_and_fail_fast(
                        result is not None,
                        f"패턴 분석 실패: 청크 크기 {len(chunk_data)}",
                    )
                    return result

                # 전체 데이터 분석 (메모리 효율적)
                if len(historical_data) > chunk_size:
                    logger.info(
                        f"대용량 데이터 감지: {len(historical_data)}개 -> {chunk_size} 단위로 처리"
                    )
                    chunk_results = process_data_chunks_optimized(
                        historical_data, chunk_size, analyze_chunk
                    )
                    validate_and_fail_fast(
                        chunk_results is not None and len(chunk_results) > 0,
                        "청크 분석 결과가 비어 있음",
                    )

                    # 청크 결과 병합
                    analysis_result = pattern_analyzer.merge_analysis_results(
                        chunk_results
                    )
                    validate_and_fail_fast(
                        analysis_result is not None, "청크 결과 병합 실패"
                    )
                else:
                    analysis_result = analyze_chunk(historical_data)

                analysis_duration = time.time() - analysis_start
                performance_tracker.track_processing_time(
                    "pattern_analysis", analysis_duration
                )

                logger.info(f"✅ 패턴 분석 완료 ({analysis_duration:.2f}초)")
                performance_tracker.track_memory_usage()

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
                            result = future.result()
                            validate_and_fail_fast(
                                result is not None, f"{analysis_type} 분석 실패"
                            )
                            analysis_tasks[analysis_type] = result
                            logger.info(f"✅ {analysis_type} 분석 완료")

                        # ThreadPoolExecutor 정리 확인
                        logger.debug("추가 분석 ThreadPoolExecutor 정리 완료")

                    return analysis_tasks

                additional_results = run_additional_analysis()
                validate_and_fail_fast(
                    additional_results is not None and len(additional_results) > 0,
                    "추가 분석 결과가 비어 있음",
                )

                additional_duration = time.time() - additional_analysis_start
                performance_tracker.track_processing_time(
                    "additional_analysis", additional_duration
                )

                logger.info(f"✅ 추가 분석 완료 ({additional_duration:.2f}초)")
                performance_tracker.track_memory_usage()

            # 5. 최적화된 벡터화 - 실패 시 즉시 종료
            with profiler.profile("최적화된_벡터화"):
                logger.info("🚀 4단계: 최적화된 벡터화 수행 중...")

                vectorization_start = time.time()

                # 분석 결과를 병합
                combined_analysis = dict(analysis_result)
                combined_analysis.update(
                    {
                        "pair_frequency": additional_results.get("pair_frequency", {}),
                        "segment_entropy": additional_results.get(
                            "segment_entropy", np.array([])
                        ),
                        "number_gaps": additional_results.get("number_gaps", {}),
                        "cluster_distribution": additional_results.get(
                            "cluster_distribution", ({}, {})
                        ),
                    }
                )

                validate_and_fail_fast(
                    combined_analysis is not None and len(combined_analysis) > 0,
                    "병합된 분석 결과가 비어 있음",
                )

                # 벡터화 수행
                feature_vectors, feature_names = (
                    pattern_vectorizer.vectorize_extended_features(combined_analysis)
                )

                validate_and_fail_fast(
                    feature_vectors is not None and feature_names is not None,
                    "벡터화 실패",
                )
                validate_and_fail_fast(
                    len(feature_vectors) > 0 and len(feature_names) > 0,
                    f"벡터화 결과가 비어 있음: vectors={len(feature_vectors)}, names={len(feature_names)}",
                )

                vectorization_duration = time.time() - vectorization_start
                performance_tracker.track_processing_time(
                    "vectorization", vectorization_duration
                )

                logger.info(
                    f"✅ 벡터화 완료: {feature_vectors.shape} ({vectorization_duration:.2f}초)"
                )

            # 6. 결과 저장 및 검증 - 실패 시 즉시 종료
            with profiler.profile("결과_저장_검증"):
                logger.info("🚀 5단계: 결과 저장 및 검증 중...")

                # 벡터 및 특성 이름 저장
                vector_file = cache_dir / "feature_vectors_full.npy"
                names_file = cache_dir / "feature_vector_full.names.json"

                np.save(vector_file, feature_vectors)
                validate_and_fail_fast(
                    vector_file.exists(), f"벡터 파일 저장 실패: {vector_file}"
                )

                with open(names_file, "w", encoding="utf-8") as f:
                    json.dump(feature_names, f, indent=2, ensure_ascii=False)
                validate_and_fail_fast(
                    names_file.exists(), f"특성 이름 파일 저장 실패: {names_file}"
                )

                # 분석 결과 저장
                analysis_file = analysis_dir / "optimized_analysis_result.json"
                with open(analysis_file, "w", encoding="utf-8") as f:
                    json.dump(
                        combined_analysis,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        cls=LotteryJSONEncoder,
                    )
                validate_and_fail_fast(
                    analysis_file.exists(), f"분석 결과 파일 저장 실패: {analysis_file}"
                )

                logger.info("✅ 모든 결과 저장 완료")

            # 최종 성능 리포트
            total_duration = time.time() - start_time
            performance_summary = performance_tracker.get_performance_summary()

            logger.info("✅ 최적화된 데이터 분석 완료!")
            logger.info(f"📊 전체 실행 시간: {total_duration:.2f}초")
            logger.info(f"📈 성능 요약: {performance_summary}")

            # 성능 리포트 저장
            performance_file = analysis_dir / "performance_report.json"
            with open(performance_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "total_duration": total_duration,
                        "performance_summary": performance_summary,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            return True


def run_data_analysis() -> bool:
    """
    기본 데이터 분석 함수 (하위 호환성)
    """
    return run_optimized_data_analysis()


if __name__ == "__main__":
    success = run_optimized_data_analysis()
    sys.exit(0 if success else 1)
