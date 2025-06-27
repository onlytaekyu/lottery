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
import psutil

from src.utils.error_handler_refactored import (
    get_logger,
    log_exception_with_trace,
    StrictErrorHandler,
    strict_error_handler,
    validate_and_fail_fast,
)
from src.utils.state_vector_cache import get_cache
from src.utils.data_loader import load_draw_history, LotteryJSONEncoder
from src.shared.types import LotteryNumber
from src.utils.unified_config import load_config
from src.utils.unified_performance import get_profiler

# 최적화 시스템 import
from src.utils.process_pool_manager import get_process_pool_manager
from src.utils.hybrid_optimizer import get_hybrid_optimizer, optimize
from src.utils.memory_manager import get_memory_manager

from src.analysis.pattern_analyzer import PatternAnalyzer
from src.analysis.pattern_vectorizer import PatternVectorizer
from src.utils.unified_report import safe_convert, save_physical_performance_report
from src.analysis.pair_analyzer import PairAnalyzer

# from src.utils.feature_vector_validator import (
#     validate_feature_vector_with_config,
#     check_vector_dimensions,
# )

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
        try:
            process_pool_config = optimization_config.get(
                "process_pool",
                {
                    "max_workers": min(4, psutil.cpu_count()),
                    "chunk_size": 100,
                    "timeout": 300,
                },
            )
            process_pool_manager = get_process_pool_manager(process_pool_config)
            logger.info("ProcessPool 관리자 초기화 완료")
        except Exception as e:
            logger.warning(f"ProcessPool 관리자 초기화 실패: {e}")
            process_pool_manager = None

        # 메모리 관리자 초기화
        try:
            from src.utils.memory_manager import MemoryConfig

            memory_config_dict = optimization_config.get("memory", {})
            memory_config = MemoryConfig(
                max_memory_usage=memory_config_dict.get("max_memory_usage", 0.85),
                cache_size=memory_config_dict.get("cache_size", 256 * 1024 * 1024),
                use_memory_pooling=memory_config_dict.get("use_memory_pooling", True),
                auto_cleanup_interval=memory_config_dict.get(
                    "auto_cleanup_interval", 60.0
                ),
            )
            memory_manager = get_memory_manager(memory_config)
            logger.info("메모리 관리자 초기화 완료")
        except Exception as e:
            logger.warning(f"메모리 관리자 초기화 실패: {e}")
            memory_manager = None

        # 하이브리드 최적화 시스템 초기화
        try:
            hybrid_config = optimization_config.get(
                "hybrid",
                {
                    "auto_optimization": True,
                    "memory_threshold": 0.8,
                    "cpu_threshold": 75.0,
                },
            )
            hybrid_optimizer = get_hybrid_optimizer(hybrid_config)
            logger.info("하이브리드 최적화 시스템 초기화 완료")
        except Exception as e:
            logger.warning(f"하이브리드 최적화 시스템 초기화 실패: {e}")
            hybrid_optimizer = None

        # 최적화 시스템 상태 로깅
        initialized_systems = []
        if process_pool_manager is not None:
            initialized_systems.append("ProcessPool")
        if memory_manager is not None:
            initialized_systems.append("MemoryManager")
        if hybrid_optimizer is not None:
            initialized_systems.append("HybridOptimizer")

        logger.info(f"초기화된 최적화 시스템: {', '.join(initialized_systems)}")

    except Exception as e:
        logger.error(f"최적화 시스템 초기화 실패: {e}")
        # 최적화 시스템 없이도 동작할 수 있도록 None으로 설정
        process_pool_manager = None
        hybrid_optimizer = None
        memory_manager = None


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
    최적화된 데이터 분석 파이프라인 실행

    Returns:
        bool: 실행 성공 여부
    """
    start_time = time.time()

    try:
        logger.info("🚀 최적화된 데이터 분석 파이프라인 시작")

        # 설정 로드
        config = load_config()

        # 최적화 시스템 초기화
        initialize_optimization_systems(config)

        # 1단계: 데이터 로드 및 고급 검증
        logger.info("📊 1단계: 데이터 로드 및 고급 검증")
        historical_data = load_draw_history()

        if not historical_data:
            logger.error("로또 데이터를 로드할 수 없습니다.")
            return False

        logger.info(f"로또 데이터 로드 완료: {len(historical_data)}회차")

        # 🔍 고급 데이터 검증 시스템 적용
        from src.pipeline.data_validation import DataValidator

        validator = DataValidator(config)
        validation_result = validator.validate_lottery_data(historical_data)

        if not validation_result.is_valid:
            logger.error(f"데이터 검증 실패: {len(validation_result.errors)}개 오류")
            for error in validation_result.errors[:5]:  # 처음 5개 오류만 표시
                logger.error(f"  - {error}")
            return False

        logger.info(
            f"✅ 데이터 검증 완료 (품질 점수: {validation_result.quality_score})"
        )

        # 경고 사항 로깅
        if validation_result.warnings:
            logger.warning(f"데이터 품질 경고 {len(validation_result.warnings)}개:")
            for warning in validation_result.warnings[:3]:
                logger.warning(f"  - {warning}")

        # 이상치 정보 로깅
        if validation_result.anomalies:
            logger.info(f"감지된 이상치: {len(validation_result.anomalies)}개")

        # 품질 보고서 생성 및 저장
        quality_report = validator.generate_quality_report(historical_data)
        validator.save_quality_report(quality_report)

        # 2단계: 분석기 초기화
        logger.info("🔧 2단계: 분석기 초기화")

        def init_analyzer(analyzer_type: str):
            """분석기 초기화 헬퍼 함수"""
            try:
                # ConfigProxy를 딕셔너리로 안전하게 변환
                if hasattr(config, "_config"):
                    config_dict = config._config
                elif hasattr(config, "to_dict"):
                    config_dict = config.to_dict()
                elif isinstance(config, dict):
                    config_dict = config
                else:
                    # ConfigProxy 객체의 속성을 딕셔너리로 변환
                    config_dict = {}
                    if hasattr(config, "__dict__"):
                        for key, value in config.__dict__.items():
                            if not key.startswith("_"):
                                config_dict[key] = value

                    # 기본 설정 추가
                    if not config_dict:
                        config_dict = {
                            "analysis": {},
                            "paths": {
                                "cache_dir": "data/cache",
                                "result_dir": "data/result",
                            },
                            "vectorizer": {"use_cache": True, "normalize_output": True},
                            "filtering": {
                                "remove_low_variance_features": True,
                                "variance_threshold": 0.01,
                            },
                            "caching": {
                                "enable_feature_cache": True,
                                "max_cache_size": 10000,
                            },
                        }

                if analyzer_type == "pattern":
                    return PatternAnalyzer(config_dict)
                elif analyzer_type == "distribution":
                    from src.analysis.distribution_analyzer import DistributionAnalyzer

                    return DistributionAnalyzer(config_dict)
                elif analyzer_type == "roi":
                    from src.analysis.roi_analyzer import ROIAnalyzer

                    return ROIAnalyzer(config_dict)
                elif analyzer_type == "pair":
                    return PairAnalyzer(config_dict)
                elif analyzer_type == "vectorizer":
                    return PatternVectorizer(config_dict)
                else:
                    raise ValueError(f"알 수 없는 분석기 타입: {analyzer_type}")
            except Exception as e:
                logger.error(f"{analyzer_type} 분석기 초기화 실패: {e}")
                # 기본 설정으로 재시도
                try:
                    # 기본 설정 딕셔너리 생성
                    default_config = {
                        "analysis": {},
                        "paths": {
                            "cache_dir": "data/cache",
                            "result_dir": "data/result",
                        },
                        "vectorizer": {"use_cache": True, "normalize_output": True},
                        "filtering": {
                            "remove_low_variance_features": True,
                            "variance_threshold": 0.01,
                        },
                        "caching": {
                            "enable_feature_cache": True,
                            "max_cache_size": 10000,
                        },
                    }

                    if analyzer_type == "pattern":
                        return PatternAnalyzer(default_config)
                    elif analyzer_type == "distribution":
                        from src.analysis.distribution_analyzer import (
                            DistributionAnalyzer,
                        )

                        return DistributionAnalyzer(default_config)
                    elif analyzer_type == "roi":
                        from src.analysis.roi_analyzer import ROIAnalyzer

                        return ROIAnalyzer(default_config)
                    elif analyzer_type == "pair":
                        return PairAnalyzer(default_config)
                    elif analyzer_type == "vectorizer":
                        return PatternVectorizer(default_config)
                except Exception as e2:
                    logger.error(f"{analyzer_type} 분석기 기본 초기화도 실패: {e2}")
                    return None

        # 분석기들 초기화
        pattern_analyzer = init_analyzer("pattern")
        distribution_analyzer = init_analyzer("distribution")
        roi_analyzer = init_analyzer("roi")
        pair_analyzer = init_analyzer("pair")
        vectorizer = init_analyzer("vectorizer")

        # 초기화 실패 체크
        analyzers = {
            "pattern": pattern_analyzer,
            "distribution": distribution_analyzer,
            "roi": roi_analyzer,
            "pair": pair_analyzer,
            "vectorizer": vectorizer,
        }

        failed_analyzers = [
            name for name, analyzer in analyzers.items() if analyzer is None
        ]
        if failed_analyzers:
            logger.error(f"다음 분석기 초기화 실패: {failed_analyzers}")
            return False

        logger.info("모든 분석기 초기화 완료")

        # 3단계: 병렬 분석 실행
        logger.info("⚡ 3단계: 병렬 분석 실행")
        analysis_results = {}

        # 메모리 관리 스코프 내에서 실행
        if memory_manager:
            with memory_manager.allocation_scope():
                analysis_results = run_parallel_analysis(
                    historical_data, analyzers, config
                )
        else:
            analysis_results = run_parallel_analysis(historical_data, analyzers, config)

        if not analysis_results:
            logger.error("분석 실행 실패")
            return False

        logger.info(f"분석 완료: {len(analysis_results)}개 결과")

        # 4단계: 고급 특성 추출 및 벡터 생성
        logger.info("🔢 4단계: 고급 특성 추출 및 벡터 생성")

        try:
            # 통합 분석 결과 생성
            unified_analysis = merge_analysis_results(analysis_results)

            # 🧠 고급 특성 추출 엔진 적용
            from src.analysis.feature_extractor import FeatureExtractor

            feature_extractor = FeatureExtractor(config)
            extraction_result = feature_extractor.extract_all_features(unified_analysis)

            logger.info(
                f"특성 추출 완료: {len(extraction_result.feature_names)}개 특성"
            )
            logger.info(f"특성 그룹: {list(extraction_result.feature_groups.keys())}")

            # 특성 최적화 (선택적)
            if extraction_result.feature_matrix.size > 0:
                optimized_features, optimized_names, importance_scores = (
                    feature_extractor.optimize_feature_selection(
                        extraction_result.feature_matrix,
                        extraction_result.feature_names,
                    )
                )
                logger.info(f"특성 최적화 완료: {len(optimized_names)}개 특성 선택")
            else:
                optimized_features = extraction_result.feature_matrix
                optimized_names = extraction_result.feature_names

            # 벡터화 실행 (기존 벡터화 시스템과 병행)
            feature_vector = None
            feature_names = []

            if vectorizer is not None:
                try:
                    feature_vector, feature_names = (
                        vectorizer.vectorize_extended_features(unified_analysis)
                    )
                    if feature_vector is not None and len(feature_vector) > 0:
                        logger.info(f"기존 벡터화 시스템: {len(feature_vector)}차원")
                except Exception as e:
                    logger.warning(f"기존 벡터화 시스템 오류: {e}")
                    feature_vector = None
                    feature_names = []

            # 두 벡터 시스템 결합
            if feature_vector is not None and len(feature_vector) > 0:
                if optimized_features.size > 0:
                    # 차원이 다를 수 있으므로 안전하게 결합
                    combined_vector = np.concatenate(
                        [feature_vector.flatten(), optimized_features.flatten()]
                    )
                    combined_names = feature_names + optimized_names
                    logger.info(f"벡터 시스템 결합 완료: {len(combined_vector)}차원")
                else:
                    combined_vector = feature_vector
                    combined_names = feature_names
            else:
                # 기존 벡터화 실패시 새로운 특성 추출 결과 사용
                combined_vector = (
                    optimized_features.flatten()
                    if optimized_features.size > 0
                    else np.array([])
                )
                combined_names = optimized_names

            if len(combined_vector) == 0:
                logger.error("특성 벡터 생성 실패")
                return False

            logger.info(f"최종 특성 벡터 생성 완료: {len(combined_vector)}차원")

            # 벡터 품질 검증
            if not validate_feature_vector(combined_vector, combined_names, config):
                logger.error("특성 벡터 품질 검증 실패")
                return False

        except Exception as e:
            logger.error(f"특성 벡터 생성 중 오류: {e}")
            log_exception_with_trace(e)
            return False

        # 5단계: 결과 저장
        logger.info("💾 5단계: 결과 저장")

        try:
            # 분석 결과 저장
            save_analysis_results(unified_analysis, config)

            # 특성 벡터 저장
            if vectorizer is not None:
                vector_path = vectorizer.save_vector_to_file(combined_vector)
                names_path = vectorizer.save_names_to_file(combined_names)
            else:
                # vectorizer가 None인 경우 직접 저장
                from pathlib import Path
                import json

                cache_dir = Path("data/cache")
                cache_dir.mkdir(parents=True, exist_ok=True)

                vector_path = cache_dir / "feature_vector_full.npy"
                names_path = cache_dir / "feature_vector_full.names.json"

                np.save(vector_path, combined_vector)
                with open(names_path, "w", encoding="utf-8") as f:
                    json.dump(combined_names, f, ensure_ascii=False, indent=2)

                vector_path = str(vector_path)
                names_path = str(names_path)

            logger.info(f"결과 저장 완료:")
            logger.info(f"  - 벡터 파일: {vector_path}")
            logger.info(f"  - 이름 파일: {names_path}")

        except Exception as e:
            logger.error(f"결과 저장 중 오류: {e}")
            return False

        # 6단계: 성능 보고서 생성
        logger.info("📈 6단계: 성능 보고서 생성")

        try:
            execution_time = time.time() - start_time

            # 성능 보고서 데이터 수집
            performance_data = {
                "execution_time": execution_time,
                "data_size": len(historical_data),
                "vector_dimensions": len(combined_vector),
                "analysis_results_count": len(analysis_results),
                "memory_usage": get_memory_usage(),
                "data_quality_score": validation_result.quality_score,
                "feature_extraction_quality": extraction_result.quality_metrics,
                "optimization_used": {
                    "process_pool": process_pool_manager is not None,
                    "memory_manager": memory_manager is not None,
                    "hybrid_optimizer": hybrid_optimizer is not None,
                },
            }

            # 성능 보고서 저장
            save_performance_report(performance_data, "optimized_data_analysis")

            logger.info(f"성능 보고서 생성 완료 (실행시간: {execution_time:.2f}초)")

        except Exception as e:
            logger.warning(f"성능 보고서 생성 중 오류: {e}")

        # 최적화 시스템 정리
        cleanup_optimization_systems()

        logger.info(
            f"✅ 최적화된 데이터 분석 파이프라인 완료 (총 {time.time() - start_time:.2f}초)"
        )
        return True

    except Exception as e:
        logger.error(f"❌ 데이터 분석 파이프라인 실행 중 오류: {e}")
        return False


def validate_lottery_data(historical_data: List[LotteryNumber]) -> bool:
    """로또 데이터 검증"""
    try:
        if not historical_data:
            logger.error("빈 데이터셋")
            return False

        for i, draw in enumerate(historical_data):
            # 번호 개수 검증
            if len(draw.numbers) != 6:
                logger.error(
                    f"회차 {draw.draw_no}: 번호 개수 오류 ({len(draw.numbers)}개)"
                )
                return False

            # 번호 범위 검증
            for number in draw.numbers:
                if not (1 <= number <= 45):
                    logger.error(f"회차 {draw.draw_no}: 번호 범위 오류 ({number})")
                    return False

            # 중복 번호 검증
            if len(set(draw.numbers)) != 6:
                logger.error(f"회차 {draw.draw_no}: 중복 번호 존재")
                return False

        logger.info(f"데이터 검증 완료: {len(historical_data)}회차 모두 정상")
        return True

    except Exception as e:
        logger.error(f"데이터 검증 중 오류: {e}")
        return False


def run_parallel_analysis(
    historical_data: List[LotteryNumber],
    analyzers: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """병렬 분석 실행"""
    analysis_results = {}

    try:
        logger.info("병렬 분석 시작")

        # 분석 작업 정의
        analysis_tasks = [
            ("pattern", analyzers["pattern"]),
            ("distribution", analyzers["distribution"]),
            ("roi", analyzers["roi"]),
            ("pair", analyzers["pair"]),
        ]

        # 프로세스 풀이 있으면 병렬 실행, 없으면 순차 실행
        if process_pool_manager and len(analysis_tasks) > 1:
            logger.info("프로세스 풀을 사용한 병렬 분석")

            with ThreadPoolExecutor(max_workers=len(analysis_tasks)) as executor:
                future_to_name = {}

                for name, analyzer in analysis_tasks:
                    if analyzer:
                        future = executor.submit(
                            safe_analysis_execution, name, analyzer, historical_data
                        )
                        future_to_name[future] = name

                # 결과 수집
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        result = future.result(timeout=300)  # 5분 타임아웃
                        if result:
                            analysis_results[name] = result
                            logger.info(f"{name} 분석 완료")
                        else:
                            logger.warning(f"{name} 분석 결과 없음")
                    except Exception as e:
                        logger.error(f"{name} 분석 실패: {e}")
        else:
            logger.info("순차 분석 실행")

            for name, analyzer in analysis_tasks:
                if analyzer:
                    try:
                        result = safe_analysis_execution(
                            name, analyzer, historical_data
                        )
                        if result:
                            analysis_results[name] = result
                            logger.info(f"{name} 분석 완료")
                    except Exception as e:
                        logger.error(f"{name} 분석 실패: {e}")

        logger.info(f"병렬 분석 완료: {len(analysis_results)}개 결과")
        return analysis_results

    except Exception as e:
        logger.error(f"병렬 분석 실행 중 오류: {e}")
        return {}


def safe_analysis_execution(
    name: str, analyzer: Any, historical_data: List[LotteryNumber]
) -> Optional[Dict[str, Any]]:
    """안전한 분석 실행"""
    try:
        logger.info(f"{name} 분석 시작")
        result = analyzer.analyze(historical_data)
        logger.info(f"{name} 분석 성공")
        return result
    except Exception as e:
        logger.error(f"{name} 분석 중 오류: {e}")
        return None


def merge_analysis_results(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """분석 결과들을 통합"""
    try:
        unified_analysis = {
            "metadata": {
                "analysis_count": len(analysis_results),
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
            }
        }

        # 각 분석 결과를 통합
        for analysis_type, result in analysis_results.items():
            if result:
                unified_analysis[analysis_type] = result

        logger.info(f"분석 결과 통합 완료: {len(unified_analysis)}개 항목")
        return unified_analysis

    except Exception as e:
        logger.error(f"분석 결과 통합 중 오류: {e}")
        return {}


def validate_feature_vector(
    feature_vector: np.ndarray, feature_names: List[str], config: Dict[str, Any]
) -> bool:
    """특성 벡터 품질 검증"""
    try:
        # 기본 검증
        if feature_vector is None or len(feature_vector) == 0:
            logger.error("빈 특성 벡터")
            return False

        if feature_names is None or len(feature_names) == 0:
            logger.error("빈 특성 이름 리스트")
            return False

        # 차원 일치 검증
        if len(feature_vector) != len(feature_names):
            logger.error(
                f"벡터 차원 불일치: 벡터={len(feature_vector)}, 이름={len(feature_names)}"
            )
            return False

        # 최소 차원 검증
        try:
            min_dimensions = config["vector"]["min_required_dimension"]
            if len(feature_vector) < min_dimensions:
                logger.error(
                    f"벡터 차원 부족: {len(feature_vector)} < {min_dimensions}"
                )
                return False
        except (KeyError, TypeError):
            logger.warning("최소 차원 설정을 찾을 수 없음")

        # NaN/Inf 검증
        if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
            logger.error("벡터에 NaN 또는 Inf 값 포함")
            return False

        logger.info(f"특성 벡터 검증 완료: {len(feature_vector)}차원")
        return True

    except Exception as e:
        logger.error(f"특성 벡터 검증 중 오류: {e}")
        return False


def save_analysis_results(
    unified_analysis: Dict[str, Any], config: Dict[str, Any]
) -> None:
    """분석 결과 저장"""
    try:
        # 결과 디렉토리 생성
        result_dir = Path("data/result/analysis")
        result_dir.mkdir(parents=True, exist_ok=True)

        # JSON 파일로 저장
        result_file = result_dir / "optimized_analysis_result.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(
                unified_analysis,
                f,
                ensure_ascii=False,
                indent=2,
                cls=LotteryJSONEncoder,
            )

        logger.info(f"분석 결과 저장 완료: {result_file}")

    except Exception as e:
        logger.error(f"분석 결과 저장 중 오류: {e}")
        raise


def save_performance_report(performance_data: Dict[str, Any], module_name: str) -> None:
    """성능 보고서 저장"""
    try:
        # 보고서 디렉토리 생성
        report_dir = Path("data/result/performance_reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        # 보고서 파일 저장
        report_file = report_dir / f"{module_name}_performance_report.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(performance_data, f, ensure_ascii=False, indent=2)

        logger.info(f"성능 보고서 저장 완료: {report_file}")

    except Exception as e:
        logger.error(f"성능 보고서 저장 중 오류: {e}")


def get_memory_usage() -> Dict[str, float]:
    """현재 메모리 사용량 조회"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # MB
            "vms_mb": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent(),
        }
    except Exception:
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0}


def cleanup_optimization_systems() -> None:
    """최적화 시스템 정리"""
    global process_pool_manager, hybrid_optimizer, memory_manager

    try:
        if process_pool_manager:
            process_pool_manager.shutdown()
            logger.info("프로세스 풀 정리 완료")

        if memory_manager:
            memory_manager.cleanup()
            logger.info("메모리 관리자 정리 완료")

        if hybrid_optimizer:
            hybrid_optimizer.cleanup()
            logger.info("하이브리드 최적화 시스템 정리 완료")

    except Exception as e:
        logger.warning(f"최적화 시스템 정리 중 오류: {e}")
    finally:
        # 전역 변수 초기화
        process_pool_manager = None
        hybrid_optimizer = None
        memory_manager = None


def run_data_analysis() -> bool:
    """
    기본 데이터 분석 함수 (하위 호환성)
    """
    return run_optimized_data_analysis()


def run_fully_optimized_analysis():
    """완전 최적화된 데이터 분석 파이프라인"""

    # 🚀 전역 최적화 시스템 초기화
    from src.utils.memory_manager import get_memory_manager, MemoryConfig
    from src.utils.cuda_optimizers import CudaConfig
    from src.utils.unified_performance import get_profiler

    logger.info("🎉 완전 최적화된 데이터 분석 파이프라인 시작")

    # 전역 최적화 시스템 초기화
    memory_config = MemoryConfig(
        max_memory_usage=0.85,
        use_memory_pooling=True,
        pool_size=32,
        auto_cleanup_interval=60.0,
    )
    memory_manager = get_memory_manager(memory_config)

    cuda_config = CudaConfig(
        use_amp=True,
        batch_size=128,
        use_cudnn=True,
    )

    # 프로파일러 초기화
    profiler = get_profiler()

    # 🧠 전역 메모리 관리 컨텍스트
    with memory_manager.allocation_scope():
        # 📈 전체 성능 모니터링
        with profiler.profile("완전_최적화_분석"):
            # 기본 최적화된 분석 실행
            return run_optimized_data_analysis()


def create_optimized_analyzer(
    memory_manager, cuda_optimizer, process_pool_manager, hybrid_optimizer
):
    """최적화된 분석기 생성"""
    from src.analysis.unified_analyzer import UnifiedAnalyzer

    config = load_config()
    analyzer = UnifiedAnalyzer(config)

    # 최적화 시스템 주입
    if hasattr(analyzer, "set_optimizers"):
        analyzer.set_optimizers(
            memory_manager=memory_manager,
            cuda_optimizer=cuda_optimizer,
            process_pool_manager=process_pool_manager,
            hybrid_optimizer=hybrid_optimizer,
        )

    return analyzer


def create_optimized_vectorizer(memory_manager, cuda_optimizer):
    """최적화된 벡터라이저 생성"""
    from src.analysis.pattern_vectorizer import PatternVectorizer

    config = load_config()
    vectorizer = PatternVectorizer(config)

    # 최적화 시스템 주입
    if hasattr(vectorizer, "set_optimizers"):
        vectorizer.set_optimizers(
            memory_manager=memory_manager, cuda_optimizer=cuda_optimizer
        )

    return vectorizer


def load_draw_history_optimized(memory_manager=None):
    """최적화된 데이터 로드"""
    from src.utils.data_loader import load_draw_history

    if memory_manager:
        with memory_manager.allocation_scope():
            return load_draw_history()
    else:
        return load_draw_history()


def save_results_optimized(analysis_result, vectors, memory_manager=None):
    """최적화된 결과 저장"""
    try:
        if memory_manager:
            with memory_manager.allocation_scope():
                # 메모리 효율적 저장
                _save_with_memory_optimization(analysis_result, vectors)
        else:
            _save_standard(analysis_result, vectors)

        logger.info("✅ 최적화된 결과 저장 완료")

    except Exception as e:
        logger.error(f"결과 저장 중 오류: {e}")


def _save_with_memory_optimization(analysis_result, vectors):
    """메모리 최적화된 저장"""
    import json
    import numpy as np
    from pathlib import Path

    # 분석 결과 저장
    analysis_path = Path("data/result/analysis/lottery_data_analysis.json")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)

    # 벡터 저장
    if vectors is not None:
        vector_path = Path("data/cache/feature_vectors_full.npy")
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vector_path, vectors)


def _save_standard(analysis_result, vectors):
    """표준 저장"""
    _save_with_memory_optimization(analysis_result, vectors)


def cleanup_optimized_resources(*optimizers):
    """최적화 리소스 정리"""
    for optimizer in optimizers:
        if optimizer and hasattr(optimizer, "cleanup"):
            try:
                optimizer.cleanup()
            except Exception as e:
                logger.warning(f"리소스 정리 중 오류: {e}")

    logger.info("✅ 최적화 리소스 정리 완료")


def benchmark_optimization_performance():
    """최적화 성능 벤치마크"""
    logger.info("최적화 성능 벤치마크 시작")

    try:
        # 기본 분석 실행
        start_time = time.time()
        result = run_optimized_data_analysis()
        duration = time.time() - start_time

        logger.info(f"벤치마크 완료: {duration:.2f}초, 성공: {result}")
        return {"duration": duration, "success": result}

    except Exception as e:
        logger.error(f"벤치마크 실행 중 오류: {e}")
        return {"duration": 0, "success": False, "error": str(e)}


if __name__ == "__main__":
    success = run_optimized_data_analysis()
    sys.exit(0 if success else 1)
