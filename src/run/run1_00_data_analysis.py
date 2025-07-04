#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAEBAK AI 데이터 준비 통합 파이프라인

이 스크립트는 ML 학습을 위한 완전한 데이터 준비 파이프라인을 제공합니다:
- Phase 1: 데이터 분석 (회차별 구조/통계/트렌드 특성 추출)
- Phase 2: 벡터화 (150~200차원 최적 특성 벡터 생성)
- Phase 3: Negative 샘플링 (ML 학습용 비당첨 조합 생성)

3단계를 연속적으로 실행하여 효율성을 극대화합니다.
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
import time
import gc

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 설정
os.environ["PYTHONPATH"] = str(project_root)

# 핵심 유틸리티들
from src.utils.unified_logging import get_logger
from src.utils.unified_config import get_config
from src.utils.memory_manager import get_memory_manager
from src.utils.unified_performance import performance_monitor
from src.utils.cache_paths import get_cache_dir
from src.utils.data_loader import load_draw_history

# 분석 관련 모듈들 (실제 사용되는 것만)
from src.analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer
from src.analysis.negative_sample_generator import NegativeSampleGenerator

# 파이프라인 관리자들
from src.pipeline.unified_preprocessing_pipeline import UnifiedPreprocessingPipeline

# 성능 최적화 도구
from src.utils.performance_optimizer import launch_max_performance

# 공유 타입들
from src.shared.types import LotteryNumber

logger = get_logger(__name__)


class DataPreparationPipeline:
    """데이터 준비 통합 파이프라인"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = get_config("main") if config is None else config
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()

        # 결과 저장 경로들
        self.cache_dir = get_cache_dir()
        self.result_dir = Path("data/result/analysis")
        self.performance_dir = Path("data/result/performance_reports")

        # 디렉토리 생성
        for directory in [self.cache_dir, self.result_dir, self.performance_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # 분석기들 초기화 (지연 초기화)
        self._vectorizer = None
        self._negative_generator = None

        # 실행 옵션
        self.execution_options = {
            "enable_caching": True,
            "parallel_processing": True,
            "chunk_size": 10000,
            "memory_limit_ratio": 0.8,
            "vector_dimensions": [150, 200],
            "negative_sample_ratio": 3.0,
            "max_memory_usage_mb": 1024,  # 1GB 제한
            "performance_monitoring": True,
        }

        self.preproc_manager = UnifiedPreprocessingPipeline(self.config)

        self.logger.info("데이터 준비 통합 파이프라인 초기화 완료")

    def execute_full_pipeline(
        self,
        clear_cache: bool = False,
        steps: List[str] = None,
        debug: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행

        Args:
            clear_cache: 캐시 삭제 여부
            steps: 실행할 단계 리스트 (기본값: 모든 단계)
            debug: 디버그 모드
            verbose: 상세 로깅

        Returns:
            Dict[str, Any]: 실행 결과 요약
        """
        start_time = time.time()

        if steps is None:
            steps = ["analysis", "vectorization", "negative_sampling"]

        # 로깅 레벨 설정
        if verbose:
            self.logger.setLevel("DEBUG")

        self.logger.info("=" * 80)
        self.logger.info("🚀 DAEBAK AI 데이터 준비 통합 파이프라인 시작")
        self.logger.info(f"📋 실행 단계: {', '.join(steps)}")
        self.logger.info(
            f"💾 메모리 제한: {self.execution_options['max_memory_usage_mb']}MB"
        )
        self.logger.info(
            f"🎯 벡터 차원 목표: {self.execution_options['vector_dimensions']}"
        )
        self.logger.info("=" * 80)

        # 실행 결과 추적
        pipeline_results = {
            "start_time": datetime.now().isoformat(),
            "steps_executed": [],
            "steps_failed": [],
            "performance_metrics": {},
            "output_files": {},
            "warnings": [],
        }

        try:
            # 캐시 정리 (선택적)
            if clear_cache:
                self.logger.info("🧹 캐시 정리 중...")
                self._clear_pipeline_cache()

            # 1. 데이터 분석 단계
            if "analysis" in steps:
                self.logger.info("📊 Phase 1: 데이터 분석 실행 중...")
                analysis_result = self.run_data_analysis()

                if analysis_result["success"]:
                    pipeline_results["steps_executed"].append("analysis")
                    pipeline_results["performance_metrics"]["analysis"] = (
                        analysis_result["metrics"]
                    )
                    pipeline_results["output_files"].update(
                        analysis_result["output_files"]
                    )
                    self.logger.info("✅ 데이터 분석 완료")
                else:
                    pipeline_results["steps_failed"].append("analysis")
                    self.logger.error("❌ 데이터 분석 실패")
                    if not debug:
                        return pipeline_results

            # 2. 벡터화 단계
            if "vectorization" in steps:
                self.logger.info("🔢 Phase 2: 벡터화 실행 중...")

                # 분석 결과가 필요한 경우 로드
                if "analysis" not in steps:
                    analysis_result = self._load_analysis_result()

                vectorization_result = self.run_vectorization(analysis_result)

                if vectorization_result["success"]:
                    pipeline_results["steps_executed"].append("vectorization")
                    pipeline_results["performance_metrics"]["vectorization"] = (
                        vectorization_result["metrics"]
                    )
                    pipeline_results["output_files"].update(
                        vectorization_result["output_files"]
                    )
                    self.logger.info("✅ 벡터화 완료")
                else:
                    pipeline_results["steps_failed"].append("vectorization")
                    self.logger.error("❌ 벡터화 실패")
                    if not debug:
                        return pipeline_results

            # 3. Negative 샘플링 단계
            if "negative_sampling" in steps:
                self.logger.info("🎯 Phase 3: Negative 샘플링 실행 중...")

                # 벡터화 결과가 필요한 경우 로드
                if "vectorization" not in steps:
                    vectorization_result = self._load_vectorization_result()

                negative_sampling_result = self.run_negative_sampling(
                    vectorization_result
                )

                if negative_sampling_result["success"]:
                    pipeline_results["steps_executed"].append("negative_sampling")
                    pipeline_results["performance_metrics"]["negative_sampling"] = (
                        negative_sampling_result["metrics"]
                    )
                    pipeline_results["output_files"].update(
                        negative_sampling_result["output_files"]
                    )
                    self.logger.info("✅ Negative 샘플링 완료")
                else:
                    pipeline_results["steps_failed"].append("negative_sampling")
                    self.logger.error("❌ Negative 샘플링 실패")
                    if not debug:
                        return pipeline_results

            # 4. 결과 검증 및 저장
            validation_result = self.validate_and_save_results(pipeline_results)
            pipeline_results["validation"] = validation_result

            # 5. 성능 통계 출력
            total_time = time.time() - start_time
            pipeline_results["total_execution_time"] = total_time
            pipeline_results["end_time"] = datetime.now().isoformat()

            self._print_performance_summary(pipeline_results)

            # 6. 종합 보고서 저장
            self._save_pipeline_report(pipeline_results)

            self.logger.info("=" * 80)
            self.logger.info("✅ 데이터 준비 통합 파이프라인 완료")
            self.logger.info("=" * 80)

            return pipeline_results

        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류 발생: {e}")
            pipeline_results["error"] = str(e)
            pipeline_results["end_time"] = datetime.now().isoformat()
            return pipeline_results
        finally:
            # 메모리 정리
            gc.collect()

    def run_data_analysis(self) -> Dict[str, Any]:
        """데이터 분석 단계 (기존 코드 활용)"""
        start_time = time.time()

        try:
            # 기존 최적화된 분석 파이프라인 활용
            success = run_optimized_data_analysis()

            if not success:
                return {
                    "success": False,
                    "error": "기존 분석 파이프라인 실행 실패",
                    "metrics": {},
                    "output_files": {},
                }

            # 분석 결과 파일들 확인
            output_files = self._check_analysis_output_files()

            # 성능 메트릭 수집
            execution_time = time.time() - start_time
            metrics = {
                "execution_time": execution_time,
                "memory_usage": self.memory_manager.get_memory_usage(),
                "output_files_count": len(output_files),
            }

            self.logger.info(
                f"데이터 분석 완료: {execution_time:.2f}초, {len(output_files)}개 파일 생성"
            )

            return {"success": True, "metrics": metrics, "output_files": output_files}

        except Exception as e:
            self.logger.error(f"데이터 분석 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {},
                "output_files": {},
            }

    def run_vectorization(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """벡터화 단계 (150~200차원)"""
        start_time = time.time()

        try:
            # 메모리 사용량 확인
            memory_usage = self.memory_manager.get_memory_usage()
            if memory_usage > 0.8:
                self.logger.warning(f"메모리 사용량 높음: {memory_usage:.1%}")
                # 메모리 정리
                gc.collect()

            # 벡터화기 초기화
            if self._vectorizer is None:
                self._vectorizer = EnhancedPatternVectorizer(self.config)

            # 분석 결과 로드
            analysis_data = self._load_unified_analysis()

            if not analysis_data:
                self.logger.error("분석 데이터를 로드할 수 없습니다")
                return {
                    "success": False,
                    "error": "분석 데이터를 로드할 수 없습니다",
                    "metrics": {},
                    "output_files": {},
                }

            self.logger.info(f"분석 데이터 로드 완료: {len(analysis_data)} 항목")

            # 벡터화 실행
            feature_vector = self._vectorizer.vectorize_full_analysis_enhanced(
                analysis_data
            )

            # 벡터 검증
            if feature_vector is None or len(feature_vector) == 0:
                self.logger.error("벡터화 결과가 비어있습니다")
                return {
                    "success": False,
                    "error": "벡터화 결과가 비어있습니다",
                    "metrics": {},
                    "output_files": {},
                }

            feature_names = self._vectorizer.get_feature_names()

            # 차원 검증 (150~200 범위)
            vector_dim = len(feature_vector)
            target_range = self.execution_options["vector_dimensions"]

            if not (target_range[0] <= vector_dim <= target_range[1]):
                self.logger.warning(
                    f"벡터 차원 {vector_dim}이 목표 범위 {target_range} 밖입니다"
                )

            # 벡터 품질 검증
            quality_metrics = self._validate_vector_quality(
                feature_vector, feature_names
            )

            # 결과 저장
            output_files = {}

            # 특성 벡터 저장
            vector_path = self.cache_dir / "feature_vector_full.npy"
            try:
                np.save(vector_path, feature_vector)
                output_files["feature_vector"] = str(vector_path)
                self.logger.info(f"특성 벡터 저장: {vector_path}")
            except Exception as e:
                self.logger.error(f"특성 벡터 저장 실패: {e}")

            # 특성 이름 저장
            names_path = self.cache_dir / "feature_vector_full.names.json"
            try:
                with open(names_path, "w", encoding="utf-8") as f:
                    json.dump(feature_names, f, ensure_ascii=False, indent=2)
                output_files["feature_names"] = str(names_path)
                self.logger.info(f"특성 이름 저장: {names_path}")
            except Exception as e:
                self.logger.error(f"특성 이름 저장 실패: {e}")

            # 벡터 메타데이터 저장
            metadata = {
                "vector_dimension": vector_dim,
                "feature_count": len(feature_names),
                "quality_metrics": quality_metrics,
                "generated_at": datetime.now().isoformat(),
                "config": self.execution_options,
            }

            metadata_path = self.cache_dir / "feature_vector_metadata.json"
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                output_files["metadata"] = str(metadata_path)
                self.logger.info(f"메타데이터 저장: {metadata_path}")
            except Exception as e:
                self.logger.error(f"메타데이터 저장 실패: {e}")

            # 성능 메트릭 수집
            execution_time = time.time() - start_time
            metrics = {
                "execution_time": execution_time,
                "vector_dimension": vector_dim,
                "feature_count": len(feature_names),
                "memory_usage": self.memory_manager.get_memory_usage(),
                "quality_score": quality_metrics.get("overall_score", 0.0),
            }

            self.logger.info(
                f"벡터화 완료: {vector_dim}차원, 품질점수: {quality_metrics.get('overall_score', 0.0):.3f}"
            )

            return {
                "success": True,
                "metrics": metrics,
                "output_files": output_files,
                "vector_data": {
                    "vector": feature_vector,
                    "names": feature_names,
                    "metadata": metadata,
                },
            }

        except Exception as e:
            self.logger.error(f"벡터화 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {},
                "output_files": {},
            }

    def run_negative_sampling(
        self, vectorization_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Negative 샘플링 단계"""
        start_time = time.time()

        try:
            # Negative 샘플 생성기 초기화
            if self._negative_generator is None:
                self._negative_generator = NegativeSampleGenerator(self.config)

            # 과거 데이터 로드
            historical_data = load_draw_history()
            if not historical_data:
                return {
                    "success": False,
                    "error": "과거 당첨 데이터를 로드할 수 없습니다",
                    "metrics": {},
                    "output_files": {},
                }

            # 특성 벡터 정보 가져오기
            if "vector_data" in vectorization_result:
                feature_vector = vectorization_result["vector_data"]["vector"]
                vector_dim = len(feature_vector)
            else:
                # 파일에서 로드
                vector_path = self.cache_dir / "feature_vector_full.npy"
                if vector_path.exists():
                    feature_vector = np.load(vector_path)
                    vector_dim = len(feature_vector)
                else:
                    return {
                        "success": False,
                        "error": "특성 벡터를 찾을 수 없습니다",
                        "metrics": {},
                        "output_files": {},
                    }

            # 샘플 수 계산
            positive_count = len(historical_data)
            negative_count = int(
                positive_count * self.execution_options["negative_sample_ratio"]
            )

            self.logger.info(
                f"Negative 샘플 생성: 양성 {positive_count}개 → 음성 {negative_count}개"
            )

            # Negative 샘플 생성
            negative_result = self._negative_generator.generate_samples(
                historical_data, sample_size=negative_count
            )

            if not negative_result.get("success", False):
                self.logger.error(
                    f"Negative 샘플 생성 실패: {negative_result.get('error', 'Unknown error')}"
                )
                return {
                    "success": False,
                    "error": f"Negative 샘플 생성 실패: {negative_result.get('error', 'Unknown error')}",
                    "metrics": {},
                    "output_files": {},
                }

            # 결과 파일들 정리
            output_files = {}

            # 생성된 샘플 파일들 확인
            if "raw_path" in negative_result:
                output_files["raw_samples"] = negative_result["raw_path"]
            if "vector_path" in negative_result:
                output_files["vector_samples"] = negative_result["vector_path"]
            if "report_path" in negative_result:
                output_files["performance_report"] = negative_result["report_path"]

            # 메타데이터 업데이트
            sample_count = negative_result.get("sample_count", 0)
            self.logger.info(f"생성된 Negative 샘플 수: {sample_count:,}개")

            # 메타데이터 저장
            metadata = {
                "total_negative_samples": sample_count,
                "positive_samples": positive_count,
                "negative_ratio": self.execution_options["negative_sample_ratio"],
                "vector_dimension": vector_dim,
                "generated_at": datetime.now().isoformat(),
                "generation_config": self.execution_options,
                "generation_time": negative_result.get("elapsed_time", 0),
                "memory_used_mb": negative_result.get("memory_used_mb", 0),
            }

            metadata_path = self.cache_dir / "negative_sampling_metadata.json"
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                output_files["metadata"] = str(metadata_path)
                self.logger.info(f"메타데이터 저장: {metadata_path}")
            except Exception as e:
                self.logger.error(f"메타데이터 저장 실패: {e}")

            # 성능 메트릭 수집
            execution_time = time.time() - start_time
            metrics = {
                "execution_time": execution_time,
                "total_samples": sample_count,
                "memory_usage": self.memory_manager.get_memory_usage(),
                "generation_rate": (
                    sample_count / execution_time if execution_time > 0 else 0
                ),
                "generation_time": negative_result.get("elapsed_time", 0),
                "memory_used_mb": negative_result.get("memory_used_mb", 0),
            }

            self.logger.info(
                f"Negative 샘플링 완료: {sample_count:,}개 생성 ({execution_time:.2f}초)"
            )

            return {
                "success": True,
                "metrics": metrics,
                "output_files": output_files,
                "sample_data": {
                    "total_count": sample_count,
                    "metadata": metadata,
                },
            }

        except Exception as e:
            self.logger.error(f"Negative 샘플링 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {},
                "output_files": {},
            }

    def validate_and_save_results(
        self, pipeline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """결과 검증 및 저장"""
        validation_result = {
            "success": True,
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
        }

        try:
            # 1. 필수 출력 파일 존재 확인
            required_files = [
                "feature_vector_full.npy",
                "feature_vector_full.names.json",
            ]

            for filename in required_files:
                file_path = self.cache_dir / filename
                if file_path.exists():
                    validation_result["checks_passed"].append(f"파일 존재: {filename}")
                else:
                    validation_result["checks_failed"].append(f"파일 누락: {filename}")
                    validation_result["success"] = False

            # 2. 벡터 차원 검증
            vector_path = self.cache_dir / "feature_vector_full.npy"
            if vector_path.exists():
                vector = np.load(vector_path)
                vector_dim = len(vector)
                target_range = self.execution_options["vector_dimensions"]

                if target_range[0] <= vector_dim <= target_range[1]:
                    validation_result["checks_passed"].append(
                        f"벡터 차원 적합: {vector_dim}"
                    )
                else:
                    validation_result["warnings"].append(
                        f"벡터 차원 범위 외: {vector_dim} (목표: {target_range})"
                    )

            # 3. Negative 샘플 검증 (선택적)
            negative_train_path = self.cache_dir / "negative_samples_train.npy"
            if negative_train_path.exists():
                train_samples = np.load(negative_train_path)
                validation_result["checks_passed"].append(
                    f"Negative 샘플 생성: {len(train_samples)}개"
                )

            # 4. 전체 요약 저장
            summary_path = self.cache_dir / "data_preparation_summary.json"
            summary = {
                "pipeline_execution": pipeline_results,
                "validation": validation_result,
                "generated_at": datetime.now().isoformat(),
            }

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            validation_result["summary_file"] = str(summary_path)

            self.logger.info(
                f"검증 완료: {len(validation_result['checks_passed'])}개 통과, {len(validation_result['checks_failed'])}개 실패"
            )

        except Exception as e:
            self.logger.error(f"결과 검증 중 오류: {e}")
            validation_result["success"] = False
            validation_result["error"] = str(e)

        return validation_result

    # ========== 내부 헬퍼 메서드들 ==========

    def _clear_pipeline_cache(self):
        """파이프라인 캐시 정리"""
        cache_files = [
            "feature_vector_full.npy",
            "feature_vector_full.names.json",
            "feature_vector_metadata.json",
            "negative_samples_train.npy",
            "negative_samples_test.npy",
            "negative_sampling_metadata.json",
            "data_preparation_summary.json",
        ]

        for filename in cache_files:
            file_path = self.cache_dir / filename
            if file_path.exists():
                file_path.unlink()
                self.logger.debug(f"캐시 파일 삭제: {filename}")

    def _check_analysis_output_files(self) -> Dict[str, str]:
        """분석 결과 파일들 확인"""
        output_files = {}

        # 주요 분석 결과 파일들
        analysis_files = [
            "unified_analysis.json",
            "pattern_analysis.json",
            "pair_analysis.json",
            "distribution_analysis.json",
            "roi_analysis.json",
        ]

        for filename in analysis_files:
            file_path = self.result_dir / filename
            if file_path.exists():
                output_files[filename.replace(".json", "")] = str(file_path)

        return output_files

    def _load_analysis_result(self) -> Dict[str, Any]:
        """분석 결과 로드 (캐시에서)"""
        try:
            unified_path = self.result_dir / "unified_analysis.json"
            if unified_path.exists():
                with open(unified_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"분석 결과 로드 실패: {e}")

        return {}

    def _load_vectorization_result(self) -> Dict[str, Any]:
        """벡터화 결과 로드 (캐시에서)"""
        try:
            metadata_path = self.cache_dir / "feature_vector_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # 벡터 데이터 로드
                vector_path = self.cache_dir / "feature_vector_full.npy"
                names_path = self.cache_dir / "feature_vector_full.names.json"

                if vector_path.exists() and names_path.exists():
                    vector = np.load(vector_path)
                    with open(names_path, "r", encoding="utf-8") as f:
                        names = json.load(f)

                    return {
                        "success": True,
                        "vector_data": {
                            "vector": vector,
                            "names": names,
                            "metadata": metadata,
                        },
                    }
        except Exception as e:
            self.logger.warning(f"벡터화 결과 로드 실패: {e}")

        return {"success": False}

    def _load_unified_analysis(self) -> Dict[str, Any]:
        """통합 분석 데이터 로드"""
        try:
            unified_path = self.result_dir / "unified_analysis.json"
            if unified_path.exists():
                with open(unified_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not data:
                        self.logger.warning("통합 분석 결과가 비어있습니다")
                        return {}
                    self.logger.info(f"통합 분석 데이터 로드 완료: {len(data)} 항목")
                    return data
            else:
                self.logger.warning(
                    "통합 분석 파일이 없습니다. 기본 분석 실행이 필요합니다."
                )
                # 대체 경로들 시도
                alternative_paths = [
                    self.result_dir / "optimized_analysis_result.json",
                    self.result_dir / "analysis_results.json",
                    Path("data/result/analysis/analysis_results.json"),
                    Path("data/result/analysis/optimized_analysis_result.json"),
                    Path("data/result/unified_analysis.json"),
                ]

                for alt_path in alternative_paths:
                    if alt_path.exists():
                        self.logger.info(f"대체 분석 파일 사용: {alt_path}")
                        with open(alt_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if data:
                                return data

                self.logger.error("사용 가능한 분석 결과 파일을 찾을 수 없습니다")
                return {}
        except Exception as e:
            self.logger.error(f"통합 분석 데이터 로드 실패: {e}")
            return {}

    def _validate_vector_quality(
        self, vector: np.ndarray, names: List[str]
    ) -> Dict[str, Any]:
        """벡터 품질 검증"""
        quality_metrics = {}

        try:
            # 기본 통계
            quality_metrics["mean"] = float(np.mean(vector))
            quality_metrics["std"] = float(np.std(vector))
            quality_metrics["min"] = float(np.min(vector))
            quality_metrics["max"] = float(np.max(vector))

            # 0값 비율 (낮을수록 좋음)
            zero_ratio = np.sum(vector == 0) / len(vector)
            quality_metrics["zero_ratio"] = float(zero_ratio)

            # 엔트로피 (높을수록 좋음)
            hist, _ = np.histogram(vector, bins=50)
            hist = hist[hist > 0]  # 0이 아닌 빈도만
            if len(hist) > 0:
                prob = hist / hist.sum()
                entropy = -np.sum(prob * np.log2(prob))
                quality_metrics["entropy"] = float(entropy)
            else:
                quality_metrics["entropy"] = 0.0

            # 전체 품질 점수 (0~1)
            entropy_score = min(
                quality_metrics["entropy"] / 6.0, 1.0
            )  # 6은 대략적인 최대 엔트로피
            zero_score = 1.0 - zero_ratio  # 0값이 적을수록 좋음
            variance_score = min(
                (
                    quality_metrics["std"] / quality_metrics["mean"]
                    if quality_metrics["mean"] > 0
                    else 0
                ),
                1.0,
            )

            overall_score = (entropy_score + zero_score + variance_score) / 3.0
            quality_metrics["overall_score"] = float(overall_score)

        except Exception as e:
            self.logger.warning(f"벡터 품질 검증 중 오류: {e}")
            quality_metrics["overall_score"] = 0.0

        return quality_metrics

    def _print_performance_summary(self, results: Dict[str, Any]):
        """성능 요약 출력"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 파이프라인 실행 요약")
        self.logger.info("=" * 60)

        # 실행된 단계
        executed = results.get("steps_executed", [])
        failed = results.get("steps_failed", [])

        self.logger.info(
            f"✅ 성공한 단계: {', '.join(executed) if executed else '없음'}"
        )
        if failed:
            self.logger.info(f"❌ 실패한 단계: {', '.join(failed)}")

        # 성능 메트릭
        metrics = results.get("performance_metrics", {})
        total_time = results.get("total_execution_time", 0)

        self.logger.info(f"⏱️  총 실행 시간: {total_time:.2f}초")

        for step, step_metrics in metrics.items():
            execution_time = step_metrics.get("execution_time", 0)
            self.logger.info(f"   - {step}: {execution_time:.2f}초")

        # 출력 파일 요약
        output_files = results.get("output_files", {})
        if output_files:
            self.logger.info(f"📁 생성된 파일: {len(output_files)}개")
            for category, path in output_files.items():
                self.logger.info(f"   - {category}: {Path(path).name}")

        # 메모리 사용량
        memory_usage = self.memory_manager.get_memory_usage()
        self.logger.info(f"💾 메모리 사용량: {memory_usage:.1f}MB")

        self.logger.info("=" * 60)

    def _save_pipeline_report(self, results: Dict[str, Any]):
        """파이프라인 실행 보고서 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.performance_dir / f"pipeline_report_{timestamp}.json"

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"실행 보고서 저장: {report_path}")

        except Exception as e:
            self.logger.warning(f"보고서 저장 실패: {e}")


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="DAEBAK AI 데이터 준비 통합 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--clear-cache", action="store_true", help="캐시 초기화 후 실행"
    )

    parser.add_argument(
        "--steps",
        type=str,
        default="analysis,vectorization,negative_sampling",
        help="실행할 단계 (쉼표로 구분): analysis, vectorization, negative_sampling",
    )

    parser.add_argument(
        "--debug", action="store_true", help="디버그 모드 (오류 시에도 계속 진행)"
    )

    parser.add_argument("--verbose", action="store_true", help="상세 로깅 활성화")

    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_arguments()

    # 최대 성능 최적화 모드 시작
    optimizer = launch_max_performance()

    try:
        pipeline = DataPreparationPipeline()

        # CLI 인자를 기반으로 실행 옵션 설정
        steps = args.steps.split(",") if args.steps else None

        # 파이프라인 실행
        pipeline.execute_full_pipeline(
            clear_cache=args.clear_cache,
            steps=steps,
            debug=args.debug,
            verbose=args.verbose,
        )

    except Exception as e:
        logger.error(f"메인 실행 중 심각한 오류 발생: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if optimizer:
            optimizer.cleanup()


if __name__ == "__main__":
    main()
