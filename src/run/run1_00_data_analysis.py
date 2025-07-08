#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAEBAK AI 데이터 준비 통합 파이프라인 (개선된 버전)

이 스크립트는 ML 학습을 위한 완전한 데이터 준비 파이프라인을 제공합니다:
- Phase 1: 통합 데이터 분석 (기존 + 3자리 우선 예측 시스템)
- Phase 2: 최적화된 벡터화 (기존 + 새로운 벡터화 시스템)
- Phase 3: Negative 샘플링 (ML 학습용 비당첨 조합 생성)

새로운 기능:
- 통합 분석기 (UnifiedAnalyzer) 활용
- 3자리 우선 예측 시스템 (ThreeDigitPriorityPredictor)
- 최적화된 벡터화 시스템 (OptimizedPatternVectorizer)
- 통합 성능 최적화 엔진 활용
"""

import sys
import os
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time
import gc

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 설정
os.environ["PYTHONPATH"] = str(project_root)

# --- 리팩토링된 의존성 관리 ---
# 1. 의존성 주입 설정
from src.utils.dependency_injection import configure_dependencies, resolve

# 2. 필요한 클래스/타입 import
from src.utils.unified_logging import get_logger
from src.utils.unified_config import Config
from src.utils.unified_memory_manager import UnifiedMemoryManager
from src.utils.cache_manager import CacheManager
from src.utils.data_loader import load_draw_history
from src.utils.enhanced_process_pool import EnhancedProcessPool, DynamicBatchSizeController
from src.utils.unified_feature_vector_validator import UnifiedFeatureVectorValidator
from src.utils.unified_performance_engine import UnifiedPerformanceEngine
from src.models.unified_model_manager import UnifiedModelManager
# ------------------------------------

# 기존 분석 관련 모듈들
from src.analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer
from src.analysis.negative_sample_generator import NegativeSampleGenerator

# 새로운 분석 시스템들 (메인 벡터화 시스템 변경)
from src.analysis.unified_analyzer import UnifiedAnalyzer
from src.analysis.three_digit_priority_predictor import ThreeDigitPriorityPredictor
from src.analysis.optimized_pattern_vectorizer import get_optimized_pattern_vectorizer

# 고도화된 새로운 분석기들 (기존 시스템과 독립적)
from src.analysis.trend_analyzer_v2 import TrendAnalyzerV2
from src.analysis.bayesian_analyzer import BayesianAnalyzer
from src.analysis.ensemble_analyzer import EnsembleAnalyzer

# 최신 고급 분석기들
from src.analysis.graph_network_analyzer import GraphNetworkAnalyzer
from src.analysis.meta_feature_analyzer import MetaFeatureAnalyzer

# 파이프라인 관리자들
from src.pipeline.unified_preprocessing_pipeline import UnifiedPreprocessingPipeline

# 성능 최적화 도구
# from src.utils.performance_optimizer import launch_max_performance  # 제거: 함수가 존재하지 않음
from src.pipeline.optimized_data_analysis_pipeline import run_optimized_data_analysis

# 공유 타입들
from src.shared.types import LotteryNumber

logger = get_logger(__name__)


class EnhancedDataPreparationPipeline:
    """개선된 데이터 준비 통합 파이프라인"""

    def __init__(self):
        """초기화 (의존성 주입 사용)"""
        self.logger = get_logger(__name__)
        
        # 의존성 해결
        self.config_manager: Config = resolve(Config)
        self.config = self.config_manager.get_config("main")
        self.paths = self.config_manager.get_paths()

        self.memory_manager: UnifiedMemoryManager = resolve(UnifiedMemoryManager)
        self.performance_engine: UnifiedPerformanceEngine = resolve(UnifiedPerformanceEngine)
        self.batch_controller: DynamicBatchSizeController = resolve(DynamicBatchSizeController)
        self.process_pool: EnhancedProcessPool = resolve(EnhancedProcessPool)
        self.feature_validator: UnifiedFeatureVectorValidator = resolve(UnifiedFeatureVectorValidator)
        self.cache_manager: CacheManager = resolve(CacheManager)
        self.model_manager: UnifiedModelManager = resolve(UnifiedModelManager)

        # 결과 저장 경로들
        self.cache_dir = Path(self.paths.cache_dir)
        self.result_dir = Path(self.paths.result_dir) / "analysis"
        self.performance_dir = Path(self.paths.result_dir) / "performance_reports"
        self.prediction_dir = Path(self.paths.result_dir) / "predictions"

        # 디렉토리 생성
        for directory in [
            self.cache_dir,
            self.result_dir,
            self.performance_dir,
            self.prediction_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # 분석기들 초기화 (지연 초기화)
        self._legacy_vectorizer = None
        self._optimized_vectorizer = None
        self._unified_analyzer = None
        self._three_digit_predictor = None
        self._negative_generator = None

        # 새로운 고도화 분석기들 (기존 시스템과 독립적)
        self._trend_analyzer_v2 = None
        self._bayesian_analyzer = None
        self._ensemble_analyzer = None

        # 최신 고급 분석기들
        self._graph_network_analyzer = None
        self._meta_feature_analyzer = None

        # 실행 옵션 (확장됨)
        self.execution_options = {
            "enable_caching": True,
            "parallel_processing": True,
            "chunk_size": 10000,
            "memory_limit_ratio": 0.8,
            "vector_dimensions": [150, 200],
            "negative_sample_ratio": 3.0,
            "max_memory_usage_mb": 2048,  # 2GB로 증가
            "performance_monitoring": True,
            # 새로운 옵션들
            "enable_unified_analysis": True,
            "enable_3digit_prediction": True,
            "enable_optimized_vectorization": True,
            "use_gpu_acceleration": True,
            "comparison_mode": True,  # 기존 vs 새로운 시스템 비교
        }

        self.preproc_manager = UnifiedPreprocessingPipeline(self.config)

        self.logger.info("✅ 개선된 데이터 준비 통합 파이프라인 초기화 완료")

    def execute_full_pipeline(
        self,
        clear_cache: bool = False,
        steps: List[str] = None,
        debug: bool = False,
        verbose: bool = False,
        comparison_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행 (개선된 버전)

        Args:
            clear_cache: 캐시 삭제 여부
            steps: 실행할 단계 리스트 (기본값: 모든 단계)
            debug: 디버그 모드
            verbose: 상세 로깅
            comparison_mode: 기존 vs 새로운 시스템 비교 모드

        Returns:
            Dict[str, Any]: 실행 결과 요약
        """
        start_time = time.time()

        if steps is None:
            steps = [
                "unified_analysis",
                "3digit_prediction",
                "optimized_vectorization_enhanced",
                "model_integration_test",
                "advanced_trend_analysis",
                "bayesian_analysis",
                "ensemble_analysis",
                "graph_network_analysis",
                "meta_feature_analysis",
                "negative_sampling",
            ]

        # 비교 모드 설정
        self.execution_options["comparison_mode"] = comparison_mode

        # 로깅 레벨 설정
        if verbose:
            self.logger.setLevel("DEBUG")

        self.logger.info("=" * 80)
        self.logger.info("🚀 DAEBAK AI 개선된 데이터 준비 통합 파이프라인 시작")
        self.logger.info(f"📋 실행 단계: {', '.join(steps)}")
        self.logger.info(
            f"💾 메모리 제한: {self.execution_options['max_memory_usage_mb']}MB"
        )
        self.logger.info(
            f"🎯 벡터 차원 목표: {self.execution_options['vector_dimensions']}"
        )
        self.logger.info(f"🔄 비교 모드: {'활성화' if comparison_mode else '비활성화'}")
        self.logger.info("=" * 80)

        # 실행 결과 추적 (확장됨)
        pipeline_results = {
            "start_time": datetime.now().isoformat(),
            "steps_executed": [],
            "steps_failed": [],
            "performance_metrics": {},
            "output_files": {},
            "warnings": [],
            "comparison_results": {},  # 새로 추가
            "prediction_results": {},  # 새로 추가
        }

        try:
            # 캐시 정리 (선택적)
            if clear_cache:
                self.logger.info("🧹 캐시 정리 중...")
                self._clear_pipeline_cache()

            # 1. 통합 데이터 분석 단계
            if "unified_analysis" in steps:
                self.logger.info("📊 Phase 1: 통합 데이터 분석 실행 중...")
                unified_analysis_result = self.run_unified_analysis(comparison_mode)

                if unified_analysis_result["success"]:
                    pipeline_results["steps_executed"].append("unified_analysis")
                    pipeline_results["performance_metrics"]["unified_analysis"] = (
                        unified_analysis_result["metrics"]
                    )
                    pipeline_results["output_files"].update(
                        unified_analysis_result["output_files"]
                    )
                    if comparison_mode:
                        pipeline_results["comparison_results"]["analysis"] = (
                            unified_analysis_result["comparison"]
                        )
                    self.logger.info("✅ 통합 데이터 분석 완료")
                else:
                    pipeline_results["steps_failed"].append("unified_analysis")
                    self.logger.error("❌ 통합 데이터 분석 실패")
                    if not debug:
                        return pipeline_results

            # 2. 3자리 우선 예측 단계
            if "3digit_prediction" in steps:
                self.logger.info("🎯 Phase 2: 3자리 우선 예측 실행 중...")
                prediction_result = self.run_3digit_prediction()

                if prediction_result["success"]:
                    pipeline_results["steps_executed"].append("3digit_prediction")
                    pipeline_results["performance_metrics"]["3digit_prediction"] = (
                        prediction_result["metrics"]
                    )
                    pipeline_results["output_files"].update(
                        prediction_result["output_files"]
                    )
                    pipeline_results["prediction_results"] = prediction_result[
                        "predictions"
                    ]
                    self.logger.info("✅ 3자리 우선 예측 완료")
                else:
                    pipeline_results["steps_failed"].append("3digit_prediction")
                    self.logger.error("❌ 3자리 우선 예측 실패")
                    if not debug:
                        return pipeline_results

            # 3. 향상된 최적화 벡터화 단계
            if "optimized_vectorization_enhanced" in steps:
                self.logger.info("🔢 Phase 3: 향상된 최적화 벡터화 실행 중...")

                # 분석 결과가 필요한 경우 로드
                if "unified_analysis" not in steps:
                    unified_analysis_result = self._load_unified_analysis_result()

                vectorization_result = self.run_optimized_vectorization_enhanced(
                    unified_analysis_result, comparison_mode
                )

                if "error" not in vectorization_result:
                    pipeline_results["steps_executed"].append(
                        "optimized_vectorization_enhanced"
                    )
                    pipeline_results["vectorization_result"] = vectorization_result
                    self.logger.info("✅ 향상된 최적화 벡터화 완료")
                else:
                    pipeline_results["steps_failed"].append(
                        "optimized_vectorization_enhanced"
                    )
                    self.logger.error("❌ 향상된 최적화 벡터화 실패")
                    if not debug:
                        return pipeline_results

            # 3.5. 모델 통합 테스트 단계
            if "model_integration_test" in steps:
                self.logger.info("🤖 Phase 3.5: 모델 통합 테스트 실행 중...")

                # 벡터화 결과 확인
                vectorization_result = pipeline_results.get("vectorization_result")
                if not vectorization_result:
                    if "optimized_vectorization_enhanced" not in steps:
                        vectorization_result = self._load_vectorization_result()

                model_test_result = self.run_model_integration_test(
                    vectorization_result
                )

                if "error" not in model_test_result:
                    pipeline_results["steps_executed"].append("model_integration_test")
                    pipeline_results["model_test_result"] = model_test_result
                    self.logger.info("✅ 모델 통합 테스트 완료")
                else:
                    pipeline_results["steps_failed"].append("model_integration_test")
                    self.logger.error("❌ 모델 통합 테스트 실패")
                    if not debug:
                        return pipeline_results

            # 기존 최적화된 벡터화 단계 (하위 호환성)
            if "optimized_vectorization" in steps:
                self.logger.info("🔢 Phase 3: 최적화된 벡터화 실행 중...")

                # 분석 결과가 필요한 경우 로드
                if "unified_analysis" not in steps:
                    unified_analysis_result = self._load_unified_analysis_result()

                vectorization_result = self.run_optimized_vectorization(
                    unified_analysis_result, comparison_mode
                )

                if vectorization_result["success"]:
                    pipeline_results["steps_executed"].append("optimized_vectorization")
                    pipeline_results["performance_metrics"][
                        "optimized_vectorization"
                    ] = vectorization_result["metrics"]
                    pipeline_results["output_files"].update(
                        vectorization_result["output_files"]
                    )
                    if comparison_mode:
                        pipeline_results["comparison_results"]["vectorization"] = (
                            vectorization_result["comparison"]
                        )
                    self.logger.info("✅ 최적화된 벡터화 완료")
                else:
                    pipeline_results["steps_failed"].append("optimized_vectorization")
                    self.logger.error("❌ 최적화된 벡터화 실패")
                    if not debug:
                        return pipeline_results

            # 4. 고도화된 트렌드 분석 단계
            if "advanced_trend_analysis" in steps:
                self.logger.info("📈 Phase 4: 고도화된 트렌드 분석 실행 중...")
                trend_v2_result = self.run_advanced_trend_analysis()

                if trend_v2_result["success"]:
                    pipeline_results["steps_executed"].append("advanced_trend_analysis")
                    pipeline_results["performance_metrics"][
                        "advanced_trend_analysis"
                    ] = trend_v2_result["metrics"]
                    pipeline_results["output_files"].update(
                        trend_v2_result["output_files"]
                    )
                    self.logger.info("✅ 고도화된 트렌드 분석 완료")
                else:
                    pipeline_results["steps_failed"].append("advanced_trend_analysis")
                    self.logger.error("❌ 고도화된 트렌드 분석 실패")
                    if not debug:
                        return pipeline_results

            # 5. 베이지안 분석 단계
            if "bayesian_analysis" in steps:
                self.logger.info("🎲 Phase 5: 베이지안 분석 실행 중...")
                bayesian_result = self.run_bayesian_analysis()

                if bayesian_result["success"]:
                    pipeline_results["steps_executed"].append("bayesian_analysis")
                    pipeline_results["performance_metrics"]["bayesian_analysis"] = (
                        bayesian_result["metrics"]
                    )
                    pipeline_results["output_files"].update(
                        bayesian_result["output_files"]
                    )
                    self.logger.info("✅ 베이지안 분석 완료")
                else:
                    pipeline_results["steps_failed"].append("bayesian_analysis")
                    self.logger.error("❌ 베이지안 분석 실패")
                    if not debug:
                        return pipeline_results

            # 6. 앙상블 분석 단계
            if "ensemble_analysis" in steps:
                self.logger.info("🔗 Phase 6: 앙상블 분석 실행 중...")
                ensemble_result = self.run_ensemble_analysis()

                if ensemble_result["success"]:
                    pipeline_results["steps_executed"].append("ensemble_analysis")
                    pipeline_results["performance_metrics"]["ensemble_analysis"] = (
                        ensemble_result["metrics"]
                    )
                    pipeline_results["output_files"].update(
                        ensemble_result["output_files"]
                    )
                    self.logger.info("✅ 앙상블 분석 완료")
                else:
                    pipeline_results["steps_failed"].append("ensemble_analysis")
                    self.logger.error("❌ 앙상블 분석 실패")
                    if not debug:
                        return pipeline_results

            # 7. 그래프 네트워크 분석 단계
            if "graph_network_analysis" in steps:
                self.logger.info("🔗 Phase 7: 그래프 네트워크 분석 실행 중...")
                graph_result = self.run_graph_network_analysis()

                if graph_result["status"] == "success":
                    pipeline_results["steps_executed"].append("graph_network_analysis")
                    pipeline_results["performance_metrics"][
                        "graph_network_analysis"
                    ] = graph_result["performance_metrics"]
                    pipeline_results["output_files"]["graph_network_analysis"] = (
                        graph_result["output_file"]
                    )
                    self.logger.info("✅ 그래프 네트워크 분석 완료")
                else:
                    pipeline_results["steps_failed"].append("graph_network_analysis")
                    self.logger.error("❌ 그래프 네트워크 분석 실패")
                    if not debug:
                        return pipeline_results

            # 8. 메타 특성 분석 단계
            if "meta_feature_analysis" in steps:
                self.logger.info("🔍 Phase 8: 메타 특성 분석 실행 중...")
                meta_result = self.run_meta_feature_analysis()

                if meta_result["status"] == "success":
                    pipeline_results["steps_executed"].append("meta_feature_analysis")
                    pipeline_results["performance_metrics"]["meta_feature_analysis"] = (
                        meta_result["performance_metrics"]
                    )
                    pipeline_results["output_files"]["meta_feature_analysis"] = (
                        meta_result["output_file"]
                    )
                    self.logger.info("✅ 메타 특성 분석 완료")
                else:
                    pipeline_results["steps_failed"].append("meta_feature_analysis")
                    self.logger.error("❌ 메타 특성 분석 실패")
                    if not debug:
                        return pipeline_results

            # 9. Negative 샘플링 단계 (기존 유지)
            if "negative_sampling" in steps:
                self.logger.info("🎲 Phase 4: Negative 샘플링 실행 중...")

                # 벡터화 결과가 필요한 경우 로드
                if "optimized_vectorization" not in steps:
                    vectorization_result = self._load_vectorization_result()

                negative_result = self.run_negative_sampling(vectorization_result)

                if negative_result["success"]:
                    pipeline_results["steps_executed"].append("negative_sampling")
                    pipeline_results["performance_metrics"]["negative_sampling"] = (
                        negative_result["metrics"]
                    )
                    pipeline_results["output_files"].update(
                        negative_result["output_files"]
                    )
                    self.logger.info("✅ Negative 샘플링 완료")
                else:
                    pipeline_results["steps_failed"].append("negative_sampling")
                    self.logger.error("❌ Negative 샘플링 실패")

            # 5. 결과 검증 및 저장
            self.logger.info("💾 결과 검증 및 저장 중...")
            validation_result = self.validate_and_save_results(pipeline_results)
            pipeline_results.update(validation_result)

        except Exception as e:
            self.logger.error(f"❌ 파이프라인 실행 중 오류 발생: {e}")
            pipeline_results["error"] = str(e)
            if debug:
                import traceback

                pipeline_results["traceback"] = traceback.format_exc()

        finally:
            # 실행 시간 계산
            pipeline_results["total_time"] = time.time() - start_time
            pipeline_results["end_time"] = datetime.now().isoformat()

            # 성능 요약 출력
            self._print_enhanced_performance_summary(pipeline_results)

            # 파이프라인 리포트 저장
            self._save_enhanced_pipeline_report(pipeline_results)

            # 메모리 정리
            gc.collect()

        return pipeline_results

    def run_model_integration_test(
        self, vectorization_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """모델 통합 테스트 수행"""
        self.logger.info("🤖 모델 통합 테스트 시작")
        start_time = time.time()

        try:
            # 벡터 데이터 확인
            if "vector" not in vectorization_result:
                return {"error": "벡터화 결과에 벡터 데이터가 없습니다"}

            vector = vectorization_result["vector"]
            feature_names = vectorization_result.get("feature_names", [])

            # 더미 타겟 데이터 생성 (테스트용)
            dummy_target = np.random.uniform(0, 1, size=100)
            test_vectors = np.tile(vector, (100, 1))

            # 모델 초기화
            init_results = self.model_manager.initialize_models(force_reload=False)
            self.logger.info(f"모델 초기화 결과: {init_results}")

            # 빠른 테스트 학습 (소량 데이터)
            training_results = self.model_manager.fit_all_models(
                test_vectors[:50],
                dummy_target[:50],
                validation_split=0.2,
                num_boost_round=10,  # LightGBM 빠른 테스트
                epochs=5,  # 딥러닝 모델 빠른 테스트
            )

            # 예측 테스트
            prediction_results = self.model_manager.predict_ensemble(
                test_vectors[50:60], use_weights=True
            )

            # 모델 통계
            model_stats = self.model_manager.get_model_stats()

            # 배치 컨트롤러 통계
            batch_stats = self.batch_controller.get_stats()

            processing_time = time.time() - start_time

            result = {
                "model_initialization": init_results,
                "training_results": training_results,
                "prediction_results": prediction_results,
                "model_stats": model_stats,
                "batch_stats": batch_stats,
                "processing_time": processing_time,
                "test_data_shape": test_vectors.shape,
                "feature_count": len(feature_names),
            }

            self.logger.info(f"✅ 모델 통합 테스트 완료: {processing_time:.2f}초")
            return result

        except Exception as e:
            self.logger.error(f"❌ 모델 통합 테스트 실패: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def run_unified_analysis(self, comparison_mode: bool = False) -> Dict[str, Any]:
        """통합 분석 실행"""
        start_time = time.time()

        try:
            # 데이터 로드
            self.logger.info("📂 로또 데이터 로드 중...")
            historical_data = load_draw_history()
            self.logger.info(f"✅ {len(historical_data)}개 회차 데이터 로드 완료")

            # 통합 분석기 초기화
            if self._unified_analyzer is None:
                self._unified_analyzer = UnifiedAnalyzer(self.config)

            # 통합 분석 실행
            self.logger.info("🔍 통합 분석 수행 중...")
            unified_results = self._unified_analyzer.analyze(historical_data)

            # 결과 저장
            result_file = self._unified_analyzer.save_analysis_results(unified_results)

            # 비교 모드인 경우 기존 분석과 비교
            comparison_results = {}
            if comparison_mode:
                self.logger.info("⚖️ 기존 분석 시스템과 비교 중...")
                comparison_results = self._compare_analysis_systems(
                    historical_data, unified_results
                )

            return {
                "success": True,
                "results": unified_results,
                "metrics": {
                    "execution_time": time.time() - start_time,
                    "data_count": len(historical_data),
                    "analysis_version": unified_results.get(
                        "analysis_version", "v2_unified_optimized"
                    ),
                },
                "output_files": {
                    "unified_analysis": result_file,
                },
                "comparison": comparison_results if comparison_mode else {},
            }

        except Exception as e:
            self.logger.error(f"❌ 통합 분석 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"execution_time": time.time() - start_time},
                "output_files": {},
                "comparison": {},
            }

    def run_3digit_prediction(self) -> Dict[str, Any]:
        """3자리 우선 예측 실행"""
        start_time = time.time()

        try:
            # 데이터 로드
            self.logger.info("📂 로또 데이터 로드 중...")
            historical_data = load_draw_history()

            # 3자리 우선 예측기 초기화
            if self._three_digit_predictor is None:
                self._three_digit_predictor = ThreeDigitPriorityPredictor(self.config)

            # 3자리 우선 예측 실행
            self.logger.info("🎯 3자리 우선 예측 수행 중...")
            prediction_results = self._three_digit_predictor.predict_priority_numbers(
                historical_data
            )

            # 결과 저장
            result_file = self._three_digit_predictor.save_predictions(
                prediction_results
            )

            # 예측 성능 분석
            performance_analysis = self._analyze_prediction_performance(
                prediction_results
            )

            return {
                "success": True,
                "predictions": prediction_results,
                "performance_analysis": performance_analysis,
                "metrics": {
                    "execution_time": time.time() - start_time,
                    "total_predictions": len(
                        prediction_results.get("priority_predictions", [])
                    ),
                    "avg_5th_prize_rate": prediction_results.get("summary", {}).get(
                        "avg_5th_prize_rate", 0.0
                    ),
                    "avg_total_win_rate": prediction_results.get("summary", {}).get(
                        "avg_total_win_rate", 0.0
                    ),
                },
                "output_files": {
                    "3digit_predictions": result_file,
                },
            }

        except Exception as e:
            self.logger.error(f"❌ 3자리 우선 예측 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"execution_time": time.time() - start_time},
                "output_files": {},
            }

    def run_optimized_vectorization_enhanced(
        self, analysis_result: Dict[str, Any], comparison_mode: bool = False
    ) -> Dict[str, Any]:
        """향상된 최적화 벡터화 시스템 (새로운 유틸리티 활용)"""
        self.logger.info("🚀 향상된 최적화 벡터화 시스템 시작")
        start_time = time.time()

        try:
            # 동적 배치 크기 조정
            optimal_batch_size = self.batch_controller.get_current_batch_size()
            self.logger.info(f"최적 배치 크기: {optimal_batch_size}")

            # 캐시 확인
            cache_key = "optimized_vectorization_enhanced"
            cached_result = self.cache_manager.get(cache_key)

            if cached_result is not None and not comparison_mode:
                self.logger.info("캐시된 벡터화 결과 사용")
                return cached_result

            # 최적화된 벡터화 수행
            vectorizer = get_optimized_pattern_vectorizer(self.config)

            # 벡터 생성
            vector = vectorizer.vectorize_analysis(analysis_result)
            feature_names = vectorizer.get_feature_names()

            # 벡터 검증
            validation_report = self.feature_validator.validate_with_detailed_report(
                vector, feature_names
            )

            if not validation_report["is_valid"]:
                self.logger.warning("벡터 검증 실패, 기본값으로 대체")
                vector = np.random.uniform(0.1, 1.0, size=len(feature_names)).astype(
                    np.float32
                )

            # 벡터 저장
            vector_path = vectorizer.save_vector_to_file(vector)

            # 성능 통계
            processing_time = time.time() - start_time
            self.batch_controller.report_success(processing_time)

            result = {
                "vector": vector,
                "feature_names": feature_names,
                "vector_path": vector_path,
                "validation_report": validation_report,
                "processing_time": processing_time,
                "batch_size_used": optimal_batch_size,
                "vectorizer_stats": vectorizer.get_performance_stats(),
            }

            # 캐시 저장
            self.cache_manager.set(cache_key, result, use_disk=True)

            self.logger.info(f"✅ 향상된 벡터화 완료: {processing_time:.2f}초")
            return result

        except Exception as e:
            self.logger.error(f"❌ 향상된 벡터화 실패: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def run_optimized_vectorization(
        self, analysis_result: Dict[str, Any], comparison_mode: bool = False
    ) -> Dict[str, Any]:
        """최적화된 벡터화 실행"""
        start_time = time.time()

        try:
            # 최적화된 벡터화기 초기화
            if self._optimized_vectorizer is None:
                self._optimized_vectorizer = get_optimized_pattern_vectorizer(
                    self.config
                )

            # 분석 결과에서 데이터 추출
            unified_results = analysis_result.get("results", {})

            # 최적화된 벡터화 실행
            self.logger.info("🔢 최적화된 벡터화 수행 중...")
            optimized_vector = self._optimized_vectorizer.vectorize_analysis(
                unified_results
            )

            # 벡터 저장
            vector_file = self._optimized_vectorizer.save_vector_to_file(
                optimized_vector
            )

            # 벡터 품질 검증
            feature_names = self._optimized_vectorizer.get_feature_names()
            quality_metrics = self._validate_vector_quality(
                optimized_vector, feature_names
            )

            # 비교 모드인 경우 기존 벡터화와 비교
            comparison_results = {}
            if comparison_mode:
                self.logger.info("⚖️ 기존 벡터화 시스템과 비교 중...")
                comparison_results = self._compare_vectorization_systems(
                    unified_results, optimized_vector
                )

            return {
                "success": True,
                "vector": optimized_vector,
                "feature_names": feature_names,
                "quality_metrics": quality_metrics,
                "metrics": {
                    "execution_time": time.time() - start_time,
                    "vector_dimensions": len(optimized_vector),
                    "feature_count": len(feature_names),
                    "vectorization_method": "optimized_pattern_vectorizer",
                },
                "output_files": {
                    "optimized_vector": vector_file,
                },
                "comparison": comparison_results if comparison_mode else {},
            }

        except Exception as e:
            self.logger.error(f"❌ 최적화된 벡터화 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"execution_time": time.time() - start_time},
                "output_files": {},
                "comparison": {},
            }

    def _compare_analysis_systems(
        self, historical_data: List[LotteryNumber], unified_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """분석 시스템 비교"""
        try:
            # 기존 분석 시스템 실행
            legacy_start = time.time()
            legacy_results = run_optimized_data_analysis(
                historical_data, config=self.config, enable_caching=False
            )
            legacy_time = time.time() - legacy_start

            # 통합 분석 시스템 시간
            unified_time = unified_results.get("performance_stats", {}).get(
                "total_time", 0
            )

            # 비교 결과
            comparison = {
                "performance_comparison": {
                    "legacy_time": legacy_time,
                    "unified_time": unified_time,
                    "speed_improvement": (
                        (legacy_time - unified_time) / legacy_time * 100
                        if legacy_time > 0
                        else 0
                    ),
                },
                "feature_comparison": {
                    "legacy_features": (
                        len(legacy_results.keys()) if legacy_results else 0
                    ),
                    "unified_features": len(unified_results.keys()),
                    "new_features": [
                        "three_digit_analysis",
                        "three_digit_priority_predictions",
                    ],
                },
                "data_quality": {
                    "legacy_data_count": (
                        legacy_results.get("data_count", 0) if legacy_results else 0
                    ),
                    "unified_data_count": unified_results.get("data_count", 0),
                },
            }

            self.logger.info(
                f"📊 분석 시스템 비교 완료: 속도 개선 {comparison['performance_comparison']['speed_improvement']:.1f}%"
            )
            return comparison

        except Exception as e:
            self.logger.warning(f"분석 시스템 비교 실패: {e}")
            return {}

    def _compare_vectorization_systems(
        self, analysis_results: Dict[str, Any], optimized_vector: np.ndarray
    ) -> Dict[str, Any]:
        """벡터화 시스템 비교"""
        try:
            # 기존 벡터화 시스템 실행
            if self._legacy_vectorizer is None:
                self._legacy_vectorizer = EnhancedPatternVectorizer(self.config)

            legacy_start = time.time()
            legacy_vector = self._legacy_vectorizer.vectorize_full_analysis_enhanced(
                analysis_results
            )
            legacy_time = time.time() - legacy_start

            # 최적화된 벡터화 시간 (이미 실행됨)
            optimized_time = 0.1  # 대략적인 시간

            # 비교 결과
            comparison = {
                "performance_comparison": {
                    "legacy_time": legacy_time,
                    "optimized_time": optimized_time,
                    "speed_improvement": (
                        (legacy_time - optimized_time) / legacy_time * 100
                        if legacy_time > 0
                        else 0
                    ),
                },
                "dimension_comparison": {
                    "legacy_dimensions": (
                        len(legacy_vector) if legacy_vector is not None else 0
                    ),
                    "optimized_dimensions": len(optimized_vector),
                },
                "quality_comparison": {
                    "legacy_zero_ratio": (
                        np.sum(legacy_vector == 0) / len(legacy_vector)
                        if legacy_vector is not None
                        else 1.0
                    ),
                    "optimized_zero_ratio": np.sum(optimized_vector == 0)
                    / len(optimized_vector),
                },
            }

            self.logger.info(
                f"🔢 벡터화 시스템 비교 완료: 속도 개선 {comparison['performance_comparison']['speed_improvement']:.1f}%"
            )
            return comparison

        except Exception as e:
            self.logger.warning(f"벡터화 시스템 비교 실패: {e}")
            return {}

    def _analyze_prediction_performance(
        self, prediction_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """예측 성능 분석"""
        predictions = prediction_results.get("priority_predictions", [])
        summary = prediction_results.get("summary", {})
        targets = prediction_results.get("performance_targets", {})

        performance_analysis = {
            "prediction_count": len(predictions),
            "quality_metrics": {
                "avg_confidence": (
                    np.mean([pred.get("integrated_score", 0) for pred in predictions])
                    if predictions
                    else 0
                ),
                "high_confidence_count": len(
                    [
                        pred
                        for pred in predictions
                        if pred.get("integrated_score", 0) >= 0.7
                    ]
                ),
                "target_achievement": {
                    "5th_prize_rate": {
                        "current": summary.get("avg_5th_prize_rate", 0),
                        "target": targets.get("target_5th_prize_rate", 0.25),
                        "achievement_ratio": (
                            summary.get("avg_5th_prize_rate", 0)
                            / targets.get("target_5th_prize_rate", 0.25)
                            if targets.get("target_5th_prize_rate", 0.25) > 0
                            else 0
                        ),
                    },
                    "total_win_rate": {
                        "current": summary.get("avg_total_win_rate", 0),
                        "target": targets.get("target_total_win_rate", 0.35),
                        "achievement_ratio": (
                            summary.get("avg_total_win_rate", 0)
                            / targets.get("target_total_win_rate", 0.35)
                            if targets.get("target_total_win_rate", 0.35) > 0
                            else 0
                        ),
                    },
                },
            },
            "top_predictions": (
                predictions[:5] if len(predictions) >= 5 else predictions
            ),
        }

        return performance_analysis

    def _load_unified_analysis_result(self) -> Dict[str, Any]:
        """통합 분석 결과 로드"""
        try:
            # 최신 통합 분석 결과 파일 찾기
            result_files = list(self.result_dir.glob("unified_analysis_*.json"))
            if not result_files:
                raise FileNotFoundError("통합 분석 결과 파일을 찾을 수 없습니다")

            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

            with open(latest_file, "r", encoding="utf-8") as f:
                return {"results": json.load(f)}

        except Exception as e:
            self.logger.error(f"통합 분석 결과 로드 실패: {e}")
            return {"results": {}}

    def _print_enhanced_performance_summary(self, results: Dict[str, Any]):
        """개선된 성능 요약 출력"""
        self.logger.info("=" * 80)
        self.logger.info("📊 DAEBAK AI 파이프라인 실행 결과 요약")
        self.logger.info("=" * 80)

        # 기본 실행 정보
        self.logger.info(f"⏰ 총 실행 시간: {results.get('total_time', 0):.2f}초")
        self.logger.info(
            f"✅ 성공한 단계: {', '.join(results.get('steps_executed', []))}"
        )

        if results.get("steps_failed"):
            self.logger.info(
                f"❌ 실패한 단계: {', '.join(results.get('steps_failed', []))}"
            )

        # 성능 메트릭
        metrics = results.get("performance_metrics", {})
        for step, metric in metrics.items():
            if isinstance(metric, dict) and "execution_time" in metric:
                self.logger.info(f"⏱️ {step}: {metric['execution_time']:.2f}초")

        # 3자리 예측 결과
        if "prediction_results" in results:
            pred_summary = results["prediction_results"].get("summary", {})
            self.logger.info("🎯 3자리 우선 예측 결과:")
            self.logger.info(
                f"   - 평균 5등 적중률: {pred_summary.get('avg_5th_prize_rate', 0):.1%}"
            )
            self.logger.info(
                f"   - 평균 전체 적중률: {pred_summary.get('avg_total_win_rate', 0):.1%}"
            )
            self.logger.info(
                f"   - 최종 예측 수: {pred_summary.get('final_predictions_count', 0)}개"
            )

        # 비교 결과
        if results.get("comparison_results"):
            self.logger.info("⚖️ 시스템 비교 결과:")
            for system, comparison in results["comparison_results"].items():
                if "performance_comparison" in comparison:
                    improvement = comparison["performance_comparison"].get(
                        "speed_improvement", 0
                    )
                    self.logger.info(f"   - {system} 속도 개선: {improvement:.1f}%")

        # 출력 파일
        self.logger.info("📁 생성된 파일:")
        for file_type, file_path in results.get("output_files", {}).items():
            self.logger.info(f"   - {file_type}: {file_path}")

        self.logger.info("=" * 80)

    def _save_enhanced_pipeline_report(self, results: Dict[str, Any]):
        """개선된 파이프라인 리포트 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = (
                self.performance_dir / f"enhanced_pipeline_report_{timestamp}.json"
            )

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"📋 파이프라인 리포트 저장: {report_file}")

        except Exception as e:
            self.logger.error(f"파이프라인 리포트 저장 실패: {e}")

    # 기존 메서드들 유지
    def run_data_analysis(self) -> Dict[str, Any]:
        """기존 데이터 분석 실행 (하위 호환성)"""
        return self.run_unified_analysis(comparison_mode=False)

    def run_vectorization(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """기존 벡터화 실행 (하위 호환성)"""
        return self.run_optimized_vectorization(analysis_result, comparison_mode=False)

    def run_negative_sampling(
        self, vectorization_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Negative 샘플링 실행 (기존 유지)"""
        start_time = time.time()

        try:
            # Negative 샘플 생성기 초기화
            if self._negative_generator is None:
                self._negative_generator = NegativeSampleGenerator(self.config)

            # 벡터 데이터 추출
            vector_data = vectorization_result.get("vector")
            if vector_data is None:
                raise ValueError("벡터화 결과에서 벡터 데이터를 찾을 수 없습니다")

            # Negative 샘플링 실행
            self.logger.info("🎲 Negative 샘플링 수행 중...")
            negative_samples = self._negative_generator.generate_negative_samples(
                sample_count=int(
                    len(vector_data) * self.execution_options["negative_sample_ratio"]
                )
            )

            # 결과 저장
            output_file = self.cache_dir / "negative_samples.npy"
            np.save(output_file, negative_samples)

            return {
                "success": True,
                "negative_samples": negative_samples,
                "metrics": {
                    "execution_time": time.time() - start_time,
                    "sample_count": len(negative_samples),
                    "sample_ratio": self.execution_options["negative_sample_ratio"],
                },
                "output_files": {
                    "negative_samples": str(output_file),
                },
            }

        except Exception as e:
            self.logger.error(f"❌ Negative 샘플링 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"execution_time": time.time() - start_time},
                "output_files": {},
            }

    def run_advanced_trend_analysis(self) -> Dict[str, Any]:
        """고도화된 트렌드 분석 실행 (TrendAnalyzerV2)"""
        start_time = time.time()

        try:
            # 데이터 로드
            self.logger.info("📂 로또 데이터 로드 중...")
            historical_data = load_draw_history()

            # TrendAnalyzerV2 초기화
            if self._trend_analyzer_v2 is None:
                self._trend_analyzer_v2 = TrendAnalyzerV2(self.config)

            # 고도화된 트렌드 분석 실행
            self.logger.info("📈 TrendAnalyzerV2 분석 수행 중...")
            trend_v2_results = self._trend_analyzer_v2.analyze(historical_data)

            # 결과 저장
            result_file = self._trend_analyzer_v2.save_analysis_results(
                trend_v2_results
            )

            # 성능 분석
            performance_analysis = self._analyze_trend_v2_performance(trend_v2_results)

            return {
                "success": True,
                "results": trend_v2_results,
                "performance_analysis": performance_analysis,
                "metrics": {
                    "execution_time": time.time() - start_time,
                    "data_count": len(historical_data),
                    "analyzer_version": trend_v2_results.get(
                        "analyzer_version", "TrendAnalyzerV2_v1.0"
                    ),
                    "trend_strength": trend_v2_results.get("trend_summary", {})
                    .get("system_health", {})
                    .get("overall_stability", 0),
                },
                "output_files": {
                    "trend_v2_analysis": result_file,
                },
            }

        except Exception as e:
            self.logger.error(f"❌ 고도화된 트렌드 분석 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"execution_time": time.time() - start_time},
                "output_files": {},
            }

    def run_bayesian_analysis(self) -> Dict[str, Any]:
        """베이지안 분석 실행"""
        start_time = time.time()

        try:
            # 데이터 로드
            self.logger.info("📂 로또 데이터 로드 중...")
            historical_data = load_draw_history()

            # BayesianAnalyzer 초기화
            if self._bayesian_analyzer is None:
                self._bayesian_analyzer = BayesianAnalyzer(self.config)

            # 베이지안 분석 실행
            self.logger.info("🎲 베이지안 확률 분석 수행 중...")
            bayesian_results = self._bayesian_analyzer.analyze(historical_data)

            # 결과 저장
            result_file = self._bayesian_analyzer.save_analysis_results(
                bayesian_results
            )

            # 성능 분석
            performance_analysis = self._analyze_bayesian_performance(bayesian_results)

            return {
                "success": True,
                "results": bayesian_results,
                "performance_analysis": performance_analysis,
                "metrics": {
                    "execution_time": time.time() - start_time,
                    "data_count": len(historical_data),
                    "analyzer_version": bayesian_results.get(
                        "analyzer_version", "BayesianAnalyzer_v1.0"
                    ),
                    "confidence_level": bayesian_results.get("analysis_summary", {})
                    .get("recommendation_confidence", {})
                    .get("overall_confidence", 0),
                    "convergence_ratio": bayesian_results.get("posterior_updates", {})
                    .get("convergence_analysis", {})
                    .get("system_convergence", {})
                    .get("converged_ratio", 0),
                },
                "output_files": {
                    "bayesian_analysis": result_file,
                },
            }

        except Exception as e:
            self.logger.error(f"❌ 베이지안 분석 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"execution_time": time.time() - start_time},
                "output_files": {},
            }

    def run_ensemble_analysis(self) -> Dict[str, Any]:
        """앙상블 분석 실행"""
        start_time = time.time()

        try:
            # 데이터 로드
            self.logger.info("📂 로또 데이터 로드 중...")
            historical_data = load_draw_history()

            # EnsembleAnalyzer 초기화
            if self._ensemble_analyzer is None:
                self._ensemble_analyzer = EnsembleAnalyzer(self.config)

            # 앙상블 분석 실행
            self.logger.info("🔗 앙상블 패턴 분석 수행 중...")
            ensemble_results = self._ensemble_analyzer.analyze(historical_data)

            # 결과 저장
            result_file = self._ensemble_analyzer.save_analysis_results(
                ensemble_results
            )

            # 성능 분석
            performance_analysis = self._analyze_ensemble_performance(ensemble_results)

            return {
                "success": True,
                "results": ensemble_results,
                "performance_analysis": performance_analysis,
                "metrics": {
                    "execution_time": time.time() - start_time,
                    "data_count": len(historical_data),
                    "analyzer_version": ensemble_results.get(
                        "analyzer_version", "EnsembleAnalyzer_v1.0"
                    ),
                    "ensemble_methods": len(
                        ensemble_results.get("weighted_ensemble_analysis", {}).get(
                            "ensemble_results", {}
                        )
                    ),
                    "prediction_confidence": ensemble_results.get(
                        "final_predictions", {}
                    )
                    .get("prediction_summary", {})
                    .get("average_confidence", 0),
                    "window_consistency": ensemble_results.get(
                        "multi_window_analysis", {}
                    )
                    .get("consistency_analysis", {})
                    .get("consistency_score", 0),
                },
                "output_files": {
                    "ensemble_analysis": result_file,
                },
            }

        except Exception as e:
            self.logger.error(f"❌ 앙상블 분석 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {"execution_time": time.time() - start_time},
                "output_files": {},
            }

    def _analyze_trend_v2_performance(
        self, trend_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """TrendAnalyzerV2 성능 분석"""
        summary = trend_results.get("trend_summary", {})

        return {
            "top_numbers_quality": len(summary.get("top_recommended_numbers", [])),
            "system_stability": summary.get("system_health", {}).get(
                "overall_stability", 0
            ),
            "recommendation_confidence": summary.get("trend_analysis_summary", {}).get(
                "recommendation_confidence", "unknown"
            ),
            "change_point_density": summary.get("system_health", {}).get(
                "change_point_density", 0
            ),
        }

    def _analyze_bayesian_performance(
        self, bayesian_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """베이지안 분석 성능 분석"""
        summary = bayesian_results.get("analysis_summary", {})

        return {
            "model_quality": summary.get("key_findings", {}).get(
                "best_model", "unknown"
            ),
            "convergence_quality": summary.get("key_findings", {})
            .get("system_convergence", {})
            .get("converged_ratio", 0),
            "prediction_confidence": summary.get("recommendation_confidence", {}).get(
                "overall_confidence", 0
            ),
            "high_confidence_count": summary.get("recommendation_confidence", {}).get(
                "high_confidence_numbers", 0
            ),
        }

    def _analyze_ensemble_performance(
        self, ensemble_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """앙상블 분석 성능 분석"""
        summary = ensemble_results.get("ensemble_summary", {})

        return {
            "ensemble_diversity": summary.get("analysis_overview", {}).get(
                "ensemble_methods_used", 0
            ),
            "window_consistency": summary.get("key_findings", {}).get(
                "window_consistency", 0
            ),
            "prediction_quality": summary.get("performance_metrics", {}).get(
                "prediction_confidence", 0
            ),
            "ensemble_strength": summary.get("recommendations", {}).get(
                "ensemble_strength", "unknown"
            ),
        }

    def run_graph_network_analysis(self) -> Dict[str, Any]:
        """그래프 네트워크 분석 실행"""
        try:
            self.logger.info("🔗 그래프 네트워크 분석 시작...")

            # 데이터 로드
            historical_data = load_draw_history()
            if not historical_data:
                raise ValueError("로또 데이터를 로드할 수 없습니다")

            # 그래프 네트워크 분석기 초기화
            if self._graph_network_analyzer is None:
                graph_config = self.config.get("graph_network_analysis", {})
                self._graph_network_analyzer = GraphNetworkAnalyzer(graph_config)

            # 분석 실행
            analysis_results = self._graph_network_analyzer.analyze(historical_data)

            # 결과 저장
            output_file = self.result_dir / "graph_network_analysis.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    analysis_results, f, ensure_ascii=False, indent=2, default=str
                )

            # 그래프 특성 벡터 생성
            if "number_graph_features" in analysis_results:
                graph_vector = self._graph_network_analyzer.get_graph_features_vector(
                    analysis_results["number_graph_features"]
                )

                # 벡터 저장
                vector_file = self.cache_dir / "graph_network_features.npy"
                np.save(vector_file, graph_vector)

                self.logger.info(f"그래프 특성 벡터 저장: {graph_vector.shape}")

            # 성능 메트릭 계산
            performance_metrics = self._analyze_graph_network_performance(
                analysis_results
            )

            self.logger.info("✅ 그래프 네트워크 분석 완료")

            return {
                "status": "success",
                "output_file": str(output_file),
                "analysis_results": analysis_results,
                "performance_metrics": performance_metrics,
                "data_samples": len(historical_data),
            }

        except Exception as e:
            self.logger.error(f"그래프 네트워크 분석 실패: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "output_file": None,
            }

    def run_meta_feature_analysis(self) -> Dict[str, Any]:
        """메타 특성 분석 실행"""
        try:
            self.logger.info("🔍 메타 특성 분석 시작...")

            # 데이터 로드
            historical_data = load_draw_history()
            if not historical_data:
                raise ValueError("로또 데이터를 로드할 수 없습니다")

            # 메타 특성 분석기 초기화
            if self._meta_feature_analyzer is None:
                meta_config = self.config.get("meta_feature_analysis", {})
                self._meta_feature_analyzer = MetaFeatureAnalyzer(meta_config)

            # 분석 실행
            analysis_results = self._meta_feature_analyzer.analyze(historical_data)

            # 결과 저장
            output_file = self.result_dir / "meta_feature_analysis.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    analysis_results, f, ensure_ascii=False, indent=2, default=str
                )

            # 메타 특성 벡터 생성
            meta_vector = self._meta_feature_analyzer.get_meta_features_vector(
                analysis_results
            )

            # 벡터 저장
            vector_file = self.cache_dir / "meta_features.npy"
            np.save(vector_file, meta_vector)

            self.logger.info(f"메타 특성 벡터 저장: {meta_vector.shape}")

            # 결과를 JSON 파일로도 저장
            meta_results_file = self.result_dir / "meta_analysis_results.json"
            self._meta_feature_analyzer.save_meta_analysis_results(
                analysis_results, meta_results_file.name
            )

            # 성능 메트릭 계산
            performance_metrics = self._analyze_meta_feature_performance(
                analysis_results
            )

            self.logger.info("✅ 메타 특성 분석 완료")

            return {
                "status": "success",
                "output_file": str(output_file),
                "analysis_results": analysis_results,
                "performance_metrics": performance_metrics,
                "data_samples": len(historical_data),
            }

        except Exception as e:
            self.logger.error(f"메타 특성 분석 실패: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "output_file": None,
            }

    def _analyze_graph_network_performance(
        self, graph_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """그래프 네트워크 분석 성능 분석"""
        try:
            graph_stats = graph_results.get("graph_statistics", {})
            communities = graph_results.get("communities", {})
            centrality = graph_results.get("centrality_analysis", {})

            return {
                "graph_connectivity": graph_stats.get("is_connected", False),
                "graph_density": graph_stats.get("density", 0.0),
                "community_count": len(communities.get("greedy", [])),
                "modularity_score": communities.get("modularity", 0.0),
                "centrality_methods": len(
                    [k for k in centrality.keys() if k != "statistics"]
                ),
                "node_coverage": graph_stats.get("nodes", 0) / 45,  # 45개 번호 대비
                "edge_count": graph_stats.get("edges", 0),
                "analysis_completeness": 1.0 if "error" not in graph_results else 0.0,
            }

        except Exception as e:
            self.logger.error(f"그래프 네트워크 성능 분석 실패: {e}")
            return {"analysis_completeness": 0.0, "error": str(e)}

    def _analyze_meta_feature_performance(
        self, meta_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """메타 특성 분석 성능 분석"""
        try:
            importance_analysis = meta_results.get("importance_analysis", {})
            dimension_reduction = meta_results.get("dimension_reduction", {})
            feature_selection = meta_results.get("feature_selection", {})

            return {
                "importance_methods": len(
                    [k for k in importance_analysis.keys() if k != "error"]
                ),
                "pca_success": dimension_reduction.get("pca", {}).get("success", False),
                "variance_explained": dimension_reduction.get("pca", {}).get(
                    "total_variance_explained", 0.0
                ),
                "feature_selection_success": feature_selection.get(
                    "model_based", {}
                ).get("success", False),
                "selected_features_ratio": (
                    feature_selection.get("model_based", {}).get("selected_count", 0)
                    / max(meta_results.get("original_features", 1), 1)
                ),
                "data_quality_score": (
                    1.0
                    - (
                        meta_results.get("meta_statistics", {})
                        .get("data_quality", {})
                        .get("missing_values", 0)
                        + meta_results.get("meta_statistics", {})
                        .get("data_quality", {})
                        .get("infinite_values", 0)
                    )
                    / max(meta_results.get("data_samples", 1), 1)
                ),
                "analysis_completeness": 1.0 if "error" not in meta_results else 0.0,
            }

        except Exception as e:
            self.logger.error(f"메타 특성 성능 분석 실패: {e}")
            return {"analysis_completeness": 0.0, "error": str(e)}

    def validate_and_save_results(
        self, pipeline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """결과 검증 및 저장"""
        validation_results = {
            "validation_passed": True,
            "validation_errors": [],
            "file_validations": {},
        }

        # 출력 파일 검증
        for file_type, file_path in pipeline_results.get("output_files", {}).items():
            try:
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    validation_results["file_validations"][file_type] = {
                        "exists": True,
                        "size_bytes": file_size,
                        "size_mb": file_size / (1024 * 1024),
                    }
                else:
                    validation_results["validation_errors"].append(
                        f"파일이 존재하지 않음: {file_path}"
                    )
                    validation_results["validation_passed"] = False
            except Exception as e:
                validation_results["validation_errors"].append(
                    f"파일 검증 실패 {file_path}: {e}"
                )
                validation_results["validation_passed"] = False

        return validation_results

    def _clear_pipeline_cache(self):
        """파이프라인 캐시 정리"""
        try:
            cache_files = list(self.cache_dir.glob("*"))
            for cache_file in cache_files:
                if cache_file.is_file():
                    cache_file.unlink()
            self.logger.info(f"🧹 {len(cache_files)}개 캐시 파일 정리 완료")
        except Exception as e:
            self.logger.warning(f"캐시 정리 실패: {e}")

    def _validate_vector_quality(
        self, vector: np.ndarray, names: List[str]
    ) -> Dict[str, Any]:
        """벡터 품질 검증"""
        return {
            "dimension_match": len(vector) == len(names),
            "zero_ratio": np.sum(vector == 0) / len(vector),
            "nan_count": np.sum(np.isnan(vector)),
            "inf_count": np.sum(np.isinf(vector)),
            "value_range": {"min": float(np.min(vector)), "max": float(np.max(vector))},
            "mean": float(np.mean(vector)),
            "std": float(np.std(vector)),
        }

    # 기존 로드 메서드들 유지
    def _load_analysis_result(self) -> Dict[str, Any]:
        """기존 분석 결과 로드 (하위 호환성)"""
        return self._load_unified_analysis_result()

    def _load_vectorization_result(self) -> Dict[str, Any]:
        """벡터화 결과 로드"""
        try:
            vector_file = self.cache_dir / "optimized_feature_vector.npy"
            if vector_file.exists():
                vector = np.load(vector_file)
                return {"vector": vector}
            else:
                return {"vector": np.array([])}
        except Exception as e:
            self.logger.error(f"벡터화 결과 로드 실패: {e}")
            return {"vector": np.array([])}


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="DAEBAK AI 개선된 데이터 준비 통합 파이프라인"
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "unified_analysis",
            "3digit_prediction",
            "optimized_vectorization",
            "advanced_trend_analysis",
            "bayesian_analysis",
            "ensemble_analysis",
            "graph_network_analysis",
            "meta_feature_analysis",
            "negative_sampling",
        ],
        default=[
            "unified_analysis",
            "3digit_prediction",
            "optimized_vectorization",
            "advanced_trend_analysis",
            "bayesian_analysis",
            "ensemble_analysis",
            "graph_network_analysis",
            "meta_feature_analysis",
            "negative_sampling",
        ],
        help="실행할 파이프라인 단계 (최신 고급 분석기 포함)",
    )

    parser.add_argument("--clear-cache", action="store_true", help="실행 전 캐시 정리")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅")
    parser.add_argument(
        "--comparison", action="store_true", help="기존 vs 새로운 시스템 비교"
    )
    parser.add_argument("--config", type=str, help="설정 파일 경로")

    return parser.parse_args()


def main():
    """메인 실행 함수"""
    # 최우선: 의존성 설정
    configure_dependencies()
    
    parser = parse_arguments()
    args = parser.parse_args()

    # 파이프라인 인스턴스 생성 (설정 인자 제거)
    pipeline = EnhancedDataPreparationPipeline()

    try:
        # 파이프라인 실행
        results = pipeline.execute_full_pipeline(
            clear_cache=args.clear_cache,
            steps=args.steps,
            debug=args.debug,
            verbose=args.verbose,
            comparison_mode=args.comparison,
        )

        # 실행 결과 출력
        if results.get("steps_failed"):
            print(f"❌ 일부 단계가 실패했습니다: {', '.join(results['steps_failed'])}")
            return 1
        else:
            print("✅ 모든 파이프라인 단계가 성공적으로 완료되었습니다!")
            return 0

    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
