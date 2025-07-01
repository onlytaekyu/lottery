#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAEBAK AI 로또 데이터 분석 및 최적화된 전처리 실행 스크립트

이 스크립트는 새로 구현된 고급 전처리 파이프라인을 통합하여 실행합니다:
- Phase 1: 즉시 적용 가능한 핵심 최적화 (Smart Feature Selection, Outlier Handling)
- Phase 2: 성능 향상을 위한 고급 특성 엔지니어링 (Feature Interactions, Meta Features)
- 모델별 특화 전처리 (LightGBM, AutoEncoder, TCN, RandomForest)
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 설정
os.environ["PYTHONPATH"] = str(project_root)

from src.utils.unified_logging import get_logger
from src.utils.unified_config import get_config
from src.utils.memory_manager import get_memory_manager
from src.utils.unified_performance import performance_monitor
from src.utils.cache_paths import get_cache_dir

# 새로 구현된 전처리 파이프라인들
from src.pipeline.preprocessing_manager import PreprocessingManager
from src.pipeline.advanced_preprocessing_pipeline import AdvancedPreprocessingPipeline
from src.pipeline.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.pipeline.model_specific_preprocessors import create_model_preprocessor

# 기존 분석 파이프라인
from src.pipeline.optimized_data_analysis_pipeline import (
    run_optimized_data_analysis,
    clear_analysis_cache,
)

logger = get_logger(__name__)


class IntegratedDataAnalysisRunner:
    """통합 데이터 분석 및 전처리 실행기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = get_config("main") if config is None else get_config("main")
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()

        # 전처리 매니저 초기화
        self.preprocessing_manager = PreprocessingManager(config)

        # 결과 저장 경로들
        self.cache_dir = get_cache_dir()
        self.result_dir = Path("data/result/analysis")
        self.performance_dir = Path("data/result/performance_reports")

        # 디렉토리 생성
        for directory in [self.cache_dir, self.result_dir, self.performance_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("통합 데이터 분석 실행기 초기화 완료")

    def run_complete_analysis(
        self,
        clear_cache: bool = False,
        enable_preprocessing: bool = True,
        preprocessing_phase: str = "both",  # "phase1", "phase2", "both"
        target_models: List[str] = None,
    ) -> bool:
        """
        완전한 데이터 분석 및 전처리 실행

        Args:
            clear_cache: 캐시 삭제 여부
            enable_preprocessing: 전처리 활성화 여부
            preprocessing_phase: 전처리 단계 ("phase1", "phase2", "both")
            target_models: 대상 모델 리스트

        Returns:
            bool: 성공 여부
        """
        start_time = datetime.now()
        self.logger.info("=" * 80)
        self.logger.info("🚀 DAEBAK AI 통합 데이터 분석 및 전처리 시작")
        self.logger.info("=" * 80)

        if target_models is None:
            target_models = ["lightgbm", "autoencoder", "tcn", "random_forest"]

        try:
            # 1. 캐시 정리 (선택적)
            if clear_cache:
                self.logger.info("🧹 캐시 정리 중...")
                clear_analysis_cache()

            # 2. 기본 데이터 분석 실행
            self.logger.info("📊 기본 데이터 분석 실행 중...")
            basic_success = run_optimized_data_analysis()

            if not basic_success:
                self.logger.error("기본 데이터 분석 실패")
                return False

            # 3. 고급 전처리 실행 (선택적)
            if enable_preprocessing:
                preprocessing_success = self._run_advanced_preprocessing(
                    preprocessing_phase, target_models
                )

                if not preprocessing_success:
                    self.logger.error("고급 전처리 실패")
                    return False

            # 4. 결과 검증 및 리포트 생성
            validation_success = self._validate_and_report_results()

            if not validation_success:
                self.logger.warning("결과 검증에서 일부 문제 발견")

            # 5. 성능 통계 출력
            total_time = (datetime.now() - start_time).total_seconds()
            self._print_performance_summary(total_time)

            self.logger.info("=" * 80)
            self.logger.info("✅ 통합 데이터 분석 및 전처리 완료")
            self.logger.info("=" * 80)

            return True

        except Exception as e:
            self.logger.error(f"실행 중 오류 발생: {e}")
            return False

    def _run_advanced_preprocessing(self, phase: str, target_models: List[str]) -> bool:
        """고급 전처리 실행"""
        self.logger.info(f"🔧 고급 전처리 실행 중... (Phase: {phase})")

        try:
            # 기본 특성 벡터 로드
            feature_vector_path = self.cache_dir / "feature_vector_full.npy"
            feature_names_path = self.cache_dir / "feature_vector_full.names.json"

            if not feature_vector_path.exists():
                self.logger.error(f"특성 벡터 파일이 없습니다: {feature_vector_path}")
                return False

            # 데이터 로드
            X = np.load(feature_vector_path)

            import json

            if feature_names_path.exists():
                with open(feature_names_path, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            self.logger.info(
                f"원본 데이터 로드: {X.shape}, 특성명: {len(feature_names)}개"
            )

            # 전처리 권장사항 분석
            recommendations = (
                self.preprocessing_manager.get_preprocessing_recommendations(
                    X, feature_names
                )
            )
            self._save_preprocessing_recommendations(recommendations)

            # 모델별 전처리 실행
            preprocessing_results = {}

            for model_type in target_models:
                self.logger.info(f"📋 {model_type} 모델 전처리 중...")

                try:
                    result = self.preprocessing_manager.preprocess_for_model(
                        X=X,
                        feature_names=feature_names,
                        model_type=model_type,
                        y=None,  # 비지도 학습
                        phase=phase,
                        use_cache=True,
                    )

                    preprocessing_results[model_type] = result

                    # 모델별 결과 저장
                    self._save_model_preprocessing_result(model_type, result)

                    self.logger.info(
                        f"✓ {model_type}: {X.shape} → {result.X_processed.shape} "
                        f"({result.processing_time:.2f}초)"
                    )

                except Exception as e:
                    self.logger.error(f"❌ {model_type} 전처리 실패: {e}")
                    continue

            # 전처리 결과 종합 리포트 생성
            self._generate_preprocessing_report(preprocessing_results, phase)

            self.logger.info(
                f"🎯 고급 전처리 완료: {len(preprocessing_results)}개 모델"
            )
            return len(preprocessing_results) > 0

        except Exception as e:
            self.logger.error(f"고급 전처리 실행 중 오류: {e}")
            return False

    def _save_model_preprocessing_result(self, model_type: str, result) -> None:
        """모델별 전처리 결과 저장"""
        try:
            # 전처리된 특성 벡터 저장
            model_cache_dir = self.cache_dir / "preprocessed" / model_type
            model_cache_dir.mkdir(parents=True, exist_ok=True)

            # 특성 벡터 저장
            vector_path = model_cache_dir / "feature_vector_preprocessed.npy"
            np.save(vector_path, result.X_processed)

            # 특성 이름 저장
            names_path = model_cache_dir / "feature_names_preprocessed.json"
            import json

            with open(names_path, "w", encoding="utf-8") as f:
                json.dump(result.feature_names, f, ensure_ascii=False, indent=2)

            # 전처리 통계 저장
            stats_path = model_cache_dir / "preprocessing_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(result.preprocessing_stats, f, ensure_ascii=False, indent=2)

            self.logger.info(f"📁 {model_type} 전처리 결과 저장 완료")

        except Exception as e:
            self.logger.warning(f"전처리 결과 저장 실패 ({model_type}): {e}")

    def _save_preprocessing_recommendations(
        self, recommendations: Dict[str, Any]
    ) -> None:
        """전처리 권장사항 저장"""
        try:
            recommendations_path = (
                self.result_dir / "preprocessing_recommendations.json"
            )

            import json

            with open(recommendations_path, "w", encoding="utf-8") as f:
                json.dump(recommendations, f, ensure_ascii=False, indent=2)

            self.logger.info(f"📋 전처리 권장사항 저장: {recommendations_path}")

        except Exception as e:
            self.logger.warning(f"전처리 권장사항 저장 실패: {e}")

    def _generate_preprocessing_report(
        self, results: Dict[str, Any], phase: str
    ) -> None:
        """전처리 종합 리포트 생성"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "phase": phase,
                "total_models": len(results),
                "successful_models": len(results),
                "models": {},
                "summary": {
                    "average_processing_time": 0.0,
                    "average_feature_reduction": 0.0,
                    "total_cache_hits": self.preprocessing_manager.performance_stats[
                        "cache_hits"
                    ],
                    "total_processed": self.preprocessing_manager.performance_stats[
                        "total_processed"
                    ],
                },
            }

            processing_times = []
            feature_reductions = []

            for model_type, result in results.items():
                model_report = {
                    "original_shape": result.preprocessing_stats.get(
                        "original_shape", [0, 0]
                    ),
                    "final_shape": result.X_processed.shape,
                    "processing_time": result.processing_time,
                    "feature_reduction_ratio": result.preprocessing_stats.get(
                        "total_feature_reduction", 0.0
                    ),
                    "cache_key": result.cache_key,
                    "feature_count": len(result.feature_names),
                }

                report["models"][model_type] = model_report
                processing_times.append(result.processing_time)

                reduction_ratio = result.preprocessing_stats.get(
                    "total_feature_reduction", 0.0
                )
                feature_reductions.append(reduction_ratio)

            # 평균 통계 계산
            if processing_times:
                report["summary"]["average_processing_time"] = np.mean(processing_times)
            if feature_reductions:
                report["summary"]["average_feature_reduction"] = np.mean(
                    feature_reductions
                )

            # 리포트 저장
            report_path = (
                self.performance_dir
                / f"preprocessing_report_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            import json

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            self.logger.info(f"📊 전처리 리포트 생성: {report_path}")

        except Exception as e:
            self.logger.warning(f"전처리 리포트 생성 실패: {e}")

    def _validate_and_report_results(self) -> bool:
        """결과 검증 및 리포트"""
        self.logger.info("🔍 결과 검증 중...")

        validation_results = {
            "basic_analysis": False,
            "preprocessing_results": {},
            "file_checks": {},
        }

        # 기본 분석 결과 파일 확인
        basic_files = [
            "data/cache/feature_vector_full.npy",
            "data/cache/feature_vector_full.names.json",
            "data/result/analysis/optimized_analysis_result.json",
        ]

        for file_path in basic_files:
            exists = os.path.exists(file_path)
            validation_results["file_checks"][file_path] = {
                "exists": exists,
                "size": os.path.getsize(file_path) if exists else 0,
            }

        validation_results["basic_analysis"] = all(
            result["exists"] for result in validation_results["file_checks"].values()
        )

        # 전처리 결과 파일 확인
        preprocessed_dir = self.cache_dir / "preprocessed"
        if preprocessed_dir.exists():
            for model_dir in preprocessed_dir.iterdir():
                if model_dir.is_dir():
                    model_type = model_dir.name
                    model_files = [
                        model_dir / "feature_vector_preprocessed.npy",
                        model_dir / "feature_names_preprocessed.json",
                        model_dir / "preprocessing_stats.json",
                    ]

                    model_validation = {
                        "files_exist": all(f.exists() for f in model_files),
                        "file_sizes": {
                            f.name: f.stat().st_size if f.exists() else 0
                            for f in model_files
                        },
                    }

                    validation_results["preprocessing_results"][
                        model_type
                    ] = model_validation

        # 검증 결과 저장
        validation_path = self.result_dir / "validation_results.json"
        import json

        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)

        # 결과 출력
        self.logger.info("📋 검증 결과:")
        self.logger.info(
            f"  기본 분석: {'✓' if validation_results['basic_analysis'] else '✗'}"
        )

        for model_type, result in validation_results["preprocessing_results"].items():
            status = "✓" if result["files_exist"] else "✗"
            self.logger.info(f"  {model_type} 전처리: {status}")

        return validation_results["basic_analysis"]

    def _print_performance_summary(self, total_time: float) -> None:
        """성능 요약 출력"""
        self.logger.info("📈 성능 요약:")
        self.logger.info(f"  총 실행 시간: {total_time:.2f}초")

        # 전처리 매니저 통계
        stats = self.preprocessing_manager.performance_stats
        self.logger.info(f"  전처리 통계:")
        self.logger.info(f"    - 총 처리 건수: {stats['total_processed']}")
        self.logger.info(f"    - 캐시 히트: {stats['cache_hits']}")

        if stats["processing_times"]:
            avg_time = np.mean(stats["processing_times"])
            self.logger.info(f"    - 평균 처리 시간: {avg_time:.2f}초")

        if stats["feature_reduction_ratios"]:
            avg_reduction = np.mean(stats["feature_reduction_ratios"])
            self.logger.info(f"    - 평균 특성 감소율: {avg_reduction:.2%}")

        # 메모리 사용량
        memory_info = self.memory_manager.get_memory_info()
        self.logger.info(
            f"  메모리 사용량: {memory_info.get('current_usage', 0):.2f}MB"
        )


def main():
    """메인 실행 함수"""
    logger.info("=" * 80)
    logger.info("🚀 DAEBAK AI 통합 데이터 분석 및 최적화된 전처리 시작")
    logger.info("=" * 80)

    try:
        # 통합 실행기 초기화
        runner = IntegratedDataAnalysisRunner()

        # 실행 옵션 설정
        execution_options = {
            "clear_cache": False,  # True로 설정하면 캐시 삭제 후 실행
            "enable_preprocessing": True,  # 고급 전처리 활성화
            "preprocessing_phase": "both",  # "phase1", "phase2", "both"
            "target_models": ["lightgbm", "autoencoder", "tcn", "random_forest"],
        }

        logger.info("⚙️ 실행 옵션:")
        for key, value in execution_options.items():
            logger.info(f"  {key}: {value}")

        # 통합 분석 실행
        success = runner.run_complete_analysis(**execution_options)

        if success:
            logger.info("=" * 80)
            logger.info("✅ 모든 작업이 성공적으로 완료되었습니다!")
            logger.info("=" * 80)

            logger.info("📁 생성된 주요 파일들:")

            # 기본 분석 결과
            basic_files = [
                "data/cache/feature_vector_full.npy",
                "data/cache/feature_vector_full.names.json",
                "data/result/analysis/optimized_analysis_result.json",
            ]

            for file_path in basic_files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.info(f"  ✓ {file_path} ({file_size:,} bytes)")
                else:
                    logger.warning(f"  ✗ {file_path} (파일 없음)")

            # 전처리 결과 (각 모델별)
            preprocessed_dir = Path("data/cache/preprocessed")
            if preprocessed_dir.exists():
                logger.info("  📋 모델별 전처리 결과:")
                for model_dir in preprocessed_dir.iterdir():
                    if model_dir.is_dir():
                        model_type = model_dir.name
                        vector_file = model_dir / "feature_vector_preprocessed.npy"
                        if vector_file.exists():
                            vector_shape = np.load(vector_file).shape
                            logger.info(f"    ✓ {model_type}: {vector_shape}")

            return True
        else:
            logger.error("❌ 작업 실행 중 오류가 발생했습니다.")
            return False

    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        import traceback

        logger.error(f"상세 오류: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
