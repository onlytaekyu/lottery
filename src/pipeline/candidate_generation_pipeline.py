#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML 기반 로또 번호 후보 생성 파이프라인
"""

import os
import sys
import time
import random
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.core.ml_candidate_generator import MLCandidateGenerator
from src.utils.data_loader import load_draw_history
from src.utils.unified_config import load_config
from src.utils.error_handler_refactored import get_logger
from src.utils.unified_performance import get_profiler
from src.utils.pattern_filter import get_pattern_filter

# 로거 설정
logger = get_logger(__name__)


def run_ml_candidate_generation(
    candidates_count: int = 200,
    enable_structured: bool = True,
    enable_roi: bool = True,
    enable_model: bool = True,
    save_results: bool = True,
) -> bool:
    """
    ML 기반 초기 후보 생성 실행

    Args:
        candidates_count: 생성할 후보 조합 개수
        enable_structured: 구조화된 후보 생성 활성화 여부
        enable_roi: ROI 기반 후보 생성 활성화 여부
        enable_model: 모델 기반 후보 생성 활성화 여부
        save_results: 결과를 파일로 저장할지 여부

    Returns:
        bool: 작업 성공 여부
    """
    start_time = time.time()

    # 랜덤 시드 설정 (재현성 보장)
    random.seed(42)
    np.random.seed(42)

    # 설정 로드
    config = load_config()

    # 프로파일러 초기화
    profiler = get_profiler()

    try:
        # 1. 데이터 로드
        with profiler.profile("데이터 로드"):
            logger.info("2단계: 과거 당첨 번호 데이터 로드 중...")
            historical_data = load_draw_history()

            if not historical_data:
                logger.error("당첨 번호 데이터를 로드할 수 없습니다.")
                return False

            logger.info(f"데이터 로드 완료: {len(historical_data)}개 회차")

            # 캐시 디렉토리 확인
            cache_dir = Path(config.safe_get("paths.cache_dir", "data/cache"))
            if not cache_dir.exists():
                logger.warning(f"캐시 디렉토리를 찾을 수 없습니다: {cache_dir}")
                logger.warning("먼저 1단계(데이터 분석 및 전처리)를 실행해야 합니다.")
                cache_dir.mkdir(parents=True, exist_ok=True)

            # 예측 결과 디렉토리 생성
            prediction_dir = Path(
                config.safe_get("recommendation_output.path", "data/predictions")
            )
            prediction_dir.mkdir(parents=True, exist_ok=True)

        # 2. 설정 업데이트 (명령줄 인자 반영)
        with profiler.profile("설정 업데이트"):
            # 기존 설정을 딕셔너리로 변환
            config_dict = config.to_dict()

            # generation 섹션이 없으면 생성
            if "generation" not in config_dict:
                config_dict["generation"] = {}

            # 명령줄 인자로 설정 업데이트
            config_dict["generation"][
                "enable_structured_generation"
            ] = enable_structured
            config_dict["generation"]["enable_roi_guided_generation"] = enable_roi
            config_dict["generation"]["enable_model_guided_generation"] = enable_model
            config_dict["statistical_model"] = config_dict.get("statistical_model", {})
            config_dict["statistical_model"]["candidate_count"] = candidates_count

            # 업데이트된 설정으로 ConfigProxy 재생성
            config = load_config()
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        config.update(f"{key}.{subkey}", subvalue)
                else:
                    config.update(key, value)

            logger.info(f"설정 업데이트 완료 (후보 수: {candidates_count})")

            # 생성 비율 로깅
            structured_ratio = config.safe_get("generation.structured_ratio", 0.3)
            roi_ratio = config.safe_get("generation.roi_ratio", 0.3)
            model_ratio = config.safe_get("generation.model_ratio", 0.4)

            logger.info(
                f"후보 생성 비율: 구조화={structured_ratio:.1f}, ROI={roi_ratio:.1f}, 모델={model_ratio:.1f}"
            )

        # 3. ML 후보 생성기 초기화
        with profiler.profile("ML 후보 생성기 초기화"):
            logger.info("ML 후보 생성기 초기화 중...")
            generator = MLCandidateGenerator(config)

            # ML 모델 로드
            lgbm_loaded, xgb_loaded = generator.load_ml_models()
            logger.info(f"모델 로드 상태: LightGBM={lgbm_loaded}, XGBoost={xgb_loaded}")

            if not (lgbm_loaded or xgb_loaded):
                logger.warning(
                    "ML 모델을 로드할 수 없습니다. 점수 예측 성능이 저하될 수 있습니다."
                )

        # 4. 후보 생성 및 점수 부여
        with profiler.profile("후보 생성 및 점수 부여"):
            logger.info(f"ML 기반 후보 {candidates_count}개 생성 중...")

            # 후보 생성 (MLCandidateGenerator 사용)
            scored_candidates = generator.generate_candidates(historical_data)

            if not scored_candidates:
                logger.error("후보 생성에 실패했습니다.")
                return False

            logger.info(f"후보 생성 완료: {len(scored_candidates)}개")

            # 생성 소스별 통계
            source_stats = {}
            for candidate in scored_candidates:
                source = candidate.get("source", "unknown")
                source_stats[source] = source_stats.get(source, 0) + 1

            for source, count in source_stats.items():
                logger.info(
                    f"- {source}: {count}개 ({count/len(scored_candidates)*100:.1f}%)"
                )

        # 5. 결과 분석 및 정렬
        with profiler.profile("결과 분석 및 정렬"):
            # 최종 점수 기준으로 내림차순 정렬
            scored_candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            # 상위 10개 로깅
            logger.info("상위 10개 후보:")
            for i, candidate in enumerate(scored_candidates[:10]):
                logger.info(
                    f"#{i+1}: {candidate['numbers']} "
                    f"(점수: {candidate.get('final_score', 0):.4f}, "
                    f"위험도: {candidate.get('risk_score', 0):.4f})"
                )

            # 점수 분포 분석
            scores = [c.get("final_score", 0) for c in scored_candidates]
            risk_scores = [c.get("risk_score", 0) for c in scored_candidates]

            logger.info(
                f"점수 범위: {min(scores):.4f} ~ {max(scores):.4f} (평균: {sum(scores)/len(scores):.4f})"
            )
            logger.info(
                f"위험도 범위: {min(risk_scores):.4f} ~ {max(risk_scores):.4f} (평균: {sum(risk_scores)/len(risk_scores):.4f})"
            )

        # 6. 결과 저장
        if save_results:
            with profiler.profile("결과 저장"):
                # 현재 시간을 파일명에 포함
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 점수가 매겨진 후보 풀 저장
                candidates_file = (
                    prediction_dir / f"ml_scored_candidates_{timestamp}.json"
                )
                cache_file = cache_dir / "latest_ml_candidates.json"

                # 저장할 데이터 준비
                result_data = {
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_candidates": len(scored_candidates),
                    "models_used": {
                        "lightgbm": lgbm_loaded,
                        "xgboost": xgb_loaded,
                    },
                    "source_stats": source_stats,
                    "candidates": scored_candidates,
                }

                # JSON으로 저장
                with open(candidates_file, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)

                # 캐시 파일로도 저장
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)

                logger.info(f"결과 저장 완료: {candidates_file}")
                logger.info(f"캐시 저장 완료: {cache_file}")

        # 실행 시간 기록
        execution_time = time.time() - start_time
        logger.info(f"ML 기반 초기 후보 생성 완료 (소요시간: {execution_time:.2f}초)")

        # 성능 보고서 생성
        if config.safe_get("reporting.enable_performance_report", True):
            report_dir = Path(config.safe_get("reporting.report_dir", "logs/reports"))
            report_dir.mkdir(parents=True, exist_ok=True)

            report_data = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "candidate_count": len(scored_candidates),
                "structured_ratio": structured_ratio,
                "roi_ratio": roi_ratio,
                "model_ratio": model_ratio,
                "score_range": [min(scores), max(scores)],
                "risk_score_range": [min(risk_scores), max(risk_scores)],
                "execution_time_sec": round(execution_time, 2),
            }

            report_path = report_dir / "ml_candidate_generation_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            logger.info(f"성능 리포트 저장 완료: {report_path}")

        # 프로파일링 결과 출력
        profiler.log_report()

        return True

    except Exception as e:
        logger.error(f"ML 기반 초기 후보 생성 중 오류 발생: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())
        return False
