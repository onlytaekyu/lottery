#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터 분석 및 전처리 실행 모듈

이 모듈은 완성된 데이터 분석 파이프라인을 실행합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.error_handler_refactored import get_logger
from src.pipeline.optimized_data_analysis_pipeline import run_optimized_data_analysis

# 로거 설정
logger = get_logger(__name__)


def main():
    """메인 실행 함수"""
    try:
        logger.info("=" * 60)
        logger.info("🚀 DAEBAK AI 로또 데이터 분석 및 전처리 시작")
        logger.info("=" * 60)

        # 최적화된 데이터 분석 파이프라인 실행
        success = run_optimized_data_analysis()

        if success:
            logger.info("=" * 60)
            logger.info("✅ 데이터 분석 및 전처리 완료")
            logger.info("=" * 60)

            # 결과 파일 확인
            result_files = [
                "data/cache/feature_vector_full.npy",
                "data/cache/feature_vector_full.names.json",
                "data/result/analysis/optimized_analysis_result.json",
                "data/result/performance_reports/optimized_data_analysis_performance_report.json",
            ]

            logger.info("📁 생성된 파일들:")
            for file_path in result_files:
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    logger.info(f"  ✓ {file_path} ({file_size:,} bytes)")
                else:
                    logger.warning(f"  ✗ {file_path} (파일 없음)")

            return 0
        else:
            logger.error("=" * 60)
            logger.error("❌ 데이터 분석 및 전처리 실패")
            logger.error("=" * 60)
            return 1

    except Exception as e:
        logger.error(f"실행 중 예외 발생: {e}")
        import traceback

        logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
