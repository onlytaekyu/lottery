#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAEBAK AI 로또 데이터 분석 및 전처리 실행 스크립트

이 스크립트는 최적화된 데이터 분석 파이프라인을 실행합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 설정
os.environ["PYTHONPATH"] = str(project_root)

from src.utils.unified_logging import get_logger
from src.pipeline.optimized_data_analysis_pipeline import (
    run_optimized_data_analysis,
    clear_analysis_cache,
)


def main():
    """메인 실행 함수"""
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("🚀 DAEBAK AI 로또 데이터 분석 및 전처리 시작")
    logger.info("=" * 60)

    try:
        # 🔧 캐시 새로고침 옵션 (실제 분석 실행을 위해)
        # clear_analysis_cache()  # 주석 해제하면 캐시 삭제 후 실제 분석 실행

        # 최적화된 데이터 분석 파이프라인 실행
        success = run_optimized_data_analysis()

        if success:
            logger.info("=" * 60)
            logger.info("✅ 데이터 분석 및 전처리 완료")
            logger.info("=" * 60)

            # 생성된 파일들 확인
            logger.info("📁 생성된 파일들:")

            files_to_check = [
                "data/cache/feature_vector_full.npy",
                "data/cache/feature_vector_full.names.json",
                "data/result/analysis/optimized_analysis_result.json",
                "data/result/performance_reports/optimized_data_analysis_performance_report.json",
            ]

            for file_path in files_to_check:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.info(f"  ✓ {file_path} ({file_size:,} bytes)")
                else:
                    logger.warning(f"  ✗ {file_path} (파일 없음)")

            return True
        else:
            logger.error("데이터 분석 파이프라인 실행 실패")
            return False

    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
