#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBM 모델 학습 스크립트

이 스크립트는 과거 로또 당첨 번호 데이터를 사용하여 LightGBM 모델을 학습시킵니다.
"""

import sys
from pathlib import Path

# 상위 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.train_pipeline import train_lightgbm_model
from src.utils.error_handler import get_logger, log_exception_with_trace

# 모듈 로거 설정
logger = get_logger(__name__)


if __name__ == "__main__":
    try:
        logger.info("LightGBM 모델 학습 시작")
        result = train_lightgbm_model()

        if result:
            logger.info("LightGBM 모델 학습 및 저장 완료")
            sys.exit(0)
        else:
            logger.error("LightGBM 모델 학습 실패")
            sys.exit(1)
    except Exception as e:
        log_exception_with_trace(logger, e, "LightGBM 모델 학습 중 치명적 오류 발생")
        sys.exit(1)
