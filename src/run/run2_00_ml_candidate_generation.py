#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML 기반 초기 후보 생성 (2단계)

이 스크립트는 다음 작업을 수행합니다:
1. 전처리된 데이터와 캐시된 벡터 로드
2. MLCandidateGenerator를 사용하여 초기 후보 생성
3. 규칙 필터와 위험도 점수 적용
4. LightGBM + XGBoost로 후보에 점수 부여
5. 점수가 매겨진 후보 풀을 캐시에 저장
"""

import sys
from pathlib import Path

# 상위 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.candidate_generation_pipeline import run_ml_candidate_generation


if __name__ == "__main__":
    run_ml_candidate_generation()
