#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
네거티브 샘플링 (3단계)

이 스크립트는 다음 작업을 수행합니다:
1. 비당첨 로또 조합 생성
2. 생성된 조합 벡터화
3. 원시 데이터와 벡터 데이터를 /data/cache/에 저장
4. 성능 보고서 생성
"""

import sys
from pathlib import Path

# 상위 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.negative_sampling_pipeline import run_negative_sampling


if __name__ == "__main__":
    run_negative_sampling()
