#!/usr/bin/env python
"""
상위 ROI 쌍 계산 및 저장 스크립트

이 스크립트는 역사적 로또 데이터를 분석하여 최상위 ROI 값을 가진 번호 쌍을 계산하고 저장합니다.
"""

import os
import sys
import json
from pathlib import Path
from typing import cast, List

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.roi_analyzer import ROIAnalyzer
from src.utils.data_loader import load_draw_history
from src.shared.types import LotteryNumber
from src.utils.error_handler import get_logger

logger = get_logger(__name__)


def main():
    """메인 함수"""
    try:
        # 데이터 경로 설정
        data_path = project_root / "data" / "raw" / "lottery.csv"

        # 경로 확인
        if not data_path.exists():
            logger.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            return 1

        # 로또 데이터 로드
        logger.info(f"로또 데이터 로드 중: {data_path}")
        lottery_data = load_draw_history(str(data_path))
        logger.info(f"데이터 로드 완료: {len(lottery_data)}개 회차 데이터")

        # ROI 분석기 초기화
        logger.info("ROI 분석기 초기화 중...")
        roi_analyzer = ROIAnalyzer()

        # 타입 캐스팅을 통해 타입 호환성 문제 해결
        # load_draw_history의 반환 타입과 ROIAnalyzer가 기대하는 타입이 미묘하게 다름
        history_data = cast(List[LotteryNumber], lottery_data)

        # 상위 ROI 쌍 계산 및 저장
        logger.info("상위 ROI 쌍 계산 중...")
        roi_pairs = roi_analyzer.get_top_roi_pairs(history_data, top_n=20)

        # 파일로 저장
        logger.info("ROI 쌍 저장 중...")
        success = roi_analyzer.save_top_roi_pairs(history_data, top_n=20)

        if success:
            # 저장된 쌍 로그 출력
            logger.info(f"상위 ROI 쌍 저장 완료: {len(roi_pairs)}개")
            for i, pair in enumerate(roi_pairs[:5]):
                logger.info(f"Top {i+1}: {pair}")
            return 0
        else:
            logger.error("ROI 쌍 저장 실패")
            return 1

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
