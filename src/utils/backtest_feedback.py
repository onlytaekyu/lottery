"""
로또 추천 결과 평가 모듈

이 모듈은 특정 회차의 추천 번호와 실제 당첨 번호를 비교하여 평가 결과를 제공합니다.
평가 지표는 일치 개수, ROI 추정치 등을 포함합니다.
"""

import json
import pandas as pd
import os
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from .error_handler import get_logger

# 로거 설정
logger = get_logger(__name__)


def generate_pattern_hash(numbers: List[int]) -> str:
    """
    번호 조합의 패턴 해시 생성

    Args:
        numbers: 정렬된 번호 리스트

    Returns:
        패턴 해시 문자열
    """
    # 번호 정렬 확인
    sorted_numbers = sorted(numbers)

    # 홀짝 패턴 (짝수 개수)
    even_count = sum(1 for num in sorted_numbers if num % 2 == 0)
    odd_count = len(sorted_numbers) - even_count

    # 고저 패턴 (23 이하 번호 개수)
    low_count = sum(1 for num in sorted_numbers if num <= 23)
    high_count = len(sorted_numbers) - low_count

    # 번호 분포 패턴 (1-10, 11-20, 21-30, 31-40, 41-45 구간별 개수)
    ranges = [0, 0, 0, 0, 0]
    for num in sorted_numbers:
        if 1 <= num <= 10:
            ranges[0] += 1
        elif 11 <= num <= 20:
            ranges[1] += 1
        elif 21 <= num <= 30:
            ranges[2] += 1
        elif 31 <= num <= 40:
            ranges[3] += 1
        elif 41 <= num <= 45:
            ranges[4] += 1

    # 패턴 문자열 생성
    pattern_str = (
        f"e{even_count}_o{odd_count}_l{low_count}_h{high_count}_"
        f"r{ranges[0]}_{ranges[1]}_{ranges[2]}_{ranges[3]}_{ranges[4]}"
    )

    # 해시 생성
    return hashlib.md5(pattern_str.encode()).hexdigest()


def update_failed_patterns(failed_numbers: List[List[int]]) -> Dict[str, int]:
    """
    실패한 패턴의 카운터를 업데이트

    Args:
        failed_numbers: 실패한 번호 조합 리스트

    Returns:
        업데이트된 실패 패턴 카운터
    """
    # 실패 패턴 파일 경로
    base_dir = Path(__file__).parent.parent.parent
    filters_dir = base_dir / "data" / "filters"
    filters_dir.mkdir(parents=True, exist_ok=True)
    failed_patterns_file = filters_dir / "failed_patterns.json"

    # 기존 데이터 로드
    failed_patterns = {}
    if failed_patterns_file.exists():
        try:
            with open(failed_patterns_file, "r", encoding="utf-8") as f:
                failed_patterns = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("실패 패턴 파일을 불러오는 중 오류 발생, 새 파일 생성")
            failed_patterns = {}

    # 실패 패턴 업데이트
    for numbers in failed_numbers:
        pattern_hash = generate_pattern_hash(numbers)
        failed_patterns[pattern_hash] = failed_patterns.get(pattern_hash, 0) + 1

    # 업데이트된 데이터 저장
    try:
        with open(failed_patterns_file, "w", encoding="utf-8") as f:
            json.dump(failed_patterns, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"실패 패턴 저장 중 오류: {str(e)}")

    return failed_patterns


def evaluate_round(round_number: int) -> dict:
    """
    특정 회차의 추천 번호 평가

    Args:
        round_number: 평가할 회차 번호

    Returns:
        평가 결과 (딕셔너리)
    """
    # 필요한 디렉토리 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    recommendations_dir = base_dir / "data" / "recommendations"
    evaluation_dir = base_dir / "data" / "evaluation"
    lottery_data_path = base_dir / "data" / "raw" / "lottery.csv"

    # 디렉토리 생성 (존재하지 않는 경우)
    recommendations_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # 추천 데이터 파일 경로
    recommendation_file = recommendations_dir / f"{round_number}.json"

    # 추천 데이터 로드
    try:
        with open(recommendation_file, "r", encoding="utf-8") as f:
            recommendations = json.load(f)
        logger.info(
            f"{round_number}회차 추천 데이터 로드 성공: {len(recommendations)}개"
        )
    except FileNotFoundError:
        error_msg = f"{round_number}회차 추천 데이터 파일이 없습니다."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError:
        error_msg = (
            f"{round_number}회차 추천 데이터 파일이 올바른 JSON 형식이 아닙니다."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 로또 당첨 데이터 로드
    try:
        df = pd.read_csv(lottery_data_path)

        # 해당 회차 데이터 추출
        winning_data = df[df["seqNum"] == round_number]

        if winning_data.empty:
            error_msg = f"{round_number}회차 당첨 데이터가 없습니다."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 당첨 번호 추출
        actual_numbers = []
        for i in range(1, 7):
            col_name = f"num{i}"
            if col_name in winning_data.columns:
                num = winning_data.iloc[0][col_name]
                actual_numbers.append(int(num))

        # 당첨 번호 정렬
        actual_numbers = sorted(actual_numbers)
        logger.info(f"{round_number}회차 당첨 번호: {actual_numbers}")

    except FileNotFoundError:
        error_msg = "로또 당첨 데이터 파일이 없습니다."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"당첨 데이터 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 평가 결과
    results = []
    total_match_count = 0
    total_roi = 0.0

    # 실패한 조합 저장 (match_count <= 1)
    failed_combinations = []

    # 각 추천 번호에 대한 평가
    for rec in recommendations:
        # 추천 번호 추출 (리스트 또는 숫자 목록 형태)
        if isinstance(rec, dict) and "numbers" in rec:
            numbers = rec["numbers"]
        elif isinstance(rec, list):
            numbers = rec
        else:
            logger.warning(f"알 수 없는 추천 형식, 건너뜁니다: {rec}")
            continue

        # 번호 형식 변환 및 정렬
        if not all(isinstance(n, int) for n in numbers):
            numbers = [int(n) for n in numbers]
        numbers = sorted(numbers)

        # 일치하는 번호 개수 계산
        match_count = len(set(numbers) & set(actual_numbers))

        # 실패한 조합 저장 (1개 이하 일치)
        if match_count <= 1:
            failed_combinations.append(numbers)

        # ROI 추정 (등수에 따른 수익률)
        roi_estimate = 0.0
        if match_count == 3:
            roi_estimate = 0.5  # 5등 (5,000원)
        elif match_count == 4:
            roi_estimate = 1.5  # 4등 (약 50,000원)
        elif match_count == 5:
            roi_estimate = 5.0  # 3등 (약 1,500,000원)
        elif match_count == 6:
            roi_estimate = 50.0  # 1등 (수십억원)

        # 타임스탬프와 전략 추출
        timestamp = (
            rec.get("timestamp", datetime.now().isoformat())
            if isinstance(rec, dict)
            else datetime.now().isoformat()
        )
        strategy = (
            rec.get("strategy", "unknown") if isinstance(rec, dict) else "unknown"
        )

        # 결과 추가
        result = {
            "numbers": numbers,
            "match_count": match_count,
            "roi_estimate": roi_estimate,
            "timestamp": timestamp,
            "strategy": strategy,
        }
        results.append(result)

        # 통계 계산
        total_match_count += match_count
        total_roi += roi_estimate

    # 실패한 패턴 업데이트 (실패 조합이 있는 경우에만)
    if failed_combinations:
        updated_patterns = update_failed_patterns(failed_combinations)
        logger.info(f"저성능 패턴 {len(failed_combinations)}개 업데이트 완료")

    # 평가 결과 딕셔너리 생성
    evaluation_result = {
        "round": round_number,
        "actual_numbers": actual_numbers,
        "results": results,
        "summary": {
            "recommendation_count": len(results),
            "avg_match_count": (
                round(total_match_count / len(results), 2) if results else 0
            ),
            "avg_roi": round(total_roi / len(results), 2) if results else 0,
            "failed_combinations_count": len(failed_combinations),
        },
    }

    # 평가 결과 저장
    evaluation_file = evaluation_dir / f"{round_number}_eval.json"
    try:
        with open(evaluation_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        logger.info(f"{round_number}회차 평가 결과 저장 완료: {evaluation_file}")
    except Exception as e:
        logger.error(f"평가 결과 저장 중 오류 발생: {str(e)}")

    return evaluation_result
