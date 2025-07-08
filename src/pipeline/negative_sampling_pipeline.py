#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
네거티브 샘플링 파이프라인 모듈
"""

import time
import random
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple

from src.utils.data_loader import load_draw_history
from src.utils.unified_config import load_config
from src.utils.unified_logging import get_logger
from src.utils.unified_performance_engine import get_auto_performance_monitor
from src.analysis.pattern_analyzer import PatternAnalyzer
from src.analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer

# 로거 설정
logger = get_logger(__name__)


def run_negative_sampling(
    negative_count: int = 1000,
    balanced_sampling: bool = True,
    save_results: bool = True,
    generate_report: bool = True,
) -> bool:
    """
    네거티브 샘플링 실행

    Args:
        negative_count: 생성할 네거티브 샘플 수
        balanced_sampling: 균형 잡힌 샘플링 사용 여부
        save_results: 결과를 파일로 저장할지 여부
        generate_report: 성능 보고서 생성 여부

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
    profiler = get_auto_performance_monitor()

    try:
        # 1. 데이터 로드
        with profiler.track("데이터 로드"):
            logger.info("3단계: 과거 당첨 번호 데이터 로드 중...")
            historical_data = load_draw_history()

            if not historical_data:
                logger.error("당첨 번호 데이터를 로드할 수 없습니다.")
                return False

            logger.info(f"데이터 로드 완료: {len(historical_data)}개 회차")

            # 캐시 및 결과 디렉토리 설정
            cache_dir = Path(config.safe_get("paths.cache_dir", "data/cache"))
            report_dir = Path(
                config.safe_get("paths.report_dir", "logs/performance_reports")
            )
            negative_dir = cache_dir / "negative_samples"

            # 디렉토리 생성
            cache_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)
            negative_dir.mkdir(parents=True, exist_ok=True)

            # 당첨 번호 세트 생성 (빠른 조회용)
            winning_combinations = set()
            for draw in historical_data:
                winning_combinations.add(tuple(sorted(draw.numbers)))

            logger.info(f"당첨 번호 세트 생성 완료: {len(winning_combinations)}개")

        # 2. 패턴 분석기 초기화
        with profiler.track("패턴 분석기 초기화"):
            pattern_analyzer = PatternAnalyzer(config.to_dict())
            pattern_vectorizer = EnhancedPatternVectorizer(config.to_dict())
            logger.info("패턴 분석기 초기화 완료")

        # 3. 네거티브 샘플 생성
        with profiler.track("네거티브 샘플 생성"):
            logger.info(f"네거티브 샘플 {negative_count}개 생성 중...")

            if balanced_sampling:
                logger.info("균형 잡힌 샘플링 사용 (각 카테고리별 균등 비율)")
                # 다양한 패턴 카테고리 정의
                categories = [
                    "high_variance",  # 분산이 큰 조합
                    "low_variance",  # 분산이 작은 조합
                    "consecutive_heavy",  # 연속 번호가 많은 조합
                    "segment_balanced",  # 세그먼트별 균형 있는 조합
                    "segment_skewed",  # 세그먼트별 치우친 조합
                    "random",  # 완전 랜덤 조합
                ]

                # 각 카테고리별 생성할 샘플 수 계산
                per_category = negative_count // len(categories)
                remaining = negative_count % len(categories)

                # 카테고리별로 네거티브 샘플 생성
                negative_samples = []
                for i, category in enumerate(categories):
                    count = per_category + (1 if i < remaining else 0)
                    samples = generate_negative_samples_by_category(
                        count, category, winning_combinations
                    )
                    negative_samples.extend(samples)
                    logger.info(f"- {category}: {len(samples)}개 생성")
            else:
                logger.info("랜덤 샘플링 사용")
                negative_samples = generate_negative_samples(
                    negative_count, winning_combinations
                )

            logger.info(f"네거티브 샘플 생성 완료: {len(negative_samples)}개")

        # 4. 샘플 벡터화
        with profiler.track("샘플 벡터화"):
            logger.info("네거티브 샘플 벡터화 중...")

            # 샘플 데이터 준비
            negative_data = []
            negative_vectors = []

            for i, numbers in enumerate(negative_samples):
                # 패턴 특성 추출 - 튜플을 리스트로 변환
                features = pattern_analyzer.extract_pattern_features(
                    list(numbers), historical_data
                )

                # 특성 벡터화
                feature_vector = pattern_vectorizer.vectorize_pattern_features(features)

                # 데이터 준비
                sample_data = {
                    "id": i + 1,
                    "numbers": list(numbers),
                    "is_winning": False,
                    "features": features,
                }

                negative_data.append(sample_data)
                negative_vectors.append(feature_vector)

            # 벡터 배열로 변환
            negative_vectors_array = np.array(negative_vectors)
            logger.info(f"벡터화 완료: {negative_vectors_array.shape}")

        # 5. 결과 저장
        if save_results:
            with profiler.track("결과 저장"):
                # 현재 시간을 파일명에 포함
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 샘플 데이터 저장
                data_file = negative_dir / f"negative_samples_{timestamp}.json"
                with open(data_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "generated_at": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "total_samples": len(negative_samples),
                            "balanced_sampling": balanced_sampling,
                            "samples": negative_data,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                # 벡터 데이터 저장
                vector_file = negative_dir / f"negative_vectors_{timestamp}.npy"
                np.save(vector_file, negative_vectors_array)

                # 최신 파일 참조용 심볼릭 링크 또는 복사본 생성
                latest_data_file = negative_dir / "latest_negative_samples.json"
                latest_vector_file = negative_dir / "latest_negative_vectors.npy"

                # 파일 복사 (윈도우 호환성 위해 심볼릭 링크 대신 복사)
                import shutil

                shutil.copy2(data_file, latest_data_file)
                shutil.copy2(vector_file, latest_vector_file)

                logger.info(f"데이터 저장 완료: {data_file}")
                logger.info(f"벡터 저장 완료: {vector_file}")
                logger.info("최신 파일 참조 생성 완료")

        # 6. 네거티브 샘플 분석 및 보고서 생성
        if generate_report:
            with profiler.track("네거티브 샘플 분석"):
                logger.info("네거티브 샘플 분석 중...")

                # 샘플 분석 수행
                analysis_result = analyze_negative_samples(
                    negative_samples, historical_data
                )

                # 보고서 파일 저장
                report_file = report_dir / f"negative_samples_report_{timestamp}.json"
                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "generated_at": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "total_samples": len(negative_samples),
                            "balanced_sampling": balanced_sampling,
                            "analysis": analysis_result,
                            "execution_time_sec": time.time() - start_time,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                logger.info(f"분석 보고서 저장 완료: {report_file}")

                # 주요 분석 결과 로깅
                logger.info("네거티브 샘플 주요 특성:")
                logger.info(
                    f"- 평균 합계: {analysis_result['sum_stats']['mean']:.2f} (표준편차: {analysis_result['sum_stats']['std']:.2f})"
                )
                logger.info(
                    f"- 평균 홀수 개수: {analysis_result['odd_count_stats']['mean']:.2f}"
                )
                logger.info(
                    f"- 연속 번호 조합 비율: {analysis_result['consecutive_pairs_ratio']:.2%}"
                )
                logger.info(
                    f"- 구간별 분포 균형도: {analysis_result['segment_balance']:.2f} (높을수록 균형)"
                )

        # 실행 시간 기록
        execution_time = time.time() - start_time
        logger.info(f"네거티브 샘플링 완료 (소요시간: {execution_time:.2f}초)")

        # 프로파일링 결과 출력
        performance_summary = profiler.get_performance_summary()
        logger.info(f"성능 요약: {performance_summary}")

        return True

    except Exception as e:
        logger.error(f"네거티브 샘플링 중 오류 발생: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def generate_negative_samples(
    count: int, winning_combinations: Set[Tuple[int, ...]]
) -> List[Tuple[int, ...]]:
    """
    기본적인 네거티브 샘플 생성

    Args:
        count: 생성할 샘플 수
        winning_combinations: 실제 당첨 번호 세트 (제외 목록)

    Returns:
        생성된 네거티브 샘플 목록
    """
    samples = []
    attempts = 0
    max_attempts = count * 10  # 최대 시도 횟수

    while len(samples) < count and attempts < max_attempts:
        # 6개의 번호 랜덤 선택 (1~45)
        numbers = tuple(sorted(random.sample(range(1, 46), 6)))

        # 이미 당첨된 적 있는 조합이거나 이미 생성된 샘플인지 확인
        if numbers not in winning_combinations and numbers not in samples:
            samples.append(numbers)

        attempts += 1

    return samples


def generate_negative_samples_by_category(
    count: int, category: str, winning_combinations: Set[Tuple[int, ...]]
) -> List[Tuple[int, ...]]:
    """
    카테고리별 특성을 가진 네거티브 샘플 생성

    Args:
        count: 생성할 샘플 수
        category: 샘플 카테고리 (high_variance, low_variance, consecutive_heavy, segment_balanced, segment_skewed, random)
        winning_combinations: 실제 당첨 번호 세트 (제외 목록)

    Returns:
        생성된 네거티브 샘플 목록
    """
    samples = []
    attempts = 0
    max_attempts = count * 20  # 최대 시도 횟수

    # 세그먼트 정의 (1-9, 10-18, 19-27, 28-36, 37-45)
    segments = [
        list(range(1, 10)),
        list(range(10, 19)),
        list(range(19, 28)),
        list(range(28, 37)),
        list(range(37, 46)),
    ]

    while len(samples) < count and attempts < max_attempts:
        numbers = []

        # 카테고리별 생성 로직
        if category == "high_variance":
            # 분산이 큰 조합 (각 세그먼트에서 골고루 선택)
            for segment in segments:
                if len(numbers) < 6:
                    numbers.append(random.choice(segment))

            # 부족하면 추가
            while len(numbers) < 6:
                num = random.randint(1, 45)
                if num not in numbers:
                    numbers.append(num)

        elif category == "low_variance":
            # 분산이 작은 조합 (좁은 범위에서 선택)
            start = random.randint(1, 30)
            end = min(start + 15, 45)
            pool = list(range(start, end + 1))

            if len(pool) >= 6:
                numbers = random.sample(pool, 6)
            else:
                # 범위가 좁다면 다른 번호로 보충
                numbers = random.sample(pool, len(pool))
                remaining = 6 - len(numbers)
                additional = random.sample(
                    [n for n in range(1, 46) if n not in numbers], remaining
                )
                numbers.extend(additional)

        elif category == "consecutive_heavy":
            # 연속 번호가 많은 조합
            # 연속된 3-4개 번호 그룹 추가
            start = random.randint(1, 42)
            consecutive_length = random.randint(3, 4)
            consecutive_numbers = list(
                range(start, min(start + consecutive_length, 46))
            )

            numbers.extend(consecutive_numbers)

            # 부족한 번호 추가
            while len(numbers) < 6:
                num = random.randint(1, 45)
                if num not in numbers:
                    numbers.append(num)

        elif category == "segment_balanced":
            # 세그먼트별 균형 있는 조합 (각 세그먼트에서 최대 2개씩)
            selected_segments = random.sample(segments, min(4, len(segments)))
            for segment in selected_segments:
                nums_to_pick = min(random.randint(1, 2), 6 - len(numbers))
                if nums_to_pick > 0 and len(segment) >= nums_to_pick:
                    numbers.extend(random.sample(segment, nums_to_pick))

                if len(numbers) >= 6:
                    break

            # 부족하면 추가
            while len(numbers) < 6:
                num = random.randint(1, 45)
                if num not in numbers:
                    numbers.append(num)

        elif category == "segment_skewed":
            # 세그먼트별 치우친 조합 (1-2개 세그먼트에 집중)
            selected_segments = random.sample(segments, random.randint(1, 2))
            for segment in selected_segments:
                nums_to_pick = min(random.randint(3, 5), 6 - len(numbers))
                if nums_to_pick > 0 and len(segment) >= nums_to_pick:
                    numbers.extend(random.sample(segment, nums_to_pick))

                if len(numbers) >= 6:
                    break

            # 부족하면 추가
            while len(numbers) < 6:
                num = random.randint(1, 45)
                if num not in numbers:
                    numbers.append(num)

        else:  # random
            # 완전 랜덤 조합
            numbers = random.sample(range(1, 46), 6)

        # 중복 제거 및 정렬
        numbers = sorted(set(numbers))

        # 6개가 아니면 다시 시도
        if len(numbers) != 6:
            attempts += 1
            continue

        # 튜플로 변환하여 저장
        number_tuple = tuple(numbers)

        # 실제 당첨 조합이거나 이미 생성된 샘플이면 제외
        if number_tuple not in winning_combinations and number_tuple not in samples:
            samples.append(number_tuple)

        attempts += 1

    return samples


def analyze_negative_samples(
    negative_samples: List[Tuple[int, ...]], historical_data
) -> Dict[str, Any]:
    """
    생성된 네거티브 샘플의 특성 분석

    Args:
        negative_samples: 분석할 네거티브 샘플 목록
        historical_data: 과거 당첨 번호 데이터 (참조용)

    Returns:
        분석 결과 (통계 정보)
    """
    # 분석 결과 저장용 딕셔너리
    analysis = {}

    # 샘플 수
    sample_count = len(negative_samples)
    analysis["sample_count"] = sample_count

    # 당첨 번호 통계 계산 (비교용)
    winning_sums = []
    winning_odd_counts = []
    winning_consecutive_counts = []
    winning_segment_distributions = []

    for draw in historical_data:
        numbers = draw.numbers
        winning_sums.append(sum(numbers))
        winning_odd_counts.append(sum(1 for n in numbers if n % 2 == 1))

        # 연속 번호 계산
        sorted_numbers = sorted(numbers)
        consecutive_count = 0
        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                consecutive_count += 1
        winning_consecutive_counts.append(consecutive_count)

        # 세그먼트 분포 계산
        segment_dist = [0] * 5
        for num in numbers:
            if 1 <= num <= 9:
                segment_dist[0] += 1
            elif 10 <= num <= 18:
                segment_dist[1] += 1
            elif 19 <= num <= 27:
                segment_dist[2] += 1
            elif 28 <= num <= 36:
                segment_dist[3] += 1
            else:  # 37-45
                segment_dist[4] += 1
        winning_segment_distributions.append(segment_dist)

    # 네거티브 샘플 통계 계산
    sums = []
    odd_counts = []
    consecutive_counts = []
    segment_distributions = []

    for sample in negative_samples:
        numbers = list(sample)
        sums.append(sum(numbers))
        odd_counts.append(sum(1 for n in numbers if n % 2 == 1))

        # 연속 번호 계산
        sorted_numbers = sorted(numbers)
        consecutive_count = 0
        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                consecutive_count += 1
        consecutive_counts.append(consecutive_count)

        # 세그먼트 분포 계산
        segment_dist = [0] * 5
        for num in numbers:
            if 1 <= num <= 9:
                segment_dist[0] += 1
            elif 10 <= num <= 18:
                segment_dist[1] += 1
            elif 19 <= num <= 27:
                segment_dist[2] += 1
            elif 28 <= num <= 36:
                segment_dist[3] += 1
            else:  # 37-45
                segment_dist[4] += 1
        segment_distributions.append(segment_dist)

    # 합계 통계
    analysis["sum_stats"] = {
        "mean": float(np.mean(sums)),
        "std": float(np.std(sums)),
        "min": min(sums),
        "max": max(sums),
        "winning_mean": float(np.mean(winning_sums)),
        "winning_std": float(np.std(winning_sums)),
    }

    # 홀수 개수 통계
    analysis["odd_count_stats"] = {
        "mean": float(np.mean(odd_counts)),
        "distribution": {str(i): odd_counts.count(i) / sample_count for i in range(7)},
        "winning_mean": float(np.mean(winning_odd_counts)),
    }

    # 연속 번호 통계
    analysis["consecutive_stats"] = {
        "mean": float(np.mean(consecutive_counts)),
        "distribution": {
            str(i): consecutive_counts.count(i) / sample_count for i in range(6)
        },
        "winning_mean": float(np.mean(winning_consecutive_counts)),
    }

    # 연속 번호 쌍이 있는 조합 비율
    consecutive_pairs_count = sum(1 for count in consecutive_counts if count > 0)
    analysis["consecutive_pairs_ratio"] = consecutive_pairs_count / sample_count

    # 세그먼트 분포 균형도 (표준편차 역수로 표현, 높을수록 균형)
    segment_stds = []
    for dist in segment_distributions:
        segment_stds.append(np.std(dist))

    analysis["segment_balance"] = 1.0 / (
        float(np.mean(segment_stds)) + 0.01
    )  # 0으로 나눔 방지

    # 세그먼트별 평균 번호 수
    segment_means = np.mean(segment_distributions, axis=0)
    analysis["segment_distribution"] = {
        "1-9": float(segment_means[0]),
        "10-18": float(segment_means[1]),
        "19-27": float(segment_means[2]),
        "28-36": float(segment_means[3]),
        "37-45": float(segment_means[4]),
    }

    return analysis
