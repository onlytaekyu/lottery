"""
패턴 필터링 모듈

이 모듈은 로또 번호 패턴을 필터링하는 기능을 제공합니다.
"""

import os
import hashlib
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional, Callable, Union, cast

from ..shared.types import LotteryNumber, PatternFeatures
from ..utils.error_handler import get_logger
from ..utils.config_loader import ConfigProxy, load_config

# 싱글톤 인스턴스
_pattern_filter_instance = None

# 로거 설정
logger = get_logger(__name__)


class PatternFilter:
    """로또 번호 패턴 필터링 클래스"""

    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigProxy]] = None):
        """
        초기화

        Args:
            config: 필터 설정
        """
        # 설정이 없으면 기본 설정 로드
        if config is None:
            config = load_config()

        # ConfigProxy로 변환
        if not isinstance(config, ConfigProxy):
            self.config = ConfigProxy(config if isinstance(config, dict) else {})
        else:
            self.config = config

        # 기본 데이터 디렉토리 경로 설정
        try:
            self.data_dir = Path(self.config["data_dir"])
        except KeyError:
            logger.warning(
                "설정에서 'data_dir'를 찾을 수 없습니다. 기본값 'data'를 사용합니다."
            )
            self.data_dir = Path("data")

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 실패 패턴 파일 경로
        self.failed_patterns_file = self.data_dir / "failed_patterns.json"

        # 저성능 패턴 파일 경로
        self.low_performance_patterns_file = (
            self.data_dir / "low_performance_patterns.json"
        )

        # 최소 실패 횟수 (이 값 이상이면 필터링)
        try:
            self.min_failure_count = self.config["min_failure_count"]
        except KeyError:
            logger.warning(
                "설정에서 'min_failure_count'를 찾을 수 없습니다. 기본값 3을 사용합니다."
            )
            self.min_failure_count = 3

        # 저성능 패턴 맵
        self.low_performance_patterns = self._load_low_performance_patterns()

        # 실패 패턴 맵
        self.failed_patterns = self._load_failed_patterns()

        # 패턴 변경 여부 플래그
        self.failed_patterns_changed = False

        # 필터 함수 목록
        self.filters: Dict[str, Callable] = {
            "even_odd_balance": self.filter_even_odd_balance,
            "high_low_balance": self.filter_high_low_balance,
            "sum_range": self.filter_sum_range,
            "consecutive_numbers": self.filter_consecutive_numbers,
            "number_gaps": self.filter_number_gaps,
            "historical_match": self.filter_historical_match,
            "failed_pattern": self.filter_failed_pattern,
        }

        logger.info(
            f"패턴 필터 초기화 완료 (저성능 패턴: {len(self.low_performance_patterns)}개, "
            f"실패 패턴: {len(self.failed_patterns)}개)"
        )

    def _get_max_consecutive_length(self, numbers: List[int]) -> int:
        """
        최대 연속 번호 길이 계산

        Args:
            numbers: 번호 목록 (정렬됨)

        Returns:
            최대 연속 번호 길이
        """
        if not numbers:
            return 0

        sorted_numbers = sorted(numbers)
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        return max_consecutive

    def get_pattern_features(self, numbers: List[int]) -> PatternFeatures:
        """
        번호 조합의 패턴 특성 추출

        Args:
            numbers: 번호 목록

        Returns:
            패턴 특성 딕셔너리
        """
        try:
            # 번호 정렬 확인
            sorted_numbers = sorted(numbers)

            # 기본 특성
            max_consecutive_length = self._get_max_consecutive_length(sorted_numbers)
            total_sum = sum(sorted_numbers)

            # 홀짝 패턴
            even_count, odd_count = self._get_even_odd_pattern(sorted_numbers)

            # 고저 패턴 (23 이하 번호 개수)
            low_count, high_count = self._get_low_high_pattern(sorted_numbers)

            # 번호 간격
            gaps = [
                sorted_numbers[i + 1] - sorted_numbers[i]
                for i in range(len(sorted_numbers) - 1)
            ]
            gap_avg = sum(gaps) / len(gaps) if gaps else 0
            gap_std = np.std(gaps) if gaps else 0

            # 번호 분포 패턴 (1-10, 11-20, 21-30, 31-40, 41-45 구간별 개수)
            range_counts = [0, 0, 0, 0, 0]
            for num in sorted_numbers:
                if 1 <= num <= 10:
                    range_counts[0] += 1
                elif 11 <= num <= 20:
                    range_counts[1] += 1
                elif 21 <= num <= 30:
                    range_counts[2] += 1
                elif 31 <= num <= 40:
                    range_counts[3] += 1
                elif 41 <= num <= 45:
                    range_counts[4] += 1

            # 연속성 점수 (연속된 번호가 적을수록 높은 점수)
            consecutive_score = 1.0 - (max_consecutive_length / 6.0)

            # 트렌드 점수 (임의 설정, 실제로는 과거 데이터 기반 계산 필요)
            trend_score_avg = 0.5
            trend_score_max = 0.5
            trend_score_min = 0.5

            # 리스크 점수 (합계, 간격 등 기반)
            # 합계 135를 기준으로 거리에 따라 증가 (90-210 범위 내에서)
            sum_score = 1.0 - min(abs(total_sum - 135) / 45.0, 1.0)  # pyright: ignore
            # 간격의 표준편차가 클수록 좋음 (다양성 증가)
            gap_score = min(gap_std / 10.0, 1.0)  # pyright: ignore
            # 구간 분포 균형성 (이상적인 분포: [1, 1, 1, 1, 2])
            ideal_dist = np.array([1, 1, 1, 1, 2])
            dist_diff = np.sum(np.abs(np.array(range_counts) - ideal_dist)) / 10.0
            dist_score = 1.0 - min(dist_diff, 1.0)  # pyright: ignore

            # 리스크 점수 종합 (각 요소 동일 가중치)
            risk_score = (sum_score + gap_score + dist_score) / 3.0

            # 패턴 해시
            pattern_hash = self.generate_pattern_hash(sorted_numbers)

            # 결과 구성
            features: PatternFeatures = {  # pyright: ignore
                "max_consecutive_length": max_consecutive_length,
                "total_sum": total_sum,
                "odd_count": odd_count,
                "even_count": even_count,
                "gap_avg": gap_avg,
                "gap_std": gap_std,
                "range_counts": range_counts,
                "consecutive_score": consecutive_score,
                "trend_score_avg": trend_score_avg,
                "trend_score_max": trend_score_max,
                "trend_score_min": trend_score_min,
                "risk_score": risk_score,
                "cluster_overlap_ratio": 0.0,  # 실제로는 클러스터 분석 후 설정
                "frequent_pair_score": 0.0,  # 실제로는 페어 빈도 분석 후 설정
                "roi_weight": 0.0,  # 실제로는 ROI 분석 후 설정
                "metadata": {
                    "pattern_hash": pattern_hash,
                    "low_count": low_count,
                    "high_count": high_count,
                },
            }

            return features

        except Exception as e:
            logger.error(f"패턴 특성 추출 중 오류 발생: {str(e)}")
            # 기본 특성 반환
            return {
                "max_consecutive_length": 0,
                "total_sum": sum(numbers),
                "odd_count": 3,
                "even_count": 3,
                "gap_avg": 0.0,
                "gap_std": 0.0,
                "range_counts": [0, 0, 0, 0, 0],
                "consecutive_score": 0.5,
                "trend_score_avg": 0.5,
                "trend_score_max": 0.5,
                "trend_score_min": 0.5,
                "risk_score": 0.5,
                "cluster_overlap_ratio": 0.0,
                "frequent_pair_score": 0.0,
                "roi_weight": 0.0,
                "metadata": {"pattern_hash": "unknown"},
            }

    def _load_low_performance_patterns(self) -> Dict[str, int]:
        """
        저성능 패턴 로드

        Returns:
            패턴 해시 -> 실패 횟수 매핑
        """
        if not self.low_performance_patterns_file.exists():
            logger.info(
                f"저성능 패턴 파일이 존재하지 않습니다: {self.low_performance_patterns_file}"
            )
            return {}

        try:
            with open(self.low_performance_patterns_file, "r", encoding="utf-8") as f:
                patterns = json.load(f)

            if not isinstance(patterns, dict):
                logger.warning("저성능 패턴 파일 형식이 올바르지 않습니다.")
                return {}

            # 문자열 키와 정수 값인지 확인
            validated_patterns = {}
            for key, value in patterns.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    validated_patterns[key] = int(value)

            logger.info(f"저성능 패턴 {len(validated_patterns)}개 로드 완료")
            return validated_patterns

        except Exception as e:
            logger.error(f"저성능 패턴 로드 중 오류: {str(e)}")
            return {}

    def _load_failed_patterns(self) -> Dict[str, int]:
        """
        실패 패턴 로드

        Returns:
            패턴 해시 -> 실패 횟수 매핑
        """
        if not self.failed_patterns_file.exists():
            logger.info(
                f"실패 패턴 파일이 존재하지 않습니다: {self.failed_patterns_file}"
            )
            return {}

        try:
            with open(self.failed_patterns_file, "r", encoding="utf-8") as f:
                patterns = json.load(f)

            if not isinstance(patterns, dict):
                logger.warning("실패 패턴 파일 형식이 올바르지 않습니다.")
                return {}

            # 문자열 키와 정수 값인지 확인
            validated_patterns = {}
            for key, value in patterns.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    validated_patterns[key] = int(value)

            logger.info(f"실패 패턴 {len(validated_patterns)}개 로드 완료")
            return validated_patterns

        except Exception as e:
            logger.error(f"실패 패턴 로드 중 오류: {str(e)}")
            return {}

    def save_failed_pattern(self, pattern_hash: str, failure_count: int = 1) -> None:
        """
        실패 패턴 저장

        Args:
            pattern_hash: 패턴 해시
            failure_count: 실패 횟수 (기본값: 1)
        """
        try:
            # 기존 패턴 업데이트 또는 추가
            if pattern_hash in self.failed_patterns:
                # 기존 실패 횟수에 추가
                old_count = self.failed_patterns[pattern_hash]
                self.failed_patterns[pattern_hash] += failure_count

                # 값이 변경된 경우에만 플래그 설정
                if old_count != self.failed_patterns[pattern_hash]:
                    self.failed_patterns_changed = True
            else:
                # 새 패턴 추가
                self.failed_patterns[pattern_hash] = failure_count
                self.failed_patterns_changed = True

            # 실패 횟수가 임계값 이상이면 로그 기록
            if self.failed_patterns[pattern_hash] >= self.min_failure_count:
                logger.info(
                    f"실패 패턴 감지: {pattern_hash} (실패 횟수: {self.failed_patterns[pattern_hash]})"
                )

            # 변경된 경우에만 파일에 저장
            if self.failed_patterns_changed:
                self._save_failed_patterns_to_file()

        except Exception as e:
            logger.error(f"실패 패턴 저장 중 오류: {str(e)}")

    def _save_failed_patterns_to_file(self) -> None:
        """실패 패턴을 파일에 저장"""
        try:
            with open(self.failed_patterns_file, "w", encoding="utf-8") as f:
                json.dump(self.failed_patterns, f, indent=2)
            self.failed_patterns_changed = False
            logger.debug(f"실패 패턴 {len(self.failed_patterns)}개 저장 완료")
        except Exception as e:
            logger.error(f"실패 패턴 파일 저장 중 오류: {str(e)}")

    def get_pattern_hash(self, numbers: List[int]) -> str:
        """
        번호 조합의 패턴 해시 생성

        Args:
            numbers: 번호 리스트

        Returns:
            패턴 해시 문자열
        """
        return self.generate_pattern_hash(numbers)

    def generate_pattern_hash(self, numbers: List[int]) -> str:
        """
        번호 조합의 패턴 해시 생성

        Args:
            numbers: 번호 리스트

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

    def _get_even_odd_pattern(self, numbers: List[int]) -> Tuple[int, int]:
        """홀수와 짝수의 개수 반환"""
        even_count = sum(1 for num in numbers if num % 2 == 0)
        odd_count = len(numbers) - even_count
        return (even_count, odd_count)

    def _get_low_high_pattern(self, numbers: List[int]) -> Tuple[int, int]:
        """낮은 숫자(1-23)와 높은 숫자(24-45)의 개수 반환"""
        low_count = sum(1 for num in numbers if num <= 23)
        high_count = len(numbers) - low_count
        return (low_count, high_count)

    def is_low_performance_pattern(self, pattern_hash: str) -> bool:
        """
        저성능 패턴 여부 확인

        Args:
            pattern_hash: 패턴 해시

        Returns:
            저성능 패턴 여부
        """
        # 패턴의 실패 횟수가 임계값 이상이면 저성능으로 판단
        return (
            self.low_performance_patterns.get(pattern_hash, 0) >= self.min_failure_count
        )

    def is_failed_pattern(self, pattern_hash: str) -> bool:
        """
        실패 패턴 여부 확인

        Args:
            pattern_hash: 패턴 해시

        Returns:
            실패 패턴 여부
        """
        # 패턴의 실패 횟수가 임계값 이상이면 실패 패턴으로 판단
        return self.failed_patterns.get(pattern_hash, 0) >= self.min_failure_count

    def should_filter(self, numbers: List[int]) -> bool:
        """
        번호 조합을 필터링해야 하는지 여부 판단 (analysis/pattern_filter.py에서 통합)

        Args:
            numbers: 번호 조합

        Returns:
            필터링 여부 (True면 제거)
        """
        # 실패 패턴 필터링
        pattern_hash = self.generate_pattern_hash(numbers)
        return self.failed_patterns.get(pattern_hash, 0) >= self.min_failure_count

    def filter_combinations(
        self, combinations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        번호 조합 리스트 필터링

        Args:
            combinations: 번호 조합 리스트 (각 항목은 'numbers' 키를 포함하는 딕셔너리)

        Returns:
            필터링된 번호 조합 리스트
        """
        if not combinations:
            return []

        filtered_combinations = []

        for combo in combinations:
            numbers = combo.get("numbers", [])
            if not numbers:
                continue

            # 패턴 해시 생성
            pattern_hash = self.get_pattern_hash(numbers)

            # 저성능 패턴 및 실패 패턴 필터링
            if not self.is_low_performance_pattern(
                pattern_hash
            ) and not self.is_failed_pattern(pattern_hash):
                filtered_combinations.append(combo)

        logger.info(
            f"패턴 필터링 결과: {len(combinations)}개 중 {len(filtered_combinations)}개 통과"
        )
        return filtered_combinations

    def add_failed_pattern(self, numbers: List[int]) -> None:
        """
        실패한 패턴 추가

        Args:
            numbers: 실패한 번호 조합
        """
        pattern_hash = self.get_pattern_hash(numbers)
        self.save_failed_pattern(pattern_hash)
        logger.debug(f"실패 패턴 추가: {pattern_hash}")

    def reload_patterns(self) -> None:
        """패턴 데이터 다시 로드"""
        self.low_performance_patterns = self._load_low_performance_patterns()
        self.failed_patterns = self._load_failed_patterns()
        self.failed_patterns_changed = False
        logger.info(
            f"패턴 재로드 완료 (저성능 패턴: {len(self.low_performance_patterns)}개, "
            f"실패 패턴: {len(self.failed_patterns)}개)"
        )

    # analysis/pattern_filter.py에서 통합한 필터 함수
    def filter_even_odd_balance(
        self, numbers: List[int], historical_data: Optional[List[LotteryNumber]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        홀수/짝수 균형 필터

        Args:
            numbers: 필터링할 번호 목록
            historical_data: 과거 당첨 번호 데이터

        Returns:
            (필터링 여부, 필터링 정보)
        """
        # 홀수/짝수 개수 계산
        odds = sum(1 for n in numbers if n % 2 == 1)
        evens = len(numbers) - odds

        # 설정에서 최소/최대 짝수 개수 가져오기
        try:
            min_even_numbers = self.config["filters"]["min_even_numbers"]
        except KeyError:
            logger.warning(
                "설정에서 'filters.min_even_numbers'를 찾을 수 없습니다. 기본값 2를 사용합니다."
            )
            min_even_numbers = 2

        try:
            max_even_numbers = self.config["filters"]["max_even_numbers"]
        except KeyError:
            logger.warning(
                "설정에서 'filters.max_even_numbers'를 찾을 수 없습니다. 기본값 4를 사용합니다."
            )
            max_even_numbers = 4

        # 허용 범위 내에 있는지 확인
        is_balanced = min_even_numbers <= evens <= max_even_numbers

        # 필터링 정보
        info = {
            "odd_count": odds,
            "even_count": evens,
            "balanced": is_balanced,
            "min_even": min_even_numbers,
            "max_even": max_even_numbers,
        }

        # 필터링 결과 반환
        return not is_balanced, info

    def filter_high_low_balance(
        self, numbers: List[int], historical_data: Optional[List[LotteryNumber]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        낮은 번호/높은 번호 균형 필터

        Args:
            numbers: 필터링할 번호 목록
            historical_data: 과거 당첨 번호 데이터

        Returns:
            (필터링 여부, 필터링 정보)
        """
        # 낮은 번호(1-22)와 높은 번호(23-45) 개수 계산
        low_threshold = 23
        low_count = sum(1 for n in numbers if n < low_threshold)
        high_count = len(numbers) - low_count

        # 설정에서 최소/최대 낮은 번호 개수 가져오기
        try:
            min_low_numbers = self.config["filters"]["min_low_numbers"]
        except KeyError:
            logger.warning(
                "설정에서 'filters.min_low_numbers'를 찾을 수 없습니다. 기본값 2를 사용합니다."
            )
            min_low_numbers = 2

        try:
            max_low_numbers = self.config["filters"]["max_low_numbers"]
        except KeyError:
            logger.warning(
                "설정에서 'filters.max_low_numbers'를 찾을 수 없습니다. 기본값 4를 사용합니다."
            )
            max_low_numbers = 4

        # 균형이 맞는지 확인
        is_balanced = min_low_numbers <= low_count <= max_low_numbers

        # 필터링 정보
        info = {
            "low_count": low_count,
            "high_count": high_count,
            "balanced": is_balanced,
            "min_low": min_low_numbers,
            "max_low": max_low_numbers,
        }

        # 필터링 결과 반환
        return not is_balanced, info

    def filter_sum_range(
        self, numbers: List[int], historical_data: Optional[List[LotteryNumber]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        합계 범위 필터

        Args:
            numbers: 필터링할 번호 목록
            historical_data: 과거 당첨 번호 데이터

        Returns:
            (필터링 여부, 필터링 정보)
        """
        total_sum = sum(numbers)

        # 설정에서 최소/최대 합계 가져오기
        try:
            min_sum = self.config["filters"]["min_sum"]
        except KeyError:
            logger.warning(
                "설정에서 'filters.min_sum'를 찾을 수 없습니다. 기본값 90을 사용합니다."
            )
            min_sum = 90

        try:
            max_sum = self.config["filters"]["max_sum"]
        except KeyError:
            logger.warning(
                "설정에서 'filters.max_sum'를 찾을 수 없습니다. 기본값 210을 사용합니다."
            )
            max_sum = 210

        # 합계가 허용 범위 내에 있는지 확인
        is_within_range = min_sum <= total_sum <= max_sum

        # 필터링 정보
        info = {
            "total_sum": total_sum,
            "min_sum": min_sum,
            "max_sum": max_sum,
            "within_range": is_within_range,
        }

        # 필터링 결과 반환
        return not is_within_range, info

    def filter_consecutive_numbers(
        self, numbers: List[int], historical_data: Optional[List[LotteryNumber]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        연속 번호 필터

        Args:
            numbers: 필터링할 번호 목록
            historical_data: 과거 당첨 번호 데이터

        Returns:
            (필터링 여부, 필터링 정보)
        """
        # 번호 정렬
        sorted_numbers = sorted(numbers)

        # 연속된 번호 세그먼트 찾기
        segments = []
        current_segment = [sorted_numbers[0]]

        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                current_segment.append(sorted_numbers[i])
            else:
                if len(current_segment) > 1:
                    segments.append(current_segment)
                current_segment = [sorted_numbers[i]]

        if len(current_segment) > 1:
            segments.append(current_segment)

        # 가장 긴 연속 번호 세그먼트 찾기
        max_consecutive = 0
        if segments:
            max_consecutive = max(len(segment) for segment in segments)

        # 설정에서 최대 연속 번호 개수 가져오기
        try:
            max_consecutive_allowed = self.config["filters"]["max_consecutive"]
        except KeyError:
            logger.warning(
                "설정에서 'filters.max_consecutive'를 찾을 수 없습니다. 기본값 4를 사용합니다."
            )
            max_consecutive_allowed = 4

        # 연속된 번호가 너무 많은지 확인
        too_many_consecutive = max_consecutive > max_consecutive_allowed

        # 필터링 정보
        info = {
            "max_consecutive": max_consecutive,
            "max_allowed": max_consecutive_allowed,
            "segments": segments,
            "too_many_consecutive": too_many_consecutive,
        }

        # 필터링 결과 반환
        return too_many_consecutive, info

    def filter_number_gaps(
        self, numbers: List[int], historical_data: Optional[List[LotteryNumber]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        번호 간격 검사

        Args:
            numbers: 검사할 번호
            historical_data: 과거 데이터 (사용되지 않음)

        Returns:
            (통과 여부, 상세 정보)
        """
        # 정렬된 번호
        sorted_numbers = sorted(numbers)

        # 번호 간 간격
        gaps = [
            sorted_numbers[i + 1] - sorted_numbers[i]
            for i in range(len(sorted_numbers) - 1)
        ]
        max_gap = max(gaps) if gaps else 0

        # 일반적으로 간격이 너무 크면 좋지 않음
        try:
            max_acceptable_gap = self.config["filters"]["max_acceptable_gap"]
        except KeyError:
            logger.warning(
                "설정에서 'filters.max_acceptable_gap'를 찾을 수 없습니다. 기본값 15를 사용합니다."
            )
            max_acceptable_gap = 15

        is_acceptable = max_gap <= max_acceptable_gap

        return is_acceptable, {
            "gaps": gaps,
            "max_gap": max_gap,
            "max_acceptable": max_acceptable_gap,
            "is_acceptable": is_acceptable,
        }

    def filter_historical_match(
        self, numbers: List[int], historical_data: Optional[List[LotteryNumber]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        과거 당첨 번호와의 일치 필터

        Args:
            numbers: 필터링할 번호 목록
            historical_data: 과거 당첨 번호 데이터

        Returns:
            (필터링 여부, 필터링 정보)
        """
        if not historical_data:
            return False, {"error": "과거 데이터가 없습니다"}

        # 번호 집합으로 변환
        number_set = set(numbers)

        # 과거 당첨 번호와 완전히 일치하는지 확인
        exact_matches = []

        for draw in historical_data:
            if set(draw.numbers) == number_set:
                exact_matches.append(draw.draw_no)

        # 설정에서 완전 일치 제외 여부 가져오기
        try:
            exclude_exact_match = self.config["filters"]["exclude_exact_past_match"]
        except KeyError:
            logger.warning(
                "설정에서 'filters.exclude_exact_past_match'를 찾을 수 없습니다. 기본값 True를 사용합니다."
            )
            exclude_exact_match = True

        # 완전 일치하는 경우 필터링
        has_exact_match = len(exact_matches) > 0
        should_filter = has_exact_match and exclude_exact_match

        # 필터링 정보
        info = {
            "exact_matches": exact_matches,
            "has_exact_match": has_exact_match,
            "exclude_policy": exclude_exact_match,
        }

        # 필터링 결과 반환
        return should_filter, info

    def filter_failed_pattern(
        self, numbers: List[int], historical_data: Optional[List[LotteryNumber]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        실패 패턴 검사

        Args:
            numbers: 검사할 번호
            historical_data: 과거 데이터 (사용되지 않음)

        Returns:
            (통과 여부, 상세 정보)
        """
        pattern_hash = self.generate_pattern_hash(numbers)

        is_failed_pattern = self.is_failed_pattern(pattern_hash)

        return not is_failed_pattern, {
            "pattern_hash": pattern_hash,
            "is_failed_pattern": is_failed_pattern,
            "failure_count": self.failed_patterns.get(pattern_hash, 0),
        }

    def filter_numbers(
        self,
        numbers: List[int],
        historical_data: Optional[List[LotteryNumber]] = None,
        filter_names: Optional[List[str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        여러 필터를 적용하여 번호 조합을 필터링합니다.

        Args:
            numbers: 필터링할 번호 목록
            historical_data: 과거 당첨 번호 데이터
            filter_names: 적용할 필터 이름 목록 (None이면 모든 필터 적용)

        Returns:
            (필터링 여부, 필터링 정보 디렉토리)
        """
        # 적용할 필터 목록
        if filter_names is None:
            filter_names = list(self.filters.keys())

        # 필터 결과와 정보
        results = {}
        should_filter = False

        # 각 필터 적용
        for name in filter_names:
            if name in self.filters:
                filter_func = self.filters[name]
                filter_result, filter_info = filter_func(numbers, historical_data)

                results[name] = {
                    "filtered": filter_result,
                    "info": filter_info,
                }

                # 하나라도 필터링되면 전체 결과는 필터링
                if filter_result:
                    should_filter = True

        # 필터 결과 및 정보 반환
        return should_filter, results


def get_pattern_filter(
    config: Optional[Union[Dict[str, Any], ConfigProxy]] = None,
) -> PatternFilter:
    """
    패턴 필터 인스턴스를 가져옵니다 (싱글톤 패턴).

    Args:
        config: 필터 설정

    Returns:
        PatternFilter: 패턴 필터 인스턴스
    """
    global _pattern_filter_instance

    if _pattern_filter_instance is None:
        _pattern_filter_instance = PatternFilter(config)
    elif config is not None:
        # 기존 인스턴스가 있지만 새 설정이 있는 경우, 새 인스턴스 생성
        _pattern_filter_instance = PatternFilter(config)

    return _pattern_filter_instance
