"""
패턴 트레이너 모듈

이 모듈은 로또 번호의 패턴을 분석하여 모델 학습에 사용할 패턴 분석 결과를 생성합니다.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter, defaultdict
import time
import math
from pathlib import Path

from ..shared.types import LotteryNumber, PatternAnalysis
from ..utils.error_handler import get_logger

# 로거 설정
logger = get_logger(__name__)


class PatternTrainer:
    """로또 번호 패턴 분석기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        패턴 트레이너 초기화

        Args:
            config: 설정 (선택사항)
        """
        # 기본 설정
        default_config = {
            "hot_threshold": 0.6,  # 핫 넘버 임계값
            "cold_threshold": 0.3,  # 콜드 넘버 임계값
            "recency_window": 20,  # 최근성 분석 창
            "pair_significance": 0.05,  # 쌍 유의성 임계값
            "probability_smoothing": 0.01,  # 확률 스무딩 계수
        }

        # 설정 병합
        self.config = default_config.copy()
        if config:
            self.config.update(config)

        # 분석 결과 저장 변수
        self.frequency_map = {}
        self.recency_map = {}
        self.pair_frequency = {}
        self.hot_numbers = []
        self.cold_numbers = []
        self.sum_distribution = []
        self.gap_patterns = {}
        self.probability_matrix = {}

        logger.info("패턴 트레이너 초기화 완료")

    def analyze_patterns(self, historical_data: List[LotteryNumber]) -> PatternAnalysis:
        """
        로또 번호 패턴 분석 (analyze 메서드의 별칭)

        analyze 메서드를 래핑하여 호환성을 제공합니다.

        Args:
            historical_data: 과거 로또 당첨 번호 데이터

        Returns:
            패턴 분석 결과
        """
        return self.analyze(historical_data)

    def analyze(self, historical_data: List[LotteryNumber]) -> PatternAnalysis:
        """
        로또 번호 패턴 분석

        Args:
            historical_data: 과거 로또 당첨 번호 데이터

        Returns:
            패턴 분석 결과
        """
        logger.info(f"패턴 분석 시작 - {len(historical_data)}개 데이터")
        start_time = time.time()

        # 데이터가 없으면 빈 분석 결과 반환
        if not historical_data:
            return self._create_empty_analysis()

        # 변환 필요한 경우 처리
        processed_data = self._preprocess_data(historical_data)

        # 각 패턴 분석 수행
        self._analyze_frequency(processed_data)
        self._analyze_recency(processed_data)
        self._analyze_pair_frequency(processed_data)
        self._analyze_gap_patterns(processed_data)
        self._analyze_sum_distribution(processed_data)
        self._analyze_probability_matrix(processed_data)
        self._classify_numbers()

        # PatternAnalysis 객체 생성
        analysis = PatternAnalysis(
            frequency_map=self.frequency_map,
            recency_map=self.recency_map,
            pair_frequency=self.pair_frequency,
            hot_numbers=set(self.hot_numbers),
            cold_numbers=set(self.cold_numbers),
            sum_distribution=self.sum_distribution,
            gap_patterns=self.gap_patterns,
            probability_matrix=self.probability_matrix,
            metadata={
                "analysis_time": time.time() - start_time,
                "data_count": len(processed_data),
                "sum_stats": self.sum_stats if hasattr(self, "sum_stats") else {},
            },
        )

        logger.info(
            f"패턴 분석 완료 - 시간: {time.time() - start_time:.1f}초, "
            f"핫 넘버: {len(self.hot_numbers)}개, 콜드 넘버: {len(self.cold_numbers)}개"
        )

        return analysis

    def _preprocess_data(self, data: List[Any]) -> List[List[int]]:
        """
        데이터 전처리

        Args:
            data: 입력 데이터

        Returns:
            전처리된 번호 목록
        """
        processed_data = []

        for item in data:
            numbers = []
            if isinstance(item, LotteryNumber):
                # LotteryNumber 객체인 경우
                numbers = item.numbers
            elif isinstance(item, dict) and "numbers" in item:
                # 사전 형태인 경우
                numbers = item["numbers"]
            elif isinstance(item, list) and all(isinstance(n, int) for n in item):
                # 번호 리스트인 경우
                numbers = item

            # 배열 형태로 변환하고 오름차순 정렬 후 추가
            if numbers:
                # 중복 번호 제거 및 유효한 범위 필터링 (1-45)
                valid_numbers = [n for n in numbers if 1 <= n <= 45]
                unique_numbers = list(set(valid_numbers))  # 중복 제거

                # 6개 미만이면 무시
                if len(unique_numbers) < 6:
                    logger.warning(f"6개 미만의 번호 제외: {unique_numbers}")
                    continue

                # 6개를 초과하면 처음 6개만 사용
                if len(unique_numbers) > 6:
                    unique_numbers = unique_numbers[:6]
                    logger.warning(f"6개 초과 번호 절단: 처음 6개만 사용")

                # 오름차순 정렬
                sorted_numbers = sorted(unique_numbers)

                processed_data.append(sorted_numbers)

        return processed_data

    def _create_empty_analysis(self) -> PatternAnalysis:
        """빈 패턴 분석 결과 생성"""
        # 모든 번호에 대해 균등한 빈도 설정
        uniform_freq = 1 / 45
        frequency_map = {i: uniform_freq for i in range(1, 46)}
        recency_map = {i: uniform_freq for i in range(1, 46)}

        return PatternAnalysis(
            frequency_map=frequency_map,
            recency_map=recency_map,
            pair_frequency={},
            hot_numbers=set(),
            cold_numbers=set(),
            sum_distribution=[],
            gap_patterns={},
            probability_matrix={},
            metadata={"analysis_time": 0, "data_count": 0, "is_empty": True},
        )

    def _analyze_frequency(self, data: List[List[int]]):
        """
        번호 빈도 분석

        Args:
            data: 전처리된 번호 데이터
        """
        # 빈도 카운트
        counter = Counter()
        for numbers in data:
            counter.update(numbers)

        # 총 회차 수
        total_draws = len(data)

        # 빈도 정규화 (출현 확률)
        if total_draws > 0:
            self.frequency_map = {
                num: count / (total_draws * 6) for num, count in counter.items()
            }

            # 모든 가능한 번호(1-45)에 대한 값 확인
            for num in range(1, 46):
                if num not in self.frequency_map:
                    self.frequency_map[num] = 0
        else:
            self.frequency_map = {i: 1 / 45 for i in range(1, 46)}

    def _analyze_recency(self, data: List[List[int]]):
        """
        번호 최근성 분석

        Args:
            data: 전처리된 데이터
        """
        # 최근 회차만 분석
        recency_window = min(self.config["recency_window"], len(data))
        recent_data = data[-recency_window:]

        # 최근 데이터의 빈도 카운트
        counter = Counter()
        for numbers in recent_data:
            counter.update(numbers)

        # 빈도 정규화
        total_recent = len(recent_data) * 6
        if total_recent > 0:
            self.recency_map = {
                num: count / total_recent for num, count in counter.items()
            }

            # 모든 가능한 번호(1-45)에 대한 값 확인
            for num in range(1, 46):
                if num not in self.recency_map:
                    self.recency_map[num] = 0
        else:
            self.recency_map = {i: 1 / 45 for i in range(1, 46)}

    def _analyze_pair_frequency(self, data: List[List[int]]):
        """
        번호 쌍 빈도 분석

        Args:
            data: 전처리된 번호 데이터
        """
        # 쌍 빈도 카운트
        pair_counter = Counter()

        # 모든 번호 쌍에 대한 빈도 카운트
        for numbers in data:
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    # 쌍 추출
                    pair = (min(numbers[i], numbers[j]), max(numbers[i], numbers[j]))
                    pair_counter[pair] += 1

        # 총 회차 수
        total_draws = len(data)

        # 쌍 빈도 정규화 (출현 확률)
        if total_draws > 0:
            self.pair_frequency = {
                pair: count / total_draws for pair, count in pair_counter.items()
            }
        else:
            self.pair_frequency = {}

    def _analyze_gap_patterns(self, data: List[List[int]]):
        """
        번호 간격 패턴 분석

        Args:
            data: 전처리된 데이터
        """
        # 간격 패턴 카운트
        gap_counter = Counter()

        # 각 번호 세트의 인접한 번호 간 간격 계산
        for numbers in data:
            # 번호가 오름차순으로 정렬되어 있는지 확인
            sorted_numbers = sorted(numbers)

            # 간격 계산
            for i in range(1, len(sorted_numbers)):
                gap = sorted_numbers[i] - sorted_numbers[i - 1]
                gap_counter[gap] += 1

        # 총 간격 개수
        total_gaps = sum(gap_counter.values())

        # 빈도 정규화
        if total_gaps > 0:
            self.gap_patterns = {
                gap: count / total_gaps for gap, count in gap_counter.items()
            }
        else:
            self.gap_patterns = {}

    def _analyze_sum_distribution(self, data: List[List[int]]):
        """
        합계 분포 분석

        Args:
            data: 전처리된 데이터
        """
        # 합계 분포 계산
        sums = [sum(numbers) for numbers in data]

        # 최소, 최대, 평균, 중앙값, 표준편차 계산
        if sums:
            self.sum_distribution = sums
            # 메타데이터로 추가 정보 저장
            self.sum_stats = {
                "min": min(sums),
                "max": max(sums),
                "mean": sum(sums) / len(sums),
                "median": sorted(sums)[len(sums) // 2],
                "std": math.sqrt(
                    sum((x - sum(sums) / len(sums)) ** 2 for x in sums) / len(sums)
                ),
                "distribution": Counter(sums),
            }
        else:
            self.sum_distribution = []
            self.sum_stats = {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "std": 0,
                "distribution": {},
            }

    def _analyze_probability_matrix(self, data: List[List[int]]):
        """
        확률 행렬 분석

        Args:
            data: 전처리된 데이터
        """
        # 총 발생 횟수
        total_count = len(data)

        if total_count == 0:
            self.probability_matrix = {}
            return

        # 조건부 확률 행렬 초기화
        probability_matrix = {}

        # 각 쌍의 공동 발생 확률 계산
        for i in range(1, 46):
            probability_matrix[i] = {}
            for j in range(1, 46):
                if i == j:
                    continue

                # i가 발생했을 때 j가 함께 발생한 횟수 카운트
                count_i = 0  # i가 발생한 총 횟수
                count_i_with_j = 0  # i와 j가 함께 발생한 횟수

                for numbers in data:
                    if i in numbers:
                        count_i += 1
                        if j in numbers:
                            count_i_with_j += 1

                # 스무딩 파라미터 적용 (부드러운 추정)
                smoothing = self.config["probability_smoothing"]
                probability = (count_i_with_j + smoothing) / (count_i + 2 * smoothing)

                probability_matrix[i][j] = probability

        self.probability_matrix = probability_matrix

    def _classify_numbers(self):
        """
        빈도를 기반으로 인기 및 비인기 번호를 분류합니다.
        """
        sorted_numbers = sorted(
            self.frequency_map.items(), key=lambda x: x[1], reverse=True
        )
        hot_threshold = np.percentile([f for _, f in sorted_numbers], 75)
        cold_threshold = np.percentile([f for _, f in sorted_numbers], 25)

        # 핫 넘버 및 콜드 넘버 분류
        self.hot_numbers = set(
            num for num, freq in self.frequency_map.items() if freq >= hot_threshold
        )
        self.cold_numbers = set(
            num for num, freq in self.frequency_map.items() if freq <= cold_threshold
        )

    def save_analysis(self, analysis: PatternAnalysis, path: str) -> bool:
        """
        분석 결과 저장

        Args:
            analysis: 저장할 분석 결과
            path: 저장 경로

        Returns:
            저장 성공 여부
        """
        try:
            # 경로 파싱
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 데이터 변환 (직렬화 가능한 형태로)
            data = {
                "frequency_map": dict(analysis.frequency_map),
                "recency_map": dict(analysis.recency_map),
                "pair_frequency": {
                    f"{k[0]}_{k[1]}": v for k, v in analysis.pair_frequency.items()
                },
                "hot_numbers": list(analysis.hot_numbers),
                "cold_numbers": list(analysis.cold_numbers),
                "sum_distribution": analysis.sum_distribution,
                "gap_patterns": analysis.gap_patterns,
                "metadata": analysis.metadata or {},
            }

            # JSON 추가 정보
            data["metadata"]["timestamp"] = time.time()
            data["metadata"]["version"] = "1.0"

            # 파일 저장
            import json

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"분석 결과가 {save_path}에 저장되었습니다.")
            return True
        except Exception as e:
            logger.error(f"분석 결과 저장 실패: {str(e)}")
            return False

    def load_analysis(self, path: str) -> Optional[PatternAnalysis]:
        """
        저장된 분석 결과 로드

        Args:
            path: 로드할 파일 경로

        Returns:
            로드된 분석 결과
        """
        try:
            # 파일 확인
            load_path = Path(path)
            if not load_path.exists():
                logger.error(f"분석 결과 파일을 찾을 수 없습니다: {path}")
                return None

            # 파일 로드
            import json

            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 데이터 변환 (PatternAnalysis 형태로)
            pair_frequency = {}
            for key, value in data.get("pair_frequency", {}).items():
                if "_" in key:
                    num1, num2 = map(int, key.split("_"))
                    pair_frequency[(num1, num2)] = value

            # Set 형태로 변환
            hot_numbers = set(data.get("hot_numbers", []))
            cold_numbers = set(data.get("cold_numbers", []))

            # sum_distribution이 없는 경우 처리
            sum_distribution = data.get("sum_distribution", [])
            if isinstance(sum_distribution, dict):
                # 이전 버전 형식 변환
                sum_distribution = list(range(min(170, 270), max(100, 230)))

            # PatternAnalysis 객체 생성
            analysis = PatternAnalysis(
                frequency_map=data.get("frequency_map", {}),
                recency_map=data.get("recency_map", {}),
                pair_frequency=pair_frequency,
                hot_numbers=set(hot_numbers),  # 명시적으로 Set[int]로 변환
                cold_numbers=set(cold_numbers),  # 명시적으로 Set[int]로 변환
                sum_distribution=sum_distribution,
                gap_patterns=data.get("gap_patterns", {}),
                probability_matrix=data.get("probability_matrix", {}),
                metadata=data.get("metadata", {}),
            )

            logger.info(f"분석 결과가 {path}에서 로드되었습니다.")
            return analysis
        except Exception as e:
            logger.error(f"분석 결과 로드 실패: {str(e)}")
            return None
