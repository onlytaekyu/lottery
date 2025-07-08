"""
3자리 → 6자리 확장 엔진

3자리 패턴을 기반으로 6자리 로또 번호를 생성하는 고성능 확장 엔진
GPU 가속, 멀티쓰레드 처리, 캐시 최적화를 통한 고속 처리
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import defaultdict

from ..utils.unified_logging import get_logger
from ..utils.unified_performance_engine import ComputeExecutor
from ..utils.cache_manager import get_cache_manager
from ..utils.memory_manager import get_memory_manager
from ..shared.types import LotteryNumber

logger = get_logger(__name__)


@dataclass
class ExpansionCandidate:
    """확장 후보 데이터 클래스"""

    three_digit_combo: Tuple[int, int, int]
    six_digit_combo: Tuple[int, int, int, int, int, int]
    confidence_score: float
    expansion_method: str
    additional_info: Dict[str, Any]


class ThreeDigitExpansionEngine:
    """3자리 → 6자리 확장 엔진"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        확장 엔진 초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # 성능 최적화 시스템 초기화
        self.compute_executor = ComputeExecutor()
        self.cache_manager = get_cache_manager()
        self.memory_manager = get_memory_manager()

        # 확장 캐시
        self.expansion_cache = {}
        self.pattern_cache = {}

        # 스레드 풀 설정
        self.max_workers = self.config.get("max_workers", 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        # 확장 전략 설정
        self.expansion_strategies = {
            "frequency_based": self._expand_by_frequency,
            "pattern_based": self._expand_by_pattern,
            "ml_based": self._expand_by_ml,
            "hybrid": self._expand_by_hybrid,
        }

        self.logger.info("✅ 3자리 확장 엔진 초기화 완료")

    def expand_top_candidates(
        self,
        top_3digit_candidates: List[Dict[str, Any]],
        historical_data: List[LotteryNumber],
        target_count: int = 50,
    ) -> List[ExpansionCandidate]:
        """
        상위 3자리 후보들을 6자리로 확장

        Args:
            top_3digit_candidates: 상위 3자리 후보 리스트
            historical_data: 과거 당첨 번호 데이터
            target_count: 목표 생성 개수

        Returns:
            List[ExpansionCandidate]: 확장된 6자리 후보들
        """
        try:
            self.logger.info(
                f"🚀 상위 3자리 후보 확장 시작: {len(top_3digit_candidates)}개 → {target_count}개 목표"
            )
            start_time = time.time()

            all_candidates = []

            # 병렬 처리로 각 3자리 후보 확장
            futures = []
            for candidate in top_3digit_candidates:
                three_combo = candidate["combination"]
                future = self.thread_pool.submit(
                    self._expand_single_candidate,
                    three_combo,
                    historical_data,
                    candidate,
                )
                futures.append(future)

            # 결과 수집
            for future in as_completed(futures):
                try:
                    expanded_candidates = future.result()
                    all_candidates.extend(expanded_candidates)
                except Exception as e:
                    self.logger.error(f"후보 확장 중 오류: {e}")

            # 점수 기준 정렬하여 상위 선택
            all_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
            final_candidates = all_candidates[:target_count]

            elapsed_time = time.time() - start_time
            self.logger.info(
                f"✅ 3자리 확장 완료: {len(final_candidates)}개 생성 ({elapsed_time:.2f}초)"
            )

            return final_candidates

        except Exception as e:
            self.logger.error(f"3자리 확장 중 오류: {e}")
            return []

    def _expand_single_candidate(
        self,
        three_combo: Tuple[int, int, int],
        historical_data: List[LotteryNumber],
        candidate_info: Dict[str, Any],
    ) -> List[ExpansionCandidate]:
        """단일 3자리 후보를 확장"""
        try:
            # 캐시 확인
            cache_key = f"{three_combo}_{len(historical_data)}"
            if cache_key in self.expansion_cache:
                return self.expansion_cache[cache_key]

            expanded_candidates = []

            # 다중 확장 전략 적용
            strategies = ["frequency_based", "pattern_based", "hybrid"]

            for strategy in strategies:
                if strategy in self.expansion_strategies:
                    try:
                        strategy_candidates = self.expansion_strategies[strategy](
                            three_combo, historical_data, candidate_info
                        )
                        expanded_candidates.extend(strategy_candidates)
                    except Exception as e:
                        self.logger.warning(f"확장 전략 {strategy} 실패: {e}")

            # 중복 제거 및 점수 재계산
            unique_candidates = self._deduplicate_candidates(expanded_candidates)

            # 캐시에 저장
            self.expansion_cache[cache_key] = unique_candidates

            return unique_candidates

        except Exception as e:
            self.logger.error(f"단일 후보 확장 중 오류: {e}")
            return []

    def _expand_by_frequency(
        self,
        three_combo: Tuple[int, int, int],
        historical_data: List[LotteryNumber],
        candidate_info: Dict[str, Any],
    ) -> List[ExpansionCandidate]:
        """빈도 기반 확장"""
        try:
            candidates = []
            three_set = set(three_combo)

            # 3자리 조합과 함께 나온 번호들의 빈도 계산
            co_occurrence_freq = defaultdict(int)

            for draw in historical_data:
                draw_set = set(draw.numbers)
                if three_set.issubset(draw_set):
                    remaining = draw_set - three_set
                    for num in remaining:
                        co_occurrence_freq[num] += 1

            # 빈도 기준 상위 번호들 선택
            sorted_freq = sorted(
                co_occurrence_freq.items(), key=lambda x: x[1], reverse=True
            )
            top_numbers = [num for num, freq in sorted_freq[:15]]

            # 상위 번호들로 3자리 조합 생성
            if len(top_numbers) >= 3:
                for remaining_combo in combinations(top_numbers, 3):
                    six_combo = tuple(sorted(three_combo + remaining_combo))

                    # 신뢰도 점수 계산
                    confidence = self._calculate_frequency_confidence(
                        three_combo,
                        remaining_combo,
                        historical_data,
                        co_occurrence_freq,
                    )

                    candidate = ExpansionCandidate(
                        three_digit_combo=three_combo,
                        six_digit_combo=six_combo,
                        confidence_score=confidence,
                        expansion_method="frequency_based",
                        additional_info={
                            "co_occurrence_scores": [
                                co_occurrence_freq[num] for num in remaining_combo
                            ],
                            "base_3digit_score": candidate_info.get(
                                "composite_score", 0
                            ),
                        },
                    )

                    candidates.append(candidate)

            return candidates[:10]  # 상위 10개만 반환

        except Exception as e:
            self.logger.error(f"빈도 기반 확장 중 오류: {e}")
            return []

    def _expand_by_pattern(
        self,
        three_combo: Tuple[int, int, int],
        historical_data: List[LotteryNumber],
        candidate_info: Dict[str, Any],
    ) -> List[ExpansionCandidate]:
        """패턴 기반 확장"""
        try:
            candidates = []
            three_nums = sorted(three_combo)

            # 3자리 조합의 패턴 특성 분석
            pattern_features = self._analyze_pattern_features(three_nums)

            # 균형잡힌 6자리 조합을 위한 후보 번호 선별
            remaining_candidates = []

            for num in range(1, 46):
                if num not in three_combo:
                    # 패턴 균형 점수 계산
                    balance_score = self._calculate_pattern_balance_score(
                        three_nums, num, pattern_features, historical_data
                    )
                    remaining_candidates.append((num, balance_score))

            # 균형 점수 기준 정렬
            remaining_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [num for num, _ in remaining_candidates[:12]]

            # 상위 후보들로 3자리 조합 생성
            for remaining_combo in combinations(top_candidates, 3):
                six_combo = tuple(sorted(three_combo + remaining_combo))

                # 패턴 기반 신뢰도 계산
                confidence = self._calculate_pattern_confidence(
                    three_combo, remaining_combo, pattern_features, historical_data
                )

                candidate = ExpansionCandidate(
                    three_digit_combo=three_combo,
                    six_digit_combo=six_combo,
                    confidence_score=confidence,
                    expansion_method="pattern_based",
                    additional_info={
                        "pattern_balance_scores": [
                            remaining_candidates[i][1]
                            for i, (num, _) in enumerate(remaining_candidates)
                            if num in remaining_combo
                        ],
                        "base_3digit_score": candidate_info.get("composite_score", 0),
                    },
                )

                candidates.append(candidate)

            return candidates[:8]  # 상위 8개만 반환

        except Exception as e:
            self.logger.error(f"패턴 기반 확장 중 오류: {e}")
            return []

    def _expand_by_hybrid(
        self,
        three_combo: Tuple[int, int, int],
        historical_data: List[LotteryNumber],
        candidate_info: Dict[str, Any],
    ) -> List[ExpansionCandidate]:
        """하이브리드 확장 (빈도 + 패턴)"""
        try:
            candidates = []

            # 빈도 기반 후보 생성
            freq_candidates = self._expand_by_frequency(
                three_combo, historical_data, candidate_info
            )

            # 패턴 기반 후보 생성
            pattern_candidates = self._expand_by_pattern(
                three_combo, historical_data, candidate_info
            )

            # 두 방법의 결과를 결합하여 하이브리드 점수 계산
            all_combos = set()
            combo_scores = {}

            # 빈도 기반 점수 수집
            for candidate in freq_candidates:
                combo = candidate.six_digit_combo
                all_combos.add(combo)
                combo_scores[combo] = {
                    "frequency_score": candidate.confidence_score,
                    "pattern_score": 0.0,
                }

            # 패턴 기반 점수 수집
            for candidate in pattern_candidates:
                combo = candidate.six_digit_combo
                all_combos.add(combo)
                if combo not in combo_scores:
                    combo_scores[combo] = {"frequency_score": 0.0, "pattern_score": 0.0}
                combo_scores[combo]["pattern_score"] = candidate.confidence_score

            # 하이브리드 점수 계산
            for combo in all_combos:
                scores = combo_scores[combo]

                # 가중 평균 (빈도 60%, 패턴 40%)
                hybrid_score = (
                    scores["frequency_score"] * 0.6 + scores["pattern_score"] * 0.4
                )

                # 두 방법 모두에서 점수가 있는 경우 보너스
                if scores["frequency_score"] > 0 and scores["pattern_score"] > 0:
                    hybrid_score *= 1.2

                candidate = ExpansionCandidate(
                    three_digit_combo=three_combo,
                    six_digit_combo=combo,
                    confidence_score=hybrid_score,
                    expansion_method="hybrid",
                    additional_info={
                        "frequency_score": scores["frequency_score"],
                        "pattern_score": scores["pattern_score"],
                        "base_3digit_score": candidate_info.get("composite_score", 0),
                    },
                )

                candidates.append(candidate)

            # 하이브리드 점수 기준 정렬
            candidates.sort(key=lambda x: x.confidence_score, reverse=True)

            return candidates[:12]  # 상위 12개만 반환

        except Exception as e:
            self.logger.error(f"하이브리드 확장 중 오류: {e}")
            return []

    def _expand_by_ml(
        self,
        three_combo: Tuple[int, int, int],
        historical_data: List[LotteryNumber],
        candidate_info: Dict[str, Any],
    ) -> List[ExpansionCandidate]:
        """ML 기반 확장 (향후 구현)"""
        self.logger.info("ML 기반 확장은 향후 구현 예정")
        return []

    def _calculate_frequency_confidence(
        self,
        three_combo: Tuple[int, int, int],
        remaining_combo: Tuple[int, int, int],
        historical_data: List[LotteryNumber],
        co_occurrence_freq: Dict[int, int],
    ) -> float:
        """빈도 기반 신뢰도 계산"""
        try:
            # 공출현 빈도 점수
            freq_scores = [co_occurrence_freq[num] for num in remaining_combo]
            avg_freq = sum(freq_scores) / len(freq_scores) if freq_scores else 0

            # 전체 데이터 대비 정규화
            total_draws = len(historical_data)
            normalized_freq = avg_freq / total_draws if total_draws > 0 else 0

            # 추가 보정 요소들
            # 1. 번호 분산 (너무 집중되지 않게)
            six_combo = three_combo + remaining_combo
            number_variance = np.var(six_combo)
            variance_score = min(number_variance / 200, 1.0)  # 정규화

            # 2. 홀짝 균형
            odd_count = sum(1 for num in six_combo if num % 2 == 1)
            balance_score = 1 - abs(odd_count - 3) / 3  # 3:3이 이상적

            # 종합 신뢰도 계산
            confidence = (
                normalized_freq * 0.6 + variance_score * 0.2 + balance_score * 0.2
            )

            return min(confidence, 1.0)

        except Exception as e:
            self.logger.error(f"빈도 신뢰도 계산 중 오류: {e}")
            return 0.0

    def _calculate_pattern_confidence(
        self,
        three_combo: Tuple[int, int, int],
        remaining_combo: Tuple[int, int, int],
        pattern_features: Dict[str, Any],
        historical_data: List[LotteryNumber],
    ) -> float:
        """패턴 기반 신뢰도 계산"""
        try:
            six_combo = sorted(three_combo + remaining_combo)

            # 1. 전체 조합의 균형 점수
            total_balance = self._calculate_total_balance_score(six_combo)

            # 2. 간격 분포 점수
            gaps = [six_combo[i + 1] - six_combo[i] for i in range(len(six_combo) - 1)]
            gap_balance = 1 / (1 + np.std(gaps))  # 간격이 균등할수록 높은 점수

            # 3. 구간 분포 점수
            segment_balance = self._calculate_segment_balance_score(six_combo)

            # 4. 과거 패턴 유사도
            pattern_similarity = self._calculate_pattern_similarity(
                six_combo, historical_data
            )

            # 종합 신뢰도 계산
            confidence = (
                total_balance * 0.3
                + gap_balance * 0.25
                + segment_balance * 0.25
                + pattern_similarity * 0.2
            )

            return min(confidence, 1.0)

        except Exception as e:
            self.logger.error(f"패턴 신뢰도 계산 중 오류: {e}")
            return 0.0

    def _analyze_pattern_features(self, three_nums: List[int]) -> Dict[str, Any]:
        """3자리 패턴 특성 분석"""
        features = {
            "sum": sum(three_nums),
            "range": three_nums[-1] - three_nums[0],
            "gaps": [
                three_nums[i + 1] - three_nums[i] for i in range(len(three_nums) - 1)
            ],
            "odd_count": sum(1 for n in three_nums if n % 2 == 1),
            "segments": self._get_segment_distribution(three_nums),
        }

        features["gap_avg"] = sum(features["gaps"]) / len(features["gaps"])
        features["gap_std"] = np.std(features["gaps"])

        return features

    def _calculate_pattern_balance_score(
        self,
        three_nums: List[int],
        candidate_num: int,
        pattern_features: Dict[str, Any],
        historical_data: List[LotteryNumber],
    ) -> float:
        """패턴 균형 점수 계산"""
        try:
            test_combo = sorted(three_nums + [candidate_num])

            # 1. 전체 합 균형
            total_sum = sum(test_combo)
            expected_sum = 138  # 6자리 평균 합 (1+2+...+45)*6/45 ≈ 138
            sum_balance = 1 / (1 + abs(total_sum - expected_sum) / 50)

            # 2. 분산 균형
            variance = np.var(test_combo)
            variance_balance = min(variance / 200, 1.0)

            # 3. 과거 데이터와의 유사도
            similarity = self._calculate_pattern_similarity(test_combo, historical_data)

            # 종합 점수
            balance_score = (
                sum_balance * 0.4 + variance_balance * 0.3 + similarity * 0.3
            )

            return balance_score

        except Exception as e:
            self.logger.error(f"패턴 균형 점수 계산 중 오류: {e}")
            return 0.0

    def _calculate_total_balance_score(self, six_combo: List[int]) -> float:
        """전체 조합의 균형 점수 계산"""
        try:
            # 1. 합 균형
            total_sum = sum(six_combo)
            expected_sum = 138
            sum_score = 1 / (1 + abs(total_sum - expected_sum) / 50)

            # 2. 분산 균형
            variance = np.var(six_combo)
            variance_score = min(variance / 200, 1.0)

            # 3. 홀짝 균형
            odd_count = sum(1 for num in six_combo if num % 2 == 1)
            odd_even_score = 1 - abs(odd_count - 3) / 3

            return (sum_score + variance_score + odd_even_score) / 3

        except Exception as e:
            self.logger.error(f"전체 균형 점수 계산 중 오류: {e}")
            return 0.0

    def _calculate_segment_balance_score(self, six_combo: List[int]) -> float:
        """구간 분포 균형 점수 계산"""
        try:
            # 3구간 분포 (1-15, 16-30, 31-45)
            segments = [0, 0, 0]

            for num in six_combo:
                if num <= 15:
                    segments[0] += 1
                elif num <= 30:
                    segments[1] += 1
                else:
                    segments[2] += 1

            # 이상적인 분포는 2:2:2
            ideal_dist = [2, 2, 2]
            diff = sum(abs(segments[i] - ideal_dist[i]) for i in range(3))

            return 1 / (1 + diff)

        except Exception as e:
            self.logger.error(f"구간 균형 점수 계산 중 오류: {e}")
            return 0.0

    def _calculate_pattern_similarity(
        self, six_combo: List[int], historical_data: List[LotteryNumber]
    ) -> float:
        """과거 패턴과의 유사도 계산"""
        try:
            if not historical_data:
                return 0.5

            # 최근 100회 데이터와 비교
            recent_data = historical_data[-100:]

            similarities = []
            for draw in recent_data:
                # 공통 번호 개수
                common_count = len(set(six_combo) & set(draw.numbers))
                similarity = common_count / 6
                similarities.append(similarity)

            # 평균 유사도 반환 (너무 유사하지도, 너무 다르지도 않게)
            avg_similarity = (
                sum(similarities) / len(similarities) if similarities else 0
            )

            # 적절한 유사도 범위 (0.1-0.4)에서 높은 점수
            if 0.1 <= avg_similarity <= 0.4:
                return 1.0
            elif avg_similarity < 0.1:
                return avg_similarity / 0.1
            else:
                return max(0, 1 - (avg_similarity - 0.4) / 0.6)

        except Exception as e:
            self.logger.error(f"패턴 유사도 계산 중 오류: {e}")
            return 0.5

    def _get_segment_distribution(self, numbers: List[int]) -> List[int]:
        """구간별 분포 계산"""
        segments = [0, 0, 0]  # 1-15, 16-30, 31-45

        for num in numbers:
            if num <= 15:
                segments[0] += 1
            elif num <= 30:
                segments[1] += 1
            else:
                segments[2] += 1

        return segments

    def _deduplicate_candidates(
        self, candidates: List[ExpansionCandidate]
    ) -> List[ExpansionCandidate]:
        """중복 후보 제거 및 점수 통합"""
        try:
            unique_combos = {}

            for candidate in candidates:
                combo = candidate.six_digit_combo

                if combo not in unique_combos:
                    unique_combos[combo] = candidate
                else:
                    # 더 높은 점수의 후보로 교체
                    if (
                        candidate.confidence_score
                        > unique_combos[combo].confidence_score
                    ):
                        unique_combos[combo] = candidate

            return list(unique_combos.values())

        except Exception as e:
            self.logger.error(f"중복 제거 중 오류: {e}")
            return candidates

    def get_expansion_statistics(self) -> Dict[str, Any]:
        """확장 통계 정보 반환"""
        return {
            "cache_size": len(self.expansion_cache),
            "pattern_cache_size": len(self.pattern_cache),
            "available_strategies": list(self.expansion_strategies.keys()),
            "max_workers": self.max_workers,
        }

    def clear_cache(self):
        """캐시 정리"""
        self.expansion_cache.clear()
        self.pattern_cache.clear()
        self.logger.info("확장 엔진 캐시 정리 완료")

    def shutdown(self):
        """리소스 정리"""
        try:
            self.thread_pool.shutdown(wait=True)
            self.clear_cache()
            self.logger.info("3자리 확장 엔진 종료 완료")
        except Exception as e:
            self.logger.error(f"확장 엔진 종료 중 오류: {e}")
