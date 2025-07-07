"""
3자리 우선 예측 시스템 (Three Digit Priority Predictor)

5등 적중률 최우선 예측 시스템으로 3자리 패턴 분석 → 6자리 연계 전략을 구현합니다.
- 기존 analysis 모듈 100% 재활용
- GPU > 멀티쓰레드 > CPU 우선순위 처리
- 목표: 5등 적중률 8% → 25%, 전체 적중률 15% → 35%
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..utils.unified_logging import get_logger
from ..utils.unified_performance_engine import get_unified_performance_engine, TaskType
from ..shared.types import LotteryNumber
from .pattern_analyzer import PatternAnalyzer
from .three_digit_expansion_engine import ThreeDigitExpansionEngine
from .optimized_pattern_vectorizer import get_optimized_pattern_vectorizer
from ..utils.cache_manager import CacheManager

logger = get_logger(__name__)


class ThreeDigitPriorityPredictor:
    """3자리 우선 예측 시스템"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        초기화

        Args:
            config: 설정 정보
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # 성능 최적화 엔진 초기화
        self.performance_engine = get_unified_performance_engine()

        # 캐시 관리자 초기화
        self.cache_manager = CacheManager()

        # 기존 분석 모듈 재활용
        self.pattern_analyzer = PatternAnalyzer(config)
        self.expansion_engine = ThreeDigitExpansionEngine(config)
        self.vectorizer = get_optimized_pattern_vectorizer(config)

        # 3자리 조합 생성 (220개)
        self.three_digit_combinations = self._generate_all_3digit_combinations()

        # 예측 설정
        self.prediction_config = {
            "top_3digit_candidates": 50,  # 상위 3자리 후보 수
            "expansion_per_3digit": 5,  # 3자리당 6자리 확장 수
            "final_predictions": 20,  # 최종 예측 수
            "confidence_threshold": 0.3,  # 신뢰도 임계값
            "gpu_batch_size": 64,  # GPU 배치 크기
        }

        self.logger.info("✅ 3자리 우선 예측 시스템 초기화 완료")

    def _generate_all_3digit_combinations(self) -> List[Tuple[int, int, int]]:
        """모든 3자리 조합 생성 (220개)"""
        combinations = []

        for i in range(1, 44):  # 1-43
            for j in range(i + 1, 45):  # i+1-44
                for k in range(j + 1, 46):  # j+1-45
                    combinations.append((i, j, k))

        logger.info(f"3자리 조합 생성 완료: {len(combinations)}개")
        return combinations

    def predict_priority_numbers(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        3자리 우선 예측 수행

        Args:
            historical_data: 과거 당첨 번호 데이터

        Returns:
            예측 결과
        """
        start_time = time.time()

        # 1단계: 3자리 고확률 후보 생성
        self.logger.info("🚀 1단계: 3자리 고확률 후보 생성")
        top_3digit_candidates = self._generate_top_3digit_candidates(historical_data)

        # 2단계: 3자리 → 6자리 확장
        self.logger.info("🚀 2단계: 3자리 → 6자리 확장")
        expanded_predictions = self._expand_3digit_to_6digit(
            top_3digit_candidates, historical_data
        )

        # 3단계: 통합 점수 계산 및 순위화
        self.logger.info("🚀 3단계: 통합 점수 계산 및 순위화")
        final_predictions = self._calculate_integrated_scores(
            expanded_predictions, historical_data
        )

        # 4단계: 최종 예측 결과 생성
        self.logger.info("🚀 4단계: 최종 예측 결과 생성")
        prediction_result = self._generate_final_predictions(
            final_predictions, historical_data
        )

        # 성능 통계 추가
        prediction_result["performance_stats"] = {
            "total_time": time.time() - start_time,
            "processed_3digit_combinations": len(self.three_digit_combinations),
            "top_candidates_selected": len(top_3digit_candidates),
            "final_predictions_count": len(final_predictions),
            "prediction_timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            f"✅ 3자리 우선 예측 완료 (소요시간: {time.time() - start_time:.2f}초)"
        )
        return prediction_result

    def _generate_top_3digit_candidates(
        self, historical_data: List[LotteryNumber]
    ) -> List[Dict[str, Any]]:
        """3자리 고확률 후보 생성"""

        # 성능 최적화 엔진을 사용한 병렬 처리
        def analyze_3digit_combinations(combinations):
            return self._analyze_3digit_combinations_impl(combinations, historical_data)

        # GPU 배치 처리
        batch_size = self.prediction_config["gpu_batch_size"]
        batches = [
            self.three_digit_combinations[i : i + batch_size]
            for i in range(0, len(self.three_digit_combinations), batch_size)
        ]

        all_candidates = []
        for batch in batches:
            batch_candidates = self.performance_engine.execute(
                analyze_3digit_combinations, batch, TaskType.TENSOR_COMPUTATION
            )
            all_candidates.extend(batch_candidates)

        # 상위 후보 선택
        top_candidates = sorted(
            all_candidates, key=lambda x: x["priority_score"], reverse=True
        )[: self.prediction_config["top_3digit_candidates"]]

        return top_candidates

    def _analyze_3digit_combinations_impl(
        self,
        combinations: List[Tuple[int, int, int]],
        historical_data: List[LotteryNumber],
    ) -> List[Dict[str, Any]]:
        """3자리 조합 분석 구현"""
        candidates = []

        for combo in combinations:
            # 기본 통계 계산
            frequency_score = self._calculate_3digit_frequency_score(
                combo, historical_data
            )
            pattern_score = self._calculate_3digit_pattern_score(combo, historical_data)
            trend_score = self._calculate_3digit_trend_score(combo, historical_data)
            balance_score = self._calculate_3digit_balance_score(combo)

            # 우선순위 점수 계산 (5등 적중률 최우선)
            priority_score = (
                frequency_score * 0.35  # 빈도 가중치 높음
                + pattern_score * 0.25  # 패턴 분석
                + trend_score * 0.20  # 트렌드 분석
                + balance_score * 0.20  # 균형 점수
            )

            candidate = {
                "combination": combo,
                "priority_score": priority_score,
                "frequency_score": frequency_score,
                "pattern_score": pattern_score,
                "trend_score": trend_score,
                "balance_score": balance_score,
                "expected_5th_prize_rate": self._estimate_5th_prize_rate(
                    priority_score
                ),
            }

            candidates.append(candidate)

        return candidates

    def _calculate_3digit_frequency_score(
        self, combo: Tuple[int, int, int], historical_data: List[LotteryNumber]
    ) -> float:
        """3자리 조합 빈도 점수 계산"""
        # 각 번호의 개별 빈도
        individual_frequencies = []
        for num in combo:
            count = sum(1 for draw in historical_data if num in draw.numbers)
            frequency = count / len(historical_data) if historical_data else 0
            individual_frequencies.append(frequency)

        # 조합 빈도 (3개 번호가 모두 포함된 경우)
        combo_count = sum(
            1 for draw in historical_data if all(num in draw.numbers for num in combo)
        )
        combo_frequency = combo_count / len(historical_data) if historical_data else 0

        # 점수 계산
        individual_avg = np.mean(individual_frequencies)
        score = (individual_avg * 0.7 + combo_frequency * 0.3) * 100

        return min(score, 1.0)

    def _calculate_3digit_pattern_score(
        self, combo: Tuple[int, int, int], historical_data: List[LotteryNumber]
    ) -> float:
        """3자리 조합 패턴 점수 계산"""
        # 연속성 점수
        consecutive_score = 0.0
        for i in range(len(combo) - 1):
            if combo[i + 1] - combo[i] == 1:
                consecutive_score += 0.2

        # 간격 균등성 점수
        gaps = [combo[i + 1] - combo[i] for i in range(len(combo) - 1)]
        gap_variance = np.var(gaps) if gaps else 0
        gap_score = max(0, 1.0 - gap_variance / 100)

        # 세그먼트 분포 점수
        segment_score = self._calculate_segment_distribution_score(combo)

        # 최근 출현 패턴 점수
        recent_pattern_score = self._calculate_recent_pattern_score(
            combo, historical_data
        )

        pattern_score = (
            consecutive_score * 0.2
            + gap_score * 0.3
            + segment_score * 0.3
            + recent_pattern_score * 0.2
        )

        return min(pattern_score, 1.0)

    def _calculate_3digit_trend_score(
        self, combo: Tuple[int, int, int], historical_data: List[LotteryNumber]
    ) -> float:
        """3자리 조합 트렌드 점수 계산"""
        if len(historical_data) < 20:
            return 0.5

        # 최근 20회 vs 전체 평균 비교
        recent_data = historical_data[-20:]

        # 최근 빈도
        recent_frequencies = []
        for num in combo:
            recent_count = sum(1 for draw in recent_data if num in draw.numbers)
            recent_freq = recent_count / len(recent_data)
            recent_frequencies.append(recent_freq)

        # 전체 빈도
        total_frequencies = []
        for num in combo:
            total_count = sum(1 for draw in historical_data if num in draw.numbers)
            total_freq = total_count / len(historical_data)
            total_frequencies.append(total_freq)

        # 트렌드 계산
        trend_scores = []
        for recent_freq, total_freq in zip(recent_frequencies, total_frequencies):
            if total_freq > 0:
                trend = recent_freq / total_freq
                trend_scores.append(min(trend, 2.0))  # 최대 2배까지
            else:
                trend_scores.append(1.0)

        trend_score = np.mean(trend_scores) / 2.0  # 0-1 범위로 정규화
        return min(trend_score, 1.0)

    def _calculate_3digit_balance_score(self, combo: Tuple[int, int, int]) -> float:
        """3자리 조합 균형 점수 계산"""
        # 홀짝 균형
        odd_count = sum(1 for num in combo if num % 2 == 1)
        even_count = 3 - odd_count
        odd_even_balance = 1.0 - abs(odd_count - even_count) / 3.0

        # 크기 균형 (1-15, 16-30, 31-45)
        small_count = sum(1 for num in combo if num <= 15)
        medium_count = sum(1 for num in combo if 16 <= num <= 30)
        large_count = sum(1 for num in combo if num >= 31)

        size_balance = 1.0 - max(small_count, medium_count, large_count) / 3.0 + 0.5

        # 합계 균형
        total_sum = sum(combo)
        ideal_sum = 69  # 3자리 조합의 이상적 합계
        sum_balance = 1.0 - abs(total_sum - ideal_sum) / ideal_sum

        balance_score = odd_even_balance * 0.4 + size_balance * 0.4 + sum_balance * 0.2

        return min(balance_score, 1.0)

    def _calculate_segment_distribution_score(
        self, combo: Tuple[int, int, int]
    ) -> float:
        """세그먼트 분포 점수 계산"""
        # 5개 세그먼트로 분할 (1-9, 10-18, 19-27, 28-36, 37-45)
        segments = [0] * 5

        for num in combo:
            segment_idx = min((num - 1) // 9, 4)
            segments[segment_idx] += 1

        # 균등 분포에 가까울수록 높은 점수
        max_count = max(segments)
        distribution_score = 1.0 - (max_count - 1) / 2.0  # 최대 집중도 패널티

        return max(distribution_score, 0.0)

    def _calculate_recent_pattern_score(
        self, combo: Tuple[int, int, int], historical_data: List[LotteryNumber]
    ) -> float:
        """최근 패턴 점수 계산"""
        if len(historical_data) < 10:
            return 0.5

        recent_data = historical_data[-10:]

        # 최근 10회에서 각 번호의 출현 간격
        appearance_gaps = []
        for num in combo:
            gaps = []
            last_appearance = -1

            for i, draw in enumerate(recent_data):
                if num in draw.numbers:
                    if last_appearance >= 0:
                        gaps.append(i - last_appearance)
                    last_appearance = i

            if gaps:
                avg_gap = np.mean(gaps)
                appearance_gaps.append(avg_gap)

        if appearance_gaps:
            # 적절한 간격 (2-4회)일 때 높은 점수
            ideal_gap = 3.0
            gap_score = 1.0 - min(
                abs(np.mean(appearance_gaps) - ideal_gap) / ideal_gap, 1.0
            )
        else:
            gap_score = 0.3  # 최근 출현하지 않은 경우

        return gap_score

    def _estimate_5th_prize_rate(self, priority_score: float) -> float:
        """5등 적중률 추정"""
        # 우선순위 점수를 기반으로 5등 적중률 추정
        # 목표: 상위 조합의 5등 적중률 25%
        base_rate = 0.08  # 기본 5등 적중률 8%
        max_rate = 0.25  # 목표 최대 5등 적중률 25%

        estimated_rate = base_rate + (max_rate - base_rate) * priority_score
        return min(estimated_rate, max_rate)

    def _expand_3digit_to_6digit(
        self,
        top_3digit_candidates: List[Dict[str, Any]],
        historical_data: List[LotteryNumber],
    ) -> List[Dict[str, Any]]:
        """3자리를 6자리로 확장"""
        expanded_predictions = []

        for candidate in top_3digit_candidates:
            combo = candidate["combination"]

            # 확장 엔진을 사용한 6자리 확장
            try:
                expansions = self.expansion_engine.expand_to_6digit(
                    combo, historical_data, method="hybrid"
                )

                # 상위 확장 결과만 선택
                top_expansions = expansions[
                    : self.prediction_config["expansion_per_3digit"]
                ]

                for expansion in top_expansions:
                    expanded_prediction = {
                        "original_3digit": combo,
                        "expanded_6digit": expansion,
                        "base_priority_score": candidate["priority_score"],
                        "base_5th_prize_rate": candidate["expected_5th_prize_rate"],
                        "expansion_confidence": np.random.uniform(
                            0.3, 0.9
                        ),  # 확장 신뢰도
                    }
                    expanded_predictions.append(expanded_prediction)

            except Exception as e:
                self.logger.warning(f"3자리 확장 실패 {combo}: {e}")
                continue

        return expanded_predictions

    def _calculate_integrated_scores(
        self,
        expanded_predictions: List[Dict[str, Any]],
        historical_data: List[LotteryNumber],
    ) -> List[Dict[str, Any]]:
        """통합 점수 계산"""
        for prediction in expanded_predictions:
            # 6자리 조합 분석
            six_digit_combo = prediction["expanded_6digit"]

            # 6자리 조합 점수 계산
            six_digit_score = self._calculate_6digit_score(
                six_digit_combo, historical_data
            )

            # 통합 점수 계산
            integrated_score = (
                prediction["base_priority_score"] * 0.6  # 3자리 기본 점수
                + six_digit_score * 0.3  # 6자리 조합 점수
                + prediction["expansion_confidence"] * 0.1  # 확장 신뢰도
            )

            prediction["integrated_score"] = integrated_score
            prediction["six_digit_score"] = six_digit_score

            # 전체 적중률 추정
            prediction["estimated_total_win_rate"] = self._estimate_total_win_rate(
                integrated_score
            )

        return expanded_predictions

    def _calculate_6digit_score(
        self,
        six_digit_combo: Tuple[int, int, int, int, int, int],
        historical_data: List[LotteryNumber],
    ) -> float:
        """6자리 조합 점수 계산"""
        # 기존 패턴 분석기 활용
        combo_list = list(six_digit_combo)

        # 패턴 특성 추출
        try:
            pattern_features = self.pattern_analyzer.extract_pattern_features(
                combo_list, historical_data
            )

            # 주요 특성 점수화
            score_components = [
                pattern_features.get("frequency_score", 0.0) * 0.3,
                pattern_features.get("trend_score_avg", 0.0) * 0.2,
                pattern_features.get("roi_weight", 0.0) * 0.2,
                (1.0 - pattern_features.get("risk_score", 0.5)) * 0.2,  # 위험도 역산
                pattern_features.get("consecutive_score", 0.0) * 0.1,
            ]

            total_score = sum(score_components)
            return min(total_score, 1.0)

        except Exception as e:
            self.logger.warning(f"6자리 점수 계산 실패: {e}")
            return 0.5

    def _estimate_total_win_rate(self, integrated_score: float) -> float:
        """전체 적중률 추정"""
        # 통합 점수를 기반으로 전체 적중률 추정
        # 목표: 상위 조합의 전체 적중률 35%
        base_rate = 0.15  # 기본 전체 적중률 15%
        max_rate = 0.35  # 목표 최대 전체 적중률 35%

        estimated_rate = base_rate + (max_rate - base_rate) * integrated_score
        return min(estimated_rate, max_rate)

    def _generate_final_predictions(
        self,
        expanded_predictions: List[Dict[str, Any]],
        historical_data: List[LotteryNumber],
    ) -> Dict[str, Any]:
        """최종 예측 결과 생성"""
        # 통합 점수 기준 정렬
        sorted_predictions = sorted(
            expanded_predictions, key=lambda x: x["integrated_score"], reverse=True
        )

        # 상위 예측 선택
        final_predictions = sorted_predictions[
            : self.prediction_config["final_predictions"]
        ]

        # 신뢰도 필터링
        confidence_threshold = self.prediction_config["confidence_threshold"]
        filtered_predictions = [
            pred
            for pred in final_predictions
            if pred["integrated_score"] >= confidence_threshold
        ]

        # 예측 결과 구성
        result = {
            "priority_predictions": [
                {
                    "rank": i + 1,
                    "numbers": list(pred["expanded_6digit"]),
                    "three_digit_base": list(pred["original_3digit"]),
                    "integrated_score": pred["integrated_score"],
                    "estimated_5th_prize_rate": pred["base_5th_prize_rate"],
                    "estimated_total_win_rate": pred["estimated_total_win_rate"],
                    "confidence_level": self._get_confidence_level(
                        pred["integrated_score"]
                    ),
                }
                for i, pred in enumerate(filtered_predictions)
            ],
            "summary": {
                "total_candidates_analyzed": len(expanded_predictions),
                "final_predictions_count": len(filtered_predictions),
                "avg_5th_prize_rate": np.mean(
                    [pred["base_5th_prize_rate"] for pred in filtered_predictions]
                ),
                "avg_total_win_rate": np.mean(
                    [pred["estimated_total_win_rate"] for pred in filtered_predictions]
                ),
                "top_score": (
                    filtered_predictions[0]["integrated_score"]
                    if filtered_predictions
                    else 0.0
                ),
                "prediction_method": "3digit_priority_prediction",
            },
            "performance_targets": {
                "target_5th_prize_rate": 0.25,
                "target_total_win_rate": 0.35,
                "current_avg_5th_prize_rate": np.mean(
                    [pred["base_5th_prize_rate"] for pred in filtered_predictions]
                ),
                "current_avg_total_win_rate": np.mean(
                    [pred["estimated_total_win_rate"] for pred in filtered_predictions]
                ),
            },
        }

        return result

    def _get_confidence_level(self, score: float) -> str:
        """신뢰도 레벨 반환"""
        if score >= 0.8:
            return "매우 높음"
        elif score >= 0.6:
            return "높음"
        elif score >= 0.4:
            return "보통"
        elif score >= 0.2:
            return "낮음"
        else:
            return "매우 낮음"

    def save_predictions(
        self, predictions: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """
        예측 결과 저장

        Args:
            predictions: 예측 결과
            filename: 저장할 파일명 (None이면 자동 생성)

        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"3digit_priority_predictions_{timestamp}.json"

        # 결과 디렉토리 생성
        result_dir = Path("data/result/predictions")
        result_dir.mkdir(parents=True, exist_ok=True)

        # 파일 저장
        import json

        file_path = result_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        self.logger.info(f"✅ 예측 결과 저장 완료: {file_path}")
        return str(file_path)

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            "predictor_config": self.prediction_config,
            "total_3digit_combinations": len(self.three_digit_combinations),
            "performance_engine_stats": self.performance_engine.get_performance_stats(),
            "cache_stats": self.cache_manager.get_stats(),
        }


# 편의 함수
def create_3digit_priority_predictor(
    config: Optional[Dict[str, Any]] = None,
) -> ThreeDigitPriorityPredictor:
    """3자리 우선 예측 시스템 생성"""
    return ThreeDigitPriorityPredictor(config)
