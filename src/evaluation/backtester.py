import os
import json
import time
from typing import List, Dict, Set, Tuple, Optional, Any, Union, cast, TypeVar
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
import hashlib
from collections import defaultdict, Counter

# 상대 경로로 임포트 수정
from ..shared.types import LotteryNumber, ModelPrediction
from ..utils.cache_manager import CacheManager
from ..utils.performance_tracker import PerformanceTracker
from ..utils.error_handler import get_logger
from ..utils.cache_paths import BACKTESTING_CACHE_DIR

# 순환 참조 방지를 위해 타입 힌트에서만 사용
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 타입 체크 시에만 임포트
    from ..core.recommendation_engine import RecommendationEngine

logger = get_logger(__name__)

# 경로 설정
BASE_DIR = Path(__file__).parent.parent.parent  # 프로젝트 루트 디렉토리
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR = LOGS_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# 캐시 디렉토리 통합
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class Backtester:
    """백테스팅 시스템: 추천 조합의 성능을 평가하고 점수를 부여합니다"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        백테스터 초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.performance_tracker = PerformanceTracker()

        # 캐시 매니저 설정 - 캐시 경로 통합
        cache_dir = self.config.get("cache_dir", str(CACHE_DIR))
        self.cache_manager = CacheManager(
            pattern_analyzer=self,  # 자기 자신을 pattern_analyzer로 전달
            cache_dir=cache_dir,
            max_memory_size=50 * 1024 * 1024,  # 50MB
            max_disk_size=200 * 1024 * 1024,  # 200MB
            enable_compression=True,
            default_ttl=30 * 24 * 60 * 60,  # 30일(초 단위)
        )

        # 기본 스코어링 파라미터 설정
        self.scoring_params = {
            "matching_pair_score": 1.0,  # 맞는 쌍 점수
            "range_match_score": 0.5,  # 맞는 범위(구간) 점수
            "hit4_boost_multiplier": 1.2,  # 4개 이상 맞추면 패턴 점수 승수
            "roi_pair_boost": 1.5,  # ROI 프리미엄 쌍 맞추면 추가 점수
            "score_decay_factor": 0.9,  # 세대별 점수 감소율
            "even_odd_pattern_score": 0.8,  # 홀짝 패턴 점수
            "low_high_pattern_score": 0.8,  # 고저 패턴 점수
        }

        # 번호 범위 정의 (1-9, 10-19, 20-29, 30-39, 40-45)
        self.number_ranges = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)]

        # ROI 프리미엄 쌍 (상위 20개 쌍에서 추출, 실제로는 동적 계산 필요)
        # 여기서는 예시로 몇 가지 쌍만 포함
        self.premium_roi_pairs = [
            (3, 24),
            (14, 34),
            (1, 43),
            (7, 19),
            (12, 27),
            (16, 38),
            (21, 29),
            (11, 22),
            (8, 33),
            (4, 25),
        ]

        # 로우 패턴 매치 로그 파일
        log_dir = Path(self.config.get("log_dir", str(LOGS_DIR)))
        eval_dir = log_dir / "eval"
        self.low_match_log_file = eval_dir / "low_pattern_matches.json"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # 반복적으로 실패하는 패턴 관리 파일
        cache_dir_path = Path(cache_dir)
        self.persistent_failures_file = cache_dir_path / "low_pattern_matches.json"
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        # 실패 패턴 로드
        self.persistent_failures = self._load_persistent_failures()

    def get_cached_pattern_scores(self, cache_key: str) -> Optional[Dict[str, float]]:
        """캐시에서 패턴 점수를 가져옵니다."""
        return self.cache_manager.get(cache_key)

    def save_pattern_scores(self, cache_key: str, scores: Dict[str, float]) -> None:
        """패턴 점수를 캐시에 저장합니다."""
        self.cache_manager.set(cache_key, scores)

    def evaluate_patterns(
        self,
        recommender: "RecommendationEngine",
        validation_draws: List[LotteryNumber],
        count_per_draw: int = 50,
        save_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        성능 중심 평가 방법을 사용하여 추천 엔진의 패턴을 평가합니다.

        Args:
            recommender: 추천 엔진
            validation_draws: 검증에 사용될 실제 당첨 번호 리스트
            count_per_draw: 각 추첨당 생성할 추천 조합의 수 (기본값: 50)
            save_summary: 평가 결과를 파일로 저장할지 여부 (기본값: True)

        Returns:
            평가 결과 딕셔너리
        """
        logger.info(f"성능 중심 패턴 평가 시작 (추첨당 {count_per_draw}개 조합)")

        # 평가 결과 초기화
        evaluation_results = {
            "summary": {
                "total_draws": len(validation_draws),
                "avg_best_score": 0.0,
                "avg_avg_score": 0.0,
                "low_match_draws": [],
            },
            "per_draw_scores": {},
        }

        # 모든 추첨에 대한 점수 리스트
        all_best_scores = []
        all_avg_scores = []

        # 저빈도 매치 추첨 추적
        low_match_draws = []

        # 저빈도 매치 추천 조합 상세 추적
        low_pattern_matches = []

        # 반복적으로 실패하는 패턴 추적
        repeated_failures = {}

        # 패턴 실패 추적을 위한 카운터
        pattern_failures = {}

        # 기존 로우 패턴 매치 로그 로드
        persistent_low_matches = self._load_low_match_log()
        new_persistent_low_matches = persistent_low_matches.copy()

        # 각 추첨에 대해 평가
        for draw_index, draw in enumerate(validation_draws):
            draw_id = getattr(draw, "id", draw_index + 1)
            seq_num = getattr(draw, "seq_num", draw_id)

            # 추첨 번호 기록 (로깅용)
            draw_numbers = draw.numbers
            draw_date = getattr(draw, "draw_date", "N/A")

            logger.info(
                f"추첨 {seq_num} 평가 중 ({draw_index+1}/{len(validation_draws)}) - "
                f"번호: {draw_numbers}, 날짜: {draw_date}"
            )

            with self.performance_tracker.track(f"draw_{seq_num}_evaluation"):
                # 고정된 수의 추천 조합 생성
                try:
                    recommendations = []
                    # 추천 엔진으로부터 여러 세트 생성
                    for _ in range(int(count_per_draw / 5) + 1):  # 한 번에 여러 개 생성
                        batch_size = min(5, count_per_draw - len(recommendations))
                        if batch_size <= 0:
                            break

                        batch_recommendations = recommender.recommend(
                            count=batch_size, strategy="hybrid"
                        )
                        if batch_recommendations:
                            # ModelPrediction을 딕셔너리로 변환
                            batch_dict_recommendations = []
                            for rec in batch_recommendations:
                                if hasattr(rec, "numbers"):
                                    # ModelPrediction 객체인 경우
                                    batch_dict_recommendations.append(
                                        {
                                            "numbers": rec.numbers,
                                            "confidence": getattr(
                                                rec, "confidence", 0.5
                                            ),
                                            "source": getattr(
                                                rec, "model_type", "unknown"
                                            ),
                                            "strategy": getattr(
                                                rec, "model_type", "unknown"
                                            ),
                                        }
                                    )
                                else:
                                    # 이미 딕셔너리인 경우
                                    batch_dict_recommendations.append(rec)

                            recommendations.extend(batch_dict_recommendations)

                    if len(recommendations) < count_per_draw:
                        logger.warning(
                            f"목표 {count_per_draw}개 중 {len(recommendations)}개만 생성됨"
                        )
                except Exception as e:
                    logger.error(f"추천 생성 중 오류: {str(e)}")
                    recommendations = []

                # 각 조합을 평가하고 점수 계산
                scores = []
                combo_scores = []  # 각 조합별 점수와 정보 추적

                for idx, combo in enumerate(recommendations):
                    # 안전하게 numbers 필드 접근
                    numbers = None

                    # ModelPrediction 객체 또는 딕셔너리 처리
                    if hasattr(combo, "numbers") and not isinstance(combo, dict):
                        numbers = combo.numbers  # type: ignore
                    else:
                        # 딕셔너리인 경우 key 접근으로 시도
                        try:
                            if isinstance(combo, dict):
                                numbers = combo.get("numbers", [])
                            else:
                                numbers = []
                        except (AttributeError, TypeError):
                            continue

                    if not numbers or len(numbers) != 6:
                        continue

                    # 맞춘 번호 수 계산
                    matches = set(numbers) & set(draw.numbers)
                    hit_count = len(matches)

                    # 패턴 정보 추출
                    even_odd_pattern = self._get_even_odd_pattern(numbers)
                    low_high_pattern = self._get_low_high_pattern(numbers)
                    sum_value = sum(numbers)

                    # 패턴 해시 생성 (고유 식별자로 사용)
                    pattern_hash = self._create_pattern_hash(numbers)

                    # 조합 정보 기록
                    combo_info = {
                        "index": idx,
                        "numbers": numbers,
                        "hit_count": hit_count,
                        "matches": list(matches),
                        "confidence": combo.get("confidence", 0),
                        "source_method": getattr(combo, "model_type", "unknown"),
                        "pattern_hash": pattern_hash,
                    }

                    # 점수 할당
                    score = 0
                    if hit_count == 3:
                        score = 1  # 5등: 1점
                    elif hit_count == 4:
                        score = 10  # 4등: 10점
                    elif hit_count == 5:
                        score = 40  # 3등: 40점
                    elif hit_count == 6:
                        score = 100  # 1등: 100점
                    else:
                        score = 0

                    scores.append(score)
                    combo_info["score"] = score
                    combo_scores.append(combo_info)

                    # 저빈도 매치 검사 (2개 이하로 맞춘 경우)
                    if hit_count < 3:
                        # 패턴 정보 추가
                        low_pattern_matches.append(
                            {
                                "draw_seq": seq_num,
                                "draw_date": str(draw_date),
                                "numbers": numbers,
                                "hit_count": hit_count,
                                "matches": list(matches),
                                "pattern_hash": pattern_hash,
                                "even_odd": even_odd_pattern,
                                "low_high": low_high_pattern,
                            }
                        )

                        # 패턴 실패 추적
                        if pattern_hash in pattern_failures:
                            pattern_failures[pattern_hash] += 1
                        else:
                            pattern_failures[pattern_hash] = 1

                # 추첨별 최고/평균 점수 계산
                best_score = max(scores) if scores else 0
                avg_score = sum(scores) / len(scores) if scores else 0

                # 저빈도 매치 확인 (최고 점수가 1 미만인 경우)
                if best_score < 1:
                    low_match_draws.append(seq_num)

                # 결과 저장
                evaluation_results["per_draw_scores"][seq_num] = {
                    "best_score": best_score,
                    "avg_score": avg_score,
                    "combinations": combo_scores,
                }

                # 점수 통계 업데이트
                all_best_scores.append(best_score)
                all_avg_scores.append(avg_score)

        # 평균 점수 계산
        evaluation_results["summary"]["avg_best_score"] = (
            sum(all_best_scores) / len(all_best_scores) if all_best_scores else 0
        )
        evaluation_results["summary"]["avg_avg_score"] = (
            sum(all_avg_scores) / len(all_avg_scores) if all_avg_scores else 0
        )
        evaluation_results["summary"]["low_match_draws"] = low_match_draws

        # 저성능 패턴 저장
        self._save_low_performance_patterns(pattern_failures)

        # 저빈도 매치 상세 저장
        if low_pattern_matches:
            self._save_low_pattern_match_details(low_pattern_matches)

        # 요약 결과 저장
        if save_summary:
            self._save_evaluation_summary(evaluation_results)

        return evaluation_results

    def _load_low_match_log(self) -> Dict[str, int]:
        """저빈도 매치 로그 로드"""
        if not self.low_match_log_file.exists():
            return {}

        try:
            with open(self.low_match_log_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"저빈도 매치 로그 로드 중 오류: {str(e)}")
            return {}

    def _save_low_match_log(self, low_matches: Dict[str, int]) -> None:
        """저빈도 매치 로그 저장"""
        try:
            with open(self.low_match_log_file, "w") as f:
                json.dump(low_matches, f, indent=2)
        except Exception as e:
            logger.error(f"저빈도 매치 로그 저장 중 오류: {str(e)}")

    def _save_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """평가 결과 요약 저장"""
        try:
            filename = (
                f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            filepath = EVAL_DIR / filename

            with open(filepath, "w") as f:
                # 필요한 요약 정보만 추출
                summary = {
                    "summary": results["summary"],
                    "timestamp": datetime.now().isoformat(),
                }
                json.dump(summary, f, indent=2)

            logger.info(f"평가 요약 저장됨: {filepath}")
        except Exception as e:
            logger.error(f"평가 요약 저장 중 오류: {str(e)}")

    def _load_persistent_failures(self) -> Dict[str, int]:
        """지속적인 실패 패턴 로드"""
        if not self.persistent_failures_file.exists():
            return {}

        try:
            with open(self.persistent_failures_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"지속 실패 패턴 로드 중 오류: {str(e)}")
            return {}

    def _save_persistent_failures(self, failures: Dict[str, int]) -> None:
        """지속적인 실패 패턴 저장"""
        try:
            with open(self.persistent_failures_file, "w") as f:
                json.dump(failures, f, indent=2)
        except Exception as e:
            logger.error(f"지속 실패 패턴 저장 중 오류: {str(e)}")

    def _create_pattern_hash(self, numbers: List[int]) -> str:
        """
        번호 조합의 패턴 해시 생성

        Args:
            numbers: 번호 리스트

        Returns:
            패턴 해시 문자열
        """
        # 정렬된 번호
        sorted_numbers = sorted(numbers)

        # 홀짝 패턴
        even_odd = self._get_even_odd_pattern(sorted_numbers)

        # 고저 패턴
        low_high = self._get_low_high_pattern(sorted_numbers)

        # 합계 범위
        sum_value = sum(sorted_numbers)
        sum_range = sum_value // 10

        # 패턴 문자열 생성 및 해시
        pattern_str = f"{even_odd}_{low_high}_{sum_range}"
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]  # 짧은 해시 사용

    def update_persistent_failures(self, current_failures: Dict[str, int]) -> None:
        """
        지속적 실패 패턴 업데이트

        Args:
            current_failures: 현재 실패 패턴 및 횟수
        """
        try:
            # 기존 실패 로드
            failures = self._load_persistent_failures()

            # 현재 실패 정보 병합
            for pattern, count in current_failures.items():
                if pattern in failures:
                    failures[pattern] += count
                else:
                    failures[pattern] = count

            # 저장
            self._save_persistent_failures(failures)

            # 3회 이상 실패한 패턴 로깅
            high_failure_patterns = {k: v for k, v in failures.items() if v >= 3}
            if high_failure_patterns:
                logger.info(f"3회 이상 실패 패턴: {len(high_failure_patterns)}개")

        except Exception as e:
            logger.error(f"지속 실패 패턴 업데이트 중 오류: {str(e)}")

    def _save_low_pattern_match_details(
        self, low_pattern_matches: List[Dict[str, Any]]
    ) -> None:
        """
        저빈도 매치 패턴 상세 저장

        Args:
            low_pattern_matches: 저빈도 매치 패턴 정보
        """
        try:
            # 파일명 생성
            filename = f"low_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = EVAL_DIR / filename

            # 저장
            with open(filepath, "w") as f:
                json.dump(
                    {
                        "low_matches": low_pattern_matches,
                        "timestamp": datetime.now().isoformat(),
                        "total_count": len(low_pattern_matches),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"저빈도 매치 상세 저장됨: {filepath} ({len(low_pattern_matches)}개)"
            )
        except Exception as e:
            logger.error(f"저빈도 매치 상세 저장 중 오류: {str(e)}")

    def _get_even_odd_pattern(self, numbers: List[int]) -> str:
        """홀짝 패턴 반환"""
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = len(numbers) - odd_count
        return f"{odd_count}_{even_count}"

    def _get_low_high_pattern(self, numbers: List[int]) -> str:
        """고저 패턴 반환"""
        low_count = sum(1 for n in numbers if 1 <= n <= 23)
        high_count = len(numbers) - low_count
        return f"{low_count}_{high_count}"

    def is_duplicate(
        self, numbers: List[int], historical_draws: List[LotteryNumber]
    ) -> bool:
        """과거 당첨 번호와 중복 여부 확인"""
        numbers_set = set(numbers)
        for draw in historical_draws:
            if set(draw.numbers) == numbers_set:
                return True
        return False

    def apply_score_decay(
        self, pattern_scores: Dict[str, float], decay_factor: Optional[float] = None
    ) -> Dict[str, float]:
        """패턴 점수에 감쇠 적용"""
        if decay_factor is None:
            decay_factor = self.scoring_params.get("score_decay_factor", 0.9)

        return {
            pattern: score * decay_factor for pattern, score in pattern_scores.items()
        }

    def run(
        self,
        engine: Optional["RecommendationEngine"] = None,
        training_data: Optional[List[LotteryNumber]] = None,
        test_data: Optional[List[LotteryNumber]] = None,
        model_type: Optional[str] = None,
        recommendations: Optional[List[ModelPrediction]] = None,
        validation_draws: Optional[List[LotteryNumber]] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        추천 엔진을 사용하여 백테스팅 실행

        Args:
            engine: 추천 엔진
            training_data: 학습에 사용할 데이터 (추천 엔진에 제공)
            test_data: 테스트에 사용할 데이터
            model_type: 백테스팅에 사용할 모델 유형 (선택적)
            recommendations: 이미 생성된 추천 조합 목록 (선택적, engine이 없을 경우 사용)
            validation_draws: 이전 API와의 호환성을 위한 테스트 데이터 (선택적)
            cache_key: 캐시 키 (선택적)

        Returns:
            Dict[str, Any]: 평가 결과 (점수 맵)
        """
        # 이전 API와의 호환성 처리
        if validation_draws and not test_data:
            test_data = validation_draws

        # 캐시 확인
        if cache_key:
            cached_result = self.get_cached_pattern_scores(cache_key)
            if cached_result:
                logger.info(f"캐시된 백테스팅 결과 사용: {cache_key}")
                return cached_result

        with self.performance_tracker.track("backtesting"):
            # 추천 엔진을 사용하여 추천 생성
            if engine and training_data and test_data:
                # 추천 개수 설정 (테스트 데이터당 5개)
                count_per_test = self.config.get("recommendations_per_test", 5)
                total_count = len(test_data) * count_per_test

                logger.info(f"추천 엔진을 사용하여 {total_count}개 추천 생성 시작")

                # 학습 데이터로 추천 엔진 초기화
                try:
                    # 모델 유형에 따른 추천 생성
                    recommendations = []

                    # 배치로 추천 생성 (메모리 효율성)
                    batch_size = min(50, total_count)
                    for i in range(0, total_count, batch_size):
                        current_batch = min(batch_size, total_count - i)

                        # 추천 생성
                        batch_recommendations = engine.recommend(
                            count=current_batch,
                            data=training_data,
                            model_types=[model_type] if model_type else None,
                        )

                        if batch_recommendations:
                            recommendations.extend(batch_recommendations)
                except Exception as e:
                    logger.error(f"추천 생성 중 오류: {str(e)}")
                    recommendations = []

            # 오류 처리: 추천 조합이 없는 경우
            if not recommendations:
                logger.error("백테스팅을 위한 추천 조합이 없습니다.")
                return {
                    "error": "추천 조합이 없습니다.",
                    "roi_estimate": -1.0,
                    "avg_score": 0.0,
                    "success": False,  # 성공 여부 필드 추가
                }

            logger.info(
                f"{len(recommendations)}개 추천 조합에 대해 {len(test_data if test_data else [])}개 당첨 번호로 백테스팅 실행"
            )

            # 검증 데이터 설정
            validation_data = (
                test_data if test_data else validation_draws if validation_draws else []
            )
            if not validation_data:
                logger.error("백테스팅을 위한 검증 데이터가 없습니다.")
                return {
                    "error": "검증 데이터가 없습니다.",
                    "roi_estimate": -1.0,
                    "avg_score": 0.0,
                    "success": False,  # 성공 여부 필드 추가
                }

            # 결과 초기화
            result_scores = {
                "pattern_scores": {},
                "hit_counts": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
                "avg_score": 0.0,
                "roi_estimate": 0.0,
                "total_combinations": len(recommendations),
                # 모델별 통계 추가
                "by_model": {},
                "total": {"wins": 0, "roi": 0.0, "matches": 0, "investment": 0},
            }

            # 패턴별 점수 집계
            pattern_scores = {}

            # 모델별 성능 추적
            model_stats = {}

            # 모든 추천 번호 평가
            total_score = 0
            total_hits = 0
            total_investment = 0
            total_roi = 0.0
            total_matches = 0
            total_wins = 0

            for combo in recommendations:
                # ModelPrediction에서 정보 추출
                numbers = combo.numbers if hasattr(combo, "numbers") else []
                model_type = (
                    combo.model_type if hasattr(combo, "model_type") else "unknown"
                )

                # 타당성 검사
                if not numbers or len(numbers) != 6:
                    continue

                # 모델별 통계 초기화
                if model_type not in model_stats:
                    model_stats[model_type] = {
                        "wins": 0,
                        "roi": 0.0,
                        "matches": 0,
                        "investment": 0,
                    }

                # 투자금 계산 (1000원/조합)
                investment = 1000
                total_investment += investment
                model_stats[model_type]["investment"] += investment

                # 최대 맞춘 번호 수
                max_hit_count = 0
                best_score = 0

                # 각 검증 데이터에 대해 평가
                for draw in validation_data:
                    # 맞춘 번호 수 계산
                    matches = set(numbers) & set(draw.numbers)
                    hit_count = len(matches)

                    # 맞춘 횟수 업데이트
                    if hit_count > max_hit_count:
                        max_hit_count = hit_count

                    # 점수 계산
                    if hit_count == 3:
                        score = 1  # 5등: 1점
                    elif hit_count == 4:
                        score = 10  # 4등: 10점
                    elif hit_count == 5:
                        score = 40  # 3등: 40점
                    elif hit_count == 6:
                        score = 100  # 1등: 100점
                    else:
                        score = 0

                    if score > best_score:
                        best_score = score

                # 맞춘 횟수 통계
                result_scores["hit_counts"][max_hit_count] += 1

                # 총 매칭 및 승리 업데이트
                total_matches += max_hit_count
                model_stats[model_type]["matches"] += max_hit_count

                # 당첨 횟수 (3개 이상 맞춘 경우)
                if max_hit_count >= 3:
                    total_wins += 1
                    model_stats[model_type]["wins"] += 1

                # ROI 계산 및 업데이트
                roi_value = best_score - investment
                total_roi += roi_value
                model_stats[model_type]["roi"] += roi_value

                # 총점 누적
                total_score += best_score
                total_hits += max_hit_count

                # 패턴 정보 추출
                pattern_hash = self._create_pattern_hash(numbers)

                # 패턴 점수 누적
                if pattern_hash in pattern_scores:
                    pattern_scores[pattern_hash] += best_score
                else:
                    pattern_scores[pattern_hash] = best_score

            # 평균 점수 계산
            if recommendations:
                avg_score = total_score / len(recommendations)
                avg_hits = total_hits / len(recommendations)
            else:
                avg_score = 0
                avg_hits = 0

            # ROI 추정
            roi_estimate = total_roi / total_investment if total_investment > 0 else 0

            # 결과 업데이트
            result_scores["pattern_scores"] = pattern_scores
            result_scores["avg_score"] = avg_score
            result_scores["roi_estimate"] = roi_estimate
            result_scores["avg_hits"] = avg_hits

            # 전체 통계 업데이트
            result_scores["total"]["wins"] = total_wins
            result_scores["total"]["roi"] = total_roi
            result_scores["total"]["matches"] = total_matches
            result_scores["total"]["investment"] = total_investment

            # 모델별 통계 업데이트
            result_scores["by_model"] = model_stats

            # 실패한 패턴 추적
            pattern_failures = {}
            for combo in recommendations:
                numbers = combo.numbers if hasattr(combo, "numbers") else []

                # 타당성 검사
                if not numbers or len(numbers) != 6:
                    continue

                # 맞춘 번호 수 계산
                max_hit_count = 0
                for draw in validation_data:
                    hit_count = len(set(numbers) & set(draw.numbers))
                    if hit_count > max_hit_count:
                        max_hit_count = hit_count

                # 낮은 히트 카운트는 실패로 간주
                if max_hit_count < 3:
                    pattern_hash = self._create_pattern_hash(numbers)
                    if pattern_hash in pattern_failures:
                        pattern_failures[pattern_hash] += 1
                    else:
                        pattern_failures[pattern_hash] = 1

            # 지속적 실패 패턴 업데이트
            self.update_persistent_failures(pattern_failures)

            # 저성능 패턴 저장
            self._save_low_performance_patterns(pattern_failures)

            # 결과 캐싱
            if cache_key:
                self.save_pattern_scores(cache_key, result_scores)

            # 결과 로깅
            logger.info(
                f"백테스팅 결과: 평균 점수={avg_score:.2f}, "
                f"평균 맞춘 수={avg_hits:.2f}, "
                f"ROI 추정={roi_estimate:.4f}, "
                f"3개 이상 맞춘 비율={sum(result_scores['hit_counts'][i] for i in range(3, 7))/max(1, len(recommendations)):.2%}"
            )

            return result_scores

    def _save_strategy_stats(
        self, strategy_performance: Dict[str, Dict[str, float]]
    ) -> None:
        """
        전략별 성능 통계를 저장합니다.

        Args:
            strategy_performance: 전략별 성능 통계
        """
        try:
            # 저장 경로 설정 (Path 객체 사용)
            performance_dir = BASE_DIR / "data" / "performance"
            performance_dir.mkdir(parents=True, exist_ok=True)
            output_file = performance_dir / "strategy_stats.json"

            # 기존 데이터 로드 (있는 경우)
            existing_stats = {}
            if output_file.exists():
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing_stats = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"기존 전략 통계 파일 손상됨: {output_file}")

            # 새 데이터로 기존 데이터 업데이트
            for strategy, stats in strategy_performance.items():
                if strategy not in existing_stats:
                    existing_stats[strategy] = stats
                else:
                    # 기존 데이터와 새 데이터의 가중 평균 계산
                    old_count = existing_stats[strategy].get("count", 1)
                    new_count = stats.get("count", 1)
                    total_count = old_count + new_count

                    # 가중 평균 계산
                    existing_stats[strategy]["avg_match"] = (
                        existing_stats[strategy]["avg_match"] * old_count
                        + stats["avg_match"] * new_count
                    ) / total_count

                    existing_stats[strategy]["avg_roi"] = (
                        existing_stats[strategy]["avg_roi"] * old_count
                        + stats["avg_roi"] * new_count
                    ) / total_count

                    # 총 카운트 업데이트
                    existing_stats[strategy]["count"] = total_count

            # 업데이트된 통계 저장
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing_stats, f, ensure_ascii=False, indent=2)

            logger.info(f"전략별 성능 통계가 {output_file}에 저장되었습니다.")

        except Exception as e:
            logger.error(f"전략 통계 저장 중 오류: {str(e)}")

    def _save_low_performance_patterns(self, pattern_failures: Dict[str, int]) -> None:
        """저성능 패턴 저장"""
        try:
            # 기존 데이터 로드
            existing_patterns = self._load_persistent_failures()

            # 업데이트
            for pattern, count in pattern_failures.items():
                if pattern in existing_patterns:
                    existing_patterns[pattern] += count
                else:
                    existing_patterns[pattern] = count

            # 저장
            self._save_persistent_failures(existing_patterns)

            # 특정 임계값 이상 실패한 패턴 로깅
            high_failures = {k: v for k, v in existing_patterns.items() if v >= 3}
            logger.info(f"3회 이상 실패 패턴: {len(high_failures)}개")

        except Exception as e:
            logger.error(f"저성능 패턴 저장 중 오류: {str(e)}")

    def calculate_match_count(
        self, recommended: List[int], actual: List[int]
    ) -> Dict[str, Any]:
        """
        추천 번호와 실제 당첨 번호의 일치 개수 계산

        Args:
            recommended: 추천 번호 리스트 (6개)
            actual: 실제 당첨 번호 리스트 (6개)

        Returns:
            매칭 결과 정보
        """
        if not recommended or not actual:
            return {"match_count": 0, "rank": 0}

        # 입력 데이터 정리 및 검증
        recommended = sorted([int(n) for n in recommended if 1 <= int(n) <= 45])[:6]

        # 당첨 번호
        actual_main = sorted([int(n) for n in actual if 1 <= int(n) <= 45])

        # 매칭 개수 계산
        match_count = len(set(recommended) & set(actual_main))

        # 등수 계산
        rank = self._calculate_rank(match_count)

        return {
            "match_count": match_count,
            "rank": rank,
            "recommended": recommended,
            "actual": actual_main,
        }

    def _calculate_rank(self, match_count: int) -> int:
        """
        매칭 개수에 따른 등수 계산

        Args:
            match_count: 일치하는 번호 개수

        Returns:
            등수 (1-5, 0은 미당첨)
        """
        if match_count == 6:
            return 1  # 1등
        elif match_count == 5:
            return 3  # 3등
        elif match_count == 4:
            return 4  # 4등
        elif match_count == 3:
            return 5  # 5등
        else:
            return 0  # 미당첨

    def evaluate_patterns_legacy(
        self,
        recommended_combinations: List[List[int]],
        validation_draws: List[LotteryNumber],
        apply_decay: bool = False,
        previous_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        레거시 패턴 평가 메서드 (이전 버전과의 호환성용)

        Args:
            recommended_combinations: 추천 번호 조합 리스트
            validation_draws: 검증용 당첨 번호 리스트
            apply_decay: 감쇠 적용 여부
            previous_scores: 이전 점수 맵

        Returns:
            Dict[str, float]: 패턴 점수 맵
        """
        # 이전 점수 복사 (제공된 경우)
        pattern_scores = {}
        if previous_scores:
            pattern_scores = previous_scores.copy()

        # 감쇠 적용 (이전 점수의 중요도 감소)
        if apply_decay and pattern_scores:
            pattern_scores = self.apply_score_decay(pattern_scores)

        # 각 추천 조합 평가
        for numbers in recommended_combinations:
            if not numbers or len(numbers) != 6:
                continue

            # 정렬된 번호
            sorted_numbers = sorted(numbers)

            # 패턴 해시 생성
            pattern_hash = self._create_pattern_hash(sorted_numbers)

            # 최대 점수 및 히트 카운트 추적
            max_score = 0
            max_hit_count = 0

            # 각 검증 데이터에 대해 평가
            for draw in validation_draws:
                # 기본 히트 점수
                hit_count = len(set(sorted_numbers) & set(draw.numbers))
                score = 0

                # 점수 할당
                if hit_count == 3:
                    score = 1  # 5등: 1점
                elif hit_count == 4:
                    score = 10  # 4등: 10점
                elif hit_count == 5:
                    score = 40  # 3등: 40점
                elif hit_count == 6:
                    score = 100  # 1등: 100점

                # 추가 점수: 맞는 쌍 평가
                matching_pairs = self._evaluate_matching_pairs(
                    sorted_numbers, draw.numbers
                )
                pair_score = (
                    len(matching_pairs) * self.scoring_params["matching_pair_score"]
                )

                # 추가 점수: 맞는 구간 평가
                matching_ranges = self._evaluate_matching_ranges(
                    sorted_numbers, draw.numbers
                )
                range_score = (
                    len(matching_ranges) * self.scoring_params["range_match_score"]
                )

                # 추가 점수: ROI 프리미엄 쌍 평가
                roi_pair_count = sum(
                    1 for pair in matching_pairs if pair in self.premium_roi_pairs
                )
                roi_pair_score = roi_pair_count * self.scoring_params["roi_pair_boost"]

                # 총점 계산
                total_score = score + pair_score + range_score + roi_pair_score

                # 4개 이상 맞춘 경우 패턴 점수 승수 적용
                if hit_count >= 4:
                    total_score *= self.scoring_params["hit4_boost_multiplier"]

                # 최대 점수 업데이트
                if total_score > max_score:
                    max_score = total_score

                # 최대 히트 카운트 업데이트
                if hit_count > max_hit_count:
                    max_hit_count = hit_count

            # 패턴 점수 업데이트
            if pattern_hash in pattern_scores:
                pattern_scores[pattern_hash] = (
                    pattern_scores[pattern_hash] + max_score
                ) / 2
            else:
                pattern_scores[pattern_hash] = max_score

            # 낮은 히트 카운트 패턴 추적
            if max_hit_count < 3:
                self._track_low_performing_pattern(sorted_numbers, max_hit_count)

        return pattern_scores

    def _evaluate_matching_pairs(
        self, combination: List[int], draw_numbers: List[int]
    ) -> List[Tuple[int, int]]:
        """
        일치하는 번호 쌍 평가

        Args:
            combination: 추천 번호 조합
            draw_numbers: 당첨 번호

        Returns:
            List[Tuple[int, int]]: 일치하는 번호 쌍 목록
        """
        # 두 번호 리스트 정렬
        sorted_combo = sorted(combination)
        sorted_draw = sorted(draw_numbers)

        # 일치하는 쌍 찾기
        matching_pairs = []
        for i in range(len(sorted_combo) - 1):
            for j in range(i + 1, len(sorted_combo)):
                pair = (sorted_combo[i], sorted_combo[j])

                # 당첨 번호에 이 쌍이 있는지 확인
                if sorted_combo[i] in sorted_draw and sorted_combo[j] in sorted_draw:
                    matching_pairs.append(pair)

        return matching_pairs

    def _evaluate_matching_ranges(
        self, combination: List[int], draw_numbers: List[int]
    ) -> List[int]:
        """
        일치하는 번호 범위 평가

        Args:
            combination: 추천 번호 조합
            draw_numbers: 당첨 번호

        Returns:
            List[int]: 일치하는 범위 인덱스 목록
        """
        # 추천 조합의 범위별 카운트
        combo_range_counts = [0] * len(self.number_ranges)
        for num in combination:
            for i, (start, end) in enumerate(self.number_ranges):
                if start <= num <= end:
                    combo_range_counts[i] += 1
                    break

        # 당첨 번호의 범위별 카운트
        draw_range_counts = [0] * len(self.number_ranges)
        for num in draw_numbers:
            for i, (start, end) in enumerate(self.number_ranges):
                if start <= num <= end:
                    draw_range_counts[i] += 1
                    break

        # 일치하는 범위 찾기
        matching_ranges = []
        for i in range(len(self.number_ranges)):
            # 동일한 범위에 동일한 개수의 번호가 있는 경우
            if (
                combo_range_counts[i] > 0
                and combo_range_counts[i] == draw_range_counts[i]
            ):
                matching_ranges.append(i)

        return matching_ranges

    def _track_low_performing_pattern(self, numbers: List[int], hit_count: int) -> None:
        """
        저성능 패턴 추적

        Args:
            numbers: 번호 조합
            hit_count: 맞춘 번호 수
        """
        try:
            # 패턴 해시 생성
            pattern_hash = self._create_pattern_hash(numbers)

            # 실패 카운트 증가
            failures = self._load_persistent_failures()

            if pattern_hash in failures:
                failures[pattern_hash] += 1
            else:
                failures[pattern_hash] = 1

            # 저장
            self._save_persistent_failures(failures)

        except Exception as e:
            logger.error(f"저성능 패턴 추적 중 오류: {str(e)}")

    # 레거시 메서드 별칭 (호환성 유지)
    run_backtesting = run
