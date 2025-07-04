"""
향상된 강화학습 모델 학습 모듈

이 모듈은 로또 번호 예측을 위한 향상된 강화학습 모델의 학습을 담당합니다.
기본 RLTrainer를 확장하여 추가 성능 향상 기능을 제공합니다.
"""

import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import gc
from ..utils.unified_logging import get_logger
from ..utils.dynamic_batch_size_utils import get_safe_batch_size
from ..core.state_vector_builder import StateVectorBuilder

# RLTrainer 임포트 (상속)
from .train_rl import RLTrainer

# 로거 설정
logger = get_logger(__name__)


class EnhancedRLTrainer(RLTrainer):
    """향상된 강화학습 모델 학습 클래스"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        train_data: Optional[List[LotteryNumber]] = None,
        val_data: Optional[List[LotteryNumber]] = None,
        pattern_analysis: Optional[PatternAnalysis] = None,
        state_builder: Optional[StateVectorBuilder] = None,
    ):
        """
        향상된 강화학습 트레이너 초기화

        Args:
            config: 학습 설정
            train_data: 훈련 데이터
            val_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과
            state_builder: 상태 벡터 생성기
        """
        # 기본 트레이너 초기화
        super().__init__(config)

        logger.info("향상된 강화학습 트레이너 초기화")

        # 데이터 및 분석 결과 저장
        self.train_data = train_data
        self.val_data = val_data
        self.pattern_analysis = pattern_analysis
        self.state_builder = state_builder or StateVectorBuilder()

        # 추가 설정
        self.config.update(
            {
                "use_amp": self.config.get("use_amp", True),  # 혼합 정밀도 사용
                "use_prioritized_replay": self.config.get(
                    "use_prioritized_replay", True
                ),  # 우선순위 리플레이 사용
                "use_double_dqn": self.config.get(
                    "use_double_dqn", True
                ),  # Double DQN 사용
                "use_dueling_dqn": self.config.get(
                    "use_dueling_dqn", True
                ),  # Dueling DQN 사용
                "use_multi_step": self.config.get(
                    "use_multi_step", True
                ),  # 다중 스텝 학습 사용
                "n_steps": self.config.get("n_steps", 3),  # 다중 스텝 수
                "update_target_freq": self.config.get(
                    "update_target_freq", 10
                ),  # 타겟 네트워크 업데이트 주기
                "validation_freq": self.config.get("validation_freq", 50),  # 검증 주기
                "early_stopping_patience": self.config.get(
                    "early_stopping_patience", 20
                ),  # 조기 종료 인내심
                "early_stopping_min_delta": self.config.get(
                    "early_stopping_min_delta", 0.01
                ),  # 조기 종료 최소 변화량
            }
        )

        # 검증 결과 저장
        self.validation_scores = []
        self.no_improvement_count = 0

        # 결과 기록
        self.best_reward = float("-inf")
        self.best_episode = 0
        self.history = {"rewards": [], "validation": []}

        # 커스텀 모델 경로
        self.model_save_dir = (
            Path(__file__).parent.parent.parent / "savedModels" / "rl" / "enhanced"
        )
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.model_save_dir / "best_model.pt"
        self.latest_model_path = self.model_save_dir / "latest_model.pt"

        # 추가 특성 초기화
        self.env_setup = False
        self._training_env = None
        self.start_time = time.time()

    def _setup_training_env(self, data: List[LotteryNumber]) -> None:
        """
        훈련 환경 설정

        Args:
            data: 학습 데이터
        """
        # 환경 설정
        self.env = self.prepare_environment(data)
        self.env_setup = True

    def train(
        self,
        data: List[LotteryNumber],
        validation_data: Optional[List[LotteryNumber]] = None,
        pattern_analysis: Optional[PatternAnalysis] = None,
        episodes: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        향상된 강화학습 모델 훈련

        Args:
            data: 훈련 데이터
            validation_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과
            episodes: 훈련 에피소드 수 (설정 없으면 config에서 가져옴)
            batch_size: 배치 크기 (설정 없으면 config에서 가져옴)

        Returns:
            Dict[str, Any]: 훈련 결과
        """
        logger.info("향상된 강화학습 모델 훈련 시작")

        # 설정 매개변수 적용
        episodes_count = episodes or self.config.get("episodes", 100)
        batch_size_value = batch_size or self.config.get("batch_size", 32)

        # 훈련 설정
        with self.profiler.profile("train_enhanced_rl"):
            try:
                # 환경 설정
                if not self.env_setup:
                    self._setup_training_env(data)

                # 데이터 설정 - 인자로 전달된 데이터와 초기화 시 설정된 데이터 중 사용 가능한 것 선택
                train_data = data if data is not None else self.train_data
                if train_data is None:
                    raise ValueError("학습 데이터가 제공되지 않았습니다.")

                # 검증 데이터 설정
                val_data = (
                    validation_data if validation_data is not None else self.val_data
                )

                # 학습 수행 - 부모 클래스의 train 메서드를 호출하되 명시적으로 data와 episodes 매개변수 전달
                result = super().train(
                    train_data, episodes=episodes_count, batch_size=batch_size_value
                )

                # 학습 시간
                training_time = time.time() - self.start_time
                logger.info(f"향상된 학습 완료: {training_time:.1f}초")

                # 추가 정보 포함
                result["enhanced"] = True
                result["training_time"] = training_time

                return result
            except Exception as e:
                logger.error(f"학습 중 오류 발생: {e}")
                raise

    def evaluate(self, data: List[LotteryNumber], **kwargs) -> Dict[str, Any]:
        """
        향상된 모델 평가 수행

        Args:
            data: 평가 데이터
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        logger.info(f"향상된 RL 모델 평가 시작: {len(data)} 데이터")
        start_time = time.time()

        # 기본 구현으로 대체
        eval_result = super().evaluate(data)

        # 평가 시간
        eval_time = time.time() - start_time
        logger.info(
            f"향상된 평가 완료: {eval_time:.1f}초, "
            f"성공률: {eval_result.get('success', False)}"
        )

        # 기본 결과에 추가 정보 포함
        eval_result.update(
            {
                "enhanced": True,
                "evaluation_time": eval_time,
                "timestamp": time.time(),
            }
        )

        return eval_result

    def generate_recommendations(
        self, data: List[LotteryNumber], num_sets: int = 5
    ) -> List[Dict[str, Any]]:
        """
        향상된 추천 번호 생성

        Args:
            data: 입력 데이터
            num_sets: 생성할 추천 세트 수

        Returns:
            추천 번호 목록
        """
        logger.info(f"향상된 RL 모델 추천 번호 생성: {num_sets}개 세트")

        # 모델이 None인 경우
        if self.model is None:
            logger.error("모델이 초기화되지 않았습니다")
            return []

        # 예측 수행
        predictions = []
        for _ in range(num_sets):
            # 기본 predict 메서드 호출
            prediction = self.predict(data, count=1)
            if prediction:
                predictions.extend(prediction)

        # 중복 번호 체크 및 필터링
        unique_predictions = []
        seen_numbers = set()

        for pred in predictions:
            if hasattr(pred, "numbers"):
                numbers_tuple = tuple(sorted(pred.numbers))
                if numbers_tuple not in seen_numbers:
                    seen_numbers.add(numbers_tuple)
                    unique_predictions.append(pred)

        # 추천 부족시 추가 생성
        if len(unique_predictions) < num_sets:
            logger.info(
                f"중복 제거 후 {len(unique_predictions)}개 세트만 남음, 추가 생성 필요"
            )

            # 추가 생성
            additional_count = num_sets - len(unique_predictions)
            additional_preds = self.predict(data, count=additional_count * 2)

            # 중복 제거하면서 추가
            for pred in additional_preds:
                if hasattr(pred, "numbers"):
                    numbers_tuple = tuple(sorted(pred.numbers))
                    if (
                        numbers_tuple not in seen_numbers
                        and len(unique_predictions) < num_sets
                    ):
                        seen_numbers.add(numbers_tuple)
                        unique_predictions.append(pred)

        # 결과 로깅
        for i, pred in enumerate(unique_predictions[:num_sets]):
            if hasattr(pred, "numbers") and hasattr(pred, "confidence"):
                logger.info(
                    f"추천 {i+1}: {sorted(pred.numbers)} (신뢰도: {pred.confidence:.2f})"
                )

        return unique_predictions[:num_sets]

    def analyze_recommendations(
        self, recommendations: List[ModelPrediction]
    ) -> Dict[str, Any]:
        """
        생성된 추천 번호에 대한 심층 분석 수행

        Args:
            recommendations: 분석할 추천 번호 목록

        Returns:
            분석 결과
        """
        # 이 메소드는 base 버전에는 없는 향상된 기능
        logger.info(f"{len(recommendations)}개 추천 번호 분석 시작")

        results = {
            "number_frequency": {},  # 각 번호의 출현 빈도
            "sum_stats": {},  # 합계 통계
            "distribution_stats": {},  # 분포 통계
            "odd_even_stats": {},  # 홀짝 통계
            "pattern_analysis": {},  # 패턴 분석
        }

        # 번호 빈도 분석
        all_numbers = []
        for rec in recommendations:
            all_numbers.extend(rec.numbers)

        # 번호 빈도 계산
        for num in range(1, 46):
            results["number_frequency"][num] = all_numbers.count(num)

        # 합계 통계
        sums = [sum(rec.numbers) for rec in recommendations]
        results["sum_stats"] = {
            "min": min(sums),
            "max": max(sums),
            "mean": sum(sums) / len(sums) if sums else 0,
            "values": sums,
        }

        # 홀짝 통계
        odd_counts = [
            sum(1 for n in rec.numbers if n % 2 == 1) for rec in recommendations
        ]
        results["odd_even_stats"] = {
            "odd_counts": odd_counts,
            "even_counts": [6 - odd for odd in odd_counts],
            "avg_odd": sum(odd_counts) / len(odd_counts) if odd_counts else 0,
        }

        # 번호 범위 분포
        ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
        range_counts = {}

        for i, (start, end) in enumerate(ranges):
            range_name = f"{start}-{end}"
            counts = []

            for rec in recommendations:
                count = sum(1 for n in rec.numbers if start <= n <= end)
                counts.append(count)

            range_counts[range_name] = {
                "counts": counts,
                "avg": sum(counts) / len(counts) if counts else 0,
            }

        results["distribution_stats"] = range_counts

        logger.info("추천 번호 분석 완료")
        return results

    def load_model(self, path: str) -> bool:
        """
        향상된 RL 모델 로드

        Args:
            path: 모델 파일 경로

        Returns:
            로드 성공 여부
        """
        try:
            # 부모 클래스의 load_model 메서드 호출
            success = super().load_model(path)

            if success:
                logger.info(f"향상된 RL 모델 로드 완료: {path}")
                return True
            else:
                logger.error(f"향상된 RL 모델 로드 실패: {path}")
                return False

        except Exception as e:
            logger.error(f"향상된 RL 모델 로드 중 오류: {str(e)}")
            return False
