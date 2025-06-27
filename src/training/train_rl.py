# type: ignore
"""
강화학습 모델 훈련 스크립트 (RL Model Training)

이 모듈은 로또 번호 예측을 위한 강화학습 모델을 훈련하는 스크립트입니다.
"""

import os
import sys
import torch
import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import argparse
from datetime import datetime
import gc

# 상대 경로 임포트 설정
from ..models.rl_model import RLModel
from ..shared.types import LotteryNumber, PatternAnalysis, ModelPrediction
from ..utils.data_loader import load_draw_history, DataLoader
from ..utils.error_handler_refactored import get_logger
from ..utils.profiler import Profiler, ProfilerConfig
from ..utils.pattern_filter import get_pattern_filter
from ..utils.cuda_optimizers import AMPTrainer
from ..utils.batch_controller import DynamicBatchSizeController
from ..utils.cache_manager import ThreadLocalCache
from ..utils.performance_utils import MemoryTracker
from ..utils.config_loader import ConfigProxy

# 로거 설정
logger = get_logger(__name__)


# LotteryDataset 클래스
class LotteryDataset:
    """로또 데이터셋 (간단한 구현)"""

    def __init__(self, data_path: str):
        """
        초기화

        Args:
            data_path: 데이터 파일 경로
        """
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self) -> List[LotteryNumber]:
        """데이터 로드"""
        try:
            return load_draw_history(file_path=self.data_path)
        except Exception as e:
            logger.error(f"데이터 로드 중 오류: {str(e)}")
            return []

    def __len__(self) -> int:
        """데이터셋 길이 반환"""
        return len(self.data)

    def get_all_data(self) -> List[LotteryNumber]:
        """모든 데이터 반환"""
        return self.data

    def get_lottery_numbers(self) -> List[LotteryNumber]:
        """
        모든 로또 번호 데이터 반환

        Returns:
            로또 번호 데이터 리스트
        """
        return self.data

    def get_batch(self, batch_size: int) -> List[LotteryNumber]:
        """배치 데이터 반환"""
        if batch_size >= len(self.data):
            return self.data
        indices = np.random.choice(len(self.data), batch_size, replace=False)
        return [self.data[i] for i in indices]


class RLTrainer:
    """강화학습 모델 훈련기"""

    def __init__(self, config):
        """
        강화학습 모델 훈련기 초기화

        Args:
            config: 훈련 설정
        """
        # 설정 초기화
        if isinstance(config, dict):
            self.config = ConfigProxy(config)
        else:
            self.config = config

        # 기기 설정
        use_cuda = self.get_config_value("use_cuda", True) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # 추가 특성 초기화
        self.setup_env = False
        self.env = None

        # 랜덤 시드 설정
        seed_value = self.get_config_value("seed", 42)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)  # type: ignore
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)  # type: ignore

        # 모델 및 데이터셋 초기화
        self.model = None
        self.dataset = None

        # 훈련 메트릭
        self.train_losses = []
        self.valid_metrics = []
        self.best_metric = float("inf")
        self.best_epoch = 0

        # 프로파일러 설정
        profiler_config = ConfigProxy(
            {
                "enable_cpu_profiling": True,
                "enable_memory_profiling": True,
                "enable_cuda_profiling": use_cuda,
            }
        )
        self.profiler = Profiler(profiler_config)

        # 모델 디렉토리 생성
        model_dir = self.get_config_value("model_dir", "savedModels")
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)  # type: ignore

        # 배치 크기 설정
        self.batch_size = self.get_config_value("batch_size", 32)

        # 패턴 필터 초기화
        self.pattern_filter = get_pattern_filter()

        self.amp_trainer = AMPTrainer(self.config)
        self.batch_controller = DynamicBatchSizeController(
            config=self.config,
            initial_batch_size=int(self.get_config_value("batch_size", 32)),  # type: ignore
        )
        self.cache = ThreadLocalCache()
        self.memory_tracker = MemoryTracker()
        self.dataloader = None
        self.loss_fn = torch.nn.MSELoss()

        logger.info(f"RLTrainer 초기화 완료 - 장치: {self.device}")

    def get_config_value(self, key, default_value=None):
        """
        설정에서 값을 안전하게 가져오는 헬퍼 메서드

        Args:
            key: 설정 키
            default_value: 기본값

        Returns:
            설정 값 또는 기본값
        """
        return self.config.safe_get(key, default_value)

    def to_gpu(self):
        """모델을 GPU로 이동"""
        if self.model is not None and torch.cuda.is_available():
            self.model.to(self.device)
            logger.info(f"모델을 {self.device}로 이동했습니다.")

    def load_data(self, data_path: str) -> None:
        """
        데이터 로드

        Args:
            data_path: 데이터 파일 경로
        """
        try:
            # 간소화된 LotteryDataset 사용
            self.dataset = LotteryDataset(data_path)
            logger.info(f"데이터 로드 완료 - {len(self.dataset)} 항목")

            # 모델 초기화
            self.model = RLModel(
                {
                    "learning_rate": self.config.safe_get("learning_rate", 0.001),
                    "batch_size": self.config.safe_get("batch_size", 32),
                    "experience_replay": self.config.safe_get(
                        "experience_replay", True
                    ),
                    "max_memory": self.config.safe_get("max_memory", 10000),
                    "use_amp": self.config.safe_get("use_amp", True),
                }
            )

            # GPU 설정
            if self.device.type == "cuda":
                self.to_gpu()

            logger.info("RL 모델 초기화 완료")

        except Exception as e:
            logger.error(f"데이터 로드 중 오류: {str(e)}")
            raise

    def prepare_environment(self, data: List[LotteryNumber]) -> Any:
        """
        훈련 환경 준비

        Args:
            data: 학습에 사용할 데이터

        Returns:
            훈련 환경 객체
        """
        # 환경 설정 표시
        self.setup_env = True

        # 임시 환경 객체 (예시)
        env = {"data": data}

        # 환경 반환
        return env

    def train(
        self,
        data: List[LotteryNumber],
        validation_data: Optional[List[LotteryNumber]] = None,
        valid_data: Optional[List[LotteryNumber]] = None,
        pattern_analysis: Optional[PatternAnalysis] = None,
        episodes: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        모델 훈련

        Args:
            data: 훈련 데이터
            validation_data: 검증 데이터 (valid_data와 동일)
            valid_data: 검증 데이터 (validation_data와 동일, 호환성용)
            pattern_analysis: 패턴 분석 결과
            episodes: 훈련 에피소드 수 (선택적)
            batch_size: 배치 크기 (선택적)

        Returns:
            훈련 결과
        """
        # 로거 초기화 (없을 경우)
        if not hasattr(self, "logger"):
            from ..utils.error_handler_refactored import get_logger

            self.logger = get_logger(__name__)

        self.logger.info("강화학습 훈련 시작")

        try:
            self.memory_tracker.start()

            with self.profiler.profile("rl_training"):
                # 환경 초기화
                env = self.prepare_environment(data)

                # 훈련 파라미터
                episodes_count = episodes or self.get_config_value("num_episodes", 1000)
                batch_size_value = batch_size or self.get_config_value("batch_size", 32)

                # 학습 기록
                losses = []
                valid_losses = []
                best_valid_loss = float("inf")
                patience_counter = 0

                # 훈련 루프
                for episode in range(episodes_count):
                    episode_loss = 0.0
                    num_batches = 0

                    # 배치 처리
                    for i in range(0, len(data), batch_size_value):
                        batch_data = data[i : i + batch_size_value]

                        # 배치 크기 동적 조정
                        batch_size_value = self.batch_controller.adjust_batch_size(
                            self.dataloader
                        )

                        # AMP를 사용한 학습 스텝
                        metrics = self.amp_trainer.train_step(
                            self.model,
                            self.model.optimizer,
                            self.loss_fn,
                            {"data": batch_data, "env": env},
                        )

                        episode_loss += metrics["loss"]
                        num_batches += 1

                    # 에피소드 평균 손실
                    avg_loss = episode_loss / num_batches if num_batches > 0 else 0.0
                    losses.append(avg_loss)

                    # 검증
                    if validation_data:
                        valid_loss = self.amp_trainer.evaluate(
                            self.model,
                            self.loss_fn,
                            self._create_dataloader(validation_data, env),
                        )["loss"]
                        valid_losses.append(valid_loss)

                        # 조기 종료 체크
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            patience_counter = 0
                            self._save_model()
                        else:
                            patience_counter += 1
                            if patience_counter >= self.get_config_value(
                                "patience", 10
                            ):
                                logger.info(f"조기 종료: {episode + 1} 에피소드")
                                break

                    logger.info(
                        f"에피소드 {episode + 1}/{episodes_count} - 훈련 손실: {avg_loss:.4f}"
                    )

                return {
                    "losses": losses,
                    "valid_losses": valid_losses,
                    "best_valid_loss": best_valid_loss,
                    "memory_usage": self.memory_tracker.get_memory_log(),
                }

        except Exception as e:
            logger.error(f"RL 훈련 중 오류 발생: {str(e)}")
            return {"error": str(e)}

        finally:
            self.memory_tracker.stop()

    def evaluate(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        모델 평가

        Args:
            data: 평가 데이터

        Returns:
            평가 결과
        """
        if self.model is None:
            logger.error("모델이 초기화되지 않았습니다")
            return {"error": "모델이 초기화되지 않음"}

        logger.info(f"RL 모델 평가 시작 - 데이터 크기: {len(data)}")

        try:
            # 평가 수행
            with self.profiler.profile("evaluate_rl_model"):
                # 평가 결과는 모델이 정의한 형태로 반환
                eval_result = {
                    "loss": 0.0,
                    "hit_rate": 0.0,
                    "avg_matches": 0.0,
                    "success": True,
                }

                # 여기서 실제 모델 평가 수행
                # 구현이 필요함

            return eval_result

        except Exception as e:
            logger.error(f"평가 중 오류 발생: {str(e)}")
            return {"error": str(e), "success": False}

    def predict(
        self, data: List[LotteryNumber], count: int = 1
    ) -> List[ModelPrediction]:
        """
        번호 예측

        Args:
            data: 예측에 사용할 데이터
            count: 생성할 예측 수

        Returns:
            예측 결과 목록 (ModelPrediction 객체 리스트)
        """
        if not self.model:
            logger.error("모델이 초기화되지 않았습니다.")
            return []

        try:
            # 모델을 평가 모드로 설정
            self.model.eval()  # type: ignore

            # 예측 실행
            with torch.no_grad():
                predictions = []

                for _ in range(count):
                    # 모델을 사용한 번호 예측
                    result = self.model.predict(data)  # type: ignore

                    if isinstance(result, ModelPrediction):
                        predictions.append(result)
                    elif isinstance(result, dict):
                        # 딕셔너리를 ModelPrediction으로 변환
                        numbers = (
                            result.get("numbers", [])
                            if isinstance(result, dict)
                            else []
                        )
                        confidence = (
                            result.get("confidence", 0.5)
                            if isinstance(result, dict)
                            else 0.5
                        )
                        prediction = ModelPrediction(
                            numbers=sorted(numbers),
                            confidence=float(confidence),
                            model_type="rl",
                        )
                        predictions.append(prediction)
                    else:
                        # 기타 형식 처리
                        if hasattr(result, "numbers"):
                            numbers = getattr(result, "numbers")
                            confidence = getattr(result, "confidence", 0.5)
                            prediction = ModelPrediction(
                                numbers=sorted(numbers),
                                confidence=float(confidence),
                                model_type="rl",
                            )
                            predictions.append(prediction)

                return predictions

        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def generate_recommendations(
        self, num_sets: int = 5, data: Optional[List[LotteryNumber]] = None
    ) -> List[List[int]]:
        """
        추천 번호 생성

        Args:
            num_sets: 생성할 번호 세트 수
            data: 참조할 과거 데이터 (없으면 self.dataset 사용)

        Returns:
            추천 번호 목록 (6개 숫자의 리스트 형태)
        """
        if self.model is None:
            logger.error("모델이 초기화되지 않았습니다")
            # 오류 시 무작위 번호 생성
            import random

            return [sorted(random.sample(range(1, 46), 6)) for _ in range(num_sets)]

        logger.info(f"{num_sets}개의 번호 세트 생성 중...")

        try:
            # 모델을 평가 모드로 설정 (가능한 경우)
            try:
                self.model.eval()  # type: ignore
            except Exception as e:
                logger.warning(f"모델을 평가 모드로 설정 중 오류: {str(e)}")

            # 데이터 확인
            reference_data = data if data is not None else []

            # 데이터셋이 있으면 사용
            if not reference_data and self.dataset is not None:
                # 데이터셋에 get_lottery_numbers 메서드가 있는지 확인
                if hasattr(self.dataset, "get_lottery_numbers"):
                    reference_data = self.dataset.get_lottery_numbers()
                elif hasattr(self.dataset, "get_all_data"):
                    reference_data = self.dataset.get_all_data()
                else:
                    reference_data = (
                        self.dataset.data if hasattr(self.dataset, "data") else []
                    )

            # 데이터가 없으면 오류
            if not reference_data:
                logger.error("참조할 데이터가 없습니다")
                raise ValueError("참조할 데이터가 없음")

            # 모델로부터 추천 생성
            recommendations = []
            for _ in range(num_sets):
                # 예측 수행
                prediction = self.model.predict(reference_data)

                # 예측 결과 처리 - 안전한 방식으로 타입 처리
                try:
                    # 예측 결과가 ModelPrediction 타입인 경우
                    if hasattr(prediction, "numbers"):
                        numbers = prediction.numbers  # type: ignore
                        if isinstance(numbers, (list, tuple)) and all(
                            isinstance(n, int) for n in numbers
                        ):
                            recommendations.append(sorted(numbers))
                    # 예측 결과가 정수 리스트인 경우
                    elif (
                        isinstance(prediction, list)
                        and prediction
                        and all(isinstance(n, int) for n in prediction)
                    ):
                        recommendations.append(sorted(prediction))  # type: ignore
                    # 예측 결과가 ModelPrediction 리스트인 경우
                    elif (
                        isinstance(prediction, list)
                        and prediction
                        and isinstance(prediction[0], object)
                        and hasattr(prediction[0], "numbers")
                    ):
                        nums = prediction[0].numbers  # type: ignore
                        if isinstance(nums, (list, tuple)) and all(
                            isinstance(n, int) for n in nums
                        ):
                            recommendations.append(sorted(nums))
                    # 예측 결과가 딕셔너리인 경우
                    elif isinstance(prediction, dict) and "numbers" in prediction:
                        nums = prediction.get("numbers", [])  # type: ignore
                        if isinstance(nums, (list, tuple)) and all(
                            isinstance(n, int) for n in nums
                        ):
                            recommendations.append(sorted(nums))
                    else:
                        logger.warning(
                            f"지원되지 않는 예측 결과 형식: {type(prediction)}"
                        )
                except Exception as e:
                    logger.error(f"예측 결과 처리 중 오류: {str(e)}")

            # 중복 제거
            unique_recommendations = []
            seen = set()
            for rec in recommendations:
                if rec:  # 빈 리스트가 아닌 경우만 처리
                    rec_tuple = tuple(sorted(rec))
                    if rec_tuple not in seen:
                        seen.add(rec_tuple)
                        unique_recommendations.append(list(rec_tuple))

            # 필요한 만큼 추가 번호 생성
            while len(unique_recommendations) < num_sets:
                # 추가 예측
                prediction = self.model.predict(reference_data)
                numbers = None

                # 예측 결과에서 번호 추출 - 안전한 방식으로
                try:
                    if hasattr(prediction, "numbers"):
                        numbers = prediction.numbers  # type: ignore
                    elif (
                        isinstance(prediction, list)
                        and prediction
                        and all(isinstance(n, int) for n in prediction)
                    ):
                        numbers = prediction
                    elif (
                        isinstance(prediction, list)
                        and prediction
                        and isinstance(prediction[0], object)
                        and hasattr(prediction[0], "numbers")
                    ):
                        numbers = prediction[0].numbers  # type: ignore
                    elif isinstance(prediction, dict) and "numbers" in prediction:
                        numbers = prediction.get("numbers", [])  # type: ignore
                except Exception as e:
                    logger.error(f"번호 추출 중 오류: {str(e)}")

                # 중복 체크 및 추가
                if numbers:
                    try:
                        new_rec = tuple(sorted(numbers))  # type: ignore
                        if new_rec not in seen:
                            seen.add(new_rec)
                            unique_recommendations.append(list(new_rec))
                    except Exception as e:
                        logger.error(f"번호 정렬 중 오류: {str(e)}")

            logger.info(f"{len(unique_recommendations)}개의 추천 번호 생성 완료")
            return unique_recommendations[:num_sets]

        except Exception as e:
            logger.error(f"추천 번호 생성 중 오류: {str(e)}")
            # 오류 시 무작위 번호 생성
            import random

            return [sorted(random.sample(range(1, 46), 6)) for _ in range(num_sets)]

    def _create_state_vector(self, draw_history: List[LotteryNumber]) -> np.ndarray:
        """
        훈련에 사용할 상태 벡터 생성

        Args:
            draw_history: 당첨 내역 데이터

        Returns:
            상태 벡터
        """
        # 빈도 데이터 생성
        frequency = np.zeros(46)  # 인덱스 1-45 사용
        for draw in draw_history:
            if hasattr(draw, "numbers") and draw.numbers:
                for number in draw.numbers:
                    if 1 <= number <= 45:
                        frequency[number] += 1

        # 정규화
        if np.sum(frequency) > 0:
            frequency = frequency / np.sum(frequency)

        # 상태 벡터 구성
        state = frequency[1:]  # 인덱스 0 제외 (1-45만 사용)

        return state

    def load_model(self, path: str) -> bool:
        """
        저장된 모델 로드

        Args:
            path: 모델 파일 경로

        Returns:
            로드 성공 여부
        """
        try:
            if self.model is None:
                # 모델이 초기화되지 않았다면 초기화
                self.model = RLModel(self.config.to_dict())

            # 모델 로드
            success = self.model.load(path)

            if success:
                logger.info(f"RL 모델 로드 완료: {path}")
                if self.device.type == "cuda":
                    self.to_gpu()
                return True
            else:
                logger.error(f"RL 모델 로드 실패: {path}")
                return False

        except Exception as e:
            logger.error(f"RL 모델 로드 중 오류: {str(e)}")
            return False

    def _create_dataloader(self, validation_data, env):
        """
        데이터로더 생성

        Args:
            validation_data: 검증 데이터
            env: 환경 객체

        Returns:
            검증 데이터에 대한 데이터로더
        """
        from torch.utils.data import Dataset, DataLoader

        # 간단한 데이터셋 클래스
        class SimpleDataset(Dataset):
            def __init__(self, data, env):
                self.data = data
                self.env = env

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {"data": self.data[idx], "env": self.env}

        # 데이터셋 및 데이터로더 생성
        dataset = SimpleDataset(validation_data, env)
        return DataLoader(
            dataset, batch_size=self.get_config_value("batch_size", 32), shuffle=False
        )

    def _save_model(self):
        """모델 저장 메서드"""
        if self.model is None:
            self.logger.warning("저장할 모델이 없습니다")
            return

        try:
            model_dir = self.get_config_value("model_dir", "savedModels")
            model_name = self.get_config_value("model_name", "rl_model")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(model_dir, f"{model_name}_{timestamp}.pt")

            if hasattr(self.model, "save"):
                self.model.save(save_path)
            else:
                if hasattr(self.model, "state_dict"):
                    torch.save(self.model.state_dict(), save_path)
                elif hasattr(self.model, "model") and hasattr(
                    self.model.model, "state_dict"
                ):
                    torch.save(self.model.model.state_dict(), save_path)
                else:
                    # 모델 자체를 저장
                    torch.save(self.model, save_path)

            self.logger.info(f"모델 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류: {str(e)}")


def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="RL 모델 훈련 스크립트")
    parser.add_argument("--data", type=str, required=True, help="훈련 데이터 파일 경로")
    parser.add_argument("--epochs", type=int, default=200, help="훈련 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--no_cuda", action="store_true", help="CUDA 사용 안 함")
    parser.add_argument(
        "--model_dir", type=str, default="savedModels", help="모델 저장 디렉토리"
    )
    parser.add_argument("--model_name", type=str, default="rl_model", help="모델 이름")
    parser.add_argument(
        "--eval_split", type=float, default=0.2, help="검증 데이터 비율"
    )
    parser.add_argument(
        "--no_early_stopping", action="store_true", help="조기 종료 사용 안 함"
    )

    args = parser.parse_args()

    # 설정 생성
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "use_cuda": not args.no_cuda and torch.cuda.is_available(),
        "model_dir": args.model_dir,
        "model_name": args.model_name,
        "early_stopping": not args.no_early_stopping,
    }

    # 모델 디렉토리 생성
    os.makedirs(args.model_dir, exist_ok=True)

    # 세션 ID 생성 (실행 시간 기반)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 로그 디렉토리 설정
    log_dir = Path("logs") / "train"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 로그 파일 설정
    log_file = log_dir / f"rl_train_{session_id}.log"

    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"훈련 세션 시작 - 세션 ID: {session_id}")
    logger.info(f"설정: {config}")

    try:
        # 훈련기 초기화
        trainer = RLTrainer(config)

        # 데이터 로드
        trainer.load_data(args.data)

        # 데이터셋 분할
        if trainer.dataset is None:
            logger.error("데이터셋을 로드하지 못했습니다.")
            return

        # 데이터 가져오기
        all_data = trainer.dataset.get_all_data()
        if not all_data:
            logger.error("데이터를 가져오지 못했습니다.")
            return

        # 데이터 분할
        split_idx = int(len(all_data) * (1 - args.eval_split))
        train_data = all_data[:split_idx]
        valid_data = all_data[split_idx:]

        logger.info(
            f"데이터 분할 - 훈련: {len(train_data)}개, 검증: {len(valid_data)}개"
        )

        # 모델 훈련
        training_result = trainer.train(train_data, valid_data=valid_data)

        # 결과 저장
        result_file = Path(args.model_dir) / f"rl_training_result_{session_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            # json 직렬화 가능한 형태로 변환
            serializable_result = {}
            for k, v in training_result.items():
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    serializable_result[k] = v
                else:
                    serializable_result[k] = str(v)

            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

        logger.info(f"훈련 결과가 {result_file}에 저장되었습니다.")

        # 모델 평가
        eval_result = trainer.evaluate(valid_data)

        # 평가 결과 저장
        eval_file = Path(args.model_dir) / f"rl_eval_result_{session_id}.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            # json 직렬화 가능한 형태로 변환
            serializable_result = {}
            for k, v in eval_result.items():
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    serializable_result[k] = v
                else:
                    serializable_result[k] = str(v)

            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

        logger.info(f"평가 결과가 {eval_file}에 저장되었습니다.")

        logger.info("RL 모델 훈련 세션 완료")

    except Exception as e:
        logger.error(f"훈련 세션 중 오류 발생: {str(e)}")
        raise
    finally:
        # 로그 핸들러 제거
        logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()
