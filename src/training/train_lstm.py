# type: ignore
"""
LSTM 모델 훈련 스크립트 (LSTM Model Training)

이 모듈은 로또 번호 예측을 위한 LSTM 기반 모델을 훈련하는 스크립트입니다.
시퀀스 패턴을 학습하여 미래 번호 출현 확률을 예측합니다.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import logging
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict, deque
from torch.utils.data import DataLoader, Dataset
from ..utils.config_loader import ConfigProxy
from ..shared.types import LotteryNumber, ModelPrediction
from ..utils.error_handler_refactored import get_logger
from .unified_trainer import UnifiedTrainer
from ..models.lstm_model import LSTMModel
from ..utils.cuda_optimizers import AMPTrainer
from ..utils.profiler import Profiler
from ..utils.batch_controller import DynamicBatchSizeController
from ..utils.cache_manager import ThreadLocalCache
from ..utils.performance_utils import MemoryTracker

# 상대 경로 임포트 설정
from ..models.lstm_model import LSTMNetwork  # type: ignore
from ..utils.error_handler_refactored import get_logger
from ..utils.data_loader import load_draw_history  # 필요시 사용

# 파일 기반 로딩 제거# from ..utils.data_loader import load_draw_history

# 로거 설정
logger = get_logger(__name__)


class LotteryDataset(Dataset):
    def __init__(self, data: List[LotteryNumber], sequence_length: int):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.data[idx : idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]

        return {
            "input": torch.tensor([n.numbers for n in sequence], dtype=torch.float),
            "target": torch.tensor(target.numbers, dtype=torch.float),
        }


class LSTMTrainer(UnifiedTrainer):
    def __init__(self, config: ConfigProxy):
        super().__init__(LSTMModel, config, "lstm")
        self.amp_trainer = AMPTrainer(config)
        self.batch_controller = DynamicBatchSizeController(
            config, initial_batch_size=config.safe_get("batch_size", default=32)
        )
        self.cache = ThreadLocalCache()
        self.memory_tracker = MemoryTracker()
        self.profiler = Profiler(config)

        # 장치 설정
        use_cuda = config.safe_get("use_cuda", True) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def _create_dataloader(self, inputs, targets):
        """데이터로더를 생성합니다."""
        from torch.utils.data import TensorDataset, DataLoader

        # 입력과 타겟이 리스트인 경우 텐서로 변환
        if isinstance(inputs, list):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.float32)

        # 데이터셋 생성
        dataset = TensorDataset(inputs, targets)

        # 데이터로더 생성
        return DataLoader(
            dataset, batch_size=self.config.safe_get("batch_size", 32), shuffle=False
        )

    def _save_checkpoint(self):
        """현재 모델을 체크포인트로 저장합니다."""
        if self.model is None:
            return

        try:
            # 모델 디렉토리 생성
            model_dir = Path(self.config.safe_get("model_dir", "savedModels"))
            model_dir.mkdir(parents=True, exist_ok=True)

            # 저장 경로 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = model_dir / f"lstm_model_{timestamp}.pt"

            # 모델 저장
            if hasattr(self.model, "save"):
                self.model.save(str(filepath))
            else:
                if hasattr(self.model, "state_dict"):
                    torch.save(self.model.state_dict(), str(filepath))
                elif hasattr(self.model, "model") and hasattr(
                    self.model.model, "state_dict"
                ):
                    torch.save(self.model.model.state_dict(), str(filepath))
                else:
                    # 모델 자체를 저장 시도
                    torch.save(self.model, str(filepath))

            self.logger.info(f"모델 체크포인트 저장 완료: {filepath}")
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류: {str(e)}")

    def set_data(self, draw_data: List[LotteryNumber]) -> None:
        """학습 데이터를 설정합니다."""
        super().set_data(draw_data)

        # 시퀀스 길이 설정
        sequence_length = self.config.safe_get("sequence_length", default=10)

        # 데이터셋 생성
        dataset = LotteryDataset(draw_data, sequence_length)

        # 데이터 분할
        train_size = int(len(dataset) * 0.7)
        val_size = int(len(dataset) * 0.15)
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        # 데이터 로더 생성
        batch_size = self.config.safe_get("batch_size", default=32)
        self._train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self._val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self._test_loader = DataLoader(test_dataset, batch_size=batch_size)

    def _get_loss_fn(self) -> torch.nn.Module:
        """손실 함수를 반환합니다."""
        return torch.nn.BCELoss()

    def _prepare_sequences(
        self, data: List[LotteryNumber]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        시퀀스 데이터 준비

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            (입력 시퀀스, 타겟 시퀀스) 튜플
        """
        seq_length = self.config["seq_length"]
        input_sequences = []
        target_sequences = []

        # 데이터가 충분한지 확인
        if len(data) <= seq_length:
            raise ValueError(f"데이터 부족: 최소 {seq_length+1}개의 항목이 필요합니다.")

        for i in range(len(data) - seq_length):
            # 입력 시퀀스 (과거 n회차)
            seq_data = data[i : i + seq_length]

            # 타겟 (n+1회차)
            target = data[i + seq_length]

            # 원-핫 인코딩 시퀀스 생성
            input_onehot = np.zeros((seq_length, 45))
            for j, lottery in enumerate(seq_data):
                for num in lottery.numbers:
                    input_onehot[j, num - 1] = 1

            # 타겟 원-핫 인코딩
            target_onehot = np.zeros(45)
            for num in target.numbers:
                target_onehot[num - 1] = 1

            input_sequences.append(input_onehot)
            target_sequences.append(target_onehot)

        return input_sequences, target_sequences

    def _augment_data(
        self, input_sequences: List[np.ndarray], target_sequences: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        데이터 증강

        Args:
            input_sequences: 입력 시퀀스
            target_sequences: 타겟 시퀀스

        Returns:
            증강된 입력 및 타겟 시퀀스
        """
        if not self.config["augmentation"]:
            return input_sequences, target_sequences

        augmented_inputs = []
        augmented_targets = []

        # 기존 데이터 추가
        augmented_inputs.extend(input_sequences)
        augmented_targets.extend(target_sequences)

        # 순서를 조금 변경한 증강 데이터 생성
        for i, (inputs, target) in enumerate(zip(input_sequences, target_sequences)):
            # 시퀀스 내에서 두 회차의 순서 변경
            if i % 2 == 0 and inputs.shape[0] > 1:
                new_inputs = inputs.copy()
                # 첫 번째와 마지막 회차 교환
                new_inputs[0], new_inputs[-1] = (
                    new_inputs[-1].copy(),
                    new_inputs[0].copy(),
                )
                augmented_inputs.append(new_inputs)
                augmented_targets.append(target)

        logger.info(
            f"데이터 증강 완료: {len(input_sequences)} → {len(augmented_inputs)} 샘플"
        )

        return augmented_inputs, augmented_targets

    def _evaluate_model(
        self, inputs: List[np.ndarray], targets: List[np.ndarray]
    ) -> float:
        """
        모델 평가

        Args:
            inputs: 입력 시퀀스 목록
            targets: 타겟 시퀀스 목록

        Returns:
            평균 손실
        """
        logger.info("LSTM 모델 평가 중...")

        try:
            # 모델이 None이면 경고 로그 남기고 기본값 반환
            if self.model is None:
                logger.warning("모델이 없습니다. 평가를 건너뜁니다.")
                return 1.0

            # 모델을 평가 모드로 설정
            if hasattr(self.model, "eval") and callable(getattr(self.model, "eval")):
                self.model.eval()  # type: ignore
            elif hasattr(self.model, "model") and hasattr(self.model.model, "eval"):  # type: ignore
                # 중첩 모델 구조
                self.model.model.eval()  # type: ignore

            # 손실 계산
            total_loss = 0.0
            count = 0

            # 입력 데이터를 미니 배치로 처리
            batch_size = min(32, len(inputs))
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i : i + batch_size]
                batch_targets = targets[i : i + batch_size]

                try:
                    # 배치 변환
                    # 3차원 텐서로 변환 [batch_size, seq_len, feature_dim]
                    X_batch = np.stack(batch_inputs)
                    y_batch = np.stack(batch_targets)

                    # PyTorch 텐서로 변환
                    X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(
                        self.device
                    )
                    y_tensor = torch.tensor(y_batch, dtype=torch.float32).to(
                        self.device
                    )

                    # 예측 수행
                    with torch.no_grad():
                        try:
                            # 모델 호출 시도
                            if hasattr(self.model, "__call__"):
                                outputs = self.model(X_tensor)
                            elif hasattr(self.model, "forward"):
                                outputs = self.model.forward(X_tensor)
                            elif hasattr(self.model, "model"):
                                outputs = self.model.model(X_tensor)
                            else:
                                raise AttributeError(
                                    "모델에 호출 가능한 메서드가 없습니다"
                                )
                        except Exception as e:
                            logger.error(f"모델 호출 중 오류: {e}")
                            raise

                    # 손실 계산
                    if isinstance(outputs, torch.Tensor) and isinstance(
                        y_tensor, torch.Tensor
                    ):
                        # MSE 손실 계산
                        batch_loss = torch.nn.functional.mse_loss(
                            outputs, y_tensor
                        ).item()
                        total_loss += batch_loss
                        count += 1
                    else:
                        logger.warning(f"출력 유형 불일치: {type(outputs)}")

                except Exception as e:
                    logger.warning(f"배치 평가 중 오류: {e}")
                    continue

            # 평균 손실 반환
            if count > 0:
                avg_loss = total_loss / count
                logger.info(f"평가 완료: 평균 손실 = {avg_loss:.4f}")
                return avg_loss
            else:
                logger.warning("평가할 배치가 없습니다.")
                return 1.0

        except Exception as e:
            logger.error(f"모델 평가 중 오류: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return 1.0

    def _train_epoch(self) -> Dict[str, float]:
        """한 에폭의 학습을 수행합니다."""
        if self.model is None or self.optimizer is None or self._train_loader is None:
            raise ValueError("모델, 옵티마이저 또는 데이터 로더가 설정되지 않았습니다.")

        self.model.train()
        total_loss = 0.0
        batch_count = 0

        # 손실 함수 가져오기
        loss_fn = self._get_loss_fn()

        for batch in self._train_loader:
            # 배치 크기 동적 조정
            self.batch_controller.adjust_batch_size(self._train_loader)

            # 배치 데이터 가져오기
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            # 옵티마이저 초기화
            self.optimizer.zero_grad()

            # 순전파
            outputs = self.model(inputs)

            # 손실 계산
            loss = loss_fn(outputs, targets)

            # 역전파
            loss.backward()

            # 가중치 업데이트
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        return {"loss": total_loss / max(1, batch_count)}

    def train(
        self,
        data: List[LotteryNumber],
        validation_data: Optional[List[LotteryNumber]] = None,
    ) -> Dict[str, Any]:
        """
        모델 훈련

        Args:
            data: 훈련 데이터
            validation_data: 검증 데이터

        Returns:
            훈련 결과
        """
        if len(data) < self.config["seq_length"] + 1:
            raise ValueError(
                f"학습 데이터가 너무 적습니다. 최소 {self.config['seq_length'] + 1}개의 항목이 필요합니다."
            )

        logger.info(
            f"LSTM 모델 훈련 시작 - 에폭: {self.config['epochs']}, 배치 크기: {self.config['batch_size']}"
        )

        start_time = time.time()

        try:
            self.memory_tracker.start()

            with self.profiler.profile("lstm_training"):
                # 모델이 초기화되지 않은 경우 초기화
                if self.model is None:
                    # 설정을 딕셔너리로 변환하여 전달
                    model_config = self.config.to_dict()
                    self.model = LSTMModel(model_config)
                    self.model.to(self.device)
                    logger.info("LSTM 모델 초기화 완료")

                # 시퀀스 데이터 준비
                input_sequences, target_sequences = self._prepare_sequences(data)

                # 데이터 증강
                if self.config.safe_get("augmentation", default=False):
                    input_sequences, target_sequences = self._augment_data(
                        input_sequences, target_sequences
                    )

                # 검증 데이터 준비
                if validation_data:
                    valid_inputs, valid_targets = self._prepare_sequences(
                        validation_data
                    )
                else:
                    split_idx = int(len(input_sequences) * 0.8)
                    train_inputs = input_sequences[:split_idx]
                    train_targets = target_sequences[:split_idx]
                    valid_inputs = input_sequences[split_idx:]
                    valid_targets = target_sequences[split_idx:]
                    input_sequences = train_inputs
                    target_sequences = train_targets

                # 훈련 파라미터
                epochs = self.config.safe_get("epochs", default=100)
                batch_size = self.config.safe_get("batch_size", default=32)

                # 학습 기록
                losses = []
                valid_losses = []
                best_valid_loss = float("inf")
                patience_counter = 0

                # 훈련 루프
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    num_batches = 0

                    # 배치 처리
                    for i in range(0, len(input_sequences), batch_size):
                        batch_inputs = input_sequences[i : i + batch_size]
                        batch_targets = target_sequences[i : i + batch_size]

                        # 데이터 로더 생성
                        dataloader = self._create_dataloader(
                            batch_inputs, batch_targets
                        )

                        # 배치 크기 동적 조정
                        batch_size = self.batch_controller.adjust_batch_size(dataloader)

                        # AMP를 사용한 학습 스텝
                        metrics = self.amp_trainer.train_step(
                            self.model,
                            self.model.optimizer,
                            self._get_loss_fn(),
                            {"input": batch_inputs, "target": batch_targets},  # type: ignore
                        )

                        epoch_loss += metrics["loss"]
                        num_batches += 1

                    # 에폭 평균 손실
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                    losses.append(avg_loss)

                    # 검증
                    valid_loss = self.amp_trainer.evaluate(
                        self.model,
                        self._get_loss_fn(),
                        self._create_dataloader(valid_inputs, valid_targets),
                    )["loss"]
                    valid_losses.append(valid_loss)

                    # 조기 종료 체크
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        patience_counter = 0
                        self._save_checkpoint()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.safe_get(
                            "patience", default=10
                        ):
                            logger.info(f"조기 종료: {epoch + 1} 에폭")
                            break

                    logger.info(
                        f"에폭 {epoch + 1}/{epochs} - 훈련 손실: {avg_loss:.4f}, 검증 손실: {valid_loss:.4f}"
                    )

                return {
                    "losses": losses,
                    "valid_losses": valid_losses,
                    "best_valid_loss": best_valid_loss,
                    "memory_usage": self.memory_tracker.get_memory_log(),
                }

        except Exception as e:
            logger.error(f"LSTM 훈련 중 오류 발생: {str(e)}")
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
            raise ValueError("모델이 초기화되지 않았습니다")

        logger.info(f"LSTM 모델 평가 시작 - {len(data)}개 데이터")

        try:
            # 시퀀스 데이터 준비
            input_sequences, target_sequences = self._prepare_sequences(data)

            # 모델 평가
            loss = self._evaluate_model(input_sequences, target_sequences)

            # 평가 결과
            result = {
                "loss": loss,
                "data_count": len(data),
            }

            logger.info(f"LSTM 모델 평가 완료 - 손실: {loss:.4f}")
            return result

        except Exception as e:
            logger.error(f"평가 중 오류: {str(e)}")
            return {"error": str(e)}

    def predict(
        self, data: List[LotteryNumber], count: int = 1
    ) -> List[ModelPrediction]:
        """
        로또 번호 예측

        Args:
            data: 학습 데이터
            count: 예측할 번호 세트 수

        Returns:
            예측 결과 목록 (ModelPrediction 객체)
        """
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다")

        logger.info(f"LSTM 모델 예측 시작 - {count}개 세트 예측")

        # 모델을 평가 모드로 설정 (안전하게 수행)
        try:
            # 모델이 복합 객체일 경우 내부 모델에 접근
            model_obj = self.model
            if hasattr(model_obj, "model"):
                model_obj = model_obj.model

            # 다양한 객체 타입 처리 - 간접 호출 방식 사용
            if hasattr(model_obj, "eval") and callable(getattr(model_obj, "eval")):
                # PyTorch 모듈인 경우
                getattr(model_obj, "eval")()
            elif hasattr(model_obj, "set_eval") and callable(
                getattr(model_obj, "set_eval")
            ):
                # 커스텀 eval 메소드가 있는 경우
                getattr(model_obj, "set_eval")()
            elif hasattr(model_obj, "evaluate") and callable(
                getattr(model_obj, "evaluate")
            ):
                # evaluate 메소드가 있는 경우 - 매개변수 확인
                evaluate_method = getattr(model_obj, "evaluate")
                if evaluate_method.__code__.co_argcount > 1:  # self 제외한 매개변수 수
                    # 데이터 매개변수가 필요한 경우
                    logger.warning(
                        "evaluate 메서드에 데이터 매개변수가 필요하나 제공하지 않음"
                    )
                else:
                    # 매개변수가 필요 없는 evaluate
                    evaluate_method()
            else:
                # eval 모드를 설정할 수 없는 경우
                logger.info(
                    "모델이 eval 모드를 지원하지 않습니다. 기본 모드로 계속 진행합니다."
                )
        except Exception as e:
            logger.warning(f"모델 평가 모드 설정 중 오류: {e}")

        results = []

        try:
            # 마지막 N회차 데이터 추출
            seq_length = self.config["seq_length"]
            if len(data) < seq_length:
                raise ValueError(f"최소 {seq_length}개의 데이터가 필요합니다")

            # 최근 데이터 사용
            recent_data = data[-seq_length:]

            # 입력 시퀀스 생성 (원-핫 인코딩)
            input_onehot = np.zeros((seq_length, 45))
            for j, lottery in enumerate(recent_data):
                for num in lottery.numbers:
                    input_onehot[j, num - 1] = 1

            # 배치 차원 추가
            input_tensor = torch.FloatTensor(input_onehot).unsqueeze(0).to(self.device)

            # count 세트 예측
            for _ in range(count):
                try:
                    with torch.no_grad():
                        # 예측 메소드 선택 및 호출 (안전하게)
                        output = None
                        model_obj = self.model
                        if hasattr(model_obj, "model"):
                            model_obj = model_obj.model

                        # 여러 방식의 모델 호출 시도
                        if hasattr(model_obj, "forward") and callable(
                            getattr(model_obj, "forward")
                        ):
                            forward_method = getattr(model_obj, "forward")
                            output = forward_method(input_tensor)
                        elif callable(model_obj):
                            # callable 객체인 경우 직접 호출
                            output = model_obj(input_tensor)
                        else:
                            raise ValueError("호출 가능한 predict 메소드가 없습니다")

                        if output is None:
                            raise ValueError("모델 출력이 없습니다")

                        # 시그모이드 적용하여 확률 변환
                        probabilities = torch.sigmoid(output).squeeze().cpu().numpy()

                        # Top-k 번호 선택 (k = numbers_to_predict)
                        numbers_to_predict = self.config["numbers_to_predict"]
                        top_indices = np.argsort(probabilities)[-numbers_to_predict:]

                        # 인덱스를 실제 번호로 변환 (1-45)
                        top_numbers = [idx + 1 for idx in top_indices]

                        # 정렬
                        predicted_numbers = sorted(top_numbers)

                        # 신뢰도 추정
                        confidence = float(np.mean(probabilities[top_indices]))

                        # ModelPrediction 객체 생성
                        results.append(
                            ModelPrediction(
                                numbers=predicted_numbers,
                                confidence=confidence,
                                model_type="LSTM",
                            )
                        )
                except Exception as e:
                    logger.error(f"단일 예측 생성 중 오류: {str(e)}")
                    # 예측 실패 시 임의의 번호 생성
                    import random

                    random_numbers = sorted(random.sample(range(1, 46), 6))
                    results.append(
                        ModelPrediction(
                            numbers=random_numbers,
                            confidence=0.1,  # 낮은 신뢰도 표시
                            model_type="LSTM",
                        )
                    )

            logger.info(f"LSTM 모델 예측 완료 - {len(results)}개 세트")
            return results

        except Exception as e:
            logger.error(f"예측 중 오류: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

            # 오류 발생 시 무작위 예측 반환
            import random

            fallback_results = []
            for _ in range(count):
                random_numbers = sorted(random.sample(range(1, 46), 6))
                fallback_results.append(
                    ModelPrediction(
                        numbers=random_numbers,
                        confidence=0.1,  # 낮은 신뢰도
                        model_type="LSTM",
                    )
                )
            return fallback_results

    def load_model(self, path: str) -> bool:
        """
        저장된 모델 로드

        Args:
            path: 모델 파일 경로

        Returns:
            로드 성공 여부
        """
        try:
            # 모델 생성
            self.model = LSTMModel(self.config)

            # 모델 로드
            success = self.model.load(path)

            if success:
                logger.info(f"LSTM 모델 로드 완료: {path}")
                return True
            else:
                logger.error(f"LSTM 모델 로드 실패: {path}")
                return False

        except Exception as e:
            logger.error(f"LSTM 모델 로드 중 오류: {str(e)}")
            return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LSTM 모델 훈련 스크립트")
    parser.add_argument(
        "--data", type=str, default="data/lottery.csv", help="훈련 데이터 파일 경로"
    )
    parser.add_argument("--epochs", type=int, default=100, help="훈련 에폭 수")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--hidden-dim", type=int, default=128, help="은닉층 차원")
    parser.add_argument("--seq-length", type=int, default=5, help="시퀀스 길이")
    parser.add_argument("--dropout", type=float, default=0.3, help="드롭아웃 비율")
    parser.add_argument(
        "--save-dir", type=str, default="savedModels", help="모델 저장 디렉토리"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="CUDA 비활성화"
    )
    parser.add_argument(
        "--no-augment", action="store_true", default=False, help="데이터 증강 비활성화"
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    args = parser.parse_args()

    # 설정 준비
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "hidden_dim": args.hidden_dim,
        "seq_length": args.seq_length,
        "dropout": args.dropout,
        "model_dir": args.save_dir,
        "use_cuda": not args.no_cuda and torch.cuda.is_available(),
        "seed": args.seed,
        "augmentation": not args.no_augment,
    }

    # 훈련기 생성
    trainer = LSTMTrainer(config)

    try:
        # 데이터 로드
        data = trainer.load_data(args.data)

        # 훈련 및 검증 데이터 분할
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        # 모델 훈련
        result = trainer.train(train_data, val_data)

        # 결과 출력
        logger.info(f"훈련 결과: {result}")

        # 평가
        eval_result = trainer.evaluate(val_data)
        logger.info(f"평가 결과: {eval_result}")

    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
