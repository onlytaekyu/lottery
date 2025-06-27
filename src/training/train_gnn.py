# type: ignore
"""
그래프 신경망 모델 훈련 스크립트

이 모듈은 로또 번호 예측을 위한 GNN(Graph Neural Network) 모델을 훈련하는 기능을 제공합니다.
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

# 상대 경로 임포트 설정
from ..models.gnn_model import GNNModel
from ..shared.types import LotteryNumber, PatternAnalysis, ModelPrediction
from ..utils.error_handler_refactored import get_logger
from ..utils.unified_performance import Profiler, ProfilerConfig
from ..utils.dynamic_batch_size_utils import get_safe_batch_size
from ..utils.data_loader import load_draw_history
from ..utils.cuda_optimizers import AMPTrainer
from ..utils.batch_controller import DynamicBatchSizeController
from ..utils.cache_manager import ThreadLocalCache
from ..utils.performance_utils import MemoryTracker
from ..utils.unified_config import ConfigProxy

# 로거 설정
logger = get_logger(__name__)


class GNNTrainer:
    """그래프 신경망 모델 훈련기"""

    def __init__(self, config):
        """
        GNN 모델 훈련기 초기화

        Args:
            config: 훈련 설정
        """
        # 설정 초기화
        if isinstance(config, dict):
            self.config = ConfigProxy(config)
        else:
            self.config = config

        # 기기 설정
        use_cuda = self.config.safe_get("use_cuda", True) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # 랜덤 시드 설정
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["seed"])

        # 모델 및 데이터셋 초기화
        self.model = None
        self.data = None

        # 훈련 메트릭
        self.train_losses = []
        self.valid_losses = []
        self.best_loss = float("inf")
        self.best_model_state = None
        self.early_stop_counter = 0

        # 프로파일러 초기화
        profiler_config = ConfigProxy(
            {
                "enable_cpu_profiling": True,
                "enable_memory_profiling": True,
                "enable_cuda_profiling": self.device.type == "cuda",
                "profile_dir": "logs/profiler",
            }
        )
        self.profiler = Profiler(profiler_config)

        # 로거 초기화
        self.logger = get_logger(__name__)

        # 모델 디렉토리 생성
        os.makedirs(self.config["model_dir"], exist_ok=True)

        self.amp_trainer = AMPTrainer(self.config)
        self.batch_controller = DynamicBatchSizeController(
            config=self.config,
            initial_batch_size=self.config.safe_get("batch_size", 32),
        )
        self.cache = ThreadLocalCache()
        self.memory_tracker = MemoryTracker()

        # dataloader 속성 초기화를 추가합니다
        self.dataloader = None
        self.loss_fn = self._get_loss_fn()

        logger.info(f"GNNTrainer 초기화 완료 - 장치: {self.device}")

    def load_data(self, data_path: str) -> None:
        """
        데이터 로드

        Args:
            data_path: 데이터 파일 경로
        """
        try:
            # 데이터 직접 로드
            self.data = load_draw_history(file_path=data_path)
            logger.info(f"데이터 로드 완료 - {len(self.data)} 항목")

            # 모델 초기화
            self.model = GNNModel(self.config.to_dict())

            # GPU 설정
            self.model.to(self.device)

            logger.info("GNN 모델 초기화 완료")

        except Exception as e:
            logger.error(f"데이터 로드 중 오류: {str(e)}")
            raise

    def prepare_data(self, data: List[LotteryNumber]) -> Tuple[List, List]:
        """
        훈련 데이터 준비 (동시 출현 관계 분석)

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            (양성 쌍, 음성 쌍) 튜플
        """
        # 동시 출현 여부 추적
        cooccurred = defaultdict(bool)
        positive_pairs = []

        # 양성 쌍 수집 (함께 출현한 번호 쌍)
        for item in data:
            numbers = item.numbers
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = (numbers[i] - 1, numbers[j] - 1)  # 0-44 인덱스로 변환
                    positive_pairs.append(pair)
                    cooccurred[pair] = True
                    cooccurred[(pair[1], pair[0])] = True

        # 음성 쌍 샘플링 (함께 출현하지 않은 쌍)
        negative_pairs = []
        all_nodes = list(range(self.config["num_nodes"]))

        while len(negative_pairs) < len(positive_pairs):
            i = random.choice(all_nodes)
            j = random.choice(all_nodes)
            if i != j and not cooccurred.get((i, j), False):
                negative_pairs.append((i, j))
                cooccurred[(i, j)] = True
                cooccurred[(j, i)] = True

        logger.info(
            f"훈련 데이터 준비 완료 - 양성 쌍: {len(positive_pairs)}, 음성 쌍: {len(negative_pairs)}"
        )
        return positive_pairs, negative_pairs

    def _get_loss_fn(self) -> nn.Module:
        """손실 함수 반환"""
        return nn.BCEWithLogitsLoss()

    def train(
        self,
        data: List[LotteryNumber],
        validation_data: Optional[List[LotteryNumber]] = None,
        pattern_analysis: Optional[PatternAnalysis] = None,
    ) -> Dict[str, Any]:
        """
        모델 훈련

        Args:
            data: 훈련 데이터
            validation_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과

        Returns:
            훈련 결과
        """
        logger.info(
            f"GNN 모델 훈련 시작 - 에폭: {self.config['epochs']}, 배치 크기: {self.config['batch_size']}"
        )

        try:
            self.memory_tracker.start()

            with self.profiler.profile("gnn_training"):
                # 데이터 준비
                train_pos_pairs, train_neg_pairs = self.prepare_data(data)

                if validation_data:
                    valid_pos_pairs, valid_neg_pairs = self.prepare_data(
                        validation_data
                    )
                else:
                    # 훈련 데이터의 일부를 검증용으로 사용
                    split_point = int(0.8 * len(train_pos_pairs))
                    valid_pos_pairs = train_pos_pairs[split_point:]
                    valid_neg_pairs = train_neg_pairs[split_point:]
                    train_pos_pairs = train_pos_pairs[:split_point]
                    train_neg_pairs = train_neg_pairs[:split_point]

                # 데이터로더 초기화
                self.dataloader = self._create_dataloader(
                    train_pos_pairs, train_neg_pairs
                )
                valid_dataloader = self._create_dataloader(
                    valid_pos_pairs, valid_neg_pairs
                )

                # 훈련 파라미터
                epochs = self.config["epochs"]
                batch_size = self.config["batch_size"]

                # 학습 기록
                losses = []
                valid_losses = []
                best_valid_loss = float("inf")
                patience_counter = 0

                # 훈련 루프
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    num_batches = 0

                    # 배치 크기 동적 조정
                    if self.dataloader is not None:
                        batch_size = self.batch_controller.adjust_batch_size(
                            self.dataloader
                        )

                    # 손실 함수 가져오기
                    loss_fn = self._get_loss_fn()

                    # 모델이 초기화되지 않았다면 초기화
                    if self.model is None:
                        self.model = GNNModel(self.config.to_dict())
                        self.model.to(self.device)

                    # 옵티마이저가 초기화되지 않았다면 초기화
                    if not hasattr(self, "optimizer") or self.optimizer is None:
                        self.optimizer = torch.optim.Adam(
                            self.model.parameters(),
                            lr=self.config.safe_get("learning_rate", 0.001),
                        )

                    # AMP를 사용한 학습 스텝
                    if self.dataloader is not None:
                        for batch_data, batch_labels in self.dataloader:
                            metrics = self.amp_trainer.train_step(
                                model=self.model,
                                optimizer=self.optimizer,
                                loss_fn=loss_fn,
                                batch={"input": batch_data, "target": batch_labels},
                            )

                            epoch_loss += metrics.get("loss", 0.0)
                            num_batches += 1

                    # 에폭 평균 손실
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                    losses.append(avg_loss)

                    # 검증
                    valid_loss = self.amp_trainer.evaluate(
                        self.model,
                        self.loss_fn,
                        valid_dataloader,
                    )["loss"]
                    valid_losses.append(valid_loss)

                    # 조기 종료 체크
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        patience_counter = 0
                        self._save_model()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.safe_get("patience", 10):
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
            logger.error(f"GNN 훈련 중 오류 발생: {str(e)}")
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
        logger.info(f"GNN 모델 평가 시작 - 데이터 크기: {len(data)}")

        if self.model is None:
            logger.error("모델이 초기화되지 않았습니다.")
            return {"error": "모델이 초기화되지 않음", "success": False}

        # 평가 모드로 설정
        self.model.eval()

        # 양성/음성 쌍 준비
        pos_pairs, neg_pairs = self.prepare_data(data)

        # 계산 장치로 이동
        targets = [1] * len(pos_pairs) + [0] * len(neg_pairs)

        # 링크 예측 평가
        eval_result = self.model._evaluate_link_prediction(pos_pairs, neg_pairs)

        return {
            "auc": eval_result,
            "success": True,
            "data_size": len(data),
        }

    def predict(
        self, data: List[LotteryNumber], count: int = 1
    ) -> List[ModelPrediction]:
        """
        번호 예측

        Args:
            data: 참조 데이터
            count: 예측 개수

        Returns:
            List[ModelPrediction]: ModelPrediction 객체 리스트
        """
        logger.info(f"GNN 모델로 {count}개 번호 예측 시작")

        try:
            # 모델이 없으면 초기화
            if self.model is None:
                logger.warning("GNN 모델이 초기화되지 않음. 기본 모델 생성")
                self.model = GNNModel(self.config.to_dict())

            # 학습 데이터 준비
            X, _ = self.prepare_data(data)

            # 예측 수행
            with torch.no_grad():
                # 안전하게 eval 모드 설정
                if hasattr(self.model, "eval"):
                    self.model.eval()  # type: ignore

                # 예측 실행
                predictions = []

                try:
                    # 노드 확률 계산
                    node_probs = None

                    # 방법 1: predict_probabilities 메소드 확인
                    if hasattr(self.model, "predict_probabilities"):
                        try:
                            node_probs = self.model.predict_probabilities(X)  # type: ignore
                        except Exception as e:
                            logger.warning(f"predict_probabilities 호출 실패: {e}")
                            node_probs = None

                    # 방법 2: get_node_probabilities 메소드 확인
                    if node_probs is None and hasattr(
                        self.model, "get_node_probabilities"
                    ):
                        try:
                            node_probs = self.model.get_node_probabilities(X)  # type: ignore
                        except Exception as e:
                            logger.warning(f"get_node_probabilities 호출 실패: {e}")
                            node_probs = None

                    # 대체 구현
                    if node_probs is None:
                        logger.warning(
                            "확률 계산 메서드를 사용할 수 없어 대체 구현 사용"
                        )
                        node_probs = np.random.random(45)

                        # 숫자 빈도 반영
                        number_freq = {}
                        for item in data:
                            for num in item.numbers:
                                if 1 <= num <= 45:
                                    number_freq[num - 1] = (
                                        number_freq.get(num - 1, 0) + 1
                                    )

                        # 빈도에 따라 확률 조정
                        if number_freq:
                            for num, freq in number_freq.items():
                                if 0 <= num < 45:
                                    node_probs[num] += freq * 0.01

                    # node_probs가 Tensor인 경우 NumPy로 변환
                    if isinstance(node_probs, torch.Tensor):
                        node_probs = node_probs.cpu().numpy()

                    # 예측 개수만큼 반복
                    for _ in range(count):
                        try:
                            # 상위 확률 10개 노드 추출
                            top_indices = np.argsort(node_probs)[-10:][::-1]
                            top_probs = node_probs[top_indices]

                            # 확률을 신뢰도로 변환 (0-1 사이로 정규화)
                            confidences = (
                                top_probs / top_probs.sum()
                                if top_probs.sum() > 0
                                else top_probs
                            )

                            # 번호로 변환 (+1 필요, 인덱스는 0부터 시작하므로)
                            valid_numbers = [
                                int(idx) + 1 for idx in top_indices if 0 <= idx < 45
                            ]
                            valid_confidences = confidences.tolist()[
                                : len(valid_numbers)
                            ]

                            # 번호가 6개 미만이면 채우기
                            missing = 6 - len(valid_numbers)
                            if missing > 0:
                                available_numbers = [
                                    n for n in range(1, 46) if n not in valid_numbers
                                ]
                                if available_numbers:
                                    additional_numbers = random.sample(
                                        available_numbers,
                                        min(missing, len(available_numbers)),
                                    )
                                    valid_numbers.extend(additional_numbers)
                                    valid_confidences.extend(
                                        [0.3] * len(additional_numbers)
                                    )

                            # 번호 정렬
                            if valid_numbers:
                                # numpy 배열이 아닌 리스트로 처리
                                combined = list(zip(valid_numbers, valid_confidences))
                                combined.sort(key=lambda x: x[0])  # 번호 기준 정렬
                                valid_numbers = [item[0] for item in combined]
                                valid_confidences = [item[1] for item in combined]

                            # 6개 번호 선택 (정확히 6개로 맞추기)
                            if len(valid_numbers) > 6:
                                valid_numbers = valid_numbers[:6]
                                valid_confidences = valid_confidences[:6]

                            # 번호가 6개 미만이면 무작위로 채우기 (안전장치)
                            if len(valid_numbers) < 6:
                                remaining = set(range(1, 46)) - set(valid_numbers)
                                if remaining:
                                    fill_count = 6 - len(valid_numbers)
                                    fill_numbers = sorted(
                                        random.sample(list(remaining), fill_count)
                                    )
                                    valid_numbers.extend(fill_numbers)
                                    valid_confidences.extend([0.1] * fill_count)

                            # 신뢰도 계산 및 제한 (0-1 사이 값으로)
                            avg_confidence = (
                                sum(valid_confidences) / len(valid_confidences)
                                if valid_confidences
                                else 0.5
                            )
                            # 신뢰도 범위 제한
                            avg_confidence = min(0.98, max(0.1, avg_confidence))

                            # 최종 번호가 정확히 6개인지 확인 (안전장치)
                            if len(valid_numbers) != 6:
                                logger.warning(
                                    f"예측 번호 개수가 6개가 아님: {len(valid_numbers)}개"
                                )
                                # 무작위로 생성
                                valid_numbers = sorted(random.sample(range(1, 46), 6))
                                avg_confidence = 0.3  # 낮은 신뢰도

                            # ModelPrediction 객체 생성 (shared.types에서 가져온 클래스)
                            from ..shared.types import ModelPrediction

                            prediction = ModelPrediction(
                                numbers=valid_numbers,
                                confidence=float(avg_confidence),
                                model_type="gnn",
                            )
                            predictions.append(prediction)

                        except Exception as e:
                            logger.error(f"예측 생성 중 오류: {str(e)}")
                            import traceback

                            logger.debug(traceback.format_exc())
                            # 오류 시 대체 예측
                            random_numbers = sorted(random.sample(range(1, 46), 6))
                            from ..shared.types import ModelPrediction

                            predictions.append(
                                ModelPrediction(
                                    numbers=random_numbers,
                                    confidence=0.3,
                                    model_type="gnn",
                                )
                            )
                            continue

                except Exception as e:
                    logger.error(f"노드 확률 계산 중 오류: {str(e)}")
                    # 오류 시 무작위 예측
                    for _ in range(count):
                        random_numbers = sorted(random.sample(range(1, 46), 6))
                        from ..shared.types import ModelPrediction

                        predictions.append(
                            ModelPrediction(
                                numbers=random_numbers, confidence=0.3, model_type="gnn"
                            )
                        )

            logger.info(f"GNN 모델로 {len(predictions)}개 예측 생성 완료")
            return predictions

        except Exception as e:
            logger.error(f"GNN 예측 생성 중 오류: {str(e)}")
            # 오류 시 무작위 예측
            predictions = []
            for _ in range(count):
                random_numbers = sorted(random.sample(range(1, 46), 6))
                from ..shared.types import ModelPrediction

                predictions.append(
                    ModelPrediction(
                        numbers=random_numbers, confidence=0.3, model_type="gnn"
                    )
                )
            return predictions

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
            self.model = GNNModel(self.config.to_dict())

            # 모델 로드
            success = self.model.load(path)

            if success:
                logger.info(f"GNN 모델 로드 완료: {path}")
                return True
            else:
                logger.error(f"GNN 모델 로드 실패: {path}")
                return False

        except Exception as e:
            logger.error(f"GNN 모델 로드 중 오류: {str(e)}")
            return False

    def _create_dataloader(self, pos_pairs, neg_pairs):
        """
        데이터로더 생성

        Args:
            pos_pairs: 양성 쌍 목록
            neg_pairs: 음성 쌍 목록

        Returns:
            데이터로더
        """
        # 데이터 통합
        all_pairs = pos_pairs + neg_pairs
        labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

        # 텐서 변환
        pairs_tensor = torch.LongTensor(all_pairs)
        labels_tensor = torch.FloatTensor(labels)

        # TensorDataset 생성
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(pairs_tensor, labels_tensor)

        # DataLoader 생성
        return DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)

    def _save_model(self):
        """모델 저장 메서드"""
        if self.model is None:
            self.logger.warning("저장할 모델이 없습니다")
            return

        try:
            # 모델 저장 경로 설정
            model_dir = self.config.safe_get("model_dir", "savedModels")
            os.makedirs(model_dir, exist_ok=True)

            model_name = self.config.safe_get("model_name", "gnn_model")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(model_dir, f"{model_name}_{timestamp}.pt")

            # 모델 저장
            if hasattr(self.model, "save"):
                self.model.save(save_path)
            elif hasattr(self.model, "state_dict"):
                torch.save(
                    (
                        self.model.state_dict()
                        if hasattr(self.model, "state_dict")
                        else self.model.model.state_dict()
                    ),
                    save_path,
                )
            else:
                self.logger.warning("모델에 저장 메서드가 없습니다.")
                return

            self.logger.info(f"모델 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류: {str(e)}")


def main():
    """메인 함수"""
    # 인자 파싱
    parser = argparse.ArgumentParser(description="GNN 모델 훈련 스크립트")
    parser.add_argument(
        "--data", type=str, default="data/lottery_history.csv", help="데이터 파일 경로"
    )
    parser.add_argument("--epochs", type=int, default=100, help="훈련 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--hidden_dim", type=int, default=64, help="은닉층 차원")
    parser.add_argument("--dropout", type=float, default=0.3, help="드롭아웃 비율")
    parser.add_argument(
        "--device", type=str, default="auto", help="사용할 장치 (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="모델 저장 경로 (옵션)"
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    # 장치 설정
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device_str = "cuda" if use_cuda and args.device != "cpu" else "cpu"

    # 설정 객체 생성
    config = {
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "use_cuda": use_cuda,
        "seed": args.seed,
    }

    # 훈련기 초기화
    trainer = GNNTrainer(config)

    try:
        # 데이터 로드
        trainer.load_data(args.data)

        # 훈련 데이터 직접 사용
        if trainer.data is None:
            print("데이터를 로드할 수 없습니다.")
            sys.exit(1)

        train_data = trainer.data

        # 훈련 실행
        result = trainer.train(train_data)

        # 결과 출력
        print("\n===== GNN 모델 훈련 결과 =====")
        print(f"최종 손실: {result.get('loss', 'N/A')}")
        print(f"최고 검증 손실: {result.get('best_valid_loss', 'N/A')}")
        print(f"훈련 시간: {result.get('memory_usage', 'N/A')}")
        print(f"에폭: {args.epochs}")

        # 모델 저장
        if trainer.model is not None:
            save_path = (
                args.save_path
                if args.save_path
                else str(Path("savedModels/gnn_model.pt"))
            )
            if hasattr(trainer.model, "save") and callable(
                getattr(trainer.model, "save")
            ):
                trainer.model.save(save_path)
                print(f"모델 저장 완료: {save_path}")
            else:
                # 기본 PyTorch 모델 저장
                try:
                    # 내부 pytorch 모델에 접근하여 저장
                    if hasattr(trainer.model, "model"):
                        torch.save(trainer.model.model, save_path)
                    elif hasattr(trainer.model, "network"):
                        torch.save(trainer.model.network, save_path)
                    else:
                        # 직접 객체 저장
                        torch.save(trainer.model, save_path)
                    print(f"모델 저장 완료: {save_path}")
                except Exception as e:
                    print(f"모델 저장 중 오류 발생: {str(e)}")
        else:
            print("모델이 초기화되지 않아 저장할 수 없습니다.")

    except Exception as e:
        print(f"훈련 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("훈련 완료!")


if __name__ == "__main__":
    main()
