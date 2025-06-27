# type: ignore
"""
통합 훈련 인터페이스 (Training Interface)

이 모듈은 다양한 모델(RL, 통계, 패턴 등)의 훈련을 위한 통합 인터페이스를 제공합니다.
TrainingOptimizer를 활용한 최적화된 훈련 및 병렬 처리를 지원합니다.
"""

import os
import sys
import torch
import time
import gc
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, cast, Type, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import threading

from ..utils.error_handler_refactored import get_logger
from ..utils.data_loader import DataLoader
from ..shared.types import LotteryNumber, PatternAnalysis
from .performance_optimizer import (
    TrainingOptimizer,
    OptimizerConfig,
    get_training_optimizer,
)

# 모델 타입 임포트 (타입 체크 및 런타임 구분)
from ..models.base_model import BaseModel

# 모델 타입 정의 (순환 참조 방지)
if TYPE_CHECKING:
    # 타입 체킹 시에만 사용할 타입 정의
    from ..models.rl_model import RLModel
    from ..models.statistical_model import StatisticalModel
else:
    # 런타임에는 BaseModel 상속한 클래스 정의 (타입 호환성 용도)
    class RLModel(BaseModel):
        """RL 모델 클래스"""

        pass

    class StatisticalModel(BaseModel):
        """통계 모델 클래스"""

        pass


# 로거 설정 - 이 모듈을 위한 로거는 여기서 먼저 정의합니다.
logger = get_logger(__name__)

# 실제 모델 클래스 임포트 (런타임 임포트)
try:
    # 나중에 실제 RLModel을 가져오기 위한 런타임 임포트
    def get_real_rl_model():
        try:
            from ..models.rl_model import RLModel as RealRLModel

            return RealRLModel
        except ImportError:
            logger.warning("RLModel을 임포트할 수 없습니다.")
            return None

except:
    logger.warning("RLModel 임포트 정의를 설정할 수 없습니다.")

try:
    # 나중에 실제 StatisticalModel을 가져오기 위한 런타임 임포트
    def get_real_statistical_model():
        try:
            from ..models.statistical_model import (
                StatisticalModel as RealStatisticalModel,
            )

            return RealStatisticalModel
        except ImportError:
            logger.warning("StatisticalModel을 임포트할 수 없습니다.")
            return None

except:
    logger.warning("StatisticalModel 임포트 정의를 설정할 수 없습니다.")


@dataclass
class TrainingConfig:
    """훈련 설정 클래스"""

    # 일반 설정
    model_dir: str = "savedModels"
    log_dir: str = "logs/training"

    # 훈련 파라미터
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2

    # 조기 종료 설정
    early_stopping: bool = True
    patience: int = 10

    # 체크포인트 설정
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    save_best: bool = True

    # 최적화 설정
    use_optimizer: bool = True
    optimizer_config: Optional[OptimizerConfig] = None


class ModelTrainer:
    """모델 훈련 클래스"""

    def __init__(self, model: BaseModel, config: Optional[TrainingConfig] = None):
        """
        모델 훈련기 초기화

        Args:
            model: 훈련할 모델
            config: 훈련 설정
        """
        self.model = model
        self.config = config or TrainingConfig()

        # 디렉토리 설정
        self.model_dir = Path(self.config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 세션 ID 생성 (실행 시간 기반)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 성능 최적화기
        if self.config.use_optimizer:
            self.optimizer = get_training_optimizer(self.config.optimizer_config)
        else:
            self.optimizer = None

        logger.info("모델 훈련기 초기화 완료")

    def train(
        self,
        train_data: List[LotteryNumber],
        valid_data: Optional[List[LotteryNumber]] = None,
    ) -> Dict[str, Any]:
        """
        모델 훈련

        Args:
            train_data: 훈련 데이터
            valid_data: 검증 데이터 (None이면 train_data에서 분할)

        Returns:
            훈련 결과
        """
        logger.info(f"모델 훈련 시작: {self.model.__class__.__name__}")
        start_time = time.time()

        try:
            # 검증 데이터 준비
            if valid_data is None and len(train_data) > 10:
                split_idx = int(len(train_data) * (1 - self.config.validation_split))
                valid_data = train_data[split_idx:]
                train_data = train_data[:split_idx]
                logger.info(
                    f"검증 데이터 분할: 훈련 {len(train_data)}개, 검증 {len(valid_data)}개"
                )

            # 모델 최적화 (성능 최적화기 사용)
            if self.optimizer:
                logger.info("성능 최적화 적용 중...")
                # 훈련 함수 래핑
                train_fn = lambda params: self._train_with_params(
                    train_data, valid_data, params
                )

                # 하이퍼파라미터 그리드
                param_grid = self._get_param_grid()

                # 훈련 함수 최적화
                best_params, optimized_train_fn = (
                    self.optimizer.optimize_training_function(train_fn, param_grid)
                )

                # 최적화된 함수로 훈련
                result = optimized_train_fn(train_data, valid_data)
            else:
                # 일반 훈련
                result = self._train_internal(train_data, valid_data)

            # 훈련 시간 추가
            training_time = time.time() - start_time
            result["training_time"] = training_time

            logger.info(f"모델 훈련 완료: {training_time:.2f}초")
            self._save_training_result(result)

            return result

        except Exception as e:
            logger.error(f"훈련 중 오류: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

            return {
                "success": False,
                "error": str(e),
                "training_time": time.time() - start_time,
            }

    def _train_internal(
        self,
        train_data: List[LotteryNumber],
        valid_data: Optional[List[LotteryNumber]] = None,
    ) -> Dict[str, Any]:
        """
        내부 훈련 로직

        Args:
            train_data: 훈련 데이터
            valid_data: 검증 데이터

        Returns:
            훈련 결과
        """
        # 구현체 불러오기
        RealRLModel = get_real_rl_model() if "get_real_rl_model" in globals() else None
        RealStatisticalModel = (
            get_real_statistical_model()
            if "get_real_statistical_model" in globals()
            else None
        )

        # 모델 타입에 따른 훈련 수행
        if RealRLModel and isinstance(self.model, RealRLModel):
            # RL 모델 훈련
            result = self.model.train(
                train_data,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                valid_data=valid_data,
            )
        elif RealStatisticalModel and isinstance(self.model, RealStatisticalModel):
            # 통계 모델 훈련
            result = self.model.train(
                train_data,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                valid_data=valid_data,
            )
        else:
            # 기본 훈련 인터페이스 사용
            result = self.model.train(
                train_data,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
            )

        return result

    def _train_with_params(
        self,
        train_data: List[LotteryNumber],
        valid_data: Optional[List[LotteryNumber]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        지정된 하이퍼파라미터로 훈련

        Args:
            train_data: 훈련 데이터
            valid_data: 검증 데이터
            params: 하이퍼파라미터

        Returns:
            훈련 결과
        """
        logger.info(f"하이퍼파라미터 세트로 훈련: {params}")

        # 구현체 불러오기
        RealRLModel = get_real_rl_model() if "get_real_rl_model" in globals() else None
        RealStatisticalModel = (
            get_real_statistical_model()
            if "get_real_statistical_model" in globals()
            else None
        )

        # 모델 클론 (깊은 복사)
        try:
            # 원본 모델의 가중치 백업 (나중에 복원하기 위해)
            original_state_dict = None
            try:
                if hasattr(self.model, "state_dict") and callable(
                    getattr(self.model, "state_dict", None)
                ):
                    original_state_dict = self.model.state_dict()  # type: ignore
            except Exception as e:
                logger.warning(f"모델 상태 저장 중 오류: {str(e)}")

            # 모델 타입에 따른 훈련
            if RealRLModel and isinstance(self.model, RealRLModel):
                result = self.model.train(
                    train_data,
                    epochs=params.get("epochs", self.config.epochs),
                    batch_size=params.get("batch_size", self.config.batch_size),
                    learning_rate=params.get(
                        "learning_rate", self.config.learning_rate
                    ),
                    valid_data=valid_data,
                )
            elif RealStatisticalModel and isinstance(self.model, RealStatisticalModel):
                result = self.model.train(
                    train_data,
                    learning_rate=params.get(
                        "learning_rate", self.config.learning_rate
                    ),
                    batch_size=params.get("batch_size", self.config.batch_size),
                    epochs=params.get("epochs", self.config.epochs),
                    valid_data=valid_data,
                )
            else:
                # 기본 훈련 인터페이스 사용
                result = self.model.train(train_data, **params)

            # 모델 가중치 복원 (튜닝 중에만)
            try:
                if (
                    hasattr(self.model, "load_state_dict")
                    and callable(getattr(self.model, "load_state_dict", None))
                    and original_state_dict is not None
                ):
                    self.model.load_state_dict(original_state_dict)  # type: ignore
            except Exception as e:
                logger.warning(f"모델 상태 복원 중 오류: {str(e)}")

            # 모델 평가
            if valid_data:
                eval_result = self.model.evaluate(valid_data)
                result.update(eval_result)

            # 최적화 지표 추출
            metric = self._get_optimization_metric(result)
            result["optimization_metric"] = metric

            return result

        except Exception as e:
            logger.error(f"하이퍼파라미터 훈련 중 오류: {str(e)}")
            return {"error": str(e), "optimization_metric": float("inf")}

    def _get_optimization_metric(self, result: Dict[str, Any]) -> float:
        """
        최적화 지표 추출

        Args:
            result: 훈련 결과

        Returns:
            최적화 지표 값 (낮을수록 좋음)
        """
        # 가능한 지표 확인
        if "loss" in result:
            return float(result["loss"])
        elif "avg_loss" in result:
            return float(result["avg_loss"])
        elif "avg_reward" in result and isinstance(result["avg_reward"], (int, float)):
            # 보상은 높을수록 좋으므로 부호 반전
            return -float(result["avg_reward"])
        else:
            # 기본 지표 - 성공 여부
            return 0.0 if result.get("success", False) else float("inf")

    def _get_param_grid(self) -> Dict[str, List[Any]]:
        """
        하이퍼파라미터 그리드 정의

        Returns:
            하이퍼파라미터 그리드
        """
        # 구현체 불러오기
        RealRLModel = get_real_rl_model() if "get_real_rl_model" in globals() else None
        RealStatisticalModel = (
            get_real_statistical_model()
            if "get_real_statistical_model" in globals()
            else None
        )

        # 모델 타입에 따른 파라미터 그리드
        if RealRLModel and isinstance(self.model, RealRLModel):
            return {
                "learning_rate": [0.01, 0.001, 0.0001],
                "batch_size": [16, 32, 64, 128],
                "gamma": [0.9, 0.95, 0.99],
                "epsilon_decay": [0.95, 0.99, 0.995],
            }
        elif RealStatisticalModel and isinstance(self.model, RealStatisticalModel):
            return {
                "learning_rate": [0.01, 0.001, 0.0001],
                "batch_size": [16, 32, 64],
                "weight_recent": [0.5, 0.7, 0.9],
                "smoothing_factor": [0.1, 0.3, 0.5],
            }
        else:
            # 기본 파라미터 그리드
            return {
                "learning_rate": [0.01, 0.001, 0.0001],
                "batch_size": [16, 32, 64],
                "epochs": [50, 100, 200],
            }

    def _save_training_result(self, result: Dict[str, Any]) -> None:
        """
        훈련 결과 저장

        Args:
            result: 훈련 결과
        """
        if not self.config.save_checkpoints:
            return

        try:
            # 결과 파일 경로
            result_file = (
                self.log_dir
                / f"training_{self.model.__class__.__name__}_{self.session_id}.json"
            )

            # JSON 직렬화 가능한 형태로 변환
            serializable_result = {}
            for k, v in result.items():
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    serializable_result[k] = v
                else:
                    serializable_result[k] = str(v)

            # 저장
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)

            logger.info(f"훈련 결과 저장 완료: {result_file}")

        except Exception as e:
            logger.error(f"훈련 결과 저장 중 오류: {str(e)}")


class TrainInterface:
    """훈련 인터페이스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        훈련 인터페이스 초기화

        Args:
            config: 설정 객체
        """
        self.config = config or {}
        self.trainers = {}

        # AutoTuner 초기화
        try:
            from ..utils.auto_tuner import AutoTuner

            self.auto_tuner = AutoTuner(config)  # type: ignore
        except ImportError:
            logger.warning(
                "AutoTuner를 임포트할 수 없습니다. 자동 튜닝이 비활성화됩니다."
            )
            self.auto_tuner = None

        logger.info("훈련 인터페이스 초기화 완료")

    def train_all_models(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]] = None,
    ) -> Dict[str, Any]:
        """
        모든 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터

        Returns:
            각 모델의 훈련 결과
        """
        results = {}

        # 패턴 분석 수행
        pattern_analysis = self._analyze_patterns(train_data)

        # RL 모델 훈련
        results["rl"] = self.train_rl(train_data, val_data, pattern_analysis)

        # LSTM 모델 훈련
        results["lstm"] = self.train_lstm(train_data, val_data)

        # GNN 모델 훈련
        results["gnn"] = self.train_gnn(train_data, val_data, pattern_analysis)

        # 통계 모델 훈련
        results["statistical"] = self.train_statistical(train_data, val_data)

        return results

    def train_rl(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]] = None,
        pattern_analysis: Optional[PatternAnalysis] = None,
    ) -> Dict[str, Any]:
        """
        강화학습 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과

        Returns:
            훈련 결과

        Raises:
            ImportError: 훈련기 모듈을 찾을 수 없는 경우
            RuntimeError: 훈련 중 오류가 발생한 경우
        """
        logger.info("강화학습 모델 훈련 시작")

        # 자동 하이퍼파라미터 튜닝
        rl_config = {}
        if self.auto_tuner:
            try:
                rl_config = self.auto_tuner.tune_hyperparameters(
                    "rl", train_data, val_data
                )
            except Exception as e:
                logger.error(f"하이퍼파라미터 튜닝 오류: {str(e)}")
                raise RuntimeError(f"RL 모델 하이퍼파라미터 튜닝 실패: {str(e)}")

        # 훈련기 초기화
        try:
            from ..training.train_rl import RLTrainer

            trainer = RLTrainer(rl_config)
        except ImportError as e:
            logger.error(f"RLTrainer 모듈 임포트 실패: {str(e)}")
            raise ImportError(f"RLTrainer 모듈을 찾을 수 없습니다: {str(e)}")

        # 훈련 실행
        try:
            # 훈련 파라미터 설정
            train_kwargs = {
                "validation_data": val_data,
                "pattern_analysis": pattern_analysis,
            }

            # 훈련 실행
            result = trainer.train(train_data, **train_kwargs)

            # 훈련기 저장
            self.trainers["rl"] = trainer

            logger.info("강화학습 모델 훈련 완료")
            return result

        except Exception as e:
            logger.error(f"RL 모델 훈련 실패: {str(e)}")
            # 실패 시 스택 트레이스 로깅
            import traceback

            logger.error(traceback.format_exc())
            raise RuntimeError(f"RL 모델 훈련 실패: {str(e)}")

    def train_lstm(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]] = None,
    ) -> Dict[str, Any]:
        """
        LSTM 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터

        Returns:
            훈련 결과

        Raises:
            ImportError: 트레이너 모듈을 찾을 수 없는 경우
            KeyError: 필수 설정값이 없는 경우
            RuntimeError: 훈련 중 오류가 발생한 경우
            IOError: 성능 보고서 저장 실패 시
        """
        logger.info("LSTM 모델 훈련 시작")

        try:
            # LSTM 트레이너 초기화
            from ..training.train_lstm import LSTMTrainer

            # 필수 설정값 목록
            required_keys = [
                "batch_size",
                "learning_rate",
                "epochs",
                "sequence_length",
                "dropout",
                "hidden_dim",
                "use_amp",
                "model_dir",
                "use_cuda",
                "seed",
                "patience",
                "augmentation",
            ]

            # 필수 설정값 확인
            config_dict = {}
            for key in required_keys:
                if key not in self.config:
                    raise KeyError(f"LSTM 모델 필수 설정값 누락: {key}")
                config_dict[key] = self.config[key]

            # CUDA 사용 가능 여부 확인
            config_dict["use_cuda"] = (
                config_dict["use_cuda"] and torch.cuda.is_available()
            )

            # ConfigProxy 생성
            from ..utils.config_loader import ConfigProxy

            config_proxy = ConfigProxy(config_dict)

            # 트레이너 생성
            trainer = LSTMTrainer(config_proxy)

            # 훈련 시작
            start_time = time.time()
            result = trainer.train(train_data, validation_data=val_data)
            duration = time.time() - start_time

            # 트레이너 저장
            self.trainers["lstm"] = trainer

            # 결과에 추가 정보 포함
            if isinstance(result, dict):
                result.update(
                    {
                        "model_type": "lstm",
                        "duration": duration,
                        "config": config_dict,
                    }
                )
            else:
                result = {
                    "model_type": "lstm",
                    "duration": duration,
                    "success": True,
                    "config": config_dict,
                }

            # 성능 리포트 작성
            from ..utils.report_writer import save_performance_report

            report_data = {
                "model_type": "lstm",
                "training_time": duration,
                "epochs": config_dict["epochs"],
                "batch_size": config_dict["batch_size"],
                "loss": result.get("best_valid_loss", 0),
                "use_amp": config_dict["use_amp"],
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }

            save_performance_report(report_data, "lstm_training_report")
            logger.info("LSTM 모델 성능 리포트 작성 완료")

            logger.info(f"LSTM 모델 훈련 완료: {duration:.2f}초")
            return result

        except ImportError as e:
            logger.error(f"LSTM 트레이너 모듈 임포트 실패: {str(e)}")
            raise ImportError(f"LSTM 트레이너 모듈을 찾을 수 없습니다: {str(e)}")
        except KeyError as e:
            logger.error(f"LSTM 모델 설정값 오류: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"LSTM 모델 훈련 중 오류: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            raise RuntimeError(f"LSTM 모델 훈련 실패: {str(e)}")

    def train_gnn(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]] = None,
        pattern_analysis: Optional[PatternAnalysis] = None,
    ) -> Dict[str, Any]:
        """
        그래프 신경망 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과

        Returns:
            훈련 결과
        """
        logger.info("GNN 모델 훈련 시작")

        # 자동 하이퍼파라미터 튜닝
        gnn_config = {}
        if self.auto_tuner:
            try:
                gnn_config = self.auto_tuner.tune_hyperparameters("gnn", train_data, val_data)  # type: ignore
            except Exception as e:
                logger.warning(f"하이퍼파라미터 튜닝 오류: {str(e)}")

        # 훈련기 초기화
        try:
            from ..training.train_gnn import GNNTrainer

            trainer = GNNTrainer(gnn_config)
        except ImportError:
            logger.error("GNNTrainer를 임포트할 수 없습니다.")
            return {"error": "GNNTrainer 모듈을 찾을 수 없습니다."}

        # 데이터 로드
        data_path = str(Path(__file__).parent.parent.parent / "data" / "lottery.csv")
        trainer.load_data(data_path)

        # 훈련 실행 (다양한 매개변수 이름 지원)
        try:
            # 매개변수를 kwargs로 전달하여 유연하게 처리
            train_kwargs = {
                "validation_data": val_data,
                "valid_data": val_data,
                "val_data": val_data,
                "pattern_analysis": pattern_analysis,
            }

            # 키워드 인자로 훈련 실행
            result = trainer.train(train_data, **train_kwargs)
        except Exception as e:
            try:
                # 위치 인자로 시도
                result = trainer.train(train_data, val_data, pattern_analysis)
            except Exception as e2:
                logger.error(f"GNN 모델 훈련 실패: {str(e)}, {str(e2)}")
                return {"error": f"훈련 실패: {str(e2)}"}

        # 훈련기 저장
        self.trainers["gnn"] = trainer

        logger.info("GNN 모델 훈련 완료")
        return result

    def train_statistical(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]] = None,
    ) -> Dict[str, Any]:
        """
        통계 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터

        Returns:
            훈련 결과
        """
        logger.info("통계 모델 훈련 시작")

        # 로또 훈련기 초기화 (다양한 모듈 경로 시도)
        trainer = None
        error_msgs = []

        # 가능한 임포트 경로 목록
        import_paths = [
            ("src.training.statistical_trainer", "LotteryTrainer"),
            ("src.training.train_statistical", "LotteryTrainer"),
            ("training.statistical_trainer", "LotteryTrainer"),
            ("training.train_statistical", "LotteryTrainer"),
        ]

        # 경로 시도
        for module_path, class_name in import_paths:
            try:
                module = __import__(module_path, fromlist=[class_name])
                trainer_class = getattr(module, class_name)
                trainer = trainer_class(self.config)
                break
            except (ImportError, AttributeError) as e:
                error_msgs.append(f"{module_path}.{class_name}: {str(e)}")

        # 임포트 실패 처리
        if trainer is None:
            logger.error(f"LotteryTrainer를 임포트할 수 없습니다: {error_msgs}")
            return {"error": "통계 모델 훈련기를 찾을 수 없습니다."}

        # 훈련 실행 (유연한 메서드 호출)
        result = None
        error_msgs = []

        # 다양한 메서드 이름 시도
        methods = ["train_statistical_model", "train_statistical", "train"]

        for method_name in methods:
            if hasattr(trainer, method_name) and callable(
                getattr(trainer, method_name)
            ):
                try:
                    # 메서드 실행 (키워드 인자 포함)
                    kwargs = (
                        {"validation_data": val_data} if val_data is not None else {}
                    )
                    method = getattr(trainer, method_name)
                    result = method(train_data, **kwargs)
                    break
                except Exception as e:
                    error_msgs.append(f"{method_name}: {str(e)}")
                    try:
                        # 위치 인자만으로 시도
                        result = method(train_data, val_data)
                        break
                    except Exception as e2:
                        error_msgs.append(f"{method_name} (위치 인자): {str(e2)}")

        # 모든 메서드 호출 실패 처리
        if result is None:
            logger.error(f"통계 모델 훈련 메서드 호출 실패: {error_msgs}")
            return {"error": "통계 모델 훈련 실패: 호환되는 메서드를 찾을 수 없습니다."}

        # 훈련기 저장
        self.trainers["statistical"] = trainer

        logger.info("통계 모델 훈련 완료")
        return result

    def evaluate_all_models(self, test_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        모든 모델 평가

        Args:
            test_data: 테스트 데이터

        Returns:
            각 모델의 평가 결과
        """
        results = {}

        # 각 모델 평가
        for model_name, trainer in self.trainers.items():
            results[model_name] = self.evaluate_model(model_name, test_data)

        return results

    def evaluate_model(
        self, model_name: str, test_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        특정 모델 평가

        Args:
            model_name: 모델 이름
            test_data: 테스트 데이터

        Returns:
            평가 결과
        """
        logger.info(f"{model_name} 모델 평가 시작")

        # 훈련기 가져오기
        trainer = self.trainers.get(model_name)
        if not trainer:
            logger.warning(f"{model_name} 모델이 훈련되지 않았습니다.")
            return {"error": f"{model_name} 모델이 훈련되지 않았습니다."}

        # 모델 평가
        try:
            if hasattr(trainer, "evaluate"):
                result = trainer.evaluate(test_data)
                logger.info(f"{model_name} 모델 평가 완료")
                return result
            else:
                logger.warning(f"{model_name} 모델에는 평가 메소드가 없습니다.")
                return {"error": f"{model_name} 모델에는 평가 메소드가 없습니다."}
        except Exception as e:
            logger.error(f"{model_name} 모델 평가 중 오류: {str(e)}")
            return {"error": str(e)}

    def _analyze_patterns(self, data: List[LotteryNumber]) -> PatternAnalysis:
        """
        패턴 분석 수행

        Args:
            data: 분석할 데이터

        Returns:
            패턴 분석 결과
        """
        # 패턴 분석기 사용
        try:
            from ..analysis.pattern_analyzer import PatternAnalyzer

            analyzer = PatternAnalyzer()
            return analyzer.analyze_patterns(data)
        except ImportError:
            # 패턴 분석기 없을 경우 기본 구현
            logger.warning(
                "PatternAnalyzer를 임포트할 수 없습니다. 기본 패턴 분석을 수행합니다."
            )
            # 간단한 기본 분석 결과 반환
            return cast(
                PatternAnalysis,
                {
                    "frequency_map": {},
                    "recency_map": {},
                    "pair_frequency": {},
                    "hot_numbers": set(),
                    "cold_numbers": set(),
                    "trending_numbers": [],
                    "clusters": [],
                    "roi_matrix": {},
                },
            )


# 싱글톤 인스턴스
_train_interface_instance = None


def get_train_interface(config: Optional[Dict[str, Any]] = None) -> TrainInterface:
    """
    훈련 인터페이스 인스턴스 반환 (싱글톤)

    Args:
        config: 설정 객체

    Returns:
        TrainInterface 인스턴스
    """
    global _train_interface_instance

    if _train_interface_instance is None:
        _train_interface_instance = TrainInterface(config)

    return _train_interface_instance


def train_models(args, config):
    """
    모델 훈련 진입점 함수

    Args:
        args: 명령행 인자 (mode 포함)
        config: 설정 객체

    Returns:
        훈련 결과

    Raises:
        ValueError: 지원되지 않는 모드 또는 설정 오류 시
        RuntimeError: 훈련 중 오류 발생 시
        IOError: 데이터 로드 실패 시
    """
    logger.info("모델 훈련 시작")

    # 모드 확인
    mode = getattr(args, "mode", 1)

    # 설정 객체 확인
    if not config:
        logger.warning("설정 객체가 없습니다. 기본 설정을 사용합니다.")
        config = {}

    # 데이터 로드 로직 (데이터는 메모리에서 가져와야 함)
    from lottery_run import load_global_data, get_data_split

    # 이미 로드된 데이터가 있는지 확인
    try:
        data = load_global_data()
        if not data:
            logger.error("데이터를 로드할 수 없습니다.")
            raise IOError("데이터 로드 실패")

        # 데이터 분할
        train_data, val_data, test_data = get_data_split()
        logger.info(
            f"데이터 분할 완료: 훈련={len(train_data)}, 검증={len(val_data)}, 테스트={len(test_data)}"
        )

    except Exception as e:
        logger.error(f"데이터 로드 중 오류: {str(e)}")
        raise IOError(f"데이터 로드 실패: {str(e)}")

    # 모드에 따라 처리
    try:
        # ConfigProxy 생성
        from ..utils.config_loader import ConfigProxy

        config_proxy = ConfigProxy(config)

        # 1: 통합 트레이너 사용
        if mode == 1:  # 전체 모델 학습
            logger.info("모든 모델 학습 모드")

            # UnifiedTrainer 사용
            try:
                from ..training.unified_trainer import UnifiedTrainer
                from ..models.lstm_model import LSTMModel

                trainer = UnifiedTrainer(
                    model_class=LSTMModel,
                    config=config_proxy,
                    model_type="lstm",  # 기본 모델 타입
                )

                # 모든 모델 학습
                result = trainer.train(
                    train_data=train_data,
                    val_data=val_data,
                    model_types=None,  # None은 모든 모델을 의미
                )

                return result

            except ImportError:
                # TrainInterface 사용 시도
                logger.info("UnifiedTrainer를 찾을 수 없어 TrainInterface 사용")
                interface = TrainInterface(config)
                return interface.train_all_models(train_data, val_data)

        # 2: RL 모델 학습
        elif mode == 2:
            logger.info("RL 모델 학습 모드")

            try:
                # UnifiedTrainer 사용 시도
                from ..training.unified_trainer import UnifiedTrainer
                from ..models.lstm_model import LSTMModel

                trainer = UnifiedTrainer(
                    model_class=LSTMModel, config=config_proxy, model_type="lstm"
                )

                # RL 모델만 학습
                result = trainer.train(
                    train_data=train_data, val_data=val_data, model_types=["rl"]
                )

                return result

            except ImportError:
                # RLTrainer 직접 사용
                logger.info("UnifiedTrainer를 찾을 수 없어 RLTrainer 직접 사용")
                from ..training.train_rl import RLTrainer

                trainer = RLTrainer(config_proxy)

                # 임시 데이터 파일 생성 없이 직접 훈련
                return trainer.train(data=train_data, validation_data=val_data)

        # 3: GNN 모델 학습
        elif mode == 3:
            logger.info("GNN 모델 학습 모드")

            try:
                # UnifiedTrainer 사용 시도
                from ..training.unified_trainer import UnifiedTrainer
                from ..models.lstm_model import LSTMModel

                trainer = UnifiedTrainer(
                    model_class=LSTMModel, config=config_proxy, model_type="lstm"
                )

                # GNN 모델만 학습
                result = trainer.train(
                    train_data=train_data, val_data=val_data, model_types=["gnn"]
                )

                return result

            except ImportError:
                # GNNTrainer 직접 사용
                logger.info("UnifiedTrainer를 찾을 수 없어 GNNTrainer 직접 사용")
                from ..training.train_gnn import GNNTrainer

                trainer = GNNTrainer(config_proxy)

                # 임시 데이터 파일 생성 필요 없음
                result = trainer.train(data=train_data, validation_data=val_data)

                return result

        # 4: 통계 모델 학습
        elif mode == 4:
            logger.info("통계 모델 학습 모드")

            # TrainInterface 사용 (UnifiedTrainer는 통계 모델 지원 안함)
            interface = TrainInterface(config)
            return interface.train_statistical(train_data, val_data)

        # 5: LSTM 모델 학습
        elif mode == 5:
            logger.info("LSTM 모델 학습 모드")

            try:
                # UnifiedTrainer 사용 시도
                from ..training.unified_trainer import UnifiedTrainer
                from ..models.lstm_model import LSTMModel

                trainer = UnifiedTrainer(
                    model_class=LSTMModel, config=config_proxy, model_type="lstm"
                )

                # LSTM 모델만 학습
                result = trainer.train(
                    train_data=train_data, val_data=val_data, model_types=["lstm"]
                )

                return result

            except ImportError:
                # LSTMTrainer 직접 사용
                logger.info("UnifiedTrainer를 찾을 수 없어 LSTMTrainer 직접 사용")
                from ..training.train_lstm import LSTMTrainer

                trainer = LSTMTrainer(config_proxy)

                # 데이터 설정
                trainer.set_data(train_data)

                # 모델 훈련
                result = trainer.train(data=train_data, validation_data=val_data)

                return result

        else:
            logger.error(f"알 수 없는 모드: {mode}")
            raise ValueError(f"지원되지 않는 훈련 모드: {mode}")

    except Exception as e:
        logger.error(f"모델 훈련 중 오류: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise RuntimeError(f"모델 훈련 실패: {str(e)}")
