# type: ignore
"""
로또 번호 예측 모델 통합 훈련 모듈

여러 모델의 훈련을 통합적으로 관리하고 결과를 합치는 기능을 제공합니다.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Type
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..shared.types import LotteryNumber, PatternAnalysis, ModelPrediction
from ..models.base_model import BaseModel
from ..models.rl_model import RLModel
from ..models.gnn_model import GNNModel
from ..utils.error_handler_refactored import get_logger
from ..utils.dynamic_batch_size_utils import get_safe_batch_size
from ..analysis.pattern_analyzer import PatternAnalyzer
from ..training.train_rl import RLTrainer
from ..training.train_rl_extended import EnhancedRLTrainer
from ..core.state_vector_builder import StateVectorBuilder
from ..utils.config_loader import ConfigProxy
from ..utils.cuda_optimizers import AMPTrainer
from ..utils.performance_report_writer import save_report
from ..utils.model_saver import save_model
from .base_trainer import BaseTrainer
from ..utils.unified_performance import performance_monitor
from ..utils.profiler import Profiler

# 로거 설정
logger = get_logger(__name__)


class UnifiedTrainer(BaseTrainer):
    """
    통합 훈련 관리자

    여러 모델의 훈련을 통합적으로 관리하고 결과를 합치는 클래스입니다.
    """

    def __init__(
        self, model_class: Type[torch.nn.Module], config: ConfigProxy, model_type: str
    ):
        super().__init__(config)
        self.model_class = model_class
        self.model_type = model_type
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self.trainers = {}  # 추가: 트레이너를 저장할 딕셔너리
        self.models = {}  # 추가: 모델을 저장할 딕셔너리
        self.model_dir = Path(self.config.safe_get("model_dir", "savedModels"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 패턴 분석기 초기화
        from ..analysis.pattern_analyzer import PatternAnalyzer

        self.pattern_analyzer = PatternAnalyzer(config)

    def set_data(self, draw_data: List[LotteryNumber]) -> None:
        """학습 데이터를 설정합니다."""
        super().set_data(draw_data)
        # 데이터 로더 설정 로직 구현
        # TODO: 각 모델 타입에 맞는 데이터 로더 구현

    def train_base(self) -> None:
        """모델을 기본 훈련 루프로 훈련합니다."""
        if self._data is None or self.model is None or self.optimizer is None:
            raise ValueError("훈련 데이터, 모델 또는 옵티마이저가 설정되지 않았습니다.")

        # 기본 훈련 에폭 실행
        self._train_epoch()

    def base_evaluate(self) -> Dict[str, float]:
        """기본 모델 평가 메서드"""
        if self._data is None or self.model is None:
            raise ValueError("평가 데이터나 모델이 설정되지 않았습니다.")

        return self._validate()

    def base_predict(self) -> List[List[int]]:
        """기본 예측 메서드"""
        if self._data is None or self.model is None:
            raise ValueError("예측 데이터나 모델이 설정되지 않았습니다.")

        predictions = []
        self.model.eval()

        try:
            # 데이터로더가 None이면 에러 처리
            if self._test_loader is None:
                return []

            with torch.no_grad():
                for batch in self._test_loader:
                    output = self.model(batch)
                    # TODO: 각 모델 타입에 맞는 예측 처리 구현
                    predictions.extend(output.tolist())

            return predictions
        except Exception as e:
            self.logger.error(f"예측 중 오류: {str(e)}")
            return []

    def train(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]] = None,
        pattern_analysis: Optional[PatternAnalysis] = None,
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과
            model_types: 훈련할 모델 유형 목록

        Returns:
            훈련 결과
        """
        results = {}

        # 기본 모델 유형 설정
        if model_types is None:
            model_types = ["rl", "enhanced_rl", "gnn", "lstm"]

        # 패턴 분석이 제공되지 않은 경우 수행
        if pattern_analysis is None:
            self.logger.info("패턴 분석 수행 중...")
            pattern_analysis = self.pattern_analyzer.analyze_patterns(train_data)

        # 검증 데이터가 없는 경우 훈련 데이터의 일부를 사용
        if val_data is None and train_data:
            val_size = max(1, int(len(train_data) * 0.1))  # 10%
            val_data = train_data[-val_size:]
            train_data = train_data[:-val_size]

        # 각 모델 유형에 대해 훈련 수행
        for model_type in model_types:
            try:
                self.logger.info(f"{model_type} 모델 훈련 시작")

                # 모델 유형에 따라 다른 훈련 메서드 호출
                if model_type == "rl":
                    results[model_type] = self.train_rl_model(
                        train_data, val_data, pattern_analysis
                    )
                elif model_type == "enhanced_rl":
                    results[model_type] = self.train_enhanced_rl_model(
                        train_data, val_data, pattern_analysis
                    )
                elif model_type == "gnn":
                    results[model_type] = self.train_gnn_model(
                        train_data, val_data, pattern_analysis
                    )
                elif model_type == "lstm":
                    results[model_type] = self.train_lstm_model(
                        train_data, val_data, pattern_analysis
                    )
                else:
                    self.logger.warning(f"알 수 없는 모델 유형: {model_type}")

            except Exception as e:
                self.logger.error(f"{model_type} 모델 훈련 중 오류 발생: {str(e)}")
                results[model_type] = {"success": False, "error": str(e)}

        return results

    def train_rl_model(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]],
        pattern_analysis: PatternAnalysis,
    ) -> Dict[str, Any]:
        """
        강화학습 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과

        Returns:
            훈련 결과
        """
        # RLTrainer 생성
        trainer = RLTrainer(
            config={
                "learning_rate": self.config.safe_get("rl_learning_rate", 0.001),
                "batch_size": self.config.safe_get("rl_batch_size", 32),
                "gamma": self.config.safe_get("rl_gamma", 0.99),
                "epsilon": self.config.safe_get("rl_epsilon", 0.1),
                "action_dim": 45,
                "state_dim": 256,
                "experience_replay": True,
                "max_memory": self.config.safe_get("rl_max_memory", 10000),
                "optimizer": "adam",
                "loss": "mse",
                "num_episodes": self.config.safe_get("rl_episodes", 1000),
                "model_dir": self.config.safe_get("model_dir", "savedModels"),
                "model_name": "rl_model",
                "use_amp": self.config.safe_get("use_amp", True),
                "early_stopping": self.config.safe_get("early_stopping", True),
                "patience": self.config.safe_get("rl_patience", 10),
                "save_interval": self.config.safe_get("save_interval", 10),
                "seed": self.config.safe_get("seed", 42),  # 시드 추가
            }
        )

        # 데이터 로드 (임시 파일 생성)
        temp_data_path = self._save_temp_data(train_data)
        if temp_data_path:
            trainer.load_data(temp_data_path)

        # 트레이너 저장
        self.trainers["rl"] = trainer

        # 훈련 실행
        self.logger.info("강화학습 모델 훈련 시작")
        start_time = time.time()

        # 훈련 설정
        num_episodes = self.config.safe_get("epochs", 100)
        batch_size = self.config.safe_get("batch_size", 32)

        # 훈련 실행
        training_kwargs = {
            "validation_data": val_data,
            "pattern_analysis": pattern_analysis,
            "episodes": num_episodes,  # RLTrainer는 episodes 매개변수 사용
            "batch_size": batch_size,
        }
        training_result = trainer.train(train_data, **training_kwargs)

        # 결과 요약
        duration = time.time() - start_time
        result = {
            "success": True,
            "duration": duration,
            "model_type": "rl",
            "best_reward": training_result.get("best_reward", 0),
            "best_episode": training_result.get("best_episode", 0),
        }

        self.logger.info(f"강화학습 모델 훈련 완료: {duration:.1f}초")

        # 모델 저장
        model_path = self.model_dir / "rl_model.pt"
        if hasattr(trainer, "model") and trainer.model is not None:
            try:
                trainer.model.save(str(model_path))
                self.logger.info(f"강화학습 모델 저장 완료: {model_path}")
            except Exception as e:
                self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")

        # 임시 파일 정리
        if hasattr(self, "_temp_data_path") and self._temp_data_path:
            try:
                os.unlink(self._temp_data_path)
                self._temp_data_path = None
            except Exception as e:
                self.logger.error(f"임시 파일 삭제 실패: {str(e)}")

        return result

    def _save_temp_data(self, data: List[LotteryNumber]) -> Optional[str]:
        """
        데이터를 임시 파일로 저장

        Args:
            data: 저장할 데이터

        Returns:
            임시 파일 경로 또는 None
        """
        try:
            import tempfile
            import json

            # LotteryNumber 데이터를 JSON으로 직렬화 가능한 형식으로 변환
            json_data = []
            for item in data:
                if isinstance(item, LotteryNumber):
                    json_data.append(
                        {
                            "draw_no": item.draw_no,
                            "numbers": item.numbers,
                            "date": item.date,
                        }
                    )
                else:
                    self.logger.warning(f"지원되지 않는 데이터 타입: {type(item)}")

            # 임시 파일 생성
            fd, temp_path = tempfile.mkstemp(suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            self._temp_data_path = temp_path
            self.logger.debug(f"임시 데이터 파일 생성: {temp_path}")
            return temp_path
        except Exception as e:
            self.logger.error(f"임시 데이터 파일 생성 실패: {str(e)}")
            return None

    def train_enhanced_rl_model(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]],
        pattern_analysis: PatternAnalysis,
    ) -> Dict[str, Any]:
        """
        향상된 강화학습 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과

        Returns:
            Dict[str, Any]: 훈련 결과
        """
        self.logger.info("향상된 강화학습 모델 훈련 시작")

        try:
            # EnhancedRLTrainer 생성
            from ..training.train_rl_extended import EnhancedRLTrainer

            trainer = EnhancedRLTrainer(
                config={
                    "learning_rate": self.config.safe_get("learning_rate", 0.001),
                    "batch_size": self.config.safe_get("batch_size", 32),
                    "use_amp": self.config.safe_get("use_amp", True),
                    "use_compile": self.config.safe_get("use_compile", True),
                    "seed": self.config.safe_get("seed", 42),
                }
            )

            # 트레이너 저장
            self.trainers["enhanced_rl"] = trainer

            # 임시 데이터 저장 및 로드
            temp_data_path = self._save_temp_data(train_data)
            if temp_data_path:
                trainer.load_data(temp_data_path)

            # 훈련 설정
            num_episodes = self.config.safe_get("epochs", 100)
            batch_size = self.config.safe_get("batch_size", 32)

            # 동적 배치 크기 조정 (메모리 최적화)
            from ..utils.dynamic_batch_size_utils import get_safe_batch_size

            adjusted_batch_size = get_safe_batch_size(
                initial_batch_size=batch_size,
                device=trainer.device if hasattr(trainer, "device") else None,
            )

            if adjusted_batch_size != batch_size:
                self.logger.info(
                    f"배치 크기 조정: {batch_size} → {adjusted_batch_size}"
                )
                batch_size = adjusted_batch_size

            # 훈련 실행
            start_time = time.time()
            training_result = trainer.train(
                data=train_data,
                validation_data=val_data,
                pattern_analysis=pattern_analysis,
                episodes=num_episodes,
                batch_size=batch_size,
            )
            duration = time.time() - start_time

            # 결과 요약
            result = {
                "success": True,
                "duration": duration,
                "model_type": "enhanced_rl",
                "best_reward": training_result.get("best_reward", 0),
                "episodes": num_episodes,
            }

            self.logger.info(f"향상된 강화학습 모델 훈련 완료: {duration:.1f}초")

            # 모델 저장
            model_path = self.model_dir / "enhanced_rl_model.pt"
            if hasattr(trainer, "model") and trainer.model is not None:
                try:
                    trainer.model.save(str(model_path))
                    self.logger.info(f"향상된 강화학습 모델 저장 완료: {model_path}")
                except Exception as e:
                    self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")

            # 임시 파일 정리
            if temp_data_path:
                try:
                    os.unlink(temp_data_path)
                    self.logger.debug(f"임시 파일 삭제 완료: {temp_data_path}")
                except Exception as e:
                    self.logger.error(f"임시 파일 삭제 실패: {str(e)}")

            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            self.logger.error(f"향상된 강화학습 모델 훈련 중 오류: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "model_type": "enhanced_rl"}

    def train_gnn_model(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]],
        pattern_analysis: PatternAnalysis,
    ) -> Dict[str, Any]:
        """
        그래프 신경망 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과

        Returns:
            Dict[str, Any]: 훈련 결과
        """
        self.logger.info("GNN 모델 훈련 시작")

        try:
            # GNNTrainer 가져오기
            from ..training.train_gnn import GNNTrainer

            # 훈련기 초기화
            trainer = GNNTrainer(
                config={
                    "learning_rate": self.config.safe_get("learning_rate", 0.001),
                    "batch_size": self.config.safe_get("batch_size", 32),
                    "epochs": self.config.safe_get("epochs", 100),
                    "use_amp": self.config.safe_get("use_amp", True),
                    "seed": self.config.safe_get("seed", 42),
                    "use_compile": self.config.safe_get(
                        "use_compile", False
                    ),  # GNN은 compile이 항상 작동하지 않음
                    "hidden_dim": self.config.safe_get("hidden_dim", 64),
                    "dropout": self.config.safe_get("dropout", 0.3),
                }
            )

            # 트레이너 저장
            self.trainers["gnn"] = trainer

            # 임시 데이터 저장 및 로드
            temp_data_path = self._save_temp_data(train_data)
            if temp_data_path:
                trainer.load_data(temp_data_path)

            # 훈련 설정 (동적 배치 크기 조정)
            batch_size = self.config.safe_get("batch_size", 32)
            epochs = self.config.safe_get("epochs", 100)

            from ..utils.dynamic_batch_size_utils import get_safe_batch_size

            adjusted_batch_size = get_safe_batch_size(
                initial_batch_size=batch_size,
                device=trainer.device if hasattr(trainer, "device") else None,
            )

            if adjusted_batch_size != batch_size:
                self.logger.info(
                    f"GNN 배치 크기 조정: {batch_size} → {adjusted_batch_size}"
                )
                trainer.config["batch_size"] = adjusted_batch_size

            # 훈련 실행
            start_time = time.time()
            training_result = trainer.train(
                data=train_data,
                validation_data=val_data,
                pattern_analysis=pattern_analysis,
            )
            duration = time.time() - start_time

            # 결과 요약
            result = {
                "success": True,
                "duration": duration,
                "model_type": "gnn",
                "epochs": epochs,
            }

            # 추가 메트릭이 있으면 병합
            if isinstance(training_result, dict):
                result.update(training_result)

            self.logger.info(f"GNN 모델 훈련 완료: {duration:.1f}초")

            # 모델 저장
            model_path = self.model_dir / "gnn_model.pt"
            if hasattr(trainer, "model") and trainer.model is not None:
                try:
                    trainer.model.save(str(model_path))
                    self.logger.info(f"GNN 모델 저장 완료: {model_path}")
                except Exception as e:
                    self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")

            # 임시 파일 정리
            if temp_data_path:
                try:
                    os.unlink(temp_data_path)
                    self.logger.debug(f"임시 파일 삭제 완료: {temp_data_path}")
                except Exception as e:
                    self.logger.error(f"임시 파일 삭제 실패: {str(e)}")

            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            self.logger.error(f"GNN 모델 훈련 중 오류: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "model_type": "gnn"}

    def train_lstm_model(
        self,
        train_data: List[LotteryNumber],
        val_data: Optional[List[LotteryNumber]],
        pattern_analysis: PatternAnalysis,
    ) -> Dict[str, Any]:
        """
        LSTM 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과

        Returns:
            훈련 결과
        """
        self.logger.info("LSTM 모델 훈련 시작")

        try:
            # LSTMTrainer 가져오기
            from ..training.train_lstm import LSTMTrainer

            # 훈련기 초기화
            trainer = LSTMTrainer(config=self.config)

            # 트레이너 저장
            self.trainers["lstm"] = trainer

            # 데이터 설정
            trainer.set_data(train_data)

            # 훈련 설정
            batch_size = self.config.safe_get("batch_size", 32)

            # 동적 배치 크기 조정 (메모리 최적화)
            from ..utils.dynamic_batch_size_utils import get_safe_batch_size

            adjusted_batch_size = get_safe_batch_size(
                initial_batch_size=batch_size,
                device=trainer.device if hasattr(trainer, "device") else None,
            )

            if adjusted_batch_size != batch_size:
                self.logger.info(
                    f"LSTM 배치 크기 조정: {batch_size} → {adjusted_batch_size}"
                )
                trainer.config.set("batch_size", adjusted_batch_size)

            # 훈련 실행
            start_time = time.time()
            training_result = trainer.train(data=train_data, validation_data=val_data)
            duration = time.time() - start_time

            # 결과 요약
            result = {
                "success": True,
                "duration": duration,
                "model_type": "lstm",
                "best_valid_loss": training_result.get("best_valid_loss", 0),
            }

            self.logger.info(f"LSTM 모델 훈련 완료: {duration:.1f}초")

            # 모델 저장
            model_path = self.model_dir / "lstm_model.pt"
            if hasattr(trainer, "model") and trainer.model is not None:
                try:
                    trainer.model.save(str(model_path))
                    self.logger.info(f"LSTM 모델 저장 완료: {model_path}")
                except Exception as e:
                    self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")

            # 성능 리포트 작성
            try:
                # 성능 추적기가 없으면 생성
                if not hasattr(self, "performance_tracker"):
                    from ..utils.unified_performance import performance_monitor

                    # 통합 성능 모니터링 사용

                # 프로파일러가 없으면 생성
                if not hasattr(self, "profiler"):
                    from ..utils.profiler import Profiler

                    self.profiler = Profiler(self.config)

                # 물리적 성능 정보만 저장하는 리포트 생성
                save_report(
                    self.profiler,
                    self.performance_tracker,
                    self.config,
                    "lstm_training",
                    {
                        "model_type": "lstm",
                        "training_time": duration,
                        "epochs": self.config.safe_get("epochs", 100),
                        "batch_size": adjusted_batch_size,
                        "valid_loss": training_result.get("best_valid_loss", 0),
                        "use_amp": self.config.safe_get("use_amp", True),
                    },
                    include_data_metrics=False,
                )
                self.logger.info("LSTM 모델 성능 리포트 작성 완료")
            except Exception as e:
                self.logger.error(f"성능 리포트 작성 중 오류: {str(e)}")

            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            self.logger.error(f"LSTM 모델 훈련 중 오류: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "model_type": "lstm"}

    def predict(
        self,
        model_type: str = "enhanced_rl",
        count: int = 5,
        pattern_analysis: Optional[PatternAnalysis] = None,
    ) -> List[ModelPrediction]:
        """
        예측 생성

        Args:
            model_type: 사용할 모델 유형
            count: 생성할 추천 조합 수
            pattern_analysis: 패턴 분석 결과

        Returns:
            예측 결과 목록
        """
        recommendations = []

        try:
            # 모델 유형 확인
            if model_type == "rl" and "rl" in self.trainers:
                # RL 모델 예측
                trainer = self.trainers["rl"]
                recommendations = trainer.generate_recommendations(count)

            elif model_type == "enhanced_rl" and "enhanced_rl" in self.trainers:
                # 향상된 RL 모델 예측
                trainer = self.trainers["enhanced_rl"]
                recommendations = trainer.generate_recommendations(count)

            elif model_type == "gnn" and "gnn" in self.models:
                # GNN 모델 예측
                model = self.models["gnn"]
                recommendations = model.predict(
                    count=count,
                    pattern_analysis=pattern_analysis,
                )

            elif model_type == "combined":
                # 모든 가용 모델의 예측 조합
                combined_recommendations = []

                # RL 모델
                if "rl" in self.trainers:
                    rl_recommendations = self.trainers["rl"].generate_recommendations(
                        count=max(1, count // 3)
                    )
                    combined_recommendations.extend(rl_recommendations)

                # 향상된 RL 모델
                if "enhanced_rl" in self.trainers:
                    erl_recommendations = self.trainers[
                        "enhanced_rl"
                    ].generate_recommendations(count=max(1, count // 3))
                    combined_recommendations.extend(erl_recommendations)

                # GNN 모델
                if "gnn" in self.models:
                    gnn_recommendations = self.models["gnn"].predict(
                        count=max(1, count // 3),
                        pattern_analysis=pattern_analysis,
                    )
                    combined_recommendations.extend(gnn_recommendations)

                # 결과 선택 및 정렬
                combined_recommendations.sort(key=lambda x: x.confidence, reverse=True)
                recommendations = combined_recommendations[:count]

            else:
                self.logger.error(f"사용 가능한 {model_type} 모델이 없습니다")
                return []

        except Exception as e:
            self.logger.error(f"예측 생성 중 오류 발생: {str(e)}")
            return []

        # 번호 정렬 및 중복 제거
        unique_recommendations = []
        seen_numbers = set()

        for rec in recommendations:
            # 중복 체크
            numbers_tuple = tuple(sorted(rec.numbers))
            if numbers_tuple not in seen_numbers:
                seen_numbers.add(numbers_tuple)

                # 번호 정렬 (오름차순)
                rec.numbers = sorted(rec.numbers)
                unique_recommendations.append(rec)

                # 충분한 수의 추천을 얻었을 경우 중단
                if len(unique_recommendations) >= count:
                    break

        return unique_recommendations[:count]

    def evaluate(
        self,
        test_data: List[LotteryNumber],
        model_type: str = "enhanced_rl",
        pattern_analysis: Optional[PatternAnalysis] = None,
    ) -> Dict[str, Any]:
        """
        모델 평가

        Args:
            test_data: 테스트 데이터
            model_type: 평가할 모델 유형
            pattern_analysis: 패턴 분석 결과

        Returns:
            평가 결과
        """
        # 패턴 분석 수행
        if pattern_analysis is None:
            pattern_analysis = self.pattern_analyzer.analyze_patterns(test_data)

        # 평가 결과
        result = {
            "model_type": model_type,
            "total_draws": len(test_data),
            "hit_counts": {i: 0 for i in range(7)},  # 0~6개 일치 횟수
            "avg_hits": 0.0,
            "roi": 0.0,
        }

        try:
            # 예측 생성
            recommendations = self.predict(
                model_type=model_type,
                count=self.config.safe_get("eval_recommendations", 10),
                pattern_analysis=pattern_analysis,
            )

            if not recommendations:
                return {"error": f"예측 생성 실패: {model_type}"}

            # 테스트 데이터와 비교
            total_hits = 0

            for draw in test_data:
                draw_set = set(draw.numbers)

                # 각 추천에 대해 맞힌 개수 계산
                max_hits = 0
                for rec in recommendations:
                    hits = len(draw_set.intersection(set(rec.numbers)))
                    max_hits = max(max_hits, hits)

                # 결과 집계
                result["hit_counts"][max_hits] += 1
                total_hits += max_hits

            # 평균 맞춘 개수
            result["avg_hits"] = total_hits / len(test_data) if test_data else 0

            # ROI 계산 (새로운 점수 체계)
            if len(test_data) > 0:
                # 각 등수별 상금 (새로운 점수 체계)
                prize_money = {
                    3: 1,  # 5등: 1점
                    4: 10,  # 4등: 10점
                    5: 40,  # 3등: 40점
                    6: 100,  # 1등: 100점
                }

                # 총 상금 계산
                total_prize = sum(
                    result["hit_counts"].get(hits, 0) * prize_money.get(hits, 0)
                    for hits in prize_money
                )

                # 총 투자 비용 (1,000원 * 추천 수 * 추첨 횟수)
                total_cost = len(recommendations) * len(test_data) * 1000

                # ROI 계산 ((수익 - 비용) / 비용)
                if total_cost > 0:
                    result["roi"] = (total_prize - total_cost) / total_cost

        except Exception as e:
            self.logger.error(f"모델 평가 중 오류 발생: {str(e)}")
            result["error"] = str(e)

        return result

    def cleanup(self) -> None:
        """자원 정리"""
        self.logger.info("자원 정리 시작")

        # 트레이너 자원 정리
        for trainer_name, trainer in self.trainers.items():
            if hasattr(trainer, "cleanup"):
                try:
                    trainer.cleanup()
                    self.logger.debug(f"{trainer_name} 트레이너 자원 정리 완료")
                except Exception as e:
                    self.logger.warning(
                        f"{trainer_name} 트레이너 자원 정리 실패: {str(e)}"
                    )

        # 모델 자원 정리
        for model_name, model in self.models.items():
            if hasattr(model, "cleanup"):
                try:
                    model.cleanup()
                    self.logger.debug(f"{model_name} 모델 자원 정리 완료")
                except Exception as e:
                    self.logger.warning(f"{model_name} 모델 자원 정리 실패: {str(e)}")

        # GPU 메모리 정리
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                self.logger.debug("CUDA 메모리 정리 완료")
            except Exception as e:
                self.logger.warning(f"CUDA 메모리 정리 실패: {str(e)}")

        # 메모리 정리
        gc.collect()

        self.logger.info("자원 정리 완료")

    def _cleanup_temp_files(self) -> None:
        """임시 파일 정리"""
        try:
            import tempfile
            import glob

            # 임시 디렉토리에서 특정 패턴의 파일 찾기
            temp_dir = tempfile.gettempdir()
            patterns = ["lottery_*.tmp", "train_*.npz", "model_*.pt", "data_*.csv"]

            files_to_remove = []
            for pattern in patterns:
                files_to_remove.extend(glob.glob(os.path.join(temp_dir, pattern)))

            # 3일 이상 지난 파일 삭제
            current_time = time.time()
            max_age = 3 * 24 * 60 * 60  # 3일(초 단위)

            removed_count = 0
            for file_path in files_to_remove:
                try:
                    if os.path.exists(file_path):
                        # 파일 수정 시간 확인
                        mtime = os.path.getmtime(file_path)
                        if current_time - mtime > max_age:
                            os.remove(file_path)
                            removed_count += 1
                except Exception as e:
                    self.logger.warning(
                        f"임시 파일 삭제 실패: {file_path}, 오류: {str(e)}"
                    )

            if removed_count > 0:
                self.logger.info(f"{removed_count}개 임시 파일 정리 완료")

        except Exception as e:
            self.logger.error(f"임시 파일 정리 중 오류 발생: {str(e)}")

    def _train_epoch(self) -> Dict[str, float]:
        """훈련 에폭 구현"""
        self.logger.info("훈련 에폭 실행 중")
        # 구현은 하위 클래스에서 담당
        return {"loss": 0.0}

    def _validate(self) -> Dict[str, float]:
        """검증 구현"""
        self.logger.info("검증 실행 중")
        # 구현은 하위 클래스에서 담당
        return {"val_loss": 0.0}
