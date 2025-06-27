# type: ignore
"""
모델 통합 및 앙상블 시스템

이 모듈은 다양한 머신러닝/딥러닝 모델을 통합하고 최적화하는 시스템을 제공합니다.
"""

import time
import torch
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from .memory_manager import MemoryManager, MemoryConfig
from .error_handler_refactored import get_logger
from dataclasses import dataclass
import os
import json
import numpy as np

from ..models.base_model import BaseModel
from ..models.rl_model import RLModel
from ..models.gnn_model import GNNModel
from ..shared.types import ModelPrediction, LotteryNumber

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """모델 설정 데이터 클래스"""

    name: str
    type: str
    params: Dict[str, Any]
    device: str
    batch_size: int
    precision: str
    use_amp: bool
    num_workers: int
    pin_memory: bool


@dataclass
class EnsembleConfig:
    """앙상블 설정 데이터 클래스"""

    method: str = "voting"
    weights: Optional[List[float]] = None
    n_splits: int = 5
    random_state: int = 42
    n_trials: int = 10
    timeout: int = 600


@dataclass
class IntegratorConfig:
    """통합기 설정"""

    batch_size: int = 32
    num_workers: int = 4
    enable_cache: bool = True
    cache_dir: str = "cache/integrator"
    max_memory_cache: int = 1 << 30  # 1GB
    max_disk_cache: int = 1 << 33  # 8GB
    model_timeout: float = 30.0
    enable_parallel: bool = True
    weight_update_interval: float = 1.0

    def __post_init__(self):
        # 캐시 디렉토리 경로를 상대 경로로 변환
        self.cache_dir = str(Path(__file__).parent.parent.parent / self.cache_dir)


class ModelWrapper:
    """모델 래퍼"""

    def __init__(self, model: BaseModel, name: str, weight: float = 1.0):
        self.model = model
        self.name = name
        self.weight = weight
        self.is_active = True
        self.error_count = 0
        self.last_inference_time = 0.0
        self.total_inferences = 0
        self.total_errors = 0

    def predict(self, inputs: Any) -> Any:
        """예측 수행"""
        try:
            start_time = time.time()
            outputs = self.model.predict(inputs)
            self.last_inference_time = time.time() - start_time
            self.total_inferences += 1
            return outputs
        except Exception as e:
            self.error_count += 1
            self.total_errors += 1
            logger.error(f"모델 {self.name} 예측 실패: {str(e)}")
            raise


class ModelIntegrator:
    """모델 통합 시스템"""

    def __init__(self, config: Optional[IntegratorConfig] = None):
        """
        모델 통합 시스템 초기화

        Args:
            config: 통합기 설정
        """
        self.config = config or IntegratorConfig()

        # MemoryConfig 초기화
        memory_config = MemoryConfig(
            max_memory_usage=0.8,  # 80% 메모리 사용
            cache_size=self.config.max_memory_cache,
            memory_track_interval=1,
            memory_frags_threshold=0.3,
            memory_usage_warning=0.8,
            num_workers=self.config.num_workers,
            prefetch_factor=2,
            pool_size=100,
            compression_threshold=1 << 20,  # 1MB
            alignment_size=256,
            use_memory_pooling=True,
            use_memory_alignment=True,
            use_memory_compression=True,
            use_memory_prefetching=True,
            use_memory_reuse=True,
            use_memory_tracking=True,
            use_memory_optimization=True,
            use_memory_compaction=True,
            use_memory_pinning=True,
            use_memory_streaming=True,
            use_memory_events=True,
            use_memory_graphs=True,
            use_memory_peer_access=True,
            use_memory_unified=False,
            use_memory_multi_gpu=False,
            gpu_ids=[0],
        )

        self.memory_manager = MemoryManager(memory_config)
        self.models: Dict[str, BaseModel] = {}
        self.model_weights: Dict[str, float] = {}
        self.model_queue = queue.Queue()
        self.weight_update_thread = None
        self.stop_event = threading.Event()
        self.recommendation_cache: Dict[str, List[ModelPrediction]] = {}
        self.weighting_strategy: str = (
            "uniform"  # 기본값: "uniform", "weighted", "stacked"
        )
        self.model_wrappers: Dict[str, ModelWrapper] = {}

        if self.config.enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)

        # 캐시 디렉토리 생성
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        self._start_weight_updater()

    def _start_weight_updater(self):
        """가중치 업데이트 스레드 시작"""
        if not self.config.enable_parallel:
            return

        def update_weights():
            while not self.stop_event.is_set():
                try:
                    self._update_model_weights()
                    time.sleep(self.config.weight_update_interval)
                except Exception as e:
                    logger.error(f"가중치 업데이트 중 오류: {str(e)}")

        self.weight_update_thread = threading.Thread(target=update_weights, daemon=True)
        self.weight_update_thread.start()

    def _update_model_weights(self):
        """모델 가중치 동적 업데이트"""
        # ModelWrapper 객체 기반 가중치 업데이트
        total_inferences = sum(
            wrapper.total_inferences for wrapper in self.model_wrappers.values()
        )
        if total_inferences == 0:
            return

        for wrapper in self.model_wrappers.values():
            if wrapper.total_inferences == 0:
                continue

            # 성공률 기반 가중치 조정
            success_rate = 1.0 - (wrapper.total_errors / wrapper.total_inferences)
            # 속도 기반 가중치 조정
            speed_factor = 1.0 / (wrapper.last_inference_time + 1e-6)

            # 새로운 가중치 계산
            new_weight = success_rate * speed_factor
            # 점진적 업데이트
            wrapper.weight = 0.9 * wrapper.weight + 0.1 * new_weight

    def register_model(
        self, model_id: str, model: BaseModel, weight: float = 1.0
    ) -> None:
        """
        모델 등록

        Args:
            model_id: 모델 식별자
            model: 등록할 모델
            weight: 모델 가중치
        """
        if model_id in self.models:
            logger.warning(
                f"모델 ID '{model_id}'가 이미 등록되어 있습니다. 덮어씁니다."
            )

        self.models[model_id] = model
        self.model_weights[model_id] = weight
        # ModelWrapper 생성 및 등록
        self.model_wrappers[model_id] = ModelWrapper(model, model_id, weight)

        logger.info(f"모델 '{model_id}' 등록 완료 (가중치: {weight})")

    def unregister_model(self, model_id: str) -> bool:
        """
        모델 등록 해제

        Args:
            model_id: 모델 식별자

        Returns:
            성공 여부
        """
        if model_id not in self.models:
            logger.warning(f"모델 ID '{model_id}'가 등록되어 있지 않습니다.")
            return False

        # 모델 제거
        del self.models[model_id]

        # 가중치 제거
        if model_id in self.model_weights:
            del self.model_weights[model_id]

        # ModelWrapper 제거
        if model_id in self.model_wrappers:
            del self.model_wrappers[model_id]

        logger.info(f"모델 '{model_id}' 등록 해제 완료")
        return True

    def load_models_from_directory(self, dir_path: str) -> int:
        """
        디렉토리에서 모델 로드

        Args:
            dir_path: 모델 파일이 있는 디렉토리 경로

        Returns:
            로드된 모델 수
        """
        model_path = Path(dir_path)
        if not model_path.exists() or not model_path.is_dir():
            logger.error(f"디렉토리가 존재하지 않습니다: {dir_path}")
            return 0

        # 모델 파일 찾기
        model_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.pth"))

        # 로드된 모델 수
        loaded_count = 0

        for model_file in model_files:
            try:
                # 파일 이름에서 모델 타입 추출
                file_name = model_file.stem

                if "rl" in file_name.lower():
                    # IntegratorConfig를 dict로 변환하여 전달
                    model_config = (
                        vars(self.config)
                        if hasattr(self.config, "__dict__")
                        else self.config
                    )
                    model = RLModel(config=model_config)
                    model_type = "RL"
                elif "gnn" in file_name.lower():
                    # IntegratorConfig를 dict로 변환하여 전달
                    model_config = (
                        vars(self.config)
                        if hasattr(self.config, "__dict__")
                        else self.config
                    )
                    model = GNNModel(config=model_config)
                    model_type = "GNN"
                else:
                    logger.warning(f"알 수 없는 모델 타입: {file_name}")
                    continue

                # 모델 로드
                success = model.load(str(model_file))

                if success:
                    # 모델 등록
                    model_id = f"{model_type}_{file_name}"
                    self.register_model(model_id, model)
                    loaded_count += 1

                    logger.info(f"모델 '{model_id}'를 {model_file}에서 로드했습니다.")
            except Exception as e:
                logger.error(f"모델 로드 중 오류 발생: {str(e)}")

        logger.info(f"총 {loaded_count}개 모델이 로드되었습니다.")
        return loaded_count

    def predict(
        self,
        count: int = 5,
        data: Optional[List[Union[Dict[str, Any], LotteryNumber]]] = None,
    ) -> List[ModelPrediction]:
        """
        앙상블 예측 수행

        Args:
            count: 생성할 추천 세트 수
            data: 예측에 사용할 데이터 (선택사항)

        Returns:
            추천 목록
        """
        if not self.models:
            logger.error("등록된 모델이 없습니다.")
            return []

        # 캐시 키 생성
        cache_key = f"count_{count}"

        # 캐시된 결과가 있으면 반환
        if cache_key in self.recommendation_cache:
            logger.info(f"캐시된 추천 결과를 반환합니다: {cache_key}")
            return self.recommendation_cache[cache_key]

        # 각 모델에서 예측 수행
        all_predictions: Dict[str, List[ModelPrediction]] = {}

        for model_id, model in self.models.items():
            try:
                # 모델이 학습되지 않았으면 스킵
                if not model.is_trained:
                    logger.warning(
                        f"모델 '{model_id}'가 학습되지 않았습니다. 예측을 건너뜁니다."
                    )
                    continue

                # 예측 수행
                predictions = model.predict(input_data=data, count=count)
                if isinstance(predictions, ModelPrediction):
                    predictions = [predictions]  # 단일 예측을 리스트로 변환

                # 결과 저장
                all_predictions[model_id] = predictions

                logger.info(f"모델 '{model_id}'에서 {len(predictions)}개 추천 생성")
            except Exception as e:
                logger.error(f"모델 '{model_id}' 예측 중 오류 발생: {str(e)}")

        # 모든 모델의 예측 결과 통합
        ensemble_predictions = self._combine_predictions(all_predictions, count)

        # 결과 캐시
        self.recommendation_cache[cache_key] = ensemble_predictions

        logger.info(f"앙상블 예측 완료: {len(ensemble_predictions)}개 추천")
        return ensemble_predictions

    def _combine_predictions(
        self, all_predictions: Dict[str, List[ModelPrediction]], count: int
    ) -> List[ModelPrediction]:
        """
        여러 모델의 예측 결과를 통합

        Args:
            all_predictions: 각 모델의 예측 결과
            count: 생성할 추천 세트 수

        Returns:
            통합된 추천 목록
        """
        if not all_predictions:
            return []

        # 통합 전략에 따라 처리
        if self.weighting_strategy == "uniform":
            return self._uniform_ensemble(all_predictions, count)
        elif self.weighting_strategy == "weighted":
            return self._weighted_ensemble(all_predictions, count)
        elif self.weighting_strategy == "stacked":
            return self._stacked_ensemble(all_predictions, count)
        else:
            logger.warning(f"지원되지 않는 가중치 전략: {self.weighting_strategy}")
            return self._uniform_ensemble(all_predictions, count)

    def _uniform_ensemble(
        self, all_predictions: Dict[str, List[ModelPrediction]], count: int
    ) -> List[ModelPrediction]:
        """
        균등 가중치 앙상블

        Args:
            all_predictions: 각 모델의 예측 결과
            count: 생성할 추천 세트 수

        Returns:
            통합된 추천 목록
        """
        # 모든 예측 결과 통합
        combined_predictions: List[ModelPrediction] = []

        for model_id, predictions in all_predictions.items():
            combined_predictions.extend(predictions)

        # 중복 제거 (동일한 번호 조합)
        unique_combinations = {}

        for pred in combined_predictions:
            # 번호 조합을 키로 사용
            key = tuple(sorted(pred.numbers))

            if key not in unique_combinations:
                unique_combinations[key] = pred
            else:
                # 동일한 조합이 있으면 신뢰도 평균 계산
                existing_pred = unique_combinations[key]
                new_confidence = (existing_pred.confidence + pred.confidence) / 2

                # 메타데이터 병합
                merged_metadata = (
                    existing_pred.metadata.copy() if existing_pred.metadata else {}
                )

                # 두 모델 표시
                merged_metadata["models"] = merged_metadata.get(
                    "models", [existing_pred.model_type]
                )
                merged_metadata["models"].append(pred.model_type)

                # 신뢰도 업데이트
                unique_combinations[key] = ModelPrediction(
                    numbers=existing_pred.numbers,
                    confidence=new_confidence,
                    model_type="Ensemble",
                    metadata=merged_metadata,
                )

        # 신뢰도 기준 정렬
        sorted_predictions = sorted(
            unique_combinations.values(), key=lambda p: p.confidence, reverse=True
        )

        # 최대 개수만큼 반환
        result = sorted_predictions[:count]

        # 메타데이터에 랭크 추가
        for i, pred in enumerate(result):
            if pred.metadata is None:
                pred.metadata = {}
            pred.metadata["rank"] = i + 1

        return result

    def _weighted_ensemble(
        self, all_predictions: Dict[str, List[ModelPrediction]], count: int
    ) -> List[ModelPrediction]:
        """
        가중치 기반 앙상블

        Args:
            all_predictions: 각 모델의 예측 결과
            count: 생성할 추천 세트 수

        Returns:
            통합된 추천 목록
        """
        # 모든 예측 결과 통합
        combined_predictions: List[ModelPrediction] = []

        for model_id, predictions in all_predictions.items():
            # 가중치 적용
            weight = self.model_weights.get(model_id, 1.0)

            # 각 예측에 가중치 적용
            for pred in predictions:
                # 신뢰도에 가중치 적용
                weighted_pred = ModelPrediction(
                    numbers=pred.numbers,
                    confidence=pred.confidence * weight,
                    model_type=pred.model_type,
                    metadata={
                        "original_confidence": pred.confidence,
                        "weight": weight,
                        "source_model": model_id,
                        **(pred.metadata or {}),
                    },
                )

                combined_predictions.append(weighted_pred)

        # 중복 제거 및 통합
        unique_combinations = {}

        for pred in combined_predictions:
            # 번호 조합을 키로 사용
            key = tuple(sorted(pred.numbers))

            if key not in unique_combinations:
                unique_combinations[key] = pred
            else:
                # 가중치 합산으로 신뢰도 계산
                existing_pred = unique_combinations[key]

                # 원본 값과 가중치 추출
                orig1 = existing_pred.metadata.get(
                    "original_confidence", existing_pred.confidence
                )
                weight1 = existing_pred.metadata.get("weight", 1.0)
                orig2 = pred.metadata.get("original_confidence", pred.confidence)
                weight2 = pred.metadata.get("weight", 1.0)

                # 가중 평균 계산
                new_confidence = (orig1 * weight1 + orig2 * weight2) / (
                    weight1 + weight2
                )

                # 메타데이터 병합
                merged_metadata = (
                    existing_pred.metadata.copy() if existing_pred.metadata else {}
                )

                # 모델 소스 추적
                models = merged_metadata.get(
                    "source_models", [merged_metadata.get("source_model", "unknown")]
                )
                if "source_model" in pred.metadata:
                    models.append(pred.metadata["source_model"])

                merged_metadata["source_models"] = models

                # 신뢰도 업데이트
                unique_combinations[key] = ModelPrediction(
                    numbers=existing_pred.numbers,
                    confidence=new_confidence,
                    model_type="Ensemble",
                    metadata=merged_metadata,
                )

        # 신뢰도 기준 정렬
        sorted_predictions = sorted(
            unique_combinations.values(), key=lambda p: p.confidence, reverse=True
        )

        # 최대 개수만큼 반환
        result = sorted_predictions[:count]

        # 메타데이터에 랭크 추가
        for i, pred in enumerate(result):
            if pred.metadata is None:
                pred.metadata = {}
            pred.metadata["rank"] = i + 1

        return result

    def _stacked_ensemble(
        self, all_predictions: Dict[str, List[ModelPrediction]], count: int
    ) -> List[ModelPrediction]:
        """
        스택 앙상블 (각 모델에서 상위 예측만 선택)

        Args:
            all_predictions: 각 모델의 예측 결과
            count: 생성할 추천 세트 수

        Returns:
            통합된 추천 목록
        """
        # 각 모델에서 상위 예측만 선택
        top_predictions = []

        # 각 모델마다 얼마나 선택할지 계산
        num_models = len(all_predictions)
        if num_models == 0:
            return []

        selections_per_model = max(1, count // num_models)

        # 각 모델에서 상위 예측 선택
        for model_id, predictions in all_predictions.items():
            # 예측 결과가 없으면 스킵
            if not predictions:
                continue

            # 신뢰도 기준 정렬
            sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)

            # 상위 예측 선택
            selected = sorted_preds[:selections_per_model]

            # 메타데이터 추가
            for pred in selected:
                if pred.metadata is None:
                    pred.metadata = {}
                pred.metadata["source_model"] = model_id

            top_predictions.extend(selected)

        # 중복 제거
        unique_combinations = {}

        for pred in top_predictions:
            # 번호 조합을 키로 사용
            key = tuple(sorted(pred.numbers))

            if (
                key not in unique_combinations
                or pred.confidence > unique_combinations[key].confidence
            ):
                unique_combinations[key] = pred

        # 신뢰도 기준 정렬
        sorted_predictions = sorted(
            unique_combinations.values(), key=lambda p: p.confidence, reverse=True
        )

        # 최대 개수만큼 반환
        result = sorted_predictions[:count]

        # 메타데이터에 랭크 추가
        for i, pred in enumerate(result):
            if pred.metadata is None:
                pred.metadata = {}
            pred.metadata["rank"] = i + 1
            pred.metadata["ensemble_method"] = "stacked"

        return result

    def save_ensemble_config(self, path: str) -> bool:
        """
        앙상블 설정 저장

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        try:
            # 경로 생성
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 설정 준비
            ensemble_config = {
                "weighting_strategy": self.weighting_strategy,
                "model_weights": self.model_weights,
                "models": {
                    model_id: {
                        "type": model.__class__.__name__,
                        "is_trained": model.is_trained,
                    }
                    for model_id, model in self.models.items()
                },
                "timestamp": time.time(),
            }

            # JSON으로 저장
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(ensemble_config, f, indent=2)

            logger.info(f"앙상블 설정이 {save_path}에 저장되었습니다.")
            return True

        except Exception as e:
            logger.error(f"앙상블 설정 저장 중 오류 발생: {str(e)}")
            return False

    def load_ensemble_config(self, path: str) -> bool:
        """
        앙상블 설정 로드

        Args:
            path: 로드할 설정 파일 경로

        Returns:
            로드 성공 여부
        """
        try:
            load_path = Path(path)
            if not load_path.exists():
                logger.error(f"설정 파일을 찾을 수 없습니다: {path}")
                return False

            # JSON 설정 로드
            with open(load_path, "r", encoding="utf-8") as f:
                ensemble_config = json.load(f)

            # 설정 적용
            self.weighting_strategy = ensemble_config.get(
                "weighting_strategy", "uniform"
            )
            self.model_weights = ensemble_config.get("model_weights", {})

            logger.info(f"앙상블 설정이 {path}에서 로드되었습니다.")
            return True

        except Exception as e:
            logger.error(f"앙상블 설정 로드 중 오류 발생: {str(e)}")
            return False

    def cleanup(self) -> None:
        """등록된 모델 자원 정리"""
        for model_id, model in self.models.items():
            try:
                model.cleanup()
                logger.debug(f"모델 '{model_id}' 자원 정리 완료")
            except Exception as e:
                logger.error(f"모델 '{model_id}' 자원 정리 중 오류 발생: {str(e)}")

        # 사전 비우기
        self.models.clear()
        self.model_weights.clear()
        self.model_wrappers.clear()
        self.recommendation_cache.clear()

        logger.info("모든 모델 자원 정리 완료")
