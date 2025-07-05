"""
고성능 앙상블 시스템 (v3 - GPU 병렬 최적화)

GPU 병렬 추론, 모델 캐싱, 동적 가중치 조정을 통해
앙상블 성능과 메모리 효율성을 극대화한 시스템
"""

import time
import torch
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import asyncio

from .unified_logging import get_logger
from .cache_manager import get_cache_manager
from .cuda_optimizers import get_cuda_optimizer
from .error_handler import get_error_handler

logger = get_logger(__name__)
error_handler = get_error_handler()


@dataclass
class ModelWrapper:
    """모델 래퍼 (성능 추적 기능 강화)"""

    model: Any
    name: str
    weight: float = 1.0
    is_active: bool = True
    error_count: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0

    def predict(self, inputs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            if hasattr(self.model, "predict"):
                outputs = self.model.predict(inputs)
            else:  # torch.nn.Module
                with torch.no_grad():
                    outputs = self.model(inputs)

            latency = (time.perf_counter() - start_time) * 1000
            self.success_count += 1
            self.total_latency_ms += latency
            return outputs
        except Exception as e:
            self.error_count += 1
            logger.error(f"모델 {self.name} 예측 실패: {e}")
            raise

    def get_performance_score(self) -> float:
        """성공률과 지연시간을 종합한 성능 점수"""
        total_preds = self.success_count + self.error_count
        if total_preds == 0:
            return 0.7  # 초기 가중치

        success_rate = self.success_count / total_preds
        avg_latency = (
            self.total_latency_ms / self.success_count
            if self.success_count > 0
            else float("inf")
        )

        # 지연시간이 길수록 페널티 (100ms 기준)
        latency_penalty = max(0, 1 - (avg_latency / 1000))

        return success_rate * latency_penalty


class GPUEnsembleIntegrator:
    """고성능 앙상블 통합 시스템 (GPU 병렬 추론)"""

    def __init__(self, gpu_parallel: bool = True, max_workers: int = 0):

        self.model_wrappers: Dict[str, ModelWrapper] = {}
        self.gpu_parallel = gpu_parallel and torch.cuda.is_available()

        if self.gpu_parallel:
            self.max_workers = (
                max_workers if max_workers > 0 else min(8, (os.cpu_count() or 1) * 2)
            )
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="EnsembleWorker"
            )
            self.cuda_optimizer = get_cuda_optimizer()
            logger.info(
                f"✅ 고성능 앙상블 초기화 (GPU 병렬 모드, 워커: {self.max_workers})"
            )
        else:
            logger.info("고성능 앙상블 초기화 (순차 모드)")

        self.cache_manager = get_cache_manager()

    def register_model(
        self,
        model_id: str,
        model: Any,
        weight: float = 1.0,
        input_example: Optional[torch.Tensor] = None,
    ):
        """모델 등록 (TensorRT 최적화 옵션 포함)"""
        if (
            self.gpu_parallel
            and isinstance(model, torch.nn.Module)
            and input_example is not None
        ):
            optimized_model = self.cuda_optimizer.tensorrt_optimize(
                model, [input_example], model_name=model_id
            )
            wrapper = ModelWrapper(optimized_model, model_id, weight)
        else:
            wrapper = ModelWrapper(model, model_id, weight)

        self.model_wrappers[model_id] = wrapper
        logger.info(f"모델 등록 완료: {model_id} (가중치: {weight})")

    @error_handler.async_auto_recoverable(max_retries=2, delay=0.1)
    async def parallel_inference(self, data: Any) -> Dict[str, Any]:
        """GPU 병렬 추론"""
        if not self.gpu_parallel:
            return {
                model_id: wrapper.predict(data)
                for model_id, wrapper in self.model_wrappers.items()
                if wrapper.is_active
            }

        loop = asyncio.get_running_loop()
        futures = {}

        for model_id, wrapper in self.model_wrappers.items():
            if not wrapper.is_active:
                continue
            # 각 모델 예측을 별도의 스레드에서 실행
            future = loop.run_in_executor(self.executor, wrapper.predict, data)
            futures[future] = model_id

        results = {}
        for future in asyncio.as_completed(futures):
            model_id = futures[future]
            try:
                result = await future
                results[model_id] = result
            except Exception as e:
                logger.error(f"병렬 추론 중 모델 {model_id} 오류: {e}")
                self.model_wrappers[model_id].is_active = False  # 오류 발생 시 비활성화

        return results

    def _combine_predictions(self, all_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """동적 가중치 기반 예측 결과 통합"""
        number_scores: Dict[int, float] = {}
        total_weight = 0.0

        for model_id, predictions in all_predictions.items():
            wrapper = self.model_wrappers[model_id]
            perf_score = wrapper.get_performance_score()
            weight = wrapper.weight * perf_score
            total_weight += weight

            # 예측 결과 처리 (PyTorch Tensor 또는 Dict 형태 지원)
            if isinstance(predictions, torch.Tensor):
                # 예시: (Top-K Indices, Confidences)
                confidences, indices = torch.topk(
                    torch.softmax(predictions, dim=-1), k=6
                )
                for conf, num_idx in zip(confidences.flatten(), indices.flatten()):
                    number = num_idx.item()
                    number_scores[number] = (
                        number_scores.get(number, 0.0) + conf.item() * weight
                    )
            elif isinstance(predictions, list) and predictions:
                pred = predictions[0]  # 가장 확률 높은 예측 1개 사용
                if "numbers" in pred and "confidence" in pred:
                    for number in pred["numbers"]:
                        number_scores[number] = (
                            number_scores.get(number, 0.0) + pred["confidence"] * weight
                        )

        # 정규화 및 최종 선택
        if total_weight > 0:
            for num in number_scores:
                number_scores[num] /= total_weight

        sorted_numbers = sorted(
            number_scores.items(), key=lambda x: x[1], reverse=True
        )[:6]

        final_numbers = [num for num, _ in sorted_numbers]
        avg_confidence = (
            sum(score for _, score in sorted_numbers) / len(sorted_numbers)
            if sorted_numbers
            else 0.0
        )

        return {
            "numbers": final_numbers,
            "confidence": avg_confidence,
            "model": "high_perf_ensemble",
        }

    @error_handler.async_auto_recoverable(max_retries=2, delay=0.1)
    async def predict(self, data: Any) -> Dict[str, Any]:
        """고성능 앙상블 예측"""
        # 캐시 키 생성 (입력 데이터 기반)
        cache_key = f"ensemble_pred_{hash(str(data))}"

        # 동기 캐시 호출 (await 제거)
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        all_predictions = await self.parallel_inference(data)
        if not all_predictions:
            return {}

        final_prediction = self._combine_predictions(all_predictions)

        # 동기 캐시 호출 (await 제거)
        self.cache_manager.set(
            cache_key, final_prediction, use_disk=True
        )  # 잦은 예측은 디스크 사용
        return final_prediction

    def shutdown(self):
        if self.gpu_parallel:
            self.executor.shutdown(wait=True)
        logger.info("고성능 앙상블 시스템 종료")


# 하위 호환성을 위한 ModelIntegrator
ModelIntegrator = GPUEnsembleIntegrator
