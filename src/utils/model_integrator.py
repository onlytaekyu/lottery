"""모델 통합 시스템 (성능 최적화 버전)

핵심 앙상블 기능만 제공하는 간소화된 모델 통합 시스템
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

from .unified_logging import get_logger

logger = get_logger(__name__)


@dataclass
class IntegratorConfig:
    """통합기 설정 (간소화)"""

    batch_size: int = 32
    enable_cache: bool = True
    cache_dir: str = "cache/integrator"
    model_timeout: float = 10.0  # 10초로 단축

    def __post_init__(self):
        self.cache_dir = str(Path(__file__).parent.parent.parent / self.cache_dir)


class ModelWrapper:
    """모델 래퍼 (간소화)"""

    def __init__(self, model, name: str, weight: float = 1.0):
        self.model = model
        self.name = name
        self.weight = weight
        self.is_active = True
        self.error_count = 0
        self.success_count = 0

    def predict(self, inputs: Any) -> Any:
        """예측 수행"""
        try:
            # 모델이 predict 메서드를 가지고 있는지 확인
            if hasattr(self.model, "predict"):
                outputs = self.model.predict(inputs)
            else:
                # 기본 예측 형태 반환
                outputs = [{"numbers": [1, 2, 3, 4, 5], "confidence": 0.5}]

            self.success_count += 1
            return outputs
        except Exception as e:
            self.error_count += 1
            logger.error(f"모델 {self.name} 예측 실패: {e}")
            raise

    def get_success_rate(self) -> float:
        """성공률 반환"""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0


class ModelIntegrator:
    """모델 통합 시스템 (간소화)"""

    def __init__(self, config: Optional[IntegratorConfig] = None):
        self.config = config or IntegratorConfig()
        self.model_wrappers: Dict[str, ModelWrapper] = {}
        self.prediction_cache: Dict[str, List] = {}

        # 캐시 디렉토리 생성
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info("모델 통합기 초기화 완료")

    def register_model(self, model_id: str, model, weight: float = 1.0) -> None:
        """모델 등록"""
        try:
            wrapper = ModelWrapper(model, model_id, weight)
            self.model_wrappers[model_id] = wrapper
            logger.info(f"모델 등록 완료: {model_id} (가중치: {weight})")
        except Exception as e:
            logger.error(f"모델 등록 실패: {model_id}, {e}")

    def unregister_model(self, model_id: str) -> bool:
        """모델 등록 해제"""
        if model_id in self.model_wrappers:
            del self.model_wrappers[model_id]
            logger.info(f"모델 등록 해제: {model_id}")
            return True
        return False

    def predict(self, count: int = 5, data: Optional[List] = None) -> List:
        """앙상블 예측"""
        if not self.model_wrappers:
            logger.warning("등록된 모델이 없습니다")
            return []

        try:
            # 각 모델에서 예측 수행
            all_predictions: Dict[str, List] = {}

            for model_id, wrapper in self.model_wrappers.items():
                if not wrapper.is_active:
                    continue

                try:
                    start_time = time.time()
                    predictions = wrapper.predict({"count": count, "data": data})

                    # 타임아웃 체크
                    if time.time() - start_time > self.config.model_timeout:
                        logger.warning(f"모델 {model_id} 타임아웃")
                        wrapper.is_active = False
                        continue

                    all_predictions[model_id] = predictions

                except Exception as e:
                    logger.error(f"모델 {model_id} 예측 실패: {e}")
                    continue

            # 예측 결과 통합
            return self._combine_predictions(all_predictions, count)

        except Exception as e:
            logger.error(f"앙상블 예측 실패: {e}")
            return []

    def _combine_predictions(
        self, all_predictions: Dict[str, List], count: int
    ) -> List:
        """예측 결과 통합 (가중 평균)"""
        if not all_predictions:
            return []

        try:
            # 번호별 점수 집계
            number_scores: Dict[int, float] = {}
            total_weight = 0.0

            for model_id, predictions in all_predictions.items():
                wrapper = self.model_wrappers[model_id]
                weight = wrapper.weight * wrapper.get_success_rate()
                total_weight += weight

                # 예측 결과 처리 (다양한 형식 지원)
                if isinstance(predictions, list) and predictions:
                    pred = predictions[0]
                    if hasattr(pred, "numbers") and hasattr(pred, "confidence"):
                        for number in pred.numbers:
                            if number not in number_scores:
                                number_scores[number] = 0.0
                            number_scores[number] += pred.confidence * weight
                    elif (
                        isinstance(pred, dict)
                        and "numbers" in pred
                        and "confidence" in pred
                    ):
                        for number in pred["numbers"]:
                            if number not in number_scores:
                                number_scores[number] = 0.0
                            number_scores[number] += pred["confidence"] * weight

            # 정규화
            if total_weight > 0:
                for number in number_scores:
                    number_scores[number] /= total_weight

            # 상위 번호 선택
            sorted_numbers = sorted(
                number_scores.items(), key=lambda x: x[1], reverse=True
            )[:count]

            # 결과 반환
            final_numbers = [num for num, _ in sorted_numbers]
            avg_confidence = (
                sum(score for _, score in sorted_numbers) / len(sorted_numbers)
                if sorted_numbers
                else 0.0
            )

            # 간단한 딕셔너리 형태로 반환
            return [
                {
                    "numbers": final_numbers,
                    "confidence": avg_confidence,
                    "model_name": "ensemble",
                }
            ]

        except Exception as e:
            logger.error(f"예측 결과 통합 실패: {e}")
            return []

    def update_weights(self):
        """모델 가중치 업데이트"""
        try:
            total_success = sum(
                wrapper.get_success_rate() for wrapper in self.model_wrappers.values()
            )

            if total_success == 0:
                return

            # 성공률 기반 가중치 정규화
            for wrapper in self.model_wrappers.values():
                success_rate = wrapper.get_success_rate()
                wrapper.weight = (
                    success_rate / total_success if total_success > 0 else 1.0
                )

            logger.debug("모델 가중치 업데이트 완료")

        except Exception as e:
            logger.error(f"가중치 업데이트 실패: {e}")

    def get_model_stats(self) -> Dict[str, Any]:
        """모델 통계 반환"""
        stats = {}

        for model_id, wrapper in self.model_wrappers.items():
            stats[model_id] = {
                "weight": wrapper.weight,
                "success_rate": wrapper.get_success_rate(),
                "error_count": wrapper.error_count,
                "success_count": wrapper.success_count,
                "is_active": wrapper.is_active,
            }

        return stats

    def clear_cache(self):
        """캐시 정리"""
        self.prediction_cache.clear()
        logger.debug("예측 캐시 정리 완료")

    def cleanup(self):
        """리소스 정리"""
        try:
            self.clear_cache()
            self.model_wrappers.clear()
            logger.info("모델 통합기 정리 완료")
        except Exception as e:
            logger.error(f"정리 중 오류: {e}")
