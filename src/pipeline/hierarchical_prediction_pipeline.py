"""
계층적 예측 파이프라인

3자리 우선 → 6자리 연계 로또 예측 시스템
1단계: 3자리 고확률 후보 생성 (상위 100개)
2단계: 각 3자리 → 나머지 3자리 최적 확장
3단계: 6자리 통합 점수 계산 및 순위화
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import asyncio

from ..utils.unified_logging import get_logger
from ..utils.compute_strategy import ComputeExecutor, TaskType
from ..utils.performance_optimizer import get_performance_optimizer
from ..utils.unified_async_manager import get_async_manager
from ..utils.model_integrator import GPUEnsembleIntegrator
from ..shared.types import LotteryNumber, ModelPrediction
from ..analysis.pattern_analyzer import PatternAnalyzer
from ..analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer
from ..analysis.three_digit_expansion_engine import (
    ThreeDigitExpansionEngine,
    ExpansionCandidate,
)

logger = get_logger(__name__)


@dataclass
class HierarchicalPrediction:
    """계층적 예측 결과 데이터 클래스"""

    three_digit_combo: Tuple[int, int, int]
    six_digit_combo: Tuple[int, int, int, int, int, int]
    stage1_score: float  # 3자리 단계 점수
    stage2_score: float  # 확장 단계 점수
    stage3_score: float  # 통합 단계 점수
    final_score: float  # 최종 종합 점수
    confidence: float  # 신뢰도
    metadata: Dict[str, Any]


class HierarchicalPredictionPipeline:
    """계층적 예측 파이프라인"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        파이프라인 초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # 성능 최적화 시스템 초기화
        self.compute_executor = ComputeExecutor()
        self.performance_optimizer = get_performance_optimizer()
        self.async_manager = get_async_manager()

        # 분석 및 예측 엔진 초기화
        self.pattern_analyzer = PatternAnalyzer(config)
        self.vectorizer = EnhancedPatternVectorizer(config)
        self.expansion_engine = ThreeDigitExpansionEngine(config)

        # 모델 통합기 초기화
        self.model_integrator = GPUEnsembleIntegrator(
            gpu_parallel=True, max_workers=self.config.get("max_workers", 4)
        )

        # 파이프라인 설정
        self.stage1_top_k = self.config.get("stage1_top_k", 100)
        self.stage2_expansions_per_combo = self.config.get(
            "stage2_expansions_per_combo", 20
        )
        self.final_top_k = self.config.get("final_top_k", 50)

        # 가중치 설정
        self.score_weights = {
            "stage1": self.config.get("stage1_weight", 0.4),
            "stage2": self.config.get("stage2_weight", 0.35),
            "stage3": self.config.get("stage3_weight", 0.25),
        }

        self.logger.info("✅ 계층적 예측 파이프라인 초기화 완료")

    async def predict_hierarchical(
        self,
        historical_data: List[LotteryNumber],
        models: Optional[Dict[str, Any]] = None,
    ) -> List[HierarchicalPrediction]:
        """
        계층적 예측 수행

        Args:
            historical_data: 과거 당첨 번호 데이터
            models: 사용할 모델들 (None이면 기본 모델 사용)

        Returns:
            List[HierarchicalPrediction]: 계층적 예측 결과 리스트
        """
        try:
            self.logger.info("🚀 계층적 예측 파이프라인 시작")
            start_time = time.time()

            # 1단계: 3자리 고확률 후보 생성
            stage1_results = await self._stage1_generate_3digit_candidates(
                historical_data, models
            )

            # 2단계: 3자리 → 6자리 확장
            stage2_results = await self._stage2_expand_to_6digit(
                stage1_results, historical_data
            )

            # 3단계: 통합 점수 계산 및 순위화
            stage3_results = await self._stage3_integrate_and_rank(
                stage2_results, historical_data, models
            )

            # 최종 결과 정리
            final_predictions = self._finalize_predictions(stage3_results)

            total_time = time.time() - start_time

            self.logger.info(
                f"✅ 계층적 예측 완료: {len(final_predictions)}개 결과 ({total_time:.2f}초)"
            )

            return final_predictions

        except Exception as e:
            self.logger.error(f"계층적 예측 중 오류: {e}")
            return []

    async def _stage1_generate_3digit_candidates(
        self,
        historical_data: List[LotteryNumber],
        models: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        1단계: 3자리 고확률 후보 생성

        Args:
            historical_data: 과거 당첨 번호 데이터
            models: 사용할 모델들

        Returns:
            List[Dict[str, Any]]: 3자리 후보 리스트
        """
        try:
            self.logger.info("📊 1단계: 3자리 패턴 분석 및 후보 생성")

            # 3자리 패턴 분석 수행
            analysis_future = self.async_manager.run_async_task(
                self.pattern_analyzer.analyze_3digit_patterns, historical_data
            )

            three_digit_analysis = await analysis_future

            if "error" in three_digit_analysis:
                self.logger.error(
                    f"3자리 패턴 분석 실패: {three_digit_analysis['error']}"
                )
                return []

            # 상위 후보 추출
            top_candidates = three_digit_analysis.get("top_candidates", [])[
                : self.stage1_top_k
            ]

            # ML 모델을 사용한 추가 검증 (모델이 있는 경우)
            if models and "3digit_models" in models:
                enhanced_candidates = await self._enhance_candidates_with_ml(
                    top_candidates, historical_data, models["3digit_models"]
                )
                top_candidates = enhanced_candidates

            self.logger.info(f"1단계 완료: {len(top_candidates)}개 3자리 후보 생성")
            return top_candidates

        except Exception as e:
            self.logger.error(f"1단계 처리 중 오류: {e}")
            return []

    async def _stage2_expand_to_6digit(
        self,
        stage1_candidates: List[Dict[str, Any]],
        historical_data: List[LotteryNumber],
    ) -> List[ExpansionCandidate]:
        """
        2단계: 3자리 → 6자리 확장

        Args:
            stage1_candidates: 1단계 3자리 후보들
            historical_data: 과거 당첨 번호 데이터

        Returns:
            List[ExpansionCandidate]: 확장된 6자리 후보들
        """
        try:
            self.logger.info("🔄 2단계: 3자리 → 6자리 확장")

            # 병렬 확장 처리
            expansion_future = self.async_manager.run_async_task(
                self.expansion_engine.expand_top_candidates,
                stage1_candidates,
                historical_data,
                self.stage2_expansions_per_combo * len(stage1_candidates),
            )

            expanded_candidates = await expansion_future

            self.logger.info(
                f"2단계 완료: {len(expanded_candidates)}개 6자리 후보 생성"
            )
            return expanded_candidates

        except Exception as e:
            self.logger.error(f"2단계 처리 중 오류: {e}")
            return []

    async def _stage3_integrate_and_rank(
        self,
        stage2_candidates: List[ExpansionCandidate],
        historical_data: List[LotteryNumber],
        models: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        3단계: 통합 점수 계산 및 순위화

        Args:
            stage2_candidates: 2단계 확장 후보들
            historical_data: 과거 당첨 번호 데이터
            models: 사용할 모델들

        Returns:
            List[Dict[str, Any]]: 통합 점수가 계산된 후보들
        """
        try:
            self.logger.info("🎯 3단계: 통합 점수 계산 및 순위화")

            scored_candidates = []

            # 각 후보에 대해 통합 점수 계산
            for candidate in stage2_candidates:
                # 기본 패턴 분석 점수
                pattern_score = await self._calculate_pattern_score(
                    candidate.six_digit_combo, historical_data
                )

                # ML 모델 앙상블 점수 (모델이 있는 경우)
                ml_score = 0.0
                if models and "ensemble_models" in models:
                    ml_score = await self._calculate_ml_ensemble_score(
                        candidate.six_digit_combo,
                        historical_data,
                        models["ensemble_models"],
                    )

                # 통합 점수 계산
                integrated_score = self._calculate_integrated_score(
                    candidate, pattern_score, ml_score
                )

                scored_candidate = {
                    "candidate": candidate,
                    "pattern_score": pattern_score,
                    "ml_score": ml_score,
                    "integrated_score": integrated_score,
                }

                scored_candidates.append(scored_candidate)

            # 통합 점수 기준 정렬
            scored_candidates.sort(key=lambda x: x["integrated_score"], reverse=True)

            self.logger.info(f"3단계 완료: {len(scored_candidates)}개 후보 점수 계산")
            return scored_candidates[: self.final_top_k]

        except Exception as e:
            self.logger.error(f"3단계 처리 중 오류: {e}")
            return []

    async def _enhance_candidates_with_ml(
        self,
        candidates: List[Dict[str, Any]],
        historical_data: List[LotteryNumber],
        ml_models: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """ML 모델을 사용한 후보 강화"""
        try:
            enhanced_candidates = []

            for candidate in candidates:
                # 3자리 조합 특성 벡터화
                combo_features = self._vectorize_3digit_combo(
                    candidate["combination"], historical_data
                )

                # ML 모델 예측
                ml_scores = []
                for model_name, model in ml_models.items():
                    if hasattr(model, "predict_3digit_combinations"):
                        try:
                            predictions = model.predict_3digit_combinations(
                                combo_features.reshape(1, -1), top_k=1
                            )
                            if predictions:
                                ml_scores.append(predictions[0][1])
                        except Exception as e:
                            self.logger.warning(f"모델 {model_name} 예측 실패: {e}")

                # ML 점수 통합
                if ml_scores:
                    candidate["ml_enhanced_score"] = (
                        candidate.get("composite_score", 0) * 0.7
                        + np.mean(ml_scores) * 0.3
                    )
                else:
                    candidate["ml_enhanced_score"] = candidate.get("composite_score", 0)

                enhanced_candidates.append(candidate)

            # ML 강화 점수 기준 재정렬
            enhanced_candidates.sort(key=lambda x: x["ml_enhanced_score"], reverse=True)

            return enhanced_candidates

        except Exception as e:
            self.logger.error(f"ML 후보 강화 중 오류: {e}")
            return candidates

    async def _calculate_pattern_score(
        self,
        six_digit_combo: Tuple[int, int, int, int, int, int],
        historical_data: List[LotteryNumber],
    ) -> float:
        """패턴 분석 점수 계산"""
        try:
            # 패턴 특성 추출
            pattern_features = self.pattern_analyzer.extract_pattern_features(
                list(six_digit_combo), historical_data
            )

            # 위험 점수 계산
            risk_score = self.pattern_analyzer.calculate_risk_score(
                list(six_digit_combo), historical_data
            )

            # 패턴 점수 계산 (위험 점수가 낮을수록 높은 점수)
            pattern_score = (1 - risk_score) * pattern_features.get(
                "balance_score", 0.5
            )

            return pattern_score

        except Exception as e:
            self.logger.error(f"패턴 점수 계산 중 오류: {e}")
            return 0.0

    async def _calculate_ml_ensemble_score(
        self,
        six_digit_combo: Tuple[int, int, int, int, int, int],
        historical_data: List[LotteryNumber],
        ensemble_models: Dict[str, Any],
    ) -> float:
        """ML 앙상블 점수 계산"""
        try:
            # 6자리 조합 특성 벡터화
            combo_features = self._vectorize_6digit_combo(
                six_digit_combo, historical_data
            )

            # 앙상블 예측 수행
            ensemble_result = await self.model_integrator.predict(combo_features)

            # 앙상블 점수 추출
            ml_score = ensemble_result.get("avg_confidence", 0.0)

            return ml_score

        except Exception as e:
            self.logger.error(f"ML 앙상블 점수 계산 중 오류: {e}")
            return 0.0

    def _calculate_integrated_score(
        self, candidate: ExpansionCandidate, pattern_score: float, ml_score: float
    ) -> float:
        """통합 점수 계산"""
        try:
            # 각 단계별 점수
            stage1_score = candidate.additional_info.get("base_3digit_score", 0.0)
            stage2_score = candidate.confidence_score
            stage3_score = (pattern_score + ml_score) / 2

            # 가중 평균으로 통합 점수 계산
            integrated_score = (
                stage1_score * self.score_weights["stage1"]
                + stage2_score * self.score_weights["stage2"]
                + stage3_score * self.score_weights["stage3"]
            )

            return integrated_score

        except Exception as e:
            self.logger.error(f"통합 점수 계산 중 오류: {e}")
            return 0.0

    def _vectorize_3digit_combo(
        self, combo: Tuple[int, int, int], historical_data: List[LotteryNumber]
    ) -> np.ndarray:
        """3자리 조합 벡터화"""
        try:
            # 기본 특성 추출
            features = []

            # 조합 기본 특성
            features.extend(
                [
                    sum(combo),  # 합
                    max(combo) - min(combo),  # 범위
                    np.mean(combo),  # 평균
                    np.std(combo),  # 표준편차
                    sum(1 for n in combo if n % 2 == 1),  # 홀수 개수
                ]
            )

            # 과거 데이터와의 관계
            if historical_data:
                recent_data = historical_data[-50:]  # 최근 50회

                # 공출현 빈도
                co_occurrence = 0
                for draw in recent_data:
                    if set(combo).issubset(set(draw.numbers)):
                        co_occurrence += 1

                features.append(co_occurrence / len(recent_data))
            else:
                features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"3자리 조합 벡터화 중 오류: {e}")
            return np.zeros(6, dtype=np.float32)

    def _vectorize_6digit_combo(
        self,
        combo: Tuple[int, int, int, int, int, int],
        historical_data: List[LotteryNumber],
    ) -> np.ndarray:
        """6자리 조합 벡터화"""
        try:
            # 패턴 분석기를 사용한 특성 추출
            pattern_features = self.pattern_analyzer.extract_pattern_features(
                list(combo), historical_data
            )

            # 벡터화
            vectorized = self.pattern_analyzer.vectorize_pattern_features(
                pattern_features
            )

            return vectorized

        except Exception as e:
            self.logger.error(f"6자리 조합 벡터화 중 오류: {e}")
            return np.zeros(50, dtype=np.float32)

    def _finalize_predictions(
        self, stage3_results: List[Dict[str, Any]]
    ) -> List[HierarchicalPrediction]:
        """최종 예측 결과 정리"""
        try:
            final_predictions = []

            for result in stage3_results:
                candidate = result["candidate"]

                prediction = HierarchicalPrediction(
                    three_digit_combo=candidate.three_digit_combo,
                    six_digit_combo=candidate.six_digit_combo,
                    stage1_score=candidate.additional_info.get(
                        "base_3digit_score", 0.0
                    ),
                    stage2_score=candidate.confidence_score,
                    stage3_score=(result["pattern_score"] + result["ml_score"]) / 2,
                    final_score=result["integrated_score"],
                    confidence=result["integrated_score"],
                    metadata={
                        "expansion_method": candidate.expansion_method,
                        "pattern_score": result["pattern_score"],
                        "ml_score": result["ml_score"],
                        "additional_info": candidate.additional_info,
                    },
                )

                final_predictions.append(prediction)

            return final_predictions

        except Exception as e:
            self.logger.error(f"최종 예측 결과 정리 중 오류: {e}")
            return []

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """파이프라인 통계 정보 반환"""
        return {
            "stage1_top_k": self.stage1_top_k,
            "stage2_expansions_per_combo": self.stage2_expansions_per_combo,
            "final_top_k": self.final_top_k,
            "score_weights": self.score_weights,
            "expansion_statistics": self.expansion_engine.get_expansion_statistics(),
        }

    def shutdown(self):
        """리소스 정리"""
        try:
            self.expansion_engine.shutdown()
            self.model_integrator.shutdown()
            self.async_manager.shutdown()
            self.logger.info("계층적 예측 파이프라인 종료 완료")
        except Exception as e:
            self.logger.error(f"파이프라인 종료 중 오류: {e}")


# 편의 함수
async def run_hierarchical_prediction(
    historical_data: List[LotteryNumber],
    config: Optional[Dict[str, Any]] = None,
    models: Optional[Dict[str, Any]] = None,
) -> List[HierarchicalPrediction]:
    """
    계층적 예측 실행 편의 함수

    Args:
        historical_data: 과거 당첨 번호 데이터
        config: 파이프라인 설정
        models: 사용할 모델들

    Returns:
        List[HierarchicalPrediction]: 예측 결과
    """
    pipeline = HierarchicalPredictionPipeline(config)

    try:
        results = await pipeline.predict_hierarchical(historical_data, models)
        return results
    finally:
        pipeline.shutdown()
