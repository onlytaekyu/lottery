#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DAEBAK_AI 로또 추천 엔진 (Lottery Recommendation Engine)

다양한 모델과 추천 전략을 통합하여 최종 로또 번호를 추천하는 엔진입니다.
"""

from __future__ import annotations

import logging
import random
import numpy as np
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union,
    Tuple,
    cast,
    Type,
    TypeVar,
    Callable,
    Set,
    TYPE_CHECKING,
    Iterable,
)
from pathlib import Path
import json
import time
import importlib
import sys
import os
import csv
from datetime import datetime
from collections import Counter
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.spatial.distance import pdist, squareform
import platform
import psutil

from ..shared.types import LotteryNumber, ModelPrediction, PatternAnalysis
from ..utils.unified_logging import get_logger
from ..utils.data_loader import load_draw_history
from ..utils.pattern_filter import get_pattern_filter, PatternFilter
from ..models.rl_model import RLModel
from ..models.statistical_model import StatisticalModel
from ..models.lstm_model import LSTMModel
from ..models.base_model import BaseModel
from ..utils.unified_config import ConfigProxy
from ..analysis.pattern_analyzer import PatternAnalyzer
from ..utils.unified_report import save_performance_report

# 로거 설정
logger = get_logger(__name__)

# 추천 결과 표현을 위한 타입 정의
RecommendationDict = Dict[str, Any]
RecommendationResult = Union[ModelPrediction, RecommendationDict]
RecommendationList = List[RecommendationResult]


class RecommendationEngine:
    """로또 번호 추천 엔진"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        추천 엔진 초기화

        Args:
            config: 설정 객체
        """
        # 기본 설정
        default_config = {
            "model_weights": {
                "rl": 0.20,  # 강화학습 모델 가중치
                "statistical": 0.20,  # 통계 기반 모델 가중치
                "pattern": 0.15,  # 패턴 기반 모델 가중치
                "lstm": 0.10,  # LSTM 모델 가중치
                "gnn": 0.25,  # GNN 모델 가중치
                "autoencoder": 0.10,  # 오토인코더 모델 가중치
            },
            "ensemble_size": 5,  # 앙상블 크기 (생성할 추천 수)
            "filter_failed_patterns": True,  # 실패 패턴 필터링 여부
            "model_paths": {
                "rl": "savedModels/rl_model.pt",
                "statistical": "savedModels/statistical_model.pt",
                "lstm": "savedModels/lstm_model.pt",
                "gnn": "savedModels/gnn_model.pt",
                "autoencoder": "savedModels/autoencoder_model.pt",
            },
            "diversity_weight": 0.3,  # 다양성 가중치
            "confidence_threshold": 0.2,  # 최소 신뢰도 임계값
            "analysis_cache": {
                "enabled": True,
                "path": "data/cache/",
            },
            "scope_weights": {
                "full": 0.6,  # 전체 데이터 가중치
                "mid": 0.25,  # 중기(최근 100회) 가중치
                "short": 0.15,  # 단기(최근 20회) 가중치
            },
            # 다양성 관련 기본 설정 추가
            "recommendation": {
                "enable_jaccard_filter": True,
                "jaccard_threshold": 0.5,
                "use_adjusted_score": True,
                "clustering_diversity": True,
                "candidate_cluster_method": "agglomerative",  # "dbscan" 또는 "agglomerative"
            },
        }

        # 설정 파일에서 설정 로드
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
            if config_path.exists():
                import yaml

                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)

                if config_data and "recommendation" in config_data:
                    rec_config = config_data["recommendation"]
                    # 모델 사용 플래그
                    if "use_models" in rec_config:
                        # 사용 중지된 모델의 가중치를 0으로 설정
                        for model, enabled in rec_config["use_models"].items():
                            if not enabled and model in default_config["model_weights"]:
                                default_config["model_weights"][model] = 0.0

                    # 모델 가중치
                    if "model_weights" in rec_config:
                        weights = rec_config["model_weights"]
                        # 가중치 정규화
                        weight_sum = sum(weights.values())
                        if weight_sum > 0:
                            default_config["model_weights"] = {
                                k: v / weight_sum for k, v in weights.items()
                            }
        except Exception as e:
            logger.warning(f"설정 파일 로드 중 오류: {str(e)}")

        # 설정 병합
        self.config = default_config.copy()
        if config:
            self.config.update(config)

        # 모델 초기화
        self.models: Dict[str, BaseModel] = {}
        self.pattern_filter = get_pattern_filter()
        self.pattern_analyzer = PatternAnalyzer()
        self.pattern_analyses: Dict[str, PatternAnalysis] = {}

        # 모델 로드
        self._load_models()

        logger.info("추천 엔진 초기화 완료")

    def _load_models(self) -> None:
        """모델 로드"""
        try:
            # 강화학습 모델 로드
            if (
                "rl" in self.config["model_weights"]
                and self.config["model_weights"]["rl"] > 0
            ):
                rl_model = RLModel()
                rl_path = self.config["model_paths"].get(
                    "rl", "savedModels/rl_model.pt"
                )
                if rl_model.load(rl_path):
                    self.models["rl"] = rl_model
                    logger.info(f"RL 모델 로드 성공: {rl_path}")
                else:
                    logger.warning(f"RL 모델 로드 실패: {rl_path}")

            # 통계 기반 모델 로드
            if (
                "statistical" in self.config["model_weights"]
                and self.config["model_weights"]["statistical"] > 0
            ):
                statistical_model = StatisticalModel()
                stat_path = self.config["model_paths"].get(
                    "statistical", "savedModels/statistical_model.pt"
                )
                if statistical_model.load(stat_path):
                    self.models["statistical"] = statistical_model
                    logger.info(f"통계 모델 로드 성공: {stat_path}")
                else:
                    logger.warning(f"통계 모델 로드 실패: {stat_path}")

            # LSTM 모델 로드
            if (
                "lstm" in self.config["model_weights"]
                and self.config["model_weights"]["lstm"] > 0
            ):
                lstm_model = LSTMModel()
                lstm_path = self.config["model_paths"].get(
                    "lstm", "savedModels/lstm_model.pt"
                )
                if lstm_model.load(lstm_path):
                    self.models["lstm"] = lstm_model
                    logger.info(f"LSTM 모델 로드 성공: {lstm_path}")
                else:
                    logger.warning(f"LSTM 모델 로드 실패: {lstm_path}")

            # 모델이 없으면 경고
            if not self.models:
                logger.warning("로드된 모델이 없습니다. 기본 추천 방식을 사용합니다.")

        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")

    def _load_default_data(self) -> List[LotteryNumber]:
        """기본 데이터 로드"""
        try:
            data_path = self.config.get("data_path")
            if not data_path:
                # 상대 경로로 기본 데이터 파일 지정
                data_path = (
                    Path(__file__).parent.parent.parent / "data" / "lottery_numbers.csv"
                )

            loaded_data = load_draw_history(file_path=str(data_path))
            # 타입 변환 처리를 통해 타입 호환성 보장
            result = []
            for item in loaded_data:
                if hasattr(item, "numbers") and hasattr(item, "draw_no"):
                    result.append(
                        LotteryNumber(
                            draw_no=item.draw_no,
                            numbers=item.numbers,
                            date=getattr(item, "date", None),
                        )
                    )
            return result
        except Exception as e:
            logger.error(f"기본 데이터 로드 실패: {str(e)}")
            # 오류 시 빈 리스트 반환
            return []

    def _generate_random_recommendations(self, count: int) -> List[ModelPrediction]:
        """
        무작위 추천 생성

        Args:
            count: 추천 수

        Returns:
            무작위 추천 목록
        """
        recommendations = []
        seen = set()

        for _ in range(count):
            # 중복되지 않는 번호 생성
            while True:
                numbers = sorted(random.sample(range(1, 46), 6))
                numbers_tuple = tuple(numbers)
                if numbers_tuple not in seen:
                    seen.add(numbers_tuple)
                    break

            # 낮은 신뢰도의 무작위 추천
            confidence = random.uniform(0.1, 0.3)

            # ModelPrediction 객체 생성 (딕셔너리 대신)
            recommendations.append(
                ModelPrediction(
                    numbers=numbers, confidence=confidence, model_type="random"
                )
            )

        return recommendations

    def _log_recommendations(
        self, recommendations: List[ModelPrediction], strategy: str
    ) -> None:
        """로그에 추천 정보 기록"""
        logger.info(f"추천 전략: {strategy}")
        for i, rec in enumerate(recommendations):
            logger.info(
                f"추천 {i+1}: {rec.numbers} (신뢰도: {rec.confidence:.4f}, 모델: {rec.model_type})"
            )

    def _apply_scoring(
        self, recommendations: List[ModelPrediction], data: List[LotteryNumber]
    ) -> List[ModelPrediction]:
        """
        연속 번호 점수만 적용하여 신뢰도를 조정합니다.
        메타데이터를 수정하지 않습니다.
        """
        try:
            analyzer = PatternAnalyzer()

            # 연속 번호 분포 분석 - 반환값을 명시적으로 활용
            pattern_analysis = analyzer.analyze_consecutive_length_distribution(data)

            # 패턴 분석 결과 로그 출력
            consecutive_dist = pattern_analysis.metadata.get(
                "consecutive_length_distribution", {}
            )
            if consecutive_dist:
                logger.info(f"연속 번호 길이 분포: {consecutive_dist}")

            # 새로운 결과 목록
            result = []

            # 각 추천에 대해 처리
            for rec in recommendations:
                # 연속 번호 점수 계산
                consecutive_score = analyzer.score_by_consecutive_pattern(rec.numbers)
                new_confidence = min(1.0, max(0.1, rec.confidence + consecutive_score))

                # 신뢰도만 조정된 새 객체 생성
                new_rec = ModelPrediction(
                    numbers=rec.numbers,
                    confidence=new_confidence,
                    model_type=rec.model_type,
                    metadata=rec.metadata.copy(),  # 메타데이터 복사
                )

                # 메타데이터에 연속 번호 점수 추가 (선택적)
                new_rec.metadata["consecutive_score"] = consecutive_score
                new_rec.metadata["max_consecutive_length"] = (
                    analyzer.get_max_consecutive_length(rec.numbers)
                )

                result.append(new_rec)

            logger.info("연속 번호 점수 적용 완료")
            return result
        except Exception as e:
            logger.warning(f"연속 번호 점수 적용 중 오류: {str(e)}")
            return recommendations  # 오류 시 원본 반환

    def run_pattern_analysis(
        self, data: List[LotteryNumber]
    ) -> Dict[str, PatternAnalysis]:
        """
        모든 스코프에 대한 패턴 분석을 실행합니다.

        Args:
            data: 로또 당첨 번호 데이터

        Returns:
            Dict[str, PatternAnalysis]: 스코프별 패턴 분석 결과
        """
        logger.info("다단계 패턴 분석 시작")

        # 캐시 설정 확인
        cache_enabled = self.config.get("analysis_cache", {}).get("enabled", False)
        cache_path = self.config.get("analysis_cache", {}).get("path", "data/cache/")

        # 캐시에서 로드 시도
        if cache_enabled:
            logger.info(f"패턴 분석 캐시 로드 시도 (경로: {cache_path})")
            if self.pattern_analyzer.load_analyses_from_cache(cache_path):
                self.pattern_analyses = self.pattern_analyzer.get_all_analyses()
                logger.info("패턴 분석 캐시 로드 완료")
                return self.pattern_analyses

        # 캐시 로드 실패 또는 비활성화된 경우 새로 분석 수행
        logger.info("모든 스코프에 대한 패턴 분석 수행 중...")
        self.pattern_analyses = self.pattern_analyzer.run_all_analyses(data)

        # 캐시 저장
        if cache_enabled:
            logger.info(f"패턴 분석 결과 캐시 저장 중... (경로: {cache_path})")
            self.pattern_analyzer.save_analyses_to_cache(cache_path)

        return self.pattern_analyses

    def recommend(
        self,
        count: int = 5,
        strategy: str = "hybrid",
        data: Optional[List[LotteryNumber]] = None,
        model_types: Optional[List[str]] = None,
    ) -> List[ModelPrediction]:
        """
        로또 번호 추천

        Args:
            count: 추천할 번호 세트 수
            strategy: 추천 전략 ("hybrid", "statistical", "rl", "pattern" 등)
            data: 참조할 로또 당첨 번호 데이터 (None이면 자동 로드)
            model_types: 사용할 모델 유형 리스트

        Returns:
            ModelPrediction 객체 리스트
        """
        start_time = time.time()

        logger.info(f"번호 추천 요청 (전략: {strategy}, 개수: {count})")

        # 데이터 로드
        if data is None:
            data = self._load_default_data()
            if not data:
                logger.error("데이터 로드 실패, 빈 목록 반환")
                return []

        # 모든 스코프에 대한 패턴 분석 실행 (초기 단계에서)
        if not self.pattern_analyses:
            self.pattern_analyses = self.run_pattern_analysis(data)

        try:
            # 전략에 따라 추천 수행
            recommendations = self._get_recommendations_by_strategy(
                strategy, count, data, model_types
            )

            # 연속 번호 점수 적용
            recommendations = self._apply_scoring(recommendations, data)

            # 신뢰도 기준으로 재정렬
            recommendations.sort(key=lambda x: x.confidence, reverse=True)

            # 결과 로깅
            self._log_recommendations(recommendations, strategy)

            # 실행 시간 측정
            duration = time.time() - start_time
            logger.info(f"추천 완료: {duration:.2f}초 소요")

            # 성능 보고서 생성
            self._generate_performance_report(
                recommendations=recommendations[:count],
                strategy=strategy,
                model_types=model_types,
                duration=duration,
            )

            return recommendations[:count]

        except Exception as e:
            logger.error(f"추천 중 오류 발생: {str(e)}")
            import traceback

            logger.debug(traceback.format_exc())

            # 폴백 전략 사용
            recommendations = self._fallback_recommend(
                count, data, strategy, model_types
            )

            # 오류 보고서 생성
            self._generate_performance_report(
                recommendations=recommendations,
                strategy=strategy,
                model_types=model_types,
                duration=time.time() - start_time,
                error=str(e),
            )

            return recommendations

    def _get_recommendations_by_strategy(
        self,
        strategy: str,
        count: int,
        data: List[LotteryNumber],
        model_types: Optional[List[str]] = None,
    ) -> List[ModelPrediction]:
        """
        전략에 따른 추천 수행

        Args:
            strategy: 추천 전략
            count: 추천 개수
            data: 참조 데이터
            model_types: 사용할 모델 유형 목록

        Returns:
            ModelPrediction 객체 리스트
        """
        raw_recommendations: List[RecommendationResult] = []

        # 전략에 따라 추천 수행
        if strategy.lower() == "hybrid":
            raw_recommendations = self._get_model_recommendations(
                "hybrid", count, data, model_types
            )
        elif strategy.lower() == "rl":
            raw_recommendations = self._rl_recommend(count, data)
        elif strategy.lower() == "statistical":
            raw_recommendations = self._statistical_recommend(count, data)
        elif strategy.lower() == "pattern":
            raw_recommendations = self._pattern_recommend(count, data)
        elif strategy.lower() == "neural":
            raw_recommendations = self._neural_recommend(count, data)
        elif strategy.lower() == "lstm":
            raw_recommendations = self._lstm_recommend(count, data)
        else:
            logger.warning(f"알 수 없는 전략: {strategy}, 하이브리드 방식으로 대체")
            raw_recommendations = self._get_model_recommendations(
                "hybrid", count, data, model_types
            )

        # Dict -> ModelPrediction 변환
        return self._convert_to_model_predictions(raw_recommendations)

    def _hybrid_recommend(
        self,
        count: int,
        data: List[LotteryNumber],
        model_types: Optional[List[str]] = None,
    ) -> List[ModelPrediction]:
        """
        하이브리드 추천 전략 (다양한 모델의 조합)

        Args:
            count: 추천할 번호 세트 수
            data: 참조할 로또 당첨 번호 데이터
            model_types: 사용할 모델 유형 리스트

        Returns:
            ModelPrediction 객체 리스트
        """
        raw_recommendations = self._hybrid_recommend_raw(count, data, model_types)
        return self._convert_to_model_predictions(raw_recommendations)

    def _get_model_recommendations(
        self,
        model_type: str,
        count: int,
        data: List[LotteryNumber],
        model_types: Optional[List[str]] = None,
    ) -> List[RecommendationResult]:
        """
        특정 모델 유형의 추천 결과 반환

        Args:
            model_type: 모델 유형
            count: 추천 개수
            data: 참조 데이터
            model_types: 사용할 모델 유형 목록 (하이브리드 전략에만 사용)

        Returns:
            추천 결과 목록 (Dict 또는 ModelPrediction)
        """
        # 하이브리드 모델 처리
        if model_type == "hybrid":
            return self._hybrid_recommend_raw(count, data, model_types)

        # 개별 모델 처리
        if model_type == "rl":
            return self._rl_recommend(count, data)
        elif model_type == "statistical":
            return self._statistical_recommend(count, data)
        elif model_type == "pattern":
            return self._pattern_recommend(count, data)
        elif model_type == "neural":
            return self._neural_recommend(count, data)
        elif model_type == "lstm":
            return self._lstm_recommend(count, data)
        else:
            logger.warning(f"알 수 없는 모델 유형: {model_type}, 빈 목록 반환")
            return []

    def _hybrid_recommend_raw(
        self,
        count: int,
        data: List[LotteryNumber],
        model_types: Optional[List[str]] = None,
    ) -> List[RecommendationResult]:
        """
        하이브리드 추천 내부 로직 - 다양한 모델을 조합하여 추천

        Args:
            count: 추천할 번호 세트 수
            data: 참조할 로또 당첨 번호 데이터
            model_types: 사용할 모델 유형 리스트

        Returns:
            RecommendationResult 객체 리스트 (Dict 또는 ModelPrediction)
        """
        logger.info(f"하이브리드 추천 전략 실행 (요청 수: {count})")

        # 모델 가중치 계산
        weights = self._get_model_weights()

        # 사용할 모델 유형 필터링
        if model_types:
            available_models = {k: v for k, v in weights.items() if k in model_types}
            if available_models:
                # 가중치 재정규화
                total = sum(available_models.values())
                if total > 0:
                    weights = {k: v / total for k, v in available_models.items()}
                else:
                    # 모든 필터된 모델의 가중치가 0인 경우, 동일한 가중치 부여
                    equal_weight = 1.0 / len(available_models)
                    weights = {k: equal_weight for k in available_models}
            else:
                # 일치하는 모델이 없는 경우, 원래 가중치 사용
                logger.warning(
                    "지정된 모델 유형에 맞는 모델이 없습니다. 전체 모델 사용"
                )

        # 각 모델 유형별 개수 계산
        model_counts = self._calculate_model_counts(weights, count, model_types)
        logger.info(f"모델 유형별 추천 수: {model_counts}")

        # 각 모델별로 추천 수행
        raw_recommendations = []
        for model_type, model_count in model_counts.items():
            if model_count <= 0:
                continue

            logger.debug(f"{model_type} 모델로 {model_count}개 추천 시작")

            try:
                # 모델 유형별 단일 추천 모델 사용
                if model_type != "hybrid":
                    model_recs = self._get_model_recommendations(
                        model_type, model_count, data
                    )
                    raw_recommendations.extend(model_recs)
            except Exception as e:
                logger.error(f"{model_type} 모델 추천 중 오류: {str(e)}")
                continue

        # 요청한 개수보다 추천이 적은 경우 무작위 추천으로 보충
        if len(raw_recommendations) < count:
            logger.warning(
                f"추천이 부족합니다. 무작위 추천으로 보충 ({len(raw_recommendations)}/{count})"
            )
            # 여기서는 딕셔너리 형태로 추가
            additional_recs = []
            for _ in range(count - len(raw_recommendations)):
                numbers = sorted(random.sample(range(1, 46), 6))
                additional_recs.append(
                    {
                        "numbers": numbers,
                        "confidence": random.uniform(0.1, 0.3),
                        "source": "random",
                        "model_type": "random",
                    }
                )
            raw_recommendations.extend(additional_recs)

        # 패턴 필터링 적용 (딕셔너리 형태로 작업)
        if self.config.get("filter_failed_patterns", True):
            filtered_recommendations = []

            for rec in raw_recommendations:
                if isinstance(rec, dict):
                    numbers = rec["numbers"]
                elif hasattr(rec, "numbers"):
                    numbers = rec.numbers
                else:
                    continue

                pattern_hash = self.pattern_filter.get_pattern_hash(numbers)

                # 실패 패턴이 아니면 추가
                if not self.pattern_filter.is_failed_pattern(pattern_hash):
                    filtered_recommendations.append(rec)

            logger.info(
                f"패턴 필터링: {len(raw_recommendations)}개 중 {len(filtered_recommendations)}개 통과"
            )
            raw_recommendations = filtered_recommendations

            # 필터링 후 부족한 경우 다시 보충
            if len(raw_recommendations) < count:
                additional_recs = []
                for _ in range(count - len(raw_recommendations)):
                    numbers = sorted(random.sample(range(1, 46), 6))
                    additional_recs.append(
                        {
                            "numbers": numbers,
                            "confidence": random.uniform(0.1, 0.3),
                            "source": "random",
                            "model_type": "random",
                        }
                    )
                raw_recommendations.extend(additional_recs)

        # 다양성 필터링 적용 (2배 수의 후보를 생성한 후 필터링)
        diversity_candidates = raw_recommendations[: count * 2]
        diverse_recommendations = self._apply_diversity_filtering(
            diversity_candidates, count
        )

        logger.info(
            f"다양성 필터링: {len(diversity_candidates)}개 중 {len(diverse_recommendations)}개 선택"
        )

        return diverse_recommendations

    def _calculate_model_counts(
        self,
        weights: Dict[str, float],
        count: int,
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        각 모델별 추천 개수 계산

        Args:
            weights: 모델별 가중치
            count: 총 추천 개수
            model_types: 사용할 모델 유형 목록

        Returns:
            모델별 추천 개수
        """
        model_counts = {}
        remaining = count

        # 가중치에 따라 각 모델 유형별 추천 개수 계산
        for model_type, weight in sorted(
            weights.items(), key=lambda x: x[1], reverse=True
        ):
            if remaining <= 0:
                break

            # 모델 유형 필터링 적용
            if model_types and model_type not in model_types:
                continue

            # 가중치에 비례한 추천 개수 계산 (최소 1개)
            model_count = max(1, int(count * weight))
            if model_count > remaining:
                model_count = remaining

            model_counts[model_type] = model_count
            remaining -= model_count

        # 남은 개수를 첫 번째 모델에 할당
        if remaining > 0 and model_counts:
            first_model = next(iter(model_counts))
            model_counts[first_model] += remaining

        return model_counts

    def _recommend_with_model(
        self, model_name: str, count: int, data: List[LotteryNumber]
    ) -> List[RecommendationResult]:
        """
        특정 모델을 사용한 추천

        Args:
            model_name: 모델 이름
            count: 추천 수
            data: 참조 데이터

        Returns:
            추천 목록
        """
        logger.info(f"{model_name} 모델로 {count}개 추천 시작")

        # 모델이 로드되어 있지 않은 경우 처리
        if model_name not in self.models:
            logger.warning(f"{model_name} 모델이 로드되지 않았습니다.")
            return []

        try:
            model = self.models[model_name]
            model_type = model_name

            # 각 모델에 따른 스코프 선택
            scope_to_use = "full"  # 기본값
            if model_name == "statistical":
                # 통계 모델: 스코프 혼합 사용
                scope_weights = self.config.get(
                    "scope_weights", {"full": 0.6, "mid": 0.25, "short": 0.15}
                )

                # 스코프별 분석 결과가 있는 경우에만 혼합
                if len(self.pattern_analyses) >= 3:
                    # 모델 특성에 분석 결과 혼합을 위한 준비
                    blended_frequency = self.pattern_analyzer.blend_frequency_maps(
                        scope_weights.get("full", 0.6),
                        scope_weights.get("mid", 0.25),
                        scope_weights.get("short", 0.15),
                    )

                    # 스코프별 분석 결과를 모델에 제공
                    # 참고: StatisticalModel에 적용 가능한 메서드 개발 필요
                    logger.info("통계 모델에 혼합된 빈도 맵 적용 가능")

            elif model_name == "rl":
                # RL 모델: 모델별 정책에 따라 다양한 스코프 선택 가능
                # 예시: 정책에 따라 full, mid, short 스코프 선택
                scope_to_use = self.config.get("rl_scope", "full")

                # 스코프별 분석 결과가 있는지 확인
                if scope_to_use in self.pattern_analyses:
                    pattern_analysis = self.pattern_analyses[scope_to_use]
                    logger.info(
                        f"RL 모델을 위한 {scope_to_use} 스코프 분석 결과 준비 완료"
                    )

            elif model_name == "lstm":
                # LSTM 모델: 기본적으로 full 스코프 사용
                scope_to_use = "full"

                # 스코프별 분석 결과가 있는지 확인
                if scope_to_use in self.pattern_analyses:
                    pattern_analysis = self.pattern_analyses[scope_to_use]
                    logger.info(
                        f"LSTM 모델을 위한 {scope_to_use} 스코프 분석 결과 준비 완료"
                    )

            # 모델 예측 실행
            predictions = []
            for i in range(count):
                result = model.predict(data)

                # 딕셔너리 형식으로 변환 (일관성 유지)
                if isinstance(result, ModelPrediction):
                    prediction = {
                        "numbers": result.numbers,
                        "confidence": result.confidence,
                        "model_type": model_type,
                        "source": model_name,
                    }

                    # 메타데이터 추가
                    if hasattr(result, "metadata") and result.metadata:
                        prediction["metadata"] = result.metadata

                    # 패턴 기여도 추가
                    if (
                        hasattr(result, "pattern_contributions")
                        and result.pattern_contributions
                    ):
                        prediction["pattern_contributions"] = (
                            result.pattern_contributions
                        )

                    # ROI 추정치 추가
                    if hasattr(result, "roi_estimate") and result.roi_estimate:
                        prediction["roi_estimate"] = result.roi_estimate

                    predictions.append(prediction)
                else:
                    predictions.append(result)

            return predictions

        except Exception as e:
            logger.error(f"{model_name} 모델 추천 중 오류: {str(e)}")
            return []

    def _rl_recommend(
        self, count: int, data: List[LotteryNumber]
    ) -> List[RecommendationResult]:
        """강화학습 모델 기반 추천"""
        return self._recommend_with_model("rl", count, data)

    def _statistical_recommend(
        self, count: int, data: List[LotteryNumber]
    ) -> List[RecommendationResult]:
        """통계 기반 추천"""
        return self._recommend_with_model("statistical", count, data)

    def _neural_recommend(
        self, count: int, data: List[LotteryNumber]
    ) -> List[RecommendationResult]:
        """신경망 기반 추천"""
        try:
            return self._recommend_with_model("neural", count, data)
        except Exception as e:
            logger.warning(f"신경망 모델 추천 실패: {str(e)}")
            return self._recommend_with_model("statistical", count, data)

    def _lstm_recommend(
        self, count: int, data: List[LotteryNumber]
    ) -> List[RecommendationResult]:
        """LSTM 기반 추천"""
        try:
            return self._recommend_with_model("lstm", count, data)
        except Exception as e:
            logger.warning(f"LSTM 모델 추천 실패: {str(e)}")
            return self._recommend_with_model("statistical", count, data)

    def _pattern_recommend(
        self, count: int, data: List[LotteryNumber]
    ) -> List[RecommendationResult]:
        """패턴 기반 추천"""
        # 스코프 기반 패턴 분석 결과 활용
        try:
            # 단기 스코프와 중기 스코프를 혼합하여 사용
            mid_analysis = self.pattern_analyses.get("mid")
            short_analysis = self.pattern_analyses.get("short")

            # 분석 결과가 없는 경우 _recommend_with_patterns 메서드로 폴백
            if not mid_analysis or not short_analysis:
                logger.warning("패턴 분석 결과가 없습니다. 기본 패턴 추천으로 폴백")
                return self._recommend_with_patterns(count, data)

            # 각 스코프의 핫/콜드 번호 취합
            hot_numbers_mid = mid_analysis.hot_numbers
            hot_numbers_short = short_analysis.hot_numbers

            # 클러스터 정보 취합
            clusters_mid = mid_analysis.clusters
            clusters_short = short_analysis.clusters

            # 최근 당첨 번호 패턴 분석
            recent_draws = data[-10:]  # 최근 10회 당첨 번호

            # 패턴 분석
            even_odd_patterns = []
            low_high_patterns = []

            for draw in recent_draws:
                numbers = draw.numbers

                # 홀짝 패턴
                even_count = sum(1 for n in numbers if n % 2 == 0)
                odd_count = 6 - even_count
                even_odd_patterns.append((even_count, odd_count))

                # 고저 패턴
                low_count = sum(1 for n in numbers if n <= 23)
                high_count = 6 - low_count
                low_high_patterns.append((low_count, high_count))

            # 가장 많이 등장한 패턴 추출
            common_even_odd = Counter(even_odd_patterns).most_common(3)
            common_low_high = Counter(low_high_patterns).most_common(3)

            # 추천 생성
            recommendations = []
            for _ in range(count * 2):  # 필터링을 고려해 2배 생성
                # 패턴 선택
                even_odd = random.choice(common_even_odd)[0]
                low_high = random.choice(common_low_high)[0]

                # 기본 번호 생성
                numbers = self._generate_numbers_with_pattern(even_odd, low_high)

                # 핫 넘버 우선 선택 (30% 확률)
                if random.random() < 0.3:
                    hot_pool = list(hot_numbers_mid.union(hot_numbers_short))
                    if hot_pool:
                        # 1-3개의 핫 넘버 포함
                        hot_count = min(random.randint(1, 3), len(hot_pool))
                        selected_hot = random.sample(hot_pool, hot_count)

                        # 기존 번호에서 일부 교체
                        for hot_num in selected_hot:
                            if len(numbers) > 0 and hot_num not in numbers:
                                remove_idx = random.randint(0, len(numbers) - 1)
                                numbers[remove_idx] = hot_num

                # 클러스터 활용 (20% 확률)
                if random.random() < 0.2:
                    all_clusters = clusters_mid + clusters_short
                    if all_clusters:
                        # 랜덤 클러스터 선택
                        selected_cluster = random.choice(all_clusters)

                        # 클러스터에서 2-3개 번호 선택
                        cluster_count = min(random.randint(2, 3), len(selected_cluster))
                        selected_nums = random.sample(selected_cluster, cluster_count)

                        # 기존 번호에서 일부 교체
                        for cluster_num in selected_nums:
                            if len(numbers) > 0 and cluster_num not in numbers:
                                remove_idx = random.randint(0, len(numbers) - 1)
                                numbers[remove_idx] = cluster_num

                # 중복 제거 및 정렬
                unique_numbers = sorted(set(numbers))

                # 번호가 6개 미만이면 보충
                while len(unique_numbers) < 6:
                    new_num = random.randint(1, 45)
                    if new_num not in unique_numbers:
                        unique_numbers.append(new_num)

                # 6개 초과면 제거
                if len(unique_numbers) > 6:
                    unique_numbers = sorted(random.sample(unique_numbers, 6))

                # 신뢰도 계산: 패턴 품질에 기반한 점수 부여
                confidence = 0.6  # 기본 신뢰도

                # 최근 트렌드 반영 시 신뢰도 증가
                if set(unique_numbers).intersection(hot_numbers_short):
                    confidence += 0.1

                # ModelPrediction 객체 생성
                recommendations.append(
                    ModelPrediction(
                        numbers=sorted(unique_numbers),
                        confidence=min(0.9, confidence),  # 최대 0.9 제한
                        model_type="pattern",
                        metadata={"strategy": "scoped_pattern"},
                    )
                )

            return recommendations

        except Exception as e:
            logger.error(f"스코프 기반 패턴 추천 중 오류: {str(e)}")
            # 오류 발생 시 기존 패턴 추천으로 폴백
            return self._recommend_with_patterns(count, data)

    def _generate_numbers_with_pattern(
        self, even_odd: Tuple[int, int], low_high: Tuple[int, int]
    ) -> List[int]:
        """
        패턴에 맞는 번호 생성

        Args:
            even_odd: (짝수 개수, 홀수 개수) 튜플
            low_high: (낮은 번호 개수, 높은 번호 개수) 튜플

        Returns:
            생성된 번호 목록
        """
        even_count, odd_count = even_odd
        low_count, high_count = low_high

        # 짝수/홀수 풀
        even_pool = [n for n in range(2, 46, 2)]  # 2, 4, 6, ..., 44
        odd_pool = [n for n in range(1, 46, 2)]  # 1, 3, 5, ..., 45

        # 낮은/높은 번호 풀
        low_pool = list(range(1, 24))  # 1-23
        high_pool = list(range(24, 46))  # 24-45

        # 번호 선택
        selected = []

        # 짝수 중 낮은 번호
        even_low = list(set(even_pool) & set(low_pool))
        even_low_count = min(even_count, low_count)
        if even_low_count > 0 and even_low:
            selected.extend(random.sample(even_low, even_low_count))

        # 짝수 중 높은 번호
        even_high = list(set(even_pool) & set(high_pool))
        even_high_count = even_count - even_low_count
        if even_high_count > 0 and even_high:
            selected.extend(random.sample(even_high, even_high_count))

        # 홀수 중 낮은 번호
        odd_low = list(set(odd_pool) & set(low_pool))
        odd_low_count = low_count - even_low_count
        if odd_low_count > 0 and odd_low:
            selected.extend(random.sample(odd_low, odd_low_count))

        # 홀수 중 높은 번호
        odd_high = list(set(odd_pool) & set(high_pool))
        odd_high_count = odd_count - odd_low_count
        if odd_high_count > 0 and odd_high:
            selected.extend(random.sample(odd_high, odd_high_count))

        # 부족한 경우 무작위로 채움
        while len(selected) < 6:
            num = random.randint(1, 45)
            if num not in selected:
                selected.append(num)

        return selected

    def _filter_failed_patterns(
        self, recommendations: List[ModelPrediction]
    ) -> List[ModelPrediction]:
        """
        실패 패턴 필터링

        Args:
            recommendations: 추천 목록 (ModelPrediction 객체 리스트)

        Returns:
            필터링된 추천 목록
        """
        if not self.pattern_filter:
            return recommendations

        filtered = []
        for rec in recommendations:
            numbers = rec.numbers
            pattern_hash = self.pattern_filter.get_pattern_hash(numbers)

            # 실패 패턴이 아니면 추가
            if not self.pattern_filter.is_failed_pattern(pattern_hash):
                filtered.append(rec)

        logger.info(f"패턴 필터링: {len(recommendations)}개 중 {len(filtered)}개 통과")
        return filtered

    def _convert_to_model_predictions(
        self, recommendations: Union[List[RecommendationResult], List[ModelPrediction]]
    ) -> List[ModelPrediction]:
        """
        추천 결과를 ModelPrediction 객체로 변환

        Args:
            recommendations: 추천 목록 (Dict 또는 ModelPrediction)

        Returns:
            ModelPrediction 객체 리스트
        """
        result = []
        for rec in recommendations:
            if isinstance(rec, dict):
                # 메타데이터 추출
                metadata = rec.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}

                # ModelPrediction 객체 생성
                model_pred = ModelPrediction(
                    numbers=rec.get("numbers", []),
                    confidence=rec.get("confidence", 0.5),
                    model_type=rec.get("model_type", rec.get("source", "unknown")),
                    metadata=metadata,
                )
                result.append(model_pred)
            elif isinstance(rec, ModelPrediction):
                # 이미 ModelPrediction 객체인 경우
                result.append(rec)
            else:
                logger.warning(f"변환할 수 없는 추천 유형: {type(rec)}")

        return result

    def update_model_weights(self, weights: Dict[str, float]) -> None:
        """
        모델 가중치 업데이트

        Args:
            weights: 모델 유형별 가중치
        """
        # 가중치 검증
        total = sum(weights.values())
        if total <= 0:
            logger.warning("모든 가중치가 0 이하입니다. 기본 가중치를 사용합니다.")
            return

        # 가중치 정규화
        normalized = {k: v / total for k, v in weights.items()}
        self.config["model_weights"] = normalized
        logger.info(f"모델 가중치 업데이트 완료: {normalized}")

    def get_model_weights(self) -> Dict[str, float]:
        """
        현재 모델 가중치 조회

        Returns:
            모델 유형별 가중치
        """
        return self._get_model_weights()

    def _get_model_weights(self) -> Dict[str, float]:
        """모델 가중치 반환"""
        try:
            # 기본 가중치
            default_weights = {
                "rl": 0.20,
                "statistical": 0.20,
                "pattern": 0.15,
                "lstm": 0.10,
                "gnn": 0.25,
                "autoencoder": 0.10,
            }

            # 설정 파일에서 가중치 로드
            if "model_weights" in self.config:
                return self.config.get("model_weights", default_weights)
            else:
                config_path = (
                    Path(__file__).parent.parent.parent / "config" / "config.yaml"
                )

                if config_path.exists():
                    try:
                        import yaml

                        with open(config_path, "r", encoding="utf-8") as f:
                            config_data = yaml.safe_load(f)

                        if (
                            config_data
                            and "recommendation" in config_data
                            and "model_weights" in config_data["recommendation"]
                        ):
                            weights = config_data["recommendation"]["model_weights"]
                            # 가중치 합이 1인지 확인하고 정규화
                            weight_sum = sum(weights.values())
                            if weight_sum > 0:
                                return {k: v / weight_sum for k, v in weights.items()}
                            else:
                                logger.warning(
                                    "모델 가중치 합이 0입니다. 기본 가중치를 사용합니다."
                                )
                                return default_weights
                        else:
                            return default_weights
                    except Exception as e:
                        logger.error(f"설정 파일 로드 중 오류: {str(e)}")
                        return default_weights
                else:
                    return default_weights
        except (AttributeError, KeyError):
            # 기본 가중치 반환
            return {
                "rl": 0.20,
                "statistical": 0.20,
                "pattern": 0.15,
                "lstm": 0.10,
                "gnn": 0.25,
                "autoencoder": 0.10,
            }

    def _fallback_recommend(
        self,
        count: int,
        data: Optional[List[LotteryNumber]],
        strategy: str,
        model_types: Optional[List[str]] = None,
    ) -> List[ModelPrediction]:
        """
        임시/대체 추천 실행 (오류 시 호출)

        Args:
            count: 추천 수
            data: 로또 데이터
            strategy: 추천 전략
            model_types: 사용할 모델 타입 목록

        Returns:
            추천 결과 목록
        """
        logger.warning(f"전략 '{strategy}'에 대한 임시/대체 추천 실행")

        # 기본 데이터 로드
        if data is None:
            data = self._load_default_data()

        # 무작위 추천
        if not data or len(data) < 10:
            return self._generate_random_recommendations(count)

        # 패턴 기반 추천 시도
        try:
            pattern_predictions = self._pattern_recommend(count, data)
            converted = self._convert_to_model_predictions(pattern_predictions)
            if converted and len(converted) >= count:
                return converted[:count]
        except Exception as e:
            logger.error(f"패턴 기반 추천 중 오류 발생: {str(e)}")

        # 최후의 수단으로 무작위 추천
        return self._generate_random_recommendations(count)

    def save_recommendation(
        self,
        recommendations: List[ModelPrediction],
        round_number: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        추천 결과를 파일로 저장

        Args:
            recommendations: 추천 목록
            round_number: 회차 번호
            config: 설정 객체 (선택적)

        Returns:
            저장된 파일 경로 또는 None (실패 시)
        """
        import csv
        from datetime import datetime
        from pathlib import Path

        # 설정 로드
        cfg = ConfigProxy(config or {})
        if not cfg:
            cfg = self.config

        # 출력 기능이 비활성화된 경우
        try:
            enabled = cfg["recommendation_output"]["enabled"]
        except KeyError:
            # 기본값 사용
            enabled = True
            logger.warning(
                "'recommendation_output.enabled' 설정이 없습니다. 기본값(True)을 사용합니다."
            )

        if not enabled:
            logger.info("추천 결과 저장 기능이 비활성화되어 있습니다.")
            return None

        # 기본 경로 설정
        try:
            base_path = cfg["recommendation_output"]["path"]
        except KeyError:
            # 기본 경로 사용
            base_path = "data/predictions/"
            logger.warning(
                f"'recommendation_output.path' 설정이 없습니다. 기본 경로({base_path})를 사용합니다."
            )

        # 경로 확인 및 생성
        path = Path(base_path)
        path.mkdir(parents=True, exist_ok=True)

        # 파일명 생성: {회차번호}_pre_lottery.csv
        filename = f"{round_number}_pre_lottery.csv"
        file_path = path / filename

        # 현재 타임스탬프
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 파일 존재 여부 확인
        file_exists = file_path.exists()

        try:
            # 파일 열기 (존재하면 추가 모드, 없으면 생성)
            mode = "a" if file_exists else "w"
            with open(file_path, mode, newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # 새 파일의 경우 헤더 작성
                if not file_exists:
                    writer.writerow(
                        ["timestamp", "recommended_numbers", "model_used", "confidence"]
                    )

                # 각 추천 결과 작성
                for rec in recommendations:
                    # 번호를 문자열로 변환하여 저장
                    numbers_str = ",".join(map(str, rec.numbers))
                    writer.writerow(
                        [
                            timestamp,
                            f"[{numbers_str}]",
                            rec.model_type,
                            f"{rec.confidence:.4f}",
                        ]
                    )

            logger.info(f"추천 결과가 성공적으로 저장되었습니다: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"추천 결과 저장 중 오류 발생: {str(e)}")
            return None

    def _recommend_with_patterns(
        self, count: int, data: List[LotteryNumber]
    ) -> List[RecommendationResult]:
        """
        패턴 기반 추천 (스코프 기반 분석 없을 경우 대비 폴백 메서드)

        Args:
            count: 추천 수
            data: 참조 데이터

        Returns:
            추천 목록
        """
        # 최근 당첨 번호 패턴 분석
        recent_draws = data[-10:]  # 최근 10회 당첨 번호

        # 패턴 분석
        even_odd_patterns = []
        low_high_patterns = []

        for draw in recent_draws:
            numbers = draw.numbers

            # 홀짝 패턴
            even_count = sum(1 for n in numbers if n % 2 == 0)
            odd_count = 6 - even_count
            even_odd_patterns.append((even_count, odd_count))

            # 고저 패턴
            low_count = sum(1 for n in numbers if n <= 23)
            high_count = 6 - low_count
            low_high_patterns.append((low_count, high_count))

        # 가장 많이 등장한 패턴 추출
        common_even_odd = Counter(even_odd_patterns).most_common(3)
        common_low_high = Counter(low_high_patterns).most_common(3)

        # 추천 생성
        recommendations = []
        for _ in range(count * 2):  # 필터링을 고려해 2배 생성
            # 패턴 선택
            even_odd = random.choice(common_even_odd)[0]
            low_high = random.choice(common_low_high)[0]

            # 번호 생성
            numbers = self._generate_numbers_with_pattern(even_odd, low_high)

            # ModelPrediction 객체 직접 생성
            recommendations.append(
                ModelPrediction(
                    numbers=sorted(numbers),
                    confidence=0.6,  # 기본 신뢰도
                    model_type="pattern",
                    metadata={"strategy": "pattern"},
                )
            )

        return recommendations

    def jaccard(self, a: List[int], b: List[int]) -> float:
        """
        두 집합 간의 Jaccard 유사도 계산

        Args:
            a: 첫 번째 번호 집합
            b: 두 번째 번호 집합

        Returns:
            두 집합 간의 Jaccard 유사도 (0-1 사이 값)
        """
        set_a = set(a)
        set_b = set(b)
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def _calculate_similarity_matrix(
        self, candidates: List[RecommendationResult]
    ) -> np.ndarray:
        """
        후보들 간의 유사도 행렬 계산

        Args:
            candidates: 추천 결과 목록

        Returns:
            유사도 행렬 (numpy 배열)
        """
        n = len(candidates)
        similarity_matrix = np.zeros((n, n))

        # 각 후보의 번호 리스트를 미리 추출
        candidate_numbers = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                numbers = candidate["numbers"]
            else:
                numbers = candidate.numbers
            candidate_numbers.append(numbers)

        # 유사도 행렬 계산 - 최적화된 방식
        for i in range(n):
            for j in range(i + 1, n):
                # Jaccard 유사도 계산
                similarity = self.jaccard(candidate_numbers[i], candidate_numbers[j])
                # 대칭 행렬이므로 양쪽에 동일하게 설정
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        # 유사도 행렬 로깅 (디버그 모드)
        high_similarity_count = np.sum(similarity_matrix > 0.5)
        logger.debug(
            f"유사도 행렬 계산 완료: {n}x{n} 크기, 높은 유사도(>0.5) 페어: {int(high_similarity_count/2)}개"
        )

        return similarity_matrix

    def _select_diverse_candidates(
        self,
        candidates: List[RecommendationResult],
        count: int,
        similarity_threshold: float = 0.5,
    ) -> List[RecommendationResult]:
        """
        Jaccard 유사도를 기반으로 다양성 있는 후보 선택

        Args:
            candidates: 후보 목록
            count: 선택할 후보 수
            similarity_threshold: 유사도 임계값 (이 값보다 유사도가 낮아야 함)

        Returns:
            선택된 다양한 후보 목록
        """
        if len(candidates) <= count:
            return candidates

        # 후보 복사 및 신뢰도/점수 기준 정렬
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x["confidence"] if isinstance(x, dict) else x.confidence,
            reverse=True,
        )

        selected_indices = []  # 선택된 후보의 인덱스
        selected_candidates = []  # 선택된 후보 목록

        # 첫 번째 후보 선택 (최고 점수)
        selected_indices.append(0)
        selected_candidates.append(sorted_candidates[0])

        # 나머지 후보 순회하며 다양성 기준 만족하는 후보 선택
        for i in range(1, len(sorted_candidates)):
            if len(selected_candidates) >= count:
                break

            current_candidate = sorted_candidates[i]
            current_numbers = (
                current_candidate["numbers"]
                if isinstance(current_candidate, dict)
                else current_candidate.numbers
            )

            # 이미 선택된 후보들과의 유사도 확인
            diverse_enough = True
            similarities = []

            for selected in selected_candidates:
                selected_numbers = (
                    selected["numbers"]
                    if isinstance(selected, dict)
                    else selected.numbers
                )
                similarity = self.jaccard(current_numbers, selected_numbers)
                similarities.append(similarity)

                if similarity >= similarity_threshold:
                    diverse_enough = False
                    break

            # 모든 기존 선택 후보와 충분히 다르다면 선택
            if diverse_enough:
                selected_indices.append(i)
                selected_candidates.append(current_candidate)
                logger.debug(
                    f"후보 {i} 선택: 유사도 = {similarities}, 번호 = {current_numbers}"
                )
            else:
                logger.debug(
                    f"후보 {i} 제외: 높은 유사도 = {similarities}, 번호 = {current_numbers}"
                )

        # 충분한 다양성을 가진 후보를 찾지 못한 경우, 임계값을 점진적으로 완화
        if len(selected_candidates) < count:
            logger.warning(
                f"충분한 다양성을 가진 후보를 찾지 못했습니다. 임계값 완화: {similarity_threshold} -> {similarity_threshold * 1.2}"
            )
            remaining = count - len(selected_candidates)

            # 아직 선택되지 않은 후보들 중에서 선택
            for i in range(len(sorted_candidates)):
                if i in selected_indices or len(selected_candidates) >= count:
                    continue

                current_candidate = sorted_candidates[i]
                current_numbers = (
                    current_candidate["numbers"]
                    if isinstance(current_candidate, dict)
                    else current_candidate.numbers
                )

                # 평균 유사도 계산
                avg_similarity = 0.0
                if selected_candidates:
                    total_similarity = 0.0
                    for selected in selected_candidates:
                        selected_numbers = (
                            selected["numbers"]
                            if isinstance(selected, dict)
                            else selected.numbers
                        )
                        total_similarity += self.jaccard(
                            current_numbers, selected_numbers
                        )
                    avg_similarity = total_similarity / len(selected_candidates)

                # 완화된 기준으로 후보 선택
                if avg_similarity < similarity_threshold * 1.2:
                    selected_indices.append(i)
                    selected_candidates.append(current_candidate)
                    logger.debug(
                        f"완화된 기준으로 후보 {i} 선택: 평균 유사도 = {avg_similarity}, 번호 = {current_numbers}"
                    )

        return selected_candidates

    def _adjust_scores_based_on_similarity(
        self,
        candidates: List[RecommendationResult],
        selected_candidates: List[RecommendationResult],
    ) -> List[RecommendationResult]:
        """
        선택된 후보와의 유사도에 기반하여 점수 조정

        Args:
            candidates: 모든 후보 목록
            selected_candidates: 이미 선택된 후보 목록

        Returns:
            점수가 조정된 후보 목록
        """
        if not selected_candidates:
            return candidates

        adjusted_candidates = []

        for candidate in candidates:
            # 후보 번호 추출
            if isinstance(candidate, dict):
                candidate_numbers = candidate["numbers"]
            else:
                candidate_numbers = candidate.numbers

            # 원래 점수/신뢰도 추출
            if isinstance(candidate, dict):
                original_score = candidate["confidence"]
            else:
                original_score = candidate.confidence

            # 이미 선택된 후보들과의 평균 유사도 계산
            total_similarity = 0.0
            for selected in selected_candidates:
                if isinstance(selected, dict):
                    selected_numbers = selected["numbers"]
                else:
                    selected_numbers = selected.numbers
                total_similarity += self.jaccard(candidate_numbers, selected_numbers)

            avg_similarity = (
                total_similarity / len(selected_candidates)
                if selected_candidates
                else 0.0
            )

            # 조정된 점수 계산: original_score * (1 - avg_similarity)
            adjusted_score = original_score * (1 - avg_similarity)

            # 점수 조정이 큰 경우 로깅 (30% 이상 차이가 나는 경우)
            if avg_similarity > 0.3:
                logger.debug(
                    f"높은 유사도 조정: 번호 {candidate_numbers}, 원점수 {original_score:.4f} → 조정점수 {adjusted_score:.4f} (유사도: {avg_similarity:.4f})"
                )

            # 결과 저장
            if isinstance(candidate, dict):
                adjusted_candidate = candidate.copy()
                adjusted_candidate["original_confidence"] = original_score
                adjusted_candidate["similarity"] = avg_similarity
                adjusted_candidate["confidence"] = adjusted_score
                adjusted_candidates.append(adjusted_candidate)
            else:
                # ModelPrediction 객체인 경우
                metadata = candidate.metadata.copy() if candidate.metadata else {}
                metadata["original_confidence"] = original_score
                metadata["similarity"] = avg_similarity

                adjusted_candidate = ModelPrediction(
                    numbers=candidate.numbers,
                    confidence=adjusted_score,
                    model_type=candidate.model_type,
                    metadata=metadata,
                )
                adjusted_candidates.append(adjusted_candidate)

        return adjusted_candidates

    def _cluster_based_selection(
        self,
        candidates: List[RecommendationResult],
        count: int,
        method: str = "agglomerative",
    ) -> List[RecommendationResult]:
        """
        클러스터링 기반 다양한 후보 선택

        Args:
            candidates: 후보 목록
            count: 선택할 후보 수
            method: 클러스터링 방법 ("agglomerative" 또는 "dbscan")

        Returns:
            클러스터별로 선택된 후보 목록
        """
        if len(candidates) <= count:
            return candidates

        # 후보 번호를 벡터로 변환
        X = []
        for candidate in candidates:
            numbers = (
                candidate["numbers"]
                if isinstance(candidate, dict)
                else candidate.numbers
            )
            # 6개 로또 번호를 그대로 특성 벡터로 사용
            X.append(numbers)

        X = np.array(X)

        try:
            # 클러스터링 적용
            if method == "agglomerative":
                # 군집 수를 count의 2배로 설정하여 다양성 확보
                n_clusters = min(count * 2, len(candidates) // 2)
                if n_clusters < 2:
                    n_clusters = 2  # 최소 2개 클러스터

                # 계층적 클러스터링 적용
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, metric="euclidean", linkage="ward"
                ).fit(X)
                labels = clustering.labels_

            elif method == "dbscan":
                # DBSCAN 적용 (밀도 기반 클러스터링)
                # eps와 min_samples 파라미터는 데이터에 맞게 조정 필요
                clustering = DBSCAN(eps=3.0, min_samples=2).fit(X)
                labels = clustering.labels_

            else:
                logger.warning(
                    f"알 수 없는 클러스터링 방법: {method}, 'agglomerative'로 대체"
                )
                clustering = AgglomerativeClustering(n_clusters=count * 2).fit(X)
                labels = clustering.labels_

            # 클러스터별 후보 그룹화
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)

            # 각 클러스터에서 최고 점수 후보 선택
            selected_candidates = []
            cluster_labels = sorted(clusters.keys())

            for label in cluster_labels:
                if len(selected_candidates) >= count:
                    break

                cluster_indices = clusters[label]
                # 클러스터 내 후보를 점수 기준으로 정렬
                cluster_candidates = [candidates[i] for i in cluster_indices]
                sorted_cluster = sorted(
                    cluster_candidates,
                    key=lambda x: (
                        x["confidence"] if isinstance(x, dict) else x.confidence
                    ),
                    reverse=True,
                )

                # 클러스터에서 최고 점수 후보 선택
                if sorted_cluster:
                    selected_candidates.append(sorted_cluster[0])

            # 필요한 수만큼 선택되지 않았다면 점수 기준으로 나머지 채우기
            if len(selected_candidates) < count:
                logger.warning(
                    f"클러스터 기반 선택으로 충분한 후보를 찾지 못했습니다: {len(selected_candidates)}/{count}"
                )

                # 점수 기준으로 정렬된 전체 후보
                sorted_candidates = sorted(
                    candidates,
                    key=lambda x: (
                        x["confidence"] if isinstance(x, dict) else x.confidence
                    ),
                    reverse=True,
                )

                # 아직 선택되지 않은 후보 중에서 추가
                for candidate in sorted_candidates:
                    if (
                        candidate not in selected_candidates
                        and len(selected_candidates) < count
                    ):
                        selected_candidates.append(candidate)

            return selected_candidates[:count]

        except Exception as e:
            logger.error(f"클러스터링 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 점수 기반 선택으로 폴백
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x["confidence"] if isinstance(x, dict) else x.confidence,
                reverse=True,
            )
            return sorted_candidates[:count]

    def _apply_diversity_filtering(
        self, recommendations: List[RecommendationResult], count: int
    ) -> List[RecommendationResult]:
        """
        다양성 필터링 적용 (Jaccard 유사도 및 클러스터링 기반)

        Args:
            recommendations: 추천 후보 목록
            count: 선택할 후보 수

        Returns:
            다양성이 확보된 추천 목록
        """
        # 설정에서 다양성 필터링 옵션 로드
        try:
            config_proxy = ConfigProxy(self.config)
            try:
                enable_jaccard = config_proxy["recommendation"]["enable_jaccard_filter"]
            except KeyError:
                logger.warning(
                    "'recommendation.enable_jaccard_filter' 설정이 없습니다. 기본값(True)을 사용합니다."
                )
                enable_jaccard = True

            try:
                jaccard_threshold = config_proxy["recommendation"]["jaccard_threshold"]
            except KeyError:
                logger.warning(
                    "'recommendation.jaccard_threshold' 설정이 없습니다. 기본값(0.5)을 사용합니다."
                )
                jaccard_threshold = 0.5

            try:
                use_adjusted_score = config_proxy["recommendation"][
                    "use_adjusted_score"
                ]
            except KeyError:
                logger.warning(
                    "'recommendation.use_adjusted_score' 설정이 없습니다. 기본값(True)을 사용합니다."
                )
                use_adjusted_score = True

            try:
                enable_clustering = config_proxy["recommendation"][
                    "clustering_diversity"
                ]
            except KeyError:
                logger.warning(
                    "'recommendation.clustering_diversity' 설정이 없습니다. 기본값(True)을 사용합니다."
                )
                enable_clustering = True

            try:
                cluster_method = config_proxy["recommendation"][
                    "candidate_cluster_method"
                ]
            except KeyError:
                logger.warning(
                    "'recommendation.candidate_cluster_method' 설정이 없습니다. 기본값('agglomerative')을 사용합니다."
                )
                cluster_method = "agglomerative"
        except Exception as e:
            logger.warning(f"다양성 필터링 설정 로드 중 오류: {str(e)}")
            logger.error(f"성능 보고서 생성 중 오류 발생: {str(e)}")
            return ""


def get_recommendation_engine(
    config: Optional[Dict[str, Any]] = None,
) -> RecommendationEngine:
    """
    추천 엔진 인스턴스 생성

    Args:
        config: 설정 객체

    Returns:
        RecommendationEngine 인스턴스
    """
    return RecommendationEngine(config)
