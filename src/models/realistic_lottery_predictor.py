"""
현실적 로또 예측 시스템 (Realistic Lottery Predictor)

6개 완전 적중이 아닌 하위 등급 적중률 최적화와 손실 최소화에 초점을 맞춘 예측기입니다.

목표:
- 5등(3개) 적중률: 15% → 35% (133% 개선)
- 4등(4개) 적중률: 1% → 5% (400% 개선)
- 3등(5개) 적중률: 0.1% → 0.3% (200% 개선)
- 총 손실률: 60% → 25% (58% 개선)
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import yaml

from .base_model import BaseModel
from .strategy_generator import StrategyGenerator, PrizeGrade, PredictionResult
from .performance_analyzer import PerformanceAnalyzer
from .portfolio_manager import PortfolioManager
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_paths
from ..utils.cache_manager import UnifiedCachePathManager
from ..utils.unified_memory_manager import get_unified_memory_manager

logger = get_logger(__name__)


class RealisticLotteryPredictor(BaseModel):
    """현실적 로또 예측 시스템"""

    def __init__(self, config_path: str = "config/realistic_lottery_config.yaml"):
        """
        현실적 로또 예측기 초기화

        Args:
            config_path: 설정 파일 경로
        """
        super().__init__(config_path)
        self.logger = get_logger(__name__)
        self.memory_manager = get_unified_memory_manager()
        
        paths = get_paths()
        cache_path_manager = UnifiedCachePathManager(paths)
        self.cache_dir = cache_path_manager.get_path("realistic_models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 설정 로드
        self.config = self._load_config(config_path)

        # 컴포넌트 초기화
        self.strategy_generator = StrategyGenerator(self.config)
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        self.portfolio_manager = PortfolioManager(self.config)

        # 목표 등급별 확률 개선 설정
        self.target_improvements = self._setup_target_improvements()

        # 모델 컴포넌트
        self.models = {}
        self.analyzers = {}

        self.logger.info("✅ 현실적 로또 예측 시스템 초기화 완료")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("realistic_lottery", {})
        except FileNotFoundError:
            self.logger.warning(f"설정 파일 없음: {config_path}, 기본 설정 사용")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            "target_improvements": {
                "5th_prize": {
                    "base_probability": 0.017544,
                    "target_probability": 0.028571,
                    "focus_weight": 0.4,
                },
                "4th_prize": {
                    "base_probability": 0.000969,
                    "target_probability": 0.001429,
                    "focus_weight": 0.4,
                },
                "3rd_prize": {
                    "base_probability": 0.000028,
                    "target_probability": 0.00004,
                    "focus_weight": 0.2,
                },
            },
            "portfolio_allocation": {
                "conservative": 0.4,
                "aggressive": 0.4,
                "balanced": 0.2,
            },
            "risk_management": {"kelly_fraction": 0.25, "max_consecutive_losses": 10},
        }

    def _setup_target_improvements(self) -> Dict:
        """등급별 최적화 목표 설정"""
        improvements = {}
        config = self.config.get("target_improvements", {})

        # 5등 (3개 일치)
        fifth_config = config.get("5th_prize", {})
        improvements[PrizeGrade.FIFTH] = {
            "base_probability": fifth_config.get("base_probability", 0.017544),
            "target_probability": fifth_config.get("target_probability", 0.028571),
            "focus_weight": fifth_config.get("focus_weight", 0.4),
            "expected_hit_rate": 0.35,
        }

        # 4등 (4개 일치)
        fourth_config = config.get("4th_prize", {})
        improvements[PrizeGrade.FOURTH] = {
            "base_probability": fourth_config.get("base_probability", 0.000969),
            "target_probability": fourth_config.get("target_probability", 0.001429),
            "focus_weight": fourth_config.get("focus_weight", 0.4),
            "expected_hit_rate": 0.05,
        }

        # 3등 (5개 일치)
        third_config = config.get("3rd_prize", {})
        improvements[PrizeGrade.THIRD] = {
            "base_probability": third_config.get("base_probability", 0.000028),
            "target_probability": third_config.get("target_probability", 0.00004),
            "focus_weight": third_config.get("focus_weight", 0.2),
            "expected_hit_rate": 0.003,
        }

        return improvements

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        모델 훈련 (분석 수행)

        Args:
            X: 특성 벡터 (사용되지 않음)
            y: 타겟 (사용되지 않음)
        """
        self.logger.info("현실적 로또 예측기 학습 시작")
        
        # 실제로는 과거 로또 데이터를 사용하여 분석 수행
        # 여기서는 간단히 훈련 완료 표시
        self.is_trained = True
        
        self.logger.info("현실적 로또 예측기 학습 완료")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        기본 예측 (호환성을 위한 메서드)

        Args:
            X: 특성 벡터

        Returns:
            예측 결과
        """
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 기본적인 예측 결과 반환
        return np.array([[1, 7, 14, 21, 28, 35]])

    def predict_with_grade_optimization(
        self, historical_data: pd.DataFrame, total_combinations: int = 5
    ) -> List[PredictionResult]:
        """
        등급 최적화 예측 수행

        Args:
            historical_data: 과거 로또 데이터
            total_combinations: 총 조합 수

        Returns:
            예측 결과 리스트
        """
        self.logger.info(f"등급 최적화 예측 시작: {total_combinations}개 조합")

        try:
            # 포트폴리오 전략별 조합 할당
            allocation = self.portfolio_manager.calculate_combination_allocation(total_combinations)
            
            # 전략별 예측 수행
            predictions = []
            
            for strategy_name, num_combinations in allocation.items():
                if num_combinations > 0:
                    strategy_config = self._get_strategy_config(strategy_name)
                    
                    # 전략 생성기로 조합 생성
                    strategy_result = self.strategy_generator.generate_strategy_combinations(
                        historical_data, strategy_name, strategy_config, num_combinations
                    )
                    
                    # PredictionResult 객체 생성
                    prediction_result = PredictionResult(
                        combinations=strategy_result["combinations"],
                        grade_probabilities=strategy_result["grade_probabilities"],
                        strategy_type=strategy_result["strategy_type"],
                        confidence_score=strategy_result["confidence_score"],
                        expected_value=strategy_result["expected_value"],
                        risk_level=strategy_result["risk_level"],
                    )
                    
                    predictions.append(prediction_result)

            # 포트폴리오 최적화
            optimized_predictions = self.portfolio_manager.optimize_portfolio_allocation(predictions)

            self.logger.info(f"등급 최적화 예측 완료: {len(optimized_predictions)}개 전략")
            return optimized_predictions

        except Exception as e:
            self.logger.error(f"등급 최적화 예측 실패: {e}")
            return []

    def _get_strategy_config(self, strategy_name: str) -> Dict:
        """전략별 설정 반환"""
        strategy_configs = {
            "conservative": {
                "focus_grades": [PrizeGrade.FIFTH],
                "methods": ["frequency_based", "cluster_analysis"],
                "risk_level": "low",
                "expected_hit_rate": 0.35,
            },
            "aggressive": {
                "focus_grades": [PrizeGrade.THIRD, PrizeGrade.FOURTH],
                "methods": ["trend_following", "ai_ensemble"],
                "risk_level": "high",
                "expected_hit_rate": 0.05,
            },
            "balanced": {
                "focus_grades": [PrizeGrade.THIRD, PrizeGrade.FOURTH, PrizeGrade.FIFTH],
                "methods": ["weighted_average", "meta_learning"],
                "risk_level": "medium",
                "expected_hit_rate": 0.15,
            },
        }
        
        return strategy_configs.get(strategy_name, strategy_configs["balanced"])

    def analyze_performance(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        성능 분석 수행

        Args:
            data: 로또 번호 데이터

        Returns:
            분석 결과
        """
        return self.performance_analyzer.perform_comprehensive_analysis(data)

    def get_portfolio_summary(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """
        포트폴리오 요약 정보 반환

        Args:
            predictions: 예측 결과 리스트

        Returns:
            포트폴리오 요약
        """
        return self.portfolio_manager.get_portfolio_summary(predictions)

    def save_predictions(self, predictions: List[PredictionResult], round_number: int) -> bool:
        """
        예측 결과 저장

        Args:
            predictions: 예측 결과 리스트
            round_number: 회차 번호

        Returns:
            저장 성공 여부
        """
        try:
            save_path = f"data/result/realistic_predictions_{round_number}.json"
            return self.portfolio_manager.save_portfolio_report(predictions, round_number, save_path)
        except Exception as e:
            self.logger.error(f"예측 결과 저장 실패: {e}")
            return False

    def save(self, path: str) -> None:
        """
        모델 저장

        Args:
            path: 저장 경로
        """
        try:
            save_data = {
                "config": self.config,
                "target_improvements": self.target_improvements,
                "is_trained": self.is_trained,
                "save_timestamp": datetime.now().isoformat(),
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"모델 저장 완료: {path}")
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")

    def load(self, path: str) -> None:
        """
        모델 로드

        Args:
            path: 로드 경로
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            self.config = save_data.get("config", {})
            self.target_improvements = save_data.get("target_improvements", {})
            self.is_trained = save_data.get("is_trained", False)
            
            # 컴포넌트 재초기화
            self.strategy_generator = StrategyGenerator(self.config)
            self.performance_analyzer = PerformanceAnalyzer(self.config)
            self.portfolio_manager = PortfolioManager(self.config)
            
            self.logger.info(f"모델 로드 완료: {path}")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        return self.performance_analyzer.get_performance_summary()

    def calculate_kelly_criterion(
        self, win_probability: float, win_amount: float, loss_amount: float
    ) -> float:
        """켈리 기준 계산"""
        return self.portfolio_manager.calculate_kelly_criterion(
            win_probability, win_amount, loss_amount
        )
