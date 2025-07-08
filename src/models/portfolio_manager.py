"""
포트폴리오 관리자 (Portfolio Manager)

로또 예측 포트폴리오를 관리하고 최적화하는 모듈입니다.
"""

import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


class PrizeGrade(Enum):
    """로또 등급"""
    FIRST = 1   # 6개 일치
    SECOND = 2  # 5개 일치 + 보너스
    THIRD = 3   # 5개 일치
    FOURTH = 4  # 4개 일치
    FIFTH = 5   # 3개 일치


@dataclass
class PredictionResult:
    """예측 결과"""
    combinations: List[List[int]]
    grade_probabilities: Dict[PrizeGrade, float]
    strategy_type: str
    confidence_score: float
    expected_value: float
    risk_level: str


class PortfolioManager:
    """포트폴리오 관리자"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        포트폴리오 관리자 초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # 포트폴리오 설정
        self.portfolio_allocation = self.config.get("portfolio_allocation", {
            "conservative": 0.4,
            "aggressive": 0.4,
            "balanced": 0.2,
        })
        
        # 리스크 관리 설정
        self.risk_management = self.config.get("risk_management", {
            "kelly_fraction": 0.25,
            "max_consecutive_losses": 10,
        })
        
        # 등급별 상금 정보
        self.prize_amounts = {
            PrizeGrade.FIRST: 2000000000,    # 1등: 20억
            PrizeGrade.SECOND: 50000000,     # 2등: 5천만
            PrizeGrade.THIRD: 1500000,       # 3등: 150만
            PrizeGrade.FOURTH: 50000,        # 4등: 5만
            PrizeGrade.FIFTH: 5000,          # 5등: 5천
        }
        
        # 게임당 비용
        self.cost_per_game = 1000
        
        self.logger.info("포트폴리오 관리자 초기화 완료")

    def calculate_combination_allocation(self, total_combinations: int) -> Dict[str, int]:
        """
        전략별 조합 할당 계산

        Args:
            total_combinations: 총 조합 수

        Returns:
            전략별 할당 조합 수
        """
        allocation = {}
        
        for strategy, weight in self.portfolio_allocation.items():
            allocated = int(total_combinations * weight)
            allocation[strategy] = max(1, allocated)  # 최소 1개는 할당
        
        # 합계 조정
        total_allocated = sum(allocation.values())
        if total_allocated != total_combinations:
            # 가장 큰 비중 전략에 차이만큼 조정
            max_strategy = max(allocation.keys(), key=lambda k: allocation[k])
            allocation[max_strategy] += total_combinations - total_allocated
        
        self.logger.info(f"조합 할당 완료: {allocation}")
        return allocation

    def calculate_expected_value(
        self,
        combinations: List[List[int]],
        grade_probabilities: Dict[PrizeGrade, float],
    ) -> float:
        """
        포트폴리오 기대값 계산

        Args:
            combinations: 조합 리스트
            grade_probabilities: 등급별 확률

        Returns:
            기대값
        """
        try:
            # 총 투자 비용
            total_cost = len(combinations) * self.cost_per_game
            
            # 기대 수익 계산
            expected_winnings = 0
            for grade, probability in grade_probabilities.items():
                prize = self.prize_amounts.get(grade, 0)
                expected_winnings += probability * prize
            
            # 기대값 = (기대 수익 - 비용) / 비용
            expected_value = (expected_winnings - total_cost) / total_cost
            
            return expected_value

        except Exception as e:
            self.logger.error(f"기대값 계산 실패: {e}")
            return -1.0

    def calculate_portfolio_risk(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        """
        포트폴리오 리스크 계산

        Args:
            predictions: 예측 결과 리스트

        Returns:
            리스크 메트릭
        """
        try:
            # 전략별 리스크 점수
            risk_scores = {
                "conservative": 0.2,
                "balanced": 0.5,
                "aggressive": 0.8,
            }
            
            # 포트폴리오 전체 리스크 계산
            total_combinations = sum(len(pred.combinations) for pred in predictions)
            weighted_risk = 0
            
            for pred in predictions:
                strategy_weight = len(pred.combinations) / total_combinations
                strategy_risk = risk_scores.get(pred.strategy_type, 0.5)
                weighted_risk += strategy_weight * strategy_risk
            
            # 분산화 효과 계산
            strategy_count = len(set(pred.strategy_type for pred in predictions))
            diversification_factor = min(1.0, strategy_count / 3)  # 최대 3개 전략
            
            # 신뢰도 기반 리스크 조정
            avg_confidence = np.mean([pred.confidence_score for pred in predictions])
            confidence_factor = 1.0 - avg_confidence
            
            # 최종 리스크 점수
            portfolio_risk = weighted_risk * (1 - diversification_factor * 0.2) * (1 + confidence_factor * 0.3)
            
            return {
                "portfolio_risk": min(1.0, portfolio_risk),
                "diversification_factor": diversification_factor,
                "average_confidence": avg_confidence,
                "strategy_count": strategy_count,
            }

        except Exception as e:
            self.logger.error(f"포트폴리오 리스크 계산 실패: {e}")
            return {"portfolio_risk": 0.5}

    def optimize_portfolio_allocation(
        self, 
        predictions: List[PredictionResult], 
        target_risk: float = 0.5
    ) -> List[PredictionResult]:
        """
        포트폴리오 할당 최적화

        Args:
            predictions: 예측 결과 리스트
            target_risk: 목표 리스크 수준

        Returns:
            최적화된 예측 결과 리스트
        """
        try:
            self.logger.info(f"포트폴리오 할당 최적화 시작 (목표 리스크: {target_risk})")
            
            # 전략별 성과 평가
            strategy_scores = {}
            for pred in predictions:
                score = (
                    pred.confidence_score * 0.4 +
                    max(0, pred.expected_value + 1) * 0.4 +  # 기대값 정규화
                    (1 - {"low": 0.2, "medium": 0.5, "high": 0.8}.get(pred.risk_level, 0.5)) * 0.2
                )
                strategy_scores[pred.strategy_type] = score
            
            # 최적 조합 수 계산
            sum(len(pred.combinations) for pred in predictions)
            optimized_predictions = []
            
            for pred in predictions:
                strategy_score = strategy_scores.get(pred.strategy_type, 0.5)
                
                # 성과 기반 조합 수 조정
                adjustment_factor = strategy_score / np.mean(list(strategy_scores.values()))
                new_combination_count = int(len(pred.combinations) * adjustment_factor)
                new_combination_count = max(1, min(new_combination_count, len(pred.combinations)))
                
                # 상위 조합 선택
                optimized_combinations = pred.combinations[:new_combination_count]
                
                # 새로운 예측 결과 생성
                optimized_pred = PredictionResult(
                    combinations=optimized_combinations,
                    grade_probabilities=pred.grade_probabilities,
                    strategy_type=pred.strategy_type,
                    confidence_score=pred.confidence_score,
                    expected_value=pred.expected_value,
                    risk_level=pred.risk_level,
                )
                optimized_predictions.append(optimized_pred)
            
            self.logger.info(f"포트폴리오 할당 최적화 완료: {len(optimized_predictions)}개 전략")
            return optimized_predictions

        except Exception as e:
            self.logger.error(f"포트폴리오 할당 최적화 실패: {e}")
            return predictions

    def get_portfolio_summary(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """
        포트폴리오 요약 정보 생성

        Args:
            predictions: 예측 결과 리스트

        Returns:
            포트폴리오 요약
        """
        try:
            total_combinations = sum(len(pred.combinations) for pred in predictions)
            total_investment = total_combinations * self.cost_per_game
            
            # 전략별 분포
            strategy_distribution = {}
            for pred in predictions:
                strategy = pred.strategy_type
                if strategy not in strategy_distribution:
                    strategy_distribution[strategy] = {
                        "combinations": 0,
                        "investment": 0,
                        "expected_value": 0,
                        "confidence": 0,
                    }
                
                strategy_distribution[strategy]["combinations"] += len(pred.combinations)
                strategy_distribution[strategy]["investment"] += len(pred.combinations) * self.cost_per_game
                strategy_distribution[strategy]["expected_value"] += pred.expected_value
                strategy_distribution[strategy]["confidence"] += pred.confidence_score

            # 평균 계산
            for strategy_info in strategy_distribution.values():
                strategy_info["expected_value"] /= len(predictions)
                strategy_info["confidence"] /= len(predictions)

            # 전체 기대값 계산
            total_expected_value = 0
            for pred in predictions:
                pred_expected_value = self.calculate_expected_value(
                    pred.combinations, pred.grade_probabilities
                )
                total_expected_value += pred_expected_value * len(pred.combinations)
            
            portfolio_expected_value = total_expected_value / total_combinations if total_combinations > 0 else 0

            # 리스크 계산
            risk_metrics = self.calculate_portfolio_risk(predictions)

            # 등급별 전체 확률
            combined_probabilities = {}
            for grade in PrizeGrade:
                grade_prob = 0
                for pred in predictions:
                    weight = len(pred.combinations) / total_combinations
                    grade_prob += pred.grade_probabilities.get(grade, 0) * weight
                combined_probabilities[grade] = grade_prob

            summary = {
                "total_combinations": total_combinations,
                "total_investment": total_investment,
                "portfolio_expected_value": portfolio_expected_value,
                "strategy_distribution": strategy_distribution,
                "risk_metrics": risk_metrics,
                "combined_probabilities": {grade.name: prob for grade, prob in combined_probabilities.items()},
                "recommendations": self._generate_recommendations(predictions, risk_metrics),
                "summary_date": datetime.now().isoformat(),
            }

            return summary

        except Exception as e:
            self.logger.error(f"포트폴리오 요약 생성 실패: {e}")
            return {"error": str(e)}

    def _generate_recommendations(
        self, 
        predictions: List[PredictionResult], 
        risk_metrics: Dict[str, float]
    ) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        try:
            # 리스크 기반 추천
            portfolio_risk = risk_metrics.get("portfolio_risk", 0.5)
            if portfolio_risk > 0.7:
                recommendations.append("포트폴리오 리스크가 높습니다. 보수적 전략 비중을 늘려보세요.")
            elif portfolio_risk < 0.3:
                recommendations.append("포트폴리오가 매우 보수적입니다. 수익성 향상을 위해 공격적 전략을 고려해보세요.")
            
            # 분산화 추천
            diversification = risk_metrics.get("diversification_factor", 0)
            if diversification < 0.5:
                recommendations.append("전략 다양화가 부족합니다. 다양한 전략을 조합해보세요.")
            
            # 신뢰도 기반 추천
            avg_confidence = risk_metrics.get("average_confidence", 0.5)
            if avg_confidence < 0.4:
                recommendations.append("예측 신뢰도가 낮습니다. 더 많은 데이터로 모델을 개선해보세요.")
            
            # 기대값 기반 추천
            expected_values = [pred.expected_value for pred in predictions]
            if all(ev < -0.8 for ev in expected_values):
                recommendations.append("모든 전략의 기대값이 매우 낮습니다. 투자 규모를 줄이거나 전략을 재검토하세요.")
            
            if not recommendations:
                recommendations.append("현재 포트폴리오 구성이 적절합니다.")
            
        except Exception as e:
            self.logger.error(f"추천사항 생성 실패: {e}")
            recommendations = ["추천사항 생성 중 오류가 발생했습니다."]
        
        return recommendations

    def save_portfolio_report(
        self, 
        predictions: List[PredictionResult], 
        round_number: int,
        save_path: str = None
    ) -> bool:
        """
        포트폴리오 보고서 저장

        Args:
            predictions: 예측 결과 리스트
            round_number: 회차 번호
            save_path: 저장 경로

        Returns:
            저장 성공 여부
        """
        try:
            if save_path is None:
                save_path = f"data/result/portfolio_report_{round_number}.json"
            
            # 포트폴리오 요약 생성
            summary = self.get_portfolio_summary(predictions)
            
            # 예측 결과 직렬화
            serialized_predictions = []
            for pred in predictions:
                serialized_pred = {
                    "combinations": pred.combinations,
                    "grade_probabilities": {grade.name: prob for grade, prob in pred.grade_probabilities.items()},
                    "strategy_type": pred.strategy_type,
                    "confidence_score": pred.confidence_score,
                    "expected_value": pred.expected_value,
                    "risk_level": pred.risk_level,
                }
                serialized_predictions.append(serialized_pred)
            
            # 전체 보고서 구성
            report = {
                "round_number": round_number,
                "predictions": serialized_predictions,
                "portfolio_summary": summary,
                "generation_date": datetime.now().isoformat(),
                "config": {
                    "portfolio_allocation": self.portfolio_allocation,
                    "risk_management": self.risk_management,
                },
            }
            
            # 파일 저장
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"포트폴리오 보고서 저장 완료: {save_path}")
            return True

        except Exception as e:
            self.logger.error(f"포트폴리오 보고서 저장 실패: {e}")
            return False

    def load_portfolio_report(self, file_path: str) -> Dict[str, Any]:
        """
        포트폴리오 보고서 로드

        Args:
            file_path: 파일 경로

        Returns:
            로드된 보고서 데이터
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            self.logger.info(f"포트폴리오 보고서 로드 완료: {file_path}")
            return report

        except Exception as e:
            self.logger.error(f"포트폴리오 보고서 로드 실패: {e}")
            return {}

    def calculate_kelly_criterion(
        self, 
        win_probability: float, 
        win_amount: float, 
        loss_amount: float
    ) -> float:
        """
        켈리 기준 계산

        Args:
            win_probability: 승률
            win_amount: 승리 시 수익
            loss_amount: 패배 시 손실

        Returns:
            최적 배팅 비율
        """
        try:
            if win_probability <= 0 or win_probability >= 1:
                return 0.0
            
            if win_amount <= 0 or loss_amount <= 0:
                return 0.0
            
            # 켈리 공식: f = (bp - q) / b
            # b = win_amount / loss_amount (배당률)
            # p = win_probability (승률)
            # q = 1 - p (패배 확률)
            
            b = win_amount / loss_amount
            p = win_probability
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # 켈리 비율 제한 (최대 25%)
            max_fraction = self.risk_management.get("kelly_fraction", 0.25)
            kelly_fraction = max(0, min(kelly_fraction, max_fraction))
            
            return kelly_fraction

        except Exception as e:
            self.logger.error(f"켈리 기준 계산 실패: {e}")
            return 0.0 