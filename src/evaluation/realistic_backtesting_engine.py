"""
현실적 백테스팅 엔진 (Realistic Backtesting Engine)

등급별 적중률 최적화와 손실 최소화에 특화된 백테스팅 시스템입니다.

주요 기능:
- 등급별 적중률 추적 (5등, 4등, 3등)
- ROI 계산 및 분석
- 손실률 분석
- 성능 지표 계산
- 전략별 성과 비교
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

from ..utils.unified_logging import get_logger
from ..utils.cache_paths import get_cache_dir
from ..models.realistic_lottery_predictor import (
    RealisticLotteryPredictor,
    PrizeGrade,
    PredictionResult,
)
from ..utils.memory_manager import get_memory_manager

logger = get_logger(__name__)


class BacktestPeriod(Enum):
    """백테스트 기간"""

    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class BacktestResult:
    """백테스트 결과"""

    round_number: int
    predictions: List[List[int]]
    actual_numbers: List[int]
    hit_results: Dict[str, Any]
    roi: float
    profit_loss: float
    strategy_performance: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """성과 지표"""

    total_rounds: int
    hit_rates: Dict[PrizeGrade, float]
    average_roi: float
    total_profit_loss: float
    win_rate: float
    max_consecutive_losses: int
    sharpe_ratio: float
    strategy_performance: Dict[str, Dict[str, float]]


class RealisticBacktestingEngine:
    """현실적 백테스팅 엔진"""

    def __init__(
        self,
        predictor: RealisticLotteryPredictor,
        config_path: str = "config/realistic_lottery_config.yaml",
    ):
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()
        self.predictor = predictor

        # 설정 로드
        self.config = self._load_config(config_path)

        # 백테스트 결과 저장
        self.backtest_results: List[BacktestResult] = []

        # 성과 지표 초기화
        self.performance_metrics = {
            "hit_rates_by_grade": {grade: [] for grade in PrizeGrade},
            "roi_by_strategy": {},
            "loss_reduction": [],
            "prediction_accuracy": [],
            "consecutive_losses": 0,
            "max_consecutive_losses": 0,
            "total_investment": 0,
            "total_return": 0,
        }

        # 캐시 디렉토리 설정
        self.cache_dir = get_cache_dir("backtesting")

        # 등급별 상금 설정 (평균값)
        self.prize_amounts = {
            PrizeGrade.FIRST: 2000000000,  # 20억
            PrizeGrade.SECOND: 50000000,  # 5천만
            PrizeGrade.THIRD: 1500000,  # 150만
            PrizeGrade.FOURTH: 50000,  # 5만
            PrizeGrade.FIFTH: 5000,  # 5천
        }

        self.cost_per_combination = 1000  # 1천원

        # 성능 최적화를 위한 NumPy 배열 준비
        self._roi_cache = {}

        # GPU 가속 준비
        try:
            import cupy as cp  # type: ignore

            self.use_gpu = True
            self.cp = cp
            self.logger.info("GPU 가속 활성화")
        except ImportError:
            self.use_gpu = False
            self.cp = np
            self.logger.info("CPU 모드로 실행")

        self.logger.info("✅ 현실적 백테스팅 엔진 초기화 완료")

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("realistic_lottery", {}).get("backtesting", {})
        except FileNotFoundError:
            self.logger.warning(f"설정 파일 없음: {config_path}, 기본 설정 사용")
            return {
                "test_period_weeks": 52,
                "validation_split": 0.2,
                "minimum_test_rounds": 100,
            }

    def run_historical_backtest(
        self, historical_data: pd.DataFrame, start_round: int, end_round: int
    ) -> PerformanceMetrics:
        """과거 데이터 백테스팅"""
        try:
            self.logger.info(f"백테스팅 시작: {start_round}회차 ~ {end_round}회차")

            # 백테스트 결과 초기화
            self.backtest_results = []
            self._reset_performance_metrics()

            # 각 회차별 백테스트 실행
            for round_num in range(start_round, end_round + 1):
                if round_num >= len(historical_data):
                    break

                # 해당 회차까지의 데이터로 예측
                training_data = historical_data.iloc[:round_num]
                if len(training_data) < 10:  # 최소 데이터 확보
                    continue

                # 예측 실행
                predictions = self.predictor.predict_with_grade_optimization(
                    training_data, total_combinations=5
                )

                # 실제 결과 가져오기
                actual_numbers = self._get_actual_numbers(historical_data, round_num)
                if not actual_numbers:
                    continue

                # 백테스트 결과 계산
                backtest_result = self._calculate_backtest_result(
                    round_num, predictions, actual_numbers
                )

                # 결과 저장
                self.backtest_results.append(backtest_result)

                # 성과 지표 업데이트
                self._update_performance_metrics(backtest_result)

                if round_num % 10 == 0:
                    self.logger.info(f"백테스팅 진행: {round_num}회차 완료")

            # 최종 성과 지표 계산
            final_metrics = self._calculate_final_metrics()

            # 결과 저장
            self._save_backtest_results(start_round, end_round, final_metrics)

            self.logger.info(f"백테스팅 완료: 총 {len(self.backtest_results)}회차")
            return final_metrics

        except Exception as e:
            self.logger.error(f"백테스팅 실행 오류: {e}")
            raise

    def _get_actual_numbers(
        self, historical_data: pd.DataFrame, round_num: int
    ) -> List[int]:
        """실제 당첨 번호 가져오기"""
        try:
            if round_num >= len(historical_data):
                return []

            row = historical_data.iloc[round_num]
            numbers = [int(row[f"number_{i}"]) for i in range(1, 7)]
            return numbers

        except Exception as e:
            self.logger.error(f"실제 번호 가져오기 오류 (회차 {round_num}): {e}")
            return []

    def _calculate_backtest_result(
        self,
        round_number: int,
        predictions: List[PredictionResult],
        actual_numbers: List[int],
    ) -> BacktestResult:
        """백테스트 결과 계산"""
        try:
            # 모든 예측 조합 수집
            all_combinations = []
            strategy_combinations = {}

            for pred_result in predictions:
                all_combinations.extend(pred_result.combinations)
                strategy_combinations[pred_result.strategy_type] = (
                    pred_result.combinations
                )

            # 적중 결과 계산
            hit_results = self._calculate_hit_results(all_combinations, actual_numbers)

            # ROI 및 손익 계산
            roi, profit_loss = self._calculate_roi_and_profit(
                hit_results, len(all_combinations)
            )

            # 전략별 성과 계산
            strategy_performance = self._calculate_strategy_performance(
                strategy_combinations, actual_numbers
            )

            return BacktestResult(
                round_number=round_number,
                predictions=all_combinations,
                actual_numbers=actual_numbers,
                hit_results=hit_results,
                roi=roi,
                profit_loss=profit_loss,
                strategy_performance=strategy_performance,
            )

        except Exception as e:
            self.logger.error(f"백테스트 결과 계산 오류 (회차 {round_number}): {e}")
            return BacktestResult(
                round_number=round_number,
                predictions=[],
                actual_numbers=actual_numbers,
                hit_results={},
                roi=0.0,
                profit_loss=0.0,
                strategy_performance={},
            )

    def _calculate_hit_results(
        self, combinations: List[List[int]], actual_numbers: List[int]
    ) -> Dict[str, Any]:
        """적중 결과 계산"""
        try:
            hit_results = {
                "total_combinations": len(combinations),
                "hits_by_grade": {grade.value: 0 for grade in PrizeGrade},
                "hit_combinations": [],
                "total_matches": 0,
            }

            for combo in combinations:
                matches = len(set(combo) & set(actual_numbers))
                hit_results["total_matches"] += matches

                # 등급별 적중 판정
                if matches >= 3:  # 3개 이상 일치
                    grade = self._determine_prize_grade(matches)
                    if grade:
                        hit_results["hits_by_grade"][grade.value] += 1
                        hit_results["hit_combinations"].append(
                            {
                                "combination": combo,
                                "matches": matches,
                                "grade": grade.value,
                            }
                        )

            return hit_results

        except Exception as e:
            self.logger.error(f"적중 결과 계산 오류: {e}")
            return {}

    def _determine_prize_grade(self, matches: int) -> Optional[PrizeGrade]:
        """적중 개수에 따른 등급 결정"""
        if matches == 6:
            return PrizeGrade.FIRST
        elif matches == 5:
            return PrizeGrade.THIRD  # 보너스 번호는 고려하지 않음
        elif matches == 4:
            return PrizeGrade.FOURTH
        elif matches == 3:
            return PrizeGrade.FIFTH
        return None

    def _calculate_roi_and_profit(
        self, hit_results: Dict[str, Any], total_combinations: int
    ) -> Tuple[float, float]:
        """ROI 및 손익 계산"""
        try:
            total_cost = total_combinations * self.cost_per_combination
            total_prize = 0

            # 등급별 상금 합산
            for grade_value, hit_count in hit_results.get("hits_by_grade", {}).items():
                if hit_count > 0:
                    grade = PrizeGrade(grade_value)
                    if grade in self.prize_amounts:
                        total_prize += hit_count * self.prize_amounts[grade]

            # 손익 계산
            profit_loss = total_prize - total_cost

            # ROI 계산
            roi = (profit_loss / total_cost) * 100 if total_cost > 0 else 0

            return roi, profit_loss

        except Exception as e:
            self.logger.error(f"ROI 손익 계산 오류: {e}")
            return 0.0, 0.0

    def _calculate_strategy_performance(
        self,
        strategy_combinations: Dict[str, List[List[int]]],
        actual_numbers: List[int],
    ) -> Dict[str, float]:
        """전략별 성과 계산"""
        try:
            strategy_performance = {}

            for strategy_name, combinations in strategy_combinations.items():
                if not combinations:
                    strategy_performance[strategy_name] = 0.0
                    continue

                # 전략별 적중 결과 계산
                hit_results = self._calculate_hit_results(combinations, actual_numbers)
                roi, _ = self._calculate_roi_and_profit(hit_results, len(combinations))

                strategy_performance[strategy_name] = roi

            return strategy_performance

        except Exception as e:
            self.logger.error(f"전략별 성과 계산 오류: {e}")
            return {}

    def _update_performance_metrics(self, backtest_result: BacktestResult):
        """성과 지표 업데이트"""
        try:
            # 등급별 적중률 업데이트
            for grade_value, hit_count in backtest_result.hit_results.get(
                "hits_by_grade", {}
            ).items():
                grade = PrizeGrade(grade_value)
                hit_rate = hit_count / backtest_result.hit_results.get(
                    "total_combinations", 1
                )
                self.performance_metrics["hit_rates_by_grade"][grade].append(hit_rate)

            # 전략별 ROI 업데이트
            for strategy_name, roi in backtest_result.strategy_performance.items():
                if strategy_name not in self.performance_metrics["roi_by_strategy"]:
                    self.performance_metrics["roi_by_strategy"][strategy_name] = []
                self.performance_metrics["roi_by_strategy"][strategy_name].append(roi)

            # 연속 손실 추적
            if backtest_result.profit_loss < 0:
                self.performance_metrics["consecutive_losses"] += 1
                self.performance_metrics["max_consecutive_losses"] = max(
                    self.performance_metrics["max_consecutive_losses"],
                    self.performance_metrics["consecutive_losses"],
                )
            else:
                self.performance_metrics["consecutive_losses"] = 0

            # 총 투자 및 수익 누적
            total_combinations = backtest_result.hit_results.get(
                "total_combinations", 0
            )
            self.performance_metrics["total_investment"] += (
                total_combinations * self.cost_per_combination
            )
            self.performance_metrics["total_return"] += backtest_result.profit_loss + (
                total_combinations * self.cost_per_combination
            )

        except Exception as e:
            self.logger.error(f"성과 지표 업데이트 오류: {e}")

    def _calculate_final_metrics(self) -> PerformanceMetrics:
        """최종 성과 지표 계산"""
        try:
            # 등급별 평균 적중률 계산
            hit_rates = {}
            for grade, rates in self.performance_metrics["hit_rates_by_grade"].items():
                hit_rates[grade] = np.mean(rates) if rates else 0.0

            # 평균 ROI 계산
            all_rois = []
            for strategy_rois in self.performance_metrics["roi_by_strategy"].values():
                all_rois.extend(strategy_rois)
            average_roi = np.mean(all_rois) if all_rois else 0.0

            # 총 손익 계산
            total_profit_loss = (
                self.performance_metrics["total_return"]
                - self.performance_metrics["total_investment"]
            )

            # 승률 계산
            win_rounds = sum(
                1 for result in self.backtest_results if result.profit_loss > 0
            )
            win_rate = (
                win_rounds / len(self.backtest_results)
                if self.backtest_results
                else 0.0
            )

            # 샤프 비율 계산
            sharpe_ratio = self._calculate_sharpe_ratio(all_rois)

            # 전략별 성과 요약
            strategy_performance = {}
            for strategy_name, rois in self.performance_metrics[
                "roi_by_strategy"
            ].items():
                strategy_performance[strategy_name] = {
                    "average_roi": np.mean(rois) if rois else 0.0,
                    "win_rate": (
                        sum(1 for roi in rois if roi > 0) / len(rois) if rois else 0.0
                    ),
                    "max_roi": max(rois) if rois else 0.0,
                    "min_roi": min(rois) if rois else 0.0,
                }

            return PerformanceMetrics(
                total_rounds=len(self.backtest_results),
                hit_rates=hit_rates,
                average_roi=average_roi,
                total_profit_loss=total_profit_loss,
                win_rate=win_rate,
                max_consecutive_losses=self.performance_metrics[
                    "max_consecutive_losses"
                ],
                sharpe_ratio=sharpe_ratio,
                strategy_performance=strategy_performance,
            )

        except Exception as e:
            self.logger.error(f"최종 성과 지표 계산 오류: {e}")
            return PerformanceMetrics(
                total_rounds=0,
                hit_rates={},
                average_roi=0.0,
                total_profit_loss=0.0,
                win_rate=0.0,
                max_consecutive_losses=0,
                sharpe_ratio=0.0,
                strategy_performance={},
            )

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """샤프 비율 계산"""
        try:
            if not returns or len(returns) < 2:
                return 0.0

            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return == 0:
                return 0.0

            # 무위험 수익률을 0으로 가정
            sharpe_ratio = mean_return / std_return
            return sharpe_ratio

        except Exception as e:
            self.logger.error(f"샤프 비율 계산 오류: {e}")
            return 0.0

    def _reset_performance_metrics(self):
        """성과 지표 초기화"""
        self.performance_metrics = {
            "hit_rates_by_grade": {grade: [] for grade in PrizeGrade},
            "roi_by_strategy": {},
            "loss_reduction": [],
            "prediction_accuracy": [],
            "consecutive_losses": 0,
            "max_consecutive_losses": 0,
            "total_investment": 0,
            "total_return": 0,
        }

    def generate_performance_report(
        self, metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """성과 리포트 생성"""
        try:
            report = {
                "summary": {
                    "total_rounds": metrics.total_rounds,
                    "average_roi": round(metrics.average_roi, 2),
                    "total_profit_loss": round(metrics.total_profit_loss, 0),
                    "win_rate": round(metrics.win_rate * 100, 1),
                    "max_consecutive_losses": metrics.max_consecutive_losses,
                    "sharpe_ratio": round(metrics.sharpe_ratio, 3),
                },
                "hit_rates": {
                    f"{grade.value}등": f"{round(rate * 100, 2)}%"
                    for grade, rate in metrics.hit_rates.items()
                },
                "strategy_performance": {
                    strategy: {
                        "average_roi": f"{round(perf['average_roi'], 2)}%",
                        "win_rate": f"{round(perf['win_rate'] * 100, 1)}%",
                        "max_roi": f"{round(perf['max_roi'], 2)}%",
                        "min_roi": f"{round(perf['min_roi'], 2)}%",
                    }
                    for strategy, perf in metrics.strategy_performance.items()
                },
                "target_achievement": self._calculate_target_achievement(metrics),
                "recommendations": self._generate_recommendations(metrics),
            }

            return report

        except Exception as e:
            self.logger.error(f"성과 리포트 생성 오류: {e}")
            return {}

    def _calculate_target_achievement(
        self, metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """목표 달성도 계산"""
        try:
            # 목표 적중률 (설정에서 가져오기)
            target_hit_rates = {
                PrizeGrade.FIFTH: 0.35,  # 35%
                PrizeGrade.FOURTH: 0.05,  # 5%
                PrizeGrade.THIRD: 0.003,  # 0.3%
            }

            achievement = {}
            for grade, target_rate in target_hit_rates.items():
                actual_rate = metrics.hit_rates.get(grade, 0.0)
                achievement_rate = (
                    (actual_rate / target_rate) * 100 if target_rate > 0 else 0
                )

                achievement[f"{grade.value}등"] = {
                    "target": f"{target_rate * 100}%",
                    "actual": f"{actual_rate * 100:.2f}%",
                    "achievement": f"{achievement_rate:.1f}%",
                }

            return achievement

        except Exception as e:
            self.logger.error(f"목표 달성도 계산 오류: {e}")
            return {}

    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """개선 권고사항 생성"""
        try:
            recommendations = []

            # 적중률 기반 권고
            if metrics.hit_rates.get(PrizeGrade.FIFTH, 0) < 0.2:
                recommendations.append(
                    "5등 적중률이 낮습니다. 보수적 전략 비중을 늘려보세요."
                )

            # ROI 기반 권고
            if metrics.average_roi < -50:
                recommendations.append(
                    "평균 ROI가 매우 낮습니다. 리스크 관리를 강화하세요."
                )

            # 연속 손실 기반 권고
            if metrics.max_consecutive_losses > 15:
                recommendations.append(
                    "연속 손실이 많습니다. 포트폴리오 다양화를 고려하세요."
                )

            # 전략별 권고
            if metrics.strategy_performance:
                best_strategy = max(
                    metrics.strategy_performance.items(),
                    key=lambda x: x[1]["average_roi"],
                )
                recommendations.append(
                    f"'{best_strategy[0]}' 전략이 가장 우수합니다. 비중을 늘려보세요."
                )

            return recommendations

        except Exception as e:
            self.logger.error(f"권고사항 생성 오류: {e}")
            return []

    def _save_backtest_results(
        self, start_round: int, end_round: int, metrics: PerformanceMetrics
    ):
        """백테스트 결과 저장"""
        try:
            # 결과 파일 경로
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = (
                self.cache_dir / f"backtest_{start_round}_{end_round}_{timestamp}.json"
            )

            # 직렬화 가능한 형태로 변환
            serializable_results = {
                "backtest_info": {
                    "start_round": start_round,
                    "end_round": end_round,
                    "total_rounds": len(self.backtest_results),
                    "timestamp": timestamp,
                },
                "performance_metrics": {
                    "total_rounds": metrics.total_rounds,
                    "hit_rates": {
                        grade.value: rate for grade, rate in metrics.hit_rates.items()
                    },
                    "average_roi": metrics.average_roi,
                    "total_profit_loss": metrics.total_profit_loss,
                    "win_rate": metrics.win_rate,
                    "max_consecutive_losses": metrics.max_consecutive_losses,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "strategy_performance": metrics.strategy_performance,
                },
                "detailed_results": [
                    {
                        "round_number": result.round_number,
                        "predictions_count": len(result.predictions),
                        "hit_results": result.hit_results,
                        "roi": result.roi,
                        "profit_loss": result.profit_loss,
                        "strategy_performance": result.strategy_performance,
                    }
                    for result in self.backtest_results
                ],
            }

            # 파일 저장
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)

            self.logger.info(f"백테스트 결과 저장 완료: {results_file}")

        except Exception as e:
            self.logger.error(f"백테스트 결과 저장 오류: {e}")

    def compare_strategies(
        self, strategy_a: str, strategy_b: str, test_weeks: int = 12
    ) -> Dict[str, Any]:
        """전략 간 성과 비교"""
        try:
            # 전략별 성과 데이터 추출
            strategy_a_rois_raw = self.performance_metrics["roi_by_strategy"].get(
                strategy_a, []
            )
            strategy_b_rois_raw = self.performance_metrics["roi_by_strategy"].get(
                strategy_b, []
            )

            if not strategy_a_rois_raw or not strategy_b_rois_raw:
                return {"error": "비교할 전략 데이터가 부족합니다."}

            # ROI 값들을 안전하게 float로 변환
            strategy_a_rois = [
                self._extract_roi_value(roi) for roi in strategy_a_rois_raw
            ]
            strategy_b_rois = [
                self._extract_roi_value(roi) for roi in strategy_b_rois_raw
            ]

            # 기본 통계 계산
            comparison = {
                "strategy_a": {
                    "name": strategy_a,
                    "average_roi": np.mean(strategy_a_rois),
                    "std_roi": np.std(strategy_a_rois),
                    "win_rate": sum(1 for roi in strategy_a_rois if roi > 0)
                    / len(strategy_a_rois),
                    "max_roi": max(strategy_a_rois),
                    "min_roi": min(strategy_a_rois),
                },
                "strategy_b": {
                    "name": strategy_b,
                    "average_roi": np.mean(strategy_b_rois),
                    "std_roi": np.std(strategy_b_rois),
                    "win_rate": sum(1 for roi in strategy_b_rois if roi > 0)
                    / len(strategy_b_rois),
                    "max_roi": max(strategy_b_rois),
                    "min_roi": min(strategy_b_rois),
                },
            }

            # 통계적 유의성 검정
            significance_test = self._statistical_significance_test(
                strategy_a_rois, strategy_b_rois
            )
            comparison["statistical_test"] = significance_test

            # 승자 결정
            if (
                comparison["strategy_a"]["average_roi"]
                > comparison["strategy_b"]["average_roi"]
            ):
                comparison["winner"] = strategy_a
            else:
                comparison["winner"] = strategy_b

            return comparison

        except Exception as e:
            self.logger.error(f"전략 비교 오류: {e}")
            return {}

    def _statistical_significance_test(
        self, results_a: List[float], results_b: List[float]
    ) -> Dict[str, Any]:
        """통계적 유의성 검정"""
        try:
            from scipy import stats

            # t-검정 수행
            t_stat, p_value = stats.ttest_ind(results_a, results_b)

            # 결과 해석 - 타입 안전성 확보
            p_value_float = (
                float(p_value)
                if not isinstance(p_value, (tuple, list))
                else float(p_value[0]) if p_value else 1.0
            )
            is_significant = p_value_float < 0.05

            return {
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_significant": is_significant,
                "interpretation": (
                    "통계적으로 유의한 차이가 있습니다."
                    if is_significant
                    else "통계적으로 유의한 차이가 없습니다."
                ),
            }

        except ImportError:
            return {"error": "scipy 라이브러리가 필요합니다."}
        except Exception as e:
            self.logger.error(f"통계적 유의성 검정 오류: {e}")
            return {"error": str(e)}

    def _extract_roi_value(self, roi_value: Union[float, tuple, list, Any]) -> float:
        """
        ROI 값을 안전하게 float로 변환

        Args:
            roi_value: 변환할 ROI 값 (float, tuple, list 등)

        Returns:
            float: 변환된 ROI 값
        """
        try:
            if isinstance(roi_value, (tuple, list)):
                # tuple이나 list인 경우 첫 번째 요소 사용
                return float(roi_value[0]) if roi_value else 0.0
            elif isinstance(roi_value, (int, float)):
                # 숫자인 경우 직접 변환
                return float(roi_value)
            elif roi_value is None:
                return 0.0
            else:
                # 기타 타입인 경우 문자열로 변환 후 float 시도
                return float(str(roi_value))
        except (TypeError, ValueError, IndexError):
            # 변환 실패 시 기본값 반환
            self.logger.warning(f"ROI 값 변환 실패: {roi_value}, 기본값 0.0 사용")
            return 0.0

    def _validate_roi_type(self, roi_value: Any) -> float:
        """
        ROI 값 타입 검증 및 변환

        Args:
            roi_value: 검증할 ROI 값

        Returns:
            float: 검증된 ROI 값

        Raises:
            TypeError: 지원하지 않는 타입인 경우
        """
        if not isinstance(roi_value, (int, float, tuple, list, type(None))):
            raise TypeError(
                f"ROI 값은 숫자, tuple, list 타입이어야 합니다: {type(roi_value)}"
            )
        return self._extract_roi_value(roi_value)

    def _calculate_batch_roi(self, results: List[BacktestResult]) -> np.ndarray:
        """
        배치 ROI 계산으로 성능 향상

        Args:
            results: 백테스트 결과 리스트

        Returns:
            np.ndarray: ROI 값 배열
        """
        roi_values = np.array([self._extract_roi_value(r.roi) for r in results])

        # GPU 가속 적용 (데이터가 충분히 큰 경우)
        if self.use_gpu and len(roi_values) > 1000:
            try:
                gpu_data = self.cp.asarray(roi_values)
                # GPU에서 계산 수행
                processed_data = self.cp.asnumpy(gpu_data)
                return processed_data
            except Exception as e:
                self.logger.warning(f"GPU 가속 실패, CPU로 대체: {e}")
                return roi_values

        return roi_values

    def get_performance_summary(self) -> Dict[str, Any]:
        """성과 요약 정보 반환"""
        try:
            if not self.backtest_results:
                return {"error": "백테스트 결과가 없습니다."}

            # 최근 성과 계산
            recent_results = (
                self.backtest_results[-10:]
                if len(self.backtest_results) >= 10
                else self.backtest_results
            )

            # 최고/최악 라운드 계산 - 타입 안전성 확보
            best_round = None
            worst_round = None
            if self.backtest_results:
                # 벡터화된 ROI 계산으로 성능 향상
                roi_values = self._calculate_batch_roi(self.backtest_results)

                if len(roi_values) > 0:
                    best_idx = np.argmax(roi_values)
                    worst_idx = np.argmin(roi_values)

                    best_round = self.backtest_results[best_idx]
                    worst_round = self.backtest_results[worst_idx]

            # 안전한 ROI 계산
            recent_roi_values = [self._extract_roi_value(r.roi) for r in recent_results]
            all_roi_values = [
                self._extract_roi_value(r.roi) for r in self.backtest_results
            ]

            summary = {
                "total_rounds": len(self.backtest_results),
                "recent_performance": {
                    "average_roi": (
                        np.mean(recent_roi_values) if recent_roi_values else 0.0
                    ),
                    "win_rate": (
                        sum(1 for r in recent_results if r.profit_loss > 0)
                        / len(recent_results)
                        if recent_results
                        else 0.0
                    ),
                    "total_profit_loss": sum(r.profit_loss for r in recent_results),
                },
                "overall_performance": {
                    "average_roi": np.mean(all_roi_values) if all_roi_values else 0.0,
                    "total_profit_loss": sum(
                        r.profit_loss for r in self.backtest_results
                    ),
                    "max_consecutive_losses": self.performance_metrics[
                        "max_consecutive_losses"
                    ],
                },
                "best_round": (
                    {
                        "round_number": best_round.round_number if best_round else None,
                        "roi": (
                            self._extract_roi_value(best_round.roi)
                            if best_round
                            else 0.0
                        ),
                        "profit_loss": best_round.profit_loss if best_round else 0.0,
                    }
                    if best_round
                    else None
                ),
                "worst_round": (
                    {
                        "round_number": (
                            worst_round.round_number if worst_round else None
                        ),
                        "roi": (
                            self._extract_roi_value(worst_round.roi)
                            if worst_round
                            else 0.0
                        ),
                        "profit_loss": worst_round.profit_loss if worst_round else 0.0,
                    }
                    if worst_round
                    else None
                ),
            }

            return summary

        except Exception as e:
            self.logger.error(f"성과 요약 생성 오류: {e}")
            return {"error": str(e)}
