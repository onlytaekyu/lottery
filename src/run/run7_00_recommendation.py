"""
DAEBAK_AI 로또 예측 시스템 - 7단계: 최종 추천 결과 저장
이전 6단계의 모든 결과를 통합하여 최종 추천 조합을 생성하고 CSV로 저장
"""

import csv
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter
from dataclasses import dataclass, asdict

# 3. 프로젝트 내부 (리팩토링된 의존성 관리)
from ..utils.dependency_injection import configure_dependencies, resolve
from ..utils.unified_logging import get_logger
from ..utils.unified_memory_manager import UnifiedMemoryManager
from ..utils.unified_performance_engine import AutoPerformanceMonitor
from ..utils.unified_config import Config
from ..core.recommendation_engine import RecommendationEngine


@dataclass
class RecommendationConfig:
    """추천 시스템 설정"""
    # 추천 전략 가중치
    strategy_weights: Dict[str, float] = None
    
    # 생성할 추천 수
    total_recommendations: int = 100
    top_recommendations: int = 20
    
    # 품질 기준
    confidence_threshold: float = 0.6
    diversity_threshold: float = 0.8
    risk_threshold: float = 0.3
    
    # CSV 저장 설정
    csv_path: str = "D:/VSworkSpace/DAEBAK_AI/lottery/data/predictions/pred_lottery.csv"
    
    def __post_init__(self):
        if self.strategy_weights is None:
            self.strategy_weights = {
                "score_integrated": 0.25,    # run6 점수 통합 결과
                "risk_filtered": 0.20,       # run5 리스크 필터 결과
                "trend_corrected": 0.20,     # run4 트렌드 보정 결과
                "ml_predictions": 0.15,      # run2 ML 예측 결과
                "three_digit_priority": 0.10, # run1 3자리 우선 예측
                "balanced": 0.10             # 균형 조합 전략
            }


@dataclass
class FinalRecommendation:
    """최종 추천 조합"""
    combination_id: int
    numbers: List[int]
    total_score: float
    confidence: float
    strategy: str
    risk_score: float
    trend_strength: float
    diversity_metrics: Dict[str, Any]
    stage_contributions: Dict[str, float]


class Recommendation:
    """최종 추천 엔진"""
    
    def __init__(self):
        """
        추천 엔진 초기화 (의존성 주입 사용)
        """
        self.logger = get_logger(__name__)
        
        # --- 의존성 해결 ---
        self.config_manager: Config = resolve(Config)
        self.config = self.config_manager.get_config("main")
        self.paths = self.config_manager.get_paths()
        self.memory_manager: UnifiedMemoryManager = resolve(UnifiedMemoryManager)
        self.performance_monitor: AutoPerformanceMonitor = resolve(AutoPerformanceMonitor)
        self.recommendation_engine: RecommendationEngine = resolve(RecommendationEngine)
        # --------------------

        # 추천 설정
        self.recommendation_config = RecommendationConfig()
        
        # 캐시 디렉토리
        self.cache_dir = Path(self.paths.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장 경로 설정
        self.csv_path = Path(self.recommendation_config.csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ 최종 추천 엔진 초기화 완료")

    def load_all_stage_results(self) -> Dict[str, Any]:
        """
        이전 6단계의 모든 결과를 로드합니다.
        
        Returns:
            Dict[str, Any]: 통합된 모든 단계 결과
        """
        try:
            self.logger.info("📊 이전 6단계 결과 로드 시작...")
            
            with self.performance_monitor.track("load_all_stage_results"):
                results = {
                    "run1_analysis": {},
                    "run2_predictions": {},
                    "run3_anomaly": {},
                    "run4_trend": {},
                    "run5_risk": {},
                    "run6_integration": {},
                    "metadata": {}
                }
                
                # 각 단계별 결과 디렉토리 정의
                stage_dirs = {
                    "run1_analysis": Path(self.paths.result_dir) / "analysis",
                    "run2_predictions": Path(self.paths.predictions_dir),
                    "run3_anomaly": Path(self.paths.result_dir) / "anomaly_detection",
                    "run4_trend": Path(self.paths.result_dir) / "trend_correction",
                    "run5_risk": Path(self.paths.result_dir) / "risk_filter",
                    "run6_integration": Path(self.paths.result_dir) / "score_integration"
                }
                
                # 각 단계별 결과 로드
                for stage_name, stage_dir in stage_dirs.items():
                    if stage_dir.exists():
                        for file_path in stage_dir.glob("*.json"):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    results[stage_name][file_path.stem] = data
                            except Exception as e:
                                self.logger.warning(f"{stage_name} 파일 로드 실패 {file_path}: {e}")
                
                # 메타데이터 추가
                results["metadata"] = {
                    "load_timestamp": datetime.now().isoformat(),
                    "stage_file_counts": {
                        stage: len(data) for stage, data in results.items() 
                        if stage != "metadata"
                    },
                    "total_files": sum(
                        len(data) for stage, data in results.items() 
                        if stage != "metadata"
                    )
                }
            
            self.logger.info(f"✅ 전체 단계 결과 로드 완료: "
                           f"run1({len(results['run1_analysis'])}), "
                           f"run2({len(results['run2_predictions'])}), "
                           f"run3({len(results['run3_anomaly'])}), "
                           f"run4({len(results['run4_trend'])}), "
                           f"run5({len(results['run5_risk'])}), "
                           f"run6({len(results['run6_integration'])})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"단계 결과 로드 실패: {e}")
            raise

    def generate_final_recommendations(self, all_results: Dict[str, Any]) -> List[FinalRecommendation]:
        """
        최종 추천 조합을 생성합니다.
        
        Args:
            all_results: 모든 단계 결과
            
        Returns:
            List[FinalRecommendation]: 최종 추천 조합 목록
        """
        try:
            self.logger.info("🎯 최종 추천 조합 생성 시작...")
            
            with self.performance_monitor.track("generate_final_recommendations"):
                # 1. 다중 전략 기반 조합 생성
                strategy_combinations = {
                    "score_integrated": self._get_score_integrated_combinations(all_results),
                    "risk_filtered": self._get_risk_filtered_combinations(all_results),
                    "trend_corrected": self._get_trend_corrected_combinations(all_results),
                    "ml_predictions": self._get_ml_prediction_combinations(all_results),
                    "three_digit_priority": self._get_three_digit_combinations(all_results),
                    "balanced": self._get_balanced_combinations(all_results)
                }
                
                # 2. 전략별 가중치 적용 및 통합
                weighted_combinations = self._apply_strategy_weights(strategy_combinations)
                
                # 3. 중복 제거 및 다양성 보장
                diverse_combinations = self._ensure_diversity(weighted_combinations)
                
                # 4. 품질 검증 및 최종 순위화
                final_recommendations = self._apply_quality_validation(diverse_combinations, all_results)
                
                # 5. 상위 추천 선별
                top_recommendations = final_recommendations[:self.recommendation_config.total_recommendations]
                
            self.logger.info(f"✅ 최종 추천 조합 생성 완료: {len(top_recommendations)}개")
            return top_recommendations
            
        except Exception as e:
            self.logger.error(f"최종 추천 생성 실패: {e}")
            raise

    def _get_score_integrated_combinations(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """run6 점수 통합 결과 기반 조합 추출"""
        combinations = []
        
        for name, data in all_results["run6_integration"].items():
            if "integration_result" in data:
                integration_result = data["integration_result"]
                final_recommendations = integration_result.get("final_recommendations", {})
                
                if "final_top_combinations" in final_recommendations:
                    for combo in final_recommendations["final_top_combinations"]:
                        combinations.append({
                            "numbers": combo["numbers"],
                            "total_score": combo["total_score"],
                            "confidence": combo["confidence"],
                            "strategy": "score_integrated",
                            "source_stage": "run6"
                        })
                break
        
        return combinations[:50]  # 상위 50개

    def _get_risk_filtered_combinations(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """run5 리스크 필터 결과 기반 조합 추출"""
        combinations = []
        
        for name, data in all_results["run5_risk"].items():
            if "risk_filtered_combinations" in data:
                risk_combinations = data["risk_filtered_combinations"]
                
                for combo_id, combo_data in risk_combinations.items():
                    if "numbers" in combo_data and "confidence" in combo_data:
                        combinations.append({
                            "numbers": combo_data["numbers"],
                            "total_score": combo_data["confidence"],
                            "confidence": combo_data["confidence"],
                            "strategy": "risk_filtered",
                            "source_stage": "run5",
                            "risk_score": combo_data.get("risk_score", 0.2)
                        })
                break
        
        return combinations[:40]  # 상위 40개

    def _get_trend_corrected_combinations(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """run4 트렌드 보정 결과 기반 조합 생성"""
        combinations = []
        
        for name, data in all_results["run4_trend"].items():
            if "trend_correction_scores" in data:
                trend_scores = data["trend_correction_scores"]
                
                # 트렌드 점수 기반 상위 번호 선별
                sorted_numbers = sorted(
                    trend_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                top_numbers = [int(num) for num, score in sorted_numbers[:20]]
                
                # 조합 생성
                for combo in combinations(top_numbers, 6):
                    combo_list = list(combo)
                    total_score = sum(trend_scores[str(num)] for num in combo_list) / 6
                    
                    combinations.append({
                        "numbers": combo_list,
                        "total_score": total_score,
                        "confidence": min(total_score, 1.0),
                        "strategy": "trend_corrected",
                        "source_stage": "run4",
                        "trend_strength": total_score
                    })
                    
                    if len(combinations) >= 30:
                        break
                break
        
        return combinations[:30]

    def _get_ml_prediction_combinations(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """run2 ML 예측 결과 기반 조합 추출"""
        combinations = []
        
        for name, data in all_results["run2_predictions"].items():
            if "predictions" in data:
                predictions = data["predictions"]
                
                # ML 예측 점수 기반 상위 번호 선별
                if "pattern_scores" in predictions:
                    pattern_scores = predictions["pattern_scores"]
                    if isinstance(pattern_scores, list):
                        # 배열 형태인 경우 번호별 점수로 변환
                        ml_scores = {str(i+1): score for i, score in enumerate(pattern_scores)}
                    else:
                        ml_scores = predictions.get("predictions", {})
                elif "predictions" in predictions:
                    ml_scores = predictions["predictions"]
                else:
                    continue
                
                # 상위 번호 선별
                sorted_numbers = sorted(
                    ml_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                top_numbers = [int(num) for num, score in sorted_numbers[:18]]
                
                # 조합 생성
                for combo in combinations(top_numbers, 6):
                    combo_list = list(combo)
                    total_score = sum(ml_scores.get(str(num), 0) for num in combo_list) / 6
                    
                    combinations.append({
                        "numbers": combo_list,
                        "total_score": total_score,
                        "confidence": min(total_score, 1.0),
                        "strategy": "ml_predictions",
                        "source_stage": "run2"
                    })
                    
                    if len(combinations) >= 25:
                        break
                break
        
        return combinations[:25]

    def _get_three_digit_combinations(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """run1 3자리 우선 예측 결과 기반 조합 추출"""
        combinations = []
        
        for name, data in all_results["run1_analysis"].items():
            if "3digit_priority_predictions" in name or "three_digit" in name:
                if "priority_predictions" in data:
                    priority_predictions = data["priority_predictions"]
                    
                    for pred in priority_predictions[:20]:  # 상위 20개
                        if "numbers" in pred:
                            combinations.append({
                                "numbers": pred["numbers"],
                                "total_score": pred.get("integrated_score", 0.5),
                                "confidence": pred.get("integrated_score", 0.5),
                                "strategy": "three_digit_priority",
                                "source_stage": "run1"
                            })
                    break
                elif "prediction_results" in data:
                    pred_results = data["prediction_results"]
                    if "priority_predictions" in pred_results:
                        priority_predictions = pred_results["priority_predictions"]
                        
                        for pred in priority_predictions[:20]:
                            if "numbers" in pred:
                                combinations.append({
                                    "numbers": pred["numbers"],
                                    "total_score": pred.get("integrated_score", 0.5),
                                    "confidence": pred.get("integrated_score", 0.5),
                                    "strategy": "three_digit_priority",
                                    "source_stage": "run1"
                                })
                        break
        
        return combinations[:20]

    def _get_balanced_combinations(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """균형잡힌 조합 생성 (구간별, 홀짝별 균형)"""
        combinations = []
        
        # 구간별 균형 조합 생성
        segments = {
            "low": list(range(1, 10)),      # 1-9
            "mid_low": list(range(10, 19)), # 10-18
            "mid": list(range(19, 28)),     # 19-27
            "mid_high": list(range(28, 37)), # 28-36
            "high": list(range(37, 46))     # 37-45
        }
        
        # 각 구간에서 번호 선택하여 조합 생성
        for low_count in range(1, 3):      # 낮은 구간에서 1-2개
            for mid_count in range(2, 4):  # 중간 구간들에서 2-3개
                for high_count in range(1, 3):  # 높은 구간에서 1-2개
                    if low_count + mid_count + high_count == 6:
                        # 각 구간에서 번호 선택
                        selected_numbers = []
                        selected_numbers.extend(np.random.choice(segments["low"], low_count, replace=False))
                        selected_numbers.extend(np.random.choice(segments["mid_low"] + segments["mid"] + segments["mid_high"], mid_count, replace=False))
                        selected_numbers.extend(np.random.choice(segments["high"], high_count, replace=False))
                        
                        # 홀짝 균형 체크
                        odd_count = sum(1 for num in selected_numbers if num % 2 == 1)
                        even_count = 6 - odd_count
                        
                        # 균형 점수 계산
                        balance_score = 1.0 - abs(odd_count - even_count) / 6.0
                        balance_score += 0.1 if 2 <= odd_count <= 4 else 0.0  # 홀짝 균형 보너스
                        
                        combinations.append({
                            "numbers": sorted(selected_numbers),
                            "total_score": balance_score,
                            "confidence": balance_score,
                            "strategy": "balanced",
                            "source_stage": "generated",
                            "balance_metrics": {
                                "odd_count": odd_count,
                                "even_count": even_count,
                                "segment_distribution": [low_count, mid_count, high_count]
                            }
                        })
                        
                        if len(combinations) >= 15:
                            break
                if len(combinations) >= 15:
                    break
            if len(combinations) >= 15:
                break
        
        return combinations[:15]

    def _apply_strategy_weights(self, strategy_combinations: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """전략별 가중치를 적용하여 조합들을 통합합니다."""
        weighted_combinations = []
        
        for strategy, combinations in strategy_combinations.items():
            weight = self.recommendation_config.strategy_weights.get(strategy, 0.1)
            
            for combo in combinations:
                # 가중치 적용
                weighted_score = combo["total_score"] * weight
                
                # 추가 메타데이터
                combo_with_weight = combo.copy()
                combo_with_weight.update({
                    "weighted_score": weighted_score,
                    "strategy_weight": weight,
                    "original_score": combo["total_score"]
                })
                
                weighted_combinations.append(combo_with_weight)
        
        # 가중 점수 기준 정렬
        weighted_combinations.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        return weighted_combinations

    def _ensure_diversity(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거 및 다양성을 보장합니다."""
        diverse_combinations = []
        seen_combinations = set()
        
        for combo in combinations:
            # 번호 조합을 튜플로 변환하여 중복 체크
            combo_tuple = tuple(sorted(combo["numbers"]))
            
            if combo_tuple not in seen_combinations:
                seen_combinations.add(combo_tuple)
                
                # 다양성 메트릭 계산
                diversity_metrics = self._calculate_diversity_metrics(combo["numbers"])
                combo["diversity_metrics"] = diversity_metrics
                combo["diversity_score"] = diversity_metrics["overall_diversity"]
                
                # 다양성 기준을 만족하는 조합만 추가
                if diversity_metrics.get("overall_diversity", 0.5) >= self.recommendation_config.diversity_threshold:
                    diverse_combinations.append(combo)
        
        return diverse_combinations

    def _calculate_diversity_metrics(self, numbers: List[int]) -> Dict[str, Any]:
        """조합의 다양성 메트릭을 계산합니다."""
        # 홀짝 분포
        odd_count = sum(1 for num in numbers if num % 2 == 1)
        even_count = 6 - odd_count
        odd_even_ratio = f"{odd_count}:{even_count}"
        
        # 구간별 분포
        segments = [0, 0, 0, 0, 0]  # 1-9, 10-18, 19-27, 28-36, 37-45
        for num in numbers:
            if 1 <= num <= 9:
                segments[0] += 1
            elif 10 <= num <= 18:
                segments[1] += 1
            elif 19 <= num <= 27:
                segments[2] += 1
            elif 28 <= num <= 36:
                segments[3] += 1
            elif 37 <= num <= 45:
                segments[4] += 1
        
        # 연속성 체크
        consecutive_count = 0
        sorted_numbers = sorted(numbers)
        for i in range(len(sorted_numbers) - 1):
            if sorted_numbers[i+1] - sorted_numbers[i] == 1:
                consecutive_count += 1
        
        # 전체 다양성 점수 계산
        odd_even_balance = 1.0 - abs(odd_count - even_count) / 6.0
        segment_balance = 1.0 - np.std(segments) / 2.0
        consecutive_penalty = max(0, 1.0 - consecutive_count * 0.2)
        
        overall_diversity = (odd_even_balance + segment_balance + consecutive_penalty) / 3.0
        
        return {
            "odd_even_ratio": odd_even_ratio,
            "segment_distribution": segments,
            "consecutive_count": consecutive_count,
            "odd_even_balance": odd_even_balance,
            "segment_balance": segment_balance,
            "consecutive_penalty": consecutive_penalty,
            "overall_diversity": overall_diversity
        }

    def _apply_quality_validation(self, combinations: List[Dict[str, Any]], all_results: Dict[str, Any]) -> List[FinalRecommendation]:
        """품질 검증 및 최종 순위화를 적용합니다."""
        validated_recommendations = []
        
        for idx, combo in enumerate(combinations):
            # 기본 품질 기준 체크
            if combo.get("confidence", 0.0) < self.recommendation_config.confidence_threshold:
                continue
            
            # 리스크 점수 계산
            risk_score = self._calculate_risk_score(combo, all_results)
            if risk_score > self.recommendation_config.risk_threshold:
                continue
            
            # 트렌드 강도 계산
            trend_strength = self._calculate_trend_strength(combo, all_results)
            
            # 단계별 기여도 계산
            stage_contributions = self._calculate_stage_contributions(combo, all_results)
            
            # 최종 추천 객체 생성
            final_recommendation = FinalRecommendation(
                combination_id=idx + 1,
                numbers=sorted(combo["numbers"]),
                total_score=combo["weighted_score"],
                confidence=combo["confidence"],
                strategy=combo["strategy"],
                risk_score=risk_score,
                trend_strength=trend_strength,
                diversity_metrics=combo.get("diversity_metrics", {}),
                stage_contributions=stage_contributions
            )
            
            validated_recommendations.append(final_recommendation)
        
        # 최종 점수 기준 정렬
        validated_recommendations.sort(key=lambda x: x.total_score, reverse=True)
        
        return validated_recommendations

    def _calculate_risk_score(self, combo: Dict[str, Any], all_results: Dict[str, Any]) -> float:
        """조합의 리스크 점수를 계산합니다."""
        # 기본 리스크 점수
        base_risk = combo.get("risk_score", 0.2)
        
        # 과도한 연속성 패널티
        consecutive_penalty = combo.get("diversity_metrics", {}).get("consecutive_count", 0) * 0.1
        
        # 구간 불균형 패널티
        segment_distribution = combo.get("diversity_metrics", {}).get("segment_distribution", [1,1,1,1,1])
        segment_imbalance = np.std(segment_distribution) * 0.05
        
        # 최종 리스크 점수
        total_risk = base_risk + consecutive_penalty + segment_imbalance
        
        return min(total_risk, 1.0)

    def _calculate_trend_strength(self, combo: Dict[str, Any], all_results: Dict[str, Any]) -> float:
        """조합의 트렌드 강도를 계산합니다."""
        trend_strength = combo.get("trend_strength", 0.5)
        
        # run4 트렌드 보정 결과와 비교
        for name, data in all_results["run4_trend"].items():
            if "trend_correction_scores" in data:
                trend_scores = data["trend_correction_scores"]
                combo_trend_score = sum(trend_scores.get(str(num), 0) for num in combo["numbers"]) / 6
                trend_strength = max(trend_strength, combo_trend_score)
                break
        
        return trend_strength

    def _calculate_stage_contributions(self, combo: Dict[str, Any], all_results: Dict[str, Any]) -> Dict[str, float]:
        """각 단계별 기여도를 계산합니다."""
        contributions = {
            "run1_analysis": 0.15,
            "run2_predictions": 0.25,
            "run3_anomaly": 0.15,
            "run4_trend": 0.20,
            "run5_risk": 0.10,
            "run6_integration": 0.15
        }
        
        # 전략에 따른 기여도 조정
        strategy = combo.get("strategy", "balanced")
        if strategy == "score_integrated":
            contributions["run6_integration"] = 0.4
            contributions["run2_predictions"] = 0.3
        elif strategy == "risk_filtered":
            contributions["run5_risk"] = 0.4
            contributions["run3_anomaly"] = 0.3
        elif strategy == "trend_corrected":
            contributions["run4_trend"] = 0.5
        elif strategy == "ml_predictions":
            contributions["run2_predictions"] = 0.5
        elif strategy == "three_digit_priority":
            contributions["run1_analysis"] = 0.5
        
        # 정규화
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        
        return contributions

    def save_predictions_to_csv(self, recommendations: List[FinalRecommendation]) -> str:
        """
        추천 결과를 CSV 파일로 저장합니다.
        
        Args:
            recommendations: 최종 추천 목록
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            self.logger.info(f"📄 CSV 저장 시작: {self.csv_path}")
            
            with self.performance_monitor.track("save_csv"):
                # CSV 헤더 정의
                headers = [
                    "draw_no", "combination_id", "numbers", "total_score", "confidence",
                    "strategy", "risk_score", "trend_strength", "odd_even_ratio",
                    "segment_distribution", "consecutive_count", "stage_run1", "stage_run2",
                    "stage_run3", "stage_run4", "stage_run5", "stage_run6"
                ]
                
                # CSV 파일 작성
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # 헤더 작성
                    writer.writerow(headers)
                    
                    # 데이터 작성
                    for rec in recommendations:
                        # 번호를 쉼표로 구분된 문자열로 변환
                        numbers_str = ",".join(map(str, rec.numbers))
                        
                        # 구간 분포를 문자열로 변환
                        segment_dist_str = ",".join(map(str, rec.diversity_metrics.get("segment_distribution", [0,0,0,0,0])))
                        
                        row = [
                            "NEXT",  # draw_no
                            rec.combination_id,
                            f'"{numbers_str}"',  # 쉼표가 포함된 문자열이므로 따옴표로 감싸기
                            f"{rec.total_score:.6f}",
                            f"{rec.confidence:.6f}",
                            rec.strategy,
                            f"{rec.risk_score:.6f}",
                            f"{rec.trend_strength:.6f}",
                            rec.diversity_metrics.get("odd_even_ratio", "3:3"),
                            f'"{segment_dist_str}"',
                            rec.diversity_metrics.get("consecutive_count", 0),
                            f"{rec.stage_contributions.get('run1_analysis', 0):.4f}",
                            f"{rec.stage_contributions.get('run2_predictions', 0):.4f}",
                            f"{rec.stage_contributions.get('run3_anomaly', 0):.4f}",
                            f"{rec.stage_contributions.get('run4_trend', 0):.4f}",
                            f"{rec.stage_contributions.get('run5_risk', 0):.4f}",
                            f"{rec.stage_contributions.get('run6_integration', 0):.4f}"
                        ]
                        
                        writer.writerow(row)
            
            self.logger.info(f"✅ CSV 저장 완료: {self.csv_path}")
            return str(self.csv_path)
            
        except Exception as e:
            self.logger.error(f"CSV 저장 실패: {e}")
            raise

    def generate_summary_report(self, recommendations: List[FinalRecommendation]) -> Dict[str, Any]:
        """추천 결과 요약 리포트를 생성합니다."""
        if not recommendations:
            return {"error": "추천 결과가 없습니다"}
        
        # 전략별 분포
        strategy_counts = Counter(rec.strategy for rec in recommendations)
        
        # 점수 통계
        scores = [rec.total_score for rec in recommendations]
        confidences = [rec.confidence for rec in recommendations]
        risk_scores = [rec.risk_score for rec in recommendations]
        
        # 다양성 통계
        odd_even_distributions = [rec.diversity_metrics.get("odd_even_ratio", "3:3") for rec in recommendations]
        odd_even_counts = Counter(odd_even_distributions)
        
        summary_report = {
            "generation_metadata": {
                "total_recommendations": len(recommendations),
                "avg_confidence": np.mean(confidences),
                "avg_total_score": np.mean(scores),
                "avg_risk_score": np.mean(risk_scores),
                "score_range": {
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "std": np.std(scores)
                }
            },
            "strategy_distribution": dict(strategy_counts),
            "diversity_analysis": {
                "odd_even_distribution": dict(odd_even_counts),
                "avg_consecutive_count": np.mean([
                    rec.diversity_metrics.get("consecutive_count", 0) 
                    for rec in recommendations
                ])
            },
            "stage_contributions_avg": {
                f"run{i}_weight": np.mean([
                    rec.stage_contributions.get(f"run{i}_{'analysis' if i == 1 else 'predictions' if i == 2 else 'anomaly' if i == 3 else 'trend' if i == 4 else 'risk' if i == 5 else 'integration'}", 0)
                    for rec in recommendations
                ]) for i in range(1, 7)
            },
            "top_recommendations_preview": [
                {
                    "combination_id": rec.combination_id,
                    "numbers": rec.numbers,
                    "total_score": rec.total_score,
                    "confidence": rec.confidence,
                    "strategy": rec.strategy
                }
                for rec in recommendations[:10]
            ]
        }
        
        return summary_report

    def run_full_recommendation_pipeline(self) -> Dict[str, Any]:
        """전체 추천 파이프라인을 실행합니다."""
        self.logger.info("🚀 최종 추천 시스템 시작")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. 이전 단계 결과 로드
            self.logger.info("📊 1단계: 이전 6단계 결과 로드")
            all_results = self.load_all_stage_results()
            
            # 2. 최종 추천 조합 생성
            self.logger.info("🎯 2단계: 최종 추천 조합 생성")
            final_recommendations = self.generate_final_recommendations(all_results)
            
            # 3. CSV 파일로 저장
            self.logger.info("📄 3단계: CSV 파일 저장")
            csv_path = self.save_predictions_to_csv(final_recommendations)
            
            # 4. 요약 리포트 생성
            self.logger.info("📋 4단계: 요약 리포트 생성")
            summary_report = self.generate_summary_report(final_recommendations)
            
            # 5. 성능 통계 수집
            end_time = time.time()
            execution_time = end_time - start_time
            
            performance_summary = {
                "execution_time_seconds": execution_time,
                "performance_monitor": self.performance_monitor.get_performance_summary(),
                "memory_manager_stats": self.memory_manager.get_simple_stats(),
            }
            
            # 6. 최종 결과 구성
            final_result = {
                "final_recommendations": [asdict(rec) for rec in final_recommendations],
                "summary_report": summary_report,
                "csv_path": csv_path,
                "performance_summary": performance_summary,
                "execution_timestamp": datetime.now().isoformat(),
                "system_config": {
                    "recommendation_config": asdict(self.recommendation_config),
                    "total_stage_files": all_results["metadata"]["total_files"]
                }
            }
            
            self.logger.info(f"✅ 최종 추천 시스템 완료 (소요시간: {execution_time:.2f}초)")
            return final_result
            
        except Exception as e:
            self.logger.error(f"최종 추천 시스템 실행 실패: {e}")
            raise


def main():
    """메인 실행 함수"""
    # 1. 의존성 설정
    configure_dependencies()
    
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("🚀 7단계: 최종 추천 시스템 시작")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 추천 엔진 생성
        engine = Recommendation()
        
        # 전체 파이프라인 실행
        final_results = engine.run_full_recommendation_pipeline()

        total_time = time.time() - start_time
        logger.info(f"✅ 최종 추천 완료! 총 실행 시간: {total_time:.2f}초")
        logger.info(f"📁 최종 추천 파일: {final_results.get('csv_path')}")
        
    except Exception as e:
        logger.error(f"❌ 최종 추천 실패: {e}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    main()
