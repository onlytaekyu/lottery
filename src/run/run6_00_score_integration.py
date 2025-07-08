"""
DAEBAK_AI 로또 예측 시스템 - 6단계: 점수 통합 시스템
Enhanced Meta Weight Layer + ROI 기반 가중치로 최종 점수 통합
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass, asdict

# 3. 프로젝트 내부 (리팩토링된 의존성 관리)
from ..utils.dependency_injection import configure_dependencies, resolve
from ..utils.unified_logging import get_logger
from ..utils.unified_memory_manager import UnifiedMemoryManager
from ..utils.unified_performance_engine import AutoPerformanceMonitor
from ..utils.unified_config import Config
from ..pipeline.enhanced_meta_weight_layer import EnhancedMetaWeightLayer, MetaWeightConfig
from ..models.unified_model_manager import UnifiedModelManager
from ..utils.config_validator import ConfigValidator
from ..pipeline.model_performance_benchmark import ModelPerformanceBenchmark
from ..models.adaptive_weight_system import AdaptiveWeightSystem


@dataclass
class ScoreIntegrationConfig:
    """점수 통합 설정"""
    # 가중치 설정
    stage_weights: Dict[str, float] = None
    roi_weight: float = 0.3
    diversity_weight: float = 0.2
    performance_weight: float = 0.5
    
    # Meta Weight Layer 설정
    meta_config: MetaWeightConfig = None
    
    # 앙상블 설정
    ensemble_size: int = 100
    top_k_candidates: int = 50
    confidence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.stage_weights is None:
            self.stage_weights = {
                "run1_analysis": 0.25,      # 패턴 분석
                "run2_predictions": 0.25,   # LightGBM 
                "run3_anomaly": 0.2,        # AutoEncoder
                "run4_trend": 0.2,          # TCN
                "run5_risk": 0.1            # RandomForest
            }
        
        if self.meta_config is None:
            self.meta_config = MetaWeightConfig(
                num_models=5,  # 5단계 결과
                adaptation_rate=0.01,
                momentum=0.9,
                confidence_threshold=0.7,
                roi_weight=self.roi_weight,
                diversity_weight=self.diversity_weight,
                performance_weight=self.performance_weight
            )


class ScoreIntegrationEngine:
    """점수 통합 엔진"""
    
    def __init__(self):
        """
        점수 통합 엔진 초기화 (의존성 주입 사용)
        """
        self.logger = get_logger(__name__)

        # --- 의존성 해결 ---
        self.config_manager: Config = resolve(Config)
        self.config = self.config_manager.get_config("main")
        self.paths = self.config_manager.get_paths()
        self.memory_manager: UnifiedMemoryManager = resolve(UnifiedMemoryManager)
        self.performance_monitor: AutoPerformanceMonitor = resolve(AutoPerformanceMonitor)
        self.validator: ConfigValidator = resolve(ConfigValidator)
        # --------------------
        
        # 점수 통합 설정
        self.integration_config = ScoreIntegrationConfig()
        
        # Enhanced Meta Weight Layer 초기화
        self.meta_weight_layer = EnhancedMetaWeightLayer(self.config)
        
        # 통합 모델 매니저
        self.model_manager = resolve(UnifiedModelManager)
        
        # 결과 저장 경로
        self.result_dir = Path(self.paths.result_dir) / "score_integration"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 캐시 디렉토리
        self.cache_dir = Path(self.paths.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.adaptive_weight_system = AdaptiveWeightSystem(self.config)
        self.benchmark = ModelPerformanceBenchmark(self.config)
        
        self.logger.info("✅ 점수 통합 엔진 초기화 완료")

    def load_all_stage_results(self) -> Dict[str, Any]:
        """
        이전 5단계의 모든 결과를 로드합니다.
        
        Returns:
            Dict[str, Any]: 통합된 모든 단계 결과
        """
        try:
            self.logger.info("📊 이전 5단계 결과 로드 시작...")
            
            with self.performance_monitor.track("load_all_stage_results"):
                results = {
                    "run1_analysis": {},
                    "run2_predictions": {},
                    "run3_anomaly": {},
                    "run4_trend": {},
                    "run5_risk": {},
                    "metadata": {}
                }
                
                # run1 분석 결과 로드
                analysis_dir = Path(self.paths.result_dir) / "analysis"
                if analysis_dir.exists():
                    for file_path in analysis_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run1_analysis"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"run1 파일 로드 실패 {file_path}: {e}")
                
                # run2 예측 결과 로드  
                predictions_dir = Path(self.paths.predictions_dir)
                if predictions_dir.exists():
                    for file_path in predictions_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run2_predictions"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"run2 파일 로드 실패 {file_path}: {e}")
                
                # run3 이상감지 결과 로드
                anomaly_dir = Path(self.paths.result_dir) / "anomaly_detection"
                if anomaly_dir.exists():
                    for file_path in anomaly_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run3_anomaly"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"run3 파일 로드 실패 {file_path}: {e}")
                
                # run4 트렌드 보정 결과 로드
                trend_dir = Path(self.paths.result_dir) / "trend_correction"
                if trend_dir.exists():
                    for file_path in trend_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run4_trend"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"run4 파일 로드 실패 {file_path}: {e}")
                
                # run5 리스크 필터 결과 로드
                risk_dir = Path(self.paths.result_dir) / "risk_filter"
                if risk_dir.exists():
                    for file_path in risk_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run5_risk"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"run5 파일 로드 실패 {file_path}: {e}")
                
                # 메타데이터 추가
                results["metadata"] = {
                    "load_timestamp": datetime.now().isoformat(),
                    "run1_files": len(results["run1_analysis"]),
                    "run2_files": len(results["run2_predictions"]),
                    "run3_files": len(results["run3_anomaly"]),
                    "run4_files": len(results["run4_trend"]),
                    "run5_files": len(results["run5_risk"]),
                    "total_files": sum([
                        len(results["run1_analysis"]),
                        len(results["run2_predictions"]),
                        len(results["run3_anomaly"]),
                        len(results["run4_trend"]),
                        len(results["run5_risk"])
                    ])
                }
            
            self.logger.info(f"✅ 전체 단계 결과 로드 완료: "
                           f"run1({len(results['run1_analysis'])}), "
                           f"run2({len(results['run2_predictions'])}), "
                           f"run3({len(results['run3_anomaly'])}), "
                           f"run4({len(results['run4_trend'])}), "
                           f"run5({len(results['run5_risk'])})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"단계 결과 로드 실패: {e}")
            raise

    def integrate_stage_scores(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        5단계 결과를 통합하여 최종 점수를 계산합니다.
        
        Args:
            all_results: 모든 단계 결과
            
        Returns:
            Dict[str, Any]: 통합된 점수 정보
        """
        try:
            self.logger.info("🔄 단계별 점수 통합 시작...")
            
            with self.performance_monitor.track("integrate_stage_scores"):
                # 1. 각 단계별 점수 추출
                stage_scores = self._extract_stage_scores(all_results)
                
                # 2. Meta Weight Layer를 위한 모델 예측 매트릭스 구성
                model_predictions = self._build_model_predictions_matrix(stage_scores)
                
                # 3. ROI 기반 가중치 계산
                roi_weights = self._calculate_roi_weights(all_results)
                
                # 4. Enhanced Meta Weight Layer로 앙상블 예측
                ensemble_prediction, meta_metadata = self.meta_weight_layer.compute_ensemble_prediction(
                    model_predictions, 
                    historical_roi=roi_weights
                )
                
                # 5. 최종 통합 점수 계산
                integrated_scores = self._compute_final_scores(
                    ensemble_prediction, stage_scores, meta_metadata
                )
                
                # 6. 다양성 및 성능 기반 조정
                adjusted_scores = self._apply_diversity_performance_adjustment(
                    integrated_scores, all_results
                )
                
                # 7. 최종 순위화 및 후보 조합 생성
                final_recommendations = self._generate_final_recommendations(
                    adjusted_scores, all_results
                )
                
                integration_result = {
                    "stage_scores": stage_scores,
                    "model_predictions": {k: v.tolist() for k, v in model_predictions.items()},
                    "roi_weights": roi_weights,
                    "ensemble_prediction": ensemble_prediction.tolist(),
                    "integrated_scores": integrated_scores,
                    "adjusted_scores": adjusted_scores,
                    "final_recommendations": final_recommendations,
                    "meta_metadata": meta_metadata,
                    "integration_config": asdict(self.integration_config)
                }
                
            self.logger.info("✅ 단계별 점수 통합 완료")
            return integration_result
            
        except Exception as e:
            self.logger.error(f"점수 통합 실패: {e}")
            raise

    def _extract_stage_scores(self, all_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """각 단계별 점수를 추출합니다."""
        stage_scores = {}
        
        # run1 분석 점수 추출
        stage_scores["run1_analysis"] = {}
        for name, data in all_results["run1_analysis"].items():
            if "unified_analysis" in name and "comprehensive_scores" in data:
                scores = data["comprehensive_scores"]
                for num in range(1, 46):
                    stage_scores["run1_analysis"][str(num)] = scores.get(str(num), 0.0)
                break
        
        # run2 ML 예측 점수 추출
        stage_scores["run2_predictions"] = {}
        for name, data in all_results["run2_predictions"].items():
            if "ml_predictions" in name and "predictions" in data:
                predictions = data["predictions"]
                for num in range(1, 46):
                    stage_scores["run2_predictions"][str(num)] = predictions.get(str(num), 0.0)
                break
        
        # run3 이상감지 점수 추출 (이상도가 낮을수록 좋음)
        stage_scores["run3_anomaly"] = {}
        for name, data in all_results["run3_anomaly"].items():
            if "anomaly_detection" in name and "anomaly_scores" in data:
                anomaly_scores = data["anomaly_scores"]
                for num in range(1, 46):
                    anomaly_score = anomaly_scores.get(str(num), 0.5)
                    stage_scores["run3_anomaly"][str(num)] = 1.0 - anomaly_score  # 역변환
                break
        
        # run4 트렌드 보정 점수 추출
        stage_scores["run4_trend"] = {}
        for name, data in all_results["run4_trend"].items():
            if "trend_correction_scores" in data:
                trend_scores = data["trend_correction_scores"]
                for num in range(1, 46):
                    stage_scores["run4_trend"][str(num)] = trend_scores.get(str(num), 0.0)
                break
        
        # run5 리스크 필터 점수 추출 (필터링 후 남은 조합들의 점수)
        stage_scores["run5_risk"] = {}
        for name, data in all_results["run5_risk"].items():
            if "risk_filtered_combinations" in data:
                # 리스크 필터에서는 조합별 점수가 있으므로 번호별로 집계
                combinations = data["risk_filtered_combinations"]
                number_scores = defaultdict(list)
                
                for combo_data in combinations.values():
                    if "numbers" in combo_data and "confidence" in combo_data:
                        numbers = combo_data["numbers"]
                        confidence = combo_data["confidence"]
                        for num in numbers:
                            number_scores[num].append(confidence)
                
                # 번호별 평균 점수 계산
                for num in range(1, 46):
                    if num in number_scores:
                        stage_scores["run5_risk"][str(num)] = float(np.mean(number_scores[num]))
                    else:
                        stage_scores["run5_risk"][str(num)] = 0.0
                break
        
        # 누락된 단계 기본값 설정
        for stage in ["run1_analysis", "run2_predictions", "run3_anomaly", "run4_trend", "run5_risk"]:
            if stage not in stage_scores or not stage_scores[stage]:
                stage_scores[stage] = {str(num): 0.0 for num in range(1, 46)}
        
        return stage_scores

    def _build_model_predictions_matrix(self, stage_scores: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
        """Meta Weight Layer를 위한 모델 예측 매트릭스를 구성합니다."""
        model_predictions = {}
        
        for stage_name, scores in stage_scores.items():
            # 번호별 점수를 numpy 배열로 변환
            score_array = np.array([scores[str(num)] for num in range(1, 46)])
            
            # 정규화 (0-1 범위)
            if score_array.max() > 0:
                score_array = score_array / score_array.max()
            
            model_predictions[stage_name] = score_array
        
        return model_predictions

    def _calculate_roi_weights(self, all_results: Dict[str, Any]) -> Dict[str, List[float]]:
        """ROI 기반 가중치를 계산합니다."""
        roi_weights = {}
        
        # 각 단계별 성능 기반 ROI 추정
        for stage in ["run1_analysis", "run2_predictions", "run3_anomaly", "run4_trend", "run5_risk"]:
            stage_data = all_results.get(stage, {})
            
            # 성능 메트릭이 있는 경우 활용
            performance_roi = []
            
            for name, data in stage_data.items():
                if "performance_summary" in data:
                    perf_summary = data["performance_summary"]
                    # 실행 시간 기반 효율성 계산
                    exec_time = perf_summary.get("execution_time_seconds", 1.0)
                    efficiency = 1.0 / (1.0 + exec_time / 60.0)  # 분 단위로 정규화
                    performance_roi.append(efficiency)
                elif "execution_stats" in data:
                    exec_stats = data["execution_stats"]
                    total_time = exec_stats.get("total_time", 1.0)
                    efficiency = 1.0 / (1.0 + total_time / 60.0)
                    performance_roi.append(efficiency)
                elif "performance_metrics" in data:
                    # 분석 성능 메트릭 활용
                    perf_metrics = data["performance_metrics"]
                    if isinstance(perf_metrics, dict) and "execution_time" in perf_metrics:
                        exec_time = perf_metrics["execution_time"]
                        efficiency = 1.0 / (1.0 + exec_time / 60.0)
                        performance_roi.append(efficiency)
            
            # 기본 ROI 값 설정
            if not performance_roi:
                performance_roi = [0.5, 0.6, 0.7, 0.8, 0.9]  # 기본 성능 곡선
            
            roi_weights[stage] = performance_roi
        
        return roi_weights

    def _compute_final_scores(self, ensemble_prediction: np.ndarray, 
                            stage_scores: Dict[str, Dict[str, float]], 
                            meta_metadata: Dict[str, Any]) -> Dict[str, float]:
        """최종 통합 점수를 계산합니다."""
        final_scores = {}
        
        # Meta Weight Layer에서 나온 앙상블 예측을 기본으로 사용
        for i in range(45):
            num = str(i + 1)
            
            # 앙상블 예측 점수
            ensemble_score = float(ensemble_prediction[i])
            
            # 단계별 가중 점수 추가
            weighted_stage_score = 0.0
            for stage_name, weight in self.integration_config.stage_weights.items():
                stage_score = stage_scores.get(stage_name, {}).get(num, 0.0)
                weighted_stage_score += stage_score * weight
            
            # 최종 점수 = 앙상블 점수 70% + 가중 단계 점수 30%
            final_score = ensemble_score * 0.7 + weighted_stage_score * 0.3
            
            final_scores[num] = float(final_score)
        
        return final_scores

    def _apply_diversity_performance_adjustment(self, integrated_scores: Dict[str, float], 
                                              all_results: Dict[str, Any]) -> Dict[str, float]:
        """다양성 및 성능 기반 조정을 적용합니다."""
        adjusted_scores = integrated_scores.copy()
        
        # 1. 다양성 보장 조정
        # 홀짝 균형 조정
        odd_numbers = [num for num in range(1, 46, 2)]
        even_numbers = [num for num in range(2, 46, 2)]
        
        odd_avg = np.mean([adjusted_scores[str(num)] for num in odd_numbers])
        even_avg = np.mean([adjusted_scores[str(num)] for num in even_numbers])
        
        # 불균형 시 조정
        if abs(odd_avg - even_avg) > 0.1:
            adjustment_factor = 0.05
            if odd_avg > even_avg:
                for num in even_numbers:
                    adjusted_scores[str(num)] += adjustment_factor
            else:
                for num in odd_numbers:
                    adjusted_scores[str(num)] += adjustment_factor
        
        # 2. 구간별 균형 조정
        segments = [
            list(range(1, 10)),    # 1-9
            list(range(10, 19)),   # 10-18
            list(range(19, 28)),   # 19-27
            list(range(28, 37)),   # 28-36
            list(range(37, 46))    # 37-45
        ]
        
        segment_avgs = []
        for segment in segments:
            segment_avg = np.mean([adjusted_scores[str(num)] for num in segment])
            segment_avgs.append(segment_avg)
        
        # 구간별 균형 조정
        overall_avg = np.mean(segment_avgs)
        for i, segment in enumerate(segments):
            if segment_avgs[i] < overall_avg * 0.8:  # 너무 낮은 구간 상향 조정
                adjustment = (overall_avg * 0.8 - segment_avgs[i]) * 0.5
                for num in segment:
                    adjusted_scores[str(num)] += adjustment
        
        return adjusted_scores

    def _generate_final_recommendations(self, adjusted_scores: Dict[str, float], 
                                      all_results: Dict[str, Any]) -> Dict[str, Any]:
        """최종 추천 조합을 생성합니다."""
        try:
            # 1. 상위 점수 번호들 선별
            sorted_numbers = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
            top_numbers = [int(num) for num, score in sorted_numbers[:self.integration_config.top_k_candidates]]
            
            # 2. 다양한 전략으로 조합 생성
            recommendations = {
                "top_score_combinations": self._generate_top_score_combinations(top_numbers, adjusted_scores),
                "balanced_combinations": self._generate_balanced_combinations(adjusted_scores),
                "diversity_combinations": self._generate_diversity_combinations(adjusted_scores),
                "meta_weight_combinations": self._generate_meta_weight_combinations(adjusted_scores, all_results)
            }
            
            # 3. 각 전략별 최고 조합 선정
            final_recommendations = {}
            for strategy, combos in recommendations.items():
                if combos:
                    final_recommendations[strategy] = combos[:5]  # 상위 5개
            
            # 4. 통합 최종 추천 (모든 전략 고려)
            all_combinations = []
            for strategy_combos in recommendations.values():
                all_combinations.extend(strategy_combos)
            
            # 중복 제거 및 점수 기준 정렬
            unique_combinations = {}
            for combo in all_combinations:
                combo_key = tuple(sorted(combo["numbers"]))
                if combo_key not in unique_combinations or combo["total_score"] > unique_combinations[combo_key]["total_score"]:
                    unique_combinations[combo_key] = combo
            
            final_top_combinations = sorted(
                unique_combinations.values(), 
                key=lambda x: x["total_score"], 
                reverse=True
            )[:self.integration_config.ensemble_size]
            
            return {
                "strategy_recommendations": final_recommendations,
                "final_top_combinations": final_top_combinations,
                "recommendation_summary": {
                    "total_combinations_generated": len(all_combinations),
                    "unique_combinations": len(unique_combinations),
                    "final_count": len(final_top_combinations),
                    "avg_confidence": np.mean([combo["confidence"] for combo in final_top_combinations]),
                    "top_score": final_top_combinations[0]["total_score"] if final_top_combinations else 0.0
                }
            }
            
        except Exception as e:
            self.logger.error(f"최종 추천 생성 실패: {e}")
            return {}

    def _generate_top_score_combinations(self, top_numbers: List[int], scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """최고 점수 기반 조합 생성"""
        from itertools import combinations
        
        combinations_list = []
        for combo in combinations(top_numbers[:20], 6):  # 상위 20개 중 6개 조합
            combo_list = list(combo)
            total_score = sum(scores[str(num)] for num in combo_list)
            avg_score = total_score / 6
            
            combinations_list.append({
                "numbers": combo_list,
                "total_score": total_score,
                "avg_score": avg_score,
                "confidence": min(avg_score, 1.0),
                "strategy": "top_score"
            })
        
        return sorted(combinations_list, key=lambda x: x["total_score"], reverse=True)[:20]

    def _generate_balanced_combinations(self, scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """균형잡힌 조합 생성 (구간별 균형)"""
        from itertools import combinations
        
        # 구간별 상위 번호 선정
        segments = {
            "low": list(range(1, 10)),      # 1-9
            "mid_low": list(range(10, 19)), # 10-18  
            "mid": list(range(19, 28)),     # 19-27
            "mid_high": list(range(28, 37)), # 28-36
            "high": list(range(37, 46))     # 37-45
        }
        
        segment_tops = {}
        for segment_name, numbers in segments.items():
            segment_scores = [(num, scores[str(num)]) for num in numbers]
            segment_scores.sort(key=lambda x: x[1], reverse=True)
            segment_tops[segment_name] = [num for num, score in segment_scores[:4]]
        
        combinations_list = []
        
        # 각 구간에서 1-2개씩 선택하여 조합 생성
        for combo in combinations(
            segment_tops["low"][:2] + segment_tops["mid_low"][:2] + 
            segment_tops["mid"][:2] + segment_tops["mid_high"][:2] + 
            segment_tops["high"][:2], 6
        ):
            combo_list = list(combo)
            
            # 구간별 분포 확인
            segment_count = {name: 0 for name in segments}
            for num in combo_list:
                for segment_name, segment_nums in segments.items():
                    if num in segment_nums:
                        segment_count[segment_name] += 1
                        break
            
            # 균형 점수 계산 (각 구간에 최소 1개씩 있으면 보너스)
            balance_bonus = 0.1 if all(count > 0 for count in segment_count.values()) else 0.0
            
            total_score = sum(scores[str(num)] for num in combo_list) + balance_bonus
            avg_score = total_score / 6
            
            combinations_list.append({
                "numbers": combo_list,
                "total_score": total_score,
                "avg_score": avg_score,
                "confidence": min(avg_score, 1.0),
                "strategy": "balanced",
                "segment_distribution": segment_count
            })
        
        return sorted(combinations_list, key=lambda x: x["total_score"], reverse=True)[:15]

    def _generate_diversity_combinations(self, scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """다양성 기반 조합 생성"""
        from itertools import combinations
        
        # 홀수/짝수 분류
        odd_numbers = [(num, scores[str(num)]) for num in range(1, 46, 2)]
        even_numbers = [(num, scores[str(num)]) for num in range(2, 46, 2)]
        
        odd_numbers.sort(key=lambda x: x[1], reverse=True)
        even_numbers.sort(key=lambda x: x[1], reverse=True)
        
        combinations_list = []
        
        # 홀짝 균형 조합 (3:3 또는 4:2, 2:4)
        for odd_count in [2, 3, 4]:
            even_count = 6 - odd_count
            
            if even_count < 2 or even_count > 4:
                continue
                
            top_odds = [num for num, score in odd_numbers[:min(8, len(odd_numbers))]]
            top_evens = [num for num, score in even_numbers[:min(8, len(even_numbers))]]
            
            for odd_combo in combinations(top_odds, odd_count):
                for even_combo in combinations(top_evens, even_count):
                    combo_list = list(odd_combo) + list(even_combo)
                    combo_list.sort()
                    
                    # 다양성 보너스 계산
                    diversity_bonus = 0.05 if abs(odd_count - even_count) <= 1 else 0.0
                    
                    total_score = sum(scores[str(num)] for num in combo_list) + diversity_bonus
                    avg_score = total_score / 6
                    
                    combinations_list.append({
                        "numbers": combo_list,
                        "total_score": total_score,
                        "avg_score": avg_score,
                        "confidence": min(avg_score, 1.0),
                        "strategy": "diversity",
                        "odd_count": odd_count,
                        "even_count": even_count
                    })
        
        return sorted(combinations_list, key=lambda x: x["total_score"], reverse=True)[:15]

    def _generate_meta_weight_combinations(self, scores: Dict[str, float], all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Meta Weight Layer 기반 조합 생성"""
        from itertools import combinations
        
        # Meta Weight에서 계산된 가중치 활용
        meta_weights = self.meta_weight_layer.get_weight_analysis().get("current_weights", {})
        
        combinations_list = []
        
        # 각 단계별 상위 번호 활용하여 조합 생성
        stage_top_numbers = {}
        for stage_name in ["run1_analysis", "run2_predictions", "run3_anomaly", "run4_trend", "run5_risk"]:
            stage_scores = [(num, score) for num, score in scores.items()]
            stage_scores.sort(key=lambda x: x[1], reverse=True)
            stage_top_numbers[stage_name] = [int(num) for num, score in stage_scores[:10]]
        
        # Meta Weight 기반 번호 풀 구성
        weighted_numbers = set()
        for stage_name, numbers in stage_top_numbers.items():
            weight = meta_weights.get(stage_name, 0.2)
            # 가중치가 높은 단계에서 더 많은 번호 선택
            count = max(2, int(weight * 20))
            weighted_numbers.update(numbers[:count])
        
        # 조합 생성
        for combo in combinations(list(weighted_numbers)[:25], 6):
            combo_list = list(combo)
            
            # Meta Weight 보너스 계산
            meta_bonus = 0.0
            for num in combo_list:
                for stage_name, numbers in stage_top_numbers.items():
                    if num in numbers[:5]:  # 상위 5개에 있으면 보너스
                        weight = meta_weights.get(stage_name, 0.2)
                        meta_bonus += weight * 0.05
            
            total_score = sum(scores[str(num)] for num in combo_list) + meta_bonus
            avg_score = total_score / 6
            
            combinations_list.append({
                "numbers": combo_list,
                "total_score": total_score,
                "avg_score": avg_score,
                "confidence": min(avg_score, 1.0),
                "strategy": "meta_weight",
                "meta_bonus": meta_bonus
            })
        
        return sorted(combinations_list, key=lambda x: x["total_score"], reverse=True)[:15]

    def save_integration_results(self, integration_result: Dict[str, Any]) -> str:
        """통합 결과를 저장합니다."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.result_dir / f"score_integration_{timestamp}.json"
            
            # 저장할 데이터 준비
            save_data = {
                "timestamp": timestamp,
                "config": asdict(self.integration_config),
                "integration_result": integration_result,
                "system_info": {
                    "cuda_available": torch.cuda.is_available(),
                    "memory_manager_stats": self.memory_manager.get_stats(),
                }
            }
            
            # JSON 파일 저장
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"✅ 점수 통합 결과 저장 완료: {result_file}")
            return str(result_file)
            
        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
            raise

    def run_full_score_integration(self) -> Dict[str, Any]:
        """전체 점수 통합 파이프라인 실행"""
        self.logger.info("🚀 점수 통합 시스템 시작")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. 이전 단계 결과 로드
            self.logger.info("📊 1단계: 이전 단계 결과 로드")
            all_results = self.load_all_stage_results()
            
            # 2. 점수 통합 실행
            self.logger.info("🔄 2단계: 점수 통합 실행")
            integration_result = self.integrate_stage_scores(all_results)
            
            # 3. 결과 저장
            self.logger.info("💾 3단계: 결과 저장")
            saved_path = self.save_integration_results(integration_result)
            
            # 4. 성능 통계 수집
            end_time = time.time()
            execution_time = end_time - start_time
            
            performance_summary = {
                "execution_time_seconds": execution_time,
                "performance_monitor": self.performance_monitor.get_performance_summary(),
                "memory_manager_stats": self.memory_manager.get_stats(),
            }
            
            # 5. 최종 결과 구성
            final_result = {
                "integration_result": integration_result,
                "saved_path": saved_path,
                "performance_summary": performance_summary,
                "execution_timestamp": datetime.now().isoformat(),
                "system_config": {
                    "cuda_available": torch.cuda.is_available(),
                    "integration_config": asdict(self.integration_config)
                }
            }
            
            self.logger.info(f"✅ 점수 통합 시스템 완료 (소요시간: {execution_time:.2f}초)")
            return final_result
            
        except Exception as e:
            self.logger.error(f"점수 통합 시스템 실행 실패: {e}")
            raise


def main():
    """메인 실행 함수"""
    # 1. 로거 초기화
    logger = get_logger(__name__)

    # 2. 의존성 설정
    configure_dependencies()

    logger.info("=" * 80)
    logger.info("🚀 6단계: 점수 통합 시스템 시작")
    logger.info("=" * 80)
    
    start_time = time.time()

    try:
        # 점수 통합 엔진 생성 (설정 제거)
        engine = ScoreIntegrationEngine()
        
        # 전체 파이프라인 실행
        final_results = engine.run_full_score_integration()

        total_time = time.time() - start_time
        logger.info(f"✅ 점수 통합 완료! 총 실행 시간: {total_time:.2f}초")
        logger.info(f"📁 최종 결과 파일: {final_results.get('saved_path')}")
        
    except Exception as e:
        logger.error(f"❌ 점수 통합 실패: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    main()
