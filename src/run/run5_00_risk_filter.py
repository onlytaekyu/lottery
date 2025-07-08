#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
리스크 필터 시스템 (5단계) - RandomForest 기반 고위험 조합 제거

이전 4단계(run1~run4)의 모든 결과를 통합하여 RandomForest 모델로 
고위험 조합을 필터링하는 시스템입니다.

주요 기능:
- 이전 단계 결과 완전 통합 (run1~run4)
- RandomForest 기반 리스크 점수 계산
- GPU 최적화 및 메모리 풀 관리
- 고위험 조합 자동 제거
- 다양성 보장 메커니즘
"""

# 1. 표준 라이브러리
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import warnings
from itertools import combinations
from collections import defaultdict

warnings.filterwarnings("ignore")

# 2. 서드파티
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    f1_score
)

# 3. 프로젝트 내부 (리팩토링된 의존성 관리)
from ..utils.dependency_injection import configure_dependencies, resolve
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from ..utils.unified_performance_engine import UnifiedPerformanceEngine, AutoPerformanceMonitor
from ..utils.cuda_optimizers import CudaOptimizer
from ..utils.unified_memory_manager import UnifiedMemoryManager
from ..utils.unified_config import Config
from ..utils.data_loader import load_draw_history
from ..models.ml.random_forest_model import RandomForestModel
from ..utils.cache_manager import UnifiedCachePathManager

logger = get_logger(__name__)


class RiskFeatureExtractor:
    """
    리스크 특성 추출기
    
    조합별 다양한 리스크 특성을 추출하여 RandomForest 모델의 입력으로 사용합니다.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # 역사적 데이터 저장
        self.historical_data = []
        self.winning_numbers_set = set()
        
    def load_historical_data(self, historical_data: List[LotteryNumber]):
        """역사적 데이터 로드 및 전처리"""
        self.historical_data = historical_data
        for draw in historical_data:
            self.winning_numbers_set.update(draw.numbers)
            
    def extract_combination_features(self, combination: List[int]) -> np.ndarray:
        """
        단일 조합의 리스크 특성 추출
        
        Args:
            combination: 로또 번호 조합 [1-45]
            
        Returns:
            np.ndarray: 특성 벡터
        """
        features = []
        
        # 1. 기본 분포 특성
        features.extend(self._extract_distribution_features(combination))
        
        # 2. 패턴 특성
        features.extend(self._extract_pattern_features(combination))
        
        # 3. 역사적 유사도 특성
        features.extend(self._extract_historical_features(combination))
        
        # 4. 통계적 특성
        features.extend(self._extract_statistical_features(combination))
        
        # 5. 구간별 분포 특성
        features.extend(self._extract_section_features(combination))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_distribution_features(self, combination: List[int]) -> List[float]:
        """분포 관련 특성 추출"""
        features = []
        
        # 홀짝 비율
        odd_count = sum(1 for num in combination if num % 2 == 1)
        features.append(odd_count / len(combination))
        
        # 연속 번호 개수
        sorted_nums = sorted(combination)
        consecutive_count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive_count += 1
        features.append(consecutive_count / (len(combination) - 1))
        
        # 번호 간 평균 간격
        if len(sorted_nums) > 1:
            gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]
            features.append(np.mean(gaps))
            features.append(np.std(gaps))
        else:
            features.extend([0.0, 0.0])
            
        # 최대/최소 번호
        features.append(min(combination) / 45.0)
        features.append(max(combination) / 45.0)
        
        # 번호 범위
        features.append((max(combination) - min(combination)) / 45.0)
        
        return features
    
    def _extract_pattern_features(self, combination: List[int]) -> List[float]:
        """패턴 관련 특성 추출"""
        features = []
        
        # 끝자리 분포
        last_digits = [num % 10 for num in combination]
        unique_last_digits = len(set(last_digits))
        features.append(unique_last_digits / 10.0)
        
        # 십의 자리 분포
        tens_digits = [num // 10 for num in combination]
        unique_tens = len(set(tens_digits))
        features.append(unique_tens / 5.0)  # 0,1,2,3,4
        
        # 3의 배수 개수
        multiples_of_3 = sum(1 for num in combination if num % 3 == 0)
        features.append(multiples_of_3 / len(combination))
        
        # 5의 배수 개수
        multiples_of_5 = sum(1 for num in combination if num % 5 == 0)
        features.append(multiples_of_5 / len(combination))
        
        # 7의 배수 개수
        multiples_of_7 = sum(1 for num in combination if num % 7 == 0)
        features.append(multiples_of_7 / len(combination))
        
        return features
    
    def _extract_historical_features(self, combination: List[int]) -> List[float]:
        """역사적 유사도 특성 추출"""
        features = []
        
        if not self.historical_data:
            # 역사적 데이터가 없으면 기본값
            features.extend([0.5] * 5)
            return features
        
        # 각 번호의 역사적 출현 빈도
        number_frequencies = defaultdict(int)
        for draw in self.historical_data:
            for num in draw.numbers:
                number_frequencies[num] += 1
                
        total_draws = len(self.historical_data)
        combination_frequency_score = 0.0
        for num in combination:
            frequency = number_frequencies[num] / total_draws if total_draws > 0 else 0.0
            combination_frequency_score += frequency
        
        features.append(combination_frequency_score / len(combination))
        
        # 과거 당첨번호와의 최대 겹침 개수
        max_overlap = 0
        recent_overlap = 0
        
        for i, draw in enumerate(self.historical_data):
            overlap = len(set(combination) & set(draw.numbers))
            max_overlap = max(max_overlap, overlap)
            
            # 최근 10회차와의 겹침 (가중치 적용)
            if i >= len(self.historical_data) - 10:
                weight = (i - (len(self.historical_data) - 10) + 1) / 10
                recent_overlap += overlap * weight
        
        features.append(max_overlap / len(combination))
        features.append(recent_overlap / len(combination))
        
        # 번호 조합의 새로움 정도
        novelty_score = 0.0
        for pair in combinations(combination, 2):
            pair_seen = False
            for draw in self.historical_data:
                if pair[0] in draw.numbers and pair[1] in draw.numbers:
                    pair_seen = True
                    break
            if not pair_seen:
                novelty_score += 1
                
        total_pairs = len(list(combinations(combination, 2)))
        features.append(novelty_score / total_pairs if total_pairs > 0 else 0.0)
        
        # 최근 트렌드와의 일치도
        recent_numbers = set()
        recent_count = min(5, len(self.historical_data))
        for draw in self.historical_data[-recent_count:]:
            recent_numbers.update(draw.numbers)
            
        trend_match = len(set(combination) & recent_numbers)
        features.append(trend_match / len(combination))
        
        return features
    
    def _extract_statistical_features(self, combination: List[int]) -> List[float]:
        """통계적 특성 추출"""
        features = []
        
        # 기본 통계
        features.append(np.mean(combination) / 45.0)
        features.append(np.std(combination) / 45.0)
        features.append(np.median(combination) / 45.0)
        
        # 분산과 편차
        variance = np.var(combination)
        features.append(variance / (45.0 ** 2))
        
        # 왜도와 첨도 (scipy 없이 간단 계산)
        mean_val = np.mean(combination)
        std_val = np.std(combination)
        
        if std_val > 0:
            skewness = np.mean([(x - mean_val) ** 3 for x in combination]) / (std_val ** 3)
            kurtosis = np.mean([(x - mean_val) ** 4 for x in combination]) / (std_val ** 4) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0
            
        features.append(skewness)
        features.append(kurtosis)
        
        return features
    
    def _extract_section_features(self, combination: List[int]) -> List[float]:
        """구간별 분포 특성 추출"""
        features = []
        
        # 5개 구간으로 나누어 분포 확인 (1-9, 10-18, 19-27, 28-36, 37-45)
        sections = [0] * 5
        for num in combination:
            section_idx = min(4, (num - 1) // 9)
            sections[section_idx] += 1
            
        # 각 구간의 비율
        for count in sections:
            features.append(count / len(combination))
            
        # 구간 간 분산
        section_variance = np.var(sections)
        features.append(section_variance)
        
        # 빈 구간 개수
        empty_sections = sum(1 for count in sections if count == 0)
        features.append(empty_sections / 5.0)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """특성 이름 반환"""
        names = []
        
        # 분포 특성
        names.extend([
            "odd_ratio", "consecutive_ratio", "avg_gap", "gap_std",
            "min_num_norm", "max_num_norm", "range_norm"
        ])
        
        # 패턴 특성
        names.extend([
            "unique_last_digits", "unique_tens", "multiples_of_3",
            "multiples_of_5", "multiples_of_7"
        ])
        
        # 역사적 특성
        names.extend([
            "avg_frequency", "max_overlap", "recent_overlap",
            "novelty_score", "trend_match"
        ])
        
        # 통계적 특성
        names.extend([
            "mean_norm", "std_norm", "median_norm", "variance_norm",
            "skewness", "kurtosis"
        ])
        
        # 구간별 특성
        names.extend([
            "section_0", "section_1", "section_2", "section_3", "section_4",
            "section_variance", "empty_sections_ratio"
        ])
        
        return names


class RiskFilterEngine:
    """
    고위험 조합을 필터링하는 리스크 필터 시스템의 메인 클래스
    """

    def __init__(self):
        """
        RiskFilterEngine 초기화 (의존성 주입 사용)
        """
        self.logger = get_logger(__name__)

        # --- 의존성 해결 ---
        self.config_manager: Config = resolve(Config)
        self.config = self.config_manager.get_config("main")
        self.paths = self.config_manager.get_paths()
        self.performance_engine: UnifiedPerformanceEngine = resolve(UnifiedPerformanceEngine)
        self.performance_monitor: AutoPerformanceMonitor = resolve(AutoPerformanceMonitor)
        self.cuda_optimizer: CudaOptimizer = resolve(CudaOptimizer)
        self.memory_manager: UnifiedMemoryManager = resolve(UnifiedMemoryManager)
        # --------------------

        self._setup_directories()

        self.risk_config = self._setup_risk_config()
        self.feature_extractor = RiskFeatureExtractor(self.risk_config)
        self.historical_data = self.load_historical_data()
        self.feature_extractor.load_historical_data(self.historical_data)

        # RandomForest 모델 초기화 (CachePathManager 주입)
        self.rf_config = self._prepare_rf_config()
        cache_path_manager = UnifiedCachePathManager(self.paths)
        self.risk_model = RandomForestModel(
            config=self.rf_config,
            cache_path_manager=cache_path_manager
        )
        self.scaler = RobustScaler()

        self.logger.info("✅ 리스크 필터 엔진 초기화 완료")

    def _prepare_rf_config(self) -> Dict[str, Any]:
        """RandomForest 모델 설정 준비"""
        base_config = self.config.get("random_forest", {})
        return {
            **base_config,
            "use_gpu": torch.cuda.is_available(),
            "random_forest": {
                "n_estimators": base_config.get("n_estimators", 200),
                "max_depth": base_config.get("max_depth", 15),
                "min_samples_split": base_config.get("min_samples_split", 5),
                "min_samples_leaf": base_config.get("min_samples_leaf", 2),
                "max_features": base_config.get("max_features", "sqrt"),
                "bootstrap": True,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced",  # 불균형 데이터 처리
            }
        }

    def _setup_directories(self):
        """필요한 디렉토리 설정"""
        self.output_dir = Path(self.paths.result_dir) / "risk_filter"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = Path(self.paths.models_dir) / "random_forest"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(self.paths.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _setup_risk_config(self) -> Dict[str, Any]:
        """리스크 필터링 관련 주요 설정"""
        return {
            "risk_threshold": self.config.get("risk_filter", {}).get("threshold", 0.7),
            "min_combinations": self.config.get("risk_filter", {}).get("min_combinations", 100),
            "max_combinations": self.config.get("risk_filter", {}).get("max_combinations", 1000),
            "diversity_weight": self.config.get("risk_filter", {}).get("diversity_weight", 0.3),
            "cross_validation_folds": 5,
            "feature_importance_threshold": 0.01,
            "ensemble_voting": "soft",  # hard, soft
        }

    def load_all_previous_results(self) -> Dict[str, Any]:
        """
        이전 단계(run1~run4)의 모든 결과를 로드합니다.
        
        Returns:
            Dict[str, Any]: 통합된 이전 결과
        """
        try:
            self.logger.info("이전 단계 결과 로드 시작...")
            
            with self.performance_monitor.track("load_previous_results"):
                results = {
                    "run1_analysis": {},
                    "run2_predictions": {},
                    "run3_anomaly": {},
                    "run4_trend_correction": {},
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
                            self.logger.warning(f"파일 로드 실패 {file_path}: {e}")
                
                # run2 예측 결과 로드
                predictions_dir = Path(self.paths.predictions_dir)
                if predictions_dir.exists():
                    for file_path in predictions_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run2_predictions"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"파일 로드 실패 {file_path}: {e}")
                
                # run3 이상감지 결과 로드
                anomaly_dir = Path(self.paths.result_dir) / "anomaly_detection"
                if anomaly_dir.exists():
                    for file_path in anomaly_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run3_anomaly"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"파일 로드 실패 {file_path}: {e}")
                
                # run4 트렌드 보정 결과 로드
                trend_dir = Path(self.paths.result_dir) / "trend_correction"
                if trend_dir.exists():
                    for file_path in trend_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run4_trend_correction"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"파일 로드 실패 {file_path}: {e}")
                
                # 메타데이터 추가
                results["metadata"] = {
                    "load_timestamp": datetime.now().isoformat(),
                    "run1_files": len(results["run1_analysis"]),
                    "run2_files": len(results["run2_predictions"]),
                    "run3_files": len(results["run3_anomaly"]),
                    "run4_files": len(results["run4_trend_correction"]),
                }
            
            self.logger.info(f"✅ 이전 결과 로드 완료: "
                           f"run1({len(results['run1_analysis'])}), "
                           f"run2({len(results['run2_predictions'])}), "
                           f"run3({len(results['run3_anomaly'])}), "
                           f"run4({len(results['run4_trend_correction'])})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"이전 결과 로드 실패: {e}")
            raise

    def load_historical_data(self) -> List[LotteryNumber]:
        """
        과거 로또 데이터를 로드합니다.
        
        Returns:
            List[LotteryNumber]: 과거 로또 당첨 번호 리스트
        """
        try:
            self.logger.info("과거 로또 데이터 로드 시작...")
            
            with self.performance_monitor.track("load_historical_data"):
                # 설정에서 데이터 경로 가져오기
                config = self.config_manager.get_config("main")
                data_path = config.get("data", {}).get("historical_data_path", "data/raw/lottery.csv")
                
                # 데이터 로드
                historical_data = load_draw_history(data_path)
                
                if not historical_data:
                    self.logger.warning("과거 데이터를 찾을 수 없습니다. 더미 데이터를 생성합니다.")
                    historical_data = self._create_dummy_historical_data()
                
                # 회차 순으로 정렬
                historical_data.sort(key=lambda x: x.draw_no)
            
            self.logger.info(f"✅ 과거 로또 데이터 로드 완료: {len(historical_data)}개 회차")
            return historical_data
            
        except Exception as e:
            self.logger.error(f"과거 로또 데이터 로드 실패: {e}")
            # 비상용 더미 데이터 반환
            return self._create_dummy_historical_data()

    def _create_dummy_historical_data(self) -> List[LotteryNumber]:
        """더미 과거 로또 데이터 생성"""
        dummy_data = []
        for i in range(1, 201):  # 200회차 더미 데이터
            numbers = sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())
            dummy_data.append(LotteryNumber(
                draw_no=i,
                numbers=numbers,
                date=f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
            ))
        return dummy_data

    def generate_candidate_combinations(self, all_results: Dict[str, Any]) -> List[Tuple[List[int], float]]:
        """
        이전 단계 결과를 기반으로 후보 조합들을 생성합니다.
        
        Args:
            all_results: 이전 단계 결과들
            
        Returns:
            List[Tuple[List[int], float]]: (조합, 신뢰도) 리스트
        """
        try:
            self.logger.info("후보 조합 생성 시작...")
            
            with self.performance_monitor.track("generate_candidates"):
                # 각 단계별 점수 통합
                integrated_scores = self._integrate_stage_scores(all_results)
                
                # 상위 점수 번호들 선별
                top_numbers = self._select_top_numbers(integrated_scores)
                
                # 조합 생성 전략
                candidates = []
                
                # 1. 최고 점수 기반 조합
                high_score_combinations = self._generate_high_score_combinations(
                    top_numbers, integrated_scores
                )
                candidates.extend(high_score_combinations)
                
                # 2. 균형잡힌 조합
                balanced_combinations = self._generate_balanced_combinations(
                    top_numbers, integrated_scores
                )
                candidates.extend(balanced_combinations)
                
                # 3. 다양성 보장 조합
                diverse_combinations = self._generate_diverse_combinations(
                    top_numbers, integrated_scores
                )
                candidates.extend(diverse_combinations)
                
                # 중복 제거 및 정렬
                unique_candidates = self._deduplicate_combinations(candidates)
                
                # 최대 개수 제한
                max_candidates = self.risk_config["max_combinations"]
                if len(unique_candidates) > max_candidates:
                    unique_candidates = unique_candidates[:max_candidates]
            
            self.logger.info(f"✅ 후보 조합 생성 완료: {len(unique_candidates)}개")
            return unique_candidates
            
        except Exception as e:
            self.logger.error(f"후보 조합 생성 실패: {e}")
            # 비상용 랜덤 조합 생성
            return self._generate_fallback_combinations()

    def _integrate_stage_scores(self, all_results: Dict[str, Any]) -> Dict[int, float]:
        """각 단계별 점수를 통합하여 번호별 종합 점수 계산"""
        integrated_scores = defaultdict(float)
        total_weight = 0
        
        # run1 통합 분석 점수 (가중치: 0.3)
        for name, data in all_results["run1_analysis"].items():
            if "unified_analysis" in name and "comprehensive_scores" in data:
                scores = data["comprehensive_scores"]
                for num in range(1, 46):
                    integrated_scores[num] += scores.get(str(num), 0.0) * 0.3
                total_weight += 0.3
                break
        
        # run2 ML 예측 점수 (가중치: 0.3)
        for name, data in all_results["run2_predictions"].items():
            if "ml_predictions" in name and "predictions" in data:
                predictions = data["predictions"]
                for num in range(1, 46):
                    integrated_scores[num] += predictions.get(str(num), 0.0) * 0.3
                total_weight += 0.3
                break
        
        # run3 이상감지 점수 (가중치: 0.2) - 이상치가 낮을수록 좋음
        for name, data in all_results["run3_anomaly"].items():
            if "anomaly_detection" in name and "anomaly_scores" in data:
                anomaly_scores = data["anomaly_scores"]
                for num in range(1, 46):
                    anomaly_score = anomaly_scores.get(str(num), 0.5)
                    integrated_scores[num] += (1.0 - anomaly_score) * 0.2
                total_weight += 0.2
                break
        
        # run4 트렌드 보정 점수 (가중치: 0.2)
        for name, data in all_results["run4_trend_correction"].items():
            if "trend_correction_scores" in data:
                trend_scores = data["trend_correction_scores"]
                for num in range(1, 46):
                    integrated_scores[num] += trend_scores.get(str(num), 0.0) * 0.2
                total_weight += 0.2
                break
        
        # 가중치 정규화
        if total_weight > 0:
            for num in integrated_scores:
                integrated_scores[num] /= total_weight
        
        return dict(integrated_scores)

    def _select_top_numbers(self, integrated_scores: Dict[int, float], top_k: int = 20) -> List[int]:
        """상위 점수 번호들 선별"""
        sorted_numbers = sorted(integrated_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in sorted_numbers[:top_k]]

    def _generate_high_score_combinations(self, top_numbers: List[int], scores: Dict[int, float]) -> List[Tuple[List[int], float]]:
        """최고 점수 기반 조합 생성"""
        combinations_list = []
        
        # 상위 번호들로 조합 생성
        for combo in combinations(top_numbers[:12], 6):
            combo_list = list(combo)
            confidence = np.mean([scores.get(num, 0.0) for num in combo_list])
            combinations_list.append((combo_list, confidence))
        
        # 신뢰도 기준 정렬
        combinations_list.sort(key=lambda x: x[1], reverse=True)
        return combinations_list[:50]  # 상위 50개

    def _generate_balanced_combinations(self, top_numbers: List[int], scores: Dict[int, float]) -> List[Tuple[List[int], float]]:
        """균형잡힌 조합 생성"""
        combinations_list = []
        
        # 구간별 균형 고려
        sections = {
            0: [n for n in top_numbers if 1 <= n <= 9],
            1: [n for n in top_numbers if 10 <= n <= 18],
            2: [n for n in top_numbers if 19 <= n <= 27],
            3: [n for n in top_numbers if 28 <= n <= 36],
            4: [n for n in top_numbers if 37 <= n <= 45],
        }
        
        # 각 구간에서 1-2개씩 선택
        for _ in range(30):
            combo = []
            for section_nums in sections.values():
                if section_nums and len(combo) < 6:
                    selected = np.random.choice(section_nums, 
                                              size=min(2, len(section_nums), 6-len(combo)), 
                                              replace=False)
                    combo.extend(selected)
            
            if len(combo) == 6:
                confidence = np.mean([scores.get(num, 0.0) for num in combo])
                combinations_list.append((sorted(combo), confidence))
        
        return combinations_list

    def _generate_diverse_combinations(self, top_numbers: List[int], scores: Dict[int, float]) -> List[Tuple[List[int], float]]:
        """다양성 보장 조합 생성"""
        combinations_list = []
        
        # 홀짝 균형, 연속성 제한 등을 고려한 다양한 조합
        for _ in range(50):
            combo = []
            
            # 홀짝 균형 (3:3 또는 4:2)
            odd_target = np.random.choice([3, 4])
            even_target = 6 - odd_target
            
            odd_candidates = [n for n in top_numbers if n % 2 == 1]
            even_candidates = [n for n in top_numbers if n % 2 == 0]
            
            if len(odd_candidates) >= odd_target and len(even_candidates) >= even_target:
                selected_odds = np.random.choice(odd_candidates, odd_target, replace=False)
                selected_evens = np.random.choice(even_candidates, even_target, replace=False)
                
                combo = sorted(list(selected_odds) + list(selected_evens))
                confidence = np.mean([scores.get(num, 0.0) for num in combo])
                combinations_list.append((combo, confidence))
        
        return combinations_list

    def _deduplicate_combinations(self, candidates: List[Tuple[List[int], float]]) -> List[Tuple[List[int], float]]:
        """조합 중복 제거"""
        seen = set()
        unique_candidates = []
        
        for combo, confidence in candidates:
            combo_tuple = tuple(sorted(combo))
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                unique_candidates.append((list(combo_tuple), confidence))
        
        # 신뢰도 기준 정렬
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        return unique_candidates

    def _generate_fallback_combinations(self) -> List[Tuple[List[int], float]]:
        """비상용 랜덤 조합 생성"""
        combinations_list = []
        for _ in range(100):
            combo = sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())
            confidence = 0.5  # 기본 신뢰도
            combinations_list.append((combo, confidence))
        return combinations_list

    def prepare_training_data(self, candidates: List[Tuple[List[int], float]], historical_data: List[LotteryNumber]) -> Tuple[np.ndarray, np.ndarray]:
        """
        RandomForest 훈련을 위한 데이터 준비
        
        Args:
            candidates: 후보 조합들
            historical_data: 역사적 데이터
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_features, y_labels)
        """
        try:
            self.logger.info("훈련 데이터 준비 시작...")
            
            with self.performance_monitor.track("prepare_training_data"):
                # 특성 추출기에 역사적 데이터 로드
                self.feature_extractor.load_historical_data(historical_data)
                
                X_features = []
                y_labels = []
                
                # 후보 조합들의 특성 추출
                for combo, confidence in candidates:
                    features = self.feature_extractor.extract_combination_features(combo)
                    X_features.append(features)
                    
                    # 리스크 레이블 생성 (높은 신뢰도 = 낮은 리스크)
                    risk_label = 1 if confidence < 0.5 else 0  # 1: 고위험, 0: 저위험
                    y_labels.append(risk_label)
                
                # NumPy 배열로 변환
                X = np.array(X_features, dtype=np.float32)
                y = np.array(y_labels, dtype=np.int32)
                
                # 특성 스케일링
                X_scaled = self.scaler.fit_transform(X)
                
                # 클래스 균형 확인
                unique, counts = np.unique(y, return_counts=True)
                class_distribution = dict(zip(unique, counts))
                self.logger.info(f"클래스 분포: {class_distribution}")
                
                # 클래스 불균형이 심한 경우 데이터 증강
                if len(unique) == 2 and min(counts) / max(counts) < 0.3:
                    X_scaled, y = self._balance_training_data(X_scaled, y)
            
            self.logger.info(f"✅ 훈련 데이터 준비 완료: X={X_scaled.shape}, y={y.shape}")
            return X_scaled, y
            
        except Exception as e:
            self.logger.error(f"훈련 데이터 준비 실패: {e}")
            raise

    def _balance_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """클래스 불균형 해결을 위한 데이터 증강"""
        try:
            from sklearn.utils import resample
            
            # 소수 클래스와 다수 클래스 분리
            unique_classes = np.unique(y)
            class_counts = [np.sum(y == cls) for cls in unique_classes]
            
            if len(unique_classes) < 2:
                return X, y
            
            minority_class = unique_classes[np.argmin(class_counts)]
            majority_class = unique_classes[np.argmax(class_counts)]
            
            # 소수 클래스와 다수 클래스 데이터 분리
            minority_mask = y == minority_class
            majority_mask = y == majority_class
            
            X_minority = X[minority_mask]
            y_minority = y[minority_mask]
            
            # 타겟 샘플 수 (다수 클래스의 80% 수준)
            target_samples = int(np.sum(majority_mask) * 0.8)
            
            if len(X_minority) > 0 and len(X_minority) < target_samples:
                resampled_data = resample(
                    X_minority, y_minority,
                    n_samples=target_samples,
                    replace=True,
                    random_state=42
                )
                # resample 결과가 tuple인지 확인하고 언패킹
                if isinstance(resampled_data, tuple) and len(resampled_data) == 2:
                    X_minority_resampled, y_minority_resampled = resampled_data
                else:
                    # 예외 상황: 원본 데이터 사용
                    X_minority_resampled, y_minority_resampled = X_minority, y_minority
                
                # 결합
                X_balanced = np.vstack([X[majority_mask], X_minority_resampled])
                y_balanced = np.hstack([y[majority_mask], y_minority_resampled])
            else:
                X_balanced = X
                y_balanced = y
            
            return X_balanced, y_balanced
            
        except Exception as e:
            self.logger.warning(f"데이터 균형 조정 실패: {e}, 원본 데이터 반환")
            return X, y

    def train_risk_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        RandomForest 리스크 모델 훈련
        
        Args:
            X: 특성 데이터
            y: 레이블 데이터
            
        Returns:
            Dict[str, Any]: 훈련 결과
        """
        try:
            self.logger.info("RandomForest 리스크 모델 훈련 시작...")
            
            training_results = {}
            
            if self.cuda_optimizer.is_cuda_available():
                with self.cuda_optimizer.gpu_memory_scope(size_mb=1024, device_id=0):
                    with self.performance_monitor.track("risk_model_training"):
                        training_results = self._execute_risk_model_training(X, y)
            else:
                with self.performance_monitor.track("risk_model_training"):
                    training_results = self._execute_risk_model_training(X, y)
            
            # 모델 저장
            model_path = self.model_dir / "risk_filter_model.joblib"
            save_success = self.risk_model.save(str(model_path))
            training_results["model_saved"] = save_success
            training_results["model_path"] = str(model_path)
            
            self.logger.info("✅ RandomForest 리스크 모델 훈련 완료")
            return training_results
            
        except Exception as e:
            self.logger.error(f"리스크 모델 훈련 실패: {e}")
            return {"error": str(e), "training_completed": False}

    def _execute_risk_model_training(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """리스크 모델 훈련 실행"""
        try:
            # 교차 검증을 위한 데이터 분할
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 특성 이름 생성
            feature_names = self.feature_extractor.get_feature_names()
            
            # 모델 훈련
            training_results = self.risk_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                feature_names=feature_names
            )
            
            # 예측 및 평가
            y_pred = self.risk_model.predict(X_test)
            y_pred_proba = None
            
            # 확률 예측 (가능한 경우)
            if hasattr(self.risk_model, "model") and self.risk_model.model is not None and hasattr(self.risk_model.model, "predict_proba"):
                y_pred_proba = self.risk_model.model.predict_proba(X_test)
            
            # 교차 검증
            cv_scores = self._perform_cross_validation(X, y)
            
            # 성능 지표 계산
            performance_metrics = self._calculate_performance_metrics(
                y_test, y_pred, y_pred_proba
            )
            
            # 특성 중요도 분석
            feature_importance = self._analyze_feature_importance(feature_names)
            
            # 결과 통합
            training_results.update({
                "cross_validation": cv_scores,
                "performance_metrics": performance_metrics,
                "feature_importance": feature_importance,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": X.shape[1],
            })
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"리스크 모델 훈련 실행 실패: {e}")
            return {"error": str(e)}

    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """교차 검증 수행"""
        try:
            cv_folds = self.risk_config["cross_validation_folds"]
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # 다양한 지표로 교차 검증
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
            cv_results = {}
            
            for metric in scoring_metrics:
                if self.risk_model.model is not None:
                    scores = cross_val_score(
                        self.risk_model.model, X, y, 
                        cv=skf, scoring=metric, n_jobs=-1
                    )
                    cv_results[metric] = {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "scores": scores.tolist()
                    }
                else:
                    cv_results[metric] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "scores": [0.0] * cv_folds
                    }
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"교차 검증 실패: {e}")
            return {"error": str(e)}

    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """성능 지표 계산"""
        try:
            metrics = {}
            
            # 기본 분류 지표
            metrics["accuracy"] = float(np.mean(y_true == y_pred))
            metrics["f1_score"] = float(f1_score(y_true, y_pred, average='weighted'))
            
            # 분류 리포트
            classification_rep = classification_report(y_true, y_pred, output_dict=True)
            metrics["classification_report"] = classification_rep
            
            # 혼동 행렬
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # ROC AUC (확률 예측이 있는 경우)
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics["roc_auc"] = float(roc_auc)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"성능 지표 계산 실패: {e}")
            return {"error": str(e)}

    def _analyze_feature_importance(self, feature_names: List[str]) -> Dict[str, Any]:
        """특성 중요도 분석"""
        try:
            if not self.risk_model.is_trained:
                return {"error": "모델이 훈련되지 않음"}
            
            # 특성 중요도 가져오기
            importance_dict = self.risk_model.get_feature_importance()
            
            if not importance_dict:
                return {"error": "특성 중요도를 가져올 수 없음"}
            
            # 상위 중요 특성
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_importance[:10]
            
            # 임계값 이상 특성
            threshold = self.risk_config["feature_importance_threshold"]
            important_features = [(name, score) for name, score in sorted_importance if score >= threshold]
            
            return {
                "all_importance": importance_dict,
                "top_10_features": dict(top_features),
                "important_features": dict(important_features),
                "total_features": len(feature_names),
                "important_count": len(important_features),
            }
            
        except Exception as e:
            self.logger.error(f"특성 중요도 분석 실패: {e}")
            return {"error": str(e)}

    def filter_high_risk_combinations(self, candidates: List[Tuple[List[int], float]]) -> Dict[str, Any]:
        """
        고위험 조합 필터링
        
        Args:
            candidates: 후보 조합들
            
        Returns:
            Dict[str, Any]: 필터링 결과
        """
        try:
            self.logger.info("고위험 조합 필터링 시작...")
            
            with self.performance_monitor.track("risk_filtering"):
                if not self.risk_model.is_trained:
                    raise ValueError("리스크 모델이 훈련되지 않았습니다.")
                
                # 각 조합의 특성 추출
                X_candidates = []
                for combo, confidence in candidates:
                    features = self.feature_extractor.extract_combination_features(combo)
                    X_candidates.append(features)
                
                X_candidates = np.array(X_candidates, dtype=np.float32)
                X_candidates_scaled = self.scaler.transform(X_candidates)
                
                # 리스크 예측
                risk_predictions = self.risk_model.predict(X_candidates_scaled)
                
                # 리스크 확률 (가능한 경우)
                risk_probabilities = None
                if hasattr(self.risk_model, "model") and self.risk_model.model is not None and hasattr(self.risk_model.model, "predict_proba"):
                    risk_probabilities = self.risk_model.model.predict_proba(X_candidates_scaled)
                
                # 필터링 적용
                filtered_combinations = []
                high_risk_combinations = []
                
                risk_threshold = self.risk_config["risk_threshold"]
                
                for i, (combo, confidence) in enumerate(candidates):
                    risk_pred = risk_predictions[i]
                    risk_prob = risk_probabilities[i][1] if risk_probabilities is not None else risk_pred
                    
                    combination_data = {
                        "numbers": combo,
                        "confidence": confidence,
                        "risk_prediction": int(risk_pred),
                        "risk_probability": float(risk_prob),
                    }
                    
                    if risk_prob < risk_threshold:  # 낮은 리스크
                        filtered_combinations.append(combination_data)
                    else:  # 높은 리스크
                        high_risk_combinations.append(combination_data)
                
                # 다양성 보장
                if len(filtered_combinations) > 0:
                    diverse_combinations = self._ensure_diversity(filtered_combinations)
                else:
                    diverse_combinations = []
                
                # 최소 조합 수 보장
                min_combinations = self.risk_config["min_combinations"]
                if len(diverse_combinations) < min_combinations:
                    self.logger.warning(f"필터링된 조합이 부족합니다. 추가 조합을 선택합니다.")
                    additional_needed = min_combinations - len(diverse_combinations)
                    
                    # 리스크가 낮은 순으로 추가 선택
                    sorted_high_risk = sorted(high_risk_combinations, key=lambda x: x["risk_probability"])
                    diverse_combinations.extend(sorted_high_risk[:additional_needed])
                
                # 결과 정리
                filtering_results = {
                    "total_evaluated": len(candidates),
                    "low_risk_count": len(filtered_combinations),
                    "high_risk_count": len(high_risk_combinations),
                    "final_combinations": diverse_combinations,
                    "risk_threshold": risk_threshold,
                    "filtering_stats": self._calculate_filtering_stats(
                        filtered_combinations, high_risk_combinations
                    ),
                }
            
            self.logger.info(f"✅ 리스크 필터링 완료: {len(diverse_combinations)}개 조합 선택")
            return filtering_results
            
        except Exception as e:
            self.logger.error(f"리스크 필터링 실패: {e}")
            raise

    def _ensure_diversity(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """다양성 보장 메커니즘"""
        try:
            if len(combinations) <= 10:
                return combinations
            
            # 신뢰도와 다양성 점수 계산
            scored_combinations = []
            
            for combo_data in combinations:
                combo = combo_data["numbers"]
                confidence = combo_data["confidence"]
                combo_data["risk_probability"]
                
                # 다양성 점수 계산
                diversity_score = self._calculate_diversity_score(combo, combinations)
                
                # 최종 점수 (신뢰도 + 다양성)
                diversity_weight = self.risk_config["diversity_weight"]
                final_score = (1 - diversity_weight) * confidence + diversity_weight * diversity_score
                
                combo_data["diversity_score"] = diversity_score
                combo_data["final_score"] = final_score
                scored_combinations.append(combo_data)
            
            # 최종 점수 기준 정렬
            scored_combinations.sort(key=lambda x: x["final_score"], reverse=True)
            
            # 상위 조합 선택
            max_combinations = min(self.risk_config["max_combinations"], len(scored_combinations))
            return scored_combinations[:max_combinations]
            
        except Exception as e:
            self.logger.error(f"다양성 보장 실패: {e}")
            return combinations

    def _calculate_diversity_score(self, combo: List[int], all_combinations: List[Dict[str, Any]]) -> float:
        """조합의 다양성 점수 계산"""
        try:
            if len(all_combinations) <= 1:
                return 1.0
            
            # 다른 조합들과의 평균 차이 계산
            total_difference = 0
            count = 0
            
            for other_combo_data in all_combinations:
                other_combo = other_combo_data["numbers"]
                if combo != other_combo:
                    # 겹치는 번호 수 계산
                    overlap = len(set(combo) & set(other_combo))
                    difference = 1.0 - (overlap / 6.0)  # 6개 중 겹치지 않는 비율
                    total_difference += difference
                    count += 1
            
            return total_difference / count if count > 0 else 1.0
            
        except Exception:
            return 0.5

    def _calculate_filtering_stats(self, low_risk: List[Dict], high_risk: List[Dict]) -> Dict[str, Any]:
        """필터링 통계 계산"""
        try:
            stats = {}
            
            # 기본 통계
            total = len(low_risk) + len(high_risk)
            stats["total_combinations"] = total
            stats["low_risk_rate"] = len(low_risk) / total if total > 0 else 0
            stats["high_risk_rate"] = len(high_risk) / total if total > 0 else 0
            
            # 신뢰도 통계
            if low_risk:
                low_risk_confidences = [combo["confidence"] for combo in low_risk]
                stats["low_risk_confidence"] = {
                    "mean": float(np.mean(low_risk_confidences)),
                    "std": float(np.std(low_risk_confidences)),
                    "min": float(np.min(low_risk_confidences)),
                    "max": float(np.max(low_risk_confidences)),
                }
            
            if high_risk:
                high_risk_confidences = [combo["confidence"] for combo in high_risk]
                stats["high_risk_confidence"] = {
                    "mean": float(np.mean(high_risk_confidences)),
                    "std": float(np.std(high_risk_confidences)),
                    "min": float(np.min(high_risk_confidences)),
                    "max": float(np.max(high_risk_confidences)),
                }
            
            # 리스크 확률 통계
            if low_risk:
                low_risk_probs = [combo["risk_probability"] for combo in low_risk]
                stats["low_risk_probability"] = {
                    "mean": float(np.mean(low_risk_probs)),
                    "std": float(np.std(low_risk_probs)),
                }
            
            if high_risk:
                high_risk_probs = [combo["risk_probability"] for combo in high_risk]
                stats["high_risk_probability"] = {
                    "mean": float(np.mean(high_risk_probs)),
                    "std": float(np.std(high_risk_probs)),
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"필터링 통계 계산 실패: {e}")
            return {"error": str(e)}

    def save_risk_filter_results(self, results: Dict[str, Any]) -> str:
        """
        리스크 필터 결과를 저장합니다.
        
        Args:
            results: 저장할 결과
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_filter_results_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # 메타데이터 추가
            results["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "version": "RiskFilterEngine_v1.0",
                "config": self.config,
                "risk_config": self.risk_config,
                "device": str(self.device),
                "performance_stats": self.performance_monitor.get_performance_summary(),
            }
            
            # GPU 메모리 통계 추가
            if self.cuda_optimizer.is_cuda_available():
                results["metadata"]["gpu_memory_stats"] = self.cuda_optimizer.get_gpu_memory_pool().get_stats()
            
            # 결과 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ 리스크 필터 결과 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"리스크 필터 결과 저장 실패: {e}")
            raise

    def run_full_risk_filter(self) -> Dict[str, Any]:
        """
        전체 리스크 필터 프로세스를 실행합니다.
        
        Returns:
            Dict[str, Any]: 리스크 필터 결과
        """
        try:
            self.logger.info("🚀 리스크 필터 시스템 시작")
            start_time = time.time()
            
            # 1. 이전 단계 결과 로드
            all_results = self.load_all_previous_results()
            
            # 2. 과거 로또 데이터 로드
            historical_data = self.load_historical_data()
            
            # 3. 후보 조합 생성
            candidates = self.generate_candidate_combinations(all_results)
            
            # 4. 훈련 데이터 준비
            X_train, y_train = self.prepare_training_data(candidates, historical_data)
            
            # 5. 리스크 모델 훈련
            training_results = self.train_risk_model(X_train, y_train)
            
            # 6. 고위험 조합 필터링
            filtering_results = self.filter_high_risk_combinations(candidates)
            
            # 7. 결과 통합
            results = {
                "risk_filtered_combinations": {
                    f"combination_{i+1}": {
                        "numbers": combo["numbers"],
                        "risk_score": combo["risk_probability"],
                        "confidence": combo["confidence"],
                        "diversity_score": combo.get("diversity_score", 0.0),
                        "final_score": combo.get("final_score", 0.0),
                    }
                    for i, combo in enumerate(filtering_results["final_combinations"])
                },
                "risk_analysis": {
                    "total_combinations_evaluated": filtering_results["total_evaluated"],
                    "high_risk_filtered": filtering_results["high_risk_count"],
                    "remaining_combinations": len(filtering_results["final_combinations"]),
                    "risk_threshold": filtering_results["risk_threshold"],
                    "filtering_stats": filtering_results["filtering_stats"],
                },
                "model_performance": training_results.get("performance_metrics", {}),
                "cross_validation": training_results.get("cross_validation", {}),
                "feature_importance": training_results.get("feature_importance", {}),
                "training_summary": {
                    "training_samples": training_results.get("training_samples", 0),
                    "test_samples": training_results.get("test_samples", 0),
                    "feature_count": training_results.get("feature_count", 0),
                    "model_saved": training_results.get("model_saved", False),
                    "model_path": training_results.get("model_path", ""),
                },
                "candidate_generation": {
                    "total_candidates": len(candidates),
                    "data_sources": {
                        "run1_files": len(all_results["run1_analysis"]),
                        "run2_files": len(all_results["run2_predictions"]),
                        "run3_files": len(all_results["run3_anomaly"]),
                        "run4_files": len(all_results["run4_trend_correction"]),
                        "historical_data_count": len(historical_data),
                    },
                },
            }
            
            # 8. 결과 저장
            saved_path = self.save_risk_filter_results(results)
            results["saved_path"] = saved_path
            
            # 9. 성능 통계
            end_time = time.time()
            execution_time = end_time - start_time
            
            performance_summary = {
                "execution_time_seconds": execution_time,
                "performance_monitor": self.performance_monitor.get_performance_summary(),
                "memory_manager_stats": self.memory_manager.get_stats(),
            }
            
            if self.cuda_optimizer.is_cuda_available():
                performance_summary["gpu_memory_stats"] = self.cuda_optimizer.get_gpu_memory_pool().get_stats()
            
            results["performance_summary"] = performance_summary
            
            self.logger.info(f"✅ 리스크 필터 시스템 완료 (소요시간: {execution_time:.2f}초)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"리스크 필터 시스템 실행 실패: {e}")
            raise


def main():
    """메인 실행 함수"""
    # 1. 의존성 설정
    configure_dependencies()

    logger.info("=" * 80)
    logger.info("🚀 5단계: 리스크 필터 시스템 시작")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # 리스크 필터 엔진 생성 (설정 제거)
        engine = RiskFilterEngine()
        
        # 전체 파이프라인 실행
        final_results = engine.run_full_risk_filter()

        total_time = time.time() - start_time
        logger.info(f"✅ 리스크 필터링 완료! 총 실행 시간: {total_time:.2f}초")
        logger.info(f"📁 최종 결과 파일: {final_results.get('saved_path')}")

    except Exception as e:
        logger.error(f"❌ 리스크 필터링 실패: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    main()
