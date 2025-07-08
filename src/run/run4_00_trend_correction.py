#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
트렌드 보정 시스템 (4단계) - TCN 모델 기반 시계열 트렌드 보정

기존 TCN 모델과 TrendAnalyzerV2를 완전히 통합하여 시계열 트렌드 보정을 수행합니다.
이전 3단계(run1, run2, run3)의 모든 결과를 시계열 형태로 통합하고,
최근 회차 변화를 반영한 트렌드 보정을 적용합니다.

주요 기능:
- 이전 단계 결과 완전 통합 (run1 분석, run2 예측, run3 이상감지)
- TrendAnalyzerV2 기반 회차별 트렌드 분석
- TCN 모델을 통한 시계열 패턴 학습 및 예측
- GPU 최적화 및 메모리 풀 관리
- 적응적 트렌드 보정 가중치 적용
"""

# 1. 표준 라이브러리
import json
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 2. 서드파티
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 3. 프로젝트 내부 (리팩토링된 의존성 관리)
from ..utils.dependency_injection import configure_dependencies, resolve
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from ..utils.unified_performance_engine import UnifiedPerformanceEngine, AutoPerformanceMonitor
from ..utils.cuda_optimizers import CudaOptimizer, CudaConfig
from ..utils.unified_memory_manager import UnifiedMemoryManager
from ..utils.unified_config import Config
from ..utils.data_loader import load_draw_history
from ..models.dl.tcn_model import TCNModel
from ..analysis.trend_analyzer_v2 import TrendAnalyzerV2
from ..pipeline.optimized_tcn_preprocessor import OptimizedTCNPreprocessor
from ..utils.cache_manager import UnifiedCachePathManager, CacheManager

logger = get_logger(__name__)


class TrendCorrectionEngine:
    """
    트렌드 보정 엔진
    
    TCN 모델과 TrendAnalyzerV2를 완전히 통합한 시계열 트렌드 보정 시스템입니다.
    GPU 최적화, 메모리 관리, 에러 처리를 포함한 고성능 처리가 특징입니다.
    """

    def __init__(self):
        """
        트렌드 보정 엔진 초기화 (의존성 주입 사용)
        """
        self.logger = get_logger(__name__)

        # --- 의존성 해결 ---
        self.config_manager: Config = resolve(Config)
        self.config = self.config_manager.get_config("main")
        self.paths = self.config_manager.get_paths()
        self.performance_engine: UnifiedPerformanceEngine = resolve(UnifiedPerformanceEngine)
        self.performance_monitor: AutoPerformanceMonitor = resolve(AutoPerformanceMonitor)
        self.memory_manager: UnifiedMemoryManager = resolve(UnifiedMemoryManager)
        self.cuda_optimizer: CudaOptimizer = resolve(CudaOptimizer)
        self.cache_path_manager: UnifiedCachePathManager = resolve(UnifiedCachePathManager)
        # --------------------

        # GPU 최적화 설정
        cuda_config_data = self.config.get("cuda_config", {})
        self.cuda_config = CudaConfig(
            use_amp=cuda_config_data.get("use_amp", True),
            use_cudnn_benchmark=cuda_config_data.get("use_cudnn_benchmark", True),
            use_tensorrt=cuda_config_data.get("use_tensorrt", False),
            tensorrt_precision="fp16",
        )
        self.cuda_optimizer.configure(self.cuda_config)

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"디바이스 설정: {self.device}")
        
        # TCN 모델 설정 및 초기화
        self.tcn_config = self._prepare_tcn_config()
        tcn_cache_manager = CacheManager(
            path_manager=self.cache_path_manager,
            cache_type="tcn_model",
            config=self.tcn_config.get("cache_config")
        )
        self.tcn_model = TCNModel(config=self.tcn_config, cache_manager=tcn_cache_manager)
        
        # TrendAnalyzerV2 초기화
        self.trend_analyzer = TrendAnalyzerV2(self.config.get("trend_analyzer_v2", {}))
        
        # TCN 전처리기 초기화
        self.tcn_preprocessor = OptimizedTCNPreprocessor(
            self.config.get("tcn_preprocessor", {})
        )
        
        # 스케일러 초기화
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # 경로 설정
        self._setup_directories()
        
        # 트렌드 보정 설정
        self.trend_config = self._setup_trend_config()
        
        self.logger.info("✅ TrendCorrectionEngine 초기화 완료")

    def _prepare_tcn_config(self) -> Dict[str, Any]:
        """TCN 모델 설정 준비"""
        base_config = self.config.get("tcn_model", {})
        return {
            **base_config,
            "use_gpu": torch.cuda.is_available(),
            "tcn": {
                "input_dim": base_config.get("input_dim", 135),  # 45*3 (unified+trend+ml)
                "num_channels": base_config.get("num_channels", [64, 128, 256, 128, 64]),
                "kernel_size": base_config.get("kernel_size", 3),
                "dropout": base_config.get("dropout", 0.2),
                "sequence_length": base_config.get("sequence_length", 50),
                "output_dim": 45,  # 로또 번호 45개
                "learning_rate": base_config.get("learning_rate", 0.001),
                "batch_size": base_config.get("batch_size", 32),
            }
        }

    def _setup_directories(self):
        """필요한 디렉토리 설정"""
        self.output_dir = Path(self.paths.result_dir) / "trend_correction"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = Path(self.paths.models_dir) / "tcn"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(self.paths.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _setup_trend_config(self) -> Dict[str, Any]:
        """트렌드 보정 설정"""
        return {
            "correction_weight": self.config.get("trend_correction", {}).get("weight", 0.3),
            "sequence_length": self.config.get("tcn_model", {}).get("sequence_length", 50),
            "prediction_horizon": self.config.get("trend_correction", {}).get("prediction_horizon", 5),
            "min_data_points": 100,  # 최소 데이터 포인트
            "trend_threshold": 0.1,  # 트렌드 임계값
            "adaptive_weight": True,  # 적응적 가중치 사용
        }

    def load_all_previous_results(self) -> Dict[str, Any]:
        """
        이전 단계(run1, run2, run3)의 모든 결과를 로드합니다.
        
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
                    "metadata": {}
                }
                
                # run1 분석 결과 로드
                analysis_dir = self.paths.get_analysis_path()
                if analysis_dir.exists():
                    for file_path in analysis_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run1_analysis"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"파일 로드 실패 {file_path}: {e}")
                
                # run2 예측 결과 로드
                predictions_dir = self.paths.get_ml_predictions_path()
                if predictions_dir.exists():
                    for file_path in predictions_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run2_predictions"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"파일 로드 실패 {file_path}: {e}")
                
                # run3 이상감지 결과 로드
                anomaly_dir = self.paths.get_anomaly_detection_path()
                if anomaly_dir.exists():
                    for file_path in anomaly_dir.glob("*.json"):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results["run3_anomaly"][file_path.stem] = data
                        except Exception as e:
                            self.logger.warning(f"파일 로드 실패 {file_path}: {e}")
                
                # 메타데이터 추가
                results["metadata"] = {
                    "load_timestamp": datetime.now().isoformat(),
                    "run1_files": len(results["run1_analysis"]),
                    "run2_files": len(results["run2_predictions"]),
                    "run3_files": len(results["run3_anomaly"]),
                }
            
            self.logger.info(f"✅ 이전 결과 로드 완료: "
                           f"run1({len(results['run1_analysis'])}), "
                           f"run2({len(results['run2_predictions'])}), "
                           f"run3({len(results['run3_anomaly'])})")
            
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

    def prepare_time_series_data(self, all_results: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        이전 단계 결과들을 시계열 데이터로 변환합니다.
        
        Args:
            all_results: 이전 단계 결과들
            
        Returns:
            Tuple[np.ndarray, List[str]]: 시계열 데이터와 특성명 리스트
        """
        try:
            self.logger.info("시계열 데이터 준비 시작...")
            
            with self.performance_monitor.track("prepare_time_series"):
                features_list = []
                feature_names = []
                
                # run1 분석 결과에서 특성 추출
                for name, data in all_results["run1_analysis"].items():
                    if "unified_analysis" in name and "comprehensive_scores" in data:
                        scores = data["comprehensive_scores"]
                        for num in range(1, 46):
                            features_list.append(scores.get(str(num), 0.0))
                            feature_names.append(f"unified_score_{num}")
                    
                    elif "trend_analyzer_v2_results" in name and "comprehensive_trend_score" in data:
                        trend_scores = data["comprehensive_trend_score"]
                        for num in range(1, 46):
                            features_list.append(trend_scores.get(str(num), 0.0))
                            feature_names.append(f"trend_score_{num}")
                
                # run2 예측 결과에서 특성 추출
                for name, data in all_results["run2_predictions"].items():
                    if "ml_predictions" in name and "predictions" in data:
                        predictions = data["predictions"]
                        for num in range(1, 46):
                            features_list.append(predictions.get(str(num), 0.0))
                            feature_names.append(f"ml_prediction_{num}")
                
                # run3 이상감지 결과에서 특성 추출
                for name, data in all_results["run3_anomaly"].items():
                    if "anomaly_detection" in name and "anomaly_scores" in data:
                        anomaly_scores = data["anomaly_scores"]
                        for num in range(1, 46):
                            features_list.append(anomaly_scores.get(str(num), 0.0))
                            feature_names.append(f"anomaly_score_{num}")
                
                # 특성이 없는 경우 더미 데이터 생성
                if not features_list:
                    self.logger.warning("이전 결과가 없어 더미 시계열 데이터를 생성합니다.")
                    features_list = np.random.random(135).tolist()  # 45*3
                    feature_names = [f"dummy_feature_{i}" for i in range(135)]
                
                # 시계열 데이터 구성
                features_array = np.array(features_list)
                if features_array.ndim == 1:
                    # 1차원 배열을 2차원으로 변환 (1 x features)
                    time_series_data = features_array.reshape(1, -1)
                else:
                    time_series_data = features_array
                
                # 데이터 정규화
                if time_series_data.shape[0] > 1:
                    time_series_data = self.feature_scaler.fit_transform(time_series_data)
                else:
                    # 단일 샘플인 경우 정규화 스킵
                    pass
                
                self.logger.info(f"✅ 시계열 데이터 준비 완료: {time_series_data.shape}")
                return time_series_data, feature_names
                
        except Exception as e:
            self.logger.error(f"시계열 데이터 준비 실패: {e}")
            # 비상용 더미 데이터 반환
            dummy_data = np.random.random((10, 135))
            dummy_names = [f"feature_{i}" for i in range(135)]
            return dummy_data, dummy_names

    def analyze_trends_v2(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        TrendAnalyzerV2를 사용하여 트렌드를 분석합니다.
        
        Args:
            historical_data: 과거 로또 데이터
            
        Returns:
            Dict[str, Any]: 트렌드 분석 결과
        """
        try:
            self.logger.info("TrendAnalyzerV2 트렌드 분석 시작...")
            
            with self.performance_monitor.track("trend_analysis_v2"):
                if not historical_data:
                    self.logger.warning("과거 데이터가 없어 빈 트렌드 분석 결과를 반환합니다.")
                    return {"error": "no_historical_data"}
                
                trend_results = self.trend_analyzer.analyze(historical_data)
            
            self.logger.info("✅ TrendAnalyzerV2 분석 완료")
            return trend_results
            
        except Exception as e:
            self.logger.error(f"TrendAnalyzerV2 분석 실패: {e}")
            return {"error": str(e)}

    def create_tcn_sequences(self, time_series_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        TCN을 위한 시퀀스 데이터를 생성합니다.
        
        Args:
            time_series_data: 시계열 데이터
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_sequences, y_targets)
        """
        try:
            sequence_length = self.trend_config["sequence_length"]
            
            if len(time_series_data) < sequence_length:
                self.logger.warning(f"데이터 길이({len(time_series_data)})가 시퀀스 길이({sequence_length})보다 작습니다.")
                # 패딩으로 데이터 확장
                padded_data = np.pad(
                    time_series_data,
                    ((sequence_length - len(time_series_data), 0), (0, 0)),
                    mode='edge'
                )
                time_series_data = padded_data
            
            X_sequences = []
            y_targets = []
            
            for i in range(len(time_series_data) - sequence_length):
                # 입력 시퀀스
                X_sequences.append(time_series_data[i:i+sequence_length])
                
                # 타겟: 다음 시점의 첫 45개 특성 (로또 번호 점수)
                y_targets.append(time_series_data[i+sequence_length, :45])
            
            if not X_sequences:
                # 시퀀스가 생성되지 않은 경우 더미 데이터
                X_sequences = [time_series_data[-sequence_length:]]
                y_targets = [np.random.random(45)]
            
            return np.array(X_sequences), np.array(y_targets)
            
        except Exception as e:
            self.logger.error(f"TCN 시퀀스 생성 실패: {e}")
            # 비상용 더미 시퀀스
            dummy_X = np.random.random((1, sequence_length, time_series_data.shape[1]))
            dummy_y = np.random.random((1, 45))
            return dummy_X, dummy_y

    def train_tcn_model(self, X_sequences: np.ndarray, y_targets: np.ndarray) -> Dict[str, Any]:
        """
        TCN 모델을 훈련합니다.
        
        Args:
            X_sequences: 입력 시퀀스
            y_targets: 타겟 데이터
            
        Returns:
            Dict[str, Any]: 훈련 결과
        """
        try:
            self.logger.info("TCN 모델 훈련 시작...")
            
            training_results = {}
            
            if self.cuda_optimizer.is_cuda_available():
                with self.cuda_optimizer.gpu_memory_scope(size_mb=1024, device_id=0):
                    with self.performance_monitor.track("tcn_training"):
                        training_results = self._execute_tcn_training(X_sequences, y_targets)
            else:
                with self.performance_monitor.track("tcn_training"):
                    training_results = self._execute_tcn_training(X_sequences, y_targets)
            
            # 모델 저장
            model_path = self.model_dir / "trend_correction_tcn.pt"
            save_success = self.tcn_model.save(str(model_path))
            training_results["model_saved"] = save_success
            training_results["model_path"] = str(model_path)
            
            self.logger.info("✅ TCN 모델 훈련 완료")
            return training_results
            
        except Exception as e:
            self.logger.error(f"TCN 모델 훈련 실패: {e}")
            return {"error": str(e), "training_completed": False}

    def _execute_tcn_training(self, X_sequences: np.ndarray, y_targets: np.ndarray) -> Dict[str, Any]:
        """TCN 훈련 실행"""
        try:
            # TCN 전처리 적용
            X_processed, preprocess_metadata = self.tcn_preprocessor.preprocess_for_tcn(X_sequences)
            
            # 타겟 데이터 정규화
            y_normalized = self.target_scaler.fit_transform(y_targets)
            
            # TCN 모델 훈련
            training_results = self.tcn_model.fit(X_processed, y_normalized)
            
            # 메타데이터 추가
            training_results["preprocess_metadata"] = preprocess_metadata
            training_results["target_scaler_params"] = {
                "scale_": self.target_scaler.scale_.tolist() if hasattr(self.target_scaler, 'scale_') else None,
                "min_": self.target_scaler.min_.tolist() if hasattr(self.target_scaler, 'min_') else None,
            }
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"TCN 훈련 실행 실패: {e}")
            return {"error": str(e)}

    def predict_trend_corrections(self, recent_data: np.ndarray) -> np.ndarray:
        """
        최근 데이터를 기반으로 트렌드 보정을 예측합니다.
        
        Args:
            recent_data: 최근 시계열 데이터
            
        Returns:
            np.ndarray: 트렌드 보정 예측값
        """
        try:
            self.logger.info("트렌드 보정 예측 시작...")
            
            with self.performance_monitor.track("trend_prediction"):
                # 시퀀스 길이에 맞게 데이터 준비
                sequence_length = self.trend_config["sequence_length"]
                
                if len(recent_data) < sequence_length:
                    # 패딩으로 시퀀스 길이 맞춤
                    padded_data = np.pad(
                        recent_data,
                        ((sequence_length - len(recent_data), 0), (0, 0)),
                        mode='edge'
                    )
                    prediction_input = padded_data[-sequence_length:].reshape(1, sequence_length, -1)
                else:
                    prediction_input = recent_data[-sequence_length:].reshape(1, sequence_length, -1)
                
                # TCN 전처리 적용
                X_processed = self.tcn_preprocessor.transform_new_data(prediction_input)
                
                # TCN 예측
                normalized_predictions = self.tcn_model.predict(X_processed)
                
                # 정규화 해제
                if hasattr(self.target_scaler, 'scale_') and normalized_predictions.ndim == 2:
                    trend_corrections = self.target_scaler.inverse_transform(normalized_predictions)
                else:
                    trend_corrections = normalized_predictions
                
                # 1차원으로 변환
                if trend_corrections.ndim > 1:
                    trend_corrections = trend_corrections.flatten()[:45]  # 45개 번호만
                
                # 범위 조정 (-1 ~ 1)
                trend_corrections = np.clip(trend_corrections, -1, 1)
            
            self.logger.info("✅ 트렌드 보정 예측 완료")
            return trend_corrections
            
        except Exception as e:
            self.logger.error(f"트렌드 보정 예측 실패: {e}")
            # 비상용 중립 보정값 반환
            return np.zeros(45)

    def apply_trend_corrections(self, base_scores: np.ndarray, trend_corrections: np.ndarray) -> np.ndarray:
        """
        기본 점수에 트렌드 보정을 적용합니다.
        
        Args:
            base_scores: 기본 점수 (45개 번호)
            trend_corrections: 트렌드 보정값 (45개 번호)
            
        Returns:
            np.ndarray: 보정된 점수
        """
        try:
            self.logger.info("트렌드 보정 적용 시작...")
            
            # 적응적 보정 가중치 계산
            if self.trend_config["adaptive_weight"]:
                correction_weight = self._calculate_adaptive_weight(trend_corrections)
            else:
                correction_weight = self.trend_config["correction_weight"]
            
            # 트렌드 보정 적용
            corrected_scores = base_scores + (trend_corrections * correction_weight)
            
            # 점수 정규화 (0-1 범위)
            corrected_scores = np.clip(corrected_scores, 0, 1)
            
            # 확률 분포로 정규화
            score_sum = np.sum(corrected_scores)
            if score_sum > 0:
                corrected_scores = corrected_scores / score_sum
            else:
                # 모든 점수가 0인 경우 균등 분포
                corrected_scores = np.ones(45) / 45
            
            self.logger.info(f"✅ 트렌드 보정 적용 완료 (보정 가중치: {correction_weight:.3f})")
            return corrected_scores
            
        except Exception as e:
            self.logger.error(f"트렌드 보정 적용 실패: {e}")
            # 기본 점수 반환
            return base_scores

    def _calculate_adaptive_weight(self, trend_corrections: np.ndarray) -> float:
        """적응적 보정 가중치 계산"""
        try:
            # 트렌드 강도 측정
            trend_strength = np.std(trend_corrections)
            
            # 기본 가중치
            base_weight = self.trend_config["correction_weight"]
            
            # 강도에 따른 가중치 조정
            if trend_strength > 0.5:
                adaptive_weight = base_weight * 1.5  # 강한 트렌드
            elif trend_strength > 0.2:
                adaptive_weight = base_weight * 1.0  # 보통 트렌드
            else:
                adaptive_weight = base_weight * 0.5  # 약한 트렌드
            
            # 범위 제한
            adaptive_weight = np.clip(adaptive_weight, 0.1, 0.8)
            
            return adaptive_weight
            
        except Exception:
            return self.trend_config["correction_weight"]

    def generate_base_scores(self, all_results: Dict[str, Any]) -> np.ndarray:
        """이전 단계 결과로부터 기본 점수를 생성합니다."""
        try:
            base_scores = np.zeros(45)
            weight_sum = 0
            
            # run1 통합 분석 점수 (가중치: 0.4)
            for name, data in all_results["run1_analysis"].items():
                if "unified_analysis" in name and "comprehensive_scores" in data:
                    scores = data["comprehensive_scores"]
                    for i in range(45):
                        base_scores[i] += scores.get(str(i+1), 0.0) * 0.4
                    weight_sum += 0.4
                    break
            
            # run2 ML 예측 점수 (가중치: 0.4)
            for name, data in all_results["run2_predictions"].items():
                if "ml_predictions" in name and "predictions" in data:
                    predictions = data["predictions"]
                    for i in range(45):
                        base_scores[i] += predictions.get(str(i+1), 0.0) * 0.4
                    weight_sum += 0.4
                    break
            
            # run3 이상감지 점수 (가중치: 0.2)
            for name, data in all_results["run3_anomaly"].items():
                if "anomaly_detection" in name and "anomaly_scores" in data:
                    anomaly_scores = data["anomaly_scores"]
                    for i in range(45):
                        # 이상 점수가 낮을수록 정상이므로 역수 사용
                        anomaly_score = anomaly_scores.get(str(i+1), 0.5)
                        base_scores[i] += (1.0 - anomaly_score) * 0.2
                    weight_sum += 0.2
                    break
            
            # 가중치 정규화
            if weight_sum > 0:
                base_scores = base_scores / weight_sum
            else:
                # 기본 점수가 없는 경우 균등 분포
                base_scores = np.ones(45) / 45
            
            # 확률 분포로 정규화
            base_scores = np.clip(base_scores, 0, 1)
            score_sum = np.sum(base_scores)
            if score_sum > 0:
                base_scores = base_scores / score_sum
            
            return base_scores
            
        except Exception as e:
            self.logger.error(f"기본 점수 생성 실패: {e}")
            return np.ones(45) / 45  # 균등 분포

    def save_trend_results(self, results: Dict[str, Any]) -> str:
        """
        트렌드 보정 결과를 저장합니다.
        
        Args:
            results: 저장할 결과
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_correction_results_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # 메타데이터 추가
            results["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "version": "TrendCorrectionEngine_v1.0",
                "config": self.config,
                "trend_config": self.trend_config,
                "device": str(self.device),
                "performance_stats": self.performance_monitor.get_performance_summary(),
            }
            
            # GPU 메모리 통계 추가
            if self.cuda_optimizer.is_cuda_available():
                results["metadata"]["gpu_memory_stats"] = self.cuda_optimizer.get_gpu_memory_pool().get_stats()
            
            # 결과 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ 트렌드 보정 결과 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"트렌드 보정 결과 저장 실패: {e}")
            raise

    def run_full_trend_correction(self) -> Dict[str, Any]:
        """
        전체 트렌드 보정 프로세스를 실행합니다.
        
        Returns:
            Dict[str, Any]: 트렌드 보정 결과
        """
        try:
            self.logger.info("🚀 트렌드 보정 시스템 시작")
            start_time = time.time()
            
            # 1. 이전 단계 결과 로드
            all_results = self.load_all_previous_results()
            
            # 2. 과거 로또 데이터 로드
            historical_data = self.load_historical_data()
            
            # 3. 시계열 데이터 준비
            time_series_data, feature_names = self.prepare_time_series_data(all_results)
            
            # 4. TrendAnalyzerV2 트렌드 분석
            trend_analysis = self.analyze_trends_v2(historical_data)
            
            # 5. TCN 시퀀스 생성
            X_sequences, y_targets = self.create_tcn_sequences(time_series_data)
            
            # 6. TCN 모델 훈련
            training_results = self.train_tcn_model(X_sequences, y_targets)
            
            # 7. 트렌드 보정 예측
            trend_corrections = self.predict_trend_corrections(time_series_data)
            
            # 8. 기본 점수 생성
            base_scores = self.generate_base_scores(all_results)
            
            # 9. 트렌드 보정 적용
            corrected_scores = self.apply_trend_corrections(base_scores, trend_corrections)
            
            # 10. 결과 통합
            results = {
                "trend_correction_scores": {
                    str(i+1): float(corrected_scores[i]) for i in range(45)
                },
                "base_scores": {
                    str(i+1): float(base_scores[i]) for i in range(45)
                },
                "trend_corrections": {
                    str(i+1): float(trend_corrections[i]) for i in range(45)
                },
                "trend_analysis": trend_analysis,
                "training_results": training_results,
                "feature_names": feature_names,
                "data_summary": {
                    "time_series_shape": time_series_data.shape,
                    "sequence_count": len(X_sequences),
                    "historical_data_count": len(historical_data),
                },
                "correction_config": self.trend_config,
            }
            
            # 11. 결과 저장
            saved_path = self.save_trend_results(results)
            results["saved_path"] = saved_path
            
            # 12. 성능 통계
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
            
            self.logger.info(f"✅ 트렌드 보정 시스템 완료 (소요시간: {execution_time:.2f}초)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"트렌드 보정 시스템 실행 실패: {e}")
            raise


def main():
    """메인 실행 함수"""
    # 1. 의존성 설정
    configure_dependencies()

    logger.info("=" * 80)
    logger.info("🚀 4단계: 트렌드 보정 시스템 시작")
    logger.info("=" * 80)
    
    start_time = time.time()

    try:
        # TrendCorrectionEngine 인스턴스 생성 (설정 제거)
        engine = TrendCorrectionEngine()
        
        # 전체 파이프라인 실행
        final_results = engine.run_full_trend_correction()

        total_time = time.time() - start_time
        logger.info(f"✅ 트렌드 보정 완료! 총 실행 시간: {total_time:.2f}초")
        logger.info(f"📁 최종 결과 파일: {final_results.get('saved_path')}")

    except Exception as e:
        logger.error(f"❌ 트렌드 보정 실패: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    main()
