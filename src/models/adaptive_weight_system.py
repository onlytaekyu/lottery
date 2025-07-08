"""
적응형 가중치 시스템 (Adaptive Weight System)

성과 기반 가중치 적응과 메타 러닝을 통한 동적 전략 가중치 조정 시스템입니다.

주요 기능:
- 실시간 성과 기반 가중치 조정
- 메타 러닝 기반 가중치 최적화 (GPU 가속)
- 다중 목표 최적화 (ROI, 적중률, 리스크)
- 시간 가중 성과 평가
- 적응형 학습률 조정
- 가중치 안정성 보장
- GPU 기반 특성 추출 및 모델 학습

✅ v2.0 업데이트: src/utils 통합 시스템 적용
- 통합 메모리 관리자 
- 비동기 처리 지원
- 스마트 캐시 시스템
- 병렬 특성 추출
- GPU 최적화 강화
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import math
from scipy.optimize import minimize
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ✅ src/utils 통합 시스템 활용
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_config
from ..utils.cache_manager import UnifiedCachePathManager
from ..utils.unified_memory_manager import get_unified_memory_manager, MemoryConfig
from ..utils import (
    get_cuda_optimizer,
    get_enhanced_process_pool,
    get_unified_async_manager,
    get_pattern_filter
)
from ..monitoring.performance_tracker import PerformanceMetrics

logger = get_logger(__name__)


@dataclass
class WeightUpdateConfig:
    """가중치 업데이트 설정"""

    learning_rate: float = 0.01  # 학습률
    momentum: float = 0.9  # 모멘텀
    decay_rate: float = 0.95  # 시간 감쇠율
    min_weight: float = 0.05  # 최소 가중치
    max_weight: float = 0.6  # 최대 가중치
    stability_threshold: float = 0.1  # 안정성 임계값
    adaptation_window: int = 20  # 적응 윈도우 크기
    performance_memory: int = 100  # 성능 메모리 크기
    multi_objective_weights: Dict[str, float] = None  # 다중 목표 가중치
    use_gpu: bool = True  # GPU 사용 여부
    batch_size: int = 64  # 배치 크기
    memory_limit: float = 0.8  # 메모리 사용 제한
    # ✅ 새로운 최적화 설정
    enable_async_processing: bool = True  # 비동기 처리 활성화
    use_smart_caching: bool = True  # 스마트 캐시 활성화
    parallel_workers: int = 4  # 병렬 처리 워커 수
    cache_ttl: int = 3600  # 캐시 TTL (초)


@dataclass
class StrategyWeight:
    """전략 가중치 정보"""

    strategy: str
    current_weight: float
    target_weight: float
    momentum: float
    performance_score: float
    stability_score: float
    confidence: float
    last_update: datetime
    update_count: int


@dataclass
class WeightOptimizationResult:
    """가중치 최적화 결과"""

    optimized_weights: Dict[str, float]
    expected_performance: float
    optimization_score: float
    convergence_iterations: int
    stability_metrics: Dict[str, float]
    recommendations: List[str]


class GPUAcceleratedMetaLearner:
    """
    🚀 GPU 가속 메타 러닝 시스템 (v2.0)
    
    src/utils 통합 시스템 기반 고성능 메타 러닝:
    - 통합 메모리 관리
    - 비동기 처리 지원
    - 스마트 캐시 시스템
    - 병렬 특성 추출
    - GPU 최적화 강화
    """

    def __init__(self, config: WeightUpdateConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # ✅ src/utils 통합 시스템 초기화
        try:
            self.memory_mgr = get_unified_memory_manager()
            self.cuda_opt = get_cuda_optimizer()
            self.process_pool = get_enhanced_process_pool()
            self.async_mgr = get_unified_async_manager()
            self.pattern_filter = get_pattern_filter()
            self._unified_system_available = True
            self.logger.info("✅ 통합 시스템 초기화 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ 통합 시스템 초기화 실패, 기본 모드로 폴백: {e}")
            self._unified_system_available = False
            self._init_fallback_systems()

        # GPU 사용 설정
        self.use_gpu = config.use_gpu
        self.meta_model = None
        self.using_gpu = False
        self._initialize_model()

        # ✅ 스마트 캐시 시스템 
        if config.use_smart_caching and self._unified_system_available:
            self.smart_cache = True
            self.cache_ttl = config.cache_ttl
            self.feature_cache = {}  # 스마트 캐시로 관리
        else:
            # 기존 캐시 시스템 폴백
            self.smart_cache = False
            self.feature_cache = {}
            self.cache_max_size = 1000

        # ✅ 병렬 처리 설정
        self.parallel_workers = config.parallel_workers
        self.enable_async = config.enable_async_processing

        # 학습 데이터 저장소 (메모리 효율적)
        self.training_data = deque(maxlen=config.performance_memory)
        self.model_trained = False

        # ✅ 벡터화된 특성 추출기 (성능 최적화)
        self.feature_extractors = {
            "performance_trend": self._extract_performance_trend_vectorized,
            "volatility_features": self._extract_volatility_features_vectorized,
            "correlation_features": self._extract_correlation_features_vectorized,
            "temporal_features": self._extract_temporal_features_vectorized,
        }
        
        # 특성 크기 미리 계산
        self.feature_sizes = {
            "performance_trend": 10,
            "volatility_features": 10,
            "correlation_features": 10,
            "temporal_features": 10,
        }
        self.total_feature_size = sum(self.feature_sizes.values())

    def _init_fallback_systems(self):
        """폴백 시스템 초기화"""
        # 기본 메모리 관리자
        memory_config = MemoryConfig(
            max_memory_usage=self.config.memory_limit,
            use_memory_pooling=True,
            pool_size=32,
            auto_cleanup_interval=30.0,
        )
        self.memory_manager = get_unified_memory_manager()
        
        # 기본 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)

    def _initialize_model(self):
        """
        모델 초기화 (통합 시스템 GPU 최적화 적용)
        """
        try:
            if self.use_gpu:
                # ✅ CUDA 최적화기 활용
                if self._unified_system_available and self.cuda_opt:
                    self.cuda_opt.set_tf32_enabled(True)
                    self.cuda_opt.set_memory_pool_enabled(True)
                    self.logger.info("🚀 CUDA 최적화 활성화")
                
                # cuML 시도
                try:
                    from cuml.ensemble import RandomForestRegressor as CuMLRandomForestRegressor  # type: ignore
                    self.meta_model = CuMLRandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_streams=4,  # GPU 스트림 수
                    )
                    self.logger.info("✅ GPU 가속 cuML RandomForest 초기화 완료")
                    self.using_gpu = True
                except ImportError:
                    self.logger.warning("cuML 없음, scikit-learn으로 fallback")
                    self._initialize_cpu_model()
                    self.using_gpu = False
            else:
                self._initialize_cpu_model()
                self.using_gpu = False
                
        except Exception as e:
            self.logger.error(f"모델 초기화 실패: {e}")
            self._initialize_cpu_model()
            self.using_gpu = False

    def _initialize_cpu_model(self):
        """CPU 모델 초기화"""
        from sklearn.ensemble import RandomForestRegressor
        self.meta_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        )

    async def extract_features_batch_async(
        self, 
        performance_histories: List[Dict[str, List[PerformanceMetrics]]]
    ) -> np.ndarray:
        """
        🚀 비동기 배치 특성 추출 (통합 시스템 활용)
        """
        if not self.enable_async or not self._unified_system_available:
            return self.extract_features_batch(performance_histories)
        
        try:
            batch_size = len(performance_histories)
            features_batch = np.zeros((batch_size, self.total_feature_size))
            
            # ✅ 비동기 처리로 병렬 특성 추출
            async with self.async_mgr.semaphore(self.parallel_workers):
                tasks = []
                for batch_idx, performance_history in enumerate(performance_histories):
                    task = self._extract_features_async_worker(
                        performance_history, batch_idx
                    )
                    tasks.append(task)
                
                # 모든 태스크 완료 대기
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 결과 수집
                for batch_idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.warning(f"배치 {batch_idx} 특성 추출 실패: {result}")
                        features_batch[batch_idx] = np.zeros(self.total_feature_size)
                    else:
                        features_batch[batch_idx] = result

            return features_batch

        except Exception as e:
            self.logger.error(f"비동기 배치 특성 추출 실패: {e}")
            # 동기 방식 폴백
            return self.extract_features_batch(performance_histories)

    async def _extract_features_async_worker(
        self, 
        performance_history: Dict[str, List[PerformanceMetrics]], 
        batch_idx: int
    ) -> np.ndarray:
        """비동기 특성 추출 워커"""
        try:
            # ✅ 스마트 캐시 확인
            if self.smart_cache:
                cache_key = await self._get_cache_key_async(performance_history)
                if cache_key in self.feature_cache:
                    # 캐시 만료 확인
                    cached_data = self.feature_cache[cache_key]
                    if self._is_cache_valid(cached_data):
                        return cached_data['features']
            
            # 특성 추출
            feature_vector = await self._extract_features_async(performance_history)
            
            # ✅ 스마트 캐시 저장
            if self.smart_cache:
                cache_data = {
                    'features': feature_vector,
                    'timestamp': datetime.now().timestamp(),
                    'ttl': self.cache_ttl
                }
                self.feature_cache[cache_key] = cache_data
                
                # 캐시 크기 관리
                await self._manage_cache_size()
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"비동기 특성 추출 워커 실패: {e}")
            return np.zeros(self.total_feature_size)

    def extract_features_batch(
        self, 
        performance_histories: List[Dict[str, List[PerformanceMetrics]]]
    ) -> np.ndarray:
        """
        배치 단위 특성 추출 (개선된 동기 버전)
        """
        try:
            batch_size = len(performance_histories)
            features_batch = np.zeros((batch_size, self.total_feature_size))
            
            # ✅ 통합 메모리 관리자로 메모리 할당
            if self._unified_system_available and self.memory_mgr:
                with self.memory_mgr.temporary_allocation(
                    size=batch_size * self.total_feature_size * 8,  # float64 기준
                    prefer_device="cpu"
                ) as work_mem:
                    # 배치 단위로 특성 추출
                    for batch_idx, performance_history in enumerate(performance_histories):
                        features_batch[batch_idx] = self._extract_features_with_cache(
                            performance_history
                        )
            else:
                # 폴백: 기본 방식
                for batch_idx, performance_history in enumerate(performance_histories):
                    features_batch[batch_idx] = self.extract_features(performance_history)

            return features_batch

        except Exception as e:
            self.logger.error(f"배치 특성 추출 실패: {e}")
            return np.zeros((len(performance_histories), self.total_feature_size))

    def _extract_features_with_cache(
        self, 
        performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> np.ndarray:
        """캐시를 활용한 특성 추출"""
        try:
            # 캐시 키 생성
            cache_key = self._get_cache_key(performance_history)
            
            # 캐시 확인
            if cache_key in self.feature_cache:
                if self.smart_cache:
                    cached_data = self.feature_cache[cache_key]
                    if self._is_cache_valid(cached_data):
                        return cached_data['features']
                else:
                    return self.feature_cache[cache_key]
            
            # 특성 추출
            feature_vector = self.extract_features(performance_history)
            
            # 캐시 저장
            if self.smart_cache:
                cache_data = {
                    'features': feature_vector,
                    'timestamp': datetime.now().timestamp(),
                    'ttl': self.cache_ttl
                }
                self.feature_cache[cache_key] = cache_data
            else:
                # 기본 캐시 (크기 제한)
                if len(self.feature_cache) < self.cache_max_size:
                    self.feature_cache[cache_key] = feature_vector
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"캐시 특성 추출 실패: {e}")
            return np.zeros(self.total_feature_size)

    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """캐시 유효성 검사"""
        try:
            current_time = datetime.now().timestamp()
            cache_time = cached_data.get('timestamp', 0)
            ttl = cached_data.get('ttl', self.cache_ttl)
            
            return (current_time - cache_time) < ttl
            
        except Exception:
            return False

    async def _manage_cache_size(self):
        """캐시 크기 관리 (비동기)"""
        try:
            if len(self.feature_cache) > self.cache_max_size * 1.5:
                # 오래된 캐시 항목 제거
                datetime.now().timestamp()
                expired_keys = []
                
                for key, cached_data in self.feature_cache.items():
                    if not self._is_cache_valid(cached_data):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.feature_cache[key]
                
                self.logger.info(f"만료된 캐시 {len(expired_keys)}개 제거")
                
        except Exception as e:
            self.logger.error(f"캐시 관리 실패: {e}")

    async def _get_cache_key_async(self, performance_history: Dict[str, List[PerformanceMetrics]]) -> str:
        """비동기 캐시 키 생성"""
        return self._get_cache_key(performance_history)

    async def _extract_features_async(
        self, 
        performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> np.ndarray:
        """비동기 특성 추출"""
        # CPU 집약적 작업을 스레드 풀에서 실행
        if self._unified_system_available and self.process_pool:
            return await self.process_pool.run_in_thread(
                self.extract_features, performance_history
            )
        else:
            return self.extract_features(performance_history)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        🚀 성능 통계 반환 (통합 시스템 정보 포함)
        """
        stats = {
            "model_type": "GPUAcceleratedMetaLearner",
            "using_gpu": self.using_gpu,
            "unified_system_available": self._unified_system_available,
            "smart_cache_enabled": self.smart_cache,
            "async_processing_enabled": self.enable_async,
            "parallel_workers": self.parallel_workers,
            "cache_size": len(self.feature_cache),
            "model_trained": self.model_trained,
            "feature_size": self.total_feature_size
        }
        
        # 통합 시스템 통계
        if self._unified_system_available:
            if self.memory_mgr:
                try:
                    stats["memory_performance"] = self.memory_mgr.get_performance_metrics()
                except Exception as e:
                    self.logger.debug(f"메모리 성능 통계 조회 실패: {e}")
            
            if self.cuda_opt:
                try:
                    stats["cuda_optimization"] = self.cuda_opt.get_optimization_stats()
                except Exception as e:
                    self.logger.debug(f"CUDA 최적화 통계 조회 실패: {e}")
        
        return stats

    def optimize_memory_usage(self):
        """
        🚀 메모리 사용량 최적화
        """
        if self._unified_system_available and self.memory_mgr:
            self.memory_mgr.cleanup_unused_memory()
            self.logger.info("🧹 통합 메모리 최적화 완료")
        
        # 캐시 정리
        if self.smart_cache:
            asyncio.create_task(self._manage_cache_size())
        else:
            # 기본 캐시 정리
            if len(self.feature_cache) > self.cache_max_size:
                self.feature_cache.clear()
                self.logger.info("캐시 정리 완료")

    def get_optimal_batch_size(self, data_size: int) -> int:
        """
        🚀 최적 배치 크기 계산
        """
        if self._unified_system_available and self.memory_mgr:
            try:
                memory_stats = self.memory_mgr.get_memory_status()
                cpu_util = memory_stats.get("cpu_utilization", 0.5)
                
                # CPU 사용률에 따른 동적 조정
                if cpu_util < 0.3:
                    base_batch = 128
                elif cpu_util < 0.7:
                    base_batch = 64
                else:
                    base_batch = 32
                
                return min(max(base_batch, 1), data_size)
                
            except Exception as e:
                self.logger.debug(f"최적 배치 크기 계산 실패: {e}")
        
        # 폴백: 기본 배치 크기
        return min(self.config.batch_size, data_size)

    def _get_cache_key(self, performance_history: Dict[str, List[PerformanceMetrics]]) -> str:
        """캐시 키 생성"""
        try:
            # 성능 데이터의 해시 생성
            key_data = []
            for strategy, metrics_list in performance_history.items():
                if metrics_list:
                    latest_metric = metrics_list[-1]
                    key_data.append(f"{strategy}_{latest_metric.roi:.4f}_{latest_metric.win_rate:.4f}")
            return "_".join(sorted(key_data))
        except:
            return f"cache_{hash(str(performance_history)) % 10000}"

    def extract_features(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> np.ndarray:
        """성능 이력에서 특성 추출 (벡터화 최적화)"""
        try:
            features = np.zeros(self.total_feature_size)
            feature_idx = 0

            for extractor_name, extractor_func in self.feature_extractors.items():
                try:
                    feature_size = self.feature_sizes[extractor_name]
                    feature_vector = extractor_func(performance_history)
                    
                    # 크기 맞춤
                    if len(feature_vector) > feature_size:
                        feature_vector = feature_vector[:feature_size]
                    elif len(feature_vector) < feature_size:
                        feature_vector = np.pad(feature_vector, (0, feature_size - len(feature_vector)))
                    
                    features[feature_idx:feature_idx + feature_size] = feature_vector
                    feature_idx += feature_size
                    
                except Exception as e:
                    self.logger.warning(f"특성 추출 실패 ({extractor_name}): {e}")
                    # 기본값으로 채움
                    feature_idx += self.feature_sizes[extractor_name]

            return features

        except Exception as e:
            self.logger.error(f"특성 추출 중 오류: {e}")
            return np.zeros(self.total_feature_size)

    def _extract_performance_trend_vectorized(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> np.ndarray:
        """성능 트렌드 특성 추출 (벡터화)"""
        try:
            features = np.zeros(10)

            # 모든 전략의 데이터를 배열로 변환
            roi_data = []
            win_rate_data = []
            
            for strategy, metrics_list in performance_history.items():
                if len(metrics_list) >= 2:
                    recent_metrics = metrics_list[-10:]
                    roi_values = np.array([m.roi for m in recent_metrics])
                    win_rate_values = np.array([m.win_rate for m in recent_metrics])
                    
                    roi_data.append(roi_values)
                    win_rate_data.append(win_rate_values)

            if roi_data and win_rate_data:
                # 벡터화된 트렌드 계산
                roi_array = np.array(roi_data)
                win_rate_array = np.array(win_rate_data)
                
                # 전체 트렌드 (평균)
                if roi_array.size > 0:
                    roi_trend = np.mean([np.polyfit(range(len(vals)), vals, 1)[0] 
                                       for vals in roi_array if len(vals) > 1])
                    features[0] = roi_trend if not np.isnan(roi_trend) else 0.0
                
                if win_rate_array.size > 0:
                    win_rate_trend = np.mean([np.polyfit(range(len(vals)), vals, 1)[0] 
                                            for vals in win_rate_array if len(vals) > 1])
                    features[1] = win_rate_trend if not np.isnan(win_rate_trend) else 0.0
                
                # 변동성 메트릭
                features[2] = np.mean([np.std(vals) for vals in roi_array if len(vals) > 1])
                features[3] = np.mean([np.std(vals) for vals in win_rate_array if len(vals) > 1])
                
                # 최근 성과
                features[4] = np.mean([vals[-1] for vals in roi_array if len(vals) > 0])
                features[5] = np.mean([vals[-1] for vals in win_rate_array if len(vals) > 0])

            return features

        except Exception as e:
            self.logger.error(f"성능 트렌드 특성 추출 실패: {e}")
            return np.zeros(10)

    def _extract_volatility_features_vectorized(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> np.ndarray:
        """변동성 특성 추출 (벡터화)"""
        try:
            features = np.zeros(10)
            
            all_roi_values = []
            all_sharpe_values = []
            all_volatility_values = []

            for strategy, metrics_list in performance_history.items():
                if len(metrics_list) >= 2:
                    recent_metrics = metrics_list[-10:]
                    roi_values = [m.roi for m in recent_metrics]
                    sharpe_values = [m.sharpe_ratio for m in recent_metrics]
                    
                    all_roi_values.extend(roi_values)
                    all_sharpe_values.extend(sharpe_values)
                    
                    # 개별 전략 변동성
                    if len(roi_values) > 1:
                        all_volatility_values.append(np.std(roi_values))

            if all_roi_values:
                # 전체 변동성 메트릭
                features[0] = np.std(all_roi_values)
                features[1] = np.mean(all_sharpe_values) if all_sharpe_values else 0.0
                features[2] = np.var(all_roi_values)
                features[3] = np.mean(all_volatility_values) if all_volatility_values else 0.0
                
                # 분위수 기반 특성
                features[4] = np.percentile(all_roi_values, 25)
                features[5] = np.percentile(all_roi_values, 75)
                features[6] = np.percentile(all_roi_values, 90) - np.percentile(all_roi_values, 10)

            return features

        except Exception as e:
            self.logger.error(f"변동성 특성 추출 실패: {e}")
            return np.zeros(10)

    def _extract_correlation_features_vectorized(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> np.ndarray:
        """상관관계 특성 추출 (벡터화)"""
        try:
            features = np.zeros(10)
            
            strategies = list(performance_history.keys())
            if len(strategies) < 2:
                return features

            # ROI 데이터 행렬 구성
            roi_matrix = []
            win_rate_matrix = []
            
            min_length = min(len(metrics_list) for metrics_list in performance_history.values())
            if min_length < 2:
                return features

            for strategy in strategies:
                metrics_list = performance_history[strategy][-min_length:]
                roi_values = [m.roi for m in metrics_list]
                win_rate_values = [m.win_rate for m in metrics_list]
                
                roi_matrix.append(roi_values)
                win_rate_matrix.append(win_rate_values)

            # 상관관계 행렬 계산
            roi_matrix = np.array(roi_matrix)
            win_rate_matrix = np.array(win_rate_matrix)
            
            if roi_matrix.shape[1] > 1:
                roi_corr = np.corrcoef(roi_matrix)
                win_rate_corr = np.corrcoef(win_rate_matrix)
                
                # 상관관계 특성
                features[0] = np.mean(roi_corr[np.triu_indices_from(roi_corr, k=1)])
                features[1] = np.std(roi_corr[np.triu_indices_from(roi_corr, k=1)])
                features[2] = np.mean(win_rate_corr[np.triu_indices_from(win_rate_corr, k=1)])
                features[3] = np.std(win_rate_corr[np.triu_indices_from(win_rate_corr, k=1)])
                
                # 다이버시티 메트릭
                features[4] = 1.0 - np.mean(np.abs(roi_corr[np.triu_indices_from(roi_corr, k=1)]))

            return features

        except Exception as e:
            self.logger.error(f"상관관계 특성 추출 실패: {e}")
            return np.zeros(10)

    def _extract_temporal_features_vectorized(
        self, performance_history: Dict[str, List[PerformanceMetrics]]
    ) -> np.ndarray:
        """시간적 특성 추출 (벡터화)"""
        try:
            features = np.zeros(10)
            
            current_time = datetime.now()
            all_timestamps = []
            all_performance_values = []

            for strategy, metrics_list in performance_history.items():
                for metric in metrics_list:
                    all_timestamps.append(metric.timestamp)
                    all_performance_values.append(metric.roi)

            if all_timestamps:
                # 시간 기반 특성
                time_diffs = [(current_time - ts).total_seconds() / 3600 for ts in all_timestamps]  # 시간 단위
                
                features[0] = np.mean(time_diffs)
                features[1] = np.std(time_diffs)
                features[2] = min(time_diffs) if time_diffs else 0
                features[3] = max(time_diffs) if time_diffs else 0
                
                # 시간 가중 성과
                weights = np.exp(-np.array(time_diffs) / 24)  # 24시간 반감기
                if len(all_performance_values) == len(weights):
                    features[4] = np.average(all_performance_values, weights=weights)

            return features

        except Exception as e:
            self.logger.error(f"시간적 특성 추출 실패: {e}")
            return np.zeros(10)

    def train_meta_model_gpu(self, training_data: List[Dict[str, Any]]) -> bool:
        """GPU 가속 메타 모델 훈련"""
        try:
            if not training_data:
                self.logger.warning("훈련 데이터가 없습니다.")
                return False

            self.logger.info(f"GPU 메타 모델 훈련 시작: {len(training_data)}개 샘플")

            # 특성과 타겟 준비
            features_list = []
            targets = []

            for data_point in training_data:
                performance_history = data_point["performance_history"]
                target_performance = data_point["target_performance"]
                
                # 배치 단위 특성 추출
                features = self.extract_features(performance_history)
                features_list.append(features)
                targets.append(target_performance)

            X = np.array(features_list)
            y = np.array(targets)

            if X.shape[0] == 0:
                self.logger.warning("특성 행렬이 비어있습니다.")
                return False

            # GPU 메모리 체크
            if self.using_gpu:
                try:
                    import cupy as cp  # type: ignore
                    memory_info = cp.cuda.runtime.memGetInfo()
                    available_memory = memory_info[0] / (1024**3)  # GB
                    required_memory = X.nbytes * 3 / (1024**3)  # 추정치
                    
                    if required_memory > available_memory * 0.8:
                        self.logger.warning(f"GPU 메모리 부족 예상, CPU로 fallback")
                        self._initialize_cpu_model()
                        self.using_gpu = False
                except:
                    pass

            # ✅ 메타 모델 존재 확인
            if self.meta_model is None:
                raise RuntimeError("메타 모델이 초기화되지 않았습니다")

            # 모델 훈련
            self.meta_model.fit(X, y)
            self.model_trained = True

            # 훈련 성과 평가
            if hasattr(self.meta_model, 'score'):
                score = self.meta_model.score(X, y)
                self.logger.info(f"메타 모델 훈련 완료: R² score = {score:.4f}")
            else:
                self.logger.info("메타 모델 훈련 완료")

            return True

        except Exception as e:
            self.logger.error(f"GPU 메타 모델 훈련 실패: {e}")
            return False

    def predict_performance_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """배치 단위 성능 예측 (GPU 가속)"""
        try:
            if not self.model_trained or self.meta_model is None:
                self.logger.warning("모델이 훈련되지 않았습니다.")
                return np.zeros(len(features_batch))

            predictions = self.meta_model.predict(features_batch)
            
            # GPU 텐서를 numpy로 변환 (cuML 사용 시)
            if self.using_gpu and hasattr(predictions, 'get'):
                predictions = predictions.get()

            return predictions

        except Exception as e:
            self.logger.error(f"배치 예측 실패: {e}")
            return np.zeros(len(features_batch))

    def get_feature_importance(self) -> Dict[str, float]:
        """특성 중요도 반환"""
        try:
            if not self.model_trained or self.meta_model is None:
                return {}

            importance = self.meta_model.feature_importances_
            
            # GPU 텐서를 numpy로 변환 (cuML 사용 시)
            if self.using_gpu and hasattr(importance, 'get'):
                importance = importance.get()

            # 특성 이름과 매핑
            feature_names = []
            for extractor_name, size in self.feature_sizes.items():
                for i in range(size):
                    feature_names.append(f"{extractor_name}_{i}")

            return dict(zip(feature_names, importance))

        except Exception as e:
            self.logger.error(f"특성 중요도 계산 실패: {e}")
            return {}

    def clear_cache(self):
        """캐시 정리"""
        self.feature_cache.clear()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 반환"""
        try:
            memory_info = {"cache_size": len(self.feature_cache)}
            
            if self.using_gpu:
                try:
                    import cupy as cp  # type: ignore
                    gpu_memory = cp.cuda.runtime.memGetInfo()
                    memory_info["gpu_total_gb"] = gpu_memory[1] / (1024**3)
                    memory_info["gpu_available_gb"] = gpu_memory[0] / (1024**3)
                    memory_info["gpu_used_gb"] = (gpu_memory[1] - gpu_memory[0]) / (1024**3)
                except:
                    pass
                    
            return memory_info
        except Exception as e:
            self.logger.error(f"메모리 사용량 조회 실패: {e}")
            return {}


class AdaptiveWeightSystem:
    """적응형 가중치 시스템 v2.0"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        AdaptiveWeightSystem 초기화

        Args:
            config: 설정 객체
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # 통합 시스템 초기화
        self.opt_config = self._setup_optimization_config()
        self.memory_manager = get_unified_memory_manager()
        self.cuda_optimizer = get_cuda_optimizer()
        self.process_pool = get_enhanced_process_pool()

        # 데이터 저장 설정
        paths = get_config()
        cache_path_manager = UnifiedCachePathManager(paths)
        self.cache_dir = cache_path_manager.get_path("adaptive_weights")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 전략별 가중치 및 성능 데이터
        self.strategy_weights: Dict[str, StrategyWeight] = {}
        self.performance_history: Dict[str, deque[PerformanceMetrics]] = defaultdict(
            lambda: deque(maxlen=self.opt_config.performance_memory)
        )

        # 최적화 이력
        self.optimization_history = []

        # 초기 가중치 설정
        self.initialize_weights()

        self.logger.info("적응형 가중치 시스템 초기화 완료")

    def _setup_optimization_config(self) -> WeightUpdateConfig:
        """가중치 업데이트 설정 초기화"""
        weight_config = self.config.get("weight_update", {})
        multi_obj_weights = weight_config.get(
            "multi_objective_weights",
            {"roi": 0.4, "win_rate": 0.3, "stability": 0.2, "risk": 0.1},
        )

        return WeightUpdateConfig(
            learning_rate=weight_config.get("learning_rate", 0.01),
            momentum=weight_config.get("momentum", 0.9),
            decay_rate=weight_config.get("decay_rate", 0.95),
            min_weight=weight_config.get("min_weight", 0.05),
            max_weight=weight_config.get("max_weight", 0.6),
            stability_threshold=weight_config.get("stability_threshold", 0.1),
            adaptation_window=weight_config.get("adaptation_window", 20),
            performance_memory=weight_config.get("performance_memory", 100),
            multi_objective_weights=multi_obj_weights,
            use_gpu=weight_config.get("use_gpu", True),
            batch_size=weight_config.get("batch_size", 64),
            memory_limit=weight_config.get("memory_limit", 0.8),
            enable_async_processing=weight_config.get("enable_async_processing", True),
            use_smart_caching=weight_config.get("use_smart_caching", True),
            parallel_workers=weight_config.get("parallel_workers", 4),
            cache_ttl=weight_config.get("cache_ttl", 3600),
        )

    def initialize_weights(self, strategies: Optional[List[str]] = None) -> None:
        """가중치 초기화"""
        try:
            if strategies is None:
                strategies = [
                    "frequency_based",
                    "cluster_analysis",
                    "trend_following",
                    "ai_ensemble",
                    "contrarian",
                ]

            # 균등 가중치로 초기화
            equal_weight = 1.0 / len(strategies)

            for strategy in strategies:
                self.strategy_weights[strategy] = StrategyWeight(
                    strategy=strategy,
                    current_weight=equal_weight,
                    target_weight=equal_weight,
                    momentum=0.0,
                    performance_score=0.0,
                    stability_score=1.0,
                    confidence=0.5,
                    last_update=datetime.now(),
                    update_count=0,
                )

            self.logger.info(f"가중치 초기화 완료: {len(strategies)}개 전략")

        except Exception as e:
            self.logger.error(f"가중치 초기화 중 오류: {str(e)}")

    def update_weights(
        self, performance_metrics: Dict[str, PerformanceMetrics]
    ) -> Dict[str, float]:
        """성과 기반 가중치 업데이트"""
        try:
            self.logger.info("성과 기반 가중치 업데이트 시작")

            # 성과 점수 계산
            performance_scores = self._calculate_performance_scores(performance_metrics)

            # 타겟 가중치 계산
            target_weights = self._calculate_target_weights(performance_scores)

            # 가중치 업데이트 (모멘텀 적용)
            updated_weights = self._apply_momentum_update(target_weights)

            # 안정성 검사
            stable_weights = self._ensure_weight_stability(updated_weights)

            # 가중치 정규화
            normalized_weights = self._normalize_weights(stable_weights)

            # 가중치 객체 업데이트
            self._update_weight_objects(normalized_weights, performance_scores)

            # 이력 저장
            self._save_weight_history(normalized_weights)

            self.logger.info("성과 기반 가중치 업데이트 완료")
            return normalized_weights

        except Exception as e:
            self.logger.error(f"가중치 업데이트 중 오류: {str(e)}")
            return self.get_current_weights()

    def _calculate_performance_scores(
        self, performance_metrics: Dict[str, PerformanceMetrics]
    ) -> Dict[str, float]:
        """성과 점수 계산"""
        try:
            scores = {}

            for strategy, metrics in performance_metrics.items():
                # 다중 목표 점수 계산
                roi_score = self._normalize_score(metrics.roi, -1.0, 1.0)
                win_rate_score = metrics.win_rate
                stability_score = (
                    1.0 / (1.0 + metrics.volatility) if metrics.volatility > 0 else 1.0
                )
                risk_score = 1.0 - min(1.0, max(0.0, metrics.max_drawdown))

                # 가중 평균 계산
                weighted_score = (
                    self.opt_config.multi_objective_weights["roi"] * roi_score
                    + self.opt_config.multi_objective_weights["win_rate"]
                    * win_rate_score
                    + self.opt_config.multi_objective_weights["stability"]
                    * stability_score
                    + self.opt_config.multi_objective_weights["risk"] * risk_score
                )

                # 시간 가중 적용
                time_weight = self._calculate_time_weight(metrics.timestamp)
                final_score = weighted_score * time_weight

                scores[strategy] = max(0.0, min(1.0, final_score))

            return scores

        except Exception as e:
            self.logger.error(f"성과 점수 계산 중 오류: {str(e)}")
            return {}

    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """점수 정규화"""
        try:
            if max_val == min_val:
                return 0.5

            normalized = (value - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))

        except Exception as e:
            return 0.5

    def _calculate_time_weight(self, timestamp: datetime) -> float:
        """시간 가중치 계산"""
        try:
            now = datetime.now()
            time_diff = (now - timestamp).total_seconds() / 3600  # 시간 단위

            # 지수 감쇠
            time_weight = math.exp(-time_diff * 0.1)  # 10시간 반감기

            return max(0.1, min(1.0, time_weight))

        except Exception as e:
            return 1.0

    def _calculate_target_weights(
        self, performance_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """타겟 가중치 계산"""
        try:
            if not performance_scores:
                return self.get_current_weights()

            # 소프트맥스 기반 가중치 계산
            scores = np.array(list(performance_scores.values()))

            # 온도 매개변수 적용 (과도한 집중 방지)
            temperature = 2.0
            exp_scores = np.exp(scores / temperature)
            softmax_weights = exp_scores / np.sum(exp_scores)

            # 최소/최대 가중치 제한 적용
            clipped_weights = np.clip(
                softmax_weights,
                self.opt_config.min_weight,
                self.opt_config.max_weight,
            )

            # 정규화
            normalized_weights = clipped_weights / np.sum(clipped_weights)

            # 딕셔너리로 변환
            target_weights = {}
            for i, strategy in enumerate(performance_scores.keys()):
                target_weights[strategy] = float(normalized_weights[i])

            return target_weights

        except Exception as e:
            self.logger.error(f"타겟 가중치 계산 중 오류: {str(e)}")
            return self.get_current_weights()

    def _apply_momentum_update(
        self, target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """모멘텀 적용 가중치 업데이트"""
        try:
            updated_weights = {}

            for strategy, target_weight in target_weights.items():
                if strategy in self.strategy_weights:
                    current_weight = self.strategy_weights[strategy].current_weight
                    current_momentum = self.strategy_weights[strategy].momentum

                    # 그래디언트 계산
                    gradient = target_weight - current_weight

                    # 모멘텀 업데이트
                    new_momentum = (
                        self.opt_config.momentum * current_momentum
                        + self.opt_config.learning_rate * gradient
                    )

                    # 새로운 가중치 계산
                    new_weight = current_weight + new_momentum

                    updated_weights[strategy] = new_weight

                    # 모멘텀 저장
                    self.strategy_weights[strategy].momentum = new_momentum
                else:
                    updated_weights[strategy] = target_weight

            return updated_weights

        except Exception as e:
            self.logger.error(f"모멘텀 업데이트 중 오류: {str(e)}")
            return target_weights

    def _ensure_weight_stability(self, weights: Dict[str, float]) -> Dict[str, float]:
        """가중치 안정성 보장"""
        try:
            stable_weights = {}

            for strategy, weight in weights.items():
                if strategy in self.strategy_weights:
                    current_weight = self.strategy_weights[strategy].current_weight

                    # 변화량 제한
                    max_change = self.opt_config.stability_threshold
                    weight_change = weight - current_weight

                    if abs(weight_change) > max_change:
                        # 변화량 제한 적용
                        limited_change = np.sign(weight_change) * max_change
                        stable_weight = current_weight + limited_change
                    else:
                        stable_weight = weight

                    stable_weights[strategy] = stable_weight
                else:
                    stable_weights[strategy] = weight

            return stable_weights

        except Exception as e:
            self.logger.error(f"가중치 안정성 보장 중 오류: {str(e)}")
            return weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """가중치 정규화"""
        try:
            total_weight = sum(weights.values())

            if total_weight == 0:
                # 균등 분배
                equal_weight = 1.0 / len(weights)
                return {strategy: equal_weight for strategy in weights.keys()}

            # 정규화
            normalized = {
                strategy: weight / total_weight for strategy, weight in weights.items()
            }

            # 최소/최대 가중치 제한 재적용
            clipped = {}
            for strategy, weight in normalized.items():
                clipped[strategy] = max(
                    self.opt_config.min_weight,
                    min(self.opt_config.max_weight, weight),
                )

            # 재정규화
            total_clipped = sum(clipped.values())
            if total_clipped > 0:
                final_weights = {
                    strategy: weight / total_clipped
                    for strategy, weight in clipped.items()
                }
            else:
                equal_weight = 1.0 / len(clipped)
                final_weights = {strategy: equal_weight for strategy in clipped.keys()}

            return final_weights

        except Exception as e:
            self.logger.error(f"가중치 정규화 중 오류: {str(e)}")
            return weights

    def _update_weight_objects(
        self, weights: Dict[str, float], performance_scores: Dict[str, float]
    ) -> None:
        """가중치 객체 업데이트"""
        try:
            current_time = datetime.now()

            for strategy, weight in weights.items():
                if strategy in self.strategy_weights:
                    weight_obj = self.strategy_weights[strategy]

                    # 가중치 업데이트
                    weight_obj.target_weight = weight
                    weight_obj.current_weight = weight
                    weight_obj.performance_score = performance_scores.get(strategy, 0.0)
                    weight_obj.last_update = current_time
                    weight_obj.update_count += 1

                    # 안정성 점수 계산
                    if len(self.weight_history[strategy]) > 1:
                        recent_weights = list(self.weight_history[strategy])[-5:]
                        weight_variance = (
                            np.var(recent_weights) if len(recent_weights) > 1 else 0
                        )
                        weight_obj.stability_score = 1.0 / (1.0 + weight_variance)

                    # 신뢰도 계산
                    weight_obj.confidence = min(1.0, weight_obj.update_count / 10.0)

        except Exception as e:
            self.logger.error(f"가중치 객체 업데이트 중 오류: {str(e)}")

    def _save_weight_history(self, weights: Dict[str, float]) -> None:
        """가중치 이력 저장"""
        try:
            for strategy, weight in weights.items():
                self.weight_history[strategy].append(weight)

                # 이력 크기 제한
                if (
                    len(self.weight_history[strategy])
                    > self.opt_config.performance_memory
                ):
                    self.weight_history[strategy].popleft()

        except Exception as e:
            self.logger.error(f"가중치 이력 저장 중 오류: {str(e)}")

    def optimize_weights_with_meta_learning(
        self,
        performance_history: Dict[str, List[PerformanceMetrics]],
        target_performance: float = 0.1,
    ) -> WeightOptimizationResult:
        """메타 러닝을 통한 가중치 최적화"""
        try:
            self.logger.info("메타 러닝 기반 가중치 최적화 시작")

            # 특성 추출
            features = self.meta_learner.extract_features(performance_history)

            # 메타 모델 예측
            predicted_performance = self.meta_learner.predict_performance_batch([features])[0]

            # 최적화 목적함수 정의
            def objective(weights):
                """가중치 최적화 목적함수"""
                try:
                    # 제약 조건 확인
                    if np.sum(weights) != 1.0 or np.any(
                        weights < self.opt_config.min_weight
                    ):
                        return 1e10

                    # 예상 성능 계산
                    expected_perf = 0
                    for i, strategy in enumerate(self.strategy_weights.keys()):
                        if (
                            strategy in performance_history
                            and performance_history[strategy]
                        ):
                            recent_metrics = performance_history[strategy][-1]
                            strategy_score = self._calculate_strategy_score(
                                recent_metrics
                            )
                            expected_perf += weights[i] * strategy_score

                    # 목표 성능과의 차이 + 다양성 패널티
                    performance_loss = abs(expected_perf - target_performance)
                    diversity_penalty = -np.sum(
                        weights * np.log(weights + 1e-10)
                    )  # 엔트로피

                    return performance_loss - 0.1 * diversity_penalty

                except Exception as e:
                    return 1e10

            # 제약 조건 설정
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

            bounds = [
                (self.opt_config.min_weight, self.opt_config.max_weight)
                for _ in range(len(self.strategy_weights))
            ]

            # 초기값 설정
            initial_weights = np.array(
                [w.current_weight for w in self.strategy_weights.values()]
            )

            # 최적화 실행
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    initial_weights,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000, "ftol": 1e-6},
                )

            # 결과 처리
            if result.success:
                optimized_weights = result.x
                optimization_score = -result.fun
                convergence_iterations = result.nit
            else:
                self.logger.warning("가중치 최적화 실패, 현재 가중치 유지")
                optimized_weights = initial_weights
                optimization_score = 0.0
                convergence_iterations = 0

            # 가중치 딕셔너리 생성
            weight_dict = {}
            for i, strategy in enumerate(self.strategy_weights.keys()):
                weight_dict[strategy] = float(optimized_weights[i])

            # 안정성 메트릭 계산
            stability_metrics = self._calculate_stability_metrics(weight_dict)

            # 추천사항 생성
            recommendations = self._generate_optimization_recommendations(
                weight_dict, predicted_performance, optimization_score
            )

            # 결과 객체 생성
            optimization_result = WeightOptimizationResult(
                optimized_weights=weight_dict,
                expected_performance=predicted_performance,
                optimization_score=optimization_score,
                convergence_iterations=convergence_iterations,
                stability_metrics=stability_metrics,
                recommendations=recommendations,
            )

            # 최적화 이력 저장
            self.optimization_history.append(
                {
                    "timestamp": datetime.now(),
                    "result": optimization_result,
                    "target_performance": target_performance,
                }
            )

            self.logger.info("메타 러닝 기반 가중치 최적화 완료")
            return optimization_result

        except Exception as e:
            self.logger.error(f"가중치 최적화 중 오류: {str(e)}")
            return WeightOptimizationResult({}, 0, 0, 0, {}, ["최적화 실패"])

    def _calculate_strategy_score(self, metrics: PerformanceMetrics) -> float:
        """전략 점수 계산"""
        try:
            # 다중 목표 점수
            roi_score = self._normalize_score(metrics.roi, -1.0, 1.0)
            win_rate_score = metrics.win_rate
            stability_score = (
                1.0 / (1.0 + metrics.volatility) if metrics.volatility > 0 else 1.0
            )
            risk_score = 1.0 - min(1.0, max(0.0, metrics.max_drawdown))

            # 가중 평균
            weighted_score = (
                self.opt_config.multi_objective_weights["roi"] * roi_score
                + self.opt_config.multi_objective_weights["win_rate"]
                * win_rate_score
                + self.opt_config.multi_objective_weights["stability"]
                * stability_score
                + self.opt_config.multi_objective_weights["risk"] * risk_score
            )

            return max(0.0, min(1.0, weighted_score))

        except Exception as e:
            return 0.0

    def _calculate_stability_metrics(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """안정성 메트릭 계산"""
        try:
            metrics = {}

            # 가중치 분산
            weight_values = list(weights.values())
            metrics["weight_variance"] = float(np.var(weight_values))

            # 가중치 엔트로피 (다양성)
            entropy = -np.sum([w * np.log(w + 1e-10) for w in weight_values])
            metrics["weight_entropy"] = float(entropy)

            # 최대 가중치 비율
            max_weight = max(weight_values)
            metrics["max_weight_ratio"] = float(max_weight)

            # 유효 전략 수
            effective_strategies = 1 / np.sum([w**2 for w in weight_values])
            metrics["effective_strategies"] = float(effective_strategies)

            return metrics

        except Exception as e:
            self.logger.error(f"안정성 메트릭 계산 중 오류: {str(e)}")
            return {}

    def _generate_optimization_recommendations(
        self,
        weights: Dict[str, float],
        predicted_performance: float,
        optimization_score: float,
    ) -> List[str]:
        """최적화 추천사항 생성"""
        recommendations = []

        try:
            # 성능 기반 추천
            if predicted_performance < 0:
                recommendations.append(
                    "예상 성능이 음수입니다. 전략 재검토가 필요합니다."
                )
            elif predicted_performance < 0.05:
                recommendations.append(
                    "예상 성능이 낮습니다. 보다 보수적인 접근을 권장합니다."
                )

            # 가중치 분포 기반 추천
            max_weight = max(weights.values())
            if max_weight > 0.5:
                recommendations.append(
                    "특정 전략에 과도하게 집중되어 있습니다. 다양성을 고려하세요."
                )

            min_weight = min(weights.values())
            if min_weight < 0.1:
                recommendations.append(
                    "일부 전략의 가중치가 너무 낮습니다. 포트폴리오 균형을 검토하세요."
                )

            # 최적화 점수 기반 추천
            if optimization_score < 0.1:
                recommendations.append(
                    "최적화 효과가 제한적입니다. 목표 성능을 재조정하거나 전략을 변경하세요."
                )

            # 긍정적 추천
            if predicted_performance > 0.1 and optimization_score > 0.2:
                recommendations.append(
                    "최적화 결과가 양호합니다. 현재 가중치를 적용해보세요."
                )

        except Exception as e:
            recommendations.append("추천사항 생성 중 오류가 발생했습니다.")

        return recommendations

    def get_current_weights(self) -> Dict[str, float]:
        """현재 가중치 반환"""
        try:
            return {
                strategy: weight_obj.current_weight
                for strategy, weight_obj in self.strategy_weights.items()
            }
        except Exception as e:
            self.logger.error(f"현재 가중치 반환 중 오류: {str(e)}")
            return {}

    def get_weight_summary(self) -> Dict[str, Any]:
        """가중치 요약 정보 반환"""
        try:
            summary = {
                "current_weights": self.get_current_weights(),
                "weight_objects": {},
                "stability_metrics": {},
                "optimization_history_count": len(self.optimization_history),
                "last_update": None,
            }

            # 가중치 객체 정보
            for strategy, weight_obj in self.strategy_weights.items():
                summary["weight_objects"][strategy] = {
                    "current_weight": weight_obj.current_weight,
                    "target_weight": weight_obj.target_weight,
                    "performance_score": weight_obj.performance_score,
                    "stability_score": weight_obj.stability_score,
                    "confidence": weight_obj.confidence,
                    "update_count": weight_obj.update_count,
                    "last_update": weight_obj.last_update.isoformat(),
                }

                if summary[
                    "last_update"
                ] is None or weight_obj.last_update > datetime.fromisoformat(
                    summary["last_update"]
                ):
                    summary["last_update"] = weight_obj.last_update.isoformat()

            # 안정성 메트릭
            current_weights = self.get_current_weights()
            if current_weights:
                summary["stability_metrics"] = self._calculate_stability_metrics(
                    current_weights
                )

            return summary

        except Exception as e:
            self.logger.error(f"가중치 요약 정보 반환 중 오류: {str(e)}")
            return {}

    def save_weights(self) -> None:
        """가중치 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 가중치 데이터 준비
            weight_data = {
                "timestamp": timestamp,
                "strategy_weights": {},
                "weight_history": {},
                "optimization_history": self.optimization_history,
                "config": asdict(self.opt_config),
            }

            # 전략 가중치
            for strategy, weight_obj in self.strategy_weights.items():
                weight_data["strategy_weights"][strategy] = {
                    "strategy": weight_obj.strategy,
                    "current_weight": weight_obj.current_weight,
                    "target_weight": weight_obj.target_weight,
                    "momentum": weight_obj.momentum,
                    "performance_score": weight_obj.performance_score,
                    "stability_score": weight_obj.stability_score,
                    "confidence": weight_obj.confidence,
                    "last_update": weight_obj.last_update.isoformat(),
                    "update_count": weight_obj.update_count,
                }

            # 가중치 이력
            for strategy, history in self.weight_history.items():
                weight_data["weight_history"][strategy] = list(history)

            # 파일 저장
            weights_file = self.cache_dir / f"adaptive_weights_{timestamp}.json"
            with open(weights_file, "w", encoding="utf-8") as f:
                json.dump(weight_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"가중치 저장 완료: {weights_file}")

        except Exception as e:
            self.logger.error(f"가중치 저장 중 오류: {str(e)}")

    def load_weights(self, date: Optional[str] = None) -> bool:
        """가중치 로드"""
        try:
            if date:
                pattern = f"adaptive_weights_{date}*.json"
            else:
                pattern = "adaptive_weights_*.json"

            weight_files = list(self.cache_dir.glob(pattern))
            if not weight_files:
                self.logger.warning(f"가중치 파일을 찾을 수 없습니다: {pattern}")
                return False

            # 최신 파일 선택
            latest_file = max(weight_files, key=lambda x: x.name)

            with open(latest_file, "r", encoding="utf-8") as f:
                weight_data = json.load(f)

            # 전략 가중치 복원
            for strategy, data in weight_data.get("strategy_weights", {}).items():
                self.strategy_weights[strategy] = StrategyWeight(
                    strategy=data["strategy"],
                    current_weight=data["current_weight"],
                    target_weight=data["target_weight"],
                    momentum=data["momentum"],
                    performance_score=data["performance_score"],
                    stability_score=data["stability_score"],
                    confidence=data["confidence"],
                    last_update=datetime.fromisoformat(data["last_update"]),
                    update_count=data["update_count"],
                )

            # 가중치 이력 복원
            for strategy, history in weight_data.get("weight_history", {}).items():
                self.weight_history[strategy] = deque(
                    history, maxlen=self.opt_config.performance_memory
                )

            # 최적화 이력 복원
            self.optimization_history = weight_data.get("optimization_history", [])

            self.logger.info(f"가중치 로드 완료: {latest_file}")
            return True

        except Exception as e:
            self.logger.error(f"가중치 로드 중 오류: {str(e)}")
            return False
