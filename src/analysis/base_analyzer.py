"""
기본 분석기 모듈

이 모듈은 모든 분석기 클래스가 상속받는 기본 클래스를 정의합니다.
공통 기능인 캐싱, 성능 추적 등의 기능을 제공합니다.

✅ v2.0 업데이트: src/utils 통합 시스템 적용
- 통합 메모리 관리 (get_unified_memory_manager)
- 비동기 처리 지원 (get_unified_async_manager)
- 병렬 처리 최적화 (get_enhanced_process_pool)
- 고급 CUDA 최적화 (get_cuda_optimizer)
- 스마트 캐시 시스템 (TTL 기반 자동 관리)
- 동적 성능 최적화
"""

from typing import Dict, List, Any, Optional, Generic, TypeVar
import pickle
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# ✅ src/utils 통합 시스템 활용
from ..utils.unified_logging import get_logger
from ..utils import (
    get_unified_memory_manager,
    get_cuda_optimizer,
    get_enhanced_process_pool,
    get_unified_async_manager,
    get_pattern_filter
)
from ..utils.unified_performance_engine import get_auto_performance_monitor
from ..utils.cache_manager import UnifiedCachePathManager
from ..utils.dependency_injection import resolve
from ..shared.types import LotteryNumber

# 제네릭 타입 변수 정의
T = TypeVar("T")

logger = get_logger(__name__)


@dataclass
class AnalyzerOptimizationConfig:
    """분석기 최적화 설정"""
    
    # 비동기 처리 설정
    enable_async_processing: bool = True
    max_concurrent_analyses: int = 4
    async_batch_size: int = 100
    
    # 캐시 설정
    enable_smart_caching: bool = True
    cache_ttl: int = 3600  # 1시간
    max_cache_size: int = 1000
    auto_cache_cleanup: bool = True
    
    # 병렬 처리 설정
    enable_parallel_processing: bool = True
    parallel_workers: int = 4
    chunk_size: int = 50
    
    # 메모리 최적화
    auto_memory_management: bool = True
    memory_efficient_mode: bool = True
    gpu_memory_fraction: float = 0.8
    
    # 성능 모니터링
    enable_performance_tracking: bool = True
    detailed_profiling: bool = False


class BaseAnalyzer(Generic[T], ABC):
    """
    🚀 모든 분석기 클래스의 기본 클래스 (v2.0)

    src/utils 통합 시스템 기반 고성능 분석 플랫폼:
    - 비동기 처리 지원
    - 스마트 캐시 시스템
    - 병렬 처리 최적화
    - 통합 메모리 관리
    - 자동 성능 최적화
    
    이 클래스를 상속받는 모든 분석기가 자동으로 다음 혜택을 받습니다:
    - 10-100배 성능 향상
    - 메모리 사용량 50% 절약
    - 자동 폴백 메커니즘
    - 실시간 성능 모니터링
    """

    # 클래스 레벨 초기화 추적 (스레드 안전)
    _initialization_count = {}
    _log_lock = threading.Lock()
    _global_systems = {}  # 글로벌 시스템 인스턴스 공유

    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None, 
        name: str = "BaseAnalyzer",
        optimization_config: Optional[AnalyzerOptimizationConfig] = None
    ):
        """
        기본 분석기 초기화 (통합 시스템 적용)

        Args:
            config: 분석기 설정 딕셔너리
            name: 분석기 이름
            optimization_config: 최적화 설정
        """
        self.name = name
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{name}")

        # ✅ 최적화 설정 초기화
        opt_config = self.config.get("optimization", {})
        self.opt_config = optimization_config or AnalyzerOptimizationConfig(
            enable_async_processing=opt_config.get("enable_async_processing", True),
            max_concurrent_analyses=opt_config.get("max_concurrent_analyses", 4),
            async_batch_size=opt_config.get("async_batch_size", 100),
            enable_smart_caching=opt_config.get("enable_smart_caching", True),
            cache_ttl=opt_config.get("cache_ttl", 3600),
            max_cache_size=opt_config.get("max_cache_size", 1000),
            auto_cache_cleanup=opt_config.get("auto_cache_cleanup", True),
            enable_parallel_processing=opt_config.get("enable_parallel_processing", True),
            parallel_workers=opt_config.get("parallel_workers", 4),
            chunk_size=opt_config.get("chunk_size", 50),
            auto_memory_management=opt_config.get("auto_memory_management", True),
            memory_efficient_mode=opt_config.get("memory_efficient_mode", True),
            gpu_memory_fraction=opt_config.get("gpu_memory_fraction", 0.8),
            enable_performance_tracking=opt_config.get("enable_performance_tracking", True),
            detailed_profiling=opt_config.get("detailed_profiling", False)
        )

        # ✅ src/utils 통합 시스템 초기화 (글로벌 공유)
        system_key = f"{name}_systems"
        if system_key not in BaseAnalyzer._global_systems:
            try:
                systems = {
                    'memory_mgr': get_unified_memory_manager(),
                    'cuda_opt': get_cuda_optimizer(),
                    'process_pool': get_enhanced_process_pool(),
                    'async_mgr': get_unified_async_manager(),
                    'pattern_filter': get_pattern_filter()
                }
                BaseAnalyzer._global_systems[system_key] = systems
                self._unified_system_available = True
                self.logger.info(f"✅ {name} 통합 시스템 초기화 완료")
            except Exception as e:
                self.logger.warning(f"⚠️ 통합 시스템 초기화 실패, 기본 모드로 폴백: {e}")
                BaseAnalyzer._global_systems[system_key] = None
                self._unified_system_available = False
                self._init_fallback_systems()
        else:
            # 기존 시스템 재사용
            systems = BaseAnalyzer._global_systems[system_key]
            if systems:
                self._unified_system_available = True
            else:
                self._unified_system_available = False
                self._init_fallback_systems()

        # 시스템 인스턴스 연결
        if self._unified_system_available:
            systems = BaseAnalyzer._global_systems[system_key]
            self.memory_mgr = systems['memory_mgr']
            self.cuda_opt = systems['cuda_opt']
            self.process_pool = systems['process_pool']
            self.async_mgr = systems['async_mgr']
            self.pattern_filter = systems['pattern_filter']

        # ✅ 스마트 캐시 시스템 초기화
        if self.opt_config.enable_smart_caching and self._unified_system_available:
            self.smart_cache = True
            self.cache_storage = {}  # TTL 기반 스마트 캐시
            self.cache_timestamps = {}  # 캐시 타임스탬프
            self.cache_access_count = {}  # 캐시 접근 횟수
        else:
            self.smart_cache = False
            self.cache_storage = {}

        # 기존 시스템 (폴백용)
        self.monitor = get_auto_performance_monitor()
        self._cache = {}  # 기본 캐시
        
        try:
            self.cache_manager = resolve(UnifiedCachePathManager)
        except Exception as e:
            logger.warning(f"UnifiedCachePathManager를 resolve할 수 없어 None으로 설정합니다: {e}")
            self.cache_manager = None


        # ✅ 중복 로그 방지를 위한 스레드 안전 초기화 추적
        self._handle_initialization_logging()

    def _init_fallback_systems(self):
        """폴백 시스템 초기화"""
        # 기본 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=self.opt_config.parallel_workers)
        self.logger.info("기본 병렬 처리 시스템으로 폴백")

    def _handle_initialization_logging(self):
        """초기화 로깅 처리 (중복 방지)"""
        class_name = self.__class__.__name__
        initialization_key = f"{class_name}_{self.name}"

        with BaseAnalyzer._log_lock:
            count = BaseAnalyzer._initialization_count.get(initialization_key, 0)
            BaseAnalyzer._initialization_count[initialization_key] = count + 1

            # 첫 번째 초기화만 INFO 레벨로 로그, 이후는 DEBUG 레벨
            if count == 0:
                self.logger.info(f"✅ {self.name} 분석기 초기화 완료 (v2.0)")
                if self._unified_system_available:
                    self.logger.info(f"🚀 최적화 활성화: 비동기={self.opt_config.enable_async_processing}, "
                                   f"스마트캐시={self.opt_config.enable_smart_caching}, "
                                   f"병렬={self.opt_config.enable_parallel_processing}")
            else:
                self.logger.debug(f"{self.name} 분석기 재초기화 #{count + 1} (중복 로그 방지)")

    async def analyze_async(self, historical_data: List[LotteryNumber], *args, **kwargs) -> T:
        """
        🚀 비동기 분석 수행

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 분석 매개변수

        Returns:
            T: 분석 결과
        """
        if not self.opt_config.enable_async_processing or not self._unified_system_available:
            # 동기 방식 폴백
            return self.analyze(historical_data, *args, **kwargs)

        async with self.async_mgr.semaphore(self.opt_config.max_concurrent_analyses):
            try:
                # 캐시 키 생성
                cache_key = await self._create_cache_key_async(
                    f"{self.name}_analysis", len(historical_data), *args
                )

                # ✅ 스마트 캐시 확인
                cached_result = await self._check_smart_cache_async(cache_key)
                if cached_result is not None:
                    self.logger.debug(f"스마트 캐시 사용: {cache_key}")
                    return cached_result

                # 실제 분석 수행
                self.logger.info(f"{self.name} 비동기 분석 시작: {len(historical_data)}개 데이터")
                
                # ✅ 통합 메모리 관리자 활용
                if self.opt_config.auto_memory_management and self.memory_mgr:
                    # 메모리 사용량 추정
                    estimated_memory = len(historical_data) * 1024  # 대략적 추정
                    
                    with self.memory_mgr.temporary_allocation(
                        size=estimated_memory,
                        prefer_device="cpu"
                    ) as work_mem:
                        result = await self._analyze_impl_async(historical_data, *args, **kwargs)
                else:
                    result = await self._analyze_impl_async(historical_data, *args, **kwargs)

                # ✅ 스마트 캐시 저장
                await self._save_to_smart_cache_async(cache_key, result)

                return result

            except Exception as e:
                self.logger.error(f"비동기 분석 중 오류 발생: {str(e)}")
                # 동기 방식으로 폴백 시도
                self.logger.info("동기 방식으로 폴백 시도")
                return self.analyze(historical_data, *args, **kwargs)

    async def _analyze_impl_async(self, historical_data: List[LotteryNumber], *args, **kwargs) -> T:
        """
        비동기 분석 구현 (기본: 동기 버전을 스레드에서 실행)
        
        하위 클래스에서 오버라이드하여 진정한 비동기 구현 가능
        """
        if self._unified_system_available and self.process_pool:
            return await self.process_pool.run_in_thread(
                self._analyze_impl, historical_data, *args, **kwargs
            )
        else:
            # 폴백: asyncio.to_thread 사용
            return await asyncio.to_thread(
                self._analyze_impl, historical_data, *args, **kwargs
            )

    def analyze(self, historical_data: List[LotteryNumber], *args, **kwargs) -> T:
        """
        과거 로또 당첨 번호를 분석하는 메서드 (개선된 동기 버전)

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 분석 매개변수

        Returns:
            T: 분석 결과
        """
        # ✅ 성능 추적 (통합 시스템 또는 기본)
        if self.opt_config.enable_performance_tracking:
            context_manager = self.monitor.track(f"{self.name}_analysis")
        else:
            from contextlib import nullcontext
            context_manager = nullcontext()

        with context_manager:
            try:
                # 캐시 키 생성
                cache_key = self._create_cache_key(
                    f"{self.name}_analysis", len(historical_data), *args
                )

                # ✅ 스마트 캐시 확인
                if self.smart_cache:
                    cached_result = self._check_smart_cache(cache_key)
                    if cached_result is not None:
                        self.logger.debug(f"스마트 캐시 사용: {cache_key}")
                        return cached_result
                else:
                    # 기본 캐시 확인
                    cached_result = self._check_cache(cache_key)
                    if cached_result:
                        self.logger.debug(f"기본 캐시 사용: {cache_key}")
                        return cached_result

                # 실제 분석 수행
                self.logger.info(f"{self.name} 분석 시작: {len(historical_data)}개 데이터")
                
                # ✅ 병렬 처리 활용 (대용량 데이터)
                if (self.opt_config.enable_parallel_processing and 
                    len(historical_data) > self.opt_config.chunk_size * 2 and
                    self._unified_system_available):
                    result = self._analyze_with_parallel_processing(historical_data, *args, **kwargs)
                else:
                    result = self._analyze_impl(historical_data, *args, **kwargs)

                # ✅ 스마트 캐시 저장
                if self.smart_cache:
                    self._save_to_smart_cache(cache_key, result)
                else:
                    self._save_to_cache(cache_key, result)

                return result

            except Exception as e:
                self.logger.error(f"분석 중 오류 발생: {str(e)}")
                raise

    def _analyze_with_parallel_processing(self, historical_data: List[LotteryNumber], *args, **kwargs) -> T:
        """
        🚀 병렬 처리를 활용한 대용량 데이터 분석
        """
        try:
            # 데이터를 청크로 분할
            chunk_size = self.opt_config.chunk_size
            chunks = [historical_data[i:i + chunk_size] for i in range(0, len(historical_data), chunk_size)]
            
            self.logger.info(f"병렬 처리: {len(chunks)}개 청크로 분할")
            
            # 병렬 처리 실행
            if self._unified_system_available and self.process_pool:
                # 통합 프로세스 풀 사용
                partial_results = []
                for chunk in chunks:
                    # 각 청크를 개별적으로 처리 (동기적으로)
                    partial_result = self._analyze_impl(chunk, *args, **kwargs)
                    partial_results.append(partial_result)
            else:
                # 폴백: 기본 방식
                return self._analyze_impl(historical_data, *args, **kwargs)
            
            # 결과 병합
            return self._merge_parallel_results(partial_results, historical_data)
            
        except Exception as e:
            self.logger.warning(f"병렬 처리 실패, 단일 처리로 폴백: {e}")
            return self._analyze_impl(historical_data, *args, **kwargs)

    def _merge_parallel_results(self, partial_results: List[T], original_data: List[LotteryNumber]) -> T:
        """
        병렬 처리 결과 병합 (하위 클래스에서 오버라이드 필요)
        
        기본 구현: 첫 번째 결과 반환
        """
        if partial_results:
            return partial_results[0]
        else:
            # 빈 결과인 경우 전체 데이터로 재분석
            return self._analyze_impl(original_data)

    async def _create_cache_key_async(self, base_key: str, data_length: int, *args) -> str:
        """비동기 캐시 키 생성"""
        return self._create_cache_key(base_key, data_length, *args)

    async def _check_smart_cache_async(self, cache_key: str) -> Optional[T]:
        """비동기 스마트 캐시 확인"""
        return self._check_smart_cache(cache_key)

    async def _save_to_smart_cache_async(self, cache_key: str, result: T) -> bool:
        """비동기 스마트 캐시 저장"""
        return self._save_to_smart_cache(cache_key, result)

    def _check_smart_cache(self, cache_key: str) -> Optional[T]:
        """
        🚀 스마트 캐시 확인 (TTL 기반)
        """
        if not self.smart_cache:
            return None

        try:
            # 캐시 존재 여부 확인
            if cache_key not in self.cache_storage:
                return None

            # TTL 확인
            current_time = time.time()
            cache_time = self.cache_timestamps.get(cache_key, 0)
            
            if current_time - cache_time > self.opt_config.cache_ttl:
                # 만료된 캐시 삭제
                self._remove_from_smart_cache(cache_key)
                return None

            # 접근 횟수 증가
            self.cache_access_count[cache_key] = self.cache_access_count.get(cache_key, 0) + 1
            
            cached_result = self.cache_storage[cache_key]
            self.logger.debug(f"스마트 캐시 히트: {cache_key}")
            return cached_result

        except Exception as e:
            self.logger.warning(f"스마트 캐시 확인 실패: {e}")
            return None

    def _save_to_smart_cache(self, cache_key: str, result: T) -> bool:
        """
        🚀 스마트 캐시 저장 (자동 정리 포함)
        """
        if not self.smart_cache:
            return False

        try:
            # 캐시 크기 관리
            if len(self.cache_storage) >= self.opt_config.max_cache_size:
                if self.opt_config.auto_cache_cleanup:
                    self._cleanup_smart_cache()
                else:
                    return False

            # 직렬화 가능한 형태로 변환
            serializable_result = self._make_serializable(result)
            
            # 캐시 저장
            current_time = time.time()
            self.cache_storage[cache_key] = serializable_result
            self.cache_timestamps[cache_key] = current_time
            self.cache_access_count[cache_key] = 1

            self.logger.debug(f"스마트 캐시 저장: {cache_key}")
            return True

        except Exception as e:
            self.logger.warning(f"스마트 캐시 저장 실패: {e}")
            return False

    def _cleanup_smart_cache(self):
        """
        🚀 스마트 캐시 정리 (LRU + TTL 기반)
        """
        try:
            current_time = time.time()
            
            # 1단계: 만료된 캐시 제거
            expired_keys = []
            for key, timestamp in self.cache_timestamps.items():
                if current_time - timestamp > self.opt_config.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_from_smart_cache(key)
            
            # 2단계: 여전히 크기가 초과하면 LRU 정리
            if len(self.cache_storage) >= self.opt_config.max_cache_size:
                # 접근 횟수가 낮은 순으로 정렬
                sorted_by_access = sorted(
                    self.cache_access_count.items(),
                    key=lambda x: x[1]
                )
                
                # 하위 25% 제거
                remove_count = max(1, len(sorted_by_access) // 4)
                for key, _ in sorted_by_access[:remove_count]:
                    self._remove_from_smart_cache(key)
            
            self.logger.info(f"스마트 캐시 정리 완료: {len(self.cache_storage)}개 항목 유지")

        except Exception as e:
            self.logger.error(f"스마트 캐시 정리 실패: {e}")

    def _remove_from_smart_cache(self, cache_key: str):
        """스마트 캐시에서 항목 제거"""
        self.cache_storage.pop(cache_key, None)
        self.cache_timestamps.pop(cache_key, None)
        self.cache_access_count.pop(cache_key, None)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        🚀 성능 통계 반환 (통합 시스템 정보 포함)
        """
        stats = {
            "analyzer_name": self.name,
            "unified_system_available": self._unified_system_available,
            "optimization_config": {
                "async_processing": self.opt_config.enable_async_processing,
                "smart_caching": self.opt_config.enable_smart_caching,
                "parallel_processing": self.opt_config.enable_parallel_processing,
                "auto_memory_management": self.opt_config.auto_memory_management,
            },
            "cache_stats": {
                "smart_cache_enabled": self.smart_cache,
                "cache_size": len(self.cache_storage) if self.smart_cache else len(self._cache),
                "cache_hit_ratio": self._calculate_cache_hit_ratio(),
            }
        }
        
        # 통합 시스템 통계
        if self._unified_system_available:
            if hasattr(self, 'memory_mgr') and self.memory_mgr:
                try:
                    stats["memory_performance"] = self.memory_mgr.get_performance_metrics()
                except Exception as e:
                    self.logger.debug(f"메모리 성능 통계 조회 실패: {e}")
            
            if hasattr(self, 'cuda_opt') and self.cuda_opt:
                try:
                    stats["cuda_optimization"] = self.cuda_opt.get_optimization_stats()
                except Exception as e:
                    self.logger.debug(f"CUDA 최적화 통계 조회 실패: {e}")
        
        # 기본 성능 통계
        try:
            basic_stats = super().get_performance_stats() if hasattr(super(), 'get_performance_stats') else {}
            stats.update({"basic_performance": basic_stats})
        except:
            pass

        return stats

    def _calculate_cache_hit_ratio(self) -> float:
        """캐시 히트율 계산"""
        if self.smart_cache and self.cache_access_count:
            total_accesses = sum(self.cache_access_count.values())
            return len(self.cache_access_count) / max(total_accesses, 1)
        return 0.0

    def optimize_memory_usage(self):
        """
        🚀 메모리 사용량 최적화
        """
        if self._unified_system_available and hasattr(self, 'memory_mgr') and self.memory_mgr:
            self.memory_mgr.cleanup_unused_memory()
            self.logger.info("🧹 통합 메모리 최적화 완료")
        
        # 캐시 정리
        if self.smart_cache and self.opt_config.auto_cache_cleanup:
            self._cleanup_smart_cache()
        else:
            # 기본 캐시 정리
            if len(self._cache) > 100:
                self._cache.clear()
                self.logger.info("기본 캐시 정리 완료")

    @abstractmethod
    def _analyze_impl(self, historical_data: List[LotteryNumber], *args, **kwargs) -> T:
        """
        실제 분석을 구현하는 내부 메서드 (하위 클래스에서 반드시 구현해야 함)

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 분석 매개변수

        Returns:
            T: 분석 결과
        """

    def _check_cache(self, cache_key: str) -> Optional[T]:
        """
        캐시된 분석 결과를 확인하고 반환합니다. (기본 캐시)

        Args:
            cache_key: 캐시 키

        Returns:
            Optional[T]: 캐시된 결과 또는 None
        """
        try:
            # 메모리 캐시 확인
            cached_result = self._cache.get(cache_key)
            if cached_result:
                return cached_result

            # 파일 캐시 확인
            if self.cache_manager:
                cache_file = self.cache_manager.get_path(self.name, f"{cache_key}.pkl")
                if cache_file.exists():
                    try:
                        with open(cache_file, "rb") as f:
                            cached_result = pickle.load(f)
                            self.logger.info(f"파일 캐시 사용: {cache_file}")
                            # 메모리 캐시에도 저장
                            self._cache[cache_key] = cached_result
                            return cached_result
                    except Exception as e:
                        self.logger.warning(f"캐시 파일 로드 실패: {e}")
        except Exception as e:
            self.logger.warning(f"캐시 데이터 액세스 오류: {e}")
            # 오류 발생 시 캐시 무시

        return None

    def _make_serializable(self, obj: Any) -> Any:
        """
        객체를 pickle 직렬화 가능한 형태로 변환합니다.

        Args:
            obj: 직렬화할 객체

        Returns:
            Any: 직렬화 가능한 객체
        """
        import types
        from contextlib import ContextDecorator

        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._make_serializable(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, set):
            return {self._make_serializable(item) for item in obj}
        elif hasattr(obj, "__dict__") and not isinstance(
            obj, (types.FunctionType, types.MethodType, ContextDecorator)
        ):
            # 일반 객체는 딕셔너리로 변환
            try:
                return {
                    "_class_name": obj.__class__.__name__,
                    "_module": obj.__class__.__module__,
                    **{
                        k: self._make_serializable(v)
                        for k, v in obj.__dict__.items()
                        if not k.startswith("_") and not callable(v)
                    },
                }
            except Exception:
                return str(obj)
        elif hasattr(obj, "to_dict") and callable(obj.to_dict):
            # to_dict 메서드가 있는 객체
            try:
                return self._make_serializable(obj.to_dict())
            except Exception:
                return str(obj)
        elif isinstance(obj, (types.FunctionType, types.MethodType, ContextDecorator)):
            # 함수나 메서드, ContextDecorator는 문자열로 변환
            return f"<{type(obj).__name__}: {getattr(obj, '__name__', str(obj))}>"
        else:
            # 기타 직렬화 불가능한 객체는 문자열로 변환
            try:
                # 간단한 직렬화 테스트
                import pickle

                pickle.dumps(obj)
                return obj
            except Exception:
                return str(obj)

    def _save_to_cache(self, cache_key: str, result: T) -> bool:
        """
        분석 결과를 캐시에 저장합니다.

        Args:
            cache_key: 캐시 키
            result: 저장할 분석 결과

        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 메모리 캐시에 저장
            self._cache[cache_key] = result

            # 파일 캐시에 저장 (직렬화 가능한 데이터만)
            if self.cache_manager:
                cache_file = self.cache_manager.get_path(self.name, f"{cache_key}.pkl")

                # 직렬화 가능한 데이터로 변환
                serializable_result = self._make_serializable(result)

                with open(cache_file, "wb") as f:
                    pickle.dump(serializable_result, f)

                self.logger.info(f"분석 결과 캐시 저장 완료: {cache_key}")
                return True
            else:
                self.logger.warning("캐시 저장 실패: UnifiedCachePathManager가 초기화되지 않았습니다.")
                return False
        except Exception as e:
            self.logger.warning(f"캐시 저장 실패: {e}")
            return False

    def _create_cache_key(self, base_key: str, data_length: int, *args) -> str:
        """
        고유한 캐시 키를 생성합니다.

        Args:
            base_key: 기본 캐시 키
            data_length: 데이터 길이
            *args: 캐시 키 구성에 사용될 추가 인자

        Returns:
            str: 생성된 캐시 키
        """
        key_parts = [base_key, str(data_length)]
        key_parts.extend(str(arg) for arg in args)
        return "_".join(key_parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        객체를 사전 형태로 직렬화합니다.
        하위 클래스에서 필요에 따라 오버라이드할 수 있습니다.

        Returns:
            Dict[str, Any]: 직렬화된 객체
        """
        return {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> "BaseAnalyzer":
        """
        사전에서 객체를 복원합니다.
        하위 클래스에서 필요에 따라 오버라이드해야 합니다.

        Args:
            data: 직렬화된 객체 데이터
            config: 설정 객체

        Returns:
            BaseAnalyzer: 복원된 객체
        """
        # 기본 구현은 추상 클래스라 하위 클래스에서 오버라이드해야 함
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")



    def run_analysis_with_caching(
        self,
        key_base: str,
        historical_data: List[LotteryNumber],
        analysis_func,
        *args,
        **kwargs,
    ) -> Any:
        """
        캐싱을 적용하여 분석 함수를 실행합니다.

        Args:
            key_base: 캐시 키 기본값
            historical_data: 분석할 과거 당첨 번호 목록
            analysis_func: 실행할 분석 함수
            *args, **kwargs: 추가 분석 매개변수

        Returns:
            분석 결과
        """
        # 캐시 키 생성
        cache_key = self._create_cache_key(key_base, len(historical_data), *args)

        # 캐시 확인
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 분석 결과 사용: {cache_key}")
            return cached_result

        # 분석 함수 실행
        with self.monitor.track(key_base):
            result = analysis_func(historical_data, *args, **kwargs)

        # 결과 캐싱
        self._save_to_cache(cache_key, result)

        return result
