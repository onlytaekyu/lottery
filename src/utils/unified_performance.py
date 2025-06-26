"""
통합 성능 측정 시스템

시간 측정, 메모리 추적, 프로파일링 기능을 통합하여 중복 코드를 제거합니다.
"""

import time
import threading
import psutil
import torch
import gc
import cProfile
import pstats
import io
import functools
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Union, Generator, ContextManager
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .unified_logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceConfig:
    """성능 측정 설정"""

    enable_profiling: bool = True
    enable_memory_tracking: bool = True
    enable_cpu_profiling: bool = True
    enable_gpu_profiling: bool = True
    time_threshold: float = 0.1  # 시간 임계값 (초)
    memory_threshold: int = 100 * 1024 * 1024  # 메모리 임계값 (100MB)
    sampling_interval: float = 0.1  # 샘플링 간격
    max_reports: int = 100
    detailed_logging: bool = False
    save_profiles: bool = False
    profile_output_dir: str = "logs/profiles"


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    start_memory: int = 0
    end_memory: int = 0
    memory_used: int = 0
    peak_memory: int = 0
    cpu_percent: float = 0.0
    gpu_memory_allocated: int = 0
    gpu_memory_reserved: int = 0
    thread_count: int = 0
    additional_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def memory_used_mb(self) -> float:
        """메모리 사용량 (MB)"""
        return self.memory_used / (1024 * 1024)

    @property
    def gpu_memory_allocated_mb(self) -> float:
        """GPU 메모리 할당량 (MB)"""
        return self.gpu_memory_allocated / (1024 * 1024)


class UnifiedPerformanceTracker:
    """통합 성능 추적기"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, config: Optional[PerformanceConfig] = None):
        """싱글톤 패턴 구현"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Optional[PerformanceConfig] = None):
        if self._initialized:
            return

        self.config = config or PerformanceConfig()
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.active_sessions: Dict[str, PerformanceMetrics] = {}
        self.profilers: Dict[str, cProfile.Profile] = {}
        self.logger = get_logger("performance")
        self._initialized = True

        # 프로파일 저장 디렉토리 생성
        if self.config.save_profiles:
            Path(self.config.profile_output_dir).mkdir(parents=True, exist_ok=True)

    def start_tracking(self, name: str, **kwargs) -> PerformanceMetrics:
        """
        성능 추적 시작

        Args:
            name: 추적 세션 이름
            **kwargs: 추가 메타데이터

        Returns:
            성능 메트릭 객체
        """
        with self._lock:
            if name in self.active_sessions:
                self.logger.warning(f"세션 '{name}'이 이미 활성화되어 있습니다.")
                return self.active_sessions[name]

            # 성능 메트릭 초기화
            metrics = PerformanceMetrics(name=name)
            metrics.start_time = time.time()
            metrics.additional_data.update(kwargs)

            # 시스템 상태 측정
            if self.config.enable_memory_tracking:
                process = psutil.Process()
                metrics.start_memory = process.memory_info().rss
                metrics.cpu_percent = process.cpu_percent()
                metrics.thread_count = threading.active_count()

            # GPU 메모리 측정
            if self.config.enable_gpu_profiling and torch.cuda.is_available():
                try:
                    metrics.gpu_memory_allocated = torch.cuda.memory_allocated()
                    metrics.gpu_memory_reserved = torch.cuda.memory_reserved()
                except Exception as e:
                    self.logger.debug(f"GPU 메모리 측정 실패: {e}")

            # CPU 프로파일링 시작
            if self.config.enable_cpu_profiling:
                profiler = cProfile.Profile()
                profiler.enable()
                self.profilers[name] = profiler

            self.active_sessions[name] = metrics

            if self.config.detailed_logging:
                self.logger.debug(f"성능 추적 시작: {name}")

            return metrics

    def stop_tracking(self, name: str) -> PerformanceMetrics:
        """
        성능 추적 중지

        Args:
            name: 추적 세션 이름

        Returns:
            완료된 성능 메트릭
        """
        with self._lock:
            if name not in self.active_sessions:
                self.logger.warning(f"활성 세션 '{name}'을 찾을 수 없습니다.")
                return PerformanceMetrics(name=name)

            metrics = self.active_sessions[name]

            # 종료 시간 기록
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time

            # 시스템 상태 측정
            if self.config.enable_memory_tracking:
                process = psutil.Process()
                metrics.end_memory = process.memory_info().rss
                metrics.memory_used = metrics.end_memory - metrics.start_memory
                metrics.peak_memory = max(metrics.start_memory, metrics.end_memory)

            # GPU 메모리 측정
            if self.config.enable_gpu_profiling and torch.cuda.is_available():
                try:
                    gpu_allocated = torch.cuda.memory_allocated()
                    gpu_reserved = torch.cuda.memory_reserved()
                    metrics.gpu_memory_allocated = (
                        gpu_allocated - metrics.gpu_memory_allocated
                    )
                    metrics.gpu_memory_reserved = (
                        gpu_reserved - metrics.gpu_memory_reserved
                    )
                except Exception as e:
                    self.logger.debug(f"GPU 메모리 측정 실패: {e}")

            # CPU 프로파일링 중지
            if name in self.profilers:
                profiler = self.profilers[name]
                profiler.disable()

                if self.config.save_profiles:
                    self._save_profile(name, profiler)

                del self.profilers[name]

            # 활성 세션에서 제거하고 완료된 메트릭에 저장
            del self.active_sessions[name]
            self.metrics[name] = metrics

            # 임계값 확인 및 로깅
            self._check_thresholds(metrics)

            if self.config.detailed_logging:
                self.logger.debug(
                    f"성능 추적 완료: {name} "
                    f"(시간: {metrics.duration:.2f}초, "
                    f"메모리: {metrics.memory_used_mb:.1f}MB)"
                )

            return metrics

    def _check_thresholds(self, metrics: PerformanceMetrics):
        """임계값 확인 및 경고"""
        if metrics.duration > self.config.time_threshold:
            self.logger.info(
                f"실행 시간 임계값 초과: {metrics.name} "
                f"({metrics.duration:.2f}초 > {self.config.time_threshold}초)"
            )

        if metrics.memory_used > self.config.memory_threshold:
            self.logger.info(
                f"메모리 사용량 임계값 초과: {metrics.name} "
                f"({metrics.memory_used_mb:.1f}MB > {self.config.memory_threshold / (1024*1024):.1f}MB)"
            )

    def _save_profile(self, name: str, profiler: cProfile.Profile):
        """프로파일 결과 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_file = (
                Path(self.config.profile_output_dir) / f"{name}_{timestamp}.prof"
            )

            # 프로파일 통계 저장
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(50)  # 상위 50개 함수

            with open(profile_file.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(s.getvalue())

            self.logger.debug(f"프로파일 저장 완료: {profile_file}")

        except Exception as e:
            self.logger.error(f"프로파일 저장 실패: {e}")

    @contextmanager
    def track(self, name: str, **kwargs) -> Generator[PerformanceMetrics, None, None]:
        """
        컨텍스트 매니저로 성능 추적

        Args:
            name: 추적 세션 이름
            **kwargs: 추가 메타데이터

        Yields:
            성능 메트릭 객체
        """
        metrics = self.start_tracking(name, **kwargs)
        try:
            yield metrics
        finally:
            self.stop_tracking(name)

    def get_metrics(
        self, name: Optional[str] = None
    ) -> Union[PerformanceMetrics, Dict[str, PerformanceMetrics]]:
        """
        성능 메트릭 조회

        Args:
            name: 특정 세션 이름 (None이면 전체 반환)

        Returns:
            성능 메트릭 또는 전체 메트릭 딕셔너리
        """
        if name is None:
            return self.metrics.copy()
        return self.metrics.get(name, PerformanceMetrics(name=name))

    def get_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        if not self.metrics:
            return {"total_sessions": 0}

        durations = [m.duration for m in self.metrics.values()]
        memory_usage = [m.memory_used_mb for m in self.metrics.values()]

        return {
            "total_sessions": len(self.metrics),
            "total_time": sum(durations),
            "avg_time": np.mean(durations),
            "max_time": max(durations),
            "total_memory_mb": sum(memory_usage),
            "avg_memory_mb": np.mean(memory_usage),
            "max_memory_mb": max(memory_usage),
            "slowest_session": max(self.metrics.items(), key=lambda x: x[1].duration)[
                0
            ],
            "memory_intensive_session": max(
                self.metrics.items(), key=lambda x: x[1].memory_used
            )[0],
        }

    def reset(self, name: Optional[str] = None):
        """메트릭 초기화"""
        with self._lock:
            if name is None:
                self.metrics.clear()
                self.active_sessions.clear()
                self.profilers.clear()
                self.logger.debug("모든 성능 메트릭을 초기화했습니다.")
            else:
                self.metrics.pop(name, None)
                self.active_sessions.pop(name, None)
                self.profilers.pop(name, None)
                self.logger.debug(f"'{name}' 세션 메트릭을 초기화했습니다.")


# 데코레이터 함수들
def performance_monitor(name: Optional[str] = None, **kwargs):
    """
    함수 성능 모니터링 데코레이터

    Args:
        name: 추적 세션 이름 (기본값: 함수명)
        **kwargs: 추가 메타데이터
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            tracker = UnifiedPerformanceTracker()
            session_name = name or func.__name__

            with tracker.track(session_name, **kwargs):
                return func(*args, **func_kwargs)

        return wrapper

    return decorator


def memory_monitor(threshold_mb: float = 100.0):
    """
    메모리 사용량 모니터링 데코레이터

    Args:
        threshold_mb: 메모리 임계값 (MB)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = UnifiedPerformanceTracker()

            with tracker.track(func.__name__, memory_threshold_mb=threshold_mb):
                result = func(*args, **kwargs)

                # 메트릭 확인
                metrics = tracker.get_metrics(func.__name__)
                if metrics.memory_used_mb > threshold_mb:
                    logger.warning(
                        f"함수 {func.__name__} 메모리 사용량 초과: "
                        f"{metrics.memory_used_mb:.1f}MB > {threshold_mb}MB"
                    )

                return result

        return wrapper

    return decorator


# 편의 함수들
def get_performance_tracker(
    config: Optional[PerformanceConfig] = None,
) -> UnifiedPerformanceTracker:
    """통합 성능 추적기 반환"""
    return UnifiedPerformanceTracker(config)


@contextmanager
def track_performance(name: str, **kwargs) -> Generator[PerformanceMetrics, None, None]:
    """성능 추적 컨텍스트 매니저 (편의 함수)"""
    tracker = UnifiedPerformanceTracker()
    with tracker.track(name, **kwargs) as metrics:
        yield metrics


def measure_time(func: Callable) -> Callable:
    """시간 측정 데코레이터 (간단 버전)"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        logger.debug(f"함수 {func.__name__} 실행 시간: {duration:.2f}초")
        return result

    return wrapper


def get_system_info() -> Dict[str, Any]:
    """시스템 정보 반환"""
    try:
        process = psutil.Process()

        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": process.memory_info().rss / (1024**2),
            "process_cpu_percent": process.cpu_percent(),
            "thread_count": threading.active_count(),
        }

        # GPU 정보 추가
        if torch.cuda.is_available():
            info.update(
                {
                    "gpu_available": True,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated()
                    / (1024**2),
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                    "gpu_memory_total_mb": torch.cuda.get_device_properties(
                        0
                    ).total_memory
                    / (1024**2),
                }
            )
        else:
            info["gpu_available"] = False

        return info

    except Exception as e:
        logger.error(f"시스템 정보 수집 실패: {e}")
        return {"error": str(e)}
