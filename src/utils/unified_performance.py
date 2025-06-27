"""
통합 성능 관리 시스템

이 모듈은 프로파일링, 메모리 추적, 성능 보고서 작성을 모두 통합 관리합니다.
기존 profiler.py, performance_utils.py, performance_tracker.py의 모든 기능을 포함합니다.
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
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from .unified_logging import get_logger

logger = get_logger(__name__)

@dataclass
class PerformanceConfig:
    """통합 성능 설정"""

    enable_profiling: bool = True
    enable_memory_tracking: bool = True
    enable_cpu_monitoring: bool = True
    enable_gpu_monitoring: bool = True
    time_threshold: float = 0.1
    memory_threshold: int = 1_000_000
    report_threshold: float = 0.1
    save_reports: bool = True
    report_dir: str = "logs/performance"
    detailed_memory: bool = False
    sampling_interval: float = 0.1
    cpu_threshold: float = 80.0
    thread_threshold: int = 100

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""

    duration: float = 0.0
    memory_used: int = 0
    cpu_usage: float = 0.0
    gpu_memory: int = 0
    timestamp: float = field(default_factory=time.time)

class UnifiedPerformanceTracker:
    """통합 성능 추적기 - 모든 성능 관련 기능을 통합"""

    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.profiles: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
        self.memory_stats: Dict[str, Any] = {}
        self.reports: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # 기존 PerformanceTracker 호환성
        self.metrics = {}
        self.batch_metrics = {}
        self.memory_usage = {}

        # 기존 Profiler 호환성
        self._is_profiling = False
        self._memory_stats = []
        self._thread_stats = []

    @contextmanager
    def track(self, name: str):
        """통합 성능 추적 - 모든 메트릭을 한번에 수집"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_threads = threading.active_count()

        # GPU 메모리 (가능한 경우)
        gpu_memory_start = 0
        if torch.cuda.is_available():
            gpu_memory_start = torch.cuda.memory_allocated()

        # 프로파일링 시작
        profiler = None
        if self.config.enable_profiling:
            profiler = cProfile.Profile()
            profiler.enable()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            end_threads = threading.active_count()

            duration = end_time - start_time
            memory_used = end_memory - start_memory
            threads_created = end_threads - start_threads

            # GPU 메모리 (가능한 경우)
            gpu_memory_used = 0
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() - gpu_memory_start

            # 프로파일링 종료
            profile_stats = None
            if profiler:
                profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                ps.print_stats(20)
                profile_stats = s.getvalue()

            # 결과 저장
            with self._lock:
                self._save_metrics(
                    name,
                    {
                        "duration": duration,
                        "memory_used": memory_used,
                        "threads_created": threads_created,
                        "gpu_memory_used": gpu_memory_used,
                        "profile_stats": profile_stats,
                        "timestamp": time.time(),
                    },
                )

    def _save_metrics(self, name: str, metrics: Dict[str, Any]):
        """메트릭 저장"""
        if name not in self.sessions:
            self.sessions[name] = {
                "durations": [],
                "memory_usage": [],
                "gpu_memory": [],
                "thread_counts": [],
                "profile_data": [],
            }

        session = self.sessions[name]
        session["durations"].append(metrics["duration"])
        session["memory_usage"].append(metrics["memory_used"])
        session["gpu_memory"].append(metrics["gpu_memory_used"])
        session["thread_counts"].append(metrics["threads_created"])

        if metrics["profile_stats"]:
            session["profile_data"].append(metrics["profile_stats"])

        # 임계값 체크 및 로깅
        if metrics["duration"] > self.config.time_threshold:
            logger.warning(f"성능 임계값 초과: {name} - {metrics['duration']:.2f}초")

        if metrics["memory_used"] > self.config.memory_threshold:
            logger.warning(
                f"메모리 임계값 초과: {name} - {metrics['memory_used']/1024/1024:.2f}MB"
            )

    # 기존 Profiler 호환 메서드들
    @contextmanager
    def profile(self, name: str):
        """프로파일링 컨텍스트 (기존 호환성)"""
        with self.track(name):
            yield

    def start(self, name: str) -> None:
        """프로파일링 시작 (기존 호환성)"""
        if name not in self.profiles:
            self.profiles[name] = []
        self.start_times[name] = time.time()
        self._is_profiling = True

    def stop(self, name: str) -> None:
        """프로파일링 중지 (기존 호환성)"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.profiles[name].append(duration)
            del self.start_times[name]
        self._is_profiling = False

    def get_profile_stats(self) -> Dict[str, Dict[str, float]]:
        """프로파일링 통계 반환 (기존 호환성)"""
        stats = {}
        for name, durations in self.profiles.items():
            if durations:
                stats[name] = {
                    "mean": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "total": sum(durations),
                    "count": len(durations),
                }
        return stats

    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 사용량 통계 반환 (기존 호환성)"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        return {
            "allocated": torch.cuda.memory_allocated(),
            "cached": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_cached": torch.cuda.max_memory_reserved(),
        }

    def clear(self) -> None:
        """프로파일링 데이터 초기화 (기존 호환성)"""
        self.profiles.clear()
        self.start_times.clear()
        self.sessions.clear()
        self.metrics.clear()
        self.batch_metrics.clear()
        self.memory_usage.clear()

    # 기존 PerformanceTracker 호환 메서드들
    def start_tracking(self, name: str):
        """추적 시작 (기존 호환성)"""
        self.start(name)

    def stop_tracking(self, name: str):
        """추적 중지 (기존 호환성)"""
        self.stop(name)

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """메트릭 업데이트 (기존 호환성)"""
        self.metrics.update(metrics)

    def record_batch(self, name: str, batch_size: int, batch_time: float):
        """배치 기록 (기존 호환성)"""
        if name not in self.batch_metrics:
            self.batch_metrics[name] = []

        self.batch_metrics[name].append(
            {
                "batch_size": batch_size,
                "batch_time": batch_time,
                "throughput": batch_size / batch_time if batch_time > 0 else 0,
            }
        )

    def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """메트릭 반환 (기존 호환성)"""
        if name is None:
            return self.metrics
        return self.metrics.get(name, {})

    def get_summary(self, name: str) -> Dict[str, float]:
        """요약 통계 반환 (기존 호환성)"""
        if name not in self.sessions:
            return {}

        session = self.sessions[name]
        durations = session.get("durations", [])

        if not durations:
            return {}

        return {
            "mean_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations),
            "call_count": len(durations),
        }

    def generate_report(self) -> str:
        """통합 보고서 생성"""
        report = ["=== 통합 성능 보고서 ==="]

        # 세션별 통계
        report.append("\n[세션별 성능 통계]")
        for name, session in self.sessions.items():
            durations = session.get("durations", [])
            if durations:
                report.append(f"\n{name}:")
                report.append(f"  평균 시간: {sum(durations)/len(durations):.4f}초")
                report.append(f"  최소 시간: {min(durations):.4f}초")
                report.append(f"  최대 시간: {max(durations):.4f}초")
                report.append(f"  총 시간: {sum(durations):.4f}초")
                report.append(f"  호출 횟수: {len(durations)}회")

        # 시스템 정보
        report.append("\n[시스템 정보]")
        report.append(f"  CPU 사용률: {psutil.cpu_percent():.1f}%")
        report.append(f"  메모리 사용률: {psutil.virtual_memory().percent:.1f}%")

        if torch.cuda.is_available():
            report.append(
                f"  GPU 메모리: {torch.cuda.memory_allocated()/1024**2:.1f}MB"
            )

        return "\n".join(report)

    def save_report(self, filename: str = None):
        """보고서 저장"""
        if not self.config.save_reports:
            return

        report_dir = Path(self.config.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"performance_report_{int(time.time())}.json"

        filepath = report_dir / filename

        report_data = {
            "sessions": self.sessions,
            "metrics": self.metrics,
            "batch_metrics": self.batch_metrics,
            "system_info": self._get_system_info(),
            "timestamp": time.time(),
        }

        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"성능 보고서 저장: {filepath}")

    def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "cuda_available": torch.cuda.is_available(),
            "python_version": f"{psutil.version_info}",
            "timestamp": datetime.now().isoformat(),
        }

# 전역 인스턴스
_performance_tracker = None

def get_performance_manager() -> UnifiedPerformanceTracker:
    """전역 성능 관리자 반환"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = UnifiedPerformanceTracker()
    return _performance_tracker

# 기존 호환성을 위한 함수들
def get_profiler() -> UnifiedPerformanceTracker:
    """프로파일러 반환 (기존 호환성)"""
    return get_performance_manager()

def profile(name: str):
    """성능 추적 데코레이터"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with get_performance_manager().track(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator

@contextmanager
def performance_monitor(name: str):
    """성능 모니터링 컨텍스트"""
    with get_performance_manager().track(name):
        yield

# 편의 함수들
def clear_memory() -> None:
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_device_info() -> Dict[str, Any]:
    """디바이스 정보 반환"""
    info = {
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory,
            }
        )

    return info

# 기존 호환성을 위한 클래스들
class Profiler:
    """기존 호환성을 위한 Profiler 클래스"""

    def __init__(self, config=None):
        self._tracker = get_performance_manager()

    def start(self, name: str):
        """프로파일링 시작"""
        self._tracker.start(name)

    def stop(self, name: str):
        """프로파일링 중지"""
        self._tracker.stop(name)

    def get_profile_stats(self):
        """프로파일 통계 반환"""
        return self._tracker.get_profile_stats()

    def generate_report(self):
        """보고서 생성"""
        return self._tracker.generate_report()

    @contextmanager
    def profile(self, name: str):
        """프로파일링 컨텍스트"""
        with self._tracker.track(name):
            yield

    def clear(self):
        """데이터 정리"""
        self._tracker.clear()

class PerformanceTracker:
    """기존 호환성을 위한 PerformanceTracker 클래스"""

    def __init__(self):
        self._tracker = get_performance_manager()

    def start_tracking(self, name: str):
        """추적 시작"""
        self._tracker.start_tracking(name)

    def stop_tracking(self, name: str):
        """추적 중지"""
        self._tracker.stop_tracking(name)

    def get_summary(self, name: str = None):
        """요약 반환"""
        if name:
            return self._tracker.get_summary(name)
        return self._tracker.sessions

    def update_metrics(self, metrics: dict):
        """메트릭 업데이트"""
        self._tracker.update_metrics(metrics)

    def get_metrics(self, name: str = None):
        """메트릭 반환"""
        return self._tracker.get_metrics(name)

    def clear(self):
        """데이터 정리"""
        self._tracker.clear()

class MemoryTracker:
    """기존 호환성을 위한 MemoryTracker 클래스"""

    def __init__(self):
        self._tracker = get_performance_manager()

    def start(self):
        """메모리 추적 시작"""
        self._start_memory = psutil.virtual_memory().used

    def stop(self):
        """메모리 추적 중지"""
        self._end_memory = psutil.virtual_memory().used

    def get_memory_log(self):
        """메모리 로그 반환"""
        if hasattr(self, "_start_memory") and hasattr(self, "_end_memory"):
            return {
                "memory_used_mb": (self._end_memory - self._start_memory)
                / (1024 * 1024)
            }
        return {"memory_used_mb": 0.0}

    def get_memory_usage(self):
        """현재 메모리 사용량 반환"""
        return psutil.virtual_memory().used / (1024 * 1024)  # MB 단위
