"""
성능 프로파일링 도구

이 모듈은 시간 측정, 메모리 사용량 분석, CPU/GPU 사용률 추적 등의 기능을 제공합니다.
"""

import time
import os
import sys
import logging
import functools
import tracemalloc
import psutil
import torch
from typing import Callable, Dict, List, Any, Optional, Union, TypedDict, cast
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from .error_handler import get_logger
from .config_loader import ConfigProxy

# 로거 설정
logger = get_logger(__name__)


@dataclass
class ProfilerConfig:
    """프로파일러 설정"""

    enable_profiling: bool = True  # 프로파일링 활성화 여부
    time_threshold: float = 0.1  # 시간 측정 임계값 (초)
    memory_threshold: int = 1_000_000  # 메모리 사용량 임계값 (바이트)
    log_to_console: bool = True  # 콘솔 로깅 여부
    log_to_file: bool = False  # 파일 로깅 여부 - 기본값 False로 변경
    detailed_memory: bool = False  # 상세 메모리 추적 여부
    report_interval: int = 10  # 리포트 간격 (함수 호출 횟수)
    max_reports: int = 100  # 최대 리포트 수
    enable_cpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_thread_profiling: bool = True
    sampling_interval: float = 0.1
    cpu_threshold: float = 80.0
    thread_threshold: int = 100


class Profiler:
    """성능 프로파일링 클래스"""

    def __init__(self, config: ConfigProxy):
        self.config = config
        self.profiles: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
        # 로거 설정 - 공유 로거 사용
        self.logger = logger

    @contextmanager
    def profile(self, name: str):
        """코드 섹션 프로파일링"""
        try:
            self.start(name)
            yield
        finally:
            self.stop(name)

    def start(self, name: str) -> None:
        """프로파일링 시작"""
        if name not in self.profiles:
            self.profiles[name] = []
        self.start_times[name] = time.time()

    def stop(self, name: str) -> None:
        """프로파일링 중지"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.profiles[name].append(duration)
            del self.start_times[name]

    def get_profile_stats(self) -> Dict[str, Dict[str, float]]:
        """프로파일링 통계 반환"""
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
        """메모리 사용량 통계 반환"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        return {
            "allocated": torch.cuda.memory_allocated(),
            "cached": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_cached": torch.cuda.max_memory_reserved(),
        }

    def clear(self) -> None:
        """프로파일링 데이터 초기화"""
        self.profiles.clear()
        self.start_times.clear()

    def generate_report(self) -> str:
        """프로파일링 리포트 생성"""
        profile_stats = self.get_profile_stats()
        memory_stats = self.get_memory_stats()

        report = ["=== 성능 프로파일링 리포트 ==="]

        # 프로파일링 통계
        report.append("\n[프로파일링 통계]")
        for name, stats in profile_stats.items():
            report.append(f"\n{name}:")
            for stat_name, value in stats.items():
                report.append(f"  {stat_name}: {value:.4f}초")

        # 메모리 통계
        report.append("\n[메모리 통계]")
        for name, value in memory_stats.items():
            if isinstance(value, (int, float)):
                report.append(f"  {name}: {value / 1024**2:.2f}MB")
            else:
                report.append(f"  {name}: {value}")

        return "\n".join(report)

    def profile_function(self, func=None, *, name=None):
        """
        함수 프로파일링 데코레이터

        Args:
            func: 프로파일링할 함수
            name: 측정 이름 (기본값: 함수 이름)

        Returns:
            프로파일링 데코레이터가 적용된 함수
        """

        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                profile_name = name or f.__qualname__
                with self.profile(profile_name):
                    return f(*args, **kwargs)

            return wrapper

        if func is None:
            return decorator
        return decorator(func)

    def get_stats(self, span_name: Optional[str] = None) -> Dict[str, Any]:
        """
        프로파일링 통계 가져오기

        Args:
            span_name: 특정 구간 이름 (None이면 모든 구간)

        Returns:
            프로파일링 통계
        """
        if span_name is not None:
            if span_name not in self.profiles:
                return {}

            durations = self.profiles[span_name]
            if not durations:
                return {}

            return {
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "total": sum(durations),
                "count": len(durations),
            }

        # 모든 구간의 통계
        all_stats = {}
        for name, durations in self.profiles.items():
            if durations:
                all_stats[name] = self.get_stats(name)
        return all_stats

    def print_report(self, span_name: Optional[str] = None):
        """
        프로파일링 보고서 출력

        Args:
            span_name: 특정 구간 이름 (None이면 모든 구간)
        """
        if span_name is not None:
            stats = self.get_stats(span_name)
            if not stats:
                print(f"No statistics for {span_name}")
                return

            print(f"Statistics for {span_name}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.4f}초")
        else:
            all_stats = self.get_stats()
            if not all_stats:
                print("No profiling statistics available")
                return

            print("Profiling Report:")
            print("================")

            # 평균 시간 기준 정렬
            sorted_stats = sorted(
                all_stats.items(), key=lambda x: x[1]["mean"], reverse=True
            )

            for name, stats in sorted_stats:
                print(f"\n{name}:")
                for stat_name, value in stats.items():
                    print(f"  {stat_name}: {value:.4f}초")

    def reset(self):
        """측정 통계 초기화"""
        self.profiles.clear()
        self.start_times.clear()

    def log_report(self, span_name: Optional[str] = None):
        """
        프로파일링 보고서를 로그에 기록

        Args:
            span_name: 특정 구간 이름 (None이면 모든 구간)
        """
        if span_name is not None:
            stats = self.get_stats(span_name)
            if not stats:
                self.logger.info(f"No statistics for {span_name}")
                return

            self.logger.info(f"Statistics for {span_name}:")
            for stat_name, value in stats.items():
                self.logger.info(f"  {stat_name}: {value:.4f}초")
        else:
            all_stats = self.get_stats()
            if not all_stats:
                self.logger.info("No profiling statistics available")
                return

            self.logger.info("Profiling Report:")

            # 평균 시간 기준 정렬
            sorted_stats = sorted(
                all_stats.items(), key=lambda x: x[1]["mean"], reverse=True
            )

            for name, stats in sorted_stats:
                self.logger.info(f"{name}:")
                for stat_name, value in stats.items():
                    self.logger.info(f"  {stat_name}: {value:.4f}초")

    def get_execution_time_safe(self, name: str) -> float:
        """
        안전하게 특정 구간의 총 실행 시간을 반환합니다.

        Args:
            name: 구간 이름

        Returns:
            float: 총 실행 시간 (초), 오류 발생 시 0.0 반환
        """
        try:
            return self.get_stats(name).get("total", 0.0)
        except Exception:
            self.logger.warning(f"구간 '{name}'의 실행 시간을 가져올 수 없습니다.")
            return 0.0


# 전역 프로파일러 인스턴스
_global_profiler = None


def get_profiler() -> Profiler:
    """
    전역 프로파일러 인스턴스 가져오기

    Returns:
        Profiler 인스턴스
    """
    global _global_profiler
    if _global_profiler is None:
        config = ConfigProxy({})  # 빈 설정으로 초기화
        _global_profiler = Profiler(config)
    return _global_profiler


@contextmanager
def profile(span_name: Union[str, Callable]):
    """
    코드 블록 프로파일링

    Args:
        span_name: 프로파일링 구간 이름 또는 프로파일링할 함수

    Example:
        with profile("my_operation"):
            # code to profile

        @profile
        def my_function():
            # code to profile

        @profile("custom_name")
        def another_function():
            # code to profile
    """
    # 데코레이터로 사용된 경우
    if callable(span_name) and not isinstance(span_name, str):
        return profile_function(span_name)

    # 이름이 지정된 컨텍스트 매니저로 사용된 경우
    profiler = get_profiler()
    with profiler.profile(span_name):
        yield


def profile_function(func=None, *, name=None):
    """
    함수 프로파일링 데코레이터

    Args:
        func: 프로파일링할 함수
        name: 측정 이름 (기본값: 함수 이름)

    Returns:
        프로파일링 데코레이터가 적용된 함수

    Example:
        @profile_function
        def my_function():
            # code to profile

        @profile_function(name="custom_name")
        def another_function():
            # code to profile
    """
    profiler = get_profiler()
    return profiler.profile_function(func, name=name)


def print_report(span_name: Optional[str] = None):
    """
    프로파일링 보고서 출력

    Args:
        span_name: 특정 구간 이름 (None이면 모든 구간)
    """
    profiler = get_profiler()
    profiler.print_report(span_name)


def reset_profiler():
    """전역 프로파일러 초기화"""
    profiler = get_profiler()
    profiler.reset()


def configure_profiler(config: ProfilerConfig):
    """
    프로파일러 설정 (전역 프로파일러 사용)

    Args:
        config: 프로파일러 설정
    """
    global _global_profiler
    config_dict = {
        "enable_profiling": config.enable_profiling,
        "time_threshold": config.time_threshold,
        "memory_threshold": config.memory_threshold,
        "log_to_console": config.log_to_console,
        "log_to_file": config.log_to_file,
        "detailed_memory": config.detailed_memory,
        "report_interval": config.report_interval,
        "max_reports": config.max_reports,
        "enable_cpu_profiling": config.enable_cpu_profiling,
        "enable_memory_profiling": config.enable_memory_profiling,
        "enable_thread_profiling": config.enable_thread_profiling,
        "sampling_interval": config.sampling_interval,
        "cpu_threshold": config.cpu_threshold,
        "thread_threshold": config.thread_threshold,
    }
    _global_profiler = Profiler(ConfigProxy(config_dict))
