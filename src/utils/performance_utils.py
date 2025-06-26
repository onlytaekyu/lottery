"""
중앙화된 성능 유틸리티 모듈

이 모듈은 프로젝트 전체에서 사용되는 프로파일링, 메모리 추적, AMP 관련 로직을 통합합니다.
"""

import cProfile
import pstats
import io
import threading
import time
import psutil
import torch
import gc
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
import logging
from contextlib import contextmanager
import json

from .error_handler import get_logger

logger = get_logger(__name__)


def profile(func: Callable) -> Callable:
    """
    함수 실행 시간과 메모리 사용량을 프로파일링하는 데코레이터

    Args:
        func: 프로파일링할 함수

    Returns:
        래핑된 함수
    """

    def wrapper(*args, **kwargs):
        # 프로파일러 시작
        profiler = cProfile.Profile()
        profiler.enable()

        # 메모리 사용량 측정 시작
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # 함수 실행
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # 메모리 사용량 측정 종료
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        # 프로파일러 종료
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # 상위 20개 함수만 출력

        # 로깅
        logger.info(f"함수 {func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        logger.info(f"함수 {func.__name__} 메모리 사용량: {mem_used:.2f}MB")
        logger.debug(f"프로파일링 결과:\n{s.getvalue()}")

        return result

    return wrapper


class MemoryTracker:
    """메모리 사용량 추적 클래스"""

    def __init__(self):
        self.start_memory = 0
        self.start_time = 0
        self.memory_log = {}

    def start(self) -> None:
        """메모리 추적 시작"""
        self.start_memory = self._get_current_memory()
        self.start_time = time.time()

    def stop(self) -> float:
        """
        메모리 추적 중지

        Returns:
            메모리 사용량(MB)
        """
        end_memory = self._get_current_memory()
        end_time = time.time()

        memory_used_mb = (end_memory - self.start_memory) / (1024 * 1024)

        self.memory_log = {
            "start_memory_mb": self.start_memory / (1024 * 1024),
            "end_memory_mb": end_memory / (1024 * 1024),
            "memory_used_mb": memory_used_mb,
            "duration_seconds": end_time - self.start_time,
        }

        return memory_used_mb

    def get_memory_log(self) -> Dict[str, float]:
        """메모리 로그 반환"""
        return self.memory_log

    def get_memory_usage(self) -> float:
        """
        현재 메모리 사용량 반환(MB)

        Returns:
            현재 메모리 사용량(MB)
        """
        current = self._get_current_memory()
        return current / (1024 * 1024)

    def _get_current_memory(self) -> int:
        """현재 메모리 사용량 반환"""
        process = psutil.Process()
        return process.memory_info().rss


class PerformanceProfiler:
    """성능 프로파일링 클래스"""

    def __init__(self):
        self.sections: Dict[str, Dict[str, Any]] = {}
        self.current_section: Optional[str] = None

    @contextmanager
    def profile_section(self, name: str):
        """성능 프로파일링 섹션"""
        try:
            self.start_section(name)
            yield
        finally:
            self.end_section()

    def start_section(self, name: str) -> None:
        """섹션 시작"""
        if name in self.sections:
            logger.warning(f"섹션 {name}이(가) 이미 시작되었습니다.")
            return

        self.current_section = name
        self.sections[name] = {
            "start_time": time.time(),
            "start_memory": self._get_current_memory(),
            "gpu_memory_start": (
                self._get_gpu_memory() if torch.cuda.is_available() else 0
            ),
        }

    def end_section(self) -> None:
        """섹션 종료"""
        if not self.current_section:
            return

        end_time = time.time()
        end_memory = self._get_current_memory()
        gpu_memory_end = self._get_gpu_memory() if torch.cuda.is_available() else 0

        self.sections[self.current_section].update(
            {
                "end_time": end_time,
                "end_memory": end_memory,
                "gpu_memory_end": gpu_memory_end,
                "duration": end_time
                - self.sections[self.current_section]["start_time"],
                "memory_used": end_memory
                - self.sections[self.current_section]["start_memory"],
                "gpu_memory_used": gpu_memory_end
                - self.sections[self.current_section]["gpu_memory_start"],
            }
        )

        self.current_section = None

    def get_section_stats(self, name: str) -> Dict[str, Any]:
        """섹션 통계 반환"""
        if name not in self.sections:
            return {}
        return self.sections[name]

    def _get_current_memory(self) -> int:
        """현재 메모리 사용량 반환"""
        process = psutil.Process()
        return process.memory_info().rss

    def _get_gpu_memory(self) -> int:
        """GPU 메모리 사용량 반환"""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated()


def clear_memory() -> None:
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device_info() -> Dict[str, Any]:
    """디바이스 정보 반환"""
    info: Dict[str, Any] = {
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
    }

    if torch.cuda.is_available():
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_cached": torch.cuda.memory_reserved(),
        }
        info.update(gpu_info)

    return info


def train_with_amp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    AMP(Automatic Mixed Precision)를 사용하여 모델을 학습합니다.

    Args:
        model: 학습할 모델
        optimizer: 최적화기
        loss_fn: 손실 함수
        data_loader: 데이터 로더
        device: 학습에 사용할 디바이스
        scaler: GradScaler 인스턴스 (None인 경우 자동 생성)

    Returns:
        학습 메트릭 딕셔너리
    """
    if scaler is None and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        optimizer.zero_grad()

        # AMP 컨텍스트 내에서 순전파
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(batch)
            loss = loss_fn(outputs, batch)

        # 역전파 및 최적화
        if device.type == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / num_batches}


class PerformanceMonitor:
    """실시간 성능 추적 클래스"""

    def __init__(self):
        self.stage_times = {}
        self.memory_usage = {}
        self.start_times = {}

    def track_stage(self, stage_name: str):
        """성능 추적 컨텍스트 매니저 반환"""
        return self.timer_context(stage_name)

    @contextmanager
    def timer_context(self, name: str):
        """타이머 컨텍스트 매니저"""
        start = time.time()
        memory_before = psutil.virtual_memory().used

        try:
            logger.info(f"[성능추적] {name} 시작")
            yield
        finally:
            duration = time.time() - start
            memory_after = psutil.virtual_memory().used
            memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB

            # 결과 저장
            self.stage_times[name] = duration
            self.memory_usage[name] = memory_increase

            logger.info(
                f"[성능추적] {name}: {duration:.2f}초, "
                f"메모리 증가: {memory_increase:.1f}MB"
            )

    def get_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        total_time = sum(self.stage_times.values())
        total_memory = sum(self.memory_usage.values())

        return {
            "total_time": total_time,
            "total_memory_increase": total_memory,
            "stage_breakdown": {"times": self.stage_times, "memory": self.memory_usage},
            "slowest_stage": (
                max(self.stage_times.items(), key=lambda x: x[1])
                if self.stage_times
                else None
            ),
            "memory_intensive_stage": (
                max(self.memory_usage.items(), key=lambda x: x[1])
                if self.memory_usage
                else None
            ),
        }

    def save_report(self, file_path: str):
        """성능 보고서 저장"""
        summary = self.get_summary()
        summary["timestamp"] = time.time()
        summary["system_info"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "memory_available": psutil.virtual_memory().available / (1024**3),  # GB
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"성능 보고서 저장 완료: {file_path}")
