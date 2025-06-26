"""
성능 추적 시스템 (Performance Tracker)

이 모듈은 모델 학습 및 추론 과정에서의 성능 지표를 수집, 분석 및 시각화하는 기능을 제공합니다.
시간 측정, 메모리 사용량 추적, 계산 효율성 분석 등의 기능을 포함합니다.
"""

import time
import logging
import numpy as np
import torch
import psutil
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Callable, Tuple, Union, Generator
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import pstats
import io
import threading
import functools
import subprocess
import json
import os
from dataclasses import dataclass, field

# 로거 설정
from .error_handler import get_logger
from .config_loader import ConfigProxy

logger = get_logger(__name__)


@dataclass
class ProfilerConfig:
    """프로파일러 설정"""

    enable_cpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_thread_profiling: bool = True
    sampling_interval: float = 0.1
    memory_threshold: int = 2 << 30  # 2GB (1GB에서 2GB로 증가)
    cpu_threshold: float = 80.0
    thread_threshold: int = 100
    # 메모리 사용량 경고 제외 대상 작업명 목록
    memory_warning_exclude: List[str] = field(
        default_factory=lambda: ["optimize_model", "cleanup"]
    )


class Profiler:
    """성능 프로파일러"""

    def __init__(self, config: Optional[ProfilerConfig] = None):
        """
        프로파일러 초기화

        Args:
            config: 프로파일러 설정
        """
        self.config = config or ProfilerConfig()
        self.logger = get_logger(__name__)
        self._profiles = {}
        self._start_times = {}
        self._is_profiling = False
        self._lock = threading.Lock()
        self._memory_stats = []
        self._thread_stats = []
        self._cpu_profiler = pstats.Stats()
        self._start_time = time.time()

        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()
        else:
            self._device = None

    def start(self, name: str) -> None:
        """
        프로파일링 시작

        Args:
            name: 프로파일링 이름
        """
        if not self._is_profiling:
            self._is_profiling = True
            self._start_times[name] = time.time()
            self._profiles[name] = {
                "cpu_usage": [],
                "memory_usage": [],
                "thread_count": [],
                "duration": 0.0,
            }

    def stop(self, name: str) -> None:
        """
        프로파일링 종료

        Args:
            name: 프로파일링 이름
        """
        if self._is_profiling and name in self._start_times:
            end_time = time.time()
            duration = end_time - self._start_times[name]
            self._profiles[name]["duration"] = duration
            self._is_profiling = False

    def get_duration(self, name: str) -> float:
        """
        프로파일링 기간 조회

        Args:
            name: 프로파일링 이름

        Returns:
            프로파일링 기간 (초)
        """
        return self._profiles.get(name, {}).get("duration", 0.0)

    def get_profile(self, name: str) -> Dict[str, Any]:
        """
        프로파일링 결과 조회

        Args:
            name: 프로파일링 이름

        Returns:
            프로파일링 결과
        """
        return self._profiles.get(name, {})

    def reset(self) -> None:
        """프로파일링 결과 초기화"""
        self._profiles.clear()
        self._start_times.clear()
        self._is_profiling = False

    def log_profile(self, name: str) -> None:
        """
        프로파일링 결과 로깅

        Args:
            name: 프로파일링 이름
        """
        try:
            if name in self._profiles:
                profile = self._profiles[name]
                self.logger.info(f"프로파일링 결과 - {name}:")
                self.logger.info(f"  기간: {profile['duration']:.2f}초")
                if profile["cpu_usage"]:
                    self.logger.info(
                        f"  CPU 사용률: {np.mean(profile['cpu_usage']):.2f}%"
                    )
                if profile["memory_usage"]:
                    self.logger.info(
                        f"  메모리 사용량: {np.mean(profile['memory_usage']):.2f}GB"
                    )
                if profile["thread_count"]:
                    self.logger.info(
                        f"  스레드 수: {np.mean(profile['thread_count']):.2f}"
                    )
        except Exception as e:
            self.logger.error(f"프로파일링 결과 로깅 중 오류 발생: {str(e)}")

    @contextmanager
    def profile(self, name: str):
        """코드 블록 프로파일링"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_threads = threading.active_count()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            end_threads = threading.active_count()

            duration = end_time - start_time
            memory_used = end_memory - start_memory
            threads_created = end_threads - start_threads

            with self._lock:
                self._memory_stats.append(
                    {
                        "name": name,
                        "timestamp": time.time(),
                        "memory_used": memory_used,
                    }
                )
                self._thread_stats.append(
                    {
                        "name": name,
                        "timestamp": time.time(),
                        "threads_created": threads_created,
                    }
                )

            # 제외 목록에 있는 작업에 대해서는 경고 표시하지 않음
            if name not in self.config.memory_warning_exclude:
                if memory_used > self.config.memory_threshold * 1.5:
                    logger.warning(
                        f"메모리 사용량 임계값 크게 초과: {name} ({memory_used / (1 << 20):.2f}MB)"
                    )
                elif memory_used > self.config.memory_threshold:
                    logger.debug(
                        f"메모리 프로파일: {name} ({memory_used / (1 << 20):.2f}MB, "
                        f"임계값: {self.config.memory_threshold / (1 << 20):.2f}MB)"
                    )
            else:
                # 제외 목록에 있는 작업은 디버그 수준으로만 기록
                if memory_used > self.config.memory_threshold:
                    logger.debug(
                        f"높은 메모리 사용 (무시됨): {name} ({memory_used / (1 << 20):.2f}MB)"
                    )

            if threads_created > self.config.thread_threshold:
                logger.warning(f"스레드 생성 임계값 초과: {name} ({threads_created})")

    def profile_function(self, func: Callable) -> Callable:
        """함수 프로파일링 데코레이터"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile(func.__name__):
                return func(*args, **kwargs)

        return wrapper

    def start_cpu_profiling(self):
        """CPU 프로파일링 시작"""
        if self.config.enable_cpu_profiling:
            self._cpu_profiler.enable()  # type: ignore

    def stop_cpu_profiling(self):
        """CPU 프로파일링 중지"""
        if self.config.enable_cpu_profiling:
            self._cpu_profiler.disable()  # type: ignore

    def get_cpu_stats(self) -> str:
        """CPU 프로파일링 통계 반환"""
        if not self.config.enable_cpu_profiling:
            return "CPU 프로파일링이 비활성화되어 있습니다."

        s = io.StringIO()
        ps = pstats.Stats(self._cpu_profiler, stream=s).sort_stats("cumulative")  # type: ignore
        ps.print_stats()
        return s.getvalue()

    def get_memory_stats(self) -> List[Dict[str, Any]]:
        """메모리 사용 통계 반환"""
        return self._memory_stats.copy()

    def get_thread_stats(self) -> List[Dict[str, Any]]:
        """스레드 통계 반환"""
        return self._thread_stats.copy()

    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 리소스 통계 반환"""
        return {
            "cpu_percent": psutil.Process().cpu_percent(),
            "memory_percent": psutil.Process().memory_percent(),
            "num_threads": psutil.Process().num_threads(),
            "num_fds": (lambda p: None if not hasattr(p, "num_fds") else p.num_fds())(  # type: ignore
                psutil.Process()
            ),
            "create_time": psutil.Process().create_time(),
            "uptime": time.time() - self._start_time,
        }


class GPUMonitor:
    """GPU 메모리 및 사용량 모니터링 클래스"""

    def __init__(self, log_interval: int = 30, log_file: Optional[str] = None):
        """
        GPU 모니터 초기화

        Args:
            log_interval: 로그 기록 간격(초)
            log_file: 사용하지 않음 (이전 버전과의 호환성을 위해 유지)
        """
        self.logger = get_logger(__name__)
        self.log_interval = log_interval

        # 별도의 로그 파일을 사용하지 않음
        self.log_file = None

        self.running = False
        self.monitoring_thread = None
        self.use_pynvml = False

        # PYNVML 사용 가능한지 확인
        try:
            # pynvml 모듈 임포트 시도
            try:
                import pynvml

                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.use_pynvml = True
                self.logger.info("GPU 모니터링 초기화 완료 - NVML 사용")
            except ImportError:
                self.logger.info(
                    "pynvml 모듈을 찾을 수 없습니다 - nvidia-smi 명령어로 대체"
                )
                self.use_pynvml = False
                self.pynvml = None
            except Exception as e:
                self.logger.info(
                    f"GPU 모니터링 초기화 중 오류: {str(e)} - nvidia-smi 명령어로 대체"
                )
                self.use_pynvml = False
                self.pynvml = None
        except Exception as e:
            self.logger.warning(f"GPU 모니터링 초기화 실패: {str(e)}")
            self.use_pynvml = False
            self.pynvml = None

    def start_monitoring(self):
        """GPU 모니터링 시작"""
        if self.running:
            self.logger.warning("이미 GPU 모니터링이 실행 중입니다.")
            return

        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"GPU 모니터링 시작 (간격: {self.log_interval}초)")

    def stop_monitoring(self):
        """GPU 모니터링 중지"""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        self.logger.info("GPU 모니터링 중지")

        # PYNVML 사용 중인 경우 종료 처리
        if self.use_pynvml and hasattr(self, "pynvml") and self.pynvml is not None:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass

    def _monitoring_loop(self):
        """모니터링 메인 루프"""
        try:
            while self.running:
                # 시스템 자원 사용량 측정
                stats = self.get_system_usage()

                # 로그에 기록
                self._write_log(stats)

                # 로그 간격만큼 대기
                time.sleep(self.log_interval)
        except Exception as e:
            self.logger.error(f"GPU 모니터링 중 오류 발생: {str(e)}")
            self.running = False

    def _write_log(self, stats: Dict[str, Any]):
        """측정 결과를 로그에 기록"""
        try:
            # 로그 파일에 직접 쓰는 대신 로거 사용
            gpu_info = stats.get("gpu_pytorch", [])
            timestamp = stats.get("timestamp", "")

            # 주요 정보만 로그에 기록
            if gpu_info:
                for gpu in gpu_info:
                    self.logger.info(
                        f"GPU {gpu['index']} ({gpu['name']}): "
                        f"할당: {gpu['memory_allocated_gb']:.2f}GB, "
                        f"예약: {gpu['memory_reserved_gb']:.2f}GB"
                    )
            else:
                cpu_percent = stats.get("cpu_percent", 0)
                memory_percent = stats.get("memory_percent", 0)
                self.logger.info(
                    f"시스템 자원: CPU {cpu_percent:.1f}%, "
                    f"메모리 {memory_percent:.1f}%"
                )
        except Exception as e:
            self.logger.error(f"GPU 모니터링 로그 기록 실패: {str(e)}")

    def get_system_usage(self) -> Dict[str, Any]:
        """
        시스템 자원 사용량 측정

        Returns:
            Dict[str, Any]: 시스템 자원 사용량 정보
        """
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
        }

        # PyTorch GPU 정보
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu_info.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": torch.cuda.memory_allocated(i),
                        "memory_reserved": torch.cuda.memory_reserved(i),
                        "memory_allocated_gb": torch.cuda.memory_allocated(i)
                        / (1024**3),
                        "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                    }
                )
            result["gpu_pytorch"] = gpu_info

        # NVML 정보 (사용 가능한 경우)
        if self.use_pynvml and self.pynvml is not None:
            try:
                nvml_info = []
                device_count = self.pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

                    nvml_info.append(
                        {
                            "index": i,
                            "memory_total": int(memory_info.total),
                            "memory_used": int(memory_info.used),
                            "memory_free": int(memory_info.free),
                            "memory_total_gb": int(memory_info.total) / (1024**3),
                            "memory_used_gb": int(memory_info.used) / (1024**3),
                            "memory_free_gb": int(memory_info.free) / (1024**3),
                            "gpu_utilization": int(utilization.gpu),
                            "memory_utilization": int(utilization.memory),
                        }
                    )
                result["gpu_nvml"] = nvml_info
            except Exception as e:
                self.logger.debug(f"NVML 정보 수집 중 오류: {str(e)}")

        return result

    def get_current_usage(self) -> Dict[str, Any]:
        """
        현재 시스템 자원 사용량 측정 (즉시 실행)

        Returns:
            현재 시스템 자원 사용량 정보
        """
        return self.get_system_usage()


class PerformanceTracker:
    """
    성능 추적 시스템

    모델 학습, 추론 및 데이터 처리 과정에서의 성능 측정 및 분석 기능을 제공합니다.
    다음 지표를 측정합니다:
    - 실행 시간 (전체 및 단계별)
    - GPU/CPU 메모리 사용량
    - 배치 처리 시간
    - 학습/추론 속도 (샘플/초)
    - GPU 활용률
    """

    def __init__(self):
        """
        성능 추적기 초기화
        """
        self.logger = logger
        self._metrics = {}
        self._start_times = {}
        self._is_tracking = False
        self.active_tracking = set()
        self.gpu_available = torch.cuda.is_available()

        # 측정 간격 (초)
        self.interval = 0.5
        self.last_memory_check = 0

        # 프로파일러 초기화
        self.profiler = Profiler()

        # GPU 모니터 초기화
        self.gpu_monitor = GPUMonitor(log_interval=30)

        # 메모리 모니터링 이력
        self.memory_history = []

        logger.info("성능 추적기가 초기화되었습니다.")

    def start_tracking(self, name: str):
        """
        특정 작업의 성능 추적 시작

        Args:
            name: 추적할 작업 이름
        """
        if not self._is_tracking:
            self._is_tracking = True
            self._start_times[name] = time.time()
            self._metrics[name] = {
                "duration": [],
                "gpu_memory": [],
                "cpu_memory": [],
                "gpu_utilization": [],
                "batch_times": [],
                "samples_per_second": [],
                "timestamps": [],
            }

            # 초기 메모리 상태 기록
            self._record_memory_status(name)

            # 프로파일러도 시작
            self.profiler.start(name)

            logger.debug(f"'{name}' 작업 성능 추적을 시작합니다.")

    def stop_tracking(self, name: str):
        """
        특정 작업의 성능 추적 종료

        Args:
            name: 추적 중인 작업 이름
        """
        if self._is_tracking and name in self._start_times:
            end_time = time.time()
            duration = end_time - self._start_times[name]
            self._metrics[name]["duration"] = duration
            self._is_tracking = False

            # 최종 상태 기록
            self._record_memory_status(name)

            # 프로파일러도 중지
            self.profiler.stop(name)

            logger.debug(
                f"'{name}' 작업 성능 추적을 종료합니다. 소요 시간: {duration:.2f}초"
            )

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        성능 메트릭 업데이트

        Args:
            metrics: 업데이트할 메트릭
        """
        try:
            for name, value in metrics.items():
                if name in self._metrics:
                    self._metrics[name].update(value)
                else:
                    self._metrics[name] = value
        except Exception as e:
            self.logger.error(f"성능 메트릭 업데이트 중 오류 발생: {str(e)}")

    def record_batch(self, name: str, batch_size: int, batch_time: float):
        """
        배치 처리 시간 기록

        Args:
            name: 작업 이름
            batch_size: 배치 크기
            batch_time: 배치 처리 시간 (초)
        """
        if name not in self._metrics:
            self.start_tracking(name)

        self._metrics[name]["batch_times"].append(batch_time)

        # 배치당 처리 속도 계산
        if batch_time > 0:
            samples_per_second = batch_size / batch_time
            self._metrics[name]["samples_per_second"].append(samples_per_second)

        # 주기적으로 메모리 상태 업데이트
        current_time = time.time()
        if current_time - self.last_memory_check >= self.interval:
            self._record_memory_status(name)
            self.last_memory_check = current_time

    def _record_memory_status(self, name: str):
        """
        현재 메모리 상태 기록

        Args:
            name: 작업 이름
        """
        # CPU 메모리 사용량 측정
        cpu_percent = psutil.virtual_memory().percent
        self._metrics[name]["cpu_memory"].append(cpu_percent)

        # GPU가 사용 가능한 경우 GPU 메모리 측정
        if self.gpu_available:
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / (
                    1024**3
                )  # GB 단위
                gpu_memory_reserved = torch.cuda.memory_reserved() / (
                    1024**3
                )  # GB 단위
                gpu_utilization = (
                    gpu_memory_allocated / gpu_memory_reserved
                    if gpu_memory_reserved > 0
                    else 0
                )

                self._metrics[name]["gpu_memory"].append(gpu_memory_allocated)
                self._metrics[name]["gpu_utilization"].append(
                    gpu_utilization * 100
                )  # 백분율
            except Exception as e:
                logger.error(f"GPU 메모리 측정 중 오류 발생: {e}")

    def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        성능 메트릭 조회

        Args:
            name: 메트릭 이름 (None이면 전체 메트릭 반환)

        Returns:
            성능 메트릭
        """
        if name is None:
            return self._metrics
        return self._metrics.get(name, {})

    def get_summary(self, name: str) -> Dict[str, float]:
        """
        특정 작업의 성능 지표 요약 계산

        Args:
            name: 작업 이름

        Returns:
            요약 통계가 포함된 딕셔너리
        """
        if name not in self._metrics:
            return {}

        metrics = self._metrics[name]
        summary = {}

        # 기간 통계
        if metrics["duration"]:
            summary["avg_duration"] = np.mean(metrics["duration"])
            summary["total_duration"] = np.sum(metrics["duration"])

        # 메모리 통계
        if metrics["cpu_memory"]:
            summary["avg_cpu_memory"] = np.mean(metrics["cpu_memory"])
            summary["max_cpu_memory"] = np.max(metrics["cpu_memory"])

        if self.gpu_available and metrics["gpu_memory"]:
            summary["avg_gpu_memory_gb"] = np.mean(metrics["gpu_memory"])
            summary["max_gpu_memory_gb"] = np.max(metrics["gpu_memory"])
            summary["avg_gpu_utilization"] = np.mean(metrics["gpu_utilization"])

        # 처리 속도 통계
        if metrics["samples_per_second"]:
            summary["avg_samples_per_second"] = np.mean(metrics["samples_per_second"])
            summary["max_samples_per_second"] = np.max(metrics["samples_per_second"])

        if metrics["batch_times"]:
            summary["avg_batch_time"] = np.mean(metrics["batch_times"])
            summary["min_batch_time"] = np.min(metrics["batch_times"])

        return summary

    def plot_metrics(self, name: str, save_path: Optional[str] = None):
        """
        특정 작업의 성능 지표 시각화

        Args:
            name: 작업 이름
            save_path: 저장 경로 (None인 경우 화면에 표시)
        """
        if name not in self._metrics:
            logger.warning(f"'{name}' 작업에 대한 지표가 없습니다.")
            return

        metrics = self._metrics[name]

        # 시각화를 위한 그래프 생성
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{name} 성능 지표", fontsize=16)

        # 실행 시간 그래프
        if metrics["duration"]:
            axs[0, 0].plot(metrics["duration"], "b-", label="실행 시간")
            axs[0, 0].set_title("실행 시간 (초)")
            axs[0, 0].set_xlabel("실행 인덱스")
            axs[0, 0].set_ylabel("시간 (초)")
            axs[0, 0].grid(True)

        # 메모리 사용량 그래프
        if metrics["cpu_memory"]:
            ax1 = axs[0, 1]
            ax1.plot(metrics["cpu_memory"], "r-", label="CPU 메모리")
            ax1.set_title("메모리 사용량")
            ax1.set_xlabel("측정 인덱스")
            ax1.set_ylabel("CPU 메모리 (%)", color="r")
            ax1.tick_params(axis="y", labelcolor="r")
            ax1.grid(True)

            if self.gpu_available and metrics["gpu_memory"]:
                ax2 = ax1.twinx()
                ax2.plot(metrics["gpu_memory"], "g-", label="GPU 메모리")
                ax2.set_ylabel("GPU 메모리 (GB)", color="g")
                ax2.tick_params(axis="y", labelcolor="g")

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # 배치 처리 시간 그래프
        if metrics["batch_times"]:
            axs[1, 0].plot(metrics["batch_times"], "m-", label="배치 시간")
            axs[1, 0].set_title("배치 처리 시간")
            axs[1, 0].set_xlabel("배치 인덱스")
            axs[1, 0].set_ylabel("시간 (초)")
            axs[1, 0].grid(True)

        # 처리 속도 그래프
        if metrics["samples_per_second"]:
            axs[1, 1].plot(metrics["samples_per_second"], "c-", label="처리 속도")
            axs[1, 1].set_title("샘플 처리 속도")
            axs[1, 1].set_xlabel("측정 인덱스")
            axs[1, 1].set_ylabel("샘플/초")
            axs[1, 1].grid(True)

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        if save_path:
            # 저장 디렉토리 생성
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"성능 지표 그래프를 {save_path}에 저장했습니다.")
        else:
            plt.show()

        plt.close()

    def reset(self, name: Optional[str] = None):
        """
        성능 지표 초기화

        Args:
            name: 초기화할 작업 이름 (None이면 모든 작업)
        """
        if name is not None:
            if name in self._metrics:
                self._metrics[name] = {
                    "duration": [],
                    "gpu_memory": [],
                    "cpu_memory": [],
                    "gpu_utilization": [],
                    "batch_times": [],
                    "samples_per_second": [],
                    "timestamps": [],
                }
                logger.debug(f"'{name}' 작업의 성능 지표를 초기화했습니다.")

                # 프로파일러 초기화
                self.profiler.reset()
        else:
            self._metrics.clear()
            self._start_times.clear()
            self._is_tracking = False

            # 프로파일러 초기화
            self.profiler.reset()

            logger.debug("모든 성능 지표를 초기화했습니다.")

    def save_metrics(self, filepath: str, name: Optional[str] = None):
        """
        성능 지표를 파일로 저장

        Args:
            filepath: 저장할 파일 경로
            name: 저장할 작업 이름 (None이면 모든 작업)
        """
        import json

        # 저장할 데이터 준비
        if name is not None:
            if name not in self._metrics:
                logger.warning(f"'{name}' 작업에 대한 지표가 없습니다.")
                return
            data = {name: self._metrics[name]}
        else:
            data = self._metrics

        # NumPy 배열을 목록으로 변환
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj

        save_data = convert_numpy(data)

        # 파일 저장
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logger.info(f"성능 지표를 {filepath}에 저장했습니다.")

    def load_metrics(self, filepath: str):
        """
        파일에서 성능 지표 로드

        Args:
            filepath: 로드할 파일 경로

        Returns:
            로드 성공 여부
        """
        import json

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)

            # 기존 지표와 병합
            for name, metrics in loaded_data.items():
                self._metrics[name] = metrics

            logger.info(f"{filepath}에서 성능 지표를 로드했습니다.")
            return True

        except Exception as e:
            logger.error(f"성능 지표 로드 중 오류 발생: {e}")
            return False

    def log_metrics(self) -> None:
        """성능 메트릭 로깅"""
        try:
            for name, metrics in self._metrics.items():
                self.logger.info(f"성능 메트릭 - {name}:")
                for key, value in metrics.items():
                    self.logger.info(f"  {key}: {value}")

            # 프로파일링 결과도 로깅
            for name in self._metrics:
                self.profiler.log_profile(name)
        except Exception as e:
            self.logger.error(f"성능 메트릭 로깅 중 오류 발생: {str(e)}")

    @contextmanager
    def track(self, operation_name: str) -> Generator[None, None, None]:
        """Track execution time of an operation."""
        start_time = time.time()
        # 키가 없으면 기본 구조 생성
        if operation_name not in self._metrics:
            self._metrics[operation_name] = {
                "duration": [],
                "gpu_memory": [],
                "cpu_memory": [],
                "gpu_utilization": [],
                "batch_times": [],
                "samples_per_second": [],
                "timestamps": [],
            }
        # 프로파일러도 함께 시작
        with self.profiler.profile(operation_name):
            try:
                yield
            finally:
                end_time = time.time()
                duration = end_time - start_time
                self._metrics[operation_name]["duration"].append(duration)
                self.logger.debug(
                    f"Operation '{operation_name}' took {duration:.2f} seconds"
                )

    def get_metric(self, operation_name: str) -> Optional[float]:
        """Get metric for a specific operation."""
        return self._metrics.get(operation_name, {}).get("duration")

    def clear_metrics(self):
        """Clear all tracked metrics."""
        self._metrics.clear()
        self.profiler.reset()

    # 프로파일링 관련 추가 메서드들
    def profile_function(self, func):
        """함수 프로파일링 데코레이터"""
        return self.profiler.profile_function(func)

    def get_profile(self, name: str) -> Dict[str, Any]:
        """프로파일링 결과 조회"""
        return self.profiler.get_profile(name)

    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 리소스 통계 반환"""
        return self.profiler.get_system_stats()

    def start_gpu_monitoring(self):
        """GPU 모니터링 시작"""
        self.gpu_monitor.start_monitoring()

    def stop_gpu_monitoring(self):
        """GPU 모니터링 중지"""
        self.gpu_monitor.stop_monitoring()

    def get_gpu_memory_usage(self) -> Dict[str, Any]:
        """
        현재 GPU 메모리 사용량 측정

        Returns:
            GPU 메모리 사용량 정보
        """
        return self.gpu_monitor.get_current_usage()

    def get_memory_usage(self, config=None) -> Dict[str, float]:
        """
        현재 메모리 사용량 측정

        Args:
            config: 설정 객체 (선택적, 이전 버전 호환성을 위해 유지)

        Returns:
            메모리 사용량 정보
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()

        return {
            "rss_gb": memory_info.rss / (1024**3),  # 실제 사용 메모리 (GB)
            "vms_gb": memory_info.vms / (1024**3),  # 가상 메모리 (GB)
            "percent": process.memory_percent(),  # 프로세스 메모리 사용률
            "system_used_gb": system_memory.used / (1024**3),  # 시스템 사용 메모리
            "system_total_gb": system_memory.total / (1024**3),  # 시스템 전체 메모리
            "system_percent": system_memory.percent,  # 시스템 메모리 사용률
        }


def save_performance_report(
    report_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
) -> str:
    """
    성능 보고서 저장 (비활성화됨)

    Args:
        report_data: 보고서 데이터
        config: 설정 객체 또는 딕셔너리

    Returns:
        빈 문자열 (항상 비활성화됨)
    """
    logger.info("성능 보고서 생성이 비활성화되어 있습니다.")
    return ""


def update_performance_metrics(
    metrics: Dict[str, Any], model_name: str, config: Optional[Dict[str, Any]] = None
) -> None:
    """
    성능 지표 업데이트 및 보고서 저장

    Args:
        metrics: 성능 지표
        model_name: 모델 이름
        config: 설정
    """
    # 현재 시간
    current_time = datetime.now()

    # 보고서 데이터 생성
    report_data = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "metrics": metrics,
        "config": config,
    }

    # 보고서 저장
    save_performance_report(report_data, config)
