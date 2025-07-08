"""
통합 성능 최적화 엔진 (Unified Performance Engine)

기존 performance_optimizer.py와 compute_strategy.py의 중복 기능을 통합하여
GPU > 멀티쓰레드 > CPU 우선순위 처리와 성능 모니터링을 단일 시스템으로 제공합니다.
"""

import os
import time
import torch
import psutil
import threading
import numpy as np
import json
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .unified_config import get_config
from .unified_logging import get_logger
from .unified_memory_manager import get_unified_memory_manager
from .enhanced_process_pool import get_enhanced_process_pool
from .cuda_optimizers import get_cuda_optimizer
from .auto_recovery_system import auto_recoverable
from .factory import get_singleton_instance

logger = get_logger(__name__)


class ComputeStrategy(Enum):
    """연산 전략 열거형"""

    GPU = "gpu"
    MULTITHREAD_CPU = "multithread_cpu"
    SINGLE_CPU = "single_cpu"


class TaskType(Enum):
    """작업 유형 열거형"""

    TENSOR_COMPUTATION = "tensor_computation"
    DATA_PROCESSING = "data_processing"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"


@dataclass
class SystemState:
    """시스템 상태 정보"""

    gpu_available: bool
    gpu_memory_free: float  # GB
    gpu_utilization: float  # 0-100%
    cpu_count: int
    cpu_usage: float  # 0-100%
    memory_available: float  # GB
    memory_usage: float  # 0-100%


@dataclass
class ComputeRequest:
    """연산 요청 정보"""

    data_size: int  # bytes
    task_type: TaskType
    priority: int = 1  # 1=low, 5=high
    requires_gpu: bool = False
    max_workers: Optional[int] = None


class CUDAStreamManager:
    """CUDA 스트림 관리자 - GPU 최적화 강화"""

    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size
        self.streams = []
        self.current_stream = 0
        self.gpu_available = torch.cuda.is_available()

        if self.gpu_available:
            self.streams = [torch.cuda.Stream() for _ in range(pool_size)]
            logger.info(f"✅ CUDA 스트림 풀 초기화 완료: {pool_size}개 스트림")
        else:
            logger.warning("GPU 미사용 환경 - CUDA 스트림 관리자 비활성화")

    def get_next_stream(self) -> Optional[torch.cuda.Stream]:
        """다음 사용 가능한 CUDA 스트림 반환"""
        if not self.gpu_available or not self.streams:
            return None

        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return stream

    @contextmanager
    def stream_context(self):
        """CUDA 스트림 컨텍스트 관리자"""
        if not self.gpu_available:
            yield None
            return

        stream = self.get_next_stream()
        if stream is None:
            yield None
            return

        try:
            with torch.cuda.stream(stream):
                yield stream
        finally:
            if stream:
                stream.synchronize()

    def synchronize_all(self):
        """모든 스트림 동기화"""
        if self.gpu_available:
            for stream in self.streams:
                stream.synchronize()

    def get_stats(self) -> Dict[str, Any]:
        """스트림 관리자 통계"""
        return {
            "pool_size": self.pool_size,
            "gpu_available": self.gpu_available,
            "current_stream_index": self.current_stream,
            "active_streams": len(self.streams),
        }


class SystemMonitor:
    """실시간 시스템 상태 모니터링 (Enhanced GPU Monitoring)"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_history = []
        self.monitoring_interval = 0.1  # 100ms
        self.max_history_size = 100

    def get_current_state(self) -> SystemState:
        """현재 시스템 상태 반환"""
        # CPU 정보
        cpu_count = os.cpu_count() or 1
        cpu_usage = psutil.cpu_percent(interval=0.1)

        # 메모리 정보
        memory = psutil.virtual_memory()
        memory_available = memory.available / (1024**3)  # GB
        memory_usage = memory.percent

        # GPU 정보
        gpu_memory_free = 0.0
        gpu_utilization = 0.0

        if self.gpu_available:
            try:
                # 정확한 GPU 메모리 정보
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_free = gpu_memory_total - gpu_memory_allocated
                
                # GPU 사용률 계산 (더 정확한 방법)
                gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100

                # GPU 모니터링 히스토리 업데이트
                self._update_gpu_history(gpu_memory_allocated, gpu_memory_reserved, gpu_memory_total)

            except Exception as e:
                logger.warning(f"GPU 상태 확인 실패: {e}")
                gpu_memory_free = 0.0
                gpu_utilization = 100.0

        return SystemState(
            gpu_available=self.gpu_available,
            gpu_memory_free=gpu_memory_free,
            gpu_utilization=gpu_utilization,
            cpu_count=cpu_count,
            cpu_usage=cpu_usage,
            memory_available=memory_available,
            memory_usage=memory_usage,
        )

    def _update_gpu_history(self, allocated: float, reserved: float, total: float):
        """GPU 메모리 사용량 히스토리 업데이트"""
        timestamp = time.time()
        gpu_stats = {
            "timestamp": timestamp,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization": (allocated / total) * 100,
        }
        
        self.gpu_history.append(gpu_stats)
        
        # 히스토리 크기 제한
        if len(self.gpu_history) > self.max_history_size:
            self.gpu_history.pop(0)

    def get_gpu_trend(self) -> Dict[str, Any]:
        """GPU 사용량 트렌드 분석"""
        if not self.gpu_history or len(self.gpu_history) < 2:
            return {
                "trend": "stable",
                "avg_utilization": 0.0,
                "peak_utilization": 0.0,
                "memory_efficiency": 0.0,
            }
        
        utilizations = [stat["utilization"] for stat in self.gpu_history]
        allocated_gbs = [stat["allocated_gb"] for stat in self.gpu_history]
        
        # 트렌드 분석
        recent_avg = np.mean(utilizations[-10:]) if len(utilizations) >= 10 else np.mean(utilizations)
        overall_avg = np.mean(utilizations)
        
        if recent_avg > overall_avg * 1.2:
            trend = "increasing"
        elif recent_avg < overall_avg * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "avg_utilization": round(overall_avg, 2),
            "peak_utilization": round(max(utilizations), 2),
            "memory_efficiency": round((np.mean(allocated_gbs) / self.gpu_history[-1]["total_gb"]) * 100, 2),
            "samples": len(self.gpu_history),
        }

    def recommend_batch_size(self, current_batch_size: int, data_size_per_item: int) -> int:
        """GPU 메모리 상태를 기반으로 최적 배치 크기 추천"""
        if not self.gpu_available or not self.gpu_history:
            return current_batch_size
        
        current_state = self.get_current_state()
        gpu_trend = self.get_gpu_trend()
        
        # 메모리 여유 공간 (GB)
        memory_free_gb = current_state.gpu_memory_free
        
        # 데이터 크기 추정 (MB per item)
        data_size_mb = data_size_per_item / (1024**2)
        
        # 안전 여유 (20% 버퍼)
        safe_memory_gb = memory_free_gb * 0.8
        
        # 추천 배치 크기 계산
        if safe_memory_gb > 0.5:  # 500MB 이상 여유
            max_possible_batch = int((safe_memory_gb * 1024) / data_size_mb)
            
            # 트렌드 기반 조정
            if gpu_trend["trend"] == "increasing":
                # 메모리 사용량 증가 중 - 보수적 접근
                recommended_batch = min(max_possible_batch, current_batch_size)
            elif gpu_trend["trend"] == "decreasing":
                # 메모리 사용량 감소 중 - 적극적 접근
                recommended_batch = min(max_possible_batch, current_batch_size * 2)
            else:
                # 안정적 - 점진적 조정
                recommended_batch = min(max_possible_batch, int(current_batch_size * 1.2))
        else:
            # 메모리 부족 - 배치 크기 감소
            recommended_batch = max(1, current_batch_size // 2)
        
        return max(1, recommended_batch)

    def get_memory_pressure_level(self) -> str:
        """메모리 압박 수준 반환"""
        if not self.gpu_available:
            return "unknown"
        
        current_state = self.get_current_state()
        memory_usage_pct = current_state.gpu_utilization
        
        if memory_usage_pct > 90:
            return "critical"
        elif memory_usage_pct > 80:
            return "high"
        elif memory_usage_pct > 60:
            return "medium"
        else:
            return "low"


class OptimalComputeSelector:
    """최적 연산 전략 선택기"""

    def __init__(self):
        self.monitor = SystemMonitor()
        self.config = get_config("main").get_nested("utils.compute_strategy", {})

        # 임계값 설정
        self.gpu_memory_threshold = self.config.get("gpu_memory_threshold_gb", 1.0)
        self.gpu_utilization_threshold = self.config.get(
            "gpu_utilization_threshold", 80.0
        )
        self.multithread_data_threshold = self.config.get(
            "multithread_data_threshold_mb", 10
        )
        self.cpu_usage_threshold = self.config.get("cpu_usage_threshold", 80.0)
        self.min_workers_for_multithread = self.config.get(
            "min_workers_for_multithread", 2
        )

        logger.info(
            f"✅ 최적 연산 전략 선택기 초기화 (GPU 임계값: {self.gpu_memory_threshold}GB)"
        )

    def select_strategy(self, request: ComputeRequest) -> ComputeStrategy:
        """최적 연산 전략 선택"""
        state = self.monitor.get_current_state()
        data_size_mb = request.data_size / (1024**2)

        # 1. GPU 전략 검토
        if self._should_use_gpu(request, state, data_size_mb):
            logger.debug(f"GPU 전략 선택 (데이터: {data_size_mb:.1f}MB)")
            return ComputeStrategy.GPU

        # 2. 멀티쓰레드 CPU 전략 검토
        if self._should_use_multithread_cpu(request, state, data_size_mb):
            logger.debug(
                f"멀티쓰레드 CPU 전략 선택 (데이터: {data_size_mb:.1f}MB, 워커: {state.cpu_count})"
            )
            return ComputeStrategy.MULTITHREAD_CPU

        # 3. 단일 CPU 전략 (기본값)
        logger.debug(f"단일 CPU 전략 선택 (데이터: {data_size_mb:.1f}MB)")
        return ComputeStrategy.SINGLE_CPU

    def _should_use_gpu(
        self, request: ComputeRequest, state: SystemState, data_size_mb: float
    ) -> bool:
        """GPU 사용 여부 결정"""
        if not state.gpu_available:
            return False

        # 강제 GPU 요구사항
        if request.requires_gpu:
            return True

        # GPU 메모리 부족
        if state.gpu_memory_free < self.gpu_memory_threshold:
            logger.debug(
                f"GPU 메모리 부족: {state.gpu_memory_free:.1f}GB < {self.gpu_memory_threshold}GB"
            )
            return False

        # GPU 과부하
        if state.gpu_utilization > self.gpu_utilization_threshold:
            logger.debug(
                f"GPU 과부하: {state.gpu_utilization:.1f}% > {self.gpu_utilization_threshold}%"
            )
            return False

        # 텐서 연산은 GPU 우선
        if request.task_type == TaskType.TENSOR_COMPUTATION:
            return True

        # 메모리 집약적 작업은 GPU 메모리 여유 시에만
        if request.task_type == TaskType.MEMORY_INTENSIVE:
            return state.gpu_memory_free > self.gpu_memory_threshold * 2

        # 데이터 크기 기반 결정 (큰 데이터는 GPU가 유리)
        return data_size_mb > self.multithread_data_threshold * 5

    def _should_use_multithread_cpu(
        self, request: ComputeRequest, state: SystemState, data_size_mb: float
    ) -> bool:
        """멀티쓰레드 CPU 사용 여부 결정"""
        # CPU 코어 수 부족
        if state.cpu_count < self.min_workers_for_multithread:
            return False

        # CPU 과부하
        if state.cpu_usage > self.cpu_usage_threshold:
            return False

        # 데이터 크기 기반 결정
        if data_size_mb > self.multithread_data_threshold:
            return True

        # I/O 집약적 작업은 멀티쓰레드가 유리
        if request.task_type == TaskType.IO_INTENSIVE:
            return True

        return False


class AutoPerformanceMonitor:
    """자동 성능 모니터링 (Enhanced Performance Profiling)"""

    def __init__(self):
        self.performance_history = {}
        self.lock = threading.Lock()
        self.profiling_enabled = True
        self.max_history_size = 1000
        self.performance_cache_dir = "data/cache/performance_profiles"
        self.auto_save_interval = 100  # 100회 실행마다 자동 저장
        self.execution_count = 0

    @contextmanager
    def track(self, name: str, track_gpu: bool = True, track_memory_pool: bool = True):
        """성능 추적 컨텍스트 매니저"""
        start_time = time.time()
        gpu_memory_start = 0
        memory_pool_start = {}

        # GPU 메모리 추적
        if track_gpu and torch.cuda.is_available():
            try:
                gpu_memory_start = torch.cuda.memory_allocated()
            except Exception:
                pass

        # 메모리 풀 추적
        if track_memory_pool:
            try:
                memory_manager = get_unified_memory_manager()
                memory_pool_start = memory_manager.get_pool_stats()
            except Exception:
                pass

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # 성능 정보 수집
            perf_info = {
                "duration": duration,
                "timestamp": end_time,
            }

            # GPU 메모리 사용량
            if track_gpu and torch.cuda.is_available():
                try:
                    gpu_memory_end = torch.cuda.memory_allocated()
                    perf_info["gpu_memory_used"] = gpu_memory_end - gpu_memory_start
                except Exception:
                    pass

            # 메모리 풀 사용량
            if track_memory_pool:
                try:
                    memory_manager = get_unified_memory_manager()
                    memory_pool_end = memory_manager.get_pool_stats()
                    perf_info["memory_pool_delta"] = {
                        pool_name: memory_pool_end.get(pool_name, 0)
                        - memory_pool_start.get(pool_name, 0)
                        for pool_name in memory_pool_start.keys()
                    }
                except Exception:
                    pass

            # 성능 히스토리 저장
            with self.lock:
                if name not in self.performance_history:
                    self.performance_history[name] = []
                self.performance_history[name].append(perf_info)

                # 최대 히스토리 크기 유지
                if len(self.performance_history[name]) > self.max_history_size:
                    self.performance_history[name] = self.performance_history[name][
                        -self.max_history_size:
                    ]
                
                # 실행 횟수 증가 및 자동 저장
                self.execution_count += 1
                if self.execution_count % self.auto_save_interval == 0:
                    self._auto_save_performance_profile()

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        with self.lock:
            summary = {}
            for name, history in self.performance_history.items():
                if not history:
                    continue

                durations = [h["duration"] for h in history]
                summary[name] = {
                    "count": len(history),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_duration": sum(durations),
                    "last_run": history[-1]["timestamp"],
                }

                # GPU 메모리 통계
                gpu_memories = [
                    h.get("gpu_memory_used", 0)
                    for h in history
                    if "gpu_memory_used" in h
                ]
                if gpu_memories:
                    summary[name]["avg_gpu_memory"] = sum(gpu_memories) / len(
                        gpu_memories
                    )
                    summary[name]["max_gpu_memory"] = max(gpu_memories)

            return summary

    def clear_history(self):
        """성능 히스토리 초기화"""
        with self.lock:
            self.performance_history.clear()
            self.execution_count = 0

    def _auto_save_performance_profile(self):
        """자동 성능 프로파일 저장"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"auto_performance_profile_{timestamp}.json"
            self.save_performance_profile(filename)
        except Exception as e:
            logger.warning(f"자동 성능 프로파일 저장 실패: {e}")

    def save_performance_profile(self, filename: Optional[str] = None) -> str:
        """성능 프로파일 데이터 저장"""
        if not self.performance_history:
            logger.warning("저장할 성능 데이터가 없습니다.")
            return ""
        
        try:
            # 저장 디렉토리 생성
            cache_dir = Path(self.performance_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일명 생성
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"performance_profile_{timestamp}.json"
            
            filepath = cache_dir / filename
            
            with self.lock:
                # 성능 데이터 및 요약 정보 저장
                profile_data = {
                    "timestamp": time.time(),
                    "execution_count": self.execution_count,
                    "summary": self.get_performance_summary(),
                    "detailed_history": dict(self.performance_history),
                }
            
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2, default=str)
            
            logger.info(f"✅ 성능 프로파일 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"성능 프로파일 저장 실패: {e}")
            return ""

    def load_performance_profile(self, filename: str) -> bool:
        """성능 프로파일 데이터 로드"""
        try:
            filepath = Path(self.performance_cache_dir) / filename
            
            if not filepath.exists():
                logger.warning(f"성능 프로파일 파일이 존재하지 않습니다: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                profile_data = json.load(f)
            
            # 기존 히스토리와 병합
            with self.lock:
                if "detailed_history" in profile_data:
                    for name, history in profile_data["detailed_history"].items():
                        if name not in self.performance_history:
                            self.performance_history[name] = []
                        
                        self.performance_history[name].extend(history)
                        
                        # 히스토리 크기 제한
                        if len(self.performance_history[name]) > self.max_history_size:
                            self.performance_history[name] = self.performance_history[name][-self.max_history_size:]
                
                # 실행 횟수 업데이트
                if "execution_count" in profile_data:
                    self.execution_count = max(self.execution_count, profile_data["execution_count"])
            
            logger.info(f"✅ 성능 프로파일 로드 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"성능 프로파일 로드 실패: {e}")
            return False

    def get_performance_report(self) -> str:
        """성능 리포트 생성"""
        summary = self.get_performance_summary()
        
        if not summary:
            return "성능 데이터가 없습니다."
        
        report = f"""
=== 성능 모니터링 리포트 ===
총 실행 횟수: {self.execution_count}
추적 중인 함수: {len(summary)}개

=== 함수별 성능 통계 ===
"""
        
        for name, stats in summary.items():
            report += f"""
[{name}]
  실행 횟수: {stats['count']}
  평균 실행 시간: {stats['avg_duration']:.3f}초
  최소/최대 실행 시간: {stats['min_duration']:.3f}초 / {stats['max_duration']:.3f}초
  총 실행 시간: {stats['total_duration']:.3f}초
  마지막 실행: {time.ctime(stats['last_run'])}
"""
            
            if 'avg_gpu_memory' in stats:
                report += f"  평균 GPU 메모리 사용: {stats['avg_gpu_memory']:.2f} bytes\n"
                report += f"  최대 GPU 메모리 사용: {stats['max_gpu_memory']:.2f} bytes\n"
        
        return report

    def get_optimization_suggestions(self) -> List[str]:
        """최적화 제안 생성"""
        suggestions = []
        summary = self.get_performance_summary()
        
        if not summary:
            return suggestions
        
        # 성능 분석 기반 제안
        for name, stats in summary.items():
            # 긴 실행 시간 체크
            if stats['avg_duration'] > 1.0:  # 1초 이상
                suggestions.append(f"{name}: 평균 실행 시간이 {stats['avg_duration']:.3f}초로 긴 편입니다. 최적화 검토 필요")
            
            # GPU 메모리 사용량 체크
            if 'avg_gpu_memory' in stats and stats['avg_gpu_memory'] > 1024**3:  # 1GB 이상
                suggestions.append(f"{name}: GPU 메모리 사용량이 높습니다. 배치 크기 조정 고려")
            
            # 실행 횟수 대비 성능 체크
            if stats['count'] > 10:
                duration_variance = stats['max_duration'] - stats['min_duration']
                if duration_variance > stats['avg_duration']:
                    suggestions.append(f"{name}: 실행 시간 편차가 큽니다. 입력 데이터 크기 또는 시스템 부하 확인 필요")
        
        return suggestions


class ComputeExecutor:
    """연산 전략별 실행기 (compute_strategy.py에서 통합)"""

    def __init__(self):
        self.selector = OptimalComputeSelector()
        self.gpu_executor = None
        self.cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    def execute(
        self,
        func: Callable,
        data: Any,
        task_type: TaskType = TaskType.DATA_PROCESSING,
        **kwargs,
    ) -> Any:
        """최적 전략으로 함수 실행"""

        # 요청 정보 생성
        data_size = self._estimate_data_size(data)
        request = ComputeRequest(data_size=data_size, task_type=task_type, **kwargs)

        # 전략 선택
        strategy = self.selector.select_strategy(request)

        # 전략별 실행
        if strategy == ComputeStrategy.GPU:
            return self._execute_gpu(func, data, **kwargs)
        elif strategy == ComputeStrategy.MULTITHREAD_CPU:
            return self._execute_multithread_cpu(func, data, **kwargs)
        else:
            return self._execute_single_cpu(func, data, **kwargs)

    def _execute_gpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """GPU에서 실행"""
        device = torch.device("cuda")

        # 데이터를 GPU로 이동
        if hasattr(data, "to"):
            data = data.to(device)
        elif isinstance(data, (list, tuple)):
            data = [item.to(device) if hasattr(item, "to") else item for item in data]

        try:
            with torch.cuda.amp.autocast():
                result = func(data, **kwargs)

            # 결과를 CPU로 이동 (필요시)
            if hasattr(result, "cpu"):
                result = result.cpu()

            return result

        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU 메모리 부족, CPU로 폴백")
            torch.cuda.empty_cache()
            return self._execute_multithread_cpu(func, data, **kwargs)

    def _execute_multithread_cpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """멀티쓰레드 CPU에서 실행"""
        # 데이터를 CPU로 이동
        if hasattr(data, "cpu"):
            data = data.cpu()

        # 리스트/배치 데이터인 경우 병렬 처리
        if isinstance(data, (list, tuple)) and len(data) > 1:
            futures = []
            for item in data:
                future = self.cpu_executor.submit(func, item, **kwargs)
                futures.append(future)

            results = [future.result() for future in futures]
            return results
        else:
            return func(data, **kwargs)

    def _execute_single_cpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """단일 CPU에서 실행"""
        # 데이터를 CPU로 이동
        if hasattr(data, "cpu"):
            data = data.cpu()

        return func(data, **kwargs)

    def _estimate_data_size(self, data: Any) -> int:
        """데이터 크기 추정"""
        if hasattr(data, "nbytes"):
            return data.nbytes
        elif hasattr(data, "element_size") and hasattr(data, "nelement"):
            return data.element_size() * data.nelement()
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_data_size(item) for item in data)
        else:
            return len(str(data).encode("utf-8"))

    def shutdown(self):
        """리소스 정리"""
        self.cpu_executor.shutdown(wait=True)


class UnifiedPerformanceEngine:
    """통합 성능 최적화 엔진 (v2.0 - 자동 복구 통합)"""

    def __init__(self):
        """초기화"""
        # 설정 로드
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.memory_manager = get_unified_memory_manager()
        self.process_pool = get_enhanced_process_pool()
        self.cuda_optimizer = get_cuda_optimizer()

        # 핵심 구성 요소 (싱글톤)
        self.system_monitor = get_singleton_instance(SystemMonitor)
        self.compute_selector = get_singleton_instance(OptimalComputeSelector)
        self.performance_monitor = get_singleton_instance(AutoPerformanceMonitor)
        
        # 스레드 풀 및 스트림 관리자
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_cpu_workers", os.cpu_count() or 1)
        )
        self.stream_manager = CUDAStreamManager(
            pool_size=self.config.get("cuda_stream_pool_size", 4)
        )
        
        # 메모리 관리자
        self.memory_manager = get_unified_memory_manager()

        logger.info("✅ UnifiedPerformanceEngine 초기화 완료")

    @auto_recoverable() # 새로운 데코레이터 적용
    def execute(
        self,
        func: Callable,
        data: Any,
        task_type: TaskType = TaskType.DATA_PROCESSING,
        **kwargs,
    ) -> Any:
        """
        최적의 연산 전략을 사용하여 함수를 실행하고 성능을 추적합니다.
        GPU OOM 및 기타 오류 발생 시 자동으로 복구합니다.

        Args:
            func: 실행할 함수
            data: 함수에 전달할 데이터
            task_type: 작업 유형
            **kwargs: 추가 매개변수 (e.g., priority, requires_gpu)

        Returns:
            함수 실행 결과
        """
        # 강제 CPU 실행 플래그 확인
        force_cpu = kwargs.get("force_cpu", False)

        # 연산 요청 생성
        request = ComputeRequest(
            data_size=self._estimate_data_size(data),
            task_type=task_type,
            priority=kwargs.get("priority", 1),
            requires_gpu=kwargs.get("requires_gpu", False) and not force_cpu,
            max_workers=kwargs.get("max_workers"),
        )

        # 최적 실행 전략 선택
        strategy = self.compute_selector.select_strategy(request)
        if force_cpu and strategy == ComputeStrategy.GPU:
            strategy = ComputeStrategy.MULTITHREAD_CPU
            logger.info("CPU 실행 강제됨. GPU 전략을 멀티스레드 CPU로 변경.")

        # 성능 추적
        with self.performance_monitor.track(
            name=f"{func.__name__}_{strategy.value}",
            track_gpu=(strategy == ComputeStrategy.GPU),
        ) as tracker:
            
            # 전략에 따라 실행
            if strategy == ComputeStrategy.GPU:
                result = self._execute_gpu(func, data, tracker=tracker, **kwargs)
            elif strategy == ComputeStrategy.MULTITHREAD_CPU:
                result = self._execute_multithread_cpu(func, data, **kwargs)
            else:  # SINGLE_CPU
                result = self._execute_single_cpu(func, data, **kwargs)
            
            # 추적기에 결과 크기 기록
            if tracker:
                tracker.end_size_bytes = self._estimate_data_size(result)
            
            return result

    def _execute_gpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """GPU 실행"""
        device = torch.device("cuda")

        # CUDA 스트림 컨텍스트 사용
        if self.stream_manager:
            with self.stream_manager.stream_context() as stream:
                return self._execute_gpu_with_stream(
                    func, data, device, stream, **kwargs
                )
        else:
            return self._execute_gpu_with_stream(func, data, device, None, **kwargs)

    def _execute_gpu_with_stream(
        self,
        func: Callable,
        data: Any,
        device: torch.device,
        stream: Optional[torch.cuda.Stream],
        **kwargs,
    ) -> Any:
        """스트림을 사용한 GPU 실행 (최적화된 메모리 전송 및 동기화)"""
        original_stream = None
        gpu_data = None
        
        try:
            # 현재 스트림 저장
            if stream is not None:
                original_stream = torch.cuda.current_stream()
                torch.cuda.set_stream(stream)
            
            # 메모리 전송 최적화
            if isinstance(data, torch.Tensor):
                # 이미 올바른 디바이스에 있는지 확인
                if data.device == device:
                    gpu_data = data
                else:
                    # 메모리 압박 확인 후 전송
                    if self._check_gpu_memory_available(data.numel() * data.element_size()):
                        gpu_data = data.to(device, non_blocking=True)
                    else:
                        # 메모리 부족 시 CPU 폴백
                        logger.warning("GPU 메모리 부족 - CPU 폴백")
                        return self._execute_single_cpu(func, data, **kwargs)
                        
            elif isinstance(data, np.ndarray):
                # NumPy 배열의 효율적인 GPU 전송
                data_size = data.nbytes
                if self._check_gpu_memory_available(data_size):
                    gpu_data = torch.from_numpy(data).to(device, non_blocking=True)
                else:
                    logger.warning("GPU 메모리 부족 - CPU 폴백")
                    return self._execute_single_cpu(func, data, **kwargs)
                    
            elif isinstance(data, list):
                # 리스트의 경우 텐서로 변환 시도
                try:
                    # 메모리 사용량 추정
                    estimated_size = len(data) * 8  # 대략적인 추정
                    if self._check_gpu_memory_available(estimated_size):
                        gpu_data = torch.tensor(data, device=device)
                    else:
                        logger.warning("GPU 메모리 부족 - CPU 폴백")
                        return self._execute_single_cpu(func, data, **kwargs)
                except Exception as e:
                    logger.debug(f"텐서 변환 실패: {e}")
                    # 변환 실패 시 CPU에서 실행
                    return self._execute_single_cpu(func, data, **kwargs)
            else:
                # 기타 데이터 타입은 CPU에서 실행
                return self._execute_single_cpu(func, data, **kwargs)

            # 스트림 동기화 (데이터 전송 완료 확인)
            if stream is not None and gpu_data is not None:
                stream.synchronize()

            # 함수 실행
            result = func(gpu_data, **kwargs)

            # 결과 처리 및 메모리 전송 최적화
            if isinstance(result, torch.Tensor) and result.is_cuda:
                # 결과 크기 확인 후 CPU로 이동
                if kwargs.get("keep_on_gpu", False):
                    # GPU 결과 유지 옵션
                    return result
                else:
                    # 스트림 동기화 후 CPU 전송
                    if stream is not None:
                        stream.synchronize()
                    result = result.cpu()

            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU 메모리 부족: {e}")
            # 메모리 정리 시도
            self._emergency_gpu_cleanup()
            # CPU 폴백
            return self._execute_single_cpu(func, data, **kwargs)
            
        except Exception as e:
            logger.error(f"GPU 실행 중 오류 발생: {e}")
            # CPU 폴백
            return self._execute_single_cpu(func, data, **kwargs)
            
        finally:
            # 스트림 복원
            if original_stream is not None:
                torch.cuda.set_stream(original_stream)
            
            # 메모리 정리 (필요한 경우)
            if gpu_data is not None and gpu_data is not data:
                del gpu_data
                
    def _check_gpu_memory_available(self, required_bytes: int) -> bool:
        """GPU 메모리 사용 가능 여부 확인"""
        try:
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            # 10% 여유 공간 확보
            return available_memory > required_bytes * 1.1
        except:
            return False
            
    def _emergency_gpu_cleanup(self):
        """긴급 GPU 메모리 정리"""
        try:
            torch.cuda.empty_cache()
            # 통합 메모리 관리자를 통한 정리
            if hasattr(self, 'memory_manager'):
                self.memory_manager.optimize_memory_usage()
        except Exception as e:
            logger.error(f"긴급 메모리 정리 실패: {e}")

    def _execute_multithread_cpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """멀티쓰레드 CPU 실행"""
        if not isinstance(data, list):
            return self._execute_single_cpu(func, data, **kwargs)

        # 데이터 청크 분할
        chunk_size = max(1, len(data) // (os.cpu_count() or 1))
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 병렬 실행
        def func_wrapper(chunk):
            return func(chunk, **kwargs)

        futures = [self.cpu_executor.submit(func_wrapper, chunk) for chunk in chunks]
        results = [future.result() for future in futures]

        # 결과 병합
        if all(isinstance(r, list) for r in results):
            return [item for sublist in results for item in sublist]
        else:
            return results

    def _execute_single_cpu(self, func: Callable, data: Any, **kwargs) -> Any:
        """단일 CPU 실행"""
        return func(data, **kwargs)

    def _estimate_data_size(self, data: Any) -> int:
        """데이터 크기 추정"""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, list):
            return len(data) * 8  # 대략적인 추정
        elif isinstance(data, dict):
            return len(str(data).encode())
        else:
            return len(str(data).encode())

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = {
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "system_state": self.compute_selector.monitor.get_current_state().__dict__,
        }

        if self.stream_manager:
            stats["cuda_streams"] = self.stream_manager.get_stats()

        return stats

    def shutdown(self):
        """엔진 종료"""
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
        if self.stream_manager:
            self.stream_manager.synchronize_all()
        logger.info("통합 성능 최적화 엔진 종료 완료")


# 싱글톤 인스턴스
_unified_performance_engine = None
_engine_lock = threading.Lock()


def get_unified_performance_engine() -> UnifiedPerformanceEngine:
    """통합 성능 최적화 엔진 싱글톤 인스턴스 반환"""
    global _unified_performance_engine
    if _unified_performance_engine is None:
        with _engine_lock:
            if _unified_performance_engine is None:
                _unified_performance_engine = UnifiedPerformanceEngine()
    return _unified_performance_engine


def smart_execute(
    func: Callable, data: Any, task_type: TaskType = TaskType.DATA_PROCESSING, **kwargs
) -> Any:
    """
    스마트 실행 함수 - 최적 전략을 자동 선택하여 함수 실행

    Args:
        func: 실행할 함수
        data: 입력 데이터
        task_type: 작업 유형
        **kwargs: 추가 인수

    Returns:
        함수 실행 결과
    """
    engine = get_unified_performance_engine()
    return engine.execute(func, data, task_type, **kwargs)


# 하위 호환성을 위한 별칭
def get_smart_computation_engine() -> UnifiedPerformanceEngine:
    """하위 호환성을 위한 별칭"""
    return get_unified_performance_engine()


def get_auto_performance_monitor() -> AutoPerformanceMonitor:
    """성능 모니터 반환"""
    engine = get_unified_performance_engine()
    return engine.performance_monitor


# 기존 compute_strategy.py 호환성 함수들
def get_compute_executor() -> ComputeExecutor:
    """ComputeExecutor 싱글톤 인스턴스 반환"""
    return get_singleton_instance(ComputeExecutor)


def get_optimal_compute_selector() -> OptimalComputeSelector:
    """OptimalComputeSelector 싱글톤 인스턴스 반환"""
    return get_singleton_instance(OptimalComputeSelector)


def get_cuda_stream_manager() -> Optional[CUDAStreamManager]:
    """CUDA 스트림 관리자 반환"""
    engine = get_unified_performance_engine()
    return engine.stream_manager
