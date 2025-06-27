"""
배치 크기 동적 조절 (Dynamic Batch Size)

이 모듈은 모델 학습 및 추론 과정에서 메모리 사용량과 성능에 따라
배치 크기를 동적으로 조절하는 기능을 제공합니다.
"""

import numpy as np
import torch
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
import time
import os
from dataclasses import dataclass, field
from threading import Lock
from pathlib import Path
import gc

from .error_handler_refactored import get_logger

# 로거 설정
logger = get_logger(__name__)

try:
    # 상대 경로로 임포트 수정
    from .cuda_optimizers import BaseCudaOptimizer, CudaConfig

    CUDA_OPTIMIZER_AVAILABLE = True
except ImportError:
    CUDA_OPTIMIZER_AVAILABLE = False


@dataclass
class BatchSizeConfig:
    """배치 크기 설정 클래스"""

    # 배치 크기 설정
    initial_batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: Optional[int] = 256
    optimal_batch_size: Optional[int] = None

    # 조정 계수
    reduction_factor: float = 0.5
    growth_factor: float = 1.25

    # 임계값 설정
    memory_threshold: float = 0.85
    stability_steps: int = 5

    # GPU 관련 설정
    use_gpu: bool = True
    gpu_memory_reserve: float = 0.1  # 10%의 GPU 메모리 예약
    use_mixed_precision: bool = True
    adaptive_mixed_precision: bool = True

    # CPU 관련 설정
    cpu_threads: int = max(4, min(os.cpu_count() or 4, 16))
    use_thread_pinning: bool = True

    # 자동 튜닝 설정
    enable_auto_tuning: bool = True
    tuning_metric: str = "throughput"  # "throughput", "latency", "memory"
    tuning_iterations: int = 5
    tuning_search_space: List[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128, 256]
    )

    # 성능 추적 설정
    track_performance: bool = True
    performance_window: int = 10

    def __post_init__(self):
        """설정 초기화 후처리"""
        # GPU 사용 가능 여부 확인
        self.gpu_available = torch.cuda.is_available()

        # GPU 사용 불가능하면 CPU 설정으로 전환
        if not self.gpu_available:
            self.use_gpu = False
            self.use_mixed_precision = False

        # 최적 배치 크기가 없으면 초기 배치 크기 사용
        if self.optimal_batch_size is None:
            self.optimal_batch_size = self.initial_batch_size


class DynamicBatchSize:
    """
    배치 크기 동적 조절 클래스

    메모리 사용량, OOM 발생 여부, 성능 메트릭 등을 모니터링하여
    최적의 배치 크기를 동적으로 조절합니다.
    """

    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        reduction_factor: float = 0.5,
        growth_factor: float = 1.25,
        memory_threshold: float = 0.85,
        stability_steps: int = 5,
        callback: Optional[Callable] = None,
        config: Optional[BatchSizeConfig] = None,
    ):
        """
        배치 크기 동적 관리자 초기화

        Args:
            initial_batch_size: 초기 배치 크기
            min_batch_size: 최소 배치 크기 (기본값: 1)
            max_batch_size: 최대 배치 크기 (기본값: None, 제한 없음)
            reduction_factor: 배치 크기 감소 비율 (기본값: 0.5)
            growth_factor: 배치 크기 증가 비율 (기본값: 1.25)
            memory_threshold: 메모리 사용률 임계값, 이 값을 초과하면 배치 크기 감소 (기본값: 0.85)
            stability_steps: 배치 크기 조정 전 안정화를 위한 단계 수 (기본값: 5)
            callback: 배치 크기 변경 시 호출할 콜백 함수 (기본값: None)
            config: 배치 크기 설정 (기본값: None)
        """
        # 설정 초기화
        self.config = config or BatchSizeConfig(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            reduction_factor=reduction_factor,
            growth_factor=growth_factor,
            memory_threshold=memory_threshold,
            stability_steps=stability_steps,
        )

        self.batch_size = self.config.initial_batch_size
        self.callback = callback
        self._lock = Lock()

        # 배치 크기 조정 추적용 변수
        self.steps_since_last_adjustment = 0
        self.adjustment_history = []
        self.performance_history = []
        self.last_adjustment_time = time.time()
        self.last_memory_check = time.time()

        # GPU 설정
        self.gpu_available = torch.cuda.is_available()
        self.use_gpu = self.config.use_gpu and self.gpu_available
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # CUDA 최적화 모듈 초기화 (사용 가능한 경우)
        self.cuda_optimizer = None
        if self.use_gpu and CUDA_OPTIMIZER_AVAILABLE:
            try:
                # CudaConfig 초기화 시 None 안전하게 처리
                max_batch_size = (
                    self.config.max_batch_size
                    if self.config.max_batch_size is not None
                    else 256
                )
                optimal_batch_size = (
                    self.config.optimal_batch_size
                    if self.config.optimal_batch_size is not None
                    else self.batch_size
                )

                cuda_config = CudaConfig(
                    batch_size=self.batch_size,
                    min_batch_size=self.config.min_batch_size,
                    max_batch_size=max_batch_size,
                    optimal_batch_size=optimal_batch_size,
                    use_amp=self.config.use_mixed_precision,
                )
                self.cuda_optimizer = BaseCudaOptimizer(cuda_config)
                logger.info("CUDA 최적화 모듈 초기화 성공")
            except Exception as e:
                logger.warning(f"CUDA 최적화 모듈 초기화 실패: {str(e)}")

        # 초기 메모리 상태 확인
        self.initial_memory_usage = self._get_memory_usage()

        # 콜백 호출 (초기화 완료)
        if self.callback:
            try:
                self.callback(self.batch_size, "initialize")
            except Exception as e:
                logger.error(f"콜백 함수 호출 실패: {str(e)}")

        logger.info(
            f"동적 배치 크기 관리자가 초기화되었습니다. 초기 배치 크기: {self.batch_size}, "
            f"GPU 사용: {self.use_gpu}, 메모리 사용률: {self.initial_memory_usage:.2f}"
        )

        # 자동 튜닝 수행 (설정된 경우)
        if self.config.enable_auto_tuning:
            self._schedule_auto_tuning()

    def _get_memory_usage(self) -> float:
        """
        현재 메모리 사용률 확인

        Returns:
            메모리 사용률 (0~1)
        """
        if self.use_gpu:
            try:
                # GPU 메모리 사용량 확인
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()

                if reserved > 0:
                    return allocated / reserved

                # 예약된 메모리가 없으면 총 메모리 대비 사용량 확인
                total = torch.cuda.get_device_properties(0).total_memory
                return allocated / total
            except Exception as e:
                logger.warning(f"GPU 메모리 사용량 확인 실패: {str(e)}")
                return 0.0
        else:
            # CPU 메모리 사용량 확인
            try:
                import psutil

                return psutil.virtual_memory().percent / 100.0
            except:
                return 0.5  # 기본값

    def get_batch_size(self) -> int:
        """
        현재 배치 크기 반환

        Returns:
            현재 배치 크기
        """
        with self._lock:
            return self.batch_size

    def update_on_memory_status(
        self, memory_usage: Optional[float] = None, oom_detected: bool = False
    ) -> int:
        """
        메모리 사용률에 따라 배치 크기 조정

        Args:
            memory_usage: 현재 메모리 사용률 (0~1), None이면 자동 측정
            oom_detected: OOM(Out of Memory) 발생 여부

        Returns:
            조정된 배치 크기
        """
        with self._lock:
            # 메모리 사용률 자동 측정 (지정되지 않은 경우)
            if memory_usage is None:
                memory_usage = self._get_memory_usage()

            # 메모리 임계값 초과 또는 OOM 발생 시 배치 크기 감소
            if oom_detected or memory_usage > self.config.memory_threshold:
                # OOM 발생 시 더 급격히 감소
                reduction_factor = 0.5 if not oom_detected else 0.25
                self._reduce_batch_size(
                    oom=oom_detected, reduction_factor=reduction_factor
                )

                # 조정 이력 추가
                self.adjustment_history.append(
                    {
                        "time": time.time(),
                        "action": "reduce",
                        "batch_size": self.batch_size,
                        "memory_usage": memory_usage,
                        "oom": oom_detected,
                    }
                )

                # 콜백 호출
                if self.callback:
                    try:
                        self.callback(
                            self.batch_size,
                            "reduce_memory" if not oom_detected else "reduce_oom",
                        )
                    except Exception as e:
                        logger.error(f"콜백 함수 호출 실패: {str(e)}")

                logger.info(
                    f"메모리 상태에 따라 배치 크기 감소: {self.batch_size} "
                    f"(메모리: {memory_usage:.2f}, OOM: {oom_detected})"
                )
            else:
                # 안정적인 메모리 사용률이면 배치 크기 증가 가능성 검토
                current_time = time.time()
                time_since_last_check = current_time - self.last_memory_check
                self.last_memory_check = current_time

                # 최소 10초마다 메모리 상태 기반 증가 검토
                if (
                    time_since_last_check >= 10.0
                    and memory_usage < self.config.memory_threshold * 0.7
                    and self.steps_since_last_adjustment >= self.config.stability_steps
                ):

                    # 메모리 여유가 있으면 배치 크기 증가
                    self._increase_batch_size()

                    # 조정 이력 추가
                    self.adjustment_history.append(
                        {
                            "time": time.time(),
                            "action": "increase",
                            "batch_size": self.batch_size,
                            "memory_usage": memory_usage,
                        }
                    )

                    # 콜백 호출
                    if self.callback:
                        try:
                            self.callback(self.batch_size, "increase_memory")
                        except Exception as e:
                            logger.error(f"콜백 함수 호출 실패: {str(e)}")

                    logger.info(
                        f"메모리 여유에 따라 배치 크기 증가: {self.batch_size} "
                        f"(메모리: {memory_usage:.2f})"
                    )

            # 내부 카운터 업데이트
            self.steps_since_last_adjustment += 1

            # 이력 크기 제한
            if len(self.adjustment_history) > 100:
                self.adjustment_history = self.adjustment_history[-100:]

            return self.batch_size

    def update_on_performance(
        self, samples_per_second: float, batch_time: float
    ) -> int:
        """
        성능 메트릭에 따라 배치 크기 조정

        Args:
            samples_per_second: 샘플 처리 속도 (초당 샘플 수)
            batch_time: 배치 처리 시간 (초)

        Returns:
            조정된 배치 크기
        """
        with self._lock:
            # 성능 메트릭 기록
            self.performance_history.append(
                {
                    "timestamp": time.time(),
                    "samples_per_second": samples_per_second,
                    "batch_time": batch_time,
                    "batch_size": self.batch_size,
                    "memory_usage": self._get_memory_usage(),
                }
            )

            # 성능 기록 제한 (최근 N개만 유지)
            if len(self.performance_history) > self.config.performance_window:
                self.performance_history = self.performance_history[
                    -self.config.performance_window :
                ]

            # 3개 이상의 성능 메트릭 기록이 있는 경우 배치 크기 조정 고려
            if len(self.performance_history) >= 3:
                self.steps_since_last_adjustment += 1

                # 성능 기반 조정 주기 확인
                current_time = time.time()
                adjustment_elapsed = current_time - self.last_adjustment_time

                if (
                    self.steps_since_last_adjustment >= self.config.stability_steps
                    and adjustment_elapsed >= 5.0
                ):  # 최소 5초 간격으로 조정
                    self._adjust_based_on_performance()
                    self.steps_since_last_adjustment = 0
                    self.last_adjustment_time = current_time

            return self.batch_size

    def handle_exception(self, exception: Exception) -> int:
        """
        예외 발생 시 배치 크기 조정

        Args:
            exception: 발생한 예외

        Returns:
            조정된 배치 크기
        """
        with self._lock:
            # CUDA OOM 예외 발생 시 배치 크기 감소
            if isinstance(
                exception, torch.cuda.OutOfMemoryError
            ) or "CUDA out of memory" in str(exception):
                logger.warning("CUDA OOM 발생. 배치 크기 감소")
                return self.update_on_memory_status(1.0, oom_detected=True)

            # 기타 예외 발생 시 배치 크기 감소
            logger.warning(f"예외 발생: {str(exception)[:100]}... 배치 크기 감소")
            self._reduce_batch_size(reduction_factor=0.75)  # 임시로 0.75로 설정

            return self.batch_size

    def _reduce_batch_size(
        self, oom: bool = False, reduction_factor: Optional[float] = None
    ) -> None:
        """
        배치 크기 감소

        Args:
            oom: OOM 발생 여부
            reduction_factor: 감소 비율 (기본값: None, 즉 설정된 감소 비율 사용)
        """
        if reduction_factor is None:
            reduction_factor = self.config.reduction_factor

        # OOM 발생 시 더 큰 감소 비율 적용
        if oom and reduction_factor > 0.3:
            reduction_factor = min(reduction_factor, 0.3)

        old_batch_size = self.batch_size
        self.batch_size = max(
            int(self.batch_size * reduction_factor), self.config.min_batch_size
        )

        # CUDA 최적화 모듈 동기화 (있는 경우)
        if self.cuda_optimizer is not None:
            try:
                self.cuda_optimizer.config.batch_size = self.batch_size
            except Exception as e:
                logger.warning(f"CUDA 최적화 모듈 배치 크기 업데이트 실패: {str(e)}")

        # 배치 크기 감소 기록
        self.adjustment_history.append(
            {
                "timestamp": time.time(),
                "action": "reduce",
                "old_size": old_batch_size,
                "new_size": self.batch_size,
                "reason": "oom" if oom else "high_memory_usage",
            }
        )

        logger.info(
            f"배치 크기 감소: {old_batch_size} -> {self.batch_size}"
            + (" (OOM 발생)" if oom else "")
        )

        # 콜백 함수 호출
        if self.callback:
            try:
                self.callback(self.batch_size, "reduce")
            except Exception as e:
                logger.error(f"콜백 함수 호출 실패: {str(e)}")

    def _increase_batch_size(self) -> None:
        """배치 크기 증가"""
        # 최대 배치 크기 확인
        if (
            self.config.max_batch_size is not None
            and self.batch_size >= self.config.max_batch_size
        ):
            return

        # 증가율에 따라 배치 크기 증가
        old_batch_size = self.batch_size
        new_batch_size = int(self.batch_size * self.config.growth_factor)

        # 최대값 제한
        if self.config.max_batch_size is not None:
            new_batch_size = min(new_batch_size, self.config.max_batch_size)

        # 배치 크기 변경이 없으면 1 증가 (최소 변화)
        if new_batch_size == old_batch_size:
            new_batch_size = old_batch_size + 1

        # 최적 배치 크기 방향으로 조정
        if (
            self.config.optimal_batch_size is not None
            and old_batch_size < self.config.optimal_batch_size <= new_batch_size
        ):
            new_batch_size = self.config.optimal_batch_size

        # 배치 크기 업데이트
        self.batch_size = new_batch_size

        # CUDA 최적화 모듈 동기화 (있는 경우)
        if self.cuda_optimizer is not None:
            try:
                self.cuda_optimizer.config.batch_size = self.batch_size
            except Exception as e:
                logger.warning(f"CUDA 최적화 모듈 배치 크기 업데이트 실패: {str(e)}")

        # 배치 크기 증가 기록
        self.adjustment_history.append(
            {
                "timestamp": time.time(),
                "action": "increase",
                "old_size": old_batch_size,
                "new_size": new_batch_size,
                "reason": "low_memory_usage",
            }
        )

        logger.info(f"배치 크기 증가: {old_batch_size} -> {new_batch_size}")

        # 콜백 함수 호출
        if self.callback:
            try:
                self.callback(self.batch_size, "increase")
            except Exception as e:
                logger.error(f"콜백 함수 호출 실패: {str(e)}")

    def _adjust_based_on_performance(self) -> None:
        """성능 메트릭에 따라 배치 크기 조정"""
        if len(self.performance_history) < 2:
            return

        # 최근 성능 데이터 추출
        current = self.performance_history[-1]
        previous = self.performance_history[-2]

        # 성능 변화 계산
        throughput_change = (
            current["samples_per_second"] - previous["samples_per_second"]
        ) / previous["samples_per_second"]

        # 성능 판단을 위한 임계값
        improvement_threshold = 0.05  # 5% 향상
        degradation_threshold = -0.1  # 10% 저하

        # 현재 메모리 사용량 확인
        memory_usage = self._get_memory_usage()
        memory_critical = memory_usage > 0.9  # 90% 이상 사용 중인지 확인

        # 성능이 저하되고 메모리 사용량이 높으면 배치 크기 감소
        if throughput_change < degradation_threshold and memory_usage > 0.7:
            logger.info(
                f"성능 저하 감지 ({throughput_change:.2%}), 메모리 사용량: {memory_usage:.2f}. 배치 크기 감소."
            )
            self._reduce_batch_size(reduction_factor=0.8)
            return

        # 심각한 메모리 사용량이면 배치 크기 감소
        if memory_critical:
            logger.info(f"높은 메모리 사용량 감지: {memory_usage:.2f}. 배치 크기 감소.")
            self._reduce_batch_size(reduction_factor=0.9)
            return

        # 성능이 향상되면 배치 크기 증가
        if throughput_change > improvement_threshold and memory_usage < 0.8:
            logger.info(
                f"성능 향상 감지 ({throughput_change:.2%}), 메모리 사용량: {memory_usage:.2f}. 배치 크기 증가."
            )
            self._increase_batch_size()

    def _schedule_auto_tuning(self) -> None:
        """자동 튜닝 일정 잡기"""
        # 별도의 스레드에서 자동 튜닝 수행 (백그라운드)
        import threading

        def delayed_auto_tuning():
            # 5초 대기 후 자동 튜닝 시작
            time.sleep(5)
            try:
                self._auto_tune_batch_size()
            except Exception as e:
                logger.error(f"자동 튜닝 중 오류 발생: {str(e)}")

        tuning_thread = threading.Thread(target=delayed_auto_tuning)
        tuning_thread.daemon = True
        tuning_thread.start()

    def _auto_tune_batch_size(self) -> None:
        """배치 크기 자동 튜닝"""
        if not self.config.enable_auto_tuning:
            return

        # cuda_optimizer가 초기화되었는지 확인
        if not hasattr(self, "cuda_optimizer") or self.cuda_optimizer is None:
            logger.warning(
                "CUDA 최적화기가 초기화되지 않았습니다. 자동 튜닝을 건너뜁니다."
            )
            return

        logger.info("배치 크기 자동 튜닝 시작...")
        try:
            # 현재 메모리 상태를 확인하여 최적 배치 크기 추정
            memory_usage = self._get_memory_usage()

            # 메모리 여유도에 따라 배치 크기 추정
            if memory_usage < 0.3:  # 메모리 사용량이 30% 미만
                batch_scale_factor = 2.0
            elif memory_usage < 0.5:  # 메모리 사용량이 50% 미만
                batch_scale_factor = 1.5
            elif memory_usage < 0.7:  # 메모리 사용량이 70% 미만
                batch_scale_factor = 1.0
            else:  # 메모리 사용량이 70% 이상
                batch_scale_factor = 0.8

            # 추정 배치 크기 계산 (현재 배치 크기 기준)
            estimated_batch_size = max(
                self.config.min_batch_size,
                min(
                    int(self.batch_size * batch_scale_factor),
                    self.config.max_batch_size or 256,
                ),
            )

            # 최적 배치 크기 적용
            if estimated_batch_size != self.batch_size:
                logger.info(f"자동 튜닝 결과: 최적 배치 크기 = {estimated_batch_size}")
                old_batch_size = self.batch_size
                self.batch_size = estimated_batch_size
                self.config.optimal_batch_size = estimated_batch_size

                # CUDA 최적화 모듈 동기화
                if hasattr(self.cuda_optimizer, "config"):
                    self.cuda_optimizer.config.batch_size = self.batch_size

                # 배치 크기 변경 기록
                self.adjustment_history.append(
                    {
                        "timestamp": time.time(),
                        "action": "auto_tune",
                        "old_size": old_batch_size,
                        "new_size": self.batch_size,
                        "reason": "auto_tuning",
                    }
                )

                # 콜백 함수 호출
                if self.callback:
                    try:
                        self.callback(self.batch_size, "auto_tune")
                    except Exception as e:
                        logger.error(f"콜백 함수 호출 실패: {str(e)}")
            else:
                logger.info(f"현재 배치 크기 {self.batch_size}가 이미 최적입니다.")

        except Exception as e:
            logger.error(f"배치 크기 자동 튜닝 실패: {str(e)}")

    def get_adjustment_history(self) -> list:
        """
        배치 크기 조정 이력 반환

        Returns:
            배치 크기 조정 이력
        """
        return self.adjustment_history.copy()

    def get_performance_history(self) -> list:
        """
        성능 이력 반환

        Returns:
            성능 이력
        """
        return self.performance_history.copy()

    def reset(self, batch_size: Optional[int] = None) -> None:
        """
        상태 초기화

        Args:
            batch_size: 새 배치 크기 (기본값: None, 즉 초기 배치 크기 사용)
        """
        with self._lock:
            # 배치 크기 초기화
            self.batch_size = batch_size or self.config.initial_batch_size

            # 상태 초기화
            self.steps_since_last_adjustment = 0
            self.adjustment_history = []
            self.performance_history = []
            self.last_adjustment_time = time.time()

            # CUDA 최적화 모듈 동기화 (있는 경우)
            if self.cuda_optimizer is not None:
                try:
                    self.cuda_optimizer.config.batch_size = self.batch_size
                except Exception as e:
                    logger.warning(
                        f"CUDA 최적화 모듈 배치 크기 업데이트 실패: {str(e)}"
                    )

            logger.info(f"동적 배치 크기 관리자 초기화. 배치 크기: {self.batch_size}")

            # 콜백 함수 호출
            if self.callback:
                try:
                    self.callback(self.batch_size, "reset")
                except Exception as e:
                    logger.error(f"콜백 함수 호출 실패: {str(e)}")

    def cleanup(self):
        """리소스 정리"""
        if self.cuda_optimizer:
            try:
                self.cuda_optimizer.cleanup()
            except Exception as e:
                logger.warning(f"CUDA 최적화 모듈 정리 실패: {str(e)}")

    def __del__(self):
        """소멸자"""
        self.cleanup()


def get_available_memory(device: Union[str, torch.device]) -> Tuple[int, int]:
    """
    사용 가능한 메모리 용량 확인

    Args:
        device: 메모리를 확인할 장치 (CPU 또는 CUDA)

    Returns:
        (전체 메모리, 사용 가능한 메모리) 튜플 (단위: 바이트)
    """
    try:
        if isinstance(device, str):
            device = torch.device(device)

        # None 체크와 타입 체크를 안전하게 수행
        if hasattr(device, "type") and device.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA가 요청되었지만 사용할 수 없습니다. CPU로 대체합니다."
                )
                return get_available_memory("cpu")

            # GPU 메모리 정보 얻기
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory

            return total_memory, free_memory
        else:
            # CPU 메모리의 경우 (시스템에 따라 다를 수 있음)
            import psutil

            total_memory = psutil.virtual_memory().total
            available_memory = psutil.virtual_memory().available

            return total_memory, available_memory
    except Exception as e:
        logger.error(f"메모리 정보 획득 중 오류 발생: {str(e)}")
        # 기본값 반환 (단위: 바이트, 약 4GB 사용 가능)
        return 8 * (1024**3), 4 * (1024**3)


def estimate_batch_size(
    sample_input_size: int,
    model_size: Optional[int] = None,
    memory_margin: float = 0.2,
    device: Optional[Union[str, torch.device]] = None,
) -> int:
    """
    입력 크기와 모델 크기를 기반으로 적절한 배치 크기 추정

    Args:
        sample_input_size: 샘플 입력의 크기 (바이트)
        model_size: 모델 메모리 사용량 (바이트, 기본값: None)
        memory_margin: 안전 마진 (사용 가능한 메모리의 비율, 기본값: 0.2)
        device: 메모리를 확인할 장치 (기본값: None, 자동 감지)

    Returns:
        추정된 배치 크기
    """
    # 장치 설정
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # 사용 가능한 메모리 확인
    _, available_memory = get_available_memory(device)

    # 사용 가능한 메모리에서 안전 마진 확보
    usable_memory = available_memory * (1 - memory_margin)

    # 모델 크기를 고려
    if model_size is not None:
        usable_memory = max(0, usable_memory - model_size)

    # 배치 크기 계산
    if sample_input_size <= 0:
        logger.warning("샘플 입력 크기가 0 이하입니다. 기본 배치 크기 32를 사용합니다.")
        return 32

    estimated_batch_size = int(usable_memory / sample_input_size)

    # 최소 배치 크기 보장
    if estimated_batch_size < 1:
        logger.warning("메모리 부족으로 배치 크기를 1로 조정합니다.")
        return 1

    return estimated_batch_size


def get_safe_batch_size(
    initial_batch_size: int = 32,
    min_batch_size: int = 1,
    max_batch_size: int = 256,
    device: Optional[Union[str, torch.device]] = None,
    tensor_type: torch.dtype = torch.float32,
    try_gpu_first: bool = True,
) -> int:
    """
    메모리 상황을 고려하여 안전한 배치 크기 계산

    Args:
        initial_batch_size: 초기 배치 크기 (기본값: 32)
        min_batch_size: 최소 배치 크기 (기본값: 1)
        max_batch_size: 최대 배치 크기 (기본값: 256)
        device: 메모리를 확인할 장치 (기본값: None, 자동 감지)
        tensor_type: 텐서 데이터 타입 (기본값: torch.float32)
        try_gpu_first: GPU 사용을 우선 시도 (기본값: True)

    Returns:
        안전한 배치 크기
    """
    # 장치 설정
    if device is None:
        if try_gpu_first and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # 메모리 부족 시 배치 크기 자동 조정
    batch_size = initial_batch_size

    # 배치 크기 범위 보장
    batch_size = max(min_batch_size, min(batch_size, max_batch_size))

    # 메모리 테스트를 위한 임시 텐서 생성 시도
    for attempt in range(3):  # 3번 시도
        try:
            # 가비지 컬렉션 강제 실행
            gc.collect()
            if isinstance(device, torch.device) and device.type == "cuda":
                torch.cuda.empty_cache()

            # 임시 텐서를 생성하여 메모리 사용량 테스트
            tmp = torch.zeros(
                (batch_size, 1000, 1000), dtype=tensor_type, device=device
            )
            del tmp

            # 성공하면 이 배치 크기는 안전함
            logger.info(
                f"배치 크기 {batch_size}가 {device} 장치에서 안전하게 사용 가능합니다."
            )
            return batch_size

        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                # 배치 크기 감소
                prev_batch_size = batch_size
                batch_size = max(min_batch_size, batch_size // 2)

                logger.warning(
                    f"메모리 부족으로 배치 크기를 {prev_batch_size}에서 {batch_size}로 감소시킵니다."
                )

                # 이미 최소 크기면 더 이상 시도하지 않음
                if batch_size == min_batch_size:
                    break
            else:
                # 다른 오류는 재시도하지 않고 기본값 반환
                logger.error(f"메모리 테스트 중 오류 발생: {str(e)}")
                return min_batch_size

    # 최소 배치 크기로 설정
    logger.warning(
        f"안전한 배치 크기를 찾을 수 없어 최소 배치 크기 {min_batch_size}를 사용합니다."
    )
    return min_batch_size


def adjust_batch_size_for_gpu_memory(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    initial_batch_size: int = 32,
    min_batch_size: int = 1,
    max_batch_size: int = 256,
    tensor_type: torch.dtype = torch.float32,
) -> int:
    """
    모델과 입력 모양을 기반으로 GPU 메모리에 맞게 배치 크기 자동 조정

    Args:
        model: PyTorch 모델
        input_shape: 배치 차원을 제외한 입력 텐서 모양
        initial_batch_size: 초기 배치 크기 (기본값: 32)
        min_batch_size: 최소 배치 크기 (기본값: 1)
        max_batch_size: 최대 배치 크기 (기본값: 256)
        tensor_type: 입력 텐서 타입 (기본값: torch.float32)

    Returns:
        조정된 배치 크기
    """
    # GPU 사용 가능 여부 확인
    if not torch.cuda.is_available():
        logger.info("GPU를 사용할 수 없습니다. CPU 배치 크기를 반환합니다.")
        return min(initial_batch_size, max_batch_size)

    device = torch.device("cuda")

    # 모델을 GPU로 이동
    model.to(device)

    # 배치 크기 이진 탐색
    low = min_batch_size
    high = max_batch_size
    best_batch_size = low

    while low <= high:
        # 현재 배치 크기 (중간 값)
        batch_size = (low + high) // 2

        try:
            # 가비지 컬렉션 및 캐시 정리
            gc.collect()
            torch.cuda.empty_cache()

            # 임시 입력 생성
            tmp_input = torch.zeros(
                (batch_size,) + input_shape, dtype=tensor_type, device=device
            )

            # 순전파 테스트
            with torch.no_grad():
                _ = model(tmp_input)

            # 성공하면 더 큰 배치 크기 시도
            best_batch_size = batch_size
            low = batch_size + 1

            # 메모리 정리
            del tmp_input
            torch.cuda.empty_cache()

        except RuntimeError as e:
            # 메모리 오류 발생 시 더 작은 배치 크기 시도
            high = batch_size - 1

            # 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"자동 조정된 배치 크기: {best_batch_size}")
    return best_batch_size


class DynamicBatchSizeController:
    """동적 배치 크기 컨트롤러"""

    def __init__(
        self,
        initial_batch_size: int = 1000,
        min_batch_size: int = 100,
        max_batch_size: int = 10000,
        growth_rate: float = 1.2,
        reduction_rate: float = 0.5,
    ):
        """
        초기화

        Args:
            initial_batch_size: 초기 배치 크기
            min_batch_size: 최소 배치 크기
            max_batch_size: 최대 배치 크기
            growth_rate: 증가율 (1보다 큰 값: 예: 1.2는 20% 증가)
            reduction_rate: 감소율 (0과 1 사이의 값: 예: 0.5는 50% 감소)
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.growth_rate = growth_rate
        self.reduction_rate = reduction_rate

        # 성능 추적
        self.adjustment_history = []
        self.adjustment_count = 0

    def get_batch_size(self) -> int:
        """
        현재 배치 크기 반환

        Returns:
            현재 배치 크기
        """
        return self.current_batch_size

    def increase_batch_size(self) -> int:
        """
        배치 크기 증가

        Returns:
            증가된 배치 크기
        """
        old_size = self.current_batch_size
        self.current_batch_size = min(
            self.max_batch_size, int(self.current_batch_size * self.growth_rate)
        )

        # 크기가 변경된 경우만 이력 추가
        if old_size != self.current_batch_size:
            self.adjustment_history.append(
                {
                    "type": "increase",
                    "from": old_size,
                    "to": self.current_batch_size,
                    "count": self.adjustment_count,
                }
            )
            self.adjustment_count += 1

        return self.current_batch_size

    def reduce_batch_size(self) -> int:
        """
        배치 크기 감소

        Returns:
            감소된 배치 크기
        """
        old_size = self.current_batch_size
        self.current_batch_size = max(
            self.min_batch_size, int(self.current_batch_size * self.reduction_rate)
        )

        # 크기가 변경된 경우만 이력 추가
        if old_size != self.current_batch_size:
            self.adjustment_history.append(
                {
                    "type": "reduce",
                    "from": old_size,
                    "to": self.current_batch_size,
                    "count": self.adjustment_count,
                }
            )
            self.adjustment_count += 1

        return self.current_batch_size

    def set_batch_size(self, new_size: int) -> int:
        """
        배치 크기 직접 설정

        Args:
            new_size: 설정할 배치 크기

        Returns:
            설정된 배치 크기 (최소/최대 범위 내로 조정됨)
        """
        old_size = self.current_batch_size
        self.current_batch_size = max(
            self.min_batch_size, min(self.max_batch_size, new_size)
        )

        # 크기가 변경된 경우만 이력 추가
        if old_size != self.current_batch_size:
            self.adjustment_history.append(
                {
                    "type": "set",
                    "from": old_size,
                    "to": self.current_batch_size,
                    "count": self.adjustment_count,
                }
            )
            self.adjustment_count += 1

        return self.current_batch_size

    def get_adjustment_history(self) -> list:
        """
        배치 크기 조정 이력 반환

        Returns:
            배치 크기 조정 이력
        """
        return self.adjustment_history

    def reset(self, new_initial_size: Optional[int] = None) -> None:
        """
        배치 크기 컨트롤러 초기화

        Args:
            new_initial_size: 새 초기 배치 크기 (None이면 현재 값 유지)
        """
        if new_initial_size is not None:
            self.current_batch_size = max(
                self.min_batch_size, min(self.max_batch_size, new_initial_size)
            )

        self.adjustment_history = []
        self.adjustment_count = 0
