"""
모델 훈련 성능 최적화 모듈

이 모듈은 모델 훈련 시 메모리, CPU, GPU 최적화를 위한 유틸리티를 제공합니다.
"""

import os
import sys
import time
import gc
import json
import threading
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, cast
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

from ..utils.error_handler_refactored import get_logger
from ..utils.auto_tuner import AutoTuner, TuningConfig, HyperParameter
from ..utils.memory_manager import MemoryManager, MemoryConfig
from ..utils.cuda_optimizers import AMPTrainer
from ..utils.profiler import Profiler
from ..utils.batch_controller import DynamicBatchSizeController
from ..utils.cache_manager import ThreadLocalCache
from ..utils.performance_utils import MemoryTracker
from ..utils.config_loader import ConfigProxy

# 로거 설정
logger = get_logger(__name__)


@dataclass
class OptimizerConfig:
    """성능 최적화 설정"""

    enable_memory_tracking: bool = True
    enable_dynamic_batch_size: bool = True
    enable_amp: bool = True
    gc_threshold: float = 0.8  # 80% 메모리 사용 시 GC 실행
    cuda_empty_cache_threshold: float = 0.9  # 90% GPU 메모리 사용 시 캐시 비우기
    max_batch_memory_ratio: float = 0.3  # 최대 배치가 전체 메모리의 30%까지 사용 가능
    min_batch_size: int = 4  # 최소 배치 크기
    max_batch_size: int = 128  # 최대 배치 크기


class PerformanceOptimizer:
    """모델 훈련 성능 최적화 클래스"""

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        초기화

        Args:
            config: 최적화 설정
        """
        self.config = config or OptimizerConfig()
        self.memory_log = []
        self.is_tracking = False

    def start_tracking(self) -> None:
        """메모리 추적 시작"""
        if not self.config.enable_memory_tracking:
            return

        self.is_tracking = True
        self.memory_log = []
        logger.info("메모리 사용량 추적 시작")

    def stop_tracking(self) -> None:
        """메모리 추적 중지"""
        self.is_tracking = False
        logger.info(f"메모리 사용량 추적 종료 - {len(self.memory_log)}개 기록")

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        현재 메모리 사용량 반환

        Returns:
            메모리 사용량 정보
        """
        system_memory = psutil.virtual_memory()

        # GPU 메모리 (CUDA 가능한 경우)
        gpu_stats = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_stats[f"gpu_{i}"] = {
                    "total": torch.cuda.get_device_properties(i).total_memory,
                    "used": torch.cuda.memory_allocated(i),
                    "cached": torch.cuda.memory_reserved(i),
                    "free": torch.cuda.get_device_properties(i).total_memory
                    - torch.cuda.memory_allocated(i),
                    "utilization": torch.cuda.utilization(i),
                }

        # 메모리 사용량 정보
        memory_info = {
            "system": {
                "total": system_memory.total,
                "available": system_memory.available,
                "used": system_memory.used,
                "percent": system_memory.percent,
            },
            "process": {
                "rss": psutil.Process().memory_info().rss,
                "vms": psutil.Process().memory_info().vms,
            },
            "gpu": gpu_stats,
            "timestamp": time.time(),
        }

        if self.is_tracking:
            self.memory_log.append(memory_info)

            # 임계값 초과 시 메모리 최적화
            if system_memory.percent > self.config.gc_threshold * 100:
                self.optimize_memory()

            # GPU 메모리 임계값 초과 시 캐시 비우기
            if torch.cuda.is_available() and any(
                stats["used"] / stats["total"] > self.config.cuda_empty_cache_threshold
                for stats in gpu_stats.values()
            ):
                self.clear_gpu_cache()

        return memory_info

    def optimize_memory(self) -> None:
        """메모리 최적화 수행"""
        logger.info("메모리 최적화 수행")

        # 가비지 컬렉션 실행
        gc.collect()

        # CUDA 캐시 비우기
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("메모리 최적화 완료")

    def clear_gpu_cache(self) -> None:
        """GPU 캐시 비우기"""
        if torch.cuda.is_available():
            logger.info("GPU 캐시 비우기")
            torch.cuda.empty_cache()

    def get_optimal_batch_size(
        self,
        initial_batch_size: int,
        input_shape: Tuple,
        model: Optional[torch.nn.Module] = None,
    ) -> int:
        """
        최적의 배치 크기 계산

        Args:
            initial_batch_size: 초기 배치 크기
            input_shape: 입력 텐서 형태
            model: 모델 (선택적)

        Returns:
            최적의 배치 크기
        """
        if not self.config.enable_dynamic_batch_size:
            return initial_batch_size

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 메모리 사용량 추정
        try:
            # 샘플 입력 생성
            sample_input = torch.zeros(1, *input_shape[1:], device=device)

            # 메모리 부하 테스트 (모델 제공 시)
            mem_before = 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()

            # 모델 추론 테스트
            if model is not None:
                _ = model(sample_input)

            # 메모리 사용량 계산
            mem_per_sample = 0
            if torch.cuda.is_available():
                mem_per_sample = torch.cuda.memory_allocated() - mem_before

            # 시스템 메모리 확인
            sys_mem = psutil.virtual_memory()

            # GPU 메모리 제한 (CUDA 가능 시)
            if torch.cuda.is_available():
                total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
                available_gpu_mem = total_gpu_mem - torch.cuda.memory_allocated()
                max_samples = int(
                    available_gpu_mem
                    * self.config.max_batch_memory_ratio
                    / max(1, mem_per_sample)
                )

                # 최소/최대 배치 크기 적용
                optimal_batch_size = max(
                    self.config.min_batch_size,
                    min(max_samples, self.config.max_batch_size),
                )
                logger.info(
                    f"GPU 기반 최적 배치 크기: {optimal_batch_size} (메모리 사용량: {mem_per_sample/1024**2:.2f}MB/샘플)"
                )

                return optimal_batch_size
            else:
                # CPU 메모리 기반 배치 크기 계산
                available_mem = sys_mem.available
                max_samples = int(
                    available_mem
                    * self.config.max_batch_memory_ratio
                    / max(1, sample_input.element_size() * sample_input.nelement())
                )

                # 최소/최대 배치 크기 적용
                optimal_batch_size = max(
                    self.config.min_batch_size,
                    min(max_samples, self.config.max_batch_size),
                )
                logger.info(f"CPU 기반 최적 배치 크기: {optimal_batch_size}")

                return optimal_batch_size

        except Exception as e:
            logger.warning(f"배치 크기 최적화 중 오류: {str(e)}")
            return initial_batch_size

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        메모리 사용량 요약 정보

        Returns:
            메모리 사용량 요약
        """
        if not self.memory_log:
            return {"avg_memory_used": 0, "max_memory_used": 0}

        # 시스템 메모리 통계
        sys_percent = [log["system"]["percent"] for log in self.memory_log]
        avg_sys_percent = sum(sys_percent) / len(sys_percent) if sys_percent else 0
        max_sys_percent = max(sys_percent) if sys_percent else 0

        # 프로세스 메모리 통계
        proc_rss = [log["process"]["rss"] for log in self.memory_log]
        avg_proc_rss = sum(proc_rss) / len(proc_rss) if proc_rss else 0
        max_proc_rss = max(proc_rss) if proc_rss else 0

        # GPU 메모리 통계 (있는 경우)
        gpu_stats = {}
        if (
            self.memory_log
            and "gpu" in self.memory_log[0]
            and self.memory_log[0]["gpu"]
        ):
            for gpu_id in self.memory_log[0]["gpu"]:
                gpu_used = [
                    log["gpu"].get(gpu_id, {}).get("used", 0)
                    for log in self.memory_log
                    if "gpu" in log
                ]
                if gpu_used:
                    gpu_stats[gpu_id] = {
                        "avg_used": sum(gpu_used) / len(gpu_used),
                        "max_used": max(gpu_used),
                    }

        return {
            "system": {
                "avg_percent": avg_sys_percent,
                "max_percent": max_sys_percent,
            },
            "process": {
                "avg_rss_mb": avg_proc_rss / (1024 * 1024),
                "max_rss_mb": max_proc_rss / (1024 * 1024),
            },
            "gpu": gpu_stats,
            "logs_count": len(self.memory_log),
        }


class TrainingOptimizer:
    """모델 훈련 성능 최적화 클래스"""

    def __init__(self, config):
        """
        훈련 최적화기 초기화

        Args:
            config: 훈련 설정
        """
        # 설정 초기화
        if isinstance(config, dict):
            self.config = ConfigProxy(config)
        else:
            self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_trainer = AMPTrainer(self.config)
        self.batch_controller = DynamicBatchSizeController(
            config=self.config,
            initial_batch_size=self.config.safe_get("batch_size", default=32),
        )
        self.cache = ThreadLocalCache()
        self.memory_tracker = MemoryTracker()
        self.profiler = Profiler(self.config)

        # 캐시 관련 초기화
        self.cached_results = {}
        self.cache_lock = threading.Lock()
        self.enable_caching = self.config.safe_get("enable_caching", True)
        self.cache_dir = self.config.safe_get("cache_dir", "cache/performance")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # 자동 튜닝 설정
        self.enable_auto_tuning = self.config.safe_get("enable_auto_tuning", True)
        self.study = None
        self.tuning_trials = self.config.safe_get("tuning_trials", 10)
        self.tuning_timeout = self.config.safe_get("tuning_timeout", 600)
        self.max_workers = self.config.safe_get("max_workers", 4)

        logger.info("훈련 최적화기 초기화 완료")

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        모델 최적화

        Args:
            model: 최적화할 PyTorch 모델

        Returns:
            최적화된 모델
        """
        if not isinstance(model, nn.Module):
            logger.warning(f"지원되지 않는 모델 유형: {type(model)}")
            return model

        # 메모리 사용량 추적 시작
        self.memory_tracker.start()

        try:
            # 모델을 GPU로 이동 (가능한 경우)
            model = model.to(self.device)

            # 모델 컴파일 (PyTorch 2.0 이상 & 설정 활성화된 경우)
            if self.config.safe_get("enable_compile", False) and hasattr(
                torch, "compile"
            ):
                logger.info(
                    f"모델 컴파일 중 (모드: {self.config.safe_get('compile_mode', 'default')})..."
                )
                try:
                    # 타입 오류 방지를 위한 타입 무시
                    model = torch.compile(model, mode=self.config.safe_get("compile_mode", "default"))  # type: ignore
                    logger.info("모델 컴파일 완료")
                except Exception as e:
                    logger.error(f"모델 컴파일 실패: {str(e)}")
        except Exception as e:
            logger.error(f"모델 최적화 중 오류: {str(e)}")

        return model

    def optimize_training_function(
        self, train_fn: Callable, param_grid: Dict[str, List[Any]]
    ) -> Tuple[Dict[str, Any], Callable]:
        """
        훈련 함수 최적화

        Args:
            train_fn: 최적화할 훈련 함수
            param_grid: 하이퍼파라미터 그리드

        Returns:
            (최적 파라미터, 최적화된 훈련 함수) 튜플
        """
        # 자동 튜닝 비활성화 시 기본값 사용
        if not self.enable_auto_tuning:
            # 기본 파라미터 선택
            default_params = {k: v[0] for k, v in param_grid.items()}
            logger.info("자동 튜닝 건너뜀, 기본 파라미터 사용")
            return default_params, train_fn

        # 객체를 튜닝 함수로 전달하기 위한 클로저
        def objective(trial):
            # 하이퍼파라미터 샘플링
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name,
                        min(param_values),
                        max(param_values),
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        min(param_values),
                        max(param_values),
                    )
                else:
                    # 카테고리형 파라미터
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )

            # 훈련 함수 실행 및 결과 평가
            try:
                result = train_fn(**params)
                # 결과에서 메트릭 추출 (다양한 반환 형식 처리)
                if isinstance(result, dict) and "loss" in result:
                    metric = result["loss"]
                elif isinstance(result, dict) and "val_loss" in result:
                    metric = result["val_loss"]
                elif isinstance(result, float):
                    metric = result
                else:
                    metric = 1.0  # 기본값
                return metric
            except Exception as e:
                logger.error(f"튜닝 중 오류: {str(e)}")
                return float("inf")  # 오류 발생 시 최대값 반환

        # 튜닝 실행
        import optuna

        logger.info("하이퍼파라미터 튜닝 시작")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=self.tuning_trials,
            timeout=self.tuning_timeout,
            n_jobs=1,  # 병렬 처리는 상위 레벨에서 수행
        )

        # 최적 파라미터 획득
        best_params = study.best_params
        logger.info(f"최적 파라미터 발견: {best_params}")

        # 최적화된 훈련 함수 생성
        def optimized_train_fn(*args, **kwargs):
            # 최적 파라미터 주입
            updated_kwargs = kwargs.copy()
            updated_kwargs.update(best_params)

            # AMP 사용
            if self.config.safe_get("enable_amp", True) and torch.cuda.is_available():
                return self.train_with_amp(train_fn, *args, **updated_kwargs)

            # 일반 훈련 실행
            return train_fn(*args, **updated_kwargs)

        return best_params, optimized_train_fn

    def parallel_train(
        self,
        train_fn: Callable,
        data_list: List[Any],
        shared_args: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        병렬 훈련 실행

        Args:
            train_fn: 훈련 함수
            data_list: 데이터 목록
            shared_args: 공유 인수

        Returns:
            각 훈련의 결과 목록
        """
        if not data_list:
            return []

        # 공유 인수 초기화
        if shared_args is None:
            shared_args = {}

        # 병렬 처리가 비활성화된 경우 순차 실행
        if not self.config.safe_get("enable_parallel", True):
            results = []
            for data in data_list:
                results.append(train_fn(data, **shared_args))
            return results

        # 작업자 수 설정
        n_workers = min(self.config.safe_get("max_workers", 4), len(data_list))

        # 병렬 처리를 위한 실행 함수
        def _execute_train(data):
            try:
                # 메모리 최적화
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # AMP 사용 (설정된 경우)
                if (
                    self.config.safe_get("enable_amp", True)
                    and torch.cuda.is_available()
                ):
                    result = self.train_with_amp(train_fn, data, **shared_args)
                else:
                    result = train_fn(data, **shared_args)

                return result
            except Exception as e:
                logger.error(f"병렬 훈련 중 오류: {str(e)}")
                return {"error": str(e)}

        # 병렬 처리 실행
        if self.config.safe_get("use_processes", False):
            # 프로세스 풀 사용
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_execute_train, data_list))
        else:
            # 스레드 풀 사용
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_execute_train, data_list))

        return results

    def train_with_amp(self, train_fn: Callable, *args, **kwargs) -> Any:
        """
        혼합 정밀도를 사용하여 훈련

        Args:
            train_fn: 훈련 함수
            *args, **kwargs: 훈련 함수에 전달할 인자

        Returns:
            훈련 함수의 결과
        """
        if (
            not self.config.safe_get("enable_amp", True)
            or not torch.cuda.is_available()
        ):
            # 혼합 정밀도 비활성화 또는 CUDA 없으면 일반 훈련
            return train_fn(*args, **kwargs)

        # 혼합 정밀도 적용
        try:
            # 가장 단순한 방식으로 호출
            with torch.cuda.amp.autocast():  # 매개변수 없이 호출
                return train_fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"AMP 사용 중 오류 발생: {str(e)}")
            return train_fn(*args, **kwargs)

    def cleanup(self):
        """리소스 정리"""
        # 메모리 추적 중단
        self.memory_tracker.stop()

        # 가비지 컬렉션 강제 실행
        if self.config.safe_get("aggressive_gc", default=False):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("훈련 최적화기 리소스 정리 완료")

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.cleanup()

    def optimize_batch_size(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        min_batch: int = 4,
        max_batch: int = 128,
        step: int = 4,
        max_memory_usage: float = 0.8,
    ) -> int:
        """
        메모리 사용량에 따라 최적의 배치 크기 계산

        Args:
            model: 최적화할 모델
            input_shape: 입력 텐서 형태 (배치 차원 제외)
            min_batch: 최소 배치 크기
            max_batch: 최대 배치 크기
            step: 배치 크기 증가 단계
            max_memory_usage: 최대 메모리 사용률 (0.0 ~ 1.0)

        Returns:
            최적의 배치 크기
        """
        if not torch.cuda.is_available():
            logger.info("GPU를 사용할 수 없습니다. 기본 배치 크기를 반환합니다.")
            return min_batch

        # 모델을 GPU로 이동
        device = torch.device("cuda")
        model.to(device)

        # 메모리 초기화
        torch.cuda.empty_cache()
        gc.collect()

        # 초기 메모리 사용량 기록
        initial_memory = torch.cuda.memory_allocated()

        # 최대 메모리 계산
        total_memory = torch.cuda.get_device_properties(0).total_memory
        max_allowed_memory = total_memory * max_memory_usage

        optimal_batch = min_batch

        try:
            # 배치 크기를 점진적으로 증가시키며 메모리 사용량 테스트
            for batch_size in range(min_batch, max_batch + 1, step):
                # 메모리 초기화
                torch.cuda.empty_cache()
                gc.collect()

                try:
                    # 테스트 입력 생성
                    dummy_input = torch.randn(
                        (batch_size,) + input_shape, device=device
                    )

                    # 순전파 (메모리 사용량 테스트)
                    with torch.no_grad():
                        _ = model(dummy_input)

                    # 현재 메모리 사용량 확인
                    current_memory = torch.cuda.memory_allocated()

                    # 메모리 한도 초과 여부 확인
                    if current_memory > max_allowed_memory:
                        break

                    # 현재 배치 크기가 가능하므로 업데이트
                    optimal_batch = batch_size

                    logger.debug(
                        f"배치 크기 {batch_size} 테스트 성공: {current_memory / 1024**2:.2f} MB"
                    )

                except RuntimeError as e:
                    # 메모리 부족 오류가 발생하면 이전 배치 크기가 최적
                    if "CUDA out of memory" in str(e):
                        break
                    else:
                        logger.error(f"배치 크기 최적화 중 오류: {str(e)}")
                        break

            logger.info(f"최적의 배치 크기 계산 완료: {optimal_batch}")

        except Exception as e:
            logger.error(f"배치 크기 최적화 중 오류: {str(e)}")
            optimal_batch = min_batch

        # 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()

        return optimal_batch

    def optimize_training_loop(
        self,
        train_fn: Callable,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        data_loader: Any,
        epochs: int,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """훈련 루프 최적화"""
        try:
            self.memory_tracker.start()

            with self.profiler.profile("training_optimization"):
                # 모델 최적화
                model = self.optimize_model(model)

                # 훈련 결과 저장
                results = {
                    "epochs": epochs,
                    "losses": [],
                    "training_time": 0,
                    "memory_usage": {},
                }

                # 훈련 루프
                for epoch in range(epochs):
                    epoch_start = time.time()
                    epoch_loss = 0.0
                    batches = 0

                    # 배치 처리
                    for batch in data_loader:
                        # 배치 크기 동적 조정
                        batch_size = self.batch_controller.adjust_batch_size(
                            data_loader
                        )

                        # AMP를 사용한 학습 스텝
                        metrics = self.amp_trainer.train_step(
                            model, optimizer, loss_fn, batch
                        )

                        epoch_loss += metrics["loss"]
                        batches += 1

                    # 에폭 평균 손실
                    avg_loss = epoch_loss / batches if batches > 0 else 0.0
                    results["losses"].append(avg_loss)

                    # 에폭 정보 출력
                    epoch_time = time.time() - epoch_start
                    logger.info(
                        f"에폭 {epoch+1}/{epochs} - 손실: {avg_loss:.4f}, 소요 시간: {epoch_time:.2f}초"
                    )

                    # 메모리 정리
                    if (
                        self.config.safe_get("aggressive_gc", default=False)
                        and epoch % 5 == 0
                    ):
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # 훈련 시간 계산
                results["training_time"] = time.time() - epoch_start

                # 메모리 사용량 보고서 추가
                results["memory_usage"] = self.memory_tracker.get_memory_log()

                return model, results

        except Exception as e:
            logger.error(f"훈련 최적화 중 오류 발생: {str(e)}")
            return model, {"error": str(e)}

        finally:
            self.memory_tracker.stop()


# 싱글톤 인스턴스
_optimizer_instance = None


def get_training_optimizer(
    config: Optional[OptimizerConfig] = None,
) -> TrainingOptimizer:
    """
    훈련 최적화기 인스턴스 반환 (싱글톤)

    Args:
        config: 최적화 설정

    Returns:
        TrainingOptimizer 인스턴스
    """
    global _optimizer_instance

    if _optimizer_instance is None:
        _optimizer_instance = TrainingOptimizer(config)

    return _optimizer_instance
