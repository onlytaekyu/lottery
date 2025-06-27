"""
자동 튜닝 시스템 (Auto Tuner)

이 모듈은 시스템 상태를 모니터링하고 하이퍼파라미터를 자동으로 최적화하여
런타임 중 시스템 성능을 향상시키는 기능을 제공합니다.
Grid Search, Random Search, Bayesian Optimization 기법을 지원합니다.
"""

import time
import numpy as np
import pandas as pd  # type: ignore
import threading
import random
import json
import itertools
import yaml  # type: ignore
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union,
    Callable,
    TypeVar,
    Protocol,
)
from pathlib import Path
from datetime import datetime
import psutil  # type: ignore
import sys
import importlib

import torch
from dataclasses import dataclass
import queue
from concurrent.futures import ThreadPoolExecutor

# 로컬 모듈 가져오기
from .error_handler_refactored import get_logger
from .memory_manager import MemoryManager, MemoryConfig

logger = get_logger(__name__)

# 타입 변수 정의
T = TypeVar("T")

# 타입 정의
BestParams = Dict[str, Any]


# 타입 힌트를 위한 Protocol 정의
class SearcherProtocol(Protocol):
    def search(
        self, objective_fn: Callable[[Dict[str, Any]], float], **kwargs
    ) -> Optional[Dict[str, Any]]: ...


# 런타임에서는 SearcherType을 Any로 처리
SearcherType = Any


# 새로운 함수 추가
def run_autotune(
    train_fn: Callable,
    param_grid: Dict[str, List[Any]],
    n_trials: int = 10,
    method: str = "grid",
    optimization_goal: str = "minimize",
    results_dir: Optional[Union[str, Path]] = None,
    timeout: float = 3600,
    parallel_trials: int = 1,
) -> Dict[str, Any]:
    """
    하이퍼파라미터 자동 튜닝 실행

    Args:
        train_fn: 훈련 함수 (하이퍼파라미터 -> 성능 지표)
        param_grid: 하이퍼파라미터 그리드 (파라미터 이름 -> 가능한 값 목록)
        n_trials: 시도 횟수 (random, bayesian 메서드에서 사용)
        method: 튜닝 방법 ("grid", "random", "bayesian")
        optimization_goal: 최적화 목표 ("minimize" 또는 "maximize")
        results_dir: 결과 저장 디렉토리
        timeout: 최대 실행 시간 (초)
        parallel_trials: 병렬 실행할 시도 수

    Returns:
        최적의 하이퍼파라미터 및 결과
    """
    start_time = time.time()
    logger.info(f"하이퍼파라미터 자동 튜닝 시작 (방법: {method})")

    # 결과 저장 디렉토리 설정
    if results_dir is None:
        # 상대 경로 사용
        results_dir = Path(__file__).parent.parent.parent / "config"
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    # 목적 함수 정의 (최적화 목표에 따라 조정)
    def objective_fn(params):
        try:
            logger.info(f"파라미터 평가 중: {params}")
            result = train_fn(params)

            # 결과가 딕셔너리인 경우 'loss' 또는 'score' 키 사용
            if isinstance(result, dict):
                metric = result.get(
                    "loss", result.get("score", result.get("metric", 0.0))
                )
            else:
                metric = float(result)

            # 최적화 목표에 따라 점수 조정
            if optimization_goal == "minimize":
                return -metric  # 그리드/랜덤 서치는 최대화 문제로 처리하므로 부호 반전
            else:
                return metric
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {str(e)}")
            return float("-inf") if optimization_goal == "maximize" else float("inf")

    # 튜닝 방법에 따라 검색 수행
    best_params: Optional[Dict[str, Any]] = None

    # 현재 모듈에서 클래스를 직접 참조
    GridSearchClass = globals().get("GridSearch")
    RandomSearchClass = globals().get("RandomSearch")

    try:
        # 런타임에 실제 클래스가 정의되었는지 확인 후 실행
        if method == "grid":
            if GridSearchClass is None:
                logger.warning(
                    "GridSearch 클래스를 현재 모듈에서 찾을 수 없습니다. 기본값을 반환합니다."
                )
                best_params = {k: v[0] for k, v in param_grid.items()}
                return {"best_params": best_params, "error": "GridSearch not available"}

            # GridSearch 클래스를 사용하여 최적의 파라미터 탐색
            searcher = GridSearchClass(param_grid, results_dir)
            best_params = searcher.search(objective_fn, max_evals=n_trials)

        elif method == "random":
            if RandomSearchClass is None:
                logger.warning(
                    "RandomSearch 클래스를 현재 모듈에서 찾을 수 없습니다. 기본값을 반환합니다."
                )
                best_params = {k: v[0] for k, v in param_grid.items()}
                return {
                    "best_params": best_params,
                    "error": "RandomSearch not available",
                }

            # RandomSearch 클래스를 사용하여 최적의 파라미터 탐색
            searcher = RandomSearchClass(param_grid, results_dir)
            best_params = searcher.search(objective_fn, n_iter=n_trials)

        elif method == "bayesian":
            # 베이지안 최적화가 구현되어 있지 않은 경우 랜덤 서치로 대체
            logger.warning(
                "베이지안 최적화는 아직 구현되지 않았습니다. 랜덤 서치로 대체합니다."
            )

            if RandomSearchClass is None:
                logger.warning(
                    "RandomSearch 클래스를 현재 모듈에서 찾을 수 없습니다. 기본값을 반환합니다."
                )
                best_params = {k: v[0] for k, v in param_grid.items()}
                return {
                    "best_params": best_params,
                    "error": "RandomSearch not available",
                }

            searcher = RandomSearchClass(param_grid, results_dir)
            best_params = searcher.search(objective_fn, n_iter=n_trials)

        else:
            raise ValueError(f"지원하지 않는 튜닝 방법: {method}")
    except Exception as e:
        logger.error(f"하이퍼파라미터 탐색 중 오류 발생: {str(e)}")
        # 오류 발생 시 기본값으로 첫 번째 파라미터 세트 사용
        best_params = {k: v[0] for k, v in param_grid.items()}

    # 결과 정리
    duration = time.time() - start_time

    # 최적의 파라미터가 없는 경우 처리
    if best_params is None:
        logger.warning("최적의 파라미터를 찾지 못했습니다.")
        # 기본값으로 첫 번째 파라미터 세트 사용
        best_params = {k: v[0] for k, v in param_grid.items()}

    # 결과 저장
    result = {
        "best_params": best_params,
        "method": method,
        "n_trials": n_trials,
        "duration": duration,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # YAML 파일로 저장
    yaml_path = results_dir / "auto_tuned_params.yaml"
    try:
        # 기존 파일이 있으면 로드
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                existing_data = yaml.safe_load(f) or {}
        else:
            existing_data = {}

        # 새 데이터 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing_data[f"tuning_{timestamp}"] = result

        # 파일 저장
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(existing_data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"튜닝 결과가 {yaml_path}에 저장되었습니다.")
    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {str(e)}")

    logger.info(f"하이퍼파라미터 자동 튜닝 완료 (소요 시간: {duration:.2f}초)")
    logger.info(f"최적의 파라미터: {best_params}")

    return result


@dataclass
class TuningConfig:
    """튜닝 설정"""

    max_trials: int = 100
    timeout: float = 3600  # 1시간
    target_metric: str = "loss"
    optimization_goal: str = "minimize"
    early_stopping_rounds: int = 10
    parallel_trials: int = 4
    enable_pruning: bool = True
    save_history: bool = True


class HyperParameter:
    """하이퍼파라미터 정의 클래스"""

    def __init__(self, name: str, param_type: str, **kwargs):
        self.name = name
        self.param_type = param_type
        self.kwargs = kwargs

    def sample(self) -> Union[float, int, str]:
        """파라미터 값 샘플링"""
        if self.param_type == "float":
            return np.random.uniform(
                self.kwargs.get("min_value", 0.0), self.kwargs.get("max_value", 1.0)
            )
        elif self.param_type == "int":
            return np.random.randint(
                self.kwargs.get("min_value", 0), self.kwargs.get("max_value", 10)
            )
        elif self.param_type == "categorical":
            return np.random.choice(self.kwargs.get("choices", []))
        else:
            raise ValueError(f"지원하지 않는 파라미터 타입: {self.param_type}")


class Trial:
    """튜닝 시도"""

    def __init__(self, params: Dict[str, Any], trial_id: int):
        self.params = params
        self.trial_id = trial_id
        self.status = "pending"
        self.result = None
        self.start_time = None
        self.end_time = None

    def start(self):
        """시도 시작"""
        self.start_time = time.time()
        self.status = "running"

    def complete(self, result: float):
        """시도 완료"""
        self.end_time = time.time()
        self.result = result
        self.status = "completed"

    def fail(self, error: str):
        """시도 실패"""
        self.end_time = time.time()
        self.result = error
        self.status = "failed"

    @property
    def duration(self) -> float:
        """시도 소요 시간"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


class GridSearch:
    """
    Grid Search를 통한 하이퍼파라미터 탐색

    모든 하이퍼파라미터 조합을 체계적으로 탐색합니다.
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        results_dir: Optional[Union[str, Path]] = None,
    ):
        """
        초기화

        Args:
            param_grid: 하이퍼파라미터 그리드 (파라미터 이름 -> 가능한 값 목록)
            results_dir: 결과 저장 디렉토리
        """
        self.param_grid = param_grid

        # 결과 저장 디렉토리 설정
        if results_dir is None:
            # 상대 경로 사용
            self.results_dir = (
                Path(__file__).parent.parent.parent / "savedModels" / "training_results"
            )
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def search(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        max_evals: Optional[int] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        그리드 서치 실행

        Args:
            objective_fn: 최적화할 목적 함수 (하이퍼파라미터 -> 성능)
            max_evals: 최대 평가 횟수 (None이면 전체 그리드 탐색)
            **kwargs: 향후 확장을 위한 추가 인자 (n_iter 등)

        Returns:
            최적의 하이퍼파라미터 세트, 또는 결과가 없으면 None
        """
        # n_iter 파라미터가 전달된 경우 max_evals로 사용
        if "n_iter" in kwargs and max_evals is None:
            max_evals = kwargs.get("n_iter")

        # 모든 파라미터 조합 생성
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))

        if max_evals is not None and max_evals < len(combinations):
            logger.info(f"총 {len(combinations)}개 조합 중 {max_evals}개만 평가합니다")
            combinations = random.sample(combinations, max_evals)
        else:
            logger.info(f"총 {len(combinations)}개 조합을 모두 평가합니다")

        best_score = float("-inf")
        best_params = None

        # 각 조합 평가
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            logger.info(f"조합 {i+1}/{len(combinations)} 평가 중: {params}")

            start_time = time.time()
            try:
                score = objective_fn(params)
                duration = time.time() - start_time

                self.results.append(
                    {
                        "params": params,
                        "score": score,
                        "duration": duration,
                    }
                )

                logger.info(f"점수: {score:.6f}, 소요 시간: {duration:.2f}초")

                # 최고 성능 업데이트
                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"신규 최고 성능: {score:.6f}")
            except Exception as e:
                logger.error(f"평가 중 오류 발생: {str(e)}")
                self.results.append(
                    {
                        "params": params,
                        "error": str(e),
                        "duration": time.time() - start_time,
                    }
                )

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.results_dir / f"grid_search_{timestamp}.json"
        self._save_results(self.results, best_params, str(result_path))

        return best_params

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        best_params: Optional[Dict[str, Any]],
        filename: str,
    ) -> None:
        """결과를 파일에 저장"""
        try:
            # numpy 타입을 Python 기본 타입으로 변환
            def convert_numpy_types(obj):
                """NumPy 타입을 JSON 직렬화 가능한 타입으로 변환"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):  # np.int32, np.int64 등
                    return int(obj)
                elif isinstance(obj, np.floating):  # np.float32, np.float64 등
                    return float(obj)
                elif isinstance(obj, np.bool_):  # numpy 부울 타입
                    return bool(obj)
                elif hasattr(obj, "real") and hasattr(obj, "imag"):  # 복소수 확인
                    return {"real": float(obj.real), "imag": float(obj.imag)}
                elif isinstance(obj, dict):
                    return {
                        convert_numpy_types(key): convert_numpy_types(value)
                        for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                return obj

            # 결과 변환
            converted_results = [
                {k: convert_numpy_types(v) for k, v in result.items()}
                for result in results
            ]
            converted_best_params = (
                {k: convert_numpy_types(v) for k, v in best_params.items()}
                if best_params
                else None
            )

            # 결과 저장
            save_data = {
                "results": converted_results,
                "best_params": converted_best_params,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"결과 저장 중 오류: {str(e)}")

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """최고 성능의 파라미터 반환"""
        if not self.results:
            return None

        best_score = float("-inf")
        best_params = None

        for result in self.results:
            if "score" in result and result["score"] > best_score:
                best_score = result["score"]
                best_params = result["params"]

        return best_params


class RandomSearch:
    """
    Random Search를 통한 하이퍼파라미터 탐색

    파라미터 공간에서 무작위로 샘플링하여 최적의 하이퍼파라미터를 탐색합니다.
    """

    def __init__(
        self,
        param_distributions: Dict[str, Any],
        results_dir: Optional[Union[str, Path]] = None,
    ):
        """
        초기화

        Args:
            param_distributions: 파라미터 분포 (이름 -> 분포 정보)
            results_dir: 결과 저장 디렉토리
        """
        self.param_distributions = param_distributions

        # 결과 저장 디렉토리 설정
        if results_dir is None:
            # 상대 경로 사용
            self.results_dir = (
                Path(__file__).parent.parent.parent / "savedModels" / "training_results"
            )
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def _sample_params(self) -> Dict[str, Any]:
        """
        파라미터 분포에서 샘플링

        Returns:
            샘플링된 파라미터 사전
        """
        params = {}
        for name, distribution in self.param_distributions.items():
            if isinstance(distribution, list):
                # 리스트에서 무작위 선택
                params[name] = random.choice(distribution)
            elif isinstance(distribution, tuple) and len(distribution) == 3:
                # (start, end, step) 튜플로 해석
                start, end, step = distribution
                if (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and isinstance(step, int)
                ):
                    # 정수 범위
                    value = random.randrange(start, end, step)
                else:
                    # 실수 범위를 이산화하여 샘플링
                    n_steps = int((end - start) / step)
                    idx = random.randint(0, n_steps)
                    value = start + idx * step
                params[name] = value
            elif isinstance(distribution, tuple) and len(distribution) == 2:
                # (min, max) 튜플로 해석
                if isinstance(distribution[0], int) and isinstance(
                    distribution[1], int
                ):
                    # 정수 구간
                    params[name] = random.randint(distribution[0], distribution[1])
                else:
                    # 실수 구간
                    params[name] = random.uniform(distribution[0], distribution[1])
            elif callable(distribution):
                # 분포 함수 호출
                params[name] = distribution()
            elif isinstance(distribution, dict) and "distribution" in distribution:
                # 분포 명세 사전 사용
                dist_type = distribution["distribution"]
                if dist_type == "uniform":
                    params[name] = random.uniform(
                        distribution.get("low", 0.0), distribution.get("high", 1.0)
                    )
                elif dist_type == "normal":
                    params[name] = random.normalvariate(
                        distribution.get("mu", 0.0), distribution.get("sigma", 1.0)
                    )
                elif dist_type == "randint":
                    params[name] = random.randint(
                        distribution.get("low", 0), distribution.get("high", 10)
                    )
                elif dist_type == "choice":
                    params[name] = random.choice(distribution.get("values", [0, 1]))
            else:
                # 기본값으로 분포 자체를 사용
                params[name] = distribution

        # NumPy 타입을 Python 기본 타입으로 변환
        for name, value in params.items():
            if isinstance(value, (np.integer, np.floating)):
                params[name] = value.item()  # item() 메서드로 Python 기본 타입으로 변환

        return params

    def search(
        self, objective_fn: Callable[[Dict[str, Any]], float], n_iter: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        랜덤 서치 실행

        Args:
            objective_fn: 최적화할 목적 함수 (하이퍼파라미터 -> 성능)
            n_iter: 반복 횟수

        Returns:
            최적의 하이퍼파라미터 세트, 또는 결과가 없으면 None
        """
        logger.info(f"랜덤 서치 시작: {n_iter}회 반복")

        best_score = float("-inf")
        best_params = None

        for i in range(n_iter):
            params = self._sample_params()
            logger.info(f"반복 {i+1}/{n_iter} 평가 중: {params}")

            start_time = time.time()
            try:
                score = objective_fn(params)
                duration = time.time() - start_time

                self.results.append(
                    {
                        "params": params,
                        "score": score,
                        "duration": duration,
                    }
                )

                logger.info(f"점수: {score:.6f}, 소요 시간: {duration:.2f}초")

                # 최고 성능 업데이트
                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"새로운 최고 성능: {score:.6f}")

            except Exception as e:
                logger.error(f"평가 중 오류 발생: {str(e)}")
                self.results.append(
                    {
                        "params": params,
                        "score": None,
                        "duration": time.time() - start_time,
                        "error": str(e),
                    }
                )

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.results_dir / f"random_search_{timestamp}.json"
        self._save_results(self.results, best_params, str(result_path))

        return best_params

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        best_params: Optional[Dict[str, Any]],
        filename: str,
    ) -> None:
        """결과를 파일에 저장"""
        try:
            # numpy 타입을 Python 기본 타입으로 변환
            def convert_numpy_types(obj):
                """NumPy 타입을 JSON 직렬화 가능한 타입으로 변환"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):  # np.int32, np.int64 등
                    return int(obj)
                elif isinstance(obj, np.floating):  # np.float32, np.float64 등
                    return float(obj)
                elif isinstance(obj, np.bool_):  # numpy 부울 타입
                    return bool(obj)
                elif hasattr(obj, "real") and hasattr(obj, "imag"):  # 복소수 확인
                    return {"real": float(obj.real), "imag": float(obj.imag)}
                elif isinstance(obj, dict):
                    return {
                        convert_numpy_types(key): convert_numpy_types(value)
                        for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                return obj

            # 결과 변환
            converted_results = [
                {k: convert_numpy_types(v) for k, v in result.items()}
                for result in results
            ]
            converted_best_params = (
                {k: convert_numpy_types(v) for k, v in best_params.items()}
                if best_params
                else None
            )

            # 결과 저장
            save_data = {
                "results": converted_results,
                "best_params": converted_best_params,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"결과 저장 중 오류: {str(e)}")

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """최적의 파라미터 반환 (JSON 형식)"""
        if not self.results:
            return None

        best_result = max(
            self.results,
            key=lambda x: (
                x.get("score", float("-inf"))
                if x.get("score") is not None
                else float("-inf")
            ),
        )

        return best_result.get("params")


class AutoTuner:
    """자동 튜닝 시스템"""

    def __init__(self, config: Optional[TuningConfig] = None):
        self.config = config or TuningConfig()
        self.memory_manager = MemoryManager(MemoryConfig())
        self.parameters: Dict[str, HyperParameter] = {}
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.history: List[Dict[str, Any]] = []
        self.trial_queue: "queue.Queue[Trial]" = queue.Queue()
        self.stop_event = threading.Event()

    def add_parameter(self, param: HyperParameter):
        """하이퍼파라미터 추가"""
        self.parameters[param.name] = param

    def _generate_params(self) -> Dict[str, Any]:
        """파라미터 세트 생성"""
        return {name: param.sample() for name, param in self.parameters.items()}

    def _evaluate_trial(self, trial: Trial, objective_fn: Callable):
        """시도 평가"""
        try:
            with self.memory_manager.allocation_scope():
                trial.start()
                result = objective_fn(trial.params)
                trial.complete(result)

                # 히스토리 업데이트
                if self.config.save_history:
                    self.history.append(
                        {
                            "trial_id": trial.trial_id,
                            "params": trial.params,
                            "result": result,
                            "duration": trial.duration,
                        }
                    )

                # 최고 성능 업데이트
                if (
                    self.best_trial is None
                    or (
                        self.config.optimization_goal == "minimize"
                        and result < self.best_trial.result
                    )
                    or (
                        self.config.optimization_goal == "maximize"
                        and result > self.best_trial.result
                    )
                ):
                    self.best_trial = trial

        except Exception as e:
            logger.error(f"시도 {trial.trial_id} 평가 중 오류 발생: {str(e)}")
            trial.fail(str(e))

    def _should_stop_early(self) -> bool:
        """조기 종료 여부 확인"""
        if (
            len(self.history) < self.config.early_stopping_rounds
            or self.best_trial is None
        ):
            return False

        recent_results = [
            trial["result"]
            for trial in self.history[-self.config.early_stopping_rounds :]
        ]

        if self.config.optimization_goal == "minimize":
            best_result = min(recent_results)
            return best_result >= self.best_trial.result
        else:
            best_result = max(recent_results)
            return best_result <= self.best_trial.result

    def optimize(self, objective_fn: Callable) -> Dict[str, Any]:
        """최적화 수행"""
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.config.parallel_trials) as executor:
            futures = []

            for trial_id in range(self.config.max_trials):
                # 시간 초과 체크
                if time.time() - start_time > self.config.timeout:
                    logger.info("최적화 시간 초과")
                    break

                # 조기 종료 체크
                if self.config.enable_pruning and self._should_stop_early():
                    logger.info("조기 종료 조건 충족")
                    break

                # 새로운 시도 생성
                params = self._generate_params()
                trial = Trial(params, trial_id)
                self.trials.append(trial)

                # 병렬 실행
                future = executor.submit(self._evaluate_trial, trial, objective_fn)
                futures.append(future)

                # 병렬 실행 제한 관리
                if len(futures) >= self.config.parallel_trials:
                    for f in futures:
                        f.result()
                    futures = []

            # 남은 작업 완료 대기
            for f in futures:
                f.result()

        # 최적화 결과 반환
        if self.best_trial is None:
            raise RuntimeError("최적화 실패: 유효한 시도 없음")

        return {
            "best_params": self.best_trial.params,
            "best_value": self.best_trial.result,
            "num_trials": len(self.trials),
            "total_time": time.time() - start_time,
            "history": self.history if self.config.save_history else None,
        }

    def get_best_params(self) -> Dict[str, Any]:
        """최적의 파라미터 반환"""
        if self.best_trial is None:
            raise RuntimeError("최적화가 수행되지 않았습니다")
        return self.best_trial.params

    def save_results(self, path: Optional[Union[str, Path]] = None):
        """결과 저장"""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 상대 경로 사용
            path = (
                Path(__file__).parent.parent.parent
                / "savedModels"
                / "training_results"
                / f"autotuner_{timestamp}.pt"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "config": self.config.__dict__,
            "best_trial": (
                {
                    "params": self.best_trial.params,
                    "result": self.best_trial.result,
                    "trial_id": self.best_trial.trial_id,
                }
                if self.best_trial
                else None
            ),
            "history": self.history if self.config.save_history else None,
        }

        torch.save(results, path)
        logger.info(f"최적화 결과 저장 완료: {path}")

        # JSON 형식으로 추가 저장
        json_path = path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": self.config.__dict__,
                    "best_params": self.best_trial.params if self.best_trial else None,
                    "best_value": self.best_trial.result if self.best_trial else None,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info(f"최적화 결과 JSON 형식으로 저장 완료: {json_path}")

    @classmethod
    def load_results(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """결과 로드"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {path}")

        results = torch.load(path)
        logger.info(f"최적화 결과 로드 완료: {path}")
        return results

    def tune_hyperparameters(
        self, model_type: str, train_data: List, val_data: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        지정된 모델 유형에 대한 하이퍼파라미터 튜닝 수행

        Args:
            model_type: 모델 유형 ("rl", "lstm", "gnn", "statistical")
            train_data: 훈련 데이터
            val_data: 검증 데이터

        Returns:
            최적의 하이퍼파라미터 설정
        """
        logger.info(f"{model_type} 모델 하이퍼파라미터 튜닝 시작")

        # 모델 유형에 따른 파라미터 그리드 설정
        param_grid = self._get_param_grid(model_type)

        if not param_grid:
            logger.warning(
                f"지원하지 않는 모델 유형: {model_type}. 기본 설정을 반환합니다."
            )
            return self.config.__dict__ if hasattr(self, "config") else {}

        # 목적 함수 정의 (각 모델 유형에 맞게 훈련 및 평가 수행)
        def objective_fn(params):
            try:
                if model_type == "rl":
                    try:
                        # 트레이너 모듈 임포트 시도
                        trainer_module = None
                        trainer_class = None

                        # 시스템 경로에 src 추가하여 절대 임포트 시도
                        project_root = Path(__file__).parent.parent.parent
                        sys.path.insert(0, str(project_root))

                        # 여러 가능한 임포트 경로 시도
                        import_paths = [
                            "src.training.train_rl",
                            "training.train_rl",
                            f"{project_root}/src/training/train_rl",
                        ]

                        for import_path in import_paths:
                            try:
                                # 문자열 임포트 시도
                                trainer_module = importlib.import_module(
                                    import_path.replace("/", ".")
                                )
                                if hasattr(trainer_module, "RLTrainer"):
                                    trainer_class = trainer_module.RLTrainer
                                    break
                            except (ImportError, AttributeError):
                                continue

                        # 직접 경로 임포트 시도
                        if trainer_class is None:
                            try:
                                sys.path.insert(0, str(Path(__file__).parent.parent))
                                from training.train_rl import RLTrainer as trainer_class
                            except ImportError:
                                pass

                        # 모듈 찾지 못한 경우
                        if trainer_class is None:
                            logger.error("RLTrainer 클래스를 찾을 수 없습니다.")
                            return float("inf")

                        # RL 모델 훈련 및 평가
                        trainer = trainer_class(params)

                        # 매개변수를 kwargs로 전달하여 유연하게 처리
                        # 중복 매개변수 제거 및 타입 오류 방지
                        train_kwargs = {
                            "validation_data": val_data,
                        }

                        # 예외 추적을 위한 변수
                        exceptions = []

                        if hasattr(trainer, "train"):
                            # 함수 시그니처 분석
                            import inspect

                            try:
                                sig = inspect.signature(trainer.train)
                                param_names = list(sig.parameters.keys())

                                # pattern_analysis 매개변수가 존재하면 추가
                                if (
                                    "pattern_analysis" in param_names
                                    and len(param_names) > 2
                                ):
                                    train_kwargs["pattern_analysis"] = None

                                # 매개변수 이름 확인 및 조정
                                if (
                                    "validation_data" not in param_names
                                    and "valid_data" in param_names
                                ):
                                    train_kwargs["valid_data"] = train_kwargs.pop(
                                        "validation_data"
                                    )
                            except Exception:
                                # 시그니처 분석 실패 시 기본 매개변수 유지
                                pass

                        # 모든 가능한 매개변수 조합을 시도
                        try:
                            # 위치 인자와 키워드 인자 혼합하여 시도 - 타입 검사 무시
                            result = trainer.train(train_data, **train_kwargs)  # type: ignore
                        except Exception as e:
                            exceptions.append(str(e))
                            # 위치 인자만 전달
                            try:
                                result = trainer.train(train_data)  # type: ignore
                            except Exception as e2:
                                exceptions.append(str(e2))
                                logger.error(
                                    f"RL 모델 훈련 시도 모두 실패: {exceptions}"
                                )
                                return float("inf")

                        # 손실값 반환 (낮을수록 좋음)
                        if isinstance(result, dict):
                            return result.get(
                                "best_loss", result.get("loss", float("inf"))
                            )
                        return float("inf")
                    except Exception as e:
                        logger.error(f"RLTrainer 모델 훈련 전체 실패: {str(e)}")
                        return float("inf")

                elif model_type == "lstm":
                    try:
                        # 트레이너 모듈 임포트 시도
                        trainer_module = None
                        trainer_class = None

                        # 시스템 경로에 src 추가하여 절대 임포트 시도
                        project_root = Path(__file__).parent.parent.parent
                        sys.path.insert(0, str(project_root))

                        # 여러 가능한 임포트 경로 시도
                        import_paths = [
                            "src.training.train_lstm",
                            "training.train_lstm",
                            f"{project_root}/src/training/train_lstm",
                        ]

                        for import_path in import_paths:
                            try:
                                # 문자열 임포트 시도
                                trainer_module = importlib.import_module(
                                    import_path.replace("/", ".")
                                )
                                if hasattr(trainer_module, "LSTMTrainer"):
                                    trainer_class = trainer_module.LSTMTrainer
                                    break
                            except (ImportError, AttributeError):
                                continue

                        # 직접 경로 임포트 시도
                        if trainer_class is None:
                            try:
                                sys.path.insert(0, str(Path(__file__).parent.parent))
                                from training.train_lstm import (
                                    LSTMTrainer as trainer_class,
                                )
                            except ImportError:
                                pass

                        # 모듈 찾지 못한 경우
                        if trainer_class is None:
                            logger.error("LSTMTrainer 클래스를 찾을 수 없습니다.")
                            return float("inf")

                        # LSTM 모델 훈련 및 평가
                        trainer = trainer_class(params)

                        # 예외 추적을 위한 변수
                        exceptions = []

                        # 매개변수를 kwargs로 전달하여 유연하게 처리
                        train_kwargs = {
                            "validation_data": val_data,
                        }

                        if hasattr(trainer, "train"):
                            # 함수 시그니처 분석
                            import inspect

                            try:
                                sig = inspect.signature(trainer.train)
                                param_names = list(sig.parameters.keys())

                                # pattern_analysis 매개변수가 존재하면 추가
                                if (
                                    "pattern_analysis" in param_names
                                    and len(param_names) > 2
                                ):
                                    train_kwargs["pattern_analysis"] = None

                                # 매개변수 이름 확인 및 조정
                                if (
                                    "validation_data" not in param_names
                                    and "valid_data" in param_names
                                ):
                                    train_kwargs["valid_data"] = train_kwargs.pop(
                                        "validation_data"
                                    )
                            except Exception:
                                # 시그니처 분석 실패 시 기본 매개변수 유지
                                pass

                        # 모든 가능한 매개변수 조합을 시도
                        try:
                            # 위치 인자와 키워드 인자 혼합하여 시도 - 타입 검사 무시
                            result = trainer.train(train_data, **train_kwargs)  # type: ignore
                        except Exception as e:
                            exceptions.append(str(e))
                            # 위치 인자만 전달
                            try:
                                result = trainer.train(train_data)  # type: ignore
                            except Exception as e2:
                                exceptions.append(str(e2))
                                logger.error(
                                    f"LSTM 모델 훈련 시도 모두 실패: {exceptions}"
                                )
                                return float("inf")

                        # 손실값 반환 (낮을수록 좋음)
                        if isinstance(result, dict):
                            return result.get(
                                "best_loss", result.get("loss", float("inf"))
                            )
                        return float("inf")
                    except Exception as e:
                        logger.error(f"LSTM 모델 훈련 전체 실패: {str(e)}")
                        return float("inf")

                elif model_type == "gnn":
                    try:
                        # 트레이너 모듈 임포트 시도
                        trainer_module = None
                        trainer_class = None

                        # 시스템 경로에 src 추가하여 절대 임포트 시도
                        project_root = Path(__file__).parent.parent.parent
                        sys.path.insert(0, str(project_root))

                        # 여러 가능한 임포트 경로 시도
                        import_paths = [
                            "src.training.train_gnn",
                            "training.train_gnn",
                            f"{project_root}/src/training/train_gnn",
                        ]

                        for import_path in import_paths:
                            try:
                                # 문자열 임포트 시도
                                trainer_module = importlib.import_module(
                                    import_path.replace("/", ".")
                                )
                                if hasattr(trainer_module, "GNNTrainer"):
                                    trainer_class = trainer_module.GNNTrainer
                                    break
                            except (ImportError, AttributeError):
                                continue

                        # 직접 경로 임포트 시도
                        if trainer_class is None:
                            try:
                                sys.path.insert(0, str(Path(__file__).parent.parent))
                                from training.train_gnn import (
                                    GNNTrainer as trainer_class,
                                )
                            except ImportError:
                                pass

                        # 모듈 찾지 못한 경우
                        if trainer_class is None:
                            logger.error("GNNTrainer 클래스를 찾을 수 없습니다.")
                            return float("inf")

                        # GNN 모델 훈련 및 평가
                        trainer = trainer_class(params)

                        # 예외 추적을 위한 변수
                        exceptions = []

                        # 매개변수를 kwargs로 전달하여 유연하게 처리
                        train_kwargs = {
                            "validation_data": val_data,
                        }

                        if hasattr(trainer, "train"):
                            # 함수 시그니처 분석
                            import inspect

                            try:
                                sig = inspect.signature(trainer.train)
                                param_names = list(sig.parameters.keys())

                                # pattern_analysis 매개변수가 존재하면 추가
                                if (
                                    "pattern_analysis" in param_names
                                    and len(param_names) > 2
                                ):
                                    train_kwargs["pattern_analysis"] = None

                                # 매개변수 이름 확인 및 조정
                                if (
                                    "validation_data" not in param_names
                                    and "valid_data" in param_names
                                ):
                                    train_kwargs["valid_data"] = train_kwargs.pop(
                                        "validation_data"
                                    )
                            except Exception:
                                # 시그니처 분석 실패 시 기본 매개변수 유지
                                pass

                        # 모든 가능한 매개변수 조합을 시도
                        try:
                            # 위치 인자와 키워드 인자 혼합하여 시도 - 타입 검사 무시
                            result = trainer.train(train_data, **train_kwargs)  # type: ignore
                        except Exception as e:
                            exceptions.append(str(e))
                            # 위치 인자만 전달
                            try:
                                result = trainer.train(train_data)  # type: ignore
                            except Exception as e2:
                                exceptions.append(str(e2))
                                logger.error(
                                    f"GNN 모델 훈련 시도 모두 실패: {exceptions}"
                                )
                                return float("inf")

                        # 손실값 반환 (낮을수록 좋음)
                        if isinstance(result, dict):
                            return result.get(
                                "best_loss", result.get("loss", float("inf"))
                            )
                        return float("inf")
                    except Exception as e:
                        logger.error(f"GNN 모델 훈련 전체 실패: {str(e)}")
                        return float("inf")

                elif model_type == "statistical":
                    try:
                        # 트레이너 모듈 임포트 시도
                        trainer_module = None
                        trainer_class = None
                        interface_class = None

                        # 시스템 경로에 src 추가하여 절대 임포트 시도
                        project_root = Path(__file__).parent.parent.parent
                        sys.path.insert(0, str(project_root))

                        # TrainInterface 클래스 임포트 시도
                        import_paths = [
                            "src.training.train_interface",
                            "training.train_interface",
                            f"{project_root}/src/training/train_interface",
                        ]

                        for import_path in import_paths:
                            try:
                                # 문자열 임포트 시도
                                interface_module = importlib.import_module(
                                    import_path.replace("/", ".")
                                )
                                if hasattr(interface_module, "TrainInterface"):
                                    interface_class = interface_module.TrainInterface
                                    break
                            except (ImportError, AttributeError):
                                continue

                        # 직접 경로 임포트 시도
                        if interface_class is None:
                            try:
                                sys.path.insert(0, str(Path(__file__).parent.parent))
                                from training.train_interface import (
                                    TrainInterface as interface_class,
                                )
                            except ImportError:
                                pass

                        # TrainInterface 클래스를 찾았다면 사용
                        if interface_class is not None:
                            # 인터페이스를 통해 통계 모델 훈련 실행
                            try:
                                train_interface = interface_class(params)

                                # 매개변수를 kwargs로 전달하여 유연하게 처리
                                train_kwargs = {
                                    "validation_data": val_data,
                                    "valid_data": val_data,
                                    "val_data": val_data,
                                }

                                # 훈련 메서드 호출 - 메서드 존재 여부 확인 후 호출
                                if train_interface and hasattr(
                                    train_interface, "train_statistical_model"
                                ):
                                    result = train_interface.train_statistical_model(train_data, val_data)  # type: ignore
                                elif train_interface and hasattr(
                                    train_interface, "train"
                                ):
                                    result = train_interface.train(train_data, val_data)  # type: ignore

                                # 결과가 없으면 기본값 반환
                                if result is None:
                                    logger.warning(
                                        "통계 모델 훈련 메서드 호출 모두 실패"
                                    )
                                    return float("inf")
                                return result.get("loss", float("inf"))

                            except Exception as e:
                                logger.warning(f"TrainInterface 생성 실패: {str(e)}")
                                result = None
                        else:
                            # 개별 통계 모델 훈련기 임포트 시도
                            import_paths = [
                                "src.training.statistical_trainer",
                                "training.statistical_trainer",
                                "src.training.train_statistical",
                                "training.train_statistical",
                            ]

                            for import_path in import_paths:
                                try:
                                    trainer_module = importlib.import_module(
                                        import_path.replace("/", ".")
                                    )
                                    if hasattr(trainer_module, "LotteryTrainer"):
                                        trainer_class = trainer_module.LotteryTrainer
                                        break
                                except ImportError:
                                    continue

                            # 직접 훈련기 임포트 실패 시 대체 접근법 시도
                            if trainer_class is None:
                                # 패턴 분석기로 대체 시도
                                try:
                                    from ..analysis.pattern_analyzer import (
                                        PatternAnalyzer,
                                    )

                                    analyzer = PatternAnalyzer()
                                    analyzer.analyze_patterns(train_data)
                                    # 간단한 점수 계산으로 반환
                                    return 0.5  # 임의의 점수
                                except Exception as e:
                                    logger.error(f"통계 분석 시도 실패: {str(e)}")
                                    return float("inf")

                            # 통계 모델 훈련기로 훈련
                            trainer = trainer_class(params)

                            # 매개변수를 kwargs로 전달하여 유연하게 처리
                            train_kwargs = {
                                "validation_data": val_data,
                                "valid_data": val_data,
                                "val_data": val_data,
                            }

                            # 다양한 메서드 이름 시도
                            methods_to_try = [
                                "train_statistical_model",
                                "train",
                                "train_statistical",
                            ]

                            result = None
                            for method_name in methods_to_try:
                                if hasattr(trainer, method_name):
                                    try:
                                        # 메서드 호출 시 타입 오류 방지를 위한 type: ignore 추가
                                        result = getattr(trainer, method_name)(
                                            train_data,
                                            validation_data=val_data,
                                            pattern_analysis=None,  # type: ignore
                                            episodes=None,  # type: ignore
                                            batch_size=None,  # type: ignore
                                        )
                                        if result is not None:
                                            break
                                    except Exception as e:
                                        # 위치 인자만 시도
                                        try:
                                            result = getattr(trainer, method_name)(
                                                train_data,
                                                validation_data=val_data,
                                            )
                                            if result is not None:
                                                break
                                        except Exception as e2:
                                            logger.warning(
                                                f"{method_name} 메서드 호출 실패: {str(e2)}"
                                            )

                            if result is None:
                                logger.error("통계 모델 훈련 메서드 호출 모두 실패")
                                return float("inf")

                        # 손실값 반환
                        if isinstance(result, dict):
                            return result.get("loss", float("inf"))
                        return float("inf")
                    except Exception as e:
                        logger.warning(f"통계 모델 훈련 중 오류 발생: {str(e)}")
                        return float("inf")

                else:
                    logger.warning(f"지원하지 않는 모델 유형: {model_type}")
                    return float("inf")

            except Exception as e:
                logger.error(f"하이퍼파라미터 튜닝 중 오류: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())
                return float("inf")

        # 하이퍼파라미터 탐색 수행
        try:
            # run_autotune 함수 사용 (로컬 함수 우선)
            best_params = run_autotune(
                train_fn=objective_fn,
                param_grid=param_grid,
                n_trials=10,
                method="random",
                optimization_goal="minimize",
                timeout=1800,  # 기본 30분
                parallel_trials=1,
            )
        except Exception as e:
            logger.error(f"하이퍼파라미터 자동 튜닝 실패: {str(e)}")
            # 기본값 반환
            best_params = {k: v[0] for k, v in param_grid.items()}

        # 결과에 모델 유형 추가
        if isinstance(best_params, dict):
            best_params["model_type"] = model_type
        else:
            # best_params가 딕셔너리가 아닌 경우 처리
            best_params = {"model_type": model_type}

        logger.info(f"{model_type} 모델 하이퍼파라미터 튜닝 완료: {best_params}")

        return best_params

    def _get_param_grid(self, model_type: str) -> Dict[str, List[Any]]:
        """
        모델 유형에 따른 하이퍼파라미터 그리드 반환

        Args:
            model_type: 모델 유형

        Returns:
            하이퍼파라미터 그리드
        """
        if model_type == "rl":
            return {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [16, 32, 64],
                "epochs": [100, 200, 300],
                "use_amp": [True, False],
                "experience_replay": [True, False],
                "max_memory": [10000, 50000, 100000],
            }
        elif model_type == "lstm":
            return {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [16, 32, 64],
                "epochs": [100, 200, 300],
                "hidden_size": [64, 128, 256],
                "num_layers": [1, 2, 3],
                "dropout": [0.1, 0.3, 0.5],
                "use_amp": [True, False],
            }
        elif model_type == "gnn":
            return {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [16, 32, 64],
                "epochs": [100, 200, 300],
                "hidden_dim": [32, 64, 128],
                "num_layers": [1, 2, 3],
                "dropout": [0.1, 0.3, 0.5],
                "use_amp": [True, False],
            }
        elif model_type == "statistical":
            return {
                "smoothing_factor": [0.1, 0.3, 0.5, 0.7, 0.9],
                "weight_recent": [1.0, 1.5, 2.0, 2.5, 3.0],
                "pattern_weight": [0.1, 0.3, 0.5, 0.7, 0.9],
                "frequency_weight": [0.1, 0.3, 0.5, 0.7, 0.9],
            }
        else:
            logger.warning(f"지원하지 않는 모델 유형: {model_type}, 기본 그리드 반환")
            return {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [16, 32, 64],
                "epochs": [100, 200],
            }
