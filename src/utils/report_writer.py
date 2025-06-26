"""
학습 보고서 작성 유틸리티

학습 및 평가 메트릭을 타임스탬프가 있는 JSON 파일로 저장합니다.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import platform
import psutil
import torch
import numpy as np
import time

from .error_handler import get_logger
from .config_loader import load_config, ConfigProxy

logger = get_logger(__name__)

# 모듈에서 export할 항목들
__all__ = [
    "NumpyEncoder",
    "safe_convert",
    "write_performance_report",
    "write_performance_report_simple",
    "write_test_report",
    "read_performance_report",
    "list_performance_reports",
    "get_system_info",
    "write_training_report",
    "save_performance_report",
    "save_report",
    "validate_report_schema",
    "save_physical_performance_report",
]

# 보고서 저장 절대 경로 설정
REPORTS_DIR = Path("D:/VSworkSpace/DAEBAK_AI/lottery/logs/reports")


def validate_report_schema(report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    보고서 스키마를 검증합니다.

    Args:
        report: 검증할 보고서 데이터

    Returns:
        Tuple[bool, List[str]]: 검증 성공 여부와 누락된 키 목록
    """
    # 보고서 유형에 따른 필수 키 정의
    required_keys = {
        "data_analysis": ["vector_feature_stats", "cluster_embedding_quality"],
        "performance": ["execution_time_sec", "pattern_outlier_rate", "cache_hit_rate"],
        "training": ["model_name", "epochs", "learning_rate", "metrics"],
        "evaluation": ["metrics", "model_name", "timestamp"],
        "recommendation": ["recommended_sets", "execution_time_sec"],
    }

    # 보고서 유형 추론
    report_type = None
    if "vector_feature_stats" in report:
        report_type = "data_analysis"
    elif "model_name" in report and "epochs" in report:
        report_type = "training"
    elif "model_name" in report and "metrics" in report:
        report_type = "evaluation"
    elif "recommended_sets" in report:
        report_type = "recommendation"
    elif "execution_time_sec" in report:
        report_type = "performance"
    else:
        report_type = "unknown"

    # 알 수 없는 보고서 유형은 항상 유효하다고 간주
    if report_type == "unknown":
        logger.warning("알 수 없는 보고서 유형: 스키마 검증을 건너뜁니다.")
        return True, []

    # 필수 키 검증
    required = required_keys.get(report_type, [])
    missing_keys = [key for key in required if key not in report]

    # 결과 로깅
    if missing_keys:
        logger.warning(
            f"보고서({report_type})에 필수 키가 누락되었습니다: {', '.join(missing_keys)}"
        )
        return False, missing_keys

    logger.info(f"보고서({report_type}) 스키마 검증 성공")
    return True, []


def save_report(
    report_data: Dict[str, Any], report_name: str, subdir: Optional[str] = None
) -> str:
    """
    보고서를 저장하는 통합 함수입니다.

    Args:
        report_data: 보고서 데이터
        report_name: 보고서 이름 (파일명으로 사용)
        subdir: 저장할 하위 디렉토리 (None인 경우 REPORTS_DIR 직접 사용)

    Returns:
        저장된 보고서 파일 경로

    Raises:
        IOError: 보고서 저장 실패 시
    """
    try:
        # 보고서 스키마 검증
        is_valid, missing_keys = validate_report_schema(report_data)
        if not is_valid:
            logger.warning(
                f"보고서 스키마 검증 실패: 누락된 키 - {', '.join(missing_keys)}"
            )
            # 누락된 키에 대한 기본값 추가
            for key in missing_keys:
                report_data[key] = "N/A"
            logger.info("누락된 키에 기본값을 추가했습니다.")

        # 저장 디렉토리 설정
        if subdir:
            report_dir = REPORTS_DIR / subdir
        else:
            report_dir = REPORTS_DIR

        # 디렉토리 생성
        report_dir.mkdir(parents=True, exist_ok=True)

        # 타임스탬프 추가 (없는 경우)
        if "timestamp" not in report_data:
            report_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 파일명 생성
        filename = f"{report_name}.json"
        filepath = report_dir / filename

        # JSON 파일로 저장 (기존 파일은 덮어쓰기)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                safe_convert(report_data),
                f,
                indent=2,
                ensure_ascii=False,
                cls=NumpyEncoder,
            )

        logger.info(f"보고서 저장 완료: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"보고서 저장 실패: {str(e)}")
        raise IOError(f"보고서 저장 실패: {str(e)}")


# NumPy 타입을 JSON 직렬화 가능한 타입으로 변환하는 함수
def safe_convert(obj):
    """
    NumPy 타입을 포함한 객체를 JSON 직렬화 가능한 형태로 변환합니다.

    Args:
        obj: 변환할 객체

    Returns:
        JSON 직렬화 가능한 객체
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, dict):
        # 튜플 키를 문자열로 변환 (예: (1,2) -> "1_2")
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                # 튜플 키를 문자열로 변환
                new_key = "_".join(str(i) for i in k)
                new_dict[new_key] = safe_convert(v)
            else:
                new_dict[
                    str(k) if not isinstance(k, (str, int, float, bool)) else k
                ] = safe_convert(v)
        return new_dict
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [safe_convert(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    # 객체가 to_dict 메서드 가지고 있는 경우
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return safe_convert(obj.to_dict())
    # None, 기본 타입은 그대로 반환
    elif obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    # 그 외의 경우 문자열로 변환
    try:
        return str(obj)
    except Exception as e:
        logger.warning(f"직렬화할 수 없는 객체: {type(obj)}")
        return "UNKNOWN_OBJECT"


# numpy 타입을 처리할 수 있는 JSON 인코더
class NumpyEncoder(json.JSONEncoder):
    """NumPy 타입을 JSON으로 변환하는 인코더"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)


def write_performance_report(
    mode: str,
    model_weights: Dict[str, float],
    metrics: Dict[str, float],
    pattern_insights: Dict[str, Any],
    recommended_sets: List[List[int]],
    excluded_info: Dict[str, Any],
    logs: Dict[str, Any],
    config: ConfigProxy,
) -> str:
    """
    학습 성능 보고서를 작성합니다.

    Args:
        mode: 학습 모드 ("training", "evaluation", "prediction")
        model_weights: 모델 가중치
        metrics: 평가 지표
        pattern_insights: 패턴 분석 결과
        recommended_sets: 추천 번호 조합
        excluded_info: 제외된 정보
        logs: 로그 정보
        config: 설정

    Returns:
        저장된 보고서 파일 경로

    Raises:
        IOError: 보고서 저장 실패 시
    """
    # 보고서 데이터 구성
    report = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "mode": mode,
        "model_weights": model_weights,
        "metrics": metrics,
        "pattern_insights": pattern_insights,
        "recommended_sets": recommended_sets,
        "excluded_info": excluded_info,
        "logs": logs,
        "config_snapshot": config.to_dict(),
    }

    # 통합 함수 호출
    return save_report(report, f"{mode}_report")


def write_performance_report_simple(
    execution_mode: str,
    model_weights: Dict[str, float],
    metrics: Dict[str, Any],
    config: ConfigProxy,
) -> str:
    """
    간소화된 성능 보고서를 작성합니다.

    Args:
        execution_mode: 실행 모드 ("recommendation", "training", "backtesting")
        model_weights: 모델 가중치
        metrics: 성능 메트릭
        config: 설정

    Returns:
        저장된 보고서 파일 경로

    Raises:
        IOError: 보고서 저장 실패 시
    """
    # 시스템 정보 획득
    system_info = get_system_info()

    # 보고서 데이터 구성
    report = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "execution_mode": execution_mode,
        "model_weights": model_weights,
        "metrics": metrics,
        "system_info": system_info,
        "config_snapshot": config.to_dict(),
    }

    # 통합 함수 호출
    return save_report(report, f"performance_{execution_mode}", "performance")


def write_test_report(
    test_name: str,
    test_results: Dict[str, Any],
    verifications: Optional[Dict[str, bool]] = None,
    config: Optional[ConfigProxy] = None,
) -> Tuple[str, str]:
    """
    테스트 결과 보고서를 작성합니다.

    Args:
        test_name: 테스트 이름 ("data_analysis", "statistical_model" 등)
        test_results: 테스트 결과 데이터
        verifications: 검증 결과 (통과 여부)
        config: 설정 (None인 경우 load_config()로 로드)

    Returns:
        저장된 텍스트 및 JSON 파일 경로의 튜플

    Raises:
        IOError: 보고서 저장 실패 시
        KeyError: 필수 설정값이 없는 경우
    """
    # 설정 로드
    if config is None:
        config = load_config()

    # 검증 결과가 없는 경우 빈 딕셔너리 생성
    if verifications is None:
        verifications = {}

    # 통과 여부 판정 (모든 검증이 통과해야 PASS)
    verdict = "PASS" if all(verifications.values()) else "FAIL"

    try:
        # 저장 디렉토리 생성 (항상 고정 경로에 저장)
        test_results_dir = REPORTS_DIR / "test_results"
        test_results_dir.mkdir(parents=True, exist_ok=True)

        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 형식 보고서 구성
        report = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            **test_results,
            "verifications": verifications,
            "verdict": verdict,
        }

        # JSON 파일 경로 생성
        json_filename = f"test_{test_name}.json"
        json_filepath = test_results_dir / json_filename

        # JSON 파일로 저장 (통합 함수 호출)
        json_filepath_str = save_report(report, f"test_{test_name}", "test_results")
        json_filepath = Path(json_filepath_str)

        # 텍스트 파일 경로 생성
        txt_filename = f"{test_name}.txt"
        txt_filepath = test_results_dir / txt_filename

        # 텍스트 파일로 저장
        with open(txt_filepath, "w", encoding="utf-8") as f:
            f.write(f"===== {test_name.upper()} 테스트 결과 =====\n")
            f.write(f"타임스탬프: {timestamp}\n\n")

            # 번호별 출현 횟수 출력 (Top 10)
            if "number_frequencies" in test_results:
                f.write("[번호별 출현 횟수 (Top 10)]\n")
                for num, freq in test_results["number_frequencies"]:
                    f.write(f"{int(num)}번: {freq}회\n")
                f.write("\n")

            # 테스트 결과 출력
            for key, value in test_results.items():
                # number_frequencies와 failure_reasons는 별도로 처리
                if key in ["number_frequencies", "failure_reasons"]:
                    continue

                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for sub_key, sub_value in value.items():
                        f.write(f"  {sub_key}: {sub_value}\n")
                elif (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], (list, tuple))
                ):
                    f.write(f"{key}:\n")
                    for i, item in enumerate(value):
                        f.write(f"  {i+1}: {item}\n")
                else:
                    f.write(f"{key}: {value}\n")

            # 검증 결과 출력
            f.write("\n===== 검증 결과 =====\n")
            for check_name, passed in verifications.items():
                symbol = "✔" if passed else "✘"
                f.write(f"{symbol} {check_name}: {'통과' if passed else '실패'}\n")

            # 실패 이유 출력
            if "failure_reasons" in test_results and test_results["failure_reasons"]:
                f.write("\n===== 실패 이유 =====\n")
                for reason in test_results["failure_reasons"]:
                    f.write(f"- {reason}\n")

            # 최종 판정 출력
            f.write(f"\n최종 판정: {verdict}\n")

        logger.info(f"테스트 보고서 저장 완료: {json_filepath}, {txt_filepath}")
        return str(txt_filepath), str(json_filepath)

    except Exception as e:
        logger.error(f"테스트 보고서 작성 실패: {str(e)}")
        raise IOError(f"테스트 보고서 작성 실패: {str(e)}")


def read_performance_report(filename: str) -> Dict[str, Any]:
    """
    성능 보고서를 읽습니다.

    Args:
        filename: 읽을 파일명

    Returns:
        보고서 데이터

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        ValueError: 파일 형식이 올바르지 않을 때
    """
    # 절대 경로가 아니면 보고서 디렉토리에서 찾음
    filepath = Path(filename)
    if not filepath.is_absolute():
        filepath = REPORTS_DIR / filename

    if not filepath.exists():
        error_msg = f"보고서 파일이 존재하지 않습니다: {filepath}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            report = json.load(f)
        return report
    except json.JSONDecodeError as e:
        error_msg = f"보고서 파일 형식이 올바르지 않습니다: {filepath} - {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"보고서 파일 읽기 실패: {filepath} - {str(e)}"
        logger.error(error_msg)
        raise IOError(error_msg)


def list_performance_reports() -> List[str]:
    """
    모든 성능 보고서 파일 목록을 반환합니다.

    Returns:
        보고서 파일 경로 목록

    Raises:
        IOError: 디렉토리 접근 실패 시
    """
    try:
        # 디렉토리 생성 (없는 경우)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # 모든 JSON 파일 찾기
        json_files = list(REPORTS_DIR.glob("*.json"))

        # 하위 디렉토리 검색
        for subdir in REPORTS_DIR.iterdir():
            if subdir.is_dir():
                json_files.extend(list(subdir.glob("*.json")))

        # 파일 경로를 문자열로 변환
        return [
            str(f)
            for f in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)
        ]

    except Exception as e:
        logger.error(f"성능 보고서 목록 조회 실패: {str(e)}")
        raise IOError(f"성능 보고서 목록 조회 실패: {str(e)}")


def get_system_info() -> Dict[str, Any]:
    """
    시스템 정보를 수집합니다.

    Returns:
        시스템 정보 딕셔너리
    """
    info = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    # CUDA 정보 추가
    if torch.cuda.is_available():
        info["cuda"] = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
        }
    else:
        info["cuda"] = {"available": False}

    return info


def write_training_report(report_data: Dict[str, Any], report_name: str) -> str:
    """
    학습 보고서를 작성합니다.

    Args:
        report_data: 보고서 데이터
        report_name: 보고서 이름 (파일명의 일부로 사용)

    Returns:
        저장된 보고서 파일 경로

    Raises:
        IOError: 보고서 저장 실패 시
    """
    # 통합 함수 호출
    return save_report(report_data, report_name, "training")


def save_performance_report(report_data: Dict[str, Any], report_name: str) -> str:
    """
    성능 보고서를 저장합니다. write_performance_report_simple의 내부 구현용으로
    직접 호출하지 마세요.

    Args:
        report_data: 보고서 데이터
        report_name: 보고서 이름 (파일명으로 사용)

    Returns:
        저장된 보고서 파일 경로

    Raises:
        IOError: 보고서 저장 실패 시
    """
    return save_report(report_data, report_name, "performance")


def save_physical_performance_report(
    module_name: str,
    execution_times: Dict[str, float],
    cache_stats: Optional[Dict[str, Any]] = None,
    gpu_stats: Optional[Dict[str, Any]] = None,
    config: Optional[Union[Dict[str, Any], ConfigProxy]] = None,
) -> str:
    """
    물리적 성능 보고서를 생성하고 저장합니다.
    이 함수는 모든 모듈(데이터 분석, ML, DL, RL 등)에서 공통으로 사용되는
    통합 성능 보고서 형식을 제공합니다. 오직 하드웨어 및 실행 수준 지표만 포함됩니다.

    Args:
        module_name: 모듈 이름 (예: 'data_analysis', 'lightgbm_training', 'lstm_training')
        execution_times: 각 단계별 실행 시간 딕셔너리
        cache_stats: 캐시 통계 정보 (선택적)
        gpu_stats: GPU 통계 정보 (선택적)
        config: 설정 객체 (선택적)

    Returns:
        저장된 보고서 파일 경로

    Raises:
        IOError: 보고서 저장 실패 시
    """
    if config is None:
        # 설정이 제공되지 않은 경우 로드
        config = load_config()

    # ConfigProxy 객체를 딕셔너리로 변환
    config_dict = config.to_dict() if hasattr(config, "to_dict") else config

    # 보고서 저장 디렉토리 가져오기
    try:
        report_dir = Path(config_dict["paths"]["performance_report_dir"])
    except (KeyError, TypeError):
        logger.warning(
            "설정에서 'paths.performance_report_dir'를 찾을 수 없습니다. 기본값 'data/result/performance_reports'를 사용합니다."
        )
        report_dir = Path("data/result/performance_reports")

    # 디렉토리 생성
    report_dir.mkdir(parents=True, exist_ok=True)

    # 메모리 사용량 측정
    memory_info = psutil.virtual_memory()
    memory_used_mb = memory_info.used / (1024 * 1024)

    # 가능하면 프로세스별 메모리 측정
    process = psutil.Process(os.getpid())
    try:
        process_memory_info = process.memory_info()
        process_memory_mb = process_memory_info.rss / (1024 * 1024)
    except:
        process_memory_mb = 0

    # CPU 사용률 측정
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
    except:
        cpu_percent = 0

    # CUDA 정보 수집
    cuda_driver_version = None
    cudnn_version = None

    if torch.cuda.is_available():
        try:
            cuda_driver_version = torch.version.cuda
        except:
            pass

        # CUDNN 버전 정보
        try:
            cudnn_version = torch.backends.cudnn.version()
            if cudnn_version:
                cudnn_version = f"{cudnn_version // 1000}.{(cudnn_version % 1000) // 100}.{cudnn_version % 100}"
        except:
            pass

    # 총 실행 시간 계산 (없으면 sum)
    total_execution_time = execution_times.get("total", sum(execution_times.values()))

    # 기본 보고서 구조 생성 - 오직 하드웨어 및 실행 수준 지표만 포함
    report_data = {
        "hardware": "gpu" if torch.cuda.is_available() else "cpu",
        "gpu_device": (
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else None
        ),
        "torch_amp_enabled": config_dict.get("training", {}).get("use_amp", False),
        "parallel_execution": config_dict.get("execution", {}).get("parallel", False),
        "threading_backend": config_dict.get("execution", {}).get(
            "backend", "threading"
        ),
        "max_threads": config_dict.get("execution", {}).get(
            "max_threads", os.cpu_count()
        ),
        "batch_size": config_dict.get("training", {}).get("batch_size", None),
        "memory_usage": {
            "total_used_mb": round(memory_used_mb, 2),
            "peak_used_mb": round(process_memory_mb, 2),
        },
        "cache_hit_rate": None,
        "cache_memory_hit_count": None,
        "cache_memory_miss_count": None,
        "vector_processing_count": None,
        "execution_time_sec": round(total_execution_time, 2),
        "module_execution_times": {k: round(v, 2) for k, v in execution_times.items()},
        "cpu_usage_percent": round(cpu_percent, 1),
        "gpu_utilization_percent": None,
        "cuda_driver_version": cuda_driver_version,
        "cudnn_version": cudnn_version,
    }

    # 캐시 통계 정보 추가 (제공된 경우)
    if cache_stats:
        try:
            report_data["cache_hit_rate"] = round(cache_stats.get("hit_ratio", 0.0), 2)
            report_data["cache_memory_hit_count"] = cache_stats.get("hits", 0)
            report_data["cache_memory_miss_count"] = cache_stats.get("misses", 0)
            # vector_processing_count 계산
            total_cache_ops = cache_stats.get("hits", 0) + cache_stats.get("misses", 0)
            report_data["vector_processing_count"] = total_cache_ops
        except Exception as e:
            logger.warning(f"캐시 통계 정보 처리 중 오류 발생: {str(e)}")

    # GPU 통계 정보 추가 (제공된 경우)
    if gpu_stats:
        try:
            report_data["gpu_utilization_percent"] = round(
                gpu_stats.get("utilization", 0.0), 1
            )
            # 이미 설정되지 않은 경우에만 설정
            if not report_data["cuda_driver_version"]:
                report_data["cuda_driver_version"] = gpu_stats.get("driver_version")
            if not report_data["cudnn_version"]:
                report_data["cudnn_version"] = gpu_stats.get("cudnn_version")
        except Exception as e:
            logger.warning(f"GPU 통계 정보 처리 중 오류 발생: {str(e)}")

    # 파일명 생성
    filename = f"{module_name}_performance_report"

    # 보고서 저장
    filepath = report_dir / f"{filename}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            safe_convert(report_data),
            f,
            indent=2,
            ensure_ascii=False,
            cls=NumpyEncoder,
        )

    # 키 개수 계산
    valid_keys = sum(1 for k, v in report_data.items() if v is not None)
    logger.info(f"Physical report saved to: {filepath} with {valid_keys} keys")

    # 누락된 필수 필드 경고
    for key, value in report_data.items():
        if value is None:
            logger.warning(f"Missing physical report field: {key}")

    return str(filepath)
