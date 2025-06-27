"""
통합 보고서 작성 시스템

모든 종류의 보고서(성능, 학습, 분석, 평가)를 통합 관리합니다.
"""

import json
import platform
import psutil
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

from .unified_logging import get_logger

logger = get_logger(__name__)


class UnifiedReportWriter:
    """통합 보고서 작성기"""

    def __init__(self, report_dir: str = "data/result/performance_reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_report(
        self, data: Dict[str, Any], name: str, category: str = "general"
    ) -> str:
        """통합 보고서 저장"""
        # 보고서 데이터 향상
        enhanced_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "system_info": self._get_system_info(),
            },
            "data": self._safe_convert(data),
        }

        # 파일명 생성 (날짜 제외)
        filename = f"{category}_{name}.json"
        filepath = self.report_dir / filename

        # JSON 저장
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                enhanced_data, f, indent=2, ensure_ascii=False, cls=self.NumpyEncoder
            )

        logger.info(f"보고서 저장: {filepath}")
        return str(filepath)

    def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        info = {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_device_name": torch.cuda.get_device_name(),
                    "cuda_memory_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                    ),
                }
            )

        return info

    def _safe_convert(self, obj):
        """NumPy 타입을 JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): self._safe_convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._safe_convert(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj

    class NumpyEncoder(json.JSONEncoder):
        """NumPy 타입 JSON 인코더"""

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)


# 전역 인스턴스
_report_writer = UnifiedReportWriter()


# 편의 함수들
def save_report(data: Dict[str, Any], name: str, category: str = "general") -> str:
    """보고서 저장 (전역 함수)"""
    return _report_writer.save_report(data, name, category)


def save_performance_report(data: Dict[str, Any], name: str) -> str:
    """성능 보고서 저장"""
    return save_report(data, name, "performance")


def save_analysis_report(data: Dict[str, Any], name: str) -> str:
    """분석 보고서 저장"""
    return save_report(data, name, "analysis")


def save_training_report(data: Dict[str, Any], name: str) -> str:
    """학습 보고서 저장"""
    return save_report(data, name, "training")


def save_evaluation_report(data: Dict[str, Any], name: str) -> str:
    """평가 보고서 저장"""
    return save_report(data, name, "evaluation")


# 기존 호환성을 위한 함수들
def save_analysis_performance_report(
    profiler,
    performance_tracker,
    config,
    module_name: str = "data_analysis",
    data_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """분석 성능 보고서 저장 (기존 호환성)"""
    # 기본 데이터 수집
    report_data = {
        "module_name": module_name,
        "execution_time_sec": 0.0,
        "memory_usage_mb": 0.0,
        "cpu_usage_percent": psutil.cpu_percent(),
        "timestamp": datetime.now().isoformat(),
        # 필수 필드들
        "hardware": "gpu" if torch.cuda.is_available() else "cpu",
        "gpu_device": (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "none"
        ),
        "parallel_execution": True,
        "max_threads": psutil.cpu_count(),
        "batch_size": 64,
        "memory_usage": psutil.virtual_memory().percent,
        "cache_hit_rate": 0.0,
        "vector_processing_count": 0,
        "module_execution_times": {},
        "gpu_utilization_percent": 0.0,
        "torch_amp_enabled": torch.cuda.is_available(),
        "threading_backend": "threading",
        "cache_memory_hit_count": 0,
        "cache_memory_miss_count": 0,
        "cuda_driver_version": (
            torch.version.cuda if torch.cuda.is_available() else "none"
        ),
        "cudnn_version": (
            str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "none"
        ),
    }

    # 프로파일러 데이터 추가
    if hasattr(profiler, "get_profile_stats"):
        report_data["profiler_stats"] = profiler.get_profile_stats()
    elif hasattr(profiler, "sessions"):
        report_data["profiler_stats"] = profiler.sessions

    # 성능 추적기 데이터 추가
    if hasattr(performance_tracker, "get_summary"):
        if hasattr(performance_tracker, "sessions"):
            # UnifiedPerformanceTracker
            for name, session in performance_tracker.sessions.items():
                durations = session.get("durations", [])
                if durations:
                    report_data["execution_time_sec"] = max(
                        report_data["execution_time_sec"], max(durations)
                    )
        else:
            # 기존 PerformanceTracker
            summary = performance_tracker.get_summary()
            if isinstance(summary, dict) and "max_time" in summary:
                report_data["execution_time_sec"] = summary["max_time"]

    # 데이터 메트릭 추가
    if data_metrics:
        report_data["data_metrics"] = data_metrics

    return save_performance_report(report_data, f"{module_name}_performance_report")


# 기존 함수명 호환성
write_performance_report = save_performance_report
write_training_report = save_training_report
save_physical_performance_report = save_performance_report


# 호환성을 위한 safe_convert 함수
def safe_convert(obj):
    """NumPy 타입을 JSON 직렬화 가능한 형태로 변환 (전역 함수)"""
    return _report_writer._safe_convert(obj)


# 시스템 정보 함수
def get_system_info() -> Dict[str, Any]:
    """시스템 정보 반환"""
    return _report_writer._get_system_info()


# export할 항목들
__all__ = [
    "UnifiedReportWriter",
    "save_report",
    "save_performance_report",
    "save_analysis_report",
    "save_training_report",
    "save_evaluation_report",
    "save_analysis_performance_report",
    "get_system_info",
    "safe_convert",
    # 기존 호환성
    "write_performance_report",
    "write_training_report",
    "save_physical_performance_report",
]
