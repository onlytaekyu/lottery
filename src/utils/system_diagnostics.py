"""
시스템 진단 및 모니터링 도구

이 모듈은 DAEBAK_AI 시스템의 전체적인 상태를 진단하고 모니터링합니다.
성능 문제, 구성 오류, 의존성 문제 등을 자동으로 감지합니다.
"""

# 1. 표준 라이브러리
import os
import sys
import json
import time
import importlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import traceback

# 2. 서드파티
import numpy as np
import pandas as pd
import psutil

# 3. 프로젝트 내부
from .unified_logging import get_logger
from .config_loader import load_config

logger = get_logger(__name__)


class SystemDiagnostics:
    """
    시스템 진단 클래스

    전체 시스템의 상태를 체크하고 문제점을 진단합니다.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        시스템 진단기 초기화

        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = None
        self.logger = get_logger(__name__)

        # 진단 결과 저장
        self.diagnostic_results = {}
        self.health_status = "unknown"

        self.logger.info("시스템 진단기 초기화 완료")

    def run_full_diagnostics(self) -> Dict[str, Any]:
        """
        전체 시스템 진단 실행

        Returns:
            진단 결과
        """
        self.logger.info("🔍 전체 시스템 진단 시작")
        start_time = time.time()

        diagnostic_results = {
            "timestamp": datetime.now().isoformat(),
            "diagnostics": {},
        }

        # 각 진단 항목 실행
        diagnostics = [
            ("environment", self._check_environment),
            ("dependencies", self._check_dependencies),
            ("configuration", self._check_configuration),
            ("file_structure", self._check_file_structure),
            ("model_availability", self._check_model_availability),
            ("data_integrity", self._check_data_integrity),
            ("performance", self._check_performance),
            ("memory_usage", self._check_memory_usage),
            ("disk_space", self._check_disk_space),
            ("import_health", self._check_import_health),
        ]

        for diagnostic_name, diagnostic_func in diagnostics:
            try:
                self.logger.info(f"진단 중: {diagnostic_name}")
                result = diagnostic_func()
                diagnostic_results["diagnostics"][diagnostic_name] = result

            except Exception as e:
                self.logger.error(f"{diagnostic_name} 진단 실패: {e}")
                diagnostic_results["diagnostics"][diagnostic_name] = {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

        # 전체 건강 상태 평가
        overall_health = self._assess_overall_health(diagnostic_results["diagnostics"])
        diagnostic_results["overall_health"] = overall_health

        # 권장사항 생성
        recommendations = self._generate_recommendations(
            diagnostic_results["diagnostics"]
        )
        diagnostic_results["recommendations"] = recommendations

        # 실행 시간
        diagnostic_results["execution_time"] = time.time() - start_time

        self.diagnostic_results = diagnostic_results
        self.health_status = overall_health["status"]

        self.logger.info(f"✅ 시스템 진단 완료: {overall_health['status']}")
        return diagnostic_results

    def _check_environment(self) -> Dict[str, Any]:
        """환경 설정 체크"""
        try:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

            # 필요한 환경 변수 체크
            required_env_vars = ["PYTHONPATH"]

            env_status = {}
            for var in required_env_vars:
                env_status[var] = {
                    "exists": var in os.environ,
                    "value": os.environ.get(var, "Not set"),
                }

            return {
                "status": "ok",
                "python_version": python_version,
                "platform": sys.platform,
                "environment_variables": env_status,
                "working_directory": os.getcwd(),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_dependencies(self) -> Dict[str, Any]:
        """의존성 패키지 체크"""
        try:
            required_packages = [
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "lightgbm",
                "xgboost",
                "catboost",
                "scipy",
                "psutil",
                "pyyaml",
            ]

            package_status = {}
            missing_packages = []

            for package in required_packages:
                try:
                    module = importlib.import_module(package)
                    version = getattr(module, "__version__", "unknown")
                    package_status[package] = {"available": True, "version": version}
                except ImportError:
                    package_status[package] = {"available": False, "version": None}
                    missing_packages.append(package)

            status = "ok" if not missing_packages else "warning"

            return {
                "status": status,
                "packages": package_status,
                "missing_packages": missing_packages,
                "total_checked": len(required_packages),
                "available_count": len(required_packages) - len(missing_packages),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_configuration(self) -> Dict[str, Any]:
        """설정 파일 체크"""
        try:
            config_checks = {
                "config_file_exists": os.path.exists(self.config_path),
                "config_readable": False,
                "required_sections": [],
                "missing_sections": [],
            }

            if config_checks["config_file_exists"]:
                try:
                    self.config = load_config(self.config_path)
                    config_checks["config_readable"] = True

                    # 필수 섹션 체크
                    required_sections = ["data", "models", "training", "evaluation"]

                    for section in required_sections:
                        if hasattr(self.config, section) or (
                            isinstance(self.config, dict) and section in self.config
                        ):
                            config_checks["required_sections"].append(section)
                        else:
                            config_checks["missing_sections"].append(section)

                except Exception as e:
                    config_checks["config_error"] = str(e)

            status = "ok"
            if not config_checks["config_file_exists"]:
                status = "error"
            elif not config_checks["config_readable"]:
                status = "error"
            elif config_checks["missing_sections"]:
                status = "warning"

            return {
                "status": status,
                "config_path": self.config_path,
                "checks": config_checks,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_file_structure(self) -> Dict[str, Any]:
        """파일 구조 체크"""
        try:
            required_directories = [
                "src/analysis",
                "src/models/ml",
                "src/models/dl",
                "src/models/meta",
                "src/core",
                "src/evaluation",
                "src/pipeline",
                "src/utils",
                "config",
                "data",
                "savedModels",
            ]

            directory_status = {}
            missing_directories = []

            for directory in required_directories:
                exists = os.path.exists(directory)
                directory_status[directory] = {
                    "exists": exists,
                    "is_directory": os.path.isdir(directory) if exists else False,
                }

                if not exists:
                    missing_directories.append(directory)

            # 중요 파일 체크
            important_files = [
                "src/core/recommendation_engine.py",
                "src/evaluation/backtester.py",
                "src/evaluation/diversity_evaluator.py",
                "src/models/meta/meta_learner_model.py",
            ]

            file_status = {}
            missing_files = []

            for file_path in important_files:
                exists = os.path.exists(file_path)
                file_status[file_path] = {
                    "exists": exists,
                    "size": os.path.getsize(file_path) if exists else 0,
                }

                if not exists:
                    missing_files.append(file_path)

            status = "ok"
            if missing_directories or missing_files:
                status = (
                    "warning"
                    if len(missing_directories) + len(missing_files) < 3
                    else "error"
                )

            return {
                "status": status,
                "directories": directory_status,
                "files": file_status,
                "missing_directories": missing_directories,
                "missing_files": missing_files,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_model_availability(self) -> Dict[str, Any]:
        """모델 가용성 체크"""
        try:
            model_files = [
                "savedModels/lightgbm_model.pkl",
                "savedModels/autoencoder_model.pt",
                "savedModels/tcn_model.pt",
                "savedModels/random_forest_model.pkl",
            ]

            model_status = {}
            available_models = []
            missing_models = []

            for model_file in model_files:
                exists = os.path.exists(model_file)
                model_status[model_file] = {
                    "exists": exists,
                    "size_mb": (
                        os.path.getsize(model_file) / 1024 / 1024 if exists else 0
                    ),
                    "last_modified": (
                        datetime.fromtimestamp(os.path.getmtime(model_file)).isoformat()
                        if exists
                        else None
                    ),
                }

                if exists:
                    available_models.append(model_file)
                else:
                    missing_models.append(model_file)

            status = "ok" if len(available_models) >= 2 else "warning"
            if not available_models:
                status = "error"

            return {
                "status": status,
                "models": model_status,
                "available_models": available_models,
                "missing_models": missing_models,
                "availability_ratio": len(available_models) / len(model_files),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_data_integrity(self) -> Dict[str, Any]:
        """데이터 무결성 체크"""
        try:
            data_files = [
                "data/cache/feature_vectors_full.npy",
                "data/lottery_data.csv",
            ]

            data_status = {}
            valid_files = []
            corrupted_files = []

            for data_file in data_files:
                if os.path.exists(data_file):
                    try:
                        # 파일 형식에 따른 검증
                        if data_file.endswith(".npy"):
                            data = np.load(data_file)
                            data_status[data_file] = {
                                "exists": True,
                                "valid": True,
                                "shape": data.shape,
                                "dtype": str(data.dtype),
                                "size_mb": os.path.getsize(data_file) / 1024 / 1024,
                            }
                        elif data_file.endswith(".csv"):
                            data = pd.read_csv(data_file, nrows=5)  # 일부만 읽어서 검증
                            data_status[data_file] = {
                                "exists": True,
                                "valid": True,
                                "columns": len(data.columns),
                                "sample_rows": len(data),
                                "size_mb": os.path.getsize(data_file) / 1024 / 1024,
                            }

                        valid_files.append(data_file)

                    except Exception as e:
                        data_status[data_file] = {
                            "exists": True,
                            "valid": False,
                            "error": str(e),
                        }
                        corrupted_files.append(data_file)
                else:
                    data_status[data_file] = {"exists": False, "valid": False}

            status = "ok" if len(valid_files) > 0 and not corrupted_files else "warning"

            return {
                "status": status,
                "data_files": data_status,
                "valid_files": valid_files,
                "corrupted_files": corrupted_files,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_performance(self) -> Dict[str, Any]:
        """성능 체크"""
        try:
            # CPU 성능 테스트
            start_time = time.time()
            test_array = np.random.rand(1000, 1000)
            np.dot(test_array, test_array.T)
            cpu_test_time = time.time() - start_time

            # 메모리 성능 테스트
            start_time = time.time()
            large_array = np.random.rand(5000, 1000)
            del large_array
            memory_test_time = time.time() - start_time

            # 시스템 리소스
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            performance_metrics = {
                "cpu_test_time": cpu_test_time,
                "memory_test_time": memory_test_time,
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "available_memory_gb": memory.available / 1024 / 1024 / 1024,
                "cpu_count": psutil.cpu_count(),
            }

            # 성능 상태 평가
            status = "ok"
            if cpu_percent > 80 or memory.percent > 85:
                status = "warning"
            if cpu_percent > 95 or memory.percent > 95:
                status = "critical"

            return {
                "status": status,
                "metrics": performance_metrics,
                "performance_grade": self._calculate_performance_grade(
                    performance_metrics
                ),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 체크"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()

            memory_status = {
                "process_memory_mb": memory_info.rss / 1024 / 1024,
                "process_memory_percent": (memory_info.rss / system_memory.total) * 100,
                "system_memory_total_gb": system_memory.total / 1024 / 1024 / 1024,
                "system_memory_available_gb": system_memory.available
                / 1024
                / 1024
                / 1024,
                "system_memory_percent": system_memory.percent,
            }

            status = "ok"
            if memory_status["system_memory_percent"] > 80:
                status = "warning"
            if memory_status["system_memory_percent"] > 90:
                status = "critical"

            return {"status": status, "memory_status": memory_status}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_disk_space(self) -> Dict[str, Any]:
        """디스크 공간 체크"""
        try:
            disk_usage = psutil.disk_usage(".")

            disk_status = {
                "total_gb": disk_usage.total / 1024 / 1024 / 1024,
                "used_gb": disk_usage.used / 1024 / 1024 / 1024,
                "free_gb": disk_usage.free / 1024 / 1024 / 1024,
                "usage_percent": (disk_usage.used / disk_usage.total) * 100,
            }

            status = "ok"
            if disk_status["usage_percent"] > 80:
                status = "warning"
            if disk_status["usage_percent"] > 90:
                status = "critical"

            return {"status": status, "disk_status": disk_status}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_import_health(self) -> Dict[str, Any]:
        """임포트 상태 체크"""
        try:
            critical_modules = [
                "src.analysis.pattern_analyzer",
                "src.models.ml.lightgbm_model",
                "src.models.dl.autoencoder_model",
                "src.core.recommendation_engine",
                "src.evaluation.backtester",
            ]

            import_status = {}
            failed_imports = []

            for module_name in critical_modules:
                try:
                    importlib.import_module(module_name)
                    import_status[module_name] = {"importable": True, "error": None}
                except Exception as e:
                    import_status[module_name] = {"importable": False, "error": str(e)}
                    failed_imports.append(module_name)

            status = "ok" if not failed_imports else "error"

            return {
                "status": status,
                "imports": import_status,
                "failed_imports": failed_imports,
                "success_rate": (len(critical_modules) - len(failed_imports))
                / len(critical_modules),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _calculate_performance_grade(self, metrics: Dict[str, Any]) -> str:
        """성능 등급 계산"""
        try:
            cpu_score = 100 - metrics["cpu_usage_percent"]
            memory_score = 100 - metrics["memory_usage_percent"]
            speed_score = max(0, 100 - (metrics["cpu_test_time"] * 100))

            overall_score = (cpu_score + memory_score + speed_score) / 3

            if overall_score >= 80:
                return "A"
            elif overall_score >= 60:
                return "B"
            elif overall_score >= 40:
                return "C"
            else:
                return "D"

        except Exception:
            return "Unknown"

    def _assess_overall_health(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """전체 건강 상태 평가"""
        try:
            status_counts = {"ok": 0, "warning": 0, "error": 0, "critical": 0}

            for diagnostic_name, diagnostic_result in diagnostics.items():
                status = diagnostic_result.get("status", "error")
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    status_counts["error"] += 1

            total_checks = sum(status_counts.values())

            # 전체 상태 결정
            if status_counts["critical"] > 0:
                overall_status = "critical"
            elif status_counts["error"] > 0:
                overall_status = "error"
            elif status_counts["warning"] > total_checks * 0.3:
                overall_status = "warning"
            else:
                overall_status = "ok"

            health_score = (
                (
                    (
                        status_counts["ok"] * 100
                        + status_counts["warning"] * 70
                        + status_counts["error"] * 30
                        + status_counts["critical"] * 0
                    )
                    / total_checks
                )
                if total_checks > 0
                else 0
            )

            return {
                "status": overall_status,
                "health_score": health_score,
                "status_counts": status_counts,
                "total_checks": total_checks,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _generate_recommendations(self, diagnostics: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        try:
            # 의존성 문제
            if diagnostics.get("dependencies", {}).get("status") != "ok":
                missing = diagnostics["dependencies"].get("missing_packages", [])
                if missing:
                    recommendations.append(f"누락된 패키지 설치: {', '.join(missing)}")

            # 설정 문제
            if diagnostics.get("configuration", {}).get("status") != "ok":
                recommendations.append("설정 파일을 확인하고 수정하세요")

            # 파일 구조 문제
            file_struct = diagnostics.get("file_structure", {})
            if file_struct.get("status") != "ok":
                missing_dirs = file_struct.get("missing_directories", [])
                missing_files = file_struct.get("missing_files", [])
                if missing_dirs:
                    recommendations.append(
                        f"누락된 디렉토리 생성: {', '.join(missing_dirs)}"
                    )
                if missing_files:
                    recommendations.append(
                        f"누락된 파일 확인: {', '.join(missing_files)}"
                    )

            # 성능 문제
            performance = diagnostics.get("performance", {})
            if performance.get("status") in ["warning", "critical"]:
                metrics = performance.get("metrics", {})
                if metrics.get("cpu_usage_percent", 0) > 80:
                    recommendations.append(
                        "CPU 사용량이 높습니다. 실행 중인 프로세스를 확인하세요"
                    )
                if metrics.get("memory_usage_percent", 0) > 80:
                    recommendations.append(
                        "메모리 사용량이 높습니다. 메모리 정리를 실행하세요"
                    )

            # 메모리 문제
            memory = diagnostics.get("memory_usage", {})
            if memory.get("status") in ["warning", "critical"]:
                recommendations.append("메모리 최적화를 실행하세요")

            # 디스크 공간 문제
            disk = diagnostics.get("disk_space", {})
            if disk.get("status") in ["warning", "critical"]:
                recommendations.append(
                    "디스크 공간을 확보하세요. 불필요한 파일을 삭제하세요"
                )

            # 임포트 문제
            imports = diagnostics.get("import_health", {})
            if imports.get("status") != "ok":
                failed = imports.get("failed_imports", [])
                if failed:
                    recommendations.append(
                        f"임포트 실패 모듈 확인: {', '.join(failed)}"
                    )

            if not recommendations:
                recommendations.append("시스템이 정상 상태입니다")

        except Exception as e:
            recommendations.append(f"권장사항 생성 중 오류: {e}")

        return recommendations

    def save_diagnostic_report(self, output_dir: str = "data/diagnostics") -> str:
        """진단 보고서 저장"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = output_path / f"system_diagnostics_{timestamp}.json"

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.diagnostic_results,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            self.logger.info(f"진단 보고서 저장: {report_file}")
            return str(report_file)

        except Exception as e:
            self.logger.error(f"진단 보고서 저장 실패: {e}")
            return ""

    def get_health_summary(self) -> Dict[str, Any]:
        """건강 상태 요약"""
        if not self.diagnostic_results:
            return {"status": "not_diagnosed", "message": "진단이 실행되지 않았습니다"}

        overall_health = self.diagnostic_results.get("overall_health", {})
        recommendations = self.diagnostic_results.get("recommendations", [])

        return {
            "status": overall_health.get("status", "unknown"),
            "health_score": overall_health.get("health_score", 0),
            "total_checks": overall_health.get("total_checks", 0),
            "critical_issues": len(
                [r for r in recommendations if "critical" in r.lower()]
            ),
            "recommendations_count": len(recommendations),
            "last_check": self.diagnostic_results.get("timestamp"),
            "execution_time": self.diagnostic_results.get("execution_time", 0),
        }


# 편의 함수들
def run_system_diagnostics(config_path: Optional[str] = None) -> Dict[str, Any]:
    """시스템 진단 실행"""
    diagnostics = SystemDiagnostics(config_path)
    return diagnostics.run_full_diagnostics()


def get_system_health_summary(config_path: Optional[str] = None) -> Dict[str, Any]:
    """시스템 건강 상태 요약"""
    diagnostics = SystemDiagnostics(config_path)
    diagnostics.run_full_diagnostics()
    return diagnostics.get_health_summary()
