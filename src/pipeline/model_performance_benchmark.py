"""
Model Performance Benchmark System
모델 성능 벤치마킹 시스템 - A/B 테스트, 성능 추적, 종합 분석
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import time
import psutil
import threading
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from datetime import datetime
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from ..utils.unified_logging import get_logger
from ..utils.unified_config import load_config
from ..shared.types import ModelPerformanceMetrics

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    processing_time: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    throughput: float = 0.0  # samples/second

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""

    model_name: str
    dataset_name: str
    timestamp: str
    metrics: PerformanceMetrics
    configuration: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "metrics": self.metrics.to_dict(),
            "configuration": self.configuration,
        }


class SystemResourceMonitor:
    """시스템 리소스 모니터"""

    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 0.1):
        """모니터링 시작"""
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []

        self.monitor_thread = threading.Thread(
            target=self._monitor_resources, args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """모니터링 중지 및 결과 반환"""
        self.monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        return {
            "avg_cpu_usage": np.mean(self.cpu_usage) if self.cpu_usage else 0.0,
            "max_cpu_usage": np.max(self.cpu_usage) if self.cpu_usage else 0.0,
            "avg_memory_usage": (
                np.mean(self.memory_usage) if self.memory_usage else 0.0
            ),
            "max_memory_usage": np.max(self.memory_usage) if self.memory_usage else 0.0,
            "avg_gpu_usage": np.mean(self.gpu_usage) if self.gpu_usage else 0.0,
            "max_gpu_usage": np.max(self.gpu_usage) if self.gpu_usage else 0.0,
        }

    def _monitor_resources(self, interval: float):
        """리소스 모니터링 루프"""
        while self.monitoring:
            try:
                # CPU 사용률
                cpu_percent = psutil.cpu_percent()
                self.cpu_usage.append(cpu_percent)

                # 메모리 사용률
                memory_info = psutil.virtual_memory()
                self.memory_usage.append(memory_info.percent)

                # GPU 사용률 (NVIDIA GPU인 경우)
                try:
                    import GPUtil

                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_load = gpus[0].load * 100
                        self.gpu_usage.append(gpu_load)
                except ImportError:
                    self.gpu_usage.append(0.0)

                time.sleep(interval)

            except Exception as e:
                logger.warning(f"리소스 모니터링 중 오류: {e}")
                time.sleep(interval)


class ABTester:
    """A/B 테스터"""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.test_results = []

    def run_ab_test(
        self,
        model_a: Any,
        model_b: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> Dict[str, Any]:
        """
        A/B 테스트 실행

        Args:
            model_a: 모델 A
            model_b: 모델 B
            X_test: 테스트 데이터
            y_test: 테스트 라벨
            model_a_name: 모델 A 이름
            model_b_name: 모델 B 이름

        Returns:
            Dict: A/B 테스트 결과
        """

        logger.info(f"A/B 테스트 시작: {model_a_name} vs {model_b_name}")

        # 모델 A 성능 측정
        metrics_a = self._measure_model_performance(model_a, X_test, y_test)

        # 모델 B 성능 측정
        metrics_b = self._measure_model_performance(model_b, X_test, y_test)

        # 통계적 유의성 검정
        significance_test = self._statistical_significance_test(
            metrics_a, metrics_b, len(X_test)
        )

        # 결과 정리
        ab_result = {
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "model_a_metrics": metrics_a.to_dict(),
            "model_b_metrics": metrics_b.to_dict(),
            "winner": self._determine_winner(metrics_a, metrics_b),
            "performance_improvement": self._calculate_improvement(
                metrics_a, metrics_b
            ),
            "statistical_significance": significance_test,
            "recommendation": self._generate_recommendation(
                metrics_a, metrics_b, significance_test
            ),
        }

        self.test_results.append(ab_result)
        logger.info(f"A/B 테스트 완료: 승자 = {ab_result['winner']}")

        return ab_result

    def _measure_model_performance(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> PerformanceMetrics:
        """모델 성능 측정"""

        # 시스템 리소스 모니터링 시작
        monitor = SystemResourceMonitor()
        monitor.start_monitoring()

        # 성능 측정 시작
        start_time = time.time()

        try:
            # 예측 수행
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X)
                y_pred_proba = None

            # 처리 시간 계산
            processing_time = time.time() - start_time

            # 리소스 사용량 측정
            resource_usage = monitor.stop_monitoring()

            # 성능 메트릭 계산
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

            # AUC 계산 (이진 분류 또는 확률 예측이 가능한 경우)
            auc = 0.0
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y)) == 2:
                        auc = roc_auc_score(y, y_pred_proba[:, 1])
                    else:
                        auc = roc_auc_score(
                            y, y_pred_proba, multi_class="ovr", average="weighted"
                        )
                except Exception:
                    auc = 0.0

            # 처리량 계산
            throughput = len(X) / processing_time if processing_time > 0 else 0

            metrics = PerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_score=auc,
                processing_time=processing_time,
                memory_usage=resource_usage["max_memory_usage"],
                gpu_utilization=resource_usage["max_gpu_usage"],
                throughput=throughput,
            )

        except Exception as e:
            logger.error(f"성능 측정 중 오류: {e}")
            monitor.stop_monitoring()
            metrics = PerformanceMetrics()

        return metrics

    def _statistical_significance_test(
        self,
        metrics_a: PerformanceMetrics,
        metrics_b: PerformanceMetrics,
        sample_size: int,
    ) -> Dict[str, Any]:
        """통계적 유의성 검정"""

        # 간단한 z-test 근사 (정확한 검정을 위해서는 더 정교한 방법 필요)
        accuracy_diff = metrics_b.accuracy - metrics_a.accuracy

        # 표준 오차 추정
        p_pooled = (metrics_a.accuracy + metrics_b.accuracy) / 2
        se = np.sqrt(2 * p_pooled * (1 - p_pooled) / sample_size)

        # Z 점수 계산
        z_score = accuracy_diff / se if se > 0 else 0

        # P-value 근사 (양측 검정)
        from scipy import stats

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        is_significant = p_value < (1 - self.confidence_level)

        return {
            "z_score": z_score,
            "p_value": p_value,
            "is_significant": is_significant,
            "confidence_level": self.confidence_level,
        }

    def _determine_winner(
        self, metrics_a: PerformanceMetrics, metrics_b: PerformanceMetrics
    ) -> str:
        """승자 결정"""

        # 종합 점수 계산 (가중 평균)
        score_a = (
            0.4 * metrics_a.accuracy
            + 0.2 * metrics_a.f1_score
            + 0.2 * (1 / (1 + metrics_a.processing_time))  # 처리 시간 역수
            + 0.2 * (metrics_a.throughput / 1000)  # 처리량 정규화
        )

        score_b = (
            0.4 * metrics_b.accuracy
            + 0.2 * metrics_b.f1_score
            + 0.2 * (1 / (1 + metrics_b.processing_time))
            + 0.2 * (metrics_b.throughput / 1000)
        )

        if score_b > score_a:
            return "Model B"
        elif score_a > score_b:
            return "Model A"
        else:
            return "Tie"

    def _calculate_improvement(
        self, metrics_a: PerformanceMetrics, metrics_b: PerformanceMetrics
    ) -> Dict[str, float]:
        """성능 개선률 계산"""

        def safe_improvement(old_val, new_val):
            if old_val == 0:
                return 0.0
            return ((new_val - old_val) / old_val) * 100

        return {
            "accuracy_improvement": safe_improvement(
                metrics_a.accuracy, metrics_b.accuracy
            ),
            "f1_improvement": safe_improvement(metrics_a.f1_score, metrics_b.f1_score),
            "speed_improvement": safe_improvement(
                1 / metrics_a.processing_time, 1 / metrics_b.processing_time
            ),
            "throughput_improvement": safe_improvement(
                metrics_a.throughput, metrics_b.throughput
            ),
        }

    def _generate_recommendation(
        self,
        metrics_a: PerformanceMetrics,
        metrics_b: PerformanceMetrics,
        significance_test: Dict[str, Any],
    ) -> str:
        """권장사항 생성"""

        if not significance_test["is_significant"]:
            return "통계적으로 유의한 차이가 없습니다. 추가 데이터로 재테스트를 권장합니다."

        winner = self._determine_winner(metrics_a, metrics_b)

        if winner == "Model B":
            return "Model B가 더 우수한 성능을 보입니다. Model B 사용을 권장합니다."
        elif winner == "Model A":
            return "Model A가 더 우수한 성능을 보입니다. Model A 사용을 권장합니다."
        else:
            return "두 모델의 성능이 비슷합니다. 다른 요소(메모리 사용량, 처리 시간 등)를 고려하여 선택하세요."


class ModelPerformanceBenchmark:
    """모델 성능 벤치마킹 시스템"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config if config else load_config()
        self.output_dir = Path(
            self.config.get_nested("data.result.performance_reports", "data/result/performance_reports")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

        # 벤치마크 결과 저장
        self.benchmark_results = []
        self.comparison_results = []

        # A/B 테스터 초기화
        self.ab_tester = ABTester(confidence_level=self.config.get("confidence_level", 0.95))

        # 성능 임계값 설정
        self.performance_thresholds = {
            "accuracy_threshold": self.config.get("accuracy_threshold", 0.7),
            "processing_time_threshold": self.config.get("processing_time_threshold", 30.0),
            "memory_threshold": self.config.get("memory_threshold", 80.0),
            "throughput_threshold": self.config.get("throughput_threshold", 100.0),
        }

    def benchmark_preprocessing(
        self, X: np.ndarray, y: np.ndarray, model_type: str, preprocessor: Any
    ) -> Dict[str, Any]:
        """
        전처리 성능 벤치마크

        Args:
            X: 입력 데이터
            y: 타겟 데이터
            model_type: 모델 타입
            preprocessor: 전처리기

        Returns:
            Dict: 벤치마크 결과
        """

        self.logger.info(f"{model_type} 전처리 벤치마크 시작")

        # 시스템 리소스 모니터링
        monitor = SystemResourceMonitor()
        monitor.start_monitoring()

        # 처리 시간 측정
        start_time = time.time()

        try:
            # 전처리 수행
            if (
                hasattr(preprocessor, "preprocess_for_lightgbm")
                and model_type == "lightgbm"
            ):
                X_processed, metadata = preprocessor.preprocess_for_lightgbm(X, y)
            elif (
                hasattr(preprocessor, "preprocess_for_autoencoder")
                and model_type == "autoencoder"
            ):
                X_processed, metadata = preprocessor.preprocess_for_autoencoder(X, y)
            elif hasattr(preprocessor, "preprocess_for_tcn") and model_type == "tcn":
                X_processed, metadata = preprocessor.preprocess_for_tcn(X, y)
            else:
                X_processed = X
                metadata = {}

            processing_time = time.time() - start_time

            # 리소스 사용량 측정
            resource_usage = monitor.stop_monitoring()

            # 성능 메트릭 계산
            throughput = len(X) / processing_time if processing_time > 0 else 0

            benchmark_result = {
                "model_type": model_type,
                "processing_speed": throughput,
                "memory_usage": resource_usage["max_memory_usage"],
                "gpu_utilization": resource_usage["max_gpu_usage"],
                "processing_time": processing_time,
                "input_shape": X.shape,
                "output_shape": (
                    X_processed.shape if hasattr(X_processed, "shape") else None
                ),
                "metadata": metadata,
                "performance_gain": self._calculate_preprocessing_gain(
                    X, X_processed, metadata
                ),
            }

            self.logger.info(f"{model_type} 전처리 벤치마크 완료")

        except Exception as e:
            self.logger.error(f"전처리 벤치마크 중 오류: {e}")
            monitor.stop_monitoring()
            benchmark_result = {
                "model_type": model_type,
                "error": str(e),
                "processing_speed": 0,
                "memory_usage": 0,
                "gpu_utilization": 0,
                "processing_time": 0,
            }

        return benchmark_result

    def _calculate_preprocessing_gain(
        self, X_original: np.ndarray, X_processed: Any, metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """전처리 성능 향상 계산"""

        gain = {}

        # 차원 축소 효과
        if hasattr(X_processed, "shape"):
            if len(X_original.shape) == len(X_processed.shape):
                compression_ratio = X_original.shape[1] / X_processed.shape[1]
                gain["compression_ratio"] = compression_ratio
            else:
                gain["compression_ratio"] = 1.0

        # 메타데이터에서 성능 향상 정보 추출
        if "compression_ratio" in metadata:
            gain["metadata_compression"] = metadata["compression_ratio"]

        if "reconstruction_loss" in metadata:
            gain["reconstruction_quality"] = 1.0 / (
                1.0 + metadata["reconstruction_loss"]
            )

        return gain

    def run_comprehensive_benchmark(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        cross_validation: bool = True,
    ) -> Dict[str, Any]:
        """
        종합 벤치마크 실행

        Args:
            models: 모델 딕셔너리
            X_test: 테스트 데이터
            y_test: 테스트 라벨
            cross_validation: 교차 검증 여부

        Returns:
            Dict: 종합 벤치마크 결과
        """

        self.logger.info("종합 벤치마크 시작")

        benchmark_results = {}

        # 각 모델별 벤치마크
        for model_name, model in models.items():
            self.logger.info(f"{model_name} 벤치마크 실행")

            # 단일 모델 벤치마크
            model_result = self._benchmark_single_model(
                model, X_test, y_test, model_name
            )

            # 교차 검증 (옵션)
            if cross_validation:
                cv_result = self._cross_validation_benchmark(model, X_test, y_test)
                model_result["cross_validation"] = cv_result

            benchmark_results[model_name] = model_result

        # 모델 간 비교
        comparison_result = self._compare_models(benchmark_results)

        # 종합 결과
        comprehensive_result = {
            "individual_results": benchmark_results,
            "comparison": comparison_result,
            "recommendations": self._generate_model_recommendations(benchmark_results),
            "performance_summary": self._generate_performance_summary(
                benchmark_results
            ),
        }

        self.logger.info("종합 벤치마크 완료")

        return comprehensive_result

    def _benchmark_single_model(
        self, model: Any, X: np.ndarray, y: np.ndarray, model_name: str
    ) -> Dict[str, Any]:
        """단일 모델 벤치마크"""

        # 성능 측정
        metrics = self.ab_tester._measure_model_performance(model, X, y)

        # 성능 임계값 검사
        performance_check = self._check_performance_thresholds(metrics)

        # 벤치마크 결과 생성
        result = BenchmarkResult(
            model_name=model_name,
            dataset_name="test_dataset",
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            configuration={},
        )

        self.benchmark_results.append(result)

        return {
            "metrics": metrics.to_dict(),
            "performance_check": performance_check,
            "benchmark_result": result.to_dict(),
        }

    def _cross_validation_benchmark(
        self, model: Any, X: np.ndarray, y: np.ndarray, cv_folds: int = 5
    ) -> Dict[str, Any]:
        """교차 검증 벤치마크"""

        try:
            # 교차 검증 수행
            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring="accuracy",
            )

            cv_result = {
                "cv_scores": cv_scores.tolist(),
                "mean_score": np.mean(cv_scores),
                "std_score": np.std(cv_scores),
                "confidence_interval": [
                    np.mean(cv_scores) - 1.96 * np.std(cv_scores),
                    np.mean(cv_scores) + 1.96 * np.std(cv_scores),
                ],
            }

        except Exception as e:
            self.logger.warning(f"교차 검증 중 오류: {e}")
            cv_result = {
                "error": str(e),
                "cv_scores": [],
                "mean_score": 0.0,
                "std_score": 0.0,
            }

        return cv_result

    def _check_performance_thresholds(
        self, metrics: PerformanceMetrics
    ) -> Dict[str, bool]:
        """성능 임계값 검사"""

        return {
            "accuracy_pass": metrics.accuracy
            >= self.performance_thresholds["accuracy_threshold"],
            "speed_pass": metrics.processing_time
            <= self.performance_thresholds["processing_time_threshold"],
            "memory_pass": metrics.memory_usage
            <= self.performance_thresholds["memory_threshold"],
            "throughput_pass": metrics.throughput
            >= self.performance_thresholds["throughput_threshold"],
        }

    def _compare_models(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """모델 간 비교"""

        comparison = {
            "best_accuracy": "",
            "best_speed": "",
            "best_memory_efficiency": "",
            "best_overall": "",
        }

        best_accuracy = 0
        best_speed = float("inf")
        best_memory = float("inf")
        best_overall_score = 0

        for model_name, result in benchmark_results.items():
            metrics = result["metrics"]

            # 최고 정확도
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                comparison["best_accuracy"] = model_name

            # 최고 속도
            if metrics["processing_time"] < best_speed:
                best_speed = metrics["processing_time"]
                comparison["best_speed"] = model_name

            # 최고 메모리 효율성
            if metrics["memory_usage"] < best_memory:
                best_memory = metrics["memory_usage"]
                comparison["best_memory_efficiency"] = model_name

            # 종합 점수
            overall_score = (
                0.4 * metrics["accuracy"]
                + 0.2 * metrics["f1_score"]
                + 0.2 * (1 / (1 + metrics["processing_time"]))
                + 0.2 * (metrics["throughput"] / 1000)
            )

            if overall_score > best_overall_score:
                best_overall_score = overall_score
                comparison["best_overall"] = model_name

        return comparison

    def _generate_model_recommendations(
        self, benchmark_results: Dict[str, Any]
    ) -> List[str]:
        """모델 권장사항 생성"""

        recommendations = []

        for model_name, result in benchmark_results.items():
            performance_check = result["performance_check"]

            if not performance_check["accuracy_pass"]:
                recommendations.append(
                    f"{model_name}: 정확도 개선 필요 (현재: {result['metrics']['accuracy']:.3f})"
                )

            if not performance_check["speed_pass"]:
                recommendations.append(
                    f"{model_name}: 처리 속도 개선 필요 (현재: {result['metrics']['processing_time']:.2f}초)"
                )

            if not performance_check["memory_pass"]:
                recommendations.append(
                    f"{model_name}: 메모리 사용량 최적화 필요 (현재: {result['metrics']['memory_usage']:.1f}%)"
                )

            if not performance_check["throughput_pass"]:
                recommendations.append(
                    f"{model_name}: 처리량 개선 필요 (현재: {result['metrics']['throughput']:.1f} samples/sec)"
                )

        if not recommendations:
            recommendations.append("모든 모델이 성능 임계값을 만족합니다.")

        return recommendations

    def _generate_performance_summary(
        self, benchmark_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """성능 요약 생성"""

        all_accuracies = [
            result["metrics"]["accuracy"] for result in benchmark_results.values()
        ]
        all_times = [
            result["metrics"]["processing_time"]
            for result in benchmark_results.values()
        ]
        all_memories = [
            result["metrics"]["memory_usage"] for result in benchmark_results.values()
        ]

        return {
            "average_accuracy": np.mean(all_accuracies),
            "accuracy_std": np.std(all_accuracies),
            "average_processing_time": np.mean(all_times),
            "time_std": np.std(all_times),
            "average_memory_usage": np.mean(all_memories),
            "memory_std": np.std(all_memories),
            "total_models_tested": len(benchmark_results),
        }

    def save_benchmark_results(self, filepath: str):
        """벤치마크 결과 저장"""

        results_data = {
            "benchmark_results": [
                result.to_dict() for result in self.benchmark_results
            ],
            "ab_test_results": self.ab_tester.test_results,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"벤치마크 결과 저장 완료: {filepath}")

    def load_benchmark_results(self, filepath: str):
        """벤치마크 결과 로드"""

        with open(filepath, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        # 벤치마크 결과 복원
        self.benchmark_results = []
        for result_dict in results_data["benchmark_results"]:
            metrics = PerformanceMetrics(**result_dict["metrics"])
            result = BenchmarkResult(
                model_name=result_dict["model_name"],
                dataset_name=result_dict["dataset_name"],
                timestamp=result_dict["timestamp"],
                metrics=metrics,
                configuration=result_dict["configuration"],
            )
            self.benchmark_results.append(result)

        # A/B 테스트 결과 복원
        self.ab_tester.test_results = results_data["ab_test_results"]

        self.logger.info(f"벤치마크 결과 로드 완료: {filepath}")

    def print_benchmark_summary(self, comprehensive_result: Dict[str, Any]):
        """벤치마크 요약 출력"""

        print("=" * 80)
        print("📊 모델 성능 벤치마크 결과")
        print("=" * 80)

        # 개별 모델 결과
        print("\n🔍 개별 모델 성능:")
        for model_name, result in comprehensive_result["individual_results"].items():
            metrics = result["metrics"]
            print(f"\n  📈 {model_name}:")
            print(f"    • 정확도: {metrics['accuracy']:.4f}")
            print(f"    • F1 점수: {metrics['f1_score']:.4f}")
            print(f"    • 처리 시간: {metrics['processing_time']:.2f}초")
            print(f"    • 메모리 사용량: {metrics['memory_usage']:.1f}%")
            print(f"    • 처리량: {metrics['throughput']:.1f} samples/sec")

        # 모델 비교
        print(f"\n🏆 모델 비교:")
        comparison = comprehensive_result["comparison"]
        print(f"  • 최고 정확도: {comparison['best_accuracy']}")
        print(f"  • 최고 속도: {comparison['best_speed']}")
        print(f"  • 최고 메모리 효율성: {comparison['best_memory_efficiency']}")
        print(f"  • 종합 최고 성능: {comparison['best_overall']}")

        # 성능 요약
        print(f"\n📊 성능 요약:")
        summary = comprehensive_result["performance_summary"]
        print(
            f"  • 평균 정확도: {summary['average_accuracy']:.4f} (±{summary['accuracy_std']:.4f})"
        )
        print(
            f"  • 평균 처리 시간: {summary['average_processing_time']:.2f}초 (±{summary['time_std']:.2f})"
        )
        print(
            f"  • 평균 메모리 사용량: {summary['average_memory_usage']:.1f}% (±{summary['memory_std']:.1f})"
        )

        # 권장사항
        print(f"\n💡 권장사항:")
        for i, recommendation in enumerate(comprehensive_result["recommendations"], 1):
            print(f"  {i}. {recommendation}")

        print("=" * 80)

    def run_all_benchmarks(self) -> Dict[str, ModelPerformanceMetrics]:
        """모든 모델에 대한 벤치마크를 실행합니다."""
        # ... existing code ...
