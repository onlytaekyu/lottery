# 1. 표준 라이브러리
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# 2. 서드파티
import psutil
import torch

# 3. 프로젝트 내부 (없음)

__all__ = [
    "MaxPerformanceConfig",
    "PerformanceMonitor",
    "MaxPerformanceOptimizer",
]


@dataclass
class MaxPerformanceConfig:
    """최대 성능 설정 (적극적 접근)"""

    # CPU 최대 활용 (7800X3D: 8코어 16스레드)
    max_workers: int = 12  # 물리 코어 + 하이퍼스레딩 활용
    process_pool_size: int = 8  # 프로세스 풀
    thread_pool_size: int = 16  # 모든 논리 스레드 활용
    chunk_size_factor: float = 0.5  # L3 캐시 최적화 (96MB / 2)

    # GPU 최대 활용 (RTX 4090: 24GB VRAM)
    gpu_batch_size: int = 8192  # 대용량 배치
    gpu_memory_fraction: float = 0.9  # 90% GPU 메모리 활용 (22GB)
    gpu_streams: int = 8  # 다중 CUDA 스트림
    tensor_cores_enabled: bool = True  # Tensor Core 완전 활용

    # RAM 최대 활용 (32GB)
    memory_limit_gb: int = 24  # 24GB 활용
    cache_size_gb: int = 8  # 8GB 대용량 캐시
    buffer_size_gb: int = 4  # 4GB 버퍼
    memory_mapping_enabled: bool = True  # 메모리 매핑 활용

    # 모니터링 & 자동 조절 설정
    monitoring_enabled: bool = True
    auto_adjustment: bool = True
    adjustment_interval: float = 10.0  # 10초마다 체크

    # 임계값 (이 수치 초과 시 자동 다운그레이드)
    cpu_usage_threshold: float = 90.0  # CPU 90% 초과 시
    memory_usage_threshold: float = 85.0  # 메모리 85% 초과 시
    gpu_memory_threshold: float = 95.0  # GPU 메모리 95% 초과 시
    temperature_threshold: int = 80  # GPU 온도 80°C 초과 시


class PerformanceMonitor:
    """실시간 성능 모니터링 & 자동 조절"""

    def __init__(self, config: MaxPerformanceConfig):
        self.config = config
        self.monitoring: bool = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.adjustment_history: List[Dict[str, Any]] = []
        self.current_settings: Dict[str, Any] = asdict(config)
        self.lock = threading.RLock()

        # 조절 단계별 설정
        self.adjustment_levels: Dict[str, Dict[str, Any]] = {
            "level_0": {  # 최대 성능 (기본)
                "max_workers": 12,
                "gpu_batch_size": 8192,
                "memory_limit_gb": 24,
                "gpu_memory_fraction": 0.9,
            },
            "level_1": {  # 90% 성능
                "max_workers": 10,
                "gpu_batch_size": 6144,
                "memory_limit_gb": 20,
                "gpu_memory_fraction": 0.8,
            },
            "level_2": {  # 75% 성능
                "max_workers": 8,
                "gpu_batch_size": 4096,
                "memory_limit_gb": 16,
                "gpu_memory_fraction": 0.7,
            },
            "level_3": {  # 안전 모드 (50% 성능)
                "max_workers": 6,
                "gpu_batch_size": 2048,
                "memory_limit_gb": 12,
                "gpu_memory_fraction": 0.6,
            },
        }

        self.current_level: str = "level_0"

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------

    def start_monitoring(self) -> None:
        """모니터링 시작"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MaxPerformanceMonitor",
        )
        self.monitor_thread.start()
        print("🚀 최대 성능 모니터링 시작 - 실시간 자동 조절 활성화")

    def stop_monitoring(self) -> None:
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("📊 모니터링 중지")

    def get_current_settings(self) -> Dict[str, Any]:
        """현재 성능 설정 반환"""
        with self.lock:
            return {
                "level": self.current_level,
                "max_workers": self.current_settings["max_workers"],
                "gpu_batch_size": self.current_settings["gpu_batch_size"],
                "memory_limit_gb": self.current_settings["memory_limit_gb"],
                "gpu_memory_fraction": self.current_settings["gpu_memory_fraction"],
            }

    # ---------------------------------------------------------------------
    # monitor loop
    # ---------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        consecutive_high_usage = 0

        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                issues = self._detect_issues(metrics)

                if issues:
                    consecutive_high_usage += 1
                    print(
                        f"⚠️ 시스템 부하 감지: {', '.join(issues)} (연속 {consecutive_high_usage}회)"
                    )

                    # 3회 연속 문제 발생 시 자동 조절
                    if consecutive_high_usage >= 3 and self.config.auto_adjustment:
                        self._auto_adjust_down()
                        consecutive_high_usage = 0
                else:
                    consecutive_high_usage = 0
                    # 시스템이 안정적이면 점진적 성능 향상 시도
                    if consecutive_high_usage == 0:
                        self._try_performance_upgrade()

                self._log_metrics(metrics, issues)
                time.sleep(self.config.adjustment_interval)

            except Exception as exc:  # pylint: disable=broad-except
                print(f"❌ 모니터링 오류: {exc}")
                time.sleep(30)

    # ---------------------------------------------------------------------
    # metrics helpers
    # ---------------------------------------------------------------------

    def _collect_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # CPU 메트릭
        metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
        metrics["cpu_count"] = psutil.cpu_count()

        # 메모리 메트릭
        memory = psutil.virtual_memory()
        metrics["memory_percent"] = memory.percent
        metrics["memory_available_gb"] = memory.available / (1024**3)

        # GPU 메트릭 (가능한 경우)
        if torch.cuda.is_available():
            metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)
            metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )
            metrics["gpu_memory_percent"] = (
                metrics["gpu_memory_allocated"] / total_gpu_memory * 100.0
            )

            # GPU 온도
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                metrics["gpu_temperature"] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                metrics["gpu_utilization"] = pynvml.nvmlDeviceGetUtilizationRates(
                    handle
                ).gpu
            except Exception:  # pylint: disable=broad-except
                metrics["gpu_temperature"] = 0
                metrics["gpu_utilization"] = 0

        metrics["timestamp"] = time.time()
        return metrics

    def _detect_issues(self, metrics: Dict[str, float]) -> List[str]:
        issues: List[str] = []

        if metrics["cpu_percent"] > self.config.cpu_usage_threshold:
            issues.append(f"CPU 과부하 {metrics['cpu_percent']:.1f}%")

        if metrics["memory_percent"] > self.config.memory_usage_threshold:
            issues.append(f"메모리 부족 {metrics['memory_percent']:.1f}%")

        if metrics.get("gpu_memory_percent", 0) > self.config.gpu_memory_threshold:
            issues.append(f"GPU 메모리 부족 {metrics['gpu_memory_percent']:.1f}%")

        if metrics.get("gpu_temperature", 0) > self.config.temperature_threshold:
            issues.append(f"GPU 과열 {metrics['gpu_temperature']}°C")

        return issues

    # ---------------------------------------------------------------------
    # adjustment helpers
    # ---------------------------------------------------------------------

    def _auto_adjust_down(self) -> None:
        with self.lock:
            current_idx = list(self.adjustment_levels).index(self.current_level)
            if current_idx < len(self.adjustment_levels) - 1:
                new_level = list(self.adjustment_levels)[current_idx + 1]
                old_level = self.current_level
                self.current_level = new_level

                new_settings = self.adjustment_levels[new_level]
                self.current_settings.update(new_settings)

                adjustment = {
                    "timestamp": time.time(),
                    "action": "downgrade",
                    "from_level": old_level,
                    "to_level": new_level,
                    "settings": new_settings.copy(),
                }
                self.adjustment_history.append(adjustment)

                print(
                    f"🔽 성능 하향 조절: {old_level} → {new_level}\n"
                    f"   워커: {new_settings['max_workers']}, "
                    f"배치: {new_settings['gpu_batch_size']}"
                )

                self._save_current_settings()
            else:
                print("⚠️ 이미 최저 성능 단계입니다")

    def _try_performance_upgrade(self) -> None:
        with self.lock:
            recent_adjustments = [
                adj
                for adj in self.adjustment_history
                if time.time() - adj["timestamp"] < 1800
            ]
            if any(adj["action"] == "downgrade" for adj in recent_adjustments):
                return

            current_idx = list(self.adjustment_levels).index(self.current_level)
            if current_idx > 0:
                new_level = list(self.adjustment_levels)[current_idx - 1]
                old_level = self.current_level
                self.current_level = new_level

                new_settings = self.adjustment_levels[new_level]
                self.current_settings.update(new_settings)

                adjustment = {
                    "timestamp": time.time(),
                    "action": "upgrade",
                    "from_level": old_level,
                    "to_level": new_level,
                    "settings": new_settings.copy(),
                }
                self.adjustment_history.append(adjustment)

                print(
                    f"🔼 성능 상향 조절: {old_level} → {new_level}\n"
                    f"   워커: {new_settings['max_workers']}, "
                    f"배치: {new_settings['gpu_batch_size']}"
                )

                self._save_current_settings()

    # ---------------------------------------------------------------------
    # misc helpers
    # ---------------------------------------------------------------------

    def _log_metrics(self, metrics: Dict[str, float], issues: List[str]) -> None:
        if int(time.time()) % 600 == 0:  # 10분마다 출력
            print(
                f"📊 시스템 상태 - CPU: {metrics['cpu_percent']:.1f}%, "
                f"RAM: {metrics['memory_percent']:.1f}%, "
                f"GPU: {metrics.get('gpu_memory_percent', 0):.1f}%"
                f"{', 온도: ' + str(int(metrics.get('gpu_temperature', 0))) + '°C' if metrics.get('gpu_temperature', 0) > 0 else ''}"
            )

    def _save_current_settings(self) -> None:
        try:
            settings_file = Path("data/cache/current_performance_settings.json")
            settings_file.parent.mkdir(parents=True, exist_ok=True)

            save_data: Dict[str, Any] = {
                "current_level": self.current_level,
                "settings": self.current_settings,
                "last_updated": time.time(),
                "adjustment_history": self.adjustment_history[-10:],
            }
            settings_file.write_text(
                json.dumps(save_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ 설정 저장 실패: {exc}")


class MaxPerformanceOptimizer:
    """최대 성능 최적화 매니저"""

    def __init__(self) -> None:
        self.config = MaxPerformanceConfig()
        self.monitor = PerformanceMonitor(self.config)
        self.initialized = False

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """최대 성능 초기화"""
        try:
            print("🚀 최대 성능 최적화 시스템 초기화...")
            print("⚡ 7800X3D + RTX 4090 + 32GB RAM 풀파워 모드!")

            self._apply_max_performance_settings()

            if self.config.monitoring_enabled:
                self.monitor.start_monitoring()

            self.initialized = True
            self._print_max_performance_summary()

            print("✅ 최대 성능 시스템 가동 완료!")
            print("🔍 실시간 모니터링 중... 문제 발생 시 자동으로 조절됩니다.")
            return True
        except Exception as exc:  # pylint: disable=broad-except
            print(f"❌ 최대 성능 초기화 실패: {exc}")
            return False

    def get_current_performance_config(self) -> Dict[str, Any]:
        if not self.initialized:
            return {}

        settings = self.monitor.get_current_settings()
        settings.update(
            {
                "gpu_streams": self.config.gpu_streams,
                "memory_mapping": self.config.memory_mapping_enabled,
                "tensor_cores": self.config.tensor_cores_enabled,
                "monitoring": self.config.monitoring_enabled,
            }
        )
        return settings

    def force_performance_level(self, level: str) -> None:
        if level in self.monitor.adjustment_levels:
            self.monitor.current_level = level
            self.monitor.current_settings.update(self.monitor.adjustment_levels[level])
            print(f"🔧 성능 단계 강제 설정: {level}")
            self.monitor._save_current_settings()  # pylint: disable=protected-access
        else:
            print(f"❌ 잘못된 성능 단계: {level}")

    def get_performance_report(self) -> Dict[str, Any]:
        return {
            "current_level": self.monitor.current_level,
            "current_settings": self.monitor.get_current_settings(),
            "adjustment_history": self.monitor.adjustment_history[-5:],
            "available_levels": list(self.monitor.adjustment_levels.keys()),
            "monitoring_active": self.config.monitoring_enabled,
        }

    def cleanup(self) -> None:
        if self.config.monitoring_enabled:
            self.monitor.stop_monitoring()
        print("🚀 최대 성능 시스템 정리 완료")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _apply_max_performance_settings(self) -> None:
        os.environ["OMP_NUM_THREADS"] = str(self.config.thread_pool_size)
        os.environ["MKL_NUM_THREADS"] = str(self.config.thread_pool_size)
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.config.thread_pool_size)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            if self.config.tensor_cores_enabled:
                torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]

            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)

            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "max_split_size_mb:2048,"
                "expandable_segments:True,"
                "roundup_power2_divisions:8"
            )

            self._warmup_gpu()

    def _warmup_gpu(self) -> None:
        try:
            print("🔥 GPU 워밍업 중...")
            dummy = torch.randn(2048, 2048, device="cuda")
            for _ in range(5):
                _ = torch.mm(dummy, dummy)
            torch.cuda.synchronize()
            del dummy  # noqa: F821
            print("✅ GPU 워밍업 완료")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ GPU 워밍업 실패: {exc}")

    def _print_max_performance_summary(self) -> None:
        print("\n🚀 최대 성능 설정 적용:")
        print("┌─ CPU 최적화 (7800X3D)")
        print(f"├── 워커 수: {self.config.max_workers}개 (물리 8코어 + 하이퍼스레딩)")
        print(f"├── 프로세스 풀: {self.config.process_pool_size}개")
        print(f"├── 스레드 풀: {self.config.thread_pool_size}개")
        print("└── L3 캐시: 96MB 최적화")
        print("┌─ GPU 최적화 (RTX 4090)")
        print(f"├── 배치 크기: {self.config.gpu_batch_size:,}개 (4배 증가)")
        print(
            f"├── GPU 메모리: {self.config.gpu_memory_fraction*100:.0f}% 활용 (~22GB)"
        )
        print(f"├── CUDA 스트림: {self.config.gpu_streams}개")
        print(
            f"└── Tensor Core: {'활성화' if self.config.tensor_cores_enabled else '비활성화'}"
        )
        print("┌─ RAM 최적화 (32GB)")
        print(f"├── 메모리 한계: {self.config.memory_limit_gb}GB")
        print(f"├── 캐시 크기: {self.config.cache_size_gb}GB")
        print(f"├── 버퍼 크기: {self.config.buffer_size_gb}GB")
        print(
            f"└── 메모리 매핑: {'활성화' if self.config.memory_mapping_enabled else '비활성화'}"
        )
        print("└─ 모니터링: 실시간 자동 조절 활성화")


# =============================================================================
# 사용 예시 (직접 실행 시)
# =============================================================================


def launch_max_performance() -> Optional[MaxPerformanceOptimizer]:
    optimizer = MaxPerformanceOptimizer()
    if not optimizer.initialize():
        print("❌ 최대 성능 모드 시작 실패")
        return None

    print("\n🎯 추천 사용법:")
    print("1. 처음 10분간 시스템 상태 관찰")
    print("2. 문제 발생 시 자동으로 성능 조절됨")
    print("3. 수동 조절: optimizer.force_performance_level('level_1')")
    print("4. 상태 확인: optimizer.get_current_performance_config()")
    return optimizer


if __name__ == "__main__":
    OPTIMIZER = launch_max_performance()
    if OPTIMIZER:
        try:
            print("\n⏳ 30초간 시스템 모니터링...")
            time.sleep(30)
            current_cfg = OPTIMIZER.get_current_performance_config()
            print("\n📊 현재 성능 설정:")
            print(f"   성능 레벨: {current_cfg.get('level', 'unknown')}")
            print(f"   워커 수: {current_cfg.get('max_workers', 0)}개")
            print(f"   배치 크기: {current_cfg.get('gpu_batch_size', 0):,}개")
            print(f"   메모리 한계: {current_cfg.get('memory_limit_gb', 0)}GB")
        finally:
            OPTIMIZER.cleanup()
