"""
통합 시스템 테스트 스크립트
모든 새로운 시스템들의 통합 테스트
"""

import asyncio
import torch
import numpy as np
import time
import threading
from typing import Dict, Any, List

from .unified_logging import get_logger
from .unified_config import get_config
from .compute_strategy import (
    get_compute_executor,
    get_optimal_compute_selector,
    smart_execute,
    TaskType,
    ComputeStrategy,
)
from .unified_memory_manager import (
    get_unified_memory_manager,
    smart_allocate,
    get_memory_status,
    DeviceType,
)
from .unified_async_manager import (
    get_unified_async_manager,
    async_read_file,
    async_write_file,
    async_cache_get,
    async_cache_set,
)
from .auto_recovery_system import (
    get_auto_recovery_system,
    auto_recoverable,
    start_auto_recovery,
    stop_auto_recovery,
    RecoveryStrategy,
)
from .dependency_container import get_container, register, resolve

logger = get_logger(__name__)


class IntegratedSystemTester:
    """통합 시스템 테스터"""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logger

    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """모든 테스트 실행"""
        self.logger.info("🚀 통합 시스템 테스트 시작")

        # 1. GPU 연산 우선순위 시스템 테스트
        self.test_compute_strategy_system()

        # 2. 통합 메모리 관리 시스템 테스트
        self.test_unified_memory_system()

        # 3. 비동기 처리 시스템 테스트
        asyncio.run(self.test_async_system())

        # 4. 자동 복구 시스템 테스트
        self.test_auto_recovery_system()

        # 5. 의존성 주입 시스템 테스트
        self.test_dependency_injection_system()

        # 6. 통합 성능 테스트
        self.test_integrated_performance()

        self.logger.info("✅ 통합 시스템 테스트 완료")
        return self.test_results

    def test_compute_strategy_system(self):
        """GPU 연산 우선순위 시스템 테스트"""
        self.logger.info("🔧 GPU 연산 우선순위 시스템 테스트")

        test_result = {
            "gpu_available": torch.cuda.is_available(),
            "strategy_selection": {},
            "smart_execution": {},
            "performance": {},
        }

        try:
            # 1. 전략 선택 테스트
            selector = get_optimal_compute_selector()

            # 작은 데이터 (CPU 선택 예상)
            small_data = np.random.rand(100)
            small_size = small_data.nbytes

            # 큰 데이터 (GPU 선택 예상)
            large_data = np.random.rand(10000, 100)
            large_size = large_data.nbytes

            test_result["strategy_selection"] = {
                "small_data_size": small_size,
                "large_data_size": large_size,
                "gpu_available": torch.cuda.is_available(),
            }

            # 2. 스마트 실행 테스트
            def test_function(data):
                return np.sum(data)

            start_time = time.time()
            result_small = smart_execute(
                test_function, small_data, TaskType.DATA_PROCESSING
            )
            small_time = time.time() - start_time

            start_time = time.time()
            result_large = smart_execute(
                test_function, large_data, TaskType.DATA_PROCESSING
            )
            large_time = time.time() - start_time

            test_result["smart_execution"] = {
                "small_data_result": float(result_small),
                "large_data_result": float(result_large),
                "small_data_time": small_time,
                "large_data_time": large_time,
            }

            # 3. 성능 테스트
            executor = get_compute_executor()

            # 텐서 연산 테스트
            if torch.cuda.is_available():
                tensor_data = torch.randn(1000, 1000)

                def tensor_operation(data):
                    return torch.mm(data, data.T)

                start_time = time.time()
                tensor_result = executor.execute(
                    tensor_operation, tensor_data, task_type=TaskType.TENSOR_COMPUTATION
                )
                tensor_time = time.time() - start_time

                test_result["performance"]["tensor_operation_time"] = tensor_time
                test_result["performance"]["tensor_result_shape"] = list(
                    tensor_result.shape
                )

            test_result["status"] = "success"

        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
            self.logger.error(f"GPU 연산 우선순위 시스템 테스트 실패: {e}")

        self.test_results["compute_strategy"] = test_result
        self.logger.info(
            f"✅ GPU 연산 우선순위 시스템 테스트 완료: {test_result['status']}"
        )

    def test_unified_memory_system(self):
        """통합 메모리 관리 시스템 테스트"""
        self.logger.info("🧠 통합 메모리 관리 시스템 테스트")

        test_result = {
            "memory_allocation": {},
            "memory_status": {},
            "memory_cleanup": {},
        }

        try:
            memory_manager = get_unified_memory_manager()

            # 1. 메모리 할당 테스트
            tensor1, alloc_id1 = smart_allocate(1000, torch.float32, DeviceType.CPU)
            tensor2, alloc_id2 = smart_allocate(2000, torch.float32, DeviceType.GPU)

            test_result["memory_allocation"] = {
                "cpu_allocation": {
                    "id": alloc_id1,
                    "size": tensor1.numel(),
                    "device": str(tensor1.device),
                },
                "gpu_allocation": {
                    "id": alloc_id2,
                    "size": tensor2.numel() if tensor2 is not None else 0,
                    "device": str(tensor2.device) if tensor2 is not None else "none",
                },
            }

            # 2. 메모리 상태 확인
            memory_status = get_memory_status()
            test_result["memory_status"] = memory_status

            # 3. 메모리 해제 테스트
            cleanup_success1 = memory_manager.deallocate(alloc_id1)
            cleanup_success2 = memory_manager.deallocate(alloc_id2)

            test_result["memory_cleanup"] = {
                "cpu_cleanup": cleanup_success1,
                "gpu_cleanup": cleanup_success2,
            }

            # 4. 임시 메모리 할당 테스트
            with memory_manager.temporary_allocation(500, torch.float32) as temp_tensor:
                test_result["temporary_allocation"] = {
                    "size": temp_tensor.numel(),
                    "device": str(temp_tensor.device),
                }

            test_result["status"] = "success"

        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
            self.logger.error(f"통합 메모리 관리 시스템 테스트 실패: {e}")

        self.test_results["unified_memory"] = test_result
        self.logger.info(
            f"✅ 통합 메모리 관리 시스템 테스트 완료: {test_result['status']}"
        )

    async def test_async_system(self):
        """비동기 처리 시스템 테스트"""
        self.logger.info("⚡ 비동기 처리 시스템 테스트")

        test_result = {
            "file_operations": {},
            "cache_operations": {},
            "task_management": {},
        }

        try:
            async_manager = get_unified_async_manager()

            # 1. 비동기 파일 작업 테스트
            test_content = "비동기 파일 테스트 내용"
            test_file = "test_async_file.txt"

            # 파일 쓰기
            await async_write_file(test_file, test_content)

            # 파일 읽기
            read_content = await async_read_file(test_file)

            test_result["file_operations"] = {
                "write_success": True,
                "read_success": read_content == test_content,
                "content_match": read_content == test_content,
            }

            # 2. 비동기 캐시 작업 테스트
            cache_key = "test_cache_key"
            cache_value = {"data": "test_value", "timestamp": time.time()}

            # 캐시 저장
            await async_cache_set(cache_key, cache_value)

            # 캐시 조회
            cached_data = await async_cache_get(cache_key)

            test_result["cache_operations"] = {
                "cache_set_success": True,
                "cache_get_success": cached_data is not None,
                "cache_data_match": (
                    cached_data == cache_value if cached_data else False
                ),
            }

            # 3. 비동기 작업 관리 테스트
            async def test_async_task():
                await asyncio.sleep(0.1)
                return "async_task_result"

            # 세션 컨텍스트 내에서 작업 실행
            async with async_manager.session():
                task_id = await async_manager.submit_task(test_async_task())
                # 작업 완료 대기 (간단한 구현)
                await asyncio.sleep(0.2)

            test_result["task_management"] = {
                "task_submission": task_id is not None,
                "task_id": task_id,
            }

            # 파일 정리
            try:
                import os

                os.remove(test_file)
            except:
                pass

            test_result["status"] = "success"

        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
            self.logger.error(f"비동기 처리 시스템 테스트 실패: {e}")

        self.test_results["async_system"] = test_result
        self.logger.info(f"✅ 비동기 처리 시스템 테스트 완료: {test_result['status']}")

    def test_auto_recovery_system(self):
        """자동 복구 시스템 테스트"""
        self.logger.info("🔄 자동 복구 시스템 테스트")

        test_result = {"system_start": {}, "error_handling": {}, "recovery_stats": {}}

        try:
            recovery_system = get_auto_recovery_system()

            # 1. 시스템 시작 테스트
            start_auto_recovery()
            test_result["system_start"]["started"] = True

            # 2. 에러 처리 테스트
            test_errors = [
                RuntimeError("CUDA out of memory"),
                MemoryError("Memory allocation failed"),
                TimeoutError("Operation timed out"),
            ]

            recovery_results = []
            for error in test_errors:
                recovery_context = recovery_system.handle_error(error, {"test": True})
                recovery_results.append(
                    {
                        "error_type": recovery_context.error_type,
                        "strategy": recovery_context.strategy.value,
                        "success": recovery_context.success,
                        "recovery_time": recovery_context.recovery_time,
                        "attempted_actions": [
                            action.value
                            for action in recovery_context.attempted_actions
                        ],
                    }
                )

            test_result["error_handling"] = recovery_results

            # 3. 복구 통계 테스트
            recovery_stats = recovery_system.get_recovery_stats()
            test_result["recovery_stats"] = recovery_stats

            # 4. 자동 복구 데코레이터 테스트
            @auto_recoverable(strategy=RecoveryStrategy.GPU_OOM)
            def test_recoverable_function():
                # 의도적으로 예외 발생
                if torch.cuda.is_available():
                    # GPU 메모리 부족 시뮬레이션
                    large_tensor = torch.randn(10000, 10000, device="cuda")
                    return large_tensor.sum()
                else:
                    return 42

            try:
                result = test_recoverable_function()
                test_result["decorator_test"] = {
                    "success": True,
                    "result": float(result),
                }
            except Exception as e:
                test_result["decorator_test"] = {"success": False, "error": str(e)}

            # 시스템 정지
            stop_auto_recovery()

            test_result["status"] = "success"

        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
            self.logger.error(f"자동 복구 시스템 테스트 실패: {e}")

        self.test_results["auto_recovery"] = test_result
        self.logger.info(f"✅ 자동 복구 시스템 테스트 완료: {test_result['status']}")

    def test_dependency_injection_system(self):
        """의존성 주입 시스템 테스트"""
        self.logger.info("🔗 의존성 주입 시스템 테스트")

        test_result = {
            "container_operations": {},
            "dependency_resolution": {},
            "circular_dependency_detection": {},
        }

        try:
            container = get_container()

            # 1. 컨테이너 기본 동작 테스트
            class TestService:
                def __init__(self):
                    self.value = "test_service"

                def get_value(self):
                    return self.value

            # 서비스 등록
            register(TestService)

            # 서비스 해결
            service = resolve(TestService)

            test_result["container_operations"] = {
                "registration": True,
                "resolution": service is not None,
                "service_value": service.get_value() if service else None,
            }

            # 2. 의존성 해결 테스트
            class DependentService:
                def __init__(self, test_service: TestService):
                    self.test_service = test_service

                def get_dependent_value(self):
                    return f"dependent_{self.test_service.get_value()}"

            register(DependentService)
            dependent_service = resolve(DependentService)

            test_result["dependency_resolution"] = {
                "dependent_service_created": dependent_service is not None,
                "dependency_injected": (
                    dependent_service.get_dependent_value()
                    if dependent_service
                    else None
                ),
            }

            # 3. 순환 참조 탐지 테스트
            cycles = container.detect_circular_dependencies()
            test_result["circular_dependency_detection"] = {
                "cycles_detected": len(cycles),
                "cycles": cycles,
            }

            test_result["status"] = "success"

        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
            self.logger.error(f"의존성 주입 시스템 테스트 실패: {e}")

        self.test_results["dependency_injection"] = test_result
        self.logger.info(f"✅ 의존성 주입 시스템 테스트 완료: {test_result['status']}")

    def test_integrated_performance(self):
        """통합 성능 테스트"""
        self.logger.info("🏎️ 통합 성능 테스트")

        test_result = {
            "system_integration": {},
            "performance_metrics": {},
            "resource_usage": {},
        }

        try:
            # 1. 시스템 통합 테스트
            memory_manager = get_unified_memory_manager()
            compute_executor = get_compute_executor()

            # 대용량 데이터 처리 테스트
            large_data = np.random.rand(5000, 1000)

            def complex_operation(data):
                # 복잡한 연산 시뮬레이션
                result = np.dot(data, data.T)
                return np.sum(result)

            # 메모리 상태 확인
            memory_status_before = get_memory_status()

            # 통합 실행
            start_time = time.time()
            with memory_manager.temporary_allocation(
                large_data.nbytes, torch.float32
            ) as temp_tensor:
                result = smart_execute(
                    complex_operation, large_data, TaskType.DATA_PROCESSING
                )
            execution_time = time.time() - start_time

            # 메모리 상태 확인
            memory_status_after = get_memory_status()

            test_result["system_integration"] = {
                "execution_success": True,
                "result": float(result),
                "execution_time": execution_time,
            }

            # 2. 성능 메트릭
            test_result["performance_metrics"] = {
                "data_size_mb": large_data.nbytes / (1024 * 1024),
                "throughput_mb_per_sec": (large_data.nbytes / (1024 * 1024))
                / execution_time,
                "memory_efficiency": (
                    "good" if execution_time < 1.0 else "needs_improvement"
                ),
            }

            # 3. 리소스 사용량
            test_result["resource_usage"] = {
                "memory_before": memory_status_before,
                "memory_after": memory_status_after,
                "gpu_available": torch.cuda.is_available(),
            }

            test_result["status"] = "success"

        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
            self.logger.error(f"통합 성능 테스트 실패: {e}")

        self.test_results["integrated_performance"] = test_result
        self.logger.info(f"✅ 통합 성능 테스트 완료: {test_result['status']}")

    def print_test_summary(self):
        """테스트 요약 출력"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("📊 통합 시스템 테스트 요약")
        self.logger.info("=" * 50)

        total_tests = len(self.test_results)
        successful_tests = sum(
            1
            for result in self.test_results.values()
            if result.get("status") == "success"
        )

        self.logger.info(f"총 테스트: {total_tests}")
        self.logger.info(f"성공: {successful_tests}")
        self.logger.info(f"실패: {total_tests - successful_tests}")
        self.logger.info(f"성공률: {(successful_tests / total_tests * 100):.1f}%")

        self.logger.info("\n상세 결과:")
        for test_name, result in self.test_results.items():
            status = result.get("status", "unknown")
            status_emoji = "✅" if status == "success" else "❌"
            self.logger.info(f"{status_emoji} {test_name}: {status}")

            if status == "error":
                self.logger.error(f"   오류: {result.get('error', 'Unknown error')}")


def run_integrated_system_test():
    """통합 시스템 테스트 실행"""
    tester = IntegratedSystemTester()
    results = tester.run_all_tests()
    tester.print_test_summary()
    return results


if __name__ == "__main__":
    run_integrated_system_test()
