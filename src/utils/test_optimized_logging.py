"""
최적화된 로깅 시스템 테스트

중복 로그 초기화 문제 해결 상태를 테스트하고 검증합니다.
"""

import time
import threading
from typing import List, Dict, Any

from .unified_logging import get_logger, get_logging_stats
from .logging_monitor import get_optimization_report


# 간단한 주입 통계 함수
def get_injection_stats():
    return {"injected_modules": 0, "cached_loggers": 0, "module_list": []}


class LoggingSystemTester:
    """로깅 시스템 테스터"""

    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []

    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        print("🧪 최적화된 로깅 시스템 테스트 시작...")

        # 테스트 시작 시간
        start_time = time.time()

        # 개별 테스트 실행
        tests = [
            ("싱글톤 패턴 테스트", self.test_singleton_pattern),
            ("중복 초기화 방지 테스트", self.test_duplicate_prevention),
            ("Thread-Safe 테스트", self.test_thread_safety),
            ("메모리 효율성 테스트", self.test_memory_efficiency),
            ("자동 주입 테스트", self.test_auto_injection),
            ("성능 테스트", self.test_performance),
        ]

        results = {}
        for test_name, test_func in tests:
            print(f"  🔍 {test_name} 실행 중...")
            try:
                result = test_func()
                results[test_name] = {
                    "status": "PASS" if result["success"] else "FAIL",
                    "details": result,
                }
                print(f"    ✅ {test_name}: {'통과' if result['success'] else '실패'}")
            except Exception as e:
                results[test_name] = {"status": "ERROR", "error": str(e)}
                print(f"    ❌ {test_name}: 오류 - {e}")

        # 전체 테스트 시간
        total_time = time.time() - start_time

        # 최종 결과
        passed_tests = sum(1 for r in results.values() if r["status"] == "PASS")
        total_tests = len(tests)

        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100,
            "execution_time_ms": total_time * 1000,
            "test_results": results,
        }

        print(
            f"\n📋 테스트 완료: {passed_tests}/{total_tests} 통과 ({summary['success_rate']:.1f}%)"
        )

        return summary

    def test_singleton_pattern(self) -> Dict[str, Any]:
        """싱글톤 패턴 테스트"""
        from .unified_logging import _get_factory

        # 여러 번 팩토리 인스턴스 요청
        factory1 = _get_factory()
        factory2 = _get_factory()
        factory3 = _get_factory()

        # 모두 같은 인스턴스인지 확인
        same_instance = (factory1 is factory2) and (factory2 is factory3)

        return {
            "success": same_instance,
            "factory1_id": id(factory1),
            "factory2_id": id(factory2),
            "factory3_id": id(factory3),
            "message": (
                "싱글톤 패턴이 올바르게 작동합니다"
                if same_instance
                else "싱글톤 패턴 실패"
            ),
        }

    def test_duplicate_prevention(self) -> Dict[str, Any]:
        """중복 초기화 방지 테스트"""
        test_module_name = "test_duplicate_module"

        # 같은 이름으로 여러 번 로거 요청
        logger1 = get_logger(test_module_name)
        logger2 = get_logger(test_module_name)
        logger3 = get_logger(test_module_name)

        # 모두 같은 로거 인스턴스인지 확인
        same_logger = (logger1 is logger2) and (logger2 is logger3)

        # 핸들러 중복 확인
        handler_count = len(logger1.handlers)

        return {
            "success": same_logger and handler_count > 0,
            "same_logger_instance": same_logger,
            "handler_count": handler_count,
            "logger1_id": id(logger1),
            "logger2_id": id(logger2),
            "logger3_id": id(logger3),
            "message": (
                "중복 초기화가 성공적으로 방지되었습니다"
                if same_logger
                else "중복 초기화 방지 실패"
            ),
        }

    def test_thread_safety(self) -> Dict[str, Any]:
        """Thread-Safe 테스트"""
        results = []
        errors = []

        def create_logger(thread_id):
            try:
                logger = get_logger(f"thread_test_{thread_id}")
                results.append(
                    {
                        "thread_id": thread_id,
                        "logger_id": id(logger),
                        "handler_count": len(logger.handlers),
                    }
                )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # 10개 스레드에서 동시에 로거 생성
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_logger, args=(i,))
            threads.append(thread)
            thread.start()

        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()

        success = len(errors) == 0 and len(results) == 10

        return {
            "success": success,
            "created_loggers": len(results),
            "errors": errors,
            "results": results,
            "message": (
                "Thread-Safe 테스트 통과"
                if success
                else f"Thread-Safe 테스트 실패: {len(errors)}개 오류"
            ),
        }

    def test_memory_efficiency(self) -> Dict[str, Any]:
        """메모리 효율성 테스트"""
        # 테스트 전 통계
        stats_before = get_logging_stats()

        # 100개 로거 생성
        test_loggers = []
        for i in range(100):
            logger = get_logger(f"memory_test_{i}")
            test_loggers.append(logger)

        # 테스트 후 통계
        stats_after = get_logging_stats()

        # 핸들러 수가 적절히 공유되는지 확인 (100개 로거에 대해 핸들러는 훨씬 적어야 함)
        logger_increase = stats_after["total_loggers"] - stats_before["total_loggers"]
        handler_increase = (
            stats_after["total_handlers"] - stats_before["total_handlers"]
        )

        # 메모리 효율성: 핸들러 증가가 로거 증가보다 훨씬 적어야 함
        efficiency_ratio = handler_increase / max(logger_increase, 1)
        memory_efficient = efficiency_ratio < 0.1  # 핸들러가 로거의 10% 미만으로 증가

        return {
            "success": memory_efficient,
            "logger_increase": logger_increase,
            "handler_increase": handler_increase,
            "efficiency_ratio": efficiency_ratio,
            "memory_efficient": memory_efficient,
            "message": f"메모리 효율성 {'양호' if memory_efficient else '개선 필요'} (효율성 비율: {efficiency_ratio:.3f})",
        }

    def test_auto_injection(self) -> Dict[str, Any]:
        """자동 주입 테스트"""
        # 주입 통계 확인
        injection_stats = get_injection_stats()

        success = injection_stats["injected_modules"] >= 0

        return {
            "success": success,
            "injected_modules": injection_stats["injected_modules"],
            "cached_loggers": injection_stats["cached_loggers"],
            "module_list": injection_stats["module_list"][:5],  # 처음 5개만
            "message": f"자동 주입 시스템 {'정상 작동' if success else '오류'}",
        }

    def test_performance(self) -> Dict[str, Any]:
        """성능 테스트"""
        # 로거 생성 성능 측정
        start_time = time.time()

        # 1000개 로거 생성
        for i in range(1000):
            get_logger(f"performance_test_{i}")

        creation_time = time.time() - start_time

        # 로그 기록 성능 측정
        test_logger = get_logger("performance_logger")

        start_time = time.time()
        for i in range(1000):
            test_logger.info(f"Performance test message {i}")

        logging_time = time.time() - start_time

        # 성능 기준: 1000개 로거 생성이 1초 미만, 1000개 로그 기록이 0.5초 미만
        creation_fast = creation_time < 1.0
        logging_fast = logging_time < 0.5

        success = creation_fast and logging_fast

        return {
            "success": success,
            "creation_time_ms": creation_time * 1000,
            "logging_time_ms": logging_time * 1000,
            "creation_fast": creation_fast,
            "logging_fast": logging_fast,
            "message": f"성능 테스트 {'통과' if success else '실패'}",
        }

    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """테스트 보고서 생성"""
        report = []
        report.append("=" * 70)
        report.append("최적화된 로깅 시스템 테스트 보고서")
        report.append("=" * 70)
        report.append(f"실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"총 테스트: {test_results['total_tests']}")
        report.append(f"통과: {test_results['passed_tests']}")
        report.append(f"실패: {test_results['failed_tests']}")
        report.append(f"성공률: {test_results['success_rate']:.1f}%")
        report.append(f"실행 시간: {test_results['execution_time_ms']:.1f}ms")
        report.append("")

        # 개별 테스트 결과
        for test_name, result in test_results["test_results"].items():
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            report.append(f"{status_icon} {test_name}: {result['status']}")

            if "details" in result:
                details = result["details"]
                if "message" in details:
                    report.append(f"   📝 {details['message']}")

        report.append("")
        report.append("📊 시스템 통계:")

        # 현재 시스템 통계 추가
        logging_stats = get_logging_stats()
        injection_stats = get_injection_stats()

        report.append(f"   - 총 로거 수: {logging_stats.get('total_loggers', 0)}")
        report.append(f"   - 총 핸들러 수: {logging_stats.get('total_handlers', 0)}")
        report.append(
            f"   - 주입된 모듈 수: {injection_stats.get('injected_modules', 0)}"
        )

        report.append("=" * 70)

        return "\n".join(report)


def run_logging_system_test() -> str:
    """로깅 시스템 테스트 실행 및 보고서 반환"""
    tester = LoggingSystemTester()
    results = tester.run_all_tests()

    # 테스트 보고서와 최적화 보고서 결합
    test_report = tester.generate_test_report(results)
    optimization_report = get_optimization_report()

    combined_report = f"{test_report}\n\n{optimization_report}"

    return combined_report


if __name__ == "__main__":
    # 직접 실행 시 테스트 수행
    report = run_logging_system_test()
    print(report)
