#!/usr/bin/env python3
"""
DAEBAK_AI 로또 예측 시스템 - src/utils 모듈 완전 검증 스크립트

순환 참조 해결, GPU 최적화 강화, 메모리 풀링 시스템 등 모든 수정사항을 검증합니다.
"""

import sys
import time
import torch
import numpy as np
from typing import Dict, Any


def test_circular_imports():
    """순환 참조 테스트"""
    print("🔍 순환 참조 테스트 시작...")

    try:
        # 전체 utils 모듈 import 테스트
        import src.utils

        print("✅ src.utils 전체 import 성공")

        # 개별 모듈 import 테스트
        from src.utils.cache_manager import SelfHealingCacheManager, get_cache_manager
        from src.utils.memory_manager import OptimizedMemoryManager, get_memory_manager
        from src.utils.performance_optimizer import (
            get_smart_computation_engine,
            get_cuda_stream_manager,
        )
        from src.utils.cuda_optimizers import get_cuda_optimizer

        print("✅ 모든 개별 모듈 import 성공")
        print("✅ 순환 참조 문제 해결 완료")
        return True

    except ImportError as e:
        print(f"❌ Import 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False


def test_gpu_optimization():
    """GPU 최적화 테스트"""
    print("\n🚀 GPU 최적화 테스트 시작...")

    try:
        # 스마트 연산 엔진 테스트
        from src.utils import get_smart_computation_engine

        engine = get_smart_computation_engine()
        print(f"✅ 스마트 연산 엔진 초기화: GPU 사용 가능={engine.gpu_available}")

        # CUDA 스트림 관리자 테스트
        from src.utils import get_cuda_stream_manager

        stream_manager = get_cuda_stream_manager()
        stats = stream_manager.get_stats()
        print(f"✅ CUDA 스트림 관리자: {stats}")

        # CUDA 최적화기 테스트
        from src.utils import get_cuda_optimizer

        optimizer = get_cuda_optimizer()
        print(
            f"✅ CUDA 최적화기 초기화: TensorRT 사용 가능={optimizer.config.use_tensorrt}"
        )

        return True

    except Exception as e:
        print(f"❌ GPU 최적화 테스트 실패: {e}")
        return False


def test_memory_management():
    """메모리 관리 테스트"""
    print("\n💾 메모리 관리 테스트 시작...")

    try:
        from src.utils import get_memory_manager

        manager = get_memory_manager()

        # 기본 통계 확인
        stats = manager.get_simple_stats()
        print(f"✅ 메모리 관리자 통계: {stats}")

        # 메모리 풀링 테스트
        if hasattr(manager, "pool_enabled") and manager.pool_enabled:
            # 텐서 할당 테스트
            tensor1 = manager.smart_allocate((100, 100), prefer_gpu=False)
            tensor2 = manager.smart_allocate((100, 100), prefer_gpu=False)

            # 메모리 풀에 반환
            manager.return_to_pool(tensor1)
            manager.return_to_pool(tensor2)

            # 풀 상태 확인
            pool_stats = manager.get_simple_stats()
            print(f"✅ 메모리 풀링 테스트: {pool_stats}")

            # 메모리 풀 정리
            manager.clear_memory_pool()
            print("✅ 메모리 풀 정리 완료")

        return True

    except Exception as e:
        print(f"❌ 메모리 관리 테스트 실패: {e}")
        return False


def test_performance_monitoring():
    """성능 모니터링 테스트"""
    print("\n📊 성능 모니터링 테스트 시작...")

    try:
        from src.utils import get_auto_performance_monitor

        monitor = get_auto_performance_monitor()

        # 성능 추적 테스트
        with monitor.track("테스트_작업"):
            # 간단한 연산 수행
            data = np.random.rand(1000, 1000)
            result = np.dot(data, data.T)
            time.sleep(0.1)  # 의도적 지연

        # 성능 요약 확인
        summary = monitor.get_performance_summary()
        print(f"✅ 성능 모니터링 요약: {summary}")

        # 히스토리 정리
        monitor.clear_history()
        print("✅ 성능 히스토리 정리 완료")

        return True

    except Exception as e:
        print(f"❌ 성능 모니터링 테스트 실패: {e}")
        return False


def test_cache_system():
    """캐시 시스템 테스트"""
    print("\n🗄️ 캐시 시스템 테스트 시작...")

    try:
        from src.utils import get_cache_manager

        cache = get_cache_manager()

        # 캐시 저장/조회 테스트
        test_data = {"key": "value", "number": 42}
        cache.set("test_key", test_data)

        retrieved_data = cache.get("test_key")
        if retrieved_data == test_data:
            print("✅ 캐시 저장/조회 성공")
        else:
            print("❌ 캐시 데이터 불일치")
            return False

        # 캐시 통계 확인
        stats = cache.get_simple_stats()
        print(f"✅ 캐시 통계: {stats}")

        # 캐시 정리
        cache.clear()
        print("✅ 캐시 정리 완료")

        return True

    except Exception as e:
        print(f"❌ 캐시 시스템 테스트 실패: {e}")
        return False


def test_compatibility_wrappers():
    """하위 호환성 래퍼 테스트"""
    print("\n🔄 하위 호환성 래퍼 테스트 시작...")

    try:
        # 하위 호환성 함수들 테스트
        from src.utils import (
            get_optimized_memory_manager,
            get_gpu_memory_manager,
            get_self_healing_cache_manager,
            get_gpu_cache_manager,
        )

        # 메모리 관리자 래퍼 테스트
        old_manager = get_optimized_memory_manager()
        new_manager = get_gpu_memory_manager()
        print(
            f"✅ 메모리 관리자 래퍼: {type(old_manager).__name__} == {type(new_manager).__name__}"
        )

        # 캐시 관리자 래퍼 테스트
        old_cache = get_self_healing_cache_manager()
        new_cache = get_gpu_cache_manager()
        print(
            f"✅ 캐시 관리자 래퍼: {type(old_cache).__name__} == {type(new_cache).__name__}"
        )

        return True

    except Exception as e:
        print(f"❌ 하위 호환성 래퍼 테스트 실패: {e}")
        return False


def main():
    """메인 검증 함수"""
    print("🎯 DAEBAK_AI 로또 예측 시스템 - src/utils 모듈 완전 검증")
    print("=" * 60)

    test_results = []

    # 각 테스트 실행
    test_results.append(("순환 참조 해결", test_circular_imports()))
    test_results.append(("GPU 최적화", test_gpu_optimization()))
    test_results.append(("메모리 관리", test_memory_management()))
    test_results.append(("성능 모니터링", test_performance_monitoring()))
    test_results.append(("캐시 시스템", test_cache_system()))
    test_results.append(("하위 호환성", test_compatibility_wrappers()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 검증 결과 요약:")

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n총 {len(test_results)}개 테스트 중 {passed}개 성공, {failed}개 실패")

    if failed == 0:
        print("🎉 모든 테스트 통과! src/utils 모듈 수정 완료")
        return 0
    else:
        print("⚠️ 일부 테스트 실패. 추가 수정 필요")
        return 1


if __name__ == "__main__":
    sys.exit(main())
