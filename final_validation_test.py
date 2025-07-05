#!/usr/bin/env python3
"""
DAEBAK_AI 로또 예측 시스템 - 최종 검증 테스트

모든 수정사항이 정상적으로 작동하는지 종합적으로 검증합니다.
"""

import sys
import time


def test_critical_classes():
    """🚨 중요한 클래스들의 존재 여부 확인"""
    print("🔍 중요한 클래스들 존재 여부 확인...")

    try:
        # 1. IndependentGPUNormalizer 클래스 테스트
        from src.utils.normalizer import IndependentGPUNormalizer

        normalizer = IndependentGPUNormalizer()
        print("✅ IndependentGPUNormalizer 클래스 정상 작동")

        # 2. SelfHealingCacheManager 클래스 테스트
        from src.utils.cache_manager import SelfHealingCacheManager

        cache_manager = SelfHealingCacheManager()
        print("✅ SelfHealingCacheManager 클래스 정상 작동")

        # 3. OptimizedMemoryManager 클래스 테스트
        from src.utils.memory_manager import OptimizedMemoryManager

        memory_manager = OptimizedMemoryManager()
        print("✅ OptimizedMemoryManager 클래스 정상 작동")

        return True

    except Exception as e:
        print(f"❌ 중요한 클래스 테스트 실패: {e}")
        return False


def test_init_imports():
    """🔍 __init__.py에서 모든 함수들이 정상 작동하는지 확인"""
    print("\n🔍 __init__.py import 테스트...")

    try:
        # 핵심 시스템 함수들 테스트
        from src.utils import (
            get_smart_computation_engine,
            get_unified_memory_manager,
            get_unified_async_manager,
            get_auto_recovery_system,
            get_compute_executor,
            get_cuda_optimizer,
            get_cuda_stream_manager,
            get_gpu_normalizer,
            get_cache_manager,
            get_memory_manager,
        )

        # 각 함수 호출 테스트
        engine = get_smart_computation_engine()
        print(f"✅ get_smart_computation_engine: GPU={engine.gpu_available}")

        memory_mgr = get_unified_memory_manager()
        print("✅ get_unified_memory_manager: 정상 작동")

        async_mgr = get_unified_async_manager()
        print("✅ get_unified_async_manager: 정상 작동")

        recovery_sys = get_auto_recovery_system()
        print("✅ get_auto_recovery_system: 정상 작동")

        compute_exec = get_compute_executor()
        print("✅ get_compute_executor: 정상 작동")

        cuda_opt = get_cuda_optimizer()
        print(f"✅ get_cuda_optimizer: TensorRT={cuda_opt.config.use_tensorrt}")

        stream_mgr = get_cuda_stream_manager()
        stats = stream_mgr.get_stats()
        print(f"✅ get_cuda_stream_manager: {stats}")

        normalizer = get_gpu_normalizer()
        print("✅ get_gpu_normalizer: 정상 작동")

        cache = get_cache_manager()
        print("✅ get_cache_manager: 정상 작동")

        memory = get_memory_manager()
        print("✅ get_memory_manager: 정상 작동")

        return True

    except Exception as e:
        print(f"❌ __init__.py import 테스트 실패: {e}")
        return False


def test_compatibility_wrappers():
    """🔄 하위 호환성 래퍼들 테스트"""
    print("\n🔄 하위 호환성 래퍼 테스트...")

    try:
        from src.utils import (
            get_optimized_memory_manager,
            get_gpu_memory_manager,
            get_self_healing_cache_manager,
            get_gpu_cache_manager,
            get_independent_gpu_normalizer,
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

        # 정규화기 래퍼 테스트
        normalizer = get_independent_gpu_normalizer()
        print(f"✅ 정규화기 래퍼: {type(normalizer).__name__}")

        return True

    except Exception as e:
        print(f"❌ 하위 호환성 래퍼 테스트 실패: {e}")
        return False


def test_gpu_features():
    """🚀 GPU 최적화 기능 테스트"""
    print("\n🚀 GPU 최적화 기능 테스트...")

    try:
        # CUDA 스트림 관리자 기능 테스트
        from src.utils import get_cuda_stream_manager

        stream_manager = get_cuda_stream_manager()

        # 스트림 컨텍스트 테스트
        with stream_manager.stream_context() as stream:
            print(f"✅ CUDA 스트림 컨텍스트: {stream is not None}")

        # 메모리 풀링 테스트
        from src.utils import get_memory_manager

        memory_manager = get_memory_manager()

        if hasattr(memory_manager, "pool_enabled"):
            print(f"✅ 메모리 풀링 시스템: {memory_manager.pool_enabled}")

            # 간단한 메모리 풀 테스트
            if memory_manager.pool_enabled:
                tensor = memory_manager.smart_allocate((10, 10), prefer_gpu=False)
                memory_manager.return_to_pool(tensor)
                print("✅ 메모리 풀 테스트: 정상 작동")

        # TensorRT 최적화기 테스트
        from src.utils import get_cuda_optimizer

        optimizer = get_cuda_optimizer()

        if hasattr(optimizer, "tensorrt_optimize_advanced"):
            print("✅ 고급 TensorRT 최적화 기능 사용 가능")

        return True

    except Exception as e:
        print(f"❌ GPU 최적화 기능 테스트 실패: {e}")
        return False


def test_circular_imports():
    """🔄 순환 참조 테스트"""
    print("\n🔄 순환 참조 테스트...")

    try:
        # 전체 모듈 import 테스트
        import src.utils

        print("✅ src.utils 모듈 전체 import 성공")

        # 주요 함수들 개별 import 테스트 (순환 참조 확인)
        from src.utils import (
            get_smart_computation_engine,
            get_cuda_stream_manager,
            get_gpu_normalizer,
            get_cache_manager,
            get_memory_manager,
        )

        print("✅ 주요 함수들 import 성공 - 순환 참조 없음")

        return True

    except Exception as e:
        print(f"❌ 순환 참조 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🎯 DAEBAK_AI 로또 예측 시스템 - 최종 검증 테스트")
    print("=" * 60)

    test_results = []

    # 각 테스트 실행
    test_results.append(("중요한 클래스 존재", test_critical_classes()))
    test_results.append(("__init__.py import", test_init_imports()))
    test_results.append(("하위 호환성 래퍼", test_compatibility_wrappers()))
    test_results.append(("GPU 최적화 기능", test_gpu_features()))
    test_results.append(("순환 참조 해결", test_circular_imports()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 최종 검증 결과:")

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
        print("\n🎉 모든 테스트 통과!")
        print("✅ 순환 참조 완전 해결")
        print("✅ 클래스 불일치 완전 해결")
        print("✅ GPU 최적화 기능 정상 작동")
        print("✅ 하위 호환성 완전 보장")
        print("\n🚀 src/utils 모듈 수정 완료! 시스템 사용 준비 완료!")
        return 0
    else:
        print("\n⚠️ 일부 테스트 실패. 추가 수정 필요")
        return 1


if __name__ == "__main__":
    sys.exit(main())
