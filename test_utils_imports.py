#!/usr/bin/env python3
"""
src/utils 모듈 import 테스트
순환 참조 및 클래스 불일치 문제 탐지
"""

import sys
import traceback
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import_individual_modules():
    """개별 모듈 import 테스트"""
    print("🔍 개별 모듈 import 테스트")
    
    modules_to_test = [
        'src.utils.unified_logging',
        'src.utils.factory',
        'src.utils.unified_config',
        'src.utils.error_handler',
        'src.utils.memory_manager',
        'src.utils.cache_manager',
        'src.utils.normalizer',
        'src.utils.performance_optimizer',
        'src.utils.cuda_optimizers',
        'src.utils.compute_strategy',
        'src.utils.auto_recovery_system',
        'src.utils.vector_exporter',
        'src.utils.model_saver',
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            results[module_name] = "✅ 성공"
            print(f"  {module_name}: ✅")
        except Exception as e:
            results[module_name] = f"❌ 실패: {e}"
            print(f"  {module_name}: ❌ {e}")
    
    return results

def test_utils_init():
    """utils __init__.py import 테스트"""
    print("\n🔍 utils __init__.py import 테스트")
    
    try:
        from src.utils import (
            get_smart_computation_engine,
            get_cache_manager,
            get_memory_manager,
            get_gpu_normalizer,
            get_cuda_optimizer,
            get_compute_executor,
            get_auto_recovery_system,
            get_unified_memory_manager,
            get_unified_async_manager,
            get_gpu_vector_exporter,
            get_advanced_model_saver,
        )
        
        print("✅ 모든 함수 import 성공")
        
        # 실제 인스턴스 생성 테스트
        test_instances = {}
        functions_to_test = [
            ('smart_computation_engine', get_smart_computation_engine),
            ('cache_manager', get_cache_manager),
            ('memory_manager', get_memory_manager),
            ('gpu_normalizer', get_gpu_normalizer),
            ('cuda_optimizer', get_cuda_optimizer),
            ('compute_executor', get_compute_executor),
            ('auto_recovery_system', get_auto_recovery_system),
            ('unified_memory_manager', get_unified_memory_manager),
            ('unified_async_manager', get_unified_async_manager),
            ('gpu_vector_exporter', get_gpu_vector_exporter),
            ('advanced_model_saver', get_advanced_model_saver),
        ]
        
        for name, func in functions_to_test:
            try:
                instance = func()
                test_instances[name] = instance
                print(f"  {name}: ✅ 인스턴스 생성 성공")
            except Exception as e:
                print(f"  {name}: ❌ 인스턴스 생성 실패: {e}")
                traceback.print_exc()
        
        return test_instances
        
    except Exception as e:
        print(f"❌ utils __init__.py import 실패: {e}")
        traceback.print_exc()
        return None

def test_class_consistency():
    """클래스 일관성 테스트"""
    print("\n🔍 클래스 일관성 테스트")
    
    try:
        # cache_manager 클래스 확인
        from src.utils.cache_manager import GPUCache, SelfHealingCacheManager
        print("  cache_manager 클래스들: ✅")
        
        # memory_manager 클래스 확인  
        from src.utils.memory_manager import GPUMemoryManager, OptimizedMemoryManager
        print("  memory_manager 클래스들: ✅")
        
        # normalizer 클래스 확인
        from src.utils.normalizer import GPUNormalizer, get_gpu_normalizer
        normalizer = get_gpu_normalizer()
        print("  normalizer 클래스들: ✅")
        
        # performance_optimizer 클래스 확인
        from src.utils.performance_optimizer import (
            SmartComputationEngine,
            CUDAStreamManager,
            get_smart_computation_engine,
            get_cuda_stream_manager
        )
        print("  performance_optimizer 클래스들: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ 클래스 일관성 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_circular_imports():
    """순환 참조 테스트"""
    print("\n🔍 순환 참조 테스트")
    
    # 모듈 로드 순서 테스트
    import_order = [
        'src.utils.unified_logging',
        'src.utils.factory', 
        'src.utils.unified_config',
        'src.utils.error_handler',
        'src.utils.memory_manager',
        'src.utils.cache_manager',
        'src.utils.performance_optimizer',
        'src.utils.normalizer',
        'src.utils.cuda_optimizers',
        'src.utils.compute_strategy',
        'src.utils.auto_recovery_system',
        'src.utils',  # 최종 __init__.py
    ]
    
    for module_name in import_order:
        try:
            __import__(module_name)
            print(f"  {module_name}: ✅")
        except Exception as e:
            print(f"  {module_name}: ❌ {e}")
            return False
    
    return True

def main():
    """메인 테스트 실행"""
    print("🚀 src/utils 모듈 통합 테스트 시작\n")
    
    # 1. 개별 모듈 import 테스트
    individual_results = test_import_individual_modules()
    
    # 2. utils __init__.py 테스트
    init_results = test_utils_init()
    
    # 3. 클래스 일관성 테스트
    class_consistency = test_class_consistency()
    
    # 4. 순환 참조 테스트
    circular_imports = test_circular_imports()
    
    # 결과 요약
    print("\n" + "="*50)
    print("📊 테스트 결과 요약")
    print("="*50)
    
    print(f"개별 모듈 import: {sum(1 for r in individual_results.values() if '✅' in r)}/{len(individual_results)} 성공")
    print(f"utils __init__.py: {'✅ 성공' if init_results else '❌ 실패'}")
    print(f"클래스 일관성: {'✅ 성공' if class_consistency else '❌ 실패'}")
    print(f"순환 참조 테스트: {'✅ 성공' if circular_imports else '❌ 실패'}")
    
    # 실패한 모듈 상세 정보
    failed_modules = [name for name, result in individual_results.items() if '❌' in result]
    if failed_modules:
        print(f"\n❌ 실패한 모듈들:")
        for module in failed_modules:
            print(f"  - {module}: {individual_results[module]}")
    
    # 전체 결과
    all_passed = (
        all('✅' in r for r in individual_results.values()) and
        init_results and
        class_consistency and
        circular_imports
    )
    
    print(f"\n🎯 전체 테스트 결과: {'✅ 모두 통과' if all_passed else '❌ 일부 실패'}")
    
    return all_passed

if __name__ == "__main__":
    main()
