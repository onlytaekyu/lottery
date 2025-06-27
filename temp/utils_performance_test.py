#!/usr/bin/env python3
"""
src/utils 모듈 성능 측정 스크립트
"""

import time
import psutil
import gc
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def measure_memory():
    """현재 메모리 사용량 측정 (MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def main():
    """메인 테스트 실행"""
    print("🚀 src/utils 성능 측정 및 기능 테스트")
    print("=" * 50)
    
    # 초기 메모리 측정
    gc.collect()
    initial_memory = measure_memory()
    print(f"초기 메모리: {initial_memory:.1f} MB")
    
    # Import 시간 측정
    start_time = time.time()
    
    try:
        # src.utils import
        import src.utils
        
        import_time = time.time() - start_time
        
        # Import 후 메모리 측정
        after_import_memory = measure_memory()
        memory_increase = after_import_memory - initial_memory
        
        print(f"✅ Import 성공!")
        print(f"Import 시간: {import_time * 1000:.1f}ms")
        print(f"Import 후 메모리: {after_import_memory:.1f} MB")
        print(f"메모리 증가: {memory_increase:.1f} MB")
        
        # 기본 기능 테스트
        logger = src.utils.get_logger("test")
        logger.info("로거 테스트 성공")
        print("✅ 로거 작동 확인")
        
        cache_dir = src.utils.get_cache_dir()
        print(f"✅ 캐시 디렉토리: {cache_dir}")
        
        cache_dir_const = src.utils.CACHE_DIR
        print(f"✅ CACHE_DIR 상수: {cache_dir_const}")
        
        print("\n🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    main()
