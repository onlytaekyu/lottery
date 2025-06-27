import time
import psutil
import gc
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def measure_memory():
    return psutil.Process().memory_info().rss / 1024 / 1024


print("🔍 최적화된 src/utils 성능 측정")
print("=" * 40)

# 시작 메모리
gc.collect()
start_memory = measure_memory()
print(f"시작 메모리: {start_memory:.1f}MB")

# Import 시간 측정
start_time = time.time()
import src.utils

import_time = (time.time() - start_time) * 1000

# Import 후 메모리
after_memory = measure_memory()
memory_increase = after_memory - start_memory

print(f"Import 시간: {import_time:.1f}ms")
print(f"Import 후 메모리: {after_memory:.1f}MB")
print(f"메모리 증가: {memory_increase:.1f}MB")

# Lazy loading 테스트
if hasattr(src.utils, "get_import_stats"):
    stats = src.utils.get_import_stats()
    print(f'로드된 모듈: {stats["loaded_modules"]}개')
    print(f'캐시된 항목: {stats["cached_items"]}개')
    print(f'사용 가능한 lazy 모듈: {stats["available_lazy_modules"]}개')

# ThreadLocalCache 테스트
try:
    cache = src.utils.ThreadLocalCache()
    print("✅ ThreadLocalCache 정상 작동")
except Exception as e:
    print(f"❌ ThreadLocalCache 오류: {e}")

print("\n🎉 측정 완료!")
