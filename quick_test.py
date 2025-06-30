#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
빠른 벡터화 시스템 테스트
"""

import sys
import numpy as np
from pathlib import Path


def quick_test():
    try:
        print("🚀 빠른 벡터화 시스템 테스트 시작")

        # 1. 검증 모듈 테스트
        from src.utils.feature_vector_validator import ESSENTIAL_FEATURES

        print(f"✅ 필수 특성 {len(ESSENTIAL_FEATURES)}개 로드 성공")

        # 2. 완전한 설정 준비
        config = {
            "paths": {"cache_dir": "data/cache"},
            "caching": {
                "enable_feature_cache": True,
                "max_cache_size": 10000,
                "cache_log_level": "INFO",
                "cache_metrics": {
                    "save": True,
                    "report_interval": 1000,
                    "file_path": "logs/cache_stats.json",
                },
                "persistent_cache": True,
                "cache_file": "state_vector_cache.npz",
            },
            "vectorizer": {"use_cache": True},
            "filtering": {
                "remove_low_variance_features": False,
                "variance_threshold": 0.01,
            },
        }

        # 3. 향상된 벡터화 시스템 테스트
        from src.analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer

        vectorizer = EnhancedPatternVectorizer(config)
        print("✅ 향상된 벡터화 시스템 초기화 성공")

        # 4. 테스트 데이터 생성
        test_data = {
            "frequency_analysis": {
                f"num_{i}": np.random.randint(1, 50) for i in range(1, 11)
            },
            "gap_patterns": {f"gap_{i}": np.random.uniform(1, 10) for i in range(1, 6)},
            "pair_frequency": {
                f"pair_{i}_{j}": np.random.randint(1, 20)
                for i in range(1, 4)
                for j in range(i + 1, 5)
            },
            "segment_distribution": {
                f"segment_{i}": np.random.randint(5, 25) for i in range(1, 6)
            },
            "position_analysis": {
                f"position_{i}": {
                    f"pos_val_{j}": np.random.randint(1, 45) for j in range(1, 6)
                }
                for i in range(1, 4)
            },
        }
        print("✅ 테스트 데이터 생성 완료")

        # 5. 벡터화 실행
        vector = vectorizer.vectorize_full_analysis_enhanced(test_data)
        names = vectorizer.get_feature_names()

        print(f"✅ 벡터 생성 성공: {len(vector)}차원")
        print(f"✅ 특성 이름 생성: {len(names)}개")

        # 6. 차원 일치성 검증
        dimension_match = len(vector) == len(names)
        print(f"✅ 차원 일치: {dimension_match}")

        # 7. 품질 지표 계산
        vector_array = np.array(vector)
        zero_ratio = np.sum(vector_array == 0) / len(vector_array)

        # 엔트로피 계산
        hist, _ = np.histogram(vector_array, bins=20)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

        print(f"📊 품질 지표:")
        print(f"   - 0값 비율: {zero_ratio*100:.1f}%")
        print(f"   - 엔트로피: {entropy:.3f}")
        print(f"   - 평균값: {np.mean(vector_array):.3f}")
        print(f"   - 표준편차: {np.std(vector_array):.3f}")

        # 8. 성공 기준 검증
        success_criteria = {
            "차원 일치": dimension_match,
            "0값 비율 < 50%": zero_ratio < 0.5,
            "엔트로피 > 0": entropy > 0,
            "벡터 크기 > 100": len(vector) > 100,
        }

        print("\n📋 성공 기준 검증:")
        passed = 0
        for criterion, result in success_criteria.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {criterion}: {status}")
            if result:
                passed += 1

        success_rate = passed / len(success_criteria) * 100
        print(f"\n🎯 성공률: {passed}/{len(success_criteria)} ({success_rate:.1f}%)")

        if passed == len(success_criteria):
            print("🎉 모든 기준 통과! 벡터화 시스템이 정상 작동합니다.")
            return True
        else:
            print("⚠️  일부 기준 미달 - 추가 개선 필요")
            return False

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 디렉토리 생성
    Path("data/cache").mkdir(parents=True, exist_ok=True)

    # 테스트 실행
    success = quick_test()
    sys.exit(0 if success else 1)
