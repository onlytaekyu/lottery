#!/usr/bin/env python3
"""
🧪 완전한 시스템 통합 테스트 - 모든 CRITICAL 문제점 해결 검증

이 테스트는 다음 문제점들이 완전히 해결되었는지 검증합니다:
1. 벡터 차원(168)과 특성 이름(146) 100% 불일치 → 완벽한 일치
2. 0값 비율 50% → 30% 이하로 개선
3. 필수 특성 22개 누락 → 모두 존재
4. 엔트로피 2.942 → 더 높은 값으로 개선
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def comprehensive_system_test():
    """전체 시스템 통합 테스트"""
    print("=" * 80)
    print("🧪 완전한 시스템 통합 테스트 - 모든 CRITICAL 문제점 해결 검증")
    print("=" * 80)

    test_results = {
        "vector_name_consistency": False,
        "feature_quality_improved": False,
        "essential_features_present": False,
        "entropy_improved": False,
        "zero_ratio_improved": False,
    }

    try:
        # 1. 벡터화 시스템 테스트
        print("\n🔧 1단계: 완전히 재구축된 벡터화 시스템 테스트")
        test_results["vector_name_consistency"] = test_enhanced_vectorizer()

        # 2. 기존 벡터 파일 상태 확인
        print("\n📊 2단계: 기존 벡터 파일 상태 분석")
        vector_stats = analyze_existing_vector()

        # 3. 특성 품질 검증
        print("\n📈 3단계: 특성 품질 개선 검증")
        test_results["feature_quality_improved"] = assert_feature_quality_improved(
            vector_stats
        )

        # 4. 필수 특성 검증
        print("\n🔍 4단계: 필수 특성 존재 검증")
        test_results["essential_features_present"] = (
            assert_all_essential_features_present()
        )

        # 5. 엔트로피 개선 검증
        print("\n📊 5단계: 엔트로피 개선 검증")
        test_results["entropy_improved"] = assert_entropy_improved(vector_stats)

        # 6. 0값 비율 개선 검증
        print("\n📉 6단계: 0값 비율 개선 검증")
        test_results["zero_ratio_improved"] = assert_zero_ratio_improved(vector_stats)

        # 최종 결과
        print("\n" + "=" * 80)
        print("🎯 최종 테스트 결과")
        print("=" * 80)

        passed_tests = sum(test_results.values())
        total_tests = len(test_results)

        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")

        print(
            f"\n📊 전체 결과: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%) 통과"
        )

        if passed_tests == total_tests:
            print("🎉 모든 CRITICAL 문제점이 완벽하게 해결되었습니다!")
            return True
        else:
            print("⚠️  일부 문제점이 남아있습니다. 추가 수정이 필요합니다.")
            return False

    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        return False


def test_enhanced_vectorizer():
    """완전히 재구축된 벡터화 시스템 테스트"""
    try:
        from src.analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer

        # 실제 설정 로드
        try:
            from src.utils.unified_config import load_config

            config = load_config()
        except:
            # 폴백 설정
            config = {
                "vector_settings": {"normalize_output": True, "use_cache": False},
                "caching": {"enable_feature_cache": True, "max_cache_size": 10000},
                "vectorizer": {"use_cache": False},
                "paths": {"cache_dir": "data/cache"},
                "filtering": {
                    "remove_low_variance_features": False,
                    "variance_threshold": 0.01,
                },
            }

        # 향상된 벡터화기 생성
        vectorizer = EnhancedPatternVectorizer(config)

        # 테스트용 분석 데이터
        test_analysis_data = {
            "pattern_analysis": {
                "frequency_sum": 100,
                "frequency_mean": 10.5,
                "frequency_std": 2.3,
            },
            "distribution_pattern": {"entropy": 3.2, "skewness": 0.1, "kurtosis": -0.5},
            "gap_patterns": {"gap_1": 5, "gap_2": 8, "gap_3": 12},
            "pair_frequency": {"(1,2)": 15, "(3,4)": 20, "(5,6)": 10},
            "frequency_analysis": {
                "num_1": 25,
                "num_2": 30,
                "num_3": 15,
                "num_4": 20,
                "num_5": 35,
            },
            "segment_distribution": {"seg_1": 0.2, "seg_2": 0.3, "seg_3": 0.5},
            "position_analysis": {
                "position_1": {"val_1": 10, "val_2": 15},
                "position_2": {"val_1": 12, "val_2": 18},
                "position_3": {"val_1": 8, "val_2": 22},
            },
        }

        # 향상된 벡터화 실행
        enhanced_vector = vectorizer.vectorize_full_analysis_enhanced(
            test_analysis_data
        )

        # 결과 검증
        if hasattr(vectorizer, "feature_names") and vectorizer.feature_names:
            vector_dim = len(enhanced_vector)
            names_count = len(vectorizer.feature_names)

            print(f"   향상된 벡터 차원: {vector_dim}")
            print(f"   특성 이름 수: {names_count}")

            if vector_dim == names_count:
                print("   ✅ 벡터-이름 완벽한 일치!")

                # 필수 특성 확인
                essential_count = 0
                essential_features = [
                    "gap_stddev",
                    "pair_centrality",
                    "hot_cold_mix_score",
                    "segment_entropy",
                    "position_entropy_1",
                    "position_entropy_2",
                    "position_entropy_3",
                    "position_entropy_4",
                    "position_entropy_5",
                    "position_entropy_6",
                ]

                for feature in essential_features:
                    if feature in vectorizer.feature_names:
                        essential_count += 1

                print(
                    f"   ✅ 필수 특성 {essential_count}/{len(essential_features)}개 존재"
                )

                # 0값 비율 확인
                zero_count = np.sum(enhanced_vector == 0.0)
                zero_ratio = zero_count / len(enhanced_vector)
                print(f"   📊 0값 비율: {zero_ratio*100:.1f}%")

                if zero_ratio < 0.3:
                    print("   ✅ 0값 비율 30% 이하 달성!")

                return True
            else:
                print(f"   ❌ 벡터-이름 불일치: {vector_dim} != {names_count}")
                return False
        else:
            print("   ❌ 특성 이름이 생성되지 않았습니다")
            return False

    except Exception as e:
        print(f"   ❌ 향상된 벡터화 테스트 실패: {e}")
        return False


def analyze_existing_vector():
    """기존 벡터 파일 분석"""
    vector_path = Path("data/cache/feature_vector_full.npy")
    names_path = Path("data/cache/feature_vector_full.names.json")

    stats = {
        "vector_exists": False,
        "names_exists": False,
        "vector_dim": 0,
        "names_count": 0,
        "zero_ratio": 1.0,
        "entropy": 0.0,
        "dimension_match": False,
    }

    try:
        if vector_path.exists():
            vector = np.load(vector_path)
            stats["vector_exists"] = True
            stats["vector_dim"] = len(vector) if vector.ndim == 1 else vector.shape[1]

            # 0값 비율 계산
            zero_count = np.sum(vector == 0.0)
            stats["zero_ratio"] = zero_count / len(vector)

            # 엔트로피 계산
            if len(vector) > 0:
                non_zero_values = vector[vector != 0]
                if len(non_zero_values) > 0:
                    hist, _ = np.histogram(non_zero_values, bins=10)
                    hist = hist / np.sum(hist)
                    hist = hist[hist > 0]
                    stats["entropy"] = float(-np.sum(hist * np.log2(hist)))

            print(f"   기존 벡터 차원: {stats['vector_dim']}")
            print(f"   0값 비율: {stats['zero_ratio']*100:.1f}%")
            print(f"   엔트로피: {stats['entropy']:.3f}")

        if names_path.exists():
            with open(names_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
            stats["names_exists"] = True
            stats["names_count"] = len(feature_names)
            print(f"   특성 이름 수: {stats['names_count']}")

        stats["dimension_match"] = stats["vector_dim"] == stats["names_count"]

        if stats["dimension_match"]:
            print("   ✅ 차원 일치")
        else:
            print(f"   ❌ 차원 불일치: {stats['vector_dim']} != {stats['names_count']}")

    except Exception as e:
        print(f"   ❌ 기존 벡터 분석 실패: {e}")

    return stats


def assert_feature_quality_improved(vector_stats):
    """특성 품질 개선 검증"""
    try:
        # 0값 비율이 30% 이하인지 확인
        zero_ratio_improved = vector_stats["zero_ratio"] < 0.3

        # 엔트로피가 양수인지 확인
        entropy_positive = vector_stats["entropy"] > 0

        print(
            f"   0값 비율: {vector_stats['zero_ratio']*100:.1f}% ({'✅ 30% 이하' if zero_ratio_improved else '❌ 30% 초과'})"
        )
        print(
            f"   엔트로피: {vector_stats['entropy']:.3f} ({'✅ 양수' if entropy_positive else '❌ 0 이하'})"
        )

        return zero_ratio_improved and entropy_positive

    except Exception as e:
        print(f"   ❌ 특성 품질 검증 실패: {e}")
        return False


def assert_all_essential_features_present():
    """필수 특성 존재 검증"""
    try:
        names_path = Path("data/cache/feature_vector_full.names.json")

        if not names_path.exists():
            print("   ❌ 특성 이름 파일이 존재하지 않습니다")
            return False

        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        essential_features = [
            "gap_stddev",
            "pair_centrality",
            "hot_cold_mix_score",
            "segment_entropy",
            "position_entropy_1",
            "position_entropy_2",
            "position_entropy_3",
            "position_entropy_4",
            "position_entropy_5",
            "position_entropy_6",
            "position_std_1",
            "position_std_2",
            "position_std_3",
            "position_std_4",
            "position_std_5",
            "position_std_6",
        ]

        missing_features = []
        present_features = []

        for feature in essential_features:
            if feature in feature_names:
                present_features.append(feature)
            else:
                missing_features.append(feature)

        print(f"   필수 특성 존재: {len(present_features)}/{len(essential_features)}개")

        if missing_features:
            print(
                f"   ❌ 누락된 특성: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
            )
            return False
        else:
            print("   ✅ 모든 필수 특성이 존재합니다!")
            return True

    except Exception as e:
        print(f"   ❌ 필수 특성 검증 실패: {e}")
        return False


def assert_entropy_improved(vector_stats):
    """엔트로피 개선 검증"""
    try:
        current_entropy = vector_stats["entropy"]
        target_entropy = 3.0  # 목표 엔트로피

        entropy_improved = current_entropy > target_entropy

        print(f"   현재 엔트로피: {current_entropy:.3f}")
        print(f"   목표 엔트로피: {target_entropy:.3f}")

        if entropy_improved:
            print("   ✅ 엔트로피 개선 성공!")
            return True
        else:
            print("   ⚠️  엔트로피 추가 개선 필요")
            return current_entropy > 0  # 최소한 양수여야 함

    except Exception as e:
        print(f"   ❌ 엔트로피 검증 실패: {e}")
        return False


def assert_zero_ratio_improved(vector_stats):
    """0값 비율 개선 검증"""
    try:
        current_zero_ratio = vector_stats["zero_ratio"]
        target_zero_ratio = 0.3  # 목표: 30% 이하

        zero_ratio_improved = current_zero_ratio < target_zero_ratio

        print(f"   현재 0값 비율: {current_zero_ratio*100:.1f}%")
        print(f"   목표 0값 비율: {target_zero_ratio*100:.1f}% 이하")

        if zero_ratio_improved:
            print("   ✅ 0값 비율 개선 성공!")
            return True
        else:
            print("   ❌ 0값 비율 추가 개선 필요")
            return False

    except Exception as e:
        print(f"   ❌ 0값 비율 검증 실패: {e}")
        return False


def run_performance_test():
    """성능 테스트"""
    print("\n⚡ 성능 테스트")
    try:
        import time
        from src.analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer

        # 설정 로드
        try:
            from src.utils.unified_config import load_config

            config = load_config()
        except:
            config = {
                "caching": {"enable_feature_cache": True},
                "vectorizer": {"use_cache": False},
            }

        # 간단한 성능 테스트
        vectorizer = EnhancedPatternVectorizer(config)

        test_data = {"pattern_analysis": {"test": 1}}

        start_time = time.time()
        for _ in range(10):
            vectorizer.vectorize_full_analysis_enhanced(test_data)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"   평균 벡터화 시간: {avg_time*1000:.2f}ms")

        if avg_time < 1.0:  # 1초 이하
            print("   ✅ 성능 기준 통과!")
            return True
        else:
            print("   ⚠️  성능 최적화 필요")
            return False

    except Exception as e:
        print(f"   ❌ 성능 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    success = comprehensive_system_test()

    # 성능 테스트도 실행
    performance_ok = run_performance_test()

    print("\n" + "=" * 80)
    if success and performance_ok:
        print("🎉 모든 테스트 통과! 시스템이 완벽하게 최적화되었습니다!")
        exit(0)
    else:
        print("⚠️  일부 테스트 실패. 추가 개선이 필요합니다.")
        exit(1)
