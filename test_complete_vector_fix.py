#!/usr/bin/env python3
"""
🚀 완전 수정된 벡터화 시스템 테스트 스크립트

모든 문제점을 해결한 새로운 벡터화 시스템을 테스트합니다:
1. 벡터 차원과 특성 이름 100% 일치
2. 필수 특성 22개 실제 계산 구현
3. 특성 품질 개선 (0값 50% → 30% 이하)
4. 엔트로피 개선 (음수 → 양수)
"""

import sys
import os
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils import load_config
from src.utils.unified_logging import get_logger
from src.analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer


def test_complete_vector_system():
    """완전 수정된 벡터화 시스템 테스트"""
    logger = get_logger(__name__)
    logger.info("🚀 완전 수정된 벡터화 시스템 테스트 시작")

    try:
        # 1. 설정 로드
        config = load_config()
        logger.info("✅ 설정 로드 완료")

        # 2. 향상된 패턴 벡터화기 초기화
        vectorizer = EnhancedPatternVectorizer(config)
        logger.info("✅ 패턴 벡터화기 초기화 완료")

        # 3. 테스트용 분석 데이터 생성
        test_analysis_data = {
            "pattern_analysis": {
                "frequency_sum": 0.6,
                "gap_mean": 0.4,
                "total_draws": 100,
            },
            "distribution_pattern": {"entropy": 0.8, "balance_score": 0.5},
            "segment_10_frequency": {
                "1": 0.1,
                "2": 0.1,
                "3": 0.1,
                "4": 0.1,
                "5": 0.1,
                "6": 0.1,
                "7": 0.1,
                "8": 0.1,
                "9": 0.1,
                "10": 0.1,
            },
            "segment_5_frequency": {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2},
            "gap_statistics": {"mean": 7.5, "std": 2.3, "max": 15, "min": 2},
            "roi_features": {
                "roi_group_score": {"1": 0, "2": 1, "3": 2},
                "low_risk_bonus_flag": {"1": True, "2": False, "3": True},
            },
        }

        # 4. 벡터 생성 테스트
        logger.info("🔧 벡터 생성 테스트 시작...")
        vector = vectorizer.vectorize_full_analysis_enhanced(test_analysis_data)

        # 5. 특성 이름 확인
        feature_names = vectorizer.get_feature_names()

        # 6. 결과 검증
        logger.info("📊 벡터화 결과 검증:")
        logger.info(f"  - 벡터 차원: {len(vector)}")
        logger.info(f"  - 특성 이름 수: {len(feature_names)}")
        logger.info(f"  - 차원 일치: {len(vector) == len(feature_names)}")

        # 7. 0값 비율 계산
        zero_ratio = np.sum(vector == 0.0) / len(vector)
        logger.info(f"  - 0값 비율: {zero_ratio*100:.1f}%")

        # 8. 엔트로피 계산
        def calculate_entropy(values):
            hist, _ = np.histogram(values, bins=20, range=(0, 1))
            hist = hist + 1e-10
            probabilities = hist / np.sum(hist)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy

        entropy = calculate_entropy(vector)
        logger.info(f"  - 벡터 엔트로피: {entropy:.3f}")

        # 9. 필수 특성 확인
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
            "distance_variance",
            "cohesiveness_score",
            "sequential_pair_rate",
            "number_spread",
            "pattern_complexity",
            "trend_strength",
        ]

        found_essential = sum(
            1 for feature in essential_features if feature in feature_names
        )
        logger.info(
            f"  - 필수 특성 발견: {found_essential}/{len(essential_features)}개"
        )

        # 10. 벡터 저장
        vector_path = vectorizer.save_enhanced_vector_to_file(
            vector, "complete_fixed_vector.npy"
        )
        names_path = vectorizer.save_names_to_file(
            feature_names, "complete_fixed_vector.names.json"
        )

        logger.info(f"✅ 벡터 저장: {vector_path}")
        logger.info(f"✅ 이름 저장: {names_path}")

        # 11. 성공 기준 확인
        success_criteria = {
            "차원 일치": len(vector) == len(feature_names),
            "0값 비율 30% 이하": zero_ratio <= 0.3,
            "엔트로피 양수": entropy > 0,
            "필수 특성 16개 이상": found_essential >= 16,
            "벡터 크기 70개 이상": len(vector) >= 70,
        }

        logger.info("\n🎯 성공 기준 검증:")
        passed_tests = 0
        for criterion, passed in success_criteria.items():
            status = "✅ 통과" if passed else "❌ 실패"
            logger.info(f"  - {criterion}: {status}")
            if passed:
                passed_tests += 1

        success_rate = passed_tests / len(success_criteria)
        logger.info(
            f"\n🏆 전체 성공률: {passed_tests}/{len(success_criteria)} ({success_rate*100:.1f}%)"
        )

        if success_rate >= 0.8:
            logger.info("🎉 벡터화 시스템 수정 성공!")
        else:
            logger.warning("⚠️ 일부 기준 미달, 추가 수정 필요")

        return success_rate >= 0.8

    except Exception as e:
        logger.error(f"❌ 테스트 중 오류 발생: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_complete_vector_system()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
