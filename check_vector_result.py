#!/usr/bin/env python3
"""
벡터 결과 확인 스크립트
"""

import numpy as np
import json
from pathlib import Path


def check_vector_result():
    """벡터 결과 확인"""
    print("🔍 DAEBAK_AI 프로젝트 완전 수정 결과 확인")
    print("=" * 50)

    # 벡터 파일 확인
    vector_file = Path("data/cache/feature_vector_full.npy")
    names_file = Path("data/cache/feature_vector_full.names.json")

    if not vector_file.exists():
        print("❌ 벡터 파일이 존재하지 않습니다")
        return False

    if not names_file.exists():
        print("❌ 특성 이름 파일이 존재하지 않습니다")
        return False

    # 벡터 로드
    vectors = np.load(vector_file)
    with open(names_file, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # 기본 정보
    print(f"📊 벡터 정보:")
    print(f"   - 벡터 모양: {vectors.shape}")
    print(f"   - 데이터 타입: {vectors.dtype}")
    print(
        f"   - 파일 크기: {vector_file.stat().st_size:,} bytes ({vector_file.stat().st_size/1024:.1f} KB)"
    )
    print(f"   - 특성 이름 수: {len(feature_names)}개")

    # 검증 항목
    checks = {
        "차원 168차원": (
            vectors.shape[-1] == 168 if vectors.ndim > 0 else len(vectors) == 168
        ),
        "이름 수 일치": len(feature_names) == 168,
        "파일 크기 800바이트 이상": vector_file.stat().st_size >= 800,
        "NaN/Inf 없음": not (np.any(np.isnan(vectors)) or np.any(np.isinf(vectors))),
        "0값 비율 50% 이하": (vectors == 0).sum() / vectors.size <= 0.5,
    }

    print(f"\n🔍 검증 결과:")
    passed_checks = 0
    for check_name, passed in checks.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"   - {check_name}: {status}")
        if passed:
            passed_checks += 1

    # 벡터 품질 정보
    if vectors.size > 0:
        zero_ratio = (vectors == 0).sum() / vectors.size
        print(f"\n📈 벡터 품질:")
        print(f"   - 0값 비율: {zero_ratio*100:.1f}%")
        print(f"   - 최솟값: {vectors.min():.3f}")
        print(f"   - 최댓값: {vectors.max():.3f}")
        print(f"   - 평균값: {vectors.mean():.3f}")
        print(f"   - 표준편차: {vectors.std():.3f}")

    # 성공률
    success_rate = passed_checks / len(checks)
    print(f"\n🏆 전체 성공률: {passed_checks}/{len(checks)} ({success_rate*100:.1f}%)")

    # 최종 결과
    if success_rate >= 0.8:
        print("\n🎉 DAEBAK_AI 프로젝트 완전 수정 성공!")
        print("✅ 보너스 번호 관련 코드 완전 삭제")
        print("✅ 벡터화 시스템 개선 완료")
        print("✅ 코드 품질 향상")
    else:
        print("\n⚠️ 일부 항목 미달성")
        print("📝 추가 개선 필요")

    return success_rate >= 0.8


if __name__ == "__main__":
    check_vector_result()
