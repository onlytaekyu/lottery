#!/usr/bin/env python3
"""현재 벡터 상태 확인"""

import numpy as np
import json
from pathlib import Path


def check_current_vector_status():
    """현재 벡터 파일 상태 확인"""
    print("=" * 60)
    print("🔍 현재 벡터 파일 상태 분석")
    print("=" * 60)

    # 벡터 파일 확인
    vector_path = Path("data/cache/feature_vector_full.npy")
    names_path = Path("data/cache/feature_vector_full.names.json")

    if not vector_path.exists():
        print("❌ 벡터 파일이 존재하지 않습니다")
        return

    if not names_path.exists():
        print("❌ 특성 이름 파일이 존재하지 않습니다")
        return

    # 벡터 로드
    try:
        vector = np.load(vector_path)
        print(f"✅ 벡터 파일 로드 성공")
        print(f"   - 파일 크기: {vector_path.stat().st_size} bytes")
        print(f"   - 벡터 형태: {vector.shape}")
        print(f"   - 벡터 타입: {vector.dtype}")

        # 1차원인지 2차원인지 확인
        if vector.ndim == 1:
            vector_dim = len(vector)
            print(f"   - 벡터 차원: {vector_dim} (1차원 벡터)")
        else:
            vector_dim = vector.shape[1] if vector.shape[0] == 1 else vector.shape[0]
            print(f"   - 벡터 차원: {vector_dim} (다차원 벡터)")

    except Exception as e:
        print(f"❌ 벡터 로드 실패: {e}")
        return

    # 특성 이름 로드
    try:
        with open(names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        names_count = len(feature_names)
        print(f"✅ 특성 이름 파일 로드 성공")
        print(f"   - 파일 크기: {names_path.stat().st_size} bytes")
        print(f"   - 특성 이름 수: {names_count}")

    except Exception as e:
        print(f"❌ 특성 이름 로드 실패: {e}")
        return

    # 차원 일치성 확인
    print("\n📊 차원 일치성 분석:")
    if vector_dim == names_count:
        print(f"✅ 차원 일치: 벡터({vector_dim}) = 이름({names_count})")
    else:
        print(f"❌ 차원 불일치: 벡터({vector_dim}) ≠ 이름({names_count})")
        diff = abs(vector_dim - names_count)
        print(f"   - 차이: {diff}개 ({diff/max(vector_dim, names_count)*100:.1f}%)")

    # 벡터 품질 분석
    print("\n📈 벡터 품질 분석:")
    zero_count = np.sum(vector == 0.0)
    zero_ratio = zero_count / vector_dim
    print(f"   - 0값 개수: {zero_count}/{vector_dim} ({zero_ratio*100:.1f}%)")

    # 엔트로피 계산
    if vector_dim > 0:
        # 간단한 엔트로피 계산 (정규화된 값 기준)
        non_zero_values = vector[vector != 0]
        if len(non_zero_values) > 0:
            # 히스토그램 기반 엔트로피
            hist, _ = np.histogram(non_zero_values, bins=10)
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]  # 0 제거
            entropy = -np.sum(hist * np.log2(hist))
            print(f"   - 엔트로피: {entropy:.3f}")
        else:
            print(f"   - 엔트로피: 계산 불가 (모든 값이 0)")

    # 특성 이름 샘플 출력
    print("\n📝 특성 이름 샘플 (처음 10개):")
    for i, name in enumerate(feature_names[:10]):
        print(f"   {i+1:2d}. {name}")

    if len(feature_names) > 10:
        print(f"   ... (총 {len(feature_names)}개 중 10개만 표시)")

    # 필수 특성 확인
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

    print(f"\n🔍 필수 특성 확인 (총 {len(essential_features)}개):")
    missing_features = []
    for feature in essential_features:
        if feature in feature_names:
            print(f"   ✅ {feature}")
        else:
            print(f"   ❌ {feature} (누락)")
            missing_features.append(feature)

    if missing_features:
        print(f"\n⚠️  누락된 필수 특성: {len(missing_features)}개")
        for feature in missing_features:
            print(f"   - {feature}")
    else:
        print(f"\n✅ 모든 필수 특성이 존재합니다!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    check_current_vector_status()
