import json
import numpy as np
from pathlib import Path


def main():
    # 경로 설정
    cache_dir = Path("data/cache")
    full_feature_path = cache_dir / "feature_vector_full.names.json"
    roi_feature_path = cache_dir / "roi_features_vector.names.json"

    # 전체 특성 이름 로드
    with open(full_feature_path, "r", encoding="utf-8") as f:
        full_features = json.load(f)

    # ROI 특성 이름 로드
    with open(roi_feature_path, "r", encoding="utf-8") as f:
        roi_features = json.load(f)

    # 전체 특성 수와 이름 출력
    print(f"전체 특성 수: {len(full_features)}")
    print(f"전체 특성 이름: {full_features}")
    print("\n")

    # ROI 특성 수와 이름 출력
    print(f"ROI 특성 수: {len(roi_features)}")
    print(f"ROI 특성 이름: {roi_features}")
    print("\n")

    # ROI 특성 중 전체 특성에 포함되지 않은 항목 확인
    missing_features = [f for f in roi_features if f not in full_features]
    print(f"전체 특성에 포함되지 않은 ROI 특성: {missing_features}")

    # 전체 특성에 포함된 ROI 특성 확인
    included_features = [f for f in roi_features if f in full_features]
    print(
        f"전체 특성에 포함된 ROI 특성 수: {len(included_features)}/{len(roi_features)}"
    )


if __name__ == "__main__":
    main()
