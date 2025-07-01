# 1. 표준 라이브러리
import itertools
from typing import Any, Dict, List, Optional

# 2. 서드파티
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# 3. 프로젝트 내부
# (없음)


def lightgbm_features(
    df: pd.DataFrame, cat_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """LightGBM용 특성: 범주형 인코딩, 상호작용, 통계 집계"""
    out = df.copy()
    # 1. 범주형 인코딩
    if cat_cols:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        cat_encoded = enc.fit_transform(out[cat_cols])
        cat_names = enc.get_feature_names_out(cat_cols)
        out = out.drop(columns=cat_cols)
        out[cat_names] = cat_encoded
    # 2. 상호작용 특성 (쌍 곱)
    num_cols = out.select_dtypes(include=[np.number]).columns
    for a, b in itertools.combinations(num_cols, 2):
        out[f"{a}_x_{b}"] = out[a] * out[b]
    # 3. 통계 집계 (평균, 분산, 최댓값, 최솟값)
    out["row_mean"] = out[num_cols].mean(axis=1)
    out["row_std"] = out[num_cols].std(axis=1)
    out["row_max"] = out[num_cols].max(axis=1)
    out["row_min"] = out[num_cols].min(axis=1)
    return out


def tcn_lstm_time_features(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """LSTM/TCN용 시계열 특성: 이동평균, 계절성, 자기상관"""
    arr = np.asarray(arr)
    # 1. 이동평균
    ma = pd.Series(arr).rolling(window, min_periods=1).mean().values
    # 2. 지수평활
    ewma = pd.Series(arr).ewm(span=window, adjust=False).mean().values
    # 3. 계절성 (월, 분기)
    month = (np.arange(len(arr)) % 12) + 1
    quarter = ((month - 1) // 3) + 1
    # 4. 자기상관 (lag=1)
    acf1 = np.concatenate([[0], pd.Series(arr).autocorr(lag=1) * np.ones(len(arr) - 1)])
    # 합치기
    features = np.column_stack([arr, ma, ewma, month, quarter, acf1])
    return features


def autoencoder_normalized_features(
    arr: np.ndarray, clip_min: float = 0.0, clip_max: float = 1.0
) -> np.ndarray:
    """AutoEncoder용: [0,1] 정규화, 이상치 클리핑"""
    arr = np.asarray(arr)
    scaler = MinMaxScaler(feature_range=(clip_min, clip_max))
    arr_scaled = scaler.fit_transform(arr)
    arr_clipped = np.clip(arr_scaled, clip_min, clip_max)
    return arr_clipped
