"""
패턴 벡터화 모듈

이 모듈은 패턴 분석 결과를 벡터화하여 ML/DL 모델 입력으로 사용할 수 있게 합니다.
"""

import numpy as np
import os
import logging
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime
from collections import Counter
import gc
import torch

from ..utils.error_handler_refactored import get_logger
from ..utils.unified_performance import performance_monitor
from ..utils.unified_config import ConfigProxy
from ..utils.normalizer import Normalizer
from ..shared.types import LotteryNumber, PatternAnalysis, PatternFeatures
from ..shared.graph_utils import calculate_segment_entropy

# type: ignore 주석 추가
# 로거 설정
logger = get_logger(__name__)


class PatternVectorizer:
    """
    패턴 벡터화기 (단순 래퍼)

    EnhancedPatternVectorizer를 래핑하는 단순한 인터페이스
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigProxy]] = None):
        """단순 래퍼 초기화"""
        # 🚨 중복 초기화 완전 방지
        if hasattr(self, "_wrapper_initialized"):
            return
        self._wrapper_initialized = True

        # 기본 설정
        self.config = config if config is not None else {}
        self.logger = get_logger(__name__)

        # 향상된 시스템 직접 연결 (단순화)
        try:
            from .enhanced_pattern_vectorizer import EnhancedPatternVectorizer

            self._enhanced = EnhancedPatternVectorizer(self.config)
            self.feature_names = self._enhanced.get_feature_names()
            self.vector_dimensions = len(self.feature_names)
            self.logger.info("✅ 래퍼 초기화 완료")
        except Exception as e:
            self.logger.error(f"래퍼 초기화 실패: {e}")
            self._enhanced = None
            self.feature_names = []
            self.vector_dimensions = 0

    def vectorize_full_analysis(self, full_analysis: Dict[str, Any]) -> np.ndarray:
        """전체 분석 결과를 벡터화 (래퍼)"""
        if self._enhanced:
            try:
                vector, names = self._enhanced.vectorize_extended_features(
                    full_analysis
                )
                return vector
            except Exception as e:
                self.logger.error(f"향상된 벡터화 실패: {e}")
                return np.zeros(168, dtype=np.float32)
        else:
            self.logger.warning("향상된 시스템 없음 - 영벡터 반환")
            return np.zeros(168, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """특성 이름 반환"""
        if self._enhanced:
            return self._enhanced.get_feature_names()
        return [f"feature_{i+1}" for i in range(168)]

    def save_vector_to_file(
        self,
        vector: np.ndarray,
        feature_names: List[str],
        filename: str = "feature_vector_full.npy",
    ) -> bool:
        """벡터를 파일로 저장"""
        try:
            from pathlib import Path
            import json

            cache_path = Path("data/cache")
            cache_path.mkdir(parents=True, exist_ok=True)

            # 벡터 저장
            vector_path = cache_path / filename
            np.save(vector_path, vector)

            # 특성 이름 저장
            names_filename = filename.replace(".npy", ".names.json")
            names_path = cache_path / names_filename
            with open(names_path, "w", encoding="utf-8") as f:
                json.dump(feature_names, f, ensure_ascii=False, indent=2)

            # 벡터 품질 정보
            zero_ratio = (vector == 0).sum() / len(vector) * 100

            # 엔트로피 계산 수정 (정규화된 방식)
            if len(vector) > 0:
                # 벡터를 확률 분포로 정규화
                vector_normalized = (
                    vector / np.sum(vector) if np.sum(vector) > 0 else vector
                )
                # 0이 아닌 값들에 대해서만 엔트로피 계산
                non_zero_mask = vector_normalized > 0
                if np.any(non_zero_mask):
                    entropy = -np.sum(
                        vector_normalized[non_zero_mask]
                        * np.log(vector_normalized[non_zero_mask])
                    )
                else:
                    entropy = 0.0
            else:
                entropy = 0.0

            self.logger.info(
                f"✅ 벡터 저장 완료: {vector_path} ({vector_path.stat().st_size:,} bytes)"
            )
            self.logger.info(f"   - 벡터 차원: {vector.shape}")
            self.logger.info(f"   - 데이터 타입: {vector.dtype}")
            self.logger.info(f"   - 특성 이름 수: {len(feature_names)}")
            self.logger.info(f"✅ 특성 이름 저장 완료: {names_path}")
            self.logger.info(f"📊 벡터 품질:")
            self.logger.info(f"   - 0값 비율: {zero_ratio:.1f}%")
            self.logger.info(f"   - 엔트로피: {entropy:.3f}")
            self.logger.info(f"   - 최솟값: {vector.min():.3f}")
            self.logger.info(f"   - 최댓값: {vector.max():.3f}")
            self.logger.info(f"   - 평균값: {vector.mean():.3f}")

            return True

        except Exception as e:
            self.logger.error(f"벡터 저장 실패: {e}")
            return False

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """설정 값 안전하게 가져오기"""
        try:
            keys = key.split(".")
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
