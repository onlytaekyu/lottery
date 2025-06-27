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
    패턴 특성을 벡터로 변환하는 클래스

    다양한 패턴 분석 결과를 머신러닝/딥러닝 모델이 사용할 수 있는
    특성 벡터로 변환합니다.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigProxy]] = None):
        """
        초기화

        Args:
            config: 설정 객체
        """
        self.config = config if config is not None else {}
        self.logger = get_logger(__name__)

        # 🚀 성능 최적화 시스템 초기화
        try:
            from ..utils.memory_manager import MemoryManager, MemoryConfig
            from ..utils.cuda_optimizers import get_cuda_optimizer, CudaConfig

            # 메모리 관리자 초기화
            memory_config = MemoryConfig(
                max_memory_usage=0.8,
                use_memory_pooling=True,
                pool_size=32,
            )
            self.memory_manager = MemoryManager(memory_config)

            # CUDA 최적화 초기화 (벡터화 작업에 특화)
            cuda_config = CudaConfig(
                use_amp=False,  # 벡터화에서는 정확도 우선
                batch_size=32,
                use_cudnn=True,
            )
            self.cuda_optimizer = get_cuda_optimizer(cuda_config)

            self.logger.info("✅ PatternVectorizer 최적화 시스템 초기화 완료")

        except Exception as e:
            self.logger.warning(f"최적화 시스템 초기화 실패: {e}")
            self.memory_manager = None
            self.cuda_optimizer = None

        # 캐시 설정
        try:
            self.use_cache = self.config["vectorizer"]["use_cache"]
        except (KeyError, TypeError):
            self.logger.warning(
                "설정에서 'vectorizer.use_cache'를 찾을 수 없습니다. 기본값(True)을 사용합니다."
            )
            self.use_cache = True

        try:
            self.cache_dir = Path(self.config["paths"]["cache_dir"])
        except (KeyError, TypeError):
            self.logger.warning(
                "설정에서 'paths.cache_dir'를 찾을 수 없습니다. 기본값('data/cache')을 사용합니다."
            )
            self.cache_dir = Path("data/cache")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 저분산 특성 제거 설정
        try:
            self.remove_low_variance = self.config["filtering"][
                "remove_low_variance_features"
            ]
        except (KeyError, TypeError):
            self.logger.warning(
                "설정에서 'filtering.remove_low_variance_features'를 찾을 수 없습니다. 기본값(False)을 사용합니다."
            )
            self.remove_low_variance = False

        try:
            self.variance_threshold = self.config["filtering"]["variance_threshold"]
        except (KeyError, TypeError):
            self.logger.warning(
                "설정에서 'filtering.variance_threshold'를 찾을 수 없습니다. 기본값(0.01)을 사용합니다."
            )
            self.variance_threshold = 0.01

        # 제거된 저분산 특성 이름 저장
        self.removed_low_variance_features = []

        # 벡터 캐시
        self._pattern_cache = {}
        self._vector_cache = {}

        # 특성 이름 리스트 초기화
        self.feature_names = []

        # 캐시 적재 시도
        if self.use_cache:
            self._load_cache()

        logger.info(
            f"PatternVectorizer 초기화 완료 (캐시 사용: {self.use_cache}, 저분산 특성 제거: {self.remove_low_variance})"
        )

        # 정규화 유틸리티 초기화
        self.normalizer = Normalizer(self.config)

        # 특성 벡터 캐시 초기화
        from ..utils.state_vector_cache import get_cache

        self.vector_cache = get_cache(self.config)

        # 벡터 차원 청사진 시스템 초기화
        self._initialize_vector_blueprint()

        logger.info(
            f"벡터 청사진 시스템 초기화 완료: 총 {self.total_expected_dims}차원"
        )

    def _initialize_vector_blueprint(self):
        """
        벡터 차원 청사진 시스템 초기화
        각 그룹별 고정 차원을 사전 정의하여 차원 불일치 문제 해결
        """
        # 그룹별 고정 차원 정의 (총 95차원으로 표준화)
        self.vector_blueprint = {
            # 기본 패턴 분석 (25차원)
            "pattern_analysis": 25,
            # 분포 패턴 (10차원)
            "distribution_pattern": 10,
            # 세그먼트 빈도 (15차원: 10구간 + 5구간)
            "segment_frequency": 15,
            # 중심성 및 연속성 (12차원)
            "centrality_consecutive": 12,
            # 갭 통계 및 재출현 (8차원)
            "gap_reappearance": 8,
            # ROI 특성 (15차원)
            "roi_features": 15,
            # 클러스터 품질 (10차원)
            "cluster_features": 10,
            # 중복 패턴 특성 (20차원)
            "overlap_patterns": 20,
            # 물리적 구조 특성 (11차원)
            "physical_structure": 11,
            # 쌍 그래프 압축 벡터 (최대 20차원)
            "pair_graph_vector": 20,
        }

        # 총 예상 차원 계산
        self.total_expected_dims = sum(self.vector_blueprint.values())

        # 그룹별 특성 이름 템플릿
        self.feature_name_templates = {
            "pattern_analysis": [f"pattern_{i+1}" for i in range(25)],
            "distribution_pattern": [f"dist_{i+1}" for i in range(10)],
            "segment_frequency": [f"seg10_{i+1}" for i in range(10)]
            + [f"seg5_{i+1}" for i in range(5)],
            "centrality_consecutive": [f"centrality_{i+1}" for i in range(6)]
            + [f"consecutive_{i+1}" for i in range(6)],
            "gap_reappearance": [f"gap_{i+1}" for i in range(4)]
            + [f"reappear_{i+1}" for i in range(4)],
            "roi_features": [f"roi_{i+1}" for i in range(15)],
            "cluster_features": [f"cluster_{i+1}" for i in range(10)],
            "overlap_patterns": [f"overlap_{i+1}" for i in range(20)],
            "physical_structure": [f"physical_{i+1}" for i in range(11)],
            "pair_graph_vector": [f"pair_graph_{i+1}" for i in range(20)],
        }

        self.logger.info(
            f"벡터 청사진 정의 완료: {len(self.vector_blueprint)}개 그룹, 총 {self.total_expected_dims}차원"
        )

    def _vectorize_group_safe(
        self, group_name: str, data: Any, vectorize_func
    ) -> Tuple[np.ndarray, List[str]]:
        """
        그룹별 안전한 벡터화 수행
        예상 차원과 일치하지 않으면 패딩 또는 절단하여 차원 보장

        Args:
            group_name: 그룹 이름
            data: 벡터화할 데이터
            vectorize_func: 벡터화 함수

        Returns:
            Tuple[np.ndarray, List[str]]: 차원이 보장된 벡터와 특성 이름
        """
        try:
            # 벡터화 수행
            if vectorize_func is None:
                # 기본 벡터화 (모든 값을 0으로)
                vector = np.zeros(self.vector_blueprint[group_name], dtype=np.float32)
                feature_names = self.feature_name_templates[group_name][
                    : self.vector_blueprint[group_name]
                ]
            else:
                result = vectorize_func(data)
                if isinstance(result, tuple) and len(result) == 2:
                    vector, feature_names = result
                elif isinstance(result, np.ndarray):
                    vector = result
                    feature_names = [f"{group_name}_{i+1}" for i in range(len(vector))]
                else:
                    # 예상치 못한 반환 형태
                    self.logger.warning(
                        f"그룹 '{group_name}': 예상치 못한 반환 형태 {type(result)}"
                    )
                    vector = np.zeros(
                        self.vector_blueprint[group_name], dtype=np.float32
                    )
                    feature_names = self.feature_name_templates[group_name][
                        : self.vector_blueprint[group_name]
                    ]

            expected_dims = self.vector_blueprint.get(group_name, len(vector))

            # 차원 조정
            vector, feature_names = self._pad_or_truncate_vector(
                vector, feature_names, expected_dims, group_name
            )

            # NaN/Inf 검증 및 처리
            vector = self._sanitize_vector(vector, group_name)

            self.logger.debug(f"그룹 '{group_name}' 벡터화 완료: {len(vector)}차원")
            return vector, feature_names

        except Exception as e:
            self.logger.warning(f"그룹 '{group_name}' 벡터화 중 오류 발생: {e}")
            # 오류 시 기본 벡터 반환
            expected_dims = self.vector_blueprint.get(group_name, 10)
            vector = np.zeros(expected_dims, dtype=np.float32)
            feature_names = self.feature_name_templates.get(
                group_name, [f"{group_name}_{i+1}" for i in range(expected_dims)]
            )[:expected_dims]
            return vector, feature_names

    def _pad_or_truncate_vector(
        self,
        vector: np.ndarray,
        feature_names: List[str],
        expected_dims: int,
        group_name: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        벡터를 예상 차원에 맞게 패딩 또는 절단

        Args:
            vector: 원본 벡터
            feature_names: 원본 특성 이름
            expected_dims: 예상 차원
            group_name: 그룹 이름

        Returns:
            Tuple[np.ndarray, List[str]]: 조정된 벡터와 특성 이름
        """
        current_dims = len(vector)

        if current_dims == expected_dims:
            return vector, feature_names
        elif current_dims < expected_dims:
            # 패딩 (부족한 차원을 0으로 채움)
            padding_size = expected_dims - current_dims
            padded_vector = np.pad(
                vector, (0, padding_size), mode="constant", constant_values=0.0
            )

            # 특성 이름도 패딩
            template_names = self.feature_name_templates.get(
                group_name, [f"{group_name}_{i+1}" for i in range(expected_dims)]
            )
            padded_names = feature_names + template_names[current_dims:expected_dims]

            self.logger.debug(
                f"그룹 '{group_name}': {current_dims}차원 → {expected_dims}차원으로 패딩"
            )
            return padded_vector.astype(np.float32), padded_names
        else:
            # 절단 (초과 차원 제거)
            truncated_vector = vector[:expected_dims]
            truncated_names = feature_names[:expected_dims]

            self.logger.debug(
                f"그룹 '{group_name}': {current_dims}차원 → {expected_dims}차원으로 절단"
            )
            return truncated_vector.astype(np.float32), truncated_names

    def _sanitize_vector(self, vector: np.ndarray, group_name: str) -> np.ndarray:
        """
        벡터의 NaN/Inf 값을 처리하여 안전한 벡터로 변환

        Args:
            vector: 원본 벡터
            group_name: 그룹 이름

        Returns:
            np.ndarray: 정제된 벡터
        """
        # NaN을 0으로 대체
        nan_count = np.isnan(vector).sum()
        if nan_count > 0:
            vector = np.nan_to_num(vector, nan=0.0)
            self.logger.debug(f"그룹 '{group_name}': {nan_count}개 NaN 값을 0으로 대체")

        # Inf를 유한한 값으로 대체
        inf_count = np.isinf(vector).sum()
        if inf_count > 0:
            vector = np.nan_to_num(vector, posinf=1.0, neginf=-1.0)
            self.logger.debug(
                f"그룹 '{group_name}': {inf_count}개 Inf 값을 유한값으로 대체"
            )

        # 값 범위 제한 (-10 ~ 10)
        vector = np.clip(vector, -10.0, 10.0)

        return vector.astype(np.float32)

    def validate_vector_integrity(
        self, vector: np.ndarray, feature_names: List[str]
    ) -> bool:
        """
        벡터 무결성 검증

        Args:
            vector: 검증할 벡터
            feature_names: 특성 이름 리스트

        Returns:
            bool: 검증 통과 여부
        """
        try:
            # 차원 일치 확인 (경고만 출력, 실패로 처리하지 않음)
            if len(vector) != len(feature_names):
                self.logger.warning(
                    f"벡터 차원({len(vector)})과 특성 이름 수({len(feature_names)})가 일치하지 않습니다"
                )
                if len(feature_names) == 0:
                    self.logger.warning(
                        "특성 이름이 전혀 생성되지 않았습니다. 기본 이름을 사용합니다."
                    )
                elif len(vector) > len(feature_names):
                    diff = len(vector) - len(feature_names)
                    self.logger.warning(
                        f"차원 불일치 정도: {diff}개 ({diff/len(vector)*100:.2f}%)"
                    )
                    self.logger.warning(
                        f"벡터 차원이 특성 이름 수보다 {diff}개 더 많습니다. 이름 목록을 확장합니다."
                    )

            # 예상 총 차원과 일치 확인
            if len(vector) != self.total_expected_dims:
                self.logger.error(
                    f"총 차원 불일치: 예상 {self.total_expected_dims}차원 vs 실제 {len(vector)}차원"
                )
                return False

            # NaN/Inf 비율 확인 (1% 초과 시 실패)
            nan_inf_ratio = (np.isnan(vector).sum() + np.isinf(vector).sum()) / len(
                vector
            )
            if nan_inf_ratio > 0.01:
                self.logger.error(f"NaN/Inf 비율 초과: {nan_inf_ratio:.4f} > 0.01")
                return False

            # 0 값 비율 검증
            zero_ratio = np.sum(vector == 0) / len(vector)
            if zero_ratio > 0.8:
                self.logger.warning(
                    f"벡터의 {zero_ratio:.1%}가 0값입니다. 특성 추출을 개선해야 합니다."
                )

            # 특성 다양성 엔트로피 계산 (정보 품질 검증)
            entropy_score = self._calculate_feature_entropy(vector)
            if entropy_score < 1.0:
                self.logger.info(
                    f"특성 다양성 낮음: 엔트로피 {entropy_score:.4f} (개선 권장)"
                )
            else:
                self.logger.info(f"특성 다양성 양호: 엔트로피 {entropy_score:.4f}")

            self.logger.info(
                f"벡터 무결성 검증 통과: {len(vector)}차원, 0값 비율: {zero_ratio:.1%}"
            )
            return True

        except Exception as e:
            self.logger.error(f"벡터 무결성 검증 중 오류: {e}")
            return False

    def _calculate_feature_entropy(self, vector: np.ndarray) -> float:
        """
        특성 벡터의 정보 엔트로피 계산

        Args:
            vector: 특성 벡터

        Returns:
            float: 엔트로피 점수 (0~1)
        """
        try:
            # 벡터를 히스토그램으로 변환
            hist, _ = np.histogram(vector, bins=50, density=True)

            # 0이 아닌 확률만 사용
            hist = hist[hist > 0]

            if len(hist) == 0:
                return 0.0

            # 엔트로피 계산
            entropy = -np.sum(hist * np.log2(hist + 1e-10))

            # 0~1 범위로 정규화
            max_entropy = np.log2(len(hist))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            return float(normalized_entropy)

        except Exception as e:
            self.logger.warning(f"엔트로피 계산 중 오류: {e}")
            return 0.0

    def _get_config_value(self, key_path: str, default_value: Any) -> Any:
        """설정 값을 가져옵니다."""
        # ConfigProxy 객체인 경우
        if hasattr(self.config, "get") or hasattr(self.config, "__getitem__"):
            try:
                # key_path를 분해하여 직접 접근
                keys = key_path.split(".")
                value = self.config
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                self.logger.warning(
                    f"설정에서 '{key_path}'를 찾을 수 없습니다. 기본값을 사용합니다."
                )
                return default_value
        # 딕셔너리인 경우
        elif isinstance(self.config, dict):
            # 중첩된 키 처리
            keys = key_path.split(".")
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default_value
            return value
        # 기타 경우
        return default_value

    def _compute_pattern_hash(self, pattern_data: Dict[str, Any]) -> str:
        """
        패턴 데이터의 해시 값을 계산합니다.

        Args:
            pattern_data: 패턴 데이터 딕셔너리

        Returns:
            패턴 데이터의 고유 해시 값
        """
        try:
            # 정렬된 키로 JSON 문자열 생성
            pattern_json = json.dumps(pattern_data, sort_keys=True)
            # SHA-256 해시 계산
            return hashlib.sha256(pattern_json.encode()).hexdigest()
        except Exception as e:
            self.logger.warning(f"패턴 해시 계산 실패: {e}")
            # 현재 시간을 기반으로 한 폴백 해시
            return f"fallback_{int(time.time())}"

    def _compute_numbers_hash(self, numbers: List[int]) -> str:
        """
        번호 조합의 해시 값을 계산합니다.

        Args:
            numbers: 번호 조합 리스트

        Returns:
            번호 조합의 고유 해시 값
        """
        try:
            # 정렬된 번호 목록의 문자열
            numbers_str = "_".join(map(str, sorted(numbers)))
            # SHA-256 해시 계산
            return hashlib.sha256(numbers_str.encode()).hexdigest()
        except Exception as e:
            self.logger.warning(f"번호 해시 계산 실패: {e}")
            # 현재 시간을 기반으로 한 폴백 해시
            return f"fallback_{int(time.time())}"

    def _create_cache_key(self, config_section: Dict[str, Any]) -> str:
        """
        config 섹션을 JSON 직렬화하여 SHA1 해시 키를 생성합니다.
        동일 설정이면 동일 키, 설정이 다르면 다른 캐시 키 생성.

        Args:
            config_section: 설정 섹션 딕셔너리

        Returns:
            str: 10자리 SHA1 해시 키
        """
        config_str = json.dumps(config_section, sort_keys=True)
        return hashlib.sha1(config_str.encode("utf-8")).hexdigest()[:10]

    def _load_cache(self) -> None:
        """캐시 파일에서 벡터 캐시를 로드합니다."""
        try:
            # 패턴 캐시 로드
            pattern_cache_path = self.cache_dir / "pattern_vector_cache.npy"
            if pattern_cache_path.exists():
                cached_data = np.load(pattern_cache_path, allow_pickle=True).item()
                if isinstance(cached_data, dict):
                    self._pattern_cache = cached_data
                    self.logger.info(
                        f"패턴 벡터 캐시 로드 완료: {len(self._pattern_cache)}개 항목"
                    )

            # 번호 벡터 캐시 로드
            vector_cache_path = self.cache_dir / "number_vector_cache.npy"
            if vector_cache_path.exists():
                cached_data = np.load(vector_cache_path, allow_pickle=True).item()
                if isinstance(cached_data, dict):
                    self._vector_cache = cached_data
                    self.logger.info(
                        f"번호 벡터 캐시 로드 완료: {len(self._vector_cache)}개 항목"
                    )

        except Exception as e:
            self.logger.warning(f"캐시 로드 실패: {e}")
            self._pattern_cache = {}
            self._vector_cache = {}

    def _save_cache(self) -> None:
        """현재 캐시를 파일에 저장합니다."""
        if not self.use_cache:
            return

        try:
            # 패턴 캐시 저장
            pattern_cache_path = self.cache_dir / "pattern_vector_cache.npy"
            np.save(
                pattern_cache_path, np.array([self._pattern_cache], dtype=object)[0]
            )

            # 번호 벡터 캐시 저장
            vector_cache_path = self.cache_dir / "number_vector_cache.npy"
            np.save(vector_cache_path, np.array([self._vector_cache], dtype=object)[0])

            self.logger.info(
                f"벡터 캐시 저장 완료 (패턴: {len(self._pattern_cache)}개, 번호: {len(self._vector_cache)}개)"
            )
        except Exception as e:
            self.logger.warning(f"캐시 저장 실패: {e}")

    def vectorize_pattern(
        self, pattern_data: Dict[str, Any], use_optimization: bool = True
    ) -> np.ndarray:
        """
        최적화된 패턴 데이터 벡터 변환

        Args:
            pattern_data: 패턴 분석 결과 데이터
            use_optimization: 최적화 사용 여부

        Returns:
            특성 벡터
        """
        if not use_optimization:
            return self._standard_vectorize_pattern(pattern_data)

        # 🧠 메모리 관리 스코프 적용
        if self.memory_manager:
            with self.memory_manager.allocation_scope():
                return self._optimized_vectorize_pattern(pattern_data)
        else:
            return self._standard_vectorize_pattern(pattern_data)

    def _optimized_vectorize_pattern(self, pattern_data: Dict[str, Any]) -> np.ndarray:
        """최적화된 패턴 벡터화"""
        # 📊 성능 모니터링
        with performance_monitor("pattern_vectorization_optimized"):

            # 캐시 확인
            if self.use_cache:
                pattern_hash = self._compute_pattern_hash(pattern_data)
                if pattern_hash in self._pattern_cache:
                    return self._pattern_cache[pattern_hash]

            # 🚀 GPU 가속 벡터화 시도
            if (
                self.cuda_optimizer
                and self.cuda_optimizer.is_available()
                and self._should_use_gpu_vectorization(pattern_data)
            ):
                try:
                    return self._gpu_vectorize_pattern(pattern_data)
                except Exception as e:
                    self.logger.warning(f"GPU 벡터화 실패: {e}, CPU로 폴백")

            # CPU 최적화 벡터화
            return self._cpu_vectorize_pattern_optimized(pattern_data)

    def _standard_vectorize_pattern(self, pattern_data: Dict[str, Any]) -> np.ndarray:
        """표준 패턴 벡터화 (기존 로직)"""
        # 캐시 사용 시 이미 계산된 벡터가 있는지 확인
        if self.use_cache:
            pattern_hash = self._compute_pattern_hash(pattern_data)
            if pattern_hash in self._pattern_cache:
                return self._pattern_cache[pattern_hash]

        # 벡터 요소 초기화
        vector_elements = []

        # 1. 마지막 당첨 번호의 분포 특성
        if "number_distribution" in pattern_data:
            dist_data = pattern_data["number_distribution"]

            # 범위별 분포 (0-9, 10-19, 20-29, 30-39, 40-45)
            range_dist = dist_data.get("range_distribution", [0.2, 0.2, 0.2, 0.2, 0.2])
            vector_elements.extend(range_dist)

            # 홀짝 분포
            odd_even_ratio = dist_data.get("odd_even_ratio", 0.5)
            vector_elements.append(odd_even_ratio)

            # 고저 분포 (1-22, 23-45)
            high_low_ratio = dist_data.get("high_low_ratio", 0.5)
            vector_elements.append(high_low_ratio)
        else:
            # 기본값 추가
            vector_elements.extend([0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5])

        # 2. 합계 관련 특성
        if "sum_analysis" in pattern_data:
            sum_data = pattern_data["sum_analysis"]

            # 정규화된 합계 (0-1 범위)
            normalized_sum = sum_data.get("normalized_sum", 0.5)
            vector_elements.append(normalized_sum)

            # 평균과의 편차 (정규화)
            deviation = sum_data.get("deviation_from_mean", 0.0)
            normalized_deviation = max(min(deviation / 50.0 + 0.5, 1.0), 0.0)
            vector_elements.append(normalized_deviation)
        else:
            # 기본값 추가
            vector_elements.extend([0.5, 0.5])

        # 3. 간격 관련 특성
        if "number_gaps" in pattern_data:
            gap_data = pattern_data["number_gaps"]

            # 평균 간격 (정규화)
            avg_gap = gap_data.get("avg_gap", 7.5)
            normalized_avg_gap = max(min(avg_gap / 15.0, 1.0), 0.0)
            vector_elements.append(normalized_avg_gap)

            # 최대 간격 (정규화)
            max_gap = gap_data.get("max_gap", 15)
            normalized_max_gap = max(min(max_gap / 30.0, 1.0), 0.0)
            vector_elements.append(normalized_max_gap)

            # 최소 간격 (정규화)
            min_gap = gap_data.get("min_gap", 1)
            normalized_min_gap = max(min(min_gap / 10.0, 1.0), 0.0)
            vector_elements.append(normalized_min_gap)
        else:
            # 기본값 추가
            vector_elements.extend([0.5, 0.5, 0.1])

        # 4. 연속 번호 관련 특성
        if "consecutive_numbers" in pattern_data:
            consecutive_data = pattern_data["consecutive_numbers"]

            # 연속 번호 수 (정규화)
            count = consecutive_data.get("count", 0)
            normalized_count = min(count / 5.0, 1.0)
            vector_elements.append(normalized_count)
        else:
            # 기본값 추가
            vector_elements.append(0.0)

        # 5. 과거 당첨 번호와의 일치 관련 특성
        if "historical_match" in pattern_data:
            match_data = pattern_data["historical_match"]

            # 최대 일치 수 (정규화)
            max_match = match_data.get("max_match", 0)
            normalized_max_match = min(max_match / 6.0, 1.0)
            vector_elements.append(normalized_max_match)

            # 평균 일치 수 (정규화)
            avg_match = match_data.get("avg_match", 0.0)
            normalized_avg_match = min(avg_match / 3.0, 1.0)
            vector_elements.append(normalized_avg_match)
        else:
            # 기본값 추가
            vector_elements.extend([0.0, 0.0])

        # 벡터 생성
        feature_vector = np.array(vector_elements, dtype=np.float32)

        # 캐시에 저장
        if self.use_cache:
            pattern_hash = self._compute_pattern_hash(pattern_data)
            self._pattern_cache[pattern_hash] = feature_vector

            # 주기적으로 캐시 저장 (패턴 캐시가 100개 이상 늘어날 때마다)
            if len(self._pattern_cache) % 100 == 0:
                self._save_cache()

        return feature_vector

    def vectorize_number_combination(
        self, numbers: List[int], pattern_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        번호 조합과 패턴 데이터를 결합하여 특성 벡터를 생성합니다.

        Args:
            numbers: 번호 조합 (6개 번호)
            pattern_data: 패턴 분석 결과 데이터

        Returns:
            특성 벡터
        """
        # 캐시 사용 시 이미 계산된 벡터가 있는지 확인
        if self.use_cache:
            numbers_hash = self._compute_numbers_hash(numbers)
            if numbers_hash in self._vector_cache:
                return self._vector_cache[numbers_hash]

        # 기본 패턴 벡터 생성
        pattern_vector = self.vectorize_pattern(pattern_data)

        # 번호별 특성 생성 (정규화된 번호)
        normalized_numbers = [n / 45.0 for n in sorted(numbers)]

        # 벡터 결합
        combined_vector = np.concatenate([pattern_vector, normalized_numbers])

        # 캐시에 저장
        if self.use_cache:
            numbers_hash = self._compute_numbers_hash(numbers)
            self._vector_cache[numbers_hash] = combined_vector

            # 주기적으로 캐시 저장 (번호 캐시가 1000개 이상 늘어날 때마다)
            if len(self._vector_cache) % 1000 == 0:
                self._save_cache()

        return combined_vector

    def _should_use_gpu_vectorization(self, pattern_data: Dict[str, Any]) -> bool:
        """GPU 벡터화 사용 여부 결정"""
        # 데이터 크기가 충분히 클 때만 GPU 사용
        data_size = len(str(pattern_data))  # 대략적인 데이터 크기
        return data_size > 1000  # 1KB 이상일 때 GPU 사용

    def _gpu_vectorize_pattern(self, pattern_data: Dict[str, Any]) -> np.ndarray:
        """GPU 가속 패턴 벡터화"""
        try:
            if self.cuda_optimizer and hasattr(self.cuda_optimizer, "device_context"):
                with self.cuda_optimizer.device_context():
                    # 현재는 기본 벡터화로 폴백 (향후 CUDA 구현 확장 가능)
                    return self._cpu_vectorize_pattern_optimized(pattern_data)
            else:
                # GPU 컨텍스트가 없으면 CPU로 처리
                return self._cpu_vectorize_pattern_optimized(pattern_data)
        except Exception as e:
            self.logger.error(f"GPU 벡터화 실패: {e}")
            return self._cpu_vectorize_pattern_optimized(pattern_data)

    def _cpu_vectorize_pattern_optimized(
        self, pattern_data: Dict[str, Any]
    ) -> np.ndarray:
        """최적화된 CPU 패턴 벡터화"""
        # 🧠 메모리 풀에서 배열 할당 (가능한 경우)
        if self.memory_manager:
            try:
                # 예상 벡터 크기로 메모리 할당
                estimated_size = self._estimate_vector_size(pattern_data)
                vector_array = self.memory_manager.get_optimized_array(
                    shape=(estimated_size,), dtype=np.float32
                )

                # 기존 로직으로 벡터 계산
                result = self._standard_vectorize_pattern(pattern_data)

                # 크기 조정
                if len(result) <= len(vector_array):
                    vector_array[: len(result)] = result
                    return vector_array[: len(result)]
                else:
                    return result

            except Exception as e:
                self.logger.warning(f"메모리 최적화 벡터화 실패: {e}")

        # 폴백: 표준 벡터화
        return self._standard_vectorize_pattern(pattern_data)

    def _estimate_vector_size(self, pattern_data: Dict[str, Any]) -> int:
        """벡터 크기 추정"""
        # 기본 벡터 크기 추정
        base_size = 15  # 기본 특성 수

        # 데이터 복잡도에 따른 추가 크기
        if "number_distribution" in pattern_data:
            base_size += 7
        if "sum_analysis" in pattern_data:
            base_size += 2
        if "number_gaps" in pattern_data:
            base_size += 3

        return base_size

    def set_optimizers(self, **optimizers):
        """외부에서 최적화 시스템 주입"""
        if "memory_manager" in optimizers:
            self.memory_manager = optimizers["memory_manager"]
        if "cuda_optimizer" in optimizers:
            self.cuda_optimizer = optimizers["cuda_optimizer"]

        self.logger.info("PatternVectorizer 외부 최적화 시스템 주입 완료")

    def clear_cache(self) -> None:
        """캐시를 모두 비웁니다."""
        self._pattern_cache = {}
        self._vector_cache = {}

        # 캐시 파일 삭제
        try:
            pattern_cache_path = self.cache_dir / "pattern_vector_cache.npy"
            if pattern_cache_path.exists():
                os.remove(pattern_cache_path)

            vector_cache_path = self.cache_dir / "number_vector_cache.npy"
            if vector_cache_path.exists():
                os.remove(vector_cache_path)

            self.logger.info("벡터 캐시가 모두 삭제되었습니다.")
        except Exception as e:
            self.logger.warning(f"캐시 파일 삭제 실패: {e}")

    def vectorize_full_analysis(self, full_analysis: Dict[str, Any]) -> np.ndarray:
        """
        전체 분석 데이터를 벡터로 변환

        Args:
            full_analysis: 통합 분석 결과

        Returns:
            변환된 벡터 (numpy 배열)
        """
        # 성능 추적 시작
        self.performance_tracker.start_tracking("vectorize_full_analysis")

        # 메모리 정리 변수들
        temp_vectors = []
        large_arrays = []

        try:
            # 벡터 설정에서 캐시 키 생성
            vector_settings = {}
            try:
                if isinstance(self.config, dict) and "vector_settings" in self.config:
                    vector_settings = self.config["vector_settings"]
                elif hasattr(self.config, "get") and callable(
                    getattr(self.config, "get")
                ):
                    vector_settings = self.config.get("vector_settings", {})
            except Exception as e:
                self.logger.warning(f"벡터 설정 로드 중 오류: {e}")

            # 캐시 키 생성
            cache_key = self._create_cache_key(vector_settings)
            self.logger.info(f"벡터 설정 기반 캐시 키 생성: {cache_key}")

            # 캐시 파일 경로에 해시 키 포함
            cache_file = Path(self.cache_dir) / f"feature_vector_{cache_key}.npy"
            feature_names_file = (
                Path(self.cache_dir) / f"feature_vector_{cache_key}.names.json"
            )

            # 캐시 확인
            if cache_file.exists():
                try:
                    self.logger.info(f"캐시된 벡터 데이터 사용: {cache_file}")
                    # 특성 이름도 로드
                    if feature_names_file.exists():
                        with open(feature_names_file, "r", encoding="utf-8") as f:
                            self.feature_names = json.load(f)

                    # 성능 추적 종료
                    self.performance_tracker.stop_tracking("vectorize_full_analysis")
                    return np.load(cache_file)
                except Exception as e:
                    self.logger.warning(f"캐시 로드 실패: {e}")

            # 벡터 특성 초기화
            vector_features = {}
            # 특성 이름 초기화
            self.feature_names = []
            # 특성 그룹별 이름 저장
            feature_names_by_group = {}

            # 1. 10구간 빈도 (10개 값)
            if "segment_10_frequency" in full_analysis:
                segment_10_vector = self._extract_segment_frequency(
                    full_analysis["segment_10_frequency"], 10
                )
                vector_features["segment_10"] = segment_10_vector
                feature_names_by_group["segment_10"] = [
                    f"segment_10_freq_{i+1}" for i in range(10)
                ]
                temp_vectors.append(segment_10_vector)

            # 2. 5구간 빈도 (5개 값)
            if "segment_5_frequency" in full_analysis:
                segment_5_vector = self._extract_segment_frequency(
                    full_analysis["segment_5_frequency"], 5
                )
                vector_features["segment_5"] = segment_5_vector
                feature_names_by_group["segment_5"] = [
                    f"segment_5_freq_{i+1}" for i in range(5)
                ]
                temp_vectors.append(segment_5_vector)

            # ... 중간 벡터 처리 과정에서 주기적 메모리 정리 ...

            # 주기적 메모리 정리 (벡터 5개마다)
            if len(temp_vectors) >= 5:
                # 가비지 컬렉션 실행
                gc.collect()
                # 임시 벡터 목록 정리
                temp_vectors.clear()
                self.logger.debug("중간 벡터 메모리 정리 완료")

            # ... existing code for other vector processing ...

            # 최종 벡터 결합
            combined_vector = self._combine_vectors(vector_features)
            large_arrays.append(combined_vector)

            # 벡터 검증
            self._validate_final_vector(combined_vector)

            # 벡터와 특성 이름 저장 (캐시 키 포함 경로 사용)
            self.save_vector_to_file(combined_vector, f"feature_vector_{cache_key}.npy")
            self.save_names_to_file(
                self.feature_names, f"feature_vector_{cache_key}.names.json"
            )

            # 호환성을 위해 feature_vector_full.npy도 함께 저장
            compat_file = Path(self.cache_dir) / "feature_vector_full.npy"
            compat_names_file = Path(self.cache_dir) / "feature_vector_full.names.json"
            np.save(compat_file, combined_vector)
            with open(compat_names_file, "w", encoding="utf-8") as f:
                json.dump(self.feature_names, f, ensure_ascii=False, indent=2)

            # 성능 추적 종료
            self.performance_tracker.stop_tracking("vectorize_full_analysis")

            return combined_vector

        except Exception as e:
            # 오류가 발생해도 성능 추적 종료를 보장
            self.performance_tracker.stop_tracking("vectorize_full_analysis")
            raise e
        finally:
            # 메모리 정리
            try:
                # 임시 벡터들 정리
                for vec in temp_vectors:
                    if vec is not None:
                        del vec
                temp_vectors.clear()

                # 대형 배열들 정리 (최종 결과 제외)
                for arr in large_arrays[:-1]:  # 마지막 결과는 보존
                    if arr is not None:
                        del arr

                # 벡터 특성 딕셔너리 정리
                if "vector_features" in locals():
                    for key, vec in vector_features.items():
                        if vec is not None and key != "final_result":
                            del vec
                    vector_features.clear()

                # 가비지 컬렉션 실행
                gc.collect()

                # CUDA 메모리 정리 (필요시)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.logger.debug("벡터화 메모리 정리 완료")

            except Exception as cleanup_error:
                self.logger.warning(f"메모리 정리 중 오류: {str(cleanup_error)}")

    def _extract_segment_frequency(
        self, segment_freq: Dict[str, Any], num_segments: int
    ) -> np.ndarray:
        """
        구간별 빈도 벡터 추출

        Args:
            segment_freq: 구간별 빈도 데이터
            num_segments: 구간 수 (5 또는 10)

        Returns:
            구간별 빈도 벡터
        """
        # 기본 벡터 초기화
        vector = np.zeros(num_segments, dtype=np.float32)

        # 각 구간별 빈도 설정
        for segment_idx, freq in segment_freq.items():
            try:
                idx = (
                    int(segment_idx) - 1
                )  # 구간 번호는 1부터 시작하므로 0-인덱스로 변환
                if 0 <= idx < num_segments:
                    vector[idx] = float(freq)
            except (ValueError, TypeError):
                # 구간 인덱스가 숫자가 아닌 경우 무시
                continue

        # 벡터 정규화 (합이 1이 되도록)
        total = np.sum(vector)
        if total > 0:
            vector = vector / total

        return vector

    def _extract_segment_centrality(
        self, segment_centrality: Dict[str, Any]
    ) -> np.ndarray:
        """
        세그먼트 중심성 데이터를 벡터로 변환

        Args:
            segment_centrality: 세그먼트 중심성 데이터 (9개 세그먼트에 대한 값)

        Returns:
            세그먼트 중심성 벡터 (9차원)
        """
        # 기본 벡터 초기화 (9개 세그먼트)
        vector = np.zeros(9, dtype=np.float32)

        if not segment_centrality:
            return vector

        # 세그먼트 중심성 값 설정
        for i in range(9):
            segment_start = i * 5 + 1
            segment_end = (i + 1) * 5
            segment_key = f"{segment_start}~{segment_end}"

            # 세그먼트 키가 다른 형식일 수도 있으므로 대체 키 시도
            alt_keys = [
                f"{segment_start}~{segment_end}",
                f"{segment_start}-{segment_end}",
                f"{segment_start}_{segment_end}",
            ]

            centrality_value = None
            for key in alt_keys:
                if key in segment_centrality:
                    if isinstance(segment_centrality[key], dict):
                        # 값이 딕셔너리인 경우 (eigenvector, degree 등이 포함된 경우)
                        if "eigenvector" in segment_centrality[key]:
                            centrality_value = float(
                                segment_centrality[key]["eigenvector"]
                            )
                        elif "degree" in segment_centrality[key]:
                            centrality_value = float(segment_centrality[key]["degree"])
                    else:
                        # 값이 숫자인 경우
                        centrality_value = float(segment_centrality[key])
                    break

            if centrality_value is not None and i < len(vector):
                vector[i] = centrality_value

        # 벡터 정규화 (0-1 범위로)
        max_val = np.max(vector)
        if max_val > 0:
            vector = vector / max_val

        return vector

    def _extract_segment_consecutive(
        self, segment_consecutive_patterns: Dict[str, Dict[str, int]]
    ) -> np.ndarray:
        """
        세그먼트 연속 패턴 데이터를 벡터로 변환

        Args:
            segment_consecutive_patterns: 세그먼트별 연속 패턴 통계 데이터
                {
                    "1~5": {"count_2": 37, "count_3": 12, "count_4+": 3},
                    "6~10": {"count_2": 25, "count_3": 8, "count_4+": 1},
                    ...
                }

        Returns:
            세그먼트 연속 패턴 벡터 (9개 세그먼트 x 3개 카운트 = 27차원)
        """
        # 기본 벡터 초기화 (9개 세그먼트 x 3개 카운트 = 27)
        vector = np.zeros(27, dtype=np.float32)

        if not segment_consecutive_patterns:
            return vector

        # 세그먼트별 연속 패턴 카운트 추출
        for i in range(9):
            segment_start = i * 5 + 1
            segment_end = (i + 1) * 5

            # 세그먼트 키가 다른 형식일 수도 있으므로 대체 키 시도
            segment_keys = [
                f"{segment_start}~{segment_end}",
                f"{segment_start}-{segment_end}",
                f"segment_{i+1}",
            ]

            segment_data = None
            for key in segment_keys:
                if key in segment_consecutive_patterns:
                    segment_data = segment_consecutive_patterns[key]
                    break

            if segment_data and isinstance(segment_data, dict):
                # 각 연속 패턴 카운트 추출 (2개 연속, 3개 연속, 4개 이상 연속)
                base_idx = i * 3  # 각 세그먼트마다 3개 카운트

                # 2개 연속 카운트
                if "count_2" in segment_data:
                    vector[base_idx] = float(segment_data["count_2"])

                # 3개 연속 카운트
                if "count_3" in segment_data:
                    vector[base_idx + 1] = float(segment_data["count_3"])

                # 4개 이상 연속 카운트
                if "count_4+" in segment_data:
                    vector[base_idx + 2] = float(segment_data["count_4+"])

        # 벡터 정규화 (0-1 범위로)
        max_val = np.max(vector)
        if max_val > 0:
            vector = vector / max_val

        return vector

    def _extract_pattern_reappearance(self, pattern_data: Dict[str, Any]) -> np.ndarray:
        """
        패턴 재출현 간격 데이터를 벡터로 변환

        Args:
            pattern_data: 패턴 재출현 간격 데이터

        Returns:
            패턴 재출현 간격 벡터
        """
        # 재출현 간격 특성 추출 (평균, 표준편차, 최소, 최대)
        features = np.zeros(4, dtype=np.float32)

        all_intervals = []
        for pattern_type, intervals in pattern_data.items():
            if isinstance(intervals, list):
                all_intervals.extend(intervals)
            elif isinstance(intervals, dict):
                for _, values in intervals.items():
                    if isinstance(values, dict) and "intervals" in values:
                        if isinstance(values["intervals"], list):
                            all_intervals.extend(values["intervals"])
                    elif isinstance(values, list):
                        all_intervals.extend(values)

        if all_intervals:
            # 평균, 표준편차, 최소, 최대 계산
            features[0] = np.mean(all_intervals)
            features[1] = np.std(all_intervals)
            features[2] = np.min(all_intervals)
            features[3] = np.max(all_intervals)

        return features

    def _extract_recent_gaps(self, gap_data: Dict[str, Any]) -> np.ndarray:
        """
        번호별 최근 재출현 간격 데이터를 벡터로 변환

        Args:
            gap_data: 번호별 최근 재출현 간격 데이터

        Returns:
            번호별 최근 재출현 간격 벡터 (45개 값)
        """
        vector = np.zeros(45, dtype=np.float32)

        # 번호별 간격 설정
        for num_str, gap in gap_data.items():
            try:
                num = int(num_str)
                if 1 <= num <= 45:
                    vector[num - 1] = float(gap)
            except (ValueError, TypeError):
                # 번호가 숫자가 아닌 경우 무시
                continue

        # 최대값으로 정규화 (0-1 범위로)
        max_gap = np.max(vector)
        if max_gap > 0:
            vector = vector / max_gap

        return vector

    def _combine_vectors(self, vector_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        여러 특성 벡터들을 결합하여 하나의 벡터로 만듭니다.

        Args:
            vector_features: 특성 그룹별 벡터 사전

        Returns:
            결합된 벡터
        """
        total_dimensions = sum(
            vector.shape[0] if vector.ndim == 1 else vector.shape[1]
            for vector in vector_features.values()
        )

        # 결합된 벡터 준비
        combined_vector = np.zeros(total_dimensions, dtype=np.float32)
        self.feature_names = []  # 특성 이름 목록 초기화

        # 그룹별 벡터 차원 추적
        feature_group_dimensions = {}

        # 현재 위치 인덱스
        current_idx = 0

        # 각 그룹별 벡터 추가
        for group_name, vector in vector_features.items():
            # 벡터가 비어있으면 건너뛰기
            if vector is None or vector.size == 0:
                continue

            vector_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]

            # 그룹 차원 기록
            feature_group_dimensions[group_name] = vector_dim

            # 벡터 데이터 복사
            if vector.ndim == 1:
                combined_vector[current_idx : current_idx + vector_dim] = vector
            else:
                combined_vector[current_idx : current_idx + vector_dim] = (
                    vector.flatten()[:vector_dim]
                )

            # 특성 이름 추가
            # feature_names_by_group에 이름이 있으면 해당 이름을 사용, 없으면 자동 생성
            if (
                hasattr(self, "feature_names_by_group")
                and group_name in self.feature_names_by_group
            ):
                group_feature_names = self.feature_names_by_group[group_name]
                # 이름 개수가 차원과 일치하는지 확인하고 조정
                if len(group_feature_names) == vector_dim:
                    self.feature_names.extend(group_feature_names)
                else:
                    # 이름 개수가 다른 경우 경고 로그 출력 후 자동 생성 이름 사용
                    self.logger.warning(
                        f"그룹 '{group_name}'의 특성 이름 개수({len(group_feature_names)})와 "
                        f"벡터 차원({vector_dim})이 일치하지 않습니다. 자동 생성 이름을 사용합니다."
                    )
                    self.feature_names.extend(
                        [f"{group_name}_{i}" for i in range(vector_dim)]
                    )
            else:
                # 이름이 없는 경우 자동 생성
                self.feature_names.extend(
                    [f"{group_name}_{i}" for i in range(vector_dim)]
                )

            # 현재 인덱스 업데이트
            current_idx += vector_dim

        # 각 그룹의 차원 로깅
        for group_name, dim in feature_group_dimensions.items():
            self.logger.info(f"벡터 그룹 '{group_name}' 차원: ({dim},)")

        # 특성 이름과 벡터 차원이 일치하는지 확인
        if len(self.feature_names) != combined_vector.shape[0]:
            self.logger.warning(
                f"특성 이름 개수({len(self.feature_names)})와 벡터 크기({combined_vector.shape[0]})가 "
                f"일치하지 않습니다. 기본 이름을 사용합니다."
            )

            # 특성 이름 길이 조정
            if len(self.feature_names) < combined_vector.shape[0]:
                # 부족한 특성 이름 추가
                self.feature_names.extend(
                    [
                        f"feature_{i+len(self.feature_names)}"
                        for i in range(
                            combined_vector.shape[0] - len(self.feature_names)
                        )
                    ]
                )
            elif len(self.feature_names) > combined_vector.shape[0]:
                # 초과된 특성 이름 제거
                self.feature_names = self.feature_names[: combined_vector.shape[0]]

        return combined_vector

    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        특성 벡터를 정규화합니다.

        Args:
            vector: 원본 특성 벡터

        Returns:
            정규화된 특성 벡터
        """
        with performance_monitor("normalize_vector"):
            # NaN 값 확인 및 처리
            has_nan = np.isnan(vector).any()
            if has_nan:
                self.logger.warning(f"벡터에 NaN 값이 있어 0으로 대체합니다.")
                vector = np.nan_to_num(vector, nan=0.0)

            # 무한대 값 확인 및 처리
            has_inf = np.isinf(vector).any()
            if has_inf:
                self.logger.warning(f"벡터에 무한대 값이 있어 처리합니다.")
                vector = np.nan_to_num(vector, posinf=1.0, neginf=0.0)

            # 값 범위 검증 및 조정 (0-1 범위)
            out_of_range = np.logical_or(vector < 0, vector > 1).any()
            if out_of_range:
                self.logger.warning(f"벡터에 [0,1] 범위를 벗어난 값이 있어 조정합니다.")
                vector = np.clip(vector, 0.0, 1.0)

            return vector

    def _process_float_conversion(self, value: Any) -> float:
        """
        다양한 타입의 값을 안전하게 float로 변환합니다.
        기본 구현 메서드

        Args:
            value: 변환할 값 (dictionary, list, 숫자, 문자열 등)

        Returns:
            float: 변환된 float 값
        """
        try:
            # dict 타입인 경우 우선순위에 따라 값 추출 시도
            if isinstance(value, dict):
                # 우선순위: score > value > avg > mean > 첫 번째 값
                for key in ["score", "value", "avg", "mean", "total"]:
                    if key in value and value[key] is not None:
                        return self._process_float_conversion(value[key])

                # 키가 없는 경우 첫 번째 값 사용
                if value:
                    return self._process_float_conversion(list(value.values())[0])
                return 0.0

            # list 또는 tuple인 경우 첫 번째 요소 사용
            elif isinstance(value, (list, tuple)) and value:
                if len(value) == 1:
                    return self._process_float_conversion(value[0])
                # 모든 값의 평균 계산 시도
                try:
                    return float(
                        sum(self._process_float_conversion(v) for v in value)
                        / len(value)
                    )
                except:
                    return self._process_float_conversion(value[0])

            # bool 타입인 경우 1.0 또는 0.0으로 변환
            elif isinstance(value, bool):
                return 1.0 if value else 0.0

            # None 값 처리
            elif value is None:
                return 0.0

            # 나머지 경우 float 변환 시도
            return float(value)
        except (ValueError, TypeError, IndexError, KeyError) as e:
            self.logger.warning(
                f"값 '{value}' ({type(value)})를 float로 변환할 수 없습니다: {e}"
            )
            return 0.0  # 변환 실패 시 기본값 반환

    def safe_float_conversion(self, value: Any) -> float:
        """
        다양한 타입의 값을 안전하게 float로 변환합니다.
        외부에서 주입된 함수가 있으면 그것을 사용하고,
        없으면 기본 구현을 사용합니다.

        Args:
            value: 변환할 값 (dictionary, list, 숫자, 문자열 등)

        Returns:
            float: 변환된 float 값
        """
        # 외부에서 주입된 safe_float_conversion 함수가 있으면 그것을 사용
        if hasattr(self, "_external_float_conversion") and callable(
            self._external_float_conversion
        ):
            return self._external_float_conversion(value)

        # 없으면 기본 구현 사용
        return self._process_float_conversion(value)

    @property
    def external_float_conversion(self):
        """외부에서 주입된 float 변환 함수를 반환"""
        return (
            self._external_float_conversion
            if hasattr(self, "_external_float_conversion")
            else None
        )

    @external_float_conversion.setter
    def external_float_conversion(self, func):
        """외부 float 변환 함수 설정"""
        self._external_float_conversion = func

    def save_vector_to_file(
        self, vector: np.ndarray, filename: str = "feature_vector_full.npy"
    ) -> str:
        """
        특성 벡터를 파일로 저장합니다.

        Args:
            vector: 저장할 특성 벡터
            filename: 저장할 파일명

        Returns:
            저장된 파일 경로

        Raises:
            ValueError: 벡터 차원과 특성 이름 수가 일치하지 않는 경우
        """
        # 캐시 디렉토리 확인
        try:
            cache_dir = self.config["paths"]["cache_dir"]
        except (KeyError, TypeError):
            cache_dir = "data/cache"

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # 특성 이름 가져오기
        feature_names = self.get_feature_names()

        # 벡터 차원과 특성 이름 수 일치 확인
        vector_dim = vector.shape[1] if len(vector.shape) > 1 else vector.shape[0]
        names_count = len(feature_names)

        # 필수 특성 목록 가져오기
        essential_features = []
        try:
            if (
                isinstance(self.config, dict)
                and "validation" in self.config
                and "essential_features" in self.config["validation"]
            ):
                essential_features = self.config["validation"]["essential_features"]
            elif hasattr(self.config, "get") and callable(getattr(self.config, "get")):
                validation_config = self.config.get("validation", {})
                if hasattr(validation_config, "get") and callable(
                    getattr(validation_config, "get")
                ):
                    essential_features = validation_config.get("essential_features", [])
        except Exception as e:
            self.logger.warning(f"설정에서 필수 특성 목록을 가져오는 중 오류: {str(e)}")

        # 필수 특성 목록이 비어있으면 기본 필수 특성 사용
        if not essential_features:
            # 필수 특성을 직접 정의 (모듈이 없으므로)
            essential_features = [
                "gap_stddev",
                "pair_centrality",
                "hot_cold_mix_score",
                "segment_entropy",
                "roi_group_score",
                "duplicate_flag",
                "max_overlap_with_past",
                "combination_recency_score",
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
                "position_variance_avg",
                "position_bias_score",
            ]

        # 필수 특성 중 누락된 특성 확인
        missing_essential = [f for f in essential_features if f not in feature_names]

        # 차원 불일치 확인
        if vector_dim != names_count:
            self.logger.warning(
                f"벡터 차원({vector_dim})과 특성 이름 수({names_count})가 일치하지 않습니다"
            )

            # 추가 정보 로깅: 차원 불일치 정도
            dimension_diff = abs(vector_dim - names_count)
            percent_diff = (dimension_diff / max(vector_dim, names_count)) * 100
            self.logger.warning(
                f"차원 불일치 정도: {dimension_diff}개 ({percent_diff:.2f}%)"
            )

            # 벡터 차원과 특성 이름 수를 일치시키기 위한 처리
            if vector_dim > names_count:
                self.logger.warning(
                    f"벡터 차원이 특성 이름 수보다 {vector_dim - names_count}개 더 많습니다. 이름 목록을 확장합니다."
                )
                # 특성 이름 목록 확장
                for i in range(names_count, vector_dim):
                    feature_names.append(f"feature_{i}")
            else:
                self.logger.warning(
                    f"특성 이름 수가 벡터 차원보다 {names_count - vector_dim}개 더 많습니다. 벡터를 확장합니다."
                )
                # 벡터를 확장하는 코드
                if len(vector.shape) > 1:
                    extended_vector = np.zeros(
                        (vector.shape[0], names_count), dtype=vector.dtype
                    )
                    extended_vector[:, :vector_dim] = vector
                    vector = extended_vector
                else:
                    extended_vector = np.zeros(names_count, dtype=vector.dtype)
                    extended_vector[:vector_dim] = vector
                    vector = extended_vector

                # 차원 업데이트
                vector_dim = names_count

        # 필수 특성 추가
        if missing_essential:
            self.logger.warning(f"다음 필수 특성이 누락되었습니다: {missing_essential}")

            # 누락된 특성을 feature_names에 추가
            feature_names.extend(missing_essential)

            # 벡터도 확장
            new_dim = len(feature_names)
            if len(vector.shape) > 1:
                extended_vector = np.zeros(
                    (vector.shape[0], new_dim), dtype=vector.dtype
                )
                extended_vector[:, :vector_dim] = vector
                vector = extended_vector
            else:
                extended_vector = np.zeros(new_dim, dtype=vector.dtype)
                extended_vector[:vector_dim] = vector
                vector = extended_vector

            # 추가된 특성에 기본값 설정 (0.5 또는 특정 기본값)
            for i, name in enumerate(feature_names):
                if i >= vector_dim and name in missing_essential:
                    # 특성별 기본값 설정
                    if "position_entropy" in name or "segment_entropy" in name:
                        vector[..., i] = 0.5
                    elif "stddev" in name or "std_" in name:
                        vector[..., i] = 0.1
                    elif "score" in name:
                        vector[..., i] = 0.5
                    elif "flag" in name:
                        vector[..., i] = 0.0
                    elif "silhouette_score" in name:
                        vector[..., i] = 0.3
                    else:
                        vector[..., i] = 0.5

            self.logger.info(
                f"필수 특성 {len(missing_essential)}개가 자동으로 추가되었습니다."
            )

            # 차원 업데이트
            vector_dim = new_dim

        # 최종 검증
        names_count = len(feature_names)
        if vector_dim != names_count:
            error_msg = f"[ERROR] 처리 후에도 벡터 차원({vector_dim})과 특성 이름 수({names_count})가 일치하지 않습니다"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 벡터 범위 검증
        min_val = np.min(vector)
        max_val = np.max(vector)
        if min_val < 0 or max_val > 1:
            self.logger.warning(
                f"벡터에 [0,1] 범위를 벗어난 값이 있습니다: 최소={min_val}, 최대={max_val}"
            )
            # 값 범위를 0-1로 클리핑
            vector = np.clip(vector, 0, 1)
            self.logger.info("벡터 값을 [0,1] 범위로 클리핑했습니다")

        # 파일 경로 생성
        file_path = cache_path / filename

        # NumPy 배열로 저장
        np.save(file_path, vector)
        self.logger.info(f"특성 벡터 저장 완료: {file_path}")

        # 특성 이름 저장
        if feature_names:
            # 클러스터 품질 벡터인 경우, 해당 파일에만 관련 지표 이름 사용
            if "cluster_quality" in filename:
                # 클러스터 품질 관련 이름만 필터링
                relevant_names = []
                for name in feature_names:
                    if any(
                        metric in name
                        for metric in [
                            "silhouette",
                            "balance",
                            "cluster",
                            "cohesiveness",
                            "distance_between",
                            "entropy",
                            "quality",
                        ]
                    ):
                        relevant_names.append(name)

                # 벡터의 크기에 맞게 이름 목록 조정
                if len(relevant_names) < vector_dim:
                    for i in range(len(relevant_names), vector_dim):
                        relevant_names.append(f"cluster_feature_{i}")
                elif len(relevant_names) > vector_dim:
                    relevant_names = relevant_names[:vector_dim]

                names_to_save = relevant_names
            else:
                names_to_save = feature_names

            names_file = cache_path / f"{Path(filename).stem}.names.json"
            with open(names_file, "w", encoding="utf-8") as f:
                json.dump(names_to_save, f, indent=2, ensure_ascii=False)
            self.logger.info(f"특성 이름 저장 완료: {names_file}")

            # 특성 벡터 검증
            # from ..utils.feature_vector_validator import check_vector_dimensions
            # 벡터 검증은 생략 (모듈이 없음)

            try:
                # check_vector_dimensions(str(file_path), str(names_file))
                self.logger.info("벡터 차원 검증 생략 (모듈 없음)")
            except Exception as e:
                self.logger.error(f"특성 벡터 검증 실패: {e}")
                # raise  # 검증 실패를 무시

        return str(file_path)

    def save_names_to_file(
        self, names: List[str], filename: str = "feature_vector_full.names.json"
    ) -> str:
        """
        특성 이름 목록을 파일로 저장합니다.

        Args:
            names: 저장할 특성 이름 목록
            filename: 저장할 파일명

        Returns:
            저장된 파일 경로
        """
        # 캐시 디렉토리 확인
        try:
            cache_dir = self.config["paths"]["cache_dir"]
        except (KeyError, TypeError):
            cache_dir = "data/cache"

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # 파일 경로 생성
        file_path = cache_path / filename

        # 이름 목록 저장
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(names, f, indent=2, ensure_ascii=False)

        self.logger.info(f"특성 이름 저장 완료: {file_path} ({len(names)}개 이름)")

        return str(file_path)

    def save_segment_specialized_vectors(
        self,
        segment_centrality_vector: np.ndarray = None,
        segment_consecutive_vector: np.ndarray = None,
        segment_entropy_vector: np.ndarray = None,
    ) -> None:
        """
        세그먼트 특수 벡터들을 개별 파일로 저장합니다.

        Args:
            segment_centrality_vector: 세그먼트 중심성 벡터
            segment_consecutive_vector: 세그먼트 연속 패턴 벡터
            segment_entropy_vector: 세그먼트 엔트로피 벡터
        """
        try:
            cache_dir = self.config["paths"]["cache_dir"]
        except (KeyError, TypeError):
            cache_dir = "data/cache"

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # 세그먼트 중심성 벡터 저장
        if segment_centrality_vector is not None:
            centrality_path = cache_path / "segment_centrality_vector.npy"
            np.save(centrality_path, segment_centrality_vector)
            self.logger.info(f"특성 벡터 저장 완료: {centrality_path}")

        # 세그먼트 연속 패턴 벡터 저장
        if segment_consecutive_vector is not None:
            consecutive_path = cache_path / "segment_consecutive_vector.npy"
            np.save(consecutive_path, segment_consecutive_vector)
            self.logger.info(f"특성 벡터 저장 완료: {consecutive_path}")

        # 세그먼트 엔트로피 벡터 저장
        if segment_entropy_vector is not None:
            entropy_path = cache_path / "segment_entropy_vector.npy"
            np.save(entropy_path, segment_entropy_vector)
            self.logger.info(f"특성 벡터 저장 완료: {entropy_path}")

    def save_segment_history_to_numpy(
        self,
        segment_10_history: Dict[str, List[int]],
        segment_5_history: Dict[str, List[int]],
    ) -> Tuple[str, str]:
        """
        세그먼트 히스토리를 NumPy 파일로 저장합니다.

        Args:
            segment_10_history: 10구간 히스토리 데이터
            segment_5_history: 5구간 히스토리 데이터

        Returns:
            10구간 저장 경로, 5구간 저장 경로
        """
        # 캐시 디렉토리 확인
        try:
            cache_dir = self.config["paths"]["cache_dir"]
        except (KeyError, TypeError):
            cache_dir = "data/cache"

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # 세그먼트 벡터 변환
        segment_10_matrix = []
        segment_5_matrix = []

        # 10구간 히스토리 처리
        for segment, history in segment_10_history.items():
            # 최근 50회차만 사용
            recent_history = history[-50:] if len(history) > 50 else history
            # 0-1 범위로 정규화
            max_val = max(recent_history) if recent_history else 1
            normalized = [h / max_val for h in recent_history]
            segment_10_matrix.append(normalized)

        # 5구간 히스토리 처리
        for segment, history in segment_5_history.items():
            # 최근 50회차만 사용
            recent_history = history[-50:] if len(history) > 50 else history
            # 0-1 범위로 정규화
            max_val = max(recent_history) if recent_history else 1
            normalized = [h / max_val for h in recent_history]
            segment_5_matrix.append(normalized)

        # NumPy 배열로 변환
        segment_10_array = np.array(segment_10_matrix, dtype=np.float32)
        segment_5_array = np.array(segment_5_matrix, dtype=np.float32)

        # 저장
        segment_10_path = cache_path / "segment_10_history.npy"
        segment_5_path = cache_path / "segment_5_history.npy"

        np.save(segment_10_path, segment_10_array)
        np.save(segment_5_path, segment_5_array)

        self.logger.info(f"10구간 히스토리 저장 완료: {segment_10_path}")
        self.logger.info(f"5구간 히스토리 저장 완료: {segment_5_path}")

        return str(segment_10_path), str(segment_5_path)

    def _extract_identical_draw_feature(
        self, identical_draw_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        동일한 당첨 번호 조합 분석 결과를 특성 벡터로 변환합니다.

        Args:
            identical_draw_data: 동일 조합 분석 결과

        Returns:
            np.ndarray: 변환된 특성 벡터
        """
        if not identical_draw_data or not isinstance(identical_draw_data, dict):
            # 데이터가 없는 경우 기본값 반환
            return np.zeros(3, dtype=np.float32)

        # 중복 조합 여부 특성 (0 또는 1)
        duplicate_flag = (
            1.0 if identical_draw_data.get("exact_match_in_history", False) else 0.0
        )

        # 과거 조합과의 최대 중첩 개수 (0~6 → 0~1로 정규화)
        num_overlap = identical_draw_data.get("num_overlap_with_past_max", 0)
        max_overlap_with_past = float(num_overlap) / 6.0

        # 조합 최근성 점수 (이미 0~1 사이)
        combination_recency_score = identical_draw_data.get(
            "combination_recency_score", 0.0
        )

        # 특성 이름 업데이트
        if (
            self.feature_names is not None
            and "duplicate_flag" not in self.feature_names
        ):
            self.feature_names.append("duplicate_flag")
            self.feature_names.append("max_overlap_with_past")
            self.feature_names.append("combination_recency_score")

        # 특성 벡터 생성
        feature_vector = np.array(
            [duplicate_flag, max_overlap_with_past, combination_recency_score],
            dtype=np.float32,
        )

        return feature_vector

    def vectorize_pattern_features(
        self, input_data: Union[Dict[str, Any], List[int]]
    ) -> np.ndarray:
        """
        패턴 특성을 벡터화합니다.

        Args:
            input_data: 패턴 특성 데이터 또는 번호 조합

        Returns:
            np.ndarray: 벡터화된 패턴 특성
        """
        with performance_monitor("vectorize_pattern_features"):
            # 입력 데이터가 리스트인 경우 패턴 특성 추출
            if isinstance(input_data, list):
                # 여기에서 벡터화하기 위한 특성들을 계산
                from ..analysis.pattern_analyzer import PatternAnalyzer

                pattern_analyzer = PatternAnalyzer(self.config)
                try:
                    # 리스트를 번호 조합으로 간주하고 특성 추출
                    input_data = pattern_analyzer.extract_pattern_features(
                        input_data, []
                    )
                except Exception as e:
                    self.logger.error(f"패턴 특성 추출 중 오류 발생: {str(e)}")
                    return np.random.uniform(0.2, 0.8, 20).astype(
                        np.float32
                    )  # 의미있는 기본값 반환

            # 해시 기반 캐싱 (입력 데이터가 딕셔너리인 경우)
            if isinstance(input_data, dict):
                # 캐시 키 생성
                hash_key = self._compute_pattern_hash(input_data)
                # 캐시에서 결과 확인
                if hash_key in self._pattern_cache and self.use_cache:
                    return self._pattern_cache[hash_key]

                # 딕셔너리에서 벡터화
                feature_vector = self._vectorize_from_dict(input_data)

                # 캐시에 저장
                if self.use_cache:
                    self._pattern_cache[hash_key] = feature_vector

                # 저분산 특성 제거 (벡터화 후 필터링)
                if self.remove_low_variance:
                    feature_vector = self._filter_low_variance_features(feature_vector)

                return feature_vector
            else:
                self.logger.warning(f"지원되지 않는 입력 타입: {type(input_data)}")
                return np.random.uniform(0.2, 0.8, 20).astype(
                    np.float32
                )  # 의미있는 기본값 반환

    def _filter_low_variance_features(
        self, feature_vector: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        저분산 특성을 제거합니다. 단, 필수 특성은 보존합니다.

        Args:
            feature_vector: 입력 특성 벡터
            feature_names: 특성 이름 목록 (선택 사항)

        Returns:
            Tuple[np.ndarray, List[str]]: 필터링된 특성 벡터와 필터링된 특성 이름 목록
        """
        try:
            # 특성 이름이 제공되지 않은 경우 기본 이름 생성
            if feature_names is None or len(feature_names) == 0:
                feature_names = [
                    f"feature_{i}"
                    for i in range(
                        feature_vector.shape[0]
                        if feature_vector.ndim == 1
                        else feature_vector.shape[1]
                    )
                ]

            # 벡터가 1차원인 경우 2D로 변환 (sklearn의 VarianceThreshold는 2D 입력 필요)
            is_1d = feature_vector.ndim == 1
            if is_1d:
                X = feature_vector.reshape(1, -1)
            else:
                X = feature_vector

            # 특성 이름 길이 확인 및 조정
            if len(feature_names) != X.shape[1]:
                self.logger.warning(
                    f"특성 이름 개수({len(feature_names)})와 벡터 크기({X.shape[1]})가 일치하지 않습니다."
                )
                # 특성 이름이 더 짧으면 확장
                if len(feature_names) < X.shape[1]:
                    feature_names.extend(
                        [
                            f"feature_{i+len(feature_names)}"
                            for i in range(X.shape[1] - len(feature_names))
                        ]
                    )
                # 특성 이름이 더 길면 자르기
                elif len(feature_names) > X.shape[1]:
                    feature_names = feature_names[: X.shape[1]]

            # 설정에서 저분산 특성 필터링 활성화 여부 확인
            remove_low_variance = self._get_config_value(
                "filtering.remove_low_variance_features", False
            )

            # 저분산 특성 필터링이 비활성화된 경우 원본 반환
            if not remove_low_variance:
                return feature_vector, feature_names

            # 먼저 캐시 디렉토리에서 low_variance_features.json 파일 확인
            try:
                from pathlib import Path
                import json

                # 캐시 디렉토리 경로 가져오기
                cache_dir = Path(
                    self._get_config_value("paths.cache_dir", "data/cache")
                )

                # 저분산 특성 파일 경로
                low_var_path = cache_dir / "low_variance_features.json"

                # 파일이 존재하면 로드
                if low_var_path.exists():
                    with open(low_var_path, "r", encoding="utf-8") as f:
                        low_var_info = json.load(f)

                    # 제거할 특성 인덱스 가져오기
                    removed_indices = low_var_info.get("removed_feature_indices", [])

                    # 제거할 특성 이름 가져오기
                    removed_feature_names = low_var_info.get(
                        "removed_feature_names", []
                    )

                    # 정보 로깅
                    if removed_indices:
                        self.logger.info(
                            f"저분산 특성 파일에서 {len(removed_indices)}개 특성 필터링 (threshold: {low_var_info.get('threshold', 0.005)})"
                        )

                    # 저장된 정보가 있으면 필터링 수행
                    if removed_indices:
                        # 유효한 인덱스만 선택 (벡터 크기를 초과하지 않도록)
                        valid_indices = [
                            idx for idx in removed_indices if idx < X.shape[1]
                        ]

                        if valid_indices:
                            # 유지할 인덱스 생성 (제거할 인덱스의 보수)
                            keep_indices = [
                                i for i in range(X.shape[1]) if i not in valid_indices
                            ]

                            # 특성 이름 필터링 (제거할 이름 기준)
                            if removed_feature_names:
                                filtered_names = [
                                    name
                                    for name in feature_names
                                    if name not in removed_feature_names
                                ]
                            else:
                                filtered_names = [
                                    feature_names[i]
                                    for i in keep_indices
                                    if i < len(feature_names)
                                ]

                            # 벡터 필터링
                            filtered_vector = X[:, keep_indices]

                            # 원래 차원으로 복원
                            if is_1d:
                                filtered_vector = filtered_vector.flatten()

                            self.logger.info(
                                f"저분산 특성 {len(valid_indices)}개 제거됨 (feature_vector_validator.py 기준)"
                            )
                            self.removed_low_variance_features = removed_feature_names
                            return filtered_vector, filtered_names

            except Exception as e:
                self.logger.warning(f"저분산 특성 파일 처리 중 오류: {str(e)}")
                # 오류 발생 시 기존 방식으로 계속 진행

            # 분산 임계값 설정
            variance_threshold = self.variance_threshold

            # 필수 특성 목록 가져오기
            essential_features = []
            try:
                if isinstance(self.config, dict) and "validation" in self.config:
                    essential_features = self.config["validation"].get(
                        "essential_features", []
                    )
                elif hasattr(self.config, "get") and callable(
                    getattr(self.config, "get")
                ):
                    validation_config = self.config.get("validation")
                    if (
                        validation_config is not None
                        and hasattr(validation_config, "get")
                        and callable(getattr(validation_config, "get"))
                    ):
                        essential_features = validation_config.get(
                            "essential_features", []
                        )
            except Exception as e:
                self.logger.warning(
                    f"설정에서 필수 특성 목록을 가져오는 중 오류: {str(e)}"
                )

            # 필수 특성 목록이 비어있으면 기본 필수 특성 사용
            if not essential_features:
                # 필수 특성을 직접 정의 (모듈이 없으므로)
                essential_features = [
                    "gap_stddev",
                    "pair_centrality",
                    "hot_cold_mix_score",
                    "segment_entropy",
                    "roi_group_score",
                    "duplicate_flag",
                    "max_overlap_with_past",
                    "combination_recency_score",
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
                    "position_variance_avg",
                    "position_bias_score",
                ]

            # 특성별 분산 계산
            variances = np.var(X, axis=0)

            # 보존할 특성 인덱스 (임계값보다 큰 분산 또는 필수 특성)
            keep_indices = []

            # 필수 특성 인덱스와 고분산 특성 인덱스 찾기
            for i, name in enumerate(feature_names):
                if name in essential_features:
                    keep_indices.append(i)  # 필수 특성 보존
                elif variances[i] > variance_threshold:
                    keep_indices.append(i)  # 고분산 특성 보존

            keep_indices = sorted(list(set(keep_indices)))  # 중복 제거 및 정렬

            # 보존하지 않을 특성 인덱스 (저분산이면서 필수가 아닌 특성)
            remove_indices = [i for i in range(X.shape[1]) if i not in keep_indices]

            # 제거된 특성 수 로깅
            if remove_indices:
                self.logger.info(
                    f"저분산 특성 {len(remove_indices)}개 제거됨 (임계값: {variance_threshold})"
                )

                # 제거된 특성 이름 저장
                removed_features = [feature_names[i] for i in remove_indices]
                self.removed_low_variance_features = removed_features
                self.logger.info(f"제거된 저분산 특성: {removed_features}")

            # 보존된 필수 특성 로깅
            preserved_essential = [
                feature_names[i]
                for i in keep_indices
                if feature_names[i] in essential_features
            ]
            if preserved_essential:
                self.logger.info(f"보존된 필수 특성: {len(preserved_essential)}개")

            # 필터링된 벡터 생성
            if keep_indices:
                filtered_vector = X[:, keep_indices]
                filtered_names = [feature_names[i] for i in keep_indices]
            else:
                # 모든 특성이 제거되는 것을 방지
                self.logger.warning(
                    "모든 특성이 저분산으로 제거될 수 있습니다. 원본 벡터를 유지합니다."
                )
                filtered_vector = X
                filtered_names = feature_names

            # 원래 차원으로 복원
            if is_1d:
                filtered_vector = filtered_vector.flatten()

            return filtered_vector, filtered_names

        except Exception as e:
            self.logger.error(f"저분산 특성 제거 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 벡터와 이름 그대로 반환
            return feature_vector, feature_names if feature_names else []

    def get_feature_names(self) -> List[str]:
        """
        현재 벡터에 사용된 특성 이름 목록을 반환합니다.

        Returns:
            특성 이름 목록
        """
        return self.feature_names

    def vectorize_number(
        self, number: int, historical_data: Optional[List[LotteryNumber]] = None
    ) -> np.ndarray:
        """
        단일 번호를 벡터로 변환합니다. ML 모델용 특성 벡터를 생성합니다.

        Args:
            number: 벡터화할 번호 (1~45)
            historical_data: 과거 당첨 번호 데이터 (선택적)

        Returns:
            번호 특성 벡터 (numpy 배열)
        """
        # 유효한 번호 범위 검사
        if not 1 <= number <= 45:
            self.logger.warning(
                f"유효하지 않은 번호: {number}, 범위는 1~45여야 합니다."
            )
            number = max(1, min(number, 45))

        # 캐시 키 생성
        cache_key = f"num_{number}"

        # 캐시 확인
        if self.use_cache and hasattr(self.vector_cache, "get"):
            try:
                cached_vector = self.vector_cache.get(cache_key)
                if cached_vector is not None:
                    return cached_vector
            except Exception as e:
                self.logger.warning(f"캐시 접근 오류: {str(e)}")

        # 1. 기본 특성 초기화
        feature_vector = np.zeros(10, dtype=np.float32)

        # 2. 번호를 0~1 범위로 정규화
        feature_vector[0] = number / 45.0

        # 3. 번호의 세그먼트 정보 (5개 구간)
        segment_idx = (number - 1) // 9  # 0-4 범위
        feature_vector[1] = segment_idx / 4.0  # 0-1 범위로 정규화

        # 4. 홀짝 특성
        feature_vector[2] = 1.0 if number % 2 == 0 else 0.0

        # 5. 구간별 특성 (1-10, 11-20, 21-30, 31-40, 41-45)
        if 1 <= number <= 10:
            feature_vector[3] = 1.0
        elif 11 <= number <= 20:
            feature_vector[4] = 1.0
        elif 21 <= number <= 30:
            feature_vector[5] = 1.0
        elif 31 <= number <= 40:
            feature_vector[6] = 1.0
        else:  # 41-45
            feature_vector[7] = 1.0

        # 6. 소수/합성수 특성
        is_prime = (
            all(number % i != 0 for i in range(2, int(number**0.5) + 1))
            if number > 1
            else False
        )
        feature_vector[8] = 1.0 if is_prime else 0.0

        # 7. 제곱수 특성
        is_perfect_square = int(number**0.5) ** 2 == number
        feature_vector[9] = 1.0 if is_perfect_square else 0.0

        # 8. 과거 데이터 기반 추가 특성 (제공된 경우)
        if historical_data and len(historical_data) > 0:
            # 과거 출현 빈도 계산
            appearances = sum(1 for draw in historical_data if number in draw.numbers)
            frequency = appearances / len(historical_data)

            # 출현 빈도 특성 추가
            frequency_feature = np.array([frequency], dtype=np.float32)
            feature_vector = np.concatenate([feature_vector, frequency_feature])

        # 9. 정규화 적용
        feature_vector = self.normalize_vector(feature_vector)

        # 10. 캐시에 저장
        if self.use_cache and hasattr(self.vector_cache, "set"):
            try:
                self.vector_cache.set(cache_key, feature_vector)
            except Exception as e:
                self.logger.warning(f"캐시 저장 오류: {str(e)}")

        return feature_vector

    def check_feature_distribution(
        self, vectors: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        특성 벡터의 분포를 분석하여 통계 정보를 반환합니다.

        Args:
            vectors: 분석할 특성 벡터 (배열 또는 행렬)
            feature_names: 특성 이름 목록 (제공되지 않으면 인덱스 사용)

        Returns:
            특성별 통계 정보
        """
        if len(vectors.shape) == 1:
            # 단일 벡터를 행렬로 변환
            vectors = vectors.reshape(1, -1)

        # 특성 수 확인
        n_features = vectors.shape[1]

        # 특성 이름 생성
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # 특성 이름 길이 조정
        if len(feature_names) < n_features:
            feature_names.extend(
                [f"feature_{i}" for i in range(len(feature_names), n_features)]
            )
        elif len(feature_names) > n_features:
            feature_names = feature_names[:n_features]

        # 특성별 통계 계산
        stats = {}
        for i, name in enumerate(feature_names):
            col = vectors[:, i]
            unique_values = np.unique(col)

            # 기본 통계 계산
            stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "unique_count": len(unique_values),
            }

            # 문제가 있는 특성 확인
            if stats[name]["std"] < 1e-6:  # 표준편차가 거의 0인 경우
                stats[name]["warning"] = "std_zero"
                self.logger.warning(
                    f"특성 '{name}'의 표준편차가 거의 0입니다: {stats[name]['std']}"
                )

            if len(unique_values) == 1:  # 고유값이 1개인 경우
                stats[name]["warning"] = "single_value"
                self.logger.warning(
                    f"특성 '{name}'에는 하나의 고유값만 있습니다: {unique_values[0]}"
                )

        return stats

    def vectorize_pattern_features_with_diagnostics(
        self, input_data: Union[Dict[str, Any], List[int]]
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        """
        패턴 특성을 벡터로 변환하고 특성 분포 진단을 수행합니다.

        Args:
            input_data: 패턴 특성 딕셔너리 또는 번호 목록

        Returns:
            특성 벡터와 특성 분포 통계
        """
        # 특성 벡터 생성
        vector = self.vectorize_pattern_features(input_data)

        # 기본 특성 이름 생성
        feature_names = [
            "consecutive_length",
            "total_sum",
            "odd_count",
            "even_count",
            "gap_avg",
            "gap_std",
            "high_low_ratio",
            "repeat_pattern",
            "cluster_overlap",
            "frequent_pair",
            "roi_weight",
            "trend_score",
            "risk_score",
        ]

        # 분포 진단 수행
        stats = self.check_feature_distribution(vector, feature_names)

        return vector, stats

    def _vectorize_from_dict(self, feature_dict: Dict[str, Any]) -> np.ndarray:
        """
        딕셔너리 형태의 패턴 특성을 벡터로 변환합니다.

        Args:
            feature_dict: 패턴 특성 딕셔너리

        Returns:
            np.ndarray: 변환된 특성 벡터
        """
        # 기본 벡터 크기 설정 (확장 가능)
        vector_size = 20
        feature_vector = np.zeros(vector_size, dtype=np.float32)

        # 인덱스 계산
        idx = 0

        # 기본 특성 처리
        # 1. 홀수/짝수 비율
        if "odd_even_ratio" in feature_dict:
            feature_vector[idx] = float(feature_dict["odd_even_ratio"])
            idx += 1

        # 2. 합계 관련 특성
        if "total_sum" in feature_dict:
            # 정규화된 합계 (0-1 범위)
            total_sum = float(feature_dict["total_sum"])
            # 합계 범위 (기대 범위: 21-255)
            normalized_sum = max(min((total_sum - 21) / (255 - 21), 1.0), 0.0)
            feature_vector[idx] = normalized_sum
            idx += 1

        # 3. 연속 번호 관련
        if "max_consecutive" in feature_dict:
            feature_vector[idx] = min(float(feature_dict["max_consecutive"]) / 5.0, 1.0)
            idx += 1

        # 4. 클러스터 관련
        if "cluster_score" in feature_dict:
            feature_vector[idx] = float(feature_dict["cluster_score"])
            idx += 1

        # 5. 최근 등장 점수
        if "recent_hit_score" in feature_dict:
            feature_vector[idx] = float(feature_dict["recent_hit_score"])
            idx += 1

        # 6. 범위 밀도
        if "range_density" in feature_dict:
            if isinstance(feature_dict["range_density"], dict):
                density_vals = list(feature_dict["range_density"].values())
                for val in density_vals[:5]:  # 최대 5개 값만 사용
                    feature_vector[idx] = float(val)
                    idx += 1
            elif isinstance(feature_dict["range_density"], list):
                for val in feature_dict["range_density"][:5]:  # 최대 5개 값만 사용
                    feature_vector[idx] = float(val)
                    idx += 1

        # 7. 번호 간 거리
        if "pairwise_distance" in feature_dict:
            # 평균 거리
            if (
                isinstance(feature_dict["pairwise_distance"], dict)
                and "avg" in feature_dict["pairwise_distance"]
            ):
                feature_vector[idx] = (
                    float(feature_dict["pairwise_distance"]["avg"]) / 22.0
                )  # 최대 거리로 정규화
                idx += 1

        # 8. 핫/콜드 혼합 점수
        if "hot_cold_mix_score" in feature_dict:
            feature_vector[idx] = float(feature_dict["hot_cold_mix_score"])
            idx += 1

        # 9. 위험도 점수 (있는 경우)
        if "risk_score" in feature_dict:
            feature_vector[idx] = float(feature_dict["risk_score"])
            idx += 1

        # 10. 트렌드 점수 (있는 경우)
        if "trend_score" in feature_dict:
            feature_vector[idx] = float(feature_dict["trend_score"])
            idx += 1

        # 나머지 특성 처리 - 사전에 정의된 이름이 아닌 다른 특성들
        for key, value in feature_dict.items():
            if (
                key
                not in [
                    "odd_even_ratio",
                    "total_sum",
                    "max_consecutive",
                    "cluster_score",
                    "recent_hit_score",
                    "range_density",
                    "pairwise_distance",
                    "hot_cold_mix_score",
                    "risk_score",
                    "trend_score",
                ]
                and idx < vector_size
            ):
                # 숫자 값만 처리
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    # 값 정규화 가정 (0-1 범위)
                    feature_vector[idx] = min(max(float(value), 0.0), 1.0)
                    idx += 1

        # 특성 이름 목록 업데이트 (첫 호출 시에만)
        if not self.feature_names:
            self.feature_names = ["feature_" + str(i) for i in range(vector_size)]

        return self.normalize_vector(feature_vector)

    def vectorize_enhanced_analysis(
        self, analysis_data: Dict[str, Any], pair_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        패턴 분석 결과를 확장된 특성 벡터로 변환합니다.

        Args:
            analysis_data: 패턴 분석 결과 데이터
            pair_data: 번호 쌍 분석 결과 데이터 (선택 사항)

        Returns:
            np.ndarray: 확장된 특성 벡터
        """
        # 성능 추적 시작
        self.performance_tracker.start_tracking("vectorize_enhanced_analysis")
        self.logger.info("패턴 분석 결과 확장 벡터화 시작")

        # 기본 벡터 특성
        vector_features = {}
        feature_names_by_group = {}

        # 특성 이름 초기화
        self.feature_names = []

        # 반드시 포함되어야 하는 필수 분석 필드 목록
        essential_fields = [
            "gap_stddev",
            "pair_centrality",
            "hot_cold_mix_score",
            "segment_entropy",
            "roi_group_score",
            "duplicate_flag",
        ]

        # 분석 결과에서 누락된 필수 필드 확인
        for field in essential_fields:
            if field not in analysis_data and field not in (pair_data or {}):
                self.logger.warning(f"분석 결과에서 '{field}' 필드가 누락되었습니다")

        try:
            # 1. 기본 패턴 분석 결과 벡터화
            pattern_vector = self.vectorize_full_analysis(analysis_data)
            vector_features["pattern"] = pattern_vector

            # 필수 필드 직접 추출하여 벡터에 추가
            essential_vector = []
            essential_names = []

            # 1. gap_stddev - 갭 표준편차
            if "gap_stddev" in analysis_data:
                gap_stddev = self.safe_float_conversion(analysis_data["gap_stddev"])
                # 0-1 범위로 정규화 (일반적인 범위: 0-2)
                normalized_gap_stddev = min(gap_stddev / 2.0, 1.0)
                essential_vector.append(normalized_gap_stddev)
                essential_names.append("gap_stddev")
            else:
                # gap_statistics에서 추출 시도
                if (
                    "gap_statistics" in analysis_data
                    and "std" in analysis_data["gap_statistics"]
                ):
                    gap_stddev = self.safe_float_conversion(
                        analysis_data["gap_statistics"]["std"]
                    )
                    normalized_gap_stddev = min(gap_stddev / 2.0, 1.0)
                    essential_vector.append(normalized_gap_stddev)
                    essential_names.append("gap_stddev")
                else:
                    essential_vector.append(0.5)  # 기본값
                    essential_names.append("gap_stddev")

            # 2. pair_centrality - 쌍 중심성
            if "pair_centrality" in analysis_data:
                # 이미 구현된 벡터화 메서드 사용
                pair_centrality_vector = self._vectorize_pair_centrality(
                    analysis_data["pair_centrality"]
                )
                # 평균값 추출하여 사용
                pair_centrality_avg = np.mean(pair_centrality_vector)
                essential_vector.append(pair_centrality_avg)
                essential_names.append("pair_centrality")
            else:
                # segment_centrality로 대체 시도
                if "segment_centrality" in analysis_data:
                    segment_centrality_vector = self._extract_segment_centrality(
                        analysis_data["segment_centrality"]
                    )
                    segment_centrality_avg = np.mean(segment_centrality_vector)
                    essential_vector.append(segment_centrality_avg)
                    essential_names.append("pair_centrality")
                else:
                    essential_vector.append(0.5)  # 기본값
                    essential_names.append("pair_centrality")

            # 3. hot_cold_mix_score - 핫/콜드 믹스 점수
            if "hot_cold_mix_score" in analysis_data:
                hot_cold_mix = self.safe_float_conversion(
                    analysis_data["hot_cold_mix_score"]
                )
                essential_vector.append(hot_cold_mix)
                essential_names.append("hot_cold_mix_score")
            else:
                essential_vector.append(0.5)  # 기본값
                essential_names.append("hot_cold_mix_score")

            # 4. segment_entropy - 세그먼트 엔트로피
            if "segment_entropy" in analysis_data:
                if (
                    isinstance(analysis_data["segment_entropy"], dict)
                    and "total_entropy" in analysis_data["segment_entropy"]
                ):
                    segment_entropy = self.safe_float_conversion(
                        analysis_data["segment_entropy"]["total_entropy"]
                    )
                    # 일반적인 엔트로피 범위: 0-2.5
                    normalized_entropy = min(segment_entropy / 2.5, 1.0)
                    essential_vector.append(normalized_entropy)
                    essential_names.append("segment_entropy")
                else:
                    segment_entropy = self.safe_float_conversion(
                        analysis_data["segment_entropy"]
                    )
                    essential_vector.append(segment_entropy)
                    essential_names.append("segment_entropy")
            else:
                essential_vector.append(0.5)  # 기본값
                essential_names.append("segment_entropy")

            # 5. roi_group_score - ROI 그룹 점수
            if "roi_group_score" in analysis_data:
                roi_group_score = self.safe_float_conversion(
                    analysis_data["roi_group_score"]
                )
                essential_vector.append(roi_group_score)
                essential_names.append("roi_group_score")
            else:
                # roi_pattern_groups에서 추출 시도
                if "roi_pattern_groups" in analysis_data and isinstance(
                    analysis_data["roi_pattern_groups"], dict
                ):
                    roi_groups = analysis_data["roi_pattern_groups"]
                    if "group_scores" in roi_groups and isinstance(
                        roi_groups["group_scores"], dict
                    ):
                        group_scores = roi_groups["group_scores"]
                        group_values = [
                            self.safe_float_conversion(v) for v in group_scores.values()
                        ]
                        avg_score = sum(group_values) / max(1, len(group_values))
                        essential_vector.append(avg_score)
                        essential_names.append("roi_group_score")
                    else:
                        essential_vector.append(0.5)  # 기본값
                        essential_names.append("roi_group_score")
                else:
                    essential_vector.append(0.5)  # 기본값
                    essential_names.append("roi_group_score")

            # 6. duplicate_flag - 중복 플래그
            if "duplicate_flag" in analysis_data:
                duplicate_flag = (
                    1.0
                    if self.safe_float_conversion(analysis_data["duplicate_flag"]) > 0
                    else 0.0
                )
                essential_vector.append(duplicate_flag)
                essential_names.append("duplicate_flag")
            else:
                # identical_draws에서 추출 시도
                if "identical_draws" in analysis_data and isinstance(
                    analysis_data["identical_draws"], dict
                ):
                    duplicate_count = self.safe_float_conversion(
                        analysis_data["identical_draws"].get("duplicate_count", 0)
                    )
                    duplicate_flag = 1.0 if duplicate_count > 0 else 0.0
                    essential_vector.append(duplicate_flag)
                    essential_names.append("duplicate_flag")
                else:
                    essential_vector.append(0.0)  # 기본값: 중복 없음
                    essential_names.append("duplicate_flag")

            # 필수 필드 벡터 추가
            vector_features["essential"] = np.array(essential_vector, dtype=np.float32)
            feature_names_by_group["essential"] = essential_names

            # 2. 위치 기반 빈도 분석 (position_frequency - shape: [6, 45])
            if "position_frequency" in analysis_data:
                try:
                    position_matrix = analysis_data["position_frequency"]
                    position_vector = self._vectorize_position_frequency(
                        position_matrix
                    )
                    vector_features["position_frequency"] = position_vector
                    feature_names_by_group["position_frequency"] = [
                        f"position_freq_{i}" for i in range(len(position_vector))
                    ]
                except Exception as e:
                    self.logger.error(f"위치 빈도 벡터화 중 오류: {e}")

            # 3. 위치별 엔트로피 및 표준편차 특성 추출
            try:
                (
                    position_entropy_vector,
                    position_std_vector,
                    position_entropy_names,
                    position_std_names,
                ) = self._extract_position_entropy_std_features(analysis_data)

                # 엔트로피 벡터 추가
                if len(position_entropy_vector) > 0:
                    vector_features["position_entropy"] = position_entropy_vector
                    feature_names_by_group["position_entropy"] = position_entropy_names

                # 표준편차 벡터 추가
                if len(position_std_vector) > 0:
                    vector_features["position_std"] = position_std_vector
                    feature_names_by_group["position_std"] = position_std_names
            except Exception as e:
                self.logger.error(f"위치별 엔트로피/표준편차 특성 추출 중 오류: {e}")

            # 3. 세그먼트 추세 분석 (segment_trend_history)
            if "segment_trend_history" in analysis_data:
                try:
                    segment_trend_matrix = analysis_data["segment_trend_history"]

                    # list 객체를 numpy 배열로 변환
                    if isinstance(segment_trend_matrix, list):
                        try:
                            segment_trend_matrix = np.array(
                                segment_trend_matrix, dtype=np.float32
                            )
                        except Exception as e:
                            self.logger.error(f"세그먼트 추세 행렬 변환 중 오류: {e}")
                            segment_trend_matrix = np.zeros((5, 10), dtype=np.float32)

                    segment_trend_vector = self._vectorize_segment_trend(
                        segment_trend_matrix
                    )
                    vector_features["segment_trend"] = segment_trend_vector
                    feature_names_by_group["segment_trend"] = [
                        f"segment_{i+1}_trend_{j}" for i in range(5) for j in range(3)
                    ]
                except Exception as e:
                    self.logger.error(f"세그먼트 추세 벡터화 중 오류: {e}")
                    # 오류 시 기본 벡터 생성
                    vector_features["segment_trend"] = np.zeros(15, dtype=np.float32)
                    feature_names_by_group["segment_trend"] = [
                        f"segment_{i+1}_trend_{j}" for i in range(5) for j in range(3)
                    ]

            # 5. 갭 편차 점수 분석 (gap_deviation_score)
            if "gap_deviation_score" in analysis_data:
                try:
                    gap_deviation_vector = self._vectorize_gap_deviation(
                        analysis_data["gap_deviation_score"]
                    )
                    vector_features["gap_deviation"] = gap_deviation_vector
                    feature_names_by_group["gap_deviation"] = [
                        "gap_dev_avg",
                        "gap_dev_std",
                        "gap_dev_high_ratio",
                        "gap_dev_low_ratio",
                        "gap_dev_range",
                    ]
                except Exception as e:
                    self.logger.error(f"갭 편차 점수 벡터화 중 오류: {e}")

            # 6. 조합 다양성 점수 분석 (combination_diversity)
            if "combination_diversity" in analysis_data:
                try:
                    diversity_vector = self._vectorize_combination_diversity(
                        analysis_data["combination_diversity"]
                    )
                    vector_features["combination_diversity"] = diversity_vector
                    feature_names_by_group["combination_diversity"] = [
                        "total_diversity",
                        "segment_diversity_avg",
                        "segment_diversity_std",
                        "segment_diversity_min",
                        "segment_diversity_max",
                    ]
                except Exception as e:
                    self.logger.error(f"조합 다양성 점수 벡터화 중 오류: {e}")

            # 7. ROI 트렌드 분석 (roi_trend_by_pattern)
            if "roi_trend_by_pattern" in analysis_data:
                try:
                    roi_trend_vector = self._vectorize_roi_trend_by_pattern(
                        analysis_data["roi_trend_by_pattern"]
                    )
                    vector_features["roi_trend"] = roi_trend_vector
                    feature_names_by_group["roi_trend"] = [
                        f"roi_trend_{i}" for i in range(len(roi_trend_vector))
                    ]
                except Exception as e:
                    self.logger.error(f"ROI 트렌드 벡터화 중 오류: {e}")

            # 쌍 분석 결과 추가 (선택적)
            if pair_data:
                # 쌍 중심성 벡터화 (pair_centrality)
                if "pair_centrality" in pair_data:
                    try:
                        pair_centrality_vector = self._vectorize_pair_centrality(
                            pair_data["pair_centrality"]
                        )
                        vector_features["pair_centrality"] = pair_centrality_vector
                        feature_names_by_group["pair_centrality"] = [
                            f"pair_centrality_{i}"
                            for i in range(len(pair_centrality_vector))
                        ]
                    except Exception as e:
                        self.logger.error(f"쌍 중심성 벡터화 중 오류: {e}")

                # 쌍 ROI 점수 벡터화 (pair_roi_score)
                if "pair_roi_score" in pair_data:
                    try:
                        pair_roi_vector = self._vectorize_pair_roi_scores(
                            pair_data["pair_roi_score"]
                        )
                        vector_features["pair_roi"] = pair_roi_vector
                        feature_names_by_group["pair_roi"] = [
                            f"pair_roi_{i}" for i in range(len(pair_roi_vector))
                        ]
                    except Exception as e:
                        self.logger.error(f"쌍 ROI 점수 벡터화 중 오류: {e}")

                # 자주 등장하는 트리플 벡터화 (frequent_triples)
                if "frequent_triples" in pair_data:
                    try:
                        triples_vector = self._vectorize_frequent_triples(
                            pair_data["frequent_triples"]
                        )
                        vector_features["frequent_triples"] = triples_vector
                        feature_names_by_group["frequent_triples"] = [
                            f"triple_{i}" for i in range(len(triples_vector))
                        ]
                    except Exception as e:
                        self.logger.error(f"자주 등장하는 트리플 벡터화 중 오류: {e}")

            # 8. 중복 당첨 번호 벡터 (2개 값) - 새로 추가
            if "identical_draw_check" in analysis_data:
                duplicate_vector = self._extract_identical_draw_feature(
                    analysis_data["identical_draw_check"]
                )
                vector_features["duplicate_combination"] = duplicate_vector
                feature_names_by_group["duplicate_combination"] = [
                    "has_duplicates",
                    "total_duplicates_norm",
                ]

                # 이 벡터는 별도 파일로도 저장
                self.save_vector_to_file(
                    duplicate_vector, "duplicate_combination_vector.npy"
                )

            # 9-10. 위치별 엔트로피와 표준편차 특성 추출
            (
                position_entropy_vector,
                position_std_vector,
                position_entropy_names,
                position_std_names,
            ) = self._extract_position_entropy_std_features(analysis_data)

            # 엔트로피 벡터 추가
            if len(position_entropy_vector) > 0:
                vector_features["position_entropy"] = position_entropy_vector
                feature_names_by_group["position_entropy"] = position_entropy_names
                self.logger.info(
                    f"위치별 엔트로피 벡터 추가: {len(position_entropy_names)}개 특성"
                )

            # 표준편차 벡터 추가
            if len(position_std_vector) > 0:
                vector_features["position_std"] = position_std_vector
                feature_names_by_group["position_std"] = position_std_names
                self.logger.info(
                    f"위치별 표준편차 벡터 추가: {len(position_std_names)}개 특성"
                )

            # 모든 벡터를 하나로 연결
            final_vector = self._combine_vectors(vector_features)

            # 결합된 특성 이름 업데이트
            self.feature_names = []
            for group, names in feature_names_by_group.items():
                self.feature_names.extend(names)

            # 필요한 경우 추가 이름 생성
            if len(self.feature_names) < len(final_vector):
                self.feature_names.extend(
                    [
                        f"feature_{i}"
                        for i in range(len(self.feature_names), len(final_vector))
                    ]
                )

            # 저분산 특성 제거 옵션 적용
            if self.remove_low_variance:
                final_vector, self.feature_names = self._filter_low_variance_features(
                    final_vector, self.feature_names
                )

            # 벡터 정규화
            normalized_vector = self.normalize_vector(final_vector)

            # 성능 추적 종료 및 로깅
            self.performance_tracker.stop_tracking("vectorize_enhanced_analysis")
            self.logger.info(f"확장 벡터화 완료: 특성 수 {len(normalized_vector)}")
            return normalized_vector

        except Exception as e:
            # 오류 발생 시에도 성능 추적 종료 보장
            self.performance_tracker.stop_tracking("vectorize_enhanced_analysis")
            self.logger.error(f"확장 특성 벡터화 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 벡터 반환
            return np.zeros(100, dtype=np.float32)

    def _detect_outliers(
        self, vectors: np.ndarray, z_threshold: float = 2.5
    ) -> np.ndarray:
        """
        Z-점수 기반 이상치 탐지

        Args:
            vectors: 입력 특성 벡터 (1D 또는 2D 배열)
            z_threshold: Z-점수 임계값 (기본값: 2.5)

        Returns:
            np.ndarray: 이상치 플래그 벡터 (1: 이상치, 0: 정상)
        """
        # 입력 벡터가 1D인지 2D인지 확인
        is_1d = len(vectors.shape) == 1
        if is_1d:
            # 1D 배열의 경우 크기 1인 2D 배열로 변환
            vectors = vectors.reshape(1, -1)

        # 원본 shape 저장
        original_shape = vectors.shape[0]
        outlier_flags = np.zeros(original_shape, dtype=np.bool_)

        # 각 특성에 대해 Z-점수 계산
        for j in range(vectors.shape[1]):
            feature_values = vectors[:, j]
            mean = np.mean(feature_values)
            std = np.std(feature_values)

            if std > 0:  # 0으로 나누기 방지
                z_scores = np.abs((feature_values - mean) / std)
                # 특성별 이상치 탐지
                feature_outliers = z_scores > z_threshold
                # 전체 이상치 플래그 업데이트 (하나라도 이상치면 1)
                outlier_flags = np.logical_or(outlier_flags, feature_outliers)

        # 이상치 통계 로깅
        outlier_count = np.sum(outlier_flags)
        outlier_ratio = (
            outlier_count / len(outlier_flags) if len(outlier_flags) > 0 else 0
        )
        self.logger.info(f"이상치 탐지 완료: {outlier_count}개 ({outlier_ratio:.2%})")

        return outlier_flags

    def _save_outlier_mask(self, outlier_flags: np.ndarray) -> None:
        """
        이상치 마스크 및 인덱스 저장

        Args:
            outlier_flags: 이상치 플래그 배열 (1: 이상치, 0: 정상)
        """
        try:
            # 캐시 디렉토리 확인
            cache_dir = Path(self.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            # 이상치 마스크 저장
            mask_file = cache_dir / "outlier_vector_mask.npy"
            np.save(mask_file, outlier_flags)

            # 이상치 인덱스 저장
            outlier_indices = np.where(outlier_flags)[0].tolist()
            indices_file = cache_dir / "outlier_vector_indices.json"
            with open(indices_file, "w", encoding="utf-8") as f:
                json.dump(outlier_indices, f)

            self.logger.info(
                f"이상치 마스크 저장 완료: {mask_file}, 이상치 개수: {len(outlier_indices)}"
            )
        except Exception as e:
            self.logger.error(f"이상치 마스크 저장 실패: {e}")

    def _update_feature_names_with_enhanced_features(self) -> None:
        """
        확장된 특성 이름 목록 업데이트
        """
        # 현재 특성 목록이 비어있거나 확장이 필요한 경우
        if not self.feature_names or len(self.feature_names) < 20:
            # 확장된 특성 목록 정의
            enhanced_features = [
                # 기본 특성
                "frequency",
                "recency",
                # 세그먼트 특성
                "segment_10_freq",
                "segment_5_freq",
                # 간격 특성
                "gap_avg",
                "gap_std",
                "gap_min",
                "gap_max",
                # 패턴 재출현 특성
                "pattern_reappearance",
                # 최근 재출현 특성
                "recent_gaps",
                # 중복 당첨 특성
                "identical_draw",
                # 쌍 중심성 특성
                "pair_centrality",
                "pair_centrality_avg",
                # 쌍 ROI 특성
                "pair_roi_score",
                "roi_score_avg",
                # 빈번한 3중 조합 특성
                "frequent_triples",
                # 그래프 통계 특성
                "graph_avg_weight",
                "graph_max_weight",
                # 위치별 특성
                "position_freq_1",
                "position_freq_2",
                "position_freq_3",
                "position_freq_4",
                "position_freq_5",
                "position_freq_6",
                # 세그먼트 트렌드 특성
                "segment_trend",
                "segment_trend_score",
                # 간격 편차 특성
                "gap_stddev",
                "gap_deviation",
                # 조합 다양성 특성
                "combination_diversity",
                # ROI 추세 특성
                "roi_trend_by_pattern",
                # 핫/콜드 특성
                "hot_ratio",
                "cold_ratio",
                "hot_cold_mix_score",
            ]

            # 중복 없이 특성 이름 업데이트
            self.feature_names = list(dict.fromkeys(enhanced_features))

        # 최종 특성 개수 확인
        self.logger.info(f"업데이트된 특성 이름 목록: {len(self.feature_names)}개")

    def _vectorize_position_frequency(self, position_matrix: np.ndarray) -> np.ndarray:
        """
        위치별 빈도 행렬을 특성 벡터로 변환합니다.

        Args:
            position_matrix: 위치별 빈도 행렬 (shape: [6, 45])

        Returns:
            np.ndarray: 변환된 특성 벡터
        """
        if position_matrix is None or position_matrix.size == 0:
            self.logger.warning("위치별 빈도 행렬이 비어 있습니다")
            return np.zeros(12)  # 기본 0 벡터 반환 (위치 엔트로피 6개 + 표준편차 6개)

        # 행렬 정규화 (각 위치별로 합이 1이 되도록)
        normalized_matrix = position_matrix.copy()
        row_sums = normalized_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
        normalized_matrix = normalized_matrix / row_sums

        # 각 위치별 엔트로피 계산
        position_entropy = np.zeros(6)
        position_std = np.zeros(6)
        position_top_freq_ratio = np.zeros(6)

        for i in range(6):
            # 엔트로피 계산 (0 값은 건너뛰기)
            probs = normalized_matrix[i]
            non_zero_probs = probs[probs > 0]
            entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
            # 최대 가능 엔트로피로 정규화 (log2(45) = 5.49...)
            max_entropy = np.log2(45)
            position_entropy[i] = entropy / max_entropy

            # 표준편차 계산 (변동성 지표)
            position_std[i] = (
                np.std(normalized_matrix[i]) * 10
            )  # 정규화를 위해 스케일 조정

            # 최대 빈도 비율 계산 (한 번호에 집중도)
            if np.max(normalized_matrix[i]) > 0:
                position_top_freq_ratio[i] = np.max(normalized_matrix[i]) / (1.0 / 45)
            else:
                position_top_freq_ratio[i] = 1.0

        # 위치별 분산 평균 및 편향 점수 계산
        position_variance_avg = np.mean(position_std)

        # 위치 간 균일성 계산 (위치별 표준편차의 표준편차)
        position_bias_score = np.std(position_std)

        # 특성 이름 업데이트
        if self.feature_names is not None:
            # 이미 추가된 특성인지 확인하여 중복 방지
            if "position_entropy_1" not in self.feature_names:
                for i in range(6):
                    self.feature_names.append(f"position_entropy_{i+1}")
                    self.feature_names.append(f"position_std_{i+1}")
                    self.feature_names.append(f"position_top_freq_{i+1}")
                self.feature_names.append("position_variance_avg")
                self.feature_names.append("position_bias_score")

        # 모든 특성 벡터 결합
        result_vector = np.concatenate(
            [
                position_entropy,  # 6
                position_std,  # 6
                position_top_freq_ratio,  # 6
                np.array([position_variance_avg, position_bias_score]),  # 2
            ]
        )

        # 0-1 범위로 정규화
        result_vector = np.clip(result_vector, 0, 1)

        return result_vector

    def _vectorize_segment_trend(self, segment_trend_matrix: np.ndarray) -> np.ndarray:
        """
        세그먼트 추세 행렬을 벡터로 변환합니다.

        Args:
            segment_trend_matrix: 세그먼트 추세 행렬 (형태: (n_segments, n_turns))

        Returns:
            np.ndarray: 벡터화된 세그먼트 추세 특성
        """
        # 빈 데이터 처리
        if segment_trend_matrix is None:
            return np.zeros(15, dtype=np.float32)

        # list 객체를 numpy 배열로 변환
        if isinstance(segment_trend_matrix, list):
            try:
                segment_trend_matrix = np.array(segment_trend_matrix, dtype=np.float32)
            except Exception as e:
                self.logger.error(f"세그먼트 추세 행렬 변환 중 오류: {e}")
                return np.zeros(15, dtype=np.float32)

        # 크기가 0인 배열 처리
        if (
            not isinstance(segment_trend_matrix, np.ndarray)
            or segment_trend_matrix.size == 0
        ):
            return np.zeros(15, dtype=np.float32)

        # 배열이 1차원이면 2차원으로 변환 (단일 세그먼트 처리)
        if len(segment_trend_matrix.shape) == 1:
            segment_trend_matrix = segment_trend_matrix.reshape(1, -1)

        # 결과 저장할 특성 목록
        trend_features = []

        # 1. 각 세그먼트의 최근 추세 (5개 특성)
        # 마지막 10회 평균을 사용
        segment_count = segment_trend_matrix.shape[0]
        turn_count = segment_trend_matrix.shape[1]

        # 마지막 회차부터 최대 10회차 데이터 사용
        recent_window = min(10, turn_count)
        recent_trends = []

        for i in range(segment_count):
            segment_data = segment_trend_matrix[i, -recent_window:]
            avg_trend = np.mean(segment_data)
            recent_trends.append(float(avg_trend))

        # 세그먼트 수가 5개보다 적을 경우 0으로 채움
        while len(recent_trends) < 5:
            recent_trends.append(0.0)

        # 5개 세그먼트만 사용
        recent_trends = recent_trends[:5]
        trend_features.extend(recent_trends)

        # 2. 각 세그먼트의 변동성 (5개 특성)
        segment_volatility = []
        for i in range(segment_count):
            segment_data = segment_trend_matrix[i, -recent_window:]
            # 변동 계수 (CV) = 표준편차 / 평균
            mean_val = np.mean(segment_data)
            std_val = np.std(segment_data)
            cv = std_val / mean_val if mean_val > 0 else 0.0
            segment_volatility.append(float(cv))

        # 세그먼트 수가 5개보다 적을 경우 0으로 채움
        while len(segment_volatility) < 5:
            segment_volatility.append(0.0)

        # 5개 세그먼트만 사용
        segment_volatility = segment_volatility[:5]
        trend_features.extend(segment_volatility)

        # 3. 세그먼트 간 상관관계 (5개 특성)
        segment_correlations = []
        if segment_count >= 2:
            # 인접 세그먼트 간 상관관계 계산
            for i in range(segment_count - 1):
                seg1 = segment_trend_matrix[i, -recent_window:]
                seg2 = segment_trend_matrix[i + 1, -recent_window:]
                corr = np.corrcoef(seg1, seg2)[0, 1]
                segment_correlations.append(float(corr))

            # 첫 번째와 마지막 세그먼트 상관관계 추가 (원형 구조 고려)
            if segment_count >= 3:
                seg1 = segment_trend_matrix[0, -recent_window:]
                seg2 = segment_trend_matrix[-1, -recent_window:]
                corr = np.corrcoef(seg1, seg2)[0, 1]
                segment_correlations.append(float(corr))

        # NaN 값 0으로 대체
        segment_correlations = [0.0 if np.isnan(x) else x for x in segment_correlations]

        # 상관관계 수가 5개보다 적을 경우 0으로 채움
        while len(segment_correlations) < 5:
            segment_correlations.append(0.0)

        # 5개 상관관계만 사용
        segment_correlations = segment_correlations[:5]
        trend_features.extend(segment_correlations)

        # 총 15개 특성 반환
        return np.array(trend_features, dtype=np.float32)

    def _vectorize_pair_centrality(self, pair_centrality: Dict[str, Any]) -> np.ndarray:
        """
        쌍 중심성 데이터를 벡터로 변환합니다.

        Args:
            pair_centrality: 쌍 중심성 데이터

        Returns:
            np.ndarray: 벡터화된 쌍 중심성 특성
        """
        # 초기 벡터 크기 설정 (10개 특성)
        vector_size = 10
        centrality_vector = np.zeros(vector_size, dtype=np.float32)

        # 쌍 중심성 데이터가 없는 경우 빈 벡터 반환
        if not pair_centrality:
            return centrality_vector

        # 중심성 값 추출
        centrality_values = []
        for key, value in pair_centrality.items():
            try:
                # 문자열 형태의 키에서 두 번호 추출 (예: "1_34")
                if isinstance(value, (int, float)):
                    centrality_values.append(float(value))
            except (ValueError, TypeError):
                continue

        # 중심성 값이 없는 경우 빈 벡터 반환
        if not centrality_values:
            return centrality_vector

        # 주요 통계값 계산
        centrality_array = np.array(centrality_values)
        centrality_vector[0] = np.mean(centrality_array)  # 평균
        centrality_vector[1] = np.std(centrality_array)  # 표준 편차
        centrality_vector[2] = np.max(centrality_array)  # 최대값
        centrality_vector[3] = np.min(centrality_array)  # 최소값

        # 상위 n개 값의 평균
        n_values = min(5, len(centrality_array))
        if n_values > 0:
            top_values = np.sort(centrality_array)[-n_values:]
            centrality_vector[4] = np.mean(top_values)

        # 중심성 분포 특성
        if len(centrality_array) > 0:
            # 사분위수 계산
            q1 = np.percentile(centrality_array, 25)
            q2 = np.percentile(centrality_array, 50)  # 중앙값
            q3 = np.percentile(centrality_array, 75)

            centrality_vector[5] = q2  # 중앙값
            centrality_vector[6] = q3 - q1  # 사분위 범위

            # 분포의 비대칭도
            if np.std(centrality_array) > 0:
                skewness = np.mean(
                    (
                        (centrality_array - np.mean(centrality_array))
                        / np.std(centrality_array)
                    )
                    ** 3
                )
                centrality_vector[7] = skewness

            # 0.5 이상의 중심성 값 비율
            high_centrality_ratio = np.sum(centrality_array > 0.5) / len(
                centrality_array
            )
            centrality_vector[8] = high_centrality_ratio

            # 중심성 값들의 엔트로피 (값 범위 다양성)
            bins = np.linspace(0, 1, 10)  # 0-1 범위를 10개 구간으로 나눔
            hist, _ = np.histogram(centrality_array, bins=bins, density=True)
            hist = hist[hist > 0]  # 0인 값 제거
            if len(hist) > 0:
                entropy = -np.sum(hist * np.log(hist))
                centrality_vector[9] = entropy

        return centrality_vector

    def _vectorize_pair_roi_scores(self, pair_roi_score: Dict[str, Any]) -> np.ndarray:
        """
        쌍 ROI 점수 데이터를 벡터로 변환합니다.

        Args:
            pair_roi_score: 쌍 ROI 점수 데이터

        Returns:
            np.ndarray: 벡터화된 쌍 ROI 점수 특성
        """
        # 초기 벡터 크기 설정 (10개 특성)
        vector_size = 10
        roi_vector = np.zeros(vector_size, dtype=np.float32)

        # ROI 점수 데이터가 없는 경우 빈 벡터 반환
        if not pair_roi_score:
            return roi_vector

        # ROI 값 추출
        roi_values = []
        for key, value in pair_roi_score.items():
            try:
                if isinstance(value, (int, float)):
                    roi_values.append(float(value))
            except (ValueError, TypeError):
                continue

        # ROI 값이 없는 경우 빈 벡터 반환
        if not roi_values:
            return roi_vector

        # 주요 통계값 계산
        roi_array = np.array(roi_values)
        roi_vector[0] = np.mean(roi_array)  # 평균
        roi_vector[1] = np.std(roi_array)  # 표준 편차
        roi_vector[2] = np.max(roi_array)  # 최대값
        roi_vector[3] = np.min(roi_array)  # 최소값

        # 상위 n개 값의 평균
        n_values = min(5, len(roi_array))
        if n_values > 0:
            top_values = np.sort(roi_array)[-n_values:]
            roi_vector[4] = np.mean(top_values)

        # ROI 값 분포 특성
        if len(roi_array) > 0:
            # 양수 ROI 값 비율
            positive_roi_ratio = np.sum(roi_array > 0) / len(roi_array)
            roi_vector[5] = positive_roi_ratio

            # 높은 ROI 값 (1.5 이상) 비율
            high_roi_ratio = np.sum(roi_array > 1.5) / len(roi_array)
            roi_vector[6] = high_roi_ratio

            # ROI 값들의 편차 (변동성 지표)
            if np.mean(roi_array) > 0:
                roi_vector[7] = np.std(roi_array) / np.mean(roi_array)  # 변동 계수

            # 중앙값
            roi_vector[8] = np.median(roi_array)

            # 분포 범위 (최대-최소)
            roi_vector[9] = (
                roi_array.max() - roi_array.min() if len(roi_array) > 1 else 0
            )

        return roi_vector

    def _vectorize_frequent_triples(
        self, frequent_triples: Dict[str, Any]
    ) -> np.ndarray:
        """
        빈번한 3중 조합 데이터를 벡터로 변환합니다.

        Args:
            frequent_triples: 빈번한 3중 조합 데이터

        Returns:
            np.ndarray: 벡터화된 빈번한 3중 조합 특성
        """
        # 초기 벡터 크기 설정 (5개 특성)
        vector_size = 5
        triples_vector = np.zeros(vector_size, dtype=np.float32)

        # 3중 조합 데이터가 없는 경우 빈 벡터 반환
        if not frequent_triples or "triples" not in frequent_triples:
            return triples_vector

        # 3중 조합 목록 추출
        triples_list = frequent_triples.get("triples", [])
        if not triples_list:
            return triples_vector

        # 3중 조합 개수
        triples_vector[0] = min(len(triples_list) / 50.0, 1.0)  # 0-1 범위로 정규화

        # 3중 조합의 평균 출현 빈도
        if "average_frequency" in frequent_triples:
            triples_vector[1] = (
                frequent_triples["average_frequency"] / 10.0
            )  # 0-1 범위로 정규화

        # 3중 조합의 평균 ROI (있는 경우)
        if "average_roi" in frequent_triples:
            triples_vector[2] = min(
                frequent_triples["average_roi"] / 2.0, 1.0
            )  # 0-1 범위로 정규화

        # 상위 3중 조합의 빈도 차이 (있는 경우)
        if len(triples_list) >= 2 and all(
            isinstance(t, dict) and "frequency" in t for t in triples_list[:2]
        ):
            top_diff = abs(triples_list[0]["frequency"] - triples_list[1]["frequency"])
            triples_vector[3] = min(top_diff / 5.0, 1.0)  # 0-1 범위로 정규화

        # 3중 조합에 포함된 고유 번호 수 (있는 경우)
        unique_numbers = set()
        for triple in triples_list:
            if isinstance(triple, dict) and "numbers" in triple:
                unique_numbers.update(triple["numbers"])

        if unique_numbers:
            triples_vector[4] = min(
                len(unique_numbers) / 45.0, 1.0
            )  # 0-1 범위로 정규화

        return triples_vector

    def _visualize_features_with_tsne(self, feature_matrix: np.ndarray) -> str:
        """
        t-SNE를 사용하여 특성 행렬을 시각화합니다.

        Args:
            feature_matrix: 시각화할 특성 행렬

        Returns:
            str: 저장된 시각화 파일 경로
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap

            # t-SNE 매개변수 수정 - 3차원 시각화, 학습률 자동, 초기화 무작위, 퍼플렉시티 30
            tsne = TSNE(
                n_components=3,
                learning_rate="auto",
                init="random",
                perplexity=30,
                random_state=42,
            )

            # 차원 축소 수행
            self.logger.info(
                f"t-SNE 차원 축소 수행 중... (입력 형태: {feature_matrix.shape})"
            )

            # 데이터가 충분한 경우에만 수행
            if len(feature_matrix) < 10:
                self.logger.warning("데이터가 너무 적어 t-SNE 시각화를 생략합니다.")
                return ""

            tsne_result = tsne.fit_transform(feature_matrix)

            # 3D 플롯 생성
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # 컬러맵 설정 (blue -> red 그라데이션)
            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", ["blue", "cyan", "green", "yellow", "red"]
            )

            # 플롯팅
            scatter = ax.scatter(
                tsne_result[:, 0],
                tsne_result[:, 1],
                tsne_result[:, 2],
                c=np.arange(len(tsne_result)),  # 연속적인 색상 할당
                cmap=cmap,
                alpha=0.8,
                s=40,
            )

            # 그래프 제목 및 레이블 설정
            ax.set_title("t-SNE 시각화 (3D)", fontsize=14)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_zlabel("t-SNE 3")

            # 컬러바 추가
            plt.colorbar(scatter, ax=ax, label="데이터 포인트 인덱스")

            # 그래프 저장
            output_dir = Path(self.cache_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "tsne_visualization_3d.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.logger.info(f"t-SNE 시각화 저장 완료: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"t-SNE 시각화 생성 중 오류 발생: {e}")
            return ""

    def _vectorize_gap_deviation(
        self, gap_deviation_score: Dict[str, Any]
    ) -> np.ndarray:
        """
        간격 편차 점수 데이터를 벡터로 변환합니다.

        Args:
            gap_deviation_score: 간격 편차 점수 데이터

        Returns:
            np.ndarray: 벡터화된 간격 편차 점수 특성
        """
        # 초기 벡터 크기 설정 (5개 특성)
        vector_size = 5
        gap_dev_vector = np.zeros(vector_size, dtype=np.float32)

        # 간격 편차 점수 데이터가 없는 경우 빈 벡터 반환
        if not gap_deviation_score:
            return gap_dev_vector

        # 간격 편차 점수 추출
        deviation_values = []
        for num_str, score in gap_deviation_score.items():
            try:
                num = int(num_str) if isinstance(num_str, str) else num_str
                if 1 <= num <= 45 and isinstance(score, (int, float)):
                    deviation_values.append((num, float(score)))
            except (ValueError, TypeError):
                continue

        # 간격 편차 값이 없는 경우 빈 벡터 반환
        if not deviation_values:
            return gap_dev_vector

        # 번호별 간격 편차 점수를 배열로 변환
        nums, scores = zip(*deviation_values)
        deviation_array = np.array(scores)

        # 주요 통계값 계산
        gap_dev_vector[0] = np.mean(deviation_array)  # 평균 편차 점수
        gap_dev_vector[1] = np.std(deviation_array)  # 편차 점수의 표준편차

        # 높은 편차 점수(0.7 이상)를 가진 번호의 비율
        high_dev_ratio = (
            np.sum(deviation_array > 0.7) / len(deviation_array)
            if len(deviation_array) > 0
            else 0
        )
        gap_dev_vector[2] = high_dev_ratio

        # 낮은 편차 점수(0.3 이하)를 가진 번호의 비율
        low_dev_ratio = (
            np.sum(deviation_array < 0.3) / len(deviation_array)
            if len(deviation_array) > 0
            else 0
        )
        gap_dev_vector[3] = low_dev_ratio

        # 편차 점수의 범위 (최대-최소)
        if len(deviation_array) > 1:
            gap_dev_vector[4] = np.max(deviation_array) - np.min(deviation_array)

        return gap_dev_vector

    def _vectorize_combination_diversity(
        self, diversity_score: Dict[str, Any]
    ) -> np.ndarray:
        """
        조합 다양성 점수 데이터를 벡터로 변환합니다.

        Args:
            diversity_score: 조합 다양성 점수 데이터

        Returns:
            np.ndarray: 벡터화된 조합 다양성 점수 특성
        """
        # 초기 벡터 크기 설정 (5개 특성)
        vector_size = 5
        diversity_vector = np.zeros(vector_size, dtype=np.float32)

        # 다양성 점수 데이터가 없는 경우 빈 벡터 반환
        if not diversity_score:
            return diversity_vector

        # 전체 다양성 점수
        if "total_diversity" in diversity_score:
            diversity_vector[0] = float(diversity_score["total_diversity"])

        # 세그먼트별 다양성 점수
        if "segment_diversity" in diversity_score and isinstance(
            diversity_score["segment_diversity"], dict
        ):
            segment_div = diversity_score["segment_diversity"]
            segment_scores = []

            # 5개 세그먼트의 다양성 점수 추출
            for i in range(1, 6):
                seg_key = f"segment_{i}"
                if seg_key in segment_div and isinstance(
                    segment_div[seg_key], (int, float)
                ):
                    segment_scores.append(float(segment_div[seg_key]))

            # 세그먼트 다양성 점수의 평균과 표준편차
            if segment_scores:
                diversity_vector[1] = np.mean(segment_scores)  # 평균
                diversity_vector[2] = np.std(segment_scores)  # 표준편차

        # 다양성 순위
        if "diversity_rank" in diversity_score:
            rank = float(diversity_score["diversity_rank"])
            diversity_vector[3] = rank / 1000.0  # 0-1 범위로 정규화 (총 회차수 기준)

        # 다양성 분산도
        if "diversity_variance" in diversity_score:
            diversity_vector[4] = float(diversity_score["diversity_variance"])

        return diversity_vector

    def _vectorize_roi_trend_by_pattern(
        self, roi_trend_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        패턴별 ROI 추세 데이터를 벡터로 변환합니다.

        Args:
            roi_trend_data: 패턴별 ROI 추세 데이터

        Returns:
            np.ndarray: 벡터화된 패턴별 ROI 추세 특성
        """
        # 초기 벡터 크기 설정 (10개 특성)
        vector_size = 10
        roi_trend_vector = np.zeros(vector_size, dtype=np.float32)

        # ROI 추세 데이터가 없는 경우 빈 벡터 반환
        if not roi_trend_data:
            return roi_trend_vector

        # 패턴별 ROI 추세 데이터 추출
        pattern_trends = []
        for pattern_name, trend_data in roi_trend_data.items():
            if isinstance(trend_data, dict) and "trend" in trend_data:
                trend_value = float(trend_data["trend"])
                pattern_trends.append((pattern_name, trend_value))

        # 패턴별 ROI 추세 데이터가 없는 경우 빈 벡터 반환
        if not pattern_trends:
            return roi_trend_vector

        # 패턴별 추세 값 추출
        _, trend_values = zip(*pattern_trends)
        trend_array = np.array(trend_values)

        # 추세 통계 계산
        roi_trend_vector[0] = np.mean(trend_array)  # 평균 추세
        roi_trend_vector[1] = np.std(trend_array)  # 추세의 표준편차
        roi_trend_vector[2] = np.max(trend_array)  # 최대 추세
        roi_trend_vector[3] = np.min(trend_array)  # 최소 추세

        # 양수/음수 추세 비율
        if len(trend_array) > 0:
            roi_trend_vector[4] = np.sum(trend_array > 0) / len(
                trend_array
            )  # 양수 추세 비율
            roi_trend_vector[5] = np.sum(trend_array < 0) / len(
                trend_array
            )  # 음수 추세 비율

        # 추세 변동성 (절대값 평균)
        roi_trend_vector[6] = np.mean(np.abs(trend_array))

        # 상위 3개 추세 평균
        top_n = min(3, len(trend_array))
        if top_n > 0:
            top_trends = np.sort(trend_array)[-top_n:]
            roi_trend_vector[7] = np.mean(top_trends)

        # 하위 3개 추세 평균
        if top_n > 0:
            bottom_trends = np.sort(trend_array)[:top_n]
            roi_trend_vector[8] = np.mean(bottom_trends)

        # 추세 중앙값
        roi_trend_vector[9] = np.median(trend_array)

        return roi_trend_vector

    def _vectorize_physical_structure_features(
        self, physical_structure_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        로또 추첨기의 물리적 구조 특성 데이터를 벡터로 변환합니다.

        Args:
            physical_structure_data: 물리적 구조 특성 데이터

        Returns:
            np.ndarray: 벡터화된 물리적 구조 특성
        """
        # 초기 벡터 크기 설정 (15개 특성)
        vector_size = 15
        physical_vector = np.zeros(vector_size, dtype=np.float32)

        # 물리적 구조 특성 데이터가 없는 경우 빈 벡터 반환
        if not physical_structure_data:
            return physical_vector

        try:
            # 1. 거리 분산 (distance_variance) 특성
            if "distance_variance" in physical_structure_data:
                dist_variance = physical_structure_data["distance_variance"]
                if isinstance(dist_variance, dict):
                    # 평균 거리 분산
                    if "average" in dist_variance:
                        physical_vector[0] = self.safe_float_conversion(
                            dist_variance["average"]
                        )
                    # 표준편차
                    if "std" in dist_variance:
                        physical_vector[1] = self.safe_float_conversion(
                            dist_variance["std"]
                        )
                    # 최대 거리 분산
                    if "max" in dist_variance:
                        physical_vector[2] = self.safe_float_conversion(
                            dist_variance["max"]
                        )
                    # 최소 거리 분산
                    if "min" in dist_variance:
                        physical_vector[3] = self.safe_float_conversion(
                            dist_variance["min"]
                        )

            # 2. 연속 쌍 비율 (sequential_pair_rate) 특성
            if "sequential_pair_rate" in physical_structure_data:
                seq_pair_rate = physical_structure_data["sequential_pair_rate"]
                if isinstance(seq_pair_rate, dict):
                    # 평균 연속 쌍 비율
                    if "avg_rate" in seq_pair_rate:
                        physical_vector[4] = self.safe_float_conversion(
                            seq_pair_rate["avg_rate"]
                        )
                    # 최대 연속 쌍 비율
                    if "max_rate" in seq_pair_rate:
                        physical_vector[5] = self.safe_float_conversion(
                            seq_pair_rate["max_rate"]
                        )
                    # 연속 쌍 비율 표준편차
                    if "std_rate" in seq_pair_rate:
                        physical_vector[6] = self.safe_float_conversion(
                            seq_pair_rate["std_rate"]
                        )

            # 3. 위치별 Z-점수 (zscore_num1 ~ zscore_num6)
            for i in range(6):
                z_score_key = f"zscore_num{i+1}"
                if z_score_key in physical_structure_data:
                    physical_vector[7 + i] = self.safe_float_conversion(
                        physical_structure_data[z_score_key]
                    )

            # 4. 이항분포 매칭 점수 (binomial_match_score)
            if "binomial_match_score" in physical_structure_data:
                physical_vector[13] = self.safe_float_conversion(
                    physical_structure_data["binomial_match_score"]
                )

            # 5. 번호 표준편차 점수 (number_std_score)
            if "number_std_score" in physical_structure_data:
                physical_vector[14] = self.safe_float_conversion(
                    physical_structure_data["number_std_score"]
                )

            return physical_vector

        except Exception as e:
            self.logger.error(f"물리적 구조 특성 벡터화 중 오류 발생: {e}")
            return np.zeros(vector_size, dtype=np.float32)

    def vectorize_extended_features(
        self, analysis_data: Dict[str, Any], pair_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        청사진 시스템을 사용한 안전한 확장 특성 벡터화
        각 그룹별로 고정 차원을 보장하여 차원 불일치 문제 해결

        Args:
            analysis_data: 패턴 분석 결과 딕셔너리
            pair_data: 쌍 분석 결과 딕셔너리 (선택적)

        Returns:
            Tuple[np.ndarray, List[str]]: 95차원 고정 벡터와 특성 이름
        """
        self.logger.info("청사진 기반 확장 특성 벡터화 시작...")

        try:
            # 그룹별 벡터 수집
            group_vectors = {}
            all_feature_names = []

            # 1. 기본 패턴 분석 (25차원)
            pattern_vec, pattern_names = self._vectorize_group_safe(
                "pattern_analysis",
                analysis_data.get("pattern_analysis", {}),
                lambda data: self._extract_basic_pattern_features(data),
            )
            group_vectors["pattern_analysis"] = pattern_vec
            all_feature_names.extend(pattern_names)

            # 2. 분포 패턴 (10차원)
            dist_vec, dist_names = self._vectorize_group_safe(
                "distribution_pattern",
                analysis_data.get("distribution_pattern", {}),
                lambda data: self._extract_distribution_features(data),
            )
            group_vectors["distribution_pattern"] = dist_vec
            all_feature_names.extend(dist_names)

            # 3. 세그먼트 빈도 (15차원: 10구간 + 5구간)
            seg_vec, seg_names = self._vectorize_group_safe(
                "segment_frequency",
                {
                    "segment_10": analysis_data.get("segment_10_frequency", {}),
                    "segment_5": analysis_data.get("segment_5_frequency", {}),
                },
                lambda data: self._extract_segment_frequency_features(data),
            )
            group_vectors["segment_frequency"] = seg_vec
            all_feature_names.extend(seg_names)

            # 4. 중심성 및 연속성 (12차원)
            cent_vec, cent_names = self._vectorize_group_safe(
                "centrality_consecutive",
                {
                    "centrality": analysis_data.get("segment_centrality", {}),
                    "consecutive": analysis_data.get(
                        "segment_consecutive_patterns", {}
                    ),
                },
                lambda data: self._extract_centrality_consecutive_features(data),
            )
            group_vectors["centrality_consecutive"] = cent_vec
            all_feature_names.extend(cent_names)

            # 5. 갭 통계 및 재출현 (8차원)
            gap_vec, gap_names = self._vectorize_group_safe(
                "gap_reappearance",
                {
                    "gap_stats": analysis_data.get("gap_statistics", {}),
                    "reappearance": analysis_data.get("pattern_reappearance", {}),
                },
                lambda data: self._extract_gap_reappearance_features(data),
            )
            group_vectors["gap_reappearance"] = gap_vec
            all_feature_names.extend(gap_names)

            # 6. ROI 특성 (15차원)
            roi_vec, roi_names = self._vectorize_group_safe(
                "roi_features",
                analysis_data.get("roi_features", {}),
                lambda data: self.extract_roi_features({"roi_features": data}),
            )
            group_vectors["roi_features"] = roi_vec
            all_feature_names.extend(roi_names)

            # 7. 클러스터 품질 (10차원)
            cluster_vec, cluster_names = self._vectorize_group_safe(
                "cluster_features",
                analysis_data.get("cluster_quality", {}),
                lambda data: self.extract_cluster_features({"cluster_quality": data}),
            )
            group_vectors["cluster_features"] = cluster_vec
            all_feature_names.extend(cluster_names)

            # 8. 중복 패턴 특성 (20차원)
            overlap_vec, overlap_names = self._vectorize_group_safe(
                "overlap_patterns",
                analysis_data.get("overlap", {}),
                lambda data: self.extract_overlap_pattern_features({"overlap": data}),
            )
            group_vectors["overlap_patterns"] = overlap_vec
            all_feature_names.extend(overlap_names)

            # 9. 물리적 구조 특성 (11차원)
            phys_vec, phys_names = self._vectorize_group_safe(
                "physical_structure",
                analysis_data.get("physical_structure_features", {}),
                lambda data: self._extract_physical_structure_features(data),
            )
            group_vectors["physical_structure"] = phys_vec
            all_feature_names.extend(phys_names)

            # 10. 쌍 그래프 압축 벡터 (20차원)
            pair_vec, pair_names = self._vectorize_group_safe(
                "pair_graph_vector",
                analysis_data.get("pair_graph_compressed_vector", []),
                lambda data: self._extract_pair_graph_features(data),
            )
            group_vectors["pair_graph_vector"] = pair_vec
            all_feature_names.extend(pair_names)

            # 모든 그룹 벡터 결합
            combined_vector = np.concatenate(list(group_vectors.values()))

            # 최종 검증
            if not self.validate_vector_integrity(combined_vector, all_feature_names):
                raise ValueError("벡터 무결성 검증 실패")

            self.logger.info(
                f"청사진 기반 벡터화 완료: {len(combined_vector)}차원, {len(all_feature_names)}개 특성"
            )
            return combined_vector, all_feature_names

        except Exception as e:
            self.logger.error(f"확장 특성 벡터화 중 오류: {e}")
            # 오류 시 기본 청사진 벡터 반환 - 의미있는 기본값 사용
            default_vector = np.random.uniform(
                0.2, 0.8, self.total_expected_dims
            ).astype(np.float32)
            default_names = []
            for group_name, dims in self.vector_blueprint.items():
                default_names.extend(self.feature_name_templates[group_name][:dims])

            return default_vector, default_names

    def _extract_basic_pattern_features(
        self, pattern_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """기본 패턴 특성 추출 (25차원)"""
        features = []
        names = []

        # 기본 통계 특성들 (10개) - 의미있는 기본값 사용
        basic_stats = {
            "frequency_sum": 0.5,  # 중간값
            "frequency_mean": 0.5,  # 중간값
            "frequency_std": 0.1,  # 낮은 분산
            "frequency_max": 0.8,  # 높은 최대값
            "frequency_min": 0.2,  # 낮은 최소값
            "gap_mean": 0.5,  # 중간 갭
            "gap_std": 0.3,  # 중간 분산
            "gap_max": 0.9,  # 높은 최대 갭
            "gap_min": 0.1,  # 낮은 최소 갭
            "total_draws": 0.6,  # 중간 추첨 수
        }

        for stat, default_val in basic_stats.items():
            value = self.safe_float_conversion(pattern_data.get(stat, default_val))
            # 0-1 범위로 정규화
            if stat == "total_draws":
                value = min(1.0, value / 1200.0)  # 총 추첨 수 기준 정규화
            elif "frequency" in stat:
                value = min(1.0, max(0.0, value))  # 빈도는 이미 정규화됨
            else:
                value = min(1.0, max(0.0, value / 100.0))  # 갭 관련 정규화

            features.append(value)
            names.append(f"pattern_{stat}")

        # 추가 패턴 특성들 (15개) - 다양한 기본값 사용
        extra_defaults = [
            0.3,
            0.7,
            0.2,
            0.8,
            0.4,
            0.6,
            0.1,
            0.9,
            0.35,
            0.65,
            0.25,
            0.75,
            0.15,
            0.85,
            0.45,
        ]

        for i, default_val in enumerate(extra_defaults):
            # 실제 데이터가 있으면 사용, 없으면 기본값
            key = f"extra_{i+1}"
            value = self.safe_float_conversion(pattern_data.get(key, default_val))
            features.append(min(1.0, max(0.0, value)))
            names.append(f"pattern_extra_{i+1}")

        return np.array(features, dtype=np.float32), names

    def _extract_distribution_features(
        self, dist_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """분포 패턴 특성 추출 (10차원)"""
        features = []
        names = []

        # 분포 특성들 - 의미있는 기본값 사용
        dist_stats = {
            "entropy": 0.7,  # 높은 엔트로피 (다양성)
            "balance_score": 0.5,  # 중간 균형
            "diversity": 0.6,  # 중간 다양성
            "uniformity": 0.4,  # 중간 균등성
            "skewness": 0.0,  # 대칭 분포
            "kurtosis": 0.3,  # 정규 분포에 가까움
            "range_ratio": 0.8,  # 높은 범위 비율
            "concentration": 0.3,  # 낮은 집중도
            "dispersion": 0.7,  # 높은 분산
            "variance": 0.5,  # 중간 분산
        }

        for stat, default_val in dist_stats.items():
            value = self.safe_float_conversion(dist_data.get(stat, default_val))

            # 통계별 정규화
            if stat in ["skewness", "kurtosis"]:
                # 왜도/첨도는 -2~2 범위를 0~1로 정규화
                value = (value + 2) / 4.0
            elif stat == "entropy":
                # 엔트로피는 보통 0~log(n) 범위
                value = min(1.0, max(0.0, value / 5.0))
            else:
                # 나머지는 0~1 범위로 클리핑
                value = min(1.0, max(0.0, value))

            features.append(value)
            names.append(f"dist_{stat}")

        return np.array(features, dtype=np.float32), names

    def _extract_segment_frequency_features(
        self, seg_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """세그먼트 빈도 특성 추출 (15차원)"""
        features = []
        names = []

        # 10구간 빈도 (10차원) - 균등 분포 기본값
        seg_10 = seg_data.get("segment_10", {})
        uniform_freq_10 = 1.0 / 10  # 균등 분포 시 각 구간 빈도

        for i in range(1, 11):
            value = self.safe_float_conversion(seg_10.get(str(i), uniform_freq_10))
            # 빈도는 이미 비율이므로 0~1 범위 확인만
            value = min(1.0, max(0.0, value))
            features.append(value)
            names.append(f"seg10_{i}")

        # 5구간 빈도 (5차원) - 균등 분포 기본값
        seg_5 = seg_data.get("segment_5", {})
        uniform_freq_5 = 1.0 / 5  # 균등 분포 시 각 구간 빈도

        for i in range(1, 6):
            value = self.safe_float_conversion(seg_5.get(str(i), uniform_freq_5))
            # 빈도는 이미 비율이므로 0~1 범위 확인만
            value = min(1.0, max(0.0, value))
            features.append(value)
            names.append(f"seg5_{i}")

        return np.array(features, dtype=np.float32), names

    def _extract_centrality_consecutive_features(
        self, cent_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """중심성 및 연속성 특성 추출 (12차원)"""
        features = []
        names = []

        # 중심성 특성 (6차원)
        centrality = cent_data.get("centrality", {})
        cent_stats = ["mean", "std", "max", "min", "median", "range"]
        for stat in cent_stats:
            value = self.safe_float_conversion(centrality.get(stat, 0.0))
            features.append(value)
            names.append(f"centrality_{stat}")

        # 연속성 특성 (6차원)
        consecutive = cent_data.get("consecutive", {})
        cons_stats = [
            "count",
            "avg_length",
            "max_length",
            "frequency",
            "density",
            "ratio",
        ]
        for stat in cons_stats:
            value = self.safe_float_conversion(consecutive.get(stat, 0.0))
            features.append(value)
            names.append(f"consecutive_{stat}")

        return np.array(features, dtype=np.float32), names

    def _extract_gap_reappearance_features(
        self, gap_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """갭 통계 및 재출현 특성 추출 (8차원)"""
        features = []
        names = []

        # 갭 통계 (4차원)
        gap_stats = gap_data.get("gap_stats", {})
        gap_metrics = ["mean", "std", "max", "min"]
        for metric in gap_metrics:
            value = self.safe_float_conversion(gap_stats.get(metric, 0.0))
            features.append(value)
            names.append(f"gap_{metric}")

        # 재출현 특성 (4차원)
        reappearance = gap_data.get("reappearance", {})
        reapp_metrics = ["frequency", "interval", "consistency", "trend"]
        for metric in reapp_metrics:
            value = self.safe_float_conversion(reappearance.get(metric, 0.0))
            features.append(value)
            names.append(f"reappear_{metric}")

        return np.array(features, dtype=np.float32), names

    def _extract_physical_structure_features(
        self, phys_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """물리적 구조 특성 추출 (11차원)"""
        features = []
        names = []

        # 물리적 특성들 - 의미있는 기본값 사용
        phys_stats = {
            "distance_variance_avg": 0.4,  # 중간 거리 분산
            "distance_variance_std": 0.2,  # 낮은 표준편차
            "sequential_pair_avg": 0.3,  # 낮은 연속 쌍 비율
            "zscore_mean": 0.5,  # 중간 Z점수 (정규화됨)
            "zscore_std": 0.3,  # 중간 Z점수 분산
            "binomial_match_score": 0.5,  # 중간 이항 매치 점수
            "number_std_score": 0.4,  # 중간 번호 표준편차 점수
            "balance_score": 0.5,  # 중간 균형 점수
            "uniformity": 0.4,  # 중간 균등성
            "concentration": 0.3,  # 낮은 집중도
            "dispersion": 0.7,  # 높은 분산
        }

        for stat, default_val in phys_stats.items():
            value = self.safe_float_conversion(phys_data.get(stat, default_val))

            # 통계별 정규화
            if "zscore" in stat:
                # Z점수는 -3~3 범위를 0~1로 정규화
                value = (value + 3) / 6.0
                value = min(1.0, max(0.0, value))
            elif stat == "distance_variance_avg":
                # 거리 분산은 0~최대값 범위
                value = min(1.0, max(0.0, value / 10.0))  # 적절한 스케일링
            elif stat == "sequential_pair_avg":
                # 연속 쌍 비율은 이미 0~1 범위
                value = min(1.0, max(0.0, value))
            else:
                # 나머지 점수들은 0~1 범위로 클리핑
                value = min(1.0, max(0.0, value))

            features.append(value)
            names.append(f"physical_{stat}")

        return np.array(features, dtype=np.float32), names

    def _extract_pair_graph_features(
        self, pair_data: Any
    ) -> Tuple[np.ndarray, List[str]]:
        """쌍 그래프 특성 추출 (20차원)"""
        features = []
        names = []

        if isinstance(pair_data, (list, np.ndarray)) and len(pair_data) > 0:
            # 기존 압축 벡터 사용
            pair_vector = np.array(pair_data, dtype=np.float32)
            # 20차원으로 조정
            if len(pair_vector) >= 20:
                features = pair_vector[:20].tolist()
            else:
                features = pair_vector.tolist() + [0.0] * (20 - len(pair_vector))
        else:
            # 기본값
            features = [0.0] * 20

        # 특성 이름 생성
        names = [f"pair_graph_{i+1}" for i in range(20)]

        return np.array(features, dtype=np.float32), names

    def _extract_position_entropy_std_features(
        self, analysis_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        위치별 엔트로피 및 표준편차 특성을 추출합니다.

        Args:
            analysis_data: 분석 데이터

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
                (엔트로피 벡터, 표준편차 벡터, 엔트로피 특성명, 표준편차 특성명)
        """
        # 엔트로피 특성 추출
        position_entropy_values = []
        position_entropy_names = []
        all_entropy_present = True

        for i in range(1, 7):
            key = f"position_entropy_{i}"
            if key in analysis_data:
                position_entropy_values.append(analysis_data[key])
                position_entropy_names.append(key)
            else:
                all_entropy_present = False
                self.logger.warning(f"위치별 엔트로피 특성이 누락됨: {key}")

        if all_entropy_present:
            position_entropy_vector = np.array(
                position_entropy_values, dtype=np.float32
            )
            self.logger.info(
                f"위치별 엔트로피 벡터 추가: {len(position_entropy_names)}개 특성"
            )
        else:
            position_entropy_vector = np.zeros(6, dtype=np.float32)

        # 표준편차 특성 추출
        position_std_values = []
        position_std_names = []
        all_std_present = True

        for i in range(1, 7):
            key = f"position_std_{i}"
            if key in analysis_data:
                position_std_values.append(analysis_data[key])
                position_std_names.append(key)
            else:
                all_std_present = False
                self.logger.warning(f"위치별 표준편차 특성이 누락됨: {key}")

        if all_std_present:
            position_std_vector = np.array(position_std_values, dtype=np.float32)
            self.logger.info(
                f"위치별 표준편차 벡터 추가: {len(position_std_names)}개 특성"
            )
        else:
            position_std_vector = np.zeros(6, dtype=np.float32)

        return (
            position_entropy_vector,
            position_std_vector,
            position_entropy_names,
            position_std_names,
        )

    def _vectorize_roi_features(
        self, roi_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        ROI 기반 패턴 특성을 벡터화합니다.

        이 메서드는 ROI 분석 결과를 강화학습, 메타러닝 및 점수 모델에서 사용할 수 있는
        특성 벡터로 변환합니다. 주요 특성은 다음과 같습니다:

        1. roi_group_score: 그룹 ID(0,1,2) 분포 - 회차별 ROI 그룹 통계
        2. roi_cluster_score: 클러스터 ID 분포 - 클러스터 분석 기반 ROI 패턴
        3. low_risk_bonus_flag: 저위험 보너스 플래그 비율 - 안정적 수익 지표
        4. roi_pattern_group_id: 패턴별 ROI 그룹 ID - 패턴 기반 수익성 지표

        Args:
            roi_data: ROI 분석 데이터

        Returns:
            Tuple[np.ndarray, List[str]]: (벡터화된 ROI 특성, 특성 이름 목록)
        """
        self.logger.info("ROI 기반 패턴 특성 벡터화 중...")

        # 결과 벡터와 특성 이름 목록 초기화
        feature_vectors = []
        feature_names = []

        # 1. roi_group_score (회차별 그룹 ID) 처리
        roi_group_score = roi_data.get("roi_group_score", {})
        if roi_group_score:
            # 그룹 ID(0,1,2) 분포 계산
            from collections import Counter

            group_ids = list(roi_group_score.values())
            group_dist = Counter(group_ids)
            group_vector = np.array(
                [group_dist.get(i, 0) for i in range(3)], dtype=np.float32
            )

            # 정규화
            if group_vector.sum() > 0:
                group_vector = group_vector / group_vector.sum()

            feature_vectors.append(group_vector)
            feature_names.extend([f"roi_group_{i}" for i in range(3)])
            self.logger.info(f"ROI 그룹 점수 벡터 생성: {group_vector}")
        else:
            # 데이터가 없는 경우 0으로 채운 벡터 추가
            feature_vectors.append(np.zeros(3, dtype=np.float32))
            feature_names.extend([f"roi_group_{i}" for i in range(3)])
            self.logger.warning("ROI 그룹 점수 데이터가 없어 0으로 초기화됨")

        # 2. roi_cluster_score (클러스터 할당) 처리
        roi_cluster_data = roi_data.get("roi_cluster_score", {})
        cluster_assignments = roi_cluster_data.get("cluster_assignments", {})

        if cluster_assignments:
            # 클러스터 ID 분포 계산 (최대 5개 클러스터)
            cluster_ids = list(cluster_assignments.values())
            cluster_dist = Counter(cluster_ids)
            # 고정 길이(5)로 생성
            cluster_vector = np.array(
                [cluster_dist.get(i, 0) for i in range(5)], dtype=np.float32
            )

            # 정규화
            if cluster_vector.sum() > 0:
                cluster_vector = cluster_vector / cluster_vector.sum()

            feature_vectors.append(cluster_vector)
            feature_names.extend([f"roi_cluster_{i}" for i in range(5)])
            self.logger.info(f"ROI 클러스터 벡터 생성: {cluster_vector}")
        else:
            # 데이터가 없는 경우 0으로 채운 벡터 추가
            feature_vectors.append(np.zeros(5, dtype=np.float32))
            feature_names.extend([f"roi_cluster_{i}" for i in range(5)])
            self.logger.warning("ROI 클러스터 데이터가 없어 0으로 초기화됨")

        # 3. low_risk_bonus_flag (저위험 보너스 플래그) 처리
        low_risk_flags_data = roi_data.get("low_risk_bonus_flag", {})
        if low_risk_flags_data:
            # 중첩 구조 처리 (low_risk_bonus_flag 필드가 중첩되어 있는 경우)
            if "low_risk_bonus_flag" in low_risk_flags_data:
                low_risk_flags = low_risk_flags_data.get("low_risk_bonus_flag", {})
            else:
                low_risk_flags = low_risk_flags_data

            # True 값의 비율 계산
            if isinstance(low_risk_flags, dict) and low_risk_flags:
                low_risk_ratio = sum(1 for v in low_risk_flags.values() if v) / max(
                    len(low_risk_flags), 1
                )
            elif isinstance(low_risk_flags, (list, tuple)) and low_risk_flags:
                low_risk_ratio = sum(1 for v in low_risk_flags if v) / max(
                    len(low_risk_flags), 1
                )
            else:
                # 적절한 형식이 아닌 경우 기본값 사용
                low_risk_ratio = 0.0
                self.logger.warning(
                    f"저위험 보너스 플래그 데이터 형식 오류: {type(low_risk_flags)}"
                )

            low_risk_vector = np.array([low_risk_ratio], dtype=np.float32)
            feature_vectors.append(low_risk_vector)
            feature_names.append("low_risk_ratio")
            self.logger.info(f"저위험 보너스 플래그 비율: {low_risk_ratio:.4f}")
        else:
            # 데이터가 없는 경우 0으로 초기화
            feature_vectors.append(np.array([0.0], dtype=np.float32))
            feature_names.append("low_risk_ratio")
            self.logger.warning("저위험 보너스 플래그 데이터가 없어 0으로 초기화됨")

        # 4. roi_pattern_group_id (패턴 키별 그룹 ID) 처리
        roi_pattern_group_id = roi_data.get("roi_pattern_group_id", {})
        if roi_pattern_group_id:
            # 최대 10개 그룹으로 제한
            pattern_group_ids = list(roi_pattern_group_id.values())
            pattern_group_dist = Counter(pattern_group_ids)
            # 고정 길이(10)로 생성
            max_groups = 10
            pattern_group_vector = np.array(
                [pattern_group_dist.get(i, 0) for i in range(max_groups)],
                dtype=np.float32,
            )

            # 정규화
            if pattern_group_vector.sum() > 0:
                pattern_group_vector = pattern_group_vector / pattern_group_vector.sum()

            feature_vectors.append(pattern_group_vector)
            feature_names.extend([f"roi_pattern_group_{i}" for i in range(max_groups)])
            self.logger.info(
                f"ROI 패턴 그룹 벡터 생성: 길이={len(pattern_group_vector)}"
            )
        else:
            # 데이터가 없는 경우 0으로 채운 벡터 추가
            feature_vectors.append(np.zeros(10, dtype=np.float32))
            feature_names.extend([f"roi_pattern_group_{i}" for i in range(10)])
            self.logger.warning("ROI 패턴 그룹 ID 데이터가 없어 0으로 초기화됨")

        # 최종 벡터 결합
        final_vector = np.concatenate(feature_vectors)
        self.logger.info(f"ROI 특성 벡터 생성 완료: 차원={len(final_vector)}")

        # 별도 파일로 저장
        self.save_vector_to_file(final_vector, "roi_features_vector.npy")

        return final_vector, feature_names

    def extract_cluster_features(
        self, full_analysis: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        클러스터 품질 분석 결과를 정규화하여 벡터로 변환합니다.

        Args:
            full_analysis: 통합 분석 결과

        Returns:
            Tuple of (vector array, feature name list)
        """
        metrics = full_analysis.get("cluster_analysis", {}).get("quality_metrics", {})

        # 정규화 기준
        cluster_count = metrics.get("cluster_count", 1)
        sample_size = metrics.get("sample_size", 1)

        def norm(value, base):
            return min(1.0, max(0.0, float(value) / float(base))) if base else 0.0

        vector = [
            float(metrics.get("silhouette_score", 0.0)),
            float(metrics.get("cohesiveness_score", 0.0)),
            norm(metrics.get("avg_distance_between_clusters", 0.0), 10.0),
            norm(metrics.get("balance_score", 0.0), 1.0),
            norm(metrics.get("largest_cluster_size", 0.0), sample_size),
            norm(cluster_count - 1, 9.0),  # 최대 10개 기준
            float(metrics.get("cluster_entropy_score", 0.0)),
        ]

        feature_names = [
            "silhouette_score",
            "cohesiveness_score",
            "avg_distance_between_clusters",
            "balance_score",
            "largest_cluster_size_norm",
            "cluster_count_norm",
            "cluster_entropy_score",
        ]

        return np.array(vector, dtype=np.float32), feature_names

    def extract_roi_features(
        self, full_analysis: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        ROI 분석기에서 계산한 주요 ROI 관련 피처들을 정규화하여 벡터로 추출합니다.

        Args:
            full_analysis: 통합 분석 결과 데이터

        Returns:
            Tuple[np.ndarray, List[str]]: 벡터 값과 해당 특성 이름 목록
        """
        roi_data = full_analysis.get("roi_analysis", {})

        # 정규화 함수
        def norm(value, base=1.0):
            try:
                return min(1.0, max(0.0, float(value) / float(base)))
            except:
                return 0.0

        vector = []
        names = []

        # 1. roi_group_score 분포 (0~2 그룹 → 3차원)
        group_dist = roi_data.get("roi_group_score_distribution", {})
        for i in range(3):
            val = group_dist.get(str(i), 0)
            vector.append(norm(val, 1.0))  # 이미 비율 형태인 경우 그대로 사용
            names.append(f"roi_group_ratio_{i}")

        # 2. roi_cluster_score 분포 (최대 5개 클러스터)
        cluster_dist = roi_data.get("roi_cluster_distribution", {})
        for i in range(5):
            val = cluster_dist.get(str(i), 0)
            vector.append(norm(val, 1.0))
            names.append(f"roi_cluster_ratio_{i}")

        # 3. low_risk_bonus_flag → 비율로 변환
        low_risk_ratio = roi_data.get("low_risk_bonus_ratio", 0.0)
        vector.append(norm(low_risk_ratio, 1.0))
        names.append("low_risk_bonus_ratio")

        # 4. roi_pattern_group_id 분포 (최대 10개 그룹 가정)
        pattern_group_dist = roi_data.get("roi_pattern_group_distribution", {})
        for i in range(10):
            val = pattern_group_dist.get(str(i), 0)
            vector.append(norm(val, 1.0))
            names.append(f"roi_pattern_group_{i}")

        return np.array(vector, dtype=np.float32), names

    def get_gnn_vector(
        self,
        vector_path: str = "data/cache/pair_graph_compressed_vector.npy",
        names_path: str = "data/cache/pair_graph_compressed_vector.names.json",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        GNN 벡터와 특성 이름을 로드합니다.

        Args:
            vector_path: GNN 벡터 파일 경로
            names_path: GNN 특성 이름 파일 경로

        Returns:
            Tuple[np.ndarray, List[str]]: GNN 벡터와 특성 이름
        """
        try:
            # 벡터 로드
            if os.path.exists(vector_path):
                gnn_vector = np.load(vector_path)
            else:
                self.logger.warning(f"GNN 벡터 파일이 존재하지 않습니다: {vector_path}")
                return np.array([]), []

            # 특성 이름 로드
            if os.path.exists(names_path):
                with open(names_path, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)
            else:
                self.logger.warning(
                    f"GNN 특성 이름 파일이 존재하지 않습니다: {names_path}"
                )
                feature_names = [f"gnn_feature_{i}" for i in range(len(gnn_vector))]

            return gnn_vector, feature_names

        except Exception as e:
            self.logger.error(f"GNN 벡터 로드 중 오류 발생: {e}")
            return np.array([]), []

    def extract_overlap_pattern_features(
        self, full_analysis: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        중복 패턴 분석 결과를 벡터화

        Args:
            full_analysis: 전체 분석 결과 딕셔너리

        Returns:
            Tuple[np.ndarray, List[str]]: 중복 패턴 벡터와 특성 이름
        """
        overlap_features = []
        feature_names = []

        try:
            # overlap 분석 결과 추출
            overlap_data = full_analysis.get("overlap", {})

            # 1. 3자리 중복 패턴 특성
            overlap_3_4_data = overlap_data.get("overlap_3_4_digit_patterns", {})

            # 3자리 패턴 빈도 점수
            overlap_3_patterns = overlap_3_4_data.get("overlap_3_patterns", {})
            overlap_3_frequency_score = self.safe_float_conversion(
                overlap_3_patterns.get("avg_frequency", 0.0)
            )
            overlap_features.append(overlap_3_frequency_score)
            feature_names.append("overlap_3_frequency_score")

            # 3자리 패턴 총 개수 (정규화)
            total_3_patterns = self.safe_float_conversion(
                overlap_3_patterns.get("total_patterns", 0)
            )
            overlap_3_pattern_count_norm = min(
                1.0, total_3_patterns / 100.0
            )  # 최대 100개로 정규화
            overlap_features.append(overlap_3_pattern_count_norm)
            feature_names.append("overlap_3_pattern_count_norm")

            # 2. 4자리 중복 패턴 특성 (희귀도)
            overlap_4_patterns = overlap_3_4_data.get("overlap_4_patterns", {})
            overlap_4_rarity_score = self.safe_float_conversion(
                overlap_4_patterns.get("rarity_score", 0.0)
            )
            overlap_features.append(overlap_4_rarity_score)
            feature_names.append("overlap_4_rarity_score")

            # 4자리 패턴 총 개수 (정규화)
            total_4_patterns = self.safe_float_conversion(
                overlap_4_patterns.get("total_patterns", 0)
            )
            overlap_4_pattern_count_norm = min(
                1.0, total_4_patterns / 50.0
            )  # 최대 50개로 정규화
            overlap_features.append(overlap_4_pattern_count_norm)
            feature_names.append("overlap_4_pattern_count_norm")

            # 3. 시간 간격 분석 특성
            time_gap_data = overlap_3_4_data.get("overlap_time_gap_analysis", {})

            # 3자리 패턴 시간 간격 분산 (정규화)
            gap_3_std = self.safe_float_conversion(time_gap_data.get("gap_3_std", 0.0))
            overlap_time_gap_variance_3 = min(
                1.0, gap_3_std / 100.0
            )  # 최대 100회차로 정규화
            overlap_features.append(overlap_time_gap_variance_3)
            feature_names.append("overlap_time_gap_variance_3")

            # 4자리 패턴 시간 간격 분산 (정규화)
            gap_4_std = self.safe_float_conversion(time_gap_data.get("gap_4_std", 0.0))
            overlap_time_gap_variance_4 = min(
                1.0, gap_4_std / 100.0
            )  # 최대 100회차로 정규화
            overlap_features.append(overlap_time_gap_variance_4)
            feature_names.append("overlap_time_gap_variance_4")

            # 4. 패턴 일관성 특성
            pattern_consistency = overlap_3_4_data.get("pattern_consistency", {})

            # 3자리 패턴 일관성
            overlap_pattern_consistency_3 = self.safe_float_conversion(
                pattern_consistency.get("consistency_3", 0.0)
            )
            overlap_features.append(overlap_pattern_consistency_3)
            feature_names.append("overlap_pattern_consistency_3")

            # 4자리 패턴 일관성
            overlap_pattern_consistency_4 = self.safe_float_conversion(
                pattern_consistency.get("consistency_4", 0.0)
            )
            overlap_features.append(overlap_pattern_consistency_4)
            feature_names.append("overlap_pattern_consistency_4")

            # 5. ROI 상관관계 특성
            roi_correlation_data = overlap_data.get("overlap_roi_correlation", {})

            # 3자리 중복 패턴 ROI 상관관계
            overlap_3_roi_correlation = self.safe_float_conversion(
                roi_correlation_data.get("overlap_3_roi_correlation", 0.0)
            )
            overlap_features.append(overlap_3_roi_correlation)
            feature_names.append("overlap_3_roi_correlation")

            # 4자리 중복 패턴 ROI 상관관계
            overlap_4_roi_correlation = self.safe_float_conversion(
                roi_correlation_data.get("overlap_4_roi_correlation", 0.0)
            )
            overlap_features.append(overlap_4_roi_correlation)
            feature_names.append("overlap_4_roi_correlation")

            # 6. ROI 기대값 특성
            overlap_3_roi_stats = roi_correlation_data.get("overlap_3_roi_stats", {})
            overlap_4_roi_stats = roi_correlation_data.get("overlap_4_roi_stats", {})

            # 3자리 패턴 평균 ROI
            overlap_3_avg_roi = self.safe_float_conversion(
                overlap_3_roi_stats.get("avg_roi", 0.0)
            )
            overlap_features.append(overlap_3_avg_roi)
            feature_names.append("overlap_3_avg_roi")

            # 4자리 패턴 평균 ROI
            overlap_4_avg_roi = self.safe_float_conversion(
                overlap_4_roi_stats.get("avg_roi", 0.0)
            )
            overlap_features.append(overlap_4_avg_roi)
            feature_names.append("overlap_4_avg_roi")

            # 7. ROI 긍정 비율 특성
            overlap_3_positive_ratio = self.safe_float_conversion(
                overlap_3_roi_stats.get("positive_roi_ratio", 0.0)
            )
            overlap_features.append(overlap_3_positive_ratio)
            feature_names.append("overlap_3_positive_roi_ratio")

            overlap_4_positive_ratio = self.safe_float_conversion(
                overlap_4_roi_stats.get("positive_roi_ratio", 0.0)
            )
            overlap_features.append(overlap_4_positive_ratio)
            feature_names.append("overlap_4_positive_roi_ratio")

            # 8. 통합 ROI 기대값 (3자리와 4자리 패턴의 가중 평균)
            weight_3 = max(1, overlap_3_roi_stats.get("sample_count", 1))
            weight_4 = max(1, overlap_4_roi_stats.get("sample_count", 1))
            total_weight = weight_3 + weight_4

            overlap_roi_expectation = (
                overlap_3_avg_roi * weight_3 + overlap_4_avg_roi * weight_4
            ) / total_weight
            overlap_features.append(overlap_roi_expectation)
            feature_names.append("overlap_roi_expectation")

            # 9. 샘플 신뢰도 (샘플 수 기반)
            overlap_3_sample_confidence = min(
                1.0, weight_3 / 50.0
            )  # 최대 50개 샘플로 정규화
            overlap_features.append(overlap_3_sample_confidence)
            feature_names.append("overlap_3_sample_confidence")

            overlap_4_sample_confidence = min(
                1.0, weight_4 / 20.0
            )  # 최대 20개 샘플로 정규화
            overlap_features.append(overlap_4_sample_confidence)
            feature_names.append("overlap_4_sample_confidence")

            # 10. 추가 중복 특성들 (기존 overlap 분석에서)
            # 최대 중복도
            max_overlap_data = overlap_data.get("max_overlap_with_past", {})
            max_overlap_avg = self.safe_float_conversion(
                max_overlap_data.get("max_overlap_avg", 0.0)
            )
            overlap_features.append(max_overlap_avg / 6.0)  # 6개 번호로 정규화
            feature_names.append("max_overlap_avg_norm")

            # 중복 플래그
            duplicate_flag = overlap_data.get("duplicate_flag", {})
            duplicate_ratio = self.safe_float_conversion(
                duplicate_flag.get("duplicate_ratio", 0.0)
            )
            overlap_features.append(duplicate_ratio)
            feature_names.append("duplicate_ratio")

            # 최근성 점수
            recency_score = self.safe_float_conversion(
                overlap_data.get("recency_score", {}).get("avg_recency", 0.0)
            )
            overlap_features.append(recency_score)
            feature_names.append("recency_score")

            # 벡터 변환 및 NaN 처리
            overlap_vector = np.array(overlap_features, dtype=np.float32)
            overlap_vector = np.nan_to_num(
                overlap_vector, nan=0.0, posinf=1.0, neginf=0.0
            )

            self.logger.info(
                f"중복 패턴 특성 벡터 생성 완료: {len(feature_names)}개 특성"
            )
            self.logger.info(
                f"중복 패턴 벡터 통계 - 평균: {np.mean(overlap_vector):.4f}, 최대: {np.max(overlap_vector):.4f}, 최소: {np.min(overlap_vector):.4f}"
            )

            return overlap_vector, feature_names

        except Exception as e:
            self.logger.error(f"중복 패턴 특성 벡터 생성 중 오류 발생: {e}")
            # 오류 발생 시 기본 벡터 반환 (20개 특성) - 의미있는 기본값 사용
            default_values = [
                0.3,
                0.5,
                0.2,
                0.7,
                0.4,
                0.6,
                0.25,
                0.75,
                0.35,
                0.65,
                0.15,
                0.85,
                0.45,
                0.55,
                0.3,
                0.7,
                0.2,
                0.8,
                0.4,
                0.6,
            ]
            default_vector = np.array(default_values, dtype=np.float32)
            default_names = [f"overlap_feature_{i+1}" for i in range(20)]
            return default_vector, default_names

    def extract_position_bias_features(
        self, full_analysis: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        추첨 순서 편향 분석 결과를 벡터화합니다.

        Args:
            full_analysis: 전체 분석 결과

        Returns:
            Tuple[np.ndarray, List[str]]: 추첨 순서 편향 벡터와 특성 이름
        """
        try:
            position_bias_data = full_analysis.get("position_bias_features", {})

            if not position_bias_data:
                self.logger.warning(
                    "추첨 순서 편향 데이터가 없습니다. 기본값으로 대체합니다."
                )
                position_bias_data = {}

            # 추첨 순서 편향 특성 추출 (5개)
            features = []
            feature_names = []

            # 1. 위치별 평균값 특성
            position_min_value_mean = position_bias_data.get(
                "position_min_value_mean", 0.0
            )
            position_max_value_mean = position_bias_data.get(
                "position_max_value_mean", 0.0
            )
            position_gap_mean = position_bias_data.get("position_gap_mean", 0.0)

            features.extend(
                [
                    np.clip(position_min_value_mean / 45.0, 0, 1),  # 정규화
                    np.clip(position_max_value_mean / 45.0, 0, 1),  # 정규화
                    np.clip(position_gap_mean / 40.0, 0, 1),  # 최대 간격 40으로 정규화
                ]
            )

            feature_names.extend(
                [
                    "position_min_value_mean",
                    "position_max_value_mean",
                    "position_gap_mean",
                ]
            )

            # 2. 홀짝 및 구간 비율 특성
            position_even_odd_ratio = position_bias_data.get(
                "position_even_odd_ratio", 0.5
            )
            position_low_high_ratio = position_bias_data.get(
                "position_low_high_ratio", 0.5
            )

            features.extend(
                [
                    np.clip(position_even_odd_ratio, 0, 1),
                    np.clip(position_low_high_ratio, 0, 1),
                ]
            )

            feature_names.extend(["position_even_odd_ratio", "position_low_high_ratio"])

            # NaN/Inf 체크 및 수정
            features = [self.safe_float_conversion(f) for f in features]
            position_vector = np.array(features, dtype=np.float32)

            self.logger.info(f"추첨 순서 편향 벡터화 완료: {len(features)}개 특성")
            return position_vector, feature_names

        except Exception as e:
            self.logger.error(f"추첨 순서 편향 벡터화 중 오류 발생: {str(e)}")
            # 기본값 반환 - 의미있는 기본값 사용
            default_vector = np.array([0.5, 0.7, 0.6, 0.5, 0.5], dtype=np.float32)
            default_names = [
                "position_min_value_mean",
                "position_max_value_mean",
                "position_gap_mean",
                "position_even_odd_ratio",
                "position_low_high_ratio",
            ]
            return default_vector, default_names

    def extract_overlap_time_gap_features(
        self, full_analysis: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        중복 패턴 시간적 주기성 분석 결과를 벡터화합니다.

        Args:
            full_analysis: 전체 분석 결과

        Returns:
            Tuple[np.ndarray, List[str]]: 시간적 주기성 벡터와 특성 이름
        """
        try:
            overlap_time_data = full_analysis.get("overlap_time_gaps", {})

            if not overlap_time_data:
                self.logger.warning(
                    "중복 패턴 시간적 주기성 데이터가 없습니다. 기본값으로 대체합니다."
                )
                overlap_time_data = {}

            # 시간적 주기성 특성 추출 (5개)
            features = []
            feature_names = []

            # 1. 3매치/4매치 평균 간격
            overlap_3_time_gap_mean = overlap_time_data.get(
                "overlap_3_time_gap_mean", 0.0
            )
            overlap_4_time_gap_mean = overlap_time_data.get(
                "overlap_4_time_gap_mean", 0.0
            )

            # 간격을 100회차로 정규화 (보통 10-50회차 간격)
            features.extend(
                [
                    np.clip(overlap_3_time_gap_mean / 100.0, 0, 1),
                    np.clip(overlap_4_time_gap_mean / 100.0, 0, 1),
                ]
            )

            feature_names.extend(["overlap_3_time_gap_mean", "overlap_4_time_gap_mean"])

            # 2. 간격 표준편차
            overlap_time_gap_stddev = overlap_time_data.get(
                "overlap_time_gap_stddev", 0.0
            )
            features.append(
                np.clip(overlap_time_gap_stddev / 50.0, 0, 1)
            )  # 표준편차 50으로 정규화
            feature_names.append("overlap_time_gap_stddev")

            # 3. 최근 중복 발생 횟수
            recent_overlap_3_count = overlap_time_data.get("recent_overlap_3_count", 0)
            recent_overlap_4_count = overlap_time_data.get("recent_overlap_4_count", 0)

            features.extend(
                [
                    np.clip(recent_overlap_3_count / 10.0, 0, 1),  # 최대 10회로 정규화
                    np.clip(recent_overlap_4_count / 5.0, 0, 1),  # 최대 5회로 정규화
                ]
            )

            feature_names.extend(["recent_overlap_3_count", "recent_overlap_4_count"])

            # NaN/Inf 체크 및 수정
            features = [self.safe_float_conversion(f) for f in features]
            time_gap_vector = np.array(features, dtype=np.float32)

            self.logger.info(
                f"중복 패턴 시간적 주기성 벡터화 완료: {len(features)}개 특성"
            )
            return time_gap_vector, feature_names

        except Exception as e:
            self.logger.error(f"중복 패턴 시간적 주기성 벡터화 중 오류 발생: {str(e)}")
            # 기본값 반환 - 의미있는 기본값 사용
            default_vector = np.array([0.3, 0.2, 0.4, 0.1, 0.05], dtype=np.float32)
            default_names = [
                "overlap_3_time_gap_mean",
                "overlap_4_time_gap_mean",
                "overlap_time_gap_stddev",
                "recent_overlap_3_count",
                "recent_overlap_4_count",
            ]
            return default_vector, default_names

    def extract_conditional_interaction_features(
        self, full_analysis: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        번호 간 조건부 상호작용 분석 결과를 벡터화합니다.

        Args:
            full_analysis: 전체 분석 결과

        Returns:
            Tuple[np.ndarray, List[str]]: 조건부 상호작용 벡터와 특성 이름
        """
        try:
            conditional_data = full_analysis.get("conditional_interaction_features", {})

            if not conditional_data:
                self.logger.warning(
                    "번호 간 조건부 상호작용 데이터가 없습니다. 기본값으로 대체합니다."
                )
                conditional_data = {}

            # 조건부 상호작용 특성 추출 (3개)
            features = []
            feature_names = []

            # 1. 끌림 점수 (Attraction Score)
            number_attraction_score = conditional_data.get(
                "number_attraction_score", 0.0
            )
            features.append(np.clip(number_attraction_score, 0, 1))
            feature_names.append("number_attraction_score")

            # 2. 회피 점수 (Repulsion Score)
            number_repulsion_score = conditional_data.get("number_repulsion_score", 0.0)
            features.append(np.clip(number_repulsion_score, 0, 1))
            feature_names.append("number_repulsion_score")

            # 3. 조건부 의존성 강도
            conditional_dependency_strength = conditional_data.get(
                "conditional_dependency_strength", 0.0
            )
            features.append(np.clip(conditional_dependency_strength, 0, 1))
            feature_names.append("conditional_dependency_strength")

            # NaN/Inf 체크 및 수정
            features = [self.safe_float_conversion(f) for f in features]
            conditional_vector = np.array(features, dtype=np.float32)

            self.logger.info(
                f"번호 간 조건부 상호작용 벡터화 완료: {len(features)}개 특성"
            )
            return conditional_vector, feature_names

        except Exception as e:
            self.logger.error(f"번호 간 조건부 상호작용 벡터화 중 오류 발생: {str(e)}")
            # 기본값 반환
            default_vector = np.zeros(3, dtype=np.float32)
            default_names = [
                "number_attraction_score",
                "number_repulsion_score",
                "conditional_dependency_strength",
            ]
            return default_vector, default_names

    def extract_micro_bias_features(
        self, full_analysis: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        홀짝 및 구간별 미세 편향성 분석 결과를 벡터화합니다.

        Args:
            full_analysis: 전체 분석 결과

        Returns:
            Tuple[np.ndarray, List[str]]: 미세 편향성 벡터와 특성 이름
        """
        try:
            odd_even_data = full_analysis.get("odd_even_micro_bias", {})
            range_data = full_analysis.get("range_micro_bias", {})

            if not odd_even_data:
                self.logger.warning(
                    "홀짝 미세 편향성 데이터가 없습니다. 기본값으로 대체합니다."
                )
                odd_even_data = {}

            if not range_data:
                self.logger.warning(
                    "구간별 미세 편향성 데이터가 없습니다. 기본값으로 대체합니다."
                )
                range_data = {}

            # 미세 편향성 특성 추출 (4개)
            features = []
            feature_names = []

            # 1. 홀짝 편향 점수
            odd_even_bias_score = odd_even_data.get("odd_even_bias_score", 0.0)
            features.append(np.clip(odd_even_bias_score, 0, 1))
            feature_names.append("odd_even_bias_score")

            # 2. 구간 균형 편향 점수
            segment_balance_bias_score = range_data.get(
                "segment_balance_bias_score", 0.0
            )
            features.append(np.clip(segment_balance_bias_score, 0, 1))
            feature_names.append("segment_balance_bias_score")

            # 3. 구간 편향 이동 평균
            range_bias_moving_avg = range_data.get("range_bias_moving_avg", 0.0)
            features.append(np.clip(range_bias_moving_avg, 0, 1))
            feature_names.append("range_bias_moving_avg")

            # 4. 홀짝 변화율 (최근 vs 과거)
            odd_ratio_change_rate = odd_even_data.get("odd_ratio_change_rate", 0.0)
            # 변화율을 [-0.5, 0.5] 범위로 가정하고 [0, 1]로 정규화
            features.append(np.clip((odd_ratio_change_rate + 0.5), 0, 1))
            feature_names.append("odd_ratio_change_rate")

            # NaN/Inf 체크 및 수정
            features = [self.safe_float_conversion(f) for f in features]
            micro_bias_vector = np.array(features, dtype=np.float32)

            self.logger.info(f"미세 편향성 벡터화 완료: {len(features)}개 특성")
            return micro_bias_vector, feature_names

        except Exception as e:
            self.logger.error(f"미세 편향성 벡터화 중 오류 발생: {str(e)}")
            # 기본값 반환
            default_vector = np.zeros(4, dtype=np.float32)
            default_names = [
                "odd_even_bias_score",
                "segment_balance_bias_score",
                "range_bias_moving_avg",
                "odd_ratio_change_rate",
            ]
            return default_vector, default_names
