"""
머신러닝 기반 로또 번호 후보 생성기

이 모듈은 통계 모델과 머신러닝 모델을 결합하여 로또 번호 후보를 생성합니다.
StatisticalModel로 후보를 생성하고, LightGBM과 XGBoost 모델로 점수를 예측합니다.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, cast
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import joblib
import gc
import random
import json
from collections import defaultdict
from sklearn.cluster import KMeans
from datetime import datetime

from ..utils.unified_logging import get_logger
from ..utils.unified_config import ConfigProxy, load_config
from ..utils.performance_utils import MemoryTracker
from ..utils.batch_controller import CPUBatchProcessor
from ..utils.unified_performance import get_profiler, Profiler
from ..utils.pattern_filter import PatternFilter, get_pattern_filter
from ..utils.normalizer import Normalizer
from ..models.statistical_model import StatisticalModel
from ..models.lightgbm_model import LightGBMModel
from ..models.xgboost_model import XGBoostModel
from ..analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer
from ..shared.types import LotteryNumber
from ..shared.graph_utils import calculate_pair_frequency
from ..utils.unified_report import save_performance_report

# 로거 설정
logger = get_logger(__name__)


class MLCandidateGenerator:
    """머신러닝 기반 로또 번호 후보 생성기"""

    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigProxy]] = None):
        """
        초기화

        Args:
            config: 설정 객체
        """
        # 설정 로드
        if config is None:
            config = load_config()
        elif isinstance(config, dict):
            config = ConfigProxy(config)

        self.config = config
        self.logger = get_logger(__name__)
        self.profiler = get_profiler()  # 전역 프로파일러 인스턴스 사용
        self.memory_tracker = MemoryTracker()

        # ConfigProxy를 Dict로 변환
        config_dict = None
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        else:
            config_dict = config if isinstance(config, dict) else {}

        # 패턴 필터 초기화
        self.pattern_filter = get_pattern_filter(config_dict)

        # 통계 모델 초기화
        self.statistical_model = StatisticalModel(config_dict)

        # 벡터라이저 초기화
        self.pattern_vectorizer = EnhancedPatternVectorizer(config_dict)

        # ML 모델 초기화
        self.lightgbm_model = None
        self.xgboost_model = None

        # 배치 처리기 초기화
        cpu_count = os.cpu_count() or 4
        self.batch_processor = CPUBatchProcessor(
            n_jobs=max(1, cpu_count - 1),  # 하나의 코어는 메인 프로세스에 남겨둠
            batch_size=self.config.safe_get("batch_size", 100),
            max_batch_size=self.config.safe_get("max_batch_size", 500),
            backend=self.config.safe_get("parallel_backend", "threading"),
        )

        # 모델 가중치 설정
        self.lgbm_weight = self.config.safe_get(
            "recommendation.model_weights.lgbm", 0.5
        )
        self.xgb_weight = self.config.safe_get("recommendation.model_weights.xgb", 0.5)

        # 생성 설정 (새로 추가)
        self.generation_config = self.config.safe_get("generation", {})
        self.structured_ratio = self.generation_config.get("structured_ratio", 0.3)
        self.roi_ratio = self.generation_config.get("roi_ratio", 0.3)
        self.model_ratio = self.generation_config.get("model_ratio", 0.4)

        # 정규화 유틸리티 초기화
        self.normalizer = Normalizer(self.config)

        # 🚀 성능 최적화 시스템 초기화
        from ..utils.memory_manager import MemoryManager, MemoryConfig
        from ..utils.cuda_optimizers import get_cuda_optimizer, CudaConfig
        from ..utils.process_pool_manager import get_process_pool_manager
        from ..utils.hybrid_optimizer import get_hybrid_optimizer

        # 메모리 관리자 초기화
        try:
            memory_config = MemoryConfig(
                max_memory_usage=0.85,  # ML 후보 생성은 더 많은 메모리 사용
                use_memory_pooling=True,
                pool_size=256,  # 대용량 후보 생성을 위한 큰 풀
            )
            self.memory_manager = MemoryManager(memory_config)
            self.logger.info("✅ ML 후보 생성기 메모리 관리자 초기화 완료")
        except Exception as e:
            self.logger.warning(f"메모리 관리자 초기화 실패: {e}")
            self.memory_manager = None

        # CUDA 최적화 초기화 (싱글톤 사용)
        try:
            from ..utils.cuda_singleton_manager import (
                get_singleton_cuda_optimizer,
                CudaSingletonConfig,
            )

            cuda_config = CudaSingletonConfig(
                use_amp=True,
                batch_size=256,  # ML 후보 생성은 큰 배치 사용
                use_cudnn=True,
            )
            self.cuda_optimizer = get_singleton_cuda_optimizer(
                config=cuda_config, requester_name="ml_candidate_generator"
            )

            if self.cuda_optimizer:
                self.logger.debug("CUDA 최적화기 연결 완료 (ml_candidate_generator)")
            else:
                self.logger.debug("CUDA 사용 불가능 (ml_candidate_generator)")
        except Exception as e:
            self.logger.debug(f"CUDA 최적화기 연결 실패 (ml_candidate_generator): {e}")
            self.cuda_optimizer = None

        # 프로세스 풀 관리자 초기화
        try:
            pool_config = {
                "max_workers": min(
                    8, os.cpu_count() or 1
                ),  # ML 후보 생성은 최대 워커 사용
                "chunk_size": 200,
                "timeout": 900,  # 더 긴 타임아웃
            }
            self.process_pool_manager = get_process_pool_manager(pool_config)
            self.logger.info("✅ ML 후보 생성기 프로세스 풀 초기화 완료")
        except Exception as e:
            self.logger.warning(f"프로세스 풀 초기화 실패: {e}")
            self.process_pool_manager = None

        # 하이브리드 최적화 초기화
        try:
            hybrid_config = {
                "auto_optimization": True,
                "memory_threshold": 0.8,
                "cpu_threshold": 70.0,
                "gpu_threshold": 0.8,
            }
            self.hybrid_optimizer = get_hybrid_optimizer(hybrid_config)
            self.logger.info("✅ ML 후보 생성기 하이브리드 최적화 초기화 완료")
        except Exception as e:
            self.logger.warning(f"하이브리드 최적화 초기화 실패: {e}")
            self.hybrid_optimizer = None

        # 벡터 캐시 초기화
        try:
            from ..utils.state_vector_cache import get_cache

            self.vector_cache = get_cache(self.config)
        except ImportError:
            self.logger.warning(
                "벡터 캐시 초기화 실패: state_vector_cache 모듈을 가져올 수 없습니다."
            )
            self.vector_cache = None

        self.logger.info("🎉 ML 후보 생성기 성능 최적화 시스템 초기화 완료")

    def _safe_float(self, value: Any) -> float:
        """
        어떤 값이든 안전하게 float로 변환합니다.

        Args:
            value: 변환할 값

        Returns:
            변환된 float 값 (변환 실패 시 0.5 반환)
        """
        try:
            # numpy 배열 또는 스칼라 특별 처리
            if hasattr(value, "item") and callable(getattr(value, "item")):
                return float(value.item())
            # 일반 값은 직접 float 변환
            return float(value)
        except (TypeError, ValueError, AttributeError):
            # 변환 실패 시 기본값 반환
            return 0.5

    def load_ml_models(self) -> Tuple[bool, bool]:
        """
        ML 모델 로드

        Returns:
            LightGBM 로드 성공 여부, XGBoost 로드 성공 여부
        """
        lgbm_success = False
        xgb_success = False

        try:
            # ConfigProxy를 Dict로 변환
            config_dict = None
            if hasattr(self.config, "to_dict"):
                config_dict = self.config.to_dict()
            else:
                config_dict = self.config if isinstance(self.config, dict) else {}

            # LightGBM 모델 로드
            with self.profiler.profile("load_lightgbm_model"):
                self.lightgbm_model = LightGBMModel(config_dict)
                model_path = self.config.safe_get("paths.model_save_dir", "savedModels")
                lgbm_path = Path(model_path) / "lightgbm_model.pkl"

                if lgbm_path.exists():
                    lgbm_success = self.lightgbm_model.load(str(lgbm_path))
                    if lgbm_success:
                        self.logger.info(f"LightGBM 모델 로드 성공: {lgbm_path}")
                    else:
                        self.logger.warning(f"LightGBM 모델 로드 실패: {lgbm_path}")
                else:
                    self.logger.warning(
                        f"LightGBM 모델 파일이 존재하지 않습니다: {lgbm_path}"
                    )

            # XGBoost 모델 로드
            with self.profiler.profile("load_xgboost_model"):
                self.xgboost_model = XGBoostModel(config_dict)
                xgb_path = Path(model_path) / "xgboost_model.pkl"

                if xgb_path.exists():
                    xgb_success = self.xgboost_model.load(str(xgb_path))
                    if xgb_success:
                        self.logger.info(f"XGBoost 모델 로드 성공: {xgb_path}")
                    else:
                        self.logger.warning(f"XGBoost 모델 로드 실패: {xgb_path}")
                else:
                    self.logger.warning(
                        f"XGBoost 모델 파일이 존재하지 않습니다: {xgb_path}"
                    )

            return lgbm_success, xgb_success

        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return False, False

    def generate_candidates(
        self, historical_data: List[LotteryNumber]
    ) -> List[Dict[str, Any]]:
        """
        후보 번호 조합 생성

        Args:
            historical_data: 과거 당첨 번호 데이터

        Returns:
            스코어링된 후보 번호 조합 목록
        """
        with self.profiler.profile("generate_candidates"):
            self.memory_tracker.start()

            # 설정에서 원하는 후보 개수 가져오기
            target_count = self.config.safe_get(
                "statistical_model.candidate_count", 200
            )

            # 새로운 최적화 후보 생성 전략 사용
            if self.generation_config.get("enable_structured_generation", False):
                n_structured = int(target_count * self.structured_ratio)
            else:
                n_structured = 0

            if self.generation_config.get("enable_roi_guided_generation", False):
                n_roi = int(target_count * self.roi_ratio)
            else:
                n_roi = 0

            if self.generation_config.get("enable_model_guided_generation", False):
                n_model = int(target_count * self.model_ratio)
            else:
                n_model = 0

            n_statistical = target_count - (n_structured + n_roi + n_model)

            raw_candidates = []
            candidate_sources = {}

            # 1. 구조화된 후보 생성 (쌍 빈도 + 중심성 + 클러스터 기반)
            if n_structured > 0:
                with self.profiler.profile("structured_generation"):
                    self.logger.info(
                        f"구조화된 후보 생성 중... (목표: {n_structured}개)"
                    )
                    structured_candidates = self._generate_structured_candidates(
                        historical_data, n_structured
                    )
                    raw_candidates.extend(structured_candidates)
                    for cand in structured_candidates:
                        candidate_sources[tuple(cand)] = "structured"
                    self.logger.info(
                        f"구조화된 후보 생성 완료: {len(structured_candidates)}개"
                    )

            # 2. ROI 기반 후보 생성
            if n_roi > 0:
                with self.profiler.profile("roi_generation"):
                    self.logger.info(f"ROI 기반 후보 생성 중... (목표: {n_roi}개)")
                    roi_candidates = self._generate_roi_guided_candidates(
                        historical_data, n_roi
                    )
                    raw_candidates.extend(roi_candidates)
                    for cand in roi_candidates:
                        candidate_sources[tuple(cand)] = "roi"
                    self.logger.info(
                        f"ROI 기반 후보 생성 완료: {len(roi_candidates)}개"
                    )

            # 3. 모델 기반 후보 생성
            if n_model > 0:
                with self.profiler.profile("model_generation"):
                    self.logger.info(f"모델 기반 후보 생성 중... (목표: {n_model}개)")
                    # 역샘플링 기반 후보 생성 사용 여부 확인
                    if self.generation_config.get("enable_inverse_sampling", False):
                        model_candidates = (
                            self._generate_model_guided_candidates_advanced(
                                historical_data, n_model
                            )
                        )
                        generation_method = "역샘플링"
                    else:
                        model_candidates = self._generate_model_guided_candidates(
                            historical_data, n_model
                        )
                        generation_method = "기본"

                    raw_candidates.extend(model_candidates)
                    for cand in model_candidates:
                        candidate_sources[tuple(cand)] = f"model_{generation_method}"
                    self.logger.info(
                        f"모델 기반 후보 생성 완료({generation_method}): {len(model_candidates)}개"
                    )

            # 4. 통계 기반 후보 생성 (기존 방식)
            if n_statistical > 0:
                with self.profiler.profile("statistical_model_generation"):
                    self.logger.info(
                        f"통계 기반 후보 생성 중... (목표: {n_statistical}개)"
                    )
                    statistical_candidates = self._generate_statistical_candidates(
                        historical_data, n_statistical
                    )
                    raw_candidates.extend(statistical_candidates)
                    for cand in statistical_candidates:
                        candidate_sources[tuple(cand)] = "statistical"
                    self.logger.info(
                        f"통계 기반 후보 생성 완료: {len(statistical_candidates)}개"
                    )

            # 중복 제거
            unique_candidates = []
            unique_set = set()
            for cand in raw_candidates:
                cand_tuple = tuple(cand)
                if cand_tuple not in unique_set:
                    unique_set.add(cand_tuple)
                    unique_candidates.append(cand)

            self.logger.info(f"중복 제거 후 총 후보 수: {len(unique_candidates)}")

            # 메모리 정리
            gc.collect()

            # 5. 필터링 및 위험도 점수 계산
            with self.profiler.profile("filtering_risk_scoring"):
                self.logger.info("후보 필터링 및 위험도 점수 계산 중...")
                filtered_candidates = self._filter_and_score_candidates(
                    unique_candidates, historical_data
                )

                # 소스 정보 추가
                for candidate in filtered_candidates:
                    cand_tuple = tuple(candidate["numbers"])
                    if cand_tuple in candidate_sources:
                        candidate["source"] = candidate_sources[cand_tuple]
                    else:
                        candidate["source"] = "unknown"

                self.logger.info(f"필터링 후 후보 수: {len(filtered_candidates)}")

            # 메모리 정리
            gc.collect()

            # 6. ML 모델 로드
            lgbm_loaded, xgb_loaded = self.load_ml_models()

            # 7. ML 모델로 점수 예측
            with self.profiler.profile("ml_scoring"):
                self.logger.info("ML 모델을 사용하여 후보 점수 계산 중...")
                scored_candidates = self._score_candidates_with_ml(
                    filtered_candidates, historical_data, lgbm_loaded, xgb_loaded
                )
                self.logger.info(f"최종 후보 수: {len(scored_candidates)}")

            # 8. 생성된 후보를 파일로 저장
            self._save_candidates_to_file(scored_candidates)

            # 9. 성능 보고서 생성
            self._save_performance_report(scored_candidates, historical_data)

            # 메모리 정리
            self.memory_tracker.stop()
            gc.collect()

            return scored_candidates

    def _generate_structured_candidates(
        self, historical_data: List[LotteryNumber], n_candidates: int
    ) -> List[List[int]]:
        """
        구조화된 후보 생성 (쌍 빈도 + 중심성 + 클러스터 기반)

        Args:
            historical_data: 과거 당첨 번호 데이터
            n_candidates: 생성할 후보 수

        Returns:
            후보 번호 조합 목록
        """
        candidates = []

        try:
            # 쌍 빈도 데이터 로드
            pair_frequency = {}
            pair_freq_path = Path("data/cache/pair_frequency.npy")

            if not pair_freq_path.exists():
                # 파일이 없으면 쌍 빈도 분석 실행
                self.logger.info("쌍 빈도 데이터 파일이 없어 계산합니다.")
                pair_frequency = calculate_pair_frequency(
                    historical_data, logger=self.logger
                )

                # 계산된 값 저장
                cache_dir = Path("data/cache")
                cache_dir.mkdir(exist_ok=True, parents=True)

                # Dictionary를 NumPy 배열로 변환하여 저장
                # 1. 쌍 인덱스 및 값 추출
                pairs = list(pair_frequency.keys())
                values = list(pair_frequency.values())

                # 2. NumPy 배열로 변환
                pair_data = np.array(
                    [(p[0], p[1], v) for p, v in zip(pairs, values)],
                    dtype=[("num1", "i4"), ("num2", "i4"), ("freq", "f4")],
                )

                # 3. 저장
                np.save(pair_freq_path, pair_data)
                self.logger.info(f"쌍 빈도 데이터 저장 완료: {pair_freq_path}")
            else:
                try:
                    # 파일에서 로드
                    pair_data = np.load(pair_freq_path, allow_pickle=True)

                    # NumPy 배열을 Dictionary로 변환
                    for item in pair_data:
                        pair = (int(item["num1"]), int(item["num2"]))
                        pair_frequency[pair] = float(item["freq"])

                    self.logger.info(
                        f"쌍 빈도 데이터 로드 완료: {len(pair_frequency)} 항목"
                    )
                except Exception as e:
                    self.logger.warning(f"쌍 빈도 데이터 로드 실패: {e}")
                    pair_frequency = calculate_pair_frequency(
                        historical_data, logger=self.logger
                    )

            # 2. 세그먼트 중심성 데이터 로드
            segment_centrality_vector = None
            centrality_path = Path("data/cache/segment_centrality_vector.npy")

            if centrality_path.exists():
                try:
                    segment_centrality_vector = np.load(centrality_path)
                    self.logger.info(
                        f"세그먼트 중심성 벡터 로드 완료: {segment_centrality_vector.shape}"
                    )
                except Exception as e:
                    self.logger.warning(f"세그먼트 중심성 벡터 로드 실패: {e}")

            # 3. 클러스터 임베딩 데이터 로드
            cluster_embeddings = None
            embedding_path = Path("data/cache/cluster_embeddings.npy")

            if embedding_path.exists():
                try:
                    cluster_embeddings = np.load(embedding_path)
                    self.logger.info(
                        f"클러스터 임베딩 로드 완료: {cluster_embeddings.shape}"
                    )
                except Exception as e:
                    self.logger.warning(f"클러스터 임베딩 로드 실패: {e}")

            # 4. 쌍 빈도 데이터를 고빈도와 중빈도로 분리 (다양성 향상)
            top_pairs = []
            mid_pairs = []

            if pair_frequency:
                # 빈도 기준 정렬
                sorted_pairs = sorted(
                    pair_frequency.items(), key=lambda x: x[1], reverse=True
                )

                # 상위 50개 고빈도 쌍
                top_pairs = [pair for pair, _ in sorted_pairs[:50]]

                # 중간 빈도 쌍 (100~300 순위)
                mid_range_start = min(100, len(sorted_pairs) - 1)
                mid_range_end = min(300, len(sorted_pairs))
                if mid_range_end > mid_range_start:
                    # 중간 빈도 쌍에서 최대 200개까지 샘플링
                    mid_freq_pairs = [
                        pair for pair, _ in sorted_pairs[mid_range_start:mid_range_end]
                    ]
                    # 랜덤 샘플링 (최대 100개)
                    sample_size = min(100, len(mid_freq_pairs))
                    if sample_size > 0:
                        mid_pairs = random.sample(mid_freq_pairs, sample_size)

            self.logger.info(
                f"고빈도 쌍: {len(top_pairs)}개, 중빈도 쌍: {len(mid_pairs)}개"
            )

            # 5. 상위 10개 중심성 번호 추출
            high_centrality_numbers = []
            if segment_centrality_vector is not None:
                # 세그먼트 중심성 벡터를 개별 번호 중심성으로 변환
                number_centrality = {}
                for i in range(9):  # 9개 세그먼트
                    seg_start = i * 5 + 1
                    seg_end = min((i + 1) * 5, 45)

                    for num in range(seg_start, seg_end + 1):
                        number_centrality[num] = segment_centrality_vector[i]

                # 상위 10개 중심성 번호 추출
                high_centrality_numbers = sorted(
                    number_centrality.keys(),
                    key=lambda x: number_centrality.get(x, 0),
                    reverse=True,
                )[:10]
            else:
                # 중심성 데이터가 없으면 빈도 기준 상위 번호 사용
                frequency_map = self.statistical_model.frequency_map
                high_centrality_numbers = sorted(
                    frequency_map.keys(),
                    key=lambda x: frequency_map.get(x, 0),
                    reverse=True,
                )[:10]

            # 6. 클러스터링 적용 (있는 경우)
            number_clusters = {}
            n_clusters = 5  # 기본 클러스터 수

            if (
                cluster_embeddings is not None
                and isinstance(cluster_embeddings, np.ndarray)
                and cluster_embeddings.size > 0
            ):
                try:
                    # KMeans로 번호 클러스터링
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    kmeans.fit(np.array(cluster_embeddings, dtype=np.float64))

                    # 각 번호의 클러스터 할당
                    for i in range(45):
                        number_clusters[i + 1] = int(kmeans.labels_[i])

                    self.logger.info(f"번호 클러스터링 완료: {n_clusters}개 클러스터")
                except Exception as e:
                    self.logger.warning(f"클러스터링 실패: {e}")

            # 7. 후보 생성
            while len(candidates) < n_candidates:
                new_candidate = self._create_structured_candidate(
                    top_pairs,
                    high_centrality_numbers,
                    cluster_embeddings,
                    mid_pairs=mid_pairs,
                    number_clusters=number_clusters,
                )

                if new_candidate not in candidates:
                    candidates.append(new_candidate)

            self.logger.info(f"구조화된 후보 생성 완료: {len(candidates)}개")

        except Exception as e:
            self.logger.error(f"구조화된 후보 생성 중 오류 발생: {str(e)}")
            # 최소한의 후보 생성
            while len(candidates) < n_candidates:
                candidates.append(sorted(random.sample(range(1, 46), 6)))

        return candidates

    def _create_structured_candidate(
        self,
        top_pairs: List[Tuple[int, int]],
        high_centrality_numbers: List[int],
        cluster_embeddings: Optional[np.ndarray] = None,
        mid_pairs: List[Tuple[int, int]] = [],
        number_clusters: Dict[int, int] = {},
    ) -> List[int]:
        """
        구조화된 단일 후보 생성 (쌍 빈도 + 중심성 + 선택적 클러스터 기반)

        Args:
            top_pairs: 상위 빈도 쌍 목록
            high_centrality_numbers: 높은 중심성 번호 목록
            cluster_embeddings: 클러스터 임베딩 (선택적)
            mid_pairs: 중빈도 쌍 목록
            number_clusters: 번호 클러스터 할당 딕셔너리

        Returns:
            후보 번호 조합 (정렬됨)
        """
        # 선택된 번호 집합
        selected_numbers = set()

        # 클러스터 다양성 추적을 위한 선택된 클러스터
        selected_clusters = set()

        # 클러스터 정보가 있는지 확인
        has_cluster_info = bool(number_clusters)

        # 1. 고빈도 및 중빈도 쌍을 혼합하여 사용 (다양성 향상)
        # 80% 확률로 top_pairs, 20% 확률로 mid_pairs에서 시작
        use_mid_pairs_first = random.random() < 0.2 and mid_pairs

        # 쌍 선택 소스 결정
        primary_pairs = mid_pairs if use_mid_pairs_first and mid_pairs else top_pairs
        secondary_pairs = top_pairs if use_mid_pairs_first and mid_pairs else mid_pairs

        # 첫 번째 쌍 선택 (primary_pairs에서)
        if primary_pairs:
            # 랜덤하게 쌍 선택
            pair_idx = random.randint(0, min(len(primary_pairs) - 1, 49))
            selected_pair = primary_pairs[pair_idx]

            # 번호 추가
            selected_numbers.add(selected_pair[0])
            selected_numbers.add(selected_pair[1])

            # 클러스터 추적
            if has_cluster_info:
                selected_clusters.add(number_clusters.get(selected_pair[0], -1))
                selected_clusters.add(number_clusters.get(selected_pair[1], -1))

        # 두 번째 쌍 선택 (secondary_pairs에서, 있다면)
        if len(selected_numbers) < 4 and secondary_pairs:
            # 10번 시도하여 중복되지 않는 쌍 찾기
            for _ in range(10):
                pair_idx = random.randint(0, min(len(secondary_pairs) - 1, 49))
                selected_pair = secondary_pairs[pair_idx]

                # 기존 번호와 중복되지 않는지 확인
                if (
                    selected_pair[0] not in selected_numbers
                    and selected_pair[1] not in selected_numbers
                ):
                    selected_numbers.add(selected_pair[0])
                    selected_numbers.add(selected_pair[1])

                    # 클러스터 추적
                    if has_cluster_info:
                        selected_clusters.add(number_clusters.get(selected_pair[0], -1))
                        selected_clusters.add(number_clusters.get(selected_pair[1], -1))
                    break

        # 2. 높은 중심성 번호에서 1개 추가
        if len(selected_numbers) < 5 and high_centrality_numbers:
            # 아직 선택되지 않은 중심성 번호만 고려
            available_centrality = [
                num for num in high_centrality_numbers if num not in selected_numbers
            ]

            if available_centrality:
                # 랜덤하게 하나 선택
                selected_num = random.choice(available_centrality)
                selected_numbers.add(selected_num)

                # 클러스터 추적
                if has_cluster_info:
                    selected_clusters.add(number_clusters.get(selected_num, -1))

        # 3. 클러스터 다양성을 위한 추가 - 최소 3개 이상의 클러스터에서 번호 선택
        # 클러스터 정보가 있고 아직 번호를 더 선택해야 하는 경우
        if has_cluster_info and len(selected_numbers) < 6:
            # 클러스터 수 결정 (최소 3개, 최대 5개)
            n_clusters = max(3, min(5, len(number_clusters.values())))

            # 현재 선택된 고유 클러스터 수 확인
            if len(selected_clusters) < 3:
                # 아직 포함되지 않은 클러스터 찾기
                missing_clusters = [
                    c
                    for c in range(n_clusters)
                    if c not in selected_clusters and c != -1
                ]

                # 각 클러스터에서 번호 추가 (최소 3개 클러스터까지)
                for cluster_id in missing_clusters:
                    if len(selected_numbers) >= 6 or len(selected_clusters) >= 3:
                        break

                    # 해당 클러스터에 속한 번호 중 아직 선택되지 않은 번호 찾기
                    cluster_numbers = [
                        num
                        for num, cluster in number_clusters.items()
                        if cluster == cluster_id and num not in selected_numbers
                    ]

                    if cluster_numbers:
                        selected_num = random.choice(cluster_numbers)
                        selected_numbers.add(selected_num)
                        selected_clusters.add(number_clusters.get(selected_num, -1))

        # 4. 나머지 번호는 빈도 기반으로 채우기
        while len(selected_numbers) < 6:
            if (
                hasattr(self.statistical_model, "frequency_map")
                and self.statistical_model.frequency_map
            ):
                frequency_map = self.statistical_model.frequency_map

                # 클러스터 다양성 고려
                if has_cluster_info and len(selected_clusters) < 3:
                    # 아직 포함되지 않은 클러스터 찾기
                    missing_clusters = [
                        c
                        for c in range(len(set(number_clusters.values())))
                        if c not in selected_clusters and c != -1
                    ]

                    if missing_clusters:
                        # 해당 클러스터에 속한 번호들 중 아직 선택되지 않은 번호 찾기
                        available_numbers = []
                        for cluster_id in missing_clusters:
                            cluster_numbers = [
                                num
                                for num, cluster in number_clusters.items()
                                if cluster == cluster_id and num not in selected_numbers
                            ]
                            available_numbers.extend(cluster_numbers)

                        if available_numbers:
                            # 빈도 기준으로 정렬
                            available_numbers.sort(
                                key=lambda x: frequency_map.get(x, 0), reverse=True
                            )
                            selected_num = available_numbers[0]
                            selected_numbers.add(selected_num)
                            selected_clusters.add(number_clusters.get(selected_num, -1))
                            continue

                # 일반적인 빈도 기반 선택
                top_numbers = sorted(
                    [num for num in range(1, 46) if num not in selected_numbers],
                    key=lambda x: frequency_map.get(x, 0),
                    reverse=True,
                )[:20]

                if top_numbers:
                    selected_numbers.add(random.choice(top_numbers))
                    continue

            # 빈도 맵이 없거나 문제가 있으면 무작위 선택
            remaining_numbers = [
                num for num in range(1, 46) if num not in selected_numbers
            ]
            selected_num = random.choice(remaining_numbers)
            selected_numbers.add(selected_num)

            # 클러스터 추적
            if has_cluster_info:
                selected_clusters.add(number_clusters.get(selected_num, -1))

        # 최종 리스트로 변환 및 정렬
        return sorted(list(selected_numbers))

    def _generate_statistical_candidates(
        self, historical_data: List[LotteryNumber], n_candidates: int = 200
    ) -> List[List[int]]:
        """
        StatisticalModel을 사용하여 초기 후보 생성

        Args:
            historical_data: 과거 당첨 번호 데이터
            n_candidates: 생성할 후보 수

        Returns:
            후보 번호 조합 목록
        """
        candidates = []

        try:
            # StatisticalModel의 predict 메서드를 사용하여 기본 예측 생성
            prediction = self.statistical_model.predict(historical_data)
            if prediction and hasattr(prediction, "numbers"):
                candidates.append(prediction.numbers)

            # 추가 후보 생성
            frequency_map = {}
            recency_map = {}

            # StatisticalModel 학습
            self.statistical_model.train(historical_data)

            # 빈도 및 최근성 맵 가져오기
            if hasattr(self.statistical_model, "frequency_map"):
                frequency_map = self.statistical_model.frequency_map
            if hasattr(self.statistical_model, "recency_map"):
                recency_map = self.statistical_model.recency_map

            # 전체 번호 목록
            all_numbers = list(range(1, 46))

            # 번호별 통합 점수 계산
            number_scores = {}
            freq_weight = self.config.safe_get("frequency_weights.long_term", 0.6)
            recent_weight = 1.0 - freq_weight

            for num in all_numbers:
                freq_score = frequency_map.get(num, 0.0)
                recent_score = 1.0 - recency_map.get(
                    num, 1.0
                )  # 최근성 점수 반전 (1에 가까울수록 좋음)
                combined_score = freq_weight * freq_score + recent_weight * recent_score
                number_scores[num] = combined_score

            # 점수 기준 상위 번호 선택
            sorted_numbers = sorted(
                all_numbers, key=lambda x: number_scores.get(x, 0.0), reverse=True
            )

            # 필요한 후보 수
            target_count = n_candidates

            # 상위 번호들로 다양한 조합 생성
            top_n = min(25, len(sorted_numbers))  # 상위 25개 번호 사용
            top_numbers = sorted_numbers[:top_n]

            while len(candidates) < target_count and len(top_numbers) >= 6:
                # 상위 번호들에서 6개 선택
                combination = sorted(random.sample(top_numbers, 6))

                if combination not in candidates:
                    candidates.append(combination)

            # 후보 수가 여전히 부족하면 무작위 조합 추가
            while len(candidates) < target_count:
                combination = sorted(random.sample(all_numbers, 6))

                if combination not in candidates:
                    candidates.append(combination)

            self.logger.info(f"통계 모델로 {len(candidates)}개 후보 생성 완료")

        except Exception as e:
            self.logger.error(f"후보 생성 중 오류 발생: {str(e)}")
            # 최소한의 후보 생성
            for _ in range(10):
                candidates.append(sorted(random.sample(range(1, 46), 6)))

        return candidates

    def _filter_and_score_candidates(
        self, candidates: List[List[int]], historical_data: List[LotteryNumber]
    ) -> List[Dict[str, Any]]:
        """
        후보 필터링 및 점수 계산

        Args:
            candidates: 후보 번호 조합 목록
            historical_data: 과거 당첨 번호 데이터

        Returns:
            필터링 및 점수 계산된 후보 목록

        Raises:
            ValueError: 필터링 후 남은 후보가 없는 경우
            RuntimeError: 필터링 또는 점수 계산 중 오류 발생 시
        """
        # 필터링 설정
        filtering_config = self.config.get("filtering", {})
        use_rule_filter = filtering_config.get("use_rule_filter", True)
        use_risk_filter = filtering_config.get("use_risk_filter", True)
        use_roi_filter = filtering_config.get("use_roi_filter", True)

        # 리스크 필터 임계값
        risk_threshold = filtering_config.get("risk_threshold", 0.8)

        # ROI 필터 임계값
        roi_threshold = filtering_config.get("roi_threshold", -0.3)

        # 패턴 필터 가져오기
        pattern_filter = get_pattern_filter()

        # 결과 목록
        filtered_candidates = []

        # 유효한 후보 수 추적
        valid_count = 0

        # 필터별 거부 수 추적
        rule_rejected = 0
        risk_rejected = 0
        roi_rejected = 0

        # 패턴 분석기 가져오기 (재사용)
        from ..analysis.pattern_analyzer import PatternAnalyzer

        pattern_analyzer = PatternAnalyzer(self.config)

        # 각 후보에 대해 처리
        for numbers in candidates:
            # 번호 오름차순 정렬 (일관성 유지)
            numbers = sorted(numbers)

            # 1. 규칙 기반 필터링 (홀짝, 범위 등)
            if use_rule_filter:
                if not pattern_filter.is_valid_combination(numbers):
                    rule_rejected += 1
                    continue

            # 2. 패턴 특성 추출
            pattern_features = pattern_analyzer.extract_pattern_features(
                numbers, historical_data
            )

            # 3. 리스크 점수 계산 및 필터링
            risk_score = pattern_features.get("risk_score", 0.5)
            if use_risk_filter and risk_score > risk_threshold:
                risk_rejected += 1
                continue

            # 4. ROI 가중치 계산 및 필터링
            roi_weight = pattern_features.get("roi_weight", 0.0)
            if use_roi_filter and roi_weight < roi_threshold:
                roi_rejected += 1
                continue

            # 유효한 후보로 카운트
            valid_count += 1

            # 결과 사전에 추가 (번호, 점수, 특성 등)
            candidate_result = {
                "numbers": numbers,
                "risk_score": risk_score,
                "roi_weight": roi_weight,
                "pattern_features": pattern_features,
            }

            # 추가 점수 계산 (빈도, 트렌드 등)
            if "trend_score_avg" in pattern_features:
                candidate_result["trend_score"] = pattern_features["trend_score_avg"]

            if "frequent_pair_score" in pattern_features:
                candidate_result["pair_score"] = pattern_features["frequent_pair_score"]

            # 결과 목록에 추가
            filtered_candidates.append(candidate_result)

        # 필터링 결과 로깅
        self.logger.info(
            f"필터링 결과: 총 {len(candidates)}개 중 {valid_count}개 통과 "
            f"(규칙 필터: {rule_rejected}개 거부, 리스크 필터: {risk_rejected}개 거부, ROI 필터: {roi_rejected}개 거부)"
        )

        # 필터링 후 후보가 없을 때 경고
        if not filtered_candidates:
            error_msg = "모든 후보가 필터링되었습니다. 필터 임계값을 조정하세요."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return filtered_candidates

    def _score_candidates_with_ml(
        self,
        candidates: List[Dict[str, Any]],
        historical_data: List[LotteryNumber],
        lgbm_loaded: bool,
        xgb_loaded: bool,
    ) -> List[Dict[str, Any]]:
        """
        ML 모델을 사용하여 후보 점수 계산

        Args:
            candidates: 후보 번호 조합 목록 (사전 형태)
            historical_data: 과거 당첨 번호 데이터
            lgbm_loaded: LightGBM 모델 로드 여부
            xgb_loaded: XGBoost 모델 로드 여부

        Returns:
            ML 점수가 추가된 후보 목록

        Raises:
            ValueError: 모든 ML 모델 로드에 실패한 경우
            RuntimeError: 점수 계산 중 오류 발생 시
        """
        # 모델 로드 여부 확인
        if not lgbm_loaded and not xgb_loaded:
            error_msg = "ML 모델이 로드되지 않았습니다. 점수 계산을 진행할 수 없습니다."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 모델 객체 확인
        if lgbm_loaded and self.lightgbm_model is None:
            error_msg = "LightGBM 모델이 로드되었지만 객체가 없습니다."
            self.logger.error(error_msg)
            lgbm_loaded = False

        if xgb_loaded and self.xgboost_model is None:
            error_msg = "XGBoost 모델이 로드되었지만 객체가 없습니다."
            self.logger.error(error_msg)
            xgb_loaded = False

        # 다시 확인 (둘 다 없으면 오류)
        if not lgbm_loaded and not xgb_loaded:
            error_msg = "유효한 ML 모델이 없습니다. 점수 계산을 진행할 수 없습니다."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 타이머 시작
        start_time = time.time()
        self.logger.info(f"{len(candidates)}개 후보에 대해 ML 점수 계산 시작")

        # 각 후보에 대해 ML 점수 계산
        for i, candidate in enumerate(candidates):
            numbers = candidate["numbers"]
            lgbm_score = 0.0
            xgb_score = 0.0
            combined_score = 0.0

            # LightGBM 모델 사용
            if lgbm_loaded and self.lightgbm_model is not None:
                # 특성 추출
                lgbm_features = self.lightgbm_model.extract_features_for_prediction(
                    numbers, historical_data
                )

                # 특성 행렬 준비 (각 번호별로 동일한 특성 + 번호 자체)
                X_lgbm = np.zeros((6, len(lgbm_features) + 1))
                for j, num in enumerate(numbers):
                    X_lgbm[j, :-1] = lgbm_features
                    X_lgbm[j, -1] = num / 45.0  # 정규화된 번호

                # 확률 예측
                lgbm_probas = self.lightgbm_model.predict_proba(X_lgbm)
                lgbm_score = float(np.mean(lgbm_probas))
                candidate["lgbm_score"] = lgbm_score

            # XGBoost 모델 사용
            if xgb_loaded and self.xgboost_model is not None:
                # 특성 추출
                xgb_features = self.xgboost_model.extract_features_for_prediction(
                    numbers, historical_data
                )

                # 특성 행렬 준비 (각 번호별로 동일한 특성 + 번호 자체)
                X_xgb = np.zeros((6, len(xgb_features) + 1))
                for j, num in enumerate(numbers):
                    X_xgb[j, :-1] = xgb_features
                    X_xgb[j, -1] = num / 45.0  # 정규화된 번호

                # 확률 예측
                xgb_probas = self.xgboost_model.predict_proba(X_xgb)
                xgb_score = float(np.mean(xgb_probas))
                candidate["xgb_score"] = xgb_score

            # 점수 결합 (가중 평균)
            if lgbm_loaded and xgb_loaded:
                lgbm_weight = self.config.get("model_weights", {}).get("lgbm", 0.5)
                xgb_weight = self.config.get("model_weights", {}).get("xgb", 0.5)
                combined_score = (lgbm_score * lgbm_weight) + (xgb_score * xgb_weight)
            elif lgbm_loaded:
                combined_score = lgbm_score
            elif xgb_loaded:
                combined_score = xgb_score

            # 리스크 점수를 반영한 최종 점수 계산
            risk_score = candidate.get("risk_score", 0.5)
            risk_weight = self.config.get("risk_weight", 0.3)

            # ROI 가중치를 반영한 점수 계산
            roi_weight = candidate.get("roi_weight", 0.0)
            roi_factor = self.config.get("roi_factor", 0.2)

            # 최종 점수 = ML 점수 * (1 - 리스크 가중치 * 리스크 점수) + ROI 가중치 * ROI 계수
            final_score = (
                combined_score * (1 - risk_weight * risk_score)
                + roi_weight * roi_factor
            )

            # 0-1 범위로 클리핑
            final_score = max(0.0, min(1.0, final_score))

            # 점수 저장
            candidate["ml_score"] = combined_score
            candidate["combined_score"] = final_score

            # 진행 상황 로깅 (100개마다)
            if (i + 1) % 100 == 0:
                self.logger.debug(f"ML 점수 계산 진행 중: {i+1}/{len(candidates)}")

        # 타이머 종료
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"ML 점수 계산 완료: {len(candidates)}개 후보, 소요 시간: {elapsed_time:.2f}초"
        )

        # 점수 기준 내림차순 정렬
        candidates.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)

        return candidates
