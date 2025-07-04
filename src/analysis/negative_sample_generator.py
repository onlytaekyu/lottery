"""
비당첨 샘플 생성 모듈

이 모듈은 학습 데이터의 다양성을 높이기 위해 비당첨 번호 조합을 생성하는 기능을 제공합니다.
"""

import numpy as np
import random
import time
import os
import gc
import threading
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional, Union

# logging 제거 - unified_logging 사용
import json
import psutil

from ..utils.unified_logging import get_logger
from ..utils.memory_manager import get_memory_manager
from ..utils.batch_controller import DynamicBatchSizeController, CPUBatchProcessor
from ..utils.cuda_singleton_manager import get_singleton_cuda_optimizer
from ..shared.types import LotteryNumber
from .pattern_analyzer import PatternAnalyzer
from .base_analyzer import BaseAnalyzer

import torch

# 로거 설정
logger = get_logger(__name__)


class NegativeSampleGenerator(BaseAnalyzer[Dict[str, Any]]):
    """비당첨 샘플 생성 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        super().__init__(config, name="negative_sampler")

        # GPU 가속 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            logger.info(f"🚀 GPU 가속 활성화: {torch.cuda.get_device_name()}")
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        else:
            logger.warning("⚠️ GPU 사용 불가, CPU 모드로 실행")

        # 캐시 디렉토리 설정
        try:
            self.cache_dir = self.config.get("paths", {}).get("cache_dir", "data/cache")
        except KeyError:
            raise KeyError("설정에서 'paths.cache_dir' 키를 찾을 수 없습니다.")
        Path(self.cache_dir).mkdir(exist_ok=True, parents=True)

        negative_sampler_config = self.config.get("negative_sampler", {})

        # 컨트롤러 초기화
        self.batch_controller = DynamicBatchSizeController(
            config=self.config,
            initial_batch_size=negative_sampler_config.get("batch_size", 2000),
            min_batch_size=500,
            max_batch_size=5000,
        )

        # 패턴 분석기 초기화
        self.pattern_analyzer = PatternAnalyzer(config)

        # CPU/GPU별 최적화 도구 설정
        if self.use_gpu:
            self.cuda_optimizer = get_singleton_cuda_optimizer()
            self.amp_scaler = torch.cuda.amp.GradScaler()
            self.cpu_batch_processor = None
        else:
            self.cuda_optimizer = None
            self.cpu_batch_processor = CPUBatchProcessor(
                n_jobs=negative_sampler_config.get("vectorize_workers", -1),
                batch_size=negative_sampler_config.get("vectorize_batch", 1000),
                backend="multiprocessing",  # GIL 회피
            )

        # 진행 상황 추적 변수
        self.progress = 0
        self.total = 0

        # 메모리 관리자 (싱글톤)
        self.memory_tracker = get_memory_manager()

        # 진행 상황 추적 잠금
        self._progress_lock = threading.Lock()

        # 성능 모니터링
        self.performance_stats = {
            "total_samples": 0,
            "gpu_utilization": 0.0,
            "memory_usage_mb": 0.0,
            "processing_time": 0.0,
            "samples_per_second": 0.0,
        }

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """BaseAnalyzer 인터페이스 구현"""
        sample_size = kwargs.get("sample_size", 100000)
        return self.generate_samples(historical_data, sample_size)

    def generate_samples(
        self, draw_data: List[LotteryNumber], sample_size: int = 100000
    ) -> Dict[str, Any]:
        """
        비당첨 조합 샘플 생성

        Args:
            draw_data: 당첨 번호 데이터
            sample_size: 생성할 샘플 크기

        Returns:
            생성 결과 정보
        """
        self.logger.info(f"비당첨 조합 생성 시작: 목표 {sample_size:,}개")

        # 전역 시작 시간
        global_start_time = time.time()

        # 경고 메시지 저장 리스트
        warnings_list: List[str] = []

        # 이미 당첨된 번호 조합 set으로 저장 (빠른 검색)
        existing_combinations = self._extract_winning_combinations(draw_data)

        # 진행 상황 초기화
        self.progress = 0
        self.total = sample_size

        # 비당첨 조합 생성
        negative_samples = self._generate_samples(existing_combinations, sample_size)

        # 벡터화 수행
        vector_path = self._vectorize_samples(negative_samples, draw_data, sample_size)

        # 전체 실행 시간 계산
        global_end_time = time.time()
        elapsed_time = global_end_time - global_start_time

        # 메모리 사용량 추정
        memory_used_mb = len(negative_samples) * 6 * 4 / (1024 * 1024)  # 대략적 계산

        # 성능 보고서 생성
        report_path = self._generate_performance_report(
            start_time=global_start_time,
            end_time=global_end_time,
            sample_count=len(negative_samples),
            memory_used_mb=memory_used_mb,
            vector_path=vector_path,
            warnings=warnings_list,
        )

        self.logger.info(
            f"비당첨 조합 생성 완료: {len(negative_samples):,}개 ({elapsed_time:.2f}초)"
        )
        self.logger.info(f"메모리 사용: {memory_used_mb:.2f}MB")
        self.logger.info(f"성능 보고서: {report_path}")

        return {
            "success": True,
            "elapsed_time": elapsed_time,
            "sample_count": len(negative_samples),
            "memory_used_mb": memory_used_mb,
            "raw_path": self._save_raw_samples(negative_samples, sample_size),
            "vector_path": vector_path,
            "report_path": report_path,
        }

    def _extract_winning_combinations(
        self, draw_data: List[LotteryNumber]
    ) -> Set[Tuple[int, ...]]:
        """
        당첨 번호 조합 추출

        Args:
            draw_data: 당첨 번호 데이터

        Returns:
            당첨 번호 조합 집합
        """
        self.logger.info(f"당첨 번호 조합 추출 시작: {len(draw_data)}개")
        combinations = set()

        for draw in draw_data:
            # 정렬된 번호를 튜플로 변환하여 집합에 추가
            combinations.add(tuple(sorted(draw.numbers)))

        self.logger.info(f"당첨 번호 조합 추출 완료: {len(combinations)}개")
        return combinations

    def _generate_samples(
        self, existing_combinations: Set[Tuple[int, ...]], sample_size: int
    ) -> List[List[int]]:
        """
        비당첨 조합 생성 (균형잡힌 샘플링)
        - GPU 우선, CPU 폴백
        """
        if self.use_gpu:
            logger.info("🚀 GPU를 사용하여 비당첨 조합 생성")
            return self._generate_samples_gpu(existing_combinations, sample_size)
        else:
            logger.info("💻 CPU를 사용하여 비당첨 조합 생성")
            n_random = int(sample_size * 0.7)
            n_pattern = sample_size - n_random

            samples = self._random_negative_samples_cpu(existing_combinations, n_random)
            samples.extend(
                self._pattern_based_negative_samples(existing_combinations, n_pattern)
            )

            # 중복 제거 및 최종 반환
            final_samples = []
            seen = set(map(tuple, samples))
            for s in samples:
                if tuple(s) in seen:
                    final_samples.append(s)
                    seen.remove(tuple(s))
            return final_samples[:sample_size]

    def _generate_samples_gpu(
        self, existing_combinations: Set[Tuple[int, ...]], sample_size: int
    ) -> List[List[int]]:
        """
        GPU를 사용한 고성능 비당첨 조합 생성
        - 중복 제거 포함, 완전히 벡터화된 방식
        """
        all_numbers = torch.arange(1, 46, device=self.device, dtype=torch.int16)
        existing_tensor = torch.tensor(
            list(existing_combinations), device=self.device, dtype=torch.int16
        )

        final_samples = torch.empty((0, 6), device=self.device, dtype=torch.int16)

        # 목표 수량보다 더 많이 생성하여 필터링 손실 보상
        OVERSAMPLING_FACTOR = 1.5

        with torch.no_grad():
            while len(final_samples) < sample_size:
                needed = sample_size - len(final_samples)
                batch_size = int(needed * OVERSAMPLING_FACTOR)
                batch_size = min(batch_size, 200000)  # 메모리 제한

                # 1. 랜덤 인덱스 생성
                # torch.rand에서 직접 topk를 사용하여 고유 인덱스 추출
                _, random_indices = torch.topk(
                    torch.rand(batch_size, 45, device=self.device), k=6, dim=1
                )

                # 2. 인덱스를 번호로 변환
                new_samples = all_numbers[random_indices]
                new_samples, _ = torch.sort(new_samples, dim=1)

                # 3. 기존 당첨 번호와 중복 제거
                # (batch, 1, 6) vs (1, M, 6) -> (batch, M)
                is_in_existing = (
                    (new_samples.unsqueeze(1) == existing_tensor.unsqueeze(0))
                    .all(dim=2)
                    .any(dim=1)
                )
                new_samples = new_samples[~is_in_existing]

                # 4. 생성된 배치 내 중복 제거
                # 정렬된 텐서를 사용하여 고유값 찾기 (더 효율적)
                unique_mask = torch.cat(
                    [
                        torch.tensor([True], device=self.device),
                        (new_samples[1:] != new_samples[:-1]).any(dim=1),
                    ]
                )
                new_samples = new_samples[unique_mask]

                # 5. 최종 샘플셋과 중복 제거
                if len(final_samples) > 0:
                    is_in_final = (
                        (new_samples.unsqueeze(1) == final_samples.unsqueeze(0))
                        .all(dim=2)
                        .any(dim=1)
                    )
                    new_samples = new_samples[~is_in_final]

                final_samples = torch.cat([final_samples, new_samples], dim=0)

                # 메모리 정리
                del new_samples, random_indices, unique_mask
                if "is_in_existing" in locals():
                    del is_in_existing
                if "is_in_final" in locals():
                    del is_in_final
                torch.cuda.empty_cache()

        return final_samples[:sample_size].cpu().tolist()

    def _random_negative_samples_cpu(
        self, existing_combinations: Set[Tuple[int, ...]], n: int
    ) -> List[List[int]]:
        """단순 랜덤 비당첨 조합 생성 (CPU)"""
        samples = []
        all_numbers = list(range(1, 46))

        seen_combinations = existing_combinations.copy()

        while len(samples) < n:
            combo = tuple(sorted(random.sample(all_numbers, 6)))
            if combo not in seen_combinations:
                samples.append(list(combo))
                seen_combinations.add(combo)
        return samples

    def _pattern_based_negative_samples(
        self, existing_combinations: Set[Tuple[int, ...]], n: int
    ) -> List[List[int]]:
        """패턴 기반(홀짝, 연속, 분포 등) 극단/정상 케이스 포함"""
        samples = []
        all_numbers = np.arange(1, 46)
        # 극단: 모두 홀수
        if n > 0:
            odd = [i for i in all_numbers if i % 2 == 1]
            if len(odd) >= 6:
                samples.append(sorted(random.sample(odd, 6)))
        # 극단: 모두 짝수
        if n > 1:
            even = [i for i in all_numbers if i % 2 == 0]
            if len(even) >= 6:
                samples.append(sorted(random.sample(even, 6)))
        # 극단: 연속번호
        if n > 2:
            start = random.randint(1, 40)
            samples.append(list(range(start, start + 6)))
        # 정상 분포: 홀짝 3:3, 분산 높은 조합
        while len(samples) < n:
            combo = random.sample(list(all_numbers), 6)
            odds = sum(1 for x in combo if x % 2 == 1)
            evens = 6 - odds
            if odds == 3 and evens == 3:
                samples.append(sorted(combo))
        # 중복/당첨 제외
        filtered = []
        seen = set()
        for s in samples:
            t = tuple(sorted(s))
            if t not in existing_combinations and t not in seen:
                filtered.append(list(t))
                seen.add(t)
            if len(filtered) >= n:
                break
        return filtered

    def auto_label(
        self, samples: List[List[int]], positive: bool = False
    ) -> List[Dict[str, Any]]:
        """ML 학습용 레이블 자동 부여 (positive: 당첨, negative: 비당첨)"""
        label = 1 if positive else 0
        return [{"numbers": s, "label": label} for s in samples]

    def _save_raw_samples(
        self, negative_samples: List[List[int]], sample_size: int
    ) -> str:
        """
        비당첨 조합 저장

        Args:
            negative_samples: 비당첨 번호 조합 목록
            sample_size: 샘플 크기

        Returns:
            저장 파일 경로
        """
        # 고정 파일 경로 (타임스탬프 제거)
        file_path = Path(self.cache_dir) / f"negative_samples_{sample_size}.npy"

        # NumPy 배열로 변환하여 저장
        np_array = np.array(negative_samples, dtype=np.int16)
        np.save(file_path, np_array)

        # 최신 버전 링크 (덮어쓰기)
        latest_path = Path(self.cache_dir) / "negative_samples_latest.npy"
        np.save(latest_path, np_array)

        self.logger.info(f"비당첨 조합 저장 완료: {file_path}")
        self.logger.info(f"최신 버전: {latest_path}")

        return str(file_path)

    def _vectorize_samples(
        self,
        negative_samples: List[List[int]],
        draw_data: List[LotteryNumber],
        sample_size: int,
    ) -> str:
        """
        GPU 가속 비당첨 조합 벡터화

        Args:
            negative_samples: 비당첨 번호 조합 목록
            draw_data: 당첨 번호 데이터
            sample_size: 샘플 크기

        Returns:
            저장 파일 경로
        """
        self.logger.info(f"🚀 GPU 가속 벡터화 시작: {len(negative_samples):,}개")
        start_time = time.time()

        # GPU 사용 가능 여부에 따라 처리 방식 선택
        if self.use_gpu and len(negative_samples) > 1000:
            vector_path = self._vectorize_samples_gpu(
                negative_samples, draw_data, sample_size
            )
        else:
            vector_path = self._vectorize_samples_cpu(
                negative_samples, draw_data, sample_size
            )

        # 성능 통계 업데이트
        processing_time = time.time() - start_time
        self.performance_stats.update(
            {
                "total_samples": len(negative_samples),
                "processing_time": processing_time,
                "samples_per_second": len(negative_samples) / processing_time,
            }
        )

        if self.use_gpu:
            self.performance_stats["gpu_utilization"] = self._get_gpu_utilization()
            self.performance_stats["memory_usage_mb"] = self._get_gpu_memory_usage()

        self.logger.info(
            f"✅ 벡터화 완료: {processing_time:.2f}초 ({self.performance_stats['samples_per_second']:.0f} samples/sec)"
        )

        return vector_path

    def _vectorize_samples_gpu(
        self,
        negative_samples: List[List[int]],
        draw_data: List[LotteryNumber],
        sample_size: int,
    ) -> str:
        """GPU 가속 벡터화"""
        self.logger.info("🔥 GPU 가속 벡터화 실행")

        try:
            # 배치 크기 최적화
            batch_size = self.batch_controller.get_current_batch_size()
            self.logger.info(f"배치 크기: {batch_size}")

            # 특성 벡터 크기 확인
            expected_num_features = self._estimate_feature_vector_size()
            actual_num_features = self._get_actual_vector_size(
                negative_samples[0], draw_data
            )

            # 결과 저장 배열
            feature_vectors = np.zeros(
                (len(negative_samples), actual_num_features), dtype=np.float32
            )

            # GPU 메모리 사전 할당
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

                # 벡터화 결과 저장
                processed_count = 0

                # 배치 단위로 GPU 처리
                for i in range(0, len(negative_samples), batch_size):
                    batch = negative_samples[i : i + batch_size]

                    try:
                        # GPU 배치 벡터화
                        batch_vectors = self._process_batch_gpu(batch, draw_data)

                        # 결과 저장
                        for j, vector in enumerate(batch_vectors):
                            if i + j < len(feature_vectors):
                                feature_vectors[i + j] = vector

                        processed_count += len(batch_vectors)

                        # 진행 상황 업데이트
                        with self._progress_lock:
                            self.progress = processed_count
                            progress_pct = (self.progress / len(negative_samples)) * 100
                            self.logger.info(f"GPU 벡터화 진행: {progress_pct:.1f}%")

                        # 동적 배치 크기 조정
                        if i % (batch_size * 3) == 0:  # 3배치마다 체크
                            self._adjust_batch_size_based_on_memory()
                            batch_size = self.batch_controller.get_current_batch_size()

                    except torch.cuda.OutOfMemoryError:
                        self.logger.warning("GPU 메모리 부족, 배치 크기 감소")
                        batch_size = self.batch_controller.handle_oom()
                        torch.cuda.empty_cache()

                        # 작은 배치로 재시도
                        small_batch = batch[:batch_size]
                        batch_vectors = self._process_batch_gpu(small_batch, draw_data)

                        # 결과 저장
                        for j, vector in enumerate(batch_vectors):
                            if i + j < len(feature_vectors):
                                feature_vectors[i + j] = vector

                        processed_count += len(batch_vectors)

                # GPU 메모리 정리
                torch.cuda.empty_cache()

        except Exception as e:
            self.logger.error(f"GPU 벡터화 실패: {e}")
            # CPU 폴백
            return self._vectorize_samples_cpu(negative_samples, draw_data, sample_size)

        # 결과 저장
        return self._save_vectorized_results(feature_vectors, sample_size)

    def _process_batch_gpu(
        self, batch: List[List[int]], draw_data: List[LotteryNumber]
    ) -> List[np.ndarray]:
        """GPU 배치 처리"""
        try:
            # PyTorch 2.0+ AMP API 직접 사용
            with torch.cuda.amp.autocast():
                batch_vectors = []

                # 배치 내 각 조합 처리
                for combination in batch:
                    # 빠른 패턴 특성 추출
                    features = self._extract_pattern_features_fast(
                        combination, draw_data
                    )

                    # 벡터화
                    vector = self._vectorize_features_gpu(features)
                    batch_vectors.append(vector)

                return batch_vectors

        except Exception as e:
            self.logger.error(f"GPU 배치 처리 실패: {e}")
            # CPU 폴백
            return [
                self._vectorize_combination_cpu(combo, draw_data) for combo in batch
            ]

    def _extract_pattern_features_fast(
        self, combination: List[int], draw_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """빠른 패턴 특성 추출 (GPU 최적화)"""
        try:
            # 기본 특성 빠른 계산
            features = {
                "max_consecutive_length": self._calc_consecutive_fast(combination),
                "total_sum": sum(combination),
                "odd_count": sum(1 for x in combination if x % 2 == 1),
                "even_count": sum(1 for x in combination if x % 2 == 0),
                "gap_avg": self._calc_gap_avg_fast(combination),
                "gap_std": self._calc_gap_std_fast(combination),
                "range_counts": self._calc_range_counts_fast(combination),
                "cluster_overlap_ratio": 0.3,  # 기본값 (빠른 처리)
                "frequent_pair_score": 0.05,  # 기본값
                "roi_weight": 1.0,  # 기본값
                "consecutive_score": 0.0,  # 기본값
                "trend_score_avg": 0.5,  # 기본값
                "trend_score_max": 0.8,  # 기본값
                "trend_score_min": 0.2,  # 기본값
                "risk_score": 0.5,  # 기본값
            }

            return features

        except Exception as e:
            self.logger.debug(f"빠른 특성 추출 실패: {e}")
            # 기본 특성 반환
            return {
                "max_consecutive_length": 0,
                "total_sum": sum(combination),
                "odd_count": 3,
                "even_count": 3,
                "gap_avg": 7.5,
                "gap_std": 5.0,
                "range_counts": [1, 1, 1, 1, 2],
                "cluster_overlap_ratio": 0.3,
                "frequent_pair_score": 0.05,
                "roi_weight": 1.0,
                "consecutive_score": 0.0,
                "trend_score_avg": 0.5,
                "trend_score_max": 0.8,
                "trend_score_min": 0.2,
                "risk_score": 0.5,
            }

    def _calc_consecutive_fast(self, combination: List[int]) -> int:
        """빠른 연속 번호 계산"""
        if len(combination) < 2:
            return 0

        sorted_combo = sorted(combination)
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(sorted_combo)):
            if sorted_combo[i] == sorted_combo[i - 1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        return max_consecutive

    def _calc_gap_avg_fast(self, combination: List[int]) -> float:
        """빠른 간격 평균 계산"""
        if len(combination) < 2:
            return 0.0

        sorted_combo = sorted(combination)
        gaps = [
            sorted_combo[i] - sorted_combo[i - 1] for i in range(1, len(sorted_combo))
        ]
        return sum(gaps) / len(gaps)

    def _calc_gap_std_fast(self, combination: List[int]) -> float:
        """빠른 간격 표준편차 계산"""
        if len(combination) < 2:
            return 0.0

        sorted_combo = sorted(combination)
        gaps = [
            sorted_combo[i] - sorted_combo[i - 1] for i in range(1, len(sorted_combo))
        ]

        if len(gaps) < 2:
            return 0.0

        mean_gap = sum(gaps) / len(gaps)
        variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
        return variance**0.5

    def _calc_range_counts_fast(self, combination: List[int]) -> List[int]:
        """빠른 범위별 개수 계산"""
        ranges = [0, 0, 0, 0, 0]  # 1-9, 10-18, 19-27, 28-36, 37-45

        for num in combination:
            if 1 <= num <= 9:
                ranges[0] += 1
            elif 10 <= num <= 18:
                ranges[1] += 1
            elif 19 <= num <= 27:
                ranges[2] += 1
            elif 28 <= num <= 36:
                ranges[3] += 1
            elif 37 <= num <= 45:
                ranges[4] += 1

        return ranges

    def _vectorize_features_gpu(self, features: Dict[str, Any]) -> np.ndarray:
        """GPU 기반 특성 벡터화"""
        try:
            # 패턴 분석기와 동일한 벡터화 (19차원)
            vector = np.array(
                [
                    features["max_consecutive_length"] / 6.0,
                    features["total_sum"] / 270.0,
                    features["odd_count"] / 6.0,
                    features["even_count"] / 6.0,
                    features["gap_avg"] / 20.0,
                    features["gap_std"] / 15.0,
                    *[count / 6.0 for count in features["range_counts"][:5]],
                    features["cluster_overlap_ratio"],
                    features["frequent_pair_score"] * 10.0,
                    features["roi_weight"] / 2.0,
                    features["consecutive_score"] + 0.3,
                    features["trend_score_avg"] * 10.0,
                    features["trend_score_max"] * 10.0,
                    features["trend_score_min"] * 10.0,
                    features["risk_score"],
                ],
                dtype=np.float32,
            )

            return vector

        except Exception as e:
            self.logger.error(f"GPU 벡터화 실패: {e}")
            # 기본 벡터 반환
            return np.array([0.5] * 19, dtype=np.float32)

    def _vectorize_samples_cpu(
        self,
        negative_samples: List[List[int]],
        draw_data: List[LotteryNumber],
        sample_size: int,
    ) -> str:
        """CPU 기반 벡터화 (폴백)"""
        self.logger.info(f"💻 CPU 벡터화 실행 (CPUBatchProcessor 사용)")

        if not self.cpu_batch_processor:
            logger.error(
                "CPUBatchProcessor가 초기화되지 않았습니다. CPU 모드에서 실행할 수 없습니다."
            )
            # 빈 결과를 저장하고 경로를 반환하거나 예외를 발생시킬 수 있습니다.
            # 여기서는 빈 결과를 저장합니다.
            empty_vectors = np.array([])
            return self._save_vectorized_results(empty_vectors, sample_size)

        actual_num_features = self._get_actual_vector_size(
            negative_samples[0], draw_data
        )

        # 벡터화 작업을 처리할 함수 정의
        def vectorize_worker(combination_batch):
            return [
                self._vectorize_combination_cpu(combo, draw_data)
                for combo in combination_batch
            ]

        # CPUBatchProcessor를 사용하여 병렬 처리
        results = self.cpu_batch_processor.process_batches(
            negative_samples, vectorize_worker
        )

        # 결과 배열 생성 및 채우기
        feature_vectors = np.zeros(
            (len(negative_samples), actual_num_features), dtype=np.float32
        )
        for i, vector in enumerate(results):
            if vector is not None and i < len(feature_vectors):
                feature_vectors[i] = vector

        return self._save_vectorized_results(feature_vectors, sample_size)

    def _get_actual_vector_size(
        self, sample_combination: List[int], draw_data: List[LotteryNumber]
    ) -> int:
        """실제 벡터 크기 확인"""
        try:
            # 빠른 특성 추출로 벡터 크기 확인
            features = self._extract_pattern_features_fast(
                sample_combination, draw_data
            )
            vector = self._vectorize_features_gpu(features)
            return len(vector)
        except Exception as e:
            self.logger.warning(f"벡터 크기 확인 실패: {e}")
            return 19  # 기본 크기

    def _vectorize_combination_cpu(
        self, combination: List[int], draw_data: List[LotteryNumber]
    ) -> np.ndarray:
        """CPU 기반 단일 조합 벡터화"""
        try:
            # 빠른 특성 추출
            features = self._extract_pattern_features_fast(combination, draw_data)

            # 벡터화
            return self._vectorize_features_gpu(features)  # 동일한 벡터화 함수 사용

        except Exception as e:
            self.logger.debug(f"CPU 벡터화 실패: {e}")
            return np.array([0.5] * 19, dtype=np.float32)

    def _save_vectorized_results(
        self, feature_vectors: np.ndarray, sample_size: int
    ) -> str:
        """벡터화 결과 저장"""
        try:
            # 고정 파일 경로 (타임스탬프 제거)
            file_path = Path(self.cache_dir) / f"negative_vectors_{sample_size}.npy"
            latest_path = Path(self.cache_dir) / "negative_vectors_latest.npy"

            # 저장
            np.save(file_path, feature_vectors)
            np.save(latest_path, feature_vectors)

            # 메모리 사용량 추정
            mem_used = feature_vectors.nbytes / (1024 * 1024)  # MB

            self.logger.info(f"벡터화 결과 저장: {file_path}")
            self.logger.info(
                f"벡터 형태: {feature_vectors.shape}, 메모리: {mem_used:.1f}MB"
            )

            return str(file_path)

        except Exception as e:
            self.logger.error(f"벡터 저장 실패: {e}")
            raise

    def _adjust_batch_size_based_on_memory(self):
        """메모리 사용량 기반 배치 크기 조정"""
        if not self.use_gpu:
            return

        try:
            # GPU 메모리 사용률 확인
            memory_usage = self._get_gpu_memory_usage()
            memory_total = torch.cuda.get_device_properties(
                self.device
            ).total_memory / (
                1024**3
            )  # GB
            usage_ratio = memory_usage / (memory_total * 1024)  # 비율

            if usage_ratio > 0.8:  # 80% 이상 사용시
                self.batch_controller.reduce_batch_size()
                self.logger.info(
                    f"메모리 사용률 {usage_ratio*100:.1f}% - 배치 크기 감소"
                )
            elif usage_ratio < 0.5:  # 50% 미만 사용시
                self.batch_controller.increase_batch_size()
                self.logger.info(
                    f"메모리 사용률 {usage_ratio*100:.1f}% - 배치 크기 증가"
                )

        except Exception as e:
            self.logger.debug(f"배치 크기 조정 실패: {e}")

    def _get_gpu_utilization(self) -> float:
        """GPU 사용률 조회"""
        try:
            if self.cuda_optimizer and hasattr(
                self.cuda_optimizer, "get_gpu_utilization"
            ):
                return self.cuda_optimizer.get_gpu_utilization()
            return 0.0
        except Exception:
            return 0.0

    def _get_gpu_memory_usage(self) -> float:
        """GPU 메모리 사용량 조회 (MB)"""
        try:
            if self.use_gpu:
                return torch.cuda.memory_allocated(self.device) / (1024**2)
            return 0.0
        except Exception:
            return 0.0

    def _estimate_feature_vector_size(self) -> int:
        """
        특성 벡터 크기 추정

        Returns:
            예상 벡터 크기
        """
        # 기본값 (패턴 벡터라이저의 일반적인 출력 크기)
        default_size = 100

        # 기존 feature_vector_full.npy 파일이 있으면 그 크기를 참조
        vector_file = Path(self.cache_dir) / "feature_vector_full.npy"
        if vector_file.exists():
            try:
                sample_vector = np.load(vector_file)
                if len(sample_vector.shape) > 0:
                    return sample_vector.shape[0]
            except Exception as e:
                self.logger.warning(f"벡터 파일 로딩 실패: {e}")

        return default_size

    def _check_memory_usage(self, memory_limit_bytes: Union[int, float]) -> float:
        """
        메모리 사용량 확인 및 조치

        Args:
            memory_limit_bytes: 메모리 한계 (바이트)

        Returns:
            현재 메모리 사용량 (바이트)
        """
        # 현재 메모리 사용량 확인
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage = memory_info.rss  # 현재 실제 메모리 사용량

            # 메모리 한계에 근접하면 로깅
            if memory_usage > memory_limit_bytes * 0.8:
                self.logger.warning(
                    f"메모리 사용량 경고: {memory_usage / (1024**2):.1f}MB / "
                    f"{memory_limit_bytes / (1024**2):.1f}MB ({memory_usage / memory_limit_bytes * 100:.1f}%)"
                )

                # 메모리 한계를 초과하면 GC 강제 실행
                if memory_usage > memory_limit_bytes:
                    self.logger.warning("메모리 한계 초과, 가비지 컬렉션 실행")
                    gc.collect()

            return memory_usage

        except ImportError:
            # psutil이 없으면 로깅만
            self.logger.info(f"메모리 사용량 모니터링을 위해 psutil 패키지 설치 권장")
            return 0.0
        except Exception as e:
            self.logger.warning(f"메모리 사용량 확인 중 오류: {str(e)}")
            return 0.0

    def _generate_performance_report(
        self,
        start_time: float,
        end_time: float,
        sample_count: int,
        memory_used_mb: float,
        vector_path: str,
        warnings: Optional[List[str]] = None,
    ) -> str:
        """
        성능 보고서 생성 및 저장

        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            sample_count: 샘플 수
            memory_used_mb: 메모리 사용량 (MB)
            vector_path: 벡터 파일 경로
            warnings: 경고 메시지 목록

        Returns:
            성능 보고서 파일 경로
        """
        # 성능 보고서 데이터
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "execution_time": end_time - start_time,
            "sample_count": sample_count,
            "memory_used_mb": memory_used_mb,
            "vector_path": vector_path,
            "warnings": warnings or [],
            "config": {
                "batch_size": self.batch_controller.get_batch_size(),
                "cache_dir": self.cache_dir,
            },
        }

        # 성능 보고서 저장
        try:
            # logs_dir 설정이 없으면 기본값 사용
            logs_dir = self.config.get("paths", {}).get("logs_dir", "logs")
            Path(logs_dir).mkdir(exist_ok=True, parents=True)

            report_filename = (
                f"negative_sample_performance_{sample_count}_{int(start_time)}.json"
            )
            report_path = Path(logs_dir) / report_filename

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            return str(report_path)

        except Exception as e:
            self.logger.error(f"성능 보고서 저장 실패: {e}")
            return ""


def generate_negative_samples(
    draw_data: List[LotteryNumber], sample_size: int = 100000
) -> Dict[str, Any]:
    """
    비당첨 조합 샘플 생성 (모듈 레벨 함수)

    Args:
        draw_data: 당첨 번호 데이터
        sample_size: 생성할 샘플 크기

    Returns:
        생성 결과 정보
    """
    generator = NegativeSampleGenerator()
    return generator.generate_samples(draw_data, sample_size)
