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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional, Union, cast

# logging 제거 - unified_logging 사용
from datetime import datetime
import json
import platform
import psutil

from ..utils.unified_logging import get_logger
from ..utils.unified_performance import performance_monitor
from ..utils.memory_manager import MemoryManager
from ..utils.unified_config import ConfigProxy
from ..utils.cuda_optimizers import DynamicBatchSizeController
from ..shared.types import LotteryNumber
from .pattern_analyzer import PatternAnalyzer, PatternFeatures
from .enhanced_pattern_vectorizer import EnhancedPatternVectorizer
from .base_analyzer import BaseAnalyzer


# 로거 설정
logger = get_logger(__name__)

# 글로벌 분석기 인스턴스 - 멀티프로세싱용
_global_pattern_analyzer = None


def init_worker(config_dict=None):
    """
    워커 프로세스 초기화 함수

    Args:
        config_dict: 설정 사전
    """
    global _global_pattern_analyzer
    _global_pattern_analyzer = PatternAnalyzer(config_dict)
    logger.debug("워커 프로세스 분석기 초기화 완료")


def vectorize_combination(params):
    """
    멀티프로세싱을 위한 글로벌 벡터화 함수

    Args:
        params: (combination, draw_data, expected_features) 튜플

    Returns:
        벡터화된 특성
    """
    global _global_pattern_analyzer

    combination, draw_data, expected_features = params

    try:
        # 글로벌 패턴 분석기가 없으면 새로 생성
        if _global_pattern_analyzer is None:
            _global_pattern_analyzer = PatternAnalyzer()

        # 특성 추출 및 벡터화
        features = _global_pattern_analyzer.extract_pattern_features(
            combination, draw_data
        )
        vector = _global_pattern_analyzer.vectorize_pattern_features(features)

        # 벡터 크기 조정 (필요한 경우)
        if len(vector) != expected_features:
            adjusted_vector = np.zeros(expected_features, dtype=np.float32)
            # 더 작은 크기까지만 복사
            copy_size = min(len(vector), expected_features)
            adjusted_vector[:copy_size] = vector[:copy_size]
            vector = adjusted_vector

        return vector
    except Exception as e:
        logger.error(f"벡터화 오류 (조합: {combination}): {str(e)}")
        # 오류 시 기본 벡터 할당
        return np.zeros(expected_features, dtype=np.float32)


def process_batch(params):
    """
    배치 처리 함수

    Args:
        params: (batch, draw_data, start_idx, vector_size) 튜플

    Returns:
        (시작 인덱스, 결과 리스트) 튜플
    """
    batch, draw_data, start_idx, vector_size = params

    # 각 조합별 벡터화를 위한 파라미터 준비
    vectorize_params = [(combo, draw_data, vector_size) for combo in batch]

    # 배치별 개별 벡터화 수행
    results = list(map(vectorize_combination, vectorize_params))

    # 결과 반환 - (시작 인덱스, 결과 리스트) 형태
    return (start_idx, results)


def generate_batch_samples(
    existing_combinations: Set[Tuple[int, ...]], batch_size: int
) -> List[List[int]]:
    """
    비당첨 조합 배치 생성 (독립 함수)

    Args:
        existing_combinations: 이미 존재하는 당첨 조합
        batch_size: 생성할 배치 크기

    Returns:
        비당첨 번호 조합 목록
    """
    batch_samples = []

    # NumPy 기반 벡터화된 생성 로직
    BATCH_MULTIPLIER = 2  # 필터링 손실을 고려하여 더 많이 생성
    all_numbers = np.arange(1, 46)

    # 필요한 양보다 약간 더 많이 생성
    oversample_size = min(batch_size * BATCH_MULTIPLIER, 10000)

    # 고유 조합만 필터링하기 위한 집합
    unique_combinations = set()

    while len(batch_samples) < batch_size:
        # 랜덤 인덱스 배열 생성 (각 행은 6개 고유 인덱스)
        random_indices = np.zeros((oversample_size, 6), dtype=np.int32)

        for i in range(6):
            if i == 0:
                # 첫 번째 숫자 선택
                random_indices[:, i] = np.random.randint(0, 45, oversample_size)
            else:
                # 이전 선택 숫자와 겹치지 않도록 조정
                for j in range(oversample_size):
                    # 이미 선택된 인덱스는 피하기
                    mask = np.ones(45, dtype=bool)
                    mask[random_indices[j, :i]] = False
                    valid_indices = np.arange(45)[mask]
                    if len(valid_indices) > 0:
                        idx = np.random.choice(valid_indices)
                        random_indices[j, i] = idx

        # 인덱스 → 실제 번호 변환
        random_combinations = all_numbers[random_indices]

        # 각 행을 정렬
        random_combinations = np.sort(random_combinations, axis=1)

        # 배치 처리
        for combo in random_combinations:
            combo_tuple = tuple(combo)

            # 당첨 번호와 중복 아닌지, 이미 선택되지 않았는지 확인
            if (
                combo_tuple not in existing_combinations
                and combo_tuple not in unique_combinations
            ):
                batch_samples.append(combo.tolist())
                unique_combinations.add(combo_tuple)

                # 충분한 샘플을 얻었으면 중단
                if len(batch_samples) >= batch_size:
                    break

    return batch_samples


class NegativeSampleGenerator(BaseAnalyzer[Dict[str, Any]]):
    """비당첨 샘플 생성 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        초기화

        Args:
            config: 설정
        """
        super().__init__(config or {}, "negative_sample")
        self.config = config or {}
        self.pattern_analyzer = PatternAnalyzer(config)
        self.pattern_vectorizer = EnhancedPatternVectorizer(config)

        # 캐시 디렉토리 설정
        try:
            self.cache_dir = self.config.get("paths", {}).get("cache_dir", "data/cache")
        except KeyError:
            raise KeyError("설정에서 'paths.cache_dir' 키를 찾을 수 없습니다.")
        Path(self.cache_dir).mkdir(exist_ok=True, parents=True)

        # 배치 크기 컨트롤러
        negative_sampler_config = self.config.get("negative_sampler", {})
        self.batch_controller = DynamicBatchSizeController(
            initial_batch_size=negative_sampler_config.get("batch_size", 1000),
            min_batch_size=100,
            max_batch_size=10000,
            growth_rate=1.2,
            reduction_rate=0.5,
        )

        # 진행 상황 추적 변수
        self.progress = 0
        self.total = 0

        # 메모리 추적기
        self.memory_tracker = MemoryManager()

        # 진행 상황 추적 잠금
        self._progress_lock = threading.Lock()

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """BaseAnalyzer 인터페이스 구현"""
        sample_size = kwargs.get("sample_size", 100000)
        return self.generate_samples(historical_data, sample_size)

    # @profile("generate_negative_samples")
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
        - random(30%), pattern-based(40%), adversarial(30%)
        """
        n_random = int(sample_size * 0.3)
        n_pattern = int(sample_size * 0.4)
        n_adv = sample_size - n_random - n_pattern
        negative_samples = []
        # 1. Random Sampling
        negative_samples.extend(
            self._random_negative_samples(existing_combinations, n_random)
        )
        # 2. Pattern-based Sampling
        negative_samples.extend(
            self._pattern_based_negative_samples(existing_combinations, n_pattern)
        )
        # 3. Adversarial Sampling
        negative_samples.extend(
            self._adversarial_negative_samples(existing_combinations, n_adv)
        )
        return negative_samples

    def _random_negative_samples(
        self, existing_combinations: Set[Tuple[int, ...]], n: int
    ) -> List[List[int]]:
        """단순 랜덤 비당첨 조합 생성"""
        return generate_batch_samples(existing_combinations, n)

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

    def _adversarial_negative_samples(
        self, existing_combinations: Set[Tuple[int, ...]], n: int
    ) -> List[List[int]]:
        """패턴 분석 기반, 당첨 확률 낮은(패턴상 불리) 조합 생성"""
        samples = []
        all_numbers = np.arange(1, 46)
        attempts = 0
        while len(samples) < n and attempts < n * 10:
            combo = random.sample(list(all_numbers), 6)
            features = self.pattern_analyzer.extract_pattern_features(combo, None)
            # 예: 패턴상 당첨 확률이 매우 낮은 조합(예: 동일 끝수 4개 이상, 고분산 등)
            if features.get("same_end_digit", 0) >= 4 or features.get("spread", 0) > 35:
                t = tuple(sorted(combo))
                if t not in existing_combinations:
                    samples.append(list(t))
            attempts += 1
        return samples[:n]

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
        비당첨 조합 벡터화

        Args:
            negative_samples: 비당첨 번호 조합 목록
            draw_data: 당첨 번호 데이터
            sample_size: 샘플 크기

        Returns:
            저장 파일 경로
        """
        self.logger.info(f"비당첨 조합 벡터화 시작: {len(negative_samples):,}개")
        start_time = time.time()

        # 병렬 처리 설정
        negative_sampler_config = self.config.get("negative_sampler", {})
        max_processes = min(
            negative_sampler_config.get("vectorize_workers", 4),
            multiprocessing.cpu_count(),
        )
        batch_size = negative_sampler_config.get("vectorize_batch", 1000)

        self.logger.info(
            f"프로세스 풀 사용: {max_processes}개 프로세스, 배치 크기: {batch_size}"
        )

        # 특성 벡터 크기 예측
        expected_num_features = self._estimate_feature_vector_size()
        self.logger.info(f"예상 특성 벡터 크기: {expected_num_features}")

        # 첫 조합으로 실제 벡터 크기 확인
        actual_num_features = expected_num_features
        try:
            # 벡터 크기 테스트
            test_features = self.pattern_analyzer.extract_pattern_features(
                negative_samples[0], draw_data
            )
            test_vector = self.pattern_analyzer.vectorize_pattern_features(
                test_features
            )
            actual_num_features = len(test_vector)

            if actual_num_features != expected_num_features:
                self.logger.warning(
                    f"벡터 크기 불일치: 예상={expected_num_features}, 실제={actual_num_features}. 실제 크기로 조정"
                )
        except Exception as e:
            self.logger.warning(f"벡터 크기 테스트 실패: {e}. 예상 크기 사용")

        # 결과 저장 배열 - 실제 벡터 크기 사용
        feature_vectors = np.zeros(
            (len(negative_samples), actual_num_features), dtype=np.float32
        )

        # 진행 상황 초기화
        self.progress = 0
        self.total = len(negative_samples)

        # 배치 단위로 처리할 파라미터 준비
        batch_params = []
        for i in range(0, len(negative_samples), batch_size):
            end_idx = min(i + batch_size, len(negative_samples))
            batch = negative_samples[i:end_idx]
            batch_params.append((batch, draw_data, i, actual_num_features))

        self.logger.info(f"전체 {len(batch_params)}개 배치로 처리")

        try:
            # GIL 우회를 위한 ProcessPoolExecutor 사용
            with ProcessPoolExecutor(
                max_workers=max_processes,
                initializer=init_worker,
                initargs=(self.config,),
            ) as executor:
                # 배치 단위로 병렬 처리
                futures = []
                for params in batch_params:
                    futures.append(executor.submit(process_batch, params))

                # 결과 수집 및 진행상황 업데이트
                for future in futures:
                    start_idx, results = future.result()

                    # 결과를 feature_vectors에 저장
                    for i, vector in enumerate(results):
                        idx = start_idx + i
                        if idx < len(feature_vectors):
                            feature_vectors[idx] = vector

                    # 진행 상황 업데이트
                    with self._progress_lock:
                        self.progress += len(results)
                        progress_pct = (self.progress / self.total) * 100
                        elapsed = time.time() - start_time
                        speed = self.progress / elapsed if elapsed > 0 else 0

                        # 메모리 사용량 추정
                        mem_usage = (
                            self.progress * actual_num_features * 4 / (1024 * 1024)
                        )  # MB

                        self.logger.info(
                            f"벡터화 진행: {self.progress:,}/{self.total:,} ({progress_pct:.1f}%) - "
                            f"속도: {speed:.1f}개/초, 메모리: {mem_usage:.1f}MB"
                        )

                        # 메모리 확보를 위한 GC 강제 호출
                        if self.progress % (batch_size * 5) == 0:
                            gc.collect()

        except Exception as e:
            self.logger.error(f"벡터화 과정에서 오류 발생: {str(e)}")

        # 고정 파일 경로 (타임스탬프 제거)
        file_path = Path(self.cache_dir) / f"negative_vectors_{sample_size}.npy"

        # 최신 버전 링크 (덮어쓰기)
        latest_path = Path(self.cache_dir) / "negative_vectors_latest.npy"

        # 저장
        np.save(file_path, feature_vectors)
        np.save(latest_path, feature_vectors)

        # 메모리 사용량 추정
        mem_used = len(negative_samples) * actual_num_features * 4 / (1024 * 1024)  # MB

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"비당첨 조합 벡터화 완료: {len(negative_samples):,}개 ({elapsed_time:.2f}초), 메모리: {mem_used:.1f}MB"
        )
        self.logger.info(f"벡터화 결과 저장: {file_path}")
        self.logger.info(f"최신 버전: {latest_path}")

        return str(file_path)

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
