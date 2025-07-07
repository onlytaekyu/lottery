"""
GPU 가속 3자리 조합 처리 모듈

3자리 조합 고속 처리를 위한 GPU 가속 유틸리티
220개 조합을 0.1초 이내에 처리하는 것을 목표로 함
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
import threading

from .unified_logging import get_logger
from .compute_strategy import ComputeExecutor, TaskType
from .memory_manager import MemoryManager
from .cache_manager import get_cache_manager

logger = get_logger(__name__)


class GPUThreeDigitAccelerator:
    """GPU 가속 3자리 조합 처리기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        GPU 가속기 초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available() and self.config.get("use_gpu", True)

        # 메모리 관리
        self.memory_manager = MemoryManager()
        self.cache_manager = get_cache_manager()

        # 3자리 조합 미리 계산
        self.three_digit_combinations = self._generate_combinations()
        self.combination_tensor = self._create_combination_tensor()

        # 배치 처리 설정
        self.batch_size = self.config.get("batch_size", 512)
        self.max_combinations = self.config.get("max_combinations", 220)

        # 성능 통계
        self.performance_stats = {
            "total_processed": 0,
            "avg_processing_time": 0.0,
            "gpu_utilization": 0.0,
        }

        self.logger.info(
            f"✅ GPU 3자리 가속기 초기화: 장치={self.device}, GPU 사용={self.use_gpu}"
        )

    def _generate_combinations(self) -> List[Tuple[int, int, int]]:
        """3자리 조합 생성"""
        try:
            # C(45, 3) = 14190개 조합 생성
            all_combinations = list(combinations(range(1, 46), 3))

            # 상위 220개만 선택 (빈도 기반 또는 기타 기준)
            # 여기서는 단순히 처음 220개를 선택
            selected_combinations = all_combinations[: self.max_combinations]

            self.logger.info(f"3자리 조합 생성 완료: {len(selected_combinations)}개")
            return selected_combinations

        except Exception as e:
            self.logger.error(f"3자리 조합 생성 중 오류: {e}")
            return []

    def _create_combination_tensor(self) -> torch.Tensor:
        """조합을 GPU 텐서로 변환"""
        try:
            if not self.three_digit_combinations:
                return torch.empty(0, 3, device=self.device)

            # 조합을 numpy 배열로 변환
            combo_array = np.array(self.three_digit_combinations, dtype=np.int32)

            # GPU 텐서로 변환
            combo_tensor = torch.from_numpy(combo_array).to(self.device)

            self.logger.info(
                f"조합 텐서 생성 완료: {combo_tensor.shape}, 장치={combo_tensor.device}"
            )
            return combo_tensor

        except Exception as e:
            self.logger.error(f"조합 텐서 생성 중 오류: {e}")
            return torch.empty(0, 3, device=self.device)

    def process_combinations_batch(
        self, feature_vectors: np.ndarray, scoring_function: str = "frequency_based"
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        배치 단위로 3자리 조합 처리

        Args:
            feature_vectors: 특성 벡터 배열
            scoring_function: 점수 계산 함수 ("frequency_based", "pattern_based", "ml_based")

        Returns:
            List[Tuple[Tuple[int, int, int], float]]: (조합, 점수) 리스트
        """
        try:
            self.logger.info(
                f"GPU 배치 처리 시작: {len(self.three_digit_combinations)}개 조합"
            )
            start_time = time.time()

            # 특성 벡터를 GPU 텐서로 변환
            if isinstance(feature_vectors, np.ndarray):
                feature_tensor = (
                    torch.from_numpy(feature_vectors).float().to(self.device)
                )
            else:
                feature_tensor = feature_vectors.to(self.device)

            # 점수 계산 함수 선택
            if scoring_function == "frequency_based":
                scores = self._calculate_frequency_scores_gpu(feature_tensor)
            elif scoring_function == "pattern_based":
                scores = self._calculate_pattern_scores_gpu(feature_tensor)
            elif scoring_function == "ml_based":
                scores = self._calculate_ml_scores_gpu(feature_tensor)
            else:
                scores = self._calculate_combined_scores_gpu(feature_tensor)

            # 결과 정리
            results = []
            scores_cpu = scores.cpu().numpy()

            for i, combo in enumerate(self.three_digit_combinations):
                if i < len(scores_cpu):
                    results.append((combo, float(scores_cpu[i])))

            # 점수 기준 정렬
            results.sort(key=lambda x: x[1], reverse=True)

            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(
                processing_time, len(self.three_digit_combinations)
            )

            self.logger.info(
                f"GPU 배치 처리 완료: {len(results)}개 결과 ({processing_time:.4f}초)"
            )
            return results

        except Exception as e:
            self.logger.error(f"GPU 배치 처리 중 오류: {e}")
            return []

    def _calculate_frequency_scores_gpu(
        self, feature_tensor: torch.Tensor
    ) -> torch.Tensor:
        """GPU 기반 빈도 점수 계산"""
        try:
            # 조합별 빈도 특성 추출
            combo_features = self._extract_combo_features_gpu(self.combination_tensor)

            # 특성 벡터와의 유사도 계산
            similarity_scores = torch.cosine_similarity(
                combo_features.unsqueeze(1), feature_tensor.unsqueeze(0), dim=2
            )

            # 평균 유사도 계산
            freq_scores = torch.mean(similarity_scores, dim=1)

            # 정규화
            freq_scores = F.softmax(freq_scores, dim=0)

            return freq_scores

        except Exception as e:
            self.logger.error(f"빈도 점수 계산 중 오류: {e}")
            return torch.zeros(len(self.three_digit_combinations), device=self.device)

    def _calculate_pattern_scores_gpu(
        self, feature_tensor: torch.Tensor
    ) -> torch.Tensor:
        """GPU 기반 패턴 점수 계산"""
        try:
            # 조합별 패턴 특성 계산
            pattern_features = self._calculate_pattern_features_gpu(
                self.combination_tensor
            )

            # 균형 점수 계산
            balance_scores = self._calculate_balance_scores_gpu(pattern_features)

            # 분산 점수 계산
            variance_scores = self._calculate_variance_scores_gpu(
                self.combination_tensor
            )

            # 통합 패턴 점수
            pattern_scores = 0.6 * balance_scores + 0.4 * variance_scores

            # 정규화
            pattern_scores = F.softmax(pattern_scores, dim=0)

            return pattern_scores

        except Exception as e:
            self.logger.error(f"패턴 점수 계산 중 오류: {e}")
            return torch.zeros(len(self.three_digit_combinations), device=self.device)

    def _calculate_ml_scores_gpu(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """GPU 기반 ML 점수 계산"""
        try:
            # 간단한 신경망 기반 점수 계산
            # 실제 구현에서는 훈련된 모델을 사용

            # 조합 특성 추출
            combo_features = self._extract_combo_features_gpu(self.combination_tensor)

            # 단순 선형 변환 (실제로는 훈련된 가중치 사용)
            weights = torch.randn(combo_features.shape[1], 1, device=self.device) * 0.1
            ml_scores = torch.matmul(combo_features, weights).squeeze()

            # 시그모이드 활성화
            ml_scores = torch.sigmoid(ml_scores)

            return ml_scores

        except Exception as e:
            self.logger.error(f"ML 점수 계산 중 오류: {e}")
            return torch.zeros(len(self.three_digit_combinations), device=self.device)

    def _calculate_combined_scores_gpu(
        self, feature_tensor: torch.Tensor
    ) -> torch.Tensor:
        """GPU 기반 통합 점수 계산"""
        try:
            # 각 점수 계산
            freq_scores = self._calculate_frequency_scores_gpu(feature_tensor)
            pattern_scores = self._calculate_pattern_scores_gpu(feature_tensor)
            ml_scores = self._calculate_ml_scores_gpu(feature_tensor)

            # 가중 평균
            combined_scores = (
                0.4 * freq_scores + 0.35 * pattern_scores + 0.25 * ml_scores
            )

            return combined_scores

        except Exception as e:
            self.logger.error(f"통합 점수 계산 중 오류: {e}")
            return torch.zeros(len(self.three_digit_combinations), device=self.device)

    def _extract_combo_features_gpu(self, combo_tensor: torch.Tensor) -> torch.Tensor:
        """GPU 기반 조합 특성 추출"""
        try:
            # 기본 통계 특성
            combo_sum = torch.sum(combo_tensor, dim=1).float()
            combo_mean = torch.mean(combo_tensor.float(), dim=1)
            combo_std = torch.std(combo_tensor.float(), dim=1)
            combo_range = (
                torch.max(combo_tensor, dim=1)[0] - torch.min(combo_tensor, dim=1)[0]
            )

            # 홀수 개수
            odd_count = torch.sum(combo_tensor % 2, dim=1).float()

            # 특성 결합
            features = torch.stack(
                [combo_sum, combo_mean, combo_std, combo_range.float(), odd_count],
                dim=1,
            )

            # 정규화
            features = F.normalize(features, p=2, dim=1)

            return features

        except Exception as e:
            self.logger.error(f"조합 특성 추출 중 오류: {e}")
            return torch.zeros(combo_tensor.shape[0], 5, device=self.device)

    def _calculate_pattern_features_gpu(
        self, combo_tensor: torch.Tensor
    ) -> torch.Tensor:
        """GPU 기반 패턴 특성 계산"""
        try:
            # 간격 계산
            sorted_combos, _ = torch.sort(combo_tensor, dim=1)
            gaps = sorted_combos[:, 1:] - sorted_combos[:, :-1]

            # 간격 통계
            gap_mean = torch.mean(gaps.float(), dim=1)
            gap_std = torch.std(gaps.float(), dim=1)

            # 구간 분포 (1-15, 16-30, 31-45)
            segment1 = torch.sum(
                (combo_tensor >= 1) & (combo_tensor <= 15), dim=1
            ).float()
            segment2 = torch.sum(
                (combo_tensor >= 16) & (combo_tensor <= 30), dim=1
            ).float()
            segment3 = torch.sum(
                (combo_tensor >= 31) & (combo_tensor <= 45), dim=1
            ).float()

            # 패턴 특성 결합
            pattern_features = torch.stack(
                [gap_mean, gap_std, segment1, segment2, segment3], dim=1
            )

            return pattern_features

        except Exception as e:
            self.logger.error(f"패턴 특성 계산 중 오류: {e}")
            return torch.zeros(combo_tensor.shape[0], 5, device=self.device)

    def _calculate_balance_scores_gpu(
        self, pattern_features: torch.Tensor
    ) -> torch.Tensor:
        """GPU 기반 균형 점수 계산"""
        try:
            # 구간 균형 점수 (2:1:0 또는 1:1:1이 이상적)
            segment_balance = pattern_features[:, 2:]  # segment1, segment2, segment3

            # 이상적 분포와의 거리 계산
            ideal_dist = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            balance_distances = torch.norm(segment_balance - ideal_dist, dim=1)

            # 거리가 작을수록 높은 점수
            balance_scores = 1.0 / (1.0 + balance_distances)

            return balance_scores

        except Exception as e:
            self.logger.error(f"균형 점수 계산 중 오류: {e}")
            return torch.zeros(pattern_features.shape[0], device=self.device)

    def _calculate_variance_scores_gpu(
        self, combo_tensor: torch.Tensor
    ) -> torch.Tensor:
        """GPU 기반 분산 점수 계산"""
        try:
            # 조합별 분산 계산
            variances = torch.var(combo_tensor.float(), dim=1)

            # 적절한 분산 범위에서 높은 점수
            # 너무 낮거나 높으면 점수 감소
            optimal_variance = 200.0  # 경험적 최적값
            variance_scores = 1.0 / (
                1.0 + torch.abs(variances - optimal_variance) / 100.0
            )

            return variance_scores

        except Exception as e:
            self.logger.error(f"분산 점수 계산 중 오류: {e}")
            return torch.zeros(combo_tensor.shape[0], device=self.device)

    def _update_performance_stats(self, processing_time: float, num_combinations: int):
        """성능 통계 업데이트"""
        try:
            self.performance_stats["total_processed"] += num_combinations

            # 이동 평균으로 평균 처리 시간 업데이트
            alpha = 0.1  # 이동 평균 계수
            if self.performance_stats["avg_processing_time"] == 0:
                self.performance_stats["avg_processing_time"] = processing_time
            else:
                self.performance_stats["avg_processing_time"] = (
                    alpha * processing_time
                    + (1 - alpha) * self.performance_stats["avg_processing_time"]
                )

            # GPU 사용률 계산 (간단한 추정)
            if self.use_gpu:
                self.performance_stats["gpu_utilization"] = min(
                    processing_time / 0.1, 1.0  # 목표 0.1초 대비
                )

        except Exception as e:
            self.logger.error(f"성능 통계 업데이트 중 오류: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            **self.performance_stats,
            "device": str(self.device),
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size,
            "max_combinations": self.max_combinations,
            "combinations_per_second": (
                self.performance_stats["total_processed"]
                / max(self.performance_stats["avg_processing_time"], 0.001)
            ),
        }

    def benchmark_performance(self, num_iterations: int = 10) -> Dict[str, Any]:
        """성능 벤치마크 실행"""
        try:
            self.logger.info(f"성능 벤치마크 시작: {num_iterations}회 반복")

            # 더미 특성 벡터 생성
            dummy_features = np.random.randn(50, 100).astype(np.float32)

            times = []
            for i in range(num_iterations):
                start_time = time.time()
                results = self.process_combinations_batch(dummy_features)
                elapsed = time.time() - start_time
                times.append(elapsed)

                if i % 5 == 0:
                    self.logger.info(
                        f"벤치마크 진행: {i+1}/{num_iterations}, 시간: {elapsed:.4f}초"
                    )

            # 통계 계산
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)

            benchmark_results = {
                "iterations": num_iterations,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "std_time": std_time,
                "combinations_per_second": self.max_combinations / avg_time,
                "target_achieved": avg_time <= 0.1,  # 목표 0.1초 달성 여부
                "device": str(self.device),
                "use_gpu": self.use_gpu,
            }

            self.logger.info(
                f"벤치마크 완료: 평균 {avg_time:.4f}초, 목표 달성: {benchmark_results['target_achieved']}"
            )
            return benchmark_results

        except Exception as e:
            self.logger.error(f"벤치마크 중 오류: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """캐시 정리"""
        try:
            self.cache_manager.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("GPU 가속기 캐시 정리 완료")
        except Exception as e:
            self.logger.error(f"캐시 정리 중 오류: {e}")

    def shutdown(self):
        """리소스 정리"""
        try:
            self.clear_cache()
            self.logger.info("GPU 3자리 가속기 종료 완료")
        except Exception as e:
            self.logger.error(f"가속기 종료 중 오류: {e}")


# 편의 함수
def create_gpu_accelerator(
    config: Optional[Dict[str, Any]] = None,
) -> GPUThreeDigitAccelerator:
    """GPU 가속기 생성 편의 함수"""
    return GPUThreeDigitAccelerator(config)


def benchmark_gpu_performance(
    config: Optional[Dict[str, Any]] = None, iterations: int = 10
) -> Dict[str, Any]:
    """GPU 성능 벤치마크 편의 함수"""
    accelerator = create_gpu_accelerator(config)

    try:
        return accelerator.benchmark_performance(iterations)
    finally:
        accelerator.shutdown()
