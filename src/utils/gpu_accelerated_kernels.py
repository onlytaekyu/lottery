"""
GPU 가속 CUDA 커널 - 패턴 분석 및 벡터화 가속

이 모듈은 로또 번호 패턴 분석과 벡터화를 위한 고성능 CUDA 커널을 제공합니다.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any
from ..utils.unified_logging import get_logger
from ..utils.cuda_optimizers import get_cuda_optimizer, CudaConfig
from ..utils.unified_memory_manager import get_unified_memory_manager

logger = get_logger(__name__)


class GPUPatternKernels:
    """GPU 패턴 분석 커널 모음"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.logger = get_logger(__name__)

        # CUDA 최적화기 및 메모리 풀 초기화
        self.cuda_optimizer = get_cuda_optimizer(config=CudaConfig())
        self.memory_manager = get_unified_memory_manager()

        # 커널 캐시
        self.kernel_cache = {}

        self.logger.info(f"✅ GPU 패턴 커널 초기화 완료 (디바이스: {device_id})")

    def frequency_analysis_kernel(
        self, lottery_numbers: torch.Tensor, max_number: int = 45
    ) -> torch.Tensor:
        """
        빈도 분석 GPU 커널

        Args:
            lottery_numbers: [batch_size, 6] 형태의 로또 번호 텐서
            max_number: 최대 번호 (기본값: 45)

        Returns:
            [max_number] 형태의 빈도 텐서
        """
        try:
            with self.cuda_optimizer.device_context():
                # 메모리 풀에서 할당
                with self.memory_manager.allocate(
                    max_number * 4, torch.float32
                ) as freq_tensor:

                    if freq_tensor is None:
                        # 풀 할당 실패 시 직접 할당
                        freq_tensor = torch.zeros(max_number, device=self.device)
                    else:
                        freq_tensor = freq_tensor[:max_number].zero_()

                    # GPU에서 빈도 계산
                    lottery_numbers = lottery_numbers.to(self.device)

                    # 각 번호의 빈도 계산 (벡터화)
                    for i in range(max_number):
                        freq_tensor[i] = (lottery_numbers == (i + 1)).float().sum()

                    return freq_tensor.cpu()

        except Exception as e:
            self.logger.error(f"빈도 분석 커널 실패: {e}")
            # CPU 폴백
            return self._frequency_analysis_cpu(lottery_numbers, max_number)

    def gap_analysis_kernel(
        self, lottery_numbers: torch.Tensor, max_number: int = 45
    ) -> torch.Tensor:
        """
        간격 분석 GPU 커널

        Args:
            lottery_numbers: [batch_size, 6] 형태의 로또 번호 텐서
            max_number: 최대 번호

        Returns:
            [max_number] 형태의 평균 간격 텐서
        """
        try:
            with self.cuda_optimizer.device_context():
                batch_size = lottery_numbers.size(0)
                lottery_numbers = lottery_numbers.to(self.device)

                # 각 번호별 마지막 등장 위치 추적
                last_seen = torch.full((max_number,), -1, device=self.device)
                gap_sums = torch.zeros(max_number, device=self.device)
                gap_counts = torch.zeros(max_number, device=self.device)

                # 배치별 처리
                for batch_idx in range(batch_size):
                    current_numbers = lottery_numbers[batch_idx]

                    # 현재 배치에서 나온 번호들 처리
                    for num in current_numbers:
                        num_idx = num.long() - 1  # 0-based 인덱스
                        if 0 <= num_idx < max_number:
                            if last_seen[num_idx] >= 0:
                                gap = batch_idx - last_seen[num_idx]
                                gap_sums[num_idx] += gap
                                gap_counts[num_idx] += 1
                            last_seen[num_idx] = batch_idx

                # 평균 간격 계산
                avg_gaps = torch.where(
                    gap_counts > 0,
                    gap_sums / gap_counts,
                    torch.tensor(float("inf"), device=self.device),
                )

                return avg_gaps.cpu()

        except Exception as e:
            self.logger.error(f"간격 분석 커널 실패: {e}")
            return self._gap_analysis_cpu(lottery_numbers, max_number)

    def pattern_similarity_kernel(
        self,
        target_pattern: torch.Tensor,
        historical_patterns: torch.Tensor,
        similarity_threshold: float = 0.7,
    ) -> torch.Tensor:
        """
        패턴 유사도 계산 GPU 커널

        Args:
            target_pattern: [6] 형태의 타겟 패턴
            historical_patterns: [batch_size, 6] 형태의 과거 패턴들
            similarity_threshold: 유사도 임계값

        Returns:
            [batch_size] 형태의 유사도 텐서
        """
        try:
            with self.cuda_optimizer.device_context():
                target_pattern = target_pattern.to(self.device)
                historical_patterns = historical_patterns.to(self.device)

                # 코사인 유사도 계산
                target_norm = F.normalize(target_pattern.float(), dim=0)
                historical_norm = F.normalize(historical_patterns.float(), dim=1)

                # 배치 연산으로 유사도 계산
                similarities = torch.mm(
                    historical_norm, target_norm.unsqueeze(1)
                ).squeeze(1)

                return similarities.cpu()

        except Exception as e:
            self.logger.error(f"패턴 유사도 커널 실패: {e}")
            return self._pattern_similarity_cpu(
                target_pattern, historical_patterns, similarity_threshold
            )

    def consecutive_analysis_kernel(
        self, lottery_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        연속 번호 분석 GPU 커널

        Args:
            lottery_numbers: [batch_size, 6] 형태의 로또 번호 텐서

        Returns:
            [batch_size] 형태의 연속 번호 개수 텐서
        """
        try:
            with self.cuda_optimizer.device_context():
                lottery_numbers = lottery_numbers.to(self.device)
                batch_size = lottery_numbers.size(0)

                # 각 행을 정렬
                sorted_numbers, _ = torch.sort(lottery_numbers, dim=1)

                # 연속 번호 개수 계산
                consecutive_counts = torch.zeros(batch_size, device=self.device)

                for i in range(batch_size):
                    row = sorted_numbers[i]
                    consecutive_count = 0

                    for j in range(5):  # 6개 번호 중 5개 간격 확인
                        if row[j + 1] - row[j] == 1:
                            consecutive_count += 1

                    consecutive_counts[i] = consecutive_count

                return consecutive_counts.cpu()

        except Exception as e:
            self.logger.error(f"연속 번호 분석 커널 실패: {e}")
            return self._consecutive_analysis_cpu(lottery_numbers)

    def segment_distribution_kernel(
        self,
        lottery_numbers: torch.Tensor,
        segment_size: int = 9,  # 1-9, 10-18, 19-27, 28-36, 37-45
    ) -> torch.Tensor:
        """
        구간 분포 분석 GPU 커널

        Args:
            lottery_numbers: [batch_size, 6] 형태의 로또 번호 텐서
            segment_size: 구간 크기

        Returns:
            [batch_size, 5] 형태의 구간별 개수 텐서
        """
        try:
            with self.cuda_optimizer.device_context():
                lottery_numbers = lottery_numbers.to(self.device)
                batch_size = lottery_numbers.size(0)
                num_segments = 5

                # 구간별 분포 계산
                segment_counts = torch.zeros(
                    (batch_size, num_segments), device=self.device
                )

                for i in range(batch_size):
                    numbers = lottery_numbers[i]

                    for num in numbers:
                        segment_idx = min(
                            (num.long() - 1) // segment_size, num_segments - 1
                        )
                        segment_counts[i, segment_idx] += 1

                return segment_counts.cpu()

        except Exception as e:
            self.logger.error(f"구간 분포 분석 커널 실패: {e}")
            return self._segment_distribution_cpu(lottery_numbers, segment_size)

    def batch_pattern_vectorization_kernel(
        self, lottery_numbers: torch.Tensor, feature_configs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        배치 패턴 벡터화 GPU 커널

        Args:
            lottery_numbers: [batch_size, 6] 형태의 로또 번호 텐서
            feature_configs: 특성 설정 딕셔너리

        Returns:
            [batch_size, feature_dim] 형태의 특성 벡터 텐서
        """
        try:
            with self.cuda_optimizer.device_context():
                lottery_numbers = lottery_numbers.to(self.device)
                lottery_numbers.size(0)

                # 다양한 특성 계산
                features = []

                # 1. 빈도 기반 특성
                if feature_configs.get("use_frequency", True):
                    freq_features = self._compute_frequency_features(lottery_numbers)
                    features.append(freq_features)

                # 2. 간격 기반 특성
                if feature_configs.get("use_gaps", True):
                    gap_features = self._compute_gap_features(lottery_numbers)
                    features.append(gap_features)

                # 3. 패턴 기반 특성
                if feature_configs.get("use_patterns", True):
                    pattern_features = self._compute_pattern_features(lottery_numbers)
                    features.append(pattern_features)

                # 4. 통계 기반 특성
                if feature_configs.get("use_stats", True):
                    stat_features = self._compute_statistical_features(lottery_numbers)
                    features.append(stat_features)

                # 특성 결합
                combined_features = torch.cat(features, dim=1)

                return combined_features.cpu()

        except Exception as e:
            self.logger.error(f"배치 벡터화 커널 실패: {e}")
            return self._batch_vectorization_cpu(lottery_numbers, feature_configs)

    def _compute_frequency_features(
        self, lottery_numbers: torch.Tensor
    ) -> torch.Tensor:
        """빈도 기반 특성 계산"""
        batch_size = lottery_numbers.size(0)

        # 각 배치에서 번호별 빈도 계산
        freq_features = torch.zeros((batch_size, 45), device=self.device)

        for i in range(batch_size):
            numbers = lottery_numbers[i]
            for num in numbers:
                if 1 <= num <= 45:
                    freq_features[i, num.long() - 1] += 1

        return freq_features

    def _compute_gap_features(self, lottery_numbers: torch.Tensor) -> torch.Tensor:
        """간격 기반 특성 계산"""
        batch_size = lottery_numbers.size(0)

        # 정렬된 번호들 간의 간격 계산
        sorted_numbers, _ = torch.sort(lottery_numbers, dim=1)

        # 간격 특성 (5개 간격)
        gap_features = torch.zeros((batch_size, 5), device=self.device)

        for i in range(batch_size):
            for j in range(5):
                gap_features[i, j] = sorted_numbers[i, j + 1] - sorted_numbers[i, j]

        return gap_features

    def _compute_pattern_features(self, lottery_numbers: torch.Tensor) -> torch.Tensor:
        """패턴 기반 특성 계산"""
        batch_size = lottery_numbers.size(0)

        # 패턴 특성들
        pattern_features = torch.zeros((batch_size, 10), device=self.device)

        for i in range(batch_size):
            numbers = lottery_numbers[i]
            sorted_nums, _ = torch.sort(numbers)

            # 특성 1: 합계
            pattern_features[i, 0] = numbers.sum()

            # 특성 2: 평균
            pattern_features[i, 1] = numbers.float().mean()

            # 특성 3: 표준편차
            pattern_features[i, 2] = numbers.float().std()

            # 특성 4: 범위
            pattern_features[i, 3] = sorted_nums[-1] - sorted_nums[0]

            # 특성 5: 홀수 개수
            pattern_features[i, 4] = (numbers % 2 == 1).float().sum()

            # 특성 6: 짝수 개수
            pattern_features[i, 5] = (numbers % 2 == 0).float().sum()

            # 특성 7-10: 구간별 개수 (1-10, 11-20, 21-30, 31-45)
            for segment in range(4):
                start = segment * 10 + 1
                end = (segment + 1) * 10 if segment < 3 else 45
                count = ((numbers >= start) & (numbers <= end)).float().sum()
                pattern_features[i, 6 + segment] = count

        return pattern_features

    def _compute_statistical_features(
        self, lottery_numbers: torch.Tensor
    ) -> torch.Tensor:
        """통계 기반 특성 계산"""
        batch_size = lottery_numbers.size(0)

        # 통계 특성들
        stat_features = torch.zeros((batch_size, 8), device=self.device)

        for i in range(batch_size):
            numbers = lottery_numbers[i].float()

            # 기본 통계
            stat_features[i, 0] = numbers.min()
            stat_features[i, 1] = numbers.max()
            stat_features[i, 2] = numbers.median()
            stat_features[i, 3] = numbers.mean()
            stat_features[i, 4] = numbers.std()
            stat_features[i, 5] = numbers.var()

            # 왜도와 첨도 (근사치)
            mean_val = numbers.mean()
            std_val = numbers.std()
            if std_val > 0:
                normalized = (numbers - mean_val) / std_val
                stat_features[i, 6] = (normalized**3).mean()  # 왜도
                stat_features[i, 7] = (normalized**4).mean()  # 첨도

        return stat_features

    # CPU 폴백 메서드들
    def _frequency_analysis_cpu(
        self, lottery_numbers: torch.Tensor, max_number: int
    ) -> torch.Tensor:
        """CPU 빈도 분석 폴백"""
        lottery_numbers = lottery_numbers.cpu()
        freq_tensor = torch.zeros(max_number)

        for i in range(max_number):
            freq_tensor[i] = (lottery_numbers == (i + 1)).float().sum()

        return freq_tensor

    def _gap_analysis_cpu(
        self, lottery_numbers: torch.Tensor, max_number: int
    ) -> torch.Tensor:
        """CPU 간격 분석 폴백"""
        lottery_numbers = lottery_numbers.cpu()
        batch_size = lottery_numbers.size(0)

        last_seen = [-1] * max_number
        gap_sums = [0] * max_number
        gap_counts = [0] * max_number

        for batch_idx in range(batch_size):
            current_numbers = lottery_numbers[batch_idx]

            for num in current_numbers:
                num_idx = int(num) - 1
                if 0 <= num_idx < max_number:
                    if last_seen[num_idx] >= 0:
                        gap = batch_idx - last_seen[num_idx]
                        gap_sums[num_idx] += gap
                        gap_counts[num_idx] += 1
                    last_seen[num_idx] = batch_idx

        avg_gaps = torch.tensor(
            [
                gap_sums[i] / gap_counts[i] if gap_counts[i] > 0 else float("inf")
                for i in range(max_number)
            ]
        )

        return avg_gaps

    def _pattern_similarity_cpu(
        self,
        target_pattern: torch.Tensor,
        historical_patterns: torch.Tensor,
        similarity_threshold: float,
    ) -> torch.Tensor:
        """CPU 패턴 유사도 폴백"""
        target_pattern = target_pattern.cpu().float()
        historical_patterns = historical_patterns.cpu().float()

        # 코사인 유사도 계산
        target_norm = F.normalize(target_pattern, dim=0)
        historical_norm = F.normalize(historical_patterns, dim=1)

        similarities = torch.mm(historical_norm, target_norm.unsqueeze(1)).squeeze(1)

        return similarities

    def _consecutive_analysis_cpu(self, lottery_numbers: torch.Tensor) -> torch.Tensor:
        """CPU 연속 번호 분석 폴백"""
        lottery_numbers = lottery_numbers.cpu()
        batch_size = lottery_numbers.size(0)

        consecutive_counts = torch.zeros(batch_size)

        for i in range(batch_size):
            sorted_numbers, _ = torch.sort(lottery_numbers[i])
            consecutive_count = 0

            for j in range(5):
                if sorted_numbers[j + 1] - sorted_numbers[j] == 1:
                    consecutive_count += 1

            consecutive_counts[i] = consecutive_count

        return consecutive_counts

    def _segment_distribution_cpu(
        self, lottery_numbers: torch.Tensor, segment_size: int
    ) -> torch.Tensor:
        """CPU 구간 분포 분석 폴백"""
        lottery_numbers = lottery_numbers.cpu()
        batch_size = lottery_numbers.size(0)
        num_segments = 5

        segment_counts = torch.zeros((batch_size, num_segments))

        for i in range(batch_size):
            numbers = lottery_numbers[i]

            for num in numbers:
                segment_idx = min((int(num) - 1) // segment_size, num_segments - 1)
                segment_counts[i, segment_idx] += 1

        return segment_counts

    def _batch_vectorization_cpu(
        self, lottery_numbers: torch.Tensor, feature_configs: Dict[str, Any]
    ) -> torch.Tensor:
        """CPU 배치 벡터화 폴백"""
        lottery_numbers = lottery_numbers.cpu()
        batch_size = lottery_numbers.size(0)

        # 간단한 특성 벡터 생성
        features = []

        # 기본 통계 특성
        for i in range(batch_size):
            numbers = lottery_numbers[i].float()
            feature_vector = torch.tensor(
                [
                    numbers.sum(),
                    numbers.mean(),
                    numbers.std(),
                    numbers.min(),
                    numbers.max(),
                    (numbers % 2 == 1).float().sum(),  # 홀수 개수
                    (numbers % 2 == 0).float().sum(),  # 짝수 개수
                ]
            )
            features.append(feature_vector)

        return torch.stack(features)

    def get_kernel_stats(self) -> Dict[str, Any]:
        """커널 통계 정보 반환"""
        return {
            "device_id": self.device_id,
            "device_name": (
                torch.cuda.get_device_name(self.device_id)
                if torch.cuda.is_available()
                else "CPU"
            ),
            "memory_manager_stats": self.memory_manager.get_stats(),
            "kernel_cache_size": len(self.kernel_cache),
            "cuda_available": torch.cuda.is_available(),
        }


# 전역 커널 인스턴스
_gpu_kernels = None


def get_gpu_pattern_kernels(device_id: int = 0) -> GPUPatternKernels:
    """GPU 패턴 커널 인스턴스 반환 (싱글톤)"""
    global _gpu_kernels

    if _gpu_kernels is None:
        _gpu_kernels = GPUPatternKernels(device_id)

    return _gpu_kernels


def cleanup_gpu_kernels():
    """GPU 커널 정리"""
    global _gpu_kernels

    if _gpu_kernels is not None:
        _gpu_kernels.memory_manager.shutdown()
        _gpu_kernels = None
