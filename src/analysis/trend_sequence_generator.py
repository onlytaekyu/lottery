"""
추세 시퀀스 생성기 모듈

이 모듈은 로또 당첨 번호의 시계열 특성 시퀀스를 생성합니다.
LSTM 또는 Transformer 모델에 입력으로 사용할 수 있는 슬라이딩 윈도우 방식의 시퀀스를 생성합니다.
TrendAnalyzer에서 추출한 추세 특성을 시퀀스 형태로 변환합니다.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from ..shared.types import LotteryNumber
from ..utils.error_handler import get_logger
from ..utils.config_loader import ConfigProxy
from ..utils.performance_tracker import PerformanceTracker

# 로거 설정
logger = get_logger(__name__)


class TrendSequenceGenerator:
    """
    추세 시퀀스 생성기 클래스

    과거 당첨 번호 데이터를 바탕으로 시계열 특성 시퀀스를 생성합니다.
    TrendAnalyzer를 사용하여 각 회차의 추세 특성을 추출하고,
    이를 슬라이딩 윈도우 방식으로 시퀀스화합니다.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], ConfigProxy]] = None):
        """
        초기화

        Args:
            config: 설정 객체
        """
        # 설정 초기화
        self.config = config if config is not None else {}
        self.logger = get_logger(__name__)
        self.performance_tracker = PerformanceTracker()

        # 캐시 디렉토리 설정
        try:
            if isinstance(self.config, dict):
                self.cache_dir = Path(
                    self.config.get("paths", {}).get("cache_dir", "data/cache")
                )
            else:
                self.cache_dir = Path(self.config.get("paths.cache_dir", "data/cache"))
        except (KeyError, TypeError, AttributeError):
            self.logger.warning(
                "설정에서 'paths.cache_dir'를 찾을 수 없습니다. 기본값('data/cache')을 사용합니다."
            )
            self.cache_dir = Path("data/cache")

        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_trend_sequence(
        self,
        historical_data: List[LotteryNumber],
        window_size: int = 30,
        normalize: bool = True,
        save_to_cache: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        과거 당첨 번호 데이터를 바탕으로 추세 시퀀스를 생성합니다.

        Args:
            historical_data: 과거 당첨 번호 데이터 리스트 (draw_no 기준 오름차순 정렬)
            window_size: 시퀀스 윈도우 크기
            normalize: 특성별 정규화 여부
            save_to_cache: 생성된 시퀀스를 캐시에 저장할지 여부

        Returns:
            np.ndarray: 생성된 시퀀스 텐서 (형태: [num_sequences, window_size, trend_dim])
            또는 (시퀀스 텐서, 마스크 텐서) 튜플
        """
        with self.performance_tracker.track("generate_trend_sequence"):
            self.logger.info(
                f"추세 시퀀스 생성 시작: 데이터 {len(historical_data)}개, 윈도우 크기 {window_size}"
            )

            # 데이터 개수 검증
            if len(historical_data) < window_size + 1:
                self.logger.warning(
                    f"데이터가 부족합니다. 최소 {window_size + 1}개가 필요하지만 {len(historical_data)}개가 제공되었습니다."
                )
                return np.zeros((1, window_size, 9), dtype=np.float32)

            # 트렌드 특성 추출
            trend_vectors = self._extract_trend_vectors(historical_data)

            # 데이터가 부족하면 오류 반환
            if len(trend_vectors) < window_size:
                self.logger.warning(
                    f"추출된 트렌드 벡터가 부족합니다: {len(trend_vectors)}개 (윈도우 크기: {window_size})"
                )
                return np.zeros((1, window_size, 9), dtype=np.float32)

            # 시퀀스 생성
            sequences = []
            masks = []

            for i in range(len(trend_vectors) - window_size + 1):
                # 윈도우 시퀀스 추출
                window = trend_vectors[i : i + window_size]
                sequences.append(window)

                # 마스크 생성 (모두 유효)
                masks.append(np.ones(window_size, dtype=np.float32))

            # numpy 배열로 변환
            sequence_tensor = np.array(sequences, dtype=np.float32)
            mask_tensor = np.array(masks, dtype=np.float32)

            # 특성별 정규화 (선택적)
            if normalize:
                sequence_tensor = self._normalize_features(sequence_tensor)

            # 캐시에 저장 (선택적)
            if save_to_cache:
                self._save_to_cache(sequence_tensor, mask_tensor, window_size)

            # 텐서 형태 로깅
            self.logger.info(
                f"시퀀스 텐서 생성 완료: 형태 {sequence_tensor.shape} "
                f"[시퀀스 수, 윈도우 크기, 특성 차원]"
            )

            # 결과 반환
            return sequence_tensor, mask_tensor

    def _extract_trend_vectors(
        self, historical_data: List[LotteryNumber]
    ) -> List[np.ndarray]:
        """
        각 회차의 트렌드 특성 벡터를 추출합니다.

        Args:
            historical_data: 과거 당첨 번호 데이터 리스트

        Returns:
            List[np.ndarray]: 트렌드 특성 벡터 리스트
        """
        # TrendAnalyzer 임포트
        from ..analysis.trend_analyzer import TrendAnalyzer

        trend_vectors = []
        trend_analyzer = TrendAnalyzer(self.config)

        self.logger.info(f"트렌드 특성 벡터 추출 시작: {len(historical_data)}개 데이터")

        # 프로그레스 업데이트 간격
        progress_interval = max(1, len(historical_data) // 10)

        for i, draw in enumerate(historical_data):
            # 현재 회차까지의 데이터만 사용
            current_history = historical_data[: i + 1]

            try:
                # 트렌드 분석 수행
                trend_analysis = trend_analyzer.analyze(current_history)

                # 트렌드 특성 추출
                if "trend_features" in trend_analysis:
                    trend_features = trend_analysis["trend_features"]

                    # 9차원 벡터 생성
                    trend_vector = np.array(
                        [
                            trend_features.get("position_trend_slope_1", 0.0),
                            trend_features.get("position_trend_slope_2", 0.0),
                            trend_features.get("position_trend_slope_3", 0.0),
                            trend_features.get("position_trend_slope_4", 0.0),
                            trend_features.get("position_trend_slope_5", 0.0),
                            trend_features.get("position_trend_slope_6", 0.0),
                            trend_features.get("delta_mean", 0.0),
                            trend_features.get("delta_std", 0.0),
                            trend_features.get("segment_repeat_score", 0.5),
                        ],
                        dtype=np.float32,
                    )

                    trend_vectors.append(trend_vector)
                else:
                    # 기본 벡터 사용 (모든 특성이 0)
                    trend_vectors.append(np.zeros(9, dtype=np.float32))
            except Exception as e:
                self.logger.warning(
                    f"회차 {draw.draw_no} 트렌드 특성 추출 중 오류: {str(e)}"
                )
                # 오류 발생 시 기본 벡터 사용
                trend_vectors.append(np.zeros(9, dtype=np.float32))

            # 진행 상황 로깅
            if (i + 1) % progress_interval == 0 or i == len(historical_data) - 1:
                self.logger.info(
                    f"트렌드 벡터 추출 진행: {i+1}/{len(historical_data)} ({(i+1)/len(historical_data)*100:.1f}%)"
                )

        self.logger.info(f"트렌드 특성 벡터 추출 완료: {len(trend_vectors)}개 벡터")
        return trend_vectors

    def _normalize_features(self, sequence_tensor: np.ndarray) -> np.ndarray:
        """
        특성별 정규화를 수행합니다.

        Args:
            sequence_tensor: 정규화할 시퀀스 텐서

        Returns:
            np.ndarray: 정규화된 시퀀스 텐서
        """
        # 특성 차원
        _, _, feat_dim = sequence_tensor.shape

        # 결과 텐서 초기화
        normalized_tensor = np.zeros_like(sequence_tensor)

        # 각 특성별로 정규화
        for i in range(feat_dim):
            # 특성 데이터 추출
            feature_data = sequence_tensor[:, :, i].flatten()

            # 최대/최소값 계산
            min_val = np.min(feature_data)
            max_val = np.max(feature_data)

            # 정규화 (최대-최소가 0인 경우 0.5로 설정)
            if max_val > min_val:
                normalized_tensor[:, :, i] = (sequence_tensor[:, :, i] - min_val) / (
                    max_val - min_val
                )
            else:
                normalized_tensor[:, :, i] = 0.5

        return normalized_tensor

    def _save_to_cache(
        self, sequence_tensor: np.ndarray, mask_tensor: np.ndarray, window_size: int
    ) -> None:
        """
        생성된 시퀀스 텐서를 캐시에 저장합니다.

        Args:
            sequence_tensor: 저장할 시퀀스 텐서
            mask_tensor: 저장할 마스크 텐서
            window_size: 윈도우 크기
        """
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d")
        sequence_file = self.cache_dir / f"trend_sequences_w{window_size}.npy"
        mask_file = self.cache_dir / f"trend_sequences_mask_w{window_size}.npy"
        metadata_file = self.cache_dir / f"trend_sequences_metadata_w{window_size}.json"

        try:
            # 시퀀스 텐서 저장
            np.save(sequence_file, sequence_tensor)

            # 마스크 텐서 저장
            np.save(mask_file, mask_tensor)

            # 메타데이터 저장
            metadata = {
                "window_size": window_size,
                "sequence_shape": sequence_tensor.shape,
                "feature_names": [
                    "position_trend_slope_1",
                    "position_trend_slope_2",
                    "position_trend_slope_3",
                    "position_trend_slope_4",
                    "position_trend_slope_5",
                    "position_trend_slope_6",
                    "delta_mean",
                    "delta_std",
                    "segment_repeat_score",
                ],
                "generated_at": timestamp,
                "normalized": True,
            }

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.logger.info(f"시퀀스 데이터 캐시 저장 완료: {sequence_file}")
        except Exception as e:
            self.logger.error(f"시퀀스 데이터 캐시 저장 중 오류 발생: {str(e)}")

    def load_from_cache(
        self, window_size: int = 30
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        캐시에서 시퀀스 데이터를 로드합니다.

        Args:
            window_size: 윈도우 크기

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: (시퀀스 텐서, 마스크 텐서) 튜플 또는 None
        """
        # 파일 경로
        sequence_file = self.cache_dir / f"trend_sequences_w{window_size}.npy"
        mask_file = self.cache_dir / f"trend_sequences_mask_w{window_size}.npy"

        try:
            # 파일 존재 확인
            if not sequence_file.exists() or not mask_file.exists():
                self.logger.warning(f"캐시 파일이 존재하지 않습니다: {sequence_file}")
                return None

            # 데이터 로드
            sequence_tensor = np.load(sequence_file)
            mask_tensor = np.load(mask_file)

            self.logger.info(f"캐시에서 시퀀스 데이터 로드 완료: {sequence_file}")
            self.logger.info(f"시퀀스 텐서 형태: {sequence_tensor.shape}")

            return sequence_tensor, mask_tensor
        except Exception as e:
            self.logger.error(f"캐시에서 시퀀스 데이터 로드 중 오류 발생: {str(e)}")
            return None

    def generate_trend_sequences_with_targets(
        self,
        historical_data: List[LotteryNumber],
        window_size: int = 30,
        normalize: bool = True,
        target_type: str = "numbers",  # "numbers" or "trend"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        타겟 값이 포함된 트렌드 시퀀스를 생성합니다.
        모델 학습에 사용할 수 있는 입력-타겟 쌍을 생성합니다.

        Args:
            historical_data: 과거 당첨 번호 데이터 리스트
            window_size: 시퀀스 윈도우 크기
            normalize: 특성별 정규화 여부
            target_type: 타겟 유형 ("numbers": 다음 회차 번호, "trend": 다음 회차 트렌드)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (시퀀스 텐서, 타겟 텐서) 튜플
        """
        with self.performance_tracker.track("generate_trend_sequences_with_targets"):
            self.logger.info(
                f"타겟이 포함된 추세 시퀀스 생성 시작: 윈도우 크기 {window_size}, 타겟 유형 {target_type}"
            )

            # 데이터 개수 검증
            if len(historical_data) < window_size + 1:
                self.logger.warning(
                    f"데이터가 부족합니다. 최소 {window_size + 1}개가 필요하지만 {len(historical_data)}개가 제공되었습니다."
                )
                return np.zeros((1, window_size, 9), dtype=np.float32), np.zeros(
                    (1, 6), dtype=np.float32
                )

            # 트렌드 특성 추출
            trend_vectors = self._extract_trend_vectors(historical_data)

            # 시퀀스와 타겟 생성
            sequences = []
            targets = []

            for i in range(len(trend_vectors) - window_size):
                # 윈도우 시퀀스 추출
                window = trend_vectors[i : i + window_size]
                sequences.append(window)

                # 타겟 추출
                if target_type == "numbers":
                    # 다음 회차 번호를 타겟으로 사용
                    next_draw = historical_data[i + window_size]
                    normalized_numbers = [num / 45.0 for num in next_draw.numbers]
                    targets.append(normalized_numbers)
                else:
                    # 다음 회차 트렌드 벡터를 타겟으로 사용
                    next_trend = trend_vectors[i + window_size]
                    targets.append(next_trend)

            # numpy 배열로 변환
            sequence_tensor = np.array(sequences, dtype=np.float32)
            target_tensor = np.array(targets, dtype=np.float32)

            # 특성별 정규화 (선택적)
            if normalize:
                sequence_tensor = self._normalize_features(sequence_tensor)

            # 텐서 형태 로깅
            self.logger.info(
                f"시퀀스 텐서 생성 완료: 형태 {sequence_tensor.shape}, "
                f"타겟 텐서 형태: {target_tensor.shape}"
            )

            return sequence_tensor, target_tensor


def generate_trend_sequence(
    historical_data: List[LotteryNumber],
    window_size: int = 30,
    normalize: bool = True,
    save_to_cache: bool = True,
    config: Optional[Union[Dict[str, Any], ConfigProxy]] = None,
) -> np.ndarray:
    """
    과거 당첨 번호 데이터를 바탕으로 추세 시퀀스를 생성합니다.

    Args:
        historical_data: 과거 당첨 번호 데이터 리스트 (draw_no 기준 오름차순 정렬)
        window_size: 시퀀스 윈도우 크기
        normalize: 특성별 정규화 여부
        save_to_cache: 생성된 시퀀스를 캐시에 저장할지 여부
        config: 설정 객체

    Returns:
        np.ndarray: 생성된 시퀀스 텐서 (형태: [num_sequences, window_size, trend_dim])
    """
    generator = TrendSequenceGenerator(config)
    sequence_tensor, _ = generator.generate_trend_sequence(
        historical_data,
        window_size=window_size,
        normalize=normalize,
        save_to_cache=save_to_cache,
    )

    # 샘플 값 로깅
    if len(sequence_tensor) > 0:
        sample_idx = min(3, len(sequence_tensor) - 1)
        logger.info(f"샘플 시퀀스 (인덱스 {sample_idx}):")
        logger.info(f"첫 번째 timestep: {sequence_tensor[sample_idx, 0]}")
        logger.info(f"마지막 timestep: {sequence_tensor[sample_idx, -1]}")

    return sequence_tensor


if __name__ == "__main__":
    """CLI 진입점"""
    import argparse
    from ..utils.data_loader import load_draw_history

    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description="추세 시퀀스 생성")
    parser.add_argument("--window", type=int, default=30, help="윈도우 크기")
    parser.add_argument("--normalize", action="store_true", help="특성별 정규화 수행")
    parser.add_argument(
        "--data", type=str, default="data/lottery_history.csv", help="데이터 파일 경로"
    )

    args = parser.parse_args()

    # 데이터 로드
    try:
        historical_data = load_draw_history(args.data)
        logger.info(f"{len(historical_data)}개 당첨 번호 데이터 로드 완료")

        # 시퀀스 생성
        sequence_tensor = generate_trend_sequence(
            historical_data, window_size=args.window, normalize=args.normalize
        )

        logger.info(f"시퀀스 생성 완료: 형태 {sequence_tensor.shape}")
    except Exception as e:
        logger.error(f"시퀀스 생성 중 오류 발생: {str(e)}")
