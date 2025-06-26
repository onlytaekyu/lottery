import os
import sys
import argparse
import logging
import numpy as np
import torch
from typing import List, Dict, Optional, Any
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from ..shared.types import LotteryNumber, PatternAnalysis
from ..core.state_vector_builder import StateVectorBuilder
from ..training.train_rl_extended import EnhancedRLTrainer
from ..utils.memory_manager import MemoryManager
from ..utils.error_handler import get_logger


class RLPipeline:
    """로또 예측을 위한 강화학습 파이프라인"""

    def __init__(self, config_path: Optional[str] = None):
        """초기화"""
        # 로거 설정
        self.logger = get_logger("rl_pipeline")

        # 기본 설정 경로 (상대 경로 사용)
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent.parent / "config" / "rl_config.json"
            )

        # 설정 로드
        self.config = self._load_config(config_path)

        # 메모리 관리자 초기화
        self.memory_manager = MemoryManager()

        # 경로 설정 (상대 경로 사용)
        base_dir = Path(__file__).parent.parent.parent
        self.data_path = str(
            base_dir / self.config.get("data_path", "data/lottery_history.csv")
        )
        self.model_path = str(
            base_dir / self.config.get("model_path", "models/rl/policy_latest.pt")
        )
        self.log_dir = str(
            base_dir / "logs" / "rl" / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # 데이터 로드
        self.draw_history = self._load_data()

        # 패턴 분석
        self.pattern_analysis = self._analyze_patterns()

        # 상태 벡터 생성기 초기화
        embedding_dim = self.config.get("embedding_dim", 32)
        self.state_builder = StateVectorBuilder(embedding_dim=embedding_dim)

        # RL 트레이너 초기화
        self.trainer = EnhancedRLTrainer(
            config=self.config,
            train_data=self.draw_history,
            val_data=self.draw_history[-10:] if len(self.draw_history) > 10 else None,
            pattern_analysis=self.pattern_analysis,
            state_builder=self.state_builder,
        )

    def run(self, num_episodes: int = 1000, batch_size: int = 64) -> Dict[str, Any]:
        """
        파이프라인 실행

        Args:
            num_episodes: 학습 에피소드 수
            batch_size: 배치 크기

        Returns:
            실행 결과
        """
        self.logger.info("=== [시작] 강화학습 파이프라인 실행 ===")

        # 1. 모델 학습
        self.logger.info("1. 모델 학습 시작")
        train_result = self.trainer.train(
            self.draw_history, episodes=num_episodes, batch_size=batch_size
        )
        self.logger.info(
            f"학습 완료. 최고 보상: {train_result.get('best_reward', 0):.4f}"
        )

        # 2. 번호 추천
        self.logger.info("2. 번호 추천 생성")
        recommendations = self.trainer.generate_recommendations(
            data=self.draw_history, num_sets=5
        )
        self.logger.info(f"추천 생성 완료: {len(recommendations)}개")

        # 3. 결과 반환
        results = {
            "train_result": train_result,
            "recommendations": recommendations,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.logger.info("=== [종료] 강화학습 파이프라인 실행 ===")
        return results

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config
        except Exception as e:
            self.logger.error(f"설정 파일 로드 오류: {str(e)}")
            # 기본 설정 반환
            return {
                "data_path": "data/lottery_history.csv",
                "num_episodes": 1000,
                "batch_size": 64,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "embedding_dim": 32,
                "num_recommendations": 5,
                "run_backtest": True,
            }

    def _load_data(self) -> List[LotteryNumber]:
        """데이터 로드 및 전처리"""
        try:
            # CSV 파일에서 로또 당첨 번호 데이터 로드
            df = pd.read_csv(self.data_path)

            # 데이터프레임을 LotteryNumber 객체 리스트로 변환
            lottery_numbers = []
            for _, row in df.iterrows():
                # 필요한 컬럼 추출 (데이터 형식에 맞게 수정 필요)
                try:
                    round_num = int(row.get("seqNum", 0))

                    # 당첨 번호 추출 (컬럼 이름에 맞게 수정 필요)
                    numbers = []
                    for i in range(1, 7):
                        col_name = f"num{i}"
                        if col_name in row and pd.notna(row[col_name]):
                            numbers.append(int(row[col_name]))

                    # 유효한 번호 세트만 추가
                    if len(numbers) == 6 and all(1 <= n <= 45 for n in numbers):
                        lottery_number = LotteryNumber(
                            draw_no=round_num,
                            numbers=sorted(numbers),
                            date=None,  # 날짜 정보가 없는 경우
                        )
                        lottery_numbers.append(lottery_number)
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"데이터 변환 오류 (회차 {row.get('seqNum', 'Unknown')}): {e}"
                    )
                    continue

            self.logger.info(f"{len(lottery_numbers)}개의 로또 당첨 번호 로드 완료")
            return lottery_numbers

        except Exception as e:
            self.logger.error(f"데이터 로드 오류: {str(e)}")
            raise

    def _analyze_patterns(self) -> PatternAnalysis:
        """패턴 분석 수행"""
        # 빈도 계산
        frequency = {}
        for num in range(1, 46):
            frequency[num] = 0

        for draw in self.draw_history:
            for num in draw.numbers:
                frequency[num] = frequency.get(num, 0) + 1

        # 핫넘버/콜드넘버 계산
        sorted_numbers = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

        # 상위 30% 핫넘버, 하위 30% 콜드넘버
        hot_size = int(len(sorted_numbers) * 0.3)
        hot_numbers = set(num for num, _ in sorted_numbers[:hot_size])
        cold_numbers = set(num for num, _ in sorted_numbers[-hot_size:])

        # 번호 쌍 빈도 계산
        pair_frequency = {}
        for draw in self.draw_history:
            for i, num1 in enumerate(draw.numbers):
                for num2 in draw.numbers[i + 1 :]:
                    pair = (min(num1, num2), max(num1, num2))  # 항상 오름차순 정렬
                    pair_frequency[pair] = pair_frequency.get(pair, 0) + 1

        # ROI 메트릭 딕셔너리로 변환
        roi_metrics = {}
        for num1 in range(1, 46):
            for num2 in range(num1 + 1, 46):
                pair_key = (num1, num2)
                roi_metrics[pair_key] = (
                    pair_frequency.get(pair_key, 0) / len(self.draw_history)
                    if self.draw_history
                    else 0
                )

        # PatternAnalysis 객체 생성 및 반환
        pattern_analysis = PatternAnalysis(
            frequency_map=frequency,
            recency_map=frequency.copy(),  # 임시로 frequency와 동일하게 설정
            pair_frequency=pair_frequency,
            hot_numbers=hot_numbers,
            cold_numbers=cold_numbers,
            sum_distribution=[],  # 빈 리스트로 초기화
            gap_patterns={},
            probability_matrix={},
            metadata={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_count": len(self.draw_history),
            },
        )

        self.logger.info("패턴 분석 완료")
        return pattern_analysis

    def _save_recommendations(self, recommendations: List[List[int]]) -> None:
        """추천 번호 저장"""
        # 추천 결과 객체 생성
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommendations": [
                {"numbers": sorted(nums), "confidence": 0.0}  # 오름차순 정렬 보장
                for nums in recommendations
            ],
        }

        # 테스트 중 파일 저장 여부 확인
        save_during_test = self.config.get("recommendation_output", {}).get(
            "save_during_test", False
        )
        if not save_during_test:
            # 테스트 중이므로 로그만 남기고 파일은 저장하지 않음
            self.logger.info(f"테스트 중 추천 번호가 생성되었습니다 (저장하지 않음)")
            for i, numbers in enumerate(recommendations):
                self.logger.info(f"추천 세트 {i+1}: {sorted(numbers)}")
            return

        # JSON 형식으로 저장 (테스트 중 저장이 활성화된 경우만)
        result_path = os.path.join(self.log_dir, "recommendations.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        # 콘솔에 결과 출력
        self.logger.info(f"추천 번호 저장 완료: {result_path}")
        for i, numbers in enumerate(recommendations):
            self.logger.info(
                f"추천 세트 {i+1}: {sorted(numbers)}"
            )  # 오름차순 정렬 출력

    def _run_backtest(self, recommendations: List[List[int]]) -> None:
        """백테스트 실행"""
        if not self.config.get("run_backtest", False):
            return

        self.logger.info("백테스트 시작...")

        # 실제 당첨 번호 목록 (최근 30회)
        recent_draws = (
            self.draw_history[-30:]
            if len(self.draw_history) >= 30
            else self.draw_history
        )

        total_matches = 0
        match_counts = {3: 0, 4: 0, 5: 0, 6: 0}

        # 각 추천 번호 세트에 대해 백테스트
        for recommendation in recommendations:
            sorted_rec = sorted(recommendation)  # 오름차순 정렬 보장
            for draw in recent_draws:
                sorted_draw = sorted(draw.numbers)  # 오름차순 정렬 보장

                # 일치하는 번호 개수 계산
                match_count = len(set(sorted_rec) & set(sorted_draw))

                # 3개 이상 일치 시 기록
                if match_count >= 3:
                    match_counts[match_count] = match_counts.get(match_count, 0) + 1
                    total_matches += 1

        # 결과 출력
        self.logger.info(f"백테스트 결과: 총 {total_matches}회 당첨 (3개 이상 일치)")
        for count, occurrences in sorted(match_counts.items()):
            self.logger.info(f"  {count}개 일치: {occurrences}회")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="RL 파이프라인 실행")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "config" / "rl_config.json"),
        help="설정 파일 경로",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="학습 에피소드 수")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    args = parser.parse_args()

    # 파이프라인 초기화 및 실행
    pipeline = RLPipeline(config_path=args.config)
    results = pipeline.run(num_episodes=args.episodes, batch_size=args.batch_size)

    # 추천 결과 출력
    print("\n추천 번호:")
    for i, rec in enumerate(results.get("recommendations", [])):
        print(f"세트 {i+1}: {sorted(rec)}")


if __name__ == "__main__":
    main()
