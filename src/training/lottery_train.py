"""
로또 번호 예측 모델 훈련 파이프라인

이 모듈은 로또 번호 예측을 위한 다양한 모델(RL, 통계 등)의 통합 훈련 파이프라인을 제공합니다.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import gc
import json
from datetime import datetime

import torch
import numpy as np

from ..shared.types import LotteryNumber, PatternAnalysis, ModelPrediction
from ..utils.error_handler_refactored import get_logger
from ..utils.dynamic_batch_size_utils import get_safe_batch_size
from ..core.state_vector_builder import StateVectorBuilder

# 훈련 모듈 임포트
from .train_rl import RLTrainer
from .train_rl_extended import EnhancedRLTrainer
from .pattern_trainer import PatternTrainer

# 로거 설정
logger = get_logger(__name__)


class LotteryTrainer:
    """로또 번호 예측 통합 훈련 클래스"""

    def __init__(self, config_path: Optional[str] = None):
        """
        로또 훈련 파이프라인 초기화

        Args:
            config_path: 설정 파일 경로 (선택적)
        """
        self.logger = get_logger(__name__)
        self.logger.info("로또 훈련 파이프라인 초기화")

        # 설정 로드
        self.config = self._load_config(config_path)

        # 경로 설정 (상대 경로 사용)
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "savedModels"
        self.log_dir = self.base_dir / "logs"

        # 각 디렉토리 생성
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 모델 저장소
        self.models = {}
        self.trainers = {}

        # 상태 벡터 생성기
        self.embedding_dim = self.config.get("embedding_dim", 32)
        self.state_builder = StateVectorBuilder(embedding_dim=self.embedding_dim)

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        설정 파일 로드

        Args:
            config_path: 설정 파일 경로

        Returns:
            설정 사전
        """
        # 기본 설정
        default_config = {
            "use_cuda": torch.cuda.is_available(),
            "batch_size": 64,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "num_episodes": 1000,
            "hidden_dim": 128,
            "embedding_dim": 32,
            "use_enhanced_rl": True,
        }

        # 설정 파일이 제공된 경우 로드
        if config_path is not None:
            try:
                config_file = Path(config_path)
                if config_file.exists():
                    with open(config_file, "r", encoding="utf-8") as f:
                        file_config = json.load(f)
                        # 기본 설정에 파일 설정 병합
                        default_config.update(file_config)
                        self.logger.info(f"설정 파일을 로드했습니다: {config_path}")
            except Exception as e:
                self.logger.error(f"설정 파일 로드 실패: {str(e)}")

        return default_config

    def load_data(self, data_path: Optional[str] = None) -> List[LotteryNumber]:
        """
        로또 데이터 로드

        Args:
            data_path: 데이터 파일 경로 (선택적)

        Returns:
            로또 번호 목록
        """
        import pandas as pd

        # 기본 경로 설정
        if data_path is None:
            data_path = str(self.data_dir / "lottery_history.csv")

        self.logger.info(f"로또 데이터 로드: {data_path}")

        try:
            # CSV 파일 로드
            df = pd.read_csv(data_path)

            # LotteryNumber 객체로 변환
            lottery_numbers = []
            for _, row in df.iterrows():
                try:
                    # 필요한 컬럼 추출
                    draw_no = int(row.get("seqNum", 0))

                    # 당첨 번호 추출
                    numbers = []
                    for i in range(1, 7):
                        col_name = f"num{i}"
                        if col_name in row and not pd.isna(row[col_name]):
                            numbers.append(int(row[col_name]))

                    # 유효한 번호 세트만 추가
                    if len(numbers) == 6 and all(1 <= n <= 45 for n in numbers):
                        lottery_number = LotteryNumber(
                            draw_no=draw_no,
                            numbers=sorted(numbers),  # 오름차순 정렬
                            date=None,  # 날짜 정보가 없을 경우
                        )
                        lottery_numbers.append(lottery_number)
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"데이터 변환 오류 (회차 {row.get('seqNum', 'Unknown')}): {e}"
                    )

            self.logger.info(f"{len(lottery_numbers)}개 로또 번호 데이터 로드 완료")
            return lottery_numbers

        except Exception as e:
            self.logger.error(f"데이터 로드 오류: {str(e)}")
            return []

    def split_data(
        self, data: List[LotteryNumber], val_ratio: float = 0.1, test_ratio: float = 0.1
    ) -> Tuple[List[LotteryNumber], List[LotteryNumber], List[LotteryNumber]]:
        """
        데이터 분할

        Args:
            data: 전체 데이터
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율

        Returns:
            (훈련 데이터, 검증 데이터, 테스트 데이터) 튜플
        """
        total_count = len(data)

        # 데이터가 너무 적으면 분할 비율 조정
        if total_count < 20:
            val_ratio = min(0.2, val_ratio)
            test_ratio = min(0.1, test_ratio)

        # 테스트 데이터 크기 계산
        test_size = max(1, int(total_count * test_ratio))
        # 검증 데이터 크기 계산
        val_size = max(1, int(total_count * val_ratio))
        # 훈련 데이터 크기 계산
        train_size = total_count - test_size - val_size

        # 데이터 분할 (시간순 정렬 유지)
        train_data = data[:train_size]
        val_data = data[train_size : train_size + val_size]
        test_data = data[train_size + val_size :]

        self.logger.info(
            f"데이터 분할: 훈련={len(train_data)}개, 검증={len(val_data)}개, 테스트={len(test_data)}개"
        )

        return train_data, val_data, test_data

    def analyze_patterns(self, data: List[LotteryNumber]) -> PatternAnalysis:
        """
        패턴 분석 수행

        Args:
            data: 분석할 로또 데이터

        Returns:
            패턴 분석 결과
        """
        self.logger.info(f"패턴 분석 시작: {len(data)}개 데이터")

        # 패턴 분석기 초기화
        pattern_trainer = PatternTrainer()

        # 패턴 분석 수행
        pattern_analysis = pattern_trainer.analyze_patterns(data)

        self.logger.info("패턴 분석 완료")
        return pattern_analysis

    def train_rl_model(
        self,
        train_data: List[LotteryNumber],
        val_data: List[LotteryNumber],
        pattern_analysis: PatternAnalysis,
        enhanced: bool = True,
    ) -> Dict[str, Any]:
        """
        강화학습 모델 훈련

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            pattern_analysis: 패턴 분석 결과
            enhanced: 향상된 RL 모델 사용 여부

        Returns:
            훈련 결과
        """
        self.logger.info(f"{'향상된 ' if enhanced else ''}강화학습 모델 훈련 시작")

        # 트레이너 선택
        if enhanced:
            trainer = EnhancedRLTrainer(
                config=self.config,
                state_builder=self.state_builder,
            )
        else:
            trainer = RLTrainer(
                config=self.config,
            )

        # 트레이너 저장
        model_type = "enhanced_rl" if enhanced else "rl"
        self.trainers[model_type] = trainer

        # 훈련 수행
        start_time = time.time()
        train_result = trainer.train(
            data=train_data,
            validation_data=val_data,
            pattern_analysis=pattern_analysis,
            episodes=self.config.get("num_episodes", 1000),
            batch_size=self.config.get("batch_size", 64),
        )
        training_time = time.time() - start_time

        # 결과 요약
        result = {
            "model_type": model_type,
            "training_time": training_time,
            "best_reward": train_result.get("best_reward", 0),
            "best_episode": train_result.get("best_episode", 0),
            "success": train_result.get("success", False),
        }

        self.logger.info(
            f"{'향상된 ' if enhanced else ''}강화학습 모델 훈련 완료: {training_time:.1f}초"
        )
        return result

    def recommend(
        self, model_type: str = "enhanced_rl", count: int = 5
    ) -> List[ModelPrediction]:
        """
        모델을 사용한 번호 추천 생성

        Args:
            model_type: 사용할 모델 유형
            count: 생성할 추천 세트 수

        Returns:
            추천 번호 목록
        """
        self.logger.info(f"{model_type} 모델 추천 번호 생성: {count}개")

        # 트레이너 존재 확인
        if model_type not in self.trainers:
            self.logger.error(f"사용 가능한 {model_type} 모델이 없습니다")
            return []

        # 추천 생성
        trainer = self.trainers[model_type]
        recommendations = trainer.generate_recommendations(count)

        # 로깅
        for i, rec in enumerate(recommendations):
            self.logger.info(
                f"추천 {i+1}: {rec.numbers} (신뢰도: {rec.confidence:.2f})"
            )

        return recommendations

    def run_pipeline(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        전체 훈련 파이프라인 실행

        Args:
            data_path: 데이터 파일 경로 (선택적)

        Returns:
            파이프라인 실행 결과
        """
        pipeline_result = {
            "start_time": datetime.now().isoformat(),
            "models": {},
            "recommendations": {},
        }

        # 1. 데이터 로드
        data = self.load_data(data_path)
        if not data:
            self.logger.error("데이터 로드 실패")
            pipeline_result["error"] = "데이터 로드 실패"
            return pipeline_result

        # 2. 데이터 분할
        train_data, val_data, test_data = self.split_data(data)

        # 3. 패턴 분석
        pattern_analysis = self.analyze_patterns(train_data)

        # 4. 강화학습 모델 훈련
        use_enhanced = self.config.get("use_enhanced_rl", True)

        if use_enhanced:
            # 향상된 RL 모델 훈련
            enhanced_rl_result = self.train_rl_model(
                train_data=train_data,
                val_data=val_data,
                pattern_analysis=pattern_analysis,
                enhanced=True,
            )
            pipeline_result["models"]["enhanced_rl"] = enhanced_rl_result

            # 추천 생성
            enhanced_recommendations = self.recommend(
                model_type="enhanced_rl",
                count=self.config.get("num_recommendations", 5),
            )

            # 결과 저장
            pipeline_result["recommendations"]["enhanced_rl"] = [
                {"numbers": rec.numbers, "confidence": rec.confidence}
                for rec in enhanced_recommendations
            ]
        else:
            # 기본 RL 모델 훈련
            rl_result = self.train_rl_model(
                train_data=train_data,
                val_data=val_data,
                pattern_analysis=pattern_analysis,
                enhanced=False,
            )
            pipeline_result["models"]["rl"] = rl_result

            # 추천 생성
            rl_recommendations = self.recommend(
                model_type="rl", count=self.config.get("num_recommendations", 5)
            )

            # 결과 저장
            pipeline_result["recommendations"]["rl"] = [
                {"numbers": rec.numbers, "confidence": rec.confidence}
                for rec in rl_recommendations
            ]

        # 5. 결과 정리
        pipeline_result["end_time"] = datetime.now().isoformat()

        # 결과 저장
        self._save_result(pipeline_result)

        return pipeline_result

    def _save_result(self, result: Dict[str, Any]) -> None:
        """
        파이프라인 결과 저장

        Args:
            result: 저장할 결과
        """
        try:
            # 결과 저장 디렉토리
            result_dir = self.log_dir / "results"
            result_dir.mkdir(parents=True, exist_ok=True)

            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = result_dir / f"pipeline_result_{timestamp}.json"

            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

            self.logger.info(f"파이프라인 결과 저장 완료: {result_path}")
        except Exception as e:
            self.logger.error(f"결과 저장 실패: {str(e)}")

    def cleanup(self) -> None:
        """자원 정리"""
        self.logger.info("자원 정리 시작")

        # 트레이너 자원 정리
        for trainer_name, trainer in self.trainers.items():
            if hasattr(trainer, "cleanup"):
                try:
                    trainer.cleanup()
                    self.logger.debug(f"{trainer_name} 트레이너 자원 정리 완료")
                except Exception as e:
                    self.logger.warning(
                        f"{trainer_name} 트레이너 자원 정리 실패: {str(e)}"
                    )

        # GPU 메모리 정리
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                self.logger.debug("CUDA 메모리 정리 완료")
            except Exception as e:
                self.logger.warning(f"CUDA 메모리 정리 실패: {str(e)}")

        self.logger.info("자원 정리 완료")


def main() -> None:
    """메인 함수"""
    # 설정 파일 경로
    config_path = (
        Path(__file__).parent.parent.parent / "config" / "lottery_train_config.json"
    )

    # 훈련 파이프라인 초기화
    trainer = LotteryTrainer(
        config_path=str(config_path) if config_path.exists() else None
    )

    # 파이프라인 실행
    result = trainer.run_pipeline()

    # 자원 정리
    trainer.cleanup()

    # 추천 번호 출력
    print("\n===== 추천 번호 =====")
    for model_type, recommendations in result.get("recommendations", {}).items():
        print(f"\n{model_type} 모델 추천:")
        for i, rec in enumerate(recommendations):
            numbers = rec["numbers"]
            confidence = rec["confidence"]
            print(f"{i+1}. {numbers} (신뢰도: {confidence:.2f})")


if __name__ == "__main__":
    main()
