"""
모델 평가 모듈 (Evaluator)

이 모듈은 로또 번호 추천 시스템의 모델 평가 기능을 제공합니다.
다양한 모델의 성능을 평가하고 비교합니다.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import json
from datetime import datetime

from ..shared.types import LotteryNumber
from ..utils.error_handler_refactored import get_logger
from ..models.rl_model import RLModel
from ..models.statistical_model import StatisticalModel
from ..models.base_model import BaseModel

# 로거 설정
logger = get_logger(__name__)


class Evaluator:
    """모델 평가 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        평가기 초기화

        Args:
            config: 설정 객체
        """
        # 기본 설정
        default_config = {
            "cost_per_ticket": 1000,  # 로또 1장 가격
            "prize_tiers": {
                6: 1000000000,  # 1등 (6개 일치): 10억원
                5: 1500000,  # 3등 (5개 일치): 150만원
                4: 50000,  # 4등 (4개 일치): 5만원
                3: 5000,  # 5등 (3개 일치): 5천원
            },
            "save_results": True,  # 결과 저장 여부
            "results_dir": "data/evaluation",  # 결과 저장 디렉토리
            "model_paths": {
                "rl": "savedModels/rl_model.pt",
                "statistical": "savedModels/statistical_model.pt",
                "lstm": "savedModels/lstm_model.pt",
                "gnn": "savedModels/gnn_model.pt",
            },
        }

        # 설정 병합
        self.config = default_config.copy()
        if config:
            if "evaluation" in config:
                self.config.update(config["evaluation"])
            else:
                self.config.update(config)

        # 결과 저장 디렉토리 생성
        if self.config["save_results"]:
            results_dir = Path(self.config["results_dir"])
            results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("평가기 초기화 완료")

    def evaluate_model(
        self, model_name: str, test_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        단일 모델 평가

        Args:
            model_name: 모델 이름 ("rl", "statistical", "lstm", "gnn")
            test_data: 테스트 데이터

        Returns:
            평가 결과
        """
        logger.info(f"{model_name} 모델 평가 시작")
        start_time = time.time()

        # 모델 로드
        model = self._load_model(model_name)
        if not model:
            logger.error(f"{model_name} 모델 로드 실패")
            return {
                model_name: {
                    "success": False,
                    "error": f"{model_name} 모델 로드 실패",
                }
            }

        # 평가 수행
        try:
            eval_result = model.evaluate(test_data)

            # 결과 로깅
            logger.info(
                f"{model_name} 모델 평가 완료: "
                f"평균 일치 번호 수 {eval_result.get('avg_matches', 0):.2f}, "
                f"평균 보상 {eval_result.get('avg_reward', 0):.2f}"
            )

            # 평가 시간 추가
            eval_result["evaluation_time"] = time.time() - start_time
            eval_result["success"] = True

            return {model_name: eval_result}

        except Exception as e:
            logger.error(f"{model_name} 모델 평가 중 오류: {str(e)}")
            return {
                model_name: {
                    "success": False,
                    "error": str(e),
                    "evaluation_time": time.time() - start_time,
                }
            }

    def evaluate_all_models(self, test_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        모든 모델 평가

        Args:
            test_data: 테스트 데이터

        Returns:
            모든 모델의 평가 결과
        """
        logger.info("모든 모델 평가 시작")
        start_time = time.time()

        # 평가할 모델 목록
        model_names = ["rl", "statistical"]

        # 각 모델 평가
        results = {}
        for model_name in model_names:
            model_result = self.evaluate_model(model_name, test_data)
            results.update(model_result)

        # 결과 저장
        if self.config["save_results"]:
            self._save_evaluation_results(results)

        logger.info(f"모든 모델 평가 완료: 소요 시간 {time.time() - start_time:.2f}초")

        return results

    def _load_model(self, model_name: str) -> Optional[BaseModel]:
        """
        모델 로드

        Args:
            model_name: 모델 이름

        Returns:
            로드된 모델 (실패 시 None)
        """
        try:
            # 모델 경로
            model_path = self.config["model_paths"].get(model_name)
            if not model_path:
                logger.error(f"{model_name} 모델의 경로가 설정되지 않았습니다.")
                return None

            # 모델 타입에 따라 인스턴스 생성
            if model_name == "rl":
                model = RLModel()
            elif model_name == "statistical":
                model = StatisticalModel()
            else:
                logger.error(f"지원되지 않는 모델 타입: {model_name}")
                return None

            # 모델 로드
            if model.load(model_path):
                logger.info(f"{model_name} 모델 로드 성공: {model_path}")
                return model
            else:
                logger.error(f"{model_name} 모델 로드 실패: {model_path}")
                return None

        except Exception as e:
            logger.error(f"{model_name} 모델 로드 중 오류: {str(e)}")
            return None

    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        평가 결과 저장

        Args:
            results: 평가 결과
        """
        try:
            # 결과 파일 경로
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"
            file_path = Path(self.config["results_dir"]) / filename

            # JSON 파일로 저장
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "results": results,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            logger.info(f"평가 결과 저장 완료: {file_path}")

        except Exception as e:
            logger.error(f"평가 결과 저장 중 오류: {str(e)}")


# 싱글톤 인스턴스
_evaluator_instance = None


def get_evaluator(config: Optional[Dict[str, Any]] = None) -> Evaluator:
    """
    평가기 인스턴스 반환 (싱글톤)

    Args:
        config: 설정 객체

    Returns:
        Evaluator 인스턴스
    """
    global _evaluator_instance

    if _evaluator_instance is None:
        _evaluator_instance = Evaluator(config)

    return _evaluator_instance
