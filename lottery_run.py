#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DAEBAK_AI 로또 추천 시스템 (Lottery Recommendation System)

이 스크립트는 DAEBAK_AI 로또 추천 시스템의 메인 인터페이스입니다.
다양한 모델과 알고리즘을 통합하여 로또 번호를 추천하고 학습시키는 기능을 제공합니다.

주요 기능:
- 로또 번호 추천 (클러스터링, 패턴 분석, 비지도 학습 및 추론 로직 활용)
- 모델 학습 (오토인코더, RL 에이전트, GNN 등)
- 백테스팅 및 강화 학습
"""

import os
import sys
import time
import logging
import yaml
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple

# 경고 억제
warnings.filterwarnings("ignore")

# 현재 디렉토리 경로 설정
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

# 로그 디렉토리 설정
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 중앙 로깅 시스템 초기화
from src.utils.error_handler_refactored import get_logger

# 로거 가져오기
logger = get_logger("lottery")

# 중앙 집중식 데이터 로딩
from src.utils.data_loader import load_draw_history
from src.shared.types import LotteryNumber
from src.analysis.pattern_analyzer import PatternAnalyzer

# 전역 데이터 변수
global_lottery_data: List[LotteryNumber] = []
global_pattern_analyses: Dict[str, Any] = {}


def load_global_data(data_path: Optional[str] = None) -> List[LotteryNumber]:
    """
    중앙 집중식 데이터 로딩 함수

    Args:
        data_path: 데이터 파일 경로 (기본값: None, config 파일에서 지정한 경로 사용)

    Returns:
        List[LotteryNumber]: 로또 당첨 번호 데이터
    """
    global global_lottery_data

    # 설정 파일 로드
    config = load_config()

    # 데이터 경로가 지정되지 않은 경우 설정에서 가져옴
    if data_path is None:
        if config and "data" in config and "historical_data_path" in config["data"]:
            data_path = config["data"]["historical_data_path"]
        else:
            # 기본 경로 사용
            data_path = str(BASE_DIR / "data" / "raw" / "lottery.csv")

    # 데이터 로드
    try:
        logger.info(f"중앙 데이터 로딩 시작: {data_path}")
        global_lottery_data = load_draw_history(data_path)
        logger.info(f"중앙 데이터 로딩 완료: {len(global_lottery_data)}개 항목")
        return global_lottery_data
    except Exception as e:
        logger.error(f"데이터 로딩 실패: {str(e)}")
        return []


def get_data_split(
    test_size: float = 0.2, val_size: float = 0.1
) -> Tuple[List[LotteryNumber], List[LotteryNumber], List[LotteryNumber]]:
    """
    데이터를 훈련, 검증, 테스트 세트로 분할

    Args:
        test_size: 테스트 세트 비율
        val_size: 검증 세트 비율

    Returns:
        Tuple[List[LotteryNumber], List[LotteryNumber], List[LotteryNumber]]: 훈련, 검증, 테스트 세트
    """
    global global_lottery_data

    # 데이터가 로드되지 않은 경우 로드
    if not global_lottery_data:
        load_global_data()

    # 데이터가 없는 경우 빈 리스트 반환
    if not global_lottery_data:
        return [], [], []

    # 데이터 순서대로 정렬 (회차 번호 기준)
    sorted_data = sorted(global_lottery_data, key=lambda x: x.draw_no)

    # 데이터 분할
    total_size = len(sorted_data)
    test_idx = int(total_size * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))

    train_data = sorted_data[:val_idx]
    val_data = sorted_data[val_idx:test_idx]
    test_data = sorted_data[test_idx:]

    logger.info(
        f"데이터 분할: 훈련 {len(train_data)}개, 검증 {len(val_data)}개, 테스트 {len(test_data)}개"
    )

    return train_data, val_data, test_data


def get_latest_data(count: int = 10) -> List[LotteryNumber]:
    """
    최근 count개의 데이터 반환

    Args:
        count: 가져올 데이터 개수

    Returns:
        List[LotteryNumber]: 최근 데이터
    """
    global global_lottery_data

    # 데이터가 로드되지 않은 경우 로드
    if not global_lottery_data:
        load_global_data()

    # 데이터가 없는 경우 빈 리스트 반환
    if not global_lottery_data:
        return []

    # 데이터 회차 번호 기준으로 정렬
    sorted_data = sorted(global_lottery_data, key=lambda x: x.draw_no, reverse=True)

    # 최근 데이터 반환
    return sorted_data[:count]


def load_config(config_path=None) -> Dict[str, Any]:
    """
    설정 파일 로드

    Args:
        config_path: 설정 파일 경로 (기본값: None, 'config/config.yaml' 사용)

    Returns:
        설정 객체
    """
    # 기본 설정 파일 경로
    if config_path is None:
        config_path = str(BASE_DIR / "config" / "config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"설정 파일 로드 완료: {config_path}")
        return config
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {str(e)}")
        return {}


def clear_screen():
    """화면 지우기"""
    os.system("cls" if os.name == "nt" else "clear")


def display_title():
    """시스템 타이틀 표시"""
    print("\n===== DAEBAK_AI Lottery System =====\n")


def display_menu():
    """메인 메뉴 표시"""
    print("1. 시스템 테스트")
    print("2. 모델 학습")
    print("3. 로또 번호 추천")
    print("4. 백테스트 및 강화 학습")
    print("0. 종료")
    print("\n선택: ", end="")


def display_test_menu():
    """시스템 테스트 메뉴 표시"""
    print("\n--- 시스템 테스트 ---")
    print("1. 데이터 분석 테스트")
    print("2. 머신러닝 테스트")
    print("0. 이전으로")
    print("\n선택: ", end="")


def run_data_analysis_test(config: Dict[str, Any]) -> bool:
    """
    데이터 분석 테스트 기능

    Args:
        config: 설정 객체

    Returns:
        bool: 성공 여부
    """
    logger.info("데이터 분석 테스트 시작")
    print("\n데이터 분석 테스트를 시작합니다...\n")

    try:
        # 테스트 스크립트 실행
        from src.test.test_data_analysis import run_test

        run_test()
        return True
    except Exception as e:
        logger.error(f"데이터 분석 테스트 실패: {str(e)}")
        print(f"오류: {str(e)}")
        return False


def run_statistical_model_test(config: Dict[str, Any]) -> bool:
    """
    머신러닝 테스트 기능

    Args:
        config: 설정 객체

    Returns:
        bool: 성공 여부
    """
    logger.info("머신러닝 테스트 시작")
    print("\n머신러닝 테스트를 시작합니다...\n")

    try:
        # 테스트 스크립트 실행
        from src.test.test_statistical_model import run_test

        run_test()
        return True
    except Exception as e:
        logger.error(f"머신러닝 테스트 실패: {str(e)}")
        print(f"오류: {str(e)}")
        return False


def option_1_test(config: Dict[str, Any]) -> bool:
    """
    시스템 테스트 기능

    Args:
        config: 설정 객체

    Returns:
        bool: 성공 여부
    """
    while True:
        clear_screen()
        display_title()
        display_test_menu()

        choice = input().strip()

        if choice == "1":
            run_data_analysis_test(config)
            input("\n계속하려면 Enter 키를 누르세요...")
        elif choice == "2":
            run_statistical_model_test(config)
            input("\n계속하려면 Enter 키를 누르세요...")
        elif choice == "0":
            return True
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")
            time.sleep(2)


def run_pattern_analysis(
    data: List[LotteryNumber], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    모든 스코프에 대한 패턴 분석을 실행합니다.

    Args:
        data: 로또 당첨 번호 데이터
        config: 설정 객체

    Returns:
        Dict[str, Any]: 스코프별 패턴 분석 결과
    """
    global global_pattern_analyses

    # 이미 분석된 결과가 있는 경우 반환
    if global_pattern_analyses:
        logger.info("기존 패턴 분석 결과 사용")
        return global_pattern_analyses

    logger.info("다단계 패턴 분석 시작")

    try:
        # 패턴 분석기 초기화
        pattern_analyzer = PatternAnalyzer(config.get("pattern_analysis", {}))

        # 캐시 설정 확인
        cache_config = config.get("analysis_cache", {})
        cache_enabled = cache_config.get("enabled", True)
        cache_path = cache_config.get("path", "data/cache/")

        # 캐시에서 로드 시도
        if cache_enabled:
            logger.info(f"패턴 분석 캐시 로드 시도: {cache_path}")
            if pattern_analyzer.load_analyses_from_cache(cache_path):
                logger.info("캐시에서 패턴 분석 결과 로드 성공")
                global_pattern_analyses = pattern_analyzer.get_all_analyses()
                return global_pattern_analyses

        # 캐시에 없으면 새로 계산
        logger.info("패턴 분석 계산 시작 (캐시 미사용)")
        analyses = pattern_analyzer.run_all_analyses(data)

        # 분석 보고서 생성 - 운영 모드에서만 파일 생성
        test_mode = config.get("test_mode", False)
        report_path = pattern_analyzer.write_analysis_report(test_mode=test_mode)
        if report_path:
            logger.info(f"패턴 분석 보고서 생성 완료: {report_path}")

        # 캐시 저장 시도
        if cache_enabled:
            logger.info(f"패턴 분석 결과 캐시 저장: {cache_path}")
            pattern_analyzer.save_analyses_to_cache(cache_path)

        # 전역 변수에 저장
        global_pattern_analyses = analyses

        return analyses
    except Exception as e:
        logger.error(f"패턴 분석 중 오류 발생: {str(e)}")
        if global_pattern_analyses:
            logger.warning("기존 패턴 분석 결과 사용")
            return global_pattern_analyses
        return {}


def apply_pattern_analysis_to_models(
    analyses: Dict[str, Any], models: Dict[str, Any]
) -> None:
    """
    패턴 분석 결과를 모델에 적용합니다.

    Args:
        analyses: 스코프별 패턴 분석 결과
        models: 적용할 모델 사전
    """
    if not analyses or not models:
        logger.warning("패턴 분석 결과 또는 모델이 없습니다")
        return

    logger.info(f"패턴 분석 결과를 {len(models)}개 모델에 적용합니다")

    # 각 모델별로 적절한 패턴 분석 결과 적용
    for model_name, model in models.items():
        try:
            # 모델별 적용 방식 결정
            if model_name == "statistical":
                # 통계 모델: 스코프별 빈도 맵 적용
                if hasattr(model, "set_frequency_maps"):
                    # 빈도 맵 추출
                    frequency_maps = {}
                    for scope, analysis in analyses.items():
                        if hasattr(analysis, "frequency_map"):
                            frequency_maps[scope] = analysis.frequency_map

                    # 스코프별 빈도 맵 설정
                    model.set_frequency_maps(frequency_maps)
                    logger.info(f"{model_name} 모델에 스코프별 빈도 맵 적용 완료")

                # 기본 패턴 분석 결과도 함께 적용
                if hasattr(model, "set_pattern_analysis"):
                    model.set_pattern_analysis(analyses.get("full", {}))
                    logger.info(f"{model_name} 모델에 기본 패턴 분석 결과 적용 완료")

            elif model_name == "lstm":
                # LSTM 모델: 주로 full 스코프 사용
                if hasattr(model, "set_pattern_analysis"):
                    model.set_pattern_analysis(analyses.get("full", {}))
                    logger.info(
                        f"{model_name} 모델에 full 스코프 패턴 분석 결과 적용 완료"
                    )

            elif model_name == "rl":
                # RL 모델: full과 mid 스코프 모두 중요
                if hasattr(model, "set_pattern_analysis"):
                    # 기본적으로 mid 스코프 적용 (최근 데이터가 더 중요)
                    model.set_pattern_analysis(
                        analyses.get("mid", analyses.get("full", {}))
                    )
                    logger.info(
                        f"{model_name} 모델에 mid 스코프 패턴 분석 결과 적용 완료"
                    )

            else:
                # 다른 모델: 기본적으로 full 스코프 적용
                if hasattr(model, "set_pattern_analysis"):
                    model.set_pattern_analysis(analyses.get("full", {}))
                    logger.info(f"{model_name} 모델에 패턴 분석 결과 적용 완료")

        except Exception as e:
            logger.error(f"{model_name} 모델에 패턴 분석 결과 적용 중 오류: {str(e)}")

    logger.info("모든 모델에 패턴 분석 결과 적용 완료")


def option_2_train(config: Dict[str, Any]) -> bool:
    """
    시스템 학습 기능

    Args:
        config: 설정 객체

    Returns:
        bool: 성공 여부
    """
    logger.info("시스템 학습 시작")
    print("\n시스템 학습을 시작합니다...\n")

    try:
        # 중앙 집중식 데이터 사용이 이미 로드 되어 있음
        data = load_global_data()

        # 패턴 분석 먼저 실행 (모든 스코프에 대해)
        print("모델 학습 전 패턴 분석을 실행합니다...")
        pattern_analyses = run_pattern_analysis(data, config)
        if not pattern_analyses:
            print("경고: 패턴 분석이 실패했습니다. 제한된 기능으로 계속 진행합니다.")
        else:
            print("패턴 분석 완료. 학습 구성 중...")

        # 지연 임포트: 필요할 때만 임포트
        # pylint: disable=import-outside-toplevel
        # 임포트 오류가 발생할 수 있지만, 이는 사용자 모듈이 나중에 생성될 것을 가정
        try:
            # 학습 모드 선택
            print("학습 모드를 선택하세요:")
            print("1. 전체 모델 학습")
            print("2. RL 모델 학습")
            print("3. GNN 모델 학습")
            print("4. 통계 모델 학습")
            print("5. LSTM 모델 학습")
            print("0. 취소")

            choice = input("\n선택: ")

            if choice == "0":
                print("학습이 취소되었습니다.")
                return False

            if choice not in ["1", "2", "3", "4", "5"]:
                print("잘못된 선택입니다. 메인 메뉴로 돌아갑니다.")
                return False

            # 학습 모드 매핑
            train_modes = {
                "1": "all",
                "2": "rl",
                "3": "gnn",
                "4": "statistical",
                "5": "lstm",
            }

            # 에폭 수 입력
            epochs = 100  # 기본값
            try:
                epochs_input = input("에폭 수를 입력하세요 (기본값: 100): ")
                if epochs_input.strip():
                    epochs = int(epochs_input)
            except ValueError:
                print("유효하지 않은 에폭 수입니다. 기본값 100을 사용합니다.")

            # 학습 시작
            print(
                f"\n{train_modes[choice]} 모델 학습을 {epochs}회 에폭으로 시작합니다..."
            )

            # 학습 설정 구성
            train_config = config.copy() if config else {}
            train_config.update(
                {
                    "epochs": epochs,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "win_rate": 0.5,
                    "avg_matches": 0.5,
                    "use_amp": True,
                    "pattern_analyses": pattern_analyses,  # 패턴 분석 결과 추가
                }
            )

            # 명령행 인자 설정 (모드 정보 포함)
            class Args:
                mode: int

                def __init__(self):
                    self.mode = 1

            args = Args()
            args.mode = int(choice)

            # train_models 함수 호출
            from src.training.train_interface import train_models

            result = train_models(args, train_config)

            # 결과 처리
            if result and not isinstance(result, dict):
                success = True
            elif isinstance(result, dict) and result.get("error") is None:
                success = True
            else:
                if isinstance(result, dict) and "error" in result:
                    print(f"오류: {result['error']}")
                success = False

            if success:
                print("\n학습이 성공적으로 완료되었습니다.")
            else:
                print("\n학습 중 문제가 발생했습니다.")

            return True

        except ImportError as e:
            logger.error(f"train_models 함수 임포트 실패: {str(e)}")
            print(f"오류: train_models 함수를 로드할 수 없습니다. {str(e)}")

            # 대체 방식으로 시도
            try:
                # 직접 run_training 함수 임포트 시도
                from src.training.train_interface import run_training  # type: ignore

                # 학습 모드 선택
                print("학습 모드를 선택하세요:")
                print("1. 전체 모델 학습")
                print("2. RL 모델 학습")
                print("3. GNN 모델 학습")
                print("4. 통계 모델 학습")
                print("5. LSTM 모델 학습")
                print("0. 취소")

                choice = input("\n선택: ")

                if choice == "0":
                    print("학습이 취소되었습니다.")
                    return False

                if choice not in ["1", "2", "3", "4", "5"]:
                    print("잘못된 선택입니다. 메인 메뉴로 돌아갑니다.")
                    return False

                # 학습 모드 매핑
                train_modes = {
                    "1": "all",
                    "2": "rl",
                    "3": "gnn",
                    "4": "statistical",
                    "5": "lstm",
                }

                # 에폭 수 입력
                epochs = 100  # 기본값
                try:
                    epochs_input = input("에폭 수를 입력하세요 (기본값: 100): ")
                    if epochs_input.strip():
                        epochs = int(epochs_input)
                except ValueError:
                    print("유효하지 않은 에폭 수입니다. 기본값 100을 사용합니다.")

                # 학습 시작
                print(
                    f"\n{train_modes[choice]} 모델 학습을 {epochs}회 에폭으로 시작합니다..."
                )
                result = run_training(
                    model_type=train_modes[choice], epochs=epochs, config=config
                )

                if result:
                    print("\n학습이 성공적으로 완료되었습니다.")
                else:
                    print("\n학습 중 문제가 발생했습니다.")

                return True

            except ImportError as e2:
                logger.error(f"run_training 함수 임포트 실패: {str(e2)}")
                print(f"오류: 필요한 훈련 모듈을 로드할 수 없습니다. {str(e2)}")
                return False

    except Exception as e:
        logger.error(f"시스템 학습 중 오류 발생: {str(e)}")
        print(f"오류: 시스템 학습 중 문제가 발생했습니다. {str(e)}")

    input("\n계속하려면 Enter 키를 누르세요...")
    return False


def option_1_predict(config: Dict[str, Any]) -> bool:
    """
    로또 번호 예측 기능

    Args:
        config: 설정 객체

    Returns:
        bool: 성공 여부
    """
    logger.info("로또 번호 예측 시작")
    print("\n로또 번호 예측을 시작합니다...\n")

    try:
        # 중앙 집중식 데이터 로드
        data = load_global_data()

        # 패턴 분석 먼저 실행 (모든 스코프에 대해)
        print("번호 추천 전 패턴 분석을 실행합니다...")
        pattern_analyses = run_pattern_analysis(data, config)
        if not pattern_analyses:
            print("경고: 패턴 분석이 실패했습니다. 제한된 기능으로 계속 진행합니다.")
        else:
            print("패턴 분석 완료. 추천 엔진 준비 중...")

        # 지연 임포트: 필요할 때만 임포트
        # pylint: disable=import-outside-toplevel
        try:
            # 모델 타입 선택
            print("예측에 사용할 모델을 선택하세요:")
            print("1. 통합 하이브리드 모델 (기본값)")
            print("2. 강화학습 모델")
            print("3. GNN 모델")
            print("4. 통계 모델")
            print("5. LSTM 모델")
            print("0. 취소")

            choice = input("\n선택: ")

            if choice == "0":
                print("예측이 취소되었습니다.")
                return False

            # 추천할 번호 세트 개수 입력
            try:
                count_input = input(
                    "\n추천할 번호 세트 개수를 입력하세요 (기본값: 5): "
                )
                count = 5  # 기본값
                if count_input.strip():
                    count = int(count_input)
                    if count <= 0:
                        print("유효하지 않은 개수입니다. 기본값 5를 사용합니다.")
                        count = 5
            except ValueError:
                print("유효하지 않은 입력입니다. 기본값 5를 사용합니다.")
                count = 5

            # 현재 회차 번호 입력
            try:
                round_input = input("\n현재 회차 번호를 입력하세요: ")
                round_number = None
                if round_input.strip():
                    round_number = int(round_input)
                    if round_number <= 0:
                        print("유효하지 않은 회차 번호입니다.")
                        round_number = None
            except ValueError:
                print("유효하지 않은 회차 번호입니다.")
                round_number = None

            # 회차 번호가 입력되지 않은 경우 저장하지 않음을 알림
            if round_number is None:
                print("\n회차 번호를 입력하지 않아 추천 결과는 화면에만 표시됩니다.")

            # 모델 타입 매핑
            model_types = {
                "1": "hybrid",
                "2": "rl",
                "3": "gnn",
                "4": "statistical",
                "5": "lstm",
            }

            # 기본값 설정
            model_type = model_types.get(choice, "hybrid")

            # 중앙 집중식 데이터 사용이 이미 로드 되어 있음
            load_global_data()

            # 설정 및 데이터 가져오기
            from src.core.recommendation_engine import get_recommendation_engine  # type: ignore

            # 추천 엔진 초기화
            engine = get_recommendation_engine(config)

            # 패턴 분석 결과 설정 (추천 엔진 내 분석 실행은 실행되지 않도록 미리 설정)
            engine.pattern_analyses = pattern_analyses

            # 추천 실행
            print(
                f"\n{model_type} 모델을 사용하여 {count}개의 번호 세트를 추천합니다..."
            )
            recommendations = engine.recommend(
                count=count,
                strategy=model_type,
                data=data,  # 데이터 명시적 전달
                model_types=None if model_type == "hybrid" else [model_type],
            )

            # 추천 결과 출력
            print("\n추천 번호:")
            if recommendations and len(recommendations) > 0:
                for i, rec in enumerate(recommendations):
                    numbers_str = " ".join([f"{n:2d}" for n in rec.numbers])
                    print(
                        f"추천 {i+1:2d}: [{numbers_str}] (신뢰도: {rec.confidence:.4f}, 모델: {rec.model_type})"
                    )
            else:
                print("추천할 번호가 없습니다.")

            print("================================\n")

            # 회차 번호가 있을 경우만 결과 저장
            if round_number is not None:
                # 추천 엔진의 save_recommendation 메서드를 사용하여 저장
                saved_path = engine.save_recommendation(
                    recommendations=recommendations,
                    round_number=round_number,
                    config=config,
                )

                if saved_path:
                    print(f"추천 결과가 저장되었습니다: {saved_path}")
                else:
                    print("추천 결과 저장에 실패했습니다.")

            return True

        except ImportError as e:
            logger.error(f"모듈 로드 실패: {str(e)}")
            print(f"오류: 필요한 모듈을 로드할 수 없습니다. {str(e)}")
        except Exception as e:
            logger.error(f"로또 번호 예측 중 오류 발생: {str(e)}")
            print(f"오류: 로또 번호 예측 중 문제가 발생했습니다. {str(e)}")

        input("\n계속하려면 Enter 키를 누르세요...")
        return False

    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {str(e)}")
        print(f"\n오류가 발생했습니다: {str(e)}")
        input("\n계속하려면 Enter 키를 누르세요...")

    return False


def option_3_backtest(config: Dict[str, Any]) -> bool:
    """
    백테스트 및 학습 강화 기능

    Args:
        config: 설정 객체

    Returns:
        bool: 성공 여부
    """
    logger.info("백테스트 및 학습 강화 시작")
    print("\n백테스트 및 학습 강화를 시작합니다...\n")

    try:
        # 지연 임포트: 필요할 때만 임포트
        # pylint: disable=import-outside-toplevel
        # 임포트 오류가 발생할 수 있지만, 이는 사용자 모듈이 나중에 생성될 것을 가정
        backtester_func = None
        reinforcer_func = None

        try:
            # 기존 Backtester 클래스 임포트 시도
            from src.evaluation.backtester import Backtester
            from src.training.unified_trainer import UnifiedTrainer

            # 백테스트 함수 정의
            def run_backtest_with_class(file_path, conf):
                """클래스 기반 백테스트 함수"""
                if file_path is None or not file_path.exists():
                    return None

                # 파일에서 추천 번호 읽기
                recommendations = []
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "추천" in line and "[" in line and "]" in line:
                            # 추천 번호 파싱
                            start_idx = line.find("[") + 1
                            end_idx = line.find("]")
                            numbers_str = line[start_idx:end_idx].strip()
                            numbers = [
                                int(n)
                                for n in numbers_str.split()
                                if n.strip().isdigit()
                            ]
                            if len(numbers) == 6:
                                from src.shared.types import ModelPrediction

                                recommendations.append(
                                    ModelPrediction(
                                        numbers=numbers,
                                        confidence=0.5,  # 기본값
                                        model_type="Unknown",  # 기본값
                                    )
                                )

                # 백테스터 초기화 및 실행
                backtester = Backtester(conf)

                # 중앙 집중식 데이터 사용
                _, _, test_data = get_data_split()

                # 백테스팅 수행
                results = backtester.run(
                    recommendations=recommendations, validation_draws=test_data
                )

                # 결과 변환
                return {
                    "total_investment": results["total"]["investment"],
                    "total_winnings": results["total"]["roi"]
                    * results["total"]["investment"],
                    "roi": results["total"]["roi"],
                    "win_count": results["total"]["wins"],
                    "avg_matches": (
                        results["total"]["matches"] / len(recommendations)
                        if recommendations
                        else 0
                    ),
                }

            # 강화 학습 함수 정의
            def reinforce_with_class(results, conf):
                """클래스 기반 강화 학습 함수"""
                if not results:
                    return False

                # 트레이너 초기화
                from src.models.lstm_model import LSTMModel

                trainer = UnifiedTrainer(
                    model_class=LSTMModel, config=conf, model_type="lstm"  # type: ignore
                )

                # ROI 기반 강화 학습 (reinforce 메서드가 없는 경우 학습 메서드 대신 사용)
                try:
                    # reinforce 메서드 사용 시도 - 메서드가 없을 경우 AttributeError 발생
                    # type: ignore 주석으로 린터 경고 무시
                    success = trainer.reinforce(  # type: ignore
                        roi=results["roi"],
                        win_rate=results["win_count"] / 10,  # 예상 게임 수로 나눔
                        avg_matches=results["avg_matches"],
                    )
                except AttributeError:
                    # reinforce 메서드가 없는 경우 학습 메서드 사용
                    from src.utils.data_loader import DataLoader

                    data_loader = DataLoader(conf)
                    train_data, val_data = data_loader.get_train_val_split()

                    # 학습 설정에 ROI 정보 추가
                    train_config = {
                        "epochs": 5,  # 짧은 튜닝
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "roi_boost": results["roi"],  # ROI 정보 추가
                        "win_rate": results["win_count"] / 10,
                        "avg_matches": results["avg_matches"],
                    }

                    # 모델 재학습
                    success = trainer.train(  # type: ignore
                        train_data=train_data,
                        val_data=val_data,
                        **train_config,  # config= 대신 키워드 인자로 전달
                    )

                return success

            backtester_func = run_backtest_with_class
            reinforcer_func = reinforce_with_class

        except ImportError:
            # 직접 함수 임포트 시도
            from src.evaluation.backtester import run_backtest as run_backtest_external  # type: ignore
            from src.training.train_interface import reinforce_from_backtest as reinforce_from_backtest_external  # type: ignore

            # 함수 래퍼 정의
            def run_backtest_with_func(file_path, conf):
                """함수 기반 백테스트 래퍼"""
                if file_path is None:
                    return None
                return run_backtest_external(file_path, conf)

            def reinforce_with_func(results, conf):
                """함수 기반 강화 학습 래퍼"""
                return reinforce_from_backtest_external(results, conf)

            backtester_func = run_backtest_with_func
            reinforcer_func = reinforce_with_func

        # 백테스트 실행
        print("백테스트 대상을 선택하세요:")
        print("1. 최근 추천 번호")
        print("2. 특정 파일의 추천 번호")
        print("0. 취소")

        choice = input("\n선택: ")

        if choice == "0":
            print("백테스트가 취소되었습니다.")
            return False

        if choice not in ["1", "2"]:
            print("잘못된 선택입니다. 메인 메뉴로 돌아갑니다.")
            return False

        recommendation_file = None

        if choice == "1":
            # 최근 추천 번호 찾기
            recommendation_dir = BASE_DIR / "data" / "recommendations"
            if not recommendation_dir.exists() or not list(
                recommendation_dir.glob("*.txt")
            ):
                print("저장된 추천 번호가 없습니다.")
                return False

            # 가장 최근 파일 찾기
            recommendation_files = sorted(
                recommendation_dir.glob("*.txt"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            recommendation_file = recommendation_files[0]
            print(f"최근 추천 파일: {recommendation_file.name}")

        elif choice == "2":
            # 추천 파일 목록 표시
            recommendation_dir = BASE_DIR / "data" / "recommendations"
            if not recommendation_dir.exists() or not list(
                recommendation_dir.glob("*.txt")
            ):
                print("저장된 추천 번호가 없습니다.")
                return False

            recommendation_files = sorted(
                recommendation_dir.glob("*.txt"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )

            print("\n추천 파일 목록:")
            for i, file in enumerate(recommendation_files):
                file_date = datetime.fromtimestamp(file.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"{i+1}. {file.name} ({file_date})")

            file_choice = input("\n파일 번호를 선택하세요: ")
            try:
                file_index = int(file_choice) - 1
                if 0 <= file_index < len(recommendation_files):
                    recommendation_file = recommendation_files[file_index]
                else:
                    print("잘못된 파일 번호입니다.")
                    return False
            except ValueError:
                print("유효하지 않은 입력입니다.")
                return False

        # recommendation_file이 None인지 확인
        if recommendation_file is None:
            print("추천 파일을 찾을 수 없습니다.")
            return False

        # 백테스트 수행
        print(f"\n{recommendation_file.name}에 대한 백테스트를 수행합니다...")
        backtest_results = backtester_func(recommendation_file, config)

        if not backtest_results:
            print("백테스트 중 문제가 발생했습니다.")
            return False

        # 백테스트 결과 출력
        print("\n===== 백테스트 결과 =====")
        print(f"총 투자: {backtest_results['total_investment']:,}원")
        print(f"총 당첨금: {backtest_results['total_winnings']:,}원")
        print(
            f"ROI: {backtest_results['roi']:.4f} ({(backtest_results['roi'] - 1) * 100:.2f}%)"
        )
        print(f"당첨 횟수: {backtest_results['win_count']}회")
        print(f"매칭 번호 평균: {backtest_results['avg_matches']:.2f}개")
        print("=========================\n")

        # 강화 학습 실행 여부 확인
        reinforce_choice = input(
            "백테스트 결과를 바탕으로 강화 학습을 진행하시겠습니까? (y/n): "
        )

        if reinforce_choice.lower() == "y":
            print("\n강화 학습을 시작합니다...")
            reinforce_result = reinforcer_func(backtest_results, config)

            if reinforce_result:
                print("강화 학습이 성공적으로 완료되었습니다.")
            else:
                print("강화 학습 중 문제가 발생했습니다.")
        else:
            print("강화 학습이 취소되었습니다.")

        return True

    except ImportError as e:
        logger.error(f"모듈 로드 실패: {str(e)}")
        print(f"오류: 필요한 모듈을 로드할 수 없습니다. {str(e)}")
    except Exception as e:
        logger.error(f"백테스트 중 오류 발생: {str(e)}")
        print(f"오류: 백테스트 중 문제가 발생했습니다. {str(e)}")

    input("\n계속하려면 Enter 키를 누르세요...")
    return False


def main():
    """메인 함수"""
    # 중앙 데이터 로드
    load_global_data()

    # 설정 로드
    config = load_config()

    while True:
        clear_screen()
        display_title()
        display_menu()

        choice = input().strip()

        if choice == "1":
            option_1_test(config)
        elif choice == "2":
            option_2_train(config)
        elif choice == "3":
            option_1_predict(config)
        elif choice == "4":
            option_3_backtest(config)
        elif choice == "0":
            print("\n시스템을 종료합니다...")
            break
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")
            time.sleep(2)

    print("DAEBAK_AI 로또 시스템을 이용해 주셔서 감사합니다!")


if __name__ == "__main__":
    sys.exit(main())
