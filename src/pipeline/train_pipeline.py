import json
from pathlib import Path
from typing import List

from src.models.ml.lightgbm_model import LightGBMModel
from src.models.ml.xgboost_model import XGBoostModel
from src.utils.data_loader import load_draw_history
from src.utils.unified_config import load_config
from src.utils.unified_logging import get_logger, log_exception_with_trace
from src.analysis.enhanced_pattern_vectorizer import EnhancedPatternVectorizer

# 로거 설정
logger = get_logger(__name__)


def load_feature_names() -> List[str]:
    """
    캐시에서 특성 이름을 로드합니다.

    Returns:
        특성 이름 목록

    Raises:
        FileNotFoundError: 특성 이름 파일을 찾을 수 없는 경우
        ValueError: 특성 이름을 로드하거나 생성할 수 없는 경우
    """
    # 설정 로드
    try:
        config = load_config()
        cache_dir = Path(config["paths"]["cache_dir"])
    except KeyError as e:
        log_exception_with_trace(
            logger, e, "특성 이름 로드: 캐시 디렉토리 경로 설정 누락"
        )
        raise

    # feature_name_tracker 유틸리티 사용 시도
    try:
        from src.utils.feature_name_tracker import load_feature_names as load_names

        # 새로운 네이밍 규칙으로 파일 확인
        feature_names_file = cache_dir / "feature_vector_full.names.json"
        if feature_names_file.exists():
            feature_names = load_names(str(feature_names_file))
            if feature_names:
                logger.info(f"특성 이름 로드 완료: {len(feature_names)}개 (새 형식)")
                return feature_names

        # 기존 네이밍 규칙으로 파일 확인
        legacy_names_file = cache_dir / "feature_vector_full_feature_names.json"
        if legacy_names_file.exists():
            feature_names = load_names(str(legacy_names_file))
            if feature_names:
                logger.info(f"특성 이름 로드 완료: {len(feature_names)}개 (기존 형식)")
                return feature_names

        # 벡터라이저에서 생성한 특성 이름 파일 확인
        vectorizer_names_file = cache_dir / "feature_names.json"
        if vectorizer_names_file.exists():
            feature_names = load_names(str(vectorizer_names_file))
            if feature_names:
                logger.info(f"벡터라이저 특성 이름 로드 완료: {len(feature_names)}개")
                return feature_names

        logger.warning(
            "특성 이름 파일을 찾을 수 없습니다. 벡터라이저를 통해 생성을 시도합니다."
        )
    except ImportError as e:
        log_exception_with_trace(logger, e, "feature_name_tracker 모듈 임포트 오류")

        # 기존 방식으로 대체
        feature_names_file = cache_dir / "feature_vector_full_feature_names.json"
        if feature_names_file.exists():
            try:
                with open(feature_names_file, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)
                    logger.info(
                        f"특성 이름 로드 완료: {len(feature_names)}개 (기존 방식)"
                    )
                    return feature_names
            except Exception as e:
                log_exception_with_trace(
                    logger, e, f"특성 이름 파일 로드 실패: {feature_names_file}"
                )
                # 계속 진행, 다음 대안을 시도

    # 벡터라이저를 통해 특성 이름 생성 시도
    try:
        vectorizer = EnhancedPatternVectorizer(config)
        feature_names = vectorizer.get_feature_names()

        if not feature_names or len(feature_names) == 0:
            error_msg = "특성 이름을 로드하거나 생성할 수 없습니다."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"벡터라이저에서 특성 이름 생성 완료: {len(feature_names)}개")

        # 특성 이름 저장 시도
        try:
            from src.utils.feature_name_tracker import save_feature_names

            # 새로운 네이밍 규칙으로 저장
            feature_names_file = cache_dir / "feature_names.json"
            save_feature_names(feature_names, str(feature_names_file))
            logger.info(f"생성된 특성 이름 저장 완료: {str(feature_names_file)}")
        except ImportError:
            logger.warning(
                "feature_name_tracker 모듈을 가져올 수 없어 특성 이름을 저장하지 못했습니다."
            )

        return feature_names
    except Exception as e:
        log_exception_with_trace(logger, e, "벡터라이저를 통한 특성 이름 생성 실패")
        raise ValueError(f"특성 이름 생성 실패: {str(e)}")


def train_lightgbm_model() -> bool:
    """
    LightGBM 모델 학습 및 저장 함수

    Returns:
        bool: 학습 성공 여부
    """
    try:
        # 설정 로드
        try:
            config = load_config()
        except Exception as e:
            log_exception_with_trace(logger, e, "LightGBM 모델 학습: 설정 로드 실패")
            return False

        # 저분산 필터링된 벡터 사용 여부 로깅
        use_filtered_vector = config.get("training", {}).get(
            "use_filtered_vector", True
        )
        logger.info(f"LightGBM 모델 학습: 필터링된 벡터 사용: {use_filtered_vector}")

        logger.info("LightGBM 모델 학습 시작")

        # 로또 당첨 번호 데이터 로드
        try:
            logger.info("로또 당첨 번호 데이터 로드 중...")
            draw_data = load_draw_history()
            logger.info(f"로또 당첨 번호 데이터 로드 완료: {len(draw_data)}개")
        except Exception as e:
            log_exception_with_trace(
                logger, e, "LightGBM 모델 학습: 로또 당첨 번호 데이터 로드 실패"
            )
            return False

        # 모델 저장 디렉토리 확인 및 생성
        try:
            model_dir = Path(config["paths"]["model_save_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
        except KeyError as e:
            log_exception_with_trace(
                logger, e, "LightGBM 모델 학습: 모델 저장 디렉토리 설정 누락"
            )
            return False
        except Exception as e:
            log_exception_with_trace(
                logger, e, "LightGBM 모델 학습: 모델 저장 디렉토리 생성 실패"
            )
            return False

        # 모델 인스턴스 생성 및 학습
        logger.info("LightGBM 모델 학습 중...")
        try:
            model = LightGBMModel(config)
        except Exception as e:
            log_exception_with_trace(logger, e, "LightGBM 모델 인스턴스 생성 실패")
            return False

        try:
            result = model.train(draw_data)  # 학습 실패 시 예외 발생해야 함

            # 학습 결과 요약 - 직접 인덱싱 접근
            logger.info(f"학습 결과: 성공 (학습 시간: {result['training_time']:.2f}초)")
            logger.info(
                f"샘플 수: {result['n_samples']}, 특성 수: {result['feature_count']}"
            )
        except Exception as e:
            log_exception_with_trace(logger, e, "LightGBM 모델 학습 중 오류")
            return False

        # 모델 저장
        try:
            model_path = model_dir / "lightgbm_model.pkl"
            model.save(str(model_path))
            logger.info(f"LightGBM 모델 저장 완료: {model_path}")
        except Exception as e:
            log_exception_with_trace(logger, e, "LightGBM 모델 저장 실패")
            return False

        # 특성 이름 로드
        try:
            feature_names = load_feature_names()
        except Exception as e:
            log_exception_with_trace(logger, e, "LightGBM 특성 이름 로드 실패")
            feature_names = [f"feature_{i}" for i in range(model.feature_count)]
            logger.warning("특성 이름 로드 실패로 기본 이름 사용")

        # 특성 중요도 명시적 저장 (고정 파일명)
        try:
            feature_importance_path = model_dir / "lgbm_feature_importance_latest.json"
            model.save_feature_importance(str(feature_importance_path), feature_names)

            # 사람이 읽을 수 있는 이름의 특성 중요도 파일 추가 저장
            named_importance_path = model_dir / "lgbm_feature_importance_named.json"
            model.save_feature_importance(str(named_importance_path), feature_names)

            logger.info(f"LightGBM 특성 중요도 저장 완료: {feature_importance_path}")
            logger.info(
                f"LightGBM 특성 중요도(명명) 저장 완료: {named_importance_path}"
            )
        except Exception as e:
            log_exception_with_trace(logger, e, "LightGBM 특성 중요도 저장 실패")
            # 특성 중요도 저장 실패는 학습 성공 여부에 영향을 주지 않음

        logger.info(f"LightGBM 모델 학습 및 저장 완료. 모델 파일: {model_path}")
        return True

    except Exception as e:
        log_exception_with_trace(
            logger, e, "LightGBM 모델 학습 및 저장 중 예상치 못한 오류"
        )
        return False


def train_xgboost_model() -> bool:
    """
    XGBoost 모델 학습 및 저장 함수

    Returns:
        bool: 학습 성공 여부
    """
    try:
        # 설정 로드
        try:
            config = load_config()
        except Exception as e:
            log_exception_with_trace(logger, e, "XGBoost 모델 학습: 설정 로드 실패")
            return False

        # 저분산 필터링된 벡터 사용 여부 로깅
        use_filtered_vector = config.get("training", {}).get(
            "use_filtered_vector", True
        )
        logger.info(f"XGBoost 모델 학습: 필터링된 벡터 사용: {use_filtered_vector}")

        logger.info("XGBoost 모델 학습 시작")

        # 로또 당첨 번호 데이터 로드
        try:
            logger.info("로또 당첨 번호 데이터 로드 중...")
            draw_data = load_draw_history()
            logger.info(f"로또 당첨 번호 데이터 로드 완료: {len(draw_data)}개")
        except Exception as e:
            log_exception_with_trace(
                logger, e, "XGBoost 모델 학습: 로또 당첨 번호 데이터 로드 실패"
            )
            return False

        # 모델 저장 디렉토리 확인 및 생성
        try:
            model_dir = Path(config["paths"]["model_save_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
        except KeyError as e:
            log_exception_with_trace(
                logger, e, "XGBoost 모델 학습: 모델 저장 디렉토리 설정 누락"
            )
            return False
        except Exception as e:
            log_exception_with_trace(
                logger, e, "XGBoost 모델 학습: 모델 저장 디렉토리 생성 실패"
            )
            return False

        # 모델 인스턴스 생성 및 학습
        logger.info("XGBoost 모델 학습 중...")
        try:
            model = XGBoostModel(config)
        except Exception as e:
            log_exception_with_trace(logger, e, "XGBoost 모델 인스턴스 생성 실패")
            return False

        try:
            result = model.train(draw_data)  # 학습 실패 시 예외 발생해야 함

            # 학습 결과 요약 - 직접 인덱싱 접근
            logger.info(f"학습 결과: 성공 (학습 시간: {result['training_time']:.2f}초)")
            logger.info(
                f"샘플 수: {result['n_samples']}, 특성 수: {result['feature_count']}"
            )
        except Exception as e:
            log_exception_with_trace(logger, e, "XGBoost 모델 학습 중 오류")
            return False

        # 모델 저장
        try:
            model_path = model_dir / "xgboost_model.pkl"
            model.save(str(model_path))
            logger.info(f"XGBoost 모델 저장 완료: {model_path}")
        except Exception as e:
            log_exception_with_trace(logger, e, "XGBoost 모델 저장 실패")
            return False

        # 특성 이름 로드
        try:
            feature_names = load_feature_names()
        except Exception as e:
            log_exception_with_trace(logger, e, "XGBoost 특성 이름 로드 실패")
            feature_names = [f"feature_{i}" for i in range(model.feature_count)]
            logger.warning("특성 이름 로드 실패로 기본 이름 사용")

        # 특성 중요도 명시적 저장 (고정 파일명)
        try:
            feature_importance_path = model_dir / "xgb_feature_importance_latest.json"
            model.save_feature_importance(str(feature_importance_path), feature_names)

            # 사람이 읽을 수 있는 이름의 특성 중요도 파일 추가 저장
            named_importance_path = model_dir / "xgb_feature_importance_named.json"
            model.save_feature_importance(str(named_importance_path), feature_names)

            logger.info(f"XGBoost 특성 중요도 저장 완료: {feature_importance_path}")
            logger.info(f"XGBoost 특성 중요도(명명) 저장 완료: {named_importance_path}")
        except Exception as e:
            log_exception_with_trace(logger, e, "XGBoost 특성 중요도 저장 실패")
            # 특성 중요도 저장 실패는 학습 성공 여부에 영향을 주지 않음

        logger.info(f"XGBoost 모델 학습 및 저장 완료. 모델 파일: {model_path}")
        return True

    except Exception as e:
        log_exception_with_trace(
            logger, e, "XGBoost 모델 학습 및 저장 중 예상치 못한 오류"
        )
        return False
