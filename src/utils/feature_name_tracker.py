"""
특성 이름 관리 유틸리티

이 모듈은 머신러닝/딥러닝 모델에서 사용되는 특성 이름을 일관되게 관리하기 위한
유틸리티 함수를 제공합니다.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .error_handler_refactored import get_logger
from .cache_paths import CACHE_DIR

# 로거 설정
logger = get_logger(__name__)

# 특성 추적 강제 활성화 (설정 파일 무시)
FEATURE_TRACKING_ENABLED = True

# 특성 이름 저장 경로 (상대경로 사용)
FEATURE_NAMES_DIR = CACHE_DIR / "feature_names"


def save_feature_names(names: List[str], path: str) -> None:
    """
    특성 이름을 JSON 파일로 저장합니다.

    Args:
        names: 저장할 특성 이름 목록
        path: 저장할 파일 경로

    Raises:
        IOError: 파일 저장 실패 시
        TypeError: 특성 이름이 유효하지 않은 경우
    """
    # 경로 객체 생성
    file_path = Path(path)

    # 디렉토리 생성
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 특성 이름 저장
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(names, f, indent=2, ensure_ascii=False)
        logger.info(f"특성 이름 {len(names)}개를 {path}에 저장했습니다.")
    except Exception as e:
        logger.error(f"특성 이름 저장 실패: {str(e)}")
        raise IOError(f"특성 이름 저장 실패: {str(e)}")


def load_feature_names(path: str) -> List[str]:
    """
    JSON 파일에서 특성 이름을 로드합니다.

    Args:
        path: 로드할 파일 경로

    Returns:
        특성 이름 목록

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        ValueError: 파일 형식이 올바르지 않을 때
        IOError: 파일 읽기에 실패했을 때
    """
    # 파일 존재 확인
    file_path = Path(path)
    if not file_path.exists():
        logger.error(f"특성 이름 파일이 존재하지 않습니다: {path}")
        raise FileNotFoundError(f"특성 이름 파일이 존재하지 않습니다: {path}")

    # 특성 이름 로드
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            names = json.load(f)

        if not isinstance(names, list):
            logger.error(f"특성 이름 파일이 리스트 형식이 아닙니다: {path}")
            raise ValueError(f"특성 이름 파일이 리스트 형식이 아닙니다: {path}")

        logger.info(f"{path}에서 특성 이름 {len(names)}개를 로드했습니다.")
        return names
    except json.JSONDecodeError as e:
        logger.error(f"특성 이름 파일 형식이 올바르지 않습니다: {path}")
        raise ValueError(f"특성 이름 파일 형식이 올바르지 않습니다: {path} - {str(e)}")
    except Exception as e:
        logger.error(f"특성 이름 로드 실패: {str(e)}")
        raise IOError(f"특성 이름 로드 실패: {str(e)}")


def save_feature_index_mapping(feature_names: List[str], path: str) -> None:
    """
    특성 이름과 인덱스 매핑을 JSON 파일로 저장합니다.

    Args:
        feature_names: 저장할 특성 이름 목록
        path: 저장할 파일 경로

    Raises:
        IOError: 파일 저장 실패 시
        TypeError: 특성 이름이 유효하지 않은 경우
    """
    # 경로 객체 생성
    file_path = Path(path)

    # 디렉토리 생성
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 특성 이름-인덱스 매핑 생성
    index_mapping = {f"index_{i}": name for i, name in enumerate(feature_names)}

    # 매핑 저장
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(index_mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"특성 인덱스 매핑 저장 완료: {file_path}")
    except Exception as e:
        logger.error(f"특성 인덱스 매핑 저장 실패: {str(e)}")
        raise IOError(f"특성 인덱스 매핑 저장 실패: {str(e)}")


def get_vector_name_path(vector_path: str) -> str:
    """
    벡터 파일의 경로에서 이름 파일 경로를 생성합니다.

    Args:
        vector_path: 벡터 파일 경로

    Returns:
        이름 파일 경로
    """
    # 경로 객체 생성
    file_path = Path(vector_path)

    # 확장자 제거 후 .names.json 추가
    name_path = file_path.with_suffix("").with_suffix(".names.json")

    return str(name_path)


def is_feature_tracking_enabled(
    config: Optional[Union[Dict[str, Any], Any]] = None,
) -> bool:
    """
    특성 이름 추적이 활성화되어 있는지 확인합니다.
    이 시스템에서는 항상 활성화되어 있어야 합니다.

    Args:
        config: 설정 객체 (무시됨)

    Returns:
        True (항상 활성화)
    """
    # 설정과 관계없이 항상 True 반환
    return FEATURE_TRACKING_ENABLED
