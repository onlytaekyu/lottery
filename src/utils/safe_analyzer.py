"""
안전한 분석 래퍼 유틸리티

이 모듈은 분석기 객체를 안전하게 사용할 수 있게 해주는 래퍼 함수를 제공합니다.
None 객체에 메서드 호출을 방지하고, 예외를 처리하여 안정성을 높입니다.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Callable
from .error_handler import get_logger

# 로거 설정
logger = get_logger(__name__)

T = TypeVar("T")
U = TypeVar("U")


def safe_analyze(analyzer: Optional[Any], data: Any, *args, **kwargs) -> Optional[Any]:
    """
    분석기의 analyze 메서드를 안전하게 호출

    Args:
        analyzer: 분석기 객체 (None일 수 있음)
        data: 분석할 데이터
        *args: 추가 인자
        **kwargs: 추가 키워드 인자

    Returns:
        분석 결과 또는 None (오류 시)
    """
    if analyzer is None:
        logger.debug("분석기가 초기화되지 않았습니다.")
        return None

    try:
        # analyze 메서드 존재 여부 확인
        if hasattr(analyzer, "analyze") and callable(getattr(analyzer, "analyze")):
            return analyzer.analyze(data, *args, **kwargs)
        # process 메서드 시도 (일부 분석기는 다른 메서드명 사용)
        elif hasattr(analyzer, "process") and callable(getattr(analyzer, "process")):
            return analyzer.process(data, *args, **kwargs)
        # run 메서드 시도
        elif hasattr(analyzer, "run") and callable(getattr(analyzer, "run")):
            return analyzer.run(data, *args, **kwargs)
        else:
            logger.debug(f"{type(analyzer).__name__} 객체에 분석 메서드가 없습니다.")
            return None
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        return None


def safe_get(data: Optional[Dict[Any, T]], key: Any, default: U = None) -> Union[T, U]:
    """
    사전에서 키 값을 안전하게 가져오기

    사전이 None이거나 접근 중 예외가 발생하는 경우를 처리합니다.

    Args:
        data: 사전 (None일 수 있음)
        key: 검색할 키
        default: 기본값 (기본값: None)

    Returns:
        키에 해당하는 값 또는 기본값
    """
    if data is None:
        return default

    try:
        return data.get(key, default)
    except Exception as e:
        logger.debug(f"키 접근 중 오류 발생: {str(e)}")
        return default


def str_to_int(value: Any) -> Any:
    """
    문자열을 정수로 안전하게 변환

    변환할 수 없는 경우 원래 값을 반환합니다.

    Args:
        value: 변환할 값

    Returns:
        변환된 정수 값 또는 원본 값
    """
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    return value


def safe_cluster_lookup(
    clusters: Optional[Dict[Any, Any]], number: Any
) -> Optional[Any]:
    """
    클러스터 맵에서 번호에 해당하는 클러스터를 안전하게 찾기

    여러 형태의 키로 시도합니다 (정수, 문자열 등).

    Args:
        clusters: 클러스터 맵 (None일 수 있음)
        number: 찾을 번호

    Returns:
        클러스터 ID 또는 None
    """
    if clusters is None:
        return None

    # 다양한 키 유형 시도
    try:
        # 정수로 직접 접근
        if number in clusters:
            return clusters[number]

        # 문자열로 변환 후 접근
        if isinstance(number, int) and str(number) in clusters:
            return clusters[str(number)]

        # 정수로 변환 후 접근
        if isinstance(number, str):
            try:
                int_key = int(number)
                if int_key in clusters:
                    return clusters[int_key]
            except (ValueError, TypeError):
                pass

        return None
    except Exception as e:
        logger.debug(f"클러스터 조회 중 오류 발생: {str(e)}")
        return None


def safe_call(func: Optional[Callable], *args, **kwargs) -> Optional[Any]:
    """
    함수를 안전하게 호출

    함수가 None이거나 호출 중 예외가 발생하는 경우를 처리합니다.

    Args:
        func: 호출할 함수 (None일 수 있음)
        *args: 함수에 전달할 위치 인자
        **kwargs: 함수에 전달할 키워드 인자

    Returns:
        함수 호출 결과 또는 None (오류 시)
    """
    if func is None or not callable(func):
        return None

    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"함수 호출 중 오류 발생: {str(e)}")
        return None
