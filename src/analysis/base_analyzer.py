"""
기본 분석기 모듈

이 모듈은 모든 분석기 클래스가 상속받는 기본 클래스를 정의합니다.
공통 기능인 캐싱, 성능 추적 등의 기능을 제공합니다.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Generic, TypeVar, Union, Tuple
import json
import time
import pickle
import os
from datetime import datetime
from abc import ABC, abstractmethod

from ..utils.unified_performance import performance_monitor
from ..utils.cache_manager import CacheManager
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from ..utils.unified_performance import Profiler
from ..utils.unified_performance import PerformanceTracker

# 제네릭 타입 변수 정의
T = TypeVar("T")

logger = get_logger(__name__)


class BaseAnalyzer(Generic[T], ABC):
    """
    모든 분석기 클래스의 기본 클래스

    이 클래스는 다음 공통 기능을 제공합니다:
    - 캐시 관리
    - 성능 추적
    - 기본 분석 인터페이스
    - 직렬화 및 역직렬화
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, name: str = "BaseAnalyzer"
    ):
        """
        기본 분석기 초기화

        Args:
            config: 분석기 설정 딕셔너리
            name: 분석기 이름
        """
        self.name = name
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{name}")

        # 성능 측정 도구 초기화
        self.profiler = Profiler()
        self.performance_tracker = PerformanceTracker()

        # 분석 결과 캐시
        self._cache = {}

        # 중복 로그 방지를 위한 전역 카운터 기반 추적
        import threading

        class_name = self.__class__.__name__
        initialization_key = f"{class_name}_{name}"

        # 전역 초기화 카운터 (스레드 안전)
        if not hasattr(BaseAnalyzer, "_initialization_count"):
            BaseAnalyzer._initialization_count = {}
            BaseAnalyzer._log_lock = threading.Lock()

        # 스레드 안전한 로그 중복 방지
        with BaseAnalyzer._log_lock:
            count = BaseAnalyzer._initialization_count.get(initialization_key, 0)
            BaseAnalyzer._initialization_count[initialization_key] = count + 1

            # 첫 번째 초기화만 INFO 레벨로 로그, 이후는 DEBUG 레벨
            if count == 0:
                self.logger.info(f"{name} 분석기 초기화 완료")
            else:
                self.logger.debug(
                    f"{name} 분석기 재초기화 #{count + 1} (중복 로그 방지)"
                )

    def analyze(self, historical_data: List[LotteryNumber], *args, **kwargs) -> T:
        """
        과거 로또 당첨 번호를 분석하는 메서드

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 분석 매개변수

        Returns:
            T: 분석 결과
        """
        with performance_monitor(f"{self.name}_analysis"):
            try:
                # 캐시 키 생성
                cache_key = self._create_cache_key(
                    f"{self.name}_analysis", len(historical_data), *args
                )

                # 캐시 확인
                cached_result = self._check_cache(cache_key)
                if cached_result:
                    self.logger.debug(f"캐시된 분석 결과 사용: {cache_key}")
                    return cached_result

                # 실제 분석 수행 (하위 클래스에서 구현)
                self.logger.info(
                    f"{self.name} 분석 시작: {len(historical_data)}개 데이터"
                )
                result = self._analyze_impl(historical_data, *args, **kwargs)

                # 결과 캐싱
                self._save_to_cache(cache_key, result)

                return result
            except Exception as e:
                self.logger.error(f"분석 중 오류 발생: {str(e)}")
                # 오류 발생 시 예외 다시 발생
                raise

    @abstractmethod
    def _analyze_impl(self, historical_data: List[LotteryNumber], *args, **kwargs) -> T:
        """
        실제 분석을 구현하는 내부 메서드 (하위 클래스에서 반드시 구현해야 함)

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            *args, **kwargs: 추가 분석 매개변수

        Returns:
            T: 분석 결과
        """
        pass

    def _check_cache(self, cache_key: str) -> Optional[T]:
        """
        캐시된 분석 결과를 확인하고 반환합니다.

        Args:
            cache_key: 캐시 키

        Returns:
            Optional[T]: 캐시된 결과 또는 None
        """
        try:
            # 메모리 캐시 확인
            cached_result = self._cache.get(cache_key)
            if cached_result:
                return cached_result

            # 파일 캐시 확인
            cache_dir = Path(
                self.config.get("paths", {}).get("cache_dir", "data/cache")
            )
            cache_file = cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        cached_result = pickle.load(f)
                        self.logger.info(f"파일 캐시 사용: {cache_file}")
                        # 메모리 캐시에도 저장
                        self._cache[cache_key] = cached_result
                        return cached_result
                except Exception as e:
                    self.logger.warning(f"캐시 파일 로드 실패: {e}")
        except Exception as e:
            self.logger.warning(f"캐시 데이터 액세스 오류: {e}")
            # 오류 발생 시 캐시 무시

        return None

    def _make_serializable(self, obj: Any) -> Any:
        """
        객체를 pickle 직렬화 가능한 형태로 변환합니다.

        Args:
            obj: 직렬화할 객체

        Returns:
            Any: 직렬화 가능한 객체
        """
        import types
        from contextlib import ContextDecorator

        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._make_serializable(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, set):
            return {self._make_serializable(item) for item in obj}
        elif hasattr(obj, "__dict__") and not isinstance(
            obj, (types.FunctionType, types.MethodType, ContextDecorator)
        ):
            # 일반 객체는 딕셔너리로 변환
            try:
                return {
                    "_class_name": obj.__class__.__name__,
                    "_module": obj.__class__.__module__,
                    **{
                        k: self._make_serializable(v)
                        for k, v in obj.__dict__.items()
                        if not k.startswith("_") and not callable(v)
                    },
                }
            except Exception:
                return str(obj)
        elif hasattr(obj, "to_dict") and callable(obj.to_dict):
            # to_dict 메서드가 있는 객체
            try:
                return self._make_serializable(obj.to_dict())
            except Exception:
                return str(obj)
        elif isinstance(obj, (types.FunctionType, types.MethodType, ContextDecorator)):
            # 함수나 메서드, ContextDecorator는 문자열로 변환
            return f"<{type(obj).__name__}: {getattr(obj, '__name__', str(obj))}>"
        else:
            # 기타 직렬화 불가능한 객체는 문자열로 변환
            try:
                # 간단한 직렬화 테스트
                import pickle

                pickle.dumps(obj)
                return obj
            except Exception:
                return str(obj)

    def _save_to_cache(self, cache_key: str, result: T) -> bool:
        """
        분석 결과를 캐시에 저장합니다.

        Args:
            cache_key: 캐시 키
            result: 저장할 분석 결과

        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 메모리 캐시에 저장
            self._cache[cache_key] = result

            # 파일 캐시에 저장 (직렬화 가능한 데이터만)
            cache_dir = Path(
                self.config.get("paths", {}).get("cache_dir", "data/cache")
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{cache_key}.pkl"

            # 직렬화 가능한 데이터로 변환
            serializable_result = self._make_serializable(result)

            with open(cache_file, "wb") as f:
                pickle.dump(serializable_result, f)

            self.logger.info(f"분석 결과 캐시 저장 완료: {cache_key}")
            return True
        except Exception as e:
            self.logger.warning(f"캐시 저장 실패: {e}")
            return False

    def _create_cache_key(self, base_key: str, data_length: int, *args) -> str:
        """
        고유한 캐시 키를 생성합니다.

        Args:
            base_key: 기본 캐시 키
            data_length: 데이터 길이
            *args: 캐시 키 구성에 사용될 추가 인자

        Returns:
            str: 생성된 캐시 키
        """
        key_parts = [base_key, str(data_length)]
        key_parts.extend(str(arg) for arg in args)
        return "_".join(key_parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        객체를 사전 형태로 직렬화합니다.
        하위 클래스에서 필요에 따라 오버라이드할 수 있습니다.

        Returns:
            Dict[str, Any]: 직렬화된 객체
        """
        return {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> "BaseAnalyzer":
        """
        사전에서 객체를 복원합니다.
        하위 클래스에서 필요에 따라 오버라이드해야 합니다.

        Args:
            data: 직렬화된 객체 데이터
            config: 설정 객체

        Returns:
            BaseAnalyzer: 복원된 객체
        """
        # 기본 구현은 추상 클래스라 하위 클래스에서 오버라이드해야 함
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")

    def get_performance_stats(self) -> Dict[str, float]:
        """
        성능 통계를 반환합니다.

        Returns:
            Dict[str, float]: 성능 통계
        """
        return self.performance_tracker.get_stats()

    def run_analysis_with_caching(
        self,
        key_base: str,
        historical_data: List[LotteryNumber],
        analysis_func,
        *args,
        **kwargs,
    ) -> Any:
        """
        캐싱을 적용하여 분석 함수를 실행합니다.

        Args:
            key_base: 캐시 키 기본값
            historical_data: 분석할 과거 당첨 번호 목록
            analysis_func: 실행할 분석 함수
            *args, **kwargs: 추가 분석 매개변수

        Returns:
            분석 결과
        """
        # 캐시 키 생성
        cache_key = self._create_cache_key(key_base, len(historical_data), *args)

        # 캐시 확인
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.logger.info(f"캐시된 분석 결과 사용: {cache_key}")
            return cached_result

        # 분석 함수 실행
        with performance_monitor(key_base):
            result = analysis_func(historical_data, *args, **kwargs)

        # 결과 캐싱
        self._save_to_cache(cache_key, result)

        return result
