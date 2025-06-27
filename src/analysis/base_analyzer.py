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
import logging
from datetime import datetime
from abc import ABC, abstractmethod

from ..utils.unified_performance import performance_monitor
from ..utils.cache_manager import CacheManager
from ..shared.types import LotteryNumber
from ..utils.error_handler_refactored import get_logger
from ..utils.unified_config import ConfigProxy

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
        self,
        config: Optional[Union[Dict[str, Any], ConfigProxy]] = None,
        analyzer_type: str = "base",
    ):
        """
        BaseAnalyzer 초기화

        Args:
            config: 분석에 사용할 설정
            analyzer_type: 분석기 유형(예: "pattern", "distribution", "roi", "unified")
        """
        # 설정 초기화
        self.config = ConfigProxy(config or {})
        self.analyzer_type = analyzer_type
        # 통합 성능 모니터링 사용
        self.logger = get_logger(f"{__name__}.{analyzer_type}")

        # 캐시 디렉토리 설정
        try:
            cache_dir = self.config["paths"]["cache_dir"]
            # 특정 분석기용 하위 디렉토리 생성
            cache_dir = Path(cache_dir) / analyzer_type
        except (KeyError, TypeError):
            logger.warning(
                f"설정에서 캐시 디렉토리를 찾을 수 없습니다. 기본값('data/cache/{analyzer_type}')을 사용합니다."
            )
            cache_dir = (
                Path(__file__).parent.parent.parent / "data" / "cache" / analyzer_type
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir

        # 캐시 설정 가져오기
        try:
            max_cache_size = self.config["caching"]["max_cache_size"]
        except (KeyError, TypeError):
            max_cache_size = 1000  # 기본값

        try:
            enable_compression = self.config["caching"]["enable_compression"]
        except (KeyError, TypeError):
            enable_compression = True  # 기본값

        try:
            cache_duration = self.config["caching"]["cache_duration"]
        except (KeyError, TypeError):
            cache_duration = 86400  # 기본값 (1일)

        # 캐시 관리자 초기화
        self.cache_manager = CacheManager(
            pattern_analyzer=self,
            max_memory_size=max_cache_size * 1024,
            max_disk_size=max_cache_size * 1024 * 10,
            cache_dir=str(cache_dir),
            enable_compression=enable_compression,
            default_ttl=cache_duration,
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
        with performance_monitor(f"{self.analyzer_type}_analysis"):
            try:
                # 캐시 키 생성
                cache_key = self._create_cache_key(
                    f"{self.analyzer_type}_analysis", len(historical_data), *args
                )

                # 캐시 확인
                cached_result = self._check_cache(cache_key)
                if cached_result:
                    self.logger.info(f"캐시된 분석 결과 사용: {cache_key}")
                    return cached_result

                # 실제 분석 수행 (하위 클래스에서 구현)
                self.logger.info(
                    f"{self.analyzer_type} 분석 시작: {len(historical_data)}개 데이터"
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
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            # 파일 캐시 확인
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        cached_result = pickle.load(f)
                        self.logger.info(f"파일 캐시 사용: {cache_file}")
                        # 메모리 캐시에도 저장
                        self.cache_manager.set(cache_key, cached_result)
                        return cached_result
                except Exception as e:
                    self.logger.warning(f"캐시 파일 로드 실패: {e}")
        except Exception as e:
            self.logger.warning(f"캐시 데이터 액세스 오류: {e}")
            # 오류 발생 시 캐시 무시

        return None

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
            self.cache_manager.set(cache_key, result)

            # 파일 캐시에 저장
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

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
            "analyzer_type": self.analyzer_type,
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
