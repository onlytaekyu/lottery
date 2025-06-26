"""
스레드 로컬 저장소 모듈

이 모듈은 스레드별 데이터 저장을 위한 유틸리티를 제공합니다.
"""

import threading
from typing import Dict, Any, Optional, TypeVar, Generic

T = TypeVar("T")


class ThreadLocalStorage(Generic[T]):
    """스레드별 데이터 저장소"""

    def __init__(self):
        """초기화"""
        self._local = threading.local()
        self._default_factory = None

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        키에 해당하는 값 조회

        Args:
            key: 조회할 키
            default: 키가 없을 경우 반환할 기본값

        Returns:
            저장된 값 또는 기본값
        """
        if not hasattr(self._local, "storage"):
            self._local.storage = {}
        return self._local.storage.get(key, default)

    def set(self, key: str, value: T) -> None:
        """
        값 설정

        Args:
            key: 설정할 키
            value: 저장할 값
        """
        if not hasattr(self._local, "storage"):
            self._local.storage = {}
        self._local.storage[key] = value

    def delete(self, key: str) -> None:
        """
        키 삭제

        Args:
            key: 삭제할 키
        """
        if hasattr(self._local, "storage") and key in self._local.storage:
            del self._local.storage[key]

    def clear(self) -> None:
        """모든 데이터 삭제"""
        if hasattr(self._local, "storage"):
            self._local.storage.clear()

    def get_all(self) -> Dict[str, Any]:
        """
        모든 데이터 조회

        Returns:
            모든 키-값 쌍
        """
        if not hasattr(self._local, "storage"):
            self._local.storage = {}
        return dict(self._local.storage)

    def set_default_factory(self, factory):
        """
        기본값 생성 함수 설정

        Args:
            factory: 기본값 생성 함수
        """
        self._default_factory = factory

    def __getitem__(self, key: str) -> T:
        """
        키에 해당하는 값 조회 ([] 연산자)

        Args:
            key: 조회할 키

        Returns:
            저장된 값

        Raises:
            KeyError: 키가 없고 기본값 생성 함수도 없는 경우
        """
        if not hasattr(self._local, "storage"):
            self._local.storage = {}

        if key not in self._local.storage:
            if self._default_factory is not None:
                self._local.storage[key] = self._default_factory()
            else:
                raise KeyError(key)

        return self._local.storage[key]

    def __setitem__(self, key: str, value: T) -> None:
        """
        값 설정 ([] 연산자)

        Args:
            key: 설정할 키
            value: 저장할 값
        """
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """
        키 삭제 (del 연산자)

        Args:
            key: 삭제할 키

        Raises:
            KeyError: 키가 없는 경우
        """
        if not hasattr(self._local, "storage") or key not in self._local.storage:
            raise KeyError(key)
        del self._local.storage[key]

    def __contains__(self, key: str) -> bool:
        """
        키 존재 여부 확인 (in 연산자)

        Args:
            key: 확인할 키

        Returns:
            키 존재 여부
        """
        return hasattr(self._local, "storage") and key in self._local.storage
