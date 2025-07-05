"""
중앙집중식 싱글톤 인스턴스 팩토리
"""

import threading
from typing import Dict, Any, Type, TypeVar

T = TypeVar("T")
_instances: Dict[Type, Any] = {}
_lock = threading.Lock()


def get_singleton_instance(cls: Type[T], *args, **kwargs) -> T:
    """
    클래스 타입을 키로 사용하여 싱글톤 인스턴스를 생성하고 반환합니다.
    이미 인스턴스가 존재하면 그것을 반환하고, 없으면 새로 생성합니다.
    생성자에 인자를 전달할 수 있습니다.
    """
    if cls not in _instances:
        with _lock:
            if cls not in _instances:
                _instances[cls] = cls(*args, **kwargs)
    return _instances[cls]
