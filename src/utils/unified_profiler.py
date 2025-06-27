"""
통합 프로파일러

성능 측정 및 프로파일링을 위한 통합 모듈입니다.
"""

import time
from typing import Dict, Any, Optional
from .error_handler_refactored import get_logger

logger = get_logger(__name__)


class Profiler:
    """성능 프로파일러"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.timers = {}
        self.start_times = {}

    def start(self, name: str):
        """타이머 시작"""
        self.start_times[name] = time.time()

    def stop(self, name: str):
        """타이머 종료"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.timers[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0.0

    def get_times(self) -> Dict[str, float]:
        """모든 타이머 결과 반환"""
        return self.timers.copy()

    def reset(self):
        """프로파일러 초기화"""
        self.timers.clear()
        self.start_times.clear()
