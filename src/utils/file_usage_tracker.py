"""파일 사용 추적 시스템

이 모듈은 시스템 내에서 파일 사용을 추적하고 분석합니다.
"""

import os
import sys
import time
import logging
import inspect
import threading
from pathlib import Path
from typing import Dict, Set, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

# 로거 설정
logger = logging.getLogger(__name__)


@dataclass
class FileUsageInfo:
    """파일 사용 정보"""

    path: str
    count: int = 0
    callers: Set[str] = field(default_factory=set)
    last_access: float = 0.0

    def __post_init__(self):
        self.last_access = time.time()

    def update(self, caller: str):
        """사용 정보 업데이트"""
        self.count += 1
        self.callers.add(caller)
        self.last_access = time.time()


class FileUsageTracker:
    """파일 사용 추적 시스템"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(FileUsageTracker, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """시스템 초기화"""
        self.files: Dict[str, FileUsageInfo] = {}
        self.project_root = Path(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 로그 파일 설정
        self.log_file = self.log_dir / "file_usage.log"
        self._init_log_file()

        # 통계 데이터
        self.stats = defaultdict(int)

        # 스레드 안전성
        self._file_lock = threading.Lock()

    def _init_log_file(self):
        """로그 파일 초기화"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(
                f"파일 사용 추적 로그 - 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 80 + "\n")

    def _get_caller_info(self, frame_back: int = 2) -> str:
        """호출자 정보 추출"""
        frame = inspect.currentframe()
        if frame is None:
            return "unknown"

        for _ in range(frame_back):
            if frame.f_back is None:
                break
            frame = frame.f_back

        module = frame.f_globals.get("__name__", "unknown")
        function = frame.f_code.co_name
        line = frame.f_lineno
        return f"{module}.{function}:{line}"

    def track_file_usage(self, file_path: str, caller_info: Optional[str] = None):
        """파일 사용 추적"""
        try:
            rel_path = os.path.relpath(file_path, self.project_root)
        except ValueError:
            rel_path = file_path

        caller = caller_info or self._get_caller_info()

        with self._file_lock:
            if rel_path not in self.files:
                self.files[rel_path] = FileUsageInfo(rel_path)

            self.files[rel_path].update(caller)
            self.stats["total_accesses"] += 1

            # 로깅
            log_message = f"파일 사용: {rel_path} (호출자: {caller})"
            logger.debug(log_message)

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_message}\n"
                )

    def get_usage_report(
        self, sort_by: str = "count", limit: Optional[int] = None
    ) -> str:
        """사용 보고서 생성"""
        report = [
            "파일 사용 보고서",
            "=" * 80,
            f"총 접근 횟수: {self.stats['total_accesses']}",
            f"추적 중인 파일 수: {len(self.files)}",
            "-" * 80,
        ]

        # 정렬 기준 설정
        if sort_by == "count":
            key_func = lambda x: x[1].count
        elif sort_by == "last_access":
            key_func = lambda x: x[1].last_access
        else:
            key_func = lambda x: x[0]

        # 파일 목록 정렬
        sorted_files = sorted(self.files.items(), key=key_func, reverse=True)
        if limit:
            sorted_files = sorted_files[:limit]

        # 보고서 생성
        for path, info in sorted_files:
            report.extend(
                [
                    f"파일: {path}",
                    f"사용 횟수: {info.count}",
                    f"마지막 접근: {datetime.fromtimestamp(info.last_access).strftime('%Y-%m-%d %H:%M:%S')}",
                    "호출자:",
                ]
            )
            report.extend(f"  - {caller}" for caller in sorted(info.callers))
            report.append("-" * 80)

        return "\n".join(report)

    def save_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
        sort_by: str = "count",
        limit: Optional[int] = None,
    ):
        """보고서 저장"""
        if output_path is None:
            output_path = str(self.log_dir / "file_usage_report.txt")
        else:
            output_path = str(output_path)

        report = self.get_usage_report(sort_by=sort_by, limit=limit)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"파일 사용 보고서가 저장되었습니다: {output_path}")

    def get_most_used_files(self, limit: int = 10) -> List[tuple]:
        """가장 많이 사용된 파일 목록"""
        return sorted(
            [(path, info.count) for path, info in self.files.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

    def clear_stats(self):
        """통계 초기화"""
        with self._file_lock:
            self.files.clear()
            self.stats.clear()
            self._init_log_file()


# 싱글톤 인스턴스
_tracker = FileUsageTracker()


def track_file(file_path: str, caller_info: Optional[str] = None):
    """파일 사용 추적"""
    _tracker.track_file_usage(file_path, caller_info)


def get_report(sort_by: str = "count", limit: Optional[int] = None) -> str:
    """사용 보고서 가져오기"""
    return _tracker.get_usage_report(sort_by=sort_by, limit=limit)


def save_report(
    output_path: Optional[Union[str, Path]] = None,
    sort_by: str = "count",
    limit: Optional[int] = None,
):
    """보고서 저장"""
    _tracker.save_report(output_path, sort_by=sort_by, limit=limit)


def get_most_used_files(limit: int = 10) -> List[tuple]:
    """가장 많이 사용된 파일 목록"""
    return _tracker.get_most_used_files(limit)


def clear_stats():
    """통계 초기화"""
    _tracker.clear_stats()


def track_imports():
    """현재 임포트된 모듈 추적"""
    for name, module in sys.modules.items():
        if hasattr(module, "__file__") and module.__file__:
            if _tracker.project_root.as_posix() in Path(module.__file__).as_posix():
                track_file(module.__file__, f"import in {name}")
