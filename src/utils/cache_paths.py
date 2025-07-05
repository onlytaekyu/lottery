"""
캐시 경로 관리 유틸리티

모든 캐시 파일을 중앙집중화된 위치에 저장하기 위한 모듈입니다.
프로젝트 전체에서 일관된 캐시 경로를 사용하도록 합니다.
"""

from pathlib import Path
from typing import Optional
import threading
import shutil

from .unified_logging import get_logger
from .factory import get_singleton_instance

# 로거 설정
logger = get_logger(__name__)

# 프로젝트 루트 디렉토리 - 상대 경로 사용
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 중앙 캐시 디렉토리 - .cache에서 cache로 변경
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# 하위 캐시 디렉토리
PATTERN_CACHE_DIR = CACHE_DIR / "patterns"
ROI_CACHE_DIR = CACHE_DIR / "roi"
FILTER_CACHE_DIR = CACHE_DIR / "filters"
BACKTESTING_CACHE_DIR = CACHE_DIR / "backtesting"
MODEL_CACHE_DIR = CACHE_DIR / "models"
TENSORRT_CACHE_DIR = CACHE_DIR / "tensorrt"
ONNX_CACHE_DIR = CACHE_DIR / "onnx"
CALIBRATION_CACHE_DIR = CACHE_DIR / "calibration"
TRAINER_CACHE_DIR = CACHE_DIR / "trainers"
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"
CONFIG_CACHE_DIR = CACHE_DIR / "config"
REPORT_CACHE_DIR = CACHE_DIR / "reports"
TEMP_CACHE_DIR = CACHE_DIR / "temp"

# 초기화 상태 추적
_initialized = False
_init_lock = threading.Lock()


def _ensure_cache_dirs_initialized():
    """캐시 디렉토리가 초기화되었는지 확인하고 필요시 초기화합니다."""
    global _initialized
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return
        init_cache_dirs()
        _initialized = True


def get_cache_dir(category: Optional[str] = None) -> Path:
    """
    특정 카테고리의 캐시 디렉토리 경로를 반환합니다.

    Args:
        category: 캐시 카테고리 (기본값: None)

    Returns:
        캐시 디렉토리 경로
    """
    _ensure_cache_dirs_initialized()

    if category is None:
        return CACHE_DIR

    cache_dir = CACHE_DIR / category
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def init_cache_dirs():
    """모든 캐시 디렉토리를 초기화합니다."""
    try:
        # 이전 .cache 디렉토리 마이그레이션 체크
        old_cache_dir = PROJECT_ROOT / "data" / ".cache"

        # 새 캐시 디렉토리 생성
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 모든 하위 캐시 디렉토리 생성
        cache_subdirs = [
            PATTERN_CACHE_DIR,
            ROI_CACHE_DIR,
            FILTER_CACHE_DIR,
            BACKTESTING_CACHE_DIR,
            MODEL_CACHE_DIR,
            TENSORRT_CACHE_DIR,
            ONNX_CACHE_DIR,
            CALIBRATION_CACHE_DIR,
            TRAINER_CACHE_DIR,
            EMBEDDING_CACHE_DIR,
            CONFIG_CACHE_DIR,
            REPORT_CACHE_DIR,
            TEMP_CACHE_DIR,
        ]

        for cache_subdir in cache_subdirs:
            cache_subdir.mkdir(parents=True, exist_ok=True)

        # 이전 캐시 디렉토리가 있으면 파일 마이그레이션 시도
        if old_cache_dir.exists():
            _migrate_old_cache(old_cache_dir)

        logger.info(
            f"✅ 캐시 디렉토리 초기화 완료: {len(cache_subdirs)}개 디렉토리 생성"
        )

    except Exception as e:
        logger.error(f"캐시 디렉토리 초기화 실패: {str(e)}")
        raise RuntimeError(f"캐시 디렉토리 초기화 실패: {str(e)}")


def _migrate_old_cache(old_cache_dir: Path):
    """이전 캐시 디렉토리에서 파일을 마이그레이션합니다."""
    try:
        # 이미 마이그레이션됐는지 확인하는 플래그 파일
        migration_flag = CACHE_DIR / ".migrated"

        if not migration_flag.exists():
            logger.info("기존 캐시 파일을 새 위치로 마이그레이션 시작...")

            # 각 하위 디렉토리를 새 위치로 복사
            for old_dir in old_cache_dir.glob("*"):
                if old_dir.is_dir():
                    new_dir = CACHE_DIR / old_dir.name
                    if not new_dir.exists():
                        shutil.copytree(old_dir, new_dir)
                    else:
                        # 파일 단위 복사
                        for old_file in old_dir.glob("*"):
                            if old_file.is_file():
                                new_file = new_dir / old_file.name
                                if not new_file.exists():
                                    shutil.copy2(old_file, new_file)

            # 마이그레이션 완료 표시
            migration_flag.touch()
            logger.info(f"캐시 마이그레이션 완료: {old_cache_dir} -> {CACHE_DIR}")
    except Exception as e:
        logger.error(f"캐시 마이그레이션 중 오류 발생: {str(e)}")


class UnifiedCachePathManager:
    """모든 캐시 경로를 중앙에서 일관되게 관리하는 싱글톤 클래스"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_dir: str = "data/cache"):
        if self._initialized:
            return

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.info(f"✅ 통합 캐시 경로 관리자 초기화 (기본 경로: {self.base_dir})")

    def get_path(self, cache_type: str, filename: Optional[str] = None) -> Path:
        """
        지정된 유형의 캐시 경로를 가져오고, 필요 시 디렉토리를 생성합니다.

        Args:
            cache_type: 캐시 유형 (e.g., 'model', 'feature_vector')
            filename: 특정 파일 이름 (선택 사항)

        Returns:
            캐시 경로 (Path 객체)
        """
        type_dir = self.base_dir / cache_type
        type_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            return type_dir / filename
        return type_dir

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        특정 유형의 캐시 또는 전체 캐시를 정리합니다.

        Args:
            cache_type: 정리할 캐시 유형. None이면 전체 캐시를 정리합니다.
        """
        if cache_type:
            target_dir = self.base_dir / cache_type
            if target_dir.exists():
                shutil.rmtree(target_dir)
                logger.info(f"캐시 유형 '{cache_type}' 정리 완료.")
            else:
                logger.warning(f"캐시 유형 '{cache_type}'을(를) 찾을 수 없습니다.")
        else:
            shutil.rmtree(self.base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)  # 기본 디렉토리 다시 생성
            logger.info("모든 캐시 정리 완료.")


def get_cache_path_manager() -> UnifiedCachePathManager:
    """UnifiedCachePathManager의 싱글톤 인스턴스를 반환합니다."""
    return get_singleton_instance(UnifiedCachePathManager)


# 모듈 로드 시 자동 초기화 제거 - lazy 초기화 사용
