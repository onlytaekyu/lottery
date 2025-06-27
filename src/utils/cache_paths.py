"""
캐시 경로 관리 유틸리티

모든 캐시 파일을 중앙집중화된 위치에 저장하기 위한 모듈입니다.
프로젝트 전체에서 일관된 캐시 경로를 사용하도록 합니다.
"""

from pathlib import Path
from typing import Optional

from .error_handler_refactored import get_logger

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


def get_cache_dir(category: Optional[str] = None) -> Path:
    """
    특정 카테고리의 캐시 디렉토리 경로를 반환합니다.

    Args:
        category: 캐시 카테고리 (기본값: None)

    Returns:
        캐시 디렉토리 경로
    """
    if category is None:
        return CACHE_DIR

    cache_dir = CACHE_DIR / category
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# 캐시 디렉토리 초기화
def init_cache_dirs():
    """모든 캐시 디렉토리를 초기화합니다."""
    # 이전 .cache 디렉토리 마이그레이션 체크
    old_cache_dir = PROJECT_ROOT / "data" / ".cache"

    # 새 캐시 디렉토리 생성
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 모든 하위 캐시 디렉토리 생성
    for cache_subdir in [
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
    ]:
        cache_subdir.mkdir(parents=True, exist_ok=True)

    # 이전 캐시 디렉토리가 있으면 파일 마이그레이션 시도
    if old_cache_dir.exists():
        try:
            import shutil

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


# 모듈 로드 시 캐시 디렉토리 초기화
init_cache_dirs()
