# src/pipeline/__init__.py

# 1. 표준 라이브러리
from typing import Any, Dict, Optional

# 2. 서드파티
# (필요 시 추가)

# 3. 프로젝트 내부
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_config

# --- 파이프라인 모듈 임포트 ---
# 각 파이프라인 클래스를 명시적으로 임포트합니다.
# 파일 이름과 클래스 이름이 일치한다고 가정합니다.
# 실제 클래스 이름에 따라 수정이 필요할 수 있습니다.

from .data_validation import DataValidation
from .enhanced_data_validation import EnhancedDataValidation
from .feature_engineering_pipeline import FeatureEngineeringPipeline
from .unified_preprocessing_pipeline import UnifiedPreprocessingPipeline
from .train_pipeline import TrainPipeline
from .negative_sampling_pipeline import NegativeSamplingPipeline
from .hierarchical_prediction_pipeline import HierarchicalPredictionPipeline
from .enhanced_meta_weight_layer import EnhancedMetaWeightLayer
from .model_performance_benchmark import ModelPerformanceBenchmark
from .optimized_data_analysis_pipeline import OptimizedDataAnalysisPipeline

logger = get_logger(__name__)

# 사용 가능한 파이프라인 클래스들을 딕셔너리에 매핑
AVAILABLE_PIPELINES = {
    "data_validation": DataValidation,
    "enhanced_data_validation": EnhancedDataValidation,
    "feature_engineering": FeatureEngineeringPipeline,
    "preprocessing": UnifiedPreprocessingPipeline,
    "train": TrainPipeline,
    "negative_sampling": NegativeSamplingPipeline,
    "hierarchical_prediction": HierarchicalPredictionPipeline,
    "meta_weighting": EnhancedMetaWeightLayer,
    "performance_benchmark": ModelPerformanceBenchmark,
    "data_analysis": OptimizedDataAnalysisPipeline,
}

def create_pipeline(pipeline_name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    파이프라인 팩토리 함수

    주요 기능:
    - 동적 파이프라인 생성
    - GPU/CPU 자동 선택 로직 (설정 기반)
    - 기본 에러 핸들링

    Args:
        pipeline_name (str): 생성할 파이프라인의 이름 (e.g., "preprocessing")
        config (Optional[Dict[str, Any]]): 파이프라인에 전달할 설정.
                                           None이면 전역 설정을 사용합니다.

    Returns:
        Any: 생성된 파이프라인 인스턴스

    Raises:
        ValueError: 요청된 파이프라인을 사용할 수 없는 경우
    """
    pipeline_name = pipeline_name.lower()
    if pipeline_name not in AVAILABLE_PIPELINES:
        logger.error(f"'{pipeline_name}'은(는) 유효한 파이프라인이 아닙니다. "
                     f"사용 가능한 파이프라인: {list(AVAILABLE_PIPELINES.keys())}")
        raise ValueError(f"알 수 없는 파이프라인: {pipeline_name}")

    # config가 None이면 전역 설정에서 가져오거나 빈 dict로 초기화
    if config is None:
        main_config = get_config("main")
        config = main_config.get(pipeline_name, {})

    # GPU/CPU 자동 선택 로직 (설정에 'use_gpu'가 없는 경우)
    if not isinstance(config, dict) or 'use_gpu' not in config:
        main_config = get_config("main")
        use_gpu = main_config.get('use_gpu', False)
        if isinstance(config, dict):
            config['use_gpu'] = use_gpu
        else:
            config = {'use_gpu': use_gpu}
        logger.info(f"전역 GPU 설정 적용: use_gpu={use_gpu}")
    
    pipeline_class = AVAILABLE_PIPELINES[pipeline_name]
    
    try:
        logger.info(f"'{pipeline_name}' 파이프라인 생성 중...")
        instance = pipeline_class(config)
        logger.info(f"'{pipeline_name}' 파이프라인이 성공적으로 생성되었습니다.")
        return instance
    except Exception as e:
        logger.error(f"'{pipeline_name}' 파이프라인 생성 중 오류 발생: {e}", exc_info=True)
        # 생성 실패 시 None을 반환하거나, 예외를 다시 발생시킬 수 있습니다.
        # 여기서는 예외를 다시 발생시켜 호출하는 쪽에서 처리하도록 합니다.
        raise

__all__ = [
    "create_pipeline",
    "DataValidation",
    "EnhancedDataValidation",
    "FeatureEngineeringPipeline",
    "UnifiedPreprocessingPipeline",
    "TrainPipeline",
    "NegativeSamplingPipeline",
    "HierarchicalPredictionPipeline",
    "EnhancedMetaWeightLayer",
    "ModelPerformanceBenchmark",
    "OptimizedDataAnalysisPipeline",
]
