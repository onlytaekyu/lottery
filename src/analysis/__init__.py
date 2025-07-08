"""
로또 번호 분석 모듈 - 완전판

모든 분석 기능을 통합한 최종 패키지입니다.
"""

# ===== 핵심 분석 모듈 =====
from .base_analyzer import BaseAnalyzer
from .unified_analyzer import UnifiedAnalyzer
from .pattern_analyzer import PatternAnalyzer
from .distribution_analyzer import DistributionAnalyzer, DistributionPattern
from .roi_analyzer import ROIAnalyzer, ROIMetrics
from .cluster_analyzer import ClusterAnalyzer
from .trend_analyzer import TrendAnalyzer

# ===== 고급 분석 모듈 (신규) =====
from .bayesian_analyzer import BayesianAnalyzer
from .trend_analyzer_v2 import TrendAnalyzerV2
from .ensemble_analyzer import EnsembleAnalyzer
from .graph_network_analyzer import GraphNetworkAnalyzer
from .meta_feature_analyzer import MetaFeatureAnalyzer

# ===== 벡터화 시스템 =====
from .enhanced_pattern_vectorizer import EnhancedPatternVectorizer
from .optimized_pattern_vectorizer import OptimizedPatternVectorizer

# ===== 3자리 시스템 =====
from .three_digit_expansion_engine import ThreeDigitExpansionEngine
from .three_digit_priority_predictor import ThreeDigitPriorityPredictor

# ===== 지원 모듈 =====
# from .negative_sample_generator import NegativeSampleGenerator  # 임시 제외
from .analyzer_factory import AnalyzerFactory, get_analyzer

# ===== 전문 분석기 =====
from .overlap_analyzer import OverlapAnalyzer
from .structural_analyzer import StructuralAnalyzer
from .statistical_analyzer import StatisticalAnalyzer
from .technical_analyzer import TechnicalAnalyzer

# from .pair_analyzer import PairAnalyzer  # 임시 제외

# ===== 패턴 필터 =====
from ..utils.pattern_filter import GPUPatternFilter, get_pattern_filter

__all__ = [
    # 핵심 분석 모듈
    "BaseAnalyzer",
    "UnifiedAnalyzer",
    "PatternAnalyzer",
    "DistributionAnalyzer",
    "DistributionPattern",
    "ROIAnalyzer",
    "ROIMetrics",
    "ClusterAnalyzer",
    "TrendAnalyzer",
    # 고급 분석 모듈
    "BayesianAnalyzer",
    "TrendAnalyzerV2",
    "EnsembleAnalyzer",
    "GraphNetworkAnalyzer",
    "MetaFeatureAnalyzer",
    # 벡터화 시스템
    "EnhancedPatternVectorizer",
    "OptimizedPatternVectorizer",
    # 3자리 시스템
    "ThreeDigitExpansionEngine",
    "ThreeDigitPriorityPredictor",
    # 지원 모듈
    # "NegativeSampleGenerator",  # 임시 제외
    "AnalyzerFactory",
    "get_analyzer",
    # 전문 분석기
    "OverlapAnalyzer",
    "StructuralAnalyzer",
    "StatisticalAnalyzer",
    "TechnicalAnalyzer",
    # "PairAnalyzer",  # 임시 제외
    # 패턴 필터
    "GPUPatternFilter",
    "get_pattern_filter",
]

# 버전 정보
__version__ = "2.0.0"
__author__ = "DAEBAK_AI Team"
__description__ = "완전한 로또 번호 분석 시스템"
