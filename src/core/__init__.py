"""
코어 모듈 패키지

로또 번호 추천 시스템의 핵심 기능을 제공하는 모듈들을 포함합니다.
"""

from .recommendation_engine import RecommendationEngine, get_recommendation_engine

__all__ = ["RecommendationEngine", "get_recommendation_engine"]
