#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
분석기 팩토리 모듈

중복 초기화를 방지하고 분석기 인스턴스를 효율적으로 관리합니다.
"""

import hashlib
import json
import threading
from typing import Dict, Any, Optional, Union
from src.utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class AnalyzerFactory:
    """분석기 팩토리 클래스 - 싱글톤 패턴으로 분석기 인스턴스 관리"""

    _instance = None
    _lock = threading.Lock()
    _analyzers: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            logger.debug("AnalyzerFactory 초기화 완료")

    def _create_config_hash(self, config: Dict[str, Any]) -> str:
        """설정 딕셔너리의 해시값을 생성합니다."""
        try:
            # 설정을 정렬된 JSON 문자열로 변환하여 해시 생성
            config_str = json.dumps(config, sort_keys=True, default=str)
            return hashlib.md5(config_str.encode()).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"설정 해시 생성 실패: {e}")
            return "default"

    def get_analyzer(self, analyzer_type: str, config: Dict[str, Any]) -> Any:
        """분석기 인스턴스를 가져오거나 생성합니다."""
        # 설정 해시 대신 간단한 타입 기반 캐싱으로 변경 (더 강력한 중복 방지)
        cache_key = f"{analyzer_type}_singleton"

        with self._lock:
            # 캐시된 인스턴스가 있으면 반환
            if cache_key in self._analyzers:
                logger.debug(f"캐시된 {analyzer_type} 분석기 재사용: {cache_key}")
                return self._analyzers[cache_key]

            # 새로운 인스턴스 생성
            logger.debug(f"새로운 {analyzer_type} 분석기 생성 시작...")
            analyzer = self._create_analyzer(analyzer_type, config)
            if analyzer is not None:
                self._analyzers[cache_key] = analyzer
                logger.info(f"✅ {analyzer_type} 분석기 팩토리 생성 완료")
            else:
                logger.error(f"❌ {analyzer_type} 분석기 생성 실패")

            return analyzer

    def _create_analyzer(
        self, analyzer_type: str, config: Dict[str, Any]
    ) -> Optional[Any]:
        """실제 분석기 인스턴스를 생성합니다."""
        try:
            if analyzer_type == "pattern":
                from src.analysis.pattern_analyzer import PatternAnalyzer

                return PatternAnalyzer(config)

            elif analyzer_type == "distribution":
                from src.analysis.distribution_analyzer import DistributionAnalyzer

                return DistributionAnalyzer(config)

            elif analyzer_type == "roi":
                from src.analysis.roi_analyzer import ROIAnalyzer

                return ROIAnalyzer(config)

            elif analyzer_type == "pair":
                from src.analysis.pair_analyzer import PairAnalyzer

                return PairAnalyzer(config)

            elif analyzer_type == "vectorizer":
                from src.analysis.pattern_vectorizer import PatternVectorizer

                return PatternVectorizer(config)

            else:
                raise ValueError(f"알 수 없는 분석기 타입: {analyzer_type}")

        except Exception as e:
            logger.error(f"{analyzer_type} 분석기 생성 실패: {e}")
            return None

    def clear_cache(self):
        """캐시된 분석기 인스턴스를 모두 제거합니다."""
        with self._lock:
            self._analyzers.clear()
            logger.info("분석기 캐시 초기화 완료")

    def get_cache_info(self) -> Dict[str, int]:
        """캐시 정보를 반환합니다."""
        with self._lock:
            return {
                "cached_analyzers": len(self._analyzers),
                "analyzer_types": list(
                    set(key.split("_")[0] for key in self._analyzers.keys())
                ),
            }


# 전역 팩토리 인스턴스
analyzer_factory = AnalyzerFactory()


def get_analyzer(analyzer_type: str, config: Dict[str, Any]) -> Any:
    """분석기 인스턴스를 가져오는 편의 함수"""
    return analyzer_factory.get_analyzer(analyzer_type, config)


def clear_analyzer_cache():
    """분석기 캐시를 초기화하는 편의 함수"""
    analyzer_factory.clear_cache()
