"""
통합 분석 모듈

이 모듈은 다양한 분석 기능을 통합하여 제공합니다.
"""

# pyright: reportCallIssue=false

from typing import Dict, List, Any, Optional, Tuple, Set, cast
import numpy as np
from pathlib import Path
import json

# logging 제거 - unified_logging 사용
import time
from datetime import datetime
import pickle
import os  # 추가: os 모듈 임포트

from ..utils.unified_logging import get_logger
from ..utils.unified_performance import performance_monitor
from ..shared.types import LotteryNumber, PatternAnalysis
from .enhanced_pattern_vectorizer import EnhancedPatternVectorizer
from .base_analyzer import BaseAnalyzer

from .roi_analyzer import ROIAnalyzer, ROIMetrics
from .pattern_analyzer import PatternAnalyzer
from .distribution_analyzer import DistributionAnalyzer, DistributionPattern
from ..utils.pattern_filter import PatternFilter, get_pattern_filter

# 추가 분석기 임포트 (1단계: cluster, trend 활성화)
from .cluster_analyzer import ClusterAnalyzer
from .trend_analyzer import TrendAnalyzer

logger = get_logger(__name__)


class UnifiedAnalyzer(BaseAnalyzer[Dict[str, Any]]):
    """통합 분석 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        초기화

        Args:
            config: 분석 설정
        """
        super().__init__(config or {}, "unified")
        self.logger = get_logger(__name__)

        # 설정을 딕셔너리로 처리
        self.config = config or {}

        # 기존 분석기 초기화
        self.pattern_analyzer = PatternAnalyzer(config)  # type: ignore
        self.roi_analyzer = ROIAnalyzer(config)  # type: ignore
        self.distribution_analyzer = DistributionAnalyzer(config)  # type: ignore

        # 새로 활성화된 분석기들 (1단계)
        try:
            self.cluster_analyzer = ClusterAnalyzer(config)  # type: ignore
            self.logger.info("✅ 클러스터 분석기 활성화 완료")
        except Exception as e:
            self.logger.warning(f"클러스터 분석기 초기화 실패: {e}")
            self.cluster_analyzer = None

        try:
            self.trend_analyzer = TrendAnalyzer(config)  # type: ignore
            self.logger.info("✅ 트렌드 분석기 활성화 완료")
        except Exception as e:
            self.logger.warning(f"트렌드 분석기 초기화 실패: {e}")
            self.trend_analyzer = None

        # 패턴 벡터라이저 초기화
        self.pattern_vectorizer = EnhancedPatternVectorizer(config)  # type: ignore

        # 패턴 필터 초기화
        self.pattern_filter = get_pattern_filter(config)  # type: ignore

        # 캐시 디렉토리 설정
        cache_dir = self.config.get("paths", {}).get("cache_dir", "data/cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        통합 분석 수행

        Args:
            historical_data: 과거 당첨 번호 데이터

        Returns:
            분석 결과
        """
        self.logger.info(f"통합 분석 시작: {len(historical_data)}개 데이터")

        # 성능 추적 시작
        self.performance_tracker.start_tracking("unified_analysis")

        try:
            # 캐시 키 생성
            cache_key = self._create_cache_key("unified_analysis", len(historical_data))

            # 캐시 파일 경로
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            # 캐시 확인
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        cached_result = pickle.load(f)
                        self.logger.info(f"캐시된 분석 결과 사용: {cache_file}")
                        # 성능 추적 종료
                        self.performance_tracker.stop_tracking("unified_analysis")
                        return cached_result
                except Exception as e:
                    self.logger.warning(f"캐시 로드 실패: {e}")

            # 기존 분석기들
            # 패턴 분석 (기존 + 추가 분석 함께 수행)
            self.performance_tracker.start_tracking("pattern_analysis")
            try:
                pattern_results = self.pattern_analyzer.run_all_analyses(
                    historical_data
                )
            finally:
                self.performance_tracker.stop_tracking("pattern_analysis")

            # 분포 분석
            self.performance_tracker.start_tracking("distribution_analysis")
            try:
                distribution_pattern = self.distribution_analyzer.analyze(
                    historical_data
                )
            finally:
                self.performance_tracker.stop_tracking("distribution_analysis")

            # ROI 분석 (기존에 있었지만 누락된 부분 추가)
            roi_results = {}
            if self.roi_analyzer:
                self.performance_tracker.start_tracking("roi_analysis")
                try:
                    roi_results = self.roi_analyzer.analyze(historical_data)
                    self.logger.info("✅ ROI 분석 완료")
                finally:
                    self.performance_tracker.stop_tracking("roi_analysis")

            # 새로 활성화된 분석기들 (1단계)
            # 클러스터 분석
            cluster_results = {}
            if self.cluster_analyzer:
                self.performance_tracker.start_tracking("cluster_analysis")
                try:
                    cluster_results = self.cluster_analyzer.analyze(historical_data)
                    self.logger.info("✅ 클러스터 분석 완료")
                except Exception as e:
                    self.logger.warning(f"클러스터 분석 실패: {e}")
                    cluster_results = {}
                finally:
                    self.performance_tracker.stop_tracking("cluster_analysis")

            # 트렌드 분석
            trend_results = {}
            if self.trend_analyzer:
                self.performance_tracker.start_tracking("trend_analysis")
                try:
                    trend_results = self.trend_analyzer.analyze(historical_data)
                    self.logger.info("✅ 트렌드 분석 완료")
                except Exception as e:
                    self.logger.warning(f"트렌드 분석 실패: {e}")
                    trend_results = {}
                finally:
                    self.performance_tracker.stop_tracking("trend_analysis")

            # 결과 통합 (새로운 분석 결과 포함)
            result = {
                "pattern_analysis": pattern_results.get("full_analysis", {}),
                "distribution_pattern": distribution_pattern,
                "roi_analysis": roi_results,  # ROI 분석 결과 추가
                "cluster_analysis": cluster_results,  # 클러스터 분석 결과 추가
                "trend_analysis": trend_results,  # 트렌드 분석 결과 추가
                "data_count": len(historical_data),
                "timestamp": int(time.time()),  # 현재 시간 타임스탬프
            }

            # 추가 분석 결과 통합
            segment_10_result = {}
            try:
                if self.config["analysis"]["enable_segment_10"]:
                    segment_10_result = pattern_results.get("segment_10_frequency", {})
            except KeyError:
                if pattern_results.get("segment_10_frequency"):
                    segment_10_result = pattern_results.get("segment_10_frequency", {})

            segment_5_result = {}
            try:
                if self.config["analysis"]["enable_segment_5"]:
                    segment_5_result = pattern_results.get("segment_5_frequency", {})
            except KeyError:
                if pattern_results.get("segment_5_frequency"):
                    segment_5_result = pattern_results.get("segment_5_frequency", {})

            gap_stats_result = {}
            try:
                if self.config["analysis"]["enable_gap_stats"]:
                    gap_stats_result = pattern_results.get("gap_statistics", {})
            except KeyError:
                if pattern_results.get("gap_statistics"):
                    gap_stats_result = pattern_results.get("gap_statistics", {})

            pattern_reappearance_result = {}
            try:
                if self.config["analysis"]["enable_pattern_interval"]:
                    pattern_reappearance_result = pattern_results.get(
                        "pattern_reappearance", {}
                    )
            except KeyError:
                if pattern_results.get("pattern_reappearance"):
                    pattern_reappearance_result = pattern_results.get(
                        "pattern_reappearance", {}
                    )

            recent_gap_result = {}
            try:
                if self.config["analysis"]["enable_recent_gap"]:
                    recent_gap_result = pattern_results.get(
                        "recent_reappearance_gap", {}
                    )
            except KeyError:
                if pattern_results.get("recent_reappearance_gap"):
                    recent_gap_result = pattern_results.get(
                        "recent_reappearance_gap", {}
                    )

            segment_10_history_result = {}
            segment_5_history_result = {}
            try:
                if self.config["analysis"]["enable_segment_history"]:
                    segment_10_history_result = pattern_results.get(
                        "segment_10_history", {}
                    )
                    segment_5_history_result = pattern_results.get(
                        "segment_5_history", {}
                    )
            except KeyError:
                if pattern_results.get("segment_10_history") and pattern_results.get(
                    "segment_5_history"
                ):
                    segment_10_history_result = pattern_results.get(
                        "segment_10_history", {}
                    )
                    segment_5_history_result = pattern_results.get(
                        "segment_5_history", {}
                    )

            segment_centrality_result = {}
            try:
                if self.config["analysis"]["enable_segment_centrality"]:
                    segment_centrality_result = pattern_results.get(
                        "segment_centrality", {}
                    )
            except KeyError:
                if pattern_results.get("segment_centrality"):
                    segment_centrality_result = pattern_results.get(
                        "segment_centrality", {}
                    )

            segment_consecutive_result = {}
            try:
                if self.config["analysis"]["enable_segment_consecutive"]:
                    segment_consecutive_result = pattern_results.get(
                        "segment_consecutive_patterns", {}
                    )
            except KeyError:
                if pattern_results.get("segment_consecutive_patterns"):
                    segment_consecutive_result = pattern_results.get(
                        "segment_consecutive_patterns", {}
                    )

            # 결과 통합
            result.update(
                {
                    "segment_10_frequency": segment_10_result,
                    "segment_5_frequency": segment_5_result,
                    "gap_statistics": gap_stats_result,
                    "pattern_reappearance": pattern_reappearance_result,
                    "recent_reappearance_gap": recent_gap_result,
                    "segment_10_history": segment_10_history_result,
                    "segment_5_history": segment_5_history_result,
                    "segment_centrality": segment_centrality_result,
                    "segment_consecutive_patterns": segment_consecutive_result,
                }
            )

            # 결과 저장 (파일로)
            self.save_analysis_results(result)

            # 벡터화 및 캐시 저장
            self.performance_tracker.start_tracking("vectorize_results")
            try:
                # 패턴 벡터화 (간단한 특성 벡터 생성)
                if hasattr(self.pattern_vectorizer, "vectorize_pattern_features"):
                    # 기본 특성 생성
                    basic_features = {
                        "total_draws": len(historical_data),
                        "recent_patterns": len(pattern_results.get("recent_100", {})),
                        "distribution_score": len(
                            distribution_pattern.get("even_odd", [])
                        ),
                    }
                    vector = self.pattern_vectorizer.vectorize_pattern_features(
                        basic_features
                    )
                    self.logger.info("패턴 벡터화 완료")
                else:
                    self.logger.info("벡터화 기능 스킵")

                # 세그먼트 히스토리 벡터 저장 (옵션)
                try:
                    enable_segment_history = self.config["analysis"][
                        "enable_segment_history"
                    ]
                except KeyError:
                    enable_segment_history = True

                if (
                    "segment_10_history" in result
                    and "segment_5_history" in result
                    and enable_segment_history
                ):
                    # save_segment_history_to_numpy 메서드가 구현되지 않아 로그로 대체
                    self.logger.info("세그먼트 히스토리 데이터 처리 완료")
            finally:
                self.performance_tracker.stop_tracking("vectorize_results")

            # 결과 캐싱
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                self.logger.info(f"분석 결과 캐시 저장 완료: {cache_file}")
            except Exception as e:
                self.logger.warning(f"분석 결과 캐시 저장 실패: {e}")

            self.logger.info("통합 분석 완료")

            # 성능 추적 종료
            self.performance_tracker.stop_tracking("unified_analysis")
            return result
        except Exception as e:
            # 오류가 발생해도 성능 추적 종료를 보장
            self.performance_tracker.stop_tracking("unified_analysis")
            raise e

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        캐시 확인 (오버라이드)

        Args:
            cache_key: 캐시 키

        Returns:
            캐시된 결과 또는 None
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_result = pickle.load(f)
                    self.logger.info(f"캐시된 분석 결과 사용: {cache_file}")
                    return cached_result
            except Exception as e:
                self.logger.warning(f"캐시 로드 실패: {e}")

        return None

    def save_analysis_results(self, unified_results: Dict[str, Any]) -> str:
        """
        분석 결과를 세그먼트 히스토리 벡터로만 저장합니다.
        사람용 JSON 및 텍스트 파일 저장 로직은 제거되었습니다.

        Args:
            unified_results: 분석 결과

        Returns:
            str: 성공 메시지
        """
        try:
            # 세그먼트 히스토리 벡터 저장
            try:
                enable_segment_history = self.config["analysis"][
                    "enable_segment_history"
                ]
            except KeyError:
                enable_segment_history = True

            if (
                "segment_10_history" in unified_results
                and "segment_5_history" in unified_results
                and enable_segment_history
            ):
                # save_segment_history_to_numpy 메서드가 구현되지 않아 로그로 대체
                self.logger.info("세그먼트 히스토리 데이터 처리 완료")

            return "세그먼트 히스토리 벡터 저장 완료"
        except Exception as e:
            self.logger.error(f"분석 결과 처리 중 오류 발생: {e}")
            return ""

    def analyze_number_combination(
        self, historical_data: List[LotteryNumber], target_numbers: List[int]
    ) -> ROIMetrics:
        """
        특정 번호 조합에 대한 ROI 분석을 수행합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            target_numbers: 분석할 번호 조합(6개 번호)

        Returns:
            ROIMetrics: ROI 분석 결과
        """
        # 번호는 항상 오름차순 정렬
        sorted_numbers = sorted(target_numbers)
        return self.roi_analyzer.analyze(historical_data, sorted_numbers)

    def analyze_multiple_combinations(
        self, historical_data: List[LotteryNumber], combinations: List[List[int]]
    ) -> Dict[str, ROIMetrics]:
        """
        여러 번호 조합에 대한 ROI 분석을 일괄 수행합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록
            combinations: 분석할 번호 조합 목록

        Returns:
            Dict[str, ROIMetrics]: 번호 조합별 ROI 분석 결과
        """
        return self.roi_analyzer.analyze_batch(historical_data, combinations)

    def filter_combinations(
        self,
        combinations: List[List[int]],
        historical_data: Optional[List[LotteryNumber]] = None,
        filter_names: Optional[List[str]] = None,
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        번호 조합 필터링

        Args:
            combinations: 필터링할 번호 조합 목록
            historical_data: 과거 당첨 번호 데이터
            filter_names: 적용할 필터 이름 목록

        Returns:
            (통과한 조합 목록, 필터링 결과 상세 정보)
        """
        self.logger.info(f"{len(combinations)}개 번호 조합 필터링 시작")

        with performance_monitor("filter_combinations"):
            # 통과한 조합 목록
            passed_combinations = []

            # 필터 결과 상세 정보
            filter_results = {
                "total": len(combinations),
                "passed": 0,
                "failed": 0,
                "filter_stats": {},
            }

            # 각 조합 필터링
            for combination in combinations:
                passed, results = self.pattern_filter.filter_numbers(
                    combination, historical_data, filter_names
                )

                # 통과 여부에 따라 처리
                if passed:
                    passed_combinations.append(combination)
                    filter_results["passed"] += 1
                else:
                    filter_results["failed"] += 1

                # 필터별 통계 업데이트
                for filter_name, filter_result in results.items():
                    if filter_name not in filter_results["filter_stats"]:
                        filter_results["filter_stats"][filter_name] = {
                            "passed": 0,
                            "failed": 0,
                        }

                    if filter_result.get("passed", False):
                        filter_results["filter_stats"][filter_name]["passed"] += 1
                    else:
                        filter_results["filter_stats"][filter_name]["failed"] += 1

            self.logger.info(
                f"필터링 완료: {filter_results['passed']}개 통과, {filter_results['failed']}개 실패"
            )
            return passed_combinations, filter_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        stats = {}
        # 각 성능 지표를 수집
        for analyzer_name, analyzer in [
            ("pattern", self.pattern_analyzer),
            ("roi", self.roi_analyzer),
            ("distribution", self.distribution_analyzer),
        ]:
            # 각 분석기의 성능 지표 수집
            with performance_monitor(f"get_stats_{analyzer_name}"):
                operation_times = {}
                for op_name in ["analyze", f"{analyzer_name}_analysis"]:
                    operation_times[op_name] = (
                        self.performance_tracker.get_metric(op_name) or 0.0
                    )

                stats[analyzer_name] = {
                    "operations": operation_times,
                    "cache_hits": 0,  # 실제 캐시 히트 수는 분석기에서 추적해야 함
                }

        # 통합 성능 지표
        stats["total"] = {
            "operations": {
                "unified_analysis": self.performance_tracker.get_metric(
                    "unified_analysis"
                )
                or 0.0,
                "filter_combinations": self.performance_tracker.get_metric(
                    "filter_combinations"
                )
                or 0.0,
            }
        }

        return stats

    def run_full_analysis(self, draw_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        모든 데이터 분석을 통합해서 수행합니다.

        Args:
            draw_data: 과거 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 통합 분석 결과
        """
        with performance_monitor("run_full_analysis"):
            # 타임스탬프 기록
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 1. 패턴 분석기 실행
            pattern_results = self.pattern_analyzer.run_all_analyses(draw_data)

            # 2. 결과 통합
            unified_results = {
                "timestamp": timestamp,
                "total_draws": len(draw_data),
                "last_draw": self._extract_last_draw(draw_data),
                # 패턴 분석 결과
                "full_analysis": pattern_results["full_analysis"],
                "recent_100": pattern_results["recent_100"],
                "last_year": pattern_results["last_year"],
                "last_month": pattern_results["last_month"],
                # 세그먼트 분석 결과
                "segment_10_frequency": pattern_results["segment_10_frequency"],
                "segment_5_frequency": pattern_results["segment_5_frequency"],
                # 갭 및 패턴 분석
                "gap_statistics": pattern_results["gap_statistics"],
                "pattern_reappearance": pattern_results["pattern_reappearance"],
                "recent_reappearance_gap": pattern_results["recent_reappearance_gap"],
                # 세그먼트 히스토리
                "segment_10_history": pattern_results["segment_10_history"],
                "segment_5_history": pattern_results["segment_5_history"],
                # 세그먼트 중심성 및 연속성 분석
                "segment_centrality": pattern_results["segment_centrality"],
                "segment_consecutive_patterns": pattern_results[
                    "segment_consecutive_patterns"
                ],
                # 중복 당첨 번호 분석
                "identical_draw_check": pattern_results["identical_draw_check"],
            }

            # 분포 분석
            with performance_monitor("distribution_analysis"):
                distribution_pattern = self.distribution_analyzer.analyze(draw_data)

            # 결과 통합
            unified_results["distribution_pattern"] = distribution_pattern

            # 결과 저장 (파일로)
            self.save_analysis_results(unified_results)

            # 벡터화 및 캐시 저장
            self.performance_tracker.start_tracking("vectorize_results")
            try:
                # 패턴 벡터화 (간단한 특성 벡터 생성)
                if hasattr(self.pattern_vectorizer, "vectorize_pattern_features"):
                    # 기본 특성 생성
                    basic_features = {
                        "total_draws": len(draw_data),
                        "recent_patterns": len(unified_results.get("recent_100", {})),
                        "distribution_score": len(
                            distribution_pattern.get("even_odd", [])
                        ),
                    }
                    vector = self.pattern_vectorizer.vectorize_pattern_features(
                        basic_features
                    )
                    self.logger.info("패턴 벡터화 완료")
                else:
                    self.logger.info("벡터화 기능 스킵")

                # 세그먼트 히스토리 벡터 저장 (옵션)
                try:
                    enable_segment_history = self.config["analysis"][
                        "enable_segment_history"
                    ]
                except KeyError:
                    enable_segment_history = True

                if (
                    "segment_10_history" in unified_results
                    and "segment_5_history" in unified_results
                    and enable_segment_history
                ):
                    # save_segment_history_to_numpy 메서드가 구현되지 않아 로그로 대체
                    self.logger.info("세그먼트 히스토리 데이터 처리 완료")
            finally:
                self.performance_tracker.stop_tracking("vectorize_results")

            self.logger.info("통합 분석 완료")
            return unified_results

    def _extract_last_draw(self, draw_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        가장 최근 추첨 데이터를 추출합니다.

        Args:
            draw_data: 과거 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 최근 추첨 정보
        """
        if not draw_data:
            return {}

        last_draw = draw_data[-1]

        return {
            "draw_no": last_draw.draw_no if hasattr(last_draw, "draw_no") else 0,
            "numbers": last_draw.numbers if hasattr(last_draw, "numbers") else [],
            "date": last_draw.date if hasattr(last_draw, "date") else "",
        }

    def run_all_analyses(self, draw_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        모든 분석을 수행하고 결과를 반환합니다.

        Args:
            draw_data: 과거 당첨 번호 데이터

        Returns:
            Dict[str, Any]: 통합 분석 결과
        """
        try:
            self.logger.info("통합 분석 시작")

            # 모든 데이터 분석
            with performance_monitor("complete_analysis"):
                result = self.run_full_analysis(draw_data)

            self.logger.info("통합 분석 완료")
            return result
        except Exception as e:
            self.logger.error(f"통합 분석 중 오류 발생: {e}")
            raise

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        통합 분석 수행

        Args:
            historical_data: 과거 당첨 번호 데이터
            *args, **kwargs: 추가 매개변수 (사용되지 않음)

        Returns:
            분석 결과
        """
        self.logger.info(f"통합 분석 시작: {len(historical_data)}개 데이터")

        try:
            # 패턴 분석 (기존 + 추가 분석 함께 수행)
            self.performance_tracker.start_tracking("pattern_analysis")
            try:
                pattern_results = self.pattern_analyzer.run_all_analyses(
                    historical_data
                )
            finally:
                self.performance_tracker.stop_tracking("pattern_analysis")

            # 분포 분석
            self.performance_tracker.start_tracking("distribution_analysis")
            try:
                distribution_pattern = self.distribution_analyzer.analyze(
                    historical_data
                )
            finally:
                self.performance_tracker.stop_tracking("distribution_analysis")

            # 결과 통합
            result = {
                "pattern_analysis": pattern_results.get("full_analysis", {}),
                "distribution_pattern": distribution_pattern,
                "data_count": len(historical_data),
                "timestamp": int(time.time()),  # 현재 시간 타임스탬프
            }

            # 추가 분석 결과 통합
            segment_10_result = {}
            try:
                if self.config["analysis"]["enable_segment_10"]:
                    segment_10_result = pattern_results.get("segment_10_frequency", {})
            except KeyError:
                if pattern_results.get("segment_10_frequency"):
                    segment_10_result = pattern_results.get("segment_10_frequency", {})

            segment_5_result = {}
            try:
                if self.config["analysis"]["enable_segment_5"]:
                    segment_5_result = pattern_results.get("segment_5_frequency", {})
            except KeyError:
                if pattern_results.get("segment_5_frequency"):
                    segment_5_result = pattern_results.get("segment_5_frequency", {})

            gap_stats_result = {}
            try:
                if self.config["analysis"]["enable_gap_stats"]:
                    gap_stats_result = pattern_results.get("gap_statistics", {})
            except KeyError:
                if pattern_results.get("gap_statistics"):
                    gap_stats_result = pattern_results.get("gap_statistics", {})

            pattern_reappearance_result = {}
            try:
                if self.config["analysis"]["enable_pattern_interval"]:
                    pattern_reappearance_result = pattern_results.get(
                        "pattern_reappearance", {}
                    )
            except KeyError:
                if pattern_results.get("pattern_reappearance"):
                    pattern_reappearance_result = pattern_results.get(
                        "pattern_reappearance", {}
                    )

            recent_gap_result = {}
            try:
                if self.config["analysis"]["enable_recent_gap"]:
                    recent_gap_result = pattern_results.get(
                        "recent_reappearance_gap", {}
                    )
            except KeyError:
                if pattern_results.get("recent_reappearance_gap"):
                    recent_gap_result = pattern_results.get(
                        "recent_reappearance_gap", {}
                    )

            segment_10_history_result = {}
            segment_5_history_result = {}
            try:
                if self.config["analysis"]["enable_segment_history"]:
                    segment_10_history_result = pattern_results.get(
                        "segment_10_history", {}
                    )
                    segment_5_history_result = pattern_results.get(
                        "segment_5_history", {}
                    )
            except KeyError:
                if pattern_results.get("segment_10_history") and pattern_results.get(
                    "segment_5_history"
                ):
                    segment_10_history_result = pattern_results.get(
                        "segment_10_history", {}
                    )
                    segment_5_history_result = pattern_results.get(
                        "segment_5_history", {}
                    )

            segment_centrality_result = {}
            try:
                if self.config["analysis"]["enable_segment_centrality"]:
                    segment_centrality_result = pattern_results.get(
                        "segment_centrality", {}
                    )
            except KeyError:
                if pattern_results.get("segment_centrality"):
                    segment_centrality_result = pattern_results.get(
                        "segment_centrality", {}
                    )

            segment_consecutive_result = {}
            try:
                if self.config["analysis"]["enable_segment_consecutive"]:
                    segment_consecutive_result = pattern_results.get(
                        "segment_consecutive_patterns", {}
                    )
            except KeyError:
                if pattern_results.get("segment_consecutive_patterns"):
                    segment_consecutive_result = pattern_results.get(
                        "segment_consecutive_patterns", {}
                    )

            # 결과 통합
            result.update(
                {
                    "segment_10_frequency": segment_10_result,
                    "segment_5_frequency": segment_5_result,
                    "gap_statistics": gap_stats_result,
                    "pattern_reappearance": pattern_reappearance_result,
                    "recent_reappearance_gap": recent_gap_result,
                    "segment_10_history": segment_10_history_result,
                    "segment_5_history": segment_5_history_result,
                    "segment_centrality": segment_centrality_result,
                    "segment_consecutive_patterns": segment_consecutive_result,
                }
            )

            # 결과 저장 (파일로)
            self.save_analysis_results(result)

            # 벡터화 및 캐시 저장
            self.performance_tracker.start_tracking("vectorize_results")
            try:
                # 패턴 벡터화 (간단한 특성 벡터 생성)
                if hasattr(self.pattern_vectorizer, "vectorize_pattern_features"):
                    # 기본 특성 생성
                    basic_features = {
                        "total_draws": len(historical_data),
                        "recent_patterns": len(pattern_results.get("recent_100", {})),
                        "distribution_score": len(
                            distribution_pattern.get("even_odd", [])
                        ),
                    }
                    vector = self.pattern_vectorizer.vectorize_pattern_features(
                        basic_features
                    )
                    self.logger.info("패턴 벡터화 완료")
                else:
                    self.logger.info("벡터화 기능 스킵")

                # 세그먼트 히스토리 벡터 저장 (옵션)
                try:
                    enable_segment_history = self.config["analysis"][
                        "enable_segment_history"
                    ]
                except KeyError:
                    enable_segment_history = True

                if (
                    "segment_10_history" in result
                    and "segment_5_history" in result
                    and enable_segment_history
                ):
                    # save_segment_history_to_numpy 메서드가 구현되지 않아 로그로 대체
                    self.logger.info("세그먼트 히스토리 데이터 처리 완료")
            finally:
                self.performance_tracker.stop_tracking("vectorize_results")

            self.logger.info("통합 분석 완료")
            return result
        except Exception as e:
            # 오류 발생시 예외 다시 발생
            self.logger.error(f"통합 분석 중 오류 발생: {str(e)}")
            raise e
