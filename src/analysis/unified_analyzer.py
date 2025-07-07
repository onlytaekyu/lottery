"""
통합 분석 모듈 (개선된 버전)

이 모듈은 다양한 분석 기능을 통합하여 제공하며, 중복 기능을 제거하고
3자리 우선 예측 시스템을 포함합니다.
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
from ..utils.unified_performance_engine import get_unified_performance_engine, TaskType
from ..shared.types import LotteryNumber, PatternAnalysis
from .enhanced_pattern_vectorizer import EnhancedPatternVectorizer
from .base_analyzer import BaseAnalyzer

from .roi_analyzer import ROIAnalyzer, ROIMetrics
from .pattern_analyzer import PatternAnalyzer
from .distribution_analyzer import DistributionAnalyzer, DistributionPattern
from ..utils.pattern_filter import GPUPatternFilter, get_pattern_filter

# 추가 분석기 임포트 (핵심 분석기만 유지)
from .cluster_analyzer import ClusterAnalyzer
from .trend_analyzer import TrendAnalyzer
from .three_digit_expansion_engine import ThreeDigitExpansionEngine

# 통합 성능 최적화 엔진 임포트
from ..utils.unified_performance_engine import get_unified_performance_engine

logger = get_logger(__name__)


class UnifiedAnalyzer(BaseAnalyzer[Dict[str, Any]]):
    """
    통합 분석 클래스 (개선된 버전)

    - 중복 기능 제거
    - 3자리 우선 예측 시스템 통합
    - 성능 최적화 엔진 활용
    """

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

        # 통합 성능 최적화 엔진 초기화
        self.performance_engine = get_unified_performance_engine()

        # 핵심 분석기만 초기화 (중복 제거)
        self.pattern_analyzer = PatternAnalyzer(config)  # 3자리 분석 포함
        self.roi_analyzer = ROIAnalyzer(config)
        self.distribution_analyzer = DistributionAnalyzer(config)

        # 선택적 분석기들
        try:
            self.cluster_analyzer = ClusterAnalyzer(config)
            self.logger.info("✅ 클러스터 분석기 활성화 완료")
        except Exception as e:
            self.logger.warning(f"클러스터 분석기 초기화 실패: {e}")
            self.cluster_analyzer = None

        try:
            self.trend_analyzer = TrendAnalyzer(config)
            self.logger.info("✅ 트렌드 분석기 활성화 완료")
        except Exception as e:
            self.logger.warning(f"트렌드 분석기 초기화 실패: {e}")
            self.trend_analyzer = None

        # 3자리 확장 엔진 초기화
        try:
            self.three_digit_engine = ThreeDigitExpansionEngine(config)
            self.logger.info("✅ 3자리 확장 엔진 활성화 완료")
        except Exception as e:
            self.logger.warning(f"3자리 확장 엔진 초기화 실패: {e}")
            self.three_digit_engine = None

        # 패턴 벡터라이저 초기화 (단일 시스템)
        self.pattern_vectorizer = EnhancedPatternVectorizer(config)

        # 패턴 필터 초기화
        self.pattern_filter = get_pattern_filter(config)

        # 캐시 디렉토리 설정
        cache_dir = self.config.get("paths", {}).get("cache_dir", "data/cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("✅ 통합 분석기 초기화 완료 (중복 제거, 3자리 시스템 통합)")

    def _validate_cache_file(self, cache_file: Path) -> bool:
        """
        캐시 파일의 유효성을 검증합니다.

        Args:
            cache_file: 검증할 캐시 파일 경로

        Returns:
            bool: 파일이 유효하면 True, 아니면 False
        """
        try:
            # 파일 크기 확인 (최소 크기)
            if cache_file.stat().st_size < 10:
                return False

            # pickle 헤더 확인
            with open(cache_file, "rb") as f:
                # pickle 매직 넘버 확인
                header = f.read(2)
                if len(header) < 2:
                    return False

                # Python pickle 프로토콜 버전 확인
                if (
                    header[0] not in [0x80, ord("("), ord("]"), ord("}")]
                    and header != b"\x80\x03"
                ):
                    return False

            return True
        except Exception:
            return False

    def analyze(self, historical_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        통합 분석 수행 (3자리 우선 예측 포함)

        Args:
            historical_data: 과거 당첨 번호 데이터

        Returns:
            분석 결과 (3자리 우선 예측 포함)
        """
        self.logger.info(f"통합 분석 시작: {len(historical_data)}개 데이터")

        # 성능 추적 시작
        self.performance_tracker.start_tracking("unified_analysis")

        try:
            # 캐시 키 생성
            cache_key = self._create_cache_key(
                "unified_analysis_v2", len(historical_data)
            )
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            # 캐시 확인
            if cache_file.exists():
                try:
                    if not self._validate_cache_file(cache_file):
                        self.logger.warning(f"손상된 캐시 파일 삭제: {cache_file}")
                        cache_file.unlink()
                    else:
                        with open(cache_file, "rb") as f:
                            cached_result = pickle.load(f)
                            self.logger.info(f"캐시된 분석 결과 사용: {cache_file}")
                            self.performance_tracker.stop_tracking("unified_analysis")
                            return cached_result
                except Exception as e:
                    self.logger.warning(f"캐시 파일 손상으로 삭제: {cache_file} - {e}")
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass

            # 통합 분석 수행 (성능 최적화 엔진 활용)
            unified_results = self._perform_unified_analysis(historical_data)

            # 결과 캐싱
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(unified_results, f)
                self.logger.info(f"분석 결과 캐시 저장: {cache_file}")
            except Exception as e:
                self.logger.warning(f"캐시 저장 실패: {e}")

            return unified_results

        finally:
            self.performance_tracker.stop_tracking("unified_analysis")

    def _perform_unified_analysis(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """
        통합 분석 수행 (성능 최적화 적용)

        Args:
            historical_data: 과거 당첨 번호 데이터

        Returns:
            통합 분석 결과
        """
        # 병렬 분석 작업 정의
        analysis_tasks = [
            ("pattern_analysis", self._perform_pattern_analysis),
            ("distribution_analysis", self._perform_distribution_analysis),
            ("roi_analysis", self._perform_roi_analysis),
            ("cluster_analysis", self._perform_cluster_analysis),
            ("trend_analysis", self._perform_trend_analysis),
            ("three_digit_analysis", self._perform_three_digit_analysis),
        ]

        # 성능 최적화 엔진을 사용한 병렬 분석
        results = {}
        for task_name, task_func in analysis_tasks:
            try:
                self.performance_tracker.start_tracking(task_name)

                # 성능 최적화 엔진으로 실행
                result = self.performance_engine.execute(
                    task_func, historical_data, TaskType.DATA_PROCESSING
                )

                results[task_name] = result
                self.logger.info(f"✅ {task_name} 완료")

            except Exception as e:
                self.logger.error(f"❌ {task_name} 실패: {e}")
                results[task_name] = {}
            finally:
                self.performance_tracker.stop_tracking(task_name)

        # 결과 통합 및 메타데이터 추가
        unified_results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "data_count": len(historical_data),
            "analysis_version": "v2_unified_optimized",
            **results,
        }

        # 3자리 우선 예측 결과 추가
        if "three_digit_analysis" in results and results["three_digit_analysis"]:
            unified_results["three_digit_priority_predictions"] = (
                self._generate_three_digit_priority_predictions(
                    results["three_digit_analysis"]
                )
            )

        return unified_results

    def _perform_pattern_analysis(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """패턴 분석 수행"""
        return self.pattern_analyzer.run_all_analyses(historical_data)

    def _perform_distribution_analysis(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """분포 분석 수행"""
        distribution_pattern = self.distribution_analyzer.analyze(historical_data)
        return (
            distribution_pattern.__dict__
            if hasattr(distribution_pattern, "__dict__")
            else {}
        )

    def _perform_roi_analysis(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """ROI 분석 수행"""
        if self.roi_analyzer:
            return self.roi_analyzer.analyze(historical_data)
        return {}

    def _perform_cluster_analysis(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """클러스터 분석 수행"""
        if self.cluster_analyzer:
            return self.cluster_analyzer.analyze(historical_data)
        return {}

    def _perform_trend_analysis(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """트렌드 분석 수행"""
        if self.trend_analyzer:
            return self.trend_analyzer.analyze(historical_data)
        return {}

    def _perform_three_digit_analysis(
        self, historical_data: List[LotteryNumber]
    ) -> Dict[str, Any]:
        """3자리 패턴 분석 수행"""
        # PatternAnalyzer의 3자리 분석 기능 활용
        three_digit_results = self.pattern_analyzer.analyze_3digit_patterns(
            historical_data
        )

        # 3자리 확장 엔진 결과 추가
        if self.three_digit_engine:
            try:
                expansion_results = self.three_digit_engine.analyze_expansion_patterns(
                    historical_data
                )
                three_digit_results["expansion_analysis"] = expansion_results
            except Exception as e:
                self.logger.warning(f"3자리 확장 분석 실패: {e}")

        return three_digit_results

    def _generate_three_digit_priority_predictions(
        self, three_digit_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        3자리 우선 예측 결과 생성

        Args:
            three_digit_analysis: 3자리 분석 결과

        Returns:
            3자리 우선 예측 리스트
        """
        predictions = []

        # 상위 3자리 조합 추출
        top_candidates = three_digit_analysis.get("top_candidates", [])

        for candidate in top_candidates[:20]:  # 상위 20개
            prediction = {
                "three_digit_combo": candidate.get("combination", []),
                "confidence_score": candidate.get("composite_score", 0.0),
                "pattern_quality": candidate.get("pattern_quality_score", 0.0),
                "expansion_success_rate": candidate.get("expansion_success_rate", 0.0),
                "frequency_score": candidate.get("frequency_score", 0.0),
                "predicted_6digit_expansions": [],
            }

            # 6자리 확장 예측 추가
            if self.three_digit_engine:
                try:
                    expansions = self.three_digit_engine.expand_to_6digit(
                        tuple(candidate.get("combination", [])), method="hybrid"
                    )
                    prediction["predicted_6digit_expansions"] = expansions[
                        :5
                    ]  # 상위 5개
                except Exception as e:
                    self.logger.warning(f"6자리 확장 예측 실패: {e}")

            predictions.append(prediction)

        return predictions

    def analyze_number_combination(
        self, historical_data: List[LotteryNumber], target_numbers: List[int]
    ) -> ROIMetrics:
        """
        특정 번호 조합 분석

        Args:
            historical_data: 과거 당첨 번호 데이터
            target_numbers: 분석할 번호 조합

        Returns:
            ROI 메트릭스
        """
        if not self.roi_analyzer:
            raise ValueError("ROI 분석기가 초기화되지 않았습니다.")

        return self.roi_analyzer.analyze_number_combination(
            historical_data, target_numbers
        )

    def analyze_multiple_combinations(
        self, historical_data: List[LotteryNumber], combinations: List[List[int]]
    ) -> Dict[str, ROIMetrics]:
        """
        여러 번호 조합 분석

        Args:
            historical_data: 과거 당첨 번호 데이터
            combinations: 분석할 번호 조합들

        Returns:
            조합별 ROI 메트릭스
        """
        if not self.roi_analyzer:
            raise ValueError("ROI 분석기가 초기화되지 않았습니다.")

        return self.roi_analyzer.analyze_multiple_combinations(
            historical_data, combinations
        )

    def filter_combinations(
        self,
        combinations: List[List[int]],
        historical_data: Optional[List[LotteryNumber]] = None,
        filter_names: Optional[List[str]] = None,
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        번호 조합 필터링

        Args:
            combinations: 필터링할 번호 조합들
            historical_data: 과거 당첨 번호 데이터
            filter_names: 적용할 필터 이름들

        Returns:
            필터링된 조합들과 필터링 통계
        """
        if not self.pattern_filter:
            return combinations, {}

        # 성능 최적화 엔진을 사용한 필터링
        def filter_func(data):
            return self.pattern_filter.filter_combinations(
                data, historical_data, filter_names
            )

        return self.performance_engine.execute(
            filter_func, combinations, TaskType.DATA_PROCESSING
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        성능 통계 반환

        Returns:
            성능 통계 정보
        """
        stats = {
            "unified_analyzer_stats": self.performance_tracker.get_stats(),
            "performance_engine_stats": self.performance_engine.get_performance_stats(),
        }

        return stats

    def run_full_analysis(self, draw_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        전체 분석 수행 (하위 호환성)

        Args:
            draw_data: 로또 당첨 번호 데이터

        Returns:
            전체 분석 결과
        """
        return self.analyze(draw_data)

    def _extract_last_draw(self, draw_data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        마지막 당첨 번호 추출

        Args:
            draw_data: 로또 당첨 번호 데이터

        Returns:
            마지막 당첨 번호 정보
        """
        if not draw_data:
            return {}

        last_draw = draw_data[-1]
        return {
            "round_number": last_draw.round_number,
            "numbers": last_draw.numbers,
            "bonus_number": last_draw.bonus_number,
            "draw_date": last_draw.draw_date,
        }

    def _analyze_impl(
        self, historical_data: List[LotteryNumber], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        분석 구현 (BaseAnalyzer 인터페이스)

        Args:
            historical_data: 과거 당첨 번호 데이터

        Returns:
            분석 결과
        """
        return self.analyze(historical_data)

    def _make_serializable(self, obj: Any) -> Any:
        """
        객체를 직렬화 가능한 형태로 변환

        Args:
            obj: 변환할 객체

        Returns:
            직렬화 가능한 객체
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        else:
            return obj

    def save_analysis_results(self, unified_results: Dict[str, Any]) -> str:
        """
        분석 결과 저장

        Args:
            unified_results: 통합 분석 결과

        Returns:
            저장된 파일 경로
        """
        try:
            # 직렬화 가능한 형태로 변환
            serializable_results = self._make_serializable(unified_results)

            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unified_analysis_{timestamp}.json"

            # 결과 디렉토리 생성
            result_dir = Path("data/result/analysis")
            result_dir.mkdir(parents=True, exist_ok=True)

            # 파일 저장
            file_path = result_dir / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✅ 분석 결과 저장 완료: {file_path}")
            return str(file_path)

        except Exception as e:
            self.logger.error(f"❌ 분석 결과 저장 실패: {e}")
            raise

    def shutdown(self):
        """리소스 정리"""
        try:
            if hasattr(self, "performance_engine"):
                self.performance_engine.shutdown()
            self.logger.info("✅ 통합 분석기 리소스 정리 완료")
        except Exception as e:
            self.logger.warning(f"리소스 정리 중 오류: {e}")
