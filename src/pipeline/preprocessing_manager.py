"""
전처리 매니저

모든 전처리 파이프라인을 통합 관리:
- Phase 1: 즉시 적용 가능한 핵심 최적화
- Phase 2: 성능 향상을 위한 고급 특성 엔지니어링
- 모델별 특화 전처리 조정
- 전처리 결과 캐싱 및 성능 모니터링
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
from datetime import datetime

from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_config
from ..utils.memory_manager import get_memory_manager
from ..utils.unified_performance import performance_monitor
from ..utils.cache_paths import get_cache_dir

from .advanced_preprocessing_pipeline import AdvancedPreprocessingPipeline
from .feature_engineering_pipeline import FeatureEngineeringPipeline
from .model_specific_preprocessors import create_model_preprocessor

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """전처리 결과"""

    X_processed: np.ndarray
    feature_names: List[str]
    preprocessing_stats: Dict[str, Any]
    model_type: str
    processing_time: float
    cache_key: Optional[str] = None


class PreprocessingManager:
    """전처리 매니저"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = get_config("main") if config is None else get_config("main")
        self.logger = get_logger(__name__)
        self.memory_manager = get_memory_manager()

        # 파이프라인 컴포넌트들
        self.advanced_pipeline = AdvancedPreprocessingPipeline(config)
        self.feature_engineering_pipeline = FeatureEngineeringPipeline(config)

        # 모델별 전처리기 캐시
        self.model_preprocessors = {}

        # 캐시 설정
        self.cache_dir = get_cache_dir() / "preprocessing"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_caching = self.config.get("preprocessing", {}).get(
            "enable_caching", True
        )

        # 성능 통계
        self.performance_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "processing_times": [],
            "feature_reduction_ratios": [],
        }

        self.logger.info("전처리 매니저 초기화 완료")

    def preprocess_for_model(
        self,
        X: np.ndarray,
        feature_names: List[str],
        model_type: str,
        y: Optional[np.ndarray] = None,
        phase: str = "both",  # "phase1", "phase2", "both"
        use_cache: bool = True,
    ) -> PreprocessingResult:
        """
        모델별 전처리 실행

        Args:
            X: 특성 행렬
            feature_names: 특성 이름 리스트
            model_type: 모델 타입 ("lightgbm", "autoencoder", "tcn", "random_forest")
            y: 타겟 변수 (선택적)
            phase: 적용할 Phase ("phase1", "phase2", "both")
            use_cache: 캐시 사용 여부

        Returns:
            PreprocessingResult: 전처리 결과
        """
        start_time = datetime.now()

        self.logger.info(f"모델별 전처리 시작: {model_type}, {X.shape}, Phase: {phase}")

        # 캐시 확인
        cache_key = None
        if use_cache and self.enable_caching:
            cache_key = self._generate_cache_key(X, feature_names, model_type, phase)
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats["cache_hits"] += 1
                self.logger.info(f"캐시에서 전처리 결과 로드: {cache_key}")
                return cached_result

        # 전처리 실행
        X_processed = X.copy()
        current_feature_names = feature_names.copy()
        combined_stats = {"original_shape": X.shape}

        # Phase 1: 핵심 최적화
        if phase in ["phase1", "both"]:
            X_processed, current_feature_names, phase1_stats = (
                self.advanced_pipeline.fit_transform(
                    X_processed, current_feature_names, model_type
                )
            )
            combined_stats["phase1"] = phase1_stats
            self.logger.info(f"Phase 1 완료: {X.shape} → {X_processed.shape}")

        # Phase 2: 특성 엔지니어링 (선택적)
        if phase in ["phase2", "both"]:
            X_processed, current_feature_names, phase2_stats = (
                self.feature_engineering_pipeline.fit_transform(
                    X_processed, current_feature_names, y, model_type
                )
            )
            combined_stats["phase2"] = phase2_stats
            self.logger.info(f"Phase 2 완료: 특성 엔지니어링 적용")

        # 모델별 특화 전처리 (최종 단계)
        model_preprocessor = self._get_model_preprocessor(model_type)
        if model_preprocessor is not None:
            X_final = model_preprocessor.fit_transform(
                X_processed, y, current_feature_names
            )
            current_feature_names = (
                model_preprocessor.get_feature_names_out() or current_feature_names
            )
            combined_stats["model_specific"] = (
                model_preprocessor.get_preprocessing_stats()
            )
        else:
            X_final = X_processed

        # 최종 통계 계산
        processing_time = (datetime.now() - start_time).total_seconds()
        combined_stats.update(
            {
                "final_shape": X_final.shape,
                "total_feature_reduction": 1 - (X_final.shape[1] / X.shape[1]),
                "processing_time": processing_time,
                "model_type": model_type,
                "phase": phase,
            }
        )

        # 결과 생성
        result = PreprocessingResult(
            X_processed=X_final,
            feature_names=current_feature_names,
            preprocessing_stats=combined_stats,
            model_type=model_type,
            processing_time=processing_time,
            cache_key=cache_key,
        )

        # 캐시 저장
        if use_cache and self.enable_caching and cache_key:
            self._save_to_cache(cache_key, result)

        # 성능 통계 업데이트
        self._update_performance_stats(result)

        self.logger.info(
            f"전처리 완료: {X.shape} → {X_final.shape}, 시간: {processing_time:.2f}초"
        )
        return result

    def preprocess_batch(
        self,
        datasets: List[
            Tuple[np.ndarray, List[str], str]
        ],  # [(X, feature_names, model_type), ...]
        y: Optional[np.ndarray] = None,
        phase: str = "both",
        parallel: bool = True,
    ) -> List[PreprocessingResult]:
        """배치 전처리"""
        self.logger.info(f"배치 전처리 시작: {len(datasets)}개 데이터셋")

        results = []

        if parallel and len(datasets) > 1:
            # 병렬 처리 (향후 구현)
            self.logger.info("병렬 처리는 향후 구현 예정, 순차 처리로 대체")

        # 순차 처리
        for i, (X, feature_names, model_type) in enumerate(datasets):
            self.logger.info(f"배치 처리 중: {i+1}/{len(datasets)} - {model_type}")
            result = self.preprocess_for_model(X, feature_names, model_type, y, phase)
            results.append(result)

        self.logger.info(f"배치 전처리 완료: {len(results)}개 결과")
        return results

    def get_preprocessing_recommendations(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """전처리 권장사항 분석"""
        self.logger.info("전처리 권장사항 분석 중...")

        recommendations = {
            "data_characteristics": {},
            "recommended_phases": [],
            "model_specific_notes": {},
            "optimization_suggestions": [],
        }

        # 데이터 특성 분석
        n_samples, n_features = X.shape

        recommendations["data_characteristics"] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_density": np.mean(X != 0),
            "missing_ratio": np.mean(np.isnan(X)) if np.any(np.isnan(X)) else 0.0,
            "outlier_ratio": self._estimate_outlier_ratio(X),
            "feature_correlation": self._analyze_feature_correlation(X),
        }

        # Phase 권장사항
        if n_features > 80:
            recommendations["recommended_phases"].append("phase1")
            recommendations["optimization_suggestions"].append(
                f"특성 수({n_features})가 많아 Phase 1 최적화 권장"
            )

        if n_samples > 1000:
            recommendations["recommended_phases"].append("phase2")
            recommendations["optimization_suggestions"].append(
                f"샘플 수({n_samples})가 충분하여 Phase 2 특성 엔지니어링 권장"
            )

        # 모델별 권장사항
        recommendations["model_specific_notes"] = {
            "lightgbm": "RobustScaler + 특성 선택(70개) 적용",
            "autoencoder": "MinMax(0,1) + PCA 차원축소 적용",
            "tcn": "StandardScaler + 시계열 윈도우 적용",
            "random_forest": "원본 데이터 유지 (스케일링 불필요)",
        }

        return recommendations

    def _get_model_preprocessor(self, model_type: str):
        """모델별 전처리기 가져오기"""
        if model_type not in self.model_preprocessors:
            try:
                self.model_preprocessors[model_type] = create_model_preprocessor(
                    model_type, self.config
                )
            except ValueError as e:
                self.logger.warning(f"모델 전처리기 생성 실패: {e}")
                return None

        return self.model_preprocessors[model_type]

    def _generate_cache_key(
        self, X: np.ndarray, feature_names: List[str], model_type: str, phase: str
    ) -> str:
        """캐시 키 생성"""
        import hashlib

        # 데이터 해시
        data_hash = hashlib.md5(X.tobytes()).hexdigest()[:8]

        # 특성 이름 해시
        features_hash = hashlib.md5("".join(feature_names).encode()).hexdigest()[:8]

        # 캐시 키 조합
        cache_key = f"preprocess_{model_type}_{phase}_{data_hash}_{features_hash}"

        return cache_key

    def _load_from_cache(self, cache_key: str) -> Optional[PreprocessingResult]:
        """캐시에서 로드"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                return result
            except Exception as e:
                self.logger.warning(f"캐시 로드 실패: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: PreprocessingResult):
        """캐시에 저장"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            self.logger.warning(f"캐시 저장 실패: {e}")

    def _update_performance_stats(self, result: PreprocessingResult):
        """성능 통계 업데이트"""
        self.performance_stats["total_processed"] += 1
        self.performance_stats["processing_times"].append(result.processing_time)

        reduction_ratio = result.preprocessing_stats.get("total_feature_reduction", 0)
        self.performance_stats["feature_reduction_ratios"].append(reduction_ratio)

    def _estimate_outlier_ratio(self, X: np.ndarray) -> float:
        """이상치 비율 추정"""
        try:
            from scipy import stats

            outlier_count = 0
            total_values = 0

            for i in range(min(10, X.shape[1])):  # 최대 10개 특성만 샘플링
                column = X[:, i]
                z_scores = np.abs(stats.zscore(column))
                outlier_count += np.sum(z_scores > 2.5)
                total_values += len(column)

            return outlier_count / total_values if total_values > 0 else 0.0
        except:
            return 0.0

    def _analyze_feature_correlation(self, X: np.ndarray) -> Dict[str, float]:
        """특성 상관관계 분석"""
        try:
            if X.shape[1] > 1:
                corr_matrix = np.corrcoef(X.T)
                upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

                return {
                    "mean_correlation": float(np.mean(np.abs(upper_tri))),
                    "max_correlation": float(np.max(np.abs(upper_tri))),
                    "high_correlation_pairs": int(np.sum(np.abs(upper_tri) > 0.9)),
                }
            else:
                return {
                    "mean_correlation": 0.0,
                    "max_correlation": 0.0,
                    "high_correlation_pairs": 0,
                }
        except:
            return {
                "mean_correlation": 0.0,
                "max_correlation": 0.0,
                "high_correlation_pairs": 0,
            }

    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        stats = self.performance_stats

        report = {
            "총_처리_건수": stats["total_processed"],
            "캐시_적중률": stats["cache_hits"] / max(stats["total_processed"], 1),
            "평균_처리_시간": (
                np.mean(stats["processing_times"]) if stats["processing_times"] else 0
            ),
            "평균_특성_감소율": (
                np.mean(stats["feature_reduction_ratios"])
                if stats["feature_reduction_ratios"]
                else 0
            ),
            "처리_시간_분포": {
                "최소": (
                    np.min(stats["processing_times"])
                    if stats["processing_times"]
                    else 0
                ),
                "최대": (
                    np.max(stats["processing_times"])
                    if stats["processing_times"]
                    else 0
                ),
                "표준편차": (
                    np.std(stats["processing_times"])
                    if stats["processing_times"]
                    else 0
                ),
            },
        }

        return report

    def clear_cache(self):
        """캐시 정리"""
        try:
            import shutil

            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("전처리 캐시 정리 완료")
        except Exception as e:
            self.logger.error(f"캐시 정리 실패: {e}")

    def save_performance_report(self, save_path: Optional[Path] = None):
        """성능 리포트 저장"""
        if save_path is None:
            save_path = self.cache_dir / "preprocessing_performance_report.json"

        report = self.get_performance_report()

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"성능 리포트 저장: {save_path}")
        except Exception as e:
            self.logger.error(f"성능 리포트 저장 실패: {e}")
