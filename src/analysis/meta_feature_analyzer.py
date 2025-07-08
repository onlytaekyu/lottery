"""
메타 특성 분석기
- 특성 중요도 분석 (SHAP/LIME)
- 특성 상호작용 분석
- 차원 축소 및 특성 선택
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import json

# 기존 시스템 import
from ..utils.unified_logging import get_logger
from ..shared.types import LotteryNumber
from ..analysis.base_analyzer import BaseAnalyzer

logger = get_logger(__name__)


class MetaFeatureAnalyzer(BaseAnalyzer):
    """
    메타 특성 분석기
    - 특성 중요도 분석
    - 특성 상호작용 탐지
    - 차원 축소
    - 특성 선택
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {}, "meta_feature")

        # 설정 초기화
        self.use_shap = self.config.get("use_shap", False)
        self.use_lime = self.config.get("use_lime", False)
        self.polynomial_degree = self.config.get("polynomial_degree", 2)
        self.pca_components = self.config.get("pca_components", 50)
        self.tsne_components = self.config.get("tsne_components", 2)
        self.select_k_best = self.config.get("select_k_best", 100)

        # 내부 상태
        self.feature_importance_scores = {}
        self.interaction_features = None
        self.dimension_reduction_models = {}

        logger.info("✅ MetaFeatureAnalyzer 초기화 완료")

    def _analyze_impl(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """메타 특성 분석 메인 로직"""
        try:
            logger.info(f"🔍 메타 특성 분석 시작: {len(data)}개 회차")

            # 데이터가 부족한 경우 기본 분석만 수행
            if len(data) < 50:
                logger.warning("데이터 부족으로 기본 분석만 수행")
                return self._basic_meta_analysis(data)

            # 1. 기본 특성 행렬 생성
            feature_matrix, target_vector = self._create_feature_matrix(data)

            # 2. 특성 중요도 분석
            importance_analysis = self._analyze_feature_importance(
                feature_matrix, target_vector
            )

            # 3. 특성 상호작용 분석
            interaction_analysis = self._analyze_feature_interactions(
                feature_matrix, target_vector
            )

            # 4. 차원 축소 분석
            dimension_reduction = self._perform_dimension_reduction(feature_matrix)

            # 5. 특성 선택
            feature_selection = self._perform_feature_selection(
                feature_matrix, target_vector
            )

            # 6. 메타 특성 통계
            meta_statistics = self._calculate_meta_statistics(
                feature_matrix, target_vector
            )

            result = {
                "importance_analysis": importance_analysis,
                "interaction_analysis": interaction_analysis,
                "dimension_reduction": dimension_reduction,
                "feature_selection": feature_selection,
                "meta_statistics": meta_statistics,
                "original_features": (
                    feature_matrix.shape[1] if feature_matrix is not None else 0
                ),
                "data_samples": len(data),
            }

            logger.info("✅ 메타 특성 분석 완료")
            return result

        except Exception as e:
            logger.error(f"메타 특성 분석 실패: {e}")
            return self._get_fallback_result()

    def _create_feature_matrix(
        self, data: List[LotteryNumber]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """기본 특성 행렬 생성"""
        try:
            features = []
            targets = []

            for i, draw in enumerate(data):
                # 기본 특성 추출
                feature_vector = self._extract_basic_features(draw)
                features.append(feature_vector)

                # 타겟 변수 (다음 회차 예측을 위한 간단한 타겟)
                if i < len(data) - 1:
                    next_draw = data[i + 1]
                    target = self._create_target_variable(draw, next_draw)
                    targets.append(target)

            # 마지막 샘플 제거 (타겟이 없음)
            if features:
                features = features[:-1]

            if len(features) == 0 or len(targets) == 0:
                logger.warning("특성 행렬 생성 실패: 데이터 부족")
                return None, None

            feature_matrix = np.array(features, dtype=np.float32)
            target_vector = np.array(targets, dtype=np.float32)

            logger.info(f"특성 행렬 생성 완료: {feature_matrix.shape}")
            return feature_matrix, target_vector

        except Exception as e:
            logger.error(f"특성 행렬 생성 실패: {e}")
            return None, None

    def _extract_basic_features(self, draw: LotteryNumber) -> List[float]:
        """기본 특성 추출"""
        try:
            features = []
            numbers = sorted(draw.numbers)

            # 기본 통계
            features.extend(
                [
                    np.sum(numbers),  # 합계
                    np.mean(numbers),  # 평균
                    np.std(numbers),  # 표준편차
                    np.min(numbers),  # 최솟값
                    np.max(numbers),  # 최댓값
                    np.median(numbers),  # 중앙값
                    numbers[-1] - numbers[0],  # 범위
                ]
            )

            # 간격 정보
            gaps = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
            features.extend(
                [
                    np.mean(gaps),  # 평균 간격
                    np.std(gaps),  # 간격 표준편차
                    np.max(gaps),  # 최대 간격
                    np.min(gaps),  # 최소 간격
                ]
            )

            # 홀짝 정보
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            even_count = sum(1 for n in numbers if n % 2 == 0)
            features.extend(
                [
                    odd_count,  # 홀수 개수
                    even_count,  # 짝수 개수
                    odd_count / len(numbers),  # 홀수 비율
                ]
            )

            # 구간 분포 (1-9, 10-18, 19-27, 28-36, 37-45)
            segments = [0, 0, 0, 0, 0]
            for num in numbers:
                if 1 <= num <= 9:
                    segments[0] += 1
                elif 10 <= num <= 18:
                    segments[1] += 1
                elif 19 <= num <= 27:
                    segments[2] += 1
                elif 28 <= num <= 36:
                    segments[3] += 1
                elif 37 <= num <= 45:
                    segments[4] += 1

            features.extend(segments)

            # 연속 번호 정보
            consecutive_count = 0
            for i in range(len(numbers) - 1):
                if numbers[i + 1] - numbers[i] == 1:
                    consecutive_count += 1
            features.append(consecutive_count)

            # 패턴 특성 (간단한 버전)
            features.extend(
                [
                    len(set(numbers)),  # 고유 번호 수 (항상 6이지만 일관성 위해)
                    sum(numbers) % 10,  # 합계의 일의 자리
                    np.var(numbers),  # 분산
                    np.sum(np.diff(numbers)),  # 총 간격 합
                ]
            )

            return features

        except Exception as e:
            logger.error(f"기본 특성 추출 실패: {e}")
            return [0.0] * 25  # 기본 특성 25개

    def _create_target_variable(
        self, current_draw: LotteryNumber, next_draw: LotteryNumber
    ) -> float:
        """타겟 변수 생성 (다음 회차와의 관계)"""
        try:
            # 간단한 타겟: 다음 회차와 현재 회차의 공통 번호 개수
            current_set = set(current_draw.numbers)
            next_set = set(next_draw.numbers)
            overlap_count = len(current_set & next_set)

            # 정규화 (0~1 범위)
            return overlap_count / 6.0

        except Exception as e:
            logger.error(f"타겟 변수 생성 실패: {e}")
            return 0.0

    def _analyze_feature_importance(
        self, feature_matrix: np.ndarray, target_vector: np.ndarray
    ) -> Dict[str, Any]:
        """특성 중요도 분석"""
        try:
            importance_result = {}

            # 1. Random Forest 기반 특성 중요도
            rf_importance = self._calculate_rf_importance(feature_matrix, target_vector)
            importance_result["random_forest"] = rf_importance

            # 2. 상관관계 기반 중요도
            correlation_importance = self._calculate_correlation_importance(
                feature_matrix, target_vector
            )
            importance_result["correlation"] = correlation_importance

            # 3. 상호정보 기반 중요도
            mutual_info_importance = self._calculate_mutual_info_importance(
                feature_matrix, target_vector
            )
            importance_result["mutual_info"] = mutual_info_importance

            # 4. SHAP 값 (선택적)
            if self.use_shap:
                shap_importance = self._calculate_shap_importance(
                    feature_matrix, target_vector
                )
                importance_result["shap"] = shap_importance

            # 5. LIME 설명 (선택적)
            if self.use_lime:
                lime_importance = self._calculate_lime_importance(
                    feature_matrix, target_vector
                )
                importance_result["lime"] = lime_importance

            # 6. 종합 중요도 점수
            combined_importance = self._combine_importance_scores(importance_result)
            importance_result["combined"] = combined_importance

            return importance_result

        except Exception as e:
            logger.error(f"특성 중요도 분석 실패: {e}")
            return {"error": str(e)}

    def _calculate_rf_importance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Random Forest 기반 특성 중요도"""
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)

            importance_scores = {}
            for i, importance in enumerate(rf.feature_importances_):
                importance_scores[f"feature_{i}"] = float(importance)

            return importance_scores

        except Exception as e:
            logger.error(f"RF 중요도 계산 실패: {e}")
            return {}

    def _calculate_correlation_importance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """상관관계 기반 중요도"""
        try:
            importance_scores = {}

            for i in range(X.shape[1]):
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                # NaN 처리
                if np.isnan(correlation):
                    correlation = 0.0
                importance_scores[f"feature_{i}"] = float(abs(correlation))

            return importance_scores

        except Exception as e:
            logger.error(f"상관관계 중요도 계산 실패: {e}")
            return {}

    def _calculate_mutual_info_importance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """상호정보 기반 중요도"""
        try:
            mutual_info_scores = mutual_info_regression(X, y)

            importance_scores = {}
            for i, score in enumerate(mutual_info_scores):
                importance_scores[f"feature_{i}"] = float(score)

            return importance_scores

        except Exception as e:
            logger.error(f"상호정보 중요도 계산 실패: {e}")
            return {}

    def _calculate_shap_importance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """SHAP 값 기반 중요도"""
        try:
            import shap

            # 모델 훈련
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)

            # SHAP 값 계산
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)

            # 평균 절댓값으로 중요도 계산
            importance_scores = {}
            for i in range(X.shape[1]):
                importance_scores[f"feature_{i}"] = float(
                    np.mean(np.abs(shap_values[:, i]))
                )

            return importance_scores

        except ImportError:
            logger.warning("SHAP 라이브러리 없음")
            return {}
        except Exception as e:
            logger.error(f"SHAP 중요도 계산 실패: {e}")
            return {}

    def _calculate_lime_importance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """LIME 기반 중요도"""
        try:
            from lime import lime_tabular

            # 모델 훈련
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)

            # LIME 설명기 생성
            explainer = lime_tabular.LimeTabularExplainer(
                X,
                mode="regression",
                feature_names=[f"feature_{i}" for i in range(X.shape[1])],
            )

            # 샘플 설명
            importance_scores = {}
            sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)

            for idx in sample_indices:
                explanation = explainer.explain_instance(X[idx], rf.predict)
                for feature_name, importance in explanation.as_list():
                    if feature_name not in importance_scores:
                        importance_scores[feature_name] = []
                    importance_scores[feature_name].append(abs(importance))

            # 평균 중요도 계산
            for feature_name in importance_scores:
                importance_scores[feature_name] = float(
                    np.mean(importance_scores[feature_name])
                )

            return importance_scores

        except ImportError:
            logger.warning("LIME 라이브러리 없음")
            return {}
        except Exception as e:
            logger.error(f"LIME 중요도 계산 실패: {e}")
            return {}

    def _combine_importance_scores(
        self, importance_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """여러 중요도 점수를 종합"""
        try:
            combined_scores = {}

            # 사용 가능한 중요도 방법들
            methods = ["random_forest", "correlation", "mutual_info", "shap", "lime"]
            available_methods = [
                m for m in methods if m in importance_result and importance_result[m]
            ]

            if not available_methods:
                return {}

            # 특성 이름 추출
            feature_names = set()
            for method in available_methods:
                feature_names.update(importance_result[method].keys())

            # 각 특성별 종합 점수 계산
            for feature_name in feature_names:
                scores = []
                for method in available_methods:
                    if feature_name in importance_result[method]:
                        scores.append(importance_result[method][feature_name])

                if scores:
                    combined_scores[feature_name] = float(np.mean(scores))

            return combined_scores

        except Exception as e:
            logger.error(f"중요도 점수 종합 실패: {e}")
            return {}

    def _analyze_feature_interactions(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """특성 상호작용 분석"""
        try:
            interaction_result = {}

            # 1. 다항식 특성 생성
            polynomial_features = self._generate_polynomial_features(X)
            interaction_result["polynomial"] = polynomial_features

            # 2. 특성 간 상관관계 행렬
            correlation_matrix = self._calculate_feature_correlations(X)
            interaction_result["correlation_matrix"] = correlation_matrix

            # 3. 상호작용 중요도 (다항식 특성 기반)
            if polynomial_features["success"]:
                interaction_importance = self._calculate_interaction_importance(
                    polynomial_features["features"], y
                )
                interaction_result["interaction_importance"] = interaction_importance

            return interaction_result

        except Exception as e:
            logger.error(f"특성 상호작용 분석 실패: {e}")
            return {"error": str(e)}

    def _generate_polynomial_features(self, X: np.ndarray) -> Dict[str, Any]:
        """다항식 특성 생성"""
        try:
            # 메모리 사용량 고려하여 특성 수 제한
            max_features = min(X.shape[1], 15)  # 최대 15개 특성만 사용
            X_subset = X[:, :max_features]

            poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
            X_poly = poly.fit_transform(X_subset)

            # 원본 특성 제거 (상호작용 항만 추출)
            X_interactions = X_poly[:, max_features:]

            result = {
                "success": True,
                "features": X_interactions,
                "feature_names": poly.get_feature_names_out()[:max_features],
                "interaction_features": poly.get_feature_names_out()[max_features:],
                "original_features": max_features,
                "interaction_count": X_interactions.shape[1],
            }

            logger.info(
                f"다항식 특성 생성 완료: {X_interactions.shape[1]}개 상호작용 특성"
            )
            return result

        except Exception as e:
            logger.error(f"다항식 특성 생성 실패: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_feature_correlations(self, X: np.ndarray) -> List[List[float]]:
        """특성 간 상관관계 행렬"""
        try:
            correlation_matrix = np.corrcoef(X, rowvar=False)
            # NaN 처리
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            return correlation_matrix.tolist()

        except Exception as e:
            logger.error(f"상관관계 행렬 계산 실패: {e}")
            return [[0.0] * X.shape[1] for _ in range(X.shape[1])]

    def _calculate_interaction_importance(
        self, X_interactions: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """상호작용 특성 중요도"""
        try:
            # Random Forest로 상호작용 중요도 계산
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_interactions, y)

            importance_scores = {}
            for i, importance in enumerate(rf.feature_importances_):
                importance_scores[f"interaction_{i}"] = float(importance)

            return importance_scores

        except Exception as e:
            logger.error(f"상호작용 중요도 계산 실패: {e}")
            return {}

    def _perform_dimension_reduction(self, X: np.ndarray) -> Dict[str, Any]:
        """차원 축소 수행"""
        try:
            reduction_result = {}

            # 1. PCA
            pca_result = self._perform_pca(X)
            reduction_result["pca"] = pca_result

            # 2. t-SNE (데이터가 충분한 경우만)
            if X.shape[0] > 30:
                tsne_result = self._perform_tsne(X)
                reduction_result["tsne"] = tsne_result

            # 3. 특성 선택을 통한 차원 축소
            feature_selection_result = self._perform_univariate_selection(X)
            reduction_result["feature_selection"] = feature_selection_result

            return reduction_result

        except Exception as e:
            logger.error(f"차원 축소 실패: {e}")
            return {"error": str(e)}

    def _perform_pca(self, X: np.ndarray) -> Dict[str, Any]:
        """PCA 수행"""
        try:
            # 컴포넌트 수 조정
            n_components = min(self.pca_components, X.shape[1], X.shape[0] - 1)

            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)

            result = {
                "success": True,
                "transformed_features": X_pca.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance_ratio": np.cumsum(
                    pca.explained_variance_ratio_
                ).tolist(),
                "n_components": n_components,
                "total_variance_explained": float(
                    np.sum(pca.explained_variance_ratio_)
                ),
            }

            # 모델 저장
            self.dimension_reduction_models["pca"] = pca

            return result

        except Exception as e:
            logger.error(f"PCA 수행 실패: {e}")
            return {"success": False, "error": str(e)}

    def _perform_tsne(self, X: np.ndarray) -> Dict[str, Any]:
        """t-SNE 수행"""
        try:
            # 퍼플렉시티 조정
            perplexity = min(30, X.shape[0] // 4)

            tsne = TSNE(
                n_components=self.tsne_components,
                perplexity=perplexity,
                random_state=42,
            )
            X_tsne = tsne.fit_transform(X)

            result = {
                "success": True,
                "transformed_features": X_tsne.tolist(),
                "n_components": self.tsne_components,
                "perplexity": perplexity,
            }

            return result

        except Exception as e:
            logger.error(f"t-SNE 수행 실패: {e}")
            return {"success": False, "error": str(e)}

    def _perform_univariate_selection(self, X: np.ndarray) -> Dict[str, Any]:
        """단변량 특성 선택"""
        try:
            # k 값 조정
            k = min(self.select_k_best, X.shape[1])

            # F-score 기반 선택
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(
                X, np.random.rand(X.shape[0])
            )  # 더미 타겟

            result = {
                "success": True,
                "selected_features": X_selected.tolist(),
                "selected_indices": selector.get_support(indices=True).tolist(),
                "scores": selector.scores_.tolist(),
                "k": k,
            }

            return result

        except Exception as e:
            logger.error(f"단변량 특성 선택 실패: {e}")
            return {"success": False, "error": str(e)}

    def _perform_feature_selection(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """특성 선택 수행"""
        try:
            selection_result = {}

            # 1. 분산 기반 선택
            variance_selection = self._variance_based_selection(X)
            selection_result["variance_based"] = variance_selection

            # 2. 상관관계 기반 선택
            correlation_selection = self._correlation_based_selection(X, y)
            selection_result["correlation_based"] = correlation_selection

            # 3. 모델 기반 선택
            model_selection = self._model_based_selection(X, y)
            selection_result["model_based"] = model_selection

            return selection_result

        except Exception as e:
            logger.error(f"특성 선택 실패: {e}")
            return {"error": str(e)}

    def _variance_based_selection(self, X: np.ndarray) -> Dict[str, Any]:
        """분산 기반 특성 선택"""
        try:
            from sklearn.feature_selection import VarianceThreshold

            # 분산 임계값 설정
            threshold = 0.01
            selector = VarianceThreshold(threshold=threshold)
            X_selected = selector.fit_transform(X)

            result = {
                "success": True,
                "selected_features": X_selected.tolist(),
                "selected_indices": selector.get_support(indices=True).tolist(),
                "threshold": threshold,
                "selected_count": X_selected.shape[1],
            }

            return result

        except Exception as e:
            logger.error(f"분산 기반 선택 실패: {e}")
            return {"success": False, "error": str(e)}

    def _correlation_based_selection(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """상관관계 기반 특성 선택"""
        try:
            # 타겟과의 상관관계 계산
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)

            # 상위 특성 선택
            top_k = min(20, len(correlations))
            top_indices = np.argsort(correlations)[-top_k:]

            result = {
                "success": True,
                "correlations": correlations,
                "top_indices": top_indices.tolist(),
                "top_correlations": [correlations[i] for i in top_indices],
                "selected_count": top_k,
            }

            return result

        except Exception as e:
            logger.error(f"상관관계 기반 선택 실패: {e}")
            return {"success": False, "error": str(e)}

    def _model_based_selection(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """모델 기반 특성 선택"""
        try:
            from sklearn.feature_selection import SelectFromModel

            # Random Forest 기반 선택
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = SelectFromModel(rf, threshold="median")
            X_selected = selector.fit_transform(X, y)

            result = {
                "success": True,
                "selected_features": X_selected.tolist(),
                "selected_indices": selector.get_support(indices=True).tolist(),
                "selected_count": X_selected.shape[1],
                "feature_importances": rf.feature_importances_.tolist(),
            }

            return result

        except Exception as e:
            logger.error(f"모델 기반 선택 실패: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_meta_statistics(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """메타 통계 계산"""
        try:
            stats = {
                "data_shape": X.shape,
                "feature_statistics": {
                    "mean": np.mean(X, axis=0).tolist(),
                    "std": np.std(X, axis=0).tolist(),
                    "min": np.min(X, axis=0).tolist(),
                    "max": np.max(X, axis=0).tolist(),
                    "median": np.median(X, axis=0).tolist(),
                },
                "target_statistics": {
                    "mean": float(np.mean(y)),
                    "std": float(np.std(y)),
                    "min": float(np.min(y)),
                    "max": float(np.max(y)),
                    "median": float(np.median(y)),
                },
                "data_quality": {
                    "missing_values": int(np.sum(np.isnan(X))),
                    "infinite_values": int(np.sum(np.isinf(X))),
                    "zero_values": int(np.sum(X == 0)),
                    "constant_features": int(np.sum(np.std(X, axis=0) == 0)),
                },
            }

            return stats

        except Exception as e:
            logger.error(f"메타 통계 계산 실패: {e}")
            return {"error": str(e)}

    def _basic_meta_analysis(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """기본 메타 분석 (데이터 부족 시)"""
        try:
            basic_stats = {
                "data_count": len(data),
                "status": "basic_analysis_only",
                "reason": "insufficient_data",
                "min_required": 50,
                "basic_statistics": {},
            }

            if data:
                # 기본 통계만 계산
                all_numbers = []
                for draw in data:
                    all_numbers.extend(draw.numbers)

                basic_stats["basic_statistics"] = {
                    "number_count": len(all_numbers),
                    "unique_numbers": len(set(all_numbers)),
                    "mean_number": float(np.mean(all_numbers)),
                    "std_number": float(np.std(all_numbers)),
                    "min_number": int(np.min(all_numbers)),
                    "max_number": int(np.max(all_numbers)),
                }

            return basic_stats

        except Exception as e:
            logger.error(f"기본 메타 분석 실패: {e}")
            return {"error": str(e)}

    def _get_fallback_result(self) -> Dict[str, Any]:
        """폴백 결과 반환"""
        return {
            "importance_analysis": {"error": "analysis_failed"},
            "interaction_analysis": {"error": "analysis_failed"},
            "dimension_reduction": {"error": "analysis_failed"},
            "feature_selection": {"error": "analysis_failed"},
            "meta_statistics": {"error": "analysis_failed"},
            "original_features": 0,
            "data_samples": 0,
        }

    def get_meta_features_vector(self, meta_analysis: Dict[str, Any]) -> np.ndarray:
        """메타 분석 결과를 벡터로 변환"""
        try:
            features = []

            # 1. 특성 중요도 요약
            importance_analysis = meta_analysis.get("importance_analysis", {})
            if "combined" in importance_analysis:
                importance_values = list(importance_analysis["combined"].values())
                if importance_values:
                    features.extend(
                        [
                            np.mean(importance_values),
                            np.std(importance_values),
                            np.max(importance_values),
                            np.min(importance_values),
                        ]
                    )
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # 2. 차원 축소 결과
            dimension_reduction = meta_analysis.get("dimension_reduction", {})
            if "pca" in dimension_reduction and dimension_reduction["pca"].get(
                "success", False
            ):
                pca_result = dimension_reduction["pca"]
                features.extend(
                    [
                        pca_result.get("total_variance_explained", 0.0),
                        pca_result.get("n_components", 0),
                        len(pca_result.get("explained_variance_ratio", [])),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0])

            # 3. 특성 선택 결과
            feature_selection = meta_analysis.get("feature_selection", {})
            if "model_based" in feature_selection and feature_selection[
                "model_based"
            ].get("success", False):
                model_result = feature_selection["model_based"]
                features.extend(
                    [
                        model_result.get("selected_count", 0),
                        meta_analysis.get("original_features", 0),
                        model_result.get("selected_count", 0)
                        / max(meta_analysis.get("original_features", 1), 1),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0])

            # 4. 메타 통계 요약
            meta_statistics = meta_analysis.get("meta_statistics", {})
            if "data_quality" in meta_statistics:
                data_quality = meta_statistics["data_quality"]
                features.extend(
                    [
                        data_quality.get("missing_values", 0),
                        data_quality.get("infinite_values", 0),
                        data_quality.get("zero_values", 0),
                        data_quality.get("constant_features", 0),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # 5. 추가 메타 특성
            features.extend(
                [
                    meta_analysis.get("data_samples", 0),
                    meta_analysis.get("original_features", 0),
                    len(importance_analysis.get("combined", {})),
                    1.0 if "error" not in meta_analysis else 0.0,  # 성공 여부
                ]
            )

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"메타 특성 벡터 변환 실패: {e}")
            return np.zeros(18, dtype=np.float32)  # 기본 18차원 벡터

    def save_meta_analysis_results(
        self, results: Dict[str, Any], filename: str = "meta_analysis_results.json"
    ) -> bool:
        """메타 분석 결과 저장"""
        try:
            from pathlib import Path

            # 결과 디렉토리 생성
            results_dir = Path("data/analysis_results")
            results_dir.mkdir(parents=True, exist_ok=True)

            # 결과 저장
            output_path = results_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"메타 분석 결과 저장 완료: {output_path}")
            return True

        except Exception as e:
            logger.error(f"메타 분석 결과 저장 실패: {e}")
            return False
