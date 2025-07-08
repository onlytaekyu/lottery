"""
성능 분석기 (Performance Analyzer)

로또 예측 모델의 성능을 분석하고 최적화하는 모듈입니다.
"""

import numpy as np
from typing import Dict, List, Any
from collections import Counter, defaultdict
from datetime import datetime

from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)


class PerformanceAnalyzer:
    """성능 분석기"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        성능 분석기 초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # 분석 결과 저장
        self.frequency_analysis = {}
        self.pattern_analysis = {}
        self.cluster_analysis = {}
        self.trend_analysis = {}
        self.roi_analysis = {}
        
        # 성능 메트릭
        self.performance_metrics = {
            "hit_rates": {},
            "roi_metrics": {},
            "accuracy_scores": {},
            "consistency_scores": {},
        }
        
        self.logger.info("성능 분석기 초기화 완료")

    def perform_comprehensive_analysis(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        종합 성능 분석 수행

        Args:
            data: 로또 번호 데이터

        Returns:
            분석 결과
        """
        self.logger.info(f"종합 성능 분석 시작: {len(data)}개 데이터")

        try:
            # 각종 분석 수행
            self._analyze_frequency(data)
            self._analyze_patterns(data)
            self._analyze_clusters(data)
            self._analyze_trends(data)
            self._analyze_roi(data)
            self._learn_grade_optimization_strategies(data)

            # 결과 종합
            analysis_results = {
                "frequency_analysis": self.frequency_analysis,
                "pattern_analysis": self.pattern_analysis,
                "cluster_analysis": self.cluster_analysis,
                "trend_analysis": self.trend_analysis,
                "roi_analysis": self.roi_analysis,
                "performance_metrics": self.performance_metrics,
                "analyzed_data_count": len(data),
                "analysis_timestamp": datetime.now().isoformat(),
            }

            self.logger.info("종합 성능 분석 완료")
            return analysis_results

        except Exception as e:
            self.logger.error(f"종합 성능 분석 실패: {e}")
            return {"error": str(e)}

    def _analyze_frequency(self, data: List[LotteryNumber]) -> None:
        """빈도 분석"""
        try:
            self.logger.info("빈도 분석 시작")

            # 번호별 출현 빈도
            number_freq = Counter()
            position_freq = defaultdict(Counter)
            
            # 최근 데이터에 가중치 적용
            total_draws = len(data)
            
            for i, draw in enumerate(data):
                # 최근 데이터일수록 높은 가중치
                weight = 1.0 + (i / total_draws) * 0.5
                
                for pos, number in enumerate(draw.numbers):
                    number_freq[number] += weight
                    position_freq[pos][number] += weight

            # 통계 계산
            freq_stats = {
                "most_common": number_freq.most_common(10),
                "least_common": number_freq.most_common()[-10:],
                "average_frequency": np.mean(list(number_freq.values())),
                "frequency_std": np.std(list(number_freq.values())),
                "position_preferences": {
                    pos: dict(pos_counter.most_common(5))
                    for pos, pos_counter in position_freq.items()
                },
            }

            # 핫/콜드 번호 분류
            freq_values = list(number_freq.values())
            hot_threshold = np.percentile(freq_values, 75)
            cold_threshold = np.percentile(freq_values, 25)

            hot_numbers = [num for num, freq in number_freq.items() if freq >= hot_threshold]
            cold_numbers = [num for num, freq in number_freq.items() if freq <= cold_threshold]

            self.frequency_analysis = {
                "statistics": freq_stats,
                "hot_numbers": hot_numbers,
                "cold_numbers": cold_numbers,
                "number_frequencies": dict(number_freq),
                "analysis_date": datetime.now().isoformat(),
            }

            self.logger.info(f"빈도 분석 완료: 핫 번호 {len(hot_numbers)}개, 콜드 번호 {len(cold_numbers)}개")

        except Exception as e:
            self.logger.error(f"빈도 분석 실패: {e}")
            self.frequency_analysis = {"error": str(e)}

    def _analyze_patterns(self, data: List[LotteryNumber]) -> None:
        """패턴 분석"""
        try:
            self.logger.info("패턴 분석 시작")

            # 연속 번호 패턴
            consecutive_patterns = []
            # 홀짝 패턴
            odd_even_patterns = []
            # 구간별 분포 패턴
            range_patterns = []
            # 합계 패턴
            sum_patterns = []

            for draw in data:
                numbers = sorted(draw.numbers)
                
                # 연속 번호 카운트
                consecutive_count = 0
                for i in range(len(numbers) - 1):
                    if numbers[i + 1] - numbers[i] == 1:
                        consecutive_count += 1
                consecutive_patterns.append(consecutive_count)

                # 홀짝 분포
                odd_count = sum(1 for num in numbers if num % 2 == 1)
                odd_even_patterns.append(odd_count)

                # 구간별 분포 (1-15, 16-30, 31-45)
                range_dist = [0, 0, 0]
                for num in numbers:
                    if num <= 15:
                        range_dist[0] += 1
                    elif num <= 30:
                        range_dist[1] += 1
                    else:
                        range_dist[2] += 1
                range_patterns.append(range_dist)

                # 번호 합계
                sum_patterns.append(sum(numbers))

            # 패턴 통계
            pattern_stats = {
                "consecutive_numbers": {
                    "average": np.mean(consecutive_patterns),
                    "std": np.std(consecutive_patterns),
                    "distribution": dict(Counter(consecutive_patterns)),
                },
                "odd_even_distribution": {
                    "average_odd_count": np.mean(odd_even_patterns),
                    "std": np.std(odd_even_patterns),
                    "distribution": dict(Counter(odd_even_patterns)),
                },
                "range_distribution": {
                    "average_per_range": np.mean(range_patterns, axis=0).tolist(),
                    "std_per_range": np.std(range_patterns, axis=0).tolist(),
                },
                "sum_statistics": {
                    "average_sum": np.mean(sum_patterns),
                    "std_sum": np.std(sum_patterns),
                    "min_sum": min(sum_patterns),
                    "max_sum": max(sum_patterns),
                },
            }

            self.pattern_analysis = {
                "statistics": pattern_stats,
                "analysis_date": datetime.now().isoformat(),
            }

            self.logger.info("패턴 분석 완료")

        except Exception as e:
            self.logger.error(f"패턴 분석 실패: {e}")
            self.pattern_analysis = {"error": str(e)}

    def _analyze_clusters(self, data: List[LotteryNumber]) -> None:
        """클러스터 분석"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            self.logger.info("클러스터 분석 시작")

            # 특성 벡터 생성
            features = []
            for draw in data:
                numbers = sorted(draw.numbers)
                
                # 다양한 특성 추출
                feature_vector = [
                    # 기본 통계
                    np.mean(numbers),
                    np.std(numbers),
                    min(numbers),
                    max(numbers),
                    # 홀짝 비율
                    sum(1 for num in numbers if num % 2 == 1) / len(numbers),
                    # 구간별 분포
                    sum(1 for num in numbers if num <= 15),
                    sum(1 for num in numbers if 16 <= num <= 30),
                    sum(1 for num in numbers if num >= 31),
                    # 간격 통계
                    np.mean([numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]),
                    # 연속 번호 수
                    sum(1 for i in range(len(numbers)-1) if numbers[i+1] - numbers[i] == 1),
                ]
                features.append(feature_vector)

            if len(features) < 10:
                self.cluster_analysis = {"error": "데이터 부족"}
                return

            features = np.array(features)
            
            # 정규화
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # K-means 클러스터링
            optimal_k = min(5, len(features) // 10)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)

            # 클러스터별 특성 분석
            cluster_info = {}
            for k in range(optimal_k):
                cluster_mask = cluster_labels == k
                cluster_data = [data[i] for i in range(len(data)) if cluster_mask[i]]
                
                if cluster_data:
                    # 클러스터 내 번호 빈도
                    cluster_numbers = []
                    for draw in cluster_data:
                        cluster_numbers.extend(draw.numbers)
                    
                    cluster_info[k] = {
                        "size": len(cluster_data),
                        "percentage": len(cluster_data) / len(data) * 100,
                        "common_numbers": dict(Counter(cluster_numbers).most_common(10)),
                        "average_features": np.mean(features[cluster_mask], axis=0).tolist(),
                    }

            self.cluster_analysis = {
                "optimal_clusters": optimal_k,
                "cluster_info": cluster_info,
                "feature_names": [
                    "mean", "std", "min", "max", "odd_ratio",
                    "range1_count", "range2_count", "range3_count",
                    "avg_gap", "consecutive_count"
                ],
                "analysis_date": datetime.now().isoformat(),
            }

            self.logger.info(f"클러스터 분석 완료: {optimal_k}개 클러스터")

        except Exception as e:
            self.logger.error(f"클러스터 분석 실패: {e}")
            self.cluster_analysis = {"error": str(e)}

    def _analyze_trends(self, data: List[LotteryNumber]) -> None:
        """트렌드 분석"""
        try:
            self.logger.info("트렌드 분석 시작")

            # 시간별 트렌드 분석
            window_size = min(20, len(data) // 4)
            trends = {}

            # 번호별 트렌드
            for number in range(1, 46):
                appearances = []
                for i in range(len(data)):
                    if number in data[i].numbers:
                        appearances.append(i)
                
                if len(appearances) >= 2:
                    # 최근 출현 빈도 vs 과거 출현 빈도
                    recent_count = sum(1 for i in appearances if i >= len(data) * 0.7)
                    total_count = len(appearances)
                    recent_ratio = recent_count / (len(data) * 0.3) if len(data) * 0.3 > 0 else 0
                    overall_ratio = total_count / len(data)
                    
                    trend_score = recent_ratio / overall_ratio if overall_ratio > 0 else 1
                    trends[number] = {
                        "trend_score": trend_score,
                        "recent_count": recent_count,
                        "total_count": total_count,
                        "last_appearance": max(appearances),
                    }

            # 트렌드 분류
            trend_scores = [info["trend_score"] for info in trends.values()]
            if trend_scores:
                hot_threshold = np.percentile(trend_scores, 75)
                cold_threshold = np.percentile(trend_scores, 25)

                hot_trend_numbers = [num for num, info in trends.items() 
                                   if info["trend_score"] >= hot_threshold]
                cold_trend_numbers = [num for num, info in trends.items() 
                                    if info["trend_score"] <= cold_threshold]
            else:
                hot_trend_numbers = []
                cold_trend_numbers = []

            # 주기성 분석
            periodicity = {}
            for number in range(1, 46):
                appearances = [i for i, draw in enumerate(data) if number in draw.numbers]
                if len(appearances) >= 3:
                    intervals = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                    periodicity[number] = {
                        "average_interval": np.mean(intervals),
                        "interval_std": np.std(intervals),
                        "regularity_score": 1 / (1 + np.std(intervals)),  # 낮은 표준편차 = 높은 규칙성
                    }

            self.trend_analysis = {
                "number_trends": trends,
                "hot_trend_numbers": hot_trend_numbers,
                "cold_trend_numbers": cold_trend_numbers,
                "periodicity": periodicity,
                "analysis_window": window_size,
                "analysis_date": datetime.now().isoformat(),
            }

            self.logger.info(f"트렌드 분석 완료: 핫 트렌드 {len(hot_trend_numbers)}개, 콜드 트렌드 {len(cold_trend_numbers)}개")

        except Exception as e:
            self.logger.error(f"트렌드 분석 실패: {e}")
            self.trend_analysis = {"error": str(e)}

    def _analyze_roi(self, data: List[LotteryNumber]) -> None:
        """ROI 분석"""
        try:
            self.logger.info("ROI 분석 시작")

            # 가상 투자 시뮬레이션
            simulation_results = []
            
            # 다양한 전략으로 시뮬레이션
            strategies = ["frequency", "trend", "cluster", "random"]
            
            for strategy in strategies:
                total_investment = 0
                total_winnings = 0
                hit_counts = {3: 0, 4: 0, 5: 0, 6: 0}  # 등급별 적중 수
                
                # 최근 100회 시뮬레이션
                simulation_data = data[-100:] if len(data) >= 100 else data
                
                for i, actual_draw in enumerate(simulation_data[10:]):  # 처음 10회는 학습용
                    # 예측 생성 (간단한 시뮬레이션)
                    predictions = self._generate_virtual_predictions(actual_draw)
                    
                    # 투자 비용
                    investment = len(predictions) * 1000  # 게임당 1000원
                    total_investment += investment
                    
                    # 적중 확인 및 상금 계산
                    for prediction in predictions:
                        matches = len(set(prediction) & set(actual_draw.numbers))
                        if matches >= 3:
                            hit_counts[matches] += 1
                            
                            # 상금 계산 (대략적)
                            prize_map = {3: 5000, 4: 50000, 5: 1500000, 6: 2000000000}
                            total_winnings += prize_map.get(matches, 0)

                # ROI 계산
                roi = (total_winnings - total_investment) / total_investment if total_investment > 0 else -1
                hit_rate = sum(hit_counts.values()) / (len(simulation_data[10:]) * len(predictions)) if simulation_data[10:] else 0

                simulation_results.append({
                    "strategy": strategy,
                    "total_investment": total_investment,
                    "total_winnings": total_winnings,
                    "roi": roi,
                    "hit_rate": hit_rate,
                    "hit_counts": hit_counts,
                })

            # 최적 전략 선택
            best_strategy = max(simulation_results, key=lambda x: x["roi"])

            self.roi_analysis = {
                "simulation_results": simulation_results,
                "best_strategy": best_strategy,
                "analysis_period": len(simulation_data),
                "analysis_date": datetime.now().isoformat(),
            }

            self.logger.info(f"ROI 분석 완료: 최적 전략 '{best_strategy['strategy']}' (ROI: {best_strategy['roi']:.3f})")

        except Exception as e:
            self.logger.error(f"ROI 분석 실패: {e}")
            self.roi_analysis = {"error": str(e)}

    def _learn_grade_optimization_strategies(self, data: List[LotteryNumber]) -> None:
        """등급별 최적화 전략 학습"""
        try:
            self.logger.info("등급별 최적화 전략 학습 시작")

            # 등급별 성공 패턴 분석
            grade_patterns = {3: [], 4: [], 5: [], 6: []}  # 적중 개수별
            
            # 가상 예측과 실제 결과 비교
            for i, actual_draw in enumerate(data[20:]):  # 충분한 학습 데이터 확보
                predictions = self._generate_virtual_predictions(actual_draw)
                
                for prediction in predictions:
                    matches = len(set(prediction) & set(actual_draw.numbers))
                    if matches >= 3:
                        # 성공한 예측의 특성 저장
                        pattern_features = {
                            "prediction": prediction,
                            "actual": actual_draw.numbers,
                            "matches": matches,
                            "prediction_sum": sum(prediction),
                            "prediction_range": max(prediction) - min(prediction),
                            "odd_count": sum(1 for num in prediction if num % 2 == 1),
                        }
                        grade_patterns[matches].append(pattern_features)

            # 등급별 성공 패턴 요약
            grade_strategies = {}
            for grade, patterns in grade_patterns.items():
                if patterns:
                    strategies = {
                        "optimal_sum_range": (
                            np.percentile([p["prediction_sum"] for p in patterns], 25),
                            np.percentile([p["prediction_sum"] for p in patterns], 75)
                        ),
                        "optimal_range_span": np.mean([p["prediction_range"] for p in patterns]),
                        "optimal_odd_count": round(np.mean([p["odd_count"] for p in patterns])),
                        "success_rate": len(patterns) / len(data[20:]) if data[20:] else 0,
                        "sample_size": len(patterns),
                    }
                    grade_strategies[grade] = strategies

            # 성과 메트릭 업데이트
            self.performance_metrics["grade_strategies"] = grade_strategies
            self.performance_metrics["learning_data_size"] = len(data)

            self.logger.info(f"등급별 최적화 전략 학습 완료: {len(grade_strategies)}개 등급")

        except Exception as e:
            self.logger.error(f"등급별 최적화 전략 학습 실패: {e}")

    def _generate_virtual_predictions(self, draw: LotteryNumber) -> List[List[int]]:
        """가상 예측 생성 (시뮬레이션용)"""
        try:
            predictions = []
            
            # 간단한 예측 전략들
            strategies = [
                self._frequency_strategy(),
                self._trend_strategy(),
                self._random_strategy(),
            ]
            
            for strategy in strategies:
                prediction = strategy
                if len(prediction) == 6 and all(1 <= num <= 45 for num in prediction):
                    predictions.append(sorted(prediction))
            
            return predictions[:5]  # 최대 5개 예측

        except Exception as e:
            self.logger.error(f"가상 예측 생성 실패: {e}")
            return [sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())]

    def _frequency_strategy(self) -> List[int]:
        """빈도 기반 전략"""
        if self.frequency_analysis.get("hot_numbers"):
            hot_numbers = self.frequency_analysis["hot_numbers"]
            return sorted(np.random.choice(hot_numbers, min(6, len(hot_numbers)), replace=False).tolist())
        return sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())

    def _trend_strategy(self) -> List[int]:
        """트렌드 기반 전략"""
        if self.trend_analysis.get("hot_trend_numbers"):
            trend_numbers = self.trend_analysis["hot_trend_numbers"]
            return sorted(np.random.choice(trend_numbers, min(6, len(trend_numbers)), replace=False).tolist())
        return sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())

    def _random_strategy(self) -> List[int]:
        """랜덤 전략"""
        return sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        try:
            summary = {
                "analysis_completion": {
                    "frequency": bool(self.frequency_analysis),
                    "patterns": bool(self.pattern_analysis),
                    "clusters": bool(self.cluster_analysis),
                    "trends": bool(self.trend_analysis),
                    "roi": bool(self.roi_analysis),
                },
                "key_insights": {},
                "recommendations": [],
            }

            # 주요 인사이트 추출
            if self.frequency_analysis:
                hot_count = len(self.frequency_analysis.get("hot_numbers", []))
                summary["key_insights"]["hot_numbers_count"] = hot_count

            if self.roi_analysis and "best_strategy" in self.roi_analysis:
                best_roi = self.roi_analysis["best_strategy"]["roi"]
                summary["key_insights"]["best_roi"] = best_roi

            if self.trend_analysis:
                hot_trend_count = len(self.trend_analysis.get("hot_trend_numbers", []))
                summary["key_insights"]["trending_numbers_count"] = hot_trend_count

            # 추천사항 생성
            if summary["key_insights"].get("best_roi", -1) > -0.5:
                summary["recommendations"].append("현재 전략이 양호한 성과를 보입니다.")
            else:
                summary["recommendations"].append("전략 개선이 필요합니다.")

            return summary

        except Exception as e:
            self.logger.error(f"성능 요약 생성 실패: {e}")
            return {"error": str(e)} 