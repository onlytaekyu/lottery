"""
실시간 성능 추적 시스템 (Performance Tracker)

로또 예측 시스템의 실시간 성능을 추적하고 모니터링하는 시스템입니다.

주요 기능:
- 실시간 성능 지표 추적
- 등급별 적중률 모니터링
- ROI 및 손실률 추적
- 전략별 성과 비교
- 대시보드 시각화
- 알림 및 경고 시스템
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from queue import Queue

from ..shared.types import LotteryNumber, ModelPrediction
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_config
from ..utils.cache_manager import UnifiedCachePathManager

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 지표 데이터 클래스"""

    timestamp: datetime
    strategy: str
    total_games: int
    wins: int
    losses: int
    win_rate: float
    loss_rate: float
    total_investment: float
    total_return: float
    net_profit: float
    roi: float
    hit_rates: Dict[str, float]  # 등급별 적중률
    avg_return_per_game: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    streak_type: str  # 'win' or 'loss'
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    recovery_time: Optional[int]  # 손실 회복 시간 (게임 수)


@dataclass
class AlertConfig:
    """알림 설정 데이터 클래스"""

    win_rate_threshold: float = 0.4  # 승률 임계값
    loss_rate_threshold: float = 0.7  # 손실률 임계값
    roi_threshold: float = -0.3  # ROI 임계값
    consecutive_loss_threshold: int = 10  # 연속 손실 임계값
    drawdown_threshold: float = 0.2  # 최대 손실 임계값
    enable_email_alerts: bool = False  # 이메일 알림 활성화
    enable_sound_alerts: bool = True  # 사운드 알림 활성화
    alert_cooldown: int = 300  # 알림 쿨다운 (초)


@dataclass
class SystemHealth:
    """시스템 상태 정보"""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: float        # 추가
    gpu_memory: float       # 추가
    prediction_latency: float
    error_count: int
    last_update: datetime
    status: str  # 'healthy', 'warning', 'critical'


class AlertManager:
    """알림 관리 시스템"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.last_alert_time = {}
        self.alert_queue = Queue()

    def check_alerts(self, metrics: PerformanceMetrics) -> List[str]:
        """알림 조건 확인"""
        alerts = []
        current_time = datetime.now()

        # 승률 알림
        if metrics.win_rate < self.config.win_rate_threshold:
            alert_key = f"win_rate_{metrics.strategy}"
            if self._should_send_alert(alert_key, current_time):
                alerts.append(
                    f"⚠️ {metrics.strategy} 전략의 승률이 임계값 미만입니다: {metrics.win_rate:.1%}"
                )

        # 손실률 알림
        if metrics.loss_rate > self.config.loss_rate_threshold:
            alert_key = f"loss_rate_{metrics.strategy}"
            if self._should_send_alert(alert_key, current_time):
                alerts.append(
                    f"🚨 {metrics.strategy} 전략의 손실률이 임계값 초과입니다: {metrics.loss_rate:.1%}"
                )

        # ROI 알림
        if metrics.roi < self.config.roi_threshold:
            alert_key = f"roi_{metrics.strategy}"
            if self._should_send_alert(alert_key, current_time):
                alerts.append(
                    f"📉 {metrics.strategy} 전략의 ROI가 임계값 미만입니다: {metrics.roi:.1%}"
                )

        # 연속 손실 알림
        if (
            metrics.current_streak >= self.config.consecutive_loss_threshold
            and metrics.streak_type == "loss"
        ):
            alert_key = f"consecutive_loss_{metrics.strategy}"
            if self._should_send_alert(alert_key, current_time):
                alerts.append(
                    f"🔴 {metrics.strategy} 전략에서 {metrics.current_streak}회 연속 손실 발생"
                )

        # 최대 손실 알림
        if metrics.max_drawdown > self.config.drawdown_threshold:
            alert_key = f"drawdown_{metrics.strategy}"
            if self._should_send_alert(alert_key, current_time):
                alerts.append(
                    f"📊 {metrics.strategy} 전략의 최대 손실이 임계값 초과: {metrics.max_drawdown:.1%}"
                )

        return alerts

    def _should_send_alert(self, alert_key: str, current_time: datetime) -> bool:
        """알림 전송 여부 확인 (쿨다운 적용)"""
        if alert_key not in self.last_alert_time:
            self.last_alert_time[alert_key] = current_time
            return True

        time_diff = (current_time - self.last_alert_time[alert_key]).total_seconds()
        if time_diff >= self.config.alert_cooldown:
            self.last_alert_time[alert_key] = current_time
            return True

        return False

    def send_alert(self, message: str) -> None:
        """알림 전송"""
        try:
            self.logger.warning(f"ALERT: {message}")

            if self.config.enable_sound_alerts:
                self._play_alert_sound()

            if self.config.enable_email_alerts:
                self._send_email_alert(message)

        except Exception as e:
            self.logger.error(f"알림 전송 중 오류: {str(e)}")

    def _play_alert_sound(self) -> None:
        """알림 사운드 재생"""
        try:
            # 간단한 비프음 (Windows)
            if os.name == "nt":
                import winsound

                winsound.Beep(1000, 500)
        except Exception as e:
            self.logger.debug(f"사운드 재생 실패: {str(e)}")

    def _send_email_alert(self, message: str) -> None:
        """이메일 알림 전송 (구현 필요)"""
        # 이메일 전송 로직 구현


class PerformanceTracker:
    """실시간 성능 추적 시스템"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        성능 추적기 초기화

        Args:
            config: 설정 객체
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # 알림 설정
        alert_config_dict = self.config.get("alerts", {})
        self.alert_config = AlertConfig(**alert_config_dict)
        self.alert_manager = AlertManager(self.alert_config)

        # 성능 데이터 저장소
        self.performance_history = defaultdict(deque)  # 전략별 성능 이력
        self.current_metrics = {}  # 현재 성능 지표
        self.game_results = defaultdict(list)  # 게임 결과 이력

        # 실시간 모니터링 설정
        self.monitoring_active = False
        self.monitoring_thread = None
        self.update_interval = self.config.get("update_interval", 5)  # 5초마다 업데이트

        # 데이터 저장 설정
        paths = get_config()
        cache_path_manager = UnifiedCachePathManager(paths)
        self.cache_dir = cache_path_manager.get_path("performance_tracking")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 시스템 상태 모니터링
        self.system_health = SystemHealth(
            cpu_usage=0, 
            memory_usage=0, 
            disk_usage=0, 
            gpu_usage=0,
            gpu_memory=0,
            prediction_latency=0, 
            error_count=0, 
            last_update=datetime.now(), 
            status="healthy"
        )

        # 대시보드 설정
        self.dashboard_data = {
            "strategies": [],
            "performance_timeline": [],
            "alerts": [],
            "system_health": {},
        }

        # 성능 임계값 설정
        self.performance_thresholds = {
            "excellent": {"win_rate": 0.6, "roi": 0.2, "loss_rate": 0.3},
            "good": {"win_rate": 0.4, "roi": 0.0, "loss_rate": 0.5},
            "poor": {"win_rate": 0.2, "roi": -0.3, "loss_rate": 0.7},
        }

        self.logger.info("실시간 성능 추적 시스템 초기화 완료")

    def start_monitoring(self) -> None:
        """실시간 모니터링 시작"""
        try:
            if self.monitoring_active:
                self.logger.warning("모니터링이 이미 활성화되어 있습니다")
                return

            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self.monitoring_thread.start()

            self.logger.info("실시간 모니터링 시작")

        except Exception as e:
            self.logger.error(f"모니터링 시작 중 오류: {str(e)}")

    def stop_monitoring(self) -> None:
        """실시간 모니터링 중지"""
        try:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)

            self.logger.info("실시간 모니터링 중지")

        except Exception as e:
            self.logger.error(f"모니터링 중지 중 오류: {str(e)}")

    def _monitoring_loop(self) -> None:
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 성능 지표 업데이트
                self._update_performance_metrics()

                # 시스템 상태 업데이트
                self._update_system_health()

                # 알림 확인
                self._check_and_send_alerts()

                # 대시보드 데이터 업데이트
                self._update_dashboard_data()

                time.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"모니터링 루프 중 오류: {str(e)}")
                time.sleep(self.update_interval)

    def record_game_result(
        self,
        strategy: str,
        prediction: ModelPrediction,
        actual_result: LotteryNumber,
        investment: float = 1000,
    ) -> None:
        """게임 결과 기록"""
        try:
            # 적중 개수 계산
            matches = len(set(prediction.numbers) & set(actual_result.numbers))

            # 등급 및 당첨금 계산
            grade, prize = self._calculate_grade_and_prize(matches)

            # 게임 결과 생성
            game_result = {
                "timestamp": datetime.now(),
                "strategy": strategy,
                "prediction": prediction.numbers,
                "actual": actual_result.numbers,
                "matches": matches,
                "grade": grade,
                "prize": prize,
                "investment": investment,
                "net_return": prize - investment,
                "is_win": prize > investment,
            }

            # 결과 저장
            self.game_results[strategy].append(game_result)

            # 메모리 누수 방지: 오래된 결과 제거 (최근 1000개만 유지)
            if len(self.game_results[strategy]) > 1000:
                self.game_results[strategy] = self.game_results[strategy][-1000:]

            # 성능 지표 업데이트
            self._update_strategy_metrics(strategy)

            self.logger.debug(
                f"게임 결과 기록: {strategy} - 적중: {matches}개, 등급: {grade}, 수익: {prize - investment}"
            )

        except Exception as e:
            self.logger.error(f"게임 결과 기록 중 오류: {str(e)}")

    def _calculate_grade_and_prize(self, matches: int) -> Tuple[Optional[str], float]:
        """적중 개수에 따른 등급 및 당첨금 계산"""
        try:
            prize_table = {
                6: ("1st", 2000000000),  # 1등
                5: ("2nd", 60000000),  # 2등 (보너스볼 미고려 단순화)
                4: ("4th", 50000),  # 4등
                3: ("5th", 5000),  # 5등
            }

            return prize_table.get(matches, (None, 0))

        except Exception as e:
            self.logger.error(f"등급 및 당첨금 계산 중 오류: {str(e)}")
            return None, 0

    def _update_strategy_metrics(self, strategy: str) -> None:
        """전략별 성능 지표 업데이트"""
        try:
            if strategy not in self.game_results or not self.game_results[strategy]:
                return

            results = self.game_results[strategy]

            # 기본 통계
            total_games = len(results)
            wins = sum(1 for r in results if r["is_win"])
            losses = total_games - wins
            win_rate = wins / total_games if total_games > 0 else 0
            loss_rate = losses / total_games if total_games > 0 else 0

            # 투자 및 수익
            total_investment = sum(r["investment"] for r in results)
            total_return = sum(r["prize"] for r in results)
            net_profit = total_return - total_investment
            roi = net_profit / total_investment if total_investment > 0 else 0
            avg_return_per_game = total_return / total_games if total_games > 0 else 0

            # 등급별 적중률
            grade_counts = defaultdict(int)
            for r in results:
                if r["grade"]:
                    grade_counts[r["grade"]] += 1

            hit_rates = {}
            for grade, count in grade_counts.items():
                hit_rates[grade] = count / total_games

            # 연속 기록 계산
            (
                max_consecutive_wins,
                max_consecutive_losses,
                current_streak,
                streak_type,
            ) = self._calculate_streaks(results)

            # 변동성 및 리스크 지표
            returns = [r["net_return"] for r in results]
            volatility = np.std(returns) if len(returns) > 1 else 0

            # 샤프 비율
            mean_return = np.mean(returns) if returns else 0
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0

            # 최대 손실 계산
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = (
                abs(np.min(drawdowns)) / total_investment
                if total_investment > 0 and len(drawdowns) > 0
                else 0
            )

            # 회복 시간 계산
            recovery_time = self._calculate_recovery_time(results)

            # 성능 지표 생성
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                strategy=strategy,
                total_games=total_games,
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                loss_rate=loss_rate,
                total_investment=total_investment,
                total_return=total_return,
                net_profit=net_profit,
                roi=roi,
                hit_rates=hit_rates,
                avg_return_per_game=avg_return_per_game,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                current_streak=current_streak,
                streak_type=streak_type,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                recovery_time=recovery_time,
            )

            # 현재 지표 업데이트
            self.current_metrics[strategy] = metrics

            # 성능 이력 저장
            self.performance_history[strategy].append(metrics)

            # 메모리 누수 방지: 성능 이력 크기 제한 (최근 500개만 유지)
            if len(self.performance_history[strategy]) > 500:
                self.performance_history[strategy] = deque(
                    list(self.performance_history[strategy])[-500:], 
                    maxlen=500
                )

        except Exception as e:
            self.logger.error(f"전략별 성능 지표 업데이트 중 오류: {str(e)}")

    def _calculate_streaks(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[int, int, int, str]:
        """연속 기록 계산"""
        try:
            if not results:
                return 0, 0, 0, "none"

            max_wins = 0
            max_losses = 0
            current_wins = 0
            current_losses = 0

            # 최근 결과부터 역순으로 확인
            for result in reversed(results):
                if result["is_win"]:
                    current_wins += 1
                    current_losses = 0
                    max_wins = max(max_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    max_losses = max(max_losses, current_losses)

            # 현재 연속 기록
            if results[-1]["is_win"]:
                current_streak = current_wins
                streak_type = "win"
            else:
                current_streak = current_losses
                streak_type = "loss"

            return max_wins, max_losses, current_streak, streak_type

        except Exception as e:
            self.logger.error(f"연속 기록 계산 중 오류: {str(e)}")
            return 0, 0, 0, "none"

    def _calculate_recovery_time(self, results: List[Dict[str, Any]]) -> Optional[int]:
        """손실 회복 시간 계산"""
        try:
            if not results:
                return None

            # 누적 수익 계산
            cumulative_profit = 0
            max_loss_point = 0
            recovery_games = 0

            for i, result in enumerate(results):
                cumulative_profit += result["net_return"]

                if cumulative_profit < 0:
                    max_loss_point = i
                    recovery_games = 0
                elif max_loss_point > 0 and cumulative_profit >= 0:
                    recovery_games = i - max_loss_point
                    break

            return recovery_games if recovery_games > 0 else None

        except Exception as e:
            self.logger.error(f"회복 시간 계산 중 오류: {str(e)}")
            return None

    def _update_performance_metrics(self) -> None:
        """성능 지표 업데이트"""
        try:
            # 모든 전략의 성능 지표 업데이트
            for strategy in self.game_results.keys():
                self._update_strategy_metrics(strategy)

        except Exception as e:
            self.logger.error(f"성능 지표 업데이트 중 오류: {str(e)}")

    def _update_system_health(self) -> None:
        """시스템 상태 업데이트"""
        try:
            import psutil
            import torch  # 추가

            # CPU 사용률
            cpu_usage = psutil.cpu_percent(interval=1)

            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # 디스크 사용률
            disk = psutil.disk_usage("/")
            disk_usage = disk.percent

            # GPU 모니터링 추가 (간단)
            gpu_usage = 0
            gpu_memory = 0
            if torch.cuda.is_available():
                try:
                    # GPU 사용률 (nvidia-ml-py 없이는 0으로 설정)
                    gpu_usage = 0  # torch.cuda.utilization()은 nvidia-ml-py 필요
                    
                    # GPU 메모리 사용률 계산 (간단한 방식)
                    gpu_memory_used = torch.cuda.memory_allocated()
                    gpu_memory_reserved = torch.cuda.memory_reserved()
                    if gpu_memory_reserved > 0:
                        gpu_memory = (gpu_memory_used / gpu_memory_reserved) * 100
                    else:
                        gpu_memory = 0
                except:
                    gpu_usage = 0
                    gpu_memory = 0

            # 예측 지연시간 (더미 데이터)
            prediction_latency = 0.1  # 실제 구현에서는 측정값 사용

            # 오류 개수 (로그 기반)
            error_count = 0  # 실제 구현에서는 로그 파싱

            # 상태 판정 (GPU 포함)
            status = "healthy"
            if cpu_usage > 80 or memory_usage > 80 or disk_usage > 90 or gpu_usage > 90:
                status = "critical"
            elif cpu_usage > 60 or memory_usage > 60 or disk_usage > 80 or gpu_usage > 70:
                status = "warning"

            self.system_health = SystemHealth(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                gpu_usage=gpu_usage,        # 추가
                gpu_memory=gpu_memory,      # 추가
                prediction_latency=prediction_latency,
                error_count=error_count,
                last_update=datetime.now(),
                status=status,
            )

        except Exception as e:
            self.logger.error(f"시스템 상태 업데이트 중 오류: {str(e)}")

    def _check_and_send_alerts(self) -> None:
        """알림 확인 및 전송"""
        try:
            for strategy, metrics in self.current_metrics.items():
                alerts = self.alert_manager.check_alerts(metrics)
                for alert in alerts:
                    self.alert_manager.send_alert(alert)

        except Exception as e:
            self.logger.error(f"알림 확인 및 전송 중 오류: {str(e)}")

    def _update_dashboard_data(self) -> None:
        """대시보드 데이터 업데이트"""
        try:
            # 전략 목록
            self.dashboard_data["strategies"] = list(self.current_metrics.keys())

            # 성능 타임라인
            timeline = []
            for strategy, metrics in self.current_metrics.items():
                timeline.append(
                    {
                        "timestamp": metrics.timestamp.isoformat(),
                        "strategy": strategy,
                        "win_rate": metrics.win_rate,
                        "roi": metrics.roi,
                        "total_games": metrics.total_games,
                    }
                )

            self.dashboard_data["performance_timeline"] = timeline

            # 시스템 상태
            self.dashboard_data["system_health"] = asdict(self.system_health)

        except Exception as e:
            self.logger.error(f"대시보드 데이터 업데이트 중 오류: {str(e)}")

    def get_performance_summary(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        try:
            if strategy:
                if strategy in self.current_metrics:
                    metrics = self.current_metrics[strategy]
                    return {
                        "strategy": strategy,
                        "performance": asdict(metrics),
                        "status": self._get_performance_status(metrics),
                        "recommendations": self._get_performance_recommendations(
                            metrics
                        ),
                    }
                else:
                    return {"error": f"전략 {strategy}를 찾을 수 없습니다"}
            else:
                # 전체 요약
                summary = {
                    "total_strategies": len(self.current_metrics),
                    "strategies": {},
                    "overall_performance": self._calculate_overall_performance(),
                    "system_health": asdict(self.system_health),
                }

                for strat, metrics in self.current_metrics.items():
                    summary["strategies"][strat] = {
                        "performance": asdict(metrics),
                        "status": self._get_performance_status(metrics),
                    }

                return summary

        except Exception as e:
            self.logger.error(f"성능 요약 정보 반환 중 오류: {str(e)}")
            return {"error": str(e)}

    def _get_performance_status(self, metrics: PerformanceMetrics) -> str:
        """성능 상태 평가"""
        try:
            # 성능 점수 계산
            score = 0

            # 승률 점수
            if metrics.win_rate >= self.performance_thresholds["excellent"]["win_rate"]:
                score += 3
            elif metrics.win_rate >= self.performance_thresholds["good"]["win_rate"]:
                score += 2
            elif metrics.win_rate >= self.performance_thresholds["poor"]["win_rate"]:
                score += 1

            # ROI 점수
            if metrics.roi >= self.performance_thresholds["excellent"]["roi"]:
                score += 3
            elif metrics.roi >= self.performance_thresholds["good"]["roi"]:
                score += 2
            elif metrics.roi >= self.performance_thresholds["poor"]["roi"]:
                score += 1

            # 손실률 점수 (역산)
            if (
                metrics.loss_rate
                <= self.performance_thresholds["excellent"]["loss_rate"]
            ):
                score += 3
            elif metrics.loss_rate <= self.performance_thresholds["good"]["loss_rate"]:
                score += 2
            elif metrics.loss_rate <= self.performance_thresholds["poor"]["loss_rate"]:
                score += 1

            # 상태 판정
            if score >= 7:
                return "excellent"
            elif score >= 5:
                return "good"
            elif score >= 3:
                return "fair"
            else:
                return "poor"

        except Exception as e:
            self.logger.error(f"성능 상태 평가 중 오류: {str(e)}")
            return "unknown"

    def _get_performance_recommendations(
        self, metrics: PerformanceMetrics
    ) -> List[str]:
        """성능 개선 추천사항"""
        recommendations = []

        try:
            # 승률 기반 추천
            if metrics.win_rate < 0.3:
                recommendations.append(
                    "승률이 낮습니다. 예측 모델을 재검토하거나 전략을 변경해보세요."
                )

            # ROI 기반 추천
            if metrics.roi < -0.2:
                recommendations.append(
                    "ROI가 매우 낮습니다. 베팅 크기를 줄이거나 전략을 중단하는 것을 고려하세요."
                )

            # 연속 손실 기반 추천
            if metrics.current_streak > 5 and metrics.streak_type == "loss":
                recommendations.append(
                    "연속 손실이 발생하고 있습니다. 일시적으로 베팅을 중단하고 전략을 재평가하세요."
                )

            # 변동성 기반 추천
            if metrics.volatility > metrics.avg_return_per_game * 2:
                recommendations.append(
                    "수익 변동성이 높습니다. 리스크 관리를 강화하고 베팅 크기를 조정하세요."
                )

            # 최대 손실 기반 추천
            if metrics.max_drawdown > 0.3:
                recommendations.append(
                    "최대 손실이 큽니다. 손실 제한 규칙을 설정하고 엄격히 준수하세요."
                )

            # 긍정적인 추천
            if metrics.win_rate > 0.5 and metrics.roi > 0.1:
                recommendations.append(
                    "우수한 성과를 보이고 있습니다. 현재 전략을 유지하되 과도한 자신감은 주의하세요."
                )

        except Exception as e:
            self.logger.error(f"성능 추천사항 생성 중 오류: {str(e)}")
            recommendations.append("추천사항 생성 중 오류가 발생했습니다.")

        return recommendations

    def _calculate_overall_performance(self) -> Dict[str, Any]:
        """전체 성능 계산"""
        try:
            if not self.current_metrics:
                return {}

            # 전체 통계 계산
            total_games = sum(m.total_games for m in self.current_metrics.values())
            total_wins = sum(m.wins for m in self.current_metrics.values())
            total_investment = sum(
                m.total_investment for m in self.current_metrics.values()
            )
            total_return = sum(m.total_return for m in self.current_metrics.values())

            overall_win_rate = total_wins / total_games if total_games > 0 else 0
            overall_roi = (
                (total_return - total_investment) / total_investment
                if total_investment > 0
                else 0
            )

            # 최고 성과 전략
            best_strategy = max(self.current_metrics.items(), key=lambda x: x[1].roi)
            worst_strategy = min(self.current_metrics.items(), key=lambda x: x[1].roi)

            return {
                "total_games": total_games,
                "overall_win_rate": overall_win_rate,
                "overall_roi": overall_roi,
                "total_investment": total_investment,
                "total_return": total_return,
                "best_strategy": best_strategy[0],
                "best_strategy_roi": best_strategy[1].roi,
                "worst_strategy": worst_strategy[0],
                "worst_strategy_roi": worst_strategy[1].roi,
                "active_strategies": len(self.current_metrics),
            }

        except Exception as e:
            self.logger.error(f"전체 성능 계산 중 오류: {str(e)}")
            return {}

    def generate_performance_report(
        self, strategy: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """성능 리포트 생성"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            report = {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days,
                },
                "strategies": {},
                "summary": {},
            }

            if strategy:
                if strategy in self.current_metrics:
                    report["strategies"][strategy] = self._generate_strategy_report(
                        strategy, start_date, end_date
                    )
                else:
                    report["error"] = f"전략 {strategy}를 찾을 수 없습니다"
            else:
                for strat in self.current_metrics.keys():
                    report["strategies"][strat] = self._generate_strategy_report(
                        strat, start_date, end_date
                    )

            # 요약 정보
            report["summary"] = self._generate_report_summary(report["strategies"])

            return report

        except Exception as e:
            self.logger.error(f"성능 리포트 생성 중 오류: {str(e)}")
            return {"error": str(e)}

    def _generate_strategy_report(
        self, strategy: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """전략별 리포트 생성"""
        try:
            # 기간 내 게임 결과 필터링
            filtered_results = [
                r
                for r in self.game_results[strategy]
                if start_date <= r["timestamp"] <= end_date
            ]

            if not filtered_results:
                return {"message": "해당 기간에 게임 결과가 없습니다"}

            # 통계 계산
            total_games = len(filtered_results)
            wins = sum(1 for r in filtered_results if r["is_win"])
            total_investment = sum(r["investment"] for r in filtered_results)
            total_return = sum(r["prize"] for r in filtered_results)

            # 일별 성과
            daily_performance = defaultdict(
                lambda: {"games": 0, "wins": 0, "investment": 0, "return": 0}
            )

            for result in filtered_results:
                date_key = result["timestamp"].date().isoformat()
                daily_performance[date_key]["games"] += 1
                daily_performance[date_key]["wins"] += 1 if result["is_win"] else 0
                daily_performance[date_key]["investment"] += result["investment"]
                daily_performance[date_key]["return"] += result["prize"]

            return {
                "total_games": total_games,
                "wins": wins,
                "win_rate": wins / total_games if total_games > 0 else 0,
                "total_investment": total_investment,
                "total_return": total_return,
                "net_profit": total_return - total_investment,
                "roi": (
                    (total_return - total_investment) / total_investment
                    if total_investment > 0
                    else 0
                ),
                "daily_performance": dict(daily_performance),
                "best_day": (
                    max(
                        daily_performance.items(),
                        key=lambda x: x[1]["return"] - x[1]["investment"],
                    )
                    if daily_performance
                    else None
                ),
                "worst_day": (
                    min(
                        daily_performance.items(),
                        key=lambda x: x[1]["return"] - x[1]["investment"],
                    )
                    if daily_performance
                    else None
                ),
            }

        except Exception as e:
            self.logger.error(f"전략별 리포트 생성 중 오류: {str(e)}")
            return {"error": str(e)}

    def _generate_report_summary(
        self, strategies_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """리포트 요약 생성"""
        try:
            if not strategies_data:
                return {}

            # 전체 통계
            total_games = sum(
                data.get("total_games", 0)
                for data in strategies_data.values()
                if isinstance(data, dict)
            )
            total_investment = sum(
                data.get("total_investment", 0)
                for data in strategies_data.values()
                if isinstance(data, dict)
            )
            total_return = sum(
                data.get("total_return", 0)
                for data in strategies_data.values()
                if isinstance(data, dict)
            )

            # 최고/최악 전략
            valid_strategies = {
                k: v
                for k, v in strategies_data.items()
                if isinstance(v, dict) and "roi" in v
            }

            if valid_strategies:
                best_strategy = max(valid_strategies.items(), key=lambda x: x[1]["roi"])
                worst_strategy = min(
                    valid_strategies.items(), key=lambda x: x[1]["roi"]
                )
            else:
                best_strategy = worst_strategy = None

            return {
                "total_games": total_games,
                "total_investment": total_investment,
                "total_return": total_return,
                "overall_roi": (
                    (total_return - total_investment) / total_investment
                    if total_investment > 0
                    else 0
                ),
                "best_strategy": best_strategy[0] if best_strategy else None,
                "best_strategy_roi": best_strategy[1]["roi"] if best_strategy else None,
                "worst_strategy": worst_strategy[0] if worst_strategy else None,
                "worst_strategy_roi": (
                    worst_strategy[1]["roi"] if worst_strategy else None
                ),
                "active_strategies": len(valid_strategies),
            }

        except Exception as e:
            self.logger.error(f"리포트 요약 생성 중 오류: {str(e)}")
            return {}

    def save_performance_data(self) -> None:
        """성능 데이터 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 현재 지표 저장
            metrics_file = self.cache_dir / f"performance_metrics_{timestamp}.json"
            metrics_data = {}
            for strategy, metrics in self.current_metrics.items():
                metrics_data[strategy] = asdict(metrics)

            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2, default=str)

            # 게임 결과 저장
            results_file = self.cache_dir / f"game_results_{timestamp}.json"
            results_data = {}
            for strategy, results in self.game_results.items():
                results_data[strategy] = results

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"성능 데이터 저장 완료: {metrics_file}, {results_file}")

        except Exception as e:
            self.logger.error(f"성능 데이터 저장 중 오류: {str(e)}")

    def load_performance_data(self, date: Optional[str] = None) -> bool:
        """성능 데이터 로드"""
        try:
            if date:
                pattern = f"performance_metrics_{date}*.json"
            else:
                pattern = "performance_metrics_*.json"

            metrics_files = list(self.cache_dir.glob(pattern))
            if not metrics_files:
                self.logger.warning(f"성능 데이터 파일을 찾을 수 없습니다: {pattern}")
                return False

            # 최신 파일 선택
            latest_file = max(metrics_files, key=lambda x: x.name)

            with open(latest_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            # 데이터 복원
            for strategy, data in metrics_data.items():
                # datetime 객체 복원
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                self.current_metrics[strategy] = PerformanceMetrics(**data)

            self.logger.info(f"성능 데이터 로드 완료: {latest_file}")
            return True

        except Exception as e:
            self.logger.error(f"성능 데이터 로드 중 오류: {str(e)}")
            return False

    def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드 데이터 반환"""
        return self.dashboard_data.copy()

    def export_performance_csv(
        self, strategy: Optional[str] = None, days: int = 30
    ) -> str:
        """성능 데이터 CSV 내보내기"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # 데이터 수집
            export_data = []

            strategies_to_export = (
                [strategy] if strategy else list(self.game_results.keys())
            )

            for strat in strategies_to_export:
                if strat in self.game_results:
                    for result in self.game_results[strat]:
                        if start_date <= result["timestamp"] <= end_date:
                            export_data.append(
                                {
                                    "timestamp": result["timestamp"].isoformat(),
                                    "strategy": strat,
                                    "prediction": str(result["prediction"]),
                                    "actual": str(result["actual"]),
                                    "matches": result["matches"],
                                    "grade": result["grade"],
                                    "prize": result["prize"],
                                    "investment": result["investment"],
                                    "net_return": result["net_return"],
                                    "is_win": result["is_win"],
                                }
                            )

            # CSV 파일 생성
            if export_data:
                df = pd.DataFrame(export_data)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = self.cache_dir / f"performance_export_{timestamp}.csv"
                df.to_csv(csv_file, index=False, encoding="utf-8")

                self.logger.info(f"성능 데이터 CSV 내보내기 완료: {csv_file}")
                return str(csv_file)
            else:
                self.logger.warning("내보낼 데이터가 없습니다")
                return ""

        except Exception as e:
            self.logger.error(f"성능 데이터 CSV 내보내기 중 오류: {str(e)}")
            return ""


def test_gpu_monitoring():
    """GPU 모니터링 기능 테스트"""
    try:
        import torch
        
        print("=== GPU 모니터링 테스트 ===")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU 개수: {torch.cuda.device_count()}")
            print(f"현재 GPU: {torch.cuda.current_device()}")
            
            # GPU 사용률 테스트
            try:
                # GPU 사용률 (nvidia-ml-py 없이는 0으로 설정)
                gpu_usage = 0  # torch.cuda.utilization()은 nvidia-ml-py 필요
                
                # GPU 메모리 사용률 계산 (간단한 방식)
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_memory_reserved = torch.cuda.memory_reserved()
                if gpu_memory_reserved > 0:
                    gpu_memory = (gpu_memory_used / gpu_memory_reserved) * 100
                else:
                    gpu_memory = 0
                print(f"GPU 사용률: {gpu_usage}%")
                print(f"GPU 메모리: {gpu_memory:.1f}%")
                print(f"GPU 메모리 사용: {gpu_memory_used / 1024**2:.1f} MB")
                print(f"GPU 메모리 예약: {gpu_memory_reserved / 1024**2:.1f} MB")
            except Exception as e:
                print(f"GPU 모니터링 오류: {e}")
        else:
            print("GPU를 사용할 수 없습니다")
            
        # PerformanceTracker 테스트
        tracker = PerformanceTracker()
        tracker._update_system_health()
        
        print(f"시스템 상태: {tracker.system_health.status}")
        print(f"GPU 사용률: {tracker.system_health.gpu_usage}%")
        print(f"GPU 메모리: {tracker.system_health.gpu_memory}%")
        print("테스트 완료!")
        
    except Exception as e:
        print(f"테스트 중 오류: {e}")


if __name__ == "__main__":
    test_gpu_monitoring()
