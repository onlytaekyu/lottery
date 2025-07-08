
# 1. 표준 라이브러리
from typing import Dict, List, Any, Optional
import collections

# 2. 서드파티
import pandas as pd

# 3. 프로젝트 내부
from .base_analyzer import BaseAnalyzer
from ..shared.types import LotteryNumber
from ..utils.unified_logging import get_logger

logger = get_logger(__name__)

class MovingAveragesAnalyzer:
    """이동평균 분석기"""
    def calculate(self, series: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
        series = series.astype(float)
        results = {}
        for window in windows:
            results[f'sma_{window}'] = series.rolling(window=window).mean()
            results[f'ema_{window}'] = series.ewm(span=window, adjust=False).mean()
        return results

class MACDAnalyzer:
    """MACD 분석기"""
    def calculate(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        series = series.astype(float)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return {'macd': macd, 'signal': signal_line, 'histogram': histogram}

class RSIAnalyzer:
    """RSI 분석기"""
    def calculate(self, series: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        series = series.astype(float)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return {'rsi': rsi}

class StochasticOscillatorAnalyzer:
    """스토캐스틱 오실레이터 분석기"""
    def calculate(self, series: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        series = series.astype(float)
        low_min = series.rolling(window=k_window).min()
        high_max = series.rolling(window=k_window).max()
        percent_k = 100 * ((series - low_min) / (high_max - low_min))
        percent_d = percent_k.rolling(window=d_window).mean()
        return {'%k': percent_k, '%d': percent_d}

class BollingerBandsAnalyzer:
    """볼린저 밴드 분석기"""
    def calculate(self, series: pd.Series, window: int = 20, num_std_dev: int = 2) -> Dict[str, pd.Series]:
        series = series.astype(float)
        sma = series.rolling(window=window).mean()
        std_dev = series.rolling(window=window).std()
        upper_band = sma + (std_dev * num_std_dev)
        lower_band = sma - (std_dev * num_std_dev)
        return {'upper': upper_band, 'middle': sma, 'lower': lower_band}


class TechnicalAnalyzer(BaseAnalyzer):
    """
    로또 번호 합계의 시계열 데이터에 대한 기술적 분석을 수행하는 분석기.
    - 이동평균 (단순, 지수)
    - MACD
    - RSI
    - 스토캐스틱 오실레이터
    - 볼린저 밴드
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config, name="TechnicalAnalyzer")
        self.ma_analyzer = MovingAveragesAnalyzer()
        self.macd_analyzer = MACDAnalyzer()
        self.rsi_analyzer = RSIAnalyzer()
        self.stochastic_analyzer = StochasticOscillatorAnalyzer()
        self.bollinger_analyzer = BollingerBandsAnalyzer()
        
        # 설정에서 파라미터 로드, 없으면 기본값 사용
        self.ma_windows = self.config.get('ma_windows', [5, 10, 20])
        self.macd_params = self.config.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
        self.rsi_window = self.config.get('rsi_window', 14)
        self.stochastic_params = self.config.get('stochastic', {'k_window': 14, 'd_window': 3})
        self.bollinger_params = self.config.get('bollinger', {'window': 20, 'num_std_dev': 2})


    def _analyze_impl(self, historical_data: List[LotteryNumber], *args, **kwargs) -> Dict[str, Any]:
        """
        과거 로또 데이터의 합계를 기반으로 기술적 지표를 계산합니다.

        Args:
            historical_data: 분석할 과거 당첨 번호 목록

        Returns:
            분석된 기술적 지표 딕셔너리
        """
        if len(historical_data) < max(list(self.stochastic_params.values()) + [self.rsi_window] + self.ma_windows + [self.bollinger_params['window']]):
            logger.warning("데이터가 부족하여 기술적 분석을 수행할 수 없습니다.")
            return {}

        df = self._prepare_data(historical_data)
        series = df['sum']

        results = collections.defaultdict(dict)

        # 각 분석기 실행
        try:
            results['moving_averages'] = self._run_ma_analysis(series)
            results['macd'] = self._run_macd_analysis(series)
            results['rsi'] = self._run_rsi_analysis(series)
            results['stochastic'] = self._run_stochastic_analysis(series)
            results['bollinger_bands'] = self._run_bollinger_analysis(series)
        except Exception as e:
            logger.error(f"기술적 분석 중 오류 발생: {e}", exc_info=True)
            return {}

        # 결과를 직렬화 가능한 형태로 변환
        final_results = self._format_results(results, series)
        
        return final_results

    def _prepare_data(self, historical_data: List[LotteryNumber]) -> pd.DataFrame:
        """LotteryNumber 리스트를 분석에 적합한 Pandas DataFrame으로 변환합니다."""
        if not historical_data:
            return pd.DataFrame(columns=['draw_no', 'sum']).set_index('draw_no')
            
        data = [
            {"draw_no": d.draw_no, "sum": sum(d.numbers)}
            for d in historical_data
        ]
        df = pd.DataFrame(data).set_index('draw_no').sort_index()
        if 'sum' in df.columns:
            df['sum'] = df['sum'].astype(float)
        return df

    def _run_ma_analysis(self, series: pd.Series) -> Dict:
        return {k: v.to_dict() for k, v in self.ma_analyzer.calculate(series, self.ma_windows).items()}

    def _run_macd_analysis(self, series: pd.Series) -> Dict:
        return {k: v.to_dict() for k, v in self.macd_analyzer.calculate(series, **self.macd_params).items()}

    def _run_rsi_analysis(self, series: pd.Series) -> Dict:
        return {k: v.to_dict() for k, v in self.rsi_analyzer.calculate(series, self.rsi_window).items()}

    def _run_stochastic_analysis(self, series: pd.Series) -> Dict:
        return {k: v.to_dict() for k, v in self.stochastic_analyzer.calculate(series, **self.stochastic_params).items()}

    def _run_bollinger_analysis(self, series: pd.Series) -> Dict:
        return {k: v.to_dict() for k, v in self.bollinger_analyzer.calculate(series, **self.bollinger_params).items()}

    def _format_results(self, results: Dict, series: pd.Series) -> Dict[str, Any]:
        """분석 결과를 최종 출력 형식으로 변환합니다."""
        formatted = {}
        
        if series.empty:
            return {
                'moving_averages': {}, 'macd': {}, 'rsi': {}, 
                'stochastic': {}, 'bollinger_bands': {}, 'last_sum': None
            }

        # 최신 데이터(마지막 회차)에 대한 지표만 추출
        last_draw = series.index.max()
        
        for analysis_type, indicators in results.items():
            formatted[analysis_type] = {}
            for indicator_name, values in indicators.items():
                if last_draw in values:
                    value = values[last_draw]
                    # NaN 값은 None으로 변환
                    formatted[analysis_type][indicator_name] = value if pd.notna(value) else None
                else:
                    formatted[analysis_type][indicator_name] = None
        
        # 마지막 회차의 번호 합계 추가
        if last_draw in series.index:
            formatted['last_sum'] = series.loc[last_draw]
        else:
            formatted['last_sum'] = None
        
        return formatted 