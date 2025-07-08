"""
공유 유형 정의

이 모듈은 로또 번호 예측 시스템에서 사용되는 공유 데이터 유형을 정의합니다.
"""

from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Set,
    Any,
    TYPE_CHECKING,
    TypedDict,
    Protocol,
    runtime_checkable,
)
import numpy as np
from datetime import datetime

# 타입 호환성을 위한 조건부 임포트
if TYPE_CHECKING:
    pass


@runtime_checkable
class LotteryDataProtocol(Protocol):
    """로또 데이터 프로토콜 - 타입 검증용"""
    draw_no: int
    numbers: List[int]
    date: Optional[str]


@dataclass
class LotteryNumber:
    """로또 번호 데이터 클래스"""

    draw_no: int
    """회차 번호"""

    numbers: List[int]
    """당첨 번호 (정렬됨)"""

    date: Optional[str] = None
    """추첨일 (YYYY-MM-DD 형식)"""

    def __post_init__(self):
        """초기화 후 검증 및 정렬"""
        # 번호 유효성 검사
        if not all(1 <= num <= 45 for num in self.numbers):
            raise ValueError("로또 번호는 1-45 범위여야 합니다")

        # 중복 번호 검사
        if len(self.numbers) != len(set(self.numbers)):
            raise ValueError("중복된 번호가 있습니다")

        # 번호 정렬
        self.numbers = sorted(self.numbers)

    def __repr__(self) -> str:
        date_str = self.date
        if isinstance(self.date, datetime):
            date_str = self.date.strftime("%Y-%m-%d")
        return f"Round {self.draw_no} ({date_str}): {self.numbers}"

    def __str__(self) -> str:
        return f"Round {self.draw_no}: {self.numbers}"

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, LotteryNumber):
            return False
        return self.numbers == other.numbers

    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(tuple(self.numbers))

    # utils.data_loader.LotteryNumber와 호환성을 위한 변환 메서드
    @classmethod
    def from_data_loader(cls, data_loader_lottery: Any) -> "LotteryNumber":
        """utils.data_loader.LotteryNumber 객체에서 shared.types.LotteryNumber 객체를 생성합니다."""
        try:
            return cls(
                draw_no=data_loader_lottery.draw_no,
                numbers=data_loader_lottery.numbers,
                date=data_loader_lottery.date,
            )
        except AttributeError:
            # 필요한 속성이 없는 경우
            return cls(
                draw_no=getattr(data_loader_lottery, "draw_no", 0),
                numbers=getattr(data_loader_lottery, "numbers", []),
                date=getattr(data_loader_lottery, "date", None),
            )


@dataclass
class PatternAnalysis:
    """패턴 분석 결과 클래스"""

    frequency_map: Dict[int, float]
    """번호별 출현 빈도 (1-45)"""

    recency_map: Dict[int, float]
    """번호별 최근성 점수 (1-45)"""

    pair_frequency: Dict[Tuple[int, int], float] = field(default_factory=dict)
    """번호 쌍 출현 빈도"""

    hot_numbers: Set[int] = field(default_factory=set)
    """핫 넘버 목록"""

    cold_numbers: Set[int] = field(default_factory=set)
    """콜드 넘버 목록"""

    trending_numbers: List[int] = field(default_factory=list)
    """상승세 번호 목록"""

    clusters: List[List[int]] = field(default_factory=list)
    """번호 클러스터 목록"""

    co_occurrence: Optional[np.ndarray] = None
    """동시 출현 행렬"""

    roi_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    """ROI 행렬"""

    sum_distribution: List[int] = field(default_factory=list)
    """당첨 번호 합계 분포"""

    gap_patterns: Dict[int, float] = field(default_factory=dict)
    """번호 간격 패턴"""

    probability_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    """번호 간 조건부 확률 행렬"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """기타 메타데이터"""

    def get_normalized_frequency(self) -> np.ndarray:
        """정규화된 빈도 반환"""
        freq_array = np.zeros(45)
        for num, count in self.frequency_map.items():
            if 1 <= num <= 45:
                freq_array[num - 1] = count

        if freq_array.max() > 0:
            return freq_array / freq_array.max()
        return freq_array

    def __getitem__(self, key):
        """지원: pattern_analysis['frequency'] 형태로 접근"""
        return getattr(self, key)

    def to_dict(self) -> Dict:
        """사전 형태로 변환"""
        return {
            "frequency": self.frequency_map,
            "hot_numbers": self.hot_numbers,
            "cold_numbers": self.cold_numbers,
            "sum_distribution": self.sum_distribution,
            "gap_patterns": self.gap_patterns,
            "probability_matrix": self.probability_matrix,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PatternAnalysis":
        """사전에서 PatternAnalysis 객체를 생성합니다."""
        # 데이터가 None이면 빈 딕셔너리로 초기화
        if data is None:
            data = {}

        # 각 키가 없을 경우 기본값 설정
        if "frequency" not in data:
            data["frequency"] = {}
        if "hot_numbers" not in data:
            data["hot_numbers"] = set()
        if "cold_numbers" not in data:
            data["cold_numbers"] = set()
        if "sum_distribution" not in data:
            data["sum_distribution"] = []
        if "gap_patterns" not in data:
            data["gap_patterns"] = {}
        if "probability_matrix" not in data:
            data["probability_matrix"] = {}
        if "recency_map" not in data:
            data["recency_map"] = {}
        if "pair_frequency" not in data:
            data["pair_frequency"] = {}
        if "trending_numbers" not in data:
            data["trending_numbers"] = []
        if "clusters" not in data:
            data["clusters"] = []
        if "roi_matrix" not in data:
            data["roi_matrix"] = {}

        return cls(
            frequency_map=data["frequency"],
            recency_map=data["recency_map"],
            pair_frequency=data["pair_frequency"],
            hot_numbers=data["hot_numbers"],
            cold_numbers=data["cold_numbers"],
            sum_distribution=data["sum_distribution"],
            gap_patterns=data["gap_patterns"],
            probability_matrix=data["probability_matrix"],
            trending_numbers=data["trending_numbers"],
            clusters=data["clusters"],
            roi_matrix=data["roi_matrix"],
        )


@dataclass
class ModelPrediction:
    """모델 예측 결과 클래스"""

    numbers: List[int]
    """예측된 번호 (정렬됨)"""

    confidence: float
    """예측 신뢰도 (0-1 범위)"""

    model_type: str = "Generic"
    """예측 모델 유형 (예: "RL", "GNN", "Statistical")"""

    pattern_contributions: Dict[str, float] = field(default_factory=dict)
    """패턴별 기여도"""

    roi_estimate: float = 0.0
    """ROI 추정치"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """추가 메타데이터"""

    def __post_init__(self):
        """초기화 후 유효성 검사 및 정렬"""
        # 번호 유효성 검사
        if not all(1 <= num <= 45 for num in self.numbers):
            raise ValueError("예측 번호는 1-45 범위여야 합니다")

        # 중복 번호 검사
        if len(self.numbers) != len(set(self.numbers)):
            raise ValueError("중복된 예측 번호가 있습니다")

        # 신뢰도 범위 검사
        if not 0 <= self.confidence <= 1:
            raise ValueError("신뢰도는 0-1 범위여야 합니다")

        # 번호 정렬
        self.numbers = sorted(self.numbers)


@dataclass
class EnsemblePrediction:
    """앙상블 예측 결과 클래스"""

    numbers: List[int]
    """예측된 번호 (정렬됨)"""

    confidence: float
    """통합 신뢰도 (0-1 범위)"""

    model_contributions: Dict[str, float]
    """각 모델의 기여도"""

    individual_predictions: List[ModelPrediction] = field(default_factory=list)
    """개별 모델 예측 결과"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """추가 메타데이터"""

    def __post_init__(self):
        """초기화 후 검증 및 정렬"""
        # 번호 정렬
        self.numbers = sorted(self.numbers)


@dataclass
class SystemConfig:
    # Soft Blending 가중치
    long_term_weight: float = 0.6  # 장기 출현율 가중치 (60%)
    mid_term_weight: float = 0.23  # 중기 출현율 가중치 (23%)
    short_term_weight: float = 0.17  # 단기 출현율 가중치 (17%)

    # 기간 설정
    mid_term_period: float = 0.025  # 중기 = 최근 2.5%
    short_term_period: float = 0.012  # 단기 = 최근 1.2%

    # ROI 관련 설정
    min_roi_pairs: int = 1  # 최소 ROI 번호쌍 수
    max_roi_pairs: int = 2  # 최대 ROI 번호쌍 수
    min_cluster_pairs: int = 2  # 최소 클러스터 연결쌍 수
    premium_roi_pairs_count: int = 20  # 프리미엄 ROI 동반쌍 수

    # 패턴 제약 설정
    min_even_numbers: int = 2  # 최소 짝수 개수
    max_even_numbers: int = 4  # 최대 짝수 개수
    min_low_numbers: int = 2  # 최소 저범위(1-23) 번호 개수
    max_low_numbers: int = 4  # 최대 저범위(1-23) 번호 개수
    pattern_percentile_threshold: float = 0.95  # 패턴 허용 백분위수 임계값

    # 번호 구간 분할
    number_ranges: List[Tuple[int, int]] = field(
        default_factory=lambda: [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
    )

    # 캐시 설정
    cache_duration: int = 3600  # 초 단위
    max_cache_size: int = 1000  # 최대 캐시 항목 수

    # 경로 설정 (리팩토링 추가)
    log_dir: str = "logs"
    cache_dir: str = "data/cache"
    model_save_dir: str = "savedModels"

    # 모델 디렉토리 설정
    rl_model_dir: str = "savedModels/rl"
    gnn_model_dir: str = "savedModels/gnn"
    statistical_model_dir: str = "savedModels/statistical"

    # 보고서 디렉토리 설정
    report_dir: str = "logs/report"

    def get(self, key: str, default: Any = None) -> Any:
        """사전 호환 메서드: 속성이 존재하면 반환하고, 없으면 기본값 반환"""
        return getattr(self, key, default)

    def __post_init__(self):
        """Initialize the number ranges if not provided."""
        if self.number_ranges is None:
            self.number_ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]


class PatternFeatures(TypedDict, total=False):
    max_consecutive_length: int
    total_sum: int
    odd_count: int
    even_count: int
    gap_avg: float
    gap_std: float
    range_counts: list[int]
    cluster_overlap_ratio: float
    frequent_pair_score: float
    roi_weight: float
    consecutive_score: float
    trend_score_avg: float
    trend_score_max: float
    trend_score_min: float
    risk_score: float
    metadata: dict[str, Any]
