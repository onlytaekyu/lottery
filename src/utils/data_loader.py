"""
로또 번호 예측을 위한 데이터 로더

이 모듈은 로또 번호 데이터를 로드하고 전처리하는 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import (
    Any,
    List,
    Tuple,
    Optional,
    Union,
    Dict,
    TypeVar,
)
from pathlib import Path
import json
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from queue import Queue
import threading
from datetime import datetime

# 상대 임포트 사용
from .error_handler_refactored import get_logger
from .memory_manager import MemoryManager
from ..shared.types import LotteryNumber
from ..analysis.pattern_vectorizer import PatternVectorizer

# 타입 변수 정의
T = TypeVar("T")

# 로거 설정
logger = get_logger(__name__)

# 모듈 수준 캐시 변수
_DRAW_HISTORY_CACHE: Dict[str, List[LotteryNumber]] = {}
_VECTORIZED_DATA_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


class DataQualityValidator:
    """데이터 품질 검증 클래스"""

    def __init__(self):
        self.logger = get_logger(__name__)

    def validate_lottery_data(self, data: List[LotteryNumber]) -> Dict[str, Any]:
        """
        로또 데이터 품질 검증

        Args:
            data: 로또 번호 데이터 리스트

        Returns:
            검증 결과 딕셔너리
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_records": len(data),
                "valid_records": 0,
                "invalid_records": 0,
            },
        }

        if not data:
            validation_result["is_valid"] = False
            validation_result["errors"].append("데이터가 비어 있습니다")
            return validation_result

        valid_count = 0
        invalid_count = 0
        draw_numbers = set()

        for i, lottery_draw in enumerate(data):
            record_valid = True

            # 1. 번호 범위 검증 (1-45)
            if hasattr(lottery_draw, "numbers"):
                numbers = lottery_draw.numbers
                for num in numbers:
                    if not isinstance(num, int) or num < 1 or num > 45:
                        validation_result["errors"].append(
                            f"회차 {i+1}: 잘못된 번호 범위 - {num} (1-45 범위 벗어남)"
                        )
                        record_valid = False

                # 2. 중복 번호 검증
                if len(numbers) != len(set(numbers)):
                    validation_result["errors"].append(
                        f"회차 {i+1}: 중복 번호 발견 - {numbers}"
                    )
                    record_valid = False

                # 3. 번호 개수 검증 (6개)
                if len(numbers) != 6:
                    validation_result["errors"].append(
                        f"회차 {i+1}: 번호 개수 오류 - {len(numbers)}개 (6개 필요)"
                    )
                    record_valid = False
            else:
                validation_result["errors"].append(f"회차 {i+1}: 번호 속성이 없습니다")
                record_valid = False

            # 4. 회차 연속성 검증
            if hasattr(lottery_draw, "draw_no"):
                draw_no = lottery_draw.draw_no
                if draw_no in draw_numbers:
                    validation_result["errors"].append(f"중복 회차 번호: {draw_no}")
                    record_valid = False
                draw_numbers.add(draw_no)

            if record_valid:
                valid_count += 1
            else:
                invalid_count += 1

        # 5. 데이터 완정성 검증
        if len(draw_numbers) != len(data):
            validation_result["warnings"].append(
                "회차 번호 불일치: 일부 회차에 번호가 없거나 중복됩니다"
            )

        # 통계 업데이트
        validation_result["stats"]["valid_records"] = valid_count
        validation_result["stats"]["invalid_records"] = invalid_count

        # 전체 유효성 판단
        if invalid_count > 0:
            validation_result["is_valid"] = False

        # 경고 임계값 검사
        if invalid_count / len(data) > 0.05:  # 5% 이상 오류
            validation_result["warnings"].append(
                f"높은 오류율: {invalid_count}/{len(data)} ({invalid_count/len(data)*100:.1f}%)"
            )

        return validation_result

    def log_validation_result(self, result: Dict[str, Any]):
        """검증 결과 로깅"""
        if result["is_valid"]:
            self.logger.info(
                f"✅ 데이터 검증 통과: {result['stats']['valid_records']}개 유효 레코드"
            )
        else:
            self.logger.error(
                f"❌ 데이터 검증 실패: {len(result['errors'])}개 오류 발견"
            )
            for error in result["errors"][:5]:  # 최대 5개만 표시
                self.logger.error(f"  - {error}")

            if len(result["errors"]) > 5:
                self.logger.error(f"  ... 및 {len(result['errors']) - 5}개 추가 오류")

        if result["warnings"]:
            for warning in result["warnings"]:
                self.logger.warning(f"⚠️ {warning}")


def load_draw_history(
    file_path: Optional[str] = None, validate_data: bool = True
) -> List[LotteryNumber]:
    """
    로또 당첨 번호 이력을 로드합니다.

    Args:
        file_path: CSV 파일 경로 (None이면 기본 경로 사용)
        validate_data: 데이터 품질 검증 여부

    Returns:
        당첨 번호 이력 리스트

    Raises:
        FileNotFoundError: 파일을 찾을 수 없는 경우
        ValueError: 데이터 검증 실패 시
    """
    logger = get_logger(__name__)

    # 기본 파일 경로 설정
    if file_path is None:
        project_root = Path(__file__).parent.parent.parent
        file_path = project_root / "data" / "raw" / "lottery.csv"

    if not Path(file_path).exists():
        error_msg = f"로또 데이터 파일을 찾을 수 없습니다: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"로또 데이터 로드 중: {file_path}")

    try:
        # CSV 파일 읽기
        import pandas as pd

        df = pd.read_csv(file_path, encoding="utf-8")

        # 컬럼 확인 - 실제 CSV 파일의 컬럼명에 맞게 수정
        required_columns = ["seqNum", "num1", "num2", "num3", "num4", "num5", "num6"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg = f"필수 컬럼이 없습니다: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # LotteryNumber 객체 생성
        lottery_numbers = []
        for _, row in df.iterrows():
            try:
                numbers = [
                    int(row["num1"]),
                    int(row["num2"]),
                    int(row["num3"]),
                    int(row["num4"]),
                    int(row["num5"]),
                    int(row["num6"]),
                ]

                lottery_number = LotteryNumber(
                    draw_no=int(row["seqNum"]),  # seqNum을 draw_no로 매핑
                    numbers=numbers,
                )

                lottery_numbers.append(lottery_number)

            except (ValueError, TypeError) as e:
                logger.warning(f"행 {row.name} 처리 중 오류: {e}")
                continue

        logger.info(f"데이터 로드 완료: {len(lottery_numbers)}개 회차")

        # 데이터 품질 검증
        if validate_data:
            validator = DataQualityValidator()
            validation_result = validator.validate_lottery_data(lottery_numbers)
            validator.log_validation_result(validation_result)

            if not validation_result["is_valid"]:
                error_msg = (
                    f"데이터 품질 검증 실패: {len(validation_result['errors'])}개 오류"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        return lottery_numbers

    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise


def clear_draw_history_cache() -> None:
    """
    로또 당첨 번호 이력 캐시를 초기화합니다.
    """
    global _DRAW_HISTORY_CACHE
    _DRAW_HISTORY_CACHE = {}
    logger.info("로또 당첨 번호 이력 캐시 초기화 완료")


def _generate_sample_data(count: int = 100) -> List[LotteryNumber]:
    """
    샘플 로또 데이터 생성 (실제 데이터를 찾을 수 없는 경우 사용)

    Args:
        count: 생성할 샘플 데이터 개수

    Returns:
        List[LotteryNumber]: 샘플 로또 당첨 번호 리스트
    """
    np.random.seed(42)  # 결정론적인 결과를 위한 시드 설정
    lottery_numbers = []

    for draw_no in range(1, count + 1):
        # 1~45 사이의 중복 없는 6개의 번호 생성
        numbers = sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())

        # 현재 날짜로부터 샘플 날짜 생성 (일주일씩 과거로 이동)
        sample_date = datetime.now()
        sample_date = sample_date.replace(
            day=sample_date.day - (draw_no * 7)
        )  # 7일씩 이전 날짜
        date_str = sample_date.strftime("%Y-%m-%d")

        lottery_numbers.append(
            LotteryNumber(
                draw_no=draw_no,
                numbers=numbers,
                date=date_str,
            )
        )

    logger.warning(f"{count}개의 샘플 로또 데이터 생성됨")
    return lottery_numbers


@dataclass
class DataConfig:
    """데이터 설정"""

    min_number: int = 1
    max_number: int = 45
    sequence_length: int = 6
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    memory_map: bool = True
    cache_size: int = 1 << 30  # 1GB
    historical_data_path: str = ""  # 로또 데이터 CSV 파일 경로 추가

    # GPU 관련 설정
    use_gpu: bool = True  # GPU 사용 여부 (사용 가능한 경우)
    gpu_buffer_size: int = 1 << 28  # 256MB의 GPU 메모리 버퍼
    use_pinned_memory: bool = True  # GPU 전송을 위한 고정 메모리 사용
    use_async_loading: bool = True  # 비동기 데이터 로딩
    mixed_precision: bool = True  # 혼합 정밀도 사용
    tensor_cores: bool = True  # 텐서 코어 사용 (지원되는 경우)

    # CPU 멀티스레딩 관련 설정
    use_multithreading: bool = True  # 멀티스레딩 사용
    thread_pool_size: int = max(4, min(16, (os.cpu_count() or 4)))  # 스레드 풀 크기
    parallel_transforms: bool = True  # 변환 병렬 처리

    def __post_init__(self):
        """설정 초기화 후처리"""
        # GPU 사용 가능 여부 확인
        self.gpu_available = torch.cuda.is_available()

        # 기본 데이터 경로 설정 (비어있을 경우)
        if not self.historical_data_path:
            self.historical_data_path = str(
                Path(__file__).parent.parent.parent / "data" / "raw" / "lottery.csv"
            )

        # GPU를 사용할 수 없는 경우 CPU 설정 최적화
        if not self.gpu_available:
            self.use_gpu = False
            self.use_pinned_memory = False

            # CPU 코어 수에 맞게 워커 수 조정
            cpu_count = os.cpu_count() or 4
            self.num_workers = max(2, min(cpu_count - 1, 8))
            self.thread_pool_size = max(2, min(cpu_count - 1, 16))

        # 워커 수가 너무 많으면 성능이 저하될 수 있음
        if self.num_workers > 8:
            self.prefetch_factor = max(1, min(3, self.prefetch_factor))


class LotteryJSONEncoder(json.JSONEncoder):
    """JSON 직렬화를 위한 사용자 정의 인코더"""

    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):  # type: ignore
            return obj.item()
        return super().default(obj)


class LotteryDataset(Dataset):
    """로또 번호 데이터셋"""

    def __init__(
        self,
        data_path: str,
        transform: Optional[Any] = None,
        to_tensor: bool = True,
        device: Optional[torch.device] = None,
        config: Optional[DataConfig] = None,
    ):
        """
        로또 데이터셋 초기화

        Args:
            data_path: 데이터 파일 경로
            transform: 데이터 변환 함수
            to_tensor: 데이터를 텐서로 변환 여부
            device: 데이터를 저장할 장치
            config: 데이터 설정
        """
        self.data_path = data_path
        self.transform = transform
        self.to_tensor = to_tensor
        self.device = device
        self.config = config or DataConfig()
        self.tempfile = None
        self.preloaded = False
        self.targets = None
        self.thread_pool = None

        # 데이터 로드
        self.data = load_draw_history(data_path)

    def __len__(self) -> int:
        """데이터셋 길이 반환"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[LotteryNumber, torch.Tensor]:
        """인덱스로 항목 접근"""
        return self.data[idx]

    def get_data(self) -> Union[List[LotteryNumber], torch.Tensor]:
        """전체 데이터 반환"""
        return self.data

    def get_all_data(self) -> Union[List[LotteryNumber], torch.Tensor]:
        """전체 데이터 반환 (별칭)"""
        return self.data

    def get_batch(
        self, batch_size: int
    ) -> Union[
        List[LotteryNumber], torch.Tensor, List[Union[LotteryNumber, torch.Tensor]]
    ]:
        """무작위 배치 데이터 반환"""
        if batch_size >= len(self.data):
            return self.data

        # 무작위 인덱스 선택
        indices = np.random.choice(len(self.data), batch_size, replace=False)
        return [self.data[i] for i in indices]

    def pin_memory(self) -> "LotteryDataset":
        """
        데이터셋 메모리 고정 (GPU 전송 최적화)

        Returns:
            고정 메모리 데이터셋
        """
        if (
            not self.preloaded
            and self.config.use_pinned_memory
            and torch.cuda.is_available()
        ):
            self.config.use_pinned_memory = True
        return self

    def to(self, device: torch.device) -> "LotteryDataset":
        """
        데이터셋을 특정 장치로 이동

        Args:
            device: 대상 장치

        Returns:
            이동된 데이터셋
        """
        self.device = device
        if self.preloaded:
            if isinstance(self.data, torch.Tensor):
                self.data = self.data.to(device)
            if isinstance(self.targets, torch.Tensor):
                self.targets = self.targets.to(device)
        return self

    def __del__(self):
        """소멸자"""
        try:
            # 리소스 정리
            if hasattr(self, "thread_pool") and self.thread_pool is not None:
                self.thread_pool.shutdown(wait=False)

            if hasattr(self, "tempfile") and self.tempfile is not None:
                import os

                try:
                    os.unlink(self.tempfile.name)
                except Exception:
                    pass

            # GPU 메모리 해제
            if (
                hasattr(self, "preloaded")
                and self.preloaded
                and hasattr(self, "device")
                and self.device is not None
                and self.device.type == "cuda"
            ):
                if hasattr(self, "data"):
                    del self.data
                if hasattr(self, "targets") and self.targets is not None:
                    del self.targets
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"데이터셋 정리 중 오류 발생: {str(e)}")


class DataManager:
    """데이터 관리자"""

    def __init__(self, config=None):
        """
        데이터 관리자 초기화

        Args:
            config: 데이터 설정 (선택 사항)
        """
        # 설정 초기화
        self.config = config or DataConfig()

        # 데이터 관련 변수 초기화
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.scaler = MinMaxScaler()

        # 데이터 변환 관련 변수
        self.transform = None
        self.to_tensor = False

        # 비동기 로딩 관련 변수
        self.load_queue = Queue()
        self.worker_thread = None
        self.is_loading = False
        self.memory_manager = MemoryManager()
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """데이터 로드 - 비동기 처리 지원"""
        try:
            # historical_data_path가 없는 경우 기본 경로 사용
            if (
                hasattr(self.config, "historical_data_path")
                and self.config.historical_data_path
            ):
                data_path = Path(self.config.historical_data_path)
            else:
                # 상대 경로 사용
                data_path = (
                    Path(__file__).parent.parent.parent / "data" / "raw" / "lottery.csv"
                )

            if not data_path.exists():
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

            # 메모리 맵핑 모드 사용
            self.data = pd.read_csv(data_path, memory_map=self.config.memory_map)

            # 데이터 유효성 검사
            self._validate_data(self.data)

            # 데이터 전처리
            self._preprocess_data()

            logger.info(f"데이터 로드 완료: {len(self.data)} 행")
            logger.info(f"컬럼: {self.data.columns.tolist()}")

        except Exception as e:
            logger.error(f"데이터 로드 실패: {str(e)}")
            raise

    def load_default_data(self) -> pd.DataFrame:
        """
        기본 로또 데이터 로드

        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
        # 상대 경로 사용
        data_path = self.data_dir / "raw" / "lottery.csv"

        if not data_path.exists():
            # 디렉토리 생성
            data_path.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(f"기본 데이터 파일이 없습니다: {data_path}")
            # 샘플 데이터 생성
            sample_data = _generate_sample_data(200)
            df = self._convert_lottery_numbers_to_df(sample_data)
            # 샘플 데이터 저장
            if not df.empty:
                df.to_csv(data_path, index=False)
                logger.info(f"샘플 데이터를 저장했습니다: {data_path}")
            return df

        # CSV 파일 로드
        return pd.read_csv(data_path)

    def _validate_data(self, df: pd.DataFrame) -> None:
        """데이터 유효성 검사"""
        # 숫자 범위 검사
        number_columns = ["num1", "num2", "num3", "num4", "num5", "num6"]
        for col in number_columns:
            if (
                not df[col]
                .between(self.config.min_number, self.config.max_number)
                .all()
            ):
                raise ValueError(f"숫자 범위가 잘못되었습니다: {col}")

        # 중복 번호 검사
        for idx, row in df.iterrows():
            numbers = [row[col] for col in number_columns]
            if len(set(numbers)) != len(numbers):
                raise ValueError(
                    f"중복된 번호가 있습니다: {numbers} (회차: {row['seqNum']})"
                )

        # 회차 순서 검사
        if not df["seqNum"].is_monotonic_increasing:
            raise ValueError("회차가 순차적으로 증가하지 않습니다.")

    def split_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터를 학습, 검증, 테스트 세트로 분할"""
        if data is None or data.empty:
            raise ValueError("분할할 데이터가 없습니다.")

        logger.info(f"데이터 분할 시작 - 전체 데이터: {len(data)} 행")

        # 데이터 정렬
        if "seqNum" in data.columns:
            data = data.sort_values(by="seqNum").reset_index(drop=True)
        else:
            logger.warning("'seqNum' 컬럼이 없습니다. 데이터를 그냥 사용합니다.")
            data = data.reset_index(drop=True)

        # 테스트 세트 분할
        train_val_data, test_data = train_test_split(
            data, test_size=test_size, shuffle=False, random_state=random_state
        )

        # 검증 세트 분할
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_size, shuffle=False, random_state=random_state
        )

        logger.info(
            f"데이터 분할 완료 - 학습: {len(train_data)}, 검증: {len(val_data)}, 테스트: {len(test_data)}"
        )
        return train_data, val_data, test_data

    def _preprocess_data(self, batch_data: Optional[pd.DataFrame] = None) -> None:
        """데이터 전처리"""
        try:
            # 메모리 사용량 확인
            if not self.memory_manager.check_memory_for_batch(self.config.batch_size):
                self.config.batch_size = self.memory_manager.get_safe_batch_size()
                logger.warning(
                    f"메모리 부족으로 배치 크기를 {self.config.batch_size}로 조정합니다."
                )

            # 데이터 전처리
            data_to_process = batch_data if batch_data is not None else self.data
            if data_to_process is None:
                raise ValueError(
                    "데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요."
                )
            number_columns = ["num1", "num2", "num3", "num4", "num5", "num6"]
            data_to_process[number_columns] = self.scaler.fit_transform(
                data_to_process[number_columns]
            )

        except Exception as e:
            logger.error(f"데이터 전처리 실패: {str(e)}")
            raise

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """학습 데이터 반환"""
        try:
            if self.data is None:
                raise ValueError(
                    "데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요."
                )
            number_columns = ["num1", "num2", "num3", "num4", "num5", "num6"]
            X = self.data[number_columns].values
            y = (
                self.data[number_columns].shift(-1).values[:-1]
            )  # 다음 회차 번호를 타겟으로
            return X[:-1], y
        except Exception as e:
            logger.error(f"학습 데이터 생성 실패: {str(e)}")
            raise

    def get_latest_numbers(self) -> List[int]:
        """최신 당첨 번호 반환"""
        try:
            if self.data is None:
                raise ValueError(
                    "데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요."
                )
            number_columns = ["num1", "num2", "num3", "num4", "num5", "num6"]
            latest = np.array(self.data.iloc[-1][number_columns].values)
            return self.scaler.inverse_transform(latest.reshape(1, -1))[0].tolist()
        except Exception as e:
            logger.error(f"최신 번호 조회 실패: {str(e)}")
            raise

    def save_processed_data(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        전처리된 데이터 저장

        Args:
            file_path: 저장할 파일 경로 (기본값: None, data/processed/lottery_processed.npz 사용)
        """
        if self.X_train is None or self.y_train is None:
            logger.error("저장할 전처리 데이터가 없습니다")
            return

        if file_path is None:
            # 상대 경로 사용
            file_path = self.data_dir / "processed" / "lottery_processed.npz"
            # 디렉토리 생성
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # None이 아닌 데이터만 저장
            save_dict = {"X_train": self.X_train, "y_train": self.y_train}

            # None이 아닌 경우에만 추가
            if self.X_val is not None:
                save_dict["X_val"] = self.X_val
            if self.y_val is not None:
                save_dict["y_val"] = self.y_val
            if self.X_test is not None:
                save_dict["X_test"] = self.X_test
            if self.y_test is not None:
                save_dict["y_test"] = self.y_test

            np.savez(file_path, **save_dict)  # type: ignore
            logger.info(f"전처리된 데이터 저장 완료: {file_path}")
        except Exception as e:
            logger.error(f"데이터 저장 중 오류 발생: {e}")

    def load_async(self, callback=None) -> None:
        """비동기 데이터 로드"""

        def _load_data_thread():
            try:
                self.load_data()
                if callback:
                    callback()
            except Exception as e:
                logger.error(f"비동기 데이터 로드 실패: {str(e)}")

        self.worker_thread = threading.Thread(target=_load_data_thread)
        self.worker_thread.start()

    def batch_process(self, batch_size: int = 1000, process_fn=None) -> None:
        """배치 처리"""
        try:
            if self.data is None:
                raise ValueError(
                    "데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요."
                )
            if process_fn is None:
                process_fn = self._preprocess_data

            # 배치 크기 조정
            if not self.memory_manager.check_memory_for_batch(batch_size):
                batch_size = self.memory_manager.get_safe_batch_size()
                logger.warning(
                    f"메모리 부족으로 배치 크기를 {batch_size}로 조정합니다."
                )

            # 배치 처리
            for i in range(0, len(self.data), batch_size):
                batch = self.data.iloc[i : i + batch_size]
                process_fn(batch)

        except Exception as e:
            logger.error(f"배치 처리 실패: {str(e)}")
            raise

    def get_training_data_async(self, callback=None) -> None:
        """비동기 학습 데이터 생성"""

        def _prepare_training_data_thread():
            try:
                X, y = self.get_training_data()
                if callback:
                    callback(X, y)
            except Exception as e:
                logger.error(f"비동기 학습 데이터 생성 실패: {str(e)}")

        self.worker_thread = threading.Thread(target=_prepare_training_data_thread)
        self.worker_thread.start()

    def get_loaders(self) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        학습, 검증, 테스트용 DataLoader 반환

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: 학습, 검증, 테스트 데이터 로더
        """
        try:
            # 데이터 준비
            if not hasattr(self, "X_train") or self.X_train is None:
                # 데이터 로드
                self.load_data()
                # 데이터 전처리 및 분할
                df = self.load_default_data()
                train_df, val_df, test_df = self.split_data(df)

                # 데이터 전처리
                X_train, y_train = self._process_dataframe(train_df)
                X_val, y_val = self._process_dataframe(val_df)
                X_test, y_test = self._process_dataframe(test_df)

                # 객체 속성으로 저장
                self.X_train, self.y_train = X_train, y_train
                self.X_val, self.y_val = X_val, y_val
                self.X_test, self.y_test = X_test, y_test

            # None 체크 추가
            if (
                self.X_train is None
                or self.y_train is None
                or self.X_val is None
                or self.y_val is None
                or self.X_test is None
                or self.y_test is None
            ):
                raise ValueError(
                    "데이터 준비 중 오류가 발생했습니다. 데이터가 None입니다."
                )

            # 배치 크기 설정 (메모리 상태에 따라 안전한 크기 계산)
            batch_size = self.config.batch_size

            # 메모리 관리자 가져오기
            memory_manager = MemoryManager()

            # 메모리 상태에 따라 안전한 배치 크기 계산
            try:
                # 안전한 배치 크기 계산 시도
                memory_batch_size = memory_manager.get_safe_batch_size(batch_size)
                if memory_batch_size < batch_size:
                    logger.info(
                        f"메모리 상태에 따라 배치 크기 조정: {batch_size} → {memory_batch_size}"
                    )
                    batch_size = memory_batch_size
            except Exception as e:
                logger.warning(f"배치 크기 자동 조정 실패, 기본값 사용: {str(e)}")

            # 데이터셋 생성
            torch_device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
            )

            # 기존 데이터 경로를 문자열로 변환 또는 기본 경로 사용
            train_data_path = str(Path(self.data_dir) / "processed" / "train_data.csv")
            val_data_path = str(Path(self.data_dir) / "processed" / "val_data.csv")
            test_data_path = str(Path(self.data_dir) / "processed" / "test_data.csv")

            # 데이터 저장
            try:
                # 필요시 디렉토리 생성
                Path(train_data_path).parent.mkdir(parents=True, exist_ok=True)

                # ndarrays를 CSV로 저장
                pd.DataFrame(self.X_train).to_csv(train_data_path, index=False)
                pd.DataFrame(self.X_val).to_csv(val_data_path, index=False)
                pd.DataFrame(self.X_test).to_csv(test_data_path, index=False)
            except Exception as e:
                logger.warning(f"데이터셋 변환 중 오류: {str(e)}")
                # 오류 발생 시 기본 데이터셋 사용
                train_data_path = str(Path(self.data_dir) / "raw" / "lottery.csv")
                val_data_path = train_data_path
                test_data_path = train_data_path

            train_dataset = LotteryDataset(
                data_path=train_data_path,
                transform=self.transform,
                to_tensor=self.to_tensor,
                device=torch_device,
                config=self.config,
            )

            val_dataset = LotteryDataset(
                data_path=val_data_path,
                transform=self.transform,
                to_tensor=self.to_tensor,
                device=torch_device,
                config=self.config,
            )

            test_dataset = LotteryDataset(
                data_path=test_data_path,
                transform=self.transform,
                to_tensor=self.to_tensor,
                device=torch_device,
                config=self.config,
            )

            # 데이터 로더 생성
            train_loader = TorchDataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=self.config.shuffle,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and torch.cuda.is_available(),
                drop_last=self.config.drop_last,
                prefetch_factor=self.config.prefetch_factor,
                persistent_workers=self.config.persistent_workers
                and self.config.num_workers > 0,
            )

            # 검증 및 테스트 데이터 로더는 무작위 섞기 없이 생성
            val_loader = TorchDataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and torch.cuda.is_available(),
                prefetch_factor=self.config.prefetch_factor,
                persistent_workers=self.config.persistent_workers
                and self.config.num_workers > 0,
            )

            test_loader = TorchDataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and torch.cuda.is_available(),
                prefetch_factor=self.config.prefetch_factor,
                persistent_workers=self.config.persistent_workers
                and self.config.num_workers > 0,
            )

            logger.info(
                f"데이터 로더 생성 완료: 학습={len(train_loader.dataset)}개, "  # type: ignore
                f"검증={len(val_loader.dataset)}개, 테스트={len(test_loader.dataset)}개"  # type: ignore
            )

            return train_loader, val_loader, test_loader

        except Exception as e:
            logger.error(f"데이터 로더 생성 중 오류 발생: {str(e)}")
            raise

    def _process_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """데이터프레임을 처리하여 특성과 타겟으로 변환합니다."""
        try:
            # 특성 추출
            X = df.iloc[:, :-1].values
            # 타겟 추출 - np.array로 명시적 변환하여 타입 일치시킴
            y = np.array(df.iloc[:, -1].values)
            # 필요에 따라 추가 전처리 수행
            return X, y
        except Exception as e:
            logger.error(f"데이터프레임 처리 중 오류: {str(e)}")
            # 기본값으로 빈 배열 반환
            return np.array([]), np.array([])

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """스케일링 역변환"""
        try:
            return self.scaler.inverse_transform(data)
        except Exception as e:
            logger.error(f"스케일링 역변환 실패: {str(e)}")
            raise

    def __del__(self):
        """리소스 정리"""
        self.is_loading = True
        if self.worker_thread:
            self.worker_thread.join()
        # clear 메서드 대신 cleanup 메서드 호출로 변경
        if hasattr(self.memory_manager, "cleanup"):
            self.memory_manager.cleanup()

    def _convert_lottery_numbers_to_df(
        self, lottery_numbers: List[LotteryNumber]
    ) -> pd.DataFrame:
        """
        로또 번호 리스트를 데이터프레임으로 변환

        Args:
            lottery_numbers: 로또 번호 객체 리스트

        Returns:
            pd.DataFrame: 변환된 데이터프레임
        """
        data = []
        for ln in lottery_numbers:
            row = {
                "draw_number": ln.draw_no,
                "date": (
                    ln.date.strftime("%Y-%m-%d")
                    if isinstance(ln.date, datetime)
                    else ln.date
                ),
            }
            # 번호 추가
            for i, num in enumerate(ln.numbers, 1):
                row[f"number{i}"] = num

            data.append(row)

        return pd.DataFrame(data)


class DataLoader:
    """로또 데이터 로더 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        데이터 로더 초기화

        Args:
            config: 설정 객체
        """
        # 기본 설정
        default_config = {
            "data_path": str(
                Path(__file__).parent.parent.parent / "data" / "raw" / "lottery.csv"
            ),
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
        }

        # 설정 병합
        self.config = default_config.copy()
        if config:
            if "data" in config:
                self.config.update(config["data"])
            else:
                self.config.update(config)

        # 데이터 로드
        self.data = load_draw_history(file_path=self.config.get("data_path"))
        logger.info(f"데이터 로더 초기화 완료: {len(self.data)}개 데이터")

    def get_train_val_split(self) -> Tuple[List[LotteryNumber], List[LotteryNumber]]:
        """
        학습 및 검증 데이터 분할

        Returns:
            Tuple[List[LotteryNumber], List[LotteryNumber]]: 학습 데이터, 검증 데이터
        """
        if not self.data:
            logger.error("데이터가 없습니다.")
            return [], []

        # 데이터 개수 계산
        total_count = len(self.data)
        train_count = int(total_count * self.config["train_ratio"])
        val_count = int(total_count * self.config["val_ratio"])

        # 데이터 분할 (시간 순서 유지)
        train_data = self.data[:train_count]
        val_data = self.data[train_count : train_count + val_count]

        logger.info(f"데이터 분할: 학습 {len(train_data)}개, 검증 {len(val_data)}개")
        return train_data, val_data

    def get_test_data(self) -> List[LotteryNumber]:
        """
        테스트 데이터 가져오기

        Returns:
            List[LotteryNumber]: 테스트 데이터
        """
        if not self.data:
            logger.error("데이터가 없습니다.")
            return []

        # 데이터 개수 계산
        total_count = len(self.data)
        train_count = int(total_count * self.config["train_ratio"])
        val_count = int(total_count * self.config["val_ratio"])

        # 테스트 데이터 (나머지)
        test_data = self.data[train_count + val_count :]

        logger.info(f"테스트 데이터: {len(test_data)}개")
        return test_data

    def get_all_data(self) -> List[LotteryNumber]:
        """
        전체 데이터 가져오기

        Returns:
            List[LotteryNumber]: 전체 데이터
        """
        return self.data

    def get_latest_data(self, count: int = 10) -> List[LotteryNumber]:
        """
        최근 데이터 가져오기

        Args:
            count: 가져올 데이터 개수

        Returns:
            List[LotteryNumber]: 최근 데이터
        """
        if not self.data:
            logger.error("데이터가 없습니다.")
            return []

        # 최근 데이터 반환
        return self.data[-count:]

    def get_lottery_numbers(self) -> List[LotteryNumber]:
        """
        로또 번호 데이터 가져오기 (전체 데이터와 동일)

        Returns:
            List[LotteryNumber]: 로또 번호 데이터
        """
        return self.data


def load_vectorized_training_data(
    config: Any, force_reload: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    벡터화된 학습 데이터를 로드합니다.
    이 데이터는 모델 학습을 위한 특성 행렬과 라벨 벡터로 구성됩니다.

    Args:
        config: 설정 객체
        force_reload: 캐시를 무시하고 강제로 다시 로드할지 여부

    Returns:
        Tuple[np.ndarray, np.ndarray]: 특성 행렬(X), 라벨 벡터(y)
    """
    # 경로 설정
    cache_key = "default_vectorized_data"

    # 캐시 확인
    if not force_reload and cache_key in _VECTORIZED_DATA_CACHE:
        logger.info("캐시된 벡터화 데이터 사용")
        return _VECTORIZED_DATA_CACHE[cache_key]

    try:
        logger.info("벡터화된 학습 데이터 생성 중...")

        # 과거 당첨 번호 데이터 로드
        historical_data = load_draw_history()
        if not historical_data:
            logger.error("과거 당첨 번호 데이터를 로드할 수 없습니다.")
            # 비어있는 데이터 반환 (실패 시)
            return np.zeros((0, 10), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        # 패턴 벡터라이저 초기화
        pattern_vectorizer = PatternVectorizer(config)

        # 특성 벡터와 라벨 준비
        X = []  # 특성 행렬
        y = []  # 라벨 벡터

        # 제외할 회차 수 (가장 최근 회차는 미래 예측용으로 사용)
        exclude_recent = 1

        # 벡터화된 데이터 생성
        for i in range(exclude_recent, len(historical_data)):
            # 현재 회차를 위한 특성 벡터 생성
            current_draw = historical_data[i]
            prev_draws = historical_data[
                i + 1 :
            ]  # 과거 데이터 (최신 회차가 리스트 앞에 있으므로 i+1부터 사용)

            # 1~45 각 번호에 대한 특성 벡터와 라벨 생성
            for num in range(1, 46):
                # 번호가 당첨되었는지 여부 (라벨)
                is_winning = 1 if num in current_draw.numbers else 0

                # 번호에 대한 특성 벡터 생성
                features = pattern_vectorizer.vectorize_number(num, prev_draws)

                # 데이터셋에 추가
                X.append(features)
                y.append(is_winning)

        # 넘파이 배열로 변환
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.int32)

        # 로그 출력
        logger.info(
            f"벡터화된 학습 데이터 생성 완료: X 형태 {X_array.shape}, y 형태 {y_array.shape}"
        )

        # 결과 캐시에 저장
        _VECTORIZED_DATA_CACHE[cache_key] = (X_array, y_array)

        return X_array, y_array

    except Exception as e:
        logger.error(f"벡터화된 학습 데이터 생성 중 오류 발생: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())
        # 비어있는 데이터 반환 (실패 시)
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0,), dtype=np.int32)
