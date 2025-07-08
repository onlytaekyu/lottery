"""
기본 모델 클래스 (Base Model Class)

이 모듈은 모든 모델이 상속받는 기본 모델 클래스를 정의합니다.
표준 인터페이스를 제공하여 모델 간 호환성을 보장합니다.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import time
from contextlib import nullcontext

# ✅ src/utils 통합 시스템 활용
from ..utils.unified_logging import get_logger
from ..utils import (
    get_unified_memory_manager,
    get_cuda_optimizer
)

logger = get_logger(__name__)


class GPUDeviceManager:
    """
    GPU 장치 관리자 - src/utils 통합 시스템 기반 고성능 메모리 관리
    
    기존 API 호환성을 유지하면서 src/utils의 강력한 기능들을 통합:
    - 스마트 메모리 할당
    - GPU 메모리 풀링
    - 자동 메모리 정리
    - OOM 자동 복구
    """
    
    def __init__(self):
        # ✅ 기존 호환성 유지
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()
        
        # ✅ src/utils 통합 시스템 초기화
        try:
            self.memory_mgr = get_unified_memory_manager()
            self.cuda_opt = get_cuda_optimizer()
            self._unified_system_available = True
            logger.info("✅ 통합 메모리 관리 시스템 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ 통합 시스템 초기화 실패, 기본 모드로 폴백: {e}")
            self.memory_mgr = None
            self.cuda_opt = None
            self._unified_system_available = False
        
        if self.gpu_available:
            logger.info(f"✅ GPU 사용 가능: {torch.cuda.get_device_name()}")
            logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # ✅ CUDA 최적화 설정 (통합 시스템이 있는 경우)
            if self._unified_system_available and self.cuda_opt:
                self.cuda_opt.set_tf32_enabled(True)
                logger.info("🚀 TF32 최적화 활성화")
        else:
            logger.info("⚠️ GPU 사용 불가능, CPU 모드로 실행")
    
    def to_device(self, tensor_or_data: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """
        데이터를 적절한 device로 이동 (스마트 메모리 할당 적용)
        
        Args:
            tensor_or_data: 이동할 데이터
            
        Returns:
            device로 이동된 텐서
        """
        try:
            # ✅ 통합 시스템이 있으면 스마트 할당 사용
            if self._unified_system_available and self.memory_mgr:
                return self._smart_to_device(tensor_or_data)
            else:
                # 기존 방식 폴백
                return self._legacy_to_device(tensor_or_data)
                
        except Exception as e:
            logger.warning(f"⚠️ 스마트 변환 실패, 기본 방식으로 폴백: {e}")
            return self._legacy_to_device(tensor_or_data)
    
    def _smart_to_device(self, tensor_or_data: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """스마트 메모리 할당을 사용한 device 이동"""
        # ✅ 메모리 관리자 존재 확인 (타입 체커 명확화)
        if self.memory_mgr is None:
            raise RuntimeError("통합 메모리 관리자가 초기화되지 않았습니다")
        
        # 데이터 크기 추정
        if isinstance(tensor_or_data, torch.Tensor):
            size = tensor_or_data.numel()
            tensor = tensor_or_data
        elif isinstance(tensor_or_data, np.ndarray):
            size = tensor_or_data.size
            tensor = torch.from_numpy(tensor_or_data)
        elif isinstance(tensor_or_data, (list, tuple)):
            size = len(tensor_or_data)
            tensor = torch.tensor(tensor_or_data)
        else:
            raise TypeError(f"지원되지 않는 타입: {type(tensor_or_data)}")
        
        # ✅ 스마트 메모리 할당으로 device 이동
        device_type = "gpu" if self.device.type == "cuda" else "cpu"
        
        # 임시 할당 컨텍스트 사용
        with self.memory_mgr.temporary_allocation(
            size=size * 4,  # float32 기준
            prefer_device=device_type
        ) as work_tensor:
            return tensor.to(self.device, non_blocking=True)
    
    def _legacy_to_device(self, tensor_or_data: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """기존 방식의 device 이동 (폴백용)"""
        if isinstance(tensor_or_data, torch.Tensor):
            return tensor_or_data.to(self.device, non_blocking=True)
        elif isinstance(tensor_or_data, np.ndarray):
            return torch.from_numpy(tensor_or_data).to(self.device, non_blocking=True)
        elif isinstance(tensor_or_data, (list, tuple)):
            return torch.tensor(tensor_or_data).to(self.device, non_blocking=True)
        else:
            raise TypeError(f"지원되지 않는 타입: {type(tensor_or_data)}")
    
    def check_memory_usage(self) -> Dict[str, float]:
        """
        GPU 메모리 사용량 확인 (통합 시스템 정보 포함)
        
        Returns:
            메모리 사용량 정보
        """
        if not self.gpu_available:
            return {"gpu_available": False}
        
        # ✅ 기본 GPU 메모리 정보
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        result = {
            "gpu_available": True,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "usage_percent": (allocated / total) * 100
        }
        
        # ✅ 통합 시스템 정보 추가
        if self._unified_system_available and self.memory_mgr:
            try:
                unified_stats = self.memory_mgr.get_memory_status()
                result["unified_memory"] = unified_stats
                result["memory_efficiency"] = unified_stats.get("efficiency", 0.0)
                result["pool_utilization"] = unified_stats.get("pool_utilization", 0.0)
            except Exception as e:
                logger.debug(f"통합 메모리 정보 조회 실패: {e}")
        
        return result
    
    def clear_cache(self):
        """GPU 캐시 정리 (통합 시스템 연동)"""
        if self.gpu_available:
            # ✅ 통합 시스템이 있으면 스마트 정리
            if self._unified_system_available and self.memory_mgr:
                try:
                    # 통합 메모리 관리자의 정리 기능 사용
                    self.memory_mgr.cleanup_unused_memory()
                    logger.info("🧹 스마트 GPU 메모리 정리 완료")
                except Exception as e:
                    logger.warning(f"⚠️ 스마트 정리 실패, 기본 정리로 폴백: {e}")
                    torch.cuda.empty_cache()
                    logger.info("GPU 캐시 정리 완료 (기본 모드)")
            else:
                # 기존 방식
                torch.cuda.empty_cache()
                logger.info("GPU 캐시 정리 완료")
    
    def get_optimal_batch_size(self, data_size: int, model_complexity: float = 1.0) -> int:
        """
        🚀 새로운 기능: 현재 메모리 상태에 따른 최적 배치 크기 계산
        
        Args:
            data_size: 처리할 데이터 크기
            model_complexity: 모델 복잡도 (1.0 = 기본)
            
        Returns:
            최적 배치 크기
        """
        if not self.gpu_available:
            return min(32, data_size)  # CPU 기본값
        
        if self._unified_system_available and self.memory_mgr:
            try:
                # 통합 시스템의 지능적 배치 크기 계산
                memory_stats = self.memory_mgr.get_memory_status()
                gpu_util = memory_stats.get("gpu_utilization", 0.5)
                
                # GPU 사용률에 따른 동적 조정
                if gpu_util < 0.3:
                    base_batch = 128
                elif gpu_util < 0.7:
                    base_batch = 64
                else:
                    base_batch = 32
                
                # 모델 복잡도 반영
                optimal_batch = int(base_batch / model_complexity)
                
                return min(max(optimal_batch, 1), data_size)
                
            except Exception as e:
                logger.debug(f"지능적 배치 크기 계산 실패: {e}")
        
        # 폴백: 기본 계산
        memory_info = self.check_memory_usage()
        usage_percent = memory_info.get("usage_percent", 50)
        
        if usage_percent < 30:
            return min(64, data_size)
        elif usage_percent < 70:
            return min(32, data_size)
        else:
            return min(16, data_size)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        🚀 새로운 기능: 성능 통계 반환
        
        Returns:
            상세한 성능 통계
        """
        stats = {
            "device": str(self.device),
            "gpu_available": self.gpu_available,
            "unified_system": self._unified_system_available,
            "memory_info": self.check_memory_usage()
        }
        
        if self._unified_system_available:
            if self.memory_mgr:
                try:
                    stats["memory_performance"] = self.memory_mgr.get_performance_metrics()
                except Exception as e:
                    logger.debug(f"메모리 성능 통계 조회 실패: {e}")
            
            if self.cuda_opt:
                try:
                    stats["cuda_optimization"] = self.cuda_opt.get_optimization_stats()
                except Exception as e:
                    logger.debug(f"CUDA 최적화 통계 조회 실패: {e}")
        
        return stats


class BaseModel(ABC):
    """
    모든 DAEBAK_AI 모델의 기본 클래스

    이 클래스는 로또 번호 예측 모델이 구현해야 하는 기본 인터페이스를 정의합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        기본 모델 초기화 (src/utils 통합 시스템 적용)

        Args:
            config: 모델 설정
        """
        # 기본 설정
        self.config = config or {}

        # ✅ GPU 장치 관리자 초기화 (통합 시스템 포함)
        self.device_manager = GPUDeviceManager()
        self.device = self.device_manager.device

        # 모델 상태
        self.is_trained = False
        self.training_history = []
        self.model_name = self.__class__.__name__

        # ✅ 성능 최적화 설정
        self.enable_smart_batching = self.config.get("enable_smart_batching", True)
        self.auto_memory_optimization = self.config.get("auto_memory_optimization", True)

        # 모델 메타데이터 (통합 시스템 정보 포함)
        self.metadata = {
            "model_type": self.model_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "2.0.0",  # src/utils 통합으로 버전 업
            "device": str(self.device),
            "gpu_available": self.device_manager.gpu_available,
            "unified_system": self.device_manager._unified_system_available,
            "smart_features": {
                "smart_batching": self.enable_smart_batching,
                "auto_memory_optimization": self.auto_memory_optimization,
                "tf32_enabled": (self.device_manager._unified_system_available and 
                               self.device_manager.cuda_opt is not None)
            }
        }

        logger.info(f"✅ {self.model_name} 초기화 완료: 장치 {self.device}")
        
        # 통합 시스템 상태 로그
        if self.device_manager._unified_system_available:
            logger.info("🚀 통합 성능 최적화 시스템 활성화")
        else:
            logger.info("⚠️ 기본 모드로 실행 (통합 시스템 비활성화)")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 훈련 (표준 인터페이스)

        Args:
            X: 특성 벡터
            y: 레이블/타겟
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과 및 메타데이터
        """
        raise NotImplementedError("fit 메서드를 구현해야 합니다.")

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        모델 예측 수행 (표준 인터페이스)

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            예측값
        """
        raise NotImplementedError("predict 메서드를 구현해야 합니다.")

    @abstractmethod
    def save(self, path: str) -> bool:
        """
        모델 저장 (표준 인터페이스)

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        raise NotImplementedError("save 메서드를 구현해야 합니다.")

    @abstractmethod
    def load(self, path: str) -> bool:
        """
        모델 로드 (표준 인터페이스)

        Args:
            path: 로드할 모델 경로

        Returns:
            성공 여부
        """
        raise NotImplementedError("load 메서드를 구현해야 합니다.")

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 평가 (선택적 구현)

        Args:
            X: 특성 벡터
            y: 레이블/타겟
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        if not self.is_trained:
            return {"error": "훈련되지 않은 모델은 평가할 수 없습니다."}

        return {
            "message": "evaluate 메서드가 구현되지 않았습니다.",
            "status": "not_implemented",
        }

    def _ensure_directory(self, path: str) -> None:
        """
        저장 경로의 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.

        Args:
            path: 파일 경로
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"디렉토리 생성: {directory}")

    def get_feature_vector(
        self,
        feature_path: str = "data/cache/feature_vector_full.npy",
        names_path: str = "data/cache/feature_vector_full.names.json",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        특성 벡터와 특성 이름을 로드합니다.

        Args:
            feature_path: 특성 벡터 파일 경로
            names_path: 특성 이름 파일 경로

        Returns:
            특성 벡터와 특성 이름의 튜플
        """
        try:
            # 벡터 데이터 로드
            if not os.path.exists(feature_path):
                raise FileNotFoundError(
                    f"특성 벡터 파일이 존재하지 않습니다: {feature_path}"
                )

            vector = np.load(feature_path)

            # 특성 이름 로드
            if not os.path.exists(names_path):
                raise FileNotFoundError(
                    f"특성 이름 파일이 존재하지 않습니다: {names_path}"
                )

            with open(names_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)

            logger.info(
                f"특성 벡터 로드 완료: {feature_path}, 형태={vector.shape}, 특성 수={len(feature_names)}"
            )
            return vector, feature_names

        except Exception as e:
            logger.error(f"특성 벡터 로드 중 오류: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환 (통합 시스템 정보 포함)

        Returns:
            모델 메타데이터와 상태 정보
        """
        info = self.metadata.copy()
        info.update({
            "is_trained": self.is_trained,
            "training_history_length": len(self.training_history),
            "device": str(self.device),
            "gpu_memory_info": self.device_manager.check_memory_usage(),
            "performance_stats": self.device_manager.get_performance_stats(),
        })
        return info
    
    def get_optimal_batch_size(self, data_size: int, model_complexity: float = 1.0) -> int:
        """
        🚀 스마트 배치 크기 계산
        
        Args:
            data_size: 처리할 데이터 크기
            model_complexity: 모델 복잡도 (1.0 = 기본)
            
        Returns:
            최적 배치 크기
        """
        if not self.enable_smart_batching:
            return min(32, data_size)  # 스마트 배치 비활성화시 기본값
        
        return self.device_manager.get_optimal_batch_size(data_size, model_complexity)
    
    def smart_data_transfer(self, data: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """
        🚀 스마트 데이터 전송 (통합 메모리 관리 활용)
        
        Args:
            data: 전송할 데이터
            
        Returns:
            device로 이동된 텐서
        """
        if not self.auto_memory_optimization:
            # 기존 방식 사용
            return self.device_manager._legacy_to_device(data)
        
        return self.device_manager.to_device(data)
    
    def optimize_memory_usage(self):
        """
        🚀 메모리 사용량 최적화
        """
        if self.auto_memory_optimization:
            self.device_manager.clear_cache()
            logger.info(f"{self.model_name}: 메모리 최적화 완료")
        else:
            logger.debug(f"{self.model_name}: 메모리 최적화 비활성화")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        🚀 성능 요약 정보 반환
        
        Returns:
            성능 요약 통계
        """
        self.device_manager.get_performance_stats()
        memory_info = self.device_manager.check_memory_usage()
        
        summary = {
            "model_name": self.model_name,
            "device": str(self.device),
            "is_trained": self.is_trained,
            "unified_system_active": self.device_manager._unified_system_available,
            "memory_efficiency": memory_info.get("memory_efficiency", 0.0),
            "gpu_utilization": memory_info.get("usage_percent", 0.0),
            "smart_features_enabled": {
                "smart_batching": self.enable_smart_batching,
                "auto_memory_optimization": self.auto_memory_optimization
            }
        }
        
        return summary


class ThreeDigitMixin:
    """3자리 예측 모드를 위한 Mixin 클래스"""
    
    def __init__(self):
        self.supports_3digit_mode = True
        self.is_3digit_mode = False
        self.three_digit_model = None

    def enable_3digit_mode(self) -> bool:
        """3자리 예측 모드 활성화"""
        if not self.supports_3digit_mode:
            logger.warning(f"{self.model_name}은 3자리 예측 모드를 지원하지 않습니다.")
            return False
        
        self.is_3digit_mode = True
        logger.info(f"{self.model_name}: 3자리 예측 모드 활성화")
        return True

    def disable_3digit_mode(self) -> bool:
        """3자리 예측 모드 비활성화"""
        self.is_3digit_mode = False
        logger.info(f"{self.model_name}: 3자리 예측 모드 비활성화")
        return True

    def predict_3digit_combinations(
        self, X: np.ndarray, top_k: int = 100, **kwargs
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        3자리 조합 예측 (구현 필요)

        Args:
            X: 특성 벡터
            top_k: 상위 k개 조합 반환
            **kwargs: 추가 매개변수

        Returns:
            3자리 조합과 확률의 리스트
        """
        if not self.is_3digit_mode:
            raise ValueError("3자리 예측 모드가 활성화되지 않았습니다.")
        
        if self.three_digit_model is None:
            raise ValueError("3자리 예측 모델이 훈련되지 않았습니다.")
        
        # 하위 클래스에서 구현
        raise NotImplementedError("3자리 예측 메서드를 구현해야 합니다.")

    def fit_3digit_mode(
        self, X: np.ndarray, y_3digit: np.ndarray, **kwargs
    ) -> Dict[str, Any]:
        """
        3자리 예측 모드 훈련 (구현 필요)

        Args:
            X: 특성 벡터
            y_3digit: 3자리 조합 레이블
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과
        """
        if not self.supports_3digit_mode:
            return {"error": "3자리 예측 모드를 지원하지 않습니다."}
        
        # 하위 클래스에서 구현
        raise NotImplementedError("3자리 모드 훈련 메서드를 구현해야 합니다.")


class ModelWithAMP(BaseModel):
    """
    Automatic Mixed Precision (AMP)를 지원하는 모델 기본 클래스

    PyTorch 모델에 AMP를 적용하기 위한 공통 기능을 제공합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        AMP 지원 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # AMP 관련 설정
        self.use_amp = (
            self.config.get("use_amp", True) if torch.cuda.is_available() else False
        )

        # Scaler 초기화
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info(f"{self.model_name}: AMP 활성화됨")
        else:
            self.scaler = None
            logger.info(f"{self.model_name}: AMP 비활성화됨")

    def get_amp_context(self):
        """AMP 컨텍스트 반환 - nullcontext 사용으로 개선"""
        return torch.cuda.amp.autocast() if self.use_amp else nullcontext()

    def train_step_with_amp(self, model, inputs, targets, optimizer, loss_fn, **kwargs):
        """
        AMP를 적용한 훈련 단계

        Args:
            model: 훈련할 모델
            inputs: 입력 데이터
            targets: 타겟 데이터
            optimizer: 옵티마이저
            loss_fn: 손실 함수
            **kwargs: 추가 매개변수

        Returns:
            손실값
        """
        # 모델을 훈련 모드로 설정
        model.train()

        # 그래디언트 초기화
        optimizer.zero_grad()

        # 데이터를 적절한 device로 이동
        inputs = self.device_manager.to_device(inputs)
        targets = self.device_manager.to_device(targets)

        try:
            with self.get_amp_context():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            if self.use_amp and self.scaler:
                # 스케일된 그래디언트로 역전파
                self.scaler.scale(loss).backward()
                # 스케일된 그래디언트로 옵티마이저 스텝
                self.scaler.step(optimizer)
                # 스케일러 업데이트
                self.scaler.update()
            else:
                # 일반 역전파
                loss.backward()
                optimizer.step()

            return loss.item()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU 메모리 부족, 캐시 정리 후 재시도")
                self.device_manager.clear_cache()
                raise MemoryError(f"GPU 메모리 부족: {e}")
            else:
                raise


class EnsembleBaseModel(BaseModel):
    """
    앙상블 모델 기본 클래스

    여러 모델의 예측을 결합하는 앙상블 모델을 위한 기본 클래스입니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        앙상블 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 및 가중치 목록
        self.models = []
        self.weights = []

    def add_model(self, model: BaseModel, weight: float = 1.0):
        """
        앙상블에 모델 추가

        Args:
            model: 추가할 모델
            weight: 모델 가중치 (기본값: 1.0)
        """
        self.models.append(model)
        self.weights.append(weight)
        logger.info(f"앙상블에 {model.model_name} 추가 (가중치: {weight})")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모든 구성 모델 훈련

        Args:
            X: 특성 벡터
            y: 레이블/타겟
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과
        """
        results = {}

        for i, model in enumerate(self.models):
            logger.info(
                f"앙상블 구성 모델 {i+1}/{len(self.models)} ({model.model_name}) 훈련 중..."
            )
            try:
                result = model.fit(X, y, **kwargs)
                results[model.model_name] = result
            except Exception as e:
                logger.error(f"모델 {model.model_name} 훈련 실패: {e}")
                results[model.model_name] = {"error": str(e)}

        self.is_trained = all(model.is_trained for model in self.models)

        return {
            "ensemble_results": results,
            "is_trained": self.is_trained,
            "model_count": len(self.models),
            "gpu_memory_info": self.device_manager.check_memory_usage(),
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        앙상블 예측 수행

        Args:
            X: 특성 벡터
            **kwargs: 추가 매개변수

        Returns:
            앙상블 예측값
        """
        if not self.models:
            raise ValueError("앙상블에 모델이 없습니다.")

        if not self.is_trained:
            raise ValueError("훈련되지 않은 앙상블 모델입니다.")

        predictions = []

        for model, weight in zip(self.models, self.weights):
            try:
                pred = model.predict(X, **kwargs)
                predictions.append((pred, weight))
            except Exception as e:
                logger.warning(f"모델 {model.model_name} 예측 실패: {e}")
                continue

        if not predictions:
            raise RuntimeError("모든 모델 예측이 실패했습니다.")

        return self._combine_predictions(predictions)

    def _combine_predictions(self, predictions) -> np.ndarray:
        """
        모델 예측값 결합

        Args:
            predictions: (예측값, 가중치) 튜플 리스트

        Returns:
            가중 평균 예측값
        """
        if not predictions:
            raise ValueError("결합할 예측값이 없습니다.")

        # 가중 평균 계산
        weighted_sum = None
        total_weight = 0

        for pred, weight in predictions:
            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 and weighted_sum is not None else (weighted_sum or predictions[0][0])

    def save(self, path: str) -> bool:
        """
        앙상블 모델 저장

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        try:
            self._ensure_directory(path)

            # 앙상블 메타데이터 저장
            ensemble_data = {
                "model_count": len(self.models),
                "weights": self.weights,
                "model_names": [model.model_name for model in self.models],
                "metadata": self.metadata,
            }

            # 각 모델 개별 저장
            model_paths = []
            for i, model in enumerate(self.models):
                model_path = f"{path}_model_{i}_{model.model_name}"
                success = model.save(model_path)
                if success:
                    model_paths.append(model_path)
                else:
                    logger.warning(f"모델 {model.model_name} 저장 실패")

            ensemble_data["model_paths"] = model_paths

            # 앙상블 정보 저장
            with open(f"{path}_ensemble.json", "w", encoding="utf-8") as f:
                json.dump(ensemble_data, f, ensure_ascii=False, indent=2)

            logger.info(f"앙상블 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"앙상블 모델 저장 실패: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        앙상블 모델 로드

        Args:
            path: 로드할 모델 경로

        Returns:
            성공 여부
        """
        try:
            # 앙상블 정보 로드
            with open(f"{path}_ensemble.json", "r", encoding="utf-8") as f:
                ensemble_data = json.load(f)

            # 모델 리스트 초기화
            self.models = []
            self.weights = ensemble_data["weights"]

            # 각 모델 로드
            for i, model_path in enumerate(ensemble_data["model_paths"]):
                model_name = ensemble_data["model_names"][i]
                model = self._create_model_instance(model_name)
                
                if model and model.load(model_path):
                    self.models.append(model)
                else:
                    logger.warning(f"모델 {model_name} 로드 실패")

            self.is_trained = len(self.models) > 0
            self.metadata = ensemble_data.get("metadata", self.metadata)

            logger.info(f"앙상블 모델 로드 완료: {len(self.models)}개 모델")
            return True

        except Exception as e:
            logger.error(f"앙상블 모델 로드 실패: {e}")
            return False

    def _create_model_instance(self, model_type: str) -> Optional[BaseModel]:
        """모델 타입 문자열로부터 모델 인스턴스를 생성합니다."""
        if model_type == "TCNModel":
            from .dl.tcn_model import TCNModel
            # TODO: 로드된 모델에 대한 캐시 관리자 주입 방법을 고려해야 합니다.
            # 현재는 캐시를 비활성화합니다.
            return TCNModel(config=self.config, cache_manager=None)
        elif model_type == "AutoencoderModel":
            from .dl.autoencoder_model import AutoencoderModel
            return AutoencoderModel(self.config)
        elif model_type == "LightGBMModel":
            from .ml.lightgbm_model import LightGBMModel
            return LightGBMModel(self.config)
        else:
            logger.error(f"알 수 없는 모델 타입: {model_type}")
            return None


__all__ = ["BaseModel", "ModelWithAMP", "EnsembleBaseModel"]
