"""
src/models 모듈 업그레이드 템플릿
src/utils 시스템을 완전히 활용한 고성능 모델 구현 가이드
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ✅ src/utils 핵심 API 완전 활용
from src.utils import (
    get_unified_memory_manager,
    get_unified_async_manager,
    get_enhanced_process_pool,
    get_cuda_optimizer,
    get_feature_validator,
    get_config_validator,
    initialize_all_systems
)
from src.utils.unified_logging import get_logger
from src.shared.types import ModelPrediction


@dataclass
class ModelConfig:
    """모델 설정"""
    use_gpu: bool = True
    use_tensorrt: bool = True
    use_amp: bool = True
    batch_size: int = 64
    max_batch_size: int = 1024
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    enable_caching: bool = True
    optimize_memory: bool = True
    async_inference: bool = True


class OptimizedModelTemplate(nn.Module, ABC):
    """
    🚀 src/utils 완전 활용 모델 템플릿
    
    모든 models 모듈이 따라야 할 표준 구조:
    - TensorRT 자동 최적화
    - GPU 메모리 최적화
    - 비동기 배치 추론
    - 자동 데이터 검증
    - 스마트 메모리 관리
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        
        self.config = config or ModelConfig()
        
        # ✅ 1. src/utils 시스템 초기화
        self.memory_mgr = get_unified_memory_manager()
        self.async_mgr = get_unified_async_manager()
        self.process_pool = get_enhanced_process_pool()
        self.cuda_opt = get_cuda_optimizer()
        self.validator = get_feature_validator()
        self.config_validator = get_config_validator()
        self.logger = get_logger(__name__)
        
        # ✅ 2. 디바이스 설정
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda_opt.set_tf32_enabled(True)
        else:
            self.device = torch.device("cpu")
        
        # ✅ 3. 모델 상태 관리
        self.is_optimized = False
        self.optimized_model = None
        self.model_cache = {}
        
        # ✅ 4. 성능 통계
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.batch_sizes = []
        
        self.logger.info(f"✅ {self.__class__.__name__} 초기화 완료 (GPU: {self.config.use_gpu})")
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """모델 구조 정의 (하위 클래스에서 구현)"""
    
    @abstractmethod
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """모델 forward 로직 (하위 클래스에서 구현)"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """최적화된 forward 패스"""
        # ✅ AMP 컨텍스트에서 실행
        if self.config.use_amp and self.device.type == "cuda":
            with self.cuda_opt.amp_context():
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    async def optimize_for_inference(self, sample_inputs: List[torch.Tensor]) -> 'OptimizedModelTemplate':
        """
        🚀 추론용 TensorRT 최적화
        
        Args:
            sample_inputs: 대표 입력 샘플들
            
        Returns:
            최적화된 모델
        """
        if self.is_optimized:
            self.logger.info("🔄 이미 최적화된 모델 사용")
            return self
        
        try:
            self.logger.info("🚀 TensorRT 최적화 시작...")
            
            # ✅ 입력 데이터 검증
            validated_inputs = []
            for inp in sample_inputs:
                if isinstance(inp, torch.Tensor):
                    validated_inputs.append(inp.to(self.device))
                else:
                    # numpy 배열 등을 텐서로 변환
                    tensor_inp = torch.tensor(inp, device=self.device, dtype=torch.float32)
                    validated_inputs.append(tensor_inp)
            
            # ✅ TensorRT 최적화 실행
            self.optimized_model = self.cuda_opt.tensorrt_optimize_advanced(
                model=self,
                input_examples=validated_inputs,
                precision=self.config.tensorrt_precision,
                dynamic_shapes=True,
                max_batch_size=self.config.max_batch_size,
                model_name=f"{self.__class__.__name__}_optimized"
            )
            
            self.is_optimized = True
            self.logger.info("✅ TensorRT 최적화 완료")
            
            return self
            
        except Exception as e:
            self.logger.error(f"❌ TensorRT 최적화 실패: {e}")
            self.logger.info("🔄 원본 모델로 폴백")
            return self
    
    async def predict_async(self, inputs: Union[torch.Tensor, np.ndarray, List]) -> ModelPrediction:
        """
        🚀 비동기 단일 예측
        
        Args:
            inputs: 입력 데이터
            
        Returns:
            예측 결과
        """
        # ✅ 입력 데이터 전처리
        processed_input = await self._preprocess_input_async(inputs)
        
        # ✅ 스마트 메모리 할당으로 추론
        with self.memory_mgr.temporary_allocation(
            size=processed_input.numel(),
            prefer_device=self.device.type
        ) as work_tensor:
            
            # 추론 실행
            start_time = asyncio.get_event_loop().time()
            
            if self.is_optimized and self.optimized_model:
                prediction = self.optimized_model(processed_input)
            else:
                prediction = self.forward(processed_input)
            
            end_time = asyncio.get_event_loop().time()
            
            # 성능 통계 업데이트
            self.inference_count += 1
            self.total_inference_time += (end_time - start_time)
            self.batch_sizes.append(processed_input.shape[0] if len(processed_input.shape) > 0 else 1)
            
            # 결과 후처리
            result = await self._postprocess_prediction_async(prediction)
            
            return result
    
    async def predict_batch_async(self, inputs: List[Union[torch.Tensor, np.ndarray]]) -> List[ModelPrediction]:
        """
        🚀 비동기 배치 예측
        
        Args:
            inputs: 입력 데이터 리스트
            
        Returns:
            예측 결과 리스트
        """
        if len(inputs) == 0:
            return []
        
        # ✅ 동적 배치 크기 결정
        optimal_batch_size = self._calculate_optimal_batch_size(inputs)
        
        # ✅ 입력 데이터 배치 분할
        batches = [inputs[i:i+optimal_batch_size] for i in range(0, len(inputs), optimal_batch_size)]
        
        # ✅ 향상된 프로세스 풀을 사용한 병렬 처리
        batch_results = await self.process_pool.async_process_batch(
            batches,
            process_func=self._predict_single_batch,
            use_gpu=self.config.use_gpu,
            max_workers=4,
            gpu_memory_limit=0.8
        )
        
        # 결과 병합
        all_predictions = []
        for batch_result in batch_results:
            all_predictions.extend(batch_result)
        
        return all_predictions
    
    async def _preprocess_input_async(self, inputs: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """비동기 입력 전처리"""
        # ✅ 입력 타입별 처리
        if isinstance(inputs, torch.Tensor):
            tensor_input = inputs.to(self.device)
        elif isinstance(inputs, np.ndarray):
            tensor_input = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        elif isinstance(inputs, (list, tuple)):
            tensor_input = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"지원되지 않는 입력 타입: {type(inputs)}")
        
        # ✅ 입력 차원 조정
        if len(tensor_input.shape) == 1:
            tensor_input = tensor_input.unsqueeze(0)  # 배치 차원 추가
        
        # ✅ 입력 검증
        if self.config.optimize_memory:
            validation_result = self.validator.validate_vector(
                tensor_input.cpu().numpy(),
                check_range=True,
                normalize=False
            )
            
            if not validation_result.is_valid:
                self.logger.warning(f"⚠️ 입력 검증 실패: {validation_result.errors}")
                if validation_result.corrected_vector is not None:
                    tensor_input = torch.tensor(
                        validation_result.corrected_vector,
                        device=self.device,
                        dtype=torch.float32
                    )
        
        return tensor_input
    
    async def _postprocess_prediction_async(self, prediction: torch.Tensor) -> ModelPrediction:
        """비동기 예측 결과 후처리"""
        # CPU로 이동
        prediction_cpu = prediction.cpu()
        
        # 확률 계산 (소프트맥스 적용)
        if prediction_cpu.dim() > 1 and prediction_cpu.shape[-1] > 1:
            probabilities = torch.softmax(prediction_cpu, dim=-1)
        else:
            probabilities = torch.sigmoid(prediction_cpu)
        
        # 예측 결과 생성
        result = ModelPrediction(
            predictions=prediction_cpu.numpy(),
            probabilities=probabilities.numpy(),
            model_name=self.__class__.__name__,
            is_optimized=self.is_optimized,
            device=str(self.device)
        )
        
        return result
    
    def _predict_single_batch(self, batch: List[Union[torch.Tensor, np.ndarray]]) -> List[ModelPrediction]:
        """단일 배치 예측 (병렬 처리용)"""
        predictions = []
        
        for inp in batch:
            # 동기 처리 (병렬 워커 내부)
            with self.memory_mgr.temporary_allocation(
                size=self._estimate_input_size(inp),
                prefer_device=self.device.type
            ) as work_tensor:
                
                # 입력 전처리
                if isinstance(inp, torch.Tensor):
                    processed_inp = inp.to(self.device)
                else:
                    processed_inp = torch.tensor(inp, device=self.device, dtype=torch.float32)
                
                # 추론 실행
                if self.is_optimized and self.optimized_model:
                    prediction = self.optimized_model(processed_inp)
                else:
                    prediction = self.forward(processed_inp)
                
                # 후처리
                result = self._postprocess_prediction_sync(prediction)
                predictions.append(result)
        
        return predictions
    
    def _postprocess_prediction_sync(self, prediction: torch.Tensor) -> ModelPrediction:
        """동기 예측 결과 후처리"""
        prediction_cpu = prediction.cpu()
        
        if prediction_cpu.dim() > 1 and prediction_cpu.shape[-1] > 1:
            probabilities = torch.softmax(prediction_cpu, dim=-1)
        else:
            probabilities = torch.sigmoid(prediction_cpu)
        
        return ModelPrediction(
            predictions=prediction_cpu.numpy(),
            probabilities=probabilities.numpy(),
            model_name=self.__class__.__name__,
            is_optimized=self.is_optimized,
            device=str(self.device)
        )
    
    def _calculate_optimal_batch_size(self, inputs: List) -> int:
        """최적 배치 크기 계산"""
        if not inputs:
            return self.config.batch_size
        
        # ✅ GPU 메모리 상태 확인
        if self.device.type == "cuda":
            gpu_memory_free = self.memory_mgr.get_gpu_memory_available()
            
            # 메모리 상태에 따른 동적 배치 크기 조정
            if gpu_memory_free > 0.8:
                return min(self.config.max_batch_size, len(inputs))
            elif gpu_memory_free > 0.5:
                return min(self.config.batch_size * 2, len(inputs))
            else:
                return min(self.config.batch_size // 2, len(inputs))
        
        return min(self.config.batch_size, len(inputs))
    
    def _estimate_input_size(self, inp: Union[torch.Tensor, np.ndarray, List]) -> int:
        """입력 크기 추정"""
        if isinstance(inp, torch.Tensor):
            return inp.numel() * inp.element_size()
        elif isinstance(inp, np.ndarray):
            return inp.nbytes
        elif isinstance(inp, (list, tuple)):
            return len(inp) * 4  # float32 기준
        else:
            return 1024  # 기본값
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        avg_inference_time = self.total_inference_time / max(self.inference_count, 1)
        avg_batch_size = sum(self.batch_sizes) / max(len(self.batch_sizes), 1)
        
        return {
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "avg_inference_time": avg_inference_time,
            "avg_batch_size": avg_batch_size,
            "is_optimized": self.is_optimized,
            "device": str(self.device),
            "memory_stats": self.memory_mgr.get_memory_status(),
            "cuda_stats": self.cuda_opt.get_cache_stats() if self.device.type == "cuda" else {}
        }
    
    def save_model(self, path: str, save_optimized: bool = True):
        """모델 저장"""
        try:
            # ✅ 기본 모델 저장
            model_state = {
                "model_state_dict": self.state_dict(),
                "config": self.config.__dict__,
                "model_class": self.__class__.__name__,
                "is_optimized": self.is_optimized,
                "performance_stats": self.get_performance_stats()
            }
            
            torch.save(model_state, path)
            
            # ✅ 최적화된 모델 저장 (선택적)
            if save_optimized and self.is_optimized and self.optimized_model:
                optimized_path = path.replace('.pt', '_optimized.pt')
                torch.save(self.optimized_model, optimized_path)
                self.logger.info(f"✅ 최적화된 모델 저장: {optimized_path}")
            
            self.logger.info(f"✅ 모델 저장 완료: {path}")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 저장 실패: {e}")
            raise
    
    def load_model(self, path: str, load_optimized: bool = True):
        """모델 로드"""
        try:
            # ✅ 기본 모델 로드
            model_state = torch.load(path, map_location=self.device)
            
            self.load_state_dict(model_state["model_state_dict"])
            self.is_optimized = model_state.get("is_optimized", False)
            
            # ✅ 최적화된 모델 로드 (선택적)
            if load_optimized and self.is_optimized:
                optimized_path = path.replace('.pt', '_optimized.pt')
                try:
                    self.optimized_model = torch.load(optimized_path, map_location=self.device)
                    self.logger.info(f"✅ 최적화된 모델 로드: {optimized_path}")
                except FileNotFoundError:
                    self.logger.warning(f"⚠️ 최적화된 모델 파일 없음: {optimized_path}")
                    self.is_optimized = False
            
            self.logger.info(f"✅ 모델 로드 완료: {path}")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def cleanup(self):
        """리소스 정리"""
        if self.optimized_model:
            del self.optimized_model
        
        self.model_cache.clear()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"✅ {self.__class__.__name__} 정리 완료")


# =====================================
# 실제 모델 구현 예시
# =====================================

class EnhancedLotteryPredictor(OptimizedModelTemplate):
    """
    향상된 로또 예측 모델 - 템플릿 활용 예시
    기존 모델들을 src/utils 기반으로 완전 재구성
    """
    
    def __init__(self, input_dim: int = 100, hidden_dims: List[int] = [256, 128, 64], 
                 output_dim: int = 45, config: Optional[ModelConfig] = None):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # 모델 구조 구축
        self.model = self._build_model()
        self.to(self.device)
        
        self.logger.info(f"✅ LotteryPredictor 구축 완료: {input_dim}->{hidden_dims}->{output_dim}")
    
    def _build_model(self) -> nn.Module:
        """로또 예측 모델 구조 정의"""
        layers = []
        
        # 입력 레이어
        layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # 히든 레이어들
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # 출력 레이어
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """모델 forward 패스"""
        return self.model(x)
    
    async def predict_lottery_numbers(self, features: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """로또 번호 예측"""
        # ✅ 비동기 예측 실행
        prediction = await self.predict_async(features)
        
        # ✅ 로또 번호 형태로 변환
        probabilities = prediction.probabilities
        if len(probabilities.shape) > 1:
            probabilities = probabilities[0]  # 첫 번째 배치
        
        # 상위 6개 번호 선택
        top_6_indices = np.argsort(probabilities)[-6:]
        lottery_numbers = [int(idx + 1) for idx in top_6_indices]  # 1-45 범위로 변환
        lottery_numbers.sort()
        
        # 신뢰도 계산
        confidence = np.mean(probabilities[top_6_indices])
        
        return {
            "numbers": lottery_numbers,
            "confidence": float(confidence),
            "all_probabilities": probabilities.tolist(),
            "model_performance": self.get_performance_stats()
        }


# =====================================
# 모델 팩토리
# =====================================

class OptimizedModelFactory:
    """최적화된 모델 생성 팩토리"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config_validator = get_config_validator()
    
    async def create_model(self, model_type: str, config_path: str, **kwargs) -> OptimizedModelTemplate:
        """모델 생성"""
        # ✅ 설정 검증
        if not self.config_validator.validate_config_file(config_path):
            raise ValueError(f"모델 설정 파일 검증 실패: {config_path}")
        
        # 모델 타입별 생성
        if model_type == "lottery_predictor":
            return EnhancedLotteryPredictor(**kwargs)
        elif model_type == "pattern_analyzer":
            # 다른 모델 타입들...
            pass
        else:
            raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
    
    async def optimize_model(self, model: OptimizedModelTemplate, 
                            sample_inputs: List[torch.Tensor]) -> OptimizedModelTemplate:
        """모델 최적화"""
        return await model.optimize_for_inference(sample_inputs)


# =====================================
# 사용 예시
# =====================================

async def main():
    """템플릿 사용 예시"""
    
    # ✅ 1. 시스템 초기화
    initialize_all_systems()
    
    # ✅ 2. 모델 생성
    config = ModelConfig(
        use_gpu=True,
        use_tensorrt=True,
        batch_size=64,
        tensorrt_precision="fp16"
    )
    
    model = EnhancedLotteryPredictor(
        input_dim=100,
        hidden_dims=[256, 128, 64],
        output_dim=45,
        config=config
    )
    
    # ✅ 3. 모델 최적화
    sample_input = torch.randn(1, 100)
    optimized_model = await model.optimize_for_inference([sample_input])
    
    # ✅ 4. 예측 실행
    try:
        test_features = np.random.randn(100)
        result = await optimized_model.predict_lottery_numbers(test_features)
        
        print("✅ 예측 완료!")
        print(f"예측 번호: {result['numbers']}")
        print(f"신뢰도: {result['confidence']:.3f}")
        
        # 성능 통계
        stats = optimized_model.get_performance_stats()
        print(f"평균 추론 시간: {stats['avg_inference_time']:.4f}초")
        print(f"TensorRT 최적화: {'✅' if stats['is_optimized'] else '❌'}")
        
    finally:
        # ✅ 5. 리소스 정리
        optimized_model.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 