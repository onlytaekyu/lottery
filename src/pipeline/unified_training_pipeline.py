# 1. 표준 라이브러리
import time
from typing import Dict, Any, Optional

# 2. 서드파티
import numpy as np
from tqdm import tqdm

# 3. 프로젝트 내부
from ..utils.unified_logging import get_logger
from ..utils.unified_config import get_config
from ..utils.model_saver import ModelSaver
from ..models.base_model import BaseModel
from ..models import create_model  # src/models/__init__.py에 팩토리 함수가 있다고 가정

logger = get_logger(__name__)


class UnifiedTrainingPipeline:
    """
    통합 훈련 파이프라인

    - 모든 모델 타입 지원 (BaseModel 기반)
    - 자동 체크포인트 및 재시작
    - 실시간 성능 모니터링 (tqdm, logging)
    - 분산 훈련 지원 (향후 확장용 프레임워크)
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = get_config("training") if config is None else config
        self.model_saver = ModelSaver(
            base_path=self.config.get("checkpoint_dir", "savedModels/checkpoints")
        )
        self.use_gpu = self.config.get("use_gpu", False)
        logger.info(f"통합 훈련 파이프라인 초기화. GPU 사용: {self.use_gpu}")

    def run(self, model_name: str, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> BaseModel:
        """
        훈련 파이프라인 실행

        Args:
            model_name (str): 훈련할 모델의 이름 (e.g., 'LightGBMModel')
            X (np.ndarray): 훈련 데이터
            y (np.ndarray): 훈련 라벨
            X_val (Optional[np.ndarray]): 검증 데이터
            y_val (Optional[np.ndarray]): 검증 라벨

        Returns:
            BaseModel: 훈련된 모델 인스턴스
        """
        # 1. 모델 생성 또는 체크포인트에서 로드
        model = self._get_or_create_model(model_name)

        # 2. 훈련 실행
        logger.info(f"'{model_name}' 모델 훈련 시작...")
        start_time = time.time()

        try:
            # 훈련 인자 준비
            fit_kwargs = {}
            if X_val is not None and y_val is not None:
                fit_kwargs['eval_set'] = [(X_val, y_val)]
            
            # epochs/iterations 설정 (tqdm을 위함)
            n_epochs = self.config.get("epochs", 1)
            
            for epoch in tqdm(range(n_epochs), desc=f"Training {model_name}"):
                metrics = model.fit(X, y, **fit_kwargs)
                
                # 실시간 성능 모니터링 (로그)
                logger.info(f"Epoch {epoch + 1}/{n_epochs} 완료. Metrics: {metrics}")

                # 3. 자동 체크포인트
                self._save_checkpoint(model, model_name, epoch, metrics)

        except Exception as e:
            logger.error(f"'{model_name}' 훈련 중 오류 발생: {e}", exc_info=True)
            raise

        total_time = time.time() - start_time
        logger.info(f"'{model_name}' 모델 훈련 완료. 총 소요 시간: {total_time:.2f}초")

        # 최종 모델 저장
        self.model_saver.save_model(model, model_name, "final")
        logger.info(f"최종 모델이 'savedModels/{model_name}/final'에 저장되었습니다.")
        
        return model

    def _get_or_create_model(self, model_name: str) -> BaseModel:
        """체크포인트에서 모델을 로드하거나 새로 생성합니다."""
        latest_checkpoint = self.model_saver.get_latest_checkpoint(model_name)
        if latest_checkpoint:
            logger.info(f"'{model_name}'의 체크포인트 '{latest_checkpoint}'에서 훈련을 재개합니다.")
            return self.model_saver.load_model(model_name, latest_checkpoint)
        
        logger.info(f"'{model_name}'에 대한 체크포인트가 없습니다. 새로운 모델을 생성합니다.")
        model_config = get_config("main").get(model_name, {})
        # `create_model` 팩토리 함수를 사용하여 모델 인스턴스화
        return create_model(model_name, config=model_config)

    def _save_checkpoint(self, model: BaseModel, model_name: str, epoch: int, metrics: Dict[str, Any]):
        """훈련 중 주기적으로 체크포인트를 저장합니다."""
        # 여기서는 매 epoch마다 저장. 실제로는 특정 조건(e.g., 검증 성능 향상)에 따라 저장하는 것이 좋음.
        tag = f"epoch_{epoch+1}"
        self.model_saver.save_model(model, model_name, tag, metrics=metrics)
        logger.debug(f"체크포인트 저장 완료: {model_name}/{tag}")

    def _setup_distributed_training(self):
        """(향후 확장) 분산 훈련 환경을 설정합니다."""
        # DDP (DistributedDataParallel) 또는 Horovod 설정 로직이 여기에 위치합니다.
        # 예:
        # if 'WORLD_SIZE' in os.environ:
        #     dist.init_process_group(backend='nccl', init_method='env://')
        logger.info("분산 훈련은 현재 지원되지 않지만, 향후 확장을 위해 프레임워크가 준비되었습니다.")

if __name__ == '__main__':
    # 간단한 테스트 및 사용 예시
    logger.info("통합 훈련 파이프라인 테스트 시작")
    
    # 가상 데이터 생성
    X_train_sample = np.random.rand(1000, 20)
    y_train_sample = (np.sum(X_train_sample, axis=1) > 10).astype(int)
    X_val_sample = np.random.rand(200, 20)
    y_val_sample = (np.sum(X_val_sample, axis=1) > 10).astype(int)

    # 파이프라인 인스턴스 생성
    # 실제 사용 시에는 config.yaml 파일에 설정을 정의
    test_config = {
        "use_gpu": False,
        "epochs": 5,
        "checkpoint_dir": "savedModels/test_checkpoints"
    }
    training_pipeline = UnifiedTrainingPipeline(config=test_config)
    
    # LightGBM 모델 훈련 (팩토리 함수가 'LightGBMModel'을 인식해야 함)
    try:
        trained_model = training_pipeline.run(
            model_name="LightGBMModel",
            X=X_train_sample,
            y=y_train_sample,
            X_val=X_val_sample,
            y_val=y_val_sample
        )
        logger.info("테스트 모델 훈련 성공")
        
        # 예측 테스트
        predictions = trained_model.predict(X_val_sample)
        logger.info(f"테스트 예측 결과 (첫 5개): {predictions[:5]}")
        
    except Exception as e:
        logger.error(f"파이프라인 테스트 중 오류 발생: {e}")

    logger.info("통합 훈련 파이프라인 테스트 종료") 