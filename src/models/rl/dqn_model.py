"""
DQN 강화학습 모델

이 모듈은 Deep Q-Network 알고리즘 기반 로또 번호 예측 모델을 구현합니다.
경험 재생과 타겟 네트워크를 활용하여 안정적인 학습을 지원합니다.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Any, Optional, Union, Tuple, Deque
from pathlib import Path
import time
from collections import deque, namedtuple

from .rl_base_model import RLBaseModel
from ...utils.error_handler import get_logger

logger = get_logger(__name__)

# 경험 재생을 위한 namedtuple 정의
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """
    경험 재생 버퍼

    모델 학습 중 얻은 경험을 저장하고, 미니배치로 샘플링합니다.
    """

    def __init__(self, capacity: int = 10000):
        """
        버퍼 초기화

        Args:
            capacity: 버퍼 용량
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        경험 추가

        Args:
            state: 현재 상태
            action: 수행한 액션
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        경험 샘플링

        Args:
            batch_size: 샘플 크기

        Returns:
            상태, 액션, 보상, 다음 상태, 종료 여부 배치
        """
        experiences = random.sample(self.buffer, k=batch_size)

        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences]))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """버퍼 크기 반환"""
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q-네트워크

    상태를 입력으로 받아 각 행동의 Q-값을 출력합니다.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Q-네트워크 초기화

        Args:
            state_size: 상태 공간 크기
            action_size: 행동 공간 크기
            hidden_size: 은닉층 크기
        """
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 상태 텐서

        Returns:
            Q-값 텐서
        """
        # 첫 번째 레이어
        x = F.relu(self.bn1(self.fc1(x)))

        # 두 번째 레이어
        x = F.relu(self.bn2(self.fc2(x)))

        # 출력 레이어
        return self.fc3(x)


class DQNModel(RLBaseModel):
    """
    DQN 기반 로또 번호 예측 모델

    Deep Q-Network 알고리즘을 사용하여 로또 번호 선택 전략을 학습합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        DQN 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config, model_name="DQNModel")

        # 모델 설정
        dqn_config = self.config.get("dqn", {})

        # 모델 하이퍼파라미터
        self.state_size = dqn_config.get("state_size", 70)
        self.action_size = dqn_config.get("action_size", 45)  # 로또 번호 범위 (1-45)
        self.hidden_size = dqn_config.get("hidden_size", 128)
        self.learning_rate = dqn_config.get("learning_rate", 0.001)
        self.gamma = dqn_config.get("gamma", 0.99)  # 할인 계수
        self.tau = dqn_config.get("tau", 0.001)  # 타겟 네트워크 소프트 업데이트 계수
        self.epsilon_start = dqn_config.get("epsilon_start", 1.0)
        self.epsilon_end = dqn_config.get("epsilon_end", 0.01)
        self.epsilon_decay = dqn_config.get("epsilon_decay", 0.995)
        self.batch_size = dqn_config.get("batch_size", 64)
        self.buffer_size = dqn_config.get("buffer_size", 10000)
        self.update_freq = dqn_config.get(
            "update_freq", 4
        )  # 타겟 네트워크 업데이트 주기

        # 현재 학습 상태
        self.epsilon = self.epsilon_start
        self.steps = 0

        # 모델 구성
        self._build_model()

        # 경험 재생 버퍼
        self.memory = ReplayBuffer(self.buffer_size)

        logger.info(
            f"DQN 모델 초기화 완료: 상태 크기={self.state_size}, "
            f"행동 크기={self.action_size}, 은닉층 크기={self.hidden_size}, "
            f"학습률={self.learning_rate}, 감마={self.gamma}"
        )

    def _build_model(self):
        """모델 구성"""
        # 정책 네트워크 (현재 사용하는 네트워크)
        self.policy_net = QNetwork(
            self.state_size, self.action_size, self.hidden_size
        ).to(self.device)

        # 타겟 네트워크 (학습 안정성을 위한 네트워크)
        self.target_net = QNetwork(
            self.state_size, self.action_size, self.hidden_size
        ).to(self.device)

        # 타겟 네트워크를 정책 네트워크와 동일하게 초기화
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 평가 모드로 설정

        # 최적화기 설정
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # 손실 함수
        self.criterion = nn.MSELoss()

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        행동 선택

        Args:
            state: 상태
            eval_mode: 평가 모드 여부 (True일 경우 탐험 없음)

        Returns:
            선택된 행동
        """
        # 랜덤 탐험 (epsilon-greedy)
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # 모델 기반 행동 선택
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()

        # 최대 Q값을 가진 행동 선택
        return q_values.max(1)[1].item()

    def learn(self):
        """
        모델 학습 단계 수행
        """
        # 경험 리플레이 버퍼에 충분한 샘플이 없으면 학습 스킵
        if len(self.memory) < self.batch_size:
            return

        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # AMP 사용 여부에 따라 학습 방식 결정
        if self.use_amp:
            with torch.cuda.amp.autocast():
                # 현재 Q 값 계산
                q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

                # 타겟 Q 값 계산
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0]
                    target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
                    target_q_values = target_q_values.unsqueeze(1)

                # 손실 계산
                loss = self.criterion(q_values, target_q_values)

            # 최적화 단계
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 현재 Q 값 계산
            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

            # 타겟 Q 값 계산
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
                target_q_values = target_q_values.unsqueeze(1)

            # 손실 계산
            loss = self.criterion(q_values, target_q_values)

            # 최적화 단계
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 타겟 네트워크 소프트 업데이트 (일정 주기마다)
        if self.steps % self.update_freq == 0:
            self._soft_update()

        # 탐험률 감소
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1

    def _soft_update(self):
        """
        타겟 네트워크 소프트 업데이트

        policy_net의 가중치를 tau 비율로 target_net에 반영
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        DQN 모델 훈련

        Args:
            X: 초기 상태 벡터
            y: 보상 벡터 (사용되지 않을 수 있음)
            **kwargs: 추가 훈련 파라미터

        Returns:
            훈련 결과 및 메타데이터
        """
        # 훈련 하이퍼파라미터 설정
        episodes = kwargs.get("episodes", 1000)
        max_steps = kwargs.get("max_steps", 100)
        early_stopping_patience = kwargs.get("early_stopping_patience", 100)
        log_interval = kwargs.get("log_interval", 10)

        logger.info(
            f"DQN 모델 훈련 시작: 에피소드={episodes}, 최대 스텝={max_steps}, "
            f"조기 종료 인내={early_stopping_patience}"
        )

        # 훈련 메트릭 초기화
        episode_rewards = []
        best_avg_reward = -float("inf")
        no_improvement_count = 0

        # 학습 시작 시간
        start_time = time.time()

        # 에피소드 반복
        for episode in range(1, episodes + 1):
            # 환경 초기화 (초기 상태 사용)
            if X.shape[0] > 0:
                state_idx = np.random.randint(0, X.shape[0])
                state = X[state_idx].copy()
            else:
                # 임의 상태 생성
                state = np.random.randn(self.state_size)

            # 에피소드 리셋
            episode_reward = 0
            done = False
            step = 0

            # 에피소드 단계 반복
            while not done and step < max_steps:
                # 행동 선택
                action = self.select_action(state)

                # 행동 실행 및 다음 상태, 보상 계산 (간단한 환경 시뮬레이션)
                next_state = state.copy()
                # 상태 변화 시뮬레이션 (실제 환경에서는 환경의 응답을 사용)
                next_state += np.random.normal(0, 0.1, size=next_state.shape)

                # 보상 계산 (간단한 보상 모델)
                reward = 1.0 if action % 7 == 0 else -0.1  # 예시 보상 함수
                done = step >= max_steps - 1

                # 경험 저장
                self.memory.add(state, action, reward, next_state, done)

                # 학습 단계 수행
                self.learn()

                # 상태 업데이트
                state = next_state
                episode_reward += reward
                step += 1

            # 에피소드 보상 기록
            episode_rewards.append(episode_reward)
            self.train_episode_rewards.append(episode_reward)

            # 로깅
            if episode % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                logger.info(
                    f"에피소드 {episode}/{episodes}, 평균 보상: {avg_reward:.2f}, "
                    f"탐험률: {self.epsilon:.4f}"
                )

                # 모델 개선 체크
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # 조기 종료 체크
                if no_improvement_count >= early_stopping_patience:
                    logger.info(
                        f"조기 종료: {early_stopping_patience} 에피소드 동안 개선 없음"
                    )
                    break

        # 훈련 시간 계산
        train_time = time.time() - start_time

        # 메타데이터 업데이트
        self.metadata.update(
            {
                "episodes": episode,
                "max_steps": max_steps,
                "best_avg_reward": best_avg_reward,
                "train_time": train_time,
                "final_epsilon": self.epsilon,
            }
        )

        # 훈련 완료 표시
        self.is_trained = True
        logger.info(
            f"DQN 모델 훈련 완료: 최종 에피소드={episode}, 최고 평균 보상={best_avg_reward:.2f}, "
            f"훈련 시간={train_time:.2f}초"
        )

        return {
            "episodes": episode,
            "best_avg_reward": best_avg_reward,
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
        }

    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        DQN 모델 예측 구현

        Args:
            X: 상태 벡터
            **kwargs: 추가 예측 파라미터

        Returns:
            예측된 행동 또는 Q값
        """
        return_q_values = kwargs.get("return_q_values", False)
        batch_size = kwargs.get("batch_size", 128)

        logger.info(f"DQN 예측 수행: 입력 형태={X.shape}")

        # 결과 초기화
        if return_q_values:
            # Q값 전체 반환
            results = np.zeros((X.shape[0], self.action_size))
        else:
            # 최적 행동만 반환
            results = np.zeros(X.shape[0], dtype=np.int64)

        # 배치 처리
        num_batches = (X.shape[0] + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X.shape[0])
            batch_X = X[start_idx:end_idx]

            # 상태 텐서 변환
            states = torch.FloatTensor(batch_X).to(self.device)

            # 예측 수행
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(states).cpu().numpy()
            self.policy_net.train()

            # 결과 저장
            if return_q_values:
                results[start_idx:end_idx] = q_values
            else:
                results[start_idx:end_idx] = np.argmax(q_values, axis=1)

        return results

    def _evaluate_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        DQN 모델 평가 구현

        Args:
            X: 상태 벡터
            y: 보상 벡터 (사용되지 않을 수 있음)
            **kwargs: 추가 평가 파라미터

        Returns:
            평가 결과
        """
        # 평가 하이퍼파라미터 설정
        episodes = kwargs.get("episodes", 10)
        max_steps = kwargs.get("max_steps", 100)
        log_interval = kwargs.get("log_interval", 5)

        logger.info(f"DQN 모델 평가 시작: 에피소드={episodes}, 최대 스텝={max_steps}")

        # 평가 메트릭 초기화
        episode_rewards = []

        # 에피소드 반복
        for episode in range(1, episodes + 1):
            # 환경 초기화 (초기 상태 사용)
            if X.shape[0] > 0:
                state_idx = np.random.randint(0, X.shape[0])
                state = X[state_idx].copy()
            else:
                # 임의 상태 생성
                state = np.random.randn(self.state_size)

            # 에피소드 리셋
            episode_reward = 0
            done = False
            step = 0

            # 에피소드 단계 반복
            while not done and step < max_steps:
                # 행동 선택 (평가 모드 - 탐험 없음)
                action = self.select_action(state, eval_mode=True)

                # 행동 실행 및 다음 상태, 보상 계산 (간단한 환경 시뮬레이션)
                next_state = state.copy()
                # 상태 변화 시뮬레이션 (실제 환경에서는 환경의 응답을 사용)
                next_state += np.random.normal(0, 0.05, size=next_state.shape)

                # 보상 계산 (간단한 보상 모델)
                reward = 1.0 if action % 7 == 0 else -0.1  # 예시 보상 함수
                done = step >= max_steps - 1

                # 상태 업데이트
                state = next_state
                episode_reward += reward
                step += 1

            # 에피소드 보상 기록
            episode_rewards.append(episode_reward)
            self.eval_episode_rewards.append(episode_reward)

            # 로깅
            if episode % log_interval == 0:
                logger.info(
                    f"평가 에피소드 {episode}/{episodes}, 보상: {episode_reward:.2f}"
                )

        # 평가 결과 계산
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        max_reward = np.max(episode_rewards)
        min_reward = np.min(episode_rewards)

        logger.info(
            f"DQN 모델 평가 완료: 평균 보상={mean_reward:.2f}, "
            f"표준편차={std_reward:.2f}, 최대={max_reward:.2f}, 최소={min_reward:.2f}"
        )

        return {
            "average_reward": mean_reward,
            "std_reward": std_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "episodes": episodes,
            "model_type": self.model_name,
        }

    def _save_model(self, path: str) -> None:
        """
        DQN 모델 저장 구현

        Args:
            path: 저장 경로
        """
        # 모델 상태 저장
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def _load_model(self, path: str) -> None:
        """
        DQN 모델 로드 구현

        Args:
            path: 모델 파일 경로
        """
        # 모델 상태 로드
        checkpoint = torch.load(path, map_location=self.device)

        # 모델 재구성
        self._build_model()

        # 상태 딕셔너리 로드
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.steps = checkpoint.get("steps", 0)

    def _get_additional_metadata(self) -> Dict[str, Any]:
        """
        추가 메타데이터 획득

        Returns:
            추가 메타데이터 딕셔너리
        """
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "steps": self.steps,
        }

    def _load_additional_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        추가 메타데이터 로드

        Args:
            metadata: 메타데이터 딕셔너리
        """
        self.state_size = metadata.get("state_size", self.state_size)
        self.action_size = metadata.get("action_size", self.action_size)
        self.hidden_size = metadata.get("hidden_size", self.hidden_size)
        self.learning_rate = metadata.get("learning_rate", self.learning_rate)
        self.gamma = metadata.get("gamma", self.gamma)
        self.epsilon = metadata.get("epsilon", self.epsilon)
        self.steps = metadata.get("steps", self.steps)


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
