"""
PPO(Proximal Policy Optimization) 강화학습 모델

이 모듈은 PPO 알고리즘 기반 로또 번호 예측 모델을 구현합니다.
Actor-Critic 구조를 사용하여 안정적인 학습을 지원합니다.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time

from ..base_model import ModelWithAMP
from ...utils.error_handler_refactored import get_logger

logger = get_logger(__name__)


class ActorNetwork(nn.Module):
    """
    Actor 네트워크 (정책 네트워크)

    로또 번호 선택 확률 분포를 출력하는 네트워크입니다.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Actor 네트워크 초기화

        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원 (로또 번호 범위)
            hidden_dim: 은닉층 차원
        """
        super().__init__()

        # 네트워크 레이어
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 활성화 함수
        self.relu = nn.ReLU()

        # 출력 활성화 함수 (확률 분포)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            state: 상태 텐서

        Returns:
            행동 확률 분포
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # 출력을 확률 분포로 변환
        action_probs = self.softmax(x)

        return action_probs


class CriticNetwork(nn.Module):
    """
    Critic 네트워크 (가치 네트워크)

    상태의 가치를 추정하는 네트워크입니다.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        Critic 네트워크 초기화

        Args:
            state_dim: 상태 차원
            hidden_dim: 은닉층 차원
        """
        super().__init__()

        # 네트워크 레이어
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # 활성화 함수
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            state: 상태 텐서

        Returns:
            상태 가치 추정값
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)

        return value


class PPOModel(ModelWithAMP):
    """
    PPO 기반 로또 번호 예측 모델

    Proximal Policy Optimization 알고리즘을 사용하여 로또 번호 선택 전략을 학습합니다.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        PPO 모델 초기화

        Args:
            config: 모델 설정
        """
        super().__init__(config)

        # 모델 설정
        self.config = config or {}
        ppo_config = self.config.get("ppo", {})

        # 모델 하이퍼파라미터
        self.state_dim = ppo_config.get("state_dim", 70)
        self.action_dim = ppo_config.get("action_dim", 45)  # 로또 번호 범위 (1-45)
        self.hidden_dim = ppo_config.get("hidden_dim", 128)

        # PPO 하이퍼파라미터
        self.lr_actor = ppo_config.get("lr_actor", 0.0003)
        self.lr_critic = ppo_config.get("lr_critic", 0.001)
        self.gamma = ppo_config.get("gamma", 0.99)  # 할인 계수
        self.eps_clip = ppo_config.get("eps_clip", 0.2)  # PPO 클리핑 파라미터
        self.k_epochs = ppo_config.get("k_epochs", 4)  # PPO 업데이트 반복 횟수

        # 훈련 설정
        self.update_timestep = ppo_config.get("update_timestep", 1000)  # 업데이트 주기
        self.batch_size = ppo_config.get("batch_size", 64)
        self.max_episodes = ppo_config.get("max_episodes", 1000)
        self.max_timesteps = ppo_config.get("max_timesteps", 200)

        # 모델 이름
        self.model_name = "PPOModel"

        # 네트워크 및 최적화기 초기화
        self._init_networks()

        # 전체 타임스텝 카운터
        self.time_step = 0

        # 버퍼 초기화
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

        logger.info(
            f"PPO 모델 초기화 완료: 상태 차원={self.state_dim}, "
            f"행동 차원={self.action_dim}, 은닉층 차원={self.hidden_dim}"
        )

    def _init_networks(self):
        """
        네트워크 및 최적화기 초기화
        """
        # Actor 네트워크
        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(
            self.device
        )

        # Critic 네트워크
        self.critic = CriticNetwork(self.state_dim, self.hidden_dim).to(self.device)

        # 최적화기
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Old Actor 네트워크 (안정적인 업데이트를 위한 복사본)
        self.old_actor = ActorNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(self.device)
        self.old_actor.load_state_dict(self.actor.state_dict())

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        행동 선택

        Args:
            state: 상태

        Returns:
            선택된 행동 및 로그 확률의 튜플
        """
        with torch.no_grad():
            # 상태를 텐서로 변환
            state_tensor = torch.FloatTensor(state).to(self.device)

            # 행동 확률 분포 계산
            action_probs = self.old_actor(state_tensor)

            # 확률 분포에서 샘플링
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()

            # 행동의 로그 확률 계산
            logprob = action_distribution.log_prob(action)

            return action.item(), logprob.item()

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        상태와 행동 평가

        Args:
            state: 상태 텐서
            action: 행동 텐서

        Returns:
            행동 로그 확률, 상태 가치, 엔트로피의 튜플
        """
        # 행동 확률 분포 계산
        action_probs = self.actor(state)

        # 행동 분포 생성
        action_distribution = torch.distributions.Categorical(action_probs)

        # 행동의 로그 확률 계산
        logprobs = action_distribution.log_prob(action)

        # 분포의 엔트로피 계산
        entropy = action_distribution.entropy()

        # 상태 가치 계산
        state_value = self.critic(state)

        return logprobs, state_value, entropy

    def update(self):
        """
        PPO 업데이트 수행
        """
        # 리워드 정규화
        rewards = []
        discounted_reward = 0

        # 역순으로 할인된 리워드 계산
        for reward, is_terminal in zip(
            reversed(self.rewards), reversed(self.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 리워드 정규화
        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 메모리에서 데이터 변환
        old_states = torch.FloatTensor(np.array(self.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.logprobs)).to(self.device)

        # k_epochs 동안 최적화
        for _ in range(self.k_epochs):
            # 현재 정책에서 행동 평가
            logprobs, state_values, entropy = self.evaluate(old_states, old_actions)

            # 가치 손실 계산을 위해 차원 조정
            state_values = state_values.squeeze()

            # 비율 계산 (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 이점 계산
            advantages = rewards - state_values.detach()

            # PPO 손실 계산
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # 최종 손실
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * F.mse_loss(state_values, rewards)
            entropy_loss = -0.01 * entropy.mean()  # 엔트로피 보너스

            # 역전파 및 최적화
            if self.use_amp:
                # AMP를 사용한 최적화
                # Actor 네트워크 업데이트
                self.optimizer_actor.zero_grad()
                with torch.cuda.amp.autocast():
                    actor_total_loss = actor_loss + entropy_loss

                self.scaler.scale(actor_total_loss).backward()
                self.scaler.step(self.optimizer_actor)

                # Critic 네트워크 업데이트
                self.optimizer_critic.zero_grad()
                with torch.cuda.amp.autocast():
                    critic_total_loss = critic_loss

                self.scaler.scale(critic_total_loss).backward()
                self.scaler.step(self.optimizer_critic)

                # 스케일러 업데이트
                self.scaler.update()
            else:
                # 일반 최적화
                # Actor 네트워크 업데이트
                self.optimizer_actor.zero_grad()
                (actor_loss + entropy_loss).backward()
                self.optimizer_actor.step()

                # Critic 네트워크 업데이트
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

        # 이전 정책 업데이트
        self.old_actor.load_state_dict(self.actor.state_dict())

        # 메모리 비우기
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 훈련

        Args:
            X: 상태 벡터
            y: 보상 값
            **kwargs: 추가 매개변수

        Returns:
            훈련 결과 및 메타데이터
        """
        # 로깅
        logger.info(f"PPO 모델 훈련 시작: X 형태={X.shape}, y 형태={y.shape}")

        # 훈련 매개변수
        episodes = kwargs.get("episodes", self.max_episodes)
        max_timesteps = kwargs.get("max_timesteps", self.max_timesteps)

        # 훈련 지표
        episode_rewards = []
        episode_lengths = []
        avg_rewards = []

        # 훈련 시작
        start_time = time.time()

        for episode in range(episodes):
            episode_reward = 0
            episode_length = 0

            # 초기 상태 선택
            idx = random.randint(0, len(X) - 1)
            state = X[idx]

            for t in range(max_timesteps):
                # 타임스텝 증가
                self.time_step += 1
                episode_length += 1

                # 행동 선택
                action, logprob = self.select_action(state)

                # 다음 상태 및 보상 (시뮬레이션)
                next_idx = (idx + 1) % len(X)
                next_state = X[next_idx]
                reward = y[next_idx]  # 보상은 타겟 값으로 사용
                done = (
                    t == max_timesteps - 1 or random.random() < 0.05
                )  # 일정 확률로 조기 종료

                # 메모리에 저장
                self.states.append(state)
                self.actions.append(action)
                self.logprobs.append(logprob)
                self.rewards.append(reward)
                self.is_terminals.append(done)

                # 상태 업데이트
                state = next_state
                idx = next_idx
                episode_reward += reward

                # 업데이트 수행
                if self.time_step % self.update_timestep == 0:
                    self.update()

                if done:
                    break

            # 에피소드 결과 기록
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # 이동 평균 계산
            if len(episode_rewards) > 100:
                avg_reward = np.mean(episode_rewards[-100:])
            else:
                avg_reward = np.mean(episode_rewards)

            avg_rewards.append(avg_reward)

            # 로깅
            if (episode + 1) % 10 == 0:
                logger.info(
                    f"에피소드 {episode+1}/{episodes}: 보상={episode_reward:.2f}, "
                    f"길이={episode_length}, 100에피소드 평균={avg_reward:.2f}"
                )

        # 남은 데이터로 최종 업데이트
        if len(self.states) > 0:
            self.update()

        # 훈련 시간 계산
        train_time = time.time() - start_time

        # 모델 훈련 완료 표시
        self.is_trained = True

        # 메타데이터 업데이트
        self.metadata.update(
            {
                "episodes": episodes,
                "max_timesteps": max_timesteps,
                "update_timestep": self.update_timestep,
                "final_avg_reward": avg_rewards[-1] if avg_rewards else 0,
                "train_time": train_time,
            }
        )

        logger.info(
            f"PPO 모델 훈련 완료: 최종 평균 보상={avg_rewards[-1] if avg_rewards else 0:.2f}, "
            f"소요 시간={train_time:.2f}초"
        )

        return {
            "average_reward": avg_rewards[-1] if avg_rewards else 0,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "train_time": train_time,
            "is_trained": self.is_trained,
            "model_type": self.model_name,
        }

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        모델 예측 수행

        Args:
            X: 상태 벡터
            **kwargs: 추가 매개변수

        Returns:
            예측 행동 가치
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 평가 모드로 설정
        self.actor.eval()
        self.critic.eval()

        # 샘플 수
        num_samples = kwargs.get("num_samples", 10)
        use_greedy = kwargs.get("use_greedy", False)

        predictions = []

        for state in X:
            with torch.no_grad():
                # 상태를 텐서로 변환
                state_tensor = torch.FloatTensor(state).to(self.device)

                # 행동 확률 분포 계산
                action_probs = self.actor(state_tensor).cpu().numpy()

                if use_greedy:
                    # 탐욕적 선택 (가장 높은 확률의 행동)
                    best_action = np.argmax(action_probs)
                    predictions.append(action_probs)
                else:
                    # 확률적 선택 (여러 번 샘플링하여 평균)
                    sample_actions = []
                    for _ in range(num_samples):
                        action = np.random.choice(len(action_probs), p=action_probs)
                        sample_actions.append(action)

                    # 히스토그램 계산
                    action_hist = np.zeros(self.action_dim)
                    for action in sample_actions:
                        action_hist[action] += 1

                    # 정규화
                    action_hist = action_hist / num_samples
                    predictions.append(action_hist)

        # 예측값 반환
        return np.array(predictions)

    def evaluate_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        모델 평가

        Args:
            X: 상태 벡터
            y: 보상 값
            **kwargs: 추가 매개변수

        Returns:
            평가 결과
        """
        if not self.is_trained:
            raise ValueError(
                "모델이 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )

        # 평가 매개변수
        episodes = kwargs.get("episodes", 100)
        max_timesteps = kwargs.get("max_timesteps", 200)

        # 평가 지표
        episode_rewards = []
        episode_lengths = []
        action_distribution = np.zeros(self.action_dim)

        # 평가 시작
        self.actor.eval()
        self.critic.eval()

        for episode in range(episodes):
            episode_reward = 0
            episode_length = 0

            # 초기 상태 선택
            idx = random.randint(0, len(X) - 1)
            state = X[idx]

            for t in range(max_timesteps):
                episode_length += 1

                # 행동 선택 (평가 모드에서는 확률적 선택)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    action_probs = self.actor(state_tensor).cpu().numpy()
                    action = np.random.choice(len(action_probs), p=action_probs)

                # 행동 분포 업데이트
                action_distribution[action] += 1

                # 다음 상태 및 보상 (시뮬레이션)
                next_idx = (idx + 1) % len(X)
                next_state = X[next_idx]
                reward = y[next_idx]
                done = t == max_timesteps - 1 or random.random() < 0.05

                # 상태 업데이트
                state = next_state
                idx = next_idx
                episode_reward += reward

                if done:
                    break

            # 에피소드 결과 기록
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # 행동 분포 정규화
        action_distribution = action_distribution / action_distribution.sum()

        # 평가 결과
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        std_reward = np.std(episode_rewards)

        logger.info(
            f"PPO 모델 평가: 평균 보상={avg_reward:.2f}±{std_reward:.2f}, "
            f"평균 에피소드 길이={avg_length:.2f}"
        )

        return {
            "average_reward": avg_reward,
            "std_reward": std_reward,
            "average_length": avg_length,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "action_distribution": action_distribution.tolist(),
            "model_type": self.model_name,
        }

    def save(self, path: str) -> bool:
        """
        모델 저장

        Args:
            path: 저장 경로

        Returns:
            성공 여부
        """
        try:
            # 디렉토리 확인
            self._ensure_directory(path)

            # 저장할 데이터
            save_dict = {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer_actor": self.optimizer_actor.state_dict(),
                "optimizer_critic": self.optimizer_critic.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "hidden_dim": self.hidden_dim,
                    "lr_actor": self.lr_actor,
                    "lr_critic": self.lr_critic,
                    "gamma": self.gamma,
                    "eps_clip": self.eps_clip,
                    "k_epochs": self.k_epochs,
                    "update_timestep": self.update_timestep,
                    "batch_size": self.batch_size,
                    "max_episodes": self.max_episodes,
                    "max_timesteps": self.max_timesteps,
                },
                "runtime": {
                    "time_step": self.time_step,
                },
                "metadata": self.metadata,
                "is_trained": self.is_trained,
            }

            # 모델 저장
            torch.save(save_dict, path)

            logger.info(f"PPO 모델 저장 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"PPO 모델 저장 중 오류: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        모델 로드

        Args:
            path: 모델 경로

        Returns:
            성공 여부
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(path):
                raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {path}")

            # 모델 로드
            checkpoint = torch.load(path, map_location=self.device)

            # 설정 업데이트
            config = checkpoint.get("config", {})
            self.state_dim = config.get("state_dim", self.state_dim)
            self.action_dim = config.get("action_dim", self.action_dim)
            self.hidden_dim = config.get("hidden_dim", self.hidden_dim)
            self.lr_actor = config.get("lr_actor", self.lr_actor)
            self.lr_critic = config.get("lr_critic", self.lr_critic)
            self.gamma = config.get("gamma", self.gamma)
            self.eps_clip = config.get("eps_clip", self.eps_clip)
            self.k_epochs = config.get("k_epochs", self.k_epochs)
            self.update_timestep = config.get("update_timestep", self.update_timestep)
            self.batch_size = config.get("batch_size", self.batch_size)
            self.max_episodes = config.get("max_episodes", self.max_episodes)
            self.max_timesteps = config.get("max_timesteps", self.max_timesteps)

            # 런타임 상태 복원
            runtime = checkpoint.get("runtime", {})
            self.time_step = runtime.get("time_step", 0)

            # 네트워크 재구성
            self._init_networks()

            # 모델 가중치 로드
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.old_actor.load_state_dict(checkpoint["actor"])

            # 옵티마이저 상태 로드
            self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor"])
            self.optimizer_critic.load_state_dict(checkpoint["optimizer_critic"])

            # 메타데이터 로드
            self.metadata = checkpoint.get("metadata", {})
            self.is_trained = checkpoint.get("is_trained", False)

            logger.info(f"PPO 모델 로드 완료: {path}")
            return True

        except Exception as e:
            logger.error(f"PPO 모델 로드 중 오류: {e}")
            return False


if __name__ == "__main__":
    logger.error(
        "이 모듈은 직접 실행할 수 없습니다. run/ 디렉토리의 스크립트를 통해 실행하세요."
    )
