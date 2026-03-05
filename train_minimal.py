#!/usr/bin/env python3
"""Minimal RL training scaffold for TrackingEnv.

This script intentionally keeps learning logic lightweight while exposing:
- random policy baseline
- PD baseline
- torch module wrapped policy
- lightweight PPO integration
- placeholder path for SAC integration
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from policies import PDPolicy, TorchPolicyWrapper
from tracking_env import EnvConfig, TrackingEnv


@dataclass
class TrainConfig:
    episodes: int = 5
    seed: int = 7


class TinyPolicyNet(torch.nn.Module):
    """Small MLP returning a single scalar dM command."""

    def __init__(self, obs_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TinyValueNet(torch.nn.Module):
    """Small MLP estimating V(s)."""

    def __init__(self, obs_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class PPOAgent:
    """Lightweight on-policy PPO agent for this small demo environment."""

    def __init__(self, obs_dim: int = 5, action_clip: float = 50.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor = TinyPolicyNet(obs_dim=obs_dim, hidden_dim=64).to(self.device)
        self.critic = TinyValueNet(obs_dim=obs_dim, hidden_dim=64).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.action_std = 8.0
        self.action_clip = float(action_clip)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.train_epochs = 6

    def sample_action(self, obs: np.ndarray) -> Tuple[float, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean = self.actor(obs_t)
        std = torch.full_like(mean, self.action_std)
        dist = torch.distributions.Normal(mean, std)
        action_t = dist.sample()
        logp_t = dist.log_prob(action_t).sum(dim=-1)
        value_t = self.critic(obs_t).squeeze(-1)

        action = float(torch.clamp(action_t.squeeze(0), -self.action_clip, self.action_clip).item())
        return action, float(logp_t.item()), float(value_t.item())

    def act(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t = self.actor(obs_t)
        return float(torch.clamp(action_t.squeeze(0), -self.action_clip, self.action_clip).item())

    def update_from_episode(self, trajectory: Dict[str, List[float]]) -> None:
        obs = torch.as_tensor(np.asarray(trajectory["obs"], dtype=np.float32), device=self.device)
        acts = torch.as_tensor(np.asarray(trajectory["acts"], dtype=np.float32), device=self.device).unsqueeze(-1)
        old_logp = torch.as_tensor(np.asarray(trajectory["logp"], dtype=np.float32), device=self.device)
        rewards = np.asarray(trajectory["rews"], dtype=np.float32)
        values = np.asarray(trajectory["vals"], dtype=np.float32)

        adv, ret = self._compute_gae(rewards, values)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        for _ in range(self.train_epochs):
            mean = self.actor(obs)
            std = torch.full_like(mean, self.action_std)
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(acts).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = self.critic(obs).squeeze(-1)
            value_loss = torch.nn.functional.mse_loss(value_pred, ret_t)

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=0.5)
            self.actor_optim.step()
            self.critic_optim.step()

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = rewards.shape[0]
        adv = np.zeros_like(rewards)
        last_gae = 0.0
        next_value = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            adv[t] = last_gae
            next_value = values[t]
        returns = adv + values

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv.astype(np.float32), returns.astype(np.float32)


def run_episode(env: TrackingEnv, policy_fn: Callable[[np.ndarray], float], seed: int) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0

    while True:
        action = float(policy_fn(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            metrics = dict(info)
            metrics["episode_reward"] = float(total_reward)
            return metrics


def run_ppo_episode(env: TrackingEnv, agent: PPOAgent, seed: int) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    traj: Dict[str, List[float]] = {"obs": [], "acts": [], "logp": [], "rews": [], "vals": []}

    while True:
        action, logp, value = agent.sample_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        traj["obs"].append(obs.tolist())
        traj["acts"].append(action)
        traj["logp"].append(logp)
        traj["rews"].append(float(reward))
        traj["vals"].append(value)

        obs = next_obs
        total_reward += reward

        if terminated or truncated:
            agent.update_from_episode(traj)
            metrics = dict(info)
            metrics["episode_reward"] = float(total_reward)
            return metrics


def random_policy(_: np.ndarray) -> float:
    return float(np.random.uniform(-10.0, 10.0))


def make_policy(name: str) -> Callable[[np.ndarray], float]:
    if name == "random":
        return random_policy
    if name == "pd":
        return PDPolicy()
    if name == "torch":
        net = TinyPolicyNet()
        wrapper = TorchPolicyWrapper(net, action_clip=50.0, latency_steps=0, motor_noise_std=0.0)
        return wrapper
    if name == "ppo":
        raise RuntimeError("PPO is stateful and is handled in main().")
    if name == "sac":
        raise NotImplementedError("Policy 'sac' is a placeholder. Integrate your trainer in make_policy().")
    raise ValueError(f"Unknown policy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal tracking RL scaffold")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "pd", "torch", "ppo", "sac"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_cfg = TrainConfig(episodes=args.episodes, seed=args.seed)
    env = TrackingEnv(EnvConfig(seed=train_cfg.seed))

    if args.policy == "ppo":
        ppo_agent = PPOAgent(action_clip=env.cfg.action_clip)
        for ep in range(train_cfg.episodes):
            metrics = run_ppo_episode(env, ppo_agent, seed=train_cfg.seed + ep)
            print(
                f"episode={ep + 1:03d} "
                f"reward={metrics['episode_reward']:.3f} "
                f"rms_error={metrics['rms_error']:.3f} "
                f"stability={metrics['stability']:.3f}"
            )
        return

    try:
        policy = make_policy(args.policy)
    except NotImplementedError as exc:
        print(exc)
        return

    for ep in range(train_cfg.episodes):
        metrics = run_episode(env, policy, seed=train_cfg.seed + ep)
        print(
            f"episode={ep + 1:03d} "
            f"reward={metrics['episode_reward']:.3f} "
            f"rms_error={metrics['rms_error']:.3f} "
            f"stability={metrics['stability']:.3f}"
        )


if __name__ == "__main__":
    main()
