#!/usr/bin/env python3
"""Minimal RL training scaffold for TrackingEnv.

This script intentionally keeps learning logic lightweight while exposing:
- random policy baseline
- PD baseline
- torch module wrapped policy
- placeholder path for PPO/SAC integration
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict

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
    if name in {"ppo", "sac"}:
        raise NotImplementedError(
            f"Policy '{name}' is a placeholder. Integrate your trainer in make_policy()."
        )
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
