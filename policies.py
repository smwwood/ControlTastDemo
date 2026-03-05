"""Policy helpers for tracking control and training scaffolds."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class PDPolicy:
    """Simple proportional-derivative policy over tracking error."""

    kp: float = 0.35
    kd: float = 0.22

    def __call__(self, obs: np.ndarray) -> float:
        e = float(obs[2])
        dE = float(obs[3] - obs[4])
        return float(-self.kp * e - self.kd * dE)


class TorchPolicyWrapper:
    """Wrap a torch policy(obs)->action with control-side effects.

    Supports:
      * action clamp on dM command
      * optional fixed action latency (in timesteps)
      * optional additive gaussian motor noise
    """

    def __init__(
        self,
        module: torch.nn.Module,
        action_clip: float = 50.0,
        latency_steps: int = 0,
        motor_noise_std: float = 0.0,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.module = module
        self.action_clip = float(action_clip)
        self.latency_steps = max(0, int(latency_steps))
        self.motor_noise_std = float(max(0.0, motor_noise_std))
        self.rng = np.random.default_rng(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(self.device)
        self.module.eval()

        self._latency_queue: Deque[float] = deque([0.0] * self.latency_steps, maxlen=self.latency_steps)

    def _model_action(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.module(obs_t)
            action = float(out.squeeze().detach().cpu().item())
        return action

    def __call__(self, obs: np.ndarray) -> float:
        action = self._model_action(obs)

        if self.latency_steps > 0:
            self._latency_queue.append(action)
            delayed = self._latency_queue[0]
        else:
            delayed = action

        noisy = delayed + float(self.rng.normal(0.0, self.motor_noise_std))
        return float(np.clip(noisy, -self.action_clip, self.action_clip))


class PPOPolicy:
    """Minimal PPO policy with Gaussian actor and value critic.

    This class provides action sampling/evaluation and an in-memory rollout buffer
    for lightweight training in ``train_minimal.py``.
    """

    def __init__(
        self,
        obs_dim: int = 5,
        hidden_dim: int = 64,
        action_std: float = 8.0,
        action_clip: float = 50.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 4,
        device: Optional[str] = None,
    ):
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.action_clip = action_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(self.device)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(self.device)
        self.log_std = torch.nn.Parameter(torch.tensor(np.log(action_std), dtype=torch.float32, device=self.device))

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], lr=lr
        )

        self.reset_buffer()

    def reset_buffer(self) -> None:
        self.obs_buf: List[np.ndarray] = []
        self.act_buf: List[float] = []
        self.logp_buf: List[float] = []
        self.rew_buf: List[float] = []
        self.done_buf: List[float] = []
        self.val_buf: List[float] = []

    def _dist_and_value(self, obs_t: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        mean = self.actor(obs_t)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.critic(obs_t)
        return dist, value

    def act(self, obs: np.ndarray) -> Tuple[float, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self._dist_and_value(obs_t)
            action_t = dist.sample()
            logp_t = dist.log_prob(action_t).sum(dim=-1)
        action = float(action_t.squeeze().cpu().item())
        action = float(np.clip(action, -self.action_clip, self.action_clip))
        return action, float(logp_t.item()), float(value.squeeze().item())

    def store(self, obs: np.ndarray, action: float, logp: float, reward: float, done: bool, value: float) -> None:
        self.obs_buf.append(obs.astype(np.float32))
        self.act_buf.append(float(action))
        self.logp_buf.append(float(logp))
        self.rew_buf.append(float(reward))
        self.done_buf.append(float(done))
        self.val_buf.append(float(value))

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        if not self.obs_buf:
            return {"ppo_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        rews = np.array(self.rew_buf, dtype=np.float32)
        vals = np.array(self.val_buf + [last_value], dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        adv = np.zeros_like(rews)
        gae = 0.0
        for t in reversed(range(len(rews))):
            nonterminal = 1.0 - dones[t]
            delta = rews[t] + self.gamma * vals[t + 1] * nonterminal - vals[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + vals[:-1]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.array(self.act_buf), dtype=torch.float32, device=self.device).unsqueeze(-1)
        old_logp_t = torch.as_tensor(np.array(self.logp_buf), dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        ppo_loss_v = 0.0
        value_loss_v = 0.0
        entropy_v = 0.0

        for _ in range(self.update_epochs):
            dist, value_t = self._dist_and_value(obs_t)
            logp_t = dist.log_prob(act_t).sum(dim=-1)
            ratio = torch.exp(logp_t - old_logp_t)

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = torch.nn.functional.mse_loss(value_t.squeeze(-1), ret_t)
            entropy = dist.entropy().mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], max_norm=0.5
            )
            self.optimizer.step()

            ppo_loss_v = float(policy_loss.item())
            value_loss_v = float(value_loss.item())
            entropy_v = float(entropy.item())

        self.reset_buffer()
        return {"ppo_loss": ppo_loss_v, "value_loss": value_loss_v, "entropy": entropy_v}
