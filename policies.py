"""Policy helpers for tracking control and training scaffolds."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

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
