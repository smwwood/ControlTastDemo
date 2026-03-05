"""Core tracking environment without UI dependencies.

This module provides a small gymnasium-style environment for a 1D compensatory
tracking task:

    C_t = M_t + D_t

Where ``M`` is controller output (action-integrated), ``D`` is an AR(1)
disturbance, ``C`` is cursor, and ``T`` is target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-8


@dataclass
class EnvConfig:
    """Configuration values for :class:`TrackingEnv`."""

    fps: int = 60
    duration_seconds: float = 10.0
    m_bound: float = 400.0
    d_bound: float = 300.0
    target_bound: float = 350.0
    target_mode: str = "random_walk"  # "fixed" or "random_walk"
    target_rw_sigma: float = 1.2
    target_rw_clip_step: float = 4.0
    rho: float = 0.98
    sigma: float = 1.8
    action_scale: float = 1.0
    action_clip: float = 50.0
    lambda_action: float = 0.0005
    seed: int = 7


def compute_rms(x: np.ndarray) -> float:
    """Compute root-mean-square of a vector (0.0 for empty vectors)."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation defensively for short/degenerate vectors."""
    n = min(x.size, y.size)
    if n < 2:
        return 0.0
    x0 = x[:n]
    y0 = y[:n]
    sx = np.std(x0)
    sy = np.std(y0)
    if sx < EPS or sy < EPS:
        return 0.0
    return float(np.corrcoef(x0, y0)[0, 1])


class TrackingEnv:
    """Gymnasium-style tracking environment (no pygame dependency)."""

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self.max_steps = int(self.cfg.duration_seconds * self.cfg.fps)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.reset(seed=self.cfg.seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset state and trajectory storage.

        Args:
            seed: Optional RNG seed for deterministic episodes.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.done = False

        self.M = 0.0
        self.D = 0.0
        self.T = 0.0
        self.C = self.M + self.D

        self.ts_T: List[float] = [self.T]
        self.ts_M: List[float] = [self.M]
        self.ts_D: List[float] = [self.D]
        self.ts_C: List[float] = [self.C]

        self.ts_dT: List[float] = []
        self.ts_dM: List[float] = []
        self.ts_dD: List[float] = []
        self.ts_dC: List[float] = []
        self.ts_E: List[float] = [self.C - self.T]

        return self._get_obs(dC=0.0, dT=0.0), {}

    def _update_target(self) -> float:
        if self.cfg.target_mode == "fixed":
            return 0.0
        step = float(self.rng.normal(0.0, self.cfg.target_rw_sigma))
        step = float(np.clip(step, -self.cfg.target_rw_clip_step, self.cfg.target_rw_clip_step))
        return float(np.clip(self.T + step, -self.cfg.target_bound, self.cfg.target_bound))

    def _update_disturbance(self) -> float:
        nxt = self.cfg.rho * self.D + self.cfg.sigma * float(self.rng.normal(0.0, 1.0))
        return float(np.clip(nxt, -self.cfg.d_bound, self.cfg.d_bound))

    def _get_obs(self, dC: float, dT: float) -> np.ndarray:
        e = self.C - self.T
        return np.array([self.C, self.T, e, dC, dT], dtype=np.float32)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute episode metrics from stored trajectories."""
        e = np.asarray(self.ts_E, dtype=np.float32)
        d = np.asarray(self.ts_D, dtype=np.float32)
        dc = np.asarray(self.ts_dC, dtype=np.float32)
        dm = np.asarray(self.ts_dM, dtype=np.float32)
        dd = np.asarray(self.ts_dD, dtype=np.float32)

        stability = float(np.std(d) / (np.std(e) + EPS))
        return {
            "rms_error": compute_rms(e),
            "stability": stability,
            "corr_dC_dM": safe_corr(dc, dm),
            "corr_dM_dD": safe_corr(dm, dd),
            "corr_dC_dD": safe_corr(dc, dd),
        }

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one environment step.

        Action is interpreted as delta-M command (dM).
        """
        if self.done:
            info = self.compute_metrics()
            return self._get_obs(0.0, 0.0), 0.0, True, False, info

        dM = float(np.clip(action * self.cfg.action_scale, -self.cfg.action_clip, self.cfg.action_clip))
        prev_c, prev_t, prev_d = self.C, self.T, self.D

        self.M = float(np.clip(self.M + dM, -self.cfg.m_bound, self.cfg.m_bound))
        self.D = self._update_disturbance()
        self.T = self._update_target()
        self.C = self.M + self.D

        dC = self.C - prev_c
        dT = self.T - prev_t
        dD = self.D - prev_d
        e = self.C - self.T

        reward = -(e**2) - self.cfg.lambda_action * (dM**2)

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        self.done = terminated

        self.ts_T.append(self.T)
        self.ts_M.append(self.M)
        self.ts_D.append(self.D)
        self.ts_C.append(self.C)
        self.ts_dT.append(dT)
        self.ts_dM.append(dM)
        self.ts_dD.append(dD)
        self.ts_dC.append(dC)
        self.ts_E.append(e)

        info = self.compute_metrics() if terminated else {}
        return self._get_obs(dC=dC, dT=dT), float(reward), terminated, False, info
