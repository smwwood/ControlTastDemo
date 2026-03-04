#!/usr/bin/env python3
"""
Compensatory tracking task (1D x-axis) with two control modes:
- human: mouse controls dM each frame
- model: built-in PD controller generates dM

Control relationships:
    C_t = M_t + D_t
where
    M_t: controller output (mouse/agent state)
    D_t: hidden disturbance, AR(1)
    C_t: visible cursor
    T_t: visible target
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame

# ------------------------------
# User-facing switch
# ------------------------------
CONTROL_MODE = "human"  # "human" or "model"


@dataclass
class EnvConfig:
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
    lambda_action: float = 0.0005
    seed: int = 7


EPS = 1e-8


def compute_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def compute_stability(signal: np.ndarray) -> float:
    """
    Stability metric in [0,1], higher is more stable.
    Based on normalized velocity variance: stable traces move less frame-to-frame.
    """
    if signal.size < 2:
        return 1.0
    vel = np.diff(signal)
    rms_vel = compute_rms(vel)
    denom = compute_rms(signal) + EPS
    val = 1.0 - (rms_vel / denom)
    return float(np.clip(val, 0.0, 1.0))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
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
    """Gymnasium-style tracking environment."""

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)
        self.max_steps = int(self.cfg.duration_seconds * self.cfg.fps)
        self.reset()

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.step_count = 0
        self.done = False

        self.M = 0.0
        self.D = 0.0
        self.T = 0.0
        self.C = self.M + self.D

        self.prev_C = self.C
        self.prev_T = self.T
        self.prev_E = self.C - self.T

        # Time series storage
        self.ts_T: List[float] = [self.T]
        self.ts_M: List[float] = [self.M]
        self.ts_D: List[float] = [self.D]
        self.ts_C: List[float] = [self.C]

        self.ts_dT: List[float] = []
        self.ts_dM: List[float] = []
        self.ts_dD: List[float] = []
        self.ts_dC: List[float] = []
        self.ts_E: List[float] = [self.prev_E]

        obs = self._get_obs(dC=0.0, dT=0.0)
        return obs, {}

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
        E = self.C - self.T
        return np.array([self.C, self.T, E, dC, dT], dtype=np.float32)

    def _terminal_info(self) -> Dict:
        E = np.asarray(self.ts_E, dtype=np.float32)
        C = np.asarray(self.ts_C, dtype=np.float32)
        dC = np.asarray(self.ts_dC, dtype=np.float32)
        dM = np.asarray(self.ts_dM, dtype=np.float32)
        dD = np.asarray(self.ts_dD, dtype=np.float32)

        info = {
            "rms_error": compute_rms(E),
            "stability": compute_stability(C),
            "corr_dC_dM": safe_corr(dC, dM),
            "corr_dM_dD": safe_corr(dM, dD),
            "corr_dC_dD": safe_corr(dC, dD),
        }
        return info

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            return self._get_obs(0.0, 0.0), 0.0, True, False, self._terminal_info()

        dM = float(np.clip(action * self.cfg.action_scale, -50.0, 50.0))

        prev_M, prev_D, prev_C, prev_T = self.M, self.D, self.C, self.T

        self.M = float(np.clip(self.M + dM, -self.cfg.m_bound, self.cfg.m_bound))
        self.D = self._update_disturbance()
        self.T = self._update_target()
        self.C = self.M + self.D

        dC = self.C - prev_C
        dT = self.T - prev_T
        dD = self.D - prev_D
        E = self.C - self.T

        reward = -(E ** 2) - self.cfg.lambda_action * (dM ** 2)

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
        self.ts_E.append(E)

        info = self._terminal_info() if terminated else {}
        obs = self._get_obs(dC=dC, dT=dT)
        return obs, float(reward), terminated, False, info


def pd_controller(error: float, d_error: float, kp: float = 0.35, kd: float = 0.22) -> float:
    """Simple built-in control model dM = -Kp*E - Kd*dE."""
    return float(-kp * error - kd * d_error)


def world_to_screen_x(x_world: float, width: int) -> int:
    return int(width // 2 + x_world)


def draw_button(screen: pygame.Surface, rect: pygame.Rect, text: str, font: pygame.font.Font) -> None:
    pygame.draw.rect(screen, (70, 70, 70), rect, border_radius=6)
    pygame.draw.rect(screen, (200, 200, 200), rect, width=2, border_radius=6)
    label = font.render(text, True, (240, 240, 240))
    screen.blit(label, label.get_rect(center=rect.center))


def run_plot(env: TrackingEnv, model_trace: np.ndarray, corr_with_model: float, mode_used: str) -> None:
    t = np.arange(len(env.ts_C)) / env.cfg.fps
    C = np.asarray(env.ts_C)
    M = np.asarray(env.ts_M)
    D = np.asarray(env.ts_D)

    plt.figure(figsize=(10, 5))
    plt.plot(t, C, label="Cursor C", color="tab:blue", linewidth=2)
    plt.plot(t, M, label="Controller output M", color="tab:green", linewidth=2)
    plt.plot(t, D, label="Disturbance D", color="tab:red", linewidth=2)

    tt = np.arange(model_trace.size) / env.cfg.fps
    plt.plot(tt, model_trace, label="Model dM (black)", color="black", linewidth=1.5)

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Position / Delta")
    plt.title(f"Tracking run ({mode_used})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show(block=False)


def run_ui() -> None:
    pygame.init()
    width, height = 900, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Compensatory Tracking Task")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 20)
    small_font = pygame.font.SysFont("arial", 17)

    cfg = EnvConfig()
    env = TrackingEnv(cfg)

    button_rect = pygame.Rect(width - 150, 20, 120, 40)

    running = True
    episode_active = True
    obs, _ = env.reset()

    model_dM_trace: List[float] = []
    actor_dM_trace: List[float] = []

    # For human control, use relative mouse movement each frame.
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    terminal_info = {}

    while running:
        dM_human = 0.0
        clicked_new_run = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    clicked_new_run = True
            elif event.type == pygame.MOUSEMOTION and CONTROL_MODE == "human" and episode_active:
                dM_human += float(event.rel[0])

        if clicked_new_run:
            obs, _ = env.reset()
            episode_active = True
            terminal_info = {}
            model_dM_trace.clear()
            actor_dM_trace.clear()

        if episode_active:
            E = float(obs[2])
            dE = float(obs[3] - obs[4])
            dM_model = pd_controller(E, dE)
            model_dM_trace.append(dM_model)

            if CONTROL_MODE == "human":
                action = dM_human
            else:
                action = dM_model

            actor_dM_trace.append(float(action))
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episode_active = False
                terminal_info = info

                corr_actor_model = safe_corr(
                    np.asarray(actor_dM_trace, dtype=np.float32),
                    np.asarray(model_dM_trace, dtype=np.float32),
                )
                print("\n=== Run Summary ===")
                print(f"Mode: {CONTROL_MODE}")
                print(f"RMS Error: {info['rms_error']:.4f}")
                print(f"Stability: {info['stability']:.4f}")
                print(f"corr(dC,dM): {info['corr_dC_dM']:.4f}")
                print(f"corr(dM,dD): {info['corr_dM_dD']:.4f}")
                print(f"corr(dC,dD): {info['corr_dC_dD']:.4f}")
                print(f"corr(dM_actor,dM_model): {corr_actor_model:.4f}")

                run_plot(
                    env,
                    model_trace=np.asarray(model_dM_trace, dtype=np.float32),
                    corr_with_model=corr_actor_model,
                    mode_used=CONTROL_MODE,
                )

        screen.fill((22, 22, 22))

        # Center reference line for x=0
        pygame.draw.line(screen, (90, 90, 90), (width // 2, 0), (width // 2, height), 1)

        # Draw target and cursor as short horizontal segments.
        top_y = int(height * 0.35)
        bottom_y = int(height * 0.65)
        seg_half_len = 24

        tx = world_to_screen_x(env.T, width)
        cx = world_to_screen_x(env.C, width)

        pygame.draw.line(screen, (255, 220, 70), (tx - seg_half_len, top_y), (tx + seg_half_len, top_y), 5)
        pygame.draw.line(screen, (70, 220, 255), (cx - seg_half_len, bottom_y), (cx + seg_half_len, bottom_y), 5)

        draw_button(screen, button_rect, "New Run", font)

        status = "RUNNING" if episode_active else "DONE"
        e_now = env.C - env.T
        hud_lines = [
            f"Mode: {CONTROL_MODE}",
            f"Status: {status}",
            f"Step: {env.step_count}/{env.max_steps}",
            f"T: {env.T:.2f}   C: {env.C:.2f}   M: {env.M:.2f}",
            f"Error E=C-T: {e_now:.2f}",
        ]
        if terminal_info:
            hud_lines.extend(
                [
                    f"RMS: {terminal_info.get('rms_error', 0.0):.3f}",
                    f"Stability: {terminal_info.get('stability', 0.0):.3f}",
                ]
            )

        for i, line in enumerate(hud_lines):
            surf = small_font.render(line, True, (230, 230, 230))
            screen.blit(surf, (20, 20 + 22 * i))

        instr = small_font.render("ESC to quit. Click New Run to reset.", True, (190, 190, 190))
        screen.blit(instr, (20, height - 35))

        pygame.display.flip()
        clock.tick(cfg.fps)

    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    pygame.quit()


if __name__ == "__main__":
    run_ui()
