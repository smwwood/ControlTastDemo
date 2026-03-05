#!/usr/bin/env python3
"""Pygame demo UI for the tracking environment.

Run examples:
    python tracking_demo.py --mode human
    python tracking_demo.py --mode model
"""

from __future__ import annotations

import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pygame

from policies import PDPolicy
from tracking_env import EnvConfig, TrackingEnv, safe_corr

CONTROL_MODE = "human"  # default: "human" or "model"


def world_to_screen_x(x_world: float, width: int) -> int:
    """Convert world x-coordinate to screen x-coordinate."""
    return int(width // 2 + x_world)


def draw_button(screen: pygame.Surface, rect: pygame.Rect, text: str, font: pygame.font.Font) -> None:
    """Draw a rounded rectangular button with centered text."""
    pygame.draw.rect(screen, (70, 70, 70), rect, border_radius=6)
    pygame.draw.rect(screen, (200, 200, 200), rect, width=2, border_radius=6)
    label = font.render(text, True, (240, 240, 240))
    screen.blit(label, label.get_rect(center=rect.center))


def run_plot(env: TrackingEnv, model_trace: np.ndarray, mode_used: str) -> None:
    """Render end-of-episode trajectory plots."""
    t = np.arange(len(env.ts_C)) / env.cfg.fps
    c = np.asarray(env.ts_C)
    m = np.asarray(env.ts_M)
    d = np.asarray(env.ts_D)

    plt.figure(figsize=(10, 5))
    plt.plot(t, c, label="Cursor C", color="tab:blue", linewidth=2)
    plt.plot(t, m, label="Controller output M", color="tab:green", linewidth=2)
    plt.plot(t, d, label="Disturbance D", color="tab:red", linewidth=2)

    tt = np.arange(model_trace.size) / env.cfg.fps
    plt.plot(tt, model_trace, label="Model dM (black)", color="black", linewidth=1.3)

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Position / Delta")
    plt.title(f"Tracking run ({mode_used})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show(block=False)


def run_ui(control_mode: str = CONTROL_MODE) -> None:
    """Run the interactive pygame demo.

    Args:
        control_mode: ``human`` for mouse control or ``model`` for PD control.
    """
    if control_mode not in {"human", "model"}:
        raise ValueError("control_mode must be 'human' or 'model'")

    pygame.init()
    width, height = 900, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Compensatory Tracking Demo")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 20)
    small_font = pygame.font.SysFont("arial", 17)

    cfg = EnvConfig(fps=60, duration_seconds=10.0)
    env = TrackingEnv(cfg)
    pd_policy = PDPolicy()

    button_rect = pygame.Rect(width - 150, 20, 120, 40)

    running = True
    episode_active = True
    obs, _ = env.reset(seed=cfg.seed)
    terminal_info = {}

    model_dM_trace: List[float] = []
    actor_dM_trace: List[float] = []

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

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
            elif event.type == pygame.MOUSEMOTION and control_mode == "human" and episode_active:
                dM_human += float(event.rel[0])

        if clicked_new_run:
            obs, _ = env.reset(seed=cfg.seed)
            episode_active = True
            terminal_info = {}
            model_dM_trace.clear()
            actor_dM_trace.clear()

        if episode_active:
            dM_model = pd_policy(obs)
            model_dM_trace.append(dM_model)

            action = dM_human if control_mode == "human" else dM_model
            actor_dM_trace.append(float(action))
            obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                episode_active = False
                terminal_info = info

                corr_actor_model = safe_corr(
                    np.asarray(actor_dM_trace, dtype=np.float32),
                    np.asarray(model_dM_trace, dtype=np.float32),
                )

                print("\n=== Run Summary ===")
                print(f"Mode: {control_mode}")
                print(f"RMS Error: {info['rms_error']:.4f}")
                print(f"Stability: {info['stability']:.4f}")
                print(f"corr(dC,dM): {info['corr_dC_dM']:.4f}")
                print(f"corr(dM,dD): {info['corr_dM_dD']:.4f}")
                print(f"corr(dC,dD): {info['corr_dC_dD']:.4f}")
                print(f"corr(dM_actor,dM_model): {corr_actor_model:.4f}")

                run_plot(env, model_trace=np.asarray(model_dM_trace, dtype=np.float32), mode_used=control_mode)

        screen.fill((22, 22, 22))
        pygame.draw.line(screen, (90, 90, 90), (width // 2, 0), (width // 2, height), 1)

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
            f"Mode: {control_mode}",
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


def main() -> None:
    """CLI entrypoint for selecting control mode."""
    parser = argparse.ArgumentParser(description="Tracking demo UI")
    parser.add_argument("--mode", choices=["human", "model"], default=CONTROL_MODE)
    args = parser.parse_args()
    run_ui(control_mode=args.mode)


if __name__ == "__main__":
    main()
