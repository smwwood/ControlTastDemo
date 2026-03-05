# ControlTastDemo

A compensatory tracking task project refactored into modular components for demos and RL experiments.

## Project structure

- `tracking_env.py`: core gymnasium-style environment (no pygame dependency).
- `tracking_demo.py`: pygame UI runner with human and PD model modes.
- `policies.py`: `PDPolicy`, `TorchPolicyWrapper`, and trainable `PPOPolicy`.
- `train_minimal.py`: minimal RL scaffold for random / PD / torch / PPO rollouts.
- `tracking_task.py`: backward-compatible entrypoint that delegates to the demo.

---

## 1) Setup a virtual environment

### Prerequisites
- Python 3.10+
- `pip`

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Quick sanity check

```bash
python -m py_compile tracking_env.py policies.py tracking_demo.py train_minimal.py tracking_task.py
```

---

## 2) Run the demo

All demo runs are 10 seconds at 60Hz by default.

### Human mode

```bash
python tracking_demo.py --mode human
```

- Move mouse left/right to control the lower cursor line.
- Press `ESC` to quit.
- Click **New Run** to reset the episode.

### PD controller mode

```bash
python tracking_demo.py --mode model
```

This uses the built-in `PDPolicy` to control the cursor.

### Legacy command (backward compatible)

```bash
python tracking_task.py
```

---

## 3) Train / evaluate policies

### Random policy

```bash
python train_minimal.py --policy random --episodes 20
```

### PD baseline

```bash
python train_minimal.py --policy pd --episodes 20
```

### Torch wrapper baseline (untrained network)

```bash
python train_minimal.py --policy torch --episodes 20
```

### PPO training

```bash
python train_minimal.py --policy ppo --episodes 200
```

This trains a lightweight PPO agent (actor + critic) directly in PyTorch. Per episode, the script logs:
- `reward`
- `rms_error`
- `stability`
- PPO losses (`ppo_loss`, `value_loss`)

### SAC placeholder

```bash
python train_minimal.py --policy sac
```

SAC is still a stub and prints an integration message.

---

## Reward function used for RL

Reward is computed **every frame** as:

```text
reward_t = -0.001 * |C_t - T_t|
```

Where:
- `C_t` is cursor x-position,
- `T_t` is target x-position,
- `|C_t - T_t|` is pixel distance between the two lines.

So:
- Perfect alignment (`C_t == T_t`) gives reward `0.0`.
- Larger separation gives more negative reward.

This per-frame reward is what PPO trains against.

---

## Environment metrics

At episode end, the environment reports:
- `rms_error`: root mean square of tracking error (`E = C - T`)
- `stability`: `std(D) / (std(E) + eps)`
- `corr_dC_dM`: correlation between delta cursor and delta controller output
- `corr_dM_dD`: correlation between delta controller output and delta disturbance
- `corr_dC_dD`: correlation between delta cursor and delta disturbance
