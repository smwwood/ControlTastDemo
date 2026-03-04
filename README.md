# ControlTastDemo

A small compensatory tracking task project, refactored from a single-file prototype into modular components suitable for demos and RL experiments.

## Project structure

- `tracking_env.py`: core gymnasium-style environment (no pygame dependency).
- `tracking_demo.py`: pygame UI runner with human and PD model modes.
- `policies.py`: control policies (`PDPolicy`) and a PyTorch wrapper (`TorchPolicyWrapper`).
- `train_minimal.py`: minimal RL scaffold for random / PD / torch policy rollouts.
- `tracking_task.py`: backward-compatible entrypoint that delegates to the demo.

---

## 1) Setup: create and activate a virtual environment

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

## 2) Run the tracking demo

All demo runs are 10 seconds at 60Hz by default.

### A) Human mode
Use your mouse movement to control the controller output (`dM`).

```bash
python tracking_demo.py --mode human
```

Notes:
- Move mouse left/right to control the cursor.
- Press `ESC` to quit.
- Click **New Run** to reset and replay.
- At episode end, metrics are printed and a matplotlib plot is shown.

### B) PD controller mode (model mode)
Use the built-in proportional-derivative control policy.

```bash
python tracking_demo.py --mode model
```

This runs the same environment but uses `PDPolicy` for actions.

### Backward-compatible legacy command

```bash
python tracking_task.py
```

(Equivalent to running the demo in the default control mode.)

---

## 3) Run RL/minimal training scaffold

`train_minimal.py` provides a lightweight rollout loop for baseline comparisons and future RL integration.

### Random policy baseline

```bash
python train_minimal.py --policy random --episodes 5
```

### PD baseline

```bash
python train_minimal.py --policy pd --episodes 5
```

### Torch policy wrapper baseline

```bash
python train_minimal.py --policy torch --episodes 5
```

### Placeholder modes

```bash
python train_minimal.py --policy ppo
python train_minimal.py --policy sac
```

These currently print a placeholder message; integrate your PPO/SAC trainer in `make_policy(...)`.

---

## Metrics logged

At episode end, the environment reports:
- `rms_error`: root mean square of tracking error (`E = C - T`)
- `stability`: `std(D) / (std(E) + eps)`
- `corr_dC_dM`: correlation between delta cursor and delta controller output
- `corr_dM_dD`: correlation between delta controller output and delta disturbance
- `corr_dC_dD`: correlation between delta cursor and delta disturbance

Training logs include:
- `episode_reward`
- `rms_error`
- `stability`
