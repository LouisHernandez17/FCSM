from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import optuna

# Config
ROOT = Path(__file__).parent.parent
TRAIN = ROOT / "train.py"
STUDY_NAME = "scfm_phase1_opt"
STORAGE = "sqlite:///optuna_phase1.db"
N_TRIALS = 20
PYTHON_EXE = sys.executable

# Fixed phase-1-ish defaults (lightweight to avoid OOM)
FIXED_ARGS = [
    "--data-dir",
    "dataset",
    "--batch-size",
    "12",
    "--epochs",
    "6",  # shorter runs for faster search
    "--bnlearn-oversample",
    "2",
    "--log-interval",
    "50",
    "--num-workers",
    "0",
    "--max-nodes",
    "128",
    "--device",
    "auto",
]


def parse_best_f1(log_output: str) -> float:
    """Extract highest val_F1 seen in the run."""
    matches = re.findall(r"val_F1=([0-9.]+)", log_output)
    if not matches:
        return 0.0
    return max(float(m) for m in matches)


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    pos_weight = trial.suggest_float("pos_weight", 1.0, 15.0, step=0.5)

    run_name = f"optuna_trial_{trial.number}"
    checkpoint_dir = ROOT / "checkpoints" / "optuna_phase1" / run_name
    log_dir = ROOT / "runs" / run_name

    cmd = [
        "uv",
        "run",
        PYTHON_EXE,
        str(TRAIN),
        *FIXED_ARGS,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--log-dir",
        str(log_dir),
        "--lr",
        f"{lr}",
        "--pos-weight",
        f"{pos_weight}",
    ]

    print(f"\n[Trial {trial.number}] lr={lr:.2e} pos_weight={pos_weight:.1f}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        stdout = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[Trial {trial.number}] failed: {e}")
        if e.stdout:
            print(e.stdout.splitlines()[-5:])
        if e.stderr:
            print(e.stderr.splitlines()[-5:])
        return 0.0

    best_f1 = parse_best_f1(stdout)
    print(f"[Trial {trial.number}] best val_F1={best_f1:.4f}")
    return best_f1


def main():
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
    )

    print(f"Starting Optuna study: {STUDY_NAME} (trials={N_TRIALS})")
    print(f"Storage: {STORAGE}")

    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("Interrupted; partial results saved.")

    print("\n==== BEST TRIAL ====")
    print(f"Value (val_F1): {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
