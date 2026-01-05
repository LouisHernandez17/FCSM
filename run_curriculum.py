from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Paths
ROOT = Path(__file__).parent
PARTITION = ROOT / "scripts" / "partition_dataset.py"
TRAIN = ROOT / "train.py"
EVAL = ROOT / "evaluate_test.py"

# Recommended hyperparameters
PHASE1_CFG = {
    "data_dir": "dataset",
    "batch_size": 12,
    "epochs": 20,
    "lr": "2e-4",
    "pos_weight": "10.0",
    "gold_oversample": 2,
    "checkpoint_dir": "checkpoints/phase1_recall",
    "log_dir": "runs/phase1_recall",
    "log_interval": 50,
    "num_workers": 0,
    "max_nodes": 128,
    # Keep the set encoder trainable; it is randomly initialized.
    "freeze_encoder": False,
}

PHASE2_CFG = {
    "data_dir": "dataset",
    "batch_size": 12,
    "epochs": 10,
    "lr": "5e-5",
    "pos_weight": "3.0",
    "gold_oversample": 100,
    "resume_from": "checkpoints/phase1_recall/scfm_epoch_10.pt",
    "checkpoint_dir": "checkpoints/phase2_precision",
    "log_dir": "runs/phase2_precision",
    "log_interval": 50,
    "num_workers": 0,
    "max_nodes": 128,
    "freeze_encoder": False,
}

HELDOUT_DIR = "dataset_heldout_test"
BEST_MODEL = "checkpoints/phase2_precision/best_model.pt"


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def maybe_partition():
    dest = Path(HELDOUT_DIR) / "tier3_gold"
    if dest.exists() and any(dest.glob("*.json")):
        print(f"Held-out set already exists at {dest}; skipping partition.")
        return
    run([sys.executable, str(PARTITION)])


def phase1():
    args = [
        sys.executable,
        str(TRAIN),
        "--data-dir",
        PHASE1_CFG["data_dir"],
        "--batch-size",
        str(PHASE1_CFG["batch_size"]),
        "--epochs",
        str(PHASE1_CFG["epochs"]),
        "--lr",
        str(PHASE1_CFG["lr"]),
        "--pos-weight",
        str(PHASE1_CFG["pos_weight"]),
        "--gold-oversample",
        str(PHASE1_CFG["gold_oversample"]),
        "--checkpoint-dir",
        PHASE1_CFG["checkpoint_dir"],
        "--log-dir",
        PHASE1_CFG["log_dir"],
        "--log-interval",
        str(PHASE1_CFG["log_interval"]),
        "--num-workers",
        str(PHASE1_CFG["num_workers"]),
        "--max-nodes",
        str(PHASE1_CFG["max_nodes"]),
        "--device",
        "auto",
        "--freeze-encoder" if PHASE1_CFG["freeze_encoder"] else None,
    ]
    run(["uv", "run"] + [a for a in args if a is not None])


def phase2():
    resume_path = Path(PHASE2_CFG["resume_from"])
    if not resume_path.exists():
        print(f"ERROR: resume checkpoint not found: {resume_path}")
        sys.exit(1)
    args = [
        sys.executable,
        str(TRAIN),
        "--data-dir",
        PHASE2_CFG["data_dir"],
        "--batch-size",
        str(PHASE2_CFG["batch_size"]),
        "--epochs",
        str(PHASE2_CFG["epochs"]),
        "--lr",
        str(PHASE2_CFG["lr"]),
        "--pos-weight",
        str(PHASE2_CFG["pos_weight"]),
        "--gold-oversample",
        str(PHASE2_CFG["gold_oversample"]),
        "--resume-from",
        str(resume_path),
        "--checkpoint-dir",
        PHASE2_CFG["checkpoint_dir"],
        "--log-dir",
        PHASE2_CFG["log_dir"],
        "--log-interval",
        str(PHASE2_CFG["log_interval"]),
        "--num-workers",
        str(PHASE2_CFG["num_workers"]),
        "--max-nodes",
        str(PHASE2_CFG["max_nodes"]),
        "--device",
        "auto",
        "--freeze-encoder" if PHASE2_CFG["freeze_encoder"] else None,
    ]
    run(["uv", "run"] + [a for a in args if a is not None])


def evaluate():
    ckpt = Path(BEST_MODEL)
    if not ckpt.exists():
        print(f"Skip evaluation: checkpoint not found at {ckpt}")
        return
    if not Path(HELDOUT_DIR).exists():
        print(f"Skip evaluation: held-out dir not found at {HELDOUT_DIR}")
        return
    args = [
        sys.executable,
        str(EVAL),
        "--checkpoint",
        str(ckpt),
        "--data-dir",
        HELDOUT_DIR,
    ]
    run(["uv", "run"] + args)


def main():
    maybe_partition()
    phase1()
    phase2()
    evaluate()


if __name__ == "__main__":
    main()
