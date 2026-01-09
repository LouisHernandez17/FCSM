import random
import shutil
from pathlib import Path
from typing import List

# Config
DATASET_DIR = Path("dataset")
BNLEARN_DIR = DATASET_DIR / "bnlearn"
CONCEPTNET_DIR = DATASET_DIR / "conceptnet"
CAUSENET_DIR = DATASET_DIR / "causenet"

# TEST SET (final exam)
TEST_DIR = Path("dataset_heldout_test")
TEST_NETWORKS = ["survey", "insurance", "sachs"]

# VALIDATION SET (midterm)
VAL_DIR = Path("dataset_heldout_val")
VAL_NETWORKS = ["alarm", "child"]
VAL_CONCEPTNET_COUNT = 500
VAL_CAUSENET_COUNT = 500


def is_network_file(filename: str, net_ids: List[str]) -> bool:
    # Handles bnlearn_<net>_* patterns and raw <net> names
    for nid in net_ids:
        if filename == f"{nid}.json" or filename.startswith(f"{nid}_"):
            return True
        if filename.startswith(f"bnlearn_{nid}"):
            return True
    return False


def move_files(dest_dir: Path, files: List[Path]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.move(str(file_path), str(dest_dir / file_path.name))
    print(f"  Moved {len(files)} files to {dest_dir}")


def partition():
    if not BNLEARN_DIR.exists():
        print(f"Error: {BNLEARN_DIR} does not exist. Build data first.")
        return

    print("--- Partitioning Dataset ---")

    all_bnlearn = list(BNLEARN_DIR.glob("*.json"))
    test_files = [f for f in all_bnlearn if is_network_file(f.name, TEST_NETWORKS)]
    print(f"Partitioning TEST set: {TEST_NETWORKS} ({len(test_files)} files)...")
    move_files(TEST_DIR / "bnlearn", test_files)

    val_files = [f for f in all_bnlearn if is_network_file(f.name, VAL_NETWORKS)]
    print(f"Partitioning VALIDATION set: {VAL_NETWORKS} ({len(val_files)} files)...")
    move_files(VAL_DIR / "bnlearn", val_files)

    if CONCEPTNET_DIR.exists():
        val_cn_dir = VAL_DIR / "conceptnet"
        if val_cn_dir.exists() and any(val_cn_dir.glob("*.json")):
            print(f"Skipping ConceptNet partition: {val_cn_dir} already has files.")
        else:
            all_cn = list(CONCEPTNET_DIR.glob("*.json"))
            if len(all_cn) > VAL_CONCEPTNET_COUNT:
                random.seed(42)
                val_cn_files = random.sample(all_cn, VAL_CONCEPTNET_COUNT)
                print(
                    f"Partitioning VALIDATION ConceptNet: Random ({len(val_cn_files)} files)..."
                )
                move_files(val_cn_dir, val_cn_files)
            else:
                print(
                    f"Skipping ConceptNet partition: {len(all_cn)} files found "
                    f"(need > {VAL_CONCEPTNET_COUNT})."
                )
    else:
        print(f"Skipping ConceptNet partition: {CONCEPTNET_DIR} does not exist.")

    if CAUSENET_DIR.exists():
        val_causenet_dir = VAL_DIR / "causenet"
        if val_causenet_dir.exists() and any(val_causenet_dir.glob("*.json")):
            print(f"Skipping CauseNet partition: {val_causenet_dir} already has files.")
        else:
            all_causenet = list(CAUSENET_DIR.glob("*.json"))
            if len(all_causenet) > VAL_CAUSENET_COUNT:
                random.seed(42)
                val_causenet_files = random.sample(all_causenet, VAL_CAUSENET_COUNT)
                print(
                    f"Partitioning VALIDATION CauseNet: Random ({len(val_causenet_files)} files)..."
                )
                move_files(val_causenet_dir, val_causenet_files)
            else:
                print(
                    f"Skipping CauseNet partition: {len(all_causenet)} files found "
                    f"(need > {VAL_CAUSENET_COUNT})."
                )
    else:
        print(f"Skipping CauseNet partition: {CAUSENET_DIR} does not exist.")

    print("--- Done ---")
    print(f"Train data remains in: {BNLEARN_DIR}, {CONCEPTNET_DIR}, and {CAUSENET_DIR}")
    print(f"Val data moved to:     {VAL_DIR}")
    print(f"Test data moved to:    {TEST_DIR}")


if __name__ == "__main__":
    partition()
