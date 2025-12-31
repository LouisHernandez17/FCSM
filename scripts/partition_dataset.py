import shutil
from pathlib import Path
from typing import List

# Config
DATASET_DIR = Path("dataset")
TIER3_DIR = DATASET_DIR / "tier3_gold"

# TEST SET (final exam)
TEST_DIR = Path("dataset_heldout_test")
TEST_NETWORKS = ["survey", "insurance", "sachs"]

# VALIDATION SET (midterm)
VAL_DIR = Path("dataset_heldout_val")
VAL_NETWORKS = ["alarm", "child"]


def is_network_file(filename: str, net_ids: List[str]) -> bool:
    # Handles gold_<net>_* patterns and raw <net> names
    for nid in net_ids:
        if filename == f"{nid}.json" or filename.startswith(f"{nid}_"):
            return True
        if filename.startswith(f"gold_{nid}"):
            return True
    return False


def move_files(source_dir: Path, dest_root: Path, target_networks: List[str]):
    dest_tier3 = dest_root / "tier3_gold"
    dest_tier3.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    for file_path in source_dir.glob("*.json"):
        if is_network_file(file_path.name, target_networks):
            shutil.move(str(file_path), str(dest_tier3 / file_path.name))
            moved_count += 1

    print(f"  Moved {moved_count} files to {dest_tier3}")


def partition():
    if not TIER3_DIR.exists():
        print(f"Error: {TIER3_DIR} does not exist. Build data first.")
        return

    print("--- Partitioning Dataset ---")

    print(f"Partitioning TEST set: {TEST_NETWORKS}...")
    move_files(TIER3_DIR, TEST_DIR, TEST_NETWORKS)

    print(f"Partitioning VALIDATION set: {VAL_NETWORKS}...")
    move_files(TIER3_DIR, VAL_DIR, VAL_NETWORKS)

    print("--- Done ---")
    print(f"Train data remains in: {TIER3_DIR}")
    print(f"Val data moved to:     {VAL_DIR}")
    print(f"Test data moved to:    {TEST_DIR}")


if __name__ == "__main__":
    partition()
