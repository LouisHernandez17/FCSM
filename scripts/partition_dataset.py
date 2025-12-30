import shutil
from pathlib import Path
from typing import Iterable

DATASET_DIR = Path("dataset")
TEST_DIR = Path("dataset_heldout_test")
TIER3_DIR = DATASET_DIR / "tier3_gold"
TEST_NETWORKS = ["sachs", "survey", "insurance"]


def is_test_file(path: Path, test_ids: Iterable[str]) -> bool:
    name = path.name
    for net_id in test_ids:
        if name == f"{net_id}.json" or name.startswith(f"{net_id}_"):
            return True
    return False


def partition():
    if not TIER3_DIR.exists():
        print(f"Error: {TIER3_DIR} does not exist. Build the dataset first.")
        return

    dest_dir = TEST_DIR / "tier3_gold"
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for file_path in TIER3_DIR.glob("*.json"):
        if not is_test_file(file_path, TEST_NETWORKS):
            continue
        dest = dest_dir / file_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(dest))
        moved += 1

    print("Partition complete.")
    print(f"  Moved {moved} files to '{dest_dir}'.")
    if moved == 0:
        print("  No matching test networks were found; nothing was moved.")
    else:
        print("  These files will be excluded from training/validation and can be used for held-out testing.")


if __name__ == "__main__":
    partition()
