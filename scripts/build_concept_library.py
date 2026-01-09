import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

DATASET_DIR = Path("dataset/conceptnet")
OUTPUT_FILE = Path("dataset/node_dictionary.json")

LM_CLIENT = OpenAI(
    base_url="http://13.218.97.64:8000/v1",
    api_key="dummy",
)
MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking-FP8"


def get_all_unique_nodes() -> list[str]:
    """Scan dataset directory for unique node names."""
    unique_nodes: set[str] = set()
    files = sorted(DATASET_DIR.glob("*.json"))
    if not files:
        print(f"No JSON graphs found in {DATASET_DIR}. Run build_full_dataset.py first.")
        return []

    print(
        f"Scanning {len(files)} graphs in {DATASET_DIR} for concept names...")
    for path in tqdm(files):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {path.name}: {exc}")
            continue

        for node in data.get("nodes", []):
            if isinstance(node, dict):
                name = node.get("name") or node.get("id")
            else:
                name = str(node)
            if name:
                unique_nodes.add(name)

    print(f"Found {len(unique_nodes)} unique concepts to define.")
    return sorted(unique_nodes)


def generate_definition(term: str) -> str | None:
    """Request a concise definition for a concept."""
    try:
        response = LM_CLIENT.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise dictionary. Reply with only the definition, under 15 words.",
                },
                {
                    "role": "user",
                    "content": (
                        "Define the concept '" + term + "' for commonsense causal reasoning."
                    ),
                },
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"Error defining '{term}': {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate concise definitions via LM Studio.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("LM_WORKERS", "4")),
        help="Number of parallel LM requests (default: 4).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Write progress every N completed definitions (0 disables).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of concepts to define (0 = all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = get_all_unique_nodes()
    if not nodes:
        return

    existing: dict[str, str]
    if OUTPUT_FILE.exists():
        try:
            existing = json.loads(OUTPUT_FILE.read_text())
            print(
                f"Loaded {len(existing)} existing definitions from {OUTPUT_FILE}.")
        except Exception as exc:  # noqa: BLE001
            print(
                f"Could not read existing dictionary, starting fresh ({exc}).")
            existing = {}
    else:
        existing = {}

    todo = [name for name in nodes if name not in existing]
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]
    print(f"Definitions remaining: {len(todo)}")
    if not todo:
        print("Dictionary already complete.")
        return

    workers = max(1, args.workers)
    print(f"Generating definitions via LMStudio with {workers} worker(s)...")
    try:
        save_every = args.save_every
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_term = {executor.submit(
                generate_definition, term): term for term in todo}
            for idx, future in enumerate(
                tqdm(as_completed(future_to_term), total=len(todo))
            ):
                term = future_to_term[future]
                definition = future.result()
                if definition:
                    existing[term] = definition

                if save_every > 0 and (idx + 1) % save_every == 0:
                    OUTPUT_FILE.write_text(json.dumps(existing, indent=2))
    except KeyboardInterrupt:
        print("Interrupted, saving progress...")

    OUTPUT_FILE.write_text(json.dumps(existing, indent=2))
    print(f"Saved {len(existing)} definitions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
