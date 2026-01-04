import json
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

DATASET_DIR = Path("dataset/tier2_conceptnet")
OUTPUT_FILE = Path("dataset/node_dictionary.json")

LM_CLIENT = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)
MODEL_ID = "nvidia/nemotron-3-nano"


def get_all_unique_nodes() -> list[str]:
    """Scan dataset directory for unique node names."""
    unique_nodes: set[str] = set()
    files = sorted(DATASET_DIR.glob("*.json"))
    if not files:
        print("No JSON graphs found in dataset/tier2_conceptnet. Run build_full_dataset.py first.")
        return []

    print(f"Scanning {len(files)} graphs in {DATASET_DIR} for concept names...")
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
            temperature=0.7,
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"Error defining '{term}': {exc}")
        return None


def main() -> None:
    nodes = get_all_unique_nodes()
    if not nodes:
        return

    existing: dict[str, str]
    if OUTPUT_FILE.exists():
        try:
            existing = json.loads(OUTPUT_FILE.read_text())
            print(f"Loaded {len(existing)} existing definitions from {OUTPUT_FILE}.")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read existing dictionary, starting fresh ({exc}).")
            existing = {}
    else:
        existing = {}

    todo = [name for name in nodes if name not in existing]
    print(f"Definitions remaining: {len(todo)}")
    if not todo:
        print("Dictionary already complete.")
        return

    print("Generating definitions via LMStudio...")
    try:
        for idx, term in enumerate(tqdm(todo)):
            definition = generate_definition(term)
            if definition:
                existing[term] = definition

            if idx % 50 == 0:
                OUTPUT_FILE.write_text(json.dumps(existing, indent=2))
    except KeyboardInterrupt:
        print("Interrupted, saving progress...")

    OUTPUT_FILE.write_text(json.dumps(existing, indent=2))
    print(f"Saved {len(existing)} definitions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
