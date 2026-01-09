from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from model import SemanticCausalFoundationModel, SCFMConfig
from train import collate_graphs, mask_logits_and_labels


class TestDataset(Dataset):
    """Held-out test set reading only *_orig.json graphs."""

    def __init__(self, root: Path):
        root = Path(root)
        bnlearn_dir = root / "bnlearn"
        self.files = sorted(bnlearn_dir.glob("*_orig.json"))
        print(f"Found {len(self.files)} held-out original graphs in {bnlearn_dir}.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        data = json.loads(path.read_text())

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        node_texts: List[str] = []
        id_to_idx: Dict[str, int] = {}
        for i, n in enumerate(nodes):
            nid = n["id"] if isinstance(n, dict) else n
            id_to_idx[nid] = i
            if isinstance(n, dict):
                text = n.get("name") or str(nid)
            else:
                text = str(n)
            node_texts.append(text)

        num_nodes = len(node_texts)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        for e in edges:
            if isinstance(e, dict):
                src = id_to_idx.get(e.get("source"))
                tgt = id_to_idx.get(e.get("target"))
            elif isinstance(e, (list, tuple)) and len(e) == 2:
                src = id_to_idx.get(e[0])
                tgt = id_to_idx.get(e[1])
            else:
                src = tgt = None
            if src is not None and tgt is not None:
                adj[src, tgt] = 1.0

        return {"node_texts": node_texts, "adj": adj, "num_nodes": num_nodes, "name": path.stem}


def evaluate_checkpoint(checkpoint: Path, data_dir: Path, batch_size: int = 1, device: str = "auto"):
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    cfg = SCFMConfig(device=dev)
    model = SemanticCausalFoundationModel(cfg)
    state_dict = torch.load(checkpoint, map_location=dev)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    ds = TestDataset(data_dir)
    if len(ds) == 0:
        print("No held-out test graphs found. Did you run partition_dataset.py?")
        return

    def _collate(batch):
        names = [item["name"] for item in batch]
        collated = collate_graphs(batch)
        collated["names"] = names
        return collated

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["adj"].to(dev)
            node_texts = batch["node_texts"]

            logits, pad_mask = model(node_texts)
            pad_mask = pad_mask.to(dev)
            valid_logits, valid_labels = mask_logits_and_labels(logits, labels, pad_mask)

            if valid_logits.numel() == 0:
                print("Skipping empty batch (no valid edges).")
                continue

            probs = torch.sigmoid(valid_logits)
            preds = (probs > 0.5).long()

            p, r, f1, _ = precision_recall_fscore_support(
                valid_labels.cpu(), preds.cpu(), average="binary", zero_division=0
            )
            name = batch["names"][0] if batch.get("names") else "graph"
            print(f"Graph: {name} | P={p:.3f} R={r:.3f} F1={f1:.3f}")

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(valid_labels.cpu().tolist())

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    print("-" * 40)
    print("HELD-OUT TEST SUMMARY")
    print(f"Precision: {macro_p:.4f}")
    print(f"Recall:    {macro_r:.4f}")
    print(f"F1:        {macro_f1:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate held-out test set")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint to evaluate")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset_heldout_test"), help="Held-out dataset root")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.data_dir, batch_size=args.batch_size, device=args.device)
