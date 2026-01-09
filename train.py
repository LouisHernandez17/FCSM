from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from model import SemanticCausalFoundationModel, SCFMConfig


@dataclass
class TrainConfig:
    data_dir: Path
    model_name: str = "google/embeddinggemma-300m"
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 5
    max_nodes: int | None = None
    device: str = "auto"
    pos_weight: float = 5.0
    num_workers: int = 2
    seed: int = 17
    bnlearn_oversample: int = 1
    resume_from: Path | None = None
    freeze_encoder: bool = False
    log_file: Path | None = None
    plot_file: Path | None = None
    log_dir: Path | None = None
    log_interval: int = 0  # 0 = only per epoch
    checkpoint_dir: Path = Path("checkpoints")


class CausalGraphDataset(Dataset):
    def __init__(self, root_dir: Path, split: str, bnlearn_oversample: int = 1):
        self.files: List[Path] = []
        root = Path(root_dir)

        if split == "train":
            conceptnet_files = list((root / "conceptnet").glob("*.json"))
            causenet_files = list((root / "causenet").glob("*.json"))
            bnlearn_files = list((root / "bnlearn").glob("*.json"))

            self.files.extend(conceptnet_files)
            self.files.extend(causenet_files)
            if bnlearn_files:
                self.files.extend(bnlearn_files * max(1, bnlearn_oversample))
                print(
                    f"[train] Upsampling: {len(bnlearn_files)} BNLearn x{max(1, bnlearn_oversample)} | "
                    f"Mix: ConceptNet={len(conceptnet_files)} "
                    f"CauseNet={len(causenet_files)} "
                    f"BNLearnEff={len(bnlearn_files) * max(1, bnlearn_oversample)}"
                )
            else:
                print("[train] Warning: No BNLearn training files found (check partition?)")

        elif split == "val":
            val_root = Path("dataset_heldout_val")
            bnlearn_val = list((val_root / "bnlearn").glob("*.json"))
            causenet_val = list((val_root / "causenet").glob("*.json"))
            cn_val = list((val_root / "conceptnet").glob("*.json"))
            self.files.extend(bnlearn_val)
            self.files.extend(causenet_val)
            self.files.extend(cn_val)
            if not self.files:
                print(f"[val] Warning: No validation files found in {val_root}")
            else:
                print(
                    f"[val] Found {len(bnlearn_val)} BNLearn, "
                    f"{len(causenet_val)} CauseNet, "
                    f"and {len(cn_val)} ConceptNet graphs."
                )
        else:
            raise ValueError(f"Unknown split {split}")

        self.files = sorted(self.files)
        print(f"[{split}] Loaded {len(self.files)} graphs from {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        data = json.loads(path.read_text())

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        node_texts = []
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

        return {"node_texts": node_texts, "adj": adj, "num_nodes": num_nodes}


def collate_graphs(batch: List[Dict[str, Any]], max_nodes: int | None = None) -> Dict[str, Any]:
    max_len = max(item["num_nodes"] for item in batch)
    if max_nodes is not None:
        max_len = min(max_len, max_nodes)

    bsz = len(batch)
    padded_adj = torch.zeros((bsz, max_len, max_len), dtype=torch.float)
    pad_mask = torch.ones((bsz, max_len), dtype=torch.bool)
    texts: List[List[str]] = []

    for i, item in enumerate(batch):
        n = min(item["num_nodes"], max_len)
        padded_adj[i, :n, :n] = item["adj"][:n, :n]
        pad_mask[i, :n] = False
        texts.append(item["node_texts"][:n])

    return {"node_texts": texts, "adj": padded_adj, "pad_mask": pad_mask}


def mask_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor, pad_mask: torch.Tensor):
    # logits: (B, N, N), labels: (B, N, N), pad_mask: (B, N) True=pad
    b, n, _ = logits.shape
    valid_nodes = ~pad_mask
    node_mask = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)  # (B, N, N)
    diag = torch.eye(n, device=logits.device, dtype=torch.bool).unsqueeze(0)
    mask = node_mask & (~diag)
    return logits[mask], labels[mask]


def evaluate(model: SemanticCausalFoundationModel, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    preds_all: List[int] = []
    labels_all: List[int] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["adj"].to(device)
            node_texts = batch["node_texts"]
            logits, pad_mask = model(node_texts)
            pad_mask = pad_mask.to(device)
            valid_logits, valid_labels = mask_logits_and_labels(logits, labels, pad_mask)
            if valid_logits.numel() == 0:
                continue

            loss = criterion(valid_logits, valid_labels)
            total_loss += loss.item()

            probs = torch.sigmoid(valid_logits)
            preds = (probs > 0.5).long().cpu().tolist()
            labels_all.extend(valid_labels.long().cpu().tolist())
            preds_all.extend(preds)

    if len(preds_all) == 0:
        return float("nan"), 0.0, 0.0, 0.0
    p, r, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="binary", zero_division=0)
    return total_loss / max(1, len(loader)), p, r, f1


def compute_prf(logits: torch.Tensor, labels: torch.Tensor, pad_mask: torch.Tensor):
    valid_logits, valid_labels = mask_logits_and_labels(logits, labels, pad_mask)
    probs = torch.sigmoid(valid_logits)
    preds = (probs > 0.5).long().cpu().tolist()
    labels_all = valid_labels.long().cpu().tolist()
    if len(preds) == 0:
        return 0.0, 0.0, 0.0
    p, r, f1, _ = precision_recall_fscore_support(labels_all, preds, average="binary", zero_division=0)
    return p, r, f1


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else (cfg.device if cfg.device != "auto" else "cpu"))

    model_cfg = SCFMConfig(embedding_model=cfg.model_name, device=device)
    model = SemanticCausalFoundationModel(model_cfg)

    if cfg.resume_from:
        state_dict = torch.load(cfg.resume_from, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[resume] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[resume] Unexpected keys: {len(unexpected)}")

    if cfg.freeze_encoder:
        for p in model.set_encoder.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0.1,
    )
    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = CausalGraphDataset(cfg.data_dir, split="train", bnlearn_oversample=cfg.bnlearn_oversample)
    val_ds = CausalGraphDataset(cfg.data_dir, split="val")

    collate_fn = lambda batch: collate_graphs(batch, cfg.max_nodes)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    print(f"Starting training on {device} for {cfg.epochs} epochs")

    history = []
    writer = None
    if cfg.log_dir:
        from torch.utils.tensorboard import SummaryWriter
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(cfg.log_dir))

    global_step = 0
    step_preds: List[int] = []
    step_labels: List[int] = []
    step_loss = 0.0

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = float("-inf")

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        train_preds = []
        train_labels = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in loop:
            labels = batch["adj"].to(device)
            node_texts = batch["node_texts"]

            logits, pad_mask = model(node_texts)
            pad_mask = pad_mask.to(device)
            valid_logits, valid_labels = mask_logits_and_labels(logits, labels, pad_mask)

            if valid_logits.numel() == 0:
                continue

            loss = criterion(valid_logits, valid_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            train_labels.extend(valid_labels.long().cpu().tolist())
            train_preds.extend((torch.sigmoid(valid_logits) > 0.5).long().cpu().tolist())
            step_labels.extend(valid_labels.long().cpu().tolist())
            step_preds.extend((torch.sigmoid(valid_logits) > 0.5).long().cpu().tolist())
            step_loss += loss.item()
            global_step += 1
            loop.set_postfix(loss=loss.item())

            if cfg.log_interval and global_step % cfg.log_interval == 0:
                if len(step_preds) > 0:
                    p_step, r_step, f1_step, _ = precision_recall_fscore_support(step_labels, step_preds, average="binary", zero_division=0)
                else:
                    p_step = r_step = f1_step = 0.0
                avg_step_loss = step_loss / cfg.log_interval
                tqdm.write(
                    f"Step {global_step}: loss={avg_step_loss:.4f} P={p_step:.3f} R={r_step:.3f} F1={f1_step:.3f}"
                )
                if writer:
                    writer.add_scalar("step/loss", avg_step_loss, global_step)
                    writer.add_scalar("step/precision", p_step, global_step)
                    writer.add_scalar("step/recall", r_step, global_step)
                    writer.add_scalar("step/f1", f1_step, global_step)
                step_preds.clear()
                step_labels.clear()
                step_loss = 0.0

        if len(train_preds) > 0:
            p_train, r_train, f1_train, _ = precision_recall_fscore_support(train_labels, train_preds, average="binary", zero_division=0)
        else:
            p_train = r_train = f1_train = 0.0

        val_loss, p, r, f1 = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}: train_loss={running/max(1,len(train_loader)):.4f} "
            f"train_P={p_train:.3f} train_R={r_train:.3f} train_F1={f1_train:.3f} "
            f"val_loss={val_loss:.4f} val_P={p:.3f} val_R={r:.3f} val_F1={f1:.3f}"
        )

        ckpt_state = {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("featurizer.encoder.")
        }
        ckpt_path = cfg.checkpoint_dir / f"scfm_epoch_{epoch+1}.pt"
        torch.save(ckpt_state, ckpt_path)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(ckpt_state, cfg.checkpoint_dir / "best_model.pt")

        history.append({
            "epoch": epoch + 1,
            "train_loss": running / max(1, len(train_loader)),
            "train_precision": p_train,
            "train_recall": r_train,
            "train_f1": f1_train,
            "val_loss": val_loss,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

        if writer:
            # Epoch-aligned scalars
            writer.add_scalar("loss/train", history[-1]["train_loss"], epoch + 1)
            writer.add_scalar("loss/val", val_loss, epoch + 1)
            writer.add_scalar("metrics/train_precision", p_train, epoch + 1)
            writer.add_scalar("metrics/train_recall", r_train, epoch + 1)
            writer.add_scalar("metrics/train_f1", f1_train, epoch + 1)
            writer.add_scalar("metrics/precision", p, epoch + 1)
            writer.add_scalar("metrics/recall", r, epoch + 1)
            writer.add_scalar("metrics/f1", f1, epoch + 1)

            # Step-aligned scalars so validation curves show up alongside step charts
            writer.add_scalar("loss/val_step", val_loss, global_step)
            writer.add_scalar("metrics/val_precision_step", p, global_step)
            writer.add_scalar("metrics/val_recall_step", r, global_step)
            writer.add_scalar("metrics/val_f1_step", f1, global_step)
            writer.flush()

    # Persist history
    if cfg.log_file:
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        import json
        cfg.log_file.write_text(json.dumps(history, indent=2))

    # Optional plot
    if cfg.plot_file:
        import matplotlib.pyplot as plt
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        precision = [h["precision"] for h in history]
        recall = [h["recall"] for h in history]
        f1 = [h["f1"] for h in history]

        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs[0, 0].plot(epochs, train_loss, label="train_loss")
        axs[0, 0].plot(epochs, val_loss, label="val_loss")
        axs[0, 0].set_title("Loss")
        axs[0, 0].legend()

        axs[0, 1].plot(epochs, precision, label="val_P")
        axs[0, 1].plot(epochs, recall, label="val_R")
        axs[0, 1].plot(epochs, f1, label="val_F1")
        axs[0, 1].plot(epochs, [h["train_precision"] for h in history], label="train_P", linestyle="--")
        axs[0, 1].plot(epochs, [h["train_recall"] for h in history], label="train_R", linestyle="--")
        axs[0, 1].plot(epochs, [h["train_f1"] for h in history], label="train_F1", linestyle="--")
        axs[0, 1].set_title("PRF (train vs val)")
        axs[0, 1].legend()

        axs[1, 0].axis("off")
        axs[1, 1].axis("off")

        fig.tight_layout()
        cfg.plot_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.plot_file)
        plt.close(fig)

    if writer:
        writer.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Semantic Causal Foundation Model")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-name", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bnlearn-oversample", type=int, default=1, help="Repeat BNLearn augmented files this many times in training")
    parser.add_argument("--resume-from", type=Path, help="Checkpoint path to resume from")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze set encoder for fine-tuning edge predictor only")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--plot-file", type=Path)
    parser.add_argument("--log-dir", type=Path, help="TensorBoard log directory")
    parser.add_argument("--log-interval", type=int, default=0, help="Steps between mid-epoch train metric logs (0=only epoch)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory to store model checkpoints")
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        max_nodes=args.max_nodes,
        device=args.device,
        pos_weight=args.pos_weight,
        num_workers=args.num_workers,
        seed=args.seed,
        bnlearn_oversample=args.bnlearn_oversample,
        resume_from=args.resume_from,
        freeze_encoder=args.freeze_encoder,
        log_file=args.log_file,
        plot_file=args.plot_file,
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
