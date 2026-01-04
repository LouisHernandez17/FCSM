from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class SCFMConfig:
    embedding_model: str = "google/embeddinggemma-300m"
    hidden_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 4
    dropout: float = 0.4
    device: str | torch.device = "auto"  # "auto" -> cuda (ROCm OK) or mps, else cpu


class SemanticFeaturizer(nn.Module):
    """Frozen text encoder + mean pooling and projection."""

    def __init__(self, cfg: SCFMConfig):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model)
        self.encoder = AutoModel.from_pretrained(cfg.embedding_model)
        for p in self.encoder.parameters():
            p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, cfg.hidden_dim)
        self.embed_dropout = nn.Dropout(0.4)

    @torch.no_grad()
    def encode(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        outputs = self.encoder(**tokens)
        hidden = outputs.last_hidden_state  # (B, T, H)
        mask = tokens.attention_mask.unsqueeze(-1)  # (B, T, 1)
        masked = hidden * mask
        summed = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_pooled = summed / lengths
        return mean_pooled, tokens.attention_mask

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            pooled, _ = self.encode(texts, device)
        return self.proj(self.embed_dropout(pooled))


class SetTransformer(nn.Module):
    def __init__(self, cfg: SCFMConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=pad_mask)


class EdgePredictor(nn.Module):
    def __init__(self, hidden_dim: int, bias_init: float = -2.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.tensor(bias_init))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Bilinear score for every (i,j): h_i^T W h_j + b
        h: (B, N, D), pad_mask: (B, N) True where padding
        """
        B, N, D = h.shape
        logits = torch.einsum("b i d, d k, b j k -> b i j", h, self.weight, h) + self.bias

        # mask self loops
        diag = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(diag, float("-inf"))

        # mask padding rows/cols
        if pad_mask.any():
            row_mask = pad_mask.unsqueeze(2).expand(-1, -1, N)
            col_mask = pad_mask.unsqueeze(1).expand(-1, N, -1)
            pad_combined = row_mask | col_mask
            logits = logits.masked_fill(pad_combined, float("-inf"))
        return logits


class SemanticCausalFoundationModel(nn.Module):
    def __init__(self, cfg: SCFMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")  # covers NVIDIA and ROCm builds
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)
        self.featurizer = SemanticFeaturizer(cfg)
        self.set_encoder = SetTransformer(cfg)
        self.edge_predictor = EdgePredictor(cfg.hidden_dim)
        self.to(self.device)

    def forward(self, graphs: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute edge logits for a batch of graphs.

        Args:
            graphs: list of graphs, each graph is a list of node descriptions (strings)
        Returns:
            logits: (B, N, N) adjacency logits with -inf on pads/diagonal
            pad_mask: (B, N) boolean padding mask
        """
        device = self.device
        batch_size = len(graphs)
        max_nodes = max(len(g) for g in graphs)
        all_texts = [t for g in graphs for t in g]

        node_embs = self.featurizer(all_texts, device)  # (total_nodes, hidden)

        # split back per graph
        splits = [len(g) for g in graphs]
        embeds_per_graph = torch.split(node_embs, splits)

        padded = torch.zeros(batch_size, max_nodes, node_embs.size(-1), device=device)
        pad_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)
        for i, emb in enumerate(embeds_per_graph):
            n = emb.size(0)
            padded[i, :n] = emb
            pad_mask[i, :n] = False

        contextual = self.set_encoder(padded, pad_mask)
        logits = self.edge_predictor(contextual, pad_mask)
        return logits, pad_mask


def _demo():
    cfg = SCFMConfig(device="cpu")
    model = SemanticCausalFoundationModel(cfg)
    graphs = [
        ["Rain", "Wet ground", "Slip"],
        ["Smoke", "Fire"],
    ]
    logits, pad = model(graphs)
    print("logits shape", logits.shape)
    print("mask", pad)


if __name__ == "__main__":
    _demo()
