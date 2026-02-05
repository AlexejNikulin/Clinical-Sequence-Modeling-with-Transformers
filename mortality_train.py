from __future__ import annotations

import os
import sys

# -------------------------------------------------
# Ensure repo root is importable so "transformer" works.
# This must run BEFORE "from transformer...." imports.
# Repo root = parent directory of this file (downstream/..)
# -------------------------------------------------
'''

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
'''

# Debug (optional): uncomment to verify path at runtime
# print("REPO_ROOT:", REPO_ROOT)
# print("sys.path[0]:", sys.path[0])
# print("Contents:", os.listdir(REPO_ROOT))

import argparse
import json
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder



"""
freeze_encoder --> linear probe style: 
        python downstream/mortality_train.py \
        --train_jsonl data/mortality_train.jsonl \
        --val_jsonl data/eval_val.jsonl \
        --ckpt checkpoints/mlm_span.pt \
        --freeze_encoder \
        --epochs 10

Fine-tune:
        python downstream/mortality_train.py \
        --train_jsonl data/mortality_train.jsonl \
        --val_jsonl data/mortality_val.jsonl \
        --ckpt checkpoints/mlm_span.pt \
        --fine_tune \
        --epochs 5



"""



# Make repo root importable so "transformer" works
# regardless of how this script is invoked.

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)





# Flattening utilities (same logic as MLM trainer)
def build_joint_sequences(
    *,
    demographics: List[List[int]],
    events: List[List[int]],
    max_len: int,
    pad_id: int,
    sep_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    if len(demographics) != len(events):
        raise ValueError("demographics/events length mismatch")

    input_ids_all, attn_all, seg_all = [], [], []

    for demo_tok, event_tok in zip(demographics, events):
        seq, seg = [], []

        seq.extend(demo_tok)
        seg.extend([0] * len(demo_tok))

        if sep_id is not None:
            seq.append(sep_id)
            seg.append(0)

        seq.extend(event_tok)
        seg.extend([1] * len(event_tok))

        seq = seq[:max_len]
        seg = seg[:max_len]
        attn = [1] * len(seq)

        while len(seq) < max_len:
            seq.append(pad_id)
            attn.append(0)
            seg.append(0)

        input_ids_all.append(seq)
        attn_all.append(attn)
        seg_all.append(seg)

    return input_ids_all, attn_all, seg_all



# Dataset
class MortalityDataset(Dataset):
    def __init__(self, input_ids, attention_mask, segment_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "event_type_ids": torch.tensor(self.segment_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }



# Model: encoder + pooling + risk head
class RiskHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)  # [B]


class MortalityRiskModel(nn.Module):
    def __init__(self, encoder: CompactTransformerEncoder, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = RiskHead(d_model=d_model, dropout=dropout)

    @staticmethod
    def pool_event_mean(hidden: torch.Tensor, attention_mask: torch.Tensor, event_type_ids: torch.Tensor) -> torch.Tensor:
        valid = (attention_mask == 1) & (event_type_ids == 1)
        valid_f = valid.unsqueeze(-1).float()
        denom = valid_f.sum(dim=1).clamp(min=1.0)
        return (hidden * valid_f).sum(dim=1) / denom

    def forward(self, input_ids, attention_mask, event_type_ids) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            event_type_ids=event_type_ids,
            labels=None,
            return_hidden=True,
        )
        hidden = out["hidden_states"]  # [B, L, D]
        pooled = self.pool_event_mean(hidden, attention_mask, event_type_ids)
        logits = self.head(pooled)     # [B]
        return logits



# Metrics (no sklearn)
def compute_auroc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.detach().cpu().float()
    y_score = y_score.detach().cpu().float()

    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]

    P = y_true.sum().item()
    N = (1.0 - y_true).sum().item()
    if P == 0 or N == 0:
        return float("nan")

    tps = torch.cumsum(y_true, dim=0)
    fps = torch.cumsum(1.0 - y_true, dim=0)

    tpr = tps / P
    fpr = fps / N

    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])

    return torch.trapz(tpr, fpr).item()


def compute_auprc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.detach().cpu().float()
    y_score = y_score.detach().cpu().float()

    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]

    P = y_true.sum().item()
    if P == 0:
        return float("nan")

    tps = torch.cumsum(y_true, dim=0)
    fps = torch.cumsum(1.0 - y_true, dim=0)

    precision = tps / (tps + fps).clamp(min=1e-12)
    recall = tps / P

    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])

    return torch.trapz(precision, recall).item()


@torch.no_grad()
def evaluate(model: MortalityRiskModel, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ys: List[torch.Tensor] = []
    ps: List[torch.Tensor] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        event_type_ids = batch["event_type_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, event_type_ids)
        probs = torch.sigmoid(logits)

        ys.append(labels)
        ps.append(probs)

    y_true = torch.cat(ys, dim=0)
    y_score = torch.cat(ps, dim=0)

    return compute_auroc(y_true, y_score), compute_auprc(y_true, y_score)



# Data loading
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_dataset_from_jsonl(
    records: List[Dict[str, Any]],
    *,
    max_len: int,
    pad_id: int,
    demo_key: str = "demo_tokens",
    event_key: str = "event_tokens",
    label_key: str = "label",
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[float]]:
    demographics = [r[demo_key] for r in records]
    events = [r[event_key] for r in records]
    labels = [float(r[label_key]) for r in records]

    input_ids, attn, seg = build_joint_sequences(
        demographics=demographics,
        events=events,
        max_len=max_len,
        pad_id=pad_id,
        sep_id=None,
    )
    return input_ids, attn, seg, labels



# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)

    p.add_argument("--ckpt", type=str, required=True, help="Path to pretrained MLM checkpoint (.pt)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--freeze_encoder", action="store_true", help="Linear probe mode (freeze encoder)")
    p.add_argument("--fine_tune", action="store_true", help="Fine-tune encoder (otherwise default is freeze)")

    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--pad_id", type=int, default=0)

    p.add_argument("--pos_weight", type=float, default=1.0)

    return p.parse_args()



# Main
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "cfg" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain keys: cfg, model_state_dict")

    cfg = CompactTransformerConfig(**ckpt["cfg"])

    if args.max_len > cfg.max_len:
        raise ValueError(
            f"--max_len={args.max_len} exceeds encoder cfg.max_len={cfg.max_len}. "
            f"Either retrain with larger max_len or reduce downstream max_len."
        )

    encoder = CompactTransformerEncoder(cfg)
    encoder.load_state_dict(ckpt["model_state_dict"], strict=True)

    model = MortalityRiskModel(encoder=encoder, d_model=cfg.d_model, dropout=cfg.dropout).to(device)

    train_records = load_jsonl(args.train_jsonl)
    val_records = load_jsonl(args.val_jsonl)

    tr_in, tr_attn, tr_seg, tr_y = build_dataset_from_jsonl(
        train_records, max_len=args.max_len, pad_id=args.pad_id
    )
    va_in, va_attn, va_seg, va_y = build_dataset_from_jsonl(
        val_records, max_len=args.max_len, pad_id=args.pad_id
    )

    train_ds = MortalityDataset(tr_in, tr_attn, tr_seg, tr_y)
    val_ds = MortalityDataset(va_in, va_attn, va_seg, va_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # default behavior: freeze unless --fine_tune is set
    freeze = args.freeze_encoder or (not args.fine_tune)
    if freeze:
        for p in model.encoder.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print(f"device={device}")
    print(f"freeze_encoder={freeze}")
    print(f"train_size={len(train_ds)} val_size={len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            event_type_ids = batch["event_type_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask, event_type_ids)

            if args.pos_weight != 1.0:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=torch.tensor(args.pos_weight, device=device)
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            n += input_ids.size(0)

        avg_loss = total_loss / max(1, n)
        auroc, auprc = evaluate(model, val_loader, device=device)

        print(f"[epoch {epoch:02d}] loss={avg_loss:.4f} val_AUROC={auroc:.4f} val_AUPRC={auprc:.4f}")


if __name__ == "__main__":
    main()
