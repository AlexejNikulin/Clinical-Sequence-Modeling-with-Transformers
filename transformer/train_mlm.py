# transformer/train_mlm.py
"""
Minimal MLM training script for the compact transformer encoder.

Goal:
- Train CompactTransformerEncoder with BERT-style Masked Language Modeling (MLM)
- Works with tokenized sequences (input_ids, attention_mask) and optional event_type_ids
- Provides console + CSV logging for quick debugging and stability monitoring

Implements:
- Dataset + DataLoader for pre-tokenized sequences
- BERT-style MLM masking (80/10/10 rule)
- Training loop with AdamW
- Console + CSV logging
- Synthetic sanity run (default) to verify end-to-end correctness
"""

from __future__ import annotations

import os
import csv
import math
import time
import random
from dataclasses import asdict
from typing import Dict, Optional, List, Any

import torch
from torch.utils.data import Dataset, DataLoader

# ✅ LOCAL imports (work with: cd transformer && python train_mlm.py)
from compact_transformer_encoder import (
    CompactTransformerConfig,
    CompactTransformerEncoder,
)
from mlm_masking import mlm_mask_801010


# -------------------------------------------------
# Reproducibility (optional but recommended for debugging)
# -------------------------------------------------
def set_seed(seed: int = 0) -> None:
    """
    Sets random seeds for reproducible debugging runs.
    Note: Full determinism on GPU can be slower / not always possible depending on ops.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class SequenceDataset(Dataset):
    """
    Dataset wrapper for already-tokenized sequences.

    Each item represents one sequence:
      - input_ids:       [L]
      - attention_mask:  [L]  (1=real token, 0=PAD)
      - event_type_ids:  [L]  optional (heterogeneity signal)
    """

    def __init__(
        self,
        input_ids: List[List[int]],
        attention_mask: List[List[int]],
        event_type_ids: Optional[List[List[int]]] = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.event_type_ids = event_type_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        if self.event_type_ids is not None:
            batch["event_type_ids"] = torch.tensor(
                self.event_type_ids[idx], dtype=torch.long
            )
        return batch


def make_dataloader(
    input_ids: List[List[int]],
    attention_mask: List[List[int]],
    event_type_ids: Optional[List[List[int]]] = None,
    batch_size: int = 16,
) -> DataLoader:
    """
    Creates a shuffled DataLoader for training.

    drop_last=True:
    - keeps batch sizes constant (more stable training & logging)
    """
    ds = SequenceDataset(input_ids, attention_mask, event_type_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# -------------------------------------------------
# MLM batch preparation
# -------------------------------------------------
def make_mlm_batch(
    batch: Dict[str, torch.Tensor],
    *,
    vocab,
    vocab_size: int,
    p_mlm: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Applies BERT-style MLM masking to a batch.

    Masking logic (80/10/10):
    - 80% -> replace selected tokens with [MASK]
    - 10% -> replace with random token ID
    - 10% -> keep original token
    Labels contain original token IDs at masked positions, and -100 elsewhere (ignore_index).

    Returns dict with:
      - input_ids: masked input IDs   [B, L]
      - attention_mask:              [B, L]
      - labels:                      [B, L]  (masked positions hold target IDs, else -100)
      - event_type_ids (optional):   [B, L]
    """

    input_ids = batch["input_ids"].to(device)                 # [B, L]
    attention_mask = batch["attention_mask"].to(device)       # [B, L]

    # Token IDs from the project's vocab interface
    pad_id = vocab.token_to_id(vocab.get_padding_token())
    mask_id = vocab.token_to_id(vocab.get_masking_token())
    unk_id = vocab.token_to_id(vocab.get_unknown_token())

    masked_input_ids, labels = mlm_mask_801010(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_token_id=mask_id,
        vocab_size=vocab_size,
        p_mlm=p_mlm,
        pad_token_id=pad_id,
        # Never mask PAD/MASK/UNK to avoid trivial or degenerate training signal
        never_mask_token_ids=[pad_id, mask_id, unk_id],
    )

    out: Dict[str, torch.Tensor] = {
        "input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Optional heterogeneity: only attach event_type_ids if present in incoming batch
    if "event_type_ids" in batch:
        out["event_type_ids"] = batch["event_type_ids"].to(device)

    return out


# -------------------------------------------------
# Training utilities
# -------------------------------------------------
def global_grad_norm(model: torch.nn.Module) -> float:
    """
    Computes global L2 gradient norm across all parameters.
    Useful to detect exploding / vanishing gradients.
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.data.norm(2).item() ** 2)
    return math.sqrt(total)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_log_path(base_dir: str = "logs", prefix: str = "train_log") -> str:
    """
    Produces a run-specific CSV log file path, e.g. logs/train_log_20260112_203001.csv
    This prevents mixing logs from multiple runs.
    """
    ensure_dir(base_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{stamp}.csv")


# -------------------------------------------------
# Training loop
# -------------------------------------------------
def train_mlm(
    *,
    vocab,
    input_ids: List[List[int]],
    attention_mask: List[List[int]],
    event_type_ids: Optional[List[List[int]]],
    cfg: CompactTransformerConfig,
    steps: int,
    batch_size: int,
    lr: float,
    p_mlm: float,
    log_every: int = 10,
    seed: int = 0,
):

    # Reproducibility for debugging
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = make_dataloader(
        input_ids,
        attention_mask,
        event_type_ids if cfg.use_event_type_embeddings else None,
        batch_size,
    )

    model = CompactTransformerEncoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()

    # Run-specific log file
    log_path = make_run_log_path(base_dir="logs", prefix="train_log")

    # Add n_masked + masked_ratio for MLM sanity checks
    fieldnames = ["step", "loss", "grad_norm", "logits_max", "n_masked", "masked_ratio"]

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print("=== Training config ===")
    print(asdict(cfg))
    print(f"device={device}")
    print(f"log_path={log_path}")
    print("=======================\n")

    step = 0
    data_iter = iter(loader)

    while step < steps:
        # We cycle the dataloader indefinitely until we reach "steps"
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        mlm_batch = make_mlm_batch(
            batch,
            vocab=vocab,
            vocab_size=cfg.vocab_size,
            p_mlm=p_mlm,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        out = model(
            input_ids=mlm_batch["input_ids"],
            attention_mask=mlm_batch["attention_mask"],
            event_type_ids=mlm_batch.get("event_type_ids"),
            labels=mlm_batch["labels"],
        )

        loss = out["loss"]
        loss.backward()

        # ✅ Measure grad norm after backward and before optimizer step
        grad_norm = global_grad_norm(model)

        optimizer.step()

        # Logits magnitude is a coarse stability check (helps detect explosions)
        logits_max = out["logits"].abs().max().item()

        # MLM supervision density: how many tokens actually contribute to loss?
        labels = mlm_batch["labels"]
        n_masked = (labels != -100).sum().item()
        n_valid = (mlm_batch["attention_mask"] == 1).sum().item()
        masked_ratio = (n_masked / max(1, n_valid))

        if step % log_every == 0:
            print(
                f"[step {step:4d}] "
                f"loss={loss.item():.4f} "
                f"grad_norm={grad_norm:.3f} "
                f"logits_max={logits_max:.3f} "
                f"n_masked={n_masked} "
                f"masked_ratio={masked_ratio:.3f}"
            )

            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(
                    {
                        "step": step,
                        "loss": loss.item(),
                        "grad_norm": grad_norm,
                        "logits_max": logits_max,
                        "n_masked": n_masked,
                        "masked_ratio": masked_ratio,
                    }
                )

        step += 1

    print("\nTraining finished.")
    print(f"Logs written to {log_path}")


# -------------------------------------------------
# Synthetic data + main (sanity run)
# -------------------------------------------------
def make_synth_data(
    *,
    n_seq: int,
    max_len: int,
    vocab_size: int,
    pad_id: int,
    n_event_types: int,
):
    """
    Creates synthetic toy sequences for sanity checking.

    Tokens:
      - Use IDs from [3..vocab_size-1] to avoid PAD/MASK/UNK (0/1/2) collisions.
    Padding:
      - padded with pad_id up to max_len
      - attention_mask=1 for real tokens, 0 for PAD
    """
    rng = random.Random(0)
    input_ids, attn, event_types = [], [], []

    for _ in range(n_seq):
        L = rng.randint(max_len // 4, max_len)

        # avoid special IDs: assume 0=[PAD],1=[MASK],2=[UNK]
        seq = [rng.randint(3, vocab_size - 1) for _ in range(L)]
        et = [rng.randint(0, n_event_types - 1) for _ in range(L)]
        mask = [1] * L

        while len(seq) < max_len:
            seq.append(pad_id)
            et.append(0)
            mask.append(0)

        input_ids.append(seq)
        attn.append(mask)
        event_types.append(et)

    return input_ids, attn, event_types


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--p_mlm", type=float, default=0.15)
    parser.add_argument("--use_event_types", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vocab_size = 70000

    # Minimal vocab for synthetic runs (replace with your real vocab in real training)
    class MiniVocab:
        def get_padding_token(self): return "[PAD]"
        def get_masking_token(self): return "[MASK]"
        def get_unknown_token(self): return "[UNK]"
        def token_to_id(self, tok: str) -> int:
            return {"[PAD]": 0, "[MASK]": 1, "[UNK]": 2}.get(tok, 2)

    vocab = MiniVocab()

    cfg = CompactTransformerConfig(
        vocab_size=vocab_size,
        max_len=args.max_len,
        d_model=192,
        n_layers=3,
        n_heads=6,
        use_event_type_embeddings=args.use_event_types,
        n_event_types=6,
        pad_token_id=0,
        mask_token_id=1,
    )

    input_ids, attn, event_types = make_synth_data(
        n_seq=800,
        max_len=args.max_len,
        vocab_size=vocab_size,
        pad_id=cfg.pad_token_id,
        n_event_types=cfg.n_event_types,
    )

    train_mlm(
        vocab=vocab,
        input_ids=input_ids,
        attention_mask=attn,
        event_type_ids=event_types if cfg.use_event_type_embeddings else None,
        cfg=cfg,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        p_mlm=args.p_mlm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
