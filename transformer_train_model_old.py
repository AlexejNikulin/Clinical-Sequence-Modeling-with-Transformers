# transformer/transformer_train_model.py
"""
Transformer training script (MLM) for CompactTransformerEncoder
supporting the EXACT nested input format:

[
  [[demo_tokens], [event_tokens]],
  [[demo_tokens], [event_tokens]],
  ...
]

Example:
[
  [[0, 2, 5], [0, 55555, 32578, 12358, 1]],
  [[1, 4, 6], [0, 47856, 21471, 1, 0, 25868, 1]],
]

What this script does
---------------------
1) Converts the nested input into model-ready tensors:
     input_ids      : [N, max_len]
     attention_mask : [N, max_len]
     segment_ids    : [N, max_len]
        0 = demographic tokens
        1 = event tokens

2) Segment embeddings (DEFAULT: ON)
   - segment_ids are passed as event_type_ids to the encoder
   - cfg.use_event_type_embeddings = True by default
   - cfg.n_event_types = 2 (demo vs events)
   - Can be disabled with: --no_event_types

3) MLM masking (BERT-style 80/10/10)
   - Default: mask ONLY event tokens (segment == 1)
   - Demographic tokens are never masked
   - Optional: --mask_demo also masks demographic tokens

4) Logging
   - Console + CSV logging
   - Metrics: loss, grad_norm, logits_max, n_masked, masked_ratio

Recommended runs
----------------
1) Standard synthetic run (segment embeddings ON):
   cd transformer
   python transformer_train_model.py

2) Slack dummy example (only 2 patients → reduce batch size):
   cd transformer
   python transformer_train_model.py --use_slack_dummy --batch_size 2 --steps 200

3) Disable segment embeddings (ablation):
   python transformer_train_model.py --no_event_types
"""

from __future__ import annotations

import os
import csv
import math
import time
import random
from dataclasses import asdict
from typing import Dict, Optional, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# Local imports
from compact_transformer_encoder import (
    CompactTransformerConfig,
    CompactTransformerEncoder,
)
from mlm_masking import mlm_mask_801010


# =================================================
# Reproducibility
# =================================================
def set_seed(seed: int = 0) -> None:
    """Set RNG seeds for reproducible debug runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =================================================
# Helpers: nested joint format -> flat tensors
# =================================================
def _get_optional_sep_id(vocab) -> Optional[int]:
    """
    Optional support for a [SEP] token.
    If the vocab exposes get_separator_token() or get_sep_token(),
    it will be inserted between demographics and events.
    """
    if hasattr(vocab, "get_separator_token"):
        return vocab.token_to_id(vocab.get_separator_token())
    if hasattr(vocab, "get_sep_token"):
        return vocab.token_to_id(vocab.get_sep_token())
    return None


def build_joint_sequences(
    *,
    demographics: List[List[int]],
    events: List[List[int]],
    max_len: int,
    pad_id: int,
    sep_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Build flat sequences by concatenating:
      demographics (+ optional SEP) + events + PAD

    Returns:
      input_ids      : [N, max_len]
      attention_mask : [N, max_len]
      segment_ids    : [N, max_len]
    """
    if len(demographics) != len(events):
        raise ValueError("demographics/events length mismatch")

    input_ids_all, attn_all, seg_all = [], [], []

    for demo_tok, event_tok in zip(demographics, events):
        seq, seg = [], []

        # Demographics block
        seq.extend(demo_tok)
        seg.extend([0] * len(demo_tok))

        # Optional separator
        if sep_id is not None:
            seq.append(sep_id)
            seg.append(0)

        # Events block
        seq.extend(event_tok)
        seg.extend([1] * len(event_tok))

        # Clip
        seq = seq[:max_len]
        seg = seg[:max_len]

        attn = [1] * len(seq)

        # Pad
        while len(seq) < max_len:
            seq.append(pad_id)
            attn.append(0)
            seg.append(0)

        input_ids_all.append(seq)
        attn_all.append(attn)
        seg_all.append(seg)

    return input_ids_all, attn_all, seg_all


def build_from_joint_format(
    *,
    joint_data: List[List[List[int]]],
    max_len: int,
    pad_id: int,
    sep_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Converts EXACT format:
      [
        [[demo_tokens], [event_tokens]],
        ...
      ]
    into model-ready tensors.
    """
    demographics, events = [], []

    for i, item in enumerate(joint_data):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid entry at index {i}: {item}")
        demo_tok, event_tok = item
        demographics.append(demo_tok)
        events.append(event_tok)

    return build_joint_sequences(
        demographics=demographics,
        events=events,
        max_len=max_len,
        pad_id=pad_id,
        sep_id=sep_id,
    )


# =================================================
# Dataset / DataLoader
# =================================================
class SequenceDataset(Dataset):
    """
    Dataset holding already-built model sequences.
    segment_ids are passed as event_type_ids.
    """

    def __init__(
        self,
        input_ids: List[List[int]],
        attention_mask: List[List[int]],
        segment_ids: Optional[List[List[int]]] = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        if self.segment_ids is not None:
            out["event_type_ids"] = torch.tensor(self.segment_ids[idx], dtype=torch.long)
        return out


def make_dataloader(
    input_ids: List[List[int]],
    attention_mask: List[List[int]],
    segment_ids: Optional[List[List[int]]] = None,
    batch_size: int = 16,
) -> DataLoader:
    ds = SequenceDataset(input_ids, attention_mask, segment_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# =================================================
# MLM batch preparation
# =================================================
def make_mlm_batch(
    batch: Dict[str, torch.Tensor],
    *,
    vocab,
    vocab_size: int,
    p_mlm: float,
    device: torch.device,
    mask_demo: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Apply MLM masking.
    Default: mask ONLY event tokens (segment==1).
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    pad_id = vocab.token_to_id(vocab.get_padding_token())
    mask_id = vocab.token_to_id(vocab.get_masking_token())
    unk_id = vocab.token_to_id(vocab.get_unknown_token())

    effective_mask = attention_mask
    seg = batch.get("event_type_ids")

    if seg is not None and not mask_demo:
        seg = seg.to(device)
        effective_mask = attention_mask.clone()
        effective_mask[seg == 0] = 0  # never mask demographics

    masked_input_ids, labels = mlm_mask_801010(
        input_ids=input_ids,
        attention_mask=effective_mask,
        mask_token_id=mask_id,
        vocab_size=vocab_size,
        p_mlm=p_mlm,
        pad_token_id=pad_id,
        never_mask_token_ids=[pad_id, mask_id, unk_id],
    )

    out = {
        "input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if seg is not None:
        out["event_type_ids"] = seg

    return out


# =================================================
# Training utilities
# =================================================
def global_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.data.norm(2).item() ** 2)
    return math.sqrt(total)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_log_path(base_dir: str = "logs", prefix: str = "train_log") -> str:
    ensure_dir(base_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{prefix}_{stamp}.csv")


# =================================================
# Training loop
# =================================================
def train_mlm(
    *,
    vocab,
    input_ids,
    attention_mask,
    segment_ids,
    cfg: CompactTransformerConfig,
    steps: int,
    batch_size: int,
    lr: float,
    p_mlm: float,
    seed: int = 0,
    mask_demo: bool = False,
) -> None:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = make_dataloader(
        input_ids,
        attention_mask,
        segment_ids if cfg.use_event_type_embeddings else None,
        batch_size,
    )

    model = CompactTransformerEncoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    log_path = make_run_log_path()
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
            mask_demo=mask_demo,
        )

        optimizer.zero_grad(set_to_none=True)
        out = model(**mlm_batch)
        loss = out["loss"]
        loss.backward()

        grad_norm = global_grad_norm(model)
        optimizer.step()

        logits_max = out["logits"].abs().max().item()

        labels = mlm_batch["labels"]
        n_masked = (labels != -100).sum().item()
        n_valid = (mlm_batch["attention_mask"] == 1).sum().item()
        masked_ratio = n_masked / max(1, n_valid)

        if step % 10 == 0:
            print(
                f"[step {step:4d}] "
                f"loss={loss.item():.4f} "
                f"grad_norm={grad_norm:.3f} "
                f"logits_max={logits_max:.3f} "
                f"n_masked={n_masked} "
                f"masked_ratio={masked_ratio:.3f}"
            )

        step += 1

    print("\nTraining finished.")
    print(f"Logs written to {log_path}")


# =================================================
# Dummy data + main
# =================================================
def make_synth_joint_nested(n_seq: int, vocab_size: int) -> List[List[List[int]]]:
    rng = random.Random(0)
    joint = []
    for _ in range(n_seq):
        demo = [rng.randint(3, 200) for _ in range(3)]
        ev = [rng.randint(3, vocab_size - 1) for _ in range(rng.randint(10, 60))]
        joint.append([demo, ev])
    return joint


def main() -> None:
    vocab_size = 70000

    class MiniVocab:
        def get_padding_token(self): return "[PAD]"
        def get_masking_token(self): return "[MASK]"
        def get_unknown_token(self): return "[UNK]"
        def token_to_id(self, tok: str) -> int:
            return {"[PAD]": 0, "[MASK]": 1, "[UNK]": 2}[tok]

    vocab = MiniVocab()

    cfg = CompactTransformerConfig(
        vocab_size=vocab_size,
        max_len=128,
        d_model=192,
        n_layers=3,
        n_heads=6,
        use_event_type_embeddings=True,
        n_event_types=2,
        pad_token_id=0,
        mask_token_id=1,
    )

    joint = make_synth_joint_nested(800, vocab_size)

    input_ids, attn, seg = build_from_joint_format(
        joint_data=joint,
        max_len=cfg.max_len,
        pad_id=cfg.pad_token_id,
    )

    train_mlm(
        vocab=vocab,
        input_ids=input_ids,
        attention_mask=attn,
        segment_ids=seg,
        cfg=cfg,
        steps=300,
        batch_size=16,
        lr=3e-4,
        p_mlm=0.15,
    )


if __name__ == "__main__":
    main()
