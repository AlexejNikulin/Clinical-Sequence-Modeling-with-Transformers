from __future__ import annotations

"""
evaluation_event_type/mlm_eval.py  -----> for event type (block eval)

This prints:
  - mlm_block_top1_acc
  - mlm_block_top{k}_acc

We evaluate on "Vocabulary blocks" (next-event type), not exact token accuracy.
Token-level MRR is printed as an additional diagnostic (optional).

Examples:

# Baseline (no sampling)
python -m evaluation_event_type.mlm_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_baseline.pt \
  --topk 1,5,10 \
  --vocab_path data/vocabulary.json

# Baseline (WITH window sampling)
# - random start for each sequence window
# - keep first 3 demographic tokens always at positions 0..2
python -m evaluation_event_type.mlm_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_baseline.pt \
  --topk 1,5,10 \
  --vocab_path data/vocabulary.json \
  --sample_windows \
  --keep_prefix_n 3 \
  --seed 13

# n_event_types_7 (WITH window sampling)
python -m evaluation_event_type.mlm_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_n_event_types_7.pt \
  --topk 1,5,10 \
  --vocab_path data/vocabulary.json \
  --sample_windows \
  --keep_prefix_n 3 \
  --seed 13

# n_heads_12 (WITH window sampling)
python -m evaluation_event_type.mlm_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_n_heads_12.pt \
  --topk 1,5,10 \
  --vocab_path data/vocabulary.json \
  --sample_windows \
  --keep_prefix_n 3 \
  --seed 13
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# --- make repo importable ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import (  # noqa: E402
    IGNORE_INDEX,
    build_token_id_to_block_id_from_vocab,
    block_top1_acc_from_logits,
    block_topk_acc_from_logits,
    load_jsonl,
    mrr_from_logits,
)

from compact_transformer_encoder import (  # noqa: E402
    CompactTransformerConfig,
    CompactTransformerEncoder,
)
from mlm_masking import mlm_mask_801010  # noqa: E402


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLM evaluation by NEXT-EVENT TYPE (Vocabulary block), not exact tokens."
    )
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to vocabulary.json (required for block eval).",
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)
    p.add_argument("--topk", type=str, default="1,5,10")

    p.add_argument("--p_mlm", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)

    # If True: when selecting random tokens during masking, avoid PAD/MASK (safer)
    p.add_argument("--avoid_random_special", action="store_true")

    # If True: ignore labels in jsonl and do masking+labels on the fly
    p.add_argument("--use_on_the_fly_masking", action="store_true")

    # ✅ NEW: random window sampling
    p.add_argument(
        "--sample_windows",
        action="store_true",
        help="Randomly sample a window from each sequence (keeps prefix fixed).",
    )
    p.add_argument(
        "--keep_prefix_n",
        type=int,
        default=3,
        help="Number of prefix tokens (demographics) that must remain at positions 0..keep_prefix_n-1.",
    )
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def _jsonl_has_labels(path: str, max_check: int = 200) -> bool:
    """Check if jsonl records contain a 'labels' field."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "labels" in r:
                return True
            n += 1
            if n >= max_check:
                break
    return False


def _load_ckpt_and_model(ckpt_path: str) -> CompactTransformerEncoder:
    """
    Expected checkpoint format:
      torch.save({"cfg": cfg_dict, "model_state_dict": sd}, out_path)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt:
        cfg = CompactTransformerConfig(**ckpt["cfg"])
        model = CompactTransformerEncoder(cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model
    raise ValueError("Unsupported checkpoint format. Expected dict with keys: cfg, model_state_dict.")


def _ensure_int_list(x: Any, name: str) -> List[int]:
    """Validate that x is List[int]."""
    if x is None:
        return []
    if not isinstance(x, list):
        raise TypeError(f"{name} must be a list, got {type(x)}")
    out: List[int] = []
    for i, v in enumerate(x):
        if not isinstance(v, int):
            raise TypeError(f"{name}[{i}] must be int, got {type(v)}")
        out.append(v)
    return out


def _sample_window_keep_prefix(
    seq: List[int],
    max_len: int,
    keep_prefix_n: int,
    rng: random.Random,
) -> Tuple[List[int], int]:
    """
    Sample a window of length <= max_len.
    Keep first keep_prefix_n tokens fixed at the front.
    Return (window, start_in_body).

    Example:
      tokens=[d0,d1,d2, a,b,c,d,e], max_len=6, keep_prefix_n=3
      => prefix=[d0,d1,d2], body=[a,b,c,d,e], body_cap=3
      start random in [0..2], window = prefix + body[start:start+3]
    """
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if keep_prefix_n < 0:
        raise ValueError("keep_prefix_n must be >= 0")

    if len(seq) <= max_len:
        # only one possible window (still "sampled")
        return seq, 0

    keep_prefix_n = min(keep_prefix_n, len(seq))
    prefix = seq[:keep_prefix_n]
    body = seq[keep_prefix_n:]

    if max_len <= keep_prefix_n:
        # cannot fit full prefix -> truncate prefix
        return prefix[:max_len], 0

    body_cap = max_len - keep_prefix_n
    if len(body) <= body_cap:
        return prefix + body, 0

    start = rng.randint(0, len(body) - body_cap)
    return prefix + body[start : start + body_cap], start


# -----------------------------
# Dataset (self-contained, supports sampling)
# -----------------------------
class ClinicalSequenceDatasetSampled(Dataset):
    """
    Returns:
      - input_ids: [max_len]
      - attention_mask: [max_len]  (1 for real token, 0 for padding)
      - event_type_ids: [max_len]
      - labels: [max_len] (IGNORE_INDEX if absent)

    Supports random window sampling with keep_prefix_n and consistent sampling
    across input_ids / labels / event_type_ids.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        max_len: int,
        pad_id: int,
        default_event_type_id: int,
        sample_windows: bool,
        keep_prefix_n: int,
        seed: int,
    ) -> None:
        self.records = records
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.default_event_type_id = int(default_event_type_id)
        self.sample_windows = bool(sample_windows)
        self.keep_prefix_n = int(keep_prefix_n)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]

        # Support both naming conventions: input_ids or tokens
        input_ids = _ensure_int_list(r.get("input_ids", r.get("tokens")), "input_ids/tokens")
        if not input_ids:
            # minimal example; will be skipped later because no eval positions
            input_ids = [self.pad_id]

        # event types may be absent
        event_type_ids = _ensure_int_list(
            r.get("event_type_ids", r.get("event_types")), "event_type_ids/event_types"
        )
        if not event_type_ids:
            event_type_ids = [self.default_event_type_id] * len(input_ids)
        elif len(event_type_ids) != len(input_ids):
            raise ValueError(f"event_type_ids length != input_ids length at idx={idx}")

        labels_raw = r.get("labels", None)
        labels_list: Optional[List[int]] = None
        if labels_raw is not None:
            labels_list = _ensure_int_list(labels_raw, "labels")
            if len(labels_list) != len(input_ids):
                raise ValueError(f"labels length != input_ids length at idx={idx}")

        # Window sampling (deterministic per idx for reproducibility)
        if self.sample_windows:
            rng = random.Random(self.seed + idx)
            input_ids, _ = _sample_window_keep_prefix(input_ids, self.max_len, self.keep_prefix_n, rng)

            rng2 = random.Random(self.seed + idx)
            event_type_ids, _ = _sample_window_keep_prefix(event_type_ids, self.max_len, self.keep_prefix_n, rng2)

            if labels_list is not None:
                rng3 = random.Random(self.seed + idx)
                labels_list, _ = _sample_window_keep_prefix(labels_list, self.max_len, self.keep_prefix_n, rng3)
        else:
            input_ids = input_ids[: self.max_len]
            event_type_ids = event_type_ids[: self.max_len]
            if labels_list is not None:
                labels_list = labels_list[: self.max_len]

        # Pad to max_len
        T = len(input_ids)
        pad_len = self.max_len - T
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            event_type_ids = event_type_ids + [self.default_event_type_id] * pad_len
            if labels_list is not None:
                labels_list = labels_list + [IGNORE_INDEX] * pad_len

        attention_mask = ([1] * T + [0] * max(0, pad_len))[: self.max_len]

        if labels_list is None:
            labels_list = [IGNORE_INDEX] * self.max_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "event_type_ids": torch.tensor(event_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_mlm_block(
    model: CompactTransformerEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    vocab_size: int,
    pad_id: int,
    mask_id: int,
    p_mlm: float,
    use_on_the_fly_masking: bool,
    generator: torch.Generator,
    topks: List[int],
    token_id_to_block: Dict[int, int],
    avoid_random_special: bool = False,
) -> Dict[str, Any]:
    """
    Computes:
      - mlm_block_top1_acc
      - mlm_block_top{k}_acc
    over masked positions.

    IMPORTANT:
      We aggregate by total number of eval positions (global weighting),
      not mean over batches.
    """
    model.eval()

    avoid_random_ids = None
    if avoid_random_special:
        avoid_random_ids = [int(pad_id), int(mask_id)]

    total_eval_positions = 0
    skipped_batches = 0

    # global weighted sums
    sum_block_top1 = 0.0
    sum_block_topk = {k: 0.0 for k in topks}
    sum_mrr = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ev = batch["event_type_ids"].to(device)

        if use_on_the_fly_masking:
            masked_ids, labels = mlm_mask_801010(
                input_ids=input_ids,
                attention_mask=attn,
                mask_token_id=int(mask_id),
                vocab_size=int(vocab_size),
                p_mlm=float(p_mlm),
                pad_token_id=int(pad_id),
                generator=generator,
                avoid_random_token_ids=avoid_random_ids,
            )
            labels = labels.masked_fill(attn == 0, IGNORE_INDEX)
            x = masked_ids
        else:
            labels = batch["labels"].to(device)
            x = input_ids

        n_pos = int((labels != IGNORE_INDEX).sum().item())
        if n_pos == 0:
            skipped_batches += 1
            continue

        out = model(
            input_ids=x,
            attention_mask=attn,
            event_type_ids=ev,
            labels=labels,
            return_hidden=False,
        )
        logits = out["logits"]  # [B, T, V]

        # Block accuracies are computed internally on masked positions
        b1 = float(block_top1_acc_from_logits(logits, labels, token_id_to_block))
        sum_block_top1 += b1 * n_pos

        for k in topks:
            bk = float(block_topk_acc_from_logits(logits, labels, token_id_to_block, k=k))
            sum_block_topk[k] += bk * n_pos

        # Token-level MRR is optional diagnostic
        mrr = float(mrr_from_logits(logits, labels))
        sum_mrr += mrr * n_pos

        total_eval_positions += n_pos

    denom = max(1, total_eval_positions)

    metrics: Dict[str, Any] = {}
    metrics["mlm_total_eval_positions"] = int(total_eval_positions)
    metrics["mlm_skipped_batches_zero_eval_positions"] = int(skipped_batches)

    metrics["mlm_block_top1_acc"] = float(sum_block_top1 / denom)
    for k in topks:
        metrics[f"mlm_block_top{k}_acc"] = float(sum_block_topk[k] / denom)

    metrics["mlm_mrr_token_level"] = float(sum_mrr / denom)

    return metrics


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    model = _load_ckpt_and_model(args.ckpt)
    cfg = model.cfg if hasattr(model, "cfg") else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)

    vocab_size = (
        int(cfg.vocab_size)
        if cfg is not None and hasattr(cfg, "vocab_size")
        else int(model.token_emb.num_embeddings)
    )

    topks = [int(x) for x in args.topk.split(",") if x.strip()]

    has_labels = _jsonl_has_labels(args.jsonl, max_check=200)
    use_on_the_fly = bool(args.use_on_the_fly_masking) or (not has_labels)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    # vocab -> block mapping
    from vocabulary import Vocabulary

    vocab = Vocabulary.load(args.vocab_path)
    token_id_to_block, block_id_to_name, _ = build_token_id_to_block_id_from_vocab(vocab)

    # dataset (supports sampling)
    ds = ClinicalSequenceDatasetSampled(
        records,
        max_len=args.max_len,
        pad_id=args.pad_id,
        default_event_type_id=args.default_event_type_id,
        sample_windows=bool(args.sample_windows),
        keep_prefix_n=int(args.keep_prefix_n),
        seed=int(args.seed),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_mlm_block(
        model,
        loader,
        device=device,
        vocab_size=vocab_size,
        pad_id=int(args.pad_id),
        mask_id=int(args.mask_id),
        p_mlm=float(args.p_mlm),
        use_on_the_fly_masking=use_on_the_fly,
        generator=gen,
        topks=topks,
        token_id_to_block=token_id_to_block,
        avoid_random_special=bool(args.avoid_random_special),
    )

    print(f"device={device}")
    print(f"use_on_the_fly_masking={use_on_the_fly} (jsonl_has_labels={has_labels})")
    print("block_id_to_name=", block_id_to_name)
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()