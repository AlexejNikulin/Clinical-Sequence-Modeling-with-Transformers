# transformer/transformer_train_model.py
"""
Compact Transformer MLM training script.

This script trains a CompactTransformerEncoder using a Masked Language Modeling (MLM)
objective on clinical event sequences derived from MIMIC-IV.

--------------------------------------------------------------------
Input format (IMPORTANT)
--------------------------------------------------------------------
The script expects pre-tokenized input sequences in a *nested joint format*:

[
  [[demo_token_ids], [event_token_ids]],
  [[demo_token_ids], [event_token_ids]],
  ...
]

Where:
- demo_token_ids  : List[int]
    Token IDs representing static demographic information (e.g. gender, age).
- event_token_ids : List[int]
    Token IDs representing time-ordered clinical events (admissions, diagnoses, labs, meds, etc.).

These two lists are concatenated internally into a single sequence:
    demographics (+ optional [SEP]) + events

Additionally, a segment/event-type embedding is used:
- segment_id = 0 → demographic tokens
- segment_id = 1 → event tokens

--------------------------------------------------------------------
Supported MLM masking strategies
--------------------------------------------------------------------
Masking strategy is selectable via --masking:

1) token
   - Classic BERT-style masking (80/10/10 rule)
   - ~p_mlm of eligible tokens selected:
       * 80% replaced with [MASK]
       * 10% replaced with random token id
       * 10% unchanged

2) span
   - Masks contiguous spans of tokens
   - Span lengths sampled from [--span_min, --span_max]

3) recency
   - Recency-biased masking: later positions have higher mask probability
   - Controlled by --recency_power

--------------------------------------------------------------------
Masking scope (demographics vs. events)
--------------------------------------------------------------------
Default:
- ONLY event tokens (segment_id==1) are eligible for masking.
- demographic tokens are not masked.

If --mask_demo is set:
- demographic tokens are also eligible.

--------------------------------------------------------------------
Validation strategy (IMPORTANT for comparisons)
--------------------------------------------------------------------
Validation loss is computed as MLM cross-entropy under the SAME masking strategy.
A fixed validation seed is used for deterministic masking during evaluation.

This makes val_loss comparable across strategies.

--------------------------------------------------------------------
Recommended usage
--------------------------------------------------------------------
From repo root:

Token masking:
    python3 -m transformer.transformer_train_model --masking token

Span masking:
    python3 -m transformer.transformer_train_model --masking span

Recency-biased masking:
    python3 -m transformer.transformer_train_model --masking recency

--------------------------------------------------------------------
Outputs
--------------------------------------------------------------------
1) Training log (CSV):
   logs/train_log_YYYYMMDD_HHMMSS.csv
   columns:
     step, loss, val_loss, grad_norm, logits_max, n_masked, masked_ratio

2) Model checkpoint (compatible with downstream/mortality_train.py):
   checkpoints/mlm_*.pt containing:
   {
     "cfg": <CompactTransformerConfig as dict>,
     "model_state_dict": <state_dict>,
     "log_path": <path to CSV log>,
     "masking": <masking strategy>,
   }
"""

from __future__ import annotations

import os
import sys
import csv
import json
import math
import time
import random
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional, List, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader

# =================================================
# Robust imports
# =================================================
# Goal:
# - Preferred: run from repo root with:
#       python3 -m transformer.transformer_train_model
#   => relative imports work (.xxx)
#
# - Fallback: run directly from transformer/ folder with:
#       python3 transformer_train_model.py
#   => relative imports fail, so we add repo root to sys.path
#
# This block keeps the file self-contained and reduces import pain.
try:
    # Package-mode (recommended)
    from ..vocabulary import Vocabulary  # type: ignore
    from .compact_transformer_encoder import (  # type: ignore
        CompactTransformerConfig,
        CompactTransformerEncoder,
    )
    from .mlm_masking import (  # type: ignore
        mlm_mask_801010,
        mlm_mask_span_801010,
        mlm_mask_recency_801010,
    )
except Exception:
    # Fallback-mode: add repo root to sys.path
    # __file__ = .../repo/transformer/transformer_train_model.py
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Now retry as absolute imports from repo root
    from vocabulary import Vocabulary  # type: ignore
    from transformer.compact_transformer_encoder import (  # type: ignore
        CompactTransformerConfig,
        CompactTransformerEncoder,
    )
    from transformer.mlm_masking import (  # type: ignore
        mlm_mask_801010,
        mlm_mask_span_801010,
        mlm_mask_recency_801010,
    )

# =================================================
# Reproducibility helpers
# =================================================
def set_seed(seed: int = 0) -> None:
    """Set RNG seeds for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =================================================
# Load ids.json sequences
# =================================================
def load_joint_sequences_from_ids(path: Path) -> List[List[List[int]]]:
    """
    Load pre-tokenized sequences from ids.json.

    Expected format:
      [
        [[demo_ids], [event_ids]],
        ...
      ]
    """
    if not path.exists():
        raise FileNotFoundError(f"ids.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("ids.json must contain a list of sequences")

    # Light sanity check (first few only)
    for i, item in enumerate(data[:5]):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(
                f"Invalid sequence at index {i}: expected [[demo],[events]], got={type(item)} len={getattr(item,'__len__',None)}"
            )

    return data


# =================================================
# Optional [SEP] support (only if your vocab has it)
# =================================================
def _get_optional_sep_id(vocab: Any) -> Optional[int]:
    """
    Optional support for a [SEP] token.

    If Vocabulary exposes get_separator_token() or get_sep_token(),
    we will insert [SEP] between demo and events.

    If not available, returns None and no separator is inserted.
    """
    # NOTE: This is intentionally defensive because different repos name SEP differently.
    if hasattr(vocab, "get_separator_token"):
        tok = vocab.get_separator_token()
        # Some vocab implementations return None if token does not exist.
        if tok is not None:
            return int(vocab.token_to_id(tok))
    if hasattr(vocab, "get_sep_token"):
        tok = vocab.get_sep_token()
        if tok is not None:
            return int(vocab.token_to_id(tok))
    return None


# =================================================
# Build flat model sequences from nested format
# =================================================
def build_joint_sequences(
    *,
    demographics: List[List[int]],
    events: List[List[int]],
    max_len: int,
    pad_id: int,
    sep_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Build model-ready sequences by concatenating:

        demographics (+ optional SEP) + events

    Then clip/pad to max_len and create:
      - input_ids      : token ids
      - attention_mask : 1 for real token, 0 for PAD
      - segment_ids    : 0 for demo part, 1 for events part (PAD uses 0)
    """
    if len(demographics) != len(events):
        raise ValueError("demographics/events length mismatch")

    input_ids_all: List[List[int]] = []
    attn_all: List[List[int]] = []
    seg_all: List[List[int]] = []

    for demo_tok, event_tok in zip(demographics, events):
        seq: List[int] = []
        seg: List[int] = []

        # (A) demographics
        seq.extend(demo_tok)
        seg.extend([0] * len(demo_tok))

        # (B) optional separator (still segment 0)
        if sep_id is not None:
            seq.append(int(sep_id))
            seg.append(0)

        # (C) events
        seq.extend(event_tok)
        seg.extend([1] * len(event_tok))

        # (D) clip (keep left part; simplest default)
        seq = seq[:max_len]
        seg = seg[:max_len]

        # (E) attention mask: 1 for non-PAD
        # Note: because we pad later, we can also just use len-before-pad,
        # but this version stays explicit and safe.
        attn = [0 if tok == pad_id else 1 for tok in seq]

        # (F) pad up to max_len
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
    Convert EXACT format:
      [
        [[demo_ids], [event_ids]],
        ...
      ]
    into:
      input_ids, attention_mask, segment_ids
    """
    demographics: List[List[int]] = []
    events: List[List[int]] = []

    for i, item in enumerate(joint_data):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid entry at index {i}: expected [demo, events], got: {item}")

        demo_tok, event_tok = item

        if not isinstance(demo_tok, list) or not isinstance(event_tok, list):
            raise ValueError(f"Invalid entry at index {i}: demo/events must be lists, got: {type(demo_tok)} / {type(event_tok)}")

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
    Holds already-built sequences (lists) and returns tensors.

    - input_ids:      [N, L]
    - attention_mask: [N, L]
    - segment_ids:    [N, L]  (returned as event_type_ids for the encoder)
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
            # Encoder expects "event_type_ids" (segment embedding)
            out["event_type_ids"] = torch.tensor(self.segment_ids[idx], dtype=torch.long)
        return out


def make_dataloader(
    input_ids: List[List[int]],
    attention_mask: List[List[int]],
    segment_ids: Optional[List[List[int]]],
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """Wrap SequenceDataset into a DataLoader."""
    ds = SequenceDataset(input_ids, attention_mask, segment_ids)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


# =================================================
# Trainer
# =================================================
class TransformerTrainer:
    # -------------------------
    # Masking dispatch
    # -------------------------
    def _mask_dispatch(
        self,
        masking: str,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_id: int,
        vocab_size: int,
        p_mlm: float,
        pad_id: int,
        never_mask: List[int],
        generator: Optional[torch.Generator],
        span_len_range: Tuple[int, int],
        recency_power: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pick and apply a masking strategy, returning:
          - masked_input_ids
          - labels (original ids where masked, else -100)
        """
        if masking == "token":
            return mlm_mask_801010(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_token_id=mask_id,
                vocab_size=vocab_size,
                p_mlm=p_mlm,
                pad_token_id=pad_id,
                never_mask_token_ids=never_mask,
                generator=generator,
            )

        if masking == "span":
            return mlm_mask_span_801010(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_token_id=mask_id,
                vocab_size=vocab_size,
                p_mlm=p_mlm,
                span_len_range=span_len_range,
                pad_token_id=pad_id,
                never_mask_token_ids=never_mask,
                generator=generator,
            )

        if masking == "recency":
            return mlm_mask_recency_801010(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_token_id=mask_id,
                vocab_size=vocab_size,
                p_mlm=p_mlm,
                recency_power=recency_power,
                pad_token_id=pad_id,
                never_mask_token_ids=never_mask,
                generator=generator,
            )

        raise ValueError(f"Unknown masking strategy: {masking}")

    # -------------------------
    # Build MLM batch
    # -------------------------
    def make_mlm_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        vocab: Any,
        cfg: CompactTransformerConfig,
        p_mlm: float,
        device: torch.device,
        mask_demo: bool,
        masking: str,
        generator: Optional[torch.Generator],
        span_len_range: Tuple[int, int],
        recency_power: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a raw batch into a masked MLM batch.

        Key logic:
        - attention_mask defines valid (non-PAD) positions
        - segment ids (event_type_ids) define demo vs events
        - if mask_demo=False: forbid masking segment==0 (demographics)
        - always forbid masking special tokens like PAD/MASK/UNK
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pad_id = int(vocab.token_to_id(vocab.get_padding_token()))
        mask_id = int(vocab.token_to_id(vocab.get_masking_token()))
        unk_id = int(vocab.token_to_id(vocab.get_unknown_token()))

        # (1) Start with "valid tokens" mask from attention_mask
        effective_mask = attention_mask

        # (2) Optionally restrict to events only (segment==1)
        seg = batch.get("event_type_ids")
        if seg is not None and not mask_demo:
            seg = seg.to(device)
            effective_mask = attention_mask.clone()
            effective_mask[seg == 0] = 0  # block demographic positions

        # (3) Never mask special tokens
        never_mask = [pad_id, mask_id, unk_id]

        masked_input_ids, labels = self._mask_dispatch(
            masking,
            input_ids=input_ids,
            attention_mask=effective_mask,
            mask_id=mask_id,
            vocab_size=int(cfg.vocab_size),
            p_mlm=float(p_mlm),
            pad_id=pad_id,
            never_mask=never_mask,
            generator=generator,
            span_len_range=span_len_range,
            recency_power=float(recency_power),
        )

        out: Dict[str, torch.Tensor] = {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Encoder expects event_type_ids if event-type embeddings are enabled
        if seg is not None:
            out["event_type_ids"] = seg
        return out

    # -------------------------
    # Logging helpers
    # -------------------------
    def global_grad_norm(self, model: torch.nn.Module) -> float:
        """Compute global L2 norm of gradients (for stability monitoring)."""
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += (p.grad.data.norm(2).item() ** 2)
        return math.sqrt(total)

    def ensure_dir(self, path: str) -> None:
        """Create directory if it does not exist."""
        os.makedirs(path, exist_ok=True)

    def make_run_paths(self, base_dir: str = "logs", prefix: str = "train_log") -> str:
        """Return a timestamped log path."""
        self.ensure_dir(base_dir)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"{prefix}_{stamp}.csv")

    # -------------------------
    # Validation: MLM loss under same masking strategy
    # -------------------------
    @torch.no_grad()
    def evaluate_mlm_loss(
        self,
        model: CompactTransformerEncoder,
        val_loader: DataLoader,
        *,
        vocab: Any,
        cfg: CompactTransformerConfig,
        p_mlm: float,
        device: torch.device,
        masking: str,
        mask_demo: bool,
        val_batches: int,
        val_seed: int,
        span_len_range: Tuple[int, int],
        recency_power: float,
    ) -> float:
        """
        Compute validation MLM loss.

        IMPORTANT:
        - Uses the SAME masking strategy as training.
        - Uses a deterministic generator (val_seed) so that evaluation is stable.
        """
        model.eval()

        g = torch.Generator(device=device)
        g.manual_seed(int(val_seed))

        total = 0.0
        n = 0

        it = iter(val_loader)
        for _ in range(int(val_batches)):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(val_loader)
                batch = next(it)

            mlm_batch = self.make_mlm_batch(
                batch,
                vocab=vocab,
                cfg=cfg,
                p_mlm=p_mlm,
                device=device,
                mask_demo=mask_demo,
                masking=masking,
                generator=g,  # deterministic masking for validation
                span_len_range=span_len_range,
                recency_power=recency_power,
            )

            out = model(**mlm_batch)
            total += float(out["loss"].item())
            n += 1

        model.train()
        return total / max(1, n)

    # -------------------------
    # Training loop
    # -------------------------
    def train_mlm(
        self,
        *,
        vocab: Any,
        input_ids: List[List[int]],
        attention_mask: List[List[int]],
        segment_ids: Optional[List[List[int]]],
        val_input_ids: List[List[int]],
        val_attention_mask: List[List[int]],
        val_segment_ids: Optional[List[List[int]]],
        cfg: CompactTransformerConfig,
        steps: int,
        batch_size: int,
        lr: float,
        p_mlm: float,
        seed: int,
        mask_demo: bool,
        masking: str,
        span_len_range: Tuple[int, int],
        recency_power: float,
        val_batches: int,
        val_seed: int,
        log_every: int,
        weight_decay: float = 0.01,
    ) -> Tuple[CompactTransformerEncoder, str]:
        """
        Train MLM for a given number of steps.

        Returns:
          - trained model
          - log_path (CSV)
        """
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train loader (shuffle=True)
        loader = make_dataloader(
            input_ids,
            attention_mask,
            segment_ids if cfg.use_event_type_embeddings else None,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Validation loader (shuffle=False typical)
        val_loader = make_dataloader(
            val_input_ids,
            val_attention_mask,
            val_segment_ids if cfg.use_event_type_embeddings else None,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Model + optimizer
        model = CompactTransformerEncoder(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()

        # Logging
        log_path = self.make_run_paths()
        fieldnames = ["step", "loss", "val_loss", "grad_norm", "logits_max", "n_masked", "masked_ratio"]

        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        # Print run configuration
        print("=== Training config ===")
        print(asdict(cfg))
        print(f"device={device}")
        print(f"masking={masking}")
        print(f"mask_demo={mask_demo}")
        print(f"p_mlm={p_mlm}")
        print(f"span_len_range={span_len_range} (only used if masking=span)")
        print(f"recency_power={recency_power} (only used if masking=recency)")
        print(f"steps={steps} | batch_size={batch_size} | lr={lr}")
        print(f"val_batches={val_batches} | val_seed={val_seed}")
        print(f"log_every={log_every}")
        print(f"log_path={log_path}")
        print("=======================\n")

        data_iter = iter(loader)

        # Running average loss between logging points
        total_loss = 0.0
        total_loss_count = 0

        # Main training loop
        for step in range(int(steps)):
            # Get next batch (recycle iterator if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            # Build masked MLM batch (training masking is stochastic => generator=None)
            mlm_batch = self.make_mlm_batch(
                batch,
                vocab=vocab,
                cfg=cfg,
                p_mlm=p_mlm,
                device=device,
                mask_demo=mask_demo,
                masking=masking,
                generator=None,
                span_len_range=span_len_range,
                recency_power=recency_power,
            )

            # Forward / backward / step
            optimizer.zero_grad(set_to_none=True)
            out = model(**mlm_batch)

            loss = out["loss"]
            loss.backward()

            total_loss += float(loss.item())
            total_loss_count += 1

            grad_norm = self.global_grad_norm(model)
            optimizer.step()

            # Stability/debug metrics
            logits_max = float(out["logits"].abs().max().item())

            # Masking stats
            labels = mlm_batch["labels"]
            n_masked = int((labels != -100).sum().item())
            n_valid = int((mlm_batch["attention_mask"] == 1).sum().item())
            masked_ratio = float(n_masked / max(1, n_valid))

            # Logging + validation
            if step % int(log_every) == 0:
                train_loss = float(total_loss / max(1, total_loss_count))
                total_loss = 0.0
                total_loss_count = 0

                val_loss = self.evaluate_mlm_loss(
                    model,
                    val_loader,
                    vocab=vocab,
                    cfg=cfg,
                    p_mlm=p_mlm,
                    device=device,
                    masking=masking,
                    mask_demo=mask_demo,
                    val_batches=val_batches,
                    val_seed=val_seed,
                    span_len_range=span_len_range,
                    recency_power=recency_power,
                )

                print(
                    f"[step {step:6d}] "
                    f"loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"grad_norm={grad_norm:.3f} "
                    f"logits_max={logits_max:.3f} "
                    f"n_masked={n_masked} "
                    f"masked_ratio={masked_ratio:.3f}"
                )

                with open(log_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(
                        {
                            "step": int(step),
                            "loss": float(train_loss),
                            "val_loss": float(val_loss),
                            "grad_norm": float(grad_norm),
                            "logits_max": float(logits_max),
                            "n_masked": int(n_masked),
                            "masked_ratio": float(masked_ratio),
                        }
                    )

        print("\nTraining finished.")
        print(f"Logs written to {log_path}")
        return model, log_path


# =================================================
# Main entry point (CLI)
# =================================================
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()

    # -------------------------
    # Core train params
    # -------------------------
    parser.add_argument("--epochs", type=float, default=1.0, help="Converted to steps via N/batch_size.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)

    # Model size knobs
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=6)

    # Optimizer knobs
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # MLM knobs
    parser.add_argument("--p_mlm", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)

    # Segment / masking config
    parser.add_argument("--mask_demo", action="store_true", help="If set, also mask demographic tokens.")
    parser.add_argument("--no_event_types", action="store_true", help="Disable segment/event-type embeddings.")

    # Masking strategy selection
    parser.add_argument(
        "--masking",
        type=str,
        default="token",
        choices=["token", "span", "recency"],
        help="Masking strategy for MLM.",
    )

    # Span masking parameters
    parser.add_argument("--span_min", type=int, default=3)
    parser.add_argument("--span_max", type=int, default=10)

    # Recency masking parameters
    parser.add_argument("--recency_power", type=float, default=2.0)

    # Validation evaluation settings
    parser.add_argument("--val_batches", type=int, default=10, help="How many val mini-batches to average.")
    parser.add_argument("--val_seed", type=int, default=123, help="Deterministic seed for validation masking.")
    parser.add_argument("--log_every", type=int, default=1000)

    # Paths (defaults relative to repo root)
    parser.add_argument("--ids_path", type=str, default="out/sequences/ids.json")
    parser.add_argument("--val_ids_path", type=str, default="out/sequences/val_ids.json")
    parser.add_argument("--vocab_path", type=str, default="out/vocab/vocabulary.json")

    # Checkpoint output
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="mlm.pt")

    args = parser.parse_args()

    # -------------------------
    # Resolve paths robustly
    # -------------------------
    # For both:
    #   python -m transformer.transformer_train_model
    #   python transformer_train_model.py
    repo_root = Path(__file__).resolve().parents[1]

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (repo_root / pp)

    ids_path = _resolve(args.ids_path).resolve()
    val_ids_path = _resolve(args.val_ids_path).resolve()
    vocab_path = _resolve(args.vocab_path).resolve()

    # -------------------------
    # Load vocabulary
    # -------------------------
    vocab = Vocabulary.load(vocab_path)

    # Special ids
    pad_id = int(vocab.token_to_id(vocab.get_padding_token()))
    mask_id = int(vocab.token_to_id(vocab.get_masking_token()))
    sep_id = _get_optional_sep_id(vocab)

    # -------------------------
    # Load sequences (train/val)
    # -------------------------
    joint_sequences = load_joint_sequences_from_ids(ids_path)
    val_joint_sequences = load_joint_sequences_from_ids(val_ids_path)

    # -------------------------
    # Build transformer config
    # -------------------------
    # NOTE: vocab_size must cover the maximum token id used in inputs.
    # Usually vocab.get_size() is correct in well-formed vocabularies.
    cfg = CompactTransformerConfig(
        vocab_size=int(vocab.get_size()),
        max_len=int(args.max_len),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        use_event_type_embeddings=(not bool(args.no_event_types)),
        n_event_types=2,  # 0=demo, 1=events
        pad_token_id=int(pad_id),
        mask_token_id=int(mask_id),
    )

    # -------------------------
    # Convert nested joint format -> model sequences
    # -------------------------
    input_ids, attention_mask, segment_ids = build_from_joint_format(
        joint_data=joint_sequences,
        max_len=cfg.max_len,
        pad_id=cfg.pad_token_id,
        sep_id=sep_id,
    )

    val_input_ids, val_attention_mask, val_segment_ids = build_from_joint_format(
        joint_data=val_joint_sequences,
        max_len=cfg.max_len,
        pad_id=cfg.pad_token_id,
        sep_id=sep_id,
    )

    # -------------------------
    # Convert epochs -> steps
    # -------------------------
    # steps = epochs * (N / batch_size)
    steps = int(float(args.epochs) * (len(input_ids) / max(1, int(args.batch_size))))
    steps = max(1, steps)

    # -------------------------
    # Train
    # -------------------------
    trainer = TransformerTrainer()
    trained_model, log_path = trainer.train_mlm(
        vocab=vocab,
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=(segment_ids if cfg.use_event_type_embeddings else None),
        val_input_ids=val_input_ids,
        val_attention_mask=val_attention_mask,
        val_segment_ids=(val_segment_ids if cfg.use_event_type_embeddings else None),
        cfg=cfg,
        steps=steps,
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        p_mlm=float(args.p_mlm),
        seed=int(args.seed),
        mask_demo=bool(args.mask_demo),
        masking=str(args.masking),
        span_len_range=(int(args.span_min), int(args.span_max)),
        recency_power=float(args.recency_power),
        val_batches=int(args.val_batches),
        val_seed=int(args.val_seed),
        log_every=int(args.log_every),
        weight_decay=float(args.weight_decay),
    )

    # -------------------------
    # Save checkpoint (downstream-compatible)
    # -------------------------
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / str(args.ckpt_name)

    torch.save(
        {
            "cfg": asdict(cfg),
            "model_state_dict": trained_model.state_dict(),
            "log_path": str(log_path),
            "masking": str(args.masking),
        },
        ckpt_path,
    )

    print(f"\nSaved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
