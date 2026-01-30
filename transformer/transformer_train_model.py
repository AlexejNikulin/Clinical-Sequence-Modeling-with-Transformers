# transformer/transformer_train_model.py
"""
Transformer training script (MLM) for CompactTransformerEncoder.

Supported input format (EXACT):
[
  [[demo_token_ids], [event_token_ids]],
  [[demo_token_ids], [event_token_ids]],
  ...
]

Recommended run (IMPORTANT)
--------------------------
From repo root (preferred):
  python3 -m transformer.transformer_train_model

Or from transformer/ folder:
  cd transformer
  python3 transformer_train_model.py
"""

from __future__ import annotations

from tqdm import tqdm
import torch.nn.functional as F
import os
import csv
import json
import math
import time
import random
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional, List, Tuple, Literal
import json

import torch
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# Robust imports (works from repo root AND from transformer/)
# -------------------------------------------------
try:
    # when executed as module: python -m transformer.transformer_train_model
    from vocabulary import Vocabulary
except ModuleNotFoundError:
    # when executed directly: cd transformer && python transformer_train_model.py
    from ..vocabulary import Vocabulary  # type: ignore

from mlm_masking import mlm_mask_801010, mlm_mask_span_801010, mlm_mask_recency_801010
from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder
from evaluation.next_event_eval import _extract_logits, build_next_event_objective
from evaluation.clinical_eval_utils import IGNORE_INDEX


# =================================================
# Reproducibility
# =================================================
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =================================================
# Änderung 1: Loader für ids.json hinzufügen (oben)
# =================================================
def load_joint_sequences_from_ids(path: Path) -> List[List[List[int]]]:
    """
    Load pre-tokenized sequences from ids.json.

    Expected:
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
        raise ValueError("ids.json must contain a list")

    # minimal structure validation (first few)
    for i, item in enumerate(data[:5]):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid sequence at index {i}: expected [[demo],[events]]")

    return data


# =================================================
# Helpers: nested joint format -> flat tensors
# =================================================
def _get_optional_sep_id(vocab) -> Optional[int]:
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
    if len(demographics) != len(events):
        raise ValueError("demographics/events length mismatch")

    input_ids_all, attn_all, seg_all = [], [], []

    for demo_tok, event_tok in zip(demographics, events):
        seq: List[int] = []
        seg: List[int] = []

        # demo block
        seq.extend(demo_tok)
        seg.extend([0] * len(demo_tok))

        # optional separator (still demo segment)
        if sep_id is not None:
            seq.append(sep_id)
            seg.append(0)

        # events block
        seq.extend(event_tok)
        seg.extend([1] * len(event_tok))

        # clip
        seq = seq[:max_len]
        seg = seg[:max_len]

        # attention mask
        attn = [0 if tok == pad_id else 1 for tok in seq]

        # pad
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
# Trainer
# =================================================
class TransformerTrainer:
    def make_mlm_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        vocab,
        vocab_size: int,
        p_mlm: float,
        device: torch.device,
        mask_demo: bool = False,
        mask_mode: Literal["token", "span", "recency"] = "token"
    ) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pad_id = vocab.token_to_id(vocab.get_padding_token())
        mask_id = vocab.token_to_id(vocab.get_masking_token())
        unk_id = vocab.token_to_id(vocab.get_unknown_token())

        effective_mask = attention_mask
        seg = batch.get("event_type_ids")

        # default: do NOT mask demo tokens
        if seg is not None and not mask_demo:
            seg = seg.to(device)
            effective_mask = attention_mask.clone()
            effective_mask[seg == 0] = 0

        mask_function = mlm_mask_801010
        if mask_mode == "span":
            mask_function = mlm_mask_span_801010
        elif mask_mode == "recency":
            mask_function = mlm_mask_recency_801010

        masked_input_ids, labels = mask_function(
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

    def global_grad_norm(self, model: torch.nn.Module) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += (p.grad.data.norm(2).item() ** 2)
        return math.sqrt(total)

    def ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def make_run_log_path(self, base_dir: str = "logs", prefix: str = "train_log") -> str:
        self.ensure_dir(base_dir)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(base_dir, f"{prefix}_{stamp}.csv")
        checkpoint_path = os.path.join("checkpoints", f"{prefix}_{stamp}.pth")
        return log_path, checkpoint_path

    def train_mlm(
        self,
        *,
        vocab,
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
        seed: int = 0,
        mask_demo: bool = False,
        mask_mode: Literal["token", "span", "recency"] = "token",
        log_every: int = 10,
    ):
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loader = make_dataloader(
            input_ids,
            attention_mask,
            segment_ids if cfg.use_event_type_embeddings else None,
            batch_size,
        )

        val_loader = make_dataloader(
            val_input_ids,
            val_attention_mask,
            val_segment_ids if cfg.use_event_type_embeddings else None,
            batch_size,
        )

        model = CompactTransformerEncoder(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        model.train()

        log_path, checkpoint_path = self.make_run_log_path()
        fieldnames = ["step", "loss", "val_loss", "grad_norm", "logits_max", "n_masked", "masked_ratio"]

        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        print("=== Training config ===")
        print(asdict(cfg))
        print(f"device={device}")
        print(f"log_path={log_path}")
        print(f"steps={steps}")
        print("=======================\n")

        data_iter = iter(loader)
        val_iter = iter(val_loader)
        total_loss = 0.0
        total_loss_count = 0

        for step in tqdm(range(steps)):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            mlm_batch = self.make_mlm_batch(
                batch,
                vocab=vocab,
                vocab_size=cfg.vocab_size,
                p_mlm=p_mlm,
                device=device,
                mask_demo=mask_demo,
                mask_mode=mask_mode,
            )

            optimizer.zero_grad(set_to_none=True)
            out = model(**mlm_batch)

            loss = out["loss"]
            loss.backward()
            total_loss += float(loss.item())
            total_loss_count += 1

            grad_norm = self.global_grad_norm(model)
            optimizer.step()

            logits_max = out["logits"].abs().max().item()

            labels = mlm_batch["labels"]
            n_masked = (labels != -100).sum().item()
            n_valid = (mlm_batch["attention_mask"] == 1).sum().item()
            masked_ratio = n_masked / max(1, n_valid)

            if step % log_every == 0:
                # Calculate validation loss
                model.eval()

                VAL_BATCHES = 10
                total_val_loss = 0.0
                for _ in range(VAL_BATCHES):
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(loader)
                        val_batch = next(val_iter)
                    
                    input_ids = val_batch["input_ids"].to(device)
                    attention_mask = val_batch["attention_mask"].to(device)
                    event_type_ids = val_batch["event_type_ids"].to(device)

                    masked_ids, labels = build_next_event_objective(
                        input_ids,
                        attention_mask,
                        event_type_ids,
                        mask_token_id=vocab.token_to_id(vocab.get_masking_token()),
                    )

                    out = model(
                        input_ids=masked_ids,
                        attention_mask=attention_mask,
                        event_type_ids=event_type_ids,
                        labels=None,
                        return_hidden=False,
                    )
                    logits = _extract_logits(out)

                    val_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=IGNORE_INDEX,
                        reduction="mean",
                    )
                    total_val_loss += float(val_loss.item())

                val_loss = total_val_loss / VAL_BATCHES
                model.train()

                # Print stats
                train_loss = total_loss / total_loss_count
                total_loss = 0.0
                total_loss_count = 0
                print(
                    f"[step {step:4d}] "
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
                            "step": step,
                            "loss": train_loss,
                            "val_loss": val_loss,
                            "grad_norm": float(grad_norm),
                            "logits_max": float(logits_max),
                            "n_masked": int(n_masked),
                            "masked_ratio": float(masked_ratio),
                        }
                    )

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        print("\nTraining finished.")
        print(f"Logs written to {log_path}")
        print(f"Checkpoint written to {checkpoint_path}")


# =================================================
# Änderung 2: main() minimal anpassen
# =================================================
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=32.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--p_mlm", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_demo", action="store_true")
    parser.add_argument("--mask_mode", type=str, default="token", choices=["token", "span", "recency"])
    parser.add_argument("--no_event_types", action="store_true")

    # IMPORTANT: defaults relative to repo root
    parser.add_argument("--ids_path", type=str, default="../out/sequences/ids.json")
    parser.add_argument("--val_ids_path", type=str, default="../out/sequences/val_ids.json")
    parser.add_argument("--vocab_path", type=str, default="../out/vocab/vocabulary.json")
    args = parser.parse_args()

    # Resolve robustly (works from repo root OR transformer/)
    repo_root = Path(__file__).resolve().parents[1]
    ids_path = (Path(args.ids_path) if Path(args.ids_path).is_absolute() else repo_root / args.ids_path).resolve()
    val_ids_path = (Path(args.val_ids_path) if Path(args.val_ids_path).is_absolute() else repo_root / args.val_ids_path).resolve()
    vocab_path = (Path(args.vocab_path) if Path(args.vocab_path).is_absolute() else repo_root / args.vocab_path).resolve()

    # Load Vocabulary
    vocab = Vocabulary.load(vocab_path)

    pad_id = vocab.token_to_id(vocab.get_padding_token())
    mask_id = vocab.token_to_id(vocab.get_masking_token())
    sep_id = _get_optional_sep_id(vocab)

    # Load ids.json sequences
    joint_sequences = load_joint_sequences_from_ids(ids_path)
    val_joint_sequences = load_joint_sequences_from_ids(val_ids_path)

    cfg = CompactTransformerConfig(
        vocab_size=vocab.get_size(),
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=3,
        n_heads=6,
        use_event_type_embeddings=(not args.no_event_types),
        n_event_types=2,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
    )

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

    trainer = TransformerTrainer()
    trainer.train_mlm(
        vocab=vocab,
        input_ids=input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        val_input_ids=val_input_ids,
        val_attention_mask=val_attention_mask,
        val_segment_ids=val_segment_ids,
        cfg=cfg,
        steps=int(args.epochs * len(input_ids) / args.batch_size),
        batch_size=args.batch_size,
        lr=args.lr,
        p_mlm=args.p_mlm,
        seed=args.seed,
        mask_demo=args.mask_demo,
        mask_mode=args.mask_mode,
        log_every=1000,
    )


if __name__ == "__main__":
    main()
