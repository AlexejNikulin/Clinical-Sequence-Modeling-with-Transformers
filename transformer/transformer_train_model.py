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
  python -m transformer.transformer_train_model

Or from transformer/ folder:
  cd transformer
  python transformer_train_model.py
"""

from __future__ import annotations

# -------------------------------------------------
# tqdm is optional (fallback to plain range)
# -------------------------------------------------
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x

import os
import csv
import json
import math
import time
import random
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional, List, Tuple, Literal

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


# =================================================
# Reproducibility
# =================================================
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =================================================
# Loader for ids.json (joint format)
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
# Device helpers (GPU selection + optional MultiGPU)
# =================================================
def resolve_device(
    *,
    device_mode: Literal["auto", "cpu", "cuda"],
    cuda_device: int,
) -> torch.device:
    """
    Resolve torch.device based on CLI args.

    - device_mode=cpu  -> CPU
    - device_mode=cuda -> requires CUDA
    - device_mode=auto -> CUDA if available else CPU

    cuda_device selects which CUDA index to use (e.g., 0,1,2,...).
    """
    if device_mode == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available() and device_mode in ("auto", "cuda"):
        n = torch.cuda.device_count()
        if cuda_device < 0 or cuda_device >= n:
            raise ValueError(
                f"--cuda_device={cuda_device} is invalid. Available CUDA devices: 0..{n - 1}"
            )
        torch.cuda.set_device(cuda_device)
        return torch.device(f"cuda:{cuda_device}")

    return torch.device("cpu")


def parse_gpu_ids(gpu_ids: Optional[str]) -> Optional[List[int]]:
    """
    Parse comma-separated GPU ids, e.g. "0,1,2".
    Returns None if gpu_ids is None.
    """
    if gpu_ids is None:
        return None
    ids = [int(x.strip()) for x in gpu_ids.split(",") if x.strip() != ""]
    if len(ids) == 0:
        raise ValueError("--gpu_ids provided but empty (example: --gpu_ids \"0,1\")")
    return ids


# =================================================
# Helpers: nested joint format -> flat tensors
# =================================================
def _get_optional_sep_id(vocab) -> Optional[int]:
    if hasattr(vocab, "get_separator_token"):
        return vocab.token_to_id(vocab.get_separator_token())
    if hasattr(vocab, "get_sep_token"):
        return vocab.token_to_id(vocab.get_sep_token())
    return None

# =================================================
# Dataset / DataLoader
# =================================================
class SequenceDataset2(Dataset):
    def __init__(
            self,
            sequences: List[List[List[int]]],
            max_len: int,
            pad_id: int,
            sep_id: Optional[int] = None,
            use_event_type_embeddings: bool = False,
        ):
        self.sequences = sequences
        self.max_len = max_len
        self.pad_id = pad_id
        self.sep_id = sep_id
        self.use_event_type_embeddings = use_event_type_embeddings
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demographics, events = self.sequences[idx]

        min_budget = int(self.max_len * 0.5)
        max_budget = int(self.max_len * 1.1)
        dynamic_len = min(self.max_len, torch.randint(min_budget, max_budget + 1, (1,)).item())

        seq: List[int] = []
        seg: List[int] = []

        # Add demographic data
        seq.extend(demographics)
        seg.extend([0] * len(demographics))

        if self.sep_id is not None:
            seq.append(self.sep_id)
            seg.append(0)

        event_token_num = dynamic_len - len(seq)
        
        if len(events) <= event_token_num:
            seq.extend(events)
            seg.extend([1] * len(events))
        else:
            max_start = len(events) - event_token_num
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
            
            seq.extend(events[start_idx : start_idx + event_token_num])
            seg.extend([1] * event_token_num)

        attn = [1] * len(seq)

        padding_len = self.max_len - len(seq)
        seq.extend([self.pad_id] * padding_len)
        seg.extend([0] * padding_len)
        attn.extend([0] * padding_len)

        assert len(seq) == self.max_len
        assert len(seg) == self.max_len
        assert len(attn) == self.max_len

        out = {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
        if self.use_event_type_embeddings:
            out["event_type_ids"] = torch.tensor(seg, dtype=torch.long)
            
        return out

def make_dataloader2(
    sequences: List[List[List[int]]],
    max_len: int,
    pad_id: int,
    sep_id: Optional[int] = None,
    use_event_type_embeddings: bool = False,
    batch_size: int = 16,
) -> DataLoader:
    ds = SequenceDataset2(sequences, max_len, pad_id, sep_id, use_event_type_embeddings)
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
        mask_mode: Literal["token", "span", "recency"] = "token",
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

    def make_run_log_path(
        self,
        experiment_name: str = None,
        base_dir: str = "logs",
        prefix: str = "train_log",
    ) -> Tuple[str, str]:
        self.ensure_dir(base_dir)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        filename = f"{prefix}_{stamp}"
        if experiment_name is not None:
            filename += f"_{experiment_name}"

        log_path = os.path.join(base_dir, f"{filename}.csv")
        checkpoint_path = os.path.join("checkpoints_raw", f"{filename}.pth")
        return log_path, checkpoint_path

    def train_mlm(
        self,
        *,
        vocab,
        sequences: List[List[List[int]]],
        val_sequences: List[List[List[int]]],
        cfg: CompactTransformerConfig,
        steps: int,
        batch_size: int,
        lr: float,
        p_mlm: float,
        seed: int = 0,
        mask_demo: bool = False,
        mask_mode: Literal["token", "span", "recency"] = "token",
        log_every: int = 1000,
        experiment_name: str = None,
        lr_decay: bool = False,
        device: torch.device = torch.device("cpu"),
        gpu_ids: Optional[List[int]] = None,
        disable_validation: bool = False,
    ) -> None:
        set_seed(seed)
        loader = make_dataloader2(
            sequences,
            cfg.max_len,
            cfg.pad_token_id,
            cfg.sep_token_id,
            cfg.use_event_type_embeddings,
            batch_size,
        )

        val_loader = make_dataloader2(
            val_sequences,
            cfg.max_len,
            cfg.pad_token_id,
            cfg.sep_token_id,
            cfg.use_event_type_embeddings,
            batch_size,
        )

        # Build model
        model: torch.nn.Module = CompactTransformerEncoder(cfg)

        # Optional DataParallel for multi-GPU
        if gpu_ids is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available but --gpu_ids was set")

            n = torch.cuda.device_count()
            for gid in gpu_ids:
                if gid < 0 or gid >= n:
                    raise ValueError(f"GPU id {gid} invalid. Available: 0..{n - 1}")

            torch.cuda.set_device(gpu_ids[0])
            device = torch.device(f"cuda:{gpu_ids[0]}")

            model = model.to(device)
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:
            model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        if not lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 1.0,
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=steps,
            )

        model.train()

        log_path, checkpoint_path = self.make_run_log_path(experiment_name=experiment_name)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        fieldnames = ["step", "loss", "val_loss", "grad_norm", "logits_max", "n_masked", "masked_ratio"]

        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        print("=== Training config ===")
        print(asdict(cfg))
        print(f"device={device}")
        if gpu_ids is not None:
            print(f"multi_gpu(DataParallel) gpu_ids={gpu_ids}")
        print(f"log_path={log_path}")
        print(f"steps={steps}")
        print("=======================\n")

        data_iter = iter(loader)
        val_iter = iter(val_loader)

        total_loss = 0.0
        total_loss_count = 0

        epoch_count = 0

        for step in tqdm(range(steps)):
            try:
                batch = next(data_iter)
            except StopIteration:
                state_dict = (
                    model.module.state_dict()
                    if isinstance(model, torch.nn.DataParallel)
                    else model.state_dict()
                )
                torch.save(state_dict, checkpoint_path + "." + str(epoch_count))
                epoch_count += 1

                data_iter = iter(loader)
                batch = next(data_iter)
                print(f"Finished epoch {epoch_count}, saving checkpoint")

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
            lr_scheduler.step()

            logits_max = out["logits"].abs().max().item()

            labels = mlm_batch["labels"]
            n_masked = (labels != -100).sum().item()
            n_valid = (mlm_batch["attention_mask"] == 1).sum().item()
            masked_ratio = n_masked / max(1, n_valid)

            if step % log_every == 0:
                # Calculate validation loss (MLM, same objective as training)
                model.eval()

                VAL_BATCHES = 10 if not disable_validation else 0
                total_val_loss = 0.0
                val_batches_used = 0
                
                for _ in range(VAL_BATCHES):
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)

                    val_mlm_batch = self.make_mlm_batch(
                        val_batch,
                        vocab=vocab,
                        vocab_size=cfg.vocab_size,
                        p_mlm=p_mlm,
                        device=device,
                        mask_demo=mask_demo,
                        mask_mode=mask_mode,
                    )

                    out_val = model(**val_mlm_batch)
                    total_val_loss += float(out_val["loss"].item())
                    val_batches_used += 1

                val_loss = total_val_loss / max(1, val_batches_used)
                model.train()

                train_loss = total_loss / max(1, total_loss_count)
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

        # Save correctly when DataParallel is used
        state_dict = (
            model.module.state_dict()
            if isinstance(model, torch.nn.DataParallel)
            else model.state_dict()
        )
        torch.save(state_dict, checkpoint_path)

        print("\nTraining finished.")
        print(f"Logs written to {log_path}")
        print(f"Checkpoint written to {checkpoint_path}")


# =================================================
# main()
# =================================================
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=32.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_event_types", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--p_mlm", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr_decay", type=bool, default=False)
    parser.add_argument("--norm_first", type=bool, default=False)
    parser.add_argument("--rms_norm", type=bool, default=False)
    parser.add_argument("--disable_validation", type=bool, default=False)
    parser.add_argument("--mask_demo", action="store_true")
    parser.add_argument("--mask_mode", type=str, default="token", choices=["token", "span", "recency"])
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "relu", "silu"])
    parser.add_argument("--no_event_types", action="store_true")

    # GPU selection
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--gpu_ids", type=str, default=None, help='e.g. "0,1" for DataParallel')

    # Paths (relative to repo root)
    parser.add_argument("--ids_path", type=str, default="../out/sequences/ids.json")
    parser.add_argument("--val_ids_path", type=str, default="../out/sequences/val_ids.json")
    parser.add_argument("--vocab_path", type=str, default="../out/vocab/vocabulary.json")
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()

    # Resolve robustly (works from repo root OR transformer/)
    repo_root = Path(__file__).resolve().parents[1]
    ids_path = (Path(args.ids_path) if Path(args.ids_path).is_absolute() else repo_root / args.ids_path).resolve()
    val_ids_path = (Path(args.val_ids_path) if Path(args.val_ids_path).is_absolute() else repo_root / args.val_ids_path).resolve()
    vocab_path = (Path(args.vocab_path) if Path(args.vocab_path).is_absolute() else repo_root / args.vocab_path).resolve()

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if gpu_ids is None:
        device = resolve_device(device_mode=args.device, cuda_device=args.cuda_device)
    else:
        # device will be set inside trainer to cuda:<gpu_ids[0]>
        device = torch.device("cpu")

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
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        use_event_type_embeddings=(not args.no_event_types),
        n_event_types=args.n_event_types,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
        sep_token_id=sep_id,
        dropout=args.dropout,
        activation=args.activation,
        norm_first=args.norm_first,
        rms_norm=args.rms_norm,
        d_ff=args.d_ff
    )

    trainer = TransformerTrainer()
    trainer.train_mlm(
        vocab=vocab,
        sequences=joint_sequences,
        val_sequences=val_joint_sequences,
        cfg=cfg,
        steps=int(args.epochs * len(joint_sequences) / args.batch_size),
        batch_size=args.batch_size,
        lr=args.lr,
        p_mlm=args.p_mlm,
        seed=args.seed,
        mask_demo=args.mask_demo,
        mask_mode=args.mask_mode,
        log_every=int(1000 * 16.0 / args.batch_size),
        experiment_name=args.experiment_name,
        lr_decay=args.lr_decay,
        device=device,
        gpu_ids=gpu_ids,
        disable_validation=args.disable_validation,
    )


if __name__ == "__main__":
    main()
