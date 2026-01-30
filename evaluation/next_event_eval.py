'''
Docstring für evaluation.next_event_eval

PATH: 
    python -m evaluation.next_event_eval \
        --jsonl data/eval.jsonl \
        --ckpt checkpoints/mlm_d384.pt \
        --topk 1,5,10

    python -m evaluation.next_event_eval \
        --jsonl data/eval.jsonl \
        --ckpt checkpoints/mlm_recency.pt \
        --topk 1,5,10

    python -m evaluation.next_event_eval \
        --jsonl data/eval.jsonl \
        --ckpt checkpoints/mlm_span.pt \
        --topk 1,5,10   
'''


from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure repo root is importable : Ensure repo root is importable so "transformer" works.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import (
    IGNORE_INDEX,
    ClinicalSequenceDataset,
    TopKResult,
    load_jsonl,
    topk_accuracy_from_logits,
)
from compact_transformer_encoder import (
    CompactTransformerConfig,
    CompactTransformerEncoder,
)


def _extract_logits(model_out: Any) -> torch.Tensor:
    if isinstance(model_out, dict) and "logits" in model_out:
        return model_out["logits"]
    if hasattr(model_out, "logits"):
        return model_out.logits
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        # Heuristic: first element is logits
        return model_out[0]
    raise ValueError("Could not extract logits from model output")


def build_next_event_objective(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    event_type_ids: torch.Tensor,
    *,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask the last *event* token and create labels for that position only.

    This implements a practical "next-event" prediction with a bidirectional encoder:
      - identify the last position where (attention_mask==1 and event_type_ids==1)
      - replace that token with [MASK]
      - labels = original token at that position, else IGNORE_INDEX
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [B, L]")

    B, L = input_ids.shape
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    valid_event = (attention_mask == 1) & (event_type_ids == 1)
    for b in range(B):
        idx = torch.where(valid_event[b])[0]
        if idx.numel() == 0:
            continue
        j = int(idx[-1].item())
        labels[b, j] = input_ids[b, j]
        masked[b, j] = int(mask_token_id)

    return masked, labels


@torch.no_grad()
def run_next_event_eval(
    *,
    model: CompactTransformerEncoder,
    loader: DataLoader,
    device: torch.device,
    mask_token_id: int,
    topk: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_pred = 0
    topk_hits = {k: 0 for k in topk}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        event_type_ids = batch["event_type_ids"].to(device)

        masked_ids, labels = build_next_event_objective(
            input_ids,
            attention_mask,
            event_type_ids,
            mask_token_id=mask_token_id,
        )

        out = model(
            input_ids=masked_ids,
            attention_mask=attention_mask,
            event_type_ids=event_type_ids,
            labels=None,
            return_hidden=False,
        )
        logits = _extract_logits(out)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        n_pred = int((labels != IGNORE_INDEX).sum().item())
        total_loss += float(loss.item())
        total_pred += n_pred

        for k in topk:
            r: TopKResult = topk_accuracy_from_logits(logits, labels, k=k)
            if r.n > 0:
                topk_hits[k] += int(round(r.accuracy * r.n))

    metrics: Dict[str, Any] = {
        "n_examples": len(loader.dataset),
        "n_pred": int(total_pred),
        "next_event_loss": (total_loss / max(1, total_pred)),
        "next_event_ppl": float(torch.exp(torch.tensor(total_loss / max(1, total_pred))).item()),
    }
    for k in topk:
        metrics[f"top{k}"] = float(topk_hits[k] / max(1, total_pred))
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate next-event prediction (Top-k) via masking last event token.")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--topk", type=str, default="1,5,10")
    return p.parse_args()


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
            "Either retrain with larger max_len or reduce eval max_len."
        )

    model = CompactTransformerEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ks = tuple(int(x) for x in args.topk.split(",") if x.strip())
    records = load_jsonl(args.jsonl)
    ds = ClinicalSequenceDataset(records, max_len=args.max_len, pad_id=args.pad_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    metrics = run_next_event_eval(
        model=model,
        loader=loader,
        device=device,
        mask_token_id=int(args.mask_id),
        topk=ks,
    )

    print(f"device={device}")
    print(f"n_examples={metrics['n_examples']} n_pred={metrics['n_pred']}")
    print(f"next_event_loss={metrics['next_event_loss']:.6f} next_event_ppl={metrics['next_event_ppl']:.4f}")
    for k in ks:
        print(f"top{k}={metrics[f'top{k}']:.4f}")


if __name__ == "__main__":
    main()
