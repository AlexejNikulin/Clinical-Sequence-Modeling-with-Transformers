from __future__ import annotations
'''
Docstring für evaluation.next_event_eval

PATH eval_val.jsonl: 
python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_baseline.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_d384.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_event_types_7.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_heads_12_n_layer_6.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_heads_12.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_layer_6.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_recency.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_span.pt --topk 1,5,10 --mask_id 1


PATH test_ids.jsonl:
python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_baseline.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_event_types_7.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_heads_12_n_layer_6.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_heads_12.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_layer_6.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_p_mlm_0_1.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_p_mlm_0_2.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_mask_mode_span.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_mask_mode_recency.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_dropout_0_5.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_dropout_0_15.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_dropout_0_15.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_d_model_384.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_max_len_384.pt --topk 1,5,10 --mask_id 1


'''
"""

Token-level next-event evaluation:
Given a context window, predict the next token.

Adds window sampling:
  --sample_windows
  --keep_prefix_n

Same requirement:
- random window start for long sequences
- demographics prefix fixed at first keep_prefix_n positions
- also short sequences "sampled" (random start in body, then padding)
"""


import argparse
import os
import sys
from typing import Any, Dict, List
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import ClinicalSequenceDataset, load_jsonl, NextEventDataset, mrr_from_logits
from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Next-event (next token) evaluation with optional window sampling.")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)

    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--default_event_type_id", type=int, default=1)

    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--seed", type=int, default=0)

    # sampling flags
    p.add_argument("--sample_windows", type=bool, default=True)
    p.add_argument("--keep_prefix_n", type=int, default=3)

    return p.parse_args()


def _load_ckpt_and_model(ckpt_path: str) -> CompactTransformerEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt:
        cfg = CompactTransformerConfig(**ckpt["cfg"])
        model = CompactTransformerEncoder(cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model
    raise ValueError("Unsupported checkpoint format. Expected dict with keys: cfg, model_state_dict.")

@torch.no_grad()
def _token_topk_acc(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    logits_flat = logits[mask]     # [N, V]
    labels_flat = labels[mask]     # [N]
    topk = logits_flat.topk(k, dim=1).indices
    correct = (topk == labels_flat.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


@torch.no_grad()
def evaluate_next_event(
    model: CompactTransformerEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    topks: List[int],
) -> Dict[str, Any]:
    model.eval()

    total_eval_positions = 0
    skipped_batches = 0

    sum_topk = {k: 0.0 for k in topks}
    sum_mrr = 0.0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ev = batch["event_type_ids"].to(device)

        B, L = input_ids.shape
        labels = torch.full((B, L), -100, dtype=torch.long, device=device)

        # Mask the last valid position in each sequence
        x = input_ids.clone()
        for b in range(B):
            valid = (attn[b] == 1).nonzero(as_tuple=False).view(-1)
            if valid.numel() < 1:
                continue
            last_idx = int(valid[-1].item())
            y = int(x[b, last_idx].item())
            labels[b, last_idx] = y
            x[b, last_idx] = int(1)

        n_pos = int((labels != -100).sum().item())
        if n_pos == 0:
            skipped_batches += 1
            continue

        out = model(input_ids=x, attention_mask=attn, event_type_ids=ev, labels=labels, return_hidden=False)
        logits = out["logits"]

        for k in topks:
            acc = _token_topk_acc(logits, labels, k=k)
            sum_topk[k] += acc * n_pos

        mrr = float(mrr_from_logits(logits, labels))
        sum_mrr += mrr * n_pos

        total_eval_positions += n_pos

    denom = max(1, total_eval_positions)
    metrics: Dict[str, Any] = {
        "next_event_total_eval_positions": int(total_eval_positions),
        "next_event_skipped_batches_zero_eval_positions": int(skipped_batches),
        "next_event_mrr": float(sum_mrr / denom),
    }
    for k in topks:
        metrics[f"next_event_top{k}_acc"] = float(sum_topk[k] / denom)
    return metrics


def main() -> None:
    args = parse_args()

    model = _load_ckpt_and_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)
    topks = [int(x) for x in args.topk.split(",") if x.strip()]

    ds = ClinicalSequenceDataset(
        records,
        max_len=args.max_len,
        pad_id=args.pad_id,
        default_event_type_id=args.default_event_type_id,
        sample_windows=bool(args.sample_windows),
        keep_prefix_n=int(args.keep_prefix_n),
        seed=int(args.seed),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_next_event(model, loader, device=device, topks=topks)

    print(f"device={device}")
    print(f"sample_windows={bool(args.sample_windows)} keep_prefix_n={int(args.keep_prefix_n)} max_len={int(args.max_len)}")
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()