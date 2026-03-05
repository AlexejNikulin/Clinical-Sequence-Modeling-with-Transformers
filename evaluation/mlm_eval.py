from __future__ import annotations
'''
Docstring for evaluation.mlm_eval

ls -la data/eval_val.jsonl
ls -la data | head
find . -maxdepth 4 -name "eval.jsonl" -print



PATH: 
python -m evaluation.mlm_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_d384.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_baseline.pt --topk 1,5,10 

python -m evaluation.mlm_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_event_types_7.pt --topk 1,5,10 

python -m evaluation.mlm_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_heads_12_n_layer_6.pt --topk 1,5,10 

python -m evaluation.mlm_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_heads_12.pt --topk 1,5,10 

python -m evaluation.mlm_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_layer_6.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_recency.pt --topk 1,5,10 

python -m evaluation.mlm_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_span.pt --topk 1,5,10 


PATH FOR test_ides.jsonl: vocab_size = 77011

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_baseline.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_event_types_7.pt --topk 1,5,10 

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_heads_12_n_layer_6.pt --topk 1,5,10 

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_heads_12.pt --topk 1,5,10 

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_layer_6.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_p_mlm_0_1.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_p_mlm_0_2.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_mask_mode_span.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_mask_mode_recency.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_dropout_0_5.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_dropout_0_15.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_d_model_384.pt --topk 1,5,10

python -m evaluation.mlm_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_max_len_384.pt --topk 1,5,10
'''
"""
Token-level MLM evaluation.
Adds window sampling:
  --sample_windows
  --keep_prefix_n

Requirement:
- Random context window start if sequence exceeds context length.
- Demographics (first keep_prefix_n tokens) always stay at positions 0..keep_prefix_n-1.
- Also short sequences are "sampled" (random start in body, then padding).
"""


import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import (
    ClinicalSequenceDataset,
    IGNORE_INDEX,
    load_jsonl,
    mrr_from_logits,
)

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder
from mlm_masking import mlm_mask_801010


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Token-level MLM evaluation (with optional window sampling).")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)

    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--p_mlm", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--avoid_random_special", action="store_true")
    p.add_argument("--use_on_the_fly_masking", action="store_true")

    # ✅ sampling flags
    p.add_argument("--sample_windows", type=bool, default=True)
    p.add_argument("--keep_prefix_n", type=int, default=3)

    return p.parse_args()


def _jsonl_has_labels(path: str, max_check: int = 200) -> bool:
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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt:
        cfg = CompactTransformerConfig(**ckpt["cfg"])
        model = CompactTransformerEncoder(cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model
    raise ValueError("Unsupported checkpoint format. Expected dict with keys: cfg, model_state_dict.")


@torch.no_grad()
def _token_topk_acc(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    mask = labels != IGNORE_INDEX
    if mask.sum().item() == 0:
        return 0.0
    logits_flat = logits[mask]     # [N, V]
    labels_flat = labels[mask]     # [N]
    topk = logits_flat.topk(k, dim=1).indices
    correct = (topk == labels_flat.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


@torch.no_grad()
def evaluate_mlm_token_level(
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
    avoid_random_special: bool = False,
) -> Dict[str, Any]:
    model.eval()

    avoid_random_ids = None
    if avoid_random_special:
        avoid_random_ids = [int(pad_id), int(mask_id)]

    total_eval_positions = 0
    skipped_batches = 0

    sum_topk = {k: 0.0 for k in topks}
    sum_mrr = 0.0

    for batch in tqdm(loader):
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
        "mlm_total_eval_positions": int(total_eval_positions),
        "mlm_skipped_batches_zero_eval_positions": int(skipped_batches),
        "mlm_mrr": float(sum_mrr / denom),
    }
    for k in topks:
        metrics[f"mlm_top{k}_acc"] = float(sum_topk[k] / denom)
    return metrics


def main() -> None:
    args = parse_args()

    model = _load_ckpt_and_model(args.ckpt)
    cfg = model.cfg if hasattr(model, "cfg") else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)
    vocab_size = int(cfg.vocab_size) if cfg is not None and hasattr(cfg, "vocab_size") else int(model.token_emb.num_embeddings)

    topks = [int(x) for x in args.topk.split(",") if x.strip()]

    has_labels = _jsonl_has_labels(args.jsonl, max_check=200)
    use_on_the_fly = bool(args.use_on_the_fly_masking) or (not has_labels)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

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

    metrics = evaluate_mlm_token_level(
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
        avoid_random_special=bool(args.avoid_random_special),
    )

    print(f"device={device}")
    print(f"use_on_the_fly_masking={use_on_the_fly} (jsonl_has_labels={has_labels})")
    print(f"sample_windows={bool(args.sample_windows)} keep_prefix_n={int(args.keep_prefix_n)} max_len={int(args.max_len)}")

    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()