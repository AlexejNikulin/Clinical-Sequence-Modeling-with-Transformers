'''
Run a small evaluation suite for a pretrained clinical MLM encoder checkpoint.
It is intended for quick, consistent comparisons across multiple checkpoints
(e.g., baseline vs architecture ablations vs different MLM masking).
It does NOT perform any recursive rollout generation; for that see:
    evaluation.rollout_eval  (predict-until-discharge/death)

   

USE PATH: 

python -m evaluation.run_eval_suite \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_baseline.pt \
  --batch_size 32 \
  --max_len 256 \
  --pad_id 0 \
  --mask_id 1 \
  --default_event_type_id 1 \
  --topk 1,5,10

 python -m evaluation.run_eval_suite \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_d384.pt \
  --batch_size 32 \
  --max_len 256 \
  --pad_id 0 \
  --mask_id 1 \
  --default_event_type_id 1 \
  --topk 1,5,10

python -m evaluation.run_eval_suite \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_n_event_types_7.pt \
  --batch_size 32 \
  --max_len 256 \
  --pad_id 0 \
  --mask_id 1 \
  --default_event_type_id 1 \
  --topk 1,5,10

 python -m evaluation.run_eval_suite \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_n_heads_12_n_layer_6.pt \
  --batch_size 32 \
  --max_len 256 \
  --pad_id 0 \
  --mask_id 1 \
  --default_event_type_id 1 \
  --topk 1,5,10

python -m evaluation.run_eval_suite \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_n_heads_12.pt \
  --batch_size 32 \
  --max_len 256 \
  --pad_id 0 \
  --mask_id 1 \
  --default_event_type_id 1 \
  --topk 1,5,10

python -m evaluation.run_eval_suite \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_n_layer_6.pt \
  --batch_size 32 \
  --max_len 256 \
  --pad_id 0 \
  --mask_id 1 \
  --default_event_type_id 1 \
  --topk 1,5,10

python -m evaluation.run_eval_suite \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_recency.pt \
  --batch_size 32 \
  --max_len 256 \
  --pad_id 0 \
  --mask_id 1 \
  --default_event_type_id 1 \
  --topk 1,5,10  

python -m evaluation.run_eval_suite \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_span.pt \
  --batch_size 32 \
  --max_len 256 \
  --pad_id 0 \
  --mask_id 1 \
  --default_event_type_id 1 \
  --topk 1,5,10       
'''
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.mlm_eval import evaluate as eval_mlm
from evaluation.next_event_eval import evaluate_next_event
from evaluation.clinical_eval_utils import (
    ClinicalSequenceDataset,
    load_jsonl,
    build_token_id_to_group_from_vocab,
)
from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full evaluation suite for clinical MLM encoder model.")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--vocab_path", type=str, default=None, help="Path to vocabulary.json (for event-type eval).")

    p.add_argument("--p_mlm", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--use_on_the_fly_masking", action="store_true")
    p.add_argument("--avoid_random_special", action="store_true")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)
    p.add_argument("--topk", type=str, default="1,5,10")
    return p.parse_args()


def _load_ckpt_and_model(ckpt_path: str) -> CompactTransformerEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt:
        cfg = CompactTransformerConfig(**ckpt["cfg"])
        model = CompactTransformerEncoder(cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model
    raise ValueError("Unsupported checkpoint format. Expected dict with keys: cfg, model_state_dict.")


def main() -> None:
    args = parse_args()

    model = _load_ckpt_and_model(args.ckpt)
    cfg = model.cfg if hasattr(model, "cfg") else None
    if cfg is None:
        raise ValueError("Model has no cfg attribute; expected CompactTransformerEncoder(cfg) to keep it.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)
    ds = ClinicalSequenceDataset(records, max_len=args.max_len, pad_id=args.pad_id, default_event_type_id=args.default_event_type_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    topks = [int(x) for x in args.topk.split(",") if x.strip()]

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    token_id_to_group = None
    if args.vocab_path:
        from vocabulary import Vocabulary
        vocab = Vocabulary.load(args.vocab_path)
        token_id_to_group = build_token_id_to_group_from_vocab(vocab)

    # 1) MLM metrics
    mlm_metrics = eval_mlm(
        model,
        loader,
        device=device,
        topks=topks,
        vocab_size=int(cfg.vocab_size),
        pad_id=int(args.pad_id),
        mask_id=int(args.mask_id),
        p_mlm=float(args.p_mlm),
        use_on_the_fly_masking=bool(args.use_on_the_fly_masking),
        generator=gen,
        token_id_to_group=token_id_to_group,
        avoid_random_special=bool(args.avoid_random_special),
    )

    # 2) Next-event metrics
    next_event_metrics = evaluate_next_event(
        model,
        loader,
        device=device,
        mask_id=int(args.mask_id),
        topks=topks,
        token_id_to_group=token_id_to_group,
    )

    print(f"device={device}")
    print("=== MLM ===")
    for k, v in mlm_metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")

    print("=== Next-Event ===")
    for k, v in next_event_metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()
