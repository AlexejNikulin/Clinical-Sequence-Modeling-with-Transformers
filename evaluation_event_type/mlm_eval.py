# evaluation/mlm_eval.py. -----> for event type
'''
This prints: mlm_block_top1_acc and mlm_block_top{k}_acc
(no exact token accuracy needed; we can keep it, but we said “NOT EXACT TOKENS”).'''

'''
python -m evaluation.mlm_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_baseline.pt \
  --topk 1,5,10 \
  --vocab_path data/vocabulary.json

python -m evaluation.mlm_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_n_event_types_7.pt \
  --topk 1,5,10 \
  --vocab_path data/vocabulary.json 

python -m evaluation.mlm_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_n_heads_12.pt \
  --topk 1,5,10 \
  --vocab_path data/vocabulary.json
'''


from __future__ import annotations

import argparse
import os
import sys
import json
from typing import Any, Dict, List, Optional

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
    build_token_id_to_block_id_from_vocab,
    block_top1_acc_from_logits,
    block_topk_acc_from_logits,
)

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder
from mlm_masking import mlm_mask_801010


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLM evaluation by NEXT-EVENT TYPE (Vocabulary block), not exact tokens.")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary.json (required for block eval).")

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
    model.eval()

    avoid_random_ids = None
    if avoid_random_special:
        avoid_random_ids = [int(pad_id), int(mask_id)]

    block_top1_vals: List[float] = []
    block_topk_vals: Dict[int, List[float]] = {k: [] for k in topks}
    mrr_vals: List[float] = []

    total_eval_positions = 0
    skipped_batches = 0

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
        total_eval_positions += n_pos

        out = model(input_ids=x, attention_mask=attn, event_type_ids=ev, labels=labels, return_hidden=False)
        logits = out["logits"]

        block_top1_vals.append(block_top1_acc_from_logits(logits, labels, token_id_to_block))
        for k in topks:
            block_topk_vals[k].append(block_topk_acc_from_logits(logits, labels, token_id_to_block, k=k))

        # MRR is still token-level rank-based; keep if we want
        mrr_vals.append(mrr_from_logits(logits, labels))

    metrics: Dict[str, Any] = {}
    metrics["mlm_total_eval_positions"] = int(total_eval_positions)
    metrics["mlm_skipped_batches_zero_eval_positions"] = int(skipped_batches)

    metrics["mlm_block_top1_acc"] = float(sum(block_top1_vals) / max(1, len(block_top1_vals)))
    for k in topks:
        xs = block_topk_vals[k]
        metrics[f"mlm_block_top{k}_acc"] = float(sum(xs) / max(1, len(xs)))

    metrics["mlm_mrr_token_level"] = float(sum(mrr_vals) / max(1, len(mrr_vals)))

    return metrics


def main() -> None:
    args = parse_args()

    model = _load_ckpt_and_model(args.ckpt)
    cfg = model.cfg if hasattr(model, "cfg") else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)

    vocab_size = int(cfg.vocab_size) if cfg is not None and hasattr(cfg, "vocab_size") else int(model.token_emb.num_embeddings)

    ds = ClinicalSequenceDataset(
        records,
        max_len=args.max_len,
        pad_id=args.pad_id,
        default_event_type_id=args.default_event_type_id,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    topks = [int(x) for x in args.topk.split(",") if x.strip()]

    has_labels = _jsonl_has_labels(args.jsonl, max_check=200)
    use_on_the_fly = bool(args.use_on_the_fly_masking) or (not has_labels)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    from vocabulary import Vocabulary
    vocab = Vocabulary.load(args.vocab_path)
    token_id_to_block, block_id_to_name, block_name_to_id = build_token_id_to_block_id_from_vocab(vocab)

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
