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

from __future__ import annotations

import argparse
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
    topk_accuracy_from_logits,
    build_token_id_to_group_from_vocab,
    block_top1_acc_from_logits,
    block_topk_acc_from_logits,
)

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Next-event evaluation: predict next token at end of sequence (zero-shot).")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--vocab_path", type=str, default=None, help="Path to vocabulary.json (for event-type eval).")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)
    p.add_argument("--topk", type=str, default="1,5,10")
    return p.parse_args()



@torch.no_grad()
def evaluate_next_event(
    model: CompactTransformerEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    mask_id: int,
    topks: List[int],
    token_id_to_group: Optional[Dict[int, int]] = None,
) -> Dict[str, float]:
    model.eval()

    totals = {k: {"correct": 0, "total": 0} for k in topks}
    mrr_vals: List[float] = []

    et_top1_vals: List[float] = []
    et_topk_vals = {k: [] for k in topks}

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ev = batch["event_type_ids"].to(device)

        B, L = input_ids.shape
        labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long, device=device)

        x = input_ids.clone()
        for b in range(B):
            valid = (attn[b] == 1).nonzero(as_tuple=False).view(-1)
            if valid.numel() < 1:
                continue
            last_idx = int(valid[-1].item())
            y = int(x[b, last_idx].item())
            labels[b, last_idx] = y
            x[b, last_idx] = int(mask_id)

        out = model(input_ids=x, attention_mask=attn, event_type_ids=ev, labels=labels, return_hidden=False)
        logits = out["logits"]

        for k in topks:
            res = topk_accuracy_from_logits(logits, labels, k=k)
            totals[k]["correct"] += res.correct
            totals[k]["total"] += res.total

        mrr_vals.append(mrr_from_logits(logits, labels))

        if token_id_to_group is not None:
            et_top1_vals.append(block_top1_acc_from_logits(logits, labels, token_id_to_group))
            for k in topks:
                et_topk_vals[k].append(block_topk_acc_from_logits(logits, labels, token_id_to_group, k=k))

    metrics: Dict[str, float] = {}
    for k in topks:
        c = totals[k]["correct"]
        t = totals[k]["total"]
        metrics[f"next_event_top{k}_acc"] = c / t if t else float("nan")

    metrics["next_event_mrr"] = float(sum(mrr_vals) / max(1, len(mrr_vals)))

    if token_id_to_group is not None and len(et_top1_vals) > 0:
        metrics["next_event_event_type_top1_acc"] = float(sum(et_top1_vals) / len(et_top1_vals))
        for k in topks:
            xs = et_topk_vals[k]
            metrics[f"next_event_event_type_top{k}_acc"] = float(sum(xs) / len(xs)) if xs else float("nan")

    return metrics


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)
    ds = ClinicalSequenceDataset(records, max_len=args.max_len, pad_id=args.pad_id, default_event_type_id=args.default_event_type_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    topks = [int(x) for x in args.topk.split(",") if x.strip()]

    token_id_to_group = None
    if args.vocab_path:
        from vocabulary import Vocabulary
        vocab = Vocabulary.load(args.vocab_path)
        token_id_to_group = build_token_id_to_group_from_vocab(vocab)

    metrics = evaluate_next_event(
        model,
        loader,
        device=device,
        mask_id=int(args.mask_id),
        topks=topks,
        token_id_to_group=token_id_to_group,
    )

    print(f"device={device}")
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()
