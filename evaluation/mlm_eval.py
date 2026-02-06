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

from __future__ import annotations

import argparse
import os
import sys
import json
from tqdm import tqdm

from typing import Any, Dict, List, Optional

import torch

from torch.utils.data import DataLoader



REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import (
    ClinicalSequenceDataset,
    IGNORE_INDEX,
    TopKResult,
    load_jsonl,
    mrr_from_logits,
    topk_accuracy_from_logits,
    build_token_id_to_group_from_vocab,
    event_type_top1_acc_from_logits,
    event_type_topk_acc_from_logits,
)

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder

from mlm_masking import mlm_mask_801010


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLM evaluation (Top-k + MRR). If JSONL has no labels, creates MLM labels on-the-fly."
    )
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    # NEW: for event-type/group eval from Vocabulary
    p.add_argument("--vocab_path", type=str, default=None, help="Path to vocabulary.json (for event-type eval).")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)
    p.add_argument("--topk", type=str, default="1,5,10")

    p.add_argument("--p_mlm", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--avoid_random_special", action="store_true")
    p.add_argument("--use_on_the_fly_masking", action="store_true", help="Force on-the-fly masking even if labels exist.")
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


@torch.no_grad()
def evaluate(
    model: CompactTransformerEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    topks: List[int],
    vocab_size: int,
    pad_id: int,
    mask_id: int,
    p_mlm: float,
    use_on_the_fly_masking: bool,
    generator: torch.Generator,
    token_id_to_group: Optional[Dict[int, int]] = None,
    avoid_random_special: bool = False,
) -> Dict[str, Any]:
    model.eval()

    totals = {k: {"correct": 0, "total": 0} for k in topks}
    mrr_vals: List[float] = []

    et_top1_vals: List[float] = []
    et_topk_vals = {k: [] for k in topks}

    total_eval_positions = 0
    skipped_batches = 0

    avoid_random_ids = None
    if avoid_random_special:
        avoid_random_ids = [int(pad_id), int(mask_id)]

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
        total_eval_positions += n_pos

        out = model(input_ids=x, attention_mask=attn, event_type_ids=ev, labels=labels, return_hidden=False)
        logits = out["logits"]

        for k in topks:
            res = topk_accuracy_from_logits(logits, labels, k=k)
            totals[k]["correct"] += res.correct
            totals[k]["total"] += res.total

        mrr_vals.append(mrr_from_logits(logits, labels))

        if token_id_to_group is not None:
            et_top1_vals.append(event_type_top1_acc_from_logits(logits, labels, token_id_to_group))
            for k in topks:
                et_topk_vals[k].append(event_type_topk_acc_from_logits(logits, labels, token_id_to_group, k=k))

    metrics: Dict[str, Any] = {}
    for k in topks:
        c = totals[k]["correct"]
        t = totals[k]["total"]
        metrics[f"mlm_top{k}_acc"] = c / t if t else float("nan")

    metrics["mlm_mrr"] = float(sum(mrr_vals) / len(mrr_vals)) if mrr_vals else float("nan")
    metrics["mlm_total_eval_positions"] = int(total_eval_positions)
    metrics["mlm_skipped_batches_zero_eval_positions"] = int(skipped_batches)

    if token_id_to_group is not None and len(et_top1_vals) > 0:
        metrics["mlm_event_type_top1_acc"] = float(sum(et_top1_vals) / len(et_top1_vals))
        for k in topks:
            xs = et_topk_vals[k]
            metrics[f"mlm_event_type_top{k}_acc"] = float(sum(xs) / len(xs)) if xs else float("nan")

    return metrics


def _load_ckpt_and_model(ckpt_path: str) -> CompactTransformerEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt:
        cfg = CompactTransformerConfig(**ckpt["cfg"])
        model = CompactTransformerEncoder(cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model
    # fallback: raw state_dict checkpoints (less ideal)
    raise ValueError("Unsupported checkpoint format. Expected dict with keys: cfg, model_state_dict.")


def main() -> None:
    args = parse_args()

    model = _load_ckpt_and_model(args.ckpt)
    cfg = model.cfg if hasattr(model, "cfg") else None  # some implementations store cfg on model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)
    # ---- sanity check: token ids must fit ckpt vocab ----
    vocab_size = int(model.cfg.vocab_size) if hasattr(model, "cfg") else int(model.token_emb.num_embeddings)

    max_id = -1
    min_id = 10**18
    for r in records[:5000]:
        ids = r.get("token_ids") or r.get("ids") or r.get("input_ids") or r.get("tokens")
        if ids:
            max_id = max(max_id, max(ids))
            min_id = min(min_id, min(ids))

    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"JSONL token id range [{min_id},{max_id}] but checkpoint vocab_size={vocab_size}. "
            "Use a matching checkpoint OR rebuild the JSONL with the checkpoint's vocabulary."
        )

    
    # ---- sanity check: token ids must fit ckpt vocab ----
    max_id = -1
    for r in records[:5000]:  # reicht meist, sonst entferne die Begrenzung
        ids = r.get("token_ids") or r.get("ids") or r.get("input_ids") or r.get("tokens")
        if ids:
            max_id = max(max_id, max(ids))
    if max_id >= int(cfg.vocab_size):
        raise ValueError(
            f"JSONL contains token id {max_id} but checkpoint vocab_size={cfg.vocab_size}. "
            "You are using a dataset/vocabulary that does not match this checkpoint."
        )

    ds = ClinicalSequenceDataset(
        records,
        max_len=args.max_len,
        pad_id=args.pad_id,
        default_event_type_id=args.default_event_type_id,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    topks = [int(x) for x in args.topk.split(",") if x.strip()]

    # Decide on-the-fly masking
    has_labels = _jsonl_has_labels(args.jsonl, max_check=200)
    use_on_the_fly = bool(args.use_on_the_fly_masking) or (not has_labels)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    token_id_to_group = None
    if args.vocab_path:
        from vocabulary import Vocabulary
        vocab = Vocabulary.load(args.vocab_path)
        token_id_to_group = build_token_id_to_group_from_vocab(vocab)

    # vocab_size from cfg
    vocab_size = int(cfg.vocab_size) if cfg is not None else None
    if vocab_size is None:
        raise ValueError("Could not infer vocab_size from model cfg.")

    metrics = evaluate(
        model,
        loader,
        device=device,
        topks=topks,
        vocab_size=vocab_size,
        pad_id=int(args.pad_id),
        mask_id=int(args.mask_id),
        p_mlm=float(args.p_mlm),
        use_on_the_fly_masking=use_on_the_fly,
        generator=gen,
        token_id_to_group=token_id_to_group,
        avoid_random_special=bool(args.avoid_random_special),
    )

    print(f"device={device}")
    print(f"use_on_the_fly_masking={use_on_the_fly} (jsonl_has_labels={has_labels})")
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()
