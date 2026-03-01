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

# - mask the LAST valid token in each sequence
# - evaluate whether predicted token belongs to the same VOCAB BLOCK (event type)
# - prints block top1/topk and token-level MRR (optional)

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Next-event evaluation by VOCAB BLOCK (type), not exact tokens.")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)
    p.add_argument("--topk", type=str, default="1,5,10")

    # Optional: window sampling for long sequences
    p.add_argument("--sample_windows", action="store_true")
    p.add_argument("--keep_prefix_n", type=int, default=0)

    # Reproducibility
    p.add_argument("--seed", type=int, default=13)
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
def evaluate_next_event_block(
    model: CompactTransformerEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    mask_id: int,
    topks: List[int],
    token_id_to_block: Dict[int, int],
) -> Dict[str, float]:
    model.eval()

    block_top1_vals: List[float] = []
    block_topk_vals: Dict[int, List[float]] = {k: [] for k in topks}
    mrr_vals: List[float] = []

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ev = batch["event_type_ids"].to(device)

        B, L = input_ids.shape
        labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long, device=device)

        # Mask the last valid position in each sequence
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

        block_top1_vals.append(block_top1_acc_from_logits(logits, labels, token_id_to_block))
        for k in topks:
            block_topk_vals[k].append(block_topk_acc_from_logits(logits, labels, token_id_to_block, k=k))

        mrr_vals.append(mrr_from_logits(logits, labels))

    metrics: Dict[str, float] = {}
    metrics["next_event_type_top1_acc"] = float(sum(block_top1_vals) / max(1, len(block_top1_vals)))
    for k in topks:
        metrics[f"next_event_type_top{k}_acc"] = float(sum(block_topk_vals[k]) / max(1, len(block_topk_vals[k])))
    metrics["next_event_mrr_token_level"] = float(sum(mrr_vals) / max(1, len(mrr_vals)))
    return metrics


def main() -> None:
    args = parse_args()

    # Reproducible window sampling (if enabled)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model = _load_ckpt_and_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)
    ds = ClinicalSequenceDataset(
        records,
        max_len=args.max_len,
        pad_id=args.pad_id,
        default_event_type_id=args.default_event_type_id,
        sample_windows=bool(args.sample_windows),
        keep_prefix_n=int(args.keep_prefix_n),
        seed=int(args.seed),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    topks = [int(x) for x in args.topk.split(",") if x.strip()]

    from vocabulary import Vocabulary
    vocab = Vocabulary.load(args.vocab_path)
    token_id_to_block, block_id_to_name, _ = build_token_id_to_block_id_from_vocab(vocab)

    metrics = evaluate_next_event_block(
        model,
        loader,
        device=device,
        mask_id=int(args.mask_id),
        topks=topks,
        token_id_to_block=token_id_to_block,
    )

    print(f"device={device}")
    print("block_id_to_name=", block_id_to_name)
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()