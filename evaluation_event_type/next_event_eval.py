from __future__ import annotations
'''evaluation/next_event_eval.py.  ----> for event type 
PATH:

python -m evaluation.next_event_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_baseline.pt \
  --topk 1,5,10 \
  --vocab_path data/vocabulary.json


'''
"""

Next-event evaluation on VOCABULARY BLOCKS:
- Predict next token, but score as correct if predicted token's block == true token's block.

Adds window sampling:
  --sample_windows
  --keep_prefix_n
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

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder

from evaluation.clinical_eval_utils import (
    build_token_id_to_block_id_from_vocab,
    NextEventDataset,
    load_jsonl
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Next-event block evaluation with optional window sampling.")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)

    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--default_event_type_id", type=int, default=1)

    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--seed", type=int, default=0)

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
def evaluate_next_event_block(
    model: CompactTransformerEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    topks: List[int],
    token_id_to_block: Dict[int, int],
) -> Dict[str, Any]:
    model.eval()

    total = 0
    correct_topk = {k: 0 for k in topks}

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ev = batch["event_type_ids"].to(device)
        target_id = batch["target_id"].to(device)  # [B]

        out = model(input_ids=input_ids, attention_mask=attn, event_type_ids=ev, labels=None, return_hidden=False)
        logits = out["logits"]  # [B, T, V]

        last_pos = (attn.sum(dim=1) - 1).clamp(min=0)
        last_logits = logits[torch.arange(logits.size(0), device=device), last_pos, :]  # [B, V]

        # true blocks
        true_blocks = torch.tensor(
            [token_id_to_block.get(int(t.item()), -1) for t in target_id],
            device=device,
            dtype=torch.long,
        )

        for k in topks:
            topk_ids = last_logits.topk(k, dim=1).indices  # [B, k]
            # map to blocks
            pred_blocks = torch.empty_like(topk_ids, dtype=torch.long)
            for i in range(topk_ids.size(0)):
                for j in range(topk_ids.size(1)):
                    pred_blocks[i, j] = int(token_id_to_block.get(int(topk_ids[i, j].item()), -1))

            correct = (pred_blocks == true_blocks.unsqueeze(1)).any(dim=1).sum().item()
            correct_topk[k] += int(correct)

        total += int(target_id.numel())

    metrics: Dict[str, Any] = {"next_event_total": int(total)}
    denom = max(1, total)
    for k in topks:
        metrics[f"next_event_block_top{k}_acc"] = float(correct_topk[k] / denom)
    return metrics


def main() -> None:
    args = parse_args()

    model = _load_ckpt_and_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)
    topks = [int(x) for x in args.topk.split(",") if x.strip()]

    from vocabulary import Vocabulary
    vocab = Vocabulary.load(args.vocab_path)
    token_id_to_block, block_id_to_name, _ = build_token_id_to_block_id_from_vocab(vocab)

    ds = NextEventDataset(
        records,
        max_len=args.max_len,
        pad_id=args.pad_id,
        default_event_type_id=args.default_event_type_id,
        sample_windows=bool(args.sample_windows),
        keep_prefix_n=int(args.keep_prefix_n),
        seed=int(args.seed),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_next_event_block(model, loader, device=device, topks=topks, token_id_to_block=token_id_to_block)

    print(f"device={device}")
    print(f"sample_windows={bool(args.sample_windows)} keep_prefix_n={int(args.keep_prefix_n)} max_len={int(args.max_len)}")
    print("block_id_to_name=", block_id_to_name)
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()