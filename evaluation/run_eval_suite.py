'''
Docstring for evaluation.run_eval_suite. 

PATH: 

    python -m evaluation.run_eval_suite \
        --jsonl data/eval.jsonl \
        --ckpt checkpoints/mlm_best.pt \
        --batch_size 32 \
        --max_len 128 \
        --pad_id 0 \
        --mask_id 1 \
        --p_mlm 0.15 \
        --topk 1,5,10



'''







from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader


# -------------------------------------------------
# Ensure repo root is importable so "transformer" works.
# -------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from evaluation.clinical_eval_utils import (  # noqa: E402
    ClinicalSequenceDataset,
    load_jsonl,
    masking_policy_expected_corruption,
)
from evaluation.mlm_eval import run_mlm_eval  # noqa: E402
from evaluation.next_event_eval import run_next_event_eval  # noqa: E402
from compact_transformer_encoder import (  # noqa: E402
    CompactTransformerConfig,
    CompactTransformerEncoder,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluation suite: MLM top-k (token+span), next-event top-k; prints analytical masking expectations."
    )
    p.add_argument("--jsonl", type=str, required=True, help="Eval JSONL with demo_tokens/event_tokens")
    p.add_argument("--ckpt", type=str, required=True, help="Pretrained MLM checkpoint (.pt)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--p_mlm", type=float, default=0.15)
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--span_min", type=int, default=3)
    p.add_argument("--span_max", type=int, default=10)
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

    vocab_size = args.vocab_size
    if vocab_size is None:
        if hasattr(cfg, "vocab_size"):
            vocab_size = int(cfg.vocab_size)
        else:
            raise ValueError("--vocab_size must be provided if not present in checkpoint cfg")

    ks = tuple(int(x) for x in args.topk.split(",") if x.strip())

    records = load_jsonl(args.jsonl)
    ds = ClinicalSequenceDataset(records, max_len=args.max_len, pad_id=args.pad_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    print(f"device={device}")
    print("expected_corruption (80/10/10):", {k: round(v, 6) for k, v in masking_policy_expected_corruption(args.p_mlm).items()})

    # MLM eval: token masking
    m_token = run_mlm_eval(
        model=model,
        loader=loader,
        device=device,
        vocab_size=int(vocab_size),
        mask_token_id=int(args.mask_id),
        pad_token_id=int(args.pad_id),
        p_mlm=float(args.p_mlm),
        strategy="token",
        span_len_range=(int(args.span_min), int(args.span_max)),
        topk=ks,
    )
    print("\n[MLM: token masking]")
    print(f"n_masked={m_token['n_masked']} loss={m_token['mlm_loss']:.6f} ppl={m_token['mlm_ppl']:.4f}")
    for k in ks:
        print(f"top{k}={m_token[f'top{k}']:.4f}")

    # MLM eval: span masking
    m_span = run_mlm_eval(
        model=model,
        loader=loader,
        device=device,
        vocab_size=int(vocab_size),
        mask_token_id=int(args.mask_id),
        pad_token_id=int(args.pad_id),
        p_mlm=float(args.p_mlm),
        strategy="span",
        span_len_range=(int(args.span_min), int(args.span_max)),
        topk=ks,
    )
    print("\n[MLM: span masking]")
    print(f"n_masked={m_span['n_masked']} loss={m_span['mlm_loss']:.6f} ppl={m_span['mlm_ppl']:.4f}")
    for k in ks:
        print(f"top{k}={m_span[f'top{k}']:.4f}")

    # Next-event eval
    ne = run_next_event_eval(
        model=model,
        loader=loader,
        device=device,
        mask_token_id=int(args.mask_id),
        topk=ks,
    )
    print("\n[Next-event prediction: mask last event token]")
    print(f"n_pred={ne['n_pred']} loss={ne['next_event_loss']:.6f} ppl={ne['next_event_ppl']:.4f}")
    for k in ks:
        print(f"top{k}={ne[f'top{k}']:.4f}")

    print(
        "\nMortality downstream probe: use downstream/mortality_train.py (linear probe via --freeze_encoder)."
    )


if __name__ == "__main__":
    main()
