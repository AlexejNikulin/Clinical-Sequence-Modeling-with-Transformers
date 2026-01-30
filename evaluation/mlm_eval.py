'''
Docstring for evaluation.mlm_eval

ls -la data/eval.jsonl
ls -la data | head
find . -maxdepth 4 -name "eval.jsonl" -print


PATH: 
    Token masking:
        python -m evaluation.mlm_eval \
            --jsonl data/eval.jsonl \
            --ckpt checkpoints/mlm_d384.pt \
            --p_mlm 0.15 \
            --strategy token \
            --topk 1,5,10 2>&1 | tee /tmp/mlm_eval_err.txt
            tail -n 80 /tmp/mlm_eval_err.txt


        python -m evaluation.mlm_eval \
            --jsonl data/eval.jsonl \
            --ckpt checkpoints/mlm_recency.pt \
            --p_mlm 0.15 \
            --strategy token \
            --topk 1,5,10

       

    span masking:     
        python -m evaluation.mlm_eval \
            --jsonl data/eval.jsonl \
            --ckpt checkpoints/mlm_span.pt \
            --p_mlm 0.15 \
            --strategy token \
            --topk 1,5,10
        
   


'''

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage*",
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from clinical_eval_utils import (
    IGNORE_INDEX,
    ClinicalSequenceDataset,
    TopKResult,
    load_jsonl,
    topk_accuracy_from_logits,
    masking_policy_expected_corruption,
)
from mlm_masking import mlm_mask_801010, mlm_mask_span_801010
from compact_transformer_encoder import (
    CompactTransformerConfig,
    CompactTransformerEncoder,
)




def _extract_logits(model_out: Any) -> torch.Tensor:
    """Robustly extract logits across slightly different forward() conventions."""
    if isinstance(model_out, dict) and "logits" in model_out:
        return model_out["logits"]
    if hasattr(model_out, "logits"):
        return model_out.logits
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        return model_out[0]
    raise ValueError("Could not extract logits from model output")


@torch.no_grad()
def run_mlm_eval(
    *,
    model: CompactTransformerEncoder,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    mask_token_id: int,
    pad_token_id: int,
    p_mlm: float,
    strategy: str,
    span_len_range: Tuple[int, int] = (3, 10),
    never_mask_token_ids: Optional[Iterable[int]] = None,
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

        if strategy == "token":
            masked_ids, labels = mlm_mask_801010(
                input_ids,
                attention_mask,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,
                p_mlm=p_mlm,
                pad_token_id=pad_token_id,
                never_mask_token_ids=never_mask_token_ids,
            )
        elif strategy == "span":
            masked_ids, labels = mlm_mask_span_801010(
                input_ids,
                attention_mask,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,
                p_mlm=p_mlm,
                span_len_range=span_len_range,
                pad_token_id=pad_token_id,
                never_mask_token_ids=never_mask_token_ids,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        out = model(
            input_ids=masked_ids,
            attention_mask=attention_mask,
            event_type_ids=event_type_ids,
            labels=None,
            return_hidden=False,
        )

        logits = _extract_logits(out)  # [B, L, V]

        # Loss on masked positions only
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        n_pred = int((labels != IGNORE_INDEX).sum().item())
        total_loss += float(loss.item())
        total_pred += n_pred

        # Top-k hits
        for k in topk:
            r: TopKResult = topk_accuracy_from_logits(logits, labels, k=k)
            if r.n > 0:
                topk_hits[k] += int(round(r.accuracy * r.n))

    metrics: Dict[str, Any] = {
        "strategy": strategy,
        "p_mlm": float(p_mlm),
        "n_masked": int(total_pred),
        "mlm_loss": (total_loss / max(1, total_pred)),
        "mlm_ppl": float(torch.exp(torch.tensor(total_loss / max(1, total_pred))).item()),
    }
    for k in topk:
        acc = topk_hits[k] / max(1, total_pred)
        metrics[f"top{k}"] = float(acc)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MLM masked-token prediction (Top-k + loss).")
    p.add_argument("--jsonl", type=str, required=True, help="Eval JSONL with demo_tokens/event_tokens")
    p.add_argument("--ckpt", type=str, required=True, help="Pretrained MLM checkpoint (.pt)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--vocab_size", type=int, required=False, default=None)
    p.add_argument("--p_mlm", type=float, default=0.15)
    p.add_argument("--strategy", choices=["token", "span"], default="token")
    p.add_argument("--span_min", type=int, default=3)
    p.add_argument("--span_max", type=int, default=10)
    p.add_argument("--topk", type=str, default="1,5,10", help="Comma-separated k list")
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

    # Prefer vocab_size from checkpoint/config when available.
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

    metrics = run_mlm_eval(
        model=model,
        loader=loader,
        device=device,
        vocab_size=int(vocab_size),
        mask_token_id=int(args.mask_id),
        pad_token_id=int(args.pad_id),
        p_mlm=float(args.p_mlm),
        strategy=args.strategy,
        span_len_range=(int(args.span_min), int(args.span_max)),
        topk=ks,
    )

    # Analytical expectations for corruption under 80/10/10
    exp = masking_policy_expected_corruption(float(args.p_mlm))

    print(f"device={device}")
    print(f"strategy={metrics['strategy']} p_mlm={metrics['p_mlm']}")
    print("expected_corruption:", {k: round(v, 6) for k, v in exp.items()})
    print(f"n_masked={metrics['n_masked']} mlm_loss={metrics['mlm_loss']:.6f} mlm_ppl={metrics['mlm_ppl']:.4f}")
    for k in ks:
        print(f"top{k}={metrics[f'top{k}']:.4f}")


if __name__ == "__main__":
    main()
