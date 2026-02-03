"""
evaluation.rollout Eval----> recursive “predict until discharge/death”: For structured per-patient recursive predictions until discharge/death. 

Zero-shot *recursive* next-token rollout using a *masked* encoder model.

Goal (as discussed):
- For each patient, hold out the last sequence (visit) as ground-truth.
- Condition on all earlier sequences.
- Recursively predict tokens by repeatedly appending a [MASK] at the end and
  filling it with the model's top-1 prediction until an end token is produced
  (e.g., DISCHARGE or DEATH) or a safety limit is reached.
- Primary metric (v1): did we predict the correct end token (discharge vs death)?

Assumptions about input JSONL
----------------------------
This script supports two common formats:

(1) One record per VISIT, grouped by patient_id:
    {
      "patient_id": "...",
      "token_ids": [ ... ],
      "event_type_ids": [ ... ]  
    }

(2) One record per PATIENT with explicit sequences:
    {
      "patient_id": "...",
      "sequences": [
         {"token_ids":[...], "event_type_ids":[...]},   # visit 1
         {"token_ids":[...], "event_type_ids":[...]},   # visit 2
         ...
      ]
    }

If event_type_ids are missing, we will:
- use `--default_event_type_id` for all tokens,
- and also for all predicted tokens.

Run
---
python -m evaluation.rollout_eval \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_baseline.pt
  --end_ids DISCHARGE_ID,DEATH_ID \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_baseline.jsonl 

python -m evaluation.rollout_eval \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_d384.pt
  --end_ids DISCHARGE_ID,DEATH_ID \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_d384.jsonl 

python -m evaluation.rollout_eval \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_n_event_types_7.pt
  --end_ids DISCHARGE_ID,DEATH_ID \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_n_event_types_7.jsonl 

python -m evaluation.rollout_eval \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_n_heads_12_n_layer_6.pt
  --end_ids DISCHARGE_ID,DEATH_ID \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_n_heads_12_n_layer_6.jsonl 

python -m evaluation.rollout_eval \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_n_heads_12.pt 
  --end_ids DISCHARGE_ID,DEATH_ID \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_n_heads_12.jsonl 

python -m evaluation.rollout_eval \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_n_layer_6.pt 
  --end_ids DISCHARGE_ID,DEATH_ID \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_n_layer_6.jsonl

python -m evaluation.rollout_eval \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_recency.pt
  --end_ids DISCHARGE_ID,DEATH_ID \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_recency.jsonl   

python -m evaluation.rollout_eval \
  --jsonl data/eval_val.jsonl \
  --ckpt checkpoints/mlm_span.pt 
  --end_ids DISCHARGE_ID,DEATH_ID \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_span.jsonl         


write per-patient outputs:
  --out_jsonl /tmp/rollout_preds.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import load_jsonl, build_token_id_to_group_from_vocab
from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder



# Helpers
def _extract_logits(model_out: Any) -> torch.Tensor:
    if isinstance(model_out, dict) and "logits" in model_out:
        return model_out["logits"]
    if hasattr(model_out, "logits"):
        return model_out.logits
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        return model_out[0]
    raise ValueError("Could not extract logits from model output")


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return list(x)


@dataclass
class Visit:
    token_ids: List[int]
    event_type_ids: Optional[List[int]] = None


def _parse_patient_records(records: List[Dict[str, Any]], *, patient_key: str) -> Dict[str, List[Visit]]:
    """
    Supports:
      (1) per-patient format:
          {"patient_id":..., "sequences":[{"token_ids":[...], "event_type_ids":[...]}, ...]}
      (2) per-visit format:
          {"patient_id":..., "token_ids":[...], "event_type_ids":[...]}  # one visit per row
    """
    by_pid: Dict[str, List[Visit]] = {}

    for r in records:
        if patient_key not in r:
            raise KeyError(f"Missing '{patient_key}' in record keys={list(r.keys())}")
        pid = str(r[patient_key])

        if "sequences" in r and isinstance(r["sequences"], list):
            visits: List[Visit] = []
            for v in r["sequences"]:
                if not isinstance(v, dict) or "token_ids" not in v:
                    raise ValueError("Each element of 'sequences' must be a dict with key 'token_ids'")
                token_ids = _ensure_list(v["token_ids"])
                et = v.get("event_type_ids", None)
                visits.append(Visit(token_ids=token_ids, event_type_ids=_ensure_list(et) if et is not None else None))
            by_pid[pid] = visits
            continue

        token_ids = _ensure_list(r.get("token_ids", r.get("tokens", None)))
        if not token_ids:
            raise KeyError("Record missing 'token_ids' (or 'tokens')")

        et = r.get("event_type_ids", None)
        by_pid.setdefault(pid, []).append(Visit(token_ids=token_ids, event_type_ids=_ensure_list(et) if et is not None else None))

    return by_pid


def _concat_visits(visits: Sequence[Visit], default_event_type_id: int) -> Tuple[List[int], List[int]]:
    toks: List[int] = []
    ets: List[int] = []
    for v in visits:
        toks.extend(v.token_ids)
        if v.event_type_ids is None:
            ets.extend([int(default_event_type_id)] * len(v.token_ids))
        else:
            if len(v.event_type_ids) != len(v.token_ids):
                raise ValueError("event_type_ids length mismatch with token_ids within a visit")
            ets.extend([int(x) for x in v.event_type_ids])
    return toks, ets


def _first_end_token(seq: Sequence[int], end_ids: Tuple[int, ...]) -> Optional[int]:
    for t in seq:
        if int(t) in end_ids:
            return int(t)
    return None


def _load_ckpt_and_model(ckpt_path: str, device: torch.device) -> Tuple[CompactTransformerEncoder, CompactTransformerConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt):
        raise ValueError("Unsupported checkpoint format. Expected dict with keys: cfg, model_state_dict.")
    cfg = CompactTransformerConfig(**ckpt["cfg"])
    model = CompactTransformerEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def rollout_until_end(
    *,
    model: CompactTransformerEncoder,
    device: torch.device,
    context_tokens: List[int],
    context_event_types: List[int],
    mask_token_id: int,
    pad_token_id: int,
    default_event_type_id: int,
    end_token_ids: Tuple[int, ...],
    max_len: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    topk_sampling: int = 0,  # 0 => greedy
) -> Dict[str, Any]:
    """
    Repeatedly append [MASK] and predict the masked position.
    Stop when predicted token is in end_token_ids or limits reached.
    """
    t0 = time.time()

    # keep room for the appended mask
    tokens = list(context_tokens)[: max_len - 1]
    etypes = list(context_event_types)[: max_len - 1]
    if len(etypes) < len(tokens):
        etypes.extend([int(default_event_type_id)] * (len(tokens) - len(etypes)))

    ended = False
    end_token: Optional[int] = None
    steps = 0
    end_token_probs: List[Dict[str, float]] = []

    while steps < max_new_tokens and len(tokens) < max_len - 1:
        L = len(tokens)

        input_ids = tokens + [int(mask_token_id)]
        attn = [1] * (L + 1)
        ev = etypes + [int(default_event_type_id)]

        pad_n = max_len - len(input_ids)
        input_ids += [int(pad_token_id)] * pad_n
        attn += [0] * pad_n
        ev += [int(default_event_type_id)] * pad_n

        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        attn_t = torch.tensor([attn], dtype=torch.long, device=device)
        ev_t = torch.tensor([ev], dtype=torch.long, device=device)

        out = model(input_ids=input_ids_t, attention_mask=attn_t, event_type_ids=ev_t, labels=None, return_hidden=False)
        logits = _extract_logits(out)  # (1,max_len,V)

        mask_pos = L
        logit_vec = logits[0, mask_pos].float()

        if temperature != 1.0:
            logit_vec = logit_vec / float(temperature)

        probs = F.softmax(logit_vec, dim=-1)

        # log end-token probs per step
        step_probs = {str(tid): float(probs[tid].item()) for tid in end_token_ids if 0 <= tid < probs.numel()}
        end_token_probs.append(step_probs)

        if topk_sampling and topk_sampling > 0:
            # sample from top-k (optional)
            topk = min(int(topk_sampling), probs.numel())
            p, ids = torch.topk(probs, k=topk)
            p = p / p.sum()
            sampled = int(ids[torch.multinomial(p, 1)].item())
            pred = sampled
        else:
            pred = int(torch.argmax(logit_vec).item())

        tokens.append(pred)
        etypes.append(int(default_event_type_id))
        steps += 1

        if pred in end_token_ids:
            ended = True
            end_token = pred
            break

    dt = time.time() - t0
    return {
        "ended": bool(ended),
        "end_token": end_token,
        "steps": int(steps),
        "pred_tokens": tokens[len(context_tokens):],
        "runtime_sec": float(dt),
        "end_token_probs": end_token_probs,
        "context_truncated_to": int(min(len(context_tokens), max_len - 1)),
    }


def run_rollout_eval(
    *,
    model: CompactTransformerEncoder,
    by_pid: Dict[str, List[Visit]],
    device: torch.device,
    mask_token_id: int,
    pad_token_id: int,
    default_event_type_id: int,
    end_token_ids: Tuple[int, ...],
    max_len: int,
    max_new_tokens: int,
    temperature: float,
    topk_sampling: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n_patients = 0
    n_with_gt = 0
    correct_end = 0
    ended_any = 0
    total_steps = 0
    total_runtime = 0.0

    rows: List[Dict[str, Any]] = []

    for pid, visits in by_pid.items():
        if len(visits) == 0:
            continue
        n_patients += 1

        # Hold out last visit as GT if possible
        if len(visits) >= 2:
            ctx_visits = visits[:-1]
            gt_visit = visits[-1]
        else:
            ctx_visits = visits
            gt_visit = None

        ctx_tokens, ctx_et = _concat_visits(ctx_visits, default_event_type_id=int(default_event_type_id))
        gt_tokens = gt_visit.token_ids if gt_visit is not None else None

        pred = rollout_until_end(
            model=model,
            device=device,
            context_tokens=ctx_tokens,
            context_event_types=ctx_et,
            mask_token_id=int(mask_token_id),
            pad_token_id=int(pad_token_id),
            default_event_type_id=int(default_event_type_id),
            end_token_ids=end_token_ids,
            max_len=int(max_len),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            topk_sampling=int(topk_sampling),
        )

        ended_any += int(pred["ended"])
        total_steps += int(pred["steps"])
        total_runtime += float(pred["runtime_sec"])

        gt_end = None
        if gt_tokens is not None:
            n_with_gt += 1
            gt_end = _first_end_token(gt_tokens, end_token_ids)
            if gt_end is not None and pred["end_token"] is not None and int(pred["end_token"]) == int(gt_end):
                correct_end += 1

        row = {
            "patient_id": pid,
            "n_visits": len(visits),
            "has_ground_truth": gt_tokens is not None,
            "gt_end_token": gt_end,
            "pred_end_token": pred["end_token"],
            "ended": pred["ended"],
            "steps": pred["steps"],
            "runtime_sec": pred["runtime_sec"],
            "end_token_probs": pred["end_token_probs"],
            # optional debug:
            # "pred_tokens": pred["pred_tokens"],
        }
        rows.append(row)

    metrics: Dict[str, Any] = {
        "n_patients": int(n_patients),
        "n_with_gt": int(n_with_gt),
        "end_token_correct": int(correct_end),
        "end_token_accuracy": (correct_end / n_with_gt) if n_with_gt > 0 else float("nan"),
        "ended_fraction": (ended_any / max(1, n_patients)),
        "avg_steps": (total_steps / max(1, n_patients)),
        "avg_runtime_sec": (total_runtime / max(1, n_patients)),
    }
    return metrics, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recursive rollout eval: predict until DISCHARGE/DEATH end token.")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--patient_key", type=str, default="patient_id")

    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)

    p.add_argument("--end_ids", type=str, required=True, help="Comma-separated end-token ids, e.g. discharge_id,death_id")
    p.add_argument("--max_new_tokens", type=int, default=256)

    # optional stochasticity
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topk_sampling", type=int, default=0, help="0=greedy, else sample from top-k")

    p.add_argument("--out_jsonl", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = _load_ckpt_and_model(args.ckpt, device=device)

    if int(args.max_len) > int(cfg.max_len):
        raise ValueError(
            f"--max_len={args.max_len} exceeds encoder cfg.max_len={cfg.max_len}. "
            "Either retrain with larger max_len or reduce eval max_len."
        )

    end_ids = tuple(int(x) for x in args.end_ids.split(",") if x.strip())
    if len(end_ids) < 1:
        raise ValueError("--end_ids must contain at least one integer token id")

    records = load_jsonl(args.jsonl)
    by_pid = _parse_patient_records(records, patient_key=args.patient_key)

    out_f = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

    metrics, rows = run_rollout_eval(
        model=model,
        by_pid=by_pid,
        device=device,
        mask_token_id=int(args.mask_id),
        pad_token_id=int(args.pad_id),
        default_event_type_id=int(args.default_event_type_id),
        end_token_ids=end_ids,
        max_len=int(args.max_len),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        topk_sampling=int(args.topk_sampling),
    )

    if out_f is not None:
        for r in rows:
            out_f.write(json.dumps(r) + "\n")
        out_f.close()

    print(f"device={device}")
    print(f"n_patients={metrics['n_patients']} n_with_gt={metrics['n_with_gt']}")
    if metrics["n_with_gt"] > 0:
        print(f"end_token_accuracy={metrics['end_token_accuracy']:.4f} ({metrics['end_token_correct']}/{metrics['n_with_gt']})")
    print(f"ended_fraction={metrics['ended_fraction']:.4f}")
    print(f"avg_steps={metrics['avg_steps']:.2f}")
    print(f"avg_runtime_sec={metrics['avg_runtime_sec']:.4f}")


if __name__ == "__main__":
    main()
