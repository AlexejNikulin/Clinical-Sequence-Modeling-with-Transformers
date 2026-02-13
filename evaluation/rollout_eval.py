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
      "event_type_ids": [ ... ]  ---op
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
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_baseline.pt \
  --disch_start 77000 --disch_end 77010 \
  --death_start 77010 --death_end 77011 \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_baseline.jsonl

python -m evaluation.rollout_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_d384.pt \
  --disch_start 77000 --disch_end 77010 \
  --death_start 77010 --death_end 77011 \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_d384.jsonl

python -m evaluation.rollout_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_n_event_types_7.pt
  --disch_start 77000 --disch_end 77010 \
  --death_start 77010 --death_end 77011 \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_n_event_types_7.jsonl 

python -m evaluation.rollout_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_n_heads_12_n_layer_6.pt
  --disch_start 77000 --disch_end 77010 \
  --death_start 77010 --death_end 77011 \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_n_heads_12_n_layer_6.jsonl 

python -m evaluation.rollout_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_n_heads_12.pt 
  --disch_start 77000 --disch_end 77010 \
  --death_start 77010 --death_end 77011 \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_n_heads_12.jsonl 

python -m evaluation.rollout_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_n_layer_6.pt 
  --disch_start 77000 --disch_end 77010 \
  --death_start 77010 --death_end 77011 \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_n_layer_6.jsonl

python -m evaluation.rollout_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_recency.pt
  --disch_start 77000 --disch_end 77010 \
  --death_start 77010 --death_end 77011 \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_recency.jsonl   

python -m evaluation.rollout_eval \
  --jsonl data/test_ids.jsonl \
  --ckpt checkpoints/mlm_span.pt 
  --disch_start 77000 --disch_end 77010 \
  --death_start 77010 --death_end 77011 \
  --max_len 256 \
  --max_new_tokens 256 \
  --out_jsonl results/rollout_mlm_span.jsonl         


write per-patient outputs:
  --out_jsonl /tmp/rollout_preds.jsonl
"""

# --- rollout_eval.py (patch-style: clean change) ---

from __future__ import annotations

import argparse
import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import load_jsonl
from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


# -------------------------
# Helpers
# -------------------------
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
      (3) wrapped format:
          {"ids": {...}}   # common in some pipelines
      (4) minimal tokens-only format:
          {"ids": [ ... ]}  # no explicit patient id -> we synthesize one
    """
    by_pid: Dict[str, List[Visit]] = {}

    # helper: pick a patient id from common keys
    def _get_pid(rec: Dict[str, Any], fallback: str) -> str:
        for k in [patient_key, "patient_id", "subject_id", "pid"]:
            if k in rec:
                return str(rec[k])
        return fallback

    # helper: pick token list from common keys
    def _get_tokens(rec: Dict[str, Any]) -> List[int]:
        for k in ["token_ids", "tokens", "input_ids"]:
            if k in rec and rec[k] is not None:
                return _ensure_list(rec[k])
        # allow "ids" to be token ids if it is a list
        if "ids" in rec and isinstance(rec["ids"], list):
            return _ensure_list(rec["ids"])
        return []

    for i, r0 in enumerate(records):
        r = r0

        # unwrap {"ids": {...}} if present
        if "ids" in r and isinstance(r["ids"], dict):
            r = r["ids"]

        pid = _get_pid(r, fallback=f"row_{i:06d}")

        # case: per-patient with sequences
        if "sequences" in r and isinstance(r["sequences"], list):
            visits: List[Visit] = []
            for v in r["sequences"]:
                if not isinstance(v, dict):
                    raise ValueError("Each element of 'sequences' must be a dict")
                token_ids = _ensure_list(v.get("token_ids", v.get("tokens", v.get("input_ids", v.get("ids", None)))))
                if not token_ids:
                    raise ValueError("Each visit in 'sequences' must contain token ids (token_ids/tokens/input_ids/ids).")
                et = v.get("event_type_ids", None)
                visits.append(Visit(token_ids=token_ids, event_type_ids=_ensure_list(et) if et is not None else None))
            by_pid[pid] = visits
            continue

        # case: per-visit (single visit row)
        token_ids = _get_tokens(r)
        if not token_ids:
            raise KeyError(
                "Record missing token ids. Expected one of: token_ids, tokens, input_ids, or ids(list). "
                f"Got keys={list(r.keys())}"
            )

        et = r.get("event_type_ids", None)
        by_pid.setdefault(pid, []).append(
            Visit(token_ids=token_ids, event_type_ids=_ensure_list(et) if et is not None else None)
        )

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


def _load_ckpt_and_model(ckpt_path: str, device: torch.device) -> Tuple[CompactTransformerEncoder, CompactTransformerConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt):
        raise ValueError("Unsupported checkpoint format. Expected dict with keys: cfg, model_state_dict.")
    cfg = CompactTransformerConfig(**ckpt["cfg"])
    model = CompactTransformerEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg



# NEW: End-token definition by ranges

@dataclass(frozen=True)
class EndTokenSpec:
    """
    Defines how to detect the "end" during rollout and in ground-truth.

    Common case in our project:
      - discharge tokens live in [disch_start, disch_end_exclusive)
      - death tokens live in [death_start, death_end_exclusive) or can be a single id

    we can provide:
      - discharge range (start inclusive, end exclusive)
      - death range (start inclusive, end exclusive)
    """
    disch_start: int
    disch_end_exclusive: int
    death_start: int
    death_end_exclusive: int

    def classify(self, token_id: int) -> Optional[str]:
        tid = int(token_id)
        if self.disch_start <= tid < self.disch_end_exclusive:
            return "DISCHARGE"
        if self.death_start <= tid < self.death_end_exclusive:
            return "DEATH"
        return None

    def is_end(self, token_id: int) -> bool:
        return self.classify(token_id) is not None


def _first_end_label(seq: Sequence[int], end_spec: EndTokenSpec) -> Optional[str]:
    for t in seq:
        lab = end_spec.classify(int(t))
        if lab is not None:
            return lab
    return None


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
    end_spec: EndTokenSpec,          # <-- NEW
    max_len: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    topk_sampling: int = 0,  # 0 => greedy
) -> Dict[str, Any]:
    """
    Repeatedly append [MASK] and predict the masked position.
    Stop when predicted token is an end token according to end_spec, or limits reached.
    """
    t0 = time.time()

    # keep room for the appended mask
    tokens = list(context_tokens)[: max_len - 1]
    etypes = list(context_event_types)[: max_len - 1]
    if len(etypes) < len(tokens):
        etypes.extend([int(default_event_type_id)] * (len(tokens) - len(etypes)))

    ended = False
    end_token: Optional[int] = None
    end_label: Optional[str] = None
    steps = 0
    end_token_probs: List[Dict[str, float]] = []

    # For logging probabilities, we log the *aggregate* probability mass of discharge-range and death-range.
    # This avoids enumerating tens/hundreds of discharge IDs.
    disch_ids = torch.arange(end_spec.disch_start, end_spec.disch_end_exclusive, dtype=torch.long, device=device)
    death_ids = torch.arange(end_spec.death_start, end_spec.death_end_exclusive, dtype=torch.long, device=device)

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

        # log end-token prob mass per step
        step_probs: Dict[str, float] = {}
        if disch_ids.numel() > 0 and int(disch_ids[-1]) < probs.numel():
            step_probs["DISCHARGE_mass"] = float(probs.index_select(0, disch_ids).sum().item())
        else:
            step_probs["DISCHARGE_mass"] = 0.0

        if death_ids.numel() > 0 and int(death_ids[-1]) < probs.numel():
            step_probs["DEATH_mass"] = float(probs.index_select(0, death_ids).sum().item())
        else:
            step_probs["DEATH_mass"] = 0.0

        end_token_probs.append(step_probs)

        if topk_sampling and topk_sampling > 0:
            topk = min(int(topk_sampling), probs.numel())
            p, ids = torch.topk(probs, k=topk)
            p = p / p.sum()
            pred = int(ids[torch.multinomial(p, 1)].item())
        else:
            pred = int(torch.argmax(logit_vec).item())

        tokens.append(pred)
        etypes.append(int(default_event_type_id))
        steps += 1

        lab = end_spec.classify(pred)
        if lab is not None:
            ended = True
            end_token = pred
            end_label = lab
            break

    dt = time.time() - t0
    return {
        "ended": bool(ended),
        "end_token": end_token,
        "end_label": end_label,  # <-- NEW
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
    end_spec: EndTokenSpec,          # <-- NEW
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
            end_spec=end_spec,
            max_len=int(max_len),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            topk_sampling=int(topk_sampling),
        )

        ended_any += int(pred["ended"])
        total_steps += int(pred["steps"])
        total_runtime += float(pred["runtime_sec"])

        gt_label = None
        if gt_tokens is not None:
            n_with_gt += 1
            gt_label = _first_end_label(gt_tokens, end_spec)
            if gt_label is not None and pred["end_label"] is not None and pred["end_label"] == gt_label:
                correct_end += 1

        row = {
            "patient_id": pid,
            "n_visits": len(visits),
            "has_ground_truth": gt_tokens is not None,
            "gt_end_label": gt_label,            # <-- NEW (label, not token id)
            "pred_end_label": pred["end_label"], # <-- NEW
            "pred_end_token": pred["end_token"], # keep for debugging
            "ended": pred["ended"],
            "steps": pred["steps"],
            "runtime_sec": pred["runtime_sec"],
            "end_token_probs": pred["end_token_probs"],  # now contains probability mass per class
            # optional debug:
            # "pred_tokens": pred["pred_tokens"],
        }
        rows.append(row)

    metrics: Dict[str, Any] = {
        "n_patients": int(n_patients),
        "n_with_gt": int(n_with_gt),
        "end_label_correct": int(correct_end),
        "end_label_accuracy": (correct_end / n_with_gt) if n_with_gt > 0 else float("nan"),
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

    # NEW: range-based end token config
    p.add_argument("--disch_start", type=int, default=77000, help="inclusive start id for discharge block")
    p.add_argument("--disch_end", type=int, default=77010, help="exclusive end id for discharge block")
    p.add_argument("--death_start", type=int, default=77010, help="inclusive start id for death block")
    p.add_argument("--death_end", type=int, default=77011, help="exclusive end id for death block (use start+1 for single [DEATH])")

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

    # validate ranges
    if not (args.disch_start < args.disch_end):
        raise ValueError("--disch_start must be < --disch_end")
    if not (args.death_start < args.death_end):
        raise ValueError("--death_start must be < --death_end")

    end_spec = EndTokenSpec(
        disch_start=int(args.disch_start),
        disch_end_exclusive=int(args.disch_end),
        death_start=int(args.death_start),
        death_end_exclusive=int(args.death_end),
    )

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
        end_spec=end_spec,
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
        print(
            f"end_label_accuracy={metrics['end_label_accuracy']:.4f} "
            f"({metrics['end_label_correct']}/{metrics['n_with_gt']})"
        )
    print(f"ended_fraction={metrics['ended_fraction']:.4f}")
    print(f"avg_steps={metrics['avg_steps']:.2f}")
    print(f"avg_runtime_sec={metrics['avg_runtime_sec']:.4f}")


if __name__ == "__main__":
    main()
