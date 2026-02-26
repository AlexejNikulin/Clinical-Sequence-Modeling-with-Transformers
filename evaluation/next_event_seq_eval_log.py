from __future__ import annotations

import argparse
import os
import sys
import random
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import load_jsonl
from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Next-event evaluation (measure-mode): sample context windows and predict the following tokens (NEXT-10) via MLM mask."
    )
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--max_len", type=int, default=256, help="Context window length (full model input length).")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)

    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--horizon", type=int, default=10, help="How many next tokens to predict after the sampled context.")

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling. If omitted, evaluation is fully random.",
    )

    # speed control via subset sampling
    p.add_argument(
        "--subset_frac",
        type=float,
        default=0.05,
        help="Evaluate only this fraction of patients (0<frac<=1). Default 0.05 for faster runs.",
    )
    p.add_argument("--max_patients", type=int, default=None, help="Evaluate only first N patients after subsetting (debug).")

    return p.parse_args()


def _load_ckpt_and_model(ckpt_path: str) -> CompactTransformerEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt:
        cfg = CompactTransformerConfig(**ckpt["cfg"])
        model = CompactTransformerEncoder(cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model
    raise ValueError("Unsupported checkpoint format. Expected dict with keys: cfg, model_state_dict.")


def _parse_topk(arg: str) -> List[int]:
    ks = []
    for x in arg.split(","):
        x = x.strip()
        if not x:
            continue
        ks.append(int(x))
    ks = sorted(set(ks))
    if not ks:
        ks = [1, 5, 10]
    return ks


def _extract_sequence(record: Dict[str, Any]) -> Tuple[List[int], Optional[List[int]]]:
    if "input_ids" in record and isinstance(record["input_ids"], list):
        toks = [int(x) for x in record["input_ids"]]
    elif "token_ids" in record and isinstance(record["token_ids"], list):
        toks = [int(x) for x in record["token_ids"]]
    elif "tokens" in record and isinstance(record["tokens"], list):
        toks = [int(x) for x in record["tokens"]]
    else:
        raise KeyError("Record does not contain a token list under known keys (input_ids/token_ids/tokens).")

    ev = None
    if "event_type_ids" in record and isinstance(record["event_type_ids"], list):
        ev = [int(x) for x in record["event_type_ids"]]
    elif "event_types" in record and isinstance(record["event_types"], list):
        ev = [int(x) for x in record["event_types"]]

    if ev is not None and len(ev) != len(toks):
        raise ValueError(f"Length mismatch: len(tokens)={len(toks)} vs len(event_type_ids)={len(ev)}")

    return toks, ev


def _build_eval_example_nextn(
    tokens: List[int],
    event_types: Optional[List[int]],
    *,
    max_len: int,
    pad_id: int,
    default_event_type_id: int,
    horizon: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]]:
    L = int(max_len)
    H = int(horizon)
    if H < 1:
        raise ValueError("--horizon must be >= 1.")
    if L < 6:
        raise ValueError("max_len must be >= 6 for next-N evaluation with demographics and append slot.")
    if len(tokens) < 4:
        return None

    demo = tokens[:3]
    rest = tokens[3:]

    ctx_cap = L - 1
    rest_ctx_cap = ctx_cap - 3
    if rest_ctx_cap < 1:
        raise ValueError("max_len too small after accounting for demographics and free append slot.")

    if len(rest) >= rest_ctx_cap + H:
        max_start = len(rest) - (rest_ctx_cap + H)
        start = random.randint(0, max_start)
        ctx_rest = rest[start : start + rest_ctx_cap]
        true_next_tokens = rest[start + rest_ctx_cap : start + rest_ctx_cap + H]

        if event_types is not None:
            ev_demo = event_types[:3]
            ev_rest = event_types[3:]
            ctx_ev_rest = ev_rest[start : start + rest_ctx_cap]
        else:
            ev_demo = None
            ctx_ev_rest = None
    else:
        if len(rest) < H:
            return None
        true_next_tokens = rest[-H:]
        ctx_available = rest[:-H]
        ctx_rest = ctx_available[-rest_ctx_cap:]

        if event_types is not None:
            ev_demo = event_types[:3]
            ev_rest = event_types[3:]
            ctx_ev_available = ev_rest[:-H]
            ctx_ev_rest = ctx_ev_available[-rest_ctx_cap:]
        else:
            ev_demo = None
            ctx_ev_rest = None

    input_ids = [pad_id] * L
    attn = [0] * L
    ev_ids = [default_event_type_id] * L

    for i in range(min(3, len(demo))):
        input_ids[i] = demo[i]
        attn[i] = 1
        if event_types is not None and ev_demo is not None:
            ev_ids[i] = ev_demo[i]

    pos = 3
    for j, tok in enumerate(ctx_rest):
        if pos + j >= ctx_cap:
            break
        input_ids[pos + j] = tok
        attn[pos + j] = 1
        if event_types is not None and ctx_ev_rest is not None and j < len(ctx_ev_rest):
            ev_ids[pos + j] = ctx_ev_rest[j]

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(ev_ids, dtype=torch.long),
        [int(x) for x in true_next_tokens],
    )


def _slide_left_preserve_demo(
    xg: torch.Tensor,
    ag: torch.Tensor,
    eg: torch.Tensor,
    *,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if xg.numel() < 4:
        return xg, ag, eg

    xs = xg[3:].clone()
    as_ = ag[3:].clone()
    es = eg[3:].clone()

    xs = torch.roll(xs, shifts=-1, dims=0)
    as_ = torch.roll(as_, shifts=-1, dims=0)
    es = torch.roll(es, shifts=-1, dims=0)

    xs[-1] = int(pad_id)
    as_[-1] = 0
    if es.numel() >= 2:
        es[-1] = es[-2]

    xg2 = xg.clone()
    ag2 = ag.clone()
    eg2 = eg.clone()

    xg2[3:] = xs
    ag2[3:] = as_
    eg2[3:] = es

    return xg2, ag2, eg2


@torch.no_grad()
def evaluate_next_event_nextn_token(
    model: CompactTransformerEncoder,
    records: List[Dict[str, Any]],
    *,
    device: torch.device,
    max_len: int,
    pad_id: int,
    mask_id: int,
    default_event_type_id: int,
    topk: List[int],
    horizon: int,
) -> Dict[str, float]:
    model.eval()

    H = int(horizon)

    total_steps = 0
    correct_at_k_pairwise = {k: 0 for k in topk}

    patients_evaluated = 0
    sum_count_score_at_k = {k: 0.0 for k in topk}

    for pi, rec in enumerate(tqdm(records, desc="patients")):
        toks, ev = _extract_sequence(rec)

        ex = _build_eval_example_nextn(
            toks,
            ev,
            max_len=int(max_len),
            pad_id=int(pad_id),
            default_event_type_id=int(default_event_type_id),
            horizon=H,
        )
        if ex is None:
            continue

        x0, a0, e0, true_next_tokens = ex
        if len(true_next_tokens) != H:
            continue

        demo = toks[:3]
        print(f"\n=== patient={pi} demographics tokens={demo} ===")

        xg = x0.to(device)
        ag = a0.to(device)
        eg = e0.to(device)

        true_count: Dict[int, int] = {}
        for t in true_next_tokens:
            true_count[int(t)] = true_count.get(int(t), 0) + 1

        pred_available_count_at_k: Dict[int, Dict[int, int]] = {k: {} for k in topk}
        correct_at_k_patient_pairwise = {k: 0 for k in topk}

        for step in range(H):
            pad_pos = (ag == 0).nonzero(as_tuple=False).view(-1)
            if pad_pos.numel() == 0:
                xg, ag, eg = _slide_left_preserve_demo(xg, ag, eg, pad_id=int(pad_id))
                pad_pos = (ag == 0).nonzero(as_tuple=False).view(-1)
                if pad_pos.numel() == 0:
                    break

            gen_pos = int(pad_pos[0].item())

            xg[gen_pos] = int(mask_id)
            ag[gen_pos] = 1

            last_att = (ag == 1).nonzero(as_tuple=False).view(-1)
            if last_att.numel() > 0:
                eg[gen_pos] = eg[int(last_att[-1].item())]
            else:
                eg[gen_pos] = int(default_event_type_id)

            out = model(
                input_ids=xg.unsqueeze(0),
                attention_mask=ag.unsqueeze(0),
                event_type_ids=eg.unsqueeze(0),
                labels=None,
                return_hidden=False,
            )
            logits = out["logits"][0]
            logit_pos = logits[gen_pos].clone()

            logit_pos[int(pad_id)] = -1e9
            logit_pos[int(mask_id)] = -1e9

            kmax = max(topk)
            topk_tok_ids = torch.topk(logit_pos, k=kmax, dim=-1).indices.detach().cpu().tolist()
            pred_top1_tok = int(topk_tok_ids[0])

            true_tok = int(true_next_tokens[step])
            print(f"[step={step}] pred_tok={pred_top1_tok} true_tok={true_tok}")

            for k in topk:
                if true_tok in topk_tok_ids[:k]:
                    correct_at_k_pairwise[k] += 1
                    correct_at_k_patient_pairwise[k] += 1

                present_toks = set(int(x) for x in topk_tok_ids[:k])
                d = pred_available_count_at_k[k]
                for t in present_toks:
                    d[t] = d.get(t, 0) + 1

            total_steps += 1
            xg[gen_pos] = pred_top1_tok

        patients_evaluated += 1

        print(f"--- patient={pi} horizon={H} pairwise accuracy (TOKEN-level) ---")
        for k in topk:
            denom = float(H)
            acc_k = float(correct_at_k_patient_pairwise[k]) / denom
            print(f"pairwise_acc@{k}={acc_k:.6f}")

        print(f"--- patient={pi} horizon={H} count_score (TOKEN-level) ---")
        for k in topk:
            pred_cnt = pred_available_count_at_k[k]
            overlap = 0
            for t, c_true in true_count.items():
                overlap += min(int(c_true), int(pred_cnt.get(t, 0)))
            score = float(overlap) / float(H)
            sum_count_score_at_k[k] += score
            print(f"count_score@{k}={score:.6f}")

    metrics: Dict[str, float] = {
        "patients_evaluated": float(patients_evaluated),
        "steps_total": float(total_steps),
        "horizon": float(H),
    }

    for k in topk:
        metrics[f"pairwise_acc@{k}_global_token"] = float(correct_at_k_pairwise[k]) / max(1.0, float(total_steps))

    for k in topk:
        metrics[f"count_score@{k}_mean_patient_token"] = float(sum_count_score_at_k[k]) / max(1.0, float(patients_evaluated))

    return metrics


def main() -> None:
    args = parse_args()

    # Only seed when explicitly provided 
    if args.seed is not None:
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    model = _load_ckpt_and_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)

    frac = float(args.subset_frac)
    if frac <= 0.0 or frac > 1.0:
        raise ValueError("--subset_frac must be in (0, 1].")
    if frac < 1.0:
        n = max(1, int(round(len(records) * frac)))
        records = random.sample(records, k=n)
    if args.max_patients is not None:
        records = records[: int(args.max_patients)]

    topk = _parse_topk(args.topk)

    metrics = evaluate_next_event_nextn_token(
        model,
        records,
        device=device,
        max_len=int(args.max_len),
        pad_id=int(args.pad_id),
        mask_id=int(args.mask_id),
        default_event_type_id=int(args.default_event_type_id),
        topk=topk,
        horizon=int(args.horizon),
    )

    print(f"\ndevice={device}")
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()