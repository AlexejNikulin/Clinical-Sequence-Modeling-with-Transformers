from __future__ import annotations

import argparse
import os
import sys
import random
import csv
from typing import Any, Dict, List, Optional, Tuple, Set

from tqdm import tqdm
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import (
    load_jsonl,
    build_token_id_to_group_from_vocab,  # token_id -> vocab/block id
)

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Next-event evaluation (measure-mode): length-independent NEXT-N via iterative MLM masks, "
            "with stop-vocab early stopping, CSV logging, event-type leak toggle, and true_vocab==-1 skips."
        )
    )
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary.json (needed for token->vocab mapping).")

    p.add_argument("--max_len", type=int, default=256, help="Full model input length.")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)

    p.add_argument("--topk", type=str, default="1,5,10")

    # horizon is a MAX cap (can stop earlier via stop_vocabs or seq end)
    p.add_argument("--horizon", type=int, default=1000, help="Maximum number of next tokens to attempt after sampled context.")

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling (python + torch). If omitted, evaluation is random.",
    )

    # stop-vocab IDs (vocab/group IDs, NOT token IDs)
    p.add_argument(
        "--stop_vocabs",
        type=str,
        default="12,13",
        help="Comma-separated list of stop vocabulary/group IDs. Stop when pred OR true hits one of these (inclusive).",
    )

    # Toggle whether we "leak" true target event-type into generated MASK positions
    p.add_argument(
        "--use_target_event_type_at_mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If True and event_type_ids exist, set event_type_ids at each generated MASK position to the TRUE token's "
            "event type for that step (strong hint). If False, do NOT leak; use last attended event type or default."
        ),
    )

    # speed: evaluate only a fraction
    p.add_argument("--subset_frac", type=float, default=0.05)
    p.add_argument("--max_patients", type=int, default=None, help="Evaluate only first N patients after subsetting (debug).")

    # CSV logging
    p.add_argument("--pairs_csv", type=str, default=None, help="Output CSV path for per-step (pred, true) pairs + vocab ids.")

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
    for x in (arg or "").split(","):
        x = x.strip()
        if x:
            ks.append(int(x))
    ks = sorted(set(ks))
    return ks if ks else [1, 5, 10]


def _parse_int_set_csv(arg: str) -> Set[int]:
    s: Set[int] = set()
    for x in (arg or "").split(","):
        x = x.strip()
        if x:
            s.add(int(x))
    return s


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


def _build_eval_example_nextn_length_independent(
    tokens: List[int],
    event_types: Optional[List[int]],
    *,
    max_len: int,
    pad_id: int,
    default_event_type_id: int,
    horizon: int,
) -> Optional[
    Tuple[
        torch.Tensor,              # input_ids [L]
        torch.Tensor,              # attention_mask [L]
        torch.Tensor,              # event_type_ids [L]
        List[int],                 # true_next_tokens (len=H_eff)
        Optional[List[int]],       # true_next_event_types (len=H_eff) or None
    ]
]:
    """
    Length-independent sampling:
    - demo = tokens[:3] fixed at positions 0..2
    - rest = tokens[3:]
    - choose random pred_start in rest
    - context = up to rest_ctx_cap tokens immediately before pred_start
    - targets = up to horizon tokens starting at pred_start (clipped to end)
    - keep at least one PAD slot free for first appended MASK (ctx_cap = L-1)
    """
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
    if len(rest) < 1:
        return None

    ctx_cap = L - 1  # leave at least one free PAD slot for generation
    rest_ctx_cap = ctx_cap - 3
    if rest_ctx_cap < 1:
        raise ValueError("max_len too small after accounting for demographics and free append slot.")

    pred_start = random.randint(0, len(rest) - 1)
    ctx_left = max(0, pred_start - rest_ctx_cap)
    ctx_rest = rest[ctx_left:pred_start]

    true_next_tokens = rest[pred_start : min(len(rest), pred_start + H)]
    if len(true_next_tokens) < 1:
        return None

    true_next_event_types: Optional[List[int]] = None
    ctx_ev_rest: Optional[List[int]] = None
    ev_demo: Optional[List[int]] = None
    if event_types is not None:
        ev_demo = event_types[:3]
        ev_rest = event_types[3:]
        ctx_ev_rest = ev_rest[ctx_left:pred_start]
        true_next_event_types = ev_rest[pred_start : min(len(ev_rest), pred_start + H)]
        if len(true_next_event_types) != len(true_next_tokens):
            raise ValueError("Internal mismatch between tokens and event_types targets.")

    input_ids = [pad_id] * L
    attn = [0] * L
    ev_ids = [default_event_type_id] * L

    # demographics (always)
    for i in range(min(3, len(demo))):
        input_ids[i] = int(demo[i])
        attn[i] = 1
        if ev_demo is not None:
            ev_ids[i] = int(ev_demo[i])

    # context after demo
    pos = 3
    for j, tok in enumerate(ctx_rest):
        if pos + j >= ctx_cap:
            break
        input_ids[pos + j] = int(tok)
        attn[pos + j] = 1
        if ctx_ev_rest is not None and j < len(ctx_ev_rest):
            ev_ids[pos + j] = int(ctx_ev_rest[j])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(ev_ids, dtype=torch.long),
        [int(x) for x in true_next_tokens],
        [int(x) for x in true_next_event_types] if true_next_event_types is not None else None,
    )


def _slide_left_preserve_demo(
    xg: torch.Tensor,
    ag: torch.Tensor,
    eg: torch.Tensor,
    *,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Slide only positions [3:] left by 1; keep demographics [0..2] fixed.
    This preserves demographic tokens always.
    """
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
def evaluate_next_event_nextn_vocab(
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
    token_id_to_vocab: Dict[int, int],  # token_id -> vocab/group id
    stop_vocabs: Set[int],
    pairs_csv_path: str,
    use_target_event_type_at_mask: bool,
) -> Dict[str, float]:
    model.eval()
    Hmax = int(horizon)

    # CSV logging
    os.makedirs(os.path.dirname(pairs_csv_path) or ".", exist_ok=True)
    f_csv = open(pairs_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    writer.writerow(
        [
            "Patient_ID",
            "Step",
            "Predicted Token",
            "True Token",
            "Predicted_Token_Vocab_ID",
            "True_Token_Vocab_ID",
            "Stopped_By",  # "", "pred_stop", "true_stop"
        ]
    )

    total_steps = 0
    correct_at_k_pairwise = {k: 0 for k in topk}

    patients_evaluated = 0
    sum_count_overlap_at_k = {k: 0.0 for k in topk}
    sum_pairwise_acc_at_k = {k: 0.0 for k in topk}

    skipped_unknown_true_vocab = 0

    try:
        for pi, rec in enumerate(tqdm(records, desc="patients")):
            toks, ev = _extract_sequence(rec)

            ex = _build_eval_example_nextn_length_independent(
                toks,
                ev,
                max_len=int(max_len),
                pad_id=int(pad_id),
                default_event_type_id=int(default_event_type_id),
                horizon=Hmax,
            )
            if ex is None:
                continue

            x0, a0, e0, true_next_tokens, true_next_event_types = ex
            if len(true_next_tokens) < 1:
                continue

            pid = rec.get("patient_id", rec.get("patient", rec.get("id", None)))
            pid_str = str(pid) if pid is not None else str(pi)

            xg = x0.to(device)
            ag = a0.to(device)
            eg = e0.to(device)

            # Metrics bookkeeping (ONLY over VALID evaluated steps, i.e. true_vocab != -1)
            true_count: Dict[int, int] = {}
            pred_available_count_at_k: Dict[int, Dict[int, int]] = {k: {} for k in topk}
            correct_at_k_patient_pairwise = {k: 0 for k in topk}

            valid_steps_patient = 0

            for step in range(min(Hmax, len(true_next_tokens))):
                true_tok = int(true_next_tokens[step])
                true_vocab = int(token_id_to_vocab.get(true_tok, -1))

                # --- Skip steps where true vocab is unknown ---
                if true_vocab == -1:
                    skipped_unknown_true_vocab += 1
                    continue

                # ensure a padding slot exists; if not, slide
                pad_pos = (ag == 0).nonzero(as_tuple=False).view(-1)
                if pad_pos.numel() == 0:
                    xg, ag, eg = _slide_left_preserve_demo(xg, ag, eg, pad_id=int(pad_id))
                    pad_pos = (ag == 0).nonzero(as_tuple=False).view(-1)
                    if pad_pos.numel() == 0:
                        break

                gen_pos = int(pad_pos[0].item())

                # put MASK and include in attention
                xg[gen_pos] = int(mask_id)
                ag[gen_pos] = 1

                # event_type_ids at MASK position: leak toggle
                if bool(use_target_event_type_at_mask) and true_next_event_types is not None:
                    eg[gen_pos] = int(true_next_event_types[step])
                else:
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
                logits = out["logits"][0]  # [L, V]
                logit_pos = logits[gen_pos].clone()

                # avoid predicting PAD/MASK
                logit_pos[int(pad_id)] = -1e9
                logit_pos[int(mask_id)] = -1e9

                kmax = max(topk)
                topk_tok_ids = torch.topk(logit_pos, k=kmax, dim=-1).indices.detach().cpu().tolist()

                pred_top1_tok = int(topk_tok_ids[0])
                pred_vocab = int(token_id_to_vocab.get(pred_top1_tok, -1))

                # --- Update true_count for this VALID step (VOCAB-level) ---
                true_count[true_vocab] = true_count.get(true_vocab, 0) + 1

                # Pairwise acc@k (VOCAB-level) + bag/count bookkeeping
                topk_vocabs = [int(token_id_to_vocab.get(int(tok), -1)) for tok in topk_tok_ids]

                for k in topk:
                    if true_vocab in topk_vocabs[:k]:
                        correct_at_k_pairwise[k] += 1
                        correct_at_k_patient_pairwise[k] += 1

                    # --- No -1 in present_vocabs ---
                    present_vocabs = {v for v in topk_vocabs[:k] if v != -1}
                    d = pred_available_count_at_k[k]
                    for v in present_vocabs:
                        d[v] = d.get(v, 0) + 1

                total_steps += 1
                valid_steps_patient += 1

                # commit predicted token into context
                xg[gen_pos] = pred_top1_tok

                # --- Early stop conditions on VOCAB-level (inclusive) ---
                stopped_by = ""
                if stop_vocabs and pred_vocab in stop_vocabs:
                    stopped_by = "pred_stop"
                elif stop_vocabs and true_vocab in stop_vocabs:
                    stopped_by = "true_stop"

                # CSV row for this VALID step
                writer.writerow([pid_str, int(step), pred_top1_tok, true_tok, pred_vocab, true_vocab, stopped_by])

                if stopped_by:
                    break

            # patient aggregation only if at least 1 VALID step contributed
            if valid_steps_patient <= 0:
                continue

            patients_evaluated += 1

            # Mean-Patient pairwise accuracy aggregation
            for k in topk:
                sum_pairwise_acc_at_k[k] += float(correct_at_k_patient_pairwise[k]) / float(valid_steps_patient)

            # Mean-Patient count/Bag-of-Vocabs score aggregation
            for k in topk:
                pred_cnt = pred_available_count_at_k[k]
                overlap = 0
                for v, c_true in true_count.items():
                    overlap += min(int(c_true), int(pred_cnt.get(v, 0)))
                sum_count_overlap_at_k[k] += float(overlap) / float(valid_steps_patient)

    finally:
        f_csv.close()

    metrics: Dict[str, float] = {
        "patients_evaluated": float(patients_evaluated),
        "steps_total": float(total_steps),
        "horizon_max": float(Hmax),
        "steps_skipped_true_vocab_unknown": float(skipped_unknown_true_vocab),
    }

    # global (weighted by steps)
    for k in topk:
        metrics[f"pairwise_acc@{k}_global_vocab"] = float(correct_at_k_pairwise[k]) / max(1.0, float(total_steps))

    # mean over patients (unweighted), only patients with >=1 valid step
    for k in topk:
        metrics[f"pairwise_acc@{k}_mean_patient_vocab"] = float(sum_pairwise_acc_at_k[k]) / max(1.0, float(patients_evaluated))
        metrics[f"count_score@{k}_mean_patient_vocab"] = float(sum_count_overlap_at_k[k]) / max(1.0, float(patients_evaluated))

    return metrics


def main() -> None:
    args = parse_args()

    # Build dynamic default for CSV if not provided
    if args.pairs_csv is None:
        ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.pairs_csv = os.path.join("..", "out", "evaluation", f"next_event_seq_vocab_pairs_{ckpt_base}.csv")

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

    from vocabulary import Vocabulary

    vocab = Vocabulary.load(args.vocab_path)
    token_id_to_vocab = build_token_id_to_group_from_vocab(vocab)

    stop_vocabs = _parse_int_set_csv(args.stop_vocabs)
    topk = _parse_topk(args.topk)

    metrics = evaluate_next_event_nextn_vocab(
        model,
        records,
        device=device,
        max_len=int(args.max_len),
        pad_id=int(args.pad_id),
        mask_id=int(args.mask_id),
        default_event_type_id=int(args.default_event_type_id),
        topk=topk,
        horizon=int(args.horizon),
        token_id_to_vocab=token_id_to_vocab,
        stop_vocabs=stop_vocabs,
        pairs_csv_path=str(args.pairs_csv),
        use_target_event_type_at_mask=bool(args.use_target_event_type_at_mask),
    )

    print(f"\ndevice={device}")
    print(f"use_target_event_type_at_mask={bool(args.use_target_event_type_at_mask)}")
    print(f"pairs_csv={str(args.pairs_csv)}")
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()