from __future__ import annotations

import argparse
import os
import sys
import random
import csv
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import (
    load_jsonl,
    build_token_id_to_group_from_vocab,  # token_id -> vocab/block/group id
)

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Next-event evaluation (measure-mode): sample context windows and predict the following token via MLM mask. ACC computed on vocab/block level."
    )
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)

    p.add_argument("--max_len", type=int, default=256, help="Context window length including the final MASK position.")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)

    # Toggle whether we "leak" the true target event-type into the MASK position
    p.add_argument(
        "--use_target_event_type_at_mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If True and event_type_ids exist, set event_type_ids at the MASK position to the TRUE next token's event type "
            "(a strong hint). If False, do NOT leak the target event type; instead, use the last attended event type "
            "or default_event_type_id."
        ),
    )

    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--n_trials", type=int, default=10, help="How many random context samples per patient sequence.")

    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optional RNG seed for reproducible sampling (python + torch). Defaults to 0",
    )

    p.add_argument(
        "--subset_frac",
        type=float,
        default=0.05,
        help="Evaluate only this fraction of patients (0<frac<=1). Default 0.05 for faster runs.",
    )
    p.add_argument("--max_patients", type=int, default=None, help="Evaluate only first N patients after subsetting (debug).")

    # CSV logging of (pred, true) pairs
    p.add_argument(
        "--pairs_csv",
        type=str,
        default=None,
        help="Output CSV path for per-trial (predicted, true) pairs and their vocab/group ids.",
    )

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


def _build_eval_example(
    tokens: List[int],
    event_types: Optional[List[int]],
    *,
    max_len: int,
    pad_id: int,
    mask_id: int,
    default_event_type_id: int,
    use_target_event_type_at_mask: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    L = int(max_len)
    if L < 5:
        raise ValueError("max_len must be >= 5 to allow 3 demographic tokens + some context + MASK.")
    if len(tokens) < 4:
        return None

    demo = tokens[:3]
    rest = tokens[3:]

    ctx_len = L - 1
    rest_ctx_len = ctx_len - 3  # = L-4
    if rest_ctx_len < 1:
        raise ValueError("max_len too small after accounting for demographics and mask slot.")

    # For short sequences sample a random target position in rest
    if len(rest) >= rest_ctx_len + 1:
        max_start = len(rest) - (rest_ctx_len + 1)
        start = random.randint(0, max_start)
        ctx_rest = rest[start : start + rest_ctx_len]
        true_next = rest[start + rest_ctx_len]
        if event_types is not None:
            ev_demo = event_types[:3]
            ev_rest = event_types[3:]
            ctx_ev_rest = ev_rest[start : start + rest_ctx_len]
            true_next_ev = ev_rest[start + rest_ctx_len]
        else:
            ev_demo = None
            ctx_ev_rest = None
            true_next_ev = None
    else:
        if len(rest) < 1:
            return None
        # pick random target token in rest, context are tokens directly before it (up to rest_ctx_len)
        j = random.randint(0, len(rest) - 1)
        true_next = rest[j]
        left = max(0, j - rest_ctx_len)
        ctx_rest = rest[left:j]
        if event_types is not None:
            ev_demo = event_types[:3]
            ev_rest = event_types[3:]
            ctx_ev_rest = ev_rest[left:j]
            true_next_ev = ev_rest[j]
        else:
            ev_demo = None
            ctx_ev_rest = None
            true_next_ev = None

    input_ids = [pad_id] * L
    attn = [0] * L
    ev_ids = [default_event_type_id] * L

    for i in range(min(3, len(demo))):
        input_ids[i] = demo[i]
        attn[i] = 1
        if event_types is not None and ev_demo is not None:
            ev_ids[i] = ev_demo[i]

    pos = 3
    for j2, tok in enumerate(ctx_rest):
        if pos + j2 >= ctx_len:
            break
        input_ids[pos + j2] = tok
        attn[pos + j2] = 1
        if event_types is not None and ctx_ev_rest is not None and j2 < len(ctx_ev_rest):
            ev_ids[pos + j2] = ctx_ev_rest[j2]

    input_ids[L - 1] = mask_id
    attn[L - 1] = 1

    # Toggle whether to leak the TRUE next event-type at the MASK position
    if (
        event_types is not None
        and true_next_ev is not None
        and bool(use_target_event_type_at_mask)
    ):
        ev_ids[L - 1] = true_next_ev
    else:
        last_attended = [i for i, a in enumerate(attn[: ctx_len]) if a == 1]
        ev_ids[L - 1] = ev_ids[last_attended[-1]] if last_attended else default_event_type_id

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(ev_ids, dtype=torch.long),
        int(true_next),
    )


@torch.no_grad()
def evaluate_next_event_measure_vocab(
    model: CompactTransformerEncoder,
    records: List[Dict[str, Any]],
    *,
    device: torch.device,
    max_len: int,
    pad_id: int,
    mask_id: int,
    default_event_type_id: int,
    topk: List[int],
    n_trials: int,
    use_target_event_type_at_mask: bool,
    token_id_to_vocab: Dict[int, int],  # token_id -> vocab/block id
    pairs_csv_path: str,  
) -> Dict[str, float]:
    model.eval()

    # Open CSV once and stream rows
    os.makedirs(os.path.dirname(pairs_csv_path) or ".", exist_ok=True)
    f_csv = open(pairs_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    writer.writerow(
        ["Patient_ID", "Predicted Token", "True Token", "Predicted_Token_Vocab_ID", "True_Token_Vocab_ID"]
    )

    total_trials = 0
    correct_at_k_global = {k: 0 for k in topk}

    patients_evaluated = 0
    sum_acc_at_k_patient = {k: 0.0 for k in topk}

    skipped_unknown_true_vocab = 0  # Track skipped trials due to unknown true vocab

    try:
        for pi, rec in enumerate(tqdm(records, desc="patients")):
            toks, ev = _extract_sequence(rec)

            # Prefer a real patient id if present; fallback to index
            pid = rec.get("patient_id", rec.get("patient", rec.get("id", None)))
            pid_str = str(pid) if pid is not None else str(pi)

            demo = toks[:3]
            demo_vocabs = [token_id_to_vocab.get(int(d), -1) for d in demo]
            print(f"\n=== patient={pi} demographics tokens={demo} vocabs={demo_vocabs} ===")

            patient_trials = 0
            correct_at_k_patient = {k: 0 for k in topk}

            for t in range(int(n_trials)):
                ex = _build_eval_example(
                    toks,
                    ev,
                    max_len=int(max_len),
                    pad_id=int(pad_id),
                    mask_id=int(mask_id),
                    default_event_type_id=int(default_event_type_id),
                    use_target_event_type_at_mask=bool(use_target_event_type_at_mask),
                )
                if ex is None:
                    continue

                x, attn, ev_ids, true_next_tok = ex

                # Skip trials where the true token maps to unknown vocab/group (-1)
                true_vocab = int(token_id_to_vocab.get(int(true_next_tok), -1))
                if true_vocab == -1:
                    skipped_unknown_true_vocab += 1
                    print(f"[trial={t}] SKIP (true_tok={true_next_tok} maps to unknown vocab=-1)")
                    continue

                x = x.unsqueeze(0).to(device)
                attn = attn.unsqueeze(0).to(device)
                ev_ids = ev_ids.unsqueeze(0).to(device)

                out = model(
                    input_ids=x,
                    attention_mask=attn,
                    event_type_ids=ev_ids,
                    labels=None,
                    return_hidden=False,
                )

                logits = out["logits"][0]  # [L, V]
                pos = logits.shape[0] - 1  # MASK position
                logit_pos = logits[pos].clone()

                logit_pos[int(pad_id)] = -1e9
                logit_pos[int(mask_id)] = -1e9

                kmax = max(topk)
                topk_tok_ids = torch.topk(logit_pos, k=kmax, dim=-1).indices.detach().cpu().tolist()
                pred_top1_tok = int(topk_tok_ids[0])

                pred_vocab_top1 = int(token_id_to_vocab.get(int(pred_top1_tok), -1))

                # Write one row per (non-skipped) trial
                writer.writerow([pid_str, pred_top1_tok, int(true_next_tok), pred_vocab_top1, true_vocab])

                # print(
                #     f"[trial={t}] "
                #     f"pred_tok={pred_top1_tok} pred_vocab={pred_vocab_top1}  "
                #     f"true_tok={true_next_tok} true_vocab={true_vocab}"
                # )

                patient_trials += 1
                total_trials += 1

                topk_vocabs = [int(token_id_to_vocab.get(int(tok), -1)) for tok in topk_tok_ids]
                for k in topk:
                    if true_vocab in topk_vocabs[:k]:
                        correct_at_k_patient[k] += 1
                        correct_at_k_global[k] += 1

            if patient_trials > 0:
                patients_evaluated += 1
                print(f"--- patient={pi} trials={patient_trials} accuracy (VOCAB-level) ---")
                for k in topk:
                    acc_k = float(correct_at_k_patient[k]) / float(patient_trials)
                    sum_acc_at_k_patient[k] += acc_k
                    print(f"acc@{k}={acc_k:.6f}")
            else:
                print(f"--- patient={pi} had 0 valid trials (skipped) ---")
    finally:
        f_csv.close()

    metrics: Dict[str, float] = {
        "patients_evaluated": float(patients_evaluated),
        "trials_total": float(total_trials),
        "trials_skipped_true_vocab_unknown": float(skipped_unknown_true_vocab),
    }

    for k in topk:
        metrics[f"acc@{k}_global_vocab"] = float(correct_at_k_global[k]) / max(1.0, float(total_trials))

    for k in topk:
        metrics[f"acc@{k}_mean_patient_vocab"] = float(sum_acc_at_k_patient[k]) / max(1.0, float(patients_evaluated))

    return metrics


def main() -> None:
    args = parse_args()

    if args.pairs_csv is None:
        ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.pairs_csv = os.path.join(
            "..",
            "out",
            "evaluation",
            f"next_event_vocab_pairs_{ckpt_base}.csv",
        )

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

    topk = _parse_topk(args.topk)

    metrics = evaluate_next_event_measure_vocab(
        model,
        records,
        device=device,
        max_len=int(args.max_len),
        pad_id=int(args.pad_id),
        mask_id=int(args.mask_id),
        default_event_type_id=int(args.default_event_type_id),
        topk=topk,
        n_trials=int(args.n_trials),
        use_target_event_type_at_mask=bool(args.use_target_event_type_at_mask),
        token_id_to_vocab=token_id_to_vocab,
        pairs_csv_path=str(args.pairs_csv),  
    )

    print(f"\ndevice={device}")
    print(f"use_target_event_type_at_mask={bool(args.use_target_event_type_at_mask)}")
    print(f"pairs_csv={str(args.pairs_csv)}")
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()