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

from evaluation.clinical_eval_utils import (
    load_jsonl,
    build_token_id_to_group_from_vocab,  # token_id -> vocab/block id
)

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Next-event evaluation (measure-mode): sample 1 context window and predict next N tokens autoregressively. ACC computed on vocab/block level."
    )
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)

    p.add_argument("--max_len", type=int, default=256, help="Context window length (full sequence length in model input).")
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)

    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--seed", type=int, default=13, help="Random seed for reproducible sampling (python + torch).")

    # Predict horizon
    p.add_argument("--horizon", type=int, default=10, help="How many next tokens to predict after the sampled context.")
    # -------------------------------

    # speed: evaluate only a fraction
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


# Build one sampled context + next-N true tokens (instead of single next token)
def _build_eval_example_nextn(
    tokens: List[int],
    event_types: Optional[List[int]],
    *,
    max_len: int,
    pad_id: int,
    default_event_type_id: int,
    horizon: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]]:
    """
    Build one evaluation example:
    - input length = max_len
    - we DO NOT place an initial MASK; generation loop will place MASK into first free slot each step
    - context occupies positions 0..(some <= max_len-1), leaving at least one pad slot to append into
    - first 3 tokens are always demographics at positions 0..2
    - returns true_next_tokens: the next `horizon` tokens right after the sampled context in the original sequence
    """
    L = int(max_len)
    H = int(horizon)
    if H < 1:
        raise ValueError("--horizon must be >= 1.")
    if L < 6:
        # 3 demo + at least 2 context + 1 free slot
        raise ValueError("max_len must be >= 6 for next-N evaluation with demographics and append slot.")
    if len(tokens) < 4:
        return None

    demo = tokens[:3]
    rest = tokens[3:]

    # context capacity excluding the mandatory free append slot (we want at least one pad slot)
    ctx_cap = L - 1  # leave last slot free (pad) for the first appended MASK
    rest_ctx_cap = ctx_cap - 3
    if rest_ctx_cap < 1:
        raise ValueError("max_len too small after accounting for demographics and free append slot.")

    # We need: rest-context (<=rest_ctx_cap) followed by H true tokens.
    # If sequence is long enough: sample a contiguous window of length rest_ctx_cap, with H tokens following it.
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
        # Shorter sequences: still allow evaluation if we have at least H tokens in `rest`.
        # Use as much preceding context as possible (right-aligned) and take the last H tokens as targets.
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

    # demographics at fixed positions
    for i in range(min(3, len(demo))):
        input_ids[i] = demo[i]
        attn[i] = 1
        if event_types is not None and ev_demo is not None:
            ev_ids[i] = ev_demo[i]

    # context after demographics, left-aligned
    pos = 3
    for j, tok in enumerate(ctx_rest):
        if pos + j >= ctx_cap:
            break
        input_ids[pos + j] = tok
        attn[pos + j] = 1
        if event_types is not None and ctx_ev_rest is not None and j < len(ctx_ev_rest):
            ev_ids[pos + j] = ctx_ev_rest[j]

    # last slot remains pad/attn=0 for first append

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(ev_ids, dtype=torch.long),
        [int(x) for x in true_next_tokens],
    )
# ---------------------------------------------------------------------------


# Sliding window that preserves demographics at 0..2
def _slide_left_preserve_demo(
    xg: torch.Tensor,
    ag: torch.Tensor,
    eg: torch.Tensor,
    *,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shift positions [3:] left by 1, keep [0..2] fixed.
    Frees the last position as a pad slot.
    """
    if xg.numel() < 4:
        return xg, ag, eg

    # slice [3:]
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
    else:
        es[-1] = es[-1]

    xg2 = xg.clone()
    ag2 = ag.clone()
    eg2 = eg.clone()

    xg2[3:] = xs
    ag2[3:] = as_
    eg2[3:] = es

    return xg2, ag2, eg2
# ---------------------------------------------------------------------------


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
    token_id_to_vocab: Dict[int, int],  # token_id -> vocab/block id
) -> Dict[str, float]:
    model.eval()

    H = int(horizon)

    # Pairwise (global over steps)
    total_steps = 0
    correct_at_k_pairwise = {k: 0 for k in topk}

    # Count/Bag-of-Vocabs (average over patients)
    patients_evaluated = 0
    sum_count_overlap_at_k = {k: 0.0 for k in topk}

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
            # safety: should not happen, but keep robust
            continue

        # demographics once per patient
        demo = toks[:3]
        demo_vocabs = [token_id_to_vocab.get(int(d), -1) for d in demo]
        print(f"\n=== patient={pi} demographics tokens={demo} vocabs={demo_vocabs} ===")

        # Prepare rolling tensors
        xg = x0.to(device)
        ag = a0.to(device)
        eg = e0.to(device)

        # True vocabs (for pairwise and count metrics)
        true_vocabs = [int(token_id_to_vocab.get(int(t), -1)) for t in true_next_tokens]

        # Build true counts
        true_count: Dict[int, int] = {}
        for v in true_vocabs:
            true_count[v] = true_count.get(v, 0) + 1

        # pred_available_count@k: how often each vocab appears somewhere in top-k per step
        pred_available_count_at_k: Dict[int, Dict[int, int]] = {k: {} for k in topk}

        # per-patient pairwise correct counters for logging
        correct_at_k_patient_pairwise = {k: 0 for k in topk}

        for t in range(H):
            # Ensure we have a padding slot to append into; otherwise free one via sliding window.
            pad_pos = (ag == 0).nonzero(as_tuple=False).view(-1)
            if pad_pos.numel() == 0:
                xg, ag, eg = _slide_left_preserve_demo(xg, ag, eg, pad_id=int(pad_id))
                pad_pos = (ag == 0).nonzero(as_tuple=False).view(-1)
                if pad_pos.numel() == 0:
                    # cannot append at all
                    break

            gen_pos = int(pad_pos[0].item())

            # Place MASK at gen_pos and include it in attention so model predicts at that position
            xg[gen_pos] = int(mask_id)
            ag[gen_pos] = 1
            # event type: keep consistent with last attended token (fallback: default)
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

            # avoid predicting PAD/MASK as "real"
            logit_pos[int(pad_id)] = -1e9
            logit_pos[int(mask_id)] = -1e9

            kmax = max(topk)
            topk_tok_ids = torch.topk(logit_pos, k=kmax, dim=-1).indices.detach().cpu().tolist()

            pred_top1_tok = int(topk_tok_ids[0])
            pred_top1_vocab = int(token_id_to_vocab.get(pred_top1_tok, -1))

            true_tok = int(true_next_tokens[t])
            true_vocab = int(true_vocabs[t])

            # Log step
            print(
                f"[step={t}] "
                f"pred_tok={pred_top1_tok} pred_vocab={pred_top1_vocab}  "
                f"true_tok={true_tok} true_vocab={true_vocab}"
            )

            # Pairwise acc@k (vocab-level)
            topk_vocabs = [int(token_id_to_vocab.get(int(tok), -1)) for tok in topk_tok_ids]
            for k in topk:
                if true_vocab in topk_vocabs[:k]:
                    correct_at_k_pairwise[k] += 1
                    correct_at_k_patient_pairwise[k] += 1

                # Count metric: vocab is "available" in this step if it appears anywhere in top-k
                # We'll count presence per vocab per step (duplicates don't matter).
                present_vocabs = set(topk_vocabs[:k])
                d = pred_available_count_at_k[k]
                for v in present_vocabs:
                    d[v] = d.get(v, 0) + 1

            total_steps += 1

            # Commit predicted top-1 token into context for next step
            xg[gen_pos] = pred_top1_tok
            # (ag already 1)

        # Per-patient summaries
        patients_evaluated += 1

        print(f"--- patient={pi} horizon={H} pairwise accuracy (VOCAB-level) ---")
        for k in topk:
            denom = float(H)
            acc_k = float(correct_at_k_patient_pairwise[k]) / denom
            print(f"pairwise_acc@{k}={acc_k:.6f}")

        print(f"--- patient={pi} horizon={H} count/Bag-of-Vocabs score (VOCAB-level) ---")
        for k in topk:
            pred_cnt = pred_available_count_at_k[k]
            overlap = 0
            for v, c_true in true_count.items():
                overlap += min(int(c_true), int(pred_cnt.get(v, 0)))
            score = float(overlap) / float(H)
            sum_count_overlap_at_k[k] += score
            print(f"count_score@{k}={score:.6f}")

    metrics: Dict[str, float] = {
        "patients_evaluated": float(patients_evaluated),
        "steps_total": float(total_steps),
    }

    # Global pairwise accuracies over all steps (weighted by steps)
    for k in topk:
        metrics[f"pairwise_acc@{k}_global_vocab"] = float(correct_at_k_pairwise[k]) / max(1.0, float(total_steps))

    # Global count scores averaged over patients (unweighted)
    for k in topk:
        metrics[f"count_score@{k}_mean_patient_vocab"] = float(sum_count_overlap_at_k[k]) / max(1.0, float(patients_evaluated))

    return metrics


def main() -> None:
    args = parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model = _load_ckpt_and_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)

    # evaluate only a fraction by default (speed)
    frac = float(args.subset_frac)
    if frac <= 0.0 or frac > 1.0:
        raise ValueError("--subset_frac must be in (0, 1].")
    if frac < 1.0:
        n = max(1, int(round(len(records) * frac)))
        records = random.sample(records, k=n)
    if args.max_patients is not None:
        records = records[: int(args.max_patients)]

    # build token->vocab/block mapping
    from vocabulary import Vocabulary

    vocab = Vocabulary.load(args.vocab_path)
    token_id_to_vocab = build_token_id_to_group_from_vocab(vocab)

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
    )

    print(f"\ndevice={device}")
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()