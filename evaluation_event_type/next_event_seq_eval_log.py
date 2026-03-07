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
    build_token_id_to_group_from_vocab,
)

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Next-event evaluation (measure-mode): sample 1 context window and predict next N tokens autoregressively. ACC computed on vocab/block level."
    )
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)

    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)

    p.add_argument(
        "--use_target_event_type_at_mask",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=10)

    p.add_argument("--subset_frac", type=float, default=0.05)
    p.add_argument("--max_patients", type=int, default=None)

    p.add_argument("--pairs_csv", type=str, default=None)

    return p.parse_args()


def _load_ckpt_and_model(ckpt_path: str) -> CompactTransformerEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt:
        cfg = CompactTransformerConfig(**ckpt["cfg"])
        model = CompactTransformerEncoder(cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model
    raise ValueError("Unsupported checkpoint format.")


def _parse_topk(arg: str) -> List[int]:
    ks = []
    for x in arg.split(","):
        x = x.strip()
        if x:
            ks.append(int(x))
    ks = sorted(set(ks))
    return ks if ks else [1, 5, 10]


def _extract_sequence(record: Dict[str, Any]) -> Tuple[List[int], Optional[List[int]]]:
    if "input_ids" in record:
        toks = [int(x) for x in record["input_ids"]]
    elif "token_ids" in record:
        toks = [int(x) for x in record["token_ids"]]
    elif "tokens" in record:
        toks = [int(x) for x in record["tokens"]]
    else:
        raise KeyError("No token list found.")

    ev = None
    if "event_type_ids" in record:
        ev = [int(x) for x in record["event_type_ids"]]
    elif "event_types" in record:
        ev = [int(x) for x in record["event_types"]]

    if ev is not None and len(ev) != len(toks):
        raise ValueError("Token/Event length mismatch.")

    return toks, ev


def _build_eval_example_nextn(
    tokens: List[int],
    event_types: Optional[List[int]],
    *,
    max_len: int,
    pad_id: int,
    default_event_type_id: int,
    horizon: int,
):
    L = int(max_len)
    H = int(horizon)

    if len(tokens) < 4:
        return None

    demo = tokens[:3]
    rest = tokens[3:]

    ctx_cap = L - 1
    rest_ctx_cap = ctx_cap - 3
    if len(rest) < 1:
        return None

    pred_start = random.randint(0, len(rest) - 1)
    ctx_left = max(0, pred_start - rest_ctx_cap)
    ctx_rest = rest[ctx_left:pred_start]

    true_next_tokens = rest[pred_start: min(len(rest), pred_start + H)]
    if not true_next_tokens:
        return None

    true_next_event_types = None
    if event_types is not None:
        ev_rest = event_types[3:]
        true_next_event_types = ev_rest[pred_start: min(len(ev_rest), pred_start + H)]

    input_ids = [pad_id] * L
    attn = [0] * L
    ev_ids = [default_event_type_id] * L

    for i in range(min(3, len(demo))):
        input_ids[i] = demo[i]
        attn[i] = 1
        if event_types is not None:
            ev_ids[i] = event_types[i]

    pos = 3
    for j, tok in enumerate(ctx_rest):
        if pos + j >= ctx_cap:
            break
        input_ids[pos + j] = tok
        attn[pos + j] = 1
        if event_types is not None:
            ev_ids[pos + j] = event_types[3 + ctx_left + j]

    return (
        torch.tensor(input_ids),
        torch.tensor(attn),
        torch.tensor(ev_ids),
        true_next_tokens,
        true_next_event_types,
    )


def _slide_left_preserve_demo(xg, ag, eg, *, pad_id):
    xs = xg[3:].clone()
    as_ = ag[3:].clone()
    es = eg[3:].clone()

    xs = torch.roll(xs, -1, 0)
    as_ = torch.roll(as_, -1, 0)
    es = torch.roll(es, -1, 0)

    xs[-1] = pad_id
    as_[-1] = 0
    es[-1] = es[-2] if es.numel() >= 2 else es[-1]

    xg[3:] = xs
    ag[3:] = as_
    eg[3:] = es

    return xg, ag, eg


@torch.no_grad()
def evaluate_next_event_nextn_vocab(
    model,
    records,
    *,
    device,
    max_len,
    pad_id,
    mask_id,
    default_event_type_id,
    topk,
    horizon,
    token_id_to_vocab,
    pairs_csv_path,
    use_target_event_type_at_mask,
):
    model.eval()

    os.makedirs(os.path.dirname(pairs_csv_path) or ".", exist_ok=True)
    f_csv = open(pairs_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    writer.writerow(
        ["Patient_ID", "Step", "Predicted Token", "True Token",
         "Predicted_Token_Vocab_ID", "True_Token_Vocab_ID"]
    )

    total_steps = 0
    correct_at_k_pairwise = {k: 0 for k in topk}
    sum_count_overlap_at_k = {k: 0.0 for k in topk}
    sum_pairwise_acc_at_k = {k: 0.0 for k in topk}
    patients_evaluated = 0
    skipped_unknown_true_vocab = 0

    try:
        for pi, rec in enumerate(tqdm(records, desc="patients")):
            toks, ev = _extract_sequence(rec)

            ex = _build_eval_example_nextn(
                toks, ev,
                max_len=max_len,
                pad_id=pad_id,
                default_event_type_id=default_event_type_id,
                horizon=horizon,
            )
            if ex is None:
                continue

            x0, a0, e0, true_next_tokens, true_next_event_types = ex
            H_eff = len(true_next_tokens)

            xg = x0.to(device)
            ag = a0.to(device)
            eg = e0.to(device)

            true_count = {}
            for t in true_next_tokens:
                v = token_id_to_vocab.get(int(t), -1)
                if v != -1:
                    true_count[v] = true_count.get(v, 0) + 1

            pred_available_count_at_k = {k: {} for k in topk}
            correct_at_k_patient = {k: 0 for k in topk}
            valid_steps_patient = 0

            for step in range(H_eff):
                true_tok = int(true_next_tokens[step])
                true_vocab = token_id_to_vocab.get(true_tok, -1)
                if true_vocab == -1:
                    skipped_unknown_true_vocab += 1
                    continue

                pad_pos = (ag == 0).nonzero().view(-1)
                if pad_pos.numel() == 0:
                    xg, ag, eg = _slide_left_preserve_demo(xg, ag, eg, pad_id=pad_id)
                    pad_pos = (ag == 0).nonzero().view(-1)
                    if pad_pos.numel() == 0:
                        break

                gen_pos = int(pad_pos[0])
                xg[gen_pos] = mask_id
                ag[gen_pos] = 1

                if use_target_event_type_at_mask and true_next_event_types is not None:
                    eg[gen_pos] = true_next_event_types[step]
                else:
                    last_att = (ag == 1).nonzero().view(-1)
                    eg[gen_pos] = eg[last_att[-1]] if last_att.numel() > 0 else default_event_type_id

                out = model(
                    input_ids=xg.unsqueeze(0),
                    attention_mask=ag.unsqueeze(0),
                    event_type_ids=eg.unsqueeze(0),
                )
                logits = out["logits"][0][gen_pos]

                logits[pad_id] = -1e9
                logits[mask_id] = -1e9

                kmax = max(topk)
                topk_tok_ids = torch.topk(logits, k=kmax).indices.cpu().tolist()

                pred_top1 = int(topk_tok_ids[0])
                pred_vocab = token_id_to_vocab.get(pred_top1, -1)

                writer.writerow(
                    [pi, step, pred_top1, true_tok, pred_vocab, true_vocab]
                )

                topk_vocabs = [token_id_to_vocab.get(int(t), -1) for t in topk_tok_ids]

                for k in topk:
                    if true_vocab in topk_vocabs[:k]:
                        correct_at_k_pairwise[k] += 1
                        correct_at_k_patient[k] += 1

                    present_vocabs = {v for v in topk_vocabs[:k] if v != -1}

                    d = pred_available_count_at_k[k]
                    for v in present_vocabs:
                        d[v] = d.get(v, 0) + 1

                total_steps += 1
                valid_steps_patient += 1
                xg[gen_pos] = pred_top1

            if valid_steps_patient == 0:
                continue

            patients_evaluated += 1

            for k in topk:
                acc_k = correct_at_k_patient[k] / float(valid_steps_patient)
                sum_pairwise_acc_at_k[k] += acc_k

                overlap = 0
                pred_cnt = pred_available_count_at_k[k]
                for v, c_true in true_count.items():
                    overlap += min(c_true, pred_cnt.get(v, 0))
                sum_count_overlap_at_k[k] += overlap / float(valid_steps_patient)

    finally:
        f_csv.close()

    metrics = {
        "patients_evaluated": float(patients_evaluated),
        "steps_total": float(total_steps),
        "steps_skipped_true_vocab_unknown": float(skipped_unknown_true_vocab),
    }

    for k in topk:
        metrics[f"pairwise_acc@{k}_global_vocab"] = (
            correct_at_k_pairwise[k] / max(1.0, total_steps)
        )
        metrics[f"pairwise_acc@{k}_mean_patient_vocab"] = (
            sum_pairwise_acc_at_k[k] / max(1.0, patients_evaluated)
        )
        metrics[f"count_score@{k}_mean_patient_vocab"] = (
            sum_count_overlap_at_k[k] / max(1.0, patients_evaluated)
        )

    return metrics


def main():
    args = parse_args()

    if args.pairs_csv is None:
        ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.pairs_csv = os.path.join(
            "..", "out", "evaluation",
            f"next_event_seq_vocab_pairs_{ckpt_base}.csv"
        )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    model = _load_ckpt_and_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)

    if args.subset_frac < 1.0:
        n = max(1, int(len(records) * args.subset_frac))
        records = random.sample(records, n)

    if args.max_patients is not None:
        records = records[: args.max_patients]

    from vocabulary import Vocabulary

    vocab = Vocabulary.load(args.vocab_path)
    token_id_to_vocab = build_token_id_to_group_from_vocab(vocab)

    topk = _parse_topk(args.topk)

    metrics = evaluate_next_event_nextn_vocab(
        model,
        records,
        device=device,
        max_len=args.max_len,
        pad_id=args.pad_id,
        mask_id=args.mask_id,
        default_event_type_id=args.default_event_type_id,
        topk=topk,
        horizon=args.horizon,
        token_id_to_vocab=token_id_to_vocab,
        pairs_csv_path=args.pairs_csv,
        use_target_event_type_at_mask=args.use_target_event_type_at_mask,
    )

    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()