'''
Docstring für evaluation.next_event_eval

PATH eval_val.jsonl: 
python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_baseline.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_d384.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_event_types_7.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_heads_12_n_layer_6.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_heads_12.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_n_layer_6.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_recency.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/eval_val.jsonl --ckpt checkpoints/mlm_span.pt --topk 1,5,10 --mask_id 1


PATH test_ids.jsonl:
python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_baseline.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_event_types_7.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_heads_12_n_layer_6.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_heads_12.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_n_layer_6.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_p_mlm_0_1.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_p_mlm_0_2.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_mask_mode_span.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_mask_mode_recency.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_dropout_0_5.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_dropout_0_15.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_dropout_0_15.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_d_model_384.pt --topk 1,5,10 --mask_id 1

python -m evaluation.next_event_eval --jsonl data/test_ids.jsonl --ckpt checkpoints/mlm_max_len_384.pt --topk 1,5,10 --mask_id 1
'''

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import random  # <-- CHANGED

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.clinical_eval_utils import (
    ClinicalSequenceDataset,
    load_jsonl,
    build_token_id_to_group_from_vocab,
)

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Next-event generation: predict following tokens until stop token is predicted.")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--vocab_path", type=str, default=None, help="Path to vocabulary.json (optional).")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=1)
    p.add_argument("--default_event_type_id", type=int, default=1)
    p.add_argument("--topk", type=str, default="1,5,10")

    # --- CHANGED: generation control ---
    p.add_argument("--stop_ids", type=str, default="70000, 70001, 70002, 70003, 70004, 70005, 70006, 70007, 70008, 70010", help="Comma-separated stop token IDs.")
    p.add_argument("--gen_max_steps", type=int, default=50, help="Safety cap for generated tokens per sequence.")
    # -------------------------------

    return p.parse_args()



@torch.no_grad()
def evaluate_next_event(
    model: CompactTransformerEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    pad_id: int,
    stop_ids: set[int],
    gen_max_steps: int,
    token_id_to_group: Optional[Dict[int, int]] = None,
    max_batches: int | None = None,
    log_random_n: int = 0,
) -> Dict[str, float]:
    model.eval()

    # --- CHANGED: pick random global indices from the actually evaluated subset ---
    n_log = max(0, int(log_random_n))
    to_log_global: set[int] = set()
    if n_log > 0:
        try:
            ds_len = len(loader.dataset)  # type: ignore[attr-defined]
            if max_batches is not None:
                bs = loader.batch_size or 1
                eval_cap = min(ds_len, int(max_batches) * int(bs))
            else:
                eval_cap = ds_len
            n_log = min(n_log, eval_cap)
            to_log_global = set(random.sample(range(eval_cap), k=n_log))
        except Exception:
            to_log_global = set()
    # ---------------------------------------------------------------------------

    seen_samples = 0
    total_generated = 0
    total_sequences = 0
    total_stopped = 0

    for step, batch in enumerate(tqdm(loader), start=1):
        if max_batches is not None and step > max_batches:
            break

        input_ids = batch["input_ids"].to(device)          # [B, L]
        attn = batch["attention_mask"].to(device)          # [B, L]
        ev = batch["event_type_ids"].to(device)            # [B, L]

        B, L = input_ids.shape
        total_sequences += B

        # --- CHANGED: per-sample autoregressive-ish generation without masking ---
        # IMPORTANT: This uses logits at the current last valid position to predict the next token,
        # then appends that token into the next padding slot (attn==0), repeating until stop token.
        for b in range(B):
            gidx = seen_samples + b

            xg = input_ids[b].clone()
            ag = attn[b].clone()
            eg = ev[b].clone()

            # find last valid position (ag==1)
            valid = (ag == 1).nonzero(as_tuple=False).view(-1)
            if valid.numel() < 1:
                continue
            cur_last = int(valid[-1].item())

            preds_list: List[int] = []

            for _ in range(max(1, int(gen_max_steps))):
                # find next append position (first padding slot)
                pad_pos = (ag == 0).nonzero(as_tuple=False).view(-1)

                # --- CHANGED: if no padding slot is available, free one via sliding window ---
                if pad_pos.numel() == 0:
                    # shift left by 1 to free the last slot
                    xg = torch.roll(xg, shifts=-1, dims=0)
                    ag = torch.roll(ag, shifts=-1, dims=0)
                    eg = torch.roll(eg, shifts=-1, dims=0)

                    # make the last position an explicit free slot
                    xg[-1] = int(pad_id)
                    ag[-1] = 0
                    eg[-1] = eg[-2] if L >= 2 else eg[-1]

                    # recompute last valid position
                    valid2 = (ag == 1).nonzero(as_tuple=False).view(-1)
                    if valid2.numel() < 1:
                        break
                    cur_last = int(valid2[-1].item())

                    pad_pos = (ag == 0).nonzero(as_tuple=False).view(-1)
                    if pad_pos.numel() == 0:
                        break
                # ---------------------------------------------------------------------------

                gen_pos = int(pad_pos[0].item())

                outg = model(
                    input_ids=xg.unsqueeze(0),
                    attention_mask=ag.unsqueeze(0),
                    event_type_ids=eg.unsqueeze(0),
                    labels=None,
                    return_hidden=False,
                )
                lg = outg["logits"][0]  # [L, V]

                # Predict "next token" using logits at current last valid position (no masking)
                pred_tok = int(torch.argmax(lg[cur_last]).item())
                preds_list.append(pred_tok)

                if pred_tok in stop_ids:
                    total_stopped += 1
                    break

                # append predicted token into padding slot
                xg[gen_pos] = pred_tok
                ag[gen_pos] = 1
                eg[gen_pos] = eg[cur_last]  # keep event type consistent
                cur_last = gen_pos

            total_generated += len(preds_list)

            if gidx in to_log_global:
                seq_seed = input_ids[b].detach().cpu().tolist()
                seq_final = xg.detach().cpu().tolist()

                print("\n" + "=" * 80)
                print(f"[Random sample gidx={gidx}]")
                print(f"Seed sequence (as given):   {seq_seed}")
                print(f"Predictions token_id list:  {preds_list}")
                #print(f"Final sequence (with preds):{seq_final}")
                print("=" * 80 + "\n")
        # ----------------------------------------------------------------------

        seen_samples += B

    return {
        "sequences_seen": float(total_sequences),
        "tokens_generated_total": float(total_generated),
        "tokens_generated_avg": float(total_generated) / max(1.0, float(total_sequences)),
        "sequences_stopped": float(total_stopped),
        "sequences_stopped_frac": float(total_stopped) / max(1.0, float(total_sequences)),
    }


def _load_ckpt_and_model(ckpt_path: str) -> CompactTransformerEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt and "model_state_dict" in ckpt:
        cfg = CompactTransformerConfig(**ckpt["cfg"])
        model = CompactTransformerEncoder(cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model
    raise ValueError("Unsupported checkpoint format. Expected dict with keys: cfg, model_state_dict.")


def main() -> None:
    args = parse_args()

    model = _load_ckpt_and_model(args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_jsonl(args.jsonl)
    ds = ClinicalSequenceDataset(
        records,
        max_len=args.max_len,
        pad_id=args.pad_id,
        default_event_type_id=args.default_event_type_id
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    token_id_to_group = None
    if args.vocab_path:
        from vocabulary import Vocabulary
        vocab = Vocabulary.load(args.vocab_path)
        token_id_to_group = build_token_id_to_group_from_vocab(vocab)

    stop_ids = {int(x) for x in args.stop_ids.split(",") if x.strip()}

    metrics = evaluate_next_event(
        model,
        loader,
        device=device,
        pad_id=int(args.pad_id),
        stop_ids=stop_ids,
        gen_max_steps=int(args.gen_max_steps),
        token_id_to_group=token_id_to_group,
        max_batches=5,
        log_random_n=3,
    )

    print(f"device={device}")
    for k, v in metrics.items():
        print(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")


if __name__ == "__main__":
    main()
