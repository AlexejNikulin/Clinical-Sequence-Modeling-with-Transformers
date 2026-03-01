from __future__ import annotations

"""
evaluation/clinical_eval_utils.py

Utilities for:
- loading JSONL evaluation data
- MLM metrics (token-level MRR, token top-k)
- block/top-k metrics (via vocab blocks / token-id-to-block mapping)
- Datasets for evaluation:
    * ClinicalSequenceDataset (MLM-style batches)
    * NextEventDataset (next-token / next-event evaluation)

RANDOM WINDOW SAMPLING (core requirement):
- If a sequence is longer than max_len, do NOT always take the first max_len tokens.
- Instead, sample a random start position in the BODY (after a fixed prefix).
- The first keep_prefix_n tokens (e.g. demographic tokens) must always stay at
  positions [0..keep_prefix_n-1], regardless of the sampled window start.

Additionally:
- Even short sequences (len <= max_len) are "sampled":
  we choose a random start in the body and take the tail, then pad.
  (This reduces bias towards always showing the earliest part of sequences.)
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Common ignore index for MLM labels (masked positions use real token id, others use IGNORE_INDEX).
IGNORE_INDEX = -100


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts.

    Each line must be valid JSON.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------
# Block mapping helper (for "block" evaluation)
# ---------------------------------------------------------------------

def build_token_id_to_block_id_from_vocab(
    vocab: Any,
) -> Tuple[Dict[int, int], Dict[int, str], Dict[str, int]]:
    """
    Build token_id -> block_id mapping from your repo's Vocabulary structure.

    Supports a few common patterns:
    A) vocab.blocks is a list of blocks, each block is dict/object containing:
         - name
         - token_ids or ids
    B) vocab.block_name_to_token_ids is a dict: block_name -> list[token_ids]

    Returns:
      token_id_to_block: token_id -> block_id
      block_id_to_name: block_id -> human-readable block name
      block_name_to_id: name -> block_id
    """
    token_id_to_block: Dict[int, int] = {}
    block_id_to_name: Dict[int, str] = {}
    block_name_to_id: Dict[str, int] = {}

    # Pattern A: vocab.blocks is a list of blocks
    if hasattr(vocab, "blocks"):
        blocks = getattr(vocab, "blocks")
        if isinstance(blocks, list):
            for b_id, b in enumerate(blocks):
                if isinstance(b, dict):
                    name = str(b.get("name", f"block_{b_id}"))
                    ids = b.get("token_ids", b.get("ids", []))
                else:
                    name = str(getattr(b, "name", f"block_{b_id}"))
                    ids = getattr(b, "token_ids", getattr(b, "ids", []))

                block_id_to_name[b_id] = name
                block_name_to_id[name] = b_id

                for tid in ids:
                    token_id_to_block[int(tid)] = b_id

            return token_id_to_block, block_id_to_name, block_name_to_id

    # Pattern B: vocab.block_name_to_token_ids is a dict
    if hasattr(vocab, "block_name_to_token_ids"):
        d = getattr(vocab, "block_name_to_token_ids")
        if isinstance(d, dict):
            for b_id, (name, ids) in enumerate(d.items()):
                name = str(name)
                block_id_to_name[b_id] = name
                block_name_to_id[name] = b_id
                for tid in ids:
                    token_id_to_block[int(tid)] = b_id
            return token_id_to_block, block_id_to_name, block_name_to_id

    raise ValueError(
        "Could not infer vocabulary blocks. Adapt build_token_id_to_block_id_from_vocab() "
        "to match your Vocabulary structure."
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

@torch.no_grad()
def mrr_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Token-level Mean Reciprocal Rank (MRR) on eval positions only.

    logits: [B, T, V]
    labels: [B, T] with IGNORE_INDEX for non-eval positions

    MRR: average of (1 / rank(true_token)) across eval positions.
    Rank is 1 for best logit, 2 for second, ...
    """
    device = logits.device
    mask = labels != IGNORE_INDEX
    if mask.sum().item() == 0:
        return 0.0

    logits_flat = logits[mask]             # [N, V]
    labels_flat = labels[mask].to(device)  # [N]

    true_logits = logits_flat.gather(1, labels_flat.view(-1, 1)).squeeze(1)  # [N]
    greater = (logits_flat > true_logits.unsqueeze(1)).sum(dim=1)            # [N]
    rank = greater + 1
    return float((1.0 / rank.float()).mean().item())


@torch.no_grad()
def token_topk_acc_from_logits(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Token-level top-k accuracy on eval positions only.

    A position is correct if the true token id appears in top-k predictions.
    """
    mask = labels != IGNORE_INDEX
    if mask.sum().item() == 0:
        return 0.0

    logits_flat = logits[mask]  # [N, V]
    labels_flat = labels[mask]  # [N]

    topk = logits_flat.topk(k, dim=1).indices  # [N, k]
    correct = (topk == labels_flat.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


@torch.no_grad()
def block_top1_acc_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_id_to_block: Dict[int, int],
) -> float:
    """
    Convenience wrapper for block top-1 accuracy.

    Some evaluation scripts import block_top1_acc_from_logits.
    Internally we implement the generic block_topk_acc_from_logits,
    so this wrapper keeps backward compatibility.
    """
    return block_topk_acc_from_logits(logits, labels, token_id_to_block, k=1)

    mask = labels != IGNORE_INDEX
    if mask.sum().item() == 0:
        return 0.0

    logits_flat = logits[mask]  # [N, V]
    labels_flat = labels[mask]  # [N]

    # True blocks for each eval label
    true_blocks = torch.tensor(
        [token_id_to_block.get(int(tid), -1) for tid in labels_flat.tolist()],
        device=logits.device,
        dtype=torch.long,
    )  # [N]

    # Top-k predicted token ids -> blocks
    topk_ids = logits_flat.topk(k, dim=1).indices  # [N, k]
    pred_blocks = torch.empty_like(topk_ids, dtype=torch.long)

    # Note: loop is fine for eval sizes; vectorization possible but not necessary here.
    for i in range(topk_ids.size(0)):
        for j in range(topk_ids.size(1)):
            pred_blocks[i, j] = int(token_id_to_block.get(int(topk_ids[i, j].item()), -1))

    correct = (pred_blocks == true_blocks.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


# ---------------------------------------------------------------------
# Window Sampling (core requirement)
# ---------------------------------------------------------------------

def sample_window_keep_prefix_start(
    seq_len: int,
    *,
    max_len: int,
    keep_prefix_n: int,
    rng: random.Random,
) -> int:
    """
    Sample ONLY the start index within the BODY (0-based, in body-space).

    We sample exactly one start value and reuse it for all aligned arrays:
      - input_ids
      - event_type_ids
      - labels

    Definitions:
      prefix_len = min(keep_prefix_n, seq_len)
      body_len   = seq_len - prefix_len
      body_cap   = max_len - prefix_len    (how many body tokens can fit in window)

    Cases:
      - body_len == 0 -> return 0
      - max_len <= prefix_len -> return 0 (window can hold only prefix)
      - if body_len <= body_cap:
          * short/equal sequences: still sample start in [0..body_len-1]
          * later we take tail body[start:] and pad
      - else:
          * long sequences: sample start in [0..body_len-body_cap] (crop window)
    """
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if keep_prefix_n < 0:
        raise ValueError("keep_prefix_n must be >= 0")

    prefix_len = min(keep_prefix_n, seq_len)
    body_len = max(0, seq_len - prefix_len)

    if body_len == 0:
        return 0
    if max_len <= prefix_len:
        return 0

    body_cap = max_len - prefix_len

    if body_len <= body_cap:
        return rng.randint(0, body_len - 1)

    return rng.randint(0, body_len - body_cap)


def apply_window_keep_prefix(
    seq: List[int],
    *,
    max_len: int,
    keep_prefix_n: int,
    start_in_body: int,
) -> List[int]:
    """
    Apply the sampled start_in_body (body-space index) to a sequence.

    Rule:
      - keep prefix fixed at beginning: seq[:prefix_len]
      - body = seq[prefix_len:]
      - if body short: return prefix + body[start:]
      - if body long:  return prefix + body[start:start+body_cap]
      - padding happens later in Dataset
    """
    prefix_len = min(keep_prefix_n, len(seq))
    prefix = seq[:prefix_len]
    body = seq[prefix_len:]

    # If window cannot hold any body tokens
    if max_len <= prefix_len:
        return prefix[:max_len]

    body_cap = max_len - prefix_len

    if not body:
        return prefix

    # Safety clamp: ensure start_in_body valid
    start_in_body = max(0, min(start_in_body, len(body) - 1))

    # Short body: take tail
    if len(body) <= body_cap:
        return prefix + body[start_in_body:]

    # Long body: crop (ensure we have body_cap tokens)
    start_in_body = min(start_in_body, len(body) - body_cap)
    return prefix + body[start_in_body : start_in_body + body_cap]


# ---------------------------------------------------------------------
# Dataset(s)
# ---------------------------------------------------------------------

def _ensure_int_list(x: Any, name: str) -> List[int]:
    """
    Ensure x is a list[int]. If x is None -> [].
    """
    if x is None:
        return []
    if not isinstance(x, list):
        raise TypeError(f"{name} must be a list, got {type(x)}")
    out: List[int] = []
    for i, v in enumerate(x):
        if not isinstance(v, int):
            raise TypeError(f"{name}[{i}] must be int, got {type(v)}")
        out.append(v)
    return out


class ClinicalSequenceDataset(Dataset):
    """
    MLM evaluation dataset.

    Supported token sequence keys (first found wins):
      - input_ids
      - tokens
      - token_ids
      - ids

    Optional keys:
      - event_type_ids or event_types (must match token length)
      - labels (must match token length)

    If sample_windows=True:
      - sample one start in BODY (after prefix)
      - apply same start to input_ids, event_type_ids, labels
      - pad to max_len
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        max_len: int,
        pad_id: int,
        default_event_type_id: int,
        sample_windows: bool = False,
        keep_prefix_n: int = 3,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.default_event_type_id = int(default_event_type_id)
        self.sample_windows = bool(sample_windows)
        self.keep_prefix_n = int(keep_prefix_n)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]

        # 1) Tokens
        input_ids_raw = r.get("input_ids") or r.get("tokens") or r.get("token_ids") or r.get("ids")
        input_ids = _ensure_int_list(input_ids_raw, "input_ids/tokens/token_ids/ids")
        if not input_ids:
            input_ids = [self.pad_id]

        # 2) Event types (optional but if provided must be aligned)
        ev_raw = r.get("event_type_ids", r.get("event_types"))
        event_type_ids = _ensure_int_list(ev_raw, "event_type_ids/event_types")
        if not event_type_ids:
            event_type_ids = [self.default_event_type_id] * len(input_ids)
        elif len(event_type_ids) != len(input_ids):
            raise ValueError(f"event_type_ids length != input_ids length at idx={idx}")

        # 3) Labels (optional but if provided must be aligned)
        labels_list: Optional[List[int]] = None
        if "labels" in r and r["labels"] is not None:
            labels_list = _ensure_int_list(r["labels"], "labels")
            if len(labels_list) != len(input_ids):
                raise ValueError(f"labels length != input_ids length at idx={idx}")

        # 4) Window sampling
        if self.sample_windows:
            # Deterministic RNG per record (same seed + same idx => reproducible)
            rng = random.Random(self.seed + idx)

            start = sample_window_keep_prefix_start(
                seq_len=len(input_ids),
                max_len=self.max_len,
                keep_prefix_n=self.keep_prefix_n,
                rng=rng,
            )

            # Apply exact same start to all arrays to keep alignment intact
            input_ids = apply_window_keep_prefix(
                input_ids,
                max_len=self.max_len,
                keep_prefix_n=self.keep_prefix_n,
                start_in_body=start,
            )
            event_type_ids = apply_window_keep_prefix(
                event_type_ids,
                max_len=self.max_len,
                keep_prefix_n=self.keep_prefix_n,
                start_in_body=start,
            )
            if labels_list is not None:
                labels_list = apply_window_keep_prefix(
                    labels_list,
                    max_len=self.max_len,
                    keep_prefix_n=self.keep_prefix_n,
                    start_in_body=start,
                )
        else:
            # Deterministic "first window" behavior
            input_ids = input_ids[: self.max_len]
            event_type_ids = event_type_ids[: self.max_len]
            if labels_list is not None:
                labels_list = labels_list[: self.max_len]

        # 5) Pad to max_len
        T = len(input_ids)
        pad_len = self.max_len - T

        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            event_type_ids = event_type_ids + [self.default_event_type_id] * pad_len
            if labels_list is not None:
                labels_list = labels_list + [IGNORE_INDEX] * pad_len

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = ([1] * min(T, self.max_len)) + ([0] * max(0, pad_len))
        attention_mask = attention_mask[: self.max_len]

        if labels_list is None:
            labels_list = [IGNORE_INDEX] * self.max_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "event_type_ids": torch.tensor(event_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


class NextEventDataset(Dataset):
    """
    Next-event evaluation dataset:

    Returns:
      - input_ids: [max_len]
      - attention_mask: [max_len]
      - event_type_ids: [max_len]
      - target_id: scalar int (the next token after the context window)

    Sampling rule:
      - Prefix stays fixed.
      - Random start in BODY.
      - Must leave 1 token as the target (if possible).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        max_len: int,
        pad_id: int,
        default_event_type_id: int,
        sample_windows: bool = False,
        keep_prefix_n: int = 3,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.default_event_type_id = int(default_event_type_id)
        self.sample_windows = bool(sample_windows)
        self.keep_prefix_n = int(keep_prefix_n)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]

        seq_raw = r.get("input_ids") or r.get("tokens") or r.get("token_ids") or r.get("ids")
        seq = _ensure_int_list(seq_raw, "input_ids/tokens/token_ids/ids")
        if not seq:
            seq = [self.pad_id]

        ev_raw = r.get("event_type_ids", r.get("event_types"))
        event_type_ids = _ensure_int_list(ev_raw, "event_type_ids/event_types")
        if not event_type_ids:
            event_type_ids = [self.default_event_type_id] * len(seq)
        elif len(event_type_ids) != len(seq):
            raise ValueError(f"event_type_ids length != seq length at idx={idx}")

        # Degenerate case: no real "next" token possible
        if len(seq) < 2:
            x = seq[:1]
            y = self.pad_id
            ev = event_type_ids[:1]
        else:
            prefix_len = min(self.keep_prefix_n, len(seq))
            prefix = seq[:prefix_len]
            body = seq[prefix_len:]

            ev_prefix = event_type_ids[:prefix_len]
            ev_body = event_type_ids[prefix_len:]

            if not body:
                # Only prefix exists, no next token in body
                x = prefix[: self.max_len]
                y = self.pad_id
                ev = ev_prefix[: len(x)]
            else:
                # If max_len can't fit any body tokens, input is prefix only, target is first body token
                if self.max_len <= prefix_len:
                    x = prefix[: self.max_len]
                    y = body[0]
                    ev = ev_prefix[: len(x)]
                else:
                    body_cap = self.max_len - prefix_len

                    # Need to leave 1 token for target if possible
                    max_body_for_input = min(body_cap, len(body) - 1)

                    if max_body_for_input <= 0:
                        # Input = prefix only, target = first body token
                        x = prefix
                        y = body[0]
                        ev = ev_prefix
                    else:
                        if self.sample_windows:
                            rng = random.Random(self.seed + idx)
                            # start is chosen so that target exists after input slice
                            start = rng.randint(0, (len(body) - 1) - max_body_for_input)
                        else:
                            start = 0

                        x = prefix + body[start : start + max_body_for_input]
                        y = body[start + max_body_for_input]
                        ev = ev_prefix + ev_body[start : start + max_body_for_input]

        # Pad x to max_len
        T = len(x)
        pad_len = self.max_len - T
        if pad_len > 0:
            x = x + [self.pad_id] * pad_len
            ev = ev + [self.default_event_type_id] * pad_len

        attn = ([1] * min(T, self.max_len)) + ([0] * max(0, pad_len))
        attn = attn[: self.max_len]

        return {
            "input_ids": torch.tensor(x[: self.max_len], dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "event_type_ids": torch.tensor(ev[: self.max_len], dtype=torch.long),
            "target_id": torch.tensor(int(y), dtype=torch.long),
        }


# ---------------------------------------------------------------------
# Self-test helper (optional, for debugging)
# ---------------------------------------------------------------------

def test() -> str:
    """
    Quick self-test for random window sampling.

    This is meant to be called via:
      python - <<'PY'
      import evaluation.clinical_eval_utils as m
      print(m.test())
      PY
    """
    seq = [101, 102, 103, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    max_len = 6
    keep_prefix_n = 3

    starts = set()
    for i in range(30):
        rng = random.Random(13 + i)
        s = sample_window_keep_prefix_start(
            len(seq),
            max_len=max_len,
            keep_prefix_n=keep_prefix_n,
            rng=rng,
        )
        win = apply_window_keep_prefix(
            seq,
            max_len=max_len,
            keep_prefix_n=keep_prefix_n,
            start_in_body=s,
        )
        assert win[:3] == [101, 102, 103]
        assert len(win) <= max_len
        starts.add(s)

    return f"OK: prefix fixed, unique starts={sorted(starts)}"