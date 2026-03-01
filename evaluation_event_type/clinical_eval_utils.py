# evaluation_event_type/clinical_eval_utils.py
#
# Utilities used for event-type / block evaluation.
# NOW includes: RANDOM WINDOW SAMPLING with fixed demographic prefix.
#
# Core requirement implemented:
# - If seq_len > max_len: do NOT always take the first window. Sample a random start in the BODY.
# - keep_prefix_n (e.g., demographics) always stays at the beginning positions [0..keep_prefix_n-1].
# - Also sample short sequences (seq_len <= max_len): pick a random start in BODY and then take the tail.
#
# Notes:
# - We sample ONE start index per record and apply it consistently to:
#   input_ids, event_type_ids, labels, attention_mask (aligned slicing).
# - Padding happens at the end so tensors always have length max_len.

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

# Common ignore index for MLM labels
IGNORE_INDEX = -100


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts.

    Supports optional comment lines starting with '#':
      - '# {...}'  -> comment marker is stripped and parsed as JSON (optional legacy format)
      - '# ...'    -> ignored if no JSON remains

    Returns:
      List[Dict[str, Any]] records
    """
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # allow comment lines like "# ..." or "#£{...}"
                line = line.lstrip("#").lstrip("£").strip()
                if not line:
                    continue

            data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------
# Window Sampling Helpers (core requirement)
# ---------------------------------------------------------------------
def sample_window_keep_prefix_start(
    seq_len: int,
    *,
    max_len: int,
    keep_prefix_n: int,
    rng: random.Random,
) -> int:
    """
    Sample ONLY the start index within the BODY (0-based in body-space).

    Definitions:
      prefix_len = min(keep_prefix_n, seq_len)
      body_len   = seq_len - prefix_len
      body_cap   = max_len - prefix_len   (how many body tokens fit in the window)

    Behavior:
      - If body_len == 0: start=0
      - If max_len <= prefix_len: start=0 (no room for body tokens anyway)
      - If body_len <= body_cap:
          still sample start uniformly from [0..body_len-1]
          -> short sequences are also "sampled" (we later take tail and pad)
      - If body_len > body_cap:
          sample start uniformly from [0..(body_len - body_cap)]
          -> cropping window of exactly body_cap tokens from the body

    Returns:
      start_in_body (int): index into the BODY (not into the full sequence)
    """
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if keep_prefix_n < 0:
        raise ValueError("keep_prefix_n must be >= 0")

    prefix_len = min(keep_prefix_n, seq_len)
    body_len = max(0, seq_len - prefix_len)

    # no body -> nothing to sample
    if body_len == 0:
        return 0

    # no room for body in window
    if max_len <= prefix_len:
        return 0

    body_cap = max_len - prefix_len

    # short/equal: sample any start, later take tail
    if body_len <= body_cap:
        return rng.randint(0, body_len - 1)

    # long: sample a start where a full body_cap window fits
    return rng.randint(0, body_len - body_cap)


def apply_window_keep_prefix(
    seq: List[int],
    *,
    max_len: int,
    keep_prefix_n: int,
    start_in_body: int,
) -> List[int]:
    """
    Apply a sampled start_in_body (in body-space) to a sequence.

    - Prefix stays fixed at positions [0..prefix_len-1]
    - Window body starts at start_in_body
    - If body is short (len(body) <= body_cap):
        return prefix + body[start:]  (tail)
    - If body is long:
        return prefix + body[start:start+body_cap]  (crop)
    - Padding happens later in Dataset

    Returns:
      windowed list (length <= max_len)
    """
    prefix_len = min(keep_prefix_n, len(seq))
    prefix = seq[:prefix_len]
    body = seq[prefix_len:]

    # if window can't even fit prefix fully, truncate prefix (edge-case)
    if max_len <= prefix_len:
        return prefix[:max_len]

    body_cap = max_len - prefix_len

    if not body:
        return prefix

    # clamp start for safety (avoid out-of-range)
    start_in_body = max(0, min(start_in_body, len(body) - 1))

    # short: take tail
    if len(body) <= body_cap:
        return prefix + body[start_in_body:]

    # long: ensure we can take a full body_cap slice
    start_in_body = min(start_in_body, len(body) - body_cap)
    return prefix + body[start_in_body : start_in_body + body_cap]


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
@dataclass
class TopKResult:
    """
    Container for top-k token accuracy (debug metric).
    """
    k: int
    correct: int
    total: int

    @property
    def acc(self) -> float:
        return self.correct / self.total if self.total else float("nan")


class ClinicalSequenceDataset(Dataset):
    """
    Dataset used by event-level evaluation scripts.

    Supported record keys (token sequence):
      - token_ids
      - ids
      - input_ids
      - tokens

    Optional aligned fields:
      - event_type_ids: segment/event-type ids
      - labels: MLM labels
      - attention_mask: 1 for real token, 0 for pad

    New args:
      sample_windows:
        If True, sample a random start position in the BODY after the prefix.
      keep_prefix_n:
        Number of prefix tokens that must stay fixed at the beginning.
      seed:
        Base seed for deterministic sampling. We use rng = Random(seed + idx).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        max_len: int,
        pad_id: int,
        default_event_type_id: int = 0,
        *,
        sample_windows: bool = False,
        keep_prefix_n: int = 3,
        seed: int = 13,
    ):
        self.records = records
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.default_event_type_id = int(default_event_type_id)

        self.sample_windows = bool(sample_windows)
        self.keep_prefix_n = int(keep_prefix_n)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _get_token_list(self, r: Dict[str, Any]) -> List[int]:
        """
        Extract token sequence from a record supporting multiple possible keys.
        """
        token_ids = r.get("token_ids") or r.get("ids") or r.get("input_ids") or r.get("tokens")
        if token_ids is None:
            raise KeyError(
                f"Record missing token sequence. Expected one of token_ids/ids/input_ids/tokens. Keys={list(r.keys())}"
            )
        return list(token_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]

        # ---- Full sequences (unwindowed) ----
        token_ids_full = self._get_token_list(r)

        event_type_ids_full = r.get("event_type_ids", None)
        labels_full = r.get("labels", None)
        attn_full = r.get("attention_mask", None)

        # Convert optional fields to lists if present
        if event_type_ids_full is not None:
            event_type_ids_full = list(event_type_ids_full)
        if labels_full is not None:
            labels_full = list(labels_full)
        if attn_full is not None:
            attn_full = list(attn_full)

        # ---- Defaults (may be overwritten by sampling) ----
        token_ids = token_ids_full
        event_type_ids = event_type_ids_full
        labels = labels_full
        attention_mask = attn_full

        # ----------------------------
        # Optional: random window sampling (ONE start reused for all aligned fields)
        # ----------------------------
        if self.sample_windows:
            rng = random.Random(self.seed + idx)

            # Sample ONE start in body-space and reuse for all aligned fields
            s = sample_window_keep_prefix_start(
                seq_len=len(token_ids_full),
                max_len=self.max_len,
                keep_prefix_n=self.keep_prefix_n,
                rng=rng,
            )

            token_ids = apply_window_keep_prefix(
                token_ids_full,
                max_len=self.max_len,
                keep_prefix_n=self.keep_prefix_n,
                start_in_body=s,
            )

            if event_type_ids_full is not None:
                event_type_ids = apply_window_keep_prefix(
                    event_type_ids_full,
                    max_len=self.max_len,
                    keep_prefix_n=self.keep_prefix_n,
                    start_in_body=s,
                )

            if labels_full is not None:
                labels = apply_window_keep_prefix(
                    labels_full,
                    max_len=self.max_len,
                    keep_prefix_n=self.keep_prefix_n,
                    start_in_body=s,
                )

            if attn_full is not None:
                attention_mask = apply_window_keep_prefix(
                    attn_full,
                    max_len=self.max_len,
                    keep_prefix_n=self.keep_prefix_n,
                    start_in_body=s,
                )

        # ----------------------------
        # Ensure aligned fields exist and pad right to max_len
        # ----------------------------
        T = min(len(token_ids), self.max_len)
        token_ids = token_ids[:T]

        # event_type_ids: if missing, fill with default; if shorter, pad up to T
        if event_type_ids is None:
            event_type_ids = [self.default_event_type_id] * T
        else:
            event_type_ids = event_type_ids[:T]
            if len(event_type_ids) < T:
                event_type_ids += [self.default_event_type_id] * (T - len(event_type_ids))

        # labels: if missing, fill with IGNORE_INDEX; if shorter, pad up to T
        if labels is None:
            labels = [IGNORE_INDEX] * T
        else:
            labels = labels[:T]
            if len(labels) < T:
                labels += [IGNORE_INDEX] * (T - len(labels))

        # attention_mask: if missing, assume all tokens are real (1) for length T
        if attention_mask is None:
            attention_mask = [1] * T
        else:
            attention_mask = attention_mask[:T]
            if len(attention_mask) < T:
                attention_mask += [1] * (T - len(attention_mask))

        # Right-pad all fields to max_len
        pad_n = self.max_len - T
        if pad_n > 0:
            token_ids += [self.pad_id] * pad_n
            event_type_ids += [self.default_event_type_id] * pad_n
            labels += [IGNORE_INDEX] * pad_n
            attention_mask += [0] * pad_n

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "event_type_ids": torch.tensor(event_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# ---------------------------------------------------------------------
# Token-level metrics (kept for debugging)
# ---------------------------------------------------------------------
@torch.no_grad()
def topk_accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    ignore_index: int = IGNORE_INDEX,
) -> TopKResult:
    """
    Token-level top-k accuracy at evaluation positions only.

    logits: [B, L, V]
    labels: [B, L] (ignore_index indicates positions to ignore)

    Returns:
      TopKResult(k, correct, total)
    """
    B, L, V = logits.shape
    lbl = labels.view(-1)
    log = logits.view(-1, V)

    mask = lbl != ignore_index
    if mask.sum().item() == 0:
        return TopKResult(k=k, correct=0, total=0)

    lbl = lbl[mask]
    log = log[mask]

    topk = log.topk(k, dim=1).indices
    correct = (topk == lbl.unsqueeze(1)).any(dim=1).sum().item()
    total = lbl.numel()
    return TopKResult(k=k, correct=int(correct), total=int(total))