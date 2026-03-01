from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts. Allows comment lines starting with '#'."""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                line = line.lstrip("#").lstrip("£").strip()
                if not line:
                    continue
            data.append(json.loads(line))
    return data


# ----------------------------
# Window sampling helpers
# ----------------------------
def sample_window_keep_prefix_start(
    seq_len: int,
    *,
    max_len: int,
    keep_prefix_n: int,
    rng: random.Random,
) -> int:
    """
    Sample ONLY the start index within the BODY (0-based in body-space).
    Prefix stays fixed at positions [0..keep_prefix_n-1].

    - short sequences (body_len <= body_cap): sample start in [0..body_len-1] and later take tail.
    - long sequences: sample start in [0..body_len-body_cap] for cropping.
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
    Apply a sampled start_in_body to seq.
    """
    prefix_len = min(keep_prefix_n, len(seq))
    prefix = seq[:prefix_len]
    body = seq[prefix_len:]

    if max_len <= prefix_len:
        return prefix[:max_len]

    body_cap = max_len - prefix_len

    if not body:
        return prefix

    start_in_body = max(0, min(start_in_body, len(body) - 1))

    if len(body) <= body_cap:
        return prefix + body[start_in_body:]

    start_in_body = min(start_in_body, len(body) - body_cap)
    return prefix + body[start_in_body : start_in_body + body_cap]


@dataclass
class TopKResult:
    k: int
    correct: int
    total: int

    @property
    def acc(self) -> float:
        return self.correct / self.total if self.total else float("nan")


class ClinicalSequenceDataset(Dataset):
    """
    Dataset for event-type/block evaluation.

    New args:
      sample_windows: random window sampling with fixed prefix
      keep_prefix_n: number of prefix tokens that must stay at the beginning
      seed: base seed, RNG per record = Random(seed + idx)
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
        token_ids = r.get("token_ids") or r.get("ids") or r.get("input_ids") or r.get("tokens")
        if token_ids is None:
            raise KeyError(
                f"Record missing token sequence. Expected one of token_ids/ids/input_ids/tokens. Keys={list(r.keys())}"
            )
        return list(token_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]

        token_ids_full = self._get_token_list(r)
        event_type_ids_full = r.get("event_type_ids", None)
        labels_full = r.get("labels", None)
        attn_full = r.get("attention_mask", None)

        if event_type_ids_full is not None:
            event_type_ids_full = list(event_type_ids_full)
        if labels_full is not None:
            labels_full = list(labels_full)
        if attn_full is not None:
            attn_full = list(attn_full)

        token_ids = token_ids_full
        event_type_ids = event_type_ids_full
        labels = labels_full
        attention_mask = attn_full

        # --- window sampling (ONE start for all aligned fields) ---
        if self.sample_windows:
            rng = random.Random(self.seed + idx)
            s = sample_window_keep_prefix_start(
                seq_len=len(token_ids_full),
                max_len=self.max_len,
                keep_prefix_n=self.keep_prefix_n,
                rng=rng,
            )

            token_ids = apply_window_keep_prefix(
                token_ids_full, max_len=self.max_len, keep_prefix_n=self.keep_prefix_n, start_in_body=s
            )
            if event_type_ids_full is not None:
                event_type_ids = apply_window_keep_prefix(
                    event_type_ids_full, max_len=self.max_len, keep_prefix_n=self.keep_prefix_n, start_in_body=s
                )
            if labels_full is not None:
                labels = apply_window_keep_prefix(
                    labels_full, max_len=self.max_len, keep_prefix_n=self.keep_prefix_n, start_in_body=s
                )
            if attn_full is not None:
                attention_mask = apply_window_keep_prefix(
                    attn_full, max_len=self.max_len, keep_prefix_n=self.keep_prefix_n, start_in_body=s
                )

        # --- enforce aligned lengths + pad to max_len ---
        T = min(len(token_ids), self.max_len)
        token_ids = token_ids[:T]

        if event_type_ids is None:
            event_type_ids = [self.default_event_type_id] * T
        else:
            event_type_ids = event_type_ids[:T]
            if len(event_type_ids) < T:
                event_type_ids += [self.default_event_type_id] * (T - len(event_type_ids))

        if labels is None:
            labels = [IGNORE_INDEX] * T
        else:
            labels = labels[:T]
            if len(labels) < T:
                labels += [IGNORE_INDEX] * (T - len(labels))

        if attention_mask is None:
            attention_mask = [1] * T
        else:
            attention_mask = attention_mask[:T]
            if len(attention_mask) < T:
                attention_mask += [1] * (T - len(attention_mask))

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


@torch.no_grad()
def topk_accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    ignore_index: int = IGNORE_INDEX,
) -> TopKResult:
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
