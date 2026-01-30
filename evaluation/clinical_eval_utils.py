from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # allow comment lines like "# ..." or "# {...}"
                line = line.lstrip("#").lstrip("£").strip()
                if not line:
                    continue
            data.append(json.loads(line))
    return data



def build_joint_sequences(
    *,
    demographics: List[List[int]],
    events: List[List[int]],
    max_len: int,
    pad_id: int,
    sep_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Concatenate demographic tokens (segment 0) and event tokens (segment 1).

    Output shapes are Python lists that can be wrapped into torch.Tensors later.
    """
    if len(demographics) != len(events):
        raise ValueError("demographics/events length mismatch")

    input_ids_all: List[List[int]] = []
    attn_all: List[List[int]] = []
    seg_all: List[List[int]] = []

    for demo_tok, event_tok in zip(demographics, events):
        seq: List[int] = []
        seg: List[int] = []

        seq.extend(demo_tok)
        seg.extend([0] * len(demo_tok))

        if sep_id is not None:
            seq.append(sep_id)
            seg.append(0)

        seq.extend(event_tok)
        seg.extend([1] * len(event_tok))

        seq = seq[:max_len]
        seg = seg[:max_len]
        attn = [1] * len(seq)

        while len(seq) < max_len:
            seq.append(pad_id)
            attn.append(0)
            seg.append(0)

        input_ids_all.append(seq)
        attn_all.append(attn)
        seg_all.append(seg)

    return input_ids_all, attn_all, seg_all


class ClinicalSequenceDataset(Dataset):
    """
    Dataset for evaluation/training.   ---> keys: ['patient_id', 'token_ids']

    Supports two JSONL formats:

    A) Flat sequence:
        - token_ids: List[int]
        (optional) event_type_ids: List[int]  # if you have it; otherwise we auto-fill zeros

    B) Split sequences:
        - demo_tokens: List[int]
        - event_tokens: List[int]
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        max_len: int,
        pad_id: int,
        token_key: str = "token_ids",
        event_type_key: Optional[str] = None,
        demo_key: str = "demo_tokens",
        event_key: str = "event_tokens",
        has_split_fields: Optional[bool] = None,
    ) -> None:
        if len(records) == 0:
            raise ValueError("records is empty")

        # Auto-detect format if not specified
        if has_split_fields is None:
            r0 = records[0]
            has_split_fields = (demo_key in r0 and event_key in r0)

        if has_split_fields:
            demographics = [r[demo_key] for r in records]
            events = [r[event_key] for r in records]
            input_ids, attn, seg = build_joint_sequences(
                demographics=demographics,
                events=events,
                max_len=max_len,
                pad_id=pad_id,
                sep_id=None,
            )
        else:
            # Flat sequences: token_ids
            input_ids = []
            attn = []
            seg = []
            for r in records:
                seq = list(r[token_key])
                seq = seq[:max_len]
                a = [1] * len(seq)

                while len(seq) < max_len:
                    seq.append(pad_id)
                    a.append(0)

                input_ids.append(seq)
                attn.append(a)

                # event_type_ids if present, else zeros
                if event_type_key is not None and event_type_key in r:
                    s = list(r[event_type_key])[:max_len]
                    while len(s) < max_len:
                        s.append(0)
                else:
                    s = [0] * max_len
                seg.append(s)

        self.input_ids = input_ids
        self.attention_mask = attn
        self.event_type_ids = seg

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "event_type_ids": torch.tensor(self.event_type_ids[idx], dtype=torch.long),
        }


@dataclass
class TopKResult:
    k: int
    accuracy: float
    n: int


@torch.no_grad()
def topk_accuracy_from_logits(
    logits: torch.Tensor,  # [B, L, V]
    labels: torch.Tensor,  # [B, L] with IGNORE_INDEX
    *,
    k: int,
    ignore_index: int = IGNORE_INDEX,
) -> TopKResult:
    if logits.ndim != 3:
        raise ValueError("logits must be [B, L, V]")
    if labels.ndim != 2:
        raise ValueError("labels must be [B, L]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/seq dims must match")
    if k <= 0:
        raise ValueError("k must be > 0")

    mask = labels != ignore_index
    n = int(mask.sum().item())
    if n == 0:
        return TopKResult(k=k, accuracy=float("nan"), n=0)

    # [N, V]
    flat_logits = logits[mask]
    flat_labels = labels[mask]

    topk = torch.topk(flat_logits, k=min(k, flat_logits.size(-1)), dim=-1).indices
    correct = (topk == flat_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return TopKResult(k=k, accuracy=float(correct), n=n)


def masking_policy_expected_corruption(p_mlm: float) -> Dict[str, float]:
    """Analytical expectation for BERT 80/10/10 masking.

    Returns expected fractions over *all valid tokens* (not including PAD).
    """
    p_pred = float(p_mlm)
    return {
        "p_pred": p_pred,
        "p_replace_mask": p_pred * 0.80,
        "p_replace_random": p_pred * 0.10,
        "p_keep_original": p_pred * 0.10,
        "p_unchanged_total": 1.0 - p_pred + (p_pred * 0.10),  # not selected + selected-but-kept
        "p_corrupted_total": p_pred * 0.90,  # [MASK] or random replacement
    }
