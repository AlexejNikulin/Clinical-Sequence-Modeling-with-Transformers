# evaluation/clinical_eval_utils.py
# Block/type evaluation utilities + Dataset with optional random window sampling.
# evaluation/clinical_eval_utils.py
# Utilities for evaluation: loading JSONL, dataset, token-level metrics,
# and "event-type/block" metrics derived from Vocabulary.*_vocab ranges.

# evaluation/clinical_eval_utils.py
# ------------------------------------------------------------
# Utilities for evaluation:
# - load_jsonl
# - ClinicalSequenceDataset (optional deterministic random-window sampling)
# - token-level metrics (top-k, MRR)
# - block/event-type mapping from Vocabulary.*_vocab dicts
# - block metrics: top1/topk by block id (event-type accuracy)
#
# Notes:
# - IGNORE_INDEX = -100 matches PyTorch CrossEntropyLoss ignore_index default usage.
# - "block" here means: token belongs to a vocabulary subspace (diagnosis, lab, med, ...),
#   not the exact token itself.
# ------------------------------------------------------------

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


# ----------------------------
# IO
# ----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL records.
    Allows comment lines starting with "#". If a comment line contains JSON after "#£",
    it will still be parsed (handy for debug dumps).
    """
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # Allow comment lines like "# ..." or "#£{...}"
                line = line.lstrip("#").lstrip("£").strip()
                if not line:
                    continue

            data.append(json.loads(line))
    return data


# ----------------------------
# Dataset
# ----------------------------
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
    Expected JSONL keys (any of these for token ids):
        token_ids / ids / input_ids / tokens : List[int]

    Optional keys:
        event_type_ids : Optional[List[int]]   (segment/event-type embedding ids)
        labels         : Optional[List[int]]   (MLM labels; IGNORE_INDEX for non-eval positions)
        attention_mask : Optional[List[int]]   (1=valid token, 0=pad)

    Optional window sampling (useful if sequences are longer than max_len):
        sample_windows : bool
        keep_prefix_n  : int   keep first N tokens fixed (e.g., demographics)
        seed           : int   deterministic random window per (seed + idx)

    IMPORTANT:
        DataLoader should typically use shuffle=False for evaluation.
        Window sampling still changes selected subsequence unless you keep seed fixed.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        max_len: int,
        pad_id: int,
        default_event_type_id: int = 0,
        *,
        sample_windows: bool = False,
        keep_prefix_n: int = 0,
        seed: int = 0,
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
        token_ids = r.get("token_ids")
        if token_ids is None:
            token_ids = r.get("ids")
        if token_ids is None:
            token_ids = r.get("input_ids")
        if token_ids is None:
            token_ids = r.get("tokens")
        if token_ids is None:
            raise KeyError(
                f"Record missing token sequence. Expected one of token_ids/ids/input_ids/tokens. Keys={list(r.keys())}"
            )
        return list(token_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]

        token_ids_full = self._get_token_list(r)

        ev_full = r.get("event_type_ids")
        lbl_full = r.get("labels")
        attn_full = r.get("attention_mask")

        ev_full = list(ev_full) if ev_full is not None else None
        lbl_full = list(lbl_full) if lbl_full is not None else None
        attn_full = list(attn_full) if attn_full is not None else None

        # ---------------------------------------------------------
        # Optional deterministic random window sampling
        # ---------------------------------------------------------
        if self.sample_windows:
            kp = max(0, self.keep_prefix_n)
            L = self.max_len

            prefix_toks = token_ids_full[:kp]
            rest_toks = token_ids_full[kp:]

            prefix_ev = ev_full[:kp] if ev_full is not None else None
            rest_ev = ev_full[kp:] if ev_full is not None else None

            prefix_lbl = lbl_full[:kp] if lbl_full is not None else None
            rest_lbl = lbl_full[kp:] if lbl_full is not None else None

            prefix_attn = attn_full[:kp] if attn_full is not None else None
            rest_attn = attn_full[kp:] if attn_full is not None else None

            # how many tokens can we take from "rest"?
            rest_cap = max(0, L - len(prefix_toks))

            # choose window start in rest part
            if rest_cap > 0 and len(rest_toks) > rest_cap:
                max_start = len(rest_toks) - rest_cap
                rr = random.Random(self.seed + idx)  # deterministic per idx given seed
                start = rr.randint(0, max_start)
            else:
                start = 0

            sampled_rest = rest_toks[start : start + rest_cap]
            sampled_rest_ev = rest_ev[start : start + rest_cap] if rest_ev is not None else None
            sampled_rest_lbl = rest_lbl[start : start + rest_cap] if rest_lbl is not None else None
            sampled_rest_attn = rest_attn[start : start + rest_cap] if rest_attn is not None else None

            token_ids = prefix_toks + sampled_rest

            # event types: fill missing with default_event_type_id
            if prefix_ev is not None or sampled_rest_ev is not None:
                ev_ids = (prefix_ev if prefix_ev is not None else [self.default_event_type_id] * len(prefix_toks)) + (
                    sampled_rest_ev if sampled_rest_ev is not None else [self.default_event_type_id] * len(sampled_rest)
                )
            else:
                ev_ids = [self.default_event_type_id] * len(token_ids)

            # labels: fill missing with IGNORE_INDEX
            if prefix_lbl is not None or sampled_rest_lbl is not None:
                labels = (prefix_lbl if prefix_lbl is not None else [IGNORE_INDEX] * len(prefix_toks)) + (
                    sampled_rest_lbl if sampled_rest_lbl is not None else [IGNORE_INDEX] * len(sampled_rest)
                )
            else:
                labels = [IGNORE_INDEX] * len(token_ids)

            # attention mask: fill missing with 1, padding later sets 0
            if prefix_attn is not None or sampled_rest_attn is not None:
                attn = (prefix_attn if prefix_attn is not None else [1] * len(prefix_toks)) + (
                    sampled_rest_attn if sampled_rest_attn is not None else [1] * len(sampled_rest)
                )
            else:
                attn = [1] * len(token_ids)

        else:
            # ---------------------------------------------------------
            # Default behavior: truncate from start
            # ---------------------------------------------------------
            token_ids = token_ids_full[: self.max_len]

            if ev_full is not None:
                ev_ids = ev_full[: self.max_len]
                if len(ev_ids) < len(token_ids):
                    ev_ids += [self.default_event_type_id] * (len(token_ids) - len(ev_ids))
            else:
                ev_ids = [self.default_event_type_id] * len(token_ids)

            if lbl_full is not None:
                labels = lbl_full[: self.max_len]
                if len(labels) < len(token_ids):
                    labels += [IGNORE_INDEX] * (len(token_ids) - len(labels))
            else:
                labels = [IGNORE_INDEX] * len(token_ids)

            if attn_full is not None:
                attn = attn_full[: self.max_len]
                if len(attn) < len(token_ids):
                    attn += [1] * (len(token_ids) - len(attn))
            else:
                attn = [1] * len(token_ids)

        # ----------------------------
        # Right-pad to max_len
        # ----------------------------
        token_ids = token_ids[: self.max_len]
        ev_ids = ev_ids[: self.max_len]
        labels = labels[: self.max_len]
        attn = attn[: self.max_len]

        pad_n = self.max_len - len(token_ids)
        token_ids += [self.pad_id] * pad_n
        ev_ids += [self.default_event_type_id] * pad_n
        labels += [IGNORE_INDEX] * pad_n
        attn += [0] * pad_n

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "event_type_ids": torch.tensor(ev_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ----------------------------
# Token-level metrics (optional / debugging)
# ----------------------------
@torch.no_grad()
def topk_accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    ignore_index: int = IGNORE_INDEX,
) -> TopKResult:
    """
    Standard token-level top-k accuracy on positions where labels != IGNORE_INDEX.
    Useful for debugging but your main metric is block/event-type accuracy.
    """
    _, _, V = logits.shape
    lbl = labels.view(-1)
    log = logits.view(-1, V)

    mask = lbl != ignore_index
    if mask.sum().item() == 0:
        return TopKResult(k=k, correct=0, total=0)

    lbl_m = lbl[mask]
    log_m = log[mask]

    topk = torch.topk(log_m, k=k, dim=-1).indices  # (N,k)
    correct = (topk == lbl_m.unsqueeze(-1)).any(dim=-1).sum().item()
    total = lbl_m.numel()
    return TopKResult(k=k, correct=int(correct), total=int(total))


@torch.no_grad()
def mrr_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> float:
    """Token-level mean reciprocal rank (MRR)."""
    _, _, V = logits.shape
    lbl = labels.view(-1)
    log = logits.view(-1, V)

    mask = lbl != ignore_index
    if mask.sum().item() == 0:
        return float("nan")

    lbl_m = lbl[mask]
    log_m = log[mask]

    sorted_idx = torch.argsort(log_m, dim=-1, descending=True)  # (N,V)
    ranks = (sorted_idx == lbl_m.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1]
    return float((1.0 / (ranks.float() + 1.0)).mean().item())


# ----------------------------
# Block/event-type mapping from Vocabulary
# ----------------------------
def build_token_id_to_block_id_from_vocab(vocab) -> Tuple[Dict[int, int], Dict[int, str], Dict[str, int]]:
    """
    Build:
      - token_id_to_block_id: token_id -> block_id (event-type group)
      - block_id_to_name: block_id -> readable group name
      - block_name_to_id: group name -> block_id

    Uses vocab.<something>_vocab dicts (token_str -> token_id). Missing dicts are treated as empty.
    """
    block_id_to_name: Dict[int, str] = {
        0: "special",
        1: "time",
        2: "demographic_gender",
        3: "demographic_age",
        4: "demographic_race",
        5: "admission",
        6: "diagnosis",
        7: "labevents",
        8: "medication",
        9: "omr_bmi",
        10: "omr_weight",
        11: "omr_blood_pres",
        12: "discharge",
        13: "death",
        14: "readmission",  # optional legacy
    }
    block_name_to_id: Dict[str, int] = {v: k for k, v in block_id_to_name.items()}

    token_id_to_block: Dict[int, int] = {}

    def add_block(block_name: str, token_to_id: Dict[str, int]) -> None:
        bid = int(block_name_to_id[block_name])
        for tid in token_to_id.values():
            token_id_to_block[int(tid)] = bid

    # Known blocks in your Vocabulary
    add_block("special", getattr(vocab, "special_vocab", {}))
    add_block("time", getattr(vocab, "time_vocab", {}))
    add_block("demographic_gender", getattr(vocab, "dem_gen_vocab", {}))
    add_block("demographic_age", getattr(vocab, "dem_age_vocab", {}))
    add_block("demographic_race", getattr(vocab, "dem_race_vocab", {}))

    add_block("admission", getattr(vocab, "admission_vocab", {}))
    add_block("diagnosis", getattr(vocab, "diagnosis_vocab", {}))
    add_block("labevents", getattr(vocab, "labevents_vocab", {}))
    add_block("medication", getattr(vocab, "medication_vocab", {}))

    add_block("omr_bmi", getattr(vocab, "omr_bmi_vocab", {}))
    add_block("omr_weight", getattr(vocab, "omr_weight_vocab", {}))
    add_block("omr_blood_pres", getattr(vocab, "omr_blood_pres_vocab", {}))

    add_block("discharge", getattr(vocab, "discharge_vocab", {}))
    add_block("death", getattr(vocab, "death_vocab", {}))

    # Optional legacy
    add_block("readmission", getattr(vocab, "readmission_vocab", {}))

    return token_id_to_block, block_id_to_name, block_name_to_id


# Backward-compatible alias (some older files call it "group")
def build_token_id_to_group_from_vocab(vocab) -> Dict[int, int]:
    """
    Backward-compatible helper:
    returns only token_id -> block_id mapping.
    """
    token_id_to_block, _, _ = build_token_id_to_block_id_from_vocab(vocab)
    return token_id_to_block


# ----------------------------
# Block metrics
# ----------------------------
@torch.no_grad()
def block_top1_acc_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_id_to_block: Dict[int, int],
    ignore_index: int = IGNORE_INDEX,
    default_block_id: int = 0,
) -> float:
    """
    Block top-1 accuracy:
      correct if block(argmax_token) == block(true_token) at evaluated positions.
    """
    pred = torch.argmax(logits, dim=-1)  # (B,L)

    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return float("nan")

    pred_ids = pred[mask].view(-1).tolist()
    true_ids = labels[mask].view(-1).tolist()

    ok = 0
    for p, y in zip(pred_ids, true_ids):
        ok += int(token_id_to_block.get(int(p), default_block_id) == token_id_to_block.get(int(y), default_block_id))

    return ok / max(1, len(true_ids))


@torch.no_grad()
def block_topk_acc_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_id_to_block: Dict[int, int],
    k: int,
    ignore_index: int = IGNORE_INDEX,
    default_block_id: int = 0,
) -> float:
    """
    Block top-k accuracy:
      hit if ANY token among top-k has the same block as the true token.
    """
    topk = torch.topk(logits, k=k, dim=-1).indices  # (B,L,k)

    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return float("nan")

    topk_list = topk[mask].view(-1, k).tolist()
    true_ids = labels[mask].view(-1).tolist()

    ok = 0
    for preds, y in zip(topk_list, true_ids):
        by = token_id_to_block.get(int(y), default_block_id)
        hit = any(token_id_to_block.get(int(p), default_block_id) == by for p in preds)
        ok += int(hit)

    return ok / max(1, len(true_ids))