# evaluation/clinical_eval_utils.py -----> use for event type 
# Build token_id → block_id from Vocabulary.START_* boundaries (range-based, no token strings).
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100



# IO

def load_jsonl(path: str) -> List[Dict[str, Any]]:
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



# Dataset

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
    Expected JSONL record keys (at least token_ids/ids):
        token_ids: List[int]
        ids: List[int]              (alternative format)
        event_type_ids: Optional[List[int]]   (segment/event-type embedding ids)
        labels: Optional[List[int]]           (for MLM)
        attention_mask: Optional[List[int]]   (1=keep,0=pad)
    """

    def __init__(self, records: List[Dict[str, Any]], max_len: int, pad_id: int, default_event_type_id: int = 0):
        self.records = records
        self.max_len = max_len
        self.pad_id = pad_id
        self.default_event_type_id = default_event_type_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]

        token_ids = r.get("token_ids", None)
        if token_ids is None:
            token_ids = r.get("ids", None)
        if token_ids is None:
            token_ids = r.get("input_ids", None)
        if token_ids is None:
            token_ids = r.get("tokens", None)

        if token_ids is None:
            raise KeyError(
                f"Record missing token sequence. Expected one of token_ids/ids/input_ids/tokens. Keys={list(r.keys())}"
            )

        token_ids = list(token_ids)[: self.max_len]

        event_type_ids = r.get("event_type_ids", None)
        labels = r.get("labels", None)

        if event_type_ids is not None:
            event_type_ids = list(event_type_ids)[: self.max_len]
            if len(event_type_ids) < len(token_ids):
                event_type_ids += [self.default_event_type_id] * (len(token_ids) - len(event_type_ids))
        else:
            event_type_ids = [self.default_event_type_id] * len(token_ids)

        if labels is not None:
            labels = list(labels)[: self.max_len]
            if len(labels) < len(token_ids):
                labels += [IGNORE_INDEX] * (len(token_ids) - len(labels))
        else:
            labels = [IGNORE_INDEX] * len(token_ids)

        attn = r.get("attention_mask", None)
        if attn is not None:
            attn = list(attn)[: self.max_len]
            if len(attn) < len(token_ids):
                attn += [1] * (len(token_ids) - len(attn))
        else:
            attn = [1] * len(token_ids)

        # pad right
        pad_n = self.max_len - len(token_ids)
        token_ids += [self.pad_id] * pad_n
        event_type_ids += [self.default_event_type_id] * pad_n
        labels += [IGNORE_INDEX] * pad_n
        attn += [0] * pad_n

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "event_type_ids": torch.tensor(event_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }



# Token-level metrics (kept for debugging)

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

    lbl_m = lbl[mask]
    log_m = log[mask]

    topk = torch.topk(log_m, k=k, dim=-1).indices  # (N,k)
    correct = (topk == lbl_m.unsqueeze(-1)).any(dim=-1).sum().item()
    total = lbl_m.numel()
    return TopKResult(k=k, correct=int(correct), total=int(total))


@torch.no_grad()
def mrr_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> float:
    B, L, V = logits.shape
    lbl = labels.view(-1)
    log = logits.view(-1, V)

    mask = lbl != ignore_index
    if mask.sum().item() == 0:
        return float("nan")

    lbl_m = lbl[mask]
    log_m = log[mask]

    sorted_idx = torch.argsort(log_m, dim=-1, descending=True)  # (N,V)
    ranks = (sorted_idx == lbl_m.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1]
    rr = (1.0 / (ranks.float() + 1.0)).mean().item()
    return float(rr)


# TYPE/BLOCK mapping based on vocabulary.py START_* (NO changes to vocabulary.py)


def _vocab_boundaries_from_vocab(vocab) -> List[Tuple[int, str]]:
    """
    Return sorted (start_id, block_name) boundaries using START_* values from Vocabulary.
    """
    # These must exist in your Vocabulary class (they do in your paste).
    boundaries = [
        (int(getattr(vocab, "START_SPECIAL")), "special"),
        (int(getattr(vocab, "START_TIME")), "time"),
        (int(getattr(vocab, "START_DEM_GEN")), "demographic_gender"),
        (int(getattr(vocab, "START_DEM_AGE")), "demographic_age"),
        (int(getattr(vocab, "START_DEM_RACE")), "demographic_race"),
        (int(getattr(vocab, "START_ADM")), "admission"),
        (int(getattr(vocab, "START_DIAG")), "diagnosis"),
        (int(getattr(vocab, "START_LABEV")), "labevents"),
        (int(getattr(vocab, "START_MED")), "medication"),
        (int(getattr(vocab, "START_OMR_BMI")), "omr_bmi"),
        (int(getattr(vocab, "START_OMR_WEIGHT")), "omr_weight"),
        (int(getattr(vocab, "START_OMR_BLOOD_PRES")), "omr_blood_pres"),
        (int(getattr(vocab, "START_DISCH")), "discharge"),
        (int(getattr(vocab, "START_DEATH")), "death"),
    ]
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def build_token_id_to_group_from_vocab(vocab) -> Dict[int, int]:
    """
    token_id -> group_id mapping based on available Vocabulary blocks.

    Robust to schema differences: uses getattr(..., {}) so missing blocks
    (e.g., readmission_vocab) won't crash evaluation.
    """
    name_to_gid = {
        "special": 0,
        "time": 1,
        "demographic_gender": 2,
        "demographic_age": 3,
        "demographic_race": 4,
        "admission": 5,
        "diagnosis": 6,
        "labevents": 7,
        "medication": 8,
        "omr_bmi": 9,
        "omr_weight": 10,
        "omr_blood_pres": 11,
        "discharge": 12,
        "death": 13,
        # optional legacy name:
        "readmission": 14,
    }

    token_id_to_group: Dict[int, int] = {}

    def add_block(block_name: str, token_to_id: Dict[str, int]):
        gid = name_to_gid[block_name]
        for tid in token_to_id.values():
            token_id_to_group[int(tid)] = int(gid)

    # --- blocks that exist in your Vocabulary ---
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

    # --- optional legacy block (won't crash if missing) ---
    add_block("readmission", getattr(vocab, "readmission_vocab", {}))

    return token_id_to_group


# TYPE/BLOCK metrics from logits (THIS is what we want)


@torch.no_grad()
def block_top1_acc_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_id_to_block: Dict[int, int],
    ignore_index: int = IGNORE_INDEX,
    default_block_id: int = 0,
) -> float:
    """
    Accuracy where prediction is correct if:
      block(pred_token) == block(true_token)
    """
    pred = torch.argmax(logits, dim=-1)  # (B,L)

    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return float("nan")

    pred_ids = pred[mask].view(-1).tolist()
    true_ids = labels[mask].view(-1).tolist()

    ok = 0
    for p, y in zip(pred_ids, true_ids):
        bp = token_id_to_block.get(int(p), default_block_id)
        by = token_id_to_block.get(int(y), default_block_id)
        ok += int(bp == by)

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
    Top-k accuracy where hit if ANY predicted token in top-k has same block as true token.
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
