from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
                line = line.lstrip("#").lstrip("£").strip()
                if not line:
                    continue
            data.append(json.loads(line))
    return data


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
    Expected JSONL record keys:
        token_ids: List[int]
        event_type_ids: Optional[List[int]]  (model segment/event-type embedding ids)
        labels: Optional[List[int]]          (for MLM)
        attention_mask: Optional[List[int]]  (1=keep,0=pad)
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
        token_ids = list(r["token_ids"])
        event_type_ids = r.get("event_type_ids", None)
        labels = r.get("labels", None)

        token_ids = token_ids[: self.max_len]
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


@torch.no_grad()
def topk_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, k: int, ignore_index: int = IGNORE_INDEX) -> TopKResult:
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


# ---------------------------------------------------------
# Event-group evaluation based on Vocabulary block structure
# ---------------------------------------------------------

def build_token_id_to_group_from_vocab(vocab) -> Dict[int, int]:
    """
    token_id -> group_id mapping based on Vocabulary block membership.
    Group IDs are fixed small ints for evaluation.
    """
    name_to_gid = {
        "special": 0,
        "time": 1,
        "demographic_gender": 2,
        "demographic_age": 3,
        "admission": 4,
        "diagnosis": 5,
        "labevents": 6,
        "medication": 7,
        "readmission": 8,
        "death": 9,
    }

    token_id_to_group: Dict[int, int] = {}

    def add_block(block_name: str, token_to_id: Dict[str, int]):
        gid = name_to_gid[block_name]
        for tid in token_to_id.values():
            token_id_to_group[int(tid)] = int(gid)

    add_block("special", vocab.special_vocab)
    add_block("time", vocab.time_vocab)
    add_block("demographic_gender", vocab.dem_gen_vocab)
    add_block("demographic_age", vocab.dem_age_vocab)
    add_block("admission", vocab.admission_vocab)
    add_block("diagnosis", vocab.diagnosis_vocab)
    add_block("labevents", vocab.labevents_vocab)
    add_block("medication", vocab.medication_vocab)
    add_block("readmission", vocab.readmission_vocab)
    add_block("death", vocab.death_vocab)

    return token_id_to_group


@torch.no_grad()
def event_type_top1_acc_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_id_to_group: Dict[int, int],
    ignore_index: int = IGNORE_INDEX,
    default_gid: int = 0,
) -> float:
    pred = torch.argmax(logits, dim=-1)  # (B,L)

    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return float("nan")

    pred_ids = pred[mask].view(-1).tolist()
    true_ids = labels[mask].view(-1).tolist()

    ok = 0
    for p, y in zip(pred_ids, true_ids):
        gp = token_id_to_group.get(int(p), default_gid)
        gy = token_id_to_group.get(int(y), default_gid)
        ok += int(gp == gy)

    return ok / len(true_ids)


@torch.no_grad()
def event_type_topk_acc_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_id_to_group: Dict[int, int],
    k: int,
    ignore_index: int = IGNORE_INDEX,
    default_gid: int = 0,
) -> float:
    topk = torch.topk(logits, k=k, dim=-1).indices  # (B,L,k)

    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return float("nan")

    topk_list = topk[mask].view(-1, k).tolist()
    true_ids = labels[mask].view(-1).tolist()

    ok = 0
    for preds, y in zip(topk_list, true_ids):
        gy = token_id_to_group.get(int(y), default_gid)
        hit = any(token_id_to_group.get(int(p), default_gid) == gy for p in preds)
        ok += int(hit)

    return ok / len(true_ids)
