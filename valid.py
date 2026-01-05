from __future__ import annotations

import os
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd


def validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1 (test is remainder)")


def split_subject_ids(
    subject_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Patient-level split. Returns (train_ids, val_ids, test_ids)."""
    validate_ratios(train_ratio, val_ratio)

    subject_ids = np.array(sorted(set(map(int, subject_ids))))
    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)

    n = len(subject_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(subject_ids[:n_train])
    val_ids = set(subject_ids[n_train:n_train + n_val])
    test_ids = set(subject_ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def filter_by_subject_ids(df: pd.DataFrame, ids: Set[int]) -> pd.DataFrame:
    """Keep only rows whose subject_id is in ids."""
    if "subject_id" not in df.columns:
        raise ValueError("DataFrame must contain 'subject_id'")
    subj = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    return df[subj.isin(list(ids))].copy()


def save_subject_ids(ids: Set[int], path: str) -> None:
    pd.Series(sorted(list(ids)), name="subject_id").to_csv(path, index=False)


def load_table(path: str, usecols: Optional[list]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def write_split_table(df: pd.DataFrame, out_path: str, compression: Optional[str]) -> None:
    df.to_csv(out_path, index=False, compression=compression)