from __future__ import annotations

import os
from pathlib import Path
from typing import Set, Tuple, Optional

import numpy as np
import pandas as pd

class DataSplitter:
    def __init__(self):
        # ------------------ CONFIG -------------------
        self.SEED = 42
        self.TRAIN_RATIO = 0.70
        self.VAL_RATIO = 0.20  # test is remainder

        # Input / output
        self.COMBINED_PATH = Path("../out/merge_and_sort/combined.csv")        # or "data/combined.csv"
        self.OUT_DIR = "../out/splits_out"
        self.OUT_BASENAME = "combined"               # produces combined_train/val/test.*

        # Performance / memory
        self.CHUNKSIZE = 500_000                     # reduce to 200_000 if memory pressure
        self.OUTPUT_COMPRESSION: Optional[str] = None # "gzip" or None
        # -------------------------------------------

    def validate_ratios(self, train: float, val: float) -> None:
        if not (0 < train < 1):
            raise ValueError("TRAIN_RATIO must be in (0, 1)")
        if not (0 <= val < 1):
            raise ValueError("VAL_RATIO must be in [0, 1)")
        if train + val >= 1:
            raise ValueError("TRAIN_RATIO + VAL_RATIO must be < 1.0")

    def split_subject_ids(self, subject_ids: np.ndarray, seed: int) -> Tuple[Set[int], Set[int], Set[int]]:
        """Deterministic patient-level split of subject_ids."""
        self.validate_ratios(self.TRAIN_RATIO, self.VAL_RATIO)

        subject_ids = np.array(sorted(set(map(int, subject_ids))))
        rng = np.random.default_rng(seed)
        rng.shuffle(subject_ids)

        n = len(subject_ids)
        n_train = int(n * self.TRAIN_RATIO)
        n_val = int(n * self.VAL_RATIO)

        train_ids = set(subject_ids[:n_train])
        val_ids = set(subject_ids[n_train:n_train + n_val])
        test_ids = set(subject_ids[n_train + n_val:])

        return train_ids, val_ids, test_ids

    def save_ids(self, ids: Set[int], path: Path) -> None:
        pd.Series(sorted(ids), name="subject_id").to_csv(path, index=False)

    def discover_subject_ids(self, combined_path: str, chunksize: int) -> np.ndarray:
        """
        Pass 1: Stream-read only subject_id to collect unique patients.
        Works for very large CSVs.
        """
        seen: Set[int] = set()

        reader = pd.read_csv(
            combined_path,
            usecols=["subject_id"],
            chunksize=chunksize,
            low_memory=False,
        )
        for i, chunk in enumerate(reader, start=1):
            s = pd.to_numeric(chunk["subject_id"], errors="coerce")
            s = s.dropna().astype(int).unique()
            seen.update(s.tolist())

            if i % 20 == 0:
                print(f"[pass1] chunks={i:,} unique_patients={len(seen):,}")

        return np.array(sorted(seen))

    def chunked_write_splits(
        self,
        combined_path: str,
        out_train: Path,
        out_val: Path,
        out_test: Path,
        train_ids: Set[int],
        val_ids: Set[int],
        test_ids: Set[int],
        chunksize: int,
        compression: Optional[str],
    ) -> None:
        """
        Pass 2: Stream-read full rows and route to train/val/test files by subject_id.
        """
        # Remove stale outputs (avoid accidental append)
        for p in (out_train, out_val, out_test):
            if p.exists():
                p.unlink()

        header_written = {"train": False, "val": False, "test": False}

        reader = pd.read_csv(
            combined_path,
            chunksize=chunksize,
            low_memory=False,
        )

        for i, chunk in enumerate(reader, start=1):
            subj = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
            chunk = chunk[subj.notna()].copy()
            chunk["subject_id"] = subj[subj.notna()].astype(int)

            c_train = chunk[chunk["subject_id"].isin(train_ids)]
            c_val = chunk[chunk["subject_id"].isin(val_ids)]
            c_test = chunk[chunk["subject_id"].isin(test_ids)]

            if not c_train.empty:
                c_train.to_csv(out_train, mode="a", index=False,
                            header=not header_written["train"], compression=compression)
                header_written["train"] = True

            if not c_val.empty:
                c_val.to_csv(out_val, mode="a", index=False,
                            header=not header_written["val"], compression=compression)
                header_written["val"] = True

            if not c_test.empty:
                c_test.to_csv(out_test, mode="a", index=False,
                            header=not header_written["test"], compression=compression)
                header_written["test"] = True

            if i % 10 == 0:
                print(f"[pass2] processed chunks={i:,}")

        # Ensure files exist even if a split got zero rows (rare but possible)
        # Write header-only file if needed.
        if not header_written["train"]:
            pd.DataFrame().to_csv(out_train, index=False, compression=compression)
        if not header_written["val"]:
            pd.DataFrame().to_csv(out_val, index=False, compression=compression)
        if not header_written["test"]:
            pd.DataFrame().to_csv(out_test, index=False, compression=compression)


    def split_dataset(self) -> None:
        os.makedirs(self.OUT_DIR, exist_ok=True)

        if not os.path.exists(self.COMBINED_PATH):
            raise FileNotFoundError(f"combined file not found: {self.COMBINED_PATH}")

        print(f"Input: {self.COMBINED_PATH}")
        print(f"Output dir: {self.OUT_DIR}")
        print(f"Chunksize: {self.CHUNKSIZE:,} | Compression: {self.OUTPUT_COMPRESSION}")

        # Pass 1: collect subject_ids
        self.subject_ids = self.discover_subject_ids(self.COMBINED_PATH, self.CHUNKSIZE)
        print(f"Total unique patients discovered: {len(self.subject_ids):,}")

        # Split
        train_ids, val_ids, test_ids = self.split_subject_ids(self.subject_ids, seed=self.SEED)
        print(f"Patients per split: train={len(train_ids):,} val={len(val_ids):,} test={len(test_ids):,}")

        # Persist lists
        out_dir = Path(self.OUT_DIR)
        self.save_ids(train_ids, out_dir / "train_subject_ids.csv")
        self.save_ids(val_ids, out_dir / "val_subject_ids.csv")
        self.save_ids(test_ids, out_dir / "test_subject_ids.csv")

        # Pass 2: write split CSVs
        ext = "csv.gz" if self.OUTPUT_COMPRESSION == "gzip" else "csv"
        out_train = out_dir / f"{self.OUT_BASENAME}_train.{ext}"
        out_val = out_dir / f"{self.OUT_BASENAME}_val.{ext}"
        out_test = out_dir / f"{self.OUT_BASENAME}_test.{ext}"

        self.chunked_write_splits(
            combined_path=self.COMBINED_PATH,
            out_train=out_train,
            out_val=out_val,
            out_test=out_test,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            chunksize=self.CHUNKSIZE,
            compression=self.OUTPUT_COMPRESSION,
        )

        print("Done!")
        print(f"- {out_train}")
        print(f"- {out_val}")
        print(f"- {out_test}")
        print("Reminder: build vocabulary/tokenizer only on TRAIN split.")
