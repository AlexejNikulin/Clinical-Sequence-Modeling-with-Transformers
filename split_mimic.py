# as main.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from splitConfig import SplitConfig, PATHS, USECOLS, OUTPUT_COMPRESSION
from valid import (
    split_subject_ids,
    filter_by_subject_ids,
    save_subject_ids,
    load_table,
    write_split_table,
)


def main(cfg: SplitConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load admissions first
    print(f"Loading admissions from: {PATHS['admissions']}")
    admissions = load_table(PATHS["admissions"], USECOLS.get("admissions"))
    if "subject_id" not in admissions.columns:
        raise ValueError("admissions must contain subject_id")

    # Choose patient universe
    if cfg.universe_from_admissions_only:
        universe_ids = admissions["subject_id"].dropna().astype(int).unique()
        print("Patient universe: admissions-only")
    else:
        print("Patient universe: union across all tables")
        universe_ids = set(admissions["subject_id"].dropna().astype(int).unique())
        for name, path in PATHS.items():
            if name == "admissions":
                continue
            print(f"Loading {name} from: {path}")
            df_tmp = load_table(path, USECOLS.get(name))
            if "subject_id" not in df_tmp.columns:
                raise ValueError(f"{name} must contain subject_id")
            universe_ids.update(df_tmp["subject_id"].dropna().astype(int).unique())
        universe_ids = np.array(sorted(universe_ids))

    print(f"Total unique patients in universe: {len(universe_ids):,}")

    # Split patients
    train_ids, val_ids, test_ids = split_subject_ids(
        subject_ids=np.array(universe_ids),
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    print(
        f"Split sizes (patients): "
        f"train={len(train_ids):,} val={len(val_ids):,} test={len(test_ids):,}"
    )

    # Save subject_id lists
    save_subject_ids(train_ids, os.path.join(cfg.out_dir, "train_subject_ids.csv"))
    save_subject_ids(val_ids, os.path.join(cfg.out_dir, "val_subject_ids.csv"))
    save_subject_ids(test_ids, os.path.join(cfg.out_dir, "test_subject_ids.csv"))

    # Split each table
    for name, path in PATHS.items():
        print(f"\nProcessing table: {name}")
        df = admissions if name == "admissions" else load_table(path, USECOLS.get(name))

        df_train = filter_by_subject_ids(df, train_ids)
        df_val = filter_by_subject_ids(df, val_ids)
        df_test = filter_by_subject_ids(df, test_ids)

        ext = "csv.gz" if OUTPUT_COMPRESSION == "gzip" else "csv"
        out_train = os.path.join(cfg.out_dir, f"{name}_train.{ext}")
        out_val = os.path.join(cfg.out_dir, f"{name}_val.{ext}")
        out_test = os.path.join(cfg.out_dir, f"{name}_test.{ext}")

        write_split_table(df_train, out_train, OUTPUT_COMPRESSION)
        write_split_table(df_val, out_val, OUTPUT_COMPRESSION)
        write_split_table(df_test, out_test, OUTPUT_COMPRESSION)

        print(
            f"{name} rows: train={len(df_train):,} val={len(df_val):,} test={len(df_test):,}"
        )

        # Leakage check (patient overlap)
        tr = set(pd.to_numeric(df_train["subject_id"], errors="coerce").dropna().astype(int).unique())
        va = set(pd.to_numeric(df_val["subject_id"], errors="coerce").dropna().astype(int).unique())
        te = set(pd.to_numeric(df_test["subject_id"], errors="coerce").dropna().astype(int).unique())
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), "Leakage detected!"


    print("Reminder: build vocab/tokenizer ONLY on training split.")
    print(f"\n  Done! Output written to: {cfg.out_dir}")

if __name__ == "__main__":
    config = SplitConfig(
        seed=42,
        train_ratio=0.80,
        val_ratio=0.10,
        out_dir="splits_out",
        universe_from_admissions_only=True,
    )
    main(config)
