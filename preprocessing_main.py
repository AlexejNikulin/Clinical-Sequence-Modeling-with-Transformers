from extract_patient_level_events import PatientLevelEventExtractor
from extract_patient_level_events_short import PatientLevelEventExtractor_Short
from sort_and_merge import SortMerger
from split_combined import DataSplitter
from vocabulary import Vocabulary
from build_patient_event_sequences import EventSequencer
from tokenize_sequences import TokenSequencer

from pathlib import Path
import pandas as pd
import os


def main():
    print("START!")

    # ------------------------------------------------------------------
    # Paths (based on what your other scripts expect)
    # ------------------------------------------------------------------
    OUT_BASE = Path("../out")
    MERGE_DIR = OUT_BASE / "merge_and_sort"
    COMBINED_CSV = MERGE_DIR / "combined.csv"

    SPLIT_DIR = OUT_BASE / "splits_out"
    TRAIN_CSV = SPLIT_DIR / "combined_train.csv"
    VAL_CSV = SPLIT_DIR / "combined_val.csv"
    TEST_CSV = SPLIT_DIR / "combined_test.csv"

    FINAL_OUT = OUT_BASE / "out"
    VOCAB_PATH = FINAL_OUT / "vocab" / "vocabulary.json"
    SEQ_DIR = FINAL_OUT / "sequences"

    # ------------------------------------------------------------------
    # (1) Extract patient-level events (run if merged file missing)
    # ------------------------------------------------------------------
    # If you want the short debug version, set USE_SHORT_EXTRACT=True.
    USE_SHORT_EXTRACT = False

    if not COMBINED_CSV.exists():
        print("[1] combined.csv missing -> ensure extracted events exist (running extractor)...")
        if USE_SHORT_EXTRACT:
            extractor = PatientLevelEventExtractor_Short()
        else:
            extractor = PatientLevelEventExtractor()
        extractor.start_extraction()
    else:
        print("[1] combined.csv exists -> skipping extraction")

    # ------------------------------------------------------------------
    # (2) Sort & merge -> ensures ../out/merge_and_sort/combined.csv exists
    # ------------------------------------------------------------------
    if not COMBINED_CSV.exists():
        print(f"[2] running SortMerger to create: {COMBINED_CSV}")
        sort_merger = SortMerger()
        sort_merger.sort_and_merge()
    else:
        print(f"[2] combined.csv exists -> skipping SortMerger: {COMBINED_CSV}")

    if not COMBINED_CSV.exists():
        raise FileNotFoundError(
            f"Expected combined.csv after SortMerger, but still missing: {COMBINED_CSV.resolve()}"
        )

    # ------------------------------------------------------------------
    # (3) Split -> ensures train/val/test csv exist
    # ------------------------------------------------------------------
    if not (TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists()):
        print(f"[3] split CSVs missing -> running DataSplitter into: {SPLIT_DIR}")
        splitter = DataSplitter()
        splitter.split_dataset()
    else:
        print("[3] split CSVs exist -> skipping DataSplitter")

    # ------------------------------------------------------------------
    # (4) Vocabulary: load if exists, else build (build uses combined_train.csv)
    # ------------------------------------------------------------------
    if VOCAB_PATH.exists() and hasattr(Vocabulary, "load"):
        print(f"[4] Loading vocabulary: {VOCAB_PATH}")
        vocab = Vocabulary.load(VOCAB_PATH)
    else:
        print(f"[4] Building vocabulary (expects train split): {TRAIN_CSV}")
        vocab = Vocabulary()
        vocab.build_vocabulary()

    # ------------------------------------------------------------------
    # (5+6) Build sequences + tokenize -> ids.json / val_ids.json / test_ids.json
    # ------------------------------------------------------------------
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [
        (TRAIN_CSV, "ids.json"),
        (VAL_CSV, "val_ids.json"),
        (TEST_CSV, "test_ids.json"),
    ]

    for combined_path, out_name in jobs:
        if not combined_path.exists():
            raise FileNotFoundError(f"Missing required split file: {combined_path.resolve()}")

        out_path = SEQ_DIR / out_name
        print(f"[5+6] {combined_path} -> {out_path}")

        df = pd.read_csv(combined_path)
        event_sequencer = EventSequencer()
        sequences = event_sequencer.build_patient_event_sequences(df, vocab)
        del df

        token_sequencer = TokenSequencer()
        token_sequencer.build_sequences(sequences, False, out_path)

    print("DONE!")


if __name__ == "__main__":
    main()
