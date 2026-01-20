from extract_patient_level_events import PatientLevelEventExtractor
from sort_and_merge import SortMerger
from split_combined import DataSplitter
from vocabulary import Vocabulary
from build_patient_event_sequences import EventSequencer
from tokenize_sequences import TokenSequencer
from pathlib import Path
import pandas as pd
import json

def main():

    # 1) Extract patient-level events
    # extractor = PatientLevelEventExtractor()
    # extractor.start_extraction()

    # 2) Sort and merge events
    # sort_merger = SortMerger()
    # sort_merger.sort_and_merge()

    # 3) Train/val/test split
    # splitter = DataSplitter()
    # splitter.split_dataset()

    # 4) Build vocabulary
    # vocab = Vocabulary()
    # vocab.build_vocabulary()

    # 5) Build patient event sequences
    for COMBINED_CSV, OUT_IDS in [
        # ("../out/splits_out/combined_train.csv", "ids.json"),
        ("../out/splits_out/combined_val.csv", "val_ids.json"),
    ]:
        df = pd.read_csv(COMBINED_CSV)
        event_sequencer = EventSequencer()
        sequences = event_sequencer.build_patient_event_sequences(df)
        del df

        # 6) Tokenize sequences
        token_sequencer = TokenSequencer()
        ids = token_sequencer.build_sequences(sequences)

        # 7) Save tokenized sequences for training
        out_dir = Path("../out/sequences")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / OUT_IDS
        with open(out_path, "w") as f:
            json.dump(ids, f)

        print(f"Saved tokenized sequences to {out_path}")

if __name__ == "__main__":
    main()
