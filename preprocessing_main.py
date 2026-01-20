from extract_patient_level_events import PatientLevelEventExtractor
from extract_patient_level_events_short import PatientLevelEventExtractor_Short
from sort_and_merge import SortMerger
from split_combined import DataSplitter
from vocabulary import Vocabulary
from build_patient_event_sequences import EventSequencer
from tokenize_sequences import TokenSequencer

from pathlib import Path
import pandas as pd
import json

def main():

    print("START!")

    # # 1) Extract patient-level events
    # # extractor = PatientLevelEventExtractor()
    # # extractor.start_extraction()

    # Short version for debugging
    extractor = PatientLevelEventExtractor_Short()
    extractor.start_extraction()

    # 2) Sort and merge events
    sort_merger = SortMerger()
    sort_merger.sort_and_merge()

    # 2.1) Add time tokens
    COMBINED_CSV = Path("../out/merge_and_sort/combined.csv")
    df = pd.read_csv(COMBINED_CSV)

    event_sequencer = EventSequencer()
    sequences = event_sequencer.add_time_tokens_to_data(df)

    # 3) Train/val/test split
    splitter = DataSplitter()
    splitter.split_dataset()

    # 4) Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocabulary()

    # 5) Build patient event sequences
    for COMBINED_CSV, OUT_IDS in [
        ("../out/splits_out/combined_train.csv", "ids.json"),
        ("../out/splits_out/combined_val.csv", "val_ids.json"),
    ]:
        df = pd.read_csv(COMBINED_CSV)
        event_sequencer = EventSequencer()
        sequences = event_sequencer.build_patient_event_sequences(df, vocab)
        del df

        # 6) Tokenize sequences
        token_sequencer = TokenSequencer()
        out_dir = Path("../out/sequences")
        out_dir.mkdir(parents=True, exist_ok=True)
        token_sequencer.build_sequences(sequences, False, out_dir / OUT_IDS)

if __name__ == "__main__":
    main()
