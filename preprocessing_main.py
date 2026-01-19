from extract_patient_level_events import PatientLevelEventExtractor
from sort_and_merge import SortMerger
from split_combined import DataSplitter
from vocabulary import Vocabulary
from build_patient_event_sequences import EventSequencer
from tokenize_sequences import TokenSequencer
from pathlib import Path
from transformer.transformer_train_model import Transformer
import pandas as pd
from tqdm import tqdm
import os

# 1 - Extract_patient_level_events 
patientLevelEventExtractor = PatientLevelEventExtractor()
patientLevelEventExtractor.start_extraction()

# 2 - Sort_and_merge
sortMerger = SortMerger()
sortMerger.sort_and_merge()

# 3 - Split_combined 
dataSplitter = DataSplitter()
dataSplitter.split_dataset() 

# 4 - Vocabulary
vocabulary = Vocabulary()
vocabulary.build_vocabulary()

# 5 - Build_patient_event_sequences
COMBINED_CSV = Path("../out/splits_out/combined_train.csv")
df = pd.read_csv(COMBINED_CSV)

os.makedirs("../out/sequences", exist_ok=True)
eventSequencer = EventSequencer() 
sequences = eventSequencer.build_patient_event_sequences(df)
# write sequences to a file
with open("../out/sequences/sequences_train.txt", "w") as outfile: 
    for sequence in tqdm(sequences):
        dem, seq = sequence[0], sequence[1]
        outfile.write(",".join(dem) + "-" + ",".join(seq) + "\n")

# 6 - tokenize_sequences
sequences = []
# load sequences from the file
with open("../out/sequences/sequences_train.txt", "r") as infile:
    for line in tqdm(infile.readlines()):
        line = line.strip()
        splitpos = line.index("-")
        dem, seq = line[:splitpos], line[1+splitpos:]
        dem, seq = dem.split(","), seq.split(",")
        sequences.append([dem, seq])
# build id sequences
tokenSequencer = TokenSequencer()
ids = tokenSequencer.build_sequences(sequences)

# 7 - Train transformer
tokenSequencer = TokenSequencer()
ids = tokenSequencer.load_ids_from_json()
transformer = Transformer()
transformer.main(ids)