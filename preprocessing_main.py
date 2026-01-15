from extract_patient_level_events import PatientLevelEventExtractor
from sort_and_merge import SortMerger
from split_combined import DataSplitter
from vocabulary import Vocabulary
from build_patient_event_sequences import EventSequencer
from tokenize_sequences import TokenSequencer
from pathlib import Path
import pandas as pd

# 1 - Extract_patient_level_events - WORKS
# patientLevelEventExtractor = PatientLevelEventExtractor()
# patientLevelEventExtractor.start_extraction()

# 2 - Sort_and_merge - WORKS
# sortMerger = SortMerger()
# sortMerger.sort_and_merge()

# 3 - Split_combined - WORKS
# dataSplitter = DataSplitter()
# dataSplitter.split_dataset() 

# 4 - Vocabulary - WORKS
# vocabulary = Vocabulary()
# vocabulary.build_vocabulary()

# 5 - Build_patient_event_sequences - WORKS
COMBINED_CSV = Path("../out/splits_out/combined_train.csv")
df = pd.read_csv(COMBINED_CSV)

eventSequencer = EventSequencer()
sequences = eventSequencer.build_patient_event_sequences(df)

# 6 - tokenize_sequences - WORKS
tokenSequencer = TokenSequencer()
tokenSequencer.build_sequences(sequences)