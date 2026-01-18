from extract_patient_level_events import PatientLevelEventExtractor
from sort_and_merge import SortMerger
from split_combined import DataSplitter
from vocabulary import Vocabulary
from build_patient_event_sequences import EventSequencer
from tokenize_sequences import TokenSequencer
from pathlib import Path
from transformer.transformer_train_model import Transformer
import pandas as pd

# # 1 - Extract_patient_level_events 
# patientLevelEventExtractor = PatientLevelEventExtractor()
# patientLevelEventExtractor.start_extraction()

# # 2 - Sort_and_merge
# sortMerger = SortMerger()
# sortMerger.sort_and_merge()

# # 3 - Split_combined 
# dataSplitter = DataSplitter()
# dataSplitter.split_dataset() 

# # 4 - Vocabulary
# vocabulary = Vocabulary()
# vocabulary.build_vocabulary()

# # 5 - Build_patient_event_sequences
# COMBINED_CSV = Path("../out/splits_out/combined_train.csv")
# df = pd.read_csv(COMBINED_CSV)

# eventSequencer = EventSequencer() 
# sequences = eventSequencer.build_patient_event_sequences(df)

# # 6 - tokenize_sequences
# sequences = [
#     ["[DEM_GENDER_F]", "[DEM_AGE_19]", "[ADM_EU OBSERVATION]", "[LAB_CREATININE]", "[MED_ATORVASTATIN]", "[READM_EU OBSERVATION]"],
#     ["[DEM_GENDER_M]", "[DEM_AGE_59]", "[ADM_OBSERVATION ADMIT]", "[DIAG9_5715]", "[DEATH]"]
# ]
# tokenSequencer = TokenSequencer()
# ids = tokenSequencer.build_sequences(sequences)

# # 7 - Train transformer
# transformer = Transformer()
# transformer.main(ids)