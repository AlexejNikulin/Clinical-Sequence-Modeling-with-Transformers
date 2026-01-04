import pandas as pd
from tqdm import tqdm
import os

ADMISSION_CSV = "../out/extract_patient_level_events/events_dynamic_admissions.csv"
DIAGNOSES_CSV = "../out/extract_patient_level_events/events_dynamic_diagnoses_icd.csv"
DISCHARGES_CSV = "../out/extract_patient_level_events/events_dynamic_discharges.csv"
MEDICATION_CSV = "../out/extract_patient_level_events/events_dynamic_emar.csv"
LABEVENTS_CSV = "../out/extract_patient_level_events/events_dynamic_labevents.csv"

PATIENT_CSV_PATH = "../out/merge_and_sort/patients/"
FINAL_CSV_PATH = "../out/merge_and_sort/combined.csv"

admissions_df = pd.read_csv(ADMISSION_CSV)
subject_ids = admissions_df["subject_id"].unique()

subject_ids_with_file = set()

def write_patient_events_for_file(file, subject_ids_with_file):

    print(f"Starting to process file: {file}")

    with pd.read_csv(file, chunksize=1_000_000) as reader:
        for i, chunk in enumerate(reader):

            local_subject_ids = chunk["subject_id"].unique()

            for subject_id in tqdm(local_subject_ids, desc=f"Chunk {i}"):
                subject_events = chunk[chunk["subject_id"] == subject_id]

                file_exists = subject_id in subject_ids_with_file
                write_mode = 'a' if file_exists else 'w' # write / append
                write_header = not file_exists

                subject_events.to_csv(
                    PATIENT_CSV_PATH + str(subject_id) + ".csv", 
                    mode=write_mode, 
                    header=write_header, 
                    index=False
                )

                subject_ids_with_file.add(subject_id)

# Create one file for each patient
os.makedirs(PATIENT_CSV_PATH)
write_patient_events_for_file(ADMISSION_CSV, subject_ids_with_file)
write_patient_events_for_file(DIAGNOSES_CSV, subject_ids_with_file)
write_patient_events_for_file(DISCHARGES_CSV, subject_ids_with_file)
write_patient_events_for_file(MEDICATION_CSV, subject_ids_with_file)
write_patient_events_for_file(LABEVENTS_CSV, subject_ids_with_file)

# For each patient, sort the events, and append them to the global dataframe
patient_files = os.listdir(PATIENT_CSV_PATH)
idx = 0
for patient_file in tqdm(patient_files, desc="Merging patients"):
    patient_id = int(patient_file.split(".")[0])

    patient_df = pd.read_csv(PATIENT_CSV_PATH + patient_file).sort_values(["timestamp"])
    
    write_mode = 'w' if idx == 0 else 'a' # write / append
    write_header = (idx == 0)

    patient_df.to_csv(
        FINAL_CSV_PATH, 
        mode=write_mode, 
        header=write_header, 
        index=False
    )

    # os.remove(PATIENT_CSV_PATH + patient_file)

    idx += 1

# What's still missing
# - death
# - readmission
# - procedure