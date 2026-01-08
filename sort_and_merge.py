import pandas as pd
from tqdm import tqdm
import os


ADMISSION_CSV = "../out/extract_patient_level_events/events_dynamic_admissions.csv"
DIAGNOSES_CSV = "../out/extract_patient_level_events/events_dynamic_diagnoses_icd.csv"
DISCHARGES_CSV = "../out/extract_patient_level_events/events_dynamic_discharges.csv"
MEDICATION_CSV = "../out/extract_patient_level_events/events_dynamic_emar.csv"
LABEVENTS_CSV = "../out/extract_patient_level_events/events_dynamic_labevents.csv"

PATIENT_CSV_PATH = "../out/merge_and_sort/patients/"
PATIENTS_CSV = "../physionet.org/files/mimiciv/3.1/hosp/patients.csv"  
patients_df = pd.read_csv(PATIENTS_CSV, usecols=["subject_id","gender","anchor_age","anchor_year_group","dod"])
patients_df = patients_df.set_index("subject_id")
FINAL_CSV_PATH = "../out/merge_and_sort/combined.csv"

def yeargrp_tok(x):
    if pd.isna(x): 
        return "DEM_YEARGRP_UNK"
    s = str(x).strip().replace(" ", "").replace("-", "_")
    return f"DEM_YEARGRP_{s}"

def gender_tok(x):
    if pd.isna(x): 
        return "DEM_GENDER_UNK"
    return f"DEM_GENDER_{str(x).strip().upper()}"

def age_tok(x):
    if pd.isna(x):
        return "DEM_AGE_UNK"
    a = int(x)
    if a >= 90:
        return "DEM_AGE_90PLUS"
    return f"DEM_AGE_{a}"

def make_demo_df(subject_id):
    if subject_id not in patients_df.index:
        return pd.DataFrame(columns=["subject_id","timestamp","event_type","event_value","source"])
    
    row = patients_df.loc[subject_id]
    toks = [gender_tok(row["gender"]), age_tok(row["anchor_age"]), yeargrp_tok(row["anchor_year_group"])]

    return pd.DataFrame({
        "subject_id": [subject_id]*len(toks),
        "timestamp":  ["1900-01-01 00:00:00"]*len(toks),  
        "event_type": ["DEM"]*len(toks),
        "event_value": toks,
        "source":     ["patients"]*len(toks),
    })


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
    patient_row = patients_df.loc[patient_id]

    patient_df = pd.read_csv(PATIENT_CSV_PATH + patient_file)
    demo_df = make_demo_df(patient_id)

    patient_df = pd.concat([demo_df, patient_df], ignore_index=True)

    if patient_row["dod"] is not None and patient_row["dod"] != "NaN":
        death_timestamp = f"{patient_row['dod']} 00:00:00"
        death_row = pd.DataFrame({"subject_id": patient_id, "timestamp": death_timestamp, "event_type": 5, "event_value": "", "source": "patients"}, index=[0])
        patient_df = pd.concat([patient_df, death_row], ignore_index=True)

    patient_df["timestamp"] = pd.to_datetime(patient_df["timestamp"], errors="coerce")
    patient_df = patient_df.dropna(subset=["timestamp"])
    patient_df = patient_df.sort_values(["timestamp"])

    
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
# - procedure
