import pandas as pd
from tqdm import tqdm
import os

class SortMerger:
    def __init__(self):
        # FOR DEBUGGING
        self.ADMISSION_CSV = "../out/extract_patient_level_events/events_dynamic_admissions.csv"
        self.DIAGNOSES_CSV = "../out/extract_patient_level_events/events_dynamic_diagnoses_icd.csv"
        self.DISCHARGES_CSV = "../out/extract_patient_level_events/events_dynamic_discharges.csv"
        self.MEDICATION_CSV = "../out/extract_patient_level_events/events_dynamic_emar.csv"
        self.LABEVENTS_CSV = "../out/extract_patient_level_events/events_dynamic_labevents.csv"
        self.OMR_CSV = "../out/extract_patient_level_events/events_dynamic_omr.csv"

        self.PATIENT_CSV_PATH = "../out/merge_and_sort/patients/"
        self.PATIENTS_CSV = "../physionet.org/files/mimiciv/3.1/hosp/patients.csv"  
        self.patients_df = pd.read_csv(self.PATIENTS_CSV, usecols=["subject_id","gender","anchor_age","anchor_year_group","dod"])
        self.patients_df = self.patients_df.set_index("subject_id")

        self.ADM_CSV = "../physionet.org/files/mimiciv/3.1/hosp/admissions.csv"  
        self.adm_df = pd.read_csv(self.ADM_CSV, usecols=["subject_id","race"])
        self.adm_df = self.adm_df.set_index("subject_id")

        self.FINAL_CSV_PATH = "../out/merge_and_sort/combined.csv"

    def yeargrp_tok(self, x):
        if pd.isna(x): 
            return "DEM_YEARGRP_UNK"
        s = str(x).strip().replace(" ", "").replace("-", "_")
        return f"DEM_YEARGRP_{s}"

    def gender_tok(self, x):
        if pd.isna(x): 
            return "DEM_GENDER_UNK"
        return f"DEM_GENDER_{str(x).strip().upper()}"

    def age_tok(self, x):
        if pd.isna(x):
            return "DEM_AGE_UNK"
        a = int(x)
        return f"DEM_AGE_{a}"
    
    def race_tok(self, x):
        if pd.isna(x):
            return "DEM_RACE_UNK"
        return f"DEM_RACE_{str(x).strip().upper()}"

    def make_demo_df(self, subject_id):
        if subject_id not in self.patients_df.index or subject_id not in self.adm_df.index:
            return pd.DataFrame(columns=["subject_id","timestamp","event_type","event_value","result"])
        
        row = self.patients_df.loc[subject_id]
        row_adm_obj = self.adm_df.loc[subject_id]

        if isinstance(row_adm_obj, pd.DataFrame):
            row_adm = row_adm_obj.iloc[0]
        else:
            row_adm = row_adm_obj

        toks = [self.gender_tok(row["gender"]), self.age_tok(row["anchor_age"]), self.race_tok(row_adm["race"])] # , self.yeargrp_tok(row["anchor_year_group"])

        return pd.DataFrame({
            "subject_id": [subject_id]*len(toks),
            "timestamp":  ["1900-01-01 00:00:00"]*len(toks),  
            "event_type": ["DEM"]*len(toks),
            "event_value": toks,
            "result":     [""]*len(toks),
        })

    def write_patient_events_for_file(self, file, subject_ids_with_file):
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
                        self.PATIENT_CSV_PATH + str(subject_id) + ".csv", 
                        mode=write_mode, 
                        header=write_header, 
                        index=False
                    )

                    subject_ids_with_file.add(subject_id)

    def sort_and_merge(self):
        admissions_df = pd.read_csv(self.ADMISSION_CSV)
        subject_ids = admissions_df["subject_id"].unique()

        subject_ids_with_file = set()

        # Create one file for each patient
        os.makedirs(self.PATIENT_CSV_PATH)
        self.write_patient_events_for_file(self.ADMISSION_CSV, subject_ids_with_file)
        self.write_patient_events_for_file(self.DIAGNOSES_CSV, subject_ids_with_file)
        self.write_patient_events_for_file(self.DISCHARGES_CSV, subject_ids_with_file)
        self.write_patient_events_for_file(self.MEDICATION_CSV, subject_ids_with_file)
        self.write_patient_events_for_file(self.LABEVENTS_CSV, subject_ids_with_file)
        self.write_patient_events_for_file(self.OMR_CSV, subject_ids_with_file)

        # For each patient, sort the events, and append them to the global dataframe
        patient_files = os.listdir(self.PATIENT_CSV_PATH)
        idx = 0
        for patient_file in tqdm(patient_files, desc="Merging patients"):
            patient_id = int(patient_file.split(".")[0])
            patient_row = self.patients_df.loc[patient_id]

            patient_df = pd.read_csv(self.PATIENT_CSV_PATH + patient_file)
            demo_df = self.make_demo_df(patient_id)

            patient_df = pd.concat([demo_df, patient_df], ignore_index=True)

            if patient_row["dod"] is not None and patient_row["dod"] != "NaN":
                death_timestamp = f"{patient_row['dod']} 00:00:00"
                death_row = pd.DataFrame({"subject_id": patient_id, "timestamp": death_timestamp, "event_type": 6, "event_value": "", "result": ""}, index=[0])
                patient_df = pd.concat([patient_df, death_row], ignore_index=True)

            patient_df["timestamp"] = pd.to_datetime(patient_df["timestamp"], errors="coerce")
            patient_df = patient_df.dropna(subset=["timestamp"])
            patient_df = patient_df.sort_values(["timestamp"])

            has_admission = patient_df['event_type'].isin([0, 1]).any()
            if not has_admission:
                continue
            
            write_mode = 'w' if idx == 0 else 'a' # write / append
            write_header = (idx == 0)

            patient_df.to_csv(
                self.FINAL_CSV_PATH, 
                mode=write_mode, 
                header=write_header, 
                index=False
            )

            # os.remove(PATIENT_CSV_PATH + patient_file)

            idx += 1

        # What's still missing
        # - procedure
