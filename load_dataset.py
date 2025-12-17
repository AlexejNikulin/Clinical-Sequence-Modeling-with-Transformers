import os
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIRECTORY, "../", "physionet.org/files/mimiciv/3.1")

def load_dataframe(module, table):
    csv_path = os.path.join(DATASET_PATH, module, f"{table}.csv")
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    admissions_df = load_dataframe("hosp", "admissions")
    patients_df = load_dataframe("hosp", "patients")
    diagnoses_df = load_dataframe("hosp", "diagnoses_icd")
    diagnoses_definition_df = load_dataframe("hosp", "d_icd_diagnoses")

    # For testing, let's get some information for this specific patient
    test_subject_id = 10000068
    print()
    print("PATIENT INFORMATION")
    print(patients_df[patients_df["subject_id"] == test_subject_id])
    print()
    print("ADMISSION INFORMATION")
    print(admissions_df[admissions_df["subject_id"] == test_subject_id])
    print()
    print("DIAGNOSES INFORMATION")
    print(diagnoses_df[diagnoses_df["subject_id"] == test_subject_id])
    print()

    # For fun, let's look up the ICD definition of the first diagnosis from the corresponding table
    diagnosis = diagnoses_df[diagnoses_df["subject_id"] == test_subject_id].iloc[0]
    icd_definition = diagnoses_definition_df[(diagnoses_definition_df["icd_code"] == diagnosis["icd_code"]) & (diagnoses_definition_df["icd_version"] == diagnosis["icd_version"])]
    print("ICD DEFINITION OF FIRST DIAGNOSIS")
    print(icd_definition)