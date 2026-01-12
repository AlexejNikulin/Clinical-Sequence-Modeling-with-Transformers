# ------------------------

import pandas as pd
import re
from pathlib import Path


DATA_DIR = Path(r"../physionet.org/files/mimiciv/3.1/hosp")
OUT_DIR  = Path(r"../out/extract_patient_level_events")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
PATIENTS_CSV   = DATA_DIR / "patients.csv"
ADMISSIONS_CSV = DATA_DIR / "admissions.csv"
DIAGNOSES_CSV  = DATA_DIR / "diagnoses_icd.csv"
LABEVENTS_CSV  = DATA_DIR / "labevents.csv"
DLABITEMS_CSV  = DATA_DIR / "d_labitems.csv"
EMAR_CSV       = DATA_DIR / "emar.csv"
EMAR_DETAIL_CSV       = DATA_DIR / "emar_detail.csv"

# Output files
OUT_DYNAMIC = OUT_DIR / "events_dynamic_"

PATIENTS_CSV, ADMISSIONS_CSV


# ------------------------

def sanitize_token(x) -> str:
    """Uppercase + keep only [A-Z0-9_], collapse separators."""
    if pd.isna(x):
        return "UNK"
    s = str(x).strip().upper()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "UNK"

def to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

# ------------------------

def save_events_dynamic(input_df, file_suffix):

    events_dynamic = pd.concat([input_df], ignore_index=True)

    events_dynamic["timestamp"] = to_dt(events_dynamic["timestamp"])
    events_dynamic = events_dynamic.dropna(subset=["timestamp"])

    # sort for readability (Patrick can sort again)
    events_dynamic = events_dynamic.sort_values(["subject_id", "timestamp", "event_type"], kind="mergesort")

    outfile = Path(str(OUT_DYNAMIC) + file_suffix + ".csv")
    events_dynamic.to_csv(outfile, index=False)

    print("Wrote:", outfile, "rows:", len(events_dynamic))
    print("\nCounts by event_type:")
    print(events_dynamic["event_type"].value_counts())
    events_dynamic.head(20)

# ------------------------

admissions = pd.read_csv(ADMISSIONS_CSV, usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"])
admissions["admittime"] = to_dt(admissions["admittime"])
admissions["dischtime"] = to_dt(admissions["dischtime"])

adm_events = admissions[["subject_id", "admittime", "admission_type"]].rename(columns={"admittime": "timestamp", "admission_type": "result"})
adm_events["event_type"]  = 0
adm_events["event_value"] = ""

dis_events = admissions[["subject_id", "dischtime", "admission_type"]].rename(columns={"dischtime": "timestamp", "admission_type": "result"})
dis_events["event_type"]  = 1
dis_events["event_value"] = ""

hadm_to_dischtime = admissions.set_index("hadm_id")["dischtime"]  # for diagnoses timestamps

print("ADM rows:", len(adm_events), "DIS rows:", len(dis_events))
adm_events.head()

save_events_dynamic(
    adm_events[["subject_id", "timestamp", "event_type", "event_value", "result"]],
    "admissions"
)

save_events_dynamic(
    dis_events[["subject_id", "timestamp", "event_type", "event_value", "result"]],
    "discharges"
)

# ------------------------

diag = pd.read_csv(DIAGNOSES_CSV, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])

diag_icd10 = diag[diag["icd_version"] == 10].copy()
diag_icd9 = diag[diag["icd_version"] == 9].copy()

# proxy timestamp from admissions dischtime
diag_icd10["timestamp"] = diag_icd10["hadm_id"].map(hadm_to_dischtime)
diag_icd10 = diag_icd10.dropna(subset=["timestamp"])
diag_icd10["event_type"]  = 2
diag_icd10["event_value"] = diag_icd10["icd_code"].map(lambda c: f"10_{sanitize_token(c)}")
diag_icd10["result"]      = ""
diag10_events = diag_icd10[["subject_id", "timestamp", "event_type", "event_value", "result"]]

# proxy timestamp from admissions dischtime
diag_icd9["timestamp"] = diag_icd9["hadm_id"].map(hadm_to_dischtime)
diag_icd9 = diag_icd9.dropna(subset=["timestamp"])
diag_icd9["event_type"]  = 2
diag_icd9["event_value"] = diag_icd9["icd_code"].map(lambda c: f"9_{sanitize_token(c)}")
diag_icd9["result"]      = ""
diag9_events = diag_icd9[["subject_id", "timestamp", "event_type", "event_value", "result"]]

diag_events = pd.concat([diag10_events, diag9_events], ignore_index=True)

print("DIAG rows:", len(diag_events))
diag_events.head()

save_events_dynamic(
    diag_events,
    "diagnoses_icd"
)

# ------------------------

# Lookup table for itemid <-> label
d_lab = pd.read_csv(DLABITEMS_CSV, usecols=["itemid", "label"])
d_lab["clean_label"] = d_lab["label"].apply(lambda x: f"{sanitize_token(x)}")
item_map = d_lab.set_index("itemid")["clean_label"].to_dict()

output_file = Path(str(OUT_DYNAMIC) + "labevents.csv")
print(f"Processing to: {output_file}")

# Process in chunks, as this file is 18GB large
with pd.read_csv(
    LABEVENTS_CSV,
    usecols=["subject_id", "charttime", "itemid", "valuenum", "value", "valueuom", "flag"],
    chunksize=1_000_000
) as reader:
    
    for i, chunk in enumerate(reader):
        chunk["charttime"] = to_dt(chunk["charttime"])
        chunk = chunk.dropna(subset=["charttime"])

        chunk["event_type"] = 3
        chunk["result"] = chunk["flag"]

        chunk["event_value"] = chunk["itemid"].map(item_map)
        
        # Fallback for unknown IDs
        mask_unknown = chunk["event_value"].isna()
        if mask_unknown.any():
            chunk.loc[mask_unknown, "event_value"] = "UNK"

        chunk = chunk.rename(columns={
            "charttime": "timestamp",
            "valuenum": "value_num",
            "value": "value_text",
            "valueuom": "unit"
        })

        lab_events = chunk[[
            "subject_id", "timestamp", "event_type", "event_value",
            # "value_num", "value_text", "unit", # unused right now, would have to be merged into event_value to not break the scripts
            "result"
        ]]

        write_mode = 'w' if i == 0 else 'a' # write / append
        write_header = (i == 0)

        lab_events.to_csv(
            output_file, 
            mode=write_mode, 
            header=write_header, 
            index=False
        )

        print(f"Chunk {i} processed and appended. Rows: {len(lab_events)}")

print("Processing complete.")

# ------------------------

meds_detail = pd.read_csv(EMAR_DETAIL_CSV, usecols=["emar_id", "pharmacy_id", "dose_given", "dose_given_unit"])
meds_detail = (
    meds_detail[meds_detail["pharmacy_id"].notna() & (meds_detail["pharmacy_id"] != "")]
    .drop_duplicates("emar_id")
    .set_index("emar_id")
)
meds_detail["result"] = (
    meds_detail["dose_given"].astype(str) + 
    "_" + 
    meds_detail["dose_given_unit"].astype(str)
)
lookup_dict = meds_detail["result"].to_dict()
del meds_detail

# NOTE: assumes column name is 'medication' in your emar.csv
meds = pd.read_csv(EMAR_CSV, usecols=["subject_id", "emar_id", "charttime", "medication"])
meds["charttime"] = to_dt(meds["charttime"])
meds = meds.dropna(subset=["charttime"])

meds["event_type"]  = 4
meds["event_value"] = meds["medication"].map(lambda m: f"{sanitize_token(m)}")
meds["result"]      = ""

for index, row in meds.iterrows():
    meds.at[index, "result"] = lookup_dict.get(row["emar_id"], "")

med_events = meds.rename(columns={"charttime": "timestamp"})[["subject_id", "timestamp", "event_type", "event_value", "result"]]
print("MED rows:", len(med_events))
med_events.head()

save_events_dynamic(
    med_events,
    "emar"
)

# ------------------------

# BASE_COLS = ["subject_id","timestamp","event_type","event_value","value_num","value_text","unit","source"]

# def to_base(df):
#     # aggiunge colonne mancanti
#     for c in ["value_num", "value_text", "unit"]:
#         if c not in df.columns:
#             df[c] = pd.NA
#     return df[BASE_COLS]

# adm_base  = to_base(adm_events)   # già ha timestamp, event_type/value/source
# dis_base  = to_base(dis_events)
# diag_base = to_base(diag_events)
# lab_base  = to_base(lab_events)   # già ha value_num/value_text/unit
# med_base  = to_base(med_events)

# ------------------------

# df = pd.read_csv("events_dynamic.csv")
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# df = df.dropna(subset=["timestamp"])

# seq_len = df.groupby("subject_id").size()  # numero eventi per paziente

# print("N patients:", seq_len.shape[0])
# print("Mean events/patient:", seq_len.mean())
# print("Median:", seq_len.median())
# print("Min:", seq_len.min(), "Max:", seq_len.max())

# seq_len.describe()


