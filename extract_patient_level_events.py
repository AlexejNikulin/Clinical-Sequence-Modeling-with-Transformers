# ------------------------

import pandas as pd
import re
from pathlib import Path


DATA_DIR = Path(r"C:\Users\Leo\Desktop\magistrale\Erasmus courses\biostatics and AI for medicine\Project\physionet.org.short\files\mimiciv\3.1\hosp")
OUT_DIR  = Path(r"C:\Users\Leo\Desktop\magistrale\Erasmus courses\biostatics and AI for medicine\Project")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
PATIENTS_CSV   = DATA_DIR / "patients.csv"
ADMISSIONS_CSV = DATA_DIR / "admissions.csv"
DIAGNOSES_CSV  = DATA_DIR / "diagnoses_icd.csv"
LABEVENTS_CSV  = DATA_DIR / "labevents.csv"
DLABITEMS_CSV  = DATA_DIR / "d_labitems.csv"
EMAR_CSV       = DATA_DIR / "emar.csv"

# Output files
OUT_DYNAMIC = OUT_DIR / "events_dynamic.csv"

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

admissions = pd.read_csv(ADMISSIONS_CSV, usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
admissions["admittime"] = to_dt(admissions["admittime"])
admissions["dischtime"] = to_dt(admissions["dischtime"])

adm_events = admissions[["subject_id", "admittime"]].rename(columns={"admittime": "timestamp"})
adm_events["event_type"]  = "ADM"
adm_events["event_value"] = "HOSPITAL_ADMISSION"
adm_events["source"]      = "admissions"

dis_events = admissions[["subject_id", "dischtime"]].rename(columns={"dischtime": "timestamp"})
dis_events["event_type"]  = "DIS"
dis_events["event_value"] = "HOSPITAL_DISCHARGE"
dis_events["source"]      = "admissions"

hadm_to_dischtime = admissions.set_index("hadm_id")["dischtime"]  # for diagnoses timestamps

print("ADM rows:", len(adm_events), "DIS rows:", len(dis_events))
adm_events.head()

# ------------------------

diag = pd.read_csv(DIAGNOSES_CSV, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])

# keep only ICD-10
diag = diag[diag["icd_version"] == 10].copy()

# proxy timestamp from admissions dischtime
diag["timestamp"] = diag["hadm_id"].map(hadm_to_dischtime)
diag = diag.dropna(subset=["timestamp"])

diag["event_type"]  = "DIAG"
diag["event_value"] = diag["icd_code"].map(lambda c: f"ICD10:{sanitize_token(c)}")
diag["source"]      = "diagnoses_icd"

diag_events = diag[["subject_id", "timestamp", "event_type", "event_value", "source"]]
print("DIAG rows:", len(diag_events))
diag_events.head()


# ------------------------

labs = pd.read_csv(
    LABEVENTS_CSV,
    usecols=["subject_id", "charttime", "itemid", "valuenum", "value", "valueuom"]
)

labs["charttime"] = to_dt(labs["charttime"])
labs = labs.dropna(subset=["charttime"])

d_lab = pd.read_csv(DLABITEMS_CSV, usecols=["itemid", "label"])
item_to_label = d_lab.set_index("itemid")["label"]

def lab_token(itemid: int) -> str:
    label = item_to_label.get(itemid, f"ITEM_{itemid}")
    return f"LAB:{sanitize_token(label)}"

labs["event_type"]  = "LAB"
labs["event_value"] = labs["itemid"].map(lab_token)
labs["source"]      = "labevents"

labs = labs.rename(columns={
    "charttime": "timestamp",
    "valuenum": "value_num",
    "value": "value_text",
    "valueuom": "unit"
})

lab_events = labs[[
    "subject_id", "timestamp", "event_type", "event_value",
    "value_num", "value_text", "unit", "source"
]]

print("LAB rows:", len(lab_events))
lab_events.head()


# ------------------------

# NOTE: assumes column name is 'medication' in your emar.csv
meds = pd.read_csv(EMAR_CSV, usecols=["subject_id", "charttime", "medication"])
meds["charttime"] = to_dt(meds["charttime"])
meds = meds.dropna(subset=["charttime"])

meds["event_type"]  = "MED"
meds["event_value"] = meds["medication"].map(lambda m: f"MED:{sanitize_token(m)}")
meds["source"]      = "emar"

med_events = meds.rename(columns={"charttime": "timestamp"})[["subject_id", "timestamp", "event_type", "event_value", "source"]]
print("MED rows:", len(med_events))
med_events.head()

# ------------------------

BASE_COLS = ["subject_id","timestamp","event_type","event_value","value_num","value_text","unit","source"]

def to_base(df):
    # aggiunge colonne mancanti
    for c in ["value_num", "value_text", "unit"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df[BASE_COLS]

adm_base  = to_base(adm_events)   # già ha timestamp, event_type/value/source
dis_base  = to_base(dis_events)
diag_base = to_base(diag_events)
lab_base  = to_base(lab_events)   # già ha value_num/value_text/unit
med_base  = to_base(med_events)


# ------------------------

events_dynamic = pd.concat(
    [
        adm_events[["subject_id", "timestamp", "event_type", "event_value", "source"]],
        dis_events[["subject_id", "timestamp", "event_type", "event_value", "source"]],
        diag_events,
        lab_events,
        med_events
    ],
    ignore_index=True
)

events_dynamic["timestamp"] = to_dt(events_dynamic["timestamp"])
events_dynamic = events_dynamic.dropna(subset=["timestamp"])

# sort for readability (Patrick can sort again)
events_dynamic = events_dynamic.sort_values(["subject_id", "timestamp", "event_type"], kind="mergesort")

events_dynamic.to_csv(OUT_DYNAMIC, index=False)

print("Wrote:", OUT_DYNAMIC, "rows:", len(events_dynamic))
print("\nCounts by event_type:")
print(events_dynamic["event_type"].value_counts())
events_dynamic.head(20)

# ------------------------

df = pd.read_csv("events_dynamic.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

seq_len = df.groupby("subject_id").size()  # numero eventi per paziente

print("N patients:", seq_len.shape[0])
print("Mean events/patient:", seq_len.mean())
print("Median:", seq_len.median())
print("Min:", seq_len.min(), "Max:", seq_len.max())

seq_len.describe()


