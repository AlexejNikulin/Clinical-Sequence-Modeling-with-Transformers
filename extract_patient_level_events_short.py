# ------------------------

import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

class PatientLevelEventExtractor_Short:
    def __init__(self):
        self.DATA_DIR = Path(r"../physionet.org/files/mimiciv/3.1/hosp")
        self.OUT_DIR  = Path(r"../out/extract_patient_level_events")
        self.REF_VAL_DIR  = Path(r"ref_ranges")

        self.ROWS = 1000

        self.OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Input files
        self.PATIENTS_CSV   = self.DATA_DIR / "patients.csv"
        self.ADMISSIONS_CSV = self.DATA_DIR / "admissions.csv"
        self.DIAGNOSES_CSV  = self.DATA_DIR / "diagnoses_icd.csv"
        self.LABEVENTS_CSV  = self.DATA_DIR / "labevents.csv"
        self.DLABITEMS_CSV  = self.DATA_DIR / "d_labitems.csv"
        self.EMAR_CSV       = self.DATA_DIR / "emar.csv"
        self.EMAR_DETAIL_CSV       = self.DATA_DIR / "emar_detail.csv"

        self.REF_VAL_CSV = self.REF_VAL_DIR / "ref_ranges.csv"

        # Output files
        self.OUT_DYNAMIC = self.OUT_DIR / "events_dynamic_"

        self.PATIENTS_CSV, self.ADMISSIONS_CSV

    # ------------------------

    def sanitize_token(self, x) -> str:
        """Uppercase + keep only [A-Z0-9_], collapse separators."""
        if pd.isna(x):
            return "UNK"
        s = str(x).strip().upper()
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s if s else "UNK"

    def to_dt(self, series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce")

    # ------------------------

    def save_events_dynamic(self, input_df, file_suffix):

        events_dynamic = pd.concat([input_df], ignore_index=True)

        events_dynamic["timestamp"] = self.to_dt(events_dynamic["timestamp"])
        events_dynamic = events_dynamic.dropna(subset=["timestamp"])

        # sort for readability (Patrick can sort again)
        events_dynamic = events_dynamic.sort_values(["subject_id", "timestamp", "event_type"], kind="mergesort")

        outfile = Path(str(self.OUT_DYNAMIC) + file_suffix + ".csv")
        events_dynamic.to_csv(outfile, index=False)

        print("Wrote:", outfile, "rows:", len(events_dynamic))
        print("\nCounts by event_type:")
        print(events_dynamic["event_type"].value_counts())
        events_dynamic.head(20)

    # ------------------------

    def start_extraction(self):
        admissions = pd.read_csv(self.ADMISSIONS_CSV, usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"], nrows=self.ROWS)
        admissions["admittime"] = self.to_dt(admissions["admittime"])
        admissions["dischtime"] = self.to_dt(admissions["dischtime"])

        adm_events = admissions[["subject_id", "admittime", "admission_type"]].rename(columns={"admittime": "timestamp", "admission_type": "result"})
        adm_events["event_type"]  = 0
        adm_events["event_value"] = ""

        dis_events = admissions[["subject_id", "dischtime", "admission_type"]].rename(columns={"dischtime": "timestamp", "admission_type": "result"})
        dis_events["event_type"]  = 1
        dis_events["event_value"] = ""

        hadm_to_dischtime = admissions.set_index("hadm_id")["dischtime"]  # for diagnoses timestamps

        print("ADM rows:", len(adm_events), "DIS rows:", len(dis_events))
        adm_events.head()

        self.save_events_dynamic(
            adm_events[["subject_id", "timestamp", "event_type", "event_value", "result"]],
            "admissions"
        )

        self.save_events_dynamic(
            dis_events[["subject_id", "timestamp", "event_type", "event_value", "result"]],
            "discharges"
        )

        # ------------------------

        diag = pd.read_csv(self.DIAGNOSES_CSV, usecols=["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"], nrows=self.ROWS)

        diag_icd10 = diag[diag["icd_version"] == 10].copy()
        diag_icd9 = diag[diag["icd_version"] == 9].copy()

        # proxy timestamp from admissions dischtime
        diag_icd10["timestamp"] = diag_icd10["hadm_id"].map(hadm_to_dischtime)
        diag_icd10 = diag_icd10.dropna(subset=["timestamp"])
        diag_icd10["event_type"]  = 2
        diag_icd10["event_value"] = diag_icd10["icd_code"].map(lambda c: f"10_{self.sanitize_token(c)}")
        diag_icd10["result"]      = ""
        diag10_events = diag_icd10[["subject_id", "timestamp", "event_type", "event_value", "result", "hadm_id", "seq_num"]]

        # proxy timestamp from admissions dischtime
        diag_icd9["timestamp"] = diag_icd9["hadm_id"].map(hadm_to_dischtime) 
        diag_icd9 = diag_icd9.dropna(subset=["timestamp"])
        diag_icd9["event_type"]  = 2
        diag_icd9["event_value"] = diag_icd9["icd_code"].map(lambda c: f"9_{self.sanitize_token(c)}")
        diag_icd9["result"]      = ""
        diag9_events = diag_icd9[["subject_id", "timestamp", "event_type", "event_value", "result"]]

        diag_events = pd.concat([diag10_events, diag9_events], ignore_index=True)

        diag_events["timestamp"] -= pd.Timedelta(seconds=1) # Diagnosis timestamp -1 second so the readmission/discharge is always last in the sequence

        diag_events = diag_events.sort_values(by=["hadm_id", "seq_num"], kind="mergesort").reset_index(drop=True)
        diag_events = diag_events[["subject_id", "timestamp", "event_type", "event_value", "result"]]

        print("DIAG rows:", len(diag_events))
        diag_events.head()

        self.save_events_dynamic(
            diag_events,
            "diagnoses_icd"
        )

        # ------------------------

        # Lookup table for itemid <-> label
        d_lab = pd.read_csv(
            self.DLABITEMS_CSV,
            usecols=["itemid", "label"],
            nrows=self.ROWS
        )
        d_lab["clean_label"] = d_lab["label"].apply(
            lambda x: self.sanitize_token(x)
        )
        item_map = d_lab.set_index("itemid")["clean_label"].to_dict()

        output_file = Path(str(self.OUT_DYNAMIC) + "labevents.csv")
        print(f"Processing to: {output_file}")

        # --- load reference values ONCE ---
        ref_vals = pd.read_csv(self.REF_VAL_CSV)

        ref_vals["itemid"] = pd.to_numeric(ref_vals["itemid"], errors="coerce")
        ref_vals["ref_range_lower"] = pd.to_numeric(ref_vals["ref_range_lower"], errors="coerce")
        ref_vals["ref_range_upper"] = pd.to_numeric(ref_vals["ref_range_upper"], errors="coerce")

        ref_vals_agg = (
            ref_vals
            .dropna(subset=["itemid"])
            .groupby("itemid", as_index=False)
            .agg(
                ref_range_lower=("ref_range_lower", "min"),
                ref_range_upper=("ref_range_upper", "max")
            )
        )

        # --- chunk processing ---
        with pd.read_csv(
            self.LABEVENTS_CSV,
            usecols=["subject_id", "charttime", "itemid", "valuenum", "value"],
            chunksize=1_000_000,
            nrows=self.ROWS
        ) as reader:

            for i, chunk in enumerate(reader):

                chunk["charttime"] = self.to_dt(chunk["charttime"])
                chunk = chunk.dropna(subset=["charttime"])
                chunk = chunk.rename(columns={"charttime": "timestamp"})

                chunk["event_type"] = 3
                chunk["event_value"] = chunk["itemid"].map(item_map)

                # --- merge reference ranges ---
                chunk = chunk.merge(
                    ref_vals_agg,
                    on="itemid",
                    how="left",
                    validate="m:1"
                )

                low = chunk["ref_range_lower"]
                high = chunk["ref_range_upper"]
                vnum = chunk["valuenum"]
                vtxt = chunk["value"].astype(str).str.strip()

                # --- numeric classification ---
                chunk["result"] = np.select(
                    [
                        low.notna() & vnum.notna() & (vnum < low),
                        high.notna() & vnum.notna() & (vnum > high),
                        low.notna() & high.notna() & vnum.notna(),
                    ],
                    ["LOW", "HIGH", "NORMAL"],
                    default=""
                )

                # --- fallback: no ref-range but qualitative value ---
                no_ref = low.isna() & high.isna()
                has_value = vtxt.notna() & (vtxt != "") & (vtxt != "nan")

                chunk.loc[no_ref & has_value, "result"] = vtxt

                lab_events = chunk[
                    ["subject_id", "timestamp", "event_type", "event_value", "result"]
                ]

                write_mode = "w" if i == 0 else "a"
                write_header = (i == 0)

                lab_events.to_csv(
                    output_file,
                    mode=write_mode,
                    header=write_header,
                    index=False
                )

                print(f"Chunk {i} processed. Rows written: {len(lab_events)}")
        # ------------------------

        meds_detail = pd.read_csv(self.EMAR_DETAIL_CSV, usecols=["emar_id", "pharmacy_id", "dose_given", "dose_given_unit"], nrows=self.ROWS)
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

        meds = pd.read_csv(self.EMAR_CSV, usecols=["subject_id", "emar_id", "charttime", "medication", "event_txt"], nrows=self.ROWS)
                
        meds = meds[meds["event_txt"].notna() & ~meds["event_txt"].isin(['Not Applied', 'Not Given', 'Hold Dose', 'Not Flushed', 'Not Given per Sliding Scale', 'Not Confirmed', 'Not Started', 'Infusion Reconciliation Not Done', 'Not Stopped per Sliding Scale', 'Not Assessed', 'Pain score re-assess not done', 'TPN Rate Not Changed', 'Delayed Not Applied', 'Delayed Not Flushed', 'Delayed Not Removed', 'Not Started per Sliding Scale', 'Not Read' , 'Delayed Not Started', 'Delayed Not Confirmed', 'Delayed Not Assessed'])
]
        meds = meds.drop(columns=["event_txt"])
    
        meds["charttime"] = self.to_dt(meds["charttime"])
        meds = meds.dropna(subset=["charttime"])

        meds["event_type"]  = 4
        meds["event_value"] = meds["medication"].map(lambda m: f"{self.sanitize_token(m)}")
        meds["result"]      = ""

        for index, row in tqdm(meds.iterrows(), total=meds.shape[0]):
            meds.at[index, "result"] = lookup_dict.get(row["emar_id"], "")

        med_events = meds.rename(columns={"charttime": "timestamp"})[["subject_id", "timestamp", "event_type", "event_value", "result"]]
        print("MED rows:", len(med_events))
        med_events.head()

        self.save_events_dynamic(
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


