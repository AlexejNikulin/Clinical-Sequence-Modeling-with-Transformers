
"""
MIMIC-IV: Create Train / Validation / Test splits (patient-level) for:
- admissions
- diagnoses_icd
- prescriptions (meds)
- labevents (labs)

Key properties:
- Patient-level split by subject_id (no leakage)
- Fixed ratios (default 80/10/10)
- Saves split patient ID lists + split table files
- Ready for huge CSVs (supports csv.gz)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train_ratio: float = 0.80
    val_ratio: float = 0.10  # test is remainder
    out_dir: str = "splits_out"
    # If True, only admissions.subject_id defines patient universe.
    # If False, patient universe is union across all loaded tables.
    universe_from_admissions_only: bool = True


# Update these to local MIMIC-IV paths (csv or csv.gz).
PATHS: Dict[str, str] = {
    "admissions": "mimic-iv/hosp/admissions.csv.gz",
    "diagnoses_icd": "mimic-iv/hosp/diagnoses_icd.csv.gz",
    "prescriptions": "mimic-iv/hosp/prescriptions.csv.gz",
    "labevents": "mimic-iv/hosp/labevents.csv.gz",
}

# Optional: speed up loads by selecting columns (must include subject_id).
USECOLS: Dict[str, Optional[list]] = {
    "admissions": ["subject_id", "hadm_id", "admittime", "dischtime"],
    "diagnoses_icd": ["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"],
    "prescriptions": ["subject_id", "hadm_id", "starttime", "stoptime", "drug"],
    "labevents": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
}

# Output compression: "gzip" (recommended) or None
OUTPUT_COMPRESSION = None
