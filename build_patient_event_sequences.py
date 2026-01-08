import pandas as pd
from typing import List
from datetime import datetime


def build_patient_event_sequences(
    df: pd.DataFrame,
) -> List[List[str]]:
    """
    Convert an event table into time-ordered event sequences per patient.

    Expected columns:
        - subject_id
        - timestamp
        - event_type
        - event_value
        - source

    Returns:
        List[List[str]]: one ordered list of event strings per subject_id
    """

    # df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # df = df.sort_values(["subject_id", "timestamp"])

    def event_to_string(row) -> str:
        source = str(row["source"])
        value = row.get("event_value")

        if pd.isna(value) or str(value).strip() == "":
            return source
        else:
            return f"{source}::{value}"

    df["event_string"] = df.apply(event_to_string, axis=1)

    sequences = []

    for subject_id, group in df.groupby("subject_id"):
        patient_sequence = []

        patient_sequence.append(f"patient::{subject_id}")

        patient_sequence.extend(group["event_string"].tolist())

        sequences.append(patient_sequence)

    return sequences