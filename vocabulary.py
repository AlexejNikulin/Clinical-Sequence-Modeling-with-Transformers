from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional
import pandas as pd
import json
from pathlib import Path
 
class EventType(IntEnum):
    ADMISSION = 0
    READMISSION = 1
    DIAGNOSE = 2
    LABEVENTS = 3
    MEDICATION = 4
    DEATH = 5
 
class TokenConverter:
    def convert_row_to_token_seq(self, row):
        """
        Convert one row into a token string.
        Expects row to have:
          - event_type
          - icd_code, icd_version (for diag/proc)
          - formulary_drug_cd (for med)
        """
        raw_event = row["event_type"]

        if pd.isna(raw_event):
            return self.get_unknown_token()

        try:
            event = int(raw_event)
        except (ValueError, TypeError):
            # e.g. "DEM", "LAB", "", None → skip
            return self.get_unknown_token()
 
        if event == EventType.ADMISSION:
            return self.adm_to_token()
 
        elif event == EventType.DIAGNOSE:
            icd_code = str(row["event_value"]).strip()
            #icd_version = row["icd_version"]
            #is_icd_10 = int(icd_version) == 10)
            return self.diag_to_token(icd_code, True)
 
        elif event == EventType.LABEVENTS:
            icd_code = str(row["event_value"]).strip()
            #icd_version = row["icd_version"]
            #is_icd_10 = int(icd_version) == 10
            return self.lab_to_token(icd_code)
 
        elif event == EventType.MEDICATION:
            drug_cd = str(row["event_value"]).strip()
            return self.med_to_token(drug_cd)
 
        elif event == EventType.READMISSION:
            return self.readm_to_token()
 
        elif event == EventType.DEATH:
            return self.death_to_token()
 
        return None
 
    def adm_to_token(self) -> str:
        return "[ADM]"
 
    def readm_to_token(self) -> str:
        return "[READM]"
 
    def death_to_token(self) -> str:
        return "[DEATH]"
 
    def diag_to_token(self, icd_code: str, is_icd_10: bool) -> str:
        return f"[DIAG{icd_code}]" if is_icd_10 else f"[DIAG9_{icd_code}]"
 
    def lab_to_token(self, icd_code: str) -> str:
        return f"[LAB_{icd_code}]"
 
    def med_to_token(self, drug_cd: str) -> str:
        return f"[MED_{drug_cd}]"
    
    def get_unknown_token(self) -> str:
        return ["UNK"]
   
    def get_padding_token(self) -> str:
        return ["PAD"]
   
    def get_masking_token(self) -> str:
        return ["MASK"]
 
@dataclass
class Vocabulary:
    """
    One global special-vocabulary + per-event vocabularies in 10k blocks.
 
    Ranges:
      Special     : 0..9999
      Admission   : 10000..19999
      Diagnose    : 20000..29999
      Procedure   : 30000..39999
      Medication  : 40000..49999
      Readmission : 50000..59999
      Death       : 60000..69999
    """
    token_converter: TokenConverter = field(default_factory=TokenConverter)
 
    # token -> id maps
    special_vocab: Dict[str, int] = field(default_factory=dict)
    admission_vocab: Dict[str, int] = field(default_factory=dict)
    diagnosis_vocab: Dict[str, int] = field(default_factory=dict)
    labevents_vocab: Dict[str, int] = field(default_factory=dict)
    medication_vocab: Dict[str, int] = field(default_factory=dict)
    readmission_vocab: Dict[str, int] = field(default_factory=dict)
    death_vocab: Dict[str, int] = field(default_factory=dict)
 
    # Special tokens (global)
    PAD: str = "[PAD]"
    MASK: str = "[MASK]"
    UNK: str = "[UNK]"
 
    # Next free IDs per block
    _next_special: int = 0
    _next_adm: int = 10000
    _next_diag: int = 20000
    _next_labev: int = 30000
    _next_med: int = 40000
    _next_readm: int = 50000
    _next_death: int = 60000
 
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        vocab_path: Optional[Path] = None,
        token_converter: Optional[TokenConverter] = None,
    ):
        self.token_converter = token_converter or TokenConverter()

        self.special_vocab = {}
        self.admission_vocab = {}
        self.diagnosis_vocab = {}
        self.labevents_vocab = {}
        self.medication_vocab = {}
        self.readmission_vocab = {}
        self.death_vocab = {}

        self._init_special_tokens()

        # ------------------------------------
        # Auto load or build
        # ------------------------------------
        if vocab_path is not None and vocab_path.exists():
            self._load_from_disk(vocab_path)

        else:
            if df is None:
                raise ValueError(
                    "Vocabulary needs a DataFrame when vocab_path does not exist."
                )

            self.build_from_dataframe(df)

            if vocab_path is not None:
                vocab_path.parent.mkdir(parents=True, exist_ok=True)
                self.save(vocab_path)
    
    # ------------------------
    # Save vocabulary to disk
    # ------------------------
    def save(self, path: str | Path):
        path = Path(path)
        data = {
            "special": self.special_vocab,
            "admission": self.admission_vocab,
            "diagnosis": self.diagnosis_vocab,
            "labevents": self.labevents_vocab,
            "medication": self.medication_vocab,
            "readmission": self.readmission_vocab,
            "death": self.death_vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls.__new__(cls)

        vocab.token_converter = TokenConverter()

        vocab.special_vocab = {}
        vocab.admission_vocab = {}
        vocab.diagnosis_vocab = {}
        vocab.labevents_vocab = {}
        vocab.medication_vocab = {}
        vocab.readmission_vocab = {}
        vocab.death_vocab = {}

        vocab._next_special = 0
        vocab._next_adm = 10000
        vocab._next_diag = 20000
        vocab._next_labev = 30000
        vocab._next_med = 40000
        vocab._next_readm = 50000
        vocab._next_death = 60000

        # --------- jetzt sicher ---------
        vocab.special_vocab = data["special"]
        vocab.admission_vocab = data["admission"]
        vocab.diagnosis_vocab = data["diagnosis"]
        vocab.labevents_vocab = data["labevents"]
        vocab.medication_vocab = data["medication"]
        vocab.readmission_vocab = data["readmission"]
        vocab.death_vocab = data["death"]

        return vocab
 
    # ------------------------
    # Load vocabulary from disk
    # ------------------------
    def _load_from_disk(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.special_vocab = data["special"]
        self.admission_vocab = data["admission"]
        self.diagnosis_vocab = data["diagnosis"]
        self.labevents_vocab = data["labevents"]
        self.medication_vocab = data["medication"]
        self.readmission_vocab = data["readmission"]
        self.death_vocab = data["death"]
    
    # -------------------------
    # Special tokens init (global)
    # -------------------------
    def _init_special_tokens(self) -> None:
        # Reserve 0..9999 for special vocab; we only use first few
        self.special_vocab[self.PAD] = 0
        self.special_vocab[self.MASK] = 1
        self.special_vocab[self.UNK] = 2
        self._next_special = 3
 
    # -------------------------
    # Build logic
    # -------------------------
    def build_from_dataframe(self, df: pd.DataFrame) -> None:
        if "event_type" not in df.columns:
            raise ValueError("DataFrame missing required column: 'event_type'")

        for _, row in df.iterrows():

            # --- robust event_type parsing ---
            raw_event = row["event_type"]

            if pd.isna(raw_event):
                continue

            try:
                event = int(raw_event)
            except (ValueError, TypeError):
                # e.g. "DEM", "LAB", "", None → skip
                continue

            # ---------------------------------
            token = self.token_converter.convert_row_to_token_seq(row)
            if token is None:
                continue

            vocab = self._vocab_for_event(event)
            if vocab is None:
                continue

            self._add_token(vocab, token, event)
 
    # -------------------------
    # Internal helpers
    # -------------------------
    def _add_token(self, vocab: Dict[str, int], token: str, event: int) -> None:
        if token in vocab:
            return
 
        if event == EventType.ADMISSION:
            new_id = self._next_adm
            if new_id > 19999:
                raise RuntimeError("Admission vocab exceeded 10000..19999 range.")
            vocab[token] = new_id
            self._next_adm += 1
 
        elif event == EventType.DIAGNOSE:
            new_id = self._next_diag
            if new_id > 29999:
                raise RuntimeError("Diagnosis vocab exceeded 20000..29999 range.")
            vocab[token] = new_id
            self._next_diag += 1
 
        elif event == EventType.LABEVENTS:
            new_id = self._next_labev
            if new_id > 39999:
                raise RuntimeError("Labevents vocab exceeded 30000..39999 range.")
            vocab[token] = new_id
            self._next_labev += 1
 
        elif event == EventType.MEDICATION:
            new_id = self._next_med
            if new_id > 49999:
                raise RuntimeError("Medication vocab exceeded 40000..49999 range.")
            vocab[token] = new_id
            self._next_med += 1
 
        elif event == EventType.READMISSION:
            new_id = self._next_readm
            if new_id > 59999:
                raise RuntimeError("Readmission vocab exceeded 50000..59999 range.")
            vocab[token] = new_id
            self._next_readm += 1
 
        elif event == EventType.DEATH:
            new_id = self._next_death
            if new_id > 69999:
                raise RuntimeError("Death vocab exceeded 60000..69999 range.")
            vocab[token] = new_id
            self._next_death += 1
 
    def _vocab_for_event(self, event: int) -> Optional[Dict[str, int]]:
        if event == EventType.ADMISSION:
            return self.admission_vocab
        if event == EventType.DIAGNOSE:
            return self.diagnosis_vocab
        if event == EventType.LABEVENTS:
            return self.labevents_vocab
        if event == EventType.MEDICATION:
            return self.medication_vocab
        if event == EventType.READMISSION:
            return self.readmission_vocab
        if event == EventType.DEATH:
            return self.death_vocab
        return None
       
    # -------------------------
    # Requested: row -> token, fallback to global UNK if missing
    # -------------------------
    def row_to_token(self, row: Any) -> str:
        """
        Create token via TokenConverter, then check whether it exists in the
        corresponding vocabulary. If not, return global [UNK].
        """
        token = self.token_converter.convert_row_to_token_seq(row)
        if token is None:
            return self.UNK
        
        raw_event = row["event_type"]

        if pd.isna(raw_event):
            return self.UNK

        try:
            event = int(raw_event)
        except (ValueError, TypeError):
            # e.g. "DEM", "LAB", "", None → skip
            return self.UNK
 
        vocab = self._vocab_for_event(int(row["event_type"]))
        if vocab is None:
            return self.UNK
 
        return token if token in vocab else self.UNK
       
    # -------------------------
    # Public API
    # -------------------------
    def token_to_id(self, token: str) -> int:
        """
        Look up a token in all vocabularies.
        Raises KeyError if not found.
        """
        if token in self.special_vocab:
            return self.special_vocab[token]
        if token in self.admission_vocab:
            return self.admission_vocab[token]
        if token in self.diagnosis_vocab:
            return self.diagnosis_vocab[token]
        if token in self.labevents_vocab:
            return self.labevents_vocab[token]
        if token in self.medication_vocab:
            return self.medication_vocab[token]
        if token in self.readmission_vocab:
            return self.readmission_vocab[token]
        if token in self.death_vocab:
            return self.death_vocab[token]
        # If unknown, map to UNK id
        return self.special_vocab[self.UNK]
   
    def get_unknown_token(self) -> str:
        return self.UNK
   
    def get_padding_token(self) -> str:
        return self.PAD
   
    def get_masking_token(self) -> str:
        return self.MASK
 
 #-----------MAIN CODE--------------
if __name__ == "__main__":
 
    VOCAB_PATH = Path("../out/vocab/vocabulary.json")
    COMBINED_CSV = Path("../out/merge_and_sort/combined.csv")
 
    # ------------------------------------
    # Load or build vocabulary
    # ------------------------------------
    if VOCAB_PATH.exists():
        vocab = Vocabulary.load(VOCAB_PATH)
 
        print("Vocabulary loaded. Converting 10 random rows to tokens...\n")
 
        df = pd.read_csv(COMBINED_CSV)
 
        # 10 zufällige Zeilen
        sample_df = df.sample(n=10, random_state=42)

        print("\n=== SPECIAL VOCAB ===")
        print(vocab.special_vocab)
    
        print("\n=== ADMISSION VOCAB ===")
        print(vocab.admission_vocab)
    
        print("\n=== DIAGNOSIS VOCAB ===")
        print(vocab.diagnosis_vocab)
    
        print("\n=== PROCEDURE VOCAB ===")
        print(vocab.labevents_vocab)
    
        print("\n=== MEDICATION VOCAB ===")
        print(vocab.medication_vocab)
    
        print("\n=== READMISSION VOCAB ===")
        print(vocab.readmission_vocab)
    
        print("\n=== DEATH VOCAB ===")
        print(vocab.death_vocab)
 
        for idx, row in sample_df.iterrows():
            token = vocab.row_to_token(row)
 
            print(f"Row: {row}")
            print(f"Event type: {row['event_type']}")
            print(f"Token     : {token}")
            print("-" * 40)
 
    else:
        print("Vocabulary not found. Building new vocabulary...")
        df = pd.read_csv(COMBINED_CSV)
        vocab = Vocabulary(df, vocab_path=VOCAB_PATH)