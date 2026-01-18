from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
 
class EventType(IntEnum):
    ADMISSION = 0
    DISCHARGE = 1
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
        
        if(raw_event == "DEM"):
            return self.dem_to_token(row["event_value"])
        else:
            try:
                event = int(raw_event)
            except (ValueError, TypeError):
                return self.get_unknown_token()
 
        if event == EventType.ADMISSION:
            adm_type = str(row["result"]).strip()
            return self.adm_to_token(adm_type)
 
        elif event == EventType.DIAGNOSE:
            icd_code = str(row["event_value"]).strip()
            #icd_version = row["icd_version"]
            #is_icd_10 = int(icd_version) == 10)
            return self.diag_to_token(icd_code, True)
 
        elif event == EventType.LABEVENTS:
            icd_code = str(row["event_value"]).strip()
            result = str(row["result"]).strip()
            #icd_version = row["icd_version"]
            #is_icd_10 = int(icd_version) == 10
            return self.lab_to_token(icd_code, result)
 
        elif event == EventType.MEDICATION:
            drug_cd = str(row["event_value"]).strip()
            dose = str(row["result"]).strip()
            return self.med_to_token(drug_cd, dose)
 
        elif event == EventType.DISCHARGE:
            dis_type = str(row["result"]).strip()
            return self.readm_to_token(dis_type)
 
        elif event == EventType.DEATH:
            return self.death_to_token()
 
        return None
 
    def dem_to_token(self, event_value: str) -> str:
        return f"[{event_value}]"

    def adm_to_token(self, adm_type: str) -> str:
        return f"[ADM_{adm_type}]"
 
    def readm_to_token(self, dis_type) -> str:
        return f"[READM_{dis_type}]"
 
    def death_to_token(self) -> str:
        return "[DEATH]"
 
    def diag_to_token(self, icd_code: str, is_icd_10: bool) -> str:
        return f"[DIAG{icd_code}]" if is_icd_10 else f"[DIAG9_{icd_code}]"
 
    def lab_to_token(self, icd_code: str, result: str) -> str:
        return f"[LAB_{icd_code}_{result}]" if result != "nan" else f"[LAB_{icd_code}]"
 
    def med_to_token(self, drug_cd: str, dose: str) -> str:
        return f"[MED_{drug_cd}_{dose}]" if dose != "nan" else f"[MED_{drug_cd}]"
    
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
      Special     : 0..4
      Dem_gender  : 5..9
      Dem_age     : 10..99999
      Admission   : 100000..199999
      Diagnose    : 200000..299999
      Procedure   : 300000..399999
      Medication  : 400000..499999
      Readmission : 500000..599999
      Death       : 600000..699999
    """
    token_converter: TokenConverter = field(default_factory=TokenConverter)
 
    # token -> id maps
    special_vocab: Dict[str, int] = field(default_factory=dict)
    dem_gen_vocab: Dict[str, int] = field(default_factory=dict)
    dem_age_vocab: Dict[str, int] = field(default_factory=dict)
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
    _next_dem_gen: int = 5
    _next_dem_age: int = 10
    _next_adm: int = 100000
    _next_diag: int = 200000
    _next_labev: int = 300000
    _next_med: int = 400000
    _next_readm: int = 500000
    _next_death: int = 600000
 
    def __init__(self, df = None):
        self.VOCAB_PATH = Path("../out/vocab/vocabulary.json")
        self.COMBINED_CSV = Path("../out/splits_out/combined_train.csv")

        self.token_converter = TokenConverter()

        self.special_vocab = {}
        self.dem_gen_vocab = {}
        self.dem_age_vocab = {}
        self.admission_vocab = {}
        self.diagnosis_vocab = {}
        self.labevents_vocab = {}
        self.medication_vocab = {}
        self.readmission_vocab = {}
        self.death_vocab = {}

        self._init_special_tokens()
    
    # ------------------------
    # Save vocabulary to disk
    # ------------------------
    def save(self, path):
        path = Path(path)
        data = {
            "special": self.special_vocab,
            "demographic_gender": self.dem_gen_vocab,
            "demographic_age": self.dem_age_vocab,
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
    def load(cls, path):
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls.__new__(cls)

        vocab.token_converter = TokenConverter()

        vocab.special_vocab = {}
        vocab.dem_gen_vocab = {}
        vocab.dem_age_vocab = {}
        vocab.admission_vocab = {}
        vocab.diagnosis_vocab = {}
        vocab.labevents_vocab = {}
        vocab.medication_vocab = {}
        vocab.readmission_vocab = {}
        vocab.death_vocab = {}

        vocab._next_special = 0
        vocab._next_dem_gen = 5
        vocab._next_dem_age = 10
        vocab._next_adm = 100000
        vocab._next_diag = 200000
        vocab._next_labev = 300000
        vocab._next_med = 400000
        vocab._next_readm = 500000
        vocab._next_death = 600000

        vocab.special_vocab = data["special"]
        vocab.dem_gen_vocab = data["demographic_gender"]
        vocab.dem_age_vocab = data["demographic_age"]
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
        self.dem_gen_vocab = data["demographic_gender"]
        self.dem_age_vocab = data["demographic_age"]
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

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):

            # --- robust event_type parsing ---
            raw_event = row["event_type"]

            if pd.isna(raw_event):
                continue

            try:
                event = int(raw_event)
            except (ValueError, TypeError):
                event = raw_event

            # ---------------------------------
            token = self.token_converter.convert_row_to_token_seq(row)
            if token is None:
                continue

            vocab = self._vocab_for_event(row)
            if vocab is None:
                continue

            self._add_token(vocab, token, event)

            if(vocab == self.dem_age_vocab):
                self.dem_age_vocab = dict(sorted(self.dem_age_vocab.items(), key=lambda item: item[1]))
 
    # -------------------------
    # Internal helpers
    # -------------------------
    def _add_token(self, vocab: Dict[str, int], token: str, event) -> None:
        if token in vocab:
            return
 
        if token.startswith("[DEM_G"):
            new_id = self._next_dem_gen
            if new_id > 10:
                raise RuntimeError("Demographic gender vocab exceeded 5..10 range.")
            vocab[token] = new_id
            self._next_dem_gen += 1

        if token.startswith("[DEM_A"):
            new_id = self._next_dem_age
            if new_id > 99999:
                raise RuntimeError("Demographic age vocab exceeded 10..99999 range.")
            
            age_str = token.removeprefix("[DEM_AGE_")
            age_str = age_str.removesuffix("]")
            try:
                age = int(age_str)
                vocab[token] = age
                self._next_dem_age += 1
            except (ValueError, TypeError):
                return RuntimeError("Wrong age token!")

        if event == EventType.ADMISSION:
            new_id = self._next_adm
            if new_id > 199999:
                raise RuntimeError("Admission vocab exceeded 100000..199999 range.")
            vocab[token] = new_id
            self._next_adm += 1
 
        elif event == EventType.DIAGNOSE:
            new_id = self._next_diag
            if new_id > 299999:
                raise RuntimeError("Diagnosis vocab exceeded 200000..299999 range.")
            vocab[token] = new_id
            self._next_diag += 1
 
        elif event == EventType.LABEVENTS:
            new_id = self._next_labev
            if new_id > 399999:
                raise RuntimeError("Labevents vocab exceeded 300000..399999 range.")
            vocab[token] = new_id
            self._next_labev += 1
 
        elif event == EventType.MEDICATION:
            new_id = self._next_med
            if new_id > 499999:
                raise RuntimeError("Medication vocab exceeded 400000..499999 range.")
            vocab[token] = new_id
            self._next_med += 1
 
        elif event == EventType.DISCHARGE:
            new_id = self._next_readm
            if new_id > 599999:
                raise RuntimeError("Readmission vocab exceeded 500000..599999 range.")
            vocab[token] = new_id
            self._next_readm += 1
 
        elif event == EventType.DEATH:
            new_id = self._next_death
            if new_id > 699999:
                raise RuntimeError("Death vocab exceeded 600000..699999 range.")
            vocab[token] = new_id
            self._next_death += 1
 
    def _vocab_for_event(self, row) -> Optional[Dict[str, int]]:
        raw_event = row["event_type"]
        raw_value = row["event_value"]

        if(raw_event == "DEM"):
            if(raw_value.startswith("DEM_G")):
                return self.dem_gen_vocab
            elif(raw_value.startswith("DEM_A")):
                return self.dem_age_vocab
        else:
            try:
                event = int(raw_event)
            except (ValueError, TypeError):
                return None
            
            if event == EventType.ADMISSION:
                return self.admission_vocab
            if event == EventType.DIAGNOSE:
                return self.diagnosis_vocab
            if event == EventType.LABEVENTS:
                return self.labevents_vocab
            if event == EventType.MEDICATION:
                return self.medication_vocab
            if event == EventType.DISCHARGE:
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
 
        vocab = self._vocab_for_event(row)
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
        if token in self.dem_gen_vocab:
            return self.dem_gen_vocab[token]
        if token in self.dem_age_vocab:
            return self.dem_age_vocab[token]
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
    def build_vocabulary(self):    
        # ------------------------------------
        # Load or build vocabulary
        # ------------------------------------
        if self.VOCAB_PATH is not None and self.VOCAB_PATH.exists():
            self._load_from_disk(self.VOCAB_PATH)
            print("Vocabulary loaded!")

        else:
            df = pd.read_csv(self.COMBINED_CSV)

            if df is None:
                raise ValueError(
                    "Vocabulary needs a DataFrame when vocab_path does not exist."
                )

            self.build_from_dataframe(df)

            if self.VOCAB_PATH is not None:
                self.VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.save(self.VOCAB_PATH)
                
            print("Vocabulary built!")