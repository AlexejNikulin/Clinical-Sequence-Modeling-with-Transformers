from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import re

 
class EventType(IntEnum):
    ADMISSION = 0
    DISCHARGE = 1
    DIAGNOSE = 2
    LABEVENTS = 3
    MEDICATION = 4
    OMR = 5
    DEATH = 6
 
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
            if(row["event_value"].startswith("DEM_R")):
                return self.race_to_token(row["event_value"])
            else:
                return self.dem_to_token(row["event_value"])
        elif(raw_event == "TIME"):
            return self.time_to_token(row["event_value"]) 
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
 
        elif event == EventType.OMR:
            event_val = str(row["event_value"]).strip()
            result = str(row["result"]).strip()
            return self.omr_to_token(event_val, result)
        
        elif event == EventType.DISCHARGE:
            dis_type = str(row["result"]).strip()
            return self.disch_to_token(dis_type)
 
        elif event == EventType.DEATH:
            return self.death_to_token()
 
        return None

    def adm_to_token(self, adm_type: str) -> str:
        return f"[ADM_{adm_type}]"
    
    def dem_to_token(self, event_value: str) -> str:
        return f"[{event_value}]"
    
    def race_to_token(self, event_value: str) -> str:
        if not isinstance(event_value, str):
            return "[DEM_RACE_UNKNOWN]"

        v = event_value.upper().strip()

        if "WHITE" in v:
            return "[DEM_RACE_WHITE]"

        if "BLACK" in v or "AFRICAN" in v:
            return "[DEM_RACE_BLACK]"

        if "HISPANIC" in v or "LATINO" in v:
            return "[DEM_RACE_HISPANIC]"

        if "ASIAN" in v or "CHINESE" in v:
            return "[DEM_RACE_ASIAN]"

        if "NATIVE" in v or "PACIFIC" in v:
            return "[DEM_RACE_NATIVE]"

        if "UNKNOWN" in v or "UNABLE" in v:
            return "[DEM_RACE_UNKNOWN]"

        return "[DEM_RACE_OTHER]"
    
    def time_to_token(self, event_value: str) -> str:
        return f"[TIME_CAT{event_value}]"
 
    def disch_to_token(self, dis_type) -> str:
        return f"[DISCH_{dis_type}]"
    
    def omr_to_token(self, event_val, result) -> str:
        return f"[OMR_{event_val}_{result}]"
 
    def death_to_token(self) -> str:
        return "[DEATH]"
 
    def diag_to_token(self, icd_code: str, is_icd_10: bool) -> str:
        return f"[DIAG{icd_code}]" if is_icd_10 else f"[DIAG9_{icd_code}]"
 
    def lab_to_token(self, icd_code: str, result: str) -> str:
        return f"[LAB_{icd_code}_{result}]" if result != "nan" and result != "" else f"[LAB_{icd_code}]"
 
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
    One global special-vocabulary + per-event vocabularies in blocks by event type.
    """
    token_converter: TokenConverter = field(default_factory=TokenConverter)
 
    # token -> id maps
    special_vocab: Dict[str, int] = field(default_factory=dict) 
    time_vocab: Dict[str, int] = field(default_factory=dict) 
    dem_gen_vocab: Dict[str, int] = field(default_factory=dict)
    dem_age_vocab: Dict[str, int] = field(default_factory=dict)
    dem_race_vocab: Dict[str, int] = field(default_factory=dict)
    admission_vocab: Dict[str, int] = field(default_factory=dict)
    diagnosis_vocab: Dict[str, int] = field(default_factory=dict)
    labevents_vocab: Dict[str, int] = field(default_factory=dict)
    medication_vocab: Dict[str, int] = field(default_factory=dict)
    omr_bmi_vocab: Dict[str, int] = field(default_factory=dict)
    omr_weight_vocab: Dict[str, int] = field(default_factory=dict)
    omr_blood_pres_vocab: Dict[str, int] = field(default_factory=dict)
    discharge_vocab: Dict[str, int] = field(default_factory=dict)
    death_vocab: Dict[str, int] = field(default_factory=dict)
 
    # Special tokens (global)
    PAD: str = "[PAD]"
    MASK: str = "[MASK]"
    UNK: str = "[UNK]"

    # Start IDs per block
    START_SPECIAL: int = 0
    START_TIME: int = 3
    START_DEM_GEN: int = 15
    START_DEM_AGE: int = 19
    START_DEM_RACE: int = 120
    START_ADM: int = 150
    START_DIAG: int = 170
    START_LABEV: int = 27000
    START_MED: int = 29000
    START_OMR_BMI: int = 56000
    START_OMR_WEIGHT: int = 56500
    START_OMR_BLOOD_PRES: int = 56650
    START_DISCH: int = 70000
    START_DEATH: int = 70010
 
    # Next free IDs per block
    _next_special: int = START_SPECIAL
    _next_time: int = START_TIME
    _next_dem_gen: int = START_DEM_GEN
    _next_dem_age: int = START_DEM_AGE
    _next_dem_race: int = START_DEM_RACE
    _next_adm: int = START_ADM
    _next_diag: int = START_DIAG
    _next_labev: int = START_LABEV
    _next_med: int = START_MED
    _next_omr_bmi: int = START_OMR_BMI
    _next_omr_weight: int = START_OMR_WEIGHT
    _next_omr_blood_pres: int = START_OMR_BLOOD_PRES
    _next_disch: int = START_DISCH
    _next_death: int = START_DEATH
 
    def __init__(self, df = None):
        self.VOCAB_PATH = Path("../out/vocab/vocabulary.json")
        self.COMBINED_CSV = Path("../out/splits_out/combined_train.csv")

        self.token_converter = TokenConverter()

        self.special_vocab = {}
        self.time_vocab = {}
        self.dem_gen_vocab = {}
        self.dem_age_vocab = {}
        self.dem_race_vocab = {}
        self.dem_self_vocab = {}
        self.admission_vocab = {}
        self.diagnosis_vocab = {}
        self.labevents_vocab = {}
        self.medication_vocab = {}
        self.omr_bmi_vocab = {}
        self.omr_weight_vocab = {}
        self.omr_blood_pres_vocab = {}
        self.discharge_vocab = {}
        self.death_vocab = {}

        self._init_special_tokens()
    
    # ------------------------
    # Save vocabulary to disk
    # ------------------------
    def save(self, path):
        path = Path(path)
        data = {
            "special": self.special_vocab,
            "time": self.time_vocab,
            "demographic_gender": self.dem_gen_vocab,
            "demographic_age": self.dem_age_vocab,
            "demographic_race": self.dem_race_vocab,
            "admission": self.admission_vocab,
            "diagnosis": self.diagnosis_vocab,
            "labevents": self.labevents_vocab,
            "medication": self.medication_vocab,
            "omr_bmi": self.omr_bmi_vocab,
            "omr_weight": self.omr_weight_vocab,
            "omr_blood_pres": self.omr_blood_pres_vocab,
            "discharge": self.discharge_vocab,
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
        vocab.time_vocab = {}
        vocab.dem_gen_vocab = {}
        vocab.dem_age_vocab = {}
        vocab.dem_race_vocab = {}
        vocab.admission_vocab = {}
        vocab.diagnosis_vocab = {}
        vocab.labevents_vocab = {}
        vocab.medication_vocab = {}
        vocab.omr_bmi_vocab = {}
        vocab.omr_weight_vocab = {}
        vocab.omr_blood_pres_vocab = {}
        vocab.discharge_vocab = {}
        vocab.death_vocab = {}

        vocab._next_special = cls.START_SPECIAL
        vocab._next_time = cls.START_TIME
        vocab._next_dem_gen = cls.START_DEM_GEN
        vocab._next_dem_age = cls.START_DEM_AGE
        vocab._next_dem_race = cls.START_DEM_RACE
        vocab._next_adm = cls.START_ADM
        vocab._next_diag = cls.START_DIAG
        vocab._next_labev = cls.START_LABEV
        vocab._next_med = cls.START_MED
        vocab._next_omr_bmi = cls.START_OMR_BMI
        vocab._next_omr_weight = cls.START_OMR_WEIGHT
        vocab._next_omr_blood_pres = cls.START_OMR_BLOOD_PRES
        vocab._next_disch = cls.START_DISCH
        vocab._next_death = cls.START_DEATH

        vocab.special_vocab = data["special"]
        vocab.time_vocab = data["time"]
        vocab.dem_gen_vocab = data["demographic_gender"]
        vocab.dem_age_vocab = data["demographic_age"]
        vocab.dem_race_vocab = data["demographic_race"]
        vocab.admission_vocab = data["admission"]
        vocab.diagnosis_vocab = data["diagnosis"]
        vocab.labevents_vocab = data["labevents"]
        vocab.medication_vocab = data["medication"]
        vocab.omr_bmi_vocab = data["omr_bmi"]
        vocab.omr_weight_vocab = data["omr_weight"]
        vocab.omr_blood_pres_vocab = data["omr_blood_pres"]
        vocab.discharge_vocab = data["discharge"]
        vocab.death_vocab = data["death"]

        return vocab
 
    # ------------------------
    # Load vocabulary from disk
    # ------------------------
    def _load_from_disk(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.special_vocab = data["special"]
        self.time_vocab = data["time"]
        self.dem_gen_vocab = data["demographic_gender"]
        self.dem_age_vocab = data["demographic_age"]
        self.dem_race_vocab = data["demographic_race"]
        self.admission_vocab = data["admission"]
        self.diagnosis_vocab = data["diagnosis"]
        self.labevents_vocab = data["labevents"]
        self.medication_vocab = data["medication"]
        self.omr_bmi_vocab = data["omr_bmi"]
        self.omr_weight_vocab = data["omr_weight"]
        self.omr_blood_pres_vocab = data["omr_blood_pres"]
        self.discharge_vocab = data["discharge"]
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

            # ===============================
            # SORT ALL VOCABS (except special and time)
            # ===============================
            self.time_vocab        = self._sort_vocab(self.time_vocab,        self.START_TIME)
            self.dem_gen_vocab     = self._sort_vocab(self.dem_gen_vocab,     self.START_DEM_GEN)
            self.dem_age_vocab     = self._sort_vocab(self.dem_age_vocab,     self.START_DEM_AGE)
            self.dem_race_vocab    = self._sort_vocab(self.dem_race_vocab,    self.START_DEM_RACE)
            self.admission_vocab   = self._sort_vocab(self.admission_vocab,   self.START_ADM)
            self.diagnosis_vocab   = self._sort_vocab(self.diagnosis_vocab,   self.START_DIAG)
            self.labevents_vocab   = self._sort_vocab(self.labevents_vocab,   self.START_LABEV)
            self.medication_vocab  = self._sort_vocab(self.medication_vocab,  self.START_MED)
            self.discharge_vocab   = self._sort_vocab(self.discharge_vocab,   self.START_DISCH)
            self.death_vocab       = self._sort_vocab(self.death_vocab,       self.START_DEATH)
        self._sort_omr_vocabs_by_value()
 
    # -------------------------
    # Internal helpers
    # -------------------------
    def _add_token(self, vocab: Dict[str, int], token: str, event) -> None:
        if token in vocab:
            return
        
        if token.startswith("[TIME"):
            new_id = self._next_time
            if new_id >= self.START_DEM_GEN:
                raise RuntimeError("Time vocab exceeded allowed range.")
            vocab[token] = new_id
            self._next_time += 1
 
        if token.startswith("[DEM_G"):
            new_id = self._next_dem_gen
            if new_id >= self.START_DEM_AGE:
                raise RuntimeError("Demographic gender vocab exceeded allowed range.")
            vocab[token] = new_id
            self._next_dem_gen += 1

        if token.startswith("[DEM_R"):
            new_id = self._next_dem_race
            if new_id >= self.START_ADM:
                raise RuntimeError("Demographic race vocab exceeded allowed range.")
            vocab[token] = new_id
            self._next_dem_race += 1

        if token.startswith("[DEM_A"):
            new_id = self._next_dem_age
            if new_id >= self.START_DEM_RACE:
                raise RuntimeError("Demographic age vocab exceeded allowed range.")
            
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
            if new_id >= self.START_DIAG:
                raise RuntimeError("Admission vocab exceeded allowed range.")
            vocab[token] = new_id
            self._next_adm += 1
 
        elif event == EventType.DIAGNOSE:
            new_id = self._next_diag
            if new_id >= self.START_LABEV:
                raise RuntimeError("Diagnosis vocab exceeded allowed range.")
            vocab[token] = new_id
            self._next_diag += 1
 
        elif event == EventType.LABEVENTS:
            new_id = self._next_labev
            if new_id >= self.START_MED:
                raise RuntimeError("Labevents vocab exceeded allowed range.")
            vocab[token] = new_id
            self._next_labev += 1
 
        elif event == EventType.MEDICATION:
            new_id = self._next_med
            if new_id >= self.START_OMR_BMI:
                raise RuntimeError("Medication vocab exceeded allowed range.")
            vocab[token] = new_id
            self._next_med += 1

        elif event == EventType.OMR:
            if token.startswith("[OMR_BMI"):
                new_id = self._next_omr_bmi
                if new_id >= self.START_OMR_WEIGHT:
                    raise RuntimeError("OMR BMI vocab exceeded allowed range.")
                vocab[token] = new_id
                self._next_omr_bmi += 1
            elif token.startswith("[OMR_WEIGHT"):
                new_id = self._next_omr_weight
                if new_id >= self.START_OMR_BLOOD_PRES:
                    raise RuntimeError("OMR weight vocab exceeded allowed range.")
                vocab[token] = new_id
                self._next_omr_weight += 1
            elif token.startswith("[OMR_BLOOD"):
                new_id = self._next_omr_blood_pres
                if new_id >= self.START_DISCH:
                    raise RuntimeError("OMR blood pressure vocab exceeded allowed range.")
                vocab[token] = new_id
                self._next_omr_blood_pres += 1
 
        elif event == EventType.DISCHARGE:
            new_id = self._next_disch
            if new_id >= self.START_DEATH:
                raise RuntimeError("Discharge vocab exceeded allowed range.")
            vocab[token] = new_id
            self._next_disch += 1
 
        elif event == EventType.DEATH:
            new_id = self._next_death
            # if new_id > 699999:
            #     raise RuntimeError("Death vocab exceeded allowed range.")
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
            elif(raw_value.startswith("DEM_R")):
                return self.dem_race_vocab
        elif(raw_event == "TIME"):
            return self.time_vocab
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
            if event == EventType.OMR:
                if(raw_value.startswith("BMI")):
                    return self.omr_bmi_vocab
                elif(raw_value.startswith("WEIGHT")):
                    return self.omr_weight_vocab
                elif(raw_value.startswith("BLOOD")):
                    return self.omr_blood_pres_vocab
            if event == EventType.DISCHARGE:
                return self.discharge_vocab
            if event == EventType.DEATH:
                return self.death_vocab
            return None
        
    def vocab_name_for_token(self, token: str) -> Optional[str]:
        if token in self.special_vocab:
            return "special"
        if token in self.time_vocab:
            return "time"
        if token in self.dem_gen_vocab:
            return "demographic_gender"
        if token in self.dem_age_vocab:
            return "demographic_age"
        if token in self.dem_race_vocab:
            return "demographic_race"
        if token in self.admission_vocab:
            return "admission"
        if token in self.diagnosis_vocab:
            return "diagnosis"
        if token in self.labevents_vocab:
            return "labevents"
        if token in self.medication_vocab:
            return "medication"
        if token in self.omr_bmi_vocab:
            return "omr_bmi"
        if token in self.omr_weight_vocab:
            return "omr_weight"
        if token in self.omr_blood_pres_vocab:
            return "omr_blood_pres"
        if token in self.discharge_vocab:
            return "discharge"
        if token in self.death_vocab:
            return "death"
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
        if token in self.time_vocab:
            return self.time_vocab[token]
        if token in self.dem_gen_vocab:
            return self.dem_gen_vocab[token]
        if token in self.dem_age_vocab:
            return self.dem_age_vocab[token]
        if token in self.dem_race_vocab:
            return self.dem_race_vocab[token]
        if token in self.admission_vocab:
            return self.admission_vocab[token]
        if token in self.diagnosis_vocab:
            return self.diagnosis_vocab[token]
        if token in self.labevents_vocab:
            return self.labevents_vocab[token]
        if token in self.medication_vocab:
            return self.medication_vocab[token]
        if token in self.omr_bmi_vocab:
            return self.omr_bmi_vocab[token]
        if token in self.omr_weight_vocab:
            return self.omr_weight_vocab[token]
        if token in self.omr_blood_pres_vocab:
            return self.omr_blood_pres_vocab[token]
        if token in self.discharge_vocab:
            return self.discharge_vocab[token]
        if token in self.death_vocab:
            return self.death_vocab[token]
        # If unknown, map to UNK id
        return self.special_vocab[self.UNK]
    
    def id_to_token(self, token_id: int) -> str:
        """
        Convert token ID back to token string.
        Falls unbekannt: [UNK]
        """

        for vocab in (
            self.special_vocab,
            self.time_vocab,
            self.dem_gen_vocab,
            self.dem_age_vocab,
            self.dem_race_vocab,
            self.admission_vocab,
            self.diagnosis_vocab,
            self.labevents_vocab,
            self.medication_vocab,
            self.omr_bmi_vocab,
            self.omr_weight_vocab,
            self.omr_blood_pres_vocab,
            self.discharge_vocab,
            self.death_vocab,
        ):
            inv = {v: k for k, v in vocab.items()}
            if token_id in inv:
                return inv[token_id]

        return self.UNK
    
   
    def _token_sort_key(self, token: str):
        """
        Sort key that is alphabetical but numeric-aware.
        Example:
        [DEM_AGE_9] < [DEM_AGE_10]
        """
        parts = re.split(r"(\d+)", token)
        key = []
        for p in parts:
            if p.isdigit():
                key.append(int(p))
            else:
                key.append(p)
        return key
    
    def _sort_vocab(self, vocab: Dict[str, int], start_id: int) -> Dict[str, int]:
        """
        Sort a vocab dict by token (alphabetical / numeric-aware)
        and reassign IDs starting from start_id.
        """
        sorted_tokens = sorted(vocab.keys(), key=self._token_sort_key)

        return {
                token: start_id + i
                for i, token in enumerate(sorted_tokens)
            }

    def _extract_float_from_token(self, token: str) -> Optional[float]:
        m = re.search(r"_(-?\d+(?:\.\d+)?)\]$", token)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def _extract_bp_key(self, token: str):
        m = re.search(r"_([0-9]{2,3})/([0-9]{2,3})\]$", token)
        if not m:
            return (10**9, 10**9) 
        return (int(m.group(1)), int(m.group(2)))  

    def _reassign_sorted_ids(self, vocab: Dict[str, int], start_id: int, key_func) -> Dict[str, int]:
        items = list(vocab.keys())
        items_sorted = sorted(items, key=key_func)
        return {tok: start_id + i for i, tok in enumerate(items_sorted)}

    def _sort_omr_vocabs_by_value(self) -> None:
        # BMI
        self.omr_bmi_vocab = self._reassign_sorted_ids(
            self.omr_bmi_vocab,
            self.START_OMR_BMI,
            key_func=lambda tok: (self._extract_float_from_token(tok) is None,
                                self._extract_float_from_token(tok) if self._extract_float_from_token(tok) is not None else 10**9,
                                tok)
        )
        self._next_omr_bmi = self.START_OMR_BMI + len(self.omr_bmi_vocab)

        # WEIGHT
        self.omr_weight_vocab = self._reassign_sorted_ids(
            self.omr_weight_vocab,
            self.START_OMR_WEIGHT,
            key_func=lambda tok: (self._extract_float_from_token(tok) is None,
                                self._extract_float_from_token(tok) if self._extract_float_from_token(tok) is not None else 10**9,
                                tok)
        )
        self._next_omr_weight = self.START_OMR_WEIGHT + len(self.omr_weight_vocab)

        # BLOOD PRESSURE 
        self.omr_blood_pres_vocab = self._reassign_sorted_ids(
            self.omr_blood_pres_vocab,
            self.START_OMR_BLOOD_PRES,
            key_func=lambda tok: (*self._extract_bp_key(tok), tok)
        )
        self._next_omr_blood_pres = self.START_OMR_BLOOD_PRES + len(self.omr_blood_pres_vocab)

    
    def get_unknown_token(self) -> str:
        return self.UNK
   
    def get_padding_token(self) -> str:
        return self.PAD
   
    def get_masking_token(self) -> str:
        return self.MASK
    
    def get_size(self) -> int:
        return self.death_vocab["[DEATH]"] + 1
 
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
