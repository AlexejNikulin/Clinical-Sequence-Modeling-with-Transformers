from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional
import pandas as pd
 
class EventType(IntEnum):
    ADMISSION = 0
    DIAGNOSE = 1
    PROCEDURE = 2
    MEDICATION = 3
    READMISSION = 4
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
        event = int(row["event_type"])
 
        if event == EventType.ADMISSION:
            return self.adm_to_token()
 
        elif event == EventType.DIAGNOSE:
            icd_code = str(row["icd_code"]).strip()
            icd_version = row["icd_version"]
            is_icd_10 = int(icd_version) == 10
            return self.diag_to_token(icd_code, is_icd_10)
 
        elif event == EventType.PROCEDURE:
            icd_code = str(row["icd_code"]).strip()
            icd_version = row["icd_version"]
            is_icd_10 = int(icd_version) == 10
            return self.proc_to_token(icd_code, is_icd_10)
 
        elif event == EventType.MEDICATION:
            drug_cd = str(row["formulary_drug_cd"]).strip()
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
        return f"[DIAG_{icd_code}]" if is_icd_10 else f"[DIAG9_{icd_code}]"
 
    def proc_to_token(self, icd_code: str, is_icd_10: bool) -> str:
        return f"[PROC_{icd_code}]" if is_icd_10 else f"[PROC9_{icd_code}]"
 
    def med_to_token(self, drug_cd: str) -> str:
        return f"[MED_{drug_cd}]"
 
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
    procedure_vocab: Dict[str, int] = field(default_factory=dict)
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
    _next_proc: int = 30000
    _next_med: int = 40000
    _next_readm: int = 50000
    _next_death: int = 60000
 
    def __init__(self, df: pd.DataFrame, token_converter: Optional[TokenConverter] = None):
        self.token_converter = token_converter or TokenConverter()
 
        self.special_vocab = {}
        self.admission_vocab = {}
        self.diagnosis_vocab = {}
        self.procedure_vocab = {}
        self.medication_vocab = {}
        self.readmission_vocab = {}
        self.death_vocab = {}
 
        self._init_special_tokens()
        self.build_from_dataframe(df)
 
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
            event = int(row["event_type"])
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
 
        elif event == EventType.PROCEDURE:
            new_id = self._next_proc
            if new_id > 39999:
                raise RuntimeError("Procedure vocab exceeded 30000..39999 range.")
            vocab[token] = new_id
            self._next_proc += 1
 
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
        if event == EventType.PROCEDURE:
            return self.procedure_vocab
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
        if token in self.procedure_vocab:
            return self.procedure_vocab[token]
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
 
if __name__ == "__main__":
    import pandas as pd
 
    # -------------------------
    # Example data (toy)
    # -------------------------
    df = pd.DataFrame(
        [
            {"event_type": 0},
            {"event_type": 1, "icd_code": "I10", "icd_version": 10},
            {"event_type": 1, "icd_code": "E11", "icd_version": 10},
            {"event_type": 2, "icd_code": "5491", "icd_version": 9},
            {"event_type": 3, "formulary_drug_cd": "LISINOPRIL"},
            {"event_type": 4},
            {"event_type": 5},
        ]
    )
 
    vocab = Vocabulary(df)
 
    # -------------------------
    # 1) Print vocabularies
    # -------------------------
    print("\n=== SPECIAL VOCAB ===")
    print(vocab.special_vocab)
 
    print("\n=== ADMISSION VOCAB ===")
    print(vocab.admission_vocab)
 
    print("\n=== DIAGNOSIS VOCAB ===")
    print(vocab.diagnosis_vocab)
 
    print("\n=== PROCEDURE VOCAB ===")
    print(vocab.procedure_vocab)
 
    print("\n=== MEDICATION VOCAB ===")
    print(vocab.medication_vocab)
 
    print("\n=== READMISSION VOCAB ===")
    print(vocab.readmission_vocab)
 
    print("\n=== DEATH VOCAB ===")
    print(vocab.death_vocab)
 
    # -------------------------
    # 2) From another row -> ID (token -> id)
    # -------------------------
    row_for_id = df.iloc[4]  # MED LISINOPRIL
    token = vocab.row_to_token(row_for_id)
    token_id = vocab.token_to_id(token)
    print("\nRow -> Token -> ID")
    print("Row:", row_for_id.to_dict())
    print("Token:", token)
    print("ID:", token_id)
 
    # -------------------------
    # 3) Unseen example -> UNK token and UNK id
    # -------------------------
    unseen = pd.Series({"event_type": 3, "formulary_drug_cd": "NEW_DRUG"})
    unseen_token = vocab.row_to_token(unseen)
    unseen_id = vocab.token_to_id(unseen_token)
    print("\nUnseen row -> Token -> ID")
    print("Row:", dict(unseen))
    print("Token:", unseen_token)
    print("ID:", unseen_id)