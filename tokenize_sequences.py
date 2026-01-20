import json
from typing import List, Any, Optional
from pathlib import Path
from tqdm import tqdm

class TokenSequencer:
    """
    Converts per-patient token sequences into per-patient ID sequences.

    NEW expected token input format (per patient):
      [
        demo_tokens + event_tokens
      ]

    i.e. one list per patient:
      [
        ["[DEM_GENDER_F]", "[DEM_AGE_52]", ..., "[ADM]", "[DIAG_9_6820]", ...]
      ]

    Output ID format (per patient):
      [
        [ids]
      ]

    NOTE:
    - No padding here!
    - Padding/masks are created later.
    """

    def tokens_to_ids_combined(
        self,
        token_sequences: List[List[List[str]]],
        vocab,
        *,
        drop_empty: bool = True,
        keep_unk: bool = True,
        min_len: int = 1,
    ) -> List[List[List[int]]]:
        pad_id = vocab.token_to_id(vocab.get_padding_token())
        unk_id = vocab.token_to_id(vocab.get_unknown_token())

        all_ids: List[List[int]] = []

        for i, seq in tqdm(enumerate(token_sequences), total=len(token_sequences)):
            if not isinstance(seq, list):
                raise ValueError(
                    f"Invalid patient entry at index {i}. Expected a list of tokens, got: {type(seq)}"
                )

            ids: List[List[int]] = []

            for data_type in seq: # demographic vs. non-demographic data
                local_ids: List[int] = []
                for tok in data_type:
                    if tok is None:
                        continue
                    s = str(tok).strip()
                    if s == "":
                        continue

                    tid = vocab.token_to_id(s)

                    # optionally drop unknowns
                    if tid == unk_id and not keep_unk:
                        continue

                    # never carry PAD from upstream
                    if tid == pad_id:
                        continue

                    local_ids.append(tid)
                ids.append(local_ids)

            if drop_empty:
                if len(ids) < min_len:
                    continue

            all_ids.append(ids)

        return all_ids

    def tokens_to_ids(
        self,
        token_sequences: List[List[List[str]]],
        vocab,
        *,
        drop_empty: bool = True,
        keep_unk: bool = True,
        min_len: int = 1,
    ) -> List[List[List[int]]]:
        pad_id = vocab.token_to_id(vocab.get_padding_token())
        unk_id = vocab.token_to_id(vocab.get_unknown_token())

        all_ids: List[List[int]] = []

        for i, seq in tqdm(enumerate(token_sequences), total=len(token_sequences)):
            if not isinstance(seq, list):
                raise ValueError(
                    f"Invalid patient entry at index {i}. Expected a list of tokens, got: {type(seq)}"
                )

            ids: List[List[int]] = []

            for data_type in seq: # demographic vs. non-demographic data
                local_ids: List[int] = []
                for tok in data_type:
                    if tok is None:
                        continue
                    s = str(tok).strip()
                    if s == "":
                        continue

                    tid = vocab.token_to_id(s)

                    # optionally drop unknowns
                    if tid == unk_id and not keep_unk:
                        continue

                    # never carry PAD from upstream
                    if tid == pad_id:
                        continue

                    local_ids.append(tid)
                ids.append(local_ids)

            if drop_empty:
                if len(ids) < min_len:
                    continue

            all_ids.append(ids)

        return all_ids

    def build_sequences(
        self,
        token_sequences: List[List[List[str]]],
        combine_lists: bool,
        *,
        vocab_path: Path = Path("../out/vocab/vocabulary.json"),
        out_json: Optional[Path] = Path("../out/ids.json"),
        drop_empty: bool = True,
        keep_unk: bool = True,
        min_len: int = 1,
    ) -> List[List[int]]:
        from vocabulary import Vocabulary

        vocab = Vocabulary.load(vocab_path)

        if(combine_lists):
            ids = self.tokens_to_ids_combined(
                token_sequences,
                vocab,
                drop_empty=drop_empty,
                keep_unk=keep_unk,
                min_len=min_len,
            )
        else:
            ids = self.tokens_to_ids(
                token_sequences,
                vocab,
                drop_empty=drop_empty,
                keep_unk=keep_unk,
                min_len=min_len,
            )

        if out_json is not None:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(ids, f, indent=2)

        return ids

    def load_ids_from_json(self, path: Path = Path("../out/ids.json")):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)