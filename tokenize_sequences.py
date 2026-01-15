import json
from typing import List, Any, Tuple, Optional
from pathlib import Path


class TokenSequencer:
    """
    Converts nested token sequences into nested ID sequences for our transformer.

    Expected token input format (per patient):
      [
        [demo_tokens],   # e.g. ["[DEM_GENDER_F]", "[DEM_AGE_52]", ...]
        [event_tokens],  # e.g. ["[ADM]", "[DIAG_9_6820]", "[MED_HEPARIN]", ...]
      ]

    Output ID format (per patient):
      [
        [demo_ids],
        [event_ids],
      ]

    NOTE:
    - No padding here!
    - Padding/masks/segment_ids are created later via build_from_joint_format(...)
    """

    def tokens_to_ids_joint(
        self,
        joint_token_sequences: List[List[List[Any]]],
        vocab,
        *,
        drop_empty: bool = True,
        keep_unk: bool = True,
    ) -> List[List[List[int]]]:
        pad_id = vocab.token_to_id(vocab.get_padding_token())
        unk_id = vocab.token_to_id(vocab.get_unknown_token())

        joint_ids: List[List[List[int]]] = []

        for i, patient in enumerate(joint_token_sequences):
            if not isinstance(patient, list) or len(patient) != 2:
                raise ValueError(
                    f"Invalid patient entry at index {i}. "
                    f"Expected [[demo_tokens],[event_tokens]], got: {patient}"
                )

            demo_tokens, event_tokens = patient

            demo_ids: List[int] = []
            event_ids: List[int] = []

            # --- DEMO ---
            for tok in demo_tokens:
                if tok is None:
                    continue
                s = str(tok).strip()
                if s == "":
                    continue
                tid = vocab.token_to_id(s)
                # vocab.token_to_id returns UNK id in your implementation, but keep it explicit:
                if tid == unk_id and not keep_unk:
                    continue
                # never carry PAD from upstream
                if tid == pad_id:
                    continue
                demo_ids.append(tid)

            # --- EVENTS ---
            for tok in event_tokens:
                if tok is None:
                    continue
                s = str(tok).strip()
                if s == "":
                    continue
                tid = vocab.token_to_id(s)
                if tid == unk_id and not keep_unk:
                    continue
                if tid == pad_id:
                    continue
                event_ids.append(tid)

            if drop_empty:
                # If events are empty, MLM masking (events-only) can produce n_masked=0
                # You can decide: drop such patients, or keep them and use --mask_demo later.
                if len(event_ids) == 0:
                    continue

            joint_ids.append([demo_ids, event_ids])

        return joint_ids

    def build_sequences(
        self,
        token_sequences: List[List[List[Any]]],
        *,
        vocab_path: Path = Path("../out/vocab/vocabulary.json"),
        out_json: Optional[Path] = Path("../out/joint_ids.json"),
        drop_empty: bool = True,
    ) -> List[List[List[int]]]:
        from vocabulary import Vocabulary

        vocab = Vocabulary.load(vocab_path)

        joint_ids = self.tokens_to_ids_joint(
            token_sequences,
            vocab,
            drop_empty=drop_empty,
        )

        if out_json is not None:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(joint_ids, f, indent=2)

        return joint_ids
