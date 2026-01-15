import json
from typing import List, Tuple, Any
from pathlib import Path

class TokenSequencer:
    """
    Tokenize sequences and pad/clip for transformer input.

    Adapted to project Vocabulary API:
    - vocab.token_to_id(token)
    - vocab.get_padding_token()
    - vocab.get_unknown_token()
    """

    def tokenize_and_pad_from_tokens(
        self,
        sequences: List[List[List[Any], List[Any]]],
        vocab,
        max_len: int = 512,
        keep: str = "last",  # "last" or "first"
    ) -> Tuple[List[List[List[int], List[int]]], List[List[List[int], List[int]]]]:
        """
        Use this if sequences are already provided as token strings, e.g.:
        ["[ADM]", "[DIAG_I10]", "[MED_LISINOPRIL]", ...]

        Returns:
        input_ids: List[List[int]] shape [N, max_len]
        attention_masks: List[List[int]] shape [N, max_len]
        """
        pad_id = vocab.token_to_id(vocab.get_padding_token())
        unk_id = vocab.token_to_id(vocab.get_unknown_token())

        input_ids: List[List[List[int], List[int]]] = []
        attention_masks: List[List[List[int], List[int]]] = []

        for seq in sequences:
            event_ids: List[List[int]] = []

            for tokens in seq:
                ids: List[int] = []

                for tok in tokens:
                    if tok is None or str(tok).strip() == "":
                        continue
                    token_str = str(tok).strip()
                    token_id = vocab.token_to_id(token_str) if token_str else unk_id
                    ids.append(token_id)

                # Clip
                if len(ids) > max_len:
                    ids = ids[-max_len:] if keep == "last" else ids[:max_len]

                # Attention mask (before padding)
                mask = [1] * len(ids)

                # Pad
                while len(ids) < max_len:
                    ids.append(pad_id)
                    mask.append(0)

                event_ids.append(ids)
                attention_masks.append(mask) 
            input_ids.append(event_ids)

        return input_ids, attention_masks

    def build_sequences(self, token_sequences):
        import pandas as pd
        from vocabulary import Vocabulary

        VOCAB_PATH = Path("../out/vocab/vocabulary.json")
        vocab = Vocabulary.load(VOCAB_PATH)

        # token_sequences = [
        # [["[DEM_GENDER_F]", "[DEM_YEARGRP_2014_2016]", "[DEM_AGE_52]"], ["[ADM]", "[DIAG_9_6820]", "[MED_HEPARIN]", "[DEATH]"]],
        # [["[DEM_AGE_34]", "[DEM_YEARGRP_2017_2019]", "[DEM_AGE_74]"], ["[ADM]", "[DIAG_9_4019]", "[READM]"]]
        # ]

        ids, masks = self.tokenize_and_pad_from_tokens(token_sequences, vocab)

        path = Path("../out/ids.json")
        data = ids
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # print("\nToken-based sequences -> input_ids:")
        # for row in ids:
        #     print(row)

        # print("\nToken-based sequences -> attention_masks:")
        # for row in masks:
        #     print(row)
