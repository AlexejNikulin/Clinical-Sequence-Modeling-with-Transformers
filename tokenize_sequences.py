"""
Tokenize sequences and pad/clip for transformer input.

Adapted to project Vocabulary API:
- vocab.token_to_id(token)
- vocab.get_padding_token()
- vocab.get_unknown_token()
"""

from typing import List, Tuple, Any
from pathlib import Path


def tokenize_and_pad_from_tokens(
    sequences: List[List[str]],
    vocab,
    max_len: int = 512,
    keep: str = "last",  # "last" or "first"
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Use this if sequences are already provided as token strings, e.g.:
    ["[ADM]", "[DIAG_I10]", "[MED_LISINOPRIL]", ...]

    Returns:
      input_ids: List[List[int]] shape [N, max_len]
      attention_masks: List[List[int]] shape [N, max_len]
    """
    pad_id = vocab.token_to_id(vocab.get_padding_token())
    unk_id = vocab.token_to_id(vocab.get_unknown_token())

    input_ids: List[List[int]] = []
    attention_masks: List[List[int]] = []

    for seq in sequences:
        ids: List[int] = []

        for tok in seq:
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

        input_ids.append(ids)
        attention_masks.append(mask)

    return input_ids, attention_masks


def tokenize_and_pad_from_rows(
    sequences: List[List[Any]],
    vocab,
    max_len: int = 512,
    keep: str = "last",  # "last" or "first"
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Use this if sequences are provided as event rows (e.g. pd.Series),
    and tokens are generated via vocab.row_to_token(row).
    """
    pad_id = vocab.token_to_id(vocab.get_padding_token())
    unk_id = vocab.token_to_id(vocab.get_unknown_token())

    input_ids: List[List[int]] = []
    attention_masks: List[List[int]] = []

    for seq_rows in sequences:
        ids: List[int] = []

        for row in seq_rows:
            token = vocab.row_to_token(row)
            token_id = vocab.token_to_id(token)
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

        input_ids.append(ids)
        attention_masks.append(mask)

    return input_ids, attention_masks


if __name__ == "__main__":
    # Minimal smoke test
    import pandas as pd
    from vocabulary import Vocabulary

    # Build a toy dataframe for vocabulary construction
    # df = pd.DataFrame(
    #     [
    #         {"event_type": 0},
    #         {"event_type": 1, "event_value": "I10"},
    #         {"event_type": 1, "event_value": "E11"},
    #         {"event_type": 2, "event_value": "5491"},
    #         {"event_type": 3, "event_value": "LISINOPRIL"},
    #         {"event_type": 4},
    #         {"event_type": 5},
    #     ]
    # )

    VOCAB_PATH = Path("../out/vocab/vocabulary.json")
    vocab = Vocabulary.load(VOCAB_PATH)

    # ---- Test A: token-based sequences ----
    token_sequences = [
        ["[ADM]", "[DIAG_9_496]", "[MED_RALTEGRAVIR]", "[DEATH]"],
        ["[ADM]", "[DIAG_9_07071]", "[READM]"],
        ["[ADM]", "[MED_NEW_DRUG]"],  # should map to UNK
    ]

    ids, masks = tokenize_and_pad_from_tokens(token_sequences, vocab, max_len=6)
    print("\nToken-based sequences -> input_ids:")
    for row in ids:
        print(row)

    print("\nToken-based sequences -> attention_masks:")
    for row in masks:
        print(row)

    # ---- Test B: row-based sequences ----
    # row_sequences = [
    #     [df.iloc[0], df.iloc[1], df.iloc[4], df.iloc[6]],
    #     [df.iloc[0], df.iloc[2], df.iloc[5]],
    # ]

    # ids2, masks2 = tokenize_and_pad_from_rows(row_sequences, vocab, max_len=6)
    # print("\nRow-based sequences -> input_ids:")
    # for row in ids2:
    #     print(row)

    # print("\nRow-based sequences -> attention_masks:")
    # for row in masks2:
    #     print(row)
