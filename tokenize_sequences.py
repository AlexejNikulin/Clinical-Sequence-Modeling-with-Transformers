"""
Tokenize sequences and pad/clip for transformer input.

Adapted to project Vocabulary API:
- vocab.token_to_id(token)
- vocab.get_padding_token()
- vocab.get_unknown_token()
"""

from typing import List, Tuple, Any

def tokenize_and_pad_from_tokens(
    sequences: List[List[str]],
    vocab,
    max_len: int = 512,
    keep: str = "last",  # "last" or "first"
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Use this if Hauke already provides token strings, e.g.:
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
            # token_to_id already falls back to UNK id if unknown in your implementation
            # but we keep it explicit-safe:
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
    Use this if you receive sequences as rows/events (e.g. pd.Series dict-like),
    and you want to convert row -> token via vocab.row_to_token(row).

    Example row must contain at least:
      - event_type
      - icd_code, icd_version (for diag/proc)
      - formulary_drug_cd (for med)
    """
    pad_id = vocab.token_to_id(vocab.get_padding_token())
    unk_id = vocab.token_to_id(vocab.get_unknown_token())

    input_ids: List[List[int]] = []
    attention_masks: List[List[int]] = []

    for seq_rows in sequences:
        ids: List[int] = []
        for row in seq_rows:
            token = vocab.row_to_token(row)  # returns token or global [UNK]
            token_id = vocab.token_to_id(token)  # maps to int (UNK if unknown)
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
    # Minimal smoke test with the Vocabulary class you posted
    import pandas as pd

    # Build a toy dataframe for vocab construction
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

    # Import your Vocabulary from wherever it lives in your project.
    # If it's in vocabulary.py in the same folder, you would do:
    # from vocabulary import Vocabulary
    # For this snippet, assume Vocabulary is available in scope.
    from vocabulary import Vocabulary  # adjust path/name to your repo

    vocab = Vocabulary(df)

    # ---- Test A: already-tokenized sequences (strings) ----
    token_sequences = [
        ["[ADM]", "[DIAG_I10]", "[MED_LISINOPRIL]", "[DEATH]"],
        ["[ADM]", "[DIAG_E11]", "[READM]"],
        ["[ADM]", "[MED_NEW_DRUG]"],  # should become UNK for the med token
    ]

    ids, masks = tokenize_and_pad_from_tokens(token_sequences, vocab, max_len=6, keep="last")
    print("\nToken sequences -> input_ids:")
    for row in ids:
        print(row)
    print("\nToken sequences -> attention_masks:")
    for row in masks:
        print(row)

    # ---- Test B: row-based sequences (events) ----
    row_sequences = [
        [df.iloc[0], df.iloc[1], df.iloc[4], df.iloc[6]],  # ADM, DIAG I10, MED LISINOPRIL, DEATH
        [df.iloc[0], df.iloc[2], df.iloc[5]],              # ADM, DIAG E11, READM
    ]

    ids2, masks2 = tokenize_and_pad_from_rows(row_sequences, vocab, max_len=6, keep="last")
    print("\nRow sequences -> input_ids:")
    for row in ids2:
        print
