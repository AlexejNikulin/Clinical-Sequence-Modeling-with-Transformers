"""
Responsibilities:
- Convert token sequences (strings) into integer ID sequences using a fixed vocabulary
- Handle unknown tokens via UNK_ID
- Pad / truncate sequences to fixed max_len
- Generate attention masks (1 for real tokens, 0 for padding)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union, Optional


TokenSeq = Union[List[str], str]  # allow list of tokens OR "A -> B -> C" string


def _normalize_sequence(seq: TokenSeq) -> List[str]:
    """
    Ensures we always work with List[str].
    Accepts:
      - List[str]
      - String formatted like: "[ADM] -> [DIAG_I10] -> [DEATH]"
    """
    if seq is None:
        return []

    # If it's already a list, clean it
    if isinstance(seq, list):
        return [str(t).strip() for t in seq if t is not None and str(t).strip() != ""]

    # If it's a string, split by '->' (or fallback whitespace)
    if isinstance(seq, str):
        s = seq.strip()
        if s == "":
            return []
        if "->" in s:
            parts = [p.strip() for p in s.split("->")]
            return [p for p in parts if p]
        # fallback: whitespace split
        parts = s.split()
        return [p for p in parts if p]

    # fallback: stringify
    return [str(seq).strip()] if str(seq).strip() != "" else []


def tokenize_and_pad(
    sequences: Sequence[TokenSeq],
    vocab: Dict[str, int],
    max_len: int = 512,
    pad_id: int = 0,
    unk_id: int = 1,
    truncation: str = "keep_last",  # "keep_last" or "keep_first"
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Convert sequences to padded/clipped transformer input.

    Args:
        sequences: list of sequences; each can be List[str] or a "->" string
        vocab: token -> id mapping
        max_len: final sequence length
        pad_id: padding token id
        unk_id: unknown token id
        truncation: if too long:
            - "keep_last": keep last max_len tokens (recommended for outcomes)
            - "keep_first": keep first max_len tokens

    Returns:
        input_ids: List[List[int]] shape [N, max_len]
        attention_masks: List[List[int]] shape [N, max_len]
    """
    if max_len <= 0:
        raise ValueError("max_len must be > 0")

    if truncation not in ("keep_last", "keep_first"):
        raise ValueError("truncation must be 'keep_last' or 'keep_first'")

    input_ids: List[List[int]] = []
    attention_masks: List[List[int]] = []

    for seq in sequences:
        tokens = _normalize_sequence(seq)

        # Token -> ID
        ids = [vocab.get(tok, unk_id) for tok in tokens]

        # Clip
        if len(ids) > max_len:
            if truncation == "keep_first":
                ids = ids[:max_len]
            else:
                ids = ids[-max_len:]  # keep_last

        # Mask before padding
        mask = [1] * len(ids)

        # Pad
        if len(ids) < max_len:
            pad_n = max_len - len(ids)
            ids = ids + [pad_id] * pad_n
            mask = mask + [0] * pad_n

        input_ids.append(ids)
        attention_masks.append(mask)

    return input_ids, attention_masks


def quick_stats(
    input_ids: List[List[int]],
    attention_masks: List[List[int]],
    pad_id: int = 0,
) -> Dict[str, float]:
    """
    Optional helper to sanity-check outputs.
    """
    n = len(input_ids)
    if n == 0:
        return {"n": 0, "avg_len": 0.0, "pad_rate": 0.0}

    max_len = len(input_ids[0])
    real_lens = []
    pad_count = 0

    for ids, m in zip(input_ids, attention_masks):
        real_len = sum(m)
        real_lens.append(real_len)
        pad_count += sum(1 for x in ids if x == pad_id)

    total_tokens = n * max_len
    return {
        "n": float(n),
        "avg_len": float(sum(real_lens) / n),
        "pad_rate": float(pad_count / total_tokens),
    }


if __name__ == "__main__":
    # Simple local test
    sequences = [
        ["[ADM]", "[DIAG_I10]", "[MED_LISINOPRIL]", "[READM]"],
        "[ADM] -> [PROC9_5491] -> [DEATH]",   # also works (string with arrows)
        ["[ADM]", "[UNKNOWN_TOKEN]"],
    ]

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[ADM]": 10,
        "[READM]": 11,
        "[DEATH]": 12,
        "[DIAG_I10]": 20001,
        "[PROC9_5491]": 30001,
        "[MED_LISINOPRIL]": 40001,
    }

    input_ids, masks = tokenize_and_pad(
        sequences=sequences,
        vocab=vocab,
        max_len=8,
        pad_id=vocab["[PAD]"],
        unk_id=vocab["[UNK]"],
        truncation="keep_last",
    )

    print("Input IDs:")
    for row in input_ids:
        print(row)

    print("\nAttention Masks:")
    for row in masks:
        print(row)

    print("\nStats:", quick_stats(input_ids, masks, pad_id=vocab["[PAD]"]))
