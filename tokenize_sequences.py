"""
Tokenize sequences and pad/clip for transformer input.
Author: Hamid
"""

def tokenize_and_pad(
    sequences,
    vocab,
    max_len=512,
    pad_id=0,
    unk_id=1,
):
    """
    sequences: List[List[str]]
    vocab: Dict[str, int]
    """
    input_ids = []
    attention_masks = []

    for seq in sequences:
        # Tokenize (string -> id)
        ids = [
            vocab.get(tok, unk_id)
            for tok in seq
            if tok is not None and str(tok).strip() != ""
        ]

        # Clip to max_len (keep last tokens)
        ids = ids[-max_len:]

        # Attention mask before padding
        mask = [1] * len(ids)

        # Pad to max_len
        while len(ids) < max_len:
            ids.append(pad_id)
            mask.append(0)

        input_ids.append(ids)
        attention_masks.append(mask)

    return input_ids, attention_masks
