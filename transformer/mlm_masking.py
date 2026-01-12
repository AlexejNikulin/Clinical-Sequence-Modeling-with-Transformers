# transformer/mlm_masking.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple
import torch


def mlm_mask_801010(
    input_ids: torch.Tensor,              # [B, L] long
    attention_mask: torch.Tensor,         # [B, L] long (1=token, 0=pad)
    mask_token_id: int,
    vocab_size: int,
    p_mlm: float = 0.15,
    pad_token_id: Optional[int] = None,
    never_mask_token_ids: Optional[Iterable[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BERT-style MLM:
      - select ~p_mlm of valid tokens (non-PAD)
      - for selected tokens:
          80% -> replace with [MASK]
          10% -> replace with random token id
          10% -> keep original token
      - labels = original token where selected, else -100

    Returns:
      masked_input_ids: [B, L]
      labels: [B, L]
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have same shape")

    device = input_ids.device
    masked_input_ids = input_ids.clone()

    # Valid positions: attention_mask==1 and (optionally) not PAD id
    can_mask = (attention_mask == 1)
    if pad_token_id is not None:
        can_mask &= (input_ids != pad_token_id)

    # Exclude special tokens if provided
    if never_mask_token_ids is not None:
        for tid in never_mask_token_ids:
            can_mask &= (input_ids != int(tid))

    # Choose which positions to predict
    rand = torch.rand(input_ids.shape, device=device)
    mask_positions = (rand < p_mlm) & can_mask

    # labels: only where we predict
    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]

    # Decide replacement type for masked positions
    u = torch.rand(input_ids.shape, device=device)

    # 80% -> [MASK]
    do_mask = mask_positions & (u < 0.8)

    # 10% -> random token
    do_rand = mask_positions & (u >= 0.8) & (u < 0.9)

    # 10% -> keep original (nothing to do)

    masked_input_ids[do_mask] = mask_token_id

    if do_rand.any():
        random_tokens = torch.randint(
            low=0, high=vocab_size,
            size=(do_rand.sum().item(),),
            device=device, dtype=input_ids.dtype
        )
        masked_input_ids[do_rand] = random_tokens

    return masked_input_ids, labels
