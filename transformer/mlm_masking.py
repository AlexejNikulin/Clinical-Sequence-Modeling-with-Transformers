from __future__ import annotations
from typing import Iterable, Optional, Tuple, Set
import torch


def mlm_mask_801010(
    input_ids: torch.Tensor,              # [B, L] long
    attention_mask: torch.Tensor,         # [B, L] long (1=token, 0=pad)
    mask_token_id: int,
    vocab_size: int,
    p_mlm: float = 0.15,
    pad_token_id: Optional[int] = None,
    never_mask_token_ids: Optional[Iterable[int]] = None,
    *,
    generator: Optional[torch.Generator] = None,
    avoid_random_token_ids: Optional[Iterable[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BERT-style MLM:
      - select ~p_mlm of valid tokens (non-PAD)
      - for selected tokens:
          80% -> replace with [MASK]
          10% -> replace with random token id
          10% -> keep original token
      - labels = original token where selected, else -100

    Extras:
      - generator: for deterministic behavior in tests
      - avoid_random_token_ids: token ids that must NOT be sampled as random replacements
        (e.g. [PAD], [MASK], special tokens)
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have same shape")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must be torch.long")
    if attention_mask.dtype not in (torch.long, torch.int64, torch.int32, torch.uint8, torch.bool):
        raise ValueError("attention_mask must be integer/bool-like")
    if not (0.0 <= p_mlm <= 1.0):
        raise ValueError("p_mlm must be in [0, 1]")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    device = input_ids.device
    masked_input_ids = input_ids.clone()

    # Valid positions: attention_mask==1 and (optionally) not PAD id
    can_mask = (attention_mask == 1) if attention_mask.dtype != torch.bool else attention_mask
    if pad_token_id is not None:
        can_mask &= (input_ids != int(pad_token_id))

    # Exclude special tokens if provided (never mask)
    if never_mask_token_ids is not None:
        for tid in never_mask_token_ids:
            can_mask &= (input_ids != int(tid))

    # Choose which positions to predict
    rand = torch.rand(input_ids.shape, device=device, generator=generator)
    mask_positions = (rand < p_mlm) & can_mask

    # labels: only where we predict
    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]

    # Decide replacement type for masked positions
    u = torch.rand(input_ids.shape, device=device, generator=generator)

    # 80% -> [MASK]
    do_mask = mask_positions & (u < 0.8)

    # 10% -> random token
    do_rand = mask_positions & (u >= 0.8) & (u < 0.9)

    masked_input_ids[do_mask] = int(mask_token_id)

    if do_rand.any():
        # Build a "forbidden set" for random replacements
        forbidden: Set[int] = set()
        if pad_token_id is not None:
            forbidden.add(int(pad_token_id))
        forbidden.add(int(mask_token_id))
        if never_mask_token_ids is not None:
            forbidden.update(int(x) for x in never_mask_token_ids)
        if avoid_random_token_ids is not None:
            forbidden.update(int(x) for x in avoid_random_token_ids)

        # If nothing is forbidden, sample directly
        n = int(do_rand.sum().item())
        if len(forbidden) == 0:
            random_tokens = torch.randint(
                low=0, high=vocab_size, size=(n,),
                device=device, dtype=input_ids.dtype, generator=generator
            )
            masked_input_ids[do_rand] = random_tokens
        else:
            # Rejection sampling (safe for small forbidden sets)
            # Note: In worst case (forbidden ~ vocab), can loop; we guard that.
            if len(forbidden) >= vocab_size:
                raise ValueError("avoid_random_token_ids / forbidden ids cover the entire vocab")

            random_tokens = torch.empty((n,), device=device, dtype=input_ids.dtype)
            filled = 0
            while filled < n:
                # Oversample a bit
                k = (n - filled) * 2
                cand = torch.randint(
                    low=0, high=vocab_size, size=(k,),
                    device=device, dtype=input_ids.dtype, generator=generator
                )
                # filter
                keep = torch.ones_like(cand, dtype=torch.bool)
                for f in forbidden:
                    keep &= (cand != f)
                cand = cand[keep]
                take = min(cand.numel(), n - filled)
                if take > 0:
                    random_tokens[filled:filled + take] = cand[:take]
                    filled += take

            masked_input_ids[do_rand] = random_tokens

    return masked_input_ids, labels
