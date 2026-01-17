from __future__ import annotations
from typing import Iterable, Optional, Tuple, Set
import torch


def mlm_mask_801010(
    input_ids: torch.Tensor,              # [B, L] long
    attention_mask: torch.Tensor,         # [B, L] long (1=token, 0=pad) or bool
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
    can_mask = (attention_mask == 1) if attention_mask.dtype != torch.bool else attention_mask.clone()
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


def mlm_mask_span_801010(
    input_ids: torch.Tensor,              # [B, L] long
    attention_mask: torch.Tensor,         # [B, L] long/bool (1=token, 0=pad)
    mask_token_id: int,
    vocab_size: int,
    p_mlm: float = 0.15,
    span_len_range: Tuple[int, int] = (3, 10),
    pad_token_id: Optional[int] = None,
    never_mask_token_ids: Optional[Iterable[int]] = None,
    *,
    generator: Optional[torch.Generator] = None,
    avoid_random_token_ids: Optional[Iterable[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Span MLM:
      - choose span start positions among valid tokens
      - expand to spans (random length in [min_len, max_len])
      - aim for ~p_mlm of valid tokens masked (best-effort)
      - replacements follow BERT 80/10/10 policy
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have same shape")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must be torch.long")
    if not (0.0 <= p_mlm <= 1.0):
        raise ValueError("p_mlm must be in [0, 1]")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    min_len, max_len = span_len_range
    if not (1 <= min_len <= max_len):
        raise ValueError("span_len_range must satisfy 1 <= min_len <= max_len")

    device = input_ids.device
    B, L = input_ids.shape
    masked_input_ids = input_ids.clone()

    can_mask = (attention_mask == 1) if attention_mask.dtype != torch.bool else attention_mask.clone()
    if pad_token_id is not None:
        can_mask &= (input_ids != int(pad_token_id))
    if never_mask_token_ids is not None:
        for tid in never_mask_token_ids:
            can_mask &= (input_ids != int(tid))

    n_valid = int(can_mask.sum().item())
    target = int(round(p_mlm * n_valid))
    if target <= 0:
        labels = torch.full_like(input_ids, -100)
        return masked_input_ids, labels

    mask_positions = torch.zeros((B, L), device=device, dtype=torch.bool)

    for b in range(B):
        valid_idx = torch.where(can_mask[b])[0]
        if valid_idx.numel() == 0:
            continue

        perm = valid_idx[torch.randperm(valid_idx.numel(), generator=generator, device=device)]
        masked_count = 0

        for start in perm:
            if masked_count >= target:
                break

            span_len = int(torch.randint(min_len, max_len + 1, (1,), generator=generator, device=device).item())
            s = int(start.item())
            e = min(L, s + span_len)

            span_valid = can_mask[b, s:e]
            add = int(span_valid.sum().item())
            if add == 0:
                continue

            mask_positions[b, s:e] |= span_valid
            masked_count += add

    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]

    u = torch.rand((B, L), device=device, generator=generator)
    do_mask = mask_positions & (u < 0.8)
    do_rand = mask_positions & (u >= 0.8) & (u < 0.9)

    masked_input_ids[do_mask] = int(mask_token_id)

    if do_rand.any():
        forbidden: Set[int] = set()
        if pad_token_id is not None:
            forbidden.add(int(pad_token_id))
        forbidden.add(int(mask_token_id))
        if never_mask_token_ids is not None:
            forbidden.update(int(x) for x in never_mask_token_ids)
        if avoid_random_token_ids is not None:
            forbidden.update(int(x) for x in avoid_random_token_ids)

        if len(forbidden) >= vocab_size:
            raise ValueError("forbidden ids cover entire vocab")

        n = int(do_rand.sum().item())
        random_tokens = torch.empty((n,), device=device, dtype=input_ids.dtype)
        filled = 0
        while filled < n:
            k = (n - filled) * 2
            cand = torch.randint(0, vocab_size, (k,), device=device, generator=generator, dtype=input_ids.dtype)
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


def mlm_mask_recency_801010(
    input_ids: torch.Tensor,              # [B, L] long
    attention_mask: torch.Tensor,         # [B, L] long/bool
    mask_token_id: int,
    vocab_size: int,
    p_mlm: float = 0.15,
    recency_power: float = 2.0,           # higher => more end-focused
    pad_token_id: Optional[int] = None,
    never_mask_token_ids: Optional[Iterable[int]] = None,
    *,
    generator: Optional[torch.Generator] = None,
    avoid_random_token_ids: Optional[Iterable[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recency-biased MLM:
      - later positions have higher mask probability
      - expected overall mask rate approximately p_mlm (best-effort)
      - replacements follow BERT 80/10/10 policy
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have same shape")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must be torch.long")
    if not (0.0 <= p_mlm <= 1.0):
        raise ValueError("p_mlm must be in [0, 1]")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    device = input_ids.device
    B, L = input_ids.shape
    masked_input_ids = input_ids.clone()

    can_mask = (attention_mask == 1) if attention_mask.dtype != torch.bool else attention_mask.clone()
    if pad_token_id is not None:
        can_mask &= (input_ids != int(pad_token_id))
    if never_mask_token_ids is not None:
        for tid in never_mask_token_ids:
            can_mask &= (input_ids != int(tid))

    pos = torch.arange(L, device=device).float()
    w = ((pos + 1.0) / float(L)) ** float(recency_power)  # (0,1]
    w = w / w.mean().clamp(min=1e-8)                      # mean ~ 1

    rand = torch.rand((B, L), device=device, generator=generator)
    mask_positions = (rand < (p_mlm * w.unsqueeze(0))) & can_mask

    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]

    u = torch.rand((B, L), device=device, generator=generator)
    do_mask = mask_positions & (u < 0.8)
    do_rand = mask_positions & (u >= 0.8) & (u < 0.9)

    masked_input_ids[do_mask] = int(mask_token_id)

    if do_rand.any():
        forbidden: Set[int] = set()
        if pad_token_id is not None:
            forbidden.add(int(pad_token_id))
        forbidden.add(int(mask_token_id))
        if never_mask_token_ids is not None:
            forbidden.update(int(x) for x in never_mask_token_ids)
        if avoid_random_token_ids is not None:
            forbidden.update(int(x) for x in avoid_random_token_ids)

        if len(forbidden) >= vocab_size:
            raise ValueError("forbidden ids cover entire vocab")

        n = int(do_rand.sum().item())
        random_tokens = torch.empty((n,), device=device, dtype=input_ids.dtype)
        filled = 0
        while filled < n:
            k = (n - filled) * 2
            cand = torch.randint(0, vocab_size, (k,), device=device, generator=generator, dtype=input_ids.dtype)
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
