# Minimal test for compact_transformer_encoder modell

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder


if __name__ == "__main__":
    cfg = CompactTransformerConfig(
        vocab_size=70000,
        max_len=256,
        d_model=192,    # 128 or 192 or 256
        n_layers=3,
        n_heads=6,      # 4 or 6 or 8
        use_event_type_embeddings=True,
        n_event_types=6,
        pad_token_id=0,
        mask_token_id=1,
    )
    model = CompactTransformerEncoder(cfg)

    B, L = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))
    attention_mask = torch.ones((B, L), dtype=torch.long)

    # event types in [0..n_event_types-1]
    event_type_ids = torch.randint(0, cfg.n_event_types, (B, L))

    # MLM labels: -100 except a few masked positions
    labels = torch.full((B, L), -100, dtype=torch.long)
    labels[:, 3] = input_ids[:, 3]
    labels[:, 7] = input_ids[:, 7]
    input_ids[:, 3] = cfg.mask_token_id
    input_ids[:, 7] = cfg.mask_token_id

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        event_type_ids=event_type_ids,
        labels=labels,
    )
    print("-------------------------------------------------------")
    print("logits:", out["logits"].shape)  # [B, L, V]
    print("-------------------------------------------------------")
    print("loss:", out["loss"].detach().item())
    print("-------------------------------------------------------")
    
