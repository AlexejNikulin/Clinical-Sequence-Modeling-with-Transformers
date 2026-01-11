# transformer / compact_transformer_encoder as model

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompactTransformerConfig:
    vocab_size: int
    max_len: int = 512          # could be 256

    # Compact encoder settings
    d_model: int = 128         # we can chenge 192 or 256
    n_layers: int = 3          # 2–4 is our target range
    n_heads: int = 4          # "few heads" (must divide d_model) ---> we can chenge for d_model 192-> 6 and for 256-> 4
    d_ff: Optional[int] = None # if None -> 4*d_model
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Optional heterogeneity signal
    use_event_type_embeddings: bool = True
    n_event_types: int = 6     # adjust to EventType count

    # Special token ids (use our Vocabulary IDs)
    pad_token_id: int = 0
    mask_token_id: int = 1


class CompactTransformerEncoder(nn.Module):
    """
    Compact Transformer *encoder* for MLM. 
    Input:
      - input_ids:        [B, L]
      - attention_mask:   [B, L] (1=token, 0=pad)
      - event_type_ids:   [B, L] optional
      - labels:           [B, L] optional, use -100 for ignore positions

    Output:
      - logits: [B, L, vocab_size]
      - loss:   scalar (if labels provided)
    """

    def __init__(self, cfg: CompactTransformerConfig):
        super().__init__()

        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})."
            )
        if cfg.d_ff is None:
            cfg.d_ff = 4 * cfg.d_model

        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        self.event_type_emb = None
        if cfg.use_event_type_embeddings:
            self.event_type_emb = nn.Embedding(cfg.n_event_types, cfg.d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",           # we can switch to "SiLU", but the best default for MLM and what matches ETHOS is "GELU"!
            batch_first=True,             # keeps tensors [B, L, D]
            norm_first=False,              # typically improves stability, but can be True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.final_ln = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Optional: tie weights with token embedding for parameter efficiency
        self.lm_head.weight = self.token_emb.weight

        self.dropout = nn.Dropout(cfg.dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Conservative init: embedding ~ N(0, 0.02), linear uses default kaiming/xavier in PyTorch.
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        if self.event_type_emb is not None:
            nn.init.normal_(self.event_type_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        event_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns dict with:
          - logits
          - loss (optional)
          - hidden_states (optional)
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")

        B, L = input_ids.shape
        if L > self.cfg.max_len:
            raise ValueError(f"Sequence length L={L} exceeds cfg.max_len={self.cfg.max_len}")

        if attention_mask is None:
            attention_mask = torch.ones((B, L), device=input_ids.device, dtype=torch.long)

        # Build embeddings
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)

        if self.event_type_emb is not None:
            if event_type_ids is None:
                raise ValueError("cfg.use_event_type_embeddings=True but event_type_ids is None")
            x = x + self.event_type_emb(event_type_ids)

        x = self.dropout(x)

        # PyTorch TransformerEncoder uses src_key_padding_mask with True for PAD positions
        key_padding_mask = (attention_mask == 0)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.final_ln(x)

        logits = self.lm_head(x)  # [B, L, V]

        out: Dict[str, Any] = {"logits": logits}

        if labels is not None:
            # labels: [B, L], ignore_index=-100 for non-MLM positions
            if labels.shape != (B, L):
                raise ValueError(f"labels must be [B, L]={B,L}, got {tuple(labels.shape)}")

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        if return_hidden:
            out["hidden_states"] = x

        return out


