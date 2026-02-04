import os
import re
import torch

RAW_DIR = "checkpoints_raw"
OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "mlm_baseline.pt": {
        "ckpt": "train_log_20260131_183112_baseline.pth",
        "n_heads": 6,
        "n_layers": 3,
    },
    "mlm_n_layer_6.pt": {
        "ckpt": "train_log_20260131_220938_n_layers_6.pth",
        "n_heads": 6,
        "n_layers": 6,
    },
    "mlm_n_heads_12.pt": {
        "ckpt": "train_log_20260201_015354_n_heads_12.pth",
        "n_heads": 12,
        "n_layers": 3,
    },
    "mlm_n_heads_12_n_layer_6.pt": {
        "ckpt": "train_log_20260201_051614_n_heads_12_n_layers_6.pth",
        "n_heads": 12,
        "n_layers": 6,
    },
    "mlm_n_event_types_7.pt": {
        "ckpt": "train_log_20260201_091330_n_event_types_7.pth",
        "n_heads": 6,
        "n_layers": 3,
    },
    "mlm_span.pt": {
        "ckpt": "train_log_20260121_230427_mlm_span.pth",
        "n_heads": 6,
        "n_layers": 3,
    },
    "mlm_recency.pt": {
        "ckpt": "train_log_20260122_055311_mlm_recency.pth",
        "n_heads": 6,
        "n_layers": 3,
    },
}


def infer_cfg(sd: dict, *, n_heads: int, n_layers_expected: int | None = None) -> dict:
    vocab_size, d_model = sd["token_emb.weight"].shape
    max_len = sd["pos_emb.weight"].shape[0]
    d_ff = sd["encoder.layers.0.linear1.weight"].shape[0]

    # infer n_layers from keys (robust)
    layer_ids = set()
    for k in sd.keys():
        m = re.search(r"encoder\.layers\.(\d+)\.", k)
        if m:
            layer_ids.add(int(m.group(1)))
    n_layers_inferred = (max(layer_ids) + 1) if layer_ids else 0

    # optional sanity check against expected
    if n_layers_expected is not None and n_layers_inferred != n_layers_expected:
        raise ValueError(
            f"n_layers mismatch: inferred={n_layers_inferred} expected={n_layers_expected}"
        )

    use_event = "event_type_emb.weight" in sd
    n_event_types = int(sd["event_type_emb.weight"].shape[0]) if use_event else 0

    # sanity check: d_model must be divisible by n_heads
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")

    return {
        "vocab_size": int(vocab_size),
        "max_len": int(max_len),
        "d_model": int(d_model),
        "n_layers": int(n_layers_inferred),
        "n_heads": int(n_heads),
        "d_ff": int(d_ff),
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "use_event_type_embeddings": bool(use_event),
        "n_event_types": int(n_event_types),
        "pad_token_id": 0,
        "mask_token_id": 1,
    }


for out_name, meta in FILES.items():
    in_path = os.path.join(RAW_DIR, meta["ckpt"])
    out_path = os.path.join(OUT_DIR, out_name)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing raw checkpoint: {in_path}")

    sd = torch.load(in_path, map_location="cpu")
    cfg = infer_cfg(sd, n_heads=meta["n_heads"], n_layers_expected=meta["n_layers"])

    torch.save({"cfg": cfg, "model_state_dict": sd}, out_path)

    print(f"[OK] {out_path}")
    print(
        f"cfg: d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, n_heads={cfg['n_heads']}, "
        f"use_event={cfg['use_event_type_embeddings']}, n_event_types={cfg['n_event_types']}"
    )
