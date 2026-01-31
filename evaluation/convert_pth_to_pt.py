import os
import re
import torch

RAW_DIR = "checkpoints_raw"
OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

# >>> HIER EINMAL KORREKT SETZEN (aus Training!) <<<
N_HEADS = 6   # z.B. 6 / 8 / 12 / 16 – MUSS dem Training entsprechen

FILES = {
    "mlm_d384.pt": "train_log_20260121_001420_d_model_384.pth",
    "mlm_span.pt": "train_log_20260121_230427_mlm_span.pth",
    "mlm_recency.pt": "train_log_20260122_055311_mlm_recency.pth",
}

def infer_cfg(sd: dict) -> dict:
    vocab_size, d_model = sd["token_emb.weight"].shape
    max_len = sd["pos_emb.weight"].shape[0]
    d_ff = sd["encoder.layers.0.linear1.weight"].shape[0]

    layer_ids = set()
    for k in sd.keys():
        m = re.search(r"encoder\.layers\.(\d+)\.", k)
        if m:
            layer_ids.add(int(m.group(1)))
    n_layers = max(layer_ids) + 1

    use_event = "event_type_emb.weight" in sd
    n_event_types = sd["event_type_emb.weight"].shape[0] if use_event else 0

    return {
        "vocab_size": int(vocab_size),
        "max_len": int(max_len),
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "n_heads": int(N_HEADS),
        "d_ff": int(d_ff),
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "use_event_type_embeddings": bool(use_event),
        "n_event_types": int(n_event_types),
        "pad_token_id": 0,
        "mask_token_id": 1,
    }

for out_name, raw_name in FILES.items():
    in_path = os.path.join(RAW_DIR, raw_name)
    out_path = os.path.join(OUT_DIR, out_name)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing raw checkpoint: {in_path}")

    sd = torch.load(in_path, map_location="cpu")
    cfg = infer_cfg(sd)

    torch.save(
        {"cfg": cfg, "model_state_dict": sd},
        out_path
    )

    print(f"[OK] {out_path}")
    print(f"cfg: d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, n_heads={cfg['n_heads']}")
