from __future__ import annotations

import os
import re
import torch

RAW_DIR = "checkpoints_raw"
OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "mlm_baseline.pt": {"ckpt": "train_log_20260131_183112_baseline.pth"},
    "mlm_n_layer_6.pt": {"ckpt": "train_log_20260131_220938_n_layers_6.pth", "n_layers": 6},
    "mlm_n_heads_12.pt": {"ckpt": "train_log_20260201_015354_n_heads_12.pth", "n_heads": 12},
    "mlm_n_heads_12_n_layer_6.pt": {"ckpt": "train_log_20260201_051614_n_heads_12_n_layers_6.pth", "n_heads": 12, "n_layers": 6},
    "mlm_n_event_types_7.pt": {"ckpt": "train_log_20260201_091330_n_event_types_7.pth"},
    "mlm_max_len_384.pt": {"ckpt": "train_log_20260201_151819_max_len_384.pth"},
    "mlm_d_model_384.pt": {"ckpt": "train_log_20260201_195853_d_model_384.pth"},
    "mlm_lr_decay.pt": {"ckpt": "train_log_20260203_125439_lr_decay.pth"},
    "mlm_span.pt": {"ckpt": "train_log_20260203_160942_mask_mode_span.pth"},
    "mlm_recency.pt": {"ckpt": "train_log_20260203_235424_mask_mode_recency.pth"},
    "mlm_p_mlm_0.1.pt": {"ckpt": "train_log_20260204_030943_p_mlm_0.1.pth"},
    "mlm_p_mlm_0.2.pt": {"ckpt": "train_log_20260204_062431_p_mlm_0.2.pth"},
    "mlm_dropout_15.pt": {"ckpt": "train_log_20260204_111524_dropout_15.pth", "dropout": 0.15},
    "mlm_dropout_05.pt": {"ckpt": "train_log_20260204_143210_dropout_5.pth", "dropout": 0.05},
    "mlm_rms_norm_true_norm_first_true.pt": {"ckpt": "train_log_20260204_181630_rms_norm_true_norm_first_true.pth", "rms_norm": True, "norm_first": True},
    "mlm_rms_norm_true.pt": {"ckpt": "train_log_20260205_121849_rms_norm.pth", "rms_norm": True},
    "mlm_norm_first_true.pt": {"ckpt": "train_log_20260205_153846_norm_first_true.pth", "norm_first": True},
    "mlm_activation_silu.pt": {"ckpt": "train_log_20260205_185907_activation_silu.pth", "activation": "silu"},
    "mlm_activation_relu.pt": {"ckpt": "train_log_20260205_221730_activation_relu.pth", "activation": "relu"},
    "mlm_big_1.pt": {"ckpt": "train_log_20260213_154758_big-1.pth", "n_layers": 10, "d_model": 768, "n_heads": 12, "norm_first": True},
    "mlm_big_1_fixed.pt": {"ckpt": "train_log_20260217_195657_big-1.pth", "n_layers": 10, "d_model": 768, "n_heads": 12, "norm_first": True},
    "mlm_big_2.pt": {"ckpt": "train_log_20260219_171035_big-2.pth", "n_layers": 10, "d_model": 768, "n_heads": 12, "norm_first": True},
    "mlm_big_3.pt": {"ckpt": "train_log_20260219_171127_big-3.pth", "n_layers": 10, "d_model": 768, "n_heads": 12, "norm_first": True},
}


def infer_cfg(sd: dict, *, meta: dict) -> dict:
    if "token_emb.weight" not in sd:
        raise KeyError("Missing key in state_dict: token_emb.weight")
    if "pos_emb.weight" not in sd:
        raise KeyError("Missing key in state_dict: pos_emb.weight")
    if "encoder.layers.0.linear1.weight" not in sd:
        raise KeyError("Missing key in state_dict: encoder.layers.0.linear1.weight")

    n_heads = meta.get("n_heads", 6)
    n_layers_expected = meta.get("n_layers", None)
    dropout = meta.get("dropout", 0.1)
    activation = meta.get("activation", "gelu")
    norm_first = meta.get("norm_first", False)
    rms_norm = meta.get("rms_norm", False)

    vocab_size, d_model = sd["token_emb.weight"].shape
    max_len = sd["pos_emb.weight"].shape[0]
    d_ff = sd["encoder.layers.0.linear1.weight"].shape[0]

    layer_ids = set()
    for k in sd.keys():
        m = re.search(r"encoder\.layers\.(\d+)\.", k)
        if m:
            layer_ids.add(int(m.group(1)))
    n_layers_inferred = (max(layer_ids) + 1) if layer_ids else 0
    if n_layers_inferred <= 0:
        raise ValueError("Could not infer n_layers from state_dict keys (no encoder.layers.* found)")

    if n_layers_expected is not None and n_layers_inferred != int(n_layers_expected):
        raise ValueError(f"n_layers mismatch: inferred={n_layers_inferred} expected={n_layers_expected}")

    use_event = "event_type_emb.weight" in sd
    n_event_types = int(sd["event_type_emb.weight"].shape[0]) if use_event else 0

    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")

    return {
        "vocab_size": int(vocab_size),
        "max_len": int(max_len),
        "d_model": int(d_model),
        "n_layers": int(n_layers_inferred),
        "n_heads": int(n_heads),
        "d_ff": int(d_ff),
        "dropout": float(dropout),
        "layer_norm_eps": 1e-5,
        "use_event_type_embeddings": bool(use_event),
        "n_event_types": int(n_event_types),
        "pad_token_id": 0,
        "mask_token_id": 1,
        "activation": str(activation),
        "norm_first": bool(norm_first),
        "rms_norm": bool(rms_norm),
    }


def main() -> None:
    for out_name, meta in FILES.items():
        if "ckpt" not in meta:
            raise KeyError(f"FILES[{out_name}] missing required key: 'ckpt'")

    ok = 0
    skipped = 0

    for out_name, meta in FILES.items():
        in_path = os.path.join(RAW_DIR, meta["ckpt"])
        out_path = os.path.join(OUT_DIR, out_name)

        if not os.path.exists(in_path):
            print(f"[SKIP] missing: {in_path}")
            skipped += 1
            continue

        sd = torch.load(in_path, map_location="cpu")
        if not isinstance(sd, dict):
            raise TypeError(f"Expected a state_dict dict in {in_path}, got {type(sd)}")

        cfg = infer_cfg(sd, meta=meta)
        torch.save({"cfg": cfg, "model_state_dict": sd}, out_path)

        ok += 1
        print(f"[OK] {out_path}")

    print(f"[DONE] ok={ok} skipped={skipped} (raw_dir={RAW_DIR}, out_dir={OUT_DIR})")


if __name__ == "__main__":
    main()
