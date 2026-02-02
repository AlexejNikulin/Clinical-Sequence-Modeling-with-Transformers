import argparse
import json
from pathlib import Path
import torch

from compact_transformer_encoder import CompactTransformerConfig, CompactTransformerEncoder
from vocabulary import Vocabulary


def _get_optional_sep_id(vocab) -> int | None:
    # falls du sowas nicht hast: return None
    if hasattr(vocab, "get_separator_token"):
        return vocab.token_to_id(vocab.get_separator_token())
    if hasattr(vocab, "get_sep_token"):
        return vocab.token_to_id(vocab.get_sep_token())
    return None


def build_one_sequence(demo_ids, event_ids, max_len, pad_id, sep_id=None):
    ids = list(demo_ids)
    seg = [0] * len(demo_ids)

    if sep_id is not None:
        ids.append(sep_id)
        seg.append(0)

    ids.extend(event_ids)
    seg.extend([1] * len(event_ids))

    # truncate
    ids = ids[:max_len]
    seg = seg[:max_len]

    attn = [1] * len(ids)

    # pad right
    while len(ids) < max_len:
        ids.append(pad_id)
        seg.append(0)
        attn.append(0)

    return ids, attn, seg


@torch.no_grad()
def predict_next_token(model, input_ids, attention_mask, event_type_ids, topk=10):
    device = next(model.parameters()).device
    x  = torch.tensor([input_ids], dtype=torch.long, device=device)
    am = torch.tensor([attention_mask], dtype=torch.long, device=device)
    et = torch.tensor([event_type_ids], dtype=torch.long, device=device)

    out = model(input_ids=x, attention_mask=am, event_type_ids=et, labels=None, return_hidden=False)
    logits = out["logits"][0]  # (L, V)

    last_idx = int(am[0].sum().item()) - 1
    last_idx = max(0, last_idx)

    probs = torch.softmax(logits[last_idx], dim=-1)
    p, ids = torch.topk(probs, k=topk)

    return [(int(i.item()), float(pp.item())) for i, pp in zip(ids, p)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_path", type=str, default="../out/sequences/ids.json")
    parser.add_argument("--vocab_path", type=str, default="../out/vocab/vocabulary.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/train_log_20260130_174849.pth")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    vocab = Vocabulary.load(Path(args.vocab_path))
    pad_id = vocab.token_to_id(vocab.get_padding_token())
    mask_id = vocab.token_to_id(vocab.get_masking_token())
    sep_id = _get_optional_sep_id(vocab)

    cfg = CompactTransformerConfig(
        vocab_size=vocab.get_size(),
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        use_event_type_embeddings=True,  # muss True sein, weil dein Checkpoint event_type_emb hat
        n_event_types=2,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactTransformerEncoder(cfg).to(device)
    state = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(state)
    model.eval()

    joint = json.loads(Path(args.ids_path).read_text(encoding="utf-8"))
    # joint: [ [demo_ids, event_ids], ... ]

    for i, (demo_ids, event_ids) in enumerate(joint[: args.limit]):
        input_ids, attn, seg = build_one_sequence(demo_ids, event_ids, args.max_len, pad_id, sep_id)

        # Log letzte Tokens (als Strings)
        # nur die letzten validen (non-pad) anzeigen
        valid_len = sum(attn)
        tail_n = min(50, valid_len)

        print("\n" + "=" * 90)
        print(f"sequence_index={i}  valid_len={valid_len}  (showing last {tail_n})")
        for tid in input_ids[valid_len - tail_n : valid_len]:
            print(" ", vocab.id_to_token(tid))

        preds = predict_next_token(model, input_ids, attn, seg, topk=args.topk)

        print("\nNext-token predictions (top-k):")
        for tid, p in preds:
            print(f"  {vocab.id_to_token(tid):<45s} vocab={vocab.vocab_name_for_token(vocab.id_to_token(tid)):<20s} id={tid:<6d} p={p:.4f}")


if __name__ == "__main__":
    main()