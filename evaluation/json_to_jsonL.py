import json
from pathlib import Path

# --- Input e output ---
# for test_id
INPUT_JSON = Path("../out/sequences/test_ids.json")
OUTPUT_JSONL = Path("../out/sequences/test_val.jsonl")


# --- Carica le sequenze ---
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    sequences = json.load(f)

# --- Scrivi in JSONL ---
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
    for i, patient_seqs in enumerate(sequences):
        
        flat_seq = [token for seq in patient_seqs for token in seq]

        record = {
            "patient_id": i,       
            "token_ids": flat_seq
        }
        f_out.write(json.dumps(record) + "\n")

print(f"Converted {len(sequences)} patients in {OUTPUT_JSONL}")
