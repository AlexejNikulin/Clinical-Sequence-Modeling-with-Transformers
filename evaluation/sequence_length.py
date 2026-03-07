import json
import numpy as np

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def analyze_sequence_lengths(jsonl_path, context_length):
    data = load_jsonl(jsonl_path)

    lengths = []
    longer = 0

    for row in data:
        seq = row["input_ids"]
        lengths.append(len(seq))

        if len(seq) > context_length:
            longer += 1

    total = len(lengths)

    print("Total sequences:", total)
    print("Mean length:", np.mean(lengths))
    print("Median length:", np.median(lengths))
    print("Max length:", np.max(lengths))
    print("Min length:", np.min(lengths))
    print("Sequences > context_length:", longer)
    print("Percentage > context_length:", longer / total * 100, "%")


# Example
analyze_sequence_lengths("data/test_ids.jsonl", context_length=256)

def check_demographic_tokens(data, expected_n_demo=3):
    violations = 0

    for row in data:
        seq = row["input_ids"]

        if len(seq) < expected_n_demo:
            violations += 1
            continue
        
        DEMO_START = 0
        DEMO_END   = 100

        for i in range(expected_n_demo):
            if not (DEMO_START <= seq[i] < DEMO_END):
                violations += 1
                break

    print("Sequences violating demographic constraint:", violations)

def full_dataset_audit(jsonl_path, context_length):
    data = load_jsonl(jsonl_path)

    lengths = []
    longer = 0
    demo_violations = 0

    for row in data:
        seq = row["input_ids"]
        lengths.append(len(seq))

        if len(seq) > context_length:
            longer += 1

        if len(seq) < 3:
            demo_violations += 1

    print("----- DATASET AUDIT -----")
    print("Total sequences:", len(data))
    print("Mean length:", np.mean(lengths))
    print("Max length:", np.max(lengths))
    print("Sequences > context_length:", longer)
    print("Demographic violations:", demo_violations)