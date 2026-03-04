# Clinical-Sequence-Modeling-with-Transformers
This project focuses on building a transformer-based clinical sequence model from scratch using temporal ICD-10 event histories extracted from the MIMIC-IV database.

## How to use - Script explanation

### Setup

1. Download the MIMIC-IV dataset from physionet (e.g. with `wget -r -N -c -np --user YOUR_USER --ask-password https://physionet.org/files/mimiciv/3.1/`). The dataset needs to be located at `../physionet.org`, so that the hospital files are located at `../physionet.org/files/mimiciv/3.1/hosp/*.csv`.

2. Install dependencies: `pip install -r requirements.txt`

3. Run the preprocessing script: `python preprocessing_main.py`. This will create the folder `../out`, which contains the tokenised dataset splits, the vocabulary definition etc.

### Training

1. Run the training script: `python -m transformer.transformer_train_model`. Run with `--help` to see all the options, or check the corresponding source code for implementation details. The training script will automatically create checkpoints in `./checkpoints_raw` after every epoch, as well as when training finishes. You can view training logs in `./logs`, and visualise training and validation loss using `python plot_logs.py`.

### Evaluation

1. Convert test data to jsonL: `python evaluation/json_to_jsonL.py`.

2. Convert pytorch checkpoints: `./evaluation/convert_pth_to_pt.py` contains all the meta-information about the model checkpoints. After training a new model, you need to add a new entry to the `FILES` variable that matches your architecture. Next, run `python evaluation/convert_pth_to_pt.py`, and the converted checkpoint will be written to `./checkpoints`.

3. Run the evaluation of your choice (all scripts support `--help` and `--seed`):

#### MLM evaluation

Token-level:
`python -m evaluation.mlm_eval --jsonl ../out/sequences/test_ids.jsonl --ckpt checkpoints/YOUR_CHECKPOINT.pt`

Event-type-level:
`python -m evaluation_event_type.mlm_eval --jsonl ../out/sequences/test_ids.jsonl --vocab_path ../out/vocab/vocabulary.json --ckpt checkpoints/YOUR_CHECKPOINT.pt`

#### NE evaluation

Token-level:
`python -m evaluation.next_event_eval --jsonl ../out/sequences/test_ids.jsonl --ckpt checkpoints/YOUR_CHECKPOINT.pt`

Event-type-level:
`python -m evaluation_event_type.next_event_eval --vocab_path ../out/vocab/vocabulary.json --jsonl ../out/sequences/test_ids.jsonl --ckpt checkpoints/YOUR_CHECKPOINT.pt`

#### N-NE evaluation

Token-level:
`python -m evaluation.next_event_token_eval_log --jsonl ../out/sequences/test_ids.jsonl --vocab_path ../out/vocab/vocabulary.json --max_len CONTEXT_LENGTH --ckpt checkpoints/YOUR_CHECKPOINT.pt`

Event-type-level:
`python -m evaluation_event_type.next_event_token_eval_log --jsonl ../out/sequences/test_ids.jsonl --vocab_path ../out/vocab/vocabulary.json --max_len CONTEXT_LENGTH --ckpt checkpoints/YOUR_CHECKPOINT.pt`

#### NE-Seq-N evaluation

Token-level:
`python -m evaluation.next_event_seq_eval_log --jsonl ../out/sequences/test_ids.jsonl --vocab_path ../out/vocab/vocabulary.json --max_len CONTEXT_LENGTH --ckpt checkpoints/YOUR_CHECKPOINT.pt`

Event-type-level:
`python -m evaluation_event_type.next_event_seq_eval_log --jsonl ../out/sequences/test_ids.jsonl --vocab_path ../out/vocab/vocabulary.json --max_len CONTEXT_LENGTH --ckpt checkpoints/YOUR_CHECKPOINT.pt`

#### NE-Seq-End evaluation

Token-level:
`python -m evaluation.next_event_seq_until_end_eval_log --jsonl ../out/sequences/test_ids.jsonl --vocab_path ../out/vocab/vocabulary.json --max_len CONTEXT_LENGTH --ckpt checkpoints/YOUR_CHECKPOINT.pt`

Event-type-level:
`python -m evaluation_event_type.next_event_seq_until_end_eval_log --jsonl ../out/sequences/test_ids.jsonl --vocab_path ../out/vocab/vocabulary.json --max_len CONTEXT_LENGTH --ckpt checkpoints/YOUR_CHECKPOINT.pt`