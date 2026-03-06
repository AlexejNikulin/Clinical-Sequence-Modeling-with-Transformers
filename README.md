# Clinical-Sequence-Modeling-with-Transformers

This project focuses on building a transformer-based clinical sequence model from scratch using temporal clinical event histories extracted from the MIMIC-IV database.

The goal is to model patient trajectories by representing medical histories as ordered sequences of events such as diagnoses, laboratory measurements, medications, and admission events. A compact transformer encoder is trained on these sequences using a masked language modeling objective in order to learn contextual relationships between clinical events.

The trained model is evaluated on tasks such as masked event prediction and next-event prediction.

---

## Project Pipeline

The overall workflow of the project consists of four main steps:

### 1. Data Extraction

Clinical events are extracted from multiple tables of the **MIMIC-IV hospital module**, including:

- `patients`
- `admissions`
- `labevents`
- `diagnoses_icd`
- `emar`
- `omr`

These tables contain demographic information, diagnoses, laboratory measurements, medications, and admission-related data.

---

### 2. Data Preprocessing

The extracted events are processed to create a unified representation of patient histories.

Key steps include:

- cleaning and normalizing event values  
- aligning timestamps across different data sources  
- merging events from multiple tables  
- constructing chronological patient event sequences  

Each patient is represented as a sequence of clinical events ordered by time.

---

### 3. Tokenization

Clinical events are converted into discrete tokens that can be processed by the model.

This includes:

- creating a vocabulary of event tokens  
- adding special tokens such as `[PAD]`, `[MASK]`, and `[UNK]`  
- encoding demographic information  
- adding time tokens to represent temporal gaps between events  

The resulting sequences are used as input to the transformer model.

---

### 4. Model Training

A compact **transformer encoder** is trained on the tokenized sequences.

Key characteristics of the model:

- transformer encoder architecture  
- masked language modeling training objective  
- contextual prediction of masked clinical events  
- training on temporal patient event sequences  

The model learns contextual relationships between events in patient histories.

---

## Evaluation

The trained models are evaluated using two main tasks.

### Masked Language Modeling

Random tokens in the input sequence are masked and the model must predict the original event using the surrounding context.

### Next Event Prediction

The model receives a partial patient timeline and predicts the next clinical event.

Evaluation metrics include:

- **Top-1 accuracy**
- **Top-5 accuracy**
- **Top-10 accuracy**
- **Mean Reciprocal Rank (MRR)**

These metrics measure how well the model ranks the correct event among its predictions.

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
