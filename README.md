# Constrained Decoding for Clinical Information Extraction

A systematic comparison of four constraint levels for medical dialogue → JSON
extraction using Llama-3.2-3B-Instruct. We find that schema enforcement is a
free win, but ICD-10 vocabulary constraint *cratered* diagnosis F1 from 0.46
(medium) to 0.015 (hard), what we call **semantic substitution**.

## Key findings

- Schema validity: 3% (baseline) → 100% (medium); schema enforcement works.
- Mean F1 across list fields: peaks at medium (0.32), drops to 0.10 at hard.
- Diagnosis F1: 0.46 → 0.015 at hard, the hard condition substitutes
  vocabulary-valid but semantically wrong codes.
- Hallucination rate: lowest at hard (0.05), but at the cost of F1.
- Constraint enforces vocabulary and schema, but cannot enforce *which entity
  belongs in which field* (the "Dragon" finding).

## Setup

```bash
git clone https://github.com/<username>/houseLLM-constrained-decoding
cd houseLLM-constrained-decoding
pip install -r requirements.txt
cp .env.example .env  # then fill in your OPENAI_API_KEY
```

## Reproducing the pipeline

```bash
# Stage 1: generate gold labels (one-time, requires OpenAI API)
python labeler/make_labels.py --input data/train.csv --output labeler/labels_clean.jsonl

# Stage 2: run inference across all four conditions
python inference/run_inference.py \
    --labels labeler/labels_clean.jsonl \
    --output results/predictions/ \
    --icd10-csv data/icd10_filtered.csv \
    --conditions baseline soft medium hard

# Stage 3: evaluate
python evaluation/evaluator.py \
    --labels labeler/labels_clean.jsonl \
    --predictions-dir results/predictions/ \
    --output results/results_full.json

# Generate plots
python evaluation/make_plots.py \
    --results results/results_full.json \
    --out results/plots/
```

## Summary of How the Files Connect

```
train.csv
    ↓
labeler.py  (GPT-4o + annotation_guide + few_shot_examples)
    ↓
labels_clean.jsonl  ←── spot_check.py (quality validation)
    ↓
run_inference.py  (4 conditions)
    ├── LlamaRunner [baseline]   → predictions_baseline.jsonl
    ├── LlamaRunner [soft]       → predictions_soft.jsonl
    ├── LlamaRunner [medium]     → predictions_medium.jsonl  (schema-constrained)
    └── LlamaRunner [hard]       → predictions_hard.jsonl   (schema + ICD-10 constrained)
                                            ↓
                                    evaluator.py
                                    ├── deterministic metrics (all records)
                                    └── LLM judge (valid records only)
                                            ↓
                                    results_full.json → make_plots.py

```

## For more information about the code:

See document: [`CodeOverview.pdf`](CodeOverview.pdf)

## To visualize the constraints on UI, run ui.py

```bash
cd/ui
pip install streamlit
streamlit run ui.py
```

or https://housellm.streamlit.app/


## Results

See [`results/plots/`](results/plots/).

## Limitations

- Single 3B model; larger models likely show different patterns.
- 65 dialogues, English only.
- ICD-10 vocabulary subset of 2,763 of 73,000 codes.
- Library version dependence (lm-format-enforcer).

## Authors

[Team xyz]

Alisa Zheng, Kaushiki Singh, Sneha Jaikumar

## Individual Code Contributions
- Sneha: Implemented the medium and hard constraint pipeline using lm-format-enforcer, including JSON schema enforcement and ICD-10 vocabulary integration for diagnosis and history fields, and debugged constraint library compatibility issues (Outlines → lm-format-enforcer migration)

- Kaushiki: Built the end to end evaluation framework including deterministic metrics (set-level F1, validity rate, hallucination rate) and generated visualizations for cross-condition comparison

- Alisa: Implemented the baseline and soft inference conditions, developed and ran smoke tests across all four conditions, debugged constraint library compatibility issues (Outlines → lm-format-enforcer migration), and validated pipeline outputs
