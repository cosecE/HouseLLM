"""
Run inference under all four conditions and save predictions.

Usage:
    python run_inference.py \
        --labels labels_clean.jsonl \
        --output predictions/ \
        --icd10-csv icd10_filtered.csv \
        --conditions baseline soft medium hard

For each condition, writes predictions_{condition}.jsonl with records:
    {"encounter_id": "...", "prediction": {...}, "valid": true/false}

Few-shot example encounters (defined in few_shot_examples.py) are
automatically excluded from inference and evaluation to prevent leakage.

Resumable: skips encounters already in the output file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from llama_runner import LlamaRunner, load_icd10_vocab
from schema import ClinicalNote
from few_shot_examples import FEW_SHOT_EXAMPLES, FEW_SHOT_IDS


def load_labels(path: Path) -> list[dict]:
    """Load the cleaned ground-truth labels JSONL."""
    records = []
    with path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_already_predicted(output_path: Path) -> set[str]:
    """Resumable: track which encounter_ids have been predicted."""
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["encounter_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def validate_prediction(pred: dict) -> tuple[bool, str]:
    """Check if prediction is valid against the schema."""
    try:
        ClinicalNote(**pred)
        return True, ""
    except (ValidationError, TypeError) as e:
        return False, str(e)[:200]


def run_condition(
    condition: str,
    labels: list[dict],
    output_path: Path,
    few_shot_examples: list,
    diagnosis_vocab: list[str],
):
    """Run one condition over the dataset."""
    print(f"\n{'='*60}\nCondition: {condition}\n{'='*60}")

    already_done = load_already_predicted(output_path)
    print(f"Resuming: {len(already_done)} already predicted.")

    # Initialize runner — each condition loads the model once
    kwargs = {"condition": condition, "few_shot_examples": few_shot_examples}
    if condition == "hard":
        kwargs["diagnosis_vocab"] = diagnosis_vocab

    runner = LlamaRunner(**kwargs)

    failures = 0
    with output_path.open("a") as fout:
        for r in labels:
            eid = r["encounter_id"]
            if eid in already_done:
                continue

            t0 = time.time()
            try:
                pred = runner.generate(r["dialogue"])
                valid, error = validate_prediction(pred)
                record = {
                    "encounter_id": eid,
                    "prediction": pred,
                    "valid": valid,
                    "validation_error": error if not valid else None,
                    "latency_sec": round(time.time() - t0, 2),
                }
            except Exception as e:
                failures += 1
                record = {
                    "encounter_id": eid,
                    "prediction": {},
                    "valid": False,
                    "validation_error": f"runtime error: {e}",
                    "latency_sec": round(time.time() - t0, 2),
                }

            fout.write(json.dumps(record) + "\n")
            fout.flush()
            print(f"[{condition}] {eid} valid={record['valid']} "
                  f"t={record['latency_sec']}s")

    if failures:
        print(f"\n{condition}: {failures} runtime failures.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True,
                        help="Path to labels_clean.jsonl (ground truth).")
    parser.add_argument("--output", required=True,
                        help="Output directory for predictions.")
    parser.add_argument("--icd10-csv", default="icd10_filtered.csv",
                        help="Path to filtered ICD-10 CSV (used by hard "
                             "condition for diagnosis/history vocabulary).")
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "soft", "medium", "hard"],
                        choices=["baseline", "soft", "medium", "hard"])
    args = parser.parse_args()

    # Load all labels, then exclude any used as few-shot examples (leakage).
    all_labels = load_labels(Path(args.labels))
    labels = [r for r in all_labels if r["encounter_id"] not in FEW_SHOT_IDS]
    n_excluded = len(all_labels) - len(labels)
    print(f"Loaded {len(all_labels)} ground-truth records.")
    print(f"Excluded {n_excluded} few-shot examples "
          f"({sorted(FEW_SHOT_IDS)}); evaluating on {len(labels)} records.")

    # ICD-10 vocab is only needed when running the hard condition.
    # The hard condition constrains diagnosis and history fields to ICD-10
    # long descriptions. Symptoms are intentionally left unconstrained
    # (no widely-used external symptom vocabulary; using a label-derived
    # one would be circular). See llama_runner.py for the full rationale.
    diagnosis_vocab = []
    if "hard" in args.conditions:
        diagnosis_vocab = load_icd10_vocab(args.icd10_csv)
        print(f"Diagnosis vocab (from {args.icd10_csv}): "
              f"{len(diagnosis_vocab)} ICD-10 descriptions.")

    print(f"Using {len(FEW_SHOT_EXAMPLES)} few-shot examples in soft/"
          f"medium/hard prompts (baseline ignores them).")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions:
        out_path = output_dir / f"predictions_{cond}.jsonl"
        run_condition(
            cond, labels, out_path, FEW_SHOT_EXAMPLES, diagnosis_vocab,
        )

    print(f"\nAll done. Predictions in {output_dir}/")


if __name__ == "__main__":
    main()
