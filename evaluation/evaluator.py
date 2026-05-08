"""
Evaluator: scores predictions against ground truth.

Two layers of metrics:

1. Deterministic (cheap, run on every record):
   - JSON validity rate
   - Per-field exact match (name, age, duration)
   - Set F1 for list fields (symptoms, negated_symptoms, history, diagnosis)
   - Set F1 for treatment as (type, normalized detail) tuples
   - Hallucination rate: fraction of predicted symptoms not groundable
     in the dialogue text

2. LLM-as-judge (more expensive, run on test set):
   - Per-field semantic equivalence (1/0)
   - Per-field faithfulness (predicted item supported by dialogue?)
   - Pairwise comparison between conditions

The LLM judge is implemented but optional — pass --skip-judge to run
deterministic only.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict


# ---- Deterministic metrics ----------------------------------------------

def normalize_str(s) -> str:
    """Lowercase, collapse whitespace, strip punctuation at edges.

    Defensive against non-string inputs. Baseline predictions don't follow
    the schema, so a field expected to be a string can come back as a list,
    dict, or number. We coerce to a stable string repr (via json.dumps for
    structured types) so equality comparisons return False rather than
    crashing the evaluator.
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s, sort_keys=True)
    return re.sub(r"\s+", " ", s.lower().strip(" .,;:"))


def set_f1(gold, pred) -> dict:
    """Set-level precision, recall, F1 with light normalization.

    Defensive: if gold or pred isn't a list (baseline can produce wrong
    types), treat it as empty so the comparison reports 0 rather than
    crashing.
    """
    if not isinstance(gold, list):
        gold = []
    if not isinstance(pred, list):
        pred = []
    gold_set = {normalize_str(x) for x in gold}
    pred_set = {normalize_str(x) for x in pred}
    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0,
                "fp": 0, "fn": 0}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def treatment_set_f1(gold, pred) -> dict:
    """Treatments compared as (type, normalized_detail) tuples.

    Defensive against non-list inputs from baseline.
    """
    if not isinstance(gold, list):
        gold = []
    if not isinstance(pred, list):
        pred = []
    def to_tuples(items):
        out = set()
        for i in items:
            if isinstance(i, dict) and "type" in i and "detail" in i:
                out.add((i["type"], normalize_str(i["detail"])))
        return out
    gold_set = to_tuples(gold)
    pred_set = to_tuples(pred)
    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def hallucination_rate(predicted_symptoms, dialogue: str) -> float:
    """Fraction of predicted symptoms NOT findable as substring in dialogue.

    Crude but useful — a symptom that doesn't appear in the dialogue
    (even loosely) is likely hallucinated. Stemming/synonyms would
    improve this; for now we use word-level overlap.

    Defensive: if predicted_symptoms isn't a list (baseline can produce
    a string or dict here), wrap a single string as a one-element list,
    or return 0.0 for completely unparseable cases.
    """
    if predicted_symptoms is None:
        return 0.0
    if isinstance(predicted_symptoms, str):
        predicted_symptoms = [predicted_symptoms]
    elif not isinstance(predicted_symptoms, list):
        return 0.0
    if not predicted_symptoms:
        return 0.0
    dialogue_lower = dialogue.lower()
    hallucinated = 0
    for s in predicted_symptoms:
        words = normalize_str(s).split()
        # Symptom is "grounded" if any content word appears in dialogue
        content_words = [w for w in words if len(w) > 3]
        if not content_words:
            continue
        if not any(w in dialogue_lower for w in content_words):
            hallucinated += 1
    return hallucinated / len(predicted_symptoms)


def score_one(gold: dict, pred, dialogue: str) -> dict:
    """Compute deterministic metrics for one (gold, pred) pair.

    Defensive: if pred isn't a dict (entire baseline output unparseable),
    treat it as an empty dict so all field-level metrics correctly
    report 0/empty rather than crashing.
    """
    if not isinstance(pred, dict):
        pred = {}
    if not isinstance(gold, dict):
        gold = {}
    metrics = {
        "name_exact": int(gold.get("name") == pred.get("name")),
        "age_exact": int(gold.get("age") == pred.get("age")),
        "duration_exact": int(
            normalize_str(gold.get("duration") or "") ==
            normalize_str(pred.get("duration") or "")
        ),
    }
    for field in ["symptoms", "negated_symptoms", "history", "diagnosis"]:
        m = set_f1(gold.get(field, []), pred.get(field, []))
        for k, v in m.items():
            metrics[f"{field}_{k}"] = round(v, 3)

    tm = treatment_set_f1(gold.get("treatment", []), pred.get("treatment", []))
    for k, v in tm.items():
        metrics[f"treatment_{k}"] = round(v, 3)

    metrics["hallucination_rate"] = round(
        hallucination_rate(pred.get("symptoms", []), dialogue), 3
    )
    return metrics


# ---- LLM-as-judge -------------------------------------------------------

JUDGE_MODEL = "gpt-4o"

JUDGE_PROMPT_TEMPLATE = """You are evaluating a clinical information extraction system.

DIALOGUE:
{dialogue}

REFERENCE JSON (ground truth):
{reference}

PREDICTED JSON (model output):
{prediction}

For each field, score whether the prediction is clinically equivalent \
to the reference. For each item in the predicted symptoms and treatment \
lists, score whether it is supported by the dialogue (faithfulness).

Return JSON with this exact structure:
{{
  "name_equivalent": 0 or 1,
  "symptoms_equivalent": 0 or 1,
  "negated_symptoms_equivalent": 0 or 1,
  "diagnosis_equivalent": 0 or 1,
  "treatment_equivalent": 0 or 1,
  "predicted_symptoms_grounded": <fraction 0.0-1.0>,
  "predicted_treatments_grounded": <fraction 0.0-1.0>,
  "rationale": "<one sentence>"
}}

JSON only, no preamble."""


def llm_judge_one(client, dialogue: str, gold: dict, pred: dict) -> dict:
    """Run the LLM judge on one (gold, pred) pair."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        dialogue=dialogue,
        reference=json.dumps(gold, indent=2),
        prediction=json.dumps(pred, indent=2),
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---- Aggregation -------------------------------------------------------

def aggregate(per_record: list[dict]) -> dict:
    """Mean across all per-record metrics."""
    if not per_record:
        return {}
    sums = defaultdict(float)
    counts = defaultdict(int)
    for record in per_record:
        for k, v in record.items():
            if isinstance(v, (int, float)):
                sums[k] += v
                counts[k] += 1
    return {k: round(sums[k] / counts[k], 3) for k in sums}


# ---- Main evaluation flow ----------------------------------------------

def evaluate_condition(
    condition: str,
    labels_by_id: dict,
    predictions_path: Path,
    use_judge: bool,
    judge_client=None,
) -> dict:
    """Evaluate one condition's predictions."""
    print(f"\nEvaluating condition: {condition}")

    per_record_det = []
    per_record_judge = []
    valid_count = 0
    total_count = 0
    total_latency = 0.0

    with predictions_path.open() as f:
        for line in f:
            r = json.loads(line)
            eid = r["encounter_id"]
            if eid not in labels_by_id:
                continue
            gold_record = labels_by_id[eid]
            gold = gold_record["label"]
            dialogue = gold_record["dialogue"]
            pred = r["prediction"]

            total_count += 1
            if r["valid"]:
                valid_count += 1
            total_latency += r.get("latency_sec", 0)

            # Deterministic metrics — always run
            det = score_one(gold, pred, dialogue)
            det["encounter_id"] = eid
            per_record_det.append(det)

            # LLM judge — optional
            if use_judge and r["valid"]:
                try:
                    judge = llm_judge_one(judge_client, dialogue, gold, pred)
                    judge["encounter_id"] = eid
                    per_record_judge.append(judge)
                except Exception as e:
                    print(f"Judge failed on {eid}: {e}", file=sys.stderr)

    summary = {
        "condition": condition,
        "n_records": total_count,
        "validity_rate": round(valid_count / total_count, 3) if total_count else 0,
        "avg_latency_sec": round(total_latency / total_count, 2) if total_count else 0,
        "deterministic": aggregate(per_record_det),
    }
    if per_record_judge:
        summary["llm_judge"] = aggregate(per_record_judge)
    summary["per_record_deterministic"] = per_record_det
    if per_record_judge:
        summary["per_record_judge"] = per_record_judge
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True,
                        help="Path to labels_clean.jsonl (ground truth).")
    parser.add_argument("--predictions-dir", required=True,
                        help="Directory with predictions_{cond}.jsonl files.")
    parser.add_argument("--output", required=True,
                        help="Path to write evaluation summary JSON.")
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "soft", "medium", "hard"])
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-judge (faster, free).")
    args = parser.parse_args()

    # Load gold labels indexed by encounter_id
    labels_by_id = {}
    with open(args.labels) as f:
        for line in f:
            r = json.loads(line)
            labels_by_id[r["encounter_id"]] = r
    print(f"Loaded {len(labels_by_id)} ground-truth records.")

    # Set up judge if needed
    judge_client = None
    if not args.skip_judge:
        try:
            from openai import OpenAI
        except ImportError:
            print("Install openai for LLM-as-judge, or use --skip-judge.",
                  file=sys.stderr)
            sys.exit(1)
        if not os.environ.get("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY for LLM-as-judge, or use --skip-judge.",
                  file=sys.stderr)
            sys.exit(1)
        judge_client = OpenAI()

    # Evaluate each condition
    pred_dir = Path(args.predictions_dir)
    all_results = {}
    for cond in args.conditions:
        path = pred_dir / f"predictions_{cond}.jsonl"
        if not path.exists():
            print(f"Skipping {cond}: {path} not found.")
            continue
        all_results[cond] = evaluate_condition(
            cond, labels_by_id, path, not args.skip_judge, judge_client
        )

    # Write summary
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.output}")

    # Print headline numbers
    print("\n=== HEADLINE METRICS ===")
    print(f"{'condition':<12} {'validity':>10} {'sym_f1':>10} "
          f"{'neg_f1':>10} {'tx_f1':>10} {'halluc':>10}")
    for cond, r in all_results.items():
        det = r["deterministic"]
        print(f"{cond:<12} {r['validity_rate']:>10.3f} "
              f"{det.get('symptoms_f1', 0):>10.3f} "
              f"{det.get('negated_symptoms_f1', 0):>10.3f} "
              f"{det.get('treatment_f1', 0):>10.3f} "
              f"{det.get('hallucination_rate', 0):>10.3f}")


if __name__ == "__main__":
    main()
