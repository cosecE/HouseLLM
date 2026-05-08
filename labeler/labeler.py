"""
LLM labeler: takes a dialogue, returns a validated ClinicalNote JSON label.

Uses OpenAI's API (GPT-4o by default) with JSON mode for guaranteed
valid JSON output.

Usage:
    python labeler.py --input dataset.csv --output labels.jsonl
    python labeler.py --input dataset.csv --output labels.jsonl --limit 10

Requires:
    pip install openai pydantic pandas
    export OPENAI_API_KEY=...

Notes:
- Each output is validated against schema.ClinicalNote before being saved.
  Invalid outputs are retried up to MAX_RETRIES with the validation error
  fed back to the model.
- Outputs are written one-per-line as JSONL so the loop is resumable: if
  the script crashes, already-labeled rows are skipped on the next run.
- JSON mode (response_format={"type": "json_object"}) ensures the model
  always returns valid JSON, eliminating most parse-failure retries.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from annotation_guide import ANNOTATION_GUIDE
from few_shot_examples import FEW_SHOT_EXAMPLES
from schema import ClinicalNote


MODEL = "gpt-4o"  # gpt-4o-mini is cheaper but lower quality on extraction
MAX_RETRIES = 2
REQUEST_TIMEOUT_SEC = 120


SYSTEM_PROMPT = f"""You are a clinical note annotator. Given a doctor-patient \
dialogue, extract structured information into a JSON object that conforms \
exactly to the schema below.

{ANNOTATION_GUIDE}

OUTPUT FORMAT
- Respond with a single valid JSON object. No preamble, no markdown fences, \
no commentary.
- All list fields default to [] if there is nothing to extract.
- Scalar fields (name, age, duration) default to null if absent.
"""


def build_user_prompt(dialogue: str) -> str:
    """Build the user message: few-shot examples + the target dialogue."""
    parts = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"--- Example {i} dialogue ---\n{ex['dialogue']}")
        parts.append(
            f"--- Example {i} JSON ---\n"
            + json.dumps(ex["label"], indent=2)
        )
    parts.append(f"--- Target dialogue ---\n{dialogue}")
    parts.append("--- Target JSON ---")
    return "\n\n".join(parts)


def parse_json_response(raw: str) -> dict:
    """Strip possible markdown fences and parse JSON. Raise on failure."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
    return json.loads(raw)


def label_one_dialogue(client, dialogue: str) -> ClinicalNote:
    """Label a single dialogue. Retries on validation/parse errors."""
    user_prompt = build_user_prompt(dialogue)
    error_feedback = None

    for attempt in range(MAX_RETRIES + 1):
        messages = [{"role": "user", "content": user_prompt}]
        if error_feedback is not None:
            messages.append({
                "role": "assistant",
                "content": error_feedback["raw_response"],
            })
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous response failed validation:\n"
                    f"{error_feedback['error']}\n\n"
                    "Re-emit the JSON, fixing the errors. JSON only."
                ),
            })

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *messages,
            ],
            timeout=REQUEST_TIMEOUT_SEC,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content

        try:
            parsed = parse_json_response(raw)
            return ClinicalNote(**parsed)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            error_feedback = {"raw_response": raw, "error": str(e)}
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Labeler failed after {MAX_RETRIES + 1} attempts. "
                    f"Last error: {e}\nLast raw response:\n{raw}"
                ) from e


def load_already_labeled(output_path: Path) -> set[str]:
    """Read existing JSONL output to skip already-labeled encounter_ids."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to dataset CSV with encounter_id, dialogue columns.")
    parser.add_argument("--output", required=True,
                        help="Path to output JSONL file.")
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only label the first N rows.")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between API calls.")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("Install the OpenAI SDK: pip install openai", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    df = pd.read_csv(args.input)

    required_cols = {"encounter_id", "dialogue"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Input CSV is missing required columns: {missing}", file=sys.stderr)
        print(f"Found columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    has_dataset_col = "dataset" in df.columns

    if args.limit:
        df = df.head(args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = load_already_labeled(output_path)
    print(f"Resuming: {len(already_done)} encounters already labeled.")
    print(f"To label: {len(df) - len(already_done & set(df['encounter_id']))}")

    failures = []
    with output_path.open("a") as fout:
        for _, row in df.iterrows():
            eid = row["encounter_id"]
            if eid in already_done:
                continue
            try:
                label = label_one_dialogue(client, row["dialogue"])
                record = {
                    "encounter_id": eid,
                    "dialogue": row["dialogue"],
                    "label": label.model_dump(),
                }
                if has_dataset_col:
                    record["dataset"] = row["dataset"]
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                print(f"[ok] {eid}")
            except Exception as e:
                print(f"[fail] {eid}: {e}", file=sys.stderr)
                failures.append({"encounter_id": eid, "error": str(e)})
            time.sleep(args.sleep)

    if failures:
        fail_path = output_path.with_suffix(".failures.json")
        with fail_path.open("w") as f:
            json.dump(failures, f, indent=2)
        print(f"\n{len(failures)} failures written to {fail_path}")
    print(f"Done. Total labeled: {len(already_done) + len(df) - len(failures)}")


if __name__ == "__main__":
    main()
