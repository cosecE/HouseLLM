"""
Llama runner: produces JSON output from a dialogue under one of four
conditions. The rest of the code doesn't need to know which condition
is active — it just calls .generate(dialogue) and gets a dict back.

Conditions:
- baseline: vanilla generation, prompt asks for JSON but no enforcement
- soft:     same as baseline but with stronger prompt + few-shot
- medium:   Outlines enforces the JSON schema at decoding time
- hard:     Outlines enforces schema + ICD-10 vocabulary on diagnosis
            and history fields. Symptoms are intentionally NOT
            constrained — see _build_hard_constrained_schema for why.

Requires:
    pip install transformers torch outlines pydantic

Recommended model: meta-llama/Llama-3.2-3B-Instruct
(Fits in a Colab T4. For better quality use Llama-3.1-8B-Instruct on A100.)
"""

import csv
import json
from typing import Literal, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Outlines imports are deferred to inside methods so baseline/soft work
# without it installed.

from schema import ClinicalNote


Condition = Literal["baseline", "soft", "medium", "hard"]


# Default model — change here if you want something bigger
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MAX_NEW_TOKENS = 1024


# ---- Prompts ---------------------------------------------------------

BASELINE_SYSTEM_PROMPT = """You extract structured clinical information \
from doctor-patient dialogues. Output a JSON object with these fields: \
name, age, symptoms, duration, negated_symptoms, history, diagnosis, \
treatment. Output JSON only."""


SOFT_SYSTEM_PROMPT = """You are a clinical note annotator. Given a \
doctor-patient dialogue, extract structured information into a JSON \
object with exactly these fields:

- name: patient's first name as spoken in the dialogue, or null
- age: integer, or null
- symptoms: list of patient-reported current symptoms (lowercase strings)
- duration: duration of the chief complaint, or null
- negated_symptoms: list of symptoms the patient explicitly denied
- history: list of past medical conditions mentioned
- diagnosis: list of conditions the doctor commits to a plan for
- treatment: list of {type, detail} objects where type is one of
  medication, test, referral, counseling, follow_up

Extract only what is in the dialogue. Output a single JSON object, \
no preamble, no markdown fences."""

HARD_SYSTEM_PROMPT = """You are a clinical note annotator. Given a \
doctor-patient dialogue, extract structured information into a JSON \
object with exactly these fields:

- name: patient's first name as spoken in the dialogue, or null
- age: integer, or null
- symptoms: list of patient-reported current symptoms — sensations or \
experiences the patient describes (e.g. "joint pain", "shortness of \
breath", "dizziness"). Symptoms are NOT diagnoses.
- duration: duration of the chief complaint as a string, or null
- negated_symptoms: list of symptoms the patient explicitly denied
- history: list of past medical conditions mentioned in the dialogue
- diagnosis: list of conditions the doctor commits to a plan for
- treatment: list of {type, detail} objects where type is one of
  medication, test, referral, counseling, follow_up

EXTRACTION RULES:
- Extract everything that is clearly stated in the dialogue. Fill in \
each field with content from the dialogue when available.
- Do not invent information. If a name is not spoken, use null. If no \
symptoms are denied, use an empty list, not a list containing an empty \
string.
- Use null for missing scalar fields. Never use empty strings ("") for \
scalar fields. List entries must not be empty strings.
- Symptoms are what the patient feels or reports (e.g. "back pain", \
"nausea"). Diagnostic terms like "fracture", "pneumonia", or \
"hypothyroidism" belong in the diagnosis or history fields, not in \
symptoms.
- For diagnosis and history: your output for these fields is restricted \
to a closed vocabulary of ICD-10 long descriptions. Include only entries \
that clearly match conditions actually mentioned in the dialogue. Most \
dialogues mention 0-4 history items and 1-3 diagnoses. Do not pad these \
lists with conditions not present in the dialogue.
- If a condition mentioned in the dialogue does not clearly match any \
vocabulary entry, OMIT it rather than substituting an unrelated condition.
- Each diagnosis and history entry should appear at most once.

Output a single JSON object, no preamble, no markdown fences."""

def build_user_prompt(dialogue: str, few_shot_examples: Optional[list] = None) -> str:
    """Build the user-side prompt. Few-shot is optional."""
    parts = []
    if few_shot_examples:
        for i, ex in enumerate(few_shot_examples, 1):
            parts.append(f"--- Example {i} dialogue ---\n{ex['dialogue']}")
            parts.append(
                f"--- Example {i} JSON ---\n"
                + json.dumps(ex["label"], indent=2)
            )
    parts.append(f"--- Target dialogue ---\n{dialogue}")
    parts.append("--- Target JSON ---")
    return "\n\n".join(parts)


# ---- ICD-10 helper --------------------------------------------------

def load_icd10_vocab(csv_path: str) -> list[str]:
    """Load ICD-10 long descriptions from a CSV as a vocabulary list.

    Expects the CSV format used by the official FY2026 file:
        CODE, SHORT DESCRIPTION (...), LONG DESCRIPTION (...)
    Returns the sorted, deduplicated list of long descriptions.
    """
    descriptions = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        long_col = next(
            (c for c in reader.fieldnames if c.startswith("LONG DESCRIPTION")),
            None,
        )
        if long_col is None:
            raise ValueError(
                f"No 'LONG DESCRIPTION' column found in {csv_path}. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            desc = row[long_col].strip()
            if desc:
                descriptions.append(desc)
    return sorted(set(descriptions))


# ---- Runner ---------------------------------------------------------

class LlamaRunner:
    """Single class that handles all four conditions.

    Load once, then call .generate(dialogue) repeatedly.
    """

    def __init__(
        self,
        condition: Condition,
        model_name: str = DEFAULT_MODEL,
        few_shot_examples: Optional[list] = None,
        diagnosis_vocab: Optional[list[str]] = None,
    ):
        self.condition = condition
        self.few_shot = few_shot_examples
        self.diagnosis_vocab = diagnosis_vocab

        print(f"Loading {model_name} for condition={condition}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # For medium/hard, set up Outlines
        self.outlines_generator = None
        if condition in ("medium", "hard"):
            self._setup_constraints()

    def _setup_constraints(self):
            """Set up structured generation using lm-format-enforcer for both
            medium and hard. Replaces the earlier Outlines-based implementation
            which had FSM compilation issues with list[Literal[...]] types."""
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

            if self.condition == "medium":
                schema = ClinicalNote.model_json_schema()
            elif self.condition == "hard":
                constrained = self._build_hard_constrained_schema()
                schema = constrained.model_json_schema()
                print(
                    f"Hard condition: lm-format-enforcer constraining diagnosis/history "
                    f"to {len(self.diagnosis_vocab or [])} ICD-10 descriptions."
                )
            else:
                return  # baseline / soft don't need constraints

            parser = JsonSchemaParser(schema)
            self.prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
                self.tokenizer, parser
            )

    def _build_hard_constrained_schema(self):
        """Build a Pydantic model with Literal types for diagnosis fields.

        Constrains `diagnosis` and `history` to the ICD-10 long-description
        vocabulary (if provided). If no vocab is provided, falls back to
        the unconstrained schema (equivalent to medium).

        Why we don't constrain symptoms here:
        Patient-reported symptoms ("back pain", "feeling dizzy when
        climbing stairs") don't have a widely-used external vocabulary.
        Using a vocabulary derived from our own gold labels would be
        circular: the model would be restricted to exactly the answers
        we want it to give, inflating scores artificially. Honest
        comparison requires either an external symptom ontology
        (e.g. SNOMED CT, UMLS) or leaving symptoms unconstrained.
        We chose the latter for simplicity and methodological clarity.

        Note: Outlines + Literal lists are tractable up to ~a few thousand
        entries. For larger vocabularies you'd need a regex grammar.
        """
        from typing import Literal as TLiteral
        from pydantic import Field, create_model

        if not self.diagnosis_vocab:
            return ClinicalNote  # fall back to medium

        diagnosis_literal = TLiteral[tuple(self.diagnosis_vocab)]
        ConstrainedNote = create_model(
            "ConstrainedClinicalNote",
            __base__=ClinicalNote,
            diagnosis=(list[diagnosis_literal], Field(default_factory=list)),
            history=(list[diagnosis_literal], Field(default_factory=list)),
        )
        return ConstrainedNote

    def _build_messages(self, dialogue: str) -> list[dict]:
            """Build the chat-formatted messages."""
            if self.condition == "baseline":
                system = BASELINE_SYSTEM_PROMPT
                few_shot = None
            elif self.condition == "hard":
                system = HARD_SYSTEM_PROMPT
                few_shot = self.few_shot
            else:
                # soft and medium share the same prompt
                system = SOFT_SYSTEM_PROMPT
                few_shot = self.few_shot

            return [
                {"role": "system", "content": system},
                {"role": "user", "content": build_user_prompt(dialogue, few_shot)},
            ]

    def generate(self, dialogue: str) -> dict:
        """Generate a JSON label for one dialogue."""
        messages = self._build_messages(dialogue)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.condition in ("medium", "hard"):
            gen_kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_tokens_fn
        if self.condition == "hard":
            gen_kwargs["repetition_penalty"] = 1.3
            gen_kwargs["no_repeat_ngram_size"] = 10

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        raw = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return self._parse_loose_json(raw)

    @staticmethod
    def _parse_loose_json(raw: str) -> dict:
        """Try to extract JSON from a loose model output.

        Returns {} on failure — caller decides how to handle invalid output.
        """
        raw = raw.strip()
        # Strip markdown fences
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                p = part.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    raw = p
                    break

        # Try to find the JSON object
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Last resort: find first { ... } block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass

        return {}
