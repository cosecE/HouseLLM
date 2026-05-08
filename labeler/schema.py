"""
Schema definition for ground truth JSON labels.

This is the canonical schema used by:
- The labeler (to instruct the LLM what to produce)
- The evaluator (to validate predictions)
- The constrained decoding step (medium constraints enforce this schema)
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


TreatmentType = Literal["medication", "test", "referral", "counseling", "follow_up", "procedure", "equipment"]


class TreatmentItem(BaseModel):
    """A single treatment item with structured type."""
    type: TreatmentType = Field(
        ...,
        description="Category of treatment item.",
    )
    detail: str = Field(
        ...,
        description="Free-text description, e.g. 'lisinopril 40 mg daily' or "
                    "'CT abdomen and pelvis without contrast'.",
    )


class ClinicalNote(BaseModel):
    """Ground truth label for a single dialogue."""

    name: Optional[str] = Field(
        None,
        description="Patient's full name as it appears in the reference note. "
                    "Null if absent.",
    )
    age: Optional[int] = Field(
        None,
        description="Patient's age in years. Null if absent.",
    )
    symptoms: list[str] = Field(
        default_factory=list,
        description="Patient-reported current symptoms (lowercase). "
                    "Excludes physical exam findings (those are signs).",
    )
    duration: Optional[str] = Field(
        None,
        description="Duration of the chief complaint / presenting illness. "
                    "Null for annual exams or when not stated.",
    )
    negated_symptoms: list[str] = Field(
        default_factory=list,
        description="Symptoms explicitly denied by the patient (lowercase). "
                    "Same vocabulary as `symptoms`.",
    )
    history: list[str] = Field(
        default_factory=list,
        description="Past medical history: chronic conditions, past surgeries, "
                    "etc. Chronic conditions also appear in `diagnosis` if "
                    "actively managed.",
    )
    diagnosis: list[str] = Field(
        default_factory=list,
        description="Conditions the doctor commits to a plan for in this visit. "
                    "Includes both new diagnoses and actively managed chronic "
                    "conditions.",
    )
    treatment: list[TreatmentItem] = Field(
        default_factory=list,
        description="Structured treatment plan items.",
    )


# JSON schema used in the labeler prompt
JSON_SCHEMA = ClinicalNote.model_json_schema()
