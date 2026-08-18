"""
Pre-LLM data sanitization contract (section 10).

Purpose: dataset CONTENTS are untrusted input. Once M3 introduces an
LLM planner, that LLM will read dataset values (sample values, column
names, free-text cells) as part of its context. A malicious or
adversarial dataset could contain text designed to look like an
instruction to the LLM ("ignore previous instructions...", fake
system/developer role markers, etc.) — this tool detects and
neutralizes that BEFORE any such content reaches an LLM context.

Locked separation, never violated:

    original_dataset        <- lives in DatasetStore, untouched, always
    sanitized_llm_context    <- a SEPARATE, restricted view — this is
                                what an LLM would eventually see
    sanitization_evidence    <- what was detected/altered and why

This tool NEVER mutates the dataset in DatasetStore. Its output is a
new, separate structure. A future LLM planner reads ONLY the
sanitized_llm_context, never the raw DataFrame directly.

This is explicitly defense-in-depth, not a claim of completely solving
prompt injection — stated in every report this tool produces.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SanitizationFindingType = Literal[
    "prompt_injection_pattern",
    "role_impersonation",
    "command_like_text",
    "control_characters",
    "excessive_length",
    "suspicious_encoding",
]

MAX_SAFE_TEXT_LENGTH = 500
"""
Locked truncation threshold for values entering LLM context. Not
applied to the original dataset — only to what sanitized_llm_context
carries forward.
"""


class SanitizationFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    row_index: int = Field(..., ge=0)
    finding_type: SanitizationFindingType
    original_excerpt: str = Field(
        ..., description="A short excerpt of the flagged value (not the full value, to avoid needlessly reproducing the suspicious content at length)."
    )
    action_taken: Literal["excluded", "truncated", "neutralized"]
    reason: str


class ColumnRiskClassification(BaseModel):
    """
    Every column is classified before any per-cell scanning happens,
    so low-risk categorical columns (a handful of repeated short
    values, e.g. 'Yes'/'No'/'Month-to-month') are not needlessly
    scanned cell-by-cell the same way a genuine free-text column is —
    this is what keeps legitimate business text from being
    over-sanitized.
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    risk_level: Literal["low_categorical", "business_text", "high_risk_free_text"]
    reason: str
    unique_value_ratio: float = Field(..., ge=0.0, le=1.0)


class SanitizationReport(BaseModel):
    """
    Output of sanitize_for_llm_context(). Carries the evidence AND the
    resulting sanitized context together, so a caller never has to
    reconstruct "what did we actually give the LLM" from a separate
    source.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    column_classifications: list[ColumnRiskClassification] = Field(default_factory=list)
    findings: list[SanitizationFinding] = Field(default_factory=list)
    high_risk_columns: list[str] = Field(default_factory=list)
    values_excluded_count: int = Field(..., ge=0)
    values_truncated_count: int = Field(..., ge=0)
    original_dataset_unchanged: bool = Field(
        default=True,
        description="Always True by construction — this tool never calls store.save(). Present as an explicit, checkable claim rather than an implicit assumption.",
    )
    scope_note: str = Field(
        default=(
            "This is defense-in-depth pattern detection, not a complete "
            "solution to prompt injection. It reduces risk from "
            "adversarial dataset content reaching an LLM context "
            "unfiltered; it does not guarantee detection of every "
            "possible injection technique."
        ),
    )
