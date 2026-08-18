"""
PIPER Learn — Learn-Explain schemas (Batch 6A).

Every explanation type here is deliberately tied to the ONE real
existing schema it explains (OperationRecord, ModelComparison,
EvaluationResult, BaselineComparisonResult, ValidationCheck,
FailureInfo) rather than a generic "evidence citation" abstraction —
the field values ARE the evidence, copied by reference or lightly
templated, never fabricated or LLM-generated. See app/learning/explain.py
for the deterministic, template-based construction.

FormulaEntry/ComprehensionCheck are the two genuinely STATIC pieces
(curated once, reviewed, never per-run/per-dataset — see
app/learning/formulas.py and app/learning/comprehension.py) — they are
not tied to any specific run and carry no run evidence at all.

Locked constraints this module upholds (Batch 6A spec, CLAUDE.md):
 - Deterministic/template-based only — no LLM-generated explanations.
 - Every RunExplanation field is grounded in real evidence already
   computed elsewhere in the run — never fabricated or generic filler.
 - Read-only: nothing here can construct or mutate AgentState.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.evaluation import ModelComparisonEntry


class FormulaEntry(BaseModel):
    """One entry in the static, curated formula library — generic and
    reviewed, never generated per-run or per-dataset (locked)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    formula: str
    description: str
    when_used: str


class ComprehensionCheck(BaseModel):
    """
    Static "check your understanding" content — a question plus its
    model explanation. No grading, no scoring, no per-user state:
    presented content only (locked — Learn-Explore, Batch 6B, is the
    only place any form of guided exploration exists, and even that is
    not grading).
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    answer_explanation: str
    related_concept: str


class OperationExplanation(BaseModel):
    """Explains one real OperationRecord from cleaning_log/feature_log."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    tool_name: str
    what_happened: str = Field(..., description="Beginner-friendly restatement of OperationRecord.result_summary.")
    why: str = Field(..., description="OperationRecord.reason, verbatim — the real rationale recorded for this operation during this run.")


class ModelSelectionExplanation(BaseModel):
    """Explains state.comparison — which model was selected and why."""

    model_config = ConfigDict(extra="forbid")

    recommended_model_id: str
    recommended_algorithm: str
    justification: str = Field(..., description="ModelComparison.justification, verbatim (Pre-6A Polish item 2).")
    candidates: list[ModelComparisonEntry] = Field(default_factory=list)


class MetricExplanation(BaseModel):
    """Explains one real metric value from an EvaluationResult."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float
    meaning: str = Field(..., description="Static, formula-grounded definition of what this metric measures, with the real value plugged in.")


class EvaluationExplanation(BaseModel):
    """Explains one real EvaluationResult, plus the baseline comparison for that model when available."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    metrics: list[MetricExplanation]
    confusion_matrix_meaning: str
    baseline_comparison: Optional[str] = Field(
        default=None, description="BaselineComparisonResult.reason, verbatim, when this model_id is the one state.baseline was computed for."
    )


class GuardrailCheckExplanation(BaseModel):
    """Explains one real ValidationCheck from state.validation.checks."""

    model_config = ConfigDict(extra="forbid")

    check: str
    passed: bool
    severity: str
    meaning: str = Field(..., description="Static, generic definition of what this guardrail checks for.")
    message: str = Field(..., description="ValidationCheck.message, verbatim — the real, run-specific finding.")


class FailureExplanation(BaseModel):
    """Explains state.failure, when present."""

    model_config = ConfigDict(extra="forbid")

    category: str
    message: str
    retryable: bool
    human_intervention_required: bool
    meaning: str = Field(..., description="Static, generic definition of what this failure category means.")


class RunExplanation(BaseModel):
    """
    Top-level Learn-Explain bundle for one run — a read-only view over
    already-computed AgentState, never a new source of truth and never
    capable of influencing the run it explains.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    preprocessing: list[OperationExplanation] = Field(default_factory=list)
    feature_engineering: list[OperationExplanation] = Field(default_factory=list)
    model_selection: Optional[ModelSelectionExplanation] = None
    evaluation: list[EvaluationExplanation] = Field(default_factory=list)
    guardrail_checks: list[GuardrailCheckExplanation] = Field(default_factory=list)
    failure: Optional[FailureExplanation] = None
