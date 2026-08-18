"""
RunSummary (Pre-6A Polish, item 3).

A single, read-only, top-level aggregation of state already computed
elsewhere in a completed run — retry/REPLAN count, each candidate
model's scores, the winning model, the operations actually executed
(from cleaning_log/feature_log), and guardrail status/checks (from
PipelineValidationResult).

Deliberately NOT a new source of truth: every field here is either
copied by reference from an existing AgentState field (comparison,
validation, cleaning_log, feature_log) or a trivial derivation of one
(replanned, winning_algorithm). See app/agent/run_summary.py's
build_run_summary() for the read-only construction — nothing in this
module or that function ever mutates AgentState.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.agent.state import OperationRecord
from app.schemas.evaluation import ModelComparisonEntry
from app.schemas.guardrails import ValidationCheck


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    retry_count: int
    replanned: bool = Field(
        ..., description="True iff retry_count > 0 — at least one genuine REPLAN occurred this run."
    )

    candidate_models: list[ModelComparisonEntry] = Field(
        default_factory=list,
        description="Every candidate's metrics, copied by reference from state.comparison.models.",
    )
    winning_model_id: Optional[str] = None
    winning_algorithm: Optional[str] = None
    selection_justification: Optional[str] = Field(
        default=None,
        description="Same deterministic justification as state.comparison.justification — not recomputed here.",
    )

    operations_executed: list[OperationRecord] = Field(
        default_factory=list,
        description="state.cleaning_log + state.feature_log, in execution order.",
    )

    guardrail_valid: Optional[bool] = None
    guardrail_checks: list[ValidationCheck] = Field(default_factory=list)
    guardrail_violations: list[ValidationCheck] = Field(default_factory=list)
    guardrail_warnings: list[ValidationCheck] = Field(default_factory=list)
