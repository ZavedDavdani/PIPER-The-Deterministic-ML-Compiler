"""
V1.2 Batch 1 — productization view models.

Read-only presentation of evidence PIPER already has (trace events,
planning attempts, plan diffs, validation, adequacy, guardrails).
None of these models are consulted by graph routing, validation, or
execution. They never include LLM chain-of-thought / reasoning text.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.adequacy import AdequacyFinding
from app.schemas.baseline import BaselineComparisonResult
from app.schemas.evaluation import EvaluationResult, ModelComparison
from app.schemas.execution_timeline import ExecutionTimeline
from app.schemas.failure import FailureInfo
from app.schemas.guardrails import PipelineValidationResult
from app.schemas.reproducibility import ReproducibilityMetadata
from app.schemas.training import TrainingResult

StageId = Literal[
    "LLM_PROPOSED",
    "VALIDATED",
    "ADEQUACY",
    "REPLAN",
    "EXECUTION",
    "TRAINING",
    "EVALUATION",
    "GUARDRAILS",
    "FINAL_VERDICT",
]

StageStatus = Literal["pending", "current", "passed", "failed", "skipped", "not_reached"]

PlanningOutcome = Literal[
    "provider_error",
    "invalid",
    "duplicate",
    "inadequate",
    "accepted",
]

VerdictOutcome = Literal[
    "ACCEPTED",
    "REJECTED",
    "HUMAN_INTERVENTION_REQUIRED",
]


class ExecutableStep(BaseModel):
    """tool_name + arguments only — never reasoning."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict = Field(default_factory=dict)


class PlanningAttempt(BaseModel):
    """One planner call's executable proposal and deterministic gates."""

    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(..., ge=0)
    proposed_steps: list[ExecutableStep] = Field(default_factory=list)
    plan_hash: Optional[str] = None
    structurally_valid: bool
    adequacy_status: Optional[Literal["PASS", "FAIL"]] = None
    outcome: PlanningOutcome
    violation_count: int = 0
    material_finding_count: int = 0


class DecisionStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StageId
    label: str
    status: StageStatus
    summary: str
    attempt: Optional[int] = None
    evidence: dict = Field(default_factory=dict)


class DecisionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_status: str
    stages: list[DecisionStage]
    planning_attempts: list[PlanningAttempt] = Field(default_factory=list)
    plan_diffs: list[dict] = Field(default_factory=list)


class PiperVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    outcome: VerdictOutcome
    reason_code: str
    summary: str
    retry_count: int
    max_retries: int
    structurally_valid_plan: bool
    adequacy_passed: Optional[bool] = None
    guardrails_passed: Optional[bool] = None
    human_intervention_required: bool
    executed: bool


class HumanInterventionPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    required: bool
    headline: str
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None
    retry_count: int
    max_retries: int
    last_proposed_steps: list[ExecutableStep] = Field(default_factory=list)
    structural_violations: list[dict] = Field(default_factory=list)
    material_adequacy_findings: list[AdequacyFinding] = Field(default_factory=list)
    advisory_adequacy_findings: list[AdequacyFinding] = Field(default_factory=list)
    preserved_valid_steps: list[ExecutableStep] = Field(default_factory=list)
    implicated_steps: list[ExecutableStep] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    blocked_invalid_execution: bool = True


class EvidenceExport(BaseModel):
    """Single JSON artifact for a terminal (or in-progress) run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["piper.evidence.v1"] = "piper.evidence.v1"
    run_id: str
    dataset_id: str
    target_column: str
    status: str
    decision_trace: DecisionTrace
    verdict: Optional[PiperVerdict] = None
    intervention: HumanInterventionPackage
    summary: Optional[dict] = None
    timeline: ExecutionTimeline
    validation: Optional[PipelineValidationResult] = None
    comparison: Optional[ModelComparison] = None
    baseline: Optional[BaselineComparisonResult] = None
    failure: Optional[FailureInfo] = None
    reproducibility: Optional[ReproducibilityMetadata] = None
    model_results: list[TrainingResult] = Field(default_factory=list)
    evaluation_results: list[EvaluationResult] = Field(default_factory=list)
    executed_operations: list[ExecutableStep] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReplayResponse(BaseModel):
    """Audit/replay of a stored run. Never invokes the LLM."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    llm_invoked: Literal[False] = False
    source: Literal["persisted_events_and_state"] = "persisted_events_and_state"
    status: str
    decision_trace: DecisionTrace
    verdict: Optional[PiperVerdict] = None
    intervention: HumanInterventionPackage
    evidence: EvidenceExport
