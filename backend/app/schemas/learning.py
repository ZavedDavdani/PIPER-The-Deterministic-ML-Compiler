"""
PIPER Learn — Schemas for Student Mode & ML Education (Phase 6).

Read-only educational schemas mapping directly to deterministic pipeline state.
No LLM dependencies, no fuzzy generation. All fields are grounded in actual execution.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.evaluation import ModelComparisonEntry

ExplanationLevel = Literal["beginner", "intermediate", "advanced"]


class FormulaEntry(BaseModel):
    """One entry in the static, curated formula library — generic and reviewed."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    formula: str
    description: str
    when_used: str


class ComprehensionCheck(BaseModel):
    """Static 'check your understanding' question and explanation."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    question: str
    answer_explanation: str
    related_concept: str


class ConceptDefinition(BaseModel):
    """Static concept definition from the registry."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    key: str
    title: str
    category: str
    summary: str
    detail: str
    related_formula: Optional[str] = None


class WhyExplanation(BaseModel):
    """Structured 'Why did PIPER do this?' explanation for an action or decision."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    action: str
    what_happened: str
    why: str
    concept: str
    alternative_consideration: Optional[str] = None
    level: ExplanationLevel = "beginner"
    evidence: dict[str, Any] = Field(default_factory=dict)


class OperationExplanation(BaseModel):
    """Explains one real OperationRecord from cleaning_log/feature_log."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    operation_id: str
    tool_name: str
    what_happened: str = Field(..., description="Beginner-friendly restatement of OperationRecord.result_summary.")
    why: str = Field(..., description="OperationRecord.reason, verbatim — real recorded rationale.")
    level: ExplanationLevel = "beginner"
    concept: Optional[str] = None
    alternative_consideration: Optional[str] = None


class ModelSelectionExplanation(BaseModel):
    """Explains state.comparison — which model was selected and why."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    recommended_model_id: str
    recommended_algorithm: str
    justification: str = Field(..., description="ModelComparison.justification, verbatim.")
    candidates: list[ModelComparisonEntry] = Field(default_factory=list)
    concept: str = "Model Selection & Metric Optimization"


class MetricExplanation(BaseModel):
    """Explains one real metric value from an EvaluationResult."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    metric: str
    value: float
    meaning: str = Field(..., description="Formula-grounded definition with real value.")
    formula: Optional[str] = None
    guidance: Optional[str] = None


class ModelConceptExplanation(BaseModel):
    """Conceptual educational overview for a model family evaluated in this run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    algorithm: str
    name: str
    concept: str
    strengths: list[str]
    tradeoffs: list[str]
    how_piper_used_it: str
    is_winner: bool = False


class EvaluationExplanation(BaseModel):
    """Explains one real EvaluationResult and baseline comparison."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    algorithm: Optional[str] = None
    metrics: list[MetricExplanation]
    confusion_matrix_meaning: str
    baseline_comparison: Optional[str] = Field(default=None)
    model_concept: Optional[ModelConceptExplanation] = None


class GuardrailCheckExplanation(BaseModel):
    """Explains one real ValidationCheck from state.validation.checks."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    check: str
    passed: bool
    severity: str
    meaning: str
    message: str
    educational_action: Optional[str] = None


class FailureExplanation(BaseModel):
    """Explains state.failure, when present."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    category: str
    message: str
    retryable: bool
    human_intervention_required: bool
    meaning: str
    educational_takeaway: Optional[str] = None


class ReplanExplanation(BaseModel):
    """Explains the REPLAN workflow and differences between attempts."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    replan_occurred: bool
    total_attempts: int
    attempts_summary: list[dict[str, Any]] = Field(default_factory=list)
    plan_differences: list[dict[str, Any]] = Field(default_factory=list)
    educational_takeaway: str


class FeatureImportanceEducation(BaseModel):
    """Educational view of feature importance with non-causal disclaimer."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    available: bool
    method: str
    algorithm: Optional[str] = None
    disclaimer: str = (
        "Feature importance shows association with model predictions; it does not prove causation."
    )
    features: list[dict[str, Any]] = Field(default_factory=list)
    educational_summary: str


class LearningJourneyStage(BaseModel):
    """One of the 14 guided ML workflow stages."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    stage_id: int
    title: str
    description: str
    status: Literal["completed", "in_progress", "failed", "not_reached", "skipped"]
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    concept: str


class LearningJourney(BaseModel):
    """Complete 14-stage guided ML learning journey for a run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str
    status: str
    current_stage_id: Optional[int] = None
    stages: list[LearningJourneyStage] = Field(default_factory=list)


class PipelineNode(BaseModel):
    """Node in the student pipeline visualization."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    name: str
    stage: str
    status: Literal["passed", "failed", "pending", "not_reached", "skipped"]
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class PipelineEdge(BaseModel):
    """Directed connection between pipeline nodes."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    from_node: str
    to_node: str


class PipelineVisualization(BaseModel):
    """Full visual flowchart data for student understanding."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str
    nodes: list[PipelineNode] = Field(default_factory=list)
    edges: list[PipelineEdge] = Field(default_factory=list)


class RunExplanation(BaseModel):
    """
    Top-level Learn-Explain bundle for one run — read-only view over
    already-computed AgentState.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str
    status: str
    level: ExplanationLevel = "beginner"
    preprocessing: list[OperationExplanation] = Field(default_factory=list)
    feature_engineering: list[OperationExplanation] = Field(default_factory=list)
    model_selection: Optional[ModelSelectionExplanation] = None
    evaluation: list[EvaluationExplanation] = Field(default_factory=list)
    guardrail_checks: list[GuardrailCheckExplanation] = Field(default_factory=list)
    failure: Optional[FailureExplanation] = None
    replan: Optional[ReplanExplanation] = None
    feature_importance: Optional[FeatureImportanceEducation] = None
    model_concepts: list[ModelConceptExplanation] = Field(default_factory=list)
