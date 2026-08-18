"""
AgentState — the shared state carried through the LangGraph graph.

This is NOT part of M1's tool contract; it is the state contract
locked separately, before LangGraph was introduced. It is being coded
now, at the start of M2, because it's the type every graph node will
read and write.

Verified compatible with langgraph==1.2.10: a Pydantic BaseModel works
directly as StateGraph's state_schema (see backend/tests/test_state.py
for the equivalent proof at the AgentState level, and the ad-hoc
TinyState check that preceded this file). No custom serialization or
checkpointing is introduced here — plain Pydantic defaults are enough
for M2's scope.

Ten constraints this file must uphold (locked before writing):
 1. No raw DataFrames in AgentState.
 2. No fitted scikit-learn model objects in state.
 3. Models remain referenced through model_id only.
 4. dataset_id and run_id are references, not the underlying data.
 5. State must be serializable/structured (Pydantic — LangGraph carries it).
 6. max_retries defaults to 2.
 7. retry_count starts at 0.
 8. task_type remains restricted to "binary_classification".
 9. Invalid task type -> eventually "failed", never replanned.
10. No database persistence or extra state-management infra introduced here.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# --- Sub-schemas -------------------------------------------------------

TaskType = Literal["binary_classification"]

AgentStatus = Literal["initialized", "running", "completed", "failed", "replanning"]

PlanStepStatus = Literal["pending", "executing", "completed", "failed", "skipped"]


class PlanStep(BaseModel):
    """
    One entry in AgentState.plan — what the agent INTENDS to do.
    Distinct from OperationRecord (what actually happened) and
    ToolTraceEntry (raw execution record for SSE), per the locked
    three-log design.
    """

    model_config = ConfigDict(extra="forbid")

    step_id: str
    action: str = Field(..., description="Human-readable description, e.g. 'Impute missing MonthlyCharges'.")
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    reasoning: str = Field(..., description="Why the agent chose this step.")
    status: PlanStepStatus = "pending"


class OperationRecord(BaseModel):
    """
    One entry in cleaning_log or feature_log — what ACTUALLY changed.
    Kept separate from PlanStep because a planned step can fail; this
    only exists for operations that actually executed.
    """

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    result_summary: str
    reason: str
    timestamp: str = Field(..., description="ISO 8601 timestamp string — kept as str, not datetime, for trivial JSON/SSE serialization.")


class ToolTraceEntry(BaseModel):
    """
    One entry in tool_trace — the raw execution record every SSE event
    will be built from later (M5). This is the only log that always
    exists, success or failure, for every tool call.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    timestamp: str
    node: str = Field(..., description="Which graph node executed this call, e.g. 'cleaner'.")
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict, description="Compact summary, not the full ToolResult payload.")
    success: bool
    duration_ms: float


from app.schemas.training import TrainingResult
from app.schemas.evaluation import EvaluationResult, ModelComparison
from app.schemas.guardrails import PipelineValidationResult
from app.schemas.failure import FailureInfo
from app.schemas.baseline import BaselineComparisonResult
from app.schemas.sanitization import SanitizationReport
from app.schemas.reproducibility import ReproducibilityMetadata


# NOTE ON THIS REPLACEMENT (real graph integration, post-guardrails):
# AgentState originally defined its OWN lighter ModelResult,
# EvaluationResult, ModelComparison, PipelineValidationCheck, and
# PipelineValidationResult classes, built before the real tools
# existed. Now that train_model()/evaluate_model()/compare_models()/
# validate_pipeline() are real and return their own fuller schemas
# (TrainingResult from schemas/training.py; EvaluationResult and
# ModelComparison from schemas/evaluation.py; PipelineValidationResult
# from schemas/guardrails.py, which already contains violations,
# warnings, and scope_note internally), carrying two differently-
# shaped classes with overlapping names would be a real architectural
# smell — so AgentState now imports and uses the real schemas
# directly, rather than translating between two shapes at every node
# boundary. This was an explicit, deliberate decision (not a default),
# made because the alternative (keeping AgentState's own lighter
# types) would require a translation step in every node and risked
# the two schemas silently drifting apart over time.
#
# One consequence: AgentState.model_results is now list[TrainingResult]
# (not a custom ModelResult) — TrainingResult already satisfies
# constraints #1/#2/#3 (references only, no fitted objects, model_id
# as the only pointer into ModelStore), so nothing about those locked
# constraints changes, only which class expresses them.
#
# The standalone leakage_report/imbalance_report/constant_feature_report/
# high_cardinality_report placeholder fields (dict-typed, added when
# guardrails didn't exist yet) are REMOVED: the real
# PipelineValidationResult already carries all guardrail evidence
# internally via its `checks`/`violations`/`warnings` lists, so
# separate top-level fields for each guardrail's raw report would be
# duplicate state, not new information. If a node needs the full,
# untyped LeakageReport/ImbalanceReport/etc. (e.g. for a richer trace
# entry), it can call the guardrail tool directly and put a summary in
# tool_trace — AgentState itself doesn't need to hold the full report.


class GuardrailReport(dict):
    """
    Retained only as a type alias for backward-compatible imports
    elsewhere in the codebase that may reference it; AgentState no
    longer uses this for a dedicated field (see note above). Kept as
    a thin dict subclass rather than removed outright to avoid an
    import break if anything still references
    `from app.agent.state import GuardrailReport`.
    """

    pass


# --- AgentState ----------------------------------------------------------


class AgentState(BaseModel):
    """
    The shared state carried through every LangGraph node.

    Holds only references (run_id, dataset_id, model_id) and
    structured results — never a raw DataFrame, never a fitted
    scikit-learn model object. The dataset lives in DatasetStore; a
    trained model lives in ModelStore.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Identity ---
    run_id: str
    dataset_id: str

    # --- Task ---
    target_column: str
    task_type: Optional[TaskType] = None
    """
    None until resolved. If resolution determines the objective is NOT
    binary classification (e.g. a regression or multiclass target),
    task_type stays None and status must go straight to "failed" with
    a clear error — never "replanning". This is a scope violation, not
    a recoverable agent mistake (constraint #9).
    """

    # --- Observation ---
    profile: Optional[dict] = Field(
        default=None,
        description="DatasetProfile.model_dump() once profile_dataset() runs. Stored as dict here to avoid a circular import between state.py and the profiling schema module; graph wiring re-validates into DatasetProfile at the point of use.",
    )
    sanitization_report: Optional[SanitizationReport] = None
    """
    Section 10: evidence of what was detected/neutralized before any
    future LLM context is built. The original dataset in DatasetStore
    is never touched by sanitization — this report is purely
    informational evidence carried alongside the run.
    """

    # --- Planning ---
    plan: list[PlanStep] = Field(default_factory=list)
    plan_history: list[str] = Field(
        default_factory=list,
        description="SHA-256 canonical plan hashes (see agent/plan_canonical.py) of every plan PROPOSED so far in this run, in order — whether it passed deterministic validation or was rejected as invalid. Used by the REPLAN loop to detect and reject a duplicate executable plan (DUPLICATE_PLAN, post-validation) or a duplicate already-rejected proposal (also DUPLICATE_PLAN, pre-validation — see plan_node_v2) before wasting a retry repeating something already known to fail.",
    )

    # --- Execution logs (three distinct logs, per locked design) ---
    cleaning_log: list[OperationRecord] = Field(default_factory=list)
    feature_log: list[OperationRecord] = Field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)

    # --- ML state ---
    split_id: Optional[str] = None
    model_results: list[TrainingResult] = Field(default_factory=list)
    evaluation_results: list[EvaluationResult] = Field(default_factory=list)
    comparison: Optional[ModelComparison] = None
    reproducibility: Optional[ReproducibilityMetadata] = None
    """
    Environment versions, dataset content fingerprint, and split/model
    random-state configuration for this run — populated by
    reproducibility_node immediately after SPLIT. None until that node
    runs; survives unchanged through the rest of the graph to the
    final result.
    """
    baseline: Optional[BaselineComparisonResult] = None
    """
    Majority-class baseline comparison (section 8) for the most
    recently evaluated model — computed on the SAME split, consumed
    by validate_pipeline()'s baseline gate.
    """

    # --- Validation / guardrails ---
    # The real PipelineValidationResult (schemas/guardrails.py) already
    # carries all guardrail evidence (violations, warnings, scope_note)
    # internally — no separate per-guardrail report fields are needed
    # here (see module-level note above for why they were removed).
    validation: Optional[PipelineValidationResult] = None

    # --- Control flow ---
    retry_count: int = 0          # constraint #7
    max_retries: int = 2          # constraint #6
    steps_executed: int = 0
    """
    M4: PIPER-owned execution-step counter, incremented once per graph
    node execution (see graph.py's node-wrapping in build_graph()).
    Entirely independent of LangGraph's own internal recursion_limit
    and of max_retries — exists so the graph's termination guarantee
    ("must always reach a terminal state, never hang" — already
    asserted by tests/test_ollama_integration.py) is enforced by
    PIPER's own deterministic logic, not merely by an externally
    supplied recursion_limit that would otherwise raise an uncaught
    GraphRecursionError if max_retries were ever configured
    unreasonably high. See graph.py's MAX_EXECUTION_STEPS and
    _increment_retry_if_replanning for the actual enforcement point.
    """
    status: AgentStatus = "initialized"
    error: Optional[str] = None
    """
    Legacy free-text error message — kept for backward compatibility
    with earlier tests and for a quick human-readable summary. `failure`
    (below) is the structured, authoritative failure detail; `error`
    should generally just be `failure.message` when `failure` is set.
    """
    failure: Optional[FailureInfo] = None
    """
    Structured failure detail (section 5/6): category, evidence, which
    node/attempt, retryable, human_intervention_required. Populated
    whenever status becomes "failed", and also populated on a
    recoverable failure that triggers REPLAN (so the plan-diff/retry
    history has structured evidence for each attempt, not just the
    final one).
    """
