"""
Failure taxonomy and structured failure information.

Locked separation (section 5 of the deterministic-foundation batch):
    status   = pipeline LIFECYCLE ("initialized"/"running"/"replanning"/
               "completed"/"failed") — see AgentState.status
    FailureInfo = structured FAILURE DETAIL, populated only when
               something actually failed — category, evidence, which
               node, which attempt, whether it's retryable, and
               whether it requires a human.

A terminal run must always have a structured final result: if
status == "failed", AgentState.failure must be populated with enough
detail to answer "why did this fail" without re-reading tool_trace by
hand. Reaching a terminal graph node is never itself treated as
success — report_node only sets status="completed" when
validation.valid is actually True (this was already true before this
module existed; FailureInfo makes the failure side of that same
discipline explicit and structured).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# --- Failure taxonomy ------------------------------------------------------

TerminalFailureCategory = Literal[
    "DATA_ERROR",
    "SCHEMA_ERROR",
    "TARGET_ERROR",
    "EXECUTION_BUDGET_EXCEEDED",
]
"""
Terminal categories bypass the retry loop entirely — these are not
"the plan was bad," they're "the input or environment makes this run
impossible to complete," so retrying with a different plan cannot
help. Constraint #9 (non-binary/unresolvable task_type) is
TARGET_ERROR. Malformed/empty/zero-column datasets are DATA_ERROR or
SCHEMA_ERROR. Exceeding a hard execution budget (e.g. recursion limit)
is EXECUTION_BUDGET_EXCEEDED.
"""

RecoverableFailureCategory = Literal[
    "LEAKAGE_ERROR",
    "IMBALANCE_ERROR",
    "FEATURE_ERROR",
    "TRAINING_ERROR",
    "EVALUATION_ERROR",
    "DUPLICATE_PLAN",
    "BASELINE_GATE_FAILED",
    "PLAN_ADEQUACY",
]
"""
Recoverable categories MAY trigger REPLAN if retry_count < max_retries
— but "may" is doing real work here: a category being in this list
means the KIND of failure is, in principle, addressable by a different
plan (e.g. dropping a leaky feature next attempt). Whether a SPECIFIC
failure instance actually triggers REPLAN is a runtime decision made
by the graph router (_route_after_validate), which checks
retry_count < max_retries — not a property of the category alone.

`retryable` on a given FailureInfo instance describes whether this
category of failure is retryable in principle (mirrors
category-in-RECOVERABLE_CATEGORIES); it does NOT mean a retry is still
available right now — `human_intervention_required` is what flips to
True once the retry budget is actually exhausted (see report_node in
graph.py), even though `retryable` stays True (the failure was, and
remains, the KIND of thing that could have been fixed by replanning —
the budget simply ran out).
"""

FailureCategory = Literal[
    "DATA_ERROR",
    "SCHEMA_ERROR",
    "TARGET_ERROR",
    "EXECUTION_BUDGET_EXCEEDED",
    "LEAKAGE_ERROR",
    "IMBALANCE_ERROR",
    "FEATURE_ERROR",
    "TRAINING_ERROR",
    "EVALUATION_ERROR",
    "DUPLICATE_PLAN",
    "BASELINE_GATE_FAILED",
    "PLAN_ADEQUACY",
]

TERMINAL_CATEGORIES: frozenset = frozenset(
    {"DATA_ERROR", "SCHEMA_ERROR", "TARGET_ERROR", "EXECUTION_BUDGET_EXCEEDED"}
)
RECOVERABLE_CATEGORIES: frozenset = frozenset(
    {
        "LEAKAGE_ERROR", "IMBALANCE_ERROR", "FEATURE_ERROR", "TRAINING_ERROR",
        "EVALUATION_ERROR", "DUPLICATE_PLAN", "BASELINE_GATE_FAILED",
        "PLAN_ADEQUACY",
    }
)


class FailureInfo(BaseModel):
    """
    Structured failure detail. Populated whenever a node fails —
    whether terminally or in a way that may trigger REPLAN. This is
    what makes report_node's terminal message explainable rather than
    a bare "pipeline failed" string.
    """

    model_config = ConfigDict(extra="forbid")

    category: FailureCategory
    message: str
    evidence: dict = Field(
        default_factory=dict,
        description="Structured evidence backing the failure — e.g. the LeakageReport violations, or the exact retry_count/max_retries at exhaustion.",
    )
    node: str = Field(..., description="Which graph node produced this failure, e.g. 'validate', 'profile'.")
    attempt: int = Field(..., ge=0, description="Which attempt (0-indexed, matches retry_count at the time of failure) this occurred on.")
    retryable: bool = Field(
        ..., description="Whether THIS specific failure instance may trigger REPLAN, per the recoverability policy — not merely whether its category is in RECOVERABLE_CATEGORIES."
    )
    human_intervention_required: bool = Field(
        default=False,
        description="True once retries are exhausted on a recoverable failure, or immediately for any terminal-category failure.",
    )

    def model_post_init(self, __context) -> None:
        # A terminal-category failure is never retryable, regardless of
        # what the caller passed — enforced here so this invariant
        # can't be violated by a call-site mistake.
        if self.category in TERMINAL_CATEGORIES and self.retryable:
            object.__setattr__(self, "retryable", False)
