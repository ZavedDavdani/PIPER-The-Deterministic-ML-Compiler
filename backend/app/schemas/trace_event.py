"""
TraceEvent (Phase 5 of the deterministic-core completion batch).

A standardized, UI-independent record of one thing that happened
during a run — a tool call, a guardrail check, a node transition, a
failure. This is NOT the same as ToolTraceEntry (AgentState's existing
per-tool-call log, used by CLEAN/FEATURE_ENGINEER/etc.) — TraceEvent is
a broader, run-level event stream intended to support the future M5
observability layer (SSE, a "PIPER Control Room" UI), grouped by
attempt and structured with parent/child spans, whereas
ToolTraceEntry is narrower (one entry per deterministic tool call
inside AgentState, already used by the graph nodes).

No SSE, no frontend, no UI coupling here — this is purely the
structured data foundation. RunStore (Phase 5's other half) is what
actually accumulates these events per run.

Hierarchy example (attempt grouping):

    RUN run_001
      ATTEMPT 0
        PROFILE
        CLEAN
        TRAIN
        EVALUATE
        VALIDATE
        (REPLAN decision)
      ATTEMPT 1
        PLAN
        (DUPLICATE_PLAN)

parent_span_id lets an event reference the event that logically
contains it (e.g. a guardrail-check event's parent could be the
VALIDATE node's own event) — kept as a plain optional string ID
rather than a nested structure, so events remain flat, independently
serializable records (an SSE-friendly shape), with hierarchy
reconstructed by consumers via parent_span_id lookups, not by nesting.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

TraceEventType = Literal[
    "node_started",
    "node_completed",
    "node_failed",
    "tool_called",
    "guardrail_checked",
    "validation_result",
    "replan_triggered",
    "duplicate_plan_detected",
    "retry_exhausted",
    "run_completed",
    "run_failed",
]

TraceEventStatus = Literal["success", "failure", "info"]
TraceEventSeverity = Literal["info", "warning", "error"]


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    step_id: str = Field(..., description="Unique ID for this specific event, e.g. 'trace_a1b2c3d4'.")
    parent_span_id: Optional[str] = Field(
        None, description="step_id of the logically-containing event, if any. None for top-level (e.g. node-level) events."
    )
    attempt: int = Field(..., ge=0, description="Which REPLAN attempt this event occurred during (0-indexed, matches retry_count at the time).")
    node: str = Field(..., description="Which graph node produced this event, e.g. 'validate', 'train'.")
    event_type: TraceEventType
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")
    status: TraceEventStatus
    severity: TraceEventSeverity = "info"
    tool_name: Optional[str] = None
    guardrail_name: Optional[str] = None
    evidence: dict = Field(default_factory=dict)
    duration_ms: float = Field(default=0.0, ge=0.0)
    message: str = ""
