"""
ExecutionTimeline (Pre-6A Polish, item 4).

A high-level phase timeline (e.g. Profile -> Plan -> Validate ->
Execute -> Train -> Evaluate -> Replan -> Complete) derived entirely
from an existing TraceEvent stream (see app/agent/timeline.py's
build_execution_timeline()). TraceEvent remains the sole source of
execution-history truth — this is a read-only, computed-on-read view
over it, never a second, competing execution-state system.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TimelinePhase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str = Field(..., description="Human-readable phase label, e.g. 'Train', 'Validate', 'Complete'.")
    node: str = Field(..., description="Raw graph-node name from the underlying TraceEvent(s), e.g. 'train'.")
    attempt: int = Field(..., ge=0, description="Which REPLAN attempt this phase occurred during (0-indexed).")
    status: Literal["success", "failure"]
    event_count: int = Field(..., ge=1, description="How many underlying TraceEvents were collapsed into this phase entry.")
    started_at: str
    ended_at: str


class ExecutionTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    phases: list[TimelinePhase] = Field(default_factory=list)
    replan_count: int = Field(..., ge=0, description="Highest attempt number observed across all events — matches AgentState.retry_count at the same point in the run.")
    final_status: Optional[str] = Field(
        default=None, description="'completed'/'failed' once a run_completed/run_failed event has been observed; None while the run is still in progress."
    )
