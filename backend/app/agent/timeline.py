"""
build_execution_timeline() (Pre-6A Polish, item 4).

Derives a high-level phase timeline purely from an existing TraceEvent
stream (run_store.get_events(run_id)) — TraceEvent remains the sole
source of execution-history truth; this is a read-only view over it,
computed fresh on every call, never a second, competing execution-
state system. See ExecutionTimeline's own docstring for the schema.

Works against both live per-node events (node_started/node_completed/
node_failed, produced by stream_with_tracing() as the graph actually
executes) and the post-hoc tool-call-level events derived from
tool_trace (produced by both stream_with_tracing() and
run_with_tracing() — see tracing.py's _tool_trace_events()) — the two
event sources use different node-name vocabularies for the same
conceptual phase (e.g. "train" vs. "trainer"), both covered by
_PHASE_LABELS below so the timeline reads the same regardless of which
tracing function produced the run's events.
"""

from __future__ import annotations

from app.schemas.execution_timeline import ExecutionTimeline, TimelinePhase
from app.schemas.trace_event import TraceEvent

_PHASE_LABELS: dict[str, str] = {
    "validate_input": "Validate Input",
    "input_validator": "Validate Input",
    "profile": "Profile",
    "profiler": "Profile",
    "sanitize": "Sanitize",
    "sanitizer": "Sanitize",
    "plan_entry": "Plan",
    "plan": "Plan",
    "clean": "Clean",
    "cleaner": "Clean",
    "feature_engineer": "Feature Engineer",
    "split": "Split",
    "splitter": "Split",
    "reproducibility": "Reproducibility",
    "train": "Train",
    "trainer": "Train",
    "evaluate": "Evaluate",
    "evaluator": "Evaluate",
    "compare": "Compare",
    "comparator": "Compare",
    "baseline": "Baseline",
    "validate": "Validate",
    "validator": "Validate",
    "report": "Report",
}

_TERMINAL_EVENT_TYPES = {"run_completed", "run_failed"}


def _phase_label(node: str) -> str:
    return _PHASE_LABELS.get(node, node.replace("_", " ").title())


def build_execution_timeline(run_id: str, events: list[TraceEvent]) -> ExecutionTimeline:
    """
    Collapses consecutive events that map to the same high-level phase
    label AND the same attempt number into one TimelinePhase entry (a
    phase is reported as "failure" if ANY collapsed event failed).
    replan_count is the highest `attempt` observed across all events —
    matches AgentState.retry_count at the same point in the run, since
    every TraceEvent's `attempt` field mirrors retry_count when it was
    produced.
    """
    phases: list[TimelinePhase] = []
    current: dict | None = None
    max_attempt = 0
    final_status: str | None = None

    for event in events:
        max_attempt = max(max_attempt, event.attempt)

        if event.event_type in _TERMINAL_EVENT_TYPES:
            final_status = "completed" if event.event_type == "run_completed" else "failed"
            label = "Complete" if final_status == "completed" else "Failed"
        else:
            label = _phase_label(event.node)

        event_status = "failure" if event.status == "failure" else "success"

        if current is not None and current["phase"] == label and current["attempt"] == event.attempt:
            current["event_count"] += 1
            current["ended_at"] = event.timestamp
            if event_status == "failure":
                current["status"] = "failure"
        else:
            if current is not None:
                phases.append(TimelinePhase(**current))
            current = {
                "phase": label,
                "node": event.node,
                "attempt": event.attempt,
                "status": event_status,
                "event_count": 1,
                "started_at": event.timestamp,
                "ended_at": event.timestamp,
            }

    if current is not None:
        phases.append(TimelinePhase(**current))

    return ExecutionTimeline(
        run_id=run_id,
        phases=phases,
        replan_count=max_attempt,
        final_status=final_status,
    )
