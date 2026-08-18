"""
run_with_tracing() (Phase 5 integration) / stream_with_tracing() (M5
live-progress extension).

Connects TraceEvent/InMemoryRunStore to REAL graph execution, without
threading a RunStore argument through every individual node function
(which would touch every node in graph.py/real_nodes.py and meaningfully
increase risk on an already-stable graph for a feature that was
originally foundation-only, no UI/SSE yet).

Instead: run_with_tracing() wraps graph.invoke(), then derives
TraceEvents from the final AgentState's EXISTING tool_trace
(ToolTraceEntry list, already populated by every real node) plus
plan_history (to reconstruct attempt boundaries). This is a legitimate,
lower-risk way to make "the events usable by an observability layer"
true, without destabilizing the deterministic core to do it. It is
POST-HOC, though: run_store isn't populated until the entire run has
already finished — fine for a final trace/results view, but useless
for genuinely live progress.

stream_with_tracing() (M5) is the live counterpart: it drives the
SAME graph via graph.stream(..., stream_mode="updates") instead of
graph.invoke(), so a TraceEvent (and a live run_store.update()) is
appended immediately after EACH node finishes, in real time, while the
run is still executing. This is what the API layer's SSE
/runs/{run_id}/events endpoint and "live" GET /runs/{run_id} status
are actually built on. It still ALSO appends the same tool-call-level
detail run_with_tracing() derives (via the shared helper below), so a
streamed run ends up exactly as rich as a non-streamed one — live
progress is additive, not a replacement.

Attempt reconstruction (tool-call-level events only): since tool_trace
entries accumulate strictly in execution order, and a new attempt
always begins with a fresh CLEAN phase immediately after a prior
VALIDATE/BASELINE phase, attempt boundaries are reconstructed by
walking tool_trace and incrementing the attempt counter each time a
'cleaner' node entry immediately follows a prior 'validator'/'baseline'
node entry. This is a best-effort reconstruction, documented as such;
exact per-event attempt tags would require per-node instrumentation,
deliberately out of scope for this pass. Live per-node events (from
stream_with_tracing() only) don't need this reconstruction — they read
the actual, current retry_count directly off the accumulated state.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.schemas.trace_event import TraceEvent
from app.storage.run_store import InMemoryRunStore

logger = logging.getLogger(__name__)

_ATTEMPT_BOUNDARY_NODE = "cleaner"
_PRIOR_PHASE_NODES = {"validator", "baseline"}


def _new_step_id() -> str:
    return f"trace_{uuid.uuid4().hex[:8]}"


class _RunResultState:
    """
    Lightweight shim so run_store.update() can read the fields it
    needs via getattr() without requiring a full AgentState
    reconstruction from a result dict.

    Carries a few fields beyond what update() itself reads
    (validation/comparison/baseline/failure/model_results/
    evaluation_results/reproducibility/error/cleaning_log/feature_log)
    — update() ignores attributes it doesn't need, but the API layer
    (M5) reads these off record.final_state once a run reaches a
    terminal status, to build a real GET /runs/{run_id}/result (and,
    Pre-6A Polish, GET /runs/{run_id}/summary) response, so they need
    to actually be present here, not just the handful update()
    consults.
    """

    def __init__(self, d: dict):
        self.dataset_id = d.get("dataset_id", "")
        self.target_column = d.get("target_column", "")
        self.status = d.get("status", "unknown")
        self.retry_count = d.get("retry_count", 0)
        self.plan_history = d.get("plan_history", [])
        self.validation = d.get("validation")
        self.comparison = d.get("comparison")
        self.baseline = d.get("baseline")
        self.failure = d.get("failure")
        self.model_results = d.get("model_results", [])
        self.evaluation_results = d.get("evaluation_results", [])
        self.reproducibility = d.get("reproducibility")
        self.error = d.get("error")
        self.cleaning_log = d.get("cleaning_log", [])
        self.feature_log = d.get("feature_log", [])


def _tool_trace_events(run_id: str, tool_trace: list) -> tuple[list[TraceEvent], int]:
    """
    Derives TraceEvents from an AgentState's tool_trace (ToolTraceEntry
    list), reconstructing attempt boundaries per the module docstring.
    Returns (events, final_attempt_number) so callers needing the
    attempt count for a subsequent summary event don't recompute it.
    """
    events: list[TraceEvent] = []
    attempt = 0
    prior_node = None

    for entry in tool_trace:
        if entry.node == _ATTEMPT_BOUNDARY_NODE and prior_node in _PRIOR_PHASE_NODES:
            attempt += 1
        prior_node = entry.node

        events.append(
            TraceEvent(
                run_id=run_id,
                step_id=_new_step_id(),
                attempt=attempt,
                node=entry.node,
                event_type="tool_called",
                timestamp=entry.timestamp,
                status="success" if entry.success else "failure",
                severity="info" if entry.success else "error",
                tool_name=entry.tool_name,
                evidence=entry.result,
                duration_ms=entry.duration_ms,
                message=entry.result.get("message", "") if isinstance(entry.result, dict) else "",
            )
        )

    return events, attempt


def _final_summary_event(run_id: str, attempt: int, final_status: str, error: str) -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        step_id=_new_step_id(),
        attempt=attempt,
        node="report",
        event_type="run_completed" if final_status == "completed" else "run_failed",
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="success" if final_status == "completed" else "failure",
        severity="info" if final_status == "completed" else "error",
        evidence={"final_status": final_status},
        message=error or "",
    )


def run_with_tracing(graph, initial_state, run_store: InMemoryRunStore, config=None):
    """
    Invokes the compiled graph exactly as graph.invoke() would, then
    populates run_store with a real RunRecord and a real, derived
    TraceEvent stream for this run. Returns the same result dict
    graph.invoke() would have returned — this wrapper is purely
    additive, never changes execution behavior.
    """
    run_id = initial_state.run_id
    run_store.create(run_id, initial_state)

    try:
        result = graph.invoke(initial_state, config=config or {})
    except Exception:
        # Batch 5 fix: an unexpected exception (e.g. an sklearn fit()
        # failure, or any other bug not already converted to a
        # structured ToolResult/FailureInfo somewhere in the node
        # chain) previously propagated straight out of this function
        # uncaught. Nothing downstream ever called run_store.update()
        # with a terminal status in that case, so the run stayed
        # "running" in RunStore forever — GET /runs/{id}/result would
        # 409 indefinitely and a caller polling GET /runs/{id} would
        # never see a terminal state. Converting this to a genuine
        # terminal "failed" result (logged, not silently swallowed) is
        # purely additive: it never changes the outcome of a run that
        # doesn't hit this path.
        logger.exception("Unexpected error during graph execution for run %s", run_id)
        error_result = {"status": "failed", "error": "Unexpected internal error during execution — see server logs."}
        run_store.update(run_id, _RunResultState(error_result))
        run_store.append_event(run_id, _final_summary_event(run_id, 0, "failed", error_result["error"]))
        return error_result

    tool_trace = result.get("tool_trace", [])
    events, attempt = _tool_trace_events(run_id, tool_trace)
    for event in events:
        run_store.append_event(run_id, event)

    final_status = result.get("status", "unknown")
    final_event = _final_summary_event(
        run_id, result.get("retry_count", attempt), final_status, result.get("error") or ""
    )
    run_store.append_event(run_id, final_event)

    run_store.update(run_id, _RunResultState(result))

    return result


def stream_with_tracing(graph, initial_state, run_store: InMemoryRunStore, config=None):
    """
    M5 live-progress counterpart to run_with_tracing() — see module
    docstring for the full rationale. Drives the compiled graph via
    graph.stream(..., stream_mode="updates"), which yields one
    {node_name: partial_update_dict} per completed node — the SAME
    partial dicts every node function in real_nodes.py/graph.py
    already returns; no node internals are touched by this function.

    Locally accumulates those partial updates into `merged` via plain
    dict.update() per chunk — correct because AgentState uses no
    custom reducers, so every field is "last write wins", exactly
    matching how LangGraph itself merges state internally. This lets
    each live event report accurate attempt/status context, and lets
    this function hand back a complete, graph.invoke()-shaped result
    dict at the end.

    Intended to be called from a background thread/task by the API
    layer (a real run can take minutes with a real LLM provider) — this
    function itself is fully synchronous/blocking, matching
    run_with_tracing()'s existing contract.
    """
    run_id = initial_state.run_id
    run_store.create(run_id, initial_state)

    merged: dict = {}

    try:
        for chunk in graph.stream(initial_state, config=config or {}, stream_mode="updates"):
            node_name, update = next(iter(chunk.items()))
            merged.update(update)

            node_failed = update.get("status") == "failed"
            live_event = TraceEvent(
                run_id=run_id,
                step_id=_new_step_id(),
                attempt=merged.get("retry_count", 0),
                node=node_name,
                event_type="node_failed" if node_failed else "node_completed",
                timestamp=datetime.now(timezone.utc).isoformat(),
                status="failure" if node_failed else "success",
                severity="error" if node_failed else "info",
                evidence={"updated_fields": sorted(update.keys())},
                message=update.get("error") or "",
            )
            run_store.append_event(run_id, live_event)
            run_store.update(run_id, _RunResultState(merged))
    except Exception:
        # Batch 5 fix — see run_with_tracing()'s matching except block
        # for the full rationale. This is the path actually reachable
        # via the live API (POST /runs uses stream_with_tracing(), not
        # run_with_tracing()): without this, an unexpected exception
        # mid-run would leave RunStore permanently stuck at "running"
        # — GET /runs/{id}/result stuck at 409 forever, and the SSE
        # endpoint (GET /runs/{id}/events) looping forever waiting for
        # a terminal status that will never arrive.
        logger.exception("Unexpected error during graph execution for run %s", run_id)
        merged["status"] = "failed"
        merged["error"] = "Unexpected internal error during execution — see server logs."
        run_store.update(run_id, _RunResultState(merged))
        run_store.append_event(
            run_id, _final_summary_event(run_id, merged.get("retry_count", 0), "failed", merged["error"])
        )
        return merged

    tool_trace = merged.get("tool_trace", [])
    events, attempt = _tool_trace_events(run_id, tool_trace)
    for event in events:
        run_store.append_event(run_id, event)

    final_status = merged.get("status", "unknown")
    final_event = _final_summary_event(
        run_id, merged.get("retry_count", attempt), final_status, merged.get("error") or ""
    )
    run_store.append_event(run_id, final_event)

    run_store.update(run_id, _RunResultState(merged))

    return merged
