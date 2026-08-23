"""
InMemoryRunStore (Phase 5). Real implementation of the RunStore
interface (run_and_model_store.py) — in-memory only, no database, no
persistence across process restarts, no SSE/UI coupling. This mirrors
DatasetStore/SplitStore/ModelStore's existing in-memory pattern.

Records, per run_id:
    - run metadata (dataset_id, target_column, created_at)
    - current status / current node
    - attempt number (mirrors AgentState.retry_count)
    - plan history (mirrors AgentState.plan_history)
    - a flat list of TraceEvents (the Phase 5 event stream)
    - the final AgentState snapshot, once the run terminates

Deliberately NOT tightly coupled to individual tools — nodes call
append_event()/set_final_state() explicitly; RunStore itself has no
knowledge of what a "tool" or "guardrail" is, it just stores whatever
TraceEvents are given to it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.trace_event import TraceEvent
from app.storage.exceptions import RunNotFoundError
from app.storage.run_and_model_store import RunStore

__all__ = ["InMemoryRunStore", "RunRecord", "RunNotFoundError"]
"""
RunNotFoundError is re-exported here (not redefined) so
`from app.storage.run_store import RunNotFoundError` keeps working for
existing callers/tests, while resolving to the SAME canonical class
`app.storage.exceptions.RunNotFoundError` that every other store
(DatasetStore, ModelStore, SplitStore) already raises and that
`app.storage`'s package-level exports use. This module previously
defined its own separate, identically-shaped RunNotFoundError class —
a genuine bug: code catching `app.storage.RunNotFoundError` (the
canonical one) around an InMemoryRunStore call would NOT have caught
what InMemoryRunStore actually raised, since the two classes were
unrelated despite the identical name and message. Fixed by collapsing
onto the one canonical class.
"""


class RunRecord(BaseModel):
    """
    Everything InMemoryRunStore tracks for one run. final_state is
    left untyped (Any-like via arbitrary_types_allowed) intentionally
    — AgentState lives in agent/state.py, and importing it here would
    create app.storage -> app.agent -> app.storage circularity given
    how AgentState's own imports are structured; this keeps the
    storage layer decoupled from AgentState's specific shape.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    dataset_id: str
    target_column: str
    status: str = "initialized"
    current_node: Optional[str] = None
    attempt: int = 0
    plan_history: list[str] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    final_state: object = None
    evidence_json: Optional[dict] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InMemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def create(self, run_id: str, initial_state, *, display_dataset_id: str | None = None) -> None:
        if run_id in self._runs:
            # Batch 5 fix: idempotent. The real API path
            # (app/api/routers/runs.py) calls create() synchronously
            # with display_dataset_id BEFORE backgrounding execution
            # (so a client polling immediately after the 202 response
            # never races a not-yet-started background task) — but
            # stream_with_tracing()/run_with_tracing() ALSO call
            # create() at their own start (load-bearing for every
            # caller that invokes them directly, without a prior
            # create(), e.g. most of this suite). Without this
            # idempotency check, that second call would silently
            # clobber the first call's display_dataset_id override
            # with initial_state.dataset_id (the run-scoped clone).
            return
        self._runs[run_id] = RunRecord(
            run_id=run_id,
            dataset_id=display_dataset_id if display_dataset_id is not None else getattr(initial_state, "dataset_id", ""),
            target_column=getattr(initial_state, "target_column", ""),
            status=getattr(initial_state, "status", "initialized"),
        )

    def get(self, run_id: str) -> RunRecord:
        if run_id not in self._runs:
            raise RunNotFoundError(run_id)
        return self._runs[run_id]

    def update(self, run_id: str, state) -> None:
        """
        Updates status/attempt/plan_history from a live AgentState
        snapshot, and stores the state itself as final_state once the
        run reaches a terminal status.
        """
        if run_id not in self._runs:
            raise RunNotFoundError(run_id)
        record = self._runs[run_id]
        record.status = getattr(state, "status", record.status)
        record.attempt = getattr(state, "retry_count", record.attempt)
        record.plan_history = list(getattr(state, "plan_history", record.plan_history))
        if record.status in ("completed", "failed"):
            record.final_state = state
        record.updated_at = datetime.now(timezone.utc).isoformat()

    def list(self) -> list[RunRecord]:
        """Newest-updated first. Additive — not part of the original ABC."""
        return sorted(self._runs.values(), key=lambda r: r.updated_at, reverse=True)

    def append_trace(self, run_id: str, trace_entry) -> None:
        """Satisfies the locked RunStore interface — delegates to append_event."""
        self.append_event(run_id, trace_entry)

    def append_event(self, run_id: str, event: TraceEvent) -> None:
        if run_id not in self._runs:
            raise RunNotFoundError(run_id)
        self._runs[run_id].events.append(event)
        self._runs[run_id].current_node = event.node
        self._runs[run_id].updated_at = datetime.now(timezone.utc).isoformat()

    def get_events(self, run_id: str) -> list:
        return self.get(run_id).events

    def get_events_by_attempt(self, run_id: str) -> dict:
        """
        Groups this run's events by attempt number — the exact
        structure the master prompt's hierarchy example describes
        (RUN -> ATTEMPT N -> events).
        """
        grouped: dict = {}
        for event in self.get(run_id).events:
            grouped.setdefault(event.attempt, []).append(event)
        return grouped

    def exists(self, run_id: str) -> bool:
        return run_id in self._runs

    def save_evidence(self, run_id: str, evidence: dict) -> None:
        if run_id not in self._runs:
            raise RunNotFoundError(run_id)
        self._runs[run_id].evidence_json = evidence
        self._runs[run_id].updated_at = datetime.now(timezone.utc).isoformat()
