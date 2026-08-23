"""
SqliteRunStore — local persistence for run metadata, trace events, and
the terminal state snapshot. Same public methods as InMemoryRunStore so
the API/tracing layers do not branch.

This is storage only. It never calls an LLM, never mutates a plan, and
never bypasses validate_proposed_plan(). Replay reads what was stored.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from pydantic import BaseModel

from app.schemas.trace_event import TraceEvent
from app.storage.exceptions import RunNotFoundError
from app.storage.run_and_model_store import RunStore
from app.storage.run_store import RunRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    target_column TEXT NOT NULL,
    status TEXT NOT NULL,
    current_node TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    plan_history_json TEXT NOT NULL DEFAULT '[]',
    final_state_json TEXT,
    evidence_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def serialize_state(state: Any) -> Optional[str]:
    if state is None:
        return None
    if isinstance(state, BaseModel):
        return json.dumps(state.model_dump(mode="json"))
    payload: dict[str, Any] = {}
    for key, value in vars(state).items():
        if key.startswith("_"):
            continue
        if isinstance(value, BaseModel):
            payload[key] = value.model_dump(mode="json")
        else:
            payload[key] = value
    return json.dumps(payload, default=_json_default)


def restore_state(raw: Optional[str]) -> Any:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        return data
    return SimpleNamespace(**data)


class SqliteRunStore(RunStore):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    close = close

    def create(self, run_id: str, initial_state, *, display_dataset_id: str | None = None) -> None:
        with self._lock:
            existing = self._conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is not None:
                return
            now = _now()
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, dataset_id, target_column, status, current_node,
                    attempt, plan_history_json, final_state_json, evidence_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, 0, '[]', NULL, NULL, ?, ?)
                """,
                (
                    run_id,
                    display_dataset_id
                    if display_dataset_id is not None
                    else getattr(initial_state, "dataset_id", ""),
                    getattr(initial_state, "target_column", ""),
                    getattr(initial_state, "status", "initialized"),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            event_rows = self._conn.execute(
                "SELECT event_json FROM events WHERE run_id = ? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()
        events = [TraceEvent.model_validate_json(r["event_json"]) for r in event_rows]
        evidence = json.loads(row["evidence_json"]) if row["evidence_json"] else None
        return RunRecord(
            run_id=row["run_id"],
            dataset_id=row["dataset_id"],
            target_column=row["target_column"],
            status=row["status"],
            current_node=row["current_node"],
            attempt=row["attempt"],
            plan_history=json.loads(row["plan_history_json"] or "[]"),
            events=events,
            final_state=restore_state(row["final_state_json"]),
            evidence_json=evidence,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update(self, run_id: str, state) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status, attempt, plan_history_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            status = getattr(state, "status", row["status"])
            attempt = getattr(state, "retry_count", row["attempt"])
            plan_history = list(
                getattr(state, "plan_history", json.loads(row["plan_history_json"] or "[]"))
            )
            if status in ("completed", "failed"):
                final_json = serialize_state(state)
                self._conn.execute(
                    """
                    UPDATE runs SET status = ?, attempt = ?, plan_history_json = ?,
                        final_state_json = ?, updated_at = ? WHERE run_id = ?
                    """,
                    (status, attempt, json.dumps(plan_history), final_json, _now(), run_id),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE runs SET status = ?, attempt = ?, plan_history_json = ?,
                        updated_at = ? WHERE run_id = ?
                    """,
                    (status, attempt, json.dumps(plan_history), _now(), run_id),
                )
            self._conn.commit()

    def list(self) -> list[RunRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id FROM runs ORDER BY updated_at DESC"
            ).fetchall()
        return [self.get(r["run_id"]) for r in rows]

    def append_trace(self, run_id: str, trace_entry) -> None:
        self.append_event(run_id, trace_entry)

    def append_event(self, run_id: str, event: TraceEvent) -> None:
        payload = event.model_dump_json()
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            seq_row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) AS max_seq FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            seq = int(seq_row["max_seq"]) + 1
            self._conn.execute(
                "INSERT INTO events (run_id, seq, event_json) VALUES (?, ?, ?)",
                (run_id, seq, payload),
            )
            self._conn.execute(
                "UPDATE runs SET current_node = ?, updated_at = ? WHERE run_id = ?",
                (event.node, _now(), run_id),
            )
            self._conn.commit()

    def get_events(self, run_id: str) -> list:
        return self.get(run_id).events

    def get_events_by_attempt(self, run_id: str) -> dict:
        grouped: dict = {}
        for event in self.get_events(run_id):
            grouped.setdefault(event.attempt, []).append(event)
        return grouped

    def exists(self, run_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return row is not None

    def save_evidence(self, run_id: str, evidence: dict) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            self._conn.execute(
                "UPDATE runs SET evidence_json = ?, updated_at = ? WHERE run_id = ?",
                (json.dumps(evidence), _now(), run_id),
            )
            self._conn.commit()
