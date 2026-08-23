"""SQLite run persistence — Batch 2. Isolated from the in-memory API fixture."""

from __future__ import annotations

from app.schemas.trace_event import TraceEvent
from app.storage.exceptions import RunNotFoundError
from app.storage.sqlite_run_store import SqliteRunStore
from app.storage.store_factory import create_run_store


class _FakeState:
    def __init__(
        self,
        dataset_id="d1",
        target_column="target",
        status="running",
        retry_count=0,
        plan_history=None,
    ):
        self.dataset_id = dataset_id
        self.target_column = target_column
        self.status = status
        self.retry_count = retry_count
        self.plan_history = plan_history or []


def _event(run_id: str, step_id: str = "trace_001") -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        step_id=step_id,
        attempt=0,
        node="profile",
        event_type="node_started",
        timestamp="2026-08-23T00:00:00Z",
        status="success",
    )


class TestSqliteRunStore:
    def test_create_get_update_and_events_round_trip(self, tmp_path):
        store = SqliteRunStore(tmp_path / "runs.sqlite")
        store.create("run_001", _FakeState())
        store.append_event("run_001", _event("run_001"))
        store.update("run_001", _FakeState(status="completed", retry_count=1, plan_history=["h1"]))

        record = store.get("run_001")
        assert record.status == "completed"
        assert record.attempt == 1
        assert record.plan_history == ["h1"]
        assert len(record.events) == 1
        assert record.final_state is not None
        assert record.final_state.status == "completed"
        store.close()

        reopened = SqliteRunStore(tmp_path / "runs.sqlite")
        restored = reopened.get("run_001")
        assert restored.status == "completed"
        assert len(restored.events) == 1
        assert restored.events[0].node == "profile"
        assert restored.final_state.status == "completed"
        listed = reopened.list()
        assert [r.run_id for r in listed] == ["run_001"]
        reopened.close()

    def test_missing_run_raises_canonical_error(self, tmp_path):
        store = SqliteRunStore(tmp_path / "runs.sqlite")
        try:
            store.get("missing")
        except RunNotFoundError:
            pass
        else:
            raise AssertionError("expected RunNotFoundError")
        store.close()

    def test_create_is_idempotent(self, tmp_path):
        store = SqliteRunStore(tmp_path / "runs.sqlite")
        store.create("run_001", _FakeState(), display_dataset_id="user_ds")
        store.create("run_001", _FakeState(dataset_id="cloned"))
        assert store.get("run_001").dataset_id == "user_ds"
        store.close()

    def test_factory_memory_backend(self, monkeypatch):
        monkeypatch.setenv("PIPER_RUN_STORE", "memory")
        store = create_run_store()
        from app.storage.run_store import InMemoryRunStore

        assert isinstance(store, InMemoryRunStore)

    def test_factory_sqlite_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIPER_RUN_STORE", "sqlite")
        monkeypatch.setenv("PIPER_SQLITE_PATH", str(tmp_path / "p.sqlite"))
        store = create_run_store()
        assert isinstance(store, SqliteRunStore)
        store.close()
