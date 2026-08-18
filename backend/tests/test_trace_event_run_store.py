"""
Formal tests for Phase 5: TraceEvent schema, InMemoryRunStore, and the
run_with_tracing() wrapper that connects them to real graph execution.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from tests.conftest import heuristic_llm_provider
from app.agent.tracing import run_with_tracing
from app.schemas.trace_event import TraceEvent
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore
from app.storage.run_store import InMemoryRunStore, RunNotFoundError


class TestTraceEventSchemaValidation:
    def test_minimal_valid_event(self):
        event = TraceEvent(
            run_id="run_001", step_id="trace_001", attempt=0, node="profile",
            event_type="node_started", timestamp="2026-01-01T00:00:00Z", status="success",
        )
        assert event.run_id == "run_001"
        assert event.severity == "info"  # default

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            TraceEvent(
                run_id="run_001", step_id="trace_001", attempt=0, node="profile",
                event_type="node_started", timestamp="2026-01-01T00:00:00Z", status="success",
                made_up_field="not allowed",
            )

    def test_invalid_event_type_rejected(self):
        with pytest.raises(Exception):
            TraceEvent(
                run_id="run_001", step_id="trace_001", attempt=0, node="profile",
                event_type="not_a_real_event_type", timestamp="2026-01-01T00:00:00Z", status="success",
            )

    def test_negative_attempt_rejected(self):
        with pytest.raises(Exception):
            TraceEvent(
                run_id="run_001", step_id="trace_001", attempt=-1, node="profile",
                event_type="node_started", timestamp="2026-01-01T00:00:00Z", status="success",
            )

    def test_serialization_round_trip(self):
        event = TraceEvent(
            run_id="run_001", step_id="trace_001", parent_span_id="trace_000", attempt=1,
            node="validate", event_type="guardrail_checked", timestamp="2026-01-01T00:00:00Z",
            status="failure", severity="error", guardrail_name="data_leakage",
            evidence={"violations": []}, duration_ms=12.5,
        )
        dumped = event.model_dump_json()
        restored = TraceEvent.model_validate_json(dumped)
        assert restored == event


class TestParentChildRelationships:
    def test_parent_span_id_optional(self):
        event = TraceEvent(
            run_id="run_001", step_id="trace_001", attempt=0, node="profile",
            event_type="node_started", timestamp="t", status="success",
        )
        assert event.parent_span_id is None

    def test_parent_span_id_references_another_event(self):
        parent = TraceEvent(
            run_id="run_001", step_id="trace_parent", attempt=0, node="validate",
            event_type="node_started", timestamp="t", status="success",
        )
        child = TraceEvent(
            run_id="run_001", step_id="trace_child", parent_span_id=parent.step_id, attempt=0,
            node="validate", event_type="guardrail_checked", timestamp="t", status="success",
        )
        assert child.parent_span_id == parent.step_id


class TestInMemoryRunStore:
    class _FakeState:
        def __init__(self, dataset_id="d1", target_column="target", status="running", retry_count=0, plan_history=None):
            self.dataset_id = dataset_id
            self.target_column = target_column
            self.status = status
            self.retry_count = retry_count
            self.plan_history = plan_history or []

    def test_create_and_get(self):
        store = InMemoryRunStore()
        store.create("run_001", self._FakeState())
        record = store.get("run_001")
        assert record.run_id == "run_001"
        assert record.dataset_id == "d1"

    def test_get_missing_run_raises(self):
        store = InMemoryRunStore()
        with pytest.raises(RunNotFoundError):
            store.get("nonexistent")

    def test_run_not_found_error_is_the_canonical_storage_exception(self):
        """
        Regression test: run_store.py used to define its OWN separate
        RunNotFoundError class, distinct from
        app.storage.exceptions.RunNotFoundError (the one every other
        store raises and that app.storage exports at the package
        level) despite the identical name/message. Code catching the
        canonical `app.storage.RunNotFoundError` around an
        InMemoryRunStore call would NOT have caught what it actually
        raised. Fixed by re-exporting the canonical class instead of
        redefining it — this proves the two names now resolve to the
        exact same class.
        """
        from app.storage import RunNotFoundError as CanonicalRunNotFoundError

        assert RunNotFoundError is CanonicalRunNotFoundError

        store = InMemoryRunStore()
        with pytest.raises(CanonicalRunNotFoundError):
            store.get("nonexistent")

    def test_update_changes_status(self):
        store = InMemoryRunStore()
        store.create("run_001", self._FakeState())
        store.update("run_001", self._FakeState(status="completed"))
        assert store.get("run_001").status == "completed"

    def test_update_sets_final_state_only_on_terminal_status(self):
        store = InMemoryRunStore()
        store.create("run_001", self._FakeState())
        store.update("run_001", self._FakeState(status="running"))
        assert store.get("run_001").final_state is None
        store.update("run_001", self._FakeState(status="completed"))
        assert store.get("run_001").final_state is not None

    def test_append_event_and_retrieve(self):
        store = InMemoryRunStore()
        store.create("run_001", self._FakeState())
        event = TraceEvent(
            run_id="run_001", step_id="trace_001", attempt=0, node="profile",
            event_type="node_started", timestamp="t", status="success",
        )
        store.append_event("run_001", event)
        assert len(store.get_events("run_001")) == 1

    def test_append_trace_is_an_alias_for_append_event(self):
        """Satisfies the locked RunStore interface's append_trace() method name."""
        store = InMemoryRunStore()
        store.create("run_001", self._FakeState())
        event = TraceEvent(
            run_id="run_001", step_id="trace_001", attempt=0, node="profile",
            event_type="node_started", timestamp="t", status="success",
        )
        store.append_trace("run_001", event)
        assert len(store.get_events("run_001")) == 1

    def test_events_grouped_by_attempt(self):
        store = InMemoryRunStore()
        store.create("run_001", self._FakeState())
        store.append_event("run_001", TraceEvent(
            run_id="run_001", step_id="t1", attempt=0, node="profile",
            event_type="node_started", timestamp="t", status="success",
        ))
        store.append_event("run_001", TraceEvent(
            run_id="run_001", step_id="t2", attempt=1, node="plan",
            event_type="node_started", timestamp="t", status="success",
        ))
        grouped = store.get_events_by_attempt("run_001")
        assert set(grouped.keys()) == {0, 1}
        assert len(grouped[0]) == 1
        assert len(grouped[1]) == 1

    def test_two_runs_are_isolated(self):
        store = InMemoryRunStore()
        store.create("run_a", self._FakeState(dataset_id="dataset_a"))
        store.create("run_b", self._FakeState(dataset_id="dataset_b"))
        store.append_event("run_a", TraceEvent(
            run_id="run_a", step_id="t1", attempt=0, node="profile",
            event_type="node_started", timestamp="t", status="success",
        ))
        assert len(store.get_events("run_a")) == 1
        assert len(store.get_events("run_b")) == 0

    def test_exists(self):
        store = InMemoryRunStore()
        assert store.exists("run_001") is False
        store.create("run_001", self._FakeState())
        assert store.exists("run_001") is True


class TestRunWithTracingIntegration:
    """Tests the wrapper against the REAL graph, not a mock."""

    def test_successful_run_produces_real_events(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="run_traced", dataset_id="dataset_001", target_column="Churn")

        result = run_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        record = run_store.get("run_traced")
        assert record.status == "completed"
        assert len(record.events) > 0
        assert record.final_state is not None

    def test_wrapper_does_not_change_graph_behavior(self, telco_df: pd.DataFrame):
        """
        The wrapper must be purely additive — invoking the graph
        directly vs. through run_with_tracing must produce identical
        pipeline outcomes.
        """
        dataset_store_a = InMemoryDatasetStore()
        dataset_store_a.save("dataset_001", telco_df)
        graph_a = build_graph(dataset_store_a, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        direct_result = graph_a.invoke(
            AgentState(run_id="run_direct", dataset_id="dataset_001", target_column="Churn"),
            config={"recursion_limit": 50},
        )

        dataset_store_b = InMemoryDatasetStore()
        dataset_store_b.save("dataset_001", telco_df)
        graph_b = build_graph(dataset_store_b, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        traced_result = run_with_tracing(
            graph_b,
            AgentState(run_id="run_traced2", dataset_id="dataset_001", target_column="Churn"),
            run_store,
            config={"recursion_limit": 50},
        )

        assert direct_result["status"] == traced_result["status"] == "completed"

    def test_events_include_real_tool_names_from_actual_execution(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="run_traced3", dataset_id="dataset_001", target_column="Churn")

        run_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        tool_names = {e.tool_name for e in run_store.get_events("run_traced3") if e.tool_name}
        assert "profile_dataset" in tool_names
        assert "train_model" in tool_names
        assert "evaluate_model" in tool_names

    def test_multi_attempt_run_produces_events_across_multiple_attempts(self, telco_df: pd.DataFrame):
        leaky_df = telco_df.copy()
        leaky_df["leaky_dup"] = leaky_df["Churn"]
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_leak", leaky_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="run_traced_fail", dataset_id="dataset_leak", target_column="Churn")

        result = run_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        grouped = run_store.get_events_by_attempt("run_traced_fail")
        assert len(grouped) >= 2  # at least attempt 0 and attempt 1 present
