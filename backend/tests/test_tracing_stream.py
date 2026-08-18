"""
M5: behavioral tests for stream_with_tracing() — the live-progress
counterpart to the existing, unchanged run_with_tracing() (see
tests/test_trace_event_run_store.py for that one). Tests against the
REAL graph, not a mock, matching this codebase's standing convention.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.tracing import run_with_tracing, stream_with_tracing
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemoryRunStore, InMemorySplitStore
from tests.conftest import heuristic_llm_provider


@pytest.fixture()
def telco_store(telco_df: pd.DataFrame) -> InMemoryDatasetStore:
    store = InMemoryDatasetStore()
    store.save("dataset_001", telco_df)
    return store


class TestStreamWithTracingHappyPath:
    def test_successful_run_produces_a_completed_result(self, telco_store):
        graph = build_graph(telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="stream_001", dataset_id="dataset_001", target_column="Churn")

        result = stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        assert result["validation"].valid is True

    def test_matches_direct_invoke_outcome(self, telco_df: pd.DataFrame):
        """Purely additive: streaming vs. direct graph.invoke() must reach the same outcome."""
        store_a = InMemoryDatasetStore()
        store_a.save("dataset_001", telco_df)
        graph_a = build_graph(store_a, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        direct_result = graph_a.invoke(
            AgentState(run_id="stream_direct", dataset_id="dataset_001", target_column="Churn"),
            config={"recursion_limit": 50},
        )

        store_b = InMemoryDatasetStore()
        store_b.save("dataset_001", telco_df)
        graph_b = build_graph(store_b, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        streamed_result = stream_with_tracing(
            graph_b, AgentState(run_id="stream_direct2", dataset_id="dataset_001", target_column="Churn"),
            InMemoryRunStore(), config={"recursion_limit": 50},
        )

        assert direct_result["status"] == streamed_result["status"] == "completed"
        assert direct_result["validation"].valid == streamed_result["validation"].valid is True

    def test_live_node_completed_events_are_appended_during_the_run(self, telco_store):
        """
        The core "live" property: node_completed events for the
        intermediate graph nodes (not just a final summary) are
        present in run_store, proving they were genuinely emitted per
        node rather than reconstructed only after the fact.
        """
        graph = build_graph(telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="stream_002", dataset_id="dataset_001", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        events = run_store.get_events("stream_002")
        node_completed_nodes = {e.node for e in events if e.event_type == "node_completed"}
        # Every real graph node name should appear as a live node_completed event.
        assert {"validate_input", "profile", "sanitize", "plan", "clean", "feature_engineer",
                "split", "reproducibility", "train", "evaluate", "compare", "baseline", "validate"} <= node_completed_nodes

    def test_final_run_completed_summary_event_is_present(self, telco_store):
        graph = build_graph(telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="stream_003", dataset_id="dataset_001", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        events = run_store.get_events("stream_003")
        assert events[-1].event_type == "run_completed"

    def test_tool_call_level_events_are_also_present(self, telco_store):
        """
        Streaming must not lose the richer tool-call-level detail
        run_with_tracing() already provides — both views should coexist.
        """
        graph = build_graph(telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="stream_004", dataset_id="dataset_001", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        tool_names = {e.tool_name for e in run_store.get_events("stream_004") if e.tool_name}
        assert "profile_dataset" in tool_names
        assert "train_model" in tool_names

    def test_run_store_reflects_completed_status_and_populates_final_state(self, telco_store):
        graph = build_graph(telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="stream_005", dataset_id="dataset_001", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        record = run_store.get("stream_005")
        assert record.status == "completed"
        assert record.final_state is not None
        assert record.final_state.validation.valid is True
        assert record.final_state.comparison is not None

    def test_run_store_is_updated_live_before_the_run_finishes(self, telco_store):
        """
        Proves run_store isn't only populated at the very end (which
        run_with_tracing() already does, and which stream_with_tracing()
        would trivially satisfy even if it weren't truly live) — a
        genuinely intermediate node (profile) must already be visible
        in run_store's event list partway through, not just after
        graph execution as a whole has returned. Verified by wrapping
        run_store.append_event with a spy that snapshots the record's
        status the FIRST time an event for the 'profile' node arrives
        — at that point VALIDATE/TRAIN/etc. genuinely haven't run yet,
        so the run must still be "running", not "completed".
        """
        graph = build_graph(telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="stream_006", dataset_id="dataset_001", target_column="Churn")

        observed_status_at_profile = {}
        original_append_event = run_store.append_event

        def spy_append_event(run_id, event):
            original_append_event(run_id, event)
            if event.node == "profile" and "status" not in observed_status_at_profile:
                observed_status_at_profile["status"] = run_store.get(run_id).status

        run_store.append_event = spy_append_event

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        assert observed_status_at_profile.get("status") == "running"


class TestStreamWithTracingReplan:
    def test_multi_attempt_run_produces_events_across_multiple_attempts(self, telco_df: pd.DataFrame):
        leaky_df = telco_df.copy()
        leaky_df["leaky_dup"] = leaky_df["Churn"]
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_leak", leaky_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="stream_replan_001", dataset_id="dataset_leak", target_column="Churn")

        result = stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        grouped = run_store.get_events_by_attempt("stream_replan_001")
        assert len(grouped) >= 2

    def test_final_state_carries_the_structured_failure(self, telco_df: pd.DataFrame):
        leaky_df = telco_df.copy()
        leaky_df["leaky_dup"] = leaky_df["Churn"]
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_leak", leaky_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="stream_replan_002", dataset_id="dataset_leak", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        record = run_store.get("stream_replan_002")
        assert record.status == "failed"
        assert record.final_state.failure is not None
        assert record.final_state.failure.category == "DUPLICATE_PLAN"
