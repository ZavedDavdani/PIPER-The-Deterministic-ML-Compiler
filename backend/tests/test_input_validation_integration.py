"""
Formal tests confirming validate_input_node is genuinely wired into
the real graph as its entry point (section 9 integration), not just a
standalone tool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from tests.conftest import heuristic_llm_provider
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore


class TestInputValidationGateIntegration:
    def test_empty_dataset_fails_before_profiling(self):
        empty_df = pd.DataFrame({"a": pd.array([], dtype="float64"), "target": pd.array([], dtype="object")})
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_empty", empty_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())

        initial = AgentState(run_id="run_001", dataset_id="dataset_empty", target_column="target")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        assert result["retry_count"] == 0
        trace_nodes = [t.node for t in result["tool_trace"]]
        assert "profiler" not in trace_nodes  # never reached profiling

    def test_failure_info_is_structured(self):
        empty_df = pd.DataFrame({"a": pd.array([], dtype="float64"), "target": pd.array([], dtype="object")})
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_empty", empty_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())

        initial = AgentState(run_id="run_002", dataset_id="dataset_empty", target_column="target")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        failure = result["failure"]
        assert failure is not None
        assert failure.category == "DATA_ERROR"
        assert failure.retryable is False
        assert failure.human_intervention_required is True
        assert failure.node == "validate_input"

    def test_insufficient_samples_fails_before_profiling(self):
        small_df = pd.DataFrame({"a": np.random.rand(10), "target": np.random.choice(["Yes", "No"], 10)})
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_small", small_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())

        initial = AgentState(run_id="run_003", dataset_id="dataset_small", target_column="target")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        assert result["retry_count"] == 0

    def test_clean_telco_still_passes_through_to_completion(self, telco_df: pd.DataFrame):
        """
        Confirms adding the input-validation gate did not break the
        real happy path — the same end-to-end pipeline that already
        works must still work with VALIDATE_INPUT as the new entry
        point.
        """
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())

        initial = AgentState(run_id="run_004", dataset_id="dataset_001", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        trace_nodes = [t.node for t in result["tool_trace"]]
        assert "input_validator" in trace_nodes
        assert "profiler" in trace_nodes
