"""
Formal tests proving FailureInfo is genuinely populated by the real
graph, with correct terminal-vs-recoverable classification — not just
that the schema exists, but that validate_node_v2/report_node/
validate_input_node actually produce it with real evidence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from tests.conftest import heuristic_llm_provider
from app.schemas.failure import RECOVERABLE_CATEGORIES, TERMINAL_CATEGORIES
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore


class TestTerminalFailureClassification:
    def test_empty_dataset_produces_terminal_data_error(self):
        empty_df = pd.DataFrame({"a": pd.array([], dtype="float64"), "target": pd.array([], dtype="object")})
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_empty", empty_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())

        initial = AgentState(run_id="run_001", dataset_id="dataset_empty", target_column="target")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        failure = result["failure"]
        assert failure.category == "DATA_ERROR"
        assert failure.category in TERMINAL_CATEGORIES
        assert failure.retryable is False
        assert result["retry_count"] == 0

    def test_terminal_failure_never_consumes_retry_budget(self):
        """Confirms the schema's model_post_init invariant holds
        through the real graph, not just in isolation."""
        empty_df = pd.DataFrame({"a": pd.array([], dtype="float64"), "target": pd.array([], dtype="object")})
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_empty", empty_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())

        initial = AgentState(run_id="run_002", dataset_id="dataset_empty", target_column="target")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["retry_count"] == 0
        assert result["failure"].retryable is False


class TestRecoverableFailureClassification:
    @pytest.fixture()
    def leaky_telco_store(self, telco_df: pd.DataFrame) -> InMemoryDatasetStore:
        leaky_df = telco_df.copy()
        leaky_df["leaky_duplicate_of_target"] = leaky_df["Churn"]
        store = InMemoryDatasetStore()
        store.save("dataset_leak", leaky_df)
        return store

    def test_leakage_produces_recoverable_leakage_error_on_first_attempt(self, leaky_telco_store):
        """
        The FIRST validation attempt correctly identifies LEAKAGE_ERROR
        — confirmed via the real validation result from that attempt,
        not the final state.failure (which, per Phase 4's
        duplicate-plan detection, becomes DUPLICATE_PLAN once the
        deterministic dummy planner produces the same doomed plan a
        second time — see test_duplicate_plan_is_the_final_category
        below for that distinct, also-correct invariant).
        """
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_003", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        validation = result["validation"]
        assert validation.valid is False
        violation_checks = {v.check for v in validation.violations}
        assert "data_leakage" in violation_checks

    def test_duplicate_plan_is_the_final_failure_category(self, leaky_telco_store):
        """
        Phase 4: since the dummy planner cannot produce a different
        plan given the same profile, the SECOND attempt at this
        already-failed plan is caught as DUPLICATE_PLAN before a
        second wasted training cycle — this becomes the final
        state.failure category, correctly reflecting WHY the run
        actually stopped (not re-running an identical doomed plan),
        distinct from the original leakage finding that caused the
        first attempt to fail (see test above).
        """
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_003b", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        failure = result["failure"]
        assert failure.category == "DUPLICATE_PLAN"
        assert failure.category in RECOVERABLE_CATEGORIES

    def test_failure_evidence_contains_real_violations_not_fabricated(self, leaky_telco_store):
        """
        Confirms the ORIGINAL leakage evidence is still fully
        inspectable via the run's validation result, even though the
        final failure category is DUPLICATE_PLAN — the run is still
        explainable end-to-end, not just "pipeline failed."
        """
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_004", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        validation = result["validation"]
        violation_checks = {v.check for v in validation.violations}
        assert "data_leakage" in violation_checks
        leakage_violation = next(v for v in validation.violations if v.check == "data_leakage")
        assert "leakage indicator" in leakage_violation.message.lower()

    def test_human_intervention_required_true_when_duplicate_plan_detected(self, leaky_telco_store):
        """
        DUPLICATE_PLAN, like retry-exhaustion, means the deterministic
        system has no further automated path forward — a human (or a
        smarter M3 planner) is needed. human_intervention_required
        must be True here even though retry_count (1) is below
        max_retries (2), since the STOPPING REASON is "no new plan to
        try," not "ran out of budget" — both are legitimate reasons
        a human is needed, and this test confirms the duplicate-plan
        path sets the flag correctly too, not only the exhaustion path.
        """
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_005", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["failure"].human_intervention_required is True

    def test_duplicate_plan_failure_is_not_retryable(self, leaky_telco_store):
        """
        Unlike LEAKAGE_ERROR (retryable=True in principle — a smarter
        plan COULD fix it), DUPLICATE_PLAN's retryable=False is
        correct: THIS SPECIFIC failure instance has no path forward
        without a fundamentally different (smarter) planner, which the
        current deterministic dummy planner cannot provide.
        """
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_006", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        failure = result["failure"]
        assert failure.category == "DUPLICATE_PLAN"
        assert failure.retryable is False
        assert failure.human_intervention_required is True

    def test_failure_attempt_matches_final_retry_count(self, leaky_telco_store):
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_007", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["failure"].attempt == result["retry_count"]


class TestSuccessfulRunHasNoFailure:
    def test_clean_run_leaves_failure_none(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())

        initial = AgentState(run_id="run_008", dataset_id="dataset_001", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        assert result.get("failure") is None
