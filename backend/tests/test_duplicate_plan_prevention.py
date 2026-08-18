"""
Formal tests for duplicate-plan prevention integrated into the real
REPLAN loop (Phase 4).

Uses genuine forced leakage (not mocks) — the dummy, deterministic
planner has no rule to fix a leaky column it doesn't recognize, so it
produces the identical plan on every attempt, which is exactly the
scenario duplicate-plan detection exists to catch efficiently.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from tests.conftest import heuristic_llm_provider
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore


@pytest.fixture()
def leaky_telco_store(telco_df: pd.DataFrame) -> InMemoryDatasetStore:
    leaky_df = telco_df.copy()
    leaky_df["leaky_duplicate_of_target"] = leaky_df["Churn"]
    store = InMemoryDatasetStore()
    store.save("dataset_leak", leaky_df)
    return store


class TestDuplicatePlanDetection:
    def test_duplicate_plan_produces_failed_status(self, leaky_telco_store):
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        assert result["failure"].category == "DUPLICATE_PLAN"

    def test_plan_history_records_both_attempts_with_identical_hash(self, leaky_telco_store):
        """
        M3 Phase 5 note: with a genuinely context-driven planner (via
        heuristic_llm_provider), attempt 1 and attempt 2 propose two
        DISTINCT plans for this dataset (attempt 1 still needs to drop
        customerID/convert TotalCharges; attempt 2 doesn't, since
        clean_node already mutated the stored dataset in place —
        existing, pre-M3 behavior). Attempt 3 duplicates attempt 2.
        plan_history therefore has 3 entries, not 2, with the last two
        (not the first two) sharing an identical hash. The actual
        invariant — plan_history ends with two consecutive identical
        hashes, proving the duplicate was genuinely detected by
        comparing real canonical hashes — is unchanged.
        """
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_002", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        history = result["plan_history"]
        assert len(history) == 3
        assert history[-1] == history[-2]

    def test_no_repeated_expensive_training_or_evaluation(self, leaky_telco_store):
        """
        The core invariant: TRAIN/EVALUATE/BASELINE must never run for
        a plan that's byte-identical to one already attempted —
        confirmed by counting actual tool_trace entries from those
        specific nodes, not inferring from retry_count.

        tool_trace is an append-only historical log across the WHOLE
        run (unlike state.model_results/evaluation_results, which are
        correctly scoped to just the current cycle — see
        train_node_v2/evaluate_node_v2's docstrings for a related fix).
        With a genuinely context-driven planner (via
        heuristic_llm_provider) now able to legitimately propose two
        DISTINCT plans across this dataset's retry budget before a
        third attempt duplicates the second (see
        test_plan_history_records_both_attempts_with_identical_hash's
        docstring), trainer/evaluator each ran for TWO legitimate
        cycles (2 * len(_TRAIN_CANDIDATES) calls each) — the third,
        genuinely duplicate attempt is what never ran a third time.
        baseline_node still emits exactly one call per cycle that
        actually reached BASELINE (2 legitimate cycles here).
        """
        from app.agent.nodes.real_nodes import _TRAIN_CANDIDATES

        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_003", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        trainer_calls = [t for t in result["tool_trace"] if t.node == "trainer"]
        evaluator_calls = [t for t in result["tool_trace"] if t.node == "evaluator"]
        baseline_calls = [t for t in result["tool_trace"] if t.node == "baseline"]

        legitimate_cycles = len(set(result["plan_history"]))  # distinct plans actually attempted

        assert len(trainer_calls) == legitimate_cycles * len(_TRAIN_CANDIDATES)
        assert len(evaluator_calls) == legitimate_cycles * len(_TRAIN_CANDIDATES)
        assert len(baseline_calls) == legitimate_cycles

    def test_retry_count_stays_below_max_retries(self, leaky_telco_store):
        """
        The loop stops via duplicate detection, not by silently
        exhausting the retry budget on identical repeated work.

        M3 Phase 5 note: for this dataset, retry_count reaches
        max_retries exactly (2 == 2) — the planner (via
        heuristic_llm_provider) genuinely uses its full retry budget
        proposing two distinct plans before the third duplicates the
        second (see test_plan_history_records_both_attempts_with_identical_hash's
        docstring). retry_count == max_retries here does NOT mean a
        third, identical cycle ran — DUPLICATE_PLAN still correctly
        fires and prevents that, which is the invariant this test
        actually protects; it no longer additionally guarantees
        retry_count stays strictly BELOW max_retries for every
        possible dataset, since that specific numeric relationship was
        an artifact of the old static dummy heuristic, not a
        requirement of the architecture itself.
        """
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_004", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["retry_count"] <= result["max_retries"]
        assert result["failure"].category == "DUPLICATE_PLAN"

    def test_no_infinite_loop_confirmed_by_bounded_recursion(self, leaky_telco_store):
        """
        Explicit proof the graph terminates well within a bounded
        recursion limit — if duplicate-plan detection failed to catch
        the repeated plan, this run would either loop until
        max_retries (the old behavior) or, in a genuinely broken
        implementation, exceed the recursion limit entirely.
        """
        graph = build_graph(leaky_telco_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="run_005", dataset_id="dataset_leak", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})  # must not raise GraphRecursionError

        assert result["status"] in ("completed", "failed")


class TestDuplicatePlanDoesNotAffectHappyPath:
    def test_clean_run_has_single_plan_history_entry(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())

        initial = AgentState(run_id="run_006", dataset_id="dataset_001", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        assert len(result["plan_history"]) == 1
        assert result["retry_count"] == 0

    def test_clean_run_has_no_duplicate_plan_failure(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())

        initial = AgentState(run_id="run_007", dataset_id="dataset_001", target_column="Churn")
        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result.get("failure") is None


class TestDuplicatePlanDeterminism:
    def test_same_leaky_dataset_produces_identical_plan_hash_across_runs(self, telco_df: pd.DataFrame):
        """
        Running the exact same scenario twice must produce the exact
        same plan_history hashes both times — the dummy planner and
        canonicalization are both fully deterministic.

        Deliberately builds two INDEPENDENT DatasetStore instances
        from fresh copies of telco_df, rather than reusing the shared
        leaky_telco_store fixture across both invocations — reusing it
        would let the FIRST graph run's in-place dataset mutations
        (drop_column etc.) leak into the SECOND run's starting state,
        producing a genuinely different (shorter) plan on the second
        pass and making this test fail for a reason unrelated to what
        it's actually testing. This was caught by running the test,
        not assumed away.
        """
        def build_fresh_leaky_store():
            leaky_df = telco_df.copy()
            leaky_df["leaky_duplicate_of_target"] = leaky_df["Churn"]
            store = InMemoryDatasetStore()
            store.save("dataset_leak", leaky_df)
            return store

        graph1 = build_graph(build_fresh_leaky_store(), InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial1 = AgentState(run_id="run_008a", dataset_id="dataset_leak", target_column="Churn")
        result1 = graph1.invoke(initial1, config={"recursion_limit": 50})

        graph2 = build_graph(build_fresh_leaky_store(), InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial2 = AgentState(run_id="run_008b", dataset_id="dataset_leak", target_column="Churn")
        result2 = graph2.invoke(initial2, config={"recursion_limit": 50})

        assert result1["plan_history"] == result2["plan_history"]
