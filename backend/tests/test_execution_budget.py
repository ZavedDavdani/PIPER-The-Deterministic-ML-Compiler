"""
M4: deterministic execution-step budget — behavioral tests.

Proves the invariant that graph.invoke() must ALWAYS return a clean
terminal AgentState, never raise, regardless of how max_retries is
configured — a genuine gap before this change: the graph's only real
termination backstop was an externally supplied LangGraph
`recursion_limit`, which raises an uncaught GraphRecursionError rather
than a structured failure if exceeded, and nothing tied it to
max_retries at all. See app/agent/graph.py's MAX_EXECUTION_STEPS and
_increment_retry_if_replanning for the mechanism under test.

Two angles, both required to trust the guarantee:
  1. Normal runs (default max_retries=2) are completely unaffected —
     steps_executed is tracked but never comes close to tripping the
     budget.
  2. An adversarially-configured max_retries (far larger than could
     ever legitimately complete) is still caught DETERMINISTICALLY by
     PIPER's own budget — not by luck, not by DUPLICATE_PLAN (which
     this test deliberately defeats by having the fake provider
     propose a genuinely different, never-repeating plan every single
     attempt), and not by an uncaught GraphRecursionError.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.graph import MAX_EXECUTION_STEPS
from app.llm.provider import LLMProviderResult, ProposedPlan, ProposedPlanStep
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore
from tests.conftest import heuristic_llm_provider


@pytest.fixture()
def telco_store(telco_df: pd.DataFrame) -> InMemoryDatasetStore:
    store = InMemoryDatasetStore()
    store.save("dataset_001", telco_df)
    return store


@pytest.fixture()
def fresh_stores(telco_store):
    return telco_store, InMemorySplitStore(), InMemoryModelStore()


# Columns from the real Telco CSV never touched by the fixed
# drop_column/encode/scale steps below (customerID, Contract,
# MonthlyCharges, tenure) — used purely so each REPLAN attempt's
# convert_column_type target differs from every previous attempt,
# guaranteeing a genuinely distinct canonical plan hash each time (so
# DUPLICATE_PLAN detection never fires, and the loop is bounded ONLY
# by the new execution-step budget). None of these tools mutate the
# stored dataset in a way that conflicts with a later re-conversion of
# a DIFFERENT column, and each is used at most once across the whole
# test.
_DECOY_COLUMN_POOL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "PaymentMethod",
]


class _NeverConvergingLeakyProvider:
    """
    Deliberately adversarial fake provider for this test only: every
    single call proposes a plan that (a) never drops the injected
    leaky column, so validate_pipeline()'s real leakage guardrail
    genuinely fails EVERY attempt (never converges to a passing
    pipeline), and (b) targets a different, never-before-used decoy
    column with convert_column_type each call, so canonicalize_plan()
    produces a genuinely different hash every attempt (never triggers
    DUPLICATE_PLAN). This isolates the invariant under test: with both
    "give up early" mechanisms (passing validation, duplicate
    detection) deliberately defeated, only the new execution-step
    budget can be what stops the loop.
    """

    def __init__(self) -> None:
        self.calls = 0

    def generate_plan(self, context):
        self.calls += 1
        decoy_column = _DECOY_COLUMN_POOL[(self.calls - 1) % len(_DECOY_COLUMN_POOL)]
        steps = [
            ProposedPlanStep(action="a", tool_name="drop_column", arguments={"column": "customerID", "reason": "id"}, reasoning="r"),
            ProposedPlanStep(action="a", tool_name="convert_column_type", arguments={"column": decoy_column, "target_type": "string"}, reasoning="r"),
            ProposedPlanStep(action="a", tool_name="encode_categorical_features", arguments={"columns": ["Contract"]}, reasoning="r"),
            ProposedPlanStep(action="a", tool_name="scale_features", arguments={"columns": ["MonthlyCharges", "tenure"]}, reasoning="r"),
        ]
        return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))


@pytest.fixture()
def leaky_telco_store(telco_df: pd.DataFrame) -> InMemoryDatasetStore:
    leaky_df = telco_df.copy()
    leaky_df["leaky_duplicate_of_target"] = leaky_df["Churn"]
    store = InMemoryDatasetStore()
    store.save("dataset_leak", leaky_df)
    return store


class TestNormalRunsAreUnaffected:
    def test_happy_path_completes_and_reports_a_bounded_step_count(self, fresh_stores):
        """
        A normal, real, passing run: steps_executed is tracked (proves
        the counter genuinely runs, not dead code) but stays nowhere
        near MAX_EXECUTION_STEPS, and the run completes exactly as it
        did before this change.
        """
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="budget_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        assert result["steps_executed"] > 0
        assert result["steps_executed"] < MAX_EXECUTION_STEPS
        assert result.get("failure") is None

    def test_default_max_retries_replan_loop_stays_far_below_the_budget(self, leaky_telco_store):
        """
        The existing, legitimate REPLAN loop (default max_retries=2,
        genuine leakage, real guardrail failures) — already covered by
        test_graph.py's TestReplanLoopIsBoundedByMaxRetriesRealLeakage
        — must still terminate via its EXISTING mechanism
        (DUPLICATE_PLAN, in this case) and must never come close to
        tripping the new budget. Confirms the new mechanism is a
        backstop, not something that changes ordinary behavior.
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(leaky_telco_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="budget_002", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        assert result["failure"].category == "DUPLICATE_PLAN"  # unchanged pre-existing behavior
        assert result["steps_executed"] < MAX_EXECUTION_STEPS


class TestExcessiveRetryConfigurationTerminatesCleanly:
    def test_adversarial_max_retries_does_not_raise_and_terminates_via_budget(self, leaky_telco_store):
        """
        The core invariant: max_retries configured far beyond anything
        that could legitimately complete (1000, vs. the default of 2)
        must still result in graph.invoke() returning normally with a
        clean terminal AgentState — never an uncaught
        GraphRecursionError, never an actually-unbounded run.

        recursion_limit is set generously high (300) here specifically
        so THIS test can prove PIPER's OWN budget (not LangGraph's) is
        what stops the run — every other call site in this codebase
        keeps its existing recursion_limit=50 convention unchanged,
        since MAX_EXECUTION_STEPS (80) already terminates any of those
        long before 50 could ever be reached under a sane max_retries.
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        provider = _NeverConvergingLeakyProvider()
        graph = build_graph(leaky_telco_store, split_store, model_store, provider)
        initial = AgentState(
            run_id="budget_003", dataset_id="dataset_leak", target_column="Churn", max_retries=1000,
        )

        result = graph.invoke(initial, config={"recursion_limit": 300})  # does not raise

        assert result["status"] == "failed"
        assert result["failure"] is not None
        assert result["failure"].category == "EXECUTION_BUDGET_EXCEEDED"
        assert result["failure"].retryable is False
        assert result["failure"].human_intervention_required is True

    def test_retry_count_never_reaches_the_adversarial_max_retries(self, leaky_telco_store):
        """
        The whole point: retry_count stops climbing nowhere near the
        configured max_retries=1000 — it's bounded by the execution
        budget, not by the (absurd) configured value.
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        provider = _NeverConvergingLeakyProvider()
        graph = build_graph(leaky_telco_store, split_store, model_store, provider)
        initial = AgentState(
            run_id="budget_004", dataset_id="dataset_leak", target_column="Churn", max_retries=1000,
        )

        result = graph.invoke(initial, config={"recursion_limit": 300})

        assert result["retry_count"] < 20  # nowhere near 1000
        assert result["max_retries"] == 1000  # the configured value itself is untouched/honest

    def test_steps_executed_is_bounded_at_termination(self, leaky_telco_store):
        """
        steps_executed itself must never run away — it should land at
        or just above MAX_EXECUTION_STEPS (the budget can only be
        checked once per REPLAN cycle, so a small, bounded overshoot
        within a single in-flight cycle is expected), never grow
        without bound.
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        provider = _NeverConvergingLeakyProvider()
        graph = build_graph(leaky_telco_store, split_store, model_store, provider)
        initial = AgentState(
            run_id="budget_005", dataset_id="dataset_leak", target_column="Churn", max_retries=1000,
        )

        result = graph.invoke(initial, config={"recursion_limit": 300})

        assert MAX_EXECUTION_STEPS <= result["steps_executed"] < MAX_EXECUTION_STEPS + 20

    def test_evidence_reports_the_budget_and_configured_retry_values(self, leaky_telco_store):
        """
        The structured FailureInfo must be genuinely explainable —
        carrying both the budget that tripped and the retry
        configuration that would otherwise have kept going, not just a
        bare "pipeline failed" message.
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        provider = _NeverConvergingLeakyProvider()
        graph = build_graph(leaky_telco_store, split_store, model_store, provider)
        initial = AgentState(
            run_id="budget_006", dataset_id="dataset_leak", target_column="Churn", max_retries=1000,
        )

        result = graph.invoke(initial, config={"recursion_limit": 300})

        evidence = result["failure"].evidence
        assert evidence["max_execution_steps"] == MAX_EXECUTION_STEPS
        assert evidence["max_retries"] == 1000
        assert evidence["steps_executed"] >= MAX_EXECUTION_STEPS


class TestBudgetEnforcementMechanismDirectly:
    """
    Fast, isolated tests of the enforcement point itself
    (_increment_retry_if_replanning's budget check), by pre-seeding
    AgentState.steps_executed near the ceiling — cheaper and more
    precise than only proving it end-to-end through many real
    training cycles (covered above), and pins down the exact boundary
    behavior.

    VALIDATE_INPUT -> PROFILE -> SANITIZE run (and each increment
    steps_executed) BEFORE the very first PLAN_ENTRY call, so a
    pre-seeded initial value is offset by that fixed count (3) to
    land the budget CHECK itself (inside PLAN_ENTRY) at an exact,
    intended value.
    """

    _PRE_PLAN_ENTRY_NODE_COUNT = 3  # VALIDATE_INPUT, PROFILE, SANITIZE

    def test_trips_immediately_when_preseeded_at_the_ceiling(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(
            run_id="budget_007", dataset_id="dataset_001", target_column="Churn",
            steps_executed=MAX_EXECUTION_STEPS - self._PRE_PLAN_ENTRY_NODE_COUNT,
        )

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        assert result["failure"].category == "EXECUTION_BUDGET_EXCEEDED"

    def test_does_not_trip_one_below_the_ceiling_on_a_normal_dataset(self, fresh_stores):
        """
        One step below the ceiling at the moment PLAN_ENTRY's check
        actually runs, a normal (non-adversarial) run must still be
        allowed to proceed and complete normally — the check is a
        strict >=, not an off-by-one early trip.
        """
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(
            run_id="budget_008", dataset_id="dataset_001", target_column="Churn",
            steps_executed=MAX_EXECUTION_STEPS - 1 - self._PRE_PLAN_ENTRY_NODE_COUNT,
        )

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        assert result.get("failure") is None
