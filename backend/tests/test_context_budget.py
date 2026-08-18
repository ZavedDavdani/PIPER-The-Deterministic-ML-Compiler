"""
Deterministic LLM context-budgeting tests (Batch 7).

Proves: (1) the real reference Telco dataset's context is completely
unaffected (a genuine no-op, not just "under budget by luck"); (2) a
genuinely large/wide dataset triggers real, staged reduction; (3) the
locked minimums (column names/types, target info, missingness stats,
essential summary stats) are NEVER removed, regardless of how far over
budget; (4) only sample_values ever shrinks, and only until the
context fits (or the deterministic floor of zero samples is reached);
(5) plan_node_v2 actually uses the budgeted context, not the raw one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent.nodes.real_nodes import plan_node_v2
from app.agent.state import AgentState
from app.agent.tools.context_budget import (
    DEFAULT_MAX_CONTEXT_CHARS,
    apply_context_budget,
    estimate_context_size,
)
from app.agent.tools.sanitized_llm_context import (
    MAX_SAMPLE_VALUES_PER_COLUMN,
    SanitizedColumnContext,
    SanitizedLLMContext,
    build_sanitized_llm_context,
)
from app.llm.provider import FakeLLMProvider
from app.storage import InMemoryDatasetStore
from tests.conftest import TELCO_CSV_PATH


def _synthetic_context(num_columns: int, samples_per_column: int = MAX_SAMPLE_VALUES_PER_COLUMN) -> SanitizedLLMContext:
    columns = [
        SanitizedColumnContext(
            name=f"col_{i}",
            dtype="object",
            missing_percentage=1.5,
            unique_percentage=42.0,
            sample_values=[f"value_{i}_{j}_padding_to_make_this_realistically_long" for j in range(samples_per_column)],
        )
        for i in range(num_columns)
    ]
    return SanitizedLLMContext(
        dataset_id="synthetic", rows=1000, columns=num_columns, target_column="target",
        column_contexts=columns, sanitization_findings_count=0, high_risk_columns=[],
    )


class TestRealTelcoDatasetIsUnaffected:
    def test_the_real_reference_dataset_stays_completely_unbudgeted(self):
        if not TELCO_CSV_PATH.exists():
            pytest.skip("Telco CSV not found")
        store = InMemoryDatasetStore()
        store.save("d1", pd.read_csv(TELCO_CSV_PATH))
        raw = build_sanitized_llm_context("d1", "Churn", store).data

        budgeted, report = apply_context_budget(raw)

        assert report.reduction_applied is False
        assert report.original_size_chars == report.final_size_chars
        assert report.original_size_chars < DEFAULT_MAX_CONTEXT_CHARS
        assert budgeted == raw  # genuinely unchanged, not just "close enough"


class TestStagedReduction:
    def test_a_context_under_budget_is_returned_unchanged(self):
        context = _synthetic_context(num_columns=3)
        budgeted, report = apply_context_budget(context, max_chars=100_000)

        assert report.reduction_applied is False
        assert budgeted == context

    def test_a_wide_context_over_budget_triggers_reduction(self):
        context = _synthetic_context(num_columns=100)
        original_size = estimate_context_size(context)
        assert original_size > DEFAULT_MAX_CONTEXT_CHARS  # sanity: this fixture genuinely needs budgeting

        budgeted, report = apply_context_budget(context, max_chars=DEFAULT_MAX_CONTEXT_CHARS)

        assert report.reduction_applied is True
        assert report.original_size_chars == original_size
        assert report.final_size_chars <= report.original_size_chars
        assert report.sample_values_cap_used < MAX_SAMPLE_VALUES_PER_COLUMN

    @pytest.mark.parametrize("max_chars,expected_cap", [(100_000, 5), (9000, 2), (7000, 1), (3000, 0)])
    def test_reduction_stage_matches_the_tightest_budget_that_still_fits(self, max_chars, expected_cap):
        """
        Deterministic staged behavior: a tighter budget forces a
        smaller sample_values cap. Thresholds are the REAL measured
        sizes of this exact fixture (30 columns) at each cap — 13033
        chars at cap=5, 8203 at cap=2, 6593 at cap=1, 5043 at cap=0 —
        not guessed.
        """
        context = _synthetic_context(num_columns=30)
        budgeted, report = apply_context_budget(context, max_chars=max_chars)

        assert report.sample_values_cap_used == expected_cap
        for col in budgeted.column_contexts:
            assert len(col.sample_values) <= expected_cap

    def test_locked_minimum_fields_are_never_touched_even_at_maximum_reduction(self):
        context = _synthetic_context(num_columns=200)
        budgeted, report = apply_context_budget(context, max_chars=1)  # impossible budget — forces the floor

        assert report.sample_values_cap_used == 0
        assert len(budgeted.column_contexts) == len(context.column_contexts)
        for original_col, reduced_col in zip(context.column_contexts, budgeted.column_contexts):
            assert reduced_col.name == original_col.name
            assert reduced_col.dtype == original_col.dtype
            assert reduced_col.missing_percentage == original_col.missing_percentage
            assert reduced_col.unique_percentage == original_col.unique_percentage
            assert reduced_col.min == original_col.min
            assert reduced_col.max == original_col.max
            assert reduced_col.mean == original_col.mean
            assert reduced_col.sample_values == []
        assert budgeted.target_column == context.target_column
        assert budgeted.dataset_id == context.dataset_id
        assert budgeted.rows == context.rows
        assert budgeted.columns == context.columns

    def test_never_raises_even_when_the_floor_still_exceeds_budget(self):
        context = _synthetic_context(num_columns=500)
        budgeted, report = apply_context_budget(context, max_chars=1)
        assert report.sample_values_cap_used == 0  # floor reached, no exception

    def test_estimate_is_deterministic(self):
        context = _synthetic_context(num_columns=10)
        assert estimate_context_size(context) == estimate_context_size(context)


class TestPlanNodeUsesTheBudgetedContext:
    def test_plan_node_v2_passes_a_reduced_dataset_context_for_a_wide_dataset(self):
        """
        Builds a genuinely wide real dataset (many real columns, not a
        synthetic SanitizedLLMContext) and drives it through the real
        plan_node_v2 — the FakeLLMProvider records exactly what
        dataset_context it received, proving budgeting is actually
        wired into the real planning path, not just unit-tested in
        isolation.
        """
        if not TELCO_CSV_PATH.exists():
            pytest.skip("Telco CSV not found")
        telco = pd.read_csv(TELCO_CSV_PATH)
        wide = telco.copy()
        for i in range(60):
            wide[f"extra_col_{i}"] = telco["Contract"]

        store = InMemoryDatasetStore()
        store.save("wide_dataset", wide)
        raw_context = build_sanitized_llm_context("wide_dataset", "Churn", store).data
        assert estimate_context_size(raw_context) > DEFAULT_MAX_CONTEXT_CHARS  # sanity: genuinely needs budgeting

        provider = FakeLLMProvider(scenario="valid_plan")
        state = AgentState(
            run_id="ctx_budget_001", dataset_id="wide_dataset", target_column="Churn",
            profile={"dummy": "profile-present-is-all-plan_node_v2-checks"},
        )

        plan_node_v2(state, store, provider)

        assert len(provider.received_contexts) == 1
        received_size = len(str(provider.received_contexts[0].dataset_context))
        raw_size_as_dict_str = len(str(raw_context.model_dump(mode="json")))
        assert received_size < raw_size_as_dict_str

    def test_plan_node_v2_context_is_byte_identical_to_unbudgeted_for_the_real_telco_dataset(self):
        """The no-op case: for the real reference dataset, budgeting must
        change nothing about what the LLM provider actually receives."""
        if not TELCO_CSV_PATH.exists():
            pytest.skip("Telco CSV not found")
        store = InMemoryDatasetStore()
        store.save("dataset_001", pd.read_csv(TELCO_CSV_PATH))
        raw_context = build_sanitized_llm_context("dataset_001", "Churn", store).data

        provider = FakeLLMProvider(scenario="valid_plan")
        state = AgentState(
            run_id="ctx_budget_002", dataset_id="dataset_001", target_column="Churn",
            profile={"dummy": "profile-present-is-all-plan_node_v2-checks"},
        )

        plan_node_v2(state, store, provider)

        assert provider.received_contexts[0].dataset_context == raw_context.model_dump(mode="json")


class TestUpstreamFailureRootCauseIsPreserved:
    """
    Batch 7 regression test for the cascade finding observed live during
    Dockerized end-to-end verification: a TRAIN failure used to be
    overwritten by EVALUATE/COMPARE/BASELINE each reporting their own
    downstream "reached with no X" symptom, so the terminal result named
    the last symptom in the chain instead of the real root cause.
    """

    def _leaky_store(self, telco_df):
        leaky = telco_df.copy()
        leaky["leaky_dup"] = leaky["Churn"]
        store = InMemoryDatasetStore()
        store.save("dataset_leak", leaky)
        return store

    def test_empty_feature_set_reports_training_error_not_a_baseline_symptom(self, telco_df):
        """
        A plan with no encode/scale steps produces an empty feature set,
        so train_model() genuinely fails. The terminal FailureInfo must
        name TRAIN as the root cause, not baseline_node's downstream
        symptom.
        """
        from app.agent import build_graph
        from app.llm.provider import LLMProviderResult, ProposedPlan, ProposedPlanStep
        from app.storage import InMemoryModelStore, InMemorySplitStore

        class NoFeatureStepsProvider:
            """Only ever proposes a drop_column — never encode/scale, so
            _feature_intent_from_plan() yields an empty feature set."""

            def generate_plan(self, context):
                return LLMProviderResult(success=True, plan=ProposedPlan(steps=[
                    ProposedPlanStep(action="a", tool_name="drop_column",
                                     arguments={"column": "customerID", "reason": "id"}, reasoning="r"),
                ]))

        store = self._leaky_store(telco_df)
        graph = build_graph(store, InMemorySplitStore(), InMemoryModelStore(), NoFeatureStepsProvider())
        # max_retries=0 isolates THIS finding (the TRAIN -> EVALUATE ->
        # COMPARE -> BASELINE cascade overwriting the root cause) from
        # the separate, already-correct DUPLICATE_PLAN path: with a
        # retry budget, this deliberately-static provider re-proposes an
        # identical plan on REPLAN, so the terminal failure legitimately
        # becomes DUPLICATE_PLAN instead — correct behavior, but a
        # different code path than the one under test here.
        initial = AgentState(
            run_id="b7_cascade_001", dataset_id="dataset_leak",
            target_column="Churn", max_retries=0,
        )

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        failure = result["failure"]
        assert failure is not None
        assert failure.category == "TRAINING_ERROR", f"root cause lost, got: {failure.message}"
        assert failure.node == "train"
        assert "empty" in failure.message.lower() or "feature" in failure.message.lower()
        # The downstream cascade symptoms must NOT have overwritten it.
        assert "baseline_node reached" not in failure.message

    def test_downstream_nodes_pass_an_upstream_failure_through_unchanged(self):
        """Unit-level proof of the pass-through guard itself."""
        from app.agent.nodes.real_nodes import (
            baseline_node, compare_node, evaluate_node_v2, validate_node_v2,
        )
        from app.schemas.failure import FailureInfo
        from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore

        root_cause = FailureInfo(
            category="TRAINING_ERROR", message="the real root cause", node="train",
            attempt=0, retryable=True,
        )
        state = AgentState(
            run_id="b7_cascade_002", dataset_id="d1", target_column="t",
            status="failed", failure=root_cause,
        )
        split_store, model_store, dataset_store = (
            InMemorySplitStore(), InMemoryModelStore(), InMemoryDatasetStore()
        )

        assert evaluate_node_v2(state, split_store, model_store) == {}
        assert compare_node(state, split_store, model_store) == {}
        assert baseline_node(state, split_store, model_store) == {}
        assert validate_node_v2(state, dataset_store) == {}

    def test_guard_does_not_trigger_on_a_healthy_running_state(self):
        """Control: a normal running state must NOT be short-circuited,
        even if a stale failure from a superseded attempt is present."""
        from app.agent.nodes.real_nodes import _upstream_already_failed
        from app.schemas.failure import FailureInfo

        stale = FailureInfo(
            category="LEAKAGE_ERROR", message="stale from attempt 0", node="validate",
            attempt=0, retryable=True,
        )
        running = AgentState(
            run_id="b7_cascade_003", dataset_id="d1", target_column="t",
            status="running", failure=stale,
        )
        assert _upstream_already_failed(running) is False
