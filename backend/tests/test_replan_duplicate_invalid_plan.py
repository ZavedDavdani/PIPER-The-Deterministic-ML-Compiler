"""
Regression tests for a genuine reliability finding observed live
(run_dfcbae97, real qwen3:4b against the Telco acceptance dataset): the
LLM repeatedly proposed `drop_column` with an empty `column` argument.
`validate_proposed_plan()` correctly rejected it every time, but REPLAN
produced the SAME invalid proposal again on attempt 1 and attempt 2,
burning several minutes and the entire retry budget on a proposal
PIPER already had complete evidence was invalid.

Root cause: canonicalize_plan()/plan_hash()/plan_history (the existing
DUPLICATE_PLAN machinery) only ever ran on a plan that had ALREADY
passed validate_proposed_plan() — a REJECTED (invalid) proposal was
never given any executable identity, so nothing deterministic could
ever detect "this is the exact same invalid plan I already rejected."

Fix (plan_node_v2, app/agent/nodes/real_nodes.py): a rejected proposal
is now ALSO canonicalized and checked against state.plan_history. A
first-time-invalid proposal is still reported as a normal, retryable
EVALUATION_ERROR (unchanged routing/retry semantics) — but its hash is
recorded, so an EXACT REPEAT of that same invalid proposal is caught
and reported as a terminal, non-retryable DUPLICATE_PLAN, exactly
mirroring the existing post-validation duplicate-plan mechanism one
stage earlier. validate_proposed_plan() itself is completely
unchanged — this only stops the LLM from being given unlimited free
retries to repeat content already known to be invalid.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.nodes.real_nodes import plan_node_v2
from app.llm.provider import LLMProviderResult, ProposedPlan, ProposedPlanStep
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore
from tests.conftest import heuristic_llm_provider


class _AlwaysEmptyColumnProvider:
    """Deterministic repro of the exact observed bug: every call proposes
    drop_column with an empty (invalid) `column` argument."""

    def __init__(self):
        self.calls = 0

    def generate_plan(self, context):
        self.calls += 1
        steps = [
            ProposedPlanStep(
                action="Drop identifier column", tool_name="drop_column",
                arguments={"column": ""}, reasoning="r",
            )
        ]
        return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))


class _RecoversOnThirdCallProvider:
    """Repeats the same invalid proposal twice, then proposes a genuinely
    different (still invalid, different field) plan on the third call —
    proves the duplicate check is content-specific, not a blanket
    'second failure is always terminal' rule."""

    def __init__(self):
        self.calls = 0

    def generate_plan(self, context):
        self.calls += 1
        if self.calls <= 2:
            steps = [ProposedPlanStep(action="a", tool_name="drop_column", arguments={"column": ""}, reasoning="r")]
        else:
            steps = [ProposedPlanStep(
                action="a", tool_name="convert_column_type",
                arguments={"column": "TotalCharges", "target_type": "not_a_real_type"}, reasoning="r",
            )]
        return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))


class TestDuplicateInvalidPlanDetection:
    def test_repeating_the_same_invalid_proposal_terminates_early_not_after_the_full_retry_budget(
        self, telco_df: pd.DataFrame
    ):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _AlwaysEmptyColumnProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="replan_dup_001", dataset_id="dataset_001", target_column="Churn", max_retries=5)

        result = graph.invoke(initial, config={"recursion_limit": 50})

        # Before the fix: every one of (max_retries + 1) attempts would
        # call the LLM with the identical invalid proposal, burning the
        # entire retry budget. After the fix: the SECOND occurrence of
        # the identical invalid proposal is caught deterministically —
        # exactly 2 LLM calls, not 6.
        assert provider.calls == 2
        assert result["status"] == "failed"
        assert result["failure"].category == "DUPLICATE_PLAN"
        assert result["failure"].retryable is False
        assert result["failure"].human_intervention_required is True
        assert "REJECTED" in result["failure"].message

    def test_first_invalid_attempt_is_still_a_normal_retryable_failure(self, telco_df: pd.DataFrame):
        """Control: the FIRST occurrence of an invalid proposal is
        unaffected by this fix — still retryable, still gets a REPLAN
        chance, exactly as before."""
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _AlwaysEmptyColumnProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="replan_dup_002", dataset_id="dataset_001", target_column="Churn", max_retries=5)

        graph.invoke(initial, config={"recursion_limit": 50})

        # The provider was called a second time at all — proves the
        # FIRST invalid proposal was retryable and genuinely triggered
        # a REPLAN, not an immediate terminal failure.
        assert provider.calls == 2

    def test_a_genuinely_different_invalid_proposal_is_not_treated_as_a_duplicate(self, telco_df: pd.DataFrame):
        """Content-specific: repeating the SAME invalid plan is caught,
        but a DIFFERENT invalid plan (different tool/field) must still
        get its own normal retry — the check is a hash of executable
        content, never a blanket 'second failure is terminal' rule."""
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _RecoversOnThirdCallProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="replan_dup_003", dataset_id="dataset_001", target_column="Churn", max_retries=5)

        result = graph.invoke(initial, config={"recursion_limit": 50})

        # Two identical-invalid calls (1st retryable, 2nd caught as
        # duplicate) — the third, genuinely different invalid proposal
        # never has a chance to run because the 2nd call already
        # terminated the run. This proves duplicate detection did NOT
        # silently skip ahead to call 3; it stopped exactly at the
        # real duplicate.
        assert provider.calls == 2
        assert result["failure"].category == "DUPLICATE_PLAN"

    def test_evidence_includes_the_actual_rejected_arguments_not_just_the_field_name(self, telco_df: pd.DataFrame):
        """The locked requirement 'REPLAN receives clear actionable
        evidence about the invalid argument' — the FailureInfo evidence
        must carry the real submitted (invalid) arguments, not just an
        abstract description of the violated rule."""
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _AlwaysEmptyColumnProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="replan_dup_004", dataset_id="dataset_001", target_column="Churn", max_retries=5)

        result = graph.invoke(initial, config={"recursion_limit": 50})

        evidence = result["failure"].evidence
        assert "rejected_steps" in evidence
        assert evidence["rejected_steps"][0]["tool_name"] == "drop_column"
        assert evidence["rejected_steps"][0]["arguments"] == {"column": ""}
        assert "violations" in evidence  # the original field/reason evidence is preserved, not replaced

    def test_a_valid_recovering_plan_still_succeeds_normally(self, telco_df: pd.DataFrame):
        """Control: a run that fails validation once, then proposes a
        genuinely valid plan, must still complete normally — this fix
        must not interfere with the ordinary recovery path at all."""
        from tests.test_batch5_hardening import _FailsOnceThenValidProvider

        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _FailsOnceThenValidProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="replan_dup_005", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert provider.calls == 2
        assert result["status"] == "completed"

    def test_max_retries_zero_still_reports_a_single_retryable_failure_not_a_false_duplicate(
        self, telco_df: pd.DataFrame
    ):
        """Edge case: with no retry budget at all, the FIRST invalid
        proposal must still be reported as the normal (retryable=True
        at the category level, though no budget remains) EVALUATION_ERROR
        — never misreported as DUPLICATE_PLAN, since nothing has been
        proposed twice yet."""
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _AlwaysEmptyColumnProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="replan_dup_006", dataset_id="dataset_001", target_column="Churn", max_retries=0)

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert provider.calls == 1
        assert result["failure"].category == "EVALUATION_ERROR"
        assert result["failure"].retryable is True


class TestPlanNodeUnitLevel:
    """Direct plan_node_v2() calls — faster, isolates the node's own
    logic from full graph routing (already covered above)."""

    def test_rejected_plan_hash_is_recorded_in_plan_history(self):
        store = InMemoryDatasetStore()
        store.save("d1", pd.DataFrame({"a": [1, 2], "b": [3, 4], "Churn": ["Yes", "No"]}))
        provider = _AlwaysEmptyColumnProvider()
        state = AgentState(
            run_id="r1", dataset_id="d1", target_column="Churn",
            profile={"dummy": "profile-present-is-all-plan_node_v2-checks"},
        )

        update = plan_node_v2(state, store, provider)

        assert update["status"] == "failed"
        assert update["failure"].category == "EVALUATION_ERROR"
        assert len(update["plan_history"]) == 1

    def test_second_call_with_the_recorded_hash_present_is_a_duplicate(self):
        store = InMemoryDatasetStore()
        store.save("d1", pd.DataFrame({"a": [1, 2], "b": [3, 4], "Churn": ["Yes", "No"]}))
        provider = _AlwaysEmptyColumnProvider()
        state = AgentState(
            run_id="r1", dataset_id="d1", target_column="Churn",
            profile={"dummy": "profile-present-is-all-plan_node_v2-checks"},
        )

        first = plan_node_v2(state, store, provider)
        state_after_first = state.model_copy(update={
            "plan_history": first["plan_history"],
            "failure": first["failure"],
            "status": "replanning",
            "retry_count": 1,
        })

        second = plan_node_v2(state_after_first, store, provider)

        assert second["status"] == "failed"
        assert second["failure"].category == "DUPLICATE_PLAN"
        assert second["failure"].retryable is False
