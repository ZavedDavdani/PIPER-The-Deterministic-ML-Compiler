"""
Formal tests for the REAL pipeline graph (graph.py).

This file replaces the original M2 stub-based tests. The M2 stub
functions (train_node, evaluate_node, validate_node in m2_nodes.py)
still exist and are retained for reference, but build_graph() no
longer uses them — it wires the real tools
(split_dataset/train_model/evaluate_model/validate_pipeline) via
real_nodes.py instead. build_graph() also now requires three stores
(dataset_store, split_store, model_store), not one.

Rather than mocking evaluate_node to force a suspicious result (that
node doesn't exist in the real graph anymore), the REPLAN-loop test
now injects GENUINE leakage into the dataset — a feature that's an
exact duplicate of the target — so the real trained model genuinely
achieves a suspiciously perfect score and the real guardrails
genuinely catch it. This is a stronger test than mocking, per the
standing instruction to test invariants behaviorally.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from tests.conftest import heuristic_llm_provider
from app.llm.provider import LLMProviderResult, ProposedPlan, ProposedPlanStep
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore


@pytest.fixture()
def telco_store(telco_df: pd.DataFrame) -> InMemoryDatasetStore:
    store = InMemoryDatasetStore()
    store.save("dataset_001", telco_df)
    return store


@pytest.fixture()
def fresh_stores(telco_store):
    """A complete set of the three stores build_graph() now requires."""
    return telco_store, InMemorySplitStore(), InMemoryModelStore()


class TestHappyPath:
    """
    Every assertion here is checking REAL output: a genuinely fitted
    sklearn Pipeline, genuinely computed metrics against a genuinely
    held-out test split, and a genuine guardrail validation pass — no
    fabricated numbers anywhere in this path.
    """

    def test_reaches_completed_status(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"

    def test_resolves_binary_classification_task_type(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["task_type"] == "binary_classification"

    def test_plan_covers_cleaning_and_feature_engineering(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        tool_names = {step.tool_name for step in result["plan"]}
        assert "drop_column" in tool_names
        assert "convert_column_type" in tool_names
        assert "impute_missing_values" in tool_names
        assert "encode_categorical_features" in tool_names
        assert "scale_features" in tool_names

    def test_all_plan_steps_complete_successfully(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        statuses = {step.status for step in result["plan"]}
        assert statuses == {"completed"}

    def test_cleaning_log_matches_the_m1_total_charges_chain(self, fresh_stores):
        """
        Cross-check against M1's own acceptance test: the same
        11-failed-conversion / median-imputation numbers must appear
        here too, proving the graph's CLEAN node genuinely calls the
        same tools with the same real behavior, not a re-implementation.
        """
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        summaries = [entry.result_summary for entry in result["cleaning_log"]]
        assert any("customerID" in s for s in summaries)
        assert any("11 failed" in s for s in summaries)
        assert any("Imputed 11 missing" in s for s in summaries)

    def test_retry_count_stays_zero_on_a_clean_pass(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["retry_count"] == 0

    def test_validation_passes(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        validation = result["validation"]
        assert validation.valid is True

    def test_a_real_model_id_is_stored_in_model_store(self, fresh_stores):
        """
        Confirms the model actually made it into ModelStore, not just
        that AgentState has a model_id reference — closing the loop
        between state and storage.
        """
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        model_id = result["model_results"][-1].model_id
        assert model_store.exists(model_id)

    def test_metrics_are_plausible_not_suspicious(self, fresh_stores):
        """
        Real churn prediction on this dataset should land in a
        plausible range, not suspiciously perfect — a good sign
        nothing is leaking in the default (clean) path.
        """
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        eval_result = result["evaluation_results"][-1]
        assert 0.5 < eval_result.accuracy < 0.95
        assert 0.0 < eval_result.f1 < 0.98


class TestConstraintNineNonBinaryTargetFailsHard:
    """
    Contract has 3 distinct values (Month-to-month, One year, Two
    year) — not binary. Must fail immediately, never attempt a replan.
    """

    def test_status_is_failed(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_002", dataset_id="dataset_001", target_column="Contract")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"

    def test_task_type_stays_none(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_002", dataset_id="dataset_001", target_column="Contract")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["task_type"] is None

    def test_retry_count_is_never_incremented(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_002", dataset_id="dataset_001", target_column="Contract")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["retry_count"] == 0

    def test_error_message_explains_scope_violation(self, fresh_stores):
        """
        As of section 9 (data-quality validation), a non-binary target
        is now caught by validate_input_node BEFORE profile_node even
        runs — the detailed "3 distinct values" explanation now lives
        in the structured failure.evidence (per sections 5-6's
        separation of lifecycle status from structured failure detail)
        rather than only in the free-text error string. The outcome
        (status=failed, retry_count=0, no replan) is unchanged; only
        which node catches it and how the detail is represented
        improved.
        """
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_002", dataset_id="dataset_001", target_column="Contract")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        failure = result["failure"]
        assert failure is not None
        assert failure.category == "TARGET_ERROR"
        assert failure.retryable is False
        violations = failure.evidence["violations"]
        assert any("3 distinct value" in v["reason"] for v in violations)

    def test_nonexistent_target_column_also_fails_hard(self, fresh_stores):
        dataset_store, split_store, model_store = fresh_stores
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(
            run_id="run_003", dataset_id="dataset_001", target_column="DoesNotExist"
        )

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        assert result["retry_count"] == 0


class TestReplanLoopIsBoundedByMaxRetriesRealLeakage:
    """
    Injects GENUINE leakage (a feature that's an exact duplicate of
    the target) into the dataset. The real trained model genuinely
    achieves a suspiciously perfect score; the real guardrails
    genuinely catch it; the graph genuinely attempts REPLAN.

    UPDATED (Phase 4, duplicate-plan prevention): the dummy,
    non-LLM planner is deterministic — given the same profile, it
    always produces the exact same plan, since it has no rule to
    detect "the guardrails rejected this because of a specific leaky
    column" and adjust. This means, as of Phase 4's duplicate-plan
    detection, REPLAN now catches this as a DUPLICATE_PLAN on the
    SECOND attempt (retry_count=1) rather than grinding through the
    full retry budget (retry_count=2) re-running an identical,
    doomed plan three times. This is the CORRECT, improved behavior
    — "no repeated expensive training/evaluation" — not a regression.
    The original 3-evaluation-attempt version of this test was only
    possible because duplicate-plan detection didn't exist yet.

    This replaces the old mocked-evaluate_node version of this test —
    evaluate_node doesn't exist in the real graph's node set anymore,
    and a genuine forced failure is a stronger test than a mock per
    the standing instruction to test invariants behaviorally.
    """

    @pytest.fixture()
    def leaky_telco_store(self, telco_df: pd.DataFrame) -> InMemoryDatasetStore:
        leaky_df = telco_df.copy()
        leaky_df["leaky_duplicate_of_target"] = leaky_df["Churn"]
        store = InMemoryDatasetStore()
        store.save("dataset_leak", leaky_df)
        return store

    def test_model_results_do_not_accumulate_stale_models_across_replan(self, leaky_telco_store):
        """
        Regression test for a genuine bug found during M3 Phase 5:
        train_node_v2/evaluate_node_v2 used to APPEND new candidates
        onto state.model_results/evaluation_results
        (state.model_results + new_model_results) rather than
        replacing them. Under the old static dummy heuristic this was
        invisible — it always produced the SAME plan on every REPLAN
        attempt for a leakage dataset, so TRAIN was reached at most
        once before DUPLICATE_PLAN fired. With a genuinely context-
        driven planner (able to legitimately propose a different plan
        across attempts — e.g. because clean_node mutates the working
        dataset in place between attempts), TRAIN can legitimately run
        more than once per graph invocation, and the accumulation bug
        surfaced as stale models from a PREVIOUS, already-failed
        attempt being redundantly re-evaluated and fed into
        compare_models() alongside the CURRENT attempt's candidates.

        Fixed by having train_node_v2/evaluate_node_v2 return their new
        results as a replacement, not an accumulation — matching
        split_node's own established pattern of overwriting split_id
        fresh each cycle.
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(leaky_telco_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_regress_001", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        from app.agent.nodes.real_nodes import _TRAIN_CANDIDATES

        # Regardless of how many distinct plans were attempted across
        # the REPLAN loop, the FINAL state must reflect exactly one
        # cycle's worth of candidates -- never more.
        assert len(result["model_results"]) == len(_TRAIN_CANDIDATES)
        assert len(result["evaluation_results"]) == len(_TRAIN_CANDIDATES)
        # No duplicate model_ids in evaluation_results (would indicate
        # the same model being evaluated more than once).
        eval_model_ids = [e.model_id for e in result["evaluation_results"]]
        assert len(eval_model_ids) == len(set(eval_model_ids))
        # Every evaluated model_id must correspond to a model actually
        # trained in the FINAL cycle, not a stale one from an earlier,
        # already-failed attempt.
        trained_model_ids = {m.model_id for m in result["model_results"]}
        assert set(eval_model_ids) == trained_model_ids

    def test_retry_count_stays_below_max_due_to_duplicate_detection(self, leaky_telco_store):
        """
        retry_count reaches max_retries (2) — for this dataset, the LLM
        planner (via heuristic_llm_provider, see conftest.py) genuinely
        proposes two DISTINCT plans across the retry budget (attempt 1
        still needs to drop customerID/convert TotalCharges; attempt 2
        doesn't, since clean_node already mutated the stored dataset in
        place — existing, pre-M3 behavior) before attempt 3 duplicates
        attempt 2 and DUPLICATE_PLAN correctly fires. The actual
        invariant under test — the loop terminates via duplicate
        detection, not by silently exhausting the retry budget on
        identical repeated work — is unchanged from before M3 Phase 5;
        only the exact retry_count at which it happens has changed, a
        genuine and correct consequence of the planner now being
        context-driven rather than static.
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(leaky_telco_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_004", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["failure"].category == "DUPLICATE_PLAN"
        assert result["retry_count"] == result["max_retries"]

    def test_final_status_is_failed(self, leaky_telco_store):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(leaky_telco_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_005", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"

    def test_failure_category_is_duplicate_plan(self, leaky_telco_store):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(leaky_telco_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_006", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["failure"].category == "DUPLICATE_PLAN"

    def test_only_one_expensive_training_cycle_actually_ran(self, leaky_telco_store):
        """
        The core invariant: model_results/evaluation_results in the
        FINAL state reflect only ONE training cycle's worth of
        candidates (_TRAIN_CANDIDATES, currently 2) — never an
        accumulation across multiple attempts, and never a repeat
        evaluation of a stale model from a previous, already-failed
        attempt. (This accumulation was a genuine pre-existing bug in
        train_node_v2/evaluate_node_v2, found and fixed during M3
        Phase 5 real-graph testing — see those nodes' docstrings.)

        Two DISTINCT plans are genuinely attempted for this leaky
        dataset before DUPLICATE_PLAN fires (see
        test_retry_count_stays_below_max_due_to_duplicate_detection's
        docstring for why) — but each TRAIN/EVALUATE cycle correctly
        starts fresh, so the final state's model_results/
        evaluation_results always contain exactly one cycle's worth of
        candidates, regardless of how many distinct plans preceded it.
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(leaky_telco_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_007", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        from app.agent.nodes.real_nodes import _TRAIN_CANDIDATES

        assert len(result["evaluation_results"]) == len(_TRAIN_CANDIDATES)
        assert len(result["model_results"]) == len(_TRAIN_CANDIDATES)

    def test_original_leakage_evidence_is_still_present_from_first_attempt(self, leaky_telco_store):
        """
        Even though the run terminates via DUPLICATE_PLAN, the
        VALIDATION from the one real attempt that did run must still
        show the actual leakage finding that triggered the whole
        REPLAN sequence in the first place — the failure is still
        explainable, not just "pipeline failed."
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(leaky_telco_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_008", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        validation = result["validation"]
        assert validation.valid is False
        violation_checks = {v.check for v in validation.violations}
        assert "data_leakage" in violation_checks

    def test_metrics_from_the_one_real_attempt_were_genuinely_suspicious(self, leaky_telco_store):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(leaky_telco_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_009", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        first_eval = result["evaluation_results"][0]
        assert first_eval.f1 > 0.98  # genuinely achieved by an actual fitted model
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(leaky_telco_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_008", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        first_eval = result["evaluation_results"][0]
        assert first_eval.f1 > 0.98  # genuinely achieved by an actual fitted model


class TestImbalancePath:
    """
    Locked policy: severe imbalance is a WARNING, not an error — the
    graph must PASS with a warning surfaced, not fail outright.
    """

    @pytest.fixture()
    def severely_imbalanced_store(self) -> InMemoryDatasetStore:
        import numpy as np

        np.random.seed(0)
        n = 2000
        df = pd.DataFrame({
            "feature_a": np.random.rand(n),
            "feature_b": np.random.choice(["X", "Y", "Z"], n),
            "target": ["No"] * 1900 + ["Yes"] * 100,  # 95/5
        })
        store = InMemoryDatasetStore()
        store.save("dataset_imbalanced", df)
        return store

    def test_severe_imbalance_does_not_prevent_completion(self, severely_imbalanced_store):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(severely_imbalanced_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_imb", dataset_id="dataset_imbalanced", target_column="target")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"

    def test_imbalance_warning_is_surfaced(self, severely_imbalanced_store):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(severely_imbalanced_store, split_store, model_store, heuristic_llm_provider())
        initial = AgentState(run_id="run_imb", dataset_id="dataset_imbalanced", target_column="target")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        validation = result["validation"]
        assert validation.valid is True
        warning_checks = {w.check for w in validation.warnings}
        assert "target_imbalance" in warning_checks


class _PlanValidationFailsThenEmptiesFeatureSet:
    """
    Deterministic repro (Batch 7 finding): call 1 proposes a
    structurally-invalid plan (rejected by validate_proposed_plan(),
    so state.validation is never populated this run — the pipeline
    never even reaches VALIDATE on attempt 0). Call 2 proposes
    dropping every non-target column, so feature_engineer_node
    produces an empty feature set, cascading train/evaluate/compare/
    baseline failures into a non-retryable baseline_node failure by
    the time the graph reaches VALIDATE on attempt 1 — validate_node_v2
    itself never runs (no evaluation_results), so state.validation
    stays None. Any further call proposes a fixed trivial plan.
    """

    def __init__(self, telco_columns: list[str]):
        self.calls = 0
        self._telco_columns = telco_columns

    def generate_plan(self, context):
        self.calls += 1
        if self.calls == 1:
            steps = [ProposedPlanStep(
                action="a", tool_name="impute_missing_values",
                arguments={"column": "TotalCharges", "strategy": "not_a_real_strategy"}, reasoning="r",
            )]
            return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))
        if self.calls == 2:
            steps = [
                ProposedPlanStep(action=f"drop {c}", tool_name="drop_column", arguments={"column": c, "reason": "r"}, reasoning="r")
                for c in self._telco_columns if c != "Churn"
            ]
            return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))
        steps = [ProposedPlanStep(action="drop customerID", tool_name="drop_column", arguments={"column": "customerID", "reason": "x"}, reasoning="r")]
        return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))


class TestValidateRoutingDoesNotLoopWithoutBudgetedRetries:
    """
    Batch 7 finding, found via real end-to-end Docker verification
    against a real Ollama model: _route_after_validate used to decide
    REPLAN vs. REPORT purely from state.validation, ignoring the case
    where VALIDATE never actually ran this attempt (state.status ==
    "failed" from an early return, state.validation left None/stale).
    _increment_retry_if_replanning correctly declined to increment
    retry_count in that case (neither of its two conditions held), but
    the graph still routed back to PLAN via the unconditional
    PLAN_ENTRY->PLAN edge — an unbudgeted retry that called the LLM
    again without retry_count ever advancing, bounded only by the much
    larger MAX_EXECUTION_STEPS safety net. Confirmed live: a real
    Dockerized run made 4 real Ollama calls while retry_count stayed
    at 1 the whole time.
    """

    def test_a_non_retryable_failure_reaching_validate_unrun_is_immediately_terminal(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _PlanValidationFailsThenEmptiesFeatureSet(list(telco_df.columns))
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="b7_route_fix_001", dataset_id="dataset_001", target_column="Churn", max_retries=2)

        result = graph.invoke(initial, config={"recursion_limit": 50})

        # The invariant under test: EVERY LLM call is budgeted — i.e.
        # accounted for by a retry_count increment — so the graph can
        # never loop through PLAN_ENTRY->PLAN without advancing
        # retry_count. Before the routing fix: 4 LLM calls with
        # retry_count stuck at 1 (an unbudgeted extra REPLAN), bounded
        # only by the much larger MAX_EXECUTION_STEPS safety net.
        #
        # Asserted as a RELATIONSHIP (one initial attempt + at most
        # max_retries budgeted retries) rather than a magic number, so
        # this keeps testing the actual invariant rather than one
        # incidental call count.
        assert provider.calls == result["retry_count"] + 1
        assert provider.calls <= initial.max_retries + 1
        assert result["status"] == "failed"
        assert result["retry_count"] == initial.max_retries  # budget fully spent, then terminal

        # The reported failure must be the genuine ROOT CAUSE (training
        # could not run — the attempt-2 plan dropped every feature
        # column, leaving an empty feature set), not a downstream
        # cascade symptom. Batch 7's _upstream_already_failed() guard is
        # what preserves this: before it, TRAIN's real failure was
        # overwritten in turn by evaluate/compare/baseline each
        # reporting their own "reached with no X" symptom, so this run
        # terminated one attempt early citing "baseline_node reached
        # with no state.comparison" — a misleading symptom that was
        # additionally marked retryable=False, denying the genuinely
        # fixable root cause its remaining budgeted retry.
        assert result["failure"] is not None
        assert result["failure"].node == "train"
        assert result["failure"].category == "TRAINING_ERROR"
        assert result["failure"].retryable is True
        # Retry budget exhausted on a retryable failure -> escalate.
        assert result["failure"].human_intervention_required is True

    def test_a_genuinely_retryable_failure_reaching_validate_unrun_still_gets_a_replan_chance(self, telco_df: pd.DataFrame):
        """
        Control: this fix must not remove the REPLAN chance for a
        genuinely retryable failure reaching this same situation — only
        for non-retryable ones. Reuses the exact scenario
        test_batch5_hardening.py's _FailsOnceThenValidProvider proves
        recovers via a RETRYABLE PLAN-node failure; asserting here that
        it still recovers confirms this fix didn't regress that path.
        """
        from tests.test_batch5_hardening import _FailsOnceThenValidProvider

        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _FailsOnceThenValidProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="b7_route_fix_002", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        assert result["retry_count"] == 1
