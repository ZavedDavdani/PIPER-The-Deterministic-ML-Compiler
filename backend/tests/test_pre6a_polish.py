"""
Pre-6A Polish: regression tests for three of the four locked, scoped
changes in this batch.

1. Stale failure cleanup: report_node now clears state.failure once
   final validation succeeds, so a completed run's result never
   carries a FailureInfo left over from an earlier, superseded REPLAN
   attempt (the open finding documented in CLAUDE.md under Batch 5).
3. Structured RunSummary: build_run_summary() aggregates already-
   computed AgentState fields (comparison, validation, cleaning_log,
   feature_log, retry_count) into one read-only view — never a new
   source of truth.
4. Structured execution timeline: build_execution_timeline() derives a
   high-level phase timeline purely from an existing TraceEvent stream
   — TraceEvent remains the sole source of execution-history truth.

(Item 2, model-selection transparency, is tested alongside
compare_models()'s existing suite in test_evaluation.py, since it's a
direct extension of that tool's own contract.)
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.graph import report_node
from app.agent.run_summary import build_run_summary
from app.agent.timeline import build_execution_timeline
from app.agent.tracing import stream_with_tracing
from app.llm.provider import LLMProviderResult, ProposedPlan, ProposedPlanStep
from app.schemas.failure import FailureInfo
from app.schemas.guardrails import PipelineValidationResult, ValidationCheck
from app.schemas.trace_event import TraceEvent
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemoryRunStore, InMemorySplitStore
from tests.conftest import heuristic_llm_provider


class _FailsOnceThenValidProvider:
    """
    Same pattern as TestPlanNodeRetryRouting in
    test_batch5_hardening.py: the first proposal is structurally
    invalid (rejected by validate_proposed_plan(), so CLEAN never runs
    and the dataset stays unmutated), every later call delegates to
    the deterministic heuristic provider. Drives a genuine PLAN-
    triggered REPLAN (attempt 0 fails, populating state.failure) that
    then recovers and completes (attempt 1) — exactly the scenario
    that exposed the stale-failure finding.
    """

    def __init__(self):
        self.calls = 0
        self._heuristic = heuristic_llm_provider()

    def generate_plan(self, context):
        self.calls += 1
        if self.calls == 1:
            steps = [ProposedPlanStep(
                action="a", tool_name="impute_missing_values",
                arguments={"column": "TotalCharges", "strategy": "not_a_real_strategy"}, reasoning="r",
            )]
            return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))
        return self._heuristic.generate_plan(context)


class TestStaleFailureCleanup:
    def test_report_node_clears_failure_when_validation_passes(self):
        stale_failure = FailureInfo(
            category="EVALUATION_ERROR", message="stale from an earlier, superseded attempt",
            node="validate", attempt=0, retryable=True,
        )
        validation = PipelineValidationResult(
            dataset_id="d1", target_column="t", valid=True,
            checks=[ValidationCheck(check="leakage", passed=True, severity="info", message="ok")],
        )
        state = AgentState(
            run_id="r1", dataset_id="d1", target_column="t",
            validation=validation, failure=stale_failure, retry_count=1,
        )

        update = report_node(state)

        assert update["status"] == "completed"
        assert update["failure"] is None

    def test_report_node_still_flips_human_intervention_on_genuine_terminal_failure(self):
        """Control: the failed/retries-exhausted branch is untouched by this fix."""
        validation = PipelineValidationResult(
            dataset_id="d1", target_column="t", valid=False,
            checks=[ValidationCheck(check="leakage", passed=False, severity="error", message="bad")],
        )
        failure = FailureInfo(
            category="LEAKAGE_ERROR", message="leakage", node="validate", attempt=2, retryable=True,
        )
        state = AgentState(
            run_id="r1", dataset_id="d1", target_column="t",
            validation=validation, failure=failure, retry_count=2, max_retries=2,
        )

        update = report_node(state)

        assert update["status"] == "failed"
        assert update["failure"].human_intervention_required is True

    def test_report_node_leaves_an_already_failed_state_untouched(self):
        """Control: report_node reached with status already 'failed' and no
        retry budget left must not be perturbed by the success-path change."""
        failure = FailureInfo(
            category="DUPLICATE_PLAN", message="dup", node="plan", attempt=0,
            retryable=False, human_intervention_required=True,
        )
        state = AgentState(
            run_id="r1", dataset_id="d1", target_column="t",
            status="failed", failure=failure,
        )

        update = report_node(state)

        assert update == {}

    def test_real_graph_run_that_replans_then_succeeds_has_no_stale_failure(self, telco_df: pd.DataFrame):
        """
        End-to-end proof of the Batch 5 finding's fix: a genuine
        PLAN-triggered REPLAN (attempt 0 fails, populating
        state.failure) followed by a genuine success (attempt 1) must
        report failure=None, not attempt 0's stale FailureInfo.
        """
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _FailsOnceThenValidProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="pre6a_stale_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "completed"
        assert result["retry_count"] == 1
        assert result["failure"] is None


class TestRunSummary:
    def test_build_run_summary_from_a_real_completed_run(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="pre6a_summary_001", dataset_id="dataset_001", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        record = run_store.get("pre6a_summary_001")
        assert record.status == "completed"

        summary = build_run_summary(record.run_id, record.final_state)

        assert summary.run_id == "pre6a_summary_001"
        assert summary.status == "completed"
        assert summary.retry_count == 0
        assert summary.replanned is False
        assert len(summary.candidate_models) == 2
        assert summary.winning_model_id == record.final_state.comparison.recommended_model_id
        assert summary.winning_algorithm in ("random_forest", "logistic_regression")
        assert summary.selection_justification == record.final_state.comparison.justification
        assert len(summary.operations_executed) >= 1
        assert summary.guardrail_valid is True
        assert len(summary.guardrail_checks) >= 1

    def test_replanned_flag_and_retry_count_reflect_a_genuine_replan(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _FailsOnceThenValidProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="pre6a_summary_002", dataset_id="dataset_001", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        record = run_store.get("pre6a_summary_002")
        summary = build_run_summary(record.run_id, record.final_state)

        assert summary.retry_count == 1
        assert summary.replanned is True
        assert summary.status == "completed"

    def test_build_run_summary_on_a_bare_pre_planning_state_does_not_crash(self):
        """
        A state that never reached COMPARE/VALIDATE (e.g. a hard,
        terminal early failure) must still produce a well-formed
        RunSummary with empty/None fields, not raise.
        """
        state = AgentState(run_id="pre6a_summary_003", dataset_id="d1", target_column="t", status="failed")

        summary = build_run_summary(state.run_id, state)

        assert summary.candidate_models == []
        assert summary.winning_model_id is None
        assert summary.winning_algorithm is None
        assert summary.selection_justification is None
        assert summary.operations_executed == []
        assert summary.guardrail_valid is None
        assert summary.guardrail_checks == []


class TestExecutionTimeline:
    def test_build_execution_timeline_from_a_real_completed_run(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="pre6a_timeline_001", dataset_id="dataset_001", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        events = run_store.get_events("pre6a_timeline_001")
        timeline = build_execution_timeline("pre6a_timeline_001", events)

        assert timeline.run_id == "pre6a_timeline_001"
        assert timeline.final_status == "completed"
        assert timeline.replan_count == 0
        phase_labels = [p.phase for p in timeline.phases]
        assert "Profile" in phase_labels
        assert "Train" in phase_labels
        assert "Validate" in phase_labels
        assert phase_labels[-1] == "Complete"
        assert all(p.event_count >= 1 for p in timeline.phases)
        assert all(p.status == "success" for p in timeline.phases)

    def test_a_genuine_replan_shows_up_as_a_second_attempt_in_the_timeline(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _FailsOnceThenValidProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="pre6a_timeline_002", dataset_id="dataset_001", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        events = run_store.get_events("pre6a_timeline_002")
        timeline = build_execution_timeline("pre6a_timeline_002", events)

        assert timeline.replan_count == 1
        assert timeline.final_status == "completed"
        attempts_seen = {p.attempt for p in timeline.phases}
        assert attempts_seen == {0, 1}

    def test_consecutive_events_for_the_same_phase_and_attempt_collapse(self):
        events = [
            TraceEvent(run_id="r", step_id="s1", attempt=0, node="profile", event_type="node_completed", timestamp="t1", status="success"),
            TraceEvent(run_id="r", step_id="s2", attempt=0, node="profiler", event_type="tool_called", timestamp="t2", status="success"),
            TraceEvent(run_id="r", step_id="s3", attempt=0, node="plan", event_type="node_completed", timestamp="t3", status="success"),
            TraceEvent(run_id="r", step_id="s4", attempt=0, node="report", event_type="run_completed", timestamp="t4", status="success"),
        ]

        timeline = build_execution_timeline("r", events)

        assert [p.phase for p in timeline.phases] == ["Profile", "Plan", "Complete"]
        assert timeline.phases[0].event_count == 2
        assert timeline.phases[0].started_at == "t1"
        assert timeline.phases[0].ended_at == "t2"
        assert timeline.final_status == "completed"
        assert timeline.replan_count == 0

    def test_a_failed_event_marks_its_collapsed_phase_and_the_final_status_as_failure(self):
        events = [
            TraceEvent(run_id="r", step_id="s1", attempt=0, node="train", event_type="node_completed", timestamp="t1", status="success"),
            TraceEvent(run_id="r", step_id="s2", attempt=0, node="trainer", event_type="tool_called", timestamp="t2", status="failure", severity="error"),
            TraceEvent(run_id="r", step_id="s3", attempt=0, node="report", event_type="run_failed", timestamp="t3", status="failure"),
        ]

        timeline = build_execution_timeline("r", events)

        assert timeline.phases[0].status == "failure"
        assert timeline.phases[-1].phase == "Failed"
        assert timeline.final_status == "failed"

    def test_empty_event_list_produces_an_empty_in_progress_timeline(self):
        timeline = build_execution_timeline("r_empty", [])

        assert timeline.phases == []
        assert timeline.replan_count == 0
        assert timeline.final_status is None
