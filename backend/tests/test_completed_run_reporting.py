"""
A run that REPLANned and then SUCCEEDED must not report the superseded
attempt's failure text anywhere.

Observed live in the qwen3:8b demonstration run (run_e13cf35f): the run
reached status "completed" with failure=None (correct), but its final
REPORT TraceEvent message read "Plan is structurally valid but inadequate:
1 material finding(s) ... affecting ['Age']" — the adequacy failure from a
superseded attempt.

Cause: report_node's success branch cleared `failure` (Pre-6A Polish item
1) but not `error`, and tracing.py builds both the node event message and
the terminal run message from `state.error`.

Cosmetic at the API level, but actively misleading in the live SSE feed
and the frontend timeline — which is exactly where a demo is watched.
"""

from __future__ import annotations

import pandas as pd

from app.agent import AgentState, build_graph
from app.agent.tracing import stream_with_tracing
from app.llm.provider import LLMProviderResult, ProposedPlan, ProposedPlanStep
from app.storage import (
    InMemoryDatasetStore,
    InMemoryModelStore,
    InMemoryRunStore,
    InMemorySplitStore,
)

TARGET = "target"


def _step(tool: str, args: dict) -> ProposedPlanStep:
    return ProposedPlanStep(action="a", tool_name=tool, arguments=args, reasoning="r")


def _df(n: int = 120) -> pd.DataFrame:
    return pd.DataFrame({
        "num": [None if i % 6 == 0 else float(i % 50) for i in range(n)],
        "cat": [None if i % 12 == 0 else ("NY" if i % 2 else "LA") for i in range(n)],
        "plain": [float(i % 30) for i in range(n)],
        TARGET: [i % 2 for i in range(n)],
    })


_INADEQUATE = [
    _step("encode_categorical_features", {"columns": ["cat"]}),
    _step("scale_features", {"columns": ["plain"]}),
]
_ADEQUATE = [
    _step("impute_missing_values", {"column": "num", "strategy": "median"}),
    _step("impute_missing_values", {"column": "cat", "strategy": "mode"}),
    _step("encode_categorical_features", {"columns": ["cat"]}),
    _step("scale_features", {"columns": ["num", "plain"]}),
]


class _RecoversOnSecondAttempt:
    def __init__(self):
        self.calls = 0

    def generate_plan(self, context):
        self.calls += 1
        steps = _INADEQUATE if self.calls == 1 else _ADEQUATE
        return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))


def _stores():
    ds = InMemoryDatasetStore()
    ds.save("ds", _df())
    return ds, InMemorySplitStore(), InMemoryModelStore(), InMemoryRunStore()


class TestCompletedRunCarriesNoStaleFailureText:
    def test_completed_after_replan_clears_both_failure_and_error(self):
        ds, sp, mo, _ = _stores()
        provider = _RecoversOnSecondAttempt()
        graph = build_graph(ds, sp, mo, provider)

        result = graph.invoke(
            AgentState(run_id="r1", dataset_id="ds", target_column=TARGET, max_retries=2),
            config={"recursion_limit": 60},
        )

        assert provider.calls == 2, "this test is only meaningful if a REPLAN happened"
        assert result["status"] == "completed"
        assert result["failure"] is None
        assert not result.get("error"), f"stale error text survived: {result.get('error')!r}"

    def test_report_trace_event_of_a_completed_run_is_not_a_failure_message(self):
        """The demo-visible symptom: the final SSE event must not read as a
        failure on a run that succeeded."""
        ds, sp, mo, runs = _stores()
        graph = build_graph(ds, sp, mo, _RecoversOnSecondAttempt())

        stream_with_tracing(graph, AgentState(
            run_id="r2", dataset_id="ds", target_column=TARGET, max_retries=2,
        ), runs, config={"recursion_limit": 60})

        record = runs.get("r2")
        assert record.status == "completed"

        report_events = [e for e in runs.get_events("r2") if e.node == "report"]
        assert report_events
        for e in report_events:
            assert "inadequate" not in (e.message or "").lower()
            assert "material finding" not in (e.message or "").lower()

    def test_a_genuinely_failed_run_still_reports_its_error(self):
        """Control: clearing error on SUCCESS must not silence real failures.

        The plans must VARY — an identical repeat would (correctly) trip
        duplicate-plan detection and terminate as DUPLICATE_PLAN, which
        would test the wrong mechanism.
        """
        class _AlwaysInadequate:
            def __init__(self):
                self.calls = 0

            def generate_plan(self, context):
                self.calls += 1
                steps = [
                    _step("encode_categorical_features", {"columns": ["cat"]}),
                    _step("scale_features", {"columns": ["plain"] if self.calls == 1 else ["num"]}),
                ]
                return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))

        ds, sp, mo, _ = _stores()
        graph = build_graph(ds, sp, mo, _AlwaysInadequate())

        result = graph.invoke(
            AgentState(run_id="r3", dataset_id="ds", target_column=TARGET, max_retries=1),
            config={"recursion_limit": 60},
        )

        assert result["status"] == "failed"
        assert result["failure"] is not None
        assert result["failure"].category == "PLAN_ADEQUACY"
        assert result["failure"].human_intervention_required is True

    def test_first_attempt_success_is_unaffected(self):
        class _AlwaysAdequate:
            def generate_plan(self, context):
                return LLMProviderResult(success=True, plan=ProposedPlan(steps=_ADEQUATE))

        ds, sp, mo, _ = _stores()
        graph = build_graph(ds, sp, mo, _AlwaysAdequate())

        result = graph.invoke(
            AgentState(run_id="r4", dataset_id="ds", target_column=TARGET, max_retries=2),
            config={"recursion_limit": 60},
        )

        assert result["status"] == "completed"
        assert result["failure"] is None
        assert not result.get("error")
