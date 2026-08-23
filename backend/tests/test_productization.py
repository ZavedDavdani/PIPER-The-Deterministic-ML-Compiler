"""V1.2 Batch 1 — productization layer.

Read-only decision trace, plan diff, verdict, intervention, and
evidence export. These tests must not weaken validate_proposed_plan().
"""

from __future__ import annotations

from types import SimpleNamespace

from app.agent.plan_validation import validate_proposed_plan
from app.agent.productization import (
    build_decision_trace,
    build_evidence_export,
    build_intervention,
    build_plan_diffs,
    build_verdict,
    executable_steps_from,
    make_planning_attempt,
)
from app.agent.state import AgentState, OperationRecord, PlanStep
from app.schemas.failure import FailureInfo
from app.schemas.guardrails import PipelineValidationResult, ValidationCheck
from app.schemas.productization import PlanningAttempt
from app.schemas.trace_event import TraceEvent


SECRET_REASONING = "SECRET_CHAIN_OF_THOUGHT_DO_NOT_LEAK"


def _step(tool_name: str, arguments: dict, reasoning: str = SECRET_REASONING) -> PlanStep:
    return PlanStep(
        step_id="s1",
        action="do it",
        tool_name=tool_name,
        arguments=arguments,
        reasoning=reasoning,
        status="pending",
    )


def _event(run_id: str, node: str, event_type: str = "node_completed", status: str = "success") -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        step_id=f"trace_{node}_{event_type}",
        attempt=0,
        node=node,
        event_type=event_type,
        timestamp="t0",
        status=status,
    )


def _completed_state(**kwargs) -> AgentState:
    validation = PipelineValidationResult(
        dataset_id="d1",
        target_column="Survived",
        valid=True,
        checks=[ValidationCheck(check="leakage", passed=True, severity="info", message="ok")],
    )
    defaults = dict(
        run_id="run_ok",
        dataset_id="d1",
        target_column="Survived",
        status="completed",
        retry_count=0,
        max_retries=2,
        validation=validation,
        cleaning_log=[
            OperationRecord(
                operation_id="op1",
                tool_name="impute_missing_values",
                arguments={"column": "Age", "strategy": "median"},
                result_summary="imputed",
                reason="median imputation applied",
                timestamp="t0",
            )
        ],
        planning_attempts=[
            make_planning_attempt(
                attempt=0,
                steps=[_step("impute_missing_values", {"column": "Age", "strategy": "median"})],
                structurally_valid=True,
                outcome="accepted",
                plan_hash="abc",
                adequacy_status="PASS",
            )
        ],
    )
    defaults.update(kwargs)
    return AgentState(**defaults)


class TestReasoningIsStripped:
    def test_executable_steps_drop_reasoning(self):
        steps = executable_steps_from([_step("drop_column", {"column": "Cabin"})])
        dumped = steps[0].model_dump()
        assert dumped == {"tool_name": "drop_column", "arguments": {"column": "Cabin"}}
        assert "reasoning" not in dumped
        assert SECRET_REASONING not in str(dumped)

    def test_decision_trace_export_never_contains_reasoning(self):
        state = _completed_state()
        events = [
            _event("run_ok", "plan"),
            _event("run_ok", "clean"),
            _event("run_ok", "train"),
            _event("run_ok", "validate"),
        ]
        trace = build_decision_trace("run_ok", "completed", events, state, target_column="Survived")
        blob = trace.model_dump_json()
        assert SECRET_REASONING not in blob
        assert "reasoning" not in blob


class TestInvalidPlansDoNotExecute:
    def test_columns_array_on_drop_column_is_still_a_violation(self):
        proposed = [SimpleNamespace(tool_name="drop_column", arguments={"columns": ["Cabin"]})]
        result = validate_proposed_plan(proposed, "Survived")
        assert result.valid is False
        state = AgentState(
            run_id="run_bad",
            dataset_id="d1",
            target_column="Survived",
            status="failed",
            failure=FailureInfo(
                category="EVALUATION_ERROR",
                message="invalid",
                evidence={"violations": [v.model_dump(mode="json") for v in result.violations]},
                node="plan",
                attempt=0,
                retryable=True,
                human_intervention_required=False,
            ),
            planning_attempts=[
                make_planning_attempt(
                    attempt=0,
                    steps=proposed,
                    structurally_valid=False,
                    outcome="invalid",
                    violation_count=len(result.violations),
                )
            ],
        )
        verdict = build_verdict("run_bad", "failed", state)
        intervention = build_intervention("run_bad", "failed", state)
        assert verdict.reason_code == "REJECTED_INVALID_PLAN"
        assert verdict.executed is False
        assert intervention.blocked_invalid_execution is True
        trace = build_decision_trace(
            "run_bad",
            "failed",
            [_event("run_bad", "plan", "node_failed", "failure")],
            state,
            target_column="Survived",
        )
        by_id = {s.id: s for s in trace.stages}
        assert by_id["EXECUTION"].status == "not_reached"
        assert by_id["VALIDATED"].status == "failed"


class TestPlanDiffIsDeterministic:
    def test_added_step_is_reported_without_reasoning(self):
        first = make_planning_attempt(
            attempt=0,
            steps=[_step("impute_missing_values", {"column": "Age", "strategy": "median"})],
            structurally_valid=True,
            outcome="inadequate",
            adequacy_status="FAIL",
            material_finding_count=1,
        )
        second = make_planning_attempt(
            attempt=1,
            steps=[
                _step("impute_missing_values", {"column": "Age", "strategy": "median"}),
                _step("drop_column", {"column": "Cabin"}),
            ],
            structurally_valid=True,
            outcome="accepted",
            adequacy_status="PASS",
        )
        diffs = build_plan_diffs([first, second], "Survived")
        assert len(diffs) == 1
        assert diffs[0].is_duplicate is False
        assert len(diffs[0].added) == 1
        assert diffs[0].added[0].tool_name == "drop_column"
        assert diffs[0].added[0].arguments == {"column": "Cabin"}
        assert SECRET_REASONING not in diffs[0].model_dump_json()


class TestVerdict:
    def test_completed_valid_run_is_accepted(self):
        verdict = build_verdict("run_ok", "completed", _completed_state())
        assert verdict.outcome == "ACCEPTED"
        assert verdict.reason_code == "ACCEPTED_GUARDRAILS_PASSED"
        assert verdict.human_intervention_required is False

    def test_duplicate_plan_requires_human_intervention(self):
        state = AgentState(
            run_id="run_dup",
            dataset_id="d1",
            target_column="Survived",
            status="failed",
            retry_count=1,
            max_retries=2,
            failure=FailureInfo(
                category="DUPLICATE_PLAN",
                message="duplicate",
                evidence={"duplicate_plan_hash": "abc"},
                node="plan",
                attempt=1,
                retryable=False,
                human_intervention_required=True,
            ),
            planning_attempts=[
                make_planning_attempt(
                    attempt=1,
                    steps=[_step("scale_features", {"columns": ["Age"]})],
                    structurally_valid=True,
                    outcome="duplicate",
                    plan_hash="abc",
                )
            ],
        )
        verdict = build_verdict("run_dup", "failed", state)
        assert verdict.outcome == "HUMAN_INTERVENTION_REQUIRED"
        assert verdict.reason_code == "REJECTED_DUPLICATE_PLAN"
        package = build_intervention("run_dup", "failed", state)
        assert package.required is True
        assert package.last_proposed_steps[0].tool_name == "scale_features"
        assert SECRET_REASONING not in package.model_dump_json()


class TestEvidenceExport:
    def test_completed_export_contains_trace_verdict_and_no_reasoning(self):
        state = _completed_state()
        events = [
            _event("run_ok", "plan"),
            _event("run_ok", "clean"),
            _event("run_ok", "train"),
            _event("run_ok", "evaluate"),
            _event("run_ok", "validate"),
            _event("run_ok", "report", "run_completed"),
        ]
        export = build_evidence_export(
            "run_ok", "completed", events, state, dataset_id="d1", target_column="Survived",
        )
        blob = export.model_dump_json()
        assert SECRET_REASONING not in blob
        assert export.verdict is not None
        assert export.verdict.outcome == "ACCEPTED"
        assert export.executed_operations[0].tool_name == "impute_missing_values"
        ids = [s.id for s in export.decision_trace.stages]
        assert ids == [
            "LLM_PROPOSED", "VALIDATED", "ADEQUACY", "REPLAN",
            "EXECUTION", "TRAINING", "EVALUATION", "GUARDRAILS", "FINAL_VERDICT",
        ]

    def test_planning_attempt_model_forbids_reasoning_field(self):
        attempt = PlanningAttempt(
            attempt=0,
            proposed_steps=executable_steps_from([_step("drop_column", {"column": "x"})]),
            structurally_valid=True,
            outcome="accepted",
        )
        assert "reasoning" not in attempt.model_dump()
