"""
V1.2 Batch 1 — deterministic productization builders.

Pure functions over already-recorded run evidence. They never call an
LLM, never mutate a plan, never bypass validate_proposed_plan(), and
never copy PlanStep.reasoning / chain-of-thought into the product API.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from app.agent.plan_canonical import canonicalize_plan
from app.agent.plan_diff import PlanDiff, diff_plans
from app.agent.run_summary import build_run_summary
from app.agent.timeline import build_execution_timeline
from app.state_access import field
from app.schemas.adequacy import AdequacyFinding
from app.schemas.productization import (
    DecisionStage,
    DecisionTrace,
    EvidenceExport,
    ExecutableStep,
    HumanInterventionPackage,
    PiperVerdict,
    PlanningAttempt,
    StageId,
)
from app.schemas.trace_event import TraceEvent

STAGE_ORDER: list[tuple[StageId, str]] = [
    ("LLM_PROPOSED", "LLM proposed"),
    ("VALIDATED", "Validated"),
    ("ADEQUACY", "Adequacy"),
    ("REPLAN", "REPLAN"),
    ("EXECUTION", "Execution"),
    ("TRAINING", "Training"),
    ("EVALUATION", "Evaluation"),
    ("GUARDRAILS", "Guardrails"),
    ("FINAL_VERDICT", "Final verdict"),
]

_PLAN_NODES = frozenset({"plan", "plan_entry"})
_EXECUTION_NODES = frozenset({"clean", "feature_engineer"})
_TRAIN_NODES = frozenset({"train"})
_EVAL_NODES = frozenset({"evaluate", "compare"})
_GUARD_NODES = frozenset({"validate", "baseline"})


def executable_steps_from(steps: Any) -> list[ExecutableStep]:
    out: list[ExecutableStep] = []
    if not steps:
        return out
    for step in steps:
        if isinstance(step, ExecutableStep):
            out.append(step)
            continue
        if isinstance(step, dict):
            tool = step.get("tool_name")
            args = step.get("arguments") or {}
        else:
            tool = getattr(step, "tool_name", None)
            args = getattr(step, "arguments", None) or {}
        if isinstance(tool, str) and tool:
            out.append(ExecutableStep(tool_name=tool, arguments=dict(args)))
    return out


def make_planning_attempt(
    *,
    attempt: int,
    steps: Any,
    structurally_valid: bool,
    outcome: str,
    plan_hash: Optional[str] = None,
    adequacy_status: Optional[str] = None,
    violation_count: int = 0,
    material_finding_count: int = 0,
) -> PlanningAttempt:
    return PlanningAttempt(
        attempt=attempt,
        proposed_steps=executable_steps_from(steps),
        plan_hash=plan_hash,
        structurally_valid=structurally_valid,
        adequacy_status=adequacy_status,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        violation_count=violation_count,
        material_finding_count=material_finding_count,
    )


def _attempts_from_state(state: Any) -> list[PlanningAttempt]:
    raw = getattr(state, "planning_attempts", None) or []
    parsed: list[PlanningAttempt] = []
    for item in raw:
        if isinstance(item, PlanningAttempt):
            parsed.append(item)
        elif isinstance(item, dict):
            parsed.append(PlanningAttempt.model_validate(item))
        else:
            parsed.append(PlanningAttempt.model_validate(item.model_dump()))
    return parsed


def _as_canonicalizable(steps: list[ExecutableStep]) -> list:
    return [
        SimpleNamespace(tool_name=s.tool_name, arguments=s.arguments, status="pending")
        for s in steps
    ]


def build_plan_diffs(attempts: list[PlanningAttempt], target_column: str) -> list[PlanDiff]:
    diffs: list[PlanDiff] = []
    previous: Optional[PlanningAttempt] = None
    for current in attempts:
        if previous is not None and previous.proposed_steps and current.proposed_steps:
            old = canonicalize_plan(_as_canonicalizable(previous.proposed_steps), target_column)
            new = canonicalize_plan(_as_canonicalizable(current.proposed_steps), target_column)
            diffs.append(diff_plans(old, new))
        previous = current
    return diffs


def _nodes(events: list[TraceEvent]) -> set[str]:
    return {event.node for event in events}


def _event_type(event: TraceEvent) -> str:
    return getattr(event, "event_type", "") or ""


def _any_failed(events: list[TraceEvent], nodes: frozenset[str]) -> bool:
    return any(
        event.node in nodes
        and (
            _event_type(event) in {"node_failed", "run_failed"}
            or event.status == "failure"
        )
        for event in events
    )


def _stage(
    stage_id: StageId,
    label: str,
    status: str,
    summary: str,
    *,
    attempt: Optional[int] = None,
    evidence: Optional[dict] = None,
) -> DecisionStage:
    return DecisionStage(
        id=stage_id,
        label=label,
        status=status,  # type: ignore[arg-type]
        summary=summary,
        attempt=attempt,
        evidence=evidence or {},
    )


def _executed(state: Any) -> bool:
    return bool(
        getattr(state, "cleaning_log", None)
        or getattr(state, "cleaning_log", None)
        or getattr(state, "feature_log", None)
        or getattr(state, "feature_log", None)
    )


def build_decision_trace(
    run_id: str,
    run_status: str,
    events: list[TraceEvent],
    state: Any = None,
    *,
    target_column: str = "",
) -> DecisionTrace:
    attempts = _attempts_from_state(state) if state is not None else []
    target = target_column or (getattr(state, "target_column", "") if state is not None else "")
    diffs = build_plan_diffs(attempts, target) if target else []
    diff_payload = [item.model_dump(mode="json") for item in diffs]
    nodes = _nodes(events)
    retry_count = int(getattr(state, "retry_count", 0) or 0) if state is not None else 0
    last = attempts[-1] if attempts else None
    last_accepted = next((item for item in reversed(attempts) if item.outcome == "accepted"), None)
    executed = (_executed(state) if state is not None else False) or bool(nodes & _EXECUTION_NODES)

    stages: list[DecisionStage] = []

    if last is not None:
        if last.outcome == "provider_error" and not last.proposed_steps:
            stages.append(_stage(
                "LLM_PROPOSED", "LLM proposed", "failed",
                "The planner did not return a usable plan.",
                attempt=last.attempt,
                evidence={"step_count": 0, "outcome": last.outcome},
            ))
        else:
            stages.append(_stage(
                "LLM_PROPOSED", "LLM proposed", "passed",
                f"Planner proposed {len(last.proposed_steps)} executable step(s).",
                attempt=last.attempt,
                evidence={"step_count": len(last.proposed_steps), "outcome": last.outcome},
            ))
    elif nodes & _PLAN_NODES:
        stages.append(_stage("LLM_PROPOSED", "LLM proposed", "current", "Planning in progress."))
    else:
        stages.append(_stage("LLM_PROPOSED", "LLM proposed", "pending", "Waiting for the planner."))

    if last is None:
        stages.append(_stage("VALIDATED", "Validated", "pending", "Structural validation has not run yet."))
    elif last.outcome == "provider_error":
        stages.append(_stage("VALIDATED", "Validated", "not_reached", "No proposal reached validate_proposed_plan()."))
    elif last.structurally_valid:
        stages.append(_stage(
            "VALIDATED", "Validated", "passed",
            "validate_proposed_plan() accepted the proposal.",
            attempt=last.attempt,
            evidence={"plan_hash": last.plan_hash},
        ))
    else:
        stages.append(_stage(
            "VALIDATED", "Validated", "failed",
            f"validate_proposed_plan() rejected the proposal ({last.violation_count} violation(s)).",
            attempt=last.attempt,
            evidence={"violation_count": last.violation_count, "plan_hash": last.plan_hash},
        ))

    if last is None or last.outcome == "provider_error":
        stages.append(_stage("ADEQUACY", "Adequacy", "not_reached", "Adequacy was not evaluated."))
    elif not last.structurally_valid:
        stages.append(_stage("ADEQUACY", "Adequacy", "not_reached", "Adequacy is not evaluated for structurally invalid plans."))
    elif last.adequacy_status == "PASS":
        stages.append(_stage("ADEQUACY", "Adequacy", "passed", "evaluate_plan_adequacy() passed.", attempt=last.attempt))
    elif last.adequacy_status == "FAIL":
        stages.append(_stage(
            "ADEQUACY", "Adequacy", "failed",
            f"Material adequacy findings blocked execution ({last.material_finding_count}).",
            attempt=last.attempt,
            evidence={"material_finding_count": last.material_finding_count},
        ))
    else:
        stages.append(_stage("ADEQUACY", "Adequacy", "pending", "Adequacy result not recorded."))

    replan_event = any(
        event.node == "plan_entry" and _event_type(event) in {"node_completed", "replan_triggered"}
        for event in events
    )
    if retry_count > 0 or replan_event:
        stages.append(_stage(
            "REPLAN", "REPLAN", "passed",
            f"REPLAN occurred ({retry_count} retr{'y' if retry_count == 1 else 'ies'}).",
            evidence={"retry_count": retry_count, "diff_count": len(diff_payload)},
        ))
    elif last is not None and last.outcome in {"invalid", "inadequate", "provider_error"} and run_status not in {"completed", "failed"}:
        stages.append(_stage("REPLAN", "REPLAN", "current", "A retry may be in progress."))
    else:
        stages.append(_stage("REPLAN", "REPLAN", "skipped", "No REPLAN was required (or none has started)."))

    if executed:
        failed = _any_failed(events, _EXECUTION_NODES)
        stages.append(_stage(
            "EXECUTION", "Execution", "failed" if failed else "passed",
            "Cleaning / feature-engineering steps ran." if not failed else "Execution reported a failure.",
        ))
    elif last_accepted is not None and run_status not in {"failed", "completed"}:
        stages.append(_stage("EXECUTION", "Execution", "current", "A validated plan is executing."))
    elif last_accepted is None:
        stages.append(_stage("EXECUTION", "Execution", "not_reached", "No validated plan was executed."))
    else:
        stages.append(_stage("EXECUTION", "Execution", "pending", "Execution has not been recorded yet."))

    model_results = list(getattr(state, "model_results", []) or []) if state is not None else []
    if model_results or nodes & _TRAIN_NODES:
        failed = _any_failed(events, _TRAIN_NODES)
        stages.append(_stage(
            "TRAINING", "Training", "failed" if failed else "passed",
            f"{len(model_results)} candidate model(s) trained." if model_results else "Training node ran.",
        ))
    elif executed:
        pending = run_status not in {"failed", "completed"}
        stages.append(_stage("TRAINING", "Training", "pending" if pending else "not_reached", "Training has not completed."))
    else:
        stages.append(_stage("TRAINING", "Training", "not_reached", "Training did not run."))

    evaluation_results = list(getattr(state, "evaluation_results", []) or []) if state is not None else []
    comparison = getattr(state, "comparison", None) if state is not None else None
    if evaluation_results or comparison is not None or nodes & _EVAL_NODES:
        failed = _any_failed(events, _EVAL_NODES)
        stages.append(_stage(
            "EVALUATION", "Evaluation", "failed" if failed else "passed",
            "Candidates were evaluated and compared." if comparison is not None else "Evaluation ran.",
        ))
    elif model_results:
        pending = run_status not in {"failed", "completed"}
        stages.append(_stage("EVALUATION", "Evaluation", "pending" if pending else "not_reached", "Evaluation has not completed."))
    else:
        stages.append(_stage("EVALUATION", "Evaluation", "not_reached", "Evaluation did not run."))

    validation = field(state, "validation") if state is not None else None
    if validation is not None:
        ok = field(validation, "valid") is True
        stages.append(_stage(
            "GUARDRAILS", "Guardrails", "passed" if ok else "failed",
            "Deterministic guardrails passed." if ok else "Deterministic guardrails failed.",
            evidence={"valid": ok},
        ))
    elif nodes & _GUARD_NODES:
        failed = _any_failed(events, _GUARD_NODES)
        stages.append(_stage("GUARDRAILS", "Guardrails", "failed" if failed else "current", "Guardrail node ran."))
    else:
        stages.append(_stage("GUARDRAILS", "Guardrails", "not_reached", "Guardrails have not run."))

    if run_status == "completed":
        stages.append(_stage("FINAL_VERDICT", "Final verdict", "passed", "Run completed."))
    elif run_status == "failed":
        stages.append(_stage("FINAL_VERDICT", "Final verdict", "failed", "Run ended in a structured failure."))
    else:
        stages.append(_stage("FINAL_VERDICT", "Final verdict", "current", f"Run is '{run_status}'."))

    return DecisionTrace(
        run_id=run_id,
        run_status=run_status,
        stages=stages,
        planning_attempts=attempts,
        plan_diffs=diff_payload,
    )


def build_verdict(run_id: str, run_status: str, state: Any) -> PiperVerdict:
    attempts = _attempts_from_state(state)
    last = attempts[-1] if attempts else None
    failure = getattr(state, "failure", None)
    validation = field(state, "validation")
    retry_count = int(field(state, "retry_count", default=0) or 0)
    max_retries = int(field(state, "max_retries", default=2) or 2)
    human = False
    if failure is not None:
        human = bool(field(failure, "human_intervention_required", default=False))
    executed = _executed(state)
    structurally_valid = bool(last.structurally_valid) if last is not None else False
    if last is None or last.adequacy_status is None:
        adequacy_passed: Optional[bool] = None
    else:
        adequacy_passed = last.adequacy_status == "PASS"
    if validation is None:
        guardrails_passed: Optional[bool] = None
    else:
        guardrails_passed = field(validation, "valid") is True

    category = getattr(failure, "category", None) if failure is not None else None

    if run_status == "completed" and guardrails_passed:
        return PiperVerdict(
            run_id=run_id,
            outcome="ACCEPTED",
            reason_code="ACCEPTED_GUARDRAILS_PASSED",
            summary=(
                "PIPER accepted this run: the plan passed structural validation and adequacy, "
                "models trained, and guardrails passed."
            ),
            retry_count=retry_count,
            max_retries=max_retries,
            structurally_valid_plan=True,
            adequacy_passed=True,
            guardrails_passed=True,
            human_intervention_required=False,
            executed=executed,
        )

    reason_code = "REJECTED"
    summary = "PIPER rejected this run."
    if category == "DUPLICATE_PLAN":
        reason_code = "REJECTED_DUPLICATE_PLAN"
        summary = "PIPER stopped: the planner repeated a plan that was already proposed this run."
    elif category == "PLAN_ADEQUACY":
        reason_code = "REJECTED_INADEQUATE_PLAN"
        summary = "PIPER stopped: the plan was structurally valid but failed deterministic adequacy."
    elif category == "EVALUATION_ERROR" and last is not None and not last.structurally_valid:
        reason_code = "REJECTED_INVALID_PLAN"
        summary = "PIPER stopped: validate_proposed_plan() rejected the proposal. Nothing executed."
    elif category == "EVALUATION_ERROR":
        reason_code = "REJECTED_PLANNER_ERROR"
        summary = "PIPER stopped: the planner/provider failed before a valid plan could be accepted."
    elif category == "BASELINE_GATE_FAILED":
        reason_code = "REJECTED_GUARDRAIL"
        summary = "PIPER stopped: the trained model failed the majority-class baseline gate."
    elif category in {"DATA_ERROR", "SCHEMA_ERROR", "TARGET_ERROR"}:
        reason_code = "REJECTED_INPUT"
        summary = "PIPER stopped: the dataset or target is not usable for this V1 pipeline."
    elif category == "EXECUTION_BUDGET_EXCEEDED":
        reason_code = "REJECTED_BUDGET"
        summary = "PIPER stopped: the hard execution-step budget was reached."
    elif guardrails_passed is False:
        reason_code = "REJECTED_GUARDRAIL"
        summary = "PIPER stopped: deterministic guardrails did not pass."
    elif failure is not None:
        summary = failure.message

    outcome = "HUMAN_INTERVENTION_REQUIRED" if human else "REJECTED"
    if outcome == "HUMAN_INTERVENTION_REQUIRED":
        summary = "Human intervention is required. " + summary

    return PiperVerdict(
        run_id=run_id,
        outcome=outcome,
        reason_code=reason_code,
        summary=summary,
        retry_count=retry_count,
        max_retries=max_retries,
        structurally_valid_plan=structurally_valid,
        adequacy_passed=adequacy_passed,
        guardrails_passed=guardrails_passed,
        human_intervention_required=human,
        executed=executed,
    )


_ACTIONS: dict[str, list[str]] = {
    "DUPLICATE_PLAN": [
        "The planner repeated an already-seen plan. Inspect the last proposal and the earlier rejection.",
        "Do not expect a different executable outcome from the same plan hash.",
        "If you retry, change the dataset, target, or planner model — not PIPER's validator.",
    ],
    "PLAN_ADEQUACY": [
        "The proposal was well-formed but incomplete. Read the material adequacy findings.",
        "Preserved valid operations can stay; implicated operations must be revised.",
        "Do not hand-edit JSON into the executor — invalid plans still cannot run.",
    ],
    "EVALUATION_ERROR": [
        "Either the planner produced a structurally invalid plan, or the provider failed.",
        "Check Ollama is running and the model name is installed, then inspect structural violations if present.",
        "validate_proposed_plan() remains the only gate; do not bypass it.",
    ],
    "BASELINE_GATE_FAILED": [
        "The model did not beat the majority-class baseline. Review metrics and class balance.",
        "Consider whether the target or features make this problem ill-posed for V1 classification.",
    ],
    "DATA_ERROR": ["Fix or replace the dataset, then start a new run."],
    "SCHEMA_ERROR": ["Fix the table schema (columns/types), then start a new run."],
    "TARGET_ERROR": ["Choose a binary/multiclass classification target PIPER V1 supports."],
    "EXECUTION_BUDGET_EXCEEDED": [
        "The run hit PIPER's hard step ceiling. This is a safety stop, not a planner hint.",
    ],
}


def _findings_from_evidence(raw: Any) -> list[AdequacyFinding]:
    if not isinstance(raw, list):
        return []
    findings: list[AdequacyFinding] = []
    for item in raw:
        try:
            findings.append(AdequacyFinding.model_validate(item))
        except Exception:
            continue
    return findings


def build_intervention(run_id: str, run_status: str, state: Any) -> HumanInterventionPackage:
    attempts = _attempts_from_state(state)
    last = attempts[-1] if attempts else None
    failure = getattr(state, "failure", None)
    retry_count = int(getattr(state, "retry_count", 0) or 0)
    max_retries = int(getattr(state, "max_retries", 2) or 2)
    evidence = dict(getattr(failure, "evidence", {}) or {}) if failure is not None else {}
    category = getattr(failure, "category", None) if failure is not None else None
    message = getattr(failure, "message", None) if failure is not None else None

    if run_status == "completed":
        required = False
        headline = "No human intervention required."
        actions = ["This run completed under deterministic guardrails. Review the verdict and metrics as usual."]
    else:
        required = False
        if failure is not None:
            human_flag = bool(getattr(failure, "human_intervention_required", False))
            retryable_flag = bool(getattr(failure, "retryable", True))
            required = human_flag or not retryable_flag
        elif run_status == "failed":
            required = True
        headline = (
            "Human review is required before treating this run as a successful pipeline."
            if required
            else "Human review is recommended."
        )
        actions = list(_ACTIONS.get(category or "", [
            "Read the structured failure category and evidence.",
            "Invalid LLM plans cannot execute; do not attempt to repair them by mutating JSON.",
        ]))

    structural_violations = evidence.get("violations") if isinstance(evidence.get("violations"), list) else []
    material = _findings_from_evidence(evidence.get("findings"))
    advisory = _findings_from_evidence(evidence.get("advisory_findings"))
    preserved = executable_steps_from(evidence.get("valid_steps") or [])
    implicated = executable_steps_from(evidence.get("implicated_steps") or [])
    last_steps = last.proposed_steps if last is not None else executable_steps_from(
        evidence.get("proposed_steps") or evidence.get("rejected_steps") or []
    )

    blocked = not _executed(state)
    if last is not None and last.outcome in {"invalid", "duplicate", "inadequate", "provider_error"}:
        blocked = True

    return HumanInterventionPackage(
        run_id=run_id,
        required=required,
        headline=headline,
        failure_category=category,
        failure_message=message,
        retry_count=retry_count,
        max_retries=max_retries,
        last_proposed_steps=last_steps,
        structural_violations=list(structural_violations),
        material_adequacy_findings=material,
        advisory_adequacy_findings=advisory,
        preserved_valid_steps=preserved,
        implicated_steps=implicated,
        recommended_actions=actions,
        blocked_invalid_execution=blocked,
    )


def _executed_operations(state: Any) -> list[ExecutableStep]:
    ops: list[ExecutableStep] = []
    for log_name in ("cleaning_log", "cleaning_log", "feature_log", "feature_log"):
        ops.extend(executable_steps_from(getattr(state, log_name, None) or []))
    return ops


def build_evidence_export(
    run_id: str,
    run_status: str,
    events: list[TraceEvent],
    state: Any,
    *,
    dataset_id: str,
    target_column: str,
) -> EvidenceExport:
    trace = build_decision_trace(run_id, run_status, events, state, target_column=target_column)
    intervention = build_intervention(run_id, run_status, state)
    verdict = None
    summary = None
    if run_status in {"completed", "failed"} and state is not None:
        verdict = build_verdict(run_id, run_status, state)
        try:
            built = build_run_summary(run_id, state)
            summary = built.model_dump(mode="json") if built is not None else None
        except Exception:
            summary = None
    timeline = build_execution_timeline(run_id, events)
    return EvidenceExport(
        run_id=run_id,
        dataset_id=dataset_id,
        target_column=target_column,
        status=run_status,
        decision_trace=trace,
        verdict=verdict,
        intervention=intervention,
        summary=summary,
        timeline=timeline,
        validation=getattr(state, "validation", None) if state is not None else None,
        comparison=getattr(state, "comparison", None) if state is not None else None,
        baseline=getattr(state, "baseline", None) if state is not None else None,
        failure=getattr(state, "failure", None) if state is not None else None,
        reproducibility=getattr(state, "reproducibility", None) if state is not None else None,
        model_results=list(getattr(state, "model_results", []) or []) if state is not None else [],
        evaluation_results=list(getattr(state, "evaluation_results", []) or []) if state is not None else [],
        executed_operations=_executed_operations(state) if state is not None else [],
        notes=[
            "This export is a read-only snapshot of deterministic PIPER evidence.",
            "LLM chain-of-thought is intentionally omitted.",
            "validate_proposed_plan() remains the sole structural authority; this file does not repair plans.",
        ],
    )
