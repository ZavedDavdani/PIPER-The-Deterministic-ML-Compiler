"""
Real pipeline nodes — replaces the M2 stubs (train_node, evaluate_node,
validate_node from m2_nodes.py) with implementations that call the
actual tools built after M2: split_dataset(), train_model(),
evaluate_model(), validate_pipeline().

profile_node and clean_node are reused as-is from m2_nodes.py (already
real). plan_node is extended here (plan_node_v2) to also plan feature-
engineering steps, since the real pipeline needs a feature-engineering
intent before it can train.

New flow (per the locked graph shape):

    PROFILE -> PLAN -> CLEAN -> FEATURE_ENGINEER -> SPLIT -> TRAIN
    -> EVALUATE -> VALIDATE -> PASS/REPLAN

No ML logic lives in these node functions — every node is a thin
orchestration layer around a real tool call, per the explicit
instruction not to duplicate ML logic inside graph nodes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from app.agent.plan_adequacy import classify_plan_steps, evaluate_plan_adequacy
from app.agent.plan_canonical import canonicalize_plan
from app.agent.plan_diff import diff_plans
from app.agent.plan_validation import ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS, validate_proposed_plan
from app.agent.state import AgentState, OperationRecord, PlanStep, ToolTraceEntry
from app.agent.tools import (
    build_sanitized_llm_context,
    compare_models,
    encode_categorical_features,
    evaluate_model,
    scale_features,
    split_dataset,
    train_model,
)
from app.agent.tools.context_budget import apply_context_budget
from app.agent.tools.guardrails import validate_pipeline
from app.agent.tools.baseline import compute_baseline
from app.agent.tools.data_quality import validate_data_quality
from app.agent.tools.preparation import RANDOM_STATE as SPLIT_RANDOM_STATE
from app.agent.tools.sanitization import sanitize_for_llm_context
from app.agent.tools.training import RANDOM_STATE as MODEL_RANDOM_STATE
from app.llm.provider import LLMPlanningContext
from app.schemas.failure import FailureInfo
from app.schemas.reproducibility import (
    ReproducibilityMetadata,
    capture_environment_metadata,
    dataset_fingerprint,
)
from app.schemas.training import FeatureEngineeringIntent
from app.storage import DatasetStore, InMemoryModelStore, SplitStore
from app.storage.exceptions import SplitNotFoundError

TEST_SIZE = 0.2  # locked elsewhere (split_dataset's own tests use this)

# V1 multi-model candidate set for the graph's TRAIN stage — fixed,
# not LLM-choosable (mirrors the existing fixed-algorithm precedent
# from the prior single-model dummy default). Both algorithms are
# already fully supported by train_model()'s allowlist; no new model
# types are introduced by wiring compare_models() into the graph.
_TRAIN_CANDIDATES: list[tuple[str, dict]] = [
    ("random_forest", {"n_estimators": 200, "max_depth": 15}),
    ("logistic_regression", {"C": 1.0, "max_iter": 1000}),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _trace(node: str, tool_name: str, arguments: dict, success: bool, result_summary: dict, duration_ms: float) -> ToolTraceEntry:
    return ToolTraceEntry(
        trace_id=_new_id("trace"),
        timestamp=_now(),
        node=node,
        tool_name=tool_name,
        arguments=arguments,
        result=result_summary,
        success=success,
        duration_ms=duration_ms,
    )


def validate_input_node(state: AgentState, store: DatasetStore) -> dict:
    """
    Runs BEFORE profile_node — the "is this dataset even usable" gate
    (section 9). Every violation here is terminal: DATA_ERROR,
    SCHEMA_ERROR, or TARGET_ERROR, per DATA_QUALITY_FAILURE_CATEGORY.
    No replan can fix a malformed input, so this always routes
    straight to REPORT with retry_count untouched — the same
    precedent constraint #9 established for non-binary targets,
    generalized to the rest of the input-validation surface.
    """
    from app.schemas.data_quality import DATA_QUALITY_FAILURE_CATEGORY
    from app.schemas.failure import FailureInfo

    result = validate_data_quality(state.dataset_id, state.target_column, store)
    trace = _trace(
        "input_validator", "validate_data_quality",
        {"dataset_id": state.dataset_id, "target_column": state.target_column},
        result.success, {"message": result.message}, 0.0,
    )

    if not result.success:
        # Tool-level failure (e.g. dataset doesn't exist at all) —
        # still terminal, still DATA_ERROR.
        failure = FailureInfo(
            category="DATA_ERROR",
            message=result.message,
            evidence={"tool_error_code": result.error.code},
            node="validate_input",
            attempt=state.retry_count,
            retryable=False,
            human_intervention_required=True,
        )
        return {
            "status": "failed",
            "task_type": None,
            "error": result.message,
            "failure": failure,
            "tool_trace": state.tool_trace + [trace],
        }

    if not result.data.valid:
        # Pick the category of the FIRST violation for the top-level
        # FailureInfo.category — all violations remain visible in
        # evidence regardless, so nothing is lost by this choice, it
        # just needs one category to classify the overall failure.
        first_violation = result.data.violations[0]
        category = DATA_QUALITY_FAILURE_CATEGORY[first_violation.check_type]

        failure = FailureInfo(
            category=category,
            message=result.message,
            evidence={"violations": [v.model_dump() for v in result.data.violations]},
            node="validate_input",
            attempt=state.retry_count,
            retryable=False,
            human_intervention_required=True,
        )
        return {
            "status": "failed",
            "task_type": None,
            "error": result.message,
            "failure": failure,
            "tool_trace": state.tool_trace + [trace],
        }

    return {
        "status": "running",
        "tool_trace": state.tool_trace + [trace],
    }


def sanitize_node(state: AgentState, store: DatasetStore) -> dict:
    """
    Section 10, runs after PROFILE and before PLAN — matches the
    locked diagram (PROFILE -> SANITIZATION -> PLAN). Builds
    state.sanitization_report as evidence; never touches the dataset
    in DatasetStore. This is defense-in-depth ahead of M3's eventual
    LLM planner reading dataset sample values as context — the
    sanitization report itself doesn't currently gate anything (no
    LLM exists yet to protect), but the evidence is captured now so
    M3's planner integration has it ready rather than needing this
    built retroactively.
    """
    result = sanitize_for_llm_context(state.dataset_id, store)
    trace = _trace(
        "sanitizer", "sanitize_for_llm_context",
        {"dataset_id": state.dataset_id},
        result.success, {"message": result.message}, 0.0,
    )

    if not result.success:
        # Sanitization failing to RUN (e.g. dataset vanished between
        # nodes) is a genuine failure — but note this is different
        # from sanitization finding adversarial content, which is not
        # a failure at all, just evidence.
        return {
            "status": "failed",
            "error": f"sanitize_for_llm_context failed: {result.error.message}",
            "tool_trace": state.tool_trace + [trace],
        }

    return {
        "status": "running",
        "sanitization_report": result.data,
        "tool_trace": state.tool_trace + [trace],
    }


def _carried_forward_preserved_steps(state: AgentState) -> dict:
    """
    Carries an EARLIER attempt's already-classified `valid_steps` /
    `implicated_steps` forward across a provider parse/transport failure.

    Confirmed gap (adequacy-recovery benchmark v2, 3 of 14 real calls):
    a PLAN_ADEQUACY failure emits `valid_steps` so the next REPLAN can
    PATCH rather than regenerate — but if the very next Ollama call fails
    at the transport/parse level (timeout, malformed JSON), plan_node_v2
    returns an EVALUATION_ERROR carrying no classification at all. The
    attempt AFTER that then plans from scratch, discarding preservation
    state that nothing had actually invalidated. All 3 observed
    "chain broken" REPLANs came from exactly this path.

    This is deliberately additive EVIDENCE propagation only:

    - Nothing malformed is ever treated as valid. A parse/transport
      failure produced no plan at all, so there is no model output here
      to trust, merge, or auto-correct — the steps carried forward are
      the ones an EARLIER attempt already put through
      validate_proposed_plan() successfully.
    - They are re-validated anyway (below) before being re-stated, so a
      carried step can never instruct the model to reproduce something
      the validator would now reject.
    - Steps are copied VERBATIM in the same production
      {"tool_name": ..., "arguments": {...}} shape — never rewritten,
      renamed, or merged.
    - Validator, retry accounting, routing, and duplicate-plan semantics
      are untouched; this only changes what the next prompt is TOLD.

    Returns {} when there is nothing to carry (first attempt, a previous
    failure that never carried a classification, or carried steps that no
    longer re-validate).
    """
    if state.failure is None:
        return {}
    evidence = state.failure.evidence
    if not isinstance(evidence, dict):
        return {}

    valid_steps = evidence.get("valid_steps")
    if not isinstance(valid_steps, list) or not valid_steps:
        return {}
    if not all(
        isinstance(s, dict) and isinstance(s.get("tool_name"), str) and isinstance(s.get("arguments"), dict)
        for s in valid_steps
    ):
        return {}

    # Re-validation, not re-trust: validate_proposed_plan() is duck-typed
    # on .tool_name/.arguments, so the carried dicts can be re-checked
    # through the SAME sole authority without reconstructing provider
    # types. If any carried step no longer passes, carry nothing —
    # partially-valid preservation evidence would be worse than none.
    revalidation = validate_proposed_plan(
        [SimpleNamespace(tool_name=s["tool_name"], arguments=s["arguments"]) for s in valid_steps],
        state.target_column,
    )
    if not revalidation.valid:
        return {}

    carried = {"valid_steps": valid_steps}
    implicated = evidence.get("implicated_steps")
    if isinstance(implicated, list) and implicated:
        carried["implicated_steps"] = implicated
    return carried


def plan_node_v2(
    state: AgentState,
    store: DatasetStore,
    llm_provider,
) -> dict:
    """
    M3 Phase 5: real LLM-driven planning, replacing the M2/Phase-4
    dummy heuristic. The heuristic's SHAPE (cleaning steps, then
    feature-engineering steps) is preserved as the CONTRACT — nothing
    downstream (CLEAN/FEATURE_ENGINEER/SPLIT/... /VALIDATE) changes —
    but the plan's CONTENT now genuinely comes from llm_provider
    (Protocol: app.llm.provider.LLMProvider), not a fixed rule set.

    Architecture (every stage mandatory, in this order — see module
    docstring's locked M3 invariant: LLM proposes, PIPER decides):

        build_sanitized_llm_context()      -- never raw dataset content
            |
        apply_context_budget()             -- Batch 7: deterministic size
            |                                  estimate + reduction, no-op
            |                                  for any dataset small enough
            |                                  not to need it
            |
        llm_provider.generate_plan()       -- untrusted proposal
            |
        [provider failure?] -> structured FailureInfo, no execution
            |
        validate_proposed_plan()           -- deterministic allowlist/
            |                                  argument check (Phase 4)
        [invalid?] -> structured FailureInfo, no execution
            |
        list[PlanStep] construction        -- ONLY from a plan that
            |                                  has passed validation
        canonicalize_plan() + plan_hash()  -- unchanged from before
            |
        duplicate check against plan_history -- unchanged from before
            |
        state.plan

    The LLM NEVER: constructs a PlanStep directly (it only produces
    ProposedPlanStep, a distinct, less-trusted type — see
    app/llm/provider.py), bypasses validate_proposed_plan(), decides
    graph routing (this node still only returns
    status="running"/"failed"; _route_after_plan/_route_after_validate,
    unchanged, make every routing decision), or influences plan_hash()
    (canonicalize_plan() still excludes reasoning/action/step_id/status
    exactly as before — an LLM's rationale can never change plan
    identity).

    REPLAN: triggered the same way as before (state.validation.valid
    is False and retry budget remains — decided entirely by
    _route_after_validate, not by this node or the LLM). On a REPLAN
    entry, state.plan/state.failure hold the PREVIOUS attempt's
    executable plan and structured failure — this node builds
    failure_context from state.failure and, using diff_plans()
    (Phase 5's PlanDiff activation — previously dormant, now real
    production-path functionality), a previous_plan_summary so the LLM
    has structured evidence of what to change, not just "try again."
    """
    if state.failure is not None and state.failure.category == "EXECUTION_BUDGET_EXCEEDED":
        # M4: PLAN_ENTRY (_increment_retry_if_replanning in graph.py)
        # already determined the hard execution-step budget was
        # exhausted and set a terminal FailureInfo accordingly — this
        # node must pass that through unchanged (no LLM call, no
        # re-planning) rather than silently overriding it, so
        # _route_after_plan's existing status=="failed" -> REPORT
        # routing carries the real reason through to the final result.
        return {}

    if state.profile is None:
        return {"status": "failed", "error": "plan_node_v2 reached with no profile in state."}

    sanitized_context_result = build_sanitized_llm_context(state.dataset_id, state.target_column, store)
    if not sanitized_context_result.success:
        failure = FailureInfo(
            category="DATA_ERROR",
            message=f"Could not build sanitized LLM context: {sanitized_context_result.error.message}",
            evidence={"dataset_id": state.dataset_id},
            node="plan",
            attempt=state.retry_count,
            retryable=False,
            human_intervention_required=True,
        )
        return {"status": "failed", "error": failure.message, "failure": failure}

    # Batch 7: deterministic context-budgeting — estimates the sanitized
    # context's serialized size and, only if it exceeds
    # DEFAULT_MAX_CONTEXT_CHARS, applies a staged, non-lossy-of-essential-
    # information reduction (sample_values shrink first, down to none;
    # column names/dtypes/target/missing%/unique%/min/max/mean are never
    # touched). A no-op for any dataset small enough not to need it —
    # the real reference Telco CSV's context stays completely unbudgeted
    # (see context_budget.py's own docstring for the real measurement).
    budgeted_context, _budget_report = apply_context_budget(sanitized_context_result.data)

    # Batch 5 fix: previously required len(state.plan) > 0 too, which
    # silently dropped failure_context/previous_plan_summary on a
    # REPLAN triggered by a retryable PLAN-node failure (LLM provider
    # error, or a proposal that failed validate_proposed_plan()) — in
    # both cases state.plan is still [] at this point, since plan_node_v2
    # never got far enough to construct PlanStep objects on the failed
    # attempt. state.failure is None only on the very first PLAN entry
    # (nothing has failed yet), so its presence alone is a sufficient and
    # correct "is this a REPLAN" signal — equivalent to the old, tighter
    # condition for every previously-tested (VALIDATE-triggered) REPLAN,
    # since state.plan is always non-empty by the time VALIDATE can fail.
    is_replan = state.failure is not None

    failure_context = None
    previous_plan_summary = None

    if is_replan:
        failure_context = state.failure.model_dump(mode="json")

        # PlanDiff activation (Phase 5): the previous attempt's
        # executable plan, canonicalized the same way plan_hash()
        # already does, diffed against itself as a structural summary
        # for the LLM. There is no "new" canonical plan to diff
        # against yet at this point (the LLM hasn't proposed one) —
        # diff_plans() genuinely needs two plans to compare, so what
        # we give the LLM here is the previous plan's own canonical
        # step list (added=all steps, since there is nothing to
        # compare it against but an empty plan), which is exactly the
        # structured "what did the previous attempt actually do"
        # summary the LLM needs. A second, real diff (previous vs.
        # this attempt's NEW proposal) happens below, after the LLM
        # responds, purely for duplicate-plan detection continuity —
        # see the canonicalize_plan()/plan_hash() step further down,
        # which is what actually enforces DUPLICATE_PLAN, unchanged
        # from before. If the previous attempt failed at PLAN itself
        # (state.plan == []), this is simply an empty-vs-empty diff —
        # an accurate summary of "there was no previous executable
        # plan," not a bug.
        previous_canonical = canonicalize_plan(state.plan, state.target_column)
        empty_canonical = canonicalize_plan([], state.target_column)
        previous_plan_summary = diff_plans(empty_canonical, previous_canonical).model_dump(mode="json")

    planning_context = LLMPlanningContext(
        objective=f"Predict '{state.target_column}' from the remaining columns (binary/multiclass classification).",
        dataset_context=budgeted_context.model_dump(mode="json"),
        allowed_operations=sorted(ALLOWED_TOOL_NAMES),
        tool_schemas=TOOL_ARGUMENT_SCHEMAS,
        failure_context=failure_context,
        previous_plan_summary=previous_plan_summary,
    )

    provider_result = llm_provider.generate_plan(planning_context)

    if not provider_result.success:
        failure = FailureInfo(
            category="EVALUATION_ERROR",  # reuses the existing taxonomy; see plan_validation-adjacent nodes' precedent (compare_node) for reusing this category for non-training provider-style failures rather than inventing a new one
            message=f"LLM planner failed: {provider_result.error.message}",
            evidence={
                "provider_error_code": provider_result.error.code,
                "raw_response_excerpt": provider_result.error.raw_response_excerpt,
                # Preservation state survives a parse/transport failure —
                # see _carried_forward_preserved_steps(). Absent (key
                # omitted entirely) when there is nothing to carry, so
                # this branch's evidence is byte-identical to before for
                # every first-attempt/no-classification case.
                **_carried_forward_preserved_steps(state),
            },
            node="plan",
            attempt=state.retry_count,
            retryable=True,
            human_intervention_required=False,
        )
        return {"status": "failed", "error": failure.message, "failure": failure}

    validation_result = validate_proposed_plan(provider_result.plan.steps, state.target_column)

    if not validation_result.valid:
        # Genuine finding (observed live: run_dfcbae97, real qwen3:4b):
        # a REJECTED (invalid) proposal was never given any executable
        # identity — canonicalize_plan()/plan_hash()/plan_history only
        # ever ran on a plan that had ALREADY passed validation (see the
        # duplicate-plan block below this one). That meant an LLM could
        # propose the EXACT SAME invalid tool call (e.g.
        # drop_column with arguments={}) on every REPLAN attempt, and
        # nothing deterministic ever detected or stopped it — each
        # attempt just burned a full LLM call + retry_count increment
        # repeating a validation failure PIPER already had complete
        # evidence for. validate_proposed_plan() itself is completely
        # unchanged and remains the sole authority on validity; this
        # only adds detection of an already-rejected proposal recurring,
        # exactly mirroring the existing post-validation duplicate check
        # immediately below — same mechanism, one stage earlier.
        #
        # PlanStep accepts any tool_name/arguments structurally (it
        # never itself validates them), so a REJECTED proposal can be
        # canonicalized the same way an accepted one is, purely for
        # identity/hashing purposes — this never lets invalid content
        # anywhere near CLEAN/FEATURE_ENGINEER/execution.
        rejected_plan = [
            PlanStep(
                step_id=f"step_{i + 1:02d}",
                action=proposed.action,
                tool_name=proposed.tool_name,
                arguments=proposed.arguments,
                reasoning=proposed.reasoning,
            )
            for i, proposed in enumerate(provider_result.plan.steps)
        ]
        rejected_hash = canonicalize_plan(rejected_plan, state.target_column).plan_hash()

        # Full rejected tool_name/arguments alongside the violations —
        # not just the field name and reason — so REPLAN has the exact
        # invalid value in front of it, not just a description of the
        # rule it broke.
        evidence = {
            "violations": [v.model_dump(mode="json") for v in validation_result.violations],
            "rejected_steps": [
                {"step_index": i, "tool_name": s.tool_name, "arguments": s.arguments}
                for i, s in enumerate(provider_result.plan.steps)
            ],
        }

        if rejected_hash in state.plan_history:
            failure = FailureInfo(
                category="DUPLICATE_PLAN",
                message=(
                    "The newly LLM-proposed plan is executably identical to a plan "
                    "already REJECTED as invalid earlier this run (same tool calls and "
                    "arguments, ignoring rationale) — retrying it again would produce "
                    "the same deterministic validation failure, not a different outcome."
                ),
                evidence={**evidence, "duplicate_rejected_plan_hash": rejected_hash},
                node="plan",
                attempt=state.retry_count,
                retryable=False,
                human_intervention_required=True,
            )
            return {"status": "failed", "error": failure.message, "failure": failure}

        failure = FailureInfo(
            category="EVALUATION_ERROR",
            message=(
                f"LLM-proposed plan failed deterministic tool/argument validation "
                f"({len(validation_result.violations)} violation(s))."
            ),
            evidence=evidence,
            node="plan",
            attempt=state.retry_count,
            retryable=True,
            human_intervention_required=False,
        )
        return {
            "status": "failed",
            "error": failure.message,
            "failure": failure,
            "plan_history": state.plan_history + [rejected_hash],
        }

    # Only NOW, after the proposal has survived provider-level success
    # AND deterministic tool/argument validation, is it turned into
    # real PlanStep objects — the type every downstream node
    # (CLEAN/FEATURE_ENGINEER/execution) actually trusts and executes
    # against. An LLM's ProposedPlanStep never reaches execution
    # directly.
    plan: list[PlanStep] = [
        PlanStep(
            step_id=f"step_{i + 1:02d}",
            action=proposed.action,
            tool_name=proposed.tool_name,
            arguments=proposed.arguments,
            reasoning=proposed.reasoning,
        )
        for i, proposed in enumerate(provider_result.plan.steps)
    ]

    # --- Duplicate-plan prevention (unchanged from before Phase 5) -----
    # Canonicalize what was JUST built (pending status — nothing has
    # executed yet) and compare its hash against every plan already
    # attempted this run. Still a deterministic, hash-based check —
    # never a judgment call, never something the LLM decides. This is
    # the ONE place plan identity is ever determined, and it is
    # completely unaffected by which planner (dummy heuristic,
    # formerly; real LLM, now) produced the plan's content — a real
    # LLM CAN produce a genuinely different plan on REPLAN, closing the
    # honest limitation the old dummy heuristic had.
    candidate = canonicalize_plan(plan, state.target_column)
    candidate_hash = candidate.plan_hash()

    if candidate_hash in state.plan_history:
        failure = FailureInfo(
            category="DUPLICATE_PLAN",
            message=(
                "The newly LLM-proposed plan is executably identical to a "
                "previously attempted plan (same tool calls and arguments, "
                "ignoring rationale)."
            ),
            evidence={"duplicate_plan_hash": candidate_hash, "plan_history": state.plan_history},
            node="plan",
            attempt=state.retry_count,
            retryable=False,
            human_intervention_required=True,
        )
        return {
            "status": "failed",
            "plan": plan,
            "plan_history": state.plan_history + [candidate_hash],
            "failure": failure,
            "error": failure.message,
        }

    # --- Plan adequacy (deterministic, read-only) ----------------------
    # Structural validity (validate_proposed_plan, above) proves every
    # step is well-formed. It does NOT prove the plan is SUFFICIENT for
    # what this dataset deterministically requires — a real gap observed
    # in benchmark data, where a structurally-valid single-step plan left
    # a 19.87%-missing column entirely unaddressed.
    #
    # Deliberately placed AFTER the duplicate check above, so the
    # existing mechanism keeps its authority and ordering:
    #   attempt N   : plan is novel -> duplicate check passes -> adequacy
    #                 fails -> retryable REPLAN, hash RECORDED below
    #   attempt N+1 : same plan -> the duplicate check above fires FIRST
    #                 -> DUPLICATE_PLAN (terminal), exactly as it already
    #                 does for any other repeated plan
    # That is why the failure return below still appends candidate_hash to
    # plan_history: without it, an inadequate plan could be re-proposed
    # forever. No new retry counter, no new routing, no second budget —
    # _route_after_plan already handles a retryable PLAN-node failure.
    #
    # budgeted_context is the SAME evidence the LLM was shown, so the
    # evaluator and the planner can never disagree about the dataset.
    adequacy = evaluate_plan_adequacy(budgeted_context, provider_result.plan.steps, state.target_column)

    if adequacy.material_failure:
        # Classify plan steps so REPLAN can distinguish valid operations
        # (worth preserving) from implicated ones (must be revised).
        # Pure read-only call — no plan mutation.
        step_classification = classify_plan_steps(
            adequacy.findings, provider_result.plan.steps
        )
        failure = FailureInfo(
            category="PLAN_ADEQUACY",
            message=adequacy.summary,
            evidence={
                "adequacy_status": adequacy.status,
                "findings": [f.model_dump(mode="json") for f in adequacy.material_findings],
                "advisory_findings": [
                    f.model_dump(mode="json")
                    for f in adequacy.findings
                    if f.severity == "advisory" and f.status == "NOT_ADDRESSED"
                ],
                "proposed_steps": [
                    {"tool_name": s.tool_name, "arguments": s.arguments}
                    for s in provider_result.plan.steps
                ],
                "valid_steps": step_classification["valid_steps"],
                "implicated_steps": step_classification["implicated_steps"],
            },
            node="plan",
            attempt=state.retry_count,
            retryable=True,
            human_intervention_required=False,
        )
        return {
            "status": "failed",
            "error": failure.message,
            "failure": failure,
            "plan_history": state.plan_history + [candidate_hash],
        }

    return {
        "status": "running",
        "plan": plan,
        "plan_history": state.plan_history + [candidate_hash],
    }


def feature_engineer_node(state: AgentState, store: DatasetStore) -> dict:
    """
    Executes the encode_categorical_features()/scale_features() steps
    from the plan. These tools VALIDATE the request and return a
    preview — they do NOT fit anything (locked design, see
    schemas/feature_engineering.py). The actual fit happens later,
    inside train_model(), on the train split only.
    """
    feature_log: list = []
    new_trace: list = []
    updated_plan: list = []

    for step in state.plan:
        if step.tool_name == "encode_categorical_features":
            result = encode_categorical_features(state.dataset_id, step.arguments["columns"], store)
            new_trace.append(
                _trace("feature_engineer", step.tool_name, step.arguments, result.success, {"message": result.message}, 0.0)
            )
            if result.success:
                feature_log.append(
                    OperationRecord(
                        operation_id=_new_id("op"), tool_name=step.tool_name,
                        arguments=step.arguments, result_summary=result.message,
                        reason=step.reasoning, timestamp=_now(),
                    )
                )
                updated_plan.append(step.model_copy(update={"status": "completed"}))
            else:
                updated_plan.append(step.model_copy(update={"status": "failed"}))

        elif step.tool_name == "scale_features":
            result = scale_features(state.dataset_id, step.arguments["columns"], store)
            new_trace.append(
                _trace("feature_engineer", step.tool_name, step.arguments, result.success, {"message": result.message}, 0.0)
            )
            if result.success:
                feature_log.append(
                    OperationRecord(
                        operation_id=_new_id("op"), tool_name=step.tool_name,
                        arguments=step.arguments, result_summary=result.message,
                        reason=step.reasoning, timestamp=_now(),
                    )
                )
                updated_plan.append(step.model_copy(update={"status": "completed"}))
            else:
                updated_plan.append(step.model_copy(update={"status": "failed"}))
        else:
            # Cleaning-group steps already executed by clean_node —
            # leave their status untouched, just pass them through.
            updated_plan.append(step)

    completed_fe_steps = [
        s for s in updated_plan
        if s.tool_name in ("encode_categorical_features", "scale_features") and s.status == "completed"
    ]
    if not completed_fe_steps:
        return {
            "status": "failed",
            "error": "feature_engineer_node produced an empty feature set — no encode/scale steps succeeded.",
            "plan": updated_plan,
            "tool_trace": state.tool_trace + new_trace,
        }

    return {
        "status": "running",
        "plan": updated_plan,
        "feature_log": state.feature_log + feature_log,
        "tool_trace": state.tool_trace + new_trace,
    }


def _upstream_already_failed(state: AgentState) -> bool:
    """
    Batch 7 (found by real Dockerized end-to-end verification): TRAIN ->
    EVALUATE -> COMPARE -> BASELINE -> VALIDATE are unconditional edges
    (see graph.py) — there is deliberately no per-node routing check
    between them, because the graph's ONE guardrail decision point is
    VALIDATE. That is correct for routing, but it meant a failure at
    TRAIN cascaded through EVALUATE/COMPARE/BASELINE, each of which
    OVERWROTE state.failure with its own "reached with no X" message —
    so the terminal result reported the last symptom in the chain
    ("baseline_node reached with no state.comparison") instead of the
    real root cause ("train_model failed: empty feature set").

    This helper lets each downstream node pass an already-structured
    upstream failure through UNCHANGED rather than overwriting it —
    exactly the pattern plan_node_v2 already uses for
    EXECUTION_BUDGET_EXCEEDED. Routing is completely unaffected: state
    still arrives at VALIDATE with status="failed" and validation=None,
    so _route_after_validate still makes the same REPLAN/REPORT
    decision it always did.

    Keys off status=="failed" (not merely failure is not None) so a
    STALE failure from a previous, superseded REPLAN attempt can never
    trigger it: PLAN_ENTRY sets status="replanning" and plan_node_v2
    sets it back to "running" at the start of every new attempt, so
    status is only ever "failed" here because a node in THIS attempt
    just failed.
    """
    return state.status == "failed" and state.failure is not None


def _feature_intent_from_plan(state: AgentState) -> FeatureEngineeringIntent:
    """
    Reconstructs the FeatureEngineeringIntent from completed plan
    steps — deliberately recomputed here rather than stored as a new
    AgentState field, keeping plan[] as the single source of truth for
    intent (per the locked three-log design: plan[] is what was
    intended/executed, not a place to smuggle extra derived state).
    """
    categorical_columns: list = []
    numeric_columns: list = []
    for step in state.plan:
        if step.status != "completed":
            continue
        if step.tool_name == "encode_categorical_features":
            categorical_columns.extend(step.arguments["columns"])
        elif step.tool_name == "scale_features":
            numeric_columns.extend(step.arguments["columns"])
    return FeatureEngineeringIntent(
        categorical_columns=categorical_columns,
        numeric_columns_to_scale=numeric_columns,
    )


def split_node(state: AgentState, dataset_store: DatasetStore, split_store: SplitStore) -> dict:
    """Calls the real split_dataset(). Sets state.split_id on success."""
    result = split_dataset(state.dataset_id, state.target_column, TEST_SIZE, dataset_store, split_store)
    trace = _trace(
        "splitter", "split_dataset",
        {"dataset_id": state.dataset_id, "target": state.target_column, "test_size": TEST_SIZE},
        result.success, {"message": result.message}, 0.0,
    )

    if not result.success:
        return {
            "status": "failed",
            "error": f"split_dataset failed: {result.error.message}",
            "tool_trace": state.tool_trace + [trace],
        }

    return {
        "status": "running",
        "split_id": result.data.split_id,
        "tool_trace": state.tool_trace + [trace],
    }


def reproducibility_node(state: AgentState, split_store: SplitStore) -> dict:
    """
    Populates state.reproducibility. Placed immediately after SPLIT
    (not earlier) because that is the first point in the graph where
    everything the metadata needs to describe is concretely fixed for
    this run: the dataset content that will actually be used for
    training (the split's train+test rows — not the pre-clean raw
    dataset, since cleaning/feature-engineering already ran by this
    point and the split is what train_model()/evaluate_model() will
    actually touch), and the split's random_state (already applied).
    model_random_state is recorded from the same fixed module constant
    train_node_v2 is about to use — TRAIN hasn't run yet at this point,
    but the value is a fixed constant, not something TRAIN computes at
    runtime, so recording it here (rather than adding a second node
    after TRAIN) keeps this a single, minimal integration point.

    The dataset fingerprint is computed over the concatenation of the
    split's train and test DataFrames (train followed by test, the
    split's own given order) — this is the actual dataset content this
    run's models are trained/evaluated against, which is the more
    meaningful reproducibility signal than fingerprinting the original
    unsplit dataset (which cleaning has already mutated a copy of by
    this point anyway).

    The pipeline fingerprint reuses the EXISTING plan_canonical.py
    machinery (canonicalize_plan) rather than a second competing
    canonicalization system — state.plan at this point already
    contains the completed cleaning/feature-engineering steps for this
    attempt, exactly what canonicalize_plan() expects.
    """
    if state.split_id is None:
        return {"status": "failed", "error": "reproducibility_node reached with no split_id in state."}

    try:
        train_df, test_df = split_store.get(state.split_id)
    except SplitNotFoundError:
        return {
            "status": "failed",
            "error": f"reproducibility_node: split '{state.split_id}' does not exist.",
        }

    combined = pd.concat([train_df, test_df], axis=0)
    fingerprint = dataset_fingerprint(combined)

    pipeline_fp = None
    try:
        canonical_plan = canonicalize_plan(
            state.plan, state.target_column, model_candidates=_TRAIN_CANDIDATES
        )
        pipeline_fp = canonical_plan.plan_hash()
    except Exception:
        # Pipeline fingerprint is explicitly optional (Phase 4/5 of the
        # reproducibility spec) — if canonicalization can't be built
        # cleanly from the current plan state for any reason, this
        # remains None rather than failing the whole run over an
        # optional, non-authoritative field.
        pipeline_fp = None

    metadata = ReproducibilityMetadata(
        environment=capture_environment_metadata(),
        dataset_fingerprint=fingerprint,
        split_random_state=SPLIT_RANDOM_STATE,
        model_random_state=MODEL_RANDOM_STATE,
        pipeline_fingerprint=pipeline_fp,
    )

    return {
        "status": "running",
        "reproducibility": metadata,
    }


def train_node_v2(
    state: AgentState,
    split_store: SplitStore,
    model_store: InMemoryModelStore,
) -> dict:
    """
    Calls the real train_model() once per candidate in _TRAIN_CANDIDATES
    (random_forest + logistic_regression — both already-supported V1
    algorithms; no new model types introduced). This replaces the
    former single fixed-default training call — M3's real LLM will
    eventually choose which candidates/hyperparameters to try within
    the locked allowlists; this fixed candidate set exists only so the
    graph can be exercised end-to-end, now exercising the full
    multi-model comparison path, before M3 exists.

    Every candidate uses the same split_id/target_column/feature_intent
    (same preprocessing, same random_state=42 fixed inside train_model())
    so comparison is apples-to-apples. If ANY candidate fails to train,
    the whole node fails deterministically — a partial candidate set
    could silently bias compare_models() toward whichever candidates
    happened to succeed.

    Returns model_results as a REPLACEMENT (new_model_results), not an
    accumulation onto state.model_results — matches split_node's own
    established pattern of overwriting split_id fresh each cycle rather
    than accumulating. A REPLAN produces a fresh split/reproducibility
    snapshot every attempt (see split_node/reproducibility_node), so
    model_results/evaluation_results must be scoped to THIS attempt's
    candidates only; accumulating them across REPLAN attempts was a
    genuine bug (found and fixed during M3 Phase 5 real-graph testing,
    where a genuinely context-driven planner — unlike the old static
    dummy heuristic — could actually reach TRAIN more than once per
    run) that caused evaluate_node_v2/compare_node to redundantly
    re-evaluate and compare stale models from a PREVIOUS, already-
    failed attempt alongside the current attempt's candidates.
    """
    if state.split_id is None:
        failure = FailureInfo(
            category="TRAINING_ERROR",
            message="train_node_v2 reached with no split_id in state.",
            evidence={"split_id": None},
            node="train",
            attempt=state.retry_count,
            retryable=True,
        )
        return {"status": "failed", "error": failure.message, "failure": failure}

    feature_intent = _feature_intent_from_plan(state)

    new_model_results = []
    traces = []

    for algorithm, params in _TRAIN_CANDIDATES:
        result = train_model(
            state.split_id,
            state.target_column,
            algorithm,
            params,
            feature_intent,
            split_store,
            model_store,
        )
        traces.append(
            _trace(
                "trainer", "train_model",
                {"split_id": state.split_id, "algorithm": algorithm},
                result.success, {"message": result.message}, 0.0,
            )
        )

        if not result.success:
            # Batch 7: structured FailureInfo (was a bare `error` string)
            # so the REAL root cause survives the TRAIN -> EVALUATE ->
            # COMPARE -> BASELINE cascade — see _upstream_already_failed().
            # retryable=True because a different plan genuinely can fix
            # this (e.g. an empty feature set caused by a plan that
            # proposed no encode/scale steps), which is exactly the case
            # observed live during Batch 7's Dockerized verification.
            failure = FailureInfo(
                category="TRAINING_ERROR",
                message=f"train_model failed for candidate '{algorithm}': {result.error.message}",
                evidence={
                    "algorithm": algorithm,
                    "error_code": result.error.code,
                    "categorical_columns": feature_intent.categorical_columns,
                    "numeric_columns_to_scale": feature_intent.numeric_columns_to_scale,
                },
                node="train",
                attempt=state.retry_count,
                retryable=True,
            )
            return {
                "status": "failed",
                "error": failure.message,
                "failure": failure,
                "tool_trace": state.tool_trace + traces,
            }

        new_model_results.append(result.data)

    return {
        "status": "running",
        "model_results": new_model_results,
        "tool_trace": state.tool_trace + traces,
    }


def evaluate_node_v2(state: AgentState, split_store: SplitStore, model_store: InMemoryModelStore) -> dict:
    """
    Calls the real evaluate_model() against EVERY model trained by
    train_node_v2 in the current run — not just the most recently
    trained one. This is what makes the downstream COMPARE stage
    meaningful: compare_models() needs an evaluation for each
    candidate, not just the last.

    If ANY candidate fails to evaluate, the whole node fails
    deterministically, mirroring train_node_v2's all-or-nothing
    policy — a partial evaluation set would silently bias
    compare_models() toward whichever candidates happened to succeed.

    Returns evaluation_results as a REPLACEMENT (new_evaluation_results),
    not an accumulation — see train_node_v2's docstring for why (this
    was a genuine bug, fixed together with train_node_v2's matching fix).
    """
    if _upstream_already_failed(state):
        return {}

    if not state.model_results:
        failure = FailureInfo(
            category="EVALUATION_ERROR",
            message="evaluate_node_v2 reached with no trained model.",
            evidence={"model_results_count": 0},
            node="evaluate",
            attempt=state.retry_count,
            retryable=True,
        )
        return {"status": "failed", "error": failure.message, "failure": failure}

    new_evaluation_results = []
    traces = []

    for training_result in state.model_results:
        model_id = training_result.model_id
        result = evaluate_model(model_id, split_store, model_store)
        traces.append(
            _trace(
                "evaluator", "evaluate_model", {"model_id": model_id},
                result.success, {"message": result.message}, 0.0,
            )
        )

        if not result.success:
            return {
                "status": "failed",
                "error": f"evaluate_model failed for model '{model_id}': {result.error.message}",
                "tool_trace": state.tool_trace + traces,
            }

        new_evaluation_results.append(result.data)

    return {
        "status": "running",
        "evaluation_results": new_evaluation_results,
        "tool_trace": state.tool_trace + traces,
    }


def compare_node(state: AgentState, split_store: SplitStore, model_store: InMemoryModelStore) -> dict:
    """
    Calls the real compare_models() against every model trained for
    the current run, and stores the result in state.comparison — the
    single authoritative source baseline_node (and anything
    downstream) uses to determine the selected model. Selection
    itself remains entirely inside compare_models() (F1-max, fixed,
    never LLM-choosable) — this node is a thin orchestration wrapper,
    consistent with the rest of real_nodes.py.

    Fails deterministically (TRAINING_ERROR-adjacent — reuses
    EVALUATION_ERROR, since compare_models() operates on evaluation
    results, not a new failure category) if there are no trained
    models to compare, or if compare_models() itself reports failure
    (e.g. a model_id that no longer resolves in ModelStore).
    """
    if _upstream_already_failed(state):
        return {}

    if not state.model_results:
        return {"status": "failed", "error": "compare_node reached with no trained models to compare."}

    model_ids = [training_result.model_id for training_result in state.model_results]
    result = compare_models(model_ids, split_store, model_store)
    trace = _trace(
        "comparator", "compare_models", {"model_ids": model_ids},
        result.success, {"message": result.message}, 0.0,
    )

    if not result.success:
        failure = FailureInfo(
            category="EVALUATION_ERROR",
            message=f"compare_models failed: {result.error.message}",
            evidence={"model_ids": model_ids, "error_code": result.error.code},
            node="compare",
            attempt=state.retry_count,
            retryable=True,
        )
        return {
            "status": "failed",
            "error": failure.message,
            "failure": failure,
            "tool_trace": state.tool_trace + [trace],
        }

    return {
        "status": "running",
        "comparison": result.data,
        "tool_trace": state.tool_trace + [trace],
    }


def baseline_node(state: AgentState, split_store: SplitStore, model_store: InMemoryModelStore) -> dict:
    """
    Calls the real compute_baseline() (section 8): fits a majority-
    class DummyClassifier on the SAME split the real model used, and
    compares the real model's F1 against it under the locked
    BASELINE_POLICY (delta >= 0.05 required to pass).

    Sits between COMPARE and VALIDATE — validate_node_v2 reads
    state.baseline and folds its gate_passed result into the overall
    validation decision, per the locked policy that a model
    indistinguishable from a trivial baseline must never PASS.

    The authoritative selected model is ALWAYS
    state.comparison.recommended_model_id — this node never falls
    back to model_results[-1]/evaluation_results[-1] in the
    production path. If comparison is missing, or its
    recommended_model_id doesn't correspond to an actual evaluation
    result in state.evaluation_results, that is treated as a genuine
    integration bug and reported as a structured deterministic
    failure — not silently papered over by guessing "the last model."
    """
    if _upstream_already_failed(state):
        return {}

    if state.comparison is None:
        failure = FailureInfo(
            category="EVALUATION_ERROR",
            message="baseline_node reached with no state.comparison — COMPARE stage did not run or did not populate a result.",
            evidence={
                "model_results_count": len(state.model_results),
                "evaluation_results_count": len(state.evaluation_results),
            },
            node="baseline",
            attempt=state.retry_count,
            retryable=False,
            human_intervention_required=True,
        )
        return {"status": "failed", "error": failure.message, "failure": failure}

    selected_model_id = state.comparison.recommended_model_id

    selected_evaluation = next(
        (e for e in state.evaluation_results if e.model_id == selected_model_id), None
    )
    if selected_evaluation is None:
        failure = FailureInfo(
            category="EVALUATION_ERROR",
            message=(
                f"comparison.recommended_model_id '{selected_model_id}' does not correspond "
                "to any evaluation result in state.evaluation_results."
            ),
            evidence={
                "recommended_model_id": selected_model_id,
                "known_evaluation_model_ids": [e.model_id for e in state.evaluation_results],
            },
            node="baseline",
            attempt=state.retry_count,
            retryable=False,
            human_intervention_required=True,
        )
        return {"status": "failed", "error": failure.message, "failure": failure}

    model_f1 = selected_evaluation.f1

    result = compute_baseline(selected_model_id, model_f1, split_store, model_store)
    trace = _trace(
        "baseline", "compute_baseline", {"model_id": selected_model_id, "model_f1": model_f1},
        result.success, {"message": result.message}, 0.0,
    )

    if not result.success:
        return {
            "status": "failed",
            "error": f"compute_baseline failed: {result.error.message}",
            "tool_trace": state.tool_trace + [trace],
        }

    return {
        "status": "running",
        "baseline": result.data,
        "tool_trace": state.tool_trace + [trace],
    }


_VALIDATION_CHECK_TO_FAILURE_CATEGORY = {
    "data_leakage": "LEAKAGE_ERROR",
    "target_imbalance": "IMBALANCE_ERROR",
    "constant_features": "FEATURE_ERROR",
    "high_cardinality": "FEATURE_ERROR",
    "suspicious_evaluation_metric": "LEAKAGE_ERROR",
    "baseline_gate": "BASELINE_GATE_FAILED",
}
"""
Maps a ValidationCheck.check name to the FailureCategory it represents
when that check is the reason validation failed (severity="error",
passed=False). Every category reachable through this map is
RECOVERABLE (see schemas/failure.py's RECOVERABLE_CATEGORIES) — a
guardrail-level violation on an otherwise-valid, already-profiled
dataset is never a terminal condition; REPLAN may address it (e.g. by
excluding the leaky feature next attempt).
"""


def validate_node_v2(state: AgentState, dataset_store: DatasetStore) -> dict:
    """
    Calls the real validate_pipeline(), which internally runs all four
    guardrails, the locked suspicious-metric rule, and — if
    state.baseline is populated — the baseline gate. The returned
    PipelineValidationResult already IS AgentState.validation's real
    type — no translation needed (see state.py's replacement note).

    On validation failure, also populates state.failure with a
    structured FailureInfo — mapping the first ERROR-severity
    violation to its FailureCategory (per
    _VALIDATION_CHECK_TO_FAILURE_CATEGORY), always RECOVERABLE at this
    stage (retryable=True), so the REPLAN routing decision in graph.py
    has real, structured evidence to route on rather than only the
    bare validation.valid boolean.
    """
    if _upstream_already_failed(state):
        return {}

    if not state.evaluation_results:
        return {"status": "failed", "error": "validate_node_v2 reached with no evaluation result."}

    latest_eval = state.evaluation_results[-1]

    result = validate_pipeline(
        state.dataset_id,
        state.target_column,
        dataset_store,
        evaluation_f1=latest_eval.f1,
        evaluation_accuracy=latest_eval.accuracy,
        baseline_comparison=state.baseline,
    )
    trace = _trace(
        "validator", "validate_pipeline",
        {"dataset_id": state.dataset_id, "target_column": state.target_column},
        result.success, {"message": result.message}, 0.0,
    )

    if not result.success:
        return {
            "status": "failed",
            "error": f"validate_pipeline failed to run: {result.error.message}",
            "tool_trace": state.tool_trace + [trace],
        }

    update = {
        "status": "running",
        "validation": result.data,
        "tool_trace": state.tool_trace + [trace],
    }

    if not result.data.valid and result.data.violations:
        from app.schemas.failure import FailureInfo

        first_violation = result.data.violations[0]
        category = _VALIDATION_CHECK_TO_FAILURE_CATEGORY.get(first_violation.check, "TRAINING_ERROR")

        update["failure"] = FailureInfo(
            category=category,
            message=first_violation.message,
            evidence={"violations": [v.model_dump() for v in result.data.violations]},
            node="validate",
            attempt=state.retry_count,
            retryable=True,  # guardrail-level violations at this stage are always recoverable
            human_intervention_required=False,
        )

    return update
