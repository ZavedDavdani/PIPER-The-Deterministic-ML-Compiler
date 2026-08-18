"""
Real deterministic pipeline graph — replaces the M2 stub-based graph.

    PROFILE (real)
       |
       v
    [task_type check: failed if not binary_classification, no replan]
       |
       v
    PLAN (dummy heuristic — M3 replaces with real LLM; now also plans
          feature engineering, not just cleaning)
       |
       v
    CLEAN (real, executes cleaning-group plan steps)
       |
       v
    FEATURE_ENGINEER (real — validates encode/scale requests; does NOT
                       fit anything, per the locked no-leakage design)
       |
       v
    SPLIT (real — split_dataset(), stratified, random_state=42)
       |
       v
    REPRODUCIBILITY (real — captures environment versions, a content
                      fingerprint of the actual split dataset, and the
                      split/model random_state configuration into
                      state.reproducibility; introduces no new
                      randomness or routing decisions)
       |
       v
    TRAIN (real — train_model() called once per V1 candidate
           [random_forest, logistic_regression], each fitting its own
           ColumnTransformer + classifier as ONE Pipeline, on the
           train split only)
       |
       v
    EVALUATE (real — evaluate_model() called for EVERY trained
              candidate this run, never fits anything, proven
              behaviorally in test_evaluation.py)
       |
       v
    COMPARE (real — compare_models(), F1-max selection, fixed and
             never LLM-choosable; result stored in state.comparison)
       |
       v
    BASELINE (real — compute_baseline() against
              state.comparison.recommended_model_id, the sole
              authoritative selected model; no [-1] fallback)
       |
       v
    VALIDATE (real — validate_pipeline(), runs all four guardrails +
              the locked suspicious-metric rule)
       |
       +---- PASS  -----> REPORT (terminal, status=completed)
       |
       +---- FAIL  -----> [retry_count < max_retries?]
                              |
                         YES  |  NO
                              v    v
                          REPLAN  REPORT (terminal, status=failed,
                          (loops    caveat noted)
                          back to
                          PLAN)

REPLAN is not its own graph node — it's the routing decision to send
control back to plan_node with retry_count incremented. The graph
(not the LLM, not the dummy planner) decides PASS vs REPLAN based on
validation.valid and retry_count — this routing logic is UNCHANGED
from M2, since it was already generic against state.validation/
state.status/state.retry_count and never depended on the stub nodes'
fake data.

No fake F1 values, no fake validation, no hardcoded "successful"
training result — every metric in a run through this graph comes from
an actual fitted sklearn Pipeline evaluated against a real held-out
test split.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from app.agent.nodes.m2_nodes import clean_node, profile_node
from app.agent.nodes.real_nodes import (
    baseline_node,
    compare_node,
    evaluate_node_v2,
    feature_engineer_node,
    plan_node_v2,
    reproducibility_node,
    sanitize_node,
    split_node,
    train_node_v2,
    validate_input_node,
    validate_node_v2,
)
from app.agent.state import AgentState
from app.schemas.failure import FailureInfo
from app.storage import DatasetStore, InMemoryModelStore, SplitStore

# --- Node names (avoid magic strings scattered through routing logic) ---

VALIDATE_INPUT = "validate_input"
PROFILE = "profile"
SANITIZE = "sanitize"
PLAN_ENTRY = "plan_entry"  # bridging node: increments retry_count on REPLAN only
PLAN = "plan"
CLEAN = "clean"
FEATURE_ENGINEER = "feature_engineer"
SPLIT = "split"
REPRODUCIBILITY = "reproducibility"
TRAIN = "train"
EVALUATE = "evaluate"
COMPARE = "compare"
BASELINE = "baseline"
VALIDATE = "validate"
REPORT = "report"

MAX_EXECUTION_STEPS = 80
"""
M4: hard, PIPER-owned execution-step ceiling — deliberately independent
of max_retries and of any externally supplied LangGraph
`recursion_limit` (passed ad hoc as a `graph.invoke(..., config=...)`
argument at each call site; graph.py itself never sees or controls
it). Its job is to guarantee graph.invoke() ALWAYS returns a clean
terminal AgentState (status="failed", a real FailureInfo) instead of
ever raising an uncaught GraphRecursionError, regardless of how
max_retries is configured.

Sized well above any legitimate run: a normal default run
(max_retries=2, the constraint #6 default) uses on the order of 25-37
total node executions even in its worst case (every attempt fails
validation, no early duplicate-plan exit); a generously-configured
max_retries=5 run tops out around 70. 80 leaves real headroom above
both while still catching a genuinely misconfigured/runaway retry
budget (e.g. max_retries left at some very large value) well before it
could run indefinitely.

Enforced at the single place a graph invocation can ever loop back on
itself — PLAN_ENTRY (`_increment_retry_if_replanning`), immediately
before any REPLAN (or the very first PLAN entry) is allowed to proceed
— since that back-edge (VALIDATE -> PLAN_ENTRY) is the graph's only
cycle. See that function for the actual check.
"""


def report_node(state: AgentState) -> dict:
    """
    Terminal node. Sets final status. If validation failed and retries
    are exhausted, status becomes "failed" with the validation caveat
    preserved in `error` — per the locked slippage rule, we report
    best-so-far with a flagged caveat rather than looping forever.

    If state.status is already "failed" (e.g. from profile_node's
    task_type check), this node preserves that — it does not overwrite
    a hard failure with "completed".

    On terminal exhaustion of a recoverable failure, this node also
    flips state.failure.human_intervention_required to True (it was
    False when validate_node_v2 first populated it, since at that
    point a REPLAN was still possible) — the retry budget being spent
    is itself new information that belongs in the structured failure
    record, not just the free-text `error` caveat.

    Pre-6A Polish fix: when final validation succeeds, this node also
    clears state.failure. Without this, a run that REPLANned at least
    once before ultimately passing validation reached "completed" with
    a stale FailureInfo left over from an earlier, superseded attempt
    still attached — status/validation.valid were always correct, but
    a caller naively checking `failure is not None` on a completed run
    would be misled. Observed live in Batch 5 (see CLAUDE.md); this is
    the fix.
    """
    if state.status == "failed":
        if (
            state.failure is not None
            and state.failure.retryable
            and not state.failure.human_intervention_required
        ):
            # Batch 5 fix: a retryable PLAN-node failure (see
            # _route_after_plan) only ever reaches REPORT with
            # status == "failed" once its retry budget is genuinely
            # exhausted — mirror the VALIDATE-exhaustion treatment
            # below (which reaches this same "retries ran out"
            # moment via a different status value) so both paths
            # report human_intervention_required=True consistently.
            return {
                "failure": state.failure.model_copy(
                    update={"human_intervention_required": True, "attempt": state.retry_count}
                )
            }
        return {}  # already a fully-formed terminal failure — leave as is

    if state.validation is not None and state.validation.valid:
        # `error` is cleared for the same reason `failure` is (Pre-6A
        # Polish item 1) — it was simply missed there. tracing.py builds
        # both the REPORT node's TraceEvent message and the terminal run
        # message from `state.error`, so a run that REPLANned and then
        # SUCCEEDED was reporting the superseded attempt's failure text
        # on its final event. Observed live in the qwen3:8b demonstration
        # run (run_e13cf35f): status "completed", failure null, yet the
        # report event read "Plan is structurally valid but inadequate...".
        # Cosmetic in the API (status/validation.valid were always
        # correct) but actively misleading in the live SSE feed and the
        # frontend timeline, which is where a demo is actually watched.
        return {"status": "completed", "failure": None, "error": None}

    # Retries exhausted and validation still failing.
    caveat = (
        "Pipeline did not pass validation after "
        f"{state.retry_count} retr{'y' if state.retry_count == 1 else 'ies'}; "
        "reporting best-available result with this caveat."
    )

    update = {"status": "failed", "error": caveat}

    if state.failure is not None:
        update["failure"] = state.failure.model_copy(
            update={"human_intervention_required": True, "attempt": state.retry_count}
        )

    return update


def _route_after_validate_input(state: AgentState) -> str:
    """
    Section 9: any data-quality violation is terminal — straight to
    REPORT, never PLAN, never REPLAN. Mirrors _route_after_profile's
    existing precedent for constraint #9 (non-binary target).
    """
    if state.status == "failed":
        return REPORT
    return PROFILE


def _route_after_profile(state: AgentState) -> str:
    """
    Constraint #9: invalid/unresolved task_type routes straight to
    REPORT as a hard failure — never PLAN, never REPLAN.
    """
    if state.status == "failed" or state.task_type is None:
        return REPORT
    return PLAN


def _route_after_plan(state: AgentState) -> str:
    """
    Batch 5 fix (previously documented in CLAUDE.md as "PLAN-node
    failures never get a REPLAN chance"): a PLAN-node failure whose
    FailureInfo is genuinely retryable (an LLM provider error, or a
    proposed plan that failed validate_proposed_plan() — both set
    retryable=True in plan_node_v2) now gets the same REPLAN
    opportunity a VALIDATE-node failure already got, bounded by the
    same retry_count < max_retries check _route_after_validate already
    uses — not unconditional retries, not a new termination path.

    Still routes straight to REPORT, never PLAN_ENTRY, for:
    - a non-retryable PLAN failure (DUPLICATE_PLAN, or the sanitized-
      context-build failure — both explicitly set retryable=False), or
    - a retryable failure with no retry budget left, or
    - any status == "failed" with no FailureInfo at all (the
      defensive "profile is None" guard clause), which was never a
      retryable failure to begin with.

    Never waste execution running CLEAN/TRAIN/EVALUATE/BASELINE on a
    plan already known to be identical to one that already failed —
    that DUPLICATE_PLAN case is unaffected by this change.
    """
    if state.status == "failed":
        if (
            state.failure is not None
            and state.failure.retryable
            and state.retry_count < state.max_retries
        ):
            return PLAN  # REPLAN = route back through PLAN_ENTRY (retry_count increment)
        return REPORT
    return CLEAN


def _route_after_validate(state: AgentState) -> str:
    """
    The one deterministic decision point the locked architecture cares
    about most: the graph — not the LLM, not the dummy planner —
    decides PASS vs REPLAN based on validation.valid and retry_count.

    Batch 7 fix (real, confirmed reliability finding from end-to-end
    Docker verification against a real Ollama model): validate_node_v2
    can reach a status=="failed" early return WITHOUT ever populating
    state.validation — e.g. its own "no evaluation_results" guard
    clause, reached whenever an EARLIER node this attempt (train/
    evaluate/compare/baseline) already failed. The original version of
    this function only ever looked at state.validation, so on a run
    whose FIRST attempt failed at PLAN itself (state.validation still
    None — the pipeline never reached VALIDATE that attempt), reaching
    this situation on a LATER attempt fell through to the unconditional
    `retry_count < max_retries -> PLAN` branch, incorrectly treating
    "VALIDATE never even ran" as an ordinary "guardrail found the
    pipeline invalid" case. _increment_retry_if_replanning then
    correctly declined to increment retry_count (neither of its two
    conditions held: state.validation was still None, and the
    upstream failure carried into this node was retryable=False) —
    but the graph still routed back to PLAN via the unconditional
    PLAN_ENTRY->PLAN edge, calling the LLM again anyway. The result: a
    genuine unbudgeted retry loop, bounded only by the much larger
    MAX_EXECUTION_STEPS safety net rather than by max_retries —
    confirmed live (a real run made 4 real LLM calls while retry_count
    never advanced past 1) and reproduced deterministically with a
    FakeLLMProvider.

    Fixed by mirroring _route_after_plan's own, already-correct
    pattern: when VALIDATE didn't actually run this attempt
    (state.status == "failed" with no fresh validation), only REPLAN
    if state.failure is genuinely retryable AND budget remains —
    otherwise report a bounded terminal failure immediately, exactly
    like a non-retryable PLAN-node failure already does.
    """
    if state.validation is not None and state.validation.valid:
        return REPORT

    if state.status == "failed":
        if (
            state.failure is not None
            and state.failure.retryable
            and state.retry_count < state.max_retries
        ):
            return PLAN  # REPLAN = route back through PLAN_ENTRY (retry_count increment)
        return REPORT  # VALIDATE never ran and the carried-over failure isn't a budgeted retry

    if state.retry_count < state.max_retries:
        return PLAN  # REPLAN = route back to PLAN with retry_count incremented

    return REPORT  # retries exhausted — report best-so-far with caveat


def _increment_retry_if_replanning(state: AgentState) -> dict:
    """
    Small bridging node: increments retry_count exactly once per
    REPLAN loop, immediately before re-entering PLAN. Keeping this as
    its own step (rather than folding into plan_node) keeps plan_node
    focused purely on planning and makes the retry increment visible
    and testable on its own.

    Only increments if we're actually coming back from a genuine
    REPLAN — either a failed VALIDATE (state.validation is invalid) or
    a retryable PLAN-node failure (Batch 5 fix, see _route_after_plan)
    — not on the very first pass into PLAN (where both state.validation
    and state.failure are still None).

    Both conditions checked below are the ONLY two ways this node's
    incoming back-edge can ever be reached: _route_after_validate and
    _route_after_plan each independently verify retry_count <
    max_retries before ever routing back here, so by construction
    every arrival via the back-edge is a legitimate, budgeted retry.

    M4: also the sole enforcement point for MAX_EXECUTION_STEPS (see
    that constant's docstring) — this is the only node PLAN_ENTRY's
    incoming edge, and the graph's only cycle (VALIDATE -> PLAN_ENTRY)
    passes back through here, so checking here is sufficient to bound
    the entire graph, first entry included. On trip, this returns a
    genuine terminal FailureInfo(category="EXECUTION_BUDGET_EXCEEDED")
    instead of incrementing retry_count — plan_node_v2 recognizes this
    specific failure and passes it straight through unchanged (see its
    own guard clause), and _route_after_plan's existing
    `status == "failed" -> REPORT` routing (unchanged) takes it to a
    clean terminal state from there.
    """
    if state.steps_executed >= MAX_EXECUTION_STEPS:
        failure = FailureInfo(
            category="EXECUTION_BUDGET_EXCEEDED",
            message=(
                f"Execution step budget ({MAX_EXECUTION_STEPS} graph node "
                "executions) was reached — terminating deterministically "
                "rather than continuing an unbounded run. This is a hard, "
                "PIPER-owned safety backstop, independent of max_retries "
                "and of any externally supplied recursion_limit."
            ),
            evidence={
                "steps_executed": state.steps_executed,
                "max_execution_steps": MAX_EXECUTION_STEPS,
                "retry_count": state.retry_count,
                "max_retries": state.max_retries,
            },
            node=PLAN_ENTRY,
            attempt=state.retry_count,
            retryable=False,
            human_intervention_required=True,
        )
        return {"status": "failed", "error": failure.message, "failure": failure}

    came_from_failed_validation = state.validation is not None and not state.validation.valid
    came_from_retryable_plan_failure = (
        state.status == "failed" and state.failure is not None and state.failure.retryable
    )
    if came_from_failed_validation or came_from_retryable_plan_failure:
        return {"retry_count": state.retry_count + 1, "status": "replanning"}
    return {}


def _counted(node_name: str, fn):
    """
    M4: wraps a node callable so every real execution increments
    state.steps_executed by exactly one — the PIPER-owned execution
    budget mechanism (see MAX_EXECUTION_STEPS). Purely additive: never
    changes what the wrapped node itself returns or decides, only adds
    the steps_executed key to its result. node_name is accepted for
    parity with the rest of this module's naming (and possible future
    tracing use) even though it isn't currently read.
    """

    def wrapped(state: AgentState, **kwargs) -> dict:
        result = fn(state, **kwargs)
        update = dict(result) if result else {}
        update["steps_executed"] = state.steps_executed + 1
        return update

    return wrapped


def build_graph(
    dataset_store: DatasetStore,
    split_store: SplitStore,
    model_store: InMemoryModelStore,
    llm_provider,
):
    """
    Constructs and compiles the real pipeline graph. Stores (and, as of
    M3 Phase 5, the LLM provider) are injected via functools.partial
    into every node that needs them, rather than being part of
    AgentState — per the locked principle that AgentState holds
    references, not infrastructure.

    llm_provider must satisfy the LLMProvider protocol
    (app.llm.provider) — FakeLLMProvider for tests, OllamaProvider for
    real local execution. graph.py has no import of, or coupling to,
    Ollama specifically; it only depends on the protocol.
    """
    graph = StateGraph(AgentState)

    graph.add_node(VALIDATE_INPUT, _counted(VALIDATE_INPUT, partial(validate_input_node, store=dataset_store)))
    graph.add_node(PROFILE, _counted(PROFILE, partial(profile_node, store=dataset_store)))
    graph.add_node(SANITIZE, _counted(SANITIZE, partial(sanitize_node, store=dataset_store)))
    graph.add_node(PLAN_ENTRY, _counted(PLAN_ENTRY, _increment_retry_if_replanning))
    graph.add_node(PLAN, _counted(PLAN, partial(plan_node_v2, store=dataset_store, llm_provider=llm_provider)))
    graph.add_node(CLEAN, _counted(CLEAN, partial(clean_node, store=dataset_store)))
    graph.add_node(FEATURE_ENGINEER, _counted(FEATURE_ENGINEER, partial(feature_engineer_node, store=dataset_store)))
    graph.add_node(SPLIT, _counted(SPLIT, partial(split_node, dataset_store=dataset_store, split_store=split_store)))
    graph.add_node(REPRODUCIBILITY, _counted(REPRODUCIBILITY, partial(reproducibility_node, split_store=split_store)))
    graph.add_node(TRAIN, _counted(TRAIN, partial(train_node_v2, split_store=split_store, model_store=model_store)))
    graph.add_node(EVALUATE, _counted(EVALUATE, partial(evaluate_node_v2, split_store=split_store, model_store=model_store)))
    graph.add_node(COMPARE, _counted(COMPARE, partial(compare_node, split_store=split_store, model_store=model_store)))
    graph.add_node(BASELINE, _counted(BASELINE, partial(baseline_node, split_store=split_store, model_store=model_store)))
    graph.add_node(VALIDATE, _counted(VALIDATE, partial(validate_node_v2, dataset_store=dataset_store)))
    graph.add_node(REPORT, _counted(REPORT, report_node))

    graph.set_entry_point(VALIDATE_INPUT)

    graph.add_conditional_edges(
        VALIDATE_INPUT,
        _route_after_validate_input,
        path_map={PROFILE: PROFILE, REPORT: REPORT},
    )

    graph.add_conditional_edges(
        PROFILE,
        _route_after_profile,
        path_map={PLAN: SANITIZE, REPORT: REPORT},
    )

    graph.add_edge(SANITIZE, PLAN_ENTRY)

    graph.add_edge(PLAN_ENTRY, PLAN)
    graph.add_conditional_edges(
        PLAN,
        _route_after_plan,
        path_map={CLEAN: CLEAN, REPORT: REPORT, PLAN: PLAN_ENTRY},
    )
    graph.add_edge(CLEAN, FEATURE_ENGINEER)
    graph.add_edge(FEATURE_ENGINEER, SPLIT)
    graph.add_edge(SPLIT, REPRODUCIBILITY)
    graph.add_edge(REPRODUCIBILITY, TRAIN)
    graph.add_edge(TRAIN, EVALUATE)
    graph.add_edge(EVALUATE, COMPARE)
    graph.add_edge(COMPARE, BASELINE)
    graph.add_edge(BASELINE, VALIDATE)

    graph.add_conditional_edges(
        VALIDATE,
        _route_after_validate,
        path_map={PLAN: PLAN_ENTRY, REPORT: REPORT},
    )

    graph.add_edge(REPORT, END)

    return graph.compile()
