"""
build_run_summary() (Pre-6A Polish, item 3).

Pure, read-only aggregation over a terminal run's already-computed
state (an AgentState, or the equivalent _RunResultState shim
app/agent/tracing.py builds for the API layer — both expose the same
attributes this function reads) into one RunSummary. Never mutates its
input, never re-derives a value that already has a single source of
truth elsewhere (comparison, validation, cleaning_log, feature_log are
all read by reference, not recomputed) — see RunSummary's own
docstring for the full rationale.

run_id is accepted as a separate argument rather than read off `state`
because _RunResultState (the shim tracing.py's stream_with_tracing()/
run_with_tracing() actually populate RunStore.final_state with) never
carries run_id — no graph node's partial state update ever includes
that key, since it never changes after AgentState is constructed. The
API layer already has the real run_id from the URL path / RunRecord,
so it's passed in directly rather than sourced from `state`, exactly
like RunResultResponse's own construction in app/api/routers/runs.py.
"""

from __future__ import annotations

from app.schemas.run_summary import RunSummary


def build_run_summary(run_id: str, state) -> RunSummary:
    comparison = getattr(state, "comparison", None)
    validation = getattr(state, "validation", None)
    cleaning_log = list(getattr(state, "cleaning_log", []) or [])
    feature_log = list(getattr(state, "feature_log", []) or [])
    retry_count = getattr(state, "retry_count", 0)

    winning_algorithm = None
    if comparison is not None:
        for entry in comparison.models:
            if entry.model_id == comparison.recommended_model_id:
                winning_algorithm = entry.algorithm
                break

    return RunSummary(
        run_id=run_id,
        status=getattr(state, "status", "unknown"),
        retry_count=retry_count,
        replanned=retry_count > 0,
        candidate_models=list(comparison.models) if comparison is not None else [],
        winning_model_id=comparison.recommended_model_id if comparison is not None else None,
        winning_algorithm=winning_algorithm,
        selection_justification=comparison.justification if comparison is not None else None,
        operations_executed=cleaning_log + feature_log,
        guardrail_valid=validation.valid if validation is not None else None,
        guardrail_checks=list(validation.checks) if validation is not None else [],
        guardrail_violations=list(validation.violations) if validation is not None else [],
        guardrail_warnings=list(validation.warnings) if validation is not None else [],
    )
