"""
explore_alternative() — PIPER Learn: Learn-Explore (Batch 6B).

Controlled, single-variable exploration of an alternative to a model a
real PIPER run already trained. No new training logic: this function is
a thin orchestration wrapper around the SAME train_model()/
evaluate_model()/compare_models() the original run used — exactly the
"reuse existing training/model-comparison machinery" the locked spec
requires.

Locked, enforced here:
    - Exactly ONE variable may change relative to the base model:
      either the algorithm itself (must differ from the base's
      algorithm, and be one of the two already-supported V1
      algorithms), OR a single hyperparameter already inside its
      locked allowlist/bounds — never both, never anything else.
    - The base model MUST belong to the run being explored (its
      model_id must appear in that run's own model_results) — an
      unrelated model_id from a different run is rejected.
    - The SAME split_id the base model was trained on is reused
      (read from ModelStore's own metadata for that model, mirroring
      evaluate_model()'s existing "read split_id from the model's
      metadata" pattern) — no new splitting, no new randomness
      affecting comparability.
    - Never touches RunStore, never reconstructs or returns an
      AgentState update — structurally incapable of modifying the
      original run. Every new artifact (the new model, its evaluation,
      the comparison) is either a brand-new ModelStore entry (additive,
      never overwrites the base model_id) or a value returned to the
      caller, never written back into the original run's own record.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.agent.tools.evaluation import compare_models, evaluate_model
from app.agent.tools.training import _ALGORITHM_ALLOWED_PARAMS, train_model
from app.learning.explain import explain_evaluation, explain_model_selection
from app.schemas import ToolError, ToolResult
from app.schemas.exploration import ExplorationResult, ExplorationVariable
from app.schemas.training import Algorithm, FeatureEngineeringIntent
from app.storage import InMemoryModelStore, SplitStore
from app.storage.exceptions import ModelNotFoundError

_ALLOWED_ALGORITHMS = ("logistic_regression", "random_forest")


def _new_experiment_id() -> str:
    return f"exp_{uuid.uuid4().hex[:8]}"


def explore_alternative(
    run_id: str,
    run_model_ids: list[str],
    base_model_id: str,
    split_store: SplitStore,
    model_store: InMemoryModelStore,
    new_algorithm: str | None = None,
    hyperparameter_name: str | None = None,
    hyperparameter_value: float | None = None,
) -> ToolResult[ExplorationResult]:
    """
    run_model_ids: the ORIGINAL run's own state.model_results model_ids
    — the caller-supplied scoping boundary that proves base_model_id
    genuinely belongs to that run (ModelStore itself is a shared,
    global, in-memory store across every run, so nothing about a bare
    model_id string alone proves which run trained it).

    Exactly one of (new_algorithm) or (hyperparameter_name AND
    hyperparameter_value) must be provided — enforced here, before
    anything else is validated.
    """
    changing_model = new_algorithm is not None
    changing_hyperparameter = hyperparameter_name is not None or hyperparameter_value is not None

    if changing_model and changing_hyperparameter:
        return ToolResult[ExplorationResult](
            success=False, tool_name="explore_alternative",
            message="Exactly one variable may change per exploration — both a new algorithm and a hyperparameter were given.",
            error=ToolError(
                code="more_than_one_variable_changed",
                message="Provide either new_algorithm OR (hyperparameter_name and hyperparameter_value), never both.",
                details={"new_algorithm": new_algorithm, "hyperparameter_name": hyperparameter_name},
            ),
        )
    if not changing_model and not changing_hyperparameter:
        return ToolResult[ExplorationResult](
            success=False, tool_name="explore_alternative",
            message="No variable to explore was provided.",
            error=ToolError(
                code="no_variable_changed",
                message="Provide either new_algorithm OR (hyperparameter_name and hyperparameter_value).",
                details={},
            ),
        )
    if changing_hyperparameter and (hyperparameter_name is None or hyperparameter_value is None):
        return ToolResult[ExplorationResult](
            success=False, tool_name="explore_alternative",
            message="A hyperparameter exploration requires both hyperparameter_name and hyperparameter_value.",
            error=ToolError(
                code="incomplete_hyperparameter_request",
                message="hyperparameter_name and hyperparameter_value must both be provided.",
                details={"hyperparameter_name": hyperparameter_name, "hyperparameter_value": hyperparameter_value},
            ),
        )

    if base_model_id not in run_model_ids:
        return ToolResult[ExplorationResult](
            success=False, tool_name="explore_alternative",
            message=f"Model '{base_model_id}' does not belong to run '{run_id}'.",
            error=ToolError(
                code="model_not_from_this_run",
                message="base_model_id must be one of the model_ids this run actually trained.",
                details={"run_id": run_id, "base_model_id": base_model_id, "run_model_ids": run_model_ids},
            ),
        )

    try:
        base_artifact = model_store.get(base_model_id)
    except ModelNotFoundError:
        return ToolResult[ExplorationResult](
            success=False, tool_name="explore_alternative",
            message=f"Model '{base_model_id}' does not exist.",
            error=ToolError(code="model_not_found", message="base_model_id does not exist in ModelStore.", details={"base_model_id": base_model_id}),
        )

    base_metadata = base_artifact.metadata
    feature_intent = FeatureEngineeringIntent(
        categorical_columns=base_metadata.categorical_columns,
        numeric_columns_to_scale=base_metadata.numeric_columns,
    )

    if changing_model:
        if new_algorithm not in _ALLOWED_ALGORITHMS:
            return ToolResult[ExplorationResult](
                success=False, tool_name="explore_alternative",
                message=f"Unknown algorithm '{new_algorithm}'.",
                error=ToolError(
                    code="unknown_algorithm",
                    message=f"new_algorithm must be one of: {list(_ALLOWED_ALGORITHMS)}.",
                    details={"new_algorithm": new_algorithm},
                ),
            )
        if new_algorithm == base_metadata.algorithm:
            return ToolResult[ExplorationResult](
                success=False, tool_name="explore_alternative",
                message="new_algorithm must differ from the base model's algorithm to be an alternative.",
                error=ToolError(
                    code="not_an_alternative",
                    message=f"'{new_algorithm}' is already the base model's algorithm.",
                    details={"base_algorithm": base_metadata.algorithm},
                ),
            )
        new_full_params: dict = {}
        variable = ExplorationVariable(
            kind="model", name="algorithm", old_value=base_metadata.algorithm, new_value=new_algorithm,
        )
        train_algorithm: Algorithm = new_algorithm  # type: ignore[assignment]
    else:
        if hyperparameter_name not in _ALGORITHM_ALLOWED_PARAMS.get(base_metadata.algorithm, set()):
            return ToolResult[ExplorationResult](
                success=False, tool_name="explore_alternative",
                message=f"'{hyperparameter_name}' is not an already-supported hyperparameter for {base_metadata.algorithm}.",
                error=ToolError(
                    code="disallowed_hyperparameter",
                    message="hyperparameter_name must be inside the existing locked allowlist for the base model's algorithm.",
                    details={
                        "algorithm": base_metadata.algorithm,
                        "hyperparameter_name": hyperparameter_name,
                        "allowed": sorted(_ALGORITHM_ALLOWED_PARAMS.get(base_metadata.algorithm, set())),
                    },
                ),
            )
        new_full_params = {**base_metadata.parameters, hyperparameter_name: hyperparameter_value}
        old_value = base_metadata.parameters.get(hyperparameter_name, "default")
        variable = ExplorationVariable(
            kind="hyperparameter", name=hyperparameter_name,
            old_value=str(old_value), new_value=str(hyperparameter_value),
        )
        train_algorithm = base_metadata.algorithm  # type: ignore[assignment]

    training_result = train_model(
        base_metadata.split_id, base_metadata.target_column, train_algorithm,
        new_full_params, feature_intent, split_store, model_store,
    )
    if not training_result.success:
        return ToolResult[ExplorationResult](
            success=False, tool_name="explore_alternative",
            message=f"Training the exploration variant failed: {training_result.error.message}",
            error=training_result.error,
        )

    new_model_id = training_result.data.model_id

    evaluation_result = evaluate_model(new_model_id, split_store, model_store)
    if not evaluation_result.success:
        return ToolResult[ExplorationResult](
            success=False, tool_name="explore_alternative",
            message=f"Evaluating the exploration variant failed: {evaluation_result.error.message}",
            error=evaluation_result.error,
        )

    comparison_result = compare_models([base_model_id, new_model_id], split_store, model_store)
    if not comparison_result.success:
        return ToolResult[ExplorationResult](
            success=False, tool_name="explore_alternative",
            message=f"Comparing the exploration variant against the base model failed: {comparison_result.error.message}",
            error=comparison_result.error,
        )

    result = ExplorationResult(
        experiment_id=_new_experiment_id(),
        run_id=run_id,
        base_model_id=base_model_id,
        split_id=base_metadata.split_id,
        variable_changed=variable,
        training=training_result.data,
        evaluation=evaluation_result.data,
        comparison_vs_base=comparison_result.data,
        evaluation_explanation=explain_evaluation(evaluation_result.data, baseline=None),
        comparison_explanation=explain_model_selection(comparison_result.data),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    return ToolResult[ExplorationResult](
        success=True, tool_name="explore_alternative",
        message=(
            f"Explored {variable.kind} change ({variable.name}: {variable.old_value} -> "
            f"{variable.new_value}); new model F1={evaluation_result.data.f1}, "
            f"comparison recommends '{comparison_result.data.recommended_model_id}'."
        ),
        data=result,
    )
