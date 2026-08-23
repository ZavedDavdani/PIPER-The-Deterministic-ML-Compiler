"""
Whether a run may emit a deployable artifact.

Failed, incomplete, and guardrail-invalid runs are rejected. The
winning fitted pipeline and its evaluation split must still be
retrievable — we serialize that object, not a rebuilt plan.
"""

from __future__ import annotations

from typing import Any

from app.artifacts.errors import ArtifactEligibilityError
from app.storage.exceptions import ModelNotFoundError, SplitNotFoundError
from app.storage.model_store import InMemoryModelStore, ModelArtifact
from app.storage.split_store import SplitStore


def require_eligible_run(
    record: Any,
    model_store: InMemoryModelStore,
    split_store: SplitStore,
) -> tuple[Any, ModelArtifact]:
    if record is None:
        raise ArtifactEligibilityError("run_not_found", "Run does not exist.")

    status = getattr(record, "status", None)
    if status != "completed":
        raise ArtifactEligibilityError(
            "run_not_verified",
            f"Run status is '{status}', not 'completed'.",
            {"status": status},
        )

    state = getattr(record, "final_state", None)
    if state is None:
        raise ArtifactEligibilityError(
            "missing_final_state",
            "Completed run has no final_state snapshot.",
        )

    validation = getattr(state, "validation", None)
    if validation is None or getattr(validation, "valid", False) is not True:
        raise ArtifactEligibilityError(
            "guardrails_not_passed",
            "Deployable artifacts require validation.valid is True.",
            {"validation_present": validation is not None},
        )

    comparison = getattr(state, "comparison", None)
    winner_id = getattr(comparison, "recommended_model_id", None) if comparison is not None else None
    if not winner_id:
        raise ArtifactEligibilityError(
            "no_winning_model",
            "No recommended_model_id on the completed run.",
        )

    try:
        artifact = model_store.get(winner_id)
    except ModelNotFoundError as exc:
        raise ArtifactEligibilityError(
            "winning_pipeline_unavailable",
            "The fitted winning pipeline is not in ModelStore. "
            "Artifact export requires the in-process fitted object, "
            "not a reconstruction from the LLM plan.",
            {"model_id": winner_id},
        ) from exc

    split_id = artifact.metadata.split_id
    if not split_store.exists(split_id):
        raise ArtifactEligibilityError(
            "evaluation_split_unavailable",
            "The evaluation split used by the winning model is not available.",
            {"split_id": split_id, "model_id": winner_id},
        )

    try:
        split_store.get(split_id)
    except SplitNotFoundError as exc:
        raise ArtifactEligibilityError(
            "evaluation_split_unavailable",
            "The evaluation split used by the winning model is not available.",
            {"split_id": split_id, "model_id": winner_id},
        ) from exc

    pipeline = getattr(artifact, "pipeline", None)
    if pipeline is None or not hasattr(pipeline, "predict"):
        raise ArtifactEligibilityError(
            "winning_pipeline_unavailable",
            "ModelStore artifact has no fitted predict() pipeline.",
            {"model_id": winner_id},
        )

    return state, artifact
