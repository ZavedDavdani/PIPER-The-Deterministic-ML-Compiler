"""
ModelStore — real implementation.

Replaces the interface-only stub from run_and_model_store.py's
ModelStore ABC (kept there; this module provides the concrete
InMemoryModelStore). Holds the complete fitted artifact: a single
sklearn.Pipeline containing BOTH preprocessing (ColumnTransformer:
OneHotEncoder + StandardScaler) AND the classifier, fit once, together,
on the train split only.

This is deliberate and load-bearing: keeping preprocessing and the
classifier as one fitted Pipeline object (not separate encoder/scaler/
model pieces) is what makes it structurally impossible for
evaluate_model() to accidentally re-fit anything on test data later —
there's only one object, and it's already fit.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.storage.exceptions import ModelNotFoundError


class ModelArtifactMetadata(BaseModel):
    """
    Metadata stored alongside the fitted Pipeline. This is NOT the
    same as AgentState's ModelResult (state.py) — that's a lighter
    summary carried through the graph; this is the fuller record kept
    with the actual fitted object, since ModelStore is the source of
    truth for "can this model be reproduced and explained."
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    algorithm: str
    parameters: dict = Field(default_factory=dict)
    split_id: str
    target_column: str
    feature_columns: list[str]
    categorical_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    random_state: int
    training_rows: int


class ModelArtifact(BaseModel):
    """Metadata plus a reference to where the actual fitted object lives."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    metadata: ModelArtifactMetadata
    pipeline: Any = Field(..., description="The fitted sklearn.Pipeline object itself.")


class InMemoryModelStore:
    """
    M1–M2 implementation: a plain dict keyed by model_id. Each entry is
    a ModelArtifact (metadata + the fitted Pipeline). No copy-on-write
    isolation is applied to the Pipeline itself (unlike DatasetStore) —
    a fitted sklearn Pipeline is treated as immutable once stored;
    nothing in this codebase mutates a Pipeline after fit(), so the
    copy-isolation concern that mattered for mutable DataFrames doesn't
    apply the same way here.
    """

    def __init__(self) -> None:
        self._data: dict[str, ModelArtifact] = {}

    def save(self, model_id: str, artifact: ModelArtifact) -> None:
        self._data[model_id] = artifact

    def get(self, model_id: str) -> ModelArtifact:
        if model_id not in self._data:
            raise ModelNotFoundError(model_id)
        return self._data[model_id]

    def delete(self, model_id: str) -> None:
        if model_id not in self._data:
            raise ModelNotFoundError(model_id)
        del self._data[model_id]

    def exists(self, model_id: str) -> bool:
        return model_id in self._data
