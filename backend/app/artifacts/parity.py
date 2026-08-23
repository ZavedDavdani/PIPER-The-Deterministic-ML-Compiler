"""Holdout prediction parity after joblib round-trip. Non-negotiable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.artifacts.errors import ArtifactParityError
from app.storage.model_store import ModelArtifact
from app.storage.split_store import SplitStore


def holdout_features(artifact: ModelArtifact, split_store: SplitStore) -> pd.DataFrame:
    _, test_df = split_store.get(artifact.metadata.split_id)
    columns = list(artifact.metadata.feature_columns)
    missing = [c for c in columns if c not in test_df.columns]
    if missing:
        raise ArtifactParityError(
            "Holdout frame is missing feature columns required by the winning pipeline.",
            {"missing_columns": missing},
        )
    return test_df[columns]


def assert_joblib_parity(
    fitted_pipeline: Any,
    joblib_path: Path,
    x_holdout: pd.DataFrame,
) -> dict:
    """
    Reload pipeline.joblib and require
    np.array_equal(y_pred_memory, y_pred_artifact) on the evaluation holdout.
    """
    y_memory = np.asarray(fitted_pipeline.predict(x_holdout))
    loaded = joblib.load(joblib_path)
    if not hasattr(loaded, "predict"):
        raise ArtifactParityError(
            "Reloaded pipeline.joblib has no predict().",
            {"loaded_type": type(loaded).__name__},
        )
    y_artifact = np.asarray(loaded.predict(x_holdout))
    if y_memory.shape != y_artifact.shape:
        raise ArtifactParityError(
            "Reloaded pipeline prediction shape does not match the in-memory pipeline.",
            {
                "memory_shape": list(y_memory.shape),
                "artifact_shape": list(y_artifact.shape),
            },
        )
    if not np.array_equal(y_memory, y_artifact):
        mismatches = int(np.sum(y_memory != y_artifact))
        raise ArtifactParityError(
            "Reloaded pipeline.joblib predictions do not match the in-memory winning pipeline.",
            {
                "holdout_rows": int(y_memory.shape[0]),
                "mismatched_rows": mismatches,
            },
        )
    return {
        "parity_status": "passed",
        "holdout_rows": int(y_memory.shape[0]),
        "mismatched_rows": 0,
    }
