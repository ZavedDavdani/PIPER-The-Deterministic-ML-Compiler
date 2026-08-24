"""Independent inference over a VERIFIED joblib. Never retrains. Fail closed on parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.deployment.errors import InferenceError
from app.deployment.loader import load_verified_bundle
from app.deployment.schema import validate_inference_frame


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def assert_inference_parity(joblib_path: Path, features: pd.DataFrame, predictions: np.ndarray) -> dict:
    """Direct joblib.load().predict must match the inference-layer output."""
    try:
        direct = joblib.load(joblib_path)
        y_direct = np.asarray(direct.predict(features))
    except Exception as exc:
        raise InferenceError(
            "inference_parity_failed",
            "Direct pipeline.joblib prediction failed during the parity check.",
            {"error": str(exc)},
        ) from exc
    y_layer = np.asarray(predictions)
    if y_direct.shape != y_layer.shape or not np.array_equal(y_direct, y_layer):
        mismatches = int(np.sum(y_direct != y_layer)) if y_direct.shape == y_layer.shape else y_layer.shape[0]
        raise InferenceError(
            "inference_parity_failed",
            "Independent inference predictions do not match direct pipeline.joblib predictions.",
            {
                "mismatched_rows": mismatches,
                "row_count": int(y_layer.shape[0]),
            },
        )
    return {"parity_status": "passed", "mismatched_rows": 0, "row_count": int(y_layer.shape[0])}


def predict_unseen(
    artifact_root: Path,
    run_id: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    bundle = load_verified_bundle(artifact_root, run_id)
    features = validate_inference_frame(frame, bundle["feature_columns"])
    try:
        predictions = np.asarray(bundle["pipeline"].predict(features))
    except Exception as exc:
        raise InferenceError(
            "prediction_failed",
            "The verified pipeline could not score this input.",
            {"error": str(exc)},
        ) from exc
    parity = assert_inference_parity(bundle["joblib_path"], features, predictions)
    values = [_native(v) for v in predictions.tolist()]
    return {
        "run_id": run_id,
        "artifact_id": run_id,
        "winning_model_id": bundle["winning_model_id"],
        "algorithm": bundle["algorithm"],
        "row_count": int(len(values)),
        "predictions": values,
        "schema_status": "valid",
        "required_columns": list(bundle["feature_columns"]),
        "parity": parity,
        "data_kind": "NEW_UNSEEN_DATA",
    }
