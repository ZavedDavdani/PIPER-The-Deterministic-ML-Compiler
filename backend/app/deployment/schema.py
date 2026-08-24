"""Validate unseen inference frames against the verified artifact schema."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.deployment.errors import InferenceError

MAX_PREDICT_ROWS = 100_000


def rows_to_frame(rows: Any) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        raise InferenceError("invalid_input", "Request must include a non-empty rows array.")
    if not all(isinstance(row, dict) for row in rows):
        raise InferenceError("invalid_input", "Each row must be a JSON object.")
    return pd.DataFrame(rows)


def validate_inference_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise InferenceError("invalid_input", "Input must be a tabular frame.")
    if frame.shape[0] == 0:
        raise InferenceError("invalid_input", "Input contains no rows.")
    if frame.shape[0] > MAX_PREDICT_ROWS:
        raise InferenceError(
            "invalid_input",
            f"Input exceeds the maximum of {MAX_PREDICT_ROWS} rows.",
            {"row_count": int(frame.shape[0])},
        )
    missing = [name for name in feature_columns if name not in frame.columns]
    if missing:
        raise InferenceError(
            "missing_features",
            "Required inference features are missing.",
            {"missing": missing, "required": list(feature_columns)},
        )
    return frame.loc[:, list(feature_columns)].copy()
