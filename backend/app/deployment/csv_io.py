"""CSV helpers for Test Flight. Never mutate the uploaded bytes/file."""

from __future__ import annotations

import io

import pandas as pd

from app.deployment.errors import InferenceError

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def parse_unseen_csv(filename: str | None, raw: bytes) -> pd.DataFrame:
    name = (filename or "").lower()
    if not name.endswith(".csv"):
        raise InferenceError(
            "unsupported_file_type",
            "Test Flight accepts CSV files only.",
            {"filename": filename},
        )
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InferenceError(
            "file_too_large",
            f"CSV exceeds the maximum upload size ({MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
        )
    if not raw:
        raise InferenceError("invalid_input", "Uploaded CSV is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise InferenceError("invalid_input", "CSV could not be parsed.", {"error": str(exc)}) from exc
    return frame


def predictions_csv(original: pd.DataFrame, predictions: list) -> bytes:
    result = original.copy()
    result.insert(len(result.columns), "prediction", predictions)
    return result.to_csv(index=False).encode("utf-8")
