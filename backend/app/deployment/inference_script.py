"""Standalone inference.py for the optional deployment package. No PIPER imports."""

from __future__ import annotations

from typing import Sequence

_INFERENCE_PY = '''\
"""
Standalone batch inference for a PIPER-exported VERIFIED sklearn pipeline.

This file does not import PIPER, FastAPI, LangGraph, SQLite, or Ollama.
It never retrains. It only loads pipeline.joblib and calls predict().

Usage:
    python inference.py new_data.csv -o predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

PIPELINE_PATH = Path(__file__).resolve().parent / "pipeline.joblib"
REQUIRED_COLUMNS = {required_columns!r}
RUN_ID = {run_id!r}
TARGET_COLUMN = {target_column!r}
WINNING_ALGORITHM = {algorithm!r}


def load_pipeline():
    if not PIPELINE_PATH.is_file():
        raise FileNotFoundError(f"Missing {{PIPELINE_PATH}}")
    pipeline = joblib.load(PIPELINE_PATH)
    if not hasattr(pipeline, "predict"):
        raise TypeError("pipeline.joblib has no predict()")
    return pipeline


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[0] == 0:
        raise ValueError("Input CSV has no rows.")
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required inference columns: {{missing}}")
    return frame.loc[:, list(REQUIRED_COLUMNS)]


def predict_csv(csv_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    original = pd.read_csv(csv_path)
    features = validate_frame(original)
    pipeline = load_pipeline()
    preds = pipeline.predict(features)
    # Copy — never mutate the source CSV on disk.
    result = original.copy()
    result.insert(len(result.columns), "prediction", preds)
    if output_path is not None:
        result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score unseen CSV rows with a verified PIPER pipeline.")
    parser.add_argument("csv_path", type=Path, help="Unseen CSV (not training data).")
    parser.add_argument("-o", "--output", type=Path, help="Write predictions CSV here.")
    args = parser.parse_args()
    out = predict_csv(args.csv_path, args.output)
    if args.output is None:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
'''


def render_inference_py(
    *,
    run_id: str,
    target_column: str,
    algorithm: str,
    feature_columns: Sequence[str],
) -> str:
    return _INFERENCE_PY.format(
        required_columns=list(feature_columns),
        run_id=run_id,
        target_column=target_column,
        algorithm=algorithm,
    )
