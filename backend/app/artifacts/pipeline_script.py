"""Deterministic pipeline.py — no LLM, no PIPER imports."""

from __future__ import annotations

from typing import Sequence

_PIPELINE_PY_TEMPLATE = '''\
"""
Standalone inference for PIPER-exported sklearn pipeline.

This file is generated deterministically from a verified run.
It does not import PIPER, FastAPI, LangGraph, SQLite, or Ollama.

Dependencies: pandas, scikit-learn, joblib (and numpy via sklearn).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

PIPELINE_PATH = Path(__file__).resolve().parent / "pipeline.joblib"
REQUIRED_COLUMNS = {required_columns!r}
RUN_ID = {run_id!r}
TARGET_COLUMN = {target_column!r}
WINNING_ALGORITHM = {algorithm!r}


def load_pipeline():
    return joblib.load(PIPELINE_PATH)


def validate_columns(frame: pd.DataFrame) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required inference columns: {{missing}}")


def predict(frame: pd.DataFrame):
    """Accept a raw inference DataFrame; return a 1-d prediction array."""
    validate_columns(frame)
    pipeline = load_pipeline()
    return pipeline.predict(frame[list(REQUIRED_COLUMNS)])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with the exported PIPER pipeline.")
    parser.add_argument("csv_path", help="CSV of raw feature rows (must include REQUIRED_COLUMNS).")
    parser.add_argument("-o", "--output", help="Optional CSV path for predictions.")
    args = parser.parse_args()
    data = pd.read_csv(args.csv_path)
    preds = predict(data)
    out = pd.DataFrame({{"prediction": preds}})
    if args.output:
        out.to_csv(args.output, index=False)
    else:
        print(out.to_string(index=False))
'''


def render_pipeline_py(
    *,
    run_id: str,
    target_column: str,
    algorithm: str,
    feature_columns: Sequence[str],
) -> str:
    return _PIPELINE_PY_TEMPLATE.format(
        required_columns=list(feature_columns),
        run_id=run_id,
        target_column=target_column,
        algorithm=algorithm,
    )
