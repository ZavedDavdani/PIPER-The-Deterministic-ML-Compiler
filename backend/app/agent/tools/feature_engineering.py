"""
Feature engineering tools.

encode_categorical_features() and scale_features() validate the
request and produce a descriptive result, but do NOT fit a
transformer — the fitted OneHotEncoder/StandardScaler is assembled
into a scikit-learn Pipeline at train_model() time, fit on the train
split only. See schemas/feature_engineering.py for the full rationale.

create_date_features() is different: it genuinely does mutate the
stored dataset, because date-component extraction (year/month/day/etc)
is a deterministic, leakage-free operation regardless of train/test
split — unlike encoding/scaling, there's no "fitting" involved, so
there's no reason to defer it.
"""

from __future__ import annotations

import pandas as pd

from app.agent.tools._profiling_helpers import is_numeric_dtype
from app.schemas import ToolError, ToolResult
from app.schemas.feature_engineering import (
    DateFeatureResult,
    EncodingResult,
    ScalingResult,
)
from app.storage import DatasetNotFoundError, DatasetStore

MAX_CARDINALITY_FOR_ONE_HOT = 20  # sensible sanity bound for V1, not from the locked contract


def encode_categorical_features(
    dataset_id: str, columns: list[str], store: DatasetStore
) -> ToolResult[EncodingResult]:
    """
    Validate a one-hot encoding request and return a preview of the
    columns it would generate. Does not mutate the stored dataset or
    fit an encoder — see module docstring.

    Errors:
    - dataset doesn't exist
    - any requested column doesn't exist
    - a requested column is numeric (one-hot encoding is for
      categorical data; encoding an already-numeric column is very
      likely a planning mistake worth surfacing, not silently allowing)
    - a requested column has too many distinct values to sensibly
      one-hot encode (V1 sanity bound, prevents e.g. accidentally
      encoding an identifier column into thousands of dummy columns)
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[EncodingResult](
            success=False,
            tool_name="encode_categorical_features",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    missing_columns = [c for c in columns if c not in df.columns]
    if missing_columns:
        return ToolResult[EncodingResult](
            success=False,
            tool_name="encode_categorical_features",
            message=f"Column(s) not found: {missing_columns}.",
            error=ToolError(
                code="column_not_found",
                message="One or more requested columns do not exist.",
                details={"dataset_id": dataset_id, "missing_columns": missing_columns},
            ),
        )

    numeric_columns = [c for c in columns if is_numeric_dtype(df[c])]
    if numeric_columns:
        return ToolResult[EncodingResult](
            success=False,
            tool_name="encode_categorical_features",
            message=f"Column(s) are numeric, not categorical: {numeric_columns}.",
            error=ToolError(
                code="column_is_numeric",
                message="One-hot encoding requires categorical columns.",
                details={"dataset_id": dataset_id, "numeric_columns": numeric_columns},
            ),
        )

    high_cardinality = {
        c: int(df[c].nunique(dropna=True))
        for c in columns
        if df[c].nunique(dropna=True) > MAX_CARDINALITY_FOR_ONE_HOT
    }
    if high_cardinality:
        return ToolResult[EncodingResult](
            success=False,
            tool_name="encode_categorical_features",
            message=f"Column(s) exceed max cardinality for one-hot encoding ({MAX_CARDINALITY_FOR_ONE_HOT}): {high_cardinality}.",
            error=ToolError(
                code="cardinality_too_high",
                message="One or more columns have too many distinct values for one-hot encoding.",
                details={"dataset_id": dataset_id, "cardinality": high_cardinality},
            ),
        )

    generated_preview: list[str] = []
    for col in columns:
        for category in sorted(df[col].dropna().unique().tolist(), key=str):
            generated_preview.append(f"{col}_{category}")

    result = EncodingResult(
        dataset_id=dataset_id,
        original_columns=columns,
        generated_columns=generated_preview,
        encoding="one_hot",
    )

    return ToolResult[EncodingResult](
        success=True,
        tool_name="encode_categorical_features",
        message=(
            f"Validated one-hot encoding request for {len(columns)} column(s); "
            f"preview generates {len(generated_preview)} column(s). "
            f"Actual encoder is fit at train_model() time on the train split only."
        ),
        data=result,
    )


def scale_features(
    dataset_id: str, columns: list[str], store: DatasetStore
) -> ToolResult[ScalingResult]:
    """
    Validate a scaling request. Does not fit a scaler — see module
    docstring.

    Errors:
    - dataset doesn't exist
    - any requested column doesn't exist
    - a requested column is not numeric (StandardScaler requires
      numeric input)
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[ScalingResult](
            success=False,
            tool_name="scale_features",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    missing_columns = [c for c in columns if c not in df.columns]
    if missing_columns:
        return ToolResult[ScalingResult](
            success=False,
            tool_name="scale_features",
            message=f"Column(s) not found: {missing_columns}.",
            error=ToolError(
                code="column_not_found",
                message="One or more requested columns do not exist.",
                details={"dataset_id": dataset_id, "missing_columns": missing_columns},
            ),
        )

    non_numeric_columns = [c for c in columns if not is_numeric_dtype(df[c])]
    if non_numeric_columns:
        return ToolResult[ScalingResult](
            success=False,
            tool_name="scale_features",
            message=f"Column(s) are not numeric: {non_numeric_columns}.",
            error=ToolError(
                code="column_not_numeric",
                message="StandardScaler requires numeric columns.",
                details={"dataset_id": dataset_id, "non_numeric_columns": non_numeric_columns},
            ),
        )

    result = ScalingResult(dataset_id=dataset_id, columns=columns, scaler="StandardScaler")

    return ToolResult[ScalingResult](
        success=True,
        tool_name="scale_features",
        message=(
            f"Validated scaling request for {len(columns)} column(s). "
            f"Actual scaler is fit at train_model() time on the train split only."
        ),
        data=result,
    )


def create_date_features(
    dataset_id: str, column: str, store: DatasetStore
) -> ToolResult[DateFeatureResult]:
    """
    Extract year/month/day/day_of_week from a date column. Unlike
    encode/scale, this genuinely mutates the stored dataset — date
    component extraction is deterministic and leakage-free regardless
    of train/test split.

    Errors:
    - dataset doesn't exist
    - column doesn't exist
    - column cannot be parsed as a date at all (returns an error
      rather than a partial/garbage result)
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[DateFeatureResult](
            success=False,
            tool_name="create_date_features",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if column not in df.columns:
        return ToolResult[DateFeatureResult](
            success=False,
            tool_name="create_date_features",
            message=f"Column '{column}' does not exist in dataset '{dataset_id}'.",
            error=ToolError(
                code="column_not_found",
                message=f"Column '{column}' does not exist.",
                details={"dataset_id": dataset_id, "column": column},
            ),
        )

    import warnings

    with warnings.catch_warnings():
        # Attempting to parse an obviously non-date column (e.g. free
        # text) triggers pandas' per-element dateutil fallback warning
        # on every call — this is an expected, handled path (we check
        # non_null_parsed below and return a clean error), not a
        # genuine warning-worthy situation, so it's suppressed here
        # rather than left to alarm every caller.
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(df[column], errors="coerce")
    non_null_original = df[column].notna().sum()
    non_null_parsed = parsed.notna().sum()

    if non_null_original > 0 and non_null_parsed == 0:
        return ToolResult[DateFeatureResult](
            success=False,
            tool_name="create_date_features",
            message=f"Column '{column}' could not be parsed as a date.",
            error=ToolError(
                code="unparseable_date_column",
                message="No values in the column could be parsed as a date.",
                details={"dataset_id": dataset_id, "column": column},
            ),
        )

    updated = df.copy()
    generated_features: list[str] = []

    updated[f"{column}_year"] = parsed.dt.year
    generated_features.append("year")

    updated[f"{column}_month"] = parsed.dt.month
    generated_features.append("month")

    updated[f"{column}_day"] = parsed.dt.day
    generated_features.append("day")

    updated[f"{column}_day_of_week"] = parsed.dt.dayofweek
    generated_features.append("day_of_week")

    store.save(dataset_id, updated)

    result = DateFeatureResult(
        dataset_id=dataset_id,
        source_column=column,
        generated_features=generated_features,
    )

    return ToolResult[DateFeatureResult](
        success=True,
        tool_name="create_date_features",
        message=f"Generated {len(generated_features)} date feature(s) from '{column}'.",
        data=result,
    )
