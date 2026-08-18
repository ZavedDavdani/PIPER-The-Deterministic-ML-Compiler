"""
Cleaning tools, part 2: impute_missing_values(), convert_column_type().

convert_column_type() is the tool that produces the TotalCharges
scenario the whole chain test depends on: 11 blank strings -> coerced
to NaN on failed numeric conversion -> becomes visible to
detect_missing_values() -> resolved by impute_missing_values().
"""

from __future__ import annotations

import pandas as pd

from app.agent.tools._profiling_helpers import is_numeric_dtype
from app.schemas import ConversionResult, ImputationResult, ToolError, ToolResult
from app.storage import DatasetNotFoundError, DatasetStore

_STRATEGY_ALLOWED_FOR_NUMERIC = {"mean", "median", "mode"}
_STRATEGY_ALLOWED_FOR_CATEGORICAL = {"mode"}


def impute_missing_values(
    dataset_id: str, column: str, strategy: str, store: DatasetStore
) -> ToolResult[ImputationResult]:
    """
    Impute missing values in a column.

    Rules enforced (per contract):
    - mean/median -> numeric columns only
    - mode -> categorical or numeric
    - reject if the column has no missing values
    - reject unsupported dtype/strategy combinations
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[ImputationResult](
            success=False,
            tool_name="impute_missing_values",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if column not in df.columns:
        return ToolResult[ImputationResult](
            success=False,
            tool_name="impute_missing_values",
            message=f"Column '{column}' does not exist in dataset '{dataset_id}'.",
            error=ToolError(
                code="column_not_found",
                message=f"Column '{column}' does not exist.",
                details={"dataset_id": dataset_id, "column": column},
            ),
        )

    if strategy not in ("mean", "median", "mode"):
        return ToolResult[ImputationResult](
            success=False,
            tool_name="impute_missing_values",
            message=f"Unsupported strategy '{strategy}'.",
            error=ToolError(
                code="unsupported_strategy",
                message="strategy must be one of: mean, median, mode.",
                details={"dataset_id": dataset_id, "column": column, "strategy": strategy},
            ),
        )

    series = df[column]
    missing_before = int(series.isna().sum())

    if missing_before == 0:
        return ToolResult[ImputationResult](
            success=False,
            tool_name="impute_missing_values",
            message=f"Column '{column}' has no missing values; nothing to impute.",
            error=ToolError(
                code="no_missing_values",
                message="Column has no missing values.",
                details={"dataset_id": dataset_id, "column": column},
            ),
        )

    numeric = is_numeric_dtype(series)

    if strategy in ("mean", "median") and not numeric:
        return ToolResult[ImputationResult](
            success=False,
            tool_name="impute_missing_values",
            message=f"Strategy '{strategy}' requires a numeric column; '{column}' is not numeric.",
            error=ToolError(
                code="unsupported_dtype_strategy_combination",
                message=f"'{strategy}' is only valid for numeric columns.",
                details={"dataset_id": dataset_id, "column": column, "strategy": strategy},
            ),
        )

    if strategy == "mean":
        replacement = series.dropna().mean()
    elif strategy == "median":
        replacement = series.dropna().median()
    else:  # mode — valid for numeric or categorical
        mode_values = series.dropna().mode()
        if mode_values.empty:
            return ToolResult[ImputationResult](
                success=False,
                tool_name="impute_missing_values",
                message=f"Column '{column}' has no non-null values to compute a mode from.",
                error=ToolError(
                    code="no_mode_available",
                    message="Cannot compute mode: no non-null values present.",
                    details={"dataset_id": dataset_id, "column": column},
                ),
            )
        replacement = mode_values.iloc[0]

    updated = df.copy()
    updated[column] = updated[column].fillna(replacement)
    store.save(dataset_id, updated)

    missing_after = int(updated[column].isna().sum())
    replacement_str = (
        f"{replacement:.4f}" if isinstance(replacement, float) else str(replacement)
    )

    result = ImputationResult(
        dataset_id=dataset_id,
        column=column,
        strategy=strategy,  # type: ignore[arg-type]
        missing_before=missing_before,
        missing_after=missing_after,
        replacement_value=replacement_str,
    )

    return ToolResult[ImputationResult](
        success=True,
        tool_name="impute_missing_values",
        message=(
            f"Imputed {missing_before} missing value(s) in '{column}' "
            f"using {strategy} (replacement: {replacement_str})."
        ),
        data=result,
    )


def convert_column_type(
    dataset_id: str,
    column: str,
    target_type: str,
    store: DatasetStore,
    failure_threshold_percent: float = 10.0,
) -> ToolResult[ConversionResult]:
    """
    Convert a column to a target type.

    Critical rule: failed conversions must never silently become
    garbage values. Rows that fail to convert become NaN (missing) —
    never a corrupted value. If the failure rate exceeds
    failure_threshold_percent, the conversion is rejected outright and
    the dataset is left unchanged (applied=False).
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[ConversionResult](
            success=False,
            tool_name="convert_column_type",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if column not in df.columns:
        return ToolResult[ConversionResult](
            success=False,
            tool_name="convert_column_type",
            message=f"Column '{column}' does not exist in dataset '{dataset_id}'.",
            error=ToolError(
                code="column_not_found",
                message=f"Column '{column}' does not exist.",
                details={"dataset_id": dataset_id, "column": column},
            ),
        )

    if target_type not in ("numeric", "string", "datetime", "boolean"):
        return ToolResult[ConversionResult](
            success=False,
            tool_name="convert_column_type",
            message=f"Unsupported target_type '{target_type}'.",
            error=ToolError(
                code="unsupported_target_type",
                message="target_type must be one of: numeric, string, datetime, boolean.",
                details={"dataset_id": dataset_id, "column": column, "target_type": target_type},
            ),
        )

    series = df[column]
    original_type = str(series.dtype)
    rows = len(series)

    # Values that are already null before conversion don't count as
    # "failed conversions" — only non-null values that fail to coerce
    # into the target type count.
    already_null = series.isna()
    non_null_count = int((~already_null).sum())

    if target_type == "numeric":
        # Blank/whitespace-only strings must coerce to NaN, not to 0
        # or any other silently-wrong value. Strip first so "  " and
        # "" behave identically to a true empty value.
        stripped = series.astype(str).str.strip()
        stripped = stripped.mask(already_null, series)  # preserve true NaNs as NaN, not "nan" string
        converted = pd.to_numeric(stripped, errors="coerce")
    elif target_type == "string":
        converted = series.astype("string")
        converted = converted.mask(already_null)
    elif target_type == "datetime":
        converted = pd.to_datetime(series, errors="coerce")
    else:  # boolean
        truthy = {"true", "1", "yes", "y"}
        falsy = {"false", "0", "no", "n"}

        def _to_bool(val):
            if pd.isna(val):
                return pd.NA
            s = str(val).strip().lower()
            if s in truthy:
                return True
            if s in falsy:
                return False
            return pd.NA  # unrecognized -> failed conversion, not a guess

        converted = series.map(_to_bool).astype("boolean")

    failed_mask = converted.isna() & ~already_null
    failed_conversion_count = int(failed_mask.sum())
    converted_count = non_null_count - failed_conversion_count

    failure_rate_percent = (
        (failed_conversion_count / non_null_count) * 100 if non_null_count > 0 else 0.0
    )

    applied = failure_rate_percent <= failure_threshold_percent

    if applied:
        updated = df.copy()
        updated[column] = converted
        store.save(dataset_id, updated)
        message = (
            f"Converted '{column}' to {target_type}: {converted_count} succeeded, "
            f"{failed_conversion_count} failed (became missing)."
        )
    else:
        message = (
            f"Conversion of '{column}' to {target_type} rejected: "
            f"{failure_rate_percent:.1f}% failure rate exceeds "
            f"{failure_threshold_percent}% threshold. Dataset left unchanged."
        )

    result = ConversionResult(
        dataset_id=dataset_id,
        column=column,
        original_type=original_type,
        target_type=target_type,  # type: ignore[arg-type]
        converted_count=converted_count,
        failed_conversion_count=failed_conversion_count,
        failure_threshold_percent=failure_threshold_percent,
        applied=applied,
    )

    return ToolResult[ConversionResult](
        success=True,  # the tool executed correctly even if it declined to apply the conversion
        tool_name="convert_column_type",
        message=message,
        data=result,
    )
