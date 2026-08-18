"""
validate_data_quality() (section 9).

    RAW DATA
        |
        v
    validate_data_quality()  <-- this tool
        |
        v
    PROFILE (only reached if valid=True)

Every violation here is terminal (see schemas/data_quality.py module
docstring) — there's no "warning" tier and no scenario where a
data-quality violation triggers REPLAN. The graph node that wraps this
tool (validate_input_node in real_nodes.py) routes straight to failure
with retry_count untouched, the same precedent constraint #9 already
established for non-binary targets.
"""

from __future__ import annotations

import pandas as pd

from app.agent.tools._profiling_helpers import is_numeric_dtype
from app.schemas import ToolError, ToolResult
from app.schemas.data_quality import (
    DataQualityReport,
    DataQualityViolation,
    MINIMUM_SAMPLES_REQUIRED,
)
from app.storage import DatasetNotFoundError, DatasetStore

# Types considered "supported" for V1 tabular binary classification.
# Anything else (e.g. a column of lists/dicts, or an unrecognizable
# object dtype that isn't plain text) is flagged as
# unsupported_feature_type rather than silently passed downstream
# where cleaning/feature-engineering tools would fail confusingly.
_SUPPORTED_KIND_PREFIXES = ("i", "u", "f", "b", "O", "S", "U", "M")  # int, uint, float, bool, object/str, datetime


def validate_data_quality(
    dataset_id: str,
    target_column: str,
    store: DatasetStore,
) -> ToolResult[DataQualityReport]:
    """
    Runs every data-quality check from section 9 and returns
    structured evidence. Never mutates the dataset.

    Tool-level errors (not data-quality violations — these mean the
    tool itself couldn't run):
    - dataset doesn't exist
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[DataQualityReport](
            success=False,
            tool_name="validate_data_quality",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    violations: list[DataQualityViolation] = []

    # --- zero_columns / empty_dataset (checked first — everything else
    # assumes at least some columns/rows exist to inspect) -------------
    if df.shape[1] == 0:
        violations.append(
            DataQualityViolation(
                check_type="zero_columns",
                column=None,
                reason="Dataset has zero columns.",
                evidence={"columns": 0},
            )
        )
        report = DataQualityReport(
            dataset_id=dataset_id, target_column=target_column, valid=False,
            violations=violations, rows_checked=len(df), columns_checked=0,
        )
        return ToolResult[DataQualityReport](
            success=True, tool_name="validate_data_quality",
            message="Dataset has zero columns.", data=report,
        )

    if df.shape[0] == 0:
        violations.append(
            DataQualityViolation(
                check_type="empty_dataset",
                column=None,
                reason="Dataset has zero rows.",
                evidence={"rows": 0},
            )
        )
        report = DataQualityReport(
            dataset_id=dataset_id, target_column=target_column, valid=False,
            violations=violations, rows_checked=0, columns_checked=df.shape[1],
        )
        return ToolResult[DataQualityReport](
            success=True, tool_name="validate_data_quality",
            message="Dataset has zero rows.", data=report,
        )

    # --- duplicate_column_names ----------------------------------------
    column_list = list(df.columns)
    seen = set()
    duplicates = set()
    for col in column_list:
        if col in seen:
            duplicates.add(col)
        seen.add(col)
    if duplicates:
        violations.append(
            DataQualityViolation(
                check_type="duplicate_column_names",
                column=None,
                reason=f"Duplicate column name(s) found: {sorted(duplicates)}.",
                evidence={"duplicate_columns": sorted(duplicates)},
            )
        )

    # --- empty_column (entirely null) -----------------------------------
    # Use positional iteration (df.iloc[:, i]) rather than df[col],
    # since df[col] returns a DataFrame (not a Series) when the column
    # name is duplicated — discovered as a genuine crash during
    # testing, not assumed. This makes the empty-column and
    # unsupported-type checks correct even when duplicate_column_names
    # also fired above.
    for i, col in enumerate(column_list):
        series = df.iloc[:, i]
        if series.notna().sum() == 0:
            violations.append(
                DataQualityViolation(
                    check_type="empty_column",
                    column=col,
                    reason=f"Column '{col}' (position {i}) is entirely null.",
                    evidence={"non_null_count": 0, "column_position": i},
                )
            )

    # --- unsupported_feature_type ----------------------------------------
    for i, col in enumerate(column_list):
        series = df.iloc[:, i]
        kind = series.dtype.kind
        if kind not in _SUPPORTED_KIND_PREFIXES:
            violations.append(
                DataQualityViolation(
                    check_type="unsupported_feature_type",
                    column=col,
                    reason=f"Column '{col}' (position {i}) has an unsupported dtype ({series.dtype}); expected numeric, text, boolean, or datetime.",
                    evidence={"dtype": str(series.dtype), "dtype_kind": kind, "column_position": i},
                )
            )
        elif series.dtype == object:
            # An object-dtype column could be genuine text OR could
            # contain unhashable/complex Python objects (lists, dicts)
            # that will break downstream tools in confusing ways —
            # sample and check.
            sample = series.dropna().head(20)
            if any(isinstance(v, (list, dict, set, tuple)) for v in sample):
                violations.append(
                    DataQualityViolation(
                        check_type="unsupported_feature_type",
                        column=col,
                        reason=f"Column '{col}' (position {i}) contains complex Python objects (list/dict/set/tuple), not scalar text values.",
                        evidence={"dtype": str(series.dtype), "column_position": i},
                    )
                )

    # --- missing_target / constant_target / invalid_binary_target -------
    target_positions = [i for i, c in enumerate(column_list) if c == target_column]

    if not target_positions:
        violations.append(
            DataQualityViolation(
                check_type="missing_target",
                column=target_column,
                reason=f"Target column '{target_column}' does not exist in the dataset.",
                evidence={"available_columns": column_list},
            )
        )
    else:
        # If the target name is itself duplicated, use the first
        # occurrence — duplicate_column_names has already been flagged
        # above as its own violation, so this doesn't hide that issue.
        target_series = df.iloc[:, target_positions[0]]
        distinct_non_null = target_series.dropna().nunique()

        if distinct_non_null < 2:
            violations.append(
                DataQualityViolation(
                    check_type="constant_target",
                    column=target_column,
                    reason=f"Target column '{target_column}' has {distinct_non_null} distinct non-null value(s); a constant target cannot be predicted.",
                    evidence={"distinct_values": int(distinct_non_null)},
                )
            )
        elif distinct_non_null > 2:
            violations.append(
                DataQualityViolation(
                    check_type="invalid_binary_target",
                    column=target_column,
                    reason=f"Target column '{target_column}' has {distinct_non_null} distinct values; V1 requires exactly 2 (binary classification).",
                    evidence={"distinct_values": int(distinct_non_null)},
                )
            )

    # --- insufficient_samples --------------------------------------------
    if len(df) < MINIMUM_SAMPLES_REQUIRED:
        violations.append(
            DataQualityViolation(
                check_type="insufficient_samples",
                column=None,
                reason=f"Dataset has {len(df)} row(s); at least {MINIMUM_SAMPLES_REQUIRED} are required for a meaningful train/test split.",
                evidence={"rows": len(df), "minimum_required": MINIMUM_SAMPLES_REQUIRED},
            )
        )

    valid = len(violations) == 0

    report = DataQualityReport(
        dataset_id=dataset_id,
        target_column=target_column,
        valid=valid,
        violations=violations,
        rows_checked=len(df),
        columns_checked=df.shape[1],
    )

    message = (
        f"Data quality check passed: {df.shape[1]} column(s), {len(df)} row(s)."
        if valid
        else f"Data quality check found {len(violations)} violation(s)."
    )

    return ToolResult[DataQualityReport](
        success=True,
        tool_name="validate_data_quality",
        message=message,
        data=report,
    )
