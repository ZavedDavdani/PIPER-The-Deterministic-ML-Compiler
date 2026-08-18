"""
Detection tools: detect_missing_values(), detect_outliers().

Both are detection-only. Neither decides whether to drop, impute, or
remove anything — see profiling.py schema docstrings for why that
distinction is preserved architecturally, not just by convention.
"""

from __future__ import annotations

from app.agent.tools._profiling_helpers import is_numeric_dtype
from app.schemas import (
    MissingValueColumnEntry,
    MissingValueReport,
    OutlierColumnEntry,
    OutlierReport,
    ToolError,
    ToolResult,
)
from app.storage import DatasetNotFoundError, DatasetStore


def detect_missing_values(
    dataset_id: str, store: DatasetStore
) -> ToolResult[MissingValueReport]:
    """Identify missing-value patterns across every column."""
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[MissingValueReport](
            success=False,
            tool_name="detect_missing_values",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    rows = len(df)
    columns_with_missing: list[MissingValueColumnEntry] = []
    total_missing = 0

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            total_missing += missing_count
            columns_with_missing.append(
                MissingValueColumnEntry(
                    column=col,
                    count=missing_count,
                    percentage=round((missing_count / rows) * 100, 4) if rows else 0.0,
                )
            )

    report = MissingValueReport(
        dataset_id=dataset_id,
        total_missing=total_missing,
        columns_with_missing=columns_with_missing,
    )

    return ToolResult[MissingValueReport](
        success=True,
        tool_name="detect_missing_values",
        message=(
            f"Found {total_missing} missing values across "
            f"{len(columns_with_missing)} column(s)."
        ),
        data=report,
    )


def detect_outliers(dataset_id: str, store: DatasetStore) -> ToolResult[OutlierReport]:
    """
    Identify potential numerical outliers using the IQR method (V1
    fixed choice, per contract). Only numeric columns are examined;
    non-numeric columns are silently skipped, not reported as errors.
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[OutlierReport](
            success=False,
            tool_name="detect_outliers",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    rows = len(df)
    columns: list[OutlierColumnEntry] = []

    for col in df.columns:
        series = df[col]
        if not is_numeric_dtype(series):
            continue

        non_null = series.dropna()
        if len(non_null) < 4:
            # Not enough data for a meaningful quartile calculation —
            # skip rather than report a misleading result.
            continue

        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            # No spread — every value in the IQR range is identical,
            # so bounds would be degenerate (lower == upper). Skip
            # rather than flag everything as an outlier.
            continue

        lower_bound = float(q1 - 1.5 * iqr)
        upper_bound = float(q3 + 1.5 * iqr)

        outlier_mask = (non_null < lower_bound) | (non_null > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            columns.append(
                OutlierColumnEntry(
                    column=col,
                    method="IQR",
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    outlier_count=outlier_count,
                    outlier_percentage=round((outlier_count / rows) * 100, 4) if rows else 0.0,
                )
            )

    report = OutlierReport(dataset_id=dataset_id, columns=columns)

    return ToolResult[OutlierReport](
        success=True,
        tool_name="detect_outliers",
        message=f"Checked numeric columns for outliers; {len(columns)} column(s) flagged.",
        data=report,
    )
