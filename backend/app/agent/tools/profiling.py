"""
Profiling tools: profile_dataset(), inspect_column().

Locked principle: tools never call the LLM and never decide what
action to take. These two only observe and report.
"""

from __future__ import annotations

from app.agent.tools._profiling_helpers import (
    infer_semantic_type,
    numeric_stats,
    sample_values,
)
from app.schemas import (
    ColumnInspection,
    ColumnProfile,
    DatasetProfile,
    ToolError,
    ToolResult,
)
from app.storage import DatasetNotFoundError, DatasetStore


def profile_dataset(dataset_id: str, store: DatasetStore) -> ToolResult[DatasetProfile]:
    """
    Produce the initial dataset profile.

    Errors handled per contract:
    - dataset does not exist
    - dataset is empty (zero rows)
    - dataset has zero columns
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[DatasetProfile](
            success=False,
            tool_name="profile_dataset",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if df.shape[1] == 0:
        return ToolResult[DatasetProfile](
            success=False,
            tool_name="profile_dataset",
            message=f"Dataset '{dataset_id}' has zero columns.",
            error=ToolError(
                code="dataset_has_no_columns",
                message="Dataset contains zero columns.",
                details={"dataset_id": dataset_id},
            ),
        )

    if df.shape[0] == 0:
        return ToolResult[DatasetProfile](
            success=False,
            tool_name="profile_dataset",
            message=f"Dataset '{dataset_id}' is empty.",
            error=ToolError(
                code="dataset_is_empty",
                message="Dataset is empty (zero rows).",
                details={"dataset_id": dataset_id},
            ),
        )

    rows = len(df)
    column_profiles: list[ColumnProfile] = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        stats = numeric_stats(series)

        column_profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                missing_count=missing_count,
                missing_percentage=round((missing_count / rows) * 100, 4) if rows else 0.0,
                unique_count=unique_count,
                unique_percentage=round((unique_count / rows) * 100, 4) if rows else 0.0,
                sample_values=sample_values(series),
                **stats,  # min/max/mean/median/std, or nothing if not numeric
            )
        )

    duplicate_rows = int(df.duplicated().sum())
    memory_usage_bytes = int(df.memory_usage(deep=True).sum())

    profile = DatasetProfile(
        dataset_id=dataset_id,
        rows=rows,
        columns=df.shape[1],
        column_profiles=column_profiles,
        duplicate_rows=duplicate_rows,
        memory_usage_bytes=memory_usage_bytes,
    )

    return ToolResult[DatasetProfile](
        success=True,
        tool_name="profile_dataset",
        message=f"Profiled dataset '{dataset_id}': {rows} rows, {df.shape[1]} columns.",
        data=profile,
    )


def inspect_column(
    dataset_id: str, column: str, store: DatasetStore
) -> ToolResult[ColumnInspection]:
    """
    Obtain detailed information about one column.

    Errors handled per contract:
    - dataset doesn't exist
    - column doesn't exist
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[ColumnInspection](
            success=False,
            tool_name="inspect_column",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if column not in df.columns:
        return ToolResult[ColumnInspection](
            success=False,
            tool_name="inspect_column",
            message=f"Column '{column}' does not exist in dataset '{dataset_id}'.",
            error=ToolError(
                code="column_not_found",
                message=f"Column '{column}' does not exist.",
                details={"dataset_id": dataset_id, "column": column},
            ),
        )

    series = df[column]
    rows = len(df)
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))
    unique_percentage = round((unique_count / rows) * 100, 4) if rows else 0.0

    semantic_type = infer_semantic_type(series, unique_percentage)
    stats = numeric_stats(series)

    top_values = []
    if semantic_type in ("categorical", "boolean", "identifier"):
        value_counts = series.value_counts(dropna=True).head(10)
        for value, count in value_counts.items():
            top_values.append(
                {
                    "value": value.item() if hasattr(value, "item") else value,
                    "count": int(count),
                    "percentage": round((count / rows) * 100, 4) if rows else 0.0,
                }
            )

    inspection = ColumnInspection(
        name=column,
        dtype=str(series.dtype),
        semantic_type=semantic_type,
        missing_count=missing_count,
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        statistics=stats,
        sample_values=sample_values(series),
        top_values=top_values,
    )

    return ToolResult[ColumnInspection](
        success=True,
        tool_name="inspect_column",
        message=f"Inspected column '{column}' in dataset '{dataset_id}'.",
        data=inspection,
    )
