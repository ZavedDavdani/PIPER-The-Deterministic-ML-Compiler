"""
Cleaning tools, part 1: drop_column(), drop_duplicates().
"""

from __future__ import annotations

from app.schemas import DropColumnResult, DropDuplicatesResult, ToolError, ToolResult
from app.storage import DatasetNotFoundError, DatasetStore


def drop_column(
    dataset_id: str,
    column: str,
    reason: str,
    store: DatasetStore,
    target_column: str | None = None,
) -> ToolResult[DropColumnResult]:
    """
    Drop a column from the dataset.

    Errors handled per contract:
    - dataset doesn't exist
    - column doesn't exist
    - attempt to drop the target column
    - attempt to drop the final remaining feature
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[DropColumnResult](
            success=False,
            tool_name="drop_column",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if column not in df.columns:
        return ToolResult[DropColumnResult](
            success=False,
            tool_name="drop_column",
            message=f"Column '{column}' does not exist in dataset '{dataset_id}'.",
            error=ToolError(
                code="column_not_found",
                message=f"Column '{column}' does not exist.",
                details={"dataset_id": dataset_id, "column": column},
            ),
        )

    if target_column is not None and column == target_column:
        return ToolResult[DropColumnResult](
            success=False,
            tool_name="drop_column",
            message=f"Cannot drop '{column}': it is the target column.",
            error=ToolError(
                code="target_column_protected",
                message="The target column cannot be dropped.",
                details={"dataset_id": dataset_id, "column": column},
            ),
        )

    if df.shape[1] <= 1:
        return ToolResult[DropColumnResult](
            success=False,
            tool_name="drop_column",
            message=f"Cannot drop '{column}': it is the final remaining feature.",
            error=ToolError(
                code="final_feature_protected",
                message="Cannot drop the final remaining feature column.",
                details={"dataset_id": dataset_id, "column": column},
            ),
        )

    columns_before = df.shape[1]
    rows_affected = len(df)
    updated = df.drop(columns=[column])
    store.save(dataset_id, updated)

    result = DropColumnResult(
        dataset_id=dataset_id,
        column=column,
        reason=reason,
        rows_affected=rows_affected,
        columns_before=columns_before,
        columns_after=updated.shape[1],
    )

    return ToolResult[DropColumnResult](
        success=True,
        tool_name="drop_column",
        message=f"Dropped column '{column}'. Columns: {columns_before} -> {updated.shape[1]}.",
        data=result,
    )


def drop_duplicates(dataset_id: str, store: DatasetStore) -> ToolResult[DropDuplicatesResult]:
    """
    Remove fully duplicate rows.

    Idempotent by construction: running this twice in a row will find
    duplicates_found=0 the second time, since the first call already
    removed them from the stored dataset.
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[DropDuplicatesResult](
            success=False,
            tool_name="drop_duplicates",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    duplicates_found = int(df.duplicated().sum())
    updated = df.drop_duplicates()
    store.save(dataset_id, updated)

    result = DropDuplicatesResult(
        dataset_id=dataset_id,
        duplicates_found=duplicates_found,
        duplicates_removed=duplicates_found,
    )

    return ToolResult[DropDuplicatesResult](
        success=True,
        tool_name="drop_duplicates",
        message=f"Found and removed {duplicates_found} duplicate row(s).",
        data=result,
    )
