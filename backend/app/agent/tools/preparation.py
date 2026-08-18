"""
Preparation tool: split_dataset().

Deliberately built before feature-engineering tools (encode/scale)
even though the locked contract lists preparation after feature
engineering — sklearn-based encoding/scaling must be fit on the train
split only to avoid leakage, so a real split has to exist first for
those tools to be tested honestly.
"""

from __future__ import annotations

import uuid

from sklearn.model_selection import train_test_split

from app.schemas import ToolError, ToolResult
from app.schemas.preparation import SplitResult
from app.storage import DatasetNotFoundError, DatasetStore, SplitStore

RANDOM_STATE = 42  # locked: reproducibility


def split_dataset(
    dataset_id: str,
    target: str,
    test_size: float,
    dataset_store: DatasetStore,
    split_store: SplitStore,
) -> ToolResult[SplitResult]:
    """
    Split a dataset into train/test sets.

    For binary classification, stratified=True by default (locked
    contract). random_state is fixed at 42 for reproducibility
    (locked). The resulting train/test DataFrames are written to
    SplitStore under a new split_id; the original dataset in
    DatasetStore is left untouched.

    Errors:
    - dataset doesn't exist
    - target column doesn't exist
    - test_size not in (0, 1)
    - target has fewer than 2 classes, or a class has too few members
      to stratify (sklearn's own requirement: at least 2 members per
      class when stratifying)
    """
    try:
        df = dataset_store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[SplitResult](
            success=False,
            tool_name="split_dataset",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if target not in df.columns:
        return ToolResult[SplitResult](
            success=False,
            tool_name="split_dataset",
            message=f"Target column '{target}' does not exist in dataset '{dataset_id}'.",
            error=ToolError(
                code="column_not_found",
                message=f"Target column '{target}' does not exist.",
                details={"dataset_id": dataset_id, "target": target},
            ),
        )

    if not (0.0 < test_size < 1.0):
        return ToolResult[SplitResult](
            success=False,
            tool_name="split_dataset",
            message=f"test_size must be between 0 and 1 (exclusive); got {test_size}.",
            error=ToolError(
                code="invalid_test_size",
                message="test_size must be strictly between 0 and 1.",
                details={"dataset_id": dataset_id, "test_size": test_size},
            ),
        )

    target_series = df[target]
    distinct_classes = target_series.dropna().nunique()

    if distinct_classes != 2:
        return ToolResult[SplitResult](
            success=False,
            tool_name="split_dataset",
            message=(
                f"Target column '{target}' has {distinct_classes} distinct "
                f"value(s); split_dataset() requires exactly 2 for binary "
                f"classification stratification."
            ),
            error=ToolError(
                code="target_not_binary",
                message="Target column must have exactly 2 distinct values.",
                details={"dataset_id": dataset_id, "target": target, "distinct_classes": distinct_classes},
            ),
        )

    min_class_count = int(target_series.value_counts().min())
    if min_class_count < 2:
        return ToolResult[SplitResult](
            success=False,
            tool_name="split_dataset",
            message=(
                f"Smallest class in '{target}' has only {min_class_count} "
                f"member(s); cannot stratify a split with fewer than 2 "
                f"members in any class."
            ),
            error=ToolError(
                code="insufficient_class_members_for_stratification",
                message="A class has fewer than 2 members; cannot stratify.",
                details={"dataset_id": dataset_id, "target": target, "min_class_count": min_class_count},
            ),
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=target_series,
        random_state=RANDOM_STATE,
    )

    split_id = f"split_{uuid.uuid4().hex[:8]}"
    split_store.save(split_id, train_df, test_df)

    result = SplitResult(
        dataset_id=dataset_id,
        split_id=split_id,
        target=target,
        train_rows=len(train_df),
        test_rows=len(test_df),
        test_size=test_size,
        stratified=True,
        random_state=RANDOM_STATE,
    )

    return ToolResult[SplitResult](
        success=True,
        tool_name="split_dataset",
        message=(
            f"Split dataset '{dataset_id}' into {len(train_df)} train / "
            f"{len(test_df)} test rows (stratified on '{target}')."
        ),
        data=result,
    )
