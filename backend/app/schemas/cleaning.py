"""
Cleaning tool contracts: drop_column(), drop_duplicates(),
impute_missing_values(), convert_column_type().
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ImputationStrategy = Literal["mean", "median", "mode"]
TargetType = Literal["numeric", "string", "datetime", "boolean"]


class DropColumnResult(BaseModel):
    """
    Output of drop_column(col).

    Errors this tool must raise (not silently allow):
    - column doesn't exist
    - attempt to drop the target column
    - attempt to drop the final remaining feature
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    operation: Literal["drop_column"] = "drop_column"
    column: str
    reason: str = Field(..., description="Why the agent chose to drop this column.")
    rows_affected: int = Field(..., ge=0)
    columns_before: int = Field(..., ge=0)
    columns_after: int = Field(..., ge=0)


class DropDuplicatesResult(BaseModel):
    """
    Output of drop_duplicates().

    Must be idempotent — running it twice should not progressively
    alter the dataset (the second call should report
    duplicates_found=0, duplicates_removed=0).
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    operation: Literal["drop_duplicates"] = "drop_duplicates"
    duplicates_found: int = Field(..., ge=0)
    duplicates_removed: int = Field(..., ge=0)


class ImputationResult(BaseModel):
    """
    Output of impute_missing_values(col, strategy).

    Rules enforced at the tool level, not here:
    - mean/median → numeric columns only
    - mode → categorical or numeric
    - reject if column has no missing values
    - reject unsupported dtype/strategy combinations
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    column: str
    strategy: ImputationStrategy
    missing_before: int = Field(..., ge=0)
    missing_after: int = Field(..., ge=0)
    replacement_value: str = Field(
        ...,
        description="String representation of the value used (numeric or categorical).",
    )


class ConversionResult(BaseModel):
    """
    Output of convert_column_type(col, target_type).

    Failed conversions must never silently become garbage values. The
    tool either (a) reports the failed-row count and leaves those rows
    as missing, or (b) fails the operation outright if the failure
    rate exceeds the configured threshold (default 10% for V1).
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    column: str
    original_type: str
    target_type: TargetType
    converted_count: int = Field(..., ge=0)
    failed_conversion_count: int = Field(..., ge=0)
    failure_threshold_percent: float = Field(
        default=10.0,
        description="Configured max allowed failure rate before the tool refuses to apply the conversion.",
    )
    applied: bool = Field(
        ...,
        description="False if failure rate exceeded the threshold and the conversion was rejected.",
    )
