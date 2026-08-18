"""
Profiling tool contracts: profile_dataset(), inspect_column(),
detect_missing_values(), detect_outliers().

These tools only observe and report. None of them decide whether to
drop, impute, or remove anything — that distinction is preserved here
by keeping every model purely descriptive (no "recommended_action"
fields anywhere in this file).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SemanticType = Literal["numeric", "categorical", "identifier", "datetime", "boolean", "unknown"]


class ColumnProfile(BaseModel):
    """One column's summary as returned inside DatasetProfile."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str = Field(..., description="Pandas dtype as a string, e.g. 'int64', 'object'.")
    missing_count: int = Field(..., ge=0)
    missing_percentage: float = Field(..., ge=0.0, le=100.0)
    unique_count: int = Field(..., ge=0)
    unique_percentage: float = Field(..., ge=0.0, le=100.0)
    sample_values: list = Field(default_factory=list, max_length=10)

    # Numeric-only fields — populated only when dtype is numeric.
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None


class DatasetProfile(BaseModel):
    """Output of profile_dataset()."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    rows: int = Field(..., ge=0)
    columns: int = Field(..., ge=0)
    column_profiles: list[ColumnProfile]
    duplicate_rows: int = Field(..., ge=0)
    memory_usage_bytes: int = Field(..., ge=0)


class ColumnInspection(BaseModel):
    """Output of inspect_column(col) — one column, more detail than ColumnProfile."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    semantic_type: SemanticType
    missing_count: int = Field(..., ge=0)
    unique_count: int = Field(..., ge=0)
    unique_percentage: float = Field(..., ge=0.0, le=100.0)
    statistics: dict = Field(
        default_factory=dict,
        description="Numeric-only stats (min/max/mean/median/std) when applicable, else empty.",
    )
    sample_values: list = Field(default_factory=list, max_length=10)
    top_values: list[dict] = Field(
        default_factory=list,
        description="e.g. [{'value': 'Yes', 'count': 5174, 'percentage': 73.46}, ...]",
    )


class MissingValueColumnEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0.0, le=100.0)


class MissingValueReport(BaseModel):
    """
    Output of detect_missing_values().

    Detection only — this tool does not decide whether to drop or
    impute. That decision belongs to the agent (plan[]), executed via
    impute_missing_values() or drop_column().
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    total_missing: int = Field(..., ge=0)
    columns_with_missing: list[MissingValueColumnEntry]


class OutlierColumnEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    method: Literal["IQR"] = "IQR"
    lower_bound: float
    upper_bound: float
    outlier_count: int = Field(..., ge=0)
    outlier_percentage: float = Field(..., ge=0.0, le=100.0)


class OutlierReport(BaseModel):
    """
    Output of detect_outliers().

    Detection only, IQR method for V1. An outlier is never
    automatically removed — the agent must reason about whether it's
    bad data or a legitimate observation before acting.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    columns: list[OutlierColumnEntry]
