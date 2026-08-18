"""
Data-quality validation contract (section 9).

Runs BEFORE profiling — this is the "is this dataset even usable"
gate, distinct from check_constant_features()/check_high_cardinality()
etc. (which assume a basically-sane dataset and look for statistical
red flags). A dataset that fails data-quality validation never reaches
those guardrails at all.

Every check here classifies as a TERMINAL failure (DATA_ERROR,
SCHEMA_ERROR, or TARGET_ERROR — bypasses the retry loop entirely,
since no replan can fix "the dataset has zero rows"). There is no
RECOVERABLE data-quality failure by design: if the input itself is
malformed, retrying with a different plan cannot help — this mirrors
constraint #9's existing precedent (non-binary target -> hard failure,
no replan) and extends it to the rest of the input-validation surface.

Note on target_as_feature: this specific check is NOT performed here.
At this stage (before any plan/feature-intent exists), the target
column is just one column among many — there is no "feature list" yet
to check the target against. This check is meaningfully enforced
later, once a concrete feature set exists: train_model() already
rejects target_leakage_in_features (see agent/tools/training.py) if
the target column is ever listed inside a FeatureEngineeringIntent.
Section 9's "target-as-feature leakage" requirement is satisfied by
that existing, tested check, not duplicated here.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

DataQualityCheckType = Literal[
    "empty_dataset",
    "zero_columns",
    "empty_column",
    "duplicate_column_names",
    "missing_target",
    "constant_target",
    "insufficient_samples",
    "invalid_binary_target",
    "unsupported_feature_type",
]

# Every data-quality violation is terminal by construction (see module
# docstring) — this map exists so the failure-classification code has
# one place to look up which FailureCategory each check maps to,
# rather than switching on check_type ad hoc at every call site.
DATA_QUALITY_FAILURE_CATEGORY: dict = {
    "empty_dataset": "DATA_ERROR",
    "zero_columns": "SCHEMA_ERROR",
    "empty_column": "DATA_ERROR",
    "duplicate_column_names": "SCHEMA_ERROR",
    "missing_target": "TARGET_ERROR",
    "constant_target": "TARGET_ERROR",
    "insufficient_samples": "DATA_ERROR",
    "invalid_binary_target": "TARGET_ERROR",
    "unsupported_feature_type": "SCHEMA_ERROR",
}

MINIMUM_SAMPLES_REQUIRED = 20
"""
Locked minimum row count. Below this, a train/test split (even at a
generous 80/20) leaves too few rows per class to train or evaluate
meaningfully — this is a data-quality gate, not the same concern as
check_target_imbalance()'s class-ratio check.
"""


class DataQualityViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_type: DataQualityCheckType
    column: Optional[str] = Field(None, description="The affected column, if applicable (None for dataset-level issues like empty_dataset).")
    reason: str
    evidence: dict = Field(default_factory=dict)


class DataQualityReport(BaseModel):
    """
    Output of validate_data_quality(). valid=False if ANY violation is
    found — unlike the statistical guardrails, there is no
    warning-severity tier here: a malformed input is malformed,
    full stop.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    target_column: str
    valid: bool
    violations: list[DataQualityViolation] = Field(default_factory=list)
    rows_checked: int = Field(..., ge=0)
    columns_checked: int = Field(..., ge=0)
