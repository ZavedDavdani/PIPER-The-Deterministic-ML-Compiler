"""
Baseline comparator contract (section 8).

Locked V1 policy (named here, never scattered as a magic literal
elsewhere — validate_pipeline() imports BASELINE_POLICY, it does not
hardcode 0.05):

    primary_metric               = "f1"
    minimum_primary_metric_delta = 0.05

    model_f1 - baseline_f1 < 0.05   -> baseline gate FAILED
    model_f1 - baseline_f1 >= 0.05  -> baseline gate PASSED

The threshold is fixed policy, not something the LLM (or any caller)
can adjust — BaselinePolicy has no field for it to be overridden
through; it's a module-level constant instantiated once.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class BaselinePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_metric: str = "f1"
    minimum_primary_metric_delta: float = 0.05


BASELINE_POLICY = BaselinePolicy()


class BaselineMetrics(BaseModel):
    """
    Metrics for the majority-class DummyClassifier baseline, evaluated
    on the SAME test split as the real model — never a different or
    re-drawn split, since that would make the comparison meaningless.
    """

    model_config = ConfigDict(extra="forbid")

    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: Optional[float] = Field(
        None, description="None if mathematically undefined (e.g. majority class never predicted the positive label) — never silently coerced to 0.0."
    )
    recall: Optional[float] = None
    f1: Optional[float] = None
    majority_class: str


class BaselineComparisonResult(BaseModel):
    """
    Output of the baseline-gate check. gate_passed is computed
    strictly from BASELINE_POLICY — never a caller-supplied threshold.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    split_id: str
    baseline: BaselineMetrics
    model_primary_metric_value: float
    baseline_primary_metric_value: Optional[float] = Field(
        None, description="None if the baseline's primary metric was undefined — see BaselineMetrics.f1."
    )
    primary_metric: str
    delta: Optional[float] = Field(
        None, description="model_primary_metric_value - baseline_primary_metric_value, or None if the baseline metric was undefined."
    )
    minimum_required_delta: float
    gate_passed: bool
    reason: str = Field(
        ..., description="Human-readable explanation, especially important when gate_passed=False or the delta is undefined."
    )
