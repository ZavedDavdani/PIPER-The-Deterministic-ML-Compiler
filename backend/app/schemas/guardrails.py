"""
Guardrail contract: check_data_leakage().

Locked design principle: this tool produces EVIDENCE, not a verdict.
It reports what it found, with numbers, so the agent (and a human
reading the report) can explain the finding. It does NOT delete
features or make routing decisions — that's the graph's job,
downstream, using this evidence.

Equally important: this tool checks FEATURE-LEVEL leakage indicators
only. It does NOT and cannot verify PIPELINE-LEVEL leakage (train/test
contamination) — that class of leakage is prevented structurally by
the architecture itself (split -> fit Pipeline on train only ->
evaluate on raw test, proven by the monkeypatched-fit tests in
train_model()/evaluate_model()). This tool has no way to "prove" that
absence, so it never claims to. Its `valid=True` result means "no
feature-level leakage indicators detected by the checks implemented
here" — not "leakage is absent." This distinction is deliberate and
should never be softened into an overclaim.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LeakageCheckType = Literal[
    "high_correlation",
    "categorical_near_perfect_association",
    "identifier_like_column",
    "duplicate_of_target",
]


class LeakageViolation(BaseModel):
    """
    One piece of evidence. `evidence` always carries the actual
    number(s) behind the finding — never just a label — so the result
    is inspectable and explainable, not a bare assertion.
    """

    model_config = ConfigDict(extra="forbid")

    feature: str
    check_type: LeakageCheckType
    reason: str
    evidence: dict = Field(
        ...,
        description="Actual numbers behind the finding, e.g. {'correlation': 0.98} or {'unique_percentage': 100.0}.",
    )


class LeakageReport(BaseModel):
    """
    Output of check_data_leakage().

    valid=True means "no feature-level leakage indicators detected by
    the checks implemented here" — an explicit, honest scope
    statement, not a claim that leakage is absent. See module
    docstring for the full rationale.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    target_column: str
    leakage_detected: bool
    violations: list[LeakageViolation] = Field(default_factory=list)
    features_checked: list[str] = Field(
        default_factory=list,
        description="Which feature columns were actually examined — makes the scope of the check explicit.",
    )
    scope_note: str = Field(
        default=(
            "This check covers feature-level leakage indicators only "
            "(correlation, near-perfect categorical association, "
            "identifier-like columns, exact target duplication). "
            "Pipeline-level leakage (train/test contamination) is "
            "prevented structurally by this system's architecture "
            "(fit-on-train-only) and is not re-verified here."
        ),
        description="Explicit statement of what this check does and does not prove.",
    )


# --- check_target_imbalance() -------------------------------------------
#
# CONTRACT MIGRATION NOTICE: originally ratio-based (severely_imbalanced
# = majority/minority ratio > 9.0, i.e. >90/10, warning-only, no
# failure tier). Deliberately migrated to a minority-PERCENTAGE-based,
# three-tier contract:
#
#     minority_percentage >= 15.0            -> OK
#     5.0 <= minority_percentage < 15.0       -> WARNING
#     minority_percentage < 5.0               -> FAILURE
#
# This is an intentional, authorized contract change. severely_imbalanced
# is retained as a derived compatibility property (True iff
# severity == FAILURE) for any code still reading the old boolean shape.

from enum import Enum


class ImbalanceSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    FAILURE = "failure"


IMBALANCE_WARNING_THRESHOLD_PERCENT = 15.0
IMBALANCE_FAILURE_THRESHOLD_PERCENT = 5.0


class ClassCount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0.0, le=100.0)


class ImbalanceReport(BaseModel):
    """
    Output of check_target_imbalance().

    Locked policy (post-migration, see module note above):
        minority_percentage >= 15.0            -> OK
        5.0 <= minority_percentage < 15.0       -> WARNING
        minority_percentage < 5.0               -> FAILURE

    The 5% boundary is strict: exactly 5.0% minority is WARNING, not
    FAILURE (FAILURE requires strictly below 5.0%).

    severity is the AUTHORITATIVE field. severely_imbalanced is a
    derived, read-only compatibility property (True iff
    severity == FAILURE) — it intentionally cannot distinguish WARNING
    from OK, so any code needing that distinction must read severity
    directly, never severely_imbalanced.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    target_column: str
    class_counts: list[ClassCount]
    minority_label: str
    minority_percentage: float
    majority_label: str
    majority_percentage: float
    severity: ImbalanceSeverity
    warning_threshold_percent: float
    failure_threshold_percent: float
    reason: str

    @property
    def severely_imbalanced(self) -> bool:
        return self.severity is ImbalanceSeverity.FAILURE


# --- check_constant_features() ------------------------------------------


class ConstantFeatureEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    constant_value: str = Field(..., description="String representation of the single value present (or 'NaN-only' if entirely null).")
    non_null_count: int = Field(..., ge=0)


class ConstantFeatureReport(BaseModel):
    """
    Output of check_constant_features().

    A constant feature (zero variance — one distinct non-null value,
    or entirely null) carries no predictive information and cannot
    meaningfully contribute to a trained model. This tool reports
    which columns are constant; it does NOT remove them — that
    decision belongs to the agent/graph, per the same evidence-not-
    verdict principle used throughout this guardrail layer.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    constant_columns: list[ConstantFeatureEntry] = Field(default_factory=list)
    columns_checked: list[str] = Field(default_factory=list)


# --- check_high_cardinality() -------------------------------------------

HIGH_CARDINALITY_UNIQUENESS_THRESHOLD_PERCENT = 99.0  # locked: >99% unique


class HighCardinalityEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    unique_count: int = Field(..., ge=0)
    unique_percentage: float = Field(..., ge=0.0, le=100.0)
    reason: str


class HighCardinalityReport(BaseModel):
    """
    Output of check_high_cardinality().

    Locked rule: >99% unique is flagged as high-cardinality/
    identifier-like — but ONLY for non-numeric (categorical/text)
    columns. A continuous numeric feature being highly or fully unique
    is normal and expected (e.g. a price, a measurement), not a sign
    it's an identifier — this is the exact same distinction that was a
    real bug in check_data_leakage() (see that tool's test suite for
    the original bug and its regression test) and is deliberately
    reused here, not reinvented.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    suspicious_columns: list[HighCardinalityEntry] = Field(default_factory=list)
    columns_checked: list[str] = Field(default_factory=list)
    scope_note: str = Field(
        default=(
            "This check applies only to non-numeric (categorical/text) "
            "columns. Continuous numeric columns are excluded by design "
            "— high uniqueness in a numeric feature (e.g. a price or "
            "measurement) is expected and not an identifier signal."
        ),
    )


# --- validate_pipeline() -------------------------------------------------

ValidationSeverity = Literal["info", "warning", "error"]


class ValidationCheck(BaseModel):
    """
    One named check's outcome inside the combined validation result.
    Distinct from LeakageViolation/HighCardinalityEntry/etc. (which
    are per-COLUMN findings from an individual guardrail) — this is
    one row per GUARDRAIL, summarizing whether that guardrail's
    evidence, taken as a whole, passed or failed.
    """

    model_config = ConfigDict(extra="forbid")

    check: str
    passed: bool
    severity: ValidationSeverity
    message: str


class PipelineValidationResult(BaseModel):
    """
    Output of validate_pipeline() — the deterministic gate that
    combines evidence from check_data_leakage(), check_target_imbalance(),
    check_constant_features(), check_high_cardinality(), and (where
    required by the locked guardrail rules) model/evaluation evidence,
    into ONE valid/invalid decision.

    Locked distinction, carried forward from every individual
    guardrail: `valid=True` means "no implemented guardrail violation
    was detected" — it is NOT a proof that the pipeline is free of
    every possible problem. This is stated explicitly in
    `scope_note` and must never be softened into an overclaim, exactly
    as check_data_leakage()'s own report already does for itself.

    `violations` are severity="error" findings that make valid=False.
    `warnings` are severity="warning" findings that do NOT make
    valid=False on their own (e.g. moderate imbalance, or a single
    constant feature that doesn't block training) but are still
    surfaced as evidence.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    target_column: str
    valid: bool
    checks: list[ValidationCheck] = Field(default_factory=list)
    violations: list[ValidationCheck] = Field(
        default_factory=list, description="Subset of `checks` where passed=False and severity='error'."
    )
    warnings: list[ValidationCheck] = Field(
        default_factory=list, description="Subset of `checks` where passed=False and severity='warning'."
    )
    scope_note: str = Field(
        default=(
            "valid=True means no implemented guardrail violation was "
            "detected by the checks run here (leakage, imbalance, "
            "constant features, high cardinality, and — where "
            "applicable — a suspiciously high evaluation metric "
            "forcing a leakage re-check). It is not a proof that the "
            "pipeline is free of every possible problem, only that the "
            "specific, implemented guardrails found no violation."
        ),
    )
