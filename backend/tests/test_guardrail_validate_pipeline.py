"""
Formal tests for validate_pipeline() — the deterministic gate that
combines evidence from all four guardrails plus the locked
suspicious-metric rule into one valid/invalid decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent.tools import convert_column_type, drop_column, impute_missing_values
from app.agent.tools.guardrails import validate_pipeline
from app.storage import InMemoryDatasetStore


class TestHappyPath:
    def test_cleaned_telco_is_valid(self, telco_df: pd.DataFrame):
        store = InMemoryDatasetStore()
        store.save("d1", telco_df)
        drop_column("d1", "customerID", "identifier", store, target_column="Churn")
        convert_column_type("d1", "TotalCharges", "numeric", store)
        impute_missing_values("d1", "TotalCharges", "median", store)

        result = validate_pipeline("d1", "Churn", store)

        assert result.data.valid is True
        assert result.data.violations == []

    def test_all_four_guardrails_are_run(self, telco_df: pd.DataFrame):
        store = InMemoryDatasetStore()
        store.save("d1", telco_df)
        drop_column("d1", "customerID", "identifier", store, target_column="Churn")
        convert_column_type("d1", "TotalCharges", "numeric", store)
        impute_missing_values("d1", "TotalCharges", "median", store)

        result = validate_pipeline("d1", "Churn", store)

        check_names = {c.check for c in result.data.checks}
        assert check_names == {
            "data_leakage", "target_imbalance", "constant_features", "high_cardinality",
        }


class TestLeakageMakesInvalid:
    def test_raw_telco_with_customer_id_is_invalid(self, telco_df: pd.DataFrame):
        store = InMemoryDatasetStore()
        store.save("d2", telco_df)  # uncleaned — customerID still present

        result = validate_pipeline("d2", "Churn", store)

        assert result.data.valid is False
        assert any(v.check == "data_leakage" for v in result.data.violations)

    def test_leakage_violation_has_error_severity(self, telco_df: pd.DataFrame):
        store = InMemoryDatasetStore()
        store.save("d2", telco_df)
        result = validate_pipeline("d2", "Churn", store)
        leakage_check = next(c for c in result.data.checks if c.check == "data_leakage")
        assert leakage_check.severity == "error"


class TestImbalanceIsWarningNotError:
    """Locked: severe imbalance means F1/ROC-AUC are REQUIRED, not that
    the pipeline is invalid outright."""

    def test_severe_imbalance_does_not_make_invalid(self):
        df = pd.DataFrame({"x": np.random.rand(100), "target": ["Yes"] * 95 + ["No"] * 5})
        store = InMemoryDatasetStore()
        store.save("d3", df)

        result = validate_pipeline("d3", "target", store)

        assert result.data.valid is True
        assert any(w.check == "target_imbalance" for w in result.data.warnings)

    def test_severe_imbalance_warning_has_warning_severity(self):
        df = pd.DataFrame({"x": np.random.rand(100), "target": ["Yes"] * 95 + ["No"] * 5})
        store = InMemoryDatasetStore()
        store.save("d3", df)
        result = validate_pipeline("d3", "target", store)
        imbalance_check = next(c for c in result.data.checks if c.check == "target_imbalance")
        assert imbalance_check.severity == "warning"


class TestSuspiciousEvaluationMetric:
    def test_high_f1_alone_makes_invalid(self):
        df = pd.DataFrame({"x": np.random.rand(100), "target": np.random.choice(["Yes", "No"], 100)})
        store = InMemoryDatasetStore()
        store.save("d4", df)

        result = validate_pipeline("d4", "target", store, evaluation_f1=0.995)

        assert result.data.valid is False
        assert any(v.check == "suspicious_evaluation_metric" for v in result.data.violations)

    def test_normal_f1_does_not_make_invalid(self):
        df = pd.DataFrame({"x": np.random.rand(100), "target": np.random.choice(["Yes", "No"], 100)})
        store = InMemoryDatasetStore()
        store.save("d4", df)

        result = validate_pipeline("d4", "target", store, evaluation_f1=0.82)

        assert result.data.valid is True

    def test_no_metric_provided_skips_that_check_entirely(self):
        df = pd.DataFrame({"x": np.random.rand(100), "target": np.random.choice(["Yes", "No"], 100)})
        store = InMemoryDatasetStore()
        store.save("d4", df)

        result = validate_pipeline("d4", "target", store)  # no evaluation_f1/accuracy

        check_names = {c.check for c in result.data.checks}
        assert "suspicious_evaluation_metric" not in check_names

    def test_high_accuracy_also_triggers_the_check(self):
        df = pd.DataFrame({"x": np.random.rand(100), "target": np.random.choice(["Yes", "No"], 100)})
        store = InMemoryDatasetStore()
        store.save("d4", df)

        result = validate_pipeline("d4", "target", store, evaluation_accuracy=0.99)

        assert result.data.valid is False

    def test_leakage_plus_suspicious_metric_mentions_corroboration(self):
        df = pd.DataFrame({
            "leaky": ["Yes"] * 50 + ["No"] * 50,
            "target": ["Yes"] * 50 + ["No"] * 50,
        })
        store = InMemoryDatasetStore()
        store.save("d6", df)

        result = validate_pipeline("d6", "target", store, evaluation_f1=0.999)

        metric_violation = next(v for v in result.data.violations if v.check == "suspicious_evaluation_metric")
        assert "corroborat" in metric_violation.message.lower()

    def test_suspicious_metric_without_leakage_evidence_says_so(self):
        df = pd.DataFrame({"x": np.random.rand(100), "target": np.random.choice(["Yes", "No"], 100)})
        store = InMemoryDatasetStore()
        store.save("d4", df)

        result = validate_pipeline("d4", "target", store, evaluation_f1=0.995)

        metric_violation = next(v for v in result.data.violations if v.check == "suspicious_evaluation_metric")
        assert "no feature-level leakage evidence" in metric_violation.message.lower()


class TestScopeHonesty:
    def test_scope_note_present(self):
        df = pd.DataFrame({"x": np.random.rand(50), "target": np.random.choice(["Yes", "No"], 50)})
        store = InMemoryDatasetStore()
        store.save("d1", df)
        result = validate_pipeline("d1", "target", store)
        assert result.data.scope_note
        assert "not a proof" in result.data.scope_note.lower()


class TestEdgeCases:
    def test_dataset_not_found(self):
        store = InMemoryDatasetStore()
        result = validate_pipeline("nonexistent", "target", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_target_column_not_found(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        store = InMemoryDatasetStore()
        store.save("d1", df)
        result = validate_pipeline("d1", "DoesNotExist", store)
        assert result.success is False
        assert result.error.code == "column_not_found"

    def test_non_binary_target_does_not_crash_but_notes_check_could_not_run(self):
        """
        validate_pipeline() itself doesn't require a binary target
        (unlike the individual guardrails it calls) — when a
        sub-check can't run because the target isn't binary, that's
        reported as a warning-severity finding, not a crash.
        """
        df = pd.DataFrame({"x": np.random.rand(90), "target": ["A"] * 30 + ["B"] * 30 + ["C"] * 30})
        store = InMemoryDatasetStore()
        store.save("d2", df)

        result = validate_pipeline("d2", "target", store)

        assert result.success is True
        leakage_check = next(c for c in result.data.checks if c.check == "data_leakage")
        assert leakage_check.severity == "warning"
        assert "could not run" in leakage_check.message.lower()


class TestWarningAggregationIncludesSeverityWarnings:
    """
    Regression tests for two related, now-fixed bugs:

    1. validate_pipeline() previously read
       imbalance_result.data.severely_imbalanced (a boolean meaning
       only "severity == FAILURE") instead of severity itself,
       collapsing WARNING-tier imbalance into severity="info" and
       silently dropping it from PipelineValidationResult.warnings.

    2. The final aggregation filtered warnings_list on
       `not c.passed and c.severity == "warning"`, but WARNING-severity
       checks are constructed with passed=True in some branches (e.g.
       the fixed imbalance branch) — so a WARNING-severity check could
       never appear in warnings_list regardless of its severity value.
       Fixed by filtering on severity alone.
    """

    def test_five_percent_minority_full_chain(self):
        """
        The exact chain originally reported: 5% minority -> WARNING
        severity -> ValidationCheck(passed=True, severity="warning")
        -> valid stays True -> target_imbalance appears in warnings.
        """
        df = pd.DataFrame({
            "x": range(100),
            "target": ["Yes"] * 95 + ["No"] * 5,
        })
        store = InMemoryDatasetStore()
        store.save("d_5pct_chain", df)

        result = validate_pipeline("d_5pct_chain", "target", store)

        imbalance_check = next(c for c in result.data.checks if c.check == "target_imbalance")
        assert imbalance_check.passed is True
        assert imbalance_check.severity == "warning"

        assert result.data.valid is True
        warning_checks = {w.check for w in result.data.warnings}
        assert "target_imbalance" in warning_checks

    def test_ok_tier_imbalance_stays_info_not_warning(self):
        """Confirms the fix didn't overcorrect -- OK-tier imbalance
        (e.g. 30% minority) must still be severity='info', not warning."""
        df = pd.DataFrame({
            "x": range(100),
            "target": ["Yes"] * 70 + ["No"] * 30,
        })
        store = InMemoryDatasetStore()
        store.save("d_ok", df)

        result = validate_pipeline("d_ok", "target", store)

        imbalance_check = next(c for c in result.data.checks if c.check == "target_imbalance")
        assert imbalance_check.severity == "info"
        warning_checks = {w.check for w in result.data.warnings}
        assert "target_imbalance" not in warning_checks

    def test_failure_tier_imbalance_still_does_not_invalidate(self):
        """FAILURE-tier (e.g. 3% minority) still surfaces as a warning,
        not an error -- preserving the existing no-hard-fail-on-imbalance
        policy."""
        df = pd.DataFrame({
            "x": range(100),
            "target": ["Yes"] * 97 + ["No"] * 3,
        })
        store = InMemoryDatasetStore()
        store.save("d_failure_tier", df)

        result = validate_pipeline("d_failure_tier", "target", store)

        assert result.data.valid is True
        imbalance_check = next(c for c in result.data.checks if c.check == "target_imbalance")
        assert imbalance_check.severity == "warning"

    def test_constant_feature_warning_is_surfaced(self):
        """Same aggregation bug would have hidden this warning too --
        confirms the fix isn't imbalance-specific."""
        df = pd.DataFrame({
            "constant_col": [1] * 50,
            "normal_col": np.random.rand(50),
            "target": np.random.choice(["Yes", "No"], 50),
        })
        store = InMemoryDatasetStore()
        store.save("d_constant", df)

        result = validate_pipeline("d_constant", "target", store)

        warning_checks = {w.check for w in result.data.warnings}
        assert "constant_features" in warning_checks
        assert result.data.valid is True

    def test_high_cardinality_check_is_warning_severity_and_never_a_violation(self):
        """
        NOTE: intentionally does NOT assert valid is True on this
        dataset, because a >99% unique text column (id_col) also
        legitimately trips check_data_leakage()'s independent
        identifier_like_column check (severity="error") -- both
        checks share the same underlying signal (near-100% uniqueness)
        by design, so there is no natural dataset that trips
        high_cardinality without also tripping the leakage guardrail.
        This test instead verifies the specific, correct invariant:
        high_cardinality itself is always severity="warning" and never
        lands in violations, regardless of what else fires.
        """
        df = pd.DataFrame({
            "id_col": [f"id_{i}" for i in range(200)],
            "normal_col": np.random.rand(200),
            "target": np.random.choice(["Yes", "No"], 200),
        })
        store = InMemoryDatasetStore()
        store.save("d_high_card", df)

        result = validate_pipeline("d_high_card", "target", store)

        hc_check = next(c for c in result.data.checks if c.check == "high_cardinality")
        assert hc_check.severity == "warning"

        warning_checks = {w.check for w in result.data.warnings}
        violation_checks = {v.check for v in result.data.violations}
        assert "high_cardinality" in warning_checks
        assert "high_cardinality" not in violation_checks
