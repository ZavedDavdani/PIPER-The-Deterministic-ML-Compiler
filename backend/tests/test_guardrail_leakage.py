"""
Formal tests for check_data_leakage().

Per the user's explicit testing standard: don't just test Telco. Use
controlled synthetic datasets where the expected result is known, plus
edge cases (missing target, non-binary target, constant feature,
non-numeric features, empty feature set, NaNs/infinite values).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent.tools import convert_column_type, drop_column, impute_missing_values
from app.agent.tools.guardrails import check_data_leakage
from app.storage import InMemoryDatasetStore


class TestKnownScenarios:
    """Each test constructs data where the correct answer is known in advance."""

    def test_normal_feature_no_violation(self):
        np.random.seed(0)
        df = pd.DataFrame({
            "normal_feature": np.random.rand(200),
            "target": np.random.choice(["Yes", "No"], 200),
        })
        store = InMemoryDatasetStore()
        store.save("d1", df)

        result = check_data_leakage("d1", "target", store)

        assert result.success is True
        assert result.data.leakage_detected is False
        assert result.data.violations == []

    def test_feature_equals_target_detects_leakage(self):
        df = pd.DataFrame({
            "leaky_copy": ["Yes"] * 100 + ["No"] * 100,
            "other": np.random.rand(200),
            "target": ["Yes"] * 100 + ["No"] * 100,
        })
        store = InMemoryDatasetStore()
        store.save("d2", df)

        result = check_data_leakage("d2", "target", store)

        assert result.data.leakage_detected is True
        violation = next(v for v in result.data.violations if v.check_type == "duplicate_of_target")
        assert violation.feature == "leaky_copy"
        assert violation.evidence["identical_row_count"] == 200

    def test_feature_approximately_equals_target_high_correlation(self):
        np.random.seed(2)
        target_binary = np.random.choice([0, 1], 300)
        near_leaky = target_binary + np.random.normal(0, 0.05, 300)
        df = pd.DataFrame({
            "near_leaky_numeric": near_leaky,
            "target": ["Yes" if t == 1 else "No" for t in target_binary],
        })
        store = InMemoryDatasetStore()
        store.save("d3", df)

        result = check_data_leakage("d3", "target", store)

        violations = [v for v in result.data.violations if v.check_type == "high_correlation"]
        assert len(violations) == 1
        assert abs(violations[0].evidence["correlation"]) > 0.95

    def test_text_identifier_column_flagged(self):
        df = pd.DataFrame({
            "user_id": [f"id_{i}" for i in range(200)],
            "normal_num": np.random.rand(200),
            "target": np.random.choice(["Yes", "No"], 200),
        })
        store = InMemoryDatasetStore()
        store.save("d4", df)

        result = check_data_leakage("d4", "target", store)

        violations = [v for v in result.data.violations if v.check_type == "identifier_like_column"]
        assert len(violations) == 1
        assert violations[0].feature == "user_id"
        assert violations[0].evidence["unique_percentage"] == 100.0

    def test_continuous_numeric_feature_not_flagged_as_identifier(self):
        """
        A continuous numeric feature (e.g. a price) is EXPECTED to be
        highly/fully unique — this must NOT trigger the identifier
        check, which is specifically about non-numeric/label-like
        columns. This is a real bug that was caught during development.
        """
        np.random.seed(0)
        df = pd.DataFrame({
            "continuous_price": np.random.rand(200) * 1000,
            "target": np.random.choice(["Yes", "No"], 200),
        })
        store = InMemoryDatasetStore()
        store.save("d_continuous", df)

        result = check_data_leakage("d_continuous", "target", store)

        identifier_violations = [
            v for v in result.data.violations if v.check_type == "identifier_like_column"
        ]
        assert identifier_violations == []

    def test_legitimate_modest_correlation_not_flagged(self):
        np.random.seed(3)
        target_binary = np.random.choice([0, 1], 300)
        mildly_correlated = target_binary * 0.5 + np.random.normal(0, 1, 300)
        df = pd.DataFrame({
            "mildly_correlated": mildly_correlated,
            "target": ["Yes" if t == 1 else "No" for t in target_binary],
        })
        store = InMemoryDatasetStore()
        store.save("d5", df)

        result = check_data_leakage("d5", "target", store)

        assert result.data.leakage_detected is False

    def test_categorical_near_perfect_association_detected(self):
        df = pd.DataFrame({
            "leaky_cat": ["A"] * 100 + ["B"] * 100,
            "target": ["Yes"] * 100 + ["No"] * 100,
        })
        store = InMemoryDatasetStore()
        store.save("d6", df)

        result = check_data_leakage("d6", "target", store)

        violations = [
            v for v in result.data.violations
            if v.check_type == "categorical_near_perfect_association"
        ]
        assert len(violations) == 1
        assert violations[0].evidence["min_category_purity"] == 1.0


class TestEdgeCases:
    def test_dataset_not_found(self):
        store = InMemoryDatasetStore()
        result = check_data_leakage("nonexistent", "target", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_missing_target_column(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        store = InMemoryDatasetStore()
        store.save("d1", df)
        result = check_data_leakage("d1", "DoesNotExist", store)
        assert result.success is False
        assert result.error.code == "column_not_found"

    def test_non_binary_target_rejected(self):
        df = pd.DataFrame({
            "x": np.random.rand(90),
            "target": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        })
        store = InMemoryDatasetStore()
        store.save("d2", df)
        result = check_data_leakage("d2", "target", store)
        assert result.success is False
        assert result.error.code == "target_not_binary"

    def test_constant_feature_does_not_crash(self):
        df = pd.DataFrame({
            "constant_col": [5] * 200,
            "target": np.random.choice(["Yes", "No"], 200),
        })
        store = InMemoryDatasetStore()
        store.save("d3", df)
        result = check_data_leakage("d3", "target", store)
        assert result.success is True

    def test_mixed_numeric_and_non_numeric_features(self):
        df = pd.DataFrame({
            "num_col": np.random.rand(200),
            "text_col": np.random.choice(["red", "blue", "green"], 200),
            "target": np.random.choice(["Yes", "No"], 200),
        })
        store = InMemoryDatasetStore()
        store.save("d4", df)
        result = check_data_leakage("d4", "target", store)
        assert result.success is True
        assert set(result.data.features_checked) == {"num_col", "text_col"}

    def test_empty_feature_set_rejected(self):
        df = pd.DataFrame({"target": np.random.choice(["Yes", "No"], 50)})
        store = InMemoryDatasetStore()
        store.save("d5", df)
        result = check_data_leakage("d5", "target", store)
        assert result.success is False
        assert result.error.code == "empty_feature_set"

    def test_nans_in_numeric_feature_do_not_crash(self):
        df = pd.DataFrame({
            "num_with_nans": [1.0, 2.0, np.nan, 4.0, np.nan] * 40,
            "target": np.random.choice(["Yes", "No"], 200),
        })
        store = InMemoryDatasetStore()
        store.save("d6", df)
        result = check_data_leakage("d6", "target", store)
        assert result.success is True

    def test_infinite_values_do_not_crash_or_warn(self):
        """
        Regression test for a real bug: infinite values previously
        caused a noisy RuntimeWarning from numpy's internal
        correlation reduction. Fixed by explicitly filtering
        non-finite values before computing correlation.
        """
        import warnings

        df = pd.DataFrame({
            "num_with_inf": [1.0, 2.0, np.inf, 4.0, -np.inf] * 40,
            "target": np.random.choice(["Yes", "No"], 200),
        })
        store = InMemoryDatasetStore()
        store.save("d7", df)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = check_data_leakage("d7", "target", store)

        assert result.success is True

    def test_nan_in_target_column_does_not_crash(self):
        df = pd.DataFrame({
            "num_col": np.random.rand(200),
            "target": (["Yes"] * 90 + ["No"] * 90 + [None] * 20),
        })
        store = InMemoryDatasetStore()
        store.save("d8", df)
        result = check_data_leakage("d8", "target", store)
        assert result.success is True


class TestScopeHonesty:
    """
    The result must always be explicit about what it does and does not
    prove — never overclaim "no leakage" when it only means "no
    feature-level indicators found by these specific checks."
    """

    def test_scope_note_present_and_non_empty(self):
        df = pd.DataFrame({"x": np.random.rand(50), "target": np.random.choice(["Yes", "No"], 50)})
        store = InMemoryDatasetStore()
        store.save("d1", df)
        result = check_data_leakage("d1", "target", store)
        assert result.data.scope_note
        assert "feature-level" in result.data.scope_note.lower()

    def test_message_distinguishes_detected_vs_not_detected(self):
        store = InMemoryDatasetStore()

        clean_df = pd.DataFrame({"x": np.random.rand(50), "target": np.random.choice(["Yes", "No"], 50)})
        store.save("clean", clean_df)
        clean_result = check_data_leakage("clean", "target", store)
        assert "no leakage indicators detected" in clean_result.message.lower()

        leaky_df = pd.DataFrame({"x": ["Yes"] * 50 + ["No"] * 50, "target": ["Yes"] * 50 + ["No"] * 50})
        store.save("leaky", leaky_df)
        leaky_result = check_data_leakage("leaky", "target", store)
        assert "detected" in leaky_result.message.lower()
        assert "indicator" in leaky_result.message.lower()


class TestRealTelcoData:
    """Sanity check against real data, in addition to synthetic scenarios."""

    def test_raw_telco_flags_customer_id(self, telco_df: pd.DataFrame):
        store = InMemoryDatasetStore()
        store.save("dataset_001", telco_df)

        result = check_data_leakage("dataset_001", "Churn", store)

        assert result.data.leakage_detected is True
        id_violations = [v for v in result.data.violations if v.feature == "customerID"]
        assert len(id_violations) == 1
        assert id_violations[0].check_type == "identifier_like_column"

    def test_cleaned_telco_has_no_violations(self, telco_df: pd.DataFrame):
        store = InMemoryDatasetStore()
        store.save("dataset_001", telco_df)
        drop_column("dataset_001", "customerID", "identifier", store, target_column="Churn")
        convert_column_type("dataset_001", "TotalCharges", "numeric", store)
        impute_missing_values("dataset_001", "TotalCharges", "median", store)

        result = check_data_leakage("dataset_001", "Churn", store)

        assert result.data.leakage_detected is False
