"""
Formal tests for check_target_imbalance().

CONTRACT MIGRATION NOTICE:
This guardrail was originally built on a ratio-based "90/10" contract
(severely_imbalanced = majority/minority ratio > 9.0, warning-only,
no failure tier). It has been deliberately migrated to a
minority-PERCENTAGE-based, three-tier contract:

    minority_percentage >= 15.0             -> OK
    5.0 <= minority_percentage < 15.0        -> WARNING
    minority_percentage < 5.0                -> FAILURE

named via IMBALANCE_WARNING_THRESHOLD_PERCENT (15.0) and
IMBALANCE_FAILURE_THRESHOLD_PERCENT (5.0). The 5% boundary is strict:
exactly 5.0% minority is WARNING, not FAILURE (FAILURE requires
strictly below 5.0%). This is an intentional, authorized contract
change, not a bug fix — the old ratio-based tests (previously named
around "90_10") have been migrated below to equivalent boundary tests
for the NEW policy. Under the new contract, real Telco Churn data
(26.54% minority) now correctly resolves to OK, not WARNING, since
26.54% is comfortably above the 15% threshold.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent.tools.guardrails import check_target_imbalance
from app.schemas.guardrails import (
    ImbalanceSeverity,
    IMBALANCE_FAILURE_THRESHOLD_PERCENT,
    IMBALANCE_WARNING_THRESHOLD_PERCENT,
)
from app.storage import InMemoryDatasetStore


class TestKnownScenarios:
    def test_balanced_target_is_ok(self):
        df = pd.DataFrame({"x": range(200), "target": ["Yes"] * 100 + ["No"] * 100})
        store = InMemoryDatasetStore()
        store.save("balanced", df)

        result = check_target_imbalance("balanced", "target", store)

        assert result.success is True
        assert result.data.minority_percentage == 50.0
        assert result.data.severity is ImbalanceSeverity.OK

    def test_70_30_is_ok_under_new_contract(self):
        """
        Migrated from the old ratio-based contract, where 70/30
        (ratio ~2.33) was WARNING. Under the new 15%/5%
        minority-percentage contract, 30% minority is well above the
        15% threshold, so this is OK.
        """
        df = pd.DataFrame({"x": range(100), "target": ["Yes"] * 70 + ["No"] * 30})
        store = InMemoryDatasetStore()
        store.save("moderate", df)

        result = check_target_imbalance("moderate", "target", store)

        assert result.data.minority_percentage == 30.0
        assert result.data.severity is ImbalanceSeverity.OK

    def test_exactly_15_percent_minority_is_ok(self):
        """Boundary: the OK/WARNING threshold is inclusive at exactly
        15% -- minority_percentage >= 15.0 stays OK."""
        df = pd.DataFrame({"x": range(100), "target": ["Yes"] * 85 + ["No"] * 15})
        store = InMemoryDatasetStore()
        store.save("boundary_15", df)

        result = check_target_imbalance("boundary_15", "target", store)

        assert result.data.minority_percentage == 15.0
        assert result.data.severity is ImbalanceSeverity.OK
        assert result.data.warning_threshold_percent == IMBALANCE_WARNING_THRESHOLD_PERCENT

    def test_just_below_15_percent_is_warning(self):
        """One row below the 15% boundary must cross into WARNING."""
        df = pd.DataFrame({"x": range(100), "target": ["Yes"] * 86 + ["No"] * 14})
        store = InMemoryDatasetStore()
        store.save("just_below_15", df)

        result = check_target_imbalance("just_below_15", "target", store)

        assert result.data.minority_percentage == 14.0
        assert result.data.severity is ImbalanceSeverity.WARNING

    def test_exactly_5_percent_minority_is_warning(self):
        """Boundary: exactly 5% is WARNING, not FAILURE -- the failure
        threshold is strict '<', so 5.0% itself stays WARNING."""
        df = pd.DataFrame({"x": range(100), "target": ["Yes"] * 95 + ["No"] * 5})
        store = InMemoryDatasetStore()
        store.save("boundary_5", df)

        result = check_target_imbalance("boundary_5", "target", store)

        assert result.data.minority_percentage == 5.0
        assert result.data.severity is ImbalanceSeverity.WARNING
        assert result.data.failure_threshold_percent == IMBALANCE_FAILURE_THRESHOLD_PERCENT

    def test_just_below_5_percent_is_failure(self):
        df = pd.DataFrame({"x": range(100), "target": ["Yes"] * 96 + ["No"] * 4})
        store = InMemoryDatasetStore()
        store.save("just_below_5", df)

        result = check_target_imbalance("just_below_5", "target", store)

        assert result.data.minority_percentage == 4.0
        assert result.data.severity is ImbalanceSeverity.FAILURE

    def test_warning_message_mentions_warning_threshold(self):
        df = pd.DataFrame({"x": range(100), "target": ["Yes"] * 90 + ["No"] * 10})
        store = InMemoryDatasetStore()
        store.save("warning_msg", df)

        result = check_target_imbalance("warning_msg", "target", store)

        assert result.data.severity is ImbalanceSeverity.WARNING
        assert "warning threshold" in result.message.lower()

    def test_failure_message_mentions_failure_threshold(self):
        df = pd.DataFrame({"x": range(100), "target": ["Yes"] * 97 + ["No"] * 3})
        store = InMemoryDatasetStore()
        store.save("failure_msg", df)

        result = check_target_imbalance("failure_msg", "target", store)

        assert result.data.severity is ImbalanceSeverity.FAILURE
        assert "failure threshold" in result.message.lower()


class TestEdgeCases:
    def test_dataset_not_found(self):
        store = InMemoryDatasetStore()
        result = check_target_imbalance("nonexistent", "target", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_missing_target_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        store = InMemoryDatasetStore()
        store.save("d1", df)
        result = check_target_imbalance("d1", "DoesNotExist", store)
        assert result.success is False
        assert result.error.code == "column_not_found"

    def test_single_class_target_rejected(self):
        df = pd.DataFrame({"x": range(50), "target": ["Yes"] * 50})
        store = InMemoryDatasetStore()
        store.save("single_class", df)
        result = check_target_imbalance("single_class", "target", store)
        assert result.success is False
        assert result.error.code == "target_not_binary"

    def test_three_class_target_rejected(self):
        df = pd.DataFrame({"x": range(90), "target": ["A"] * 30 + ["B"] * 30 + ["C"] * 30})
        store = InMemoryDatasetStore()
        store.save("three_class", df)
        result = check_target_imbalance("three_class", "target", store)
        assert result.success is False
        assert result.error.code == "target_not_binary"

    def test_nan_target_values_excluded_from_counts(self):
        df = pd.DataFrame({
            "x": range(100),
            "target": ["Yes"] * 70 + ["No"] * 10 + [None] * 20,
        })
        store = InMemoryDatasetStore()
        store.save("nan_target", df)

        result = check_target_imbalance("nan_target", "target", store)

        assert result.success is True
        assert sum(entry.count for entry in result.data.class_counts) == 80

    def test_all_null_target_rejected(self):
        df = pd.DataFrame({"x": range(20), "target": [None] * 20})
        store = InMemoryDatasetStore()
        store.save("all_null", df)
        result = check_target_imbalance("all_null", "target", store)
        assert result.success is False
        assert result.error.code == "target_not_binary"


class TestRealTelcoData:
    def test_real_telco_churn_is_ok_under_new_contract(self, telco_df: pd.DataFrame):
        """
        Under the NEW 15%/5% contract, Telco's 26.54% minority class
        is comfortably above the 15% warning threshold, so this
        resolves to OK -- a deliberate change from the old ratio-based
        contract's WARNING result. See module docstring.
        """
        store = InMemoryDatasetStore()
        store.save("dataset_001", telco_df)

        result = check_target_imbalance("dataset_001", "Churn", store)

        assert result.success is True
        assert {entry.label: entry.count for entry in result.data.class_counts} == {"No": 5174, "Yes": 1869}
        assert result.data.minority_percentage == pytest.approx(26.5399, abs=0.01)
        assert result.data.severity is ImbalanceSeverity.OK
