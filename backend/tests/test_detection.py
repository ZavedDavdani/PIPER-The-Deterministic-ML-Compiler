"""
Tests for detect_missing_values() and detect_outliers().
"""

from __future__ import annotations

import numpy as np

from app.agent.tools import detect_missing_values, detect_outliers
from app.storage import InMemoryDatasetStore


class TestDetectMissingValues:
    def test_raw_telco_has_zero_true_nan(self, loaded_store: InMemoryDatasetStore):
        """
        TotalCharges' 11 problem rows are blank strings, not NaN, until
        convert_column_type() runs — see test_cleaning_chain.py for
        that transition.
        """
        result = detect_missing_values("dataset_001", loaded_store)
        assert result.data.total_missing == 0

    def test_detects_injected_missing_values(
        self, store: InMemoryDatasetStore, telco_df
    ):
        dirty = telco_df.copy()
        dirty.loc[dirty.sample(11, random_state=42).index, "MonthlyCharges"] = np.nan
        dirty.loc[dirty.sample(5, random_state=7).index, "gender"] = np.nan
        store.save("dataset_002", dirty)

        result = detect_missing_values("dataset_002", store)
        assert result.data.total_missing == 16
        columns = {e.column: e.count for e in result.data.columns_with_missing}
        assert columns["MonthlyCharges"] == 11
        assert columns["gender"] == 5

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = detect_missing_values("nonexistent", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"


class TestDetectOutliers:
    def test_clean_numeric_columns_have_no_false_positives(
        self, loaded_store: InMemoryDatasetStore
    ):
        result = detect_outliers("dataset_001", loaded_store)
        assert result.data.columns == []

    def test_senior_citizen_degenerate_iqr_is_skipped(
        self, loaded_store: InMemoryDatasetStore
    ):
        """SeniorCitizen is binary (0/1); IQR == 0, so it must be
        skipped rather than falsely flagging every row as an outlier."""
        result = detect_outliers("dataset_001", loaded_store)
        flagged = [e.column for e in result.data.columns]
        assert "SeniorCitizen" not in flagged

    def test_injected_extreme_value_is_detected(
        self, store: InMemoryDatasetStore, telco_df
    ):
        dirty = telco_df.copy()
        dirty.loc[0, "MonthlyCharges"] = 5000.0
        store.save("dataset_003", dirty)

        result = detect_outliers("dataset_003", store)
        mc_entry = next(e for e in result.data.columns if e.column == "MonthlyCharges")
        assert mc_entry.outlier_count >= 1
        assert mc_entry.method == "IQR"

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = detect_outliers("nonexistent", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"
