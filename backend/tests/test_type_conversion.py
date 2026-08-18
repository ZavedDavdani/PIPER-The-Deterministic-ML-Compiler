"""
Unit tests for impute_missing_values() and convert_column_type() in
isolation. The full end-to-end TotalCharges chain lives in its own
file, test_cleaning_chain.py, since that's the milestone's primary
acceptance test and deserves to be easy to find on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.agent.tools import convert_column_type, impute_missing_values
from app.storage import InMemoryDatasetStore


class TestImputeMissingValues:
    def test_mode_imputation_on_categorical(
        self, store: InMemoryDatasetStore, telco_df: pd.DataFrame
    ):
        dirty = telco_df.copy()
        dirty.loc[dirty.sample(5, random_state=2).index, "gender"] = np.nan
        store.save("dataset_cat", dirty)

        result = impute_missing_values("dataset_cat", "gender", "mode", store)
        assert result.success is True
        assert result.data.missing_before == 5
        assert result.data.missing_after == 0

    def test_mean_median_rejected_on_categorical_column(
        self, store: InMemoryDatasetStore, telco_df: pd.DataFrame
    ):
        dirty = telco_df.copy()
        dirty.loc[dirty.sample(5, random_state=2).index, "gender"] = np.nan
        store.save("dataset_cat", dirty)

        result = impute_missing_values("dataset_cat", "gender", "mean", store)
        assert result.success is False
        assert result.error.code == "unsupported_dtype_strategy_combination"

    def test_median_imputation_on_numeric_column(
        self, store: InMemoryDatasetStore, telco_df: pd.DataFrame
    ):
        dirty = telco_df.copy()
        dirty.loc[dirty.sample(11, random_state=42).index, "MonthlyCharges"] = np.nan
        store.save("dataset_num", dirty)

        result = impute_missing_values("dataset_num", "MonthlyCharges", "median", store)
        assert result.success is True
        assert result.data.missing_after == 0

    def test_rejects_column_with_no_missing_values(self, loaded_store: InMemoryDatasetStore):
        result = impute_missing_values("dataset_001", "MonthlyCharges", "median", loaded_store)
        assert result.success is False
        assert result.error.code == "no_missing_values"

    def test_rejects_unsupported_strategy(
        self, store: InMemoryDatasetStore, telco_df: pd.DataFrame
    ):
        dirty = telco_df.copy()
        dirty.loc[0, "MonthlyCharges"] = np.nan
        store.save("dataset_bad_strategy", dirty)

        result = impute_missing_values(
            "dataset_bad_strategy", "MonthlyCharges", "guess_randomly", store
        )
        assert result.success is False
        assert result.error.code == "unsupported_strategy"

    def test_column_not_found(self, loaded_store: InMemoryDatasetStore):
        result = impute_missing_values("dataset_001", "DoesNotExist", "median", loaded_store)
        assert result.success is False
        assert result.error.code == "column_not_found"

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = impute_missing_values("nonexistent", "gender", "mode", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"


class TestConvertColumnType:
    def test_failure_threshold_rejects_conversion(
        self, store: InMemoryDatasetStore, telco_df: pd.DataFrame
    ):
        """
        Corrupting ~30% of a text column with garbage should exceed
        the 10% default threshold, and the tool must leave the
        dataset unchanged rather than silently applying a mostly-empty
        conversion.
        """
        badly_dirty = telco_df.copy()
        n = len(badly_dirty)
        badly_dirty.loc[
            badly_dirty.sample(int(n * 0.3), random_state=1).index, "gender"
        ] = "garbage_text"
        store.save("dataset_bad", badly_dirty)

        result = convert_column_type("dataset_bad", "gender", "numeric", store)
        assert result.data.applied is False

        unchanged = store.get("dataset_bad")
        assert not pd.api.types.is_numeric_dtype(unchanged["gender"])

    def test_column_not_found(self, loaded_store: InMemoryDatasetStore):
        result = convert_column_type("dataset_001", "DoesNotExist", "numeric", loaded_store)
        assert result.success is False
        assert result.error.code == "column_not_found"

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = convert_column_type("nonexistent", "TotalCharges", "numeric", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_unsupported_target_type_rejected(self, loaded_store: InMemoryDatasetStore):
        result = convert_column_type(
            "dataset_001", "TotalCharges", "not_a_real_type", loaded_store
        )
        assert result.success is False
        assert result.error.code == "unsupported_target_type"
