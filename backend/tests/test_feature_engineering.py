"""
Formal tests for encode_categorical_features(), scale_features(),
create_date_features().
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent.tools.feature_engineering import (
    create_date_features,
    encode_categorical_features,
    scale_features,
)
from app.storage import InMemoryDatasetStore


class TestEncodeCategoricalFeatures:
    def test_happy_path_generates_correct_preview(self, loaded_store: InMemoryDatasetStore):
        result = encode_categorical_features("dataset_001", ["Contract"], loaded_store)
        assert result.success is True
        assert len(result.data.generated_columns) == 3  # Month-to-month, One year, Two year

    def test_does_not_mutate_stored_dataset(self, loaded_store: InMemoryDatasetStore):
        before = loaded_store.get("dataset_001").shape
        encode_categorical_features("dataset_001", ["Contract"], loaded_store)
        after = loaded_store.get("dataset_001").shape
        assert before == after

    def test_rejects_numeric_column(self, loaded_store: InMemoryDatasetStore):
        result = encode_categorical_features("dataset_001", ["MonthlyCharges"], loaded_store)
        assert result.success is False
        assert result.error.code == "column_is_numeric"

    def test_rejects_high_cardinality_column(self, loaded_store: InMemoryDatasetStore):
        result = encode_categorical_features("dataset_001", ["customerID"], loaded_store)
        assert result.success is False
        assert result.error.code == "cardinality_too_high"

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = encode_categorical_features("nonexistent", ["x"], store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_column_not_found(self, loaded_store: InMemoryDatasetStore):
        result = encode_categorical_features("dataset_001", ["DoesNotExist"], loaded_store)
        assert result.success is False
        assert result.error.code == "column_not_found"


class TestScaleFeatures:
    def test_happy_path(self, loaded_store: InMemoryDatasetStore):
        result = scale_features(
            "dataset_001", ["MonthlyCharges", "tenure"], loaded_store
        )
        assert result.success is True
        assert result.data.scaler == "StandardScaler"

    def test_does_not_mutate_stored_dataset(self, loaded_store: InMemoryDatasetStore):
        before = loaded_store.get("dataset_001")["MonthlyCharges"].tolist()
        scale_features("dataset_001", ["MonthlyCharges"], loaded_store)
        after = loaded_store.get("dataset_001")["MonthlyCharges"].tolist()
        assert before == after

    def test_rejects_non_numeric_column(self, loaded_store: InMemoryDatasetStore):
        result = scale_features("dataset_001", ["Contract"], loaded_store)
        assert result.success is False
        assert result.error.code == "column_not_numeric"

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = scale_features("nonexistent", ["x"], store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_column_not_found(self, loaded_store: InMemoryDatasetStore):
        result = scale_features("dataset_001", ["DoesNotExist"], loaded_store)
        assert result.success is False
        assert result.error.code == "column_not_found"


class TestCreateDateFeatures:
    def test_happy_path_generates_all_four_components(
        self, store: InMemoryDatasetStore, telco_df: pd.DataFrame
    ):
        dated = telco_df.copy()
        dated["SignupDate"] = pd.date_range("2020-01-01", periods=len(dated), freq="h")
        store.save("dataset_dated", dated)

        result = create_date_features("dataset_dated", "SignupDate", store)

        assert result.success is True
        assert set(result.data.generated_features) == {"year", "month", "day", "day_of_week"}

    def test_actually_writes_new_columns_to_stored_dataset(
        self, store: InMemoryDatasetStore, telco_df: pd.DataFrame
    ):
        dated = telco_df.copy()
        dated["SignupDate"] = pd.date_range("2020-01-01", periods=len(dated), freq="h")
        store.save("dataset_dated", dated)

        create_date_features("dataset_dated", "SignupDate", store)

        updated = store.get("dataset_dated")
        assert "SignupDate_year" in updated.columns
        assert updated["SignupDate_year"].iloc[0] == 2020

    def test_rejects_unparseable_column_without_warning(
        self, loaded_store: InMemoryDatasetStore
    ):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes a test failure
            result = create_date_features("dataset_001", "Contract", loaded_store)

        assert result.success is False
        assert result.error.code == "unparseable_date_column"

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = create_date_features("nonexistent", "x", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_column_not_found(self, loaded_store: InMemoryDatasetStore):
        result = create_date_features("dataset_001", "DoesNotExist", loaded_store)
        assert result.success is False
        assert result.error.code == "column_not_found"
