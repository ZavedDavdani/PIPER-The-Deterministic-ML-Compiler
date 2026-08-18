"""
Formal tests for split_dataset().
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent.tools import (
    convert_column_type,
    drop_column,
    impute_missing_values,
    split_dataset,
)
from app.storage import InMemoryDatasetStore, InMemorySplitStore, SplitNotFoundError


@pytest.fixture()
def cleaned_telco_store(telco_df: pd.DataFrame):
    """Telco data run through the M1 cleaning chain, as a realistic pre-split state."""
    store = InMemoryDatasetStore()
    store.save("dataset_001", telco_df)
    drop_column("dataset_001", "customerID", "identifier", store, target_column="Churn")
    convert_column_type("dataset_001", "TotalCharges", "numeric", store)
    impute_missing_values("dataset_001", "TotalCharges", "median", store)
    return store


class TestSplitDataset:
    def test_happy_path_row_counts(self, cleaned_telco_store: InMemoryDatasetStore):
        split_store = InMemorySplitStore()
        result = split_dataset("dataset_001", "Churn", 0.2, cleaned_telco_store, split_store)

        assert result.success is True
        assert result.data.train_rows + result.data.test_rows == 7043
        assert result.data.stratified is True
        assert result.data.random_state == 42

    def test_split_is_reproducible(self, cleaned_telco_store: InMemoryDatasetStore):
        split_store = InMemorySplitStore()
        r1 = split_dataset("dataset_001", "Churn", 0.2, cleaned_telco_store, split_store)
        r2 = split_dataset("dataset_001", "Churn", 0.2, cleaned_telco_store, split_store)

        train1, _ = split_store.get(r1.data.split_id)
        train2, _ = split_store.get(r2.data.split_id)

        assert list(train1.index) == list(train2.index)

    def test_stratification_preserves_class_balance(
        self, cleaned_telco_store: InMemoryDatasetStore
    ):
        split_store = InMemorySplitStore()
        result = split_dataset("dataset_001", "Churn", 0.2, cleaned_telco_store, split_store)
        train_df, test_df = split_store.get(result.data.split_id)

        original_df = cleaned_telco_store.get("dataset_001")
        original_rate = original_df["Churn"].value_counts(normalize=True)["Yes"]
        train_rate = train_df["Churn"].value_counts(normalize=True)["Yes"]
        test_rate = test_df["Churn"].value_counts(normalize=True)["Yes"]

        assert abs(original_rate - train_rate) < 0.02
        assert abs(original_rate - test_rate) < 0.02

    def test_dataset_not_found(self):
        store = InMemoryDatasetStore()
        split_store = InMemorySplitStore()
        result = split_dataset("nonexistent", "Churn", 0.2, store, split_store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_column_not_found(self, cleaned_telco_store: InMemoryDatasetStore):
        split_store = InMemorySplitStore()
        result = split_dataset(
            "dataset_001", "DoesNotExist", 0.2, cleaned_telco_store, split_store
        )
        assert result.success is False
        assert result.error.code == "column_not_found"

    def test_invalid_test_size_rejected(self, cleaned_telco_store: InMemoryDatasetStore):
        split_store = InMemorySplitStore()
        result = split_dataset("dataset_001", "Churn", 1.5, cleaned_telco_store, split_store)
        assert result.success is False
        assert result.error.code == "invalid_test_size"

    def test_non_binary_target_rejected(self, cleaned_telco_store: InMemoryDatasetStore):
        split_store = InMemorySplitStore()
        result = split_dataset(
            "dataset_001", "Contract", 0.2, cleaned_telco_store, split_store
        )
        assert result.success is False
        assert result.error.code == "target_not_binary"


class TestInMemorySplitStore:
    def test_save_and_get_roundtrip(self):
        store = InMemorySplitStore()
        train = pd.DataFrame({"a": [1, 2, 3]})
        test = pd.DataFrame({"a": [4, 5]})
        store.save("split_001", train, test)

        got_train, got_test = store.get("split_001")
        assert got_train.shape == (3, 1)
        assert got_test.shape == (2, 1)

    def test_copy_isolation(self):
        store = InMemorySplitStore()
        train = pd.DataFrame({"a": [1, 2, 3]})
        test = pd.DataFrame({"a": [4, 5]})
        store.save("split_001", train, test)

        got_train, _ = store.get("split_001")
        got_train.drop(columns=["a"], inplace=True)

        still_stored_train, _ = store.get("split_001")
        assert still_stored_train.shape == (3, 1)

    def test_get_missing_split_raises(self):
        store = InMemorySplitStore()
        with pytest.raises(SplitNotFoundError):
            store.get("nonexistent")
