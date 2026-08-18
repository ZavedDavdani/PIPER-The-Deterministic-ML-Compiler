"""
Storage-layer tests: save/get/delete/exists, and critically, copy
isolation — the property that makes cleaning_log/tool_trace trustworthy
as the single source of truth for "what actually changed," rather than
silent in-place mutation via a stale reference.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.storage import DatasetNotFoundError, InMemoryDatasetStore


class TestBasicOperations:
    def test_exists_false_on_empty_store(self, store: InMemoryDatasetStore):
        assert store.exists("dataset_001") is False

    def test_save_and_get_roundtrip(self, loaded_store: InMemoryDatasetStore, telco_df):
        retrieved = loaded_store.get("dataset_001")
        assert retrieved.shape == telco_df.shape

    def test_exists_true_after_save(self, loaded_store: InMemoryDatasetStore):
        assert loaded_store.exists("dataset_001") is True

    def test_delete_removes_dataset(self, loaded_store: InMemoryDatasetStore):
        loaded_store.delete("dataset_001")
        assert loaded_store.exists("dataset_001") is False


class TestCopyIsolation:
    """
    The property that matters most for correctness: nothing outside
    the store should be able to mutate what's stored, whether through
    the return value of get() or by re-using a reference passed to
    save().
    """

    def test_mutating_retrieved_dataframe_does_not_affect_store(
        self, loaded_store: InMemoryDatasetStore
    ):
        retrieved = loaded_store.get("dataset_001")
        original_cols = retrieved.shape[1]
        retrieved.drop(columns=["customerID"], inplace=True)

        still_stored = loaded_store.get("dataset_001")
        assert still_stored.shape[1] == original_cols

    def test_save_then_external_mutation_does_not_affect_store(
        self, store: InMemoryDatasetStore, telco_df: pd.DataFrame
    ):
        store.save("dataset_001", telco_df)
        telco_df.drop(columns=["customerID"], inplace=True)

        stored = store.get("dataset_001")
        assert stored.shape[1] == 21  # unaffected by the caller's later mutation


class TestErrorHandling:
    def test_get_missing_dataset_raises(self, store: InMemoryDatasetStore):
        with pytest.raises(DatasetNotFoundError):
            store.get("does_not_exist")

    def test_delete_missing_dataset_raises(self, store: InMemoryDatasetStore):
        with pytest.raises(DatasetNotFoundError):
            store.delete("does_not_exist")


class TestResave:
    def test_resave_persists_new_state(self, loaded_store: InMemoryDatasetStore):
        cleaned = loaded_store.get("dataset_001").drop(columns=["customerID"])
        loaded_store.save("dataset_001", cleaned)

        after = loaded_store.get("dataset_001")
        assert after.shape[1] == 20


class TestListIds:
    """M5: list_ids(), added for the API layer's dataset-listing endpoint."""

    def test_empty_store_returns_empty_list(self, store: InMemoryDatasetStore):
        assert store.list_ids() == []

    def test_returns_all_saved_dataset_ids(self, store: InMemoryDatasetStore, telco_df: pd.DataFrame):
        store.save("dataset_a", telco_df)
        store.save("dataset_b", telco_df)

        assert set(store.list_ids()) == {"dataset_a", "dataset_b"}

    def test_deleted_dataset_no_longer_listed(self, loaded_store: InMemoryDatasetStore):
        assert "dataset_001" in loaded_store.list_ids()
        loaded_store.delete("dataset_001")
        assert "dataset_001" not in loaded_store.list_ids()
