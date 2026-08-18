"""
Tests for drop_column() and drop_duplicates().
"""

from __future__ import annotations

from app.agent.tools import drop_column, drop_duplicates
from app.storage import InMemoryDatasetStore


class TestDropColumn:
    def test_drops_customer_id_successfully(self, loaded_store: InMemoryDatasetStore):
        result = drop_column(
            "dataset_001", "customerID", "identifier, >99% unique",
            loaded_store, target_column="Churn",
        )
        assert result.success is True
        assert result.data.columns_before == 21
        assert result.data.columns_after == 20

        current = loaded_store.get("dataset_001")
        assert "customerID" not in current.columns

    def test_target_column_is_protected(self, loaded_store: InMemoryDatasetStore):
        result = drop_column(
            "dataset_001", "Churn", "trying to drop target",
            loaded_store, target_column="Churn",
        )
        assert result.success is False
        assert result.error.code == "target_column_protected"

    def test_column_not_found(self, loaded_store: InMemoryDatasetStore):
        result = drop_column(
            "dataset_001", "DoesNotExist", "test",
            loaded_store, target_column="Churn",
        )
        assert result.success is False
        assert result.error.code == "column_not_found"

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = drop_column("nonexistent", "gender", "test", store, target_column="Churn")
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_final_remaining_feature_is_protected(self, store: InMemoryDatasetStore):
        import pandas as pd

        single_col_df = pd.DataFrame({"only_column": [1, 2, 3]})
        store.save("single_col_dataset", single_col_df)

        result = drop_column(
            "single_col_dataset", "only_column", "test",
            store, target_column=None,
        )
        assert result.success is False
        assert result.error.code == "final_feature_protected"


class TestDropDuplicates:
    def test_is_idempotent(self, loaded_store: InMemoryDatasetStore):
        first = drop_duplicates("dataset_001", loaded_store)
        second = drop_duplicates("dataset_001", loaded_store)

        assert second.data.duplicates_found == 0
        assert second.data.duplicates_removed == 0
        # First call's counts should match each other (found == removed).
        assert first.data.duplicates_found == first.data.duplicates_removed

    def test_dropping_identifier_reveals_duplicates(self, loaded_store: InMemoryDatasetStore):
        """
        Dropping customerID (the only thing making every row unique)
        should surface genuine duplicate rows that were previously
        hidden by the identifier.
        """
        drop_column(
            "dataset_001", "customerID", "identifier",
            loaded_store, target_column="Churn",
        )
        result = drop_duplicates("dataset_001", loaded_store)
        assert result.data.duplicates_found > 0

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = drop_duplicates("nonexistent", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"
