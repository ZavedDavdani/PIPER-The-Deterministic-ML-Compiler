"""
Tests for profile_dataset() and inspect_column() against the real
Telco CSV — asserting the specific values our locked contract was
designed around (customerID identifier, TotalCharges text, etc.), not
just "does it run."
"""

from __future__ import annotations

from app.agent.tools import inspect_column, profile_dataset
from app.storage import InMemoryDatasetStore


class TestProfileDataset:
    def test_success_matches_telco_shape(self, loaded_store: InMemoryDatasetStore):
        result = profile_dataset("dataset_001", loaded_store)
        assert result.success is True
        assert result.data.rows == 7043
        assert result.data.columns == 21
        assert result.data.duplicate_rows == 0

    def test_customer_id_is_100_percent_unique(self, loaded_store: InMemoryDatasetStore):
        result = profile_dataset("dataset_001", loaded_store)
        cid = next(c for c in result.data.column_profiles if c.name == "customerID")
        assert cid.unique_percentage == 100.0

    def test_total_charges_has_no_numeric_stats_because_its_text(
        self, loaded_store: InMemoryDatasetStore
    ):
        result = profile_dataset("dataset_001", loaded_store)
        tc = next(c for c in result.data.column_profiles if c.name == "TotalCharges")
        assert tc.min is None
        assert tc.max is None

    def test_monthly_charges_has_numeric_stats(self, loaded_store: InMemoryDatasetStore):
        result = profile_dataset("dataset_001", loaded_store)
        mc = next(c for c in result.data.column_profiles if c.name == "MonthlyCharges")
        assert mc.min is not None
        assert mc.max is not None
        assert mc.min <= mc.mean <= mc.max

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = profile_dataset("nonexistent", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"


class TestInspectColumn:
    def test_customer_id_is_identifier(self, loaded_store: InMemoryDatasetStore):
        result = inspect_column("dataset_001", "customerID", loaded_store)
        assert result.data.semantic_type == "identifier"

    def test_contract_is_categorical(self, loaded_store: InMemoryDatasetStore):
        result = inspect_column("dataset_001", "Contract", loaded_store)
        assert result.data.semantic_type == "categorical"
        assert len(result.data.top_values) > 0

    def test_monthly_charges_is_numeric(self, loaded_store: InMemoryDatasetStore):
        result = inspect_column("dataset_001", "MonthlyCharges", loaded_store)
        assert result.data.semantic_type == "numeric"
        assert "mean" in result.data.statistics

    def test_churn_distribution_matches_known_values(self, loaded_store: InMemoryDatasetStore):
        result = inspect_column("dataset_001", "Churn", loaded_store)
        counts = {v["value"]: v["count"] for v in result.data.top_values}
        assert counts["No"] == 5174
        assert counts["Yes"] == 1869

    def test_column_not_found(self, loaded_store: InMemoryDatasetStore):
        result = inspect_column("dataset_001", "DoesNotExist", loaded_store)
        assert result.success is False
        assert result.error.code == "column_not_found"

    def test_dataset_not_found(self, store: InMemoryDatasetStore):
        result = inspect_column("nonexistent", "Churn", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"
