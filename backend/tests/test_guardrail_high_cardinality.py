"""
Formal tests for check_high_cardinality().

TestAvoidsContinuousNumericBug is the most important class here: it
directly guards against repeating the exact bug found and fixed in
check_data_leakage() — flagging a continuous numeric feature as an
identifier just because it's highly unique.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent.tools.guardrails import check_high_cardinality
from app.storage import InMemoryDatasetStore


class TestKnownScenarios:
    def test_customer_id_style_column_flagged(self):
        df = pd.DataFrame({
            "customerID": [f"id_{i}" for i in range(200)],
            "other": np.random.rand(200),
        })
        store = InMemoryDatasetStore()
        store.save("d1", df)

        result = check_high_cardinality("d1", store)

        assert len(result.data.suspicious_columns) == 1
        assert result.data.suspicious_columns[0].column == "customerID"
        assert result.data.suspicious_columns[0].unique_percentage == 100.0

    def test_unique_text_column_flagged(self):
        df = pd.DataFrame({
            "random_text": [f"val_{i}_{np.random.rand()}" for i in range(200)],
        })
        store = InMemoryDatasetStore()
        store.save("d2", df)

        result = check_high_cardinality("d2", store)

        assert len(result.data.suspicious_columns) == 1

    def test_low_cardinality_categorical_not_flagged(self):
        df = pd.DataFrame({"category": np.random.choice(["A", "B", "C"], 200)})
        store = InMemoryDatasetStore()
        store.save("d3", df)

        result = check_high_cardinality("d3", store)

        assert result.data.suspicious_columns == []


class TestAvoidsContinuousNumericBug:
    """
    Regression coverage for the exact bug class found in
    check_data_leakage(): a continuous numeric feature being highly or
    fully unique is EXPECTED and must never be flagged as an
    identifier.
    """

    def test_continuous_numeric_feature_never_flagged(self):
        df = pd.DataFrame({"continuous_price": np.random.rand(200) * 1000})
        store = InMemoryDatasetStore()
        store.save("d4", df)

        result = check_high_cardinality("d4", store)

        assert result.data.suspicious_columns == []

    def test_sequential_integer_column_never_flagged(self):
        """
        An integer column that happens to be 100% unique (e.g. a
        sequential index) is still numeric and must be excluded.
        """
        df = pd.DataFrame({"sequential_int": range(200)})
        store = InMemoryDatasetStore()
        store.save("d5", df)

        result = check_high_cardinality("d5", store)

        assert result.data.suspicious_columns == []

    def test_mixed_dataset_flags_only_the_genuine_identifier(self):
        df = pd.DataFrame({
            "id_col": [f"id_{i}" for i in range(200)],
            "num_col": np.random.rand(200),
            "cat_col": np.random.choice(["X", "Y"], 200),
        })
        store = InMemoryDatasetStore()
        store.save("d6", df)

        result = check_high_cardinality("d6", store)

        flagged = [e.column for e in result.data.suspicious_columns]
        assert flagged == ["id_col"]

    def test_scope_note_explains_numeric_exclusion(self):
        df = pd.DataFrame({"x": np.random.rand(50)})
        store = InMemoryDatasetStore()
        store.save("d1", df)
        result = check_high_cardinality("d1", store)
        assert "numeric" in result.data.scope_note.lower()


class TestEdgeCases:
    def test_dataset_not_found(self):
        store = InMemoryDatasetStore()
        result = check_high_cardinality("nonexistent", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_empty_feature_set_when_only_target_present(self):
        df = pd.DataFrame({"target": np.random.choice(["Yes", "No"], 50)})
        store = InMemoryDatasetStore()
        store.save("d7", df)
        result = check_high_cardinality("d7", store, target_column="target")
        assert result.success is False
        assert result.error.code == "empty_feature_set"

    def test_target_excluded_when_target_column_passed(self):
        df = pd.DataFrame({
            "id_col": [f"id_{i}" for i in range(50)],
            "target": [f"t_{i}" for i in range(50)],  # also 100% unique
        })
        store = InMemoryDatasetStore()
        store.save("d8", df)
        result = check_high_cardinality("d8", store, target_column="target")
        flagged = [e.column for e in result.data.suspicious_columns]
        assert "target" not in flagged
        assert "id_col" in flagged


class TestRealTelcoData:
    def test_real_telco_flags_only_customer_id(self, telco_df: pd.DataFrame):
        store = InMemoryDatasetStore()
        store.save("dataset_001", telco_df)

        result = check_high_cardinality("dataset_001", store, target_column="Churn")

        assert len(result.data.suspicious_columns) == 1
        assert result.data.suspicious_columns[0].column == "customerID"
