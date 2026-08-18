"""
Formal tests for check_constant_features().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent.tools.guardrails import check_constant_features
from app.storage import InMemoryDatasetStore


class TestKnownScenarios:
    def test_one_constant_feature_detected(self):
        df = pd.DataFrame({
            "const": [5] * 100,
            "normal": np.random.rand(100),
            "target": np.random.choice(["Yes", "No"], 100),
        })
        store = InMemoryDatasetStore()
        store.save("d1", df)

        result = check_constant_features("d1", store, target_column="target")

        assert len(result.data.constant_columns) == 1
        assert result.data.constant_columns[0].column == "const"
        assert result.data.constant_columns[0].constant_value == "5"

    def test_multiple_constant_features_detected(self):
        df = pd.DataFrame({
            "const1": [1] * 100,
            "const2": ["A"] * 100,
            "normal": np.random.rand(100),
        })
        store = InMemoryDatasetStore()
        store.save("d2", df)

        result = check_constant_features("d2", store)

        assert {e.column for e in result.data.constant_columns} == {"const1", "const2"}

    def test_normal_features_not_flagged(self):
        df = pd.DataFrame({
            "a": np.random.rand(100),
            "b": np.random.choice(["x", "y", "z"], 100),
        })
        store = InMemoryDatasetStore()
        store.save("d3", df)

        result = check_constant_features("d3", store)

        assert result.data.constant_columns == []

    def test_binary_non_constant_feature_not_flagged(self):
        df = pd.DataFrame({"binary_feature": [0, 1] * 50, "other": np.random.rand(100)})
        store = InMemoryDatasetStore()
        store.save("d4", df)

        result = check_constant_features("d4", store)

        assert result.data.constant_columns == []

    def test_all_features_constant(self):
        df = pd.DataFrame({"c1": [1] * 50, "c2": ["x"] * 50, "c3": [True] * 50})
        store = InMemoryDatasetStore()
        store.save("d5", df)

        result = check_constant_features("d5", store)

        assert len(result.data.constant_columns) == 3

    def test_entirely_null_column_flagged_as_nan_only(self):
        df = pd.DataFrame({"all_nan": [np.nan] * 50, "normal": np.random.rand(50)})
        store = InMemoryDatasetStore()
        store.save("d7", df)

        result = check_constant_features("d7", store)

        nan_entry = next(e for e in result.data.constant_columns if e.column == "all_nan")
        assert nan_entry.constant_value == "NaN-only"
        assert nan_entry.non_null_count == 0


class TestTargetExclusion:
    def test_target_flagged_if_target_column_not_passed(self):
        """
        If the caller omits target_column, a constant target IS
        flagged like any other column — this is correct: the
        exclusion is opt-in, not automatic guessing.
        """
        df = pd.DataFrame({"x": np.random.rand(50), "target": ["Yes"] * 50})
        store = InMemoryDatasetStore()
        store.save("d6", df)

        result = check_constant_features("d6", store)

        assert "target" in [e.column for e in result.data.constant_columns]

    def test_target_excluded_when_target_column_passed(self):
        df = pd.DataFrame({"x": np.random.rand(50), "target": ["Yes"] * 50})
        store = InMemoryDatasetStore()
        store.save("d6", df)

        result = check_constant_features("d6", store, target_column="target")

        assert "target" not in [e.column for e in result.data.constant_columns]


class TestEdgeCases:
    def test_dataset_not_found(self):
        store = InMemoryDatasetStore()
        result = check_constant_features("nonexistent", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"

    def test_empty_feature_set_when_only_target_present(self):
        df = pd.DataFrame({"target": np.random.choice(["Yes", "No"], 50)})
        store = InMemoryDatasetStore()
        store.save("d8", df)
        result = check_constant_features("d8", store, target_column="target")
        assert result.success is False
        assert result.error.code == "empty_feature_set"


class TestRealTelcoData:
    def test_real_telco_has_no_constant_columns(self, telco_df: pd.DataFrame):
        store = InMemoryDatasetStore()
        store.save("dataset_001", telco_df)

        result = check_constant_features("dataset_001", store, target_column="Churn")

        assert result.data.constant_columns == []
