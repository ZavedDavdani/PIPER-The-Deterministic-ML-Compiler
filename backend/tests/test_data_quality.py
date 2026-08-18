"""
Formal tests for validate_data_quality() (section 9). Covers every
check named in the spec plus the duplicate-column-name crash found
and fixed during development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent.tools.data_quality import validate_data_quality
from app.storage import InMemoryDatasetStore


class TestEachCheckType:
    def test_empty_dataset_zero_rows(self):
        df = pd.DataFrame({"a": pd.array([], dtype="float64"), "target": pd.array([], dtype="object")})
        store = InMemoryDatasetStore()
        store.save("d1", df)
        result = validate_data_quality("d1", "target", store)
        assert result.data.valid is False
        assert "empty_dataset" in [v.check_type for v in result.data.violations]

    def test_zero_columns(self):
        df = pd.DataFrame(index=range(5))
        store = InMemoryDatasetStore()
        store.save("d2", df)
        result = validate_data_quality("d2", "target", store)
        assert "zero_columns" in [v.check_type for v in result.data.violations]

    def test_empty_column(self):
        df = pd.DataFrame({"a": [np.nan] * 25, "target": np.random.choice(["Yes", "No"], 25)})
        store = InMemoryDatasetStore()
        store.save("d3", df)
        result = validate_data_quality("d3", "target", store)
        assert "empty_column" in [v.check_type for v in result.data.violations]

    def test_duplicate_column_names_does_not_crash(self):
        """
        Regression test: df[col] returns a DataFrame (not Series) when
        the column name is duplicated, which crashed the original
        empty-column/unsupported-type checks. Positional iteration
        (df.iloc[:, i]) fixed this.
        """
        df = pd.DataFrame(np.random.rand(25, 3), columns=["a", "b", "a"])
        df["target"] = np.random.choice(["Yes", "No"], 25)
        store = InMemoryDatasetStore()
        store.save("d4", df)
        result = validate_data_quality("d4", "target", store)  # must not raise
        assert "duplicate_column_names" in [v.check_type for v in result.data.violations]

    def test_missing_target(self):
        df = pd.DataFrame({"a": np.random.rand(25)})
        store = InMemoryDatasetStore()
        store.save("d5", df)
        result = validate_data_quality("d5", "target", store)
        assert "missing_target" in [v.check_type for v in result.data.violations]

    def test_constant_target(self):
        df = pd.DataFrame({"a": np.random.rand(25), "target": ["Yes"] * 25})
        store = InMemoryDatasetStore()
        store.save("d6", df)
        result = validate_data_quality("d6", "target", store)
        assert "constant_target" in [v.check_type for v in result.data.violations]

    def test_insufficient_samples(self):
        df = pd.DataFrame({"a": np.random.rand(10), "target": np.random.choice(["Yes", "No"], 10)})
        store = InMemoryDatasetStore()
        store.save("d7", df)
        result = validate_data_quality("d7", "target", store)
        assert "insufficient_samples" in [v.check_type for v in result.data.violations]

    def test_invalid_non_binary_target(self):
        df = pd.DataFrame({"a": np.random.rand(30), "target": np.random.choice(["A", "B", "C"], 30)})
        store = InMemoryDatasetStore()
        store.save("d8", df)
        result = validate_data_quality("d8", "target", store)
        assert "invalid_binary_target" in [v.check_type for v in result.data.violations]

    def test_unsupported_feature_type_complex_objects(self):
        df = pd.DataFrame({"a": [[1, 2]] * 25, "target": np.random.choice(["Yes", "No"], 25)})
        store = InMemoryDatasetStore()
        store.save("d9", df)
        result = validate_data_quality("d9", "target", store)
        assert "unsupported_feature_type" in [v.check_type for v in result.data.violations]


class TestEvidenceQuality:
    def test_every_violation_has_structured_evidence(self):
        df = pd.DataFrame({"a": np.random.rand(30), "target": np.random.choice(["A", "B", "C"], 30)})
        store = InMemoryDatasetStore()
        store.save("d1", df)
        result = validate_data_quality("d1", "target", store)
        for v in result.data.violations:
            assert isinstance(v.evidence, dict)
            assert len(v.evidence) > 0

    def test_multiple_violations_all_reported_not_just_first(self):
        """A dataset with several distinct problems must surface all of them."""
        df = pd.DataFrame({"a": np.random.rand(5), "target": ["Yes"] * 5})  # insufficient_samples + constant_target
        store = InMemoryDatasetStore()
        store.save("d1", df)
        result = validate_data_quality("d1", "target", store)
        check_types = {v.check_type for v in result.data.violations}
        assert "insufficient_samples" in check_types
        assert "constant_target" in check_types


class TestHappyPath:
    def test_clean_synthetic_data_is_valid(self):
        df = pd.DataFrame({
            "a": np.random.rand(50), "b": np.random.choice(["X", "Y"], 50),
            "target": np.random.choice(["Yes", "No"], 50),
        })
        store = InMemoryDatasetStore()
        store.save("d_good", df)
        result = validate_data_quality("d_good", "target", store)
        assert result.data.valid is True
        assert result.data.violations == []

    def test_real_raw_telco_is_valid(self, telco_df: pd.DataFrame):
        """
        Raw Telco has no DATA-QUALITY violations (customerID is a
        leakage-guardrail concern, not a data-quality one) — confirms
        this check doesn't overreach into territory check_data_leakage()
        already owns.
        """
        store = InMemoryDatasetStore()
        store.save("telco", telco_df)
        result = validate_data_quality("telco", "Churn", store)
        assert result.data.valid is True


class TestErrorHandling:
    def test_dataset_not_found(self):
        store = InMemoryDatasetStore()
        result = validate_data_quality("nonexistent", "target", store)
        assert result.success is False
        assert result.error.code == "dataset_not_found"
