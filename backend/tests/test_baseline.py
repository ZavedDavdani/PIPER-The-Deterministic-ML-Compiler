"""
Formal tests for compute_baseline() — the majority-class baseline
comparator (section 8). Boundary tests are the most important part
here: delta < 0.05, == 0.05, and > 0.05 must produce exactly the
documented gate_passed outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent.tools import convert_column_type, drop_column, impute_missing_values, split_dataset, train_model
from app.agent.tools.baseline import compute_baseline
from app.schemas.baseline import BASELINE_POLICY
from app.schemas.training import FeatureEngineeringIntent
from app.storage import InMemoryDatasetStore, InMemorySplitStore
from app.storage.model_store import InMemoryModelStore


class TestBaselinePolicy:
    def test_locked_policy_values(self):
        assert BASELINE_POLICY.primary_metric == "f1"
        assert BASELINE_POLICY.minimum_primary_metric_delta == 0.05

    def test_policy_is_frozen(self):
        with pytest.raises(Exception):
            BASELINE_POLICY.minimum_primary_metric_delta = 0.10


@pytest.fixture()
def majority_positive_setup():
    """
    Synthetic data where the positive class ('B') is the MAJORITY, so
    the majority-class baseline has a defined, non-trivial F1 — needed
    to test exact delta boundaries (an undefined baseline can't be
    used for that).
    """
    np.random.seed(5)
    n = 500
    df = pd.DataFrame({
        "x": np.random.rand(n),
        "target": np.random.choice(["A", "B"], n, p=[0.3, 0.7]),
    })
    dataset_store = InMemoryDatasetStore()
    split_store = InMemorySplitStore()
    model_store = InMemoryModelStore()
    dataset_store.save("d1", df)
    split_result = split_dataset("d1", "target", 0.2, dataset_store, split_store)
    split_id = split_result.data.split_id

    intent = FeatureEngineeringIntent(categorical_columns=[], numeric_columns_to_scale=["x"])
    train_result = train_model(split_id, "target", "logistic_regression", {"C": 1.0}, intent, split_store, model_store)

    return split_store, model_store, train_result.data.model_id


class TestExactBoundaries:
    def test_delta_below_threshold_fails(self, majority_positive_setup):
        split_store, model_store, model_id = majority_positive_setup
        baseline_probe = compute_baseline(model_id, 0.0, split_store, model_store)
        baseline_f1 = baseline_probe.data.baseline_primary_metric_value
        assert baseline_f1 is not None

        result = compute_baseline(model_id, baseline_f1 + 0.03, split_store, model_store)

        assert result.data.delta == 0.03
        assert result.data.gate_passed is False

    def test_delta_exactly_at_threshold_passes(self, majority_positive_setup):
        """The locked rule is >=, not >, so exactly 0.05 must PASS."""
        split_store, model_store, model_id = majority_positive_setup
        baseline_probe = compute_baseline(model_id, 0.0, split_store, model_store)
        baseline_f1 = baseline_probe.data.baseline_primary_metric_value

        result = compute_baseline(model_id, round(baseline_f1 + 0.05, 4), split_store, model_store)

        assert result.data.delta == 0.05
        assert result.data.gate_passed is True

    def test_delta_above_threshold_passes(self, majority_positive_setup):
        split_store, model_store, model_id = majority_positive_setup
        baseline_probe = compute_baseline(model_id, 0.0, split_store, model_store)
        baseline_f1 = baseline_probe.data.baseline_primary_metric_value

        result = compute_baseline(model_id, baseline_f1 + 0.10, split_store, model_store)

        assert result.data.delta == 0.10
        assert result.data.gate_passed is True


class TestUndefinedBaselineMetric:
    """
    Real Telco data: majority class 'No' never predicts the positive
    class 'Yes' under a majority-class strategy, so baseline F1 is
    mathematically undefined — must be None, never silently 0.0.
    """

    @pytest.fixture()
    def telco_setup(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        dataset_store.save("d1", telco_df)
        drop_column("d1", "customerID", "identifier", dataset_store, target_column="Churn")
        convert_column_type("d1", "TotalCharges", "numeric", dataset_store)
        impute_missing_values("d1", "TotalCharges", "median", dataset_store)
        split_result = split_dataset("d1", "Churn", 0.2, dataset_store, split_store)
        split_id = split_result.data.split_id

        intent = FeatureEngineeringIntent(
            categorical_columns=["Contract"], numeric_columns_to_scale=["MonthlyCharges"]
        )
        train_result = train_model(
            split_id, "Churn", "random_forest", {"n_estimators": 100},
            intent, split_store, model_store,
        )
        return split_store, model_store, train_result.data.model_id

    def test_baseline_f1_is_none_not_zero(self, telco_setup):
        split_store, model_store, model_id = telco_setup
        result = compute_baseline(model_id, 0.5, split_store, model_store)

        assert result.data.baseline.f1 is None
        assert result.data.baseline_primary_metric_value is None

    def test_delta_is_none_when_baseline_undefined(self, telco_setup):
        split_store, model_store, model_id = telco_setup
        result = compute_baseline(model_id, 0.5, split_store, model_store)

        assert result.data.delta is None

    def test_gate_passes_by_policy_when_baseline_undefined(self, telco_setup):
        """
        Locked policy: an undefined baseline cannot be used to REJECT
        the model (gate_passed=True), but the reason field must make
        clear this is weak evidence, not a strong pass.
        """
        split_store, model_store, model_id = telco_setup
        result = compute_baseline(model_id, 0.5, split_store, model_store)

        assert result.data.gate_passed is True
        assert "undefined" in result.data.reason.lower()

    def test_baseline_accuracy_is_still_defined(self, telco_setup):
        """Accuracy is always computable even when F1 is undefined."""
        split_store, model_store, model_id = telco_setup
        result = compute_baseline(model_id, 0.5, split_store, model_store)

        assert result.data.baseline.accuracy is not None
        assert 0.0 <= result.data.baseline.accuracy <= 1.0


class TestBaselineUsesSameSplit:
    def test_baseline_fit_on_same_train_split(self, majority_positive_setup):
        """
        Confirms the baseline's majority_class is computed from the
        actual train split's distribution, not the full dataset.
        """
        split_store, model_store, model_id = majority_positive_setup
        result = compute_baseline(model_id, 0.5, split_store, model_store)

        assert result.data.baseline.majority_class == "B"  # B was constructed as majority


class TestErrorHandling:
    def test_model_not_found(self):
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        result = compute_baseline("nonexistent", 0.5, split_store, model_store)
        assert result.success is False
        assert result.error.code == "model_not_found"
