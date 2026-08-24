"""
Formal tests for train_model() and InMemoryModelStore.

TestLeakageProof is the acceptance test this milestone stands or falls
on: it proves the stronger invariant requested — "no preprocessing
object is fitted using test data" — via a synthetic unseen-category
scenario, not by comparing learned categories against what the test
set happens to contain (which can pass by coincidence).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.agent.tools import (
    convert_column_type,
    drop_column,
    impute_missing_values,
    split_dataset,
    train_model,
)
from app.schemas.training import FeatureEngineeringIntent
from app.storage import InMemoryDatasetStore, InMemorySplitStore
from app.storage.model_store import InMemoryModelStore, ModelNotFoundError


@pytest.fixture()
def prepared_split(telco_df: pd.DataFrame):
    """Real Telco data, cleaned and split — ready for training."""
    dataset_store = InMemoryDatasetStore()
    split_store = InMemorySplitStore()
    dataset_store.save("dataset_001", telco_df)
    drop_column("dataset_001", "customerID", "identifier", dataset_store, target_column="Churn")
    convert_column_type("dataset_001", "TotalCharges", "numeric", dataset_store)
    impute_missing_values("dataset_001", "TotalCharges", "median", dataset_store)
    split_result = split_dataset("dataset_001", "Churn", 0.2, dataset_store, split_store)
    return split_store, split_result.data.split_id


@pytest.fixture()
def model_store() -> InMemoryModelStore:
    return InMemoryModelStore()


@pytest.fixture()
def good_intent() -> FeatureEngineeringIntent:
    return FeatureEngineeringIntent(
        categorical_columns=["Contract", "PaymentMethod", "InternetService"],
        numeric_columns_to_scale=["MonthlyCharges", "TotalCharges", "tenure"],
    )


class TestHappyPath:
    def test_training_succeeds(self, prepared_split, model_store, good_intent):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "random_forest",
            {"n_estimators": 200, "max_depth": 15},
            good_intent, split_store, model_store,
        )
        assert result.success is True

    def test_model_store_holds_a_real_fitted_pipeline(
        self, prepared_split, model_store, good_intent
    ):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "random_forest", {"n_estimators": 200},
            good_intent, split_store, model_store,
        )
        artifact = model_store.get(result.data.model_id)
        assert isinstance(artifact.pipeline, Pipeline)
        assert [name for name, _ in artifact.pipeline.steps] == ["preprocessor", "classifier"]

    def test_pipeline_predicts_directly_on_raw_unencoded_test_data(
        self, prepared_split, model_store, good_intent
    ):
        """
        The core acceptance property: raw X_test -> Pipeline transforms
        internally -> predictions. No manual preprocessing required by
        the caller.
        """
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "random_forest", {"n_estimators": 200},
            good_intent, split_store, model_store,
        )
        artifact = model_store.get(result.data.model_id)

        _, test_df = split_store.get(split_id)
        feature_cols = good_intent.categorical_columns + good_intent.numeric_columns_to_scale
        X_test_raw = test_df[feature_cols]

        # Confirm the data really is raw/unencoded before predicting.
        assert X_test_raw["Contract"].dtype == object or X_test_raw["Contract"].dtype.name == "str"

        predictions = artifact.pipeline.predict(X_test_raw)
        assert len(predictions) == len(test_df)

    def test_no_deprecation_warnings_from_logistic_regression(
        self, prepared_split, model_store, good_intent
    ):
        split_store, split_id = prepared_split
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", message=r".*disp.*iprint.*", category=DeprecationWarning)
            result = train_model(
                split_id, "Churn", "logistic_regression", {"C": 1.0, "max_iter": 1000},
                good_intent, split_store, model_store,
            )
        assert result.success is True


class TestLeakageProof:
    """
    The strongest version of the leakage test: constructs data where a
    category is structurally guaranteed to exist ONLY in the test
    split, then proves (a) the fitted encoder never learned it, (b)
    prediction still succeeds via handle_unknown="ignore", and (c) no
    re-fitting occurs during prediction.
    """

    @pytest.fixture()
    def synthetic_split_with_unseen_test_category(self):
        np.random.seed(1)
        n = 200
        synthetic = pd.DataFrame({
            "color": ["Red"] * 100 + ["Blue"] * 99 + ["Purple"] * 1,
            "value": np.random.rand(n) * 100,
            "target": ([0] * 50 + [1] * 50) * 2,
        })

        purple_row = synthetic[synthetic["color"] == "Purple"]
        non_purple = synthetic[synthetic["color"] != "Purple"]
        train_df = non_purple.iloc[:150]
        test_df = pd.concat([non_purple.iloc[150:], purple_row])

        assert "Purple" not in train_df["color"].values
        assert "Purple" in test_df["color"].values

        split_store = InMemorySplitStore()
        split_store.save("split_leak_test", train_df, test_df)
        return split_store, train_df, test_df

    def test_fitted_encoder_never_learned_the_unseen_category(
        self, synthetic_split_with_unseen_test_category, model_store
    ):
        split_store, train_df, test_df = synthetic_split_with_unseen_test_category
        intent = FeatureEngineeringIntent(categorical_columns=["color"], numeric_columns_to_scale=["value"])

        result = train_model(
            "split_leak_test", "target", "logistic_regression", {"C": 1.0},
            intent, split_store, model_store,
        )
        assert result.success is True

        artifact = model_store.get(result.data.model_id)
        fitted_encoder = artifact.pipeline.named_steps["preprocessor"].named_transformers_["categorical"]
        learned_categories = fitted_encoder.categories_[0].tolist()

        assert "Purple" not in learned_categories
        assert set(learned_categories) == {"Blue", "Red"}

    def test_prediction_succeeds_on_unseen_category_via_handle_unknown_ignore(
        self, synthetic_split_with_unseen_test_category, model_store
    ):
        split_store, train_df, test_df = synthetic_split_with_unseen_test_category
        intent = FeatureEngineeringIntent(categorical_columns=["color"], numeric_columns_to_scale=["value"])

        result = train_model(
            "split_leak_test", "target", "logistic_regression", {"C": 1.0},
            intent, split_store, model_store,
        )
        artifact = model_store.get(result.data.model_id)

        X_test = test_df[["color", "value"]]
        predictions = artifact.pipeline.predict(X_test)  # must not raise

        assert len(predictions) == len(test_df)

    def test_no_refitting_occurs_during_prediction(
        self, synthetic_split_with_unseen_test_category, model_store
    ):
        split_store, train_df, test_df = synthetic_split_with_unseen_test_category
        intent = FeatureEngineeringIntent(categorical_columns=["color"], numeric_columns_to_scale=["value"])

        result = train_model(
            "split_leak_test", "target", "logistic_regression", {"C": 1.0},
            intent, split_store, model_store,
        )
        artifact = model_store.get(result.data.model_id)
        X_test = test_df[["color", "value"]]

        encoder = artifact.pipeline.named_steps["preprocessor"].named_transformers_["categorical"]
        categories_before = encoder.categories_[0].tolist()

        artifact.pipeline.predict(X_test)
        artifact.pipeline.predict(X_test)  # call twice to be sure

        categories_after = (
            artifact.pipeline.named_steps["preprocessor"]
            .named_transformers_["categorical"]
            .categories_[0]
            .tolist()
        )
        assert categories_before == categories_after


class TestRejectionPaths:
    def test_unknown_algorithm(self, prepared_split, model_store, good_intent):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "gradient_boosting_xtreme", {},
            good_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "unknown_algorithm"

    def test_disallowed_hyperparameter_blocks_arbitrary_estimator_injection(
        self, prepared_split, model_store, good_intent
    ):
        """
        The specific attack scenario: the LLM must never be able to
        smuggle an arbitrary estimator/class through the params dict.
        """
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "random_forest", {"estimator": "some_arbitrary_class"},
            good_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "disallowed_hyperparameter"

    def test_hyperparameter_out_of_bounds_random_forest(
        self, prepared_split, model_store, good_intent
    ):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "random_forest", {"n_estimators": 999999},
            good_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "hyperparameter_out_of_bounds"

    def test_hyperparameter_out_of_bounds_logistic_regression(
        self, prepared_split, model_store, good_intent
    ):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "logistic_regression", {"C": 99999},
            good_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "hyperparameter_out_of_bounds"

    def test_max_depth_none_is_explicitly_valid(self, prepared_split, model_store, good_intent):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "random_forest", {"max_depth": None},
            good_intent, split_store, model_store,
        )
        assert result.success is True

    def test_missing_split_id(self, prepared_split, model_store, good_intent):
        split_store, _ = prepared_split
        result = train_model(
            "nonexistent_split", "Churn", "random_forest", {},
            good_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "split_not_found"

    def test_invalid_target_column(self, prepared_split, model_store, good_intent):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "DoesNotExist", "random_forest", {},
            good_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "invalid_target"

    def test_target_column_as_feature_is_rejected_as_leakage(
        self, prepared_split, model_store
    ):
        split_store, split_id = prepared_split
        bad_intent = FeatureEngineeringIntent(categorical_columns=["Churn"], numeric_columns_to_scale=[])
        result = train_model(
            split_id, "Churn", "random_forest", {},
            bad_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "target_leakage_in_features"

    def test_empty_feature_set_rejected(self, prepared_split, model_store):
        split_store, split_id = prepared_split
        empty_intent = FeatureEngineeringIntent(categorical_columns=[], numeric_columns_to_scale=[])
        result = train_model(
            split_id, "Churn", "random_forest", {},
            empty_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "empty_feature_set"

    def test_nonexistent_feature_column_rejected(self, prepared_split, model_store):
        split_store, split_id = prepared_split
        bad_intent = FeatureEngineeringIntent(categorical_columns=["NotARealColumn"], numeric_columns_to_scale=[])
        result = train_model(
            split_id, "Churn", "random_forest", {},
            bad_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "feature_column_not_found"

    def test_column_listed_as_both_categorical_and_numeric_rejected(
        self, prepared_split, model_store
    ):
        split_store, split_id = prepared_split
        ambiguous_intent = FeatureEngineeringIntent(
            categorical_columns=["MonthlyCharges"], numeric_columns_to_scale=["MonthlyCharges"]
        )
        result = train_model(
            split_id, "Churn", "random_forest", {},
            ambiguous_intent, split_store, model_store,
        )
        assert result.success is False
        assert result.error.code == "ambiguous_feature_type"


class TestInMemoryModelStore:
    def test_save_and_get_roundtrip(self, prepared_split, model_store, good_intent):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "random_forest", {"n_estimators": 50},
            good_intent, split_store, model_store,
        )
        artifact = model_store.get(result.data.model_id)
        assert artifact.metadata.algorithm == "random_forest"
        assert artifact.metadata.split_id == split_id

    def test_get_missing_model_raises(self, model_store):
        with pytest.raises(ModelNotFoundError):
            model_store.get("nonexistent")

    def test_delete_removes_model(self, prepared_split, model_store, good_intent):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "random_forest", {"n_estimators": 50},
            good_intent, split_store, model_store,
        )
        model_store.delete(result.data.model_id)
        assert model_store.exists(result.data.model_id) is False

    def test_metadata_records_reproducibility_fields(
        self, prepared_split, model_store, good_intent
    ):
        split_store, split_id = prepared_split
        result = train_model(
            split_id, "Churn", "random_forest", {"n_estimators": 200, "max_depth": 15},
            good_intent, split_store, model_store,
        )
        artifact = model_store.get(result.data.model_id)
        assert artifact.metadata.random_state == 42
        assert artifact.metadata.parameters == {"n_estimators": 200, "max_depth": 15}
        assert artifact.metadata.feature_columns == (
            good_intent.categorical_columns + good_intent.numeric_columns_to_scale
        )
