"""
Formal tests for evaluate_model() and compare_models().

TestNeverFitsDuringEvaluation is the key test: it proves, behaviorally
(via monkeypatching fit() to raise), that evaluate_model() never fits
anything — not just that the code looks like it doesn't.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from app.agent.tools import (
    compare_models,
    convert_column_type,
    drop_column,
    evaluate_model,
    impute_missing_values,
    split_dataset,
    train_model,
)
from app.schemas.training import FeatureEngineeringIntent
from app.storage import InMemoryDatasetStore, InMemorySplitStore
from app.storage.model_store import InMemoryModelStore


@pytest.fixture()
def trained_setup(telco_df: pd.DataFrame):
    """Real Telco data: cleaned, split, and one model trained."""
    dataset_store = InMemoryDatasetStore()
    split_store = InMemorySplitStore()
    model_store = InMemoryModelStore()

    dataset_store.save("dataset_001", telco_df)
    drop_column("dataset_001", "customerID", "identifier", dataset_store, target_column="Churn")
    convert_column_type("dataset_001", "TotalCharges", "numeric", dataset_store)
    impute_missing_values("dataset_001", "TotalCharges", "median", dataset_store)
    split_result = split_dataset("dataset_001", "Churn", 0.2, dataset_store, split_store)
    split_id = split_result.data.split_id

    intent = FeatureEngineeringIntent(
        categorical_columns=["Contract", "PaymentMethod"],
        numeric_columns_to_scale=["MonthlyCharges", "TotalCharges", "tenure"],
    )
    rf_result = train_model(
        split_id, "Churn", "random_forest", {"n_estimators": 200, "max_depth": 15},
        intent, split_store, model_store,
    )
    lr_result = train_model(
        split_id, "Churn", "logistic_regression", {"C": 1.0, "max_iter": 1000},
        intent, split_store, model_store,
    )

    return {
        "split_store": split_store,
        "model_store": model_store,
        "split_id": split_id,
        "rf_model_id": rf_result.data.model_id,
        "lr_model_id": lr_result.data.model_id,
    }


class TestEvaluateModelHappyPath:
    def test_returns_plausible_metrics_on_real_data(self, trained_setup):
        result = evaluate_model(
            trained_setup["rf_model_id"], trained_setup["split_store"], trained_setup["model_store"]
        )
        assert result.success is True
        # Real churn prediction on this dataset should land in a
        # plausible range — not 0, not suspiciously perfect.
        assert 0.5 < result.data.accuracy < 1.0
        assert 0.0 < result.data.f1 < 1.0
        assert 0.5 < result.data.roc_auc < 1.0

    def test_test_rows_matches_split(self, trained_setup):
        result = evaluate_model(
            trained_setup["rf_model_id"], trained_setup["split_store"], trained_setup["model_store"]
        )
        assert result.data.test_rows == 1409  # 20% of 7043, matches M1's own split test

    def test_confusion_matrix_sums_to_test_rows(self, trained_setup):
        result = evaluate_model(
            trained_setup["rf_model_id"], trained_setup["split_store"], trained_setup["model_store"]
        )
        cm = result.data.confusion_matrix
        assert cm.tn + cm.fp + cm.fn + cm.tp == result.data.test_rows

    def test_model_not_found(self, trained_setup):
        result = evaluate_model(
            "nonexistent_model", trained_setup["split_store"], trained_setup["model_store"]
        )
        assert result.success is False
        assert result.error.code == "model_not_found"


class TestNeverFitsDuringEvaluation:
    """
    The behavioral proof: patch fit() on the classifier AND on a
    transformer inside the ColumnTransformer to raise if called, then
    run evaluate_model() and confirm it completes successfully without
    ever triggering either patch.
    """

    def test_evaluate_model_never_calls_fit_on_classifier_or_transformers(
        self, trained_setup
    ):
        model_id = trained_setup["rf_model_id"]
        model_store = trained_setup["model_store"]
        split_store = trained_setup["split_store"]

        artifact = model_store.get(model_id)
        classifier_type = type(artifact.pipeline.named_steps["classifier"])
        preprocessor = artifact.pipeline.named_steps["preprocessor"]
        first_transformer_type = type(preprocessor.transformers_[0][1])

        def _raise_if_called(*args, **kwargs):
            raise AssertionError(
                "fit() was called during evaluate_model() — this would mean "
                "the model or a transformer was refit, a leakage risk."
            )

        with patch.object(classifier_type, "fit", side_effect=_raise_if_called), \
             patch.object(first_transformer_type, "fit", side_effect=_raise_if_called):
            result = evaluate_model(model_id, split_store, model_store)

        assert result.success is True

    def test_running_evaluation_twice_produces_identical_metrics(self, trained_setup):
        """
        A secondary signal of no hidden fitting: evaluating the same
        model twice must produce byte-identical metrics, since nothing
        about the fitted pipeline should change between calls.
        """
        model_id = trained_setup["rf_model_id"]
        split_store = trained_setup["split_store"]
        model_store = trained_setup["model_store"]

        first = evaluate_model(model_id, split_store, model_store)
        second = evaluate_model(model_id, split_store, model_store)

        assert first.data.f1 == second.data.f1
        assert first.data.accuracy == second.data.accuracy
        assert first.data.confusion_matrix == second.data.confusion_matrix


class TestCompareModels:
    def test_recommends_highest_f1_model(self, trained_setup):
        result = compare_models(
            [trained_setup["rf_model_id"], trained_setup["lr_model_id"]],
            trained_setup["split_store"], trained_setup["model_store"],
        )
        assert result.success is True

        best = max(result.data.models, key=lambda m: m.f1)
        assert result.data.recommended_model_id == best.model_id

    def test_selection_metric_is_always_f1(self, trained_setup):
        """
        Locked: selection_metric is fixed at "f1", never left to the
        LLM to choose a metric that favors a preferred model.
        """
        result = compare_models(
            [trained_setup["rf_model_id"], trained_setup["lr_model_id"]],
            trained_setup["split_store"], trained_setup["model_store"],
        )
        assert result.data.selection_metric == "f1"

    def test_empty_model_list_rejected(self, trained_setup):
        result = compare_models([], trained_setup["split_store"], trained_setup["model_store"])
        assert result.success is False
        assert result.error.code == "empty_model_list"

    def test_nonexistent_model_in_list_fails_cleanly(self, trained_setup):
        result = compare_models(
            [trained_setup["rf_model_id"], "nonexistent"],
            trained_setup["split_store"], trained_setup["model_store"],
        )
        assert result.success is False

    def test_all_models_present_in_comparison(self, trained_setup):
        result = compare_models(
            [trained_setup["rf_model_id"], trained_setup["lr_model_id"]],
            trained_setup["split_store"], trained_setup["model_store"],
        )
        model_ids_in_result = {m.model_id for m in result.data.models}
        assert model_ids_in_result == {trained_setup["rf_model_id"], trained_setup["lr_model_id"]}


class TestModelSelectionJustification:
    """
    Pre-6A Polish (Model-Selection Transparency): compare_models() now
    also returns a deterministic, template-based justification for
    recommended_model_id, derived only from the real F1 scores already
    in `models` — never LLM-generated.
    """

    def test_justification_names_the_winner_and_the_runner_up(self, trained_setup):
        result = compare_models(
            [trained_setup["rf_model_id"], trained_setup["lr_model_id"]],
            trained_setup["split_store"], trained_setup["model_store"],
        )
        winner = max(result.data.models, key=lambda m: m.f1)
        loser = min(result.data.models, key=lambda m: m.f1)

        assert result.data.justification.startswith(winner.algorithm)
        assert str(winner.f1) in result.data.justification
        assert str(loser.f1) in result.data.justification
        assert loser.algorithm in result.data.justification

    def test_justification_is_a_pure_function_of_the_metrics_not_llm_generated(self, trained_setup):
        """
        Calling compare_models() twice against the identical inputs
        must produce byte-identical justification text — proves it's a
        deterministic function of the already-computed metrics, never
        LLM-generated (which would not be guaranteed reproducible).
        """
        first = compare_models(
            [trained_setup["rf_model_id"], trained_setup["lr_model_id"]],
            trained_setup["split_store"], trained_setup["model_store"],
        )
        second = compare_models(
            [trained_setup["rf_model_id"], trained_setup["lr_model_id"]],
            trained_setup["split_store"], trained_setup["model_store"],
        )
        assert first.data.justification == second.data.justification

    def test_single_candidate_justification_notes_no_alternative(self, trained_setup):
        result = compare_models(
            [trained_setup["rf_model_id"]],
            trained_setup["split_store"], trained_setup["model_store"],
        )
        assert "only candidate" in result.data.justification
        assert result.data.justification.startswith(result.data.models[0].algorithm)
