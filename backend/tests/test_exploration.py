"""
PIPER Learn — Learn-Explore tests (Batch 6B).

explore_alternative() (app/agent/tools/exploration.py) is a thin,
deterministic orchestration wrapper around the SAME train_model()/
evaluate_model()/compare_models() the real graph already uses — no new
training logic. These tests prove the locked constraints:
1. Exactly one variable (model OR one hyperparameter) may change per
   exploration.
2. The exploration reuses the SAME split as the original run.
3. Exploration results are isolated (their own experiment_id
   namespace) and NEVER modify the original run's RunStore record,
   AgentState-derived state, or ModelStore entries.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.tools.exploration import explore_alternative
from app.agent.tracing import stream_with_tracing
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemoryRunStore, InMemorySplitStore
from tests.conftest import heuristic_llm_provider


@pytest.fixture()
def completed_run(telco_df: pd.DataFrame):
    """
    A genuinely completed real run (both V1 candidates trained,
    evaluated, compared, validated) — the exact fixture shape
    exploration needs: a terminal RunRecord plus the SAME split_store/
    model_store instances the run actually used (mirrors what the API
    layer's shared app.state singletons provide in production).
    """
    dataset_store = InMemoryDatasetStore()
    dataset_store.save("dataset_001", telco_df)
    split_store = InMemorySplitStore()
    model_store = InMemoryModelStore()
    graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
    run_store = InMemoryRunStore()
    initial = AgentState(run_id="explore_base_001", dataset_id="dataset_001", target_column="Churn")

    stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

    record = run_store.get("explore_base_001")
    assert record.status == "completed"
    assert len(record.final_state.model_results) == 2  # both V1 candidates

    return {
        "run_store": run_store,
        "split_store": split_store,
        "model_store": model_store,
        "record": record,
    }


def _model_ids_by_algorithm(record) -> dict:
    return {m.algorithm: m.model_id for m in record.final_state.model_results}


def _run_model_ids(record) -> list:
    return [m.model_id for m in record.final_state.model_results]


class TestExactlyOneVariableEnforcement:
    def test_both_new_algorithm_and_hyperparameter_rejected(self, completed_run):
        record = completed_run["record"]
        base_model_id = record.final_state.model_results[0].model_id

        result = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="logistic_regression", hyperparameter_name="n_estimators", hyperparameter_value=100,
        )

        assert result.success is False
        assert result.error.code == "more_than_one_variable_changed"

    def test_neither_variable_provided_rejected(self, completed_run):
        record = completed_run["record"]
        base_model_id = record.final_state.model_results[0].model_id

        result = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
        )

        assert result.success is False
        assert result.error.code == "no_variable_changed"

    def test_hyperparameter_name_without_value_rejected(self, completed_run):
        record = completed_run["record"]
        base_model_id = record.final_state.model_results[0].model_id

        result = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
            hyperparameter_name="n_estimators",
        )

        assert result.success is False
        assert result.error.code == "incomplete_hyperparameter_request"


class TestModelSwapExploration:
    def test_produces_a_new_model_with_the_alternative_algorithm(self, completed_run):
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        base_model_id = by_algo["random_forest"]

        result = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="logistic_regression",
        )

        assert result.success is True
        assert result.data.training.algorithm == "logistic_regression"
        assert result.data.training.model_id != base_model_id
        assert result.data.variable_changed.kind == "model"
        assert result.data.variable_changed.name == "algorithm"
        assert result.data.variable_changed.old_value == "random_forest"
        assert result.data.variable_changed.new_value == "logistic_regression"

    def test_swapping_to_the_same_algorithm_is_rejected(self, completed_run):
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        base_model_id = by_algo["random_forest"]

        result = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="random_forest",
        )

        assert result.success is False
        assert result.error.code == "not_an_alternative"

    def test_unknown_algorithm_rejected(self, completed_run):
        record = completed_run["record"]
        base_model_id = record.final_state.model_results[0].model_id

        result = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="gradient_boosted_llm",
        )

        assert result.success is False
        assert result.error.code == "unknown_algorithm"


class TestHyperparameterExploration:
    def test_changes_exactly_one_hyperparameter_others_preserved(self, completed_run):
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        rf_model_id = by_algo["random_forest"]
        rf_result = next(m for m in record.final_state.model_results if m.model_id == rf_model_id)

        result = explore_alternative(
            record.run_id, _run_model_ids(record), rf_model_id,
            completed_run["split_store"], completed_run["model_store"],
            hyperparameter_name="n_estimators", hyperparameter_value=333,
        )

        assert result.success is True
        assert result.data.training.parameters["n_estimators"] == 333
        # max_depth was set on the base model (see _TRAIN_CANDIDATES) — must survive unchanged.
        assert result.data.training.parameters.get("max_depth") == rf_result.parameters.get("max_depth")
        assert result.data.variable_changed.kind == "hyperparameter"
        assert result.data.variable_changed.name == "n_estimators"
        assert result.data.variable_changed.old_value == str(rf_result.parameters["n_estimators"])
        assert result.data.variable_changed.new_value == "333"

    def test_disallowed_hyperparameter_for_the_algorithm_rejected(self, completed_run):
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        rf_model_id = by_algo["random_forest"]

        result = explore_alternative(
            record.run_id, _run_model_ids(record), rf_model_id,
            completed_run["split_store"], completed_run["model_store"],
            hyperparameter_name="C", hyperparameter_value=1.0,  # C is a logistic_regression param, not random_forest
        )

        assert result.success is False
        assert result.error.code == "disallowed_hyperparameter"

    def test_out_of_bounds_hyperparameter_value_rejected(self, completed_run):
        """train_model()'s own bounds check is reused unchanged — no
        duplicated allowlist/bounds logic in explore_alternative()."""
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        rf_model_id = by_algo["random_forest"]

        result = explore_alternative(
            record.run_id, _run_model_ids(record), rf_model_id,
            completed_run["split_store"], completed_run["model_store"],
            hyperparameter_name="n_estimators", hyperparameter_value=999999,
        )

        assert result.success is False
        assert result.error.code == "hyperparameter_out_of_bounds"


class TestBaseModelScoping:
    def test_model_id_not_from_this_run_is_rejected(self, completed_run):
        record = completed_run["record"]
        base_model_id = record.final_state.model_results[0].model_id

        result = explore_alternative(
            record.run_id, [], base_model_id,  # empty run_model_ids: nothing belongs to this run
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="logistic_regression",
        )

        assert result.success is False
        assert result.error.code == "model_not_from_this_run"

    def test_nonexistent_model_id_rejected(self, completed_run):
        record = completed_run["record"]

        result = explore_alternative(
            record.run_id, ["model_ghost"], "model_ghost",
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="logistic_regression",
        )

        assert result.success is False
        assert result.error.code == "model_not_found"


class TestSameSplitReused:
    def test_exploration_uses_the_identical_split_id(self, completed_run):
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        rf_model_id = by_algo["random_forest"]
        rf_result = next(m for m in record.final_state.model_results if m.model_id == rf_model_id)

        result = explore_alternative(
            record.run_id, _run_model_ids(record), rf_model_id,
            completed_run["split_store"], completed_run["model_store"],
            hyperparameter_name="min_samples_leaf", hyperparameter_value=5,
        )

        assert result.data.split_id == rf_result.split_id
        assert result.data.training.split_id == rf_result.split_id


class TestComparisonAndLearnExplainIntegration:
    def test_comparison_includes_both_the_base_and_new_model(self, completed_run):
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        base_model_id = by_algo["random_forest"]

        result = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="logistic_regression",
        )

        model_ids_in_comparison = {m.model_id for m in result.data.comparison_vs_base.models}
        assert model_ids_in_comparison == {base_model_id, result.data.training.model_id}

    def test_evaluation_and_comparison_explanations_are_grounded(self, completed_run):
        """Batch 6A integration: reuses explain_evaluation()/explain_model_selection() directly."""
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        base_model_id = by_algo["random_forest"]

        result = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="logistic_regression",
        )

        assert result.data.evaluation_explanation is not None
        assert result.data.evaluation_explanation.model_id == result.data.training.model_id
        by_metric = {m.metric: m.value for m in result.data.evaluation_explanation.metrics}
        assert by_metric["f1"] == result.data.evaluation.f1

        assert result.data.comparison_explanation is not None
        assert result.data.comparison_explanation.recommended_model_id == result.data.comparison_vs_base.recommended_model_id
        assert result.data.comparison_explanation.justification == result.data.comparison_vs_base.justification


class TestOriginalRunIsolation:
    def test_exploration_never_mutates_the_original_run_record(self, completed_run):
        run_store = completed_run["run_store"]
        record = completed_run["record"]
        run_id = record.run_id

        before_status = record.status
        before_attempt = record.attempt
        before_plan_history = list(record.plan_history)
        before_model_results = list(record.final_state.model_results)
        before_comparison = record.final_state.comparison
        before_validation = record.final_state.validation
        before_retry_count = record.final_state.retry_count

        by_algo = _model_ids_by_algorithm(record)
        explore_alternative(
            run_id, _run_model_ids(record), by_algo["random_forest"],
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="logistic_regression",
        )

        after_record = run_store.get(run_id)
        assert after_record.status == before_status
        assert after_record.attempt == before_attempt
        assert after_record.plan_history == before_plan_history
        assert after_record.final_state.model_results == before_model_results
        assert after_record.final_state.comparison == before_comparison
        assert after_record.final_state.validation == before_validation
        assert after_record.final_state.retry_count == before_retry_count

    def test_original_models_remain_unchanged_in_model_store(self, completed_run):
        model_store = completed_run["model_store"]
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        base_model_id = by_algo["random_forest"]
        before_metadata = model_store.get(base_model_id).metadata.model_dump()

        explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], model_store,
            hyperparameter_name="n_estimators", hyperparameter_value=250,
        )

        after_metadata = model_store.get(base_model_id).metadata.model_dump()
        assert before_metadata == after_metadata

    def test_exploration_results_are_isolated_in_their_own_experiment_namespace(self, completed_run):
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)

        result = explore_alternative(
            record.run_id, _run_model_ids(record), by_algo["random_forest"],
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="logistic_regression",
        )

        assert result.data.experiment_id.startswith("exp_")
        assert result.data.experiment_id != record.run_id
        # RunRecord itself carries no exploration-related attribute at all —
        # exploration results are never merged into the original run's own state.
        assert not hasattr(record, "explorations")
        assert not hasattr(record, "experiment_id")

    def test_two_explorations_from_the_same_base_get_distinct_isolated_experiment_ids(self, completed_run):
        record = completed_run["record"]
        by_algo = _model_ids_by_algorithm(record)
        base_model_id = by_algo["random_forest"]

        first = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
            new_algorithm="logistic_regression",
        )
        second = explore_alternative(
            record.run_id, _run_model_ids(record), base_model_id,
            completed_run["split_store"], completed_run["model_store"],
            hyperparameter_name="n_estimators", hyperparameter_value=100,
        )

        assert first.data.experiment_id != second.data.experiment_id
        assert first.data.training.model_id != second.data.training.model_id
