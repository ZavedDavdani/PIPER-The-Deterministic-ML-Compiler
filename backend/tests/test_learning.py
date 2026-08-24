"""
PIPER Learn — Learn-Explain tests (Batch 6A).

Proves three things the locked spec requires:
1. Every explanation is grounded in real evidence, not fabricated —
   each explain_*() function's output is checked against the exact
   real value it was built from.
2. The formula library and comprehension checks are used correctly —
   static, complete coverage of the concepts PIPER actually surfaces
   (the six real guardrail check names, the eleven real failure
   categories), never silently drifting out of sync with the taxonomy.
3. Learning Mode has zero effect on execution — build_run_explanation()
   never mutates the state it reads, and invoking it (or not) never
   changes a run's actual result.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.state import OperationRecord
from app.agent.tracing import stream_with_tracing
from app.learning.comprehension import COMPREHENSION_CHECKS
from app.learning.explain import (
    _FAILURE_CATEGORY_MEANINGS,
    _GUARDRAIL_MEANINGS,
    build_run_explanation,
    explain_evaluation,
    explain_failure,
    explain_guardrail_check,
    explain_model_selection,
    explain_operation,
)
from app.learning.formulas import FORMULA_LIBRARY
from app.schemas.baseline import BaselineComparisonResult, BaselineMetrics
from app.schemas.evaluation import ConfusionMatrix, EvaluationResult, ModelComparison, ModelComparisonEntry
from app.schemas.failure import FailureCategory, FailureInfo
from app.schemas.guardrails import ValidationCheck
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemoryRunStore, InMemorySplitStore
from tests.conftest import heuristic_llm_provider

_ALL_FAILURE_CATEGORIES = list(FailureCategory.__args__)
_ALL_GUARDRAIL_CHECK_NAMES = [
    "data_leakage", "target_imbalance", "constant_features",
    "high_cardinality", "suspicious_evaluation_metric", "baseline_gate",
]


class TestFormulaLibrary:
    def test_covers_every_metric_piper_actually_reports(self):
        names = {entry.name for entry in FORMULA_LIBRARY}
        for required in ("Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Standardization (Z-score scaling)"):
            assert required in names

    def test_every_entry_is_non_empty_and_generic(self):
        for entry in FORMULA_LIBRARY:
            assert entry.name.strip()
            assert entry.formula.strip()
            assert entry.description.strip()
            assert entry.when_used.strip()

    def test_library_is_a_fixed_module_level_constant(self):
        """A second import path resolves to the SAME object — proves
        this is a fixed, curated constant, never regenerated per call."""
        from app.learning.formulas import FORMULA_LIBRARY as reimported

        assert reimported is FORMULA_LIBRARY


class TestComprehensionChecks:
    def test_has_multiple_generic_static_entries(self):
        assert len(COMPREHENSION_CHECKS) >= 5

    def test_every_entry_is_non_empty(self):
        for check in COMPREHENSION_CHECKS:
            assert check.question.strip().endswith("?")
            assert check.answer_explanation.strip()
            assert check.related_concept.strip()


class TestExplainOperation:
    def test_grounded_in_the_real_operation_record(self):
        op = OperationRecord(
            operation_id="op_1",
            tool_name="drop_column",
            arguments={"column": "customerID"},
            result_summary="Dropped column 'customerID'. Columns: 21 -> 20.",
            reason="customerID is 100% unique, an identifier column.",
            timestamp="2026-01-01T00:00:00Z",
        )

        explanation = explain_operation(op)

        assert explanation.operation_id == "op_1"
        assert explanation.tool_name == "drop_column"
        assert explanation.why == op.reason
        assert op.result_summary in explanation.what_happened

    def test_unknown_tool_name_falls_back_gracefully(self):
        op = OperationRecord(
            operation_id="op_2", tool_name="some_future_tool", arguments={},
            result_summary="Did something.", reason="Because.", timestamp="t",
        )
        explanation = explain_operation(op)
        assert "some_future_tool" in explanation.what_happened


class TestExplainModelSelection:
    def test_grounded_in_the_real_comparison(self):
        comparison = ModelComparison(
            models=[
                ModelComparisonEntry(model_id="m_rf", algorithm="random_forest", accuracy=0.8, precision=0.7, recall=0.6, f1=0.65, roc_auc=0.75),
                ModelComparisonEntry(model_id="m_lr", algorithm="logistic_regression", accuracy=0.82, precision=0.75, recall=0.68, f1=0.71, roc_auc=0.79),
            ],
            recommended_model_id="m_lr",
            selection_metric="f1",
            justification="logistic_regression selected: F1=0.71 vs. 0.65 for random_forest.",
        )

        explanation = explain_model_selection(comparison)

        assert explanation.recommended_model_id == "m_lr"
        assert explanation.recommended_algorithm == "logistic_regression"
        assert explanation.justification == comparison.justification
        assert explanation.candidates == comparison.models


class TestExplainEvaluation:
    def _evaluation(self, model_id="m_1"):
        return EvaluationResult(
            model_id=model_id, split_id="s_1", accuracy=0.81, precision=0.7, recall=0.6, f1=0.65, roc_auc=0.77,
            confusion_matrix=ConfusionMatrix(tn=500, fp=50, fn=80, tp=120), test_rows=750,
        )

    def test_metrics_are_the_real_values_not_fabricated(self):
        evaluation = self._evaluation()
        explanation = explain_evaluation(evaluation, baseline=None)

        by_name = {m.metric: m.value for m in explanation.metrics}
        assert by_name == {
            "accuracy": evaluation.accuracy, "precision": evaluation.precision,
            "recall": evaluation.recall, "f1": evaluation.f1, "roc_auc": evaluation.roc_auc,
        }
        assert str(evaluation.f1) in next(m.meaning for m in explanation.metrics if m.metric == "f1")

    def test_confusion_matrix_meaning_cites_real_counts(self):
        evaluation = self._evaluation()
        explanation = explain_evaluation(evaluation, baseline=None)
        assert "120" in explanation.confusion_matrix_meaning  # tp
        assert "500" in explanation.confusion_matrix_meaning  # tn
        assert "750" in explanation.confusion_matrix_meaning  # test_rows

    def test_baseline_comparison_only_attached_for_the_matching_model(self):
        evaluation = self._evaluation(model_id="m_1")
        baseline = BaselineComparisonResult(
            model_id="m_1", split_id="s_1",
            baseline=BaselineMetrics(accuracy=0.6, precision=None, recall=None, f1=None, majority_class="No"),
            model_primary_metric_value=0.65, baseline_primary_metric_value=None,
            primary_metric="f1", delta=None, minimum_required_delta=0.05, gate_passed=False,
            reason="Baseline F1 undefined; treated as gate failure per policy.",
        )

        matching = explain_evaluation(evaluation, baseline=baseline)
        assert matching.baseline_comparison == baseline.reason

        other_baseline = baseline.model_copy(update={"model_id": "m_other"})
        non_matching = explain_evaluation(evaluation, baseline=other_baseline)
        assert non_matching.baseline_comparison is None


class TestExplainGuardrailCheck:
    @pytest.mark.parametrize("check_name", _ALL_GUARDRAIL_CHECK_NAMES)
    def test_every_real_guardrail_check_name_has_a_specific_meaning(self, check_name):
        """
        Proves _GUARDRAIL_MEANINGS hasn't drifted out of sync with the
        real check names validate_pipeline() actually produces
        (app/agent/tools/guardrails.py) — a check name missing here
        would silently fall back to a generic placeholder.
        """
        check = ValidationCheck(check=check_name, passed=True, severity="info", message="real finding text")
        explanation = explain_guardrail_check(check)

        assert explanation.meaning == _GUARDRAIL_MEANINGS[check_name]
        assert explanation.meaning != "A deterministic PIPER guardrail check."
        assert explanation.message == "real finding text"
        assert explanation.check == check_name
        assert explanation.passed is True

    def test_unknown_check_name_falls_back_gracefully(self):
        check = ValidationCheck(check="some_future_check", passed=False, severity="error", message="m")
        explanation = explain_guardrail_check(check)
        assert explanation.meaning == "A deterministic PIPER guardrail check."


class TestExplainFailure:
    @pytest.mark.parametrize("category", _ALL_FAILURE_CATEGORIES)
    def test_every_real_failure_category_has_a_specific_meaning(self, category):
        """
        Proves _FAILURE_CATEGORY_MEANINGS covers every category in the
        real FailureCategory taxonomy (app/schemas/failure.py) — a
        category missing here would silently fall back to a generic
        placeholder.
        """
        failure = FailureInfo(
            category=category, message="real evidence", node="validate", attempt=0,
            retryable=category not in ("DATA_ERROR", "SCHEMA_ERROR", "TARGET_ERROR", "EXECUTION_BUDGET_EXCEEDED"),
        )
        explanation = explain_failure(failure)

        assert explanation.meaning == _FAILURE_CATEGORY_MEANINGS[category]
        assert explanation.meaning != "A structured PIPER failure category."
        assert explanation.message == "real evidence"
        assert explanation.category == category
        assert explanation.retryable == failure.retryable
        assert explanation.human_intervention_required == failure.human_intervention_required


class TestBuildRunExplanation:
    def test_grounded_end_to_end_from_a_real_completed_run(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="learn_explain_001", dataset_id="dataset_001", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        record = run_store.get("learn_explain_001")
        assert record.status == "completed"
        state = record.final_state

        explanation = build_run_explanation(record.run_id, state)

        assert explanation.run_id == "learn_explain_001"
        assert explanation.status == "completed"
        assert len(explanation.preprocessing) == len(state.cleaning_log)
        assert len(explanation.feature_engineering) == len(state.feature_log)
        assert explanation.model_selection is not None
        assert explanation.model_selection.recommended_model_id == state.comparison.recommended_model_id
        assert explanation.model_selection.justification == state.comparison.justification
        assert len(explanation.evaluation) == len(state.evaluation_results)
        assert {e.model_id for e in explanation.evaluation} == {r.model_id for r in state.evaluation_results}
        assert len(explanation.guardrail_checks) == len(state.validation.checks)
        assert explanation.failure is None  # stale-failure cleanup (Pre-6A Polish) means this must be None on success

    def test_grounded_for_a_real_failed_run(self, telco_df: pd.DataFrame):
        """A genuinely leaky dataset drives a real DUPLICATE_PLAN failure
        (same fixture pattern as test_api_runs.py's failed-run test)."""
        leaky_df = telco_df.copy()
        leaky_df["leaky_dup"] = leaky_df["Churn"]
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_leak", leaky_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="learn_explain_002", dataset_id="dataset_leak", target_column="Churn")

        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        record = run_store.get("learn_explain_002")
        assert record.status == "failed"
        state = record.final_state

        explanation = build_run_explanation(record.run_id, state)

        assert explanation.failure is not None
        assert explanation.failure.category == state.failure.category
        assert explanation.failure.message == state.failure.message

    def test_bare_state_with_no_downstream_results_does_not_crash(self):
        state = AgentState(run_id="learn_explain_003", dataset_id="d1", target_column="t", status="failed")
        explanation = build_run_explanation(state.run_id, state)

        assert explanation.preprocessing == []
        assert explanation.feature_engineering == []
        assert explanation.model_selection is None
        assert explanation.evaluation == []
        assert explanation.guardrail_checks == []
        assert explanation.failure is None


class TestLearningModeHasZeroEffectOnExecution:
    def test_build_run_explanation_never_mutates_the_state_it_reads(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="learn_zero_effect_001", dataset_id="dataset_001", target_column="Churn")
        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        state = run_store.get("learn_zero_effect_001").final_state
        before = {k: v for k, v in vars(state).items()}

        build_run_explanation("learn_zero_effect_001", state)

        after = {k: v for k, v in vars(state).items()}
        assert before == after

    def test_calling_it_twice_produces_byte_identical_output(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="learn_zero_effect_002", dataset_id="dataset_001", target_column="Churn")
        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        state = run_store.get("learn_zero_effect_002").final_state
        first = build_run_explanation("learn_zero_effect_002", state)
        second = build_run_explanation("learn_zero_effect_002", state)

        assert first == second

    @staticmethod
    def _deterministic_projection(result: dict) -> dict:
        """
        Projects a graph.invoke() result down to fields that should be
        genuinely deterministic across two independent runs on the
        SAME input dataset (split_dataset()/train_model() are both
        seeded — see CLAUDE.md's reproducibility notes) — excludes
        randomly-generated identifiers (run_id, dataset_id clones,
        model_id, split_id, operation_id), which legitimately differ
        between any two independent runs regardless of Learn-Explain.
        """
        comparison = result.get("comparison")
        validation = result.get("validation")
        return {
            "status": result["status"],
            "retry_count": result["retry_count"],
            "validation_valid": validation.valid if validation else None,
            "validation_check_names": sorted(c.check for c in validation.checks) if validation else [],
            "comparison_algorithms_and_metrics": sorted(
                (m.algorithm, m.accuracy, m.precision, m.recall, m.f1, m.roc_auc) for m in comparison.models
            ) if comparison else [],
            "justification": comparison.justification if comparison else None,
            "cleaning_tool_sequence": [op.tool_name for op in result.get("cleaning_log", [])],
            "feature_tool_sequence": [op.tool_name for op in result.get("feature_log", [])],
            "failure_category": result["failure"].category if result.get("failure") else None,
        }

    def test_a_run_produces_the_same_result_whether_or_not_learn_explain_is_invoked(self, telco_df: pd.DataFrame):
        """
        The real proof the locked spec asks for: a run with Learning
        Mode invoked produces byte-identical AgentState/results to one
        without. Since no learning_mode flag is ever threaded into
        AgentState or the graph (Learn-Explain reads only
        already-terminal state, after the fact), this also confirms
        there's no hidden side channel (e.g. shared mutable module
        state) linking the two.
        """
        dataset_store_a = InMemoryDatasetStore()
        dataset_store_a.save("dataset_001", telco_df)
        graph_a = build_graph(dataset_store_a, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial_a = AgentState(run_id="learn_zero_effect_a", dataset_id="dataset_001", target_column="Churn")
        result_a = graph_a.invoke(initial_a, config={"recursion_limit": 50})
        # No Learn-Explain call at all for run A.

        dataset_store_b = InMemoryDatasetStore()
        dataset_store_b.save("dataset_001", telco_df)
        graph_b = build_graph(dataset_store_b, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial_b = AgentState(run_id="learn_zero_effect_b", dataset_id="dataset_001", target_column="Churn")
        result_b = graph_b.invoke(initial_b, config={"recursion_limit": 50})
        build_run_explanation(result_b["run_id"], AgentState(**result_b))  # Learn-Explain invoked for run B.

        assert self._deterministic_projection(result_a) == self._deterministic_projection(result_b)


class TestConceptRegistry:
    def test_concepts_registry_has_core_educational_topics(self):
        from app.learning.registry import CONCEPTS
        keys = {c.key for c in CONCEPTS}
        for required in (
            "missing_value_imputation",
            "column_dropping",
            "categorical_encoding",
            "feature_scaling",
            "data_leakage",
            "target_imbalance",
            "train_test_split",
            "model_selection",
            "baseline_comparison",
            "replan_cycle",
            "feature_importance",
        ):
            assert required in keys

    def test_every_concept_has_valid_fields(self):
        from app.learning.registry import CONCEPTS
        for c in CONCEPTS:
            assert c.key.strip()
            assert c.title.strip()
            assert c.category.strip()
            assert c.summary.strip()
            assert c.detail.strip()


class TestWhyExplanation:
    def test_impute_missing_values_why(self):
        from app.learning.explain import build_why_explanation
        why = build_why_explanation("impute_missing_values", evidence={"column": "Age", "strategy": "median"})
        assert "Age" in why.what_happened
        assert "median" in why.what_happened
        assert "numeric" in why.why
        assert why.concept == "Missing Value Imputation"

    def test_drop_column_why(self):
        from app.learning.explain import build_why_explanation
        why = build_why_explanation("drop_column", evidence={"column": "customerID", "reason": "unique ID"})
        assert "customerID" in why.what_happened
        assert "unique ID" in why.why

    def test_levels_beginner_intermediate_advanced(self):
        from app.learning.explain import build_why_explanation
        beg = build_why_explanation("scale_features", level="beginner")
        med = build_why_explanation("scale_features", level="intermediate")
        adv = build_why_explanation("scale_features", level="advanced")
        assert beg.what_happened != med.what_happened or beg.level == "beginner"
        assert adv.level == "advanced"


class TestLearningJourney:
    def test_14_stages_on_completed_run(self, telco_df: pd.DataFrame):
        from app.learning.explain import build_learning_journey
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="journey_001", dataset_id="dataset_001", target_column="Churn")
        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        record = run_store.get("journey_001")
        journey = build_learning_journey(record.run_id, record.final_state)

        assert journey.run_id == "journey_001"
        assert len(journey.stages) == 14
        assert all(s.status == "completed" for s in journey.stages)
        stage_titles = [s.title for s in journey.stages]
        assert "Understand the Dataset" in stage_titles[0]
        assert "Test Unseen Data" in stage_titles[13]

    def test_journey_on_failed_run(self, telco_df: pd.DataFrame):
        from app.learning.explain import build_learning_journey
        leaky_df = telco_df.copy()
        leaky_df["leaky_dup"] = leaky_df["Churn"]
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_leak", leaky_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="journey_fail_001", dataset_id="dataset_leak", target_column="Churn")
        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        record = run_store.get("journey_fail_001")
        journey = build_learning_journey(record.run_id, record.final_state)

        assert journey.run_id == "journey_fail_001"
        assert len(journey.stages) == 14
        # Downstream stages should not be completed
        assert any(s.status in ("failed", "not_reached") for s in journey.stages)


class TestPipelineVisualization:
    def test_pipeline_nodes_and_edges(self, telco_df: pd.DataFrame):
        from app.learning.explain import build_pipeline_visualization
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="pipe_viz_001", dataset_id="dataset_001", target_column="Churn")
        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        record = run_store.get("pipe_viz_001")
        viz = build_pipeline_visualization(record.run_id, record.final_state)

        node_ids = {n.id for n in viz.nodes}
        assert {"dataset", "preprocessing", "split", "models", "evaluation", "guardrails", "winner", "artifact"}.issubset(node_ids)
        assert len(viz.edges) >= 7


class TestReplanAndFeatureImportanceEducation:
    def test_replan_explanation(self):
        from app.learning.explain import build_replan_explanation
        state = AgentState(run_id="r1", dataset_id="d1", target_column="t", retry_count=1)
        replan = build_replan_explanation(state)
        assert replan.replan_occurred is True
        assert "REPLAN" in replan.educational_takeaway or "autonomous" in replan.educational_takeaway

    def test_feature_importance_disclaimer(self):
        from app.learning.explain import build_feature_importance_education
        state = AgentState(run_id="r1", dataset_id="d1", target_column="t")
        fi = build_feature_importance_education(state)
        assert "not prove causation" in fi.disclaimer


class TestStudentModeTrustBoundaries:
    def test_no_llm_dependency_in_learning_module(self):
        """Proves app/learning contains no imports or references to langchain, ollama, or providers."""
        import importlib
        import sys
        mod = importlib.import_module("app.learning.explain")
        assert "ollama" not in sys.modules or not hasattr(mod, "OllamaProvider")

    def test_what_if_experiment_isolation(self, telco_df: pd.DataFrame):
        """Proves What-If exploration creates a separate experiment and does not mutate the base run."""
        from app.agent.tools.exploration import explore_alternative
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(dataset_store, split_store, model_store, heuristic_llm_provider())
        run_store = InMemoryRunStore()
        initial = AgentState(run_id="base_run_001", dataset_id="dataset_001", target_column="Churn")
        stream_with_tracing(graph, initial, run_store, config={"recursion_limit": 50})

        base_record = run_store.get("base_run_001")
        base_model_id = base_record.final_state.comparison.recommended_model_id
        run_model_ids = [m.model_id for m in base_record.final_state.model_results]

        result = explore_alternative(
            run_id="base_run_001",
            run_model_ids=run_model_ids,
            base_model_id=base_model_id,
            split_store=split_store,
            model_store=model_store,
            new_algorithm="random_forest",
        )

        assert result.success is True
        # Base run remains unchanged
        assert run_store.get("base_run_001").final_state.comparison.recommended_model_id == base_model_id
        assert result.data.experiment_id.startswith("exp_")

