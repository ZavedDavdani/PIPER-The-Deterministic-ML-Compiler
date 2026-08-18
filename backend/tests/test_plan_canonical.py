"""
Formal tests for canonicalize_plan()/CanonicalPlan (Phase 2).
"""

from __future__ import annotations

import pytest

from app.agent.plan_canonical import canonicalize_plan
from app.agent.state import PlanStep


def _step(step_id, tool_name, arguments, reasoning="r", status="completed", action="a"):
    return PlanStep(step_id=step_id, action=action, tool_name=tool_name, arguments=arguments, reasoning=reasoning, status=status)


class TestRationaleIndependence:
    def test_different_step_id_same_hash(self):
        plan_a = [_step("s1", "drop_column", {"column": "x"})]
        plan_b = [_step("s99", "drop_column", {"column": "x"})]
        assert canonicalize_plan(plan_a, "target").plan_hash() == canonicalize_plan(plan_b, "target").plan_hash()

    def test_different_reasoning_same_hash(self):
        plan_a = [_step("s1", "drop_column", {"column": "x"}, reasoning="Reason A")]
        plan_b = [_step("s1", "drop_column", {"column": "x"}, reasoning="Completely different reasoning")]
        assert canonicalize_plan(plan_a, "target").plan_hash() == canonicalize_plan(plan_b, "target").plan_hash()

    def test_different_action_label_same_hash(self):
        plan_a = [_step("s1", "drop_column", {"column": "x"}, action="Drop the id column")]
        plan_b = [_step("s1", "drop_column", {"column": "x"}, action="Remove identifier")]
        assert canonicalize_plan(plan_a, "target").plan_hash() == canonicalize_plan(plan_b, "target").plan_hash()


class TestExecutableDifferencesChangeHash:
    def test_different_argument_value_different_hash(self):
        plan_a = [_step("s1", "impute_missing_values", {"column": "x", "strategy": "median"})]
        plan_b = [_step("s1", "impute_missing_values", {"column": "x", "strategy": "mean"})]
        assert canonicalize_plan(plan_a, "target").plan_hash() != canonicalize_plan(plan_b, "target").plan_hash()

    def test_different_tool_name_different_hash(self):
        plan_a = [_step("s1", "drop_column", {"column": "x"})]
        plan_b = [_step("s1", "convert_column_type", {"column": "x", "target_type": "numeric"})]
        assert canonicalize_plan(plan_a, "target").plan_hash() != canonicalize_plan(plan_b, "target").plan_hash()

    def test_different_target_column_different_hash(self):
        plan = [_step("s1", "drop_column", {"column": "x"})]
        assert canonicalize_plan(plan, "Churn").plan_hash() != canonicalize_plan(plan, "OtherTarget").plan_hash()


class TestOrderSensitivity:
    def test_reordered_steps_different_hash(self):
        step1 = _step("s1", "drop_column", {"column": "x"})
        step2 = _step("s2", "convert_column_type", {"column": "y", "target_type": "numeric"})
        hash_forward = canonicalize_plan([step1, step2], "target").plan_hash()
        hash_reversed = canonicalize_plan([step2, step1], "target").plan_hash()
        assert hash_forward != hash_reversed


class TestModelCandidateOrderInsensitivity:
    def test_reordered_model_candidates_same_hash(self):
        candidates_1 = [("random_forest", {"n_estimators": 200}), ("logistic_regression", {"C": 1.0})]
        candidates_2 = [("logistic_regression", {"C": 1.0}), ("random_forest", {"n_estimators": 200})]
        h1 = canonicalize_plan([], "target", model_candidates=candidates_1).plan_hash()
        h2 = canonicalize_plan([], "target", model_candidates=candidates_2).plan_hash()
        assert h1 == h2

    def test_different_model_candidates_different_hash(self):
        candidates_1 = [("random_forest", {"n_estimators": 200})]
        candidates_2 = [("random_forest", {"n_estimators": 100})]
        h1 = canonicalize_plan([], "target", model_candidates=candidates_1).plan_hash()
        h2 = canonicalize_plan([], "target", model_candidates=candidates_2).plan_hash()
        assert h1 != h2


class TestSkippedStepsExcluded:
    def test_skipped_step_does_not_affect_hash(self):
        plan_with_skip = [
            _step("s1", "drop_column", {"column": "x"}, status="completed"),
            _step("s2", "encode_categorical_features", {"columns": ["y"]}, status="skipped"),
        ]
        plan_without_skip = [
            _step("s1", "drop_column", {"column": "x"}, status="completed"),
        ]
        assert canonicalize_plan(plan_with_skip, "target").plan_hash() == canonicalize_plan(plan_without_skip, "target").plan_hash()

    def test_failed_step_does_not_affect_hash(self):
        plan_with_failed = [
            _step("s1", "drop_column", {"column": "x"}, status="completed"),
            _step("s2", "convert_column_type", {"column": "y", "target_type": "numeric"}, status="failed"),
        ]
        plan_without_failed = [
            _step("s1", "drop_column", {"column": "x"}, status="completed"),
        ]
        assert canonicalize_plan(plan_with_failed, "target").plan_hash() == canonicalize_plan(plan_without_failed, "target").plan_hash()


class TestListValuedArguments:
    """Regression coverage for the real bug: list-valued arguments
    (e.g. scale_features'/encode_categorical_features' columns=[...])
    must canonicalize and hash correctly, not crash."""

    def test_list_valued_arguments_hash_deterministically(self):
        plan = [_step("s1", "scale_features", {"columns": ["tenure", "MonthlyCharges", "TotalCharges"]})]
        h1 = canonicalize_plan(plan, "target").plan_hash()
        h2 = canonicalize_plan(plan, "target").plan_hash()
        assert h1 == h2

    def test_realistic_telco_shaped_plan_hashes_without_error(self):
        plan = [
            _step("s1", "drop_column", {"column": "customerID", "reason": "identifier"}),
            _step("s2", "convert_column_type", {"column": "TotalCharges", "target_type": "numeric"}),
            _step("s3", "impute_missing_values", {"column": "TotalCharges", "strategy": "median"}),
            _step("s4", "encode_categorical_features", {"columns": ["gender", "Partner", "Contract"]}),
            _step("s5", "scale_features", {"columns": ["tenure", "MonthlyCharges", "TotalCharges"]}),
        ]
        result = canonicalize_plan(plan, "Churn").plan_hash()
        assert isinstance(result, str) and len(result) == 64  # sha256 hex digest length
