"""
Formal tests for diff_plans()/PlanDiff (Phase 3).

TestMasterPromptExample reproduces the exact worked example from the
deterministic-core completion spec. TestListValuedArguments is a
regression test for the real bug caught during development: list-
valued tool arguments (scale_features'/encode_categorical_features'
columns=[...]) crashed the original hashable-key implementation.
"""

from __future__ import annotations

from app.agent.plan_canonical import canonicalize_plan
from app.agent.plan_diff import diff_plans
from app.agent.state import PlanStep


def _step(step_id, tool_name, arguments, reasoning="r", status="completed", action="a"):
    return PlanStep(step_id=step_id, action=action, tool_name=tool_name, arguments=arguments, reasoning=reasoning, status=status)


class TestMasterPromptExample:
    """
    Attempt 1: keep customerID, convert TotalCharges.
    Attempt 2: drop customerID, convert TotalCharges.
    Expected: ADDED drop_column(customerID), UNCHANGED convert_column_type.
    """

    def test_added_and_unchanged_match_the_worked_example(self):
        attempt1 = [_step("s1", "convert_column_type", {"column": "TotalCharges", "target_type": "numeric"}, reasoning="r1")]
        attempt2 = [
            _step("s2", "drop_column", {"column": "customerID", "reason": "identifier"}, reasoning="r2"),
            _step("s3", "convert_column_type", {"column": "TotalCharges", "target_type": "numeric"}, reasoning="r3"),
        ]

        cp1 = canonicalize_plan(attempt1, "Churn")
        cp2 = canonicalize_plan(attempt2, "Churn")
        diff = diff_plans(cp1, cp2)

        assert diff.is_duplicate is False
        assert len(diff.added) == 1
        assert diff.added[0].tool_name == "drop_column"
        assert diff.added[0].arguments["column"] == "customerID"
        assert diff.removed == []
        assert diff.changed == []
        assert len(diff.unchanged) == 1
        assert diff.unchanged[0].tool_name == "convert_column_type"


class TestDuplicateDetection:
    def test_identical_plans_different_rationale_is_duplicate(self):
        plan_a = [_step("s1", "drop_column", {"column": "x"}, reasoning="Reason A: this looks like an identifier")]
        plan_b = [_step("s99", "drop_column", {"column": "x"}, reasoning="Reason B: totally different explanation")]

        diff = diff_plans(canonicalize_plan(plan_a, "target"), canonicalize_plan(plan_b, "target"))

        assert diff.is_duplicate is True
        assert diff.added == []
        assert diff.removed == []
        assert diff.changed == []
        assert len(diff.unchanged) == 1

    def test_equivalent_unordered_model_candidates_is_duplicate(self):
        cp_c = canonicalize_plan([], "target", model_candidates=[("random_forest", {"n_estimators": 200}), ("logistic_regression", {"C": 1.0})])
        cp_d = canonicalize_plan([], "target", model_candidates=[("logistic_regression", {"C": 1.0}), ("random_forest", {"n_estimators": 200})])

        diff = diff_plans(cp_c, cp_d)

        assert diff.is_duplicate is True

    def test_truly_identical_plans_are_duplicate(self):
        plan = [_step("s1", "drop_column", {"column": "x"})]
        diff = diff_plans(canonicalize_plan(plan, "target"), canonicalize_plan(plan, "target"))
        assert diff.is_duplicate is True


class TestOneOperationChange:
    def test_single_argument_change_detected_as_changed(self):
        plan_e = [_step("s1", "impute_missing_values", {"column": "x", "strategy": "median"})]
        plan_f = [_step("s1", "impute_missing_values", {"column": "x", "strategy": "mean"})]

        diff = diff_plans(canonicalize_plan(plan_e, "target"), canonicalize_plan(plan_f, "target"))

        assert diff.is_duplicate is False
        assert len(diff.changed) == 1
        assert diff.changed[0].tool_name == "impute_missing_values"
        assert diff.changed[0].old_arguments["strategy"] == "median"
        assert diff.changed[0].new_arguments["strategy"] == "mean"
        assert diff.added == []
        assert diff.removed == []


class TestMultipleOperationChange:
    def test_multiple_changes_all_detected(self):
        plan_old = [
            _step("s1", "drop_column", {"column": "a"}),
            _step("s2", "impute_missing_values", {"column": "b", "strategy": "median"}),
        ]
        plan_new = [
            _step("s1", "drop_column", {"column": "c"}),  # changed argument
            _step("s2", "impute_missing_values", {"column": "b", "strategy": "mean"}),  # changed argument
        ]

        diff = diff_plans(canonicalize_plan(plan_old, "target"), canonicalize_plan(plan_new, "target"))

        assert diff.is_duplicate is False
        assert len(diff.changed) == 2
        changed_tools = {c.tool_name for c in diff.changed}
        assert changed_tools == {"drop_column", "impute_missing_values"}


class TestCompletelyDifferentPlans:
    def test_no_overlap_shows_as_pure_add_and_remove(self):
        plan_g = [_step("s1", "drop_column", {"column": "a"})]
        plan_h = [_step("s1", "scale_features", {"columns": ["b", "c"]})]

        diff = diff_plans(canonicalize_plan(plan_g, "target"), canonicalize_plan(plan_h, "target"))

        assert diff.is_duplicate is False
        assert len(diff.added) == 1
        assert len(diff.removed) == 1
        assert diff.changed == []  # different tool_name entirely -- not paired as "changed"


class TestOrderSensitivePreservation:
    def test_reordered_steps_are_not_duplicate(self):
        step1 = _step("s1", "drop_column", {"column": "x"})
        step2 = _step("s2", "convert_column_type", {"column": "y", "target_type": "numeric"})

        diff = diff_plans(
            canonicalize_plan([step1, step2], "target"),
            canonicalize_plan([step2, step1], "target"),
        )

        assert diff.is_duplicate is False


class TestListValuedArguments:
    """Regression test for the real bug: list-valued arguments crashed
    the original step_key() implementation with
    TypeError: unhashable type: 'list'."""

    def test_diff_with_list_valued_arguments_does_not_crash(self):
        plan_g = [_step("s1", "drop_column", {"column": "a"})]
        plan_h = [_step("s1", "scale_features", {"columns": ["b", "c"]})]

        diff = diff_plans(canonicalize_plan(plan_g, "target"), canonicalize_plan(plan_h, "target"))  # must not raise

        assert diff.added[0].tool_name == "scale_features"

    def test_list_valued_arguments_displayed_as_real_lists_not_corrupted(self):
        plan_g = [_step("s1", "drop_column", {"column": "a"})]
        plan_h = [_step("s1", "scale_features", {"columns": ["b", "c"]})]

        diff = diff_plans(canonicalize_plan(plan_g, "target"), canonicalize_plan(plan_h, "target"))

        assert diff.added[0].arguments == {"columns": ["b", "c"]}
        assert isinstance(diff.added[0].arguments["columns"], list)

    def test_identical_list_valued_arguments_detected_as_duplicate(self):
        plan_i = [_step("s1", "encode_categorical_features", {"columns": ["Contract", "PaymentMethod"]}, reasoning="r1")]
        plan_j = [_step("s1", "encode_categorical_features", {"columns": ["Contract", "PaymentMethod"]}, reasoning="r2 different")]

        diff = diff_plans(canonicalize_plan(plan_i, "target"), canonicalize_plan(plan_j, "target"))

        assert diff.is_duplicate is True


class TestPlanHashesPreservedInDiff:
    def test_diff_carries_the_real_plan_hashes(self):
        plan_a = [_step("s1", "drop_column", {"column": "x"})]
        plan_b = [_step("s1", "drop_column", {"column": "y"})]

        cp_a = canonicalize_plan(plan_a, "target")
        cp_b = canonicalize_plan(plan_b, "target")
        diff = diff_plans(cp_a, cp_b)

        assert diff.old_plan_hash == cp_a.plan_hash()
        assert diff.new_plan_hash == cp_b.plan_hash()
        assert diff.old_plan_hash != diff.new_plan_hash
