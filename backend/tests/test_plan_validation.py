"""
M3 Phase 4: deterministic tool/argument validation behavioral tests.

Covers app/agent/plan_validation.py — the central boundary that
rejects an LLM-proposed plan step before it ever reaches execution.
Not wired into plan_node_v2/graph.py yet — tests exercise
validate_proposed_plan() standalone, per Phase 4's scope.
"""

from __future__ import annotations

import pytest

from app.agent.plan_validation import (
    ALLOWED_TOOL_NAMES,
    TOOL_ARGUMENT_SCHEMAS,
    _VALID_CONVERT_TARGET_TYPES,
    _VALID_IMPUTE_STRATEGIES,
    validate_proposed_plan,
)
from app.llm.provider import ProposedPlanStep


def _step(tool_name: str, arguments: dict) -> ProposedPlanStep:
    return ProposedPlanStep(action="a", tool_name=tool_name, arguments=arguments, reasoning="r")


class TestValidToolsAccepted:
    def test_valid_drop_column_accepted(self):
        """(1) valid tool accepted."""
        result = validate_proposed_plan([_step("drop_column", {"column": "customerID"})], "Churn")
        assert result.valid is True
        assert result.violations == []

    def test_valid_convert_column_type_accepted(self):
        result = validate_proposed_plan(
            [_step("convert_column_type", {"column": "TotalCharges", "target_type": "numeric"})], "Churn"
        )
        assert result.valid is True

    def test_valid_impute_missing_values_accepted(self):
        result = validate_proposed_plan(
            [_step("impute_missing_values", {"column": "TotalCharges", "strategy": "median"})], "Churn"
        )
        assert result.valid is True

    def test_valid_encode_categorical_features_accepted(self):
        result = validate_proposed_plan(
            [_step("encode_categorical_features", {"columns": ["Contract", "PaymentMethod"]})], "Churn"
        )
        assert result.valid is True

    def test_valid_scale_features_accepted(self):
        result = validate_proposed_plan(
            [_step("scale_features", {"columns": ["MonthlyCharges", "tenure"]})], "Churn"
        )
        assert result.valid is True

    def test_multi_step_valid_plan_accepted(self):
        steps = [
            _step("drop_column", {"column": "customerID"}),
            _step("convert_column_type", {"column": "TotalCharges", "target_type": "numeric"}),
            _step("impute_missing_values", {"column": "TotalCharges", "strategy": "median"}),
            _step("encode_categorical_features", {"columns": ["Contract"]}),
            _step("scale_features", {"columns": ["MonthlyCharges"]}),
        ]
        result = validate_proposed_plan(steps, "Churn")
        assert result.valid is True
        assert result.violations == []


class TestUnknownToolRejected:
    def test_unknown_tool_name_rejected(self):
        """(2) unknown tool rejected."""
        result = validate_proposed_plan([_step("delete_everything", {})], "Churn")
        assert result.valid is False
        assert result.violations[0].field == "tool_name"

    def test_unknown_tool_never_executes(self, monkeypatch):
        """
        (10) rejected tool never executes — proves rejection happens
        BEFORE any real tool function is invoked, by asserting no real
        tool import is even touched for an unknown tool_name. Since
        plan_validation.py has no import of the real tool functions at
        all (confirmed structurally: it only imports pydantic/typing),
        there is no code path by which validation could accidentally
        execute anything.
        """
        import inspect

        import app.agent.plan_validation as pv

        source = inspect.getsource(pv)
        # Confirms this module never imports any of the actual
        # execution-capable tool functions — it can only ever compare
        # strings against the allowlist, never call anything.
        for forbidden_import in ("drop_column", "convert_column_type", "impute_missing_values",
                                   "encode_categorical_features", "scale_features"):
            assert f"import {forbidden_import}" not in source
            assert f"from app.agent.tools import" not in source or forbidden_import not in source


class TestArbitraryCodeExecutionRejected:
    def test_arbitrary_python_tool_name_rejected(self):
        """(8) arbitrary executable Python from the LLM is rejected."""
        result = validate_proposed_plan(
            [_step("exec_python", {"code": "import os; os.system('rm -rf /')"})], "Churn"
        )
        assert result.valid is False
        assert result.violations[0].field == "tool_name"

    def test_arbitrary_shell_command_tool_name_rejected(self):
        """(9) arbitrary shell/code execution is rejected."""
        result = validate_proposed_plan(
            [_step("run_shell_command", {"command": "rm -rf /"})], "Churn"
        )
        assert result.valid is False
        assert result.violations[0].field == "tool_name"

    def test_eval_tool_name_rejected(self):
        result = validate_proposed_plan([_step("eval", {"expression": "1+1"})], "Churn")
        assert result.valid is False

    def test_code_smuggled_as_argument_to_valid_tool_does_not_execute(self):
        """
        Even if a step names a VALID tool_name but stuffs code-like
        content into an argument value (e.g. column="__import__('os')"),
        validation only checks type/shape, never evaluates the string —
        and the real tool functions (out of scope here, already
        covered by their own existing tests) only ever use argument
        values as column-name lookups against a DataFrame, never as
        code. This test confirms validation doesn't choke on or
        specially interpret such a value — it's just a string that
        fails as an ordinary nonexistent-column error later, at
        execution, not here.
        """
        result = validate_proposed_plan(
            [_step("drop_column", {"column": "__import__('os').system('rm -rf /')"})], "Churn"
        )
        # Structurally valid (non-empty string in the right field) —
        # rejection of a NONEXISTENT column happens inside the real
        # drop_column() tool at execution time, which is unrelated to
        # and unaffected by this validation layer.
        assert result.valid is True


class TestInvalidArgumentsRejected:
    def test_missing_required_argument_rejected(self):
        """(3) invalid arguments are rejected."""
        result = validate_proposed_plan([_step("drop_column", {})], "Churn")
        assert result.valid is False
        assert result.violations[0].field == "column"

    def test_wrong_argument_type_rejected(self):
        """(4) wrong argument type rejected."""
        result = validate_proposed_plan([_step("drop_column", {"column": 12345})], "Churn")
        assert result.valid is False
        assert result.violations[0].field == "column"

    def test_columns_argument_must_be_list_not_string(self):
        result = validate_proposed_plan(
            [_step("encode_categorical_features", {"columns": "Contract"})], "Churn"
        )
        assert result.valid is False
        assert result.violations[0].field == "columns"

    def test_empty_columns_list_rejected(self):
        result = validate_proposed_plan([_step("scale_features", {"columns": []})], "Churn")
        assert result.valid is False

    def test_columns_list_with_non_string_entry_rejected(self):
        result = validate_proposed_plan(
            [_step("scale_features", {"columns": ["MonthlyCharges", 123]})], "Churn"
        )
        assert result.valid is False

    def test_arguments_not_a_dict_rejected(self):
        step = ProposedPlanStep.model_construct(
            action="a", tool_name="drop_column", arguments={}, reasoning="r"
        )
        # model_construct bypasses validation to simulate a malformed
        # object that got past Pydantic somehow — arguments forced to
        # a non-dict to confirm this layer's own defensive check.
        object.__setattr__(step, "arguments", "not a dict")
        result = validate_proposed_plan([step], "Churn")
        assert result.valid is False
        assert result.violations[0].field == "arguments"


class TestUnsupportedOperationRejected:
    def test_unsupported_strategy_rejected(self):
        """(5) unsupported operation rejected."""
        result = validate_proposed_plan(
            [_step("impute_missing_values", {"column": "x", "strategy": "interpolate"})], "Churn"
        )
        assert result.valid is False
        assert result.violations[0].field == "strategy"

    def test_unsupported_target_type_rejected(self):
        """(6) invalid model/operation-config rejected — target_type here plays the analogous role."""
        result = validate_proposed_plan(
            [_step("convert_column_type", {"column": "x", "target_type": "complex_number"})], "Churn"
        )
        assert result.valid is False
        assert result.violations[0].field == "target_type"

    def test_valid_strategies_all_accepted(self):
        for strategy in ("mean", "median", "mode"):
            result = validate_proposed_plan(
                [_step("impute_missing_values", {"column": "x", "strategy": strategy})], "Churn"
            )
            assert result.valid is True, f"strategy={strategy} should be valid"

    def test_valid_target_types_all_accepted(self):
        for target_type in ("numeric", "string", "datetime"):
            result = validate_proposed_plan(
                [_step("convert_column_type", {"column": "x", "target_type": target_type})], "Churn"
            )
            assert result.valid is True, f"target_type={target_type} should be valid"


class TestInvalidTargetReferenceRejected:
    def test_target_column_as_encode_feature_rejected(self):
        """(7) invalid target reference rejected — target listed as a feature."""
        result = validate_proposed_plan(
            [_step("encode_categorical_features", {"columns": ["Contract", "Churn"]})], "Churn"
        )
        assert result.valid is False
        assert "leakage" in result.violations[0].reason.lower()

    def test_target_column_as_scale_feature_rejected(self):
        result = validate_proposed_plan(
            [_step("scale_features", {"columns": ["MonthlyCharges", "Churn"]})], "Churn"
        )
        assert result.valid is False
        assert "leakage" in result.violations[0].reason.lower()

    def test_target_column_alone_as_only_feature_rejected(self):
        result = validate_proposed_plan(
            [_step("encode_categorical_features", {"columns": ["Churn"]})], "Churn"
        )
        assert result.valid is False


class TestMultipleViolationsReported:
    def test_all_violations_across_all_steps_are_reported(self):
        """Confirms validation doesn't stop at the first problem."""
        steps = [
            _step("unknown_tool", {}),
            _step("drop_column", {}),  # missing column
            _step("impute_missing_values", {"column": "x", "strategy": "bogus"}),
        ]
        result = validate_proposed_plan(steps, "Churn")

        assert result.valid is False
        assert len(result.violations) == 3
        assert {v.step_index for v in result.violations} == {0, 1, 2}

    def test_valid_and_invalid_steps_mixed_still_reports_invalid_ones(self):
        steps = [
            _step("drop_column", {"column": "customerID"}),  # valid
            _step("bad_tool", {}),  # invalid
        ]
        result = validate_proposed_plan(steps, "Churn")

        assert result.valid is False
        assert len(result.violations) == 1
        assert result.violations[0].step_index == 1


class TestAllowedToolNamesConstant:
    def test_allowed_tool_names_matches_the_five_real_tools(self):
        """
        Confirms the allowlist is exactly the five tool_names
        plan_node_v2 has ever produced — no more, no fewer. Adding a
        sixth here without a corresponding real tool would be a
        genuine bug this test would catch.
        """
        assert ALLOWED_TOOL_NAMES == {
            "drop_column",
            "convert_column_type",
            "impute_missing_values",
            "encode_categorical_features",
            "scale_features",
        }

    def test_empty_plan_is_valid(self):
        """An empty proposed plan has no steps to violate anything."""
        result = validate_proposed_plan([], "Churn")
        assert result.valid is True
        assert result.violations == []


class TestToolArgumentSchemasMatchValidator:
    """
    Anti-drift guard for TOOL_ARGUMENT_SCHEMAS (the LLM-facing prompt
    content added post-4-model-benchmark investigation — see the
    constant's own docstring in plan_validation.py). This is the single
    thing standing between "the prompt describes an accurate contract"
    and "the prompt silently drifts from what the real validator
    enforces" as either side changes over time — every assertion here
    is a genuine correctness guarantee, not a style check.
    """

    def test_schema_keys_exactly_match_allowed_tool_names(self):
        assert set(TOOL_ARGUMENT_SCHEMAS.keys()) == ALLOWED_TOOL_NAMES

    @pytest.mark.parametrize("tool_name", sorted(ALLOWED_TOOL_NAMES))
    def test_documented_example_passes_real_validation(self, tool_name):
        """
        Every worked example rendered into the prompt must genuinely be
        accepted by validate_proposed_plan() — proves the documentation
        is not just plausible-looking, it is actually correct against
        the real, unweakened validator.
        """
        example = TOOL_ARGUMENT_SCHEMAS[tool_name]["example"]
        result = validate_proposed_plan([_step(tool_name, example)], "Survived")
        assert result.valid is True, f"{tool_name} example {example} was rejected: {result.violations}"

    def test_impute_strategy_enum_matches_real_validator_constant(self):
        assert set(TOOL_ARGUMENT_SCHEMAS["impute_missing_values"]["arguments"]["strategy"]["enum"]) == (
            _VALID_IMPUTE_STRATEGIES
        )

    def test_convert_target_type_enum_matches_real_validator_constant(self):
        assert set(TOOL_ARGUMENT_SCHEMAS["convert_column_type"]["arguments"]["target_type"]["enum"]) == (
            _VALID_CONVERT_TARGET_TYPES
        )

    def test_drop_column_schema_declares_singular_string_column(self):
        """
        Pins the exact distinction every one of the 4 benchmarked
        models got wrong: drop_column's one argument is a singular
        string, not a list.
        """
        args = TOOL_ARGUMENT_SCHEMAS["drop_column"]["arguments"]
        assert set(args.keys()) == {"column"}
        assert args["column"]["type"] == "string"
        assert args["column"]["required"] is True

    @pytest.mark.parametrize("tool_name", ["encode_categorical_features", "scale_features"])
    def test_multi_column_tools_schema_declares_list_of_strings(self, tool_name):
        args = TOOL_ARGUMENT_SCHEMAS[tool_name]["arguments"]
        assert set(args.keys()) == {"columns"}
        assert args["columns"]["type"] == "array of strings"
        assert args["columns"]["required"] is True


class TestRealWorldBenchmarkFailurePatternsRejected:
    """
    Named regression tests reproducing the LITERAL failure patterns
    observed across the 20 real Ollama calls in the post-benchmark
    investigation (qwen3:4b, qwen3.5:4b, llama3.2:3b, gemma3:4b — see
    backend/benchmark_report.md and CLAUDE.md). validate_proposed_plan()
    already rejected every one of these correctly at the time — these
    tests exist to pin the exact scenarios as permanent regression
    coverage, not because a bug was found in the validator itself.
    """

    def test_drop_column_given_plural_columns_list_rejected(self):
        """qwen3:4b's actual observed failure: wanted to drop 2 columns in
        one step and used the multi-column list shape instead of the
        real, singular `column` field."""
        result = validate_proposed_plan(
            [_step("drop_column", {"columns": ["Name", "Ticket"]})], "Survived"
        )
        assert result.valid is False
        assert result.violations[0].field == "column"

    def test_drop_column_given_empty_string_column_rejected(self):
        result = validate_proposed_plan([_step("drop_column", {"column": ""})], "Survived")
        assert result.valid is False
        assert result.violations[0].field == "column"

    def test_encode_categorical_features_given_invented_columns_to_encode_field_rejected(self):
        """llama3.2:3b's actual observed failure."""
        result = validate_proposed_plan(
            [_step("encode_categorical_features", {"columns_to_encode": ["Sex", "Embarked"]})],
            "Survived",
        )
        assert result.valid is False
        assert result.violations[0].field == "columns"

    def test_drop_column_given_invented_columns_to_drop_field_rejected(self):
        """llama3.2:3b's actual observed failure."""
        result = validate_proposed_plan(
            [_step("drop_column", {"columns_to_drop": ["Name", "Ticket"]})], "Survived"
        )
        assert result.valid is False
        assert result.violations[0].field == "column"

    def test_impute_missing_values_given_invented_method_field_instead_of_strategy_rejected(self):
        """qwen3.5:4b's actual observed failure."""
        result = validate_proposed_plan(
            [_step("impute_missing_values", {"method": "mean"})], "Survived"
        )
        assert result.valid is False
        fields = {v.field for v in result.violations}
        assert "column" in fields
        assert "strategy" in fields

    @pytest.mark.parametrize(
        "tool_name,base_args",
        [
            ("drop_column", {}),
            ("impute_missing_values", {}),
            ("encode_categorical_features", {}),
            ("scale_features", {}),
        ],
    )
    def test_invented_column_names_field_rejected(self, tool_name, base_args):
        """gemma3:4b's actual observed failure — a wrong field name
        (`column_names`) applied consistently across nearly every tool."""
        result = validate_proposed_plan(
            [_step(tool_name, {**base_args, "column_names": ["Age"]})], "Survived"
        )
        assert result.valid is False

    def test_hallucinated_nonexistent_tool_name_rejected(self):
        """llama3.2:3b's actual observed failure: proposed tool_names that
        don't exist anywhere in ALLOWED_TOOL_NAMES."""
        for fake_tool in ("identify_categorical_columns", "select_columns", "onehot_encode"):
            result = validate_proposed_plan([_step(fake_tool, {"column_names": ["Sex"]})], "Survived")
            assert result.valid is False
            assert result.violations[0].field == "tool_name"
