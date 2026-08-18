"""
Regression tests for the adequacy dtype blind spot, observed for real in
the qwen3:8b demonstration run (run_e13cf35f, 2026-08-17).

Observed: the model proposed
    impute_missing_values(column="Embarked", strategy="median")
`Embarked` is categorical. Because the column WAS named in an impute step,
adequacy marked its missingness ADDRESSED and passed the plan — but at
execution the tool correctly rejected the operation:
    "Strategy 'median' requires a numeric column; 'Embarked' is not numeric."
so the missing values were never actually resolved.

That run still completed only because `Embarked` never entered the
effective feature set. Had it been encoded or scaled, the NaN would have
survived into the training matrix and LogisticRegression would have raised
at fit() time.

Fix: a column counts as imputed ONLY if the proposed strategy can actually
run against its dtype. Nothing is auto-corrected — median is never
silently rewritten to mode; the incompatible step is reported with its
reason and the existing REPLAN loop lets the model fix it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent.plan_adequacy import evaluate_plan_adequacy
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context
from app.agent.tools.type_conversion import impute_missing_values
from app.llm.provider import ProposedPlanStep
from app.storage import InMemoryDatasetStore

TARGET = "target"


def _step(tool_name: str, arguments: dict) -> ProposedPlanStep:
    return ProposedPlanStep(action="a", tool_name=tool_name, arguments=arguments, reasoning="r")


def _context(df: pd.DataFrame, target: str = TARGET):
    store = InMemoryDatasetStore()
    store.save("ds", df)
    res = build_sanitized_llm_context("ds", target, store)
    assert res.success
    return res.data


@pytest.fixture()
def df() -> pd.DataFrame:
    """`cat` categorical w/ missing, `num` numeric w/ missing, `keep` complete."""
    n = 100
    return pd.DataFrame({
        "cat": [None if i % 10 == 0 else ("NY" if i % 2 else "LA") for i in range(n)],
        "num": [None if i % 5 == 0 else float(i) for i in range(n)],
        "keep": [float(i) for i in range(n)],
        TARGET: [i % 2 for i in range(n)],
    })


def _finding(result, condition: str, column: str):
    for f in result.findings:
        if f.condition == condition and column in f.columns:
            return f
    return None


class TestExecutionContractIsWhatWePin:
    """The adequacy rule must mirror the REAL tool, not a guess about it."""

    @pytest.mark.parametrize("strategy", ["mean", "median"])
    def test_real_tool_rejects_numeric_only_strategy_on_categorical(self, df, strategy):
        store = InMemoryDatasetStore()
        store.save("ds", df)
        res = impute_missing_values("ds", "cat", strategy, store)
        assert res.success is False
        assert res.error.code == "unsupported_dtype_strategy_combination"

    def test_real_tool_accepts_mode_on_categorical(self, df):
        store = InMemoryDatasetStore()
        store.save("ds", df)
        assert impute_missing_values("ds", "cat", "mode", store).success is True

    @pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
    def test_real_tool_accepts_every_strategy_on_numeric(self, df, strategy):
        store = InMemoryDatasetStore()
        store.save("ds", df)
        assert impute_missing_values("ds", "num", strategy, store).success is True


class TestIncompatibleImputationDoesNotAddressMissingness:
    @pytest.mark.parametrize("strategy", ["mean", "median"])
    def test_categorical_imputed_with_numeric_only_strategy_is_not_addressed(self, df, strategy):
        """The exact qwen3:8b failure, with the column IN the feature set."""
        steps = [
            _step("impute_missing_values", {"column": "cat", "strategy": strategy}),
            _step("encode_categorical_features", {"columns": ["cat"]}),
            _step("impute_missing_values", {"column": "num", "strategy": "median"}),
            _step("scale_features", {"columns": ["num", "keep"]}),
        ]
        result = evaluate_plan_adequacy(_context(df), steps, TARGET)

        missing = _finding(result, "missing_values", "cat")
        assert missing.status == "NOT_ADDRESSED"
        assert missing.severity == "material"
        assert result.material_failure is True

    def test_incompatibility_finding_explains_why_and_names_the_fix(self, df):
        steps = [
            _step("impute_missing_values", {"column": "cat", "strategy": "median"}),
            _step("encode_categorical_features", {"columns": ["cat"]}),
        ]
        result = evaluate_plan_adequacy(_context(df), steps, TARGET)

        f = _finding(result, "imputation_strategy_compatibility", "cat")
        assert f is not None
        assert f.status == "NOT_ADDRESSED"
        assert f.severity == "material"
        assert "median" in f.evidence and "not numeric" in f.evidence
        assert "mode" in f.reason  # tells the planner what IS valid

    def test_mode_on_categorical_is_addressed(self, df):
        steps = [
            _step("impute_missing_values", {"column": "cat", "strategy": "mode"}),
            _step("encode_categorical_features", {"columns": ["cat"]}),
            _step("impute_missing_values", {"column": "num", "strategy": "median"}),
            _step("scale_features", {"columns": ["num"]}),
        ]
        result = evaluate_plan_adequacy(_context(df), steps, TARGET)

        assert _finding(result, "missing_values", "cat").status == "ADDRESSED"
        assert _finding(result, "imputation_strategy_compatibility", "cat") is None
        assert result.material_failure is False

    @pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
    def test_every_strategy_is_compatible_with_a_numeric_column(self, df, strategy):
        steps = [
            _step("impute_missing_values", {"column": "num", "strategy": strategy}),
            _step("scale_features", {"columns": ["num"]}),
            _step("drop_column", {"column": "cat"}),
        ]
        result = evaluate_plan_adequacy(_context(df), steps, TARGET)

        assert _finding(result, "missing_values", "num").status == "ADDRESSED"
        assert _finding(result, "imputation_strategy_compatibility", "num") is None
        assert result.material_failure is False


class TestSeverityFollowsTheEffectiveFeatureRule:
    def test_incompatible_impute_on_non_feature_column_is_advisory_not_blocking(self, df):
        """The literal qwen3:8b case: Embarked imputed with median but never
        encoded. remainder='drop' means it cannot reach training, so this
        must NOT block — consistent with the existing severity policy."""
        steps = [
            _step("impute_missing_values", {"column": "cat", "strategy": "median"}),
            _step("impute_missing_values", {"column": "num", "strategy": "median"}),
            _step("scale_features", {"columns": ["num", "keep"]}),
        ]
        result = evaluate_plan_adequacy(_context(df), steps, TARGET)

        assert _finding(result, "imputation_strategy_compatibility", "cat").severity == "advisory"
        assert _finding(result, "missing_values", "cat").severity == "advisory"
        assert result.material_failure is False
        assert result.status == "PASS"

    def test_the_two_findings_for_one_column_never_disagree_about_blocking(self, df):
        """Both conditions must apply the same effective-feature rule."""
        for in_feature_set in (True, False):
            steps = [_step("impute_missing_values", {"column": "cat", "strategy": "median"})]
            if in_feature_set:
                steps.append(_step("encode_categorical_features", {"columns": ["cat"]}))
            result = evaluate_plan_adequacy(_context(df), steps, TARGET)

            compat = _finding(result, "imputation_strategy_compatibility", "cat")
            missing = _finding(result, "missing_values", "cat")
            assert compat.severity == missing.severity


class TestNoAutoCorrectionAndNoMutation:
    def test_strategy_is_never_rewritten(self, df):
        steps = [
            _step("impute_missing_values", {"column": "cat", "strategy": "median"}),
            _step("encode_categorical_features", {"columns": ["cat"]}),
        ]
        before = [dict(s.arguments) for s in steps]
        evaluate_plan_adequacy(_context(df), steps, TARGET)
        assert [dict(s.arguments) for s in steps] == before
        assert steps[0].arguments["strategy"] == "median"  # NOT silently -> "mode"

    def test_evaluation_is_deterministic(self, df):
        steps = [
            _step("impute_missing_values", {"column": "cat", "strategy": "median"}),
            _step("encode_categorical_features", {"columns": ["cat"]}),
        ]
        ctx = _context(df)
        a = evaluate_plan_adequacy(ctx, steps, TARGET)
        b = evaluate_plan_adequacy(ctx, steps, TARGET)
        assert a.model_dump() == b.model_dump()

    def test_malformed_impute_step_is_skipped_not_guessed(self, df):
        """validate_proposed_plan() rejects these before adequacy runs; this
        proves adequacy cannot crash or invent a finding if one appears."""
        for bad in ({"column": "cat"}, {"strategy": "median"}, {}, {"column": "", "strategy": "median"}):
            result = evaluate_plan_adequacy(_context(df), [_step("impute_missing_values", bad)], TARGET)
            assert _finding(result, "imputation_strategy_compatibility", "cat") is None


class TestUnknownColumnIsLeftToExecution:
    def test_impute_of_column_absent_from_profile_produces_no_compatibility_finding(self, df):
        result = evaluate_plan_adequacy(
            _context(df), [_step("impute_missing_values", {"column": "nope", "strategy": "median"})], TARGET
        )
        assert _finding(result, "imputation_strategy_compatibility", "nope") is None
