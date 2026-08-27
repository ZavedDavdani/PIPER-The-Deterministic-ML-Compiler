"""
Deterministic Plan Adequacy tests.

Covers app/agent/plan_adequacy.py (the evaluator), app/schemas/adequacy.py
(the schema), and the integration into plan_node_v2 Ã¢â‚¬â€ including that
adequacy failures reuse the EXISTING REPLAN/retry/duplicate-plan
machinery rather than introducing a parallel one.

Every test here is deterministic and Ollama-independent.
"""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from app.agent.plan_adequacy import classify_plan_steps, evaluate_plan_adequacy
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context
from app.schemas.adequacy import PlanAdequacyResult
from app.schemas.failure import RECOVERABLE_CATEGORIES, FailureCategory
from app.llm.provider import ProposedPlanStep
from app.storage import InMemoryDatasetStore

TARGET = "target"


def _step(tool_name: str, arguments: dict) -> ProposedPlanStep:
    return ProposedPlanStep(action="a", tool_name=tool_name, arguments=arguments, reasoning="r")


def _context(df: pd.DataFrame, target: str = TARGET):
    store = InMemoryDatasetStore()
    store.save("ds", df)
    result = build_sanitized_llm_context("ds", target, store)
    assert result.success
    return result.data


@pytest.fixture()
def df_with_missing() -> pd.DataFrame:
    """`age` 20% missing, `city` 10% missing, `score` complete."""
    n = 100
    return pd.DataFrame({
        "age": [None if i % 5 == 0 else float(20 + i % 40) for i in range(n)],
        "city": [None if i % 10 == 0 else ("NY" if i % 2 else "LA") for i in range(n)],
        "score": [float(i) for i in range(n)],
        TARGET: [i % 2 for i in range(n)],
    })


@pytest.fixture()
def df_clean() -> pd.DataFrame:
    n = 50
    return pd.DataFrame({
        "score": [float(i) for i in range(n)],
        "city": ["NY" if i % 2 else "LA" for i in range(n)],
        TARGET: [i % 2 for i in range(n)],
    })


def _finding_for(result: PlanAdequacyResult, condition: str, column: str):
    for f in result.findings:
        if f.condition == condition and column in f.columns:
            return f
    return None


# --- 1-5: missing-value rule ------------------------------------------


class TestMissingValueAdequacy:
    def test_1_missing_feature_with_no_operation_is_not_addressed(self, df_with_missing):
        """
        (1) Missing column with no operation -> NOT_ADDRESSED.

        `age` is missing 20% and the plan neither imputes/drops it NOR
        names it in any encode/scale step, so it is NOT an effective
        training feature. Under effective-feature semantics the finding
        is still raised (status NOT_ADDRESSED Ã¢â‚¬â€ the condition is real
        and the evidence is surfaced), but its severity is `advisory`:
        train_model() selects X_train = train_df[all_feature_columns],
        so a column outside that list cannot reach either estimator.
        """
        result = evaluate_plan_adequacy(_context(df_with_missing), [_step("scale_features", {"columns": ["score"]})], TARGET)

        age = _finding_for(result, "missing_values", "age")
        assert age.status == "NOT_ADDRESSED"
        assert age.severity == "advisory"
        # An advisory finding must never, on its own, produce a failure.
        assert result.material_failure is False
        assert result.status == "PASS"

    def test_1b_missing_column_IN_the_feature_set_is_material(self, df_with_missing):
        """
        The counterpart to test_1: the identical dataset and identical
        unaddressed column, differing ONLY in that the plan now scales
        `age` Ã¢â‚¬â€ which puts it into the effective feature set, so NaN
        would genuinely reach the training matrix.
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing), [_step("scale_features", {"columns": ["age", "score"]})], TARGET
        )

        age = _finding_for(result, "missing_values", "age")
        assert age.status == "NOT_ADDRESSED"
        assert age.severity == "material"
        assert result.material_failure is True
        assert result.status == "FAIL"

    def test_1c_encoded_missing_column_is_also_material(self, df_with_missing):
        """Encoding puts a column in the feature set just as scaling does."""
        result = evaluate_plan_adequacy(
            _context(df_with_missing), [_step("encode_categorical_features", {"columns": ["city"]})], TARGET
        )

        city = _finding_for(result, "missing_values", "city")
        assert city.status == "NOT_ADDRESSED"
        assert city.severity == "material"
        assert result.material_failure is True

    def test_2_missing_feature_with_imputation_is_addressed(self, df_with_missing):
        """(2) Missing feature with supported imputation -> ADDRESSED.

        A realistic plan also includes a scale/encode step; the test pins
        that the missing_values finding is ADDRESSED and no material_failure
        fires for the imputed columns.  `score` is complete so adding
        scale_features(["score"]) does not introduce a missing-value issue.
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing),
            [_step("impute_missing_values", {"column": "age", "strategy": "median"}),
             _step("impute_missing_values", {"column": "city", "strategy": "mode"}),
             _step("scale_features", {"columns": ["age", "score"]}),
             _step("encode_categorical_features", {"columns": ["city"]})],
            TARGET,
        )

        assert _finding_for(result, "missing_values", "age").status == "ADDRESSED"
        assert _finding_for(result, "missing_values", "city").status == "ADDRESSED"
        assert result.material_failure is False
        assert result.status == "PASS"

    def test_3_missing_feature_with_drop_is_addressed(self, df_with_missing):
        """(3) Missing feature with supported drop -> ADDRESSED.

        A realistic plan also includes a scale step on the remaining
        complete column (`score`) so the empty_feature_set check does not
        fire.  The test intent Ã¢â‚¬â€ drop resolves missingness Ã¢â€ â€™ ADDRESSED Ã¢â‚¬â€
        is unchanged.
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing),
            [_step("drop_column", {"column": "age"}),
             _step("drop_column", {"column": "city"}),
             _step("scale_features", {"columns": ["score"]})],
            TARGET,
        )

        assert _finding_for(result, "missing_values", "age").status == "ADDRESSED"
        assert _finding_for(result, "missing_values", "city").status == "ADDRESSED"
        assert result.material_failure is False

    def test_4_multiple_missing_columns_are_evaluated_independently(self, df_with_missing):
        """
        (4) Multiple missing columns -> each independently evaluated.

        `age` is imputed (ADDRESSED). `city` is unaddressed AND encoded,
        putting it in the effective feature set, so it is material. This
        keeps the test's real invariant Ã¢â‚¬â€ per-column independence, with
        the addressed column never leaking into the material set Ã¢â‚¬â€ while
        making the feature-set membership explicit rather than incidental.
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing),
            [_step("impute_missing_values", {"column": "age", "strategy": "median"}),
             _step("encode_categorical_features", {"columns": ["city"]})],
            TARGET,
        )

        assert _finding_for(result, "missing_values", "age").status == "ADDRESSED"
        assert _finding_for(result, "missing_values", "city").status == "NOT_ADDRESSED"
        assert result.material_failure is True
        # Exactly one material finding Ã¢â‚¬â€ the addressed column must not leak in.
        assert [f.columns for f in result.material_findings] == [["city"]]

    def test_mixed_feature_and_non_feature_missing_columns(self, df_with_missing):
        """
        (6) Mixed: one missing column IN the feature set (material) and one
        NOT in it (advisory) -> overall FAIL, driven solely by the feature-set
        column. Pins the strict advisory invariant: advisory findings coexist
        with material ones without either masking the other.
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing),
            [_step("encode_categorical_features", {"columns": ["city"]})],  # city in-set; age not
            TARGET,
        )

        assert _finding_for(result, "missing_values", "city").severity == "material"
        assert _finding_for(result, "missing_values", "age").severity == "advisory"
        assert result.status == "FAIL"
        assert [f.columns for f in result.material_findings] == [["city"]]

    def test_only_non_feature_missing_columns_pass(self, df_with_missing):
        """
        (7) Every missing column is outside the effective feature set ->
        all advisory -> overall PASS. This is the exact false-positive class
        the effective-feature correction removes.
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing), [_step("scale_features", {"columns": ["score"]})], TARGET
        )

        advisory = [f for f in result.findings
                    if f.condition == "missing_values" and f.status == "NOT_ADDRESSED"]
        assert {c for f in advisory for c in f.columns} == {"age", "city"}
        assert all(f.severity == "advisory" for f in advisory)
        assert result.material_failure is False
        assert result.status == "PASS"

    def test_5_zero_missingness_produces_no_missing_value_failure(self, df_clean):
        """(5) Zero missingness -> no missing-value finding (NOT_APPLICABLE)."""
        result = evaluate_plan_adequacy(
            _context(df_clean), [_step("scale_features", {"columns": ["score"]})], TARGET
        )

        missing = [f for f in result.findings if f.condition == "missing_values"]
        assert len(missing) == 1
        assert missing[0].status == "NOT_APPLICABLE"
        assert missing[0].columns == []
        assert result.material_failure is False

    def test_scaling_a_missing_column_does_not_address_it(self, df_with_missing):
        """
        Explicitly pins the user-specified example: scaling a column does
        NOT resolve its missing values (StandardScaler preserves NaN Ã¢â‚¬â€
        verified against the real sklearn in plan_adequacy.py's docstring).
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing), [_step("scale_features", {"columns": ["age"]})], TARGET
        )

        assert _finding_for(result, "missing_values", "age").status == "NOT_ADDRESSED"
        assert result.material_failure is True

    def test_encoding_a_missing_column_does_not_address_it(self, df_with_missing):
        """
        Encoding does not resolve missingness either. OneHotEncoder absorbs
        NaN as its own category (verified), which is a silent decision, not
        an explicit one Ã¢â‚¬â€ the rule is deliberately estimator-independent.
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing), [_step("encode_categorical_features", {"columns": ["city"]})], TARGET
        )

        assert _finding_for(result, "missing_values", "city").status == "NOT_ADDRESSED"

    def test_target_missingness_is_never_a_feature_finding(self):
        """The missing-value rule applies to FEATURE columns; the target is excluded."""
        df = pd.DataFrame({
            "score": [float(i) for i in range(20)],
            TARGET: [None if i % 4 == 0 else i % 2 for i in range(20)],
        })
        result = evaluate_plan_adequacy(_context(df), [_step("scale_features", {"columns": ["score"]})], TARGET)

        assert _finding_for(result, "missing_values", TARGET) is None


# --- 6-8: target protection -------------------------------------------


class TestTargetProtection:
    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(_step("drop_column", {"column": TARGET}), id="6_target_dropped"),
            pytest.param(_step("scale_features", {"columns": [TARGET]}), id="7_target_scaled"),
            pytest.param(_step("encode_categorical_features", {"columns": [TARGET]}), id="8_target_encoded"),
            pytest.param(_step("impute_missing_values", {"column": TARGET, "strategy": "mode"}), id="target_imputed"),
            pytest.param(_step("convert_column_type", {"column": TARGET, "target_type": "numeric"}), id="target_converted"),
        ],
    )
    def test_target_misuse_is_a_material_failure(self, df_clean, step):
        """(6)(7)(8) Target dropped / scaled / encoded -> material failure."""
        result = evaluate_plan_adequacy(_context(df_clean), [step], TARGET)

        finding = _finding_for(result, "target_protection", TARGET)
        assert finding.status == "NOT_ADDRESSED"
        assert finding.severity == "material"
        assert result.material_failure is True
        assert result.status == "FAIL"

    def test_untouched_target_is_addressed(self, df_clean):
        result = evaluate_plan_adequacy(
            _context(df_clean), [_step("scale_features", {"columns": ["score"]})], TARGET
        )

        assert _finding_for(result, "target_protection", TARGET).status == "ADDRESSED"
        assert result.material_failure is False


# --- 9-10: identifier-like + high missingness --------------------------


class TestIdentifierAndHighMissingness:
    def test_9_identifier_like_column_explicitly_dropped_is_addressed(self):
        """(9) Identifier-like column explicitly dropped -> ADDRESSED."""
        n = 50
        df = pd.DataFrame({
            "user_id": [f"U{i:04d}" for i in range(n)],
            "score": [float(i) for i in range(n)],
            TARGET: [i % 2 for i in range(n)],
        })
        result = evaluate_plan_adequacy(_context(df), [_step("drop_column", {"column": "user_id"})], TARGET)

        assert _finding_for(result, "identifier_like_column", "user_id").status == "ADDRESSED"

    def test_identifier_like_retained_is_advisory_not_material(self):
        """
        An identifier-like column left in is recorded as evidence but must
        NEVER by itself block execution Ã¢â‚¬â€ it violates no V1 execution
        invariant, and retaining such a column can be legitimate.
        """
        n = 50
        df = pd.DataFrame({
            "user_id": [f"U{i:04d}" for i in range(n)],
            "score": [float(i) for i in range(n)],
            TARGET: [i % 2 for i in range(n)],
        })
        result = evaluate_plan_adequacy(_context(df), [_step("scale_features", {"columns": ["score"]})], TARGET)

        finding = _finding_for(result, "identifier_like_column", "user_id")
        assert finding.status == "NOT_ADDRESSED"
        assert finding.severity == "advisory"
        assert result.material_failure is False
        assert result.status == "PASS"

    def test_numeric_high_uniqueness_is_not_identifier_like(self):
        """
        Reuses the EXISTING guardrail convention: a fully-unique NUMERIC
        column is normal (a price, a measurement), not identifier evidence.
        """
        n = 50
        df = pd.DataFrame({
            "measurement": [float(i) for i in range(n)],
            TARGET: [i % 2 for i in range(n)],
        })
        result = evaluate_plan_adequacy(_context(df), [_step("scale_features", {"columns": ["measurement"]})], TARGET)

        assert _finding_for(result, "identifier_like_column", "measurement") is None

    def test_10_high_missingness_column_produces_deterministic_finding(self):
        """
        (10) High-missingness column -> correct deterministic finding carrying
        the REAL measured percentage (not a bucket or invented threshold).

        The column is encoded here, putting it in the effective feature set,
        so the finding is material. There is deliberately no missingness
        THRESHOLD anywhere in this assertion Ã¢â‚¬â€ 90% missing is material for
        exactly the same reason 0.22% missing would be: it is an unaddressed
        missing value in a column that reaches training.
        """
        n = 100
        df = pd.DataFrame({
            "mostly_missing": [None if i % 10 != 0 else "x" for i in range(n)],
            "score": [float(i) for i in range(n)],
            TARGET: [i % 2 for i in range(n)],
        })
        result = evaluate_plan_adequacy(
            _context(df),
            [_step("encode_categorical_features", {"columns": ["mostly_missing"]}),
             _step("scale_features", {"columns": ["score"]})],
            TARGET,
        )

        finding = _finding_for(result, "missing_values", "mostly_missing")
        assert finding.status == "NOT_ADDRESSED"
        assert finding.severity == "material"
        assert "90.0" in finding.evidence  # the real measured percentage, not a bucket
        assert result.material_failure is True

    def test_10b_high_missingness_outside_feature_set_is_advisory(self):
        """
        The same 90%-missing column, NOT used as a feature, is advisory Ã¢â‚¬â€
        proving severity follows effective-feature membership, not the
        magnitude of missingness. This is the `Cabin` case from the real
        Titanic benchmark.
        """
        n = 100
        df = pd.DataFrame({
            "mostly_missing": [None if i % 10 != 0 else "x" for i in range(n)],
            "score": [float(i) for i in range(n)],
            TARGET: [i % 2 for i in range(n)],
        })
        result = evaluate_plan_adequacy(_context(df), [_step("scale_features", {"columns": ["score"]})], TARGET)

        finding = _finding_for(result, "missing_values", "mostly_missing")
        assert finding.status == "NOT_ADDRESSED"
        assert finding.severity == "advisory"
        assert "90.0" in finding.evidence
        assert result.material_failure is False

    def test_high_missingness_is_not_auto_dropped_or_prescribed(self):
        """
        The evaluator must report the condition, never prescribe DROP. Both
        dropping and imputing must be accepted as resolutions.

        A `score` column (complete, numeric) is included so that a realistic
        plan can include scale_features(["score"]) without triggering
        empty_feature_set Ã¢â‚¬â€ the test's intent is solely that both DROP and
        IMPUTE resolve missingness, not that the plan is incomplete.
        """
        n = 100
        df = pd.DataFrame({
            "mostly_missing": [None if i % 10 != 0 else "x" for i in range(n)],
            "score": [float(i) for i in range(n)],
            TARGET: [i % 2 for i in range(n)],
        })
        ctx = _context(df)

        dropped = evaluate_plan_adequacy(ctx, [
            _step("drop_column", {"column": "mostly_missing"}),
            _step("scale_features", {"columns": ["score"]}),
        ], TARGET)
        imputed = evaluate_plan_adequacy(ctx, [
            _step("impute_missing_values", {"column": "mostly_missing", "strategy": "mode"}),
            _step("scale_features", {"columns": ["score"]}),
        ], TARGET)

        assert dropped.material_failure is False
        assert imputed.material_failure is False


# --- 11-14: semantics, purity, determinism -----------------------------


class TestEvaluatorSemantics:
    def test_11_existing_supported_preprocessing_is_interpreted_correctly(self, df_with_missing):
        """(11) A realistic multi-step plan using supported ops is read correctly."""
        result = evaluate_plan_adequacy(
            _context(df_with_missing),
            [
                _step("impute_missing_values", {"column": "age", "strategy": "median"}),
                _step("drop_column", {"column": "city"}),
                _step("scale_features", {"columns": ["age", "score"]}),
            ],
            TARGET,
        )

        assert result.status == "PASS"
        assert result.material_failure is False
        assert _finding_for(result, "missing_values", "age").status == "ADDRESSED"
        assert _finding_for(result, "missing_values", "city").status == "ADDRESSED"

    def test_12_unsupported_operation_is_not_silently_treated_as_addressed(self, df_with_missing):
        """
        (12) An unknown/unsupported tool_name must never satisfy a
        condition. (In the integrated flow validate_proposed_plan() rejects
        such a step first; this pins the evaluator's own behavior so it can
        never become a silent bypass.)
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing),
            [_step("magically_fix_everything", {"column": "age"}),
             _step("handle_missing", {"columns": ["age", "city"]}),
             # A REAL feature-selecting step puts both columns in the
             # effective feature set, so the unsupported ops above cannot
             # hide behind advisory severity Ã¢â‚¬â€ if they were wrongly credited
             # as "addressing" the missingness, this assertion would fail.
             _step("encode_categorical_features", {"columns": ["city"]}),
             _step("scale_features", {"columns": ["age"]})],
            TARGET,
        )

        assert _finding_for(result, "missing_values", "age").status == "NOT_ADDRESSED"
        assert _finding_for(result, "missing_values", "city").status == "NOT_ADDRESSED"
        assert _finding_for(result, "missing_values", "age").severity == "material"
        assert _finding_for(result, "missing_values", "city").severity == "material"
        assert result.material_failure is True

    def test_unsupported_tool_never_puts_a_column_in_the_feature_set(self, df_with_missing):
        """
        An unrecognised tool must not be able to make a column an effective
        feature either Ã¢â‚¬â€ feature-set membership is derived ONLY from the two
        real feature-selecting tools, matching train_model()'s intent.

        The plan contains only a fake tool name Ã¢â‚¬â€ no real FE step Ã¢â‚¬â€ so
        empty_feature_set fires and material_failure is True.  The test
        pins that the missing_values findings are ADVISORY (the fake tool
        did NOT inject those columns into the feature set), not the
        overall material_failure flag.
        """
        result = evaluate_plan_adequacy(
            _context(df_with_missing),
            [_step("encode_everything", {"columns": ["age", "city"]})],
            TARGET,
        )

        # The fake tool must not put age or city into the effective feature set.
        # If it did, they would be material; since it doesn't, they stay advisory.
        assert _finding_for(result, "missing_values", "age").severity == "advisory"
        assert _finding_for(result, "missing_values", "city").severity == "advisory"
        # empty_feature_set is the material finding that fires, not missing_values.
        material_conditions = {f.condition for f in result.material_findings}
        assert "missing_values" not in material_conditions, (
            "A fake tool must never put a missing column into the material finding set."
        )
        assert "empty_feature_set" in material_conditions


    def test_malformed_arguments_do_not_satisfy_a_condition(self, df_with_missing):
        """A right-named tool with the wrong argument shape must not count as addressing anything."""
        result = evaluate_plan_adequacy(
            _context(df_with_missing),
            [_step("drop_column", {"columns": ["age"]}),   # plural: wrong shape for drop_column
             _step("impute_missing_values", {"column": ""})],  # empty column name
            TARGET,
        )

        assert _finding_for(result, "missing_values", "age").status == "NOT_ADDRESSED"

    def test_13_proposed_plan_and_context_are_not_mutated(self, df_with_missing):
        """(13) The evaluator is read-only Ã¢â‚¬â€ it mutates neither argument."""
        ctx = _context(df_with_missing)
        steps = [
            _step("impute_missing_values", {"column": "age", "strategy": "median"}),
            _step("scale_features", {"columns": ["score"]}),
        ]
        ctx_before = ctx.model_dump(mode="json")
        steps_before = copy.deepcopy([s.model_dump(mode="json") for s in steps])

        evaluate_plan_adequacy(ctx, steps, TARGET)

        assert ctx.model_dump(mode="json") == ctx_before
        assert [s.model_dump(mode="json") for s in steps] == steps_before
        assert len(steps) == len(steps_before)  # no steps added or removed

    def test_14_same_input_produces_identical_output(self, df_with_missing):
        """(14) Deterministic: identical inputs -> byte-identical result."""
        ctx = _context(df_with_missing)
        steps = [_step("scale_features", {"columns": ["score"]})]

        first = evaluate_plan_adequacy(ctx, steps, TARGET)
        second = evaluate_plan_adequacy(ctx, steps, TARGET)

        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_evaluator_never_imports_a_store_or_llm(self):
        """
        Structural proof the layer is read-only and LLM-free: it can only
        consume the context object it is handed.
        """
        import inspect

        import app.agent.plan_adequacy as pa

        source = inspect.getsource(pa)
        for forbidden in ("DatasetStore", "store.save", "llm_provider", "generate_plan", "OllamaProvider"):
            assert forbidden not in source


# --- 15-18: integration with the existing architecture -----------------


class TestFailureTaxonomyIntegration:
    def test_15_plan_adequacy_is_a_real_recoverable_failure_category(self):
        """(15) Adequacy failure integrates with the existing FailureInfo taxonomy."""
        assert "PLAN_ADEQUACY" in FailureCategory.__args__
        assert "PLAN_ADEQUACY" in RECOVERABLE_CATEGORIES

    def test_plan_adequacy_has_a_learn_explain_meaning(self):
        """A new category must not fall through to Learn-Explain's generic placeholder."""
        from app.learning.explain import _FAILURE_CATEGORY_MEANINGS

        assert "PLAN_ADEQUACY" in _FAILURE_CATEGORY_MEANINGS
        assert len(_FAILURE_CATEGORY_MEANINGS["PLAN_ADEQUACY"]) > 40


# --- 16-18: real-graph integration -------------------------------------


def _adequacy_dataset() -> pd.DataFrame:
    """
    Valid for PIPER V1 (binary target, enough rows, a usable feature) but
    carries a 20%-missing column, so plan adequacy is genuinely in play.

    `city`/`score` deliberately use modulo patterns that do NOT align with
    the target's `i % 2`. An earlier version of this fixture made `city`
    perfectly predict the target, which tripped the real leakage guardrail
    and caused a REPLAN for reasons unrelated to adequacy Ã¢â‚¬â€ noise that
    would have made these tests assert the wrong thing.
    """
    n = 200
    return pd.DataFrame({
        "age": [None if i % 5 == 0 else float(20 + i % 40) for i in range(n)],
        "city": ["NY" if i % 3 == 0 else ("LA" if i % 3 == 1 else "SF") for i in range(n)],
        "score": [float(i % 7) for i in range(n)],
        "target": ["yes" if i % 2 else "no" for i in range(n)],
    })


class _FixedPlanProvider:
    """Returns the same fixed plan every call, and counts calls."""

    def __init__(self, steps):
        self._steps = steps
        self.calls = 0

    def generate_plan(self, context):
        from app.llm.provider import LLMProviderResult, ProposedPlan

        self.calls += 1
        return LLMProviderResult(success=True, plan=ProposedPlan(steps=list(self._steps)))


def _run_graph(df: pd.DataFrame, provider, max_retries: int = 2):
    from app.agent import AgentState, build_graph
    from app.storage import InMemoryModelStore, InMemorySplitStore

    store = InMemoryDatasetStore()
    store.save("ds_adequacy", df)
    graph = build_graph(store, InMemorySplitStore(), InMemoryModelStore(), provider)
    initial = AgentState(
        run_id="run_adequacy",
        dataset_id="ds_adequacy",
        target_column="target",
        max_retries=max_retries,
    )
    return graph.invoke(initial, config={"recursion_limit": 60})


class TestAdequacyGraphIntegration:
    def test_16_adequacy_failure_uses_the_existing_replan_route(self):
        """
        (16) An inadequate-but-valid plan must produce a RETRYABLE
        PLAN_ADEQUACY failure at the plan node Ã¢â‚¬â€ i.e. it enters the
        existing REPLAN route rather than terminating immediately or
        creating a new branch.
        """
        # Structurally valid, but SCALES `age` (20% missing) without ever
        # imputing or dropping it Ã¢â‚¬â€ `age` is therefore an effective training
        # feature carrying NaN, which is materially inadequate.
        # max_retries=0 so the FIRST adequacy failure is itself terminal and
        # therefore directly inspectable Ã¢â‚¬â€ with a retry budget it would be
        # superseded by the DUPLICATE_PLAN of the repeat (that path is
        # test 17's job).
        provider = _FixedPlanProvider([
            _step("encode_categorical_features", {"columns": ["city"]}),
            _step("scale_features", {"columns": ["age", "score"]}),
        ])
        result = _run_graph(_adequacy_dataset(), provider, max_retries=0)

        assert result["status"] == "failed"
        failure = result["failure"]
        assert failure.category == "PLAN_ADEQUACY"
        assert failure.node == "plan"
        assert failure.retryable is True, "adequacy failures must be recoverable via REPLAN"
        # Carries real deterministic evidence, not a generic message.
        assert "age" in str(failure.evidence)
        assert "missing_values" in str(failure.evidence)
        assert provider.calls == 1

    def test_17_repeated_inadequate_plan_terminates_via_existing_duplicate_detection(self):
        """
        (17) The required end-to-end behavior:
            attempt 0 -> PLAN_ADEQUACY (retryable, REPLAN)
            attempt 1 -> identical plan -> existing DUPLICATE_PLAN -> terminal

        Critically this must cost exactly 2 LLM calls, NOT the full retry
        budget Ã¢â‚¬â€ proving adequacy failures record plan identity and cannot
        consume unbounded Ollama calls.
        """
        # Same materially-inadequate plan as test 16: `age` IS an effective
        # feature (scaled) but is never imputed or dropped.
        provider = _FixedPlanProvider([
            _step("encode_categorical_features", {"columns": ["city"]}),
            _step("scale_features", {"columns": ["age", "score"]}),
        ])
        result = _run_graph(_adequacy_dataset(), provider, max_retries=2)

        assert result["status"] == "failed"
        assert result["failure"].category == "DUPLICATE_PLAN"
        assert provider.calls == 2, (
            f"Expected exactly 2 LLM calls (inadequate -> REPLAN -> duplicate), got {provider.calls}. "
            "An inadequate plan must not be able to consume the whole retry budget."
        )
        # No separate adequacy budget was introduced Ã¢â‚¬â€ the existing
        # retry_count is what moved, and it stayed within max_retries.
        assert result["retry_count"] <= 2

    def test_adequate_plan_still_executes_normally(self):
        """
        Control: a plan that DOES address the condition must pass adequacy
        and proceed into execution Ã¢â‚¬â€ adequacy must not block healthy runs.
        """
        provider = _FixedPlanProvider([
            _step("impute_missing_values", {"column": "age", "strategy": "median"}),
            _step("encode_categorical_features", {"columns": ["city"]}),
            _step("scale_features", {"columns": ["age", "score"]}),
        ])
        result = _run_graph(_adequacy_dataset(), provider, max_retries=0)

        # It must get PAST the plan node. Downstream ML outcome is out of
        # scope here Ã¢â‚¬â€ the invariant under test is only that adequacy did
        # not block a plan that genuinely addresses the condition.
        assert result["plan"], "an adequate plan must be committed to state"
        if result["status"] == "failed":
            assert result["failure"].category != "PLAN_ADEQUACY"

    def test_18_existing_schema_validation_behavior_is_unchanged(self):
        """
        (18) A structurally INVALID plan must still fail exactly as before
        Ã¢â‚¬â€ EVALUATION_ERROR from validate_proposed_plan(), never
        PLAN_ADEQUACY. Adequacy runs strictly after, and cannot pre-empt or
        weaken the existing validator.
        """
        provider = _FixedPlanProvider([
            _step("drop_column", {"columns": ["age"]}),  # wrong shape: plural
        ])
        result = _run_graph(_adequacy_dataset(), provider)

        assert result["status"] == "failed"
        assert result["failure"].category in ("EVALUATION_ERROR", "DUPLICATE_PLAN")
        assert result["failure"].category != "PLAN_ADEQUACY"

    def test_adequacy_evidence_reaches_the_replan_prompt(self):
        """
        Adequacy evidence must flow into the EXISTING build_replan_prompt()
        via the existing FailureInfo Ã¢â‚¬â€ not a separate retry prompt. This
        pins that the structured findings actually reach the model.
        """
        from app.llm.prompts import build_replan_prompt
        from app.llm.provider import LLMPlanningContext
        from app.schemas.failure import FailureInfo

        ctx = _context(_adequacy_dataset(), "target")
        # `age` must be scaled here so it is an EFFECTIVE feature and the
        # finding is genuinely material Ã¢â‚¬â€ otherwise material_findings would
        # be empty and the assertions below would pass only because `age`
        # also appears in the dataset context, which would prove nothing.
        result = evaluate_plan_adequacy(
            ctx, [_step("scale_features", {"columns": ["age", "score"]})], "target"
        )
        assert result.material_findings, "fixture must produce a material finding"
        failure = FailureInfo(
            category="PLAN_ADEQUACY",
            message=result.summary,
            evidence={"findings": [f.model_dump(mode="json") for f in result.material_findings]},
            node="plan",
            attempt=0,
            retryable=True,
            human_intervention_required=False,
        )
        prompt = build_replan_prompt(
            LLMPlanningContext(
                objective="Predict target",
                dataset_context=ctx.model_dump(mode="json"),
                allowed_operations=["drop_column", "impute_missing_values"],
                failure_context=failure.model_dump(mode="json"),
            )
        )

        assert "PLAN_ADEQUACY" in prompt
        assert "age" in prompt
        assert "missing_values" in prompt
        # It must carry real deterministic evidence, not a generic retry nudge.
        assert "20.0" in prompt


# --- classify_plan_steps(): REPLAN state preservation -------------------


class TestClassifyPlanSteps:
    """
    classify_plan_steps() is what makes REPLAN a PATCH rather than a
    blank-slate regeneration: it tells the LLM which of its own operations
    were NOT implicated in the failure and should be carried forward.
    """

    def test_no_material_findings_means_every_step_is_valid(self, df_with_missing):
        steps = [
            _step("impute_missing_values", {"column": "age", "strategy": "median"}),
            _step("impute_missing_values", {"column": "city", "strategy": "mode"}),
            _step("scale_features", {"columns": ["score"]}),
        ]
        result = evaluate_plan_adequacy(_context(df_with_missing), steps, TARGET)
        assert result.material_failure is False

        classified = classify_plan_steps(result.findings, steps)
        assert len(classified["valid_steps"]) == 3
        assert classified["implicated_steps"] == []

    def test_single_column_step_on_a_failing_column_is_implicated(self, df_with_missing):
        steps = [
            _step("scale_features", {"columns": ["age"]}),
            _step("scale_features", {"columns": ["score"]}),
        ]
        result = evaluate_plan_adequacy(_context(df_with_missing), steps, TARGET)
        assert result.material_failure is True

        classified = classify_plan_steps(result.findings, steps)
        implicated_cols = [s["arguments"]["columns"] for s in classified["implicated_steps"]]
        valid_cols = [s["arguments"]["columns"] for s in classified["valid_steps"]]
        assert ["age"] in implicated_cols
        assert ["score"] in valid_cols

    def test_unrelated_step_remains_valid(self, df_with_missing):
        steps = [
            _step("encode_categorical_features", {"columns": ["city"]}),
            _step("scale_features", {"columns": ["score"]}),
        ]
        result = evaluate_plan_adequacy(_context(df_with_missing), steps, TARGET)

        classified = classify_plan_steps(result.findings, steps)
        assert any(s["arguments"].get("columns") == ["score"] for s in classified["valid_steps"])

    def test_multi_column_step_is_implicated_in_full(self, df_with_missing):
        """
        DOCUMENTED SEMANTICS: a multi-column step is implicated in FULL when
        ANY of its columns has a material finding.

        This matches the real tool contract rather than being a convenience
        choice: scale_features takes ONE columns list and PIPER has no
        "scale_features but skip column X" variant, so the step cannot be
        partially preserved. The LLM must reissue the whole operation.
        The tool contract is asserted here, not assumed.
        """
        from app.agent.plan_validation import TOOL_ARGUMENT_SCHEMAS

        assert set(TOOL_ARGUMENT_SCHEMAS["scale_features"]["arguments"]) == {"columns"}

        steps = [_step("scale_features", {"columns": ["age", "score"]})]
        result = evaluate_plan_adequacy(_context(df_with_missing), steps, TARGET)
        assert result.material_failure is True

        classified = classify_plan_steps(result.findings, steps)
        assert classified["valid_steps"] == []
        assert len(classified["implicated_steps"]) == 1
        assert classified["implicated_steps"][0]["arguments"]["columns"] == ["age", "score"]
        assert classified["implicated_steps"][0]["reason"]

    def test_classification_does_not_mutate_the_plan(self, df_with_missing):
        steps = [_step("scale_features", {"columns": ["age", "score"]})]
        before = copy.deepcopy([s.model_dump() for s in steps])
        result = evaluate_plan_adequacy(_context(df_with_missing), steps, TARGET)
        classify_plan_steps(result.findings, steps)
        assert [s.model_dump() for s in steps] == before


# --- REPLAN prompt: exact production JSON ------------------------------


def _adequacy_failure_context(steps, df, target=TARGET):
    """Builds the real FailureInfo evidence plan_node_v2 produces on adequacy failure."""
    from app.schemas.failure import FailureInfo

    result = evaluate_plan_adequacy(_context(df, target), steps, target)
    classified = classify_plan_steps(result.findings, steps)
    fc = FailureInfo(
        category="PLAN_ADEQUACY",
        message=result.summary,
        evidence={
            "findings": [f.model_dump(mode="json") for f in result.material_findings],
            "valid_steps": classified["valid_steps"],
            "implicated_steps": classified["implicated_steps"],
        },
        node="plan",
        attempt=0,
        retryable=True,
        human_intervention_required=False,
    ).model_dump(mode="json")
    return fc, result, classified


class TestReplanPromptValidOperations:
    SECTION = "=== VALID OPERATIONS (preserve these) ==="

    def _prompt(self, failure_context, df):
        from app.llm.prompts import build_replan_prompt
        from app.llm.provider import LLMPlanningContext

        return build_replan_prompt(LLMPlanningContext(
            objective="Predict target",
            dataset_context=_context(df).model_dump(mode="json"),
            allowed_operations=["drop_column", "impute_missing_values", "scale_features"],
            failure_context=failure_context,
        ))

    def test_section_appears_when_valid_steps_exist(self, df_with_missing):
        steps = [
            _step("scale_features", {"columns": ["age"]}),
            _step("drop_column", {"column": "city"}),
        ]
        fc, _, classified = _adequacy_failure_context(steps, df_with_missing)
        assert classified["valid_steps"], "fixture must yield a preservable step"

        prompt = self._prompt(fc, df_with_missing)
        assert self.SECTION in prompt
        assert "preserve" in prompt.lower()

    def test_section_omitted_when_no_valid_steps(self, df_with_missing):
        steps = [_step("scale_features", {"columns": ["age"]})]
        fc, _, classified = _adequacy_failure_context(steps, df_with_missing)
        assert classified["valid_steps"] == []

        assert self.SECTION not in self._prompt(fc, df_with_missing)

    def test_section_omitted_for_non_adequacy_failures(self, df_with_missing):
        """Other failure types must render exactly as before this section existed."""
        from app.schemas.failure import FailureInfo

        fc = FailureInfo(
            category="EVALUATION_ERROR",
            message="structural failure",
            evidence={"violations": [{"field": "column"}]},
            node="plan", attempt=0, retryable=True, human_intervention_required=False,
        ).model_dump(mode="json")

        assert self.SECTION not in self._prompt(fc, df_with_missing)

    def test_valid_steps_render_as_exact_production_json(self, df_with_missing):
        """
        Preserved operations must appear as the LITERAL production tool JSON
        the model is required to emit Ã¢â‚¬â€ never prose, never a renamed
        argument, never a second schema.
        """
        steps = [
            _step("scale_features", {"columns": ["age"]}),
            _step("impute_missing_values", {"column": "city", "strategy": "mode"}),
        ]
        fc, _, classified = _adequacy_failure_context(steps, df_with_missing)
        prompt = self._prompt(fc, df_with_missing)

        # Isolate THIS section only Ã¢â‚¬â€ later sections (e.g. REQUIRED OUTPUT
        # FORMAT) also contain brackets and would corrupt a naive slice.
        section = prompt.split(self.SECTION)[1].split("\n=== ")[0]
        rendered = json.loads(section[section.index("["):section.rindex("]") + 1])

        assert rendered == classified["valid_steps"]
        assert rendered == [{
            "tool_name": "impute_missing_values",
            "arguments": {"column": "city", "strategy": "mode"},
        }]
        assert rendered[0]["tool_name"] == "impute_missing_values"
        assert set(rendered[0]["arguments"]) == {"column", "strategy"}
        assert rendered[0]["arguments"]["column"] == "city"
        assert rendered[0]["arguments"]["strategy"] == "mode"

    def test_rendered_valid_steps_satisfy_the_real_validator(self, df_with_missing):
        """
        Anything presented as "preserve this" must itself be a plan the real
        validator accepts Ã¢â‚¬â€ otherwise the prompt would be instructing the
        model to reproduce something that gets rejected.
        """
        from app.agent.plan_validation import validate_proposed_plan

        steps = [
            _step("scale_features", {"columns": ["age"]}),
            _step("impute_missing_values", {"column": "city", "strategy": "mode"}),
        ]
        _, _, classified = _adequacy_failure_context(steps, df_with_missing)

        revalidated = [_step(s["tool_name"], s["arguments"]) for s in classified["valid_steps"]]
        assert validate_proposed_plan(revalidated, TARGET).valid is True


class TestAdequacyEvidenceIncludesClassification:
    def test_graph_adequacy_failure_evidence_has_valid_and_implicated_steps(self):
        """End-to-end: the real graph PLAN_ADEQUACY evidence carries both lists."""
        provider = _FixedPlanProvider([
            _step("encode_categorical_features", {"columns": ["city"]}),
            _step("scale_features", {"columns": ["age", "score"]}),
        ])
        result = _run_graph(_adequacy_dataset(), provider, max_retries=0)

        failure = result["failure"]
        assert failure.category == "PLAN_ADEQUACY"
        assert "valid_steps" in failure.evidence
        assert "implicated_steps" in failure.evidence
        assert "findings" in failure.evidence
        assert any(s["tool_name"] == "encode_categorical_features"
                   for s in failure.evidence["valid_steps"])
        assert any(s["tool_name"] == "scale_features"
                   for s in failure.evidence["implicated_steps"])


# --- Empty feature-set adequacy gate ----------------------------------
#
# Regression for the Titanic end-to-end failure: a plan with no
# encode_categorical_features or scale_features step passes structural
# validation but must be rejected by adequacy so the existing REPLAN
# loop can ask the LLM to produce a corrected plan.


class TestEmptyFeatureSetAdequacy:
    """
    CASE A: plan with no feature-engineering steps Ã¢â€ â€™ adequacy FAIL, material,
            condition=empty_feature_set.
    CASE B: plan with at least one valid FE step Ã¢â€ â€™ no empty_feature_set
            material finding.
    """

    def _clean_df(self) -> "pd.DataFrame":
        """Minimal clean dataset Ã¢â‚¬â€ no missing values, so the only material
        finding that can fire is empty_feature_set."""
        import pandas as pd
        n = 50
        return pd.DataFrame({
            "age": [float(20 + i % 40) for i in range(n)],
            "city": ["NY" if i % 2 else "LA" for i in range(n)],
            TARGET: [i % 2 for i in range(n)],
        })

    # ------------------------------------------------------------------
    # CASE A Ã¢â‚¬â€ no encode_categorical_features or scale_features step
    # ------------------------------------------------------------------

    def test_case_a_plan_with_no_fe_step_fails_adequacy(self):
        """
        CASE A: a plan consisting ONLY of a drop_column step Ã¢â‚¬â€ structurally
        valid, but produces zero effective features.

        Must produce:
          - result.status == "FAIL"
          - result.material_failure is True
          - exactly one empty_feature_set finding with status=NOT_ADDRESSED,
            severity=material
        """
        df = self._clean_df()
        ctx = _context(df)
        steps = [_step("drop_column", {"column": "city"})]

        result = evaluate_plan_adequacy(ctx, steps, TARGET)

        efs = next(
            (f for f in result.findings if f.condition == "empty_feature_set"),
            None,
        )
        assert efs is not None, "empty_feature_set finding must be present"
        assert efs.status == "NOT_ADDRESSED"
        assert efs.severity == "material"
        assert result.status == "FAIL"
        assert result.material_failure is True

    def test_case_a_completely_empty_plan_fails_adequacy(self):
        """
        CASE A variant: a completely empty plan also has no FE steps.
        """
        df = self._clean_df()
        ctx = _context(df)

        result = evaluate_plan_adequacy(ctx, [], TARGET)

        efs = next(
            (f for f in result.findings if f.condition == "empty_feature_set"),
            None,
        )
        assert efs is not None
        assert efs.status == "NOT_ADDRESSED"
        assert efs.severity == "material"
        assert result.material_failure is True

    def test_case_a_routes_to_replan_via_graph(self):
        """
        CASE A integration: the real graph rejects a no-FE-step plan with
        PLAN_ADEQUACY and routes to REPLAN (retryable=True).

        Uses the same _FixedPlanProvider / _run_graph helpers as the rest
        of this module.  The plan has a drop_column step only Ã¢â‚¬â€ no
        encode_categorical_features or scale_features Ã¢â‚¬â€ so the ONLY
        material finding should be empty_feature_set (clean_df has no
        missing values to create a competing missing_values finding).
        """
        provider = _FixedPlanProvider([
            _step("drop_column", {"column": "city"}),
        ])
        result = _run_graph(self._clean_df(), provider, max_retries=0)

        assert result["status"] == "failed"
        failure = result["failure"]
        assert failure.category == "PLAN_ADEQUACY"
        assert failure.retryable is True
        findings = failure.evidence.get("findings", [])
        assert any(f["condition"] == "empty_feature_set" for f in findings), (
            "empty_feature_set must appear in the replan evidence"
        )


    # ------------------------------------------------------------------
    # CASE B Ã¢â‚¬â€ at least one valid FE step present
    # ------------------------------------------------------------------

    def test_case_b_plan_with_encode_step_has_no_empty_feature_set_finding(self):
        """
        CASE B: a plan that includes encode_categorical_features must NOT
        produce an empty_feature_set material finding.
        """
        df = self._clean_df()
        ctx = _context(df)
        steps = [_step("encode_categorical_features", {"columns": ["city"]})]

        result = evaluate_plan_adequacy(ctx, steps, TARGET)

        efs = next(
            (f for f in result.findings
             if f.condition == "empty_feature_set" and f.status == "NOT_ADDRESSED"),
            None,
        )
        assert efs is None, (
            "A plan with encode_categorical_features must not trigger the "
            "empty_feature_set material finding."
        )

    def test_case_b_plan_with_scale_step_has_no_empty_feature_set_finding(self):
        """
        CASE B: a plan that includes scale_features must NOT produce an
        empty_feature_set material finding.
        """
        df = self._clean_df()
        ctx = _context(df)
        steps = [_step("scale_features", {"columns": ["age"]})]

        result = evaluate_plan_adequacy(ctx, steps, TARGET)

        efs = next(
            (f for f in result.findings
             if f.condition == "empty_feature_set" and f.status == "NOT_ADDRESSED"),
            None,
        )
        assert efs is None

    def test_case_b_plan_with_both_fe_steps_has_no_empty_feature_set_finding(self):
        """
        CASE B: encode + scale together also must not trigger the finding.
        """
        df = self._clean_df()
        ctx = _context(df)
        steps = [
            _step("encode_categorical_features", {"columns": ["city"]}),
            _step("scale_features", {"columns": ["age"]}),
        ]

        result = evaluate_plan_adequacy(ctx, steps, TARGET)

        efs = next(
            (f for f in result.findings
             if f.condition == "empty_feature_set" and f.status == "NOT_ADDRESSED"),
            None,
        )
        assert efs is None
