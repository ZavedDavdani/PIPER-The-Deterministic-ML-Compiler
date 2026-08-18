"""
PIPER V1 deterministic test matrix.

One compact, comprehensive pass over the V1 trust boundary. Every test
here is deterministic and Ollama-independent (fake providers only).

Matrix (numbering matches the V1 hardening spec):
   1 valid plan                     13 material effective missingness
   2 malformed JSON                 14 adequacy REPLAN
   3 invalid tool                   15 parse-failure state preservation
   4 wrong argument key             16 provider-failure state preservation
   5 wrong argument type            17 timeout state preservation
   6 invalid enum                   18 duplicate plan
   7 drop_column array misuse       19 repeated failure / oscillation
   8 multi-column operation         20 target-column protection
   9 numeric missingness            21 retry ceiling
  10 categorical missingness        22 total planning deadline
  11 invalid categorical strategy   23 execution failure
  12 advisory non-effective missing 24 successful end-to-end run
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.plan_adequacy import evaluate_plan_adequacy
from app.agent.plan_validation import validate_proposed_plan
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context
from app.llm.provider import LLMProviderResult, ProposedPlan, ProposedPlanStep, ProviderError
from app.schemas.failure import RECOVERABLE_CATEGORIES, TERMINAL_CATEGORIES
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore

TARGET = "target"


def _step(tool: str, args: dict) -> ProposedPlanStep:
    return ProposedPlanStep(action="a", tool_name=tool, arguments=args, reasoning="r")


def _df(n: int = 120) -> pd.DataFrame:
    """`num` numeric w/ missing, `cat` categorical w/ missing, `plain` complete."""
    return pd.DataFrame({
        "num": [None if i % 6 == 0 else float(i % 50) for i in range(n)],
        "cat": [None if i % 12 == 0 else ("NY" if i % 2 else "LA") for i in range(n)],
        "plain": [float(i % 30) for i in range(n)],
        TARGET: [i % 2 for i in range(n)],
    })


def _ctx(df: pd.DataFrame):
    store = InMemoryDatasetStore()
    store.save("ds", df)
    res = build_sanitized_llm_context("ds", TARGET, store)
    assert res.success
    return res.data


def _adequate_steps() -> list:
    return [
        _step("impute_missing_values", {"column": "num", "strategy": "median"}),
        _step("impute_missing_values", {"column": "cat", "strategy": "mode"}),
        _step("encode_categorical_features", {"columns": ["cat"]}),
        _step("scale_features", {"columns": ["num", "plain"]}),
    ]


class _Provider:
    """Fake provider driven by a scripted list of results."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0
        self.contexts: list = []

    def generate_plan(self, context):
        self.contexts.append(context)
        self.calls += 1
        r = self.results[min(self.calls - 1, len(self.results) - 1)]
        return r() if callable(r) else r


def _ok(steps) -> LLMProviderResult:
    return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))


def _err(code: str, msg: str = "boom") -> LLMProviderResult:
    return LLMProviderResult(success=False, error=ProviderError(code=code, message=msg))


def _run(provider, max_retries: int = 2, df: pd.DataFrame | None = None):
    store = InMemoryDatasetStore()
    store.save("ds", df if df is not None else _df())
    graph = build_graph(store, InMemorySplitStore(), InMemoryModelStore(), provider)
    return graph.invoke(
        AgentState(run_id="m", dataset_id="ds", target_column=TARGET, max_retries=max_retries),
        config={"recursion_limit": 60},
    )


# ---------------------------------------------------------------- 1-8
# Structural validation is authoritative: nothing malformed may pass.


class TestStructuralValidation:
    def test_01_valid_plan_passes(self):
        assert validate_proposed_plan(_adequate_steps(), TARGET).valid is True

    def test_02_malformed_json_is_a_provider_failure_not_a_plan(self):
        """A parse failure yields NO plan — there is nothing to validate."""
        result = _run(_Provider(_err("malformed_response")), max_retries=0)
        assert result["status"] == "failed"
        assert result["failure"].evidence["provider_error_code"] == "malformed_response"
        assert not result.get("plan")

    @pytest.mark.parametrize("tool", ["fit_model", "onehot_encode", "select_columns", ""])
    def test_03_invalid_tool_is_rejected(self, tool):
        assert validate_proposed_plan([_step(tool, {"column": "num"})], TARGET).valid is False

    @pytest.mark.parametrize("args", [
        {"columns": ["num"]},          # plural key on a singular tool
        {"column_name": "num"},        # invented key
        {"columns_to_drop": ["num"]},  # invented key
    ])
    def test_04_wrong_argument_key_is_rejected(self, args):
        assert validate_proposed_plan([_step("drop_column", args)], TARGET).valid is False

    @pytest.mark.parametrize("args", [
        {"column": 123},
        {"column": None},
        {"column": ["num"]},
    ])
    def test_05_wrong_argument_type_is_rejected(self, args):
        assert validate_proposed_plan([_step("drop_column", args)], TARGET).valid is False

    @pytest.mark.parametrize("bad", ["average", "MEDIAN", "fill", ""])
    def test_06_invalid_enum_is_rejected(self, bad):
        steps = [_step("impute_missing_values", {"column": "num", "strategy": bad})]
        assert validate_proposed_plan(steps, TARGET).valid is False

    def test_07_drop_column_array_misuse_is_rejected(self):
        """The single most common real-model error across the benchmarks."""
        steps = [_step("drop_column", {"columns": ["num", "cat"]})]
        res = validate_proposed_plan(steps, TARGET)
        assert res.valid is False and res.violations

    def test_08_multi_column_operation_accepts_a_list(self):
        steps = [_step("scale_features", {"columns": ["num", "plain"]})]
        assert validate_proposed_plan(steps, TARGET).valid is True


# --------------------------------------------------------------- 9-13
# Adequacy: deterministic, effective-feature-gated.


class TestAdequacy:
    def test_09_numeric_missingness_in_feature_set_is_material(self):
        steps = [_step("scale_features", {"columns": ["num"]})]
        res = evaluate_plan_adequacy(_ctx(_df()), steps, TARGET)
        assert res.material_failure is True

    def test_10_categorical_missingness_in_feature_set_is_material(self):
        steps = [_step("encode_categorical_features", {"columns": ["cat"]})]
        res = evaluate_plan_adequacy(_ctx(_df()), steps, TARGET)
        assert res.material_failure is True

    def test_11_invalid_categorical_imputation_strategy_does_not_address_it(self):
        """median on a categorical is rejected at execution, so it cannot
        count as addressing the column (qwen3:8b run_e13cf35f)."""
        steps = [
            _step("impute_missing_values", {"column": "cat", "strategy": "median"}),
            _step("encode_categorical_features", {"columns": ["cat"]}),
        ]
        res = evaluate_plan_adequacy(_ctx(_df()), steps, TARGET)
        assert res.material_failure is True
        assert any(f.condition == "imputation_strategy_compatibility" for f in res.findings)

    def test_12_advisory_non_effective_missingness_never_blocks(self):
        """`cat` is never encoded -> remainder='drop' -> cannot reach training."""
        steps = [
            _step("impute_missing_values", {"column": "num", "strategy": "median"}),
            _step("scale_features", {"columns": ["num", "plain"]}),
        ]
        res = evaluate_plan_adequacy(_ctx(_df()), steps, TARGET)
        assert res.material_failure is False
        cat = [f for f in res.findings if "cat" in f.columns and f.condition == "missing_values"]
        assert cat and cat[0].severity == "advisory"

    def test_13_material_effective_missingness_blocks(self):
        steps = [_step("encode_categorical_features", {"columns": ["cat"]}),
                 _step("scale_features", {"columns": ["num"]})]
        res = evaluate_plan_adequacy(_ctx(_df()), steps, TARGET)
        assert res.status == "FAIL"

    def test_adequacy_is_deterministic(self):
        ctx, steps = _ctx(_df()), _adequate_steps()
        assert evaluate_plan_adequacy(ctx, steps, TARGET).model_dump() == \
               evaluate_plan_adequacy(ctx, steps, TARGET).model_dump()


# -------------------------------------------------------------- 14-19
# REPLAN, state preservation, duplicate detection, oscillation.


class TestReplanAndPreservation:
    def test_14_adequacy_failure_triggers_replan_with_preserved_state(self):
        inadequate = [_step("encode_categorical_features", {"columns": ["cat"]}),
                      _step("scale_features", {"columns": ["plain"]})]
        provider = _Provider(_ok(inadequate), _ok(_adequate_steps()))
        result = _run(provider)

        assert provider.calls == 2
        replan_ctx = provider.contexts[1]
        assert replan_ctx.failure_context["category"] == "PLAN_ADEQUACY"
        assert replan_ctx.failure_context["evidence"]["valid_steps"]

    @pytest.mark.parametrize("code,label", [
        ("malformed_response", "15 parse"),
        ("provider_unavailable", "16 provider"),
        ("timeout", "17 timeout"),
    ])
    def test_15_16_17_state_preserved_across_every_transport_failure(self, code, label):
        """After an adequacy failure, a parse/provider/timeout failure must
        NOT discard the already-validated steps."""
        inadequate = [_step("encode_categorical_features", {"columns": ["cat"]}),
                      _step("scale_features", {"columns": ["plain"]})]
        provider = _Provider(_ok(inadequate), _err(code), _ok(_adequate_steps()))
        _run(provider, max_retries=2)

        assert provider.calls == 3
        third = provider.contexts[2]
        assert third.failure_context["evidence"].get("valid_steps"), (
            f"{label}: preservation chain broken"
        )

    def test_18_duplicate_plan_terminates_early(self):
        """An identical repeated INVALID plan must not burn the whole budget."""
        invalid = [_step("drop_column", {"column": ""})]
        provider = _Provider(_ok(invalid))
        result = _run(provider, max_retries=5)

        assert provider.calls == 2, "duplicate detection must stop the second identical proposal"
        assert result["failure"].category == "DUPLICATE_PLAN"
        assert result["failure"].retryable is False

    def test_19_repeated_but_varying_failure_terminates_cleanly_within_budget(self):
        """Oscillation ('whack-a-mole') is bounded by the retry budget: it
        terminates safely, never loops, and escalates to human review."""
        a = [_step("encode_categorical_features", {"columns": ["cat"]}),
             _step("scale_features", {"columns": ["plain"]})]
        b = [_step("scale_features", {"columns": ["num"]})]
        c = [_step("encode_categorical_features", {"columns": ["cat"]})]
        provider = _Provider(_ok(a), _ok(b), _ok(c))

        result = _run(provider, max_retries=2)

        assert provider.calls == 3                      # budget respected exactly
        assert result["status"] == "failed"
        assert result["failure"].category == "PLAN_ADEQUACY"
        assert result["failure"].human_intervention_required is True
        assert not result.get("model_results")          # nothing executed


# -------------------------------------------------------------- 20-21
# Target protection and retry ceiling.


class TestTargetProtection:
    @pytest.mark.parametrize("steps", [
        [_step("drop_column", {"column": TARGET})],
        [_step("impute_missing_values", {"column": TARGET, "strategy": "median"})],
        [_step("convert_column_type", {"column": TARGET, "target_type": "string"})],
    ])
    def test_20a_adequacy_flags_any_operation_on_the_target(self, steps):
        res = evaluate_plan_adequacy(_ctx(_df()), steps, TARGET)
        tp = [f for f in res.findings if f.condition == "target_protection"]
        assert tp and tp[0].status == "NOT_ADDRESSED" and tp[0].severity == "material"
        assert res.material_failure is True

    @pytest.mark.parametrize("tool", ["encode_categorical_features", "scale_features"])
    def test_20b_structural_validation_rejects_target_as_a_feature(self, tool):
        """Leakage guard — the target may never enter the feature matrix."""
        assert validate_proposed_plan([_step(tool, {"columns": [TARGET]})], TARGET).valid is False

    def test_20c_target_survives_a_real_run_unmodified(self):
        df = _df()
        before = df[TARGET].tolist()
        store = InMemoryDatasetStore()
        store.save("ds", df.copy())
        graph = build_graph(store, InMemorySplitStore(), InMemoryModelStore(),
                            _Provider(_ok(_adequate_steps())))
        graph.invoke(
            AgentState(run_id="tgt", dataset_id="ds", target_column=TARGET, max_retries=1),
            config={"recursion_limit": 60},
        )
        assert store.get("ds")[TARGET].tolist() == before

    def test_21_retry_ceiling_is_never_exceeded(self):
        for max_retries in (0, 1, 2, 3):
            provider = _Provider(_err("provider_unavailable"))
            result = _run(provider, max_retries=max_retries)
            assert provider.calls == max_retries + 1
            assert result["status"] == "failed"


# -------------------------------------------------------------- 22-24


class TestDeadlineExecutionAndSuccess:
    def test_22_total_planning_deadline_is_configured_and_bounded(self):
        from app.llm.ollama_provider import (
            DEFAULT_TIMEOUT_SECONDS,
            DEFAULT_TOTAL_DEADLINE_SECONDS,
            OllamaProvider,
        )
        p = OllamaProvider()
        assert p.total_deadline_seconds > 0
        assert DEFAULT_TOTAL_DEADLINE_SECONDS > DEFAULT_TIMEOUT_SECONDS

    def test_23_execution_failure_is_classified_and_never_silently_swallowed(self):
        """A plan whose feature set is empty cannot train; the failure must
        be structured and must name its real origin, not a cascade symptom."""
        steps = [_step("drop_column", {"column": "num"}),
                 _step("drop_column", {"column": "cat"}),
                 _step("drop_column", {"column": "plain"})]
        result = _run(_Provider(_ok(steps)), max_retries=0)

        assert result["status"] == "failed"
        failure = result["failure"]
        assert failure is not None
        assert failure.category in TERMINAL_CATEGORIES | RECOVERABLE_CATEGORIES
        assert failure.node and failure.message

    def test_24_successful_end_to_end_run(self):
        result = _run(_Provider(_ok(_adequate_steps())), max_retries=1)

        assert result["status"] == "completed"
        assert result["validation"].valid is True
        assert len(result["model_results"]) == 2
        assert result["comparison"].recommended_model_id
        assert result["comparison"].justification
        assert result["failure"] is None, "a completed run must not carry a stale failure"


# ------------------------------------------------- failure classification


class TestFailureClassificationStability:
    """Every failure must be machine-distinguishable and carry full context.

    NOTE: EVALUATION_ERROR deliberately covers plan-validation, parse,
    provider and timeout failures. They are distinguished at the EVIDENCE
    level (`provider_error_code` vs `violations`/`rejected_steps`), not by
    category — a documented V1 design choice, pinned here so it cannot
    drift silently.
    """

    @pytest.mark.parametrize("code", ["malformed_response", "provider_unavailable", "timeout", "http_error"])
    def test_transport_failures_are_distinguishable_by_evidence(self, code):
        result = _run(_Provider(_err(code)), max_retries=0)
        failure = result["failure"]
        assert failure.category == "EVALUATION_ERROR"
        assert failure.evidence["provider_error_code"] == code

    def test_schema_failure_carries_violations_and_rejected_steps(self):
        result = _run(_Provider(_ok([_step("drop_column", {"columns": ["num"]})])), max_retries=0)
        failure = result["failure"]
        assert failure.evidence["violations"]
        assert failure.evidence["rejected_steps"]
        assert "provider_error_code" not in failure.evidence

    def test_every_failure_carries_the_required_context_fields(self):
        for provider in (
            _Provider(_err("timeout")),
            _Provider(_ok([_step("drop_column", {"columns": ["x"]})])),
            _Provider(_ok([_step("encode_categorical_features", {"columns": ["cat"]})])),
        ):
            failure = _run(provider, max_retries=0)["failure"]
            assert failure.category
            assert failure.message                      # human-readable
            assert isinstance(failure.evidence, dict)   # machine-readable
            assert failure.attempt is not None
            assert failure.node
            assert isinstance(failure.retryable, bool)
            assert isinstance(failure.human_intervention_required, bool)
