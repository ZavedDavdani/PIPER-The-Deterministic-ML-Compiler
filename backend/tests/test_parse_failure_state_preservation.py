"""
Regression tests for the confirmed gap recorded by the adequacy-recovery
benchmark v2: state preservation existed ONLY on the PLAN_ADEQUACY path.

Observed (3 of 14 real Ollama calls): attempt N fails adequacy and emits
`valid_steps`/`implicated_steps` so attempt N+1 can PATCH rather than
regenerate. But when attempt N+1's call itself fails at the
transport/parse level (timeout, malformed JSON), plan_node_v2 returned an
EVALUATION_ERROR carrying no classification at all — so attempt N+2
planned from scratch, discarding preservation state that nothing had
actually invalidated. All 3 observed "chain broken" REPLANs came from
exactly this path, and both qwen3.5:4b budget-exhaustion trials followed
this shape.

Fix (_carried_forward_preserved_steps, app/agent/nodes/real_nodes.py):
the provider-failure branch carries an EARLIER attempt's already-
validated classification forward into its own evidence — verbatim,
re-validated through validate_proposed_plan() first, and only when there
is genuinely something to carry. Nothing about the malformed output is
trusted (there IS no output on this path), nothing is auto-corrected or
merged, and the validator/retry/routing/duplicate-plan semantics are
untouched: this only changes what the next prompt is TOLD.
"""

from __future__ import annotations

import pandas as pd

from app.agent.nodes.real_nodes import _carried_forward_preserved_steps, plan_node_v2
from app.agent.state import AgentState
from app.llm.prompts import build_replan_prompt
from app.llm.provider import (
    LLMPlanningContext,
    LLMProviderResult,
    ProviderError,
)
from app.schemas.failure import FailureInfo
from app.storage import InMemoryDatasetStore

_VALID_STEPS = [
    {"tool_name": "drop_column", "arguments": {"column": "customerID"}},
    {"tool_name": "scale_features", "arguments": {"columns": ["tenure", "MonthlyCharges"]}},
]
_IMPLICATED_STEPS = [
    {
        "tool_name": "encode_categorical_features",
        "arguments": {"columns": ["Contract"]},
        "reason": "Contract has unaddressed missing values",
    }
]


def _adequacy_failure(evidence_extra: dict | None = None) -> FailureInfo:
    evidence = {
        "adequacy_status": "INADEQUATE",
        "findings": [],
        "valid_steps": _VALID_STEPS,
        "implicated_steps": _IMPLICATED_STEPS,
    }
    if evidence_extra is not None:
        evidence.update(evidence_extra)
    return FailureInfo(
        category="PLAN_ADEQUACY",
        message="material adequacy failure",
        evidence=evidence,
        node="plan",
        attempt=0,
        retryable=True,
        human_intervention_required=False,
    )


class _AlwaysTransportFailureProvider:
    """Reproduces the observed path: the call never yields a plan at all."""

    def __init__(self, code: str = "timeout"):
        self.code = code
        self.calls = 0
        self.contexts: list = []

    def generate_plan(self, context):
        self.calls += 1
        self.contexts.append(context)
        return LLMProviderResult(
            success=False,
            error=ProviderError(code=self.code, message="boom", raw_response_excerpt="..."),
        )


def _state_with_failure(failure: FailureInfo | None) -> AgentState:
    return AgentState(
        run_id="parse_pres_001",
        dataset_id="dataset_001",
        target_column="Churn",
        profile={"dummy": "profile-present-is-all-plan_node_v2-checks"},
        failure=failure,
        retry_count=1 if failure is not None else 0,
        max_retries=3,
    )


def _store(telco_df: pd.DataFrame) -> InMemoryDatasetStore:
    store = InMemoryDatasetStore()
    store.save("dataset_001", telco_df)
    return store


class TestCarriedForwardHelper:
    def test_returns_nothing_on_a_first_attempt(self):
        assert _carried_forward_preserved_steps(_state_with_failure(None)) == {}

    def test_carries_valid_and_implicated_steps_verbatim(self):
        carried = _carried_forward_preserved_steps(_state_with_failure(_adequacy_failure()))
        assert carried["valid_steps"] == _VALID_STEPS
        assert carried["implicated_steps"] == _IMPLICATED_STEPS

    def test_returns_nothing_when_the_previous_failure_carried_no_classification(self):
        failure = FailureInfo(
            category="EVALUATION_ERROR",
            message="m",
            evidence={"provider_error_code": "timeout"},
            node="plan",
            attempt=0,
            retryable=True,
            human_intervention_required=False,
        )
        assert _carried_forward_preserved_steps(_state_with_failure(failure)) == {}

    def test_returns_nothing_when_valid_steps_is_empty(self):
        failure = _adequacy_failure()
        failure.evidence["valid_steps"] = []
        assert _carried_forward_preserved_steps(_state_with_failure(failure)) == {}

    def test_carried_steps_are_revalidated_and_dropped_if_they_no_longer_pass(self):
        """The carried steps go back through validate_proposed_plan() —
        the sole authority — so the prompt can never be told to preserve
        something the validator would now reject."""
        failure = _adequacy_failure()
        failure.evidence["valid_steps"] = [
            {"tool_name": "drop_column", "arguments": {"column": "customerID"}},
            {"tool_name": "drop_column", "arguments": {"column": ""}},  # invalid
        ]
        assert _carried_forward_preserved_steps(_state_with_failure(failure)) == {}

    def test_malformed_evidence_shapes_are_ignored_not_carried(self):
        for bad in ("not a list", [{"tool_name": "drop_column"}], [{"arguments": {}}], [42]):
            failure = _adequacy_failure()
            failure.evidence["valid_steps"] = bad
            assert _carried_forward_preserved_steps(_state_with_failure(failure)) == {}

    def test_target_column_is_respected_by_revalidation(self):
        """A carried step naming the target as a feature is rejected —
        proves revalidation is real, not a shape check."""
        failure = _adequacy_failure()
        failure.evidence["valid_steps"] = [
            {"tool_name": "scale_features", "arguments": {"columns": ["Churn"]}}
        ]
        assert _carried_forward_preserved_steps(_state_with_failure(failure)) == {}

    def test_helper_does_not_mutate_the_state_or_the_failure_evidence(self):
        failure = _adequacy_failure()
        before = failure.model_dump(mode="json")
        _carried_forward_preserved_steps(_state_with_failure(failure))
        assert failure.model_dump(mode="json") == before


class TestPlanNodeProviderFailureCarriesPreservationState:
    def test_parse_failure_after_an_adequacy_failure_preserves_valid_steps(self, telco_df: pd.DataFrame):
        provider = _AlwaysTransportFailureProvider("malformed_response")
        result = plan_node_v2(_state_with_failure(_adequacy_failure()), _store(telco_df), provider)

        failure = result["failure"]
        assert failure.category == "EVALUATION_ERROR"
        assert failure.retryable is True  # routing/retry semantics unchanged
        assert failure.evidence["provider_error_code"] == "malformed_response"
        assert failure.evidence["valid_steps"] == _VALID_STEPS
        assert failure.evidence["implicated_steps"] == _IMPLICATED_STEPS

    def test_first_attempt_provider_failure_evidence_is_unchanged(self, telco_df: pd.DataFrame):
        """Control: with nothing to carry, this branch's evidence is
        exactly what it was before the fix — the keys are omitted
        entirely, not emitted empty."""
        provider = _AlwaysTransportFailureProvider()
        result = plan_node_v2(_state_with_failure(None), _store(telco_df), provider)

        assert set(result["failure"].evidence) == {"provider_error_code", "raw_response_excerpt"}

    def test_preservation_survives_two_consecutive_parse_failures(self, telco_df: pd.DataFrame):
        """The chain is what matters: adequacy -> parse fail -> parse fail
        must still know what to preserve."""
        provider = _AlwaysTransportFailureProvider()
        store = _store(telco_df)

        first = plan_node_v2(_state_with_failure(_adequacy_failure()), store, provider)
        second = plan_node_v2(_state_with_failure(first["failure"]), store, provider)

        assert second["failure"].evidence["valid_steps"] == _VALID_STEPS

    def test_no_plan_is_produced_and_plan_history_is_untouched(self, telco_df: pd.DataFrame):
        """Malformed output is never treated as valid: a parse failure
        still yields no plan and records no plan identity."""
        provider = _AlwaysTransportFailureProvider()
        result = plan_node_v2(_state_with_failure(_adequacy_failure()), _store(telco_df), provider)

        assert result["status"] == "failed"
        assert "plan" not in result
        assert "plan_history" not in result


class TestCarriedStateReachesTheReplanPrompt:
    def test_replan_prompt_renders_the_preserve_section_for_a_parse_failure(self, telco_df: pd.DataFrame):
        provider = _AlwaysTransportFailureProvider()
        result = plan_node_v2(_state_with_failure(_adequacy_failure()), _store(telco_df), provider)

        prompt = build_replan_prompt(
            LLMPlanningContext(
                objective="o",
                dataset_context={},
                allowed_operations=[],
                failure_context=result["failure"].model_dump(mode="json"),
            )
        )

        assert "=== VALID OPERATIONS (preserve these) ===" in prompt
        assert '"customerID"' in prompt

    def test_a_plain_provider_failure_still_renders_no_preserve_section(self, telco_df: pd.DataFrame):
        provider = _AlwaysTransportFailureProvider()
        result = plan_node_v2(_state_with_failure(None), _store(telco_df), provider)

        prompt = build_replan_prompt(
            LLMPlanningContext(
                objective="o",
                dataset_context={},
                allowed_operations=[],
                failure_context=result["failure"].model_dump(mode="json"),
            )
        )

        assert "VALID OPERATIONS" not in prompt
