"""
Total-planning-deadline tests.

Defect being fixed: `urllib`'s `timeout` bounds each individual SOCKET
OPERATION, not total request duration. Measured in this project's own
benchmark artifacts: single planning calls of 20,923s (5.8h) and 59,679s
(16.6h) against a configured 600s budget, each ultimately reporting "did
not respond within 600.0s". The documented 600s ceiling was therefore not
enforceable on its own.

Fix: OllamaProvider.generate_plan() runs the blocking read on a daemon
thread and abandons it if `total_deadline_seconds` passes, returning the
same structured `timeout` ProviderError a socket timeout already produced.

The deadline is transport-layer only. It performs no repair, produces no
plan, and changes no routing/retry/validation/adequacy semantics — which
is exactly what these tests pin.
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.nodes.real_nodes import plan_node_v2
from app.llm.ollama_provider import (
    DEFAULT_TOTAL_DEADLINE_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    OllamaProvider,
)
from app.llm.provider import LLMProviderResult, ProposedPlan, ProposedPlanStep, ProviderError
from app.schemas.failure import FailureInfo
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore


class TestDeadlineConfiguration:
    def test_default_is_the_documented_backstop(self, monkeypatch):
        monkeypatch.delenv("PIPER_OLLAMA_TOTAL_DEADLINE_SECONDS", raising=False)
        assert OllamaProvider().total_deadline_seconds == DEFAULT_TOTAL_DEADLINE_SECONDS

    def test_default_is_above_the_socket_timeout_so_it_is_a_backstop_not_a_new_limit(self):
        """If this inverted, the deadline would start cutting off healthy
        calls that the socket timeout would have allowed."""
        assert DEFAULT_TOTAL_DEADLINE_SECONDS > DEFAULT_TIMEOUT_SECONDS

    def test_reads_environment_variable(self, monkeypatch):
        monkeypatch.setenv("PIPER_OLLAMA_TOTAL_DEADLINE_SECONDS", "123.5")
        assert OllamaProvider().total_deadline_seconds == 123.5

    def test_constructor_arg_overrides_environment(self, monkeypatch):
        monkeypatch.setenv("PIPER_OLLAMA_TOTAL_DEADLINE_SECONDS", "123.5")
        assert OllamaProvider(total_deadline_seconds=7.0).total_deadline_seconds == 7.0

    def test_deadline_is_never_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("PIPER_OLLAMA_TOTAL_DEADLINE_SECONDS", raising=False)
        d = OllamaProvider().total_deadline_seconds
        assert d is not None and d > 0


class _HangingHTTPResponse:
    """Simulates the real defect: a read that never returns within the
    deadline, exactly as observed when the socket timeout fails to bound
    total duration."""

    def __init__(self, release: threading.Event):
        self._release = release

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        self._release.wait(30)  # far longer than the test deadline
        return b'{"response": "{}"}'


class TestDeadlineEnforcement:
    def test_deadline_breach_returns_structured_timeout_not_an_exception(self, monkeypatch):
        release = threading.Event()
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: _HangingHTTPResponse(release)
        )
        provider = OllamaProvider(total_deadline_seconds=0.5)

        started = time.time()
        result = provider.generate_plan(_planning_context())
        elapsed = time.time() - started
        release.set()

        assert result.success is False
        assert result.error.code == "timeout"
        assert "total planning deadline" in result.error.message
        assert result.plan is None

    def test_deadline_actually_bounds_wall_time(self, monkeypatch):
        release = threading.Event()
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: _HangingHTTPResponse(release)
        )
        provider = OllamaProvider(total_deadline_seconds=0.5)

        started = time.time()
        provider.generate_plan(_planning_context())
        elapsed = time.time() - started
        release.set()

        # The whole point: it returns near the deadline, not after the
        # 30s the underlying read would have taken.
        assert elapsed < 10, f"deadline did not bound wall time (took {elapsed:.1f}s)"

    def test_normal_completion_before_deadline_is_unaffected(self, monkeypatch):
        class _FastResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return (
                    b'{"response": "{\\"steps\\": [{\\"action\\": \\"a\\", '
                    b'\\"tool_name\\": \\"drop_column\\", \\"arguments\\": '
                    b'{\\"column\\": \\"x\\"}, \\"reasoning\\": \\"r\\"}]}"}'
                )

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FastResponse())
        provider = OllamaProvider(total_deadline_seconds=30.0)

        result = provider.generate_plan(_planning_context())

        assert result.success is True
        assert result.plan.steps[0].tool_name == "drop_column"

    def test_deadline_breach_produces_no_plan_so_nothing_can_execute(self, monkeypatch):
        release = threading.Event()
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: _HangingHTTPResponse(release)
        )
        result = OllamaProvider(total_deadline_seconds=0.5).generate_plan(_planning_context())
        release.set()
        assert result.plan is None


def _planning_context():
    from app.llm.provider import LLMPlanningContext

    return LLMPlanningContext(
        objective="o", dataset_context={"columns": 1}, allowed_operations=["drop_column"]
    )


# --- Graph-level: a deadline breach behaves exactly like any other
#     provider failure (no new routing, no new retry, state preserved).


class _DeadlineTimeoutProvider:
    """Returns the exact structured error the deadline produces."""

    def __init__(self, fail_times: int = 99):
        self.calls = 0
        self.fail_times = fail_times

    def generate_plan(self, context):
        self.calls += 1
        if self.calls <= self.fail_times:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="timeout",
                    message="Ollama exceeded PIPER's total planning deadline of 900.0s.",
                ),
            )
        return LLMProviderResult(
            success=True,
            plan=ProposedPlan(steps=[
                ProposedPlanStep(action="a", tool_name="impute_missing_values",
                                 arguments={"column": "age", "strategy": "median"}, reasoning="r"),
                ProposedPlanStep(action="a", tool_name="scale_features",
                                 arguments={"columns": ["age"]}, reasoning="r"),
            ]),
        )


def _df() -> pd.DataFrame:
    n = 60
    return pd.DataFrame({
        "age": [None if i % 6 == 0 else float(i) for i in range(n)],
        "target": [i % 2 for i in range(n)],
    })


class TestDeadlineAtGraphLevel:
    def test_deadline_during_initial_planning_is_a_bounded_retryable_failure(self):
        store = InMemoryDatasetStore()
        store.save("ds", _df())
        provider = _DeadlineTimeoutProvider()
        graph = build_graph(store, InMemorySplitStore(), InMemoryModelStore(), provider)

        result = graph.invoke(
            AgentState(run_id="dl1", dataset_id="ds", target_column="target", max_retries=2),
            config={"recursion_limit": 50},
        )

        assert result["status"] == "failed"
        # Retry budget UNCHANGED: max_retries + 1 attempts, no more.
        assert provider.calls == 3
        assert result["retry_count"] == 2

    def test_deadline_then_recovery_still_reaches_execution(self):
        store = InMemoryDatasetStore()
        store.save("ds", _df())
        provider = _DeadlineTimeoutProvider(fail_times=1)
        graph = build_graph(store, InMemorySplitStore(), InMemoryModelStore(), provider)

        result = graph.invoke(
            AgentState(run_id="dl2", dataset_id="ds", target_column="target", max_retries=2),
            config={"recursion_limit": 50},
        )

        assert provider.calls == 2
        assert result["plan"], "a genuine REPLAN after a deadline breach must still be able to plan"

    def test_deadline_after_preserved_state_exists_carries_valid_steps_forward(self):
        """The deadline path reuses the provider-failure branch, so
        _carried_forward_preserved_steps() must still apply."""
        store = InMemoryDatasetStore()
        store.save("ds", _df())

        prior = FailureInfo(
            category="PLAN_ADEQUACY",
            message="material adequacy failure",
            evidence={
                "valid_steps": [{"tool_name": "drop_column", "arguments": {"column": "age"}}],
                "implicated_steps": [],
            },
            node="plan", attempt=0, retryable=True, human_intervention_required=False,
        )
        state = AgentState(
            run_id="dl3", dataset_id="ds", target_column="target",
            profile={"dummy": "present"}, failure=prior, retry_count=1, max_retries=2,
        )

        out = plan_node_v2(state, store, _DeadlineTimeoutProvider())

        failure = out["failure"]
        assert failure.category == "EVALUATION_ERROR"
        assert failure.retryable is True
        assert failure.evidence["provider_error_code"] == "timeout"
        assert failure.evidence["valid_steps"] == [
            {"tool_name": "drop_column", "arguments": {"column": "age"}}
        ]

    def test_deadline_breach_writes_no_plan_and_no_plan_history(self):
        store = InMemoryDatasetStore()
        store.save("ds", _df())
        state = AgentState(
            run_id="dl4", dataset_id="ds", target_column="target",
            profile={"dummy": "present"}, max_retries=2,
        )

        out = plan_node_v2(state, store, _DeadlineTimeoutProvider())

        assert out["status"] == "failed"
        assert "plan" not in out
        assert "plan_history" not in out
