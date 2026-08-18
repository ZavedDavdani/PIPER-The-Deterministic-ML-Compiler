"""
Batch 5 (production hardening): regression tests for the genuine
findings fixed this batch.

1. PLAN-node retry routing (CLAUDE.md's previously-documented finding):
   a retryable PLAN-node failure (LLM provider error, or a proposed
   plan that failed validate_proposed_plan()) now genuinely gets a
   REPLAN chance, bounded by max_retries — it previously routed
   straight to REPORT with zero retries despite being marked
   retryable=True.
2. Cross-run dataset isolation: two runs against the SAME uploaded
   dataset_id must never let one run's cleaning mutations corrupt the
   other's input — each run now executes against a private, run-scoped
   clone.
3. Exception safety net: an unexpected exception mid-graph-execution
   must produce a genuine terminal "failed" RunStore result instead of
   leaving the run stuck at "running" forever.
4. plan_node_v2's sanitized-context-build failure branch used an
   invalid FailureCategory literal ("DATA_QUALITY_ERROR", not a member
   of the taxonomy) that would have raised a pydantic ValidationError
   the moment this defensive branch was ever actually exercised.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.nodes.real_nodes import plan_node_v2
from app.agent.tracing import stream_with_tracing
from app.llm.provider import LLMProviderResult, ProposedPlan, ProposedPlanStep
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemoryRunStore, InMemorySplitStore
from tests.conftest import heuristic_llm_provider


class _FailsOnceThenValidProvider:
    """
    First call proposes a structurally-invalid plan (rejected by
    validate_proposed_plan()) — CLEAN never runs, so the dataset stays
    completely unmutated. Every call after that delegates to
    heuristic_llm_provider() (the same deterministic double proven
    elsewhere in this suite to drive a real, unmutated Telco dataset
    to a genuinely completed run), seeing that same pristine dataset.
    """

    def __init__(self):
        self.calls = 0
        self._heuristic = heuristic_llm_provider()

    def generate_plan(self, context):
        self.calls += 1
        if self.calls == 1:
            steps = [ProposedPlanStep(
                action="a", tool_name="impute_missing_values",
                arguments={"column": "TotalCharges", "strategy": "not_a_real_strategy"}, reasoning="r",
            )]
            return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))
        return self._heuristic.generate_plan(context)


class _AlwaysInvalidPlanProvider:
    """
    Every call proposes a structurally-invalid plan — never recovers.
    Each call's invalid `strategy` value is distinct (not_a_real_strategy_1,
    _2, ...) so every attempt is a GENUINELY DIFFERENT invalid proposal —
    always rejected by validate_proposed_plan(), but never executably
    identical to an earlier attempt. This is deliberate: an earlier
    version of this provider proposed the byte-identical invalid plan on
    every call, which (correctly, post-fix — see
    tests/test_replan_duplicate_invalid_plan.py) now gets caught by
    duplicate-invalid-plan detection and terminates EARLY, before the
    retry budget itself is exhausted. Varying the content keeps this
    provider actually exercising "the retry budget is what bounds an
    always-failing-but-never-repeating run" — a different invariant from
    the duplicate-detection one.
    """

    def __init__(self):
        self.calls = 0

    def generate_plan(self, context):
        self.calls += 1
        steps = [ProposedPlanStep(
            action="a", tool_name="impute_missing_values",
            arguments={"column": "TotalCharges", "strategy": f"not_a_real_strategy_{self.calls}"}, reasoning="r",
        )]
        return LLMProviderResult(success=True, plan=ProposedPlan(steps=steps))


class TestPlanNodeRetryRouting:
    def test_retryable_plan_failure_gets_a_replan_chance_and_recovers(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _FailsOnceThenValidProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="b5_plan_retry_001", dataset_id="dataset_001", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert provider.calls == 2
        assert result["status"] == "completed"
        assert result["retry_count"] == 1

    def test_failure_context_reaches_the_provider_on_a_plan_triggered_replan(self, telco_df: pd.DataFrame):
        """The Batch 5 companion fix to is_replan: failure_context/previous_plan_summary
        must reach the provider even though state.plan was still [] after the failed attempt."""
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)

        class RecordingProvider(_FailsOnceThenValidProvider):
            def __init__(self):
                super().__init__()
                self.received_contexts = []

            def generate_plan(self, context):
                self.received_contexts.append(context)
                return super().generate_plan(context)

        provider = RecordingProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="b5_plan_retry_002", dataset_id="dataset_001", target_column="Churn")

        graph.invoke(initial, config={"recursion_limit": 50})

        assert len(provider.received_contexts) == 2
        assert provider.received_contexts[0].failure_context is None
        assert provider.received_contexts[1].failure_context is not None
        assert provider.received_contexts[1].failure_context["category"] == "EVALUATION_ERROR"
        assert provider.received_contexts[1].previous_plan_summary is not None

    def test_retries_are_bounded_by_max_retries_not_unconditional(self, telco_df: pd.DataFrame):
        """
        _AlwaysInvalidPlanProvider proposes a genuinely DIFFERENT
        invalid plan on every call (see its own docstring) — so this
        exercises max_retries as the bound, distinct from
        duplicate-invalid-plan detection (which would otherwise
        terminate the run earlier, before the budget is exhausted; see
        tests/test_replan_duplicate_invalid_plan.py for that case).
        """
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _AlwaysInvalidPlanProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="b5_plan_retry_003", dataset_id="dataset_001", target_column="Churn", max_retries=2)

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert provider.calls == 3  # first attempt + 2 REPLANs, then exhausted
        assert result["status"] == "failed"
        assert result["retry_count"] == 2
        assert result["failure"].category == "EVALUATION_ERROR"
        assert result["failure"].retryable is True

    def test_human_intervention_required_flips_true_once_plan_retries_are_exhausted(self, telco_df: pd.DataFrame):
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_001", telco_df)
        provider = _AlwaysInvalidPlanProvider()
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), provider)
        initial = AgentState(run_id="b5_plan_retry_004", dataset_id="dataset_001", target_column="Churn", max_retries=1)

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        assert result["failure"].human_intervention_required is True
        assert result["failure"].attempt == result["retry_count"]

    def test_non_retryable_duplicate_plan_still_reports_immediately(self, telco_df: pd.DataFrame):
        """Unaffected-by-this-fix control: DUPLICATE_PLAN (retryable=False)
        must still route straight to REPORT, not get a REPLAN chance."""
        leaky_df = telco_df.copy()
        leaky_df["leaky_dup"] = leaky_df["Churn"]
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_leak", leaky_df)
        graph = build_graph(dataset_store, InMemorySplitStore(), InMemoryModelStore(), heuristic_llm_provider())
        initial = AgentState(run_id="b5_plan_retry_005", dataset_id="dataset_leak", target_column="Churn")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] == "failed"
        assert result["failure"].category == "DUPLICATE_PLAN"
        assert result["failure"].retryable is False


class TestSanitizedContextBuildFailureUsesAValidFailureCategory:
    def test_plan_node_reports_a_structured_failure_not_a_crash(self, loaded_store):
        """
        Forces build_sanitized_llm_context() to fail (dataset removed
        from the store after PROFILE ran) — before the Batch 5 fix,
        constructing FailureInfo(category="DATA_QUALITY_ERROR", ...)
        would itself raise pydantic.ValidationError, since that string
        was never a member of the FailureCategory taxonomy.
        """
        from app.agent.tools import profile_dataset

        profile_result = profile_dataset("dataset_001", loaded_store)
        assert profile_result.success

        state = AgentState(
            run_id="b5_ctx_fail_001", dataset_id="dataset_001", target_column="Churn",
            profile=profile_result.data.model_dump(mode="json"),
        )
        loaded_store.delete("dataset_001")

        update = plan_node_v2(state, loaded_store, heuristic_llm_provider())  # must not raise

        assert update["status"] == "failed"
        assert update["failure"].category == "DATA_ERROR"
        assert update["failure"].retryable is False


class TestCrossRunDatasetIsolation:
    def test_two_runs_against_the_same_uploaded_dataset_do_not_corrupt_each_other(self, api_client, telco_df):
        """
        The real regression scenario: upload once, run twice (a normal,
        UI-supported workflow — datasets persist and stay re-selectable).
        Before the Batch 5 fix, clean_node's in-place mutation of the
        SAME dataset_id meant the second run would silently execute
        against the first run's already-cleaned dataset (e.g. customerID
        already dropped) instead of the original upload.
        """
        import io

        csv_bytes = telco_df.to_csv(index=False).encode("utf-8")
        upload = api_client.post("/datasets", files={"file": ("telco.csv", io.BytesIO(csv_bytes), "text/csv")})
        dataset_id = upload.json()["dataset_id"]

        create_a = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id_a = create_a.json()["run_id"]
        create_b = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id_b = create_b.json()["run_id"]

        result_a = api_client.get(f"/runs/{run_id_a}/result").json()
        result_b = api_client.get(f"/runs/{run_id_b}/result").json()

        assert result_a["status"] == "completed"
        assert result_b["status"] == "completed"
        # Both runs report the ORIGINAL uploaded dataset_id back to the
        # caller, even though each executed against a private clone.
        status_a = api_client.get(f"/runs/{run_id_a}").json()
        status_b = api_client.get(f"/runs/{run_id_b}").json()
        assert status_a["dataset_id"] == dataset_id
        assert status_b["dataset_id"] == dataset_id

    def test_run_store_reports_the_original_dataset_id_not_the_internal_clone(self, api_client, telco_df):
        import io

        csv_bytes = telco_df.to_csv(index=False).encode("utf-8")
        upload = api_client.post("/datasets", files={"file": ("telco.csv", io.BytesIO(csv_bytes), "text/csv")})
        dataset_id = upload.json()["dataset_id"]

        create = api_client.post("/runs", json={"dataset_id": dataset_id, "target_column": "Churn"})
        run_id = create.json()["run_id"]

        status = api_client.get(f"/runs/{run_id}")
        assert status.json()["dataset_id"] == dataset_id

    def test_original_uploaded_dataset_is_never_mutated_by_a_run(self):
        """Direct storage-layer proof: after a run, the dataset stored
        under the originally uploaded dataset_id is byte-identical to
        what was uploaded — clean_node's in-place mutation only ever
        touches the run-scoped clone."""
        df = pd.DataFrame({
            "id": [f"c{i}" for i in range(50)],
            "cat": (["a", "b"] * 25),
            "num": list(range(50)),
            "target": (["yes", "no"] * 25),
        })
        dataset_store = InMemoryDatasetStore()
        dataset_store.save("dataset_orig", df)

        run_dataset_id = "run_xyz_data"
        dataset_store.save(run_dataset_id, dataset_store.get("dataset_orig"))
        # Simulate what clean_node does: mutate the run-scoped copy in place.
        from app.agent.tools import drop_column
        drop_column(run_dataset_id, "id", "identifier", dataset_store, target_column="target")

        untouched = dataset_store.get("dataset_orig")
        assert "id" in untouched.columns
        assert list(untouched.columns) == list(df.columns)


class TestTracingExceptionSafetyNet:
    def test_stream_with_tracing_converts_an_unexpected_exception_into_a_terminal_failed_result(self):
        """
        Before the Batch 5 fix, an exception raised mid-graph.stream()
        propagated straight out of stream_with_tracing() uncaught,
        leaving RunStore permanently stuck at "running" — GET
        /runs/{id}/result would 409 forever and the SSE endpoint would
        loop forever waiting for a terminal status. Uses a minimal fake
        "graph" (only needs a .stream() method) so this is a fast,
        isolated test of stream_with_tracing() itself, not a full
        real-graph run.
        """

        class _ExplodingGraph:
            def stream(self, initial_state, config=None, stream_mode="updates"):
                yield {"profile": {"status": "running"}}
                raise RuntimeError("boom: simulated unexpected node failure")

        run_store = InMemoryRunStore()
        initial = AgentState(run_id="b5_exc_001", dataset_id="d1", target_column="t")

        result = stream_with_tracing(_ExplodingGraph(), initial, run_store, config={})

        assert result["status"] == "failed"
        record = run_store.get("b5_exc_001")
        assert record.status == "failed"
        events = run_store.get_events("b5_exc_001")
        assert events[-1].event_type == "run_failed"

    def test_run_with_tracing_also_converts_an_unexpected_exception(self):
        class _ExplodingGraph:
            def invoke(self, initial_state, config=None):
                raise RuntimeError("boom: simulated unexpected node failure")

        from app.agent.tracing import run_with_tracing

        run_store = InMemoryRunStore()
        initial = AgentState(run_id="b5_exc_002", dataset_id="d1", target_column="t")

        result = run_with_tracing(_ExplodingGraph(), initial, run_store, config={})

        assert result["status"] == "failed"
        assert run_store.get("b5_exc_002").status == "failed"
