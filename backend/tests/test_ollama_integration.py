"""
M3 Phase 7: real-Ollama integration test.

This is the ONLY test file in the suite that talks to a real, running
Ollama server rather than a fake/stubbed provider. It is deliberately
isolated here (not mixed into test_llm_provider.py or
test_llm_graph_integration.py) so it's obvious at a glance which file
requires a live external dependency.

GATING (per the M3 Phase 7 requirement — explicit opt-in, not
silent skip-if-convenient):

    PIPER_RUN_OLLAMA_TESTS unset or not "1"
        -> every test in this file is SKIPPED (pytest.skip), with a
           clear reason. This is the default, and it's what keeps the
           normal `pytest -q` suite (495 tests as of Phase 6) fully
           deterministic and Ollama-independent.

    PIPER_RUN_OLLAMA_TESTS=1, but the reachability probe fails
        -> the test FAILS, not skips. If a human has explicitly asked
           for the real-Ollama suite to run, a server that should be
           there but isn't is a genuine failure to report, not
           something to quietly paper over — this is the exact
           distinction the Phase 7 instructions asked for.

    PIPER_RUN_OLLAMA_TESTS=1, and Ollama is reachable
        -> tests run for real, against real_provider =
           OllamaProvider() using the documented defaults
           (PIPER_OLLAMA_HOST=http://localhost:11434,
           PIPER_LLM_MODEL=qwen3:4b), or whatever the environment
           variables are actually set to.

Run explicitly with:

    PIPER_RUN_OLLAMA_TESTS=1 pytest -m ollama -q tests/test_ollama_integration.py

or, to run only this file's tests (equivalent, since every test here
carries the marker):

    PIPER_RUN_OLLAMA_TESTS=1 pytest -q tests/test_ollama_integration.py
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pandas as pd
import pytest

from app.agent import AgentState, build_graph
from app.agent.plan_validation import validate_proposed_plan
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context
from app.llm.ollama_provider import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_HOST, OllamaProvider
from app.llm.provider import LLMPlanningContext
from app.storage import InMemoryDatasetStore, InMemoryModelStore, InMemorySplitStore

pytestmark = pytest.mark.ollama


def _ollama_run_requested() -> bool:
    return os.environ.get("PIPER_RUN_OLLAMA_TESTS") == "1"


def _ollama_host() -> str:
    return os.environ.get("PIPER_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def _probe_ollama_reachable(host: str, timeout_seconds: float = 5.0) -> tuple[bool, str]:
    """
    Cheap reachability check (GET /api/tags) — distinct from the
    provider's own generate_plan() call, so a probe failure can be
    reported with a clear "Ollama isn't reachable at all" message
    rather than conflating it with a structured-output/parsing
    failure from generate_plan() itself.
    """
    try:
        request = urllib.request.Request(url=f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response.read()
        return True, ""
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except TimeoutError:
        return False, f"timed out after {timeout_seconds}s"


@pytest.fixture(autouse=True)
def _require_explicit_opt_in():
    """
    Runs before every test in this file. Skips (not fails) when the
    real-Ollama suite was not explicitly requested — this is the
    default, keeps the normal suite green and Ollama-independent.
    """
    if not _ollama_run_requested():
        pytest.skip(
            "Real-Ollama integration tests are gated behind PIPER_RUN_OLLAMA_TESTS=1 "
            "(not set) — this is expected and does not indicate a problem. Run with "
            "PIPER_RUN_OLLAMA_TESTS=1 pytest -q tests/test_ollama_integration.py to enable."
        )


@pytest.fixture()
def real_provider() -> OllamaProvider:
    """
    Uses OllamaProvider's own environment-variable/default resolution
    (PIPER_OLLAMA_HOST/PIPER_LLM_MODEL, defaulting to
    http://localhost:11434 / qwen3:4b) — never hardcodes a
    machine-specific value here, consistent with the provider's own
    design.
    """
    return OllamaProvider()


@pytest.fixture()
def small_synthetic_dataset() -> InMemoryDatasetStore:
    """
    Small, deliberately simple synthetic dataset — NOT the full
    21-column Telco CSV. A real local qwen3:4b call is slow and the
    point of this test is proving the WIRING (provider -> validation
    -> execution -> graph termination) works, not stress-testing an
    LLM's reasoning quality on a large schema. One obvious
    identifier-like column, one categorical, one numeric, a small
    binary target — an unambiguous cleaning task any reasonable
    planner (real or heuristic) should handle correctly.
    """
    df = pd.DataFrame({
        "customer_id": [f"CUST{i:04d}" for i in range(1, 101)],
        "plan_type": (["Basic", "Standard", "Premium"] * 34)[:100],
        "monthly_spend": [20.0 + (i % 50) for i in range(100)],
        "churned": (["No", "Yes"] * 50),
    })
    store = InMemoryDatasetStore()
    store.save("dataset_synthetic", df)
    return store


class TestOllamaReachability:
    def test_ollama_is_reachable(self):
        """
        Reachability itself is asserted as a hard requirement once
        PIPER_RUN_OLLAMA_TESTS=1 has been explicitly set — an
        unreachable server here is a genuine FAILURE for this test
        file's purposes, not a skip (see module docstring: skipping is
        only for the "not requested at all" case, handled by the
        autouse fixture above).
        """
        host = _ollama_host()
        reachable, reason = _probe_ollama_reachable(host)

        assert reachable, (
            f"PIPER_RUN_OLLAMA_TESTS=1 was set, but Ollama is not reachable at "
            f"'{host}': {reason}. If Ollama is genuinely not running, either start "
            f"it or unset PIPER_RUN_OLLAMA_TESTS to skip this suite normally."
        )


class TestRealProviderProducesStructuredPlan:
    def test_provider_generates_a_structured_plan(self, real_provider, small_synthetic_dataset):
        """qwen3:4b can produce the expected structured plan response through the real provider abstraction."""
        sanitized_result = build_sanitized_llm_context("dataset_synthetic", "churned", small_synthetic_dataset)
        assert sanitized_result.success

        context = LLMPlanningContext(
            objective="Predict 'churned' from the remaining columns (binary classification).",
            dataset_context=sanitized_result.data.model_dump(mode="json"),
            allowed_operations=["drop_column", "convert_column_type", "impute_missing_values", "encode_categorical_features", "scale_features"],
        )

        result = real_provider.generate_plan(context)

        assert result.success, f"Real Ollama call failed: {result.error}"
        assert result.plan is not None
        assert len(result.plan.steps) > 0
        for step in result.plan.steps:
            assert step.tool_name  # non-empty
            assert isinstance(step.arguments, dict)

    def test_real_plan_response_reaches_deterministic_validation(self, real_provider, small_synthetic_dataset):
        """
        The real LLM's proposal is independently run through
        validate_proposed_plan() (the same function plan_node_v2 uses
        in production) here, directly — proving deterministic
        validation is a real, separate gate, not something the graph
        merely trusts the provider to have already done.
        """
        sanitized_result = build_sanitized_llm_context("dataset_synthetic", "churned", small_synthetic_dataset)
        context = LLMPlanningContext(
            objective="Predict 'churned' from the remaining columns (binary classification).",
            dataset_context=sanitized_result.data.model_dump(mode="json"),
            allowed_operations=["drop_column", "convert_column_type", "impute_missing_values", "encode_categorical_features", "scale_features"],
        )

        provider_result = real_provider.generate_plan(context)
        assert provider_result.success, f"Real Ollama call failed: {provider_result.error}"

        validation_result = validate_proposed_plan(provider_result.plan.steps, "churned")

        # Either outcome is acceptable evidence the boundary works: a
        # genuinely valid plan proves the happy path; a rejected plan
        # (with violations reported) proves the validator actually
        # inspects and can reject real, unpredictable LLM output --
        # which is arguably the more important property to prove
        # about an UNTRUSTED input source. What must never happen is
        # an exception escaping validate_proposed_plan() itself.
        assert isinstance(validation_result.valid, bool)
        if not validation_result.valid:
            assert len(validation_result.violations) > 0


class TestRealGraphWithRealOllamaProvider:
    def test_real_graph_terminates_with_real_ollama_provider(self, real_provider, small_synthetic_dataset):
        """
        The graph can execute using OllamaProvider end to end. The
        graph must reach a TERMINAL state (completed or failed) — it
        must never hang — regardless of what the model proposes,
        because deterministic validation/duplicate-detection/retry
        limits are what guarantee termination, not the model's
        cooperation.

        A "failed" status is only an acceptable outcome here if the
        failure genuinely came from somewhere AFTER the LLM was
        actually reached (e.g. the model proposed something the
        deterministic validator rejected, or a real guardrail
        genuinely failed) — a provider_unavailable/connection failure
        would mean this test never actually exercised the real
        integration at all, which must be treated as a test failure,
        not a passing "failed" outcome. (TestOllamaReachability's own
        test is what's responsible for surfacing a bare
        "Ollama isn't reachable" condition clearly; this test asserts
        a stronger, more specific thing on top of that.)
        """
        split_store = InMemorySplitStore()
        model_store = InMemoryModelStore()
        graph = build_graph(small_synthetic_dataset, split_store, model_store, real_provider)
        initial = AgentState(run_id="ollama_real_001", dataset_id="dataset_synthetic", target_column="churned")

        result = graph.invoke(initial, config={"recursion_limit": 50})

        assert result["status"] in ("completed", "failed")

        if result["status"] == "failed":
            failure = result["failure"]
            assert failure is not None
            provider_error_code = (failure.evidence or {}).get("provider_error_code")
            assert provider_error_code not in ("provider_unavailable", "timeout", "http_error"), (
                f"Graph failed due to a provider/transport problem, not a genuine "
                f"model/validation outcome — the real integration was never actually "
                f"exercised: {failure.message}"
            )

        if result["status"] == "completed":
            assert result["validation"] is not None
            assert result["validation"].valid is True

    def test_no_raw_unsanitized_dataset_content_sent_to_real_provider(self, small_synthetic_dataset):
        """
        Confirms — against the REAL provider's request construction,
        not a fake — that the sanitized context (never a raw
        DataFrame) is what actually gets sent. Intercepts the outgoing
        HTTP request the real OllamaProvider builds by pointing it at
        a local capturing server instead of the real Ollama endpoint
        for JUST this one assertion (the request-construction logic is
        identical regardless of which server receives it — this
        isolates "what did OllamaProvider send" from "did Ollama
        respond correctly", which is already covered by the other
        tests in this file).
        """
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        captured = {}

        class CapturingHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                captured["body"] = json.loads(raw.decode("utf-8"))
                response_body = json.dumps({"response": json.dumps({"steps": []})}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), CapturingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            local_provider = OllamaProvider(host=f"http://127.0.0.1:{server.server_port}", model=DEFAULT_LLM_MODEL)

            sanitized_result = build_sanitized_llm_context("dataset_synthetic", "churned", small_synthetic_dataset)
            context = LLMPlanningContext(
                objective="Predict 'churned' from the remaining columns.",
                dataset_context=sanitized_result.data.model_dump(mode="json"),
                allowed_operations=["drop_column"],
            )
            local_provider.generate_plan(context)
        finally:
            server.shutdown()

        prompt_sent = captured["body"]["prompt"]
        original_df = small_synthetic_dataset.get("dataset_synthetic")

        # The raw customer_id VALUES (not the column name, which is
        # legitimately safe metadata) must never appear verbatim as a
        # full row dump -- only the sanitized, capped sample_values
        # (see SanitizedLLMContext) may appear.
        assert "CUST0001" not in prompt_sent or prompt_sent.count("CUST") <= 5  # capped sample, not all 100 rows
        # A crude but meaningful proxy: the full raw dataset serialized
        # (e.g. via to_json/to_csv) would be far larger than what a
        # sanitized context of this dataset's size should produce.
        raw_full_dump_size = len(original_df.to_json())
        assert len(prompt_sent) < raw_full_dump_size
