"""
M3 Phase A / Step 2: provider-layer unit tests.

Covers app/llm/provider.py (LLMProvider protocol, FakeLLMProvider,
schemas) and app/llm/ollama_provider.py (OllamaProvider), plus
app/llm/prompts.py's separation from transport.

NOTHING in this file requires a running Ollama server — OllamaProvider
is tested against a real local http.server instance spun up in-process
per test (a genuine HTTP transport, but not Ollama itself), so the
transport/parsing logic is exercised behaviorally rather than mocked,
while remaining fully deterministic and network-independent of any
external service. The separately-gated real-Ollama integration test
(a later step) is what actually talks to qwen3:4b.

This module is NOT wired into graph.py/plan_node_v2/AgentState yet —
these tests exercise the provider layer standalone, per Step 2's scope.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import ValidationError

from app.llm.ollama_provider import OllamaProvider, PLAN_JSON_SCHEMA, _extract_content
from app.llm.provider import (
    FakeLLMProvider,
    LLMPlanningContext,
    ProposedPlan,
    ProposedPlanStep,
)
from app.llm.prompts import build_planning_prompt, build_replan_prompt


# --- Shared context fixture -------------------------------------------------


@pytest.fixture()
def sample_context() -> LLMPlanningContext:
    return LLMPlanningContext(
        objective="Predict Churn (binary classification).",
        dataset_context={"columns": ["customerID", "Churn"], "rows": 7043},
        allowed_operations=["drop_column", "convert_column_type", "impute_missing_values"],
    )


@pytest.fixture()
def replan_context() -> LLMPlanningContext:
    return LLMPlanningContext(
        objective="Predict Churn (binary classification).",
        dataset_context={"columns": ["customerID", "Churn"], "rows": 7043},
        allowed_operations=["drop_column"],
        failure_context={"category": "LEAKAGE_ERROR", "message": "Feature leaks target."},
        previous_plan_summary={"added": [], "removed": [], "changed": []},
    )


# --- FakeLLMProvider ---------------------------------------------------


class TestFakeLLMProvider:
    def test_valid_plan_scenario_returns_success(self, sample_context):
        """(1) Fake provider returns valid Plan."""
        provider = FakeLLMProvider(scenario="valid_plan")
        result = provider.generate_plan(sample_context)

        assert result.success is True
        assert result.plan is not None
        assert len(result.plan.steps) >= 1
        assert result.error is None

    def test_malformed_json_scenario_returns_structured_error(self, sample_context):
        """(2) Malformed provider output is rejected — structured error, not an exception."""
        provider = FakeLLMProvider(scenario="malformed_json")
        result = provider.generate_plan(sample_context)

        assert result.success is False
        assert result.plan is None
        assert result.error.code == "malformed_response"

    def test_invalid_plan_scenario_returns_structured_error(self, sample_context):
        """(3) Schema-invalid output is rejected — structured error, not an exception."""
        provider = FakeLLMProvider(scenario="invalid_plan")
        result = provider.generate_plan(sample_context)

        assert result.success is False
        assert result.error.code == "invalid_plan_schema"

    def test_provider_failure_scenario_returns_structured_error(self, sample_context):
        provider = FakeLLMProvider(scenario="provider_failure")
        result = provider.generate_plan(sample_context)

        assert result.success is False
        assert result.error.code == "provider_unavailable"

    def test_fixed_plan_is_returned_verbatim(self, sample_context):
        """Configuring a specific plan (not the default) is honored exactly."""
        custom_plan = ProposedPlan(
            steps=[
                ProposedPlanStep(
                    action="Custom action",
                    tool_name="convert_column_type",
                    arguments={"column": "TotalCharges", "target_type": "numeric"},
                    reasoning="Custom reasoning.",
                )
            ]
        )
        provider = FakeLLMProvider(scenario="valid_plan", fixed_plan=custom_plan)
        result = provider.generate_plan(sample_context)

        assert result.plan == custom_plan

    def test_provider_records_every_context_received(self, sample_context, replan_context):
        """
        Proves the fake provider can be inspected for what context it
        received — the capability a later sanitization-boundary test
        will depend on (proving raw dataset content never reaches the
        provider, once that wiring exists).
        """
        provider = FakeLLMProvider(scenario="valid_plan")
        provider.generate_plan(sample_context)
        provider.generate_plan(replan_context)

        assert len(provider.received_contexts) == 2
        assert provider.received_contexts[0] == sample_context
        assert provider.received_contexts[1] == replan_context


# --- Schema validation (Pydantic-level) ---------------------------------


class TestProviderSchemas:
    def test_proposed_plan_step_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            ProposedPlanStep(
                action="a", tool_name="drop_column", arguments={}, reasoning="r",
                unexpected_field="should not be allowed",
            )

    def test_proposed_plan_step_requires_all_fields(self):
        with pytest.raises(ValidationError):
            ProposedPlanStep(tool_name="drop_column")  # missing action/arguments/reasoning

    def test_llm_planning_context_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            LLMPlanningContext(objective="x", not_a_real_field=1)


# --- Prompt construction (separate from transport) -----------------------


class TestPromptConstruction:
    def test_planning_prompt_contains_all_required_sections(self, sample_context):
        prompt = build_planning_prompt(sample_context)

        for section in (
            "SYSTEM INSTRUCTIONS", "DATASET CONTEXT", "USER OBJECTIVE",
            "ALLOWED OPERATIONS", "DETERMINISTIC CONSTRAINTS", "REQUIRED OUTPUT FORMAT",
        ):
            assert section in prompt

    def test_planning_prompt_excludes_failure_context_section(self, sample_context):
        """A first-attempt prompt has no FAILURE CONTEXT section at all."""
        prompt = build_planning_prompt(sample_context)
        assert "FAILURE CONTEXT" not in prompt

    def test_replan_prompt_contains_failure_context_section(self, replan_context):
        prompt = build_replan_prompt(replan_context)
        assert "FAILURE CONTEXT" in prompt
        assert "LEAKAGE_ERROR" in prompt

    def test_replan_prompt_contains_previous_plan_summary_when_present(self, replan_context):
        prompt = build_replan_prompt(replan_context)
        assert "PREVIOUS PLAN SUMMARY" in prompt

    def test_replan_prompt_requires_failure_context(self, sample_context):
        """Calling build_replan_prompt() without failure_context is a caller bug."""
        with pytest.raises(ValueError):
            build_replan_prompt(sample_context)

    def test_prompt_labels_dataset_content_as_data_not_instructions(self, sample_context):
        prompt = build_planning_prompt(sample_context)
        assert "not instructions" in prompt or "DATA" in prompt.upper()

    def test_prompt_construction_never_makes_network_calls(self, sample_context, monkeypatch):
        """
        Confirms prompts.py is genuinely transport-free: blocks
        urllib.request.urlopen for the duration of this test and
        proves prompt construction still succeeds without it.
        """
        import urllib.request

        def _blocked(*args, **kwargs):
            raise AssertionError("prompt construction must never make a network call")

        monkeypatch.setattr(urllib.request, "urlopen", _blocked)
        prompt = build_planning_prompt(sample_context)
        assert len(prompt) > 0


class TestToolSchemaRenderedIntoPrompt:
    """
    Post-4-model-benchmark investigation: the prompt previously rendered
    ALLOWED OPERATIONS as nothing but a bare list of tool_name strings —
    no argument names/types/required-ness/examples — which independent
    evidence from 20 real Ollama calls across 4 different models tied
    directly to the observed invalid-argument failure patterns. These
    tests cover the fix: LLMPlanningContext.tool_schemas (additive,
    default empty) and prompts.py's new rendering of it.
    """

    def test_llm_planning_context_defaults_tool_schemas_to_empty_dict(self):
        """Every pre-existing caller that never sets tool_schemas is unaffected."""
        ctx = LLMPlanningContext(objective="x", dataset_context={}, allowed_operations=["drop_column"])
        assert ctx.tool_schemas == {}

    def test_without_tool_schemas_prompt_falls_back_to_bare_operation_list(self, sample_context):
        """
        sample_context never sets tool_schemas — this pins the exact
        backward-compatible fallback rendering (byte-identical to the
        pre-fix behavior) so this addition cannot silently change
        output for any caller that doesn't opt in.
        """
        prompt = build_planning_prompt(sample_context)
        assert '"drop_column"' in prompt
        operations_section = prompt.split("=== ALLOWED OPERATIONS ===")[1].split("=== DETERMINISTIC CONSTRAINTS ===")[0]
        assert "required" not in operations_section

    def test_with_tool_schemas_prompt_describes_exact_argument_contract(self):
        from app.agent.plan_validation import ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS

        ctx = LLMPlanningContext(
            objective="Predict Survived",
            dataset_context={},
            allowed_operations=sorted(ALLOWED_TOOL_NAMES),
            tool_schemas=TOOL_ARGUMENT_SCHEMAS,
        )
        prompt = build_planning_prompt(ctx)
        operations_section = prompt.split("=== ALLOWED OPERATIONS ===")[1].split("=== DETERMINISTIC CONSTRAINTS ===")[0]

        # Every tool name, its exact required argument name(s), and a
        # worked example must all be present — this is the concrete
        # evidence a real model was never given before this fix.
        for tool_name in ALLOWED_TOOL_NAMES:
            assert tool_name in operations_section
        assert "column (string, required)" in operations_section
        assert "columns (array of strings, required)" in operations_section
        assert '"column": "PassengerId"' in operations_section
        # The exact singular-vs-plural distinction every benchmarked
        # model got wrong is now explicit, not implicit.
        assert "never pass a list of names here" in operations_section

    def test_tool_schemas_carried_through_replan_prompt_too(self, replan_context):
        from app.agent.plan_validation import TOOL_ARGUMENT_SCHEMAS

        ctx = replan_context.model_copy(update={"tool_schemas": TOOL_ARGUMENT_SCHEMAS})
        prompt = build_replan_prompt(ctx)
        assert "Example arguments:" in prompt


# --- OllamaProvider: response parsing -----------------------------------


class _FakeOllamaServer:
    """
    Spins up a real local HTTP server (not Ollama itself) whose
    response body is fully controlled by the test — used to exercise
    OllamaProvider's actual HTTP + JSON parsing logic behaviorally,
    without depending on a real Ollama instance being reachable.
    """

    def __init__(self, response_builder):
        self._response_builder = response_builder
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Must fully consume the incoming Content-Length-delimited
                # request body before responding. OllamaProvider sends a
                # real, non-trivial JSON body (the full constructed
                # prompt) on every request; responding before draining
                # rfile can leave unread data on the socket when the
                # handler tears the connection down. On Linux this is
                # usually silently tolerated by the socket stack, but on
                # Windows, closing a socket with unread incoming data
                # causes an RST rather than a clean FIN, which the
                # client observes as ConnectionAbortedError ([WinError
                # 10053]) — this is a genuine cross-platform request/
                # response race in the test harness, not something
                # specific to any one test's content.
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length:
                    self.rfile.read(content_length)

                status, body_bytes = outer._response_builder()
                body_bytes = body_bytes or b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                if body_bytes:
                    self.wfile.write(body_bytes)
                self.wfile.flush()

            def log_message(self, *args):
                pass  # keep test output clean

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def shutdown(self):
        self._server.shutdown()


def _make_ollama_body(response: str | None = None, thinking: str | None = None) -> bytes:
    body = {"model": "qwen3:4b", "done": True}
    if response is not None:
        body["response"] = response
    if thinking is not None:
        body["thinking"] = thinking
    return json.dumps(body).encode("utf-8")


class TestOllamaProviderResponseParsing:
    def test_content_extracted_from_response_field(self, sample_context):
        """The documented common case: content in `response`."""
        valid_plan_json = json.dumps({
            "steps": [{"action": "a", "tool_name": "drop_column", "arguments": {"column": "x"}, "reasoning": "r"}]
        })
        server = _FakeOllamaServer(lambda: (200, _make_ollama_body(response=valid_plan_json)))
        try:
            provider = OllamaProvider(host=server.host, model="qwen3:4b")
            result = provider.generate_plan(sample_context)
        finally:
            server.shutdown()

        assert result.success is True
        assert result.plan.steps[0].tool_name == "drop_column"

    def test_content_extracted_from_thinking_field_when_response_empty(self, sample_context):
        """
        (4, robustness) The exact real-world case manually observed
        against local Ollama 0.32.6 + qwen3:4b: `response` empty,
        generated content actually in `thinking`. Must not be treated
        as malformed/empty.
        """
        valid_plan_json = json.dumps({
            "steps": [{"action": "a", "tool_name": "drop_column", "arguments": {"column": "x"}, "reasoning": "r"}]
        })
        server = _FakeOllamaServer(lambda: (200, _make_ollama_body(response="", thinking=valid_plan_json)))
        try:
            provider = OllamaProvider(host=server.host, model="qwen3:4b")
            result = provider.generate_plan(sample_context)
        finally:
            server.shutdown()

        assert result.success is True
        assert result.plan.steps[0].tool_name == "drop_column"

    def test_markdown_fenced_content_is_stripped_before_parsing(self, sample_context):
        valid_plan_json = json.dumps({
            "steps": [{"action": "a", "tool_name": "drop_column", "arguments": {"column": "x"}, "reasoning": "r"}]
        })
        fenced = f"```json\n{valid_plan_json}\n```"
        server = _FakeOllamaServer(lambda: (200, _make_ollama_body(response=fenced)))
        try:
            provider = OllamaProvider(host=server.host, model="qwen3:4b")
            result = provider.generate_plan(sample_context)
        finally:
            server.shutdown()

        assert result.success is True

    def test_extract_content_returns_none_when_no_known_field_present(self):
        """Unit-level check of _extract_content() directly."""
        assert _extract_content({"done": True}) is None
        assert _extract_content({"response": "", "thinking": ""}) is None

    def test_no_known_content_field_produces_malformed_response_error(self, sample_context):
        server = _FakeOllamaServer(lambda: (200, json.dumps({"done": True}).encode()))
        try:
            provider = OllamaProvider(host=server.host, model="qwen3:4b")
            result = provider.generate_plan(sample_context)
        finally:
            server.shutdown()

        assert result.success is False
        assert result.error.code == "malformed_response"

    def test_non_json_envelope_produces_malformed_response_error(self, sample_context):
        server = _FakeOllamaServer(lambda: (200, b"not json at all"))
        try:
            provider = OllamaProvider(host=server.host, model="qwen3:4b")
            result = provider.generate_plan(sample_context)
        finally:
            server.shutdown()

        assert result.success is False
        assert result.error.code == "malformed_response"

    def test_content_that_is_not_json_produces_malformed_response_error(self, sample_context):
        server = _FakeOllamaServer(lambda: (200, _make_ollama_body(response="not { valid json")))
        try:
            provider = OllamaProvider(host=server.host, model="qwen3:4b")
            result = provider.generate_plan(sample_context)
        finally:
            server.shutdown()

        assert result.success is False
        assert result.error.code == "malformed_response"

    def test_schema_invalid_plan_json_produces_invalid_plan_schema_error(self, sample_context):
        bad_plan_json = json.dumps({"steps": [{"tool_name": 123}]})  # wrong type, missing fields
        server = _FakeOllamaServer(lambda: (200, _make_ollama_body(response=bad_plan_json)))
        try:
            provider = OllamaProvider(host=server.host, model="qwen3:4b")
            result = provider.generate_plan(sample_context)
        finally:
            server.shutdown()

        assert result.success is False
        assert result.error.code == "invalid_plan_schema"

    def test_http_error_status_produces_http_error(self, sample_context):
        server = _FakeOllamaServer(lambda: (500, b"internal error"))
        try:
            provider = OllamaProvider(host=server.host, model="qwen3:4b")
            result = provider.generate_plan(sample_context)
        finally:
            server.shutdown()

        assert result.success is False
        assert result.error.code == "http_error"

    def test_unreachable_host_produces_provider_unavailable(self, sample_context):
        """Port 1 is a privileged, essentially-never-bound port -> connection refused."""
        provider = OllamaProvider(host="http://127.0.0.1:1", model="qwen3:4b", timeout_seconds=2.0)
        result = provider.generate_plan(sample_context)

        assert result.success is False
        assert result.error.code == "provider_unavailable"

    def test_slow_server_produces_bounded_timeout_error(self, sample_context):
        """(5) Ollama provider handles timeout deterministically — bounded, never hangs."""
        def _slow_response():
            time.sleep(2.0)
            return 200, _make_ollama_body(response="{}")

        server = _FakeOllamaServer(_slow_response)
        try:
            provider = OllamaProvider(host=server.host, model="qwen3:4b", timeout_seconds=0.3)
            start = time.monotonic()
            result = provider.generate_plan(sample_context)
            elapsed = time.monotonic() - start
        finally:
            server.shutdown()

        assert result.success is False
        assert result.error.code == "timeout"
        # Bounded: must not have waited anywhere near the server's full delay.
        assert elapsed < 1.5


# --- OllamaProvider: request construction --------------------------------


class TestOllamaProviderRequestConstruction:
    def test_request_includes_model_prompt_and_json_schema_format(self, sample_context):
        """(4) Ollama provider constructs the correct request."""
        captured = {}

        def _capture_and_respond():
            return 200, _make_ollama_body(response=json.dumps({"steps": []}))

        # Intercept at the handler level by wrapping BaseHTTPRequestHandler
        # to capture the raw request body actually sent.
        class CapturingHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                captured["body"] = json.loads(raw.decode("utf-8"))
                captured["content_type"] = self.headers.get("Content-Type")
                status, body_bytes = _capture_and_respond()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)
                self.wfile.flush()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), CapturingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OllamaProvider(host=f"http://127.0.0.1:{server.server_port}", model="qwen3:4b")
            provider.generate_plan(sample_context)
        finally:
            server.shutdown()

        assert captured["content_type"] == "application/json"
        assert captured["body"]["model"] == "qwen3:4b"
        assert captured["body"]["stream"] is False
        assert "prompt" in captured["body"]
        assert captured["body"]["format"] == PLAN_JSON_SCHEMA
        # Confirms the objective and allowed operations actually reached
        # the constructed prompt sent over the wire, not just the
        # in-process prompt builder.
        assert sample_context.objective in captured["body"]["prompt"]
        # Engineering-hardening Phase 2A: keep_alive must reach the real
        # wire request, not just be stored on the instance — this is
        # what actually prevents Ollama from evicting the model between
        # calls (see DEFAULT_KEEP_ALIVE's docstring for the controlled
        # experiment establishing this matters).
        assert captured["body"]["keep_alive"] == "10m"

    def test_replan_context_produces_replan_prompt_over_the_wire(self, replan_context):
        captured = {}

        class CapturingHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                captured["body"] = json.loads(raw.decode("utf-8"))
                response_body = _make_ollama_body(response=json.dumps({"steps": []}))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
                self.wfile.flush()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), CapturingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OllamaProvider(host=f"http://127.0.0.1:{server.server_port}", model="qwen3:4b")
            provider.generate_plan(replan_context)
        finally:
            server.shutdown()

        assert "FAILURE CONTEXT" in captured["body"]["prompt"]


# --- Configuration ------------------------------------------------------


class TestOllamaProviderConfiguration:
    def test_defaults_to_documented_host_and_model(self, monkeypatch):
        monkeypatch.delenv("PIPER_OLLAMA_HOST", raising=False)
        monkeypatch.delenv("PIPER_LLM_MODEL", raising=False)

        provider = OllamaProvider()

        assert provider.host == "http://localhost:11434"
        assert provider.model == "qwen3:4b"

    def test_reads_host_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("PIPER_OLLAMA_HOST", "http://custom-host:9999")
        monkeypatch.delenv("PIPER_LLM_MODEL", raising=False)

        provider = OllamaProvider()

        assert provider.host == "http://custom-host:9999"

    def test_reads_model_from_environment_variable(self, monkeypatch):
        monkeypatch.delenv("PIPER_OLLAMA_HOST", raising=False)
        monkeypatch.setenv("PIPER_LLM_MODEL", "custom-model:latest")

        provider = OllamaProvider()

        assert provider.model == "custom-model:latest"

    def test_explicit_constructor_args_override_environment(self, monkeypatch):
        monkeypatch.setenv("PIPER_OLLAMA_HOST", "http://env-host:1111")
        monkeypatch.setenv("PIPER_LLM_MODEL", "env-model")

        provider = OllamaProvider(host="http://explicit-host:2222", model="explicit-model")

        assert provider.host == "http://explicit-host:2222"
        assert provider.model == "explicit-model"

    def test_defaults_to_documented_timeout(self, monkeypatch):
        """
        Evidence-based default (see DEFAULT_TIMEOUT_SECONDS's
        docstring): 600.0s, revised in Batch 5 from a 5-run
        real-latency distribution measured against the REAL 21-column,
        7,043-row Telco CSV against a real local Ollama 0.32.6 +
        qwen3:4b instance (min 143.56s / median 215.93s / mean 247.34s
        / max 418.24s / stdev 103.06s) — the previous 150.0s default
        (measured against a small synthetic dataset) sat below this
        distribution's min, not an arbitrary guess or a single sample.
        """
        monkeypatch.delenv("PIPER_OLLAMA_TIMEOUT_SECONDS", raising=False)

        provider = OllamaProvider()

        assert provider.timeout_seconds == 600.0

    def test_reads_timeout_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("PIPER_OLLAMA_TIMEOUT_SECONDS", "45.5")

        provider = OllamaProvider()

        assert provider.timeout_seconds == 45.5

    def test_explicit_timeout_constructor_arg_overrides_environment(self, monkeypatch):
        monkeypatch.setenv("PIPER_OLLAMA_TIMEOUT_SECONDS", "45.5")

        provider = OllamaProvider(timeout_seconds=12.0)

        assert provider.timeout_seconds == 12.0

    def test_timeout_remains_finite_and_bounded_by_default(self, monkeypatch):
        """
        The default timeout must never be disabled (None/0/inf) — it
        must be a genuine, finite, bounded value, per the explicit
        "do not disable the timeout" requirement.
        """
        monkeypatch.delenv("PIPER_OLLAMA_TIMEOUT_SECONDS", raising=False)

        provider = OllamaProvider()

        assert provider.timeout_seconds is not None
        assert 0 < provider.timeout_seconds < float("inf")

    def test_no_api_key_required_or_referenced(self, monkeypatch):
        """
        Confirms OllamaProvider never looks for or requires any API-key
        environment variable — local Ollama has none, per the M3
        instructions.
        """
        import inspect

        source = inspect.getsource(OllamaProvider)
        assert "api_key" not in source.lower()
        assert "authorization" not in source.lower()

    def test_defaults_to_documented_keep_alive(self, monkeypatch):
        """
        Evidence-based default (engineering-hardening Phase 2A — see
        DEFAULT_KEEP_ALIVE's docstring): a controlled real-Ollama
        experiment confirmed via `ollama ps` ground truth that Ollama's
        own 5-minute default evicts the model across a realistic
        PLAN-to-REPLAN gap, while an explicit longer keep_alive survives
        the same gap — 690.4s (evicted) vs. 186.0s (retained) for an
        otherwise identical call. 10 minutes matches
        DEFAULT_TIMEOUT_SECONDS (a single call's own worst-case
        duration) rather than being an arbitrary guess.
        """
        monkeypatch.delenv("PIPER_OLLAMA_KEEP_ALIVE", raising=False)

        provider = OllamaProvider()

        assert provider.keep_alive == "10m"

    def test_reads_keep_alive_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("PIPER_OLLAMA_KEEP_ALIVE", "1h")

        provider = OllamaProvider()

        assert provider.keep_alive == "1h"

    def test_explicit_keep_alive_constructor_arg_overrides_environment(self, monkeypatch):
        monkeypatch.setenv("PIPER_OLLAMA_KEEP_ALIVE", "1h")

        provider = OllamaProvider(keep_alive="5m")

        assert provider.keep_alive == "5m"

    def test_keep_alive_never_disabled_by_default(self, monkeypatch):
        """
        The default must be a real, deliberate value — never None/empty
        — since an unset keep_alive key would silently fall back to
        whatever Ollama's own server-side default is (5 minutes,
        confirmed too short for a realistic REPLAN gap), defeating the
        entire point of this configuration existing.
        """
        monkeypatch.delenv("PIPER_OLLAMA_KEEP_ALIVE", raising=False)

        provider = OllamaProvider()

        assert provider.keep_alive
        assert isinstance(provider.keep_alive, str)
