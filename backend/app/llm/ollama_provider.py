"""
OllamaProvider (M3 Phase A / Step 2) — concrete LLMProvider talking to
a local Ollama HTTP server.

Uses only the Python standard library (urllib.request) for the HTTP
call — no new dependency needed for a single JSON POST with a bounded
timeout, per the instruction to prefer stdlib where reasonable and
avoid unnecessary dependencies. requirements.txt is unchanged by this
module.

ROBUST RESPONSE PARSING (important, confirmed against real behavior):
manual verification against a real local Ollama 0.32.6 + qwen3:4b
instance found that generated JSON content can appear under different
top-level keys depending on request shape/model behavior — observed
content landing in the `thinking` field with `response` left empty,
not just in `response` as Ollama's documented common case. This module
therefore does NOT assume content lives in any single field. It
inspects the full raw decoded JSON body and searches an ordered list
of candidate locations (`response`, then `thinking`, then a `message.
content` shape for a possible /api/chat-style body) for the first
non-empty string, and treats total absence of any of them as a
malformed_response provider error rather than silently returning
empty content.

This is NOT wired into graph.py or plan_node_v2 in this step — it is
only reachable via direct instantiation and the (later, separately
gated) real-Ollama integration test.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Optional

from pydantic import ValidationError

from app.llm.provider import (
    LLMPlanningContext,
    LLMProviderResult,
    ProposedPlan,
    ProviderError,
)
from app.llm.prompts import build_planning_prompt, build_replan_prompt

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3:4b"
DEFAULT_TIMEOUT_SECONDS = 600.0

DEFAULT_TOTAL_DEADLINE_SECONDS = 900.0
"""
Hard ceiling on the total wall-clock duration of ONE generate_plan()
call, enforced by PIPER rather than by the socket layer.

Why this exists: `urllib`'s `timeout` bounds each individual socket
operation, NOT total request duration. Two real measurements from this
project's benchmark artifacts show the gap is not academic — single
calls of 20,923s (5.8h) and 59,679s (16.6h) against a configured 600s
budget, each ultimately reporting "did not respond within 600.0s".

900s is deliberately ABOVE DEFAULT_TIMEOUT_SECONDS so it is a backstop,
not a new operating limit: every healthy call measured in this project
(~400-460s for qwen3:4b and qwen3:8b first attempts) completes well
inside it, so normal behavior is unchanged. Combined with the existing
bounded retry budget, total planning for a run is provably bounded by
(max_retries + 1) x this value.

Override with PIPER_OLLAMA_TOTAL_DEADLINE_SECONDS.
"""

DEFAULT_KEEP_ALIVE = "10m"
"""
Evidence-based default (engineering-hardening Phase 2A — see CLAUDE.md
for the full controlled experiment). Without an explicit keep_alive,
Ollama unloads a model after its own documented 5-minute idle default —
a real REPLAN gap (deterministic CLEAN/FEATURE_ENGINEER/SPLIT/TRAIN/
EVALUATE/... execution between a PLAN call and a possible follow-up)
can exceed that easily. A controlled real-Ollama experiment on the
real Titanic workload confirmed via `ollama ps` ground truth (not just
inferred from latency) that a 330s gap evicts the model under the
5-minute default but NOT under an explicit longer keep_alive, and the
resulting prompt-processing cost difference was stark: 690.4s
(evicted, cold) vs. 186.0s (retained, warm) for an otherwise identical
call — a 73% wall-time reduction. 10 minutes was chosen to comfortably
exceed a single planning call's own worst-case duration (matches
DEFAULT_TIMEOUT_SECONDS) without keeping the model resident
indefinitely for no reason. Overridable via PIPER_OLLAMA_KEEP_ALIVE
(same override precedence as host/model/timeout_seconds) or the
keep_alive constructor argument.
"""
"""
Evidence-based default (Batch 5 revision — see CLAUDE.md's "current
timeout finding," now resolved). The previous 150.0s default was set
from a 5-run distribution measured against a SMALL, 4-column synthetic
dataset's planning prompt (min 53.45s / median 64.51s / mean 74.72s /
max 123.88s / stdev 28.91s). During Batch 4's Docker verification, a
run against the full 21-column, 7,043-row real Telco CSV genuinely hit
`provider_error_code: timeout` — one observation, not itself
sufficient evidence, but a real signal the small-dataset measurement
didn't account for prompt/dataset size.

Batch 5 collected a proper 5-run distribution against the REAL Telco
CSV (same methodology: real local Ollama 0.32.6 + qwen3:4b, CPU
inference, actual PIPER planning request via
build_sanitized_llm_context() -> LLMPlanningContext ->
OllamaProvider.generate_plan()):

    min:    143.56s
    median: 215.93s
    mean:   247.34s
    max:   418.24s
    stdev:  103.06s
    (5/5 successful)

The old 150.0s default sat BELOW the min of this distribution — every
one of the 5 runs but one would have timed out against a realistically
sized dataset, confirming this was never a rare edge case for
production-scale data. 600.0s covers the observed max (418.24s) with
real margin (~182s, ~44%), sized generously given this distribution's
much higher variance (stdev 103s, vs. 28.91s for the small dataset) —
consistent with the same "real margin above the observed max, still
finite, do not disable the timeout" principle the original 150.0s
value was set under, just against realistic-scale evidence this time.
qwen3:4b is a thinking-mode model and OllamaProvider deliberately uses
stream=False (see generate_plan()'s docstring), so the full timeout
budget must cover the ENTIRE hidden reasoning trace, not just
time-to-first-byte — this, combined with prompt/dataset size, is the
source of both the elevated default and its variance.

Overridable via the PIPER_OLLAMA_TIMEOUT_SECONDS environment variable
or the timeout_seconds constructor argument (constructor argument
takes precedence over the environment variable, matching host/model's
existing override precedence).
"""

# Ordered candidate locations to search for generated text content in
# an Ollama response body, per the robust-parsing rationale in the
# module docstring. Checked in this order; the first non-empty string
# found wins.
_CONTENT_FIELD_CANDIDATES = ("response", "thinking")

# JSON Schema for the Plan shape, passed via Ollama's `format` field
# when requesting structured output — mirrors ProposedPlan's shape
# exactly (not the graph-internal PlanStep) so a schema mismatch here
# can never silently diverge from what this module actually validates
# the response against afterward.
PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "reasoning": {"type": "string"},
                },
                "required": ["action", "tool_name", "arguments", "reasoning"],
            },
        },
    },
    "required": ["steps"],
}


def _extract_content(body: dict) -> Optional[str]:
    """
    Searches the decoded Ollama response body for generated text
    content across the known candidate fields (see module docstring).
    Returns the first non-empty string found, or None if none of the
    candidates contain non-empty content.
    """
    for field in _CONTENT_FIELD_CANDIDATES:
        value = body.get(field)
        if isinstance(value, str) and value.strip():
            return value

    # /api/chat-style shape, checked last since /api/generate (what
    # this provider uses) does not normally produce it — kept as a
    # defensive fallback in case of a future transport change, not
    # because it's expected today.
    message = body.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    return None


def _strip_markdown_fences(text: str) -> str:
    """
    Defensive cleanup only — NOT a regex-based content extractor (per
    the instruction against fragile regex extraction of arbitrary
    text). This only strips a leading/trailing ``` or ```json fence if
    the entire response is wrapped in one; it does not attempt to hunt
    for JSON embedded within other text. If the model ignores the
    "no markdown fences" instruction in REQUIRED_OUTPUT_FORMAT and
    wraps its output, this still allows json.loads() to succeed; if
    the model returns anything else non-JSON, json.loads() below fails
    normally and is reported as malformed_response, exactly as it
    would without this cleanup.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


class OllamaProvider:
    """
    LLMProvider implementation backed by a local Ollama server's
    /api/generate endpoint.

    Configuration (per the M3 instructions):
        PIPER_OLLAMA_HOST            (default: http://localhost:11434)
        PIPER_LLM_MODEL              (default: qwen3:4b)
        PIPER_OLLAMA_TIMEOUT_SECONDS (default: 600.0 — see
                                       DEFAULT_TIMEOUT_SECONDS's
                                       docstring for the measured
                                       evidence behind this default)
        PIPER_OLLAMA_KEEP_ALIVE      (default: "10m" — see
                                       DEFAULT_KEEP_ALIVE's docstring
                                       for the measured evidence behind
                                       this default; Ollama's own
                                       `keep_alive` request field, e.g.
                                       "5m"/"1h"/"-1" for indefinite/
                                       "0" to unload immediately after
                                       each call)

    No API key is required or supported — local Ollama has none.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        keep_alive: Optional[str] = None,
        total_deadline_seconds: Optional[float] = None,
    ) -> None:
        self.host = host or os.environ.get("PIPER_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.model = model or os.environ.get("PIPER_LLM_MODEL", DEFAULT_LLM_MODEL)
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
        else:
            self.timeout_seconds = float(
                os.environ.get("PIPER_OLLAMA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
            )
        self.keep_alive = keep_alive or os.environ.get("PIPER_OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
        # Same override precedence as host/model/timeout_seconds:
        # explicit constructor arg > environment variable > documented default.
        if total_deadline_seconds is not None:
            self.total_deadline_seconds = total_deadline_seconds
        else:
            self.total_deadline_seconds = float(
                os.environ.get(
                    "PIPER_OLLAMA_TOTAL_DEADLINE_SECONDS", DEFAULT_TOTAL_DEADLINE_SECONDS
                )
            )

    def generate_plan(self, context: LLMPlanningContext) -> LLMProviderResult:
        """
        Never raises for ordinary failure modes — every failure path
        (timeout, connection failure, HTTP error, malformed JSON,
        schema-invalid plan) is caught and returned as a structured
        ProviderError, per the LLMProvider protocol's contract.
        """
        is_replan = context.failure_context is not None
        prompt = build_replan_prompt(context) if is_replan else build_planning_prompt(context)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": PLAN_JSON_SCHEMA,
            "keep_alive": self.keep_alive,
        }

        request_body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.host.rstrip('/')}/api/generate",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # --- Total-deadline enforcement ------------------------------
        # urllib's `timeout` bounds each individual SOCKET OPERATION, not
        # the total duration of the request. Measured, not theorised: a
        # single call was observed running 20,923s (5.8h) and another
        # 59,679s (16.6h) before finally reporting "did not respond within
        # 600.0s". That makes the documented 600s budget unenforceable on
        # its own, so PIPER bounds total wall time here.
        #
        # The blocking read runs on a daemon thread and is ABANDONED (not
        # killed — Python cannot safely kill a thread blocked in a socket
        # read) if the deadline passes. It is a daemon, so it can never
        # keep the process alive, and it terminates on its own once the
        # underlying socket timeout fires.
        #
        # This is transport-layer only. It performs no repair, returns no
        # plan, and cannot cause anything unvalidated to execute: a
        # deadline breach produces the same structured `timeout`
        # ProviderError that a socket timeout already produced, so
        # plan_node_v2's existing provider-failure branch handles it
        # unchanged — including carrying previously-validated steps
        # forward via _carried_forward_preserved_steps().
        _transport: dict = {}

        def _do_request() -> None:
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    _transport["raw"] = response.read()
            except BaseException as exc:  # re-raised on the calling thread below
                _transport["exc"] = exc

        worker = threading.Thread(
            target=_do_request,
            name="piper-ollama-request",
            daemon=True,
        )
        worker.start()
        worker.join(self.total_deadline_seconds)

        if worker.is_alive():
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="timeout",
                    message=(
                        f"Ollama exceeded PIPER's total planning deadline of "
                        f"{self.total_deadline_seconds}s (socket-level timeout is "
                        f"{self.timeout_seconds}s, which does not bound total request "
                        f"duration). The request was abandoned; no plan was produced."
                    ),
                ),
            )

        try:
            if "exc" in _transport:
                raise _transport["exc"]
            raw_bytes = _transport["raw"]
        except urllib.error.HTTPError as e:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="http_error",
                    message=f"Ollama returned HTTP {e.code}: {e.reason}",
                ),
            )
        except TimeoutError:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="timeout",
                    message=f"Ollama did not respond within {self.timeout_seconds}s.",
                ),
            )
        except urllib.error.URLError as e:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="provider_unavailable",
                    message=f"Could not reach Ollama at {self.host}: {e.reason}",
                ),
            )

        try:
            body = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="malformed_response",
                    message=f"Ollama response envelope was not valid JSON: {e}",
                    raw_response_excerpt=raw_bytes[:500].decode("utf-8", errors="replace"),
                ),
            )

        if not isinstance(body, dict):
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="malformed_response",
                    message="Ollama response envelope was valid JSON but not a JSON object.",
                    raw_response_excerpt=str(body)[:500],
                ),
            )

        content = _extract_content(body)
        if content is None:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="malformed_response",
                    message=(
                        "Ollama response envelope did not contain generated content in any "
                        f"known field ({', '.join(_CONTENT_FIELD_CANDIDATES)}, or message.content)."
                    ),
                    raw_response_excerpt=json.dumps(body)[:500],
                ),
            )

        cleaned_content = _strip_markdown_fences(content)

        try:
            plan_json = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="malformed_response",
                    message=f"Generated content was not valid JSON: {e}",
                    raw_response_excerpt=cleaned_content[:500],
                ),
            )

        try:
            plan = ProposedPlan.model_validate(plan_json)
        except ValidationError as e:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="invalid_plan_schema",
                    message=f"Generated JSON did not match the Plan schema: {e}",
                    raw_response_excerpt=cleaned_content[:500],
                ),
            )

        return LLMProviderResult(success=True, plan=plan)
