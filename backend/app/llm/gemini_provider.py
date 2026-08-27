"""
GeminiProvider — concrete LLMProvider implementation using Google's official
google-genai SDK.

Conforms to the unified LLMProvider protocol (app.llm.provider.LLMProvider):
- Generates structured ProposedPlan proposals only.
- Never executes code or mutates AgentState directly.
- Maps all network/authentication/rate-limit/parsing failures into structured
  ProviderError instances, never letting uncontrolled exceptions escape.
- Keeps credentials strictly private: API keys are never printed, logged,
  or exposed in error messages.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import ValidationError

from app.llm.config import resolve_gemini_model
from app.llm.plan_schema import build_plan_json_schema
from app.llm.prompts import build_planning_prompt, build_replan_prompt
from app.llm.provider import (
    LLMPlanningContext,
    LLMProviderResult,
    ProposedPlan,
    ProviderError,
)

DEFAULT_TIMEOUT_SECONDS = 120.0

_SYSTEM_INSTRUCTION = (
    "You are a deterministic ML planning assistant that outputs only valid JSON "
    "conforming to the requested schema."
)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _map_gemini_exception(exc: Exception, timeout_seconds: float) -> ProviderError:
    from google.genai import errors

    if isinstance(exc, TimeoutError):
        return ProviderError(
            code="timeout",
            message=f"Gemini did not respond within {timeout_seconds}s.",
        )

    if isinstance(exc, errors.ClientError):
        code = getattr(exc, "code", None)
        if code in (401, 403):
            return ProviderError(
                code="authentication_error",
                message="Gemini authentication failed: invalid API key.",
            )
        if code == 429:
            return ProviderError(
                code="rate_limit",
                message=f"Gemini rate limit exceeded: {exc}",
            )
        return ProviderError(
            code="http_error",
            message=f"Gemini returned HTTP {code}: {exc}",
        )

    if isinstance(exc, errors.ServerError):
        code = getattr(exc, "code", None)
        return ProviderError(
            code="http_error",
            message=f"Gemini server error (HTTP {code}): {exc}",
        )

    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        if code in (401, 403):
            return ProviderError(
                code="authentication_error",
                message="Gemini authentication failed: invalid API key.",
            )
        if code == 429:
            return ProviderError(
                code="rate_limit",
                message=f"Gemini rate limit exceeded: {exc}",
            )
        if code == 408:
            return ProviderError(
                code="timeout",
                message=f"Gemini did not respond within {timeout_seconds}s.",
            )
        return ProviderError(
            code="http_error",
            message=f"Gemini API error (HTTP {code}): {exc}",
        )

    message = str(exc).lower()
    if "timeout" in message or "timed out" in message or "deadline" in message:
        return ProviderError(
            code="timeout",
            message=f"Gemini did not respond within {timeout_seconds}s.",
        )
    if "connection" in message or "connect" in message:
        return ProviderError(
            code="provider_unavailable",
            message="Could not connect to Gemini API.",
        )

    return ProviderError(
        code="provider_unavailable",
        message="Unexpected Gemini provider failure.",
    )


class GeminiProvider:
    """
    LLMProvider implementation backed by Google's Gemini API (google-genai).

    Configuration:
        GEMINI_API_KEY                 (required for live API calls)
        GEMINI_MODEL                   (required — any valid Gemini model id)
        PIPER_GEMINI_TIMEOUT_SECONDS   (default: 120.0)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        resolved_model = model or resolve_gemini_model()
        self.model = resolved_model or ""
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
        else:
            self.timeout_seconds = float(
                os.environ.get("PIPER_GEMINI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
            )

    def generate_plan(self, context: LLMPlanningContext) -> LLMProviderResult:
        if not self.api_key:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="provider_unavailable",
                    message="Gemini API key is missing. Set the GEMINI_API_KEY environment variable.",
                ),
            )

        if not self.model.strip():
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="provider_unavailable",
                    message="Gemini model is not configured. Set the GEMINI_MODEL environment variable.",
                ),
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="provider_unavailable",
                    message=(
                        "The 'google-genai' Python package is not installed. "
                        "Install with 'pip install google-genai'."
                    ),
                ),
            )

        is_replan = context.failure_context is not None
        prompt = build_replan_prompt(context) if is_replan else build_planning_prompt(context)
        response_schema = build_plan_json_schema(context.allowed_operations, context.tool_schemas or None)

        try:
            client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
            )
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                ),
            )
        except Exception as exc:
            return LLMProviderResult(success=False, error=_map_gemini_exception(exc, self.timeout_seconds))

        raw_content = getattr(response, "text", None)
        if not raw_content or not str(raw_content).strip():
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="malformed_response",
                    message="Gemini returned empty text content in response.",
                ),
            )

        cleaned_content = _strip_markdown_fences(str(raw_content))

        try:
            plan_json = json.loads(cleaned_content)
        except json.JSONDecodeError as exc:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="malformed_response",
                    message=f"Generated content was not valid JSON: {exc}",
                    raw_response_excerpt=cleaned_content[:500],
                ),
            )

        try:
            plan = ProposedPlan.model_validate(plan_json)
        except ValidationError as exc:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="invalid_plan_schema",
                    message=f"Generated JSON did not match the Plan schema: {exc}",
                    raw_response_excerpt=cleaned_content[:500],
                ),
            )

        return LLMProviderResult(success=True, plan=plan)

    def list_available_models(self, timeout_seconds: float = 10.0) -> tuple[bool, list[str], Optional[str]]:
        """Probe Gemini model listing. Never exposes the API key."""
        if not self.api_key:
            return False, [], "GEMINI_API_KEY is not set."

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return False, [], "google-genai package is not installed."

        try:
            client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
            )
            models: list[str] = []
            for item in client.models.list():
                name = getattr(item, "name", None)
                if name:
                    models.append(str(name).removeprefix("models/"))
            return True, models, None
        except Exception as exc:
            return False, [], str(exc)
