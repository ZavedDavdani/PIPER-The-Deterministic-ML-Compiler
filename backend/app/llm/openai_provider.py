"""
OpenAIProvider — concrete LLMProvider implementation using the official
OpenAI Python SDK.

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

from app.llm.plan_schema import build_plan_json_schema
from app.llm.prompts import build_planning_prompt, build_replan_prompt
from app.llm.provider import (
    LLMPlanningContext,
    LLMProviderResult,
    ProposedPlan,
    ProviderError,
)

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_TIMEOUT_SECONDS = 120.0

_OPENAI_RESPONSE_FORMAT_BASE = {
    "type": "json_schema",
    "json_schema": {
        "name": "proposed_plan",
        "strict": True,
    },
}


def _strip_markdown_fences(text: str) -> str:
    """
    Defensive cleanup: strips leading/trailing markdown fences (``` or ```json)
    if the entire response is wrapped in them.
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


class OpenAIProvider:
    """
    LLMProvider implementation backed by OpenAI's chat completions API.

    Configuration:
        OPENAI_API_KEY                 (required for live API calls)
        OPENAI_MODEL                   (default: gpt-5.6-luna)
        OPENAI_BASE_URL                (optional custom endpoint/proxy)
        PIPER_OPENAI_TIMEOUT_SECONDS   (default: 120.0)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
        else:
            self.timeout_seconds = float(
                os.environ.get("PIPER_OPENAI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
            )

    def generate_plan(self, context: LLMPlanningContext) -> LLMProviderResult:
        """
        Generates a ProposedPlan from LLMPlanningContext using the OpenAI API.
        Never raises for expected operational failures — returns structured
        LLMProviderResult with error envelope instead.
        """
        if not self.api_key:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="provider_unavailable",
                    message="OpenAI API key is missing. Set the OPENAI_API_KEY environment variable.",
                ),
            )

        try:
            import openai
        except ImportError:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="provider_unavailable",
                    message="The 'openai' Python package is not installed. Install with 'pip install openai'.",
                ),
            )

        is_replan = context.failure_context is not None
        prompt = build_replan_prompt(context) if is_replan else build_planning_prompt(context)
        response_schema = build_plan_json_schema(context.allowed_operations, context.tool_schemas or None)
        response_format = {
            **_OPENAI_RESPONSE_FORMAT_BASE,
            "json_schema": {
                **_OPENAI_RESPONSE_FORMAT_BASE["json_schema"],
                "schema": response_schema,
            },
        }

        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a deterministic ML planning assistant that outputs only valid JSON conforming to the requested schema.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
            )
        except openai.AuthenticationError:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="authentication_error",
                    message="OpenAI authentication failed: invalid API key.",
                ),
            )
        except openai.RateLimitError as exc:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="rate_limit",
                    message=f"OpenAI rate limit exceeded: {exc.message}",
                ),
            )
        except (openai.APITimeoutError, TimeoutError):
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="timeout",
                    message=f"OpenAI did not respond within {self.timeout_seconds}s.",
                ),
            )
        except openai.APIConnectionError as exc:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="provider_unavailable",
                    message=f"Could not connect to OpenAI API: {exc.message}",
                ),
            )
        except openai.APIStatusError as exc:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="http_error",
                    message=f"OpenAI returned HTTP {exc.status_code}: {exc.message}",
                ),
            )
        except Exception:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="provider_unavailable",
                    message="Unexpected OpenAI provider failure.",
                ),
            )

        if not response.choices:
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="malformed_response",
                    message="OpenAI returned an empty choices list.",
                ),
            )

        raw_content = response.choices[0].message.content
        if not raw_content or not raw_content.strip():
            return LLMProviderResult(
                success=False,
                error=ProviderError(
                    code="malformed_response",
                    message="OpenAI returned empty text content in response message.",
                ),
            )

        cleaned_content = _strip_markdown_fences(raw_content)

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
