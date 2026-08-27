"""
Unit tests for OpenAIProvider and LLM Provider Factory.

All unit tests use mocks to remain 100% deterministic and network-independent.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.llm import create_llm_provider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.provider import (
    LLMPlanningContext,
    ProposedPlan,
    ProposedPlanStep,
)


@pytest.fixture()
def sample_context() -> LLMPlanningContext:
    return LLMPlanningContext(
        objective="Predict Survived (binary classification).",
        dataset_context={"columns": ["PassengerId", "Survived", "Age", "Sex"], "rows": 891},
        allowed_operations=["drop_column", "convert_column_type", "impute_missing_values", "encode_categorical_features", "scale_features"],
    )


@pytest.fixture()
def replan_context() -> LLMPlanningContext:
    return LLMPlanningContext(
        objective="Predict Survived (binary classification).",
        dataset_context={"columns": ["PassengerId", "Survived", "Age", "Sex"], "rows": 891},
        allowed_operations=["drop_column", "encode_categorical_features", "scale_features"],
        failure_context={"category": "PLAN_ADEQUACY", "message": "Missing values not addressed."},
        previous_plan_summary={"added": [], "removed": [], "changed": []},
    )


class TestOpenAIProviderUnit:
    def test_missing_api_key_returns_structured_error(self, sample_context, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider(api_key=None)
        result = provider.generate_plan(sample_context)

        assert result.success is False
        assert result.plan is None
        assert result.error is not None
        assert result.error.code == "provider_unavailable"
        assert "API key is missing" in result.error.message

    def test_valid_plan_response_parsed_successfully(self, sample_context):
        valid_json = json.dumps({
            "steps": [
                {
                    "action": "Drop identifier column PassengerId",
                    "tool_name": "drop_column",
                    "arguments": {"column": "PassengerId"},
                    "reasoning": "High uniqueness identifier column.",
                },
                {
                    "action": "Encode categorical column Sex",
                    "tool_name": "encode_categorical_features",
                    "arguments": {"columns": ["Sex"]},
                    "reasoning": "Categorical feature encoding.",
                }
            ]
        })

        mock_choice = MagicMock()
        mock_choice.message.content = valid_json
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        provider = OpenAIProvider(api_key="sk-test-mock-key")

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is True
            assert result.error is None
            assert result.plan is not None
            assert len(result.plan.steps) == 2
            assert result.plan.steps[0].tool_name == "drop_column"
            assert result.plan.steps[1].tool_name == "encode_categorical_features"

    def test_markdown_wrapped_json_parsed_successfully(self, sample_context):
        wrapped_json = "```json\n" + json.dumps({
            "steps": [
                {
                    "action": "Scale Age feature",
                    "tool_name": "scale_features",
                    "arguments": {"columns": ["Age"]},
                    "reasoning": "Numeric scaling.",
                }
            ]
        }) + "\n```"

        mock_choice = MagicMock()
        mock_choice.message.content = wrapped_json
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        provider = OpenAIProvider(api_key="sk-test-mock-key")

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is True
            assert result.plan is not None
            assert len(result.plan.steps) == 1

    def test_malformed_json_returns_structured_error(self, sample_context):
        mock_choice = MagicMock()
        mock_choice.message.content = "{not valid json"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        provider = OpenAIProvider(api_key="sk-test-mock-key")

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.plan is None
            assert result.error is not None
            assert result.error.code == "malformed_response"

    def test_invalid_schema_returns_structured_error(self, sample_context):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"steps": [{"tool_name": 123}]})
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        provider = OpenAIProvider(api_key="sk-test-mock-key")

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.plan is None
            assert result.error is not None
            assert result.error.code == "invalid_plan_schema"

    def test_authentication_error_returns_structured_error(self, sample_context):
        import openai

        provider = OpenAIProvider(api_key="sk-invalid-key")

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
                message="Incorrect API key provided",
                response=mock_response,
                body=None,
            )
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "authentication_error"
            assert "authentication failed" in result.error.message.lower()

    def test_rate_limit_error_returns_structured_error(self, sample_context):
        import openai

        provider = OpenAIProvider(api_key="sk-test-key")

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_client.chat.completions.create.side_effect = openai.RateLimitError(
                message="Rate limit reached",
                response=mock_response,
                body=None,
            )
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "rate_limit"
            assert "rate limit" in result.error.message.lower()

    def test_timeout_error_returns_structured_error(self, sample_context):
        import openai

        provider = OpenAIProvider(api_key="sk-test-key")

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = openai.APITimeoutError(
                request=MagicMock()
            )
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "timeout"

    def test_connection_error_returns_structured_error(self, sample_context):
        import openai

        provider = OpenAIProvider(api_key="sk-test-key")

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
                request=MagicMock()
            )
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "provider_unavailable"

    def test_replan_prompt_used_when_failure_context_present(self, replan_context):
        valid_json = json.dumps({
            "steps": [
                {
                    "action": "Encode Sex",
                    "tool_name": "encode_categorical_features",
                    "arguments": {"columns": ["Sex"]},
                    "reasoning": "Corrected plan.",
                }
            ]
        })

        mock_choice = MagicMock()
        mock_choice.message.content = valid_json
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        provider = OpenAIProvider(api_key="sk-test-key")

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(replan_context)

            assert result.success is True
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            sent_prompt = call_kwargs["messages"][1]["content"]
            assert "=== FAILURE CONTEXT" in sent_prompt
            assert "Missing values not addressed" in sent_prompt


class TestCreateLLMProviderFactory:
    def test_factory_selects_openai_when_explicitly_configured(self):
        provider = create_llm_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_factory_selects_ollama_when_explicitly_configured(self):
        provider = create_llm_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_factory_respects_piper_llm_provider_env_var(self, monkeypatch):
        monkeypatch.setenv("PIPER_LLM_PROVIDER", "openai")
        provider = create_llm_provider()
        assert isinstance(provider, OpenAIProvider)

        monkeypatch.setenv("PIPER_LLM_PROVIDER", "ollama")
        provider = create_llm_provider()
        assert isinstance(provider, OllamaProvider)

    def test_factory_defaults_to_ollama_without_explicit_provider(self, monkeypatch):
        monkeypatch.delenv("PIPER_LLM_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-auto-select")
        provider = create_llm_provider()
        assert isinstance(provider, OllamaProvider)

    def test_factory_invalid_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_provider("invalid_backend_xyz")
