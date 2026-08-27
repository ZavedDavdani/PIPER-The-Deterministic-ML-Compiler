"""
Unit tests for GeminiProvider and factory integration.

All unit tests use mocks to remain deterministic and network-independent.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.llm import create_llm_provider
from app.llm.gemini_provider import GeminiProvider
from app.llm.provider import LLMPlanningContext


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


def _valid_plan_json() -> str:
    return json.dumps({
        "steps": [
            {
                "action": "Drop identifier column PassengerId",
                "tool_name": "drop_column",
                "arguments": {"column": "PassengerId"},
                "reasoning": "High uniqueness identifier column.",
            }
        ]
    })


class TestGeminiProviderUnit:
    def test_missing_api_key_returns_structured_error(self, sample_context, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        provider = GeminiProvider(api_key=None, model="gemini-test-model")
        result = provider.generate_plan(sample_context)

        assert result.success is False
        assert result.error is not None
        assert result.error.code == "provider_unavailable"
        assert "API key is missing" in result.error.message

    def test_missing_model_returns_structured_error(self, sample_context, monkeypatch):
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        provider = GeminiProvider(api_key="test-key", model="")
        result = provider.generate_plan(sample_context)

        assert result.success is False
        assert result.error is not None
        assert result.error.code == "provider_unavailable"
        assert "model is not configured" in result.error.message

    def test_valid_plan_response_parsed_successfully(self, sample_context):
        mock_response = MagicMock()
        mock_response.text = _valid_plan_json()

        provider = GeminiProvider(api_key="test-key", model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is True
            assert result.plan is not None
            assert len(result.plan.steps) == 1
            assert result.plan.steps[0].tool_name == "drop_column"

    def test_malformed_json_returns_structured_error(self, sample_context):
        mock_response = MagicMock()
        mock_response.text = "{not valid json"

        provider = GeminiProvider(api_key="test-key", model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "malformed_response"

    def test_invalid_schema_returns_structured_error(self, sample_context):
        mock_response = MagicMock()
        mock_response.text = json.dumps({"steps": [{"tool_name": 123}]})

        provider = GeminiProvider(api_key="test-key", model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "invalid_plan_schema"

    def test_authentication_error_returns_structured_error(self, sample_context):
        from google.genai import errors

        provider = GeminiProvider(api_key="invalid-key", model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = errors.ClientError(
                401, {"error": {"message": "API key not valid"}}
            )
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "authentication_error"

    def test_rate_limit_error_returns_structured_error(self, sample_context):
        from google.genai import errors

        provider = GeminiProvider(api_key="test-key", model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = errors.ClientError(
                429, {"error": {"message": "Rate limit exceeded"}}
            )
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "rate_limit"

    def test_timeout_error_returns_structured_error(self, sample_context):
        provider = GeminiProvider(api_key="test-key", model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = TimeoutError()
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "timeout"

    def test_connection_failure_returns_structured_error(self, sample_context):
        provider = GeminiProvider(api_key="test-key", model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = ConnectionError("connection failed")
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

            assert result.success is False
            assert result.error is not None
            assert result.error.code == "provider_unavailable"

    def test_planning_prompt_used_for_initial_plan(self, sample_context):
        mock_response = MagicMock()
        mock_response.text = _valid_plan_json()
        provider = GeminiProvider(api_key="test-key", model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider.generate_plan(sample_context)

            call_kwargs = mock_client.models.generate_content.call_args[1]
            assert "FAILURE CONTEXT" not in call_kwargs["contents"]

    def test_replan_prompt_used_when_failure_context_present(self, replan_context):
        mock_response = MagicMock()
        mock_response.text = _valid_plan_json()
        provider = GeminiProvider(api_key="test-key", model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            provider.generate_plan(replan_context)

            call_kwargs = mock_client.models.generate_content.call_args[1]
            assert "=== FAILURE CONTEXT" in call_kwargs["contents"]
            assert "Missing values not addressed" in call_kwargs["contents"]

    def test_arbitrary_model_identifier_accepted(self):
        provider = GeminiProvider(api_key="test-key", model="custom-gemini-model-v9")
        assert provider.model == "custom-gemini-model-v9"

    def test_api_key_never_exposed_in_error_messages(self, sample_context):
        secret = "AIzaSy-test-secret-key"
        from google.genai import errors

        provider = GeminiProvider(api_key=secret, model="gemini-test-model")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = errors.ClientError(
                500, {"error": {"message": "server exploded"}}
            )
            mock_client_cls.return_value = mock_client

            result = provider.generate_plan(sample_context)

        assert result.error is not None
        assert secret not in result.error.message
        assert secret not in str(result.model_dump())


class TestGeminiProviderFactory:
    def test_factory_selects_gemini_when_explicitly_configured(self):
        provider = create_llm_provider("gemini", api_key="test-key", model="gemini-test-model")
        assert isinstance(provider, GeminiProvider)

    def test_factory_respects_piper_llm_provider_env_var(self, monkeypatch):
        monkeypatch.setenv("PIPER_LLM_PROVIDER", "gemini")
        provider = create_llm_provider(api_key="test-key", model="gemini-test-model")
        assert isinstance(provider, GeminiProvider)

    def test_factory_passes_configured_model(self):
        provider = create_llm_provider("gemini", api_key="test-key", model="arbitrary-gemini-model")
        assert provider.model == "arbitrary-gemini-model"
