"""
Tests for generic LLM provider settings and model-agnostic configuration.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.agent.plan_validation import validate_proposed_plan
from app.llm import create_llm_provider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.provider import ProposedPlan, ProposedPlanStep
from app.main import app


@pytest.fixture()
def settings_client():
    with TestClient(app) as client:
        yield client


class TestArbitraryModelConfiguration:
    def test_openai_provider_accepts_arbitrary_model_string(self):
        provider = OpenAIProvider(api_key="sk-test", model="my-custom-openai-model-2026")
        assert provider.model == "my-custom-openai-model-2026"

    def test_ollama_provider_accepts_arbitrary_model_string(self):
        provider = OllamaProvider(model="custom-ollama-tag:latest")
        assert provider.model == "custom-ollama-tag:latest"

    def test_factory_passes_openai_model_from_kwargs(self):
        provider = create_llm_provider("openai", model="arbitrary-gpt-variant", api_key="sk-test")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "arbitrary-gpt-variant"

    def test_factory_passes_ollama_model_from_kwargs(self):
        provider = create_llm_provider("ollama", model="qwen3:8b")
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "qwen3:8b"

    def test_factory_reads_ollama_model_env_alias(self, monkeypatch):
        monkeypatch.delenv("PIPER_LLM_MODEL", raising=False)
        monkeypatch.setenv("OLLAMA_MODEL", "alias-model:7b")
        provider = create_llm_provider("ollama")
        assert provider.model == "alias-model:7b"

    def test_factory_reads_ollama_host_env_alias(self, monkeypatch):
        monkeypatch.delenv("PIPER_OLLAMA_HOST", raising=False)
        monkeypatch.setenv("OLLAMA_HOST", "http://ollama-alias:11434")
        provider = create_llm_provider("ollama")
        assert provider.host == "http://ollama-alias:11434"

    def test_changing_model_does_not_change_deterministic_validation(self):
        plan = ProposedPlan(
            steps=[
                ProposedPlanStep(
                    action="Drop id",
                    tool_name="drop_column",
                    arguments={"column": "customerID", "reason": "identifier"},
                    reasoning="High uniqueness.",
                )
            ]
        )
        result_a = validate_proposed_plan(plan.steps, target_column="Churn")
        provider = create_llm_provider("openai", model="model-a", api_key="sk-test")
        provider.model = "model-b"
        result_b = validate_proposed_plan(plan.steps, target_column="Churn")
        assert result_a == result_b


class TestProviderSettingsApi:
    def test_get_provider_status_is_safe(self, settings_client: TestClient):
        response = settings_client.get("/settings/provider")
        assert response.status_code == 200
        body = response.json()
        assert "provider" in body
        assert "model" in body
        assert "reachable" in body
        assert "available_models" in body
        assert "api_key" not in body
        assert "OPENAI_API_KEY" not in json.dumps(body)

    def test_put_provider_switches_openai_model(self, settings_client: TestClient):
        response = settings_client.put(
            "/settings/provider",
            json={"provider": "openai", "model": "gpt-custom-test-model"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "openai"
        assert body["model"] == "gpt-custom-test-model"
        assert "api_key" not in body
        assert body["details"].get("has_api_key") in (True, False)

        follow_up = settings_client.get("/settings/provider")
        assert follow_up.json()["model"] == "gpt-custom-test-model"

    def test_put_provider_switches_ollama_model(self, settings_client: TestClient):
        response = settings_client.put(
            "/settings/provider",
            json={
                "provider": "ollama",
                "model": "qwen3:8b",
                "host": "http://localhost:11434",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "ollama"
        assert body["model"] == "qwen3:8b"
        assert body["details"]["host"] == "http://localhost:11434"

    def test_ollama_backwards_compatible_get(self, settings_client: TestClient):
        response = settings_client.get("/settings/ollama")
        assert response.status_code == 200
        body = response.json()
        assert "host" in body
        assert "model" in body
        assert "models" in body

    def test_ollama_backwards_compatible_put(self, settings_client: TestClient):
        response = settings_client.put(
            "/settings/ollama",
            json={"model": "legacy-endpoint-model:latest"},
        )
        assert response.status_code == 200
        assert response.json()["model"] == "legacy-endpoint-model:latest"

        provider_status = settings_client.get("/settings/provider").json()
        assert provider_status["model"] == "legacy-endpoint-model:latest"

    def test_put_provider_switches_gemini_model(self, settings_client: TestClient):
        response = settings_client.put(
            "/settings/provider",
            json={"provider": "gemini", "model": "gemini-custom-model"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "gemini"
        assert body["model"] == "gemini-custom-model"
        assert "api_key" not in body
        assert body["details"].get("has_api_key") in (True, False)

    def test_put_provider_rejects_unknown_provider(self, settings_client: TestClient):
        response = settings_client.put(
            "/settings/provider",
            json={"provider": "groq", "model": "llama"},
        )
        assert response.status_code == 422
