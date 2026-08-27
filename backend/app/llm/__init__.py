"""
PIPER LLM Provider Layer.

Provides the unified LLMProvider interface with concrete implementations:
- GeminiProvider: Cloud LLM provider (selected via PIPER_LLM_PROVIDER=gemini)
- OllamaProvider: Local/private/offline provider (default)
- FakeLLMProvider: Deterministic test double
"""

from __future__ import annotations

from typing import Optional

from app.llm.config import (
    resolve_gemini_model,
    resolve_llm_provider_name,
    resolve_ollama_host,
    resolve_ollama_model,
    resolve_openai_model,
)
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.provider import (
    FakeLLMProvider,
    LLMPlanningContext,
    LLMProvider,
    LLMProviderResult,
    ProposedPlan,
    ProposedPlanStep,
    ProviderError,
)


def create_llm_provider(
    provider_name: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function for instantiating the configured LLMProvider.

    Resolution order:
    1. Explicit provider_name argument (if given, case-insensitive: 'openai' or 'ollama').
    2. PIPER_LLM_PROVIDER environment variable ('openai' or 'ollama').
    3. Otherwise -> default to 'ollama'.
    """
    selected = resolve_llm_provider_name(provider_name)

    if selected == "openai":
        if "model" not in kwargs:
            env_model = resolve_openai_model()
            if env_model is not None:
                kwargs["model"] = env_model
        return OpenAIProvider(**kwargs)
    elif selected == "ollama":
        if "model" not in kwargs:
            env_model = resolve_ollama_model()
            if env_model is not None:
                kwargs["model"] = env_model
        if "host" not in kwargs:
            env_host = resolve_ollama_host()
            if env_host is not None:
                kwargs["host"] = env_host
        return OllamaProvider(**kwargs)
    elif selected == "gemini":
        if "model" not in kwargs:
            env_model = resolve_gemini_model()
            if env_model is not None:
                kwargs["model"] = env_model
        return GeminiProvider(**kwargs)
    elif selected == "fake":
        return FakeLLMProvider(**kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected}'. Supported providers are: 'openai', 'ollama', 'gemini'."
        )


__all__ = [
    "LLMProvider",
    "LLMPlanningContext",
    "LLMProviderResult",
    "ProposedPlan",
    "ProposedPlanStep",
    "ProviderError",
    "FakeLLMProvider",
    "OllamaProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "create_llm_provider",
]
