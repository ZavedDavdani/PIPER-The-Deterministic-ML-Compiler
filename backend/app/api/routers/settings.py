"""
Local operator settings: LLM provider/model configuration and status.

Changing provider or model reconstructs the appropriate LLMProvider on
app.state — never touches validate_proposed_plan(), adequacy, or graph
routing. API keys and other secrets are never exposed to clients.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.llm import create_llm_provider
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama_provider import (
    DEFAULT_KEEP_ALIVE,
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_HOST,
    OllamaProvider,
)
from app.llm.openai_provider import OpenAIProvider

router = APIRouter(prefix="/settings", tags=["settings"])


class ProviderStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    reachable: bool
    available_models: list[str] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)
    error: Optional[str] = None


class ProviderConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "ollama", "gemini"]
    model: str = Field(..., min_length=1)
    host: Optional[str] = Field(default=None, min_length=1)
    base_url: Optional[str] = Field(default=None)


class OllamaStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str
    model: str
    keep_alive: str
    reachable: bool
    models: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class OllamaConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = Field(default=None, min_length=1)
    host: Optional[str] = Field(default=None, min_length=1)


def _provider_snapshot(provider) -> tuple[str, str, str]:
    host = getattr(provider, "host", DEFAULT_OLLAMA_HOST)
    model = getattr(provider, "model", DEFAULT_LLM_MODEL)
    keep_alive = getattr(provider, "keep_alive", DEFAULT_KEEP_ALIVE)
    return host, model, keep_alive


def _get_provider(request: Request):
    return getattr(request.app.state, "llm_provider", None) or create_llm_provider()


def _build_provider_status(provider) -> ProviderStatusResponse:
    if isinstance(provider, OpenAIProvider):
        has_key = bool(provider.api_key)
        return ProviderStatusResponse(
            provider="openai",
            model=provider.model,
            reachable=has_key,
            available_models=[],
            details={
                "has_api_key": has_key,
                "base_url": provider.base_url,
            },
            error=None if has_key else "OPENAI_API_KEY is not set.",
        )

    if isinstance(provider, OllamaProvider):
        reachable, models, error = provider.list_local_models()
        return ProviderStatusResponse(
            provider="ollama",
            model=provider.model,
            reachable=reachable,
            available_models=models,
            details={
                "host": provider.host,
                "keep_alive": provider.keep_alive,
            },
            error=error,
        )

    if isinstance(provider, GeminiProvider):
        has_key = bool(provider.api_key)
        models: list[str] = []
        if has_key:
            _, models, _ = provider.list_available_models()
        return ProviderStatusResponse(
            provider="gemini",
            model=provider.model,
            reachable=has_key,
            available_models=models,
            details={"has_api_key": has_key},
            error=None if has_key else "Gemini API key is not configured.",
        )

    name = getattr(provider, "__class__", type(provider)).__name__
    return ProviderStatusResponse(
        provider=name,
        model=getattr(provider, "model", "unknown"),
        reachable=True,
        available_models=[],
        details={},
        error=None,
    )


def _rebuild_provider(
    provider_name: str,
    model: str,
    *,
    host: Optional[str] = None,
    base_url: Optional[str] = None,
    current_provider=None,
) -> object:
    if provider_name == "openai":
        timeout = getattr(current_provider, "timeout_seconds", None) if isinstance(
            current_provider, OpenAIProvider
        ) else None
        return OpenAIProvider(
            model=model,
            base_url=base_url if base_url is not None else getattr(current_provider, "base_url", None),
            timeout_seconds=timeout,
        )

    if provider_name == "gemini":
        timeout = getattr(current_provider, "timeout_seconds", None) if isinstance(
            current_provider, GeminiProvider
        ) else None
        return GeminiProvider(model=model, timeout_seconds=timeout)

    current_host = DEFAULT_OLLAMA_HOST
    keep_alive = DEFAULT_KEEP_ALIVE
    timeout = None
    deadline = None
    if isinstance(current_provider, OllamaProvider):
        current_host = current_provider.host
        keep_alive = current_provider.keep_alive
        timeout = current_provider.timeout_seconds
        deadline = current_provider.total_deadline_seconds

    return OllamaProvider(
        host=host or current_host,
        model=model,
        keep_alive=keep_alive,
        timeout_seconds=timeout,
        total_deadline_seconds=deadline,
    )


@router.get("/provider", response_model=ProviderStatusResponse)
def get_provider_status(request: Request) -> ProviderStatusResponse:
    return _build_provider_status(_get_provider(request))


@router.put("/provider", response_model=ProviderStatusResponse)
def update_provider_config(body: ProviderConfigUpdate, request: Request) -> ProviderStatusResponse:
    current = _get_provider(request)
    request.app.state.llm_provider = _rebuild_provider(
        body.provider,
        body.model,
        host=body.host,
        base_url=body.base_url,
        current_provider=current,
    )
    return _build_provider_status(request.app.state.llm_provider)


@router.get("/ollama", response_model=OllamaStatusResponse)
def get_ollama_status(request: Request) -> OllamaStatusResponse:
    provider = _get_provider(request)
    host, model, keep_alive = _provider_snapshot(provider)
    if isinstance(provider, OllamaProvider):
        reachable, models, error = provider.list_local_models()
    else:
        probe = OllamaProvider(host=host, model=model, keep_alive=keep_alive)
        reachable, models, error = probe.list_local_models()
    return OllamaStatusResponse(
        host=host,
        model=model,
        keep_alive=keep_alive,
        reachable=reachable,
        models=models,
        error=error,
    )


@router.put("/ollama", response_model=OllamaStatusResponse)
def update_ollama_config(body: OllamaConfigUpdate, request: Request) -> OllamaStatusResponse:
    if body.model is None and body.host is None:
        raise HTTPException(status_code=400, detail="Provide model and/or host.")
    current = _get_provider(request)
    host, model, keep_alive = _provider_snapshot(current)
    timeout = getattr(current, "timeout_seconds", None)
    deadline = getattr(current, "total_deadline_seconds", None)
    request.app.state.llm_provider = OllamaProvider(
        host=body.host or host,
        model=body.model or model,
        keep_alive=keep_alive,
        timeout_seconds=timeout,
        total_deadline_seconds=deadline,
    )
    return get_ollama_status(request)
