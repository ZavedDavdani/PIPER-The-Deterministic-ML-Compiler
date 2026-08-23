"""
Local operator settings: Ollama host/model/status.

Changing the model only reconstructs OllamaProvider with the same
contract as production (no temperature / options field). This never
touches validate_proposed_plan() or graph routing.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.llm.ollama_provider import (
    DEFAULT_KEEP_ALIVE,
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_HOST,
    OllamaProvider,
)

router = APIRouter(prefix="/settings", tags=["settings"])


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


@router.get("/ollama", response_model=OllamaStatusResponse)
def get_ollama_status(request: Request) -> OllamaStatusResponse:
    provider = request.app.state.llm_provider
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
    current = request.app.state.llm_provider
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
