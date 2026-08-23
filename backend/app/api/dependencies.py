"""
FastAPI dependency providers (M5).

Every dependency here pulls a shared, process-lifetime singleton off
`request.app.state` — set up once at application startup (see
app/main.py's lifespan). Tests override these via
app.dependency_overrides (FastAPI's standard testing pattern), never by
monkeypatching app.state directly.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request

from app.storage import DatasetStore, InMemoryExplorationStore, InMemoryModelStore, InMemoryRunStore, SplitStore


def get_dataset_store(request: Request) -> DatasetStore:
    return request.app.state.dataset_store


def get_split_store(request: Request) -> SplitStore:
    return request.app.state.split_store


def get_model_store(request: Request) -> InMemoryModelStore:
    return request.app.state.model_store


def get_run_store(request: Request) -> InMemoryRunStore:
    return request.app.state.run_store


def get_exploration_store(request: Request) -> InMemoryExplorationStore:
    return request.app.state.exploration_store


def get_llm_provider(request: Request):
    """
    Untyped return (matches app.llm.provider.LLMProvider's own Protocol
    style — duck-typed, no concrete class required) so this dependency
    works identically whether app.state.llm_provider is a real
    OllamaProvider (production default, see app/main.py) or a
    FakeLLMProvider/heuristic test double (via dependency_overrides).
    """
    return request.app.state.llm_provider


def get_artifact_dir(request: Request) -> Path:
    return request.app.state.artifact_dir
