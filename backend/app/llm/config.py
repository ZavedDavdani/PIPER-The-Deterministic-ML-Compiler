"""
LLM provider configuration resolution.

Reads provider/model settings from environment variables and constructor
arguments. Supports documented aliases (OLLAMA_HOST, OLLAMA_MODEL) without
hard-coding model identifiers.
"""

from __future__ import annotations

import os
from typing import Optional

SUPPORTED_LLM_PROVIDERS = ("openai", "ollama", "gemini")


def resolve_llm_provider_name(provider_name: Optional[str] = None) -> str:
    return (provider_name or os.environ.get("PIPER_LLM_PROVIDER") or "ollama").strip().lower()


def resolve_openai_model(model: Optional[str] = None) -> Optional[str]:
    if model is not None:
        return model
    value = os.environ.get("OPENAI_MODEL")
    return value if value else None


def resolve_ollama_host(host: Optional[str] = None) -> Optional[str]:
    if host is not None:
        return host
    return os.environ.get("PIPER_OLLAMA_HOST") or os.environ.get("OLLAMA_HOST") or None


def resolve_ollama_model(model: Optional[str] = None) -> Optional[str]:
    if model is not None:
        return model
    return os.environ.get("PIPER_LLM_MODEL") or os.environ.get("OLLAMA_MODEL") or None


def resolve_gemini_model(model: Optional[str] = None) -> Optional[str]:
    if model is not None:
        return model
    value = os.environ.get("GEMINI_MODEL")
    return value if value else None
