"""Structured inference/deployment failures. Never repaired silently."""

from __future__ import annotations


class InferenceError(Exception):
    """Fail-closed inference: missing artifact, bad schema, or parity failure."""

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
