"""Structured artifact-publication failures. Never repaired silently."""

from __future__ import annotations


class ArtifactEligibilityError(Exception):
    """The run is not a verified, safe candidate for a deployable artifact."""

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ArtifactParityError(Exception):
    """
    Reloaded pipeline.joblib predictions did not match the in-memory
    winning pipeline on the evaluation holdout. Publication must abort.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.code = "artifact_parity_failed"
        self.message = message
        self.details = details or {}
        super().__init__(message)

