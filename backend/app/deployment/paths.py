"""Resolve PIPER-registered artifact paths. Never accept a user filesystem path."""

from __future__ import annotations

import re
from pathlib import Path

from app.deployment.errors import InferenceError

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

PACKAGE_DIRNAME = "deployment_package"
PACKAGE_FILES = (
    "pipeline.joblib",
    "inference.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
)


def require_safe_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise InferenceError(
            "invalid_run_id",
            "run_id must be a PIPER identifier (letters, digits, underscore, hyphen).",
            {"run_id": run_id},
        )
    return run_id


def bundle_dir(artifact_root: Path, run_id: str) -> Path:
    safe = require_safe_run_id(run_id)
    root = artifact_root.resolve()
    dest = (root / safe).resolve()
    if root not in dest.parents and dest != root:
        raise InferenceError("invalid_run_id", "Artifact path is outside the artifact root.", {"run_id": run_id})
    return dest


def package_dir(artifact_root: Path, run_id: str) -> Path:
    return bundle_dir(artifact_root, run_id) / PACKAGE_DIRNAME
