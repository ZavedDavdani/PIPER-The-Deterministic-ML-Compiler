"""
Phase 3 — portable ML artifact publication.

PIPER compiles a verified run into a deployment-independent bundle.
The fitted sklearn Pipeline in ModelStore is the source of truth —
never reconstructed from the original LLM plan. Publication is gated
on holdout prediction parity after joblib round-trip.
"""

from app.artifacts.errors import ArtifactEligibilityError, ArtifactParityError
from app.artifacts.publisher import (
    DOWNLOADABLE_FILES,
    publish_run_artifacts,
    read_artifact_status,
)

__all__ = [
    "ArtifactEligibilityError",
    "ArtifactParityError",
    "DOWNLOADABLE_FILES",
    "publish_run_artifacts",
    "read_artifact_status",
]
