"""
Phase 5 — standalone inference over a VERIFIED artifact.

Loads pipeline.joblib only. Never calls an LLM, never invokes LangGraph,
never retrains, never executes generated Python.
"""

from app.deployment.package import write_deployment_package
from app.deployment.predict import predict_unseen
from app.deployment.readiness import check_deployment_readiness

__all__ = [
    "check_deployment_readiness",
    "predict_unseen",
    "write_deployment_package",
]
