"""Deterministic deployment readiness. Fail closed; never invent READY."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.artifacts.publisher import read_artifact_status
from app.deployment.errors import InferenceError
from app.deployment.loader import load_verified_bundle, verify_bundle_hashes
from app.deployment.paths import bundle_dir
from app.deployment.predict import predict_unseen


def check_deployment_readiness(artifact_root: Path, run_id: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def _record(name: str, passed: bool, detail: str | None = None) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    try:
        dest = bundle_dir(artifact_root, run_id)
        exists = dest.is_dir()
        _record("artifact_exists", exists, None if exists else "Bundle directory is missing.")
        status = read_artifact_status(artifact_root, run_id)
        verified = status.get("artifact_status") == "VERIFIED"
        _record("artifact_verified", verified, None if verified else str(status.get("artifact_status")))
        bundle = load_verified_bundle(artifact_root, run_id)
        _record("manifest_valid", True)
        verify_bundle_hashes(dest)
        _record("hashes_valid", True)
        _record("pipeline_loads", True, type(bundle["pipeline"]).__name__)
        columns = bundle["feature_columns"]
        _record("input_schema_valid", bool(columns), f"{len(columns)} required columns")
        _record("predict_available", callable(getattr(bundle["pipeline"], "predict", None)))
        smoke = bundle["manifest"].get("inference_smoke_row")
        if not isinstance(smoke, dict) or not smoke:
            raise InferenceError(
                "manifest_invalid",
                "manifest.json has no inference_smoke_row for the readiness prediction check.",
            )
        scored = predict_unseen(artifact_root, run_id, pd.DataFrame([smoke]))
        _record("prediction_succeeds", True, f"row_count={scored['row_count']}")
        parity_ok = scored.get("parity", {}).get("parity_status") == "passed"
        _record("inference_parity", parity_ok)
        if not parity_ok:
            raise InferenceError("inference_parity_failed", "Readiness parity check did not pass.")
        return {
            "run_id": run_id,
            "status": "READY",
            "artifact_status": "VERIFIED",
            "winning_model_id": bundle["winning_model_id"],
            "algorithm": bundle["algorithm"],
            "required_columns": columns,
            "checks": checks,
            "reason": None,
        }
    except InferenceError as exc:
        if not any(item["check"] == "artifact_exists" for item in checks):
            _record("artifact_exists", False, exc.message)
        return {
            "run_id": run_id,
            "status": "NOT_READY",
            "artifact_status": read_artifact_status(artifact_root, run_id).get("artifact_status"),
            "winning_model_id": None,
            "algorithm": None,
            "required_columns": [],
            "checks": checks,
            "reason": {"code": exc.code, "message": exc.message, "details": exc.details},
        }
