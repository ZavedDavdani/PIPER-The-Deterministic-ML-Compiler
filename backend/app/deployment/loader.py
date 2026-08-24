"""Load a VERIFIED pipeline.joblib. Never execute pipeline.py. Never rebuild from a plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from app.artifacts.hashes import sha256_file
from app.artifacts.publisher import read_artifact_status
from app.deployment.errors import InferenceError
from app.deployment.paths import bundle_dir


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceError(
            "manifest_invalid",
            f"Could not parse {path.name}.",
            {"filename": path.name, "error": str(exc)},
        ) from exc


def verify_bundle_hashes(dest: Path) -> dict[str, str]:
    hashes_path = dest / "hashes.json"
    if not hashes_path.is_file():
        raise InferenceError("hashes_missing", "hashes.json is missing from the artifact bundle.")
    payload = _read_json(hashes_path)
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise InferenceError("hashes_invalid", "hashes.json does not list file digests.")
    for name, expected in files.items():
        if name in {"hashes.json", "status.json"}:
            continue
        if "/" in str(name) or "\\" in str(name) or ".." in str(name):
            raise InferenceError("hashes_invalid", "hashes.json contains an unsafe filename.", {"name": name})
        path = dest / str(name)
        if not path.is_file():
            raise InferenceError(
                "hashes_mismatch",
                f"Hashed file '{name}' is missing.",
                {"name": name},
            )
        actual = sha256_file(path)
        if actual != expected:
            raise InferenceError(
                "hashes_mismatch",
                f"SHA-256 mismatch for '{name}'.",
                {"name": name},
            )
    return {str(k): str(v) for k, v in files.items()}


def load_verified_bundle(artifact_root: Path, run_id: str) -> dict[str, Any]:
    """
    Returns pipeline, manifest, status, dest. Rejects anything that is
    not a PIPER VERIFIED artifact. Does not import or exec pipeline.py.
    """
    dest = bundle_dir(artifact_root, run_id)
    status = read_artifact_status(artifact_root, run_id)
    if status.get("artifact_status") == "NOT_GENERATED" or not dest.is_dir():
        raise InferenceError(
            "artifact_missing",
            "No artifact bundle exists for this run.",
            {"run_id": run_id},
        )
    if status.get("artifact_status") != "VERIFIED":
        raise InferenceError(
            "artifact_not_verified",
            "Only a VERIFIED artifact can be used for inference.",
            {"run_id": run_id, "artifact_status": status.get("artifact_status")},
        )
    manifest_path = dest / "manifest.json"
    joblib_path = dest / "pipeline.joblib"
    if not manifest_path.is_file() or not joblib_path.is_file():
        raise InferenceError(
            "artifact_missing",
            "VERIFIED status is recorded but required bundle files are missing.",
            {"run_id": run_id},
        )
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_status") != "VERIFIED":
        raise InferenceError(
            "manifest_invalid",
            "manifest.json is not marked VERIFIED.",
            {"artifact_status": manifest.get("artifact_status")},
        )
    if manifest.get("run_id") not in (None, run_id) and manifest.get("run_id") != run_id:
        raise InferenceError(
            "manifest_invalid",
            "manifest.json run_id does not match the requested run.",
            {"manifest_run_id": manifest.get("run_id"), "run_id": run_id},
        )
    columns = manifest.get("expected_inference_columns") or (manifest.get("feature_schema") or {}).get(
        "feature_columns"
    )
    if not isinstance(columns, list) or not columns:
        raise InferenceError("manifest_invalid", "manifest.json has no expected_inference_columns.")
    verify_bundle_hashes(dest)
    try:
        pipeline = joblib.load(joblib_path)
    except Exception as exc:
        raise InferenceError(
            "pipeline_load_failed",
            "pipeline.joblib could not be loaded.",
            {"error": str(exc)},
        ) from exc
    if not hasattr(pipeline, "predict"):
        raise InferenceError(
            "pipeline_load_failed",
            "Loaded object has no predict().",
            {"loaded_type": type(pipeline).__name__},
        )
    return {
        "pipeline": pipeline,
        "manifest": manifest,
        "status": status,
        "dest": dest,
        "joblib_path": joblib_path,
        "feature_columns": [str(c) for c in columns],
        "winning_model_id": (manifest.get("winning_model") or {}).get("model_id")
        or status.get("winning_model_id"),
        "algorithm": (manifest.get("winning_model") or {}).get("algorithm") or status.get("algorithm"),
    }
