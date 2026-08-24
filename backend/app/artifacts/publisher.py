"""Compile a verified run into a portable, PIPER-independent artifact bundle."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.agent.productization import build_evidence_export
from app.artifacts.eligibility import require_eligible_run
from app.artifacts.errors import ArtifactEligibilityError, ArtifactParityError
from app.artifacts.hashes import write_hashes_manifest as write_hashes_manifest
from app.artifacts.notebook import render_training_notebook as render_training_notebook
from app.artifacts.parity import assert_joblib_parity, holdout_features
from app.artifacts.pipeline_script import render_pipeline_py
from app.storage.model_store import InMemoryModelStore
from app.storage.run_store import InMemoryRunStore
from app.storage.split_store import SplitStore

BUNDLE_FILES = (
    "pipeline.joblib",
    "pipeline.py",
    "training_reproduction.ipynb",
    "manifest.json",
    "evidence.json",
)
DOWNLOADABLE_FILES = BUNDLE_FILES + ("hashes.json",)
STATUS_FILENAME = "status.json"


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return value


def _executed_ops(state: Any) -> list[str]:
    names: list[str] = []
    for log_name in ("cleaning_log", "feature_log", "tool_trace"):
        for item in getattr(state, log_name, None) or []:
            tool = getattr(item, "tool_name", None)
            if tool is None and isinstance(item, dict):
                tool = item.get("tool_name")
            if tool:
                names.append(str(tool))
    return names


def _winner_eval(state: Any, winner_id: str) -> dict | None:
    for item in getattr(state, "evaluation_results", None) or []:
        mid = getattr(item, "model_id", None) if not isinstance(item, dict) else item.get("model_id")
        if mid == winner_id:
            return _jsonable(item)
    return None


def _status_payload(
    *,
    run_id: str,
    artifact_status: str,
    parity_status: str,
    winning_model_id: str | None,
    algorithm: str | None,
    files: list[str],
    error: dict | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "artifact_status": artifact_status,
        "parity_status": parity_status,
        "winning_model_id": winning_model_id,
        "algorithm": algorithm,
        "files": files,
        "error": error,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_failed_status(dest: Path, payload: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / STATUS_FILENAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_artifact_status(artifact_root: Path, run_id: str) -> dict:
    path = artifact_root / run_id / STATUS_FILENAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "artifact_status": "NOT_GENERATED",
        "parity_status": "not_run",
        "winning_model_id": None,
        "algorithm": None,
        "files": [],
        "error": None,
        "created_at": None,
    }


def publish_run_artifacts(
    run_id: str,
    *,
    run_store: InMemoryRunStore,
    model_store: InMemoryModelStore,
    split_store: SplitStore,
    artifact_root: Path,
) -> dict:
    """
    Serialize the ACTUAL fitted winning pipeline. Never reconstruct from
    the LLM plan. Parity failure aborts publication (no VERIFIED mark).
    """
    record = run_store.get(run_id)
    dest = artifact_root / run_id
    dest.mkdir(parents=True, exist_ok=True)

    try:
        state, artifact = require_eligible_run(record, model_store, split_store)
    except ArtifactEligibilityError as exc:
        payload = _status_payload(
            run_id=run_id,
            artifact_status="FAILED",
            parity_status="not_run",
            winning_model_id=None,
            algorithm=None,
            files=[],
            error={"code": exc.code, "message": exc.message, "details": exc.details},
        )
        _write_failed_status(dest, payload)
        raise

    meta = artifact.metadata
    winner_id = meta.model_id
    x_holdout = holdout_features(artifact, split_store)
    staging: Path | None = None

    try:
        staging = Path(tempfile.mkdtemp(prefix=f"piper-artifact-{run_id}-"))
        joblib_path = staging / "pipeline.joblib"
        joblib.dump(artifact.pipeline, joblib_path)
        try:
            parity = assert_joblib_parity(artifact.pipeline, joblib_path, x_holdout)
        except ArtifactParityError as exc:
            fail = _status_payload(
                run_id=run_id,
                artifact_status="FAILED",
                parity_status="failed",
                winning_model_id=winner_id,
                algorithm=meta.algorithm,
                files=[],
                error={"code": exc.code, "message": exc.message, "details": exc.details},
            )
            _write_failed_status(dest, fail)
            raise

        (staging / "pipeline.py").write_text(
            render_pipeline_py(
                run_id=run_id,
                target_column=meta.target_column,
                algorithm=meta.algorithm,
                feature_columns=meta.feature_columns,
            ),
            encoding="utf-8",
        )

        candidates = [_jsonable(item) for item in (getattr(state, "model_results", None) or [])]
        winner = next((c for c in candidates if c.get("model_id") == winner_id), None) or {
            "model_id": winner_id,
            "algorithm": meta.algorithm,
            "parameters": meta.parameters,
        }
        metrics = _winner_eval(state, winner_id)
        (staging / "training_reproduction.ipynb").write_text(
            render_training_notebook(
                run_id=run_id,
                dataset_id=record.dataset_id,
                target_column=meta.target_column,
                split_id=meta.split_id,
                feature_columns=meta.feature_columns,
                categorical_columns=meta.categorical_columns,
                numeric_columns=meta.numeric_columns,
                candidates=candidates,
                winner=winner,
                metrics=metrics,
                executed_operations=_executed_ops(state),
            ),
            encoding="utf-8",
        )

        events = run_store.get_events(run_id)
        evidence = build_evidence_export(
            run_id,
            record.status,
            events,
            state,
            dataset_id=record.dataset_id,
            target_column=getattr(state, "target_column", record.target_column),
        )
        (staging / "evidence.json").write_text(
            json.dumps(evidence.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )

        created_at = datetime.now(timezone.utc).isoformat()
        smoke = x_holdout.iloc[0].to_dict()
        smoke_row = {}
        for key, value in smoke.items():
            if pd.isna(value):
                smoke_row[str(key)] = None
            elif hasattr(value, "item"):
                smoke_row[str(key)] = value.item()
            else:
                smoke_row[str(key)] = value
        manifest = {
            "run_id": run_id,
            "dataset_id": record.dataset_id,
            "target": meta.target_column,
            "task_type": getattr(state, "task_type", None) or "classification",
            "feature_schema": {
                "feature_columns": list(meta.feature_columns),
                "categorical_columns": list(meta.categorical_columns),
                "numeric_columns": list(meta.numeric_columns),
            },
            "expected_inference_columns": list(meta.feature_columns),
            "inference_smoke_row": smoke_row,
            "winning_model": {
                "model_id": winner_id,
                "algorithm": meta.algorithm,
                "parameters": meta.parameters,
            },
            "metrics": metrics,
            "library_versions": {
                "scikit-learn": _pkg_version("scikit-learn"),
                "pandas": _pkg_version("pandas"),
                "numpy": _pkg_version("numpy"),
                "joblib": _pkg_version("joblib"),
            },
            "artifact_version": "piper.artifact.v1",
            "parity_status": "passed",
            "artifact_status": "VERIFIED",
            "created_at": created_at,
            "filenames": list(DOWNLOADABLE_FILES),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        write_hashes_manifest(staging, list(BUNDLE_FILES))

        status = _status_payload(
            run_id=run_id,
            artifact_status="VERIFIED",
            parity_status="passed",
            winning_model_id=winner_id,
            algorithm=meta.algorithm,
            files=list(DOWNLOADABLE_FILES),
        )
        status["created_at"] = created_at
        status["parity"] = parity
        (staging / STATUS_FILENAME).write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(staging), str(dest))
        staging = None
        return status
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
