"""SHA-256 content hashes plus explicit metadata. Not an immutability claim."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.agent.plan_canonical import CanonicalPlan, CanonicalPlanStep
from app.artifacts.publisher import BUNDLE_FILES, read_artifact_status
from app.governance.hashing import HASH_ALGORITHM, HASH_CAVEAT, sha256_bytes, sha256_canonical_json, sha256_file
from app.governance.helpers import dump, field, operation_rows
from app.schemas.governance import FingerprintManifest, HashEntry
from app.storage.dataset_store import DatasetStore
from app.storage.exceptions import DatasetNotFoundError

_ARTIFACT_CONTENT_FILES = (
    "pipeline.joblib",
    "pipeline.py",
    "training_reproduction.ipynb",
    "manifest.json",
    "evidence.json",
)


def _entry(name: str, digest: Optional[str], *, reason: str | None = None) -> HashEntry:
    return HashEntry(
        name=name,
        kind="CONTENT_HASH",
        algorithm=HASH_ALGORITHM,
        digest=digest,
        available=digest is not None,
        reason=reason,
    )


def _dataset_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _executed_plan_digest(state: Any) -> str:
    rows = operation_rows(state)
    candidates: list[tuple] = []
    for item in field(state, "model_results", default=[]) or []:
        params = field(item, "parameters", default={}) or {}
        pair = (field(item, "algorithm"), tuple(sorted(dict(params).items())))
        if pair not in candidates:
            candidates.append(pair)
    plan = CanonicalPlan(
        target_column=str(field(state, "target_column") or ""),
        steps=tuple(
            CanonicalPlanStep(tool_name=row["tool_name"], arguments=row["arguments"])
            for row in rows
        ),
        model_candidates=tuple(sorted(candidates)),
    )
    return sha256_bytes(plan.canonical_json().encode("utf-8"))


def build_fingerprint_manifest(
    run_id: str,
    state: Any,
    *,
    dataset_id: str | None,
    dataset_store: DatasetStore | None,
    artifact_root: Path | None,
    requirements_path: Path | None,
) -> FingerprintManifest:
    hashes: list[HashEntry] = []

    if dataset_store is not None and dataset_id:
        try:
            frame = dataset_store.get(dataset_id)
        except DatasetNotFoundError:
            hashes.append(_entry("dataset", None, reason="Dataset is not in DatasetStore."))
        else:
            hashes.append(_entry("dataset", sha256_bytes(_dataset_csv_bytes(frame))))
    else:
        hashes.append(_entry("dataset", None, reason="Dataset store or dataset_id is missing."))

    if state is None:
        hashes.append(_entry("executed_plan", None, reason="No final state to hash."))
    else:
        hashes.append(_entry("executed_plan", _executed_plan_digest(state)))

    bundle_dir = (artifact_root / run_id) if artifact_root is not None else None
    if bundle_dir is None or not bundle_dir.is_dir():
        for name in _ARTIFACT_CONTENT_FILES:
            hashes.append(_entry(name, None, reason="Artifact bundle has not been generated."))
    else:
        for name in _ARTIFACT_CONTENT_FILES:
            path = bundle_dir / name
            if path.is_file():
                hashes.append(_entry(name, sha256_file(path)))
            else:
                hashes.append(_entry(name, None, reason=f"{name} is not on disk."))

    if requirements_path is not None and requirements_path.is_file():
        hashes.append(_entry("requirements.txt", sha256_file(requirements_path)))
    else:
        hashes.append(_entry("requirements.txt", None, reason="requirements.txt was not found."))

    env = dump(field(field(state, "reproducibility"), "environment")) if state is not None else None
    if env:
        hashes.append(_entry("environment_snapshot", sha256_canonical_json(env)))
    else:
        hashes.append(_entry("environment_snapshot", None, reason="No recorded environment snapshot."))

    artifact_status = (
        read_artifact_status(artifact_root, run_id) if artifact_root is not None else None
    )
    repro = field(state, "reproducibility") if state is not None else None
    metadata = {
        "hash_algorithm": HASH_ALGORITHM,
        "recorded_dataset_fingerprint_sha256": field(repro, "dataset_fingerprint"),
        "run_status": field(state, "status") if state is not None else None,
        "dataset_id": dataset_id,
        "artifact_status": (artifact_status or {}).get("artifact_status"),
        "bundle_files_expected": list(BUNDLE_FILES),
        "python_version": field(field(repro, "environment"), "python_version") if repro is not None else None,
        "pandas_version": field(field(repro, "environment"), "pandas_version") if repro is not None else None,
        "numpy_version": field(field(repro, "environment"), "numpy_version") if repro is not None else None,
        "sklearn_version": field(field(repro, "environment"), "sklearn_version") if repro is not None else None,
    }
    return FingerprintManifest(
        run_id=run_id,
        hash_algorithm=HASH_ALGORITHM,
        content_hashes=hashes,
        metadata=metadata,
        caveat=HASH_CAVEAT,
    )
