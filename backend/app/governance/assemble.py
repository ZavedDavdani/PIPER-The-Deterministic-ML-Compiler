"""Assemble the Phase 4 governance bundle from recorded run evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.artifacts.publisher import read_artifact_status
from app.governance.data_card import build_data_card
from app.governance.explainability import extract_feature_importance
from app.governance.fairness import analyze_subgroups
from app.governance.fingerprints import build_fingerprint_manifest
from app.governance.model_card import build_model_card
from app.schemas.governance import GovernanceBundle
from app.storage.dataset_store import DatasetStore
from app.storage.model_store import InMemoryModelStore
from app.storage.split_store import SplitStore

_REPO_REQUIREMENTS = Path(__file__).resolve().parents[3] / "requirements.txt"

_NOTES = [
    "Governance is compiled from recorded PIPER evidence and fitted objects.",
    "The original LLM plan is not re-executed and is not a source of metrics.",
    "No language model is invoked while building this bundle.",
    "SHA-256 fingerprints are tamper-evident; they do not make files immutable.",
    "Feature importance is associative, not causal.",
    "Subgroup metrics are statistical measurements, not legal determinations.",
]


def assemble_governance_bundle(
    run_id: str,
    *,
    run_status: str,
    state: Any,
    dataset_id: str | None,
    model_store: InMemoryModelStore | None = None,
    split_store: SplitStore | None = None,
    dataset_store: DatasetStore | None = None,
    artifact_root: Path | None = None,
    subgroup_columns: Optional[list[str]] = None,
) -> GovernanceBundle:
    artifact_status = (
        read_artifact_status(artifact_root, run_id) if artifact_root is not None else None
    )
    importance = extract_feature_importance(state, model_store)
    model_card = build_model_card(
        run_id,
        state,
        dataset_id=dataset_id,
        model_store=model_store,
        artifact_status=artifact_status,
    )
    data_card = build_data_card(run_id, state, dataset_id=dataset_id)
    fingerprints = build_fingerprint_manifest(
        run_id,
        state,
        dataset_id=dataset_id,
        dataset_store=dataset_store,
        artifact_root=artifact_root,
        requirements_path=_REPO_REQUIREMENTS,
    )
    fairness = analyze_subgroups(
        state,
        columns=list(subgroup_columns or []),
        model_store=model_store,
        split_store=split_store,
    )
    limitations = list(dict.fromkeys([*model_card.limitations, *data_card.limitations, *_NOTES]))
    return GovernanceBundle(
        run_id=run_id,
        run_status=run_status,
        model_card=model_card,
        data_card=data_card,
        fingerprints=fingerprints,
        feature_importance=importance,
        fairness=fairness,
        limitations=limitations,
        artifact_status=artifact_status,
        notes=_NOTES,
    )
