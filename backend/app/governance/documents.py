"""Deterministic markdown/JSON downloads for governance documents."""

from __future__ import annotations

import json

from app.schemas.governance import GovernanceBundle

GOVERNANCE_DOCUMENT_NAMES = (
    "model_card.json",
    "model_card.md",
    "data_card.json",
    "data_card.md",
    "fingerprints.json",
    "feature_importance.json",
    "fairness.json",
)


def _json(payload) -> str:
    return json.dumps(payload.model_dump(mode="json"), indent=2) + "\n"


def _md_list(items: list[str]) -> str:
    if not items:
        return "_None recorded._\n"
    return "".join(f"- {item}\n" for item in items)


def _model_card_md(bundle: GovernanceBundle) -> str:
    card = bundle.model_card
    metrics = "\n".join(
        f"- {m.name}: {m.value}" for m in card.evaluation_metrics
    ) or "_No recorded metrics._"
    candidates = "\n".join(
        f"- {c.algorithm} (`{c.model_id}`) F1={c.f1}"
        + (" **selected**" if c.selected else "")
        for c in card.candidate_models
    ) or "_No candidates recorded._"
    importance = card.feature_importance
    top = importance.rows[:10]
    importance_block = (
        "\n".join(
            f"- `{row.transformed_feature}` importance={row.importance:.6f}"
            + (f" direction={row.direction}" if row.direction else "")
            for row in top
        )
        if importance.status == "AVAILABLE" and top
        else f"_NOT_AVAILABLE_: {importance.reason or 'not derived.'}_"
    )
    return (
        f"# Model Card\n\n"
        f"- run_id: `{card.run_id}`\n"
        f"- status: {card.status}\n"
        f"- dataset: `{card.dataset_id}`\n"
        f"- task: {card.task_type}\n"
        f"- target: `{card.target}`\n"
        f"- winning model: `{card.winning_model_id}` ({card.winning_algorithm})\n\n"
        f"## Candidates\n{candidates}\n\n"
        f"## Evaluation metrics\n{metrics}\n\n"
        f"## Feature importance\n{importance.disclaimer}\n\n{importance_block}\n\n"
        f"## Limitations\n{_md_list(card.limitations)}\n"
        + (f"\nReason: {card.reason}\n" if card.reason else "")
    )


def _data_card_md(bundle: GovernanceBundle) -> str:
    card = bundle.data_card
    features = ", ".join(f"`{name}`" for name in card.feature_list) or "_none_"
    missing = "\n".join(
        f"- {row.get('column')}: {row.get('missing_count')} ({row.get('missing_percentage')}%)"
        for row in card.missingness
    ) or "_No recorded missingness._"
    ops = "\n".join(
        f"- {row.get('tool_name')} {row.get('arguments')}"
        for row in card.preprocessing_operations
    ) or "_No executed preprocessing recorded._"
    return (
        f"# Data Card\n\n"
        f"- run_id: `{card.run_id}`\n"
        f"- dataset: `{card.dataset_id}`\n"
        f"- rows × columns: {card.rows} × {card.columns}\n"
        f"- target: `{card.target}`\n"
        f"- features: {features}\n\n"
        f"## Missingness\n{missing}\n\n"
        f"## Executed preprocessing\n{ops}\n\n"
        f"## Limitations\n{_md_list(card.limitations)}\n"
    )


def render_governance_document(bundle: GovernanceBundle, filename: str) -> tuple[str, str]:
    """Returns (media_type, body). Unknown names raise KeyError."""
    if filename not in GOVERNANCE_DOCUMENT_NAMES:
        raise KeyError(filename)
    if filename == "model_card.json":
        return "application/json", _json(bundle.model_card)
    if filename == "model_card.md":
        return "text/markdown", _model_card_md(bundle)
    if filename == "data_card.json":
        return "application/json", _json(bundle.data_card)
    if filename == "data_card.md":
        return "text/markdown", _data_card_md(bundle)
    if filename == "fingerprints.json":
        return "application/json", _json(bundle.fingerprints)
    if filename == "feature_importance.json":
        return "application/json", _json(bundle.feature_importance)
    if filename == "fairness.json":
        return "application/json", _json(bundle.fairness)
    raise KeyError(filename)
