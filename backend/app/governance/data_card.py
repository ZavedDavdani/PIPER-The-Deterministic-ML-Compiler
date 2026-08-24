"""Deterministic data card from recorded profile, logs, and split metadata."""

from __future__ import annotations

from typing import Any

from app.governance.helpers import dump, field, operation_rows, winner_evaluation, winner_training
from app.schemas.governance import DataCard, DataCardColumn


def _column_kind(dtype: str | None, semantic: str | None) -> str:
    if semantic in {"numeric", "categorical"}:
        return semantic
    if not dtype:
        return "other"
    lowered = dtype.lower()
    if any(token in lowered for token in ("int", "float", "double", "number")):
        return "numeric"
    if any(token in lowered for token in ("object", "string", "category", "bool")):
        return "categorical"
    return "other"


def build_data_card(run_id: str, state: Any, *, dataset_id: str | None) -> DataCard:
    if state is None:
        return DataCard(
            status="NOT_AVAILABLE",
            run_id=run_id,
            dataset_id=dataset_id,
            reason="No final AgentState is stored for this run.",
            limitations=["A data card cannot be compiled without recorded run state."],
        )
    profile = dump(field(state, "profile")) or {}
    target = field(state, "target_column")
    raw_columns = profile.get("column_profiles") or profile.get("column_profiles") or []
    summaries: list[DataCardColumn] = []
    numeric: list[str] = []
    categorical: list[str] = []
    missingness: list[dict[str, Any]] = []
    features: list[str] = []
    for col in raw_columns:
        payload = dump(col) or {}
        name = str(payload.get("name") or "")
        if not name:
            continue
        role = "target" if name == target else "feature"
        kind = _column_kind(payload.get("dtype"), payload.get("semantic_type"))
        summaries.append(
            DataCardColumn(
                name=name,
                dtype=payload.get("dtype"),
                missing_count=payload.get("missing_count"),
                missing_percentage=payload.get("missing_percentage"),
                unique_count=payload.get("unique_count"),
                role=role,  # type: ignore[arg-type]
                kind=kind,  # type: ignore[arg-type]
            )
        )
        if role == "feature":
            features.append(name)
            if kind == "numeric":
                numeric.append(name)
            elif kind == "categorical":
                categorical.append(name)
        missing_count = payload.get("missing_count") or 0
        if missing_count:
            missingness.append(
                {
                    "column": name,
                    "missing_count": missing_count,
                    "missing_percentage": payload.get("missing_percentage"),
                }
            )

    quality: list[str] = []
    duplicates = profile.get("duplicate_rows")
    if isinstance(duplicates, int) and duplicates > 0:
        quality.append(f"Recorded duplicate_rows={duplicates}.")
    for finding in field(state, "validation") and field(field(state, "validation"), "warnings", default=[]) or []:
        quality.append(str(field(finding, "message") or field(finding, "check")))

    train = winner_training(state)
    evaluation = winner_evaluation(state)
    repro = field(state, "reproducibility")
    train_test = None
    if train is not None or evaluation is not None or repro is not None:
        train_test = {
            "training_rows": field(train, "training_rows"),
            "test_rows": field(evaluation, "test_rows"),
            "split_id": field(train, "split_id") or field(evaluation, "split_id") or field(state, "split_id"),
            "split_random_state": field(repro, "split_random_state"),
        }

    limitations = [
        "This card uses recorded profile statistics, not a re-scan of raw row values.",
        "Sample cell values from profiling are omitted to avoid leaking sensitive raw data.",
        "Preprocessing lists operations that actually executed, not the original LLM proposal.",
    ]
    return DataCard(
        status="AVAILABLE",
        run_id=run_id,
        dataset_id=dataset_id,
        rows=profile.get("rows"),
        columns=profile.get("columns"),
        target=target,
        feature_list=features,
        column_summaries=summaries,
        numeric_features=numeric,
        categorical_features=categorical,
        missingness=missingness,
        preprocessing_operations=operation_rows(state),
        train_test=train_test,
        data_quality_findings=quality,
        limitations=limitations,
    )
