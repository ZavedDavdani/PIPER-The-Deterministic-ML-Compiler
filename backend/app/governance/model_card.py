"""Deterministic model card from recorded evaluation, comparison, and guardrails."""

from __future__ import annotations

from typing import Any, Optional

from app.governance.explainability import extract_feature_importance
from app.governance.helpers import (
    field,
    preprocessing_lines,
    winner_evaluation,
    winner_id,
    winner_training,
)
from app.schemas.governance import CandidateModelCardEntry, GuardrailCardEntry, ModelCard, RecordedMetric
from app.storage.model_store import InMemoryModelStore


def _metric_list(evaluation: Any) -> list[RecordedMetric]:
    if evaluation is None:
        return []
    metrics: list[RecordedMetric] = []
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        value = field(evaluation, name)
        if value is None:
            continue
        metrics.append(RecordedMetric(name=name, value=float(value)))
    return metrics


def _candidates(state: Any, selected_id: Optional[str]) -> list[CandidateModelCardEntry]:
    comparison = field(state, "comparison")
    entries = field(comparison, "models", default=[]) or []
    evals = {
        field(item, "model_id"): item
        for item in (field(state, "evaluation_results", "evaluation_results", default=[]) or [])
    }
    rows: list[CandidateModelCardEntry] = []
    for entry in entries:
        mid = field(entry, "model_id")
        ev = evals.get(mid)
        rows.append(
            CandidateModelCardEntry(
                model_id=str(mid),
                algorithm=str(field(entry, "algorithm") or field(ev, "algorithm") or "unknown"),
                accuracy=field(entry, "accuracy", default=field(ev, "accuracy")),
                precision=field(entry, "precision", default=field(ev, "precision")),
                recall=field(entry, "recall", default=field(ev, "recall")),
                f1=field(entry, "f1", default=field(ev, "f1")),
                roc_auc=field(entry, "roc_auc", default=field(ev, "roc_auc")),
                selected=bool(selected_id and mid == selected_id),
            )
        )
    return rows


def _guardrails(state: Any) -> list[GuardrailCardEntry]:
    validation = field(state, "validation")
    checks = field(validation, "checks", default=[]) or []
    rows: list[GuardrailCardEntry] = []
    for check in checks:
        rows.append(
            GuardrailCardEntry(
                check=str(field(check, "check") or "unknown"),
                passed=bool(field(check, "passed")),
                severity=str(field(check, "severity") or "info"),
                message=str(field(check, "message") or ""),
            )
        )
    return rows


def _split_info(state: Any) -> dict[str, Any] | None:
    train = winner_training(state)
    evaluation = winner_evaluation(state)
    repro = field(state, "reproducibility")
    payload: dict[str, Any] = {}
    if train is not None:
        payload["training_rows"] = field(train, "training_rows")
        payload["split_id"] = field(train, "split_id")
        payload["feature_count"] = field(train, "feature_count")
    if evaluation is not None:
        payload["test_rows"] = field(evaluation, "test_rows")
        payload["split_id"] = payload.get("split_id") or field(evaluation, "split_id")
    if repro is not None:
        payload["split_random_state"] = field(repro, "split_random_state")
        payload["model_random_state"] = field(repro, "model_random_state")
    return payload or None


def _baseline(state: Any) -> dict[str, Any] | None:
    baseline = field(state, "baseline")
    if baseline is None:
        return None
    dumped = baseline.model_dump(mode="json") if hasattr(baseline, "model_dump") else dict(baseline)
    return dumped


def _limitations(state: Any, *, completed: bool, valid: bool) -> list[str]:
    notes = [
        "This card is compiled from recorded PIPER evidence. Missing fields stay null.",
        "Feature importance, when present, is associative rather than causal.",
        "V1 trains only logistic_regression and random_forest on tabular classification.",
    ]
    if not completed:
        notes.append("The run did not complete; winning-model metrics may be absent.")
    if not valid:
        notes.append("Guardrails did not pass; this is not a verified deployable model.")
    failure = field(state, "failure")
    if failure is not None:
        notes.append(f"Recorded failure category: {field(failure, 'category')}.")
    return notes


def build_model_card(
    run_id: str,
    state: Any,
    *,
    dataset_id: str | None,
    model_store: InMemoryModelStore | None,
    artifact_status: dict[str, Any] | None,
) -> ModelCard:
    importance = extract_feature_importance(state, model_store)
    status = str(field(state, "status") or "")
    validation = field(state, "validation")
    valid = bool(field(validation, "valid")) if validation is not None else False
    completed = status == "completed"
    if state is None:
        return ModelCard(
            status="NOT_AVAILABLE",
            run_id=run_id,
            dataset_id=dataset_id,
            feature_importance=importance,
            reason="No final AgentState is stored for this run.",
            limitations=_limitations(None, completed=False, valid=False),
        )
    if not completed or not valid:
        reason = "Model cards for verified models require a completed run with validation.valid is True."
        return ModelCard(
            status="NOT_AVAILABLE",
            run_id=run_id,
            dataset_id=dataset_id,
            task_type=field(state, "task_type") or "binary_classification",
            target=field(state, "target_column"),
            winning_model_id=winner_id(state),
            candidate_models=_candidates(state, winner_id(state)),
            evaluation_metrics=_metric_list(winner_evaluation(state)),
            baseline_comparison=_baseline(state),
            train_test_split=_split_info(state),
            preprocessing_summary=preprocessing_lines(state),
            guardrail_results=_guardrails(state),
            limitations=_limitations(state, completed=completed, valid=valid),
            artifact_information=artifact_status,
            feature_importance=importance,
            reason=reason,
        )
    train = winner_training(state)
    return ModelCard(
        status="AVAILABLE",
        run_id=run_id,
        dataset_id=dataset_id,
        task_type=field(state, "task_type") or "binary_classification",
        target=field(state, "target_column"),
        winning_model_id=winner_id(state),
        winning_algorithm=field(train, "algorithm"),
        candidate_models=_candidates(state, winner_id(state)),
        evaluation_metrics=_metric_list(winner_evaluation(state)),
        baseline_comparison=_baseline(state),
        train_test_split=_split_info(state),
        preprocessing_summary=preprocessing_lines(state),
        guardrail_results=_guardrails(state),
        limitations=_limitations(state, completed=True, valid=True),
        artifact_information=artifact_status,
        feature_importance=importance,
    )
