"""
build_run_explanation() and friends (Batch 6A: Learn-Explain).

Every function here is a pure, read-only transformation: real,
already-computed evidence in (an OperationRecord, a ModelComparison, an
EvaluationResult, a ValidationCheck, a FailureInfo), a schema from
app/schemas/learning.py out. Nothing here ever constructs a state
update, calls a graph node, a store, or an LLM provider — this module
is structurally incapable of influencing a run, matching the locked
Learn-Explain constraint.

Every "meaning" string is a STATIC, reviewed template (the _*_MEANINGS
dicts below) — never LLM-generated, never re-derived per dataset. Real
per-run numbers (a metric's actual value, a check's actual message, a
failure's actual evidence) are plugged into those templates or passed
through verbatim, never fabricated. This mirrors exactly the pattern
already used for `ModelComparison.justification` (Pre-6A Polish item
2) and `_build_selection_justification()`.

build_run_explanation() takes `run_id` as a separate argument for the
same reason build_run_summary() does (see app/agent/run_summary.py's
docstring): the API layer's `record.final_state` is `_RunResultState`
(app/agent/tracing.py), which never carries `run_id` — no graph node's
partial update ever includes that key.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.baseline import BaselineComparisonResult
from app.schemas.evaluation import EvaluationResult, ModelComparison
from app.schemas.failure import FailureInfo
from app.schemas.guardrails import ValidationCheck
from app.schemas.learning import (
    EvaluationExplanation,
    FailureExplanation,
    GuardrailCheckExplanation,
    MetricExplanation,
    ModelSelectionExplanation,
    OperationExplanation,
    RunExplanation,
)

_TOOL_INTROS: dict[str, str] = {
    "drop_column": "PIPER removed a column entirely from the dataset.",
    "convert_column_type": "PIPER converted a column to a different data type.",
    "impute_missing_values": "PIPER filled in missing values in a column.",
    "encode_categorical_features": "PIPER one-hot encoded one or more categorical columns.",
    "scale_features": "PIPER standardized (Z-score scaled) one or more numeric columns.",
}

_METRIC_TEMPLATES: dict[str, str] = {
    "accuracy": "Accuracy = {value} — this fraction of all predictions on the held-out test split were correct.",
    "precision": "Precision = {value} — of every case the model predicted positive, this fraction actually was.",
    "recall": "Recall = {value} — of every case that was actually positive, the model correctly caught this fraction.",
    "f1": "F1 Score = {value} — the balance between precision and recall (PIPER's locked model-selection metric).",
    "roc_auc": "ROC-AUC = {value} — how well the model separates the two classes across every possible decision threshold (1.0 = perfect, 0.5 = random guessing).",
}

_GUARDRAIL_MEANINGS: dict[str, str] = {
    "data_leakage": (
        "Checks whether any feature is suspiciously predictive of the "
        "target (e.g. a near-duplicate of the target, or an "
        "identifier-like column) — a sign the model could be 'cheating' "
        "by seeing information it shouldn't have at prediction time."
    ),
    "target_imbalance": (
        "Checks how skewed the target class distribution is — a "
        "severely imbalanced target (minority class under 5%) can make "
        "a trivial always-predict-the-majority model look deceptively "
        "good."
    ),
    "constant_features": (
        "Checks for columns with only one distinct value (or entirely "
        "missing) — a constant feature carries no predictive "
        "information and cannot meaningfully contribute to a trained "
        "model."
    ),
    "high_cardinality": (
        "Checks for categorical columns that are almost entirely unique "
        "values (over 99%) — a sign the column is actually an "
        "identifier (like a customer ID), not a genuine predictive "
        "feature."
    ),
    "suspicious_evaluation_metric": (
        "Checks whether the trained model's evaluation metric is "
        "implausibly high — a common symptom of undetected data "
        "leakage."
    ),
    "baseline_gate": (
        "Checks whether the trained model meaningfully outperforms a "
        "trivial majority-class baseline on the same test split (see "
        "the Baseline Gate policy)."
    ),
}

_FAILURE_CATEGORY_MEANINGS: dict[str, str] = {
    "DATA_ERROR": (
        "The dataset or environment itself makes this run impossible "
        "to complete (e.g. it vanished mid-run) — not something a "
        "different plan could fix, so this is never retried."
    ),
    "SCHEMA_ERROR": (
        "The dataset's structure is fundamentally invalid for this "
        "task (e.g. zero columns or zero rows) — never retried."
    ),
    "TARGET_ERROR": (
        "The target column's values don't resolve to a supported task "
        "type — PIPER V1 only supports binary/multiclass classification "
        "— never retried."
    ),
    "EXECUTION_BUDGET_EXCEEDED": (
        "PIPER's own hard execution-step safety limit was reached — a "
        "deterministic backstop independent of max_retries, "
        "guaranteeing the run always reaches a clean terminal state."
    ),
    "LEAKAGE_ERROR": (
        "The data-leakage guardrail found a feature suspiciously "
        "predictive of the target — recoverable, since a different plan "
        "could drop the offending feature."
    ),
    "IMBALANCE_ERROR": (
        "The target class distribution is severely imbalanced — "
        "recoverable in principle, though a different plan can't "
        "create more minority-class data."
    ),
    "FEATURE_ERROR": (
        "A feature-engineering step failed (e.g. an invalid column "
        "reference) — recoverable via a different plan."
    ),
    "TRAINING_ERROR": (
        "Model training itself failed — recoverable via a different "
        "plan (e.g. different preprocessing)."
    ),
    "EVALUATION_ERROR": (
        "Evaluating or comparing trained models failed, or the LLM's "
        "proposed plan was rejected before execution — recoverable via "
        "a different plan."
    ),
    "DUPLICATE_PLAN": (
        "The newly proposed plan was executably identical to a plan "
        "already attempted and already failed — retrying the same plan "
        "again could never produce a different outcome, so this is "
        "never retried."
    ),
    "BASELINE_GATE_FAILED": (
        "The trained model didn't meaningfully outperform a trivial "
        "majority-class baseline — recoverable via a different plan."
    ),
    "PLAN_ADEQUACY": (
        "The proposed plan was structurally valid — every step named a "
        "real operation with correct arguments — but did not address a "
        "condition the dataset deterministically requires before "
        "training, such as a column with missing values that is neither "
        "imputed nor dropped. Recoverable: a different plan that "
        "addresses the reported condition can succeed."
    ),
}


def explain_operation(op) -> OperationExplanation:
    intro = _TOOL_INTROS.get(op.tool_name, f"PIPER ran '{op.tool_name}'.")
    return OperationExplanation(
        operation_id=op.operation_id,
        tool_name=op.tool_name,
        what_happened=f"{intro} {op.result_summary}",
        why=op.reason,
    )


def explain_model_selection(comparison: ModelComparison) -> ModelSelectionExplanation:
    recommended_algorithm = next(
        (m.algorithm for m in comparison.models if m.model_id == comparison.recommended_model_id),
        "unknown",
    )
    return ModelSelectionExplanation(
        recommended_model_id=comparison.recommended_model_id,
        recommended_algorithm=recommended_algorithm,
        justification=comparison.justification,
        candidates=list(comparison.models),
    )


def explain_evaluation(
    evaluation: EvaluationResult, baseline: Optional[BaselineComparisonResult]
) -> EvaluationExplanation:
    metrics = [
        MetricExplanation(metric=name, value=value, meaning=_METRIC_TEMPLATES[name].format(value=value))
        for name, value in (
            ("accuracy", evaluation.accuracy),
            ("precision", evaluation.precision),
            ("recall", evaluation.recall),
            ("f1", evaluation.f1),
            ("roc_auc", evaluation.roc_auc),
        )
    ]

    cm = evaluation.confusion_matrix
    confusion_matrix_meaning = (
        f"Out of {evaluation.test_rows} test rows: {cm.tp} true positives, "
        f"{cm.tn} true negatives, {cm.fp} false positives, {cm.fn} false negatives."
    )

    baseline_comparison = None
    if baseline is not None and baseline.model_id == evaluation.model_id:
        baseline_comparison = baseline.reason

    return EvaluationExplanation(
        model_id=evaluation.model_id,
        metrics=metrics,
        confusion_matrix_meaning=confusion_matrix_meaning,
        baseline_comparison=baseline_comparison,
    )


def explain_guardrail_check(check: ValidationCheck) -> GuardrailCheckExplanation:
    return GuardrailCheckExplanation(
        check=check.check,
        passed=check.passed,
        severity=check.severity,
        meaning=_GUARDRAIL_MEANINGS.get(check.check, "A deterministic PIPER guardrail check."),
        message=check.message,
    )


def explain_failure(failure: FailureInfo) -> FailureExplanation:
    return FailureExplanation(
        category=failure.category,
        message=failure.message,
        retryable=failure.retryable,
        human_intervention_required=failure.human_intervention_required,
        meaning=_FAILURE_CATEGORY_MEANINGS.get(failure.category, "A structured PIPER failure category."),
    )


def build_run_explanation(run_id: str, state) -> RunExplanation:
    """
    Aggregates every explanation type into one RunExplanation for a
    (normally terminal) run's state. Duck-types `state` exactly like
    build_run_summary() — works against a real AgentState or the
    _RunResultState shim tracing.py builds for the API layer.
    """
    cleaning_log = list(getattr(state, "cleaning_log", []) or [])
    feature_log = list(getattr(state, "feature_log", []) or [])
    comparison = getattr(state, "comparison", None)
    evaluation_results = list(getattr(state, "evaluation_results", []) or [])
    baseline = getattr(state, "baseline", None)
    validation = getattr(state, "validation", None)
    failure = getattr(state, "failure", None)

    return RunExplanation(
        run_id=run_id,
        status=getattr(state, "status", "unknown"),
        preprocessing=[explain_operation(op) for op in cleaning_log],
        feature_engineering=[explain_operation(op) for op in feature_log],
        model_selection=explain_model_selection(comparison) if comparison is not None else None,
        evaluation=[explain_evaluation(ev, baseline) for ev in evaluation_results],
        guardrail_checks=[explain_guardrail_check(c) for c in validation.checks] if validation is not None else [],
        failure=explain_failure(failure) if failure is not None else None,
    )
