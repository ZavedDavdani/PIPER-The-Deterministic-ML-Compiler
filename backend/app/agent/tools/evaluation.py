"""
evaluate_model() and compare_models().

    model_id + split_id
        |
        v
    retrieve fitted Pipeline (already fit, never refit here)
    retrieve untouched X_test
        |
        v
    pipeline.predict(X_test) / pipeline.predict_proba(X_test)
        |
        v
    accuracy / precision / recall / f1 / roc_auc / confusion matrix

The positive class for probability-based metrics (roc_auc) is looked
up from the fitted classifier's classes_ attribute rather than
assumed by column position — sklearn sorts classes_ alphabetically,
so which column is "positive" depends on the actual label values
(e.g. 'Yes' sorts after 'No', landing at index 1 — but this must never
be hardcoded, since a different dataset's labels could sort the
opposite way).
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.schemas import ToolError, ToolResult
from app.schemas.evaluation import (
    ConfusionMatrix,
    EvaluationResult,
    ModelComparison,
    ModelComparisonEntry,
)
from app.storage import SplitNotFoundError, SplitStore
from app.storage.model_store import InMemoryModelStore, ModelNotFoundError


def evaluate_model(
    model_id: str,
    split_store: SplitStore,
    model_store: InMemoryModelStore,
) -> ToolResult[EvaluationResult]:
    """
    Evaluates a trained model against its associated test split.

    Uses the split_id recorded in the model's own metadata (set by
    train_model()) rather than accepting split_id as a separate
    argument — this makes it structurally impossible to accidentally
    evaluate a model against a DIFFERENT split than the one whose
    train portion it was fit on, which would be a subtle but serious
    bug (evaluating against data that overlaps with training).

    Never calls .fit() on anything — only .predict()/.predict_proba()
    on the already-fitted Pipeline. See test_evaluation.py for a
    behavioral proof of this, not just a code-reading assumption.
    """
    try:
        artifact = model_store.get(model_id)
    except ModelNotFoundError:
        return ToolResult[EvaluationResult](
            success=False,
            tool_name="evaluate_model",
            message=f"Model '{model_id}' does not exist.",
            error=ToolError(
                code="model_not_found",
                message=f"Model '{model_id}' does not exist.",
                details={"model_id": model_id},
            ),
        )

    split_id = artifact.metadata.split_id
    try:
        _, test_df = split_store.get(split_id)
    except SplitNotFoundError:
        return ToolResult[EvaluationResult](
            success=False,
            tool_name="evaluate_model",
            message=f"Split '{split_id}' (recorded for model '{model_id}') no longer exists.",
            error=ToolError(
                code="split_not_found",
                message="The split this model was trained against no longer exists.",
                details={"model_id": model_id, "split_id": split_id},
            ),
        )

    target_column = artifact.metadata.target_column
    feature_columns = artifact.metadata.feature_columns

    if target_column not in test_df.columns:
        return ToolResult[EvaluationResult](
            success=False,
            tool_name="evaluate_model",
            message=f"Target column '{target_column}' not present in test split.",
            error=ToolError(
                code="invalid_target",
                message="Target column missing from the test split.",
                details={"model_id": model_id, "target_column": target_column},
            ),
        )

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    y_pred = artifact.pipeline.predict(X_test)

    classifier = artifact.pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    # Positive class convention for V1: whichever label is NOT the
    # first alphabetically (matches how the rest of the system already
    # treats binary targets — e.g. Telco's 'Yes'/'No' with 'Yes' as
    # the outcome of interest). Looked up dynamically, never hardcoded
    # by column position.
    positive_class = classes[1]
    positive_class_idx = classes.index(positive_class)

    y_proba = artifact.pipeline.predict_proba(X_test)[:, positive_class_idx]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    cm = sk_confusion_matrix(y_test, y_pred, labels=classes)
    # cm layout with labels=[negative, positive]:
    #   [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    result = EvaluationResult(
        model_id=model_id,
        split_id=split_id,
        accuracy=round(float(accuracy), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1=round(float(f1), 4),
        roc_auc=round(float(roc_auc), 4),
        confusion_matrix=ConfusionMatrix(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        test_rows=len(test_df),
    )

    return ToolResult[EvaluationResult](
        success=True,
        tool_name="evaluate_model",
        message=(
            f"Evaluated model '{model_id}' on {len(test_df)} test rows: "
            f"F1={result.f1}, ROC-AUC={result.roc_auc}."
        ),
        data=result,
    )


def _build_selection_justification(
    recommended: ModelComparisonEntry, entries: list[ModelComparisonEntry]
) -> str:
    """
    Pre-6A Polish (Model-Selection Transparency): a deterministic,
    template-based justification for why `recommended` was selected —
    built ONLY from the real F1 scores already computed in `entries`.
    No LLM involvement whatsoever, matching the same "deterministic
    code decides, LLM never influences the outcome" principle that
    compare_models() itself already enforces for the selection.
    """
    others = [e for e in entries if e.model_id != recommended.model_id]
    if not others:
        return f"{recommended.algorithm} selected: only candidate evaluated, F1={recommended.f1}."

    others_sorted = sorted(others, key=lambda e: e.f1, reverse=True)
    comparisons = ", ".join(f"{e.f1} for {e.algorithm}" for e in others_sorted)
    return f"{recommended.algorithm} selected: F1={recommended.f1} vs. {comparisons}."


def compare_models(
    model_ids: list[str],
    split_store: SplitStore,
    model_store: InMemoryModelStore,
) -> ToolResult[ModelComparison]:
    """
    Evaluates each model_id and recommends one using F1 as the
    selection metric — fixed for V1, never left to the LLM to choose,
    which would allow metric-shopping to favor a preferred model.

    Errors:
    - empty model_ids list
    - any model_id doesn't exist (whole comparison fails rather than
      silently comparing a partial list — a partial comparison could
      mislead the agent into thinking it saw all candidates)
    """
    if not model_ids:
        return ToolResult[ModelComparison](
            success=False,
            tool_name="compare_models",
            message="model_ids must contain at least one model.",
            error=ToolError(
                code="empty_model_list",
                message="At least one model_id is required.",
                details={},
            ),
        )

    entries: list[ModelComparisonEntry] = []
    for model_id in model_ids:
        eval_result = evaluate_model(model_id, split_store, model_store)
        if not eval_result.success:
            return ToolResult[ModelComparison](
                success=False,
                tool_name="compare_models",
                message=f"Could not evaluate model '{model_id}': {eval_result.message}",
                error=eval_result.error,
            )

        try:
            algorithm = model_store.get(model_id).metadata.algorithm
        except ModelNotFoundError:
            algorithm = "unknown"

        entries.append(
            ModelComparisonEntry(
                model_id=model_id,
                algorithm=algorithm,
                accuracy=eval_result.data.accuracy,
                precision=eval_result.data.precision,
                recall=eval_result.data.recall,
                f1=eval_result.data.f1,
                roc_auc=eval_result.data.roc_auc,
            )
        )

    recommended = max(entries, key=lambda e: e.f1)
    justification = _build_selection_justification(recommended, entries)

    comparison = ModelComparison(
        models=entries,
        recommended_model_id=recommended.model_id,
        selection_metric="f1",
        justification=justification,
    )

    return ToolResult[ModelComparison](
        success=True,
        tool_name="compare_models",
        message=(
            f"Compared {len(entries)} model(s); recommended '{recommended.model_id}' "
            f"({recommended.algorithm}) with F1={recommended.f1}."
        ),
        data=comparison,
    )
