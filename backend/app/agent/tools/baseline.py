"""
compute_baseline() (section 8).

    model_id (the already-trained/evaluated real model)
        |
        v
    look up its split_id (same discipline as evaluate_model() — never
    accept a separate split_id argument that could point somewhere else)
        |
        v
    fit DummyClassifier(strategy="most_frequent") on the SAME X_train/y_train
        |
        v
    evaluate on the SAME raw X_test/y_test
        |
        v
    compare model's primary metric vs baseline's primary metric against
    the LOCKED BASELINE_POLICY threshold (0.05) — never a scattered
    literal, never LLM-adjustable
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from app.schemas import ToolError, ToolResult
from app.schemas.baseline import BASELINE_POLICY, BaselineComparisonResult, BaselineMetrics
from app.storage import SplitNotFoundError, SplitStore
from app.storage.model_store import InMemoryModelStore, ModelNotFoundError

RANDOM_STATE = 42


def compute_baseline(
    model_id: str,
    model_primary_metric_value: float,
    split_store: SplitStore,
    model_store: InMemoryModelStore,
) -> ToolResult[BaselineComparisonResult]:
    """
    Computes a majority-class baseline on the same split the given
    model was trained/evaluated on, and applies the locked baseline
    gate policy.

    model_primary_metric_value is passed in explicitly (the caller's
    already-computed EvaluationResult.f1) rather than this tool
    re-evaluating the real model itself — that evaluation already
    happened in evaluate_model() and re-doing it here would be
    duplicate work and a second source of truth for the same number.

    Errors:
    - model doesn't exist
    - the model's recorded split no longer exists
    """
    try:
        artifact = model_store.get(model_id)
    except ModelNotFoundError:
        return ToolResult[BaselineComparisonResult](
            success=False,
            tool_name="compute_baseline",
            message=f"Model '{model_id}' does not exist.",
            error=ToolError(
                code="model_not_found",
                message=f"Model '{model_id}' does not exist.",
                details={"model_id": model_id},
            ),
        )

    split_id = artifact.metadata.split_id
    try:
        train_df, test_df = split_store.get(split_id)
    except SplitNotFoundError:
        return ToolResult[BaselineComparisonResult](
            success=False,
            tool_name="compute_baseline",
            message=f"Split '{split_id}' (recorded for model '{model_id}') no longer exists.",
            error=ToolError(
                code="split_not_found",
                message="The split this model was trained against no longer exists.",
                details={"model_id": model_id, "split_id": split_id},
            ),
        )

    target_column = artifact.metadata.target_column
    feature_columns = artifact.metadata.feature_columns

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    baseline_clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline_clf.fit(X_train, y_train)  # fit on train only — same discipline as train_model()
    y_pred_baseline = baseline_clf.predict(X_test)

    majority_class = str(y_train.value_counts().idxmax())
    classes = sorted(y_test.dropna().unique().tolist(), key=str)
    positive_class = classes[1] if len(classes) == 2 else classes[0]

    accuracy = float((y_pred_baseline == y_test.values).mean())

    # The majority-class baseline predicts the SAME label for every
    # row. If that label is not the positive class, precision/recall/
    # F1 for the positive class are mathematically undefined (zero
    # predicted positives) rather than meaningfully "0.0" — represented
    # explicitly as None, never silently coerced to a number.
    predicts_positive_at_all = (y_pred_baseline == positive_class).any()

    if predicts_positive_at_all:
        baseline_precision = float(precision_score(y_test, y_pred_baseline, pos_label=positive_class, zero_division=0))
        baseline_recall = float(recall_score(y_test, y_pred_baseline, pos_label=positive_class, zero_division=0))
        baseline_f1 = float(f1_score(y_test, y_pred_baseline, pos_label=positive_class, zero_division=0))
    else:
        baseline_precision = None
        baseline_recall = None
        baseline_f1 = None

    baseline_metrics = BaselineMetrics(
        accuracy=round(accuracy, 4),
        precision=round(baseline_precision, 4) if baseline_precision is not None else None,
        recall=round(baseline_recall, 4) if baseline_recall is not None else None,
        f1=round(baseline_f1, 4) if baseline_f1 is not None else None,
        majority_class=majority_class,
    )

    primary_metric = BASELINE_POLICY.primary_metric
    baseline_primary_value = getattr(baseline_metrics, primary_metric)

    if baseline_primary_value is None:
        delta = None
        gate_passed = True  # an undefined baseline metric cannot be beaten in the normal sense; see reason
        reason = (
            f"Baseline's {primary_metric} is mathematically undefined (majority class "
            f"'{majority_class}' never predicts the positive class '{positive_class}', so "
            f"precision/recall/F1 have no defined value). Gate passes by policy: an "
            f"undefined baseline cannot be used to reject the model, but this should be "
            f"treated as weak evidence, not strong evidence of model quality."
        )
    else:
        delta = round(model_primary_metric_value - baseline_primary_value, 4)
        gate_passed = delta >= BASELINE_POLICY.minimum_primary_metric_delta
        reason = (
            f"model {primary_metric}={model_primary_metric_value} vs baseline "
            f"{primary_metric}={baseline_primary_value} (delta={delta}); "
            f"{'meets' if gate_passed else 'does not meet'} the required minimum delta "
            f"of {BASELINE_POLICY.minimum_primary_metric_delta}."
        )

    result = BaselineComparisonResult(
        model_id=model_id,
        split_id=split_id,
        baseline=baseline_metrics,
        model_primary_metric_value=model_primary_metric_value,
        baseline_primary_metric_value=baseline_primary_value,
        primary_metric=primary_metric,
        delta=delta,
        minimum_required_delta=BASELINE_POLICY.minimum_primary_metric_delta,
        gate_passed=gate_passed,
        reason=reason,
    )

    return ToolResult[BaselineComparisonResult](
        success=True,
        tool_name="compute_baseline",
        message=reason,
        data=result,
    )
