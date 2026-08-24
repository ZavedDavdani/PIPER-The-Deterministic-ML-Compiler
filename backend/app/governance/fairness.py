"""
Optional subgroup metrics. Columns are never inferred from names.
Insufficient n yields a warning, not a fabricated rate. Results are
statistical measurements, not legal or compliance findings.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app.governance.helpers import field, winner_id
from app.schemas.governance import FairnessReport, SubgroupMetricRow
from app.storage.exceptions import ModelNotFoundError, SplitNotFoundError
from app.storage.model_store import InMemoryModelStore
from app.storage.split_store import SplitStore

MIN_GROUP_SIZE = 30

FAIRNESS_DISCLAIMER = (
    "These figures are statistical subgroup measurements on the recorded "
    "holdout split. They are not a legal, regulatory, or compliance "
    "determination, and PIPER does not treat a named column as a "
    "protected attribute unless the operator explicitly requests it."
)

_REFERENCE_RULE = (
    "Disparate-impact-style ratio uses the largest-n requested group "
    "as the statistical reference (selection_rate_group / "
    "selection_rate_reference). This is not a privileged-class designation."
)


def _positive_label(classifier: Any) -> Any:
    classes = list(getattr(classifier, "classes_", []))
    if len(classes) < 2:
        return classes[0] if classes else None
    return classes[1]


def _binary_scores(y_true: np.ndarray, y_pred: np.ndarray, positive: Any) -> dict[str, Optional[float]]:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    if len(labels) <= 2:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, pos_label=positive, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=positive, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def analyze_subgroups(
    state: Any,
    *,
    columns: list[str],
    model_store: InMemoryModelStore | None,
    split_store: SplitStore | None,
    min_group_size: int = MIN_GROUP_SIZE,
) -> FairnessReport:
    requested = [c for c in columns if isinstance(c, str) and c.strip()]
    requested = list(dict.fromkeys(requested))
    if not requested:
        return FairnessReport(
            status="NOT_REQUESTED",
            requested_columns=[],
            minimum_group_size=min_group_size,
            reference_group_rule=_REFERENCE_RULE,
            disclaimer=FAIRNESS_DISCLAIMER,
            reason="No subgroup columns were supplied. PIPER does not infer protected attributes from names.",
        )
    mid = winner_id(state)
    if not mid or model_store is None or split_store is None:
        return FairnessReport(
            status="NOT_AVAILABLE",
            requested_columns=requested,
            minimum_group_size=min_group_size,
            reference_group_rule=_REFERENCE_RULE,
            disclaimer=FAIRNESS_DISCLAIMER,
            reason="Winning pipeline or evaluation split is not available.",
        )
    try:
        artifact = model_store.get(mid)
    except ModelNotFoundError:
        return FairnessReport(
            status="NOT_AVAILABLE",
            requested_columns=requested,
            minimum_group_size=min_group_size,
            reference_group_rule=_REFERENCE_RULE,
            disclaimer=FAIRNESS_DISCLAIMER,
            reason="Fitted winning pipeline is not in ModelStore.",
        )
    meta = field(artifact, "metadata")
    split_id = field(meta, "split_id")
    target = field(meta, "target_column")
    features = list(field(meta, "feature_columns", default=[]) or [])
    try:
        _, test_df = split_store.get(split_id)
    except SplitNotFoundError:
        return FairnessReport(
            status="NOT_AVAILABLE",
            requested_columns=requested,
            minimum_group_size=min_group_size,
            reference_group_rule=_REFERENCE_RULE,
            disclaimer=FAIRNESS_DISCLAIMER,
            reason="Evaluation split is no longer in SplitStore.",
        )
    missing = [c for c in requested if c not in test_df.columns]
    warnings: list[str] = []
    if missing:
        warnings.append(f"Requested columns not present on the holdout frame: {missing}.")
    usable = [c for c in requested if c in test_df.columns and c != target]
    if target in requested:
        warnings.append("The target column cannot be used as a subgroup column.")
    if not usable:
        return FairnessReport(
            status="NOT_AVAILABLE",
            requested_columns=requested,
            minimum_group_size=min_group_size,
            reference_group_rule=_REFERENCE_RULE,
            warnings=warnings,
            disclaimer=FAIRNESS_DISCLAIMER,
            reason="No usable subgroup columns remained after validation.",
        )
    pipeline = field(artifact, "pipeline")
    named = getattr(pipeline, "named_steps", {}) or {}
    classifier = named.get("classifier") or named.get("classifier")
    positive = _positive_label(classifier)
    y_true = test_df[target].to_numpy()
    y_pred = np.asarray(pipeline.predict(test_df[features]))
    selected = (y_pred == positive) if positive is not None else np.zeros(len(y_pred), dtype=bool)

    groups: list[SubgroupMetricRow] = []
    for column in usable:
        series = test_df[column]
        counts = series.astype("string").fillna("__missing__").value_counts()
        largest = str(counts.index[0]) if len(counts) else None
        ref_rate: Optional[float] = None
        if largest is not None:
            mask_ref = series.astype("string").fillna("__missing__") == largest
            if int(mask_ref.sum()) >= min_group_size:
                ref_rate = float(selected[mask_ref.to_numpy()].mean()) if ref_rate is None else ref_rate
                ref_rate = float(np.mean(selected[mask_ref.to_numpy()]))
        for group_key, n in counts.items():
            group_name = str(group_key)
            mask = (series.astype("string").fillna("__missing__") == group_name).to_numpy()
            n_int = int(n)
            if n_int < min_group_size:
                groups.append(
                    SubgroupMetricRow(
                        column=column,
                        group=group_name,
                        n=n_int,
                        sufficient=False,
                        warning=(
                            f"n={n_int} is below the minimum group size "
                            f"({min_group_size}); rates are not reported."
                        ),
                    )
                )
                warnings.append(
                    f"Column '{column}' group '{group_name}' has n={n_int} < {min_group_size}."
                )
                continue
            scores = _binary_scores(y_true[mask], y_pred[mask], positive)
            sel = float(np.mean(selected[mask]))
            ratio = None
            if ref_rate is not None and ref_rate > 0:
                ratio = sel / ref_rate
            groups.append(
                SubgroupMetricRow(
                    column=column,
                    group=group_name,
                    n=n_int,
                    accuracy=scores["accuracy"],
                    precision=scores["precision"],
                    recall=scores["recall"],
                    f1=scores["f1"],
                    selection_rate=sel,
                    disparate_impact_ratio=ratio,
                    sufficient=True,
                )
            )
    sufficient_any = any(row.sufficient for row in groups)
    status: str = "AVAILABLE" if sufficient_any else "INSUFFICIENT_DATA"
    reason = None if sufficient_any else "Every requested subgroup is below the minimum sample size."
    return FairnessReport(
        status=status,  # type: ignore[arg-type]
        requested_columns=requested,
        minimum_group_size=min_group_size,
        positive_class=None if positive is None else str(positive),
        reference_group_rule=_REFERENCE_RULE,
        groups=groups,
        warnings=warnings,
        disclaimer=FAIRNESS_DISCLAIMER,
        reason=reason,
    )
