"""
Global feature importance from the fitted winning sklearn Pipeline.

Never causal. One-hot columns keep their transformed names. Unsupported
estimators return NOT_AVAILABLE instead of a guessed ranking.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.governance.helpers import field, winner_id
from app.schemas.governance import FeatureImportanceReport, FeatureImportanceRow
from app.storage.exceptions import ModelNotFoundError
from app.storage.model_store import InMemoryModelStore

IMPORTANCE_DISCLAIMER = (
    "These scores describe association with the model's predictions on "
    "the fitted pipeline, not causal effects. Transformed names include "
    "one-hot encoding produced by the recorded preprocessor."
)

_UNSUPPORTED = (
    "Feature importance is only derived for LogisticRegression "
    "(coefficient magnitude/sign) and RandomForestClassifier "
    "(impurity importance). Other estimators are reported as NOT_AVAILABLE."
)


def _source_feature(transformed: str) -> str:
    rest = transformed
    if "__" in rest:
        rest = rest.split("__", 1)[1]
    if "=" in rest:
        return rest.split("=", 1)[0]
    if "_" in rest:
        # OneHotEncoder get_feature_names_out uses "col_category"
        return rest.rsplit("_", 1)[0]
    return rest


def _transformed_names(pipeline: Any) -> Optional[np.ndarray]:
    preprocessor = None
    named = getattr(pipeline, "named_steps", None)
    if named is not None:
        preprocessor = named.get("preprocessor") or named.get("preprocessor") or named.get("preprocess")
    if preprocessor is None:
        return None
    getter = getattr(preprocessor, "get_feature_names_out", None)
    if getter is None:
        return None
    try:
        return np.asarray(getter())
    except Exception:
        return None


def _logistic_importance(classifier: LogisticRegression, names: np.ndarray) -> list[FeatureImportanceRow]:
    coef = np.asarray(classifier.coef_, dtype=float)
    if coef.ndim == 2:
        # Binary: one row. Multiclass: mean absolute, sign from argmax class vs rest is not well-defined.
        if coef.shape[0] == 1:
            weights = coef[0]
        else:
            weights = np.mean(coef, axis=0)
    else:
        weights = coef
    if weights.shape[0] != names.shape[0]:
        raise ValueError("coefficient length does not match transformed feature names")
    rows: list[FeatureImportanceRow] = []
    for name, weight in zip(names, weights):
        label = str(name)
        if weight > 0:
            direction: Optional[str] = "positive"
        elif weight < 0:
            direction = "negative"
        else:
            direction = "neutral"
        rows.append(
            FeatureImportanceRow(
                feature=label,
                transformed_feature=label,
                importance=float(abs(weight)),
                direction=direction,  # type: ignore[arg-type]
                source_feature=_source_feature(label),
            )
        )
    rows.sort(key=lambda row: (-row.importance, row.transformed_feature))
    return rows


def _forest_importance(classifier: RandomForestClassifier, names: np.ndarray) -> list[FeatureImportanceRow]:
    values = np.asarray(classifier.feature_importances_, dtype=float)
    if values.shape[0] != names.shape[0]:
        raise ValueError("importance length does not match transformed feature names")
    rows = [
        FeatureImportanceRow(
            feature=str(name),
            transformed_feature=str(name),
            importance=float(value),
            direction=None,
            source_feature=_source_feature(str(name)),
        )
        for name, value in zip(names, values)
    ]
    rows.sort(key=lambda row: (-row.importance, row.transformed_feature))
    return rows


def extract_feature_importance(
    state: Any,
    model_store: InMemoryModelStore | None,
) -> FeatureImportanceReport:
    mid = winner_id(state)
    if not mid:
        return FeatureImportanceReport(
            status="NOT_AVAILABLE",
            disclaimer=IMPORTANCE_DISCLAIMER,
            reason="No winning model is recorded on this run.",
        )
    if model_store is None:
        return FeatureImportanceReport(
            status="NOT_AVAILABLE",
            disclaimer=IMPORTANCE_DISCLAIMER,
            reason="ModelStore is not available in this process.",
        )
    try:
        artifact = model_store.get(mid)
    except ModelNotFoundError:
        return FeatureImportanceReport(
            status="NOT_AVAILABLE",
            disclaimer=IMPORTANCE_DISCLAIMER,
            reason="The fitted winning pipeline is no longer in ModelStore.",
        )
    pipeline = field(artifact, "pipeline")
    if pipeline is None:
        return FeatureImportanceReport(
            status="NOT_AVAILABLE",
            disclaimer=IMPORTANCE_DISCLAIMER,
            reason="Stored artifact has no fitted pipeline.",
        )
    named = getattr(pipeline, "named_steps", {}) or {}
    classifier = named.get("classifier") or named.get("classifier") or named.get("model")
    algorithm = field(field(artifact, "metadata"), "algorithm")
    names = _transformed_names(pipeline)
    if names is None:
        return FeatureImportanceReport(
            status="NOT_AVAILABLE",
            algorithm=algorithm,
            disclaimer=IMPORTANCE_DISCLAIMER,
            reason="Preprocessor does not expose get_feature_names_out().",
        )
    try:
        if isinstance(classifier, LogisticRegression):
            rows = _logistic_importance(classifier, names)
            method = "logistic_regression_coefficients"
        elif isinstance(classifier, RandomForestClassifier):
            rows = _forest_importance(classifier, names)
            method = "random_forest_impurity"
        else:
            return FeatureImportanceReport(
                status="NOT_AVAILABLE",
                algorithm=algorithm or type(classifier).__name__,
                disclaimer=IMPORTANCE_DISCLAIMER,
                reason=_UNSUPPORTED,
            )
    except Exception as exc:
        return FeatureImportanceReport(
            status="NOT_AVAILABLE",
            algorithm=algorithm,
            disclaimer=IMPORTANCE_DISCLAIMER,
            reason=f"Could not map importance onto transformed features: {exc}",
        )
    return FeatureImportanceReport(
        status="AVAILABLE",
        method=method,  # type: ignore[arg-type]
        algorithm=algorithm,
        rows=rows,
        disclaimer=IMPORTANCE_DISCLAIMER,
    )
