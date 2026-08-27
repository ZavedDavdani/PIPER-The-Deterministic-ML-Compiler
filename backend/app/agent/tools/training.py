"""
train_model() — the architectural core of the training tool group.

    SplitStore (raw X_train/X_test, untouched)
        |
        v
    build ColumnTransformer (OneHotEncoder + StandardScaler)
        |
    add classifier
        |
    fit ONLY on X_train / y_train
        |
        v
    one sklearn.Pipeline object
        |
        v
    ModelStore, keyed by model_id

The LLM (eventually) may only choose from: algorithm name, and
hyperparameters within a locked allowlist/bounds. It can never supply
an arbitrary estimator class or arbitrary parameters — every
disallowed input is rejected here, deterministically, before any
fitting happens.
"""

from __future__ import annotations

import time
import uuid

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas import ToolError, ToolResult
from app.schemas.training import FeatureEngineeringIntent, TrainingResult
from app.storage import SplitNotFoundError, SplitStore
from app.storage.model_store import InMemoryModelStore, ModelArtifact, ModelArtifactMetadata

RANDOM_STATE = 42

# --- Locked parameter allowlists (from the earlier-locked contract) ------

_LOGISTIC_REGRESSION_ALLOWED_PARAMS = {"C", "max_iter"}
_LOGISTIC_REGRESSION_BOUNDS = {
    "C": (0.001, 100.0),
    "max_iter": (100, 5000),
}
# penalty is fixed at "l2" for V1 — not exposed as a settable param at all.

_RANDOM_FOREST_ALLOWED_PARAMS = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"}
_RANDOM_FOREST_BOUNDS = {
    "n_estimators": (50, 500),
    "max_depth": (1, 50),  # None is also valid — handled separately, not in this numeric range
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 20),
}

_ALGORITHM_ALLOWED_PARAMS = {
    "logistic_regression": _LOGISTIC_REGRESSION_ALLOWED_PARAMS,
    "random_forest": _RANDOM_FOREST_ALLOWED_PARAMS,
}
_ALGORITHM_BOUNDS = {
    "logistic_regression": _LOGISTIC_REGRESSION_BOUNDS,
    "random_forest": _RANDOM_FOREST_BOUNDS,
}


def _validate_params(algorithm: str, params: dict) -> ToolError | None:
    """
    Returns a ToolError describing the first validation failure, or
    None if params are fully valid. This is the enforcement point that
    guarantees the LLM can request `n_estimators=200` but never
    `estimator=<arbitrary class>` — only keys in the allowlist are
    ever accepted, and every accepted key is bounds-checked.
    """
    allowed_keys = _ALGORITHM_ALLOWED_PARAMS[algorithm]
    bounds = _ALGORITHM_BOUNDS[algorithm]

    unknown_keys = set(params.keys()) - allowed_keys
    if unknown_keys:
        return ToolError(
            code="disallowed_hyperparameter",
            message=f"Parameter(s) not permitted for {algorithm}: {sorted(unknown_keys)}.",
            details={"algorithm": algorithm, "disallowed": sorted(unknown_keys), "allowed": sorted(allowed_keys)},
        )

    for key, value in params.items():
        if key == "max_depth" and value is None:
            continue  # None is explicitly valid for random_forest's max_depth

        low, high = bounds[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return ToolError(
                code="invalid_hyperparameter_type",
                message=f"Parameter '{key}' must be numeric; got {type(value).__name__}.",
                details={"algorithm": algorithm, "parameter": key, "value": value},
            )
        if key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_iter"):
            if isinstance(value, float) and not value.is_integer():
                return ToolError(
                    code="invalid_hyperparameter_type",
                    message=f"Parameter '{key}' must be a whole number; got {value}.",
                    details={"algorithm": algorithm, "parameter": key, "value": value},
                )
        if not (low <= value <= high):
            return ToolError(
                code="hyperparameter_out_of_bounds",
                message=f"Parameter '{key}'={value} is outside the allowed bounds [{low}, {high}] for {algorithm}.",
                details={"algorithm": algorithm, "parameter": key, "value": value, "bounds": [low, high]},
            )

    return None


def _build_classifier(algorithm: str, params: dict):
    if algorithm == "logistic_regression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=RANDOM_STATE,
        )
    else:  # random_forest
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=RANDOM_STATE,
        )


def train_model(
    split_id: str,
    target_column: str,
    algorithm: str,
    params: dict,
    feature_intent: FeatureEngineeringIntent,
    split_store: SplitStore,
    model_store: InMemoryModelStore,
) -> ToolResult[TrainingResult]:
    """
    Assembles a ColumnTransformer (OneHotEncoder for feature_intent's
    categorical_columns, StandardScaler for numeric_columns_to_scale)
    plus a classifier into one sklearn Pipeline, fits it ONLY on the
    train split, and stores the fitted Pipeline in ModelStore.

    Rejects (all checked before any fitting occurs):
    - unknown algorithm
    - disallowed hyperparameters / out-of-bounds hyperparameters
    - missing/nonexistent split_id
    - target column not present in the split
    - target column listed as a feature (categorical_columns or
      numeric_columns_to_scale) — this would be leakage by definition
    - empty feature set (no categorical AND no numeric columns given)
    - feature-intent columns that don't actually exist in the split
    """
    if algorithm not in ("logistic_regression", "random_forest"):
        return ToolResult[TrainingResult](
            success=False,
            tool_name="train_model",
            message=f"Unknown algorithm '{algorithm}'.",
            error=ToolError(
                code="unknown_algorithm",
                message="algorithm must be one of: logistic_regression, random_forest.",
                details={"algorithm": algorithm},
            ),
        )

    param_error = _validate_params(algorithm, params)
    if param_error is not None:
        return ToolResult[TrainingResult](
            success=False,
            tool_name="train_model",
            message=param_error.message,
            error=param_error,
        )

    try:
        train_df, test_df = split_store.get(split_id)
    except SplitNotFoundError:
        return ToolResult[TrainingResult](
            success=False,
            tool_name="train_model",
            message=f"Split '{split_id}' does not exist.",
            error=ToolError(
                code="split_not_found",
                message=f"Split '{split_id}' does not exist.",
                details={"split_id": split_id},
            ),
        )

    if target_column not in train_df.columns:
        return ToolResult[TrainingResult](
            success=False,
            tool_name="train_model",
            message=f"Target column '{target_column}' not present in split '{split_id}'.",
            error=ToolError(
                code="invalid_target",
                message="Target column does not exist in the split.",
                details={"split_id": split_id, "target_column": target_column},
            ),
        )

    categorical_columns = feature_intent.categorical_columns
    numeric_columns = feature_intent.numeric_columns_to_scale

    if target_column in categorical_columns or target_column in numeric_columns:
        return ToolResult[TrainingResult](
            success=False,
            tool_name="train_model",
            message=f"Target column '{target_column}' cannot also be listed as a feature.",
            error=ToolError(
                code="target_leakage_in_features",
                message="Target column must not appear in the feature-engineering intent.",
                details={"target_column": target_column},
            ),
        )

    if not categorical_columns and not numeric_columns:
        return ToolResult[TrainingResult](
            success=False,
            tool_name="train_model",
            message="Feature set is empty: no categorical or numeric columns specified.",
            error=ToolError(
                code="empty_feature_set",
                message="At least one feature column (categorical or numeric) is required.",
                details={},
            ),
        )

    all_feature_columns = categorical_columns + numeric_columns
    missing_feature_columns = [c for c in all_feature_columns if c not in train_df.columns]
    if missing_feature_columns:
        return ToolResult[TrainingResult](
            success=False,
            tool_name="train_model",
            message=f"Feature column(s) not found in split: {missing_feature_columns}.",
            error=ToolError(
                code="feature_column_not_found",
                message="One or more feature-intent columns do not exist in the split.",
                details={"missing_columns": missing_feature_columns},
            ),
        )

    duplicate_columns = set(categorical_columns) & set(numeric_columns)
    if duplicate_columns:
        return ToolResult[TrainingResult](
            success=False,
            tool_name="train_model",
            message=f"Column(s) listed as both categorical and numeric: {sorted(duplicate_columns)}.",
            error=ToolError(
                code="ambiguous_feature_type",
                message="A column cannot be both categorical and numeric-to-scale.",
                details={"columns": sorted(duplicate_columns)},
            ),
        )

    # --- Build the ColumnTransformer + classifier as ONE Pipeline ---
    transformers = []
    if categorical_columns:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        )
    if numeric_columns:
        transformers.append(("numeric", StandardScaler(), numeric_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = _build_classifier(algorithm, params)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    X_train = train_df[all_feature_columns]
    y_train = train_df[target_column]

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)  # fit happens HERE, on train data ONLY
    duration = time.perf_counter() - start

    model_id = f"model_{uuid.uuid4().hex[:8]}"

    metadata = ModelArtifactMetadata(
        model_id=model_id,
        algorithm=algorithm,
        parameters=params,
        split_id=split_id,
        target_column=target_column,
        feature_columns=all_feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        random_state=RANDOM_STATE,
        training_rows=len(train_df),
    )
    artifact = ModelArtifact(metadata=metadata, pipeline=pipeline)
    model_store.save(model_id, artifact)

    result = TrainingResult(
        model_id=model_id,
        algorithm=algorithm,
        parameters=params,
        split_id=split_id,
        training_rows=len(train_df),
        feature_count=len(all_feature_columns),
        training_duration_seconds=round(duration, 4),
    )

    return ToolResult[TrainingResult](
        success=True,
        tool_name="train_model",
        message=(
            f"Trained {algorithm} on {len(train_df)} rows, "
            f"{len(all_feature_columns)} feature(s), in {duration:.4f}s."
        ),
        data=result,
    )
