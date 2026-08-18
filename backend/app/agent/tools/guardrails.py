"""
check_data_leakage() — feature-level leakage evidence, not a verdict.

See schemas/guardrails.py module docstring for the full scope
statement (feature-level only; pipeline-level leakage is prevented
structurally elsewhere and not re-verified here).

Locked thresholds (do not loosen without explicit discussion):
    correlation > 0.95              -> high_correlation violation
    categorical purity >= 0.98      -> categorical_near_perfect_association
                                        (every category near-perfectly
                                        predicts one class)
    unique_percentage > 99.0        -> identifier_like_column
    exact equality with target      -> duplicate_of_target
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.agent.tools._profiling_helpers import is_numeric_dtype
from app.schemas import ToolError, ToolResult
from app.schemas.guardrails import (
    ClassCount,
    ConstantFeatureEntry,
    ConstantFeatureReport,
    HighCardinalityEntry,
    HighCardinalityReport,
    HIGH_CARDINALITY_UNIQUENESS_THRESHOLD_PERCENT,
    ImbalanceReport,
    ImbalanceSeverity,
    IMBALANCE_FAILURE_THRESHOLD_PERCENT,
    IMBALANCE_WARNING_THRESHOLD_PERCENT,
    LeakageReport,
    LeakageViolation,
    PipelineValidationResult,
    ValidationCheck,
)
from app.storage import DatasetNotFoundError, DatasetStore

CORRELATION_THRESHOLD = 0.95
CATEGORICAL_PURITY_THRESHOLD = 0.98
IDENTIFIER_UNIQUENESS_THRESHOLD_PERCENT = 99.0
MIN_ROWS_PER_CATEGORY_FOR_PURITY_CHECK = 2  # a category with 1 row is trivially "pure"


def check_data_leakage(
    dataset_id: str,
    target_column: str,
    store: DatasetStore,
) -> ToolResult[LeakageReport]:
    """
    Examines every non-target column for feature-level leakage
    indicators against target_column. Returns evidence (with actual
    numbers), never a bare "leakage=True/False" verdict, and never
    mutates the dataset or removes any column — that decision belongs
    to the agent/graph, using this evidence.

    Errors:
    - dataset doesn't exist
    - target column doesn't exist
    - target column is not binary (this check is specifically for the
      numeric-target-correlation and categorical-purity math below,
      both of which assume a 2-class target)
    - no feature columns to check
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[LeakageReport](
            success=False,
            tool_name="check_data_leakage",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if target_column not in df.columns:
        return ToolResult[LeakageReport](
            success=False,
            tool_name="check_data_leakage",
            message=f"Target column '{target_column}' does not exist.",
            error=ToolError(
                code="column_not_found",
                message=f"Target column '{target_column}' does not exist.",
                details={"dataset_id": dataset_id, "target_column": target_column},
            ),
        )

    target_series = df[target_column]
    distinct_target_values = target_series.dropna().nunique()

    if distinct_target_values != 2:
        return ToolResult[LeakageReport](
            success=False,
            tool_name="check_data_leakage",
            message=(
                f"Target column '{target_column}' has {distinct_target_values} "
                f"distinct value(s); this check requires exactly 2 (binary)."
            ),
            error=ToolError(
                code="target_not_binary",
                message="Target column must be binary for this leakage check.",
                details={"dataset_id": dataset_id, "target_column": target_column, "distinct_values": distinct_target_values},
            ),
        )

    feature_columns = [c for c in df.columns if c != target_column]

    if not feature_columns:
        return ToolResult[LeakageReport](
            success=False,
            tool_name="check_data_leakage",
            message="No feature columns to check (only the target column is present).",
            error=ToolError(
                code="empty_feature_set",
                message="Dataset has no columns other than the target.",
                details={"dataset_id": dataset_id},
            ),
        )

    # Binary-encode the target once for numeric correlation math.
    # Sorted so the mapping is deterministic regardless of label values
    # (e.g. 'No'/'Yes' -> 0/1, matching the same alphabetical
    # convention used in evaluate_model()'s positive-class lookup).
    target_classes = sorted(target_series.dropna().unique().tolist(), key=str)
    target_binary = target_series.map({target_classes[0]: 0, target_classes[1]: 1})

    violations: list[LeakageViolation] = []

    for col in feature_columns:
        series = df[col]

        # --- Check: exact duplicate of target -----------------------
        if series.equals(target_series):
            violations.append(
                LeakageViolation(
                    feature=col,
                    check_type="duplicate_of_target",
                    reason=f"'{col}' is identical to the target column '{target_column}'.",
                    evidence={"identical_row_count": int(len(df))},
                )
            )
            continue  # no need to run other checks on an exact duplicate

        # --- Check: identifier-like column ---------------------------
        # Deliberately excludes numeric columns: a continuous numeric
        # feature (e.g. a price or a measurement) is EXPECTED to be
        # highly or fully unique and that's normal, not suspicious.
        # Identifier risk is specifically about non-numeric columns
        # (strings/categoricals) that are almost entirely unique — the
        # customerID pattern — where high uniqueness genuinely signals
        # "this is a label, not a feature."
        if not is_numeric_dtype(series):
            unique_count = series.nunique(dropna=True)
            unique_percentage = (unique_count / len(df) * 100) if len(df) > 0 else 0.0

            if unique_percentage > IDENTIFIER_UNIQUENESS_THRESHOLD_PERCENT:
                violations.append(
                    LeakageViolation(
                        feature=col,
                        check_type="identifier_like_column",
                        reason=f"'{col}' is {unique_percentage:.2f}% unique — likely an identifier, not a genuine predictive feature.",
                        evidence={"unique_percentage": round(float(unique_percentage), 4), "unique_count": int(unique_count)},
                    )
                )

        # --- Check: numeric feature <-> binary target correlation ----
        if is_numeric_dtype(series):
            # Infinite values must be excluded explicitly before the
            # correlation call — leaving them in produces NaN via a
            # noisy RuntimeWarning from numpy's internal reduction,
            # rather than a clean, deliberate filter. The pd.notna()
            # check below still correctly ignores a NaN correlation
            # result either way, but filtering first is more robust
            # and avoids the warning entirely.
            finite_mask = np.isfinite(series.astype(float)) if series.notna().any() else series.notna()
            valid_mask = series.notna() & target_binary.notna() & finite_mask
            if valid_mask.sum() >= 2 and series[valid_mask].nunique() > 1:
                correlation = series[valid_mask].astype(float).corr(target_binary[valid_mask].astype(float))
                if pd.notna(correlation) and abs(correlation) > CORRELATION_THRESHOLD:
                    violations.append(
                        LeakageViolation(
                            feature=col,
                            check_type="high_correlation",
                            reason=f"'{col}' has {abs(correlation):.4f} correlation with the target, exceeding the {CORRELATION_THRESHOLD} threshold.",
                            evidence={"correlation": round(float(correlation), 4)},
                        )
                    )

        # --- Check: categorical near-perfect association --------------
        else:
            valid_mask = series.notna() & target_series.notna()
            if valid_mask.sum() >= 2:
                grouped = df.loc[valid_mask].groupby(col, observed=True)[target_column]
                category_sizes = grouped.size()
                # Only consider categories with enough rows to be
                # meaningful — a singleton category is trivially "pure"
                # and would otherwise flood this check with noise.
                meaningful_categories = category_sizes[category_sizes >= MIN_ROWS_PER_CATEGORY_FOR_PURITY_CHECK]

                if len(meaningful_categories) >= 1:
                    purities = grouped.apply(
                        lambda s: s.value_counts(normalize=True).iloc[0]
                    )
                    purities = purities[purities.index.isin(meaningful_categories.index)]

                    if len(purities) > 0 and (purities >= CATEGORICAL_PURITY_THRESHOLD).all():
                        min_purity = float(purities.min())
                        violations.append(
                            LeakageViolation(
                                feature=col,
                                check_type="categorical_near_perfect_association",
                                reason=(
                                    f"Every category in '{col}' (with >= {MIN_ROWS_PER_CATEGORY_FOR_PURITY_CHECK} rows) "
                                    f"predicts one target class with >= {CATEGORICAL_PURITY_THRESHOLD*100:.0f}% purity."
                                ),
                                evidence={
                                    "min_category_purity": round(min_purity, 4),
                                    "categories_checked": int(len(purities)),
                                },
                            )
                        )

    report = LeakageReport(
        dataset_id=dataset_id,
        target_column=target_column,
        leakage_detected=len(violations) > 0,
        violations=violations,
        features_checked=feature_columns,
    )

    if violations:
        message = f"Detected {len(violations)} leakage indicator(s) among {len(feature_columns)} feature(s) checked."
    else:
        message = f"No leakage indicators detected by implemented checks, among {len(feature_columns)} feature(s) checked."

    return ToolResult[LeakageReport](
        success=True,
        tool_name="check_data_leakage",
        message=message,
        data=report,
    )


def check_target_imbalance(
    dataset_id: str,
    target_column: str,
    store: DatasetStore,
) -> ToolResult[ImbalanceReport]:
    """
    Reports class distribution evidence for a binary target and
    classifies severity using the migrated, locked, minority-
    percentage-based thresholds (named constants, never scattered
    literals):

        minority_percentage >= 15.0            -> OK
        5.0 <= minority_percentage < 15.0       -> WARNING
        minority_percentage < 5.0               -> FAILURE

    The 5% boundary is strict: exactly 5.0% stays WARNING.

    Does NOT itself decide whether the pipeline is invalid — it only
    reports severity as evidence; validate_pipeline() decides routing.

    Errors:
    - dataset doesn't exist
    - target column doesn't exist
    - target column has fewer than 2 distinct non-null values (can't
      report a meaningful "imbalance" for a constant or empty target)
    - target column has more than 2 distinct values (this check is
      specifically for binary targets, matching check_data_leakage())
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[ImbalanceReport](
            success=False,
            tool_name="check_target_imbalance",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if target_column not in df.columns:
        return ToolResult[ImbalanceReport](
            success=False,
            tool_name="check_target_imbalance",
            message=f"Target column '{target_column}' does not exist.",
            error=ToolError(
                code="column_not_found",
                message=f"Target column '{target_column}' does not exist.",
                details={"dataset_id": dataset_id, "target_column": target_column},
            ),
        )

    target_series = df[target_column]
    distinct_values = target_series.dropna().nunique()

    if distinct_values < 2:
        return ToolResult[ImbalanceReport](
            success=False,
            tool_name="check_target_imbalance",
            message=(
                f"Target column '{target_column}' has {distinct_values} distinct "
                f"non-null value(s); cannot report class imbalance for a "
                f"constant or empty target."
            ),
            error=ToolError(
                code="target_not_binary",
                message="Target column must have at least 2 distinct values.",
                details={"dataset_id": dataset_id, "target_column": target_column, "distinct_values": distinct_values},
            ),
        )

    if distinct_values > 2:
        return ToolResult[ImbalanceReport](
            success=False,
            tool_name="check_target_imbalance",
            message=(
                f"Target column '{target_column}' has {distinct_values} distinct "
                f"value(s); this check requires exactly 2 (binary)."
            ),
            error=ToolError(
                code="target_not_binary",
                message="Target column must have exactly 2 distinct values for this check.",
                details={"dataset_id": dataset_id, "target_column": target_column, "distinct_values": distinct_values},
            ),
        )

    value_counts = target_series.dropna().value_counts()
    total = int(value_counts.sum())

    class_counts = [
        ClassCount(label=str(label), count=int(count), percentage=round(count / total * 100, 4))
        for label, count in value_counts.items()
    ]
    sorted_counts = sorted(class_counts, key=lambda c: c.percentage)
    minority = sorted_counts[0]
    majority = sorted_counts[-1]

    if minority.percentage < IMBALANCE_FAILURE_THRESHOLD_PERCENT:
        severity = ImbalanceSeverity.FAILURE
        reason = (
            f"Minority class '{minority.label}' is {minority.percentage:.2f}% of the data, "
            f"below the {IMBALANCE_FAILURE_THRESHOLD_PERCENT}% failure threshold — severe imbalance."
        )
    elif minority.percentage < IMBALANCE_WARNING_THRESHOLD_PERCENT:
        severity = ImbalanceSeverity.WARNING
        reason = (
            f"Minority class '{minority.label}' is {minority.percentage:.2f}% of the data, "
            f"below the {IMBALANCE_WARNING_THRESHOLD_PERCENT}% warning threshold."
        )
    else:
        severity = ImbalanceSeverity.OK
        reason = f"Minority class '{minority.label}' is {minority.percentage:.2f}% of the data — within acceptable balance."

    report = ImbalanceReport(
        dataset_id=dataset_id,
        target_column=target_column,
        class_counts=class_counts,
        minority_label=minority.label,
        minority_percentage=minority.percentage,
        majority_label=majority.label,
        majority_percentage=majority.percentage,
        severity=severity,
        warning_threshold_percent=IMBALANCE_WARNING_THRESHOLD_PERCENT,
        failure_threshold_percent=IMBALANCE_FAILURE_THRESHOLD_PERCENT,
        reason=reason,
    )

    return ToolResult[ImbalanceReport](
        success=True,
        tool_name="check_target_imbalance",
        message=reason,
        data=report,
    )


def check_constant_features(
    dataset_id: str,
    store: DatasetStore,
    target_column: str | None = None,
) -> ToolResult[ConstantFeatureReport]:
    """
    Reports columns with zero variance (a single distinct non-null
    value, or entirely null). target_column, if given, is excluded
    from the check — a constant target is already caught by
    check_target_imbalance()'s target_not_binary rejection, so
    re-flagging it here would be redundant noise, not new evidence.

    Does not remove any column — reports evidence only.

    Errors:
    - dataset doesn't exist
    - dataset has zero columns (nothing to check)
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[ConstantFeatureReport](
            success=False,
            tool_name="check_constant_features",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    columns_to_check = [c for c in df.columns if c != target_column]

    if not columns_to_check:
        return ToolResult[ConstantFeatureReport](
            success=False,
            tool_name="check_constant_features",
            message="No columns to check (dataset has no columns other than the target).",
            error=ToolError(
                code="empty_feature_set",
                message="Dataset has no columns to check.",
                details={"dataset_id": dataset_id},
            ),
        )

    constant_entries: list[ConstantFeatureEntry] = []

    for col in columns_to_check:
        series = df[col]
        non_null_count = int(series.notna().sum())
        unique_non_null = series.dropna().nunique()

        if non_null_count == 0:
            constant_entries.append(
                ConstantFeatureEntry(column=col, constant_value="NaN-only", non_null_count=0)
            )
        elif unique_non_null == 1:
            value = series.dropna().iloc[0]
            constant_entries.append(
                ConstantFeatureEntry(
                    column=col,
                    constant_value=str(value),
                    non_null_count=non_null_count,
                )
            )

    report = ConstantFeatureReport(
        dataset_id=dataset_id,
        constant_columns=constant_entries,
        columns_checked=columns_to_check,
    )

    if constant_entries:
        message = f"Found {len(constant_entries)} constant column(s) among {len(columns_to_check)} checked: {[e.column for e in constant_entries]}."
    else:
        message = f"No constant columns found among {len(columns_to_check)} checked."

    return ToolResult[ConstantFeatureReport](
        success=True,
        tool_name="check_constant_features",
        message=message,
        data=report,
    )


def check_high_cardinality(
    dataset_id: str,
    store: DatasetStore,
    target_column: str | None = None,
) -> ToolResult[HighCardinalityReport]:
    """
    Reports non-numeric columns exceeding the locked 99% uniqueness
    threshold — identifier-like columns such as customerID.

    Deliberately excludes numeric columns from this check entirely
    (same distinction as check_data_leakage()'s identifier check,
    reused rather than reimplemented): a continuous numeric feature
    (price, measurement, etc.) being highly or fully unique is normal
    and expected, not a sign it's an identifier. Only non-numeric
    (text/categorical) columns are examined.

    Does not remove any column — reports evidence only.

    Errors:
    - dataset doesn't exist
    - no columns to check (after excluding target_column, if given)
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[HighCardinalityReport](
            success=False,
            tool_name="check_high_cardinality",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    columns_to_check = [c for c in df.columns if c != target_column]

    if not columns_to_check:
        return ToolResult[HighCardinalityReport](
            success=False,
            tool_name="check_high_cardinality",
            message="No columns to check (dataset has no columns other than the target).",
            error=ToolError(
                code="empty_feature_set",
                message="Dataset has no columns to check.",
                details={"dataset_id": dataset_id},
            ),
        )

    suspicious: list[HighCardinalityEntry] = []
    rows = len(df)

    for col in columns_to_check:
        series = df[col]

        if is_numeric_dtype(series):
            continue  # deliberately excluded — see docstring/scope_note

        unique_count = series.nunique(dropna=True)
        unique_percentage = (unique_count / rows * 100) if rows > 0 else 0.0

        if unique_percentage > HIGH_CARDINALITY_UNIQUENESS_THRESHOLD_PERCENT:
            suspicious.append(
                HighCardinalityEntry(
                    column=col,
                    unique_count=int(unique_count),
                    unique_percentage=round(float(unique_percentage), 4),
                    reason=f"'{col}' is {unique_percentage:.2f}% unique (non-numeric) — likely an identifier or free-text column.",
                )
            )

    report = HighCardinalityReport(
        dataset_id=dataset_id,
        suspicious_columns=suspicious,
        columns_checked=columns_to_check,
    )

    if suspicious:
        message = f"Found {len(suspicious)} high-cardinality column(s) among {len(columns_to_check)} checked: {[e.column for e in suspicious]}."
    else:
        message = f"No high-cardinality columns found among {len(columns_to_check)} checked."

    return ToolResult[HighCardinalityReport](
        success=True,
        tool_name="check_high_cardinality",
        message=message,
        data=report,
    )


# Locked from the original guardrail contract: accuracy/F1 exceeding
# this is "suspicious" and forces a leakage re-check.
SUSPICIOUS_METRIC_THRESHOLD = 0.98


def validate_pipeline(
    dataset_id: str,
    target_column: str,
    store: DatasetStore,
    evaluation_f1: float | None = None,
    evaluation_accuracy: float | None = None,
    baseline_comparison=None,
) -> ToolResult[PipelineValidationResult]:
    """
    The deterministic gate. Runs check_data_leakage(),
    check_target_imbalance(), check_constant_features(), and
    check_high_cardinality() against the current dataset state, and —
    if evaluation_f1/evaluation_accuracy are provided — applies the
    locked suspicious-metric rule (>0.98 forces the leakage finding to
    be treated as a hard failure even if it would otherwise only be a
    warning).

    baseline_comparison, if provided, is a BaselineComparisonResult
    (from compute_baseline()) — its gate_passed field is checked and,
    if False, becomes an ERROR-severity BASELINE_GATE_FAILED finding.
    A model that cannot beat a trivial majority-class baseline by the
    locked minimum delta (section 8) must never PASS validation,
    regardless of how clean the rest of the pipeline looks.

    Severity policy (locked for V1):
    - leakage_detected=True                       -> ERROR (always)
    - severely_imbalanced=True                     -> WARNING
      (per the locked rule: severe imbalance means F1/ROC-AUC are
      REQUIRED, not that the pipeline is invalid outright — enforcing
      "F1 was actually used" is a downstream reporting concern, not
      something this tool can check by inspecting the dataset alone)
    - any constant feature present                 -> WARNING
      (a constant feature is dead weight, not inherently invalidating
      — cleaning may not have removed it yet, which is a planning gap,
      not proof of a broken pipeline)
    - any high-cardinality column present           -> WARNING
      (same reasoning as constant features — evidence for planning,
      not automatic invalidity)
    - evaluation metric (if provided) > 0.98        -> ERROR, AND
      escalates check_data_leakage()'s finding: if leakage evidence
      exists at all (even a single violation) alongside a suspicious
      metric, this is treated as strong corroborating evidence, not
      coincidence
    - baseline_comparison.gate_passed=False (if provided) -> ERROR
      (a model indistinguishable from a trivial baseline must never
      PASS)

    valid=False if and only if there is at least one ERROR-severity
    check. WARNING-severity findings are surfaced but do not block.

    Errors (tool-level, not guardrail findings):
    - dataset doesn't exist
    - target column doesn't exist
    """
    try:
        df = store.get(dataset_id)
    except DatasetNotFoundError:
        return ToolResult[PipelineValidationResult](
            success=False,
            tool_name="validate_pipeline",
            message=f"Dataset '{dataset_id}' does not exist.",
            error=ToolError(
                code="dataset_not_found",
                message=f"Dataset '{dataset_id}' does not exist.",
                details={"dataset_id": dataset_id},
            ),
        )

    if target_column not in df.columns:
        return ToolResult[PipelineValidationResult](
            success=False,
            tool_name="validate_pipeline",
            message=f"Target column '{target_column}' does not exist.",
            error=ToolError(
                code="column_not_found",
                message=f"Target column '{target_column}' does not exist.",
                details={"dataset_id": dataset_id, "target_column": target_column},
            ),
        )

    checks: list[ValidationCheck] = []

    # --- Leakage ------------------------------------------------------
    leakage_result = check_data_leakage(dataset_id, target_column, store)
    leakage_has_evidence = False
    if leakage_result.success:
        leakage_has_evidence = leakage_result.data.leakage_detected
        checks.append(
            ValidationCheck(
                check="data_leakage",
                passed=not leakage_result.data.leakage_detected,
                severity="error" if leakage_result.data.leakage_detected else "info",
                message=leakage_result.message,
            )
        )
    else:
        # The underlying check itself couldn't run (e.g. non-binary
        # target) — this is reported as a warning, not silently
        # skipped, since "we couldn't check" is itself evidence.
        checks.append(
            ValidationCheck(
                check="data_leakage",
                passed=True,
                severity="warning",
                message=f"Leakage check could not run: {leakage_result.message}",
            )
        )

    # --- Imbalance ------------------------------------------------------
    imbalance_result = check_target_imbalance(dataset_id, target_column, store)
    if imbalance_result.success:
        # BUG FIX: previously read imbalance_result.data.severely_imbalanced
        # (True iff severity == FAILURE) instead of severity itself.
        # That collapsed WARNING and OK into the same "False" value, so
        # a genuinely WARNING-tier imbalance was silently reported as
        # severity="info" and never surfaced in
        # PipelineValidationResult.warnings. Fixed by reading the
        # authoritative severity enum directly.
        imbalance_severity = imbalance_result.data.severity

        if imbalance_severity is ImbalanceSeverity.OK:
            checks.append(
                ValidationCheck(
                    check="target_imbalance",
                    passed=True,
                    severity="info",
                    message=imbalance_result.message,
                )
            )
        else:
            # Both WARNING and FAILURE tiers surface as a non-blocking
            # "warning"-severity ValidationCheck — imbalance, even at
            # its worst (FAILURE) tier, does not by itself invalidate
            # the pipeline. Only leakage and the baseline gate are
            # hard errors in the locked severity policy.
            checks.append(
                ValidationCheck(
                    check="target_imbalance",
                    passed=True,
                    severity="warning",
                    message=imbalance_result.message,
                )
            )
    else:
        checks.append(
            ValidationCheck(
                check="target_imbalance",
                passed=True,
                severity="warning",
                message=f"Imbalance check could not run: {imbalance_result.message}",
            )
        )

    # --- Constant features ------------------------------------------------
    constant_result = check_constant_features(dataset_id, store, target_column=target_column)
    if constant_result.success:
        has_constant = len(constant_result.data.constant_columns) > 0
        checks.append(
            ValidationCheck(
                check="constant_features",
                passed=not has_constant,
                severity="warning" if has_constant else "info",
                message=constant_result.message,
            )
        )
    else:
        checks.append(
            ValidationCheck(
                check="constant_features",
                passed=True,
                severity="warning",
                message=f"Constant-feature check could not run: {constant_result.message}",
            )
        )

    # --- High cardinality ---------------------------------------------
    cardinality_result = check_high_cardinality(dataset_id, store, target_column=target_column)
    if cardinality_result.success:
        has_high_cardinality = len(cardinality_result.data.suspicious_columns) > 0
        checks.append(
            ValidationCheck(
                check="high_cardinality",
                passed=not has_high_cardinality,
                severity="warning" if has_high_cardinality else "info",
                message=cardinality_result.message,
            )
        )
    else:
        checks.append(
            ValidationCheck(
                check="high_cardinality",
                passed=True,
                severity="warning",
                message=f"High-cardinality check could not run: {cardinality_result.message}",
            )
        )

    # --- Suspicious evaluation metric (locked rule) ---------------------
    if evaluation_f1 is not None or evaluation_accuracy is not None:
        suspicious_metric = (
            (evaluation_f1 is not None and evaluation_f1 > SUSPICIOUS_METRIC_THRESHOLD)
            or (evaluation_accuracy is not None and evaluation_accuracy > SUSPICIOUS_METRIC_THRESHOLD)
        )
        if suspicious_metric:
            corroboration = " Leakage evidence was also found, strongly corroborating this." if leakage_has_evidence else " No feature-level leakage evidence was found by the implemented checks, but the metric itself remains suspicious."
            checks.append(
                ValidationCheck(
                    check="suspicious_evaluation_metric",
                    passed=False,
                    severity="error",
                    message=(
                        f"Evaluation metric exceeds {SUSPICIOUS_METRIC_THRESHOLD} "
                        f"(F1={evaluation_f1}, accuracy={evaluation_accuracy}) — "
                        f"this is suspicious for real-world tabular classification "
                        f"and forces a leakage investigation." + corroboration
                    ),
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    check="suspicious_evaluation_metric",
                    passed=True,
                    severity="info",
                    message=f"Evaluation metric within plausible range (F1={evaluation_f1}, accuracy={evaluation_accuracy}).",
                )
            )

    # --- Baseline gate (section 8) ---------------------------------------
    if baseline_comparison is not None:
        if baseline_comparison.gate_passed:
            checks.append(
                ValidationCheck(
                    check="baseline_gate",
                    passed=True,
                    severity="info",
                    message=baseline_comparison.reason,
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    check="baseline_gate",
                    passed=False,
                    severity="error",
                    message=baseline_comparison.reason,
                )
            )

    violations = [c for c in checks if not c.passed and c.severity == "error"]
    warnings_list = [c for c in checks if c.severity == "warning"]
    valid = len(violations) == 0

    result = PipelineValidationResult(
        dataset_id=dataset_id,
        target_column=target_column,
        valid=valid,
        checks=checks,
        violations=violations,
        warnings=warnings_list,
    )

    if valid:
        message = f"Pipeline validation passed: no guardrail violations among {len(checks)} check(s) run ({len(warnings_list)} warning(s))."
    else:
        message = f"Pipeline validation FAILED: {len(violations)} violation(s) found among {len(checks)} check(s) run."

    return ToolResult[PipelineValidationResult](
        success=True,
        tool_name="validate_pipeline",
        message=message,
        data=result,
    )
