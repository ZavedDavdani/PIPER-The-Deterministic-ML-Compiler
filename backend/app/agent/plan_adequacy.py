"""
Deterministic Plan Adequacy evaluation.

Sits AFTER `validate_proposed_plan()` (structural validity) and BEFORE
execution. Answers a question no existing layer answered: given the
deterministic evidence PIPER already has about this dataset, does the
proposed plan actually address the conditions that matter before TRAIN?

    LLM plan
        v
    validate_proposed_plan()   -- well-formed? (unchanged, still the
        v                          sole authority on VALIDITY)
    evaluate_plan_adequacy()   -- sufficient?  (this module)
        v
    execution

READ-ONLY BY CONSTRUCTION. This module imports no store, mutates
nothing, and returns findings only. It cannot add, remove, reorder, or
rewrite a plan step, and cannot drop a column. Correction is the
existing REPLAN loop's job, driven by an LLM — never this module's.

=====================================================================
VERIFIED V1 EXECUTION CONTRACT (measured, not assumed)
=====================================================================

Everything below was verified empirically against this checkout's
actual sklearn (1.5.2) and the real `train_model()` pipeline shape, per
CLAUDE.md's standing "measure, don't guess" rule:

1. A column becomes a training feature IF AND ONLY IF it appears in an
   `encode_categorical_features` or `scale_features` step.
   `_feature_intent_from_plan()` builds the intent from exactly those
   two tools, and `train_model()` builds its ColumnTransformer with
   `remainder="drop"` — so any column not named by one of them never
   reaches the model at all.

2. `StandardScaler` PRESERVES NaN in its output. A numeric feature
   carrying NaN therefore puts NaN into the final training matrix, and
   `LogisticRegression` raises `ValueError: Input X contains NaN`.
   Verified directly.

3. `OneHotEncoder(handle_unknown="ignore")` treats NaN as its OWN
   CATEGORY (confirmed: `categories_` contains `nan`) and emits a
   NaN-free 0/1 matrix. A categorical feature carrying NaN therefore
   does NOT crash either estimator — instead it silently converts
   "value was missing" into a predictive feature.

4. `RandomForestClassifier` on sklearn 1.5.2 tolerates NaN natively.
   `LogisticRegression` does not.

5. `train_model()` has no try/except around `pipeline.fit()`, so the
   ValueError in (2) escapes as an unhandled exception.

=====================================================================
THE V1 MISSING-VALUE RULE — EFFECTIVE FEATURE SET GATING
=====================================================================

Points (3) and (4) establish that only columns that actually ENTER the
training matrix can cause crashes or silent data-quality issues.  A
column with missing values that the plan neither imputes/drops NOR
names in any encode/scale step is silently excluded by
`ColumnTransformer(remainder="drop")` and never reaches either
estimator at all.

The rule is therefore keyed to the EFFECTIVE FEATURE SET:

    (a) A column with missing_percentage > 0 that IS named in an
        encode_categorical_features or scale_features step (i.e. it IS
        an effective training feature) MUST be explicitly addressed
        (imputed or dropped).  Failure → severity "material".

    (b) A column with missing_percentage > 0 that is NOT named in any
        such step (i.e. it is NOT an effective training feature) is
        silently excluded by remainder="drop" and cannot crash or
        silently corrupt the model.  Failure to mention it explicitly
        → severity "advisory" (the finding is recorded as evidence; it
        does NOT block execution on its own).

This replaces the prior uniform-material rule.  The new rule is still
estimator-independent (it depends only on which columns reach the
ColumnTransformer, which is determined by the plan's encode/scale
steps, not by which estimator is chosen) and still carries no invented
thresholds (the only threshold used — identifier-like uniqueness — is
imported unchanged from the existing guardrails module).

Rationale for (b):
  - A non-feature column cannot crash LogisticRegression (it never
    reaches it).
  - A non-feature column cannot silently encode missingness as a
    one-hot category (OneHotEncoder never sees it).
  - RandomForest NaN tolerance is irrelevant because the column is not
    in X_train at all.
  - Requiring explicit handling of every missing column regardless of
    feature membership was producing false-positive material findings
    that blocked adequate plans and forced "whack-a-mole" REPLAN
    cycles.

`missing_percentage` is read verbatim from the SAME
`SanitizedLLMContext` the planner itself was shown; it is never
recomputed here, so the evaluator and the LLM cannot disagree about
the evidence.

There are NO invented thresholds. The only threshold used
(identifier-like uniqueness) is imported from the existing guardrails
module rather than redefined.

=====================================================================
MULTI-COLUMN STEP IMPLICATION (classify_plan_steps)
=====================================================================

A step that operates on MULTIPLE columns (encode_categorical_features,
scale_features) is classified as IMPLICATED if ANY of its columns
appear in a material NOT_ADDRESSED finding.  The entire step is
implicated — not just the affected column — because PIPER has no
mechanism to emit a partial tool call.  The LLM must replace the full
step with a corrected version (e.g. impute the problematic column
first, then re-encode the full set).

This is explicitly documented in classify_plan_steps()'s docstring and
covered by tests.
"""

from __future__ import annotations

from app.agent.tools.guardrails import IDENTIFIER_UNIQUENESS_THRESHOLD_PERCENT
from app.agent.tools.sanitized_llm_context import SanitizedLLMContext
from app.schemas.adequacy import AdequacyFinding, PlanAdequacyResult

# Tools whose argument is a SINGLE column name, under the key "column".
_SINGLE_COLUMN_TOOLS = frozenset({"drop_column", "impute_missing_values", "convert_column_type"})
# Tools whose argument is a LIST of column names, under the key "columns".
_MULTI_COLUMN_TOOLS = frozenset({"encode_categorical_features", "scale_features"})
# Tools that, per the verified contract above, put a column into the
# feature set consumed by train_model().
_FEATURE_SELECTING_TOOLS = frozenset({"encode_categorical_features", "scale_features"})

_NUMERIC_DTYPE_PREFIXES = ("int", "uint", "float")


def _is_numeric_dtype_name(dtype: str) -> bool:
    """
    Mirrors the numeric/non-numeric split the existing guardrails use,
    but from the dtype STRING carried on SanitizedColumnContext (this
    module never touches a DataFrame). Matches pandas' dtype naming for
    every numeric dtype PIPER can produce.
    """
    return dtype.lower().startswith(_NUMERIC_DTYPE_PREFIXES)


# Verified against the REAL execution contract in
# app/agent/tools/type_conversion.py::impute_missing_values(), not assumed:
#   mean / median -> numeric columns ONLY. A non-numeric column is rejected
#                    at execution with code "unsupported_dtype_strategy_combination".
#   mode          -> valid for numeric OR categorical.
# Keeping this keyed to the tool's own rule is what makes adequacy and
# execution agree; if that tool's rule ever changes, this must change with it
# (pinned by a test that reproduces the real rejection).
_NUMERIC_ONLY_IMPUTE_STRATEGIES = frozenset({"mean", "median"})


def _imputation_steps(steps: list) -> list[tuple[str, str]]:
    """
    Every (column, strategy) pair proposed by an impute_missing_values
    step. Duck-typed like _columns_touched_by(); malformed steps are
    skipped rather than guessed at (validate_proposed_plan() is what
    rejects them, and it runs before this layer).
    """
    pairs: list[tuple[str, str]] = []
    for step in steps:
        if getattr(step, "tool_name", None) != "impute_missing_values":
            continue
        arguments = getattr(step, "arguments", None)
        if not isinstance(arguments, dict):
            continue
        column = arguments.get("column")
        strategy = arguments.get("strategy")
        if isinstance(column, str) and column and isinstance(strategy, str) and strategy:
            pairs.append((column, strategy))
    return pairs


def _evaluate_imputations(
    proposed_steps: list, context, target_column: str
) -> tuple[set[str], list[tuple[str, str, str]]]:
    """
    Walks the plan IN ORDER and decides, for each impute step, whether its
    strategy can actually run against the dtype that column will have AT
    THAT POINT in the plan.

    Order is load-bearing: clean_node executes steps sequentially, so a
    convert_column_type step earlier in the plan changes what the later
    impute step sees. Judging dtype from the profile alone would reject
    the canonical Telco plan (TotalCharges is `str` at plan time, is
    converted to numeric, and is only then imputed with median — valid at
    execution precisely because the convert ran first).

    Only columns that actually HAVE missing values are evaluated: this
    condition is about whether an impute step RESOLVES missingness, so it
    does not arise for a column with none (the tool rejects such a step
    with "no_missing_values" for an unrelated reason).

    Returns (imputed_columns, incompatible) where `incompatible` is a list
    of (column, strategy, dtype_at_that_point).
    """
    numeric_now: dict[str, bool] = {
        c.name: _is_numeric_dtype_name(c.dtype) for c in context.column_contexts
    }
    dtype_now: dict[str, str] = {c.name: c.dtype for c in context.column_contexts}
    has_missing: dict[str, bool] = {
        c.name: c.missing_percentage > 0.0 for c in context.column_contexts
    }

    imputed: set[str] = set()
    incompatible: list[tuple[str, str, str]] = []

    for step in proposed_steps:
        tool_name = getattr(step, "tool_name", None)
        arguments = getattr(step, "arguments", None)
        if not isinstance(arguments, dict):
            continue

        if tool_name == "convert_column_type":
            column = arguments.get("column")
            target_type = arguments.get("target_type")
            if isinstance(column, str) and column and isinstance(target_type, str):
                # Mirrors convert_column_type()'s real effect on dtype.
                numeric_now[column] = target_type == "numeric"
                dtype_now[column] = target_type
            continue

        if tool_name != "impute_missing_values":
            continue

        column = arguments.get("column")
        strategy = arguments.get("strategy")
        if not (isinstance(column, str) and column and isinstance(strategy, str) and strategy):
            continue

        if column not in numeric_now or not has_missing.get(column, False):
            # Unknown column, or nothing to resolve — left to the tool.
            imputed.add(column)
            continue

        if strategy in _NUMERIC_ONLY_IMPUTE_STRATEGIES and not numeric_now[column]:
            incompatible.append((column, strategy, dtype_now[column]))
        else:
            imputed.add(column)

    return imputed, incompatible


def _columns_touched_by(steps: list, tool_names: frozenset[str]) -> set[str]:
    """
    Every column named by any step whose tool_name is in tool_names.
    Duck-typed on .tool_name/.arguments so this works for both
    ProposedPlanStep (provider layer) and PlanStep (graph layer)
    without importing either — the same layering discipline
    plan_validation.py already follows.
    """
    touched: set[str] = set()
    for step in steps:
        tool_name = getattr(step, "tool_name", None)
        arguments = getattr(step, "arguments", None)
        if tool_name not in tool_names or not isinstance(arguments, dict):
            continue

        if tool_name in _SINGLE_COLUMN_TOOLS:
            value = arguments.get("column")
            if isinstance(value, str) and value:
                touched.add(value)
        elif tool_name in _MULTI_COLUMN_TOOLS:
            values = arguments.get("columns")
            if isinstance(values, list):
                touched.update(v for v in values if isinstance(v, str) and v)
    return touched


def classify_plan_steps(
    findings: list[AdequacyFinding],
    proposed_steps: list,
) -> dict:
    """
    Classifies each proposed step as 'valid' or 'implicated' based on
    the current material adequacy findings.  Pure function — read-only,
    never mutates either argument.

    Returns a dict with two keys:
      "valid_steps"      — list of {"tool_name": ..., "arguments": ...}
                           for steps NOT implicated in any material failure.
                           These should be preserved by the LLM on REPLAN.
      "implicated_steps" — list of {"tool_name": ..., "arguments": ...,
                           "reason": ...} for steps that ARE implicated in
                           at least one material NOT_ADDRESSED finding.

    A step is IMPLICATED if any column it operates on appears in a
    material NOT_ADDRESSED finding's columns list.

    MULTI-COLUMN RULE: For tools that accept a list of columns
    (encode_categorical_features, scale_features), the ENTIRE step is
    implicated if ANY of its columns appear in a material finding.
    PIPER has no mechanism to partially preserve a multi-column
    operation (there is no "scale_features but skip column X" variant),
    so the LLM must replace the whole step when revising.  This
    behavior is explicit and tested — it is not a silent assumption.

    Steps with unrecognised tool names or malformed arguments are
    treated as valid (they were already rejected by
    validate_proposed_plan() in the production path; classify_plan_steps
    only runs on proposals that passed validation).
    """
    # Collect columns that appear in any material NOT_ADDRESSED finding.
    implicated_columns: set[str] = set()
    # Map column → reason text (first material finding that mentions it).
    column_to_reason: dict[str, str] = {}
    for f in findings:
        if f.severity == "material" and f.status == "NOT_ADDRESSED":
            for col in f.columns:
                implicated_columns.add(col)
                if col not in column_to_reason:
                    column_to_reason[col] = f.reason

    valid_steps: list[dict] = []
    implicated_steps: list[dict] = []

    for step in proposed_steps:
        tool_name = getattr(step, "tool_name", None)
        arguments = getattr(step, "arguments", None)
        if not isinstance(arguments, dict):
            arguments = {}

        # Collect columns this step touches.
        step_columns: set[str] = set()
        if tool_name in _SINGLE_COLUMN_TOOLS:
            col = arguments.get("column")
            if isinstance(col, str) and col:
                step_columns.add(col)
        elif tool_name in _MULTI_COLUMN_TOOLS:
            cols = arguments.get("columns")
            if isinstance(cols, list):
                step_columns.update(c for c in cols if isinstance(c, str) and c)

        # A step is implicated if ANY of its columns are implicated.
        overlap = step_columns & implicated_columns
        base = {"tool_name": tool_name, "arguments": arguments}

        if overlap:
            # Collect reasons for all implicated columns in this step.
            reasons = sorted({column_to_reason[c] for c in overlap if c in column_to_reason})
            reason_text = "; ".join(reasons) if reasons else f"Column(s) {sorted(overlap)} implicated in a material adequacy finding."
            implicated_steps.append({**base, "reason": reason_text})
        else:
            valid_steps.append(base)

    return {"valid_steps": valid_steps, "implicated_steps": implicated_steps}


def evaluate_plan_adequacy(
    context: SanitizedLLMContext,
    proposed_steps: list,
    target_column: str,
) -> PlanAdequacyResult:
    """
    Pure function: dataset evidence + proposed plan in, findings out.
    Never mutates either argument. Calling it twice with the same
    inputs always produces an identical result.

    `context` is the SAME SanitizedLLMContext the planner was given, so
    `missing_percentage`/`unique_percentage` are the exact figures the
    LLM saw — this layer can never disagree with the planner about the
    underlying evidence, only about whether the plan answers it.
    """
    findings: list[AdequacyFinding] = []

    dropped = _columns_touched_by(proposed_steps, frozenset({"drop_column"}))
    feature_columns = _columns_touched_by(proposed_steps, _FEATURE_SELECTING_TOOLS)
    all_touched = _columns_touched_by(proposed_steps, _SINGLE_COLUMN_TOOLS | _MULTI_COLUMN_TOOLS)

    # --- Imputation strategy/dtype compatibility ----------------------
    # A column counts as imputed ONLY if the proposed strategy can
    # actually run against that column's dtype. Observed for real
    # (qwen3:8b demonstration run, run_e13cf35f): a plan proposed
    # impute_missing_values(column="Embarked", strategy="median"). The
    # column IS named in an impute step, so adequacy previously marked
    # its missingness ADDRESSED and passed the plan — but at execution
    # the tool correctly rejected it ("Strategy 'median' requires a
    # numeric column; 'Embarked' is not numeric"), so the missing values
    # were never actually resolved.
    #
    # That inconsistency is only harmless when the column stays OUT of
    # the effective feature set (as it did in that run). If such a column
    # IS encoded/scaled, the NaN survives into the training matrix and
    # LogisticRegression raises at fit() time — a real crash path this
    # check now prevents at plan time.
    #
    # This does NOT auto-correct anything: median is never silently
    # rewritten to mode. The incompatible step is reported, with the
    # reason, and the existing REPLAN loop lets the model fix it.
    # ORDER MATTERS. clean_node executes plan steps sequentially, so a
    # convert_column_type step earlier in the plan changes the dtype the
    # later impute step actually sees. The canonical Telco plan relies on
    # exactly this: TotalCharges is `str` at plan time (blank strings),
    # gets converted to numeric, and is THEN imputed with median — which
    # is valid at execution precisely because the convert ran first.
    # Judging dtype from the profile alone would reject that correct plan.
    imputed, incompatible_imputations = _evaluate_imputations(
        proposed_steps, context, target_column
    )

    for column, strategy, dtype in incompatible_imputations:
        findings.append(
            AdequacyFinding(
                condition="imputation_strategy_compatibility",
                columns=[column],
                status="NOT_ADDRESSED",
                # Same effective-feature rule the missing-value condition
                # uses, so the two findings for one column can never
                # disagree about whether it blocks.
                severity="material" if column in feature_columns else "advisory",
                evidence=(
                    f"The plan imputes '{column}' with strategy '{strategy}', but '{column}' "
                    f"has dtype '{dtype}', which is not numeric."
                ),
                reason=(
                    f"impute_missing_values() accepts '{strategy}' for numeric columns only; a "
                    f"non-numeric column is rejected at execution "
                    f"(unsupported_dtype_strategy_combination), so this step would NOT resolve "
                    f"'{column}'s missing values. Use strategy 'mode', which is valid for "
                    f"categorical columns, or drop the column instead."
                ),
            )
        )

    # --- Condition: target protection --------------------------------
    # The target must remain the target. `validate_proposed_plan()`
    # already rejects the target appearing as an encode/scale FEATURE
    # (its structural leakage guard), so that specific case is
    # unreachable in the integrated flow — this check is a superset
    # covering the cases plan validation does NOT catch at plan time
    # (drop/impute/convert of the target, which today would only fail
    # later, inside the tool at execution time). It can never contradict
    # the existing guard: both reject, this one simply rejects earlier
    # and covers more.
    if target_column in all_touched:
        findings.append(
            AdequacyFinding(
                condition="target_protection",
                columns=[target_column],
                status="NOT_ADDRESSED",
                severity="material",
                evidence=f"The plan applies a feature-preprocessing operation to the target column '{target_column}'.",
                reason=(
                    f"'{target_column}' is the prediction target. It must not be dropped, imputed, "
                    f"type-converted, encoded, or scaled by feature preprocessing — doing so would "
                    f"either destroy the label or leak it into the feature matrix."
                ),
            )
        )
    else:
        findings.append(
            AdequacyFinding(
                condition="target_protection",
                columns=[target_column],
                status="ADDRESSED",
                severity="material",
                evidence=f"No plan step applies a feature-preprocessing operation to '{target_column}'.",
                reason="The target column is left untouched by feature preprocessing, as required.",
            )
        )

    # --- Condition: missing values ------------------------------------
    # Severity is keyed to the EFFECTIVE FEATURE SET (see module
    # docstring).  Only columns that actually enter the ColumnTransformer
    # (i.e. named in encode/scale steps) can crash an estimator or
    # silently encode missingness — columns outside the feature set are
    # dropped by remainder="drop" before either estimator ever sees them.
    columns_with_missing = [
        col for col in context.column_contexts
        if col.name != target_column and col.missing_percentage > 0.0
    ]

    if not columns_with_missing:
        findings.append(
            AdequacyFinding(
                condition="missing_values",
                columns=[],
                status="NOT_APPLICABLE",
                severity="material",
                evidence="No non-target column has any missing values.",
                reason="The V1 missing-value invariant cannot be violated when no feature column has missing values.",
            )
        )
    else:
        for col in columns_with_missing:
            is_addressed = col.name in imputed or col.name in dropped
            becomes_feature = col.name in feature_columns

            if is_addressed:
                how = "imputed" if col.name in imputed else "dropped"
                findings.append(
                    AdequacyFinding(
                        condition="missing_values",
                        columns=[col.name],
                        status="ADDRESSED",
                        severity="material",
                        evidence=f"'{col.name}' has {col.missing_percentage}% missing values.",
                        reason=f"The plan {how} '{col.name}', which deterministically resolves its missing values before TRAIN.",
                    )
                )
            elif becomes_feature:
                # Column is IN the effective feature set and has unaddressed
                # missing values → NaN would enter the training matrix.
                # severity = material (blocks execution).
                findings.append(
                    AdequacyFinding(
                        condition="missing_values",
                        columns=[col.name],
                        status="NOT_ADDRESSED",
                        severity="material",
                        evidence=f"'{col.name}' has {col.missing_percentage}% missing values, and no plan step imputes or drops it.",
                        reason=(
                            f"'{col.name}' is used as a training feature (named in an encode or scale "
                            f"step), so its missing values would reach the training matrix. "
                            f"PIPER V1 trains both LogisticRegression and RandomForestClassifier — "
                            f"LogisticRegression raises ValueError on NaN inputs; "
                            f"even for RandomForest, allowing NaN to enter a scaled feature is a "
                            f"silent, unintentional choice rather than a deliberate one."
                        ),
                    )
                )
            else:
                # Column is NOT in the effective feature set — it will be
                # silently dropped by ColumnTransformer(remainder="drop") and
                # never reach either estimator.
                # severity = advisory (does NOT block execution).
                findings.append(
                    AdequacyFinding(
                        condition="missing_values",
                        columns=[col.name],
                        status="NOT_ADDRESSED",
                        severity="advisory",
                        evidence=f"'{col.name}' has {col.missing_percentage}% missing values, and no plan step imputes or drops it.",
                        reason=(
                            f"'{col.name}' is not named by any encode or scale step, so it is silently "
                            f"excluded from the training matrix by train_model()'s "
                            f"ColumnTransformer(remainder='drop'). It cannot crash either estimator or "
                            f"silently encode missingness as a feature. This is advisory: the LLM may "
                            f"choose to explicitly drop or impute it, but is not required to."
                        ),
                    )
                )

    # --- Condition: identifier-like columns (advisory) ----------------
    # Reuses the EXISTING guardrail threshold and the existing
    # convention of excluding numeric columns (a continuous numeric
    # feature being highly unique is normal, not identifier evidence).
    identifier_columns = [
        col for col in context.column_contexts
        if col.name != target_column
        and not _is_numeric_dtype_name(col.dtype)
        and col.unique_percentage > IDENTIFIER_UNIQUENESS_THRESHOLD_PERCENT
    ]

    if not identifier_columns:
        findings.append(
            AdequacyFinding(
                condition="identifier_like_column",
                columns=[],
                status="NOT_APPLICABLE",
                severity="advisory",
                evidence=(
                    f"No non-numeric column exceeds the {IDENTIFIER_UNIQUENESS_THRESHOLD_PERCENT}% "
                    f"uniqueness threshold."
                ),
                reason="No identifier-like column was detected in this dataset.",
            )
        )
    else:
        for col in identifier_columns:
            addressed = col.name in dropped
            findings.append(
                AdequacyFinding(
                    condition="identifier_like_column",
                    columns=[col.name],
                    status="ADDRESSED" if addressed else "NOT_ADDRESSED",
                    severity="advisory",
                    evidence=f"'{col.name}' is {col.unique_percentage}% unique (non-numeric).",
                    reason=(
                        f"The plan drops '{col.name}'."
                        if addressed
                        else (
                            f"'{col.name}' looks identifier-like and the plan does not drop it. This is "
                            f"advisory only: it does not violate a V1 execution invariant, and retaining "
                            f"such a column can be legitimate, so it never blocks execution on its own."
                        )
                    ),
                )
            )

    # --- Condition: empty feature set ------------------------------------
    # If the plan contains no encode_categorical_features or scale_features
    # step, feature_engineer_node will find no completed FE steps and
    # return an immediate failure ("produced an empty feature set"), and
    # train_model() will then fail with "At least one feature column is
    # required."  Both failures were observed in the real end-to-end run
    # (Titanic dataset, attempt 1).
    #
    # This is checked LAST so that a plan that genuinely tried to engineer
    # features but got blocked by other material findings (e.g. a
    # missing-value finding that implicated its encode step) produces the
    # more specific material finding, not this catch-all.  The REPLAN loop
    # will surface all material findings together in its evidence, so the
    # LLM always receives a complete picture regardless of which finding
    # fires.
    #
    # `columns=[]` is correct here: the condition is about the plan as a
    # whole (no column is individually responsible), exactly as
    # NOT_APPLICABLE findings use empty columns lists.
    #
    # READ-ONLY. This check cannot add or suggest a step; it only refuses
    # the plan and lets the existing REPLAN loop ask the LLM to produce a
    # better one.
    has_feature_step = any(
        getattr(s, "tool_name", None) in _FEATURE_SELECTING_TOOLS
        for s in proposed_steps
    )
    if not has_feature_step:
        findings.append(
            AdequacyFinding(
                condition="empty_feature_set",
                columns=[],
                status="NOT_ADDRESSED",
                severity="material",
                evidence=(
                    "The plan contains no encode_categorical_features or scale_features step."
                ),
                reason=(
                    "PIPER's feature_engineer_node only processes encode_categorical_features "
                    "and scale_features steps. A plan with neither produces an empty feature "
                    "matrix: feature_engineer_node raises 'produced an empty feature set' and "
                    "train_model() subsequently raises 'At least one feature column is required'. "
                    "The plan MUST include at least one encode_categorical_features step (for "
                    "categorical columns) or one scale_features step (for numeric columns), "
                    "or both, to be executable."
                ),
            )
        )

    material = [f for f in findings if f.severity == "material" and f.status == "NOT_ADDRESSED"]
    material_failure = len(material) > 0

    if material_failure:
        conditions = sorted({f.condition for f in material})
        affected = sorted({c for f in material for c in f.columns})
        summary = (
            f"Plan is structurally valid but inadequate: {len(material)} material finding(s) "
            f"across {conditions}, affecting {affected}."
        )
    else:
        advisory_open = [f for f in findings if f.severity == "advisory" and f.status == "NOT_ADDRESSED"]
        summary = (
            f"Plan adequacy passed: no material findings"
            f"{f' ({len(advisory_open)} advisory finding(s) recorded)' if advisory_open else ''}."
        )

    return PlanAdequacyResult(
        status="FAIL" if material_failure else "PASS",
        material_failure=material_failure,
        findings=findings,
        summary=summary,
    )
