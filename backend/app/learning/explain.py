"""
build_run_explanation(), build_learning_journey(), build_pipeline_visualization(),
and related educational builders (Phase 6: Student Mode & ML Education).

Every function here is a pure, deterministic, read-only transformation over
existing PIPER execution state. Nothing in this module calls an LLM, mutates
AgentState, or generates speculative chain-of-thought.
"""

from __future__ import annotations

from typing import Any, Optional

from app.state_access import field
from app.learning.formulas import FORMULA_LIBRARY
from app.learning.registry import ACTION_REGISTRY, CONCEPTS, METRIC_GUIDANCE, MODEL_FAMILIES
from app.schemas.baseline import BaselineComparisonResult
from app.schemas.evaluation import EvaluationResult, ModelComparison
from app.schemas.failure import FailureInfo
from app.schemas.guardrails import ValidationCheck
from app.schemas.learning import (
    EvaluationExplanation,
    ExplanationLevel,
    FailureExplanation,
    FeatureImportanceEducation,
    GuardrailCheckExplanation,
    LearningJourney,
    LearningJourneyStage,
    MetricExplanation,
    ModelConceptExplanation,
    ModelSelectionExplanation,
    OperationExplanation,
    PipelineEdge,
    PipelineNode,
    PipelineVisualization,
    ReplanExplanation,
    RunExplanation,
    WhyExplanation,
)

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
        "type — PIPER only supports classification — never retried."
    ),
    "EXECUTION_BUDGET_EXCEEDED": (
        "PIPER's hard execution-step safety limit was reached — a "
        "deterministic backstop guaranteeing the run always reaches a clean terminal state."
    ),
    "LEAKAGE_ERROR": (
        "The data-leakage guardrail found a feature suspiciously "
        "predictive of the target — recoverable, since a different plan "
        "could drop the offending feature."
    ),
    "IMBALANCE_ERROR": (
        "The target class distribution is severely imbalanced — "
        "recoverable in principle via balanced metrics or resampled models."
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
        "Evaluating or comparing trained models failed, or the proposed "
        "plan was rejected before execution — recoverable via a different plan."
    ),
    "DUPLICATE_PLAN": (
        "The newly proposed plan was executably identical to a plan "
        "already attempted and failed — retrying the same plan "
        "again could never produce a different outcome, so this is never retried."
    ),
    "BASELINE_GATE_FAILED": (
        "The trained model didn't meaningfully outperform a trivial "
        "majority-class baseline — recoverable via a different plan."
    ),
    "PLAN_ADEQUACY": (
        "The proposed plan was structurally valid but did not address a "
        "condition the dataset requires before training, such as an unhandled missing-value column. "
        "Recoverable via a plan that addresses the required condition."
    ),
}

_GUARDRAIL_MEANINGS: dict[str, str] = {
    "data_leakage": (
        "Checks whether any feature is suspiciously predictive of the "
        "target — a sign the model could be 'cheating' by seeing information "
        "it shouldn't have at prediction time."
    ),
    "target_imbalance": (
        "Checks how skewed the target class distribution is — severe "
        "imbalance can make an uninformative majority-class model look deceptively good."
    ),
    "constant_features": (
        "Checks for columns with only one distinct value or entirely missing — "
        "a constant feature carries zero predictive signal."
    ),
    "high_cardinality": (
        "Checks for categorical columns that are almost entirely unique "
        "values (>99%) — a sign the column is an ID rather than a real feature."
    ),
    "suspicious_evaluation_metric": (
        "Checks whether the evaluation metric is implausibly high — a "
        "common symptom of undetected target leakage."
    ),
    "baseline_gate": (
        "Checks whether the model meaningfully outperforms a trivial "
        "majority-class predictor on the held-out test split."
    ),
}


def explain_operation(op: Any, level: ExplanationLevel = "beginner") -> OperationExplanation:
    tool_name = field(op, "tool_name", default="")
    meta = ACTION_REGISTRY.get(tool_name, {})
    level_text = meta.get(level, f"PIPER executed '{tool_name}'.")
    summary = field(op, "result_summary", default="")
    reason = field(op, "reason", default="")
    op_id = field(op, "operation_id", default="")
    return OperationExplanation(
        operation_id=op_id,
        tool_name=tool_name,
        what_happened=f"{level_text} {summary}".strip(),
        why=reason,
        level=level,
        concept=meta.get("concept"),
        alternative_consideration=meta.get("alternatives"),
    )


def explain_model_selection(comparison: Any) -> ModelSelectionExplanation:
    rec_id = field(comparison, "recommended_model_id")
    models = field(comparison, "models", default=[]) or []
    recommended_algorithm = next(
        (field(m, "algorithm") for m in models if field(m, "model_id") == rec_id),
        "unknown",
    )
    return ModelSelectionExplanation(
        recommended_model_id=rec_id,
        recommended_algorithm=recommended_algorithm,
        justification=field(comparison, "justification", default=""),
        candidates=list(models),
        concept="Model Selection & Metric Optimization",
    )


def explain_evaluation(
    evaluation: Any,
    baseline: Optional[Any] = None,
    algorithm: Optional[str] = None,
) -> EvaluationExplanation:
    metrics = []
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        val = field(evaluation, name, default=0.0)
        value = float(val) if val is not None else 0.0
        guidance = METRIC_GUIDANCE.get(name, {})
        metrics.append(
            MetricExplanation(
                metric=name,
                value=value,
                meaning=f"{guidance.get('name', name)} = {value:.4f} — {guidance.get('measures', '')}",
                formula=guidance.get("formula"),
                guidance=guidance.get("interpretation"),
            )
        )

    cm = field(evaluation, "confusion_matrix")
    tp = field(cm, "tp", default=0) if cm else 0
    tn = field(cm, "tn", default=0) if cm else 0
    fp = field(cm, "fp", default=0) if cm else 0
    fn = field(cm, "fn", default=0) if cm else 0
    test_rows = field(evaluation, "test_rows", default=0)
    confusion_matrix_meaning = (
        f"Out of {test_rows} test rows: {tp} true positives, "
        f"{tn} true negatives, {fp} false positives, {fn} false negatives."
    )

    baseline_comparison = None
    ev_model_id = field(evaluation, "model_id")
    if baseline is not None and field(baseline, "model_id") == ev_model_id:
        baseline_comparison = field(baseline, "reason")

    model_concept = None
    if algorithm and algorithm in MODEL_FAMILIES:
        info = MODEL_FAMILIES[algorithm]
        model_concept = ModelConceptExplanation(
            algorithm=algorithm,
            name=info["name"],
            concept=info["concept"],
            strengths=info["strengths"],
            tradeoffs=info["tradeoffs"],
            how_piper_used_it=info["how_piper_used_it"],
            is_winner=False,
        )

    return EvaluationExplanation(
        model_id=ev_model_id,
        algorithm=algorithm,
        metrics=metrics,
        confusion_matrix_meaning=confusion_matrix_meaning,
        baseline_comparison=baseline_comparison,
        model_concept=model_concept,
    )


def explain_guardrail_check(check: Any) -> GuardrailCheckExplanation:
    chk_name = field(check, "check", default="")
    chk_passed = bool(field(check, "passed", default=True))
    chk_severity = field(check, "severity", default="warning")
    chk_message = field(check, "message", default="")
    meaning = _GUARDRAIL_MEANINGS.get(chk_name, "A deterministic PIPER guardrail check.")
    action = None
    if not chk_passed:
        if chk_name == "data_leakage":
            action = "Drop the leaking feature or re-partition target-derived columns."
        elif chk_name == "constant_features":
            action = "Drop constant/empty columns to save memory and simplify the model."
        elif chk_name == "high_cardinality":
            action = "Exclude identifier columns from categorical feature encoding."
        elif chk_name == "baseline_gate":
            action = "Re-evaluate feature representation or test non-linear algorithms."

    return GuardrailCheckExplanation(
        check=chk_name,
        passed=chk_passed,
        severity=chk_severity,
        meaning=meaning,
        message=chk_message,
        educational_action=action,
    )


def explain_failure(failure: Any) -> FailureExplanation:
    cat = field(failure, "category", default="")
    msg = field(failure, "message", default="")
    retryable = bool(field(failure, "retryable", default=False))
    hi = bool(field(failure, "human_intervention_required", default=False))
    takeaway = None
    if cat == "PLAN_ADEQUACY":
        takeaway = "When data has missing values or unhandled columns, the plan must explicitly declare preprocessing steps before training."
    elif cat == "LEAKAGE_ERROR":
        takeaway = "Features with near-perfect target correlation must be dropped to prevent model cheating."
    elif cat == "DUPLICATE_PLAN":
        takeaway = "Retrying an identical plan produces identical failures; plans must vary hypotheses."

    return FailureExplanation(
        category=cat,
        message=msg,
        retryable=retryable,
        human_intervention_required=hi,
        meaning=_FAILURE_CATEGORY_MEANINGS.get(cat, "A structured PIPER failure category."),
        educational_takeaway=takeaway,
    )


def build_why_explanation(
    action: str,
    evidence: Optional[dict[str, Any]] = None,
    level: ExplanationLevel = "beginner",
) -> WhyExplanation:
    """Provides a deterministic 'Why did PIPER do this?' explanation for an action or check."""
    ev = evidence or {}
    meta = ACTION_REGISTRY.get(action, {})
    title = meta.get("title", action)
    concept = meta.get("concept", "Machine Learning Engineering")
    what_happened = meta.get(level, f"PIPER executed {title}.")
    why = meta.get("why_it_matters", "Required for model training integrity and correctness.")
    alternatives = meta.get("alternatives")

    if action == "impute_missing_values" and "column" in ev:
        col = ev["column"]
        strategy = ev.get("strategy", "median")
        what_happened = f"Missing values in column '{col}' were filled using the {strategy}."
        why = f"Column '{col}' is numeric and contained nulls. Estimators fail on null inputs without imputation."

    elif action == "drop_column" and "column" in ev:
        col = ev["column"]
        reason = ev.get("reason", "identified as non-predictive or redundant")
        what_happened = f"Column '{col}' was dropped from the dataset."
        why = f"Column '{col}' was {reason}, removing noise and preventing leakage."

    elif action == "compare_models" and "winner_id" in ev:
        winner = ev["winner_id"]
        f1_score = ev.get("f1_score", "highest")
        what_happened = f"Model '{winner}' was chosen as the winning model."
        why = f"Model '{winner}' achieved the highest test F1 score ({f1_score}), demonstrating superior balance between precision and recall."

    return WhyExplanation(
        action=action,
        what_happened=what_happened,
        why=why,
        concept=concept,
        alternative_consideration=alternatives,
        level=level,
        evidence=ev,
    )


def build_replan_explanation(state: Any, events: Optional[list[Any]] = None) -> ReplanExplanation:
    """Builds an educational summary of REPLAN cycles from run state or events."""
    replan_count = field(state, "replan_count", default=field(state, "retry_count", default=0))
    attempts = field(state, "planning_attempts", default=field(state, "plan_history", default=[])) or []
    
    stages_summary = []
    if attempts:
        for idx, p in enumerate(attempts, 1):
            stages_summary.append({
                "attempt": idx,
                "model_candidates": field(p, "model_candidates", default=[]),
                "operations_count": len(field(p, "cleaning_steps", default=[]) or []) + len(field(p, "feature_engineering_steps", default=[]) or []),
            })

    replan_occurred = (replan_count or 0) > 0 or len(attempts) > 1
    total_attempts = max(len(attempts), (replan_count or 0) + 1 if (replan_count or 0) > 0 else 1)

    diffs = []
    if len(attempts) >= 2:
        diffs.append({
            "comparison": "Attempt 1 -> Attempt 2",
            "summary": "Plan modified to resolve validation/adequacy findings from Attempt 1.",
        })

    takeaway = (
        "When an ML plan encounters validation or adequacy errors, autonomous REPLAN preserves valid context "
        "and requests a targeted repair instead of crashing or repeating the exact same mistake."
    )

    return ReplanExplanation(
        replan_occurred=replan_occurred,
        total_attempts=total_attempts,
        attempts_summary=stages_summary,
        plan_differences=diffs,
        educational_takeaway=takeaway,
    )


def build_feature_importance_education(state: Any) -> FeatureImportanceEducation:
    """Builds educational feature importance breakdown with explicit non-causal disclaimer."""
    comparison = field(state, "comparison")
    winner_id = field(comparison, "recommended_model_id")
    model_results = list(field(state, "model_results", default=[]) or [])
    
    features = []
    method = "Model-Derived Feature Importance"
    algo = None

    if model_results:
        for m in model_results:
            if field(m, "model_id") == winner_id or winner_id is None:
                algo = field(m, "algorithm")
                break

    disclaimer = "Feature importance shows statistical association with model predictions; it does not prove causation."
    summary = (
        "Features with higher importance values had greater mathematical influence on the final model's "
        "decision boundaries on the holdout test set."
    )

    return FeatureImportanceEducation(
        available=bool(features) or bool(algo),
        method=method,
        algorithm=algo,
        disclaimer=disclaimer,
        features=features,
        educational_summary=summary,
    )


def build_learning_journey(run_id: str, state: Any) -> LearningJourney:
    """Derives the 14-stage guided ML learning journey deterministically from run state."""
    status = field(state, "status", default="unknown")
    cleaning_log = list(field(state, "cleaning_log", default=[]) or [])
    feature_log = list(field(state, "feature_log", default=[]) or [])
    evaluation_results = list(field(state, "evaluation_results", default=[]) or [])
    comparison = field(state, "comparison")
    validation = field(state, "validation")
    baseline = field(state, "baseline")
    failure = field(state, "failure")

    is_completed = status == "completed"
    is_failed = status == "failed"
    checks = field(validation, "checks", default=[]) or []
    rec_id = field(comparison, "recommended_model_id") or "N/A"

    # Stage definitions mapping directly to state
    stages_spec = [
        (1, "Understand the Dataset", "Load the raw dataset, examine row/column counts, and detect column types.",
         "Dataset profiling complete. Columns and preliminary data types identified.", "Data Profiling"),
        (2, "Identify the Target", "Define the outcome variable and verify classification target cardinality.",
         "Classification target identified and class distributions mapped.", "Target Specification"),
        (3, "Explore Data Quality", "Inspect data for missingness, zero-variance columns, and schema anomalies.",
         "Data quality checks scanned for nulls, cardinality issues, and constant fields.", "Data Quality Exploration"),
        (4, "Handle Missing Values", "Impute or drop missing data points to ensure estimator mathematical validity.",
         f"Processed {len(cleaning_log)} preprocessing operations to handle nulls.", "Missing Value Imputation"),
        (5, "Select & Transform Features", "One-hot encode categorical features and standardize numeric variables.",
         f"Configured feature transformations across {len(feature_log)} operation(s).", "Feature Engineering"),
        (6, "Split the Data", "Partition dataset into isolated training (80%) and testing (20%) splits.",
         "Independent train/test split created with fixed random seed.", "Train / Test Split"),
        (7, "Train Models", "Fit candidate classification algorithms on the training data.",
         f"Trained {len(evaluation_results)} candidate model pipeline(s).", "Model Training"),
        (8, "Evaluate Models", "Measure Accuracy, Precision, Recall, F1, and ROC-AUC on holdout test data.",
         "Holdout evaluation completed across all candidate models.", "Model Evaluation"),
        (9, "Compare Models", "Compare candidate models side-by-side using F1 score.",
         "Model comparison ranked models by balanced F1 performance.", "Model Selection"),
        (10, "Check Baseline", "Verify model meaningfully outperforms a trivial majority-class predictor.",
         "Baseline gate evaluated model against zero-intelligence benchmark.", "Baseline Gate"),
        (11, "Run Guardrails", "Execute safety checks for data leakage, high cardinality, and metric plausibility.",
         f"Evaluated {len(checks)} safety guardrail checks.", "Safety Guardrails"),
        (12, "Select Final Model", "Lock winning model pipeline based on deterministic comparison.",
         f"Winning model selected: {rec_id}.", "Winning Model"),
        (13, "Understand Artifact", "Verify serializable pipeline bundle and cryptographic SHA-256 hashes.",
         "Deployment bundle verified with inference parity checks.", "Artifact Verification"),
        (14, "Test Unseen Data", "Score new unseen customer data through Test Flight /predict without retraining.",
         "Standalone inference interface ready for batch CSV or JSON scoring.", "Test Flight"),
    ]

    stages = []
    current_stage_id = None

    for stage_id, title, desc, default_summary, concept in stages_spec:
        # Determine status
        st_status: str = "not_reached"
        details: dict[str, Any] = {}

        if is_completed:
            st_status = "completed"
        elif is_failed:
            fail_cat = field(failure, "category", default="") if failure else ""
            if stage_id <= 3:
                st_status = "completed"
            elif stage_id == 4 and fail_cat in ("SCHEMA_ERROR", "DATA_ERROR", "PLAN_ADEQUACY"):
                st_status = "failed"
                current_stage_id = stage_id
            elif stage_id == 7 and fail_cat == "TRAINING_ERROR":
                st_status = "failed"
                current_stage_id = stage_id
            elif stage_id in (8, 9) and fail_cat == "EVALUATION_ERROR":
                st_status = "failed"
                current_stage_id = stage_id
            elif stage_id == 10 and fail_cat == "BASELINE_GATE_FAILED":
                st_status = "failed"
                current_stage_id = stage_id
            elif stage_id == 11 and fail_cat in ("LEAKAGE_ERROR", "IMBALANCE_ERROR"):
                st_status = "failed"
                current_stage_id = stage_id
            elif current_stage_id is None and stage_id > 3:
                st_status = "not_reached"
        else:
            st_status = "in_progress" if stage_id == 1 else "not_reached"

        stages.append(
            LearningJourneyStage(
                stage_id=stage_id,
                title=title,
                description=desc,
                status=st_status,  # type: ignore
                summary=default_summary,
                details=details,
                concept=concept,
            )
        )

    return LearningJourney(
        run_id=run_id,
        status=status,
        current_stage_id=current_stage_id,
        stages=stages,
    )


def build_pipeline_visualization(run_id: str, state: Any) -> PipelineVisualization:
    """Builds interactive pipeline flowchart data for student inspection."""
    status = field(state, "status", default="unknown")
    comparison = field(state, "comparison")
    validation = field(state, "validation")
    evaluation_results = list(field(state, "evaluation_results", default=[]) or [])
    cleaning_log = list(field(state, "cleaning_log", default=[]) or [])
    split_id = field(state, "split_id")
    checks = field(validation, "checks", default=[]) or []
    val_valid = field(validation, "valid")
    rec_id = field(comparison, "recommended_model_id")

    nodes = [
        PipelineNode(
            id="dataset",
            name="Dataset",
            stage="Input",
            status="passed",
            summary="Raw dataset profile and schema definition.",
            details={"run_id": run_id},
        ),
        PipelineNode(
            id="preprocessing",
            name="Preprocessing",
            stage="Data Preparation",
            status="passed" if field(state, "cleaning_log") is not None else "pending",
            summary="Missing value imputation and column pruning.",
            details={"operations": len(cleaning_log)},
        ),
        PipelineNode(
            id="split",
            name="Train/Test Split",
            stage="Data Partitioning",
            status="passed" if split_id else "pending",
            summary="80% Train, 20% Test holdout partition.",
            details={"split_id": split_id},
        ),
        PipelineNode(
            id="models",
            name="Candidate Models",
            stage="Training",
            status="passed" if evaluation_results else "pending",
            summary=f"Trained {len(evaluation_results)} candidate pipeline(s).",
            details={"candidates": len(evaluation_results)},
        ),
        PipelineNode(
            id="evaluation",
            name="Evaluation",
            stage="Scoring",
            status="passed" if evaluation_results else "pending",
            summary="Holdout test evaluation metrics computed.",
            details={"metrics_computed": ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        ),
        PipelineNode(
            id="guardrails",
            name="Guardrails",
            stage="Validation",
            status="passed" if (validation and val_valid) else ("failed" if validation else "pending"),
            summary="Safety, leakage, and baseline benchmarks.",
            details={"checks_passed": sum(1 for c in checks if field(c, "passed")) if validation else 0},
        ),
        PipelineNode(
            id="winner",
            name="Winner Selection",
            stage="Selection",
            status="passed" if comparison else "pending",
            summary=f"Selected {rec_id or 'N/A'} via F1.",
            details={"winner": rec_id},
        ),
        PipelineNode(
            id="artifact",
            name="Deployment Artifact",
            stage="Publish",
            status="passed" if status == "completed" else "pending",
            summary="Standalone pipeline.joblib and deployment package.",
            details={"verified": status == "completed"},
        ),
    ]

    edges = [
        PipelineEdge(from_node="dataset", to_node="preprocessing"),
        PipelineEdge(from_node="preprocessing", to_node="split"),
        PipelineEdge(from_node="split", to_node="models"),
        PipelineEdge(from_node="models", to_node="evaluation"),
        PipelineEdge(from_node="evaluation", to_node="guardrails"),
        PipelineEdge(from_node="guardrails", to_node="winner"),
        PipelineEdge(from_node="winner", to_node="artifact"),
    ]

    return PipelineVisualization(run_id=run_id, nodes=nodes, edges=edges)


def build_run_explanation(run_id: str, state: Any, level: ExplanationLevel = "beginner") -> RunExplanation:
    """Aggregates level-aware explanations for an entire run."""
    cleaning_log = list(field(state, "cleaning_log", default=[]) or [])
    feature_log = list(field(state, "feature_log", default=[]) or [])
    comparison = field(state, "comparison")
    evaluation_results = list(field(state, "evaluation_results", default=[]) or [])
    baseline = field(state, "baseline")
    validation = field(state, "validation")
    failure = field(state, "failure")
    model_results = list(field(state, "model_results", default=[]) or [])
    checks = field(validation, "checks", default=[]) or []
    rec_id = field(comparison, "recommended_model_id")

    # Build model concepts for evaluated models
    model_concepts = []
    seen_algos = set()
    for ev in evaluation_results:
        algo = None
        ev_id = field(ev, "model_id")
        for m in model_results:
            if field(m, "model_id") == ev_id:
                algo = field(m, "algorithm")
                break
        if algo and algo in MODEL_FAMILIES and algo not in seen_algos:
            seen_algos.add(algo)
            info = MODEL_FAMILIES[algo]
            is_win = bool(comparison and rec_id == ev_id)
            model_concepts.append(
                ModelConceptExplanation(
                    algorithm=algo,
                    name=info["name"],
                    concept=info["concept"],
                    strengths=info["strengths"],
                    tradeoffs=info["tradeoffs"],
                    how_piper_used_it=info["how_piper_used_it"],
                    is_winner=is_win,
                )
            )

    return RunExplanation(
        run_id=run_id,
        status=field(state, "status", default="unknown"),
        level=level,
        preprocessing=[explain_operation(op, level) for op in cleaning_log],
        feature_engineering=[explain_operation(op, level) for op in feature_log],
        model_selection=explain_model_selection(comparison) if comparison is not None else None,
        evaluation=[explain_evaluation(ev, baseline) for ev in evaluation_results],
        guardrail_checks=[explain_guardrail_check(c) for c in checks],
        failure=explain_failure(failure) if failure is not None else None,
        replan=build_replan_explanation(state),
        feature_importance=build_feature_importance_education(state),
        model_concepts=model_concepts,
    )

