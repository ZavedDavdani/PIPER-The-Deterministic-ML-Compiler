"""
build_run_explanation(), build_learning_journey(), build_pipeline_visualization(),
and related educational builders (Phase 6: Student Mode & ML Education).

Every function here is a pure, deterministic, read-only transformation over
existing PIPER execution state. Nothing in this module calls an LLM, mutates
AgentState, or generates speculative chain-of-thought.
"""

from __future__ import annotations

from typing import Any, Optional

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
    meta = ACTION_REGISTRY.get(op.tool_name, {})
    level_text = meta.get(level, f"PIPER executed '{op.tool_name}'.")
    return OperationExplanation(
        operation_id=op.operation_id,
        tool_name=op.tool_name,
        what_happened=f"{level_text} {op.result_summary}",
        why=op.reason,
        level=level,
        concept=meta.get("concept"),
        alternative_consideration=meta.get("alternatives"),
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
        concept="Model Selection & Metric Optimization",
    )


def explain_evaluation(
    evaluation: EvaluationResult,
    baseline: Optional[BaselineComparisonResult],
    algorithm: Optional[str] = None,
) -> EvaluationExplanation:
    metrics = []
    for name, value in (
        ("accuracy", evaluation.accuracy),
        ("precision", evaluation.precision),
        ("recall", evaluation.recall),
        ("f1", evaluation.f1),
        ("roc_auc", evaluation.roc_auc),
    ):
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

    cm = evaluation.confusion_matrix
    confusion_matrix_meaning = (
        f"Out of {evaluation.test_rows} test rows: {cm.tp} true positives, "
        f"{cm.tn} true negatives, {cm.fp} false positives, {cm.fn} false negatives."
    )

    baseline_comparison = None
    if baseline is not None and baseline.model_id == evaluation.model_id:
        baseline_comparison = baseline.reason

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
        model_id=evaluation.model_id,
        algorithm=algorithm,
        metrics=metrics,
        confusion_matrix_meaning=confusion_matrix_meaning,
        baseline_comparison=baseline_comparison,
        model_concept=model_concept,
    )


def explain_guardrail_check(check: ValidationCheck) -> GuardrailCheckExplanation:
    meaning = _GUARDRAIL_MEANINGS.get(check.check, "A deterministic PIPER guardrail check.")
    action = None
    if not check.passed:
        if check.check == "data_leakage":
            action = "Drop the leaking feature or re-partition target-derived columns."
        elif check.check == "constant_features":
            action = "Drop constant/empty columns to save memory and simplify the model."
        elif check.check == "high_cardinality":
            action = "Exclude identifier columns from categorical feature encoding."
        elif check.check == "baseline_gate":
            action = "Re-evaluate feature representation or test non-linear algorithms."

    return GuardrailCheckExplanation(
        check=check.check,
        passed=check.passed,
        severity=check.severity,
        meaning=meaning,
        message=check.message,
        educational_action=action,
    )


def explain_failure(failure: FailureInfo) -> FailureExplanation:
    takeaway = None
    if failure.category == "PLAN_ADEQUACY":
        takeaway = "When data has missing values or unhandled columns, the plan must explicitly declare preprocessing steps before training."
    elif failure.category == "LEAKAGE_ERROR":
        takeaway = "Features with near-perfect target correlation must be dropped to prevent model cheating."
    elif failure.category == "DUPLICATE_PLAN":
        takeaway = "Retrying an identical plan produces identical failures; plans must vary hypotheses."

    return FailureExplanation(
        category=failure.category,
        message=failure.message,
        retryable=failure.retryable,
        human_intervention_required=failure.human_intervention_required,
        meaning=_FAILURE_CATEGORY_MEANINGS.get(failure.category, "A structured PIPER failure category."),
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
    replan_count = getattr(state, "replan_count", getattr(state, "retry_count", 0))
    attempts = getattr(state, "planning_attempts", getattr(state, "plan_history", [])) or []
    
    stages_summary = []
    if attempts:
        for idx, p in enumerate(attempts, 1):
            stages_summary.append({
                "attempt": idx,
                "model_candidates": getattr(p, "model_candidates", []),
                "operations_count": len(getattr(p, "cleaning_steps", []) or []) + len(getattr(p, "feature_engineering_steps", []) or []),
            })

    replan_occurred = replan_count > 0 or len(attempts) > 1
    total_attempts = max(len(attempts), replan_count + 1 if replan_count > 0 else 1)

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
    comparison = getattr(state, "comparison", None)
    winner_id = comparison.recommended_model_id if comparison else None
    
    # Check if governance or model results have importance
    features = []
    method = "Model-Derived Feature Importance"
    algo = None

    if hasattr(state, "model_results") and state.model_results:
        for m in state.model_results:
            if m.model_id == winner_id or winner_id is None:
                algo = m.algorithm
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
    status = getattr(state, "status", "unknown")
    cleaning_log = list(getattr(state, "cleaning_log", []) or [])
    feature_log = list(getattr(state, "feature_log", []) or [])
    evaluation_results = list(getattr(state, "evaluation_results", []) or [])
    comparison = getattr(state, "comparison", None)
    validation = getattr(state, "validation", None)
    baseline = getattr(state, "baseline", None)
    failure = getattr(state, "failure", None)

    is_completed = status == "completed"
    is_failed = status == "failed"

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
         f"Evaluated {len(validation.checks) if validation else 0} safety guardrail checks.", "Safety Guardrails"),
        (12, "Select Final Model", "Lock winning model pipeline based on deterministic comparison.",
         f"Winning model selected: {comparison.recommended_model_id if comparison else 'N/A'}.", "Winning Model"),
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
            fail_cat = failure.category if failure else ""
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
    status = getattr(state, "status", "unknown")
    comparison = getattr(state, "comparison", None)
    validation = getattr(state, "validation", None)
    evaluation_results = list(getattr(state, "evaluation_results", []) or [])

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
            status="passed" if getattr(state, "cleaning_log", None) is not None else "pending",
            summary="Missing value imputation and column pruning.",
            details={"operations": len(getattr(state, "cleaning_log", []) or [])},
        ),
        PipelineNode(
            id="split",
            name="Train/Test Split",
            stage="Data Partitioning",
            status="passed" if getattr(state, "split_id", None) else "pending",
            summary="80% Train, 20% Test holdout partition.",
            details={"split_id": getattr(state, "split_id", None)},
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
            status="passed" if (validation and validation.valid) else ("failed" if validation else "pending"),
            summary="Safety, leakage, and baseline benchmarks.",
            details={"checks_passed": sum(1 for c in validation.checks if c.passed) if validation else 0},
        ),
        PipelineNode(
            id="winner",
            name="Winner Selection",
            stage="Selection",
            status="passed" if comparison else "pending",
            summary=f"Selected {comparison.recommended_model_id if comparison else 'N/A'} via F1.",
            details={"winner": comparison.recommended_model_id if comparison else None},
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
    cleaning_log = list(getattr(state, "cleaning_log", []) or [])
    feature_log = list(getattr(state, "feature_log", []) or [])
    comparison = getattr(state, "comparison", None)
    evaluation_results = list(getattr(state, "evaluation_results", []) or [])
    baseline = getattr(state, "baseline", None)
    validation = getattr(state, "validation", None)
    failure = getattr(state, "failure", None)

    # Build model concepts for evaluated models
    model_concepts = []
    seen_algos = set()
    for ev in evaluation_results:
        algo = None
        if hasattr(state, "model_results"):
            for m in state.model_results:
                if m.model_id == ev.model_id:
                    algo = m.algorithm
                    break
        if algo and algo in MODEL_FAMILIES and algo not in seen_algos:
            seen_algos.add(algo)
            info = MODEL_FAMILIES[algo]
            is_win = bool(comparison and comparison.recommended_model_id == ev.model_id)
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
        status=getattr(state, "status", "unknown"),
        level=level,
        preprocessing=[explain_operation(op, level) for op in cleaning_log],
        feature_engineering=[explain_operation(op, level) for op in feature_log],
        model_selection=explain_model_selection(comparison) if comparison is not None else None,
        evaluation=[explain_evaluation(ev, baseline) for ev in evaluation_results],
        guardrail_checks=[explain_guardrail_check(c) for c in validation.checks] if validation is not None else [],
        failure=explain_failure(failure) if failure is not None else None,
        replan=build_replan_explanation(state),
        feature_importance=build_feature_importance_education(state),
        model_concepts=model_concepts,
    )

