"""
Static Explanation Registry & Dictionary (Phase 6: Student Mode).

Reviewed, deterministic, static knowledge base for ML education.
Never LLM-generated, never fabricated.
"""

from __future__ import annotations

from typing import Any

from app.schemas.learning import ConceptDefinition

# Static Concept Dictionary
CONCEPTS: list[ConceptDefinition] = [
    ConceptDefinition(
        key="missing_value_imputation",
        title="Missing Value Imputation",
        category="Preprocessing",
        summary="Replacing missing data points with statistical estimates rather than discarding rows.",
        detail=(
            "Missing data is common in real-world tabular datasets. Dropping entire rows with missing "
            "values reduces training sample size and can introduce survivorship bias. Imputation estimates "
            "the missing values using central tendency statistics: median for skewed numeric data with "
            "outliers, mean for symmetric numeric data, or most frequent/constant for categorical fields."
        ),
        related_formula="Median/Mean Imputation",
    ),
    ConceptDefinition(
        key="column_dropping",
        title="Feature Dropping",
        category="Preprocessing",
        summary="Removing irrelevant, high-missingness, or leaking columns from the dataset.",
        detail=(
            "Columns that contain purely unique identifiers, excessive missing values (>90%), or "
            "direct target leakage add noise or invalidate generalization. Dropping these columns "
            "protects model generalization and simplifies training."
        ),
    ),
    ConceptDefinition(
        key="categorical_encoding",
        title="One-Hot Encoding",
        category="Feature Engineering",
        summary="Transforming discrete categories into binary (0/1) indicator columns.",
        detail=(
            "Machine learning algorithms require numeric matrix inputs. Categorical values (e.g., "
            "'Contract': 'Month-to-month', 'One year') cannot be directly multiplied by weights. "
            "One-hot encoding creates separate binary columns without imposing artificial ordinal rankings."
        ),
        related_formula="One-Hot Encoding",
    ),
    ConceptDefinition(
        key="feature_scaling",
        title="Feature Standardization (Z-score)",
        category="Feature Engineering",
        summary="Rescaling numeric features to have zero mean and unit standard deviation.",
        detail=(
            "When features are on radically different numerical scales (e.g. Age: 20-80 vs TotalCharges: "
            "100-8000), gradient descent and regularized linear models can be dominated by high-magnitude "
            "features. Standardization levels the optimization playing field. Scaling parameters must "
            "be fit strictly on the training split to prevent leakage."
        ),
        related_formula="Standardization (Z-score scaling)",
    ),
    ConceptDefinition(
        key="data_leakage",
        title="Data Leakage Prevention",
        category="Guardrails & Integrity",
        summary="Ensuring the training process never accesses information unavailable at inference time.",
        detail=(
            "Data leakage occurs when test data or future information contaminates training. Examples "
            "include calculating global mean imputation before splitting, or including a feature that "
            "is mathematically synonymous with the target."
        ),
    ),
    ConceptDefinition(
        key="target_imbalance",
        title="Class Imbalance",
        category="Guardrails & Integrity",
        summary="Handling disparities between majority and minority class frequencies.",
        detail=(
            "When one class represents 95% of data and the other 5%, raw accuracy is misleading "
            "because an uninformative model that always predicts the majority class gets 95% accuracy. "
            "Balanced metrics (F1 score, Precision, Recall) and baseline comparisons are required."
        ),
    ),
    ConceptDefinition(
        key="train_test_split",
        title="Train / Test Split",
        category="Validation",
        summary="Partitioning data into independent sets to evaluate generalization on unseen data.",
        detail=(
            "Evaluating a model on the data it trained on measures memorization, not generalization. "
            "Holding out an untouched test split provides an honest evaluation of performance on future unseen data."
        ),
    ),
    ConceptDefinition(
        key="model_selection",
        title="Model Selection via F1",
        category="Evaluation",
        summary="Selecting the optimal candidate model using a balanced performance objective.",
        detail=(
            "PIPER compares candidate models on the held-out test split using the F1 score. "
            "F1 is the harmonic mean of Precision and Recall, preventing models from winning "
            "by simply maximizing one extreme."
        ),
        related_formula="F1 Score",
    ),
    ConceptDefinition(
        key="baseline_comparison",
        title="Baseline Comparison",
        category="Evaluation",
        summary="Comparing model performance against a trivial majority-class predictor.",
        detail=(
            "A model is only useful if it learns real patterns beyond a dummy majority-class rule. "
            "PIPER computes a zero-intelligence majority class baseline on the exact same test split "
            "and requires trained models to meaningfully exceed it."
        ),
    ),
    ConceptDefinition(
        key="replan_cycle",
        title="REPLAN & Self-Correction",
        category="Orchestration",
        summary="Autonomous feedback loop that detects structural flaws and attempts revised strategies.",
        detail=(
            "When a proposed plan fails validation or adequacy (e.g. missing values unhandled), "
            "PIPER's deterministic state machine catches the failure, diagnoses the exact defect, "
            "and triggers a structured REPLAN attempt within a strict execution budget."
        ),
    ),
    ConceptDefinition(
        key="feature_importance",
        title="Feature Importance & Non-Causality",
        category="Governance",
        summary="Measuring feature contribution to model predictions without claiming real-world causation.",
        detail=(
            "Feature importance quantifies how strongly a trained model relies on each feature to make "
            "its predictions. High importance indicates strong statistical association, NOT that manipulating "
            "the feature will cause an outcome in the real world."
        ),
    ),
]

# Static Action Dictionary
ACTION_REGISTRY: dict[str, dict[str, Any]] = {
    "impute_missing_values": {
        "title": "Impute Missing Values",
        "concept": "Missing Value Imputation",
        "why_it_matters": (
            "Standard machine learning estimators (like LogisticRegression and RandomForest) cannot "
            "compute gradients or split points when tabular cells contain NaN / missing values."
        ),
        "alternatives": (
            "Alternative: Dropping rows with missing values drops training signal. "
            "Alternative: Mean imputation can be distorted by outliers; median imputation is more robust."
        ),
        "beginner": "PIPER replaced missing values in this column so the model can read every row.",
        "intermediate": "Replaced null entries using column-level statistics (median/mean/mode) to preserve dataset size.",
        "advanced": "Applied SimpleImputer transformer integrated into the preprocessing column pipeline.",
    },
    "drop_column": {
        "title": "Drop Column",
        "concept": "Feature Dropping",
        "why_it_matters": (
            "Columns that are pure identifiers, have excessive missingness, or contain leakage "
            "impair model performance or invalidate evaluation."
        ),
        "alternatives": (
            "Alternative: Keeping identifier columns leads to overfitting on customer IDs. "
            "Alternative: Imputing >90% missing columns introduces synthetic noise."
        ),
        "beginner": "PIPER removed this column because it was not helpful for predicting the target.",
        "intermediate": "Dropped column due to high missingness, unique ID cardinality, or user exclusion.",
        "advanced": "Column removed from dataset frame prior to train/test partitioning.",
    },
    "convert_column_type": {
        "title": "Convert Column Type",
        "concept": "Data Typing",
        "why_it_matters": (
            "Numbers formatted as strings (e.g., '12.50') cannot be mathematically scaled or averaged "
            "until parsed to numeric floats."
        ),
        "alternatives": (
            "Alternative: Treating numeric strings as categorical creates hundreds of single-value categories."
        ),
        "beginner": "PIPER fixed this column's format so numbers are recognized as numbers.",
        "intermediate": "Coerced string representations into numeric floats or categorical types.",
        "advanced": "Applied pd.to_numeric coercion with errors='coerce' to standardize feature dtypes.",
    },
    "encode_categorical_features": {
        "title": "Encode Categorical Features",
        "concept": "One-Hot Encoding",
        "why_it_matters": (
            "Linear models and decision trees require numeric inputs to compute weights and decision thresholds."
        ),
        "alternatives": (
            "Alternative: Label encoding (0, 1, 2) introduces a fake numerical ordering where 'Two year' > 'One year'."
        ),
        "beginner": "PIPER turned text categories into 1s and 0s that the computer can calculate.",
        "intermediate": "Expanded categorical columns into binary indicator variables via OneHotEncoder(handle_unknown='ignore').",
        "advanced": "Appended OneHotEncoder step to sklearn ColumnTransformer, preserving unobserved category tolerance.",
    },
    "scale_features": {
        "title": "Standardize Numeric Features",
        "concept": "Feature Scaling",
        "why_it_matters": (
            "Features with large ranges (e.g. TotalCharges: $8,000) would dominate regularized models over small features (e.g. Tenure: 12)."
        ),
        "alternatives": (
            "Alternative: Min-Max scaling bounds to [0,1] but is sensitive to outliers; StandardScaler (Z-score) is standard."
        ),
        "beginner": "PIPER adjusted the scale of numeric columns so no single large number overpowers the others.",
        "intermediate": "Standardized numeric columns to zero mean and unit variance using training statistics.",
        "advanced": "Added StandardScaler to numeric branch of ColumnTransformer, fit strictly on train split.",
    },
    "train_model": {
        "title": "Train Model",
        "concept": "Supervised Learning",
        "why_it_matters": "Fitting the mathematical estimator on the training data to learn patterns mapping features to target labels.",
        "alternatives": "Different model families (linear vs. tree-based) have different inductive biases and complexity.",
        "beginner": "PIPER taught the model to recognize patterns between customer information and the target outcome.",
        "intermediate": "Fit estimator parameters (weights or decision trees) on the training partition.",
        "advanced": "Executed pipeline.fit(X_train, y_train) inside an isolated execution sandbox.",
    },
    "evaluate_model": {
        "title": "Evaluate Model",
        "concept": "Model Evaluation",
        "why_it_matters": "Scoring the trained model on holdout test data to measure true generalization ability.",
        "alternatives": "Evaluating on train data produces over-optimistic scores due to memorization.",
        "beginner": "PIPER tested the model on separate test data it had never seen before to see how well it works.",
        "intermediate": "Computed confusion matrix, Accuracy, Precision, Recall, F1, and ROC-AUC on holdout test split.",
        "advanced": "Executed pipeline.predict(X_test) and generated non-refitting metrics against y_test.",
    },
    "compare_models": {
        "title": "Compare Models",
        "concept": "Model Selection",
        "why_it_matters": "Determining which algorithm performs best on the primary objective (F1 score).",
        "alternatives": "Selecting by Accuracy alone could favor models that ignore the minority class.",
        "beginner": "PIPER compared all tested models side-by-side and picked the one with the best balanced score.",
        "intermediate": "Ranked candidate models by test-split F1 score and selected the top performer.",
        "advanced": "Executed deterministic model comparison with automatic runner-up delta calculation.",
    },
    "guardrails": {
        "title": "Run Guardrails",
        "concept": "Data & Model Guardrails",
        "why_it_matters": "Pre-flight and post-flight safety checks preventing data leakage, severe imbalance, or uninformative baseline models.",
        "alternatives": "Skipping guardrails risks deploying leaking or broken models to production.",
        "beginner": "PIPER checked for safety issues like cheating data or empty columns before trusting the result.",
        "intermediate": "Evaluated data leakage, target distribution, constant columns, and baseline gate rules.",
        "advanced": "Executed validation checks returning structured severity and actionable repair diagnostics.",
    },
    "replan": {
        "title": "REPLAN Triggered",
        "concept": "REPLAN & Self-Correction",
        "why_it_matters": "Allowing the system to recover when an initial plan is invalid or fails safety checks.",
        "alternatives": "Failing immediately on first error halts automation without self-correction.",
        "beginner": "PIPER noticed a problem in the first plan, fixed it, and tried a better approach.",
        "intermediate": "Detected validation/adequacy failure, preserved valid context, and requested a corrected plan attempt.",
        "advanced": "State machine transition from PLAN_FAILED / VALIDATE_FAILED to REPLAN within attempt budget.",
    },
}

# Static Model Family Explanations
MODEL_FAMILIES: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "concept": (
            "A fundamental linear classification algorithm that models the log-odds of the positive class "
            "as a weighted linear combination of input features, mapped through a sigmoid function to output probabilities between 0 and 1."
        ),
        "strengths": [
            "Highly interpretable with direct feature weights (coefficients)",
            "Fast to train and very lightweight for inference",
            "Outputs well-calibrated class probabilities",
            "Resistant to overfitting on small datasets when regularized",
        ],
        "tradeoffs": [
            "Assumes linear decision boundaries between classes",
            "Cannot capture complex non-linear interactions without manual feature crosses",
            "Sensitive to outliers and unscaled numeric features",
        ],
        "how_piper_used_it": (
            "Trained with L2 regularization and standard scaling on all numeric features."
        ),
    },
    "random_forest": {
        "name": "Random Forest Classifier",
        "concept": (
            "An ensemble learning method that constructs a multitude of decision trees at training time "
            "and outputs the majority class vote across all individual trees (bagging + feature randomization)."
        ),
        "strengths": [
            "Naturally handles non-linear relationships and complex feature interactions",
            "Robust to outliers and invariant to monotonic feature scaling",
            "Reduces variance and overfitting compared to individual decision trees",
            "Provides built-in feature importance rankings",
        ],
        "tradeoffs": [
            "Larger artifact file size and higher memory footprint than linear models",
            "Slower inference time compared to a single linear equation",
            "Less directly interpretable than simple linear coefficients",
        ],
        "how_piper_used_it": (
            "Trained with an ensemble of decision trees with balanced bootstrap sampling and fixed random seed."
        ),
    },
}

# Static Metric Guidance
METRIC_GUIDANCE: dict[str, dict[str, str]] = {
    "accuracy": {
        "name": "Accuracy",
        "measures": "The proportion of total predictions (both positive and negative) that were correct.",
        "when_useful": "Best when the dataset is well-balanced across classes and all error types have equal cost.",
        "interpretation": "High accuracy (>0.90) is good, but can be deceptive if 95% of the data belongs to one class.",
        "formula": "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
    },
    "precision": {
        "name": "Precision",
        "measures": "Out of all instances predicted as positive, how many were actually positive.",
        "when_useful": "Critical when False Positives are expensive (e.g. spam detection, fraud flags).",
        "interpretation": "High precision means when the model says 'Yes', you can trust it.",
        "formula": "Precision = TP / (TP + FP)",
    },
    "recall": {
        "name": "Recall (Sensitivity)",
        "measures": "Out of all actual positive instances, how many did the model successfully identify.",
        "when_useful": "Critical when False Negatives are dangerous or costly (e.g. disease diagnosis, customer churn).",
        "interpretation": "High recall means the model rarely misses actual positive cases.",
        "formula": "Recall = TP / (TP + FN)",
    },
    "f1": {
        "name": "F1 Score",
        "measures": "The harmonic mean of Precision and Recall, balancing both false positives and false negatives.",
        "when_useful": "PIPER's primary selection metric; essential for imbalanced datasets.",
        "interpretation": "A high F1 score indicates both strong precision and strong recall simultaneously.",
        "formula": "F1 = 2 * (Precision * Recall) / (Precision + Recall)",
    },
    "roc_auc": {
        "name": "ROC-AUC",
        "measures": "The area under the Receiver Operating Characteristic curve across all classification thresholds.",
        "when_useful": "Assesses model discrimination capability independently of any chosen probability threshold.",
        "interpretation": "1.0 = perfect ranking; 0.5 = random guessing; <0.5 = worse than random.",
        "formula": "Area Under (True Positive Rate vs. False Positive Rate Curve)",
    },
    "baseline_accuracy": {
        "name": "Majority-Class Baseline",
        "measures": "The accuracy achieved by a trivial strategy that always predicts the most frequent label.",
        "when_useful": "The minimum benchmark any trained machine learning model must beat to demonstrate value.",
        "interpretation": "A trained model must meaningfully exceed this baseline to pass PIPER's baseline gate.",
        "formula": "Majority Class Count / Total Test Rows",
    },
}
