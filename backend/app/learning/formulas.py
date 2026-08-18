"""
FORMULA_LIBRARY (Batch 6A: Learn-Explain).

Curated, reviewed, STATIC — every entry is generic and never generated
per-run or per-dataset (locked). app/learning/explain.py plugs a real
run's actual metric values into template sentences separately (see
MetricExplanation) — the library entries themselves never change.
"""

from __future__ import annotations

from app.schemas.learning import FormulaEntry

FORMULA_LIBRARY: list[FormulaEntry] = [
    FormulaEntry(
        name="Accuracy",
        formula="Accuracy = (TP + TN) / (TP + TN + FP + FN)",
        description=(
            "The fraction of all predictions the model got right — both "
            "correctly predicted positives (TP) and correctly predicted "
            "negatives (TN), out of every prediction made."
        ),
        when_used=(
            "A quick overall summary, but misleading on an imbalanced "
            "dataset — a model that always predicts the majority class "
            "can still score a high accuracy while never catching the "
            "minority class at all."
        ),
    ),
    FormulaEntry(
        name="Precision",
        formula="Precision = TP / (TP + FP)",
        description=(
            "Out of every case the model predicted as positive, the "
            "fraction that was actually positive."
        ),
        when_used=(
            "Matters most when a false positive is costly — e.g. "
            "flagging a customer as 'will churn' when they won't wastes "
            "a retention offer on someone who didn't need it."
        ),
    ),
    FormulaEntry(
        name="Recall",
        formula="Recall = TP / (TP + FN)",
        description=(
            "Out of every case that was actually positive, the fraction "
            "the model correctly caught."
        ),
        when_used=(
            "Matters most when a false negative is costly — e.g. "
            "missing a customer who was actually about to churn means "
            "losing them with no intervention at all."
        ),
    ),
    FormulaEntry(
        name="F1 Score",
        formula="F1 = 2 * (Precision * Recall) / (Precision + Recall)",
        description=(
            "The harmonic mean of precision and recall — a single number "
            "that stays low if EITHER precision or recall is low, so a "
            "model can't inflate it by being extreme in only one "
            "direction (e.g. predicting everything positive to maximize "
            "recall alone)."
        ),
        when_used=(
            "PIPER's own locked model-selection metric (see "
            "compare_models()) — chosen specifically because it can't be "
            "gamed by favoring one of precision/recall over the other."
        ),
    ),
    FormulaEntry(
        name="ROC-AUC",
        formula="ROC-AUC = area under the ROC curve (True Positive Rate vs. False Positive Rate, across every possible decision threshold)",
        description=(
            "How well the model separates the two classes overall, "
            "independent of any single decision threshold. 1.0 is "
            "perfect separation; 0.5 is no better than random guessing."
        ),
        when_used=(
            "Reported alongside F1 as secondary context (never as the "
            "selection metric itself, to prevent metric-shopping — see "
            "compare_models())."
        ),
    ),
    FormulaEntry(
        name="Standardization (Z-score scaling)",
        formula="x_scaled = (x - mean) / standard_deviation",
        description=(
            "Rescales a numeric feature so it has mean 0 and standard "
            "deviation 1, computed from the TRAINING split only (never "
            "the test split, to avoid leaking test-set statistics into "
            "training)."
        ),
        when_used=(
            "Applied by scale_features() to numeric columns so no "
            "single feature dominates a model just because its raw "
            "values happen to be on a larger numeric scale."
        ),
    ),
    FormulaEntry(
        name="One-Hot Encoding",
        formula="A categorical column with k distinct values becomes k binary (0/1) columns, one per value.",
        description=(
            "Turns a categorical column (e.g. 'Contract': "
            "Month-to-month/One year/Two year) into a numeric form a "
            "model can actually use, without implying any false "
            "ordering between the categories."
        ),
        when_used=(
            "Applied by encode_categorical_features() to every "
            "remaining non-numeric column in the plan."
        ),
    ),
    FormulaEntry(
        name="Median/Mean Imputation",
        formula="Missing value <- median(column) or mean(column), computed over the non-missing values.",
        description=(
            "Fills in a missing numeric value using a single summary "
            "statistic of the rest of that column, so the row can still "
            "be used for training instead of being dropped."
        ),
        when_used=(
            "Applied by impute_missing_values() — median is generally "
            "preferred over mean when the column has outliers, since "
            "the median isn't pulled toward extreme values."
        ),
    ),
]
