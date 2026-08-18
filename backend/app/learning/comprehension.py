"""
COMPREHENSION_CHECKS (Batch 6A: Learn-Explain).

Static "check your understanding" content — curated once, reviewed,
generic. No grading, no scoring, no per-user/per-run state (locked).
Each entry is presented content only: a question plus its explanation.
"""

from __future__ import annotations

from app.schemas.learning import ComprehensionCheck

COMPREHENSION_CHECKS: list[ComprehensionCheck] = [
    ComprehensionCheck(
        question="A model has high recall but low precision. What does that suggest about its predictions?",
        answer_explanation=(
            "It's catching most of the actual positive cases (high "
            "recall), but a lot of what it flags as positive is "
            "actually negative (low precision) — it's over-predicting "
            "the positive class, generating a lot of false alarms "
            "along with the real catches."
        ),
        related_concept="Precision vs. Recall",
    ),
    ComprehensionCheck(
        question="Why can accuracy alone be misleading on an imbalanced dataset (e.g. 95% 'No', 5% 'Yes')?",
        answer_explanation=(
            "A model that always predicts 'No' would score 95% accuracy "
            "without ever correctly identifying a single 'Yes' case — "
            "accuracy doesn't distinguish between getting the easy "
            "majority class right and actually catching the minority "
            "class, which is usually the case that matters."
        ),
        related_concept="Accuracy vs. Imbalance",
    ),
    ComprehensionCheck(
        question="Why is F1 used as PIPER's model-selection metric instead of accuracy or precision alone?",
        answer_explanation=(
            "F1 is the harmonic mean of precision and recall, so it "
            "stays low if either one is low — a model can't win by "
            "being extreme in only one direction (e.g. predicting "
            "everything positive to max out recall). Precision or "
            "recall alone, or accuracy on imbalanced data, can all be "
            "gamed that way."
        ),
        related_concept="Model Selection",
    ),
    ComprehensionCheck(
        question="What does one-hot encoding do to a categorical column, and why not just assign each category a number (0, 1, 2, ...)?",
        answer_explanation=(
            "One-hot encoding turns a categorical column into several "
            "binary (0/1) columns, one per category. Assigning plain "
            "numbers instead (label encoding) would imply a false "
            "ordering or distance between categories that don't "
            "actually have one — e.g. 'Two year' isn't numerically "
            "'twice' 'One year'."
        ),
        related_concept="Feature Engineering",
    ),
    ComprehensionCheck(
        question="Why must feature scaling (standardization) be fit only on the training split, never the full dataset?",
        answer_explanation=(
            "Fitting scaling statistics (mean/standard deviation) on the "
            "full dataset would let information from the test split "
            "leak into training — the model would be evaluated on data "
            "whose distribution it had already partially seen, making "
            "the test metrics look better than the model would actually "
            "perform on truly unseen data."
        ),
        related_concept="Data Leakage",
    ),
    ComprehensionCheck(
        question="What does PIPER's 'baseline gate' guardrail actually protect against?",
        answer_explanation=(
            "It compares the trained model's F1 against a trivial "
            "majority-class baseline (always predicting the most common "
            "label) on the same test split. If the real model isn't "
            "meaningfully better than that trivial baseline, the "
            "guardrail fails — protecting against reporting a model "
            "that looks 'trained' but isn't actually learning anything "
            "useful."
        ),
        related_concept="Guardrails",
    ),
    ComprehensionCheck(
        question="Why does PIPER REPLAN after a failed guardrail check instead of just failing the run immediately?",
        answer_explanation=(
            "Some guardrail failures are recoverable by a different plan "
            "— e.g. dropping a leaky feature next attempt. PIPER's graph "
            "(never the LLM) decides whether to REPLAN, bounded by "
            "max_retries, so a fixable problem gets a real second "
            "attempt instead of failing on the first correctable mistake."
        ),
        related_concept="REPLAN",
    ),
]
