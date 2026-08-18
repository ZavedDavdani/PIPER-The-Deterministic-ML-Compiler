"""
Evaluation tool contracts: evaluate_model(), compare_models().

Locked design: model selection uses F1 as the primary metric (with
ROC-AUC as secondary context, not an alternate selection metric the
LLM can pick to make a preferred model look better) — this was decided
explicitly to prevent metric-shopping.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConfusionMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tn: int = Field(..., ge=0)
    fp: int = Field(..., ge=0)
    fn: int = Field(..., ge=0)
    tp: int = Field(..., ge=0)


class EvaluationResult(BaseModel):
    """Output of evaluate_model()."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    split_id: str
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    roc_auc: float = Field(..., ge=0.0, le=1.0)
    confusion_matrix: ConfusionMatrix
    test_rows: int = Field(..., ge=0)


class ModelComparisonEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    algorithm: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


class ModelComparison(BaseModel):
    """
    Output of compare_models().

    selection_metric is always "f1" for V1 (locked) — never left to
    the LLM to choose, which would allow selecting whichever metric
    happens to favor a preferred model.
    """

    model_config = ConfigDict(extra="forbid")

    models: list[ModelComparisonEntry]
    recommended_model_id: str
    selection_metric: Literal["f1"] = "f1"
    justification: str = Field(
        default="",
        description=(
            "Deterministic, template-based explanation of why "
            "recommended_model_id was selected, built only from the "
            "real metrics already present in `models` — e.g. "
            "\"logistic_regression selected: F1 0.4969 vs. 0.4848 for "
            "random_forest.\" Never LLM-generated, per the locked "
            "principle that model selection itself is deterministic "
            "and never LLM-influenced (see compare_models())."
        ),
    )
