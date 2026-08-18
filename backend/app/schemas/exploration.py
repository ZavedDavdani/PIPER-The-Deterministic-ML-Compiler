"""
PIPER Learn — Learn-Explore schemas (Batch 6B).

One exploration = exactly ONE changed variable relative to a base
model already trained during a real, terminal PIPER run — either the
algorithm itself, or a single hyperparameter already inside its locked
allowlist/bounds (see app/agent/tools/training.py). Never both, never
anything outside what train_model() already supports. See
app/agent/tools/exploration.py's explore_alternative() for the
deterministic construction — no new training logic, exactly the same
train_model()/evaluate_model()/compare_models() the original run used.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.evaluation import EvaluationResult, ModelComparison
from app.schemas.learning import EvaluationExplanation, ModelSelectionExplanation
from app.schemas.training import TrainingResult


class ExplorationVariable(BaseModel):
    """
    The ONE thing this exploration changed relative to the base model.
    old_value/new_value are stringified so this shape works uniformly
    for both an algorithm name and a numeric hyperparameter value — the
    real typed value actually used for training lives in this
    exploration's own TrainingResult.algorithm/parameters, nothing is
    lost by stringifying here.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["model", "hyperparameter"]
    name: str = Field(..., description="'algorithm' for a model swap, or the hyperparameter's name (e.g. 'n_estimators').")
    old_value: str
    new_value: str


class ExplorationResult(BaseModel):
    """
    One exploration's full result — isolated by experiment_id, never
    merged into the original run's own state. `run_id` is a reference
    to the original run only (for display/lookup); nothing here is
    ever written back into that run's AgentState/RunStore record.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    run_id: str
    base_model_id: str
    split_id: str = Field(..., description="The SAME split_id reused from the original run's base model — no new splitting.")
    variable_changed: ExplorationVariable
    training: TrainingResult
    evaluation: EvaluationResult
    comparison_vs_base: ModelComparison = Field(
        ..., description="compare_models([base_model_id, new_model_id], ...) — the same real, F1-max comparison logic the original run used."
    )
    evaluation_explanation: Optional[EvaluationExplanation] = Field(
        default=None, description="Learn-Explain integration: the same deterministic, template-based explanation build_run_explanation() would produce, for this exploration's own new model."
    )
    comparison_explanation: Optional[ModelSelectionExplanation] = Field(
        default=None, description="Learn-Explain integration: which of {base model, exploration variant} compare_models() would pick, and why."
    )
    created_at: str
