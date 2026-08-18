"""
Training tool contract: train_model().

FeatureEngineeringIntent is the explicit input that carries forward
the encode_categorical_features()/scale_features() decisions already
validated earlier in the pipeline — train_model() consumes this rather
than inspecting the dataset and silently deciding preprocessing for
itself. This preserves the chain: PLAN -> feature intent ->
train_model() -> Pipeline construction -> fit(train only).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Algorithm = Literal["logistic_regression", "random_forest"]


class FeatureEngineeringIntent(BaseModel):
    """
    What train_model() consumes instead of deciding preprocessing
    itself. Built from the EncodingResult/ScalingResult that
    encode_categorical_features()/scale_features() already produced
    and validated earlier in the run.
    """

    model_config = ConfigDict(extra="forbid")

    categorical_columns: list[str] = Field(
        default_factory=list,
        description="Columns to one-hot encode, as validated by encode_categorical_features().",
    )
    numeric_columns_to_scale: list[str] = Field(
        default_factory=list,
        description="Columns to standard-scale, as validated by scale_features().",
    )


class TrainingResult(BaseModel):
    """Output of train_model()."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    algorithm: Algorithm
    parameters: dict
    split_id: str
    training_rows: int = Field(..., ge=0)
    feature_count: int = Field(..., ge=0)
    training_duration_seconds: float = Field(..., ge=0.0)
