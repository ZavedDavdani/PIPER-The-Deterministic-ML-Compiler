"""
Preparation tool contract: split_dataset().
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SplitResult(BaseModel):
    """
    Output of split_dataset(target, test_size).

    For binary classification, stratified=True by default (locked).
    random_state is always 42 (locked, for reproducibility).
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    split_id: str
    target: str
    train_rows: int = Field(..., ge=0)
    test_rows: int = Field(..., ge=0)
    test_size: float = Field(..., gt=0.0, lt=1.0)
    stratified: bool
    random_state: int = 42
