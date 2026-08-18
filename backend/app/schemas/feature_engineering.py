"""
Feature engineering tool contracts: encode_categorical_features(),
scale_features(), create_date_features().

Locked design: these tools record WHAT was requested (which columns,
which transformation) and validate that the request is sensible
against the real data — they do NOT fit a transformer. The actual
fitted OneHotEncoder/StandardScaler is assembled into a scikit-learn
Pipeline at train_model() time, fit on the train split only, to avoid
leaking test-split information into the transformation. This mirrors
scale_features()'s original contract note almost exactly and is
applied consistently to encoding too.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EncodingResult(BaseModel):
    """
    Output of encode_categorical_features(cols).

    generated_columns is a best-effort PREVIEW of what one-hot
    encoding would produce (based on the categories actually observed
    in the current dataset state), not a guarantee — the real fitted
    encoder at train time may see slightly different categories if the
    train split doesn't contain every category observed here. This
    distinction is intentional and documented so nothing downstream
    treats generated_columns as authoritative.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    original_columns: list[str]
    generated_columns: list[str] = Field(
        ...,
        description="Preview only — see class docstring. Real columns are determined at train_model() time.",
    )
    encoding: Literal["one_hot"] = "one_hot"


class ScalingResult(BaseModel):
    """
    Output of scale_features(cols).

    Records the requested scaling; no scaler is fit here (locked
    contract — the fitted StandardScaler belongs to the training
    pipeline).
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    columns: list[str]
    scaler: Literal["StandardScaler"] = "StandardScaler"


class DateFeatureResult(BaseModel):
    """
    Output of create_date_features(col).

    generated_features lists only the components actually derivable
    from the parsed date column — e.g. if every value in the column
    has the same day_of_week (unlikely but possible), it's still
    included; this only omits a component if the column couldn't be
    parsed as a date at all, in which case the tool returns an error
    instead of a partial result.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    source_column: str
    generated_features: list[Literal["year", "month", "day", "day_of_week"]]
