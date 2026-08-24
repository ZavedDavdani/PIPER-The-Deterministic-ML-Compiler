"""Phase 5 deployment / Test Flight view models."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    rows: list[dict[str, Any]] = Field(min_length=1)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    artifact_id: str
    winning_model_id: Optional[str] = None
    algorithm: Optional[str] = None
    row_count: int
    predictions: list[Any]
    schema_status: Literal["valid"]
    required_columns: list[str]
    parity: dict[str, Any]
    data_kind: Literal["NEW_UNSEEN_DATA"] = "NEW_UNSEEN_DATA"
    sample: list[dict[str, Any]] = Field(default_factory=list)


class DeploymentReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["READY", "NOT_READY"]
    artifact_status: Optional[str] = None
    winning_model_id: Optional[str] = None
    algorithm: Optional[str] = None
    required_columns: list[str] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    reason: Optional[dict[str, Any]] = None


class DeploymentPackageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    files: list[str]
    docker_optional: bool = True
