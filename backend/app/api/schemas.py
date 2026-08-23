"""
API-layer request/response models (M5).

Deliberately thin: every ML/agent-domain field is a DIRECT reference to
the existing core schema it represents (FailureInfo,
PipelineValidationResult, ModelComparison, BaselineComparisonResult,
ReproducibilityMetadata, TrainingResult, EvaluationResult) — this layer
never redefines or duplicates their shape. Only genuinely API-specific
concerns (request bodies, envelope/status fields) get new models here,
per "FastAPI should expose the core, not replace it."
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.baseline import BaselineComparisonResult
from app.schemas.evaluation import EvaluationResult, ModelComparison
from app.schemas.failure import FailureInfo
from app.schemas.guardrails import PipelineValidationResult
from app.schemas.ingestion import DatasetFormat, SheetInfo
from app.schemas.reproducibility import ReproducibilityMetadata
from app.schemas.training import Algorithm, TrainingResult


class DatasetUploadResponse(BaseModel):
    """
    The four original fields (dataset_id/filename/rows/columns) are
    unchanged — every existing client keeps working. Multi-format
    ingestion only ADDS the detected-format evidence below, so the user
    can confirm what PIPER thought the file was, and its dimensions,
    before starting a run.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    filename: str
    rows: int
    columns: list[str]

    detected_format: Optional[DatasetFormat] = None
    column_count: Optional[int] = None
    sheet_name: Optional[str] = Field(
        default=None, description="Excel only: the worksheet actually ingested."
    )
    available_sheets: list[SheetInfo] = Field(
        default_factory=list, description="Excel only: every worksheet found in the workbook."
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Ingestion decisions worth surfacing (e.g. which Excel sheet was chosen).",
    )


class DatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_ids: list[str]


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    target_column: str
    max_retries: int = Field(
        default=2,
        ge=0,
        le=20,
        description=(
            "Matches AgentState's own default (2). Capped at 20 here as a "
            "sane API-boundary guard — this is a REST request-validation "
            "concern, not a change to AgentState's own contract, which "
            "still accepts any int when constructed directly (e.g. in "
            "tests). PIPER's internal execution-step budget "
            "(MAX_EXECUTION_STEPS, see app/agent/graph.py) is the real, "
            "unconditional termination guarantee regardless of this value."
        ),
    )


class CreateRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset_id: str
    target_column: str
    status: str
    current_node: Optional[str] = None
    attempt: int
    plan_history: list[str] = Field(default_factory=list)


class RunListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset_id: str
    target_column: str
    status: str
    current_node: Optional[str] = None
    attempt: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RunListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[RunListItem]


class RunResultResponse(BaseModel):
    """
    Only meaningful once a run has reached a terminal status
    ("completed"/"failed") — see GET /runs/{run_id}/result, which
    returns 409 before that point rather than a partially-populated
    instance of this model.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    validation: Optional[PipelineValidationResult] = None
    comparison: Optional[ModelComparison] = None
    baseline: Optional[BaselineComparisonResult] = None
    failure: Optional[FailureInfo] = None
    reproducibility: Optional[ReproducibilityMetadata] = None
    model_results: list[TrainingResult] = Field(default_factory=list)
    evaluation_results: list[EvaluationResult] = Field(default_factory=list)
    error: Optional[str] = None


class CreateExplorationRequest(BaseModel):
    """
    Batch 6B (PIPER Learn: Learn-Explore) request body for
    POST /runs/{run_id}/explore. Exactly one of `new_algorithm` or
    (`hyperparameter_name` and `hyperparameter_value`) must be
    provided — enforced by explore_alternative() itself, not
    re-validated here, so the one structured error path stays
    authoritative.
    """

    model_config = ConfigDict(extra="forbid")

    base_model_id: str
    new_algorithm: Optional[Algorithm] = None
    hyperparameter_name: Optional[str] = None
    hyperparameter_value: Optional[float] = None


class ArtifactStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    artifact_status: str
    parity_status: str
    winning_model_id: Optional[str] = None
    algorithm: Optional[str] = None
    files: list[str] = Field(default_factory=list)
    error: Optional[dict] = None
    created_at: Optional[str] = None
    parity: Optional[dict] = None


class ArtifactFileListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    artifact_status: str
    files: list[str] = Field(default_factory=list)
