"""Phase 4 governance view models. Read-only; never consulted by the graph."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Availability = Literal["AVAILABLE", "NOT_AVAILABLE", "NOT_REQUESTED"]
FairnessStatus = Literal["AVAILABLE", "NOT_REQUESTED", "INSUFFICIENT_DATA", "NOT_AVAILABLE"]
ImportanceMethod = Literal[
    "logistic_regression_coefficients",
    "random_forest_impurity",
    "NOT_AVAILABLE",
]


class RecordedMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: Optional[float] = None


class CandidateModelCardEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    algorithm: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    selected: bool = False


class GuardrailCardEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    passed: bool
    severity: str
    message: str


class FeatureImportanceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    transformed_feature: str
    importance: float
    direction: Optional[Literal["positive", "negative", "neutral"]] = None
    source_feature: Optional[str] = None


class FeatureImportanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Availability
    method: ImportanceMethod = "NOT_AVAILABLE"
    algorithm: Optional[str] = None
    rows: list[FeatureImportanceRow] = Field(default_factory=list)
    disclaimer: str
    reason: Optional[str] = None


class SubgroupMetricRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    group: str
    n: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    selection_rate: Optional[float] = None
    disparate_impact_ratio: Optional[float] = None
    sufficient: bool
    warning: Optional[str] = None


class FairnessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: FairnessStatus
    requested_columns: list[str] = Field(default_factory=list)
    minimum_group_size: int
    positive_class: Optional[str] = None
    reference_group_rule: str
    groups: list[SubgroupMetricRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str
    reason: Optional[str] = None


class HashEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["CONTENT_HASH", "METADATA"]
    algorithm: str
    digest: Optional[str] = None
    available: bool
    reason: Optional[str] = None


class FingerprintManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    hash_algorithm: str
    content_hashes: list[HashEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    caveat: str


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Availability
    run_id: str
    dataset_id: Optional[str] = None
    task_type: Optional[str] = None
    target: Optional[str] = None
    winning_model_id: Optional[str] = None
    winning_algorithm: Optional[str] = None
    candidate_models: list[CandidateModelCardEntry] = Field(default_factory=list)
    evaluation_metrics: list[RecordedMetric] = Field(default_factory=list)
    baseline_comparison: Optional[dict[str, Any]] = None
    train_test_split: Optional[dict[str, Any]] = None
    preprocessing_summary: list[str] = Field(default_factory=list)
    guardrail_results: list[GuardrailCardEntry] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    artifact_information: Optional[dict[str, Any]] = None
    feature_importance: FeatureImportanceReport
    reason: Optional[str] = None


class DataCardColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: Optional[str] = None
    missing_count: Optional[int] = None
    missing_percentage: Optional[float] = None
    unique_count: Optional[int] = None
    role: Literal["target", "feature", "unknown"] = "unknown"
    kind: Optional[Literal["numeric", "categorical", "other"]] = None


class DataCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Availability
    run_id: str
    dataset_id: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    target: Optional[str] = None
    feature_list: list[str] = Field(default_factory=list)
    column_summaries: list[DataCardColumn] = Field(default_factory=list)
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    missingness: list[dict[str, Any]] = Field(default_factory=list)
    preprocessing_operations: list[dict[str, Any]] = Field(default_factory=list)
    train_test: Optional[dict[str, Any]] = None
    data_quality_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


class GovernanceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["piper.governance.v1"] = "piper.governance.v1"
    run_id: str
    run_status: str
    model_card: ModelCard
    data_card: DataCard
    fingerprints: FingerprintManifest
    feature_importance: FeatureImportanceReport
    fairness: FairnessReport
    limitations: list[str] = Field(default_factory=list)
    artifact_status: Optional[dict[str, Any]] = None
    notes: list[str] = Field(default_factory=list)
