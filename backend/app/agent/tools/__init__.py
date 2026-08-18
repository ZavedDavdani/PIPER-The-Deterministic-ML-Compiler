"""
Deterministic tools. Every function here follows the same shape:
    (dataset_id, ..., store: DatasetStore) -> ToolResult[SomeSchema]

Tools never call the LLM and never decide what action to take — they
execute exactly what's requested and report a structured result.

M1 profiling group:
    profile_dataset, inspect_column, detect_missing_values, detect_outliers

M1 cleaning group:
    drop_column, drop_duplicates, impute_missing_values, convert_column_type

Preparation group:
    split_dataset

Feature engineering group:
    encode_categorical_features, scale_features, create_date_features

Training group:
    train_model

Evaluation group:
    evaluate_model, compare_models

Guardrails group:
    check_data_leakage, check_target_imbalance, check_constant_features,
    check_high_cardinality, validate_pipeline

Baseline group:
    compute_baseline

Exploration group (Batch 6B: Learn-Explore):
    explore_alternative

Ingestion group (multi-format upload):
    ingest_dataset, detect_format
"""

from app.agent.tools.profiling import inspect_column, profile_dataset
from app.agent.tools.detection import detect_missing_values, detect_outliers
from app.agent.tools.cleaning_ops import drop_column, drop_duplicates
from app.agent.tools.type_conversion import convert_column_type, impute_missing_values
from app.agent.tools.preparation import split_dataset
from app.agent.tools.feature_engineering import (
    create_date_features,
    encode_categorical_features,
    scale_features,
)
from app.agent.tools.training import train_model
from app.agent.tools.evaluation import compare_models, evaluate_model
from app.agent.tools.guardrails import (
    check_constant_features,
    check_data_leakage,
    check_high_cardinality,
    check_target_imbalance,
    validate_pipeline,
)
from app.agent.tools.baseline import compute_baseline
from app.agent.tools.exploration import explore_alternative
from app.agent.tools.ingestion import detect_format, ingest_dataset

from app.agent.tools.data_quality import validate_data_quality

from app.agent.tools.sanitization import sanitize_for_llm_context
from app.agent.tools.sanitized_llm_context import build_sanitized_llm_context

__all__ = [
    "profile_dataset",
    "inspect_column",
    "detect_missing_values",
    "detect_outliers",
    "drop_column",
    "drop_duplicates",
    "impute_missing_values",
    "convert_column_type",
    "split_dataset",
    "encode_categorical_features",
    "scale_features",
    "create_date_features",
    "train_model",
    "evaluate_model",
    "compare_models",
    "check_data_leakage",
    "check_target_imbalance",
    "check_constant_features",
    "check_high_cardinality",
    "validate_pipeline",
    "compute_baseline",
    "explore_alternative",
    "ingest_dataset",
    "detect_format",
    "validate_data_quality",
    "sanitize_for_llm_context",
    "build_sanitized_llm_context",
]
