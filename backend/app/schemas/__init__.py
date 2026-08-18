"""
M1 schemas: profiling + cleaning contracts, plus the shared ToolResult
envelope. Feature-engineering, training, evaluation, and guardrail
schemas are added when those tool groups are implemented (later in M1
per the locked order: profiling -> cleaning -> [rest]).
"""

from app.schemas.common import ToolError, ToolResult
from app.schemas.profiling import (
    ColumnInspection,
    ColumnProfile,
    DatasetProfile,
    MissingValueColumnEntry,
    MissingValueReport,
    OutlierColumnEntry,
    OutlierReport,
)
from app.schemas.cleaning import (
    ConversionResult,
    DropColumnResult,
    DropDuplicatesResult,
    ImputationResult,
)
from app.schemas.preparation import SplitResult
from app.schemas.feature_engineering import DateFeatureResult, EncodingResult, ScalingResult
from app.schemas.training import Algorithm, FeatureEngineeringIntent, TrainingResult
from app.schemas.evaluation import (
    ConfusionMatrix,
    EvaluationResult,
    ModelComparison,
    ModelComparisonEntry,
)
from app.schemas.guardrails import (
    ConstantFeatureEntry,
    ConstantFeatureReport,
    HighCardinalityEntry,
    HighCardinalityReport,
    ImbalanceReport,
    LeakageReport,
    LeakageViolation,
    PipelineValidationResult,
    ValidationCheck,
)
from app.schemas.baseline import BASELINE_POLICY, BaselineComparisonResult, BaselineMetrics, BaselinePolicy
from app.schemas.failure import (
    FailureCategory,
    FailureInfo,
    RECOVERABLE_CATEGORIES,
    RecoverableFailureCategory,
    TERMINAL_CATEGORIES,
    TerminalFailureCategory,
)

from app.schemas.data_quality import DataQualityReport, DataQualityViolation, DATA_QUALITY_FAILURE_CATEGORY, MINIMUM_SAMPLES_REQUIRED

from app.schemas.sanitization import ColumnRiskClassification, SanitizationFinding, SanitizationReport, MAX_SAFE_TEXT_LENGTH

__all__ = [
    "ToolError",
    "ToolResult",
    "ColumnInspection",
    "ColumnProfile",
    "DatasetProfile",
    "MissingValueColumnEntry",
    "MissingValueReport",
    "OutlierColumnEntry",
    "OutlierReport",
    "ConversionResult",
    "DropColumnResult",
    "DropDuplicatesResult",
    "ImputationResult",
    "SplitResult",
    "EncodingResult",
    "ScalingResult",
    "DateFeatureResult",
    "Algorithm",
    "FeatureEngineeringIntent",
    "TrainingResult",
    "ConfusionMatrix",
    "EvaluationResult",
    "ModelComparison",
    "ModelComparisonEntry",
    "LeakageReport",
    "LeakageViolation",
    "ImbalanceReport",
    "ConstantFeatureReport",
    "ConstantFeatureEntry",
    "HighCardinalityReport",
    "HighCardinalityEntry",
    "PipelineValidationResult",
    "ValidationCheck",
    "BaselinePolicy",
    "BASELINE_POLICY",
    "BaselineMetrics",
    "BaselineComparisonResult",
    "FailureCategory",
    "TerminalFailureCategory",
    "RecoverableFailureCategory",
    "TERMINAL_CATEGORIES",
    "RECOVERABLE_CATEGORIES",
    "FailureInfo",
    "DataQualityReport",
    "DataQualityViolation",
    "DATA_QUALITY_FAILURE_CATEGORY",
    "MINIMUM_SAMPLES_REQUIRED",
    "ColumnRiskClassification",
    "SanitizationFinding",
    "SanitizationReport",
    "MAX_SAFE_TEXT_LENGTH",
]
