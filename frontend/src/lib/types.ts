/**
 * TypeScript mirrors of the real FastAPI/pydantic schemas (backend
 * app/api/schemas.py + app/schemas/*.py). Every field here matches the
 * backend's actual JSON shape exactly — this file has no invented
 * fields and no mock/sample data; it exists purely so the rest of the
 * frontend gets type safety against the REAL API contract.
 */

// --- Dataset endpoints -------------------------------------------------

export type DatasetFormat = 'csv' | 'tsv' | 'excel' | 'json' | 'ipynb' | 'parquet'

export interface SheetInfo {
  name: string
  rows: number
  columns: number
}

export interface DatasetUploadResponse {
  dataset_id: string
  filename: string
  rows: number
  columns: string[]
  /** Multi-format ingestion: optional so older/mocked responses still typecheck. */
  detected_format?: DatasetFormat | null
  column_count?: number | null
  sheet_name?: string | null
  available_sheets?: SheetInfo[]
  notes?: string[]
}

export interface DatasetListResponse {
  dataset_ids: string[]
}

export interface ColumnProfile {
  name: string
  dtype: string
  missing_count: number
  missing_percentage: number
  unique_count: number
  unique_percentage: number
  sample_values: unknown[]
  min: number | null
  max: number | null
  mean: number | null
  median: number | null
  std: number | null
}

export interface DatasetProfile {
  dataset_id: string
  rows: number
  columns: number
  column_profiles: ColumnProfile[]
  duplicate_rows: number
  memory_usage_bytes: number
}

// --- Run lifecycle -------------------------------------------------------

export interface CreateRunRequest {
  dataset_id: string
  target_column: string
  max_retries?: number
}

export interface CreateRunResponse {
  run_id: string
  status: string
}

export type RunStatus = 'initialized' | 'running' | 'replanning' | 'completed' | 'failed'

export interface RunStatusResponse {
  run_id: string
  dataset_id: string
  target_column: string
  status: string
  current_node: string | null
  attempt: number
  plan_history: string[]
}

// --- Guardrails / validation ---------------------------------------------

export type ValidationSeverity = 'info' | 'warning' | 'error'

export interface ValidationCheck {
  check: string
  passed: boolean
  severity: ValidationSeverity
  message: string
}

export interface PipelineValidationResult {
  dataset_id: string
  target_column: string
  valid: boolean
  checks: ValidationCheck[]
  violations: ValidationCheck[]
  warnings: ValidationCheck[]
  scope_note: string
}

// --- Failure taxonomy ------------------------------------------------------

export type FailureCategory =
  | 'DATA_ERROR'
  | 'SCHEMA_ERROR'
  | 'TARGET_ERROR'
  | 'EXECUTION_BUDGET_EXCEEDED'
  | 'LEAKAGE_ERROR'
  | 'IMBALANCE_ERROR'
  | 'FEATURE_ERROR'
  | 'TRAINING_ERROR'
  | 'EVALUATION_ERROR'
  | 'DUPLICATE_PLAN'
  | 'BASELINE_GATE_FAILED'

export interface FailureInfo {
  category: FailureCategory
  message: string
  evidence: Record<string, unknown>
  node: string
  attempt: number
  retryable: boolean
  human_intervention_required: boolean
}

// --- Models / evaluation ---------------------------------------------------

export interface ConfusionMatrix {
  tn: number
  fp: number
  fn: number
  tp: number
}

export interface EvaluationResult {
  model_id: string
  split_id: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc: number
  confusion_matrix: ConfusionMatrix
  test_rows: number
}

export interface TrainingResult {
  model_id: string
  algorithm: 'logistic_regression' | 'random_forest'
  parameters: Record<string, unknown>
  split_id: string
  training_rows: number
  feature_count: number
  training_duration_seconds: number
}

export interface ModelComparisonEntry {
  model_id: string
  algorithm: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc: number
}

export interface ModelComparison {
  models: ModelComparisonEntry[]
  recommended_model_id: string
  selection_metric: 'f1'
}

// --- Baseline --------------------------------------------------------------

export interface BaselineMetrics {
  accuracy: number
  precision: number | null
  recall: number | null
  f1: number | null
  majority_class: string
}

export interface BaselineComparisonResult {
  model_id: string
  split_id: string
  baseline: BaselineMetrics
  model_primary_metric_value: number
  baseline_primary_metric_value: number | null
  primary_metric: string
  delta: number | null
  minimum_required_delta: number
  gate_passed: boolean
  reason: string
}

// --- Reproducibility ---------------------------------------------------------

export interface EnvironmentMetadata {
  python_version: string
  pandas_version: string
  numpy_version: string
  sklearn_version: string
}

export interface ReproducibilityMetadata {
  environment: EnvironmentMetadata
  dataset_fingerprint: string
  split_random_state: number
  model_random_state: number
  pipeline_fingerprint: string | null
}

// --- Result ------------------------------------------------------------------

export interface RunResultResponse {
  run_id: string
  status: string
  validation: PipelineValidationResult | null
  comparison: ModelComparison | null
  baseline: BaselineComparisonResult | null
  failure: FailureInfo | null
  reproducibility: ReproducibilityMetadata | null
  model_results: TrainingResult[]
  evaluation_results: EvaluationResult[]
  error: string | null
}

// --- Live trace events (SSE) --------------------------------------------------

export type TraceEventType =
  | 'node_started'
  | 'node_completed'
  | 'node_failed'
  | 'tool_called'
  | 'guardrail_checked'
  | 'validation_result'
  | 'replan_triggered'
  | 'duplicate_plan_detected'
  | 'retry_exhausted'
  | 'run_completed'
  | 'run_failed'

export type TraceEventStatus = 'success' | 'failure' | 'info'
export type TraceEventSeverity = 'info' | 'warning' | 'error'

export interface TraceEvent {
  run_id: string
  step_id: string
  parent_span_id: string | null
  attempt: number
  node: string
  event_type: TraceEventType
  timestamp: string
  status: TraceEventStatus
  severity: TraceEventSeverity
  tool_name: string | null
  guardrail_name: string | null
  evidence: Record<string, unknown>
  duration_ms: number
  message: string
}

export interface ApiErrorBody {
  detail: string
}
