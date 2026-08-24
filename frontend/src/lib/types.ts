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
  detail: string | { code?: string; message?: string; details?: unknown }
}

// --- V1.2 productization (decision trace / verdict / intervention / evidence) ---

export type StageId =
  | 'LLM_PROPOSED'
  | 'VALIDATED'
  | 'ADEQUACY'
  | 'REPLAN'
  | 'EXECUTION'
  | 'TRAINING'
  | 'EVALUATION'
  | 'GUARDRAILS'
  | 'FINAL_VERDICT'

export type StageStatus = 'pending' | 'current' | 'passed' | 'failed' | 'skipped' | 'not_reached'

export interface ExecutableStep {
  tool_name: string
  arguments: Record<string, unknown>
}

export interface PlanningAttempt {
  attempt: number
  proposed_steps: ExecutableStep[]
  plan_hash: string | null
  structurally_valid: boolean
  adequacy_status: 'PASS' | 'FAIL' | null
  outcome: 'provider_error' | 'invalid' | 'duplicate' | 'inadequate' | 'accepted'
  violation_count: number
  material_finding_count: number
}

export interface DecisionStage {
  id: StageId
  label: string
  status: StageStatus
  summary: string
  attempt: number | null
  evidence: Record<string, unknown>
}

export interface DecisionTrace {
  run_id: string
  run_status: string
  stages: DecisionStage[]
  planning_attempts: PlanningAttempt[]
  plan_diffs: Record<string, unknown>[]
}

export interface PiperVerdict {
  run_id: string
  outcome: 'ACCEPTED' | 'REJECTED' | 'HUMAN_INTERVENTION_REQUIRED'
  reason_code: string
  summary: string
  retry_count: number
  max_retries: number
  structurally_valid_plan: boolean
  adequacy_passed: boolean | null
  guardrails_passed: boolean | null
  human_intervention_required: boolean
  executed: boolean
}

export interface HumanInterventionPackage {
  run_id: string
  required: boolean
  headline: string
  failure_category: string | null
  failure_message: string | null
  retry_count: number
  max_retries: number
  last_proposed_steps: ExecutableStep[]
  structural_violations: Record<string, unknown>[]
  material_adequacy_findings: Record<string, unknown>[]
  advisory_adequacy_findings: Record<string, unknown>[]
  preserved_valid_steps: ExecutableStep[]
  implicated_steps: ExecutableStep[]
  recommended_actions: string[]
  blocked_invalid_execution: boolean
}

export interface EvidenceExport {
  schema_version: 'piper.evidence.v1'
  run_id: string
  dataset_id: string
  target_column: string
  status: string
  decision_trace: DecisionTrace
  verdict: PiperVerdict | null
  intervention: HumanInterventionPackage
  notes: string[]
}

export interface RunListItem {
  run_id: string
  dataset_id: string
  target_column: string
  status: string
  current_node: string | null
  attempt: number
  created_at: string | null
  updated_at: string | null
}

export interface RunListResponse {
  runs: RunListItem[]
}

export interface ReplayResponse {
  run_id: string
  llm_invoked: false
  source: 'persisted_events_and_state'
  status: string
  decision_trace: DecisionTrace
  verdict: PiperVerdict | null
  intervention: HumanInterventionPackage
  evidence: EvidenceExport
}

export interface ArtifactStatusResponse {
  run_id: string
  artifact_status: 'NOT_GENERATED' | 'VERIFIED' | 'FAILED' | string
  parity_status: string
  winning_model_id: string | null
  algorithm: string | null
  files: string[]
  error: { code?: string; message?: string; details?: Record<string, unknown> } | null
  created_at: string | null
  parity?: Record<string, unknown> | null
}

export interface ArtifactFileListResponse {
  run_id: string
  artifact_status: string
  files: string[]
}

export interface OllamaStatusResponse {
  host: string
  model: string
  keep_alive: string
  reachable: boolean
  models: string[]
  error: string | null
}

export type GovernanceAvailability = 'AVAILABLE' | 'NOT_AVAILABLE' | 'NOT_REQUESTED'
export type FairnessStatus = 'AVAILABLE' | 'NOT_REQUESTED' | 'INSUFFICIENT_DATA' | 'NOT_AVAILABLE'

export interface FeatureImportanceRow {
  feature: string
  transformed_feature: string
  importance: number
  direction: 'positive' | 'negative' | 'neutral' | null
  source_feature: string | null
}

export interface FeatureImportanceReport {
  status: GovernanceAvailability
  method: string
  algorithm: string | null
  rows: FeatureImportanceRow[]
  disclaimer: string
  reason: string | null
}

export interface HashEntry {
  name: string
  kind: 'CONTENT_HASH' | 'METADATA'
  algorithm: string
  digest: string | null
  available: boolean
  reason: string | null
}

export interface FingerprintManifest {
  run_id: string
  hash_algorithm: string
  content_hashes: HashEntry[]
  metadata: Record<string, unknown>
  caveat: string
}

export interface CandidateModelCardEntry {
  model_id: string
  algorithm: string
  accuracy: number | null
  precision: number | null
  recall: number | null
  f1: number | null
  roc_auc: number | null
  selected: boolean
}

export interface ModelCard {
  status: GovernanceAvailability
  run_id: string
  dataset_id: string | null
  task_type: string | null
  target: string | null
  winning_model_id: string | null
  winning_algorithm: string | null
  candidate_models: CandidateModelCardEntry[]
  evaluation_metrics: { name: string; value: number | null }[]
  baseline_comparison: Record<string, unknown> | null
  train_test_split: Record<string, unknown> | null
  preprocessing_summary: string[]
  guardrail_results: { check: string; passed: boolean; severity: string; message: string }[]
  limitations: string[]
  artifact_information: Record<string, unknown> | null
  feature_importance: FeatureImportanceReport
  reason: string | null
}

export interface DataCard {
  status: GovernanceAvailability
  run_id: string
  dataset_id: string | null
  rows: number | null
  columns: number | null
  target: string | null
  feature_list: string[]
  column_summaries: {
    name: string
    dtype: string | null
    missing_count: number | null
    missing_percentage: number | null
    unique_count: number | null
    role: string
    kind: string | null
  }[]
  numeric_features: string[]
  categorical_features: string[]
  missingness: Record<string, unknown>[]
  preprocessing_operations: Record<string, unknown>[]
  train_test: Record<string, unknown> | null
  data_quality_findings: string[]
  limitations: string[]
  reason: string | null
}

export interface SubgroupMetricRow {
  column: string
  group: string
  n: number
  accuracy: number | null
  precision: number | null
  recall: number | null
  f1: number | null
  selection_rate: number | null
  disparate_impact_ratio: number | null
  sufficient: boolean
  warning: string | null
}

export interface FairnessReport {
  status: FairnessStatus
  requested_columns: string[]
  minimum_group_size: number
  positive_class: string | null
  reference_group_rule: string
  groups: SubgroupMetricRow[]
  warnings: string[]
  disclaimer: string
  reason: string | null
}

export interface GovernanceBundle {
  schema_version: 'piper.governance.v1'
  run_id: string
  run_status: string
  model_card: ModelCard
  data_card: DataCard
  fingerprints: FingerprintManifest
  feature_importance: FeatureImportanceReport
  fairness: FairnessReport
  limitations: string[]
  artifact_status: Record<string, unknown> | null
  notes: string[]
}

export interface PredictResponse {
  run_id: string
  artifact_id: string
  winning_model_id: string | null
  algorithm: string | null
  row_count: number
  predictions: unknown[]
  schema_status: 'valid'
  required_columns: string[]
  parity: { parity_status: string; mismatched_rows?: number; row_count?: number }
  data_kind: 'NEW_UNSEEN_DATA'
  sample: Record<string, unknown>[]
}

export interface DeploymentReadinessResponse {
  run_id: string
  status: 'READY' | 'NOT_READY'
  artifact_status: string | null
  winning_model_id: string | null
  algorithm: string | null
  required_columns: string[]
  checks: { check: string; passed: boolean; detail: string | null }[]
  reason: { code?: string; message?: string } | null
}

export interface DeploymentPackageResponse {
  run_id: string
  status: string
  files: string[]
  docker_optional: boolean
}

// --- Phase 6: Student Mode & ML Education ------------------------------------

export type ExplanationLevel = 'beginner' | 'intermediate' | 'advanced'

export interface FormulaEntry {
  name: string
  formula: string
  description: string
  when_used: string
}

export interface ComprehensionCheck {
  question: string
  answer_explanation: string
  related_concept: string
}

export interface ConceptDefinition {
  key: string
  title: string
  category: string
  summary: string
  detail: string
  related_formula?: string | null
}

export interface WhyExplanation {
  action: string
  what_happened: string
  why: string
  concept: string
  alternative_consideration?: string | null
  level: ExplanationLevel
  evidence: Record<string, unknown>
}

export interface OperationExplanation {
  operation_id: string
  tool_name: string
  what_happened: string
  why: string
  level?: ExplanationLevel
  concept?: string | null
  alternative_consideration?: string | null
}

export interface ModelSelectionExplanation {
  recommended_model_id: string
  recommended_algorithm: string
  justification: string
  candidates: ModelComparisonEntry[]
  concept?: string
}

export interface MetricExplanation {
  metric: string
  value: number
  meaning: string
  formula?: string | null
  guidance?: string | null
}

export interface ModelConceptExplanation {
  algorithm: string
  name: string
  concept: string
  strengths: string[]
  tradeoffs: string[]
  how_piper_used_it: string
  is_winner: boolean
}

export interface EvaluationExplanation {
  model_id: string
  algorithm?: string | null
  metrics: MetricExplanation[]
  confusion_matrix_meaning: string
  baseline_comparison?: string | null
  model_concept?: ModelConceptExplanation | null
}

export interface GuardrailCheckExplanation {
  check: string
  passed: boolean
  severity: string
  meaning: string
  message: string
  educational_action?: string | null
}

export interface FailureExplanation {
  category: string
  message: string
  retryable: boolean
  human_intervention_required: boolean
  meaning: string
  educational_takeaway?: string | null
}

export interface ReplanExplanation {
  replan_occurred: boolean
  total_attempts: number
  attempts_summary: Record<string, unknown>[]
  plan_differences: Record<string, unknown>[]
  educational_takeaway: string
}

export interface FeatureImportanceEducation {
  available: boolean
  method: string
  algorithm?: string | null
  disclaimer: string
  features: Record<string, unknown>[]
  educational_summary: string
}

export interface LearningJourneyStage {
  stage_id: number
  title: string
  description: string
  status: 'completed' | 'in_progress' | 'failed' | 'not_reached' | 'skipped'
  summary: string
  details: Record<string, unknown>
  concept: string
}

export interface LearningJourney {
  run_id: string
  status: string
  current_stage_id?: number | null
  stages: LearningJourneyStage[]
}

export interface PipelineNode {
  id: string
  name: string
  stage: string
  status: 'passed' | 'failed' | 'pending' | 'not_reached' | 'skipped'
  summary: string
  details: Record<string, unknown>
}

export interface PipelineEdge {
  from_node: string
  to_node: string
}

export interface PipelineVisualization {
  run_id: string
  nodes: PipelineNode[]
  edges: PipelineEdge[]
}

export interface RunExplanation {
  run_id: string
  status: string
  level: ExplanationLevel
  preprocessing: OperationExplanation[]
  feature_engineering: OperationExplanation[]
  model_selection?: ModelSelectionExplanation | null
  evaluation: EvaluationExplanation[]
  guardrail_checks: GuardrailCheckExplanation[]
  failure?: FailureExplanation | null
  replan?: ReplanExplanation | null
  feature_importance?: FeatureImportanceEducation | null
  model_concepts: ModelConceptExplanation[]
}

export interface ExplorationComparison {
  base_model_id: string
  new_model_id: string
  primary_metric: string
  base_metric_value: number
  new_metric_value: number
  delta: number
  winner_id: string
  justification: string
}

export interface ExplorationResult {
  experiment_id: string
  run_id: string
  base_model_id: string
  variable: {
    kind: 'algorithm' | 'hyperparameter'
    name: string
    base_value: unknown
    new_value: unknown
  }
  new_model: TrainingResult
  evaluation: EvaluationResult
  comparison: ExplorationComparison
  evaluation_explanation?: EvaluationExplanation | null
}

