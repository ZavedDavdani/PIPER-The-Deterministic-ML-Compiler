/**
 * Realistic fixture payloads matching the REAL backend's response
 * shapes exactly (app/api/schemas.py + app/schemas/*.py on the Python
 * side) — used by MSW handlers in tests. Not shown anywhere in the
 * running application itself.
 */
import type {
  BaselineComparisonResult,
  DatasetProfile,
  ModelComparison,
  PipelineValidationResult,
  ReproducibilityMetadata,
  RunResultResponse,
} from '@/lib/types'

export const fixtureDatasetProfile: DatasetProfile = {
  dataset_id: 'dataset_abc12345',
  rows: 7043,
  columns: 4,
  duplicate_rows: 0,
  memory_usage_bytes: 225376,
  column_profiles: [
    {
      name: 'customerID', dtype: 'object', missing_count: 0, missing_percentage: 0,
      unique_count: 7043, unique_percentage: 100, sample_values: ['7590-VHVEG'],
      min: null, max: null, mean: null, median: null, std: null,
    },
    {
      name: 'Contract', dtype: 'object', missing_count: 0, missing_percentage: 0,
      unique_count: 3, unique_percentage: 0.04, sample_values: ['Month-to-month'],
      min: null, max: null, mean: null, median: null, std: null,
    },
    {
      name: 'MonthlyCharges', dtype: 'float64', missing_count: 0, missing_percentage: 0,
      unique_count: 1585, unique_percentage: 22.5, sample_values: [29.85],
      min: 18.25, max: 118.75, mean: 64.76, median: 70.35, std: 30.09,
    },
    {
      name: 'Churn', dtype: 'object', missing_count: 0, missing_percentage: 0,
      unique_count: 2, unique_percentage: 0.03, sample_values: ['No'],
      min: null, max: null, mean: null, median: null, std: null,
    },
  ],
}

export const fixtureValidation: PipelineValidationResult = {
  dataset_id: 'dataset_abc12345',
  target_column: 'Churn',
  valid: true,
  checks: [
    { check: 'data_leakage', passed: true, severity: 'info', message: 'No leakage indicators detected among 3 feature(s) checked.' },
    { check: 'target_imbalance', passed: true, severity: 'info', message: 'Minority class is 26.5% — above the 15% warning threshold.' },
    { check: 'constant_features', passed: true, severity: 'info', message: 'No constant-valued columns detected.' },
    { check: 'high_cardinality', passed: true, severity: 'info', message: 'No suspiciously high-cardinality categorical columns detected.' },
    { check: 'baseline_gate', passed: true, severity: 'info', message: 'Model F1 exceeds the majority-class baseline by >= 0.05.' },
  ],
  violations: [],
  warnings: [],
  scope_note: 'valid=True means no implemented guardrail violation was detected by the checks run here.',
}

export const fixtureComparison: ModelComparison = {
  models: [
    { model_id: 'model_rf001', algorithm: 'random_forest', accuracy: 0.79, precision: 0.63, recall: 0.51, f1: 0.565, roc_auc: 0.83 },
    { model_id: 'model_lr001', algorithm: 'logistic_regression', accuracy: 0.8, precision: 0.65, recall: 0.55, f1: 0.596, roc_auc: 0.84 },
  ],
  recommended_model_id: 'model_lr001',
  selection_metric: 'f1',
}

export const fixtureBaseline: BaselineComparisonResult = {
  model_id: 'model_lr001',
  split_id: 'split_001',
  baseline: { accuracy: 0.735, precision: null, recall: 0, f1: 0, majority_class: 'No' },
  model_primary_metric_value: 0.596,
  baseline_primary_metric_value: 0,
  primary_metric: 'f1',
  delta: 0.596,
  minimum_required_delta: 0.05,
  gate_passed: true,
  reason: 'Model F1 (0.596) exceeds baseline F1 (0.0) by >= 0.05.',
}

export const fixtureReproducibility: ReproducibilityMetadata = {
  environment: { python_version: '3.11.9 (main)', pandas_version: '3.0.2', numpy_version: '2.2.0', sklearn_version: '1.8.0' },
  dataset_fingerprint: 'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678901234567890abcdef1234',
  split_random_state: 42,
  model_random_state: 42,
  pipeline_fingerprint: 'f1e2d3c4b5a6978869504132241536475869708192a3b4c5d6e7f8091a2b3c',
}

export const fixtureCompletedResult: RunResultResponse = {
  run_id: 'run_test0001',
  status: 'completed',
  validation: fixtureValidation,
  comparison: fixtureComparison,
  baseline: fixtureBaseline,
  failure: null,
  reproducibility: fixtureReproducibility,
  model_results: [],
  evaluation_results: [],
  error: null,
}

export const fixtureFailedResult: RunResultResponse = {
  run_id: 'run_test0002',
  status: 'failed',
  validation: { ...fixtureValidation, valid: false, violations: [{ check: 'data_leakage', passed: false, severity: 'error', message: "Feature 'leaky_dup' is a duplicate of the target." }] },
  comparison: fixtureComparison,
  baseline: null,
  failure: {
    category: 'DUPLICATE_PLAN',
    message: 'The newly LLM-proposed plan is executably identical to a previously attempted plan.',
    evidence: { duplicate_plan_hash: 'abc123', plan_history: ['abc123'] },
    node: 'plan',
    attempt: 1,
    retryable: false,
    human_intervention_required: true,
  },
  reproducibility: fixtureReproducibility,
  model_results: [],
  evaluation_results: [],
  error: 'Pipeline did not pass validation after 1 retry; reporting best-available result with this caveat.',
}
